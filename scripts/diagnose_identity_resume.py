"""Separate serialized-state errors from repeat-execution CUDA variability.

Uses the existing real identity pilot checkpoint. No training thresholds are
relaxed and no benchmark result is produced by this diagnostic.
"""

import argparse
import copy
import json
import os
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from luna_open.avatar import Gaussians
from luna_open.data.neuman import NeuManDataset
from luna_open.losses import geometry_priors, rendering_losses
from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256, validate_body_asset, verify_sources
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import FeatureStore, prepare_item, restore_rng, rng_state


def assert_equal(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.cpu(), b.cpu()):
            raise ValueError(f"Tensor restore mismatch: {path}")
    elif isinstance(a, np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or not np.array_equal(a, b):
            raise ValueError(f"Array restore mismatch: {path}")
    elif isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError(f"Dictionary restore mismatch: {path}")
        for key in a:
            assert_equal(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError(f"Sequence restore mismatch: {path}")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_equal(x, y, f"{path}.{i}")
    elif a != b:
        raise ValueError(f"Scalar restore mismatch: {path}")


def errors(a, b):
    difference = (a.detach().float() - b.detach().float()).abs()
    return dict(
        max=float(difference.max()),
        mean=float(difference.mean()),
        rms=float(difference.square().mean().sqrt()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    state = torch.load(args.checkpoint, weights_only=False, map_location="cpu")
    cfg = state["config"]
    validate_body_asset(state, file_sha256(cfg["smpl_model"]))
    if file_sha256(cfg["manifest"]) != state["manifest_sha256"]:
        raise ValueError("Manifest differs from checkpoint")
    verify_sources(Path(cfg["data_root"]), json.loads(Path(cfg["manifest"]).read_text()))
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], size=cfg["image_size"])
    item = data.load_frame(state["scene"], state["frame"])
    item.update(scene=state["scene"], reference_names=state["references"])
    target = prepare_item(item)
    features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
    assert_equal(features.metadata, state["feature_metadata"])
    body, face = features.references(item)
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).cuda()
    anchors = teacher.shaped_anchors(target["betas"])
    perceptual = build_lpips(cfg["training"]["lpips_backbone"], "cuda")

    def make_model():
        return IdentityEncoder(
            teacher.anchors, teacher.semantic_labels, ModelConfig(**cfg["model"])
        ).cuda()

    def predict(model):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = model(body, face)
            posed = teacher(
                canonical.gaussians,
                target["pose"],
                target["betas"],
                target["body_to_camera"],
                detach=False,
            )
        prediction = render(posed, target["K"], (cfg["image_size"], cfg["image_size"]))
        loss = sum(rendering_losses(prediction, target["rgb"], target["mask"], perceptual).values())
        loss = loss + sum(geometry_priors(canonical.gaussians, anchors).values())
        return posed, prediction, loss

    identity = make_model()
    optimizer = None
    reports = {}
    reference = None
    for branch in ("first", "repeat_same_instance", "fresh_instance"):
        if branch == "fresh_instance":
            del identity, optimizer
            identity = make_model()
        identity.load_state_dict(state["identity"])
        optimizer = torch.optim.AdamW(identity.parameters(), lr=cfg["training"]["learning_rate"])
        # Non-capturable AdamW keeps CPU step tensors; clone the snapshot so
        # a branch cannot increment the next branch's saved step counters.
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        assert_equal(identity.state_dict(), state["identity"])
        assert_equal(optimizer.state_dict(), state["optimizer"])
        restore_rng(state["rng"])
        assert_equal(rng_state(), state["rng"])
        identity.train()
        optimizer.zero_grad(set_to_none=True)
        posed, prediction, loss = predict(identity)
        before = dict(
            rgb=prediction["rgb"].detach().clone(),
            means=posed.means.detach().clone(),
            loss=float(loss.detach()),
        )
        fixed_gaussians = posed.detach()
        loss.backward()
        gradients = {
            name: p.grad.detach().clone()
            for name, p in identity.named_parameters()
            if p.grad is not None
        }
        norm = torch.nn.utils.clip_grad_norm_(
            identity.parameters(), cfg["training"]["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            _, after, _ = predict(identity)
        parameters = {name: p.detach().clone() for name, p in identity.named_parameters()}
        report = dict(
            restored_model_optimizer_rng_exact=True,
            loss_before=before["loss"],
            gradient_norm=float(norm),
        )
        if reference is None:
            reference = dict(
                before=before,
                gradients=gradients,
                parameters=parameters,
                after=after["rgb"].detach().clone(),
            )
        else:
            grad_errors = {
                k: errors(v, reference["gradients"][k])["max"] for k, v in gradients.items()
            }
            parameter_errors = {
                k: errors(v, reference["parameters"][k])["max"] for k, v in parameters.items()
            }
            report.update(
                before_rgb=errors(before["rgb"], reference["before"]["rgb"]),
                before_means=errors(before["means"], reference["before"]["means"]),
                gradient_max_abs_difference=max(grad_errors.values()),
                most_changed_gradient=max(grad_errors, key=grad_errors.get),
                parameter_max_abs_difference=max(parameter_errors.values()),
                most_changed_parameter=max(parameter_errors, key=parameter_errors.get),
                after_rgb=errors(after["rgb"], reference["after"]),
            )
        reports[branch] = report
        print(json.dumps({branch: report}), flush=True)
        del gradients, parameters, posed, prediction, loss, after

    # Repeated renderer backward with identical fixed Gaussian leaves and
    # constant upstream pixel gradients isolates this custom CUDA operation.
    renderer_reports = []
    renderer_reference = None
    for repeat in range(3):
        leaves = {
            f.name: getattr(fixed_gaussians, f.name).detach().clone().requires_grad_()
            for f in fields(Gaussians)
        }
        result = render(Gaussians(**leaves), target["K"], (cfg["image_size"], cfg["image_size"]))
        score = (result["rgb"] * target["rgb"]).mean() + result["alpha"].mean()
        score.backward()
        gradients = {k: v.grad.detach().clone() for k, v in leaves.items()}
        if renderer_reference is None:
            renderer_reference = dict(rgb=result["rgb"].detach().clone(), gradients=gradients)
        else:
            renderer_reports.append(
                dict(
                    repeat=repeat,
                    forward_rgb=errors(result["rgb"], renderer_reference["rgb"]),
                    gradients={
                        k: errors(v, renderer_reference["gradients"][k])
                        for k, v in gradients.items()
                    },
                )
            )
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha256(args.checkpoint),
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        seconds=time.perf_counter() - start,
        branches=reports,
        isolated_renderer=renderer_reports,
        limitation="One checkpoint and one input; repeated steps assess local variability, not long-run reproducibility",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
