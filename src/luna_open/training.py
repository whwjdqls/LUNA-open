"""Single-GPU development training with resumable optimizer/RNG state.

Large-data DDP and synchronized multiview refinement are subsequent milestones.
No fallback to random features is allowed in this training entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from .data.neuman import NeuManDataset, frame_annotation
from .geometry import matrix_to_sincos
from .losses import geometry_priors, projection_loss, rendering_losses, structural_loss
from .metrics import aggregate_records, image_metrics
from .model import IdentityEncoder, ModelConfig, NeuralAnimator
from .perceptual import build_lpips
from .provenance import (
    file_sha256,
    validate_body_asset,
    validate_identity_transfer,
    verify_sources,
)
from .rendering import render
from .smpl import SMPLTeacher


class FeatureStore:
    def __init__(self, root: str, manifest: str, kinds: tuple[str, ...]):
        self.root = Path(root)
        self.metadata = {}
        manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
        for kind in kinds:
            meta = json.loads((self.root / kind / "metadata.json").read_text())
            if meta["manifest_sha256"] != manifest_hash:
                raise ValueError(f"Feature manifest mismatch: {kind}")
            self.metadata[kind] = meta

    def get(self, kind: str, scene: str, name: str) -> torch.Tensor:
        path = self.root / kind / scene / (Path(name).stem + ".pt")
        return torch.load(path, weights_only=True, map_location="cpu")["features"].cuda().float()

    def references(self, item: dict):
        values = []
        for kind in ("body", "face"):
            values.append(
                torch.stack([self.get(kind, item["scene"], n) for n in item["reference_names"]])[
                    None
                ]
            )
        return values


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all(),
    )


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".part")
    torch.save(state, temporary)
    temporary.replace(path)


def prepare_item(item):
    return {k: v[None].cuda() if isinstance(v, torch.Tensor) else v for k, v in item.items()}


@torch.no_grad()
def translation_statistics(teacher, data):
    values = []
    for scene, name in data.items:
        annotation = frame_annotation(data.root / scene, data.lookup[scene][name])
        pose, betas, transform = [
            torch.from_numpy(annotation[k])[None].cuda()
            for k in ("pose", "betas", "body_to_camera")
        ]
        _, translation = teacher.global_motion(pose, betas, transform)
        values.append(translation[0])
    stacked = torch.stack(values)
    return stacked.mean(0), stacked.std(0, unbiased=False).clamp_min(1e-4)


def forward_item(identity, animator, teacher, feature_store, item, stage, global_only=False):
    body, face = feature_store.references(item)
    tensor_item = prepare_item(item)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.set_grad_enabled(torch.is_grad_enabled() and stage == "identity"):
            canonical = identity(body, face)
        if stage == "identity":
            gaussians = teacher(
                canonical.gaussians,
                tensor_item["pose"],
                tensor_item["betas"],
                tensor_item["body_to_camera"],
                detach=False,
            )
            posed = None
        else:
            motion = feature_store.get("motion", item["scene"], item["frame"])[None]
            posed = animator(canonical, motion, global_only=global_only)
            gaussians = posed.gaussians
    return canonical, posed, gaussians, tensor_item


@torch.no_grad()
def evaluate(identity, animator, teacher, features, data, stage, lpips, size):
    identity.eval()
    animator.eval()
    records = []
    for index in range(len(data)):
        item = data[index]
        _, _, gaussians, target = forward_item(identity, animator, teacher, features, item, stage)
        prediction = render(gaussians, target["K"], (size, size))
        scores = image_metrics(prediction, target["rgb"], target["mask"], lpips)
        records.append(
            dict(
                scene=item["scene"],
                frame=item["frame"],
                metrics={k: float(v[0]) for k, v in scores.items()},
            )
        )
    return aggregate_records(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["identity", "animator"], required=True)
    parser.add_argument("--identity-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", choices=["val", "test"])
    parser.add_argument(
        "--stop-after-update",
        type=int,
        help="Save and exit at this absolute update without changing the configured schedule",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Training/evaluation requires a Slurm GPU allocation")
    if args.stage == "animator" and not (args.identity_checkpoint or args.resume):
        raise ValueError("Animator needs a trained identity checkpoint")
    if args.evaluate and not args.resume:
        raise ValueError("Evaluation requires --resume checkpoint")
    verify_sources(Path(cfg["data_root"]), json.loads(Path(cfg["manifest"]).read_text()))
    body_asset_hash = file_sha256(cfg["smpl_model"])
    seed_all(cfg["seed"])
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).cuda()
    model_config = ModelConfig(**cfg["model"])
    identity = IdentityEncoder(teacher.anchors, teacher.semantic_labels, model_config).cuda()
    data = NeuManDataset(
        cfg["data_root"], cfg["manifest"], size=cfg["image_size"], random_references=True
    )
    validation = NeuManDataset(
        cfg["data_root"], cfg["manifest"], split=args.evaluate or "val", size=cfg["image_size"]
    )
    mean, std = translation_statistics(teacher, data)
    animator = NeuralAnimator(cfg["num_queries"], model_config, mean, std).cuda()
    kinds = ("body", "face") if args.stage == "identity" else ("body", "face", "motion")
    features = FeatureStore(cfg["features"], cfg["manifest"], kinds)
    # Fail before training if any required feature is absent, including held-out
    # drivers for evaluation. Cached held-out inputs never update network weights.
    for scene, info in data.metadata.items():
        for row in info["frames"]:
            for kind in kinds:
                path = features.root / kind / scene / (Path(row["name"]).stem + ".pt")
                if not path.is_file():
                    raise FileNotFoundError(path)
    manifest_hash = hashlib.sha256(Path(cfg["manifest"]).read_bytes()).hexdigest()
    if args.identity_checkpoint:
        initial = torch.load(args.identity_checkpoint, weights_only=False, map_location="cpu")
        validate_body_asset(initial, body_asset_hash)
        if initial["manifest_sha256"] != manifest_hash or initial["stage"] != "identity":
            raise ValueError("Incompatible identity checkpoint")
        validate_identity_transfer(initial, cfg, features.metadata)
        identity.load_state_dict(initial["identity"])
    identity.requires_grad_(args.stage == "identity")
    animator.requires_grad_(args.stage == "animator")
    model = identity if args.stage == "identity" else animator
    options = cfg["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=options["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=options["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda n: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(n / options["updates"], 1))),
    )
    start, best = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, weights_only=False, map_location="cpu")
        validate_body_asset(state, body_asset_hash)
        if (
            state["manifest_sha256"] != manifest_hash
            or state["stage"] != args.stage
            or state["config"] != cfg
            or state["feature_metadata"] != features.metadata
        ):
            raise ValueError("Resume provenance/configuration mismatch")
        identity.load_state_dict(state["identity"])
        animator.load_state_dict(state["animator"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best = state["update"], state["best_lpips"]
    stop = options["updates"] if args.stop_after_update is None else args.stop_after_update
    if not args.evaluate and not start < stop <= options["updates"]:
        raise ValueError(
            "Stopping update must follow the checkpoint and fit the configured schedule"
        )
    perceptual = build_lpips(options["lpips_backbone"], device="cuda")
    if args.resume:
        # LPIPS construction also consumes RNG; restore after ALL initialization.
        restore_rng(state["rng"])
    output = Path(cfg["output"]) / args.stage
    output.mkdir(parents=True, exist_ok=True)
    if args.evaluate:
        result = evaluate(
            identity,
            animator,
            teacher,
            features,
            validation,
            args.stage,
            perceptual,
            cfg["image_size"],
        )
        result.update(
            manifest_sha256=manifest_hash,
            checkpoint=str(args.resume),
            protocol="seen-sequence, frame-held-out, annotated crops",
        )
        (output / f"{args.evaluate}-metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["mean_over_scenes"], indent=2))
        return
    by_scene = {
        scene: [i for i, pair in enumerate(data.items) if pair[0] == scene]
        for scene in data.metadata
    }
    for update in range(start, stop):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = {}
        warmup = args.stage == "animator" and update < options["global_warmup"]
        for _ in range(options["effective_batch"]):
            scene = random.choice(list(by_scene))
            item = data[random.choice(by_scene[scene])]
            canonical, posed, gaussians, target = forward_item(
                identity, animator, teacher, features, item, args.stage, warmup
            )
            losses = {}
            if not warmup:
                rendered = render(gaussians, target["K"], (cfg["image_size"], cfg["image_size"]))
                losses.update(rendering_losses(rendered, target["rgb"], target["mask"], perceptual))
            if args.stage == "identity":
                losses.update(
                    geometry_priors(canonical.gaussians, teacher.shaped_anchors(target["betas"]))
                )
            else:
                labeled = torch.ones(1, device="cuda", dtype=torch.bool)
                expected = teacher(
                    canonical.gaussians, target["pose"], target["betas"], target["body_to_camera"]
                )
                root_rotation, _ = teacher.global_motion(
                    target["pose"], target["betas"], target["body_to_camera"]
                )
                losses["rotation"] = F.l1_loss(
                    posed.rotation_sincos, matrix_to_sincos(root_rotation)
                )
                losses["projection"] = projection_loss(
                    gaussians.means,
                    expected.means,
                    target["K"],
                    (cfg["image_size"], cfg["image_size"]),
                    labeled,
                )
                if not warmup:
                    losses["structural"] = structural_loss(gaussians, expected, labeled)
            total = sum(losses.values()) / options["effective_batch"]
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at update {update}")
            total.backward()
            for key, value in losses.items():
                metrics[key] = (
                    metrics.get(key, 0.0) + float(value.detach()) / options["effective_batch"]
                )
        nn.utils.clip_grad_norm_(
            model.parameters(), options["gradient_clip"], error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        completed = update + 1
        improved = False
        if completed % options["validate_every"] == 0 or completed == options["updates"]:
            result = evaluate(
                identity,
                animator,
                teacher,
                features,
                validation,
                args.stage,
                perceptual,
                cfg["image_size"],
            )
            score = result["mean_over_scenes"]["lpips"]
            improved, best = score < best, min(best, score)
            metrics["validation_lpips"] = score
        record = dict(update=completed, lr=scheduler.get_last_lr()[0], **metrics)
        with (output / "train.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        if completed % 10 == 0 or completed == 1:
            print(json.dumps(record), flush=True)
        if (
            improved
            or completed % options["checkpoint_every"] == 0
            or completed == options["updates"]
            or completed == stop
        ):
            state = dict(
                config=cfg,
                stage=args.stage,
                update=completed,
                best_lpips=best,
                manifest_sha256=manifest_hash,
                smpl_asset_sha256=body_asset_hash,
                feature_metadata=features.metadata,
                identity=identity.state_dict(),
                animator=animator.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=rng_state(),
            )
            atomic_checkpoint(output / "latest.pt", state)
            if improved:
                atomic_checkpoint(output / "best.pt", state)


if __name__ == "__main__":
    main()
