"""One-frame real-data identity overfit and checkpoint continuation check.

Uses the configured model size, actual cached references, licensed SMPL teacher,
512px gsplat rendering, LPIPS and AdamW. This is a training integration diagnostic,
not a held-out benchmark or an identity model ready for animator training.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from luna_open.data.neuman import NeuManDataset
from luna_open.losses import geometry_priors, rendering_losses
from luna_open.metrics import image_metrics
from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256, verify_sources
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import (
    FeatureStore,
    atomic_checkpoint,
    prepare_item,
    restore_rng,
    rng_state,
    seed_all,
)


def save_image(tensor, path):
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--scene", default="bike")
    parser.add_argument("--frame", default="00001.png")
    args = parser.parse_args()
    if args.steps < 2 or not torch.cuda.is_available():
        raise ValueError("Use at least two steps inside a GPU allocation")
    if (args.output / "report.json").exists():
        raise FileExistsError("Use a new pilot output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())
    seed_all(cfg["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], size=cfg["image_size"])
    if args.frame not in data.metadata[args.scene]["splits"]["train"]:
        raise ValueError("The pilot target must be a training frame")
    refs = data.metadata[args.scene]["references"]
    if args.frame in refs:
        raise ValueError("Pilot target must be excluded from reference inputs")
    item = data.load_frame(args.scene, args.frame)
    item.update(scene=args.scene, frame=args.frame, reference_names=refs)
    target = prepare_item(item)
    features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
    body, face = features.references(item)
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).cuda()
    model_config = ModelConfig(**cfg["model"])
    identity = IdentityEncoder(teacher.anchors, teacher.semantic_labels, model_config).cuda()
    perceptual = build_lpips(cfg["training"]["lpips_backbone"], device="cuda")
    options = cfg["training"]

    def make_optimizer(model):
        return torch.optim.AdamW(
            model.parameters(),
            lr=options["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=options["weight_decay"],
        )

    optimizer = make_optimizer(identity)
    anchors = teacher.shaped_anchors(target["betas"])

    def forward(model):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = model(body, face)
            posed = teacher(
                canonical.gaussians,
                target["pose"],
                target["betas"],
                target["body_to_camera"],
                detach=False,
            )
        posed.validate()
        prediction = render(posed, target["K"], (cfg["image_size"], cfg["image_size"]))
        losses = rendering_losses(prediction, target["rgb"], target["mask"], perceptual)
        losses.update(geometry_priors(canonical.gaussians, anchors))
        return prediction, losses

    def step(model, opt):
        model.train()
        opt.zero_grad(set_to_none=True)
        _, losses = forward(model)
        total = sum(losses.values())
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite pilot loss")
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), options["gradient_clip"], error_if_nonfinite=True
        )
        if gradient <= 0:
            raise ValueError("Identity model has zero training gradient")
        opt.step()
        return dict(
            total=float(total.detach()),
            gradient_norm=float(gradient),
            **{k: float(v.detach()) for k, v in losses.items()},
        )

    @torch.no_grad()
    def inspect(model, label):
        model.eval()
        prediction, losses = forward(model)
        save_image(prediction["rgb"][0], args.output / f"{label}-rgb.png")
        save_image(prediction["alpha"][0].expand(3, -1, -1), args.output / f"{label}-alpha.png")
        scores = image_metrics(prediction, target["rgb"], target["mask"], perceptual)
        return dict(
            total=float(sum(losses.values())),
            losses={k: float(v) for k, v in losses.items()},
            metrics={k: float(v[0]) for k, v in scores.items()},
        )

    save_image(target["rgb"][0], args.output / "target.png")
    initial = inspect(identity, "initial")
    history = []
    for update in range(args.steps):
        record = dict(update=update + 1, **step(identity, optimizer))
        history.append(record)
        with (args.output / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if update == 0 or (update + 1) % 10 == 0:
            print(json.dumps(record), flush=True)
    final = inspect(identity, "final")
    training_peak = torch.cuda.max_memory_allocated()
    # Preserve completed training evidence even if a later continuation check fails.
    (args.output / "training-report.json").write_text(
        json.dumps(dict(initial=initial, final=final, updates=args.steps), indent=2) + "\n"
    )
    if final["total"] >= initial["total"]:
        raise ValueError("Pilot did not reduce the measured fixed-target objective")
    checkpoint = args.output / "pilot.pt"
    atomic_checkpoint(
        checkpoint,
        dict(
            kind="one-frame-identity-integration-pilot",
            config=cfg,
            steps=args.steps,
            scene=args.scene,
            frame=args.frame,
            references=refs,
            identity=identity.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(),
            manifest_sha256=file_sha256(cfg["manifest"]),
            smpl_asset_sha256=file_sha256(cfg["smpl_model"]),
            feature_metadata=features.metadata,
        ),
    )
    # Compare one uninterrupted AdamW update with a separately constructed model
    # and optimizer loaded from the actual disk checkpoint. CUDA reductions may
    # differ at roundoff level; report errors instead of claiming bitwise identity.
    uninterrupted = step(identity, optimizer)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    resumed = IdentityEncoder(teacher.anchors, teacher.semantic_labels, model_config).cuda()
    resumed.load_state_dict(state["identity"])
    resumed_optimizer = make_optimizer(resumed)
    resumed_optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    continued = step(resumed, resumed_optimizer)
    max_parameter_error = max(
        float((a.detach() - b.detach()).abs().max())
        for a, b in zip(identity.parameters(), resumed.parameters(), strict=True)
    )
    with torch.no_grad():
        a, _ = forward(identity)
        b, _ = forward(resumed)
        image_error = float((a["rgb"] - b["rgb"]).abs().max())
    resume_passed = not (
        abs(uninterrupted["total"] - continued["total"]) > 1e-5
        or max_parameter_error > 1e-4
        or image_error > 1e-3
    )
    torch.cuda.synchronize()
    result = dict(
        protocol="One training target, four distinct fixed training references; no held-out evaluation",
        job_id=os.environ.get("SLURM_JOB_ID"),
        device=torch.cuda.get_device_name(),
        scene=args.scene,
        frame=args.frame,
        references=refs,
        updates=args.steps,
        model_parameters=sum(p.numel() for p in identity.parameters()),
        image_size=cfg["image_size"],
        num_queries=cfg["num_queries"],
        config=cfg,
        manifest_sha256=state["manifest_sha256"],
        smpl_asset_sha256=state["smpl_asset_sha256"],
        initial=initial,
        final=final,
        seconds=time.perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_peak_allocated_bytes=training_peak,
        resume_check=dict(
            passed=resume_passed,
            loss_abs_difference=abs(uninterrupted["total"] - continued["total"]),
            max_parameter_abs_difference=max_parameter_error,
            max_rgb_abs_difference=image_error,
        ),
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        limitations=[
            "Effective batch 1, fixed references and target, constant learning rate",
            "Not the full training sampler/schedule or a reusable identity checkpoint",
            "Memory includes a second model/optimizer for the continuation check",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not resume_passed:
        raise ValueError(
            f"Checkpoint continuation differs: parameters {max_parameter_error}, image {image_error}"
        )


if __name__ == "__main__":
    main()
