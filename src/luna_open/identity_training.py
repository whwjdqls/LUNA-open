"""Identity retraining with shared reconstruction across several target frames.

Development choices and their evidence are recorded in docs/identity-retraining.md.
The original trainer remains available for reproducing the first NeuMan run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import time
from functools import lru_cache
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw

from .data.neuman import NeuManDataset
from .losses import geometry_priors, rendering_losses
from .metrics import aggregate_records, image_metrics
from .model import IdentityEncoder, ModelConfig
from .perceptual import build_lpips
from .provenance import file_sha256, validate_body_asset, verify_sources
from .rendering import render
from .smpl import SMPLTeacher
from .training import (
    FeatureStore,
    atomic_checkpoint,
    prepare_item,
    restore_rng,
    rng_state,
    seed_all,
)


class CachedIdentityFeatures(FeatureStore):
    """CPU FP16 cache; only selected reference tensors are copied to the GPU."""

    @lru_cache(maxsize=1024)
    def read(self, kind, scene, name):
        path = self.root / kind / scene / (Path(name).stem + ".pt")
        feature = torch.load(path, weights_only=True, map_location="cpu")["features"]
        shape = (4096, 1536) if kind == "body" else (1024, 1536)
        if feature.shape != shape or feature.dtype != torch.float16 or not feature.isfinite().all():
            raise ValueError(f"Invalid {kind} features: {path}, {feature.shape}, {feature.dtype}")
        return feature.pin_memory()

    def get(self, kind, scene, name):
        return self.read(kind, scene, name).to(
            device="cuda", dtype=torch.float32, non_blocking=True
        )


def image(value):
    array = (value.detach().float().cpu().clamp(0, 1).permute(1, 2, 0) * 255).byte().numpy()
    return Image.fromarray(array)


def identity_losses(prediction, target, canonical, anchors, perceptual, options):
    losses = rendering_losses(prediction, target["rgb"], target["mask"], perceptual)
    if options["rgb_reduction"] == "balanced_foreground_background":
        error = (prediction["rgb"] - target["rgb"]).abs()
        foreground = target["mask"].expand_as(error)
        background = 1 - foreground
        fg = (error * foreground).sum() / foreground.sum().clamp_min(1)
        bg = (error * background).sum() / background.sum().clamp_min(1)
        losses["rgb"] = 0.5 * (fg + bg)
    elif options["rgb_reduction"] != "full_image":
        raise ValueError(f"Unknown RGB reduction: {options['rgb_reduction']}")
    # Keep the original priors and their units explicit. Configured weights
    # replace the original constants rather than multiplying them twice.
    priors = geometry_priors(canonical.gaussians, anchors)
    losses["anisotropy"] = priors["anisotropy"] / 0.01 * options["anisotropy_weight"]
    losses["offset"] = priors["offset"] / 10 * options["anchor_weight"]
    return losses


@torch.no_grad()
def evaluate(identity, teacher, features, data, load_frame, perceptual, size, output, update):
    identity.eval()
    records, panels = [], []
    for scene, info in data.metadata.items():
        body, face = features.references(dict(scene=scene, reference_names=info["references"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = identity(body, face)
        for index, name in enumerate(info["splits"]["val"]):
            target = prepare_item(load_frame(scene, name))
            posed = teacher(
                canonical.gaussians, target["pose"], target["betas"], target["body_to_camera"]
            )
            prediction = render(posed, target["K"], (size, size))
            scores = {
                k: float(v[0])
                for k, v in image_metrics(
                    prediction, target["rgb"], target["mask"], perceptual
                ).items()
            }
            error = (prediction["rgb"] - target["rgb"]).abs()
            mask = target["mask"].expand_as(error)
            scores["foreground_l1"] = float((error * mask).sum() / mask.sum().clamp_min(1))
            records.append(dict(scene=scene, frame=name, metrics=scores))
            if index == len(info["splits"]["val"]) // 2:
                panel = Image.new("RGB", (2 * size, size + 35), "white")
                panel.paste(image(target["rgb"][0]), (0, 35))
                panel.paste(image(prediction["rgb"][0]), (size, 35))
                ImageDraw.Draw(panel).text(
                    (10, 10), f"{scene} {name} | GT | identity {update}", fill="black"
                )
                panels.append(panel)
    report = aggregate_records(records)
    report.update(
        update=update,
        stage="identity",
        split="val",
        reference_protocol="four fixed training references",
    )
    (output / f"val-{update:06d}.json").write_text(json.dumps(report, indent=2) + "\n")
    overview = Image.new("RGB", (2 * size, len(panels) * (size + 35)), "white")
    for i, panel in enumerate(panels):
        overview.paste(panel, (0, i * (size + 35)))
    overview.save(output / f"val-{update:06d}.jpg", quality=95)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-update", type=int)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise RuntimeError("Run on a Slurm GPU compute node")
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This Yonsei development run requires exactly one RTX 4090")
    cfg = yaml.safe_load(args.config.read_text())
    options, losses_options = cfg["training"], cfg["identity_loss"]
    output = Path(cfg["output"]) / "identity"
    if not args.resume and (output / "train.jsonl").exists():
        raise FileExistsError("Fresh identity training requires an unused output directory")
    output.mkdir(parents=True, exist_ok=True)
    targets_per_identity = options["targets_per_identity"]
    batch = options["effective_batch"]
    if targets_per_identity < 2 or batch % targets_per_identity:
        raise ValueError("Effective batch must be divisible by at least two targets per identity")
    if cfg["model"]["identity_face_encoder"] != "sapiens":
        raise ValueError("This trainer expects cached Sapiens body and face features")
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    manifest_hash, body_hash = file_sha256(cfg["manifest"]), file_sha256(cfg["smpl_model"])
    package = Path(__file__).parent
    code_hashes = {path.name: file_sha256(path) for path in sorted(package.glob("*.py"))}
    code_hashes["data/neuman.py"] = file_sha256(package / "data/neuman.py")
    seed_all(cfg["seed"])
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).cuda()
    identity = IdentityEncoder(
        teacher.anchors, teacher.semantic_labels, ModelConfig(**cfg["model"])
    ).cuda()
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], size=cfg["image_size"])
    features = CachedIdentityFeatures(cfg["features"], cfg["manifest"], ("body", "face"))
    if features.metadata["face"].get("face_backbone") != "sapiens":
        raise ValueError("Face metadata does not identify the Sapiens cache")
    for scene, info in data.metadata.items():
        if len(info["splits"]["train"]) < targets_per_identity + 4:
            raise ValueError(f"Not enough disjoint references and targets in {scene}")
        for kind in ("body", "face"):
            for name in info["splits"]["train"]:
                if not (features.root / kind / scene / (Path(name).stem + ".pt")).is_file():
                    raise FileNotFoundError(f"Missing {kind} cache for {scene}/{name}")

    @lru_cache(maxsize=512)
    def load_frame(scene, name):
        return data.load_frame(scene, name)

    perceptual = build_lpips(options["lpips_backbone"], device="cuda")
    optimizer = torch.optim.AdamW(
        identity.parameters(),
        lr=options["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=options["weight_decay"],
    )

    def schedule(update):
        warmup = options["lr_warmup"]
        if update < warmup:
            return (update + 1) / warmup
        progress = min((update - warmup) / (options["updates"] - warmup), 1)
        floor = options["final_lr_fraction"]
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, best = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_body_asset(state, body_hash)
        if (
            state["config"] != cfg
            or state["manifest_sha256"] != manifest_hash
            or state["feature_metadata"] != features.metadata
            or state["source_sha256"] != code_hashes
        ):
            raise ValueError("Resume configuration, data, features or source mismatch")
        if state["stage"] != "identity" or state.get("trainer") != "identity_multitarget_v2":
            raise ValueError("This is not an identity v2 training checkpoint")
        identity.load_state_dict(state["identity"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start, best = state["update"], state["best_lpips"]
        # Restore only after all model/metric/optimizer construction.
        restore_rng(state["rng"])
        del state
        last = json.loads((output / "train.jsonl").read_text().splitlines()[-1])
        if last["update"] != start:
            raise ValueError("Resume checkpoint must match the last recorded update")
    stop = args.stop_after_update or options["updates"]
    if not start < stop <= options["updates"]:
        raise ValueError("Invalid requested stopping update")
    receipt = dict(
        job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(),
        device=torch.cuda.get_device_name(),
        capability=list(torch.cuda.get_device_capability()),
        config=cfg,
        config_sha256=file_sha256(args.config),
        manifest_sha256=manifest_hash,
        smpl_asset_sha256=body_hash,
        source_sha256=code_hashes,
        feature_metadata=features.metadata,
        initialization="fresh reconstruction network" if not args.resume else str(args.resume),
        start_update=start,
        requested_stop=stop,
        train_frames=sum(len(s["splits"]["train"]) for s in data.metadata.values()),
        val_frames=sum(len(s["splits"]["val"]) for s in data.metadata.values()),
    )
    (output / f"startup-{start:06d}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                event="startup",
                job=receipt["job_id"],
                host=receipt["host"],
                device=receipt["device"],
                start=start,
                stop=stop,
                train_frames=receipt["train_frames"],
            )
        ),
        flush=True,
    )
    scenes = list(data.metadata)
    for update in range(start, stop):
        began = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        identity.train()
        optimizer.zero_grad(set_to_none=True)
        metrics, sampled = {}, []
        for _ in range(batch // targets_per_identity):
            scene = random.choice(scenes)
            choices = random.sample(
                data.metadata[scene]["splits"]["train"], targets_per_identity + 4
            )
            references, target_names = choices[:4], choices[4:]
            assert set(references).isdisjoint(target_names)
            sampled.append(dict(scene=scene, references=references, targets=target_names))
            body, face = features.references(dict(scene=scene, reference_names=references))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                canonical = identity(body, face)
            group_total = canonical.gaussians.means.new_zeros(())
            for name in target_names:
                target = prepare_item(load_frame(scene, name))
                posed = teacher(
                    canonical.gaussians,
                    target["pose"],
                    target["betas"],
                    target["body_to_camera"],
                    detach=False,
                )
                prediction = render(posed, target["K"], (cfg["image_size"], cfg["image_size"]))
                terms = identity_losses(
                    prediction,
                    target,
                    canonical,
                    teacher.shaped_anchors(target["betas"]),
                    perceptual,
                    losses_options,
                )
                group_total = group_total + sum(terms.values()) / batch
                for key, value in terms.items():
                    metrics[key] = metrics.get(key, 0) + float(value.detach()) / batch
            if not torch.isfinite(group_total):
                raise FloatingPointError(f"Nonfinite identity loss at update {update + 1}")
            group_total.backward()
            del group_total, canonical, prediction, posed, terms, body, face
        gradient = torch.nn.utils.clip_grad_norm_(
            identity.parameters(), options["gradient_clip"], error_if_nonfinite=True
        )
        if gradient == 0:
            raise RuntimeError("Identity reconstruction network has zero gradient")
        if update < 3 or (update + 1) % 100 == 0:
            metrics["gradient_probe_after_clip"] = {
                name: None if parameter.grad is None else float(parameter.grad.norm())
                for name, parameter in identity.named_parameters()
                if name
                in {
                    "queries",
                    "body_projection.weight",
                    "face_projection.weight",
                    "blocks.0.qkv.0.weight",
                    "decoder.4.weight",
                }
            }
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        completed = update + 1
        duration = time.perf_counter() - began
        improved = False
        if completed % options["validate_every"] == 0 or completed == options["updates"]:
            result = evaluate(
                identity,
                teacher,
                features,
                data,
                load_frame,
                perceptual,
                cfg["image_size"],
                output,
                completed,
            )
            score = result["mean_over_scenes"]["lpips"]
            improved = score < best
            best = min(best, score)
            metrics["validation_lpips"] = score
            metrics["validation_foreground_l1"] = result["mean_over_scenes"]["foreground_l1"]
        record = dict(
            update=completed,
            lr=scheduler.get_last_lr()[0],
            training_seconds=duration,
            gradient_norm_before_clip=float(gradient),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            total=sum(metrics[k] for k in ("rgb", "mask", "lpips", "anisotropy", "offset")),
            **metrics,
        )
        with (output / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if completed <= 2 or completed % 10 == 0:
            print(json.dumps(record), flush=True)
        if (
            completed == 1
            or completed % options["checkpoint_every"] == 0
            or completed == stop
            or improved
        ):
            state = dict(
                config=cfg,
                trainer="identity_multitarget_v2",
                stage="identity",
                update=completed,
                best_lpips=best,
                manifest_sha256=manifest_hash,
                smpl_asset_sha256=body_hash,
                source_sha256=code_hashes,
                feature_metadata=features.metadata,
                identity=identity.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                rng=rng_state(),
                last_sampled=sampled,
            )
            atomic_checkpoint(output / "latest.pt", state)
            if improved:
                atomic_checkpoint(output / "best.pt", state)
            del state


if __name__ == "__main__":
    main()
