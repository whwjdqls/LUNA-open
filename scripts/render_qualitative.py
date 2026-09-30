"""Render fixed held-out NeuMan comparisons on a Slurm GPU compute node.

This is a visualization of the existing evaluation path. The identity column
uses the fitted SMPL pose; the animator column uses cached driving-image
features. Neither branch is trained or updated here.
"""

import argparse
import gc
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from luna_open.data.neuman import NeuManDataset
from luna_open.metrics import aggregate_records, image_metrics
from luna_open.model import IdentityEncoder, ModelConfig, NeuralAnimator
from luna_open.provenance import file_sha256, validate_body_asset, verify_sources
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import FeatureStore, forward_item, seed_all


def load_snapshot(path):
    # Hash and load the same open file, even if training atomically replaces best.pt.
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=False)
    return state, digest


def rgb_image(tensor):
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite render")
    array = tensor.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype("uint8"))


def font(size):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)


def reference_grid(images, tile):
    canvas = Image.new("RGB", (tile, tile), "white")
    side = tile // 2
    for index, image in enumerate(images):
        canvas.paste(image.resize((side, side)), ((index % 2) * side, (index // 2) * side))
    return canvas


def comparison_panel(refs, target, identity, animator, title, tile=320):
    gap, margin, heading, footer = 10, 18, 72, 32
    canvas = Image.new("RGB", (4 * tile + 3 * gap + 2 * margin, heading + tile + footer), "#eef1f5")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 9), title, fill="#17212b", font=font(20))
    labels = (
        "Four reference images",
        "Ground truth / driver",
        "Identity + fitted SMPL pose",
        "Image-driven animator",
    )
    images = (reference_grid(refs, tile), target, identity, animator)
    for index, (label, image) in enumerate(zip(labels, images, strict=True)):
        left = margin + index * (tile + gap)
        draw.text((left, 43), label, fill="#17212b", font=font(16))
        canvas.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (left, heading))
    draw.text(
        (margin, heading + tile + 7),
        "512 px renders; fixed training references; annotated foreground crops on white",
        fill="#465465",
        font=font(14),
    )
    return canvas


def stack_panels(panels):
    canvas = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), "white")
    offset = 0
    for panel in panels:
        canvas.paste(panel, (0, offset))
        offset += panel.height
    return canvas


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--identity-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Use a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.output.exists():
        raise FileExistsError("Choose a fresh qualitative output directory")
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.25)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free; leave the running training job undisturbed")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    seed_all(cfg["seed"])
    state, checkpoint_hash = load_snapshot(args.checkpoint)
    manifest_hash = file_sha256(cfg["manifest"])
    if (
        state["stage"] != "animator"
        or state["config"] != cfg
        or state["manifest_sha256"] != manifest_hash
    ):
        raise ValueError("Animator checkpoint/configuration/manifest mismatch")
    validate_body_asset(state, file_sha256(cfg["smpl_model"]))
    identity_state, identity_hash = load_snapshot(args.identity_checkpoint)
    if identity_state["stage"] != "identity" or identity_state["config"] != cfg:
        raise ValueError("Identity checkpoint mismatch")
    if identity_state["identity"].keys() != state["identity"].keys() or not all(
        torch.equal(value, state["identity"][key])
        for key, value in identity_state["identity"].items()
    ):
        raise ValueError("Animator identity differs from selected identity checkpoint")
    identity_update = identity_state["update"]
    animator_update = state["update"]
    identity_tensors = len(identity_state["identity"])
    del identity_state
    # Preserve compact weights so this preview remains reproducible after best.pt changes.
    snapshot_keys = (
        "stage",
        "update",
        "best_lpips",
        "config",
        "identity",
        "animator",
        "manifest_sha256",
        "smpl_asset_sha256",
        "feature_metadata",
    )
    torch.save({key: state[key] for key in snapshot_keys}, args.output / "inference-snapshot.pt")
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    data = NeuManDataset(
        cfg["data_root"], cfg["manifest"], split=args.split, size=cfg["image_size"]
    )
    features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face", "motion"))
    if features.metadata != state["feature_metadata"]:
        raise ValueError("Feature provenance differs from checkpoint")
    model_config = ModelConfig(**cfg["model"])
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg.get("smpl_pose_blend_shapes", True),
    ).cuda()
    if not torch.equal(teacher.anchors.cpu(), state["identity"]["anchors"]) or not torch.equal(
        teacher.semantic_labels.cpu(), state["identity"]["semantic_labels"]
    ):
        raise ValueError("SMPL query correspondence differs")
    identity = (
        IdentityEncoder(teacher.anchors, teacher.semantic_labels, model_config)
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    identity.load_state_dict(state["identity"])
    animator = (
        NeuralAnimator(
            cfg["num_queries"],
            model_config,
            state["animator"]["translation_mean"],
            state["animator"]["translation_std"],
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    animator.load_state_dict(state["animator"])
    del state
    gc.collect()
    print(
        json.dumps(
            {
                "event": "loaded",
                "host": socket.gethostname(),
                "identity_update": identity_update,
                "animator_update": animator_update,
                "frames": len(data),
            }
        ),
        flush=True,
    )
    records = {"identity": [], "animator": []}
    by_scene = {}
    references = {}
    for index in range(len(data)):
        item = data[index]
        scene, frame = item["scene"], item["frame"]
        destination = args.output / scene
        destination.mkdir(exist_ok=True)
        if scene not in references:
            references[scene] = [rgb_image(value) for value in item["reference_images"]]
            for ref_name, image in zip(item["reference_names"], references[scene], strict=True):
                image.save(destination / f"reference-{ref_name}")
        target_image = rgb_image(item["rgb"])
        target_image.save(destination / f"target-{frame}")
        results = {}
        for stage in ("identity", "animator"):
            _, _, gaussians, target = forward_item(
                identity, animator, teacher, features, item, stage
            )
            prediction = render(gaussians, target["K"], (cfg["image_size"], cfg["image_size"]))
            results[stage] = rgb_image(prediction["rgb"][0])
            results[stage].save(destination / f"{stage}-{frame}")
            rgb_image(prediction["alpha"][0].expand(3, -1, -1)).save(
                destination / f"{stage}-alpha-{frame}"
            )
            scores = image_metrics(prediction, target["rgb"], target["mask"])
            if not all(torch.isfinite(value).all() for value in scores.values()):
                raise ValueError("Nonfinite image scores")
            records[stage].append(
                {
                    "scene": scene,
                    "frame": frame,
                    "metrics": {key: float(value[0]) for key, value in scores.items()},
                }
            )
            del gaussians, target, prediction
        by_scene.setdefault(scene, []).append(
            {"frame": frame, "reference_names": item["reference_names"]}
        )
        print(
            json.dumps({"event": "rendered", "scene": scene, "frame": frame, "index": index + 1}),
            flush=True,
        )
    galleries = []
    selection = {}
    for scene, rows in by_scene.items():
        destination = args.output / scene
        panels = []
        for row in rows:
            frame = row["frame"]
            images = [
                Image.open(destination / f"{kind}-{frame}").convert("RGB")
                for kind in ("target", "identity", "animator")
            ]
            title = f"{scene} | {args.split} {frame} | identity {identity_update:,} / animator {animator_update:,}"
            panels.append(comparison_panel(references[scene], *images, title))
        chosen = sorted({0, len(rows) // 2, len(rows) - 1})
        selection[scene] = [rows[index]["frame"] for index in chosen]
        stack_panels([panels[index] for index in chosen]).save(
            destination / "first-middle-last.jpg", quality=94
        )
        # Held-out indices are sparse; playback speed is illustrative, not capture FPS.
        panels[0].save(
            destination / "held-out-sequence.gif",
            save_all=True,
            append_images=panels[1:],
            duration=500,
            loop=0,
        )
        galleries.append(panels[len(rows) // 2])
    for index in range(0, len(galleries), 3):
        stack_panels(galleries[index : index + 3]).save(
            args.output / f"overview-{index // 3 + 1}.jpg", quality=95
        )
    torch.cuda.synchronize()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "gpu": torch.cuda.get_device_name(),
        "seconds": time.perf_counter() - started,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "animator_update": animator_update,
        "identity_checkpoint": str(args.identity_checkpoint),
        "identity_checkpoint_sha256": identity_hash,
        "identity_update": identity_update,
        "frozen_identity_tensors_equal": identity_tensors,
        "config_sha256": file_sha256(args.config),
        "manifest_sha256": manifest_hash,
        "script_sha256": file_sha256(__file__),
        "snapshot_sha256": file_sha256(args.output / "inference-snapshot.pt"),
        "split": args.split,
        "frames": len(data),
        "selection": selection,
        "selection_rule": "Every split frame rendered; first/middle/last per-scene panels; middle per-scene overview, independent of scores",
        "protocol": "Seen-sequence, frame-held-out, fixed training references, annotated foreground crops on white",
        "identity_render": "Identity model plus fitted SMPL pose/betas/body_to_camera; not image-driven animation",
        "animator_render": "Frozen identity plus cached DINO driving-image features; no target pose used by animator forward",
        "playback": "Sparse held-out frames shown in split order at 2 fps; capture timing is unverified",
        "input_features": "Existing locally prepared Sapiens / DINOv2 / DINOv3 caches; no encoder substitution",
        "initial_free_gpu_bytes": free_bytes,
        "total_gpu_bytes": total_bytes,
        "max_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "allocator_fraction_cap": 0.25,
        "metrics_without_lpips": {
            stage: aggregate_records(rows) for stage, rows in records.items()
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "seconds": report["seconds"],
                "max_allocated_gpu_bytes": report["max_allocated_gpu_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
