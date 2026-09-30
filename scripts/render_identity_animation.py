"""Show canonical IdentityEncoder output and its image-driven animation.

Visualization follows LUNA sections 3.1-3.2: G_can and T_can feed the animator.
The virtual camera is a display choice, not a learned or paper-specified camera.
No fitted target pose or SMPL teacher is used by this script.
"""

import argparse
import gc
import json
import math
import os
import socket
import time
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from render_qualitative import font, load_snapshot, rgb_image, stack_panels

from luna_open.data.neuman import NeuManDataset
from luna_open.model import IdentityEncoder, ModelConfig, NeuralAnimator
from luna_open.provenance import file_sha256, validate_body_asset
from luna_open.rendering import render
from luna_open.training import FeatureStore, seed_all


def virtual_camera(gaussians, size, degrees):
    """Canonical SMPL meters, +Y up, +Z front; OpenCV camera +Z forward, +Y down.

    Returns batched K and world-to-camera matrices. Only the camera changes;
    Gaussian positions, wxyz quaternions, scales, colors and opacity stay fixed.
    """
    means = gaussians.means[0].float()
    center = (means.amin(0) + means.amax(0)) / 2
    radius = (means - center).norm(dim=-1).max() + 3 * gaussians.scales.max()
    focal = 1.5 * size
    distance = radius * 1.12 / math.sin(math.atan((size / 2) / focal))
    angle = math.radians(degrees)
    position = center + distance * means.new_tensor([math.sin(angle), 0, math.cos(angle)])
    forward = torch.nn.functional.normalize(center - position, dim=0)
    right = torch.nn.functional.normalize(
        torch.linalg.cross(forward, means.new_tensor([0, 1, 0])), dim=0
    )
    down = torch.linalg.cross(forward, right)
    rotation = torch.stack((right, down, forward))
    view = torch.eye(4, device=means.device)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ position
    intrinsic = means.new_tensor([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]])
    return intrinsic[None], view[None]


def panel(images, labels, title, footer, tile=384):
    margin, gap, heading = 16, 12, 72
    canvas = Image.new(
        "RGB", (len(images) * tile + (len(images) - 1) * gap + 2 * margin, tile + 108), "#eef1f5"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), title, fill="#17212b", font=font(21))
    for index, (label, picture) in enumerate(zip(labels, images, strict=True)):
        left = margin + index * (tile + gap)
        draw.text((left, 44), label, fill="#17212b", font=font(18))
        canvas.paste(picture.resize((tile, tile), Image.Resampling.LANCZOS), (left, heading))
    draw.text((margin, tile + heading + 8), footer, fill="#465465", font=font(14))
    return canvas


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--preview", type=Path, required=True, help="Preserved qualitative snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Run on a Slurm compute node")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.25)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 8 * 1024**3:
        raise RuntimeError("Less than 8 GiB free in existing allocation")
    started = time.perf_counter()
    cfg = yaml.safe_load(args.config.read_text())
    prior = json.loads((args.preview / "report.json").read_text())
    state, snapshot_hash = load_snapshot(args.preview / "inference-snapshot.pt")
    if snapshot_hash != prior["snapshot_sha256"] or state["config"] != cfg:
        raise ValueError("Snapshot/configuration mismatch")
    if state["manifest_sha256"] != file_sha256(cfg["manifest"]):
        raise ValueError("Manifest mismatch")
    validate_body_asset(state, file_sha256(cfg["smpl_model"]))
    features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face", "motion"))
    if features.metadata != state["feature_metadata"]:
        raise ValueError("Feature metadata mismatch")
    seed_all(cfg["seed"])
    model_config = ModelConfig(**cfg["model"])
    identity = (
        IdentityEncoder(
            state["identity"]["anchors"], state["identity"]["semantic_labels"], model_config
        )
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
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], split="test", size=cfg["image_size"])
    args.output.mkdir(parents=True)
    groups = {}
    for index, (scene, _) in enumerate(data.items):
        groups.setdefault(scene, []).append(index)
    scene_records, overviews = {}, []
    for scene, indices in groups.items():
        destination = args.output / scene
        destination.mkdir()
        first = data[indices[0]]
        body, face = features.references(first)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = identity(body, face)
        canonical.gaussians.validate()
        torch.save(
            {
                "gaussians": {
                    f.name: getattr(canonical.gaussians, f.name).cpu()
                    for f in fields(canonical.gaussians)
                },
                "tokens": canonical.tokens.cpu(),
                "scene": scene,
                "reference_names": first["reference_names"],
                "snapshot_sha256": snapshot_hash,
                "coordinates": "SMPL canonical meters; quaternion wxyz; batch dimension retained",
            },
            destination / "canonical-output.pt",
        )
        camera_records, canonical_views = {}, {}
        for name, angle in (("front", 0), ("side", 90), ("back", 180)):
            intrinsic, view = virtual_camera(canonical.gaussians, cfg["image_size"], angle)
            result = render(
                canonical.gaussians, intrinsic, (cfg["image_size"], cfg["image_size"]), view
            )
            picture = rgb_image(result["rgb"][0])
            picture.save(destination / f"identity-{name}.png")
            canonical_views[name] = picture
            camera_records[name] = {
                "K": intrinsic.cpu().tolist(),
                "world_to_camera": view.cpu().tolist(),
            }
        panel(
            list(canonical_views.values()),
            ("Front", "Side", "Back"),
            f"{scene} | Identity encoder output | step {prior['identity_update']:,}",
            "Canonical 3D Gaussians; virtual camera views; no body pose applied",
        ).save(destination / "identity-views.jpg", quality=95)
        panels, frames = [], []
        for index in indices:
            item = data[index]
            if item["reference_names"] != first["reference_names"]:
                raise ValueError("References changed within scene")
            motion = features.get("motion", scene, item["frame"])[None]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                posed = animator(canonical, motion, global_only=False)
            result = render(
                posed.gaussians, item["K"][None].cuda(), (cfg["image_size"], cfg["image_size"])
            )
            animated = rgb_image(result["rgb"][0])
            driver = rgb_image(item["rgb"])
            animated.save(destination / f"animated-{item['frame']}")
            driver.save(destination / f"driver-{item['frame']}")
            panels.append(
                panel(
                    (canonical_views["front"], driver, animated),
                    ("Identity encoder: canonical", "Driving image", "Neural animator output"),
                    f"{scene} | test {item['frame']} | identity {prior['identity_update']:,} / animator {prior['animator_update']:,}",
                    "Left: fixed virtual view. Right: predicted motion, rendered with the annotated crop camera.",
                )
            )
            frames.append(item["frame"])
        chosen = sorted({0, len(panels) // 2, len(panels) - 1})
        stack_panels([panels[index] for index in chosen]).save(
            destination / "first-middle-last.jpg", quality=95
        )
        panels[len(panels) // 2].save(destination / "comparison.jpg", quality=95)
        panels[0].save(
            destination / "animation.gif",
            save_all=True,
            append_images=panels[1:],
            duration=500,
            loop=0,
        )
        overviews.append(panels[len(panels) // 2])
        scene_records[scene] = {
            "references": first["reference_names"],
            "frames": frames,
            "canonical_views": camera_records,
            "canonical_output_sha256": file_sha256(destination / "canonical-output.pt"),
        }
        print(
            json.dumps({"event": "scene_complete", "scene": scene, "frames": len(frames)}),
            flush=True,
        )
        del canonical, body, face, posed, result, motion
    for offset in range(0, len(overviews), 3):
        stack_panels(overviews[offset : offset + 3]).save(
            args.output / f"overview-{offset // 3 + 1}.jpg", quality=95
        )
    torch.cuda.synchronize()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
        "gpu": torch.cuda.get_device_name(),
        "seconds": time.perf_counter() - started,
        "identity_update": prior["identity_update"],
        "animator_update": prior["animator_update"],
        "snapshot": str(args.preview / "inference-snapshot.pt"),
        "snapshot_sha256": snapshot_hash,
        "source_preview_report_sha256": file_sha256(args.preview / "report.json"),
        "config_sha256": file_sha256(args.config),
        "script_sha256": file_sha256(__file__),
        "model_sha256": file_sha256(Path(__file__).resolve().parents[1] / "src/luna_open/model.py"),
        "max_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "initial_free_gpu_bytes": free_bytes,
        "total_gpu_bytes": total_bytes,
        "canonical": "Direct IdentityEncoder Gaussians and tokens; canonical SMPL coordinates, no teacher deformation",
        "animation": "Same canonical output reused for every driving frame; NeuralAnimator global_only=False; no target pose used",
        "camera": "Canonical: virtual fitted-to-bounds camera, +Y up, +Z front. Animated: annotated foreground crop camera",
        "playback": "Sparse test frames, 2 fps display; not original capture timing or interpolated frames",
        "selection": "All 41 test frames; first/middle/last per scene; middle overview; independent of metrics",
        "known_limitation": "Animator checkpoint 5000 has previously confirmed collapsed local limb motion",
        "scenes": scene_records,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output),
                "seconds": report["seconds"],
                "peak_bytes": report["max_allocated_gpu_bytes"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
