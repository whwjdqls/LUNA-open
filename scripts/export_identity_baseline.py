"""Render a frozen local identity checkpoint through SMPL for baseline evaluation."""

import argparse
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
from cache_features import face_image
from PIL import Image
from render_identity_animation import panel, virtual_camera
from render_qualitative import load_snapshot, rgb_image

from luna_open.avatar import Gaussians
from luna_open.data.neuman import NeuManDataset
from luna_open.model import IdentityEncoder, ModelConfig
from luna_open.provenance import file_sha256, validate_body_asset, verify_sources
from luna_open.rendering import render
from luna_open.smpl import SMPLTeacher
from luna_open.training import FeatureStore, prepare_item, seed_all


@torch.inference_mode()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--selection", default="frozen progress-report checkpoint, selected by validation LPIPS"
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    started = time.perf_counter()
    state, checksum = load_snapshot(args.checkpoint)
    cfg = state["config"]
    manifest = json.loads(Path(cfg["manifest"]).read_text())
    verify_sources(Path(cfg["data_root"]), manifest)
    assert state["manifest_sha256"] == file_sha256(cfg["manifest"])
    validate_body_asset(state, file_sha256(cfg["smpl_model"]))
    seed_all(cfg["seed"])
    teacher = SMPLTeacher(
        cfg["smpl_model"],
        cfg["num_queries"],
        cfg["seed"],
        pose_blend_shapes=cfg["smpl_pose_blend_shapes"],
    ).cuda()
    assert torch.equal(teacher.anchors.cpu(), state["identity"]["anchors"])
    identity = IdentityEncoder(
        teacher.anchors, teacher.semantic_labels, ModelConfig(**cfg["model"])
    )
    identity.load_state_dict(state["identity"], strict=True)
    identity = identity.cuda().eval().requires_grad_(False)
    features = FeatureStore(cfg["features"], cfg["manifest"], ("body", "face"))
    assert features.metadata == state["feature_metadata"]
    data = NeuManDataset(cfg["data_root"], cfg["manifest"], split=args.split, size=512)
    args.output.mkdir(parents=True)
    method = dict(
        method=args.name,
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=checksum,
        update=state["update"],
        split=args.split,
        reference_count=4,
        reference_frames={scene: info["references"] for scene, info in data.metadata.items()},
        manifest_sha256=state["manifest_sha256"],
        renderer="gsplat 1.5.3; luna_open.rendering",
        animation="fixed SMPL surface-anchor weights, annotation-driven LBS teacher",
        training_overlap="trained on the 344 training frames of these same six NeuMan identities",
        checkpoint_selection=args.selection,
        rgb_encoding="lossless PNG, 8-bit RGB, already white-composited; no GT-mask cleanup",
        alpha_encoding="lossless PNG, 8-bit predicted alpha",
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(),
        config=cfg,
    )
    template = Gaussians(
        teacher.anchors[None], None, torch.full_like(teacher.anchors[None], 0.008), None, None
    )
    K_virtual, viewmat = virtual_camera(template, 512, 0)
    (args.output / "canonical-camera.json").write_text(
        json.dumps(
            dict(
                K=K_virtual[0].tolist(),
                world_to_camera=viewmat[0].tolist(),
                width=512,
                height=512,
                convention="metric canonical body coordinates, +Y up, +Z front; OpenCV camera",
            ),
            indent=2,
        )
        + "\n"
    )
    for scene, info in data.metadata.items():
        scene_dir = args.output / scene
        for subdir in ("rgb", "alpha", "references", "qualitative"):
            (scene_dir / subdir).mkdir(parents=True)
        assert set(info["references"]).isdisjoint(info["splits"][args.split])
        body, face = features.references(dict(scene=scene, reference_names=info["references"]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            canonical = identity(body, face)
        canonical.gaussians.validate()
        canonical_image = rgb_image(
            render(canonical.gaussians, K_virtual, (512, 512), viewmat)["rgb"][0]
        )
        canonical_image.save(scene_dir / "qualitative/canonical-front.png")
        face_meta = {}
        for name in info["references"]:
            rgb_image(data.load_frame(scene, name)["rgb"]).save(
                scene_dir / "references" / name, format="PNG"
            )
            crop, source = face_image(data, scene, name, size=112)
            rgb_image(crop).save(scene_dir / "references" / f"face-{name}", format="PNG")
            face_meta[name] = source
        (scene_dir / "references/face-crops.json").write_text(
            json.dumps(face_meta, indent=2) + "\n"
        )
        frames = []
        for name in info["splits"][args.split]:
            target = prepare_item(data.load_frame(scene, name))
            posed = teacher(
                canonical.gaussians, target["pose"], target["betas"], target["body_to_camera"]
            )
            prediction = render(posed, target["K"], (512, 512))
            rgb = rgb_image(prediction["rgb"][0])
            rgb.save(scene_dir / "rgb" / name, format="PNG")
            alpha = prediction["alpha"][0, 0].clamp(0, 1).cpu().numpy()
            Image.fromarray(np.round(alpha * 255).astype(np.uint8)).save(
                scene_dir / "alpha" / name, format="PNG"
            )
            frame = panel(
                [rgb_image(target["rgb"][0]), canonical_image, rgb],
                ("Ground truth", "Canonical identity", "Fitted SMPL / LBS"),
                f"{scene} | {name} | {args.name}",
                f"Official {args.split}; update {state['update']}; 4 fixed training references",
                tile=512,
            )
            frame.save(scene_dir / "qualitative" / name, format="PNG")
            frames.append(frame)
        frames[0].save(
            scene_dir / "qualitative/gt-canonical-lbs.gif",
            save_all=True,
            append_images=frames[1:],
            duration=300,
            loop=0,
        )
        print(f"Rendered {scene}: {len(frames)} held-out frames", flush=True)
    method["elapsed_seconds"] = time.perf_counter() - started
    method["peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    (args.output / "method.json").write_text(json.dumps(method, indent=2) + "\n")
    print(
        json.dumps(
            {key: method[key] for key in ("method", "update", "elapsed_seconds", "peak_cuda_bytes")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
