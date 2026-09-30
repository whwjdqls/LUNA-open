"""Real image-only inference using a trained smoke checkpoint.

The dataset supplies foreground crops and camera intrinsics before the tested
inference boundary. A file-access guard rejects body-model/fitting inputs inside
that boundary. Outputs establish execution, not reconstruction quality.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from cache_features import face_image
from PIL import Image

from luna_open.data.neuman import NeuManDataset
from luna_open.pipeline import LUNAPipeline
from luna_open.provenance import file_sha256, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_sources(args.root, json.loads(args.manifest.read_text()))
    data = NeuManDataset(args.root, args.manifest, split="val", size=1024)
    scene, frame = data.items[0]
    refs = data.metadata[scene]["references"]
    references = torch.stack([data.load_frame(scene, name)["rgb"] for name in refs])[None].cuda()
    face_pairs = [face_image(data, scene, name) for name in refs]
    faces = torch.stack([p[0] for p in face_pairs])[None].cuda()
    driver = data.load_frame(scene, frame)["rgb"][None].cuda()
    output_data = NeuManDataset(args.root, args.manifest, split="val", size=512)
    target = output_data.load_frame(scene, frame)
    K = target["K"][None].cuda()

    def reject_body_file(event, arguments):
        if event != "open" or not arguments or not isinstance(arguments[0], (str, bytes)):
            return
        path = os.fsdecode(arguments[0])
        if "SMPL_NEUTRAL" in path or "smpl_output" in path or path.endswith("alignments.npy"):
            raise AssertionError(f"Inference attempted to read a body/fitting asset: {path}")

    sys.addaudithook(reject_body_file)
    torch.cuda.reset_peak_memory_stats()
    model = LUNAPipeline.from_checkpoint(args.checkpoint, args.assets)
    canonical = model.encode_identity(references, faces)
    posed = model.animate(canonical, driver)
    posed.gaussians.validate()
    prediction = model.render(posed, K, (512, 512))
    if prediction["rgb"].shape != (1, 3, 512, 512) or prediction["alpha"].shape != (1, 1, 512, 512):
        raise ValueError("Wrong inference render dimensions")
    if not all(torch.isfinite(value).all() for value in prediction.values()):
        raise ValueError("Nonfinite image-only inference output")
    for name, tensor in (
        ("rgb", prediction["rgb"][0]),
        ("alpha", prediction["alpha"][0].expand(3, -1, -1)),
        ("target", target["rgb"]),
    ):
        pixels = (tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        Image.fromarray(pixels).save(args.output / f"{name}.png")
    np.savez_compressed(
        args.output / "gaussians.npz",
        **{
            key: getattr(posed.gaussians, key)[0].cpu().numpy()
            for key in ("means", "quaternions", "scales", "opacities", "colors")
        },
    )
    torch.cuda.synchronize()
    report = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha256(args.checkpoint),
        manifest_sha256=file_sha256(args.manifest),
        scene=scene,
        frame=frame,
        references=refs,
        face_crop_sources=[p[1] for p in face_pairs],
        image_only_body_file_guard_passed=True,
        rgb_shape=list(prediction["rgb"].shape),
        num_gaussians=posed.gaussians.means.shape[1],
        alpha_min=float(prediction["alpha"].min()),
        alpha_max=float(prediction["alpha"].max()),
        foreground_fraction=float((prediction["alpha"] >= 0.5).float().mean()),
        seconds=time.perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        limitations=[
            "Eight-update training smoke; selected checkpoints may be earlier than eight",
            "Annotations used for crops, faces and K; no pose/body asset read inside inference",
            "Live encoders; no claim of exact parity with FP16 cached features",
        ],
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
