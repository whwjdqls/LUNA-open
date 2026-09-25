"""Check one real pretrained encoder on CPU while a GPU allocation is pending.

This creates no production cache; CUDA/BF16 parity and rendering remain separate.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from cache_features import face_image

from luna_open.data.neuman import NeuManDataset
from luna_open.features import DinoFeatures, SapiensFeatures
from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("face", "body", "motion"), default="face")
    args = parser.parse_args()
    start = time.monotonic()
    dataset = NeuManDataset(args.root, args.manifest, size=1024)
    scene, frame = dataset.items[0]
    if args.kind == "face":
        pixels, crop_source = face_image(dataset, scene, frame)
        model = DinoFeatures(args.assets / "dino_face", "face").eval()
        expected_shape = (1, 4, 1024, 1024)
        encoder = "DINOv2-L/14 registers; actual pretrained weights"
    elif args.kind == "body":
        pixels = dataset.load_frame(scene, frame)["rgb"]
        crop_source = "supplied_foreground_mask_body_crop"
        model = SapiensFeatures(
            args.assets / "sapiens_body" / "sapiens_1b_epoch_173_torchscript.pt2"
        ).eval()
        expected_shape = (1, 4096, 1536)
        encoder = "Sapiens-1B; actual pretrained TorchScript weights"
    else:
        pixels = dataset.load_frame(scene, frame)["rgb"]
        crop_source = "supplied_foreground_mask_body_crop"
        model = DinoFeatures(args.assets / "dino_motion", "motion").eval()
        expected_shape = (1, 1024, 1024)
        encoder = "DINOv3-L/16; actual pretrained weights"
    with torch.inference_mode():
        features = model(pixels[None])
    assert features.shape == expected_shape, features.shape
    assert torch.isfinite(features).all()
    assert features.std() > 0
    asset_key = dict(face="dino_face", body="sapiens_body", motion="dino_motion")[args.kind]
    report = dict(
        encoder=encoder,
        device="cpu",
        dtype=str(features.dtype),
        shape=list(features.shape),
        mean=float(features.mean()),
        std=float(features.std()),
        scene=scene,
        frame=frame,
        crop_source=crop_source,
        manifest_sha256=file_sha256(args.manifest),
        asset_files={
            path.name: dict(bytes=path.stat().st_size, sha256=file_sha256(path))
            for path in sorted((args.assets / asset_key).iterdir())
            if path.is_file()
        },
        elapsed_seconds=time.monotonic() - start,
        limitations="Only the named encoder tested; no GPU/BF16 parity or reconstruction quality tested",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
