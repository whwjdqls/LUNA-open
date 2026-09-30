"""Cache real frozen features, recording preprocessing and checkpoint revisions."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from luna_open.data import dataset_from_manifest
from luna_open.data.neuman import load_mask, mask_path
from luna_open.features import DinoFeatures, SapiensFeatures
from luna_open.provenance import verify_sources


def face_image(dataset, scene, name):
    if hasattr(dataset, "face_image"):
        return dataset.face_image(scene, name)
    root = dataset.root / scene
    keypoints_path = root / "keypoints" / (name + ".npy")
    keypoints = np.load(keypoints_path, allow_pickle=False)[:5]
    visible = keypoints[keypoints[:, 2] >= 0.3, :2]
    image = np.asarray(Image.open(root / "images" / name).convert("RGB")).copy()
    image[load_mask(mask_path(root, name)) == 0] = 255
    body_box = dataset.lookup[scene][name]["crop_xyxy"]
    if len(visible) >= 2:
        center = (visible.max(0) + visible.min(0)) / 2
        side = max(float(np.ptp(visible, axis=0).max()) * 2.5, (body_box[2] - body_box[0]) * 0.12)
        source = "supplied_COCO_face_keypoints"
    else:
        side = (body_box[2] - body_box[0]) * 0.35
        center = np.array([(body_box[0] + body_box[2]) / 2, body_box[1] + side / 2])
        source = "top_of_body_crop_fallback"
    left, top = np.floor(center - side / 2).astype(int)
    box = (int(left), int(top), int(left + np.ceil(side)), int(top + np.ceil(side)))
    crop = Image.fromarray(image).crop(box).resize((448, 448), Image.Resampling.BICUBIC)
    # Paste onto a white padded canvas to avoid PIL's black out-of-bounds fill.
    valid = Image.new("L", (image.shape[1], image.shape[0]), 255).crop(box)
    valid = np.asarray(valid.resize((448, 448), Image.Resampling.NEAREST)) > 0
    arr = np.asarray(crop).copy()
    arr[~valid] = 255
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=["body", "face", "motion"], required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-only limit; default all frames")
    parser.add_argument(
        "--max-cache-gib",
        type=float,
        default=32,
        help="Per-kind DNA cache budget, including existing selected features",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction must run in a GPU allocation")
    document = json.loads(args.manifest.read_text())
    verify_sources(args.data_root, document)
    dna = document.get("dataset") == "dna_rendering"
    if dna:
        if args.max_cache_gib <= 0:
            raise ValueError("DNA cache budget must be positive")
        elements = dict(body=4096 * 1536, face=4 * 1024 * 1024, motion=1024 * 1024)
        per_frame = elements[args.kind] * 2 + 1024**2  # serialization/headroom allowance
        rows = [
            (scene, row) for scene, info in document["scenes"].items() for row in info["frames"]
        ]
        if len(rows) * per_frame > args.max_cache_gib * 1024**3:
            raise ValueError(
                "Selected DNA observations exceed the cache budget; reduce the sampling plan"
            )
        args.output.mkdir(parents=True, exist_ok=True)
        missing = sum(
            not (args.output / args.kind / scene / (Path(row["name"]).stem + ".pt")).exists()
            for scene, row in rows
        )
        if missing * per_frame + 20 * 1024**3 > shutil.disk_usage(args.output).free:
            raise ValueError("Insufficient free storage for DNA features plus 20 GiB reserve")
    catalog = yaml.safe_load(Path("configs/assets.yaml").read_text())
    key = dict(body="sapiens_body", face="dino_face", motion="dino_motion")[args.kind]
    metadata = dict(
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        asset=catalog[key],
        preprocessing_version=2 if dna else 1,
        kind=args.kind,
        face_crop=(
            document["preprocessing"]["face_crop"]
            if dna
            else "supplied keypoints confidence>=.3, 2.5x bbox, fallback upper35%"
        ),
        dtype="float16",
        checkpoint_inputs="frozen real pretrained features",
    )
    folder = args.output / args.kind
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "metadata.json"
    content = json.dumps(metadata, indent=2) + "\n"
    if receipt.exists() and receipt.read_text() != content:
        raise ValueError("Incompatible feature cache; choose a new output directory")
    receipt.write_text(content)
    directory = args.assets / key
    if args.kind == "body":
        model = SapiensFeatures(directory / catalog[key]["files"][0])
    else:
        model = DinoFeatures(directory, args.kind)
    model = model.cuda().eval()
    data = dataset_from_manifest(args.data_root, args.manifest, size=1024, verify=False)
    count = 0
    for scene, info in data.metadata.items():
        for row in info["frames"]:
            name = row["name"]
            dest = folder / scene / (Path(name).stem + ".pt")
            if dest.exists():
                continue
            if args.kind == "face":
                image, crop_source = face_image(data, scene, name)
            else:
                image = data.load_frame(scene, name)["rgb"]
                crop_source = "body_mask_bbox"
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                features = model(image[None].cuda())[0].float()
            if not torch.isfinite(features).all():
                raise FloatingPointError(f"Nonfinite features: {scene}/{name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_suffix(".part")
            torch.save(dict(features=features.cpu().half(), crop_source=crop_source), temporary)
            temporary.replace(dest)
            count += 1
            if count == 1 or count % 25 == 0:
                print(scene, name, list(features.shape), "cached", count, flush=True)
            if args.limit and count >= args.limit:
                return


if __name__ == "__main__":
    main()
