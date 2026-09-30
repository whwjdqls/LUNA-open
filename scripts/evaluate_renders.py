"""Common NeuMan evaluator for external baseline renders.

Expected <renders>/<scene>/rgb/<frame>.png and alpha/<frame>.png. RGB must
already be composited onto white with predicted alpha; no target-mask cleanup.
All expected frames must exist at the exact evaluation resolution.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from luna_open.data.neuman import NeuManDataset
from luna_open.metrics import aggregate_records, image_metrics
from luna_open.perceptual import build_lpips
from luna_open.provenance import verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--regional-l1",
        action="store_true",
        help="Also report foreground/background L1 without modifying predictions",
    )
    args = parser.parse_args()
    verify_sources(args.root, json.loads(args.manifest.read_text()))
    model = build_lpips(device=args.device)
    data = NeuManDataset(args.root, args.manifest, split=args.split, size=512)
    method = json.loads((args.renders / "method.json").read_text())
    required = ("method", "checkpoint", "reference_count", "reference_frames", "manifest_sha256")
    if any(k not in method for k in required):
        raise ValueError(f"method.json requires {required}")
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    if method["manifest_sha256"] != manifest_hash:
        raise ValueError("Baseline data manifest mismatch")
    count = method["reference_count"]
    if count not in (1, 4):
        raise ValueError("Initial protocol permits 1-reference LHM or 4-reference models")
    for scene, info in data.metadata.items():
        if method["reference_frames"][scene] != info["references"][:count]:
            raise ValueError(f"Baseline reference selection mismatch: {scene}")
    records = []
    for scene, name in data.items:
        sample = data.load_frame(scene, name)
        rgb = np.asarray(Image.open(args.renders / scene / "rgb" / name).convert("RGB")).copy()
        alpha = np.asarray(Image.open(args.renders / scene / "alpha" / name).convert("L")).copy()
        if rgb.shape != (512, 512, 3) or alpha.shape != (512, 512):
            raise ValueError("Baseline renders must match 512x512 evaluation coordinates")
        prediction = dict(
            rgb=torch.from_numpy(rgb).permute(2, 0, 1)[None].to(args.device).float() / 255,
            alpha=torch.from_numpy(alpha)[None, None].to(args.device).float() / 255,
        )
        scores = image_metrics(
            prediction,
            sample["rgb"][None].to(args.device),
            sample["mask"][None].to(args.device),
            model,
        )
        if args.regional_l1:
            error = (prediction["rgb"] - sample["rgb"][None].to(args.device)).abs()
            foreground = sample["mask"][None].to(args.device).expand_as(error)
            for label, region in (("foreground_l1", foreground), ("background_l1", 1 - foreground)):
                scores[label] = (error * region).sum((1, 2, 3)) / region.sum((1, 2, 3)).clamp_min(1)
        records.append(
            dict(scene=scene, frame=name, metrics={k: float(v[0]) for k, v in scores.items()})
        )
    result = aggregate_records(records)
    result.update(
        method=method, split=args.split, metric_protocol="512-square white human crop; LPIPS-Alex"
    )
    if args.regional_l1:
        result["regional_l1_protocol"] = (
            "RGB absolute error weighted by target foreground/background masks; predictions unchanged"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["mean_over_scenes"], indent=2))


if __name__ == "__main__":
    main()
