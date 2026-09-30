"""Score every diagnostic image using the same seven metrics as the frozen benchmark."""

import argparse
import json
import os
import socket
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from luna_open.metrics import aggregate_records, image_metrics
from luna_open.perceptual import build_lpips

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
BASE = WORK / "baselines/lhm-20260928"


def tensor(path, gray=False):
    a = np.asarray(Image.open(path).convert("L" if gray else "RGB")).copy()
    if gray:
        return torch.from_numpy(a)[None, None].float().cuda() / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda() / 255


@torch.inference_mode()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["alignment", "native", "canvas", "pose", "geometry", "shape"], required=True)
    parser.add_argument("--method", choices=["lhm", "lhmpp"], help="Select one pose-oracle method")
    args = parser.parse_args()
    torch.set_num_threads(3)
    lpips = build_lpips(device="cuda")
    protocol = json.loads((BASE / "protocol-test/protocol.json").read_text())
    variants = {}
    if args.group == "alignment":
        variants = {
            f"{m}-{v}": WORK / "diagnostics/alignment-20260928" / f"{m}-{v}"
            for m in ("lhm", "lhmpp", "ours")
            for v in ("translation", "similarity", "affine")
        }
    elif args.group == "native":
        variants = {
            f"{m}-{v}": WORK / f"diagnostics/native-{m}-20260928" / v
            for m in ("lhm", "lhmpp")
            for v in ("repeat", "zero-shape", "fixed-shape", "recrop")
        }
    elif args.group == "pose":
        selected = [args.method] if args.method else ["lhm", "lhmpp"]
        for method in selected:
            for mode in ("root", "articulated"):
                marker = WORK / f"diagnostics/pose-oracle-{method}-20260928" / mode / "method.json"
                if not marker.exists():
                    raise RuntimeError(f"Pose export is not complete: {marker}")
        variants = {
            f"{m}-{v}": WORK / f"diagnostics/pose-oracle-{m}-20260928" / v
            for m in selected
            for v in ("repeat", "root", "articulated")
            if (WORK / f"diagnostics/pose-oracle-{m}-20260928" / v / "method.json").exists()
        }
    elif args.group == "geometry":
        variants = {
            f"{m}-repeat": WORK / "diagnostics/smplx-audit-20260929" / f"learned-{m}" / "renders"
            for m in ("lhm", "lhmpp")
        }
    elif args.group == "shape":
        shape_methods = ["lhm", "lhmpp"]
        for candidate in ("lhm-wide", "lhmpp-wide"):
            marker = WORK / "diagnostics/shared-shape-20260929" / candidate / "audit.json"
            if marker.exists():
                if not json.loads(marker.read_text()).get("complete"):
                    raise RuntimeError(f"Shape sensitivity experiment incomplete: {marker}")
                shape_methods.append(candidate)
        variants = {
            f"{m}-{v}": WORK / "diagnostics/shared-shape-20260929" / m / v
            for m in shape_methods
            for v in ("repeat", "shared-initial", "shared-fitted")
        }
    else:
        variants = {
            f"{m}-{v}": WORK / "diagnostics/canvas-20260928" / f"{m}-{v}"
            for m in ("lhm", "lhmpp", "ours")
            for v in ("raw", "similarity")
        }
    out = WORK / ({"geometry": "diagnostics/smplx-audit-20260929/metrics",
                   "shape": "diagnostics/shared-shape-20260929/metrics"}.get(args.group,
                   "diagnostics/alignment-metrics-20260928"))
    out.mkdir(parents=True, exist_ok=True)
    for name, root in variants.items():
        destination = out / f"{args.group}-{name}.json"
        if destination.exists():
            raise FileExistsError(destination)
        records = []
        for scene, info in protocol["scenes"].items():
            for frame in info["targets"]:
                if args.group == "canvas":
                    gt = tensor(WORK / "diagnostics/canvas-20260928/gt" / scene / frame)
                    mask = tensor(
                        WORK / "diagnostics/canvas-20260928/gt" / scene / f"mask-{frame}", True
                    )
                else:
                    gt = tensor(BASE / "protocol-test" / scene / "rgb" / frame)
                    mask = tensor(BASE / "protocol-test" / scene / "mask" / frame, True)
                pred = dict(
                    rgb=tensor(root / scene / "rgb" / frame),
                    alpha=tensor(root / scene / "alpha" / frame, True),
                )
                assert pred["rgb"].shape == gt.shape
                scores = image_metrics(pred, gt, mask, lpips)
                error = (pred["rgb"] - gt).abs()
                for key, weights in (("foreground_l1", mask), ("background_l1", 1 - mask)):
                    scores[key] = (error * weights).sum() / (weights.sum() * 3).clamp_min(1)
                records.append(
                    dict(
                        scene=scene,
                        frame=frame,
                        metrics={k: float(v.flatten()[0]) for k, v in scores.items()},
                    )
                )
        result = aggregate_records(records)
        result.update(
            group=args.group,
            variant=name,
            root=str(root),
            diagnostic_only=True,
            test_rgb_optimized=args.group == "alignment"
            or (args.group == "pose" and not name.endswith("repeat"))
            or name.endswith("similarity"),
            host=socket.gethostname(),
            job=os.environ["SLURM_JOB_ID"],
            lpips="AlexNet, [-1,1]",
            aggregation="equal scene means",
            resolution="native" if args.group == "canvas" else "512x512",
        )
        destination.write_text(json.dumps(result, indent=2))
        print(name, result["mean_over_scenes"], flush=True)


if __name__ == "__main__":
    main()
