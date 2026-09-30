"""Re-express frozen crop predictions on the native NeuMan image canvas.

Diagnostic protocol sensitivity, not the undisclosed LUNA scoring implementation.
Prediction placement uses only the recorded crop transform. No RGB alignment or
GT-mask cleanup is applied. Similarity inputs are explicitly test-RGB oracle fits.
"""

import json
import os
import socket
from pathlib import Path

import numpy as np
from PIL import Image

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
BASE = WORK / "baselines/lhm-20260928"
ROOTS = dict(
    lhm=BASE / "lhm-500m-test",
    lhmpp=WORK / "baselines/lhmpp-20260928/lhmpp-700m-test",
    ours=BASE / "ours-identity-14750-test",
)


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    out = WORK / "diagnostics/canvas-20260928"
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((WORK / "data/neuman/manifest-v2.json").read_text())
    records = []
    for scene, info in manifest["scenes"].items():
        for row in info["frames"]:
            name = row["name"]
            if name not in info["splits"]["test"]:
                continue
            original = WORK / "data/neuman/dataset" / scene
            im = Image.open(original / "images" / name).convert("RGB")
            mp = original / "segmentations" / (name + ".npy")
            mask = (
                np.load(mp).squeeze() == 0
                if mp.exists()
                else np.asarray(Image.open(original / "segmentations" / name)) == 0
            )
            gt = Image.fromarray(np.where(mask[..., None], np.asarray(im), 255).astype(np.uint8))
            gt_dir = out / "gt" / scene
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt.save(gt_dir / name)
            Image.fromarray((mask * 255).astype(np.uint8)).save(gt_dir / f"mask-{name}")
            left, top, right, bottom = row["crop_xyxy"]
            side = right - left
            for method, root in ROOTS.items():
                for mode in ("raw", "similarity"):
                    source = (
                        root
                        if mode == "raw"
                        else WORK / "diagnostics/alignment-20260928" / f"{method}-similarity"
                    )
                    directory = out / f"{method}-{mode}" / scene
                    for kind, color in (("rgb", "white"), ("alpha", 0)):
                        (directory / kind).mkdir(parents=True, exist_ok=True)
                        crop = Image.open(source / scene / kind / name).resize(
                            (side, side), Image.Resampling.BILINEAR
                        )
                        canvas = Image.new(crop.mode, (row["width"], row["height"]), color)
                        canvas.paste(crop, (left, top))
                        canvas.save(directory / kind / name)
            records.append(
                dict(
                    scene=scene,
                    frame=name,
                    crop_xyxy=row["crop_xyxy"],
                    size=im.size,
                    area_ratio_native_to_crop=row["width"] * row["height"] / side**2,
                )
            )
            print(f"Canvas {scene}/{name}", flush=True)
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                host=socket.gethostname(),
                job=os.environ["SLURM_JOB_ID"],
                records=records,
                diagnostic_only=True,
                no_gt_cleanup=True,
                interpolation="PIL bilinear resize crop back to native pixel extent; paste onto white native canvas",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
