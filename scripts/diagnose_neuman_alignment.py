"""Measure global alignment sensitivity of frozen NeuMan renders on compute nodes.

Diagnostic oracle only: each bounded 2D warp is optimized against test RGB.
This does not change official benchmark scores or establish a deployable method.
Image coordinates are pixels, +x right/+y down; matrices map source to target.
RGB and predicted alpha receive the same warp, white/zero outside the canvas.
"""

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import socket
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
BASE = WORK / "baselines/lhm-20260928"
METHODS = {
    "lhm": BASE / "lhm-500m-test",
    "lhmpp": WORK / "baselines/lhmpp-20260928/lhmpp-700m-test",
    "ours": BASE / "ours-identity-14750-test",
}
BOUNDS = [(-64, 64), (-64, 64), (0.7, 1.3), (-15, 15), (0.7, 1.3), (-0.15, 0.15)]


def read_image(path, gray=False):
    flags = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    im = cv2.imread(str(path), flags)
    if im is None:
        raise FileNotFoundError(path)
    if not gray:
        im = im[..., ::-1]
    return im.astype(np.float32) / 255


def matrix(p, size=512):
    dx, dy = p[:2]
    sx = p[2] if len(p) > 2 else 1
    angle = np.deg2rad(p[3]) if len(p) > 3 else 0
    sy = p[4] if len(p) > 4 else sx
    shear = p[5] if len(p) > 5 else 0
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    A = rotation @ np.array([[sx, shear], [0, sy]])
    center = np.full(2, (size - 1) / 2)
    t = center - A @ center + np.array([dx, dy]) * size / 512
    return np.c_[A, t].astype(np.float32)


def warp(im, p, background=1):
    return cv2.warpAffine(
        im,
        matrix(p, im.shape[0]),
        (im.shape[1], im.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(background,) * 3 if im.ndim == 3 else background,
    )


def metrics(pred, gt, alpha, mask):
    error = np.abs(pred - gt).astype(np.float64)
    mse = float((error**2).mean())
    fg, pf = mask > 0.5, alpha > 0.5
    return dict(
        psnr=float(-10 * np.log10(max(mse, 1e-10))),
        mse=mse,
        l1=float(error.mean()),
        foreground_l1=float(error[fg].mean()),
        background_l1=float(error[~fg].mean()),
        mask_iou=float((fg & pf).sum() / max(1, (fg | pf).sum())),
    )


def fit(pred, gt, initials, count):
    # Coarse-to-fine bounded Powell; candidates retained by full-resolution MSE.
    candidates = [np.array(p[:count], dtype=float) for p in initials]
    trace = []
    for size in (128, 256):
        small = cv2.resize(pred, (size, size), interpolation=cv2.INTER_AREA)
        target = cv2.resize(gt, (size, size), interpolation=cv2.INTER_AREA)

        def objective(p):
            return float(np.mean((warp(small, p) - target) ** 2))

        if size == 256:
            candidates = sorted(candidates, key=objective)[:2]
        optimized = []
        for p in candidates:
            result = minimize(
                objective,
                p,
                method="Powell",
                bounds=BOUNDS[:count],
                options=dict(maxiter=45, xtol=0.002, ftol=1e-6),
            )
            optimized.append(result.x)
            trace.append(dict(size=size, success=bool(result.success), nfev=result.nfev))
        candidates.extend(optimized)
    best = min(candidates, key=lambda p: float(np.mean((warp(pred, p) - gt) ** 2)))
    return best, trace


def one(task):
    cv2.setNumThreads(1)
    method, scene, frame, output = task
    gt = read_image(BASE / "protocol-test" / scene / "rgb" / frame)
    mask = read_image(BASE / "protocol-test" / scene / "mask" / frame, True)
    pred = read_image(METHODS[method] / scene / "rgb" / frame)
    alpha = read_image(METHODS[method] / scene / "alpha" / frame, True)
    fg, pf = mask > 0.5, alpha > 0.5
    yy, xx = np.nonzero(fg)
    py, px = np.nonzero(pf)
    shift = np.clip([xx.mean() - px.mean(), yy.mean() - py.mean()], -63, 63)
    scale = float(np.clip((yy.max() - yy.min()) / max(1, py.max() - py.min()), 0.71, 1.29))
    identity = [0, 0, 1, 0, 1, 0]
    eroded = cv2.erode(fg.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    dilated = cv2.dilate(fg.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    regions = dict(
        fg_interior=eroded,
        boundary_band=dilated & ~eroded,
        outside_dilated=~dilated,
        mask_disagreement=fg ^ pf,
    )
    sq = (pred.astype(np.float64) - gt) ** 2
    energy = {k: float(sq[v].sum() / sq.sum()) for k, v in regions.items()}
    result = dict(
        method=method,
        scene=scene,
        frame=frame,
        foreground_fraction=float(fg.mean()),
        mse_energy=energy,
        mask_centroid_shift_pixels=shift.tolist(),
        bbox_height_ratio=scale,
        variants={"raw": dict(metrics=metrics(pred, gt, alpha, mask), parameters=identity)},
    )
    p, trace = fit(pred, gt, [identity, [*shift, 1, 0, 1, 0]], 2)
    previous = [*p, 1, 0, 1, 0]
    for mode, count in (("translation", 2), ("similarity", 4), ("affine", 6)):
        if mode != "translation":
            initials = [identity, previous, [*shift, scale, 0, scale, 0]]
            p, trace = fit(pred, gt, initials, count)
        rgb, a = warp(pred, p), warp(alpha, p, 0)
        directory = Path(output) / f"{method}-{mode}" / scene
        for kind, im in (("rgb", rgb), ("alpha", a)):
            (directory / kind).mkdir(parents=True, exist_ok=True)
            im = np.round(np.clip(im, 0, 1) * 255).astype(np.uint8)
            if im.ndim == 3:
                im = im[..., ::-1]
            assert cv2.imwrite(str(directory / kind / frame), im)
        result["variants"][mode] = dict(
            metrics=metrics(rgb, gt, a, mask),
            parameters=p.tolist(),
            source_to_target=matrix(p).tolist(),
            optimization_trace=trace,
        )
        previous = [*p, p[2], 0] if count == 4 else [*p, 1, 0, 1, 0] if count == 2 else p
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((BASE / "protocol-test/protocol.json").read_text())
    # Verify warp conventions with a known synthetic source-to-target translation.
    synthetic = np.ones((512, 512, 3), np.float32)
    synthetic[190:240, 230:275] = [0.2, 0.4, 0.8]
    target = warp(synthetic, [11, -7])
    estimated, _ = fit(synthetic, target, [[0, 0], [10, -6]], 2)
    assert np.max(np.abs(estimated - [11, -7])) < 0.15, estimated
    tasks = [
        (m, s, f, str(args.output))
        for m in METHODS
        for s, v in protocol["scenes"].items()
        for f in v["targets"]
    ]
    records = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for record in pool.map(one, tasks):
            records.append(record)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            gain = (
                record["variants"]["similarity"]["metrics"]["psnr"]
                - record["variants"]["raw"]["metrics"]["psnr"]
            )
            print(
                f"{len(records)}/{len(tasks)} {record['method']}/{record['scene']}/{record['frame']} similarity {gain:+.3f} dB",
                flush=True,
            )
    summary = {}
    for method, root in METHODS.items():
        summary[method] = {}
        for mode in ("raw", "translation", "similarity", "affine"):
            per_scene = {}
            for scene in protocol["scenes"]:
                rs = [
                    r["variants"][mode]["metrics"]
                    for r in records
                    if r["method"] == method and r["scene"] == scene
                ]
                per_scene[scene] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            summary[method][mode] = dict(
                per_scene=per_scene,
                mean_over_scenes={
                    k: float(np.mean([v[k] for v in per_scene.values()])) for k in rs[0]
                },
            )
            if mode != "raw":
                metadata = json.loads((root / "method.json").read_text())
                metadata.update(
                    method=f"DIAGNOSTIC ORACLE {method}: test-RGB-fitted 2D {mode}",
                    diagnostic_only=True,
                    fit_uses_test_rgb=True,
                    original_renders=str(root),
                    alignment_bounds=BOUNDS,
                )
                (args.output / f"{method}-{mode}/method.json").write_text(
                    json.dumps(metadata, indent=2)
                )
    receipt = dict(
        host=socket.gethostname(),
        job=os.environ["SLURM_JOB_ID"],
        diagnostic_only=True,
        synthetic_translation_recovered=estimated.tolist(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        bounds=BOUNDS,
        records=records,
        summary=summary,
    )
    (args.output / "alignment.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
