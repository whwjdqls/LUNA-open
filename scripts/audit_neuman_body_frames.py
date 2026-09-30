"""Audit canonical SMPL/SMPL-X origins and frozen conversion/crop conventions."""

import json
import os
import socket
from pathlib import Path

import numpy as np
import smplx
import torch


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    torch.set_num_threads(2)
    work = Path("/scratch2/whwjdqls99/LUNA-open")
    base = work / "baselines/lhm-20260928"
    fits = json.loads((base / "fit-test/fits.json").read_text())
    protocol = json.loads((base / "protocol-test/protocol.json").read_text())
    manifest = json.loads((work / "data/neuman/manifest-v2.json").read_text())
    body = smplx.SMPL(fits["files"]["smpl"]["path"], num_betas=10).eval()
    target = smplx.SMPLX(
        fits["files"]["smplx"]["path"],
        num_betas=10,
        use_pca=False,
        flat_hand_mean=True,
        num_expression_coeffs=10,
    ).eval()
    body.posedirs.zero_()
    target.posedirs.zero_()
    results = {}
    with torch.no_grad():
        for scene, info in protocol["scenes"].items():
            name = info["references"][0]
            ann = info["annotations"][name]
            fit = fits["scenes"][scene][name]
            a = body(betas=torch.tensor(ann["betas"]).reshape(1, 10))
            b = target(betas=torch.tensor(fit["betas"]).reshape(1, 10))
            delta = a.joints[0, 0] - b.joints[0, 0]
            rows = [r for r in manifest["scenes"][scene]["frames"] if r["name"] in info["targets"]]
            results[scene] = dict(
                reference=name,
                smpl_pelvis=a.joints[0, 0].tolist(),
                smplx_pelvis=b.joints[0, 0].tolist(),
                canonical_root_translation=delta.tolist(),
                fitted_reference_translation=fit["trans"],
                smpl_bounds=torch.stack([a.vertices[0].amin(0), a.vertices[0].amax(0)]).tolist(),
                smplx_bounds=torch.stack([b.vertices[0].amin(0), b.vertices[0].amax(0)]).tolist(),
                scene_mean_native_frame_fg_fraction=float(
                    np.mean([r["foreground_fraction"] for r in rows])
                ),
                native_canvas=[rows[0]["width"], rows[0]["height"]],
                ideal_canvas_psnr_delta_db=float(
                    np.mean(
                        [
                            10
                            * np.log10(
                                r["width"]
                                * r["height"]
                                / (r["crop_xyxy"][2] - r["crop_xyxy"][0]) ** 2
                            )
                            for r in rows
                        ]
                    )
                ),
            )
    out = work / "diagnostics/body-frames-20260928.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                host=socket.gethostname(),
                job=os.environ["SLURM_JOB_ID"],
                scenes=results,
                scope="Canonical pelvis alignment only; no RGB fitting, no scale/rotation fitted. Ideal canvas PSNR delta assumes identical continuous error field, no boundary clipping/resampling.",
            ),
            indent=2,
        )
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
