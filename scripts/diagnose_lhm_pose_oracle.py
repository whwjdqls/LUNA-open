"""Freeze native LHM/LHM++ identity and fit bounded target poses to test RGB.

Diagnostic oracle, never an official test score: optimizes root orientation and
translation, then body-joint angles. No appearance/shape/checkpoint optimization.
Root/body angle vectors are additive axis-angle coordinates (radians); translation
is in annotation body-frame meters before the fixed OpenCV camera transform.
"""

import argparse
import json
import math
import os
import socket
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
from diagnose_native_avatar import motion_from_fit
from evaluate_lhm_neuman import blank_motion, image_tensor, save_tensor

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
BASE = WORK / "baselines/lhm-20260928"


def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--method", choices=["lhm", "lhmpp"], default="lhm")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    if args.method == "lhm":
        from evaluate_lhm_neuman import build_model

        model, _ = build_model(BASE / "source", WORK / "assets/lhm_500m", args.output)
    else:
        from evaluate_lhmpp_neuman import build_model
        from evaluate_lhmpp_neuman import render as render_pp

        model, _ = build_model(
            WORK / "baselines/lhmpp-20260928/source", WORK / "assets/lhmpp_700m", args.output
        )
    protocol = json.loads((BASE / "protocol-test/protocol.json").read_text())
    fits = json.loads((BASE / "fit-test/fits.json").read_text())
    records = []
    for scene, info in protocol["scenes"].items():
        first = info["references"][0]
        with torch.no_grad():
            if args.method == "lhm":
                cache = model.infer_single_view(
                    image_tensor(BASE / "protocol-test" / scene / "rgb" / first),
                    image_tensor(
                        BASE / "ours-identity-14750-test" / scene / "references" / f"face-{first}"
                    ),
                    None,
                    None,
                    None,
                    None,
                    None,
                    blank_motion(),
                )
            else:
                refs = torch.cat(
                    [
                        image_tensor(BASE / "protocol-test" / scene / "rgb" / name)
                        for name in info["references"]
                    ],
                    dim=1,
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cache = model.infer_single_view(
                        refs,
                        None,
                        None,
                        torch.eye(4, device="cuda")[None, None],
                        torch.tensor([[768.0, 0, 256], [0, 768, 256], [0, 0, 1]], device="cuda")[
                            None, None
                        ],
                        torch.ones(1, 1, 3, device="cuda"),
                        blank_motion(),
                        ref_imgs_bool=torch.ones(1, 4, device="cuda", dtype=torch.bool),
                    )
            gaussians, query, bind = cache[:3]
        targets = (
            info["targets"] if not args.preview else [info["targets"][len(info["targets"]) // 2]]
        )
        for name in targets:
            original = motion_from_fit(fits["scenes"][scene][name])
            original["transform_mat_neutral_pose"] = bind
            annotation = info["annotations"][name]
            w2c = torch.tensor(annotation["body_to_camera"], device="cuda")
            K = torch.tensor(annotation["K"], device="cuda")
            target = image_tensor(BASE / "protocol-test" / scene / "rgb" / name)[0]
            correction = torch.nn.Parameter(torch.zeros(6, device="cuda"))
            local = torch.nn.Parameter(torch.zeros(21, 3, device="cuda"))

            def forward(articulated):
                motion = dict(original)
                motion["root_pose"] = original["root_pose"] + correction[:3].tanh()[
                    None, None
                ] * math.radians(20)
                motion["trans"] = original["trans"] + correction[3:].tanh()[None, None] * 0.15
                if articulated:
                    motion["body_pose"] = original["body_pose"] + local.tanh()[
                        None, None
                    ] * math.radians(25)
                if args.method == "lhmpp":
                    rgb, alpha = render_pp(model, cache, motion, w2c, K)
                    return dict(comp_rgb=rgb[None, None], comp_mask=alpha[None, None, None])
                return model.renderer.forward_animate_gs(
                    gaussians,
                    query,
                    motion,
                    torch.linalg.inv(w2c)[None, None],
                    K[None, None],
                    512,
                    512,
                    torch.ones(1, 1, 3, device="cuda"),
                )

            with torch.no_grad():
                baseline = forward(False)
                initial_mse = float((baseline["comp_rgb"][0] - target).square().mean())
                for kind, key in (("rgb", "comp_rgb"), ("alpha", "comp_mask")):
                    folder = args.output / "repeat" / scene / kind
                    folder.mkdir(parents=True, exist_ok=True)
                    save_tensor(baseline[key][0, 0], folder / name)
            record = dict(scene=scene, frame=name, initial_mse=initial_mse, stages={})
            for stage, articulated in (("root", False), ("articulated", True)):
                variables = [correction, local] if articulated else [correction]
                optimizer = torch.optim.Adam(variables, lr=0.035)
                best = None
                curve = []
                for step in range(args.steps + 1):
                    optimizer.zero_grad()
                    prediction = forward(articulated)
                    loss = (prediction["comp_rgb"][0] - target).square().mean()
                    value = float(loss.detach())
                    curve.append(value)
                    if best is None or value < best[0]:
                        best = (value, correction.detach().clone(), local.detach().clone())
                    if step == args.steps:
                        break
                    loss.backward()
                    if correction.grad is None or not torch.isfinite(correction.grad).all():
                        raise RuntimeError(f"Missing/nonfinite pose gradient at {scene}/{name}")
                    if step == 0:
                        print(
                            f"{scene}/{name} {stage} gradient={float(correction.grad.norm()):.6g}",
                            flush=True,
                        )
                    torch.nn.utils.clip_grad_norm_(variables, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                with torch.no_grad():
                    correction.copy_(best[1])
                    local.copy_(best[2])
                    prediction = forward(articulated)
                folder = args.output / stage / scene
                for kind in ("rgb", "alpha"):
                    (folder / kind).mkdir(parents=True, exist_ok=True)
                save_tensor(prediction["comp_rgb"][0, 0], folder / "rgb" / name)
                save_tensor(prediction["comp_mask"][0, 0], folder / "alpha" / name)
                record["stages"][stage] = dict(
                    mse=best[0],
                    mse_curve=curve,
                    psnr_gain=float(10 * np.log10(initial_mse / best[0])),
                    root_axisangle_delta=(correction[:3].tanh() * math.radians(20)).tolist(),
                    translation_delta=(correction[3:].tanh() * 0.15).tolist(),
                    body_axisangle_delta=(local.tanh() * math.radians(25)).tolist(),
                )
                print(
                    f"{scene}/{name} {stage} PSNR gain {record['stages'][stage]['psnr_gain']:.3f}",
                    flush=True,
                )
            records.append(record)
            (args.output / "audit.json").write_text(
                json.dumps(
                    dict(
                        diagnostic_only=True,
                        fit_uses_test_rgb=True,
                        host=socket.gethostname(),
                        job=os.environ["SLURM_JOB_ID"],
                        steps=args.steps,
                        method=args.method,
                        records=records,
                    ),
                    indent=2,
                )
            )
        if args.preview:
            break
    if not args.preview:
        original_folder = (
            BASE / "lhm-500m-test"
            if args.method == "lhm"
            else WORK / "baselines/lhmpp-20260928/lhmpp-700m-test"
        )
        source = json.loads((original_folder / "method.json").read_text())
        for mode in ("repeat", "root", "articulated"):
            meta = dict(
                source,
                method=f"{args.method} diagnostic oracle {mode} pose refinement",
                fit_uses_test_rgb=mode != "repeat",
            )
            (args.output / mode / "method.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
