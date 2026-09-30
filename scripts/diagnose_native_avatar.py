"""Native baseline ablations and canonical orbit with explicit SMPL/SMPL-X origin alignment.

Reuses the previously audited adapters. Outputs are separate diagnostic variants.
Canonical pelvis translation is measured from body templates, not target RGB.
Reference recropping changes reconstruction input only; target cameras stay fixed.
"""

import argparse
import copy
import json
import math
import os
import socket
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
from evaluate_lhm_neuman import blank_motion, image_tensor, save_tensor
from PIL import Image

WORK = Path("/scratch2/whwjdqls99/LUNA-open")
BASE = WORK / "baselines/lhm-20260928"
PLUS = WORK / "baselines/lhmpp-20260928"


def motion_from_fit(fit):
    motion = blank_motion()
    for key in ("root_pose", "body_pose", "lhand_pose", "rhand_pose", "trans"):
        motion[key] = torch.tensor(fit[key], device="cuda")[None, None]
    motion["betas"] = torch.tensor(fit["betas"], device="cuda")[None]
    return motion


def reference(scene, name, recrop, folder):
    image = Image.open(BASE / "protocol-test" / scene / "rgb" / name).convert("RGB")
    if recrop:
        mask = np.asarray(Image.open(BASE / "protocol-test" / scene / "mask" / name)) > 127
        ys, xs = np.nonzero(mask)
        # Native inference uses a tall 5:3 canvas. A 5% margin is our declared
        # diagnostic choice; no pixels from other reference/target frames enter.
        h = math.ceil(max(np.ptp(ys) + 1, (np.ptp(xs) + 1) * 5 / 3) * 1.05)
        w = math.ceil(h * 3 / 5)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        x, y = round(cx - w / 2), round(cy - h / 2)
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(image, (-x, -y))
        image = canvas.resize((504, 840), Image.Resampling.LANCZOS)
    image.save(folder / name)
    return (
        torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().cuda()[None, None] / 255
    )


def orbit(camera, angle):
    # Rotate the shared front camera about the canonical body's vertical axis.
    front = torch.tensor(camera["world_to_camera"], device="cuda")
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    yaw = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], device="cuda")
    center = torch.tensor([-front[0, 3], front[1, 3], 0], device="cuda")
    result = front.clone()
    result[:3, :3] = front[:3, :3] @ yaw.T
    result[:3, 3] = front[:3, 3] + front[:3, :3] @ center - result[:3, :3] @ center
    return result


@torch.inference_mode()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["lhm", "lhmpp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert torch.cuda.device_count() == 1 and "4090" in torch.cuda.get_device_name()
    if args.method == "lhm":
        from evaluate_lhm_neuman import build_model

        model, _ = build_model(BASE / "source", WORK / "assets/lhm_500m", args.output)
        original = BASE / "lhm-500m-test"
    else:
        from evaluate_lhmpp_neuman import build_model

        model, _ = build_model(PLUS / "source", WORK / "assets/lhmpp_700m", args.output)
        original = PLUS / "lhmpp-700m-test"
    protocol = json.loads((BASE / "protocol-test/protocol.json").read_text())
    fits = json.loads((BASE / "fit-test/fits.json").read_text())
    camera = json.loads((BASE / "ours-identity-14750-test/canonical-camera.json").read_text())
    origins = json.loads((WORK / "diagnostics/body-frames-20260928.json").read_text())
    evidence = dict(
        host=socket.gethostname(), job=os.environ["SLURM_JOB_ID"], method=args.method, scenes={}
    )
    # Exercise the actual native rest-joint and FK path with the same converted
    # parameters. This extends the previous camera-only and external mesh audit.
    import smplx

    standard = (
        smplx.SMPLX(
            fits["files"]["smplx"]["path"],
            num_betas=10,
            use_pca=False,
            flat_hand_mean=True,
            num_expression_coeffs=10,
        )
        .cuda()
        .eval()
    )
    standard.posedirs.zero_()
    native = model.renderer.smplx_model
    native_fk = []
    for scene, info in protocol["scenes"].items():
        for name in [info["references"][0]] + info["targets"]:
            motion = motion_from_fit(fits["scenes"][scene][name])
            single = {k: v[:, 0] if k != "betas" else v for k, v in motion.items()}
            rest = native.get_zero_pose_human(single["betas"], torch.device("cuda"), None, None)
            _, joints = native.get_transform_mat_joint(None, rest, single)
            actual = joints + single["trans"][:, None]
            expected = standard(
                betas=single["betas"],
                global_orient=single["root_pose"],
                body_pose=single["body_pose"].flatten(1),
                left_hand_pose=single["lhand_pose"].flatten(1),
                right_hand_pose=single["rhand_pose"].flatten(1),
                transl=single["trans"],
            ).joints[:, :55]
            residual = (actual - expected).norm(dim=-1)
            native_fk.append(
                dict(
                    scene=scene,
                    frame=name,
                    mean_mm=float(residual.mean() * 1000),
                    max_mm=float(residual.max() * 1000),
                )
            )
    evidence["native_fk_vs_fit_body"] = native_fk
    (args.output / "audit.json").write_text(json.dumps(evidence, indent=2))
    print(f"Native FK max error {max(r['max_mm'] for r in native_fk):.6f} mm", flush=True)
    del standard

    def render(cache, motion, w2c, K):
        if args.method == "lhmpp":
            from evaluate_lhmpp_neuman import render as render_pp

            return render_pp(model, cache, motion, w2c, K)
        gaussians, query, bind = cache
        motion["transform_mat_neutral_pose"] = bind
        out = model.renderer.forward_animate_gs(
            gaussians,
            query,
            motion,
            torch.linalg.inv(w2c)[None, None],
            K[None, None],
            512,
            512,
            torch.ones(1, 1, 3, device="cuda"),
        )
        return out["comp_rgb"][0, 0], out["comp_mask"][0, 0]

    for scene, info in protocol["scenes"].items():
        first = info["references"][0]
        delta = origins["scenes"][scene]["canonical_root_translation"]
        scene_evidence = dict(canonical_translation=delta, raw_repeat_max_u8=0)
        evidence["scenes"][scene] = scene_evidence
        for recrop in (False, True):
            label = "recrop" if recrop else "original"
            refs_folder = args.output / label / scene / "references"
            refs_folder.mkdir(parents=True)
            names = info["references"][: 1 if args.method == "lhm" else 4]
            refs = torch.cat([reference(scene, name, recrop, refs_folder) for name in names], dim=1)
            if args.method == "lhm":
                face = image_tensor(
                    BASE / "ours-identity-14750-test" / scene / "references" / f"face-{first}"
                )
                cache = model.infer_single_view(
                    refs, face, None, None, None, None, None, blank_motion()
                )
            else:
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
            if not recrop:
                folder = args.output / "canonical" / scene
                folder.mkdir(parents=True)
                for angle in range(0, 360, 30):
                    motion = blank_motion()
                    motion["betas"] = torch.tensor(
                        fits["scenes"][scene][first]["betas"], device="cuda"
                    )[None]
                    motion["trans"] = torch.tensor(delta, device="cuda")[None, None]
                    rgb, alpha = render(
                        cache,
                        motion,
                        orbit(camera, angle),
                        torch.tensor(camera["K"], device="cuda"),
                    )
                    save_tensor(rgb, folder / f"{angle:03}.png")
                # Original zero-translation display for a direct before/after.
                motion["trans"].zero_()
                rgb, _ = render(
                    cache,
                    motion,
                    torch.tensor(camera["world_to_camera"], device="cuda"),
                    torch.tensor(camera["K"], device="cuda"),
                )
                save_tensor(rgb, folder / "unaligned.png")
            modes = ("recrop",) if recrop else ("repeat", "zero-shape", "fixed-shape")
            for mode in modes:
                folder = args.output / mode / scene
                for kind in ("rgb", "alpha"):
                    (folder / kind).mkdir(parents=True, exist_ok=True)
                for name in info["targets"]:
                    motion = motion_from_fit(fits["scenes"][scene][name])
                    if mode == "zero-shape":
                        motion["betas"].zero_()
                    elif mode == "fixed-shape":
                        motion["betas"] = torch.tensor(
                            fits["scenes"][scene][first]["betas"], device="cuda"
                        )[None]
                    ann = info["annotations"][name]
                    rgb, alpha = render(
                        cache,
                        motion,
                        torch.tensor(ann["body_to_camera"], device="cuda"),
                        torch.tensor(ann["K"], device="cuda"),
                    )
                    save_tensor(rgb, folder / "rgb" / name)
                    save_tensor(alpha, folder / "alpha" / name)
                    if mode == "repeat":
                        old = np.asarray(Image.open(original / scene / "rgb" / name)).astype(int)
                        new = np.asarray(Image.open(folder / "rgb" / name)).astype(int)
                        scene_evidence["raw_repeat_max_u8"] = max(
                            scene_evidence["raw_repeat_max_u8"], int(np.max(np.abs(old - new)))
                        )
                print(f"{args.method} {scene} {mode} done", flush=True)
            del cache, refs
            torch.cuda.empty_cache()
        (args.output / "audit.json").write_text(json.dumps(evidence, indent=2))
    original_method = json.loads((original / "method.json").read_text())
    for mode in ("repeat", "zero-shape", "fixed-shape", "recrop"):
        meta = copy.deepcopy(original_method)
        meta.update(
            method=f"{args.method} diagnostic {mode}",
            diagnostic=True,
            reference_preprocessing="GT-reference-mask bbox + 5% margin; 840x504 tall canvas"
            if mode == "recrop"
            else "unchanged",
            output_root=str(args.output),
            canonical_display="pelvis translation only, does not alter target renders",
        )
        (args.output / mode / "method.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    main()
