"""Run pinned LHM-500M on the common NeuMan reference/camera protocol.

This adapter calls, rather than copies, Alibaba's Apache-2.0 LHM implementation
at 4f88aaeb3629249fbbddb4d0784a06962d9e1338. Reconstruction and skinning retain
the released implementation. Its supplied gsplat renderer is used with explicit
512x512 dimensions to preserve off-center crop intrinsics. See docs/baselines.md.
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import time
import types
from pathlib import Path

# Upstream inference configuration also disables compilation.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def image_tensor(path):
    image = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(image).permute(2, 0, 1).float().div(255)[None, None].cuda()


def save_tensor(tensor, path):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Nonfinite render or reference: {path}")
    array = tensor.detach().float().clamp(0, 1).cpu().numpy()
    if array.ndim == 3:
        array = array.transpose(1, 2, 0)
        if array.shape[-1] == 1:
            array = array[..., 0]
    Image.fromarray(np.round(array * 255).astype(np.uint8)).save(path, format="PNG")


def blank_motion(betas=None):
    shapes = dict(
        root_pose=(3,),
        body_pose=(21, 3),
        jaw_pose=(3,),
        leye_pose=(3,),
        reye_pose=(3,),
        lhand_pose=(15, 3),
        rhand_pose=(15, 3),
        expr=(100,),
        trans=(3,),
    )
    params = {key: torch.zeros(1, 1, *shape, device="cuda") for key, shape in shapes.items()}
    params["betas"] = torch.zeros(1, 10, device="cuda") if betas is None else betas
    return params


def verify_projection(model, protocol, output):
    from gsplat import rasterization
    from LHM.models.rendering.gs_renderer import GaussianModel

    first_scene = next(iter(protocol["scenes"].values()))
    first = first_scene["annotations"][first_scene["targets"][0]]
    K = torch.tensor(first["K"], device="cuda", dtype=torch.float32)
    # Pick a point whose analytical projection lies inside the off-center crop.
    pixel = torch.tensor([193.0, 287.0, 1.0], device="cuda")
    camera_mean = torch.linalg.solve(K, pixel) * 2
    w2c = torch.eye(4, device="cuda")
    angle = torch.tensor(0.35, device="cuda")
    w2c[0, 0] = w2c[2, 2] = angle.cos()
    w2c[0, 2], w2c[2, 0] = angle.sin(), -angle.sin()
    w2c[:3, 3] = torch.tensor([0.1, -0.02, 1.0], device="cuda")
    mean = ((camera_mean - w2c[:3, 3]) @ w2c[:3, :3])[None]
    quat = torch.tensor([[1.0, 0, 0, 0]], device="cuda")
    scale = torch.full((1, 3), 0.01, device="cuda")
    color = torch.tensor([[0.8, 0.2, 0.1]], device="cuda")
    image, alpha, info = rasterization(
        means=mean,
        quats=quat,
        scales=scale,
        opacities=torch.ones(1, device="cuda"),
        colors=color,
        backgrounds=torch.ones(1, 3, device="cuda"),
        viewmats=w2c[None],
        Ks=K[None],
        width=512,
        height=512,
    )
    actual = info["means2d"].reshape(-1, 2)[0]
    error = float((actual - pixel[:2]).abs().max())
    if image.shape != (1, 512, 512, 3) or error > 1e-3:
        raise RuntimeError(f"Camera adapter verification failed: {image.shape}, {error}")
    gaussian = GaussianModel(
        mean, torch.ones(1, 1, device="cuda"), quat, scale, color[:, None], use_rgb=True
    )
    native = model.renderer.forward_single_batch(
        [gaussian], torch.linalg.inv(w2c)[None], K[None], 512, 512, torch.ones(1, 3, device="cuda")
    )
    native_error = float((native["comp_rgb"] - image).abs().max())
    if native_error > 1e-4:
        raise RuntimeError(
            f"Native LHM camera path differs from analytical gsplat camera: {native_error}"
        )
    (output / "projection-check.json").write_text(
        json.dumps(
            dict(
                K=K.tolist(),
                expected_pixel=pixel[:2].tolist(),
                actual_pixel=actual.tolist(),
                max_absolute_error_pixels=error,
                image_shape=list(image.shape),
                alpha_max=float(alpha.max()),
                native_camera_rgb_max_error=native_error,
                world_to_camera=w2c.tolist(),
            ),
            indent=2,
        )
        + "\n"
    )


def build_model(source, checkpoint, output):
    sys.path.insert(0, str(source))
    os.chdir(source)
    # BasicSR 1.4.2 references the former torchvision module name. Only alias
    # that module; RGB-to-grayscale behavior is supplied by installed torchvision.
    import torchvision.transforms.functional as tv_functional

    sys.modules.setdefault("torchvision.transforms.functional_tensor", tv_functional)
    from accelerate import Accelerator
    from LHM.models.encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
    from LHM.models.modeling_human_lrm import ModelHumanLRMSapdinoBodyHeadSD3_5
    from LHM.models.rendering.gsplat_renderer import GSPlatRenderer

    Accelerator()  # initializes accelerate's logging state
    cfg = json.loads((checkpoint / "config.json").read_text())
    # Face-ID is a training loss network; it is absent from the released state.
    cfg["use_face_id"] = False
    original_builder = Dinov2FusionWrapper._build_dinov2
    Dinov2FusionWrapper._build_dinov2 = staticmethod(
        lambda model_name, modulation_dim=None: original_builder(
            model_name, modulation_dim, pretrained=False
        )
    )
    try:
        print(
            "Constructing native LHM with separate Sapiens and face restoration assets", flush=True
        )
        model = ModelHumanLRMSapdinoBodyHeadSD3_5(**cfg)
    finally:
        Dinov2FusionWrapper._build_dinov2 = staticmethod(original_builder)
    weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    incompatible = model.load_state_dict(weights, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    # The frozen Sapiens network is loaded separately from its published asset.
    invalid_missing = [key for key in missing if not key.startswith("fine_encoder.")]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Incomplete reconstruction checkpoint: {invalid_missing=}, {unexpected=}"
        )
    if any(key.startswith("encoder.") for key in missing):
        raise RuntimeError("DINO reconstruction weights must all come from LHM checkpoint")
    del weights
    model.cuda()
    # Upstream overrides train() without returning self, so do not chain eval().
    model.eval()
    model.requires_grad_(False)
    print("Loaded all released reconstruction weights; model ready on CUDA", flush=True)
    # Invoke the upstream GSPlatRenderer methods on the already-created renderer
    # to avoid constructing a second SMPL-X/voxel model. No learned layer changes.
    model.renderer.get_gaussians_properties = types.MethodType(
        GSPlatRenderer.get_gaussians_properties, model.renderer
    )
    model.renderer.forward_single_view = types.MethodType(
        GSPlatRenderer.forward_single_view, model.renderer
    )
    report = dict(
        missing_separate_sapiens_keys=missing,
        unexpected_keys=unexpected,
        config=cfg,
        checkpoint_sha256=checksum(checkpoint / "model.safetensors"),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        renderer="upstream GSPlatRenderer.forward_single_view; gsplat 1.4.0",
        compatibility=[
            "BasicSR torchvision module alias",
            "skip initial DINO download; complete LHM DINO state loaded",
            "omit unused face-ID loss network",
            "disable torch.compile as upstream inference config",
        ],
    )
    (output / "model-load.json").write_text(json.dumps(report, indent=2) + "\n")
    return model, report


@torch.inference_mode()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Use a Slurm compute node")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--face-inputs", type=Path, required=True)
    parser.add_argument("--fits", type=Path)
    parser.add_argument("--canonical-camera", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", help="Limit to one scene for an integration preview")
    parser.add_argument("--canonical-only", action="store_true")
    args = parser.parse_args()
    for key in (
        "source",
        "checkpoint",
        "protocol",
        "face_inputs",
        "fits",
        "output",
        "canonical_camera",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.canonical_only and args.fits is None:
        raise ValueError("Converted SMPL-X poses required for posed evaluation")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("This Yonsei protocol requires the allocated single RTX 4090")
    args.output.mkdir(parents=True)
    protocol = json.loads((args.protocol / "protocol.json").read_text())
    fits = json.loads(args.fits.read_text()) if args.fits else None
    if fits and fits["manifest_sha256"] != protocol["manifest_sha256"]:
        raise ValueError("Fit/protocol manifest mismatch")
    if fits and not fits.get("all_passed", False):
        raise ValueError(
            "SMPL-X conversion must pass its geometry quality thresholds before scoring"
        )
    started = time.perf_counter()
    model, load_report = build_model(args.source, args.checkpoint, args.output)
    verify_projection(model, protocol, args.output)
    print("Analytical projection and native camera path checks passed", flush=True)
    for scene, info in protocol["scenes"].items():
        if args.scene and scene != args.scene:
            continue
        folder = args.output / scene
        for subdir in ("rgb", "alpha", "canonical", "references"):
            (folder / subdir).mkdir(parents=True)
        reference = info["references"][0]
        body = image_tensor(args.protocol / scene / "rgb" / reference)
        face = image_tensor(args.face_inputs / scene / "references" / f"face-{reference}")
        save_tensor(body[0, 0], folder / "references/body.png")
        save_tensor(face[0, 0], folder / "references/face.png")
        params = blank_motion()
        gaussians, query, bind = model.infer_single_view(
            body, face, None, None, None, None, None, params
        )
        params["transform_mat_neutral_pose"] = bind
        if fits:
            params["betas"] = torch.tensor(
                fits["scenes"][scene][reference]["betas"], device="cuda", dtype=torch.float32
            )[None]
        # Canonical target is the zero-pose SMPL-X body, with a front-facing
        # OpenCV camera (+Y down, +Z forward), metric units and white background.
        for view, position in (("front", [0, 0, 3.5]), ("back", [0, 0, -3.5])):
            w2c = torch.eye(4, device="cuda")
            w2c[:3, :3] = torch.diag(
                torch.tensor(
                    [1.0 if view == "front" else -1.0, -1.0, -1.0 if view == "front" else 1.0],
                    device="cuda",
                )
            )
            w2c[:3, 3] = -w2c[:3, :3] @ torch.tensor(position, device="cuda")
            K = torch.tensor([[768.0, 0, 256.0], [0, 768.0, 256.0], [0, 0, 1]], device="cuda")
            if args.canonical_camera and view == "front":
                camera = json.loads(args.canonical_camera.read_text())
                w2c = torch.tensor(camera["world_to_camera"], device="cuda")
                K = torch.tensor(camera["K"], device="cuda")
            pred = model.renderer.forward_animate_gs(
                gaussians,
                query,
                params,
                torch.linalg.inv(w2c)[None, None],
                K[None, None],
                512,
                512,
                torch.ones(1, 1, 3, device="cuda"),
            )
            save_tensor(pred["comp_rgb"][0, 0], folder / "canonical" / f"{view}.png")
        if not args.canonical_only:
            for name in info["targets"]:
                motion = blank_motion()
                fitted = fits["scenes"][scene][name]
                for key in ("root_pose", "body_pose", "lhand_pose", "rhand_pose", "trans"):
                    motion[key] = torch.tensor(fitted[key], device="cuda", dtype=torch.float32)[
                        None, None
                    ]
                motion["betas"] = torch.tensor(fitted["betas"], device="cuda", dtype=torch.float32)[
                    None
                ]
                motion["transform_mat_neutral_pose"] = bind
                annotation = info["annotations"][name]
                w2c = torch.tensor(annotation["body_to_camera"], device="cuda", dtype=torch.float32)
                K = torch.tensor(annotation["K"], device="cuda", dtype=torch.float32)
                pred = model.renderer.forward_animate_gs(
                    gaussians,
                    query,
                    motion,
                    torch.linalg.inv(w2c)[None, None],
                    K[None, None],
                    512,
                    512,
                    torch.ones(1, 1, 3, device="cuda"),
                )
                save_tensor(pred["comp_rgb"][0, 0], folder / "rgb" / name)
                save_tensor(pred["comp_mask"][0, 0], folder / "alpha" / name)
        print(f"Completed {scene}: reference {reference}", flush=True)
    report = dict(
        method="LHM-500M released checkpoint",
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=load_report["checkpoint_sha256"],
        reference_count=1,
        reference_frames={
            scene: info["references"][:1] for scene, info in protocol["scenes"].items()
        },
        manifest_sha256=protocol["manifest_sha256"],
        split=protocol["split"],
        upstream_commit="4f88aaeb3629249fbbddb4d0784a06962d9e1338",
        renderer=load_report["renderer"],
        fits=str(args.fits) if args.fits else None,
        fits_sha256=checksum(args.fits) if args.fits else None,
        canonical_camera=str(args.canonical_camera) if args.canonical_camera else None,
        canonical_camera_sha256=checksum(args.canonical_camera) if args.canonical_camera else None,
        canonical_only=args.canonical_only,
        scene_limit=args.scene,
        preprocessing="common 512 square white body crop; supplied NeuMan keypoint face crop resized 112; native Sapiens 1024 and DINO 448; native face restoration",
        mask_protocol="supplied NeuMan segmentations define reference/target crops and white input backgrounds; predicted alpha retained; no target-mask cleanup of predictions",
        pose_note="SMPL-X fits obtained from NeuMan SMPL meshes, no RGB-based baseline fine-tuning",
        skinning_note="pinned get_transform_mat_vertex computes query_skinning but multiplies stored skinning_weight; preserved",
        training_overlap="released pretraining data overlap with NeuMan is unknown; no local baseline training",
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(),
        elapsed_seconds=time.perf_counter() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.output / "method.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
