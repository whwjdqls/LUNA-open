"""Evaluate released LHM++-700M with four fixed NeuMan training references.

Calls upstream code at 906b5d9fb967ab42efb92f6fa55bf22cac86b653, retaining its
point/image transformer, diffused SMPL-X skinning and DPT neural renderer.
Geometry uses meters, OpenCV world-to-camera extrinsics and full crop K.
The DPT needs dimensions divisible by seven: render 518 square with unchanged
K, then discard the six extra bottom/right pixels to preserve the 512 crop.
No target RGB or mask is supplied to reconstruction, skinning or neural rendering.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import torch
from evaluate_lhm_neuman import blank_motion, checksum, image_tensor, save_tensor
from safetensors.torch import load_file


def build_model(source, checkpoint, output):
    source_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != "906b5d9fb967ab42efb92f6fa55bf22cac86b653":
        raise ValueError(f"Unexpected upstream source: {source_commit}")
    sys.path.insert(0, str(source))
    os.chdir(source)
    from accelerate import Accelerator
    from core.models.encoders.dinov2_wrapper import Dinov2Wrapper
    from core.models.modeling_humana4o_lrm import ModelHumanA4OLRM

    Accelerator()
    torch._dynamo.config.disable = True
    cfg = json.loads((checkpoint / "config.json").read_text())
    # ArcFace is a training loss module, not an inference input.
    cfg["use_face_id"] = False
    builder = Dinov2Wrapper._build_dinov2
    Dinov2Wrapper._build_dinov2 = staticmethod(
        lambda model_name, modulation_dim=None: builder(
            model_name, modulation_dim, pretrained=False
        )
    )
    try:
        print("Constructing native LHM++-700M", flush=True)
        model = ModelHumanA4OLRM(**cfg)
    finally:
        Dinov2Wrapper._build_dinov2 = staticmethod(builder)
    # This DPT-only subclass inherits the generic renderer constructor, which
    # allocates attention blocks that its predict_rgbs() never calls. They are
    # absent from the released checkpoint. Remove only those dormant modules;
    # the patch tokenizer and complete DPT network must load strictly.
    if cfg["neural_renderer"]["type"] != "patch_4dptonly":
        raise ValueError("Adapter audited for the released DPT-only checkpoint")
    unused_keys = [
        "neural_renderer.transformer_block." + k
        for k in model.neural_renderer.transformer_block.state_dict()
    ]
    model.neural_renderer.transformer_block = torch.nn.ModuleList()
    weights = load_file(str(checkpoint / "model.safetensors"), device="cpu")
    incompatible = model.load_state_dict(weights, strict=False)
    # Never permit an uninitialized reconstruction or neural-renderer layer.
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Incomplete LHM++ checkpoint: {incompatible}")
    tensor_count = len(weights)
    del weights
    model.cuda()
    model.eval()
    model.requires_grad_(False)
    # Native forward_gs normally sets this transient attribute. The analytical
    # camera check runs first, before reconstruction creates that runtime state.
    model.renderer.device = torch.device("cuda")
    weight_path = checkpoint / "model.safetensors"
    hash_receipt = checkpoint / "model-verified-sha256.json"
    stat = weight_path.stat()
    fingerprint = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    cached_hash = json.loads(hash_receipt.read_text()) if hash_receipt.exists() else {}
    if cached_hash.get("fingerprint") == fingerprint:
        weight_sha = cached_hash["sha256"]
    else:
        weight_sha = checksum(weight_path)
    report = dict(
        missing_keys=[],
        unexpected_keys=[],
        loaded_tensors=tensor_count,
        source_commit=source_commit,
        adapter_sha256=checksum(Path(__file__).resolve()),
        removed_unused_constructor_keys=unused_keys,
        config=cfg,
        checkpoint_sha256=weight_sha,
        checkpoint_fingerprint=fingerprint,
        checkpoint_hash_reused=(cached_hash.get("fingerprint") == fingerprint),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        compatibility=[
            "skip initial DINO download; all DINO weights in released state",
            "omit unused ArcFace training loss network",
            "remove inherited attention blocks unused by DPT-only forward and absent from checkpoint",
            "torch.compile disabled as in upstream inference",
        ],
    )
    (output / "model-load.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Loaded {tensor_count} checkpoint tensors with no missing/unexpected keys", flush=True)
    return model, report


def verify_projection(model, protocol, output):
    from core.structures.camera import Camera
    from core.structures.gaussian_model import GaussianModel
    from gsplat import rasterization

    scene = next(iter(protocol["scenes"].values()))
    K = torch.tensor(scene["annotations"][scene["targets"][0]]["K"], device="cuda")
    pixel = torch.tensor([193.0, 287.0, 1.0], device="cuda")
    w2c = torch.eye(4, device="cuda")
    angle = torch.tensor(0.35, device="cuda")
    w2c[0, 0] = w2c[2, 2] = angle.cos()
    w2c[0, 2], w2c[2, 0] = angle.sin(), -angle.sin()
    w2c[:3, 3] = torch.tensor([0.1, -0.02, 1.0], device="cuda")
    mean = ((torch.linalg.solve(K, pixel) * 2 - w2c[:3, 3]) @ w2c[:3, :3])[None]
    quat = torch.tensor([[1.0, 0, 0, 0]], device="cuda")
    scale = torch.full((1, 3), 0.01, device="cuda")
    color = torch.tensor([[0.8, 0.2, 0.1]], device="cuda")
    direct, alpha, info = rasterization(
        means=mean,
        quats=quat,
        scales=scale,
        opacities=torch.ones(1, device="cuda"),
        colors=color,
        backgrounds=torch.ones(1, 3, device="cuda"),
        viewmats=w2c[None],
        Ks=K[None],
        width=518,
        height=518,
    )
    camera = Camera.from_c2w(torch.linalg.inv(w2c), K, 518, 518)
    gaussian = GaussianModel(
        mean, torch.ones(1, 1, device="cuda"), quat, scale, color[:, None], use_rgb=True
    )
    native = model.renderer.forward_single_view(
        gaussian,
        camera,
        torch.ones(3, device="cuda"),
        features=torch.zeros(1, 128, device="cuda"),
        patch_size=1,
    )
    position_error = float((info["means2d"].reshape(-1, 2)[0] - pixel[:2]).abs().max())
    rgb_error = float((native["comp_rgb"] - direct[0]).abs().max())
    alpha_error = float((native["comp_mask"] - alpha[0]).abs().max())
    if position_error > 1e-3 or max(rgb_error, alpha_error) > 1e-4:
        raise RuntimeError(f"Camera check failed: {position_error}, {rgb_error}, {alpha_error}")
    report = dict(
        max_absolute_error_pixels=position_error,
        native_camera_rgb_max_error=rgb_error,
        native_camera_alpha_max_error=alpha_error,
        expected_pixel=pixel[:2].tolist(),
        K=K.tolist(),
        render_size=[518, 518],
        retained_crop=[0, 0, 512, 512],
    )
    (output / "projection-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Native feature renderer camera check passed", flush=True)


def render(model, cache, motion, w2c, K):
    gaussians, query, bind, latent, images, motion_embed, positions = cache
    motion["transform_mat_neutral_pose"] = bind
    rendered = model.renderer.forward_animate_gs(
        gaussians,
        query,
        motion,
        torch.linalg.inv(w2c)[None, None],
        K[None, None],
        518,
        518,
        torch.ones(1, 1, 3, device="cuda"),
        patch_size=model.neural_renderer_patch_size,
        features=latent,
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb, mask = model.neural_renderer(
            images,
            rendered["comp_features"],
            motion_embed,
            518,
            518,
            pos_emb_list=positions,
        )
    if mask.shape == (1, 1, 518, 518, 1):
        mask = mask.squeeze(-1)
    if rgb.shape != (1, 1, 518, 518, 3) or mask.shape != (1, 1, 518, 518):
        raise RuntimeError(f"Unexpected native DPT output shapes: {rgb.shape}, {mask.shape}")
    # Native DPT predicts the final RGB canvas and a separate mask. Do not
    # alpha-composite again (upstream also exports DPT RGB directly).
    return rgb[0, 0, :512, :512].permute(2, 0, 1), mask[0, 0, :512, :512]


@torch.inference_mode()
def main():
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Slurm compute node required")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "protocol", "fits", "canonical-camera", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scene", help="Single scene integration preview")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise RuntimeError("Use the selected single RTX 4090 allocation")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    args.output.mkdir(parents=True)
    protocol = json.loads((args.protocol / "protocol.json").read_text())
    fits = json.loads(args.fits.read_text())
    if not fits["all_passed"] or fits["manifest_sha256"] != protocol["manifest_sha256"]:
        raise ValueError("Require matching, audited SMPL-X conversions")
    started = time.perf_counter()
    model, load_report = build_model(args.source, args.checkpoint, args.output)
    verify_projection(model, protocol, args.output)
    scene_records = {}
    for scene, info in protocol["scenes"].items():
        if args.scene and scene != args.scene:
            continue
        folder = args.output / scene
        for kind in ("rgb", "alpha", "canonical", "references"):
            (folder / kind).mkdir(parents=True)
        images = torch.cat(
            [image_tensor(args.protocol / scene / "rgb" / name) for name in info["references"]],
            dim=1,
        )
        for name in info["references"]:
            shutil.copy2(args.protocol / scene / "rgb" / name, folder / "references" / name)
        # Match LHM reconstruction's neutral mean-shape queries. Converted
        # target shape is supplied at animation, not to the reference encoder.
        initial = blank_motion()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cache = model.infer_single_view(
                images,
                None,
                None,
                torch.eye(4, device="cuda")[None, None],
                torch.tensor([[768.0, 0, 256], [0, 768, 256], [0, 0, 1]], device="cuda")[
                    None, None
                ],
                torch.ones(1, 1, 3, device="cuda"),
                initial,
                ref_imgs_bool=torch.ones(1, 4, device="cuda", dtype=torch.bool),
            )
        del images
        gaussian_count = int(cache[1]["neutral_coords"].shape[1])
        print(f"Reconstructed {scene}: 4 references, {gaussian_count} points", flush=True)
        for view in ("front", "back"):
            canonical = blank_motion()
            canonical["betas"] = torch.tensor(
                fits["scenes"][scene][info["references"][0]]["betas"],
                device="cuda",
                dtype=torch.float32,
            )[None]
            camera = json.loads(args.canonical_camera.read_text())
            w2c = torch.tensor(camera["world_to_camera"], device="cuda")
            K = torch.tensor(camera["K"], device="cuda")
            if view == "back":
                w2c = torch.eye(4, device="cuda")
                w2c[:3, :3] = torch.diag(torch.tensor([-1.0, -1.0, 1.0], device="cuda"))
                w2c[:3, 3] = torch.tensor([0.0, 0.0, 3.5], device="cuda")
                K = torch.tensor([[768.0, 0, 256], [0, 768, 256], [0, 0, 1]], device="cuda")
            rgb, mask = render(model, cache, canonical, w2c, K)
            save_tensor(rgb, folder / "canonical" / f"{view}.png")
        for name in info["targets"]:
            fitted = fits["scenes"][scene][name]
            motion = blank_motion()
            for key in ("root_pose", "body_pose", "lhand_pose", "rhand_pose", "trans"):
                motion[key] = torch.tensor(fitted[key], device="cuda", dtype=torch.float32)[
                    None, None
                ]
            motion["betas"] = torch.tensor(fitted["betas"], device="cuda", dtype=torch.float32)[
                None
            ]
            annotation = info["annotations"][name]
            w2c = torch.tensor(annotation["body_to_camera"], device="cuda", dtype=torch.float32)
            K = torch.tensor(annotation["K"], device="cuda", dtype=torch.float32)
            rgb, mask = render(model, cache, motion, w2c, K)
            save_tensor(rgb, folder / "rgb" / name)
            save_tensor(mask, folder / "alpha" / name)
            print(f"Saved {scene}/{name}", flush=True)
        scene_records[scene] = dict(gaussian_count=gaussian_count, references=info["references"])
        del cache, rgb, mask
        torch.cuda.empty_cache()
    report = dict(
        method="LHM++-700M released checkpoint + native DPT neural renderer",
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=load_report["checkpoint_sha256"],
        checkpoint_revision="fc9f73664b9bfcc457e96210ddc06fe7caf0d559",
        upstream_commit="906b5d9fb967ab42efb92f6fa55bf22cac86b653",
        reference_count=4,
        reference_frames={s: i["references"] for s, i in protocol["scenes"].items()},
        manifest_sha256=protocol["manifest_sha256"],
        split=protocol["split"],
        renderer="native GSPlatBackFeatRenderer + PatchDPT4DecoderOnly",
        fits=str(args.fits),
        fits_sha256=checksum(args.fits),
        canonical_camera_sha256=checksum(args.canonical_camera),
        scene_limit=args.scene,
        scene_records=scene_records,
        preprocessing="common 512x512 white body crops; native DINO wrapper resizes to 504x504; four exact references",
        render_geometry="518x518 with full unchanged crop K; retain top-left 512x512; DPT patch size seven",
        rgb_protocol="native DPT final RGB with learned white background; no additional alpha multiplication",
        mask_protocol="NeuMan supplied masks for common input/GT cropping only; DPT predicted mask for IoU; no GT cleanup",
        posing="audited SMPL-to-SMPL-X fits; native diffused voxel skinning; no target RGB fitting",
        training_overlap="no local fine-tuning; released pretraining overlap with NeuMan unknown",
        dtype="native reconstruction and DPT under bfloat16 autocast; rasterizer float32",
        host=socket.gethostname(),
        job_id=os.environ["SLURM_JOB_ID"],
        gpu=torch.cuda.get_device_name(),
        elapsed_seconds=time.perf_counter() - started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
    )
    (args.output / "method.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
