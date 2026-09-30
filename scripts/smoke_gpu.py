"""CUDA smoke at small and intended model sizes; not a pretrained benchmark.

Uses actual identity features and synthetic driving tokens. No learned avatar
quality or real SMPL teacher is tested by this script.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

from luna_open.avatar import Gaussians
from luna_open.data.neuman import NeuManDataset
from luna_open.features import DinoFeatures, SapiensFeatures
from luna_open.model import IdentityEncoder, ModelConfig, NeuralAnimator
from luna_open.rendering import render


def save_image(tensor, path):
    array = (tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(array).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2026)
    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    device = torch.device("cuda")
    # Real frames exercise image/mask/camera/body-annotation loading, not SMPL forward.
    dataset = NeuManDataset(args.data_root, args.manifest, split="test", size=128)
    sample = dataset[0]
    save_image(sample["rgb"], args.output / "neuman-crop.png")
    assert sample["reference_images"].shape == (4, 3, 128, 128)
    assert torch.isfinite(sample["body_to_camera"]).all()

    n = 128
    anchors = torch.randn(n, 3, device=device) * torch.tensor([0.25, 0.5, 0.1], device=device)
    config = ModelConfig(
        width=64,
        depth=2,
        heads=4,
        decoder_width=64,
        body_dim=32,
        face_dim=24,
        motion_dim=32,
        initial_scale=0.04,
    )
    identity = IdentityEncoder(anchors, torch.arange(n, device=device) % 24, config).to(device)
    animator = NeuralAnimator(n, config, torch.tensor([0.0, 0.0, 3.0]), torch.ones(3)).to(device)
    K = torch.tensor([[[150.0, 0.0, 64.0], [0.0, 150.0, 64.0], [0.0, 0.0, 1.0]]], device=device)
    body = torch.randn(1, 4, 16, 32, device=device)
    face = torch.randn(1, 4, 4, 24, device=device)
    motion = torch.randn(1, 16, 32, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        canonical = identity(body, face)
        posed = animator(canonical, motion)
        torch.testing.assert_close(
            posed.gaussians.means,
            anchors[None] + torch.tensor([0.0, 0.0, 3.0], device=device),
            atol=1e-6,
            rtol=1e-6,
        )
        result = render(posed.gaussians, K, (128, 128))
        loss = result["rgb"].mean() + result["alpha"].mean()
    loss.backward()
    for module in (identity, animator):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    save_image(result["rgb"][0], args.output / "synthetic-model.png")

    # Optimize perturbed colors against a fixed rendered target to verify actual
    # rasterizer gradients and an optimizer update, without an expensive training run.
    target_gs = posed.gaussians.detach()
    target_gs.colors = torch.rand_like(target_gs.colors)
    with torch.no_grad():
        target = render(target_gs, K, (128, 128))["rgb"]
    colors = torch.nn.Parameter(torch.zeros_like(target_gs.colors))
    optimizer = torch.optim.Adam([colors], lr=0.15)
    losses = []
    for _ in range(30):
        prediction = Gaussians(
            target_gs.means,
            target_gs.quaternions,
            target_gs.scales,
            target_gs.opacities,
            colors.sigmoid(),
        )
        rendered = render(prediction, K, (128, 128))
        error = (rendered["rgb"] - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        error.backward()
        optimizer.step()
        losses.append(float(error.detach()))
    assert losses[-1] < 0.3 * losses[0], losses
    # Test actual downloaded identity backbones on four real reference images.
    # DINOv3 and the real SMPL teacher are covered by caching and CLI integration.
    from cache_features import face_image

    encoder_data = NeuManDataset(args.data_root, args.manifest, size=1024)
    encoder_results, reference_features, crop_sources = {}, {}, []
    for kind in ("body", "face"):
        if kind == "body":
            backbone = SapiensFeatures(
                args.assets / "sapiens_body/sapiens_1b_epoch_173_torchscript.pt2"
            )
            expected_shape = (1, 4096, 1536)
        else:
            backbone = DinoFeatures(args.assets / "dino_face", "face")
            expected_shape = (1, 4, 1024, 1024)
        backbone = backbone.cuda().eval()
        views = []
        for name in sample["reference_names"]:
            if kind == "body":
                pixels = encoder_data.load_frame(sample["scene"], name)["rgb"]
            else:
                pixels, source = face_image(encoder_data, sample["scene"], name)
                crop_sources.append(source)
            # Ordinary no_grad tensors can be saved for the trainable projection
            # backward below. inference_mode tensors cannot be saved that way.
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                actual = backbone(pixels[None].cuda())
            assert actual.shape == expected_shape, actual.shape
            assert torch.isfinite(actual).all()
            views.append(actual.float())
        reference_features[kind] = torch.stack(views, dim=1)
        encoder_results[kind] = dict(
            shape=list(reference_features[kind].shape),
            std=float(reference_features[kind].std()),
        )
        del backbone, actual, views
        torch.cuda.empty_cache()

    # Exercise the intended token counts and width before real teacher training.
    # Anchors and driving features stay synthetic and are labeled in the report.
    full_start = time.monotonic()
    full_config = ModelConfig()
    full_n = 8192
    full_anchors = torch.randn(full_n, 3, device=device) * torch.tensor(
        [0.25, 0.5, 0.1], device=device
    )
    full_identity = IdentityEncoder(
        full_anchors, torch.arange(full_n, device=device) % 24, full_config
    ).to(device)
    full_animator = NeuralAnimator(
        full_n, full_config, torch.tensor([0.0, 0.0, 3.0]), torch.ones(3)
    ).to(device)
    full_motion = torch.randn(1, 1024, full_config.motion_dim, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full_canonical = full_identity(reference_features["body"], reference_features["face"])
        full_posed = full_animator(full_canonical, full_motion)
        full_render = render(full_posed.gaussians, K, (128, 128))
        full_loss = full_render["rgb"].mean() + full_render["alpha"].mean()
    full_loss.backward()
    for module in (full_identity, full_animator):
        grads = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(gradient).all() for gradient in grads)
        assert sum(gradient.abs().sum() for gradient in grads) > 0
    torch.cuda.synchronize()
    full_model_result = dict(
        config=asdict(full_config),
        queries=full_n,
        motion_shape=list(full_motion.shape),
        driving_features="synthetic for this isolated network smoke",
        anchors="synthetic for this isolated network smoke",
        rendering_resolution=[128, 128],
        trainable_parameters=sum(
            parameter.numel()
            for module in (full_identity, full_animator)
            for parameter in module.parameters()
        ),
        elapsed_seconds=time.monotonic() - full_start,
        forward_backward="passed",
    )
    torch.cuda.synchronize()
    report = dict(
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        capability=torch.cuda.get_device_capability(),
        dataset_frames=len(dataset),
        real_frame=f"{sample['scene']}/{sample['frame']}",
        neural_forward_backward="passed (synthetic features)",
        rasterizer_optimization_initial=losses[0],
        rasterizer_optimization_final=losses[-1],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        elapsed_seconds=time.monotonic() - start,
        pretrained_identity_encoders=encoder_results,
        intended_model=full_model_result,
        reference_frames=sample["reference_names"],
        face_crop_sources=crop_sources,
        limitations="No DINOv3, licensed real-SMPL forward, or quality benchmark tested",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
