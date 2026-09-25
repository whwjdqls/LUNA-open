"""Check actual LPIPS weights, image metrics and image gradients on one real frame."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import lpips
import torch

from luna_open.data.neuman import NeuManDataset
from luna_open.losses import rendering_losses
from luna_open.metrics import image_metrics
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(2026)
    model = build_lpips(device="cpu")
    data = NeuManDataset(args.root, args.manifest, split="test", size=512)
    scene, name = data.items[0]
    frame = data.load_frame(scene, name)
    rgb, mask = frame["rgb"][None], frame["mask"][None]
    perfect = image_metrics(dict(rgb=rgb, alpha=mask), rgb, mask, model)
    assert abs(float(perfect["lpips"])) < 1e-7
    torch.testing.assert_close(perfect["ssim"], torch.ones(1))
    assert float(perfect["mask_iou"]) == 1.0
    perturbed = (rgb * (1 - 0.3 * mask)).requires_grad_(True)
    prediction = dict(rgb=perturbed, alpha=mask)
    changed = image_metrics(prediction, rgb, mask, model)
    assert float(changed["lpips"]) > 0
    losses = rendering_losses(prediction, rgb, mask, model)
    # Check the perceptual gradient itself, not a gradient supplied only by L1.
    losses["lpips"].backward()
    assert torch.isfinite(perturbed.grad).all() and perturbed.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    backbone = Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    learned = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    report = dict(
        scene=scene,
        frame=name,
        manifest_sha256=file_sha256(args.manifest),
        device="cpu",
        versions={key: version(key) for key in ("torch", "torchvision", "lpips")},
        alexnet_source="https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        alexnet_sha256=file_sha256(backbone),
        lpips_linear_weights_sha256=file_sha256(learned),
        identical={key: float(value[0]) for key, value in perfect.items()},
        darkened_foreground={key: float(value[0]) for key, value in changed.items()},
        perceptual_gradient_l1=float(perturbed.grad.abs().sum()),
        limitations="Metric/loss integration with a synthetic perturbation; no model quality score",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
