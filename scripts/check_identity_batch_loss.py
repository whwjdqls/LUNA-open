"""Check per-image loss weighting independently of BF16 encoder batching."""

import argparse
import json
import os
import socket
from pathlib import Path

import torch

from luna_open.avatar import CanonicalAvatar, Gaussians
from luna_open.identity_batching import batched_identity_losses, select_avatar
from luna_open.identity_training import identity_losses
from luna_open.perceptual import build_lpips
from luna_open.provenance import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-tf32", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or "login" in socket.gethostname():
        raise RuntimeError("Use a compute-node GPU allocation")
    torch.manual_seed(20260928)
    if args.disable_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda"
    batch, points, size = 8, 64, 128
    rgb = torch.rand(batch, 3, size, size, device=device, requires_grad=True)
    alpha = torch.rand(batch, 1, size, size, device=device, requires_grad=True)
    means = torch.randn(batch, points, 3, device=device, requires_grad=True)
    scales = torch.rand(batch, points, 3, device=device).add(0.01).requires_grad_(True)
    inputs = [rgb, alpha, means, scales]
    gaussian = Gaussians(
        means,
        torch.zeros(batch, points, 4, device=device),
        scales,
        torch.ones(batch, points, device=device),
        torch.ones_like(means),
    )
    canonical = CanonicalAvatar(gaussian, means)
    anchors = torch.randn_like(means)
    # Deliberately unequal foreground areas, including empty and full masks.
    thresholds = torch.linspace(0, 1, batch, device=device)[:, None, None, None]
    mask = (torch.rand(batch, 1, size, size, device=device) < thresholds).float()
    target = dict(rgb=torch.rand_like(rgb), mask=mask)
    prediction = dict(rgb=rgb, alpha=alpha)
    perceptual = build_lpips("alex", device=device)
    options = dict(
        rgb_reduction="balanced_foreground_background", anisotropy_weight=0.01, anchor_weight=10.0
    )
    serial = {}
    for i in range(batch):
        terms = identity_losses(
            {key: value[i : i + 1] for key, value in prediction.items()},
            {key: value[i : i + 1] for key, value in target.items()},
            select_avatar(canonical, i, i + 1),
            anchors[i : i + 1],
            perceptual,
            options,
        )
        for key, value in terms.items():
            serial[key] = serial.get(key, 0) + value / batch
    serial_gradients = torch.autograd.grad(sum(serial.values()), inputs)
    batched = batched_identity_losses(prediction, target, gaussian, anchors, perceptual, options)
    batched_gradients = torch.autograd.grad(sum(batched.values()), inputs)
    errors = {}
    for key in serial:
        try:
            torch.testing.assert_close(serial[key], batched[key], rtol=1e-5, atol=1e-6)
        except AssertionError as error:
            errors[f"loss_{key}"] = str(error)
    gradients = {}
    for name, reference, actual in zip(
        ("rgb", "alpha", "means", "scales"), serial_gradients, batched_gradients, strict=True
    ):
        try:
            torch.testing.assert_close(reference, actual, rtol=1e-3, atol=1e-7)
        except AssertionError as error:
            errors[f"gradient_{name}"] = str(error)
        gradients[name] = dict(
            max_abs=float((reference - actual).abs().max()),
            relative_l2=float((reference - actual).norm() / reference.norm().clamp_min(1e-20)),
        )
    report = dict(
        passed=not errors,
        errors=errors,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        job_id=os.environ["SLURM_JOB_ID"],
        host=socket.gethostname(),
        script_sha256=file_sha256(__file__),
        foreground_fractions=mask.mean((1, 2, 3)).cpu().tolist(),
        loss_absolute_differences={
            key: float((serial[key] - batched[key]).abs()) for key in serial
        },
        gradients=gradients,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if errors:
        raise AssertionError("Serial/batched tolerance check failed; see saved report")


if __name__ == "__main__":
    main()
