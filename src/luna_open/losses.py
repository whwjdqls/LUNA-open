"""Loss reductions are explicit to avoid accidental hybrid-label reweighting."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .avatar import Gaussians
from .geometry import normalize_quaternion, project_points


def quaternion_loss(a: Tensor, b: Tensor) -> Tensor:
    dot = (normalize_quaternion(a) * normalize_quaternion(b)).sum(-1)
    return 1 - dot.abs().clamp(max=1)


def structural_loss(prediction: Gaussians, teacher: Gaussians, labeled: Tensor) -> Tensor:
    """Full-batch mean; unlabeled samples contribute zero. Teacher is detached."""
    teacher = teacher.detach()
    mask = labeled.bool()
    if not mask.any():
        return prediction.means.sum() * 0
    # Select labels before arithmetic: multiplying missing-label NaNs by zero
    # would still poison the loss and gradients.
    position = (prediction.means[mask] - teacher.means[mask]).abs().mean(dim=(-1, -2))
    rotation = quaternion_loss(prediction.quaternions[mask], teacher.quaternions[mask]).mean(-1)
    color = (prediction.colors[mask] - teacher.colors[mask]).abs().mean(dim=(-1, -2))
    return (position + 0.5 * rotation + 0.5 * color).sum() / len(mask)


def projection_loss(
    prediction: Tensor, teacher: Tensor, K: Tensor, size: tuple[int, int], labeled: Tensor
) -> Tensor:
    selected = labeled.bool()
    if not selected.any():
        return prediction.sum() * 0
    source = teacher.detach()[selected]
    valid = torch.isfinite(source).all(-1) & (source[..., 2] > 1e-4)
    dummy = source.new_tensor([0.0, 0.0, 1.0])
    pred, _ = project_points(
        torch.where(valid[..., None], prediction[selected], dummy), K[selected]
    )
    target, _ = project_points(torch.where(valid[..., None], source, dummy), K[selected])
    normalizer = pred.new_tensor((size[1], size[0]))
    error = ((pred - target) / normalizer).abs().mean(-1)
    per_sample = (error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    return per_sample.sum() / len(selected)


def rendering_losses(rendered: dict, rgb: Tensor, mask: Tensor, lpips=None) -> dict:
    losses = dict(rgb=F.l1_loss(rendered["rgb"], rgb), mask=F.l1_loss(rendered["alpha"], mask))
    if lpips is not None:
        losses["lpips"] = lpips(2 * rendered["rgb"] - 1, 2 * rgb - 1).mean()
    return losses


def geometry_priors(gaussians: Gaussians, anchors: Tensor) -> dict:
    ratio = gaussians.scales.amax(-1) / gaussians.scales.amin(-1).clamp_min(1e-8)
    return dict(
        anisotropy=0.01 * F.relu(ratio - 5).mean(),
        offset=10 * F.relu((gaussians.means - anchors).norm(dim=-1) - 0.0525).mean(),
    )
