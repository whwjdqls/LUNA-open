"""Human-crop evaluation. SSIM uses an 11x11 Gaussian window, sigma=1.5."""

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor


def ssim(x: Tensor, y: Tensor) -> Tensor:
    axis = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-axis.square() / (2 * 1.5**2))
    g = g / g.sum()
    kernel = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    mean_x, mean_y = [F.conv2d(v, kernel, groups=3) for v in (x, y)]
    var_x = F.conv2d(x * x, kernel, groups=3) - mean_x.square()
    var_y = F.conv2d(y * y, kernel, groups=3) - mean_y.square()
    covariance = F.conv2d(x * y, kernel, groups=3) - mean_x * mean_y
    score = (
        (2 * mean_x * mean_y + 0.01**2)
        * (2 * covariance + 0.03**2)
        / ((mean_x.square() + mean_y.square() + 0.01**2) * (var_x + var_y + 0.03**2))
    )
    return score.mean(dim=(1, 2, 3))


@torch.no_grad()
def image_metrics(prediction: dict, target: Tensor, mask: Tensor, lpips=None) -> dict:
    rgb = prediction["rgb"].clamp(0, 1)
    mse = (rgb - target).square().mean(dim=(1, 2, 3))
    pred_mask = prediction["alpha"] >= 0.5
    true_mask = mask >= 0.5
    intersection = (pred_mask & true_mask).sum(dim=(1, 2, 3))
    union = (pred_mask | true_mask).sum(dim=(1, 2, 3))
    scores = dict(
        psnr=-10 * torch.log10(mse.clamp_min(1e-10)),
        l1=(rgb - target).abs().mean(dim=(1, 2, 3)),
        ssim=ssim(rgb, target),
        mask_iou=intersection / union.clamp_min(1),
    )
    if lpips is not None:
        scores["lpips"] = lpips(2 * rgb - 1, 2 * target - 1).flatten(1).mean(1)
    return scores


def aggregate_records(records: list[dict]) -> dict:
    scenes = defaultdict(list)
    for row in records:
        scenes[row["scene"]].append(row["metrics"])
    per_scene = {
        scene: {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}
        for scene, rows in scenes.items()
    }
    macro = {
        key: sum(r[key] for r in per_scene.values()) / len(per_scene)
        for key in next(iter(per_scene.values()))
    }
    return dict(per_scene=per_scene, mean_over_scenes=macro, frames=records)
