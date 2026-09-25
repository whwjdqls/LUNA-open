"""gsplat 1.5.3 wrapper. Input/output geometry is FP32 even under autocast."""

import torch
from torch import Tensor

from .avatar import Gaussians


def render(
    gaussians: Gaussians, K: Tensor, size: tuple[int, int], viewmats: Tensor | None = None
) -> dict[str, Tensor]:
    from gsplat import rasterization

    if not gaussians.means.is_cuda:
        raise RuntimeError("gsplat rendering requires CUDA; use a Slurm GPU allocation")
    height, width = size
    batch = gaussians.means.shape[0]
    if viewmats is None:
        viewmats = torch.eye(4, device=K.device).expand(batch, 4, 4)
    images, alphas, depths = [], [], []
    with torch.autocast(device_type="cuda", enabled=False):
        for b in range(batch):
            image, alpha, _ = rasterization(
                means=gaussians.means[b].float(),
                quats=gaussians.quaternions[b].float(),
                scales=gaussians.scales[b].float(),
                opacities=gaussians.opacities[b].float(),
                colors=gaussians.colors[b].float(),
                viewmats=viewmats[b : b + 1].float(),
                Ks=K[b : b + 1].float(),
                width=width,
                height=height,
                packed=False,
                backgrounds=torch.ones(1, 3, device=K.device),
                render_mode="RGB+ED",
                near_plane=0.01,
                far_plane=100.0,
            )
            images.append(image[0, ..., :3].permute(2, 0, 1))
            alphas.append(alpha[0].permute(2, 0, 1))
            depths.append(image[0, ..., 3:].permute(2, 0, 1))
    return dict(rgb=torch.stack(images), alpha=torch.stack(alphas), depth=torch.stack(depths))
