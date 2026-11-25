"""Explicit rotary coordinates at Wan's existing self-attention Q/K hook."""

from dataclasses import dataclass
from functools import wraps

import torch


def native_wan_rope(freqs, image_side, global_uv):
    """Return shared (time, row, column) coordinates and Wan rotary phases."""
    if not 1 <= image_side <= len(freqs) or len(freqs) < 2:
        raise ValueError('Image positions are outside the Wan frequency table')
    device = freqs.device
    row, col = torch.meshgrid(*(torch.arange(image_side, device=device),) * 2, indexing='ij')
    zero = torch.zeros_like(row)
    image = torch.stack((zero, row, col), -1).reshape(-1, 3)
    u, v = global_uv.to(device=device, dtype=torch.float64).unbind(-1)
    extent = image_side - 1
    uv = torch.stack((torch.ones_like(u), (1 - v) * extent, u * extent), -1)

    channels = freqs.shape[-1]
    spatial = channels // 3
    bands = freqs.split((channels - 2 * spatial, spatial, spatial), dim=-1)
    # Retain exact table entries for pixels and the temporal axis. Spatial UV
    # patch centers are fractional positions in the same [0, image_side-1]
    # domain. At position one, Wan's positive phase increments are all <= 1.
    image_phases = torch.cat([band[image[:, axis]] for axis, band in enumerate(bands)], dim=-1)
    uv_phases = [bands[0][1].expand(len(uv), -1)]
    for axis, band in enumerate(bands[1:], start=1):
        angles = uv[:, axis, None] * torch.angle(band[1])[None]
        uv_phases.append(torch.polar(torch.ones_like(angles), angles))
    phases = torch.cat((image_phases, torch.cat(uv_phases, dim=-1)), dim=0)
    return torch.cat((image, uv), dim=0), phases


@dataclass(frozen=True)
class ExplicitWanRoPE:
    phases: torch.Tensor  # [S, head_dim/2] complex, shared by all heads and batch elements

    def apply(self, x):
        if (x.shape[1], x.shape[-1] // 2) != self.phases.shape:
            raise ValueError('Explicit Wan RoPE dimensions differ from Q/K')
        with torch.autocast(device_type=x.device.type, enabled=False):
            paired = torch.view_as_complex(x.to(torch.float64).unflatten(-1, (-1, 2)))
            return torch.view_as_real(paired * self.phases[None, :, None]).flatten(-2).float()


def install_explicit_rope(module):
    """Dispatch at the rotary operation. Wan retains its Q/K/V, norms and blocks."""
    if getattr(module.rope_apply, '_has_explicit_uv_rope', False):
        return
    upstream = module.rope_apply

    @wraps(upstream)
    def rope_apply(x, grid_sizes, freqs):
        if isinstance(grid_sizes, ExplicitWanRoPE):
            return grid_sizes.apply(x)
        return upstream(x, grid_sizes, freqs)

    rope_apply._has_explicit_uv_rope = True
    module.rope_apply = rope_apply
