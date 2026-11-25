"""SMPL-X UV texels with ARG-Avatar-style transport of Gaussian offsets."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from luna_open.avatar import Gaussians
from luna_open.geometry import (
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
)


@dataclass
class PosedSurface:
    anchors: Tensor  # [..., N, 3] world metres
    rotation: Tensor  # [..., N, 4] wxyz, bind to posed


@dataclass
class PoseDelta:
    offset: Tensor  # [..., N, 3] canonical metres
    rotation: Tensor  # [..., N, 4] canonical wxyz multiplier
    log_scale: Tensor  # [..., N, 3] logarithm of the scale multiplier


def pose_gaussians(surface, offset, quaternion, log_scale, opacity_logit, rgb):
    """Render-ready attributes from canonical fields and a posed UV surface."""
    rotation = quaternion_to_matrix(surface.rotation)
    center = surface.anchors + (rotation @ offset.unsqueeze(-1)).squeeze(-1)
    return Gaussians(
        center,
        quaternion_multiply(surface.rotation, quaternion),
        log_scale.exp().expand_as(center),
        opacity_logit.squeeze(-1).sigmoid().expand(center.shape[:-1]),
        rgb.expand_as(center),
    )


def rasterize_atlas(uv: np.ndarray, uv_faces: np.ndarray, size: int, bounds=None):
    """Assign triangles and barycentrics on a full or cropped texel lattice."""
    xy = uv.astype(np.float64) * [size, -size] + [0, size]
    left, top, right, bottom = bounds if bounds is not None else (0, 0, size, size)
    xy -= [left, top]
    width, height = right - left, bottom - top
    face_ids = np.full((height, width), -1, dtype=np.int64)
    weights = np.zeros((height, width, 3), dtype=np.float32)
    for index, corners in enumerate(xy[uv_faces]):
        low = np.maximum(np.ceil(corners.min(0) - 0.5).astype(int), 0)
        high = np.minimum(np.floor(corners.max(0) - 0.5).astype(int), [width - 1, height - 1])
        if np.any(high < low):
            continue
        a, b, c = corners
        basis = np.column_stack((b - a, c - a))
        if abs(np.linalg.det(basis)) < 1e-12:
            continue
        xx, yy = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
        bc = (np.stack((xx, yy), axis=-1) + 0.5 - a) @ np.linalg.inv(basis).T
        bary = np.concatenate((1 - bc.sum(-1, keepdims=True), bc), axis=-1)
        inside = (bary.min(-1) >= -1e-9) & (face_ids[yy, xx] == -1)
        face_ids[yy[inside], xx[inside]] = index
        weights[yy[inside], xx[inside]] = bary[inside]
    return face_ids, weights


class Atlas(nn.Module):
    """Fixed texel topology shared by the canonical and every posed mesh."""

    def __init__(self, uv, uv_faces, mesh_faces, size):
        super().__init__()
        if uv_faces.shape != mesh_faces.shape:
            raise ValueError('UV faces and SMPL-X mesh faces must have the same ordering')
        face_ids, weights = rasterize_atlas(uv, uv_faces, size)
        valid = face_ids >= 0
        if not valid.any():
            raise ValueError('The UV atlas has no valid texels')
        self.size = size
        self.register_buffer('valid', torch.from_numpy(valid))
        self.register_anchors(mesh_faces, face_ids[valid], weights[valid])
        self.register_buffer('pixels', torch.from_numpy(np.flatnonzero(valid.reshape(-1))))

    def register_anchors(self, mesh_faces, face_ids, weights):
        self.register_buffer('face_ids', torch.from_numpy(face_ids))
        self.register_buffer('barycentric', torch.from_numpy(weights))
        self.register_buffer('triangles', torch.as_tensor(mesh_faces[face_ids]).long())

    @classmethod
    def from_smplx(cls, asset, size):
        with np.load(asset, allow_pickle=False) as data:
            return cls(data['vt'], data['ft'], data['f'], size)

    def sample(self, vertex_values):
        return (vertex_values[..., self.triangles, :] * self.barycentric[..., None]).sum(-2)

    def to_map(self, values):
        """Scatter [..., N, C] into [..., H, W, C] with zero invalid texels."""
        dense = values.new_zeros(*values.shape[:-2], self.size**2, values.shape[-1])
        dense[..., self.pixels, :] = values
        return dense.unflatten(-2, (self.size, self.size))

    def gather(self, dense):
        return dense.flatten(-3, -2)[..., self.pixels, :]

    def pose(self, vertices, vertex_linear):
        # ARG-Avatar Appendix C converts before barycentric interpolation and
        # uses a nonnegative scalar component to choose quaternion signs.
        q = matrix_to_quaternion(vertex_linear)
        q = torch.where(q[..., :1] < 0, -q, q)
        return PosedSurface(self.sample(vertices), normalize_quaternion(self.sample(q)))


class CanonicalGaussians(nn.Module):
    """One independently optimizable Gaussian per valid UV texel."""

    def __init__(self, anchors, colors, initial_scales):
        super().__init__()
        self.register_buffer('anchors', anchors)
        self.offset = nn.Parameter(torch.zeros_like(anchors))
        q = anchors.new_zeros(len(anchors), 4)
        q[:, 0] = 1
        self.quaternion = nn.Parameter(q)
        self.log_scale = nn.Parameter(initial_scales.log())
        self.opacity_logit = nn.Parameter(anchors.new_full((len(anchors), 1), 1.1))
        self.rgb_logit = nn.Parameter(torch.logit(colors.clamp(0.025, 0.975)))

    def fields(self):
        return {
            'offset_m': self.offset,
            'quaternion_wxyz': normalize_quaternion(self.quaternion),
            'log_scale_m': self.log_scale,
            'opacity_logit': self.opacity_logit,
            'rgb': self.rgb_logit.sigmoid(),
        }

    def canonical_surface(self):
        identity = self.quaternion.new_zeros(self.quaternion.shape)
        identity[:, 0] = 1
        return PosedSurface(self.anchors, identity)

    def forward(self, surface, delta: PoseDelta | None = None):
        offset = self.offset
        q = normalize_quaternion(self.quaternion)
        log_scale = self.log_scale
        if delta is not None:
            offset = offset + delta.offset
            q = quaternion_multiply(q, normalize_quaternion(delta.rotation))
            log_scale = log_scale + delta.log_scale
        return pose_gaussians(surface, offset, q, log_scale, self.opacity_logit, self.rgb_logit.sigmoid())
