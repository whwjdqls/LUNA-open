"""Rasterize each SMPL-X UV island at its anatomical sampling density."""

import json
import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from torch import nn

from .representation import Atlas, rasterize_atlas


def island_ids(uv_faces, num_uv_vertices):
    left = uv_faces.reshape(-1)
    right = np.roll(uv_faces, -1, axis=1).reshape(-1)
    graph = coo_matrix((np.ones_like(left), (left, right)), shape=(num_uv_vertices, num_uv_vertices))
    _, labels = connected_components(graph.tocsr(), directed=False)
    return labels[uv_faces[:, 0]]


def island_region(mesh_faces, uv_triangles, segmentation):
    area = np.abs(np.linalg.det(uv_triangles[:, 1:] - uv_triangles[:, :1]))
    region_ids = (
        ('head', [index for part in ('head', 'neck', 'leftEye', 'rightEye', 'eyeballs')
                  for index in segmentation.get(part, [])]),
        ('hands', [index for part, ids in segmentation.items() if 'Hand' in part for index in ids]),
    )
    for name, ids in region_ids:
        covered = np.isin(mesh_faces, ids).mean(axis=1)
        if (area * covered).sum() > area.sum() * 0.5:
            return name
    return 'body'


class UVIsland(nn.Module):
    """UV feature grid and valid output indices for one isolated CNN pass."""

    def __init__(self, mask, left, top, resolution):
        super().__init__()
        self.register_buffer('valid_texels', torch.from_numpy(mask), persistent=False)
        height, width = mask.shape
        pad_height, pad_width = (-height) % 8, (-width) % 8
        padded = np.pad(mask, ((0, pad_height), (0, pad_width)))
        coarse = padded.reshape(padded.shape[0] // 8, 8, padded.shape[1] // 8, 8).any((1, 3))
        y, x = np.mgrid[:coarse.shape[0], :coarse.shape[1]]
        grid = np.stack((left + 8 * x + 4, top + 8 * y + 4), axis=-1)
        self.register_buffer('grid', torch.from_numpy((grid * (2 / resolution) - 1).astype(np.float32))[None])
        self.register_buffer('coarse_mask', torch.from_numpy(coarse)[None, None])
        self.register_buffer('indices', torch.from_numpy(np.flatnonzero(padded)))


class AdaptiveAtlas(Atlas):
    """Adaptive Gaussian anchors with a uniform atlas preview for display."""

    def __init__(self, uv, uv_faces, mesh_faces, size, segmentation, body_scale=2.0):
        nn.Module.__init__(self)
        if size < 2 or size % 2 or uv_faces.shape != mesh_faces.shape:
            raise ValueError('Adaptive atlas requires an even resolution and matching UV/mesh face order')
        body_resolution = size * body_scale
        if not math.isfinite(body_resolution) or round(body_resolution) < 1:
            raise ValueError('The adaptive body scale must produce a positive finite resolution')
        self.size = size
        self.islands = nn.ModuleList()
        self.island_stats = []
        components = island_ids(uv_faces, len(uv))
        display_faces, _ = rasterize_atlas(uv, uv_faces, size)
        display_component = np.where(display_faces >= 0, components[display_faces], -1)
        display = np.full((size, size), -1, dtype=np.int64)
        all_faces, all_weights, all_coords, all_resolutions = [], [], [], []
        used = 0
        for component in np.unique(components):
            faces = np.flatnonzero(components == component)
            region = island_region(mesh_faces[faces], uv[uv_faces[faces]], segmentation)
            scale = {'hands': 0.5, 'head': 1.0, 'body': body_scale}[region]
            resolution = round(size * scale)
            coords = uv[uv_faces[faces].reshape(-1)] * [resolution, -resolution] + [0, resolution]
            left, top = np.maximum(np.floor(coords.min(axis=0)).astype(int), 0)
            right, bottom = np.minimum(np.ceil(coords.max(axis=0)).astype(int), resolution)
            if right <= left or bottom <= top:
                continue
            face_ids, weights = rasterize_atlas(uv, uv_faces[faces], resolution, (left, top, right, bottom))
            mask = face_ids >= 0
            if not mask.any():
                continue
            y, x = np.nonzero(mask)
            samples = np.stack((left + x + 0.5, top + y + 0.5), axis=-1) / resolution
            all_faces.append(faces[face_ids[mask]])
            all_weights.append(weights[mask])
            all_coords.append(samples)
            all_resolutions.append(np.full(len(samples), resolution, dtype=np.float32))
            self.islands.append(UVIsland(mask, left, top, resolution))
            self.island_stats.append(dict(region=region, scale=scale, resolution=resolution,
                                          bounds=[int(v) for v in (left, top, right, bottom)],
                                          gaussians=len(samples)))
            dy, dx = np.nonzero(display_component == component)
            if len(dx):
                preview_uv = (np.stack((dx, dy), axis=-1) + 0.5) / size
                _, nearest = cKDTree(samples).query(preview_uv)
                display[dy, dx] = used + nearest
            used += len(samples)
        if not used:
            raise ValueError('The adaptive atlas has no valid texels')
        self.register_anchors(mesh_faces, np.concatenate(all_faces), np.concatenate(all_weights))
        self.register_buffer('resolution_per_texel', torch.from_numpy(np.concatenate(all_resolutions)))
        uv_grid = torch.from_numpy(np.concatenate(all_coords).astype(np.float32) * 2 - 1)
        self.register_buffer('base_grid', uv_grid[None, None])
        self.register_buffer('preview_indices', torch.from_numpy(display))
        self.register_buffer('valid', torch.from_numpy(display >= 0))

    @classmethod
    def from_smplx(cls, asset, size, segmentation, body_scale=2.0):
        regions = json.loads(segmentation.read_text())
        with np.load(asset, allow_pickle=False) as data:
            return cls(data['vt'], data['ft'], data['f'], size, regions, body_scale=body_scale)

    def to_map(self, values):
        preview = values[..., self.preview_indices.clamp_min(0), :]
        return preview * self.valid[..., None]
