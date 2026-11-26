"""Canonical geometry on native SMPL-X islands. All distances are in meters."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from luna_open.geometry import normalize_quaternion

REFERENCE_WEIGHTS = dict(scale=1., smooth=1e-4, normal=.1, offset=.1, strain=0., pixel=0.)


# Express ARG logits -2 and +3 as log physical radius ratios relative
# to its scale at z=0. The constant scale base cancels in the ratio.
SCALE_LOWER, SCALE_UPPER = (math.log(2 / (1 + math.exp(-z))) for z in (-2., 3.))


def add_geometry_arguments(parser):
    parser.add_argument('--coarse-to-fine', action='store_true')
    parser.add_argument('--coarse-range', type=float, default=.135)
    parser.add_argument('--fine-range', type=float, default=.015)
    parser.add_argument('--canonical-regularization', action='store_true')
    for key in REFERENCE_WEIGHTS:
        parser.add_argument(f'--{key}-weight', type=float)


def geometry_weights(args):
    weights = {}
    for name, reference in REFERENCE_WEIGHTS.items():
        value = getattr(args, f'{name}_weight', None)
        weights[name] = value if value is not None else (reference if getattr(args, 'canonical_regularization', False) else 0.)
    if any(v < 0 or not math.isfinite(v) for v in weights.values()):
        raise ValueError('Geometry loss weights must be finite and nonnegative')
    coarse = getattr(args, 'coarse_to_fine', False)
    if (coarse or any(weights.values())) and args.uv_representation != 'native-islands':
        raise ValueError('Canonical geometry requires native UV islands')
    if (weights['strain'] or weights['pixel']) and not coarse:
        raise ValueError('Strain and patch-mean residuals require the coarse-to-fine decoder')
    if coarse and not (0 < args.fine_range < args.coarse_range and
                       math.isclose(args.fine_range + args.coarse_range, .15, abs_tol=1e-8)):
        raise ValueError('Positive coarse and fine ranges must sum to 0.15 m, with fine smaller')
    return weights


def update_regularizers(geometry, fields, aux, weights):
    """Evaluate each available raw term once. Disabled terms are diagnostics only."""
    names = [k for k in weights if k not in ('strain', 'pixel') or 'fine_offset' in aux]
    active = [k for k in names if weights.get(k, 0)]
    inactive = [k for k in names if not weights.get(k, 0)]
    terms = geometry.losses(fields, aux, active) if active else {}
    if inactive:
        with torch.no_grad():
            terms.update(geometry.losses(fields, aux, inactive))
    return terms


class CanonicalGeometry(nn.Module):
    """Fixed sample mesh correspondence and fully occupied native UV neighborhoods."""

    def __init__(self, atlas, layout, vertices, faces):
        super().__init__()
        vertices, faces = vertices.detach().float().cpu(), torch.as_tensor(faces).long().cpu()
        tri, bary = atlas.triangles.detach().cpu(), atlas.barycentric.detach().float().cpu()
        self.register_buffer('vertices', vertices)
        self.register_buffer('faces', faces)
        self.register_buffer('triangles', tri)
        self.register_buffer('barycentric', bary)
        parent = torch.cat([i.dense_to_patch.cpu() + start for i, start in zip(layout.islands, layout.token_offsets)])
        if len(parent) != len(tri):
            raise ValueError('Token and mesh sample counts differ')
        self.register_buffer('parent', parent)
        self.register_buffer('patch_counts', parent.bincount(minlength=layout.num_tokens))
        mass = vertices.new_zeros(len(vertices)).index_add_(0, tri.flatten(), bary.flatten())
        self.register_buffer('mass', mass)
        mesh = vertices[faces]
        area_normals = torch.linalg.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0])
        normals = torch.zeros_like(vertices)
        for corner in faces.T:
            normals.index_add_(0, corner, area_normals)
        self.register_buffer('normals', F.normalize((F.normalize(normals, dim=-1)[tri] * bary[..., None]).sum(1), dim=-1))
        edges = torch.cat((faces[:, :2], faces[:, 1:], faces[:, [2, 0]])).sort(-1).values.unique(dim=0)
        rest = vertices[edges[:, 1]] - vertices[edges[:, 0]]
        used = (mass[edges] > 0).all(-1) & (rest.norm(dim=-1) > 1e-8)
        self.register_buffer('edges', edges[used])
        self.register_buffer('rest_edges', rest[used])

        neighborhoods = []
        for island, start in zip(layout.islands, layout.gaussian_offsets):
            h, w = island.dense_shape
            ids = torch.full((h * w,), -1, dtype=torch.long)
            ids[island.dense_indices.cpu()] = torch.arange(len(island.dense_indices)) + start
            if min(h, w) >= 3:
                windows = ids.view(h, w).unfold(0, 3, 1).unfold(1, 3, 1)
                windows = windows[(windows >= 0).all((-2, -1))].reshape(-1, 9)
                neighborhoods.append(windows[:, [4, 1, 3, 5, 7]])
        self.register_buffer('neighbors', torch.cat(neighborhoods) if neighborhoods else torch.empty(0, 5, dtype=torch.long))

    def lift(self, token_offsets):
        offsets = token_offsets.float().index_select(-2, self.parent)
        vertices = offsets.new_zeros((*offsets.shape[:-2], len(self.vertices), 3))
        for corner, weights in zip(self.triangles.T, self.barycentric.T):
            vertices.index_add_(-2, corner, offsets * weights[:, None])
        vertices = vertices / torch.where(self.mass > 0, self.mass + 1e-8, 1.)[:, None]
        coarse = (vertices[..., self.triangles, :] * self.barycentric[..., None]).sum(-2)
        return vertices, coarse

    def laplacian(self, offset, residual, quaternion):
        features = torch.cat((offset, residual, .1 * offset.norm(dim=-1, keepdim=True)), -1)
        center, *neighbors = self.neighbors.T
        q0 = quaternion.index_select(-2, center)
        d = -4 * features.index_select(-2, center)
        dq = -4 * q0
        for idx in neighbors:
            q = quaternion.index_select(-2, idx)
            sign = torch.where((q * q0).sum(-1, keepdim=True) < 0, -1., 1.)
            dq = dq + sign * q
            d = d + features.index_select(-2, idx)
        return d.square().sum(-1) + dq.square().sum(-1)

    def alignment(self, offset):
        norm = offset.norm(dim=-1)
        valid = (norm > .001) & (self.normals.norm(dim=-1) > .5)
        cosine = (offset * self.normals).sum(-1) / norm.clamp_min(.001)
        return (1 - cosine.square()).clamp_min(0), valid

    def losses(self, fields, aux, names):
        """Each term averages within an outfit first, then equally across outfits."""
        offset, s, q = fields['offset'].float(), aux['scale_residual'].float(), fields['quaternion'].float()
        result = {}
        if 'scale' in names:
            result['scale'] = torch.maximum(F.relu(s - SCALE_UPPER) + F.relu(SCALE_LOWER - s), .1 * F.relu(s)).mean()
        if 'smooth' in names:
            lap = self.laplacian(offset, s, q)
            result['smooth'] = lap.sum(-1).div(max(1, lap.shape[-1])).mean()
        if 'normal' in names:
            error, valid = self.alignment(offset)
            result['normal'] = ((error * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()
        if 'offset' in names:
            result['offset'] = offset.square().sum(-1).mean()
        if 'strain' in names:
            v = aux['vertex_offset']
            e = self.rest_edges + v[..., self.edges[:, 1], :] - v[..., self.edges[:, 0], :]
            error = (e.norm(dim=-1) / self.rest_edges.norm(dim=-1) - 1).square()
            result['strain'] = error.sum(-1).div(max(1, len(self.edges))).mean()
        if 'pixel' in names:
            fine = aux['fine_offset']
            sums = fine.new_zeros((*fine.shape[:-2], len(self.patch_counts), 3)).index_add(-2, self.parent, fine)
            result['pixel'] = (sums / self.patch_counts[:, None]).square().sum(-1).mean()
        return result


def activate(raw, base_log_scale, *, token_offsets=None, geometry=None, fine_range=.015):
    offset, q, scale, opacity, color = raw.float().split((3, 4, 3, 1, 3), dim=-1)
    residual = 3 * scale.tanh()
    aux = dict(scale_raw=scale, scale_residual=residual, rgb_logit=color)
    if token_offsets is None:
        offset = .15 * offset.tanh()
    else:
        vertices, coarse = geometry.lift(token_offsets)
        fine = fine_range * offset.tanh()
        aux.update(token_offset=token_offsets, vertex_offset=vertices, coarse_offset=coarse, fine_offset=fine)
        offset = coarse + fine
    fields = dict(offset=offset, quaternion=normalize_quaternion(q + q.new_tensor([1., 0., 0., 0.])),
                  log_scale=(base_log_scale + residual).clamp(math.log(.0004), math.log(.045)),
                  opacity_logit=opacity + 1.1, rgb=color.sigmoid())
    return fields, aux
