"""Fixed standard SMPL-X vertices, anatomical routing, and differentiable LBS.

Canonical pose follows LHM cano_pose_type=1 (hips ±pi/9, jaw x=1/3).
Standard SMPL-X expression/PCA bases are retained from the dataset asset.
Unlike native LHM's voxel queries, every Gaussian has an exact vertex identity.
Geometry and labels are fixed; gradients flow through predicted attributes.
"""
import json
import math

import torch
from torch import nn

from .avatar import Gaussians
from .data.smplx_records import mvhuman_standard_smplx
from .geometry import matrix_to_quaternion, proper_rotation, quaternion_multiply


def anatomical_tables(path, count=10475):
    regions = json.loads(open(path).read())
    names = ['unlabelled'] + sorted(regions)
    labels = torch.zeros(count, dtype=torch.long)
    # Broad regions first, smallest anatomical region wins overlaps; ties by name.
    # Unlabelled vertices retain an explicit learned "unlabelled" embedding.
    for name in sorted(regions, key=lambda key: (-len(regions[key]), key)):
        ids = torch.tensor(regions[name], dtype=torch.long)
        if ids.min() < 0 or ids.max() >= count:
            raise ValueError('Segmentation incompatible with standard SMPL-X topology')
        labels[ids] = names.index(name)
    head_names = {'head', 'neck', 'leftEye', 'rightEye', 'eyeballs'}
    head = torch.zeros(count, dtype=torch.bool)
    for name in head_names:
        head[regions[name]] = True
    hands = torch.tensor(['Hand' in names[i] for i in labels]) & ~head
    lower = torch.tensor([any(s in names[i] for s in ('Leg', 'Foot', 'Toe')) for i in labels]) & ~head & ~hands
    upper = ~(head | hands | lower)
    return labels, head.nonzero().flatten(), dict(head=head, hands=hands, lower_body=lower,
                                                upper_body=upper), names


class VertexSMPLX(nn.Module):
    def __init__(self, asset, segmentation):
        super().__init__()
        import smplx

        self.body = smplx.SMPLX(str(asset), use_pca=False, flat_hand_mean=True,
                                num_betas=10, num_expression_coeffs=10).eval().requires_grad_(False)
        if self.body.v_template.shape != (10475, 3):
            raise ValueError('Expected all 10475 standard SMPL-X vertices')
        # Components are not created by SMPLX when use_pca=False; obtain from asset.
        import numpy as np
        with np.load(asset, allow_pickle=False) as data:
            self.components = {side: data[f'hands_components{side[0]}'][:6].astype('float32')
                               for side in ('left', 'right')}
        labels, head, groups, self.label_names = anatomical_tables(segmentation)
        self.register_buffer('labels', labels)
        self.register_buffer('head_indices', head)
        for name, mask in groups.items():
            self.register_buffer('group_' + name, mask)
        # Native upper-body scale constraint includes hands.
        self.register_buffer('upper_body', groups['upper_body'] | groups['hands'])
        pose = torch.zeros(1, 165)
        pose[0, 5], pose[0, 8], pose[0, 66] = math.pi / 9, -math.pi / 9, 1 / 3
        self.register_buffer('canonical_pose', pose)
        with torch.no_grad():
            surface, transforms = self.mesh(pose, torch.zeros(1, 10), torch.zeros(1, 10))
        self.register_buffer('anchors', surface[0])
        self.register_buffer('inverse_bind', torch.linalg.inv(transforms[0]))

    @torch.no_grad()
    @torch.autocast('cuda', enabled=False)
    def mesh(self, pose, betas, expression):
        from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints

        shaped = self.body.v_template + blend_shapes(betas.float(), self.body.shapedirs)
        shaped = shaped + blend_shapes(expression.float(), self.body.expr_dirs)
        joints = vertices2joints(self.body.J_regressor, shaped)
        rotations = batch_rodrigues(pose.float().reshape(-1, 3)).reshape(-1, 55, 3, 3)
        feature = (rotations[:, 1:] - torch.eye(3, device=pose.device)).reshape(len(pose), -1)
        offsets = (feature @ self.body.posedirs).reshape(-1, 10475, 3)
        _, transforms = batch_rigid_transform(rotations, joints, self.body.parents)
        vertex = torch.einsum('vj,bjkl->bvkl', self.body.lbs_weights, transforms)
        surface = torch.einsum('bvij,bvj->bvi', vertex[..., :3, :3], shaped + offsets) + vertex[..., :3, 3]
        return surface, transforms

    def records(self, records, device):
        values = [mvhuman_standard_smplx(record, self.components) for record in records]
        import numpy as np
        return {key: torch.from_numpy(np.stack([v[key] for v in values])).to(device)
                for key in ('pose', 'betas', 'expression', 'transl')}

    @torch.no_grad()
    def deformation(self, records):
        values = self.records(records, self.anchors.device)
        surface, transforms = self.mesh(values['pose'], values['betas'], values['expression'])
        # LHM composes joint transforms before blending. Exact target mesh anchor
        # adds standard SMPL-X shape/expression/pose corrections at each vertex.
        relative = transforms @ self.inverse_bind[None]
        linear = torch.einsum('vj,bjkl->bvkl', self.body.lbs_weights, relative)[..., :3, :3]
        rotation = matrix_to_quaternion(proper_rotation(linear))
        return dict(surface=surface + values['transl'][:, None], linear=linear, rotation=rotation)

    def animate(self, canonical, deformation, outfit_indices):
        with torch.autocast('cuda', enabled=False):
            delta = canonical.means[outfit_indices].float() - self.anchors[None]
            means = deformation['surface'] + torch.einsum('bvij,bvj->bvi', deformation['linear'], delta)
            return Gaussians(means, quaternion_multiply(deformation['rotation'], canonical.quaternions[outfit_indices].float()),
                             canonical.scales[outfit_indices], canonical.opacities[outfit_indices],
                             canonical.colors[outfit_indices])

    def priors(self, gaussian, settings):
        scale = gaussian.scales.float()
        ratio = scale.amax(-1) / (scale.amin(-1) + 1e-6)
        penalty = (ratio - settings['asap_ratio']).clamp_min(0)
        asap = sum(weight * penalty[:, getattr(self, 'group_' + name)].mean()
                   for name, weight in settings['asap_groups'].items())
        offset = gaussian.means.float() - self.anchors[None]
        acap = (offset.norm(dim=-1) - settings['acap_distance']).clamp_min(0).mean()
        return dict(asap=asap, acap=acap)
