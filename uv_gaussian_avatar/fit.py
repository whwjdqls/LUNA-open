"""Fit a subject's canonical UV map to calibrated MVHumanNet++ video frames."""

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
from gsplat import rasterization
from torch.nn import functional as F
from torch.utils.cpp_extension import _import_module_from_library

from luna_open.identity_v2_body import VertexSMPLX
from luna_open.identity_v2_data import IdentityData
from luna_open.official_mvh import atomic_json
from luna_open.provenance import file_sha256

from .representation import Atlas, CanonicalGaussians, PoseDelta, PosedSurface
from .visualize import export_media


def load_rasterizer(extension_dir):
    if extension_dir is not None:
        backend = ModuleType('gsplat.cuda._backend')
        backend._C = _import_module_from_library('gsplat_cuda', str(extension_dir), True)
        sys.modules[backend.__name__] = backend


def select_times(times):
    times = sorted(times)
    if len(times) < 49:
        raise ValueError('Expected at least 49 capture timestamps')
    segment = times[:49]
    return [stamp for i, stamp in enumerate(segment) if i % 4 != 2], segment[2::4], segment


def training_batches(times, num_cameras, batch_size, rng):
    """Cycle all camera-time pairs in shuffled epochs."""
    while True:
        batches = []
        for stamp in times:
            cameras = rng.sample(range(num_cameras), num_cameras)
            batches.extend((stamp, cameras[i:i + batch_size]) for i in range(0, num_cameras, batch_size))
        rng.shuffle(batches)
        yield from batches


def render(gaussians, views, size, mode='RGB'):
    h, w = size
    return rasterization(
        means=gaussians.means.contiguous(), quats=gaussians.quaternions.contiguous(),
        scales=gaussians.scales.contiguous(), opacities=gaussians.opacities.contiguous(),
        colors=gaussians.colors.contiguous(), viewmats=views['world_to_camera'], Ks=views['K'],
        width=w, height=h, packed=False, near_plane=0.01, far_plane=100, render_mode=mode,
        backgrounds=torch.ones(len(views['K']), 3, device=gaussians.means.device)
        if mode == 'RGB' else None,
    )[:2]


def stack_views(frames):
    return {
        'rgb': torch.stack([f['rgb'] for f in frames]).permute(0, 2, 3, 1).cuda(),
        'mask': torch.stack([f['mask'] for f in frames]).permute(0, 2, 3, 1).cuda(),
        'K': torch.stack([f['K'] for f in frames]).cuda(),
        'world_to_camera': torch.stack([f['world_to_camera'] for f in frames]).cuda(),
    }


def subset(views, indices):
    return {key: value[indices].contiguous() for key, value in views.items()}


def vertex_transforms(body, transforms, bind):
    relative = transforms @ torch.linalg.inv(bind)
    return torch.einsum('vj,bjkl->bvkl', body.body.lbs_weights, relative)[..., :3, :3]


@torch.no_grad()
def check_representation(atlas, model, canonical, surface):
    if not torch.equal(atlas.gather(atlas.to_map(model.offset)), model.offset):
        raise RuntimeError('UV dense/sparse round trip failed')
    identity = canonical.rotation.new_tensor([1.0, 0.0, 0.0, 0.0])
    bind_error = (canonical.rotation - identity).abs().max().item()
    if bind_error > 1e-5:
        raise RuntimeError(f'Canonical rotation changed by {bind_error}')
    sample = model.offset.new_tensor([0.02, -0.03, 0.04]).expand_as(model.offset)
    delta_rotation = identity.expand_as(model.quaternion)
    delta = PoseDelta(sample, delta_rotation, torch.full_like(model.log_scale, math.log(1.1)))
    original = model(surface)
    adjusted = model(surface, delta)
    norm_error = ((adjusted.means - original.means).norm(dim=-1) - sample.norm(dim=-1)).abs().max().item()
    scale_error = (adjusted.scales - original.scales * 1.1).abs().max().item()
    if norm_error > 1e-5 or scale_error > 1e-6:
        raise RuntimeError('Posed offset or scale transport failed')
    return {'bind_quaternion_max_error': bind_error, 'offset_length_max_error_m': norm_error,
            'scale_multiplier_max_error_m': scale_error, 'uv_round_trip': True}


@torch.no_grad()
def initialize_colors(model, faces, atlas, vertices, surface, views, size):
    triangles = vertices[faces][atlas.face_ids]
    normal = F.normalize(torch.linalg.cross(triangles[:, 1] - triangles[:, 0],
                                            triangles[:, 2] - triangles[:, 0]), dim=-1)
    depth, alpha = render(model(surface), views, size, 'ED')
    camera = views['world_to_camera']
    position = surface.anchors @ camera[:, :3, :3].transpose(1, 2) + camera[:, None, :3, 3]
    pixel = position @ views['K'].transpose(1, 2)
    uv = pixel[..., :2] / pixel[..., 2:].clamp_min(0.01)
    origin = torch.linalg.inv(camera)[:, None, :3, 3]
    incidence = (F.normalize(origin - surface.anchors, dim=-1) * normal).sum(-1).clamp_min(0)
    h, w = size
    grid = (2 * uv / uv.new_tensor([w, h]) - 1).unsqueeze(2)

    def sample(image):
        return F.grid_sample(image.permute(0, 3, 1, 2), grid, align_corners=False,
                             mode='nearest').squeeze(-1).transpose(1, 2)

    observed = (sample(views['mask'])[..., 0] > 0.95) & (sample(alpha)[..., 0] > 0.5)
    observed &= (sample(depth)[..., 0] - position[..., 2]).abs() < 0.025
    weight = observed * incidence.pow(4)
    mass = weight.sum(0)
    color = (sample(views['rgb']) * weight[..., None]).sum(0) / mass[:, None].clamp_min(1e-8)
    visible = mass > 0.01
    return color, visible


@torch.no_grad()
def measure(model, surfaces, views, pairs, size):
    rows = []
    for stamp, cameras in pairs.items():
        identifiers, observed = views[stamp]
        chosen = subset(observed, [identifiers.index(camera) for camera in cameras])
        rgb, alpha = render(model(surfaces[stamp]), chosen, size)
        error = (rgb - chosen['rgb']).square()
        foreground = chosen['mask'] > 0.5
        pred = alpha > 0.5
        psnr = -10 * error.mean((1, 2, 3)).clamp_min(1e-12).log10()
        fg_psnr = -10 * ((error * foreground).sum((1, 2, 3)) /
                         (3 * foreground.sum((1, 2, 3))).clamp_min(1)).clamp_min(1e-12).log10()
        iou = (pred & foreground).sum((1, 2, 3)) / (pred | foreground).sum((1, 2, 3)).clamp_min(1)
        for camera, all_score, fg_score, iou_score in zip(cameras, psnr.tolist(), fg_psnr.tolist(), iou.tolist()):
            rows.append(dict(frame=stamp, camera=camera, psnr=all_score,
                             foreground_psnr=fg_score, mask_iou=iou_score))
    return {'count': len(rows), **{k: float(np.mean([r[k] for r in rows]))
                                   for k in ('psnr', 'foreground_psnr', 'mask_iou')}, 'views': rows}


def fit(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(6)
    cv2.setNumThreads(2)
    rng = random.Random(args.seed)
    load_rasterizer(args.cuda_extension)
    torch.cuda.set_device(0)
    size = (args.height, args.width)
    data = IdentityData(args.protocol, args.split, render_size=size)
    if args.outfit not in data.ids:
        raise ValueError(f'{args.outfit} is absent from the {args.split} split')
    train_cameras = (0, 2, 4, 6, 8, 10, 12, 14)
    held_cameras = (1, 7, 13)
    train_times, held_times, video_times = select_times(data.document['outfits'][args.outfit]['times'])
    print(f'Loading {args.outfit}: {len(train_times)} fit poses, {len(held_times)} held-out poses', flush=True)
    body = VertexSMPLX(args.asset, args.segmentation).cuda()
    atlas = Atlas.from_smplx(args.asset, args.atlas_size).cuda()
    values = body.records([data.body(args.outfit, stamp) for stamp in video_times], 'cuda')
    fit_indices = [video_times.index(stamp) for stamp in train_times]
    betas = values['betas'][fit_indices].mean(0, keepdim=True)
    beta_drift = (values['betas'] - betas).norm(dim=-1).max().item()
    bind_pose = torch.zeros_like(values['pose'][:1])
    canonical_vertices, bind = body.mesh(bind_pose, betas, torch.zeros_like(values['expression'][:1]))
    betas = betas.expand(len(video_times), -1)
    vertices, transforms = body.mesh(values['pose'], betas, values['expression'])
    vertices = vertices + values['transl'][:, None]
    deformation = atlas.pose(vertices, vertex_transforms(body, transforms, bind))
    canonical = atlas.pose(canonical_vertices[0], vertex_transforms(body, bind, bind)[0])
    surfaces = {stamp: PosedSurface(deformation.anchors[i], deformation.rotation[i])
                for i, stamp in enumerate(video_times)}
    with np.load(args.asset, allow_pickle=False) as asset:
        faces = torch.from_numpy(asset['f'].astype('int64')).cuda()
    views = {}
    for i, stamp in enumerate(video_times):
        cams = train_cameras + held_cameras if stamp in train_times else (0, 8) + held_cameras
        frames = [data.target(args.outfit, camera, stamp) for camera in cams]
        views[stamp] = cams, stack_views(frames)
        if (i + 1) % 5 == 0:
            print(f'Loaded {i + 1}/{len(video_times)} timestamps', flush=True)

    # The UV derivatives vary by body part. Initialize a radius using the
    # physical area covered by a texel on each triangle of the bind mesh.
    initial_scales = estimate_scales(args.asset, atlas, canonical_vertices[0])
    gray = torch.full_like(canonical.anchors, 0.5)
    model = CanonicalGaussians(canonical.anchors, gray, initial_scales)
    init_stamp = train_times[len(train_times) // 2]
    color, visible = initialize_colors(model, faces, atlas, vertices[video_times.index(init_stamp)],
                                       surfaces[init_stamp], subset(views[init_stamp][1], list(range(len(train_cameras)))), size)
    with torch.no_grad():
        model.rgb_logit.copy_(torch.logit(torch.where(visible[:, None], color, gray).clamp(0.025, 0.975)))
    audit = check_representation(atlas, model, canonical, surfaces[held_times[len(held_times) // 2]])
    audit['initial_texture_coverage'] = visible.float().mean().item()
    print(f'Geometry verified. Valid anchors {len(atlas.pixels)}; color init {audit["initial_texture_coverage"]:.1%}', flush=True)

    groups = {
        'fit_views': {t: train_cameras for t in train_times},
        'held_cameras': {t: held_cameras for t in train_times},
        'held_poses': {t: (0, 8) for t in held_times},
        'held_cameras_and_poses': {t: held_cameras for t in held_times},
    }
    baseline = {name: measure(model, surfaces, views, pairs, size) for name, pairs in groups.items()}
    initial_log_scale = model.log_scale.detach().clone()
    neighbors = adjacent_texels(atlas, canonical.anchors)
    optimizer = torch.optim.Adam([
        {'params': [model.offset], 'lr': 0.0015},
        {'params': [model.quaternion, model.log_scale, model.opacity_logit], 'lr': 0.008},
        {'params': [model.rgb_logit], 'lr': 0.025},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps, eta_min=0.0002)
    history = []
    started = time.monotonic()
    stream = training_batches(train_times, len(train_cameras), args.views_per_step, rng)
    fit_coverage = set()
    for step in range(1, args.steps + 1):
        stamp, cameras = next(stream)
        fit_coverage.update((stamp, train_cameras[camera]) for camera in cameras)
        batch = subset(views[stamp][1], cameras)
        rgb, alpha = render(model(surfaces[stamp]), batch, size)
        mask = batch['mask']
        rgb_loss = ((rgb - batch['rgb']).abs() * (0.2 + mask)).mean()
        mask_loss = (alpha - mask).abs().mean()
        offset_jump = model.offset[neighbors[:, 0]] - model.offset[neighbors[:, 1]]
        regularization = (model.offset.square().mean() + 0.5 * offset_jump.square().mean()
                          + 0.00005 * (model.log_scale - initial_log_scale).square().mean())
        loss = rgb_loss + 0.4 * mask_loss + 0.35 * regularization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            model.offset.clamp_(-0.30, 0.30)
            model.log_scale.clamp_(math.log(0.0004), math.log(0.045))
        if step == 1 or step % 25 == 0 or step == args.steps:
            row = {'step': step, 'loss': loss.item(), 'rgb_loss': rgb_loss.item(), 'mask_loss': mask_loss.item()}
            if not np.isfinite(row['loss']):
                raise RuntimeError(f'Nonfinite loss at step {step}')
            history.append(row)
            if step == 1 or step % 100 == 0 or step == args.steps:
                print(json.dumps(row), flush=True)
    seconds = time.monotonic() - started
    scores = {name: measure(model, surfaces, views, pairs, size) for name, pairs in groups.items()}
    audit['train_camera_time_pairs_visited'] = len(fit_coverage)
    audit['offset_quantiles_m'] = dict(zip(('p50', 'p95', 'max'),
        torch.quantile(model.offset.detach().norm(dim=-1), model.offset.new_tensor([0.5, 0.95, 1.0])).tolist()))
    summary = dict(dataset='MVHumanNet++', split=args.split, outfit=args.outfit,
                   atlas_size=args.atlas_size, valid_texels=len(atlas.pixels), image_size=list(size),
                   train_cameras=train_cameras, held_cameras=held_cameras, train_times=train_times,
                   held_times=held_times, video_times=video_times, seed=args.seed, steps=args.steps,
                   views_per_step=args.views_per_step, fit_seconds=seconds, slurm_job=os.getenv('SLURM_JOB_ID'),
                   gpu=torch.cuda.get_device_name(), beta_drift_l2=beta_drift,
                   pose_residuals=False, audit=audit, initial=baseline, final=scores, history=history,
                   asset_sha256=file_sha256(args.asset), protocol_sha256=file_sha256(args.protocol))
    save_map(args.output, atlas, model, betas[:1], bind_pose)
    atomic_json(args.output / 'metrics.json', summary)
    print(json.dumps({name: {k: v for k, v in metrics.items() if k != 'views'} for name, metrics in scores.items()}), flush=True)
    export_media(args.output, args.outfit, atlas, model, surfaces, views, train_times, held_times,
                 video_times, held_cameras, size, render, subset)


@torch.no_grad()
def estimate_scales(asset_path, atlas, vertices):
    with np.load(asset_path, allow_pickle=False) as asset:
        corners = torch.from_numpy(asset['vt'][asset['ft']][atlas.face_ids.cpu().numpy()]).to(vertices)
    xy = corners[:, 1:] - corners[:, :1]
    uv_area = (xy[:, 0, 0] * xy[:, 1, 1] - xy[:, 0, 1] * xy[:, 1, 0]).abs().clamp_min(1e-9)
    mesh = vertices[atlas.triangles]
    world_area = torch.linalg.cross(mesh[:, 1] - mesh[:, 0], mesh[:, 2] - mesh[:, 0]).norm(dim=-1)
    resolution = getattr(atlas, 'resolution_per_texel', atlas.size)
    scale = (world_area / uv_area).sqrt() / resolution
    return (scale * 0.78).clamp(0.0007, 0.016)[:, None].expand(-1, 3).contiguous()


def adjacent_texels(atlas, anchors):
    indices = torch.full_like(atlas.valid, -1, dtype=torch.long)
    indices[atlas.valid] = torch.arange(len(anchors), device=anchors.device)
    x = torch.stack([indices[:, :-1], indices[:, 1:]], dim=-1).flatten(0, 1)
    y = torch.stack([indices[:-1], indices[1:]], dim=-1).flatten(0, 1)
    pairs = torch.cat([x, y])
    pairs = pairs[(pairs >= 0).all(-1)]
    return pairs[(anchors[pairs[:, 0]] - anchors[pairs[:, 1]]).norm(dim=-1) < 0.025]


@torch.no_grad()
def save_map(output, atlas, model, betas, bind_pose):
    fields = {key: atlas.to_map(value).cpu().numpy() for key, value in model.fields().items()}
    np.savez_compressed(output / 'canonical_uv.npz', **fields, valid=atlas.valid.cpu().numpy(),
                        texel_face=atlas.face_ids.cpu().numpy(), texel_barycentric=atlas.barycentric.cpu().numpy(),
                        canonical_anchors_m=model.anchors.cpu().numpy(), betas=betas.cpu().numpy(),
                        bind_pose_axis_angle=bind_pose.cpu().numpy())
    torch.save(model.state_dict(), output / 'avatar.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-extension', type=Path, help='Directory of a prebuilt gsplat_cuda extension')
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--segmentation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--outfit', default='203381')
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--atlas-size', type=int, default=256)
    parser.add_argument('--height', type=int, default=416)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--views-per-step', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--seed', type=int, default=20261123)
    args = parser.parse_args()
    if not os.getenv('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Run inside a Slurm allocation with one visible GPU')
    if args.steps < 1 or min(args.height, args.width, args.atlas_size) < 2:
        parser.error('Supply positive steps and resolutions of at least two pixels')
    args.output.mkdir(parents=True, exist_ok=False)
    fit(args)


if __name__ == '__main__':
    main()
