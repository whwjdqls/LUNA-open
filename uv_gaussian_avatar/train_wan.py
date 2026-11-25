"""Train the single-reference Wan-to-UV adapter using multiview render losses."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from luna_open.identity_rgb import build_identity_train_perceptual, identity_rgb_terms
from luna_open.identity_v2_body import VertexSMPLX
from luna_open.identity_v2_data import IdentityData
from luna_open.official_mvh import atomic_json
from luna_open.provenance import file_sha256

from .adaptive_atlas import AdaptiveAtlas
from .fit import (
    estimate_scales,
    load_rasterizer,
    render,
    stack_views,
    vertex_transforms,
)
from .native_islands import NativeUVLayout, add_uv_arguments, resolve_uv_arguments
from .representation import Atlas, PosedSurface
from .visualize import as_rgb, panel
from .wan_avatar import WanUVAvatar, animate_prediction, load_wan


@torch.no_grad()
def prepare_outfit(data, outfit, model, body, atlas):
    times = sorted(data.document['outfits'][outfit]['times'])
    if len(times) < 12:
        raise ValueError(f'At least 12 capture times are required for {outfit}')
    ref_stamp = times[0]
    train_times = [times[i] for i in (1, 3, 5, 7, 9)]
    held_times = [times[i] for i in (4, 10)]
    reference = data.target(outfit, 0, ref_stamp)['rgb'].cuda()
    latent = model.encode_image(reference)
    reference_body = body.records([data.body(outfit, ref_stamp)], 'cuda')
    beta = reference_body['betas']
    bind_pose = torch.zeros_like(reference_body['pose'])
    canonical, bind = body.mesh(bind_pose, beta, torch.zeros_like(reference_body['expression']))
    stamps = train_times + held_times
    motion = body.records([data.body(outfit, stamp) for stamp in stamps], 'cuda')
    vertices, transforms = body.mesh(motion['pose'], beta.expand(len(stamps), -1), motion['expression'])
    surfaces = atlas.pose(vertices + motion['transl'][:, None], vertex_transforms(body, transforms, bind))
    anchors = atlas.sample(canonical[0])
    frames = {}
    for i, stamp in enumerate(stamps):
        cameras = (0, 4, 8, 12) if stamp in train_times else (1, 7, 13)
        frames[stamp] = (PosedSurface(surfaces.anchors[i], surfaces.rotation[i]),
                         stack_views([data.target(outfit, camera, stamp) for camera in cameras]))
    return dict(outfit=outfit, reference=reference, reference_time=ref_stamp, latent=latent,
                anchors=anchors, betas=beta,
                frames=frames, train_times=train_times, held_times=held_times)


def lhm_rgb_losses(fields, surface, target, size, perceptual, settings):
    rgb, _ = render(animate_prediction(fields, surface), target, size)
    terms = identity_rgb_terms(rgb.permute(0, 3, 1, 2), target['rgb'].permute(0, 3, 1, 2),
                               target['mask'].permute(0, 3, 1, 2), perceptual)
    return sum(value * settings[key] for key, value in terms.items()), terms


@torch.no_grad()
def evaluate(model, atlas, outfits, size, output=None):
    results = {}
    for item in outfits:
        fields = model(item['latent'])
        psnr, foreground_psnr, iou = [], [], []
        for stamp in item['held_times']:
            surface, target = item['frames'][stamp]
            rgb, alpha = render(animate_prediction(fields, surface), target, size)
            error = (rgb - target['rgb']).square()
            mask = target['mask'] > 0.5
            psnr.extend((-10 * error.mean((1, 2, 3)).clamp_min(1e-12).log10()).tolist())
            fg = (error * mask).sum((1, 2, 3)) / (3 * mask.sum((1, 2, 3))).clamp_min(1)
            foreground_psnr.extend((-10 * fg.clamp_min(1e-12).log10()).tolist())
            iou.extend(((mask & (alpha > 0.5)).sum((1, 2, 3)) /
                        (mask | (alpha > 0.5)).sum((1, 2, 3)).clamp_min(1)).tolist())
        outfit = item['outfit']
        results[outfit] = dict(psnr=float(np.mean(psnr)), foreground_psnr=float(np.mean(foreground_psnr)),
                               mask_iou=float(np.mean(iou)), count=len(psnr))
        if output is not None:
            labels = ('01', '07', '13')
            tiled = atlas.to_map(fields['rgb'])
            tiled[~atlas.valid] = 0.85
            panel([
                [('Single reference', as_rgb(item['reference'].permute(1, 2, 0).cpu().numpy())),
                 ('Predicted UV color', as_rgb(tiled.cpu().numpy()))],
                [(f'Captured {c}', as_rgb(im)) for c, im in zip(labels, target['rgb'].cpu().numpy())],
                [(f'Predicted {c}', as_rgb(im)) for c, im in zip(labels, rgb.cpu().numpy())],
            ], title=f'Wan2.2 UV prediction · outfit {outfit}',
                subtitle=f'One input image · held-out cameras and pose at timestamp {stamp}',
                thumbnail=(205, 235)).save(output / f'outfit-{outfit}.png')
            map_fields = {label: atlas.to_map(fields[key]).cpu().numpy() for label, key in (
                ('offset_m', 'offset'), ('quaternion_wxyz', 'quaternion'), ('log_scale_m', 'log_scale'),
                ('opacity_logit', 'opacity_logit'), ('rgb', 'rgb'))}
            np.savez_compressed(output / f'outfit-{outfit}-canonical.npz', **map_fields,
                                valid=atlas.valid.cpu().numpy(), texel_face=atlas.face_ids.cpu().numpy(),
                                texel_barycentric=atlas.barycentric.cpu().numpy(),
                                canonical_anchors_m=item['anchors'].cpu().numpy(), betas=item['betas'].cpu().numpy(),
                                bind_pose_axis_angle=np.zeros((1, 165), dtype=np.float32))
    return results


def check_gradients(model, step):
    missing, zero, nonfinite = [], [], []
    names, norms = [], []
    suffix = '.lora_B' if step == 1 else '.lora_A'
    for name, p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None:
                raise RuntimeError(f'Frozen parameter received a gradient: {name}')
            continue
        if p.grad is None:
            missing.append(name)
        elif not name.endswith(('.lora_A', '.lora_B')) or name.endswith(suffix):
            names.append(name)
            norms.append(p.grad.detach().float().square().sum().sqrt())
    for name, norm in zip(names, torch.stack(norms).tolist()):
        if not np.isfinite(norm):
            nonfinite.append(name)
        elif norm == 0:
            zero.append(name)
    if missing or zero or nonfinite:
        raise RuntimeError(f'Gradient failure: missing {missing}, zero {zero}, nonfinite {nonfinite}')
    return dict(stage=step, checked=len(names), adapter_projections=len(model.targets))


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    rng = random.Random(args.seed)
    load_rasterizer(args.cuda_extension)
    print('Loading Wan2.2 TI2V and its frozen VAE', flush=True)
    backbone, vae, embedding = load_wan(args.wan_source, args.wan_checkpoint, args.wan_deps)
    if args.adaptive_decoder:
        atlas = AdaptiveAtlas.from_smplx(args.asset, args.atlas_size, args.segmentation,
                                       args.adaptive_body_scale).cuda()
    else:
        atlas = Atlas.from_smplx(args.asset, args.atlas_size).cuda()
    body = VertexSMPLX(args.asset, args.segmentation).cuda()
    neutral, _ = body.mesh(torch.zeros_like(body.canonical_pose),
                           body.anchors.new_zeros((1, 10)), body.anchors.new_zeros((1, 10)))
    base_log_scale = estimate_scales(args.asset, atlas, neutral[0]).log()
    native = args.uv_representation == 'native-islands'
    layout = NativeUVLayout(atlas, neutral[0], args.uv_patch_size) if native else None
    model = WanUVAvatar(backbone, vae, embedding, atlas, base_log_scale,
                        side=args.query_side, rank=args.rank, image_side=args.image_side,
                        native_layout=layout, canonical_3d_pos=args.canonical_3d_pos).cuda()
    weights = yaml.safe_load(args.lhm_loss_config.read_text())['loss']
    settings = {key: weights[key] for key in ('rgb', 'lpips')}
    print(f'Using LHM identity RGB losses from {args.lhm_loss_config}', flush=True)
    perceptual = build_identity_train_perceptual()
    size = (args.height, args.width)
    train_data = IdentityData(args.protocol, 'train', render_size=size)
    val_data = IdentityData(args.protocol, 'val', render_size=size)
    train_ids = args.train_ids or ['203381', *rng.sample(sorted(set(train_data.ids) - {'203381'}), 3)]
    val_ids = args.val_ids or rng.sample(sorted(val_data.ids), 2)
    if (not set(train_ids) <= set(train_data.ids) or not set(val_ids) <= set(val_data.ids)
            or len(set(train_ids + val_ids)) != len(train_ids + val_ids)):
        raise ValueError('Select distinct outfits from the declared split')
    outfits = []
    for data, ids in ((train_data, train_ids), (val_data, val_ids)):
        for outfit in ids:
            outfits.append(prepare_outfit(data, outfit, model, body, atlas))
            print(f'Encoded single reference for {outfit}', flush=True)

    initial = evaluate(model, atlas, outfits, size)
    adapter = [p for n, p in model.named_parameters() if p.requires_grad and '.lora_' in n]
    projections = [p for n, p in model.named_parameters() if p.requires_grad and '.lora_' not in n]
    optimizer = torch.optim.AdamW([dict(params=adapter, lr=args.lora_lr),
                                  dict(params=projections, lr=args.projection_lr)], fused=True)
    audit, history = [], []
    started = time.monotonic()
    schedule = []
    for step in range(1, args.steps + 1):
        if not schedule:
            schedule = [(item, stamp) for item in outfits[:len(train_ids)] for stamp in item['train_times']]
            rng.shuffle(schedule)
        item, stamp = schedule.pop()
        surface, target = item['frames'][stamp]
        fields = model(item['latent'])
        loss, terms = lhm_rgb_losses(fields, surface, target, size, perceptual, settings)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if step <= 2:
            audit.append(check_gradients(model, step))
        torch.nn.utils.clip_grad_norm_(adapter + projections, 1, error_if_nonfinite=True)
        optimizer.step()
        if step <= 2 or step % 20 == 0 or step == args.steps:
            row = dict(step=step, loss=loss.item(), **{key: value.item() for key, value in terms.items()})
            history.append(row)
            print(json.dumps(row), flush=True)
    fit_seconds = time.monotonic() - started
    metrics = evaluate(model, atlas, outfits, size, args.output)
    state_path = args.output / 'trainable.pt'
    torch.save(model.trainable_state(), state_path)
    with torch.no_grad():
        old = model(outfits[-1]['latent'])
        model.load_trainable_state(torch.load(state_path, weights_only=True))
        new = model.predict_image(outfits[-1]['reference'])
        for key in old:
            torch.testing.assert_close(old[key], new[key], atol=0, rtol=0)
        changed = (model(outfits[-2]['latent'])['rgb'] - new['rgb']).abs().mean().item()
        if not np.isfinite(changed) or changed <= 0:
            raise RuntimeError('The fitted UV RGB did not respond to a changed reference')
    report = dict(steps=args.steps, train_outfits=train_ids, held_out_outfits=val_ids, atlas_size=atlas.size,
                  gaussians=len(atlas.face_ids), query_side=args.query_side, rank=args.rank,
                  wan_blocks=len(backbone.blocks), adapter_targets=model.targets,
                  adapter_parameters=sum(p.numel() for p in adapter), projection_parameters=sum(p.numel() for p in projections),
                  reference_count=1, shape='supplied reference betas for render only',
                  scales='fixed neutral template prior plus learned residual', query_timestep=model.query_timestep,
                  loss_config=str(args.lhm_loss_config), loss_config_sha256=file_sha256(args.lhm_loss_config),
                  loss_weights=settings,
                  gradient_checks=audit, initial=initial, final=metrics, history=history,
                  rgb_change_with_swapped_reference=changed, state_replay='exact UV tensors',
                  checkpoint_sha256=file_sha256(state_path), fit_seconds=fit_seconds,
                  peak_cuda_mib=torch.cuda.max_memory_allocated() // (1024**2),
                  gpu=torch.cuda.get_device_name(), slurm_job=os.getenv('SLURM_JOB_ID'),
                  protocol_sha256=file_sha256(args.protocol), smplx_sha256=file_sha256(args.asset))
    if native:
        report.update(uv_patch_size=layout.patch_size, uv_tokens=layout.num_tokens,
                      image_tokens=model.image_side**2, native_layout=layout.stats,
                      canonical_3d_pos=args.canonical_3d_pos)
    atomic_json(args.output / 'metrics.json', report)
    print(json.dumps({'initial': initial, 'final': metrics, 'fit_seconds': fit_seconds}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('wan-source', 'wan-checkpoint', 'protocol', 'asset', 'segmentation', 'output'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    for key in ('wan-deps', 'cuda-extension'):
        parser.add_argument(f'--{key}', type=Path)
    parser.add_argument('--lhm-loss-config', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'configs' / 'lunav2_identity.yaml')
    for key, default in (('steps', 160), ('atlas-size', 256), ('query-side', 16), ('rank', 8),
                         ('height', 256), ('width', 224), ('seed', 20261124)):
        parser.add_argument(f'--{key}', type=int, default=default)
    parser.add_argument('--lora-lr', type=float, default=0.0001)
    parser.add_argument('--projection-lr', type=float, default=0.0003)
    parser.add_argument('--train-ids', nargs='+')
    parser.add_argument('--val-ids', nargs='+')
    add_uv_arguments(parser)
    args = resolve_uv_arguments(parser, parser.parse_args())
    if not os.getenv('SLURM_JOB_ID') or torch.cuda.device_count() != 1:
        raise RuntimeError('Run inside a Slurm step with one visible GPU')
    if min(args.steps, args.atlas_size, args.query_side, args.rank, args.height, args.width) < 1:
        parser.error('Steps and dimensions must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    train(args)


if __name__ == '__main__':
    main()
