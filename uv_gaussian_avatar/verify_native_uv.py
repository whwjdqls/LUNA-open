"""Verify native UV geometry and run a short, real MVHumanNet++ optimization."""

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from luna_open.official_mvh import atomic_json

from .fit import load_rasterizer
from .native_islands import resolve_uv_arguments
from .train_wan import check_gradients
from .train_wan_mvh import Experiment, training_parser
from .visualize import as_rgb, panel
from .wan_rope import ExplicitWanRoPE


def array_sha256(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def verify_layout(experiment, legacy_path):
    atlas, layout = experiment.atlas, experiment.model.native_layout
    legacy = json.loads(legacy_path.read_text())
    assert len(atlas.face_ids) == legacy['gaussians']
    assert atlas.island_stats == legacy['islands']
    assert all(array_sha256(getattr(atlas, key)) == digest for key, digest in legacy['hashes'].items())
    arrays = {
        name: getattr(layout, name).cpu().numpy()
        for name in ('global_uv', 'local_uv', 'canonical_xyz', 'patch_row_col', 'island_ids')
    }
    arrays.update(token_offsets=layout.token_offsets, gaussian_offsets=layout.gaussian_offsets,
                  rope_coordinates=experiment.model.token_rope_coordinates.cpu().numpy())
    for i, (island, old, report) in enumerate(zip(layout.islands, atlas.islands, layout.stats, strict=True)):
        width = island.dense_shape[1]
        y, x = island.dense_indices // width, island.dense_indices % width
        old_width = old.coarse_mask.shape[-1] * 8
        torch.testing.assert_close(old.indices, y * old_width + x, rtol=0, atol=0)
        assert len(island.dense_indices.unique()) == report['gaussians']
        local_slots = (y // layout.patch_size) * island.patch_shape[1] + (x // layout.patch_size)
        torch.testing.assert_close(island.patch_indices[island.dense_to_patch], local_slots, rtol=0, atol=0)
        assert island.valid_counts.sum() == report['gaussians'] and island.valid_counts.min() > 0
        for name in ('patch_indices', 'dense_indices', 'dense_to_patch', 'valid_counts'):
            arrays[f'island_{i}_{name}'] = getattr(island, name).cpu().numpy()
    np.savez_compressed(experiment.args.output / 'native-uv-layout.npz', **arrays)
    write_patch_board(layout.stats, arrays, layout.patch_size, experiment.args.output / 'native-uv-patches.png')
    atomic_json(experiment.args.output / 'native-uv-islands.json', dict(
        gaussians=len(atlas.face_ids), tokens=layout.num_tokens, patch_size=layout.patch_size,
        islands=layout.stats, original_atlas_preserved=legacy['hashes'],
    ))


def write_patch_board(islands, arrays, patch_size, output):
    tiles = []
    for i, island in enumerate(islands):
        label = f'{i:02}  {island["region"]}  ·  {island["valid_tokens"]} tokens'
        tiles.append((label, patch_image(island, patch_size, arrays[f'island_{i}_patch_indices'],
                                         arrays[f'island_{i}_dense_indices'])))
    panel([tiles[i:i+4] for i in range(0, len(tiles), 4)], title='Native UV patches by island',
          subtitle='Teal · Gaussian texels   Light blue · remainder of valid patches   Gray · empty patches').save(output)


def patch_image(island, patch_size, patch_indices, dense_indices):
    occupied = np.zeros(island['coarse_hw'], dtype=bool)
    occupied.flat[patch_indices] = True
    grid = np.repeat(np.repeat(occupied, patch_size, axis=0), patch_size, axis=1)
    rgb = np.full((*island['padded_hw'], 3), (218, 223, 232), dtype=np.uint8)
    rgb[grid] = (173, 212, 238)
    rgb.reshape(-1, 3)[dense_indices] = (20, 131, 124)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for x in range(0, width, patch_size):
        draw.line((x, 0, x, height - 1), fill=(80, 99, 115))
    for y in range(0, height, patch_size):
        draw.line((0, y, width - 1, y), fill=(80, 99, 115))
    cell_width, cell_height = 250, 195
    scale = min(cell_width / width, cell_height / height, 3)
    chart = img.resize((round(width * scale), round(height * scale)), Image.Resampling.NEAREST)
    cell = Image.new('RGB', (cell_width, cell_height), '#f6f5f2')
    cell.paste(chart, (0, 0))
    return np.asarray(cell)


@torch.no_grad()
def verify_forward(experiment, image):
    model, layout = experiment.model, experiment.model.native_layout
    latent = model.encode_image(image)
    side = model.image_side
    assert latent.shape == (48, 1, side * 2, side * 2)
    image_tokens = side * side
    length = image_tokens + layout.num_tokens
    backbone = model.backbone
    ops = sys.modules[type(backbone).__module__]
    q = torch.randn(1, length, backbone.num_heads, backbone.dim // backbone.num_heads, device=latent.device)
    explicit = ExplicitWanRoPE(model.token_rope_phases)
    legacy = ops.rope_apply(q[:, :image_tokens], torch.tensor([[1, side, side]]), backbone.freqs)
    torch.testing.assert_close(ops.rope_apply(q, explicit, backbone.freqs)[:, :image_tokens],
                               legacy, rtol=0, atol=0)
    assert (model.token_rope_coordinates[:image_tokens, 0] == 0).all()
    assert (model.token_rope_coordinates[image_tokens:, 0] == 1).all()
    observed = []

    def record(_, inputs):
        x, modulation, _, rotary, _, context, _ = inputs
        assert isinstance(rotary, ExplicitWanRoPE)
        assert modulation.shape == (1, length, 6, backbone.dim)
        assert context.shape == (1, image_tokens, backbone.dim)
        observed.append(tuple(x.shape))

    handles = [block.register_forward_pre_hook(record) for block in backbone.blocks]
    try:
        with patch('torch.nn.functional.grid_sample', side_effect=AssertionError('global-grid interpolation')):
            result = model(latent)
    finally:
        for handle in handles:
            handle.remove()
    assert observed == [(1, length, backbone.dim)] * len(backbone.blocks)
    widths = dict(offset=3, quaternion=4, log_scale=3, opacity_logit=1, rgb=3)
    for name, width in widths.items():
        assert result[name].shape == (len(experiment.atlas.face_ids), width)
        assert result[name].isfinite().all()
    return dict(image_tokens=image_tokens, uv_tokens=layout.num_tokens, sequence_length=length,
                hidden=backbone.dim, blocks_checked=len(observed), image_rope_bitwise_match=True,
                no_global_grid_sampling=True)


@torch.no_grad()
def attribute_previews(experiment, image):
    atlas, model = experiment.atlas, experiment.model
    result = model.predict_image(image)

    def mapped(values):
        rgb = atlas.to_map(values).clone()
        rgb[~atlas.valid] = 0.85
        return as_rgb(rgb.clamp(0, 1).cpu().numpy())

    quat = result['quaternion']
    log_scale = result['log_scale']
    low, high = math.log(0.0004), math.log(0.045)
    panel([
        [('RGB', mapped(result['rgb'])),
         ('Position offset XYZ', mapped(result['offset'] / 0.30 + 0.5)),
         ('Quaternion XYZ', mapped(quat[:, 1:] * 0.5 + 0.5))],
        [('Quaternion W', mapped((quat[:, :1] * 0.5 + 0.5).expand(-1, 3))),
         ('Log scale XYZ', mapped((log_scale - low) / (high - low))),
         ('Opacity', mapped(result['opacity_logit'].sigmoid().expand(-1, 3)))],
    ], title='Decoded Gaussian attributes', subtitle='Native island predictions displayed in SMPL-X UV space',
       thumbnail=(280, 260)).save(experiment.args.output / 'native-uv-attributes.png')


def main():
    parser = training_parser()
    parser.add_argument('--legacy-anchor-receipt', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int, default=4)
    args = resolve_uv_arguments(parser, parser.parse_args())
    if args.uv_representation != 'native-islands' or args.fork_checkpoint or args.smoke_steps < 2:
        parser.error('The gate requires native islands from scratch and at least two smoke updates')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one allocated GPU')
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(6)
    cv2.setNumThreads(1)
    load_rasterizer(args.cuda_extension)
    experiment = Experiment(args, cfg)
    model = experiment.model
    try:
        verify_layout(experiment, args.legacy_anchor_receipt)
        outfit = experiment.data['val'].ids[0]
        selected = experiment.fixed[outfit]['references'][0]
        image = experiment.fixed_reference(outfit, selected)['body'].cuda()
        sequence = verify_forward(experiment, image)
        print(json.dumps(sequence), flush=True)
        parameters = {name: param for name, param in model.named_parameters() if param.requires_grad}
        shared = {name: array_sha256(p) for name, p in parameters.items() if name != 'token_xyz_projection.weight'}
        start_queries = model.queries.detach().clone()
        settings = experiment.train_settings
        optimizer = torch.optim.AdamW(parameters.values(), lr=settings['learning_rate'],
                                      betas=settings['betas'], weight_decay=settings['weight_decay'], fused=True)
        updates = []
        for step in range(1, args.smoke_steps + 1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            terms, renders, choices = experiment.train_update(random.Random(cfg['seed']))
            grads = check_gradients(model, step)
            norm = torch.nn.utils.clip_grad_norm_(parameters.values(), settings['gradient_clip'],
                                                  error_if_nonfinite=True).item()
            optimizer.step()
            row = dict(step=step, total=sum(experiment.weights[k] * v for k, v in terms.items()),
                       terms=terms, renders=renders, gradients=grads, gradient_norm=norm,
                       seconds=time.monotonic() - tick)
            assert math.isfinite(row['total']) and renders == 64
            updates.append(row)
            print(json.dumps(row), flush=True)
        assert not torch.equal(model.queries, start_queries)
        attribute_previews(experiment, image)
        report = dict(passed=True, sequence=sequence, updates=updates,
                      outfit_choices=choices, provenance=experiment.provenance,
                      peak_memory_mib=torch.cuda.max_memory_allocated() // 2**20,
                      shared_initial_weights=shared,
                      trainable_parameters=sum(p.numel() for p in parameters.values()),
                      decoder_parameters=sum(p.numel() for p in model.head.parameters()),
                      parameter_counts=dict(queries=model.queries.numel(),
                                            lora=sum(p.numel() for n, p in parameters.items() if '.lora_' in n)))
        atomic_json(args.output / 'verification.json', report)
    finally:
        experiment.pool.shutdown()


if __name__ == '__main__':
    main()
