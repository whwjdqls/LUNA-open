import argparse
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from uv_gaussian_avatar.adaptive_atlas import AdaptiveAtlas
from uv_gaussian_avatar.native_islands import (
    IslandPatches,
    NativeUVLayout,
    add_uv_arguments,
    decoder_stages,
    resolve_uv_arguments,
)
from uv_gaussian_avatar.wan_avatar import NativeUVHead
from uv_gaussian_avatar.wan_rope import ExplicitWanRoPE, install_explicit_rope, native_wan_rope


def build_island(patch_size=16):
    mask = torch.zeros((19, 37), dtype=torch.bool)
    mask[:2, :3] = True
    mask[4, 18] = True
    mask[18, 35] = True
    positions = torch.arange(mask.sum() * 3, dtype=torch.float32).reshape(-1, 3)
    return IslandPatches(mask, (7, 9, 44, 28), 96, positions, patch_size), positions


def test_patch_indexing_and_partial_patch_coordinates():
    island, xyz = build_island()
    assert island.shape == (19, 37)
    assert island.dense_shape == (32, 48)
    assert island.patch_shape == (2, 3)
    assert island.patch_indices.tolist() == [0, 1, 5]
    assert island.patch_row_col.tolist() == [[0, 0], [0, 1], [1, 2]]
    assert island.valid_counts.tolist() == [6, 1, 1]
    assert island.dense_indices.tolist() == [0, 1, 2, 48, 49, 50, 210, 899]
    assert island.dense_to_patch.tolist() == [0, 0, 0, 0, 0, 0, 1, 2]
    torch.testing.assert_close(island.canonical_xyz, torch.stack((xyz[:6].mean(0), xyz[6], xyz[7])))
    torch.testing.assert_close(island.global_uv[-1], torch.tensor([42.5 / 96, 1 - 27.5 / 96]))
    torch.testing.assert_close(island.local_uv[-1], torch.tensor([35.5 / 37, 1 - 18.5 / 19]))
    tokens = torch.arange(1., 4.).reshape(1, 3, 1)
    torch.testing.assert_close(island.scatter(tokens), torch.tensor([[[[1., 2., 0.], [0., 0., 3.]]]]))
    dense = torch.arange(32 * 48).reshape(1, 1, 32, 48)
    assert island.gather(dense).flatten().tolist() == island.dense_indices.tolist()


@pytest.mark.parametrize('patch_size,widths', [(8, [256, 128, 64]), (16, [256, 128, 64, 32])])
def test_native_cnn_shapes_gradients_and_chart_isolation(patch_size, widths):
    assert decoder_stages(patch_size) == widths
    first, _ = build_island(patch_size)
    mask = torch.ones(7, 3, dtype=torch.bool)
    second = IslandPatches(mask, (30, 2, 33, 9), 96, torch.zeros(21, 3), patch_size)
    n1, n2 = len(first.patch_indices), len(second.patch_indices)
    layout = SimpleNamespace(islands=[first, second], token_offsets=[0, n1, n1 + n2])
    head = NativeUVHead(12, patch_size)
    tokens = torch.randn(2, n1 + n2, 12, requires_grad=True)
    with patch('torch.nn.functional.grid_sample', side_effect=AssertionError('global grid sampling')):
        out = head(tokens, layout)
        assert out.shape == (2, len(first.dense_indices) + len(second.dense_indices), 14)
        new = head(torch.cat((tokens[:, :n1], tokens[:, n1:] + 3), dim=1), layout)
    torch.testing.assert_close(new[:, :len(first.dense_indices)], out[:, :len(first.dense_indices)])
    out.square().mean().backward()
    assert tokens.grad.isfinite().all() and tokens.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.isfinite().all() for p in head.parameters())


def frequencies(length=8, head_dim=12):
    c = head_dim // 2
    sizes = (c - 2 * (c // 3), c // 3, c // 3)
    pos = torch.arange(length, dtype=torch.float64)
    phases = [torch.outer(pos, 10000 ** (-torch.arange(n, dtype=torch.float64) / n)) for n in sizes]
    return torch.polar(torch.ones((length, c), dtype=torch.float64), torch.cat(phases, dim=-1))


def test_mixed_rope_uses_original_image_positions_and_fractional_uv_coordinates():
    freqs = frequencies()
    uv = torch.tensor([[.25, .75], [.25, .75], [.8, .2]])
    coords, phases = native_wan_rope(freqs, 4, uv)
    assert coords.shape == (19, 3)
    assert coords[0].tolist() == [0, 0, 0]
    assert coords[15].tolist() == [0, 3, 3]
    assert coords[16].tolist() == [1, .75, .75]
    torch.testing.assert_close(phases[16], phases[17], rtol=0, atol=0)
    assert not torch.equal(phases[16], phases[18])
    spac = freqs.shape[1] // 3
    expected = torch.stack([
        torch.cat((freqs[0, :spac], freqs[y, spac:2 * spac], freqs[x, 2 * spac:]))
        for y in range(4) for x in range(4)
    ])
    torch.testing.assert_close(phases[:16], expected, rtol=0, atol=0)
    torch.testing.assert_close(phases[16, :spac], freqs[1, :spac], rtol=0, atol=0)
    x = torch.randn(2, 19, 3, 12, requires_grad=True)
    rope = ExplicitWanRoPE(phases)
    y = rope.apply(x)
    assert y.shape == x.shape
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))
    y.sum().backward()
    assert x.grad.isfinite().all()

    module = SimpleNamespace(rope_apply=lambda x, grid, freqs: grid)
    install_explicit_rope(module)
    wrapped = module.rope_apply
    install_explicit_rope(module)
    assert module.rope_apply is wrapped
    torch.testing.assert_close(module.rope_apply(x, rope, freqs), y)
    assert module.rope_apply(x, 'legacy', freqs) == 'legacy'


def test_overlapping_charts_keep_distinct_tokens_and_anchors():
    triangle = np.array([[.12, .17], [.81, .19], [.26, .59]], dtype=np.float32)
    uv = np.concatenate((triangle, triangle))
    faces = np.arange(6).reshape(2, 3)
    atlas = AdaptiveAtlas(uv, faces, faces, 24, {}, body_scale=1.2)
    verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    translation = torch.tensor([0., 0., 2.])
    layout = NativeUVLayout(atlas, torch.cat((verts, verts + translation)), patch_size=8)
    assert len(layout.islands) == 2
    middle = layout.token_offsets[1]
    assert middle > 1 and layout.num_tokens == 2 * middle
    assert layout.island_ids.tolist() == [0] * middle + [1] * middle
    torch.testing.assert_close(layout.global_uv[:middle], layout.global_uv[middle:], atol=0, rtol=0)
    torch.testing.assert_close(layout.canonical_xyz[:middle] + translation, layout.canonical_xyz[middle:])
    assert layout.gaussian_offsets[-1] == len(atlas.face_ids)
    assert torch.equal(layout.islands[0].dense_indices, layout.islands[1].dense_indices)


def test_native_patch_size_is_independent_of_image_size():
    parser = argparse.ArgumentParser()
    parser.set_defaults(query_side=16, atlas_size=528, rank=8)
    add_uv_arguments(parser)
    a = resolve_uv_arguments(parser, parser.parse_args(['--uv-representation', 'native-islands']))
    b = resolve_uv_arguments(parser, parser.parse_args(
        ['--uv-representation', 'native-islands', '--uv-patch-size', '8'],
    ))
    assert a.image_side == b.image_side == 16
    assert (a.uv_patch_size, b.uv_patch_size) == (16, 8)
    assert a.adaptive_decoder and b.adaptive_decoder
    assert a.adaptive_body_scale == b.adaptive_body_scale == 1.2
