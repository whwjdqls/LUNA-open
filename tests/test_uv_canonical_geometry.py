from types import SimpleNamespace

import numpy as np
import pytest
import torch

from uv_gaussian_avatar.canonical_geometry import (
    REFERENCE_WEIGHTS,
    SCALE_LOWER,
    SCALE_UPPER,
    CanonicalGeometry,
    activate,
    geometry_weights,
    update_regularizers,
)
from uv_gaussian_avatar.native_islands import IslandPatches
from uv_gaussian_avatar.representation import rasterize_atlas
from uv_gaussian_avatar.wan_avatar import NativeUVHead


def test_rasterized_anchor_uses_ordered_triangle_corners():
    uv = np.array([[0., 1.], [1., 1.], [0., 0.]])
    face = np.array([[0, 1, 2]])
    face_ids, weights = rasterize_atlas(uv, face, 8)
    assert face_ids[1, 2] == 0
    np.testing.assert_allclose(weights[1, 2] @ uv[face[0]], [(2 + .5) / 8, 1 - (1 + .5) / 8])


def fixture():
    # Disjoint UV charts share a physical triangle. A duplicate row of skin
    # samples across the seam has identical barycentrics but distinct tokens.
    mask = torch.ones(5, 5, dtype=torch.bool)
    mask[0, 0] = False
    samples = int(mask.sum())
    a = IslandPatches(mask, (0, 0, 5, 5), 8, torch.zeros(samples, 3), 4)
    b = IslandPatches(mask, (0, 0, 5, 5), 8, torch.zeros(samples, 3), 4)
    n = len(a.valid_counts)
    layout = SimpleNamespace(islands=[a, b], token_offsets=[0, n, n*2], gaussian_offsets=[0, samples, samples*2], num_tokens=n*2)
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [3., 2., 1.]])
    faces = torch.tensor([[0, 1, 2]])
    weights = torch.stack([torch.linspace(.1, .6, samples), torch.full((samples,), .2),
                           torch.linspace(.7, .2, samples)], -1).repeat(2, 1)
    atlas = SimpleNamespace(triangles=faces.expand(samples*2, -1), barycentric=weights)
    return layout, CanonicalGeometry(atlas, layout, vertices, faces)


def test_mesh_lift_weights_interpolation_seam_and_gradients():
    layout, geom = fixture()
    tokens = torch.randn(2, layout.num_tokens, 3, requires_grad=True)
    vertices, anchors = geom.lift(tokens)
    broadcast = torch.cat([tokens[:, off + i.dense_to_patch] for off, i in zip(layout.token_offsets, layout.islands)], 1)
    expected = (geom.barycentric.T @ broadcast) / geom.barycentric.sum(0)[:, None]
    torch.testing.assert_close(vertices[:, :3], expected)
    torch.testing.assert_close(vertices[:, 3], torch.zeros_like(vertices[:, 3]))
    torch.testing.assert_close(anchors, geom.barycentric @ expected)
    torch.testing.assert_close(*anchors.chunk(2, dim=1))
    anchors.sum().backward()
    assert (tokens.grad > 0).all()


def test_neighborhoods_and_sign_invariance():
    layout, g = fixture()
    # Each 5x5 chart has nine interior centers, one rejected by the missing corner.
    assert len(g.neighbors) == 16
    assert ((g.neighbors < 24).all(-1) | (g.neighbors >= 24).all(-1)).all()
    offset = torch.zeros(48, 3)
    offset[24:] = .02
    scale = torch.zeros_like(offset)
    q = torch.zeros(48, 4)
    q[:, 2] = 1
    q[::2] *= -1
    torch.testing.assert_close(g.laplacian(offset, scale, q), torch.zeros(16), atol=1e-12, rtol=0)
    # A spike at the center has a 4x amplitude in the standard 5-point stencil.
    center = g.neighbors[2, 0]
    scale[center, 0] = 1.
    assert g.laplacian(offset, scale, q)[2] == 16


def test_inactive_diagnostics_are_logged_without_gradients():
    _, geom = fixture()
    raw = torch.randn(48, 14, requires_grad=True)
    fields, aux = activate(raw, torch.full((48, 3), -6.))
    weights = dict.fromkeys(REFERENCE_WEIGHTS, 0.)
    weights['offset'] = .1
    terms = update_regularizers(geom, fields, aux, weights)
    assert set(terms) == {'scale', 'smooth', 'normal', 'offset'}
    assert {k for k, value in terms.items() if value.requires_grad} == {'offset'}
    assert terms['scale'].item() > 0
    torch.testing.assert_close(terms['offset'], fields['offset'].square().sum(-1).mean())


def test_zero_uniform_and_equal_outfit_reduction():
    _, g = fixture()
    raw = torch.zeros(2, 48, 14, requires_grad=True)
    tokens = torch.zeros(2, 8, 3, requires_grad=True)
    fields, aux = activate(raw, torch.full((48, 3), -6.), token_offsets=tokens, geometry=g)
    terms = g.losses(fields, aux, REFERENCE_WEIGHTS)
    assert all(float(v.detach()) == pytest.approx(0, abs=1e-12) for v in terms.values())
    sum(terms.values()).backward()
    assert torch.isfinite(raw.grad).all() and torch.isfinite(tokens.grad).all()
    offset = torch.zeros_like(fields['offset'])
    offset[0, 0, 0] = .02
    offset[1, :, 2] = -.03
    normal = g.losses({**fields, 'offset': offset}, aux, ['normal'])['normal']
    assert normal.item() == pytest.approx(.5)
    for sign in (1, -1):
        offset[:, :, 2] = .003 * sign
        offset[:, :, 0] = 0.
        assert g.losses({**fields, 'offset': offset}, aux, ['normal'])['normal'] == 0


def test_relative_scale_gradient_above_physical_cap_and_patch_mean():
    layout, g = fixture()
    raw = torch.zeros(48, 14)
    raw[:, 7:10] = 1.5
    raw.requires_grad_()
    fields, aux = activate(raw, torch.zeros(48, 3))
    assert fields['log_scale'].exp().max().item() == pytest.approx(.045)
    loss = g.losses(fields, aux, ['scale'])['scale']
    loss.backward()
    expected = 3 * (1 - torch.tanh(torch.tensor(1.5)).square()) / (48*3)
    torch.testing.assert_close(raw.grad[:, 7:10], expected.expand(48, 3))
    assert torch.exp(torch.tensor(SCALE_UPPER)).item() == pytest.approx(2 * torch.sigmoid(torch.tensor(3.)).item())
    assert torch.exp(torch.tensor(SCALE_LOWER)).item() == pytest.approx(2 * torch.sigmoid(torch.tensor(-2.)).item())
    fine = torch.randn(48, 3)
    expected = torch.stack([fine[g.parent == j].mean(0) for j in range(layout.num_tokens)])
    actual = g.losses(fields, {**aux, 'fine_offset': fine}, ['pixel'])['pixel']
    torch.testing.assert_close(actual, expected.square().sum(-1).mean())


def test_hierarchical_branches_backpropagate_and_preserve_budget():
    layout, g = fixture()
    head = NativeUVHead(12, 4, coarse_to_fine=True)
    tokens = torch.randn(1, layout.num_tokens, 12, requires_grad=True)
    raw, coarse = head(tokens, layout)
    fields, aux = activate(raw, torch.full((48, 3), -6.), token_offsets=coarse, geometry=g)
    torch.testing.assert_close(fields['offset'], aux['coarse_offset'] + aux['fine_offset'])
    assert fields['offset'].abs().max() <= .15
    fields['offset'].sum().backward()
    for grad in (tokens.grad, head.coarse_head[0].weight.grad, head.coarse_head[-1].weight.grad,
                 head.cnn[-1].weight.grad[:3]):
        assert grad.isfinite().all() and grad.abs().sum() > 0


def test_flags_and_original_activations():
    args = SimpleNamespace(uv_representation='native-islands', canonical_regularization=True)
    assert geometry_weights(args) == REFERENCE_WEIGHTS
    for k in REFERENCE_WEIGHTS:
        setattr(args, f'{k}_weight', 0)
    assert not any(geometry_weights(args).values())
    args.strain_weight = 1.
    with pytest.raises(ValueError):
        geometry_weights(args)
    raw = torch.randn(12, 14)
    prior = torch.full((12, 3), -6.)
    fields, aux = activate(raw, prior)
    torch.testing.assert_close(fields['offset'], raw[:, :3].tanh() * .15, rtol=0, atol=0)
    torch.testing.assert_close(fields['rgb'], raw[:, 11:].sigmoid(), rtol=0, atol=0)


def test_trainer_counts_one_canonical_penalty_per_outfit(monkeypatch):
    import random

    from uv_gaussian_avatar import train_wan_mvh as trainer

    exp = object.__new__(trainer.Experiment)
    exp.train_settings = dict(outfits_per_gpu=4, target_frames=4, target_cameras=4)
    exp.weights = {'rgb': .8}
    exp.reg_weights = {'offset': .1}
    exp.loss_weights = {**exp.weights, **exp.reg_weights}
    exp.data = {'train': SimpleNamespace(ids=[1, 2, 3, 4])}
    exp.size, exp.front, exp.detector, exp.perceptual = (), None, None, None
    theta = torch.tensor(1., requires_grad=True)

    class ToyPredictor:
        def encode_image(self, image):
            return image

        def __call__(self, image, *, return_aux):
            return {'offset': theta*image}, {}

    exp.model = ToyPredictor()
    calls = []

    def regularizers(fields, aux, active):
        calls.append(fields['offset'].item())
        return {'offset': fields['offset'].square()}

    exp.geometry = SimpleNamespace(losses=regularizers)
    monkeypatch.setattr(trainer, 'sample_references', lambda data, outfit, *args: [dict(observation=(0, outfit))])
    monkeypatch.setattr(trainer, 'sample_targets', lambda data, outfit, *args:
                        (list(range(outfit)), [(0, t) for t in range(outfit)]))
    exp.load_outfit = lambda split, outfit, ref, stamps, targets: (
        outfit, SimpleNamespace(anchors=stamps, rotation=stamps), [dict(observation=t) for t in targets])
    monkeypatch.setattr(trainer, 'animate_prediction', lambda f, s: f)
    monkeypatch.setattr(trainer, 'stack_views', lambda x: x)
    monkeypatch.setattr(trainer, 'render', lambda f, target, size: (f['offset'][None], 0))
    monkeypatch.setattr(trainer, 'render_terms', lambda rgb, *args: {'rgb': rgb.mean()})
    terms, renders, _ = exp.train_update(random.Random(0))
    assert renders == 10 and sorted(calls) == [1, 2, 3, 4]
    assert terms == pytest.approx(dict(rgb=2.5, offset=7.5))
    assert theta.grad.item() == pytest.approx(.8*2.5 + .1*2*7.5)
