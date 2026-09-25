import pytest
import torch

from luna_open.avatar import CanonicalAvatar, Gaussians
from luna_open.geometry import quaternion_to_matrix
from luna_open.losses import projection_loss, quaternion_loss, structural_loss
from luna_open.model import IdentityEncoder, ModelConfig, NeuralAnimator


def config():
    return ModelConfig(
        width=32, depth=1, heads=4, decoder_width=32, body_dim=8, face_dim=8, motion_dim=8
    )


@pytest.mark.parametrize("autocast_enabled", [False, True])
def test_animation_initialization_and_fixed_attributes(autocast_enabled):
    torch.manual_seed(3)
    cfg = config()
    encoder = IdentityEncoder(torch.randn(16, 3), torch.arange(16) % 24, cfg)
    animator = NeuralAnimator(16, cfg, torch.tensor([0.0, 0.0, 3.0]), torch.ones(3))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast_enabled):
        avatar = encoder(torch.randn(2, 4, 4, 8), torch.randn(2, 4, 2, 8))
        posed = animator(avatar, torch.randn(2, 5, 8))
    posed.gaussians.validate()
    torch.testing.assert_close(
        posed.gaussians.means, avatar.gaussians.means + torch.tensor([0.0, 0.0, 3.0])
    )
    torch.testing.assert_close(
        quaternion_to_matrix(posed.global_quaternion), torch.eye(3).expand(2, 3, 3)
    )
    assert posed.gaussians.scales is avatar.gaussians.scales
    assert posed.gaussians.opacities is avatar.gaussians.opacities


def test_teacher_loss_detaches_and_masks_samples():
    means = torch.zeros(2, 4, 3, requires_grad=True)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 4, 4)
    teacher_means = torch.ones_like(means, requires_grad=True)
    with torch.no_grad():
        teacher_means[1] = float("nan")
    pred = Gaussians(means, q, torch.ones_like(means), torch.ones(2, 4), means)
    target = Gaussians(teacher_means, -q, torch.ones_like(means), torch.ones(2, 4), means.detach())
    loss = structural_loss(pred, target, torch.tensor([True, False]))
    torch.testing.assert_close(loss, torch.tensor(0.5))
    loss.backward()
    assert teacher_means.grad is None
    assert means.grad[0].abs().sum() > 0
    assert means.grad[1].abs().sum() == 0
    assert quaternion_loss(q, -q).sum() == 0


def test_projection_ignores_missing_labels_before_arithmetic():
    prediction = torch.tensor([[[0.2, 0.0, 1.0]], [[0.0, 0.0, 1.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 0.0, 1.0]], [[float("nan"), 0.0, 1.0]]])
    loss = projection_loss(
        prediction, teacher, torch.eye(3).expand(2, 3, 3), (100, 100), torch.tensor([True, False])
    )
    torch.testing.assert_close(loss, torch.tensor(0.0005))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[1].abs().sum() == 0


def test_global_warmup_preserves_independent_heads_and_skips_local_branch():
    cfg = config()
    n = 8
    zeros = torch.zeros(1, n, 3)
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, n, 4)
    avatar = CanonicalAvatar(
        Gaussians(zeros, q, zeros + 0.01, torch.ones(1, n), zeros), torch.randn(1, n, cfg.width)
    )
    animator = NeuralAnimator(n, cfg, torch.tensor([0.0, 0.0, 3.0]), torch.ones(3))
    output = animator(avatar, torch.randn(1, 5, 8), global_only=True)
    output.translation.sum().backward()
    assert animator.local[-1].weight.grad is None
    assert animator.translation_head[-1].weight.grad is not None
    assert animator.rotation_head[-1].weight.grad is None
    animator.zero_grad(set_to_none=True)
    output = animator(avatar, torch.randn(1, 5, 8), global_only=True)
    output.rotation_sincos[..., 0].sum().backward()
    assert animator.rotation_head[-1].weight.grad is not None
    assert animator.translation_head[-1].weight.grad is None
    assert animator.local[-1].weight.grad is None
