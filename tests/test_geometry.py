import math

import numpy as np
import torch

from luna_open.data.neuman import crop_affine
from luna_open.geometry import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    matrix_to_sincos,
    project_points,
    proper_rotation,
    quaternion_multiply,
    quaternion_to_matrix,
    sincos_to_quaternion,
)


def test_quaternion_composition_and_half_turns():
    axes = torch.eye(3) * math.pi
    q = axis_angle_to_quaternion(axes)
    r = quaternion_to_matrix(q)
    torch.testing.assert_close(quaternion_to_matrix(matrix_to_quaternion(r)), r)
    a = axis_angle_to_quaternion(torch.tensor([0.3, -0.8, 1.0]))
    b = axis_angle_to_quaternion(torch.tensor([0.9, 0.1, -0.3]))
    torch.testing.assert_close(
        quaternion_to_matrix(quaternion_multiply(a, b)),
        quaternion_to_matrix(a) @ quaternion_to_matrix(b),
    )


def test_euler_targets_round_trip_and_gimbal_lock():
    angles = torch.tensor([[0.2, 0.4, -0.6], [0.3, math.pi / 2, 0.7], [0.3, -math.pi / 2, 0.7]])
    pairs = torch.stack((angles.sin(), angles.cos()), -1)
    rotation = quaternion_to_matrix(sincos_to_quaternion(pairs))
    recovered = quaternion_to_matrix(sincos_to_quaternion(matrix_to_sincos(rotation)))
    torch.testing.assert_close(recovered, rotation, atol=1e-5, rtol=1e-5)


def test_crop_preserves_projection_including_outside_padding():
    K = torch.tensor([[400.0, 0, 300.0], [0, 500.0, 200.0], [0, 0, 1.0]])
    pts = torch.tensor([[0.2, -0.1, 2.0], [-0.5, 0.2, 3.0]])
    affine = torch.from_numpy(crop_affine((-30, 15, 370, 415), 256))
    raw, _ = project_points(pts, K)
    cropped, _ = project_points(pts, affine @ K)
    expected = torch.cat((raw, torch.ones(2, 1)), -1) @ affine.T
    torch.testing.assert_close(cropped, expected[:, :2])
    np.testing.assert_allclose(affine.numpy()[2], [0, 0, 1])


def test_teacher_polar_rotation_removes_scale_and_reflection():
    r = quaternion_to_matrix(axis_angle_to_quaternion(torch.tensor([0.4, 0.7, -0.2])))
    torch.testing.assert_close(proper_rotation(r * 3), r)
    reflected = r @ torch.diag(torch.tensor([1.0, 2.0, -3.0]))
    out = proper_rotation(reflected)
    torch.testing.assert_close(out.T @ out, torch.eye(3), atol=1e-6, rtol=1e-6)
    assert torch.linalg.det(out) > 0.999
