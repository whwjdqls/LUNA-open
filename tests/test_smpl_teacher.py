"""Toy numeric rig verifies deformation algebra; contains no licensed body data."""

import pickle

import numpy as np
import torch
from smplx.lbs import lbs

from luna_open.avatar import Gaussians
from luna_open.geometry import axis_angle_to_quaternion, quaternion_to_matrix
from luna_open.smpl import SMPLTeacher


def make_teacher(tmp_path):
    # Four vertices and 24 SMPL-style joints; all vertices attach to the root.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    shape = np.zeros((4, 3, 10), np.float32)
    shape[:, 0, 0] = 0.1
    regressor = np.zeros((24, 4), np.float32)
    regressor[:, 0] = 1
    weights = np.zeros((4, 24), np.float32)
    weights[:, 0] = 1
    parents = np.zeros((2, 24), np.int64)
    parents[0, 0] = -1
    parents[1] = np.arange(24)
    model = dict(
        v_template=vertices,
        shapedirs=shape,
        posedirs=np.zeros((4, 3, 207), np.float32),
        J_regressor=regressor,
        weights=weights,
        kintree_table=parents,
        f=np.array([[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]]),
    )
    path = tmp_path / "toy.pkl"
    with path.open("wb") as f:
        pickle.dump(model, f)
    return SMPLTeacher(str(path), num_queries=16)


def test_teacher_matches_library_mesh_without_double_shape(tmp_path):
    teacher = make_teacher(tmp_path)
    beta = torch.zeros(1, 10)
    beta[:, 0] = 2
    pose = torch.zeros(1, 72)
    pose[:, 2] = 0.8
    body = teacher.body
    vertices, _ = lbs(
        beta,
        pose,
        body.v_template,
        body.shapedirs,
        body.posedirs,
        body.J_regressor,
        body.parents,
        body.lbs_weights,
    )
    means = teacher.shaped_anchors(beta).requires_grad_()
    q = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, 16, 4)
    gs = Gaussians(means, q, torch.ones_like(means) * 0.01, torch.ones(1, 16), means * 0 + 0.5)
    view = torch.eye(4)[None]
    out = teacher(gs, pose, beta, view, detach=False)
    torch.testing.assert_close(out.means, teacher.interpolate(vertices))
    out.means.sum().backward()
    assert torch.isfinite(means.grad).all() and means.grad.abs().sum() > 0
    detached = teacher(gs, pose, beta, view)
    assert not detached.means.requires_grad


def test_global_motion_includes_shape_dependent_root_pivot(tmp_path):
    teacher = make_teacher(tmp_path)
    beta = torch.zeros(1, 10)
    beta[:, 0] = 2
    pose = torch.zeros(1, 72)
    pose[:, 2] = torch.pi / 2
    view = torch.eye(4)[None]
    view[:, :3, 3] = torch.tensor([0.0, 0.0, 3.0])
    rotation, translation = teacher.global_motion(pose, beta, view)
    torch.testing.assert_close(
        rotation, quaternion_to_matrix(axis_angle_to_quaternion(pose[:, :3]))
    )
    torch.testing.assert_close(translation, torch.tensor([[0.2, -0.2, 3.0]]), atol=1e-6, rtol=1e-6)


def test_neuman_pose_correction_switch_has_expected_effect(tmp_path):
    teacher = make_teacher(tmp_path)
    # One non-root rotation changes R[1,1] from 1 to 0; use that feature
    # to impose a known +0.2m z correction on every vertex. Root skinning
    # makes the expected spatial change independent of the rotated child.
    teacher.body.posedirs[4, 2::3] = -0.2
    pose, beta = torch.zeros(1, 72), torch.zeros(1, 10)
    pose[:, 3] = torch.pi / 2
    corrected, _ = teacher.deformation(pose, beta)
    teacher.pose_blend_shapes = False
    neuman, _ = teacher.deformation(pose, beta)
    torch.testing.assert_close(neuman, teacher.anchors[None])
    expected = torch.zeros_like(corrected)
    expected[..., 2] = 0.2
    torch.testing.assert_close(corrected - neuman, expected, atol=1e-6, rtol=1e-5)
