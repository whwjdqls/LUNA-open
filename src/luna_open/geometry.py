"""Column-vector geometry; OpenCV cameras; scalar-first (wxyz) quaternions."""

import torch
import torch.nn.functional as F
from torch import Tensor


def normalize_quaternion(q: Tensor) -> Tensor:
    norm = q.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(norm > 1e-8, q / norm.clamp_min(1e-8), identity)


def quaternion_multiply(a: Tensor, b: Tensor) -> Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def quaternion_to_matrix(q: Tensor) -> Tensor:
    w, x, y, z = normalize_quaternion(q).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def axis_angle_to_quaternion(v: Tensor) -> Tensor:
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * torch.pi)) * v), -1)


def matrix_to_quaternion(m: Tensor) -> Tensor:
    """Stable largest-component construction, including 180-degree rotations."""
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    candidates = torch.stack((1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c), -1)
    r = candidates.clamp_min(1e-12).sqrt()
    d = m[..., 2, 1] - m[..., 1, 2]
    e = m[..., 0, 2] - m[..., 2, 0]
    f = m[..., 1, 0] - m[..., 0, 1]
    g = m[..., 0, 1] + m[..., 1, 0]
    h = m[..., 0, 2] + m[..., 2, 0]
    i = m[..., 1, 2] + m[..., 2, 1]
    rows = torch.stack(
        (
            torch.stack((r[..., 0] ** 2, d, e, f), -1),
            torch.stack((d, r[..., 1] ** 2, g, h), -1),
            torch.stack((e, g, r[..., 2] ** 2, i), -1),
            torch.stack((f, h, i, r[..., 3] ** 2), -1),
        ),
        -2,
    )
    rows = rows / (2 * r[..., :, None])
    index = candidates.argmax(-1)[..., None, None].expand(*m.shape[:-2], 1, 4)
    return normalize_quaternion(rows.gather(-2, index).squeeze(-2))


def proper_rotation(matrix: Tensor) -> Tensor:
    """Closest SO(3) rotation; teacher-only SVD (repeated singular values)."""
    u, _, vh = torch.linalg.svd(matrix)
    sign = torch.linalg.det(u @ vh)
    diagonal = torch.ones_like(matrix[..., 0, :])
    diagonal[..., -1] = sign
    return (u * diagonal[..., None, :]) @ vh


def sincos_to_quaternion(pairs: Tensor) -> Tensor:
    """[...,3,2] sin/cos pairs for x,y,z; composition Rz @ Ry @ Rx."""
    pairs = F.normalize(pairs, dim=-1, eps=1e-8)
    angles = torch.atan2(pairs[..., 0], pairs[..., 1])
    quats = []
    for axis in range(3):
        v = torch.zeros_like(angles)
        v[..., axis] = angles[..., axis]
        quats.append(axis_angle_to_quaternion(v))
    return normalize_quaternion(
        quaternion_multiply(quats[2], quaternion_multiply(quats[1], quats[0]))
    )


def matrix_to_sincos(rotation: Tensor) -> Tensor:
    """Euler RzRyRx targets. Gimbal lock uses z=0 and a consistent x."""
    y = torch.asin((-rotation[..., 2, 0]).clamp(-1, 1))
    lock = torch.cos(y).abs() < 1e-6
    x = torch.where(
        lock,
        torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1]),
        torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2]),
    )
    z = torch.where(
        lock, torch.zeros_like(y), torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    )
    angles = torch.stack((x, y, z), -1)
    return torch.stack((angles.sin(), angles.cos()), -1)


def transform_points(matrix: Tensor, points: Tensor) -> Tensor:
    return points @ matrix[..., :3, :3].transpose(-1, -2) + matrix[..., None, :3, 3]


def project_points(points: Tensor, K: Tensor) -> tuple[Tensor, Tensor]:
    pixel = points @ K.transpose(-1, -2)
    valid = points[..., 2] > 1e-4
    return pixel[..., :2] / pixel[..., 2:].clamp_min(1e-4), valid
