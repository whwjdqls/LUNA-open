"""LUNA equations 11–12 and explicit NeuMan temporal-coordinate conversion.

The paper's MAE expression measures acceleration magnitude, without a ground-
truth subtraction. Stable point correspondence and a fixed coordinate frame
are required. Time and length units must be declared by the caller.
"""

import math

import torch
from torch import Tensor


def uniform_interval(frame_ids: Tensor) -> float:
    """Return the frame-index spacing; reject gaps or a nonchronological order."""
    if frame_ids.ndim != 1 or len(frame_ids) < 4:
        raise ValueError("Temporal evaluation requires at least four ordered frames")
    indices = frame_ids.to(torch.float64)
    differences = indices.diff()
    if (
        not torch.isfinite(indices).all()
        or differences[0] <= 0
        or not torch.allclose(differences, differences[0].expand_as(differences))
    ):
        raise ValueError("Temporal evaluation requires uniformly spaced, increasing frame IDs")
    return float(differences[0])


@torch.no_grad()
def temporal_metrics(positions: Tensor, interval: float) -> dict[str, float]:
    """MAE and MSJ for [time, points, xyz], in length/time² and length²/time⁶.

    `interval` is in seconds only when capture timing is known. For NeuMan use
    its frame-index spacing and explicitly label the output per frame interval.
    No point matching, camera stabilization or temporal resampling is implicit.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[1] == 0:
        raise ValueError("Expected nonempty positions [time, points, 3]")
    if len(positions) < 4:
        raise ValueError("Both MAE and MSJ require at least four frames")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Time interval must be finite and positive")
    if not torch.isfinite(positions).all():
        raise ValueError("Trajectory contains nonfinite positions")
    # Temporal differences amplify rounding; do the arithmetic in float64.
    points = positions.to(torch.float64)
    acceleration = points.diff(n=2, dim=0) / interval**2
    jerk = points.diff(n=3, dim=0) / interval**3
    return dict(
        mae=float(acceleration.norm(dim=-1).mean()),
        msj=float(jerk.square().sum(-1).mean()),
    )


@torch.no_grad()
def neuman_fixed_world(
    camera_meters: Tensor,
    world_to_camera: Tensor,
    alignment_scales: Tensor,
    reference_scale: float,
) -> Tensor:
    """Recover one fixed NeuMan world frame, normalized once per sequence.

    Inputs: camera points [T,K,3] in native SMPL meters, COLMAP rigid transforms
    [T,4,4], and per-frame body-alignment scales [T]. Undo camera-space scale
    normalization, invert the rigid camera, then divide by ONE reference scale.
    Use the median training-frame alignment scale. Output units are approximate
    meters ("proxy meters"); this does not prove metric COLMAP calibration or
    remove body-fit noise. Do not normalize each recovered world frame separately.
    """
    if camera_meters.ndim != 3 or camera_meters.shape[-1] != 3:
        raise ValueError("Expected camera points [time, points, 3]")
    frames = len(camera_meters)
    if world_to_camera.shape != (frames, 4, 4) or alignment_scales.shape != (frames,):
        raise ValueError("Camera/scale sequence must match trajectory frames")
    if not math.isfinite(reference_scale) or reference_scale <= 0:
        raise ValueError("Reference scale must be finite and positive")
    if (
        not torch.isfinite(camera_meters).all()
        or not torch.isfinite(world_to_camera).all()
        or not torch.isfinite(alignment_scales).all()
        or (alignment_scales <= 0).any()
    ):
        raise ValueError("Camera geometry and positive scales must be finite")
    matrices = world_to_camera.to(device=camera_meters.device, dtype=torch.float64)
    rotation, translation = matrices[:, :3, :3], matrices[:, :3, 3]
    identity = torch.eye(3, dtype=torch.float64, device=camera_meters.device).expand(frames, 3, 3)
    if not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity, atol=1e-5):
        raise ValueError("World-to-camera rotations must be orthonormal")
    if (torch.linalg.det(rotation) <= 0).any():
        raise ValueError("World-to-camera rotations must preserve handedness")
    scales = alignment_scales.to(device=camera_meters.device, dtype=torch.float64)
    camera_colmap = camera_meters.to(torch.float64) * scales[:, None, None]
    # Row-vector form of R.T @ (camera_point - t).
    return ((camera_colmap - translation[:, None]) @ rotation) / reference_scale
