import pytest
import torch

from luna_open.geometry import axis_angle_to_quaternion, quaternion_to_matrix
from luna_open.temporal import neuman_fixed_world, temporal_metrics, uniform_interval


def test_acceleration_jerk_and_time_units_from_analytic_motion():
    interval = 0.5
    time = torch.arange(7, dtype=torch.float64) * interval
    jerk = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float64)
    # Three points with distinct static offsets and identical cubic motion.
    offsets = torch.tensor([[1, 0, 2], [0, 3, 1], [-2, 1, 0]], dtype=torch.float64)
    positions = time[:, None, None] ** 3 / 6 * jerk + offsets[None]
    scores = temporal_metrics(positions, interval)
    assert scores["mae"] == pytest.approx(4.5)
    assert scores["msj"] == pytest.approx(9.0)
    faster = temporal_metrics(positions, interval / 2)
    assert faster["mae"] == pytest.approx(scores["mae"] * 4)
    assert faster["msj"] == pytest.approx(scores["msj"] * 64)


def test_camera_and_per_frame_scale_do_not_create_world_motion():
    torch.manual_seed(31)
    frames = 7
    world_points = torch.randn(5, 3, dtype=torch.float64)
    matrices = torch.eye(4, dtype=torch.float64).repeat(frames, 1, 1)
    rotation = quaternion_to_matrix(
        axis_angle_to_quaternion(torch.randn(frames, 3, dtype=torch.float64))
    )
    matrices[:, :3, :3] = rotation
    matrices[:, :3, 3] = torch.randn(frames, 3, dtype=torch.float64)
    scales = torch.linspace(1.0, 3.0, frames, dtype=torch.float64)
    camera_points = (
        world_points[None] @ rotation.transpose(-1, -2) + matrices[:, None, :3, 3]
    ) / scales[:, None, None]
    recovered = neuman_fixed_world(camera_points, matrices, scales, reference_scale=2.0)
    torch.testing.assert_close(recovered, world_points[None].expand(frames, -1, -1) / 2)
    scores = temporal_metrics(recovered, 1.0)
    assert scores["mae"] < 1e-12
    assert scores["msj"] < 1e-24


def test_temporal_rejects_insufficient_irregular_or_invalid_data():
    assert uniform_interval(torch.tensor([2, 7, 12, 17])) == 5
    for ids in ([0, 1, 3, 4], [3, 2, 1, 0], [0, 1, 2]):
        with pytest.raises(ValueError):
            uniform_interval(torch.tensor(ids))
    for interval in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="interval"):
            temporal_metrics(torch.zeros(4, 2, 3), interval)
    with pytest.raises(ValueError, match="four frames"):
        temporal_metrics(torch.zeros(3, 2, 3), 1)
    with pytest.raises(ValueError, match="nonfinite"):
        temporal_metrics(torch.full((4, 2, 3), float("nan")), 1)
