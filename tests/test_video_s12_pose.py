from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from panorama_demo.session import RGBDFrame
from panorama_demo.video_s12_config import S12TrajectoryConfig
from panorama_demo.video_s12_pose import parse_s12_trajectory, qualify_s12_poses


def _pose(x_mm: float = 0.0, *, yaw_deg: float = 0.0) -> list[list[float]]:
    angle = np.deg2rad(yaw_deg)
    pose = np.eye(4)
    pose[:2, :2] = ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
    pose[0, 3] = x_mm
    return pose.tolist()


def _payload(*, kinds: tuple[str, ...] = ("direct_orb_pose", "direct_orb_pose")):
    return {
        "schema": "gemini305-orbslam3-trajectory/v2",
        "pose_convention": "camera_to_world",
        "translation_unit": "mm",
        "timestamp_unit": "us",
        "poses": [
            {
                "frame_id": index,
                "timestamp": index * 10_000,
                "pose_status": "tracked",
                "pose_kind": kind,
                "pose_origin": "orb_map_0",
                "camera_to_world": _pose(index * 50.0),
                "tracking_state": "tracking",
            }
            for index, kind in enumerate(kinds)
        ],
    }


def _frame(index: int) -> RGBDFrame:
    return RGBDFrame(index, Path(f"{index}.png"), Path(f"{index}_d.png"), 1.0, index * 10_000, 8)


def _config() -> S12TrajectoryConfig:
    return S12TrajectoryConfig(5.0, 1.0, 1.0, 2.0, 2, True)


def test_interpolated_pose_is_rejected_as_render_source() -> None:
    trajectory = parse_s12_trajectory(
        _payload(kinds=("direct_orb_pose", "interpolated_se3_prior"))
    )
    rows = qualify_s12_poses((_frame(0), _frame(1)), trajectory, _config())
    assert rows[0].qualified is True
    assert rows[1].qualified is False
    assert rows[1].rejection_reasons == ("not_direct_orb_pose",)


def test_mixed_pose_origin_is_structural_fatal() -> None:
    payload = _payload()
    payload["poses"][1]["pose_origin"] = "orb_map_1"
    with pytest.raises(ValueError, match="mixes pose origins"):
        parse_s12_trajectory(payload)


def test_invalid_se3_is_structural_fatal() -> None:
    payload = _payload()
    payload["poses"][1]["camera_to_world"][3][3] = 2.0
    with pytest.raises(ValueError, match="homogeneous"):
        parse_s12_trajectory(payload)


def test_trajectory_without_explicit_direct_marker_is_rejected() -> None:
    payload = _payload()
    del payload["poses"][0]["pose_kind"]
    with pytest.raises(ValueError, match="explicit pose_kind"):
        parse_s12_trajectory(payload)


def test_rotation_is_audited_when_render_rotation_is_disabled() -> None:
    payload = _payload()
    payload["poses"][1]["camera_to_world"] = _pose(50.0, yaw_deg=3.0)
    trajectory = parse_s12_trajectory(payload)
    rows = qualify_s12_poses((_frame(0), _frame(1)), trajectory, _config())
    assert rows[1].qualified is False
    assert "yaw_limit" in rows[1].rejection_reasons

