from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomConfig,
    estimate_rgb_motion_pixels_per_mm,
    render_calibrated_rgb_pushbroom,
)
from panorama_demo.session import RGBDSession, load_rgbd_session
from panorama_demo.synthetic import generate_sequence


def _make_rgb_pushbroom_input(tmp_path: Path, *, seed: int) -> tuple[RGBDSession, list[np.ndarray]]:
    root = generate_sequence(
        tmp_path / "session",
        frame_count=5,
        frame_width=320,
        frame_height=200,
        step=60,
        seed=seed,
    )
    session = load_rgbd_session(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    trajectory = manifest["known_trajectory"]
    assert trajectory["transform"] == "camera_to_world"
    return (
        session,
        [
            np.asarray(row["matrix_row_major"], dtype=np.float64).reshape(4, 4)
            for row in trajectory["poses"]
        ],
    )


def _reliable_adjacent_rgb_motions(frame_count: int) -> list[dict[str, object]]:
    """Synthetic frames advance by 60 RGB pixels at the far background."""

    return [{"dx": 60.0, "reliable": True} for _ in range(frame_count - 1)]


def test_pushbroom_uses_every_real_frame_once_with_bounded_strip_residency(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=17)
    motions = _reliable_adjacent_rgb_motions(len(session.frames))

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=motions,
    )

    metadata = result.metadata
    metrics = metadata["quality_metrics"]
    assert len(motions) == len(session.frames) - 1
    assert metadata["source_count"] == len(session.frames)
    assert metadata["frame_ids"] == [frame.frame_id for frame in session.frames]
    assert metadata["single_inverse_remap_per_source"] is True
    assert metadata["interpolated_pose_count"] == 0
    assert metrics["source_remap_count"] == len(session.frames)
    assert 2 <= metrics["maximum_resident_strips"] <= 5
    assert len(metadata["rgb_motion_scale"]["samples"]) == len(session.frames) - 1
    assert metadata["layout"]["endpoint_policy"] == "outward_half_fov"
    assert len(metadata["layout"]["endpoint_outer_owner_intervals_x"]) == 2
    assert metadata["layout"]["maximum_source_strip_width"] > 64
    supports = metadata["layout"]["source_support_intervals_x"]
    assert all(right - left <= 64 for left, right in supports[1:-1])
    assert all(count > 0 for count in metadata["source_owner_pixel_counts"])
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )
    assert metrics["endpoint_outer_half_fov_preserved"] is True
    assert metrics["endpoint_outer_half_fov_trimmed_column_counts"] == [0, 0]
    assert metrics["endpoint_outer_half_fov_trimmed_invalid_pixel_counts"] == [0, 0]


def test_pushbroom_keeps_outward_endpoint_coverage_when_virtual_x_is_reversed(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=19)
    reversed_poses = []
    for pose in poses:
        reversed_pose = pose.copy()
        reversed_pose[:3, :3] = np.diag((-1.0, 1.0, -1.0))
        reversed_poses.append(reversed_pose)

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        reversed_poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    layout = result.metadata["layout"]
    metrics = result.metadata["quality_metrics"]
    assert layout["temporal_to_virtual_x_sign"] < 0.0
    assert layout["endpoint_policy"] == "outward_half_fov"
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )


def test_pushbroom_pair_blends_are_narrow_and_never_include_rgb_risk(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=23)
    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    pairs = result.metadata["pairs"]
    metrics = result.metadata["quality_metrics"]
    assert len(pairs) == len(session.frames) - 1
    assert all(2 <= pair["blend_width_pixels"] <= 8 for pair in pairs)
    assert all(pair["blend_zone_risk_pixel_count"] == 0 for pair in pairs)
    assert metrics["blend_zone_risk_pixel_count"] == 0
    assert metrics["blend_zone_risk_fraction"] == 0.0
    assert metrics["blend_zone_fraction"] <= 0.20


def test_pushbroom_crops_from_valid_mask_and_preserves_valid_black_rgb(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=31)
    black = np.zeros(
        (session.calibration.height, session.calibration.width, 3), dtype=np.uint8
    )
    for frame in session.frames:
        assert cv2.imwrite(str(frame.color_path), black)
        # Rendering after this deletion proves output pixels are not read from
        # the strict session's aligned-depth files.
        frame.aligned_depth_path.unlink()

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    crop = result.metadata["crop"]
    assert result.metadata["depth_used_for_output_pixels"] is False
    assert result.panorama.shape[:2] == (crop["height"], crop["width"])
    assert crop["width"] > 0 and crop["height"] > 0
    assert not np.any(result.panorama)


def test_pushbroom_rejects_insufficient_reliable_rgb_motion_scale(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=41)
    invalid_motions = [
        {"dx": 0.0, "reliable": False} for _ in range(len(session.frames) - 1)
    ]

    with pytest.raises(RuntimeError, match="too few reliable adjacent RGB motion"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=invalid_motions,
        )


def test_pushbroom_rejects_unstable_rgb_motion_scale(tmp_path: Path) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=47)
    unstable_motions = [
        {"dx": value, "reliable": True} for value in (60.0, 120.0, 1200.0, 2400.0)
    ]

    with pytest.raises(RuntimeError, match="RGB-motion scale is unstable"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=unstable_motions,
        )
