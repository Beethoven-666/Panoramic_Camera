from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_s12_config import S12ScanConfig
from panorama_demo.video_s12_pose import S12PoseQualification, S12PoseRecord
from panorama_demo.video_s12_scan import select_s12_scan_segment


def _config(**overrides) -> S12ScanConfig:
    values = {
        "minimum_pose_count": 3,
        "minimum_scan_displacement_m": 0.08,
        "rail_axis_ransac_iterations": 64,
        "rail_axis_inlier_threshold_m": 0.01,
        "maximum_cross_track_error_m": 0.02,
        "maximum_backward_step_m": 0.002,
        "maximum_pause_span_frames": 30,
        "require_image_motion_direction_agreement": True,
    }
    values.update(overrides)
    return S12ScanConfig(**values)


def _qualification(frame_id: int, center_m: tuple[float, float, float]) -> S12PoseQualification:
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(center_m) * 1000.0
    record = S12PoseRecord(
        frame_id,
        frame_id * 10_000,
        "tracked",
        "direct_orb_pose",
        "orb_map_0",
        pose,
        "tracking",
    )
    return S12PoseQualification(
        frame_id,
        frame_id * 10_000,
        True,
        0.0,
        0.0,
        0.0,
        0.0,
        True,
        (),
        record,
    )


def _missing_qualification(frame_id: int) -> S12PoseQualification:
    return S12PoseQualification(
        frame_id,
        frame_id * 10_000,
        False,
        None,
        None,
        None,
        None,
        False,
        ("missing_trajectory_record",),
        None,
    )


def test_scan_axis_is_recovered_from_pose() -> None:
    direction = np.asarray((1.0, 2.0, 0.2))
    direction /= np.linalg.norm(direction)
    rows = tuple(
        _qualification(index, tuple(direction * (index * 0.05) + (0.0, 0.0, 0.001 * (-1) ** index)))
        for index in range(6)
    )
    segment = select_s12_scan_segment(rows, _config(), image_motion_agreement=[True] * 5)
    assert abs(np.dot(segment.rail_axis, direction)) > 0.999
    assert segment.candidate_frame_ids == tuple(range(6))


def test_pause_does_not_create_invented_progress() -> None:
    positions = (0.0, 0.05, 0.05, 0.10, 0.15)
    rows = tuple(_qualification(index, (value, 0.0, 0.0)) for index, value in enumerate(positions))
    segment = select_s12_scan_segment(rows, _config(), image_motion_agreement=[True] * 4)
    progress = [row.progress_m for row in segment.pose_audit]
    assert progress[1] == pytest.approx(progress[2])
    assert progress == sorted(progress)


def test_missing_direct_pose_frame_is_not_source_but_bounded_gap_is_allowed() -> None:
    rows = (
        _qualification(0, (0.0, 0.0, 0.0)),
        _qualification(1, (0.05, 0.0, 0.0)),
        _missing_qualification(2),
        _qualification(3, (0.10, 0.0, 0.0)),
        _qualification(4, (0.15, 0.0, 0.0)),
    )
    segment = select_s12_scan_segment(
        rows, _config(), image_motion_agreement={1: True, 3: True, 4: True}
    )
    assert segment.candidate_frame_ids == (0, 1, 3, 4)


def test_backward_motion_splits_scan_and_selects_longest_forward_run() -> None:
    positions = (0.0, 0.05, 0.10, 0.15, 0.08, 0.13, 0.18)
    rows = tuple(_qualification(index, (value, 0.0, 0.0)) for index, value in enumerate(positions))
    segment = select_s12_scan_segment(rows, _config(), image_motion_agreement=[True] * 6)
    assert segment.candidate_frame_ids == (0, 1, 2, 3)


def test_pose_and_image_direction_must_agree() -> None:
    rows = tuple(_qualification(index, (index * 0.05, 0.0, 0.0)) for index in range(6))
    segment = select_s12_scan_segment(
        rows,
        _config(minimum_pose_count=3, minimum_scan_displacement_m=0.08),
        image_motion_agreement={1: True, 2: True, 3: False, 4: True, 5: True},
    )
    assert segment.candidate_frame_ids == (0, 1, 2)
    assert "image_pose_direction_disagreement" in segment.pose_audit[3].rejection_reasons


def test_insufficient_direct_pose_segment_is_structural_fatal() -> None:
    rows = tuple(_qualification(index, (index * 0.02, 0.0, 0.0)) for index in range(3))
    with pytest.raises(RuntimeError, match="structural fatal"):
        select_s12_scan_segment(rows, _config(minimum_scan_displacement_m=0.5))
