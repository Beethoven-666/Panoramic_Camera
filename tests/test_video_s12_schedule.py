from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo.video_s12_schedule import (
    build_s012_schedule,
    validate_s012_schedule,
)


def _calibration(*, width: int = 10, height: int = 4, cx: float = 4.25) -> SimpleNamespace:
    return SimpleNamespace(
        width=width,
        height=height,
        fx=8.0,
        fy=8.0,
        cx=cx,
        cy=1.5,
        distortion=(),
    )


def test_voronoi_columns_partition_canvas_exactly_once() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule(
        [11, 22, 33], [calibration.cx, 7.0, 10.0], calibration
    )

    assert schedule.canvas_width == 16
    np.testing.assert_array_equal(
        np.bincount(schedule.column_assignment_index, minlength=3),
        [6, 3, 7],
    )
    assert np.all(schedule.column_frame_id >= 0)
    validate_s012_schedule(schedule)


def test_schedule_uses_half_open_intervals_and_rounded_midpoints() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule(
        [1, 2, 3], [calibration.cx, 7.0, 10.0], calibration
    )

    assert schedule.boundaries == (0, 6, 9, 16)
    assert [(item.left_x, item.right_x) for item in schedule.assignments] == [
        (0, 6),
        (6, 9),
        (9, 16),
    ]
    assert schedule.column_frame_id[5] == 1
    assert schedule.column_frame_id[6] == 2
    assert schedule.column_frame_id[8] == 2
    assert schedule.column_frame_id[9] == 3


def test_source_x_uses_calibrated_cx_not_image_midpoint() -> None:
    calibration = _calibration(width=9, cx=2.25)
    schedule = build_s012_schedule([7, 8], [2.25, 5.25], calibration)

    for assignment in schedule.assignments:
        columns = np.arange(assignment.left_x, assignment.right_x)
        np.testing.assert_allclose(
            schedule.column_source_u[assignment.left_x : assignment.right_x],
            calibration.cx + columns - assignment.center_x,
        )


def test_zero_width_source_is_marked_in_schedule() -> None:
    calibration = _calibration(width=6, cx=2.2)
    schedule = build_s012_schedule(
        [1, 2, 3], [2.2, 2.3, 2.4], calibration, hard_internal_width_px=None
    )

    assert schedule.assignments[1].zero_width is True
    assert schedule.assignments[1].width == 0
    assert 2 not in schedule.column_frame_id


def test_first_and_last_full_fields_are_automatic_and_s1_independent() -> None:
    calibration = _calibration(width=10, cx=4.25)
    schedule = build_s012_schedule([4, 9], [4.25, 9.1], calibration)

    assert schedule.canvas_left == 0
    assert schedule.canvas_right == 15
    assert schedule.assignments[0].left_x == 0
    assert schedule.assignments[-1].right_x == 15


def test_internal_width_over_24_is_structural_fatal() -> None:
    calibration = _calibration(width=10, cx=4.25)
    with pytest.raises(ValueError, match="hard width"):
        build_s012_schedule(
            [1, 2, 3], [4.25, 34.25, 64.25], calibration
        )


def test_schedule_validator_rejects_column_provenance_tampering() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule([1, 2], [4.25, 7.25], calibration)
    bad_frame_ids = schedule.column_frame_id.copy()
    bad_frame_ids[0] = 999

    with pytest.raises(ValueError, match="frame provenance"):
        validate_s012_schedule(replace(schedule, column_frame_id=bad_frame_ids))
