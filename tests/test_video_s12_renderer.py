from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo.video_s12_renderer import render_s012_stage_a
from panorama_demo.video_s12_schedule import build_s012_schedule


def _calibration(
    *,
    width: int = 8,
    height: int = 5,
    cx: float = 3.25,
    distortion: tuple[float, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        width=width,
        height=height,
        fx=10.0,
        fy=10.0,
        cx=cx,
        cy=2.0,
        distortion=distortion,
    )


def _constant_image(calibration: SimpleNamespace, value: int) -> np.ndarray:
    return np.full((calibration.height, calibration.width, 3), value, dtype=np.uint8)


def test_stage_a_has_no_color_gain_or_blend() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule([10, 20], [calibration.cx, 7.25], calibration)
    result = render_s012_stage_a(
        schedule,
        calibration,
        {10: _constant_image(calibration, 17), 20: _constant_image(calibration, 203)},
    )

    boundary = schedule.boundaries[1]
    assert np.all(result.image[:, :boundary] == 17)
    assert np.all(result.image[:, boundary:] == 203)
    assert set(np.unique(result.image)) == {17, 203}


def test_stage_a_has_no_vertical_correction() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule([10], [calibration.cx], calibration)
    image = np.zeros((calibration.height, calibration.width, 3), dtype=np.uint8)
    for row in range(calibration.height):
        image[row, :, :] = row * 31

    result = render_s012_stage_a(schedule, calibration, {10: image})

    for row in range(calibration.height):
        assert np.all(result.image[row] == row * 31)
        assert np.allclose(result.source_v[row], row)


def test_roi_remap_uses_only_assigned_widths(monkeypatch: pytest.MonkeyPatch) -> None:
    import panorama_demo.video_s12_renderer as renderer

    calibration = _calibration()
    schedule = build_s012_schedule(
        [1, 2, 3], [calibration.cx, 5.25, 7.25], calibration
    )
    calls: list[tuple[int, int]] = []
    original = renderer.accelerated_remap

    def audited_remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        calls.append(map_x.shape)
        return original(image, map_x, map_y, *args, **kwargs)

    monkeypatch.setattr(renderer, "accelerated_remap", audited_remap)
    result = render_s012_stage_a(
        schedule,
        calibration,
        {frame_id: _constant_image(calibration, frame_id) for frame_id in (1, 2, 3)},
    )

    assert calls == [
        (calibration.height, assignment.width)
        for assignment in schedule.assignments
        if not assignment.zero_width
    ]
    assert all(width < schedule.canvas_width for _, width in calls)
    assert result.remap_invocations == len(calls)


def test_zero_width_source_is_not_decoded_or_rendered() -> None:
    calibration = _calibration(width=6, cx=2.2)
    schedule = build_s012_schedule(
        [1, 2, 3], [2.2, 2.3, 2.4], calibration, hard_internal_width_px=None
    )
    decoded: list[int] = []

    def load(frame_id: int) -> np.ndarray:
        decoded.append(frame_id)
        if frame_id == 2:
            raise AssertionError("zero-width source was decoded")
        return _constant_image(calibration, frame_id * 20)

    result = render_s012_stage_a(schedule, calibration, load)

    assert decoded == [1, 3]
    assert result.decoded_frame_ids == (1, 3)
    assert 2 not in result.owner_frame_id


def test_owner_and_source_uv_are_pixel_level() -> None:
    calibration = _calibration(
        distortion=(0.03, -0.01, 0.001, -0.002, 0.0)
    )
    schedule = build_s012_schedule([10], [calibration.cx], calibration)
    result = render_s012_stage_a(
        schedule, calibration, {10: _constant_image(calibration, 0)}
    )

    assert result.owner_frame_id.shape == (calibration.height, schedule.canvas_width)
    assert result.source_u.shape == result.owner_frame_id.shape
    assert result.source_v.shape == result.owner_frame_id.shape
    assert result.assignment_index.shape == result.owner_frame_id.shape
    column_u = result.source_u[:, 1]
    assert np.ptp(column_u[np.isfinite(column_u)]) > 0.0
    assert np.array_equal(result.valid_mask, result.owner_frame_id >= 0)
    assert np.all(result.owner_frame_id[~result.valid_mask] == -1)
    assert np.isnan(result.source_u[~result.valid_mask]).all()
    assert np.isnan(result.source_v[~result.valid_mask]).all()


def test_black_rgb_remains_valid_content() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule([42], [calibration.cx], calibration)
    result = render_s012_stage_a(
        schedule, calibration, {42: _constant_image(calibration, 0)}
    )

    assert result.valid_mask.all()
    assert np.all(result.owner_frame_id == 42)
    assert np.all(result.image == 0)


def test_second_pixel_write_is_structural_fatal() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule([1, 2], [calibration.cx, 6.25], calibration)
    second = replace(schedule.assignments[1], left_x=schedule.assignments[0].right_x - 1)
    malformed = replace(schedule, assignments=(schedule.assignments[0], second))

    with pytest.raises(ValueError):
        render_s012_stage_a(
            malformed,
            calibration,
            {1: _constant_image(calibration, 1), 2: _constant_image(calibration, 2)},
        )


def test_renderer_accepts_no_depth_or_foreign_fill_input() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule([4], [calibration.cx], calibration)
    result = render_s012_stage_a(
        schedule, calibration, {4: _constant_image(calibration, 77)}
    )

    assert result.decoded_frame_ids == (4,)
    assert set(np.unique(result.owner_frame_id)) == {4}
