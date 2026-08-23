from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s13_base_renderer import render_s13_p0
from panorama_demo.video_s13_motion import (
    S13MotionEdge,
    build_basic_s13_progress,
)
from panorama_demo.video_s13_schedule import (
    build_s13_midpoint_schedule,
    select_dense_s13_sources,
)
from panorama_demo.video_s13_session import S13RenderFrame


def _frame(frame_id: int) -> S13RenderFrame:
    return S13RenderFrame(frame_id, Path(f"{frame_id}.png"), frame_id, 64, 32)


def _edge(source: int, target: int, advance_px: float) -> S13MotionEdge:
    return S13MotionEdge(
        source,
        target,
        1,
        advance_px,
        0.0,
        10.0,
        20,
        advance_px,
        0.0,
        1.0,
        advance_px,
        "grid_lk",
        False,
        (),
    )


def _render_scan(
    scene: np.ndarray,
    positions: tuple[int, ...],
    calibration: CameraIntrinsics,
):
    frames = tuple(_frame(index) for index in range(len(positions)))
    edges = tuple(
        _edge(index, index + 1, float(positions[index + 1] - positions[index]))
        for index in range(len(positions) - 1)
    )
    progress = build_basic_s13_progress(frames, edges)
    selection = select_dense_s13_sources(progress, edges, calibration)
    schedule = build_s13_midpoint_schedule(selection, calibration)
    source_images = {
        frame.frame_id: scene[:, position : position + calibration.width]
        for frame, position in zip(frames, positions, strict=True)
    }
    result = render_s13_p0(
        schedule,
        calibration,
        source_images.__getitem__,
        placement_methods=selection.placement_methods,
    )
    return progress, selection, schedule, result


def test_repeated_fast_frames_do_not_expand_the_same_scene() -> None:
    calibration = CameraIntrinsics(
        width=64,
        height=32,
        fx=50.0,
        fy=50.0,
        cx=31.5,
        cy=15.5,
        distortion=(),
    )
    scene = np.random.default_rng(13).integers(
        0,
        256,
        size=(calibration.height, 88, 3),
        dtype=np.uint8,
    )

    slow = _render_scan(scene, (0, 8, 16, 24), calibration)
    fast = _render_scan(scene, (0, 0, 0, 8, 8, 16, 16, 24), calibration)
    slow_progress, slow_selection, slow_schedule, slow_result = slow
    fast_progress, fast_selection, fast_schedule, fast_result = fast

    assert slow_progress.centers_x == (0.0, 8.0, 16.0, 24.0)
    assert fast_progress.centers_x == (0.0, 0.0, 0.0, 8.0, 8.0, 16.0, 16.0, 24.0)
    assert slow_selection.frame_ids == (0, 1, 2, 3)
    assert fast_selection.frame_ids == (0, 3, 5, 7)
    assert slow_schedule.canvas_width == fast_schedule.canvas_width == scene.shape[1]
    assert slow_result.remap_invocations == fast_result.remap_invocations == 4
    assert np.array_equal(fast_result.image, slow_result.image)
    assert np.array_equal(fast_result.image, scene)

