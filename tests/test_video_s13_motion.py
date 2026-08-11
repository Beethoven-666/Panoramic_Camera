from __future__ import annotations

from dataclasses import replace

import numpy as np

from panorama_demo.video_s13_motion import S13MotionEdge, build_basic_s13_progress
from panorama_demo.video_s13_session import S13RenderFrame


def _frame(frame_id: int) -> S13RenderFrame:
    from pathlib import Path

    return S13RenderFrame(frame_id, Path(f"{frame_id}.png"), frame_id, 64, 32)


def _edge(left: int, right: int, value: float | None, *, step: int = 1) -> S13MotionEdge:
    return S13MotionEdge(
        left, right, step, value, 0.0 if value is not None else None, 10.0 if value is not None else 0.0,
        20 if value is not None else 0, value, 0.0 if value is not None else None,
        1.0 if value is not None else 0.0, value, "grid_lk" if value is not None else "unavailable",
        value is None, ("low_lk_support",) if value is None else (),
    )


def test_basic_progress_survives_disconnected_adjacent_graph() -> None:
    frames = tuple(_frame(index) for index in range(4))
    progress = build_basic_s13_progress(frames, (_edge(0, 1, 8.0), _edge(1, 2, None), _edge(2, 3, 7.0)))
    assert progress.spatial is True
    assert progress.adjacent_reliable_graph_connected is False
    assert list(progress.centers_x) == [0.0, 8.0, 15.5, 22.5]
    assert progress.placement_methods[2] == "session_median"


def test_no_rgb_motion_is_not_mislabeled_as_spatial_panorama() -> None:
    frames = tuple(_frame(index) for index in range(3))
    progress = build_basic_s13_progress(frames, (_edge(0, 1, 0.0), _edge(1, 2, 0.0)))
    assert progress.spatial is False


def test_observed_stationary_prefix_does_not_borrow_moving_session_median() -> None:
    frames = tuple(_frame(index) for index in range(6))
    progress = build_basic_s13_progress(
        frames,
        (
            _edge(0, 1, 0.09),
            _edge(1, 2, -0.08),
            _edge(2, 3, 0.02),
            _edge(3, 4, 12.0),
            _edge(4, 5, 10.0),
        ),
    )
    assert progress.spatial is True
    assert list(progress.centers_x) == [0.0, 0.0, 0.0, 0.0, 12.0, 22.0]
    assert progress.placement_methods[1:4] == ("zero_duplicate",) * 3
    assert "session_median" not in progress.placement_methods


def test_low_confidence_near_zero_uses_missing_edge_fallback() -> None:
    frames = tuple(_frame(index) for index in range(4))
    uncertain = replace(
        _edge(1, 2, 0.05),
        lk_total_weight=0.0,
        lk_observation_count=0,
        phase_response=0.0,
        risk=True,
        telemetry_only_reasons=("low_lk_support", "low_phase_response"),
    )
    progress = build_basic_s13_progress(
        frames,
        (_edge(0, 1, 8.0), uncertain, _edge(2, 3, 8.0)),
    )
    assert progress.centers_x == (0.0, 8.0, 16.0, 24.0)
    assert progress.placement_methods[2] == "session_median"


def test_multistep_evidence_preserves_coherent_subpixel_scan() -> None:
    frames = tuple(_frame(index) for index in range(6))
    adjacent = tuple(_edge(index, index + 1, 0.2) for index in range(5))
    step_two = tuple(_edge(index, index + 2, 0.4, step=2) for index in range(4))
    step_four = tuple(_edge(index, index + 4, 0.8, step=4) for index in range(2))
    progress = build_basic_s13_progress(frames, (*adjacent, *step_two, *step_four))
    assert progress.spatial is True
    assert np.allclose(progress.centers_x, (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    assert all(method == "grid_lk_subpixel" for method in progress.placement_methods[1:])
