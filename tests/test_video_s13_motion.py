from __future__ import annotations

from panorama_demo.video_s13_motion import S13MotionEdge, build_basic_s13_progress
from panorama_demo.video_s13_session import S13RenderFrame


def _frame(frame_id: int) -> S13RenderFrame:
    from pathlib import Path

    return S13RenderFrame(frame_id, Path(f"{frame_id}.png"), frame_id, 64, 32)


def _edge(left: int, right: int, value: float | None) -> S13MotionEdge:
    return S13MotionEdge(
        left, right, 1, value, 0.0 if value is not None else None, 10.0 if value is not None else 0.0,
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
