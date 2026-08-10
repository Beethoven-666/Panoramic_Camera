"""Dense real-source selection and fixed-midpoint owner schedule for S1.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .session import CameraIntrinsics
from .video_s12_schedule import S012Schedule, build_s012_schedule
from .video_s13_motion import S13MotionEdge, S13Progress


@dataclass(frozen=True)
class S13SourceSelection:
    frame_ids: tuple[int, ...]
    centers_x: tuple[float, ...]
    placement_methods: tuple[str, ...]
    risk_by_frame_id: Mapping[int, bool]


def select_dense_s13_sources(
    progress: S13Progress,
    edges: Sequence[S13MotionEdge],
    calibration: CameraIntrinsics,
    *,
    normal_target_advance_px: float = 8.0,
    risky_target_advance_px: float = 5.0,
) -> S13SourceSelection:
    if not progress.frame_ids:
        raise ValueError("S1.3 source selection requires real frames")
    risks = {edge.target_frame_id: edge.risk for edge in edges if edge.step == 1}
    selected = [0]
    last_center = progress.centers_x[0]
    for index in range(1, len(progress.frame_ids) - 1):
        target = risky_target_advance_px if risks.get(progress.frame_ids[index], False) else normal_target_advance_px
        if progress.centers_x[index] - last_center >= target:
            selected.append(index)
            last_center = progress.centers_x[index]
    if len(progress.frame_ids) > 1 and selected[-1] != len(progress.frame_ids) - 1:
        selected.append(len(progress.frame_ids) - 1)
    cx = float(calibration.cx)
    origin = float(progress.centers_x[selected[0]])
    return S13SourceSelection(
        frame_ids=tuple(progress.frame_ids[index] for index in selected),
        centers_x=tuple(cx + float(progress.centers_x[index]) - origin for index in selected),
        placement_methods=tuple(progress.placement_methods[index] for index in selected),
        risk_by_frame_id={progress.frame_ids[index]: bool(risks.get(progress.frame_ids[index], False)) for index in selected},
    )


def build_s13_midpoint_schedule(selection: S13SourceSelection, calibration: CameraIntrinsics) -> S012Schedule:
    # S1.3 M2 intentionally has no S1.2 24 px fatal gate. The 20% rescue/panel
    # policy belongs to M3 and is not implemented in this first round.
    return build_s012_schedule(selection.frame_ids, selection.centers_x, calibration, hard_internal_width_px=None)


__all__ = ["S13SourceSelection", "build_s13_midpoint_schedule", "select_dense_s13_sources"]
