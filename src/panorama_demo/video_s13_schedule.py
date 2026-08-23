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


@dataclass(frozen=True)
class S13SchedulePlan:
    selections: tuple[S13SourceSelection, ...]
    schedules: tuple[S012Schedule, ...]
    rescued_frame_ids: tuple[int, ...]
    density_scale: float
    source_u_cap_px: int
    panel_gap_frame_ids: tuple[tuple[int, int], ...]
    cap_satisfied: bool


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


def _scaled_selection(selection: S13SourceSelection, calibration: CameraIntrinsics, scale: float) -> S13SourceSelection:
    cx = float(calibration.cx)
    return S13SourceSelection(
        frame_ids=selection.frame_ids,
        centers_x=tuple(cx + (center - cx) * scale for center in selection.centers_x),
        placement_methods=selection.placement_methods,
        risk_by_frame_id=selection.risk_by_frame_id,
    )


def _unsafe_internal(schedule: S012Schedule, cap: int) -> tuple[int, ...]:
    return tuple(index for index, assignment in enumerate(schedule.assignments[1:-1], start=1)
                 if assignment.width > cap)


def plan_s13_m3_schedule(
    progress: S13Progress,
    edges: Sequence[S13MotionEdge],
    calibration: CameraIntrinsics,
    *,
    normal_target_advance_px: float = 8.0,
    risky_target_advance_px: float = 5.0,
    source_u_fraction_cap: float = 0.20,
    segment_break_pairs: Sequence[tuple[int, int]] = (),
) -> S13SchedulePlan:
    """Rescue real frames, reduce canvas density, then split panels if needed."""

    base = select_dense_s13_sources(
        progress, edges, calibration,
        normal_target_advance_px=normal_target_advance_px,
        risky_target_advance_px=risky_target_advance_px,
    )
    cap = max(1, int(float(calibration.width) * source_u_fraction_cap))
    selected_ids = list(base.frame_ids)
    rescued: list[int] = []
    progress_index = {frame_id: index for index, frame_id in enumerate(progress.frame_ids)}
    for left_id, right_id in segment_break_pairs:
        if left_id in progress_index and left_id not in selected_ids:
            selected_ids.append(left_id)
        if right_id in progress_index and right_id not in selected_ids:
            selected_ids.append(right_id)

    def rebuild(ids: Sequence[int]) -> S13SourceSelection:
        indices = sorted(progress_index[frame_id] for frame_id in ids)
        origin = progress.centers_x[indices[0]]
        cx = float(calibration.cx)
        risk = {edge.target_frame_id: edge.risk for edge in edges if edge.step == 1}
        return S13SourceSelection(
            frame_ids=tuple(progress.frame_ids[index] for index in indices),
            centers_x=tuple(cx + progress.centers_x[index] - origin for index in indices),
            placement_methods=tuple(progress.placement_methods[index] for index in indices),
            risk_by_frame_id={progress.frame_ids[index]: bool(risk.get(progress.frame_ids[index], False)) for index in indices},
        )

    selection = rebuild(selected_ids)
    schedule = build_s13_midpoint_schedule(selection, calibration)
    while _unsafe_internal(schedule, cap):
        added = False
        for assignment_index in _unsafe_internal(schedule, cap):
            left_index = progress_index[selection.frame_ids[assignment_index - 1]]
            right_index = progress_index[selection.frame_ids[assignment_index + 1]]
            candidates = [index for index in range(left_index + 1, right_index)
                          if progress.frame_ids[index] not in selected_ids]
            if candidates:
                target = 0.5 * (progress.centers_x[left_index] + progress.centers_x[right_index])
                chosen = min(candidates, key=lambda index: (abs(progress.centers_x[index] - target), index))
                frame_id = progress.frame_ids[chosen]
                selected_ids.append(frame_id)
                rescued.append(frame_id)
                added = True
        if not added:
            break
        selection = rebuild(selected_ids)
        schedule = build_s13_midpoint_schedule(selection, calibration)
    unsafe = _unsafe_internal(schedule, cap)
    density_scale = 1.0
    if unsafe:
        widest = max(schedule.assignments[index].width for index in unsafe)
        density_scale = min(1.0, float(cap) / float(widest))
        if density_scale >= 0.25:
            selection = _scaled_selection(selection, calibration, density_scale)
            schedule = build_s13_midpoint_schedule(selection, calibration)
            unsafe = _unsafe_internal(schedule, cap)
    forced_split_indices = tuple(
        sorted({
            selection.frame_ids.index(right_id)
            for _left_id, right_id in segment_break_pairs
            if right_id in selection.frame_ids and selection.frame_ids.index(right_id) > 0
        })
    )
    if not unsafe and not forced_split_indices:
        return S13SchedulePlan((selection,), (schedule,), tuple(sorted(rescued)), density_scale, cap, (), True)

    split_indices = tuple(sorted(set(unsafe) | set(forced_split_indices)))
    boundaries = (0, *split_indices, len(selection.frame_ids))
    panel_selections: list[S13SourceSelection] = []
    panel_schedules: list[S012Schedule] = []
    gaps: list[tuple[int, int]] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        panel_ids = selection.frame_ids[start:stop]
        panel_methods = selection.placement_methods[start:stop]
        panel_centers_raw = selection.centers_x[start:stop]
        cx = float(calibration.cx)
        panel_centers = tuple(cx + value - panel_centers_raw[0] for value in panel_centers_raw)
        panel = S13SourceSelection(panel_ids, panel_centers, panel_methods,
                                   {frame_id: selection.risk_by_frame_id[frame_id] for frame_id in panel_ids})
        panel_selections.append(panel)
        panel_schedules.append(build_s13_midpoint_schedule(panel, calibration))
        if stop < len(selection.frame_ids):
            gaps.append((selection.frame_ids[stop - 1], selection.frame_ids[stop]))
    return S13SchedulePlan(tuple(panel_selections), tuple(panel_schedules), tuple(sorted(rescued)), density_scale,
                           cap, tuple(gaps), True)


__all__ = [
    "S13SchedulePlan", "S13SourceSelection", "build_s13_midpoint_schedule", "plan_s13_m3_schedule",
    "select_dense_s13_sources",
]
