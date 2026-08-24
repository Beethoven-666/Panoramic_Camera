"""M3 coherent RGB lineage and non-metric layout cascade for S1.3."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .video_s13_motion import (
    S13MotionEdge,
    S13MotionHypothesis,
    S13Progress,
    reliable_step1_direction_evidence,
)
from .video_s13_session import S13RenderFrame
from .video_s13_trajectory import S13Trajectory


@dataclass(frozen=True)
class S13LineageStep:
    source_frame_id: int
    target_frame_id: int
    selected_hypothesis_id: int
    advance_px: float
    vertical_px: float
    placement_method: str
    support_signature_sha256: str | None
    switch_cost: float
    switch_reason: str
    confidence: float


@dataclass(frozen=True)
class S13M3Layout:
    progress: S13Progress
    lineage: tuple[S13LineageStep, ...]
    layout_level: str
    metric_scale_claim: bool
    pose_supported: bool
    pixels_per_meter: float | None
    pose_calibration_pairs: int
    canonical_scan_direction: int
    segment_break_pairs: tuple[tuple[int, int], ...]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class _State:
    hypothesis_id: int
    advance: float
    vertical: float
    method: str
    signature: str | None
    cells: frozenset[str]
    centroid_x: float
    central_support: float
    confidence: float


def _state_from_hypothesis(hypothesis: S13MotionHypothesis, canonical_direction: int) -> _State:
    return _State(
        hypothesis_id=hypothesis.hypothesis_id,
        advance=float(hypothesis.advance_px) * canonical_direction,
        vertical=float(hypothesis.vertical_px),
        method="motion_hypothesis",
        signature=hypothesis.support_signature_sha256,
        cells=frozenset(hypothesis.support_cells),
        centroid_x=float(hypothesis.centroid_x),
        central_support=float(hypothesis.central_support),
        confidence=float(hypothesis.confidence),
    )


def _confident_pause(edge: S13MotionEdge | None) -> bool:
    if edge is None or edge.lk_advance_px is None or edge.phase_advance_px is None:
        return False
    return bool(
        not edge.risk
        and edge.lk_observation_count >= 8
        and edge.lk_total_weight >= 2.0
        and edge.phase_response >= 0.05
        and abs(float(edge.lk_advance_px)) < 0.25
        and abs(float(edge.phase_advance_px)) < 0.25
        and abs(float(edge.lk_advance_px) - float(edge.phase_advance_px)) < 0.25
    )


def _fallback_state(edge: S13MotionEdge | None, session_median: float, canonical_direction: int) -> _State:
    if _confident_pause(edge):
        return _State(-1, 0.0, 0.0, "F5_zero_duplicate", None, frozenset(), 0.5, 0.0, 0.95)
    if edge is not None and edge.phase_advance_px is not None and edge.phase_response >= 0.05:
        return _State(-1, float(edge.phase_advance_px) * canonical_direction, float(edge.phase_vertical_px or 0.0),
                      "F1_phase_correlation", None, frozenset(), 0.5, 0.0,
                      min(1.0, float(edge.phase_response)))
    if edge is not None and edge.selected_advance_px is not None and math.isfinite(float(edge.selected_advance_px)):
        return _State(-1, float(edge.selected_advance_px) * canonical_direction, float(edge.lk_vertical_px or 0.0),
                      "F0_grid_lk_summary", None, frozenset(), 0.5, 0.0, 0.35)
    return _State(-1, session_median, 0.0, "F4_session_median", None, frozenset(), 0.5, 0.0, 0.05)


def _canonical_scan_direction(edges: Sequence[S13MotionEdge]) -> int:
    signed = list(reliable_step1_direction_evidence(edges))
    if not signed:
        signed = [
            float(edge.selected_advance_px)
            for edge in edges
            if edge.selected_advance_px is not None
            and math.isfinite(float(edge.selected_advance_px))
            and abs(float(edge.selected_advance_px)) >= 0.25
        ]
    if not signed:
        return 1
    median = float(np.median(np.asarray(signed, dtype=np.float64)))
    return -1 if median < 0.0 else 1


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right)) / float(len(left | right))


def _pose_rail(trajectory: S13Trajectory, frame_ids: Sequence[int]) -> tuple[dict[int, float], np.ndarray | None]:
    available = [(frame_id, trajectory.poses_by_frame_id[frame_id][:3, 3]) for frame_id in frame_ids
                 if frame_id in trajectory.poses_by_frame_id]
    if len(available) < 2:
        return {}, None
    centers = np.asarray([center for _, center in available], dtype=np.float64)
    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    rail = vh[0]
    projection = centered @ rail
    if projection[-1] < projection[0]:
        rail = -rail
        projection = -projection
    return {frame_id: float(value) for (frame_id, _), value in zip(available, projection, strict=True)}, rail


def _transition_cost(
    previous: _State,
    current: _State,
    pose_direction: float | None,
    skip_advance: float | None,
) -> tuple[float, str]:
    scale = max(1.0, abs(previous.advance), abs(current.advance))
    advance_cost = abs(current.advance - previous.advance) / scale
    vertical_cost = 0.15 * abs(current.vertical - previous.vertical) / scale
    centroid_cost = 0.35 * abs(current.centroid_x - previous.centroid_x)
    support_cost = 0.4 * (1.0 - _jaccard(previous.cells, current.cells)) if previous.cells and current.cells else 0.15
    central_cost = 0.2 if previous.central_support > 0.2 and current.central_support < 0.05 else 0.0
    pose_cost = 0.0 if pose_direction is None or current.advance == 0.0 or current.advance * pose_direction >= 0.0 else 0.5
    skip_cost = 0.0
    if skip_advance is not None:
        predicted = previous.advance + current.advance
        skip_cost = 0.5 * abs(predicted - skip_advance) / max(1.0, abs(predicted), abs(skip_advance))
    cost = advance_cost + vertical_cost + centroid_cost + support_cost + central_cost + pose_cost + skip_cost
    reasons = []
    if advance_cost > 0.5:
        reasons.append("advance_change")
    if support_cost > 0.25:
        reasons.append("support_change")
    if central_cost:
        reasons.append("central_support_disappeared")
    if pose_cost:
        reasons.append("pose_direction_conflict")
    if skip_cost > 0.2:
        reasons.append("skip_edge_cumulative_inconsistency")
    return cost, "+".join(reasons) if reasons else "coherent_continuation"


def _select_lineage(
    frames: Sequence[S13RenderFrame], edges: Sequence[S13MotionEdge], pose_progress: Mapping[int, float],
    canonical_direction: int,
) -> tuple[S13LineageStep, ...]:
    adjacent = {(edge.source_frame_id, edge.target_frame_id): edge for edge in edges if edge.step == 1}
    skip_edges = {(edge.source_frame_id, edge.target_frame_id): edge for edge in edges if edge.step > 1}
    reliable = [float(edge.selected_advance_px) * canonical_direction for edge in adjacent.values()
                if edge.selected_advance_px is not None and not edge.risk
                and float(edge.selected_advance_px) * canonical_direction >= 0.25]
    session_median = float(np.median(reliable)) if reliable else 1.0
    state_rows: list[list[_State]] = []
    pairs: list[tuple[int, int]] = []
    for left, right in zip(frames[:-1], frames[1:]):
        edge = adjacent.get((left.frame_id, right.frame_id))
        states = [] if _confident_pause(edge) else [
            _state_from_hypothesis(item, canonical_direction) for item in (edge.motion_hypotheses if edge else ())
        ]
        if not states:
            states.append(_fallback_state(edge, session_median, canonical_direction))
        state_rows.append(states)
        pairs.append((left.frame_id, right.frame_id))
    if not state_rows:
        return ()
    costs = [-math.log(max(1e-6, state.confidence)) for state in state_rows[0]]
    parents: list[list[int]] = [[-1] * len(state_rows[0])]
    reasons: list[list[str]] = [["lineage_origin"] * len(state_rows[0])]
    switch_costs: list[list[float]] = [[0.0] * len(state_rows[0])]
    for row_index in range(1, len(state_rows)):
        left_id, right_id = pairs[row_index]
        pose_direction = None
        if left_id in pose_progress and right_id in pose_progress:
            pose_direction = pose_progress[right_id] - pose_progress[left_id]
        skip = skip_edges.get((pairs[row_index - 1][0], right_id))
        skip_advance = (
            float(skip.selected_advance_px) * canonical_direction
            if skip is not None and skip.selected_advance_px is not None and math.isfinite(float(skip.selected_advance_px))
            else None
        )
        row_costs: list[float] = []
        row_parents: list[int] = []
        row_reasons: list[str] = []
        row_switch_costs: list[float] = []
        for current in state_rows[row_index]:
            options = []
            for parent_index, previous in enumerate(state_rows[row_index - 1]):
                transition, reason = _transition_cost(previous, current, pose_direction, skip_advance)
                options.append((costs[parent_index] + transition, parent_index, transition, reason))
            best = min(options, key=lambda item: (item[0], item[1]))
            row_costs.append(best[0] - math.log(max(1e-6, current.confidence)))
            row_parents.append(best[1])
            row_switch_costs.append(best[2])
            row_reasons.append(best[3])
        costs = row_costs
        parents.append(row_parents)
        reasons.append(row_reasons)
        switch_costs.append(row_switch_costs)
    selected = [0] * len(state_rows)
    selected[-1] = min(range(len(costs)), key=lambda index: (costs[index], index))
    for row_index in range(len(state_rows) - 1, 0, -1):
        selected[row_index - 1] = parents[row_index][selected[row_index]]
    return tuple(
        S13LineageStep(
            source_frame_id=pairs[index][0], target_frame_id=pairs[index][1],
            selected_hypothesis_id=state_rows[index][state_index].hypothesis_id,
            advance_px=state_rows[index][state_index].advance,
            vertical_px=state_rows[index][state_index].vertical,
            placement_method=state_rows[index][state_index].method,
            support_signature_sha256=state_rows[index][state_index].signature,
            switch_cost=switch_costs[index][state_index], switch_reason=reasons[index][state_index],
            confidence=state_rows[index][state_index].confidence,
        )
        for index, state_index in enumerate(selected)
    )


def _project_signed_lineage(
    lineage: Sequence[S13LineageStep],
) -> tuple[list[float], list[str], tuple[tuple[int, int], ...], dict[str, int]]:
    positive = [step.advance_px for step in lineage if step.advance_px >= 0.25]
    moving_median = float(np.median(positive)) if positive else 1.0
    reverse_limit = max(1.0, moving_median * 0.25)
    projected: list[float] = []
    methods: list[str] = []
    reverse_indices: list[int] = []
    counts = {"pause_zero": 0, "small_negative_projected": 0, "reverse_projected": 0}
    for index, step in enumerate(lineage):
        if abs(step.advance_px) < 0.25:
            projected.append(0.0)
            methods.append("F5_zero_duplicate" if step.placement_method != "F5_zero_duplicate" else step.placement_method)
            counts["pause_zero"] += 1
        elif step.advance_px < 0.0:
            projected.append(0.0)
            methods.append(f"{step.placement_method}+signed_monotone_projection")
            if abs(step.advance_px) <= reverse_limit:
                counts["small_negative_projected"] += 1
            else:
                reverse_indices.append(index)
                counts["reverse_projected"] += 1
        else:
            projected.append(float(step.advance_px))
            methods.append(step.placement_method)
    breaks: list[tuple[int, int]] = []
    run: list[int] = []
    for index in (*reverse_indices, -1):
        if run and index != run[-1] + 1:
            if len(run) >= 2:
                first = lineage[run[0]]
                breaks.append((first.source_frame_id, first.target_frame_id))
            run = []
        if index >= 0:
            run.append(index)
    return projected, methods, tuple(breaks), counts


def _pose_endpoint_calibration(
    frame_ids: Sequence[int], projected: Sequence[float], methods: Sequence[str],
    pose_progress: Mapping[int, float], trajectory: S13Trajectory,
) -> tuple[list[float], list[tuple[int, int, int, int, float]], str | None]:
    frame_index = {frame_id: index for index, frame_id in enumerate(frame_ids)}
    posed_ids = [frame_id for frame_id in frame_ids if frame_id in pose_progress]
    ratios: list[float] = []
    intervals: list[tuple[int, int, int, int, float]] = []
    reason: str | None = None
    for left_id, right_id in zip(posed_ids[:-1], posed_ids[1:]):
        if trajectory.pose_epoch_by_frame_id.get(left_id, "default") != trajectory.pose_epoch_by_frame_id.get(right_id, "default"):
            continue
        left, right = frame_index[left_id], frame_index[right_id]
        if right <= left:
            continue
        coherent = sum(
            projected[index] for index in range(left, right)
            if methods[index].startswith(("F0_", "F1_", "motion_hypothesis"))
        )
        distance_m = (pose_progress[right_id] - pose_progress[left_id]) / 1000.0
        if distance_m > 1e-6 and coherent > 0.25:
            ratios.append(float(coherent / distance_m))
            intervals.append((left, right, left_id, right_id, distance_m))
    if len(posed_ids) < 2:
        reason = "insufficient_direct_pose_endpoints"
    elif not ratios:
        reason = "no_coherent_rgb_advance_between_posed_endpoints"
    elif len(ratios) < 3:
        reason = "insufficient_valid_posed_endpoint_intervals"
    return ratios, intervals, reason


def build_s13_m3_layout(
    frames: Sequence[S13RenderFrame], edges: Sequence[S13MotionEdge], trajectory: S13Trajectory
) -> S13M3Layout:
    frame_ids = tuple(frame.frame_id for frame in frames)
    if not frames:
        raise ValueError("S1.3 M3 layout requires at least one real RGB frame")
    pose_progress, rail = _pose_rail(trajectory, frame_ids)
    canonical_direction = _canonical_scan_direction(edges)
    lineage = _select_lineage(frames, edges, pose_progress, canonical_direction)
    if len(frames) == 1:
        progress = S13Progress(frame_ids, (0.0,), ("L4_single_view",), False, False, False)
        return S13M3Layout(progress, (), "L4_single_or_contact", False, False, None, 0,
                           canonical_direction, (), {"pose_rail": None, "lineage": []})

    projected, edge_methods, segment_breaks, projection_counts = _project_signed_lineage(lineage)
    reliable_indices = [index for index, method in enumerate(edge_methods)
                        if method.startswith(("F0_", "F1_", "motion_hypothesis")) and projected[index] > 0.0]
    for index, method in enumerate(tuple(edge_methods)):
        if method != "F4_session_median":
            continue
        neighbours = [projected[other] for other in reliable_indices if 0 < abs(other - index) <= 2]
        if neighbours:
            projected[index] = float(np.median(neighbours))
            edge_methods[index] = "F3_local_median"

    rgb_steps = [index for index, method in enumerate(edge_methods)
                 if method.startswith(("F0_", "F1_", "motion_hypothesis"))]
    rgb_coherent = len(rgb_steps) >= max(1, (len(frames) - 1) // 2)
    ratios, pose_intervals, calibration_reason = _pose_endpoint_calibration(
        frame_ids, projected, edge_methods, pose_progress, trajectory
    )
    pixels_per_meter = float(np.median(ratios)) if len(ratios) >= 3 else None
    if pixels_per_meter is not None:
        calibration_reason = None
        for left, right, _left_id, _right_id, distance_m in pose_intervals:
            target = distance_m * pixels_per_meter
            missing = [index for index in range(left, right) if edge_methods[index].startswith(("F3_", "F4_"))]
            if missing:
                inferred = target / float(right - left)
                for index in missing:
                    projected[index] = max(0.0, inferred)
                    edge_methods[index] = "F2_rgb_calibrated_pose_trend"
            current = sum(projected[left:right])
            positive_total = sum(value for value in projected[left:right] if value > 0.0)
            if current > 0.0 and positive_total > 0.0:
                correction = 0.1 * (target - current)
                for index in range(left, right):
                    if projected[index] > 0.0:
                        projected[index] = max(0.0, projected[index] + correction * projected[index] / positive_total)
                        edge_methods[index] = f"{edge_methods[index]}+pose_low_frequency_soft"

    if rgb_coherent:
        level = "L1_rgb_pose_soft" if pixels_per_meter is not None else "L0_coherent_rgb"
        centers = [0.0]
        for advance in projected:
            centers.append(centers[-1] + advance)
        methods = ["origin", *edge_methods]
        spatial = centers[-1] > 0.25
    elif len(pose_progress) >= 2:
        ordered = np.asarray([pose_progress.get(frame_id, np.nan) for frame_id in frame_ids], dtype=np.float64)
        finite = np.isfinite(ordered)
        normalized = np.zeros(len(frames), dtype=np.float64)
        values = ordered[finite]
        span = max(1e-9, float(values.max() - values.min()))
        normalized[finite] = (values - values.min()) / span * max(8.0, 8.0 * (len(frames) - 1))
        normalized = np.maximum.accumulate(np.interp(np.arange(len(frames)), np.flatnonzero(finite), normalized[finite]))
        centers = normalized.tolist()
        methods = ["F6_normalized_pose"] * len(frames)
        methods[0] = "origin"
        level, spatial = "L2_normalized_pose", centers[-1] > 0.25
    else:
        centers = [0.0] * len(frames)
        methods = ["F7_temporal_order"] * len(frames)
        methods[0] = "origin"
        level, spatial = "L3_temporal_slit", False

    adjacent_edges = [edge for edge in edges if edge.step == 1]
    progress = S13Progress(
        frame_ids=frame_ids, centers_x=tuple(centers), placement_methods=tuple(methods), spatial=spatial,
        coherent_rgb_motion=rgb_coherent,
        adjacent_reliable_graph_connected=(
            len(adjacent_edges) == len(frames) - 1
            and all(not edge.risk and edge.selected_advance_px is not None
                    and float(edge.selected_advance_px) * canonical_direction >= 0.25 for edge in adjacent_edges)
        ),
    )
    pose_supported = level == "L1_rgb_pose_soft" and len(trajectory.poses_by_frame_id) == len(frames)
    return S13M3Layout(
        progress=progress, lineage=lineage, layout_level=level, metric_scale_claim=False,
        pose_supported=pose_supported, pixels_per_meter=pixels_per_meter,
        pose_calibration_pairs=len(ratios), canonical_scan_direction=canonical_direction,
        segment_break_pairs=segment_breaks,
        audit={
            "pose_rail": None if rail is None else rail.tolist(),
            "pose_direct_frame_count": len(pose_progress),
            "pose_calibration_pairs": len(ratios),
            "pose_calibration_reason": calibration_reason,
            "pose_calibration_uses_posed_endpoint_cumulative_rgb": True,
            "pose_interpolation_used": False,
            "pixels_per_meter_rgb_calibrated": pixels_per_meter,
            "pose_is_soft_direction_low_frequency_only": True,
            "metric_scale_claim": False,
            "canonical_scan_direction": canonical_direction,
            "signed_lineage_advance_px": [step.advance_px for step in lineage],
            "projected_monotone_advance_px": projected,
            "segment_break_pairs": [list(pair) for pair in segment_breaks],
            "projection_counts": projection_counts,
            "lineage": [asdict(step) for step in lineage],
        },
    )


def selected_hypothesis_ids_for_spatial_sources(
    layout: S13M3Layout,
    frame_ids: Sequence[int],
) -> tuple[int, ...]:
    """Map temporal M3 edge hypotheses onto canvas-spatial source order."""

    ids = tuple(int(frame_id) for frame_id in frame_ids)
    if not ids:
        return ()
    by_temporal_target = {
        int(step.target_frame_id): int(step.selected_hypothesis_id)
        for step in layout.lineage
    }
    if layout.canonical_scan_direction > 0:
        return tuple(by_temporal_target.get(frame_id, -1) for frame_id in ids)
    return (
        -1,
        *(
            by_temporal_target.get(previous_spatial_id, -1)
            for previous_spatial_id in ids[:-1]
        ),
    )


__all__ = [
    "S13LineageStep", "S13M3Layout", "build_s13_m3_layout",
    "selected_hypothesis_ids_for_spatial_sources",
]
