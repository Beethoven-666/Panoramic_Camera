"""M3 coherent RGB lineage and non-metric layout cascade for S1.3."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .video_s13_motion import S13MotionEdge, S13MotionHypothesis, S13Progress
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


def _state_from_hypothesis(hypothesis: S13MotionHypothesis) -> _State:
    return _State(
        hypothesis_id=hypothesis.hypothesis_id,
        advance=max(0.0, float(hypothesis.advance_px)),
        vertical=float(hypothesis.vertical_px),
        method="motion_hypothesis",
        signature=hypothesis.support_signature_sha256,
        cells=frozenset(hypothesis.support_cells),
        centroid_x=float(hypothesis.centroid_x),
        central_support=float(hypothesis.central_support),
        confidence=float(hypothesis.confidence),
    )


def _fallback_state(edge: S13MotionEdge | None, session_median: float) -> _State:
    if edge is not None and edge.phase_advance_px is not None and edge.phase_response >= 0.05:
        return _State(-1, abs(float(edge.phase_advance_px)), float(edge.phase_vertical_px or 0.0),
                      "F1_phase_correlation", None, frozenset(), 0.5, 0.0,
                      min(1.0, float(edge.phase_response)))
    if edge is not None and edge.selected_advance_px is not None and math.isfinite(float(edge.selected_advance_px)):
        return _State(-1, abs(float(edge.selected_advance_px)), float(edge.lk_vertical_px or 0.0),
                      "F0_rgb_summary", None, frozenset(), 0.5, 0.0, 0.35)
    return _State(-1, session_median, 0.0, "F4_session_median", None, frozenset(), 0.5, 0.0, 0.05)


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
    scale = max(1.0, previous.advance, current.advance)
    advance_cost = abs(current.advance - previous.advance) / scale
    vertical_cost = 0.15 * abs(current.vertical - previous.vertical) / scale
    centroid_cost = 0.35 * abs(current.centroid_x - previous.centroid_x)
    support_cost = 0.4 * (1.0 - _jaccard(previous.cells, current.cells)) if previous.cells and current.cells else 0.15
    central_cost = 0.2 if previous.central_support > 0.2 and current.central_support < 0.05 else 0.0
    pose_cost = 0.0 if pose_direction is None or current.advance == 0.0 or pose_direction >= 0.0 else 0.5
    skip_cost = 0.0
    if skip_advance is not None:
        predicted = previous.advance + current.advance
        skip_cost = 0.5 * abs(predicted - skip_advance) / max(1.0, predicted, skip_advance)
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
    frames: Sequence[S13RenderFrame], edges: Sequence[S13MotionEdge], pose_progress: Mapping[int, float]
) -> tuple[S13LineageStep, ...]:
    adjacent = {(edge.source_frame_id, edge.target_frame_id): edge for edge in edges if edge.step == 1}
    skip_edges = {(edge.source_frame_id, edge.target_frame_id): edge for edge in edges if edge.step > 1}
    reliable = [abs(float(edge.selected_advance_px)) for edge in adjacent.values()
                if edge.selected_advance_px is not None and not edge.risk and abs(float(edge.selected_advance_px)) >= 0.25]
    session_median = float(np.median(reliable)) if reliable else 1.0
    state_rows: list[list[_State]] = []
    pairs: list[tuple[int, int]] = []
    for left, right in zip(frames[:-1], frames[1:]):
        edge = adjacent.get((left.frame_id, right.frame_id))
        states = [_state_from_hypothesis(item) for item in (edge.motion_hypotheses if edge else ())]
        if not states:
            states.append(_fallback_state(edge, session_median))
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
            abs(float(skip.selected_advance_px))
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


def build_s13_m3_layout(
    frames: Sequence[S13RenderFrame], edges: Sequence[S13MotionEdge], trajectory: S13Trajectory
) -> S13M3Layout:
    frame_ids = tuple(frame.frame_id for frame in frames)
    pose_progress, rail = _pose_rail(trajectory, frame_ids)
    lineage = _select_lineage(frames, edges, pose_progress)
    if not frames:
        raise ValueError("S1.3 M3 layout requires at least one real RGB frame")
    if len(frames) == 1:
        progress = S13Progress(frame_ids, (0.0,), ("L4_single_view",), False, False, False)
        return S13M3Layout(progress, (), "L4_single_or_contact", False, False, None, 0,
                           {"pose_rail": None, "lineage": []})
    rgb_steps = [step for step in lineage if step.placement_method in {"motion_hypothesis", "F1_phase_correlation", "F0_rgb_summary"}]
    rgb_coherent = len(rgb_steps) >= max(1, (len(frames) - 1) // 2)
    calibration: list[float] = []
    for step in lineage:
        left_pose, right_pose = pose_progress.get(step.source_frame_id), pose_progress.get(step.target_frame_id)
        if left_pose is None or right_pose is None or step.confidence < 0.25:
            continue
        distance_m = abs(right_pose - left_pose) / 1000.0
        if distance_m > 1e-6 and step.advance_px > 0.0:
            calibration.append(step.advance_px / distance_m)
    pixels_per_meter = float(np.median(calibration)) if len(calibration) >= 3 else None
    centers = [0.0]
    methods = ["origin"]
    for step in lineage:
        advance = max(0.0, step.advance_px)
        method = step.placement_method
        if pixels_per_meter is not None:
            left_pose, right_pose = pose_progress.get(step.source_frame_id), pose_progress.get(step.target_frame_id)
            if left_pose is not None and right_pose is not None:
                pose_advance = max(0.0, (right_pose - left_pose) / 1000.0 * pixels_per_meter)
                advance = 0.9 * advance + 0.1 * pose_advance
                method = f"{method}+pose_low_frequency_soft"
        centers.append(max(centers[-1], centers[-1] + advance))
        methods.append(method)
    if rgb_coherent:
        level = "L1_rgb_pose_soft" if pixels_per_meter is not None else "L0_coherent_rgb"
        spatial = centers[-1] > 0.25
    elif len(pose_progress) >= 2:
        # L2 is explicitly normalized and non-metric. Pose determines ordering,
        # never a unique metre-to-pixel conversion.
        ordered = np.asarray([pose_progress.get(frame_id, np.nan) for frame_id in frame_ids], dtype=np.float64)
        finite = np.isfinite(ordered)
        normalized = np.zeros(len(frames), dtype=np.float64)
        if finite.sum() >= 2:
            values = ordered[finite]
            span = max(1e-9, float(values.max() - values.min()))
            normalized[finite] = (values - values.min()) / span * max(8.0, 8.0 * (len(frames) - 1))
            normalized = np.maximum.accumulate(np.interp(np.arange(len(frames)), np.flatnonzero(finite), normalized[finite]))
        centers = normalized.tolist()
        methods = ["L2_normalized_pose_nonmetric"] * len(frames)
        methods[0] = "origin"
        level, spatial = "L2_normalized_pose", centers[-1] > 0.25
    else:
        centers = [float(index * 8) for index in range(len(frames))]
        methods = ["L3_temporal_slit"] * len(frames)
        methods[0] = "origin"
        level, spatial = "L3_temporal_slit", True
    progress = S13Progress(
        frame_ids=frame_ids, centers_x=tuple(centers), placement_methods=tuple(methods), spatial=spatial,
        coherent_rgb_motion=rgb_coherent,
        adjacent_reliable_graph_connected=(
            len([edge for edge in edges if edge.step == 1]) == len(frames) - 1
            and all(
                not edge.risk
                and edge.selected_advance_px is not None
                and abs(float(edge.selected_advance_px)) >= 0.25
                for edge in edges if edge.step == 1
            )
        ),
    )
    pose_supported = level == "L1_rgb_pose_soft" and len(trajectory.poses_by_frame_id) == len(frames)
    return S13M3Layout(
        progress=progress, lineage=lineage, layout_level=level, metric_scale_claim=False,
        pose_supported=pose_supported, pixels_per_meter=pixels_per_meter,
        pose_calibration_pairs=len(calibration),
        audit={
            "pose_rail": None if rail is None else rail.tolist(),
            "pose_direct_frame_count": len(pose_progress),
            "pose_calibration_pairs": len(calibration),
            "pixels_per_meter_rgb_calibrated": pixels_per_meter,
            "pose_is_soft_direction_low_frequency_only": True,
            "metric_scale_claim": False,
            "lineage": [asdict(step) for step in lineage],
        },
    )


__all__ = ["S13LineageStep", "S13M3Layout", "build_s13_m3_layout"]
