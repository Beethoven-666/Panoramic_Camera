from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .video_s1_config import S1SourceSelectionConfig, S11SourceSelectionConfig


@dataclass(frozen=True)
class S1MotionEdge:
    dx: float
    dy: float = 0.0
    inlier_ratio: float = 1.0
    grid_coverage: float = 1.0
    texture_coverage: float = 1.0
    reliable: bool = True


@dataclass(frozen=True)
class S1SourceLayout:
    frame_id: int
    source_index: int
    canvas_center_x: float
    owner_left_x: int
    owner_right_x: int
    support_left_x: int
    support_right_x: int
    match_left_x: int
    match_right_x: int
    measured_progress_from_previous_px: float
    source_quality: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.owner_left_x >= self.owner_right_x:
            raise ValueError("owner_left_x must be less than owner_right_x")
        if self.support_left_x > self.owner_left_x or self.support_right_x < self.owner_right_x:
            raise ValueError("owner must be contained by source support")
        if not self.support_left_x <= self.match_left_x <= self.owner_left_x:
            raise ValueError("left match shoulder is outside source support")
        if not self.owner_right_x <= self.match_right_x <= self.support_right_x:
            raise ValueError("right match shoulder is outside source support")


@dataclass(frozen=True)
class S1PairOverlap:
    pair_index: int
    left_frame_id: int
    right_frame_id: int
    owner_boundary_x: int
    roi_left_x: int
    roi_right_x: int
    per_row_valid_left_x: np.ndarray
    per_row_valid_right_x: np.ndarray
    common_valid_mask: np.ndarray
    overlap_width_p5: float
    overlap_width_p50: float
    overlap_width_p95: float
    s1_evaluable_fraction: float


@dataclass(frozen=True)
class S1PairLineage:
    """Stable identity of an original adjacent S01 source pair."""

    origin_left_frame_id: int
    origin_right_frame_id: int

    def __post_init__(self) -> None:
        if self.origin_left_frame_id == self.origin_right_frame_id:
            raise ValueError("origin pair lineage endpoints must be distinct")

    @property
    def frame_ids(self) -> tuple[int, int]:
        return (self.origin_left_frame_id, self.origin_right_frame_id)


@dataclass(frozen=True)
class S1SourceRescue:
    """One real-frame insertion selected inside an original S01 pair lineage."""

    frame_index: int
    frame_id: int
    origin_pair_lineage: S1PairLineage
    refinement_round: int
    parent_left_frame_id: int
    parent_right_frame_id: int
    left_child_step_fullres_px: float
    right_child_step_fullres_px: float
    maximum_child_step_fullres_px: float
    improvement_fullres_px: float
    quality_penalty: float


def _layout_from_baseline(value: S1SourceLayout | Mapping[str, Any]) -> S1SourceLayout:
    if isinstance(value, S1SourceLayout):
        fields = {
            "frame_id": value.frame_id,
            "source_index": value.source_index,
            "canvas_center_x": value.canvas_center_x,
            "owner_left_x": value.owner_left_x,
            "owner_right_x": value.owner_right_x,
            "support_left_x": value.support_left_x,
            "support_right_x": value.support_right_x,
            "match_left_x": value.match_left_x,
            "match_right_x": value.match_right_x,
            "measured_progress_from_previous_px": value.measured_progress_from_previous_px,
            "source_quality": dict(value.source_quality),
        }
    elif isinstance(value, Mapping):
        required = {
            "frame_id",
            "source_index",
            "canvas_center_x",
            "owner_left_x",
            "owner_right_x",
            "support_left_x",
            "support_right_x",
            "match_left_x",
            "match_right_x",
            "measured_progress_from_previous_px",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"baseline source layout is missing fields: {sorted(missing)}")
        fields = {name: value[name] for name in required}
        fields["source_quality"] = dict(value.get("source_quality", {}))
    else:
        raise TypeError("baseline source layout must be an S1SourceLayout or mapping")
    return S1SourceLayout(**fields)


def inherit_s01_v1_exact(
    baseline_layouts: Sequence[S1SourceLayout | Mapping[str, Any]],
    frame_ids: Sequence[int],
    *,
    expected_canvas_width: int | None = None,
) -> tuple[S1SourceLayout, ...]:
    """Validate and copy the exact S01 layout without reselecting or relaying it out.

    The copied fields include frame/source identity, centers, supports, initial midpoint
    owners and shoulders. This deliberately has no source-selection configuration: S1.1
    is not allowed to approximate the baseline by running a second selector.
    """
    if not baseline_layouts:
        raise ValueError("S01 baseline source layout cannot be empty")
    copied = tuple(_layout_from_baseline(item) for item in baseline_layouts)
    validate_owner_partition(copied)
    if copied[0].source_index != 0 or copied[-1].source_index != len(frame_ids) - 1:
        raise ValueError("S01 baseline must retain the first and last real frames")
    for item in copied:
        if item.source_index < 0 or item.source_index >= len(frame_ids):
            raise ValueError("S01 baseline source index is outside the current frame sequence")
        if int(frame_ids[item.source_index]) != item.frame_id:
            raise ValueError("S01 baseline frame ID does not match the current frame sequence")
        if not np.isfinite(item.canvas_center_x):
            raise ValueError("S01 baseline canvas centers must be finite")
    centers = np.asarray([item.canvas_center_x for item in copied], dtype=np.float64)
    if np.any(np.diff(centers) < 0):
        raise ValueError("S01 baseline canvas centers must be monotonic")
    if copied[0].owner_left_x != copied[0].support_left_x:
        raise ValueError("S01 baseline first owner must retain the complete left endpoint")
    if copied[-1].owner_right_x != copied[-1].support_right_x:
        raise ValueError("S01 baseline last owner must retain the complete right endpoint")
    for index, (left, right) in enumerate(zip(copied[:-1], copied[1:]), start=1):
        expected_midpoint = int(np.floor((left.canvas_center_x + right.canvas_center_x) * 0.5 + 0.5))
        if left.owner_right_x != expected_midpoint or right.owner_left_x != expected_midpoint:
            raise ValueError(f"S01 baseline owner boundary {index} is not its exact midpoint")
    canvas_width = copied[-1].support_right_x - copied[0].support_left_x
    if canvas_width <= 0:
        raise ValueError("S01 baseline canvas extent must be positive")
    if expected_canvas_width is not None and canvas_width != expected_canvas_width:
        raise ValueError("S01 baseline canvas width does not match the expected canvas width")
    return copied


def progress_analysis_to_full_resolution(
    progress_analysis_px: Sequence[float], *, calibrated_width: int, analysis_width: int
) -> np.ndarray:
    """Convert scan progress units before any S1.1 rescue ranking."""
    if calibrated_width <= 0 or analysis_width <= 0:
        raise ValueError("calibrated and analysis widths must be positive")
    progress = np.asarray(progress_analysis_px, dtype=np.float64)
    if progress.ndim != 1 or progress.size == 0 or not np.all(np.isfinite(progress)):
        raise ValueError("analysis progress must be a non-empty finite one-dimensional sequence")
    if np.any(np.diff(progress) < 0):
        raise ValueError("analysis progress must be monotonic")
    return progress * (float(calibrated_width) / float(analysis_width))


def select_local_real_source_rescue(
    frame_ids: Sequence[int],
    progress_analysis_px: Sequence[float],
    left_source_index: int,
    right_source_index: int,
    config: S11SourceSelectionConfig,
    *,
    calibrated_width: int,
    analysis_width: int,
    origin_pair_lineage: S1PairLineage | tuple[int, int] | None = None,
    refinement_round: int = 1,
    prior_inserted_frame_indices: Sequence[int] = (),
    rgb_available: Sequence[bool] | None = None,
    source_quality_penalties: Sequence[float] | None = None,
) -> S1SourceRescue | None:
    """Choose one local, on-disk real frame using full-resolution scan progress.

    The caller is responsible for invoking this only for an evaluable hard-risk pair
    with no safe straight boundary. Returning one insertion at a time lets the caller
    rebuild child pairs and all evidence before a later refinement round.
    """
    count = len(frame_ids)
    if len(progress_analysis_px) != count:
        raise ValueError("scan progress must align with the real frame sequence")
    if not 0 <= left_source_index < right_source_index < count:
        raise ValueError("rescue parent source indices must be ordered and in range")
    if refinement_round < 1 or refinement_round > config.maximum_refinement_rounds:
        return None
    prior = tuple(int(index) for index in prior_inserted_frame_indices)
    if len(prior) >= config.maximum_insertions_per_original_pair:
        return None
    if len(set(prior)) != len(prior):
        raise ValueError("prior inserted frame indices must be unique")
    if rgb_available is not None and len(rgb_available) != count:
        raise ValueError("rgb availability must align with the real frame sequence")
    if source_quality_penalties is not None and len(source_quality_penalties) != count:
        raise ValueError("source quality penalties must align with the real frame sequence")

    progress = progress_analysis_to_full_resolution(
        progress_analysis_px, calibrated_width=calibrated_width, analysis_width=analysis_width
    )
    parent_step = float(progress[right_source_index] - progress[left_source_index])
    lineage = origin_pair_lineage
    if lineage is None:
        lineage = S1PairLineage(
            int(frame_ids[left_source_index]), int(frame_ids[right_source_index])
        )
    elif not isinstance(lineage, S1PairLineage):
        lineage = S1PairLineage(int(lineage[0]), int(lineage[1]))
    try:
        origin_left_index = frame_ids.index(lineage.origin_left_frame_id)
        origin_right_index = frame_ids.index(lineage.origin_right_frame_id)
    except (AttributeError, ValueError):
        positions = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
        if (
            lineage.origin_left_frame_id not in positions
            or lineage.origin_right_frame_id not in positions
        ):
            raise ValueError("origin pair lineage is outside the current frame sequence") from None
        origin_left_index = positions[lineage.origin_left_frame_id]
        origin_right_index = positions[lineage.origin_right_frame_id]
    if any(index <= origin_left_index or index >= origin_right_index for index in prior):
        raise ValueError("prior insertion does not belong to the origin pair lineage")

    ranked: list[tuple[tuple[float, float, float, int], S1SourceRescue]] = []
    prior_set = set(prior)
    for candidate in range(left_source_index + 1, right_source_index):
        if candidate in prior_set or (rgb_available is not None and not rgb_available[candidate]):
            continue
        left_step = float(progress[candidate] - progress[left_source_index])
        right_step = float(progress[right_source_index] - progress[candidate])
        maximum_child = max(left_step, right_step)
        improvement = parent_step - maximum_child
        if improvement + 1e-12 < config.minimum_rescue_improvement_px:
            continue
        quality_penalty = (
            0.0 if source_quality_penalties is None else float(source_quality_penalties[candidate])
        )
        if not np.isfinite(quality_penalty):
            continue
        result = S1SourceRescue(
            frame_index=candidate,
            frame_id=int(frame_ids[candidate]),
            origin_pair_lineage=lineage,
            refinement_round=refinement_round,
            parent_left_frame_id=int(frame_ids[left_source_index]),
            parent_right_frame_id=int(frame_ids[right_source_index]),
            left_child_step_fullres_px=left_step,
            right_child_step_fullres_px=right_step,
            maximum_child_step_fullres_px=maximum_child,
            improvement_fullres_px=improvement,
            quality_penalty=quality_penalty,
        )
        rank = (maximum_child, abs(left_step - right_step), quality_penalty, candidate)
        ranked.append((rank, result))
    return min(ranked, key=lambda item: item[0])[1] if ranked else None


def insert_local_real_source_rescue(
    source_indices: Sequence[int], rescue: S1SourceRescue
) -> tuple[int, ...]:
    """Insert a selected real frame while proving endpoint and time-order invariants."""
    original = tuple(int(index) for index in source_indices)
    if not original or tuple(sorted(set(original))) != original:
        raise ValueError("source indices must be strictly increasing and unique")
    if rescue.frame_index in original:
        raise ValueError("rescue frame is already a selected source")
    refined = tuple(sorted((*original, rescue.frame_index)))
    if refined[0] != original[0] or refined[-1] != original[-1]:
        raise ValueError("local rescue cannot change the S01 endpoint sources")
    return refined


def validate_refined_layout_invariants(
    initial_layouts: Sequence[S1SourceLayout], refined_layouts: Sequence[S1SourceLayout]
) -> None:
    """Prove that local rescue kept S01 endpoint sources and the canvas extent."""
    if not initial_layouts or not refined_layouts:
        raise ValueError("initial and refined source layouts must be non-empty")
    validate_owner_partition(initial_layouts)
    validate_owner_partition(refined_layouts)
    initial_endpoints = (
        initial_layouts[0].frame_id,
        initial_layouts[-1].frame_id,
        initial_layouts[0].source_index,
        initial_layouts[-1].source_index,
    )
    refined_endpoints = (
        refined_layouts[0].frame_id,
        refined_layouts[-1].frame_id,
        refined_layouts[0].source_index,
        refined_layouts[-1].source_index,
    )
    if refined_endpoints != initial_endpoints:
        raise ValueError("refined layout changed the exact S01 endpoint sources")
    initial_extent = (
        initial_layouts[0].support_left_x,
        initial_layouts[-1].support_right_x,
    )
    refined_extent = (
        refined_layouts[0].support_left_x,
        refined_layouts[-1].support_right_x,
    )
    if refined_extent != initial_extent:
        raise ValueError("refined layout changed the exact S01 canvas extent")
    if refined_layouts[0].owner_left_x != refined_layouts[0].support_left_x:
        raise ValueError("refined layout cropped the complete left endpoint")
    if refined_layouts[-1].owner_right_x != refined_layouts[-1].support_right_x:
        raise ValueError("refined layout cropped the complete right endpoint")


def cumulative_monotonic_progress(
    edges: Sequence[S1MotionEdge], *, direction: int
) -> np.ndarray:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    progress = np.zeros(len(edges) + 1, dtype=np.float64)
    for index, edge in enumerate(edges, start=1):
        advance = direction * float(edge.dx)
        if not np.isfinite(advance):
            raise ValueError("motion dx must be finite")
        progress[index] = max(progress[index - 1], progress[index - 1] + advance)
    return progress


def build_monotonic_progress(
    motions: Sequence[S1MotionEdge], scan_direction: int
) -> np.ndarray:
    return cumulative_monotonic_progress(motions, direction=scan_direction)


def select_dynamic_real_sources(
    frame_ids: Sequence[int],
    edges: Sequence[S1MotionEdge],
    config: S1SourceSelectionConfig,
    *,
    direction: int,
    risk_edges: Sequence[bool] | None = None,
    source_qualities: Sequence[Mapping[str, float]] | None = None,
) -> tuple[int, ...]:
    """Return ordered indices of real frames selected by cumulative canvas motion."""
    if len(frame_ids) == 0:
        raise ValueError("at least one real frame is required")
    if len(edges) != len(frame_ids) - 1:
        raise ValueError("one motion edge is required between every adjacent real frame")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("frame IDs must be unique")
    if risk_edges is not None and len(risk_edges) != len(edges):
        raise ValueError("risk_edges must align with motion edges")
    if source_qualities is not None and len(source_qualities) != len(frame_ids):
        raise ValueError("source_qualities must align with real frames")
    if len(frame_ids) == 1:
        return (0,)

    progress = cumulative_monotonic_progress(edges, direction=direction)
    selected = [0]
    for index in range(1, len(frame_ids) - 1):
        edge = edges[index - 1]
        is_risk = (not edge.reliable) or bool(risk_edges[index - 1] if risk_edges is not None else False)
        target = config.risk_target_step_px if is_risk else config.normal_target_step_px
        since_last = progress[index] - progress[selected[-1]]
        edge_advance = progress[index] - progress[index - 1]
        next_advance = progress[index + 1] - progress[index]
        waiting_overshoots = since_last + next_advance > config.maximum_preferred_step_px
        forced = (
            edge_advance >= config.emergency_step_px
            or waiting_overshoots
            or (is_risk and config.adjacent_frame_rescue)
        )
        if since_last >= target or forced:
            candidate = index
            prior = index - 1
            if not forced and prior > selected[-1] and source_qualities is not None:
                def score(frame_index: int) -> tuple[float, float]:
                    quality = source_qualities[frame_index]
                    exposure_penalty = float(quality.get("dark_ratio", 0.0)) + float(
                        quality.get("saturated_ratio", 0.0)
                    )
                    evidence = (
                        0.002 * float(quality.get("sharpness", 0.0))
                        + float(quality.get("texture_coverage", 0.0))
                        - exposure_penalty
                    )
                    closeness = abs((progress[frame_index] - progress[selected[-1]]) - target)
                    return evidence, -float(closeness)

                candidate = max((prior, index), key=score)
            if candidate > selected[-1]:
                selected.append(candidate)
    if selected[-1] != len(frame_ids) - 1:
        selected.append(len(frame_ids) - 1)
    return tuple(selected)


def select_dynamic_sources(
    frame_ids: Sequence[int],
    motions: Sequence[S1MotionEdge],
    config: S1SourceSelectionConfig,
    *,
    scan_direction: int,
    risk_edges: Sequence[bool] | None = None,
    source_qualities: Sequence[Mapping[str, float]] | None = None,
) -> tuple[int, ...]:
    return select_dynamic_real_sources(
        frame_ids,
        motions,
        config,
        direction=scan_direction,
        risk_edges=risk_edges,
        source_qualities=source_qualities,
    )


def build_source_layout(
    frame_ids: Sequence[int],
    source_indices: Sequence[int],
    canvas_centers_x: Sequence[float],
    support_bounds: Sequence[tuple[int, int]],
    config: S1SourceSelectionConfig,
    *,
    source_qualities: Sequence[Mapping[str, float]] | None = None,
) -> tuple[S1SourceLayout, ...]:
    count = len(source_indices)
    if not (count == len(canvas_centers_x) == len(support_bounds)) or count == 0:
        raise ValueError("source layout inputs must have the same non-zero length")
    if tuple(source_indices) != tuple(sorted(set(source_indices))):
        raise ValueError("source indices must be strictly increasing and unique")
    if source_indices[0] < 0 or source_indices[-1] >= len(frame_ids):
        raise ValueError("source index is outside the real frame sequence")
    centers = np.asarray(canvas_centers_x, dtype=np.float64)
    if not np.all(np.isfinite(centers)) or np.any(np.diff(centers) < 0):
        raise ValueError("canvas source centers must be finite and monotonic")
    qualities = source_qualities or tuple({} for _ in range(count))
    if len(qualities) != count:
        raise ValueError("source_qualities must align with selected sources")

    boundaries = [int(support_bounds[0][0])]
    for left, right in zip(centers[:-1], centers[1:]):
        boundaries.append(int(np.floor((left + right) * 0.5 + 0.5)))
    boundaries.append(int(support_bounds[-1][1]))
    for index in range(1, len(boundaries)):
        if boundaries[index] <= boundaries[index - 1]:
            boundaries[index] = boundaries[index - 1] + 1

    layouts: list[S1SourceLayout] = []
    for position, source_index in enumerate(source_indices):
        support_left, support_right = (int(value) for value in support_bounds[position])
        owner_left, owner_right = boundaries[position], boundaries[position + 1]
        if support_left > owner_left or support_right < owner_right:
            raise ValueError("selected source support cannot cover its owner interval")
        left_available = owner_left - support_left
        right_available = support_right - owner_right
        left_shoulder = min(config.shoulder_target_px, config.shoulder_maximum_px, left_available)
        right_shoulder = min(config.shoulder_target_px, config.shoulder_maximum_px, right_available)
        previous_progress = 0.0 if position == 0 else float(centers[position] - centers[position - 1])
        layouts.append(S1SourceLayout(
            frame_id=int(frame_ids[source_index]),
            source_index=int(source_index),
            canvas_center_x=float(centers[position]),
            owner_left_x=owner_left,
            owner_right_x=owner_right,
            support_left_x=support_left,
            support_right_x=support_right,
            match_left_x=owner_left - left_shoulder,
            match_right_x=owner_right + right_shoulder,
            measured_progress_from_previous_px=previous_progress,
            source_quality=dict(qualities[position]),
        ))
    return tuple(layouts)


def build_source_layouts(
    frame_ids: Sequence[int],
    source_indices: Sequence[int],
    canvas_centers_x: Sequence[float],
    support_bounds: Sequence[tuple[int, int]],
    config: S1SourceSelectionConfig,
    *,
    source_qualities: Sequence[Mapping[str, float]] | None = None,
) -> tuple[S1SourceLayout, ...]:
    return build_source_layout(
        frame_ids, source_indices, canvas_centers_x, support_bounds, config,
        source_qualities=source_qualities,
    )


def validate_owner_partition(layouts: Sequence[S1SourceLayout]) -> None:
    if not layouts:
        raise ValueError("source layout cannot be empty")
    for left, right in zip(layouts[:-1], layouts[1:]):
        if left.source_index >= right.source_index:
            raise ValueError("source IDs are not time ordered")
        if left.owner_right_x != right.owner_left_x:
            raise ValueError("owner intervals must have neither gaps nor overlaps")
