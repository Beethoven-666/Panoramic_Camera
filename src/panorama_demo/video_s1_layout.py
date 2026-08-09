from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .video_s1_config import S1SourceSelectionConfig


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
