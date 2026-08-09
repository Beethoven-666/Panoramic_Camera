"""Sparse horizontal evidence and straight-owner-boundary solving for S1.1.

All coordinates carried by public evidence objects are explicit.  Sparse LK is
performed in the horizontally reduced calibrated target image (``A``), then
converted through the full-resolution calibrated image (``C``) to the panorama
canvas (``X``).  The module never warps RGB and held-out observations never
enter a solver decision.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HandoffCoordinateTransform:
    analysis_width: int
    calibrated_width: int
    support_left_x: float

    def __post_init__(self) -> None:
        if self.analysis_width <= 0 or self.calibrated_width <= 0:
            raise ValueError("coordinate widths must be positive")
        if not np.isfinite(self.support_left_x):
            raise ValueError("support_left_x must be finite")

    @property
    def scale_x(self) -> float:
        return self.calibrated_width / self.analysis_width

    def analysis_to_calibrated_x(self, x: float) -> float:
        """Convert pixel centres from A to C."""
        return (float(x) + 0.5) * self.scale_x - 0.5

    def calibrated_to_analysis_x(self, x: float) -> float:
        return (float(x) + 0.5) / self.scale_x - 0.5

    def calibrated_to_canvas_x(self, x: float) -> float:
        return self.support_left_x + float(x)

    def analysis_to_canvas_x(self, x: float) -> float:
        return self.calibrated_to_canvas_x(self.analysis_to_calibrated_x(x))


@dataclass(frozen=True)
class HandoffObservation:
    match_id: str
    origin_pair_lineage: str
    left_frame_id: int
    right_frame_id: int
    role: str
    left_analysis_x: float
    left_y: float
    right_analysis_x: float
    right_y: float
    left_calibrated_x: float
    right_calibrated_x: float
    left_canvas_x: float
    right_canvas_x: float
    forward_backward_error_px: float
    vertical_error_px: float
    texture_score: float
    observation_canvas_dx: float
    background_canvas_dx: float = 0.0
    relative_dx: float = 0.0
    depth_risk_weight: float = 1.0
    left_depth_mm: float | None = None
    right_depth_mm: float | None = None
    left_depth_edge: bool = False
    right_depth_edge: bool = False

    def __post_init__(self) -> None:
        if self.role not in {"unassigned", "solver", "heldout"}:
            raise ValueError("observation role must be unassigned, solver, or heldout")
        numeric = (
            self.left_analysis_x, self.left_y, self.right_analysis_x, self.right_y,
            self.left_calibrated_x, self.right_calibrated_x,
            self.left_canvas_x, self.right_canvas_x,
            self.forward_backward_error_px, self.vertical_error_px,
            self.texture_score, self.observation_canvas_dx,
            self.background_canvas_dx, self.relative_dx, self.depth_risk_weight,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("handoff observation values must be finite")
        for depth in (self.left_depth_mm, self.right_depth_mm):
            if depth is not None and (not np.isfinite(depth) or depth <= 0.0):
                raise ValueError("attached aligned depth must be finite positive millimetres")

    @property
    def interval(self) -> tuple[float, float]:
        return (
            min(self.left_canvas_x, self.right_canvas_x),
            max(self.left_canvas_x, self.right_canvas_x),
        )


@dataclass(frozen=True)
class HandoffEvidenceSplit:
    solver: tuple[HandoffObservation, ...]
    heldout: tuple[HandoffObservation, ...]


@dataclass(frozen=True)
class ForbiddenInterval:
    left_x: float
    right_x: float
    cluster_id: int
    match_ids: tuple[str, ...]
    match_count: int
    vertical_span_px: float
    median_relative_dx_px: float
    residual_sign: int

    def __post_init__(self) -> None:
        if not self.left_x < self.right_x:
            raise ValueError("forbidden interval must have positive width")
        if self.match_count != len(self.match_ids) or self.match_count < 1:
            raise ValueError("forbidden interval match audit is inconsistent")
        if self.residual_sign not in (-1, 1):
            raise ValueError("forbidden interval residual sign must be -1 or 1")

    def contains(self, canvas_x: float) -> bool:
        """Return half-open membership: ``left <= x < right``."""
        return self.left_x <= float(canvas_x) < self.right_x


@dataclass(frozen=True)
class BoundaryCandidate:
    x: int
    total_cost: float
    hard_forbidden_crossing: bool
    soft_crossing_cost: float = 0.0
    rgb_edge_cost: float = 0.0
    depth_edge_cost: float = 0.0
    midpoint_distance: float = 0.0


@dataclass(frozen=True)
class PairBoundaryDecision:
    pair_index: int
    midpoint_x: int
    boundary_x: int
    classification: str
    needs_s2: bool
    manual_review_required: bool
    protected_crossing: bool


@dataclass(frozen=True)
class StraightBoundarySolution:
    boundaries: tuple[int, ...]
    decisions: tuple[PairBoundaryDecision, ...]
    feasible: bool
    total_cost: float


def analysis_to_calibrated_x(x: float, analysis_width: int, calibrated_width: int) -> float:
    return HandoffCoordinateTransform(analysis_width, calibrated_width, 0.0).analysis_to_calibrated_x(x)


def calibrated_to_canvas_x(x: float, support_left_x: float) -> float:
    return float(support_left_x) + float(x)


def _stable_match_id(
    calibration_id: str,
    origin_pair_lineage: str,
    left_frame_id: int,
    right_frame_id: int,
    left_c: tuple[float, float],
    right_c: tuple[float, float],
) -> str:
    quarter = tuple(int(np.rint(value * 4.0)) for value in (*left_c, *right_c))
    canonical = "|".join((
        calibration_id, origin_pair_lineage, str(int(left_frame_id)),
        str(int(right_frame_id)), *(str(value) for value in quarter),
    ))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def extract_calibrated_handoff_observations(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    left_transform: HandoffCoordinateTransform,
    right_transform: HandoffCoordinateTransform,
    *,
    calibration_id: str,
    origin_pair_lineage: str,
    left_frame_id: int,
    right_frame_id: int,
    corridor_canvas: tuple[float, float],
    maximum_features: int = 500,
    quality_level: float = 0.01,
    minimum_distance_px: float = 5.0,
    lk_forward_backward_max_px: float = 0.75,
    maximum_vertical_match_error_px: float = 4.0,
) -> tuple[HandoffObservation, ...]:
    """Track calibrated A-domain GFTT points with candidate-specific LK F/B."""
    left = np.asarray(left_gray)
    right = np.asarray(right_gray)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("handoff gray images must be same-shape 2-D arrays")
    if left.shape != np.shape(left_valid) or right.shape != np.shape(right_valid):
        raise ValueError("handoff valid masks must match their images")
    if left.shape[1] != left_transform.analysis_width or right.shape[1] != right_transform.analysis_width:
        raise ValueError("analysis image width does not match coordinate transform")
    corridor_left, corridor_right = map(float, corridor_canvas)
    if not corridor_left < corridor_right:
        raise ValueError("canvas corridor must have positive width")
    if maximum_features < 1 or not 0.0 < quality_level <= 1.0 or minimum_distance_px < 0.0:
        raise ValueError("invalid GFTT configuration")

    def gray8(value: np.ndarray) -> np.ndarray:
        if value.dtype == np.uint8:
            return np.ascontiguousarray(value)
        finite = np.nan_to_num(value.astype(np.float32), nan=0.0)
        if finite.size and float(np.max(finite)) <= 1.0:
            finite *= 255.0
        return np.clip(finite, 0, 255).astype(np.uint8)

    left8, right8 = gray8(left), gray8(right)
    mask = np.asarray(left_valid, dtype=bool).copy()
    x_canvas = np.asarray(
        [left_transform.analysis_to_canvas_x(x) for x in range(left.shape[1])]
    )
    mask &= (x_canvas[None, :] >= corridor_left) & (x_canvas[None, :] < corridor_right)
    points = cv2.goodFeaturesToTrack(
        left8, maxCorners=int(maximum_features), qualityLevel=float(quality_level),
        minDistance=float(minimum_distance_px), mask=(mask.astype(np.uint8) * 255),
    )
    if points is None:
        return ()
    forward, status_forward, _ = cv2.calcOpticalFlowPyrLK(left8, right8, points, None)
    if forward is None or status_forward is None:
        return ()
    backward, status_backward, _ = cv2.calcOpticalFlowPyrLK(right8, left8, forward, None)
    if backward is None or status_backward is None:
        return ()
    result: list[HandoffObservation] = []
    h, w = left.shape
    texture_map = cv2.cornerMinEigenVal(left8, 3)
    for source, target, returned, ok_f, ok_b in zip(
        points.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2),
        status_forward.reshape(-1), status_backward.reshape(-1),
    ):
        if not (ok_f and ok_b and np.isfinite((*source, *target, *returned)).all()):
            continue
        lx, ly = map(float, source)
        rx, ry = map(float, target)
        fb = float(np.linalg.norm(source - returned))
        vertical_error = abs(ry - ly)
        li = (int(np.rint(ly)), int(np.rint(lx)))
        ri = (int(np.rint(ry)), int(np.rint(rx)))
        if not (0 <= li[0] < h and 0 <= li[1] < w and 0 <= ri[0] < h and 0 <= ri[1] < w):
            continue
        if fb > lk_forward_backward_max_px or vertical_error > maximum_vertical_match_error_px:
            continue
        if not (bool(left_valid[li]) and bool(right_valid[ri])):
            continue
        left_canvas = left_transform.analysis_to_canvas_x(lx)
        right_canvas = right_transform.analysis_to_canvas_x(rx)
        if not (
            corridor_left <= left_canvas < corridor_right
            and corridor_left <= right_canvas < corridor_right
        ):
            continue
        left_c = (left_transform.analysis_to_calibrated_x(lx), ly)
        right_c = (right_transform.analysis_to_calibrated_x(rx), ry)
        match_id = _stable_match_id(
            calibration_id, origin_pair_lineage, left_frame_id, right_frame_id,
            left_c, right_c,
        )
        # Shi-Tomasi strength is sampled in A and used only as confidence/audit.
        texture = float(texture_map[li])
        result.append(HandoffObservation(
            match_id=match_id,
            origin_pair_lineage=origin_pair_lineage,
            left_frame_id=int(left_frame_id), right_frame_id=int(right_frame_id),
            role="unassigned", left_analysis_x=lx, left_y=ly,
            right_analysis_x=rx, right_y=ry,
            left_calibrated_x=left_c[0], right_calibrated_x=right_c[0],
            left_canvas_x=left_canvas, right_canvas_x=right_canvas,
            forward_backward_error_px=fb, vertical_error_px=vertical_error,
            texture_score=texture, observation_canvas_dx=right_canvas - left_canvas,
        ))
    return tuple(sorted(result, key=lambda item: item.match_id))


def split_solver_heldout(
    observations: Sequence[HandoffObservation], *, heldout_fraction: float = 0.20
) -> HandoffEvidenceSplit:
    """Deterministically split by stable match ID, independent of input order."""
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in (0, 1)")
    solver: list[HandoffObservation] = []
    heldout: list[HandoffObservation] = []
    threshold = int(round(heldout_fraction * 65536.0))
    seen: set[str] = set()
    for observation in sorted(observations, key=lambda item: item.match_id):
        if observation.match_id in seen:
            raise ValueError("match IDs must be unique")
        seen.add(observation.match_id)
        bucket = int(observation.match_id[:4], 16)
        if bucket < threshold:
            heldout.append(replace(observation, role="heldout"))
        else:
            solver.append(replace(observation, role="solver"))
    return HandoffEvidenceSplit(tuple(solver), tuple(heldout))


def estimate_background_canvas_dx(observations: Sequence[HandoffObservation]) -> float:
    if not observations:
        raise ValueError("background motion requires solver observations")
    if any(item.role == "heldout" for item in observations):
        raise ValueError("heldout observations cannot estimate background motion")
    values = np.asarray([item.observation_canvas_dx for item in observations], dtype=np.float64)
    median = float(np.median(values))
    absolute = np.abs(values - median)
    mad = float(np.median(absolute))
    inliers = absolute <= max(0.5, 3.0 * 1.4826 * mad)
    return float(np.median(values[inliers])) if np.any(inliers) else median


def attach_background_residual(
    observations: Sequence[HandoffObservation], background_canvas_dx: float
) -> tuple[HandoffObservation, ...]:
    if not np.isfinite(background_canvas_dx):
        raise ValueError("background canvas dx must be finite")
    return tuple(replace(
        item,
        background_canvas_dx=float(background_canvas_dx),
        relative_dx=float(item.observation_canvas_dx - background_canvas_dx),
    ) for item in observations)


def attach_aligned_depth_risk(
    observations: Sequence[HandoffObservation],
    left_depth_analysis_mm: np.ndarray,
    right_depth_analysis_mm: np.ndarray,
    *,
    absolute_consistency_gate_mm: float = 20.0,
    relative_consistency_gate: float = 0.02,
    edge_risk_weight: float = 2.0,
    inconsistent_confidence_weight: float = 0.5,
) -> tuple[HandoffObservation, ...]:
    """Attach A-domain aligned-depth risk to existing RGB observations.

    Depth is deliberately incapable of creating an observation or a hard
    interval.  A local depth edge increases the cost of crossing an already
    RGB-triggered observation.  Mutually inconsistent, otherwise non-edge
    depth lowers that observation's confidence.  Missing depth is neutral and
    therefore never turns an RGB region into an automatically safe one.

    Both arrays are aligned depth expressed in millimetres and sampled in the
    same horizontally reduced calibrated target domain (A) as the sparse LK
    coordinates.
    """
    left = np.asarray(left_depth_analysis_mm, dtype=np.float64)
    right = np.asarray(right_depth_analysis_mm, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape:
        raise ValueError("aligned-depth analysis arrays must be same-shape 2-D")
    if (
        absolute_consistency_gate_mm <= 0.0
        or relative_consistency_gate <= 0.0
        or edge_risk_weight < 1.0
        or not 0.0 < inconsistent_confidence_weight <= 1.0
    ):
        raise ValueError("invalid aligned-depth risk configuration")

    def sample(array: np.ndarray, x: float, y: float) -> tuple[float | None, bool]:
        row, column = int(np.rint(y)), int(np.rint(x))
        if not (0 <= row < array.shape[0] and 0 <= column < array.shape[1]):
            return None, False
        value = float(array[row, column])
        if not np.isfinite(value) or value <= 0.0:
            return None, False
        r0, r1 = max(0, row - 1), min(array.shape[0], row + 2)
        c0, c1 = max(0, column - 1), min(array.shape[1], column + 2)
        neighbourhood = array[r0:r1, c0:c1]
        valid = neighbourhood[np.isfinite(neighbourhood) & (neighbourhood > 0.0)]
        if valid.size < 2:
            return value, False
        gate = max(
            float(absolute_consistency_gate_mm),
            float(relative_consistency_gate) * value,
        )
        edge = float(np.max(valid) - np.min(valid)) > gate
        return value, edge

    result: list[HandoffObservation] = []
    for item in observations:
        left_depth, left_edge = sample(left, item.left_analysis_x, item.left_y)
        right_depth, right_edge = sample(right, item.right_analysis_x, item.right_y)
        weight = 1.0
        if left_edge or right_edge:
            weight = float(edge_risk_weight)
        elif left_depth is not None and right_depth is not None:
            gate = max(
                float(absolute_consistency_gate_mm),
                float(relative_consistency_gate) * max(left_depth, right_depth),
            )
            if abs(left_depth - right_depth) > gate:
                weight = float(inconsistent_confidence_weight)
        result.append(replace(
            item,
            depth_risk_weight=weight,
            left_depth_mm=left_depth,
            right_depth_mm=right_depth,
            left_depth_edge=left_edge,
            right_depth_edge=right_edge,
        ))
    return tuple(result)


def _connected_components(observations: Sequence[HandoffObservation], x_gap: float, y_gap: float,
                          residual_gap: float) -> list[list[HandoffObservation]]:
    pending = set(range(len(observations)))
    components: list[list[HandoffObservation]] = []
    while pending:
        seed = min(pending)
        pending.remove(seed)
        stack = [seed]
        indices = [seed]
        while stack:
            current = stack.pop()
            a = observations[current]
            component_residuals = [observations[index].relative_dx for index in indices]
            residual_min = min(component_residuals)
            residual_max = max(component_residuals)
            neighbours = []
            for index in pending:
                b = observations[index]
                same_sign = np.sign(a.relative_dx) == np.sign(b.relative_dx)
                proposed_residual_span = max(residual_max, b.relative_dx) - min(
                    residual_min, b.relative_dx
                )
                if (
                    same_sign
                    and abs((a.left_canvas_x + a.right_canvas_x) * 0.5
                            - (b.left_canvas_x + b.right_canvas_x) * 0.5) <= x_gap
                    and abs(a.left_y - b.left_y) <= y_gap
                    # Do not let single-link chaining gradually merge distinct
                    # motion layers.  The complete cluster, not merely each
                    # adjacent edge, must stay inside the configured residual
                    # tolerance.
                    and proposed_residual_span <= residual_gap
                ):
                    neighbours.append(index)
            for index in neighbours:
                pending.remove(index)
                stack.append(index)
                indices.append(index)
        components.append([observations[index] for index in indices])
    return components


def build_forbidden_intervals(
    observations: Sequence[HandoffObservation], *,
    minimum_match_count: int = 4,
    minimum_vertical_span_px: float = 24.0,
    minimum_interval_width_px: float = 1.5,
    minimum_relative_dx_px: float = 1.0,
    cluster_neighbor_x_px: float = 24.0,
    cluster_neighbor_y_px: float = 32.0,
    cluster_relative_dx_tolerance_px: float = 1.0,
    merge_gap_px: float = 2.0,
    guard_px: float = 3.0,
) -> tuple[ForbiddenInterval, ...]:
    """Build hard intervals only from audited multi-point solver clusters."""
    if any(item.role == "heldout" for item in observations):
        raise ValueError("heldout observations cannot create forbidden intervals")
    eligible = [item for item in observations if abs(item.relative_dx) >= minimum_relative_dx_px]
    components = _connected_components(
        eligible, cluster_neighbor_x_px, cluster_neighbor_y_px,
        cluster_relative_dx_tolerance_px,
    )
    raw: list[ForbiddenInterval] = []
    for component in components:
        span = float(np.ptp([item.left_y for item in component])) if len(component) > 1 else 0.0
        left = min(item.interval[0] for item in component) - guard_px
        right = max(item.interval[1] for item in component) + guard_px
        if (
            len(component) < minimum_match_count
            or span < minimum_vertical_span_px
            or right - left < minimum_interval_width_px
        ):
            continue
        residual = float(np.median([item.relative_dx for item in component]))
        raw.append(ForbiddenInterval(
            left_x=left, right_x=right, cluster_id=len(raw),
            match_ids=tuple(sorted(item.match_id for item in component)),
            match_count=len(component), vertical_span_px=span,
            median_relative_dx_px=residual, residual_sign=1 if residual > 0 else -1,
        ))
    raw.sort(key=lambda item: (item.left_x, item.right_x))
    merged: list[ForbiddenInterval] = []
    for item in raw:
        if (
            merged
            and item.residual_sign == merged[-1].residual_sign
            and item.left_x - merged[-1].right_x <= merge_gap_px
        ):
            prior = merged.pop()
            ids = tuple(sorted(set(prior.match_ids + item.match_ids)))
            weighted = (
                prior.median_relative_dx_px * prior.match_count
                + item.median_relative_dx_px * item.match_count
            ) / (prior.match_count + item.match_count)
            merged.append(ForbiddenInterval(
                left_x=prior.left_x, right_x=max(prior.right_x, item.right_x),
                cluster_id=prior.cluster_id, match_ids=ids, match_count=len(ids),
                vertical_span_px=max(prior.vertical_span_px, item.vertical_span_px),
                median_relative_dx_px=float(weighted),
                residual_sign=1 if weighted > 0 else -1,
            ))
        else:
            merged.append(replace(item, cluster_id=len(merged)))
    return tuple(merged)


def generate_boundary_candidates(
    *, midpoint_x: int, allowed_left_x: int, allowed_right_x: int,
    forbidden_intervals: Sequence[ForbiddenInterval],
    soft_observations: Sequence[HandoffObservation] = (),
    maximum_shift_px: int = 48,
    rgb_edge_cost: Sequence[float] | None = None,
    depth_edge_cost: Sequence[float] | None = None,
    midpoint_weight: float = 0.01,
    soft_crossing_base_cost: float = 1.0,
) -> tuple[BoundaryCandidate, ...]:
    """Generate integer straight-line states; horizontal correction is never produced."""
    if any(item.role == "heldout" for item in soft_observations):
        raise ValueError("heldout observations cannot contribute boundary cost")
    left = max(int(allowed_left_x), int(midpoint_x) - int(maximum_shift_px))
    right = min(int(allowed_right_x), int(midpoint_x) + int(maximum_shift_px))
    if left > right:
        return ()
    rgb = None if rgb_edge_cost is None else np.asarray(rgb_edge_cost, dtype=np.float64)
    depth = None if depth_edge_cost is None else np.asarray(depth_edge_cost, dtype=np.float64)
    if soft_crossing_base_cost <= 0.0 or not np.isfinite(soft_crossing_base_cost):
        raise ValueError("soft crossing base cost must be finite positive")
    result = []
    for x in range(left, right + 1):
        hard = any(interval.contains(x) for interval in forbidden_intervals)
        soft = sum(
            (soft_crossing_base_cost + abs(item.relative_dx)) * item.depth_risk_weight
            for item in soft_observations
            if item.interval[0] <= x < item.interval[1]
        )
        rgb_value = float(rgb[x - left]) if rgb is not None and x - left < rgb.size else 0.0
        depth_value = float(depth[x - left]) if depth is not None and x - left < depth.size else 0.0
        distance = abs(x - midpoint_x)
        total = soft + rgb_value + depth_value + midpoint_weight * distance
        result.append(BoundaryCandidate(
            x=x, total_cost=float(total), hard_forbidden_crossing=hard,
            soft_crossing_cost=float(soft), rgb_edge_cost=rgb_value,
            depth_edge_cost=depth_value, midpoint_distance=float(distance),
        ))
    return tuple(result)


def solve_global_straight_boundaries(
    candidate_sets: Sequence[Sequence[BoundaryCandidate]], *,
    midpoint_boundaries: Sequence[int],
    first_owner_left_x: int,
    last_owner_right_x: int,
    minimum_internal_owner_width_px: int = 9,
    observable: Sequence[bool] | None = None,
) -> StraightBoundarySolution:
    """Globally solve one monotonically increasing sequence of straight boundaries."""
    count = len(candidate_sets)
    if len(midpoint_boundaries) != count:
        raise ValueError("one midpoint is required per internal boundary")
    observed = tuple(True for _ in range(count)) if observable is None else tuple(observable)
    if len(observed) != count:
        raise ValueError("observable flags must align with boundary candidate sets")
    if minimum_internal_owner_width_px < 1 or first_owner_left_x >= last_owner_right_x:
        raise ValueError("invalid owner extent or minimum width")
    if count == 0:
        return StraightBoundarySolution((), (), True, 0.0)

    states: list[dict[int, tuple[float, int | None, BoundaryCandidate]]] = []
    for pair_index, candidates in enumerate(candidate_sets):
        # Retain hard states with a dominating penalty. This lets the global
        # topology stay feasible when two neighbouring safe ranges conflict,
        # while still selecting a hard crossing only when no all-safe monotone
        # path exists. The selected pair is then explicitly marked needs_s2.
        usable = list(candidates)
        # Unobservable pairs deliberately retain the nominal midpoint as hard owner.
        if not observed[pair_index]:
            usable = [item for item in candidates if item.x == midpoint_boundaries[pair_index]]
        if not usable:
            usable = [item for item in candidates if item.x == midpoint_boundaries[pair_index]]
        current: dict[int, tuple[float, int | None, BoundaryCandidate]] = {}
        for candidate in usable:
            if pair_index == 0:
                if candidate.x - first_owner_left_x >= minimum_internal_owner_width_px:
                    candidate_cost = candidate.total_cost + (
                        1.0e9 if candidate.hard_forbidden_crossing else 0.0
                    )
                    current[candidate.x] = (candidate_cost, None, candidate)
                continue
            best: tuple[float, int | None, BoundaryCandidate] | None = None
            for previous_x, previous in states[-1].items():
                if candidate.x - previous_x < minimum_internal_owner_width_px:
                    continue
                candidate_cost = candidate.total_cost + (
                    1.0e9 if candidate.hard_forbidden_crossing else 0.0
                )
                option = (previous[0] + candidate_cost, previous_x, candidate)
                if best is None or (option[0], candidate.x, previous_x) < (best[0], best[2].x, best[1]):
                    best = option
            if best is not None:
                current[candidate.x] = best
        states.append(current)
    feasible_end = {
        x: value for x, value in states[-1].items()
        if last_owner_right_x - x >= minimum_internal_owner_width_px
    }
    if not feasible_end:
        decisions = tuple(PairBoundaryDecision(
            pair_index=index, midpoint_x=int(midpoint), boundary_x=int(midpoint),
            classification="global_boundary_conflict", needs_s2=True,
            manual_review_required=True, protected_crossing=True,
        ) for index, midpoint in enumerate(midpoint_boundaries))
        return StraightBoundarySolution(tuple(map(int, midpoint_boundaries)), decisions, False, float("inf"))
    end_x, end_state = min(feasible_end.items(), key=lambda item: (item[1][0], item[0]))
    chosen = [0] * count
    cursor = end_x
    for index in range(count - 1, -1, -1):
        chosen[index] = cursor
        previous = states[index][cursor][1]
        if previous is not None:
            cursor = previous
    decisions = []
    for index, (boundary, midpoint, candidates) in enumerate(
        zip(chosen, midpoint_boundaries, candidate_sets)
    ):
        selected = next(item for item in candidates if item.x == boundary)
        midpoint_item = next((item for item in candidates if item.x == midpoint), None)
        if not observed[index]:
            classification = "unobservable_midpoint"
            needs_s2 = False
            manual = True
        elif selected.hard_forbidden_crossing:
            classification = "straight_seam_limited"
            needs_s2 = True
            manual = True
        elif midpoint_item is None or midpoint_item.hard_forbidden_crossing:
            classification = "straight_boundary_shifted"
            needs_s2 = False
            manual = False
        else:
            classification = "midpoint_safe"
            needs_s2 = False
            manual = False
        decisions.append(PairBoundaryDecision(
            pair_index=index, midpoint_x=int(midpoint), boundary_x=int(boundary),
            classification=classification, needs_s2=needs_s2,
            manual_review_required=manual,
            protected_crossing=selected.hard_forbidden_crossing,
        ))
    return StraightBoundarySolution(
        tuple(chosen), tuple(decisions), True, float(end_state[0])
    )


def forbidden_interval_rows(intervals: Iterable[ForbiddenInterval]) -> tuple[dict[str, object], ...]:
    """Return serialization-ready audit rows without leaking numpy scalars."""
    return tuple({
        "cluster_id": item.cluster_id, "left_x": item.left_x, "right_x": item.right_x,
        "half_open": True, "match_count": item.match_count,
        "match_ids": list(item.match_ids), "vertical_span_px": item.vertical_span_px,
        "median_relative_dx_px": item.median_relative_dx_px,
        "residual_sign": item.residual_sign,
    } for item in intervals)
