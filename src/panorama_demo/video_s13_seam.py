"""Straight and monotone-DP seam candidates for the isolated S1.3 M5 stage.

The module only selects hard-owner boundaries.  It deliberately contains no
GraphCut, photometric correction, blending, depth, or production publication.
All costs are finite so that risky structure is discouraged without creating
an unbounded forbidden mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class S13SeamResult:
    """A selected per-row seam and the audit trail for its fallbacks."""

    seam_x_by_row: np.ndarray
    model: str
    total_cost: float
    cost_components: Mapping[str, float]
    audit: Mapping[str, object]
    fallback_chain: tuple[str, ...]
    candidate_seams: Mapping[str, np.ndarray]
    candidate_total_costs: Mapping[str, float]
    candidate_cost_components: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class S13SeamCandidate:
    """One independently generated seam candidate for a single pair."""

    candidate_id: str
    model_code: str
    model_name: str
    seam_x_by_row: np.ndarray
    local_objective: float
    complexity_rank: int
    generation_audit: Mapping[str, object]


_MODEL_CODES = {
    "midpoint_straight": ("S0", 0),
    "shifted_straight": ("S1", 1),
    "monotone_dp": ("S2", 2),
}


def build_s13_seam_candidates(
    pair_index: int,
    result: S13SeamResult,
) -> tuple[tuple[S13SeamCandidate, ...], tuple[Mapping[str, object], ...]]:
    """Expose S0/S1/S2 as peers, with an explicit status for every model."""

    candidates: list[S13SeamCandidate] = []
    statuses: list[Mapping[str, object]] = []
    failures = tuple(str(value) for value in result.audit.get("fallback_reasons", ()))
    for model_name, (model_code, complexity_rank) in _MODEL_CODES.items():
        path = result.candidate_seams.get(model_name)
        if path is None:
            prefix = f"{model_name}:"
            reason = next((value[len(prefix):] for value in failures if value.startswith(prefix)), None)
            statuses.append({
                "candidate_id": f"pair-{pair_index:04d}-{model_code}",
                "model_code": model_code,
                "model_name": model_name,
                "generation_status": "generation_failed",
                "reason": reason or "candidate_unavailable",
            })
            continue
        objective = float(result.candidate_total_costs[model_name])
        component_costs = dict(result.candidate_cost_components[model_name])
        candidate = S13SeamCandidate(
            candidate_id=f"pair-{pair_index:04d}-{model_code}",
            model_code=model_code,
            model_name=model_name,
            seam_x_by_row=np.asarray(path, dtype=np.int32).copy(),
            local_objective=objective,
            complexity_rank=complexity_rank,
            generation_audit={
                "generation_status": "generated",
                "seam_total_cost": objective,
                "cost_components": component_costs,
                "search_left_x": result.audit.get("search_left_x"),
                "search_right_x": result.audit.get("search_right_x"),
                "maximum_shift_px": result.audit.get("maximum_shift_px"),
                "allowed_row_steps": (-1, 0, 1),
                "seam_sha256": _sha_path(path),
            },
        )
        candidates.append(candidate)
        statuses.append({
            "candidate_id": candidate.candidate_id,
            "model_code": model_code,
            "model_name": model_name,
            "generation_status": "generated",
            "reason": None,
        })
    if not any(candidate.model_code == "S0" for candidate in candidates):
        raise RuntimeError("midpoint seam candidate was not generated")
    return tuple(candidates), tuple(statuses)


def _sha_path(path: np.ndarray) -> str:
    import hashlib

    value = np.ascontiguousarray(path)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def rank_s13_seam_candidates(
    candidates: Sequence[S13SeamCandidate],
    *,
    minimum_complex_improvement_fraction: float = 0.02,
) -> tuple[S13SeamCandidate, ...]:
    """Rank one pair locally; this ordering never authorizes a stage seal."""

    if not 0.0 <= minimum_complex_improvement_fraction < 1.0:
        raise ValueError("minimum_complex_improvement_fraction must be in [0, 1)")
    straight_costs = [
        candidate.local_objective for candidate in candidates
        if candidate.model_name in {"midpoint_straight", "shifted_straight"}
    ]
    best_straight = min(straight_costs, default=math.inf)

    def key(candidate: S13SeamCandidate) -> tuple[float, int, str]:
        objective = float(candidate.local_objective)
        if candidate.model_name == "monotone_dp" and math.isfinite(best_straight):
            improvement = (best_straight - objective) / max(abs(best_straight), 1e-6)
            if improvement < minimum_complex_improvement_fraction:
                # Keep the objective as an offline fact, but put a near-tied
                # complex path behind the simpler safe candidates.
                objective = best_straight
        return objective, candidate.complexity_rank, candidate.candidate_id

    return tuple(sorted(candidates, key=key))


@dataclass(frozen=True)
class _CostVolume:
    total: np.ndarray
    components: Mapping[str, np.ndarray]
    common_valid: np.ndarray
    search_left_x: int
    search_right_x: int


_COMPONENT_WEIGHTS: Mapping[str, float] = {
    "color": 0.24,
    "gradient_magnitude": 0.13,
    "gradient_direction": 0.10,
    "double_edge": 0.15,
    "motion_residual": 0.13,
    "thin_strong_structure": 0.15,
    "boundary_offset": 0.10,
}


def seam_search_bounds(
    width: int,
    base_boundary_x: int,
    *,
    maximum_shift_px: int = 8,
    previous_boundary_x: int | None = None,
    next_boundary_x: int | None = None,
) -> tuple[int, int]:
    """Return inclusive, ordered search bounds for one seam.

    Midpoints to adjacent *base* boundaries divide neighbouring search domains.
    Equal limits are allowed because a duplicate source may legitimately have a
    zero-width owner, while crossing is never allowed.
    """

    width = int(width)
    boundary = int(base_boundary_x)
    shift = int(maximum_shift_px)
    if width < 3:
        raise ValueError("seam images must be at least three pixels wide")
    if shift < 4 or shift > 8:
        raise ValueError("maximum_shift_px must be in the inclusive range [4, 8]")
    if boundary < 1 or boundary > width - 2:
        raise ValueError("base_boundary_x must leave pixels on both sides")
    if previous_boundary_x is not None and int(previous_boundary_x) >= boundary:
        raise ValueError("previous_boundary_x must be less than base_boundary_x")
    if next_boundary_x is not None and int(next_boundary_x) <= boundary:
        raise ValueError("next_boundary_x must be greater than base_boundary_x")

    ordered_left = 1
    ordered_right = width - 2
    if previous_boundary_x is not None:
        ordered_left = int(math.ceil(0.5 * (int(previous_boundary_x) + boundary)))
    if next_boundary_x is not None:
        ordered_right = int(math.floor(0.5 * (boundary + int(next_boundary_x))))
    search_left = max(1, boundary - shift, ordered_left)
    search_right = min(width - 2, boundary + shift, ordered_right)
    if search_left > search_right:
        raise ValueError("adjacent seam ordering leaves an empty search band")
    return search_left, search_right


def _validate_inputs(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    motion_residual: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    left = np.asarray(left_bgr)
    right = np.asarray(right_bgr)
    if left.ndim != 3 or left.shape[2] != 3 or left.dtype != np.uint8:
        raise ValueError("left_bgr must be an HxWx3 uint8 image")
    if right.shape != left.shape or right.dtype != np.uint8:
        raise ValueError("right_bgr must match left_bgr")
    lv = np.asarray(left_valid, dtype=bool)
    rv = np.asarray(right_valid, dtype=bool)
    if lv.shape != left.shape[:2] or rv.shape != left.shape[:2]:
        raise ValueError("valid masks must match the image height and width")
    residual = None if motion_residual is None else np.asarray(motion_residual, dtype=np.float32)
    if residual is not None and residual.shape != left.shape[:2]:
        raise ValueError("motion_residual must match the image height and width")
    return left, right, lv, rv, residual


def _robust_unit(values: np.ndarray, support: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite_support = np.asarray(support, dtype=bool) & np.isfinite(array)
    if not np.any(finite_support):
        return np.zeros_like(array, dtype=np.float32)
    sample = array[finite_support]
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    scale = max(1e-3, 1.4826 * mad)
    normalized = np.abs(array - median) / scale
    normalized = np.nan_to_num(normalized, nan=8.0, posinf=8.0, neginf=8.0)
    return np.clip(normalized, 0.0, 8.0).astype(np.float32)


def _features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    direction = cv2.phase(gx, gy, angleInDegrees=False)
    return lab, gray, magnitude, direction


def _angular_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.abs(left - right)
    return np.minimum(delta, (2.0 * np.pi) - delta).astype(np.float32)


def _build_cost_volume(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    motion_residual: np.ndarray | None,
    base_boundary_x: int,
    search_left_x: int,
    search_right_x: int,
) -> _CostVolume:
    left_lab, left_gray, left_mag, left_dir = _features(left)
    right_lab, right_gray, right_mag, right_dir = _features(right)
    common = left_valid & right_valid
    columns = np.arange(search_left_x, search_right_x + 1, dtype=np.int32)
    band = np.s_[:, search_left_x:search_right_x + 1]
    support = common[band]

    raw_color = np.linalg.norm(left_lab[band] - right_lab[band], axis=2)
    raw_mag = np.abs(left_mag[band] - right_mag[band])
    raw_direction = _angular_difference(left_dir[band], right_dir[band])
    # A seam is risky where both sources carry a strong, displaced edge.
    left_lap = np.abs(cv2.Laplacian(left_gray, cv2.CV_32F, ksize=3))[band]
    right_lap = np.abs(cv2.Laplacian(right_gray, cv2.CV_32F, ksize=3))[band]
    raw_double = np.minimum(left_lap, right_lap) + 0.5 * np.abs(left_lap - right_lap)
    if motion_residual is None:
        raw_motion = np.abs(left_gray[band] - right_gray[band])
    else:
        raw_motion = np.abs(motion_residual[band])
    left_canny = cv2.Canny(left, 80, 160).astype(np.float32)[band] / 255.0
    right_canny = cv2.Canny(right, 80, 160).astype(np.float32)[band] / 255.0
    raw_structure = (
        np.abs(left_canny - right_canny)
        + np.minimum(left_mag[band], right_mag[band]) / 255.0
    )
    boundary = np.broadcast_to(
        np.abs(columns.astype(np.float32) - float(base_boundary_x))[None, :]
        / max(1.0, float(max(base_boundary_x - search_left_x, search_right_x - base_boundary_x))),
        support.shape,
    )

    components = {
        "color": _robust_unit(raw_color, support),
        "gradient_magnitude": _robust_unit(raw_mag, support),
        "gradient_direction": _robust_unit(raw_direction, support),
        "double_edge": _robust_unit(raw_double, support),
        "motion_residual": _robust_unit(raw_motion, support),
        "thin_strong_structure": _robust_unit(raw_structure, support),
        "boundary_offset": boundary.astype(np.float32),
    }
    total = np.zeros(support.shape, dtype=np.float32)
    for name, weight in _COMPONENT_WEIGHTS.items():
        total += float(weight) * components[name]
    # Invalid overlap remains traversable at a high finite cost.  A separate
    # support audit decides whether an optimized candidate is admissible.
    total += np.where(support, 0.0, 12.0).astype(np.float32)
    total = np.nan_to_num(total, nan=32.0, posinf=32.0, neginf=32.0)
    return _CostVolume(
        total=total,
        components=components,
        common_valid=support,
        search_left_x=search_left_x,
        search_right_x=search_right_x,
    )


def _minimum_supported_rows(volume: _CostVolume) -> int:
    return max(8, int(math.ceil(volume.total.shape[0] * 0.20)))


def _straight_candidate(volume: _CostVolume) -> tuple[np.ndarray, float]:
    supported_per_column = np.sum(volume.common_valid, axis=0)
    eligible = supported_per_column >= _minimum_supported_rows(volume)
    if not np.any(eligible):
        raise ValueError("insufficient_common_valid_rows_for_shifted_straight")
    column_cost = np.mean(volume.total, axis=0)
    column_cost = np.where(eligible, column_cost, np.inf)
    relative_x = int(np.argmin(column_cost))
    if not np.isfinite(column_cost[relative_x]):
        raise ValueError("nonfinite_shifted_straight_cost")
    seam_x = volume.search_left_x + relative_x
    seam = np.full(volume.total.shape[0], seam_x, dtype=np.int32)
    return seam, float(column_cost[relative_x] * volume.total.shape[0])


def _dp_candidate(
    volume: _CostVolume,
    *,
    slope_weight: float,
    curvature_weight: float,
) -> tuple[np.ndarray, float]:
    height, band_width = volume.total.shape
    if height < 2 or band_width < 1:
        raise ValueError("empty_monotone_dp_domain")
    if int(np.sum(np.any(volume.common_valid, axis=1))) < _minimum_supported_rows(volume):
        raise ValueError("insufficient_common_valid_rows_for_monotone_dp")
    steps = np.asarray((-1, 0, 1), dtype=np.int32)
    inf = np.float64(np.inf)
    state = np.full((height, band_width, 3), inf, dtype=np.float64)
    back_x = np.full((height, band_width, 3), -1, dtype=np.int32)
    back_step = np.full((height, band_width, 3), -1, dtype=np.int8)
    state[0, :, 1] = volume.total[0]
    # These are invariant for every row; keeping them out of the hot loop is
    # especially material for the many narrow, 480-row video corridors.
    transitions = []
    for step in steps:
        x = np.arange(band_width, dtype=np.int32)
        previous_x = x - int(step)
        in_bounds = (previous_x >= 0) & (previous_x < band_width)
        valid_x = x[in_bounds]
        transitions.append((
            valid_x,
            previous_x[in_bounds],
            float(slope_weight) * abs(int(step))
            + float(curvature_weight) * np.abs(steps - int(step)),
        ))
    for row in range(1, height):
        # Keep the exact float64 recurrence and NumPy's first-index tie break,
        # but evaluate every x position for one step together.  The previous
        # scalar loop created millions of tiny ``argmin`` arrays over a real
        # video session; this is the same three-state DP, not a new seam rule.
        for step_index, (valid_x, previous, penalties) in enumerate(transitions):
            transition = state[row - 1, previous] + penalties[None, :]
            previous_step = np.argmin(transition, axis=1)
            best = transition[np.arange(len(valid_x)), previous_step]
            finite = np.isfinite(best)
            if not np.any(finite):
                continue
            target_x = valid_x[finite]
            state[row, target_x, step_index] = (
                best[finite] + volume.total[row, target_x]
            )
            back_x[row, target_x, step_index] = previous[finite]
            back_step[row, target_x, step_index] = previous_step[finite]
    flat_index = int(np.argmin(state[-1]))
    x, step_index = np.unravel_index(flat_index, state[-1].shape)
    total_cost = float(state[-1, x, step_index])
    if not math.isfinite(total_cost):
        raise ValueError("monotone_dp_has_no_finite_path")
    relative = np.empty(height, dtype=np.int32)
    relative[-1] = int(x)
    for row in range(height - 1, 0, -1):
        previous_x = int(back_x[row, x, step_index])
        previous_step = int(back_step[row, x, step_index])
        if previous_x < 0 or previous_step < 0:
            raise ValueError("monotone_dp_backtrack_failed")
        x, step_index = previous_x, previous_step
        relative[row - 1] = int(x)
    if np.any(np.abs(np.diff(relative)) > 1):
        raise ValueError("monotone_dp_step_audit_failed")
    return relative + volume.search_left_x, total_cost


def _path_component_costs(volume: _CostVolume, seam: np.ndarray) -> dict[str, float]:
    rows = np.arange(volume.total.shape[0], dtype=np.int32)
    relative = seam.astype(np.int32) - int(volume.search_left_x)
    return {
        name: float(np.mean(values[rows, relative]))
        for name, values in volume.components.items()
    }


def _complete_path_component_costs(
    volume: _CostVolume,
    seam: np.ndarray,
    *,
    slope_weight: float,
    curvature_weight: float,
) -> dict[str, float]:
    components = _path_component_costs(volume, seam)
    step = np.diff(np.asarray(seam, dtype=np.float64))
    components["slope"] = float(slope_weight) * (
        float(np.mean(np.abs(step))) if step.size else 0.0
    )
    curvature = np.diff(step)
    components["curvature"] = float(curvature_weight) * (
        float(np.mean(np.abs(curvature))) if curvature.size else 0.0
    )
    return components


def select_s13_seam(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    base_boundary_x: int,
    *,
    maximum_shift_px: int = 8,
    previous_boundary_x: int | None = None,
    next_boundary_x: int | None = None,
    motion_residual: np.ndarray | None = None,
    slope_weight: float = 0.20,
    curvature_weight: float = 0.35,
    minimum_dp_relative_improvement: float = 0.02,
) -> S13SeamResult:
    """Select the simplest audited seam within a bounded relative cost margin."""

    left, right, lv, rv, residual = _validate_inputs(
        left_bgr, right_bgr, left_valid, right_valid, motion_residual
    )
    if not math.isfinite(float(slope_weight)) or float(slope_weight) < 0.0:
        raise ValueError("slope_weight must be finite and non-negative")
    if not math.isfinite(float(curvature_weight)) or float(curvature_weight) < 0.0:
        raise ValueError("curvature_weight must be finite and non-negative")
    if not 0.0 <= float(minimum_dp_relative_improvement) < 1.0:
        raise ValueError("minimum_dp_relative_improvement must be in [0, 1)")
    search_left, search_right = seam_search_bounds(
        left.shape[1],
        base_boundary_x,
        maximum_shift_px=maximum_shift_px,
        previous_boundary_x=previous_boundary_x,
        next_boundary_x=next_boundary_x,
    )
    volume = _build_cost_volume(
        left, right, lv, rv, residual, int(base_boundary_x), search_left, search_right
    )
    failures: list[str] = []
    fallback_chain: list[str] = ["monotone_dp", "shifted_straight"]
    dp_result: tuple[np.ndarray, float] | None = None
    straight_result: tuple[np.ndarray, float] | None = None
    try:
        dp_result = _dp_candidate(
            volume, slope_weight=float(slope_weight), curvature_weight=float(curvature_weight)
        )
    except (ArithmeticError, ValueError) as exc:
        failures.append(f"monotone_dp:{exc}")
    try:
        straight_result = _straight_candidate(volume)
    except (ArithmeticError, ValueError) as exc:
        failures.append(f"shifted_straight:{exc}")

    midpoint_seam = np.full(left.shape[0], int(base_boundary_x), dtype=np.int32)
    midpoint_relative = int(base_boundary_x) - search_left
    midpoint_total_cost = float(np.sum(volume.total[:, midpoint_relative]))
    dp_relative_improvement: float | None = None
    if dp_result is not None and straight_result is not None:
        dp_relative_improvement = float(
            (straight_result[1] - dp_result[1]) / max(abs(straight_result[1]), 1e-6)
        )
        if dp_relative_improvement >= float(minimum_dp_relative_improvement):
            model, (seam, total_cost) = "monotone_dp", dp_result
            selection_reason = "dp_proved_relative_cost_improvement"
        else:
            model, (seam, total_cost) = "shifted_straight", straight_result
            selection_reason = "straight_preferred_without_dp_margin"
            failures.append("monotone_dp:insufficient_relative_cost_improvement")
    elif dp_result is not None:
        # Generation and local ordering are independent of the straight
        # candidate.  The caller still applies the DP candidate's own geometry
        # and hard safety audit before it can own pixels.
        model, (seam, total_cost) = "monotone_dp", dp_result
        selection_reason = "dp_ranked_by_independent_local_objective"
    elif straight_result is not None:
        model, (seam, total_cost) = "shifted_straight", straight_result
        selection_reason = "monotone_dp_unavailable"
    else:
        fallback_chain.append("midpoint_straight")
        model = "midpoint_straight"
        selection_reason = "optimized_seams_unavailable"
        seam = midpoint_seam
        total_cost = midpoint_total_cost

    relative = seam - search_left
    common_on_path = volume.common_valid[np.arange(left.shape[0]), relative]
    step_ok = bool(np.all(np.isin(np.diff(seam), (-1, 0, 1))))
    in_bounds = bool(np.all((seam >= search_left) & (seam <= search_right)))
    finite_cost = bool(math.isfinite(total_cost) and np.isfinite(volume.total).all())
    audit: dict[str, object] = {
        "base_boundary_x": int(base_boundary_x),
        "search_left_x": int(search_left),
        "search_right_x": int(search_right),
        "maximum_shift_px": int(maximum_shift_px),
        "previous_boundary_x": None if previous_boundary_x is None else int(previous_boundary_x),
        "next_boundary_x": None if next_boundary_x is None else int(next_boundary_x),
        "ordered_non_crossing_search_bounds": True,
        "finite_cost": finite_cost,
        "in_search_bounds": in_bounds,
        "allowed_row_steps_only": step_ok,
        "common_valid_path_rows": int(np.sum(common_on_path)),
        "total_rows": int(left.shape[0]),
        "slope_weight": float(slope_weight),
        "curvature_weight": float(curvature_weight),
        "minimum_dp_relative_improvement": float(minimum_dp_relative_improvement),
        "dp_total_cost": None if dp_result is None else float(dp_result[1]),
        "shifted_straight_total_cost": (
            None if straight_result is None else float(straight_result[1])
        ),
        "dp_relative_improvement": dp_relative_improvement,
        "selection_reason": selection_reason,
        "fallback_reasons": tuple(failures),
    }
    if not (finite_cost and in_bounds and step_ok):
        raise RuntimeError("selected seam failed its hard safety audit")
    candidate_seams: dict[str, np.ndarray] = {
        "midpoint_straight": midpoint_seam.copy(),
    }
    candidate_total_costs: dict[str, float] = {
        "midpoint_straight": midpoint_total_cost,
    }
    if straight_result is not None:
        candidate_seams["shifted_straight"] = straight_result[0].copy()
        candidate_total_costs["shifted_straight"] = float(straight_result[1])
    if dp_result is not None:
        candidate_seams["monotone_dp"] = dp_result[0].copy()
        candidate_total_costs["monotone_dp"] = float(dp_result[1])
    candidate_cost_components = {
        name: _complete_path_component_costs(
            volume,
            path,
            slope_weight=float(slope_weight),
            curvature_weight=float(curvature_weight),
        )
        for name, path in candidate_seams.items()
    }
    component_costs = candidate_cost_components[model]
    return S13SeamResult(
        seam_x_by_row=seam,
        model=model,
        total_cost=total_cost,
        cost_components=component_costs,
        audit=audit,
        fallback_chain=tuple(fallback_chain),
        candidate_seams=candidate_seams,
        candidate_total_costs=candidate_total_costs,
        candidate_cost_components=candidate_cost_components,
    )


__all__ = [
    "S13SeamCandidate", "S13SeamResult", "build_s13_seam_candidates",
    "rank_s13_seam_candidates", "seam_search_bounds", "select_s13_seam",
]
