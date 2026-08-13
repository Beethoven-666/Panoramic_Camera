"""Local and sequence-relative structure audits for isolated S1.3 stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence
from typing import Mapping

import cv2
import numpy as np


METRIC_NAMES = (
    "color_jump",
    "gradient_jump",
    "double_edge",
    "motion_residual",
    "thin_structure",
    "horizontal_edge_mismatch",
)


@dataclass(frozen=True)
class SeamStructureFeatures:
    """Image features cached once when many seam paths audit one panorama."""

    lab: np.ndarray
    gray: np.ndarray
    gradient: np.ndarray
    laplacian: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.gray.shape[0]), int(self.gray.shape[1])


def _finite_summary(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size < 16 else float(np.median(finite))


def prepare_seam_structure(image: np.ndarray) -> SeamStructureFeatures | None:
    """Prepare reusable, black-safe structure features for one BGR image."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return SeamStructureFeatures(
        lab=lab,
        gray=gray,
        gradient=cv2.magnitude(gx, gy),
        laplacian=np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)),
        gradient_x=gx,
        gradient_y=gy,
    )


def _horizontal_edge_mismatch(left: np.ndarray, right: np.ndarray) -> float:
    """Measure row-step disagreement only where either side has a real horizontal edge."""

    left_f = np.asarray(left, dtype=np.float64)
    right_f = np.asarray(right, dtype=np.float64)
    strength = np.maximum(np.abs(left_f), np.abs(right_f))
    finite = np.isfinite(strength) & np.isfinite(left_f) & np.isfinite(right_f)
    positive = strength[finite & (strength >= 8.0)]
    if positive.size < 16:
        return 0.0
    threshold = max(8.0, float(np.median(positive)))
    support = finite & (strength >= threshold)
    mismatch = np.abs(left_f[support] - right_f[support])
    return float(np.median(mismatch)) if mismatch.size >= 16 else 0.0


def seam_structure_metrics(
    image: np.ndarray | SeamStructureFeatures,
    seam_x_by_row: np.ndarray,
    *,
    radius_px: int = 4,
) -> dict[str, object]:
    """Measure local hard-owner structure without treating black RGB as invalid."""

    features = image if isinstance(image, SeamStructureFeatures) else prepare_seam_structure(image)
    if features is None:
        return {"evaluable": False, "reason": "invalid_image", **{name: None for name in METRIC_NAMES}}
    height, width = features.shape
    seam = np.asarray(seam_x_by_row, dtype=np.int32)
    if seam.shape != (height,) or width < 5:
        return {"evaluable": False, "reason": "invalid_seam", **{name: None for name in METRIC_NAMES}}
    rows = np.arange(height)
    usable = (seam >= 2) & (seam <= width - 3)
    if int(usable.sum()) < 16:
        return {"evaluable": False, "reason": "insufficient_corridor_rows", **{name: None for name in METRIC_NAMES}}
    rows, seam = rows[usable], seam[usable]
    left = features.lab[rows, seam - 1]
    right = features.lab[rows, seam]
    color = np.linalg.norm(left - right, axis=1)
    grad_jump = np.abs(
        features.gradient[rows, seam - 1] - features.gradient[rows, seam]
    )
    double = np.maximum(
        features.laplacian[rows, seam - 1], features.laplacian[rows, seam]
    )
    motion = np.abs(features.gray[rows, seam - 2] - features.gray[rows, seam + 1])
    thin = np.maximum(
        np.abs(features.gradient_x[rows, seam - 1]),
        np.abs(features.gradient_x[rows, seam]),
    )
    horizontal = _horizontal_edge_mismatch(
        features.gradient_y[rows, seam - 1], features.gradient_y[rows, seam]
    )
    values = {
        "color_jump": _finite_summary(color),
        "gradient_jump": _finite_summary(grad_jump),
        "double_edge": _finite_summary(double),
        "motion_residual": _finite_summary(motion),
        "thin_structure": _finite_summary(thin),
        "horizontal_edge_mismatch": horizontal,
    }
    if any(value is None for value in values.values()):
        return {"evaluable": False, "reason": "nonfinite_or_sparse_metrics", **values}
    score = float(
        0.25 * values["color_jump"]
        + 0.15 * values["gradient_jump"]
        + 0.15 * values["double_edge"]
        + 0.10 * values["motion_residual"]
        + 0.10 * values["thin_structure"]
        + 0.25 * values["horizontal_edge_mismatch"]
    )
    return {"evaluable": True, "reason": None, **values, "score": score, "radius_px": int(radius_px)}


def structurally_non_degrading(
    before: Mapping[str, object],
    after: Mapping[str, object],
    *,
    relative_tolerance: float = 0.02,
) -> tuple[bool, str | None]:
    """Require aggregate and every protected component to remain locally non-worse."""

    if before.get("evaluable") is not True or after.get("evaluable") is not True:
        return False, "insufficient_relative_structure_evidence"
    for name in (*METRIC_NAMES, "score"):
        left, right = before.get(name), after.get(name)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            return False, f"unevaluable_{name}"
        allowance = max(0.25, abs(float(left)) * float(relative_tolerance))
        if float(right) > float(left) + allowance:
            return False, f"{name}_degraded"
    return True, None


def mean_seam_structure_metrics(
    metrics: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Average already comparable seam audits without hiding an unevaluable path."""

    rows = tuple(metrics)
    if not rows or any(row.get("evaluable") is not True for row in rows):
        return {
            "evaluable": False,
            "reason": "unevaluable_symmetric_seam_path",
            **{name: None for name in METRIC_NAMES},
            "score": None,
        }
    values: dict[str, float] = {}
    for name in (*METRIC_NAMES, "score"):
        samples = [row.get(name) for row in rows]
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in samples):
            return {
                "evaluable": False,
                "reason": f"nonfinite_symmetric_{name}",
                **{metric_name: None for metric_name in METRIC_NAMES},
                "score": None,
            }
        values[name] = float(np.mean(np.asarray(samples, dtype=np.float64)))
    return {
        "evaluable": True,
        "reason": None,
        **values,
        "radius_px": max(int(row.get("radius_px", 0)) for row in rows),
        "path_count": len(rows),
    }


def symmetric_seam_structure_metrics(
    before: np.ndarray | SeamStructureFeatures,
    after: np.ndarray | SeamStructureFeatures,
    base_seam_x_by_row: np.ndarray,
    candidate_seam_x_by_row: np.ndarray,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate both images on both seam paths so moving a seam cannot game selection."""

    paths = (base_seam_x_by_row, candidate_seam_x_by_row)
    before_metrics = mean_seam_structure_metrics(
        tuple(seam_structure_metrics(before, path) for path in paths)
    )
    after_metrics = mean_seam_structure_metrics(
        tuple(seam_structure_metrics(after, path) for path in paths)
    )
    before_metrics["comparison_policy"] = "symmetric_base_and_candidate_paths"
    after_metrics["comparison_policy"] = "symmetric_base_and_candidate_paths"
    return before_metrics, after_metrics


def _horizontal_profile(
    features: SeamStructureFeatures,
    seam: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(features.shape[0], dtype=np.int32)[:, None]
    columns = seam[:, None] + offsets[None, :]
    in_bounds = (columns >= 0) & (columns < features.shape[1])
    clipped = np.clip(columns, 0, features.shape[1] - 1)
    gx = features.gradient_x[rows, clipped]
    gy = features.gradient_y[rows, clipped]
    horizontal = in_bounds & (np.abs(gy) >= np.abs(gx)) & (np.abs(gy) >= 8.0)
    count = horizontal.sum(axis=1)
    profile = np.divide(
        np.where(horizontal, gy, 0.0).sum(axis=1),
        np.maximum(count, 1),
        dtype=np.float64,
    )
    profile = cv2.GaussianBlur(profile.astype(np.float32)[:, None], (1, 5), 0).ravel()
    return profile.astype(np.float64), count > 0


def _profile_correlation(
    left: np.ndarray,
    right: np.ndarray,
    left_support: np.ndarray,
    right_support: np.ndarray,
    lag: int,
) -> tuple[float | None, int]:
    if lag < 0:
        left_values, right_values = left[:lag], right[-lag:]
        support = left_support[:lag] & right_support[-lag:]
    elif lag > 0:
        left_values, right_values = left[lag:], right[:-lag]
        support = left_support[lag:] & right_support[:-lag]
    else:
        left_values, right_values = left, right
        support = left_support & right_support
    count = int(support.sum())
    minimum_support = max(4, int(math.ceil(len(left) * 0.01)))
    if count < minimum_support:
        return None, count
    a = left_values[support] - float(np.mean(left_values[support]))
    b = right_values[support] - float(np.mean(right_values[support]))
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-6:
        return None, count
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0)), count


def long_horizontal_structure_metrics(
    image: np.ndarray | SeamStructureFeatures,
    seam_x_by_row: np.ndarray,
    *,
    maximum_lag_px: int = 8,
) -> dict[str, object]:
    """Audit long horizontal-edge continuity outside the seam cost's near pixels."""

    features = image if isinstance(image, SeamStructureFeatures) else prepare_seam_structure(image)
    if features is None:
        return {"observed": False, "reason": "invalid_image", "line_score": None}
    seam = np.asarray(seam_x_by_row, dtype=np.int32)
    if seam.shape != (features.shape[0],):
        return {"observed": False, "reason": "invalid_seam", "line_score": None}
    left, left_support = _horizontal_profile(
        features, seam, np.arange(-16, -8, dtype=np.int32)
    )
    right, right_support = _horizontal_profile(
        features, seam, np.arange(9, 17, dtype=np.int32)
    )
    correlations: dict[int, tuple[float, int]] = {}
    for lag in range(-int(maximum_lag_px), int(maximum_lag_px) + 1):
        correlation, support = _profile_correlation(
            left, right, left_support, right_support, lag
        )
        if correlation is not None:
            correlations[lag] = (correlation, support)
    if not correlations:
        return {
            "observed": False,
            "reason": "insufficient_horizontal_edge_support",
            "line_score": None,
            "supported_lag_count": 0,
        }
    best_lag, (best_correlation, best_support) = max(
        correlations.items(), key=lambda item: (item[1][0], -abs(item[0]))
    )
    zero_correlation = correlations.get(0, (-1.0, 0))[0]
    observed = bool(best_correlation >= 0.50)
    line_score = float(
        abs(best_lag) + 4.0 * max(0.0, best_correlation - zero_correlation)
    )
    return {
        "observed": observed,
        "reason": None if observed else "weak_horizontal_edge_correlation",
        "best_vertical_lag_px": int(best_lag),
        "absolute_best_vertical_lag_px": abs(int(best_lag)),
        "best_correlation": float(best_correlation),
        "zero_lag_correlation": float(zero_correlation),
        "best_minus_zero_correlation": float(best_correlation - zero_correlation),
        "line_score": line_score,
        "support_row_count": int(best_support),
        "supported_lag_count": len(correlations),
        "maximum_lag_px": int(maximum_lag_px),
        "held_out_from_near_seam_cost": True,
    }


def long_horizontal_structure_nondegrading(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> tuple[bool, str | None]:
    """Reject both lost support and newly introduced, visibly displaced structure."""

    if before.get("observed") is not True:
        if after.get("observed") is True:
            after_score = after.get("line_score")
            if not isinstance(after_score, (int, float)):
                return False, "horizontal_structure_score_unevaluable"
            if float(after_score) > 0.25:
                return False, "horizontal_structure_new_misalignment"
        return True, None
    if after.get("observed") is not True:
        return False, "horizontal_structure_support_lost"
    before_score, after_score = before.get("line_score"), after.get("line_score")
    if not isinstance(before_score, (int, float)) or not isinstance(after_score, (int, float)):
        return False, "horizontal_structure_score_unevaluable"
    allowance = max(0.25, abs(float(before_score)) * 0.10)
    if float(after_score) > float(before_score) + allowance:
        return False, "horizontal_structure_continuity_degraded"
    return True, None


def sequence_structure_decision(
    before: Sequence[Mapping[str, object]],
    after: Sequence[Mapping[str, object]],
    *,
    maximum_component_failure_fraction: float = 0.05,
    maximum_pair_score_failure_fraction: float = 0.0,
    minimum_mean_improvement_fraction: float = 0.005,
) -> tuple[bool, dict[str, object]]:
    """Make a robust sequence decision while bounding every pair's total score.

    A small number of protected-component outliers may not veto a clearly
    better long sequence, but no pair may materially regress in total score.
    Short sequences remain strict because their rounded-down failure budget is
    zero.
    """

    left = tuple(before)
    right = tuple(after)
    if len(left) != len(right) or not left:
        return False, {
            "eligible": False,
            "reason": "mismatched_or_empty_sequence_metrics",
            "pair_count": len(left),
        }
    if not 0.0 <= maximum_component_failure_fraction < 1.0:
        raise ValueError("maximum_component_failure_fraction must be in [0, 1)")
    if not 0.0 <= maximum_pair_score_failure_fraction < 1.0:
        raise ValueError("maximum_pair_score_failure_fraction must be in [0, 1)")
    if not 0.0 <= minimum_mean_improvement_fraction < 1.0:
        raise ValueError("minimum_mean_improvement_fraction must be in [0, 1)")

    decisions = [structurally_non_degrading(a, b) for a, b in zip(left, right, strict=True)]
    unevaluable = [
        index for index, (a, b) in enumerate(zip(left, right, strict=True))
        if a.get("evaluable") is not True or b.get("evaluable") is not True
    ]
    failure_indices = [index for index, (passed, _reason) in enumerate(decisions) if not passed]
    score_regressions: list[int] = []
    catastrophic_score_regressions: list[dict[str, object]] = []
    catastrophic_component_regressions: list[dict[str, object]] = []
    before_scores: list[float] = []
    after_scores: list[float] = []
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        before_score, after_score = a.get("score"), b.get("score")
        if not isinstance(before_score, (int, float)) or not isinstance(after_score, (int, float)):
            score_regressions.append(index)
            continue
        before_value, after_value = float(before_score), float(after_score)
        if not math.isfinite(before_value) or not math.isfinite(after_value):
            score_regressions.append(index)
            continue
        before_scores.append(before_value)
        after_scores.append(after_value)
        allowance = max(0.25, abs(before_value) * 0.02)
        if after_value > before_value + allowance:
            score_regressions.append(index)
        catastrophic_score_allowance = max(2.0, abs(before_value) * 0.25)
        if after_value > before_value + catastrophic_score_allowance:
            catastrophic_score_regressions.append(
                {
                    "pair_index": index,
                    "before": before_value,
                    "after": after_value,
                    "allowance": catastrophic_score_allowance,
                }
            )
        for name in METRIC_NAMES:
            before_component, after_component = a.get(name), b.get(name)
            if not isinstance(before_component, (int, float)) or not isinstance(
                after_component, (int, float)
            ):
                continue
            component_allowance = max(2.0, abs(float(before_component)) * 0.25)
            if float(after_component) > float(before_component) + component_allowance:
                catastrophic_component_regressions.append(
                    {
                        "pair_index": index,
                        "metric": name,
                        "before": float(before_component),
                        "after": float(after_component),
                        "allowance": component_allowance,
                    }
                )

    pair_count = len(left)
    failure_budget = int(math.floor(pair_count * maximum_component_failure_fraction))
    score_failure_budget = int(math.floor(pair_count * maximum_pair_score_failure_fraction))
    before_mean = float(np.mean(before_scores)) if len(before_scores) == pair_count else None
    after_mean = float(np.mean(after_scores)) if len(after_scores) == pair_count else None
    aggregate_improvement = (
        None
        if before_mean is None or after_mean is None or before_mean <= 0.0
        else float((before_mean - after_mean) / before_mean)
    )
    eligible = bool(
        not unevaluable
        and len(failure_indices) <= failure_budget
        and len(score_regressions) <= score_failure_budget
        and not catastrophic_score_regressions
        and not catastrophic_component_regressions
        and aggregate_improvement is not None
        and aggregate_improvement >= minimum_mean_improvement_fraction
    )
    reason = None
    if unevaluable:
        reason = "unevaluable_sequence_pairs"
    elif len(failure_indices) > failure_budget:
        reason = "component_failure_budget_exceeded"
    elif len(score_regressions) > score_failure_budget:
        reason = "pair_total_score_failure_budget_exceeded"
    elif catastrophic_score_regressions:
        reason = "catastrophic_pair_total_score_regression"
    elif catastrophic_component_regressions:
        reason = "catastrophic_component_regression"
    elif aggregate_improvement is None or aggregate_improvement < minimum_mean_improvement_fraction:
        reason = "aggregate_improvement_not_proven"
    return eligible, {
        "eligible": eligible,
        "reason": reason,
        "pair_count": pair_count,
        "component_pass_count": pair_count - len(failure_indices),
        "component_failure_count": len(failure_indices),
        "component_failure_indices": failure_indices,
        "component_failure_reasons": [decisions[index][1] for index in failure_indices],
        "component_failure_budget": failure_budget,
        "maximum_component_failure_fraction": float(maximum_component_failure_fraction),
        "pair_total_score_regression_indices": score_regressions,
        "pair_total_score_failure_budget": score_failure_budget,
        "maximum_pair_score_failure_fraction": float(maximum_pair_score_failure_fraction),
        "catastrophic_pair_total_score_regressions": catastrophic_score_regressions,
        "catastrophic_component_regressions": catastrophic_component_regressions,
        "before_mean_score": before_mean,
        "after_mean_score": after_mean,
        "aggregate_improvement_fraction": aggregate_improvement,
        "minimum_mean_improvement_fraction": float(minimum_mean_improvement_fraction),
    }


__all__ = [
    "METRIC_NAMES",
    "SeamStructureFeatures",
    "mean_seam_structure_metrics",
    "long_horizontal_structure_metrics",
    "long_horizontal_structure_nondegrading",
    "prepare_seam_structure",
    "seam_structure_metrics",
    "sequence_structure_decision",
    "structurally_non_degrading",
    "symmetric_seam_structure_metrics",
]
