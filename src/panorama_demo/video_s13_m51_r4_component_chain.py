"""Pure component-chain C2E primitives for the isolated S1.3 M5.1-r4 candidate.

The module deliberately has no stage-writing or renderer authority.  It turns
exact, pair-local edge evidence into deterministic chains, application
segments, source-local correction fields, a conflict-resolved patch registry,
and canonical source-map oracle arrays.  The M5 orchestrator is responsible for
supplying immutable P0-derived maps and for applying only audited patch sets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import cv2
import numpy as np


EvidenceState = Literal["actionable", "safe_anchor", "ambiguous", "unevaluable"]
CandidateDecision = Literal[
    "resolved", "improved_unresolved", "rejected", "budget_deferred"
]


def _readonly(array: np.ndarray, dtype: np.dtype | str | None = None) -> np.ndarray:
    value = np.array(array, dtype=dtype, copy=True, order="C")
    value.setflags(write=False)
    return value


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


def _canonical_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("C2E canonical payload contains a nonfinite float")
        rounded = round(number, 6)
        return 0.0 if rounded == 0.0 else rounded
    return value


def _json_sha(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        _canonical_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def canonical_s13_support_sha256(support_xy: np.ndarray) -> str:
    """Hash exact xy support with the same canonical array contract as assets."""

    return _array_sha(np.ascontiguousarray(support_xy, dtype="<i4"))


def canonical_s13_normal_search_lags(
    minimum_px: float = -3.0,
    maximum_px: float = 3.0,
    step_px: float = 0.5,
) -> np.ndarray:
    """Return the frozen inclusive lag grid, rejecting inexact endpoints."""

    values = (float(minimum_px), float(maximum_px), float(step_px))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("C2E normal-search values must be finite")
    if step_px <= 0.0 or maximum_px < minimum_px:
        raise ValueError("C2E normal-search interval is invalid")
    intervals = (maximum_px - minimum_px) / step_px
    rounded = int(round(intervals))
    if not math.isclose(intervals, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("C2E normal-search endpoints are not divisible by the step")
    lags = minimum_px + np.arange(rounded + 1, dtype=np.float64) * step_px
    lags[-1] = maximum_px
    return _readonly(lags, np.float64)


@dataclass(frozen=True)
class S13EdgeComponentObservation:
    pair_index: int
    component_id: int
    source_indices: tuple[int, int]
    global_bbox_xyxy: tuple[int, int, int, int]
    block_indices: tuple[int, ...]
    normal_x: float
    normal_y: float
    fitted_line_offset: float
    forward_best_lag_px: float
    reverse_best_lag_px: float
    correlation: float
    uniqueness_fraction: float
    orientation_difference_degrees: float
    mask_sha256: str
    evidence_state: EvidenceState
    exclusion_reasons: tuple[str, ...]
    signed_gradient_polarity: float = 1.0
    forward_valid_samples: int = 0
    reverse_valid_samples: int = 0
    reference_support_samples: int = 0
    search_boundary_hit: bool = False

    def __post_init__(self) -> None:
        if self.pair_index < 0 or self.component_id < 0:
            raise ValueError("C2E observation indices must be nonnegative")
        if self.source_indices != (self.pair_index, self.pair_index + 1):
            raise ValueError("C2E observation sources must match its adjacent pair")
        x0, y0, x1, y1 = self.global_bbox_xyxy
        if x0 >= x1 or y0 >= y1:
            raise ValueError("C2E observation bbox is empty")
        normal_norm = math.hypot(float(self.normal_x), float(self.normal_y))
        finite = (
            self.normal_x, self.normal_y, self.fitted_line_offset,
            self.forward_best_lag_px, self.reverse_best_lag_px, self.correlation,
            self.uniqueness_fraction, self.orientation_difference_degrees,
            self.signed_gradient_polarity,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("C2E observation contains nonfinite metrics")
        if not math.isclose(normal_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("C2E observation normal must be unit length")
        if len(self.mask_sha256) != 64:
            raise ValueError("C2E observation mask SHA is invalid")
        if self.evidence_state not in {"actionable", "safe_anchor", "ambiguous", "unevaluable"}:
            raise ValueError("C2E observation evidence state is invalid")
        if min(
            self.forward_valid_samples,
            self.reverse_valid_samples,
            self.reference_support_samples,
        ) < 0:
            raise ValueError("C2E observation sample counts must be nonnegative")
        object.__setattr__(self, "block_indices", tuple(sorted(set(self.block_indices))))
        object.__setattr__(self, "exclusion_reasons", tuple(self.exclusion_reasons))


@dataclass(frozen=True)
class S13ExactEdgeComponentEvidence:
    support_xy: np.ndarray
    forward_scores: np.ndarray
    reverse_scores: np.ndarray
    forward_correlations: np.ndarray
    reverse_correlations: np.ndarray
    forward_support_counts: np.ndarray
    reverse_support_counts: np.ndarray

    def __post_init__(self) -> None:
        support = np.asarray(self.support_xy)
        if support.ndim != 2 or support.shape[1] != 2:
            raise ValueError("C2E support must be Nx2 xy coordinates")
        length = np.asarray(self.forward_scores).size
        if length < 1 or any(
            np.asarray(value).shape != (length,)
            for value in (
                self.reverse_scores,
                self.forward_correlations,
                self.reverse_correlations,
                self.forward_support_counts,
                self.reverse_support_counts,
            )
        ):
            raise ValueError("C2E hypothesis evidence shapes disagree")
        object.__setattr__(self, "support_xy", _readonly(support, np.int32))
        for name in (
            "forward_scores", "reverse_scores", "forward_correlations", "reverse_correlations"
        ):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            values = np.where(np.isfinite(values), np.round(values, 6), values)
            object.__setattr__(self, name, _readonly(values, np.float64))
        for name in ("forward_support_counts", "reverse_support_counts"):
            object.__setattr__(self, name, _readonly(getattr(self, name), np.int32))


def _bilinear(array: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Keep the cached Sobel/magnitude dtype.  Casting the whole ROI to float64
    # here would copy it once per lag and per physical component; arithmetic
    # with the float64 interpolation weights already promotes sampled values.
    image = np.asarray(array)
    if image.ndim != 2:
        raise ValueError("C2E feature arrays must be two-dimensional")
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    valid = (x0 >= 0) & (y0 >= 0) & (x1 < image.shape[1]) & (y1 < image.shape[0])
    sx0, sx1 = np.clip(x0, 0, image.shape[1] - 1), np.clip(x1, 0, image.shape[1] - 1)
    sy0, sy1 = np.clip(y0, 0, image.shape[0] - 1), np.clip(y1, 0, image.shape[0] - 1)
    wx, wy = x - x0, y - y0
    top = image[sy0, sx0] * (1.0 - wx) + image[sy0, sx1] * wx
    bottom = image[sy1, sx0] * (1.0 - wx) + image[sy1, sx1] * wx
    sampled = top * (1.0 - wy) + bottom * wy
    valid &= np.isfinite(sampled)
    return sampled, valid


def _fit_weighted_line(
    support_xy: np.ndarray, weights: np.ndarray, gradient_x: np.ndarray, gradient_y: np.ndarray
) -> tuple[float, float, float, float]:
    points = np.asarray(support_xy, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if points.shape[0] < 2 or weight.shape != (points.shape[0],) or weight.sum() <= 0.0:
        raise ValueError("C2E line fit has insufficient weighted support")
    center = np.average(points, axis=0, weights=weight)
    centered = points - center
    covariance = (centered * weight[:, None]).T @ centered / weight.sum()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    # A thick antialiased edge contains two Sobel flanks whose raw signed
    # gradients nearly cancel.  Using that cancellation to orient the normal
    # makes repeated fits flip sign under harmless floating-point noise.  Use
    # a coordinate-only canonical hemisphere, then measure contrast polarity
    # from gradient sign relative to the fitted centerline; both flanks agree
    # under this product and inverted contrast remains distinguishable.
    if normal[0] < 0.0 or (normal[0] == 0.0 and normal[1] < 0.0):
        normal = -normal
    normal /= np.linalg.norm(normal)
    offset = float(np.dot(normal, center))
    projected_gradient = gradient_x * normal[0] + gradient_y * normal[1]
    signed_distance = points @ normal - offset
    polarity_terms = projected_gradient * signed_distance
    denominator = float(np.sum(weight * np.abs(polarity_terms)))
    polarity = (
        float(np.sum(weight * polarity_terms)) / denominator
        if denominator > 1e-12 else 0.0
    )
    return float(normal[0]), float(normal[1]), offset, polarity


def _hypothesis_scores(
    *,
    reference_magnitude: np.ndarray,
    moving_magnitude: np.ndarray,
    reference_gx: np.ndarray,
    reference_gy: np.ndarray,
    moving_gx: np.ndarray,
    moving_gy: np.ndarray,
    anchor_xy: np.ndarray,
    normal_xy: tuple[float, float],
    lags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchors = np.asarray(anchor_xy, dtype=np.float64)
    rx, rvalid = _bilinear(reference_magnitude, anchors[:, 0], anchors[:, 1])
    rgx, rgxvalid = _bilinear(reference_gx, anchors[:, 0], anchors[:, 1])
    rgy, rgyvalid = _bilinear(reference_gy, anchors[:, 0], anchors[:, 1])
    base_valid = rvalid & rgxvalid & rgyvalid & (rx > 0.0)
    scores = np.full(lags.shape, -math.inf, dtype=np.float64)
    correlations = np.zeros(lags.shape, dtype=np.float64)
    agreements = np.zeros(lags.shape, dtype=np.float64)
    counts = np.zeros(lags.shape, dtype=np.int32)
    nx, ny = normal_xy
    for index, lag in enumerate(lags):
        x = anchors[:, 0] + float(lag) * nx
        y = anchors[:, 1] + float(lag) * ny
        mm, mvalid = _bilinear(moving_magnitude, x, y)
        mgx, mgxvalid = _bilinear(moving_gx, x, y)
        mgy, mgyvalid = _bilinear(moving_gy, x, y)
        keep = base_valid & mvalid & mgxvalid & mgyvalid & (mm > 0.0)
        counts[index] = int(keep.sum())
        if counts[index] < 2:
            continue
        left, right = rx[keep], mm[keep]
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1e-12:
            continue
        correlation = float(np.clip(np.dot(left, right) / denominator, 0.0, 1.0))
        orientation_denominator = np.hypot(rgx[keep], rgy[keep]) * np.hypot(
            mgx[keep], mgy[keep]
        )
        usable = orientation_denominator > 1e-12
        if not np.any(usable):
            agreement = 0.0
        else:
            dot = rgx[keep][usable] * mgx[keep][usable] + rgy[keep][usable] * mgy[keep][usable]
            agreement = float(np.average(dot / orientation_denominator[usable], weights=left[usable]))
        correlations[index], agreements[index] = correlation, agreement
        # Cosine correlation alone cannot distinguish an exact response from a
        # uniformly attenuated half-pixel interpolation.  Keep correlation as
        # its own reported gate, and include a bounded magnitude-consistency
        # term only in the hypothesis ranking score.
        magnitude_scale = max(float(np.mean(np.maximum(left, right))), 1e-12)
        magnitude_consistency = float(np.clip(
            1.0 - np.mean(np.abs(left - right)) / magnitude_scale, 0.0, 1.0
        ))
        scores[index] = correlation * max(0.0, agreement) * magnitude_consistency
    return scores, correlations, agreements, counts


def _best_hypothesis(scores: np.ndarray, lags: np.ndarray) -> tuple[int, float]:
    finite = np.flatnonzero(np.isfinite(scores))
    if finite.size == 0:
        raise ValueError("C2E observation has no finite lag hypothesis")
    index = min(
        (int(item) for item in finite),
        key=lambda item: (-float(scores[item]), abs(float(lags[item])), float(lags[item])),
    )
    separated = [
        int(item) for item in finite
        if abs(float(lags[item]) - float(lags[index])) >= 1.5
    ]
    second = max((float(scores[item]) for item in separated), default=0.0)
    uniqueness = (float(scores[index]) - second) / max(abs(float(scores[index])), 1e-12)
    return index, float(uniqueness)


def make_s13_edge_component_observation(
    *,
    pair_index: int,
    component_id: int,
    source_indices: tuple[int, int],
    support_xy: np.ndarray,
    left_magnitude: np.ndarray,
    right_magnitude: np.ndarray,
    left_gradient_x: np.ndarray | None = None,
    left_gradient_y: np.ndarray | None = None,
    right_gradient_x: np.ndarray | None = None,
    right_gradient_y: np.ndarray | None = None,
    lags: np.ndarray | None = None,
    block_indices: tuple[int, ...] = (),
    minimum_correlation: float = 0.75,
    minimum_uniqueness_fraction: float = 0.10,
    maximum_orientation_difference_degrees: float = 10.0,
    maximum_forward_reverse_discrepancy_px: float = 0.5,
    minimum_signed_gradient_agreement: float = 0.10,
    precomputed_forward_scores: np.ndarray | None = None,
    precomputed_forward_correlations: np.ndarray | None = None,
    precomputed_forward_agreements: np.ndarray | None = None,
    precomputed_forward_support_counts: np.ndarray | None = None,
) -> tuple[S13EdgeComponentObservation, S13ExactEdgeComponentEvidence]:
    """Fit one exact component and evaluate symmetric normal-lag hypotheses."""

    support = np.asarray(support_xy, dtype=np.int32)
    if support.ndim != 2 or support.shape[1] != 2 or support.shape[0] < 2:
        raise ValueError("C2E exact component support must contain at least two xy points")
    order = np.lexsort((support[:, 0], support[:, 1]))
    support = support[order]
    if np.any(support[:, 0] < 0) or np.any(support[:, 1] < 0):
        raise ValueError("C2E support coordinates must be nonnegative")
    left_mag = np.asarray(left_magnitude)
    right_mag = np.asarray(right_magnitude)
    if left_mag.shape != right_mag.shape or left_mag.ndim != 2:
        raise ValueError("C2E left/right feature shapes disagree")
    if np.any(support[:, 0] >= left_mag.shape[1]) or np.any(support[:, 1] >= left_mag.shape[0]):
        raise ValueError("C2E support lies outside the feature image")
    lgx = np.zeros_like(left_mag) if left_gradient_x is None else np.asarray(left_gradient_x)
    lgy = np.zeros_like(left_mag) if left_gradient_y is None else np.asarray(left_gradient_y)
    rgx = np.zeros_like(right_mag) if right_gradient_x is None else np.asarray(right_gradient_x)
    rgy = np.zeros_like(right_mag) if right_gradient_y is None else np.asarray(right_gradient_y)
    if any(value.shape != left_mag.shape for value in (lgx, lgy, rgx, rgy)):
        raise ValueError("C2E gradient feature shapes disagree")
    x, y = support[:, 0], support[:, 1]
    weights = np.maximum(left_mag[y, x], 0.0)
    nx, ny, line_offset, polarity = _fit_weighted_line(
        support, weights, lgx[y, x], lgy[y, x]
    )
    lag_values = canonical_s13_normal_search_lags() if lags is None else _readonly(lags, np.float64)
    precomputed = (
        precomputed_forward_scores,
        precomputed_forward_correlations,
        precomputed_forward_agreements,
        precomputed_forward_support_counts,
    )
    if any(value is not None for value in precomputed):
        if not all(value is not None for value in precomputed):
            raise ValueError("C2E precomputed forward hypothesis arrays are incomplete")
        forward = (
            np.asarray(precomputed_forward_scores, dtype=np.float64),
            np.asarray(precomputed_forward_correlations, dtype=np.float64),
            np.asarray(precomputed_forward_agreements, dtype=np.float64),
            np.asarray(precomputed_forward_support_counts, dtype=np.int32),
        )
        if any(value.shape != lag_values.shape for value in forward):
            raise ValueError("C2E precomputed forward hypothesis shapes disagree")
    else:
        forward = _hypothesis_scores(
            reference_magnitude=left_mag, moving_magnitude=right_mag,
            reference_gx=lgx, reference_gy=lgy, moving_gx=rgx, moving_gy=rgy,
            anchor_xy=support, normal_xy=(nx, ny), lags=lag_values,
        )
    forward_index, forward_uniqueness = _best_hypothesis(forward[0], lag_values)
    # Hypothesis coordinates describe where the moving edge was sampled.
    # The public lag is the inverse-map correction convention used by the
    # solver: a moving edge sampled at +d requires a reported lag of -d.
    forward_sampling_lag = float(lag_values[forward_index])
    forward_lag = -forward_sampling_lag
    moving_support = (
        support.astype(np.float64)
        + forward_sampling_lag * np.asarray((nx, ny))[None, :]
    )
    reverse = _hypothesis_scores(
        reference_magnitude=right_mag, moving_magnitude=left_mag,
        reference_gx=rgx, reference_gy=rgy, moving_gx=lgx, moving_gy=lgy,
        anchor_xy=moving_support, normal_xy=(nx, ny), lags=lag_values,
    )
    reverse_index, reverse_uniqueness = _best_hypothesis(reverse[0], lag_values)
    reverse_lag = -float(lag_values[reverse_index])
    correlation = min(float(forward[1][forward_index]), float(reverse[1][reverse_index]))
    signed_agreement = min(float(forward[2][forward_index]), float(reverse[2][reverse_index]))
    orientation_difference = math.degrees(math.acos(float(np.clip(signed_agreement, -1.0, 1.0))))
    uniqueness = min(forward_uniqueness, reverse_uniqueness)
    reasons: list[str] = []
    if correlation < minimum_correlation:
        reasons.append("correlation_below_minimum")
    if uniqueness < minimum_uniqueness_fraction:
        reasons.append("normal_search_not_unique")
    if orientation_difference > maximum_orientation_difference_degrees:
        reasons.append("orientation_difference_exceeded")
    if signed_agreement < minimum_signed_gradient_agreement:
        reasons.append("signed_gradient_disagrees")
    if abs(forward_lag + reverse_lag) > maximum_forward_reverse_discrepancy_px:
        reasons.append("forward_reverse_discrepancy_exceeded")
    if min(int(forward[3][forward_index]), int(reverse[3][reverse_index])) < 2:
        reasons.append("insufficient_finite_support")
    state: EvidenceState
    if reasons:
        state = "ambiguous"
    else:
        state = "actionable" if abs(forward_lag) > 1.0 else "safe_anchor"
    bbox = (
        int(support[:, 0].min()), int(support[:, 1].min()),
        int(support[:, 0].max()) + 1, int(support[:, 1].max()) + 1,
    )
    support_hash = canonical_s13_support_sha256(support)
    observation = S13EdgeComponentObservation(
        pair_index=pair_index,
        component_id=component_id,
        source_indices=source_indices,
        global_bbox_xyxy=bbox,
        block_indices=block_indices,
        normal_x=nx,
        normal_y=ny,
        fitted_line_offset=line_offset,
        forward_best_lag_px=forward_lag,
        reverse_best_lag_px=reverse_lag,
        correlation=correlation,
        uniqueness_fraction=uniqueness,
        orientation_difference_degrees=orientation_difference,
        mask_sha256=support_hash,
        evidence_state=state,
        exclusion_reasons=tuple(reasons),
        signed_gradient_polarity=polarity,
        forward_valid_samples=int(forward[3][forward_index]),
        reverse_valid_samples=int(reverse[3][reverse_index]),
        reference_support_samples=int(support.shape[0]),
        search_boundary_hit=bool(
            forward_index in {0, len(lag_values) - 1} or reverse_index in {0, len(lag_values) - 1}
        ),
    )
    evidence = S13ExactEdgeComponentEvidence(
        support_xy=support,
        forward_scores=forward[0],
        reverse_scores=reverse[0],
        forward_correlations=forward[1],
        reverse_correlations=reverse[1],
        forward_support_counts=forward[3],
        reverse_support_counts=reverse[3],
    )
    return observation, evidence


def make_s13_unevaluable_edge_component_observation(
    *,
    pair_index: int,
    component_id: int,
    source_indices: tuple[int, int],
    support_xy: np.ndarray,
    left_magnitude: np.ndarray,
    left_gradient_x: np.ndarray,
    left_gradient_y: np.ndarray,
    lags: np.ndarray,
    block_indices: tuple[int, ...] = (),
    reason: str = "reverse_hypothesis_unevaluable",
) -> tuple[S13EdgeComponentObservation, S13ExactEdgeComponentEvidence]:
    """Freeze a detected physical component when symmetric sampling cannot run."""

    support = np.asarray(support_xy, dtype=np.int32)
    if support.ndim != 2 or support.shape[1] != 2 or support.shape[0] < 2:
        raise ValueError("C2E unevaluable support must contain at least two xy points")
    support = support[np.lexsort((support[:, 0], support[:, 1]))]
    magnitude = np.asarray(left_magnitude, dtype=np.float64)
    gx = np.asarray(left_gradient_x, dtype=np.float64)
    gy = np.asarray(left_gradient_y, dtype=np.float64)
    if magnitude.ndim != 2 or gx.shape != magnitude.shape or gy.shape != magnitude.shape:
        raise ValueError("C2E unevaluable feature shapes disagree")
    x, y = support[:, 0], support[:, 1]
    if (
        np.any(x < 0) or np.any(y < 0)
        or np.any(x >= magnitude.shape[1]) or np.any(y >= magnitude.shape[0])
    ):
        raise ValueError("C2E unevaluable support lies outside its feature image")
    weights = np.maximum(magnitude[y, x], 0.0)
    nx, ny, offset, polarity = _fit_weighted_line(
        support, weights, gx[y, x], gy[y, x]
    )
    lag_values = np.asarray(lags, dtype=np.float64)
    if lag_values.ndim != 1 or lag_values.size < 1 or not np.isfinite(lag_values).all():
        raise ValueError("C2E unevaluable lag grid is invalid")
    bbox = (
        int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1,
    )
    observation = S13EdgeComponentObservation(
        pair_index=pair_index,
        component_id=component_id,
        source_indices=source_indices,
        global_bbox_xyxy=bbox,
        block_indices=block_indices,
        normal_x=nx,
        normal_y=ny,
        fitted_line_offset=offset,
        forward_best_lag_px=0.0,
        reverse_best_lag_px=0.0,
        correlation=0.0,
        uniqueness_fraction=0.0,
        orientation_difference_degrees=0.0,
        mask_sha256=canonical_s13_support_sha256(support),
        evidence_state="unevaluable",
        exclusion_reasons=(str(reason),),
        signed_gradient_polarity=polarity,
        reference_support_samples=int(support.shape[0]),
    )
    evidence = S13ExactEdgeComponentEvidence(
        support_xy=support,
        forward_scores=np.full(lag_values.shape, -math.inf, np.float64),
        reverse_scores=np.full(lag_values.shape, -math.inf, np.float64),
        forward_correlations=np.zeros(lag_values.shape, np.float64),
        reverse_correlations=np.zeros(lag_values.shape, np.float64),
        forward_support_counts=np.zeros(lag_values.shape, np.int32),
        reverse_support_counts=np.zeros(lag_values.shape, np.int32),
    )
    return observation, evidence


@dataclass(frozen=True)
class S13BaselineC2EObligation:
    obligation_id: str
    pair_index: int
    component_id: int
    global_bbox_xyxy: tuple[int, int, int, int]
    support_sha256: str
    baseline_metrics: Mapping[str, object]
    severe: bool
    evaluable: bool
    scope: Literal["runtime_detected"] = "runtime_detected"

    def __post_init__(self) -> None:
        object.__setattr__(self, "baseline_metrics", _frozen_mapping(self.baseline_metrics))


def freeze_s13_baseline_c2e_obligations(
    observations: Sequence[S13EdgeComponentObservation],
) -> tuple[S13BaselineC2EObligation, ...]:
    obligations: list[S13BaselineC2EObligation] = []
    for observation in sorted(observations, key=lambda row: (row.pair_index, row.component_id)):
        metrics = {
            "forward_best_lag_px": observation.forward_best_lag_px,
            "reverse_best_lag_px": observation.reverse_best_lag_px,
            "correlation": observation.correlation,
            "uniqueness_fraction": observation.uniqueness_fraction,
            "evidence_state": observation.evidence_state,
        }
        payload = {
            "pair_index": observation.pair_index,
            "component_id": observation.component_id,
            "bbox": observation.global_bbox_xyxy,
            "support_sha256": observation.mask_sha256,
        }
        obligations.append(S13BaselineC2EObligation(
            obligation_id=f"c2e-obligation-{_json_sha(payload)[:20]}",
            pair_index=observation.pair_index,
            component_id=observation.component_id,
            global_bbox_xyxy=observation.global_bbox_xyxy,
            support_sha256=observation.mask_sha256,
            baseline_metrics=metrics,
            severe=abs(observation.forward_best_lag_px) > 1.0,
            evaluable=observation.evidence_state != "unevaluable",
        ))
    return tuple(obligations)


@dataclass(frozen=True)
class S13EdgeComponentChain:
    chain_id: str
    pair_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    observations: tuple[S13EdgeComponentObservation, ...]
    canonical_normal_xy: tuple[float, float]
    global_bbox_xyxy: tuple[int, int, int, int]
    chain_sha256: str


def _angle_difference(left: tuple[float, float], right: tuple[float, float]) -> float:
    dot = abs(float(np.dot(np.asarray(left), np.asarray(right))))
    return math.degrees(math.acos(float(np.clip(dot, -1.0, 1.0))))


def _line_y(observation: S13EdgeComponentObservation, x: float) -> float:
    if abs(observation.normal_y) <= 1e-9:
        return 0.5 * (observation.global_bbox_xyxy[1] + observation.global_bbox_xyxy[3] - 1)
    return (observation.fitted_line_offset - observation.normal_x * x) / observation.normal_y


def _compatibility_score(
    left: S13EdgeComponentObservation,
    right: S13EdgeComponentObservation,
    config: object,
) -> float | None:
    if right.pair_index - left.pair_index != 1:
        return None
    if left.evidence_state in {"ambiguous", "unevaluable"} or right.evidence_state in {
        "ambiguous", "unevaluable"
    }:
        return None
    ly0, ly1 = left.global_bbox_xyxy[1], left.global_bbox_xyxy[3]
    ry0, ry1 = right.global_bbox_xyxy[1], right.global_bbox_xyxy[3]
    overlap = max(0, min(ly1, ry1) - max(ly0, ry0)) / max(1, min(ly1 - ly0, ry1 - ry0))
    if overlap < float(config.minimum_component_y_overlap_fraction):
        return None
    angle = _angle_difference((left.normal_x, left.normal_y), (right.normal_x, right.normal_y))
    if angle > float(config.maximum_component_normal_difference_degrees):
        return None
    left_endpoints = np.asarray([
        (left.global_bbox_xyxy[0], _line_y(left, left.global_bbox_xyxy[0])),
        (left.global_bbox_xyxy[2] - 1, _line_y(left, left.global_bbox_xyxy[2] - 1)),
    ])
    right_endpoints = np.asarray([
        (right.global_bbox_xyxy[0], _line_y(right, right.global_bbox_xyxy[0])),
        (right.global_bbox_xyxy[2] - 1, _line_y(right, right.global_bbox_xyxy[2] - 1)),
    ])
    endpoint_distance = float(np.min(np.linalg.norm(
        left_endpoints[:, None, :] - right_endpoints[None, :, :], axis=2
    )))
    if endpoint_distance > float(config.maximum_component_endpoint_distance_px):
        return None
    seam_midpoint = 0.25 * (
        left.global_bbox_xyxy[0] + left.global_bbox_xyxy[2]
        + right.global_bbox_xyxy[0] + right.global_bbox_xyxy[2]
    )
    predicted_y = abs(_line_y(left, seam_midpoint) - _line_y(right, seam_midpoint))
    if predicted_y > float(config.maximum_predicted_y_disagreement_px):
        return None
    polarity_product = left.signed_gradient_polarity * right.signed_gradient_polarity
    if polarity_product <= 0.0:
        return None
    signed_agreement = min(abs(left.signed_gradient_polarity), abs(right.signed_gradient_polarity))
    if signed_agreement < float(config.minimum_signed_gradient_agreement):
        return None
    weights = config.component_match_weights
    correlation = min(left.correlation, right.correlation)
    uniqueness = min(left.uniqueness_fraction, right.uniqueness_fraction)
    parts = {
        "y_overlap": np.clip(overlap, 0.0, 1.0),
        "predicted_y": np.clip(1.0 - predicted_y / config.maximum_predicted_y_disagreement_px, 0.0, 1.0),
        "endpoint_distance": np.clip(1.0 - endpoint_distance / config.maximum_component_endpoint_distance_px, 0.0, 1.0),
        "correlation": np.clip(correlation, 0.0, 1.0),
        "uniqueness": np.clip(uniqueness, 0.0, 1.0),
    }
    return float(sum(float(weights[name]) * float(parts[name]) for name in parts))


def _maximum_weight_matching(
    left: Sequence[S13EdgeComponentObservation],
    right: Sequence[S13EdgeComponentObservation],
    scores: Mapping[tuple[int, int], float],
) -> tuple[tuple[int, int], ...]:
    if len(right) <= 18:
        states: dict[int, tuple[float, tuple[tuple[int, int], ...]]] = {0: (0.0, ())}
        for left_index in range(len(left)):
            updated = dict(states)
            for mask, (total, edges) in states.items():
                for right_index in range(len(right)):
                    if mask & (1 << right_index) or (left_index, right_index) not in scores:
                        continue
                    candidate = (total + scores[left_index, right_index], edges + ((left_index, right_index),))
                    key = mask | (1 << right_index)
                    current = updated.get(key)
                    if current is None or candidate[0] > current[0] + 1e-12 or (
                        math.isclose(candidate[0], current[0], abs_tol=1e-12) and candidate[1] < current[1]
                    ):
                        updated[key] = candidate
            states = updated
        return max(states.values(), key=lambda item: (item[0], -len(item[1]), tuple(
            (-a, -b) for a, b in item[1]
        )))[1]
    ordered = sorted(scores, key=lambda edge: (-scores[edge], edge))
    used_left: set[int] = set()
    used_right: set[int] = set()
    chosen = []
    for edge in ordered:
        if edge[0] not in used_left and edge[1] not in used_right:
            chosen.append(edge)
            used_left.add(edge[0])
            used_right.add(edge[1])
    return tuple(chosen)


def build_s13_edge_component_chains(
    pair_observations: Sequence[S13EdgeComponentObservation] | Mapping[
        int, Sequence[S13EdgeComponentObservation]
    ],
    *,
    config: object,
) -> tuple[S13EdgeComponentChain, ...]:
    """Build deterministic physical chains with one-to-one adjacent matching."""

    if isinstance(pair_observations, Mapping):
        observations = tuple(row for rows in pair_observations.values() for row in rows)
    else:
        observations = tuple(pair_observations)
    observations = tuple(sorted(observations, key=lambda row: (row.pair_index, row.component_id)))
    by_pair: dict[int, list[int]] = {}
    for index, observation in enumerate(observations):
        by_pair.setdefault(observation.pair_index, []).append(index)
    links: set[tuple[int, int]] = set()
    margin = float(config.minimum_component_match_margin_fraction)
    for pair_index in sorted(by_pair):
        if pair_index + 1 not in by_pair:
            continue
        left_indices, right_indices = by_pair[pair_index], by_pair[pair_index + 1]
        scores: dict[tuple[int, int], float] = {}
        for li, global_left in enumerate(left_indices):
            for ri, global_right in enumerate(right_indices):
                score = _compatibility_score(observations[global_left], observations[global_right], config)
                if score is not None:
                    scores[li, ri] = score
        ambiguous_left: set[int] = set()
        ambiguous_right: set[int] = set()
        for li in range(len(left_indices)):
            values = sorted((value for (row, _), value in scores.items() if row == li), reverse=True)
            if len(values) > 1 and (values[0] - values[1]) / max(abs(values[0]), 1e-12) < margin:
                ambiguous_left.add(li)
        for ri in range(len(right_indices)):
            values = sorted((value for (_, column), value in scores.items() if column == ri), reverse=True)
            if len(values) > 1 and (values[0] - values[1]) / max(abs(values[0]), 1e-12) < margin:
                ambiguous_right.add(ri)
        scores = {
            edge: value for edge, value in scores.items()
            if edge[0] not in ambiguous_left and edge[1] not in ambiguous_right
        }
        for li, ri in _maximum_weight_matching(
            [observations[index] for index in left_indices],
            [observations[index] for index in right_indices],
            scores,
        ):
            links.add((left_indices[li], right_indices[ri]))
    adjacency: dict[int, set[int]] = {index: set() for index in range(len(observations))}
    for left, right in links:
        adjacency[left].add(right)
        adjacency[right].add(left)
    chains: list[S13EdgeComponentChain] = []
    unseen = set(range(len(observations)))
    while unseen:
        start = min(unseen)
        stack, component = [start], []
        while stack:
            index = stack.pop()
            if index not in unseen:
                continue
            unseen.remove(index)
            component.append(index)
            stack.extend(sorted(adjacency[index], reverse=True))
        rows = tuple(sorted((observations[index] for index in component), key=lambda row: (
            row.pair_index, row.component_id
        )))
        weights = np.asarray([max(row.correlation, 1e-9) for row in rows])
        normals = np.asarray([(row.normal_x, row.normal_y) for row in rows], np.float64)
        reference = normals[0]
        normals[np.sum(normals * reference[None, :], axis=1) < 0.0] *= -1.0
        normal = np.average(normals, axis=0, weights=weights)
        normal /= np.linalg.norm(normal)
        bbox = (
            min(row.global_bbox_xyxy[0] for row in rows),
            min(row.global_bbox_xyxy[1] for row in rows),
            max(row.global_bbox_xyxy[2] for row in rows),
            max(row.global_bbox_xyxy[3] for row in rows),
        )
        payload = {
            "observations": [(row.pair_index, row.component_id, row.mask_sha256) for row in rows],
            "normal": (float(normal[0]), float(normal[1])), "bbox": bbox,
        }
        sha = _json_sha(payload)
        chains.append(S13EdgeComponentChain(
            chain_id=f"c2e-chain-{sha[:20]}",
            pair_indices=tuple(row.pair_index for row in rows),
            source_indices=tuple(sorted({source for row in rows for source in row.source_indices})),
            observations=rows,
            canonical_normal_xy=(float(normal[0]), float(normal[1])),
            global_bbox_xyxy=bbox,
            chain_sha256=sha,
        ))
    return tuple(sorted(chains, key=lambda chain: chain.chain_id))


@dataclass(frozen=True)
class S13ComponentApplicationSegment:
    segment_id: str
    parent_chain_id: str
    pair_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    observations: tuple[S13EdgeComponentObservation, ...]
    left_cut_reason: str | None
    right_cut_reason: str | None
    global_bbox_xyxy: tuple[int, int, int, int]
    segment_sha256: str

    @classmethod
    def create(
        cls,
        *,
        parent_chain_id: str,
        observations: Sequence[S13EdgeComponentObservation],
        left_cut_reason: str | None = None,
        right_cut_reason: str | None = None,
    ) -> "S13ComponentApplicationSegment":
        rows = tuple(sorted(observations, key=lambda row: (row.pair_index, row.component_id)))
        if not rows:
            raise ValueError("C2E segment must contain an observation")
        bbox = (
            min(row.global_bbox_xyxy[0] for row in rows),
            min(row.global_bbox_xyxy[1] for row in rows),
            max(row.global_bbox_xyxy[2] for row in rows),
            max(row.global_bbox_xyxy[3] for row in rows),
        )
        payload = {
            "parent_chain_id": parent_chain_id,
            "observations": [(row.pair_index, row.component_id, row.mask_sha256) for row in rows],
            "left_cut_reason": left_cut_reason, "right_cut_reason": right_cut_reason,
        }
        sha = _json_sha(payload)
        return cls(
            segment_id=f"c2e-segment-{sha[:20]}", parent_chain_id=parent_chain_id,
            pair_indices=tuple(row.pair_index for row in rows),
            source_indices=tuple(sorted({source for row in rows for source in row.source_indices})),
            observations=rows, left_cut_reason=left_cut_reason, right_cut_reason=right_cut_reason,
            global_bbox_xyxy=bbox, segment_sha256=sha,
        )


def split_s13_edge_component_chain(
    chain: S13EdgeComponentChain,
    *,
    config: object,
) -> tuple[S13ComponentApplicationSegment, ...]:
    del config  # Initial evidence segmentation is independent of recursive preview limits.
    segments: list[S13ComponentApplicationSegment] = []
    current: list[S13EdgeComponentObservation] = []
    left_reason: str | None = None
    previous_pair: int | None = None
    pending_reason: str | None = None

    def flush(right_reason: str | None) -> None:
        nonlocal current, left_reason
        if current:
            segments.append(S13ComponentApplicationSegment.create(
                parent_chain_id=chain.chain_id, observations=current,
                left_cut_reason=left_reason, right_cut_reason=right_reason,
            ))
            current = []

    for observation in sorted(chain.observations, key=lambda row: (row.pair_index, row.component_id)):
        invalid_reason = None
        if observation.evidence_state == "ambiguous":
            invalid_reason = "ambiguous_observation"
        elif observation.evidence_state == "unevaluable":
            invalid_reason = "unevaluable_observation"
        if invalid_reason is not None:
            flush(invalid_reason)
            pending_reason = invalid_reason
            previous_pair = observation.pair_index
            continue
        if previous_pair is not None and observation.pair_index != previous_pair + 1:
            flush("pair_gap")
            left_reason = "pair_gap"
        elif pending_reason is not None:
            left_reason = pending_reason
        current.append(observation)
        previous_pair = observation.pair_index
        pending_reason = None
    flush(None)
    return tuple(segments)


def select_s13_component_segment_split_pair(
    pair_audits: Sequence[Mapping[str, object]],
) -> int:
    """Choose the frozen lexicographically worst, then lowest-index split edge."""

    if not pair_audits:
        raise ValueError("C2E split selection requires at least one pair audit")
    rows: list[tuple[tuple[float, float, float, float, int], int]] = []
    for audit in pair_audits:
        pair_index = int(audit["pair_index"])
        values = (
            float(audit.get("hard_violation_count", 0)),
            float(audit.get("maximum_normalized_gate_excess", 0.0)),
            float(audit.get("normalized_solver_residual", 0.0)),
            float(audit.get("baseline_edge_excess_px", 0.0)),
        )
        if pair_index < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("C2E split audit contains invalid values")
        rows.append(((*values, -pair_index), pair_index))
    return max(rows, key=lambda row: row[0])[1]


def _observation_lag(observation: S13EdgeComponentObservation) -> float:
    return 0.5 * (observation.forward_best_lag_px - observation.reverse_best_lag_px)


def solve_s13_component_segment_source_offsets(
    segment: S13ComponentApplicationSegment,
    *,
    config: object,
    owner_support_by_source: Mapping[int, int | float] | None = None,
) -> Mapping[int, float]:
    """Solve source normal offsets with one owner-weighted zero-mean gauge."""

    sources = tuple(segment.source_indices)
    if len(sources) < 2:
        raise ValueError("C2E segment has fewer than two sources")
    source_column = {source: index for index, source in enumerate(sources)}
    pair_rows: list[np.ndarray] = []
    pair_targets: list[float] = []
    pair_base_weights: list[float] = []
    for observation in sorted(segment.observations, key=lambda row: row.pair_index):
        row = np.zeros(len(sources), dtype=np.float64)
        row[source_column[observation.source_indices[0]]] = -1.0
        row[source_column[observation.source_indices[1]]] = 1.0
        support_confidence = min(
            observation.forward_valid_samples, observation.reverse_valid_samples
        ) / max(observation.reference_support_samples, 1)
        weight = observation.correlation * np.clip(
            observation.uniqueness_fraction / 0.10, 0.0, 2.0
        ) * np.clip(support_confidence, 0.0, 1.0)
        if not math.isfinite(float(weight)) or weight <= 0.0:
            raise ValueError("C2E solver observation has zero or invalid weight")
        pair_rows.append(row)
        pair_targets.append(-_observation_lag(observation))
        pair_base_weights.append(float(weight))
    if not pair_rows:
        raise ValueError("C2E segment has no solvable observations")
    support = np.asarray([
        float((owner_support_by_source or {}).get(source, 1.0)) for source in sources
    ], np.float64)
    if not np.isfinite(support).all() or np.any(support < 0.0) or support.sum() <= 0.0:
        raise ValueError("C2E owner-support gauge is unevaluable")
    gauge = support / support.sum()
    regularization = float(config.source_offset_regularization)
    if not math.isfinite(regularization) or regularization < 0.0:
        raise ValueError("C2E solver regularization is invalid")
    second_rows = []
    if regularization > 0.0:
        for index in range(1, len(sources) - 1):
            row = np.zeros(len(sources), dtype=np.float64)
            row[index - 1:index + 2] = (1.0, -2.0, 1.0)
            second_rows.append(math.sqrt(regularization) * row)
    robust = np.ones(len(pair_rows), dtype=np.float64)
    solution = np.zeros(len(sources), dtype=np.float64)
    final_design = np.empty((0, len(sources)), np.float64)
    for _iteration in range(3):
        weighted = np.sqrt(np.asarray(pair_base_weights) * robust)
        design_parts = [np.asarray(pair_rows) * weighted[:, None]]
        target_parts = [np.asarray(pair_targets) * weighted]
        if second_rows:
            design_parts.append(np.asarray(second_rows))
            target_parts.append(np.zeros(len(second_rows)))
        design_parts.append(gauge[None, :])
        target_parts.append(np.zeros(1))
        final_design = np.vstack(design_parts)
        target = np.concatenate(target_parts)
        solution = np.linalg.lstsq(final_design, target, rcond=None)[0]
        residuals = np.asarray(pair_rows) @ solution - np.asarray(pair_targets)
        delta = float(config.solver_huber_delta_px)
        robust = np.minimum(1.0, delta / np.maximum(np.abs(residuals), 1e-6))
    condition = float(np.linalg.cond(final_design))
    if not math.isfinite(condition) or condition > float(config.maximum_solver_condition_number):
        raise ValueError("C2E solver condition number exceeded")
    residuals = np.asarray(pair_rows) @ solution - np.asarray(pair_targets)
    median = float(np.median(residuals))
    sigma = max(
        float(config.normal_search_step_px),
        1.4826 * float(np.median(np.abs(residuals - median))),
    )
    normalized = np.abs(residuals) / sigma
    if np.any(normalized > float(config.maximum_normalized_solver_residual)):
        raise ValueError("C2E solver normalized residual exceeded")
    maximum = float(config.maximum_source_normal_offset_px)
    if not np.isfinite(solution).all() or np.any(np.abs(solution) > maximum + 1e-9):
        raise ValueError("C2E solver source offset exceeded")
    if np.any(np.abs(np.diff(solution)) > maximum + 1e-9):
        raise ValueError("C2E solver adjacent offset difference exceeded")
    return MappingProxyType({source: float(solution[index]) for index, source in enumerate(sources)})


@dataclass(frozen=True)
class S13ComponentQualityEvaluation:
    decision: CandidateDecision
    rejection_reasons: tuple[str, ...]
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))
        object.__setattr__(self, "audit", _frozen_mapping(self.audit))


def _finite_metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"C2E quality metric is missing: {name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"C2E quality metric is nonfinite: {name}")
    return number


def evaluate_s13_component_candidate_quality(
    *,
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    hard_gates: Mapping[str, bool],
    config: object,
) -> S13ComponentQualityEvaluation:
    """Apply the mutually exclusive resolved/improved/rejected C2E quality rules."""

    failed_hard_gates = tuple(sorted(name for name, passed in hard_gates.items() if passed is not True))
    if failed_hard_gates:
        return S13ComponentQualityEvaluation(
            decision="rejected",
            rejection_reasons=tuple(f"hard_gate_failed:{name}" for name in failed_hard_gates),
            audit={"hard_gate_failures": failed_hard_gates},
        )
    baseline_p95 = _finite_metric(baseline, "edge_p95_px")
    candidate_p95 = _finite_metric(candidate, "edge_p95_px")
    baseline_step = _finite_metric(baseline, "maximum_step_px")
    candidate_step = _finite_metric(candidate, "maximum_step_px")
    baseline_break = _finite_metric(baseline, "break_length_px") + _finite_metric(
        baseline, "double_edge_length_px"
    )
    candidate_break = _finite_metric(candidate, "break_length_px") + _finite_metric(
        candidate, "double_edge_length_px"
    )
    baseline_non_target = _finite_metric(baseline, "non_target_p95_px")
    candidate_non_target = _finite_metric(candidate, "non_target_p95_px")
    absolute_improvement = baseline_p95 - candidate_p95
    relative_improvement = absolute_improvement / max(abs(baseline_p95), 1e-12)
    if baseline_break >= 4.0:
        break_reduction = (baseline_break - candidate_break) / baseline_break
    else:
        break_reduction = 0.0
    no_break_regression = candidate_break <= baseline_break
    search_boundary_hit = candidate.get("search_boundary_hit") is True
    resolved = bool(
        not search_boundary_hit
        and candidate_p95 <= float(config.maximum_post_edge_p95_px)
        and candidate_step <= float(config.maximum_post_edge_step_px)
        and no_break_regression
        and (
            absolute_improvement >= float(config.minimum_resolved_improvement_px)
            or candidate_break < baseline_break
        )
    )
    substantive_improvement = bool(
        absolute_improvement >= float(config.minimum_absolute_improvement_px)
        or relative_improvement >= float(config.minimum_relative_improvement_fraction)
        or (
            baseline_break >= 4.0
            and break_reduction >= float(config.minimum_break_length_reduction_fraction)
            and candidate_break <= math.floor(0.5 * baseline_break)
        )
    )
    improved = bool(
        candidate_p95 < baseline_p95
        and substantive_improvement
        and candidate_step - baseline_step <= float(config.maximum_non_target_step_regression_px)
        and candidate_non_target - baseline_non_target <= float(
            config.maximum_non_target_p95_regression_px
        )
        and no_break_regression
    )
    audit = {
        "baseline_edge_p95_px": baseline_p95,
        "candidate_edge_p95_px": candidate_p95,
        "absolute_improvement_px": absolute_improvement,
        "relative_improvement_fraction": relative_improvement,
        "baseline_break_double_edge_union_length": baseline_break,
        "candidate_break_double_edge_union_length": candidate_break,
        "break_length_reduction_fraction": break_reduction,
        "search_boundary_hit": search_boundary_hit,
        "hard_gate_failures": (),
    }
    if resolved:
        return S13ComponentQualityEvaluation("resolved", (), audit)
    if improved:
        return S13ComponentQualityEvaluation("improved_unresolved", (), audit)
    reasons = []
    if candidate_p95 >= baseline_p95:
        reasons.append("edge_p95_not_improved")
    if not substantive_improvement:
        reasons.append("no_substantive_improvement")
    if not no_break_regression:
        reasons.append("break_or_double_edge_regressed")
    if candidate_step - baseline_step > float(config.maximum_non_target_step_regression_px):
        reasons.append("maximum_step_regressed")
    if candidate_non_target - baseline_non_target > float(
        config.maximum_non_target_p95_regression_px
    ):
        reasons.append("non_target_p95_regressed")
    return S13ComponentQualityEvaluation(
        "rejected", tuple(reasons or ("quality_gain_not_accepted",)), audit
    )


@dataclass(frozen=True)
class S13SourceComponentCorrection:
    segment_id: str
    source_index: int
    frame_id: int
    x0: int
    x1: int
    y0: int
    y1: int
    delta_u: np.ndarray
    delta_v: np.ndarray
    weight: np.ndarray
    correction_sha256: str

    def __post_init__(self) -> None:
        shape = (self.y1 - self.y0, self.x1 - self.x0)
        if self.x0 < 0 or self.y0 < 0 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("C2E correction domain is invalid")
        for name in ("delta_u", "delta_v", "weight"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("C2E correction array is invalid")
            object.__setattr__(self, name, _readonly(value, np.float32))
        if np.any((self.weight < 0.0) | (self.weight > 1.0)):
            raise ValueError("C2E correction weights are outside [0,1]")
        if np.any((self.weight == 0.0) & ((self.delta_u != 0.0) | (self.delta_v != 0.0))):
            raise ValueError("C2E correction leaks outside its weight support")
        if len(self.correction_sha256) != 64:
            raise ValueError("C2E correction SHA is invalid")


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def build_s13_source_component_correction(
    *,
    segment_id: str,
    source_index: int,
    frame_id: int,
    support_mask: np.ndarray,
    source_offset_px: float,
    normal_xy: tuple[float, float],
    config: object,
) -> S13SourceComponentCorrection:
    mask = np.asarray(support_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError("C2E correction exact support is empty")
    if not math.isfinite(source_offset_px):
        raise ValueError("C2E correction offset is nonfinite")
    normal = np.asarray(normal_xy, np.float64)
    if normal.shape != (2,) or not np.isfinite(normal).all() or not math.isclose(
        float(np.linalg.norm(normal)), 1.0, abs_tol=1e-6
    ):
        raise ValueError("C2E correction normal is invalid")
    radius = int(config.normal_taper_radius_px)
    core_radius = int(config.edge_core_radius_px)
    rows, columns = np.nonzero(mask)
    x0, x1 = max(0, int(columns.min()) - radius), min(mask.shape[1], int(columns.max()) + radius + 1)
    y0, y1 = max(0, int(rows.min()) - radius), min(mask.shape[0], int(rows.max()) + radius + 1)
    local = mask[y0:y1, x0:x1]
    distance = cv2.distanceTransform((~local).astype(np.uint8), cv2.DIST_L2, 5)
    if radius <= core_radius:
        weight = (distance <= core_radius).astype(np.float32)
    else:
        weight = 1.0 - _smoothstep((distance - core_radius) / (radius - core_radius))
        weight[distance >= radius] = 0.0
    yy, xx = np.indices(local.shape, dtype=np.float64)
    global_x, global_y = xx + x0, yy + y0
    tangent = np.asarray((-normal[1], normal[0]))
    support_projection = columns * tangent[0] + rows * tangent[1]
    projection = global_x * tangent[0] + global_y * tangent[1]
    endpoint = float(config.endpoint_taper_px)
    if endpoint > 0.0:
        left = _smoothstep((projection - float(support_projection.min())) / endpoint)
        right = _smoothstep((float(support_projection.max()) - projection) / endpoint)
        endpoint_weight = np.minimum(left, right)
        if float(endpoint_weight.max()) > 0.0:
            endpoint_weight /= float(endpoint_weight.max())
        weight *= endpoint_weight.astype(np.float32)
    weight[~np.isfinite(weight)] = 0.0
    weight = np.clip(weight, 0.0, 1.0).astype(np.float32)
    delta_u = weight * np.float32(source_offset_px * normal[0])
    delta_v = weight * np.float32(source_offset_px * normal[1])
    payload = {
        "segment_id": segment_id, "source_index": source_index, "frame_id": frame_id,
        "domain_xyxy": (x0, y0, x1, y1), "array_sha256": _array_sha(delta_u, delta_v, weight),
    }
    return S13SourceComponentCorrection(
        segment_id=segment_id, source_index=source_index, frame_id=frame_id,
        x0=x0, x1=x1, y0=y0, y1=y1,
        delta_u=delta_u, delta_v=delta_v, weight=weight,
        correction_sha256=_json_sha(payload),
    )


def compose_s13_source_component_deltas(
    *,
    base_delta_u: np.ndarray,
    base_delta_v: np.ndarray,
    domain_xyxy: tuple[int, int, int, int],
    corrections: Sequence[S13SourceComponentCorrection],
    field_ids: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose ``base + winner`` once in target/canvas delta coordinates."""

    combined_u = np.array(base_delta_u, dtype=np.float32, copy=True, order="C")
    combined_v = np.array(base_delta_v, dtype=np.float32, copy=True, order="C")
    if combined_u.shape != combined_v.shape or combined_u.ndim != 2:
        raise ValueError("C2E base delta shapes disagree")
    x0, y0, x1, y1 = domain_xyxy
    if combined_u.shape != (y1 - y0, x1 - x0) or x0 < 0 or y0 < 0:
        raise ValueError("C2E base delta domain disagrees with its arrays")
    if not np.isfinite(combined_u).all() or not np.isfinite(combined_v).all():
        raise ValueError("C2E base delta contains nonfinite values")
    ordered_segments = sorted({row.segment_id for row in corrections})
    ids = dict(field_ids or {segment_id: index for index, segment_id in enumerate(ordered_segments)})
    if set(ids) != set(ordered_segments) or len(set(ids.values())) != len(ids) or any(
        not isinstance(value, int) or value < 0 for value in ids.values()
    ):
        raise ValueError("C2E correction field-ID table is invalid")
    labels = np.full(combined_u.shape, -1, dtype=np.int32)
    for correction in sorted(corrections, key=lambda row: (row.segment_id, row.source_index)):
        overlap_x0, overlap_x1 = max(x0, correction.x0), min(x1, correction.x1)
        overlap_y0, overlap_y1 = max(y0, correction.y0), min(y1, correction.y1)
        if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
            continue
        target = np.s_[
            overlap_y0 - y0:overlap_y1 - y0,
            overlap_x0 - x0:overlap_x1 - x0,
        ]
        source = np.s_[
            overlap_y0 - correction.y0:overlap_y1 - correction.y0,
            overlap_x0 - correction.x0:overlap_x1 - correction.x0,
        ]
        active = correction.weight[source] > 0.0
        if np.any(active & (labels[target] >= 0)):
            raise ValueError("C2E correction fields overlap after conflict resolution")
        target_u, target_v, target_labels = combined_u[target], combined_v[target], labels[target]
        target_u[active] += correction.delta_u[source][active]
        target_v[active] += correction.delta_v[source][active]
        target_labels[active] = ids[correction.segment_id]
    return _readonly(combined_u, np.float32), _readonly(combined_v, np.float32), _readonly(
        labels, np.int32
    )


@dataclass(frozen=True)
class S13ComponentSegmentCandidate:
    segment: S13ComponentApplicationSegment
    source_offsets_px: tuple[float, ...]
    gain: float
    corrections: tuple[S13SourceComponentCorrection, ...]
    decision: CandidateDecision
    rejection_reasons: tuple[str, ...]
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.decision not in {"resolved", "improved_unresolved", "rejected", "budget_deferred"}:
            raise ValueError("C2E segment candidate decision is invalid")
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise ValueError("C2E segment candidate gain is invalid")
        if not all(math.isfinite(value) for value in self.source_offsets_px):
            raise ValueError("C2E segment candidate offsets are nonfinite")
        object.__setattr__(self, "corrections", tuple(self.corrections))
        object.__setattr__(self, "rejection_reasons", tuple(self.rejection_reasons))
        object.__setattr__(self, "audit", _frozen_mapping(self.audit))


@dataclass(frozen=True)
class S13ComponentPatchSet:
    accepted_segment_ids: tuple[str, ...]
    rejected_segment_ids: tuple[str, ...]
    deferred_segment_ids: tuple[str, ...]
    application_state: Literal["complete", "partial", "none"]
    corrections_by_source: Mapping[int, tuple[S13SourceComponentCorrection, ...]]
    unresolved_regions: tuple[Mapping[str, object], ...]
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        corrections = {
            int(source): tuple(rows) for source, rows in self.corrections_by_source.items()
        }
        object.__setattr__(self, "corrections_by_source", MappingProxyType(corrections))
        object.__setattr__(self, "unresolved_regions", tuple(
            _frozen_mapping(row) for row in self.unresolved_regions
        ))
        object.__setattr__(self, "audit", _frozen_mapping(self.audit))


@dataclass(frozen=True)
class SourceCorrectionRegistry:
    """Frozen sole authority for every applied source-local correction field."""

    application_state: Literal["complete", "partial", "none"]
    accepted_segment_ids: tuple[str, ...]
    corrections_by_source: Mapping[int, tuple[S13SourceComponentCorrection, ...]]
    field_id_by_segment: Mapping[str, int]

    def __post_init__(self) -> None:
        accepted = tuple(sorted(str(value) for value in self.accepted_segment_ids))
        corrections = {
            int(source): tuple(sorted(rows, key=lambda row: row.segment_id))
            for source, rows in self.corrections_by_source.items()
        }
        fields = {str(key): int(value) for key, value in self.field_id_by_segment.items()}
        contributors = {
            row.segment_id for rows in corrections.values() for row in rows
        }
        if contributors != set(accepted) or set(fields) != set(accepted):
            raise ValueError("C2E correction registry authority is incomplete")
        if len(set(fields.values())) != len(fields) or any(value < 0 for value in fields.values()):
            raise ValueError("C2E correction registry field IDs are invalid")
        if self.application_state == "none" and accepted:
            raise ValueError("C2E none registry cannot contain corrections")
        if self.application_state != "none" and not accepted:
            raise ValueError("C2E applied registry must contain corrections")
        object.__setattr__(self, "accepted_segment_ids", accepted)
        object.__setattr__(self, "corrections_by_source", MappingProxyType(corrections))
        object.__setattr__(self, "field_id_by_segment", MappingProxyType(fields))


def freeze_s13_source_correction_registry(
    patch_set: S13ComponentPatchSet,
) -> SourceCorrectionRegistry:
    segment_ids = tuple(sorted(patch_set.accepted_segment_ids))
    return SourceCorrectionRegistry(
        application_state=patch_set.application_state,
        accepted_segment_ids=segment_ids,
        corrections_by_source=patch_set.corrections_by_source,
        field_id_by_segment={
            segment_id: index for index, segment_id in enumerate(segment_ids)
        },
    )


def _corrections_overlap(
    left: S13SourceComponentCorrection, right: S13SourceComponentCorrection
) -> bool:
    if left.source_index != right.source_index:
        return False
    x0, x1 = max(left.x0, right.x0), min(left.x1, right.x1)
    y0, y1 = max(left.y0, right.y0), min(left.y1, right.y1)
    if x1 <= x0 or y1 <= y0:
        return False
    left_roi = left.weight[y0 - left.y0:y1 - left.y0, x0 - left.x0:x1 - left.x0]
    right_roi = right.weight[y0 - right.y0:y1 - right.y0, x0 - right.x0:x1 - right.x0]
    return bool(np.any((left_roi > 0.0) & (right_roi > 0.0)))


def _candidate_utility(candidate: S13ComponentSegmentCandidate) -> tuple[float, ...]:
    audit = candidate.audit
    return (
        float(audit.get("rescued_severe_seam_count", 0)),
        float(audit.get("worst_seam_absolute_improvement", 0.0)),
        float(audit.get("supported_unique_edge_columns", 0)),
        -float(audit.get("post_maximum_step", math.inf)),
        -float(audit.get("correction_energy", math.inf)),
    )


def select_s13_component_patch_set(
    candidates: Sequence[S13ComponentSegmentCandidate],
    *,
    config: object,
    obligations: Sequence[S13BaselineC2EObligation] = (),
) -> S13ComponentPatchSet:
    """Select a deterministic maximum compatible set without vector merging."""

    accepted_candidates = [
        row for row in candidates if row.decision in {"resolved", "improved_unresolved"}
    ]
    rejected = [row.segment.segment_id for row in candidates if row.decision == "rejected"]
    deferred = [row.segment.segment_id for row in candidates if row.decision == "budget_deferred"]
    conflicts: set[tuple[int, int]] = set()
    for left in range(len(accepted_candidates)):
        for right in range(left + 1, len(accepted_candidates)):
            if any(
                _corrections_overlap(a, b)
                for a in accepted_candidates[left].corrections
                for b in accepted_candidates[right].corrections
            ):
                conflicts.add((left, right))
    maximum_exact = int(getattr(config, "maximum_exact_conflict_group_nodes", 12))
    chosen_indices: tuple[int, ...]
    if len(accepted_candidates) <= maximum_exact:
        compatible: list[tuple[tuple[float, ...], tuple[str, ...], tuple[int, ...]]] = []
        for mask in range(1 << len(accepted_candidates)):
            indices = tuple(index for index in range(len(accepted_candidates)) if mask & (1 << index))
            if any(left in indices and right in indices for left, right in conflicts):
                continue
            utility_rows = [_candidate_utility(accepted_candidates[index]) for index in indices]
            utility = (
                sum(row[0] for row in utility_rows),
                sum(row[1] for row in utility_rows),
                sum(row[2] for row in utility_rows),
                min((row[3] for row in utility_rows), default=0.0),
                sum(row[4] for row in utility_rows),
            )
            ids = tuple(sorted(accepted_candidates[index].segment.segment_id for index in indices))
            compatible.append((utility, ids, indices))
        best_utility = max(row[0] for row in compatible)
        chosen_indices = min(
            (row for row in compatible if row[0] == best_utility), key=lambda row: row[1]
        )[2]
    else:
        order = sorted(
            range(len(accepted_candidates)),
            key=lambda index: (tuple(-value for value in _candidate_utility(accepted_candidates[index])),
                               accepted_candidates[index].segment.segment_id),
        )
        chosen: list[int] = []
        for index in order:
            if not any((min(index, other), max(index, other)) in conflicts for other in chosen):
                chosen.append(index)
        chosen_indices = tuple(sorted(chosen))
    chosen_set = set(chosen_indices)
    conflict_losers = [
        row.segment.segment_id for index, row in enumerate(accepted_candidates)
        if index not in chosen_set
    ]
    rejected.extend(conflict_losers)
    winners = [accepted_candidates[index] for index in chosen_indices]
    corrections: dict[int, list[S13SourceComponentCorrection]] = {}
    for candidate in winners:
        for correction in candidate.corrections:
            corrections.setdefault(correction.source_index, []).append(correction)
    corrections_frozen = {
        source: tuple(sorted(rows, key=lambda row: row.segment_id))
        for source, rows in corrections.items()
    }
    accepted_ids = tuple(sorted(row.segment.segment_id for row in winners))
    unresolved = tuple(
        {"segment_id": row.segment.segment_id, "state": row.decision}
        for row in candidates if row.decision != "resolved" or row.segment.segment_id in rejected
    )
    severe_obligations = [row for row in obligations if row.severe and row.evaluable]
    resolved_observations = {
        (
            observation.pair_index,
            observation.component_id,
            observation.mask_sha256,
        )
        for candidate in winners if candidate.decision == "resolved"
        for observation in candidate.segment.observations
    }
    complete = bool(severe_obligations) and all(
        (
            obligation.pair_index,
            obligation.component_id,
            obligation.support_sha256,
        ) in resolved_observations
        for obligation in severe_obligations
    )
    state: Literal["complete", "partial", "none"]
    if not winners:
        state = "none"
    elif complete or (not obligations and not rejected and not deferred and all(
        row.decision == "resolved" for row in winners
    )):
        state = "complete"
    else:
        state = "partial"
    return S13ComponentPatchSet(
        accepted_segment_ids=accepted_ids,
        rejected_segment_ids=tuple(sorted(set(rejected))),
        deferred_segment_ids=tuple(sorted(set(deferred))),
        application_state=state,
        corrections_by_source=corrections_frozen,
        unresolved_regions=unresolved,
        audit={
            "candidate_count": len(candidates), "conflict_edges": sorted(conflicts),
            "repair_complete": complete,
        },
    )


@dataclass(frozen=True)
class SourceMapOracle:
    source_index: int
    domain_xyxy: tuple[int, int, int, int]
    u: np.ndarray
    v: np.ndarray
    valid: np.ndarray
    field_id: np.ndarray
    oracle_sha256: str

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.domain_xyxy
        shape = (y1 - y0, x1 - x0)
        if self.source_index < 0 or x0 < 0 or y0 < 0 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("C2E source-map oracle domain is invalid")
        expected = {
            "u": np.dtype("<f4"), "v": np.dtype("<f4"),
            "valid": np.dtype("u1"), "field_id": np.dtype("<i4"),
        }
        for name, dtype in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError("C2E source-map oracle shape disagrees with its domain")
            object.__setattr__(self, name, _readonly(value, dtype))
        if len(self.oracle_sha256) != 64:
            raise ValueError("C2E source-map oracle SHA is invalid")


def source_map_oracle_from_arrays(
    *,
    source_index: int,
    domain_xyxy: tuple[int, int, int, int],
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    field_id: np.ndarray,
) -> SourceMapOracle:
    valid_u8 = np.array(valid, dtype=np.uint8, copy=True, order="C")
    source_u = np.array(u, dtype="<f4", copy=True, order="C")
    source_v = np.array(v, dtype="<f4", copy=True, order="C")
    labels = np.array(field_id, dtype="<i4", copy=True, order="C")
    if source_u.shape != source_v.shape or source_u.shape != valid_u8.shape or source_u.shape != labels.shape:
        raise ValueError("C2E source-map oracle arrays have different shapes")
    valid_mask = valid_u8 != 0
    if np.any(valid_mask & (~np.isfinite(source_u) | ~np.isfinite(source_v))):
        raise ValueError("C2E source-map oracle has nonfinite valid UV")
    source_u[~valid_mask], source_v[~valid_mask], labels[~valid_mask] = 0.0, 0.0, -1
    source_u[source_u == 0.0] = 0.0
    source_v[source_v == 0.0] = 0.0
    header = {
        "schema": "gemini305-video-s13-source-map-oracle/v1",
        "source_index": int(source_index), "domain_xyxy": tuple(int(value) for value in domain_xyxy),
        "shape": source_u.shape,
        "dtypes": {"u": "<f4", "v": "<f4", "valid": "u1", "field_id": "<i4"},
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for array in (source_u, source_v, valid_u8, labels):
        digest.update(array.tobytes(order="C"))
    return SourceMapOracle(
        source_index=int(source_index), domain_xyxy=tuple(int(value) for value in domain_xyxy),
        u=source_u, v=source_v, valid=valid_u8, field_id=labels,
        oracle_sha256=digest.hexdigest(),
    )


__all__ = [
    "S13BaselineC2EObligation", "S13ComponentApplicationSegment",
    "S13ComponentPatchSet", "S13ComponentQualityEvaluation", "S13ComponentSegmentCandidate", "S13EdgeComponentChain",
    "S13EdgeComponentObservation", "S13ExactEdgeComponentEvidence",
    "S13SourceComponentCorrection", "SourceMapOracle",
    "build_s13_edge_component_chains", "build_s13_source_component_correction",
    "compose_s13_source_component_deltas",
    "canonical_s13_support_sha256",
    "canonical_s13_normal_search_lags", "freeze_s13_baseline_c2e_obligations",
    "evaluate_s13_component_candidate_quality", "make_s13_edge_component_observation",
    "make_s13_unevaluable_edge_component_observation",
    "select_s13_component_patch_set", "select_s13_component_segment_split_pair",
    "solve_s13_component_segment_source_offsets", "source_map_oracle_from_arrays",
    "split_s13_edge_component_chain",
]
