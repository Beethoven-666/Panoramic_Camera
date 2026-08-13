"""Pure component-chain C2E primitives for the isolated S1.3 M5.1-r4 candidate.

The module deliberately has no stage-writing or renderer authority.  It turns
exact, pair-local edge evidence into deterministic chains, application
segments, source-local correction fields, a conflict-resolved patch registry,
and canonical source-map oracle arrays.  The M5 orchestrator is responsible for
supplying immutable P0-derived maps and for applying only audited patch sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

import cv2
import numpy as np


EvidenceState = Literal["actionable", "safe_anchor", "ambiguous", "unevaluable"]
CandidateDecision = Literal[
    "resolved", "improved_unresolved", "rejected", "budget_deferred"
]


def locate_s13_dependency_failure_subset(
    candidate_ids: Sequence[str],
    *,
    pixel_support_by_candidate: Mapping[str, np.ndarray],
    evaluate_subset: Callable[
        [tuple[str, ...]], tuple[bool, Mapping[str, object]]
    ],
    utility_by_candidate: Mapping[str, tuple[object, ...]],
) -> tuple[str, Mapping[str, object]]:
    """Locate a composite failure before choosing one candidate to remove.

    Pairwise support overlap first excludes unrelated candidates.  Leave-one-out
    then identifies necessary offenders; when no single removal repairs the
    composite, deterministic ddmin finds a smaller failing subset.  Utility is
    consulted only inside the localized failing set.
    """

    ids = tuple(sorted(str(value) for value in candidate_ids))
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError("C2E dependency failure locator requires unique candidates")
    if set(ids) != set(pixel_support_by_candidate) or set(ids) != set(
        utility_by_candidate
    ):
        raise ValueError("C2E dependency failure locator authority is incomplete")
    canonical_support: dict[str, np.ndarray] = {}
    support_sets: dict[str, set[tuple[int, int]]] = {}
    for candidate_id in ids:
        support = np.asarray(pixel_support_by_candidate[candidate_id], np.int32)
        if support.ndim != 2 or support.shape[1] != 2:
            raise ValueError("C2E dependency pixel support must be xy coordinates")
        support = np.unique(support, axis=0)
        canonical_support[candidate_id] = support
        support_sets[candidate_id] = {
            (int(row[0]), int(row[1])) for row in support
        }
    overlap_rows: list[dict[str, object]] = []
    overlap_pixels: set[tuple[int, int]] = set()
    localized: set[str] = set()
    for left_index, left_id in enumerate(ids):
        for right_id in ids[left_index + 1:]:
            overlap = support_sets[left_id] & support_sets[right_id]
            overlap_rows.append({
                "left_segment_id": left_id,
                "right_segment_id": right_id,
                "overlap_pixel_count": len(overlap),
                "overlap_support_sha256": _array_sha(
                    np.asarray(sorted(overlap), dtype="<i4").reshape(-1, 2)
                ),
            })
            if overlap:
                localized.update((left_id, right_id))
                overlap_pixels.update(overlap)
    localized_ids = tuple(sorted(localized)) if len(localized) >= 2 else ids
    evaluations: list[dict[str, object]] = []

    def evaluate(kind: str, subset: Sequence[str]) -> bool:
        subset_ids = tuple(sorted(subset))
        passed, audit = evaluate_subset(subset_ids)
        evaluations.append({
            "kind": kind,
            "candidate_ids": list(subset_ids),
            "passed": bool(passed),
            "audit": dict(audit),
        })
        return bool(passed)

    necessary: list[str] = []
    for candidate_id in localized_ids:
        remainder = tuple(row for row in localized_ids if row != candidate_id)
        if remainder and evaluate("leave_one_out", remainder):
            necessary.append(candidate_id)
    ddmin_trace: list[dict[str, object]] = []
    if necessary:
        minimal_failure_ids = tuple(sorted(necessary))
        method = "leave_one_out"
    else:
        current = localized_ids
        granularity = 2
        while len(current) >= 2:
            chunk_size = int(math.ceil(len(current) / granularity))
            chunks = tuple(
                current[index:index + chunk_size]
                for index in range(0, len(current), chunk_size)
            )
            reduced: tuple[str, ...] | None = None
            for chunk in chunks:
                if len(chunk) >= 2 and not evaluate("ddmin_chunk", chunk):
                    reduced = chunk
                    ddmin_trace.append({
                        "operation": "failing_chunk", "candidate_ids": list(chunk)
                    })
                    break
            if reduced is None:
                for chunk in chunks:
                    complement = tuple(row for row in current if row not in set(chunk))
                    if len(complement) >= 2 and not evaluate(
                        "ddmin_complement", complement
                    ):
                        reduced = complement
                        ddmin_trace.append({
                            "operation": "failing_complement",
                            "candidate_ids": list(complement),
                        })
                        break
            if reduced is not None:
                current = tuple(sorted(reduced))
                granularity = max(2, granularity - 1)
            elif granularity >= len(current):
                break
            else:
                granularity = min(len(current), granularity * 2)
        minimal_failure_ids = current
        method = "deterministic_ddmin"
    removed = min(
        minimal_failure_ids,
        key=lambda row: (utility_by_candidate[row], row),
    )
    audit = {
        "schema": "gemini305-video-s13-dependency-failure-localization/v1",
        "candidate_ids": list(ids),
        "localized_candidate_ids": list(localized_ids),
        "minimal_failure_candidate_ids": list(minimal_failure_ids),
        "selected_removal_segment_id": removed,
        "localization_method": method,
        "overlap_pixel_count": len(overlap_pixels),
        "overlap_support_sha256": _array_sha(
            np.asarray(sorted(overlap_pixels), dtype="<i4").reshape(-1, 2)
        ),
        "candidate_support_authority": [
            {
                "segment_id": candidate_id,
                "pixel_count": int(canonical_support[candidate_id].shape[0]),
                "support_sha256": _array_sha(
                    np.asarray(canonical_support[candidate_id], dtype="<i4")
                ),
            }
            for candidate_id in ids
        ],
        "pairwise_support_overlap": overlap_rows,
        "subset_evaluations": evaluations,
        "ddmin_trace": ddmin_trace,
    }
    return removed, MappingProxyType(audit)


class S13ComponentSolverError(ValueError):
    """A localizable deterministic solver failure with frozen per-pair audits."""

    def __init__(
        self, message: str, *, pair_audits: Sequence[Mapping[str, object]] = ()
    ) -> None:
        super().__init__(message)
        self.pair_audits = tuple(MappingProxyType(dict(row)) for row in pair_audits)


def audit_s13_component_candidate_map_safety(
    *,
    base_u: np.ndarray,
    base_v: np.ndarray,
    base_valid: np.ndarray,
    candidate_u: np.ndarray,
    candidate_v: np.ndarray,
    candidate_valid: np.ndarray,
    owner_mask: np.ndarray,
    evidence_mask: np.ndarray,
    influence_mask: np.ndarray,
    source_size: tuple[int, int],
    minimum_owner_retention: float,
    minimum_evidence_retention: float,
    minimum_jacobian: float,
    maximum_displacement_px: float,
    maximum_halo_regression_px: float,
) -> Mapping[str, object]:
    """Audit one candidate source map before it can enter the frozen registry."""

    arrays = tuple(np.asarray(value) for value in (
        base_u, base_v, base_valid, candidate_u, candidate_v, candidate_valid,
        owner_mask, evidence_mask, influence_mask,
    ))
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise ValueError("C2E candidate map safety array shapes disagree")
    width, height = (int(value) for value in source_size)
    thresholds = (
        minimum_owner_retention, minimum_evidence_retention, minimum_jacobian,
        maximum_displacement_px, maximum_halo_regression_px,
    )
    if width < 1 or height < 1 or not all(math.isfinite(float(value)) for value in thresholds):
        raise ValueError("C2E candidate map safety inputs are invalid")
    base_u64 = np.asarray(base_u, np.float64)
    base_v64 = np.asarray(base_v, np.float64)
    candidate_u64 = np.asarray(candidate_u, np.float64)
    candidate_v64 = np.asarray(candidate_v, np.float64)
    base_valid_bool = np.asarray(base_valid, bool)
    candidate_valid_bool = np.asarray(candidate_valid, bool)
    owner = np.asarray(owner_mask, bool)
    evidence = np.asarray(evidence_mask, bool)
    influence = np.asarray(influence_mask, bool)

    formal_domain = owner & base_valid_bool
    formal_count = int(np.count_nonzero(formal_domain))
    retained_formal = int(np.count_nonzero(formal_domain & candidate_valid_bool))
    owner_retention = retained_formal / max(formal_count, 1)
    owner_valid_exact = bool(np.array_equal(
        base_valid_bool & owner, candidate_valid_bool & owner
    ))
    evidence_domain = evidence & owner & base_valid_bool
    evidence_count = int(np.count_nonzero(evidence_domain))
    retained_evidence = int(np.count_nonzero(evidence_domain & candidate_valid_bool))
    evidence_retention = retained_evidence / max(evidence_count, 1)

    affected = influence & base_valid_bool
    finite = bool(
        np.isfinite(candidate_u64[affected]).all()
        and np.isfinite(candidate_v64[affected]).all()
    )
    in_bounds = bool(
        finite
        and np.all((candidate_u64[affected] >= 0.0) & (candidate_u64[affected] <= width - 1))
        and np.all((candidate_v64[affected] >= 0.0) & (candidate_v64[affected] <= height - 1))
        and np.all(candidate_valid_bool[affected])
    )
    displacement = np.hypot(candidate_u64 - base_u64, candidate_v64 - base_v64)
    maximum_displacement = float(np.max(displacement[influence])) if np.any(influence) else 0.0

    def minimum_map_jacobian(u: np.ndarray, v: np.ndarray) -> float:
        if min(u.shape) < 2 or not np.any(influence):
            return math.inf
        du_dy, du_dx = np.gradient(u)
        dv_dy, dv_dx = np.gradient(v)
        determinant = du_dx * dv_dy - du_dy * dv_dx
        evaluable = influence & np.isfinite(determinant)
        return float(np.min(determinant[evaluable])) if np.any(evaluable) else -math.inf

    base_minimum_jacobian = minimum_map_jacobian(base_u64, base_v64)
    candidate_minimum_jacobian = minimum_map_jacobian(candidate_u64, candidate_v64)
    jacobian_ratio = (
        candidate_minimum_jacobian / base_minimum_jacobian
        if math.isfinite(base_minimum_jacobian) and abs(base_minimum_jacobian) > 1e-12
        else None
    )

    kernel = np.ones((3, 3), np.uint8)
    # A validation ROI may clip through the middle of a larger correction
    # field.  Only an observed transition to exterior pixels is a true segment
    # boundary; the artificial ROI border is not.
    interior_boundary = influence & cv2.dilate(
        (~influence).astype(np.uint8), kernel, iterations=1
    ).astype(bool)
    halo_regression = (
        float(np.max(displacement[interior_boundary]))
        if np.any(interior_boundary) else 0.0
    )
    displacement_curvature = cv2.Laplacian(displacement, cv2.CV_64F)
    boundary_curvature = (
        float(np.max(np.abs(displacement_curvature[interior_boundary])))
        if np.any(interior_boundary) else 0.0
    )
    exterior = ~influence
    exterior_exact = bool(
        np.array_equal(np.asarray(base_u)[exterior], np.asarray(candidate_u)[exterior])
        and np.array_equal(np.asarray(base_v)[exterior], np.asarray(candidate_v)[exterior])
        and np.array_equal(base_valid_bool[exterior], candidate_valid_bool[exterior])
    )
    passed = bool(
        formal_count > 0
        and owner_valid_exact
        and owner_retention >= float(minimum_owner_retention)
        and evidence_count > 0
        and evidence_retention >= float(minimum_evidence_retention)
        and finite
        and in_bounds
        and maximum_displacement <= float(maximum_displacement_px) + 1e-9
        and candidate_minimum_jacobian >= float(minimum_jacobian) - 1e-9
        and halo_regression <= float(maximum_halo_regression_px) + 1e-9
        and boundary_curvature <= float(maximum_halo_regression_px) + 1e-9
        and exterior_exact
    )
    return MappingProxyType({
        "passed": passed,
        "formal_owner_pixel_count": formal_count,
        "formal_owner_retained_pixel_count": retained_formal,
        "formal_owner_support_retention": owner_retention,
        "formal_owner_valid_exact": owner_valid_exact,
        "evidence_sample_count": evidence_count,
        "evidence_retained_sample_count": retained_evidence,
        "evidence_sample_retention": evidence_retention,
        "affected_maps_finite": finite,
        "affected_maps_in_bounds": in_bounds,
        "maximum_combined_map_displacement_px": maximum_displacement,
        "base_minimum_inverse_map_jacobian": base_minimum_jacobian,
        "candidate_minimum_inverse_map_jacobian": candidate_minimum_jacobian,
        "minimum_inverse_map_jacobian_ratio": jacobian_ratio,
        "halo_regression_px": halo_regression,
        "segment_boundary_step_px": halo_regression,
        "segment_boundary_curvature_spike_px": boundary_curvature,
        "exterior_map_exact": exterior_exact,
    })


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
    signed_gradient_agreement: float = 1.0
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
            self.signed_gradient_agreement,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("C2E observation contains nonfinite metrics")
        if not math.isclose(normal_norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("C2E observation normal must be unit length")
        # OpenCV/LAPACK may differ by a few last bits across otherwise exact
        # repeated runs.  Freeze the physical observation before it becomes
        # chain/match authority; five-decimal metric quantization is then
        # serialized by the existing canonical-six-decimal JSON contract.
        quantized_normal = np.asarray(
            (round(float(self.normal_x), 5), round(float(self.normal_y), 5)),
            dtype=np.float64,
        )
        quantized_norm = float(np.linalg.norm(quantized_normal))
        if quantized_norm <= 0.0:
            raise ValueError("C2E quantized observation normal is invalid")
        quantized_normal /= quantized_norm
        object.__setattr__(self, "normal_x", float(quantized_normal[0]))
        object.__setattr__(self, "normal_y", float(quantized_normal[1]))
        for name in (
            "fitted_line_offset", "forward_best_lag_px", "reverse_best_lag_px",
            "correlation", "uniqueness_fraction", "orientation_difference_degrees",
            "signed_gradient_polarity", "signed_gradient_agreement",
        ):
            object.__setattr__(self, name, round(float(getattr(self, name)), 5))
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
            values = np.where(np.isfinite(values), np.round(values, 5), values)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchors = np.asarray(anchor_xy, dtype=np.float64)
    rx, rvalid = _bilinear(reference_magnitude, anchors[:, 0], anchors[:, 1])
    rgx, rgxvalid = _bilinear(reference_gx, anchors[:, 0], anchors[:, 1])
    rgy, rgyvalid = _bilinear(reference_gy, anchors[:, 0], anchors[:, 1])
    base_valid = rvalid & rgxvalid & rgyvalid & (rx > 0.0)
    scores = np.full(lags.shape, -math.inf, dtype=np.float64)
    correlations = np.zeros(lags.shape, dtype=np.float64)
    orientation_agreements = np.zeros(lags.shape, dtype=np.float64)
    signed_agreements = np.zeros(lags.shape, dtype=np.float64)
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
            orientation_agreement = 0.0
            signed_agreement = 0.0
        else:
            dot = rgx[keep][usable] * mgx[keep][usable] + rgy[keep][usable] * mgy[keep][usable]
            signed_cosine = dot / orientation_denominator[usable]
            # Physical edge orientation is modulo pi: contrast reversal must
            # not masquerade as a 180-degree geometric disagreement.  Preserve
            # the signed cosine separately for the contrast-consistency gate.
            modulo_pi_difference = np.arccos(np.clip(np.abs(signed_cosine), 0.0, 1.0))
            orientation_agreement = math.cos(float(np.average(
                modulo_pi_difference, weights=left[usable]
            )))
            signed_agreement = float(np.average(signed_cosine, weights=left[usable]))
        correlations[index] = correlation
        orientation_agreements[index] = orientation_agreement
        signed_agreements[index] = signed_agreement
        # Cosine correlation alone cannot distinguish an exact response from a
        # uniformly attenuated half-pixel interpolation.  Keep correlation as
        # its own reported gate, and include a bounded magnitude-consistency
        # term only in the hypothesis ranking score.
        magnitude_scale = max(float(np.mean(np.maximum(left, right))), 1e-12)
        magnitude_consistency = float(np.clip(
            1.0 - np.mean(np.abs(left - right)) / magnitude_scale, 0.0, 1.0
        ))
        scores[index] = (
            correlation
            * max(0.0, orientation_agreement)
            * max(0.0, signed_agreement)
            * magnitude_consistency
        )
    return scores, correlations, orientation_agreements, signed_agreements, counts


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
    precomputed_forward_signed_agreements: np.ndarray | None = None,
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
            np.asarray(
                precomputed_forward_signed_agreements
                if precomputed_forward_signed_agreements is not None
                else precomputed_forward_agreements,
                dtype=np.float64,
            ),
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
    orientation_agreement = min(
        float(forward[2][forward_index]), float(reverse[2][reverse_index])
    )
    signed_agreement = min(
        float(forward[3][forward_index]), float(reverse[3][reverse_index])
    )
    orientation_difference = math.degrees(math.acos(float(np.clip(
        orientation_agreement, -1.0, 1.0
    ))))
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
    if min(int(forward[4][forward_index]), int(reverse[4][reverse_index])) < 2:
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
        signed_gradient_agreement=signed_agreement,
        forward_valid_samples=int(forward[4][forward_index]),
        reverse_valid_samples=int(reverse[4][reverse_index]),
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
        forward_support_counts=forward[4],
        reverse_support_counts=reverse[4],
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
    *,
    support_by_authority: Mapping[tuple[int, int, str], np.ndarray] | None = None,
    severe_threshold_px: float = 1.0,
    minimum_evaluable_columns: int = 1,
) -> tuple[S13BaselineC2EObligation, ...]:
    if not math.isfinite(severe_threshold_px) or severe_threshold_px < 0.0:
        raise ValueError("C2E obligation severe threshold is invalid")
    if minimum_evaluable_columns < 1:
        raise ValueError("C2E obligation minimum evaluable columns is invalid")
    obligations: list[S13BaselineC2EObligation] = []
    for observation in sorted(observations, key=lambda row: (row.pair_index, row.component_id)):
        authority = (
            observation.pair_index, observation.component_id, observation.mask_sha256
        )
        support = None if support_by_authority is None else support_by_authority.get(authority)
        trace_columns = 0
        trace_missing_columns = 0
        trace_double_edge_columns = 0
        trace_slope_jump_p95 = 0.0
        trace_curvature_p95 = 0.0
        if support is not None:
            xy = np.asarray(support, dtype=np.int32)
            if xy.ndim != 2 or xy.shape[1] != 2:
                raise ValueError("C2E obligation support authority is invalid")
            if canonical_s13_support_sha256(xy) != observation.mask_sha256:
                raise ValueError("C2E obligation support authority SHA disagrees")
            columns = np.unique(xy[:, 0])
            trace_columns = int(columns.size)
            trace_missing_columns = int(
                max(0, int(columns[-1] - columns[0] + 1) - columns.size)
            ) if columns.size else 0
            y_trace = np.asarray([
                np.median(xy[xy[:, 0] == column, 1]) for column in columns
            ], dtype=np.float64)
            trace_double_edge_columns = int(sum(
                np.ptp(xy[xy[:, 0] == column, 1]) > 2.0 for column in columns
            ))
            if y_trace.size >= 3:
                slope_jump = np.abs(np.diff(y_trace, n=2))
                trace_slope_jump_p95 = float(np.percentile(slope_jump, 95.0))
                trace_curvature_p95 = trace_slope_jump_p95
        symmetric_lag = 0.5 * (
            observation.forward_best_lag_px - observation.reverse_best_lag_px
        )
        discrepancy = abs(
            observation.forward_best_lag_px + observation.reverse_best_lag_px
        )
        evaluable = bool(
            observation.evidence_state != "unevaluable"
            and observation.forward_valid_samples > 0
            and observation.reverse_valid_samples > 0
            and (support is None or trace_columns >= minimum_evaluable_columns)
        )
        metrics = {
            "forward_best_lag_px": observation.forward_best_lag_px,
            "reverse_best_lag_px": observation.reverse_best_lag_px,
            "symmetric_normal_lag_px": symmetric_lag,
            "absolute_symmetric_normal_lag_px": abs(symmetric_lag),
            "forward_reverse_discrepancy_px": discrepancy,
            "correlation": observation.correlation,
            "uniqueness_fraction": observation.uniqueness_fraction,
            "evidence_state": observation.evidence_state,
            "trace_unique_column_count": trace_columns,
            "trace_missing_column_count": trace_missing_columns,
            "trace_double_edge_column_count": trace_double_edge_columns,
            "trace_slope_jump_p95_px": trace_slope_jump_p95,
            "trace_second_difference_p95_px": trace_curvature_p95,
            "severe_threshold_px": float(severe_threshold_px),
            "severe_reason": (
                "absolute_symmetric_normal_lag_exceeded"
                if evaluable and abs(symmetric_lag) > severe_threshold_px else None
            ),
            "evaluable_reason": (
                None if evaluable else "insufficient_actionable_symmetric_trace"
            ),
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
            severe=evaluable and abs(symmetric_lag) > severe_threshold_px,
            evaluable=evaluable,
        ))
    return tuple(obligations)


def audit_s13_obligation_coverage(
    obligations: Sequence[Mapping[str, object]],
    segments: Sequence[Mapping[str, object]],
    accepted_segment_ids: Sequence[str],
) -> Mapping[str, object]:
    """Independently recompute frozen obligation and split/application coverage."""

    accepted = {str(value) for value in accepted_segment_ids}
    obligation_keys: dict[tuple[int, int, str], Mapping[str, object]] = {}
    for row in obligations:
        key = (
            int(row.get("pair_index", -1)),
            int(row.get("component_id", -1)),
            str(row.get("support_sha256", "")),
        )
        if key[0] < 0 or key[1] < 0 or len(key[2]) != 64 or key in obligation_keys:
            raise ValueError("C2E frozen obligation authority is invalid or duplicated")
        if row.get("scope") != "runtime_detected" or not isinstance(
            row.get("baseline_metrics"), Mapping
        ):
            raise ValueError("C2E frozen obligation metrics/scope authority is missing")
        obligation_keys[key] = row

    segment_ids: set[str] = set()
    resolved_keys: set[tuple[int, int, str]] = set()
    covered_keys: set[tuple[int, int, str]] = set()
    child_parent: dict[str, str] = {}
    for segment in segments:
        segment_id = str(segment.get("segment_id", ""))
        if not segment_id or segment_id in segment_ids:
            raise ValueError("C2E segment authority is invalid or duplicated")
        segment_ids.add(segment_id)
        pair_indices = tuple(int(value) for value in segment.get("pair_indices", ()))
        source_indices = tuple(int(value) for value in segment.get("source_indices", ()))
        expected_sources = tuple(sorted({value for pair in pair_indices for value in (pair, pair + 1)}))
        if pair_indices != tuple(sorted(set(pair_indices))) or source_indices != expected_sources:
            raise ValueError("C2E segment pair/source authority is inconsistent")
        parent = segment.get("parent_segment_id")
        if parent is not None:
            child_parent[segment_id] = str(parent)
        for child in segment.get("child_segment_ids", ()):
            child_parent[str(child)] = segment_id
        for authority in segment.get("observation_authority", ()):
            if not isinstance(authority, Mapping):
                raise ValueError("C2E segment observation authority is invalid")
            key = (
                int(authority.get("pair_index", -1)),
                int(authority.get("component_id", -1)),
                str(authority.get("support_sha256", "")),
            )
            if key not in obligation_keys or key[0] not in pair_indices:
                raise ValueError("C2E segment references an unknown frozen obligation")
            covered_keys.add(key)
            if segment_id in accepted and segment.get("state") == "resolved":
                resolved_keys.add(key)
    if not accepted.issubset(segment_ids):
        raise ValueError("C2E accepted segment authority is unknown")
    for child, parent in child_parent.items():
        if child not in segment_ids or parent not in segment_ids or child == parent:
            raise ValueError("C2E split parent/child authority is invalid")
        seen = {child}
        cursor = parent
        while cursor in child_parent:
            if cursor in seen:
                raise ValueError("C2E split lineage contains a cycle")
            seen.add(cursor)
            cursor = child_parent[cursor]

    severe = {
        key for key, row in obligation_keys.items()
        if row.get("severe") is True and row.get("evaluable") is True
    }
    repair_complete = bool(severe) and severe.issubset(resolved_keys)
    return MappingProxyType({
        "passed": True,
        "obligation_count": len(obligation_keys),
        "covered_obligation_count": len(covered_keys),
        "uncovered_obligation_count": len(set(obligation_keys) - covered_keys),
        "severe_evaluable_obligation_count": len(severe),
        "resolved_severe_obligation_count": len(severe & resolved_keys),
        "repair_complete": repair_complete,
    })


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


def _compatibility_audit(
    left: S13EdgeComponentObservation,
    right: S13EdgeComponentObservation,
    config: object,
) -> dict[str, object]:
    """Return every frozen identity gate and weighted matching subscore."""

    ly0, ly1 = left.global_bbox_xyxy[1], left.global_bbox_xyxy[3]
    ry0, ry1 = right.global_bbox_xyxy[1], right.global_bbox_xyxy[3]
    overlap = max(0, min(ly1, ry1) - max(ly0, ry0)) / max(1, min(ly1 - ly0, ry1 - ry0))
    angle = _angle_difference((left.normal_x, left.normal_y), (right.normal_x, right.normal_y))
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
    seam_midpoint = 0.25 * (
        left.global_bbox_xyxy[0] + left.global_bbox_xyxy[2]
        + right.global_bbox_xyxy[0] + right.global_bbox_xyxy[2]
    )
    predicted_y = abs(_line_y(left, seam_midpoint) - _line_y(right, seam_midpoint))
    polarity_product = left.signed_gradient_polarity * right.signed_gradient_polarity
    signed_agreement = min(abs(left.signed_gradient_polarity), abs(right.signed_gradient_polarity))
    gates = {
        "adjacent_pair": right.pair_index - left.pair_index == 1,
        "left_evaluable": left.evidence_state not in {"ambiguous", "unevaluable"},
        "right_evaluable": right.evidence_state not in {"ambiguous", "unevaluable"},
        "minimum_y_overlap": overlap >= float(config.minimum_component_y_overlap_fraction),
        "maximum_normal_difference": angle <= float(config.maximum_component_normal_difference_degrees),
        "maximum_endpoint_distance": endpoint_distance <= float(config.maximum_component_endpoint_distance_px),
        "maximum_predicted_y_disagreement": predicted_y <= float(config.maximum_predicted_y_disagreement_px),
        "signed_gradient_polarity": polarity_product > 0.0,
        "minimum_signed_gradient_agreement": (
            signed_agreement >= float(config.minimum_signed_gradient_agreement)
        ),
    }
    weights = config.component_match_weights
    correlation = min(left.correlation, right.correlation)
    uniqueness = min(left.uniqueness_fraction, right.uniqueness_fraction)
    parts = {
        "y_overlap": float(np.clip(overlap, 0.0, 1.0)),
        "predicted_y": float(np.clip(1.0 - predicted_y / config.maximum_predicted_y_disagreement_px, 0.0, 1.0)),
        "endpoint_distance": float(np.clip(1.0 - endpoint_distance / config.maximum_component_endpoint_distance_px, 0.0, 1.0)),
        "correlation": float(np.clip(correlation, 0.0, 1.0)),
        "uniqueness": float(np.clip(uniqueness, 0.0, 1.0)),
    }
    eligible = all(gates.values())
    score = float(sum(float(weights[name]) * parts[name] for name in parts))
    return {
        "left_pair_index": int(left.pair_index),
        "left_component_id": int(left.component_id),
        "right_pair_index": int(right.pair_index),
        "right_component_id": int(right.component_id),
        "measurements": {
            "y_overlap_fraction": float(overlap),
            "normal_difference_degrees": float(angle),
            "endpoint_distance_px": float(endpoint_distance),
            "predicted_y_disagreement_px": float(predicted_y),
            "signed_gradient_product": float(polarity_product),
            "signed_gradient_agreement": float(signed_agreement),
            "minimum_correlation": float(correlation),
            "minimum_uniqueness_fraction": float(uniqueness),
        },
        "hard_gates": gates,
        "subscores": parts,
        "weighted_score": score,
        "eligible": bool(eligible),
        "reject_reasons": tuple(name for name, passed in gates.items() if not passed),
    }


def _compatibility_score(
    left: S13EdgeComponentObservation,
    right: S13EdgeComponentObservation,
    config: object,
) -> float | None:
    audit = _compatibility_audit(left, right, config)
    return float(audit["weighted_score"]) if audit["eligible"] is True else None


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
    match_matrix_sink: list[Mapping[str, object]] | None = None,
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
        matrix_rows: dict[tuple[int, int], dict[str, object]] = {}
        for li, global_left in enumerate(left_indices):
            for ri, global_right in enumerate(right_indices):
                row = _compatibility_audit(
                    observations[global_left], observations[global_right], config
                )
                matrix_rows[li, ri] = row
                if row["eligible"] is True:
                    scores[li, ri] = float(row["weighted_score"])
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
        unambiguous_scores = {
            edge: value for edge, value in scores.items()
            if edge[0] not in ambiguous_left and edge[1] not in ambiguous_right
        }
        winners = set(_maximum_weight_matching(
            [observations[index] for index in left_indices],
            [observations[index] for index in right_indices],
            unambiguous_scores,
        ))
        for li, ri in winners:
            links.add((left_indices[li], right_indices[ri]))
        if match_matrix_sink is not None:
            serial_rows: list[dict[str, object]] = []
            for edge in sorted(matrix_rows):
                li, ri = edge
                row = dict(matrix_rows[edge])
                left_values = sorted(
                    (value for (candidate_left, _), value in scores.items()
                     if candidate_left == li), reverse=True
                )
                right_values = sorted(
                    (value for (_, candidate_right), value in scores.items()
                     if candidate_right == ri), reverse=True
                )
                left_margin = (
                    (left_values[0] - left_values[1]) / max(abs(left_values[0]), 1e-12)
                    if len(left_values) > 1 else None
                )
                right_margin = (
                    (right_values[0] - right_values[1]) / max(abs(right_values[0]), 1e-12)
                    if len(right_values) > 1 else None
                )
                junction_ambiguous = li in ambiguous_left or ri in ambiguous_right
                reject_reasons = list(row["reject_reasons"])
                if row["eligible"] is True and junction_ambiguous:
                    reject_reasons.append("best_vs_second_margin_below_minimum")
                elif row["eligible"] is True and edge not in winners:
                    reject_reasons.append("not_selected_by_maximum_weight_matching")
                row.update({
                    "left_margin_fraction": (
                        None if left_margin is None else float(left_margin)
                    ),
                    "right_margin_fraction": (
                        None if right_margin is None else float(right_margin)
                    ),
                    "junction_ambiguous": bool(junction_ambiguous),
                    "winner": edge in winners,
                    "reject_reasons": reject_reasons,
                })
                serial_rows.append(row)
            match_matrix_sink.append({
                "left_pair_index": int(pair_index),
                "right_pair_index": int(pair_index + 1),
                "minimum_margin_fraction": margin,
                "candidate_count": len(serial_rows),
                "eligible_candidate_count": len(scores),
                "winner_count": len(winners),
                "candidates": serial_rows,
            })
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


def propagate_s13_edge_component_evidence(
    seed_evidence_by_pair: Mapping[int, Sequence[tuple[object, object]]],
    *,
    available_pair_indices: Sequence[int],
    pair_evidence_loader: Callable[[int], Sequence[tuple[object, object]]],
    config: object,
) -> tuple[tuple[tuple[object, object], ...], Mapping[str, object]]:
    """Lazily follow exact physical components in both pair directions.

    Only a pair adjacent to an already matched component is loaded.  A loaded
    pair becomes a new frontier solely through a winner of the same frozen
    one-to-one match matrix used by the final chain builder.  Thus safe anchors
    may extend a seed across any number of pairs without paying reverse-search
    cost eagerly for the full sequence.
    """

    available = {int(value) for value in available_pair_indices}
    evidence_by_pair: dict[int, tuple[tuple[object, object], ...]] = {
        int(pair): tuple(rows)
        for pair, rows in sorted(seed_evidence_by_pair.items())
    }
    seed_pairs = tuple(sorted(evidence_by_pair))
    active: set[tuple[int, int]] = set()
    queue: list[tuple[int, int]] = []
    for pair_index, rows in evidence_by_pair.items():
        for observation, _evidence in rows:
            if observation.evidence_state in {"ambiguous", "unevaluable"}:
                continue
            key = (int(pair_index), int(observation.component_id))
            active.add(key)
            queue.append(key)
    queue.sort()
    loader_calls: list[int] = []
    match_matrices: dict[tuple[int, int], Mapping[str, object]] = {}
    propagation_links: set[tuple[int, int, int, int]] = set()

    cursor = 0
    while cursor < len(queue):
        pair_index, component_id = queue[cursor]
        cursor += 1
        for neighbour in (pair_index - 1, pair_index + 1):
            if neighbour not in available:
                continue
            if neighbour not in evidence_by_pair:
                evidence_by_pair[neighbour] = tuple(pair_evidence_loader(neighbour))
                loader_calls.append(neighbour)
            left_pair, right_pair = sorted((pair_index, neighbour))
            junction = (left_pair, right_pair)
            if junction not in match_matrices:
                matrix_sink: list[Mapping[str, object]] = []
                build_s13_edge_component_chains(
                    tuple(
                        row[0]
                        for candidate_pair in junction
                        for row in evidence_by_pair.get(candidate_pair, ())
                    ),
                    config=config,
                    match_matrix_sink=matrix_sink,
                )
                match_matrices[junction] = (
                    matrix_sink[0]
                    if matrix_sink else {
                        "left_pair_index": left_pair,
                        "right_pair_index": right_pair,
                        "minimum_margin_fraction": float(
                            config.minimum_component_match_margin_fraction
                        ),
                        "candidate_count": 0,
                        "eligible_candidate_count": 0,
                        "winner_count": 0,
                        "candidates": [],
                    }
                )
            matrix = match_matrices[junction]
            for row in matrix["candidates"]:
                if row["winner"] is not True:
                    continue
                left_key = (
                    int(row["left_pair_index"]), int(row["left_component_id"])
                )
                right_key = (
                    int(row["right_pair_index"]), int(row["right_component_id"])
                )
                if (pair_index, component_id) == left_key:
                    destination = right_key
                elif (pair_index, component_id) == right_key:
                    destination = left_key
                else:
                    continue
                propagation_links.add((*left_key, *right_key))
                if destination not in active:
                    active.add(destination)
                    queue.append(destination)

    flattened = tuple(
        row
        for pair_index in sorted(evidence_by_pair)
        for row in sorted(
            evidence_by_pair[pair_index],
            key=lambda item: (item[0].pair_index, item[0].component_id),
        )
    )
    audit: Mapping[str, object] = {
        "policy": "visual_suspect_seed_bidirectional_lazy_safe_anchor_propagation",
        "seed_pair_indices": list(seed_pairs),
        "probed_pair_indices": sorted(set(loader_calls)),
        "reverse_evidence_loader_call_count": len(loader_calls),
        "exact_pair_count": len(evidence_by_pair),
        "exact_observation_count": len(flattened),
        "propagation_link_count": len(propagation_links),
        "propagation_links": [list(row) for row in sorted(propagation_links)],
        "match_matrices": [
            match_matrices[key] for key in sorted(match_matrices)
        ],
    }
    return flattened, _frozen_mapping(audit)


def s13_component_segment_budget_priority(
    segment: object,
    *,
    evidence_by_component: Mapping[tuple[int, int], object],
    maximum_post_edge_p95_px: float,
) -> tuple[float, float, int, str]:
    """Frozen pre-evaluation order: excess, confidence, coverage, stable ID."""

    observations = tuple(segment.observations)
    excess = max(
        max(
            0.0,
            abs(0.5 * (
                float(row.forward_best_lag_px)
                - float(row.reverse_best_lag_px)
            )) - float(maximum_post_edge_p95_px),
        )
        for row in observations
    )
    confidence = min(
        float(row.correlation) * float(row.uniqueness_fraction)
        for row in observations
    )
    expected_coverage = len({
        int(x)
        for row in observations
        for x in np.asarray(
            evidence_by_component[(row.pair_index, row.component_id)].support_xy
        )[:, 0]
    })
    return (-excess, -confidence, -expected_coverage, str(segment.segment_id))


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


def split_s13_component_application_segment_at_pair(
    segment: S13ComponentApplicationSegment,
    *,
    split_pair_index: int,
    reason: str,
    minimum_pair_count: int = 1,
) -> tuple[S13ComponentApplicationSegment, ...]:
    """Remove one failed pair edge and return deterministic viable child segments.

    A split is deliberately not a partition that assigns the failed observation
    to either child: the selected pair is the local no-op boundary.  Remaining
    observations on each side retain the outer cut authority of their parent and
    bind the newly introduced cut on the side adjacent to the removed edge.
    """

    if not isinstance(split_pair_index, int) or split_pair_index < 0:
        raise ValueError("C2E split pair index must be nonnegative")
    if not isinstance(reason, str) or not reason:
        raise ValueError("C2E split reason must be nonempty")
    if not isinstance(minimum_pair_count, int) or minimum_pair_count < 1:
        raise ValueError("C2E minimum child pair count must be positive")
    if split_pair_index not in segment.pair_indices:
        raise ValueError("C2E split pair is outside its segment")

    left = tuple(
        row for row in segment.observations if row.pair_index < split_pair_index
    )
    right = tuple(
        row for row in segment.observations if row.pair_index > split_pair_index
    )
    children: list[S13ComponentApplicationSegment] = []
    if len({row.pair_index for row in left}) >= minimum_pair_count:
        children.append(S13ComponentApplicationSegment.create(
            parent_chain_id=segment.parent_chain_id,
            observations=left,
            left_cut_reason=segment.left_cut_reason,
            right_cut_reason=reason,
        ))
    if len({row.pair_index for row in right}) >= minimum_pair_count:
        children.append(S13ComponentApplicationSegment.create(
            parent_chain_id=segment.parent_chain_id,
            observations=right,
            left_cut_reason=reason,
            right_cut_reason=segment.right_cut_reason,
        ))
    return tuple(children)


def plan_s13_failed_segment_split(
    segment: S13ComponentApplicationSegment,
    *,
    pair_audits: Sequence[Mapping[str, object]],
    reason: str,
    split_depth: int,
    failure_kind: Literal["solver", "quality"],
    config: object,
) -> tuple[int | None, tuple[S13ComponentApplicationSegment, ...]]:
    """Apply the frozen recursive-split policy to one failed segment."""

    if failure_kind not in {"solver", "quality"}:
        raise ValueError("C2E segment split failure kind is invalid")
    if not isinstance(split_depth, int) or split_depth < 0:
        raise ValueError("C2E segment split depth is invalid")
    if not getattr(config, "allow_partial_application", False):
        return None, ()
    if failure_kind == "solver" and not getattr(config, "split_on_solver_outlier", False):
        return None, ()
    maximum_depth = int(getattr(config, "maximum_segment_split_depth"))
    minimum_pairs = int(getattr(config, "minimum_application_segment_pair_count"))
    if split_depth >= maximum_depth or len(set(segment.pair_indices)) <= minimum_pairs:
        return None, ()
    if not pair_audits:
        return None, ()
    split_pair = select_s13_component_segment_split_pair(pair_audits)
    children = split_s13_component_application_segment_at_pair(
        segment,
        split_pair_index=split_pair,
        reason=reason,
        minimum_pair_count=minimum_pairs,
    )
    return (split_pair, children) if children else (None, ())


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
        limit = float(config.maximum_normalized_solver_residual)
        raise S13ComponentSolverError(
            "C2E solver normalized residual exceeded",
            pair_audits=tuple(
                {
                    "pair_index": observation.pair_index,
                    "hard_violation_count": int(value > limit),
                    "maximum_normalized_gate_excess": max(0.0, value - limit)
                    / max(abs(limit), 1e-6),
                    "normalized_solver_residual": float(value),
                    "baseline_edge_excess_px": max(
                        0.0,
                        abs(_observation_lag(observation))
                        - float(config.maximum_post_edge_p95_px),
                    ),
                }
                for observation, value in zip(
                    sorted(segment.observations, key=lambda row: row.pair_index),
                    normalized,
                    strict=True,
                )
            ),
        )
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


@dataclass(frozen=True)
class S13RuntimeEdgeTrace:
    """Deterministic owner-only, exact-seeded per-column physical edge trace."""

    y_edge_px: np.ndarray
    evaluable: np.ndarray
    double_edge: np.ndarray
    seam_side: np.ndarray
    metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        arrays = (
            _readonly(self.y_edge_px, np.float64),
            _readonly(self.evaluable, np.bool_),
            _readonly(self.double_edge, np.bool_),
            _readonly(self.seam_side, np.int8),
        )
        if any(row.ndim != 1 for row in arrays) or any(
            row.shape != arrays[0].shape for row in arrays[1:]
        ):
            raise ValueError("C2E runtime edge trace arrays disagree")
        object.__setattr__(self, "y_edge_px", arrays[0])
        object.__setattr__(self, "evaluable", arrays[1])
        object.__setattr__(self, "double_edge", arrays[2])
        object.__setattr__(self, "seam_side", arrays[3])
        object.__setattr__(self, "metrics", _frozen_mapping(self.metrics))


def trace_s13_owner_only_component_edge(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    support_xy: np.ndarray,
    normal_xy: tuple[float, float],
    signed_gradient_polarity: float,
    seam_x_by_row: np.ndarray,
    search_radius_px: int = 6,
) -> S13RuntimeEdgeTrace:
    """Trace one seeded edge in a narrow band without selecting another ROI edge."""

    pixels = np.asarray(image)
    valid = np.asarray(valid_mask, dtype=bool)
    support = np.asarray(support_xy, dtype=np.float64)
    seam = np.asarray(seam_x_by_row, dtype=np.int32)
    if pixels.ndim != 3 or pixels.shape[:2] != valid.shape:
        raise ValueError("C2E runtime trace image/valid shapes disagree")
    if support.ndim != 2 or support.shape[1] != 2 or support.shape[0] < 2:
        raise ValueError("C2E runtime trace exact support is invalid")
    if seam.shape != (pixels.shape[0],) or search_radius_px < 1:
        raise ValueError("C2E runtime trace seam/search inputs are invalid")
    normal = np.asarray(normal_xy, dtype=np.float64)
    if normal.shape != (2,) or not np.isfinite(normal).all():
        raise ValueError("C2E runtime trace normal is invalid")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9 or abs(float(normal[1])) <= 1e-6:
        raise ValueError("C2E runtime y-edge trace requires a nonvertical line")
    normal /= normal_norm
    polarity = float(np.sign(signed_gradient_polarity))
    if not math.isfinite(float(signed_gradient_polarity)) or polarity == 0.0:
        raise ValueError("C2E runtime trace polarity is invalid")
    height, width = valid.shape
    line_offset = float(np.median(support @ normal))
    columns = np.arange(width, dtype=np.float64)
    expected_y = (line_offset - normal[0] * columns) / normal[1]
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    directed = polarity * (gx * float(normal[0]) + gy * float(normal[1]))
    candidates: list[np.ndarray] = []
    eligible = np.zeros(width, dtype=bool)
    positive_samples: list[np.ndarray] = []
    for column, expected in enumerate(expected_y):
        lower = max(1, int(math.floor(expected)) - search_radius_px)
        upper = min(height - 1, int(math.ceil(expected)) + search_radius_px + 1)
        rows = np.arange(lower, upper, dtype=np.int32)
        rows = rows[valid[rows, column]] if rows.size else rows
        candidates.append(rows)
        if rows.size:
            eligible[column] = True
            values = directed[rows, column]
            positive_samples.append(values[values > 0.0])
    finite_strength = (
        np.concatenate([row for row in positive_samples if row.size])
        if any(row.size for row in positive_samples) else np.empty(0, np.float32)
    )
    threshold = (
        max(12.0, float(np.percentile(finite_strength, 65.0)))
        if finite_strength.size else math.inf
    )
    scale = max(threshold, 1.0)
    costs: list[np.ndarray] = []
    predecessors: list[np.ndarray] = []
    for column, rows in enumerate(candidates):
        if not rows.size:
            costs.append(np.empty(0, np.float64))
            predecessors.append(np.empty(0, np.int32))
            continue
        emission = (
            -directed[rows, column].astype(np.float64) / scale
            + 0.06 * np.abs(rows.astype(np.float64) - expected_y[column])
        )
        if column == 0 or not costs[column - 1].size:
            costs.append(emission)
            predecessors.append(np.full(rows.size, -1, np.int32))
            continue
        previous_rows = candidates[column - 1]
        expected_step = expected_y[column] - expected_y[column - 1]
        transition = 0.35 * np.abs(
            rows[:, None].astype(np.float64)
            - previous_rows[None, :].astype(np.float64) - expected_step
        )
        combined = transition + costs[column - 1][None, :]
        previous = np.argmin(combined, axis=1).astype(np.int32)
        costs.append(emission + combined[np.arange(rows.size), previous])
        predecessors.append(previous)
    traced = np.full(width, np.nan, dtype=np.float64)
    segment_ends = [
        column for column in range(width) if candidates[column].size
        and (column + 1 == width or not candidates[column + 1].size)
    ]
    for segment_end in segment_ends:
        state = int(np.argmin(costs[segment_end]))
        column = segment_end
        while column >= 0 and candidates[column].size:
            traced[column] = float(candidates[column][state])
            if column == 0 or not candidates[column - 1].size:
                break
            state = int(predecessors[column][state])
            column -= 1
    selected_columns = np.flatnonzero(np.isfinite(traced))
    if selected_columns.size:
        selected_rows = np.rint(traced[selected_columns]).astype(np.int32)
        orientation_ratio = directed[selected_rows, selected_columns] / np.maximum(
            magnitude[selected_rows, selected_columns], 1e-6
        )
        local_threshold = np.asarray([
            max(12.0, 0.50 * float(np.max(directed[candidates[column], column])))
            for column in selected_columns
        ], dtype=np.float64)
        supported = (
            (directed[selected_rows, selected_columns] >= local_threshold)
            & (orientation_ratio >= math.cos(math.radians(35.0)))
        )
        traced[selected_columns[~supported]] = np.nan
    double_edge = np.zeros(width, dtype=bool)
    for column in np.flatnonzero(np.isfinite(traced)):
        rows = candidates[column]
        values = directed[rows, column]
        primary = int(round(float(traced[column])))
        primary_strength = float(directed[primary, column])
        peaks = np.flatnonzero(
            (values >= np.roll(values, 1)) & (values >= np.roll(values, -1))
            & (values >= 0.50 * primary_strength)
        )
        peaks = peaks[(peaks > 0) & (peaks + 1 < values.size)]
        double_edge[column] = any(
            abs(int(rows[index]) - primary) >= 3
            and float(values[index]) >= 0.50 * primary_strength
            and float(values[index]) / max(
                float(magnitude[int(rows[index]), column]), 1e-6
            ) >= math.cos(math.radians(35.0))
            for index in peaks
        )
    evaluable = np.isfinite(traced)
    adjacency = evaluable[:-1] & evaluable[1:]
    slopes = np.diff(traced)[adjacency]
    expected_slope = float(-normal[0] / normal[1])
    slope_error = np.abs(slopes - expected_slope)
    triples = evaluable[:-2] & evaluable[1:-1] & evaluable[2:]
    second = np.abs(np.diff(traced, n=2)[triples])
    missing = eligible & ~evaluable
    longest_break = 0
    current_break = 0
    for value in missing:
        current_break = current_break + 1 if value else 0
        longest_break = max(longest_break, current_break)
    seam_side = np.zeros(width, dtype=np.int8)
    for column in np.flatnonzero(evaluable):
        row = int(np.clip(round(float(traced[column])), 0, height - 1))
        seam_side[column] = np.int8(np.sign(column - int(seam[row])))
    compressed_sides = seam_side[evaluable]
    compressed_sides = compressed_sides[compressed_sides != 0]
    seam_crossing_count = int(np.count_nonzero(
        compressed_sides[:-1] != compressed_sides[1:]
    ))
    seam_crossings = np.flatnonzero(
        evaluable & (seam_side != 0)
        & (np.arange(width) == np.asarray([
            int(seam[int(np.clip(round(float(value)), 0, height - 1))])
            if math.isfinite(float(value)) else -1 for value in traced
        ]))
    )
    metrics = {
        "evaluable_column_count": int(np.count_nonzero(evaluable)),
        "eligible_column_count": int(np.count_nonzero(eligible)),
        "coverage_fraction": float(
            np.count_nonzero(evaluable) / max(np.count_nonzero(eligible), 1)
        ),
        "slope_p50_px": float(np.percentile(slopes, 50.0)) if slopes.size else None,
        "slope_p95_px": float(np.percentile(np.abs(slopes), 95.0)) if slopes.size else None,
        "slope_error_p95_px": (
            float(np.percentile(slope_error, 95.0)) if slope_error.size else None
        ),
        "maximum_slope_error_px": float(np.max(slope_error)) if slope_error.size else None,
        "second_difference_p95_px": (
            float(np.percentile(second, 95.0)) if second.size else None
        ),
        "maximum_second_difference_px": float(np.max(second)) if second.size else None,
        "missing_column_count": int(np.count_nonzero(missing)),
        "break_length_px": int(longest_break),
        "double_edge_length_px": int(np.count_nonzero(double_edge)),
        "seam_crossing_count": seam_crossing_count,
        "seam_crossing_columns": seam_crossings.astype(np.int32).tolist(),
        "search_band_boundary_contact": bool(any(
            evaluable[column] and candidates[column].size
            and int(round(float(traced[column]))) in {
                int(candidates[column][0]), int(candidates[column][-1])
            }
            for column in range(width)
        )),
        "strength_threshold": threshold if math.isfinite(threshold) else None,
        "expected_slope_px_per_column": expected_slope,
    }
    return S13RuntimeEdgeTrace(
        y_edge_px=traced, evaluable=evaluable, double_edge=double_edge,
        seam_side=seam_side, metrics=metrics,
    )


def aggregate_s13_runtime_edge_traces(
    traces: Sequence[S13RuntimeEdgeTrace],
    *,
    search_boundary_hit: bool = False,
) -> Mapping[str, object]:
    """Aggregate exact traces into the metric names consumed by quality gates."""

    if not traces:
        raise ValueError("C2E runtime trace aggregation requires at least one trace")
    slope_errors: list[np.ndarray] = []
    second_values: list[np.ndarray] = []
    for trace in traces:
        valid = trace.evaluable
        adjacent = valid[:-1] & valid[1:]
        slopes = np.diff(trace.y_edge_px)[adjacent]
        expected_slope = float(trace.metrics["expected_slope_px_per_column"])
        slope_errors.append(np.abs(slopes - expected_slope))
        triples = valid[:-2] & valid[1:-1] & valid[2:]
        second_values.append(np.abs(np.diff(trace.y_edge_px, n=2)[triples]))
    slopes = np.concatenate(slope_errors) if slope_errors else np.empty(0, np.float64)
    second = np.concatenate(second_values) if second_values else np.empty(0, np.float64)
    evaluable_count = sum(
        int(row.metrics["evaluable_column_count"]) for row in traces
    )
    eligible_count = sum(int(row.metrics["eligible_column_count"]) for row in traces)
    return MappingProxyType({
        "edge_p50_px": float(np.percentile(slopes, 50.0)) if slopes.size else 0.0,
        "edge_p95_px": float(np.percentile(slopes, 95.0)) if slopes.size else math.inf,
        "maximum_step_px": float(np.max(slopes)) if slopes.size else math.inf,
        "break_length_px": float(max(
            int(row.metrics["break_length_px"]) for row in traces
        )),
        "double_edge_length_px": float(sum(
            int(row.metrics["double_edge_length_px"]) for row in traces
        )),
        "non_target_p95_px": (
            float(np.percentile(second, 95.0)) if second.size else 0.0
        ),
        "second_difference_p95_px": (
            float(np.percentile(second, 95.0)) if second.size else 0.0
        ),
        "evaluable_edge_columns": evaluable_count,
        "eligible_edge_columns": eligible_count,
        "coverage_fraction": float(evaluable_count / max(eligible_count, 1)),
        "seam_crossing_count": int(sum(
            int(row.metrics["seam_crossing_count"]) for row in traces
        )),
        "search_band_boundary_contact": any(
            row.metrics["search_band_boundary_contact"] is True for row in traces
        ),
        "search_boundary_hit": bool(search_boundary_hit),
        "traces": [
            {
                "y_edge_px": [
                    float(value) if math.isfinite(float(value)) else None
                    for value in row.y_edge_px
                ],
                "evaluable": row.evaluable.astype(np.uint8).tolist(),
                "double_edge": row.double_edge.astype(np.uint8).tolist(),
                "seam_side": row.seam_side.tolist(),
                "metrics": dict(row.metrics),
            }
            for row in traces
        ],
    })


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
    baseline_search_boundary_hit = baseline.get("search_boundary_hit") is True
    candidate_search_boundary_hit = candidate.get("search_boundary_hit") is True
    search_boundary_hit = baseline_search_boundary_hit or candidate_search_boundary_hit
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
        "baseline_search_boundary_hit": baseline_search_boundary_hit,
        "candidate_search_boundary_hit": candidate_search_boundary_hit,
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
    normal_field_audit: Mapping[str, object] = field(default_factory=dict)

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
        object.__setattr__(self, "normal_field_audit", _frozen_mapping(self.normal_field_audit))


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _s13_local_normal_field(
    global_x: np.ndarray,
    *,
    canonical_normal_xy: tuple[float, float],
    observations: Sequence[S13EdgeComponentObservation],
    maximum_angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    canonical = np.asarray(canonical_normal_xy, np.float64)
    canonical /= float(np.linalg.norm(canonical))
    if not observations:
        return (
            np.full(global_x.shape, canonical[0], np.float64),
            np.full(global_x.shape, canonical[1], np.float64),
            MappingProxyType({
                "policy": "single_canonical_normal_exact", "node_count": 1,
                "maximum_node_angle_degrees": 0.0,
                "maximum_pixel_angle_degrees": 0.0,
                "canonical_hemisphere_enforced": True,
            }),
        )
    nodes: list[tuple[float, float, float, float, int, int]] = []
    maximum_node_angle = 0.0
    for observation in observations:
        normal = np.asarray((observation.normal_x, observation.normal_y), np.float64)
        normal /= float(np.linalg.norm(normal))
        if float(np.dot(normal, canonical)) < 0.0:
            normal *= -1.0
        angle = math.degrees(math.acos(float(np.clip(np.dot(normal, canonical), -1.0, 1.0))))
        maximum_node_angle = max(maximum_node_angle, angle)
        x0, _y0, x1, _y1 = observation.global_bbox_xyxy
        anchor_x = 0.5 * float(x0 + x1 - 1)
        nodes.append((
            anchor_x, _line_y(observation, anchor_x),
            float(normal[0]), float(normal[1]),
            observation.pair_index, observation.component_id,
        ))
    if maximum_node_angle > maximum_angle_degrees + 1e-9:
        raise ValueError("C2E local normal node exceeds canonical angle limit")
    nodes.sort(key=lambda row: (row[0], row[1], row[4], row[5]))
    collapsed: list[tuple[float, float, float, float]] = []
    for anchor_x in sorted({row[0] for row in nodes}):
        group = [row for row in nodes if row[0] == anchor_x]
        normal = np.mean(np.asarray([(row[2], row[3]) for row in group]), axis=0)
        normal /= float(np.linalg.norm(normal))
        collapsed.append((
            anchor_x, float(np.mean([row[1] for row in group])),
            float(normal[0]), float(normal[1]),
        ))
    anchors_x = np.asarray([row[0] for row in collapsed], np.float64)
    normals_x = np.asarray([row[2] for row in collapsed], np.float64)
    normals_y = np.asarray([row[3] for row in collapsed], np.float64)
    if len(collapsed) == 1:
        nx = np.full(global_x.shape, normals_x[0], np.float64)
        ny = np.full(global_x.shape, normals_y[0], np.float64)
        policy = "single_fitted_line_exact"
    else:
        anchors_y = np.asarray([row[1] for row in collapsed], np.float64)
        arc = np.concatenate((np.asarray([0.0]), np.cumsum(
            np.hypot(np.diff(anchors_x), np.diff(anchors_y))
        )))
        local_arc = np.interp(global_x, anchors_x, arc)
        nx = np.interp(local_arc, arc, normals_x)
        ny = np.interp(local_arc, arc, normals_y)
        length = np.hypot(nx, ny)
        nx, ny = nx / length, ny / length
        policy = "fitted_line_x_to_arc_length_linear_normal"
    pixel_angle = np.degrees(np.arccos(np.clip(
        nx * canonical[0] + ny * canonical[1], -1.0, 1.0
    )))
    maximum_pixel_angle = float(np.max(pixel_angle))
    if maximum_pixel_angle > maximum_angle_degrees + 1e-9:
        raise ValueError("C2E local normal field exceeds canonical angle limit")
    return nx, ny, MappingProxyType({
        "policy": policy, "node_count": len(collapsed),
        "maximum_node_angle_degrees": maximum_node_angle,
        "maximum_pixel_angle_degrees": maximum_pixel_angle,
        "maximum_allowed_angle_degrees": float(maximum_angle_degrees),
        "canonical_hemisphere_enforced": True,
    })


def build_s13_source_component_correction(
    *,
    segment_id: str,
    source_index: int,
    frame_id: int,
    support_mask: np.ndarray,
    source_offset_px: float,
    normal_xy: tuple[float, float],
    config: object,
    local_normal_observations: Sequence[S13EdgeComponentObservation] = (),
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
    # OpenCV's distance transform may differ by a few float32 ULPs across
    # worker scheduling. Canonicalize well below the 1e-4 pixel audit scale
    # before hashing so repeated clean runs produce byte-exact fields.
    weight = np.round(
        np.clip(weight, 0.0, 1.0).astype(np.float64), decimals=6
    ).astype(np.float32)
    local_normal_x, local_normal_y, normal_field_audit = _s13_local_normal_field(
        global_x,
        canonical_normal_xy=(float(normal[0]), float(normal[1])),
        observations=local_normal_observations,
        maximum_angle_degrees=float(config.maximum_component_normal_difference_degrees),
    )
    delta_u = np.round(
        weight.astype(np.float64) * (source_offset_px * local_normal_x), decimals=6
    ).astype(np.float32)
    delta_v = np.round(
        weight.astype(np.float64) * (source_offset_px * local_normal_y), decimals=6
    ).astype(np.float32)
    payload = {
        "segment_id": segment_id, "source_index": source_index, "frame_id": frame_id,
        "domain_xyxy": (x0, y0, x1, y1), "array_sha256": _array_sha(delta_u, delta_v, weight),
    }
    return S13SourceComponentCorrection(
        segment_id=segment_id, source_index=source_index, frame_id=frame_id,
        x0=x0, x1=x1, y0=y0, y1=y1,
        delta_u=delta_u, delta_v=delta_v, weight=weight,
        correction_sha256=_json_sha(payload),
        normal_field_audit=normal_field_audit,
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


def _candidate_subset_utility(
    candidates: Sequence[S13ComponentSegmentCandidate], indices: Sequence[int]
) -> tuple[float, ...]:
    rows = [_candidate_utility(candidates[index]) for index in indices]
    return (
        sum(row[0] for row in rows),
        sum(row[1] for row in rows),
        sum(row[2] for row in rows),
        min((row[3] for row in rows), default=0.0),
        sum(row[4] for row in rows),
    )


def _s13_conflict_connected_components(
    candidates: Sequence[S13ComponentSegmentCandidate],
    conflicts: set[tuple[int, int]],
) -> tuple[tuple[int, ...], ...]:
    adjacency = {index: set() for index in range(len(candidates))}
    for left, right in conflicts:
        adjacency[left].add(right)
        adjacency[right].add(left)

    def node_key(index: int) -> str:
        return candidates[index].segment.segment_id

    unseen = set(adjacency)
    components: list[tuple[int, ...]] = []
    while unseen:
        pending = [min(unseen, key=node_key)]
        component: set[int] = set()
        while pending:
            index = pending.pop()
            if index not in unseen:
                continue
            unseen.remove(index)
            component.add(index)
            pending.extend(sorted(adjacency[index] & unseen, key=node_key, reverse=True))
        components.append(tuple(sorted(component, key=node_key)))
    return tuple(sorted(components, key=lambda row: tuple(node_key(index) for index in row)))


def _select_s13_conflict_component(
    candidates: Sequence[S13ComponentSegmentCandidate],
    component: Sequence[int],
    *,
    adjacency: Mapping[int, set[int]],
    maximum_exact: int,
) -> tuple[tuple[int, ...], str]:
    if len(component) == 1:
        # An accepted isolated candidate has no trade-off to optimize and must
        # never be discarded merely because its scalar utility is non-positive.
        return (component[0],), "isolated"
    if len(component) <= maximum_exact:
        compatible: list[tuple[tuple[float, ...], tuple[str, ...], tuple[int, ...]]] = []
        for mask in range(1 << len(component)):
            indices = tuple(
                component[local_index]
                for local_index in range(len(component))
                if mask & (1 << local_index)
            )
            selected = set(indices)
            if any(adjacency[index] & selected for index in indices):
                continue
            ids = tuple(sorted(candidates[index].segment.segment_id for index in indices))
            compatible.append((_candidate_subset_utility(candidates, indices), ids, indices))
        best_utility = max(row[0] for row in compatible)
        chosen = min(
            (row for row in compatible if row[0] == best_utility),
            key=lambda row: row[1],
        )[2]
        return tuple(sorted(chosen)), "exact"
    order = sorted(
        component,
        key=lambda index: (
            tuple(-value for value in _candidate_utility(candidates[index])),
            candidates[index].segment.segment_id,
        ),
    )
    chosen: list[int] = []
    selected: set[int] = set()
    for index in order:
        if not (adjacency[index] & selected):
            chosen.append(index)
            selected.add(index)
    return tuple(sorted(chosen)), "lexicographic_greedy"


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
    if maximum_exact < 1:
        raise ValueError("C2E exact conflict-component limit must be positive")
    adjacency = {index: set() for index in range(len(accepted_candidates))}
    for left, right in conflicts:
        adjacency[left].add(right)
        adjacency[right].add(left)
    conflict_components = _s13_conflict_connected_components(
        accepted_candidates, conflicts
    )
    chosen_rows: list[int] = []
    component_audit: list[dict[str, object]] = []
    for component in conflict_components:
        component_chosen, selection_method = _select_s13_conflict_component(
            accepted_candidates,
            component,
            adjacency=adjacency,
            maximum_exact=maximum_exact,
        )
        chosen_rows.extend(component_chosen)
        component_audit.append({
            "segment_ids": tuple(sorted(
                accepted_candidates[index].segment.segment_id for index in component
            )),
            "node_count": len(component),
            "selection_method": selection_method,
        })
    chosen_indices = tuple(sorted(chosen_rows))
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
            "conflict_components": tuple(component_audit),
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
    "S13SourceComponentCorrection", "S13ComponentSolverError",
    "S13RuntimeEdgeTrace", "SourceMapOracle",
    "aggregate_s13_runtime_edge_traces",
    "audit_s13_component_candidate_map_safety", "audit_s13_obligation_coverage",
    "build_s13_edge_component_chains", "build_s13_source_component_correction",
    "compose_s13_source_component_deltas",
    "canonical_s13_support_sha256",
    "canonical_s13_normal_search_lags", "freeze_s13_baseline_c2e_obligations",
    "evaluate_s13_component_candidate_quality", "make_s13_edge_component_observation",
    "make_s13_unevaluable_edge_component_observation",
    "locate_s13_dependency_failure_subset",
    "plan_s13_failed_segment_split",
    "propagate_s13_edge_component_evidence",
    "s13_component_segment_budget_priority",
    "select_s13_component_patch_set", "select_s13_component_segment_split_pair",
    "split_s13_component_application_segment_at_pair",
    "solve_s13_component_segment_source_offsets", "source_map_oracle_from_arrays",
    "split_s13_edge_component_chain", "trace_s13_owner_only_component_edge",
]
