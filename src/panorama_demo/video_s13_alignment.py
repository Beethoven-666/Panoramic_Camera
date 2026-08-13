"""Fail-closed pair-local geometry candidates for the isolated S1.3 M5 stage.

This module estimates only RGB, pair-local target-coordinate corrections.  It
does not render a panorama, change source ownership, read depth, fit a mesh, or
apply photometric correction.  Every candidate sampling map is composed from
the caller's immutable P0 target-to-source grids; candidates are never chained.

The public estimator exposes C0 identity, C1 an explicitly accepted vertical
row correction, C2 subpixel translation, C3 translation plus tiny rotation,
and C4 bounded light affine.  Only the explicitly named non-reference source
is adjusted, within a caller-bounded application band whose raised-cosine
taper is exactly zero at its boundary.  A moved final seam must be handled by
``reestimate_s13_final_corridor_alignment`` so the geometry is re-estimated in
that seam's corridor rather than reusing a midpoint warp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Mapping

import cv2
import numpy as np

from .video_s13_m51_r2 import S13M51R2Config


AlignmentSide = Literal["left", "right"]
QualityEvaluator = Callable[[str, np.ndarray, np.ndarray, np.ndarray], Mapping[str, object]]


@dataclass(frozen=True)
class S13AlignmentConfig:
    """Closed M5 geometry limits, expressed in full-resolution pixels."""

    alignment_shoulder_width_px: int = 96
    maximum_translation_px: float = 4.0
    maximum_rotation_degrees: float = 1.5
    affine_scale_minimum: float = 0.97
    affine_scale_maximum: float = 1.03
    maximum_affine_shear: float = 0.03
    maximum_map_displacement_px: float = 8.0
    minimum_training_points: int = 24
    minimum_held_out_points: int = 12
    held_out_non_degradation_tolerance_px: float = 0.10
    minimum_complex_model_improvement_px: float = 0.05
    maximum_estimated_model_held_out_p95_px: float = 1.50
    minimum_jacobian: float = 0.05
    minimum_support_retention: float = 0.95

    def __post_init__(self) -> None:
        if not 64 <= self.alignment_shoulder_width_px <= 128:
            raise ValueError("S1.3 M5 alignment shoulder must be in [64, 128] px")
        finite = (
            self.maximum_translation_px,
            self.maximum_rotation_degrees,
            self.affine_scale_minimum,
            self.affine_scale_maximum,
            self.maximum_affine_shear,
            self.maximum_map_displacement_px,
            self.held_out_non_degradation_tolerance_px,
            self.minimum_complex_model_improvement_px,
            self.maximum_estimated_model_held_out_p95_px,
            self.minimum_jacobian,
            self.minimum_support_retention,
        )
        if not np.isfinite(finite).all() or any(value < 0.0 for value in finite):
            raise ValueError("S1.3 M5 alignment limits must be finite and non-negative")
        if not 0.0 < self.affine_scale_minimum <= 1.0 <= self.affine_scale_maximum:
            raise ValueError("S1.3 M5 affine scale bounds must contain identity")
        if self.minimum_training_points < 6 or self.minimum_held_out_points < 4:
            raise ValueError("S1.3 M5 alignment needs bounded train and held-out evidence")
        if not 0.0 < self.minimum_jacobian <= 1.0:
            raise ValueError("S1.3 M5 minimum Jacobian must be in (0, 1]")
        if not 0.0 < self.minimum_support_retention <= 1.0:
            raise ValueError("S1.3 M5 support retention must be in (0, 1]")


@dataclass(frozen=True)
class S13ApplicationBand:
    """A per-row, half-open application band constrained by the orchestrator.

    Adjacent-band non-overlap is a sequence-level property and therefore must
    be checked by the caller.  This object validates one pair and never expands
    the supplied bounds.
    """

    left_x_by_row: np.ndarray
    right_x_by_row: np.ndarray

    @classmethod
    def straight(cls, *, height: int, left_x: int, right_x: int) -> S13ApplicationBand:
        return cls(
            np.full(int(height), int(left_x), dtype=np.int32),
            np.full(int(height), int(right_x), dtype=np.int32),
        )

    def validate(self, *, canvas_width: int, canvas_height: int, shoulder_width_px: int) -> None:
        left = np.asarray(self.left_x_by_row)
        right = np.asarray(self.right_x_by_row)
        if left.shape != (canvas_height,) or right.shape != (canvas_height,):
            raise ValueError("S1.3 M5 application band must provide one bound per canvas row")
        if not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(right.dtype, np.integer):
            raise ValueError("S1.3 M5 application band bounds must be integers")
        widths = right.astype(np.int64) - left.astype(np.int64)
        if np.any(left < 0) or np.any(right > canvas_width) or np.any(widths < 3):
            raise ValueError("S1.3 M5 application band is empty or outside the canvas")
        if np.any(widths >= int(shoulder_width_px)):
            raise ValueError("S1.3 M5 application band must be narrower than its shoulder")


@dataclass(frozen=True)
class S13AlignmentCandidate:
    """One audited map candidate, stored only over its compact band bbox."""

    model: str
    x0: int
    x1: int
    target_delta_u: np.ndarray
    target_delta_v: np.ndarray
    source_u: np.ndarray
    source_v: np.ndarray
    valid: np.ndarray
    accepted: bool
    failure_reason: str | None
    metrics: Mapping[str, object]
    audit: Mapping[str, object]


@dataclass(frozen=True)
class S13PairAlignment:
    """All C0--C4 candidates and the fail-closed selected fallback."""

    pair_index: int
    pair_frame_ids: tuple[int, int]
    non_reference_side: AlignmentSide
    alignment_shoulder: tuple[int, int]
    application_band: S13ApplicationBand
    candidates: tuple[S13AlignmentCandidate, ...]
    selected_model: str
    selected_candidate_index: int
    fallback_order: tuple[str, ...]
    reestimated_for_final_seam: bool
    seam_local_evidence: Mapping[str, object] | None = None

    @property
    def selected(self) -> S13AlignmentCandidate:
        return self.candidates[self.selected_candidate_index]


def _as_points(points: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError(f"S1.3 M5 {name} correspondences must be finite Nx2 points")
    return result


def _deterministic_split(
    reference: np.ndarray,
    moving: np.ndarray,
    *,
    minimum_training: int,
    minimum_held_out: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if len(reference) != len(moving):
        raise ValueError("S1.3 M5 correspondence arrays have different lengths")
    if len(reference) < minimum_training + minimum_held_out:
        return None
    key = np.floor(reference[:, 0]).astype(np.int64) * 17 + np.floor(reference[:, 1]).astype(np.int64) * 31
    held = key % 5 == 0
    if int(held.sum()) < minimum_held_out or int((~held).sum()) < minimum_training:
        order = np.lexsort((reference[:, 0], reference[:, 1]))
        held = np.zeros(len(reference), dtype=bool)
        held[order[::5]] = True
    if int(held.sum()) < minimum_held_out or int((~held).sum()) < minimum_training:
        return None
    return reference[~held], moving[~held], reference[held], moving[held]


def _sample_rows(values: np.ndarray, points: np.ndarray) -> np.ndarray:
    rows = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, len(values) - 1)
    return np.asarray(values, dtype=np.float64)[rows]


def _apply_matrix(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return homogeneous @ np.asarray(matrix, dtype=np.float64).T


def _fit_models(train_reference: np.ndarray, train_moving: np.ndarray) -> dict[str, np.ndarray | None]:
    translation = np.median(train_moving - train_reference, axis=0)
    translation_matrix = np.asarray(((1.0, 0.0, translation[0]), (0.0, 1.0, translation[1])), dtype=np.float64)
    rotation, _ = cv2.estimateAffinePartial2D(
        train_reference.astype(np.float32), train_moving.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=1000, confidence=0.99,
    )
    affine, _ = cv2.estimateAffine2D(
        train_reference.astype(np.float32), train_moving.astype(np.float32),
        method=cv2.RANSAC, ransacReprojThreshold=1.0, maxIters=1000, confidence=0.99,
    )
    return {
        "C2_subpixel_translation": translation_matrix,
        "C3_translation_tiny_rotation": None if rotation is None else rotation.astype(np.float64),
        "C4_bounded_light_affine": None if affine is None else affine.astype(np.float64),
    }


def _linear_audit(model: str, matrix: np.ndarray | None, config: S13AlignmentConfig) -> tuple[bool, str | None, dict[str, float | bool | None]]:
    if matrix is None or matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        return False, "model_estimation_failed", {"positive_linear_jacobian": None}
    linear = matrix[:, :2]
    determinant = float(np.linalg.det(linear))
    singular = np.linalg.svd(linear, compute_uv=False)
    rotation = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
    shear = float(0.5 * (linear[0, 1] + linear[1, 0]))
    translation = float(np.linalg.norm(matrix[:, 2]))
    audit: dict[str, float | bool | None] = {
        "linear_jacobian": determinant,
        "positive_linear_jacobian": determinant > config.minimum_jacobian,
        "translation_px": translation,
        "rotation_degrees": rotation,
        "scale_max": float(singular.max()),
        "scale_min": float(singular.min()),
        "shear": shear,
    }
    if determinant <= config.minimum_jacobian:
        return False, "nonpositive_or_small_linear_jacobian", audit
    if translation > config.maximum_translation_px:
        return False, "translation_out_of_bounds", audit
    if model == "C3_translation_tiny_rotation" and abs(rotation) > config.maximum_rotation_degrees:
        return False, "rotation_out_of_bounds", audit
    if model == "C4_bounded_light_affine":
        if abs(rotation) > config.maximum_rotation_degrees:
            return False, "affine_rotation_out_of_bounds", audit
        if singular.min() < config.affine_scale_minimum or singular.max() > config.affine_scale_maximum:
            return False, "affine_scale_out_of_bounds", audit
        if abs(shear) > config.maximum_affine_shear:
            return False, "affine_shear_out_of_bounds", audit
    return True, None, audit


def _band_arrays(band: S13ApplicationBand) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    left = np.asarray(band.left_x_by_row, dtype=np.int32)
    right = np.asarray(band.right_x_by_row, dtype=np.int32)
    x0, x1 = int(left.min()), int(right.max())
    x = np.arange(x0, x1, dtype=np.float64)[None, :]
    width = (right - left).astype(np.float64)
    phase = (x - left[:, None]) / np.maximum(1.0, width[:, None] - 1.0)
    inside = (x >= left[:, None]) & (x < right[:, None])
    # Exact zero displacement at both application boundaries, with a smooth
    # derivative and no discontinuity when the band bends with a curved seam.
    taper = np.where(inside, np.sin(np.pi * np.clip(phase, 0.0, 1.0)) ** 2, 0.0)
    y = np.arange(len(left), dtype=np.float64)[:, None]
    return x0, x1, np.broadcast_to(x, taper.shape), np.broadcast_to(y, taper.shape), taper


def _candidate_delta(
    model: str,
    matrix: np.ndarray | None,
    band: S13ApplicationBand,
    vertical_dy: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    x0, x1, x, y, taper = _band_arrays(band)
    vertical = np.broadcast_to(vertical_dy[:, None], x.shape)
    if model == "C0_identity":
        raw_u = np.zeros_like(x)
        raw_v = np.zeros_like(y)
    elif model == "C1_accepted_vertical":
        raw_u = np.zeros_like(x)
        raw_v = vertical
    else:
        if matrix is None:
            raw_u = np.full_like(x, np.nan)
            raw_v = np.full_like(y, np.nan)
        else:
            base = np.column_stack((x.ravel(), (y + vertical).ravel()))
            moved = _apply_matrix(matrix, base).reshape((*x.shape, 2))
            raw_u = moved[..., 0] - x
            raw_v = moved[..., 1] - y
    return x0, x1, raw_u * taper, raw_v * taper, taper > 0.0


def _compose_from_p0(
    p0_source_u: np.ndarray,
    p0_source_v: np.ndarray,
    p0_valid: np.ndarray,
    *,
    x0: int,
    x1: int,
    delta_u: np.ndarray,
    delta_v: np.ndarray,
    application_mask: np.ndarray,
    source_size: tuple[int, int],
    minimum_jacobian: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    height, width = p0_valid.shape
    target_x = np.broadcast_to(np.arange(x0, x1, dtype=np.float32)[None, :], delta_u.shape)
    target_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], delta_v.shape)
    query_x = target_x + delta_u.astype(np.float32)
    query_y = target_y + delta_v.astype(np.float32)
    p0_u = np.asarray(p0_source_u, dtype=np.float32)
    p0_v = np.asarray(p0_source_v, dtype=np.float32)
    valid_float = np.asarray(p0_valid, dtype=np.uint8) * 255
    source_u = cv2.remap(p0_u, query_x, query_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    source_v = cv2.remap(p0_v, query_x, query_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    sampled_valid = cv2.remap(valid_float, query_x, query_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0
    source_width, source_height = (int(value) for value in source_size)
    finite = np.isfinite(source_u) & np.isfinite(source_v) & np.isfinite(delta_u) & np.isfinite(delta_v)
    in_bounds = (
        (source_u >= 0.0) & (source_u <= source_width - 1)
        & (source_v >= 0.0) & (source_v <= source_height - 1)
    )
    map_x = query_x.astype(np.float64)
    map_y = query_y.astype(np.float64)
    du_dx = np.gradient(map_x, axis=1) if map_x.shape[1] > 1 else np.ones_like(map_x)
    du_dy = np.gradient(map_x, axis=0) if map_x.shape[0] > 1 else np.zeros_like(map_x)
    dv_dx = np.gradient(map_y, axis=1) if map_y.shape[1] > 1 else np.zeros_like(map_y)
    dv_dy = np.gradient(map_y, axis=0) if map_y.shape[0] > 1 else np.ones_like(map_y)
    jacobian = du_dx * dv_dy - du_dy * dv_dx
    # Missing immutable P0 support is not a numerical map failure.  It remains
    # invalid in the returned map and must never be filled by this geometry
    # stage; safety is audited only where this real source had P0 support.
    base_support = application_mask & np.asarray(p0_valid[:, x0:x1], dtype=bool)
    retained = base_support & sampled_valid & finite & in_bounds
    inspected = retained
    positive = bool(np.all(jacobian[inspected] > minimum_jacobian)) if np.any(inspected) else True
    maximum_displacement = float(np.max(np.hypot(delta_u[inspected], delta_v[inspected]))) if np.any(inspected) else 0.0
    valid = sampled_valid & finite & in_bounds
    audit = {
        "map_finite": bool(np.all(finite[inspected])) if np.any(inspected) else True,
        "map_in_source_bounds": bool(np.all(in_bounds[inspected])) if np.any(inspected) else True,
        "positive_jacobian": positive,
        "minimum_jacobian": float(np.min(jacobian[inspected])) if np.any(inspected) else 1.0,
        "maximum_map_displacement_px": maximum_displacement,
        "application_pixel_count": int(inspected.sum()),
        "base_support_pixel_count": int(base_support.sum()),
        "support_retention": (
            float(retained.sum() / base_support.sum()) if np.any(base_support) else 0.0
        ),
        "immutable_p0_grid_composition": True,
    }
    return source_u, source_v, valid, audit


def _correspondence_metrics(
    model: str,
    matrix: np.ndarray | None,
    held_reference: np.ndarray,
    held_moving: np.ndarray,
    vertical_dy: np.ndarray,
) -> dict[str, object]:
    base = held_reference.copy()
    if model != "C0_identity":
        base[:, 1] += _sample_rows(vertical_dy, held_reference)
    predicted = base if model in {"C0_identity", "C1_accepted_vertical"} else _apply_matrix(matrix, base)  # type: ignore[arg-type]
    residual = np.linalg.norm(predicted - held_moving, axis=1)
    finite = residual[np.isfinite(residual)]
    if finite.size == 0:
        return {"evaluable": False, "reason": "nonfinite_held_out_residual", "residual_median_px": None, "residual_p95_px": None}
    return {
        "evaluable": True,
        "reason": None,
        "held_out_count": int(finite.size),
        "residual_median_px": float(np.median(finite)),
        "residual_p95_px": float(np.percentile(finite, 95.0)),
    }


def estimate_s13_pair_alignment(
    *,
    pair_index: int,
    pair_frame_ids: tuple[int, int],
    non_reference_side: AlignmentSide,
    p0_source_u: np.ndarray,
    p0_source_v: np.ndarray,
    p0_valid: np.ndarray,
    source_size: tuple[int, int],
    reference_points_xy: np.ndarray,
    non_reference_points_xy: np.ndarray,
    application_band: S13ApplicationBand,
    accepted_vertical_dy_by_row: np.ndarray | None = None,
    alignment_shoulder: tuple[int, int] | None = None,
    vertical_accepted: bool = False,
    quality_evaluator: QualityEvaluator | None = None,
    config: S13AlignmentConfig | None = None,
    reestimated_for_final_seam: bool = False,
    seam_local_evidence: Mapping[str, object] | None = None,
) -> S13PairAlignment:
    """Estimate and audit C0--C4 from one immutable P0 non-reference grid.

    ``reference_points_xy`` and ``non_reference_points_xy`` are corresponding
    target-canvas coordinates measured in the requested alignment shoulder.
    The optional quality evaluator receives each compact candidate source map;
    it must return ``non_degrading=True`` to accept a non-identity candidate.
    Without it, held-out correspondence residual supplies the local relative
    non-degradation decision.  Sparse evidence therefore selects C0.
    """

    settings = config or S13AlignmentConfig()
    if non_reference_side not in {"left", "right"}:
        raise ValueError("S1.3 M5 must explicitly identify the non-reference side")
    valid = np.asarray(p0_valid, dtype=bool)
    if p0_source_u.shape != valid.shape or p0_source_v.shape != valid.shape or valid.ndim != 2:
        raise ValueError("S1.3 M5 immutable P0 maps must share one HxW shape")
    height, width = valid.shape
    application_band.validate(
        canvas_width=width, canvas_height=height,
        shoulder_width_px=settings.alignment_shoulder_width_px,
    )
    reference = _as_points(reference_points_xy, name="reference")
    moving = _as_points(non_reference_points_xy, name="non-reference")
    if alignment_shoulder is None:
        centre = int(np.median((application_band.left_x_by_row + application_band.right_x_by_row) * 0.5))
        half = settings.alignment_shoulder_width_px // 2
        shoulder = (max(0, centre - half), min(width, centre + half))
    else:
        shoulder = (int(alignment_shoulder[0]), int(alignment_shoulder[1]))
    if shoulder[0] < 0 or shoulder[1] > width or shoulder[1] <= shoulder[0]:
        raise ValueError("S1.3 M5 alignment shoulder is outside the canvas")
    if shoulder[1] - shoulder[0] > settings.alignment_shoulder_width_px:
        raise ValueError("S1.3 M5 alignment shoulder exceeds the configured width")
    in_shoulder = (
        (reference[:, 0] >= shoulder[0]) & (reference[:, 0] < shoulder[1])
        & (moving[:, 0] >= shoulder[0] - settings.maximum_map_displacement_px)
        & (moving[:, 0] < shoulder[1] + settings.maximum_map_displacement_px)
    )
    reference, moving = reference[in_shoulder], moving[in_shoulder]
    vertical = np.zeros(height, dtype=np.float64)
    if accepted_vertical_dy_by_row is not None:
        supplied = np.asarray(accepted_vertical_dy_by_row, dtype=np.float64)
        if supplied.shape != (height,) or not np.isfinite(supplied).all():
            raise ValueError("S1.3 M5 accepted vertical correction must be finite H-vector")
        vertical = supplied if vertical_accepted else vertical
    split = _deterministic_split(
        reference, moving,
        minimum_training=settings.minimum_training_points,
        minimum_held_out=settings.minimum_held_out_points,
    )
    models: dict[str, np.ndarray | None] = {
        "C0_identity": np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        "C1_accepted_vertical": np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        "C2_subpixel_translation": None,
        "C3_translation_tiny_rotation": None,
        "C4_bounded_light_affine": None,
    }
    held_reference = held_moving = np.empty((0, 2), dtype=np.float64)
    if split is not None:
        train_reference, train_moving, held_reference, held_moving = split
        train_base = train_reference.copy()
        train_base[:, 1] += _sample_rows(vertical, train_reference)
        models.update(_fit_models(train_base, train_moving))
    candidates: list[S13AlignmentCandidate] = []
    identity_p95: float | None = None
    prior_p95: float | None = None
    for model in models:
        matrix = models[model]
        numeric_pass, failure, linear_audit = _linear_audit(model, matrix, settings)
        if model == "C0_identity":
            numeric_pass, failure = True, None
        elif model == "C1_accepted_vertical" and not vertical_accepted:
            numeric_pass, failure = False, "vertical_parent_not_accepted"
        x0, x1, delta_u, delta_v, application_mask = _candidate_delta(model, matrix, application_band, vertical)
        source_u, source_v, map_valid, map_audit = _compose_from_p0(
            p0_source_u, p0_source_v, valid, x0=x0, x1=x1,
            delta_u=delta_u, delta_v=delta_v, application_mask=application_mask,
            source_size=source_size, minimum_jacobian=settings.minimum_jacobian,
        )
        numeric_pass = bool(
            numeric_pass
            and map_audit["map_finite"]
            and map_audit["map_in_source_bounds"]
            and map_audit["positive_jacobian"]
            and float(map_audit["maximum_map_displacement_px"]) <= settings.maximum_map_displacement_px
            and float(map_audit["support_retention"]) >= settings.minimum_support_retention
        )
        # C0 is the immutable P0 sampling identity and is always a candidate.
        # Existing P0 holes stay invalid; they are not grounds for eliminating
        # the fail-closed parent or for synthesising replacement support.
        if model == "C0_identity":
            numeric_pass, failure = True, None
        if not numeric_pass and failure is None:
            failure = "unsafe_sampling_map"
        metrics = _correspondence_metrics(model, matrix, held_reference, held_moving, vertical) if split is not None else {
            "evaluable": False, "reason": "insufficient_train_held_out_evidence",
            "residual_median_px": None, "residual_p95_px": None,
        }
        p95 = metrics.get("residual_p95_px")
        if model == "C0_identity" and isinstance(p95, (int, float)):
            identity_p95 = float(p95)
            prior_p95 = identity_p95
        quality_pass = model == "C0_identity"
        if (
            model in {
                "C2_subpixel_translation",
                "C3_translation_tiny_rotation",
                "C4_bounded_light_affine",
            }
            and numeric_pass
            and isinstance(p95, (int, float))
            and float(p95) > settings.maximum_estimated_model_held_out_p95_px
        ):
            numeric_pass = False
            failure = "held_out_residual_out_of_bounds"
        if model != "C0_identity" and numeric_pass:
            if quality_evaluator is not None:
                external = dict(quality_evaluator(model, source_u, source_v, map_valid))
                metrics = {**dict(metrics), "structure": external}
                quality_pass = external.get("non_degrading") is True
                if not quality_pass:
                    failure = str(external.get("reason") or "local_structure_degraded_or_unevaluable")
            elif isinstance(p95, (int, float)) and prior_p95 is not None:
                allowed = prior_p95 + settings.held_out_non_degradation_tolerance_px
                quality_pass = float(p95) <= allowed
                if model in {"C3_translation_tiny_rotation", "C4_bounded_light_affine"}:
                    quality_pass = quality_pass and (
                        prior_p95 - float(p95) >= settings.minimum_complex_model_improvement_px
                    )
                if not quality_pass:
                    failure = "held_out_structure_not_better_than_simpler_candidate"
            else:
                failure = "insufficient_relative_structure_evidence"
        accepted = bool(numeric_pass and quality_pass)
        if accepted and isinstance(p95, (int, float)):
            prior_p95 = min(float(p95), prior_p95 if prior_p95 is not None else float(p95))
        candidates.append(S13AlignmentCandidate(
            model=model, x0=x0, x1=x1,
            target_delta_u=delta_u.astype(np.float32), target_delta_v=delta_v.astype(np.float32),
            source_u=source_u.astype(np.float32), source_v=source_v.astype(np.float32), valid=map_valid,
            accepted=accepted, failure_reason=None if accepted else failure,
            metrics=metrics,
            audit={
                **linear_audit, **map_audit,
                "non_reference_side": non_reference_side,
                "continuous_taper_to_identity": True,
                "candidate_built_from_immutable_p0_grid": True,
                "identity_held_out_residual_p95_px": identity_p95,
                "maximum_estimated_model_held_out_p95_px": (
                    settings.maximum_estimated_model_held_out_p95_px
                ),
            },
        ))
    fallback_order = (
        "C4_bounded_light_affine", "C3_translation_tiny_rotation",
        "C2_subpixel_translation", "C1_accepted_vertical", "C0_identity",
    )
    by_model = {candidate.model: index for index, candidate in enumerate(candidates)}
    selected_model = next(model for model in fallback_order if candidates[by_model[model]].accepted)
    return S13PairAlignment(
        pair_index=int(pair_index), pair_frame_ids=(int(pair_frame_ids[0]), int(pair_frame_ids[1])),
        non_reference_side=non_reference_side, alignment_shoulder=shoulder,
        application_band=application_band, candidates=tuple(candidates),
        selected_model=selected_model, selected_candidate_index=by_model[selected_model],
        fallback_order=fallback_order, reestimated_for_final_seam=bool(reestimated_for_final_seam),
        seam_local_evidence=(
            None if seam_local_evidence is None else dict(seam_local_evidence)
        ),
    )


def final_corridor_application_band(
    seam_x_by_row: np.ndarray,
    *,
    canvas_width: int,
    half_width_px: int,
    allowed_left_x_by_row: np.ndarray | None = None,
    allowed_right_x_by_row: np.ndarray | None = None,
) -> S13ApplicationBand:
    """Build, but never expand, a caller-constrained band around a final seam."""

    seam = np.asarray(seam_x_by_row, dtype=np.int32)
    if seam.ndim != 1 or half_width_px < 2 or not np.all((seam >= 0) & (seam < canvas_width)):
        raise ValueError("S1.3 M5 final seam or application half-width is invalid")
    left = seam - int(half_width_px)
    right = seam + int(half_width_px) + 1
    if allowed_left_x_by_row is not None:
        allowed_left = np.asarray(allowed_left_x_by_row, dtype=np.int32)
        if allowed_left.shape != seam.shape:
            raise ValueError("S1.3 M5 allowed left bounds do not match final seam")
        left = np.maximum(left, allowed_left)
    if allowed_right_x_by_row is not None:
        allowed_right = np.asarray(allowed_right_x_by_row, dtype=np.int32)
        if allowed_right.shape != seam.shape:
            raise ValueError("S1.3 M5 allowed right bounds do not match final seam")
        right = np.minimum(right, allowed_right)
    left = np.maximum(left, 0)
    right = np.minimum(right, int(canvas_width))
    return S13ApplicationBand(left.astype(np.int32), right.astype(np.int32))


def filter_s13_correspondences_for_final_seam(
    reference_points_xy: np.ndarray,
    non_reference_points_xy: np.ndarray,
    final_seam_x_by_row: np.ndarray,
    *,
    evidence_half_widths_px: tuple[int, ...] = (16, 24),
    moving_margin_px: int = 8,
    minimum_required_points: int = 36,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    """Select deterministic, seam-local evidence without a shoulder fallback."""

    reference = np.asarray(reference_points_xy, dtype=np.float64)
    moving = np.asarray(non_reference_points_xy, dtype=np.float64)
    seam = np.asarray(final_seam_x_by_row)
    widths = tuple(int(value) for value in evidence_half_widths_px)
    if (
        reference.ndim != 2
        or reference.shape[1:] != (2,)
        or moving.shape != reference.shape
    ):
        raise ValueError("S1.3 M5 seam-local correspondences must be paired Nx2 arrays")
    if seam.ndim != 1 or not np.issubdtype(seam.dtype, np.integer) or seam.size == 0:
        raise ValueError("S1.3 M5 final seam must be a non-empty integer row vector")
    if (
        not widths
        or any(value <= 0 for value in widths)
        or tuple(sorted(set(widths))) != widths
        or moving_margin_px < 0
        or minimum_required_points < 1
    ):
        raise ValueError("S1.3 M5 seam-local evidence limits are invalid")

    finite = np.isfinite(reference).all(axis=1) & np.isfinite(moving).all(axis=1)
    ref_y_ok = finite & (reference[:, 1] >= 0.0) & (reference[:, 1] <= seam.size - 1.0)
    mov_y_ok = finite & (moving[:, 1] >= 0.0) & (moving[:, 1] <= seam.size - 1.0)
    rows_ok = ref_y_ok & mov_y_ok
    valid_indices = np.flatnonzero(rows_ok)
    ref_rows = np.rint(reference[valid_indices, 1]).astype(np.int32)
    mov_rows = np.rint(moving[valid_indices, 1]).astype(np.int32)
    ref_distance = np.abs(reference[valid_indices, 0] - seam[ref_rows])
    mov_distance = np.abs(moving[valid_indices, 0] - seam[mov_rows])

    selected_indices: np.ndarray | None = None
    selected_width: int | None = None
    counts_by_width: dict[str, int] = {}
    for width in widths:
        local = (ref_distance <= width) & (
            mov_distance <= width + int(moving_margin_px)
        )
        indices = valid_indices[local]
        counts_by_width[str(width)] = int(indices.size)
        if indices.size >= minimum_required_points:
            selected_indices = indices
            selected_width = width
            break

    sufficient = selected_indices is not None
    if selected_indices is None:
        selected_indices = np.empty(0, dtype=np.int64)
    audit: dict[str, object] = {
        "requested_half_widths_px": list(widths),
        "selected_half_width_px": selected_width,
        "raw_count": int(len(reference)),
        "finite_in_bounds_row_count": int(valid_indices.size),
        "selected_count": int(selected_indices.size),
        "minimum_required_count": int(minimum_required_points),
        "sufficient_for_train_held_out": sufficient,
        "wide_evidence": bool(selected_width is not None and selected_width != widths[0]),
        "full_shoulder_fallback_used": False,
        "mode": "seam_local" if sufficient else "seam_local_insufficient",
        "counts_by_half_width_px": counts_by_width,
    }
    return reference[selected_indices], moving[selected_indices], audit


def reestimate_s13_final_corridor_alignment(
    *,
    final_seam_x_by_row: np.ndarray,
    application_half_width_px: int,
    allowed_left_x_by_row: np.ndarray | None = None,
    allowed_right_x_by_row: np.ndarray | None = None,
    m51_r2_config: S13M51R2Config | None = None,
    **estimator_arguments: object,
) -> S13PairAlignment:
    """Re-estimate geometry around a moved seam from the immutable P0 grids.

    The caller supplies the same immutable P0 maps and raw correspondences used
    for candidate analysis.  No previously selected midpoint candidate is an
    input.  Allowed bounds are intended to encode sequence-level non-overlap.
    """

    p0_valid = np.asarray(estimator_arguments.get("p0_valid"), dtype=bool)
    if p0_valid.ndim != 2:
        raise ValueError("S1.3 M5 final-corridor re-estimation requires immutable P0 validity")
    band = final_corridor_application_band(
        final_seam_x_by_row, canvas_width=p0_valid.shape[1],
        half_width_px=int(application_half_width_px),
        allowed_left_x_by_row=allowed_left_x_by_row,
        allowed_right_x_by_row=allowed_right_x_by_row,
    )
    arguments = dict(estimator_arguments)
    successor = m51_r2_config or S13M51R2Config()
    if successor.enabled:
        reference, moving, evidence_audit = filter_s13_correspondences_for_final_seam(
            np.asarray(arguments.get("reference_points_xy"), dtype=np.float64),
            np.asarray(arguments.get("non_reference_points_xy"), dtype=np.float64),
            np.asarray(final_seam_x_by_row),
            evidence_half_widths_px=successor.evidence_half_widths_px,
            moving_margin_px=successor.moving_evidence_margin_px,
            minimum_required_points=36,
        )
        arguments["reference_points_xy"] = reference
        arguments["non_reference_points_xy"] = moving
        arguments["seam_local_evidence"] = evidence_audit
    arguments["application_band"] = band
    arguments["reestimated_for_final_seam"] = True
    return estimate_s13_pair_alignment(**arguments)  # type: ignore[arg-type]


__all__ = [
    "AlignmentSide", "QualityEvaluator", "S13AlignmentCandidate", "S13AlignmentConfig",
    "S13ApplicationBand", "S13PairAlignment", "estimate_s13_pair_alignment",
    "filter_s13_correspondences_for_final_seam", "final_corridor_application_band",
    "reestimate_s13_final_corridor_alignment",
]
