"""Fail-closed global photometric calibration for video render sources.

This module deliberately has no knowledge of a canvas, a seam, or an owner
map.  It consumes only already-aligned adjacent source overlap samples and
returns one linear-light correction for each *real* source frame.  Callers may
apply a result only when :attr:`GlobalPhotometricResult.accepted` is true.

The relation between two raw source images is fitted in linear BGR light as
``left ~= pair_gain * right + pair_bias``.  The per-source corrections then
map every source into source zero's photometric domain.  Invalid, incomplete,
or out-of-bounds evidence never produces a partial correction: the returned
corrections are exactly identity and the audit explains why it was rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import cv2
import numpy as np


PhotometricModel = Literal["gain_only", "gain_bias"]


@dataclass(frozen=True)
class VideoPhotometricConfig:
    """Conservative bounds for global video photometric calibration.

    All gains and biases below are in linear-light BGR.  The limits are
    intentionally narrow: a physical auto-exposure transition can be
    compensated, while an unsupported scene or correspondence mismatch is
    rejected instead of being painted over by a large colour transform.
    """

    model: PhotometricModel = "gain_bias"
    minimum_support_pixels: int = 192
    minimum_inlier_fraction: float = 0.55
    stable_gradient_quantile: float = 0.70
    pair_gain_minimum: float = 0.72
    pair_gain_maximum: float = 1.38
    pair_bias_limit: float = 0.10
    global_gain_minimum: float = 0.55
    # A scan-wide one-stop change is still within the documented fixed-video
    # exposure transition envelope.  The per-pair bounds and held-out residual
    # gates remain the primary evidence; this aggregate bound prevents an
    # otherwise well-supported sequence from being rejected solely because 50+
    # small, individually accepted corrections compose to 1.8x.
    global_gain_maximum: float = 2.00
    global_bias_limit: float = 0.18
    maximum_pair_residual_p95: float = 0.035
    maximum_pair_residual_max: float = 0.12
    # The relation must be demonstrated on pixels not used to fit it.  The
    # split is deterministic and tile based (rather than random per pixel) so
    # neighbouring samples cannot trivially leak the same texture into both
    # populations.
    held_out_tile_side_pixels: int = 16
    held_out_tile_modulus: int = 5
    held_out_tile_remainder: int = 0
    minimum_training_pixels: int = 128
    minimum_held_out_pixels: int = 64
    maximum_held_out_residual_p95: float = 0.035
    maximum_held_out_residual_max: float = 0.12
    irls_iterations: int = 4

    def validated(self) -> "VideoPhotometricConfig":
        """Return this configuration after rejecting unsafe values."""

        if self.model not in {"gain_only", "gain_bias"}:
            raise ValueError("Photometric model must be gain_only or gain_bias")
        if self.minimum_support_pixels < 32:
            raise ValueError("Photometric minimum_support_pixels must be at least 32")
        finite = np.asarray(
            (
                self.minimum_inlier_fraction,
                self.stable_gradient_quantile,
                self.pair_gain_minimum,
                self.pair_gain_maximum,
                self.pair_bias_limit,
                self.global_gain_minimum,
                self.global_gain_maximum,
                self.global_bias_limit,
                self.maximum_pair_residual_p95,
                self.maximum_pair_residual_max,
                self.maximum_held_out_residual_p95,
                self.maximum_held_out_residual_max,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(finite).all():
            raise ValueError("Photometric configuration must be finite")
        if not 0.25 <= self.minimum_inlier_fraction <= 1.0:
            raise ValueError("Photometric minimum_inlier_fraction is invalid")
        if not 0.05 <= self.stable_gradient_quantile <= 1.0:
            raise ValueError("Photometric stable_gradient_quantile is invalid")
        if not 0.0 < self.pair_gain_minimum <= self.pair_gain_maximum:
            raise ValueError("Photometric pair gain limits are invalid")
        if not 0.0 < self.global_gain_minimum <= self.global_gain_maximum:
            raise ValueError("Photometric global gain limits are invalid")
        if min(
            self.pair_bias_limit,
            self.global_bias_limit,
            self.maximum_pair_residual_p95,
            self.maximum_pair_residual_max,
        ) <= 0.0:
            raise ValueError("Photometric positive limits are invalid")
        if self.maximum_pair_residual_p95 > self.maximum_pair_residual_max:
            raise ValueError("Photometric residual limits are invalid")
        if self.maximum_held_out_residual_p95 > self.maximum_held_out_residual_max:
            raise ValueError("Photometric held-out residual limits are invalid")
        if self.held_out_tile_side_pixels < 4:
            raise ValueError("Photometric held-out tile side must be at least 4")
        if self.held_out_tile_modulus < 2:
            raise ValueError("Photometric held-out tile modulus must be at least 2")
        if not 0 <= self.held_out_tile_remainder < self.held_out_tile_modulus:
            raise ValueError("Photometric held-out tile remainder is invalid")
        if self.minimum_training_pixels < 32:
            raise ValueError("Photometric minimum_training_pixels must be at least 32")
        if self.minimum_held_out_pixels < 32:
            raise ValueError("Photometric minimum_held_out_pixels must be at least 32")
        if self.irls_iterations < 1 or self.irls_iterations > 16:
            raise ValueError("Photometric irls_iterations must be within 1..16")
        return self


@dataclass(frozen=True)
class AdjacentBGRAOverlap:
    """One direct adjacent-source overlap in a common pixel coordinate system.

    ``left_valid`` and ``right_valid`` are evidence masks, not merely alpha
    masks.  A renderer must supply only mutually visible, safe, same-layer
    background pixels; foreground, depth-edge, occlusion and seam-risk pixels
    must already be false.  The solver cannot infer that geometry from RGB.
    """

    left_source_index: int
    right_source_index: int
    left_bgra: np.ndarray
    right_bgra: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray


@dataclass(frozen=True)
class VideoPhotometricCorrection:
    """One source's linear-light BGR correction into the global domain."""

    gain_bgr: np.ndarray
    bias_bgr: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "gain_bgr": [float(value) for value in self.gain_bgr],
            "bias_bgr": [float(value) for value in self.bias_bgr],
        }


@dataclass(frozen=True)
class GlobalPhotometricResult:
    """Global source corrections plus JSON-safe scalar audit evidence."""

    accepted: bool
    corrections: tuple[VideoPhotometricCorrection, ...]
    audit: dict[str, object]


def _identity_corrections(source_count: int) -> tuple[VideoPhotometricCorrection, ...]:
    return tuple(
        VideoPhotometricCorrection(
            gain_bgr=np.ones(3, dtype=np.float64),
            bias_bgr=np.zeros(3, dtype=np.float64),
        )
        for _ in range(source_count)
    )


def _rejected_result(
    source_count: int,
    *,
    config: VideoPhotometricConfig,
    reason: str,
    pair_audit: Sequence[dict[str, object]] = (),
) -> GlobalPhotometricResult:
    return GlobalPhotometricResult(
        accepted=False,
        corrections=_identity_corrections(source_count),
        audit={
            "schema": "g305-video-global-photometric/v1",
            "accepted": False,
            "fail_closed_identity": True,
            "model": config.model,
            "rejection_reason": reason,
            "pair_count": len(pair_audit),
            "pairs": list(pair_audit),
        },
    )


def _srgb_to_linear_bgr(image: np.ndarray) -> np.ndarray:
    encoded = np.asarray(image, dtype=np.float32) / 255.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb_bgr(image: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded * 255.0, 0.0, 255.0)).astype(np.uint8)


def _validated_bgra_and_mask(
    image: np.ndarray,
    valid: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    bgra = np.asarray(image)
    if bgra.ndim != 3 or bgra.shape[2] != 4 or bgra.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8 BGRA")
    mask = np.asarray(valid)
    if mask.shape != bgra.shape[:2] or mask.dtype not in (np.bool_, np.uint8):
        raise ValueError(f"{name} valid mask must be matching bool or uint8")
    return np.ascontiguousarray(bgra), np.ascontiguousarray((mask > 0) & (bgra[:, :, 3] > 0))


def _stable_overlap_mask(
    left_bgra: np.ndarray,
    right_bgra: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    *,
    quantile: float,
) -> np.ndarray:
    common = np.asarray(left_valid, dtype=bool) & np.asarray(right_valid, dtype=bool)
    if not np.any(common):
        return common
    left = left_bgra[:, :, :3]
    right = right_bgra[:, :, :3]
    # Exclude black, clipped highlights and alpha-only pixels before estimating
    # any relation.  Valid black remains valid for rendering; it is merely not
    # identifiable evidence for a gain/bias fit.
    unclipped = (
        (np.min(left, axis=2) >= 8)
        & (np.max(left, axis=2) <= 247)
        & (np.min(right, axis=2) >= 8)
        & (np.max(right, axis=2) <= 247)
    )
    candidates = common & unclipped
    if int(np.count_nonzero(candidates)) < 32:
        return np.zeros_like(common)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    left_gradient = cv2.magnitude(
        cv2.Scharr(left_gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(left_gray, cv2.CV_32F, 0, 1),
    )
    right_gradient = cv2.magnitude(
        cv2.Scharr(right_gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(right_gray, cv2.CV_32F, 0, 1),
    )
    values = np.concatenate((left_gradient[candidates], right_gradient[candidates]))
    threshold = float(np.percentile(values, 100.0 * quantile))
    return np.ascontiguousarray(
        candidates & (left_gradient <= threshold) & (right_gradient <= threshold)
    )


def _spatial_train_held_out_masks(
    stable: np.ndarray,
    *,
    config: VideoPhotometricConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, spatially separated evidence populations.

    The caller's valid masks must already restrict the evidence to common,
    safe, same-layer background.  This function merely keeps the photometric
    fit honest: every retained tile is wholly train *or* held-out.  A global
    source correction therefore cannot report success from the same samples
    that selected its gain/bias.
    """

    mask = np.asarray(stable, dtype=bool)
    rows, columns = np.indices(mask.shape, sparse=True)
    tile_index = (
        rows // int(config.held_out_tile_side_pixels)
        + columns // int(config.held_out_tile_side_pixels)
    )
    held_out = mask & (
        np.asarray(tile_index % int(config.held_out_tile_modulus))
        == int(config.held_out_tile_remainder)
    )
    training = mask & ~held_out
    return np.ascontiguousarray(training), np.ascontiguousarray(held_out)


def _fit_channel_relation(
    left: np.ndarray,
    right: np.ndarray,
    *,
    gain_only: bool,
    iterations: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit ``left ~= gain * right + bias`` with deterministic Huber IRLS."""

    x = np.asarray(right, dtype=np.float64)
    y = np.asarray(left, dtype=np.float64)
    kept = np.ones(x.shape, dtype=bool)
    gain = 1.0
    bias = 0.0
    for _ in range(iterations):
        xx, yy = x[kept], y[kept]
        if xx.size < 3:
            break
        if gain_only:
            denominator = float(np.dot(xx, xx))
            if denominator <= 1e-12:
                break
            gain, bias = float(np.dot(xx, yy) / denominator), 0.0
        else:
            design = np.column_stack((xx, np.ones(xx.size, dtype=np.float64)))
            solution, _residuals, rank, _singular_values = np.linalg.lstsq(
                design, yy, rcond=None
            )
            if rank != 2:
                break
            gain, bias = float(solution[0]), float(solution[1])
        residual = y - (gain * x + bias)
        median = float(np.median(residual[kept]))
        mad = float(np.median(np.abs(residual[kept] - median)))
        # The fixed floor prevents quantisation noise from making a single
        # source pixel decide the global correction.
        threshold = max(0.0025, 3.5 * 1.4826 * mad)
        next_kept = np.abs(residual - median) <= threshold
        if np.array_equal(next_kept, kept):
            kept = next_kept
            break
        kept = next_kept
    residual = y - (gain * x + bias)
    return gain, bias, kept, residual


@dataclass(frozen=True)
class _PairRelation:
    left_index: int
    right_index: int
    gain_bgr: np.ndarray
    bias_bgr: np.ndarray
    support_pixels: int
    training_pixels: int
    held_out_pixels: int
    inlier_pixels: int
    residual_p95: float
    residual_max: float
    held_out_residual_p95: float
    held_out_residual_max: float
    audit: dict[str, object]


def _estimate_pair_relation(
    overlap: AdjacentBGRAOverlap,
    *,
    config: VideoPhotometricConfig,
) -> _PairRelation | None:
    left_bgra, left_valid = _validated_bgra_and_mask(
        overlap.left_bgra, overlap.left_valid, name="left BGRA"
    )
    right_bgra, right_valid = _validated_bgra_and_mask(
        overlap.right_bgra, overlap.right_valid, name="right BGRA"
    )
    if left_bgra.shape != right_bgra.shape:
        raise ValueError("Adjacent photometric BGRA overlaps must have equal shape")
    stable = _stable_overlap_mask(
        left_bgra,
        right_bgra,
        left_valid,
        right_valid,
        quantile=config.stable_gradient_quantile,
    )
    support = int(np.count_nonzero(stable))
    training_mask, held_out_mask = _spatial_train_held_out_masks(stable, config=config)
    training_count = int(np.count_nonzero(training_mask))
    held_out_count = int(np.count_nonzero(held_out_mask))
    base_audit: dict[str, object] = {
        "left_source_index": int(overlap.left_source_index),
        "right_source_index": int(overlap.right_source_index),
        "stable_support_pixels": support,
        "training_pixels": training_count,
        "held_out_pixels": held_out_count,
        "held_out_split": {
            "kind": "deterministic_spatial_tiles/v1",
            "tile_side_pixels": int(config.held_out_tile_side_pixels),
            "tile_modulus": int(config.held_out_tile_modulus),
            "tile_remainder": int(config.held_out_tile_remainder),
        },
    }
    if (
        support < config.minimum_support_pixels
        or training_count < config.minimum_training_pixels
        or held_out_count < config.minimum_held_out_pixels
    ):
        if support < config.minimum_support_pixels:
            reason = "insufficient_stable_support"
        elif training_count < config.minimum_training_pixels:
            reason = "insufficient_training_support"
        else:
            reason = "insufficient_held_out_support"
        base_audit.update({"accepted": False, "reason": reason})
        return _PairRelation(
            int(overlap.left_source_index),
            int(overlap.right_source_index),
            np.ones(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            support,
            training_count,
            held_out_count,
            0,
            math.inf,
            math.inf,
            math.inf,
            math.inf,
            base_audit,
        )
    left_linear_image = _srgb_to_linear_bgr(left_bgra[:, :, :3])
    right_linear_image = _srgb_to_linear_bgr(right_bgra[:, :, :3])
    left_linear = left_linear_image[training_mask]
    right_linear = right_linear_image[training_mask]
    gains = np.empty(3, dtype=np.float64)
    biases = np.empty(3, dtype=np.float64)
    inlier_masks: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for channel in range(3):
        gain, bias, inliers, residual = _fit_channel_relation(
            left_linear[:, channel],
            right_linear[:, channel],
            gain_only=config.model == "gain_only",
            iterations=config.irls_iterations,
        )
        gains[channel], biases[channel] = gain, bias
        inlier_masks.append(inliers)
        residuals.append(residual)
    inliers = np.logical_and.reduce(inlier_masks)
    inlier_count = int(np.count_nonzero(inliers))
    combined_residual = np.max(
        np.abs(np.column_stack(residuals)), axis=1
    )
    residual_p95 = float(np.percentile(combined_residual[inliers], 95.0)) if inlier_count else math.inf
    residual_max = float(np.max(combined_residual[inliers])) if inlier_count else math.inf
    held_out_residual = np.max(
        np.abs(
            left_linear_image[held_out_mask]
            - (
                right_linear_image[held_out_mask] * gains.reshape(1, 3)
                + biases.reshape(1, 3)
            )
        ),
        axis=1,
    )
    held_out_residual_p95 = float(np.percentile(held_out_residual, 95.0))
    held_out_residual_max = float(np.max(held_out_residual))
    accepted = bool(
        np.isfinite(gains).all()
        and np.isfinite(biases).all()
        and inlier_count >= config.minimum_support_pixels
        and inlier_count >= math.ceil(config.minimum_inlier_fraction * training_count)
        and np.all(gains >= config.pair_gain_minimum)
        and np.all(gains <= config.pair_gain_maximum)
        and (
            config.model == "gain_only"
            or np.all(np.abs(biases) <= config.pair_bias_limit)
        )
        and residual_p95 <= config.maximum_pair_residual_p95
        and residual_max <= config.maximum_pair_residual_max
        and held_out_residual_p95 <= config.maximum_held_out_residual_p95
        and held_out_residual_max <= config.maximum_held_out_residual_max
    )
    reason = "accepted"
    if not accepted:
        if not np.isfinite(gains).all() or not np.isfinite(biases).all():
            reason = "nonfinite_relation"
        elif inlier_count < config.minimum_training_pixels:
            reason = "insufficient_robust_support"
        elif inlier_count < math.ceil(config.minimum_inlier_fraction * training_count):
            reason = "insufficient_inlier_fraction"
        elif np.any(gains < config.pair_gain_minimum) or np.any(gains > config.pair_gain_maximum):
            reason = "pair_gain_out_of_bounds"
        elif config.model == "gain_bias" and np.any(np.abs(biases) > config.pair_bias_limit):
            reason = "pair_bias_out_of_bounds"
        elif (
            held_out_residual_p95 > config.maximum_held_out_residual_p95
            or held_out_residual_max > config.maximum_held_out_residual_max
        ):
            reason = "held_out_residual_out_of_bounds"
        else:
            reason = "pair_residual_out_of_bounds"
    base_audit.update(
        {
            "accepted": accepted,
            "reason": reason,
            "inlier_pixels": inlier_count,
            "inlier_fraction": float(inlier_count / support),
            "gain_bgr": [float(value) for value in gains],
            "bias_bgr": [float(value) for value in biases],
            "residual_p95_linear": residual_p95,
            "residual_max_linear": residual_max,
            "held_out_residual_p95_linear": held_out_residual_p95,
            "held_out_residual_max_linear": held_out_residual_max,
        }
    )
    return _PairRelation(
        int(overlap.left_source_index),
        int(overlap.right_source_index),
        gains,
        biases,
        support,
        training_count,
        held_out_count,
        inlier_count,
        residual_p95,
        residual_max,
        held_out_residual_p95,
        held_out_residual_max,
        base_audit,
    )


def solve_video_global_photometric(
    source_count: int,
    overlaps: Sequence[AdjacentBGRAOverlap],
    *,
    config: VideoPhotometricConfig | None = None,
) -> GlobalPhotometricResult:
    """Solve one bounded linear BGR correction for every real video source.

    ``overlaps`` must contain exactly the chronological adjacent edges
    ``(0, 1), (1, 2), ...``.  A missing, malformed, disconnected, or rejected
    edge returns identity for *all* sources.  This avoids the dangerous
    half-corrected panorama where one photometric component has no trustworthy
    relation to its neighbour.
    """

    settings = (config or VideoPhotometricConfig()).validated()
    if source_count < 2:
        raise ValueError("Global video photometry requires at least two sources")
    if len(overlaps) != source_count - 1:
        return _rejected_result(
            source_count,
            config=settings,
            reason="adjacent_edge_coverage_incomplete",
        )
    expected_edges = [(index, index + 1) for index in range(source_count - 1)]
    actual_edges = [
        (int(overlap.left_source_index), int(overlap.right_source_index))
        for overlap in overlaps
    ]
    if actual_edges != expected_edges:
        return _rejected_result(
            source_count,
            config=settings,
            reason="adjacent_edges_not_strictly_chronological",
        )
    relations: list[_PairRelation] = []
    pair_audit: list[dict[str, object]] = []
    try:
        for overlap in overlaps:
            relation = _estimate_pair_relation(overlap, config=settings)
            assert relation is not None
            relations.append(relation)
            pair_audit.append(relation.audit)
    except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
        return _rejected_result(
            source_count,
            config=settings,
            reason=f"malformed_overlap:{exc}",
            pair_audit=pair_audit,
        )
    rejected = next((relation for relation in relations if not relation.audit["accepted"]), None)
    if rejected is not None:
        return _rejected_result(
            source_count,
            config=settings,
            reason=f"rejected_pair_{rejected.left_index}_{rejected.right_index}:{rejected.audit['reason']}",
            pair_audit=pair_audit,
        )

    # A sequence is an adjacent graph, yet solve the whole log-gain system at
    # once rather than accumulating raw floats.  The anchored least-squares
    # formulation remains deterministic and gives an explicit global residual
    # audit, ready for a future non-adjacent loop-closure edge without changing
    # source-correction semantics.
    rows = np.zeros((source_count, source_count), dtype=np.float64)
    rhs = np.zeros((source_count, 3), dtype=np.float64)
    rows[0, 0] = 1.0
    for index, relation in enumerate(relations, start=1):
        rows[index, relation.left_index] = -1.0
        rows[index, relation.right_index] = 1.0
        rhs[index] = np.log(relation.gain_bgr)
    try:
        log_gains, _residuals, rank, _singular_values = np.linalg.lstsq(rows, rhs, rcond=None)
    except np.linalg.LinAlgError:
        return _rejected_result(
            source_count,
            config=settings,
            reason="global_gain_solver_failure",
            pair_audit=pair_audit,
        )
    if rank != source_count or not np.isfinite(log_gains).all():
        return _rejected_result(
            source_count,
            config=settings,
            reason="global_gain_solver_rank_failure",
            pair_audit=pair_audit,
        )
    gains = np.exp(log_gains)
    biases = np.zeros((source_count, 3), dtype=np.float64)
    for relation in relations:
        left, right = relation.left_index, relation.right_index
        biases[right] = gains[left] * relation.bias_bgr + biases[left]
    residual = rows @ log_gains - rhs
    if (
        not np.isfinite(gains).all()
        or not np.isfinite(biases).all()
        or np.any(gains < settings.global_gain_minimum)
        or np.any(gains > settings.global_gain_maximum)
        or (settings.model == "gain_bias" and np.any(np.abs(biases) > settings.global_bias_limit))
    ):
        return _rejected_result(
            source_count,
            config=settings,
            reason="global_correction_out_of_bounds",
            pair_audit=pair_audit,
        )
    corrections = tuple(
        VideoPhotometricCorrection(
            gain_bgr=np.ascontiguousarray(gains[index]),
            bias_bgr=np.ascontiguousarray(biases[index]),
        )
        for index in range(source_count)
    )
    return GlobalPhotometricResult(
        accepted=True,
        corrections=corrections,
        audit={
            "schema": "g305-video-global-photometric/v1",
            "accepted": True,
            "fail_closed_identity": False,
            "model": settings.model,
            "linear_light": True,
            "source_count": source_count,
            "pair_count": len(relations),
            "training_pixel_count": int(
                sum(relation.training_pixels for relation in relations)
            ),
            "held_out_pixel_count": int(
                sum(relation.held_out_pixels for relation in relations)
            ),
            "global_log_gain_residual_max": float(np.max(np.abs(residual))),
            "global_gain_min": float(np.min(gains)),
            "global_gain_max": float(np.max(gains)),
            "global_bias_abs_max": float(np.max(np.abs(biases))),
            "held_out_pair_residual_p95_max_linear": float(
                max(relation.held_out_residual_p95 for relation in relations)
            ),
            "held_out_pair_residual_max_linear": float(
                max(relation.held_out_residual_max for relation in relations)
            ),
            "corrections": [correction.as_dict() for correction in corrections],
            "pairs": pair_audit,
        },
    )


def apply_video_photometric_correction(
    bgra: np.ndarray,
    correction: VideoPhotometricCorrection,
) -> np.ndarray:
    """Apply one accepted correction, preserving BGRA alpha exactly.

    This primitive intentionally validates the same limits as the solver.  It
    does not inspect an acceptance flag because a correction carries no mutable
    provenance; callers must gate it with :attr:`GlobalPhotometricResult.accepted`.
    """

    image, _mask = _validated_bgra_and_mask(
        bgra, np.ones(np.asarray(bgra).shape[:2], dtype=bool), name="BGRA"
    )
    gain = np.asarray(correction.gain_bgr, dtype=np.float64)
    bias = np.asarray(correction.bias_bgr, dtype=np.float64)
    if gain.shape != (3,) or bias.shape != (3,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError("Photometric correction must contain finite BGR triplets")
    if np.any(gain < 0.55) or np.any(gain > 2.00) or np.any(np.abs(bias) > 0.18):
        raise ValueError("Photometric correction is outside fail-closed limits")
    output = image.copy()
    linear = _srgb_to_linear_bgr(image[:, :, :3])
    output[:, :, :3] = _linear_to_srgb_bgr(
        linear * gain.reshape(1, 1, 3).astype(np.float32)
        + bias.reshape(1, 1, 3).astype(np.float32)
    )
    return output
