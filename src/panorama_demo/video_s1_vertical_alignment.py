"""Local vertical-only alignment for the isolated S0/S1 video experiment.

The module deliberately has no knowledge of poses, depth, seams, or production
renderers.  It consumes already co-registered RGB overlap crops, estimates a
bounded ``dy(y)`` curve, and returns evidence which a caller may compose into
one final inverse map.  Failure is represented by missing observations and a
zero curve; it is never promoted to a pair-level rendering exception.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class VerticalAlignmentConfig:
    horizontal_scale: float = 0.5
    preserve_full_vertical_resolution: bool = True
    window_height_px: int = 32
    window_step_px: int = 16
    horizontal_subwindow_width_px: int = 80
    horizontal_subwindow_count: int = 3
    minimum_horizontal_subwindow_width_px: int = 48
    search_radius_y_px: int = 16
    top_k_candidates: int = 5
    candidate_nms_radius_px: int = 2
    allow_subpixel_refinement: bool = True
    luminance_zncc_weight: float = 0.45
    vertical_gradient_zncc_weight: float = 0.40
    lab_color_weight: float = 0.15
    minimum_valid_fraction_for_candidate: float = 0.45
    minimum_2d_texture_score: float = 0.05
    left_right_sigma_px: float = 1.5
    candidate_margin_sigma: float = 0.08
    multi_x_consensus_sigma_px: float = 1.5
    use_ransac_soft_prior: bool = True
    ransac_minimum_points: int = 6
    ransac_residual_px: float = 2.0
    dp_first_order_weight: float = 1.0
    dp_second_order_weight: float = 0.5
    dp_missing_cost: float = 1.25
    dp_soft_prior_weight: float = 0.25
    smoothing_control_spacing_px: int = 80
    smoothing_data_weight: float = 1.0
    smoothing_first_order_weight: float = 0.25
    smoothing_second_order_weight: float = 2.0
    maximum_local_slope: float = 0.08
    gain_candidates: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    gain_block_height_px: int = 64
    gain_smoothness_weight: float = 0.20
    # S1.1 opt-in controls.  ``None`` preserves the S01/v1 single-stage
    # search exactly; the S011 configuration supplies all of these values.
    normal_coarse_search_radius_y_px: int | None = None
    expanded_coarse_search_radius_y_px: int | None = None
    coarse_window_height_px: int = 32
    coarse_window_step_px: int = 16
    fine_window_height_px: int = 24
    fine_window_step_px: int = 8
    fine_search_radius_y_px: float = 2.0
    fine_search_step_px: float = 0.25
    normal_top_k_candidates: int = 3
    expanded_top_k_candidates: int = 5
    minimum_independent_subwindow_agreement: int = 2
    maximum_subwindow_spread_px: float = 0.75
    trusted_row_taper_px: int = 8
    maximum_consecutive_missing_windows: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "VerticalAlignmentConfig":
        if value is None:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown S1 vertical-alignment keys: {unknown}")
        normalized = dict(value)
        if "gain_candidates" in normalized:
            normalized["gain_candidates"] = tuple(float(item) for item in normalized["gain_candidates"])
        result = cls(**normalized)
        result.validate()
        return result

    def validate(self) -> None:
        if not 0.0 < self.horizontal_scale <= 1.0:
            raise ValueError("S1 horizontal_scale must be in (0, 1]")
        if not self.preserve_full_vertical_resolution:
            raise ValueError("S1 must preserve full vertical resolution")
        if self.window_height_px < 8 or not 1 <= self.window_step_px < self.window_height_px:
            raise ValueError("S1 windows must be overlapping and at least 8 px high")
        if self.search_radius_y_px < 1 or self.search_radius_y_px > 24:
            raise ValueError("S1 vertical search radius must be in [1, 24]")
        if self.top_k_candidates < 1 or self.candidate_nms_radius_px < 0:
            raise ValueError("S1 Top-K/NMS configuration is invalid")
        if self.normal_coarse_search_radius_y_px is not None:
            if not 1 <= self.normal_coarse_search_radius_y_px <= 16:
                raise ValueError("S1.1 normal coarse search radius must be in [1, 16]")
            if self.expanded_coarse_search_radius_y_px is None or not (
                self.normal_coarse_search_radius_y_px
                <= self.expanded_coarse_search_radius_y_px
                <= 24
            ):
                raise ValueError("S1.1 expanded coarse search radius is invalid")
            if not 0.0 < self.fine_search_step_px <= self.fine_search_radius_y_px:
                raise ValueError("S1.1 fine search step/radius is invalid")
            if self.minimum_independent_subwindow_agreement < 2:
                raise ValueError("S1.1 requires at least two agreeing subwindows")
            if not 0.0 < self.maximum_subwindow_spread_px <= 2.0:
                raise ValueError("S1.1 subwindow spread limit is invalid")
            if self.trusted_row_taper_px < 1 or self.maximum_consecutive_missing_windows < 0:
                raise ValueError("S1.1 trusted-row configuration is invalid")
        if self.horizontal_subwindow_count < 1 or self.minimum_horizontal_subwindow_width_px < 4:
            raise ValueError("S1 horizontal subwindow configuration is invalid")
        if not 0.0 < self.minimum_valid_fraction_for_candidate <= 1.0:
            raise ValueError("S1 minimum valid fraction must be in (0, 1]")
        weights = (
            self.luminance_zncc_weight,
            self.vertical_gradient_zncc_weight,
            self.lab_color_weight,
        )
        if min(weights) < 0.0 or not np.isclose(sum(weights), 1.0):
            raise ValueError("S1 matching weights must be non-negative and sum to one")
        if self.left_right_sigma_px <= 0.0 or self.multi_x_consensus_sigma_px <= 0.0:
            raise ValueError("S1 confidence sigmas must be positive")
        if self.ransac_minimum_points < 2 or self.ransac_residual_px <= 0.0:
            raise ValueError("S1 RANSAC configuration is invalid")
        if self.smoothing_control_spacing_px < 4 or not 0.0 < self.maximum_local_slope < 1.0:
            raise ValueError("S1 curve smoothing configuration is invalid")
        if self.gain_block_height_px < 4 or not self.gain_candidates:
            raise ValueError("S1 gain configuration is invalid")
        gains = np.asarray(self.gain_candidates, dtype=np.float64)
        if not np.isfinite(gains).all() or np.any((gains < 0.0) | (gains > 1.0)):
            raise ValueError("S1 gains must be finite values in [0, 1]")


def _coerce_config(value: object | None) -> VerticalAlignmentConfig:
    """Accept the experiment's wider S1 config without importing it here."""

    if value is None:
        result = VerticalAlignmentConfig()
    elif isinstance(value, VerticalAlignmentConfig):
        result = value
    elif isinstance(value, Mapping):
        result = VerticalAlignmentConfig.from_mapping(value)
    else:
        result = VerticalAlignmentConfig(
            **{
                name: getattr(value, name)
                for name in VerticalAlignmentConfig.__dataclass_fields__
                if hasattr(value, name)
            }
        )
    result.validate()
    return result


@dataclass(frozen=True)
class SourceFeatures:
    gray: np.ndarray
    lab: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    canny: np.ndarray
    structure_tensor_score: np.ndarray
    sharpness: np.ndarray
    valid_mask: np.ndarray
    horizontal_scale: float


@dataclass(frozen=True)
class VerticalCandidate:
    window_index: int
    center_y: float
    dy_px: float
    raw_cost: float
    normalized_cost: float
    luminance_cost: float
    vertical_gradient_cost: float
    color_cost: float
    valid_fraction: float
    texture_score: float
    left_right_error_px: float
    multi_x_consensus_error_px: float
    confidence: float
    is_missing: bool
    search_border_hit: bool = False


@dataclass(frozen=True)
class VerticalCurve:
    y: np.ndarray
    dy_raw: np.ndarray
    dy_smoothed: np.ndarray
    confidence: np.ndarray
    gain: np.ndarray
    dy_applied: np.ndarray
    valid_observation_mask: np.ndarray


@dataclass(frozen=True)
class RansacSoftPrior:
    available: bool
    slope: float = 0.0
    intercept: float = 0.0
    inlier_count: int = 0

    def predict(self, y: float) -> float:
        return self.slope * float(y) + self.intercept


@dataclass(frozen=True)
class VerticalAlignmentResult:
    pair_status: str
    candidates_by_window: tuple[tuple[VerticalCandidate, ...], ...]
    selected_candidates: tuple[VerticalCandidate, ...]
    ransac_prior: RansacSoftPrior
    curve: VerticalCurve
    gain_block_values: tuple[float, ...] = ()
    diagnostics: Mapping[str, object] = field(default_factory=dict)


def precompute_source_features(
    image: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    horizontal_scale: float = 0.5,
) -> SourceFeatures:
    """Compute all S1 source features once, shrinking only horizontally."""

    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("S1 source RGB must be a uint8 BGR image")
    if not 0.0 < horizontal_scale <= 1.0:
        raise ValueError("S1 horizontal feature scale must be in (0, 1]")
    height, width = image.shape[:2]
    target_width = max(1, int(round(width * horizontal_scale)))
    scaled = cv2.resize(image, (target_width, height), interpolation=cv2.INTER_AREA)
    if valid_mask is None:
        valid = np.ones((height, target_width), dtype=bool)
    else:
        source_valid = np.asarray(valid_mask, dtype=np.uint8)
        if source_valid.shape != (height, width):
            raise ValueError("S1 source valid mask shape does not match RGB")
        valid = cv2.resize(source_valid, (target_width, height), interpolation=cv2.INTER_NEAREST) > 0
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] /= 255.0
    lab[..., 1:] = (lab[..., 1:] - 128.0) / 128.0
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    canny = cv2.Canny((gray * 255.0).astype(np.uint8), 60, 150)
    tensor_xx = cv2.GaussianBlur(gradient_x * gradient_x, (5, 5), 0)
    tensor_yy = cv2.GaussianBlur(gradient_y * gradient_y, (5, 5), 0)
    tensor_xy = cv2.GaussianBlur(gradient_x * gradient_y, (5, 5), 0)
    trace = tensor_xx + tensor_yy
    discriminant = np.sqrt(np.maximum(0.0, (tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy**2))
    structure = np.maximum(0.0, 0.5 * (trace - discriminant))
    sharpness = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (5, 5), 0)
    return SourceFeatures(
        gray=np.ascontiguousarray(gray),
        lab=np.ascontiguousarray(lab),
        gradient_x=np.ascontiguousarray(gradient_x),
        gradient_y=np.ascontiguousarray(gradient_y),
        canny=np.ascontiguousarray(canny),
        structure_tensor_score=np.ascontiguousarray(structure),
        sharpness=np.ascontiguousarray(sharpness),
        valid_mask=np.ascontiguousarray(valid),
        horizontal_scale=float(horizontal_scale),
    )


def precompute_feature_cache(
    images: Mapping[int, np.ndarray],
    valid_masks: Mapping[int, np.ndarray] | None = None,
    *,
    horizontal_scale: float = 0.5,
) -> dict[int, SourceFeatures]:
    """Return a frame-keyed cache, evaluating every real source exactly once."""

    masks = valid_masks or {}
    return {
        int(frame_id): precompute_source_features(
            image, masks.get(frame_id), horizontal_scale=horizontal_scale
        )
        for frame_id, image in images.items()
    }


def _zncc_cost(first: np.ndarray, second: np.ndarray, valid: np.ndarray) -> float:
    a = np.asarray(first[valid], dtype=np.float64).reshape(-1)
    b = np.asarray(second[valid], dtype=np.float64).reshape(-1)
    if a.size < 16:
        return 1.0
    a -= np.mean(a)
    b -= np.mean(b)
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale <= 1e-9:
        return 1.0
    return float(np.clip(1.0 - np.dot(a, b) / scale, 0.0, 2.0))


def _batched_zncc_cost(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Return one masked ZNCC cost per leading ``second`` volume.

    This is algebraically the same centred dot product as :func:`_zncc_cost`,
    but reduces every vertical-shift plane together.  Accumulation remains in
    float64 so batching does not trade matching precision for speed.
    """

    reference = np.asarray(first, dtype=np.float64)[None, ...]
    samples = np.asarray(second, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if samples.ndim != reference.ndim or mask.shape != samples.shape:
        raise ValueError("S1 batched ZNCC volume shapes do not match")
    axes = tuple(range(1, samples.ndim))
    weights = mask.astype(np.float64, copy=False)
    count = np.sum(weights, axis=axes)
    sum_a = np.sum(reference * weights, axis=axes)
    sum_b = np.sum(samples * weights, axis=axes)
    sum_aa = np.sum(reference * reference * weights, axis=axes)
    sum_bb = np.sum(samples * samples * weights, axis=axes)
    sum_ab = np.sum(reference * samples * weights, axis=axes)
    safe_count = np.maximum(count, 1.0)
    covariance = sum_ab - sum_a * sum_b / safe_count
    variance_a = np.maximum(0.0, sum_aa - sum_a * sum_a / safe_count)
    variance_b = np.maximum(0.0, sum_bb - sum_b * sum_b / safe_count)
    scale = np.sqrt(variance_a * variance_b)
    costs = np.ones(count.shape, dtype=np.float64)
    usable = (count >= 16.0) & (scale > 1e-9)
    costs[usable] = np.clip(1.0 - covariance[usable] / scale[usable], 0.0, 2.0)
    return costs


def _subwindow_ranges(width: int, config: VerticalAlignmentConfig) -> tuple[tuple[int, int], ...]:
    desired = max(
        config.minimum_horizontal_subwindow_width_px,
        int(round(config.horizontal_subwindow_width_px * config.horizontal_scale)),
    )
    window_width = min(width, desired)
    if window_width < int(round(config.minimum_horizontal_subwindow_width_px * config.horizontal_scale)):
        return ()
    if config.horizontal_subwindow_count == 1 or window_width == width:
        starts = np.asarray([(width - window_width) // 2], dtype=np.int32)
    else:
        starts = np.rint(
            np.linspace(0, width - window_width, config.horizontal_subwindow_count)
        ).astype(np.int32)
    return tuple(dict.fromkeys((int(start), int(start + window_width)) for start in starts))


def _score_shift(
    left: SourceFeatures,
    right: SourceFeatures,
    *,
    y0: int,
    y1: int,
    dy: int,
    x0: int,
    x1: int,
    config: VerticalAlignmentConfig,
) -> tuple[float, float, float, float, float, float] | None:
    right_y0, right_y1 = y0 + dy, y1 + dy
    if right_y0 < 0 or right_y1 > right.gray.shape[0]:
        return None
    valid = left.valid_mask[y0:y1, x0:x1] & right.valid_mask[right_y0:right_y1, x0:x1]
    valid_fraction = float(np.mean(valid))
    if valid_fraction < config.minimum_valid_fraction_for_candidate:
        return None
    lum = _zncc_cost(left.gray[y0:y1, x0:x1], right.gray[right_y0:right_y1, x0:x1], valid)
    grad = _zncc_cost(
        left.gradient_y[y0:y1, x0:x1], right.gradient_y[right_y0:right_y1, x0:x1], valid
    )
    lab_delta = np.linalg.norm(
        left.lab[y0:y1, x0:x1] - right.lab[right_y0:right_y1, x0:x1], axis=2
    )
    color = float(np.clip(np.mean(lab_delta[valid]) / 1.5, 0.0, 2.0))
    texture = float(np.percentile(left.structure_tensor_score[y0:y1, x0:x1][valid], 75))
    texture += float(np.percentile(right.structure_tensor_score[right_y0:right_y1, x0:x1][valid], 75))
    texture = float(np.clip(texture * 20.0, 0.0, 1.0))
    total = (
        config.luminance_zncc_weight * lum
        + config.vertical_gradient_zncc_weight * grad
        + config.lab_color_weight * color
    )
    return total, lum, grad, color, valid_fraction, texture


def _integer_costs_for_window(
    left: SourceFeatures,
    right: SourceFeatures,
    y0: int,
    y1: int,
    config: VerticalAlignmentConfig,
    *,
    search_center: float = 0.0,
    search_radius: int | None = None,
) -> tuple[dict[int, tuple[float, ...]], tuple[int, ...]]:
    ranges = _subwindow_ranges(left.gray.shape[1], config)
    if not ranges:
        return {}, ()
    per_shift: dict[int, tuple[float, ...]] = {}
    per_subwindow_best: list[int] = []
    subwindow_scores: list[dict[int, tuple[float, ...]]] = []
    radius = config.search_radius_y_px if search_radius is None else int(search_radius)
    centre = int(round(search_center))
    for x0, x1 in ranges:
        scores: dict[int, tuple[float, ...]] = {}
        for dy in range(centre - radius, centre + radius + 1):
            score = _score_shift(left, right, y0=y0, y1=y1, dy=dy, x0=x0, x1=x1, config=config)
            if score is not None:
                scores[dy] = score
        if scores:
            per_subwindow_best.append(min(scores, key=lambda value: scores[value][0]))
            subwindow_scores.append(scores)
    if not subwindow_scores:
        return {}, ()
    for dy in range(centre - radius, centre + radius + 1):
        rows = [scores[dy] for scores in subwindow_scores if dy in scores]
        if not rows:
            continue
        values = np.asarray(rows, dtype=np.float64)
        per_shift[dy] = tuple(float(item) for item in np.mean(values, axis=0))
    return per_shift, tuple(per_subwindow_best)


def _score_fractional_shift(
    left: SourceFeatures,
    right: SourceFeatures,
    *,
    y0: int,
    y1: int,
    dy: float,
    x0: int,
    x1: int,
    config: VerticalAlignmentConfig,
) -> tuple[float, float, float, float, float, float] | None:
    """Score ``right[y + dy]`` without introducing a horizontal warp."""

    sample_y = np.arange(y0, y1, dtype=np.float64) + float(dy)
    inside_rows = (sample_y >= 0.0) & (sample_y <= right.gray.shape[0] - 1)
    lower = np.clip(np.floor(sample_y).astype(np.int64), 0, right.gray.shape[0] - 1)
    upper = np.clip(lower + 1, 0, right.gray.shape[0] - 1)
    fraction = np.clip(sample_y - lower, 0.0, 1.0).astype(np.float32)
    nearest = np.clip(np.rint(sample_y).astype(np.int64), 0, right.gray.shape[0] - 1)
    right_valid = right.valid_mask[nearest, x0:x1]
    valid = (
        left.valid_mask[y0:y1, x0:x1]
        & right_valid
        & inside_rows[:, None]
    )
    valid_fraction = float(np.mean(valid))
    if valid_fraction < config.minimum_valid_fraction_for_candidate:
        return None

    def interpolate(array: np.ndarray) -> np.ndarray:
        before = array[lower, x0:x1]
        after = array[upper, x0:x1]
        weights = fraction[:, None]
        if array.ndim == 3:
            weights = weights[:, :, None]
        return before * (1.0 - weights) + after * weights

    right_gray = interpolate(right.gray)
    right_gradient = interpolate(right.gradient_y)
    right_lab = interpolate(right.lab)
    right_structure = interpolate(right.structure_tensor_score)
    lum = _zncc_cost(left.gray[y0:y1, x0:x1], right_gray, valid)
    grad = _zncc_cost(left.gradient_y[y0:y1, x0:x1], right_gradient, valid)
    lab_delta = np.linalg.norm(left.lab[y0:y1, x0:x1] - right_lab, axis=2)
    color = float(np.clip(np.mean(lab_delta[valid]) / 1.5, 0.0, 2.0))
    texture = float(np.percentile(left.structure_tensor_score[y0:y1, x0:x1][valid], 75))
    texture += float(np.percentile(right_structure[valid], 75))
    texture = float(np.clip(texture * 20.0, 0.0, 1.0))
    total = (
        config.luminance_zncc_weight * lum
        + config.vertical_gradient_zncc_weight * grad
        + config.lab_color_weight * color
    )
    return total, lum, grad, color, valid_fraction, texture


def _fine_candidates_consensus(
    left: SourceFeatures,
    right: SourceFeatures,
    *,
    y0: int,
    y1: int,
    coarse_dys: Sequence[int],
    config: VerticalAlignmentConfig,
) -> dict[int, tuple[float, tuple[float, ...], tuple[float, ...]]]:
    """Refine every coarse Top-K candidate using shared forward/reverse volumes."""

    ranges = _subwindow_ranges(left.gray.shape[1], config)
    residuals = np.arange(
        -config.fine_search_radius_y_px,
        config.fine_search_radius_y_px + 0.5 * config.fine_search_step_px,
        config.fine_search_step_px,
        dtype=np.float64,
    )

    def refined_minimum(scored: list[tuple[float, float]]) -> tuple[float, float]:
        """Return cost/dy with a bounded three-point parabolic refinement."""

        ordered_scores = sorted(scored, key=lambda item: item[1])
        best_index = min(range(len(ordered_scores)), key=lambda index: ordered_scores[index][0])
        cost, dy = ordered_scores[best_index]
        if 0 < best_index < len(ordered_scores) - 1:
            before = ordered_scores[best_index - 1][0]
            after = ordered_scores[best_index + 1][0]
            denominator = before - 2.0 * cost + after
            if denominator > 1e-9:
                fraction = np.clip(0.5 * (before - after) / denominator, -1.0, 1.0)
                dy += float(fraction * config.fine_search_step_px)
        return cost, dy

    def score_grid(
        source: SourceFeatures,
        target: SourceFeatures,
        x0: int,
        x1: int,
        candidate_dy: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Batch a quarter-pixel vertical cost volume for one subwindow."""

        rows = np.arange(y0, y1, dtype=np.float64)[None, :]
        sample_y = rows + candidate_dy[:, None]
        inside = (sample_y >= 0.0) & (sample_y <= target.gray.shape[0] - 1)
        lower = np.clip(
            np.floor(sample_y).astype(np.int64), 0, target.gray.shape[0] - 1
        )
        upper = np.clip(lower + 1, 0, target.gray.shape[0] - 1)
        fraction = np.clip(sample_y - lower, 0.0, 1.0).astype(np.float32)
        columns = np.arange(x0, x1, dtype=np.int64)[None, None, :]

        def sample(array: np.ndarray) -> np.ndarray:
            before = array[lower[:, :, None], columns]
            after = array[upper[:, :, None], columns]
            weights = fraction[:, :, None]
            return before * (1.0 - weights) + after * weights

        gray_volume = sample(target.gray)
        gradient_volume = sample(target.gradient_y)
        nearest = np.clip(
            np.rint(sample_y).astype(np.int64), 0, target.gray.shape[0] - 1
        )
        valid_volume = (
            target.valid_mask[nearest[:, :, None], columns]
            & source.valid_mask[y0:y1, x0:x1][None, :, :]
            & inside[:, :, None]
        )
        source_gray = source.gray[y0:y1, x0:x1]
        source_gradient = source.gradient_y[y0:y1, x0:x1]
        valid_fraction = np.mean(valid_volume, axis=(1, 2))
        usable = valid_fraction >= config.minimum_valid_fraction_for_candidate
        if not np.any(usable):
            return []
        lum = _batched_zncc_cost(source_gray, gray_volume, valid_volume)
        grad = _batched_zncc_cost(source_gradient, gradient_volume, valid_volume)
        # Coarse ranking already audited Lab/color. Fine refinement only
        # resolves the vertical residual and therefore batches the two
        # structure-bearing terms.
        weight = config.luminance_zncc_weight + config.vertical_gradient_zncc_weight
        costs = (
            config.luminance_zncc_weight * lum
            + config.vertical_gradient_zncc_weight * grad
        ) / max(weight, 1e-9)
        indices = np.flatnonzero(usable)
        return [
            (float(costs[index]), float(candidate_dy[index]))
            for index in indices
        ]

    required = min(config.minimum_independent_subwindow_agreement, len(ranges))
    if not coarse_dys or required < 1:
        return {}

    # All Top-K hypotheses share most of the same quarter-pixel samples.  Build
    # one union volume per independent x subwindow instead of resampling the
    # images once per hypothesis.
    forward_grids = {
        int(coarse): np.asarray(float(coarse) + residuals, dtype=np.float64)
        for coarse in coarse_dys
    }
    union_forward = np.unique(np.concatenate(tuple(forward_grids.values())))
    forward_scores = [
        score_grid(left, right, x0, x1, union_forward) for x0, x1 in ranges
    ]
    forward_lookup = [
        {round(dy, 8): cost for cost, dy in scored} for scored in forward_scores
    ]

    forward_results: dict[int, tuple[float, tuple[float, ...]]] = {}
    for coarse, grid in forward_grids.items():
        per_subwindow: list[float] = []
        for lookup in forward_lookup:
            scored = [
                (lookup[round(float(dy), 8)], float(dy))
                for dy in grid
                if round(float(dy), 8) in lookup
            ]
            if scored:
                per_subwindow.append(refined_minimum(scored)[1])
        if len(per_subwindow) < required:
            continue
        ordered = np.sort(np.asarray(per_subwindow, dtype=np.float64))
        clusters = [ordered[i : i + required] for i in range(len(ordered) - required + 1)]
        agreed = min(clusters, key=lambda values: float(np.ptp(values)))
        if float(np.ptp(agreed)) <= config.maximum_subwindow_spread_px:
            forward_results[coarse] = (float(np.median(agreed)), tuple(per_subwindow))

    if not forward_results:
        return {}
    reverse_grids = {
        coarse: np.asarray(-refined + residuals, dtype=np.float64)
        for coarse, (refined, _) in forward_results.items()
    }
    union_reverse = np.unique(np.concatenate(tuple(reverse_grids.values())))
    reverse_scores = [
        score_grid(right, left, x0, x1, union_reverse) for x0, x1 in ranges
    ]
    reverse_lookup = [
        {round(dy, 8): cost for cost, dy in scored} for scored in reverse_scores
    ]
    result: dict[int, tuple[float, tuple[float, ...], tuple[float, ...]]] = {}
    for coarse, grid in reverse_grids.items():
        reverse_best: list[float] = []
        for lookup in reverse_lookup:
            scored = [
                (lookup[round(float(dy), 8)], float(dy))
                for dy in grid
                if round(float(dy), 8) in lookup
            ]
            if scored:
                reverse_best.append(refined_minimum(scored)[1])
        if len(reverse_best) >= required:
            refined, forward_best = forward_results[coarse]
            result[coarse] = (refined, forward_best, tuple(reverse_best))
    return result


def _refined_dy(costs: Mapping[int, tuple[float, ...]], dy: int) -> float:
    if dy - 1 not in costs or dy + 1 not in costs:
        return float(dy)
    before, centre, after = costs[dy - 1][0], costs[dy][0], costs[dy + 1][0]
    denominator = before - 2.0 * centre + after
    if denominator <= 1e-9:
        return float(dy)
    return float(dy + np.clip(0.5 * (before - after) / denominator, -0.75, 0.75))


def _missing_candidate(window_index: int, center_y: float, cost: float) -> VerticalCandidate:
    return VerticalCandidate(
        window_index=window_index,
        center_y=center_y,
        dy_px=0.0,
        raw_cost=cost,
        normalized_cost=cost,
        luminance_cost=1.0,
        vertical_gradient_cost=1.0,
        color_cost=1.0,
        valid_fraction=0.0,
        texture_score=0.0,
        left_right_error_px=float("inf"),
        multi_x_consensus_error_px=float("inf"),
        confidence=0.0,
        is_missing=True,
    )


def _generate_s11_vertical_candidates(
    left: SourceFeatures,
    right: SourceFeatures,
    settings: VerticalAlignmentConfig,
    *,
    risk_pair: bool,
    motion_dy_px: float,
) -> tuple[tuple[VerticalCandidate, ...], ...]:
    height = left.gray.shape[0]
    coarse_height = min(height, settings.coarse_window_height_px)
    coarse_starts = list(
        range(0, max(1, height - coarse_height + 1), settings.coarse_window_step_px)
    )
    final_coarse_start = max(0, height - coarse_height)
    if not coarse_starts or coarse_starts[-1] != final_coarse_start:
        coarse_starts.append(final_coarse_start)
    normal_radius = int(settings.normal_coarse_search_radius_y_px or 6)
    expanded_radius = int(settings.expanded_coarse_search_radius_y_px or 16)
    coarse_records: list[tuple[float, dict[int, tuple[float, ...]], tuple[int, ...], int]] = []
    for y0 in coarse_starts:
        y1 = min(height, y0 + coarse_height)
        radius = expanded_radius if risk_pair else normal_radius
        top_k = settings.expanded_top_k_candidates if risk_pair else settings.normal_top_k_candidates
        costs, _ = _integer_costs_for_window(
            left,
            right,
            y0,
            y1,
            settings,
            search_center=motion_dy_px,
            search_radius=radius,
        )
        ordered = sorted(costs, key=lambda value: costs[value][0])
        # A normal search touching either edge is inconclusive; rerun just this
        # height window with the expanded S1.1 envelope.
        if ordered and not risk_pair and abs(ordered[0] - round(motion_dy_px)) >= radius:
            radius = expanded_radius
            top_k = settings.expanded_top_k_candidates
            costs, _ = _integer_costs_for_window(
                left,
                right,
                y0,
                y1,
                settings,
                search_center=motion_dy_px,
                search_radius=radius,
            )
            ordered = sorted(costs, key=lambda value: costs[value][0])
        selected: list[int] = []
        for dy in ordered:
            if all(abs(dy - prior) > settings.candidate_nms_radius_px for prior in selected):
                selected.append(dy)
            if len(selected) == top_k:
                break
        coarse_records.append((0.5 * (y0 + y1 - 1), costs, tuple(selected), radius))

    fine_height = min(height, settings.fine_window_height_px)
    fine_starts = list(
        range(0, max(1, height - fine_height + 1), settings.fine_window_step_px)
    )
    final_fine_start = max(0, height - fine_height)
    if not fine_starts or fine_starts[-1] != final_fine_start:
        fine_starts.append(final_fine_start)
    fine_records = []
    for fine_y0 in fine_starts:
        fine_y1 = min(height, fine_y0 + fine_height)
        fine_center = 0.5 * (fine_y0 + fine_y1 - 1)
        fine_records.append(
            min(coarse_records, key=lambda record: abs(record[0] - fine_center))
        )

    def refine_window(item: tuple[int, tuple[float, dict[int, tuple[float, ...]], tuple[int, ...], int]]):
        fine_y0, record = item
        return _fine_candidates_consensus(
            left,
            right,
            y0=fine_y0,
            y1=min(height, fine_y0 + fine_height),
            coarse_dys=record[2],
            config=settings,
        )

    # NumPy releases the GIL while sampling and reducing each independent
    # height cost volume.  A small fixed pool preserves deterministic window
    # order while avoiding one Python dispatch chain per fine window.
    with ThreadPoolExecutor(max_workers=min(4, len(fine_starts))) as executor:
        refined_by_window = tuple(
            executor.map(refine_window, zip(fine_starts, fine_records, strict=True))
        )
    result: list[tuple[VerticalCandidate, ...]] = []
    for window_index, fine_y0 in enumerate(fine_starts):
        fine_y1 = min(height, fine_y0 + fine_height)
        fine_center = 0.5 * (fine_y0 + fine_y1 - 1)
        _, costs, selected, radius = fine_records[window_index]
        ordered = sorted(costs, key=lambda value: costs[value][0])
        best_cost = costs[ordered[0]][0] if ordered else 2.0
        second_cost = costs[ordered[1]][0] if len(ordered) > 1 else best_cost + 1.0
        uniqueness = 1.0 - np.exp(
            -max(0.0, second_cost - best_cost) / settings.candidate_margin_sigma
        )
        candidates: list[VerticalCandidate] = []
        refined_candidates = refined_by_window[window_index]
        for dy in selected:
            consensus = refined_candidates.get(dy)
            if consensus is None:
                continue
            refined, subwindow_best, reverse_best = consensus
            total, lum, grad, color, valid_fraction, texture = costs[dy]
            reverse_array = np.asarray(reverse_best, dtype=np.float64)
            lr_error = float(np.median(np.abs(refined + reverse_array)))
            agreeing = np.sort(np.asarray(subwindow_best, dtype=np.float64))
            required = min(settings.minimum_independent_subwindow_agreement, len(agreeing))
            spreads = [
                agreeing[i : i + required]
                for i in range(len(agreeing) - required + 1)
            ]
            consensus_error = float(np.ptp(min(spreads, key=lambda values: float(np.ptp(values)))))
            confidence = float(
                np.exp(-max(0.0, total))
                * np.exp(-(lr_error / settings.left_right_sigma_px) ** 2)
                * np.exp(-(consensus_error / settings.multi_x_consensus_sigma_px) ** 2)
                * (0.25 + 0.75 * texture)
                * (0.25 + 0.75 * uniqueness)
            )
            border = abs(dy - round(motion_dy_px)) >= radius
            # Expanded-border and failed candidate-specific FB observations
            # remain explicit missing states rather than weak corrections.
            if border or lr_error > settings.maximum_subwindow_spread_px:
                continue
            candidates.append(
                VerticalCandidate(
                    window_index=window_index,
                    center_y=fine_center,
                    dy_px=refined,
                    raw_cost=float(total),
                    normalized_cost=float(np.clip(total / 2.0, 0.0, 1.0)),
                    luminance_cost=float(lum),
                    vertical_gradient_cost=float(grad),
                    color_cost=float(color),
                    valid_fraction=float(valid_fraction),
                    texture_score=float(texture),
                    left_right_error_px=lr_error,
                    multi_x_consensus_error_px=consensus_error,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    is_missing=False,
                    search_border_hit=False,
                )
            )
        candidates.append(
            _missing_candidate(window_index, fine_center, settings.dp_missing_cost)
        )
        result.append(tuple(candidates))
    return tuple(result)


def generate_vertical_candidates(
    left: SourceFeatures,
    right: SourceFeatures,
    config: object | None = None,
    *,
    risk_pair: bool = False,
    motion_dy_px: float = 0.0,
) -> tuple[tuple[VerticalCandidate, ...], ...]:
    """Generate per-height Top-K candidates plus an explicit missing state.

    ``dy`` is the right-image row sampled for a left-image row: a feature at
    ``left[y]`` found at ``right[y + 2]`` is reported as ``dy=+2``.
    """

    settings = _coerce_config(config)
    if left.gray.shape != right.gray.shape:
        raise ValueError("S1 pair features must share one overlap shape")
    if settings.normal_coarse_search_radius_y_px is not None:
        if not np.isfinite(motion_dy_px):
            raise ValueError("S1.1 motion dy must be finite")
        return _generate_s11_vertical_candidates(
            left, right, settings, risk_pair=bool(risk_pair), motion_dy_px=float(motion_dy_px)
        )
    height = left.gray.shape[0]
    starts = list(range(0, max(1, height - settings.window_height_px + 1), settings.window_step_px))
    final_start = max(0, height - settings.window_height_px)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    result: list[tuple[VerticalCandidate, ...]] = []
    for window_index, y0 in enumerate(starts):
        y1 = min(height, y0 + settings.window_height_px)
        costs, local_best = _integer_costs_for_window(left, right, y0, y1, settings)
        reverse_costs, _ = _integer_costs_for_window(right, left, y0, y1, settings)
        ordered = sorted(costs, key=lambda value: costs[value][0])
        selected: list[int] = []
        for dy in ordered:
            if all(abs(dy - prior) > settings.candidate_nms_radius_px for prior in selected):
                selected.append(dy)
            if len(selected) == settings.top_k_candidates:
                break
        candidates: list[VerticalCandidate] = []
        best_cost = costs[ordered[0]][0] if ordered else 2.0
        second_cost = costs[ordered[1]][0] if len(ordered) > 1 else best_cost + 1.0
        uniqueness = 1.0 - np.exp(-max(0.0, second_cost - best_cost) / settings.candidate_margin_sigma)
        reverse_best = min(reverse_costs, key=lambda value: reverse_costs[value][0]) if reverse_costs else 0
        for dy in selected:
            total, lum, grad, color, valid_fraction, texture = costs[dy]
            refined = _refined_dy(costs, dy) if settings.allow_subpixel_refinement else float(dy)
            lr_error = abs(refined + float(reverse_best))
            consensus_error = (
                float(np.sqrt(np.mean((np.asarray(local_best, dtype=np.float64) - refined) ** 2)))
                if local_best
                else float(settings.search_radius_y_px)
            )
            confidence = float(
                np.exp(-max(0.0, total))
                * np.exp(-(lr_error / settings.left_right_sigma_px) ** 2)
                * np.exp(-(consensus_error / settings.multi_x_consensus_sigma_px) ** 2)
                * (0.25 + 0.75 * texture)
                * (0.25 + 0.75 * uniqueness)
            )
            border = abs(dy) == settings.search_radius_y_px
            if border:
                confidence *= 0.35
            candidates.append(
                VerticalCandidate(
                    window_index=window_index,
                    center_y=0.5 * (y0 + y1 - 1),
                    dy_px=refined,
                    raw_cost=float(total),
                    normalized_cost=float(np.clip(total / 2.0, 0.0, 1.0)),
                    luminance_cost=float(lum),
                    vertical_gradient_cost=float(grad),
                    color_cost=float(color),
                    valid_fraction=float(valid_fraction),
                    texture_score=float(texture),
                    left_right_error_px=float(lr_error),
                    multi_x_consensus_error_px=float(consensus_error),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    is_missing=False,
                    search_border_hit=border,
                )
            )
        candidates.append(
            _missing_candidate(window_index, 0.5 * (y0 + y1 - 1), settings.dp_missing_cost)
        )
        result.append(tuple(candidates))
    return tuple(result)


def fit_ransac_soft_prior(
    candidates_by_window: Sequence[Sequence[VerticalCandidate]],
    config: object | None = None,
) -> RansacSoftPrior:
    settings = _coerce_config(config)
    if not settings.use_ransac_soft_prior:
        return RansacSoftPrior(False)
    observations = [
        max((item for item in window if not item.is_missing), key=lambda item: item.confidence)
        for window in candidates_by_window
        if any(not item.is_missing for item in window)
    ]
    observations = [item for item in observations if item.confidence >= 0.05]
    if len(observations) < settings.ransac_minimum_points:
        return RansacSoftPrior(False)
    y = np.asarray([item.center_y for item in observations], dtype=np.float64)
    dy = np.asarray([item.dy_px for item in observations], dtype=np.float64)
    best: tuple[int, float, float, np.ndarray] | None = None
    for first in range(len(y) - 1):
        for second in range(first + 1, len(y)):
            if abs(y[second] - y[first]) < 1e-6:
                continue
            slope = (dy[second] - dy[first]) / (y[second] - y[first])
            intercept = dy[first] - slope * y[first]
            residual = np.abs(dy - (slope * y + intercept))
            inliers = residual <= settings.ransac_residual_px
            score = (int(np.sum(inliers)), -float(np.sum(np.minimum(residual, settings.ransac_residual_px))))
            if best is None or score > (best[0], best[1]):
                best = (score[0], score[1], slope, inliers)
    if best is None or best[0] < settings.ransac_minimum_points:
        return RansacSoftPrior(False)
    inliers = best[3]
    design = np.column_stack((y[inliers], np.ones(int(np.sum(inliers)))))
    slope, intercept = np.linalg.lstsq(design, dy[inliers], rcond=None)[0]
    if not np.isfinite((slope, intercept)).all():
        return RansacSoftPrior(False)
    return RansacSoftPrior(True, float(slope), float(intercept), int(np.sum(inliers)))


def _huber(value: float, delta: float = 1.0) -> float:
    absolute = abs(float(value))
    return 0.5 * absolute * absolute if absolute <= delta else delta * (absolute - 0.5 * delta)


def solve_candidate_path(
    candidates_by_window: Sequence[Sequence[VerticalCandidate]],
    prior: RansacSoftPrior | None = None,
    config: object | None = None,
) -> tuple[VerticalCandidate, ...]:
    """Select a continuous ordered path using first/second-order dynamic programming."""

    settings = _coerce_config(config)
    windows = tuple(tuple(window) for window in candidates_by_window)
    if not windows or any(not window for window in windows):
        return ()
    soft_prior = prior or RansacSoftPrior(False)

    def local(item: VerticalCandidate) -> float:
        if item.is_missing:
            return settings.dp_missing_cost
        value = item.normalized_cost + 0.35 * (1.0 - item.confidence)
        if soft_prior.available:
            value += settings.dp_soft_prior_weight * _huber(item.dy_px - soft_prior.predict(item.center_y), 2.0)
        return float(value)

    if len(windows) == 1:
        return (min(windows[0], key=local),)
    states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
    for first_index, first in enumerate(windows[0]):
        for second_index, second in enumerate(windows[1]):
            if not first.is_missing and not second.is_missing and second.center_y + second.dy_px <= first.center_y + first.dy_px:
                continue
            smooth = 0.0 if first.is_missing or second.is_missing else settings.dp_first_order_weight * _huber(second.dy_px - first.dy_px)
            states[(first_index, second_index)] = (local(first) + local(second) + smooth, (first_index, second_index))
    for window_index in range(2, len(windows)):
        next_states: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for (older_index, previous_index), (cost, path) in states.items():
            older = windows[window_index - 2][older_index]
            previous = windows[window_index - 1][previous_index]
            for current_index, current in enumerate(windows[window_index]):
                if not previous.is_missing and not current.is_missing and current.center_y + current.dy_px <= previous.center_y + previous.dy_px:
                    continue
                transition = 0.0
                if not previous.is_missing and not current.is_missing:
                    transition += settings.dp_first_order_weight * _huber(current.dy_px - previous.dy_px)
                if not older.is_missing and not previous.is_missing and not current.is_missing:
                    transition += settings.dp_second_order_weight * _huber(
                        current.dy_px - 2.0 * previous.dy_px + older.dy_px
                    )
                key = (previous_index, current_index)
                proposal = (cost + local(current) + transition, (*path, current_index))
                if key not in next_states or proposal[0] < next_states[key][0]:
                    next_states[key] = proposal
        states = next_states
        if not states:
            return tuple(
                next(item for item in window if item.is_missing) for window in windows
            )
    if not states:
        return tuple(next(item for item in window if item.is_missing) for window in windows)
    indices = min(states.values(), key=lambda item: item[0])[1]
    return tuple(windows[index][candidate_index] for index, candidate_index in enumerate(indices))


def fit_smooth_vertical_curve(
    selected: Sequence[VerticalCandidate],
    height: int,
    config: object | None = None,
) -> VerticalCurve:
    settings = _coerce_config(config)
    if settings.normal_coarse_search_radius_y_px is not None:
        return _fit_s11_trusted_curve(selected, height, settings)
    rows = np.arange(height, dtype=np.float64)
    observed = [item for item in selected if not item.is_missing and item.confidence > 0.0]
    if height < 1 or not observed:
        zeros = np.zeros(max(0, height), dtype=np.float32)
        return VerticalCurve(rows.astype(np.float32), zeros, zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy(), np.zeros(max(0, height), dtype=bool))
    spacing = settings.smoothing_control_spacing_px
    controls = np.arange(0, height, spacing, dtype=np.float64)
    if controls[-1] != height - 1:
        controls = np.append(controls, height - 1)
    observation_y = np.asarray([item.center_y for item in observed], dtype=np.float64)
    observation_dy = np.asarray([item.dy_px for item in observed], dtype=np.float64)
    weights = np.asarray([item.confidence for item in observed], dtype=np.float64)
    design = np.zeros((len(observed), len(controls)), dtype=np.float64)
    for row_index, value in enumerate(observation_y):
        upper = min(len(controls) - 1, int(np.searchsorted(controls, value, side="right")))
        lower = max(0, upper - 1)
        if upper == lower:
            design[row_index, lower] = 1.0
        else:
            fraction = (value - controls[lower]) / (controls[upper] - controls[lower])
            design[row_index, lower] = 1.0 - fraction
            design[row_index, upper] = fraction
    d1 = np.diff(np.eye(len(controls)), axis=0)
    d2 = np.diff(np.eye(len(controls)), n=2, axis=0)
    weighted = design * weights[:, None]
    matrix = settings.smoothing_data_weight * design.T @ weighted
    matrix += settings.smoothing_first_order_weight * d1.T @ d1
    if len(controls) > 2:
        matrix += settings.smoothing_second_order_weight * d2.T @ d2
    # Weak zero anchors prevent long unobserved tails from extrapolating.
    distance = np.min(np.abs(controls[:, None] - observation_y[None, :]), axis=1)
    zero_weight = np.clip((distance - spacing) / max(1.0, 2.0 * spacing), 0.0, 1.0)
    matrix += np.diag(0.5 * zero_weight + 1e-6)
    target = settings.smoothing_data_weight * design.T @ (weights * observation_dy)
    control_dy = np.linalg.solve(matrix, target)
    smoothed = np.interp(rows, controls, control_dy)
    # Enforce monotonic y -> y+dy without rejecting the full curve.
    maximum_delta = settings.maximum_local_slope
    forward = np.empty_like(smoothed)
    forward[0] = smoothed[0]
    for index in range(1, height):
        forward[index] = forward[index - 1] + np.clip(smoothed[index] - forward[index - 1], -maximum_delta, maximum_delta)
    backward = np.empty_like(smoothed)
    backward[-1] = smoothed[-1]
    for index in range(height - 2, -1, -1):
        backward[index] = backward[index + 1] + np.clip(smoothed[index] - backward[index + 1], -maximum_delta, maximum_delta)
    smoothed = 0.5 * (forward + backward)
    raw = np.interp(rows, observation_y, observation_dy, left=0.0, right=0.0)
    confidence = np.interp(rows, observation_y, weights, left=0.0, right=0.0)
    nearest = np.min(np.abs(rows[:, None] - observation_y[None, :]), axis=1)
    confidence *= np.clip(1.0 - nearest / max(1.0, 3.0 * spacing), 0.0, 1.0)
    valid = nearest <= 0.5 * settings.window_height_px
    ones = np.ones(height, dtype=np.float32)
    return VerticalCurve(
        y=rows.astype(np.float32),
        dy_raw=raw.astype(np.float32),
        dy_smoothed=smoothed.astype(np.float32),
        confidence=confidence.astype(np.float32),
        gain=ones,
        dy_applied=smoothed.astype(np.float32),
        valid_observation_mask=valid,
    )


def _fit_s11_trusted_curve(
    selected: Sequence[VerticalCandidate],
    height: int,
    settings: VerticalAlignmentConfig,
) -> VerticalCurve:
    """Fit independent trusted segments; missing cores are exact identity."""

    rows = np.arange(max(0, height), dtype=np.float64)
    zeros = np.zeros(max(0, height), dtype=np.float32)
    observed = sorted(
        (item for item in selected if not item.is_missing and item.confidence > 0.0),
        key=lambda item: item.window_index,
    )
    # A lone height observation cannot establish a vertical segment.
    if height < 1 or len(observed) < 2:
        return VerticalCurve(
            rows.astype(np.float32), zeros, zeros.copy(), zeros.copy(), zeros.copy(),
            zeros.copy(), np.zeros(max(0, height), dtype=bool),
        )
    segments: list[list[VerticalCandidate]] = []
    for item in observed:
        if not segments or item.window_index != segments[-1][-1].window_index + 1:
            segments.append([item])
        else:
            segments[-1].append(item)
    segments = [segment for segment in segments if len(segment) >= 2]
    if not segments:
        return VerticalCurve(
            rows.astype(np.float32), zeros, zeros.copy(), zeros.copy(), zeros.copy(),
            zeros.copy(), np.zeros(max(0, height), dtype=bool),
        )

    raw = np.zeros(height, dtype=np.float64)
    smoothed = np.zeros(height, dtype=np.float64)
    confidence = np.zeros(height, dtype=np.float64)
    trusted = np.zeros(height, dtype=bool)
    taper = settings.trusted_row_taper_px
    for segment in segments:
        ys = np.asarray([item.center_y for item in segment], dtype=np.float64)
        dys = np.asarray([item.dy_px for item in segment], dtype=np.float64)
        weights = np.asarray([item.confidence for item in segment], dtype=np.float64)
        core_start = max(0, int(np.floor(ys[0] - 0.5 * settings.fine_window_step_px)))
        core_stop = min(height, int(np.ceil(ys[-1] + 0.5 * settings.fine_window_step_px)) + 1)
        support_start = max(0, core_start - taper)
        support_stop = min(height, core_stop + taper)
        segment_rows = rows[support_start:support_stop]
        values = np.interp(segment_rows, ys, dys)
        conf = np.interp(segment_rows, ys, weights)
        fade = np.ones(segment_rows.size, dtype=np.float64)
        left_count = core_start - support_start
        right_count = support_stop - core_stop
        if left_count:
            phase = np.arange(1, left_count + 1, dtype=np.float64) / (left_count + 1)
            fade[:left_count] = 0.5 - 0.5 * np.cos(np.pi * phase)
        if right_count:
            phase = np.arange(right_count, 0, -1, dtype=np.float64) / (right_count + 1)
            fade[-right_count:] = 0.5 - 0.5 * np.cos(np.pi * phase)
        smoothed[support_start:support_stop] = values * fade
        confidence[support_start:support_stop] = conf * fade
        raw[core_start:core_stop] = np.interp(rows[core_start:core_stop], ys, dys)
        trusted[core_start:core_stop] = True

    # Every explicit missing window owns an identity core.  No fit, including
    # one from a neighbouring trusted segment, may extrapolate through it.
    missing_core = np.zeros(height, dtype=bool)
    half_core = max(1, settings.fine_window_step_px // 4)
    for item in selected:
        if item.is_missing:
            centre = int(round(item.center_y))
            missing_core[max(0, centre - half_core) : min(height, centre + half_core + 1)] = True
    trusted[missing_core] = False
    smoothed[missing_core] = 0.0
    confidence[missing_core] = 0.0

    # Bound the curve by the distance to identity rows.  This simultaneously
    # enforces the local-slope limit and makes long untrusted gaps exactly zero.
    support = np.abs(smoothed) > 0.0
    zero_indices = np.flatnonzero(~support)
    anchors = np.concatenate(([-1], zero_indices, [height]))
    insertion = np.searchsorted(anchors, np.arange(height))
    distance = np.minimum(
        np.arange(height) - anchors[np.maximum(0, insertion - 1)],
        anchors[np.minimum(len(anchors) - 1, insertion)] - np.arange(height),
    )
    envelope = distance * settings.maximum_local_slope
    smoothed = np.sign(smoothed) * np.minimum(np.abs(smoothed), envelope)
    # The envelope constrains distance from identity; these two directional
    # projections also constrain variation within a trusted segment.
    for _ in range(3):
        previous = 0.0
        for index in range(height):
            smoothed[index] = np.clip(
                smoothed[index],
                previous - settings.maximum_local_slope,
                previous + settings.maximum_local_slope,
            )
            if not support[index]:
                smoothed[index] = 0.0
            previous = smoothed[index]
        following = 0.0
        for index in range(height - 1, -1, -1):
            smoothed[index] = np.clip(
                smoothed[index],
                following - settings.maximum_local_slope,
                following + settings.maximum_local_slope,
            )
            if not support[index]:
                smoothed[index] = 0.0
            following = smoothed[index]
    applied = smoothed.astype(np.float32)
    return VerticalCurve(
        y=rows.astype(np.float32),
        dy_raw=raw.astype(np.float32),
        dy_smoothed=applied,
        confidence=confidence.astype(np.float32),
        gain=np.ones(height, dtype=np.float32),
        dy_applied=applied.copy(),
        valid_observation_mask=trusted,
    )


def solve_vertical_curve(
    window_candidates: Sequence[Sequence[VerticalCandidate]],
    height: int,
    config: object | None = None,
) -> tuple[tuple[VerticalCandidate, ...], RansacSoftPrior, VerticalCurve]:
    """Fit the soft prior, solve the discrete path, then smooth ``dy(y)``.

    The three returned objects keep diagnostics available to the experiment
    runner while providing the single-call API used by the renderer.
    """

    settings = _coerce_config(config)
    prior = fit_ransac_soft_prior(window_candidates, settings)
    selected = solve_candidate_path(window_candidates, prior, settings)
    return selected, prior, fit_smooth_vertical_curve(selected, height, settings)


def select_local_gains(
    left: SourceFeatures,
    right: SourceFeatures,
    curve: VerticalCurve,
    config: object | None = None,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """Choose a smooth gain sequence from overlap residuals at analysis scale."""

    settings = _coerce_config(config)
    height, width = left.gray.shape
    if right.gray.shape != (height, width) or curve.dy_smoothed.shape != (height,):
        raise ValueError("S1 gain inputs must share height and overlap shape")
    block_starts = list(range(0, height, settings.gain_block_height_px))
    costs = np.full((len(block_starts), len(settings.gain_candidates)), np.inf, dtype=np.float64)
    x0, x1 = max(0, width // 2 - max(4, width // 6)), min(width, width // 2 + max(4, width // 6))
    xx = np.broadcast_to(np.arange(x0, x1, dtype=np.float32)[None, :], (height, x1 - x0))
    base_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], xx.shape)
    for gain_index, gain in enumerate(settings.gain_candidates):
        half = 0.5 * float(gain) * curve.dy_smoothed[:, None]
        left_sample = cv2.remap(left.gray[:, x0:x1], xx - x0, base_y - half, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        right_sample = cv2.remap(right.gray[:, x0:x1], xx - x0, base_y + half, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        left_valid = cv2.remap(left.valid_mask[:, x0:x1].astype(np.uint8), xx - x0, base_y - half, cv2.INTER_NEAREST) > 0
        right_valid = cv2.remap(right.valid_mask[:, x0:x1].astype(np.uint8), xx - x0, base_y + half, cv2.INTER_NEAREST) > 0
        residual = np.abs(left_sample - right_sample)
        gradient_residual = np.abs(
            cv2.Sobel(left_sample, cv2.CV_32F, 0, 1, ksize=3)
            - cv2.Sobel(right_sample, cv2.CV_32F, 0, 1, ksize=3)
        )
        for block_index, start in enumerate(block_starts):
            stop = min(height, start + settings.gain_block_height_px)
            valid = left_valid[start:stop] & right_valid[start:stop]
            if int(np.sum(valid)) < 16:
                continue
            costs[block_index, gain_index] = float(np.mean(residual[start:stop][valid]))
            costs[block_index, gain_index] += 0.25 * float(np.mean(gradient_residual[start:stop][valid]))
            costs[block_index, gain_index] += 0.002 * float(gain) * float(np.mean(np.abs(curve.dy_smoothed[start:stop])))
    dp = costs[0].copy()
    back = np.zeros_like(costs, dtype=np.int32)
    for block_index in range(1, len(block_starts)):
        next_cost = np.empty(len(settings.gain_candidates), dtype=np.float64)
        for current, gain in enumerate(settings.gain_candidates):
            transition = dp + settings.gain_smoothness_weight * np.abs(
                np.asarray(settings.gain_candidates) - gain
            )
            previous = int(np.argmin(transition))
            back[block_index, current] = previous
            next_cost[current] = costs[block_index, current] + transition[previous]
        dp = next_cost
    selected = [int(np.argmin(dp))]
    for block_index in range(len(block_starts) - 1, 0, -1):
        selected.append(int(back[block_index, selected[-1]]))
    selected.reverse()
    block_gains = tuple(float(settings.gain_candidates[index]) for index in selected)
    row_gain = np.empty(height, dtype=np.float32)
    for block_index, start in enumerate(block_starts):
        row_gain[start : min(height, start + settings.gain_block_height_px)] = block_gains[block_index]
    return row_gain, block_gains


def align_vertical_pair(
    left: SourceFeatures,
    right: SourceFeatures,
    config: object | None = None,
    *,
    select_gains: bool = True,
    risk_pair: bool = False,
    motion_dy_px: float = 0.0,
) -> VerticalAlignmentResult:
    """Run the complete fail-local S1 estimator for one source pair."""

    settings = _coerce_config(config)
    try:
        candidates = generate_vertical_candidates(
            left,
            right,
            settings,
            risk_pair=risk_pair,
            motion_dy_px=motion_dy_px,
        )
        prior = fit_ransac_soft_prior(candidates, settings)
        selected = solve_candidate_path(candidates, prior, settings)
        curve = fit_smooth_vertical_curve(selected, left.gray.shape[0], settings)
        if settings.normal_coarse_search_radius_y_px is None:
            status = "s1_ok" if any(not item.is_missing for item in selected) else "s1_zero_correction"
        else:
            trusted_count = int(np.count_nonzero(curve.valid_observation_mask))
            if trusted_count == 0:
                status = "s11_zero_unobservable"
            elif any(item.is_missing for item in selected):
                status = "s11_partial_rows"
            else:
                status = "s11_full_rows"
        blocks: tuple[float, ...] = ()
        if select_gains and status in {"s1_ok", "s11_full_rows", "s11_partial_rows"}:
            row_gain, blocks = select_local_gains(left, right, curve, settings)
            curve = VerticalCurve(
                y=curve.y,
                dy_raw=curve.dy_raw,
                dy_smoothed=curve.dy_smoothed,
                confidence=curve.confidence,
                gain=row_gain,
                dy_applied=(curve.dy_smoothed * row_gain).astype(np.float32),
                valid_observation_mask=curve.valid_observation_mask,
            )
        return VerticalAlignmentResult(
            pair_status=status,
            candidates_by_window=candidates,
            selected_candidates=selected,
            ransac_prior=prior,
            curve=curve,
            gain_block_values=blocks,
            diagnostics={
                "window_count": len(candidates),
                "missing_count": sum(item.is_missing for item in selected),
                "ransac_soft_prior_used": prior.available,
            },
        )
    except (ArithmeticError, cv2.error, np.linalg.LinAlgError, ValueError) as exc:
        # Pair-local numerical/observability failures are required to degrade
        # to S0-equivalent output.  Structural programmer errors remain visible.
        height = left.gray.shape[0]
        curve = fit_smooth_vertical_curve((), height, settings)
        return VerticalAlignmentResult(
            pair_status="s1_no_valid_path",
            candidates_by_window=(),
            selected_candidates=(),
            ransac_prior=RansacSoftPrior(False),
            curve=curve,
            diagnostics={"degraded_reason": str(exc)},
        )
