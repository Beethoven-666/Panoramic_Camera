"""Held-out acceptance metrics for the isolated S1.1 renderer.

The functions in this module are deliberately downstream-only: they consume
final boundaries, validity, or rendered images and return immutable audit
records.  No result is suitable as input to handoff or warp selection.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .video_s1_handoff import HandoffObservation


@dataclass(frozen=True)
class SingleCopySample:
    match_id: str
    pair_index: int
    copy_count: int
    evaluable: bool
    left_copy_present: bool
    right_copy_present: bool


@dataclass(frozen=True)
class SingleCopyMetrics:
    total_heldout_count: int
    evaluable_count: int
    evaluable_fraction: float
    duplicate_count: int
    missing_count: int
    single_copy_count: int
    duplicate_fraction: float
    missing_fraction: float
    samples: tuple[SingleCopySample, ...]


@dataclass(frozen=True)
class EdgeRidge:
    boundary_y: float
    slope: float
    strength: float
    width_px: float
    sample_count: int
    # These two fields are populated by the image extractor.  Defaults keep
    # hand-authored audit fixtures and the public value type backwards
    # compatible.  Polarity and the normalized cross-edge profile prevent a
    # nearby but unrelated horizontal edge from being reported as a vertical
    # residual observation.
    polarity: int = 0
    descriptor: tuple[float, ...] = ()


@dataclass(frozen=True)
class EdgeResidualSample:
    left_boundary_y: float
    right_boundary_y: float
    subpixel_residual_px: float
    legacy_integer_peak_row_gap_px: int
    peak_strength: float
    edge_width_px: float


@dataclass(frozen=True)
class VerticalResidualMetrics:
    subpixel_edge_residual_p50: float
    subpixel_edge_residual_p95: float
    subpixel_edge_residual_p99: float
    subpixel_edge_residual_max: float
    legacy_integer_peak_row_gap_p50: float
    legacy_integer_peak_row_gap_p95: float
    coverage: float
    measurement_sample_count: int
    edge_peak_strength: float
    edge_width: float
    double_edge_count: int
    samples: tuple[EdgeResidualSample, ...]


@dataclass(frozen=True)
class StageVerticalMetrics:
    stage_b: VerticalResidualMetrics
    stage_c: VerticalResidualMetrics

    def as_report(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {}
        for prefix, value in (("stage_b", self.stage_b), ("stage_c", self.stage_c)):
            result.update({
                f"{prefix}_subpixel_edge_residual_p50": value.subpixel_edge_residual_p50,
                f"{prefix}_subpixel_edge_residual_p95": value.subpixel_edge_residual_p95,
                f"{prefix}_subpixel_edge_residual_p99": value.subpixel_edge_residual_p99,
                f"{prefix}_subpixel_edge_residual_max": value.subpixel_edge_residual_max,
                f"{prefix}_coverage": value.coverage,
                f"{prefix}_measurement_sample_count": value.measurement_sample_count,
                f"{prefix}_edge_peak_strength": value.edge_peak_strength,
                f"{prefix}_edge_width": value.edge_width,
                f"{prefix}_double_edge_count": value.double_edge_count,
            })
        # The legacy headline remains tied to final Stage C.
        result["legacy_integer_peak_row_gap_p50"] = (
            self.stage_c.legacy_integer_peak_row_gap_p50
        )
        result["legacy_integer_peak_row_gap_p95"] = (
            self.stage_c.legacy_integer_peak_row_gap_p95
        )
        return result


@dataclass(frozen=True)
class PairStageVerticalMetrics:
    """Stage B/C ridge audit for one physical straight boundary."""

    pair_index: int
    boundary_x: int
    stage_b: VerticalResidualMetrics
    stage_c: VerticalResidualMetrics

    def as_report(self) -> dict[str, object]:
        return {
            "pair_index": self.pair_index,
            "boundary_x": self.boundary_x,
            **StageVerticalMetrics(self.stage_b, self.stage_c).as_report(),
        }


def _sample_valid(mask: np.ndarray | None, y: float, x: float) -> bool:
    if mask is None:
        return True
    row, column = int(np.rint(y)), int(np.rint(x))
    return (
        0 <= row < mask.shape[0]
        and 0 <= column < mask.shape[1]
        and bool(mask[row, column])
    )


def evaluate_heldout_single_copy(
    observations_by_pair: Sequence[Sequence[HandoffObservation]],
    boundaries: Sequence[int],
    *,
    left_valid_masks: Sequence[np.ndarray | None] | None = None,
    right_valid_masks: Sequence[np.ndarray | None] | None = None,
) -> SingleCopyMetrics:
    """Count rendered copies using only stable held-out correspondences.

    A left projection is present only left of the pair boundary; a right
    projection is present only at or right of it.  This is the same half-open
    owner convention used by rendering.  Invalid samples are excluded from the
    evaluable denominator, never silently classified as safe.
    """
    if len(observations_by_pair) != len(boundaries):
        raise ValueError("heldout observations must align with boundaries")
    left_masks = left_valid_masks or tuple(None for _ in boundaries)
    right_masks = right_valid_masks or tuple(None for _ in boundaries)
    if len(left_masks) != len(boundaries) or len(right_masks) != len(boundaries):
        raise ValueError("valid masks must align with boundaries")
    samples: list[SingleCopySample] = []
    for pair_index, (observations, boundary, left_mask, right_mask) in enumerate(
        zip(observations_by_pair, boundaries, left_masks, right_masks)
    ):
        for item in observations:
            if item.role != "heldout":
                raise ValueError("single-copy acceptance may consume heldout observations only")
            left_valid = _sample_valid(left_mask, item.left_y, item.left_canvas_x)
            right_valid = _sample_valid(right_mask, item.right_y, item.right_canvas_x)
            evaluable = left_valid and right_valid
            left_present = evaluable and item.left_canvas_x < boundary
            right_present = evaluable and item.right_canvas_x >= boundary
            copy_count = int(left_present) + int(right_present) if evaluable else 0
            samples.append(SingleCopySample(
                match_id=item.match_id, pair_index=pair_index,
                copy_count=copy_count, evaluable=evaluable,
                left_copy_present=left_present, right_copy_present=right_present,
            ))
    total = len(samples)
    evaluable_samples = [item for item in samples if item.evaluable]
    evaluable = len(evaluable_samples)
    duplicate = sum(item.copy_count >= 2 for item in evaluable_samples)
    missing = sum(item.copy_count == 0 for item in evaluable_samples)
    single = sum(item.copy_count == 1 for item in evaluable_samples)
    denominator = max(1, evaluable)
    return SingleCopyMetrics(
        total_heldout_count=total, evaluable_count=evaluable,
        evaluable_fraction=float(evaluable / total) if total else 0.0,
        duplicate_count=duplicate, missing_count=missing, single_copy_count=single,
        duplicate_fraction=float(duplicate / denominator) if evaluable else 0.0,
        missing_fraction=float(missing / denominator) if evaluable else 0.0,
        samples=tuple(samples),
    )


def slope_compensated_residual_samples(
    left_ridges: Sequence[EdgeRidge],
    right_ridges: Sequence[EdgeRidge],
    *,
    maximum_pair_distance_px: float = 8.0,
) -> tuple[tuple[EdgeResidualSample, ...], int]:
    """Pair ridge intercepts at ``x = boundary - 0.5`` without an 8 px trim.

    ``EdgeRidge.boundary_y`` must already be the line-fit extrapolation to that
    unique physical boundary.  The nearest monotonic matching is deterministic;
    extra plausible ridges are reported as double edges rather than discarded.
    """
    if maximum_pair_distance_px <= 0.0:
        raise ValueError("maximum ridge pairing distance must be positive")
    left = sorted(left_ridges, key=lambda item: item.boundary_y)
    right = sorted(right_ridges, key=lambda item: item.boundary_y)

    def descriptor_correlation(a: EdgeRidge, b: EdgeRidge) -> float | None:
        if not a.descriptor or not b.descriptor:
            return None
        first = np.asarray(a.descriptor, dtype=np.float64)
        second = np.asarray(b.descriptor, dtype=np.float64)
        if first.shape != second.shape or first.ndim != 1:
            return float("-inf")
        return float(np.clip(np.dot(first, second), -1.0, 1.0))

    correlations = np.full((len(left), len(right)), np.nan, dtype=np.float64)
    for left_index, a in enumerate(left):
        for right_index, b in enumerate(right):
            if a.polarity and b.polarity and a.polarity != b.polarity:
                continue
            if abs(a.slope - b.slope) > 0.15:
                continue
            if min(a.strength, b.strength) / max(a.strength, b.strength, 1e-9) < 0.35:
                continue
            value = descriptor_correlation(a, b)
            if value is not None:
                correlations[left_index, right_index] = value

    def descriptor_is_unique(left_index: int, right_index: int) -> bool:
        if not np.isfinite(correlations[left_index, right_index]):
            return True
        row = correlations[left_index]
        column = correlations[:, right_index]
        row_finite = row[np.isfinite(row)]
        column_finite = column[np.isfinite(column)]
        value = correlations[left_index, right_index]
        if value + 1e-12 < np.max(row_finite) or value + 1e-12 < np.max(column_finite):
            return False
        row_second = np.partition(row_finite, -2)[-2] if row_finite.size > 1 else -1.0
        column_second = (
            np.partition(column_finite, -2)[-2] if column_finite.size > 1 else -1.0
        )
        return value - max(row_second, column_second) >= 0.01

    def compatible(left_index: int, right_index: int) -> tuple[bool, float]:
        a, b = left[left_index], right[right_index]
        if a.polarity and b.polarity and a.polarity != b.polarity:
            return False, float("inf")
        if abs(a.slope - b.slope) > 0.15:
            return False, float("inf")
        strength_ratio = min(a.strength, b.strength) / max(
            a.strength, b.strength, 1e-9
        )
        if strength_ratio < 0.35:
            return False, float("inf")
        correlation = descriptor_correlation(a, b)
        if correlation is not None and correlation < 0.80:
            return False, float("inf")
        if correlation is not None and not descriptor_is_unique(left_index, right_index):
            return False, float("inf")
        residual = abs(a.boundary_y - b.boundary_y)
        # ``maximum_pair_distance_px`` is a scale for the ordered assignment,
        # not a trimming threshold.  Strongly corresponding ridges farther
        # than 8 px remain measurements, as required by the acceptance plan.
        # Beyond the configured search envelope, require essentially
        # identical extracted profiles.  This does not trim a measured large
        # residual: a strong correspondence is still emitted at its full
        # value.  It prevents the ordered assignment from pairing unrelated
        # ridges merely to increase its match count when one side has an
        # occlusion or a newly appearing edge.
        if residual > maximum_pair_distance_px and (
            correlation is None
            or correlation < 0.995
            or residual > 2.0 * maximum_pair_distance_px
        ):
            return False, float("inf")
        appearance_cost = 0.0 if correlation is None else 0.5 * (1.0 - correlation)
        cost = residual / maximum_pair_distance_px
        cost += 2.0 * abs(a.slope - b.slope) + appearance_cost
        return True, float(cost)

    # Ordered sequence alignment is essential here: greedy nearest-neighbour
    # pairing can cross two adjacent ridges and manufacture a small residual.
    # The lexicographic state maximizes real correspondences first, then
    # minimizes their geometric/appearance cost.  Unmatched ridges remain
    # explicit double-edge evidence.
    states: list[list[tuple[int, float, tuple[tuple[int, int], ...]] | None]] = [
        [None] * (len(right) + 1) for _ in range(len(left) + 1)
    ]
    states[0][0] = (0, 0.0, ())

    def improve(
        i: int,
        j: int,
        candidate: tuple[int, float, tuple[tuple[int, int], ...]],
    ) -> None:
        prior = states[i][j]
        if prior is None or (candidate[0], -candidate[1]) > (prior[0], -prior[1]):
            states[i][j] = candidate

    for left_index in range(len(left) + 1):
        for right_index in range(len(right) + 1):
            state = states[left_index][right_index]
            if state is None:
                continue
            count, cost, pairs = state
            if left_index < len(left):
                improve(left_index + 1, right_index, (count, cost, pairs))
            if right_index < len(right):
                improve(left_index, right_index + 1, (count, cost, pairs))
            if left_index < len(left) and right_index < len(right):
                accepted, pair_cost = compatible(left_index, right_index)
                if accepted:
                    improve(
                        left_index + 1,
                        right_index + 1,
                        (count + 1, cost + pair_cost, pairs + ((left_index, right_index),)),
                    )

    matched = states[-1][-1]
    pairs = () if matched is None else matched[2]
    samples: list[EdgeResidualSample] = []
    for left_index, right_index in pairs:
        a, b = left[left_index], right[right_index]
        residual = abs(a.boundary_y - b.boundary_y)
        samples.append(EdgeResidualSample(
            left_boundary_y=a.boundary_y, right_boundary_y=b.boundary_y,
            subpixel_residual_px=float(residual),
            legacy_integer_peak_row_gap_px=abs(round(a.boundary_y) - round(b.boundary_y)),
            peak_strength=float(min(a.strength, b.strength)),
            edge_width_px=float(max(a.width_px, b.width_px)),
        ))
    # Any unmatched ridge is visible double-edge evidence. Alternative
    # combinatorial pairings of the same two ridges are ambiguity, not extra
    # rendered edges, and must not inflate the count.
    unmatched = len(left) + len(right) - 2 * len(pairs)
    return tuple(sorted(samples, key=lambda item: item.left_boundary_y)), unmatched


def _line_tracks(
    gray: np.ndarray,
    valid: np.ndarray,
    x_values: Sequence[int],
    *, minimum_strength: float,
    maximum_link_dy_px: float = 2.5,
    signed_gradient: np.ndarray | None = None,
) -> tuple[EdgeRidge, ...]:
    if signed_gradient is None:
        signed_gradient = cv2.Sobel(
            gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3
        )
    gradient = np.abs(signed_gradient)
    tracks: list[list[tuple[float, float, float, float]]] = []
    for x in x_values:
        profile = gradient[:, x]
        accepted = valid[:, x]
        peaks = np.flatnonzero(
            accepted
            & (profile >= minimum_strength)
            & (profile >= np.roll(profile, 1))
            & (profile > np.roll(profile, -1))
        )
        peaks = peaks[(peaks > 0) & (peaks + 1 < gray.shape[0])]
        assigned: set[int] = set()
        for y in peaks:
            denominator = float(profile[y - 1] - 2.0 * profile[y] + profile[y + 1])
            fraction = (
                0.5 * float(profile[y - 1] - profile[y + 1]) / denominator
                if denominator < -1e-6
                else 0.0
            )
            subpixel_y = float(y + np.clip(fraction, -0.75, 0.75))
            polarity = 1.0 if signed_gradient[y, x] >= 0.0 else -1.0
            choices = [
                (abs(track[-1][1] - subpixel_y), index)
                for index, track in enumerate(tracks)
                if (
                    abs(track[-1][0] - x) == 1
                    and abs(track[-1][1] - subpixel_y) <= maximum_link_dy_px
                    and track[-1][3] == polarity
                )
            ]
            if choices:
                _, index = min(choices)
                if index in assigned:
                    # Do not start a new ridge away from the physical seam.
                    # Such a track does not provide boundary evidence.
                    continue
                else:
                    tracks[index].append((float(x), subpixel_y, float(profile[y]), polarity))
                    assigned.add(index)
            elif not tracks or all(len(track) == 0 for track in tracks):
                tracks.append(([(float(x), subpixel_y, float(profile[y]), polarity)]))
            elif x == x_values[0]:
                tracks.append([(float(x), subpixel_y, float(profile[y]), polarity)])
    minimum_samples = max(2, int(np.ceil(len(x_values) * 0.5)))
    result = []
    if not x_values:
        return ()
    # The caller orders both shoulders from the seam outwards.  Extrapolating
    # from the seam-adjacent observation avoids an eight-column lever arm that
    # used to amplify a 0.1 px slope error into a false ~1 px residual.
    physical_boundary_x = (
        float(x_values[0]) + 0.5 if x_values[-1] < x_values[0]
        else float(x_values[0]) - 0.5
    )
    for track in tracks:
        if len(track) < minimum_samples:
            continue
        xs = np.asarray([item[0] for item in track])
        ys = np.asarray([item[1] for item in track])
        slope, intercept = np.polyfit(xs, ys, 1)
        predicted = slope * xs + intercept
        keep = np.abs(ys - predicted) <= 1.5
        if np.count_nonzero(keep) < minimum_samples:
            continue
        slope, intercept = np.polyfit(xs[keep], ys[keep], 1)
        strengths = np.asarray([item[2] for item in track])[keep]
        # Effective width is inversely related to normalized peak sharpness and
        # remains an explicit anti-blur diagnostic.
        local_scale = max(float(np.percentile(strengths, 95)), 1e-6)
        width = float(np.median(np.clip(local_scale / np.maximum(strengths, 1e-6), 1.0, 16.0)))
        boundary_y = float(track[0][1] + slope * (physical_boundary_x - track[0][0]))
        offsets = np.arange(-6, 7, dtype=np.float64)
        profiles: list[np.ndarray] = []
        for item, accepted_item in zip(track, keep):
            if not accepted_item:
                continue
            sample_y = item[1] + offsets
            lower = np.clip(np.floor(sample_y).astype(np.int64), 0, gray.shape[0] - 1)
            upper = np.clip(lower + 1, 0, gray.shape[0] - 1)
            fraction = sample_y - lower
            profile = gray[lower, int(item[0])] * (1.0 - fraction)
            profile += gray[upper, int(item[0])] * fraction
            profiles.append(profile)
        descriptor = np.mean(profiles, axis=0).astype(np.float64)
        descriptor -= np.mean(descriptor)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-9)
        result.append(EdgeRidge(
            boundary_y=boundary_y, slope=float(slope),
            strength=float(np.median(strengths)), width_px=width,
            sample_count=int(np.count_nonzero(keep)),
            polarity=int(np.sign(np.median([item[3] for item in track]))),
            descriptor=tuple(float(item) for item in descriptor),
        ))
    return tuple(result)


def extract_boundary_ridges(
    image: np.ndarray,
    boundary_x: int,
    *,
    valid_mask: np.ndarray | None = None,
    shoulder_width_px: int = 8,
    minimum_peak_strength: float = 12.0,
) -> tuple[tuple[EdgeRidge, ...], tuple[EdgeRidge, ...]]:
    """Extract and slope-compensate edge ridges on both sides of a hard seam."""
    value = np.asarray(image)
    if value.ndim == 3 and value.shape[2] == 3:
        gray = cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif value.ndim == 2:
        gray = value.astype(np.float32)
    else:
        raise ValueError("edge metric image must be gray or BGR")
    valid = np.ones(gray.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != gray.shape:
        raise ValueError("edge metric valid mask shape mismatch")
    if not 1 <= boundary_x < gray.shape[1] or shoulder_width_px < 2:
        raise ValueError("boundary or shoulder is invalid")
    signed_gradient = cv2.Sobel(
        gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3
    )
    return _extract_boundary_ridges_precomputed(
        gray,
        valid,
        signed_gradient,
        boundary_x,
        shoulder_width_px=shoulder_width_px,
        minimum_peak_strength=minimum_peak_strength,
    )


def _extract_boundary_ridges_precomputed(
    gray: np.ndarray,
    valid: np.ndarray,
    signed_gradient: np.ndarray,
    boundary_x: int,
    *,
    shoulder_width_px: int,
    minimum_peak_strength: float,
) -> tuple[tuple[EdgeRidge, ...], tuple[EdgeRidge, ...]]:
    """Extract one boundary while reusing the image-wide Sobel response."""

    left_x = tuple(
        range(boundary_x - 1, max(-1, boundary_x - shoulder_width_px - 1), -1)
    )
    right_x = tuple(range(boundary_x, min(gray.shape[1], boundary_x + shoulder_width_px)))
    left = _line_tracks(
        gray,
        valid,
        left_x,
        minimum_strength=minimum_peak_strength,
        signed_gradient=signed_gradient,
    )
    right = _line_tracks(
        gray,
        valid,
        right_x,
        minimum_strength=minimum_peak_strength,
        signed_gradient=signed_gradient,
    )
    return left, right


def summarize_vertical_residuals(
    samples: Sequence[EdgeResidualSample], *,
    expected_sample_count: int,
    double_edge_count: int = 0,
) -> VerticalResidualMetrics:
    if expected_sample_count < 0 or double_edge_count < 0:
        raise ValueError("metric counts cannot be negative")
    residuals = np.asarray([item.subpixel_residual_px for item in samples], dtype=np.float64)
    legacy = np.asarray([item.legacy_integer_peak_row_gap_px for item in samples], dtype=np.float64)
    strengths = np.asarray([item.peak_strength for item in samples], dtype=np.float64)
    widths = np.asarray([item.edge_width_px for item in samples], dtype=np.float64)

    def percentile(values: np.ndarray, q: float) -> float:
        return float(np.percentile(values, q)) if values.size else float("nan")

    return VerticalResidualMetrics(
        subpixel_edge_residual_p50=percentile(residuals, 50),
        subpixel_edge_residual_p95=percentile(residuals, 95),
        subpixel_edge_residual_p99=percentile(residuals, 99),
        subpixel_edge_residual_max=float(np.max(residuals)) if residuals.size else float("nan"),
        legacy_integer_peak_row_gap_p50=percentile(legacy, 50),
        legacy_integer_peak_row_gap_p95=percentile(legacy, 95),
        coverage=float(len(samples) / expected_sample_count) if expected_sample_count else 0.0,
        measurement_sample_count=len(samples),
        edge_peak_strength=percentile(strengths, 50),
        edge_width=percentile(widths, 50),
        double_edge_count=int(double_edge_count), samples=tuple(samples),
    )


def measure_boundary_edge_residuals(
    image: np.ndarray,
    boundaries: Sequence[int],
    *,
    valid_mask: np.ndarray | None = None,
    shoulder_width_px: int = 8,
    minimum_peak_strength: float = 12.0,
    maximum_pair_distance_px: float = 8.0,
) -> VerticalResidualMetrics:
    gray, valid, signed_gradient = _prepare_edge_metric_image(image, valid_mask)
    return _measure_precomputed_boundary_edge_residuals(
        gray,
        valid,
        signed_gradient,
        boundaries,
        shoulder_width_px=shoulder_width_px,
        minimum_peak_strength=minimum_peak_strength,
        maximum_pair_distance_px=maximum_pair_distance_px,
    )


def _prepare_edge_metric_image(
    image: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(image)
    if value.ndim == 3 and value.shape[2] == 3:
        gray = cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif value.ndim == 2:
        gray = value.astype(np.float32)
    else:
        raise ValueError("edge metric image must be gray or BGR")
    valid = (
        np.ones(gray.shape, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if valid.shape != gray.shape:
        raise ValueError("edge metric valid mask shape mismatch")
    signed_gradient = cv2.Sobel(
        gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3
    )
    return gray, valid, signed_gradient


def _measure_precomputed_boundary_edge_residuals(
    gray: np.ndarray,
    valid: np.ndarray,
    signed_gradient: np.ndarray,
    boundaries: Sequence[int],
    *,
    shoulder_width_px: int,
    minimum_peak_strength: float,
    maximum_pair_distance_px: float,
) -> VerticalResidualMetrics:
    all_samples: list[EdgeResidualSample] = []
    double_edges = 0
    expected = 0
    for boundary in boundaries:
        if not 1 <= int(boundary) < gray.shape[1] or shoulder_width_px < 2:
            raise ValueError("boundary or shoulder is invalid")
        left, right = _extract_boundary_ridges_precomputed(
            gray,
            valid,
            signed_gradient,
            int(boundary),
            shoulder_width_px=shoulder_width_px,
            minimum_peak_strength=minimum_peak_strength,
        )
        expected += max(len(left), len(right))
        samples, doubles = slope_compensated_residual_samples(
            left, right, maximum_pair_distance_px=maximum_pair_distance_px
        )
        all_samples.extend(samples)
        double_edges += doubles
    return summarize_vertical_residuals(
        all_samples, expected_sample_count=expected, double_edge_count=double_edges
    )


def compare_stage_b_to_stage_c(
    stage_b: np.ndarray,
    stage_c: np.ndarray,
    boundaries: Sequence[int],
    *,
    stage_b_valid: np.ndarray | None = None,
    stage_c_valid: np.ndarray | None = None,
    metric_options: Mapping[str, float | int] | None = None,
) -> StageVerticalMetrics:
    options = dict(metric_options or {})
    with ThreadPoolExecutor(max_workers=2) as executor:
        stage_b_future = executor.submit(
            measure_boundary_edge_residuals,
            stage_b,
            boundaries,
            valid_mask=stage_b_valid,
            **options,
        )
        stage_c_future = executor.submit(
            measure_boundary_edge_residuals,
            stage_c,
            boundaries,
            valid_mask=stage_c_valid,
            **options,
        )
        return StageVerticalMetrics(
            stage_b=stage_b_future.result(),
            stage_c=stage_c_future.result(),
        )


def compare_stage_b_to_stage_c_per_pair(
    stage_b: np.ndarray,
    stage_c: np.ndarray,
    boundaries: Sequence[int],
    *,
    stage_b_valid: np.ndarray | None = None,
    stage_c_valid: np.ndarray | None = None,
    metric_options: Mapping[str, float | int] | None = None,
) -> tuple[PairStageVerticalMetrics, ...]:
    """Return acceptance-ready P95/P99 evidence for every straight pair.

    The function deliberately returns all boundaries, including pairs with no
    matched ridge samples (their percentiles remain NaN and their double-edge
    count/coverage expose why they are not evaluable).  Callers must not drop
    those records when deciding the per-pair ``<= 0.75 px`` gate.
    """

    options = dict(metric_options or {})
    allowed = {
        "shoulder_width_px",
        "minimum_peak_strength",
        "maximum_pair_distance_px",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError(f"unknown vertical metric options: {unknown}")
    shoulder = int(options.get("shoulder_width_px", 8))
    strength = float(options.get("minimum_peak_strength", 12.0))
    maximum_distance = float(options.get("maximum_pair_distance_px", 8.0))
    prepared_b = _prepare_edge_metric_image(stage_b, stage_b_valid)
    prepared_c = _prepare_edge_metric_image(stage_c, stage_c_valid)

    def measure(pair_index: int, boundary: int) -> PairStageVerticalMetrics:
        before = _measure_precomputed_boundary_edge_residuals(
            *prepared_b,
            (int(boundary),),
            shoulder_width_px=shoulder,
            minimum_peak_strength=strength,
            maximum_pair_distance_px=maximum_distance,
        )
        after = _measure_precomputed_boundary_edge_residuals(
            *prepared_c,
            (int(boundary),),
            shoulder_width_px=shoulder,
            minimum_peak_strength=strength,
            maximum_pair_distance_px=maximum_distance,
        )
        return PairStageVerticalMetrics(pair_index, int(boundary), before, after)

    # Per-pair reports are independent read-only measurements.  A small fixed
    # pool keeps audit mode practical without oversubscribing OpenCV kernels.
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(boundaries)))) as executor:
        futures = [
            executor.submit(measure, pair_index, int(boundary))
            for pair_index, boundary in enumerate(boundaries)
        ]
        return tuple(future.result() for future in futures)
