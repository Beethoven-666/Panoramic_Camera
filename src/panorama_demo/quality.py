from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .rgbd_projection import (
        EstimatedProjectionFootprint,
        SideScanFootprintEstimate,
    )


@dataclass(frozen=True)
class FrameQuality:
    sharpness: float
    tenengrad: float
    texture_coverage: float
    dark_ratio: float
    saturated_ratio: float
    contrast: float
    detail_strength: float

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpness": self.sharpness,
            "tenengrad": self.tenengrad,
            "texture_coverage": self.texture_coverage,
            "dark_ratio": self.dark_ratio,
            "saturated_ratio": self.saturated_ratio,
            "contrast": self.contrast,
            "detail_strength": self.detail_strength,
        }


@dataclass(frozen=True)
class MotionEstimate:
    dx: float
    dy: float
    matches: int
    inlier_ratio: float
    grid_coverage: float
    method: str

    @property
    def reliable(self) -> bool:
        if self.method == "features":
            return (
                self.matches >= 24
                and self.inlier_ratio >= 0.35
                and self.grid_coverage >= 0.12
            )
        return self.inlier_ratio >= 0.12

    def as_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "matches": self.matches,
            "inlier_ratio": self.inlier_ratio,
            "grid_coverage": self.grid_coverage,
            "method": self.method,
            "reliable": self.reliable,
        }


@dataclass(frozen=True)
class AdaptiveSelection:
    indices: tuple[int, ...]
    target_displacement: float
    scan_direction: int
    unreliable_edges: int

    def as_dict(self) -> dict[str, object]:
        return {
            "indices": list(self.indices),
            "target_displacement": self.target_displacement,
            "scan_direction": self.scan_direction,
            "unreliable_edges": self.unreliable_edges,
        }


@dataclass(frozen=True)
class ScanSegment:
    start_index: int
    end_index: int
    scan_direction: int
    displacement: float
    reliable_fraction: float
    candidate_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "scan_direction": self.scan_direction,
            "displacement": self.displacement,
            "reliable_fraction": self.reliable_fraction,
            "candidate_count": self.candidate_count,
        }


def assess_capture_quality(
    qualities: list[FrameQuality],
    exposure_raw: list[int | None],
    *,
    exposure_unit_us: float = 100.0,
    maximum_exposure_us: float = 1200.0,
) -> dict[str, float | int | bool | list[str]]:
    """Fail-closed input audit for blur-producing exposure and unusable lighting."""

    if not qualities or len(qualities) != len(exposure_raw):
        raise ValueError("Capture qualities and exposure metadata must have equal length")
    if exposure_unit_us <= 0.0 or maximum_exposure_us <= 0.0:
        raise ValueError("Exposure audit limits must be positive")
    recorded_us = np.asarray(
        [value * exposure_unit_us for value in exposure_raw if value is not None],
        dtype=np.float64,
    )
    dark = np.asarray([quality.dark_ratio for quality in qualities])
    saturated = np.asarray([quality.saturated_ratio for quality in qualities])
    sharpness = np.asarray([quality.sharpness for quality in qualities])
    tenengrad = np.asarray([quality.tenengrad for quality in qualities])
    detail = np.asarray([quality.detail_strength for quality in qualities])
    failures: list[str] = []
    exposure_p95 = float(np.percentile(recorded_us, 95)) if recorded_us.size else 0.0
    exposure_max = float(recorded_us.max()) if recorded_us.size else 0.0
    if recorded_us.size and exposure_max > maximum_exposure_us:
        failures.append(
            f"recorded color exposure reached {exposure_max:.0f} us, above the "
            f"motion-safe {maximum_exposure_us:.0f} us limit"
        )
    dark_p50 = float(np.median(dark))
    saturated_p50 = float(np.median(saturated))
    sharpness_p75 = float(np.percentile(sharpness, 75))
    tenengrad_p75 = float(np.percentile(tenengrad, 75))
    detail_p75 = float(np.percentile(detail, 75))
    if dark_p50 > 0.40:
        failures.append("more than 40% of a typical frame is underexposed")
    if saturated_p50 > 0.25:
        failures.append("more than 25% of a typical frame is saturated")
    if detail_p75 < 5.4 and tenengrad_p75 < 18.0:
        failures.append("even the clearest input frames lack usable sharp detail")
    return {
        "quality_pass": not failures,
        "frames_checked": len(qualities),
        "exposure_metadata_count": int(recorded_us.size),
        "exposure_p95_us": exposure_p95,
        "exposure_max_us": exposure_max,
        "dark_ratio_p50": dark_p50,
        "saturated_ratio_p50": saturated_p50,
        "sharpness_p75": sharpness_p75,
        "tenengrad_p75": tenengrad_p75,
        "detail_strength_p75": detail_p75,
        "failure_reasons": failures,
    }


def resize_for_analysis(image: np.ndarray, width: int = 320) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Analysis expects an HxWx3 BGR image")
    if width < 160:
        raise ValueError("Analysis width must be at least 160 pixels")
    height = max(96, int(round(image.shape[0] * width / image.shape[1])))
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def analyze_frame_quality(image: np.ndarray) -> FrameQuality:
    """Measure blur, texture coverage and exposure without assuming one scene."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Frame quality expects an HxWx3 BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    tile_values: list[float] = []
    textured_tiles = 0
    tile_count = 0
    for y0, y1 in zip(
        np.linspace(0, height, 6, dtype=int)[:-1],
        np.linspace(0, height, 6, dtype=int)[1:],
        strict=True,
    ):
        for x0, x1 in zip(
            np.linspace(0, width, 9, dtype=int)[:-1],
            np.linspace(0, width, 9, dtype=int)[1:],
            strict=True,
        ):
            tile = gray[y0:y1, x0:x1]
            value = float(cv2.Laplacian(tile, cv2.CV_32F).var())
            tile_values.append(value)
            tile_count += 1
    values = np.asarray(tile_values, dtype=np.float64)
    reference = max(4.0, float(np.percentile(values, 60)) * 0.25)
    textured_tiles = int(np.count_nonzero(values >= reference))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    p10, p90 = np.percentile(gray, [10, 90])
    return FrameQuality(
        sharpness=float(np.percentile(np.log1p(values), 55)),
        tenengrad=float(np.mean(gradient)),
        texture_coverage=textured_tiles / max(1, tile_count),
        dark_ratio=float(np.mean(gray < 16)),
        saturated_ratio=float(np.mean(gray > 248)),
        contrast=float((p90 - p10) / 255.0),
        detail_strength=float(np.percentile(np.log1p(values), 80)),
    )


def relative_quality_scores(qualities: list[FrameQuality]) -> np.ndarray:
    if not qualities:
        return np.empty(0, dtype=np.float64)
    sharpness = np.asarray([quality.sharpness for quality in qualities])
    tenengrad = np.log1p([quality.tenengrad for quality in qualities])

    def robust_scale(values: np.ndarray) -> np.ndarray:
        low, high = np.percentile(values, [10, 90])
        return np.clip((values - low) / max(float(high - low), 1e-6), 0.0, 1.0)

    sharp_score = robust_scale(sharpness)
    gradient_score = robust_scale(np.asarray(tenengrad, dtype=np.float64))
    detail_score = robust_scale(
        np.asarray([quality.detail_strength for quality in qualities])
    )
    texture = np.asarray([quality.texture_coverage for quality in qualities])
    contrast = np.asarray([quality.contrast for quality in qualities])
    exposure = np.asarray(
        [
            np.clip(
                1.0 - quality.dark_ratio / 0.25 - quality.saturated_ratio / 0.15,
                0.0,
                1.0,
            )
            for quality in qualities
        ]
    )
    return (
        0.45 * sharp_score
        + 0.13 * detail_score
        + 0.18 * gradient_score
        + 0.08 * texture
        + 0.10 * exposure
        + 0.06 * np.clip(contrast / 0.45, 0.0, 1.0)
    )


def estimate_translation(
    reference: np.ndarray,
    source: np.ndarray,
) -> MotionEstimate:
    """Estimate source motion in analysis pixels with a phase fallback."""

    if reference.shape != source.shape:
        raise ValueError("Motion-estimation images must have identical shape")
    gray0 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=900, contrastThreshold=0.02, edgeThreshold=12)
    key0, desc0 = sift.detectAndCompute(gray0, None)
    key1, desc1 = sift.detectAndCompute(gray1, None)
    if desc0 is not None and desc1 is not None and len(key0) >= 12 and len(key1) >= 12:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc0, desc1, k=2)
        good = [first for first, second in pairs if first.distance < 0.74 * second.distance]
        if len(good) >= 8:
            points0 = np.float32([key0[match.queryIdx].pt for match in good])
            points1 = np.float32([key1[match.trainIdx].pt for match in good])
            affine, inliers = cv2.estimateAffinePartial2D(
                points1,
                points0,
                method=cv2.RANSAC,
                ransacReprojThreshold=2.0,
                maxIters=4000,
                confidence=0.995,
                refineIters=10,
            )
            if affine is not None and inliers is not None:
                support = inliers.ravel() > 0
                displacement = points0[support] - points1[support]
                if displacement.size:
                    grid_x = np.clip(
                        (points0[support, 0] * 6 / gray0.shape[1]).astype(int), 0, 5
                    )
                    grid_y = np.clip(
                        (points0[support, 1] * 4 / gray0.shape[0]).astype(int), 0, 3
                    )
                    cells = len(set(zip(grid_x.tolist(), grid_y.tolist(), strict=True)))
                    median = np.median(displacement, axis=0)
                    return MotionEstimate(
                        dx=float(median[0]),
                        dy=float(median[1]),
                        matches=len(good),
                        inlier_ratio=float(np.mean(support)),
                        grid_coverage=cells / 24.0,
                        method="features",
                    )

    edge0 = cv2.Laplacian(gray0, cv2.CV_32F)
    edge1 = cv2.Laplacian(gray1, cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(edge0, edge1)
    return MotionEstimate(
        dx=float(-shift[0]),
        dy=float(-shift[1]),
        matches=0,
        inlier_ratio=float(max(0.0, response)),
        grid_coverage=0.0,
        method="phase",
    )


def select_layout_indices_adaptive(
    thumbnails: list[np.ndarray],
    *,
    target_fraction: float = 0.18,
    maximum_fraction: float = 0.28,
    max_selected: int = 160,
) -> tuple[AdaptiveSelection, list[MotionEstimate]]:
    """Select variable-rate layout frames from measured adjacent image motion."""

    if len(thumbnails) < 2:
        raise ValueError("Adaptive layout selection requires at least two frames")
    shape = thumbnails[0].shape
    if any(image.shape != shape for image in thumbnails):
        raise ValueError("Adaptive-layout thumbnails must have identical shape")
    motions = [
        estimate_translation(reference, source)
        for reference, source in zip(thumbnails, thumbnails[1:])
    ]
    return (
        select_layout_from_motion_estimates(
            motions,
            frame_count=len(thumbnails),
            image_width=thumbnails[0].shape[1],
            target_fraction=target_fraction,
            maximum_fraction=maximum_fraction,
            max_selected=max_selected,
        ),
        motions,
    )


def select_primary_scan_segment(
    motions: list[MotionEstimate],
    *,
    image_width: int,
    maximum_fraction: float = 0.28,
    tolerated_unreliable_edges: int = 2,
) -> ScanSegment:
    """Trim startup, stops and return motion to the strongest one-way scan."""

    if not motions:
        raise ValueError("Scan segmentation requires adjacent motion estimates")
    if image_width < 1 or not 0.10 <= maximum_fraction <= 0.45:
        raise ValueError("Scan-segmentation limits are invalid")
    maximum_step = maximum_fraction * image_width
    minimum_progress = max(0.25, image_width * 0.001)
    candidates: list[tuple[float, int, int, int, float]] = []

    for direction in (-1, 1):
        run_start: int | None = None
        first_progress: int | None = None
        last_progress: int | None = None
        displacement = 0.0
        unreliable_gap = 0

        def finish() -> None:
            nonlocal run_start, first_progress, last_progress, displacement
            nonlocal unreliable_gap
            if first_progress is not None and last_progress is not None:
                start = first_progress
                end = last_progress + 1
                edge_count = max(1, end - start)
                reliable = sum(motion.reliable for motion in motions[start:end])
                reliable_fraction = reliable / edge_count
                if displacement >= 0.25 * image_width and end - start >= 3:
                    candidates.append(
                        (displacement, start, end, direction, reliable_fraction)
                    )
            run_start = None
            first_progress = None
            last_progress = None
            displacement = 0.0
            unreliable_gap = 0

        for edge_index, motion in enumerate(motions):
            if not motion.reliable:
                if run_start is not None and unreliable_gap < tolerated_unreliable_edges:
                    unreliable_gap += 1
                    continue
                finish()
                continue
            signed = motion.dx * direction
            vertical_outlier = abs(motion.dy) > max(
                image_width * 0.12, 2.5 * abs(motion.dx) + 2.0
            )
            invalid = (
                signed < -0.01 * image_width
                or signed > maximum_step
                or vertical_outlier
            )
            if invalid:
                finish()
                continue
            if run_start is None:
                run_start = edge_index
            unreliable_gap = 0
            if signed >= minimum_progress:
                if first_progress is None:
                    first_progress = edge_index
                last_progress = edge_index
                displacement += signed
        finish()

    if not candidates:
        raise RuntimeError("No continuous one-way scan segment has enough displacement")
    candidates.sort(key=lambda row: (row[0], row[4], row[2] - row[1]), reverse=True)
    displacement, start, end, direction, reliable_fraction = candidates[0]
    return ScanSegment(
        start_index=start,
        end_index=end,
        scan_direction=direction,
        displacement=float(displacement),
        reliable_fraction=float(reliable_fraction),
        candidate_count=len(candidates),
    )


def select_layout_from_motion_estimates(
    motions: list[MotionEstimate],
    *,
    frame_count: int,
    image_width: int,
    target_fraction: float = 0.18,
    maximum_fraction: float = 0.28,
    max_selected: int = 160,
) -> AdaptiveSelection:
    """Select layout indices from streamed adjacent-motion measurements."""

    if frame_count < 2 or len(motions) != frame_count - 1:
        raise ValueError("Adjacent motion count must be one less than frame count")
    if image_width < 1:
        raise ValueError("Analysis image width must be positive")
    if max_selected < 2:
        raise ValueError("Adaptive layout frame budget must be at least two")
    if not 0.05 <= target_fraction < maximum_fraction <= 0.45:
        raise ValueError("Adaptive-layout displacement fractions are invalid")
    reliable_dx = np.asarray(
        [motion.dx for motion in motions if motion.reliable and abs(motion.dx) >= 0.1]
    )
    if reliable_dx.size == 0:
        raise RuntimeError("Could not estimate a reliable scan direction")
    direction = 1 if float(np.median(reliable_dx)) >= 0.0 else -1
    width = image_width
    target = width * target_fraction
    maximum = width * maximum_fraction
    selected = [0]
    accumulated = 0.0
    unreliable_run = 0
    unreliable_total = 0
    recent_steps: list[float] = []
    for edge_index, motion in enumerate(motions):
        if not motion.reliable:
            unreliable_run += 1
            unreliable_total += 1
            if unreliable_run >= 3:
                raise RuntimeError(
                    "Three consecutive frame pairs lack reliable visual overlap"
                )
            if not recent_steps:
                continue
            signed = float(np.median(recent_steps[-5:]))
        else:
            unreliable_run = 0
            signed = motion.dx * direction
            if signed < -0.02 * width:
                raise RuntimeError("Camera scan direction reversed inside one sequence")
            if abs(motion.dy) > max(0.08 * width, 2.5 * abs(motion.dx) + 2.0):
                raise RuntimeError("Adjacent frames contain excessive vertical motion")
            signed = max(0.0, signed)
            recent_steps.append(signed)
        if signed > maximum:
            raise RuntimeError(
                "Adjacent captured frames moved too far to preserve safe overlap"
            )
        accumulated += max(0.0, signed)
        if accumulated >= target:
            selected.append(edge_index + 1)
            accumulated = 0.0
            if len(selected) > max_selected:
                raise RuntimeError(
                    "Adaptive layout exceeds the safe frame budget; split the route"
                )
    if selected[-1] != frame_count - 1:
        tail_motion = sum(
            max(0.0, motion.dx * direction)
            for motion in motions[selected[-1] :]
            if motion.reliable
        )
        if tail_motion >= width * 0.05:
            selected.append(frame_count - 1)
            if len(selected) > max_selected:
                raise RuntimeError(
                    "Adaptive layout exceeds the safe frame budget; split the route"
                )
    if len(selected) < 2:
        raise RuntimeError("The input does not contain enough scan motion for a panorama")
    return AdaptiveSelection(
        indices=tuple(selected),
        target_displacement=target,
        scan_direction=direction,
        unreliable_edges=unreliable_total,
    )


def select_render_indices_auto(
    qualities: list[FrameQuality],
    transforms: list[np.ndarray],
    image_shape: tuple[int, ...],
    *,
    maximum_keyframes: int = 0,
    target_spacing_fraction: float = 0.28,
    quality_gate: bool = True,
) -> tuple[list[int], dict[str, object]]:
    """Pick sharp dense render sources while preserving scan coverage."""

    if len(qualities) != len(transforms) or len(qualities) < 2:
        raise ValueError("Render qualities and transforms must have equal length")
    if maximum_keyframes == 1 or maximum_keyframes < 0:
        raise ValueError("maximum_keyframes must be zero or at least two")
    height, width = image_shape[:2]
    centers: list[np.ndarray] = []
    for transform in transforms:
        matrix = np.asarray(transform, dtype=np.float64)
        point = matrix @ np.array([width * 0.5, height * 0.5, 1.0])
        centers.append(point[:2] / point[2])
    center_array = np.asarray(centers)
    centered = center_array - center_array.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if float(np.dot(axis, center_array[-1] - center_array[0])) < 0.0:
        axis *= -1.0
    positions = center_array @ axis
    cross_axis = np.array([-axis[1], axis[0]], dtype=np.float64)
    cross_positions = center_array @ cross_axis
    cross_reference = float(np.median(cross_positions))
    cross_scale = max(1.0, 0.04 * height)
    if np.any(np.diff(positions) < -0.04 * width):
        raise RuntimeError("Dense render trajectory is not monotonic along the scan")
    span = float(positions.max() - positions.min())
    if span < 0.10 * width:
        raise RuntimeError("Dense render trajectory is too short for a panorama")
    if not 0.18 <= target_spacing_fraction <= 0.55:
        raise ValueError("Render target spacing fraction is outside the safe range")
    target_spacing = target_spacing_fraction * width
    count = max(2, int(round(span / max(target_spacing, 1.0))) + 1)
    count = min(32, count)
    if maximum_keyframes > 0:
        count = min(count, maximum_keyframes)
    scores = relative_quality_scores(qualities)
    targets = np.linspace(positions.min(), positions.max(), count)
    spacing = max(float(targets[1] - targets[0]), 1.0)
    eligible = (
        np.asarray(
            [
                quality.dark_ratio <= 0.35
                and quality.saturated_ratio <= 0.20
                and quality.texture_coverage >= 0.08
                and quality.detail_strength >= 5.4
                and quality.tenengrad >= 18.0
                for quality in qualities
            ],
            dtype=bool,
        )
        if quality_gate
        else np.ones(len(qualities), dtype=bool)
    )
    selected: list[int] = []
    rejected_bins: list[int] = []
    for target_index, target in enumerate(targets):
        endpoint = target_index in {0, len(targets) - 1}
        window = (0.10 if endpoint else 0.28) * width
        candidates = np.flatnonzero(np.abs(positions - target) <= window)
        if candidates.size == 0:
            candidates = np.asarray([int(np.argmin(np.abs(positions - target)))])
        usable = candidates[eligible[candidates]]
        if usable.size == 0:
            rejected_bins.append(target_index)
            raise RuntimeError(
                "No exposure-safe, textured render frame covers one scan region"
            )
        distance = np.abs(positions[usable] - target) / max(window, 1.0)
        distance_weight = 0.05 if endpoint else 0.22
        cross_distance = (
            np.abs(cross_positions[usable] - cross_reference) / cross_scale
        )
        utility = (
            scores[usable]
            - distance_weight * np.square(distance)
            - 0.18 * np.square(cross_distance)
        )
        for chosen in usable[np.argsort(utility)[::-1]]:
            if int(chosen) not in selected:
                selected.append(int(chosen))
                break
    selected.sort()
    cursor = 0
    while cursor + 1 < len(selected):
        left, right = selected[cursor], selected[cursor + 1]
        if positions[right] - positions[left] < 0.16 * width:
            if cursor == 0:
                remove_at = cursor + 1
            elif cursor + 1 == len(selected) - 1:
                remove_at = cursor
            else:
                remove_at = cursor if scores[left] < scores[right] else cursor + 1
            selected.pop(remove_at)
            cursor = max(0, cursor - 1)
        else:
            cursor += 1
    bridge_count = 0
    cursor = 0
    while cursor + 1 < len(selected):
        left, right = selected[cursor], selected[cursor + 1]
        left_position = float(positions[left])
        right_position = float(positions[right])
        gap = right_position - left_position
        if gap > 0.66 * width:
            midpoint = 0.5 * (left_position + right_position)
            candidates = np.arange(left + 1, right)
            candidates = candidates[eligible[candidates]]
            progress_epsilon = max(1e-6, 0.002 * width)
            candidates = candidates[
                (positions[candidates] > left_position + progress_epsilon)
                & (positions[candidates] < right_position - progress_epsilon)
            ]
            if candidates.size == 0:
                raise RuntimeError(
                    "No clear render source can bridge a required overlap gap"
                )
            midpoint_distance = np.abs(positions[candidates] - midpoint)
            near_midpoint = candidates[
                midpoint_distance
                <= float(midpoint_distance.min()) + max(1.0, 0.03 * width)
            ]
            utility = scores[near_midpoint] - 0.05 * np.square(
                np.abs(positions[near_midpoint] - midpoint) / spacing
            )
            utility -= 0.18 * np.square(
                np.abs(cross_positions[near_midpoint] - cross_reference) / cross_scale
            )
            chosen = int(near_midpoint[int(np.argmax(utility))])
            chosen_position = float(positions[chosen])
            if not (
                left_position + progress_epsilon
                < chosen_position
                < right_position - progress_epsilon
            ):
                raise RuntimeError(
                    "Automatic overlap bridging did not make geometric progress"
                )
            selected.insert(cursor + 1, chosen)
            bridge_count += 1
        else:
            cursor += 1
    if len(selected) > 32:
        raise RuntimeError(
            f"Automatic render selection requires {len(selected)} keyframes, "
            "above the hard limit of 32"
        )
    if maximum_keyframes > 0 and len(selected) > maximum_keyframes:
        raise RuntimeError(
            "The configured render-frame budget cannot preserve safe overlap"
        )
    if len(selected) < 2:
        raise RuntimeError("Automatic render selection produced fewer than two sources")
    coverage = float(
        (positions[selected[-1]] - positions[selected[0]] + width)
        / max(span + width, 1.0)
    )
    if coverage < 0.95:
        raise RuntimeError(
            f"Clear render sources cover only {coverage:.1%} of the reliable scan"
        )
    return selected, {
        "mode": "dense_quality_coverage",
        "quality_gate": quality_gate,
        "target_spacing_pixels": target_spacing,
        "scan_span_pixels": span,
        "coverage_ratio": coverage,
        "selected_count": len(selected),
        "overlap_bridge_count": bridge_count,
        "maximum_adjacent_spacing_pixels": float(
            np.max(np.diff(positions[selected]))
        ),
        "rejected_quality_bins": rejected_bins,
        "quality_scores": [float(value) for value in scores],
    }


def _metric_render_eligible(
    qualities: Sequence[FrameQuality], *, quality_gate: bool
) -> np.ndarray:
    if not quality_gate:
        return np.ones(len(qualities), dtype=bool)
    return np.asarray(
        [
            quality.dark_ratio <= 0.35
            and quality.saturated_ratio <= 0.20
            and quality.texture_coverage >= 0.08
            and quality.detail_strength >= 5.4
            and quality.tenengrad >= 18.0
            for quality in qualities
        ],
        dtype=bool,
    )


def _interval_union_length(intervals: np.ndarray) -> float:
    if not len(intervals):
        return 0.0
    ordered = intervals[np.argsort(intervals[:, 0], kind="stable")]
    union = 0.0
    left, right = (float(value) for value in ordered[0])
    for next_left, next_right in ordered[1:]:
        next_left = float(next_left)
        next_right = float(next_right)
        if next_left > right:
            union += right - left
            left, right = next_left, next_right
        else:
            right = max(right, next_right)
    return union + right - left


def _projected_interval_overlap_fraction(
    first: np.ndarray, second: np.ndarray
) -> float:
    overlap = min(float(first[1]), float(second[1])) - max(
        float(first[0]), float(second[0])
    )
    shorter = min(float(first[1] - first[0]), float(second[1] - second[0]))
    return max(0.0, overlap) / max(shorter, 1e-12)


def _validate_metric_render_poses(
    camera_to_world: Sequence[np.ndarray],
) -> np.ndarray:
    centers: list[np.ndarray] = []
    for pose_index, value in enumerate(camera_to_world):
        pose = np.asarray(value, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(
                f"camera_to_world[{pose_index}] must be a finite 4x4 SE(3) matrix"
            )
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError(
                f"camera_to_world[{pose_index}] has an invalid homogeneous row"
            )
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError(
                f"camera_to_world[{pose_index}] rotation is not orthonormal"
            )
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError(
                f"camera_to_world[{pose_index}] rotation determinant is not +1"
            )
        centers.append(pose[:3, 3].copy())
    return np.stack(centers, axis=0)


def select_rgbd_render_indices_auto(
    qualities: Sequence[FrameQuality],
    camera_to_world: Sequence[np.ndarray],
    footprints: Sequence[EstimatedProjectionFootprint] | SideScanFootprintEstimate,
    *,
    maximum_keyframes: int = 0,
    quality_gate: bool = True,
    minimum_coverage_ratio: float = 0.95,
    minimum_overlap_fraction: float = 0.34,
) -> tuple[list[int], dict[str, object]]:
    """Select real metric RGB-D pose nodes with continuous projected coverage.

    The selector consumes optimized ``camera_to_world`` nodes and their measured
    projection footprints.  It never creates a transform or interpolates a pose.
    Adjacent selected sources are connected only when their real world-strip
    intervals retain the configured overlap.
    """

    if len(qualities) != len(camera_to_world) or len(qualities) < 2:
        raise ValueError(
            "RGB-D render qualities and camera_to_world poses must have equal length"
        )
    if maximum_keyframes == 1 or maximum_keyframes < 0:
        raise ValueError("maximum_keyframes must be zero or at least two")
    if not 0.90 <= minimum_coverage_ratio <= 1.0:
        raise ValueError("minimum_coverage_ratio must be between 0.90 and 1.0")
    if not 0.05 <= minimum_overlap_fraction <= 0.80:
        raise ValueError("minimum_overlap_fraction is outside the safe range")

    footprint_items = tuple(getattr(footprints, "footprints", footprints))
    if len(footprint_items) != len(qualities):
        raise ValueError(
            "RGB-D render qualities, poses, and projection footprints must align"
        )

    pose_centers = _validate_metric_render_poses(camera_to_world)
    frame_ids: list[int] = []
    scan_positions: list[float] = []
    interval_rows: list[tuple[float, float]] = []
    for index, footprint in enumerate(footprint_items):
        try:
            frame_id = int(footprint.frame_id)
            scan_position = float(footprint.camera_center_scan_x_mm)
            interval = tuple(float(value) for value in footprint.scan_x_interval_mm)
            footprint_center = np.asarray(
                footprint.camera_center_world_mm, dtype=np.float64
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Projection footprint {index} lacks finite metric coverage metadata"
            ) from exc
        if (
            len(interval) != 2
            or footprint_center.shape != (3,)
            or not np.isfinite([scan_position, *interval]).all()
            or not np.isfinite(footprint_center).all()
            or interval[1] <= interval[0]
        ):
            raise ValueError(
                f"Projection footprint {index} has invalid metric coverage metadata"
            )
        if not np.allclose(
            footprint_center, pose_centers[index], rtol=1e-9, atol=1e-3
        ):
            raise ValueError(
                f"Projection footprint {index} does not match its camera_to_world center"
            )
        frame_ids.append(frame_id)
        scan_positions.append(scan_position)
        interval_rows.append((interval[0], interval[1]))
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Projection footprint frame_id values must be unique")

    scan_axis_value = getattr(footprints, "scan_axis", None)
    up_axis_value = getattr(footprints, "up_axis", None)
    normal_axis_value = getattr(footprints, "normal_axis", None)
    cross_track_span_mm: float | None = None
    forward_span_mm: float | None = None
    if scan_axis_value is not None:
        axes = np.asarray(
            [scan_axis_value, up_axis_value, normal_axis_value], dtype=np.float64
        )
        if (
            axes.shape != (3, 3)
            or not np.isfinite(axes).all()
            or not np.allclose(axes @ axes.T, np.eye(3), atol=1e-5)
        ):
            raise ValueError("Side-scan footprint axes must be finite and orthonormal")
        expected_scan_positions = pose_centers @ axes[0]
        if not np.allclose(
            expected_scan_positions,
            np.asarray(scan_positions),
            rtol=1e-9,
            atol=1e-3,
        ):
            raise ValueError(
                "Projection footprint scan centers do not match camera_to_world poses"
            )
        cross_track_span_mm = float(np.ptp(pose_centers @ axes[1]))
        forward_span_mm = float(np.ptp(pose_centers @ axes[2]))

    positions = np.asarray(scan_positions, dtype=np.float64)
    intervals = np.asarray(interval_rows, dtype=np.float64)
    interval_widths = intervals[:, 1] - intervals[:, 0]
    center_span = float(positions[-1] - positions[0])
    monotonic_epsilon = max(1e-6, abs(center_span) * 1e-9)
    if np.any(np.diff(positions) < -monotonic_epsilon):
        raise RuntimeError(
            "Optimized RGB-D camera trajectory is not monotonic along the scan"
        )
    if center_span < 0.10 * float(np.median(interval_widths)):
        raise RuntimeError("Optimized RGB-D camera trajectory is too short for a panorama")

    reliable_left = float(np.min(intervals[:, 0]))
    reliable_right = float(np.max(intervals[:, 1]))
    reliable_span = reliable_right - reliable_left
    reliable_union = _interval_union_length(intervals)
    gap_tolerance = max(1e-6, reliable_span * 1e-6)
    if reliable_span - reliable_union > gap_tolerance:
        raise RuntimeError(
            "Reliable RGB-D projection footprints contain an uncovered scan gap"
        )

    scores = relative_quality_scores(list(qualities))
    eligible = _metric_render_eligible(qualities, quality_gate=quality_gate)
    eligible_indices = np.flatnonzero(eligible).tolist()
    if len(eligible_indices) < 2:
        raise RuntimeError(
            "Fewer than two exposure-safe, textured RGB-D render sources remain"
        )
    eligible_coverage = _interval_union_length(intervals[eligible]) / reliable_span
    if eligible_coverage < minimum_coverage_ratio:
        raise RuntimeError(
            "Clear RGB-D render sources cover only "
            f"{eligible_coverage:.1%} of the reliable projected scan"
        )

    # Each state is the shortest real-node path to one candidate.  Equal-length
    # alternatives retain the clearer path; no synthetic bridge or pose is made.
    best_solution: list[int] | None = None
    best_solution_key: tuple[float, ...] | None = None
    for start_offset, start in enumerate(eligible_indices):
        paths: dict[int, list[int]] = {start: [start]}
        for end in eligible_indices[start_offset + 1 :]:
            end_path: list[int] | None = None
            end_key: tuple[float, ...] | None = None
            for previous in eligible_indices[start_offset:]:
                if previous >= end:
                    break
                previous_path = paths.get(previous)
                if previous_path is None:
                    continue
                if positions[end] <= positions[previous] + monotonic_epsilon:
                    continue
                overlap = _projected_interval_overlap_fraction(
                    intervals[previous], intervals[end]
                )
                if overlap + 1e-12 < minimum_overlap_fraction:
                    continue
                candidate = [*previous_path, end]
                candidate_coverage = _interval_union_length(intervals[candidate])
                candidate_scores = scores[candidate]
                candidate_key = (
                    -float(len(candidate)),
                    candidate_coverage,
                    float(np.min(candidate_scores)),
                    float(np.mean(candidate_scores)),
                )
                if end_key is None or candidate_key > end_key:
                    end_path = candidate
                    end_key = candidate_key
            if end_path is not None:
                paths[end] = end_path

        for path in paths.values():
            if len(path) < 2:
                continue
            coverage = _interval_union_length(intervals[path]) / reliable_span
            if coverage + 1e-12 < minimum_coverage_ratio:
                continue
            path_scores = scores[path]
            solution_key = (
                -float(len(path)),
                float(np.min(path_scores)),
                float(np.mean(path_scores)),
                coverage,
            )
            if best_solution_key is None or solution_key > best_solution_key:
                best_solution = path
                best_solution_key = solution_key

    if best_solution is None:
        raise RuntimeError(
            "No real RGB-D render-source path preserves safe projected overlap"
        )
    if len(best_solution) > 32:
        raise RuntimeError(
            f"Automatic RGB-D render selection requires {len(best_solution)} sources, "
            "above the hard limit of 32"
        )
    if maximum_keyframes > 0 and len(best_solution) > maximum_keyframes:
        raise RuntimeError(
            "The configured RGB-D render-frame budget cannot preserve safe overlap"
        )

    selected_coverage = _interval_union_length(intervals[best_solution])
    coverage_ratio = selected_coverage / reliable_span
    overlap_fractions = [
        _projected_interval_overlap_fraction(intervals[first], intervals[second])
        for first, second in zip(best_solution, best_solution[1:])
    ]
    selected_positions = positions[best_solution]
    return best_solution, {
        "mode": "rgbd_metric_quality_coverage",
        "quality_gate": quality_gate,
        "uses_only_optimized_pose_nodes": True,
        "interpolated_pose_count": 0,
        "coverage_ratio": float(coverage_ratio),
        "minimum_coverage_ratio": minimum_coverage_ratio,
        "reliable_scan_range_mm": [reliable_left, reliable_right],
        "reliable_scan_span_mm": reliable_span,
        "selected_coverage_mm": selected_coverage,
        "camera_center_span_mm": center_span,
        "cross_track_camera_span_mm": cross_track_span_mm,
        "forward_camera_span_mm": forward_span_mm,
        "minimum_required_overlap_fraction": minimum_overlap_fraction,
        "minimum_selected_overlap_fraction": float(min(overlap_fractions)),
        "maximum_adjacent_camera_spacing_mm": float(
            np.max(np.diff(selected_positions))
        ),
        "selected_count": len(best_solution),
        "selected_frame_ids": [frame_ids[index] for index in best_solution],
        "selected_scan_centers_mm": [
            float(positions[index]) for index in best_solution
        ],
        "selected_scan_intervals_mm": [
            [float(value) for value in intervals[index]] for index in best_solution
        ],
        "eligible_source_count": int(np.count_nonzero(eligible)),
        "quality_scores": [float(value) for value in scores],
    }
