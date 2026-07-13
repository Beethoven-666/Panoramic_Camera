from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


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
