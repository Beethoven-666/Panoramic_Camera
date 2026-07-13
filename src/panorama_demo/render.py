from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CanvasInfo:
    width: int
    height: int
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    translation: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "bounds": [self.min_x, self.min_y, self.max_x, self.max_y],
            "translation": self.translation.tolist(),
        }


@dataclass(frozen=True)
class CropInfo:
    x: int
    y: int
    width: int
    height: int

    def as_dict(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class ScanRenderInfo:
    canvas: CanvasInfo
    crop: CropInfo
    source_count: int
    scan_axis: tuple[float, float]
    color_gains: tuple[tuple[float, float, float], ...]
    seam_margin: int
    multiband_levels: int
    exposure_mode: str
    seam_mask_sigma: float
    protected_regions: tuple[tuple[int, int, int, int, int], ...]
    automatic_protected_regions: tuple[tuple[int, int, int, int, int], ...]
    quality_metrics: dict[str, float | int | bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "canvas": self.canvas.as_dict(),
            "crop": self.crop.as_dict(),
            "source_count": self.source_count,
            "scan_axis": list(self.scan_axis),
            "color_gains": [list(gain) for gain in self.color_gains],
            "seam_margin": self.seam_margin,
            "multiband_levels": self.multiband_levels,
            "exposure_mode": self.exposure_mode,
            "seam_mask_sigma": self.seam_mask_sigma,
            "protected_regions": [
                {
                    "owner_index": owner,
                    "bounds": [x0, y0, x1, y1],
                }
                for owner, x0, y0, x1, y1 in self.protected_regions
            ],
            "automatic_protected_regions": [
                {
                    "owner_index": owner,
                    "bounds": [x0, y0, x1, y1],
                }
                for owner, x0, y0, x1, y1 in self.automatic_protected_regions
            ],
            "quality_metrics": self.quality_metrics,
        }


def _normalize_homography(matrix: np.ndarray) -> np.ndarray:
    homography = np.asarray(matrix, dtype=np.float64)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("Each transform must be a finite 3x3 matrix")
    if abs(homography[2, 2]) < 1e-12:
        raise ValueError("Degenerate homography")
    return homography / homography[2, 2]


def compute_canvas(
    images: list[np.ndarray], transforms: list[np.ndarray], max_megapixels: float
) -> CanvasInfo:
    if not images or len(images) != len(transforms):
        raise ValueError("images and transforms must be non-empty and have equal length")

    all_corners: list[np.ndarray] = []
    for image, transform in zip(images, transforms, strict=True):
        height, width = image.shape[:2]
        corners = np.array(
            [[[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]]],
            dtype=np.float32,
        )
        warped = cv2.perspectiveTransform(corners, _normalize_homography(transform))[0]
        if not np.isfinite(warped).all():
            raise ValueError("A transform produced non-finite canvas corners")
        all_corners.append(warped)

    stacked = np.concatenate(all_corners, axis=0)
    min_x = float(np.floor(stacked[:, 0].min()))
    min_y = float(np.floor(stacked[:, 1].min()))
    max_x = float(np.ceil(stacked[:, 0].max()))
    max_y = float(np.ceil(stacked[:, 1].max()))
    width = max(1, int(max_x - min_x))
    height = max(1, int(max_y - min_y))
    megapixels = width * height / 1_000_000.0
    if megapixels > max_megapixels:
        raise MemoryError(
            f"Requested canvas is {width}x{height} ({megapixels:.1f} MP), "
            f"above the {max_megapixels:.1f} MP demo limit"
        )
    translation = np.array([[1.0, 0.0, -min_x], [0.0, 1.0, -min_y], [0.0, 0.0, 1.0]])
    return CanvasInfo(width, height, min_x, min_y, max_x, max_y, translation)


def _base_feather_mask(height: int, width: int, feather_pixels: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.uint8)
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    scale = max(1, feather_pixels)
    return np.clip(distance / float(scale), 0.03, 1.0).astype(np.float32)


def largest_valid_rectangle(mask: np.ndarray) -> CropInfo:
    """Return the largest axis-aligned rectangle containing only valid pixels."""

    valid = np.asarray(mask, dtype=bool)
    if valid.ndim != 2:
        raise ValueError("Valid mask must be two-dimensional")
    if not valid.any():
        raise ValueError("Valid mask contains no usable pixels")

    height, width = valid.shape
    histogram = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = CropInfo(0, 0, 1, 1)
    for y in range(height):
        histogram = np.where(valid[y], histogram + 1, 0)
        stack: list[tuple[int, int]] = []
        for x in range(width + 1):
            current = int(histogram[x]) if x < width else 0
            start = x
            while stack and stack[-1][1] > current:
                left, rectangle_height = stack.pop()
                rectangle_width = x - left
                area = rectangle_height * rectangle_width
                candidate = CropInfo(
                    left,
                    y - rectangle_height + 1,
                    rectangle_width,
                    rectangle_height,
                )
                if (
                    area > best_area
                    or (
                        area == best_area
                        and (
                            candidate.width,
                            -candidate.y,
                            -candidate.x,
                        )
                        > (best.width, -best.y, -best.x)
                    )
                ):
                    best_area = area
                    best = candidate
                start = left
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    return best


def _center_patch_color_gains(images: list[np.ndarray]) -> np.ndarray:
    """Balance frames using the area that contributes to a centre-strip mosaic."""

    medians: list[np.ndarray] = []
    for image in images:
        height, width = image.shape[:2]
        y0, y1 = int(round(height * 0.0625)), int(round(height * 0.375))
        x0, x1 = int(round(width * 0.39)), int(round(width * 0.61))
        patch = image[y0:y1, x0:x1]
        if patch.size == 0:
            raise ValueError("Image is too small for scan exposure balancing")
        medians.append(np.median(patch.reshape(-1, 3), axis=0))
    values = np.asarray(medians, dtype=np.float64)
    target = np.median(values, axis=0)
    return np.clip(target[None, :] / np.maximum(values, 1.0), 0.82, 1.18)


def _scan_axis(centers: np.ndarray) -> np.ndarray:
    if centers.shape[0] < 2:
        return np.array([1.0, 0.0], dtype=np.float64)
    centered = centers - centers.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0].astype(np.float64)
    direction = centers[-1] - centers[0]
    if float(np.dot(axis, direction)) < 0.0:
        axis *= -1.0
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return axis / norm


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _expand_depth_occlusion_surface(
    depth: np.ndarray,
    seeds: np.ndarray,
    common: np.ndarray,
) -> np.ndarray:
    """Grow occlusion-edge seeds across their local physical surface.

    A depth mismatch is normally only visible on the two exposed edges of a
    foreground object.  Growing those edges across pixels at the same depth
    keeps the seam out of the object's interior.  The growth is deliberately
    local: unrestricted connected-depth growth can turn an entire desk or wall
    into one risk component because consumer depth maps are noisy on edges.
    """

    expanded = np.zeros(common.shape, dtype=bool)
    count, seed_labels, seed_stats, _ = cv2.connectedComponentsWithStats(
        seeds.astype(np.uint8), connectivity=8
    )
    reach = max(9, min(65, int(round(depth.shape[1] * 0.05))))
    reach_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (reach * 2 + 1, reach * 2 + 1)
    )
    for seed_label in range(1, count):
        if int(seed_stats[seed_label, cv2.CC_STAT_AREA]) < 4:
            continue
        x = int(seed_stats[seed_label, cv2.CC_STAT_LEFT])
        y = int(seed_stats[seed_label, cv2.CC_STAT_TOP])
        width = int(seed_stats[seed_label, cv2.CC_STAT_WIDTH])
        height = int(seed_stats[seed_label, cv2.CC_STAT_HEIGHT])
        x0, y0 = max(0, x - reach), max(0, y - reach)
        x1 = min(depth.shape[1], x + width + reach)
        y1 = min(depth.shape[0], y + height + reach)
        seed = seed_labels[y0:y1, x0:x1] == seed_label
        depth_roi = depth[y0:y1, x0:x1]
        common_roi = common[y0:y1, x0:x1]
        seed_values = depth_roi[seed]
        if not seed_values.size:
            continue
        surface_depth = float(np.median(seed_values))
        tolerance = max(60.0, surface_depth * 0.08)
        similar = (
            common_roi
            & (depth_roi >= 400.0)
            & (np.abs(depth_roi - surface_depth) <= tolerance)
        )
        local_reach = cv2.dilate(seed.astype(np.uint8), reach_kernel) > 0
        similar &= local_reach
        similar = cv2.morphologyEx(
            similar.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        surface_count, surface_labels = cv2.connectedComponents(similar)
        touched = np.unique(surface_labels[seed & (surface_labels > 0)])
        for surface_label in touched:
            expanded[y0:y1, x0:x1] |= surface_labels == surface_label
    return expanded


def _pair_risk_mask(
    image0: np.ndarray,
    image1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    depth0: np.ndarray | None,
    depth1: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = (valid0 > 0) & (valid1 > 0)
    lab0 = cv2.cvtColor(image0, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB).astype(np.float32)
    difference = np.linalg.norm(lab0 - lab1, axis=2)
    values = difference[common]
    if values.size:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = max(18.0, median + 3.5 * max(mad, 1.0))
    else:
        threshold = 255.0
    gradient0 = _gradient_magnitude(image0)
    gradient1 = _gradient_magnitude(image1)
    gradient_values = np.concatenate((gradient0[common], gradient1[common]))
    gradient_threshold = (
        max(18.0, float(np.percentile(gradient_values, 65)))
        if gradient_values.size
        else 255.0
    )
    rgb_risk = (
        common
        & (difference >= threshold)
        & ((gradient0 >= gradient_threshold) | (gradient1 >= gradient_threshold))
    )

    depth_risk = np.zeros(common.shape, dtype=bool)
    if depth0 is not None and depth1 is not None:
        depth_valid = common & (depth0 >= 400.0) & (depth1 >= 400.0)
        minimum = np.minimum(depth0, depth1)
        # A 100 mm / 10% threshold rejects ordinary sensor noise while still
        # detecting the disparity created by a 0.5 m foreground object.
        tolerance = np.maximum(100.0, minimum * 0.10)
        first_is_nearer = depth_valid & ((depth1 - depth0) >= tolerance)
        second_is_nearer = depth_valid & ((depth0 - depth1) >= tolerance)
        # In a textured scene, isolated depth mismatches are common on shiny
        # and low-confidence surfaces.  Only promote them when the colour views
        # also show local disagreement.  A genuinely textureless scene keeps
        # the depth-only path so foreground silhouettes remain protected.
        flat_scene = bool(
            gradient_values.size
            and float(np.percentile(gradient_values, 90)) < 6.0
            and (not values.size or float(np.percentile(values, 90)) < 6.0)
        )
        depth_support = common
        if not flat_scene:
            visual_seed = common & (
                difference >= max(12.0, threshold * 0.45)
            )
            visual_seed |= rgb_risk
            depth_support = cv2.dilate(
                visual_seed.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            ) > 0
            first_is_nearer &= depth_support
            second_is_nearer &= depth_support
        depth_risk |= first_is_nearer | second_is_nearer
        # RGB residuals already outline the full visible surface in a textured
        # scene; expanding every depth edge there tends to merge a whole desk
        # into one false obstacle.  Depth-only scenes do need surface growth to
        # bridge the two exposed sides of an otherwise invisible object.
        if flat_scene:
            depth_risk |= _expand_depth_occlusion_surface(
                depth0, first_is_nearer, common
            )
            depth_risk |= _expand_depth_occlusion_surface(
                depth1, second_is_nearer, common
            )
        else:
            saturation0 = cv2.cvtColor(image0, cv2.COLOR_BGR2HSV)[:, :, 1]
            saturation1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)[:, :, 1]
            salient_color = common & (
                (saturation0 >= 80) | (saturation1 >= 80)
            )
            salient_support = cv2.dilate(
                salient_color.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
            ) > 0
            depth_risk |= _expand_depth_occlusion_surface(
                depth0, first_is_nearer & salient_color, salient_support
            )
            depth_risk |= _expand_depth_occlusion_surface(
                depth1, second_is_nearer & salient_color, salient_support
            )

    risk = (rgb_risk | depth_risk).astype(np.uint8) * 255
    risk = cv2.morphologyEx(
        risk,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    risk = cv2.dilate(
        risk,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    risk[~common] = 0
    safe_background = (
        common
        & (gradient0 < gradient_threshold)
        & (gradient1 < gradient_threshold)
    )
    return risk, difference, safe_background


def _longest_true_run(values: np.ndarray) -> tuple[int, int]:
    """Return the half-open bounds of the longest contiguous true run."""

    best_start = best_end = run_start = 0
    in_run = False
    for index, value in enumerate(np.r_[values.astype(bool), False]):
        if value and not in_run:
            run_start = index
            in_run = True
        elif not value and in_run:
            if index - run_start > best_end - best_start:
                best_start, best_end = run_start, index
            in_run = False
    return best_start, best_end


def _monotonic_pair_seam(
    image0: np.ndarray,
    image1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    depth0: np.ndarray | None,
    depth1: np.ndarray | None,
    axis: np.ndarray,
    midpoint_projection: float,
    seam_margin: int,
    blend_guard_pixels: int,
) -> tuple[np.ndarray, bool, dict[str, float]]:
    """Find one foreground-avoiding seam that is a graph over the cross axis.

    The returned vector stores the seam's scan-axis projection for every row
    (horizontal scans) or column (vertical scans).  Restricting the cut to one
    value per cross-axis coordinate prevents GraphCut islands and makes the
    final owner partition deterministic.
    """

    risk, difference, safe_background = _pair_risk_mask(
        image0, image1, valid0, valid1, depth0, depth1
    )
    common = (valid0 > 0) & (valid1 > 0)
    horizontal = abs(float(axis[0])) >= abs(float(axis[1]))
    if horizontal:
        primary_axis, cross_axis = float(axis[0]), float(axis[1])
        common_work = common
        risk_work = risk > 0
        difference_work = difference
        safe_work = safe_background
        edge_work = np.maximum(
            _gradient_magnitude(image0), _gradient_magnitude(image1)
        )
    else:
        primary_axis, cross_axis = float(axis[1]), float(axis[0])
        common_work = common.T
        risk_work = (risk > 0).T
        difference_work = difference.T
        safe_work = safe_background.T
        edge_work = np.maximum(
            _gradient_magnitude(image0), _gradient_magnitude(image1)
        ).T
    # Join the two exposed sides of a small object without horizontally
    # swallowing unrelated desk/wall surfaces.  In particular this must not
    # scale with the (often hundreds of pixels wide) seam search corridor.
    bridge_width = min(31, max(7, int(round(common_work.shape[1] * 0.012))))
    if bridge_width % 2 == 0:
        bridge_width += 1
    risk_work = cv2.morphologyEx(
        risk_work.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_width, 3)),
    ) > 0
    if abs(primary_axis) < 1e-6:
        raise RuntimeError("Scan axis is degenerate for monotonic seam search")

    cross_size, primary_size = common_work.shape
    cross_coordinates = np.arange(cross_size, dtype=np.float32)
    primary_coordinates = np.arange(primary_size, dtype=np.float32)
    pixel_projection = (
        cross_coordinates[:, None] * cross_axis
        + primary_coordinates[None, :] * primary_axis
    )
    allowed = common_work & (
        np.abs(pixel_projection - midpoint_projection) <= max(1, seam_margin)
    )
    run_start, run_end = _longest_true_run(np.any(allowed, axis=1))
    if run_end <= run_start:
        raise RuntimeError("Adjacent scan sources have no continuous seam corridor")

    # Prefer low inter-view residual and weak image edges.  Foreground/depth
    # disagreement is a large, but finite, penalty so diagnostics can reject a
    # physically impossible cut instead of producing a hole.
    center_penalty = np.abs(pixel_projection - midpoint_projection) * 0.04
    if blend_guard_pixels > 0:
        safe_distance = cv2.distanceTransform(
            (~risk_work).astype(np.uint8), cv2.DIST_L2, 3
        )
        blend_guard_penalty = (
            np.clip(
                (float(blend_guard_pixels) - safe_distance)
                / float(blend_guard_pixels),
                0.0,
                1.0,
            )
            * 180.0
        )
    else:
        blend_guard_penalty = np.zeros(risk_work.shape, dtype=np.float32)
    cost = (
        np.minimum(difference_work, 120.0)
        + np.minimum(edge_work, 160.0) * 0.08
        + (~safe_work).astype(np.float32) * 6.0
        + risk_work.astype(np.float32) * 1500.0
        + blend_guard_penalty
        + center_penalty
    ).astype(np.float32)
    cost[~allowed] = np.inf

    row_count = run_end - run_start
    predecessor = np.full((row_count, primary_size), -1, dtype=np.int32)
    accumulated = cost[run_start].copy()
    if not np.isfinite(accumulated).any():
        raise RuntimeError("The first seam row has no valid overlap")
    nominal_slope = abs(cross_axis / primary_axis)
    max_step = max(3, int(np.ceil(nominal_slope)) + 2)
    primary_index = np.arange(primary_size)
    for local_row, work_row in enumerate(range(run_start + 1, run_end), start=1):
        best = np.full(primary_size, np.inf, dtype=np.float32)
        best_predecessor = np.full(primary_size, -1, dtype=np.int32)
        for delta in range(-max_step, max_step + 1):
            previous_index = primary_index + delta
            usable = (previous_index >= 0) & (previous_index < primary_size)
            candidate = np.full(primary_size, np.inf, dtype=np.float32)
            candidate[usable] = (
                accumulated[previous_index[usable]] + 1.25 * abs(delta)
            )
            take = candidate < best
            best[take] = candidate[take]
            best_predecessor[take] = previous_index[take]
        accumulated = cost[work_row] + best
        accumulated[~allowed[work_row]] = np.inf
        predecessor[local_row] = best_predecessor
        if not np.isfinite(accumulated).any():
            raise RuntimeError("The overlap contains no continuous monotonic seam")

    endpoint = int(np.argmin(accumulated))
    run_path = np.empty(row_count, dtype=np.int32)
    run_path[-1] = endpoint
    for local_row in range(row_count - 1, 0, -1):
        endpoint = int(predecessor[local_row, endpoint])
        if endpoint < 0:
            raise RuntimeError("Monotonic seam backtracking failed")
        run_path[local_row - 1] = endpoint

    nominal = np.rint(
        (midpoint_projection - cross_coordinates * cross_axis) / primary_axis
    ).astype(np.int32)
    full_path = np.clip(nominal, 0, primary_size - 1)
    full_path[run_start:run_end] = run_path
    if run_start:
        offset = int(run_path[0] - nominal[run_start])
        full_path[:run_start] = np.clip(
            nominal[:run_start] + offset, 0, primary_size - 1
        )
    if run_end < cross_size:
        offset = int(run_path[-1] - nominal[run_end - 1])
        full_path[run_end:] = np.clip(
            nominal[run_end:] + offset, 0, primary_size - 1
        )

    seam_projection = (
        full_path.astype(np.float32) * primary_axis
        + cross_coordinates * cross_axis
    )
    run_rows = np.arange(run_start, run_end)
    run_columns = run_path
    # This is evaluated on the actual owner boundary.  Do not dilate again
    # here: `_pair_risk_mask` already includes a safety margin and a second
    # dilation would report a nearby, successfully avoided foreground as a
    # crossed foreground.
    crossed_risk = risk_work[run_rows, run_columns]
    safe_samples = ~crossed_risk
    residuals = difference_work[run_rows[safe_samples], run_columns[safe_samples]]
    if residuals.size == 0:
        residuals = np.asarray([255.0], dtype=np.float32)
    return seam_projection, horizontal, {
        "risk_fraction": float(np.mean(crossed_risk)),
        "safe_residual_p95": float(np.percentile(residuals, 95)),
        "cross_coverage_ratio": float(row_count / max(1, cross_size)),
    }


def _monotonic_owner_masks(
    warped_images: list[np.ndarray],
    warped_valid: list[np.ndarray],
    warped_depths: list[np.ndarray | None],
    center_array: np.ndarray,
    axis: np.ndarray,
    order: np.ndarray,
    seam_margin: int,
    blend_guard_pixels: int,
) -> tuple[list[np.ndarray], dict[str, float | int | bool]]:
    """Build a strict, hole-free owner partition from adjacent DP seams."""

    projections = center_array @ axis
    seam_rows: list[np.ndarray] = []
    seam_metrics: list[dict[str, float]] = []
    horizontal: bool | None = None
    for first, second in zip(order[:-1], order[1:], strict=True):
        first_index, second_index = int(first), int(second)
        midpoint = float(
            (projections[first_index] + projections[second_index]) * 0.5
        )
        seam, pair_horizontal, metrics = _monotonic_pair_seam(
            warped_images[first_index],
            warped_images[second_index],
            warped_valid[first_index],
            warped_valid[second_index],
            warped_depths[first_index],
            warped_depths[second_index],
            axis,
            midpoint,
            seam_margin,
            blend_guard_pixels,
        )
        if horizontal is not None and horizontal != pair_horizontal:
            raise RuntimeError("Adjacent seam orientations are inconsistent")
        horizontal = pair_horizontal
        seam_rows.append(seam)
        seam_metrics.append(metrics)

    seam_matrix = np.stack(seam_rows, axis=0)
    # A pair's search band can overlap the next pair's band.  Enforce ordered
    # thresholds without changing the foreground-aware path in normal cases.
    seam_matrix = np.maximum.accumulate(seam_matrix, axis=0)
    height, width = warped_valid[0].shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    pixel_projection = xx * float(axis[0]) + yy * float(axis[1])
    rank = np.zeros((height, width), dtype=np.int16)
    if horizontal:
        for seam in seam_matrix:
            rank += pixel_projection >= seam[:, None]
    else:
        for seam in seam_matrix:
            rank += pixel_projection >= seam[None, :]
    rank = np.clip(rank, 0, len(order) - 1)
    owner = order[rank].astype(np.int16)

    valid_stack = np.stack([mask > 0 for mask in warped_valid], axis=0)
    union_valid = np.any(valid_stack, axis=0)
    rows, columns = np.indices((height, width))
    assigned_valid = valid_stack[owner, rows, columns]
    needs_fill = union_valid & ~assigned_valid
    if np.any(needs_fill):
        best_distance = np.full((height, width), np.inf, dtype=np.float32)
        replacement = np.full((height, width), -1, dtype=np.int16)
        for index, projection in enumerate(projections):
            distance = np.abs(pixel_projection - float(projection))
            take = needs_fill & valid_stack[index] & (distance < best_distance)
            best_distance[take] = distance[take]
            replacement[take] = index
        if np.any(replacement[needs_fill] < 0):
            raise RuntimeError("Monotonic seam partition left an unowned valid pixel")
        owner[needs_fill] = replacement[needs_fill]
    owner[~union_valid] = -1

    masks = [
        np.where((owner == index) & valid_stack[index], 255, 0).astype(np.uint8)
        for index in range(len(warped_images))
    ]
    partition_count = np.sum(np.stack([mask > 0 for mask in masks]), axis=0)
    if np.any(partition_count[union_valid] != 1):
        raise RuntimeError("Monotonic seam masks are not a strict owner partition")
    risks = np.asarray([item["risk_fraction"] for item in seam_metrics])
    residuals = np.asarray([item["safe_residual_p95"] for item in seam_metrics])
    coverages = np.asarray([item["cross_coverage_ratio"] for item in seam_metrics])
    return masks, {
        "automatic_protected_component_count": 0,
        "automatic_dp_seam_count": len(seam_metrics),
        "seam_risk_fraction": float(np.max(risks)),
        "safe_seam_lab_residual_p95": float(np.max(residuals)),
        "minimum_seam_cross_coverage_ratio": float(np.min(coverages)),
        "strict_owner_partition": True,
    }


def _normalized_multiband_blend(
    images: list[np.ndarray],
    masks: list[np.ndarray],
    levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend full-canvas sources with normalized pyramid weights.

    OpenCV's streaming ``MultiBandBlender`` can produce zero-weight wedges
    when a strict monotonic partition pinches to a narrow region at one pyramid
    level.  Building and normalizing the weight pyramid explicitly guarantees
    that every pixel owned at level zero remains covered in the result.
    """

    if not images or len(images) != len(masks):
        raise ValueError("Multiband images and masks must be non-empty and aligned")
    height, width = masks[0].shape
    effective_levels = max(1, int(levels))
    while effective_levels > 1 and min(height, width) < 2 ** effective_levels:
        effective_levels -= 1

    sample_shapes: list[tuple[int, int]] = [(height, width)]
    for _ in range(1, effective_levels):
        previous_height, previous_width = sample_shapes[-1]
        sample_shapes.append(
            ((previous_height + 1) // 2, (previous_width + 1) // 2)
        )
    accumulators = [
        np.zeros((level_height, level_width, 3), dtype=np.float32)
        for level_height, level_width in sample_shapes
    ]
    weight_sums = [
        np.zeros((level_height, level_width), dtype=np.float32)
        for level_height, level_width in sample_shapes
    ]

    for image, mask in zip(images, masks, strict=True):
        gaussian_images = [image.astype(np.float32)]
        gaussian_masks = [mask.astype(np.float32) / 255.0]
        for _ in range(1, effective_levels):
            gaussian_images.append(cv2.pyrDown(gaussian_images[-1]))
            gaussian_masks.append(cv2.pyrDown(gaussian_masks[-1]))
        laplacians: list[np.ndarray] = []
        for level in range(effective_levels - 1):
            expanded = cv2.pyrUp(
                gaussian_images[level + 1],
                dstsize=(
                    gaussian_images[level].shape[1],
                    gaussian_images[level].shape[0],
                ),
            )
            laplacians.append(gaussian_images[level] - expanded)
        laplacians.append(gaussian_images[-1])
        for level, (laplacian, weight) in enumerate(
            zip(laplacians, gaussian_masks, strict=True)
        ):
            accumulators[level] += laplacian * weight[..., None]
            weight_sums[level] += weight

    blended_levels: list[np.ndarray] = []
    for accumulator, weight_sum in zip(accumulators, weight_sums, strict=True):
        blended_level = np.zeros_like(accumulator)
        valid = weight_sum > 1e-6
        blended_level[valid] = accumulator[valid] / weight_sum[valid, None]
        blended_levels.append(blended_level)

    blended = blended_levels[-1]
    for level in range(effective_levels - 2, -1, -1):
        blended = cv2.pyrUp(
            blended,
            dstsize=(blended_levels[level].shape[1], blended_levels[level].shape[0]),
        )
        blended += blended_levels[level]
    output_mask = np.where(weight_sums[0] > 1e-6, 255, 0).astype(np.uint8)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8), output_mask


def _evaluate_final_owner_boundaries(
    images: list[np.ndarray],
    valid_masks: list[np.ndarray],
    depth_maps: list[np.ndarray | None],
    owner_masks: list[np.ndarray],
    axis: np.ndarray,
    order: np.ndarray,
    blend_guard_pixels: int,
) -> dict[str, float | int | bool]:
    """Measure risk on the final masks after overrides and hole filling."""

    owned = np.stack([mask > 0 for mask in owner_masks], axis=0)
    valid = np.stack([mask > 0 for mask in valid_masks], axis=0)
    union_valid = np.any(valid, axis=0)
    owner_count = np.sum(owned, axis=0)
    hole_count = int(np.count_nonzero(union_valid & (owner_count == 0)))
    overlap_count = int(np.count_nonzero(owner_count > 1))
    horizontal = abs(float(axis[0])) >= abs(float(axis[1]))
    neighbor_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    exact_risks: list[float] = []
    guard_risks: list[float] = []
    residuals: list[float] = []
    coverages: list[float] = []
    boundary_pixels = 0
    boundary_pair_count = 0
    nonadjacent_boundary_pixels = 0
    unsupported_boundary_pixels = 0
    order_rank = np.empty(len(order), dtype=np.int16)
    order_rank[order] = np.arange(len(order), dtype=np.int16)

    for first in range(len(images)):
        for second in range(first + 1, len(images)):
            first_touches_second = owned[first] & (
                cv2.dilate(owned[second].astype(np.uint8), neighbor_kernel) > 0
            )
            second_touches_first = owned[second] & (
                cv2.dilate(owned[first].astype(np.uint8), neighbor_kernel) > 0
            )
            common = valid[first] & valid[second]
            topology_boundary = first_touches_second | second_touches_first
            if not np.any(topology_boundary):
                continue
            pair_pixel_count = int(np.count_nonzero(topology_boundary))
            if abs(int(order_rank[first]) - int(order_rank[second])) != 1:
                nonadjacent_boundary_pixels += pair_pixel_count
                continue
            boundary_pair_count += 1
            boundary_pixels += pair_pixel_count
            common_guard = cv2.dilate(
                common.astype(np.uint8), neighbor_kernel
            ) > 0
            unsupported_boundary_pixels += int(
                np.count_nonzero(topology_boundary & ~common_guard)
            )
            boundary = topology_boundary & common
            if not np.any(boundary):
                exact_risks.append(1.0)
                guard_risks.append(1.0)
                residuals.append(255.0)
                coverages.append(0.0)
                continue
            risk, difference, _ = _pair_risk_mask(
                images[first],
                images[second],
                valid_masks[first],
                valid_masks[second],
                depth_maps[first],
                depth_maps[second],
            )
            risk_bool = risk > 0
            exact_risks.append(float(np.mean(risk_bool[boundary])))
            if blend_guard_pixels > 0 and np.any(risk_bool):
                distance_to_risk = cv2.distanceTransform(
                    (~risk_bool).astype(np.uint8), cv2.DIST_L2, 3
                )
                guard_risks.append(
                    float(
                        np.mean(
                            distance_to_risk[boundary]
                            <= float(blend_guard_pixels)
                        )
                    )
                )
            else:
                guard_risks.append(exact_risks[-1])
            safe_boundary = boundary & ~risk_bool
            values = difference[safe_boundary]
            residuals.append(
                float(np.percentile(values, 95)) if values.size else 255.0
            )
            cross_boundary = np.any(boundary, axis=1 if horizontal else 0)
            cross_common = np.any(common, axis=1 if horizontal else 0)
            denominator = max(1, int(np.count_nonzero(cross_common)))
            coverages.append(float(np.count_nonzero(cross_boundary) / denominator))

    return {
        "automatic_protected_component_count": 0,
        "automatic_dp_seam_count": max(0, len(order) - 1),
        "final_owner_boundary_pair_count": boundary_pair_count,
        "final_owner_boundary_pixel_count": boundary_pixels,
        "nonadjacent_owner_boundary_pixel_count": nonadjacent_boundary_pixels,
        "unsupported_owner_boundary_pixel_count": unsupported_boundary_pixels,
        "seam_risk_fraction": float(max(exact_risks, default=0.0)),
        "blend_guard_risk_fraction": float(max(guard_risks, default=0.0)),
        "safe_seam_lab_residual_p95": float(max(residuals, default=0.0)),
        "minimum_seam_cross_coverage_ratio": float(min(coverages, default=1.0)),
        "unowned_valid_pixel_count": hole_count,
        "multiple_owner_pixel_count": overlap_count,
        "strict_owner_partition": hole_count == 0 and overlap_count == 0,
    }


def render_scan_panorama(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    *,
    max_megapixels: float = 250.0,
    seam_margin: int = 220,
    multiband_levels: int = 4,
    balance_color: bool = True,
    exposure_mode: str = "center_gain",
    seam_mask_sigma: float = 0.0,
    protected_regions: list[tuple[int, int, int, int, int]] | None = None,
    depth_maps_mm: list[np.ndarray | None] | None = None,
    auto_foreground: bool = True,
    quality_gate: bool = True,
) -> tuple[np.ndarray, ScanRenderInfo]:
    """Render a side scan with one owner per region and seams around foreground objects.

    Unlike :func:`render_panorama`, this renderer does not average every frame that
    covers a pixel.  Frames first receive a narrow territory along the scan axis.
    The automatic path finds depth/RGB-risk-aware monotonic seams; the explicit
    compatibility path uses graph cut.  Multiband blending is limited to those
    owner boundaries.
    """

    if not images or len(images) != len(transforms):
        raise ValueError("images and transforms must be non-empty and have equal length")
    if seam_margin < 0:
        raise ValueError("seam_margin cannot be negative")
    if multiband_levels < 1:
        raise ValueError("multiband_levels must be at least one")
    if seam_mask_sigma < 0.0:
        raise ValueError("seam_mask_sigma cannot be negative")
    if len(images) > 32:
        raise ValueError("Scan renderer expects at most 32 selected keyframes")
    if depth_maps_mm is not None and len(depth_maps_mm) != len(images):
        raise ValueError("depth_maps_mm must match the selected image count")
    effective_exposure_mode = exposure_mode if balance_color else "none"
    if effective_exposure_mode not in {"none", "center_gain", "global_gain"}:
        raise ValueError(
            "exposure_mode must be one of: none, center_gain, global_gain"
        )
    requested_region_rows = tuple(protected_regions or ())

    canvas = compute_canvas(images, transforms, max_megapixels)
    normalized_regions: list[tuple[int, int, int, int, int]] = []
    for owner, x0, y0, x1, y1 in requested_region_rows:
        if not 0 <= owner < len(images):
            raise ValueError(f"Protected-region owner index is out of range: {owner}")
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Protected-region bounds must have positive area")
        clipped = (
            owner,
            max(0, x0),
            max(0, y0),
            min(canvas.width, x1),
            min(canvas.height, y1),
        )
        if clipped[1] >= clipped[3] or clipped[2] >= clipped[4]:
            raise ValueError("A protected region does not intersect the scan canvas")
        normalized_regions.append(clipped)
    region_rows = tuple(normalized_regions)
    aggregate_megapixels = (
        canvas.width * canvas.height * len(images) / 1_000_000.0
    )
    if aggregate_megapixels > max_megapixels:
        raise MemoryError(
            "Scan renderer working set is "
            f"{canvas.width}x{canvas.height} x {len(images)} sources "
            f"({aggregate_megapixels:.1f} aggregate MP), above the "
            f"{max_megapixels:.1f} MP demo limit"
        )
    gains = (
        _center_patch_color_gains(images)
        if effective_exposure_mode == "center_gain"
        else np.ones((len(images), 3), dtype=np.float64)
    )
    warped_images: list[np.ndarray] = []
    warped_valid: list[np.ndarray] = []
    warped_depths: list[np.ndarray | None] = []
    centers: list[np.ndarray] = []
    depth_inputs = depth_maps_mm or [None] * len(images)
    for image, transform, gain, depth in zip(
        images, transforms, gains, depth_inputs, strict=True
    ):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Renderer expects BGR uint8 images")
        height, width = image.shape[:2]
        canvas_h = canvas.translation @ _normalize_homography(transform)
        adjusted = np.clip(
            image.astype(np.float32) * gain[None, None, :], 0.0, 255.0
        ).astype(np.uint8)
        warped_images.append(
            cv2.warpPerspective(
                adjusted,
                canvas_h,
                (canvas.width, canvas.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        )
        warped_valid.append(
            cv2.warpPerspective(
                np.full((height, width), 255, dtype=np.uint8),
                canvas_h,
                (canvas.width, canvas.height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
            )
        )
        if depth is None:
            warped_depths.append(None)
        else:
            depth_value = np.asarray(depth, dtype=np.float32)
            if depth_value.shape != (height, width):
                raise ValueError("Each aligned depth map must match its color image")
            warped_depths.append(
                cv2.warpPerspective(
                    depth_value,
                    canvas_h,
                    (canvas.width, canvas.height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            )
        center = canvas_h @ np.array([width * 0.5, height * 0.5, 1.0])
        centers.append(center[:2] / center[2])

    if effective_exposure_mode == "global_gain" and len(images) > 1:
        compensator = cv2.detail.ExposureCompensator_createDefault(
            cv2.detail.ExposureCompensator_GAIN
        )
        compensator.feed(
            [(0, 0)] * len(warped_images),
            warped_images,
            warped_valid,
        )
        for index, (warped, valid) in enumerate(
            zip(warped_images, warped_valid, strict=True)
        ):
            compensator.apply(index, (0, 0), warped, valid)
        scalar_gains = np.asarray(
            [
                float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
                for value in compensator.getMatGains()
            ],
            dtype=np.float64,
        )
        gains = np.repeat(scalar_gains[:, None], 3, axis=1)

    center_array = np.asarray(centers, dtype=np.float64)
    axis = _scan_axis(center_array)
    if len(images) == 1:
        crop = largest_valid_rectangle(warped_valid[0] > 0)
        panorama = warped_images[0][
            crop.y : crop.y + crop.height,
            crop.x : crop.x + crop.width,
        ]
        source_height = float(images[0].shape[0])
        crop_height_ratio = crop.height / max(1.0, source_height)
        crop_width_ratio = crop.width / max(1.0, float(canvas.width))
        gain_min = float(np.min(gains))
        gain_max = float(np.max(gains))
        failure_reasons: list[str] = []
        if crop_height_ratio < 0.90:
            failure_reasons.append("less than 90% of source height remains after crop")
        if crop_width_ratio < 0.95:
            failure_reasons.append("less than 95% of the scan canvas remains after crop")
        if gain_min < 0.45 or gain_max > 2.20:
            failure_reasons.append("exposure compensation gain is outside the safe range")
        quality_metrics: dict[str, float | int | bool] = {
            "quality_pass": not failure_reasons,
            "crop_height_ratio": float(crop_height_ratio),
            "crop_width_ratio": float(crop_width_ratio),
            "exposure_gain_min": gain_min,
            "exposure_gain_max": gain_max,
            "final_owner_boundary_pair_count": 0,
            "nonadjacent_owner_boundary_pixel_count": 0,
            "unsupported_owner_boundary_pixel_count": 0,
            "strict_owner_partition": True,
        }
        if quality_gate and failure_reasons:
            raise RuntimeError(
                "Scan render quality gate failed: " + "; ".join(failure_reasons)
            )
        info = ScanRenderInfo(
            canvas=canvas,
            crop=crop,
            source_count=1,
            scan_axis=(float(axis[0]), float(axis[1])),
            color_gains=tuple(tuple(float(value) for value in gain) for gain in gains),
            seam_margin=seam_margin,
            multiband_levels=multiband_levels,
            exposure_mode=effective_exposure_mode,
            seam_mask_sigma=seam_mask_sigma,
            protected_regions=region_rows,
            automatic_protected_regions=(),
            quality_metrics=quality_metrics,
        )
        return panorama, info
    projections = center_array @ axis
    order = np.argsort(projections)
    sorted_projections = projections[order]
    yy, xx = np.indices((canvas.height, canvas.width), dtype=np.float32)
    pixel_projection = xx * float(axis[0]) + yy * float(axis[1])

    allowed_masks: list[np.ndarray | None] = [None] * len(images)
    for rank, frame_index in enumerate(order):
        lower = -np.inf
        upper = np.inf
        if rank > 0:
            lower = (
                sorted_projections[rank - 1] + sorted_projections[rank]
            ) * 0.5 - seam_margin
        if rank + 1 < len(order):
            upper = (
                sorted_projections[rank] + sorted_projections[rank + 1]
            ) * 0.5 + seam_margin
        territory = ((pixel_projection >= lower) & (pixel_projection <= upper)).astype(
            np.uint8
        ) * 255
        allowed_masks[frame_index] = cv2.bitwise_and(
            warped_valid[frame_index], territory
        )

    cropped_images: list[np.ndarray] = []
    cropped_masks: list[np.ndarray] = []
    corners: list[tuple[int, int]] = []
    for warped, allowed in zip(warped_images, allowed_masks, strict=True):
        assert allowed is not None
        x, y, width, height = cv2.boundingRect(allowed)
        if width == 0 or height == 0:
            raise ValueError("A scan keyframe received no output territory")
        cropped_images.append(warped[y : y + height, x : x + width])
        cropped_masks.append(allowed[y : y + height, x : x + width])
        corners.append((x, y))

    automatic_regions: list[tuple[int, int, int, int, int]] = []
    quality_metrics: dict[str, float | int | bool] = {
        "automatic_protected_component_count": 0,
        "automatic_dp_seam_count": 0,
        "seam_risk_fraction": 0.0,
        "safe_seam_lab_residual_p95": 0.0,
        "minimum_seam_cross_coverage_ratio": 1.0,
        "strict_owner_partition": True,
    }
    blend_guard_pixels = min(
        64, max(4, 1 << min(multiband_levels + 1, 6))
    )
    if auto_foreground:
        seam_masks, automatic_metrics = _monotonic_owner_masks(
            warped_images,
            warped_valid,
            warped_depths,
            center_array,
            axis,
            order,
            seam_margin,
            blend_guard_pixels,
        )
        quality_metrics.update(automatic_metrics)
    else:
        seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        try:
            seam_result = seam_finder.find(
                [image.astype(np.float32) for image in cropped_images],
                corners,
                cropped_masks,
            )
        except cv2.error as exc:
            raise RuntimeError(
                "Graph-cut seam estimation failed; refusing to fall back to "
                "overlap averaging"
            ) from exc
        seam_masks = [
            np.zeros((canvas.height, canvas.width), dtype=np.uint8) for _ in images
        ]
        for index, (corner, mask) in enumerate(
            zip(corners, seam_result, strict=True)
        ):
            x, y = corner
            value = mask.get() if hasattr(mask, "get") else np.asarray(mask)
            value = np.ascontiguousarray(value, dtype=np.uint8)
            height, width = value.shape
            seam_masks[index][y : y + height, x : x + width] = value

    for owner, x0, y0, x1, y1 in region_rows:
        area = np.zeros((canvas.height, canvas.width), dtype=bool)
        area[y0:y1, x0:x1] = True
        area &= warped_valid[owner] > 0
        if not area.any():
            raise ValueError("A protected region has no coverage from its owner")
        for seam_mask in seam_masks:
            seam_mask[area] = 0
        seam_masks[owner][area] = 255

    union_valid = np.logical_or.reduce([mask > 0 for mask in warped_valid])
    union_seam = np.logical_or.reduce([mask > 0 for mask in seam_masks])
    holes = union_valid & ~union_seam
    if holes.any():
        best_distance = np.full(holes.shape, np.inf, dtype=np.float32)
        for center, valid in zip(center_array, warped_valid, strict=True):
            along = np.abs(pixel_projection - float(np.dot(center, axis)))
            take = holes & (valid > 0) & (along < best_distance)
            best_distance[take] = along[take]
        for seam_mask, center, valid in zip(
            seam_masks, center_array, warped_valid, strict=True
        ):
            along = np.abs(pixel_projection - float(np.dot(center, axis)))
            take = holes & (valid > 0) & np.isclose(along, best_distance)
            seam_mask[take] = 255
            holes[take] = False

    if auto_foreground:
        quality_metrics.update(
            _evaluate_final_owner_boundaries(
                warped_images,
                warped_valid,
                warped_depths,
                seam_masks,
                axis,
                order,
                blend_guard_pixels,
            )
        )

    blend_masks: list[np.ndarray] = []
    for seam_mask, valid in zip(seam_masks, warped_valid, strict=True):
        if seam_mask_sigma > 0.0:
            blend_mask = cv2.GaussianBlur(
                seam_mask,
                (0, 0),
                sigmaX=seam_mask_sigma,
                sigmaY=seam_mask_sigma,
                borderType=cv2.BORDER_CONSTANT,
            )
            blend_mask = np.where(valid > 0, blend_mask, 0).astype(np.uint8)
        else:
            blend_mask = seam_mask
        blend_masks.append(blend_mask)

    if auto_foreground:
        blended, blended_mask = _normalized_multiband_blend(
            warped_images, blend_masks, multiband_levels
        )
    else:
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(multiband_levels)
        blender.prepare((0, 0, canvas.width, canvas.height))
        for warped, seam_mask in zip(warped_images, blend_masks, strict=True):
            x, y, width, height = cv2.boundingRect(seam_mask)
            if width == 0 or height == 0:
                continue
            blender.feed(
                np.ascontiguousarray(
                    warped[y : y + height, x : x + width], dtype=np.int16
                ),
                np.ascontiguousarray(
                    seam_mask[y : y + height, x : x + width].copy(),
                    dtype=np.uint8,
                ),
                (x, y),
            )
        blended, blended_mask = blender.blend(None, None)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
    crop = largest_valid_rectangle(blended_mask > 0)
    panorama = blended[
        crop.y : crop.y + crop.height,
        crop.x : crop.x + crop.width,
    ]
    crop_height_ratio = crop.height / max(
        1.0, float(np.median([image.shape[0] for image in images]))
    )
    crop_width_ratio = crop.width / max(1.0, float(canvas.width))
    gain_values = np.asarray(gains, dtype=np.float64)
    gain_min = float(gain_values.min())
    gain_max = float(gain_values.max())
    quality_metrics.update(
        {
            "crop_height_ratio": float(crop_height_ratio),
            "crop_width_ratio": float(crop_width_ratio),
            "exposure_gain_min": gain_min,
            "exposure_gain_max": gain_max,
        }
    )
    failure_reasons: list[str] = []
    if crop_height_ratio < 0.90:
        failure_reasons.append("less than 90% of source height remains after crop")
    if crop_width_ratio < 0.95:
        failure_reasons.append("less than 95% of the scan canvas remains after crop")
    if float(quality_metrics["seam_risk_fraction"]) > 0.10:
        failure_reasons.append("a seam still crosses high-parallax foreground")
    if float(quality_metrics.get("blend_guard_risk_fraction", 0.0)) > 0.10:
        failure_reasons.append(
            "the multiband blend footprint still reaches high-parallax foreground"
        )
    if float(quality_metrics["minimum_seam_cross_coverage_ratio"]) < 0.80:
        failure_reasons.append("a final owner boundary covers too little of the overlap")
    if float(quality_metrics["safe_seam_lab_residual_p95"]) > 48.0:
        failure_reasons.append("safe seam color residual is too high")
    if not bool(quality_metrics["strict_owner_partition"]):
        failure_reasons.append("the final owner masks contain holes or overlaps")
    if int(quality_metrics.get("nonadjacent_owner_boundary_pixel_count", 0)) > 0:
        failure_reasons.append("the final owner map contains a non-adjacent frame boundary")
    if int(quality_metrics.get("unsupported_owner_boundary_pixel_count", 0)) > 0:
        failure_reasons.append("a final owner boundary is not supported by source overlap")
    if int(quality_metrics.get("final_owner_boundary_pair_count", 0)) != len(order) - 1:
        failure_reasons.append("one or more adjacent source boundaries disappeared")
    if gain_min < 0.45 or gain_max > 2.20:
        failure_reasons.append("exposure compensation gain is outside the safe range")
    quality_metrics["quality_pass"] = not failure_reasons
    if quality_gate and failure_reasons:
        raise RuntimeError(
            "Scan render quality gate failed: " + "; ".join(failure_reasons)
        )
    info = ScanRenderInfo(
        canvas=canvas,
        crop=crop,
        source_count=len(images),
        scan_axis=(float(axis[0]), float(axis[1])),
        color_gains=tuple(tuple(float(value) for value in gain) for gain in gains),
        seam_margin=seam_margin,
        multiband_levels=multiband_levels,
        exposure_mode=effective_exposure_mode,
        seam_mask_sigma=seam_mask_sigma,
        protected_regions=region_rows,
        automatic_protected_regions=tuple(automatic_regions),
        quality_metrics=quality_metrics,
    )
    return panorama, info


def render_panorama(
    images: list[np.ndarray],
    transforms: list[np.ndarray],
    *,
    max_megapixels: float = 250.0,
    feather_pixels: int = 64,
) -> tuple[np.ndarray, CanvasInfo]:
    info = compute_canvas(images, transforms, max_megapixels)
    accumulator = np.zeros((info.height, info.width, 3), dtype=np.float32)
    weights = np.zeros((info.height, info.width), dtype=np.float32)

    for image, transform in zip(images, transforms, strict=True):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Renderer expects BGR uint8 images")
        height, width = image.shape[:2]
        base_weight = _base_feather_mask(height, width, feather_pixels)
        canvas_h = info.translation @ _normalize_homography(transform)
        warped = cv2.warpPerspective(
            image,
            canvas_h,
            (info.width, info.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(np.float32)
        warped_weight = cv2.warpPerspective(
            base_weight,
            canvas_h,
            (info.width, info.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        alpha = cv2.warpPerspective(
            np.ones((height, width), dtype=np.uint8),
            canvas_h,
            (info.width, info.height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
        warped_weight *= alpha.astype(np.float32)
        accumulator += warped * warped_weight[..., None]
        weights += warped_weight

    valid = weights > 1e-6
    panorama = np.zeros_like(accumulator, dtype=np.uint8)
    panorama[valid] = np.clip(
        accumulator[valid] / weights[valid, None], 0.0, 255.0
    ).astype(np.uint8)
    return panorama, info
