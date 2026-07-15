from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_PROJECTED_NEAR_FOREGROUND_MM = 1000.0


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


@dataclass(frozen=True)
class ProjectedScanRenderInfo:
    """Audit record for the formal, already-projected RGB-D render path."""

    crop: CropInfo
    canvas_width: int
    canvas_height: int
    source_count: int
    source_order: tuple[int, ...]
    color_gains: tuple[tuple[float, float, float], ...]
    multiband_levels: int
    blend_radius_pixels: int
    blend_zone_pixel_count: int
    hard_owner_component_count: int
    owner_masks: tuple[np.ndarray, ...]
    quality_metrics: dict[str, float | int | bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": "graphcut_depth_constrained",
            "blend_backend": "opencv_multiband_narrow_owner_boundary",
            "crop": self.crop.as_dict(),
            "canvas": {
                "width": self.canvas_width,
                "height": self.canvas_height,
            },
            "source_count": self.source_count,
            "source_order": list(self.source_order),
            "color_gains": [list(gain) for gain in self.color_gains],
            "multiband_levels": self.multiband_levels,
            "blend_radius_pixels": self.blend_radius_pixels,
            "blend_zone_pixel_count": self.blend_zone_pixel_count,
            "hard_owner_component_count": self.hard_owner_component_count,
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
    *,
    depth_valid0: np.ndarray | None = None,
    depth_valid1: np.ndarray | None = None,
    camera_depth0: np.ndarray | None = None,
    camera_depth1: np.ndarray | None = None,
    camera_depth_valid0: np.ndarray | None = None,
    camera_depth_valid1: np.ndarray | None = None,
    world_surface_depth: bool = False,
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
        if (depth_valid0 is None) != (depth_valid1 is None):
            raise ValueError("Both surface-depth valid masks must be provided together")
        if depth_valid0 is None:
            # Legacy image-warp callers provide camera depth without a separate
            # validity mask.  The RGB-D projection path always supplies explicit
            # masks because world-normal depth may legitimately be below 400 mm
            # or negative depending on the chosen world origin.
            depth_valid = common & (depth0 >= 400.0) & (depth1 >= 400.0)
        else:
            if depth_valid0.shape != common.shape or depth_valid1.shape != common.shape:
                raise ValueError("Surface-depth valid masks must match the canvas")
            depth_valid = common & (depth_valid0 > 0) & (depth_valid1 > 0)
        minimum = np.minimum(depth0, depth1)
        # World-normal surface coordinates may be translated by an arbitrary
        # world origin, so their disagreement threshold must be translation
        # invariant.  Legacy camera-depth callers retain the relative term.
        tolerance = (
            np.full(common.shape, 100.0, dtype=np.float32)
            if world_surface_depth
            else np.maximum(100.0, minimum * 0.10)
        )
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
        if not flat_scene and not world_surface_depth:
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
        if flat_scene and not world_surface_depth:
            depth_risk |= _expand_depth_occlusion_surface(
                depth0, first_is_nearer, common
            )
            depth_risk |= _expand_depth_occlusion_surface(
                depth1, second_is_nearer, common
            )
        elif not world_surface_depth:
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

    camera_values = (
        camera_depth0,
        camera_depth1,
        camera_depth_valid0,
        camera_depth_valid1,
    )
    near_risk = np.zeros(common.shape, dtype=bool)
    if any(value is not None for value in camera_values):
        if any(value is None for value in camera_values):
            raise ValueError(
                "Both projected camera-depth arrays and valid masks are required"
            )
        assert camera_depth0 is not None
        assert camera_depth1 is not None
        assert camera_depth_valid0 is not None
        assert camera_depth_valid1 is not None
        if any(
            value.shape != common.shape
            for value in (
                camera_depth0,
                camera_depth1,
                camera_depth_valid0,
                camera_depth_valid1,
            )
        ):
            raise ValueError("Projected camera-depth arrays must match the canvas")
        near_valid = (
            common
            & (camera_depth_valid0 > 0)
            & (camera_depth_valid1 > 0)
            & np.isfinite(camera_depth0)
            & np.isfinite(camera_depth1)
            & (camera_depth0 > 0.0)
            & (camera_depth1 > 0.0)
        )
        # Near geometry is not intrinsically unsafe: with dense short-baseline
        # RGB-D tracking, matching close surfaces should remain available for a
        # local seam.  Protect it only when the two real camera ranges disagree
        # by more than sensor-scale noise.  RGB/world-surface disagreement is
        # already represented by rgb_risk/depth_risk above.
        near_depth = np.minimum(camera_depth0, camera_depth1)
        range_disagreement = np.abs(camera_depth0 - camera_depth1) >= np.maximum(
            40.0, near_depth * 0.08
        )
        near_risk = (
            near_valid
            & (near_depth < _PROJECTED_NEAR_FOREGROUND_MM)
            & range_disagreement
        )

    risk = (rgb_risk | depth_risk | near_risk).astype(np.uint8) * 255
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
        & ~near_risk
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
    *,
    depth_valid_masks: list[np.ndarray | None] | None = None,
    camera_depth_maps: list[np.ndarray | None] | None = None,
    camera_depth_valid_masks: list[np.ndarray | None] | None = None,
    world_surface_depth: bool = False,
) -> dict[str, float | int | bool]:
    """Measure risk on the final masks after overrides and hole filling."""

    owned = np.stack([mask > 0 for mask in owner_masks], axis=0)
    valid = np.stack([mask > 0 for mask in valid_masks], axis=0)
    if depth_valid_masks is not None and len(depth_valid_masks) != len(images):
        raise ValueError("Surface-depth valid masks must match the source count")
    if camera_depth_maps is not None and len(camera_depth_maps) != len(images):
        raise ValueError("Camera-depth maps must match the source count")
    if camera_depth_valid_masks is not None and len(camera_depth_valid_masks) != len(images):
        raise ValueError("Camera-depth valid masks must match the source count")
    if (camera_depth_maps is None) != (camera_depth_valid_masks is None):
        raise ValueError("Camera-depth maps and valid masks must be provided together")
    union_valid = np.any(valid, axis=0)
    owner_count = np.sum(owned, axis=0)
    hole_count = int(np.count_nonzero(union_valid & (owner_count == 0)))
    overlap_count = int(np.count_nonzero(owner_count > 1))
    invalid_owner_count = int(np.count_nonzero(owned & ~valid))
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
                depth_valid0=(
                    depth_valid_masks[first]
                    if depth_valid_masks is not None
                    else None
                ),
                depth_valid1=(
                    depth_valid_masks[second]
                    if depth_valid_masks is not None
                    else None
                ),
                camera_depth0=(
                    camera_depth_maps[first]
                    if camera_depth_maps is not None
                    else None
                ),
                camera_depth1=(
                    camera_depth_maps[second]
                    if camera_depth_maps is not None
                    else None
                ),
                camera_depth_valid0=(
                    camera_depth_valid_masks[first]
                    if camera_depth_valid_masks is not None
                    else None
                ),
                camera_depth_valid1=(
                    camera_depth_valid_masks[second]
                    if camera_depth_valid_masks is not None
                    else None
                ),
                world_surface_depth=world_surface_depth,
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
        "automatic_dp_seam_count": (
            0 if world_surface_depth else max(0, len(order) - 1)
        ),
        "validated_owner_boundary_count": boundary_pair_count,
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
        "invalid_owner_pixel_count": invalid_owner_count,
        "strict_owner_partition": (
            hole_count == 0 and overlap_count == 0 and invalid_owner_count == 0
        ),
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


def _projected_source_arrays(
    source: object,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float],
]:
    """Validate the public ``ProjectedRGBDSource`` structural contract."""

    try:
        image = np.asarray(getattr(source, "warped_rgb"))
        valid = np.asarray(getattr(source, "valid_mask"))
        depth = np.asarray(getattr(source, "surface_depth_mm"), dtype=np.float32)
        depth_valid = np.asarray(getattr(source, "surface_depth_valid_mask"))
        camera_depth = np.asarray(
            getattr(source, "camera_depth_mm"), dtype=np.float32
        )
        camera_depth_valid = np.asarray(
            getattr(source, "camera_depth_valid_mask")
        )
        center_value = getattr(source, "projected_center_xy")
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            "Each projected source must provide RGB, validity, world depth, "
            "depth validity, camera depth, and projected centre"
        ) from exc
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Projected RGB sources must be BGR uint8 images")
    shape = image.shape[:2]
    if any(
        value.shape != shape
        for value in (valid, depth, depth_valid, camera_depth, camera_depth_valid)
    ):
        raise ValueError("Projected RGB-D arrays must share one canvas shape")
    if (
        valid.dtype != np.uint8
        or depth_valid.dtype != np.uint8
        or camera_depth_valid.dtype != np.uint8
    ):
        raise ValueError("Projected validity masks must be uint8")
    if np.any((valid > 0) != (depth_valid > 0)):
        raise ValueError("Projected colour and world-depth validity must agree")
    if np.any((valid > 0) != (camera_depth_valid > 0)):
        raise ValueError("Projected colour and camera-depth validity must agree")
    if not np.any(valid):
        raise ValueError("A projected source contains no valid surface")
    if not np.isfinite(depth[depth_valid > 0]).all():
        raise ValueError("Projected world-surface depth contains non-finite values")
    if (
        not np.isfinite(camera_depth[camera_depth_valid > 0]).all()
        or np.any(camera_depth[camera_depth_valid > 0] <= 0.0)
    ):
        raise ValueError("Projected camera depth must be finite and positive")
    center = np.asarray(center_value, dtype=np.float64)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("Projected source centre must be a finite 2-vector")
    return (
        image,
        valid,
        depth,
        depth_valid,
        camera_depth,
        camera_depth_valid,
        (float(center[0]), float(center[1])),
    )


def _projected_exposure_compensation(
    images: list[np.ndarray],
    valid_masks: list[np.ndarray],
    mode: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    if mode not in {"none", "global_gain"}:
        raise ValueError("Projected exposure mode must be none or global_gain")
    adjusted = [image.copy() for image in images]
    gains = np.ones((len(images), 3), dtype=np.float64)
    if mode == "none" or len(images) < 2:
        return adjusted, gains
    compensator = cv2.detail.ExposureCompensator_createDefault(
        cv2.detail.ExposureCompensator_GAIN
    )
    try:
        compensator.feed([(0, 0)] * len(images), adjusted, valid_masks)
        for index, (image, valid) in enumerate(
            zip(adjusted, valid_masks, strict=True)
        ):
            compensator.apply(index, (0, 0), image, valid)
    except cv2.error as exc:
        raise RuntimeError(
            "Projected global exposure compensation failed; no fallback is allowed"
        ) from exc
    scalar_gains = np.asarray(
        [
            float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
            for value in compensator.getMatGains()
        ],
        dtype=np.float64,
    )
    if scalar_gains.shape != (len(images),) or not np.isfinite(scalar_gains).all():
        raise RuntimeError("OpenCV returned invalid projected exposure gains")
    return adjusted, np.repeat(scalar_gains[:, None], 3, axis=1)


def _risk_has_crossing_channel(
    common: np.ndarray,
    protected: np.ndarray,
) -> bool:
    """Return whether a vertical scan seam can pass around protected pixels."""

    active_rows = np.flatnonzero(np.any(common, axis=1))
    if active_rows.size < 2:
        return False
    first_row, last_row = int(active_rows[0]), int(active_rows[-1])
    safe = common & ~protected
    # A one-pixel sampling pinhole must not turn a physically continuous safe
    # corridor into a false pass/fail decision.  This closing is used only for
    # connectivity auditing; it never changes a GraphCut input mask.
    safe_for_connectivity = cv2.morphologyEx(
        safe.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    count, labels = cv2.connectedComponents(safe_for_connectivity, connectivity=8)
    if count <= 1:
        return False
    top_stop = min(last_row + 1, first_row + 3)
    bottom_start = max(first_row, last_row - 2)
    top = set(int(value) for value in np.unique(labels[first_row:top_stop]) if value)
    bottom = set(
        int(value) for value in np.unique(labels[bottom_start : last_row + 1]) if value
    )
    return bool(top & bottom)


def _source_preference(
    source: object,
    valid: np.ndarray,
    component: np.ndarray,
    rank: int,
    sharpness: float,
) -> tuple[float, float, float, int]:
    stats = getattr(source, "sampling_stats", {})
    sampling = 0.0
    if isinstance(stats, dict):
        sampling = float(
            stats.get("projected_sampling_ratio", stats.get("valid_depth_fraction", 0.0))
        )
    distance = cv2.distanceTransform((valid > 0).astype(np.uint8), cv2.DIST_L2, 3)
    edge_distance = float(np.percentile(distance[component], 25))
    return sampling, edge_distance, float(sharpness), -rank


def _apply_projected_hard_constraints(
    sources: list[object],
    images: list[np.ndarray],
    valid_masks: list[np.ndarray],
    depths: list[np.ndarray],
    depth_valid_masks: list[np.ndarray],
    camera_depths: list[np.ndarray],
    camera_depth_valid_masks: list[np.ndarray],
    first: int,
    second: int,
    first_allowed: np.ndarray,
    second_allowed: np.ndarray,
    blend_radius: int,
    sharpness_scores: list[float],
    first_rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    risk, _, _ = _pair_risk_mask(
        images[first],
        images[second],
        first_allowed,
        second_allowed,
        depths[first],
        depths[second],
        depth_valid0=depth_valid_masks[first],
        depth_valid1=depth_valid_masks[second],
        camera_depth0=camera_depths[first],
        camera_depth1=camera_depths[second],
        camera_depth_valid0=camera_depth_valid_masks[first],
        camera_depth_valid1=camera_depth_valid_masks[second],
        world_surface_depth=True,
    )
    risk_bool = risk > 0
    if not np.any(risk_bool):
        return first_allowed, second_allowed, risk, 0
    diameter = blend_radius * 2 + 1
    protected = cv2.dilate(
        risk,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)),
    ) > 0
    pair_union = (first_allowed > 0) | (second_allowed > 0)
    # Only common coverage is a seam decision.  Outside it, one source is
    # already the sole valid owner and must not be folded into a component that
    # demands impossible full coverage from both views.
    protected &= (first_allowed > 0) & (second_allowed > 0)
    # A protected component may span the complete pair corridor (for example,
    # a close greenhouse wall).  It is still renderable when one *real* source
    # covers the whole component: hard ownership below keeps the GraphCut
    # boundary outside that component.  Reject only when neither source can
    # provide that complete ownership, which is checked per component.

    constrained = [first_allowed.copy(), second_allowed.copy()]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        protected.astype(np.uint8), connectivity=8
    )
    component_count = 0
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) <= 0:
            continue
        component = labels == label
        candidates: list[tuple[tuple[float, float, float, int], int]] = []
        for local, source_index in enumerate((first, second)):
            if np.all(valid_masks[source_index][component] > 0):
                candidates.append(
                    (
                        _source_preference(
                            sources[source_index],
                            valid_masks[source_index],
                            component,
                            first_rank + local,
                            sharpness_scores[source_index],
                        ),
                        local,
                    )
                )
        if not candidates:
            raise RuntimeError(
                "No single reliable projected source can own a protected RGB-D region"
            )
        winner = max(candidates, key=lambda item: item[0])[1]
        loser = 1 - winner
        constrained[loser][component] = 0
        if not np.all(constrained[winner][component] > 0):
            raise RuntimeError("Hard owner assignment removed its selected source")
        component_count += 1
    constrained_union = (constrained[0] > 0) | (constrained[1] > 0)
    if np.any(pair_union & ~constrained_union):
        raise RuntimeError("Depth hard constraints created an unowned corridor pixel")
    return constrained[0], constrained[1], risk, component_count


def _graphcut_pair_owner_masks(
    first_image: np.ndarray,
    second_image: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pair_union = (first_mask > 0) | (second_mask > 0)
    x, y, width, height = cv2.boundingRect(pair_union.astype(np.uint8))
    if width == 0 or height == 0:
        raise RuntimeError("An owner boundary lacks real pair overlap")
    # The pair corridor is only this adjacent domain, never the full common
    # projection canvas. Its one-source portions remain explicit GraphCut
    # constraints rather than consuming a whole-canvas optimization.
    outputs = [np.zeros(first_mask.shape, dtype=np.uint8) for _ in range(2)]
    input_masks = [
        np.ascontiguousarray(first_mask[y : y + height, x : x + width].copy()),
        np.ascontiguousarray(second_mask[y : y + height, x : x + width].copy()),
    ]
    if not all(np.any(mask) for mask in input_masks):
        outputs = [first_mask.copy(), second_mask.copy()]
        output_count = (outputs[0] > 0).astype(np.uint8) + (
            outputs[1] > 0
        ).astype(np.uint8)
        if np.any(pair_union & (output_count != 1)):
            raise RuntimeError("A non-overlapping pair does not have a unique owner")
        return outputs[0], outputs[1]
    seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    try:
        result = seam_finder.find(
            [
                np.ascontiguousarray(
                    first_image[y : y + height, x : x + width], dtype=np.float32
                ),
                np.ascontiguousarray(
                    second_image[y : y + height, x : x + width], dtype=np.float32
                ),
            ],
            [(x, y), (x, y)],
            input_masks,
        )
    except cv2.error as exc:
        raise RuntimeError(
            "Depth-constrained GraphCut failed; no DP, feather, or averaging "
            "fallback is allowed"
        ) from exc
    if result is None:
        result = input_masks
    if len(result) != 2:
        raise RuntimeError("GraphCut returned an invalid adjacent-pair mask count")
    for index, (original, value) in enumerate(zip(input_masks, result, strict=True)):
        array = value.get() if hasattr(value, "get") else np.asarray(value)
        array = np.ascontiguousarray(array, dtype=np.uint8)
        if array.shape != original.shape:
            raise RuntimeError("GraphCut changed an adjacent-pair mask shape")
        if np.any((array > 0) & (original == 0)):
            raise RuntimeError("GraphCut assigned pixels outside its allowed corridor")
        outputs[index][y : y + height, x : x + width] = np.where(
            array > 0, 255, 0
        )
    output_count = (outputs[0] > 0).astype(np.uint8) + (outputs[1] > 0).astype(
        np.uint8
    )
    if np.any(pair_union & (output_count != 1)):
        raise RuntimeError("GraphCut left a hole or multiple owners in a pair corridor")
    if np.any((~pair_union) & (output_count != 0)):
        raise RuntimeError("GraphCut assigned an invalid projected pixel")
    return outputs[0], outputs[1]


def _opencv_multiband_owner_boundary_blend(
    images: list[np.ndarray],
    valid_masks: list[np.ndarray],
    owner_masks: list[np.ndarray],
    depths: list[np.ndarray],
    depth_valid_masks: list[np.ndarray],
    camera_depths: list[np.ndarray],
    camera_depth_valid_masks: list[np.ndarray],
    order: np.ndarray,
    levels: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    height, width = owner_masks[0].shape
    owned = [mask > 0 for mask in owner_masks]
    owner_union = np.logical_or.reduce(owned)
    direct = np.zeros((height, width, 3), dtype=np.uint8)
    for image, owner in zip(images, owned, strict=True):
        direct[owner] = image[owner]

    blend_zone = np.zeros((height, width), dtype=bool)
    maximum_zone_risk = 0.0
    neighbor_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    diameter = radius * 2 + 1
    zone_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    pair_rows: list[tuple[int, int, int]] = []
    zone_assignment = np.full((height, width), -1, dtype=np.int16)
    best_boundary_distance = np.full((height, width), np.inf, dtype=np.float32)
    for pair_rank, (first_value, second_value) in enumerate(
        zip(order[:-1], order[1:], strict=True)
    ):
        first, second = int(first_value), int(second_value)
        boundary = owned[first] & (
            cv2.dilate(owned[second].astype(np.uint8), neighbor_kernel) > 0
        )
        boundary |= owned[second] & (
            cv2.dilate(owned[first].astype(np.uint8), neighbor_kernel) > 0
        )
        if not np.any(boundary):
            # No shared boundary means this pair is already separated by
            # unique-depth ownership.  There is nothing to blend locally.
            continue
        zone = cv2.dilate(boundary.astype(np.uint8), zone_kernel) > 0
        pair_valid = (valid_masks[first] > 0) | (valid_masks[second] > 0)
        zone &= pair_valid
        distance = cv2.distanceTransform(
            (~boundary).astype(np.uint8), cv2.DIST_L2, 3
        )
        replace = zone & (distance < best_boundary_distance)
        zone_assignment[replace] = pair_rank
        best_boundary_distance[replace] = distance[replace]
        pair_rows.append((pair_rank, first, second))

    del best_boundary_distance
    blend_zone = zone_assignment >= 0
    for pair_rank, first, second in pair_rows:
        zone = zone_assignment == pair_rank
        if not np.any(zone):
            raise RuntimeError("An adjacent owner boundary has no exclusive blend zone")

        # Run one independent local blender per verified adjacent owner
        # boundary.  A single global pyramid can leak low-frequency colour
        # from source i+2 across a narrow source-i+1 territory even when its
        # level-zero mask never touches the boundary.
        support = cv2.dilate(zone.astype(np.uint8), zone_kernel) > 0
        first_mask = support & (
            owned[first] | (zone & (valid_masks[first] > 0))
        )
        second_mask = support & (
            owned[second] | (zone & (valid_masks[second] > 0))
        )
        pair_expected = first_mask | second_mask
        if not np.any(first_mask) or not np.any(second_mask):
            raise RuntimeError("A strict owner source has no MultiBand contribution")
        if np.any(zone & ~pair_expected):
            raise RuntimeError("A MultiBand boundary zone lacks a real source")
        x, y, local_width, local_height = cv2.boundingRect(
            pair_expected.astype(np.uint8)
        )
        if local_width == 0 or local_height == 0:
            raise RuntimeError("An adjacent owner boundary has no MultiBand support")
        local_masks = [
            np.ascontiguousarray(
                mask[y : y + local_height, x : x + local_width].astype(np.uint8)
                * 255
            )
            for mask in (first_mask, second_mask)
        ]
        blender = cv2.detail_MultiBandBlender()
        blender.setNumBands(int(levels))
        blender.prepare((0, 0, local_width, local_height))
        try:
            for source_index, mask in zip(
                (first, second), local_masks, strict=True
            ):
                blender.feed(
                    np.ascontiguousarray(
                        images[source_index][
                            y : y + local_height, x : x + local_width
                        ],
                        dtype=np.int16,
                    ),
                    mask,
                    (0, 0),
                )
            blended, output_mask = blender.blend(None, None)
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV MultiBand blending failed; no normalized or average "
                "fallback is allowed"
            ) from exc
        if hasattr(output_mask, "get"):
            output_mask = output_mask.get()
        output_mask = np.asarray(output_mask, dtype=np.uint8)
        expected_local = pair_expected[y : y + local_height, x : x + local_width]
        if output_mask.shape != expected_local.shape:
            raise RuntimeError("OpenCV MultiBand returned an invalid output mask shape")
        if np.any(expected_local & (output_mask == 0)):
            raise RuntimeError("OpenCV MultiBand produced a zero-weight owner wedge")
        if np.any((output_mask > 0) & ~expected_local):
            raise RuntimeError("OpenCV MultiBand wrote outside adjacent pair support")
        if hasattr(blended, "get"):
            blended = blended.get()
        blended = np.clip(np.asarray(blended), 0, 255).astype(np.uint8)
        if blended.shape != (local_height, local_width, 3):
            raise RuntimeError("OpenCV MultiBand returned an invalid image shape")
        local_zone = zone[y : y + local_height, x : x + local_width]
        direct_view = direct[y : y + local_height, x : x + local_width]
        direct_view[local_zone] = blended[local_zone]

        risk, _, _ = _pair_risk_mask(
            images[first],
            images[second],
            valid_masks[first],
            valid_masks[second],
            depths[first],
            depths[second],
            depth_valid0=depth_valid_masks[first],
            depth_valid1=depth_valid_masks[second],
            camera_depth0=camera_depths[first],
            camera_depth1=camera_depths[second],
            camera_depth_valid0=camera_depth_valid_masks[first],
            camera_depth_valid1=camera_depth_valid_masks[second],
            world_surface_depth=True,
        )
        audited = zone & (valid_masks[first] > 0) & (valid_masks[second] > 0)
        if np.any(audited):
            maximum_zone_risk = max(
                maximum_zone_risk, float(np.mean((risk > 0)[audited]))
            )
    return (
        direct,
        owner_union.astype(np.uint8) * 255,
        int(np.count_nonzero(blend_zone)),
        maximum_zone_risk,
    )


def render_projected_scan_panorama(
    projected_sources: list[object] | tuple[object, ...],
    *,
    max_megapixels: float = 200.0,
    multiband_levels: int = 5,
    exposure_mode: str = "global_gain",
    quality_gate: bool = True,
    sharpness_scores: list[float] | tuple[float, ...] | None = None,
) -> tuple[np.ndarray, ProjectedScanRenderInfo]:
    """Render full-canvas metric RGB-D projections with no geometric fallback.

    Inputs have already been projected exactly once by ``rgbd_projection``.
    Only adjacent scan-order sources compete, risky world-depth components are
    hard-owned before GraphCut, and OpenCV MultiBand is confined to a narrow
    band around the verified strict owner boundary.
    """

    sources = list(projected_sources)
    if len(sources) < 2:
        raise ValueError("Formal projected rendering requires at least two sources")
    if len(sources) > 32:
        raise ValueError("Formal projected rendering supports at most 32 sources")
    if multiband_levels < 1:
        raise ValueError("multiband_levels must be at least one")
    if not np.isfinite(max_megapixels) or max_megapixels <= 0.0:
        raise ValueError("max_megapixels must be finite and positive")

    arrays = [_projected_source_arrays(source) for source in sources]
    images = [item[0] for item in arrays]
    geometric_valid = [item[1] for item in arrays]
    depths = [item[2] for item in arrays]
    depth_valid = [item[3] for item in arrays]
    camera_depths = [item[4] for item in arrays]
    camera_depth_valid = [item[5] for item in arrays]
    centers = np.asarray([item[6] for item in arrays], dtype=np.float64)
    shape = images[0].shape[:2]
    if any(image.shape[:2] != shape for image in images):
        raise ValueError("All projected sources must use one common canvas")
    height, width = shape
    aggregate_mp = width * height * len(sources) / 1_000_000.0
    if aggregate_mp > max_megapixels:
        raise MemoryError(
            f"Projected render working set is {aggregate_mp:.1f} aggregate MP, "
            f"above the {max_megapixels:.1f} MP limit"
        )
    sharpness = (
        [float(value) for value in sharpness_scores]
        if sharpness_scores is not None
        else [float(getattr(source, "sharpness_score", 0.0)) for source in sources]
    )
    if len(sharpness) != len(sources) or not np.isfinite(sharpness).all():
        raise ValueError("sharpness_scores must be finite and match projected sources")
    corrected, gains = _projected_exposure_compensation(
        images, geometric_valid, exposure_mode
    )

    order = np.argsort(centers[:, 0], kind="stable")
    ordered_x = centers[order, 0]
    if np.any(np.diff(ordered_x) <= 1e-6):
        raise RuntimeError("Projected source centres are not strictly ordered")
    boundary_centres = 0.5 * (ordered_x[:-1] + ordered_x[1:])
    domain_edges = [0]
    for left, right in zip(boundary_centres[:-1], boundary_centres[1:], strict=True):
        domain_edges.append(
            int(np.clip(np.floor(0.5 * (left + right)), 1, width - 1))
        )
    domain_edges.append(width)
    if any(right <= left for left, right in zip(domain_edges[:-1], domain_edges[1:])):
        raise RuntimeError("Adjacent-pair corridors collapse on the projected canvas")

    owner_masks = [np.zeros(shape, dtype=np.uint8) for _ in sources]
    corridor_valid = [np.zeros(shape, dtype=np.uint8) for _ in sources]
    hard_components = 0
    # Five OpenCV bands have an effective half-width of roughly 2**5 pixels.
    # Cap that footprint on small diagnostic canvases so unrelated residual
    # components cannot merge merely because the test/source is low resolution.
    blend_radius = min(
        64,
        max(4, 1 << min(multiband_levels, 6)),
        max(4, min(height, width) // 8),
    )
    x_grid = np.arange(width)[None, :]
    for pair_rank, (first_value, second_value) in enumerate(
        zip(order[:-1], order[1:], strict=True)
    ):
        first, second = int(first_value), int(second_value)
        zone = (x_grid >= domain_edges[pair_rank]) & (
            x_grid < domain_edges[pair_rank + 1]
        )
        # GraphCut is a local boundary solver, not a whole-strip compositor.
        # Give it a bounded overlap ribbon around the physical mid-boundary;
        # outside that ribbon the scan order fixes a unique owner directly.
        left_edge = domain_edges[pair_rank]
        right_edge = domain_edges[pair_rank + 1]
        boundary_x = (left_edge + right_edge) // 2
        ribbon_half_width = min(
            max(64, blend_radius * 6), max(8, (right_edge - left_edge) // 2)
        )
        first_zone = zone & (x_grid < boundary_x + ribbon_half_width)
        second_zone = zone & (x_grid >= boundary_x - ribbon_half_width)
        first_allowed = np.where(
            first_zone & (geometric_valid[first] > 0), 255, 0
        ).astype(np.uint8)
        second_allowed = np.where(
            second_zone & (geometric_valid[second] > 0), 255, 0
        ).astype(np.uint8)
        corridor_valid[first] = cv2.bitwise_or(corridor_valid[first], first_allowed)
        corridor_valid[second] = cv2.bitwise_or(corridor_valid[second], second_allowed)
        first_allowed, second_allowed, _, component_count = (
            _apply_projected_hard_constraints(
                sources,
                corrected,
                geometric_valid,
                depths,
                depth_valid,
                camera_depths,
                camera_depth_valid,
                first,
                second,
                first_allowed,
                second_allowed,
                blend_radius,
                sharpness,
                pair_rank,
            )
        )
        first_owner, second_owner = _graphcut_pair_owner_masks(
            corrected[first],
            corrected[second],
            first_allowed,
            second_allowed,
        )
        owner_masks[first] = cv2.bitwise_or(owner_masks[first], first_owner)
        owner_masks[second] = cv2.bitwise_or(owner_masks[second], second_owner)
        hard_components += component_count

    quality_metrics = _evaluate_final_owner_boundaries(
        corrected,
        corridor_valid,
        depths,
        owner_masks,
        np.array([1.0, 0.0], dtype=np.float64),
        order,
        blend_radius,
        depth_valid_masks=depth_valid,
        camera_depth_maps=camera_depths,
        camera_depth_valid_masks=camera_depth_valid,
        world_surface_depth=True,
    )
    structural_reasons: list[str] = []
    if not bool(quality_metrics["strict_owner_partition"]):
        structural_reasons.append("owner masks contain holes, overlaps, or invalid owners")
    if (
        quality_gate
        and int(quality_metrics["nonadjacent_owner_boundary_pixel_count"]) > 0
    ):
        structural_reasons.append("non-adjacent projected owners touch")
    if structural_reasons:
        raise RuntimeError(
            "Projected owner topology validation failed: " + "; ".join(structural_reasons)
        )

    blended, blended_mask, blend_zone_pixels, zone_risk = (
        _opencv_multiband_owner_boundary_blend(
            corrected,
            corridor_valid,
            owner_masks,
            depths,
            depth_valid,
            camera_depths,
            camera_depth_valid,
            order,
            multiband_levels,
            blend_radius,
        )
    )
    quality_metrics["blend_zone_risk_fraction"] = float(zone_risk)
    # RGB-D point splats deliberately leave depth holes rather than inventing
    # triangles.  Crop to the full observed union, not its largest hole-free
    # rectangle, otherwise a valid long scan collapses to one tiny patch.
    x, y, crop_width, crop_height = cv2.boundingRect(
        (blended_mask > 0).astype(np.uint8)
    )
    if crop_width == 0 or crop_height == 0:
        raise RuntimeError("Projected RGB-D fusion produced no valid output pixels")
    crop = CropInfo(x=x, y=y, width=crop_width, height=crop_height)
    panorama = blended[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
    projected_heights = [
        float(getattr(source, "projected_height_px", cv2.boundingRect(valid)[3]))
        for source, valid in zip(sources, corridor_valid, strict=True)
    ]
    crop_height_ratio = crop.height / max(1.0, float(np.median(projected_heights)))
    crop_width_ratio = crop.width / max(1.0, float(width))
    gain_min = float(np.min(gains))
    gain_max = float(np.max(gains))
    quality_metrics.update(
        {
            "crop_height_ratio": float(crop_height_ratio),
            "crop_width_ratio": float(crop_width_ratio),
            "exposure_gain_min": gain_min,
            "exposure_gain_max": gain_max,
            "graphcut_pair_count": len(order) - 1,
            "hard_owner_component_count": hard_components,
            "multiband_output_mask_complete": True,
            "near_foreground_threshold_mm": _PROJECTED_NEAR_FOREGROUND_MM,
        }
    )
    failure_reasons: list[str] = []
    if crop_height_ratio < 0.90:
        failure_reasons.append("less than 90% of projected source height remains")
    if crop_width_ratio < 0.95:
        failure_reasons.append("less than 95% of the projected canvas remains")
    if float(quality_metrics["seam_risk_fraction"]) > 0.10:
        failure_reasons.append("a final GraphCut boundary crosses RGB-D risk")
    if float(quality_metrics.get("blend_guard_risk_fraction", 0.0)) > 0.10:
        failure_reasons.append("the MultiBand guard reaches RGB-D risk")
    if zone_risk > 0.10:
        failure_reasons.append("the actual MultiBand zone reaches RGB-D risk")
    if float(quality_metrics["minimum_seam_cross_coverage_ratio"]) < 0.80:
        failure_reasons.append("an owner boundary has insufficient cross coverage")
    if int(quality_metrics["unsupported_owner_boundary_pixel_count"]) > 0:
        failure_reasons.append("an owner boundary lacks real pair overlap")
    if int(quality_metrics["final_owner_boundary_pair_count"]) != len(order) - 1:
        failure_reasons.append("one or more adjacent owner boundaries are missing")
    if any(not np.any(mask) for mask in owner_masks):
        failure_reasons.append("one or more projected sources lost all ownership")
    if float(quality_metrics["safe_seam_lab_residual_p95"]) > 48.0:
        failure_reasons.append("safe GraphCut boundary Lab residual is too high")
    if gain_min < 0.45 or gain_max > 2.20:
        failure_reasons.append("exposure compensation gain is outside 0.45-2.20")
    quality_metrics["quality_pass"] = not failure_reasons
    if quality_gate and failure_reasons:
        raise RuntimeError(
            "Projected render quality gate failed: " + "; ".join(failure_reasons)
        )
    info = ProjectedScanRenderInfo(
        crop=crop,
        canvas_width=width,
        canvas_height=height,
        source_count=len(sources),
        source_order=tuple(int(value) for value in order),
        color_gains=tuple(tuple(float(value) for value in gain) for gain in gains),
        multiband_levels=multiband_levels,
        blend_radius_pixels=blend_radius,
        blend_zone_pixel_count=blend_zone_pixels,
        hard_owner_component_count=hard_components,
        owner_masks=tuple(mask.copy() for mask in owner_masks),
        quality_metrics=quality_metrics,
    )
    return panorama, info
