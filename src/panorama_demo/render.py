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
) -> tuple[np.ndarray, ScanRenderInfo]:
    """Render a side scan with one owner per region and seams around foreground objects.

    Unlike :func:`render_panorama`, this renderer does not average every frame that
    covers a pixel.  Frames first receive a narrow territory along the scan axis,
    graph-cut selects a low-cost seam inside adjacent territories, and multiband
    blending is limited to those seams.
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
    centers: list[np.ndarray] = []
    for image, transform, gain in zip(images, transforms, gains, strict=True):
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
                borderMode=cv2.BORDER_CONSTANT,
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

    if len(images) > 1:
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
        cropped_masks = [
            np.ascontiguousarray(
                mask.get() if hasattr(mask, "get") else np.asarray(mask),
                dtype=np.uint8,
            )
            for mask in seam_result
        ]

    seam_masks = [
        np.zeros((canvas.height, canvas.width), dtype=np.uint8) for _ in images
    ]
    for index, (corner, mask) in enumerate(zip(corners, cropped_masks, strict=True)):
        x, y = corner
        height, width = mask.shape
        seam_masks[index][y : y + height, x : x + width] = mask

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
                seam_mask[y : y + height, x : x + width].copy(), dtype=np.uint8
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
