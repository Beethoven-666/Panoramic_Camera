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
