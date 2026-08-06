"""C6 narrow, provenance-preserving MultiBand blending for safe backgrounds."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SafeMultiBandResult:
    bgr: np.ndarray
    blend_mask: np.ndarray
    dominant_owner_frame_id: np.ndarray
    audit: dict[str, object]


def blend_safe_background_multiband(
    first_bgra: np.ndarray,
    second_bgra: np.ndarray,
    owner_frame_id: np.ndarray,
    *,
    first_frame_id: int,
    second_frame_id: int,
    safe_mask: np.ndarray,
    band_pixels: int = 16,
    levels: int = 3,
) -> SafeMultiBandResult:
    """Blend only a narrow, shared-safe owner boundary; retain dominant owner.

    The output map deliberately keeps a single dominant real owner per pixel.
    The blend mask is the separate provenance signal identifying the second
    participating real source; callers must never apply it over protected
    foreground/depth-edge pixels.
    """

    first, second = np.asarray(first_bgra), np.asarray(second_bgra)
    owner, safe = np.asarray(owner_frame_id), np.asarray(safe_mask, dtype=bool)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 4:
        raise ValueError("MultiBand sources must be matching BGRA images")
    if owner.shape != first.shape[:2] or safe.shape != owner.shape:
        raise ValueError("MultiBand owner and safe masks must match the placed sources")
    if not 2 <= int(band_pixels) <= 24 or not 1 <= int(levels) <= 3:
        raise ValueError("MultiBand limits are outside the C6 envelope")
    common = (first[..., 3] > 0) & (second[..., 3] > 0)
    real_owners = (owner == int(first_frame_id)) | (owner == int(second_frame_id))
    horizontal_boundary = np.zeros_like(owner, dtype=bool)
    horizontal_boundary[:, 1:] = (
        real_owners[:, 1:]
        & real_owners[:, :-1]
        & (owner[:, 1:] != owner[:, :-1])
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_pixels * 2 + 1,) * 2)
    band = cv2.dilate(horizontal_boundary.astype(np.uint8), kernel).astype(bool)
    blend_mask = band & safe & common
    base = np.where((owner == int(second_frame_id))[..., None], second[..., :3], first[..., :3]).copy()
    if not np.any(blend_mask):
        return SafeMultiBandResult(base, blend_mask, owner.copy(), {
            "method": "candidate_safe_background_multiband/v1", "applied": False,
            "reason": "no_safe_narrow_owner_boundary", "blend_pixel_count": 0,
        })
    blender = cv2.detail_MultiBandBlender(try_gpu=0, num_bands=int(levels))
    mask = (blend_mask.astype(np.uint8) * 255)
    blender.prepare((0, 0, first.shape[1], first.shape[0]))
    blender.feed(first[..., :3].astype(np.int16), mask, (0, 0))
    blender.feed(second[..., :3].astype(np.int16), mask, (0, 0))
    blended, result_mask = blender.blend(None, None)
    applied = blend_mask & (result_mask > 0)
    base[applied] = np.clip(blended[applied], 0, 255).astype(np.uint8)
    return SafeMultiBandResult(base, applied, owner.copy(), {
        "method": "candidate_safe_background_multiband/v1", "applied": True,
        "levels": int(levels), "band_pixels": int(band_pixels),
        "blend_pixel_count": int(np.count_nonzero(applied)),
        "protected_intersection_pixel_count": 0,
        "dominant_owner_preserved": True,
        "participant_frame_ids": [int(first_frame_id), int(second_frame_id)],
    })


__all__ = ["SafeMultiBandResult", "blend_safe_background_multiband"]
