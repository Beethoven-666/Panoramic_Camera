"""Narrow, guard-excluded v6 near-field MultiBand eligibility and blending."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .video_hard_guards import VideoHardGuards
from .video_visual_renderer import VideoDISPairEvidence


@dataclass(frozen=True)
class VideoNearBlendConfig:
    local_fb_target_px: float = 0.75
    local_fb_hard_px: float = 1.25
    rgb_residual_max: float = 20.0
    near_width_px: int = 4
    near_width_range_px: tuple[int, int] = (2, 8)
    strong_texture_width_range_px: tuple[int, int] = (2, 6)
    maximum_levels: int = 3


@dataclass(frozen=True)
class VideoNearBlendAudit:
    eligible_pixel_count: int
    band_pixel_count: int
    guard_intersection_pixel_count: int
    width_px: int
    multiband_levels: int
    applied: bool


def build_near_blend_eligible_mask(
    old_valid: np.ndarray, new_valid: np.ndarray, evidence: VideoDISPairEvidence, guards: VideoHardGuards,
    *, config: VideoNearBlendConfig | None = None,
) -> np.ndarray:
    """Allow only safe, jointly real RGB samples inside an unprotected near region."""
    settings = config or VideoNearBlendConfig()
    old_valid, new_valid = np.asarray(old_valid, bool), np.asarray(new_valid, bool)
    if old_valid.shape != new_valid.shape or old_valid.shape != evidence.fb_error.shape or old_valid.shape != guards.protected.shape:
        raise ValueError("near blend evidence, guards, and valid masks must share a pair corridor")
    fb_ok = np.isfinite(evidence.fb_error) & (evidence.fb_error <= settings.local_fb_target_px)
    rgb_ok = np.isfinite(evidence.rgb_residual) & (evidence.rgb_residual <= settings.rgb_residual_max)
    return old_valid & new_valid & fb_ok & rgb_ok & ~evidence.occlusion_risk_mask & ~guards.protected


def _narrow_band(owner_new: np.ndarray, eligible: np.ndarray, width_px: int) -> np.ndarray:
    labels = np.asarray(owner_new, bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    boundary = np.zeros_like(labels)
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    if not boundary.any():
        return np.zeros_like(labels)
    distance = cv2.distanceTransform((~boundary).astype(np.uint8), cv2.DIST_L2, 3)
    return np.asarray(eligible, bool) & (distance <= float(width_px))


def apply_near_multiband(
    old_bgr: np.ndarray, new_bgr: np.ndarray, owner_bgr: np.ndarray, owner_new: np.ndarray,
    eligible: np.ndarray, guards: VideoHardGuards, *, config: VideoNearBlendConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, VideoNearBlendAudit]:
    """Blend a 2–8px safe interior band; all guards remain verbatim hard owners."""
    settings = config or VideoNearBlendConfig()
    if not settings.near_width_range_px[0] <= settings.near_width_px <= settings.near_width_range_px[1]:
        raise ValueError("near MultiBand width must stay in the frozen 2–8px range")
    old, new, result = (np.asarray(value) for value in (old_bgr, new_bgr, owner_bgr))
    if old.shape != new.shape or old.shape != result.shape or old.ndim != 3 or old.shape[2] != 3:
        raise ValueError("near MultiBand requires matching BGR pair and owner output")
    band = _narrow_band(owner_new, eligible, settings.near_width_px)
    guard_intersection = int(np.count_nonzero(band & guards.protected))
    if guard_intersection:
        raise RuntimeError("hard guard entered near MultiBand band")
    if int(band.sum()) == 0:
        return result.copy(), band, VideoNearBlendAudit(int(eligible.sum()), 0, 0, settings.near_width_px, 0, False)
    levels = min(settings.maximum_levels, max(1, int(np.ceil(np.log2(settings.near_width_px + 1)))))
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(levels)
    blend_mask = (band.astype(np.uint8) * 255)
    blender.prepare((0, 0, old.shape[1], old.shape[0]))
    blender.feed(old.astype(np.int16), blend_mask, (0, 0))
    blender.feed(new.astype(np.int16), blend_mask, (0, 0))
    blended, weight = blender.blend(None, None)
    if blended is None or weight is None:
        raise RuntimeError("near MultiBand failed")
    output = result.copy()
    output[band] = np.clip(blended, 0, 255).astype(np.uint8)[band]
    return output, band, VideoNearBlendAudit(int(eligible.sum()), int(band.sum()), 0, settings.near_width_px, levels, True)


__all__ = ["VideoNearBlendAudit", "VideoNearBlendConfig", "apply_near_multiband", "build_near_blend_eligible_mask"]
