"""RGB/DIS object evidence for the v6 near-protected alignment path."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .video_visual_renderer import VideoDISPairEvidence


@dataclass(frozen=True)
class VideoObjectMaskConfig:
    minimum_component_pixels: int = 64
    residual_mad_multiplier: float = 2.5
    residual_floor_px: float = 1.5
    minimum_stable_iou: float = 0.55
    rectangular_fill_minimum: float = 0.72
    rectangular_minimum_side_px: int = 16


@dataclass(frozen=True)
class VideoObjectComponentAudit:
    label: int
    area_pixels: int
    bounding_box_xywh: tuple[int, int, int, int]
    collar_px: int
    stable_across_pair: bool
    rectangular: bool
    homography_eligible: bool


@dataclass(frozen=True)
class VideoObjectMaskResult:
    candidate_mask: np.ndarray
    protected_mask: np.ndarray
    homography_mask: np.ndarray
    residual_threshold_px: float
    components: tuple[VideoObjectComponentAudit, ...]


def _residual_mask(flow: np.ndarray, reliable: np.ndarray, config: VideoObjectMaskConfig) -> tuple[np.ndarray, float]:
    vectors = np.asarray(flow, np.float32)[reliable]
    if vectors.size == 0:
        return np.zeros(reliable.shape, bool), float("inf")
    centre = np.median(vectors, axis=0)
    residual = np.linalg.norm(np.asarray(flow, np.float32) - centre, axis=2)
    values = residual[reliable]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(config.residual_floor_px, median + config.residual_mad_multiplier * 1.4826 * mad)
    return reliable & np.isfinite(residual) & (residual >= threshold), threshold


def _collar_for_box(width: int, height: int) -> int:
    minimum = min(width, height)
    maximum = max(width, height)
    if minimum <= 12:
        return int(np.clip(round(0.10 * maximum), 4, 10))
    if width * height >= 2_000:
        return int(np.clip(round(0.06 * maximum), 8, 20))
    return int(np.clip(round(0.07 * maximum), 6, 16))


def _rectangular(component: np.ndarray, box: tuple[int, int, int, int], config: VideoObjectMaskConfig) -> bool:
    x, y, width, height = box
    if min(width, height) < config.rectangular_minimum_side_px:
        return False
    area = int(np.count_nonzero(component))
    if area / float(width * height) < config.rectangular_fill_minimum:
        return False
    contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return False
    contour = contours[0]
    approximation = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
    return len(approximation) == 4 and bool(cv2.isContourConvex(approximation))


def build_video_object_masks(
    evidence: VideoDISPairEvidence,
    *,
    strong_protection: np.ndarray,
    config: VideoObjectMaskConfig | None = None,
) -> VideoObjectMaskResult:
    """Derive candidate objects from motion residuals, never semantics.

    Canny/thin/occlusion protection is supplied by the caller and excluded
    before component extraction.  A component may protect ownership without
    being eligible for a planar homography.
    """

    settings = config or VideoObjectMaskConfig()
    reliable = np.asarray(evidence.reliable_mask, bool) & ~np.asarray(evidence.occlusion_risk_mask, bool)
    protection = np.asarray(strong_protection, bool)
    if reliable.shape != protection.shape:
        raise ValueError("object-mask protection must match DIS evidence")
    forward, threshold = _residual_mask(evidence.flow_forward, reliable, settings)
    backward, _ = _residual_mask(evidence.flow_backward, reliable, settings)
    # ``backward`` lives in new-image coordinates.  Assess stability only at
    # the forward-DIS correspondence of each old-image candidate, rather than
    # comparing two unrelated canvas coordinates.  This keeps a translating
    # rectangular panel eligible while rejecting a changing fan or cable.
    height, width = forward.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    forward_flow = np.asarray(evidence.flow_forward, dtype=np.float32)
    backward_at_forward = cv2.remap(
        backward.astype(np.uint8), xx + forward_flow[..., 0], yy + forward_flow[..., 1],
        cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    # First keep the connected high-motion region intact.  Strong Canny/line
    # and DIS-occlusion evidence is a *classification* constraint: it locks
    # ownership and makes the region ineligible for homography, but must not
    # cut a real object into artificial fragments before its collar is built.
    candidates = forward
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates.astype(np.uint8), connectivity=8)
    candidate = np.zeros_like(candidates)
    protected = np.zeros_like(candidates)
    homography = np.zeros_like(candidates)
    audits: list[VideoObjectComponentAudit] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < settings.minimum_component_pixels:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels == label
        candidate |= component
        # This is a conservative adjacent-frame shape stability observation,
        # not an inferred semantic class.
        union = component | backward_at_forward
        stability = float(np.count_nonzero(component & backward_at_forward)) / max(1, int(np.count_nonzero(union)))
        stable = stability >= settings.minimum_stable_iou
        rectangular = _rectangular(component, (x, y, width, height), settings)
        collar = _collar_for_box(width, height)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (collar * 2 + 1, collar * 2 + 1))
        protected |= cv2.dilate(component.astype(np.uint8), kernel).astype(bool)
        eligible = bool(stable and rectangular and not np.any(component & protection))
        if eligible:
            homography |= component
        audits.append(VideoObjectComponentAudit(
            label, area, (x, y, width, height), collar, stable, rectangular, eligible,
        ))
    return VideoObjectMaskResult(candidate, protected, homography, threshold, tuple(audits))


__all__ = [
    "VideoObjectComponentAudit", "VideoObjectMaskConfig", "VideoObjectMaskResult",
    "build_video_object_masks",
]
