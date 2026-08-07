"""RGB/DIS-derived hard protection masks for v6 GraphCut and blending.

No depth or point cloud is read here.  Object masks remain explicit upstream
evidence; absent object evidence is represented by an empty mask, not a
guessed semantic region.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .video_visual_renderer import VideoDISPairEvidence


@dataclass(frozen=True)
class VideoHardGuardAudit:
    line_guard_pixels: int
    object_outer_boundary_pixels: int
    thin_structure_pixels: int
    occlusion_risk_pixels: int
    hard_owner_old_pixels: int
    hard_owner_new_pixels: int


@dataclass(frozen=True)
class VideoHardGuards:
    line_guard: np.ndarray
    object_outer_boundary: np.ndarray
    thin_structure: np.ndarray
    occlusion_risk: np.ndarray
    protected: np.ndarray
    hard_owner_old: np.ndarray
    hard_owner_new: np.ndarray
    audit: VideoHardGuardAudit


def build_video_hard_guards(
    old_bgr: np.ndarray, new_bgr: np.ndarray, evidence: VideoDISPairEvidence, *,
    object_mask: np.ndarray | None = None, prefer_new_mask: np.ndarray | None = None,
    old_valid: np.ndarray | None = None, new_valid: np.ndarray | None = None,
    edge_guard_radius_px: int = 2,
) -> VideoHardGuards:
    """Build exclusive hard-owner masks from RGB structure and cached DIS risk."""
    old, new = np.asarray(old_bgr), np.asarray(new_bgr)
    if old.shape != new.shape or old.ndim != 3 or old.shape[2] not in {3, 4}:
        raise ValueError("hard guards require matching BGR/BGRA source crops")
    shape = old.shape[:2]
    if evidence.occlusion_risk_mask.shape != shape:
        raise ValueError("DIS evidence must match hard-guard crop")
    if edge_guard_radius_px < 0 or edge_guard_radius_px > 8:
        raise ValueError("edge guard radius must be in [0, 8]")
    old_gray = cv2.cvtColor(old[..., :3], cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new[..., :3], cv2.COLOR_BGR2GRAY)
    thin = cv2.Canny(old_gray, 80, 160) > 0
    thin |= cv2.Canny(new_gray, 80, 160) > 0
    line = thin.copy()
    if edge_guard_radius_px:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_guard_radius_px * 2 + 1,) * 2)
        line = cv2.dilate(line.astype(np.uint8), kernel).astype(bool)
    # Raw Canny support protects cables and other thin structures even if they
    # are too short to be promoted to a long-line quality gate.
    object_region = np.zeros(shape, dtype=bool) if object_mask is None else np.asarray(object_mask, dtype=bool)
    if object_region.shape != shape:
        raise ValueError("object mask must match hard-guard crop")
    object_outer = cv2.morphologyEx(object_region.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    occlusion = np.asarray(evidence.occlusion_risk_mask, dtype=bool)
    # A detected object and its context collar are owner-locked in full, not
    # merely at the outer contour.  This prevents GraphCut or MultiBand from
    # traversing a foreground object while preserving a separate boundary
    # count for the audit.
    protected = line | thin | object_region | object_outer | occlusion
    prefer_new = np.zeros(shape, dtype=bool) if prefer_new_mask is None else np.asarray(prefer_new_mask, dtype=bool)
    if prefer_new.shape != shape:
        raise ValueError("preferred owner mask must match hard-guard crop")
    old_support = np.ones(shape, dtype=bool) if old_valid is None else np.asarray(old_valid, dtype=bool)
    new_support = np.ones(shape, dtype=bool) if new_valid is None else np.asarray(new_valid, dtype=bool)
    if old_support.shape != shape or new_support.shape != shape:
        raise ValueError("hard-guard valid support must match the corridor")
    # A protection label can only name a real source that actually covers the
    # pixel.  In a one-sided valid region, forcing the absent source would
    # create an unowned final pixel and must never reach GraphCut.
    protected &= old_support | new_support
    hard_new = protected & prefer_new & new_support
    hard_old = protected & ~prefer_new & old_support
    return VideoHardGuards(
        line, object_outer, thin, occlusion, protected, hard_old, hard_new,
        VideoHardGuardAudit(int(line.sum()), int(object_outer.sum()), int(thin.sum()), int(occlusion.sum()), int(hard_old.sum()), int(hard_new.sum())),
    )


def audit_guard_owner_intersection(choose_new: np.ndarray, guards: VideoHardGuards) -> int:
    """Return owner violations; zero proves every protected pixel is hard-owned."""
    labels = np.asarray(choose_new, dtype=bool)
    if labels.shape != guards.protected.shape:
        raise ValueError("GraphCut labels must match hard guards")
    return int(np.count_nonzero((guards.hard_owner_old & labels) | (guards.hard_owner_new & ~labels)))


__all__ = ["VideoHardGuardAudit", "VideoHardGuards", "audit_guard_owner_intersection", "build_video_hard_guards"]
