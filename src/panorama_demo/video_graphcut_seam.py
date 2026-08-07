"""True OpenCV GraphCut seam labels for one v6 video corridor.

This is deliberately not a dynamic-programming seam.  ``GraphCutSeamFinder``
is the only optimizer called here; on a topology failure the caller must
reroute/expand a real owner or fail closed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoGraphCutConfig:
    normal_corridor_min_px: int = 96
    normal_corridor_max_px: int = 160
    rescue_corridor_max_px: int = 192
    required_height_px: int = 480
    maximum_row_step_px: int = 1
    minimum_fragment_pixels: int = 8


@dataclass(frozen=True)
class VideoGraphCutAudit:
    graphcut_called: bool
    rescue_corridor_used: bool
    seam_x_by_row: tuple[int, ...]
    maximum_adjacent_row_step_px: int | None
    owner_island_count: int
    small_fragment_count: int
    valid_pixel_exactly_one_owner: bool
    accepted: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class VideoGraphCutResult:
    choose_new: np.ndarray
    audit: VideoGraphCutAudit


def _validate(image: np.ndarray, mask: np.ndarray, config: VideoGraphCutConfig) -> None:
    if image.ndim != 3 or image.shape[2] not in {3, 4} or image.shape[:2] != mask.shape:
        raise ValueError("GraphCut image/mask shapes must be HxWx3/4 and HxW")
    height, width = mask.shape
    if height != config.required_height_px:
        raise ValueError("v6 GraphCut requires a 480px corridor height")
    if width < config.normal_corridor_min_px or width > config.rescue_corridor_max_px:
        raise ValueError(f"GraphCut corridor width {width}px is outside v6 normal/rescue bounds")


def _components(mask: np.ndarray, minimum_pixels: int) -> tuple[int, int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    return max(0, count - 2), int(np.count_nonzero(sizes < minimum_pixels))


def _row_seam(choose_new: np.ndarray, overlap: np.ndarray) -> tuple[tuple[int, ...], int | None, bool]:
    seam: list[int] = []
    monotone = True
    for row in range(overlap.shape[0]):
        columns = np.flatnonzero(overlap[row])
        if columns.size == 0:
            seam.append(-1)
            continue
        splits = np.flatnonzero(np.diff(columns) > 1) + 1
        runs = np.split(columns, splits)
        # Invalid projection holes are not a seam transition.  Audit each
        # contiguous overlap run, and use the largest one as that row's seam
        # coordinate so row-step checks never bridge an invalid gap.
        run = max(runs, key=len)
        for candidate in runs:
            labels = choose_new[row, candidate]
            if np.count_nonzero(labels[1:] != labels[:-1]) > 1:
                monotone = False
        labels = choose_new[row, run]
        transitions = np.flatnonzero(labels[1:] != labels[:-1])
        seam.append(int(run[transitions[0] + 1]) if transitions.size == 1 else int(run[0] if labels[0] else run[-1] + 1))
    known = [value for value in seam if value >= 0]
    maximum = None if len(known) < 2 else max(abs(right - left) for left, right in zip(known, known[1:]))
    return tuple(seam), maximum, monotone


def solve_video_graphcut_seam(
    old_bgr: np.ndarray, new_bgr: np.ndarray, old_valid: np.ndarray, new_valid: np.ndarray, *,
    hard_owner_old: np.ndarray | None = None, hard_owner_new: np.ndarray | None = None,
    config: VideoGraphCutConfig | None = None,
) -> VideoGraphCutResult:
    """Run one binary GraphCut with no custom-cost or DP fallback.

    The OpenCV Python binding accepts colour/gradient graph costs rather than
    an arbitrary cost tensor.  Hard protection is represented by removing the
    competing label from the corresponding graph-cut mask.
    """
    settings = config or VideoGraphCutConfig()
    old_valid, new_valid = np.asarray(old_valid, bool), np.asarray(new_valid, bool)
    _validate(np.asarray(old_bgr), old_valid, settings)
    _validate(np.asarray(new_bgr), new_valid, settings)
    if old_valid.shape != new_valid.shape:
        raise ValueError("GraphCut valid masks must share a corridor")
    old_hard = np.zeros_like(old_valid) if hard_owner_old is None else np.asarray(hard_owner_old, bool)
    new_hard = np.zeros_like(new_valid) if hard_owner_new is None else np.asarray(hard_owner_new, bool)
    if old_hard.shape != old_valid.shape or new_hard.shape != old_valid.shape or np.any(old_hard & new_hard):
        raise ValueError("GraphCut hard owner masks must be exclusive and match the corridor")
    overlap = old_valid & new_valid
    masks = [
        ((old_valid & ~new_hard).astype(np.uint8) * 255),
        ((new_valid & ~old_hard).astype(np.uint8) * 255),
    ]
    if not np.any((masks[0] > 0) & (masks[1] > 0)):
        raise RuntimeError("GraphCut has no unprotected competing support")
    finder = cv2.detail.GraphCutSeamFinder("COST_COLOR_GRAD")
    try:
        finder.find([
            np.ascontiguousarray(np.asarray(old_bgr)[..., :3].astype(np.float32)),
            np.ascontiguousarray(np.asarray(new_bgr)[..., :3].astype(np.float32)),
        ], [(0, 0), (0, 0)], masks)
    except cv2.error as error:
        raise RuntimeError("OpenCV GraphCut failed on the protected corridor") from error
    old_label, new_label = (mask > 0 for mask in masks)
    choose_new = (new_valid & ~old_valid) | (overlap & new_label & ~old_label)
    choose_new[new_hard] = True
    choose_new[old_hard] = False
    valid = old_valid | new_valid
    exact = bool(np.all(~valid | ((~choose_new & old_valid) | (choose_new & new_valid))))
    seam, maximum_step, monotone = _row_seam(choose_new, overlap)
    islands, small = _components(choose_new & overlap, settings.minimum_fragment_pixels)
    accepted = bool(exact and monotone and islands == 0 and small == 0 and (maximum_step is None or maximum_step <= settings.maximum_row_step_px))
    reason = None if accepted else "graphcut_topology_or_row_step_gate_failed"
    return VideoGraphCutResult(choose_new, VideoGraphCutAudit(True, old_valid.shape[1] > settings.normal_corridor_max_px, seam, maximum_step, islands, small, exact, accepted, reason))


__all__ = ["VideoGraphCutAudit", "VideoGraphCutConfig", "VideoGraphCutResult", "solve_video_graphcut_seam"]
