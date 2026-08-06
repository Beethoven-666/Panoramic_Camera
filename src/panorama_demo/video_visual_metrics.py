"""Deterministic objective measurements used by video experiments.

Metrics intentionally consume only an already-rendered panorama and its owner
map.  They never feed a decision back into rendering, which keeps audit and
measurement runs pixel invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class VisualGrades:
    structural: str
    visual: str
    performance: str
    overall: str

    def as_dict(self) -> dict[str, str]:
        return {
            "structural": self.structural,
            "visual": self.visual,
            "performance": self.performance,
            "overall": self.overall,
        }


def _owner_components(owner: np.ndarray) -> tuple[int, int]:
    components = 0
    fragments = 0
    for frame_id in np.unique(owner):
        if frame_id < 0:
            continue
        mask = (owner == frame_id).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        components += max(count - 1, 0)
        fragments += int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] < 64))
    return components, fragments


def owner_topology_metrics(owner: np.ndarray) -> dict[str, int]:
    if owner.ndim != 2 or owner.size == 0:
        raise ValueError("Owner map must be a non-empty two-dimensional array")
    components, fragments = _owner_components(owner.astype(np.int32, copy=False))
    valid = owner >= 0
    transitions = int(np.count_nonzero(valid[:, 1:] & valid[:, :-1] & (owner[:, 1:] != owner[:, :-1])))
    return {
        "active_owner_count": int(np.unique(owner[valid]).size),
        "owner_components": components,
        "owner_small_fragments": fragments,
        "horizontal_owner_transitions": transitions,
    }


def panorama_detail_metrics(panorama_bgr: np.ndarray) -> dict[str, float]:
    if panorama_bgr.dtype != np.uint8 or panorama_bgr.ndim != 3 or panorama_bgr.shape[2] != 3:
        raise ValueError("Panorama must be an 8-bit BGR image")
    gray = cv2.cvtColor(panorama_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    gradients_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gradients_x, gradients_y)
    return {
        "laplacian_variance": float(laplacian.var()),
        "gradient_p50": float(np.percentile(magnitude, 50)),
        "gradient_p95": float(np.percentile(magnitude, 95)),
    }


def evaluate_visual_metrics(
    panorama_bgr: np.ndarray,
    owner: np.ndarray,
    *,
    post_capture_seconds: float | None = None,
    maximum_post_seconds: float | None = None,
    structural_ok: bool = True,
) -> tuple[dict[str, Any], VisualGrades]:
    topology = owner_topology_metrics(owner)
    detail = panorama_detail_metrics(panorama_bgr)
    structural = "A" if structural_ok else "F"
    visual = "A" if topology["owner_small_fragments"] == 0 else "C"
    performance = "A"
    if maximum_post_seconds is not None and post_capture_seconds is not None and post_capture_seconds > maximum_post_seconds:
        performance = "C"
    overall = "F" if structural == "F" else ("A" if visual == performance == "A" else "C")
    return {"owner_topology": topology, "detail": detail}, VisualGrades(structural, visual, performance, overall)
