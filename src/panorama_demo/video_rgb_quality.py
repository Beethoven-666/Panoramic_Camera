"""Fail-closed RGB-only v6 seam quality gates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .video_graphcut_seam import VideoGraphCutAudit
from .video_hard_guards import VideoHardGuards


@dataclass(frozen=True)
class VideoRGBQualityConfig:
    seam_step_p95_max_px: float = 0.75
    seam_step_abs_max_px: float = 1.5
    staircase_run_ge_1px_over_5px_max_count: int = 0
    double_edge_max_count: int = 0
    ghost_max_count: int = 0


@dataclass(frozen=True)
class VideoRGBQualityAudit:
    owner_topology_ok: bool
    seam_step_p95_px: float | None
    seam_step_abs_max_px: float | None
    staircase_run_count: int
    double_edge_count: int
    ghost_count: int
    guard_owner_violation_count: int
    strict_quality_pass: bool
    failure_reasons: tuple[str, ...]


def _seam_steps(audits: Sequence[VideoGraphCutAudit]) -> np.ndarray:
    values: list[float] = []
    for audit in audits:
        rows = [value for value in audit.seam_x_by_row if value >= 0]
        values.extend(abs(right - left) for left, right in zip(rows, rows[1:]))
    return np.asarray(values, dtype=np.float64)


def _staircase_runs(steps: np.ndarray) -> int:
    runs = 0
    length = 0
    for value in steps:
        if value >= 1.0:
            length += 1
        else:
            runs += int(length >= 5)
            length = 0
    return runs + int(length >= 5)


def _double_edge_and_ghost_count(bgr: np.ndarray, audits: Sequence[VideoGraphCutAudit]) -> tuple[int, int]:
    """Conservative RGB-only local discontinuity observation near each seam."""
    gray = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160) > 0
    double, ghost = 0, 0
    for audit in audits:
        for row, local_x in enumerate(audit.seam_x_by_row):
            seam_x = int(local_x) + int(audit.canvas_x_offset)
            if local_x < 0 or seam_x < 2 or seam_x >= edges.shape[1] - 2 or row >= edges.shape[0]:
                continue
            window = edges[row, seam_x - 2 : seam_x + 3]
            # A continuous horizontal edge crossing a vertical seam is one
            # real line, not a double edge.  Count only distinct Canny runs
            # separated by at least one non-edge pixel in the seam window.
            starts = int(window[0]) + int(np.count_nonzero(~window[:-1] & window[1:]))
            double += int(starts >= 2)
            ghost += int(starts >= 2 and bool(window[0]) and bool(window[-1]))
    return double, ghost


def assess_video_rgb_quality(
    bgr: np.ndarray, owner_frame_id: np.ndarray, valid_mask: np.ndarray, graphcut_audits: Sequence[VideoGraphCutAudit],
    *, guards: VideoHardGuards | None = None, config: VideoRGBQualityConfig | None = None,
) -> VideoRGBQualityAudit:
    """Evaluate only final RGB/owner evidence; no geometry or depth enters this gate."""
    settings = config or VideoRGBQualityConfig()
    owner, valid = np.asarray(owner_frame_id), np.asarray(valid_mask, bool)
    if owner.shape != valid.shape or np.asarray(bgr).shape[:2] != owner.shape:
        raise ValueError("RGB quality inputs must share one final canvas")
    topology = bool(np.all(valid == (owner >= 0)))
    steps = _seam_steps(graphcut_audits)
    p95 = None if not steps.size else float(np.percentile(steps, 95.0))
    maximum = None if not steps.size else float(np.max(steps))
    staircase = _staircase_runs(steps)
    double, ghost = _double_edge_and_ghost_count(bgr, graphcut_audits)
    guard_violation = 0
    if guards is not None and guards.protected.shape == owner.shape:
        # The guard owner was applied before GraphCut.  A protected invalid
        # output is always a hard topology failure; source-specific owner
        # validation occurs beside the GraphCut call.
        guard_violation = int(np.count_nonzero(guards.protected & ~valid))
    failures: list[str] = []
    if not topology:
        failures.append("owner_topology")
    if p95 is not None and p95 > settings.seam_step_p95_max_px:
        failures.append("seam_step_p95")
    if maximum is not None and maximum > settings.seam_step_abs_max_px:
        failures.append("seam_step_abs_max")
    if staircase > settings.staircase_run_ge_1px_over_5px_max_count:
        failures.append("staircase")
    if double > settings.double_edge_max_count:
        failures.append("double_edge")
    if ghost > settings.ghost_max_count:
        failures.append("ghost")
    if guard_violation:
        failures.append("guard_owner_violation")
    return VideoRGBQualityAudit(topology, p95, maximum, staircase, double, ghost, guard_violation, not failures, tuple(failures))


__all__ = ["VideoRGBQualityAudit", "VideoRGBQualityConfig", "assess_video_rgb_quality"]
