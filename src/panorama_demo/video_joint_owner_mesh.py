"""Candidate-only C12 5--7 source joint owner/final-grid optimiser.

It consumes only already resident, genuine RGB-D source evidence and emits a
local owner label plus the selected source's *actual final inverse grid*.
Its caller must use both arrays for final real-RGB sampling.  This module has
no colour-generation, pose, or source-synthesis authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class JointOwnerMeshError(ValueError):
    """A C12 candidate window has insufficient genuine, safe evidence."""


@dataclass(frozen=True)
class JointOwnerMeshConfig:
    """Closed, immutable controls for the C12 local experiment."""

    minimum_window_frames: int = 5
    maximum_window_frames: int = 7
    rgb_weight: float = 1.0
    raft_confidence_weight: float = 0.8
    depth_weight: float = 1.0
    source_center_weight: float = 0.15
    sharpness_weight: float = 0.2
    owner_switch_penalty: float = 0.6
    minimum_owner_run_pixels: int = 3

    def __post_init__(self) -> None:
        if not 5 <= self.minimum_window_frames <= self.maximum_window_frames <= 7:
            raise ValueError("C12 requires an immutable 5--7 source local window")
        if self.minimum_owner_run_pixels < 1:
            raise ValueError("C12 minimum_owner_run_pixels must be positive")
        if any(not np.isfinite(value) or value < 0.0 for value in (
            self.rgb_weight, self.raft_confidence_weight, self.depth_weight,
            self.source_center_weight, self.sharpness_weight, self.owner_switch_penalty,
        )):
            raise ValueError("C12 data-cost weights must be finite and non-negative")


@dataclass(frozen=True)
class JointOwnerMeshResult:
    """C12 selected real-owner final grids, ready for genuine RGB sampling."""

    owner_frame_id: np.ndarray
    final_grid_xy: np.ndarray
    changed_mask: np.ndarray
    audit: dict[str, object]


def _validate_ids(ids: Sequence[int], config: JointOwnerMeshConfig) -> tuple[int, ...]:
    values = tuple(ids)
    if not config.minimum_window_frames <= len(values) <= config.maximum_window_frames:
        raise JointOwnerMeshError("C12 needs a genuine chronological 5--7 source window")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise JointOwnerMeshError("C12 source ids must be non-negative integer real frame ids")
    if tuple(sorted(values)) != values or len(set(values)) != len(values):
        raise JointOwnerMeshError("C12 source ids must be unique and chronological")
    return values


def _as_cost(value: Any, shape: tuple[int, int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all() or np.any(array < 0.0):
        raise JointOwnerMeshError(f"C12 {name} must be finite non-negative KxHxW evidence")
    return array


def optimise_joint_owner_final_grids(
    *,
    source_frame_ids: Sequence[int],
    final_grid_xy: Any,
    source_valid_mask: Any,
    rgb_cost: Any,
    raft_confidence: Any,
    depth_cost: Any,
    source_center_cost: Any,
    sharpness_cost: Any,
    baseline_owner_frame_id: Any,
    seam_protected_mask: Any,
    line_protected_mask: Any,
    object_protected_mask: Any,
    config: JointOwnerMeshConfig = JointOwnerMeshConfig(),
) -> JointOwnerMeshResult:
    """Optimise labels and gather genuine final grids for one C12 window.

    Every source must supply its own final grid.  The dynamic program permits
    only chronological label advance, locks seam/line/object pixels to the
    existing owner, and enforces a minimum horizontal owner run before a
    switch.  A result which leaves the baseline wholly unchanged is rejected:
    C12 must never be reported as executed merely because its optimiser ran.
    The caller samples the returned grid from its selected real source and
    records the resulting output application.
    """

    ids = _validate_ids(source_frame_ids, config)
    grid = np.asarray(final_grid_xy, dtype=np.float64)
    if grid.ndim != 4 or grid.shape[0] != len(ids) or grid.shape[-1] != 2 or not np.isfinite(grid).all():
        raise JointOwnerMeshError("C12 final_grid_xy must be finite KxHxWx2 genuine source grids")
    k, height, width, _ = grid.shape
    shape = (k, height, width)
    valid = np.asarray(source_valid_mask, dtype=bool)
    if valid.shape != shape:
        raise JointOwnerMeshError("C12 source_valid_mask must match final grids")
    rgb = _as_cost(rgb_cost, shape, "rgb_cost")
    confidence = _as_cost(raft_confidence, shape, "raft_confidence")
    if np.any(confidence > 1.0):
        raise JointOwnerMeshError("C12 raft_confidence must lie in [0, 1]")
    depth = _as_cost(depth_cost, shape, "depth_cost")
    centre = _as_cost(source_center_cost, shape, "source_center_cost")
    sharpness = _as_cost(sharpness_cost, shape, "sharpness_cost")
    baseline = np.asarray(baseline_owner_frame_id)
    if baseline.shape != (height, width) or not np.issubdtype(baseline.dtype, np.integer):
        raise JointOwnerMeshError("C12 baseline owner must be an integer HxW map")
    protected = np.zeros((height, width), dtype=bool)
    for name, raw in (("seam", seam_protected_mask), ("line", line_protected_mask), ("object", object_protected_mask)):
        mask = np.asarray(raw, dtype=bool)
        if mask.shape != (height, width):
            raise JointOwnerMeshError(f"C12 {name} protection mask must match the local window")
        protected |= mask
    # Lower cost is better: confidence and sharpness are benefits, not new samples.
    unary = (config.rgb_weight * rgb + config.raft_confidence_weight * (1.0 - confidence)
             + config.depth_weight * depth + config.source_center_weight * centre
             + config.sharpness_weight * (1.0 / (1.0 + sharpness)))
    unary = np.where(valid, unary, np.inf)
    owner_index = np.full((height, width), -1, dtype=np.int16)
    for row in range(height):
        # state[label, capped run length]; reset at invalid gaps.
        previous: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
        parents: list[dict[tuple[int, int], tuple[int, int] | None]] = []
        costs_by_column: list[dict[tuple[int, int], float]] = []
        for col in range(width):
            candidates = np.flatnonzero(valid[:, row, col])
            if not len(candidates):
                previous = {}
                parents.append({})
                costs_by_column.append({})
                continue
            locked = np.flatnonzero(np.asarray(ids) == int(baseline[row, col])) if protected[row, col] else np.array([], dtype=int)
            if protected[row, col]:
                if len(locked) != 1 or not valid[int(locked[0]), row, col]:
                    raise JointOwnerMeshError("C12 protected seam/line/object owner lacks a genuine sample")
                candidates = locked
            current: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {}
            for label in candidates.tolist():
                best_cost, best_parent = float("inf"), None
                if not previous:
                    best_cost = float(unary[label, row, col])
                else:
                    for (prior, run), (cost, _) in previous.items():
                        if label < prior or (label != prior and run < config.minimum_owner_run_pixels):
                            continue
                        next_run = min(config.minimum_owner_run_pixels, run + 1) if label == prior else 1
                        total = cost + float(unary[label, row, col]) + (config.owner_switch_penalty if label != prior else 0.0)
                        key = (label, next_run)
                        prior_best = current.get(key, (float("inf"), None))[0]
                        if total < prior_best:
                            current[key] = (total, (prior, run))
                        if total < best_cost:
                            best_cost, best_parent = total, (prior, run)
                    continue
                current[(label, 1)] = (best_cost, best_parent)
            if not current:
                raise JointOwnerMeshError("C12 constraints leave no feasible genuine owner label")
            parents.append({state: parent for state, (_, parent) in current.items()})
            costs_by_column.append({state: cost for state, (cost, _) in current.items()})
            previous = current
        # Recover each contiguous valid island independently.
        col = width - 1
        while col >= 0:
            if not parents[col]:
                col -= 1
                continue
            end = col
            while col >= 0 and parents[col]:
                col -= 1
            start = col + 1
            terminal = min(costs_by_column[end].items(), key=lambda item: item[1])[0]
            state = terminal
            for x in range(end, start - 1, -1):
                owner_index[row, x] = state[0]
                parent = parents[x][state]
                if parent is None and x != start:
                    raise JointOwnerMeshError("C12 local owner backtracking is discontinuous")
                if parent is not None:
                    state = parent
    owners = np.full((height, width), -1, dtype=np.int32)
    for index, frame_id in enumerate(ids):
        owners[owner_index == index] = frame_id
    valid_any = valid.any(axis=0)
    if np.any(valid_any & (owners < 0)) or np.any(~valid_any & (owners >= 0)):
        raise JointOwnerMeshError("C12 result violates exact real-owner topology")
    changed = valid_any & (owners != baseline)
    if not np.any(changed):
        raise JointOwnerMeshError("C12 rejected: zero actual owner/grid changed pixels")
    rows, cols = np.indices((height, width))
    gathered = np.full((height, width, 2), np.nan, dtype=np.float64)
    selected = owner_index >= 0
    gathered[selected] = grid[owner_index[selected], rows[selected], cols[selected]]
    if not np.isfinite(gathered[selected]).all():
        raise JointOwnerMeshError("C12 selected final grid is not finite")
    return JointOwnerMeshResult(owners, gathered, changed, {
        "schema": "gemini305-video-c12-joint-owner-final-grid/v1",
        "source_frame_ids": list(ids), "window_frame_count": k,
        "data_cost_terms": ["rgb", "raft_confidence", "depth", "source_center", "sharpness"],
        "temporal_monotonicity": True, "minimum_owner_run_pixels": config.minimum_owner_run_pixels,
        "seam_line_object_protected_pixel_count": int(protected.sum()),
        "actual_changed_pixel_count": int(changed.sum()), "creates_colour": False,
        "renderer_input": True, "selected_grid_required_for_final_sampling": True,
    })


__all__ = ["JointOwnerMeshConfig", "JointOwnerMeshError", "JointOwnerMeshResult", "optimise_joint_owner_final_grids"]
