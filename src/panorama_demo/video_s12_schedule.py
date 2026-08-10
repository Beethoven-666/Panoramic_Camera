"""Fixed midpoint-Voronoi schedule for the standalone S012 renderer.

This module owns only calibrated-target horizontal coordinates.  It does not
read an S1 layout, inspect image content, or create object-specific owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Sequence

import numpy as np

from .calibrated_remap import CalibratedIntrinsics, camera_matrix


@dataclass(frozen=True)
class S012ColumnAssignment:
    """One source's half-open interval in the fixed canvas partition."""

    assignment_index: int
    source_index: int
    frame_id: int
    center_x: float
    left_x: int
    right_x: int
    zero_width: bool

    @property
    def width(self) -> int:
        return self.right_x - self.left_x


@dataclass(frozen=True)
class S012Schedule:
    """Complete fixed owner schedule, including fast column provenance."""

    canvas_left: int
    canvas_right: int
    canvas_width: int
    canvas_height: int
    assignments: tuple[S012ColumnAssignment, ...]
    boundaries: tuple[int, ...]
    column_frame_id: np.ndarray
    column_source_u: np.ndarray
    column_assignment_index: np.ndarray


def _rounded_midpoint(left: float, right: float) -> int:
    return int(floor((left + right) * 0.5 + 0.5))


def validate_s012_schedule(schedule: S012Schedule) -> None:
    """Fail closed unless columns form one exact half-open partition."""

    if schedule.canvas_left != 0:
        raise ValueError("S012 canvas_left must be zero")
    if schedule.canvas_width <= 0 or schedule.canvas_height <= 0:
        raise ValueError("S012 canvas dimensions must be positive")
    if schedule.canvas_right != schedule.canvas_width:
        raise ValueError("S012 canvas_right must equal canvas_width")
    expected_shapes = (
        schedule.column_frame_id.shape,
        schedule.column_source_u.shape,
        schedule.column_assignment_index.shape,
    )
    if any(shape != (schedule.canvas_width,) for shape in expected_shapes):
        raise ValueError("S012 column provenance does not match canvas width")
    if len(schedule.assignments) == 0:
        raise ValueError("S012 schedule must contain at least one source")
    if len(schedule.boundaries) != len(schedule.assignments) + 1:
        raise ValueError("S012 boundary count is inconsistent")
    if schedule.boundaries[0] != 0 or schedule.boundaries[-1] != schedule.canvas_width:
        raise ValueError("S012 schedule does not preserve the full first/last field")

    writes = np.zeros(schedule.canvas_width, dtype=np.uint8)
    for index, assignment in enumerate(schedule.assignments):
        if assignment.assignment_index != index or assignment.source_index != index:
            raise ValueError("S012 assignment indices must be dense and ordered")
        if assignment.left_x != schedule.boundaries[index] or assignment.right_x != schedule.boundaries[index + 1]:
            raise ValueError("S012 assignments must use the declared half-open boundaries")
        if not (0 <= assignment.left_x <= assignment.right_x <= schedule.canvas_width):
            raise ValueError("S012 assignment lies outside the canvas")
        if assignment.zero_width != (assignment.width == 0):
            raise ValueError("S012 zero_width marker is inconsistent")
        if assignment.width:
            claimed = writes[assignment.left_x : assignment.right_x]
            if np.any(claimed):
                raise ValueError("S012 duplicate column assignment")
            claimed[...] = 1
            columns = slice(assignment.left_x, assignment.right_x)
            if not np.all(schedule.column_frame_id[columns] == assignment.frame_id):
                raise ValueError("S012 column frame provenance is inconsistent")
            if not np.all(schedule.column_assignment_index[columns] == index):
                raise ValueError("S012 column assignment provenance is inconsistent")
    if not np.all(writes == 1):
        raise ValueError("S012 canvas columns must be assigned exactly once")


def build_s012_schedule(
    frame_ids: Sequence[int],
    source_centers_x: Sequence[float],
    calibration: CalibratedIntrinsics,
    *,
    hard_internal_width_px: int | None = 24,
) -> S012Schedule:
    """Build the immutable M6 midpoint-Voronoi canvas and column owners.

    ``hard_internal_width_px`` validates the M5 invariant.  Pass ``None`` only
    for diagnostic/unit constructions that intentionally exercise wider
    internal strips; it never changes the fixed boundaries.
    """

    camera_matrix(calibration)
    ids = tuple(int(value) for value in frame_ids)
    centers = np.asarray(source_centers_x, dtype=np.float64)
    if not ids or centers.shape != (len(ids),):
        raise ValueError("S012 frame ids and source centers must be non-empty and aligned")
    if len(set(ids)) != len(ids):
        raise ValueError("S012 render sources must have unique real frame ids")
    if not np.isfinite(centers).all():
        raise ValueError("S012 source centers must be finite")
    if not np.isclose(centers[0], float(calibration.cx), rtol=0.0, atol=1e-6):
        raise ValueError("S012 first source center must equal calibrated cx")
    if np.any(np.diff(centers) < 0.0):
        raise ValueError("S012 source centers must be monotonic")

    canvas_width = int(ceil(centers[-1] + (int(calibration.width) - float(calibration.cx))))
    if canvas_width <= 0:
        raise ValueError("S012 automatic canvas is empty")
    interior = tuple(_rounded_midpoint(float(a), float(b)) for a, b in zip(centers[:-1], centers[1:]))
    boundaries = (0, *interior, canvas_width)
    if any(left > right for left, right in zip(boundaries[:-1], boundaries[1:])):
        raise ValueError("S012 integer Voronoi boundaries are not monotonic")

    column_frame_id = np.full(canvas_width, -1, dtype=np.int32)
    column_source_u = np.full(canvas_width, np.nan, dtype=np.float32)
    column_assignment_index = np.full(canvas_width, -1, dtype=np.int32)
    assignments: list[S012ColumnAssignment] = []
    for index, (frame_id, center_x, left_x, right_x) in enumerate(
        zip(ids, centers, boundaries[:-1], boundaries[1:])
    ):
        assignment = S012ColumnAssignment(
            assignment_index=index,
            source_index=index,
            frame_id=frame_id,
            center_x=float(center_x),
            left_x=int(left_x),
            right_x=int(right_x),
            zero_width=bool(left_x == right_x),
        )
        assignments.append(assignment)
        if assignment.width:
            columns = np.arange(left_x, right_x, dtype=np.float64)
            column_frame_id[left_x:right_x] = frame_id
            column_source_u[left_x:right_x] = (
                float(calibration.cx) + columns - float(center_x)
            ).astype(np.float32)
            column_assignment_index[left_x:right_x] = index

    if hard_internal_width_px is not None:
        limit = int(hard_internal_width_px)
        if limit <= 0:
            raise ValueError("S012 hard internal strip width must be positive")
        too_wide = [item for item in assignments[1:-1] if item.width > limit]
        if too_wide:
            raise ValueError("S012 internal strip exceeds the hard width limit")

    schedule = S012Schedule(
        canvas_left=0,
        canvas_right=canvas_width,
        canvas_width=canvas_width,
        canvas_height=int(calibration.height),
        assignments=tuple(assignments),
        boundaries=tuple(int(value) for value in boundaries),
        column_frame_id=column_frame_id,
        column_source_u=column_source_u,
        column_assignment_index=column_assignment_index,
    )
    validate_s012_schedule(schedule)
    return schedule
