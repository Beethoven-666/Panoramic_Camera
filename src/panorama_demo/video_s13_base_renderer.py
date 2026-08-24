"""Immutable P0 owner-only renderer and provenance adapter for S1.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .session import CameraIntrinsics
from .calibrated_remap import camera_matrix, undistortion_maps
from .cuda_backend import remap as accelerated_remap
from .video_s12_renderer import render_s012_stage_a
from .video_s12_renderer import _roi_inverse_map
from .video_s12_schedule import S012ColumnAssignment, S012Schedule


@dataclass(frozen=True)
class S13P0Result:
    image: np.ndarray
    valid_mask: np.ndarray
    pixel_provenance: dict[str, np.ndarray]
    column_provenance: dict[str, np.ndarray]
    remap_invocations: int
    duplicate_write_count: int


@dataclass(frozen=True)
class S13P0AssignmentRender:
    """Exact P0 pixels and provenance for one half-open assignment ROI."""

    assignment_index: int
    frame_id: int
    left_x: int
    right_x: int
    image_roi: np.ndarray
    valid_roi: np.ndarray
    pixel_provenance_roi: dict[str, np.ndarray]
    column_provenance: dict[str, np.ndarray]


def render_s13_p0_assignment(
    assignment: S012ColumnAssignment,
    calibration: CameraIntrinsics,
    image: np.ndarray,
    *,
    placement_method: str,
    selected_hypothesis_id: int = -1,
    resident_remap: Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> S13P0AssignmentRender:
    """Render one assignment with the same inverse map as batch P0."""

    camera_matrix(calibration)
    if assignment.zero_width or assignment.right_x <= assignment.left_x:
        raise ValueError("S1.3 assignment ROI renderer requires a positive-width assignment")
    source = np.asarray(image)
    height = int(calibration.height)
    if source.shape != (height, int(calibration.width), 3) or source.dtype != np.uint8:
        raise ValueError("S1.3 assignment RGB source must be calibrated-size uint8 BGR")
    map_u, map_v, valid = _roi_inverse_map(
        calibration,
        assignment.center_x,
        assignment.left_x,
        assignment.right_x,
        undistortion_maps(calibration),
    )
    sampled = (
        resident_remap(assignment.frame_id, source, map_u, map_v)
        if resident_remap is not None
        else accelerated_remap(
            source,
            map_u,
            map_v,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    )
    image_roi = np.zeros_like(sampled)
    image_roi[valid] = sampled[valid]
    shape = valid.shape
    invalid_i32 = np.full(shape, -1, dtype=np.int32)
    owner_frame = invalid_i32.copy()
    owner_source = invalid_i32.copy()
    assignment_index = invalid_i32.copy()
    source_u = np.full(shape, np.nan, dtype=np.float32)
    source_v = np.full(shape, np.nan, dtype=np.float32)
    hypothesis = invalid_i32.copy()
    placement = np.full(shape, -1, dtype=np.int16)
    owner_frame[valid] = int(assignment.frame_id)
    owner_source[valid] = int(assignment.source_index)
    assignment_index[valid] = int(assignment.assignment_index)
    source_u[valid] = map_u[valid]
    source_v[valid] = map_v[valid]
    hypothesis[valid] = int(selected_hypothesis_id)
    placement[valid] = int(assignment.assignment_index)
    pixel = {
        "owner_frame_id": owner_frame,
        "owner_source_index": owner_source,
        "assignment_index": assignment_index,
        "source_u": source_u,
        "source_v": source_v,
        "valid": valid,
        "selected_motion_hypothesis_id": hypothesis,
        "placement_method_code": placement,
        "geometry_transaction_id": invalid_i32.copy(),
        "seam_transaction_id": invalid_i32.copy(),
        "photometric_transaction_id": invalid_i32.copy(),
        "blend_transaction_id": invalid_i32.copy(),
        "secondary_frame_id": invalid_i32.copy(),
        "secondary_source_u": np.full(shape, np.nan, dtype=np.float32),
        "secondary_source_v": np.full(shape, np.nan, dtype=np.float32),
        "secondary_weight": np.zeros(shape, dtype=np.float32),
        "placement_method_names": np.asarray((placement_method,)),
    }
    canvas_x = np.arange(assignment.left_x, assignment.right_x, dtype=np.float64)
    rectified_x = float(calibration.cx) + canvas_x - float(assignment.center_x)
    column = {
        "owner_frame_id": np.full(assignment.width, assignment.frame_id, dtype=np.int32),
        "owner_source_index": np.full(assignment.width, assignment.source_index, dtype=np.int32),
        "assignment_index": np.full(assignment.width, assignment.assignment_index, dtype=np.int32),
        "source_u": rectified_x.astype(np.float32),
        "valid": np.ones(assignment.width, dtype=bool),
    }
    return S13P0AssignmentRender(
        assignment_index=assignment.assignment_index,
        frame_id=assignment.frame_id,
        left_x=assignment.left_x,
        right_x=assignment.right_x,
        image_roi=np.ascontiguousarray(image_roi),
        valid_roi=valid,
        pixel_provenance_roi=pixel,
        column_provenance=column,
    )


def render_s13_p0(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    *,
    placement_methods: tuple[str, ...],
    selected_hypothesis_ids: tuple[int, ...] | None = None,
    resident_remap: Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    resident_device_remap: Callable[[int, np.ndarray, np.ndarray, np.ndarray], Any] | None = None,
    resident_stage: Any | None = None,
) -> S13P0Result:
    result = render_s012_stage_a(
        schedule, calibration, image_loader, resident_remap=resident_remap,
        resident_device_remap=resident_device_remap, resident_stage=resident_stage,
    )
    contributors = {assignment.frame_id for assignment in schedule.assignments if not assignment.zero_width}
    if result.remap_invocations != len(contributors) or len(result.decoded_frame_ids) != len(contributors):
        raise ValueError("S1.3 each contributor source must be formally remapped exactly once")
    height, width = result.valid_mask.shape
    source_index = result.assignment_index.copy()
    placement_codes = np.full((height, width), -1, dtype=np.int16)
    for index in range(len(schedule.assignments)):
        placement_codes[result.assignment_index == index] = index
    invalid_i32 = np.full((height, width), -1, dtype=np.int32)
    hypothesis_ids = invalid_i32.copy()
    if selected_hypothesis_ids is not None:
        if len(selected_hypothesis_ids) != len(schedule.assignments):
            raise ValueError("S1.3 motion hypothesis ids must align with source assignments")
        for index, hypothesis_id in enumerate(selected_hypothesis_ids):
            hypothesis_ids[result.assignment_index == index] = int(hypothesis_id)
    zeros = np.zeros((height, width), dtype=np.float32)
    pixel = {
        "owner_frame_id": result.owner_frame_id,
        "owner_source_index": source_index,
        "assignment_index": result.assignment_index,
        "source_u": result.source_u,
        "source_v": result.source_v,
        "valid": result.valid_mask,
        "selected_motion_hypothesis_id": hypothesis_ids,
        "placement_method_code": placement_codes,
        "geometry_transaction_id": invalid_i32.copy(),
        "seam_transaction_id": invalid_i32.copy(),
        "photometric_transaction_id": invalid_i32.copy(),
        "blend_transaction_id": invalid_i32.copy(),
        "secondary_frame_id": invalid_i32.copy(),
        "secondary_source_u": np.full((height, width), np.nan, dtype=np.float32),
        "secondary_source_v": np.full((height, width), np.nan, dtype=np.float32),
        "secondary_weight": zeros,
        "placement_method_names": np.asarray(placement_methods),
    }
    column = {
        "owner_frame_id": result.column_frame_id,
        "owner_source_index": result.column_assignment_index,
        "assignment_index": result.column_assignment_index,
        "source_u": result.column_source_u,
        "valid": result.column_frame_id >= 0,
    }
    if not np.array_equal(pixel["valid"], pixel["owner_frame_id"] >= 0):
        raise ValueError("S1.3 valid pixels and primary owners disagree")
    if np.any(pixel["secondary_weight"] != 0) or np.any(pixel["secondary_frame_id"] != -1):
        raise ValueError("S1.3 P0 must remain hard-owner only")
    return S13P0Result(result.image, result.valid_mask, pixel, column, result.remap_invocations, result.duplicate_write_count)


__all__ = [
    "S13P0AssignmentRender", "S13P0Result", "render_s13_p0",
    "render_s13_p0_assignment",
]
