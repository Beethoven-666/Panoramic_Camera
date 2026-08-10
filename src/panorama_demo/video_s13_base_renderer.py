"""Immutable P0 owner-only renderer and provenance adapter for S1.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .session import CameraIntrinsics
from .video_s12_renderer import render_s012_stage_a
from .video_s12_schedule import S012Schedule


@dataclass(frozen=True)
class S13P0Result:
    image: np.ndarray
    valid_mask: np.ndarray
    pixel_provenance: dict[str, np.ndarray]
    column_provenance: dict[str, np.ndarray]
    remap_invocations: int
    duplicate_write_count: int


def render_s13_p0(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    *,
    placement_methods: tuple[str, ...],
) -> S13P0Result:
    result = render_s012_stage_a(schedule, calibration, image_loader)
    contributors = {assignment.frame_id for assignment in schedule.assignments if not assignment.zero_width}
    if result.remap_invocations != len(contributors) or len(result.decoded_frame_ids) != len(contributors):
        raise ValueError("S1.3 each contributor source must be formally remapped exactly once")
    height, width = result.valid_mask.shape
    source_index = result.assignment_index.copy()
    placement_codes = np.full((height, width), -1, dtype=np.int16)
    for index in range(len(schedule.assignments)):
        placement_codes[result.assignment_index == index] = index
    invalid_i32 = np.full((height, width), -1, dtype=np.int32)
    zeros = np.zeros((height, width), dtype=np.float32)
    pixel = {
        "owner_frame_id": result.owner_frame_id,
        "owner_source_index": source_index,
        "assignment_index": result.assignment_index,
        "source_u": result.source_u,
        "source_v": result.source_v,
        "valid": result.valid_mask,
        "selected_motion_hypothesis_id": invalid_i32.copy(),
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


__all__ = ["S13P0Result", "render_s13_p0"]
