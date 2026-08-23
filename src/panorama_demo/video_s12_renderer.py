"""Owner-only Stage-A renderer for standalone S012 dense central slits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .calibrated_remap import CalibratedIntrinsics, camera_matrix, undistortion_maps
from .cuda_backend import remap as accelerated_remap
from .video_s12_schedule import S012Schedule, validate_s012_schedule


@dataclass(frozen=True)
class S012StageAResult:
    """Stage-A pixels and mandatory pixel/column provenance."""

    image: np.ndarray
    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    source_u: np.ndarray
    source_v: np.ndarray
    assignment_index: np.ndarray
    column_frame_id: np.ndarray
    column_source_u: np.ndarray
    column_assignment_index: np.ndarray
    decoded_frame_ids: tuple[int, ...]
    remap_invocations: int
    duplicate_write_count: int


ImageLoader = Callable[[int], np.ndarray]
ResidentRemap = Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray]
ResidentDeviceRemap = Callable[[int, np.ndarray, np.ndarray, np.ndarray], Any]


def _as_loader(images: Mapping[int, np.ndarray] | ImageLoader) -> ImageLoader:
    if callable(images):
        return images

    def load(frame_id: int) -> np.ndarray:
        if frame_id not in images:
            raise ValueError(f"Missing S012 RGB source frame {frame_id}")
        return images[frame_id]

    return load


def _roi_inverse_map(
    calibration: CalibratedIntrinsics,
    center_x: float,
    left_x: int,
    right_x: int,
    inverse_maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw RGB coordinates for exactly one assigned H×w ROI."""

    height = int(calibration.height)
    canvas_x = np.arange(left_x, right_x, dtype=np.float64)
    rectified_x = float(calibration.cx) + canvas_x - float(center_x)
    rectified_y = np.arange(height, dtype=np.float64)
    grid_x = np.broadcast_to(rectified_x[None, :], (height, rectified_x.size))
    grid_y = np.broadcast_to(rectified_y[:, None], grid_x.shape)
    rectified_valid = (
        (grid_x >= 0.0)
        & (grid_x <= int(calibration.width) - 1)
        & (grid_y >= 0.0)
        & (grid_y <= int(calibration.height) - 1)
    )
    if inverse_maps is None:
        source_u = grid_x.astype(np.float32)
        source_v = grid_y.astype(np.float32)
    else:
        source_u = cv2.remap(
            inverse_maps[0],
            grid_x.astype(np.float32),
            grid_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
        source_v = cv2.remap(
            inverse_maps[1],
            grid_x.astype(np.float32),
            grid_y.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1,
        )
    valid = (
        rectified_valid
        & np.isfinite(source_u)
        & np.isfinite(source_v)
        & (source_u >= 0.0)
        & (source_u <= int(calibration.width) - 1)
        & (source_v >= 0.0)
        & (source_v <= int(calibration.height) - 1)
    )
    return source_u, source_v, valid


def render_s012_stage_a(
    schedule: S012Schedule,
    calibration: CalibratedIntrinsics,
    images: Mapping[int, np.ndarray] | ImageLoader,
    *,
    resident_remap: ResidentRemap | None = None,
    resident_device_remap: ResidentDeviceRemap | None = None,
    resident_stage: Any | None = None,
) -> S012StageAResult:
    """Render fixed owners without vertical warp, gain, blend, fill, or depth.

    A callable loader is recommended for production use: zero-width sources
    are then never decoded.  The renderer samples RGB once, directly into each
    assigned ROI, and a second write to any geometric output pixel is fatal.
    """

    camera_matrix(calibration)
    validate_s012_schedule(schedule)
    if schedule.canvas_height != int(calibration.height):
        raise ValueError("S012 schedule height does not match calibration")
    loader = _as_loader(images)
    inverse_maps = undistortion_maps(calibration)
    height, width = schedule.canvas_height, schedule.canvas_width
    output: np.ndarray | None = None
    output_device = (
        resident_stage.new_stage_canvas(height, width)
        if resident_stage is not None else None
    )
    if (resident_device_remap is None) != (resident_stage is None):
        raise ValueError("S012 resident device remap and stage must be supplied together")
    owner_frame_id = np.full((height, width), -1, dtype=np.int32)
    valid_mask = np.zeros((height, width), dtype=bool)
    source_u_full = np.full((height, width), np.nan, dtype=np.float32)
    source_v_full = np.full((height, width), np.nan, dtype=np.float32)
    assignment_full = np.full((height, width), -1, dtype=np.int32)
    geometric_written = np.zeros((height, width), dtype=bool)
    decoded: list[int] = []
    remap_invocations = 0
    duplicate_write_count = 0

    for assignment in schedule.assignments:
        if assignment.zero_width:
            continue
        roi = np.s_[:, assignment.left_x : assignment.right_x]
        if np.any(geometric_written[roi]):
            duplicate_write_count += int(np.count_nonzero(geometric_written[roi]))
            raise ValueError("S012 duplicate pixel write is structural fatal")
        geometric_written[roi] = True

        image = np.asarray(loader(assignment.frame_id))
        decoded.append(assignment.frame_id)
        if image.shape != (height, int(calibration.width), 3) or image.dtype != np.uint8:
            raise ValueError("S012 RGB source must be calibrated-size uint8 BGR")
        map_u, map_v, sample_valid = _roi_inverse_map(
            calibration,
            assignment.center_x,
            assignment.left_x,
            assignment.right_x,
            inverse_maps,
        )
        sampled: np.ndarray | None = None
        if output_device is None:
            sampled = (
                resident_remap(assignment.frame_id, image, map_u, map_v)
                if resident_remap is not None
                else accelerated_remap(
                    image, map_u, map_v, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            )
        remap_invocations += 1
        if output_device is not None:
            resident_stage.compose_owner_roi(
                output_device, assignment.left_x, assignment.right_x,
                resident_device_remap(assignment.frame_id, image, map_u, map_v), sample_valid,
            )
        if output is None:
            output = np.zeros((height, width, 3), dtype=np.uint8)
        if output_device is None:
            assert sampled is not None
            output[roi][sample_valid] = sampled[sample_valid]
        owner_roi = owner_frame_id[roi]
        owner_roi[sample_valid] = assignment.frame_id
        valid_mask[roi] = sample_valid
        source_u_roi = source_u_full[roi]
        source_v_roi = source_v_full[roi]
        assignment_roi = assignment_full[roi]
        source_u_roi[sample_valid] = map_u[sample_valid]
        source_v_roi[sample_valid] = map_v[sample_valid]
        assignment_roi[sample_valid] = assignment.assignment_index

    if output_device is not None:
        output = resident_stage.download_stage_canvas("P0", output_device)
    if output is None:
        output = np.zeros((height, width, 3), dtype=np.uint8)
    if not np.all(geometric_written):
        raise ValueError("S012 render did not visit every scheduled canvas pixel")
    if duplicate_write_count != 0:
        raise ValueError("S012 duplicate pixel write is structural fatal")
    if not np.array_equal(valid_mask, owner_frame_id >= 0):
        raise ValueError("S012 valid mask and pixel owner provenance disagree")
    if np.any(owner_frame_id[~valid_mask] != -1) or np.any(assignment_full[~valid_mask] != -1):
        raise ValueError("S012 invalid pixels must not have an owner")

    return S012StageAResult(
        image=np.ascontiguousarray(output),
        owner_frame_id=owner_frame_id,
        valid_mask=valid_mask,
        source_u=source_u_full,
        source_v=source_v_full,
        assignment_index=assignment_full,
        column_frame_id=schedule.column_frame_id.copy(),
        column_source_u=schedule.column_source_u.copy(),
        column_assignment_index=schedule.column_assignment_index.copy(),
        decoded_frame_ids=tuple(decoded),
        remap_invocations=remap_invocations,
        duplicate_write_count=duplicate_write_count,
    )
