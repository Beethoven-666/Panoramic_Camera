"""Independent RGB-D world-coverage audit for the inspection mosaic.

The inspection image is a two-dimensional view, so owner topology alone
cannot prove that every near surface observed by the moving camera survived
the panel hand-offs.  This module compares two measured world-space sets:

* near RGB-D voxels observed by every real pose; and
* near RGB-D voxels reached by the final inspection RGB owner pixels.

It never supplies colour, fills a hole, changes a pose, or feeds the metric or
TSDF products.  Its only output is compact scalar/location evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .cuda_backend import remap as accelerated_remap
from .cuda_backend import pinhole_unproject, transform_points
from .session import CameraIntrinsics, RGBDFrame, read_aligned_depth_mm


@dataclass(frozen=True)
class InspectionWorldCoverageConfig:
    sample_stride: int = 4
    voxel_size_mm: float = 8.0
    match_radius_voxels: int = 1
    minimum_depth_mm: float = 200.0
    maximum_depth_mm: float = 3000.0
    near_margin_mm: float = 35.0
    near_margin_ratio: float = 0.04
    minimum_multiview_source_support: int = 2
    cell_size_mm: float = 80.0
    minimum_cell_voxels: int = 24

    def validate(self) -> None:
        if not 1 <= int(self.sample_stride) <= 16:
            raise ValueError("Inspection coverage sample_stride must be in [1, 16]")
        if not 2.0 <= float(self.voxel_size_mm) <= 32.0:
            raise ValueError("Inspection coverage voxel_size_mm must be in [2, 32]")
        if not 0 <= int(self.match_radius_voxels) <= 2:
            raise ValueError(
                "Inspection coverage match_radius_voxels must be in [0, 2]"
            )
        if (
            not math.isfinite(float(self.minimum_depth_mm))
            or not math.isfinite(float(self.maximum_depth_mm))
            or float(self.minimum_depth_mm) <= 0.0
            or float(self.maximum_depth_mm) <= float(self.minimum_depth_mm)
        ):
            raise ValueError("Inspection coverage depth range is invalid")
        if (
            not math.isfinite(float(self.near_margin_mm))
            or float(self.near_margin_mm) <= 0.0
            or not 0.0 < float(self.near_margin_ratio) < 1.0
        ):
            raise ValueError("Inspection coverage near-depth margin is invalid")
        if not 1 <= int(self.minimum_multiview_source_support) <= 16:
            raise ValueError(
                "Inspection coverage source-support threshold is invalid"
            )
        if (
            not math.isfinite(float(self.cell_size_mm))
            or float(self.cell_size_mm) <= 0.0
            or int(self.minimum_cell_voxels) < 1
        ):
            raise ValueError("Inspection coverage cell configuration is invalid")


def _undistortion_maps(
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray] | None:
    distortion = np.asarray(intrinsics.distortion, dtype=np.float64)
    if distortion.size == 0 or not np.any(distortion):
        return None
    return cv2.initUndistortRectifyMap(
        intrinsics.matrix,
        distortion,
        None,
        intrinsics.matrix,
        (intrinsics.width, intrinsics.height),
        cv2.CV_32FC1,
    )


def _read_depth(
    frame: RGBDFrame,
    intrinsics: CameraIntrinsics,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    depth = read_aligned_depth_mm(frame).astype(np.float32, copy=False)
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("Inspection coverage depth dimensions do not match RGB")
    if maps is None:
        return np.ascontiguousarray(depth)
    return accelerated_remap(
        depth,
        maps[0],
        maps[1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _near_solver_valid(
    depth_mm: np.ndarray,
    *,
    reference_depth_mm: float,
    config: InspectionWorldCoverageConfig,
) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    reliable = (
        np.isfinite(depth)
        & (depth >= np.float32(config.minimum_depth_mm))
        & (depth <= np.float32(config.maximum_depth_mm))
    )
    margin = max(
        float(config.near_margin_mm),
        float(config.near_margin_ratio) * float(reference_depth_mm),
    )
    near = reliable & (
        depth < np.float32(float(reference_depth_mm) - margin)
    )
    sentinel = np.float32(float(config.maximum_depth_mm) + 1000.0)
    local_max = cv2.dilate(
        np.where(reliable, depth, 0.0),
        np.ones((3, 3), dtype=np.uint8),
    )
    local_min = cv2.erode(
        np.where(reliable, depth, sentinel),
        np.ones((3, 3), dtype=np.uint8),
    )
    tolerance = np.maximum(np.float32(20.0), np.float32(0.02) * depth)
    edge = reliable & ((local_max - local_min) > tolerance)
    protected = cv2.dilate(
        edge.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    return near & ~protected


def _world_points_from_pixels(
    x: np.ndarray,
    y: np.ndarray,
    depth_mm: np.ndarray,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    camera = pinhole_unproject(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(depth_mm, dtype=np.float64),
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    matrix = np.asarray(pose, dtype=np.float64)
    return transform_points(camera, matrix[:3, :3], matrix[:3, 3])


def _voxel_keys(points_world_mm: np.ndarray, voxel_size_mm: float) -> np.ndarray:
    points = np.asarray(points_world_mm, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Inspection coverage world points must be finite Nx3")
    return np.floor(points / float(voxel_size_mm)).astype(np.int32)


def _linear_codes(
    keys: np.ndarray,
    *,
    minimum: np.ndarray,
    dimensions: np.ndarray,
) -> np.ndarray:
    shifted = np.asarray(keys, dtype=np.int64) - minimum[None, :]
    return (
        (shifted[:, 0] * dimensions[1] + shifted[:, 1])
        * dimensions[2]
        + shifted[:, 2]
    )


def _match_voxels(
    observed: np.ndarray,
    represented: np.ndarray,
    radius: int,
) -> np.ndarray:
    if observed.size == 0:
        return np.zeros(0, dtype=bool)
    if represented.size == 0:
        return np.zeros(len(observed), dtype=bool)
    combined_minimum = np.minimum(
        np.min(observed, axis=0), np.min(represented, axis=0)
    ).astype(np.int64)
    combined_maximum = np.maximum(
        np.max(observed, axis=0), np.max(represented, axis=0)
    ).astype(np.int64)
    padding = int(radius) + 1
    minimum = combined_minimum - padding
    dimensions = combined_maximum - minimum + padding + 1
    if np.any(dimensions <= 0):
        raise RuntimeError("Inspection coverage voxel-code dimensions are invalid")
    represented_codes = np.unique(
        _linear_codes(represented, minimum=minimum, dimensions=dimensions)
    )
    observed_codes = _linear_codes(
        observed, minimum=minimum, dimensions=dimensions
    )
    matched = np.zeros(len(observed), dtype=bool)
    stride_x = int(dimensions[1] * dimensions[2])
    stride_y = int(dimensions[2])
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                offset = dx * stride_x + dy * stride_y + dz
                matched |= np.isin(
                    observed_codes + offset,
                    represented_codes,
                    assume_unique=False,
                )
    return matched


def _panel_source_records(
    selected_panel_sources: Sequence[Mapping[str, object]],
    *,
    frame_id_to_position: Mapping[int, int],
    panel_count: int,
) -> list[tuple[int, int, int]]:
    records: list[tuple[int, int, int]] = []
    used_panels: set[int] = set()
    used_frames: set[int] = set()
    for item in selected_panel_sources:
        panel_index = int(item["panel_index"])
        frame_id = int(item["frame_id"])
        source_position = int(item["source_position"])
        if (
            not 0 <= panel_index < panel_count
            or frame_id not in frame_id_to_position
            or frame_id_to_position[frame_id] != source_position
            or panel_index in used_panels
            or frame_id in used_frames
        ):
            raise ValueError("Inspection coverage panel-source records are invalid")
        used_panels.add(panel_index)
        used_frames.add(frame_id)
        records.append((panel_index, source_position, frame_id))
    if len(records) != panel_count:
        raise ValueError("Inspection coverage lacks one real source per panel")
    return sorted(records)


def audit_inspection_world_coverage(
    *,
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    layout: object,
    owner_frame_id: np.ndarray,
    crop_xywh: Sequence[int],
    selected_panel_sources: Sequence[Mapping[str, object]],
    config: InspectionWorldCoverageConfig | None = None,
) -> dict[str, object]:
    """Compare observed near RGB-D voxels with final RGB-owner reachability."""

    selected = config or InspectionWorldCoverageConfig()
    selected.validate()
    if len(frames) < 2 or len(frames) != len(poses):
        raise ValueError("Inspection coverage needs aligned frames and poses")
    owner = np.asarray(owner_frame_id, dtype=np.int32)
    if owner.ndim != 2:
        raise ValueError("Inspection coverage owner raster must be two-dimensional")
    if len(crop_xywh) != 4:
        raise ValueError("Inspection coverage crop must contain x, y, width, height")
    crop_x, crop_y, crop_width, crop_height = (
        int(value) for value in crop_xywh
    )
    if (
        crop_x < 0
        or crop_y < 0
        or crop_width != owner.shape[1]
        or crop_height != owner.shape[0]
        or crop_x + crop_width > int(layout.width)
        or crop_y + crop_height > int(layout.height)
    ):
        raise ValueError("Inspection coverage crop is inconsistent with its owner")
    checked_poses: list[np.ndarray] = []
    for pose in poses:
        matrix = np.asarray(pose, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Inspection coverage poses must be finite 4x4 matrices")
        checked_poses.append(matrix)
    frame_id_to_position = {
        int(frame.frame_id): index for index, frame in enumerate(frames)
    }
    if len(frame_id_to_position) != len(frames):
        raise ValueError("Inspection coverage frame IDs must be unique")
    records = _panel_source_records(
        selected_panel_sources,
        frame_id_to_position=frame_id_to_position,
        panel_count=len(layout.panels),
    )
    maps = _undistortion_maps(intrinsics)
    observed_key_parts: list[np.ndarray] = []
    observed_source_parts: list[np.ndarray] = []
    per_source_near_samples: list[int] = []
    cached_selected_depth: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    stride = int(selected.sample_stride)
    reference_depth = float(layout.reference_depth_mm)
    for source_position, (frame, pose) in enumerate(
        zip(frames, checked_poses, strict=True)
    ):
        depth = _read_depth(frame, intrinsics, maps)
        near = _near_solver_valid(
            depth,
            reference_depth_mm=reference_depth,
            config=selected,
        )
        sample_mask = np.zeros_like(near)
        sample_mask[::stride, ::stride] = near[::stride, ::stride]
        yy, xx = np.nonzero(sample_mask)
        per_source_near_samples.append(int(xx.size))
        if xx.size:
            world = _world_points_from_pixels(
                xx,
                yy,
                depth[yy, xx],
                pose,
                intrinsics,
            )
            keys = _voxel_keys(world, selected.voxel_size_mm)
            observed_key_parts.append(keys)
            observed_source_parts.append(
                np.full(len(keys), source_position, dtype=np.int32)
            )
        if int(frame.frame_id) in {item[2] for item in records}:
            cached_selected_depth[source_position] = (depth, near)
    if observed_key_parts:
        observed_raw_keys = np.concatenate(observed_key_parts)
        observed_raw_sources = np.concatenate(observed_source_parts)
        source_keys = np.column_stack(
            (observed_raw_sources, observed_raw_keys)
        )
        unique_source_keys = np.unique(source_keys, axis=0)
        observed_keys, source_support = np.unique(
            unique_source_keys[:, 1:],
            axis=0,
            return_counts=True,
        )
    else:
        observed_raw_keys = np.empty((0, 3), dtype=np.int32)
        observed_keys = np.empty((0, 3), dtype=np.int32)
        source_support = np.empty(0, dtype=np.int64)

    represented_parts: list[np.ndarray] = []
    represented_source_samples: list[dict[str, object]] = []
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    down_axis = np.asarray(layout.down_axis, dtype=np.float64)
    normal_axis = np.asarray(layout.normal_axis, dtype=np.float64)
    for panel_index, source_position, frame_id in records:
        panel = layout.panels[panel_index]
        x0 = int(round(float(panel.canvas_offset_x)))
        owned_y, owned_x = np.nonzero(owner == frame_id)
        full_x = owned_x + crop_x
        full_y = owned_y + crop_y
        local_x = full_x - x0
        inside = (
            (local_x >= 0)
            & (local_x < intrinsics.width)
            & (full_y >= 0)
            & (full_y < intrinsics.height)
        )
        local_x = local_x[inside].astype(np.float64)
        panel_y = full_y[inside].astype(np.float64)
        if local_x.size == 0:
            represented_source_samples.append(
                {
                    "panel_index": panel_index,
                    "frame_id": frame_id,
                    "owner_pixel_count": int(owned_x.size),
                    "near_owner_sample_count": 0,
                }
            )
            continue
        q_scan = (
            (local_x - intrinsics.cx)
            * reference_depth
            / intrinsics.fx
        )
        q_down = (
            (panel_y - intrinsics.cy)
            * reference_depth
            / intrinsics.fy
        )
        center = np.asarray(panel.center_world_mm, dtype=np.float64)
        reference_world = (
            center[None, :]
            + q_scan[:, None] * scan_axis[None, :]
            + q_down[:, None] * down_axis[None, :]
            + reference_depth * normal_axis[None, :]
        )
        pose = checked_poses[source_position]
        camera = (reference_world - pose[:3, 3]) @ pose[:3, :3]
        with np.errstate(divide="ignore", invalid="ignore"):
            source_x = intrinsics.fx * camera[:, 0] / camera[:, 2] + intrinsics.cx
            source_y = intrinsics.fy * camera[:, 1] / camera[:, 2] + intrinsics.cy
        map_valid = (
            np.isfinite(source_x)
            & np.isfinite(source_y)
            & (camera[:, 2] > 0.0)
            & (source_x >= 0.0)
            & (source_x <= intrinsics.width - 1)
            & (source_y >= 0.0)
            & (source_y <= intrinsics.height - 1)
        )
        source_x_i = np.clip(
            np.rint(source_x[map_valid]).astype(np.intp),
            0,
            intrinsics.width - 1,
        )
        source_y_i = np.clip(
            np.rint(source_y[map_valid]).astype(np.intp),
            0,
            intrinsics.height - 1,
        )
        depth, near = cached_selected_depth[source_position]
        is_near = near[source_y_i, source_x_i]
        source_x_i = source_x_i[is_near]
        source_y_i = source_y_i[is_near]
        if source_x_i.size:
            world = _world_points_from_pixels(
                source_x_i,
                source_y_i,
                depth[source_y_i, source_x_i],
                pose,
                intrinsics,
            )
            represented_parts.append(
                _voxel_keys(world, selected.voxel_size_mm)
            )
        represented_source_samples.append(
            {
                "panel_index": panel_index,
                "frame_id": frame_id,
                "owner_pixel_count": int(owned_x.size),
                "near_owner_sample_count": int(source_x_i.size),
            }
        )
    represented_keys = (
        np.unique(np.concatenate(represented_parts), axis=0)
        if represented_parts
        else np.empty((0, 3), dtype=np.int32)
    )
    matched = _match_voxels(
        observed_keys,
        represented_keys,
        int(selected.match_radius_voxels),
    )
    multiview = (
        source_support >= int(selected.minimum_multiview_source_support)
    )
    multiview_count = int(np.count_nonzero(multiview))
    multiview_matched = int(np.count_nonzero(matched & multiview))

    voxel_centers = (
        observed_keys.astype(np.float64) + 0.5
    ) * float(selected.voxel_size_mm)
    basis = np.column_stack(
        (
            voxel_centers @ scan_axis,
            voxel_centers @ down_axis,
            voxel_centers @ normal_axis,
        )
    )
    cell_keys = np.floor(
        basis / float(selected.cell_size_mm)
    ).astype(np.int32)
    cell_values, cell_inverse = np.unique(
        cell_keys, axis=0, return_inverse=True
    )
    panel_anchors = np.asarray(
        [float(panel.anchor_scan_mm) for panel in layout.panels],
        dtype=np.float64,
    )
    panel_centers = np.asarray(
        [panel.center_world_mm for panel in layout.panels],
        dtype=np.float64,
    )
    panel_offsets = np.asarray(
        [float(panel.canvas_offset_x) for panel in layout.panels],
        dtype=np.float64,
    )
    low_coverage_cells: list[dict[str, object]] = []
    for cell_index, cell_key in enumerate(cell_values):
        selected_cell = (cell_inverse == cell_index) & multiview
        count = int(np.count_nonzero(selected_cell))
        if count < int(selected.minimum_cell_voxels):
            continue
        matched_count = int(np.count_nonzero(matched & selected_cell))
        ratio = matched_count / count
        if ratio >= 0.90:
            continue
        cell_points = basis[selected_cell]
        cell_world = voxel_centers[selected_cell]
        insertion = np.searchsorted(panel_anchors, cell_points[:, 0])
        right = np.clip(insertion, 0, len(panel_anchors) - 1)
        left = np.clip(insertion - 1, 0, len(panel_anchors) - 1)
        choose_right = np.abs(
            panel_anchors[right] - cell_points[:, 0]
        ) < np.abs(panel_anchors[left] - cell_points[:, 0])
        panel_index = np.where(choose_right, right, left)
        relative = cell_world - panel_centers[panel_index]
        q_scan = relative @ scan_axis
        q_down = relative @ down_axis
        q_normal = relative @ normal_axis
        with np.errstate(divide="ignore", invalid="ignore"):
            canvas_x = (
                panel_offsets[panel_index]
                + intrinsics.cx
                + intrinsics.fx * q_scan / q_normal
            )
            canvas_y = intrinsics.cy + intrinsics.fy * q_down / q_normal
        finite_canvas = (
            np.isfinite(canvas_x)
            & np.isfinite(canvas_y)
            & (q_normal > 0.0)
        )
        projected_bbox: list[int] | None = None
        projected_center: list[float] | None = None
        if np.any(finite_canvas):
            x0 = int(math.floor(float(np.min(canvas_x[finite_canvas]))))
            y0 = int(math.floor(float(np.min(canvas_y[finite_canvas]))))
            x1 = int(math.ceil(float(np.max(canvas_x[finite_canvas])))) + 1
            y1 = int(math.ceil(float(np.max(canvas_y[finite_canvas])))) + 1
            projected_bbox = [x0, y0, x1 - x0, y1 - y0]
            projected_center = [
                float(np.median(canvas_x[finite_canvas])),
                float(np.median(canvas_y[finite_canvas])),
            ]
        low_coverage_cells.append(
            {
                "cell_scan_down_normal": [
                    int(value) for value in cell_key
                ],
                "world_voxel_count": count,
                "matched_world_voxel_count": matched_count,
                "coverage_ratio": float(ratio),
                "basis_bounds_mm": [
                    [float(value) for value in np.min(cell_points, axis=0)],
                    [float(value) for value in np.max(cell_points, axis=0)],
                ],
                "full_canvas_center_xy": projected_center,
                "full_canvas_bbox_xywh": projected_bbox,
            }
        )
    low_coverage_cells.sort(
        key=lambda item: (
            float(item["coverage_ratio"]),
            -int(item["world_voxel_count"]),
        )
    )
    all_ratio = float(np.mean(matched)) if matched.size else 1.0
    multiview_ratio = (
        float(multiview_matched / multiview_count)
        if multiview_count
        else 1.0
    )
    return {
        "schema": "gemini305-inspection-world-coverage/v1",
        "role": (
            "independent_read_only_rgbd_world_surface_coverage_audit"
        ),
        "colour_or_geometry_mutation": False,
        "pose_interpolation_or_modification": False,
        "tsdf_or_metric_feedback": False,
        "voxel_size_mm": float(selected.voxel_size_mm),
        "match_radius_voxels": int(selected.match_radius_voxels),
        "sample_stride": stride,
        "near_depth_margin_mm": max(
            float(selected.near_margin_mm),
            float(selected.near_margin_ratio) * reference_depth,
        ),
        "observed_raw_sample_count": int(len(observed_raw_keys)),
        "observed_world_voxel_count": int(len(observed_keys)),
        "multiview_observed_world_voxel_count": multiview_count,
        "represented_world_voxel_count": int(len(represented_keys)),
        "matched_observed_world_voxel_count": int(np.count_nonzero(matched)),
        "matched_multiview_world_voxel_count": multiview_matched,
        "observed_world_coverage_ratio": all_ratio,
        "multiview_world_coverage_ratio": multiview_ratio,
        "per_source_near_sample_count": per_source_near_samples,
        "represented_panel_sources": represented_source_samples,
        "low_coverage_cell_count": len(low_coverage_cells),
        "low_coverage_cells": low_coverage_cells[:64],
    }


__all__ = [
    "InspectionWorldCoverageConfig",
    "audit_inspection_world_coverage",
]
