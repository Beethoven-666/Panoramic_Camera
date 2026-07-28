"""Experimental all-view RGB-D world-surface inspection raster.

This module is deliberately isolated from the formal V1 publication path.
It evaluates whether using every tracked RGB-D view, rather than one RGB
source per virtual panel, can recover complete near-field objects.

Only RGB values selected by nearest-neighbour sampling from a real source
frame are written.  Continuous source cells are accepted only after the
existing depth-boundary, panel-locality and positive-Jacobian mesh gates.
Overlapping observations are resolved by target-view z-buffering followed by
source-image centrality on the same physical layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .cuda_backend import (
    pinhole_unproject,
    remap as accelerated_remap,
    transform_points,
)
from .inspection_multiview import (
    InspectionMultiviewLayout,
    InspectionMultiviewConfig,
    _build_depth_mesh_panel_remap,
    _depth_confidence,
    _foreground_depth_layer_components,
    _read_rgbd,
    _undistortion_maps,
    estimate_inspection_layout,
)
from .session import CameraIntrinsics, RGBDFrame


@dataclass(frozen=True)
class WorldSurfaceInspectionConfig:
    """Closed prototype settings; none are formal publication controls."""

    minimum_depth_mm: float = 200.0
    maximum_depth_mm: float = 3000.0
    panel_overlap: float = 0.95
    depth_mesh_cell_size_pixels: int = 8
    minimum_jacobian: float = 0.01
    maximum_jacobian: float = 64.0
    temporal_absolute_tolerance_mm: float = 20.0
    temporal_relative_tolerance: float = 0.02
    maximum_canvas_megapixels: float = 200.0
    maximum_working_bytes: int = 4_000_000_000
    component_lock_enabled: bool = True
    component_near_margin_mm: float = 60.0
    component_minimum_pixels: int = 300
    component_maximum_pixels: int = 80_000
    component_maximum_width_fraction: float = 0.35
    component_maximum_height_fraction: float = 0.60
    component_minimum_single_owner_coverage: float = 0.90

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None = None
    ) -> "WorldSurfaceInspectionConfig":
        payload = {} if value is None else dict(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"Unknown world-surface prototype configuration keys: {unknown}"
            )
        try:
            selected = cls(**payload)
        except TypeError as exc:
            raise ValueError(
                "Invalid world-surface prototype configuration"
            ) from exc
        selected.validate()
        return selected

    def validate(self) -> None:
        for name, value in (
            ("minimum_depth_mm", self.minimum_depth_mm),
            ("maximum_depth_mm", self.maximum_depth_mm),
            (
                "temporal_absolute_tolerance_mm",
                self.temporal_absolute_tolerance_mm,
            ),
            ("temporal_relative_tolerance", self.temporal_relative_tolerance),
            ("minimum_jacobian", self.minimum_jacobian),
            ("maximum_jacobian", self.maximum_jacobian),
            ("maximum_canvas_megapixels", self.maximum_canvas_megapixels),
            ("component_near_margin_mm", self.component_near_margin_mm),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_depth_mm <= self.minimum_depth_mm:
            raise ValueError("World-surface prototype depth range is empty")
        if not 0.0 < float(self.panel_overlap) < 1.0:
            raise ValueError("panel_overlap must be in (0, 1)")
        if not 2 <= int(self.depth_mesh_cell_size_pixels) <= 32:
            raise ValueError("depth_mesh_cell_size_pixels must be in [2, 32]")
        if self.maximum_jacobian <= self.minimum_jacobian:
            raise ValueError("World-surface Jacobian range is empty")
        if (
            type(self.maximum_working_bytes) is not int
            or self.maximum_working_bytes <= 0
        ):
            raise ValueError("maximum_working_bytes must be a positive integer")
        if type(self.component_lock_enabled) is not bool:
            raise ValueError("component_lock_enabled must be boolean")
        if (
            type(self.component_minimum_pixels) is not int
            or type(self.component_maximum_pixels) is not int
            or self.component_minimum_pixels < 1
            or self.component_maximum_pixels < self.component_minimum_pixels
        ):
            raise ValueError("World-surface component pixel bounds are invalid")
        for name, value in (
            (
                "component_maximum_width_fraction",
                self.component_maximum_width_fraction,
            ),
            (
                "component_maximum_height_fraction",
                self.component_maximum_height_fraction,
            ),
            (
                "component_minimum_single_owner_coverage",
                self.component_minimum_single_owner_coverage,
            ),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class WorldSurfaceInspectionResult:
    image_bgr: np.ndarray
    owner_frame_id: np.ndarray
    target_depth_mm: np.ndarray
    valid_mask: np.ndarray
    component_locked_image_bgr: np.ndarray
    component_locked_owner_frame_id: np.ndarray
    component_locked_valid_mask: np.ndarray
    component_label: np.ndarray
    metadata: dict[str, object]

    def validate(self) -> None:
        shape = self.image_bgr.shape[:2]
        if self.image_bgr.dtype != np.uint8 or self.image_bgr.shape != (*shape, 3):
            raise RuntimeError("World-surface RGB output must be HxWx3 uint8")
        if (
            self.owner_frame_id.shape != shape
            or self.owner_frame_id.dtype != np.int32
            or self.target_depth_mm.shape != shape
            or self.target_depth_mm.dtype != np.float32
            or self.valid_mask.shape != shape
            or self.component_locked_image_bgr.shape != self.image_bgr.shape
            or self.component_locked_image_bgr.dtype != np.uint8
            or self.component_locked_owner_frame_id.shape != shape
            or self.component_locked_owner_frame_id.dtype != np.int32
            or self.component_locked_valid_mask.shape != shape
            or self.component_label.shape != shape
            or self.component_label.dtype != np.int32
        ):
            raise RuntimeError("World-surface output rasters are misaligned")
        valid = np.asarray(self.valid_mask, dtype=bool)
        if not np.any(valid):
            raise RuntimeError("World-surface prototype produced no valid surface")
        if np.any(self.owner_frame_id[valid] < 0) or np.any(
            self.owner_frame_id[~valid] != -1
        ):
            raise RuntimeError("World-surface owner contract failed")
        if np.any(~np.isfinite(self.target_depth_mm[valid])) or np.any(
            np.isfinite(self.target_depth_mm[~valid])
        ):
            raise RuntimeError("World-surface target-depth contract failed")
        locked_valid = np.asarray(self.component_locked_valid_mask, dtype=bool)
        if np.any(self.component_locked_owner_frame_id[locked_valid] < 0) or np.any(
            self.component_locked_owner_frame_id[~locked_valid] != -1
        ):
            raise RuntimeError("Component-locked owner contract failed")


def _inspection_config(
    config: WorldSurfaceInspectionConfig,
) -> InspectionMultiviewConfig:
    return InspectionMultiviewConfig(
        minimum_depth_mm=config.minimum_depth_mm,
        maximum_depth_mm=config.maximum_depth_mm,
        background_panel_overlap=config.panel_overlap,
        depth_mesh_cell_size_pixels=config.depth_mesh_cell_size_pixels,
        depth_mesh_min_jacobian=config.minimum_jacobian,
        depth_mesh_max_jacobian=config.maximum_jacobian,
        maximum_canvas_megapixels=config.maximum_canvas_megapixels,
        maximum_working_bytes=config.maximum_working_bytes,
        temporal_absolute_tolerance_mm=(
            config.temporal_absolute_tolerance_mm
        ),
        temporal_relative_tolerance=config.temporal_relative_tolerance,
    )


def _nearest_panel_for_camera(
    pose: np.ndarray,
    panel_anchors: np.ndarray,
    scan_axis: np.ndarray,
) -> int:
    scan_coordinate = float(np.asarray(pose, dtype=np.float64)[:3, 3] @ scan_axis)
    return int(np.argmin(np.abs(panel_anchors - scan_coordinate)))


@dataclass(frozen=True)
class _ComponentSourceCandidate:
    frame_id: int
    corner_x: int
    image_bgr: np.ndarray
    target_depth_mm: np.ndarray
    valid_mask: np.ndarray
    audit: dict[str, object]


def _render_component_source_candidate(
    *,
    frame: RGBDFrame,
    pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    layout: InspectionMultiviewLayout,
    mesh_config: InspectionMultiviewConfig,
    maps: tuple[np.ndarray, np.ndarray] | None,
    panel_anchors: np.ndarray,
    scan_axis: np.ndarray,
    selected: WorldSurfaceInspectionConfig,
) -> _ComponentSourceCandidate:
    image, depth, geometric_valid = _read_rgbd(frame, intrinsics, maps)
    reliable = (
        geometric_valid
        & np.isfinite(depth)
        & (depth >= selected.minimum_depth_mm)
        & (depth <= selected.maximum_depth_mm)
    )
    _, depth_edge = _depth_confidence(depth, reliable, mesh_config)
    neighbourhood = cv2.boxFilter(
        reliable.astype(np.uint8),
        cv2.CV_16U,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    solver_valid = reliable & ~depth_edge & (neighbourhood == 9)
    panel_index = _nearest_panel_for_camera(
        np.asarray(pose, dtype=np.float64), panel_anchors, scan_axis
    )
    mesh = _build_depth_mesh_panel_remap(
        source_depth_mm=depth,
        source_solver_valid=solver_valid,
        source_pose=np.asarray(pose, dtype=np.float64),
        panel_index=panel_index,
        layout=layout,
        intrinsics=intrinsics,
        config=mesh_config,
    )
    safe_x = np.where(mesh.valid_mask, mesh.map_x, -1.0).astype(
        np.float32, copy=False
    )
    safe_y = np.where(mesh.valid_mask, mesh.map_y, -1.0).astype(
        np.float32, copy=False
    )
    sampled_image = accelerated_remap(
        image,
        safe_x,
        safe_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_valid = accelerated_remap(
        solver_valid.astype(np.uint8) * 255,
        safe_x,
        safe_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (
        mesh.valid_mask
        & (sampled_valid > 0)
        & np.isfinite(mesh.relative_depth_mm)
        & (mesh.relative_depth_mm > 0.0)
    )
    return _ComponentSourceCandidate(
        frame_id=int(frame.frame_id),
        corner_x=int(mesh.corner_x),
        image_bgr=np.ascontiguousarray(sampled_image),
        target_depth_mm=np.ascontiguousarray(mesh.relative_depth_mm),
        valid_mask=np.ascontiguousarray(valid),
        audit={
            "panel_index": panel_index,
            "complete_3x3_solver_pixel_count": int(
                np.count_nonzero(solver_valid)
            ),
            "mesh": dict(mesh.audit),
            "nearest_real_rgb_only": True,
        },
    )


def _component_lock_world_surface(
    *,
    image_bgr: np.ndarray,
    owner_frame_id: np.ndarray,
    target_depth_mm: np.ndarray,
    valid_mask: np.ndarray,
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    layout: InspectionMultiviewLayout,
    mesh_config: InspectionMultiviewConfig,
    maps: tuple[np.ndarray, np.ndarray] | None,
    selected: WorldSurfaceInspectionConfig,
    vertical_crop_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Lock only demonstrably compact near components to one reprojected frame."""

    image = np.asarray(image_bgr, dtype=np.uint8)
    owner = np.asarray(owner_frame_id, dtype=np.int32)
    depth = np.asarray(target_depth_mm, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    locked_image = image.copy()
    locked_owner = owner.copy()
    locked_valid = valid.copy()
    labels = np.zeros(valid.shape, dtype=np.int32)
    if not selected.component_lock_enabled:
        return (
            locked_image,
            locked_owner,
            locked_valid,
            labels,
            {
                "enabled": False,
                "reason": "prototype_component_lock_disabled",
                "components": [],
            },
        )

    reference = float(layout.reference_depth_mm)
    near = (
        valid
        & np.isfinite(depth)
        & (
            depth
            < np.float32(reference - selected.component_near_margin_mm)
        )
    )
    sentinel = np.float32(selected.maximum_depth_mm * 2.0)
    local_max = cv2.dilate(
        np.where(valid, depth, 0.0), np.ones((3, 3), np.uint8)
    )
    local_min = cv2.erode(
        np.where(valid, depth, sentinel), np.ones((3, 3), np.uint8)
    )
    tolerance = np.maximum(
        np.float32(selected.temporal_absolute_tolerance_mm),
        np.float32(selected.temporal_relative_tolerance)
        * np.maximum(depth, 0.0),
    )
    depth_edge = valid & ((local_max - local_min) > tolerance)
    segmentation_mask = (
        near
        & ~cv2.dilate(
            depth_edge.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
    )
    # This close affects labels only.  It never makes a pixel eligible for
    # RGB output and therefore cannot fill colour or depth holes.
    segmentation_mask = cv2.morphologyEx(
        segmentation_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    )
    count, raw_labels, stats, _ = cv2.connectedComponentsWithStats(
        segmentation_mask, 8
    )
    frame_by_id = {int(frame.frame_id): frame for frame in frames}
    pose_by_id = {
        int(frame.frame_id): np.asarray(pose, dtype=np.float64)
        for frame, pose in zip(frames, poses, strict=True)
    }
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    panel_anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels], dtype=np.float64
    )
    component_records: list[dict[str, object]] = []
    selected_owner_ids: set[int] = set()
    provisional: list[dict[str, object]] = []
    next_label = 1
    for raw_label in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[raw_label].tolist()
        )
        if area < selected.component_minimum_pixels:
            continue
        component = raw_labels == raw_label
        component &= near
        measured_area = int(np.count_nonzero(component))
        if measured_area < selected.component_minimum_pixels:
            continue
        labels[component] = next_label
        owner_values, owner_counts = np.unique(
            owner[component], return_counts=True
        )
        owner_filter = owner_values >= 0
        owner_values = owner_values[owner_filter]
        owner_counts = owner_counts[owner_filter]
        order = np.argsort(-owner_counts, kind="stable")
        owner_values = owner_values[order]
        owner_counts = owner_counts[order]
        before_owner_count = int(owner_values.size)
        compact = (
            measured_area <= selected.component_maximum_pixels
            and width
            <= int(
                math.ceil(
                    selected.component_maximum_width_fraction * valid.shape[1]
                )
            )
            and height
            <= int(
                math.ceil(
                    selected.component_maximum_height_fraction * valid.shape[0]
                )
            )
        )
        record: dict[str, object] = {
            "component_id": next_label,
            "raw_connected_component_label": int(raw_label),
            "bbox_xywh": [x, y, width, height],
            "measured_pixel_count": measured_area,
            "owner_count_before": before_owner_count,
            "owner_vote_top": [
                {
                    "frame_id": int(frame_id),
                    "pixel_count": int(pixel_count),
                }
                for frame_id, pixel_count in zip(
                    owner_values[:8], owner_counts[:8], strict=False
                )
            ],
            "compact_candidate": bool(compact),
            "selected_frame_id": (
                None if not owner_values.size else int(owner_values[0])
            ),
            "accepted": False,
        }
        if not compact:
            record["rejection_reason"] = (
                "near_world_component_is_structural_or_merges_multiple_objects"
            )
            component_records.append(record)
            next_label += 1
            continue
        if not owner_values.size or int(owner_values[0]) not in frame_by_id:
            record["rejection_reason"] = "component_has_no_real_rgb_owner_vote"
            component_records.append(record)
            next_label += 1
            continue
        selected_frame_id = int(owner_values[0])
        selected_owner_ids.add(selected_frame_id)
        provisional.append(
            {
                "record": record,
                "component": component,
                "selected_frame_id": selected_frame_id,
            }
        )
        component_records.append(record)
        next_label += 1

    candidate_by_frame: dict[int, _ComponentSourceCandidate] = {}
    for frame_id in sorted(selected_owner_ids):
        candidate_by_frame[frame_id] = _render_component_source_candidate(
            frame=frame_by_id[frame_id],
            pose=pose_by_id[frame_id],
            intrinsics=intrinsics,
            layout=layout,
            mesh_config=mesh_config,
            maps=maps,
            panel_anchors=panel_anchors,
            scan_axis=scan_axis,
            selected=selected,
        )

    for item in provisional:
        record = item["record"]
        component = np.asarray(item["component"], dtype=bool)
        frame_id = int(item["selected_frame_id"])
        candidate = candidate_by_frame[frame_id]
        x0 = int(candidate.corner_x)
        x1 = x0 + candidate.valid_mask.shape[1]
        candidate_valid = candidate.valid_mask[
            vertical_crop_y : vertical_crop_y + valid.shape[0]
        ]
        candidate_depth = candidate.target_depth_mm[
            vertical_crop_y : vertical_crop_y + valid.shape[0]
        ]
        candidate_image = candidate.image_bgr[
            vertical_crop_y : vertical_crop_y + valid.shape[0]
        ]
        component_local = component[:, x0:x1]
        base_depth_local = depth[:, x0:x1]
        comparison_tolerance = np.maximum(
            np.float32(selected.temporal_absolute_tolerance_mm),
            np.float32(selected.temporal_relative_tolerance)
            * np.maximum(candidate_depth, base_depth_local),
        )
        same_layer = (
            component_local
            & candidate_valid
            & np.isfinite(base_depth_local)
            & (
                np.abs(candidate_depth - base_depth_local)
                <= comparison_tolerance
            )
        )
        support_count = int(np.count_nonzero(same_layer))
        measured_count = int(record["measured_pixel_count"])
        coverage = float(support_count / max(1, measured_count))
        record["selected_source_reprojection"] = candidate.audit
        record["single_owner_same_layer_pixel_count"] = support_count
        record["single_owner_reprojection_coverage_ratio"] = coverage
        if coverage < selected.component_minimum_single_owner_coverage:
            record["rejection_reason"] = (
                "selected_single_owner_cannot_reproject_complete_component"
            )
            record["owner_count_after"] = int(record["owner_count_before"])
            continue
        locked_image[component] = 0
        locked_owner[component] = -1
        locked_valid[component] = False
        locked_image[:, x0:x1][same_layer] = candidate_image[same_layer]
        locked_owner[:, x0:x1][same_layer] = frame_id
        locked_valid[:, x0:x1][same_layer] = True
        record["accepted"] = True
        record["rejection_reason"] = None
        record["owner_count_after"] = 1

    accepted_count = sum(
        bool(record["accepted"]) for record in component_records
    )
    compact_count = sum(
        bool(record["compact_candidate"]) for record in component_records
    )
    return (
        np.ascontiguousarray(locked_image),
        np.ascontiguousarray(locked_owner),
        np.ascontiguousarray(locked_valid),
        np.ascontiguousarray(labels),
        {
            "enabled": True,
            "formal_publication": False,
            "method": (
                "near_world_connected_component_majority_vote_then_"
                "single_real_frame_same_layer_reprojection_lock"
            ),
            "segmentation_only_close_kernel": [3, 3],
            "segmentation_morphology_writes_rgb": False,
            "reference_depth_mm": reference,
            "near_margin_mm": float(selected.component_near_margin_mm),
            "near_measured_pixel_count": int(np.count_nonzero(near)),
            "depth_edge_rejected_pixel_count": int(
                np.count_nonzero(depth_edge & near)
            ),
            "reported_component_count": len(component_records),
            "compact_candidate_count": compact_count,
            "accepted_single_owner_component_count": accepted_count,
            "rejected_component_count": len(component_records) - accepted_count,
            "selected_source_rerender_count": len(candidate_by_frame),
            "one_real_frame_per_accepted_component": True,
            "no_cross_owner_fill": True,
            "no_rgb_interpolation": True,
            "components": component_records,
        },
    )


def render_world_surface_inspection(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    config: WorldSurfaceInspectionConfig | Mapping[str, object] | None = None,
) -> WorldSurfaceInspectionResult:
    """Render all tracked RGB-D views into one world-locked display surface."""

    selected = (
        config
        if isinstance(config, WorldSurfaceInspectionConfig)
        else WorldSurfaceInspectionConfig.from_mapping(config)
    )
    selected.validate()
    if len(frames) < 2 or len(frames) != len(poses):
        raise ValueError(
            "World-surface prototype needs at least two aligned frames and poses"
        )
    frame_ids = [int(frame.frame_id) for frame in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("World-surface frame IDs must be unique")

    mesh_config = _inspection_config(selected)
    layout = estimate_inspection_layout(
        frames, poses, intrinsics, config=mesh_config
    )
    pixel_count = int(layout.width * layout.height)
    # RGB, owner, target depth, source-centrality and temporary local remaps.
    estimated_peak_bytes = pixel_count * 48 + (
        intrinsics.width * intrinsics.height * 48
    )
    if estimated_peak_bytes > selected.maximum_working_bytes:
        raise MemoryError(
            "World-surface prototype exceeds its estimated byte budget"
        )

    image = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    owner = np.full((layout.height, layout.width), -1, dtype=np.int32)
    target_depth = np.full(
        (layout.height, layout.width), np.inf, dtype=np.float32
    )
    centrality = np.full(
        (layout.height, layout.width), -np.inf, dtype=np.float32
    )
    maps = _undistortion_maps(intrinsics)
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    panel_anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels], dtype=np.float64
    )
    source_audits: list[dict[str, object]] = []
    total_nearer = 0
    total_same_layer = 0
    total_centrality_replacements = 0
    total_written = 0

    for source_position, (frame, pose) in enumerate(
        zip(frames, poses, strict=True)
    ):
        source_image, source_depth, geometric_valid = _read_rgbd(
            frame, intrinsics, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(source_depth)
            & (source_depth >= selected.minimum_depth_mm)
            & (source_depth <= selected.maximum_depth_mm)
        )
        confidence, depth_edge = _depth_confidence(
            source_depth, reliable, mesh_config
        )
        neighbourhood = cv2.boxFilter(
            reliable.astype(np.uint8),
            cv2.CV_16U,
            (3, 3),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        )
        solver_valid = reliable & ~depth_edge & (neighbourhood == 9)
        panel_index = _nearest_panel_for_camera(
            np.asarray(pose, dtype=np.float64), panel_anchors, scan_axis
        )
        mesh = _build_depth_mesh_panel_remap(
            source_depth_mm=source_depth,
            source_solver_valid=solver_valid,
            source_pose=np.asarray(pose, dtype=np.float64),
            panel_index=panel_index,
            layout=layout,
            intrinsics=intrinsics,
            config=mesh_config,
        )
        safe_x = np.where(mesh.valid_mask, mesh.map_x, -1.0).astype(
            np.float32, copy=False
        )
        safe_y = np.where(mesh.valid_mask, mesh.map_y, -1.0).astype(
            np.float32, copy=False
        )
        sampled_image = accelerated_remap(
            source_image,
            safe_x,
            safe_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sampled_solver_valid = accelerated_remap(
            solver_valid.astype(np.uint8) * 255,
            safe_x,
            safe_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sampled_confidence = accelerated_remap(
            confidence.astype(np.float32, copy=False),
            safe_x,
            safe_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        radius = np.sqrt(
            ((safe_x - intrinsics.cx) / max(1.0, intrinsics.width * 0.5)) ** 2
            + ((safe_y - intrinsics.cy) / max(1.0, intrinsics.height * 0.5)) ** 2
        )
        candidate_centrality = (
            np.clip(1.0 - radius, 0.0, 1.0).astype(np.float32)
            * np.clip(sampled_confidence, 0.0, 1.0)
        )
        candidate = (
            mesh.valid_mask
            & (sampled_solver_valid > 0)
            & np.isfinite(mesh.relative_depth_mm)
            & (mesh.relative_depth_mm > 0.0)
        )
        x0 = int(mesh.corner_x)
        x1 = x0 + mesh.valid_mask.shape[1]
        region_owner = owner[:, x0:x1]
        region_depth = target_depth[:, x0:x1]
        region_centrality = centrality[:, x0:x1]
        existing = region_owner >= 0
        comparison_depth = np.where(
            existing, region_depth, mesh.relative_depth_mm
        )
        tolerance = np.maximum(
            np.float32(selected.temporal_absolute_tolerance_mm),
            np.float32(selected.temporal_relative_tolerance)
            * np.maximum(mesh.relative_depth_mm, comparison_depth),
        )
        delta = np.zeros(mesh.relative_depth_mm.shape, dtype=np.float32)
        np.subtract(
            mesh.relative_depth_mm,
            comparison_depth,
            out=delta,
            where=candidate,
        )
        nearer = candidate & existing & (delta < -tolerance)
        same_layer = candidate & existing & (np.abs(delta) <= tolerance)
        better_center = same_layer & (
            candidate_centrality > region_centrality + np.float32(1e-6)
        )
        take = candidate & (~existing | nearer | better_center)
        if np.any(take):
            image[:, x0:x1][take] = sampled_image[take]
            region_owner[take] = int(frame.frame_id)
            region_depth[take] = mesh.relative_depth_mm[take]
            region_centrality[take] = candidate_centrality[take]
        nearer_count = int(np.count_nonzero(nearer))
        same_layer_count = int(np.count_nonzero(same_layer))
        centrality_count = int(np.count_nonzero(better_center))
        write_count = int(np.count_nonzero(take))
        total_nearer += nearer_count
        total_same_layer += same_layer_count
        total_centrality_replacements += centrality_count
        total_written += write_count
        source_audits.append(
            {
                "source_position": int(source_position),
                "frame_id": int(frame.frame_id),
                "target_panel_index": panel_index,
                "input_reliable_depth_pixel_count": int(
                    np.count_nonzero(reliable)
                ),
                "complete_3x3_solver_pixel_count": int(
                    np.count_nonzero(solver_valid)
                ),
                "depth_edge_rejected_pixel_count": int(
                    np.count_nonzero(depth_edge)
                ),
                "mesh": dict(mesh.audit),
                "candidate_pixel_count": int(np.count_nonzero(candidate)),
                "nearer_zbuffer_replacement_count": nearer_count,
                "same_layer_collision_count": same_layer_count,
                "central_view_owner_replacement_count": centrality_count,
                "written_pixel_count": write_count,
            }
        )

    valid = owner >= 0
    target_depth[~valid] = np.nan
    # Crop only the common vertical extent; keeping x coordinates unchanged
    # makes panel/world-position diagnostics easier to compare.
    rows = np.flatnonzero(np.any(valid, axis=1))
    if not rows.size:
        raise RuntimeError("World-surface prototype produced no valid surface")
    y0 = int(rows[0])
    y1 = int(rows[-1]) + 1
    image = np.ascontiguousarray(image[y0:y1])
    owner = np.ascontiguousarray(owner[y0:y1])
    target_depth = np.ascontiguousarray(target_depth[y0:y1])
    valid = np.ascontiguousarray(valid[y0:y1])
    (
        component_locked_image,
        component_locked_owner,
        component_locked_valid,
        component_label,
        component_lock_audit,
    ) = _component_lock_world_surface(
        image_bgr=image,
        owner_frame_id=owner,
        target_depth_mm=target_depth,
        valid_mask=valid,
        frames=frames,
        poses=poses,
        intrinsics=intrinsics,
        layout=layout,
        mesh_config=mesh_config,
        maps=maps,
        selected=selected,
        vertical_crop_y=y0,
    )
    metadata: dict[str, object] = {
        "schema": "gemini305-world-surface-inspection-prototype/v1",
        "formal_publication": False,
        "purpose": (
            "all_real_rgbd_view_world_surface_candidate_for_near_object_"
            "completeness"
        ),
        "method": (
            "all_pose_depth_mesh_nearest_real_rgb_world_panel_zbuffer_"
            "central_view_owner"
        ),
        "real_pose_count": len(poses),
        "all_real_poses_consumed": True,
        "pose_interpolation_count": 0,
        "layout": layout.as_dict(),
        "vertical_crop": {"y": y0, "height": y1 - y0},
        "image_shape": list(image.shape),
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "invalid_pixel_count": int(valid.size - np.count_nonzero(valid)),
        "owner_frame_count": int(np.unique(owner[valid]).size),
        "source_rgb_policy": (
            "nearest_neighbour_copy_from_one_real_rgb_source_pixel"
        ),
        "depth_policy": (
            "real_aligned_depth_gates_piecewise_affine_surface_geometry"
        ),
        "owner_policy": (
            "nearer_target_depth_then_highest_source_image_centrality_"
            "within_same_depth_layer"
        ),
        "no_rgb_interpolation": True,
        "no_multiband": True,
        "no_graphcut": True,
        "no_hole_fill": True,
        "no_tsdf": True,
        "no_generated_colour": True,
        "positive_jacobian_required": True,
        "complete_3x3_depth_support_required": True,
        "depth_boundary_crossing_allowed": False,
        "total_nearer_zbuffer_replacement_count": total_nearer,
        "total_same_layer_collision_count": total_same_layer,
        "total_central_view_owner_replacement_count": (
            total_centrality_replacements
        ),
        "total_write_count": total_written,
        "estimated_peak_bytes": int(estimated_peak_bytes),
        "component_owner_lock": component_lock_audit,
        "config": asdict(selected),
        "sources": source_audits,
    }
    result = WorldSurfaceInspectionResult(
        image_bgr=image,
        owner_frame_id=owner,
        target_depth_mm=target_depth,
        valid_mask=valid,
        component_locked_image_bgr=component_locked_image,
        component_locked_owner_frame_id=component_locked_owner,
        component_locked_valid_mask=component_locked_valid,
        component_label=component_label,
        metadata=metadata,
    )
    result.validate()
    return result


def colourise_owner(owner_frame_id: np.ndarray) -> np.ndarray:
    """Return a deterministic diagnostic colour map for source ownership."""

    owner = np.asarray(owner_frame_id, dtype=np.int32)
    output = np.zeros((*owner.shape, 3), dtype=np.uint8)
    valid = owner >= 0
    value = owner[valid].astype(np.uint32)
    output[valid, 0] = ((value * 73 + 29) % 251 + 4).astype(np.uint8)
    output[valid, 1] = ((value * 151 + 61) % 251 + 4).astype(np.uint8)
    output[valid, 2] = ((value * 199 + 113) % 251 + 4).astype(np.uint8)
    return output


@dataclass(frozen=True)
class AutomaticInstanceConfig:
    """Fail-closed settings for the isolated automatic-instance experiment."""

    minimum_locked_component_pixels: int = 300
    maximum_locked_component_pixels: int = 50_000
    minimum_source_seed_pixels: int = 48
    maximum_candidate_views_per_component: int = 4
    grabcut_iterations: int = 5
    grabcut_bbox_margin_ratio: float = 0.20
    maximum_source_bbox_fraction: float = 0.45
    maximum_grabcut_to_seed_area_ratio: float = 8.0
    minimum_instance_reliable_depth_ratio: float = 0.50
    minimum_target_bbox_iou: float = 0.40
    minimum_target_mask_iou: float = 0.25
    maximum_world_centroid_distance_mm: float = 60.0
    maximum_world_span_ratio: float = 2.50
    duplicate_target_mask_iou: float = 0.35

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None = None
    ) -> "AutomaticInstanceConfig":
        payload = {} if value is None else dict(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"Unknown automatic-instance configuration keys: {unknown}"
            )
        try:
            selected = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid automatic-instance configuration") from exc
        selected.validate()
        return selected

    def validate(self) -> None:
        integer_positive = (
            ("minimum_locked_component_pixels", self.minimum_locked_component_pixels),
            ("maximum_locked_component_pixels", self.maximum_locked_component_pixels),
            ("minimum_source_seed_pixels", self.minimum_source_seed_pixels),
            (
                "maximum_candidate_views_per_component",
                self.maximum_candidate_views_per_component,
            ),
            ("grabcut_iterations", self.grabcut_iterations),
        )
        for name, value in integer_positive:
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.maximum_locked_component_pixels
            < self.minimum_locked_component_pixels
        ):
            raise ValueError("Automatic-instance component range is empty")
        fractions = (
            ("grabcut_bbox_margin_ratio", self.grabcut_bbox_margin_ratio),
            ("maximum_source_bbox_fraction", self.maximum_source_bbox_fraction),
            (
                "minimum_instance_reliable_depth_ratio",
                self.minimum_instance_reliable_depth_ratio,
            ),
            ("minimum_target_bbox_iou", self.minimum_target_bbox_iou),
            ("minimum_target_mask_iou", self.minimum_target_mask_iou),
            ("duplicate_target_mask_iou", self.duplicate_target_mask_iou),
        )
        for name, value in fractions:
            if not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        for name, value in (
            (
                "maximum_grabcut_to_seed_area_ratio",
                self.maximum_grabcut_to_seed_area_ratio,
            ),
            (
                "maximum_world_centroid_distance_mm",
                self.maximum_world_centroid_distance_mm,
            ),
            ("maximum_world_span_ratio", self.maximum_world_span_ratio),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AutomaticInstanceResult:
    image_bgr: np.ndarray
    owner_frame_id: np.ndarray
    instance_label: np.ndarray
    metadata: dict[str, object]

    def validate(self) -> None:
        shape = self.image_bgr.shape[:2]
        if self.image_bgr.dtype != np.uint8 or self.image_bgr.shape != (*shape, 3):
            raise RuntimeError("Automatic-instance RGB output is malformed")
        if (
            self.owner_frame_id.shape != shape
            or self.owner_frame_id.dtype != np.int32
            or self.instance_label.shape != shape
            or self.instance_label.dtype != np.int32
        ):
            raise RuntimeError("Automatic-instance rasters are misaligned")
        instance = self.instance_label > 0
        if np.any(self.owner_frame_id[instance] < 0):
            raise RuntimeError("Automatic instance lacks its real RGB owner")


@dataclass(frozen=True)
class _AutomaticPanelSource:
    panel_index: int
    source_position: int
    frame_id: int
    frame: RGBDFrame
    pose: np.ndarray
    image_bgr: np.ndarray
    depth_mm: np.ndarray
    reliable_depth: np.ndarray
    mesh: object
    sampled_target_image_bgr: np.ndarray


@dataclass(frozen=True)
class _AutomaticInstanceObservation:
    frame_id: int
    panel_index: int
    source_mask: np.ndarray
    target_mask: np.ndarray
    source_seed_pixel_count: int
    source_mask_pixel_count: int
    target_pixel_count: int
    source_bbox_xywh: tuple[int, int, int, int]
    target_bbox_xywh: tuple[int, int, int, int]
    world_min_sdn_mm: tuple[float, float, float]
    world_max_sdn_mm: tuple[float, float, float]
    world_centroid_sdn_mm: tuple[float, float, float]
    median_target_depth_mm: float
    centrality: float
    sharpness: float
    audit: dict[str, object]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    yy, xx = np.nonzero(mask)
    if not xx.size:
        return None
    x0 = int(xx.min())
    y0 = int(yy.min())
    return x0, y0, int(xx.max()) + 1 - x0, int(yy.max()) + 1 - y0


def _bbox_iou_xywh(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x0 = max(ax, bx)
    y0 = max(ay, by)
    x1 = min(ax + aw, bx + bw)
    y1 = min(ay + ah, by + bh)
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def _automatic_source_seed_grabcut(
    *,
    component: np.ndarray,
    source: _AutomaticPanelSource,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    scan_axis: np.ndarray,
    down_axis: np.ndarray,
    normal_axis: np.ndarray,
    config: AutomaticInstanceConfig,
) -> tuple[_AutomaticInstanceObservation | None, dict[str, object]]:
    mesh = source.mesh
    x0 = int(mesh.corner_x)
    x1 = x0 + mesh.valid_mask.shape[1]
    local_component = component[:, x0:x1]
    target_seed = local_component & mesh.valid_mask
    target_seed_count = int(np.count_nonzero(target_seed))
    rejected: dict[str, object] = {
        "frame_id": int(source.frame_id),
        "panel_index": int(source.panel_index),
        "target_mesh_seed_pixel_count": target_seed_count,
        "accepted": False,
    }
    if target_seed_count < config.minimum_source_seed_pixels:
        rejected["rejection_reason"] = "insufficient_inverse_mesh_target_seed"
        return None, rejected

    source_seed = np.zeros(source.image_bgr.shape[:2], dtype=np.uint8)
    source_x = np.rint(mesh.map_x[target_seed]).astype(np.int32)
    source_y = np.rint(mesh.map_y[target_seed]).astype(np.int32)
    inside = (
        (source_x >= 0)
        & (source_x < intrinsics.width)
        & (source_y >= 0)
        & (source_y < intrinsics.height)
    )
    source_seed[source_y[inside], source_x[inside]] = 1
    source_seed = cv2.morphologyEx(
        source_seed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    source_seed = cv2.dilate(
        source_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    source_seed_count = int(np.count_nonzero(source_seed))
    rejected["source_seed_pixel_count"] = source_seed_count
    bbox = _bbox_from_mask(source_seed)
    if (
        bbox is None
        or source_seed_count < config.minimum_source_seed_pixels
    ):
        rejected["rejection_reason"] = "inverse_mapped_source_seed_is_empty"
        return None, rejected
    bx, by, bw, bh = bbox
    margin = max(
        8,
        int(
            math.ceil(
                config.grabcut_bbox_margin_ratio * max(bw, bh)
            )
        ),
    )
    gx0 = max(0, bx - margin)
    gy0 = max(0, by - margin)
    gx1 = min(intrinsics.width, bx + bw + margin)
    gy1 = min(intrinsics.height, by + bh + margin)
    grabcut_bbox = (gx0, gy0, gx1 - gx0, gy1 - gy0)
    bbox_fraction = float(
        grabcut_bbox[2]
        * grabcut_bbox[3]
        / (intrinsics.width * intrinsics.height)
    )
    rejected["automatic_source_bbox_xywh"] = list(grabcut_bbox)
    rejected["automatic_source_bbox_fraction"] = bbox_fraction
    if bbox_fraction > config.maximum_source_bbox_fraction:
        rejected["rejection_reason"] = "automatic_source_bbox_is_not_compact"
        return None, rejected

    grabcut_mask = np.full(
        source_seed.shape, cv2.GC_BGD, dtype=np.uint8
    )
    grabcut_mask[gy0:gy1, gx0:gx1] = cv2.GC_PR_BGD
    probable = cv2.dilate(
        source_seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ).astype(bool)
    grabcut_mask[probable] = cv2.GC_PR_FGD
    grabcut_mask[source_seed > 0] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            source.image_bgr,
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            int(config.grabcut_iterations),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        rejected["rejection_reason"] = "grabcut_solver_failed"
        rejected["grabcut_error"] = str(exc)
        return None, rejected
    foreground = (grabcut_mask == cv2.GC_FGD) | (
        grabcut_mask == cv2.GC_PR_FGD
    )
    # Retain only automatic GrabCut regions that contain inverse-mesh seeds.
    component_count, component_labels = cv2.connectedComponents(
        foreground.astype(np.uint8), 8
    )
    retained = np.zeros_like(foreground)
    seed_labels = np.unique(component_labels[source_seed > 0])
    seed_labels = seed_labels[seed_labels > 0]
    for label in seed_labels:
        retained |= component_labels == int(label)
    if component_count <= 1 or not np.any(retained):
        rejected["rejection_reason"] = "grabcut_has_no_seed_connected_foreground"
        return None, rejected
    mask_count = int(np.count_nonzero(retained))
    expansion_ratio = float(mask_count / max(1, source_seed_count))
    rejected["grabcut_source_mask_pixel_count"] = mask_count
    rejected["grabcut_to_seed_area_ratio"] = expansion_ratio
    if (
        mask_count < source_seed_count
        or expansion_ratio > config.maximum_grabcut_to_seed_area_ratio
    ):
        rejected["rejection_reason"] = "grabcut_expansion_is_unbounded"
        return None, rejected

    reliable_instance = retained & source.reliable_depth
    reliable_count = int(np.count_nonzero(reliable_instance))
    reliable_ratio = float(reliable_count / max(1, mask_count))
    rejected["reliable_depth_pixel_count"] = reliable_count
    rejected["reliable_depth_ratio"] = reliable_ratio
    if reliable_ratio < config.minimum_instance_reliable_depth_ratio:
        rejected["rejection_reason"] = "grabcut_instance_lacks_rgbd_support"
        return None, rejected

    yy, xx = np.nonzero(reliable_instance)
    stride = max(1, int(math.ceil(xx.size / 12_000)))
    xx_sample = xx[::stride]
    yy_sample = yy[::stride]
    camera = pinhole_unproject(
        xx_sample,
        yy_sample,
        source.depth_mm[yy_sample, xx_sample],
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    world = transform_points(
        camera, source.pose[:3, :3], source.pose[:3, 3]
    )
    world_sdn = np.column_stack(
        (world @ scan_axis, world @ down_axis, world @ normal_axis)
    )
    if world_sdn.shape[0] < 32 or not np.isfinite(world_sdn).all():
        rejected["rejection_reason"] = "instance_world_support_is_invalid"
        return None, rejected
    world_min = np.quantile(world_sdn, 0.02, axis=0)
    world_max = np.quantile(world_sdn, 0.98, axis=0)
    world_centroid = np.median(world_sdn, axis=0)

    safe_x = np.where(mesh.valid_mask, mesh.map_x, -1.0).astype(np.float32)
    safe_y = np.where(mesh.valid_mask, mesh.map_y, -1.0).astype(np.float32)
    target_mask_local = accelerated_remap(
        retained.astype(np.uint8) * 255,
        safe_x,
        safe_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    target_mask = np.zeros(
        (int(layout.height), int(layout.width)), dtype=bool
    )
    target_mask[:, x0:x1] = (target_mask_local > 0) & mesh.valid_mask
    target_bbox = _bbox_from_mask(target_mask)
    if target_bbox is None or np.count_nonzero(target_mask) < 32:
        rejected["rejection_reason"] = "direct_se3_target_mask_is_empty"
        return None, rejected
    source_bbox = _bbox_from_mask(retained)
    assert source_bbox is not None
    source_radius = math.hypot(
        float(np.median(xx) - intrinsics.cx)
        / max(1.0, intrinsics.width * 0.5),
        float(np.median(yy) - intrinsics.cy)
        / max(1.0, intrinsics.height * 0.5),
    )
    centrality = float(np.clip(1.0 - source_radius, 0.0, 1.0))
    gray = cv2.cvtColor(source.image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_32F)[retained].var()
    )
    target_depth = mesh.relative_depth_mm[target_mask_local > 0]
    observation = _AutomaticInstanceObservation(
        frame_id=int(source.frame_id),
        panel_index=int(source.panel_index),
        source_mask=np.ascontiguousarray(retained),
        target_mask=np.ascontiguousarray(target_mask),
        source_seed_pixel_count=source_seed_count,
        source_mask_pixel_count=mask_count,
        target_pixel_count=int(np.count_nonzero(target_mask)),
        source_bbox_xywh=source_bbox,
        target_bbox_xywh=target_bbox,
        world_min_sdn_mm=tuple(float(value) for value in world_min),
        world_max_sdn_mm=tuple(float(value) for value in world_max),
        world_centroid_sdn_mm=tuple(float(value) for value in world_centroid),
        median_target_depth_mm=float(np.median(target_depth)),
        centrality=centrality,
        sharpness=sharpness,
        audit={
            **rejected,
            "accepted": True,
            "rejection_reason": None,
            "source_bbox_xywh": list(source_bbox),
            "target_bbox_xywh": list(target_bbox),
            "target_pixel_count": int(np.count_nonzero(target_mask)),
            "world_min_sdn_mm": world_min.tolist(),
            "world_max_sdn_mm": world_max.tolist(),
            "world_centroid_sdn_mm": world_centroid.tolist(),
            "direct_se3_mesh_projection": True,
            "affine_transform_used": False,
        },
    )
    return observation, dict(observation.audit)


def _automatic_observation_consistency(
    first: _AutomaticInstanceObservation,
    second: _AutomaticInstanceObservation,
    config: AutomaticInstanceConfig,
) -> tuple[bool, dict[str, object]]:
    centroid_distance = float(
        np.linalg.norm(
            np.asarray(first.world_centroid_sdn_mm)
            - np.asarray(second.world_centroid_sdn_mm)
        )
    )
    first_span = (
        np.asarray(first.world_max_sdn_mm)
        - np.asarray(first.world_min_sdn_mm)
    )
    second_span = (
        np.asarray(second.world_max_sdn_mm)
        - np.asarray(second.world_min_sdn_mm)
    )
    bounded = np.maximum(np.minimum(first_span, second_span), 4.0)
    span_ratio = float(
        np.max(np.maximum(first_span, second_span) / bounded)
    )
    bbox_iou = _bbox_iou_xywh(
        first.target_bbox_xywh, second.target_bbox_xywh
    )
    mask_iou = _mask_iou(first.target_mask, second.target_mask)
    passed = (
        first.frame_id != second.frame_id
        and centroid_distance <= config.maximum_world_centroid_distance_mm
        and span_ratio <= config.maximum_world_span_ratio
        and bbox_iou >= config.minimum_target_bbox_iou
        and mask_iou >= config.minimum_target_mask_iou
    )
    return passed, {
        "first_frame_id": int(first.frame_id),
        "second_frame_id": int(second.frame_id),
        "world_centroid_distance_mm": centroid_distance,
        "maximum_world_span_ratio": span_ratio,
        "target_bbox_iou": bbox_iou,
        "target_mask_iou": mask_iou,
        "pass": bool(passed),
    }


def render_automatic_instance_candidates(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    v9_renderer_metadata: Mapping[str, object],
    v9_inspection_image_bgr: np.ndarray,
    config: AutomaticInstanceConfig | Mapping[str, object] | None = None,
) -> AutomaticInstanceResult:
    """Test automatic GrabCut instances from formal locked RGB-D evidence.

    The formal output is read-only input evidence.  No ROI, object label or
    source frame is supplied by the caller: components and candidate views
    come exclusively from the v9 lock/mesh audit.
    """

    selected = (
        config
        if isinstance(config, AutomaticInstanceConfig)
        else AutomaticInstanceConfig.from_mapping(config)
    )
    selected.validate()
    renderer = dict(v9_renderer_metadata)
    if renderer.get("schema") != "gemini305-inspection-multiview/v1":
        raise ValueError("Automatic-instance prototype requires v9 renderer metadata")
    if int(renderer.get("real_pose_count", -1)) != len(poses):
        raise ValueError("v9 renderer pose count does not match real poses")
    if len(frames) != len(poses):
        raise ValueError("Automatic-instance frames and poses are not aligned")
    frame_ids = [int(frame.frame_id) for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Automatic-instance frame IDs must be unique")

    formal_config_payload = renderer.get("config")
    if not isinstance(formal_config_payload, Mapping):
        raise ValueError("v9 renderer omitted its closed inspection config")
    formal_config = InspectionMultiviewConfig.from_mapping(
        formal_config_payload
    )
    layout = estimate_inspection_layout(
        frames, poses, intrinsics, config=formal_config
    )
    layout_audit = renderer.get("layout")
    if not isinstance(layout_audit, Mapping):
        raise ValueError("v9 renderer omitted its layout audit")
    if (
        int(layout_audit.get("width", -1)) != layout.width
        or int(layout_audit.get("height", -1)) != layout.height
        or int(layout_audit.get("panel_count", -1)) != len(layout.panels)
        or not math.isclose(
            float(layout_audit.get("reference_depth_mm", math.nan)),
            layout.reference_depth_mm,
            abs_tol=1e-6,
        )
    ):
        raise ValueError("Reconstructed automatic-instance layout differs from v9")

    panel_selection = renderer.get("selected_panel_sources")
    if not isinstance(panel_selection, list) or len(panel_selection) != len(
        layout.panels
    ):
        raise ValueError("v9 selected panel sources are incomplete")
    maps = _undistortion_maps(intrinsics)
    formal_source_audits_payload = renderer.get("source_audits")
    if not isinstance(formal_source_audits_payload, list):
        raise ValueError("v9 renderer omitted source mesh audits")
    formal_source_audit_by_frame = {
        int(item["frame_id"]): item
        for item in formal_source_audits_payload
        if isinstance(item, Mapping)
    }
    panel_sources: list[_AutomaticPanelSource] = []
    for expected_panel, payload in enumerate(panel_selection):
        if not isinstance(payload, Mapping):
            raise ValueError("v9 panel source audit is malformed")
        panel_index = int(payload.get("panel_index", -1))
        source_position = int(payload.get("source_position", -1))
        frame_id = int(payload.get("frame_id", -1))
        if (
            panel_index != expected_panel
            or not 0 <= source_position < len(frames)
            or int(frames[source_position].frame_id) != frame_id
        ):
            raise ValueError("v9 panel source identity cannot be reconstructed")
        frame = frames[source_position]
        pose = np.asarray(poses[source_position], dtype=np.float64)
        image, depth, geometric_valid = _read_rgbd(frame, intrinsics, maps)
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= formal_config.minimum_depth_mm)
            & (depth <= formal_config.maximum_depth_mm)
        )
        confidence, edge = _depth_confidence(
            depth, reliable, formal_config
        )
        foreground_margin = max(
            formal_config.foreground_depth_margin_mm,
            formal_config.foreground_depth_margin_ratio
            * layout.reference_depth_mm,
        )
        geometry_depth_limit = min(
            layout.reference_depth_mm - foreground_margin,
            layout.reference_depth_mm
            * formal_config.foreground_reference_depth_ratio,
        )
        geometry_depth = reliable & (depth < geometry_depth_limit)
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        normalised_radius = np.sqrt(
            ((xx - intrinsics.cx) / max(1.0, intrinsics.width * 0.5)) ** 2
            + ((yy - intrinsics.cy) / max(1.0, intrinsics.height * 0.5)) ** 2
        )
        view_centrality = np.clip(1.0 - normalised_radius, 0.0, 1.0)
        confidence[geometry_depth] *= (
            0.35 + 0.65 * view_centrality[geometry_depth]
        )
        projection_valid = (
            geometry_depth & ~edge & (confidence >= np.float32(0.50))
        )
        mesh = _build_depth_mesh_panel_remap(
            source_depth_mm=depth,
            source_solver_valid=projection_valid,
            source_pose=pose,
            panel_index=panel_index,
            layout=layout,
            intrinsics=intrinsics,
            config=formal_config,
        )
        formal_source_audit = formal_source_audit_by_frame.get(frame_id)
        if not isinstance(formal_source_audit, Mapping):
            raise RuntimeError(
                f"v9 omitted source mesh audit for frame {frame_id}"
            )
        formal_mesh_audit = formal_source_audit.get("depth_mesh")
        if (
            not isinstance(formal_mesh_audit, Mapping)
            or int(formal_mesh_audit.get("valid_target_pixel_count", -1))
            != int(mesh.audit["valid_target_pixel_count"])
            or int(formal_mesh_audit.get("accepted_cell_count", -1))
            != int(mesh.audit["accepted_cell_count"])
        ):
            raise RuntimeError(
                "Reconstructed v9 source mesh changed: "
                f"frame={frame_id}, "
                f"formal_valid={formal_mesh_audit.get('valid_target_pixel_count') if isinstance(formal_mesh_audit, Mapping) else None}, "
                f"current_valid={mesh.audit['valid_target_pixel_count']}, "
                f"formal_cells={formal_mesh_audit.get('accepted_cell_count') if isinstance(formal_mesh_audit, Mapping) else None}, "
                f"current_cells={mesh.audit['accepted_cell_count']}"
            )
        safe_x = np.where(mesh.valid_mask, mesh.map_x, -1.0).astype(
            np.float32, copy=False
        )
        safe_y = np.where(mesh.valid_mask, mesh.map_y, -1.0).astype(
            np.float32, copy=False
        )
        sampled_target = accelerated_remap(
            image,
            safe_x,
            safe_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        panel_sources.append(
            _AutomaticPanelSource(
                panel_index=panel_index,
                source_position=source_position,
                frame_id=frame_id,
                frame=frame,
                pose=pose,
                image_bgr=np.ascontiguousarray(image),
                depth_mm=np.ascontiguousarray(depth),
                reliable_depth=np.ascontiguousarray(reliable),
                mesh=mesh,
                sampled_target_image_bgr=np.ascontiguousarray(sampled_target),
            )
        )

    shape = (layout.height, layout.width)
    nearest_depth = np.full(shape, np.inf, dtype=np.float32)
    evidence = np.zeros(shape, dtype=bool)
    for source in panel_sources:
        mesh = source.mesh
        x0 = int(mesh.corner_x)
        x1 = x0 + mesh.valid_mask.shape[1]
        local_depth = nearest_depth[:, x0:x1]
        take = mesh.valid_mask & (
            ~np.isfinite(local_depth)
            | (mesh.relative_depth_mm < local_depth)
        )
        local_depth[take] = mesh.relative_depth_mm[take]
        evidence[:, x0:x1] |= mesh.valid_mask
    raw_labels, raw_components = _foreground_depth_layer_components(
        evidence,
        nearest_depth,
        layout.reference_depth_mm,
        formal_config,
    )
    reconstructed_components: dict[int, np.ndarray] = {}
    occupied = np.zeros(shape, dtype=np.int32)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for component_id, area in raw_components:
        if area < formal_config.minimum_foreground_component_pixels:
            continue
        component = (
            cv2.morphologyEx(
                (raw_labels == int(component_id)).astype(np.uint8),
                cv2.MORPH_CLOSE,
                close_kernel,
            )
            > 0
        )
        component &= occupied == 0
        if (
            np.count_nonzero(component)
            < formal_config.minimum_foreground_component_pixels
        ):
            continue
        occupied[component] = int(component_id)
        reconstructed_components[int(component_id)] = component

    seam_audit = renderer.get("background_seam_audit")
    if not isinstance(seam_audit, Mapping):
        raise ValueError("v9 renderer omitted background seam evidence")
    lock_audit = seam_audit.get("foreground_component_locks")
    if not isinstance(lock_audit, Mapping):
        raise ValueError("v9 renderer omitted foreground component locks")
    formal_components = lock_audit.get("components")
    if not isinstance(formal_components, list):
        raise ValueError("v9 foreground component lock audit is malformed")
    formal_by_id = {
        int(item["component_id"]): item
        for item in formal_components
        if isinstance(item, Mapping)
    }
    if (
        len(formal_by_id) != int(lock_audit.get("component_count", -1))
        or set(formal_by_id) != set(reconstructed_components)
    ):
        missing_reconstructed = sorted(
            set(formal_by_id) - set(reconstructed_components)
        )
        unexpected_reconstructed = sorted(
            set(reconstructed_components) - set(formal_by_id)
        )
        raise RuntimeError(
            "Automatic-instance reconstruction does not match v9 component IDs: "
            f"formal={len(formal_by_id)}, "
            f"reconstructed={len(reconstructed_components)}, "
            f"missing={missing_reconstructed[:20]}, "
            f"unexpected={unexpected_reconstructed[:20]}"
        )
    for component_id, component in reconstructed_components.items():
        expected_area = int(formal_by_id[component_id]["area_pixels"])
        if expected_area != int(np.count_nonzero(component)):
            raise RuntimeError(
                f"v9 component {component_id} area changed during reconstruction"
            )

    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    down_axis = np.asarray(layout.down_axis, dtype=np.float64)
    normal_axis = np.asarray(layout.normal_axis, dtype=np.float64)
    track_candidates: list[dict[str, object]] = []
    component_audits: list[dict[str, object]] = []
    for component_id, formal_component in sorted(formal_by_id.items()):
        component = reconstructed_components[component_id]
        area = int(np.count_nonzero(component))
        component_audit: dict[str, object] = {
            "component_id": component_id,
            "locked_area_pixels": area,
            "formal_selected_frame_id": int(
                formal_component["selected_frame_id"]
            ),
            "formal_locked_before_seam": bool(
                formal_component["locked_before_seam"]
            ),
            "automatic_roi_used": True,
            "manual_roi_used": False,
            "manual_frame_selection_used": False,
            "accepted": False,
        }
        if not (
            selected.minimum_locked_component_pixels
            <= area
            <= selected.maximum_locked_component_pixels
        ):
            component_audit["rejection_reason"] = (
                "locked_component_outside_automatic_instance_size_envelope"
            )
            component_audits.append(component_audit)
            continue
        candidate_payloads = [
            item
            for item in formal_component.get("candidates", [])
            if isinstance(item, Mapping)
            and int(item.get("depth_mesh_coverage_pixels", 0))
            >= selected.minimum_source_seed_pixels
        ]
        candidate_payloads.sort(
            key=lambda item: (
                -int(item["depth_mesh_coverage_pixels"]),
                -int(item["reference_coverage_pixels"]),
                int(item["frame_id"]),
            )
        )
        candidate_payloads = candidate_payloads[
            : selected.maximum_candidate_views_per_component
        ]
        observations: list[_AutomaticInstanceObservation] = []
        observation_audits: list[dict[str, object]] = []
        for payload in candidate_payloads:
            panel_index = int(payload["panel_index"])
            source = panel_sources[panel_index]
            if int(payload["frame_id"]) != source.frame_id:
                raise RuntimeError("v9 component candidate frame is misaligned")
            observation, observation_audit = _automatic_source_seed_grabcut(
                component=component,
                source=source,
                layout=layout,
                intrinsics=intrinsics,
                scan_axis=scan_axis,
                down_axis=down_axis,
                normal_axis=normal_axis,
                config=selected,
            )
            observation_audits.append(observation_audit)
            if observation is not None:
                observations.append(observation)
        component_audit["automatic_candidate_view_count"] = len(
            candidate_payloads
        )
        component_audit["valid_grabcut_observation_count"] = len(observations)
        component_audit["observations"] = observation_audits
        if len(observations) < 2:
            component_audit["rejection_reason"] = (
                "fewer_than_two_valid_rgbd_grabcut_views"
            )
            component_audits.append(component_audit)
            continue
        pair_audits: list[dict[str, object]] = []
        peer_count = np.zeros(len(observations), dtype=np.int32)
        for first_index, first in enumerate(observations):
            for second_index in range(first_index + 1, len(observations)):
                passed, pair_audit = _automatic_observation_consistency(
                    first, observations[second_index], selected
                )
                pair_audits.append(pair_audit)
                if passed:
                    peer_count[first_index] += 1
                    peer_count[second_index] += 1
        component_audit["cross_view_consistency"] = pair_audits
        if int(np.max(peer_count, initial=0)) < 1:
            component_audit["rejection_reason"] = (
                "two_view_world_or_target_bbox_consistency_failed"
            )
            component_audits.append(component_audit)
            continue
        eligible_indices = np.flatnonzero(peer_count > 0)
        selected_index = max(
            eligible_indices,
            key=lambda index: (
                int(peer_count[index]),
                observations[index].centrality,
                math.log1p(max(0.0, observations[index].sharpness)),
                observations[index].target_pixel_count,
                -observations[index].frame_id,
            ),
        )
        observation = observations[int(selected_index)]
        support_frames = sorted(
            {
                int(observation.frame_id),
                *(
                    int(pair["second_frame_id"])
                    if int(pair["first_frame_id"]) == observation.frame_id
                    else int(pair["first_frame_id"])
                    for pair in pair_audits
                    if pair["pass"]
                    and observation.frame_id
                    in (
                        int(pair["first_frame_id"]),
                        int(pair["second_frame_id"]),
                    )
                ),
            }
        )
        component_audit.update(
            {
                "selected_frame_id": int(observation.frame_id),
                "selected_panel_index": int(observation.panel_index),
                "two_view_support_frame_ids": support_frames,
                "selected_target_bbox_xywh": list(
                    observation.target_bbox_xywh
                ),
                "selected_target_pixel_count": int(
                    observation.target_pixel_count
                ),
                "accepted": True,
                "rejection_reason": None,
                "direct_se3_owner": True,
            }
        )
        track_candidates.append(
            {
                "component_audit": component_audit,
                "observation": observation,
                "support_frame_count": len(support_frames),
            }
        )
        component_audits.append(component_audit)

    accepted_tracks: list[dict[str, object]] = []
    for candidate in sorted(
        track_candidates,
        key=lambda item: (
            -int(item["support_frame_count"]),
            -int(item["observation"].target_pixel_count),
            int(item["component_audit"]["component_id"]),
        ),
    ):
        observation = candidate["observation"]
        duplicate = None
        for accepted in accepted_tracks:
            overlap = _mask_iou(
                observation.target_mask,
                accepted["observation"].target_mask,
            )
            if overlap >= selected.duplicate_target_mask_iou:
                duplicate = {
                    "kept_component_id": int(
                        accepted["component_audit"]["component_id"]
                    ),
                    "target_mask_iou": overlap,
                }
                break
        if duplicate is not None:
            audit = candidate["component_audit"]
            audit["accepted"] = False
            audit["rejection_reason"] = "duplicate_automatic_instance_track"
            audit["duplicate_of"] = duplicate
            continue
        accepted_tracks.append(candidate)

    v9_image = np.asarray(v9_inspection_image_bgr, dtype=np.uint8)
    crop_payload = renderer.get("crop")
    if not isinstance(crop_payload, Mapping):
        raise ValueError("v9 renderer omitted its final crop")
    crop = (
        int(crop_payload["x"]),
        int(crop_payload["y"]),
        int(crop_payload["width"]),
        int(crop_payload["height"]),
    )
    cx, cy, cw, ch = crop
    if v9_image.shape != (ch, cw, 3):
        raise ValueError("v9 inspection image does not match its crop audit")
    full_image = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    full_image[cy : cy + ch, cx : cx + cw] = v9_image
    full_owner = np.full(shape, -1, dtype=np.int32)
    full_label = np.zeros(shape, dtype=np.int32)
    instance_depth = np.full(shape, np.inf, dtype=np.float32)
    source_by_frame = {source.frame_id: source for source in panel_sources}
    rendered_track_audits: list[dict[str, object]] = []
    for instance_id, track in enumerate(
        sorted(
            accepted_tracks,
            key=lambda item: (
                -float(item["observation"].median_target_depth_mm),
                int(item["component_audit"]["component_id"]),
            ),
        ),
        start=1,
    ):
        observation = track["observation"]
        source = source_by_frame[int(observation.frame_id)]
        mesh = source.mesh
        x0 = int(mesh.corner_x)
        x1 = x0 + mesh.valid_mask.shape[1]
        local_mask = observation.target_mask[:, x0:x1]
        local_depth = mesh.relative_depth_mm
        take = (
            local_mask
            & mesh.valid_mask
            & np.isfinite(local_depth)
            & (local_depth < instance_depth[:, x0:x1])
        )
        full_image[:, x0:x1][take] = source.sampled_target_image_bgr[take]
        full_owner[:, x0:x1][take] = int(source.frame_id)
        full_label[:, x0:x1][take] = int(instance_id)
        instance_depth[:, x0:x1][take] = local_depth[take]
        rendered_track_audits.append(
            {
                "instance_id": instance_id,
                "component_id": int(
                    track["component_audit"]["component_id"]
                ),
                "owner_frame_id": int(source.frame_id),
                "support_frame_count": int(track["support_frame_count"]),
                "rendered_pixel_count": int(np.count_nonzero(take)),
                "target_bbox_xywh": list(observation.target_bbox_xywh),
                "direct_se3_mesh_projection": True,
                "nearest_real_rgb_only": True,
            }
        )

    output_image = np.ascontiguousarray(
        full_image[cy : cy + ch, cx : cx + cw]
    )
    output_owner = np.ascontiguousarray(
        full_owner[cy : cy + ch, cx : cx + cw]
    )
    output_label = np.ascontiguousarray(
        full_label[cy : cy + ch, cx : cx + cw]
    )
    metadata: dict[str, object] = {
        "schema": "gemini305-automatic-instance-prototype/v1",
        "formal_publication": False,
        "method": (
            "v9_locked_foreground_mesh_inverse_seed_automatic_bbox_"
            "grabcut_two_view_world_target_consistency_single_direct_se3_owner"
        ),
        "manual_roi_used": False,
        "manual_frame_selection_used": False,
        "semantic_instance_model_used": False,
        "grabcut_seed_source": (
            "v9_locked_foreground_component_selected_rgbd_mesh_inverse_map"
        ),
        "real_pose_count": len(poses),
        "selected_formal_panel_source_count": len(panel_sources),
        "reconstructed_v9_component_count": len(reconstructed_components),
        "candidate_component_count": sum(
            selected.minimum_locked_component_pixels
            <= int(item["locked_area_pixels"])
            <= selected.maximum_locked_component_pixels
            for item in component_audits
        ),
        "two_view_accepted_before_duplicate_suppression": len(
            track_candidates
        ),
        "accepted_instance_count": len(accepted_tracks),
        "rejected_component_count": sum(
            not bool(item["accepted"]) for item in component_audits
        ),
        "direct_se3_owner_only": True,
        "affine_transform_used": False,
        "rgb_interpolation_used": False,
        "hole_fill_used": False,
        "tsdf_used": False,
        "output_crop": {
            "x": cx,
            "y": cy,
            "width": cw,
            "height": ch,
        },
        "config": asdict(selected),
        "components": component_audits,
        "rendered_instances": rendered_track_audits,
    }
    result = AutomaticInstanceResult(
        image_bgr=output_image,
        owner_frame_id=output_owner,
        instance_label=output_label,
        metadata=metadata,
    )
    result.validate()
    return result
