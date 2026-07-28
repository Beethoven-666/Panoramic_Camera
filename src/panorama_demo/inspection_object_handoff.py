"""Fail-closed, single-RGB-owner handoff for inspection-panel objects.

This module deliberately does not discover semantic objects.  Its input mask
must already be a measured, cross-view RGB-D component.  It solves the two
remaining problems:

* fit one local source-to-inspection sampling transform from real RGB-D mesh
  correspondences, with an excluded held-out set; and
* turn all view-dependent reference-plane copies of that component into one
  contiguous panel-owner interval.

The result never changes a pose, blends an object, or invents colour.  Every
accepted output pixel is sampled from one real RGB frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

from .cuda_backend import pinhole_unproject, transform_points
from .session import CameraIntrinsics

if TYPE_CHECKING:
    from .inspection_multiview import InspectionMultiviewLayout


@dataclass(frozen=True)
class CompleteObjectOwner:
    """One accepted local RGB owner in inspection-canvas coordinates."""

    frame_id: int
    panel_index: int
    source_to_canvas: np.ndarray
    target_mask: np.ndarray
    target_image_bgr: np.ndarray
    audit: dict[str, object]


@dataclass(frozen=True)
class ObjectOwnerInterval:
    """A row-contiguous owner lock covering every view-dependent copy."""

    panel_index: int
    lock_mask: np.ndarray
    union_footprint: np.ndarray
    audit: dict[str, object]


@dataclass(frozen=True)
class ObjectHandoffSource:
    """One real panel source and its already-audited RGB-D mesh."""

    frame_id: int
    panel_index: int
    image_bgr: np.ndarray
    depth_mm: np.ndarray
    reliable_depth: np.ndarray
    camera_to_world: np.ndarray
    mesh_corner_x: int
    mesh_map_x: np.ndarray
    mesh_map_y: np.ndarray
    mesh_valid_mask: np.ndarray
    mesh_relative_depth_mm: np.ndarray


@dataclass(frozen=True)
class AutomaticObjectHandoff:
    """Cross-view-consistent automatic component and selected owner."""

    owner: CompleteObjectOwner
    source_object_mask: np.ndarray
    audit: dict[str, object]


class AutomaticObjectHandoffRejected(RuntimeError):
    """Fail-closed automatic handoff rejection with scalar-only evidence."""

    def __init__(self, message: str, audit: dict[str, object]) -> None:
        super().__init__(message)
        self.audit = audit


def _project_world_points(
    points_world_mm: np.ndarray,
    layout: "InspectionMultiviewLayout",
    intrinsics: CameraIntrinsics,
    *,
    forced_panel_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world_mm, dtype=np.float64)
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    down_axis = np.asarray(layout.down_axis, dtype=np.float64)
    normal_axis = np.asarray(layout.normal_axis, dtype=np.float64)
    anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels], dtype=np.float64
    )
    if forced_panel_index is None:
        point_scan = points @ scan_axis
        insertion = np.searchsorted(anchors, point_scan)
        right = np.clip(insertion, 0, anchors.size - 1)
        left = np.clip(insertion - 1, 0, anchors.size - 1)
        choose_right = np.abs(anchors[right] - point_scan) < np.abs(
            anchors[left] - point_scan
        )
        panel_index = np.where(choose_right, right, left).astype(np.int32)
    else:
        if not 0 <= int(forced_panel_index) < anchors.size:
            raise ValueError("Forced object panel index is outside the layout")
        panel_index = np.full(
            points.shape[0], int(forced_panel_index), dtype=np.int32
        )
    centers = np.asarray(
        [panel.center_world_mm for panel in layout.panels], dtype=np.float64
    )
    offsets = np.asarray(
        [panel.canvas_offset_x for panel in layout.panels], dtype=np.float64
    )
    relative = points - centers[panel_index]
    q_scan = relative @ scan_axis
    q_down = relative @ down_axis
    q_normal = relative @ normal_axis
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (
            offsets[panel_index]
            + intrinsics.cx
            + intrinsics.fx * q_scan / q_normal
        )
        y = intrinsics.cy + intrinsics.fy * q_down / q_normal
    return x, y, q_normal, panel_index


def _rasterize_direct_triangle(
    *,
    map_x: np.ndarray,
    map_y: np.ndarray,
    target_depth: np.ndarray,
    target_xy: np.ndarray,
    source_xy: np.ndarray,
    source_depth: np.ndarray,
) -> int:
    x0 = max(0, int(math.floor(float(np.min(target_xy[:, 0])))))
    x1 = min(
        map_x.shape[1] - 1,
        int(math.ceil(float(np.max(target_xy[:, 0])))),
    )
    y0 = max(0, int(math.floor(float(np.min(target_xy[:, 1])))))
    y1 = min(
        map_x.shape[0] - 1,
        int(math.ceil(float(np.max(target_xy[:, 1])))),
    )
    if x1 < x0 or y1 < y0:
        return 0
    first, second, third = target_xy.astype(np.float64, copy=False)
    determinant = float(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )
    if not math.isfinite(determinant) or determinant <= 0.0:
        return 0
    yy, xx = np.indices((y1 - y0 + 1, x1 - x0 + 1), dtype=np.float64)
    xx += x0
    yy += y0
    weight_second = (
        (xx - first[0]) * (third[1] - first[1])
        - (yy - first[1]) * (third[0] - first[0])
    ) / determinant
    weight_third = (
        (second[0] - first[0]) * (yy - first[1])
        - (second[1] - first[1]) * (xx - first[0])
    ) / determinant
    weight_first = 1.0 - weight_second - weight_third
    inside = (
        (weight_first >= -1e-6)
        & (weight_second >= -1e-6)
        & (weight_third >= -1e-6)
    )
    depth = (
        weight_first * source_depth[0]
        + weight_second * source_depth[1]
        + weight_third * source_depth[2]
    )
    region_depth = target_depth[y0 : y1 + 1, x0 : x1 + 1]
    take = (
        inside
        & np.isfinite(depth)
        & (depth > 0.0)
        & (~np.isfinite(region_depth) | (depth < region_depth))
    )
    if not np.any(take):
        return 0
    candidate_x = (
        weight_first * source_xy[0, 0]
        + weight_second * source_xy[1, 0]
        + weight_third * source_xy[2, 0]
    )
    candidate_y = (
        weight_first * source_xy[0, 1]
        + weight_second * source_xy[1, 1]
        + weight_third * source_xy[2, 1]
    )
    map_x[y0 : y1 + 1, x0 : x1 + 1][take] = candidate_x[take]
    map_y[y0 : y1 + 1, x0 : x1 + 1][take] = candidate_y[take]
    region_depth[take] = depth[take].astype(np.float32)
    return int(np.count_nonzero(take))


def project_complete_object_owner_from_rgbd(
    *,
    source_image_bgr: np.ndarray,
    source_depth_mm: np.ndarray,
    source_reliable_depth: np.ndarray,
    source_object_mask: np.ndarray,
    camera_to_world: np.ndarray,
    layout: "InspectionMultiviewLayout",
    intrinsics: CameraIntrinsics,
    frame_id: int,
    panel_index: int,
    minimum_cells: int = 64,
    minimum_jacobian: float = 0.01,
    maximum_jacobian: float = 64.0,
) -> CompleteObjectOwner:
    """Directly project one complete measured RGB-D component.

    Unlike :func:`fit_complete_object_owner`, this path fits no display warp.
    Every accepted 2x2 source cell is transformed by the immutable SE(3) pose
    and rasterized with its real four depths.  Cells crossing a depth edge,
    object boundary, panel boundary, fold, or scale bound are rejected.
    """

    image = np.asarray(source_image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Source object RGB must be HxWx3 uint8")
    depth = np.asarray(source_depth_mm, dtype=np.float32)
    reliable = np.asarray(source_reliable_depth, dtype=bool)
    if depth.shape != image.shape[:2] or reliable.shape != depth.shape:
        raise ValueError("Source object RGB-D rasters are not aligned")
    object_mask = _checked_object_mask(source_object_mask, depth.shape)
    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Object owner pose must be a finite 4x4 matrix")
    target_shape = (int(layout.height), int(layout.width))
    map_x = np.full(target_shape, np.nan, dtype=np.float32)
    map_y = np.full(target_shape, np.nan, dtype=np.float32)
    target_depth = np.full(target_shape, np.inf, dtype=np.float32)

    object_y, object_x = np.nonzero(object_mask)
    source_x0 = max(0, int(np.min(object_x)) - 1)
    source_x1 = min(depth.shape[1] - 1, int(np.max(object_x)) + 1)
    source_y0 = max(0, int(np.min(object_y)) - 1)
    source_y1 = min(depth.shape[0] - 1, int(np.max(object_y)) + 1)
    rows, columns = np.indices(
        (source_y1 - source_y0 + 1, source_x1 - source_x0 + 1),
        dtype=np.int32,
    )
    source_columns = columns + source_x0
    source_rows = rows + source_y0
    local_depth = depth[source_y0 : source_y1 + 1, source_x0 : source_x1 + 1]
    camera = pinhole_unproject(
        source_columns.reshape(-1),
        source_rows.reshape(-1),
        local_depth.reshape(-1),
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    world = transform_points(camera, pose[:3, :3], pose[:3, 3])
    target_x, target_y, target_z, target_panel = (
        _project_world_points(
            world,
            layout,
            intrinsics,
            forced_panel_index=int(panel_index),
        )
    )
    grid_shape = source_columns.shape
    target_x = target_x.reshape(grid_shape)
    target_y = target_y.reshape(grid_shape)
    target_z = target_z.reshape(grid_shape)
    target_panel = target_panel.reshape(grid_shape)

    accepted_cells = 0
    rejected_object_boundary = 0
    rejected_invalid_depth = 0
    rejected_depth_edge = 0
    rejected_panel = 0
    rejected_jacobian = 0
    rasterized_pixels = 0
    minimum_accepted_jacobian = math.inf
    maximum_accepted_jacobian = 0.0
    local_mask = object_mask[
        source_y0 : source_y1 + 1, source_x0 : source_x1 + 1
    ]
    local_reliable = reliable[
        source_y0 : source_y1 + 1, source_x0 : source_x1 + 1
    ]
    for row in range(grid_shape[0] - 1):
        for column in range(grid_shape[1] - 1):
            corners = (
                (row, column),
                (row, column + 1),
                (row + 1, column + 1),
                (row + 1, column),
            )
            if not all(local_mask[index] for index in corners):
                rejected_object_boundary += 1
                continue
            if not all(local_reliable[index] for index in corners):
                rejected_invalid_depth += 1
                continue
            depths = np.asarray(
                [local_depth[index] for index in corners], dtype=np.float64
            )
            median_depth = float(np.median(depths))
            tolerance = max(20.0, 0.02 * median_depth)
            if (
                not np.isfinite(depths).all()
                or np.any(depths <= 0.0)
                or float(np.max(depths) - np.min(depths)) > tolerance
            ):
                rejected_depth_edge += 1
                continue
            if any(
                int(target_panel[index]) != int(panel_index)
                for index in corners
            ):
                rejected_panel += 1
                continue
            target = np.asarray(
                [(target_x[index], target_y[index]) for index in corners],
                dtype=np.float64,
            )
            z = np.asarray(
                [target_z[index] for index in corners], dtype=np.float64
            )
            if (
                not np.isfinite(target).all()
                or not np.isfinite(z).all()
                or np.any(z <= 0.0)
            ):
                rejected_invalid_depth += 1
                continue
            source = np.asarray(
                [
                    (source_x0 + column, source_y0 + row),
                    (source_x0 + column + 1, source_y0 + row),
                    (source_x0 + column + 1, source_y0 + row + 1),
                    (source_x0 + column, source_y0 + row + 1),
                ],
                dtype=np.float64,
            )
            triangle_indices = ((0, 1, 2), (0, 2, 3))
            triangle_payload: list[
                tuple[np.ndarray, np.ndarray, np.ndarray]
            ] = []
            jacobians: list[float] = []
            for indices in triangle_indices:
                triangle = target[list(indices)]
                first = triangle[1] - triangle[0]
                second = triangle[2] - triangle[0]
                jacobian = float(
                    first[0] * second[1] - first[1] * second[0]
                )
                jacobians.append(jacobian)
                triangle_payload.append(
                    (
                        triangle,
                        source[list(indices)],
                        z[list(indices)],
                    )
                )
            if any(
                not math.isfinite(value)
                or value < minimum_jacobian
                or value > maximum_jacobian
                for value in jacobians
            ):
                rejected_jacobian += 1
                continue
            accepted_cells += 1
            minimum_accepted_jacobian = min(
                minimum_accepted_jacobian, *jacobians
            )
            maximum_accepted_jacobian = max(
                maximum_accepted_jacobian, *jacobians
            )
            for triangle, source_triangle, triangle_depth in triangle_payload:
                rasterized_pixels += _rasterize_direct_triangle(
                    map_x=map_x,
                    map_y=map_y,
                    target_depth=target_depth,
                    target_xy=triangle,
                    source_xy=source_triangle,
                    source_depth=triangle_depth,
                )
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(target_depth)
        & (map_x >= 0.0)
        & (map_x <= image.shape[1] - 1)
        & (map_y >= 0.0)
        & (map_y <= image.shape[0] - 1)
    )
    if accepted_cells < minimum_cells or not np.any(valid):
        raise RuntimeError(
            "Object direct RGB-D projection lacks complete surface support"
        )
    target_image = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return CompleteObjectOwner(
        frame_id=int(frame_id),
        panel_index=int(panel_index),
        source_to_canvas=np.empty((0, 0), dtype=np.float64),
        target_mask=np.ascontiguousarray(valid),
        target_image_bgr=np.ascontiguousarray(target_image),
        audit={
            "policy": (
                "one_measured_rgb_owner_direct_dense_rgbd_se3_"
                "piecewise_inverse_sampling"
            ),
            "frame_id": int(frame_id),
            "panel_index": int(panel_index),
            "source_object_pixel_count": int(np.count_nonzero(object_mask)),
            "candidate_cell_count": int(
                (grid_shape[0] - 1) * (grid_shape[1] - 1)
            ),
            "accepted_cell_count": accepted_cells,
            "rejected_object_boundary_cell_count": (
                rejected_object_boundary
            ),
            "rejected_invalid_depth_cell_count": rejected_invalid_depth,
            "rejected_depth_edge_cell_count": rejected_depth_edge,
            "rejected_other_panel_cell_count": rejected_panel,
            "rejected_jacobian_cell_count": rejected_jacobian,
            "minimum_accepted_jacobian": (
                minimum_accepted_jacobian
            ),
            "maximum_accepted_jacobian": (
                maximum_accepted_jacobian
            ),
            "rasterized_pixel_update_count": rasterized_pixels,
            "target_pixel_count": int(np.count_nonzero(valid)),
            "direct_world_projection": True,
            "fitted_display_warp": False,
            "rgb_generated": False,
            "pose_modified": False,
            "blend_used": False,
        },
    )


def _checked_object_mask(mask: np.ndarray, source_shape: tuple[int, int]) -> np.ndarray:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != source_shape:
        raise ValueError("Object mask does not match the source RGB")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), 8
    )
    components = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
        if int(stats[label, cv2.CC_STAT_AREA]) > 0
    ]
    if len(components) != 1:
        raise ValueError("Object mask must contain exactly one component")
    area, label = components[0]
    if area < 64:
        raise ValueError("Object mask is too small for a complete-owner handoff")
    component = labels == label
    if (
        np.any(component[0])
        or np.any(component[-1])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    ):
        raise ValueError("Object mask touches a source-image boundary")
    return np.ascontiguousarray(component)


def _automatic_source_mask_from_mesh_seed(
    *,
    source: ObjectHandoffSource,
    target_component_mask: np.ndarray,
    minimum_seed_pixels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    component = np.asarray(target_component_mask, dtype=bool)
    local_width = int(source.mesh_valid_mask.shape[1])
    x0 = int(source.mesh_corner_x)
    x1 = x0 + local_width
    if (
        source.mesh_map_x.shape != source.mesh_map_y.shape
        or source.mesh_map_x.shape != source.mesh_valid_mask.shape
        or source.mesh_map_x.shape != source.mesh_relative_depth_mm.shape
        or source.mesh_map_x.shape[0] != component.shape[0]
        or x0 < 0
        or x1 > component.shape[1]
    ):
        raise ValueError("Automatic object source mesh is not canvas-aligned")
    local_component = component[:, x0:x1]
    support = local_component & np.asarray(
        source.mesh_valid_mask, dtype=bool
    )
    target_y, target_x = np.nonzero(support)
    if target_x.size < minimum_seed_pixels:
        raise RuntimeError("Automatic object source lacks RGB-D mesh support")
    source_x = np.rint(source.mesh_map_x[target_y, target_x]).astype(
        np.int32
    )
    source_y = np.rint(source.mesh_map_y[target_y, target_x]).astype(
        np.int32
    )
    inside = (
        (source_x >= 0)
        & (source_x < source.image_bgr.shape[1])
        & (source_y >= 0)
        & (source_y < source.image_bgr.shape[0])
    )
    source_x = source_x[inside]
    source_y = source_y[inside]
    if source_x.size < minimum_seed_pixels:
        raise RuntimeError("Automatic object source seed leaves the RGB frame")
    seed = np.zeros(source.depth_mm.shape, dtype=np.uint8)
    seed[source_y, source_x] = 1
    seed = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    seed_y, seed_x = np.nonzero(seed)
    raw_x0, raw_x1 = int(np.min(seed_x)), int(np.max(seed_x)) + 1
    raw_y0, raw_y1 = int(np.min(seed_y)), int(np.max(seed_y)) + 1
    seed_width = raw_x1 - raw_x0
    seed_height = raw_y1 - raw_y0
    margin_x = max(12, int(math.ceil(0.20 * seed_width)))
    margin_y = max(12, int(math.ceil(0.25 * seed_height)))
    box_x0 = max(1, raw_x0 - margin_x)
    box_x1 = min(source.depth_mm.shape[1] - 1, raw_x1 + margin_x)
    box_y0 = max(1, raw_y0 - margin_y)
    box_y1 = min(source.depth_mm.shape[0] - 1, raw_y1 + margin_y)
    if box_x1 - box_x0 < 8 or box_y1 - box_y0 < 8:
        raise RuntimeError("Automatic object source seed box is degenerate")

    grabcut_roi = np.full(
        (box_y1 - box_y0, box_x1 - box_x0),
        cv2.GC_PR_BGD,
        dtype=np.uint8,
    )
    seed_depth = source.depth_mm[
        source_y, source_x
    ].astype(np.float32)
    seed_depth = seed_depth[
        np.isfinite(seed_depth) & (seed_depth > 0.0)
    ]
    if seed_depth.size < minimum_seed_pixels:
        raise RuntimeError("Automatic object source seed has invalid depth")
    median_depth = float(np.median(seed_depth))
    depth_tolerance = max(35.0, 0.08 * median_depth)
    probable = (
        np.asarray(source.reliable_depth, dtype=bool)
        & (np.abs(source.depth_mm - median_depth) <= depth_tolerance)
    )
    probable_roi = probable[box_y0:box_y1, box_x0:box_x1]
    grabcut_roi[probable_roi] = cv2.GC_PR_FGD
    seed_roi = seed[box_y0:box_y1, box_x0:box_x1] > 0
    grabcut_roi[seed_roi] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.setRNGSeed(0)
    cv2.grabCut(
        np.ascontiguousarray(
            source.image_bgr[box_y0:box_y1, box_x0:box_x1]
        ),
        grabcut_roi,
        None,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )
    candidate = np.zeros(source.depth_mm.shape, dtype=bool)
    candidate[box_y0:box_y1, box_x0:box_x1] = (
        (grabcut_roi == cv2.GC_FGD)
        | (grabcut_roi == cv2.GC_PR_FGD)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    best_label = 0
    best_seed_support = 0
    for label in range(1, count):
        selected = labels == label
        seed_support = int(np.count_nonzero(selected & (seed > 0)))
        if seed_support > best_seed_support:
            best_label = label
            best_seed_support = seed_support
    if best_label == 0 or best_seed_support < minimum_seed_pixels:
        raise RuntimeError("Automatic object GrabCut lost its RGB-D seed")
    mask = labels == best_label
    mask = cv2.morphologyEx(
        mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    mask = _checked_object_mask(mask, source.depth_mm.shape)
    seed_recall = float(
        np.count_nonzero(mask & (seed > 0))
        / max(1, np.count_nonzero(seed))
    )
    depth_coverage = float(
        np.count_nonzero(mask & source.reliable_depth)
        / max(1, np.count_nonzero(mask))
    )
    if seed_recall < 0.90 or depth_coverage < 0.95:
        raise RuntimeError(
            "Automatic object source mask is incomplete or depth-unreliable"
        )
    mask_y, mask_x = np.nonzero(mask)
    return np.ascontiguousarray(mask), {
        "frame_id": int(source.frame_id),
        "source_panel_index": int(source.panel_index),
        "target_mesh_support_pixel_count": int(target_x.size),
        "source_seed_pixel_count": int(np.count_nonzero(seed)),
        "source_seed_recall": seed_recall,
        "source_depth_coverage_ratio": depth_coverage,
        "source_bbox_xywh": [
            int(np.min(mask_x)),
            int(np.min(mask_y)),
            int(np.max(mask_x) - np.min(mask_x) + 1),
            int(np.max(mask_y) - np.min(mask_y) + 1),
        ],
        "manual_bbox_used": False,
        "manual_frame_id_used": False,
    }


def select_automatic_complete_object_owner(
    *,
    target_component_mask: np.ndarray,
    baseline_owner_frame_id: np.ndarray,
    sources: Sequence[ObjectHandoffSource],
    target_panel_index: int,
    layout: "InspectionMultiviewLayout",
    intrinsics: CameraIntrinsics,
    minimum_seed_pixels: int = 80,
    minimum_component_coverage_ratio: float = 0.20,
    minimum_target_component_recall: float = 0.90,
    minimum_cross_view_iou: float = 0.30,
    minimum_selected_union_coverage_ratio: float = 0.90,
) -> AutomaticObjectHandoff:
    """Select one complete owner from an existing multi-owner RGB-D component.

    Discovery starts only from the renderer's measured target component.  No
    frame ID or source bounding box is supplied.  At least two independently
    renderable sources must agree after direct world projection.
    """

    component = np.asarray(target_component_mask, dtype=bool)
    baseline_owner = np.asarray(baseline_owner_frame_id, dtype=np.int32)
    if component.shape != baseline_owner.shape or not np.any(component):
        raise ValueError("Automatic object target component is invalid")
    baseline_owners = np.unique(baseline_owner[component])
    baseline_owners = baseline_owners[baseline_owners >= 0]
    if baseline_owners.size < 2:
        raise RuntimeError(
            "Automatic object handoff only accepts baseline multi-owner "
            "components"
        )
    component_area = int(np.count_nonzero(component))
    minimum_support = max(
        int(minimum_seed_pixels),
        int(math.ceil(minimum_component_coverage_ratio * component_area)),
    )
    accepted: list[
        tuple[
            CompleteObjectOwner,
            np.ndarray,
            dict[str, object],
        ]
    ] = []
    rejected: list[dict[str, object]] = []
    for source in sources:
        try:
            source_mask, source_audit = _automatic_source_mask_from_mesh_seed(
                source=source,
                target_component_mask=component,
                minimum_seed_pixels=minimum_support,
            )
            owner = project_complete_object_owner_from_rgbd(
                source_image_bgr=source.image_bgr,
                source_depth_mm=source.depth_mm,
                source_reliable_depth=source.reliable_depth,
                source_object_mask=source_mask,
                camera_to_world=source.camera_to_world,
                layout=layout,
                intrinsics=intrinsics,
                frame_id=int(source.frame_id),
                panel_index=int(target_panel_index),
                minimum_cells=max(64, minimum_support),
            )
            intersection = int(
                np.count_nonzero(owner.target_mask & component)
            )
            union = int(np.count_nonzero(owner.target_mask | component))
            component_iou = float(intersection / union) if union else 0.0
            component_recall = float(intersection / component_area)
            if (
                component_iou < 0.15
                or component_recall < minimum_target_component_recall
            ):
                raise RuntimeError(
                    "Automatic object direct projection disagrees with "
                    "the measured target component"
                )
            accepted.append(
                (
                    owner,
                    source_mask,
                    {
                        **source_audit,
                        "target_component_iou": component_iou,
                        "target_component_recall": component_recall,
                    },
                )
            )
        except (RuntimeError, ValueError) as exc:
            rejected.append(
                {
                    "frame_id": int(source.frame_id),
                    "source_panel_index": int(source.panel_index),
                    "reason": str(exc),
                }
            )
    if len(accepted) < 2:
        raise AutomaticObjectHandoffRejected(
            "Automatic object component lacks two complete RGB-D owners",
            {
                "accepted_source_count": len(accepted),
                "rejected_source_count": len(rejected),
                "rejected_sources": rejected,
            },
        )
    mutually_consistent: list[tuple[int, int, float]] = []
    support_count = np.ones(len(accepted), dtype=np.int32)
    for first_index, (first, _, _) in enumerate(accepted):
        for second_index in range(first_index + 1, len(accepted)):
            second = accepted[second_index][0]
            intersection = int(
                np.count_nonzero(first.target_mask & second.target_mask)
            )
            union = int(
                np.count_nonzero(first.target_mask | second.target_mask)
            )
            iou = float(intersection / union) if union else 0.0
            mutually_consistent.append(
                (int(first.frame_id), int(second.frame_id), iou)
            )
            if iou >= minimum_cross_view_iou:
                support_count[first_index] += 1
                support_count[second_index] += 1
    consistent_indices = np.flatnonzero(support_count >= 2)
    if consistent_indices.size < 2:
        raise AutomaticObjectHandoffRejected(
            "Automatic object direct projections are not cross-view consistent",
            {
                "accepted_source_count": len(accepted),
                "rejected_sources": rejected,
                "cross_view_target_mask_ious": [
                    {
                        "first_frame_id": first,
                        "second_frame_id": second,
                        "iou": iou,
                    }
                    for first, second, iou in mutually_consistent
                ],
            },
        )
    union_coverage_by_index: dict[int, tuple[float, tuple[int, ...]]] = {}
    for index in (int(value) for value in consistent_indices):
        selected_target = accepted[index][0].target_mask
        cluster_indices = [index]
        for other_index, (other, _, _) in enumerate(accepted):
            if other_index == index:
                continue
            intersection = int(
                np.count_nonzero(selected_target & other.target_mask)
            )
            union = int(
                np.count_nonzero(selected_target | other.target_mask)
            )
            if union and intersection / union >= minimum_cross_view_iou:
                cluster_indices.append(other_index)
        footprint_union = np.logical_or.reduce(
            [accepted[value][0].target_mask for value in cluster_indices]
        )
        union_coverage = float(
            np.count_nonzero(selected_target & footprint_union)
            / max(1, np.count_nonzero(footprint_union))
        )
        union_coverage_by_index[index] = (
            union_coverage,
            tuple(cluster_indices),
        )
    complete_indices = [
        index
        for index, (coverage, _) in union_coverage_by_index.items()
        if coverage >= minimum_selected_union_coverage_ratio
    ]
    if not complete_indices:
        raise AutomaticObjectHandoffRejected(
            "No automatic RGB owner covers the cross-view footprint union",
            {
                "accepted_source_count": len(accepted),
                "rejected_sources": rejected,
                "candidate_union_coverage_ratios": [
                    {
                        "frame_id": int(accepted[index][0].frame_id),
                        "coverage_ratio": float(value[0]),
                    }
                    for index, value in union_coverage_by_index.items()
                ],
            },
        )
    selected_index = max(
        complete_indices,
        key=lambda index: (
            union_coverage_by_index[index][0],
            accepted[index][2]["target_component_iou"],
            accepted[index][2]["source_depth_coverage_ratio"],
            accepted[index][0].audit["target_pixel_count"],
            -accepted[index][0].frame_id,
        ),
    )
    selected_owner, selected_mask, selected_audit = accepted[selected_index]
    selected_union_coverage, selected_cluster_indices = (
        union_coverage_by_index[selected_index]
    )
    return AutomaticObjectHandoff(
        owner=selected_owner,
        source_object_mask=selected_mask,
        audit={
            "policy": (
                "existing_multi_owner_rgbd_component_automatic_mesh_seed_"
                "grabcut_two_view_world_projection_one_rgb_owner"
            ),
            "baseline_owner_frame_ids": [
                int(value) for value in baseline_owners
            ],
            "target_component_pixel_count": component_area,
            "source_candidate_count": len(sources),
            "accepted_source_count": len(accepted),
            "rejected_source_count": len(rejected),
            "rejected_sources": rejected,
            "accepted_sources": [item[2] for item in accepted],
            "cross_view_target_mask_ious": [
                {
                    "first_frame_id": first,
                    "second_frame_id": second,
                    "iou": iou,
                }
                for first, second, iou in mutually_consistent
            ],
            "selected_frame_id": int(selected_owner.frame_id),
            "selected_cross_view_source_frame_ids": [
                int(accepted[index][0].frame_id)
                for index in selected_cluster_indices
            ],
            "selected_cross_view_footprint_union_coverage_ratio": (
                selected_union_coverage
            ),
            "minimum_target_component_recall": float(
                minimum_target_component_recall
            ),
            "minimum_cross_view_iou": float(minimum_cross_view_iou),
            "minimum_selected_union_coverage_ratio": float(
                minimum_selected_union_coverage_ratio
            ),
            "selected_source_audit": selected_audit,
            "manual_bbox_used": False,
            "manual_frame_id_used": False,
            "silent_fallback_allowed": False,
        },
    )


def fit_complete_object_owner(
    *,
    source_image_bgr: np.ndarray,
    source_object_mask: np.ndarray,
    mesh_map_x: np.ndarray,
    mesh_map_y: np.ndarray,
    mesh_valid_mask: np.ndarray,
    target_corner_x: int,
    target_shape: tuple[int, int],
    frame_id: int,
    panel_index: int,
    minimum_correspondences: int = 80,
    maximum_held_out_p95_pixels: float = 2.0,
    maximum_held_out_error_pixels: float = 4.0,
) -> CompleteObjectOwner:
    """Fit and audit one local object sampling transform.

    ``mesh_map_x`` and ``mesh_map_y`` are target-to-source coordinates made
    from real RGB-D and immutable camera poses.  A deterministic 20% held-out
    subset is never used by ``estimateAffinePartial2D``.
    """

    image = np.asarray(source_image_bgr)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Source object RGB must be HxWx3 uint8")
    source_mask = _checked_object_mask(
        source_object_mask, image.shape[:2]
    )
    map_x = np.asarray(mesh_map_x, dtype=np.float32)
    map_y = np.asarray(mesh_map_y, dtype=np.float32)
    mesh_valid = np.asarray(mesh_valid_mask, dtype=bool)
    if map_x.shape != map_y.shape or map_x.shape != mesh_valid.shape:
        raise ValueError("Object RGB-D mesh rasters are not aligned")
    target_height, target_width = (int(value) for value in target_shape)
    if (
        target_height <= 0
        or target_width <= 0
        or target_corner_x < 0
        or target_corner_x + map_x.shape[1] > target_width
        or map_x.shape[0] != target_height
    ):
        raise ValueError("Object RGB-D mesh is outside the target canvas")

    nearest_x = np.zeros(map_x.shape, dtype=np.int32)
    nearest_y = np.zeros(map_y.shape, dtype=np.int32)
    finite_maps = np.isfinite(map_x) & np.isfinite(map_y)
    nearest_x[finite_maps] = np.rint(map_x[finite_maps]).astype(np.int32)
    nearest_y[finite_maps] = np.rint(map_y[finite_maps]).astype(np.int32)
    in_source = (
        mesh_valid
        & finite_maps
        & (nearest_x >= 0)
        & (nearest_x < image.shape[1])
        & (nearest_y >= 0)
        & (nearest_y < image.shape[0])
    )
    evidence = np.zeros(mesh_valid.shape, dtype=bool)
    yy, xx = np.nonzero(in_source)
    evidence[yy, xx] = source_mask[nearest_y[yy, xx], nearest_x[yy, xx]]
    target_y, target_x_local = np.nonzero(evidence)
    if target_x_local.size < minimum_correspondences:
        raise RuntimeError(
            "Object handoff lacks enough measured RGB-D correspondences"
        )
    source_xy = np.column_stack(
        (
            map_x[target_y, target_x_local],
            map_y[target_y, target_x_local],
        )
    ).astype(np.float32)
    target_xy = np.column_stack(
        (
            target_x_local.astype(np.float32) + float(target_corner_x),
            target_y.astype(np.float32),
        )
    )

    # Stable spatial hashing prevents the held-out pixels from leaking into
    # the fit while retaining samples across the whole component.
    held_out = (
        (
            np.rint(source_xy[:, 0]).astype(np.int64) * 17
            + np.rint(source_xy[:, 1]).astype(np.int64) * 31
        )
        % 5
    ) == 0
    if (
        np.count_nonzero(held_out) < max(16, minimum_correspondences // 5)
        or np.count_nonzero(~held_out) < minimum_correspondences
    ):
        raise RuntimeError("Object handoff held-out split is incomplete")
    affine, inliers = cv2.estimateAffinePartial2D(
        source_xy[~held_out],
        target_xy[~held_out],
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0,
        maxIters=4000,
        confidence=0.999,
        refineIters=30,
    )
    if affine is None or inliers is None or not np.isfinite(affine).all():
        raise RuntimeError("Object handoff local transform did not converge")
    linear = np.asarray(affine[:, :2], dtype=np.float64)
    determinant = float(np.linalg.det(linear))
    scale = float(
        math.sqrt(float(affine[0, 0]) ** 2 + float(affine[1, 0]) ** 2)
    )
    rotation_degrees = math.degrees(
        math.atan2(float(affine[1, 0]), float(affine[0, 0]))
    )
    training_inlier_ratio = float(np.mean(inliers.reshape(-1) > 0))
    if (
        determinant <= 0.0
        or not 0.50 <= scale <= 2.0
        or abs(rotation_degrees) > 25.0
        or training_inlier_ratio < 0.70
    ):
        raise RuntimeError("Object handoff local-transform geometry failed")

    held_prediction = cv2.transform(
        source_xy[held_out][None, :, :], affine
    )[0]
    held_error = np.linalg.norm(
        held_prediction - target_xy[held_out], axis=1
    )
    held_p95 = float(np.percentile(held_error, 95))
    held_maximum = float(np.max(held_error))
    if (
        held_p95 > maximum_held_out_p95_pixels
        or held_maximum > maximum_held_out_error_pixels
    ):
        raise RuntimeError("Object handoff held-out reprojection failed")

    target_mask = cv2.warpAffine(
        source_mask.astype(np.uint8),
        affine,
        (target_width, target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    target_image = cv2.warpAffine(
        image,
        affine,
        (target_width, target_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if not np.any(target_mask):
        raise RuntimeError("Object handoff produced no target RGB footprint")
    return CompleteObjectOwner(
        frame_id=int(frame_id),
        panel_index=int(panel_index),
        source_to_canvas=np.asarray(affine, dtype=np.float64),
        target_mask=np.ascontiguousarray(target_mask),
        target_image_bgr=np.ascontiguousarray(target_image),
        audit={
            "policy": (
                "one_measured_rgb_owner_local_similarity_from_rgbd_mesh_"
                "with_excluded_held_out"
            ),
            "frame_id": int(frame_id),
            "panel_index": int(panel_index),
            "source_object_pixel_count": int(np.count_nonzero(source_mask)),
            "measured_correspondence_count": int(source_xy.shape[0]),
            "training_correspondence_count": int(
                np.count_nonzero(~held_out)
            ),
            "held_out_correspondence_count": int(
                np.count_nonzero(held_out)
            ),
            "training_inlier_ratio": training_inlier_ratio,
            "held_out_p95_pixels": held_p95,
            "held_out_maximum_pixels": held_maximum,
            "determinant": determinant,
            "scale": scale,
            "rotation_degrees": rotation_degrees,
            "target_pixel_count": int(np.count_nonzero(target_mask)),
            "rgb_generated": False,
            "pose_modified": False,
            "blend_used": False,
        },
    )


def build_object_owner_interval(
    *,
    panel_index: int,
    view_dependent_footprints: Sequence[np.ndarray],
    selected_panel_valid_mask: np.ndarray,
    protected_mask: np.ndarray | None = None,
    horizontal_guard_pixels: int = 4,
    vertical_guard_pixels: int = 2,
) -> ObjectOwnerInterval:
    """Cover every shifted copy with one row-contiguous panel owner.

    The lock spans from the leftmost to the rightmost observed footprint on
    each affected row.  This is the constraint needed by an adjacent seam
    chain: neighbouring panels may be skipped on those rows, but the selected
    owner can never become a disconnected post-composition island.
    """

    if not view_dependent_footprints:
        raise ValueError("Object handoff needs at least one reference footprint")
    footprints = [np.asarray(value, dtype=bool) for value in view_dependent_footprints]
    shape = footprints[0].shape
    if any(value.shape != shape for value in footprints):
        raise ValueError("Object reference footprints are not aligned")
    selected_valid = np.asarray(selected_panel_valid_mask, dtype=bool)
    if selected_valid.shape != shape:
        raise ValueError("Selected panel validity is not canvas-aligned")
    union = np.logical_or.reduce(footprints)
    if not np.any(union):
        raise ValueError("Object reference footprints are empty")
    if horizontal_guard_pixels < 0 or vertical_guard_pixels < 0:
        raise ValueError("Object owner guards must be non-negative")
    if horizontal_guard_pixels or vertical_guard_pixels:
        kernel = np.ones(
            (
                2 * int(vertical_guard_pixels) + 1,
                2 * int(horizontal_guard_pixels) + 1,
            ),
            dtype=np.uint8,
        )
        union = cv2.dilate(union.astype(np.uint8), kernel).astype(bool)

    lock = np.zeros(shape, dtype=bool)
    affected_rows = np.flatnonzero(np.any(union, axis=1))
    for row in affected_rows:
        columns = np.flatnonzero(union[row])
        lock[row, columns[0] : columns[-1] + 1] = True
    missing_coverage = lock & ~selected_valid
    if np.any(missing_coverage):
        raise RuntimeError(
            "Selected object owner does not cover the complete handoff interval"
        )
    protected = (
        np.zeros(shape, dtype=bool)
        if protected_mask is None
        else np.asarray(protected_mask, dtype=bool)
    )
    if protected.shape != shape:
        raise ValueError("Object handoff protection mask is not aligned")
    protected_intersection = lock & protected
    if np.any(protected_intersection):
        raise RuntimeError("Object handoff interval crosses a protected region")

    return ObjectOwnerInterval(
        panel_index=int(panel_index),
        lock_mask=np.ascontiguousarray(lock),
        union_footprint=np.ascontiguousarray(union),
        audit={
            "policy": (
                "all_reference_copies_one_row_contiguous_panel_owner_"
                "before_adjacent_seam_solve"
            ),
            "panel_index": int(panel_index),
            "observation_footprint_count": len(footprints),
            "union_footprint_pixel_count": int(np.count_nonzero(union)),
            "lock_pixel_count": int(np.count_nonzero(lock)),
            "affected_row_count": int(affected_rows.size),
            "selected_owner_missing_coverage_pixel_count": 0,
            "protected_intersection_pixel_count": 0,
            "row_contiguous": True,
            "single_owner": True,
        },
    )


__all__ = [
    "AutomaticObjectHandoff",
    "AutomaticObjectHandoffRejected",
    "CompleteObjectOwner",
    "ObjectHandoffSource",
    "ObjectOwnerInterval",
    "build_object_owner_interval",
    "fit_complete_object_owner",
    "project_complete_object_owner_from_rgbd",
    "select_automatic_complete_object_owner",
]
