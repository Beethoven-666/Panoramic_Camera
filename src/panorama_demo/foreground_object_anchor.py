"""Object-level RGB-D anchoring for the inspection mosaic.

Near objects cannot be composited on the virtual reference plane: doing so
moves them by a view-dependent parallax amount and lets a background seam cut
away part of the object.  This module detects compact near-depth components,
associates repeated observations in world space, chooses one complete RGB
owner, and maps that owner to the inspection canvas with a constrained local
similarity transform fitted from real RGB-D/SE(3) correspondences.

The transform is display-only.  It never changes a camera pose, generates
colour, or feeds the metric/TSDF products.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from .cuda_backend import pinhole_unproject, transform_points
from .session import CameraIntrinsics


@dataclass(frozen=True)
class ForegroundAnchorSource:
    source_index: int
    panel_index: int
    frame_id: int
    image_bgr: np.ndarray
    depth_mm: np.ndarray
    reliable_depth: np.ndarray
    camera_to_world: np.ndarray
    reference_map_x: np.ndarray
    reference_map_y: np.ndarray


@dataclass(frozen=True)
class ForegroundObjectObservation:
    observation_id: int
    source_index: int
    panel_index: int
    frame_id: int
    source_mask: np.ndarray
    source_to_canvas: np.ndarray
    world_centroid_mm: tuple[float, float, float]
    median_normal_depth_mm: float
    target_bbox_xywh: tuple[int, int, int, int]
    source_pixel_count: int
    target_pixel_count: int
    fit_inlier_ratio: float
    fit_rmse_pixels: float
    score: float


@dataclass(frozen=True)
class ForegroundObjectAnchor:
    track_id: int
    selected: ForegroundObjectObservation
    observation_ids: tuple[int, ...]
    world_support_source_indices: tuple[int, ...]


@dataclass(frozen=True)
class ForegroundObjectAnchorPlan:
    anchors: tuple[ForegroundObjectAnchor, ...]
    observations: tuple[ForegroundObjectObservation, ...]
    background_exclusion_masks: tuple[np.ndarray, ...]
    target_mask: np.ndarray
    audit: dict[str, object]


@dataclass(frozen=True)
class ForegroundObjectOverlayResult:
    component_label: np.ndarray
    visible_mask: np.ndarray
    audit: dict[str, object]


def _project_world_to_canvas(
    points_world_mm: np.ndarray,
    layout: object,
    intrinsics: CameraIntrinsics,
    *,
    forced_panel_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world_mm, dtype=np.float64)
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    down_axis = np.asarray(layout.down_axis, dtype=np.float64)
    normal_axis = np.asarray(layout.normal_axis, dtype=np.float64)
    anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels], dtype=np.float64
    )
    scan = points @ scan_axis
    if forced_panel_index is None:
        insertion = np.searchsorted(anchors, scan)
        right = np.clip(insertion, 0, len(anchors) - 1)
        left = np.clip(insertion - 1, 0, len(anchors) - 1)
        choose_right = np.abs(anchors[right] - scan) < np.abs(
            anchors[left] - scan
        )
        panel_index = np.where(choose_right, right, left)
    else:
        selected_panel = int(forced_panel_index)
        if not 0 <= selected_panel < len(anchors):
            raise ValueError(
                "Foreground object anchor panel index is outside layout"
            )
        panel_index = np.full(
            points.shape[0], selected_panel, dtype=np.int32
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
    return x, y, q_normal


def _component_masks(
    depth_mm: np.ndarray,
    reliable_depth: np.ndarray,
    reference_depth_mm: float,
    *,
    minimum_pixels: int,
) -> list[np.ndarray]:
    depth = np.asarray(depth_mm, dtype=np.float32)
    reliable = np.asarray(reliable_depth, dtype=bool)
    margin = max(35.0, 0.04 * float(reference_depth_mm))
    near = reliable & (depth < np.float32(reference_depth_mm - margin))

    # Remove depth discontinuities before connected components so a shelf or
    # wall does not bridge otherwise independent objects.
    sentinel = np.float32(reference_depth_mm + 1000.0)
    local_max = cv2.dilate(np.where(reliable, depth, 0.0), np.ones((3, 3), np.uint8))
    local_min = cv2.erode(
        np.where(reliable, depth, sentinel), np.ones((3, 3), np.uint8)
    )
    tolerance = np.maximum(np.float32(20.0), np.float32(0.02) * depth)
    edge = reliable & ((local_max - local_min) > tolerance)
    near &= ~cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
        bool
    )
    near = cv2.morphologyEx(
        near.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(near, 8)
    image_area = int(depth.size)
    results: list[np.ndarray] = []
    for label in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[label].tolist()
        )
        if area < minimum_pixels or area > int(0.16 * image_area):
            continue
        if width < 6 or height < 6:
            continue
        if height > int(0.88 * depth.shape[0]) and width < int(
            0.18 * depth.shape[1]
        ):
            continue
        if width > int(0.88 * depth.shape[1]) and height < int(
            0.18 * depth.shape[0]
        ):
            continue
        component = (labels == label).astype(np.uint8)
        # Close small depth holes on dark/reflective object interiors, while
        # keeping the component's measured external silhouette.
        roi = component[y : y + height, x : x + width]
        contours, _ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(roi)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        hole_count = int(np.count_nonzero(filled & ~roi))
        if hole_count <= max(64, int(0.25 * area)):
            component[y : y + height, x : x + width] = filled
        results.append(component.astype(bool))
    return results


def _component_masks_excluding_world_planes(
    source: ForegroundAnchorSource,
    reference_depth_mm: float,
    intrinsics: CameraIntrinsics,
    structural_plane_models_world: Sequence[np.ndarray],
    *,
    minimum_pixels: int,
) -> list[np.ndarray]:
    """Keep measured object silhouettes after removing fitted shelf/wall planes.

    The old image-only depth-edge split fragmented the real breakers and
    chargers into small surface patches.  Here the immutable RGB-D pose first
    identifies pixels belonging to an audited large world plane.  Connected
    components are then formed from the remaining measured near surface, so
    an object's own depth variation is not mistaken for an object boundary.
    """

    depth = np.asarray(source.depth_mm, dtype=np.float32)
    reliable = np.asarray(source.reliable_depth, dtype=bool)
    margin = max(35.0, 0.04 * float(reference_depth_mm))
    near = (
        reliable
        & np.isfinite(depth)
        & (depth < np.float32(float(reference_depth_mm) - margin))
    )
    yy, xx = np.nonzero(near)
    structural = np.zeros(depth.shape, dtype=bool)
    if xx.size and structural_plane_models_world:
        camera = pinhole_unproject(
            xx,
            yy,
            depth[yy, xx],
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
        )
        pose = np.asarray(source.camera_to_world, dtype=np.float64)
        world = transform_points(camera, pose[:3, :3], pose[:3, 3])
        structural_samples = np.zeros(xx.size, dtype=bool)
        for model in structural_plane_models_world:
            plane = np.asarray(model, dtype=np.float64)
            if plane.shape != (4,) or not np.isfinite(plane).all():
                raise RuntimeError(
                    "Foreground structural plane model is invalid"
                )
            normal_length = float(np.linalg.norm(plane[:3]))
            if normal_length <= 0.0:
                raise RuntimeError(
                    "Foreground structural plane has no finite normal"
                )
            distance = np.abs(
                world @ plane[:3] + plane[3]
            ) / normal_length
            structural_samples |= distance <= 14.0
        structural[yy[structural_samples], xx[structural_samples]] = True
    structural_guard = cv2.dilate(
        structural.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    candidate = near & ~structural_guard
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, 8
    )
    image_area = int(depth.size)
    results: list[np.ndarray] = []
    for label in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[label].tolist()
        )
        if area < minimum_pixels or area > int(0.16 * image_area):
            continue
        if width < 6 or height < 6:
            continue
        if height > int(0.88 * depth.shape[0]) and width < int(
            0.18 * depth.shape[1]
        ):
            continue
        if width > int(0.88 * depth.shape[1]) and height < int(
            0.18 * depth.shape[0]
        ):
            continue
        component = (labels == label).astype(np.uint8)
        roi = component[y : y + height, x : x + width]
        contours, _ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filled = np.zeros_like(roi)
        cv2.drawContours(
            filled, contours, -1, 1, thickness=cv2.FILLED
        )
        hole_count = int(np.count_nonzero(filled & ~roi))
        if hole_count <= max(64, int(0.25 * area)):
            component[y : y + height, x : x + width] = filled
        results.append(component.astype(bool))
    return results


def _fit_observation(
    *,
    observation_id: int,
    source: ForegroundAnchorSource,
    source_mask: np.ndarray,
    fit_mask: np.ndarray | None = None,
    layout: object,
    intrinsics: CameraIntrinsics,
    target_panel_index: int,
) -> ForegroundObjectObservation | None:
    yy, xx = np.nonzero(
        (source_mask if fit_mask is None else fit_mask)
        & source.reliable_depth
    )
    if xx.size < 32:
        return None
    stride = max(1, int(math.ceil(xx.size / 4000)))
    xx_sample = xx[::stride].astype(np.float64)
    yy_sample = yy[::stride].astype(np.float64)
    depth_sample = source.depth_mm[yy[::stride], xx[::stride]].astype(np.float64)
    points_camera = pinhole_unproject(
        xx_sample,
        yy_sample,
        depth_sample,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    pose = np.asarray(source.camera_to_world, dtype=np.float64)
    points_world = transform_points(
        points_camera, pose[:3, :3], pose[:3, 3]
    )
    target_x, target_y, target_normal = _project_world_to_canvas(
        points_world,
        layout,
        intrinsics,
        forced_panel_index=int(target_panel_index),
    )
    finite = (
        np.isfinite(target_x)
        & np.isfinite(target_y)
        & np.isfinite(target_normal)
        & (target_normal > 0.0)
    )
    if np.count_nonzero(finite) < 24:
        return None
    source_xy = np.column_stack((xx_sample[finite], yy_sample[finite])).astype(
        np.float32
    )
    target_xy = np.column_stack((target_x[finite], target_y[finite])).astype(
        np.float32
    )
    affine, inliers = cv2.estimateAffinePartial2D(
        source_xy,
        target_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=3000,
        confidence=0.995,
        refineIters=25,
    )
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return None
    predicted = cv2.transform(source_xy[None, :, :], affine)[0]
    residual = np.linalg.norm(predicted - target_xy, axis=1)
    accepted = inliers.reshape(-1).astype(bool)
    if np.count_nonzero(accepted) < 24:
        return None
    inlier_ratio = float(np.mean(accepted))
    rmse = float(np.sqrt(np.mean(np.square(residual[accepted]))))
    scale = float(
        math.sqrt(float(affine[0, 0]) ** 2 + float(affine[1, 0]) ** 2)
    )
    rotation_deg = math.degrees(
        math.atan2(float(affine[1, 0]), float(affine[0, 0]))
    )
    if (
        inlier_ratio < 0.40
        or rmse > 5.0
        or not 0.40 <= scale <= 2.50
        or abs(rotation_deg) > 30.0
    ):
        return None
    target_mask = cv2.warpAffine(
        source_mask.astype(np.uint8),
        affine,
        (int(layout.width), int(layout.height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    target_yx = np.argwhere(target_mask > 0)
    if target_yx.size == 0:
        return None
    y0, x0 = np.min(target_yx, axis=0)
    y1, x1 = np.max(target_yx, axis=0) + 1
    centroid = np.median(points_world[finite], axis=0)
    central_x = float(np.median(xx_sample))
    central_y = float(np.median(yy_sample))
    centrality = 1.0 - min(
        1.0,
        math.hypot(
            (central_x - intrinsics.cx) / max(1.0, intrinsics.width * 0.5),
            (central_y - intrinsics.cy) / max(1.0, intrinsics.height * 0.5),
        ),
    )
    gray = cv2.cvtColor(source.image_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F)[source_mask].var())
    score = (
        math.log1p(int(np.count_nonzero(source_mask)))
        + 2.0 * centrality
        + 2.0 * inlier_ratio
        - 0.5 * rmse
        + 0.05 * math.log1p(max(0.0, sharpness))
    )
    return ForegroundObjectObservation(
        observation_id=observation_id,
        source_index=int(source.source_index),
        panel_index=int(source.panel_index),
        frame_id=int(source.frame_id),
        source_mask=np.ascontiguousarray(source_mask),
        source_to_canvas=np.asarray(affine, dtype=np.float64),
        world_centroid_mm=tuple(float(value) for value in centroid),
        median_normal_depth_mm=float(np.median(target_normal[finite])),
        target_bbox_xywh=(
            int(x0),
            int(y0),
            int(x1 - x0),
            int(y1 - y0),
        ),
        source_pixel_count=int(np.count_nonzero(source_mask)),
        target_pixel_count=int(np.count_nonzero(target_mask)),
        fit_inlier_ratio=inlier_ratio,
        fit_rmse_pixels=rmse,
        score=score,
    )


def _bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax0, ay0, aw, ah = first
    bx0, by0, bw, bh = second
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union else 0.0


def _split_oversized_world_cluster(
    indices: np.ndarray,
    world_basis: np.ndarray,
) -> list[np.ndarray]:
    """Partition a residual cluster without joining distinct depth layers.

    DBSCAN can legitimately connect several residual surfaces through sparse
    depth-edge samples.  The object gate is tighter along the viewing normal
    than along scan/down, so partition every axis that exceeds its object
    envelope.  Compact axes are deliberately left unsplit: an arbitrary bin
    boundary must not cut an otherwise valid object.
    """

    selected = np.asarray(indices, dtype=np.intp)
    if selected.ndim != 1:
        raise ValueError("World-cluster indices must be one-dimensional")
    if selected.size == 0:
        return []
    basis = np.asarray(world_basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != 3:
        raise ValueError("World basis must have scan/down/normal columns")
    values = basis[selected]
    if not np.isfinite(values).all():
        raise ValueError("World-cluster coordinates must be finite")

    maximum_spans = np.asarray((450.0, 450.0, 350.0), dtype=np.float64)
    bin_widths = np.asarray((300.0, 300.0, 300.0), dtype=np.float64)
    spans = np.ptp(values, axis=0)
    oversized = spans > maximum_spans
    if not np.any(oversized):
        return [selected]

    bins = np.zeros((selected.size, 3), dtype=np.int32)
    for axis in np.flatnonzero(oversized):
        origin = float(np.min(values[:, axis]))
        bins[:, axis] = np.floor(
            (values[:, axis] - origin) / bin_widths[axis]
        ).astype(np.int32)
    return [
        selected[np.all(bins == value, axis=1)]
        for value in np.unique(bins, axis=0)
        if np.any(np.all(bins == value, axis=1))
    ]


def plan_foreground_object_anchors(
    sources: Sequence[ForegroundAnchorSource],
    layout: object,
    intrinsics: CameraIntrinsics,
    *,
    minimum_component_pixels: int = 160,
) -> ForegroundObjectAnchorPlan:
    """Build world-associated object tracks and background exclusions."""

    reference_depth = float(layout.reference_depth_mm)
    margin = max(35.0, 0.04 * reference_depth)
    canvas_boundary_margin_pixels = max(
        16,
        min(
            64,
            int(round(0.06 * min(int(layout.width), int(layout.height)))),
        ),
    )
    sample_stride = 4 if intrinsics.width >= 640 else 3
    raw_world_parts: list[np.ndarray] = []
    raw_source_parts: list[np.ndarray] = []
    raw_x_parts: list[np.ndarray] = []
    raw_y_parts: list[np.ndarray] = []
    for source in sources:
        near = (
            source.reliable_depth
            & np.isfinite(source.depth_mm)
            & (source.depth_mm < np.float32(reference_depth - margin))
        )
        sampled = np.zeros_like(near)
        sampled[::sample_stride, ::sample_stride] = near[
            ::sample_stride, ::sample_stride
        ]
        yy, xx = np.nonzero(sampled)
        if xx.size == 0:
            continue
        camera = pinhole_unproject(
            xx,
            yy,
            source.depth_mm[yy, xx],
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
        )
        pose = np.asarray(source.camera_to_world, dtype=np.float64)
        world = transform_points(camera, pose[:3, :3], pose[:3, 3])
        raw_world_parts.append(world)
        raw_source_parts.append(
            np.full(xx.size, int(source.source_index), dtype=np.int16)
        )
        raw_x_parts.append(xx.astype(np.int16))
        raw_y_parts.append(yy.astype(np.int16))
    raw_world = (
        np.concatenate(raw_world_parts)
        if raw_world_parts
        else np.empty((0, 3), dtype=np.float64)
    )
    raw_source = (
        np.concatenate(raw_source_parts)
        if raw_source_parts
        else np.empty(0, dtype=np.int16)
    )
    raw_x = (
        np.concatenate(raw_x_parts)
        if raw_x_parts
        else np.empty(0, dtype=np.int16)
    )
    raw_y = (
        np.concatenate(raw_y_parts)
        if raw_y_parts
        else np.empty(0, dtype=np.int16)
    )
    voxel_size_mm = 8.0
    voxel_keys = np.floor(raw_world / voxel_size_mm).astype(np.int32)
    unique_keys, raw_to_voxel = np.unique(
        voxel_keys, axis=0, return_inverse=True
    )
    voxel_points = (
        unique_keys.astype(np.float64) + 0.5
    ) * voxel_size_mm
    structural = np.zeros(len(voxel_points), dtype=bool)
    cluster_labels = np.full(len(voxel_points), -1, dtype=np.int32)
    plane_count = 0
    structural_plane_models_world: list[np.ndarray] = []
    if len(voxel_points) >= 32:
        import open3d as o3d

        o3d.utility.random.seed(0)
        remaining = np.arange(len(voxel_points), dtype=np.int32)
        scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
        down_axis = np.asarray(layout.down_axis, dtype=np.float64)
        normal_axis = np.asarray(layout.normal_axis, dtype=np.float64)
        for _ in range(2):
            if remaining.size < 256:
                break
            cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(voxel_points[remaining])
            )
            plane_model, local_inliers = cloud.segment_plane(
                distance_threshold=10.0,
                ransac_n=3,
                num_iterations=300,
            )
            if len(local_inliers) < max(2000, int(0.08 * remaining.size)):
                break
            inliers = remaining[np.asarray(local_inliers, dtype=np.int32)]
            points = voxel_points[inliers]
            spans = np.ptp(
                np.column_stack(
                    (
                        points @ scan_axis,
                        points @ down_axis,
                        points @ normal_axis,
                    )
                ),
                axis=0,
            )
            ordered_spans = np.sort(spans)[::-1]
            if ordered_spans[0] < 350.0 or ordered_spans[1] < 120.0:
                break
            structural[inliers] = True
            structural_plane_models_world.append(
                np.asarray(plane_model, dtype=np.float64)
            )
            plane_count += 1
            remaining = remaining[~np.isin(remaining, inliers)]
        remaining = np.flatnonzero(~structural)
        if remaining.size:
            cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(voxel_points[remaining])
            )
            local_labels = np.asarray(
                cloud.cluster_dbscan(
                    eps=28.0, min_points=5, print_progress=False
                ),
                dtype=np.int32,
            )
            cluster_labels[remaining] = local_labels

    source_by_index = {item.source_index: item for item in sources}
    source_components = {
        int(source.source_index): (
            _component_masks_excluding_world_planes(
                source,
                reference_depth,
                intrinsics,
                structural_plane_models_world,
                minimum_pixels=minimum_component_pixels,
            )
            if structural_plane_models_world
            else _component_masks(
                source.depth_mm,
                source.reliable_depth,
                reference_depth,
                minimum_pixels=minimum_component_pixels,
            )
        )
        for source in sources
    }
    observations = []
    anchors: list[ForegroundObjectAnchor] = []
    raw_labels = (
        cluster_labels[raw_to_voxel]
        if raw_to_voxel.size
        else np.empty(0, dtype=np.int32)
    )
    # DBSCAN deliberately uses a tight radius so it cannot bridge through a
    # removed shelf plane.  Rejoin nearby residual surface fragments only
    # when their combined world-axis box still fits one compact object.
    initial_ids = sorted(
        int(value) for value in np.unique(raw_labels) if int(value) >= 0
    )
    if initial_ids:
        world_basis = np.column_stack(
            (
                raw_world @ np.asarray(layout.scan_axis),
                raw_world @ np.asarray(layout.down_axis),
                raw_world @ np.asarray(layout.normal_axis),
            )
        )
        bounds: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for value in initial_ids:
            points = world_basis[raw_labels == value]
            bounds[value] = (np.min(points, axis=0), np.max(points, axis=0))
        parent = {value: value for value in initial_ids}

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        for first_index, first in enumerate(initial_ids):
            first_min, first_max = bounds[first]
            for second in initial_ids[first_index + 1 :]:
                second_min, second_max = bounds[second]
                gap = np.maximum(
                    0.0,
                    np.maximum(first_min, second_min)
                    - np.minimum(first_max, second_max),
                )
                merged_min = np.minimum(first_min, second_min)
                merged_max = np.maximum(first_max, second_max)
                merged_span = merged_max - merged_min
                if (
                    float(np.linalg.norm(gap)) <= 50.0
                    and gap[2] <= 35.0
                    and merged_span[0] <= 450.0
                    and merged_span[1] <= 450.0
                    and merged_span[2] <= 350.0
                ):
                    a, b = find(first), find(second)
                    if a != b:
                        parent[max(a, b)] = min(a, b)
        root_to_label: dict[int, int] = {}
        for value in initial_ids:
            root = find(value)
            root_to_label.setdefault(root, len(root_to_label))
            raw_labels[raw_labels == value] = root_to_label[root]
    cluster_ids = sorted(
        int(value) for value in np.unique(raw_labels) if int(value) >= 0
    )
    cluster_groups: list[np.ndarray] = []
    raw_world_basis = np.column_stack(
        (
            raw_world @ np.asarray(layout.scan_axis),
            raw_world @ np.asarray(layout.down_axis),
            raw_world @ np.asarray(layout.normal_axis),
        )
    )
    oversized_cluster_count = 0
    for cluster_id in cluster_ids:
        indices = np.flatnonzero(raw_labels == cluster_id)
        if indices.size == 0:
            continue
        parts = _split_oversized_world_cluster(indices, raw_world_basis)
        if len(parts) > 1:
            oversized_cluster_count += 1
        cluster_groups.extend(parts)
    rejected_world_clusters = 0
    rejection_reasons: dict[str, int] = {}
    rejected_cluster_audits: list[dict[str, object]] = []

    def reject(
        reason: str,
        *,
        cluster_index: int,
        raw_sample_count: int,
        source_support_count: int,
        spans_mm: np.ndarray | None = None,
        valid_observation_count: int = 0,
        observation_audits: Sequence[dict[str, object]] = (),
    ) -> None:
        nonlocal rejected_world_clusters
        rejected_world_clusters += 1
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        rejected_cluster_audits.append(
            {
                "cluster_index": int(cluster_index),
                "reason": reason,
                "raw_sample_count": int(raw_sample_count),
                "world_source_support_count": int(source_support_count),
                "world_spans_mm": (
                    None
                    if spans_mm is None
                    else [float(value) for value in spans_mm]
                ),
                "valid_observation_count": int(
                    valid_observation_count
                ),
                "observations": [
                    dict(item) for item in observation_audits
                ],
            }
        )

    for cluster_index, raw_indices in enumerate(cluster_groups):
        raw_source_support = np.unique(raw_source[raw_indices])
        if raw_indices.size < max(
            12,
            minimum_component_pixels
            // (2 * sample_stride * sample_stride),
        ):
            reject(
                "insufficient_raw_samples",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(raw_source_support.size),
            )
            continue
        points = raw_world[raw_indices]
        basis = np.column_stack(
            (
                points @ np.asarray(layout.scan_axis),
                points @ np.asarray(layout.down_axis),
                points @ np.asarray(layout.normal_axis),
            )
        )
        spans = np.ptp(basis, axis=0)
        if (
            spans[0] > 450.0
            or spans[1] > 450.0
            or spans[2] > 350.0
            or np.count_nonzero(spans > 4.0) < 2
        ):
            reject(
                "world_bbox_is_structural_or_degenerate",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(raw_source_support.size),
                spans_mm=spans,
            )
            continue
        frame_ids = np.unique(raw_source[raw_indices])
        if frame_ids.size < 2 and raw_indices.size < 20:
            reject(
                "single_frame_support_too_small",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(frame_ids.size),
                spans_mm=spans,
            )
            continue
        group: list[ForegroundObjectObservation] = []
        panel_anchors = np.asarray(
            [panel.anchor_scan_mm for panel in layout.panels],
            dtype=np.float64,
        )
        target_panel_index = int(
            np.argmin(
                np.abs(
                    panel_anchors
                    - float(np.median(basis[:, 0]))
                )
            )
        )
        # Mask completion is the expensive part of object anchoring.  World
        # association keeps every supporting source for audit, but only the
        # four views with the densest measured support are needed to prove two
        # independent renderable RGB observations and select the best owner.
        render_source_ids = sorted(
            (int(value) for value in frame_ids),
            key=lambda value: (
                -int(
                    np.count_nonzero(
                        raw_source[raw_indices] == value
                    )
                ),
                value,
            ),
        )[:4]
        for source_index in render_source_ids:
            selected_indices = raw_indices[
                raw_source[raw_indices] == source_index
            ]
            if selected_indices.size < 24:
                continue
            source = source_by_index[int(source_index)]
            seed = np.zeros(source.depth_mm.shape, dtype=np.uint8)
            seed[
                raw_y[selected_indices].astype(np.intp),
                raw_x[selected_indices].astype(np.intp),
            ] = 1
            components = source_components[int(source_index)]
            if not components:
                continue
            sample_y = raw_y[selected_indices].astype(np.intp)
            sample_x = raw_x[selected_indices].astype(np.intp)
            support = np.asarray(
                [
                    np.count_nonzero(component[sample_y, sample_x])
                    for component in components
                ],
                dtype=np.int32,
            )
            selected_component = int(np.argmax(support))
            supported_samples = int(support[selected_component])
            if supported_samples < max(
                12, int(math.ceil(0.15 * selected_indices.size))
            ):
                continue
            # A complete silhouette must come from one measured near-depth
            # component.  RGB GrabCut is intentionally not a fallback here:
            # on the real shelf it expanded arbitrary binned structural
            # fragments into the wall and ceiling.
            complete = components[selected_component]
            fit_mask = (
                complete
                & source.reliable_depth
                & (
                    source.depth_mm
                    < np.float32(reference_depth - margin)
                )
            )
            if np.count_nonzero(fit_mask) < minimum_component_pixels:
                continue
            item = _fit_observation(
                observation_id=len(observations),
                source=source,
                source_mask=complete,
                fit_mask=fit_mask,
                layout=layout,
                intrinsics=intrinsics,
                target_panel_index=target_panel_index,
            )
            if item is not None:
                observations.append(item)
                group.append(item)
        if not group:
            reject(
                "no_complete_source_mask_or_valid_affine",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(frame_ids.size),
                spans_mm=spans,
            )
            continue
        observation_audit_rows = [
            {
                "frame_id": int(item.frame_id),
                "source_index": int(item.source_index),
                "target_bbox_xywh": [
                    int(value) for value in item.target_bbox_xywh
                ],
                "source_pixel_count": int(item.source_pixel_count),
                "target_pixel_count": int(item.target_pixel_count),
                "fit_inlier_ratio": float(item.fit_inlier_ratio),
                "fit_rmse_pixels": float(item.fit_rmse_pixels),
                "world_centroid_mm": [
                    float(value) for value in item.world_centroid_mm
                ],
            }
            for item in group
        ]
        if len(group) < 2 and len(sources) > 2:
            reject(
                "world_cluster_lacks_two_renderable_rgb_observations",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(frame_ids.size),
                spans_mm=spans,
                valid_observation_count=len(group),
                observation_audits=observation_audit_rows,
            )
            observations = observations[: -len(group)]
            continue
        if len(group) >= 2:
            mutually_consistent = True
            for first_index, first in enumerate(group):
                first_area = (
                    first.target_bbox_xywh[2]
                    * first.target_bbox_xywh[3]
                )
                first_centroid = np.asarray(
                    first.world_centroid_mm, dtype=np.float64
                )
                for second in group[first_index + 1 :]:
                    second_area = (
                        second.target_bbox_xywh[2]
                        * second.target_bbox_xywh[3]
                    )
                    area_ratio = max(first_area, second_area) / max(
                        1, min(first_area, second_area)
                    )
                    centroid_distance = float(
                        np.linalg.norm(
                            first_centroid
                            - np.asarray(
                                second.world_centroid_mm,
                                dtype=np.float64,
                            )
                        )
                    )
                    if (
                        _bbox_iou(
                            first.target_bbox_xywh,
                            second.target_bbox_xywh,
                        )
                        < 0.30
                        or area_ratio > 2.50
                        or centroid_distance > 100.0
                    ):
                        mutually_consistent = False
                        break
                if not mutually_consistent:
                    break
            if not mutually_consistent:
                reject(
                    "world_cluster_renderable_observations_disagree",
                    cluster_index=cluster_index,
                    raw_sample_count=int(raw_indices.size),
                    source_support_count=int(frame_ids.size),
                    spans_mm=spans,
                    valid_observation_count=len(group),
                    observation_audits=observation_audit_rows,
                )
                observations = observations[: -len(group)]
                continue
        # Cross-frame object identity is established by the immutable world
        # cluster, not by requiring every supporting view to also pass the
        # display-only affine/mask gate.  One complete renderable owner is
        # sufficient once two or more real RGB-D sources support the same
        # compact world object.
        if frame_ids.size < 2 and len(sources) > 2:
            reject(
                "world_cluster_lacks_cross_frame_support",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(frame_ids.size),
                spans_mm=spans,
                valid_observation_count=len(group),
            )
            observations = observations[: -len(group)]
            continue
        selected = max(
            group,
            key=lambda item: (
                item.score,
                item.source_pixel_count,
                -item.frame_id,
            ),
        )
        if (
            len(sources) > 2
            and selected.median_normal_depth_mm
            >= reference_depth - margin
        ):
            reject(
                "world_cluster_not_in_front_of_reference_surface",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(frame_ids.size),
                spans_mm=spans,
                valid_observation_count=len(group),
            )
            observations = observations[: -len(group)]
            continue
        x0, y0, width, height = selected.target_bbox_xywh
        if len(sources) > 2 and (
            x0 < canvas_boundary_margin_pixels
            or y0 < canvas_boundary_margin_pixels
            or x0 + width
            > int(layout.width) - canvas_boundary_margin_pixels
            or y0 + height
            > int(layout.height) - canvas_boundary_margin_pixels
        ):
            reject(
                "world_cluster_target_touches_canvas_boundary",
                cluster_index=cluster_index,
                raw_sample_count=int(raw_indices.size),
                source_support_count=int(frame_ids.size),
                spans_mm=spans,
                valid_observation_count=len(group),
            )
            observations = observations[: -len(group)]
            continue
        anchors.append(
            ForegroundObjectAnchor(
                track_id=len(anchors),
                selected=selected,
                observation_ids=tuple(
                    item.observation_id for item in group
                ),
                world_support_source_indices=tuple(
                    int(value) for value in frame_ids
                ),
            )
        )

    exclusions: list[np.ndarray] = []
    for source in sources:
        exclusion = np.zeros(source.reference_map_x.shape, dtype=bool)
        for observation in observations:
            if observation.source_index != source.source_index:
                continue
            sampled = cv2.remap(
                observation.source_mask.astype(np.uint8),
                np.asarray(source.reference_map_x, dtype=np.float32),
                np.asarray(source.reference_map_y, dtype=np.float32),
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            exclusion |= sampled > 0
        exclusions.append(
            cv2.dilate(
                exclusion.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ).astype(bool)
        )
    target_mask = np.zeros(
        (int(layout.height), int(layout.width)), dtype=bool
    )
    for anchor in anchors:
        warped = cv2.warpAffine(
            anchor.selected.source_mask.astype(np.uint8),
            anchor.selected.source_to_canvas,
            (int(layout.width), int(layout.height)),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        target_mask |= warped > 0
    return ForegroundObjectAnchorPlan(
        anchors=tuple(anchors),
        observations=tuple(observations),
        background_exclusion_masks=tuple(exclusions),
        target_mask=target_mask,
        audit={
            "policy": (
                "rgbd_world_component_track_one_complete_rgb_owner_"
                "constrained_similarity"
            ),
            "near_depth_margin_mm": max(
                35.0, 0.04 * float(layout.reference_depth_mm)
            ),
            "canvas_boundary_margin_pixels": (
                canvas_boundary_margin_pixels
            ),
            "raw_sample_point_count": int(len(raw_world)),
            "voxel_size_mm": voxel_size_mm,
            "voxel_count": int(len(voxel_points)),
            "removed_structural_plane_count": plane_count,
            "source_component_mask_policy": (
                "measured_near_rgbd_after_fitted_world_plane_exclusion"
                if structural_plane_models_world
                else "image_depth_edge_fallback_without_fitted_world_plane"
            ),
            "raw_world_cluster_count": len(cluster_groups),
            "oversized_dbscan_cluster_count": oversized_cluster_count,
            "world_cluster_split_policy": (
                "only_oversized_world_axes_scan_down_300mm_"
                "normal_300mm_no_overlap"
            ),
            "source_mask_policy": (
                "single_measured_near_depth_component_no_rgb_grabcut_"
                "fallback"
            ),
            "rejected_world_cluster_count": rejected_world_clusters,
            "world_cluster_rejection_reasons": rejection_reasons,
            "rejected_world_clusters": rejected_cluster_audits,
            "observation_count": len(observations),
            "track_count": len(anchors),
            "background_exclusion_pixel_counts": [
                int(np.count_nonzero(mask)) for mask in exclusions
            ],
            "tracks": [
                {
                    "track_id": anchor.track_id,
                    "observation_ids": list(anchor.observation_ids),
                    "selected_observation_id": anchor.selected.observation_id,
                    "selected_frame_id": anchor.selected.frame_id,
                    "world_support_source_count": len(
                        anchor.world_support_source_indices
                    ),
                    "world_support_source_indices": list(
                        anchor.world_support_source_indices
                    ),
                    "world_centroid_mm": list(
                        anchor.selected.world_centroid_mm
                    ),
                    "target_bbox_xywh": list(
                        anchor.selected.target_bbox_xywh
                    ),
                    "fit_inlier_ratio": anchor.selected.fit_inlier_ratio,
                    "fit_rmse_pixels": anchor.selected.fit_rmse_pixels,
                }
                for anchor in anchors
            ],
        },
    )


def overlay_foreground_object_anchors(
    *,
    plan: ForegroundObjectAnchorPlan,
    sources: Sequence[ForegroundAnchorSource],
    output_image: np.ndarray,
    output_owner: np.ndarray,
    output_depth: np.ndarray,
    output_confidence: np.ndarray,
) -> ForegroundObjectOverlayResult:
    """Hard-overlay each tracked object with a per-pixel depth z-buffer."""

    source_by_index = {item.source_index: item for item in sources}
    shape = output_owner.shape
    z_buffer = np.full(shape, np.inf, dtype=np.float32)
    labels = np.full(shape, -1, dtype=np.int32)
    written_counts: list[int] = []
    for anchor in sorted(
        plan.anchors,
        key=lambda item: (
            -item.selected.median_normal_depth_mm,
            item.track_id,
        ),
    ):
        observation = anchor.selected
        source = source_by_index[observation.source_index]
        warped_image = cv2.warpAffine(
            source.image_bgr,
            observation.source_to_canvas,
            (shape[1], shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            observation.source_mask.astype(np.uint8),
            observation.source_to_canvas,
            (shape[1], shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        depth = np.float32(observation.median_normal_depth_mm)
        take = warped_mask & (depth < z_buffer)
        output_image[take] = warped_image[take]
        output_owner[take] = int(observation.frame_id)
        output_depth[take] = depth
        output_confidence[take] = np.float32(
            max(1.0 / 65535.0, observation.fit_inlier_ratio)
        )
        z_buffer[take] = depth
        labels[take] = int(anchor.track_id)
        written_counts.append(int(np.count_nonzero(take)))
    visible = labels >= 0
    return ForegroundObjectOverlayResult(
        component_label=labels,
        visible_mask=visible,
        audit={
            **plan.audit,
            "overlay_policy": (
                "depth_ordered_hard_rgb_owner_no_alpha_no_multiband"
            ),
            "visible_pixel_count": int(np.count_nonzero(visible)),
            "per_track_written_pixel_counts": written_counts,
            "all_tracks_visible": bool(
                len(written_counts) == len(plan.anchors)
                and all(value > 0 for value in written_counts)
            ),
            "blend_pixel_count": 0,
        },
    )


__all__ = [
    "ForegroundAnchorSource",
    "ForegroundObjectAnchor",
    "ForegroundObjectAnchorPlan",
    "ForegroundObjectObservation",
    "ForegroundObjectOverlayResult",
    "overlay_foreground_object_anchors",
    "plan_foreground_object_anchors",
]
