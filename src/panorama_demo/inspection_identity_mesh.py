"""RGB-D inverse-mesh composition for identity-owned inspection objects.

The segmentation/tracking layer decides only *which* source mask belongs to
one object.  This module decides every output position from aligned depth,
the immutable camera-to-world pose, and the selected virtual panel.  RGB is
sampled once from that same real source; no affine placement, alpha blend,
inpainting, or generated colour is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from .cuda_backend import (
    pinhole_unproject,
    remap as accelerated_remap,
    transform_points,
)
from .session import CameraIntrinsics


class IdentityOwner(Protocol):
    group_id: int
    structure_id: int
    panel_index: int
    target_panel_index: int | None
    frame_id: int
    source_mask: np.ndarray
    target_footprint: np.ndarray


@dataclass(frozen=True)
class _ProjectionOwner:
    panel_index: int
    target_panel_index: int
    source_mask: np.ndarray


@dataclass(frozen=True)
class InspectionIdentityMeshSource:
    panel_index: int
    frame_id: int
    image_bgr: np.ndarray
    depth_mm: np.ndarray
    reliable_depth: np.ndarray
    camera_to_world: np.ndarray


@dataclass(frozen=True)
class InspectionIdentityMeshConfig:
    cell_size_pixels: int = 4
    maximum_fill_distance_pixels: float = 2.0
    maximum_fill_fraction: float = 0.06
    minimum_direct_support_ratio: float = 0.94
    minimum_depth_mm: float = 200.0
    maximum_depth_mm: float = 3000.0
    minimum_jacobian: float = 0.01
    maximum_jacobian: float = 64.0
    minimum_seed_pixels: int = 30

    def validate(self) -> None:
        if not 1 <= int(self.cell_size_pixels) <= 16:
            raise ValueError("Identity mesh cell size must be in [1, 16]")
        if not 0.0 < float(self.maximum_fill_distance_pixels) <= 2.0:
            raise ValueError(
                "Identity mesh fill distance must be in (0, 2]"
            )
        if not 0.0 <= float(self.maximum_fill_fraction) <= 0.06:
            raise ValueError(
                "Identity mesh fill fraction must be in [0, 0.06]"
            )
        if not 0.94 <= float(self.minimum_direct_support_ratio) <= 1.0:
            raise ValueError(
                "Identity mesh direct support ratio must be in [0.94, 1]"
            )
        if not 0.0 < self.minimum_depth_mm < self.maximum_depth_mm:
            raise ValueError("Identity mesh depth limits are invalid")
        if not 0.0 < self.minimum_jacobian < self.maximum_jacobian:
            raise ValueError("Identity mesh Jacobian limits are invalid")
        if int(self.minimum_seed_pixels) < 16:
            raise ValueError("Identity mesh needs at least 16 seed pixels")


def _validate_pose(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Identity mesh pose must be finite 4x4")
    rotation = pose[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
        or not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8)
    ):
        raise ValueError("Identity mesh pose must be rigid SE(3)")
    return pose


def _rasterize_triangle(
    map_x: np.ndarray,
    map_y: np.ndarray,
    target_depth: np.ndarray,
    *,
    target_xy: np.ndarray,
    source_xy: np.ndarray,
    vertex_depth: np.ndarray,
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
    a, b, c = np.asarray(target_xy, dtype=np.float64)
    determinant = float(
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )
    if not math.isfinite(determinant) or determinant <= 0.0:
        return 0
    yy, xx = np.indices((y1 - y0 + 1, x1 - x0 + 1), dtype=np.float64)
    xx += x0
    yy += y0
    weight_b = (
        (xx - a[0]) * (c[1] - a[1])
        - (yy - a[1]) * (c[0] - a[0])
    ) / determinant
    weight_c = (
        (b[0] - a[0]) * (yy - a[1])
        - (b[1] - a[1]) * (xx - a[0])
    ) / determinant
    weight_a = 1.0 - weight_b - weight_c
    inside = (
        (weight_a >= -1e-6)
        & (weight_b >= -1e-6)
        & (weight_c >= -1e-6)
    )
    depth = (
        weight_a * vertex_depth[0]
        + weight_b * vertex_depth[1]
        + weight_c * vertex_depth[2]
    )
    destination = target_depth[y0 : y1 + 1, x0 : x1 + 1]
    take = (
        inside
        & np.isfinite(depth)
        & (depth > 0.0)
        & (~np.isfinite(destination) | (depth < destination))
    )
    if not np.any(take):
        return 0
    candidate_x = (
        weight_a * source_xy[0, 0]
        + weight_b * source_xy[1, 0]
        + weight_c * source_xy[2, 0]
    )
    candidate_y = (
        weight_a * source_xy[0, 1]
        + weight_b * source_xy[1, 1]
        + weight_c * source_xy[2, 1]
    )
    map_x[y0 : y1 + 1, x0 : x1 + 1][take] = candidate_x[take]
    map_y[y0 : y1 + 1, x0 : x1 + 1][take] = candidate_y[take]
    destination[take] = depth[take].astype(np.float32)
    return int(np.count_nonzero(take))


def _project_source_grid(
    *,
    owner: IdentityOwner,
    source: InspectionIdentityMeshSource,
    layout: object,
    intrinsics: CameraIntrinsics,
    config: InspectionIdentityMeshConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    source_mask = np.asarray(owner.source_mask, dtype=bool)
    depth = np.asarray(source.depth_mm, dtype=np.float32)
    reliable = np.asarray(source.reliable_depth, dtype=bool)
    expected = (intrinsics.height, intrinsics.width)
    if (
        source_mask.shape != expected
        or depth.shape != expected
        or reliable.shape != expected
    ):
        raise ValueError("Identity mesh source masks do not match intrinsics")
    valid = (
        source_mask
        & reliable
        & np.isfinite(depth)
        & (depth >= config.minimum_depth_mm)
        & (depth <= config.maximum_depth_mm)
    )
    yy, xx = np.nonzero(source_mask)
    if xx.size == 0:
        raise RuntimeError("Identity owner source mask is empty")
    margin = 1
    x0 = max(margin, int(np.min(xx)))
    x1 = min(intrinsics.width - 1 - margin, int(np.max(xx)))
    y0 = max(margin, int(np.min(yy)))
    y1 = min(intrinsics.height - 1 - margin, int(np.max(yy)))
    step = int(config.cell_size_pixels)
    xs = list(range(x0, x1 + 1, step))
    ys = list(range(y0, y1 + 1, step))
    if xs and xs[-1] != x1:
        xs.append(x1)
    if ys and ys[-1] != y1:
        ys.append(y1)
    if len(xs) < 2 or len(ys) < 2:
        raise RuntimeError("Identity owner mask is too small for a mesh")
    grid_x, grid_y = np.meshgrid(
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.int32),
    )
    grid_depth = depth[grid_y, grid_x].astype(np.float64)
    camera = pinhole_unproject(
        grid_x.reshape(-1),
        grid_y.reshape(-1),
        grid_depth.reshape(-1),
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    pose = _validate_pose(source.camera_to_world)
    world = transform_points(camera, pose[:3, :3], pose[:3, 3]).reshape(
        (*grid_x.shape, 3)
    )
    target_panel_index = (
        int(owner.panel_index)
        if owner.target_panel_index is None
        else int(owner.target_panel_index)
    )
    panel = layout.panels[target_panel_index]
    center = np.asarray(panel.center_world_mm, dtype=np.float64)
    relative = world - center
    scan = relative @ np.asarray(layout.scan_axis, dtype=np.float64)
    down = relative @ np.asarray(layout.down_axis, dtype=np.float64)
    normal = relative @ np.asarray(layout.normal_axis, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        target_x = intrinsics.cx + intrinsics.fx * scan / normal
        target_y = (
            float(getattr(layout, "canvas_offset_y", 0.0))
            + intrinsics.cy
            + intrinsics.fy * down / normal
        )
    corner_x = int(round(float(panel.canvas_offset_x)))
    local_width = min(intrinsics.width, int(layout.width) - corner_x)
    if local_width <= 0:
        raise RuntimeError("Identity owner panel is outside the canvas")
    map_x = np.full(
        (int(layout.height), local_width), np.nan, dtype=np.float32
    )
    map_y = np.full_like(map_x, np.nan)
    target_depth = np.full_like(map_x, np.inf)
    candidate_cells = 0
    accepted_cells = 0
    rejected_mask_or_depth = 0
    rejected_discontinuous = 0
    rejected_jacobian = 0
    updates = 0
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            candidate_cells += 1
            sx0, sx1 = xs[column], xs[column + 1]
            sy0, sy1 = ys[row], ys[row + 1]
            nodes = (
                (row, column),
                (row, column + 1),
                (row + 1, column + 1),
                (row + 1, column),
            )
            source_nodes = (
                (sx0, sy0),
                (sx1, sy0),
                (sx1, sy1),
                (sx0, sy1),
            )
            if not all(valid[sy, sx] for sx, sy in source_nodes):
                rejected_mask_or_depth += 1
                continue
            node_depth = np.asarray(
                [grid_depth[node] for node in nodes], dtype=np.float64
            )
            tolerance = max(
                20.0, 0.02 * float(np.median(node_depth))
            )
            if float(np.max(node_depth) - np.min(node_depth)) > tolerance:
                rejected_discontinuous += 1
                continue
            target = np.asarray(
                [
                    (target_x[node], target_y[node])
                    for node in nodes
                ],
                dtype=np.float64,
            )
            z = np.asarray(
                [normal[node] for node in nodes], dtype=np.float64
            )
            source_xy = np.asarray(source_nodes, dtype=np.float64)
            source_area = float(max(1, (sx1 - sx0) * (sy1 - sy0)))
            triangles = ((0, 1, 2), (0, 2, 3))
            payload: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            jacobians: list[float] = []
            for indices in triangles:
                selected_target = target[list(indices)]
                vector_ab = selected_target[1] - selected_target[0]
                vector_ac = selected_target[2] - selected_target[0]
                jacobian = float(
                    vector_ab[0] * vector_ac[1]
                    - vector_ab[1] * vector_ac[0]
                ) / source_area
                jacobians.append(jacobian)
                payload.append(
                    (
                        selected_target,
                        source_xy[list(indices)],
                        z[list(indices)],
                    )
                )
            if any(
                not math.isfinite(value)
                or value < config.minimum_jacobian
                or value > config.maximum_jacobian
                for value in jacobians
            ):
                rejected_jacobian += 1
                continue
            accepted_cells += 1
            for target_triangle, source_triangle, depth_triangle in payload:
                updates += _rasterize_triangle(
                    map_x,
                    map_y,
                    target_depth,
                    target_xy=target_triangle,
                    source_xy=source_triangle,
                    vertex_depth=depth_triangle,
                )
    finite = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(target_depth)
    )
    map_x[~finite] = np.nan
    map_y[~finite] = np.nan
    target_depth[~finite] = np.inf
    return map_x, map_y, target_depth, {
        "candidate_cell_count": candidate_cells,
        "accepted_cell_count": accepted_cells,
        "rejected_mask_or_depth_cell_count": rejected_mask_or_depth,
        "rejected_discontinuous_cell_count": rejected_discontinuous,
        "rejected_jacobian_cell_count": rejected_jacobian,
        "rasterized_pixel_update_count": updates,
        "seed_pixel_count": int(np.count_nonzero(finite)),
        "panel_corner_x": corner_x,
        "panel_local_width": local_width,
    }


def _fill_maps_inside_footprint(
    *,
    footprint: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    depth: np.ndarray,
    source_mask: np.ndarray,
    maximum_distance: float,
    maximum_fill_fraction: float,
    minimum_direct_support_ratio: float,
    minimum_seed_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    seed = (
        footprint
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(depth)
    )
    seed_count = int(np.count_nonzero(seed))
    if seed_count < minimum_seed_pixels:
        raise RuntimeError(
            "Identity owner has insufficient direct inverse-mesh support"
        )
    footprint_count = int(np.count_nonzero(footprint))
    direct_support_ratio = float(seed_count / max(1, footprint_count))
    if direct_support_ratio < minimum_direct_support_ratio:
        raise RuntimeError(
            "Identity owner direct inverse-mesh support ratio is below "
            f"the fixed gate: {direct_support_ratio:.6f} < "
            f"{minimum_direct_support_ratio:.6f}"
        )
    distance_input = np.ones(footprint.shape, dtype=np.uint8)
    distance_input[seed] = 0
    distances, labels = cv2.distanceTransformWithLabels(
        distance_input,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    label_to_values: dict[int, tuple[float, float, float]] = {}
    for y, x in zip(*np.nonzero(seed), strict=True):
        label_to_values[int(labels[y, x])] = (
            float(map_x[y, x]),
            float(map_y[y, x]),
            float(depth[y, x]),
        )
    fill = footprint & ~seed
    fill_count = int(np.count_nonzero(fill))
    fill_fraction = float(fill_count / max(1, footprint_count))
    if fill_fraction > maximum_fill_fraction:
        raise RuntimeError(
            "Identity owner inverse-mesh fill fraction exceeds the fixed "
            f"gate: {fill_fraction:.6f} > {maximum_fill_fraction:.6f}"
        )
    maximum_observed = float(np.max(distances[fill], initial=0.0))
    if maximum_observed > maximum_distance:
        raise RuntimeError(
            "Identity owner depth hole exceeds bounded inverse-mesh fill: "
            f"{maximum_observed:.3f} > {maximum_distance:.3f}"
        )
    result_x = map_x.copy()
    result_y = map_y.copy()
    result_depth = depth.copy()
    for y, x in zip(*np.nonzero(fill), strict=True):
        values = label_to_values.get(int(labels[y, x]))
        if values is None:
            raise RuntimeError("Identity owner nearest mesh seed is missing")
        result_x[y, x], result_y[y, x], result_depth[y, x] = values
    rounded_x = np.rint(result_x[footprint]).astype(np.int32)
    rounded_y = np.rint(result_y[footprint]).astype(np.int32)
    inside = (
        (rounded_x >= 0)
        & (rounded_x < source_mask.shape[1])
        & (rounded_y >= 0)
        & (rounded_y < source_mask.shape[0])
    )
    sampled_inside_mask = np.zeros(inside.shape, dtype=bool)
    sampled_inside_mask[inside] = source_mask[
        rounded_y[inside], rounded_x[inside]
    ]
    escaped = np.zeros(footprint.shape, dtype=bool)
    escaped[footprint] = ~sampled_inside_mask
    if np.any(escaped):
        raise RuntimeError(
            "Identity owner inverse mesh escapes its source mask"
        )
    return result_x, result_y, result_depth, {
        "direct_inverse_mesh_pixel_count": seed_count,
        "direct_inverse_mesh_support_ratio": direct_support_ratio,
        "minimum_direct_inverse_mesh_support_ratio": float(
            minimum_direct_support_ratio
        ),
        "filled_pixel_count": fill_count,
        "fill_fraction": fill_fraction,
        "maximum_allowed_fill_fraction": float(maximum_fill_fraction),
        "maximum_fill_distance_pixels": maximum_observed,
        "maximum_allowed_fill_distance_pixels": float(maximum_distance),
        "smoothed_fill_pixel_count": 0,
        "smoothed_fill_reverted_to_nearest_pixel_count": 0,
        "source_mask_escape_pixel_count": 0,
    }


def composite_inspection_identity_owners(
    *,
    owners: Sequence[IdentityOwner],
    sources_by_frame_id: Mapping[int, InspectionIdentityMeshSource],
    layout: object,
    intrinsics: CameraIntrinsics,
    output_image: np.ndarray,
    output_depth: np.ndarray,
    output_confidence: np.ndarray,
    output_owner: np.ndarray,
    output_reliable_depth: np.ndarray,
    output_overlay_mask: np.ndarray,
    config: InspectionIdentityMeshConfig | None = None,
) -> dict[str, object]:
    """Composite each identity structure from one real RGB-D owner."""

    settings = config or InspectionIdentityMeshConfig()
    settings.validate()
    canvas_shape = (int(layout.height), int(layout.width))
    if any(
        value.shape != canvas_shape
        for value in (
            output_depth,
            output_confidence,
            output_owner,
            output_reliable_depth,
            output_overlay_mask,
        )
    ) or output_image.shape != (*canvas_shape, 3):
        raise ValueError("Identity owner outputs are not canvas-aligned")
    rows: list[dict[str, object]] = []
    seen_structures: set[tuple[int, int]] = set()
    identity_depth_buffer = np.full(
        canvas_shape, np.inf, dtype=np.float32
    )
    identity_group_buffer = np.full(
        canvas_shape, -1, dtype=np.int64
    )
    sources_by_panel_index: dict[int, InspectionIdentityMeshSource] = {}
    for source in sources_by_frame_id.values():
        panel_index = int(source.panel_index)
        if panel_index in sources_by_panel_index:
            raise ValueError(
                "Identity mesh sources contain duplicate panel indices"
            )
        sources_by_panel_index[panel_index] = source
    target_scene_depth_cache: dict[
        int, tuple[int, np.ndarray, dict[str, object]]
    ] = {}
    for owner in owners:
        key = (int(owner.group_id), int(owner.structure_id))
        if key in seen_structures:
            raise ValueError("Identity owner structure IDs must be unique")
        seen_structures.add(key)
        frame_id = int(owner.frame_id)
        source_panel_index = int(owner.panel_index)
        target_panel_index = (
            source_panel_index
            if owner.target_panel_index is None
            else int(owner.target_panel_index)
        )
        source = sources_by_frame_id.get(frame_id)
        if source is None:
            raise RuntimeError(
                "Identity owner selected source frame is unavailable"
            )
        if (
            int(source.frame_id) != frame_id
            or int(source.panel_index) != source_panel_index
        ):
            raise RuntimeError(
                "Identity owner source panel/frame mapping is inconsistent"
            )
        footprint = np.asarray(owner.target_footprint, dtype=bool)
        if footprint.shape != canvas_shape or not np.any(footprint):
            raise ValueError(
                "Identity owner target footprint is empty or misaligned"
            )
        attempt_cell_sizes = tuple(
            dict.fromkeys(
                (
                    int(settings.cell_size_pixels),
                    max(2, int(settings.cell_size_pixels) // 2),
                    1,
                )
            )
        )
        attempt_failures: list[str] = []
        for attempt_cell_size in attempt_cell_sizes:
            attempt_config = replace(
                settings, cell_size_pixels=attempt_cell_size
            )
            map_x, map_y, identity_depth, mesh_audit = (
                _project_source_grid(
                    owner=owner,
                    source=source,
                    layout=layout,
                    intrinsics=intrinsics,
                    config=attempt_config,
                )
            )
            corner_x = int(mesh_audit["panel_corner_x"])
            x1 = corner_x + map_x.shape[1]
            local_footprint = footprint[:, corner_x:x1]
            if (
                int(np.count_nonzero(local_footprint))
                != int(np.count_nonzero(footprint))
            ):
                raise RuntimeError(
                    "Identity owner target footprint escapes its real panel"
                )
            try:
                map_x, map_y, identity_depth, fill_audit = (
                    _fill_maps_inside_footprint(
                        footprint=local_footprint,
                        map_x=map_x,
                        map_y=map_y,
                        depth=identity_depth,
                        source_mask=np.asarray(
                            owner.source_mask, dtype=bool
                        ),
                        maximum_distance=(
                            settings.maximum_fill_distance_pixels
                        ),
                        maximum_fill_fraction=(
                            settings.maximum_fill_fraction
                        ),
                        minimum_direct_support_ratio=(
                            settings.minimum_direct_support_ratio
                        ),
                        minimum_seed_pixels=settings.minimum_seed_pixels,
                    )
                )
            except RuntimeError as error:
                attempt_failures.append(
                    f"{attempt_cell_size}px: {error}"
                )
                continue
            mesh_audit["configured_cell_size_pixels"] = int(
                settings.cell_size_pixels
            )
            mesh_audit["applied_cell_size_pixels"] = attempt_cell_size
            mesh_audit["adaptive_refinement_used"] = bool(
                attempt_cell_size != int(settings.cell_size_pixels)
            )
            mesh_audit["coarser_attempt_failures"] = list(
                attempt_failures
            )
            break
        else:
            raise RuntimeError(
                "Inspection identity owner inverse mesh failed for "
                f"group={int(owner.group_id)}, "
                f"structure={int(owner.structure_id)}, "
                f"frame={frame_id}: {'; '.join(attempt_failures)}"
            )
        sampled = accelerated_remap(
            np.asarray(source.image_bgr, dtype=np.uint8),
            np.where(local_footprint, map_x, -1.0).astype(np.float32),
            np.where(local_footprint, map_y, -1.0).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        target_scene_missing = 0
        target_scene_occluded = 0
        target_scene_audit: dict[str, object] | None = None
        visible_footprint = local_footprint.copy()
        if target_panel_index != source_panel_index:
            target_scene = sources_by_panel_index.get(target_panel_index)
            if target_scene is None:
                raise RuntimeError(
                    "Cross-panel identity owner lacks its target RGB-D panel"
                )
            cached_scene = target_scene_depth_cache.get(
                target_panel_index
            )
            if cached_scene is None:
                _, _, scene_depth, scene_audit = _project_source_grid(
                    owner=_ProjectionOwner(
                        panel_index=target_panel_index,
                        target_panel_index=target_panel_index,
                        source_mask=np.asarray(
                            target_scene.reliable_depth, dtype=bool
                        ),
                    ),
                    source=target_scene,
                    layout=layout,
                    intrinsics=intrinsics,
                    config=settings,
                )
                scene_corner_x = int(scene_audit["panel_corner_x"])
                cached_scene = (
                    scene_corner_x,
                    scene_depth,
                    scene_audit,
                )
                target_scene_depth_cache[target_panel_index] = cached_scene
            scene_corner_x, scene_depth, scene_audit = cached_scene
            if (
                scene_corner_x != corner_x
                or scene_depth.shape != identity_depth.shape
            ):
                raise RuntimeError(
                    "Cross-panel identity target z-buffer is misaligned"
                )
            scene_finite = np.isfinite(scene_depth)
            missing_scene = local_footprint & ~scene_finite
            target_scene_missing = int(np.count_nonzero(missing_scene))
            if target_scene_missing:
                raise RuntimeError(
                    "Cross-panel identity target scene z-buffer is missing "
                    f"{target_scene_missing} footprint pixels"
                )
            tolerance = np.maximum(
                np.float32(20.0),
                np.float32(0.02)
                * np.minimum(identity_depth, scene_depth),
            )
            occluded_mask = (
                local_footprint
                & scene_finite
                & (identity_depth > scene_depth + tolerance)
            )
            target_scene_occluded = int(
                np.count_nonzero(occluded_mask)
            )
            visible_footprint &= ~occluded_mask
            if not np.any(visible_footprint):
                raise RuntimeError(
                    "Cross-panel identity owner is fully occluded in the "
                    "target RGB-D panel"
                )
            target_scene_audit = {
                **scene_audit,
                "target_panel_index": target_panel_index,
                "missing_footprint_pixel_count": target_scene_missing,
                "occluded_footprint_pixel_count": target_scene_occluded,
                "visibility_tolerance": "max_20mm_2percent_nearer_depth",
            }
        # Background/reference-plane depth must not punch holes back through
        # an identity-owned object.  Two structures in the same audited group
        # can touch or occlude one another in the selected real view; resolve
        # only that overlap with their measured RGB-D z values.  Independently
        # planned groups are never allowed to compete silently.
        local_group = identity_group_buffer[:, corner_x:x1]
        cross_group_overlap = (
            visible_footprint
            & (local_group >= 0)
            & (local_group != int(owner.group_id))
        )
        if np.any(cross_group_overlap):
            raise RuntimeError(
                "Independent inspection identity owner groups overlap"
            )
        local_identity_depth = identity_depth_buffer[:, corner_x:x1]
        take = visible_footprint & (
            ~np.isfinite(local_identity_depth)
            | (identity_depth < local_identity_depth)
        )
        occluded = int(np.count_nonzero(visible_footprint & ~take))
        output_image[:, corner_x:x1][take] = sampled[take]
        output_depth[:, corner_x:x1][take] = identity_depth[take]
        output_confidence[:, corner_x:x1][take] = np.float32(1.0)
        output_owner[:, corner_x:x1][take] = frame_id
        output_reliable_depth[:, corner_x:x1][take] = True
        output_overlay_mask[:, corner_x:x1][take] = True
        local_identity_depth[take] = identity_depth[take]
        local_group[take] = int(owner.group_id)
        rows.append(
            {
                "group_id": int(owner.group_id),
                "structure_id": int(owner.structure_id),
                "source_panel_index": source_panel_index,
                "target_panel_index": target_panel_index,
                "frame_id": frame_id,
                "source_mask_pixel_count": int(
                    np.count_nonzero(owner.source_mask)
                ),
                "target_footprint_pixel_count": int(
                    np.count_nonzero(footprint)
                ),
                "written_pixel_count": int(np.count_nonzero(take)),
                "z_buffer_occluded_pixel_count": occluded,
                "target_scene_z_buffer_required": bool(
                    target_panel_index != source_panel_index
                ),
                "target_scene_missing_pixel_count": target_scene_missing,
                "target_scene_occluded_pixel_count": target_scene_occluded,
                "target_scene_z_buffer": target_scene_audit,
                "mesh": mesh_audit,
                "fill": fill_audit,
                "rgb_generated": False,
                "rgb_alpha_blended": False,
                "pose_modified": False,
                "position_model": (
                    "aligned_depth_camera_to_world_virtual_panel_projection"
                ),
            }
        )
    return {
        "schema": "inspection-identity-owner-inverse-mesh/v1",
        "policy": (
            "stable_instance_identity_then_true_rgbd_single_real_source_"
            "inverse_mesh_owner"
        ),
        "component_count": len(rows),
        "written_pixel_count": int(
            sum(int(item["written_pixel_count"]) for item in rows)
        ),
        "all_components_single_real_owner": True,
        "flow_used_to_warp_rgb_or_position": False,
        "rgb_generated": False,
        "rgb_alpha_blended": False,
        "pose_modified": False,
        "components": rows,
    }


__all__ = [
    "InspectionIdentityMeshConfig",
    "InspectionIdentityMeshSource",
    "composite_inspection_identity_owners",
]
