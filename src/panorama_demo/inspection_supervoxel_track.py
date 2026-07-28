"""Appearance-and-surface-gated world supervoxels for RGB-D diagnostics.

The segmentation is deliberately semantic-free.  Samples from every real
RGB-D view are fused into metric voxels, large structural planes are removed,
and neighbouring voxels connect only when Lab appearance, local normals, and
point-to-plane boundary evidence agree.  A connected world component is a
track; no image-space motion model or fitted display transform is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class WorldSupervoxelSegmentation:
    sample_track_id: np.ndarray
    voxel_track_id: np.ndarray
    voxel_points_world_mm: np.ndarray
    structural_plane_models_world: tuple[np.ndarray, ...]
    audit: dict[str, object]


def _feature_edge_is_safe(
    first: int,
    second: int,
    points: np.ndarray,
    lab: np.ndarray,
    normals: np.ndarray,
    normal_valid: np.ndarray,
    *,
    maximum_lab_delta: float,
    minimum_normal_dot: float,
    maximum_plane_residual_mm: float,
) -> bool:
    if float(np.linalg.norm(lab[first] - lab[second])) > maximum_lab_delta:
        return False
    if not (normal_valid[first] and normal_valid[second]):
        return False
    normal_dot = float(abs(normals[first] @ normals[second]))
    if normal_dot < minimum_normal_dot:
        return False
    delta = points[second] - points[first]
    if (
        abs(float(delta @ normals[first])) > maximum_plane_residual_mm
        or abs(float(delta @ normals[second])) > maximum_plane_residual_mm
    ):
        return False
    return True


def segment_world_supervoxels(
    *,
    points_world_mm: np.ndarray,
    lab: np.ndarray,
    normals_world: np.ndarray,
    normal_valid: np.ndarray,
    voxel_size_mm: float = 12.0,
    maximum_lab_delta: float = 32.0,
    maximum_normal_angle_degrees: float = 45.0,
    maximum_plane_residual_mm: float = 14.0,
    structural_plane_minimum_voxels: int = 2000,
    structural_plane_minimum_fraction: float = 0.05,
    structural_plane_distance_mm: float = 10.0,
    maximum_structural_planes: int = 4,
    remove_structural_planes: bool = True,
) -> WorldSupervoxelSegmentation:
    """Build deterministic world tracks from measured RGB-D samples."""

    points = np.asarray(points_world_mm, dtype=np.float64)
    colours = np.asarray(lab, dtype=np.float32)
    sample_normals = np.asarray(normals_world, dtype=np.float64)
    sample_normal_valid = np.asarray(normal_valid, dtype=bool)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or colours.shape != points.shape
        or sample_normals.shape != points.shape
        or sample_normal_valid.shape != (points.shape[0],)
        or not np.isfinite(points).all()
        or not np.isfinite(colours).all()
        or not math.isfinite(voxel_size_mm)
        or voxel_size_mm <= 0.0
    ):
        raise ValueError("World supervoxel inputs are invalid")
    if not 0.0 < maximum_normal_angle_degrees < 90.0:
        raise ValueError("World supervoxel normal-angle gate is invalid")
    if points.shape[0] == 0:
        return WorldSupervoxelSegmentation(
            sample_track_id=np.empty(0, dtype=np.int32),
            voxel_track_id=np.empty(0, dtype=np.int32),
            voxel_points_world_mm=np.empty((0, 3), dtype=np.float64),
            structural_plane_models_world=(),
            audit={
                "sample_count": 0,
                "voxel_count": 0,
                "track_count": 0,
            },
        )

    voxel_keys = np.floor(points / voxel_size_mm).astype(np.int32)
    unique_keys, inverse = np.unique(
        voxel_keys, axis=0, return_inverse=True
    )
    voxel_count = len(unique_keys)
    counts = np.bincount(inverse, minlength=voxel_count).astype(np.float64)
    voxel_points = np.column_stack(
        [
            np.bincount(
                inverse, weights=points[:, axis], minlength=voxel_count
            )
            / counts
            for axis in range(3)
        ]
    )
    voxel_lab = np.column_stack(
        [
            np.bincount(
                inverse, weights=colours[:, axis], minlength=voxel_count
            )
            / counts
            for axis in range(3)
        ]
    ).astype(np.float32)
    valid_weight = sample_normal_valid.astype(np.float64)
    valid_counts = np.bincount(
        inverse, weights=valid_weight, minlength=voxel_count
    )
    voxel_normals = np.column_stack(
        [
            np.bincount(
                inverse,
                weights=sample_normals[:, axis] * valid_weight,
                minlength=voxel_count,
            )
            for axis in range(3)
        ]
    )
    normal_length = np.linalg.norm(voxel_normals, axis=1)
    voxel_normal_valid = (
        (valid_counts >= np.maximum(1.0, 0.50 * counts))
        & (normal_length > 1e-8)
    )
    voxel_normals[voxel_normal_valid] /= normal_length[
        voxel_normal_valid, None
    ]
    voxel_normals[~voxel_normal_valid] = 0.0

    structural = np.zeros(voxel_count, dtype=bool)
    plane_models: list[np.ndarray] = []
    if remove_structural_planes and voxel_count >= 32:
        import open3d as o3d

        o3d.utility.random.seed(0)
        remaining = np.arange(voxel_count, dtype=np.int32)
        for _ in range(maximum_structural_planes):
            if remaining.size < structural_plane_minimum_voxels:
                break
            cloud = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(voxel_points[remaining])
            )
            model, local_inliers = cloud.segment_plane(
                distance_threshold=structural_plane_distance_mm,
                ransac_n=3,
                num_iterations=300,
            )
            required = max(
                structural_plane_minimum_voxels,
                int(
                    math.ceil(
                        structural_plane_minimum_fraction * remaining.size
                    )
                ),
            )
            if len(local_inliers) < required:
                break
            inliers = remaining[
                np.asarray(local_inliers, dtype=np.int32)
            ]
            spans = np.sort(np.ptp(voxel_points[inliers], axis=0))[::-1]
            if spans[0] < 350.0 or spans[1] < 120.0:
                break
            structural[inliers] = True
            plane_models.append(np.asarray(model, dtype=np.float64))
            remaining = np.flatnonzero(~structural)

    key_to_index = {
        (int(key[0]), int(key[1]), int(key[2])): index
        for index, key in enumerate(unique_keys)
        if not structural[index]
    }
    track = np.full(voxel_count, -1, dtype=np.int32)
    offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    )
    minimum_normal_dot = math.cos(
        math.radians(maximum_normal_angle_degrees)
    )
    component_sizes: list[int] = []
    for start in range(voxel_count):
        if structural[start] or track[start] >= 0:
            continue
        track_id = len(component_sizes)
        track[start] = track_id
        stack = [start]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            key = unique_keys[current]
            for dx, dy, dz in offsets:
                neighbour = key_to_index.get(
                    (
                        int(key[0]) + dx,
                        int(key[1]) + dy,
                        int(key[2]) + dz,
                    )
                )
                if neighbour is None or track[neighbour] >= 0:
                    continue
                if not _feature_edge_is_safe(
                    current,
                    neighbour,
                    voxel_points,
                    voxel_lab,
                    voxel_normals,
                    voxel_normal_valid,
                    maximum_lab_delta=maximum_lab_delta,
                    minimum_normal_dot=minimum_normal_dot,
                    maximum_plane_residual_mm=maximum_plane_residual_mm,
                ):
                    continue
                track[neighbour] = track_id
                stack.append(neighbour)
        component_sizes.append(size)
    return WorldSupervoxelSegmentation(
        sample_track_id=np.ascontiguousarray(track[inverse]),
        voxel_track_id=np.ascontiguousarray(track),
        voxel_points_world_mm=np.ascontiguousarray(voxel_points),
        structural_plane_models_world=tuple(plane_models),
        audit={
            "policy": (
                "all_real_rgbd_world_voxels_lab_local_normal_"
                "point_to_plane_boundary_supervoxel_graph"
            ),
            "sample_count": int(points.shape[0]),
            "voxel_size_mm": float(voxel_size_mm),
            "voxel_count": int(voxel_count),
            "structural_plane_count": len(plane_models),
            "structural_voxel_count": int(np.count_nonzero(structural)),
            "normal_reliable_voxel_count": int(
                np.count_nonzero(voxel_normal_valid)
            ),
            "maximum_lab_delta": float(maximum_lab_delta),
            "maximum_normal_angle_degrees": float(
                maximum_normal_angle_degrees
            ),
            "maximum_plane_residual_mm": float(
                maximum_plane_residual_mm
            ),
            "raw_track_count": len(component_sizes),
            "track_voxel_size_quantiles": (
                np.quantile(component_sizes, [0.5, 0.9, 0.99]).tolist()
                if component_sizes
                else [0.0, 0.0, 0.0]
            ),
        },
    )


__all__ = [
    "WorldSupervoxelSegmentation",
    "segment_world_supervoxels",
]
