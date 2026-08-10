"""Stage-A-only source and RGB-D audit helpers for the standalone S1.2 route."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .rgbd_odometry import (
    PoseQualityThresholds,
    RGBDOdometryConfig,
    create_open3d_rgbd_odometry_backend,
    estimate_pair_rgbd_odometry,
)


def percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"p50": 0.0, "p95": 0.0, "maximum": 0.0}
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


@dataclass(frozen=True)
class S12SelectedSource:
    source_index: int
    frame_id: int
    center_x: float
    anchor: bool
    image_quality: float
    zero_width: bool = False


def select_unique_sources(
    frame_ids: Sequence[int],
    centers: Sequence[float],
    *,
    anchor_frame_ids: Sequence[int],
    image_quality_by_frame: Mapping[int, float] | None = None,
    duplicate_tolerance_px: float = 0.5,
) -> tuple[S12SelectedSource, ...]:
    """Remove only sub-pixel duplicate centres with deterministic priorities."""

    if len(frame_ids) != len(centers) or not frame_ids:
        raise ValueError("S1.2 source ids and centres must be non-empty and equal length")
    values = np.asarray(centers, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0.0):
        raise ValueError("S1.2 candidate source centres must be finite and increasing")
    anchors = {int(value) for value in anchor_frame_ids}
    qualities = image_quality_by_frame or {}
    kept: list[tuple[int, float]] = []
    for frame_id, center in zip(frame_ids, values, strict=True):
        item = (int(frame_id), float(center))
        if not kept or center - kept[-1][1] >= duplicate_tolerance_px:
            kept.append(item)
            continue
        previous_id, previous_center = kept[-1]
        previous_key = (
            previous_id in anchors,
            float(qualities.get(previous_id, 0.0)),
            -previous_id,
        )
        current_key = (
            int(frame_id) in anchors,
            float(qualities.get(int(frame_id), 0.0)),
            -int(frame_id),
        )
        if current_key > previous_key:
            kept[-1] = item
        else:
            # Keep the measured centre belonging to the retained real frame;
            # never average or interpolate duplicate positions.
            kept[-1] = (previous_id, previous_center)
    if not anchors.issubset({frame_id for frame_id, _ in kept}):
        raise ValueError("S1.2 duplicate pruning removed an automatic anchor")
    return tuple(
        S12SelectedSource(
            source_index=index,
            frame_id=frame_id,
            center_x=center,
            anchor=frame_id in anchors,
            image_quality=float(qualities.get(frame_id, 0.0)),
        )
        for index, (frame_id, center) in enumerate(kept)
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def audit_open3d_source_edges(
    frames: Sequence[Any],
    camera_to_world: Sequence[np.ndarray],
    intrinsics: Any,
    *,
    odometry_config: RGBDOdometryConfig | Mapping[str, Any] | None = None,
    thresholds: PoseQualityThresholds | None = None,
    backend: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Audit every final neighbour without modifying its immutable ORB pose."""

    if len(frames) != len(camera_to_world) or len(frames) < 2:
        raise ValueError("S1.2 Open3D audit needs equal frame/pose chains of length >= 2")
    config = (
        odometry_config
        if isinstance(odometry_config, RGBDOdometryConfig)
        else RGBDOdometryConfig.from_mapping(odometry_config)
    )
    limits = thresholds or PoseQualityThresholds()
    selected_backend = backend or create_open3d_rgbd_odometry_backend()
    rows: list[dict[str, Any]] = []
    for left_frame, right_frame, left_pose_value, right_pose_value in zip(
        frames[:-1], frames[1:], camera_to_world[:-1], camera_to_world[1:], strict=True
    ):
        left_pose = np.asarray(left_pose_value, dtype=np.float64)
        right_pose = np.asarray(right_pose_value, dtype=np.float64)
        orb_source_to_reference = np.linalg.inv(left_pose) @ right_pose
        edge = estimate_pair_rgbd_odometry(
            left_frame,
            right_frame,
            intrinsics,
            config=config,
            backend=selected_backend,
            reference_node_id=int(left_frame.frame_id),
            source_node_id=int(right_frame.frame_id),
            initial_source_to_reference=orb_source_to_reference,
        )
        delta = np.linalg.inv(edge.source_to_reference) @ orb_source_to_reference
        translation_residual_mm = float(np.linalg.norm(delta[:3, 3]))
        rotation_residual_deg = _rotation_angle_deg(delta[:3, :3])
        residual_pass = (
            translation_residual_mm <= limits.maximum_edge_translation_residual_mm
            and rotation_residual_deg <= limits.maximum_edge_rotation_residual_deg
        )
        backend_pass = edge.backend == "open3d_tensor_cuda_rgbd"
        quality_pass = bool(edge.reliable and residual_pass and backend_pass)
        rows.append(
            {
                **edge.as_dict(),
                "orb_source_to_reference": orb_source_to_reference.tolist(),
                "orb_translation_residual_mm": translation_residual_mm,
                "orb_rotation_residual_deg": rotation_residual_deg,
                "residual_pass": residual_pass,
                "required_backend": "open3d_tensor_cuda_rgbd",
                "backend_pass": backend_pass,
                "quality_pass": quality_pass,
                "rgb_generation_allowed": False,
                "pose_replacement_allowed": False,
                "tsdf_allowed": False,
            }
        )
    return tuple(rows)


__all__ = [
    "S12SelectedSource",
    "audit_open3d_source_edges",
    "percentiles",
    "select_unique_sources",
]
