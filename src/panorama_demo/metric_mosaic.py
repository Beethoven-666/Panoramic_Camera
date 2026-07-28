"""Geometry-faithful 2.5-D side-scan mosaic on a fixed metric grid.

This renderer is independent from the inspection panorama.  Every valid
output pixel carries one RGB source frame, one world-normal surface depth and
one auditable confidence value.  It never uses TSDF colour or fills a depth
hole from appearance.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .rgbd_projection import (
    PinholeIntrinsics,
    RGBDProjectionFrame,
    estimate_projection_canvas,
    prepare_rgbd_undistortion_maps,
    project_rgbd_source_compact,
)
from .session import (
    CameraIntrinsics,
    RGBDFrame,
    read_aligned_depth_mm,
)

METRIC_WORKING_BYTES_PER_CANVAS_PIXEL = 72
METRIC_MINIMUM_STRICT_CONTINUOUS_SURFACE_SUPPORT_RATIO = 0.01


@dataclass(frozen=True)
class MetricMosaicConfig:
    """Closed settings for the V1.0 geometry-analysis product."""

    enabled: bool = True
    millimetres_per_pixel: float = 2.0
    minimum_depth_mm: float = 200.0
    maximum_depth_mm: float = 3000.0
    preview_width: int = 320
    chunk_rows: int = 128
    maximum_canvas_megapixels: float = 200.0
    maximum_working_bytes: int = 4_000_000_000
    temporal_absolute_tolerance_mm: float = 20.0
    temporal_relative_tolerance: float = 0.02
    minimum_consistent_views: int = 2
    depth_edge_confidence_cap: float = 0.25

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None = None
    ) -> "MetricMosaicConfig":
        payload = {} if value is None else dict(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown metric_mosaic configuration keys: {unknown}")
        try:
            config = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid metric_mosaic configuration") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.enabled is not True:
            raise ValueError("Formal metric mosaic output cannot be disabled")
        finite_positive = (
            ("millimetres_per_pixel", self.millimetres_per_pixel),
            ("minimum_depth_mm", self.minimum_depth_mm),
            ("maximum_depth_mm", self.maximum_depth_mm),
            ("maximum_canvas_megapixels", self.maximum_canvas_megapixels),
            (
                "temporal_absolute_tolerance_mm",
                self.temporal_absolute_tolerance_mm,
            ),
            ("temporal_relative_tolerance", self.temporal_relative_tolerance),
        )
        for name, value in finite_positive:
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"metric_mosaic.{name} must be finite and positive")
        if self.millimetres_per_pixel != 2.0:
            raise ValueError("Formal metric mosaic resolution is fixed at 2 mm/pixel")
        if self.maximum_depth_mm <= self.minimum_depth_mm:
            raise ValueError("metric_mosaic depth range is empty")
        if self.preview_width < 64:
            raise ValueError("metric_mosaic.preview_width must be at least 64")
        if self.chunk_rows <= 0:
            raise ValueError("metric_mosaic.chunk_rows must be positive")
        if self.maximum_working_bytes <= 0:
            raise ValueError("metric_mosaic.maximum_working_bytes must be positive")
        if self.minimum_consistent_views < 1:
            raise ValueError(
                "metric_mosaic.minimum_consistent_views must be positive"
            )
        if not 0.0 < self.depth_edge_confidence_cap <= 1.0:
            raise ValueError(
                "metric_mosaic.depth_edge_confidence_cap must be in (0, 1]"
            )


@dataclass(frozen=True)
class MetricMosaicResult:
    """Aligned products for one fixed-resolution metric mosaic."""

    image_bgr: np.ndarray
    depth_mm: np.ndarray
    confidence_u16: np.ndarray
    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, object]

    def validate(self) -> None:
        shape = self.image_bgr.shape[:2]
        if self.image_bgr.dtype != np.uint8 or self.image_bgr.shape != (*shape, 3):
            raise RuntimeError("Metric RGB must be an HxWx3 uint8 image")
        if (
            self.depth_mm.shape != shape
            or self.depth_mm.dtype != np.float32
            or self.confidence_u16.shape != shape
            or self.confidence_u16.dtype != np.uint16
            or self.owner_frame_id.shape != shape
            or self.owner_frame_id.dtype != np.int32
            or self.valid_mask.shape != shape
        ):
            raise RuntimeError("Metric mosaic sidecars are not pixel-aligned")
        valid = np.asarray(self.valid_mask, dtype=bool)
        if not np.any(valid):
            raise RuntimeError("Metric mosaic contains no valid RGB-D surface")
        if np.any(~np.isfinite(self.depth_mm[valid])):
            raise RuntimeError("Metric valid pixels contain non-finite depth")
        if np.any(np.isfinite(self.depth_mm[~valid])):
            raise RuntimeError("Metric invalid depth must be NaN")
        if np.any(self.confidence_u16[valid] == 0) or np.any(
            self.confidence_u16[~valid] != 0
        ):
            raise RuntimeError("Metric confidence validity contract failed")
        if np.any(self.owner_frame_id[valid] < 0) or np.any(
            self.owner_frame_id[~valid] != -1
        ):
            raise RuntimeError("Metric owner validity contract failed")
        if self.metadata.get("schema") == "gemini305-metric-mosaic/v1":
            strict_complete = self.metadata.get("strict_v1_metric_complete")
            strict_reasons = self.metadata.get("strict_incomplete_reasons")
            if type(strict_complete) is not bool:
                raise RuntimeError(
                    "Metric mosaic omitted strict_v1_metric_complete"
                )
            if not isinstance(strict_reasons, list) or any(
                not isinstance(reason, str) or not reason
                for reason in strict_reasons
            ):
                raise RuntimeError(
                    "Metric mosaic strict_incomplete_reasons are malformed"
                )
            if strict_complete and strict_reasons:
                raise RuntimeError(
                    "Metric mosaic claims strict completion with failure reasons"
                )


def _pinhole(intrinsics: CameraIntrinsics) -> PinholeIntrinsics:
    return PinholeIntrinsics(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        distortion=intrinsics.distortion,
    )


def _read_bgr(path: object) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode metric RGB source: {path}")
    return image


def _preview_projection_frames(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    width: int,
) -> tuple[list[RGBDProjectionFrame], PinholeIntrinsics]:
    target_width = min(intrinsics.width, int(width))
    scale_x = target_width / float(intrinsics.width)
    target_height = max(1, int(round(intrinsics.height * scale_x)))
    scale_y = target_height / float(intrinsics.height)
    camera = PinholeIntrinsics(
        width=target_width,
        height=target_height,
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x,
        cy=intrinsics.cy * scale_y,
        distortion=intrinsics.distortion,
    )
    placeholder = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    result: list[RGBDProjectionFrame] = []
    for frame, pose in zip(frames, poses, strict=True):
        depth = read_aligned_depth_mm(frame)
        if depth.shape != (target_height, target_width):
            depth = cv2.resize(
                depth,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        result.append(
            RGBDProjectionFrame(
                frame_id=frame.frame_id,
                rgb=placeholder,
                depth_mm=np.ascontiguousarray(depth, dtype=np.float32),
                camera_to_world=np.asarray(pose, dtype=np.float64),
            )
        )
    return result, camera


def _projected_source_confidence(
    camera_depth_mm: np.ndarray,
    valid_mask: np.ndarray,
    config: MetricMosaicConfig,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.isfinite(camera_depth_mm)
        & (camera_depth_mm >= config.minimum_depth_mm)
        & (camera_depth_mm <= config.maximum_depth_mm)
    )
    confidence = np.zeros(camera_depth_mm.shape, dtype=np.float32)
    if not np.any(valid):
        return confidence, np.zeros_like(valid)

    depth = np.asarray(camera_depth_mm, dtype=np.float32)
    near_taper = max(50.0, 0.10 * config.minimum_depth_mm)
    far_taper = max(200.0, 0.10 * config.maximum_depth_mm)
    range_score = np.minimum(
        np.clip((depth - config.minimum_depth_mm) / near_taper, 0.0, 1.0),
        np.clip((config.maximum_depth_mm - depth) / far_taper, 0.0, 1.0),
    )
    neighbour_count = cv2.boxFilter(
        valid.astype(np.float32),
        ddepth=-1,
        ksize=(3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    local_score = np.clip(neighbour_count / 9.0, 0.0, 1.0)

    high = np.where(valid, depth, 0.0).astype(np.float32)
    low_sentinel = np.float32(config.maximum_depth_mm * 2.0)
    low = np.where(valid, depth, low_sentinel).astype(np.float32)
    local_high = cv2.dilate(high, np.ones((3, 3), np.uint8))
    local_low = cv2.erode(low, np.ones((3, 3), np.uint8))
    local_span = local_high - local_low
    local_span[local_low >= low_sentinel] = 0.0
    edge_tolerance = np.maximum(
        config.temporal_absolute_tolerance_mm,
        config.temporal_relative_tolerance * np.maximum(depth, 0.0),
    )
    depth_edge = valid & (local_span > edge_tolerance)

    confidence[valid] = np.minimum(
        range_score[valid],
        0.25 + 0.75 * local_score[valid],
    )
    confidence[depth_edge] = np.minimum(
        confidence[depth_edge],
        config.depth_edge_confidence_cap,
    )
    confidence[valid & (confidence <= 0.0)] = 1.0 / 65535.0
    return confidence, depth_edge


def _temporal_depth_tolerance(
    source_camera_depth_mm: np.ndarray,
    winner_camera_depth_mm: np.ndarray,
    config: MetricMosaicConfig,
) -> np.ndarray:
    """Use local camera range, never world-origin surface coordinates."""

    source_range = np.asarray(source_camera_depth_mm, dtype=np.float32)
    winner_range = np.asarray(winner_camera_depth_mm, dtype=np.float32)
    comparison_range = np.where(
        np.isfinite(winner_range), winner_range, source_range
    )
    return np.maximum(
        np.float32(config.temporal_absolute_tolerance_mm),
        np.float32(config.temporal_relative_tolerance)
        * np.maximum(source_range, comparison_range),
    )


def _crop_to_valid_bbox(
    arrays: Sequence[np.ndarray],
    valid: np.ndarray,
) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
    rows, columns = np.nonzero(valid)
    if not columns.size:
        raise RuntimeError("Metric mosaic has no valid crop")
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    return [array[y0:y1, x0:x1] for array in arrays], (
        x0,
        y0,
        x1 - x0,
        y1 - y0,
    )


def render_metric_mosaic(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    config: MetricMosaicConfig | Mapping[str, object] | None = None,
) -> MetricMosaicResult:
    """Render one fixed 2 mm/px metric product from real RGB-D poses."""

    selected = (
        config
        if isinstance(config, MetricMosaicConfig)
        else MetricMosaicConfig.from_mapping(config)
    )
    selected.validate()
    if len(frames) < 2 or len(frames) != len(poses):
        raise ValueError("Metric mosaic requires at least two aligned frames and poses")
    frame_ids = [int(frame.frame_id) for frame in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("Metric mosaic frame IDs must be unique")
    if any(frame_id < 0 or frame_id > 65534 for frame_id in frame_ids):
        raise ValueError("Metric owner PNG requires frame IDs in [0, 65534]")

    previews, preview_intrinsics = _preview_projection_frames(
        frames,
        poses,
        intrinsics,
        selected.preview_width,
    )
    canvas = estimate_projection_canvas(
        previews,
        preview_intrinsics,
        max_canvas_megapixels=selected.maximum_canvas_megapixels,
        max_aggregate_megapixels=selected.maximum_canvas_megapixels,
        maximum_depth_mm=selected.maximum_depth_mm,
        millimetres_per_pixel=selected.millimetres_per_pixel,
        maximum_resident_sources=1,
    )
    del previews
    pixel_count = int(canvas.width * canvas.height)
    # Includes the resident merge state, scalar audit temporaries and the
    # final contiguous crop copies.  Compact per-source projections are
    # bounded separately by one source bbox and released after every merge.
    estimated_peak_bytes = (
        pixel_count * METRIC_WORKING_BYTES_PER_CANVAS_PIXEL
    )
    if estimated_peak_bytes > selected.maximum_working_bytes:
        raise MemoryError(
            "Metric mosaic estimated working set exceeds its byte budget: "
            f"{estimated_peak_bytes} > {selected.maximum_working_bytes}"
        )

    image = np.zeros((canvas.height, canvas.width, 3), dtype=np.uint8)
    depth = np.full((canvas.height, canvas.width), np.inf, dtype=np.float32)
    winner_camera_depth = np.full(
        (canvas.height, canvas.width), np.nan, dtype=np.float32
    )
    confidence = np.zeros((canvas.height, canvas.width), dtype=np.float32)
    owner = np.full((canvas.height, canvas.width), -1, dtype=np.int32)
    support_count = np.zeros((canvas.height, canvas.width), dtype=np.uint16)
    conflict_count = np.zeros((canvas.height, canvas.width), dtype=np.uint16)
    edge_winner = np.zeros((canvas.height, canvas.width), dtype=bool)
    full_intrinsics = _pinhole(intrinsics)
    prepared_undistortion_maps = prepare_rgbd_undistortion_maps(
        full_intrinsics
    )
    source_projection_audit: list[dict[str, object]] = []

    for frame, pose in zip(frames, poses, strict=True):
        source = project_rgbd_source_compact(
            RGBDProjectionFrame(
                frame_id=frame.frame_id,
                rgb=_read_bgr(frame.color_path),
                depth_mm=read_aligned_depth_mm(frame),
                camera_to_world=np.asarray(pose, dtype=np.float64),
            ),
            full_intrinsics,
            canvas,
            chunk_rows=selected.chunk_rows,
            maximum_depth_mm=selected.maximum_depth_mm,
            prepared_undistortion_maps=prepared_undistortion_maps,
        )
        region = source.canvas_slices
        image_region = image[region]
        depth_region = depth[region]
        winner_camera_depth_region = winner_camera_depth[region]
        confidence_region = confidence[region]
        owner_region = owner[region]
        support_count_region = support_count[region]
        conflict_count_region = conflict_count[region]
        edge_winner_region = edge_winner[region]
        source_confidence, source_edge = _projected_source_confidence(
            source.camera_depth_mm,
            source.valid_mask,
            selected,
        )
        candidate = source_confidence > 0.0
        tolerance = _temporal_depth_tolerance(
            source.camera_depth_mm,
            winner_camera_depth_region,
            selected,
        )
        empty = candidate & (owner_region < 0)
        overlap = candidate & (owner_region >= 0)
        delta = source.surface_depth_mm - depth_region
        same_layer = overlap & (np.abs(delta) <= tolerance)
        nearer = overlap & (delta < -tolerance)
        farther = overlap & (delta > tolerance)
        support_count_region[same_layer] = np.minimum(
            support_count_region[same_layer].astype(np.uint32) + 1,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        conflict_count_region[nearer | farther] = np.minimum(
            conflict_count_region[nearer | farther].astype(np.uint32) + 1,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        better_same_layer = same_layer & (
            source_confidence > confidence_region + np.float32(1e-6)
        )
        take = empty | nearer | better_same_layer
        if np.any(take):
            image_region[take] = source.warped_rgb[take]
            depth_region[take] = source.surface_depth_mm[take]
            winner_camera_depth_region[take] = source.camera_depth_mm[take]
            confidence_region[take] = source_confidence[take]
            owner_region[take] = int(frame.frame_id)
            edge_winner_region[take] = source_edge[take]
            support_count_region[empty | nearer] = 1
        source_projection_audit.append(
            {
                **source.as_dict(),
                "candidate_pixel_count": int(np.count_nonzero(candidate)),
                "same_layer_pixel_count": int(np.count_nonzero(same_layer)),
                "nearer_replacement_pixel_count": int(np.count_nonzero(nearer)),
                "farther_occluded_pixel_count": int(np.count_nonzero(farther)),
                "depth_edge_candidate_pixel_count": int(
                    np.count_nonzero(source_edge)
                ),
            }
        )
        del source

    footprint_count_fields = (
        "measured_center_candidate_count",
        "continuous_surface_sample_count",
        "footprint_candidate_count",
        "point_center_selected_zbuffer_pixel_count",
        "footprint_rasterized_pixel_count",
        "rejected_invalid_neighbourhood_sample_count",
        "rejected_depth_edge_sample_count",
        "rejected_fold_sample_count",
        "rejected_degenerate_sample_count",
        "rejected_overscale_sample_count",
        "unobserved_output_pixel_count",
    )
    footprint_flag_fields = (
        "point_centres_preserved",
        "nearest_measured_rgb_only",
        "nearest_measured_depth_only",
        "surface_footprint_continuity_gate_used",
        "surface_footprint_positive_jacobian_required",
    )
    footprint_totals = {name: 0 for name in footprint_count_fields}
    for audit in source_projection_audit:
        sampling = audit.get("sampling_stats")
        if not isinstance(sampling, Mapping):
            raise RuntimeError("Metric source omitted projection sampling audit")
        for name in footprint_count_fields:
            value = sampling.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"Metric source has invalid footprint audit field: {name}"
                )
            footprint_totals[name] += value
        if any(sampling.get(name) is not True for name in footprint_flag_fields):
            raise RuntimeError(
                "Metric source violated measured surface-footprint provenance"
            )
        if sampling.get("morphological_hole_fill_used") is not False:
            raise RuntimeError("Metric source used a prohibited hole fill")
        if sampling.get("point_splat_only") is not False:
            raise RuntimeError(
                "Metric source did not expose its conservative pixel footprint policy"
            )
    measured_centre_count = footprint_totals[
        "measured_center_candidate_count"
    ]
    continuous_support_count = footprint_totals[
        "continuous_surface_sample_count"
    ]
    continuous_support_ratio = float(
        continuous_support_count / max(1, measured_centre_count)
    )
    strict_metric_reasons: list[str] = []
    if (
        continuous_support_ratio
        < METRIC_MINIMUM_STRICT_CONTINUOUS_SURFACE_SUPPORT_RATIO
    ):
        strict_metric_reasons.append(
            "accepted depth-continuous measured surface support is below "
            f"{METRIC_MINIMUM_STRICT_CONTINUOUS_SURFACE_SUPPORT_RATIO:.3f}"
        )
    strict_v1_metric_complete = not strict_metric_reasons

    valid = owner >= 0
    if not np.any(valid):
        raise RuntimeError("Metric RGB-D reprojection produced no owned surface")
    temporal_score = np.clip(
        support_count.astype(np.float32)
        / float(selected.minimum_consistent_views),
        1.0 / float(selected.minimum_consistent_views),
        1.0,
    )
    conflict_score = 1.0 / (
        1.0 + 0.25 * conflict_count.astype(np.float32)
    )
    confidence *= temporal_score * conflict_score
    confidence[edge_winner] = np.minimum(
        confidence[edge_winner],
        selected.depth_edge_confidence_cap,
    )
    confidence[valid & (confidence <= 0.0)] = 1.0 / 65535.0
    depth[~valid] = np.nan
    confidence_u16 = np.zeros(confidence.shape, dtype=np.uint16)
    confidence_u16[valid] = np.maximum(
        1,
        np.rint(np.clip(confidence[valid], 0.0, 1.0) * 65535.0),
    ).astype(np.uint16)

    (
        cropped,
        crop,
    ) = _crop_to_valid_bbox(
        (image, depth, confidence_u16, owner, valid),
        valid,
    )
    image_crop, depth_crop, confidence_crop, owner_crop, valid_crop = cropped
    x, y, width, height = crop
    source_owner_counts = {
        str(frame_id): int(np.count_nonzero(owner_crop == frame_id))
        for frame_id in frame_ids
    }
    normal_axis = np.asarray(canvas.normal_axis, dtype=np.float64)
    scan_axis = np.asarray(canvas.scan_axis, dtype=np.float64)
    up_axis = np.asarray(canvas.up_axis, dtype=np.float64)
    min_scan, min_down, _, _ = canvas.world_bounds
    cropped_scan_origin_mm = (
        min_scan + x * selected.millimetres_per_pixel
    )
    cropped_down_origin_mm = (
        min_down + y * selected.millimetres_per_pixel
    )
    metadata: dict[str, object] = {
        "schema": "gemini305-metric-mosaic/v1",
        "method": (
            "trajectory_constrained_depth_aware_metric_2_5d_side_scan"
        ),
        "coordinate_system": {
            "world_unit": "mm",
            "pixel_size_mm": selected.millimetres_per_pixel,
            "scan_axis_world": scan_axis.tolist(),
            "up_axis_world": up_axis.tolist(),
            "normal_axis_world": normal_axis.tolist(),
            "canvas_x": "dot(world_point_mm, scan_axis_world)",
            "canvas_y": "dot(world_point_mm, -up_axis_world)",
            "depth": "dot(world_point_mm, normal_axis_world)",
            "depth_unit": "mm",
            "crop_scan_origin_mm": float(cropped_scan_origin_mm),
            "crop_down_origin_mm": float(cropped_down_origin_mm),
            "axis_handedness": "scan_cross_up_opposes_or_matches_normal_by_camera_forward",
        },
        "canvas": canvas.as_dict(),
        "crop": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "config": asdict(selected),
        "rgb_policy": "single_real_rgb_owner_no_tsdf_colour",
        "geometry_policy": (
            "depth_continuity_gated_measured_pixel_footprint_"
            "world_normal_zbuffer_no_cross_edge_fill"
        ),
        "background_policy": (
            "metric_hard_owner_only_inspection_renderer_handles_safe_multiband"
        ),
        "confidence_model": {
            "components": [
                "depth_range_taper",
                "3x3_spatial_support",
                "depth_edge_cap",
                "multi_view_same_layer_support",
                "occlusion_conflict_penalty",
                "local_camera_range_relative_tolerance",
            ],
            "combination": "conservative_product_with_edge_cap",
            "invalid_value": 0,
            "encoding": "uint16_round(confidence_0_1_times_65535)",
        },
        "owner_encoding": {
            "internal": "int32_frame_id_minus_one_invalid",
            "png": "uint16_frame_id_plus_one_zero_invalid",
        },
        "invalid_semantics": {
            "rgb": "undefined_stored_as_zero",
            "depth": "quiet_NaN",
            "confidence": 0,
            "owner_frame_id": -1,
        },
        "surface_footprint_audit": {
            "policy": (
                "complete_3x3_measured_same_depth_layer_positive_jacobian_"
                "bounded_sensor_pixel_footprint"
            ),
            "rgb_sampling": "nearest_real_source_pixel_copy_no_interpolation",
            "depth_sampling": (
                "nearest_real_source_camera_and_world_normal_depth_copy"
            ),
            "point_centres_preserved": True,
            "world_normal_zbuffer_preserved": True,
            "morphological_hole_fill_used": False,
            "invalid_depth_crossing_allowed": False,
            "depth_edge_crossing_allowed": False,
            "fold_crossing_allowed": False,
            **footprint_totals,
            "accepted_continuous_surface_support_ratio": (
                continuous_support_ratio
            ),
            "minimum_strict_continuous_surface_support_ratio": (
                METRIC_MINIMUM_STRICT_CONTINUOUS_SURFACE_SUPPORT_RATIO
            ),
        },
        "strict_v1_metric_complete": strict_v1_metric_complete,
        "strict_incomplete_reasons": strict_metric_reasons,
        "frame_count": len(frames),
        "frame_ids": frame_ids,
        "source_owner_pixel_counts": source_owner_counts,
        "source_projection_audit": source_projection_audit,
        "valid_pixel_count": int(np.count_nonzero(valid_crop)),
        "invalid_pixel_count": int(valid_crop.size - np.count_nonzero(valid_crop)),
        "single_owner_valid_pixel_count": int(np.count_nonzero(owner_crop >= 0)),
        "unowned_valid_pixel_count": int(
            np.count_nonzero(valid_crop & (owner_crop < 0))
        ),
        "depth_edge_owner_pixel_count": int(
            np.count_nonzero(edge_winner[y : y + height, x : x + width])
        ),
        "minimum_temporal_support": int(np.min(support_count[valid])),
        "maximum_temporal_support": int(
            np.max(support_count[valid], initial=0)
        ),
        "estimated_peak_bytes": estimated_peak_bytes,
        "estimated_peak_bytes_per_canvas_pixel": (
            METRIC_WORKING_BYTES_PER_CANVAS_PIXEL
        ),
        "maximum_resident_projection_sources": 1,
        "tsdf_used_for_rgb": False,
        "tsdf_used_for_depth": False,
        "temporal_tolerance_range_basis": (
            "max(source_camera_z_mm,winner_camera_z_mm)"
        ),
    }
    result = MetricMosaicResult(
        image_bgr=np.ascontiguousarray(image_crop),
        depth_mm=np.ascontiguousarray(depth_crop, dtype=np.float32),
        confidence_u16=np.ascontiguousarray(confidence_crop, dtype=np.uint16),
        owner_frame_id=np.ascontiguousarray(owner_crop, dtype=np.int32),
        valid_mask=np.ascontiguousarray(valid_crop, dtype=bool),
        metadata=metadata,
    )
    result.validate()
    return result
