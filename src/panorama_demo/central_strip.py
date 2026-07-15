"""Reference-plane central-strip diagnostic renderer.

This module is intentionally a diagnostic-only backend.  It maps a narrow,
calibrated central field of every selected real camera pose onto one measured
reference plane, then delegates colour ownership and local blending to the
shared fail-closed prewarped renderer.  It never creates a 2-D homography,
interpolates a pose, or fills a depth hole with the reference-plane distance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .calibrated_remap import (
    camera_points_to_source_pixels,
    undistort_depth_with_validity,
    undistortion_maps,
)
from .render import PrewarpedScanSource, render_prewarped_scan_panorama
from .rgbd_projection import estimate_world_axes, validate_camera_to_world
from .session import CameraIntrinsics, RGBDFrame, read_aligned_depth_mm


_HARD_MAX_POSE_NODES = 160
_HARD_MAX_RENDER_SOURCES = 32
_HARD_MAX_CANVAS_MEGAPIXELS = 200.0
_DEFAULT_MAX_AGGREGATE_MEGAPIXELS = 80.0
_PLANE_SAMPLE_LIMIT_PER_FRAME = 16_000
_PLANE_HISTOGRAM_BIN_MM = 20.0
# These are diagnostic admission criteria, not tunable user parameters.  The
# first set says whether measured depth can structurally support one *unique*
# plane.  The stricter quality set merely forces ``strip_quality_pass=false``;
# a structurally valid diagnostic may still be published for A/B inspection.
_PLANE_STRUCTURAL_MAX_P50_MM = 40.0
_PLANE_STRUCTURAL_MAX_P95_MM = 80.0
_PLANE_QUALITY_MAX_P50_MM = 25.0
_PLANE_QUALITY_MAX_P95_MM = 50.0
_PLANE_MIN_GLOBAL_INLIER_FRACTION = 0.25
_PLANE_MIN_CALIBRATED_AREA_FRACTION = 0.10
_PLANE_MIN_FRAME_COVERAGE = 0.05
_PLANE_MIN_SUPPORTING_FRAME_FRACTION = 0.80
_PLANE_CANDIDATE_NORMAL_DEDUP_DEG = 2.0
_PLANE_CANDIDATE_OFFSET_DEDUP_FLOOR_MM = 60.0
_PLANE_COMPETING_SCORE_RATIO = 0.30
_PLANE_MIN_COMPETING_GLOBAL_FRACTION = 0.10

_CENTRAL_STRIP_DEFAULTS: dict[str, object] = {
    "enabled": False,
    "reference_scale_mode": "robust_aligned_depth_plane",
    "orientation_mode": "verified_camera_to_world",
    "maximum_central_band_fraction": 0.20,
    "minimum_pair_overlap_pixels": 96,
    "exposure_mode": "global_gain",
    "multiband_levels": 5,
}


@dataclass(frozen=True)
class ScanAxes:
    """Right-handed world axes inferred only from real ``camera_to_world`` poses."""

    scan_axis: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    normal_axis: tuple[float, float, float]

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "scan_axis": list(self.scan_axis),
            "up_axis": list(self.up_axis),
            "normal_axis": list(self.normal_axis),
        }


@dataclass(frozen=True)
class ReferencePlane:
    """Measured background plane ``normal_world · X = offset_mm``."""

    normal_world: tuple[float, float, float]
    offset_mm: float
    origin_world_mm: tuple[float, float, float]
    inlier_count: int
    sampled_point_count: int
    inlier_fraction: float
    residual_p50_mm: float
    residual_p95_mm: float
    normal_view_angle_deg: float
    camera_distances_mm: tuple[float, ...]
    measured_depth_validation_coverage: float
    measured_depth_sampling_fraction: float
    per_frame_depth_coverage: tuple[float, ...]
    per_frame_calibrated_area_fraction: tuple[float, ...]
    supporting_frame_count: int
    supporting_frame_fraction: float
    maximum_unsupported_frame_run: int
    fit_quality_pass: bool
    candidate_count: int
    distinct_candidate_count: int
    runner_score_ratio: float

    def as_dict(self) -> dict[str, object]:
        return {
            "equation": {
                "normal_world": list(self.normal_world),
                "offset_mm": self.offset_mm,
                "form": "normal_world_dot_X_mm_equals_offset_mm",
            },
            "origin_world_mm": list(self.origin_world_mm),
            "sampled_point_count": self.sampled_point_count,
            "inlier_count": self.inlier_count,
            "inlier_fraction": self.inlier_fraction,
            "residual_p50_mm": self.residual_p50_mm,
            "residual_p95_mm": self.residual_p95_mm,
            "normal_view_angle_deg": self.normal_view_angle_deg,
            "camera_distances_mm": list(self.camera_distances_mm),
            "measured_depth_validation_coverage": self.measured_depth_validation_coverage,
            "measured_depth_sampling_fraction": self.measured_depth_sampling_fraction,
            "per_frame_depth_validation_coverage": list(self.per_frame_depth_coverage),
            "per_frame_calibrated_area_fraction": list(
                self.per_frame_calibrated_area_fraction
            ),
            "supporting_frame_count": self.supporting_frame_count,
            "supporting_frame_fraction": self.supporting_frame_fraction,
            "maximum_unsupported_frame_run": self.maximum_unsupported_frame_run,
            "quality": {
                "fit_quality_pass": self.fit_quality_pass,
                "selection_policy": "unique_dominant_measured_plane",
                "candidate_count": self.candidate_count,
                "distinct_candidate_count": self.distinct_candidate_count,
                "runner_score_ratio": self.runner_score_ratio,
                "structural_min_global_inlier_fraction": _PLANE_MIN_GLOBAL_INLIER_FRACTION,
                "structural_min_calibrated_area_fraction": _PLANE_MIN_CALIBRATED_AREA_FRACTION,
                "structural_max_residual_p50_mm": _PLANE_STRUCTURAL_MAX_P50_MM,
                "structural_max_residual_p95_mm": _PLANE_STRUCTURAL_MAX_P95_MM,
                "quality_max_residual_p50_mm": _PLANE_QUALITY_MAX_P50_MM,
                "quality_max_residual_p95_mm": _PLANE_QUALITY_MAX_P95_MM,
            },
        }


@dataclass(frozen=True)
class CentralStripLayout:
    """Plane canvas and real trajectory-derived central-strip ownership."""

    width: int
    height: int
    pixels_per_mm: float
    scan_min_mm: float
    scan_max_mm: float
    up_min_mm: float
    up_max_mm: float
    source_scan_positions_mm: tuple[float, ...]
    source_support_intervals_mm: tuple[tuple[float, float], ...]
    owner_intervals_mm: tuple[tuple[float, float], ...]
    owner_boundaries_x: tuple[float, ...]
    source_center_xy: tuple[tuple[float, float], ...]
    pair_overlap_pixels: tuple[float, ...]
    canvas_megapixels: float
    aggregate_megapixels: float

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinate_system": {
                "world_unit": "mm",
                "canvas_x": "dot(world_point - plane_origin, scan_axis)",
                "canvas_y": "dot(world_point - plane_origin, up_axis), top_down",
            },
            "width": self.width,
            "height": self.height,
            "pixels_per_mm": self.pixels_per_mm,
            "scan_bounds_mm": [self.scan_min_mm, self.scan_max_mm],
            "up_bounds_mm": [self.up_min_mm, self.up_max_mm],
            "source_scan_positions_mm": list(self.source_scan_positions_mm),
            "source_support_intervals_mm": [
                list(value) for value in self.source_support_intervals_mm
            ],
            "owner_intervals_mm": [list(value) for value in self.owner_intervals_mm],
            "owner_boundaries_x": list(self.owner_boundaries_x),
            "source_center_xy": [list(value) for value in self.source_center_xy],
            "pair_overlap_pixels": list(self.pair_overlap_pixels),
            "canvas_megapixels": self.canvas_megapixels,
            "aggregate_megapixels": self.aggregate_megapixels,
        }


@dataclass(frozen=True)
class CentralStripDiagnosticResult:
    """Diagnostic renderer callback result consumed by sequence orchestration."""

    panorama: np.ndarray
    metadata: dict[str, object]


def validate_central_strip_config(
    config: Mapping[str, object] | None,
    *,
    require_enabled: bool = False,
) -> dict[str, object]:
    """Validate the deliberately small, non-user-tunable diagnostic config."""

    values = dict(_CENTRAL_STRIP_DEFAULTS)
    supplied = {} if config is None else dict(config)
    unknown = sorted(set(supplied) - set(_CENTRAL_STRIP_DEFAULTS))
    if unknown:
        raise ValueError(
            "Unknown stitch.central_strip_diagnostic configuration key(s): "
            + ", ".join(unknown)
        )
    values.update(supplied)
    if not isinstance(values["enabled"], bool):
        raise ValueError("central_strip_diagnostic.enabled must be a boolean")
    if require_enabled and not bool(values["enabled"]):
        raise ValueError("central_strip_diagnostic must be enabled by its diagnostic CLI")
    if values["reference_scale_mode"] != "robust_aligned_depth_plane":
        raise ValueError(
            "central_strip_diagnostic only supports robust_aligned_depth_plane"
        )
    if values["orientation_mode"] != "verified_camera_to_world":
        raise ValueError(
            "central_strip_diagnostic only supports verified_camera_to_world"
        )
    supplied_fraction = values["maximum_central_band_fraction"]
    if not isinstance(supplied_fraction, (int, float)) or isinstance(
        supplied_fraction, bool
    ):
        raise ValueError(
            "central_strip_diagnostic.maximum_central_band_fraction must be numeric"
        )
    fraction = float(supplied_fraction)
    if not np.isfinite(fraction) or not np.isclose(fraction, 0.20):
        raise ValueError(
            "central_strip_diagnostic.maximum_central_band_fraction is fixed at 0.20"
        )
    overlap = values["minimum_pair_overlap_pixels"]
    if type(overlap) is not int or overlap != 96:
        raise ValueError(
            "central_strip_diagnostic.minimum_pair_overlap_pixels is fixed at 96"
        )
    if values["exposure_mode"] != "global_gain":
        raise ValueError("central_strip_diagnostic only supports global_gain exposure")
    if type(values["multiband_levels"]) is not int or values["multiband_levels"] != 5:
        raise ValueError(
            "central_strip_diagnostic multiband_levels is fixed at the audited value 5"
        )
    values["maximum_central_band_fraction"] = fraction
    values["minimum_pair_overlap_pixels"] = overlap
    values["multiband_levels"] = 5
    return values


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"Cannot form a finite {label} direction")
    return value / norm


def _camera_points(
    columns: np.ndarray,
    rows: np.ndarray,
    depth_mm: np.ndarray,
    calibration: CameraIntrinsics,
) -> np.ndarray:
    z = np.asarray(depth_mm, dtype=np.float64)
    x = (np.asarray(columns, dtype=np.float64) - calibration.cx) * z / calibration.fx
    y = (np.asarray(rows, dtype=np.float64) - calibration.cy) * z / calibration.fy
    return np.stack((x, y, z), axis=-1)


def _to_world(camera_points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return camera_points @ pose[:3, :3].T + pose[:3, 3]


def _validate_time_order(
    poses: Sequence[np.ndarray], axes: ScanAxes
) -> dict[str, object]:
    """Audit trajectory geometry in capture order without sorting it."""

    checked = [validate_camera_to_world(pose) for pose in poses]
    if len(checked) < 2:
        raise ValueError("Central-strip diagnostics require at least two real poses")
    centers = np.stack([pose[:3, 3] for pose in checked], axis=0)
    rotations = np.stack([pose[:3, :3] for pose in checked], axis=0)
    scan = np.asarray(axes.scan_axis, dtype=np.float64)
    up = np.asarray(axes.up_axis, dtype=np.float64)
    normal = np.asarray(axes.normal_axis, dtype=np.float64)
    positions = centers @ scan
    increments = np.diff(positions)
    span = float(positions[-1] - positions[0])
    if not np.isfinite(span) or span <= 20.0:
        raise RuntimeError("Central-strip camera centres have no positive scan span")
    positive = increments[increments > 1e-6]
    typical_step = float(np.median(positive)) if positive.size else 0.0
    reverse_limit = max(5.0, typical_step * 0.20)
    reverse_count = int(np.count_nonzero(increments < -reverse_limit))
    if reverse_count:
        raise RuntimeError(
            "Central-strip trajectory reverses in capture order; poses must not be reordered"
        )
    negative_fraction = float(np.mean(increments < 0.0))
    if negative_fraction > 0.02:
        raise RuntimeError("Central-strip trajectory has excessive capture-order backtracking")
    origin = centers[0]
    line = origin + np.outer(positions - positions[0], scan)
    line_residual = np.linalg.norm(centers - line, axis=1)
    max_line_residual = float(np.max(line_residual))
    if max_line_residual > max(60.0, span * 0.10):
        raise RuntimeError("Central-strip camera centres are not close enough to one line")
    up_samples = -rotations[:, :, 1]
    forward_samples = rotations[:, :, 2]
    if np.min(up_samples @ up) < 0.75:
        raise RuntimeError("Central-strip camera up directions are inconsistent")
    if np.min(forward_samples @ normal) < 0.50:
        raise RuntimeError("Central-strip camera forward directions are inconsistent")
    return {
        "camera_scan_positions_mm": [float(value) for value in positions],
        "scan_span_mm": span,
        "maximum_line_residual_mm": max_line_residual,
        "maximum_reverse_step_mm": float(max(0.0, -np.min(increments))),
        "negative_step_fraction": negative_fraction,
        "minimum_up_alignment": float(np.min(up_samples @ up)),
        "minimum_forward_alignment": float(np.min(forward_samples @ normal)),
    }


def _collect_plane_points(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Sample only measured aligned-depth points for robust plane fitting."""

    maps = undistortion_maps(calibration)
    all_points: list[np.ndarray] = []
    candidate_counts: list[int] = []
    sampled_counts: list[int] = []
    for frame, pose in zip(frames, poses, strict=True):
        depth_raw = read_aligned_depth_mm(frame)
        depth, geometric = undistort_depth_with_validity(depth_raw, maps)
        valid = geometric & np.isfinite(depth) & (depth > 0.0) & (depth <= 10_000.0)
        candidate_count = int(np.count_nonzero(valid))
        candidate_counts.append(candidate_count)
        if candidate_count == 0:
            sampled_counts.append(0)
            continue
        stride = max(1, int(math.ceil(math.sqrt(candidate_count / _PLANE_SAMPLE_LIMIT_PER_FRAME))))
        sampled = valid[::stride, ::stride]
        local_rows, local_columns = np.nonzero(sampled)
        rows = local_rows.astype(np.int64) * stride
        columns = local_columns.astype(np.int64) * stride
        keep = (rows < calibration.height) & (columns < calibration.width)
        rows = rows[keep]
        columns = columns[keep]
        values = depth[rows, columns]
        points = _camera_points(columns, rows, values, calibration)
        all_points.append(_to_world(points, pose))
        sampled_counts.append(int(values.size))
    if not all_points:
        raise RuntimeError("Measured aligned depth contains no points for a reference plane")
    return np.concatenate(all_points, axis=0), candidate_counts, sampled_counts


def _fit_plane_candidate(
    points: np.ndarray,
    normal_hint: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """TLS fit followed by bounded robust inlier refinement."""

    if points.shape[0] < 64:
        raise ValueError("A plane candidate needs at least 64 measured points")
    working = points
    normal = normal_hint
    offset = float(np.median(points @ normal_hint))
    inliers = np.ones(points.shape[0], dtype=bool)
    for _ in range(4):
        centroid = np.mean(working, axis=0)
        _, _, right = np.linalg.svd(working - centroid, full_matrices=False)
        normal = _unit(right[-1], "reference-plane normal")
        if float(np.dot(normal, normal_hint)) < 0.0:
            normal = -normal
        alignment = float(np.dot(normal, normal_hint))
        if alignment < 0.65:
            raise ValueError("Measured plane is not aligned with the observed background")
        offset = float(np.dot(normal, centroid))
        residual = np.abs(points @ normal - offset)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        threshold = min(90.0, max(20.0, median + 3.5 * max(1.4826 * mad, 3.0)))
        inliers = residual <= threshold
        if int(np.count_nonzero(inliers)) < 64:
            raise ValueError("Reference-plane robust fit lost all measured support")
        working = points[inliers]
    residual = np.abs(points @ normal - offset)
    return normal, offset, inliers, residual


@dataclass(frozen=True)
class _PlaneCandidate:
    """One physically distinct measured-plane hypothesis before admission."""

    normal: np.ndarray
    offset_mm: float
    all_inliers: np.ndarray
    all_residual_mm: np.ndarray
    inlier_count: int
    residual_p50_mm: float
    residual_p95_mm: float
    normal_view_angle_deg: float
    camera_distances_mm: np.ndarray
    per_frame_depth_coverage: tuple[float, ...]
    per_frame_calibrated_area_fraction: tuple[float, ...]
    supporting_frame_count: int
    supporting_frame_fraction: float
    maximum_unsupported_frame_run: int


def _per_frame_fraction(
    values: np.ndarray,
    sampled_counts: Sequence[int],
) -> tuple[float, ...]:
    """Return per-frame means for a frame-concatenated sampled-point vector."""

    coverage: list[float] = []
    cursor = 0
    for count in sampled_counts:
        segment = values[cursor : cursor + count]
        cursor += count
        coverage.append(float(np.mean(segment)) if count else 0.0)
    if cursor != values.shape[0]:
        raise RuntimeError("Reference-plane sampled-point accounting is inconsistent")
    return tuple(coverage)


def _maximum_false_run(values: Sequence[bool]) -> int:
    maximum = 0
    current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def _same_physical_plane(
    first: _PlaneCandidate,
    second: _PlaneCandidate,
) -> bool:
    """Deduplicate adjacent histogram peaks that refit the same physical plane."""

    alignment = float(np.clip(np.dot(first.normal, second.normal), -1.0, 1.0))
    normal_angle = float(np.degrees(np.arccos(alignment)))
    offset_tolerance = max(
        _PLANE_CANDIDATE_OFFSET_DEDUP_FLOOR_MM,
        2.0 * max(first.residual_p95_mm, second.residual_p95_mm),
    )
    return normal_angle <= _PLANE_CANDIDATE_NORMAL_DEDUP_DEG and abs(
        first.offset_mm - second.offset_mm
    ) <= offset_tolerance


def fit_reference_plane(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    axes: ScanAxes,
) -> ReferencePlane:
    """Fit a real background plane from aligned-depth samples only."""

    points, candidate_counts, sampled_counts = _collect_plane_points(
        frames, poses, calibration
    )
    normal_hint = np.asarray(axes.normal_axis, dtype=np.float64)
    normal_coordinates = points @ normal_hint
    bins = np.floor(normal_coordinates / _PLANE_HISTOGRAM_BIN_MM).astype(np.int64)
    values, counts = np.unique(bins, return_counts=True)
    checked = [validate_camera_to_world(pose) for pose in poses]
    centers = np.stack([pose[:3, 3] for pose in checked], axis=0)
    mean_forward = _unit(
        np.mean([pose[:3, 2] for pose in checked], axis=0), "mean viewing"
    )
    maximum_unsupported_run = max(2, int(math.ceil(len(sampled_counts) * 0.10)))
    image_area = calibration.width * calibration.height
    # Evaluate several physical depth peaks.  Histogram neighbours commonly
    # describe the same noisy plane, so they are deduplicated below before any
    # ambiguity decision.  We deliberately do not guess that nearest or
    # farthest depth is the background: a viable reference must be a single
    # dominant, scan-wide measured plane.
    peak_order = np.argsort(counts)[::-1][: min(12, len(counts))]
    candidates: list[_PlaneCandidate] = []
    for index in peak_order:
        centre = (float(values[index]) + 0.5) * _PLANE_HISTOGRAM_BIN_MM
        candidate = points[np.abs(normal_coordinates - centre) <= 90.0]
        if candidate.shape[0] < 64:
            continue
        try:
            normal, offset, inliers, residual = _fit_plane_candidate(
                candidate, normal_hint
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        all_residual = np.abs(points @ normal - offset)
        fitted_residual = residual[inliers]
        residual_p50 = float(np.percentile(fitted_residual, 50))
        residual_p95 = float(np.percentile(fitted_residual, 95))
        threshold = max(30.0, residual_p95)
        all_inliers = all_residual <= threshold
        score = int(np.count_nonzero(all_inliers))
        camera_distances = offset - centers @ normal
        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(float(np.dot(normal, mean_forward)), -1.0, 1.0)
                )
            )
        )
        per_frame = _per_frame_fraction(all_inliers, sampled_counts)
        area_fraction = tuple(
            float(candidate_count / image_area) * coverage
            for candidate_count, coverage in zip(
                candidate_counts, per_frame, strict=True
            )
        )
        supported = (
            np.asarray(area_fraction, dtype=np.float64)
            >= _PLANE_MIN_CALIBRATED_AREA_FRACTION
        ) & (
            np.asarray(per_frame, dtype=np.float64) >= _PLANE_MIN_FRAME_COVERAGE
        )
        candidates.append(
            _PlaneCandidate(
                normal=normal,
                offset_mm=float(offset),
                all_inliers=all_inliers,
                all_residual_mm=all_residual,
                inlier_count=score,
                residual_p50_mm=residual_p50,
                residual_p95_mm=residual_p95,
                normal_view_angle_deg=angle,
                camera_distances_mm=camera_distances,
                per_frame_depth_coverage=per_frame,
                per_frame_calibrated_area_fraction=area_fraction,
                supporting_frame_count=int(np.count_nonzero(supported)),
                supporting_frame_fraction=float(
                    np.mean(supported) if supported.size else 0.0
                ),
                maximum_unsupported_frame_run=_maximum_false_run(supported),
            )
        )
    if not candidates:
        raise RuntimeError("Measured aligned depth cannot support a reference plane")

    ranked = sorted(candidates, key=lambda item: item.inlier_count, reverse=True)
    distinct: list[_PlaneCandidate] = []
    for candidate in ranked:
        if not any(_same_physical_plane(candidate, prior) for prior in distinct):
            distinct.append(candidate)
    winner = distinct[0]
    runner_score_ratio = max(
        (
            candidate.inlier_count / winner.inlier_count
            for candidate in distinct[1:]
        ),
        default=0.0,
    )
    if any(
        candidate.inlier_count >= _PLANE_COMPETING_SCORE_RATIO * winner.inlier_count
        and candidate.inlier_count
        >= _PLANE_MIN_COMPETING_GLOBAL_FRACTION * points.shape[0]
        for candidate in distinct[1:]
    ):
        raise RuntimeError(
            "Measured aligned depth contains ambiguous competing reference planes"
        )

    global_inlier_fraction = winner.inlier_count / points.shape[0]
    if global_inlier_fraction < _PLANE_MIN_GLOBAL_INLIER_FRACTION:
        raise RuntimeError(
            "Measured reference plane has insufficient global inlier support"
        )
    if (
        winner.residual_p50_mm > _PLANE_STRUCTURAL_MAX_P50_MM
        or winner.residual_p95_mm > _PLANE_STRUCTURAL_MAX_P95_MM
    ):
        raise RuntimeError("Measured reference plane exceeds structural residual limits")
    if (
        winner.supporting_frame_fraction < _PLANE_MIN_SUPPORTING_FRAME_FRACTION
        or winner.maximum_unsupported_frame_run > maximum_unsupported_run
    ):
        raise RuntimeError(
            "Measured reference plane lacks calibrated image-area support across the scan"
        )
    if not np.isfinite(winner.camera_distances_mm).all() or np.any(
        winner.camera_distances_mm <= 100.0
    ):
        raise RuntimeError("Reference plane is not in front of every real camera pose")
    if winner.normal_view_angle_deg > 45.0:
        raise RuntimeError("Reference-plane normal is inconsistent with mean viewing direction")

    origin = winner.normal * winner.offset_mm
    total_candidates = sum(candidate_counts)
    total_sampled = sum(sampled_counts)
    fit_quality_pass = (
        winner.residual_p50_mm <= _PLANE_QUALITY_MAX_P50_MM
        and winner.residual_p95_mm <= _PLANE_QUALITY_MAX_P95_MM
    )
    return ReferencePlane(
        normal_world=tuple(float(value) for value in winner.normal),
        offset_mm=winner.offset_mm,
        origin_world_mm=tuple(float(value) for value in origin),
        inlier_count=winner.inlier_count,
        sampled_point_count=int(points.shape[0]),
        inlier_fraction=float(global_inlier_fraction),
        residual_p50_mm=winner.residual_p50_mm,
        residual_p95_mm=winner.residual_p95_mm,
        normal_view_angle_deg=winner.normal_view_angle_deg,
        camera_distances_mm=tuple(
            float(value) for value in winner.camera_distances_mm
        ),
        measured_depth_validation_coverage=float(
            winner.inlier_count / max(1, total_sampled)
        ),
        measured_depth_sampling_fraction=float(total_sampled / max(1, total_candidates)),
        per_frame_depth_coverage=winner.per_frame_depth_coverage,
        per_frame_calibrated_area_fraction=winner.per_frame_calibrated_area_fraction,
        supporting_frame_count=winner.supporting_frame_count,
        supporting_frame_fraction=winner.supporting_frame_fraction,
        maximum_unsupported_frame_run=winner.maximum_unsupported_frame_run,
        fit_quality_pass=fit_quality_pass,
        candidate_count=len(candidates),
        distinct_candidate_count=len(distinct),
        runner_score_ratio=float(runner_score_ratio),
    )


def _ray_plane_intersection(
    pose: np.ndarray,
    direction_camera: np.ndarray,
    plane: ReferencePlane,
) -> np.ndarray:
    rotation = pose[:3, :3]
    centre = pose[:3, 3]
    normal = np.asarray(plane.normal_world, dtype=np.float64)
    direction_world = rotation @ np.asarray(direction_camera, dtype=np.float64)
    denominator = float(np.dot(normal, direction_world))
    numerator = float(plane.offset_mm - np.dot(normal, centre))
    if not np.isfinite(denominator) or abs(denominator) < 1e-7:
        raise RuntimeError("A central-strip camera ray is parallel to the reference plane")
    distance = numerator / denominator
    if not np.isfinite(distance) or distance <= 0.0:
        raise RuntimeError("The reference plane lies behind a central-strip camera")
    return centre + distance * direction_world


def _central_band_bounds(
    pose: np.ndarray,
    plane: ReferencePlane,
    axes: ScanAxes,
    calibration: CameraIntrinsics,
    fraction: float,
) -> tuple[float, float, float, float]:
    half = 0.5 * calibration.width * fraction
    left = calibration.cx - half
    right = calibration.cx + half
    if left < 0.0 or right > calibration.width - 1.0:
        raise ValueError("Central-band fraction exceeds calibrated colour field")
    scan = np.asarray(axes.scan_axis, dtype=np.float64)
    up = np.asarray(axes.up_axis, dtype=np.float64)
    origin = np.asarray(plane.origin_world_mm, dtype=np.float64)
    points: list[np.ndarray] = []
    for column in (left, right):
        for row in (0.0, float(calibration.height - 1)):
            direction = np.array(
                [
                    (column - calibration.cx) / calibration.fx,
                    (row - calibration.cy) / calibration.fy,
                    1.0,
                ],
                dtype=np.float64,
            )
            points.append(_ray_plane_intersection(pose, direction, plane))
    values = np.stack(points, axis=0) - origin
    scan_values = values @ scan
    up_values = values @ up
    return (
        float(np.min(scan_values)),
        float(np.max(scan_values)),
        float(np.min(up_values)),
        float(np.max(up_values)),
    )


def build_central_strip_layout(
    poses: Sequence[np.ndarray],
    plane: ReferencePlane,
    axes: ScanAxes,
    calibration: CameraIntrinsics,
    *,
    maximum_central_band_fraction: float,
    minimum_pair_overlap_pixels: int,
) -> CentralStripLayout:
    """Allocate a bounded plane canvas and time-ordered midpoint ownership."""

    checked = [validate_camera_to_world(pose) for pose in poses]
    if not 2 <= len(checked) <= _HARD_MAX_RENDER_SOURCES:
        raise ValueError("Central-strip rendering requires between two and 32 sources")
    bounds = [
        _central_band_bounds(
            pose,
            plane,
            axes,
            calibration,
            maximum_central_band_fraction,
        )
        for pose in checked
    ]
    support = [(item[0], item[1]) for item in bounds]
    # Owner intervals are defined by actual camera centres, not by where an
    # optical-axis ray happens to hit the plane.  The two diverge under yaw;
    # using the latter would conceal real trajectory drift and violate the
    # metric definition x_i = scan_axis dot (C_i - plane_origin).
    origin = np.asarray(plane.origin_world_mm, dtype=np.float64)
    scan = np.asarray(axes.scan_axis, dtype=np.float64)
    up = np.asarray(axes.up_axis, dtype=np.float64)
    camera_centres = np.stack([pose[:3, 3] for pose in checked], axis=0) - origin
    positions = camera_centres @ scan
    centre_up_positions = camera_centres @ up
    steps = np.diff(positions)
    if np.any(steps <= 1e-4):
        raise RuntimeError(
            "Selected central-strip source positions are not strictly monotonic in capture order"
        )
    owner_boundaries_mm = 0.5 * (positions[:-1] + positions[1:])
    owner_edges = np.r_[support[0][0], owner_boundaries_mm, support[-1][1]]
    if np.any(np.diff(owner_edges) <= 1e-4):
        raise RuntimeError("Central-strip trajectory midpoint owner intervals collapse")
    tolerance_mm = 1.5
    for index, ((left, right), (support_left, support_right)) in enumerate(
        zip(zip(owner_edges[:-1], owner_edges[1:], strict=True), support, strict=True)
    ):
        if left < support_left - tolerance_mm or right > support_right + tolerance_mm:
            raise RuntimeError(
                "Central-strip frame spacing exceeds its calibrated central field "
                f"at render source {index}"
            )
    distances = np.asarray(plane.camera_distances_mm, dtype=np.float64)
    density = min(calibration.fx, calibration.fy) / float(np.median(distances))
    if not np.isfinite(density) or density <= 0.0:
        raise RuntimeError("Cannot derive a finite central-strip pixel density")
    scan_min = float(owner_edges[0])
    scan_max = float(owner_edges[-1])
    up_min = float(min(item[2] for item in bounds))
    up_max = float(max(item[3] for item in bounds))
    width = int(math.ceil((scan_max - scan_min) * density)) + 1
    height = int(math.ceil((up_max - up_min) * density)) + 1
    if width < 2 or height < 2:
        raise RuntimeError("Central-strip reference-plane canvas is degenerate")
    canvas_megapixels = width * height / 1_000_000.0
    aggregate_megapixels = canvas_megapixels * len(checked)
    if canvas_megapixels > _HARD_MAX_CANVAS_MEGAPIXELS:
        raise MemoryError("Central-strip canvas exceeds the 200 MP hard limit")
    if aggregate_megapixels > _DEFAULT_MAX_AGGREGATE_MEGAPIXELS:
        raise MemoryError(
            "Central-strip aggregate working set exceeds the 80 MP diagnostic budget"
        )
    overlaps = [
        (min(first[1], second[1]) - max(first[0], second[0])) * density
        for first, second in zip(support[:-1], support[1:], strict=True)
    ]
    if any(value < float(minimum_pair_overlap_pixels) for value in overlaps):
        raise RuntimeError(
            "Adjacent central strips lack the required calibrated overlap"
        )
    centres = tuple(
        (
            float((position - scan_min) * density),
            float((up_max - up_position) * density),
        )
        for position, up_position in zip(positions, centre_up_positions, strict=True)
    )
    return CentralStripLayout(
        width=width,
        height=height,
        pixels_per_mm=float(density),
        scan_min_mm=scan_min,
        scan_max_mm=scan_max,
        up_min_mm=up_min,
        up_max_mm=up_max,
        source_scan_positions_mm=tuple(float(value) for value in positions),
        source_support_intervals_mm=tuple(
            (float(left), float(right)) for left, right in support
        ),
        owner_intervals_mm=tuple(
            (float(left), float(right))
            for left, right in zip(owner_edges[:-1], owner_edges[1:], strict=True)
        ),
        owner_boundaries_x=tuple(
            float((value - scan_min) * density) for value in owner_boundaries_mm
        ),
        source_center_xy=centres,
        pair_overlap_pixels=tuple(float(value) for value in overlaps),
        canvas_megapixels=float(canvas_megapixels),
        aggregate_megapixels=float(aggregate_megapixels),
    )


def _read_bgr(frame: RGBDFrame) -> np.ndarray:
    image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Could not decode calibrated colour frame: {frame.color_path}")
    return image


def prewarp_central_strip_source(
    frame: RGBDFrame,
    pose: np.ndarray,
    calibration: CameraIntrinsics,
    plane: ReferencePlane,
    axes: ScanAxes,
    layout: CentralStripLayout,
    *,
    maximum_central_band_fraction: float,
    projected_center_xy: tuple[float, float] | None = None,
) -> tuple[PrewarpedScanSource, dict[str, object]]:
    """Sample one real source exactly once from the plane output to raw RGB-D."""

    checked_pose = validate_camera_to_world(pose)
    image = _read_bgr(frame)
    depth = read_aligned_depth_mm(frame)
    expected_shape = (calibration.height, calibration.width)
    if image.shape[:2] != expected_shape or depth.shape != expected_shape:
        raise ValueError("Central-strip source dimensions must match calibrated colour")
    height, width = layout.height, layout.width
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    color_valid = np.zeros((height, width), dtype=np.uint8)
    surface_depth = np.zeros((height, width), dtype=np.float32)
    surface_valid = np.zeros((height, width), dtype=np.uint8)
    camera_depth = np.zeros((height, width), dtype=np.float32)
    camera_valid = np.zeros((height, width), dtype=np.uint8)
    normal = np.asarray(plane.normal_world, dtype=np.float64)
    scan = np.asarray(axes.scan_axis, dtype=np.float64)
    up = np.asarray(axes.up_axis, dtype=np.float64)
    origin = np.asarray(plane.origin_world_mm, dtype=np.float64)
    centre = checked_pose[:3, 3]
    rotation = checked_pose[:3, :3]
    # The central FOV is defined in calibrated undistorted coordinates.  The
    # inverse map below then applies the real distortion directly into the raw
    # aligned RGB-D images; no pre-undistort / rotate / rescale chain exists.
    half = 0.5 * calibration.width * maximum_central_band_fraction
    central_left = calibration.cx - half
    central_right = calibration.cx + half
    source_x = layout.scan_min_mm + np.arange(width, dtype=np.float64) / layout.pixels_per_mm
    source_up = layout.up_max_mm - np.arange(height, dtype=np.float64) / layout.pixels_per_mm
    for row_start in range(0, height, 64):
        row_stop = min(height, row_start + 64)
        local_up = source_up[row_start:row_stop]
        scan_grid = np.broadcast_to(source_x[None, :], (row_stop - row_start, width))
        up_grid = np.broadcast_to(local_up[:, None], (row_stop - row_start, width))
        world = (
            origin[None, None, :]
            + scan_grid[..., None] * scan[None, None, :]
            + up_grid[..., None] * up[None, None, :]
        )
        camera_points = (world.reshape(-1, 3) - centre) @ rotation
        map_x, map_y, positive_z = camera_points_to_source_pixels(
            camera_points.reshape(row_stop - row_start, width, 3), calibration
        )
        undistorted_u = (
            calibration.fx * camera_points[:, 0] / camera_points[:, 2] + calibration.cx
        ).reshape(row_stop - row_start, width)
        in_source = (
            positive_z
            & np.isfinite(undistorted_u)
            & (undistorted_u >= central_left)
            & (undistorted_u <= central_right)
            & (map_x >= 0.0)
            & (map_x <= calibration.width - 1.0)
            & (map_y >= 0.0)
            & (map_y <= calibration.height - 1.0)
        )
        sampled_rgb = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sampled_depth = cv2.remap(
            np.asarray(depth, dtype=np.float32),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        measured = (
            in_source
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
        )
        rgb[row_start:row_stop] = sampled_rgb
        color_valid[row_start:row_stop] = np.where(in_source, 255, 0).astype(np.uint8)
        camera_depth[row_start:row_stop][measured] = sampled_depth[measured]
        camera_valid[row_start:row_stop] = np.where(measured, 255, 0).astype(np.uint8)
        if np.any(measured):
            q = camera_points.reshape(row_stop - row_start, width, 3)
            scale = np.zeros((row_stop - row_start, width), dtype=np.float64)
            scale[measured] = sampled_depth[measured] / q[..., 2][measured]
            measured_world = q * scale[..., None]
            measured_world = measured_world @ rotation.T + centre
            values = np.sum(measured_world * normal[None, None, :], axis=2)
            surface_depth[row_start:row_stop][measured] = values[measured].astype(np.float32)
            surface_valid[row_start:row_stop] = np.where(measured, 255, 0).astype(np.uint8)
    if not np.any(color_valid):
        raise RuntimeError(f"Central strip source {frame.frame_id} has no calibrated RGB coverage")
    if np.any((surface_valid > 0) & (color_valid == 0)) or np.any(
        (camera_valid > 0) & (color_valid == 0)
    ):
        raise RuntimeError("Central-strip depth validity escaped calibrated colour validity")
    projected_center = (
        tuple(float(value) for value in projected_center_xy)
        if projected_center_xy is not None
        else (float(width - 1) * 0.5, float(height - 1) * 0.5)
    )
    colour_count = int(np.count_nonzero(color_valid))
    depth_count = int(np.count_nonzero(camera_valid))
    source = PrewarpedScanSource(
        rgb=rgb,
        color_valid_mask=color_valid,
        surface_depth_mm=surface_depth,
        surface_depth_valid_mask=surface_valid,
        camera_depth_mm=camera_depth,
        camera_depth_valid_mask=camera_valid,
        projected_center_xy=projected_center,
        frame_id=int(frame.frame_id),
        projected_height_px=height,
        sampling_stats={
            "color_valid_fraction": float(colour_count / (height * width)),
            "measured_depth_valid_fraction": float(depth_count / (height * width)),
            "single_inverse_remap": True,
        },
    )
    metadata: dict[str, object] = {
        "frame_id": int(frame.frame_id),
        "color_valid_pixel_count": colour_count,
        "color_valid_fraction": float(colour_count / (height * width)),
        "measured_depth_valid_pixel_count": depth_count,
        "measured_depth_valid_fraction": float(depth_count / (height * width)),
        "rgb_without_measured_depth_pixel_count": int(colour_count - depth_count),
        "single_inverse_remap": True,
        "colour_sampling": "bilinear",
        "depth_sampling": "nearest",
    }
    return source, metadata


def _prepare_central_strip_render_domains(
    sources: Sequence[PrewarpedScanSource],
    layout: CentralStripLayout,
    *,
    minimum_pair_overlap_pixels: int,
) -> tuple[list[PrewarpedScanSource], tuple[np.ndarray, ...], list[dict[str, object]]]:
    """Make pair corridors mutually exclusive without inventing any support.

    Consecutive narrow fields can have a real three-way colour overlap.  The
    renderer intentionally refuses overlapping pair corridors, so this method
    partitions that *already measured* overlap into one narrow ribbon around
    each real midpoint.  Source colour outside its owner core or an adjacent
    ribbon is excluded from this diagnostic canvas rather than becoming an
    implicit non-adjacent blend path.
    """

    if len(sources) != len(layout.source_center_xy):
        raise ValueError("Central-strip sources and layout centres must align")
    shape = sources[0].color_valid_mask.shape
    height, width = shape
    if any(source.color_valid_mask.shape != shape for source in sources):
        raise ValueError("Central-strip sources must share one prewarped canvas")
    boundaries = np.asarray(layout.owner_boundaries_x, dtype=np.float64)
    x_grid = np.arange(width, dtype=np.float64)[None, :]
    pair_masks: list[np.ndarray] = []
    pair_audits: list[dict[str, object]] = []
    for index, boundary in enumerate(boundaries):
        lower = 0.0 if index == 0 else float(boundaries[index - 1])
        upper = float(width - 1) if index + 1 == len(boundaries) else float(boundaries[index + 1])
        available_half = min(boundary - lower, upper - boundary) * 0.45
        # The minimum pair overlap is a physical central-FOV admission test;
        # it need not allocate all 96 px to a GraphCut ribbon when capture
        # spacing is much denser.  Keep enough room for five-band blending and
        # never let neighbouring corridors touch.
        desired_half = max(8.0, float(minimum_pair_overlap_pixels) * 0.5)
        half_width = min(desired_half, available_half)
        if half_width < 4.0:
            raise RuntimeError("Central-strip midpoint corridors are too close to separate")
        ribbon = (x_grid >= boundary - half_width) & (x_grid < boundary + half_width)
        common = (
            (sources[index].color_valid_mask > 0)
            & (sources[index + 1].color_valid_mask > 0)
        )
        pair_union = (
            (sources[index].color_valid_mask > 0)
            | (sources[index + 1].color_valid_mask > 0)
        )
        active_rows = np.any(pair_union, axis=1)
        active_cross_coverage = float(np.count_nonzero(active_rows) / max(1, height))
        if active_cross_coverage < 0.80:
            raise RuntimeError(
                "Adjacent central strips do not cover 80% of the required "
                "cross-scan canvas extent"
            )
        row_widths: list[int] = []
        for row in common:
            padded = np.r_[row, False]
            starts = np.flatnonzero(padded & ~np.r_[False, padded[:-1]])
            stops = np.flatnonzero(~padded & np.r_[False, padded[:-1]])
            row_widths.append(
                int(np.max(stops - starts)) if starts.size else 0
            )
        width_array = np.asarray(row_widths, dtype=np.int32)
        supported_rows = active_rows & (width_array >= minimum_pair_overlap_pixels)
        cross_coverage = float(
            np.count_nonzero(supported_rows) / max(1, np.count_nonzero(active_rows))
        )
        if cross_coverage < 0.80:
            raise RuntimeError(
                "Adjacent central strips lack 96 pixels of actual common RGB "
                "coverage across the required cross-scan extent"
            )
        pair_audits.append(
            {
                "pair_indices": [index, index + 1],
                "minimum_required_overlap_pixels": minimum_pair_overlap_pixels,
                "expected_cross_scan_rows": height,
                "active_cross_scan_rows": int(np.count_nonzero(active_rows)),
                "active_cross_coverage_ratio": active_cross_coverage,
                "minimum_common_overlap_pixels": int(
                    np.min(width_array[active_rows]) if np.any(active_rows) else 0
                ),
                "p50_common_overlap_pixels": float(
                    np.percentile(width_array[active_rows], 50)
                    if np.any(active_rows)
                    else 0.0
                ),
                "cross_coverage_ratio": cross_coverage,
            }
        )
        pair = common & ribbon
        if not np.any(pair):
            raise RuntimeError("Adjacent central strips have no true RGB overlap ribbon")
        pair_masks.append(pair.astype(np.uint8) * 255)
    if pair_masks:
        pair_stack = np.stack([mask > 0 for mask in pair_masks], axis=0)
        if np.any(np.sum(pair_stack, axis=0) > 1):
            raise RuntimeError("Central-strip adjacent-pair ribbons are not mutually exclusive")

    restricted: list[PrewarpedScanSource] = []
    for index, source in enumerate(sources):
        lower = -np.inf if index == 0 else boundaries[index - 1]
        upper = np.inf if index + 1 == len(sources) else boundaries[index]
        core = x_grid >= lower
        if np.isfinite(upper):
            core &= x_grid < upper
        allowed = np.broadcast_to(core, shape).copy()
        if index:
            allowed |= pair_masks[index - 1] > 0
        if index + 1 < len(sources):
            allowed |= pair_masks[index] > 0
        color_valid = np.where(
            allowed & (source.color_valid_mask > 0), 255, 0
        ).astype(np.uint8)
        surface_valid = np.where(
            (color_valid > 0) & (source.surface_depth_valid_mask > 0), 255, 0
        ).astype(np.uint8)
        camera_valid = np.where(
            (color_valid > 0) & (source.camera_depth_valid_mask > 0), 255, 0
        ).astype(np.uint8)
        if not np.any(color_valid):
            raise RuntimeError("Central-strip owner domain removed an entire real source")
        restricted.append(
            PrewarpedScanSource(
                rgb=source.rgb,
                color_valid_mask=color_valid,
                surface_depth_mm=source.surface_depth_mm,
                surface_depth_valid_mask=surface_valid,
                camera_depth_mm=source.camera_depth_mm,
                camera_depth_valid_mask=camera_valid,
                projected_center_xy=source.projected_center_xy,
                frame_id=source.frame_id,
                projected_height_px=source.projected_height_px,
                sampling_stats=source.sampling_stats,
                sharpness_score=source.sharpness_score,
            )
        )
    return restricted, tuple(pair_masks), pair_audits


def _adjacent_motion_audit(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    axes: ScanAxes,
    layout: CentralStripLayout,
) -> list[dict[str, object]]:
    """Record real capture-to-capture motion and exposure blur estimates."""

    if len(frames) != len(poses):
        raise ValueError("Central-strip motion frames and poses must align")
    scan = np.asarray(axes.scan_axis, dtype=np.float64)
    up = np.asarray(axes.up_axis, dtype=np.float64)
    normal = np.asarray(axes.normal_axis, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for first_frame, second_frame, first_pose, second_pose in zip(
        frames[:-1], frames[1:], poses[:-1], poses[1:], strict=True
    ):
        if (
            first_frame.timestamp_us is None
            or second_frame.timestamp_us is None
            or first_frame.timestamp_us < 0
            or second_frame.timestamp_us <= first_frame.timestamp_us
        ):
            raise RuntimeError(
                "Central-strip adjacent motion audit requires increasing colour timestamps"
            )
        if (
            first_frame.color_exposure_raw is None
            or second_frame.color_exposure_raw is None
            or first_frame.color_exposure_raw <= 0
            or second_frame.color_exposure_raw <= 0
        ):
            raise RuntimeError(
                "Central-strip adjacent motion audit requires positive colour exposure metadata"
            )
        first = validate_camera_to_world(first_pose)
        second = validate_camera_to_world(second_pose)
        translation = second[:3, 3] - first[:3, 3]
        elapsed_seconds = (
            float(second_frame.timestamp_us - first_frame.timestamp_us) / 1_000_000.0
        )
        distance_mm = float(np.linalg.norm(translation))
        scan_mm = float(np.dot(translation, scan))
        up_mm = float(np.dot(translation, up))
        forward_mm = float(np.dot(translation, normal))
        speed_mm_per_second = distance_mm / elapsed_seconds
        scan_speed_mm_per_second = abs(scan_mm) / elapsed_seconds
        exposure_us = [
            float(first_frame.color_exposure_raw) * 100.0,
            float(second_frame.color_exposure_raw) * 100.0,
        ]
        blur_mm = [
            scan_speed_mm_per_second * value / 1_000_000.0 for value in exposure_us
        ]
        rows.append(
            {
                "from_frame_id": int(first_frame.frame_id),
                "to_frame_id": int(second_frame.frame_id),
                "timestamp_delta_seconds": elapsed_seconds,
                "translation_mm": distance_mm,
                "scan_displacement_mm": scan_mm,
                "vertical_displacement_mm": up_mm,
                "forward_displacement_mm": forward_mm,
                "speed_mm_per_second": speed_mm_per_second,
                "scan_speed_mm_per_second": scan_speed_mm_per_second,
                "source_exposure_us": exposure_us,
                "expected_scan_motion_blur_mm": blur_mm,
                "expected_scan_motion_blur_pixels": [
                    value * layout.pixels_per_mm for value in blur_mm
                ],
            }
        )
    return rows


def render_central_strip_diagnostic(
    *,
    plane_frames: Sequence[RGBDFrame],
    plane_poses: Sequence[np.ndarray],
    render_frames: Sequence[RGBDFrame],
    render_poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    config: Mapping[str, object] | None,
    sharpness_scores: Sequence[float] | None = None,
) -> CentralStripDiagnosticResult:
    """Render the independent, diagnostic-only reference-plane strip route."""

    settings = validate_central_strip_config(config, require_enabled=True)
    if len(plane_frames) != len(plane_poses):
        raise ValueError("Central-strip plane frames and poses must align")
    if len(render_frames) != len(render_poses):
        raise ValueError("Central-strip render frames and poses must align")
    if not 2 <= len(plane_frames) <= _HARD_MAX_POSE_NODES:
        raise ValueError("Central-strip plane fitting requires between two and 160 poses")
    if not 2 <= len(render_frames) <= _HARD_MAX_RENDER_SOURCES:
        raise ValueError("Central-strip rendering requires between two and 32 sources")
    checked_plane_poses = [validate_camera_to_world(pose) for pose in plane_poses]
    checked_render_poses = [validate_camera_to_world(pose) for pose in render_poses]
    scan, up, normal = estimate_world_axes(checked_plane_poses)
    axes = ScanAxes(
        scan_axis=tuple(float(value) for value in scan),
        up_axis=tuple(float(value) for value in up),
        normal_axis=tuple(float(value) for value in normal),
    )
    trajectory_audit = _validate_time_order(checked_plane_poses, axes)
    plane = fit_reference_plane(plane_frames, checked_plane_poses, calibration, axes)
    layout = build_central_strip_layout(
        checked_render_poses,
        plane,
        axes,
        calibration,
        maximum_central_band_fraction=float(settings["maximum_central_band_fraction"]),
        minimum_pair_overlap_pixels=int(settings["minimum_pair_overlap_pixels"]),
    )
    sources: list[PrewarpedScanSource] = []
    source_metadata: list[dict[str, object]] = []
    for index, (frame, pose) in enumerate(zip(render_frames, checked_render_poses, strict=True)):
        source, metadata = prewarp_central_strip_source(
            frame,
            pose,
            calibration,
            plane,
            axes,
            layout,
            maximum_central_band_fraction=float(settings["maximum_central_band_fraction"]),
            projected_center_xy=layout.source_center_xy[index],
        )
        sources.append(source)
        metadata["projected_center_xy"] = list(layout.source_center_xy[index])
        metadata["camera_scan_position_mm"] = layout.source_scan_positions_mm[index]
        metadata["owner_interval_mm"] = list(layout.owner_intervals_mm[index])
        metadata["support_interval_mm"] = list(layout.source_support_intervals_mm[index])
        source_metadata.append(metadata)
    sharpness = (
        [float(value) for value in sharpness_scores]
        if sharpness_scores is not None
        else [0.0] * len(sources)
    )
    if len(sharpness) != len(sources) or not np.isfinite(sharpness).all():
        raise ValueError("Central-strip sharpness scores must be finite and match sources")
    sources, pair_overlap_masks, pair_overlap_audits = _prepare_central_strip_render_domains(
        sources,
        layout,
        minimum_pair_overlap_pixels=int(settings["minimum_pair_overlap_pixels"]),
    )
    adjacent_motion = _adjacent_motion_audit(
        render_frames, checked_render_poses, axes, layout
    )
    panorama, render_info = render_prewarped_scan_panorama(
        sources,
        source_order=tuple(range(len(sources))),
        owner_boundaries_x=layout.owner_boundaries_x,
        max_megapixels=_DEFAULT_MAX_AGGREGATE_MEGAPIXELS,
        multiband_levels=int(settings["multiband_levels"]),
        exposure_mode=str(settings["exposure_mode"]),
        quality_gate=False,
        sharpness_scores=sharpness,
        pair_overlap_masks=pair_overlap_masks,
    )
    render_metadata = render_info.as_dict()
    quality = render_metadata.get("quality_metrics", {})
    if not isinstance(quality, dict):
        raise RuntimeError("Central-strip renderer omitted quality metrics")
    renderer_quality_pass = bool(quality.get("quality_pass", False))
    strip_quality_reasons: list[str] = []
    if not plane.fit_quality_pass:
        strip_quality_reasons.append("reference_plane_fit")
    if not renderer_quality_pass:
        strip_quality_reasons.append("render_quality")
    strip_quality_pass = plane.fit_quality_pass and renderer_quality_pass
    metadata: dict[str, object] = {
        "schema": "gemini305-central-strip-diagnostic/v1",
        "diagnostic_only": True,
        "deliverable_published": False,
        "geometry_claim": "reference_plane_only",
        "interpolated_pose_count": 0,
        "scan_axes": axes.as_dict(),
        "trajectory": trajectory_audit,
        "reference_plane": plane.as_dict(),
        "layout": layout.as_dict(),
        "actual_pair_overlap": pair_overlap_audits,
        "adjacent_motion": adjacent_motion,
        "sources": source_metadata,
        "render": render_metadata,
        "strip_quality_pass": strip_quality_pass,
        "strip_quality_reasons": strip_quality_reasons,
    }
    return CentralStripDiagnosticResult(panorama=panorama, metadata=metadata)
