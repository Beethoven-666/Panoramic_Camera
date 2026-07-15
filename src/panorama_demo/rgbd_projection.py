"""Metric RGB-D reprojection into an orthographic side-scan strip.

The projection contract is deliberately independent from Open3D.  Depth and
``camera_to_world`` translations are millimetres here; conversion to metres is
confined to the odometry adapter.  RGB values never determine geometry -- every
result carries explicit validity masks, so black is valid image content.

Only point splats are emitted.  In particular, this module never connects
neighbouring depth samples with triangles, so a foreground/background depth
discontinuity cannot be stretched across the strip.  Colliding splats are
resolved by a per-source z-buffer in the common world-normal coordinate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np


_DISTORTION_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
_SE3_ATOL = 1e-5


@runtime_checkable
class IntrinsicsLike(Protocol):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Sequence[float]


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Colour-camera intrinsics for an aligned RGB-D frame."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion": list(self.distortion),
        }


@dataclass(frozen=True)
class RGBDProjectionFrame:
    """One undistorted-or-distortable RGB-D source with a metric SE(3) pose.

    ``rgb`` channel order is preserved.  The current sequence renderer supplies
    OpenCV BGR arrays, while the geometry is unaffected by channel order.
    ``depth_mm`` and the translation column of ``camera_to_world`` are in mm.
    """

    frame_id: int
    rgb: np.ndarray
    depth_mm: np.ndarray
    camera_to_world: np.ndarray
    camera_to_world_unit: str = "mm"


@dataclass(frozen=True)
class EstimatedProjectionFootprint:
    """Cheap per-node coverage metadata used before full-resolution rendering."""

    frame_id: int
    camera_center_world_mm: tuple[float, float, float]
    camera_center_scan_x_mm: float
    scan_x_interval_mm: tuple[float, float]
    projected_height_mm: float
    sampled_world_bounds_mm: tuple[float, float, float, float]
    sample_count: int
    valid_depth_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "camera_center_world_mm": list(self.camera_center_world_mm),
            "camera_center_scan_x_mm": self.camera_center_scan_x_mm,
            "scan_x_interval_mm": list(self.scan_x_interval_mm),
            "projected_height_mm": self.projected_height_mm,
            "sampled_world_bounds_mm": list(self.sampled_world_bounds_mm),
            "sample_count": self.sample_count,
            "valid_depth_fraction": self.valid_depth_fraction,
        }


@dataclass(frozen=True)
class SideScanFootprintEstimate:
    scan_axis: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    normal_axis: tuple[float, float, float]
    footprints: tuple[EstimatedProjectionFootprint, ...]
    working_width: int

    def as_dict(self) -> dict[str, object]:
        return {
            "scan_axis": list(self.scan_axis),
            "up_axis": list(self.up_axis),
            "normal_axis": list(self.normal_axis),
            "working_width": self.working_width,
            "footprints": [item.as_dict() for item in self.footprints],
        }


@dataclass(frozen=True)
class ProjectionCanvas:
    """Common metric orthographic strip and its world-coordinate convention."""

    width: int
    height: int
    world_bounds: tuple[float, float, float, float]
    pixels_per_mm: float
    scan_axis: tuple[float, float, float]
    up_axis: tuple[float, float, float]
    normal_axis: tuple[float, float, float]
    maximum_depth_mm: float | None
    source_count: int
    canvas_megapixels: float
    aggregate_megapixels: float

    @property
    def world_bounds_mm(self) -> tuple[float, float, float, float]:
        return self.world_bounds

    @property
    def scan_axis_world(self) -> tuple[float, float, float]:
        return self.scan_axis

    @property
    def up_axis_world(self) -> tuple[float, float, float]:
        return self.up_axis

    @property
    def normal_axis_world(self) -> tuple[float, float, float]:
        return self.normal_axis

    def world_to_canvas(self, points_world_mm: np.ndarray) -> np.ndarray:
        """Project one or more world points to floating-point canvas coordinates."""

        points = np.asarray(points_world_mm, dtype=np.float64)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("World points must be finite and end in three coordinates")
        scan = np.asarray(self.scan_axis, dtype=np.float64)
        down = -np.asarray(self.up_axis, dtype=np.float64)
        min_scan, min_down, _, _ = self.world_bounds
        x = (points @ scan - min_scan) * self.pixels_per_mm
        y = (points @ down - min_down) * self.pixels_per_mm
        return np.stack((x, y), axis=-1)

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": "orthographic_side_scan",
            "world_unit": "mm",
            "width": self.width,
            "height": self.height,
            "world_bounds_mm": list(self.world_bounds),
            "pixels_per_mm": self.pixels_per_mm,
            "scan_axis": list(self.scan_axis),
            "up_axis": list(self.up_axis),
            "normal_axis": list(self.normal_axis),
            "maximum_projection_depth_mm": self.maximum_depth_mm,
            "canvas_x_coordinate": "dot(world_point_mm, scan_axis)",
            "canvas_y_coordinate": "dot(world_point_mm, -up_axis)",
            "surface_depth_coordinate": "dot(world_point_mm, normal_axis)",
            "source_count": self.source_count,
            "canvas_megapixels": self.canvas_megapixels,
            "aggregate_megapixels": self.aggregate_megapixels,
        }


@dataclass(frozen=True)
class ProjectedRGBDSource:
    """One source sampled once into the common full-resolution strip."""

    frame_id: int
    warped_rgb: np.ndarray
    valid_mask: np.ndarray
    surface_depth_mm: np.ndarray
    surface_depth_valid_mask: np.ndarray
    camera_depth_mm: np.ndarray
    camera_depth_valid_mask: np.ndarray
    projected_center_xy: tuple[float, float]
    valid_bbox: tuple[int, int, int, int]
    projected_height_px: int
    sampling_stats: dict[str, int | float | bool]
    camera_center_xy: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "surface_depth_coordinate": "dot(world_point_mm, normal_axis)",
            "camera_depth_coordinate": "source_color_camera_z_mm",
            "projected_center_xy": list(self.projected_center_xy),
            "camera_center_xy": list(self.camera_center_xy),
            "valid_bbox": list(self.valid_bbox),
            "projected_height_px": self.projected_height_px,
            "sampling_stats": dict(self.sampling_stats),
        }


@dataclass(frozen=True)
class RGBDProjectionResult:
    canvas: ProjectionCanvas
    sources: tuple[ProjectedRGBDSource, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "canvas": self.canvas.as_dict(),
            "sources": [source.as_dict() for source in self.sources],
        }


# Compatibility names for callers that prefer layout/source terminology.
ProjectionLayout = ProjectionCanvas
ProjectionSourceInput = RGBDProjectionFrame


def _value(source: object, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _distortion_values(value: object) -> tuple[float, ...]:
    distortion = _value(value, "distortion", ())
    if distortion is None:
        return ()
    if isinstance(distortion, Mapping):
        # Capture calibration stores a rational OpenCV model by named fields.
        return tuple(float(distortion.get(name, 0.0)) for name in _DISTORTION_NAMES)
    if hasattr(distortion, "k1"):
        return tuple(float(getattr(distortion, name, 0.0)) for name in _DISTORTION_NAMES)
    array = np.asarray(distortion, dtype=np.float64).reshape(-1)
    return tuple(float(item) for item in array)


def coerce_intrinsics(value: IntrinsicsLike | Mapping[str, object]) -> PinholeIntrinsics:
    """Validate a session intrinsics object without importing the session module."""

    try:
        intrinsics = PinholeIntrinsics(
            width=int(_value(value, "width")),
            height=int(_value(value, "height")),
            fx=float(_value(value, "fx")),
            fy=float(_value(value, "fy")),
            cx=float(_value(value, "cx")),
            cy=float(_value(value, "cy")),
            distortion=_distortion_values(value),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid colour camera intrinsics") from exc
    numeric = np.asarray(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, *intrinsics.distortion],
        dtype=np.float64,
    )
    if intrinsics.width <= 0 or intrinsics.height <= 0:
        raise ValueError("Camera intrinsic dimensions must be positive")
    if not np.isfinite(numeric).all() or intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("Camera intrinsics must be finite with positive focal lengths")
    if len(intrinsics.distortion) not in {0, 4, 5, 8, 12, 14}:
        raise ValueError("OpenCV distortion must contain 0, 4, 5, 8, 12, or 14 values")
    return intrinsics


def validate_camera_to_world(camera_to_world: np.ndarray) -> np.ndarray:
    """Return a checked 4x4 rigid camera-to-world transform (translation in mm)."""

    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be a finite 4x4 SE(3) matrix")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=_SE3_ATOL):
        raise ValueError("camera_to_world has an invalid homogeneous last row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_SE3_ATOL):
        raise ValueError("camera_to_world rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=_SE3_ATOL):
        raise ValueError("camera_to_world rotation determinant is not +1")
    return pose.copy()


def _validate_frame(
    frame: RGBDProjectionFrame, intrinsics: PinholeIntrinsics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.camera_to_world_unit != "mm":
        raise ValueError(
            "RGB-D projection requires camera_to_world translation explicitly in mm"
        )
    rgb = np.asarray(frame.rgb)
    depth = np.asarray(frame.depth_mm)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Each projection source must be a three-channel uint8 image")
    expected_shape = (intrinsics.height, intrinsics.width)
    if rgb.shape[:2] != expected_shape or depth.shape != expected_shape:
        raise ValueError("RGB and aligned depth dimensions must match colour intrinsics")
    if not np.issubdtype(depth.dtype, np.number) or np.issubdtype(depth.dtype, np.bool_):
        raise ValueError("Aligned depth must be a numeric millimetre array")
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size and np.any(finite_depth < 0):
        raise ValueError("Aligned depth cannot contain negative millimetre values")
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError(f"Frame {frame.frame_id} contains no valid aligned depth")
    pose = validate_camera_to_world(frame.camera_to_world)
    return rgb, depth, pose


def _normalize(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError(f"Cannot estimate a finite {label} axis")
    return np.asarray(vector, dtype=np.float64) / norm


def _estimate_world_axes(poses: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotations = np.stack([pose[:3, :3] for pose in poses], axis=0)
    centers = np.stack([pose[:3, 3] for pose in poses], axis=0)

    # Camera +y points down in an image, hence camera -y is world/image up.
    up_samples = -rotations[:, :, 1]
    up_sum = up_samples.sum(axis=0)
    if float(np.linalg.norm(up_sum)) < 0.25 * len(poses):
        raise ValueError("Camera up directions are mutually inconsistent")
    up = _normalize(up_sum, "camera-up")

    if len(poses) > 1:
        centered = centers - centers.mean(axis=0, keepdims=True)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        scan = vh[0] if singular_values[0] > 1e-6 else rotations[:, :, 0].mean(axis=0)
    else:
        scan = rotations[0, :, 0]
    scan = scan - float(np.dot(scan, up)) * up
    if float(np.linalg.norm(scan)) < 1e-9:
        scan = rotations[:, :, 0].mean(axis=0)
        scan = scan - float(np.dot(scan, up)) * up
    scan = _normalize(scan, "scan")

    displacement = centers[-1] - centers[0]
    if float(np.linalg.norm(displacement)) > 1e-6:
        if float(np.dot(scan, displacement)) < 0.0:
            scan = -scan
    elif float(np.dot(scan, rotations[:, :, 0].mean(axis=0))) < 0.0:
        scan = -scan

    normal = _normalize(np.cross(up, scan), "world-normal")
    mean_forward = rotations[:, :, 2].mean(axis=0)
    if float(np.dot(normal, mean_forward)) < 0.0:
        normal = -normal
    return scan, up, normal


def estimate_world_axes(
    poses: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the shared side-scan axes from checked real camera poses.

    This public wrapper keeps non-orthographic diagnostic projections on the
    same camera-up / scan / viewing-normal convention as the formal RGB-D
    projector.  It never reorders or otherwise alters the supplied trajectory.
    """

    checked = [validate_camera_to_world(pose) for pose in poses]
    if not checked:
        raise ValueError("At least one camera pose is required to estimate world axes")
    return _estimate_world_axes(checked)


def _camera_points(
    u: np.ndarray, v: np.ndarray, depth_mm: np.ndarray, intrinsics: PinholeIntrinsics
) -> np.ndarray:
    z = np.asarray(depth_mm, dtype=np.float64)
    x = (np.asarray(u, dtype=np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (np.asarray(v, dtype=np.float64) - intrinsics.cy) * z / intrinsics.fy
    return np.stack((x, y, z), axis=-1)


def _to_world(camera_points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return camera_points @ pose[:3, :3].T + pose[:3, 3]


def _sample_frame_points(
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: PinholeIntrinsics,
    working_width: int,
    maximum_depth_mm: float | None = None,
) -> tuple[np.ndarray, int, int]:
    stride = max(1, int(np.ceil(intrinsics.width / float(working_width))))
    rows = np.arange(0, intrinsics.height, stride, dtype=np.int32)
    cols = np.arange(0, intrinsics.width, stride, dtype=np.int32)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    sampled_depth = np.asarray(depth, dtype=np.float64)[yy, xx]
    valid = np.isfinite(sampled_depth) & (sampled_depth > 0)
    if maximum_depth_mm is not None:
        valid &= sampled_depth <= maximum_depth_mm
    if not valid.any():
        raise ValueError("Sparse projection footprint contains no valid aligned depth")
    points = _camera_points(xx[valid], yy[valid], sampled_depth[valid], intrinsics)
    return _to_world(points, pose), int(valid.sum()), int(valid.size)


def estimate_side_scan_footprints(
    frames: Sequence[RGBDProjectionFrame],
    intrinsics: IntrinsicsLike | Mapping[str, object],
    *,
    working_width: int = 640,
    maximum_depth_mm: float | None = None,
) -> SideScanFootprintEstimate:
    """Estimate node coverage without allocating full-resolution warped sources."""

    if not frames:
        raise ValueError("At least one RGB-D frame is required")
    if working_width <= 0:
        raise ValueError("working_width must be positive")
    if maximum_depth_mm is not None and (
        not np.isfinite(maximum_depth_mm) or maximum_depth_mm <= 0.0
    ):
        raise ValueError("maximum_depth_mm must be finite and positive")
    camera = coerce_intrinsics(intrinsics)
    validated = [_validate_frame(frame, camera) for frame in frames]
    ids = [int(frame.frame_id) for frame in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("Projection frame_id values must be unique")
    poses = [item[2] for item in validated]
    scan, up, normal = _estimate_world_axes(poses)
    down = -up
    footprints: list[EstimatedProjectionFootprint] = []
    for frame, (_, depth, pose) in zip(frames, validated, strict=True):
        points, sample_count, candidate_count = _sample_frame_points(
            depth, pose, camera, working_width, maximum_depth_mm
        )
        scan_values = points @ scan
        down_values = points @ down
        center = pose[:3, 3]
        bounds = (
            float(scan_values.min()),
            float(down_values.min()),
            float(scan_values.max()),
            float(down_values.max()),
        )
        footprints.append(
            EstimatedProjectionFootprint(
                frame_id=int(frame.frame_id),
                camera_center_world_mm=tuple(float(value) for value in center),
                camera_center_scan_x_mm=float(np.dot(center, scan)),
                scan_x_interval_mm=(bounds[0], bounds[2]),
                projected_height_mm=float(bounds[3] - bounds[1]),
                sampled_world_bounds_mm=bounds,
                sample_count=sample_count,
                valid_depth_fraction=float(sample_count / max(1, candidate_count)),
            )
        )
    return SideScanFootprintEstimate(
        scan_axis=tuple(float(value) for value in scan),
        up_axis=tuple(float(value) for value in up),
        normal_axis=tuple(float(value) for value in normal),
        footprints=tuple(footprints),
        working_width=min(working_width, camera.width),
    )


def _frustum_bounds(
    depth: np.ndarray,
    pose: np.ndarray,
    intrinsics: PinholeIntrinsics,
    scan: np.ndarray,
    down: np.ndarray,
    maximum_depth_mm: float | None = None,
) -> tuple[float, float, float, float]:
    valid_depth = np.asarray(depth, dtype=np.float64)
    valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 0)]
    if maximum_depth_mm is not None:
        valid_depth = valid_depth[valid_depth <= maximum_depth_mm]
    if valid_depth.size == 0:
        raise ValueError("Projection source has no depth within the selected range")
    z_min = float(valid_depth.min())
    z_max = float(valid_depth.max())
    u = np.array([0, intrinsics.width - 1] * 4, dtype=np.float64)
    v = np.array(
        [0, 0, intrinsics.height - 1, intrinsics.height - 1] * 2,
        dtype=np.float64,
    )
    z = np.array([z_min] * 4 + [z_max] * 4, dtype=np.float64)
    points = _to_world(_camera_points(u, v, z, intrinsics), pose)
    scan_values = points @ scan
    down_values = points @ down
    return (
        float(scan_values.min()),
        float(down_values.min()),
        float(scan_values.max()),
        float(down_values.max()),
    )


def _estimate_pixels_per_mm(
    depths: Sequence[np.ndarray],
    intrinsics: PinholeIntrinsics,
    maximum_depth_mm: float | None = None,
) -> float:
    samples: list[np.ndarray] = []
    for depth in depths:
        values = np.asarray(depth, dtype=np.float64)
        values = values[np.isfinite(values) & (values > 0)]
        if maximum_depth_mm is not None:
            values = values[values <= maximum_depth_mm]
        if values.size == 0:
            raise ValueError("Projection source has no depth within the selected range")
        if values.size > 8192:
            indices = np.linspace(0, values.size - 1, 8192, dtype=np.int64)
            values = values[indices]
        samples.append(values)
    combined = np.concatenate(samples)
    # A far-depth robust percentile keeps point spacing at or below one output
    # pixel for most samples, avoiding fabricated interpolation across holes.
    representative_far_depth = float(np.quantile(combined, 0.95))
    density = min(intrinsics.fx, intrinsics.fy) / representative_far_depth
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("Cannot estimate a finite orthographic pixel density")
    return float(density)


def estimate_projection_canvas(
    frames: Sequence[RGBDProjectionFrame],
    intrinsics: IntrinsicsLike | Mapping[str, object],
    *,
    max_canvas_megapixels: float = 200.0,
    max_aggregate_megapixels: float | None = None,
    adapt_density_to_budget: bool = False,
    maximum_depth_mm: float | None = None,
) -> ProjectionCanvas:
    """Build a conservative common canvas and enforce memory limits up front."""

    if not frames:
        raise ValueError("At least one RGB-D frame is required")
    if not np.isfinite(max_canvas_megapixels) or max_canvas_megapixels <= 0.0:
        raise ValueError("max_canvas_megapixels must be finite and positive")
    aggregate_limit = (
        max_canvas_megapixels
        if max_aggregate_megapixels is None
        else max_aggregate_megapixels
    )
    if not np.isfinite(aggregate_limit) or aggregate_limit <= 0.0:
        raise ValueError("max_aggregate_megapixels must be finite and positive")
    if maximum_depth_mm is not None and (
        not np.isfinite(maximum_depth_mm) or maximum_depth_mm <= 0.0
    ):
        raise ValueError("maximum_depth_mm must be finite and positive")
    camera = coerce_intrinsics(intrinsics)
    validated = [_validate_frame(frame, camera) for frame in frames]
    ids = [int(frame.frame_id) for frame in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("Projection frame_id values must be unique")
    poses = [item[2] for item in validated]
    scan, up, normal = _estimate_world_axes(poses)
    down = -up
    density = _estimate_pixels_per_mm(
        [item[1] for item in validated], camera, maximum_depth_mm
    )
    bounds = [
        _frustum_bounds(depth, pose, camera, scan, down, maximum_depth_mm)
        for _, depth, pose in validated
    ]
    min_scan = min(item[0] for item in bounds)
    min_down = min(item[1] for item in bounds)
    max_scan = max(item[2] for item in bounds)
    max_down = max(item[3] for item in bounds)
    # One empty pixel around the conservative frusta absorbs round-to-nearest.
    padding_mm = 1.0 / density
    world_bounds = (
        min_scan - padding_mm,
        min_down - padding_mm,
        max_scan + padding_mm,
        max_down + padding_mm,
    )
    width_float = (world_bounds[2] - world_bounds[0]) * density
    height_float = (world_bounds[3] - world_bounds[1]) * density
    if adapt_density_to_budget:
        # RGB-D points are splatted directly into this metric canvas.  A close
        # foreground plus a few distant depth samples can otherwise request a
        # much finer canvas than the bounded multi-source working set permits.
        # Reduce only the sampling density; preserve the observed metric bounds
        # and do not crop, invent pixels, or exceed the hard resource budget.
        target_pixels = 0.98 * min(
            max_canvas_megapixels,
            aggregate_limit / max(1, len(frames)),
        ) * 1_000_000.0
        # Do not create an orthographic grid substantially denser than the
        # real RGB-D samples feeding it.  Sparse point splats at an inflated
        # output density look like black holes; reducing this grid is metric
        # resampling, not colour/depth hole fabrication.
        source_sample_budget = 0.85 * sum(
            item[0].shape[0] * item[0].shape[1] for item in validated
        )
        target_pixels = min(target_pixels, source_sample_budget)
        requested_pixels = width_float * height_float
        if requested_pixels > target_pixels:
            density *= math.sqrt(target_pixels / requested_pixels)
            padding_mm = 1.0 / density
            world_bounds = (
                min_scan - padding_mm,
                min_down - padding_mm,
                max_scan + padding_mm,
                max_down + padding_mm,
            )
            width_float = (world_bounds[2] - world_bounds[0]) * density
            height_float = (world_bounds[3] - world_bounds[1]) * density
    if not np.isfinite([width_float, height_float]).all():
        raise MemoryError("Orthographic world bounds produce a non-finite canvas")
    width = int(np.ceil(width_float)) + 1
    height = int(np.ceil(height_float)) + 1
    if width <= 0 or height <= 0 or width > 2_147_483_647 or height > 2_147_483_647:
        raise MemoryError("Orthographic canvas dimensions are outside safe integer limits")
    canvas_megapixels = width * height / 1_000_000.0
    if canvas_megapixels > max_canvas_megapixels:
        raise MemoryError(
            f"Orthographic canvas is {width}x{height} ({canvas_megapixels:.1f} MP), "
            f"above the {max_canvas_megapixels:.1f} MP limit"
        )
    aggregate_megapixels = canvas_megapixels * len(frames)
    if aggregate_megapixels > aggregate_limit:
        raise MemoryError(
            "Orthographic projection aggregate working set is "
            f"{width}x{height} x {len(frames)} sources "
            f"({aggregate_megapixels:.1f} aggregate MP), above the "
            f"{aggregate_limit:.1f} MP limit"
        )
    return ProjectionCanvas(
        width=width,
        height=height,
        world_bounds=world_bounds,
        pixels_per_mm=density,
        scan_axis=tuple(float(value) for value in scan),
        up_axis=tuple(float(value) for value in up),
        normal_axis=tuple(float(value) for value in normal),
        maximum_depth_mm=maximum_depth_mm,
        source_count=len(frames),
        canvas_megapixels=float(canvas_megapixels),
        aggregate_megapixels=float(aggregate_megapixels),
    )


def _undistortion_maps(
    intrinsics: PinholeIntrinsics,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not intrinsics.distortion or not np.any(np.asarray(intrinsics.distortion)):
        return None
    return cv2.initUndistortRectifyMap(
        intrinsics.matrix,
        np.asarray(intrinsics.distortion, dtype=np.float64),
        None,
        intrinsics.matrix,
        (intrinsics.width, intrinsics.height),
        cv2.CV_32FC1,
    )


def _undistort_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if maps is None:
        return rgb, np.asarray(depth, dtype=np.float32), np.ones(depth.shape, dtype=bool)
    map_x, map_y = maps
    undistorted_rgb = cv2.remap(
        rgb,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    undistorted_depth = cv2.remap(
        np.asarray(depth, dtype=np.float32),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    geometric_valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= rgb.shape[1] - 1)
        & (map_y >= 0.0)
        & (map_y <= rgb.shape[0] - 1)
    )
    return undistorted_rgb, undistorted_depth, geometric_valid


def _depth_discontinuity_count(depth: np.ndarray, valid: np.ndarray) -> int:
    def count(first: np.ndarray, second: np.ndarray, pair_valid: np.ndarray) -> int:
        nearer = np.minimum(first, second)
        threshold = np.maximum(50.0, nearer * 0.05)
        return int(np.count_nonzero(pair_valid & (np.abs(first - second) > threshold)))

    horizontal = count(depth[:, :-1], depth[:, 1:], valid[:, :-1] & valid[:, 1:])
    vertical = count(depth[:-1, :], depth[1:, :], valid[:-1, :] & valid[1:, :])
    return horizontal + vertical


def _project_source(
    frame: RGBDProjectionFrame,
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    maps: tuple[np.ndarray, np.ndarray] | None,
    *,
    chunk_rows: int,
    maximum_depth_mm: float | None = None,
) -> ProjectedRGBDSource:
    rgb, depth, pose = _validate_frame(frame, intrinsics)
    rgb, depth, geometric_valid = _undistort_rgbd(rgb, depth, maps)
    depth_valid = geometric_valid & np.isfinite(depth) & (depth > 0)
    if maximum_depth_mm is not None:
        depth_valid &= depth <= maximum_depth_mm
    if not depth_valid.any():
        raise RuntimeError(f"Frame {frame.frame_id} has no valid depth after undistortion")

    warped_rgb = np.zeros((canvas.height, canvas.width, 3), dtype=np.uint8)
    surface_depth = np.full((canvas.height, canvas.width), np.inf, dtype=np.float32)
    camera_depth = np.zeros((canvas.height, canvas.width), dtype=np.float32)
    valid_mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
    flat_rgb = warped_rgb.reshape(-1, 3)
    flat_surface = surface_depth.reshape(-1)
    flat_camera_depth = camera_depth.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    scan = np.asarray(canvas.scan_axis, dtype=np.float64)
    down = -np.asarray(canvas.up_axis, dtype=np.float64)
    normal = np.asarray(canvas.normal_axis, dtype=np.float64)
    min_scan, min_down, _, _ = canvas.world_bounds
    projected_sample_count = 0
    out_of_canvas_count = 0

    for row_start in range(0, intrinsics.height, chunk_rows):
        row_stop = min(intrinsics.height, row_start + chunk_rows)
        local_valid = depth_valid[row_start:row_stop]
        local_y, x = np.nonzero(local_valid)
        if not x.size:
            continue
        y = local_y + row_start
        z = depth[y, x].astype(np.float64)
        camera_points = _camera_points(x, y, z, intrinsics)
        world = _to_world(camera_points, pose)
        scan_value = world @ scan
        down_value = world @ down
        normal_value = world @ normal
        canvas_x = np.rint((scan_value - min_scan) * canvas.pixels_per_mm).astype(
            np.int64
        )
        canvas_y = np.rint((down_value - min_down) * canvas.pixels_per_mm).astype(
            np.int64
        )
        inside = (
            np.isfinite(normal_value)
            & (canvas_x >= 0)
            & (canvas_x < canvas.width)
            & (canvas_y >= 0)
            & (canvas_y < canvas.height)
        )
        out_of_canvas_count += int(np.count_nonzero(~inside))
        if not inside.any():
            continue
        x = x[inside]
        y = y[inside]
        z = z[inside]
        canvas_x = canvas_x[inside]
        canvas_y = canvas_y[inside]
        normal_value = normal_value[inside]
        flat_index = canvas_y * canvas.width + canvas_x
        source_index = y.astype(np.int64) * intrinsics.width + x
        order = np.lexsort((source_index, normal_value, flat_index))
        sorted_flat = flat_index[order]
        first = np.empty(sorted_flat.size, dtype=bool)
        first[0] = True
        first[1:] = sorted_flat[1:] != sorted_flat[:-1]
        candidates = order[first]
        candidate_flat = flat_index[candidates]
        candidate_depth = normal_value[candidates].astype(np.float32)
        candidate_camera_depth = z[candidates].astype(np.float32)
        replace = candidate_depth < flat_surface[candidate_flat]
        destination = candidate_flat[replace]
        selected = candidates[replace]
        flat_surface[destination] = candidate_depth[replace]
        flat_camera_depth[destination] = candidate_camera_depth[replace]
        flat_rgb[destination] = rgb[y[selected], x[selected]]
        flat_valid[destination] = 255
        projected_sample_count += int(inside.sum())

    if out_of_canvas_count:
        raise RuntimeError(
            f"Frame {frame.frame_id} projected {out_of_canvas_count} valid samples "
            "outside its conservative orthographic canvas"
        )
    selected_pixel_count = int(np.count_nonzero(valid_mask))
    if selected_pixel_count == 0:
        raise RuntimeError(f"Frame {frame.frame_id} produced no valid projected surface")
    surface_depth[valid_mask == 0] = 0.0
    surface_depth_valid_mask = valid_mask.copy()
    camera_depth_valid_mask = valid_mask.copy()
    ys, xs = np.nonzero(valid_mask)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    center = (float(np.median(xs)), float(np.median(ys)))
    camera_center = canvas.world_to_canvas(pose[:3, 3])
    raw_valid_count = int(np.count_nonzero(np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)))
    undistorted_valid_count = int(np.count_nonzero(depth_valid))
    stats: dict[str, int | float | bool] = {
        "input_pixel_count": int(intrinsics.width * intrinsics.height),
        "input_valid_depth_pixel_count": raw_valid_count,
        "undistorted_valid_depth_pixel_count": undistorted_valid_count,
        "projected_sample_count": projected_sample_count,
        "selected_zbuffer_pixel_count": selected_pixel_count,
        "zbuffer_collision_count": projected_sample_count - selected_pixel_count,
        "out_of_canvas_sample_count": out_of_canvas_count,
        "depth_discontinuity_edge_count": _depth_discontinuity_count(depth, depth_valid),
        "valid_depth_fraction": float(
            undistorted_valid_count / (intrinsics.width * intrinsics.height)
        ),
        "projected_sampling_ratio": float(
            selected_pixel_count / max(1, undistorted_valid_count)
        ),
        "point_splat_only": True,
    }
    return ProjectedRGBDSource(
        frame_id=int(frame.frame_id),
        warped_rgb=warped_rgb,
        valid_mask=valid_mask,
        surface_depth_mm=surface_depth,
        surface_depth_valid_mask=surface_depth_valid_mask,
        camera_depth_mm=camera_depth,
        camera_depth_valid_mask=camera_depth_valid_mask,
        projected_center_xy=center,
        valid_bbox=(x0, y0, x1, y1),
        projected_height_px=y1 - y0,
        sampling_stats=stats,
        camera_center_xy=(float(camera_center[0]), float(camera_center[1])),
    )


def project_rgbd_source(
    frame: RGBDProjectionFrame,
    intrinsics: IntrinsicsLike | Mapping[str, object],
    canvas: ProjectionCanvas,
    *,
    chunk_rows: int = 128,
    maximum_depth_mm: float | None = None,
) -> ProjectedRGBDSource:
    """Project one selected source once into an already-budgeted canvas."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    camera = coerce_intrinsics(intrinsics)
    selected_maximum_depth = (
        canvas.maximum_depth_mm if maximum_depth_mm is None else maximum_depth_mm
    )
    return _project_source(
        frame,
        camera,
        canvas,
        _undistortion_maps(camera),
        chunk_rows=chunk_rows,
        maximum_depth_mm=selected_maximum_depth,
    )


def project_selected_rgbd_sources(
    frames: Sequence[RGBDProjectionFrame],
    intrinsics: IntrinsicsLike | Mapping[str, object],
    *,
    max_canvas_megapixels: float = 200.0,
    max_aggregate_megapixels: float | None = None,
    adapt_density_to_budget: bool = False,
    chunk_rows: int = 128,
    maximum_depth_mm: float | None = None,
) -> RGBDProjectionResult:
    """Project only final render sources; no dense-frame point clouds are retained."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    camera = coerce_intrinsics(intrinsics)
    canvas = estimate_projection_canvas(
        frames,
        camera,
        max_canvas_megapixels=max_canvas_megapixels,
        max_aggregate_megapixels=max_aggregate_megapixels,
        adapt_density_to_budget=adapt_density_to_budget,
        maximum_depth_mm=maximum_depth_mm,
    )
    maps = _undistortion_maps(camera)
    projected = tuple(
        _project_source(
            frame,
            camera,
            canvas,
            maps,
            chunk_rows=chunk_rows,
            maximum_depth_mm=maximum_depth_mm,
        )
        for frame in frames
    )
    return RGBDProjectionResult(canvas=canvas, sources=projected)


# Concise public aliases for sequence orchestration and external diagnostics.
project_rgbd_side_scan = project_selected_rgbd_sources
project_rgbd_frames = project_selected_rgbd_sources
