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

from .cuda_backend import (
    pinhole_unproject as accelerated_pinhole_unproject,
    remap as accelerated_remap,
    transform_points as accelerated_transform_points,
)

_DISTORTION_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
_SE3_ATOL = 1e-5
_SURFACE_CONTINUITY_ABSOLUTE_MM = 20.0
_SURFACE_CONTINUITY_RELATIVE = 0.02
_SURFACE_FOOTPRINT_MAX_HALF_EXTENT_PX = 3.0
_SURFACE_FOOTPRINT_MIN_JACOBIAN = 1e-4
_SURFACE_FOOTPRINT_MAX_SCALE_FACTOR = 1.75


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
            "millimetres_per_pixel": 1.0 / self.pixels_per_mm,
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
class CompactProjectedRGBDSource:
    """One source stored only inside its valid global-canvas bounding box.

    Array coordinates are local to ``valid_bbox`` while all reported geometry
    (the bounding box, projected centre, and camera centre) remains in the
    common canvas coordinate system.
    """

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

    @property
    def canvas_slices(self) -> tuple[slice, slice]:
        x0, y0, x1, y1 = self.valid_bbox
        return slice(y0, y1), slice(x0, x1)

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
    return accelerated_pinhole_unproject(
        u,
        v,
        depth_mm,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )


def _to_world(camera_points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return accelerated_transform_points(
        camera_points,
        pose[:3, :3],
        pose[:3, 3],
    )


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
    if maximum_depth_mm is None:
        z_min = float(valid_depth.min())
        z_max = float(valid_depth.max())
    else:
        # The canvas may be estimated from a nearest-neighbour depth preview
        # and later receive the complete source.  Preview extrema are not a
        # conservative bound: a sparse farther full-resolution sample can be
        # missed and then land outside the canvas.  Bound the complete
        # configured viewing frustum, including the camera centre, whenever a
        # formal maximum range is known.
        z_min = 0.0
        z_max = float(maximum_depth_mm)
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
    millimetres_per_pixel: float | None = None,
    maximum_resident_sources: int | None = None,
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
    if millimetres_per_pixel is not None and (
        not np.isfinite(millimetres_per_pixel)
        or millimetres_per_pixel <= 0.0
    ):
        raise ValueError("millimetres_per_pixel must be finite and positive")
    camera = coerce_intrinsics(intrinsics)
    validated = [_validate_frame(frame, camera) for frame in frames]
    ids = [int(frame.frame_id) for frame in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("Projection frame_id values must be unique")
    poses = [item[2] for item in validated]
    scan, up, normal = _estimate_world_axes(poses)
    down = -up
    density = (
        1.0 / float(millimetres_per_pixel)
        if millimetres_per_pixel is not None
        else _estimate_pixels_per_mm(
            [item[1] for item in validated], camera, maximum_depth_mm
        )
    )
    resident_sources = (
        len(frames)
        if maximum_resident_sources is None
        else int(maximum_resident_sources)
    )
    if resident_sources <= 0 or resident_sources > len(frames):
        raise ValueError(
            "maximum_resident_sources must be within the source count"
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
            aggregate_limit / resident_sources,
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
    aggregate_megapixels = canvas_megapixels * resident_sources
    if aggregate_megapixels > aggregate_limit:
        raise MemoryError(
            "Orthographic projection aggregate working set is "
            f"{width}x{height} x {resident_sources} resident sources "
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


def prepare_rgbd_undistortion_maps(
    intrinsics: IntrinsicsLike | Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Prepare immutable calibration maps for repeated source projection."""

    return _undistortion_maps(coerce_intrinsics(intrinsics))


def _undistort_rgbd(
    rgb: np.ndarray,
    depth: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if maps is None:
        return rgb, np.asarray(depth, dtype=np.float32), np.ones(depth.shape, dtype=bool)
    map_x, map_y = maps
    undistorted_rgb = accelerated_remap(
        rgb,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    undistorted_depth = accelerated_remap(
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


def _continuous_surface_support(
    depth: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify source samples that may expose a finite sensor-pixel footprint.

    A projected depth sample is always retained as a measured point.  Its
    finite pixel footprint is exposed only when the complete 3x3 source
    neighbourhood is measured and lies on one locally continuous depth layer.
    This deliberately leaves an invalid guard around missing depth and rejects
    both sides of a foreground/background step instead of treating either as a
    colour/depth hole to be filled.
    """

    measured = np.asarray(valid, dtype=bool)
    values = np.asarray(depth, dtype=np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    complete_neighbourhood = cv2.erode(
        measured.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    high = cv2.dilate(
        np.where(measured, values, np.float32(0.0)),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    low_sentinel = np.float32(np.finfo(np.float32).max)
    low = cv2.erode(
        np.where(measured, values, low_sentinel),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=float(low_sentinel),
    )
    span = high - low
    tolerance = np.maximum(
        np.float32(_SURFACE_CONTINUITY_ABSOLUTE_MM),
        np.float32(_SURFACE_CONTINUITY_RELATIVE) * np.maximum(values, 0.0),
    )
    continuous = complete_neighbourhood & np.isfinite(span) & (span <= tolerance)
    depth_edge = complete_neighbourhood & ~continuous
    invalid_neighbourhood = measured & ~complete_neighbourhood
    return continuous, depth_edge, invalid_neighbourhood


def _project_source_compact(
    frame: RGBDProjectionFrame,
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    maps: tuple[np.ndarray, np.ndarray] | None,
    *,
    chunk_rows: int,
    maximum_depth_mm: float | None = None,
) -> CompactProjectedRGBDSource:
    rgb, depth, pose = _validate_frame(frame, intrinsics)
    rgb, depth, geometric_valid = _undistort_rgbd(rgb, depth, maps)
    depth_valid = geometric_valid & np.isfinite(depth) & (depth > 0)
    if maximum_depth_mm is not None:
        depth_valid &= depth <= maximum_depth_mm
    if not depth_valid.any():
        raise RuntimeError(f"Frame {frame.frame_id} has no valid depth after undistortion")

    scan = np.asarray(canvas.scan_axis, dtype=np.float64)
    down = -np.asarray(canvas.up_axis, dtype=np.float64)
    normal = np.asarray(canvas.normal_axis, dtype=np.float64)
    min_scan, min_down, _, _ = canvas.world_bounds
    (
        continuous_source_support,
        rejected_depth_edge_support,
        rejected_invalid_neighbourhood_support,
    ) = _continuous_surface_support(depth, depth_valid)
    projected_sample_count = 0
    out_of_canvas_count = 0
    footprint_out_of_canvas_count = 0
    footprint_candidate_count = 0
    continuous_surface_sample_count = 0
    rejected_fold_sample_count = 0
    rejected_degenerate_sample_count = 0
    rejected_overscale_sample_count = 0
    candidate_chunks: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

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
        canvas_x_float = (scan_value - min_scan) * canvas.pixels_per_mm
        canvas_y_float = (down_value - min_down) * canvas.pixels_per_mm
        canvas_x = np.rint(canvas_x_float).astype(np.int64)
        canvas_y = np.rint(canvas_y_float).astype(np.int64)
        inside = (
            np.isfinite(normal_value)
            & np.isfinite(canvas_x_float)
            & np.isfinite(canvas_y_float)
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
        canvas_x_float = canvas_x_float[inside]
        canvas_y_float = canvas_y_float[inside]
        canvas_x = canvas_x[inside]
        canvas_y = canvas_y[inside]
        normal_value = normal_value[inside]
        source_index = y.astype(np.int64) * intrinsics.width + x
        candidate_x = [canvas_x]
        candidate_y = [canvas_y]
        candidate_normal = [normal_value]
        candidate_camera_depth = [z]
        candidate_color = [rgb[y, x]]
        candidate_distance = [
            np.square(canvas_x - canvas_x_float)
            + np.square(canvas_y - canvas_y_float)
        ]
        candidate_source_index = [source_index]
        candidate_is_footprint = [np.zeros(canvas_x.shape, dtype=bool)]

        footprint_source = continuous_source_support[y, x]
        if np.any(footprint_source):
            # Reproject a one-row halo so the local source-pixel Jacobian is
            # derived from measured neighbours on both axes without retaining
            # a full-frame world-coordinate raster.
            halo_start = max(0, row_start - 1)
            halo_stop = min(intrinsics.height, row_stop + 1)
            halo_valid = depth_valid[halo_start:halo_stop]
            halo_local_y, halo_x = np.nonzero(halo_valid)
            halo_y = halo_local_y + halo_start
            halo_z = depth[halo_y, halo_x].astype(np.float64)
            halo_camera = _camera_points(halo_x, halo_y, halo_z, intrinsics)
            halo_world = _to_world(halo_camera, pose)
            halo_projection_x = np.full(halo_valid.shape, np.nan, dtype=np.float64)
            halo_projection_y = np.full(halo_valid.shape, np.nan, dtype=np.float64)
            halo_projection_x[halo_local_y, halo_x] = (
                (halo_world @ scan) - min_scan
            ) * canvas.pixels_per_mm
            halo_projection_y[halo_local_y, halo_x] = (
                (halo_world @ down) - min_down
            ) * canvas.pixels_per_mm

            selected_indices = np.flatnonzero(footprint_source)
            selected_x = x[selected_indices]
            selected_y = y[selected_indices]
            selected_local_y = selected_y - halo_start
            centre_x = canvas_x_float[selected_indices]
            centre_y = canvas_y_float[selected_indices]
            left_x = halo_projection_x[selected_local_y, selected_x - 1]
            left_y = halo_projection_y[selected_local_y, selected_x - 1]
            right_x = halo_projection_x[selected_local_y, selected_x + 1]
            right_y = halo_projection_y[selected_local_y, selected_x + 1]
            up_x = halo_projection_x[selected_local_y - 1, selected_x]
            up_y = halo_projection_y[selected_local_y - 1, selected_x]
            down_x = halo_projection_x[selected_local_y + 1, selected_x]
            down_y = halo_projection_y[selected_local_y + 1, selected_x]
            derivative_u_x = 0.5 * (right_x - left_x)
            derivative_u_y = 0.5 * (right_y - left_y)
            derivative_v_x = 0.5 * (down_x - up_x)
            derivative_v_y = 0.5 * (down_y - up_y)
            jacobian = (
                derivative_u_x * derivative_v_y
                - derivative_u_y * derivative_v_x
            )
            finite_jacobian = (
                np.isfinite(derivative_u_x)
                & np.isfinite(derivative_u_y)
                & np.isfinite(derivative_v_x)
                & np.isfinite(derivative_v_y)
                & np.isfinite(jacobian)
            )
            fold = finite_jacobian & (
                jacobian < -_SURFACE_FOOTPRINT_MIN_JACOBIAN
            )
            degenerate = ~finite_jacobian | (
                np.abs(jacobian) <= _SURFACE_FOOTPRINT_MIN_JACOBIAN
            )
            derivative_u_norm = np.hypot(derivative_u_x, derivative_u_y)
            derivative_v_norm = np.hypot(derivative_v_x, derivative_v_y)
            selected_z = z[selected_indices]
            expected_scale = np.maximum(
                1.0,
                _SURFACE_FOOTPRINT_MAX_SCALE_FACTOR
                * np.maximum(
                    selected_z / intrinsics.fx,
                    selected_z / intrinsics.fy,
                )
                * canvas.pixels_per_mm,
            )
            half_extent_x = 0.5 * (
                np.abs(derivative_u_x) + np.abs(derivative_v_x)
            )
            half_extent_y = 0.5 * (
                np.abs(derivative_u_y) + np.abs(derivative_v_y)
            )
            overscale = (
                (derivative_u_norm > expected_scale)
                | (derivative_v_norm > expected_scale)
                | (half_extent_x > _SURFACE_FOOTPRINT_MAX_HALF_EXTENT_PX)
                | (half_extent_y > _SURFACE_FOOTPRINT_MAX_HALF_EXTENT_PX)
            )
            accepted = ~(fold | degenerate | overscale)
            rejected_fold_sample_count += int(np.count_nonzero(fold))
            rejected_degenerate_sample_count += int(np.count_nonzero(degenerate))
            rejected_overscale_sample_count += int(
                np.count_nonzero(overscale & ~fold & ~degenerate)
            )
            continuous_surface_sample_count += int(np.count_nonzero(accepted))

            if np.any(accepted):
                accepted_indices = selected_indices[accepted]
                accepted_centre_x = centre_x[accepted]
                accepted_centre_y = centre_y[accepted]
                accepted_du_x = derivative_u_x[accepted]
                accepted_du_y = derivative_u_y[accepted]
                accepted_dv_x = derivative_v_x[accepted]
                accepted_dv_y = derivative_v_y[accepted]
                accepted_jacobian = jacobian[accepted]
                accepted_base_x = canvas_x[accepted_indices]
                accepted_base_y = canvas_y[accepted_indices]
                # Most near-field sensor pixels are already substantially
                # smaller than one 2 mm output cell.  Before evaluating any
                # neighbouring destination, reject samples whose conservative
                # axis-aligned footprint cannot reach another integer grid
                # centre.  This is an exact necessary condition, not a change
                # to the accepted physical footprint.
                accepted_half_x = half_extent_x[accepted]
                accepted_half_y = half_extent_y[accepted]
                nearest_other_x = 1.0 - np.abs(
                    accepted_centre_x - accepted_base_x
                )
                nearest_other_y = 1.0 - np.abs(
                    accepted_centre_y - accepted_base_y
                )
                may_cover_additional_grid_cell = (
                    (accepted_half_x + 1e-9 >= nearest_other_x)
                    | (accepted_half_y + 1e-9 >= nearest_other_y)
                )
                if not np.any(may_cover_additional_grid_cell):
                    may_cover_additional_grid_cell = np.zeros(
                        accepted_indices.shape,
                        dtype=bool,
                    )
                accepted_indices = accepted_indices[
                    may_cover_additional_grid_cell
                ]
                accepted_centre_x = accepted_centre_x[
                    may_cover_additional_grid_cell
                ]
                accepted_centre_y = accepted_centre_y[
                    may_cover_additional_grid_cell
                ]
                accepted_du_x = accepted_du_x[may_cover_additional_grid_cell]
                accepted_du_y = accepted_du_y[may_cover_additional_grid_cell]
                accepted_dv_x = accepted_dv_x[may_cover_additional_grid_cell]
                accepted_dv_y = accepted_dv_y[may_cover_additional_grid_cell]
                accepted_jacobian = accepted_jacobian[
                    may_cover_additional_grid_cell
                ]
                accepted_base_x = accepted_base_x[
                    may_cover_additional_grid_cell
                ]
                accepted_base_y = accepted_base_y[
                    may_cover_additional_grid_cell
                ]
                maximum_offset = int(
                    math.ceil(_SURFACE_FOOTPRINT_MAX_HALF_EXTENT_PX)
                )
                for offset_y in range(-maximum_offset, maximum_offset + 1):
                    for offset_x in range(-maximum_offset, maximum_offset + 1):
                        if offset_x == 0 and offset_y == 0:
                            continue
                        destination_x = accepted_base_x + offset_x
                        destination_y = accepted_base_y + offset_y
                        delta_x = destination_x - accepted_centre_x
                        delta_y = destination_y - accepted_centre_y
                        coordinate_u = (
                            delta_x * accepted_dv_y
                            - delta_y * accepted_dv_x
                        ) / accepted_jacobian
                        coordinate_v = (
                            accepted_du_x * delta_y
                            - accepted_du_y * delta_x
                        ) / accepted_jacobian
                        covered = (
                            (np.abs(coordinate_u) <= 0.5 + 1e-9)
                            & (np.abs(coordinate_v) <= 0.5 + 1e-9)
                        )
                        inside_footprint = (
                            covered
                            & (destination_x >= 0)
                            & (destination_x < canvas.width)
                            & (destination_y >= 0)
                            & (destination_y < canvas.height)
                        )
                        footprint_out_of_canvas_count += int(
                            np.count_nonzero(covered & ~inside_footprint)
                        )
                        if not np.any(inside_footprint):
                            continue
                        chosen = accepted_indices[inside_footprint]
                        candidate_x.append(destination_x[inside_footprint])
                        candidate_y.append(destination_y[inside_footprint])
                        candidate_normal.append(normal_value[chosen])
                        candidate_camera_depth.append(z[chosen])
                        candidate_color.append(rgb[y[chosen], x[chosen]])
                        candidate_distance.append(
                            np.square(delta_x[inside_footprint])
                            + np.square(delta_y[inside_footprint])
                        )
                        candidate_source_index.append(source_index[chosen])
                        candidate_is_footprint.append(
                            np.ones(np.count_nonzero(inside_footprint), dtype=bool)
                        )
                        footprint_candidate_count += int(
                            np.count_nonzero(inside_footprint)
                        )

        merged_x = np.concatenate(candidate_x)
        merged_y = np.concatenate(candidate_y)
        merged_normal = np.concatenate(candidate_normal)
        merged_camera_depth = np.concatenate(candidate_camera_depth)
        merged_color = np.concatenate(candidate_color)
        merged_distance = np.concatenate(candidate_distance)
        merged_source_index = np.concatenate(candidate_source_index)
        merged_is_footprint = np.concatenate(candidate_is_footprint)
        flat_index = merged_y * canvas.width + merged_x
        order = np.lexsort(
            (
                merged_source_index,
                merged_distance,
                merged_normal,
                flat_index,
            )
        )
        sorted_flat = flat_index[order]
        first = np.empty(sorted_flat.size, dtype=bool)
        first[0] = True
        first[1:] = sorted_flat[1:] != sorted_flat[:-1]
        candidates = order[first]
        candidate_chunks.append(
            (
                merged_x[candidates].copy(),
                merged_y[candidates].copy(),
                merged_normal[candidates].astype(np.float32),
                merged_camera_depth[candidates].astype(np.float32),
                merged_color[candidates].copy(),
                merged_distance[candidates].astype(np.float32),
                merged_source_index[candidates].astype(np.int32),
                merged_is_footprint[candidates].copy(),
            )
        )
        projected_sample_count += int(inside.sum())

    if out_of_canvas_count:
        raise RuntimeError(
            f"Frame {frame.frame_id} projected {out_of_canvas_count} valid samples "
            "outside its conservative orthographic canvas"
        )
    if not candidate_chunks:
        raise RuntimeError(f"Frame {frame.frame_id} produced no valid projected surface")

    x0 = min(int(chunk[0].min()) for chunk in candidate_chunks)
    y0 = min(int(chunk[1].min()) for chunk in candidate_chunks)
    x1 = max(int(chunk[0].max()) for chunk in candidate_chunks) + 1
    y1 = max(int(chunk[1].max()) for chunk in candidate_chunks) + 1
    compact_width = x1 - x0
    compact_height = y1 - y0
    warped_rgb = np.zeros((compact_height, compact_width, 3), dtype=np.uint8)
    surface_depth = np.full(
        (compact_height, compact_width), np.inf, dtype=np.float32
    )
    camera_depth = np.zeros((compact_height, compact_width), dtype=np.float32)
    valid_mask = np.zeros((compact_height, compact_width), dtype=np.uint8)
    winner_distance = np.full(
        (compact_height, compact_width), np.inf, dtype=np.float32
    )
    winner_source_index = np.full(
        (compact_height, compact_width), np.iinfo(np.int32).max, dtype=np.int32
    )
    winner_is_footprint = np.zeros(
        (compact_height, compact_width), dtype=bool
    )
    flat_rgb = warped_rgb.reshape(-1, 3)
    flat_surface = surface_depth.reshape(-1)
    flat_camera_depth = camera_depth.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    flat_distance = winner_distance.reshape(-1)
    flat_source_index = winner_source_index.reshape(-1)
    flat_is_footprint = winner_is_footprint.reshape(-1)
    for (
        canvas_x,
        canvas_y,
        candidate_depth,
        candidate_camera_depth,
        colors,
        candidate_distance,
        candidate_source_index,
        candidate_is_footprint,
    ) in candidate_chunks:
        destination = (
            (canvas_y - y0) * compact_width + (canvas_x - x0)
        ).astype(np.int64)
        current_depth = flat_surface[destination]
        current_distance = flat_distance[destination]
        current_source_index = flat_source_index[destination]
        same_depth = candidate_depth == current_depth
        same_distance = candidate_distance == current_distance
        replace = (
            (candidate_depth < current_depth)
            | (same_depth & (candidate_distance < current_distance))
            | (
                same_depth
                & same_distance
                & (candidate_source_index < current_source_index)
            )
        )
        selected_destination = destination[replace]
        flat_surface[selected_destination] = candidate_depth[replace]
        flat_camera_depth[selected_destination] = candidate_camera_depth[replace]
        flat_rgb[selected_destination] = colors[replace]
        flat_distance[selected_destination] = candidate_distance[replace]
        flat_source_index[selected_destination] = candidate_source_index[replace]
        flat_is_footprint[selected_destination] = candidate_is_footprint[replace]
        flat_valid[selected_destination] = 255

    selected_pixel_count = int(np.count_nonzero(valid_mask))
    if selected_pixel_count == 0:
        raise RuntimeError(f"Frame {frame.frame_id} produced no valid projected surface")
    surface_depth[valid_mask == 0] = 0.0
    surface_depth_valid_mask = valid_mask.copy()
    camera_depth_valid_mask = valid_mask.copy()
    ys, xs = np.nonzero(valid_mask)
    center = (float(np.median(xs) + x0), float(np.median(ys) + y0))
    camera_center = canvas.world_to_canvas(pose[:3, 3])
    raw_valid_count = int(np.count_nonzero(np.isfinite(frame.depth_mm) & (frame.depth_mm > 0)))
    undistorted_valid_count = int(np.count_nonzero(depth_valid))
    zbuffer_candidate_count = projected_sample_count + footprint_candidate_count
    footprint_rasterized_pixel_count = int(
        np.count_nonzero(winner_is_footprint & (valid_mask > 0))
    )
    point_center_selected_pixel_count = (
        selected_pixel_count - footprint_rasterized_pixel_count
    )
    stats: dict[str, int | float | bool] = {
        "input_pixel_count": int(intrinsics.width * intrinsics.height),
        "input_valid_depth_pixel_count": raw_valid_count,
        "undistorted_valid_depth_pixel_count": undistorted_valid_count,
        "projected_sample_count": projected_sample_count,
        "measured_center_candidate_count": projected_sample_count,
        "continuous_surface_sample_count": continuous_surface_sample_count,
        "footprint_candidate_count": footprint_candidate_count,
        "point_center_selected_zbuffer_pixel_count": (
            point_center_selected_pixel_count
        ),
        "footprint_rasterized_pixel_count": footprint_rasterized_pixel_count,
        "selected_zbuffer_pixel_count": selected_pixel_count,
        "zbuffer_candidate_count": zbuffer_candidate_count,
        "zbuffer_collision_count": zbuffer_candidate_count - selected_pixel_count,
        "out_of_canvas_sample_count": out_of_canvas_count,
        "footprint_out_of_canvas_candidate_count": (
            footprint_out_of_canvas_count
        ),
        "depth_discontinuity_edge_count": _depth_discontinuity_count(depth, depth_valid),
        "rejected_invalid_neighbourhood_sample_count": int(
            np.count_nonzero(rejected_invalid_neighbourhood_support)
        ),
        "rejected_depth_edge_sample_count": int(
            np.count_nonzero(rejected_depth_edge_support)
        ),
        "rejected_fold_sample_count": rejected_fold_sample_count,
        "rejected_degenerate_sample_count": rejected_degenerate_sample_count,
        "rejected_overscale_sample_count": rejected_overscale_sample_count,
        "unobserved_output_pixel_count": int(
            valid_mask.size - selected_pixel_count
        ),
        "valid_depth_fraction": float(
            undistorted_valid_count / (intrinsics.width * intrinsics.height)
        ),
        "projected_sampling_ratio": float(
            selected_pixel_count / max(1, undistorted_valid_count)
        ),
        "surface_support_coverage_ratio": float(
            selected_pixel_count
            / max(1, point_center_selected_pixel_count)
        ),
        "point_centres_preserved": True,
        "point_splat_only": False,
        "nearest_measured_rgb_only": True,
        "nearest_measured_depth_only": True,
        "morphological_hole_fill_used": False,
        "surface_footprint_continuity_gate_used": True,
        "surface_footprint_positive_jacobian_required": True,
    }
    return CompactProjectedRGBDSource(
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


def expand_compact_rgbd_source(
    source: CompactProjectedRGBDSource,
    canvas: ProjectionCanvas,
) -> ProjectedRGBDSource:
    """Expand a compact source to the established full-canvas representation."""

    x0, y0, x1, y1 = source.valid_bbox
    expected_shape = (y1 - y0, x1 - x0)
    if (
        x0 < 0
        or y0 < 0
        or x1 > canvas.width
        or y1 > canvas.height
        or x1 <= x0
        or y1 <= y0
        or source.valid_mask.shape != expected_shape
        or source.warped_rgb.shape != (*expected_shape, 3)
    ):
        raise ValueError("Compact RGB-D source does not fit its projection canvas")
    warped_rgb = np.zeros((canvas.height, canvas.width, 3), dtype=np.uint8)
    valid_mask = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
    surface_depth = np.zeros((canvas.height, canvas.width), dtype=np.float32)
    surface_depth_valid_mask = np.zeros(
        (canvas.height, canvas.width), dtype=np.uint8
    )
    camera_depth = np.zeros((canvas.height, canvas.width), dtype=np.float32)
    camera_depth_valid_mask = np.zeros(
        (canvas.height, canvas.width), dtype=np.uint8
    )
    region = np.s_[y0:y1, x0:x1]
    warped_rgb[region] = source.warped_rgb
    valid_mask[region] = source.valid_mask
    surface_depth[region] = source.surface_depth_mm
    surface_depth_valid_mask[region] = source.surface_depth_valid_mask
    camera_depth[region] = source.camera_depth_mm
    camera_depth_valid_mask[region] = source.camera_depth_valid_mask
    return ProjectedRGBDSource(
        frame_id=source.frame_id,
        warped_rgb=warped_rgb,
        valid_mask=valid_mask,
        surface_depth_mm=surface_depth,
        surface_depth_valid_mask=surface_depth_valid_mask,
        camera_depth_mm=camera_depth,
        camera_depth_valid_mask=camera_depth_valid_mask,
        projected_center_xy=source.projected_center_xy,
        valid_bbox=source.valid_bbox,
        projected_height_px=source.projected_height_px,
        sampling_stats=dict(source.sampling_stats),
        camera_center_xy=source.camera_center_xy,
    )


def _project_source(
    frame: RGBDProjectionFrame,
    intrinsics: PinholeIntrinsics,
    canvas: ProjectionCanvas,
    maps: tuple[np.ndarray, np.ndarray] | None,
    *,
    chunk_rows: int,
    maximum_depth_mm: float | None = None,
) -> ProjectedRGBDSource:
    compact = _project_source_compact(
        frame,
        intrinsics,
        canvas,
        maps,
        chunk_rows=chunk_rows,
        maximum_depth_mm=maximum_depth_mm,
    )
    return expand_compact_rgbd_source(compact, canvas)


def project_rgbd_source_compact(
    frame: RGBDProjectionFrame,
    intrinsics: IntrinsicsLike | Mapping[str, object],
    canvas: ProjectionCanvas,
    *,
    chunk_rows: int = 128,
    maximum_depth_mm: float | None = None,
    prepared_undistortion_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> CompactProjectedRGBDSource:
    """Project one source into only its valid global-canvas bounding box."""

    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    camera = coerce_intrinsics(intrinsics)
    selected_maximum_depth = (
        canvas.maximum_depth_mm if maximum_depth_mm is None else maximum_depth_mm
    )
    return _project_source_compact(
        frame,
        camera,
        canvas,
        (
            prepared_undistortion_maps
            if prepared_undistortion_maps is not None
            else _undistortion_maps(camera)
        ),
        chunk_rows=chunk_rows,
        maximum_depth_mm=selected_maximum_depth,
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
    millimetres_per_pixel: float | None = None,
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
        millimetres_per_pixel=millimetres_per_pixel,
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
