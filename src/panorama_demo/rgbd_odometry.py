from __future__ import annotations

import importlib
import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


CAMERA_COORDINATE_CONVENTION = (
    "OpenCV/Open3D color camera coordinates: +x right, +y down, +z forward"
)
POSE_COORDINATE_CONVENTION = (
    "camera_to_world maps camera coordinates into the first pose-node camera frame"
)
POSE_TRANSLATION_UNIT = "mm"


class Open3DUnavailableError(RuntimeError):
    """Open3D could not be imported when the RGB-D backend was requested."""


class PoseGraphError(RuntimeError):
    """The RGB-D graph is structurally unsafe to optimize or publish."""


@dataclass(frozen=True)
class RGBDOdometryConfig:
    """Internal, safety-oriented defaults for adjacent RGB-D odometry.

    Project-facing depth and poses use millimetres. Open3D receives an explicit
    metre conversion, and its transformations are converted back at the adapter
    boundary.
    """

    working_width: int = 640
    require_aligned_depth: bool = True
    require_calibration: bool = True
    minimum_depth_mm: float = 150.0
    maximum_depth_mm: float = 10_000.0
    maximum_depth_difference_mm: float = 70.0
    evaluation_distance_mm: float = 50.0
    minimum_valid_depth_ratio: float = 0.10
    minimum_fitness: float = 0.15
    maximum_inlier_rmse_mm: float = 50.0
    maximum_pair_translation_mm: float = 750.0
    maximum_pair_vertical_mm: float = 80.0
    maximum_pair_forward_mm: float = 120.0
    maximum_pair_rotation_deg: float = 6.0
    iteration_number_per_pyramid_level: tuple[int, ...] = (20, 10, 5)

    def __post_init__(self) -> None:
        if not self.require_aligned_depth or not self.require_calibration:
            raise ValueError(
                "Formal RGB-D odometry cannot disable aligned-depth or "
                "calibration requirements"
            )
        if self.working_width < 64:
            raise ValueError("RGB-D odometry working_width must be at least 64")
        if not 0.0 < self.minimum_depth_mm < self.maximum_depth_mm:
            raise ValueError("RGB-D odometry depth range is invalid")
        if self.maximum_depth_difference_mm <= 0.0:
            raise ValueError("maximum_depth_difference_mm must be positive")
        if self.evaluation_distance_mm <= 0.0:
            raise ValueError("evaluation_distance_mm must be positive")
        if not 0.0 <= self.minimum_valid_depth_ratio <= 1.0:
            raise ValueError("minimum_valid_depth_ratio must be between zero and one")
        if not 0.0 <= self.minimum_fitness <= 1.0:
            raise ValueError("minimum_fitness must be between zero and one")
        positive_values = (
            self.maximum_inlier_rmse_mm,
            self.maximum_pair_translation_mm,
            self.maximum_pair_vertical_mm,
            self.maximum_pair_forward_mm,
            self.maximum_pair_rotation_deg,
        )
        if any(value <= 0.0 for value in positive_values):
            raise ValueError("RGB-D odometry safety limits must be positive")
        if not self.iteration_number_per_pyramid_level or any(
            value < 1 for value in self.iteration_number_per_pyramid_level
        ):
            raise ValueError("RGB-D odometry pyramid iterations must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RGBDOdometryConfig:
        return cls() if value is None else cls(**dict(value))


@dataclass(frozen=True)
class PoseGraphConfig:
    enabled: bool = True
    maximum_correspondence_distance_mm: float = 50.0
    edge_prune_threshold: float = 0.25
    preference_loop_closure: float = 0.10

    def __post_init__(self) -> None:
        if not self.enabled:
            raise ValueError("The formal RGB-D pose graph cannot be disabled")
        if self.maximum_correspondence_distance_mm <= 0.0:
            raise ValueError("Pose-graph correspondence distance must be positive")
        if not 0.0 <= self.edge_prune_threshold <= 1.0:
            raise ValueError("Pose-graph edge_prune_threshold must be in [0, 1]")
        if not 0.0 <= self.preference_loop_closure <= 1.0:
            raise ValueError("Pose-graph preference_loop_closure must be in [0, 1]")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PoseGraphConfig:
        return cls() if value is None else cls(**dict(value))


@dataclass(frozen=True)
class PoseQualityThresholds:
    """Fail-closed trajectory gates for the fixed horizontal side-scan rig."""

    minimum_scan_span_mm: float = 20.0
    maximum_reverse_step_mm: float = 5.0
    maximum_reverse_fraction: float = 0.02
    maximum_step_translation_mm: float = 750.0
    maximum_step_vertical_mm: float = 80.0
    maximum_step_forward_mm: float = 120.0
    maximum_total_vertical_drift_mm: float = 120.0
    maximum_total_forward_drift_mm: float = 150.0
    maximum_step_rotation_deg: float = 6.0
    maximum_total_rotation_deg: float = 10.0
    maximum_edge_translation_residual_mm: float = 30.0
    maximum_edge_rotation_residual_deg: float = 2.0
    maximum_consecutive_unreliable_edges: int = 2
    require_all_adjacent_edges: bool = True

    def __post_init__(self) -> None:
        numeric = (
            self.minimum_scan_span_mm,
            self.maximum_reverse_step_mm,
            self.maximum_step_translation_mm,
            self.maximum_step_vertical_mm,
            self.maximum_step_forward_mm,
            self.maximum_total_vertical_drift_mm,
            self.maximum_total_forward_drift_mm,
            self.maximum_step_rotation_deg,
            self.maximum_total_rotation_deg,
            self.maximum_edge_translation_residual_mm,
            self.maximum_edge_rotation_residual_deg,
        )
        if any(value < 0.0 for value in numeric):
            raise ValueError("Pose trajectory thresholds cannot be negative")
        if not 0.0 <= self.maximum_reverse_fraction <= 1.0:
            raise ValueError("maximum_reverse_fraction must be in [0, 1]")
        if self.maximum_consecutive_unreliable_edges < 0:
            raise ValueError("maximum_consecutive_unreliable_edges cannot be negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PoseQualityThresholds:
        return cls() if value is None else cls(**dict(value))


@dataclass(frozen=True)
class PreparedRGBDFrame:
    """An undistorted RGB-D input at odometry resolution.

    This type is intentionally small and backend-neutral so tests can inject a
    deterministic backend without importing Open3D.
    """

    frame_id: int
    color_rgb: np.ndarray
    depth_mm: np.ndarray
    valid_depth_mask: np.ndarray
    valid_depth_ratio: float


@dataclass(frozen=True)
class _Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()


@dataclass(frozen=True)
class PoseEdge:
    """A measured rigid edge mapping source-camera points to reference-camera points."""

    reference_node_id: int
    source_node_id: int
    source_to_reference: np.ndarray
    converged: bool
    fitness: float
    rmse_mm: float
    information: np.ndarray
    reference_valid_depth_ratio: float
    source_valid_depth_ratio: float
    uncertain: bool = False
    backend: str = "open3d_rgbd"
    failure_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_to_reference",
            np.asarray(self.source_to_reference, dtype=np.float64).copy(),
        )
        object.__setattr__(
            self,
            "information",
            np.asarray(self.information, dtype=np.float64).copy(),
        )
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        if self.reference_node_id == self.source_node_id:
            raise ValueError("A pose edge cannot connect a node to itself")

    @property
    def valid_depth_ratio(self) -> float:
        return min(
            float(self.reference_valid_depth_ratio),
            float(self.source_valid_depth_ratio),
        )

    @property
    def rmse_m(self) -> float:
        """Open3D-unit view of the project-facing millimetre RMSE."""

        return float(self.rmse_mm) / 1000.0

    @property
    def transform_is_valid(self) -> bool:
        return _is_valid_se3(self.source_to_reference)

    @property
    def information_is_valid(self) -> bool:
        return _is_valid_information(self.information)

    @property
    def structurally_valid(self) -> bool:
        return bool(
            self.converged
            and self.transform_is_valid
            and self.information_is_valid
            and math.isfinite(float(self.fitness))
            and math.isfinite(float(self.rmse_mm))
            and math.isfinite(float(self.reference_valid_depth_ratio))
            and math.isfinite(float(self.source_valid_depth_ratio))
        )

    @property
    def reliable(self) -> bool:
        return self.structurally_valid and not self.failure_reasons

    def as_dict(self) -> dict[str, Any]:
        translation_mm = (
            self.source_to_reference[:3, 3]
            if self.source_to_reference.shape == (4, 4)
            else np.full(3, np.nan)
        )
        rotation_deg = (
            _rotation_angle_deg(self.source_to_reference[:3, :3])
            if self.source_to_reference.shape == (4, 4)
            else math.nan
        )
        return {
            "reference_node_id": self.reference_node_id,
            "source_node_id": self.source_node_id,
            "transform_convention": "source_to_reference",
            "translation_unit": POSE_TRANSLATION_UNIT,
            "source_to_reference": self.source_to_reference.tolist(),
            "converged": bool(self.converged),
            "fitness": _finite_or_none(self.fitness),
            "rmse_mm": _finite_or_none(self.rmse_mm),
            "rmse_m": _finite_or_none(float(self.rmse_mm) / 1000.0),
            "information": self.information.tolist(),
            "reference_valid_depth_ratio": _finite_or_none(
                self.reference_valid_depth_ratio
            ),
            "source_valid_depth_ratio": _finite_or_none(
                self.source_valid_depth_ratio
            ),
            "translation_mm": [
                _finite_or_none(float(value)) for value in translation_mm
            ],
            "rotation_deg": _finite_or_none(rotation_deg),
            "finite_se3": self.transform_is_valid,
            "valid_information": self.information_is_valid,
            "uncertain": bool(self.uncertain),
            "backend": self.backend,
            "quality_pass": self.reliable,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True)
class PoseGraphResult:
    node_ids: tuple[int, ...]
    camera_to_world: tuple[np.ndarray, ...]
    edges: tuple[PoseEdge, ...]
    optimized: bool
    connected: bool
    reference_node_id: int
    backend: str = "open3d_rgbd"
    edge_residuals: tuple[dict[str, float | int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_ids", tuple(int(value) for value in self.node_ids))
        object.__setattr__(
            self,
            "camera_to_world",
            tuple(np.asarray(pose, dtype=np.float64).copy() for pose in self.camera_to_world),
        )
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "edge_residuals", tuple(self.edge_residuals))
        if len(self.node_ids) != len(self.camera_to_world):
            raise ValueError("Pose-graph node ids and camera poses must have equal length")
        if self.reference_node_id not in self.node_ids:
            raise ValueError("Pose-graph reference node is not present")

    @property
    def poses(self) -> tuple[np.ndarray, ...]:
        """Compatibility alias with an explicit camera_to_world definition."""

        return self.camera_to_world

    def pose_for(self, node_id: int) -> np.ndarray:
        try:
            index = self.node_ids.index(int(node_id))
        except ValueError as exc:
            raise KeyError(node_id) from exc
        return self.camera_to_world[index].copy()

    def as_dict(self) -> dict[str, Any]:
        residuals = [dict(value) for value in self.edge_residuals]
        return {
            "schema": "rgbd-pose-graph/v1",
            "camera_coordinates": CAMERA_COORDINATE_CONVENTION,
            "pose_convention": POSE_COORDINATE_CONVENTION,
            "translation_unit": POSE_TRANSLATION_UNIT,
            "optimized": bool(self.optimized),
            "connected": bool(self.connected),
            "backend": self.backend,
            "reference_node_id": self.reference_node_id,
            "nodes": [
                {
                    "node_id": node_id,
                    "camera_to_world": pose.tolist(),
                }
                for node_id, pose in zip(
                    self.node_ids, self.camera_to_world, strict=True
                )
            ],
            "edges": [edge.as_dict() for edge in self.edges],
            "edge_residuals": residuals,
        }


@dataclass(frozen=True)
class PoseQualityReport:
    quality_pass: bool
    failure_reasons: tuple[str, ...]
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: PoseQualityThresholds = field(default_factory=PoseQualityThresholds)

    def as_dict(self) -> dict[str, Any]:
        return {
            "quality_pass": bool(self.quality_pass),
            "failure_reasons": list(self.failure_reasons),
            "metrics": _json_safe(self.metrics),
            "thresholds": asdict(self.thresholds),
        }


class RGBDOdometryBackend(Protocol):
    def estimate_pair(
        self,
        *,
        reference: PreparedRGBDFrame,
        source: PreparedRGBDFrame,
        intrinsics: _Intrinsics,
        config: RGBDOdometryConfig,
        initial_source_to_reference: np.ndarray | None = None,
    ) -> Mapping[str, Any] | PoseEdge: ...

    def optimize_pose_graph(
        self,
        *,
        node_ids: tuple[int, ...],
        initial_camera_to_world: tuple[np.ndarray, ...],
        edges: tuple[PoseEdge, ...],
        config: PoseGraphConfig,
    ) -> Sequence[np.ndarray] | Mapping[int, np.ndarray]: ...


def _finite_or_none(value: float | int) -> float | None:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return _finite_or_none(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _is_valid_information(information: np.ndarray) -> bool:
    matrix = np.asarray(information, dtype=np.float64)
    if (
        matrix.shape != (6, 6)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-6, rtol=1e-6)
    ):
        return False
    eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    maximum = float(eigenvalues[-1])
    if not math.isfinite(maximum) or maximum <= 0.0:
        return False
    # Pose-graph information must be positive definite in all six SE(3)
    # directions.  A small relative floor tolerates floating-point roundoff
    # but rejects indefinite, singular, or numerically unusable measurements.
    minimum = float(eigenvalues[0])
    return bool(minimum > max(1e-12, maximum * 1e-12))


def _is_valid_se3(matrix: np.ndarray, *, tolerance: float = 1e-4) -> bool:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return False
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=tolerance):
        return False
    rotation = value[:3, :3]
    return bool(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance, rtol=tolerance)
        and math.isclose(
            float(np.linalg.det(rotation)), 1.0, abs_tol=tolerance, rel_tol=tolerance
        )
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.isfinite(value).all():
        return math.nan
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _pose_mm_to_m(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).copy()
    value[:3, 3] /= 1000.0
    return value


def _pose_m_to_mm(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64).copy()
    value[:3, 3] *= 1000.0
    return value


def _lookup(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _node_id(value: Any, fallback: int) -> int:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    for name in ("node_id", "frame_id"):
        found = _lookup(value, name)
        if found is not None:
            return int(found)
    return fallback


def _coerce_intrinsics(value: Any) -> _Intrinsics:
    required: dict[str, float | int] = {}
    for name in ("width", "height", "fx", "fy", "cx", "cy"):
        found = _lookup(value, name)
        if found is None:
            raise ValueError(f"RGB-D odometry intrinsics are missing {name}")
        required[name] = found
    distortion_value = _lookup(value, "distortion", ())
    if distortion_value is None:
        distortion_value = ()
    distortion = tuple(float(item) for item in distortion_value)
    intrinsics = _Intrinsics(
        width=int(required["width"]),
        height=int(required["height"]),
        fx=float(required["fx"]),
        fy=float(required["fy"]),
        cx=float(required["cx"]),
        cy=float(required["cy"]),
        distortion=distortion,
    )
    numeric = np.asarray(
        [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, *distortion],
        dtype=np.float64,
    )
    if (
        intrinsics.width < 1
        or intrinsics.height < 1
        or intrinsics.fx <= 0.0
        or intrinsics.fy <= 0.0
        or not np.isfinite(numeric).all()
    ):
        raise ValueError("RGB-D odometry intrinsics are invalid")
    return intrinsics


def _load_frame_arrays(frame: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    color_rgb_value = _lookup(frame, "color_rgb")
    color_bgr_value = _lookup(frame, "color_bgr")
    depth_mm_value = _lookup(frame, "aligned_depth_mm")
    geometry_mask_value = _lookup(frame, "valid_mask")
    if depth_mm_value is not None and (
        color_rgb_value is not None or color_bgr_value is not None
    ):
        if color_rgb_value is not None:
            color_rgb = np.asarray(color_rgb_value)
        else:
            import cv2

            color_rgb = cv2.cvtColor(np.asarray(color_bgr_value), cv2.COLOR_BGR2RGB)
        depth_mm = np.asarray(depth_mm_value, dtype=np.float32)
        geometry_mask = (
            None
            if geometry_mask_value is None
            else np.asarray(geometry_mask_value, dtype=bool)
        )
        return color_rgb, depth_mm, geometry_mask

    color_path_value = _lookup(frame, "color_path")
    aligned_depth_path_value = _lookup(frame, "aligned_depth_path")
    if color_path_value is None:
        raise ValueError("RGB-D odometry frame is missing color data")
    if aligned_depth_path_value is None:
        raise ValueError(
            "RGB-D odometry requires aligned_depth_path; raw depth is not accepted"
        )
    depth_scale = _lookup(frame, "depth_scale_mm_per_unit")
    if depth_scale is None or not math.isfinite(float(depth_scale)) or float(depth_scale) <= 0:
        raise ValueError("RGB-D odometry requires a positive depth_scale_mm_per_unit")
    import cv2

    color_path = Path(color_path_value)
    aligned_depth_path = Path(aligned_depth_path_value)
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"Could not read RGB-D color image: {color_path}")
    depth_raw = cv2.imread(str(aligned_depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(
            f"Could not read aligned RGB-D depth image: {aligned_depth_path}"
        )
    if depth_raw.ndim != 2:
        raise ValueError(f"Aligned depth image must be single-channel: {aligned_depth_path}")
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    depth_mm = depth_raw.astype(np.float32) * float(depth_scale)
    return color_rgb, depth_mm, None


def _prepare_frame(
    frame: Any,
    intrinsics: _Intrinsics,
    config: RGBDOdometryConfig,
    *,
    fallback_id: int,
) -> tuple[PreparedRGBDFrame, _Intrinsics]:
    color_rgb, depth_mm, geometry_mask = _load_frame_arrays(frame)
    if color_rgb.dtype != np.uint8 or color_rgb.ndim != 3 or color_rgb.shape[2] != 3:
        raise ValueError("RGB-D odometry color input must be an RGB uint8 image")
    if depth_mm.ndim != 2 or depth_mm.shape != color_rgb.shape[:2]:
        raise ValueError("Aligned depth dimensions must match the RGB image")
    if (color_rgb.shape[1], color_rgb.shape[0]) != (
        intrinsics.width,
        intrinsics.height,
    ):
        raise ValueError("RGB-D image dimensions do not match calibration intrinsics")
    if geometry_mask is None:
        geometry_mask = np.ones(depth_mm.shape, dtype=bool)
    elif geometry_mask.shape != depth_mm.shape:
        raise ValueError("RGB-D valid_mask dimensions must match the aligned depth")

    import cv2

    working_intrinsics = intrinsics
    if intrinsics.distortion and np.any(np.asarray(intrinsics.distortion) != 0.0):
        camera_matrix = np.asarray(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            np.asarray(intrinsics.distortion, dtype=np.float64),
            None,
            camera_matrix,
            (intrinsics.width, intrinsics.height),
            cv2.CV_32FC1,
        )
        color_rgb = cv2.remap(
            color_rgb,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        depth_mm = cv2.remap(
            depth_mm,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        geometry_mask = cv2.remap(
            geometry_mask.astype(np.uint8),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
        working_intrinsics = _Intrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
            distortion=(),
        )

    target_width = min(config.working_width, working_intrinsics.width)
    if target_width != working_intrinsics.width:
        scale = target_width / float(working_intrinsics.width)
        target_height = max(1, int(round(working_intrinsics.height * scale)))
        color_rgb = cv2.resize(
            color_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        depth_mm = cv2.resize(
            depth_mm, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
        geometry_mask = cv2.resize(
            geometry_mask.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        scale_x = target_width / float(working_intrinsics.width)
        scale_y = target_height / float(working_intrinsics.height)
        working_intrinsics = _Intrinsics(
            width=target_width,
            height=target_height,
            fx=working_intrinsics.fx * scale_x,
            fy=working_intrinsics.fy * scale_y,
            cx=working_intrinsics.cx * scale_x,
            cy=working_intrinsics.cy * scale_y,
            distortion=(),
        )

    finite_depth = np.isfinite(depth_mm)
    valid_depth = (
        geometry_mask
        & finite_depth
        & (depth_mm >= config.minimum_depth_mm)
        & (depth_mm <= config.maximum_depth_mm)
    )
    safe_depth = np.where(valid_depth, depth_mm, 0.0).astype(np.float32)
    prepared = PreparedRGBDFrame(
        frame_id=_node_id(frame, fallback_id),
        color_rgb=np.ascontiguousarray(color_rgb),
        depth_mm=np.ascontiguousarray(safe_depth),
        valid_depth_mask=np.ascontiguousarray(valid_depth),
        valid_depth_ratio=float(np.mean(valid_depth)),
    )
    return prepared, working_intrinsics


def _import_open3d() -> Any:
    try:
        return importlib.import_module("open3d")
    except (ImportError, OSError) as exc:
        raise Open3DUnavailableError(
            "Open3D is required for the open3d_rgbd pose backend but could not "
            "be loaded. Install the project Open3D dependency in the active "
            "environment; no 2-D or feature-matching fallback is permitted."
        ) from exc


def _set_open3d_option(
    option: Any,
    canonical_name: str,
    value: float | Sequence[int],
    *,
    legacy_names: tuple[str, ...] = (),
) -> str:
    selected_name = next(
        (
            name
            for name in (canonical_name, *legacy_names)
            if hasattr(option, name)
        ),
        None,
    )
    if selected_name is None:
        accepted = ", ".join((canonical_name, *legacy_names))
        raise RuntimeError(
            "Open3D OdometryOption does not expose the required property "
            f"({accepted}); install the supported Open3D version"
        )
    sequence_value = isinstance(value, Sequence) and not isinstance(
        value, (str, bytes)
    )
    try:
        setattr(option, selected_name, value)
        applied = getattr(option, selected_name)
    except TypeError as exc:
        # Open3D 0.19 exposes the iteration schedule as utility.IntVector and
        # rejects an otherwise equivalent Python list at the pybind setter.
        if not sequence_value:
            raise RuntimeError(
                f"Open3D OdometryOption could not apply {selected_name}"
            ) from exc
        try:
            converted = _import_open3d().utility.IntVector(list(value))
            setattr(option, selected_name, converted)
            applied = getattr(option, selected_name)
        except (AttributeError, TypeError, ValueError) as converted_exc:
            raise RuntimeError(
                f"Open3D OdometryOption could not apply {selected_name}"
            ) from converted_exc
    except (AttributeError, ValueError) as exc:
        raise RuntimeError(
            f"Open3D OdometryOption could not apply {selected_name}"
        ) from exc
    if sequence_value:
        if list(applied) != list(value):
            raise RuntimeError(
                f"Open3D OdometryOption did not retain {selected_name}"
            )
    elif not math.isclose(float(applied), float(value), rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"Open3D OdometryOption did not retain {selected_name}")
    return selected_name


def _configure_open3d_odometry_option(
    option: Any, config: RGBDOdometryConfig
) -> dict[str, str]:
    """Apply and verify the Open3D 0.19 option names.

    Known legacy aliases are accepted explicitly for older supported builds.
    Missing attributes or failed readback are fatal rather than silently using
    Open3D defaults.
    """

    return {
        "depth_min": _set_open3d_option(
            option,
            "depth_min",
            config.minimum_depth_mm / 1000.0,
            legacy_names=("min_depth",),
        ),
        "depth_max": _set_open3d_option(
            option,
            "depth_max",
            config.maximum_depth_mm / 1000.0,
            legacy_names=("max_depth",),
        ),
        "depth_diff_max": _set_open3d_option(
            option,
            "depth_diff_max",
            config.maximum_depth_difference_mm / 1000.0,
            legacy_names=("max_depth_diff",),
        ),
        "iteration_number_per_pyramid_level": _set_open3d_option(
            option,
            "iteration_number_per_pyramid_level",
            list(config.iteration_number_per_pyramid_level),
        ),
    }


class _Open3DBackend:
    # Open3D's hybrid RGB-D odometry accepts a full SE(3) initial guess.  A
    # sequence coordinator can advertise this capability without relying on
    # the implementation type (which is deliberately private).
    supports_initial_source_to_reference = True

    def __init__(self) -> None:
        self.o3d = _import_open3d()

    def _intrinsic(self, intrinsics: _Intrinsics) -> Any:
        return self.o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
        )

    def _rgbd(self, frame: PreparedRGBDFrame, config: RGBDOdometryConfig) -> Any:
        color = self.o3d.geometry.Image(frame.color_rgb)
        depth = self.o3d.geometry.Image(frame.depth_mm.astype(np.float32, copy=False))
        return self.o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000.0,
            depth_trunc=config.maximum_depth_mm / 1000.0,
            convert_rgb_to_intensity=True,
        )

    def estimate_pair(
        self,
        *,
        reference: PreparedRGBDFrame,
        source: PreparedRGBDFrame,
        intrinsics: _Intrinsics,
        config: RGBDOdometryConfig,
        initial_source_to_reference: np.ndarray | None = None,
    ) -> Mapping[str, Any]:
        reference_rgbd = self._rgbd(reference, config)
        source_rgbd = self._rgbd(source, config)
        intrinsic = self._intrinsic(intrinsics)
        option = self.o3d.pipelines.odometry.OdometryOption()
        _configure_open3d_odometry_option(option, config)
        initial_m = (
            np.eye(4, dtype=np.float64)
            if initial_source_to_reference is None
            else _pose_mm_to_m(initial_source_to_reference)
        )
        converged, transformation, information = (
            self.o3d.pipelines.odometry.compute_rgbd_odometry(
                source_rgbd,
                reference_rgbd,
                intrinsic,
                initial_m,
                self.o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
                option,
            )
        )

        # The legacy odometry API has no fitness/RMSE outputs. Evaluate the
        # measured transform on actual source/reference point clouds instead of
        # fabricating either metric from the 6x6 information matrix.
        source_cloud = self.o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd, intrinsic
        )
        reference_cloud = self.o3d.geometry.PointCloud.create_from_rgbd_image(
            reference_rgbd, intrinsic
        )
        evaluation = self.o3d.pipelines.registration.evaluate_registration(
            source_cloud,
            reference_cloud,
            config.evaluation_distance_mm / 1000.0,
            np.asarray(transformation, dtype=np.float64),
        )
        return {
            "converged": bool(converged),
            "source_to_reference": _pose_m_to_mm(transformation),
            "information": information,
            "fitness": float(evaluation.fitness),
            "rmse_mm": float(evaluation.inlier_rmse) * 1000.0,
            "backend": "open3d_rgbd",
        }

    def optimize_pose_graph(
        self,
        *,
        node_ids: tuple[int, ...],
        initial_camera_to_world: tuple[np.ndarray, ...],
        edges: tuple[PoseEdge, ...],
        config: PoseGraphConfig,
    ) -> Sequence[np.ndarray]:
        graph = self.o3d.pipelines.registration.PoseGraph()
        for pose in initial_camera_to_world:
            graph.nodes.append(
                self.o3d.pipelines.registration.PoseGraphNode(
                    _pose_mm_to_m(pose)
                )
            )
        index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
        for edge in edges:
            graph.edges.append(
                self.o3d.pipelines.registration.PoseGraphEdge(
                    index_by_id[edge.source_node_id],
                    index_by_id[edge.reference_node_id],
                    _pose_mm_to_m(edge.source_to_reference),
                    edge.information.copy(),
                    bool(edge.uncertain),
                )
            )
        option = self.o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=(
                config.maximum_correspondence_distance_mm / 1000.0
            ),
            edge_prune_threshold=config.edge_prune_threshold,
            preference_loop_closure=config.preference_loop_closure,
            reference_node=0,
        )
        self.o3d.pipelines.registration.global_optimization(
            graph,
            self.o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            self.o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option,
        )
        return [_pose_m_to_mm(node.pose) for node in graph.nodes]


def _measurement_value(measurement: Any, name: str, default: Any = None) -> Any:
    if isinstance(measurement, Mapping):
        return measurement.get(name, default)
    return getattr(measurement, name, default)


def _edge_quality_reasons(
    *,
    converged: bool,
    transformation: np.ndarray,
    information: np.ndarray,
    fitness: float,
    rmse_mm: float,
    reference_valid_depth_ratio: float,
    source_valid_depth_ratio: float,
    config: RGBDOdometryConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not converged:
        reasons.append("RGB-D odometry did not converge")
    if not _is_valid_se3(transformation):
        reasons.append("RGB-D odometry did not return a finite rigid SE(3)")
    if not _is_valid_information(information):
        reasons.append("RGB-D odometry information matrix is invalid")
    if not math.isfinite(fitness) or fitness < config.minimum_fitness:
        reasons.append("RGB-D odometry fitness is below the safety threshold")
    if not math.isfinite(rmse_mm) or rmse_mm > config.maximum_inlier_rmse_mm:
        reasons.append("RGB-D odometry RMSE exceeds the safety threshold")
    if min(reference_valid_depth_ratio, source_valid_depth_ratio) < (
        config.minimum_valid_depth_ratio
    ):
        reasons.append("RGB-D valid depth coverage is below the safety threshold")
    if _is_valid_se3(transformation):
        translation_mm = transformation[:3, 3]
        if float(np.linalg.norm(translation_mm)) > config.maximum_pair_translation_mm:
            reasons.append("RGB-D odometry translation exceeds the adjacent limit")
        if abs(float(translation_mm[1])) > config.maximum_pair_vertical_mm:
            reasons.append("RGB-D odometry vertical motion exceeds the adjacent limit")
        if abs(float(translation_mm[2])) > config.maximum_pair_forward_mm:
            reasons.append("RGB-D odometry forward motion exceeds the adjacent limit")
        if (
            _rotation_angle_deg(transformation[:3, :3])
            > config.maximum_pair_rotation_deg
        ):
            reasons.append("RGB-D odometry rotation exceeds the adjacent limit")
    return tuple(reasons)


def estimate_pair_rgbd_odometry(
    reference: Any,
    source: Any,
    intrinsics: Any,
    *,
    config: RGBDOdometryConfig | Mapping[str, Any] | None = None,
    backend: RGBDOdometryBackend | Any | None = None,
    reference_node_id: int | None = None,
    source_node_id: int | None = None,
    uncertain: bool = False,
    initial_source_to_reference: Any | None = None,
) -> PoseEdge:
    """Estimate one source-to-reference RGB-D edge.

    ``source_to_reference`` follows ``p_reference = T @ p_source``. Input depth,
    returned translation, returned RMSE, and an optional rigid initial guess
    are millimetres; Open3D receives metres explicitly. No feature or 2-D
    fallback is attempted when Open3D is unavailable or odometry fails.
    """

    odometry_config = (
        config
        if isinstance(config, RGBDOdometryConfig)
        else RGBDOdometryConfig.from_mapping(config)
    )
    calibrated = _coerce_intrinsics(intrinsics)
    prepared_reference, working_intrinsics = _prepare_frame(
        reference, calibrated, odometry_config, fallback_id=0
    )
    prepared_source, source_intrinsics = _prepare_frame(
        source, calibrated, odometry_config, fallback_id=1
    )
    if source_intrinsics != working_intrinsics:
        raise ValueError("Reference and source RGB-D odometry intrinsics differ")
    initial_transform: np.ndarray | None = None
    if initial_source_to_reference is not None:
        initial_transform = np.asarray(
            initial_source_to_reference, dtype=np.float64
        )
        if not _is_valid_se3(initial_transform):
            raise ValueError(
                "RGB-D odometry initial_source_to_reference must be a finite "
                "rigid SE(3) in millimetres"
            )
    reference_id = (
        prepared_reference.frame_id
        if reference_node_id is None
        else int(reference_node_id)
    )
    source_id = (
        prepared_source.frame_id if source_node_id is None else int(source_node_id)
    )
    selected_backend = _Open3DBackend() if backend is None else backend
    estimator = getattr(selected_backend, "estimate_pair", selected_backend)
    estimator_kwargs: dict[str, Any] = {
        "reference": prepared_reference,
        "source": prepared_source,
        "intrinsics": working_intrinsics,
        "config": odometry_config,
    }
    if initial_transform is not None:
        estimator_kwargs["initial_source_to_reference"] = initial_transform
    measurement = estimator(**estimator_kwargs)
    if isinstance(measurement, PoseEdge):
        transform = measurement.source_to_reference
        converged = bool(measurement.converged)
        information = measurement.information
        fitness = float(measurement.fitness)
        rmse_mm = float(measurement.rmse_mm)
        backend_name = measurement.backend
        backend_reasons = measurement.failure_reasons
    else:
        transform = np.asarray(
            _measurement_value(measurement, "source_to_reference"),
            dtype=np.float64,
        )
        converged = bool(_measurement_value(measurement, "converged", False))
        information = np.asarray(
            _measurement_value(measurement, "information"), dtype=np.float64
        )
        fitness = float(_measurement_value(measurement, "fitness", math.nan))
        rmse_value = _measurement_value(measurement, "rmse_mm")
        if rmse_value is None:
            legacy_rmse_m = _measurement_value(measurement, "rmse_m")
            rmse_mm = (
                math.nan if legacy_rmse_m is None else float(legacy_rmse_m) * 1000.0
            )
        else:
            rmse_mm = float(rmse_value)
        backend_name = str(
            _measurement_value(
                measurement,
                "backend",
                getattr(selected_backend, "name", "injected_rgbd"),
            )
        )
        backend_reasons = tuple(
            _measurement_value(measurement, "failure_reasons", ())
        )
    quality_reasons = _edge_quality_reasons(
        converged=converged,
        transformation=transform,
        information=information,
        fitness=fitness,
        rmse_mm=rmse_mm,
        reference_valid_depth_ratio=prepared_reference.valid_depth_ratio,
        source_valid_depth_ratio=prepared_source.valid_depth_ratio,
        config=odometry_config,
    )
    reasons = tuple(dict.fromkeys((*backend_reasons, *quality_reasons)))
    return PoseEdge(
        reference_node_id=reference_id,
        source_node_id=source_id,
        source_to_reference=transform,
        converged=converged,
        fitness=fitness,
        rmse_mm=rmse_mm,
        information=information,
        reference_valid_depth_ratio=prepared_reference.valid_depth_ratio,
        source_valid_depth_ratio=prepared_source.valid_depth_ratio,
        uncertain=uncertain,
        backend=backend_name,
        failure_reasons=reasons,
    )


def _edge_between(edge: PoseEdge, left: int, right: int) -> bool:
    return {
        edge.reference_node_id,
        edge.source_node_id,
    } == {left, right}


def _graph_connected(node_ids: tuple[int, ...], edges: Sequence[PoseEdge]) -> bool:
    if not node_ids:
        return False
    neighbours: dict[int, set[int]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        if edge.reference_node_id in neighbours and edge.source_node_id in neighbours:
            neighbours[edge.reference_node_id].add(edge.source_node_id)
            neighbours[edge.source_node_id].add(edge.reference_node_id)
    visited = {node_ids[0]}
    pending = [node_ids[0]]
    while pending:
        current = pending.pop()
        for neighbour in neighbours[current] - visited:
            visited.add(neighbour)
            pending.append(neighbour)
    return len(visited) == len(node_ids)


def _propagate_initial_poses(
    node_ids: tuple[int, ...], edges: Sequence[PoseEdge]
) -> tuple[np.ndarray, ...]:
    by_node: dict[int, list[PoseEdge]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        by_node[edge.reference_node_id].append(edge)
        by_node[edge.source_node_id].append(edge)
    poses: dict[int, np.ndarray] = {node_ids[0]: np.eye(4, dtype=np.float64)}
    pending: deque[int] = deque([node_ids[0]])
    while pending:
        known_id = pending.popleft()
        known_pose = poses[known_id]
        for edge in by_node[known_id]:
            if known_id == edge.reference_node_id:
                unknown_id = edge.source_node_id
                unknown_pose = known_pose @ edge.source_to_reference
            else:
                unknown_id = edge.reference_node_id
                unknown_pose = known_pose @ np.linalg.inv(edge.source_to_reference)
            if unknown_id not in poses:
                poses[unknown_id] = unknown_pose
                pending.append(unknown_id)
    if len(poses) != len(node_ids):
        raise PoseGraphError("RGB-D pose graph is disconnected")
    return tuple(poses[node_id] for node_id in node_ids)


def _normalise_optimized_poses(
    value: Any,
    node_ids: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    if isinstance(value, PoseGraphResult):
        poses = tuple(value.pose_for(node_id) for node_id in node_ids)
    elif isinstance(value, Mapping):
        try:
            poses = tuple(np.asarray(value[node_id], dtype=np.float64) for node_id in node_ids)
        except KeyError as exc:
            raise PoseGraphError(
                f"Pose backend omitted optimized node {exc.args[0]}"
            ) from exc
    else:
        poses = tuple(np.asarray(pose, dtype=np.float64) for pose in value)
        if len(poses) != len(node_ids):
            raise PoseGraphError("Pose backend returned the wrong number of node poses")
    for node_id, pose in zip(node_ids, poses, strict=True):
        if not _is_valid_se3(pose):
            raise PoseGraphError(
                f"Pose backend returned a non-finite or non-rigid SE(3) for node {node_id}"
            )
    anchor_inverse = np.linalg.inv(poses[0])
    anchored = tuple(anchor_inverse @ pose for pose in poses)
    for node_id, pose in zip(node_ids, anchored, strict=True):
        if not _is_valid_se3(pose):
            raise PoseGraphError(f"Anchored camera_to_world pose is invalid for node {node_id}")
    return anchored


def _edge_residuals(
    node_ids: tuple[int, ...],
    poses: tuple[np.ndarray, ...],
    edges: Sequence[PoseEdge],
) -> tuple[dict[str, float | int], ...]:
    pose_by_id = dict(zip(node_ids, poses, strict=True))
    residuals: list[dict[str, float | int]] = []
    for edge in edges:
        if not edge.transform_is_valid:
            continue
        predicted = (
            np.linalg.inv(pose_by_id[edge.reference_node_id])
            @ pose_by_id[edge.source_node_id]
        )
        delta = np.linalg.inv(edge.source_to_reference) @ predicted
        residuals.append(
            {
                "reference_node_id": edge.reference_node_id,
                "source_node_id": edge.source_node_id,
                "translation_residual_mm": float(
                    np.linalg.norm(delta[:3, 3])
                ),
                "rotation_residual_deg": _rotation_angle_deg(delta[:3, :3]),
            }
        )
    return tuple(residuals)


def optimize_rgbd_pose_graph(
    nodes: Sequence[Any],
    edges: Sequence[PoseEdge],
    *,
    config: PoseGraphConfig | Mapping[str, Any] | None = None,
    backend: RGBDOdometryBackend | Any | None = None,
    enforce_edge_quality: bool = True,
) -> PoseGraphResult:
    """Optimize a connected graph made exclusively from RGB-D odometry edges.

    In diagnostic use, ``enforce_edge_quality=False`` may retain low-fitness or
    high-RMSE edges, but convergence, finite SE(3), valid information, required
    adjacency, and graph connectivity remain mandatory.
    """

    graph_config = (
        config if isinstance(config, PoseGraphConfig) else PoseGraphConfig.from_mapping(config)
    )
    node_ids = tuple(_node_id(node, index) for index, node in enumerate(nodes))
    if len(node_ids) < 2:
        raise PoseGraphError("RGB-D pose graph requires at least two nodes")
    if len(set(node_ids)) != len(node_ids):
        raise PoseGraphError("RGB-D pose-graph node ids must be unique")
    node_set = set(node_ids)
    edge_rows = tuple(edges)
    if any(
        edge.reference_node_id not in node_set or edge.source_node_id not in node_set
        for edge in edge_rows
    ):
        raise PoseGraphError("RGB-D pose edge references an unknown node")
    structural_edges = tuple(edge for edge in edge_rows if edge.structurally_valid)
    usable_edges = tuple(
        edge
        for edge in structural_edges
        if edge.reliable or not enforce_edge_quality
    )
    if not _graph_connected(node_ids, structural_edges):
        raise PoseGraphError("RGB-D pose graph is disconnected")
    for left, right in zip(node_ids, node_ids[1:]):
        if not any(_edge_between(edge, left, right) for edge in usable_edges):
            qualifier = "reliable " if enforce_edge_quality else "structurally valid "
            raise PoseGraphError(
                f"Required adjacent RGB-D edge {left}<->{right} has no "
                f"{qualifier}measurement"
            )
    if not _graph_connected(node_ids, usable_edges):
        raise PoseGraphError("Reliable RGB-D pose graph is disconnected")
    initial = _propagate_initial_poses(node_ids, usable_edges)
    selected_backend = _Open3DBackend() if backend is None else backend
    optimizer = getattr(selected_backend, "optimize_pose_graph", None)
    if optimizer is None:
        raise TypeError("RGB-D pose backend does not implement optimize_pose_graph")
    optimized_value = optimizer(
        node_ids=node_ids,
        initial_camera_to_world=initial,
        edges=usable_edges,
        config=graph_config,
    )
    poses = _normalise_optimized_poses(optimized_value, node_ids)
    residuals = _edge_residuals(node_ids, poses, usable_edges)
    backend_name = str(getattr(selected_backend, "name", "open3d_rgbd"))
    return PoseGraphResult(
        node_ids=node_ids,
        camera_to_world=poses,
        edges=edge_rows,
        optimized=True,
        connected=True,
        reference_node_id=node_ids[0],
        backend=backend_name,
        edge_residuals=residuals,
    )


def _maximum_consecutive_unreliable(
    node_ids: tuple[int, ...], edges: Sequence[PoseEdge]
) -> int:
    maximum = 0
    current = 0
    for left, right in zip(node_ids, node_ids[1:]):
        reliable = any(
            edge.reliable and _edge_between(edge, left, right) for edge in edges
        )
        if reliable:
            current = 0
        else:
            current += 1
            maximum = max(maximum, current)
    return maximum


def validate_pose_trajectory(
    result: PoseGraphResult,
    *,
    thresholds: PoseQualityThresholds | Mapping[str, Any] | None = None,
) -> PoseQualityReport:
    """Audit optimized camera-to-world poses and RGB-D edge residuals.

    The formal scan axis is world +x/-x, selected from the endpoint direction.
    World +y is vertical (camera down) and world +z is forward. The returned
    report never silently repairs, projects, or replaces an invalid pose.
    """

    limits = (
        thresholds
        if isinstance(thresholds, PoseQualityThresholds)
        else PoseQualityThresholds.from_mapping(thresholds)
    )
    failures: list[str] = []
    metrics: dict[str, Any] = {
        "node_count": len(result.node_ids),
        "edge_count": len(result.edges),
        "reliable_edge_count": sum(edge.reliable for edge in result.edges),
        "optimized": bool(result.optimized),
        "connected": bool(result.connected),
        "translation_unit": POSE_TRANSLATION_UNIT,
        "scan_axis_world": "x",
        "vertical_axis_world": "y",
        "forward_axis_world": "z",
    }
    if len(result.node_ids) < 2:
        failures.append("Pose trajectory contains fewer than two nodes")
    if len(set(result.node_ids)) != len(result.node_ids):
        failures.append("Pose trajectory node ids are not unique")
    if not result.optimized:
        failures.append("Pose graph was not optimized")
    structural_edges = tuple(edge for edge in result.edges if edge.structurally_valid)
    graph_connected = _graph_connected(result.node_ids, structural_edges)
    if not result.connected or not graph_connected:
        failures.append("Pose graph is disconnected")

    maximum_unreliable = _maximum_consecutive_unreliable(
        result.node_ids, result.edges
    )
    metrics["maximum_consecutive_unreliable_edges"] = maximum_unreliable
    if maximum_unreliable > limits.maximum_consecutive_unreliable_edges:
        failures.append("Three or more consecutive reliable RGB-D edges are missing")
    if limits.require_all_adjacent_edges:
        for left, right in zip(result.node_ids, result.node_ids[1:]):
            if not any(
                edge.reliable and _edge_between(edge, left, right)
                for edge in result.edges
            ):
                failures.append(
                    f"Required adjacent RGB-D edge {left}<->{right} failed quality"
                )

    invalid_nodes = [
        node_id
        for node_id, pose in zip(
            result.node_ids, result.camera_to_world, strict=True
        )
        if not _is_valid_se3(pose)
    ]
    metrics["invalid_se3_node_ids"] = invalid_nodes
    if invalid_nodes:
        failures.append("Optimized trajectory contains a non-finite or non-rigid SE(3)")
    if len(result.camera_to_world) != len(result.node_ids):
        failures.append("Pose trajectory has inconsistent node and pose counts")

    if not invalid_nodes and len(result.camera_to_world) >= 2:
        poses = result.camera_to_world
        positions = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
        steps = np.diff(positions, axis=0)
        x_steps = steps[:, 0]
        endpoint_x = float(positions[-1, 0] - positions[0, 0])
        if abs(endpoint_x) > 1e-12:
            direction = 1 if endpoint_x > 0.0 else -1
        else:
            nonzero = x_steps[np.abs(x_steps) > 1e-12]
            direction = 0 if nonzero.size == 0 else (1 if np.median(nonzero) > 0 else -1)
        directed_steps = x_steps * direction if direction else np.zeros_like(x_steps)
        reverse_distances = np.maximum(-directed_steps, 0.0)
        scan_span_mm = abs(endpoint_x)
        reverse_distance_mm = float(reverse_distances.sum())
        forward_distance_mm = float(np.maximum(directed_steps, 0.0).sum())
        reverse_fraction = reverse_distance_mm / max(
            reverse_distance_mm + forward_distance_mm, 1e-12
        )
        step_rotation_deg = np.asarray(
            [
                _rotation_angle_deg(
                    (np.linalg.inv(left) @ right)[:3, :3]
                )
                for left, right in zip(poses, poses[1:])
            ],
            dtype=np.float64,
        )
        total_rotation_deg = np.asarray(
            [
                _rotation_angle_deg((np.linalg.inv(poses[0]) @ pose)[:3, :3])
                for pose in poses
            ],
            dtype=np.float64,
        )
        metrics.update(
            {
                "scan_direction": direction,
                "scan_span_mm": scan_span_mm,
                "maximum_reverse_step_mm": float(reverse_distances.max()),
                "reverse_fraction": reverse_fraction,
                "maximum_step_translation_mm": float(
                    np.linalg.norm(steps, axis=1).max()
                ),
                "maximum_step_vertical_mm": float(np.abs(steps[:, 1]).max()),
                "maximum_step_forward_mm": float(np.abs(steps[:, 2]).max()),
                "maximum_total_vertical_drift_mm": float(
                    np.abs(positions[:, 1] - positions[0, 1]).max()
                ),
                "maximum_total_forward_drift_mm": float(
                    np.abs(positions[:, 2] - positions[0, 2]).max()
                ),
                "maximum_step_rotation_deg": float(step_rotation_deg.max()),
                "maximum_total_rotation_deg": float(total_rotation_deg.max()),
            }
        )
        if scan_span_mm < limits.minimum_scan_span_mm:
            failures.append("Pose trajectory has insufficient lateral scan span")
        if (
            metrics["maximum_reverse_step_mm"] > limits.maximum_reverse_step_mm
            or reverse_fraction > limits.maximum_reverse_fraction
        ):
            failures.append("Pose trajectory is not continuous unidirectional side motion")
        comparisons = (
            ("maximum_step_translation_mm", limits.maximum_step_translation_mm, "translation"),
            ("maximum_step_vertical_mm", limits.maximum_step_vertical_mm, "vertical motion"),
            ("maximum_step_forward_mm", limits.maximum_step_forward_mm, "forward motion"),
            (
                "maximum_total_vertical_drift_mm",
                limits.maximum_total_vertical_drift_mm,
                "vertical drift",
            ),
            (
                "maximum_total_forward_drift_mm",
                limits.maximum_total_forward_drift_mm,
                "forward drift",
            ),
            ("maximum_step_rotation_deg", limits.maximum_step_rotation_deg, "step rotation"),
            ("maximum_total_rotation_deg", limits.maximum_total_rotation_deg, "total rotation"),
        )
        for metric_name, maximum, label in comparisons:
            if float(metrics[metric_name]) > maximum:
                failures.append(f"Pose trajectory {label} exceeds the safety threshold")

    translation_residuals = [
        float(row["translation_residual_mm"]) for row in result.edge_residuals
    ]
    rotation_residuals = [
        float(row["rotation_residual_deg"]) for row in result.edge_residuals
    ]
    maximum_translation_residual = max(translation_residuals, default=math.inf)
    maximum_rotation_residual = max(rotation_residuals, default=math.inf)
    metrics["maximum_edge_translation_residual_mm"] = maximum_translation_residual
    metrics["maximum_edge_rotation_residual_deg"] = maximum_rotation_residual
    if not result.edge_residuals:
        failures.append("Pose graph has no auditable RGB-D edge residuals")
    else:
        if (
            not math.isfinite(maximum_translation_residual)
            or maximum_translation_residual
            > limits.maximum_edge_translation_residual_mm
        ):
            failures.append("Pose-graph translation residual exceeds the safety threshold")
        if (
            not math.isfinite(maximum_rotation_residual)
            or maximum_rotation_residual > limits.maximum_edge_rotation_residual_deg
        ):
            failures.append("Pose-graph rotation residual exceeds the safety threshold")

    unique_failures = tuple(dict.fromkeys(failures))
    return PoseQualityReport(
        quality_pass=not unique_failures,
        failure_reasons=unique_failures,
        metrics=metrics,
        thresholds=limits,
    )


__all__ = [
    "CAMERA_COORDINATE_CONVENTION",
    "POSE_COORDINATE_CONVENTION",
    "POSE_TRANSLATION_UNIT",
    "Open3DUnavailableError",
    "PoseEdge",
    "PoseGraphConfig",
    "PoseGraphError",
    "PoseGraphResult",
    "PoseQualityReport",
    "PoseQualityThresholds",
    "PreparedRGBDFrame",
    "RGBDOdometryConfig",
    "estimate_pair_rgbd_odometry",
    "optimize_rgbd_pose_graph",
    "validate_pose_trajectory",
]
