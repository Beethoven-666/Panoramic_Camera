"""Strict, audit-only loaders for optional V1 input sidecars.

``frames.csv`` and ``calibration.json`` remain authoritative for captured
pixels.  Likewise, :mod:`panorama_demo.orbslam3_bridge` remains the formal
trajectory producer.  The loaders in this module validate a supplied
``camera.yaml`` or ``transforms.json`` and make the contents available to an
explicit caller; merely placing either file in a session never changes the
formal calibration or trajectory.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from .session import CameraIntrinsics, RGBDSession


V1_IMAGE_WIDTH = 1280
V1_IMAGE_HEIGHT = 800
_DISTORTION_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
_OPENCV_COORDINATE_NAMES = {
    "opencv",
    "opencv_camera",
    "opencv_x_right_y_down_z_forward",
    "x_right_y_down_z_forward",
}


def _declares_opencv_coordinates(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().lower().split())
    if normalized in _OPENCV_COORDINATE_NAMES:
        return True
    # Compatibility with rgbd_odometry.CAMERA_COORDINATE_CONVENTION.  Require
    # every signed axis, rather than accepting an arbitrary sentence that only
    # happens to mention OpenCV.
    return (
        "opencv" in normalized
        and "+x right" in normalized
        and "+y down" in normalized
        and "+z forward" in normalized
    )


def _declares_camera_to_world(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().lower().split())
    return normalized == "camera_to_world" or normalized.startswith(
        "camera_to_world maps camera coordinates "
    )


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"camera.yaml contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _finite_float(value: Any, *, context: str, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{context} must be {qualifier}")
    return result


def _positive_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return int(value)


def _yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = yaml.load(
            path.read_text(encoding="utf-8"),
            Loader=_UniqueKeySafeLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid camera.yaml: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("camera.yaml must contain a mapping")
    if not all(isinstance(key, str) for key in payload):
        raise ValueError("camera.yaml keys must be strings")
    return payload


def _json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"transforms.json contains non-finite number {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"transforms.json contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid transforms.json: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("transforms.json must contain an object")
    return payload


def _opencv_matrix_data(value: Any, *, context: str) -> list[Any]:
    if isinstance(value, Mapping):
        data = value.get("data")
    else:
        data = value
    if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
        raise ValueError(f"{context} must contain a numeric data sequence")
    return list(data)


def _camera_dimensions(payload: Mapping[str, Any]) -> tuple[int, int]:
    candidates: list[tuple[Any, Any, str]] = []
    if "image_width" in payload or "image_height" in payload:
        candidates.append(
            (payload.get("image_width"), payload.get("image_height"), "image")
        )
    if "width" in payload or "height" in payload:
        candidates.append((payload.get("width"), payload.get("height"), "root"))
    resolution = payload.get("resolution")
    if resolution is not None:
        if (
            not isinstance(resolution, Sequence)
            or isinstance(resolution, (str, bytes))
            or len(resolution) != 2
        ):
            raise ValueError("camera.yaml resolution must be [width, height]")
        candidates.append((resolution[0], resolution[1], "resolution"))
    intrinsics = payload.get("intrinsics")
    if isinstance(intrinsics, Mapping) and (
        "width" in intrinsics or "height" in intrinsics
    ):
        candidates.append(
            (intrinsics.get("width"), intrinsics.get("height"), "intrinsics")
        )
    if not candidates:
        raise ValueError("camera.yaml is missing image dimensions")
    parsed = [
        (
            _positive_int(width, context=f"camera.yaml {label} width"),
            _positive_int(height, context=f"camera.yaml {label} height"),
        )
        for width, height, label in candidates
    ]
    if any(value != parsed[0] for value in parsed[1:]):
        raise ValueError("camera.yaml contains conflicting image dimensions")
    width, height = parsed[0]
    if (width, height) != (V1_IMAGE_WIDTH, V1_IMAGE_HEIGHT):
        raise ValueError(
            "V1 camera.yaml requires an image resolution of "
            f"{V1_IMAGE_WIDTH}x{V1_IMAGE_HEIGHT}"
        )
    return width, height


def _camera_pinhole(payload: Mapping[str, Any]) -> tuple[float, float, float, float]:
    direct = payload.get("intrinsics")
    direct_values: tuple[float, float, float, float] | None = None
    if isinstance(direct, Mapping) and all(
        key in direct for key in ("fx", "fy", "cx", "cy")
    ):
        direct_values = tuple(
            _finite_float(
                direct[key],
                context=f"camera.yaml intrinsics.{key}",
                positive=key in {"fx", "fy"},
            )
            for key in ("fx", "fy", "cx", "cy")
        )
    elif all(key in payload for key in ("fx", "fy", "cx", "cy")):
        direct_values = tuple(
            _finite_float(
                payload[key],
                context=f"camera.yaml {key}",
                positive=key in {"fx", "fy"},
            )
            for key in ("fx", "fy", "cx", "cy")
        )

    matrix_values: tuple[float, float, float, float] | None = None
    if "camera_matrix" in payload:
        data = _opencv_matrix_data(
            payload["camera_matrix"], context="camera.yaml camera_matrix"
        )
        if len(data) != 9:
            raise ValueError("camera.yaml camera_matrix must contain 9 values")
        matrix = np.asarray(
            [
                _finite_float(value, context="camera.yaml camera_matrix")
                for value in data
            ],
            dtype=np.float64,
        ).reshape(3, 3)
        if not np.allclose(
            matrix,
            np.array(
                [
                    [matrix[0, 0], 0.0, matrix[0, 2]],
                    [0.0, matrix[1, 1], matrix[1, 2]],
                    [0.0, 0.0, 1.0],
                ]
            ),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("camera.yaml camera_matrix is not a pinhole K matrix")
        matrix_values = (
            _finite_float(matrix[0, 0], context="camera.yaml fx", positive=True),
            _finite_float(matrix[1, 1], context="camera.yaml fy", positive=True),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
        )
    if direct_values is None and matrix_values is None:
        raise ValueError("camera.yaml is missing camera intrinsics")
    if (
        direct_values is not None
        and matrix_values is not None
        and not np.allclose(direct_values, matrix_values, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("camera.yaml contains conflicting camera intrinsics")
    return direct_values if direct_values is not None else matrix_values  # type: ignore[return-value]


def _normalized_distortion(payload: Mapping[str, Any]) -> tuple[float, ...]:
    raw = payload.get("distortion_coefficients", payload.get("distortion"))
    if raw is None:
        raise ValueError("camera.yaml is missing distortion coefficients")
    if isinstance(raw, Mapping) and "data" not in raw:
        unknown = set(raw) - set(_DISTORTION_NAMES)
        if unknown:
            raise ValueError(
                "camera.yaml distortion contains unsupported keys: "
                + ", ".join(sorted(str(value) for value in unknown))
            )
        values = [
            _finite_float(
                raw.get(key, 0.0),
                context=f"camera.yaml distortion.{key}",
            )
            for key in _DISTORTION_NAMES
        ]
    else:
        values = [
            _finite_float(value, context="camera.yaml distortion")
            for value in _opencv_matrix_data(
                raw, context="camera.yaml distortion_coefficients"
            )
        ]
        if len(values) not in {4, 5, 8}:
            raise ValueError(
                "camera.yaml distortion must contain 4, 5, or 8 OpenCV coefficients"
            )
        values.extend([0.0] * (8 - len(values)))
    return tuple(values)


def _depth_scale(payload: Mapping[str, Any]) -> float:
    if "depth_scale_mm_per_unit" in payload:
        value = payload["depth_scale_mm_per_unit"]
    else:
        depth = payload.get("depth")
        if not isinstance(depth, Mapping) or "scale_mm_per_unit" not in depth:
            raise ValueError("camera.yaml is missing depth_scale_mm_per_unit")
        value = depth["scale_mm_per_unit"]
    return _finite_float(
        value,
        context="camera.yaml depth_scale_mm_per_unit",
        positive=True,
    )


@dataclass(frozen=True)
class V1CameraAudit:
    """Validated V1 camera sidecar, consistent with the formal session."""

    path: Path
    intrinsics: CameraIntrinsics
    depth_scale_mm_per_unit: float
    depth_alignment: str
    authority: str = "audit_only_frames_csv_and_calibration_json_remain_authoritative"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "resolution": [self.intrinsics.width, self.intrinsics.height],
            "intrinsics": {
                "fx": self.intrinsics.fx,
                "fy": self.intrinsics.fy,
                "cx": self.intrinsics.cx,
                "cy": self.intrinsics.cy,
                "distortion": list(self.intrinsics.distortion),
            },
            "depth_scale_mm_per_unit": self.depth_scale_mm_per_unit,
            "depth_alignment": self.depth_alignment,
            "authority": self.authority,
        }


def load_v1_camera_yaml(
    path: str | Path, *, session: RGBDSession
) -> V1CameraAudit:
    """Validate ``camera.yaml`` against an already strict RGB-D session."""

    camera_path = Path(path).expanduser().resolve()
    payload = _yaml_mapping(camera_path)
    width, height = _camera_dimensions(payload)
    fx, fy, cx, cy = _camera_pinhole(payload)
    if not 0.0 <= cx < width or not 0.0 <= cy < height:
        raise ValueError("camera.yaml principal point lies outside the image")
    distortion = _normalized_distortion(payload)
    scale = _depth_scale(payload)

    alignment = payload.get("depth_alignment", "color")
    if isinstance(alignment, Mapping):
        alignment = alignment.get("aligned_to", alignment.get("target"))
    if not isinstance(alignment, str) or alignment.strip().lower() not in {
        "color",
        "rgb",
        "to_color",
        "aligned_to_color",
    }:
        raise ValueError("camera.yaml must declare depth aligned to color")

    coordinate_name = payload.get("camera_coordinates")
    if coordinate_name is not None and not _declares_opencv_coordinates(
        coordinate_name
    ):
        raise ValueError("camera.yaml camera_coordinates is not OpenCV-compatible")

    intrinsics = CameraIntrinsics(width, height, fx, fy, cx, cy, distortion)
    formal = session.calibration
    if (formal.width, formal.height) != (width, height):
        raise ValueError("camera.yaml resolution does not match calibration.json")
    formal_values = np.asarray(
        [formal.fx, formal.fy, formal.cx, formal.cy, *formal.distortion],
        dtype=np.float64,
    )
    sidecar_values = np.asarray(
        [fx, fy, cx, cy, *distortion],
        dtype=np.float64,
    )
    if formal_values.shape != sidecar_values.shape or not np.allclose(
        formal_values, sidecar_values, rtol=0.0, atol=1e-9
    ):
        raise ValueError("camera.yaml intrinsics/distortion do not match calibration.json")
    frame_scales = np.asarray(
        [frame.depth_scale_mm_per_unit for frame in session.frames],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(frame_scales))
        or np.any(frame_scales <= 0.0)
        or not np.allclose(frame_scales, scale, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("camera.yaml depth scale does not match every frames.csv row")
    return V1CameraAudit(
        path=camera_path,
        intrinsics=intrinsics,
        depth_scale_mm_per_unit=scale,
        depth_alignment="color",
    )


def _rigid_camera_to_world(value: Any, *, node_id: int) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"transforms.json node {node_id} camera_to_world is not numeric"
        ) from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(
            f"transforms.json node {node_id} camera_to_world must be a finite 4x4 matrix"
        )
    if not np.allclose(
        pose[3], np.array([0.0, 0.0, 0.0, 1.0]), rtol=0.0, atol=1e-9
    ):
        raise ValueError(
            f"transforms.json node {node_id} has an invalid homogeneous row"
        )
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
        raise ValueError(
            f"transforms.json node {node_id} rotation is not orthonormal"
        )
    if not math.isclose(
        float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(
            f"transforms.json node {node_id} rotation determinant is not +1"
        )
    return pose.copy()


def _id_sequence(value: Any, *, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    result = tuple(
        _nonnegative_int(item, context=f"{context} entry") for item in value
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicate frame ids")
    return result


def _finite_json_tree(value: Any, *, context: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{context} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json_tree(item, context=f"{context}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} contains a non-string key")
            _finite_json_tree(item, context=f"{context}.{key}")
        return
    raise ValueError(f"{context} contains an unsupported JSON value")


@dataclass(frozen=True)
class V1TrajectoryAudit:
    """Validated external trajectory sidecar.

    This type intentionally does not implement the pose-graph optimizer
    protocol.  Formal code must explicitly compare/consume this audit after
    ORB-SLAM3 has run; it is never an automatic trajectory fallback.
    """

    path: Path
    node_ids: tuple[int, ...]
    camera_to_world_mm: tuple[np.ndarray, ...]
    tracked_frame_ids: tuple[int, ...]
    edge_residuals: tuple[dict[str, Any], ...]
    global_trajectory: dict[str, Any]
    backend: str
    authority: str = "external_sidecar_audit_only_not_an_orbslam3_fallback"

    def pose_for_audit(self, frame_id: int) -> np.ndarray:
        try:
            index = self.node_ids.index(int(frame_id))
        except ValueError as exc:
            raise KeyError(frame_id) from exc
        return self.camera_to_world_mm[index].copy()

    def poses_for_audit(self, frame_ids: Iterable[int]) -> tuple[np.ndarray, ...]:
        return tuple(self.pose_for_audit(frame_id) for frame_id in frame_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "translation_unit": "mm",
            "pose_convention": "camera_to_world",
            "node_ids": list(self.node_ids),
            "tracked_frame_ids": list(self.tracked_frame_ids),
            "backend": self.backend,
            "edge_residual_count": len(self.edge_residuals),
            "authority": self.authority,
        }


def load_v1_transforms_json(
    path: str | Path,
    *,
    required_frame_ids: Iterable[int] | None = None,
) -> V1TrajectoryAudit:
    """Strictly parse an audit trajectory with millimetre camera-to-world SE(3)."""

    transforms_path = Path(path).expanduser().resolve()
    payload = _json_mapping(transforms_path)
    if payload.get("translation_unit") != "mm":
        raise ValueError("transforms.json translation_unit must be 'mm'")
    if not _declares_camera_to_world(payload.get("pose_convention")):
        raise ValueError("transforms.json pose_convention must be 'camera_to_world'")
    camera_coordinates = payload.get("camera_coordinates")
    if not _declares_opencv_coordinates(camera_coordinates):
        raise ValueError(
            "transforms.json must declare OpenCV x-right/y-down/z-forward coordinates"
        )

    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("transforms.json nodes must be a non-empty array")
    node_ids: list[int] = []
    poses: list[np.ndarray] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise ValueError(f"transforms.json nodes[{index}] must be an object")
        node_id = _nonnegative_int(
            node.get("node_id"),
            context=f"transforms.json nodes[{index}].node_id",
        )
        if node_id in node_ids:
            raise ValueError(f"transforms.json contains duplicate node_id {node_id}")
        node_ids.append(node_id)
        poses.append(_rigid_camera_to_world(node.get("camera_to_world"), node_id=node_id))

    edge_residuals_value = payload.get("edge_residuals")
    if not isinstance(edge_residuals_value, list):
        raise ValueError("transforms.json edge_residuals must be an array")
    edge_residuals: list[dict[str, Any]] = []
    for index, residual in enumerate(edge_residuals_value):
        if not isinstance(residual, dict):
            raise ValueError(
                f"transforms.json edge_residuals[{index}] must be an object"
            )
        _finite_json_tree(
            residual, context=f"transforms.json edge_residuals[{index}]"
        )
        edge_residuals.append(dict(residual))

    global_value = payload.get("global_trajectory")
    if not isinstance(global_value, dict):
        raise ValueError("transforms.json global_trajectory must be an object")
    _finite_json_tree(global_value, context="transforms.json global_trajectory")
    if global_value.get("translation_unit") != "mm":
        raise ValueError("global_trajectory translation_unit must be 'mm'")
    if not _declares_camera_to_world(global_value.get("pose_convention")):
        raise ValueError(
            "global_trajectory pose_convention must be 'camera_to_world'"
        )
    backend = global_value.get("backend")
    if not isinstance(backend, str) or "orbslam3" not in backend.lower():
        raise ValueError("global_trajectory backend must identify ORB-SLAM3 RGB-D")
    tracked = _id_sequence(
        global_value.get("tracked_frame_ids"),
        context="global_trajectory tracked_frame_ids",
    )
    root_tracked = payload.get("tracked_frame_ids")
    if root_tracked is not None and _id_sequence(
        root_tracked, context="transforms.json tracked_frame_ids"
    ) != tracked:
        raise ValueError(
            "root and global_trajectory tracked_frame_ids do not match"
        )
    missing_tracked_nodes = sorted(set(node_ids) - set(tracked))
    if missing_tracked_nodes:
        raise ValueError(
            "transforms.json nodes are absent from tracked_frame_ids: "
            + ", ".join(str(value) for value in missing_tracked_nodes)
        )

    required = (
        tuple(
            _nonnegative_int(
                value, context="required_frame_ids entry"
            )
            for value in required_frame_ids
        )
        if required_frame_ids is not None
        else ()
    )
    if len(set(required)) != len(required):
        raise ValueError("required_frame_ids contains duplicates")
    missing_nodes = sorted(set(required) - set(node_ids))
    missing_tracking = sorted(set(required) - set(tracked))
    if missing_nodes or missing_tracking:
        reasons: list[str] = []
        if missing_nodes:
            reasons.append(
                "pose nodes " + ", ".join(str(value) for value in missing_nodes)
            )
        if missing_tracking:
            reasons.append(
                "tracked ids " + ", ".join(str(value) for value in missing_tracking)
            )
        raise ValueError(
            "transforms.json does not completely cover required frames: "
            + "; ".join(reasons)
        )
    return V1TrajectoryAudit(
        path=transforms_path,
        node_ids=tuple(node_ids),
        camera_to_world_mm=tuple(poses),
        tracked_frame_ids=tracked,
        edge_residuals=tuple(edge_residuals),
        global_trajectory=dict(global_value),
        backend=backend,
    )


@dataclass(frozen=True)
class V1InputSidecarAudit:
    root: Path
    camera: V1CameraAudit | None
    trajectory: V1TrajectoryAudit | None
    required_frame_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "camera_yaml": None if self.camera is None else self.camera.as_dict(),
            "transforms_json": (
                None if self.trajectory is None else self.trajectory.as_dict()
            ),
            "required_frame_ids": list(self.required_frame_ids),
            "sidecars_change_formal_authority": False,
        }


def audit_v1_input_sidecars(
    session: RGBDSession,
    *,
    used_frame_ids: Iterable[int] | None = None,
) -> V1InputSidecarAudit:
    """Audit optional sidecars without changing formal session/ORB authority.

    When ``used_frame_ids`` is omitted, all frames in the strict session are
    treated as used.  A present ``transforms.json`` must completely cover that
    set.  Absence of either sidecar preserves the existing input behavior.
    """

    required = (
        tuple(int(value) for value in used_frame_ids)
        if used_frame_ids is not None
        else tuple(frame.frame_id for frame in session.frames)
    )
    known = {frame.frame_id for frame in session.frames}
    unknown = sorted(set(required) - known)
    if unknown:
        raise ValueError(
            "used_frame_ids are absent from frames.csv: "
            + ", ".join(str(value) for value in unknown)
        )
    camera_path = session.root / "camera.yaml"
    transforms_path = session.root / "transforms.json"
    camera = (
        load_v1_camera_yaml(camera_path, session=session)
        if camera_path.exists()
        else None
    )
    trajectory = (
        load_v1_transforms_json(
            transforms_path,
            required_frame_ids=required,
        )
        if transforms_path.exists()
        else None
    )
    return V1InputSidecarAudit(
        root=session.root,
        camera=camera,
        trajectory=trajectory,
        required_frame_ids=required,
    )


__all__ = [
    "V1CameraAudit",
    "V1InputSidecarAudit",
    "V1TrajectoryAudit",
    "audit_v1_input_sidecars",
    "load_v1_camera_yaml",
    "load_v1_transforms_json",
]
