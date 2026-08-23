"""Explicit direct-ORB trajectory parsing and pose qualification for S1.2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .session import RGBDFrame
from .video_s12_config import S12TrajectoryConfig


S12_TRAJECTORY_SCHEMAS = {
    "gemini305-orbslam3-trajectory/v1",
    "gemini305-orbslam3-trajectory/v2",
    "gemini305-video-s12-trajectory/v1",
}
_DIRECT_KINDS = {"direct_orb_pose", "direct_orbslam3", "direct_orb_anchor"}
_TRACKED_STATES = {"tracked", "tracking", "ok", "valid"}


@dataclass(frozen=True)
class S12PoseRecord:
    frame_id: int
    timestamp_us: int
    pose_status: str
    pose_kind: str
    pose_origin: str
    camera_to_world: np.ndarray | None
    tracking_state: str

    @property
    def is_direct(self) -> bool:
        return (
            self.pose_kind in _DIRECT_KINDS
            and self.pose_status.lower() in _TRACKED_STATES
            and self.tracking_state.lower() in _TRACKED_STATES
            and self.camera_to_world is not None
        )

    @property
    def camera_center_m(self) -> np.ndarray:
        if self.camera_to_world is None:
            raise ValueError(f"Frame {self.frame_id} has no camera pose")
        return np.asarray(self.camera_to_world[:3, 3], dtype=np.float64) / 1000.0


@dataclass(frozen=True)
class S12Trajectory:
    path: Path
    schema: str
    pose_origin: str
    records: tuple[S12PoseRecord, ...]
    pose_convention: str = "camera_to_world"
    translation_unit: str = "mm"

    @property
    def direct_records(self) -> tuple[S12PoseRecord, ...]:
        return tuple(record for record in self.records if record.is_direct)

    def by_frame_id(self) -> dict[int, S12PoseRecord]:
        return {record.frame_id: record for record in self.records}


@dataclass(frozen=True)
class S12PoseQualification:
    frame_id: int
    timestamp_us: int
    direct_pose: bool
    timestamp_delta_ms: float | None
    roll_deg: float | None
    pitch_deg: float | None
    yaw_deg: float | None
    qualified: bool
    rejection_reasons: tuple[str, ...]
    record: S12PoseRecord | None


def validate_se3(value: object, *, frame_id: int) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError(f"S1.2 pose for frame {frame_id} is not a finite 4x4 SE(3)")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError(f"S1.2 pose for frame {frame_id} has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError(f"S1.2 pose for frame {frame_id} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"S1.2 pose for frame {frame_id} rotation determinant is not +1")
    result = pose.copy()
    result.setflags(write=False)
    return result


def _required_string(row: Mapping[str, Any], key: str, *, frame_id: object) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"S1.2 trajectory frame {frame_id} requires explicit {key}")
    return value.strip()


def _timestamp_us(row: Mapping[str, Any], *, frame_id: int, timestamp_unit: object) -> int:
    if "timestamp_us" in row:
        value = row["timestamp_us"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"S1.2 trajectory frame {frame_id} has invalid timestamp_us")
        return value
    value = row.get("timestamp")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"S1.2 trajectory frame {frame_id} requires an explicit timestamp")
    scales = {"us": 1.0, "ms": 1000.0, "s": 1_000_000.0}
    if timestamp_unit not in scales:
        raise ValueError("S1.2 trajectory timestamp requires explicit timestamp_unit us/ms/s")
    result = int(round(float(value) * scales[str(timestamp_unit)]))
    if result < 0:
        raise ValueError(f"S1.2 trajectory frame {frame_id} has a negative timestamp")
    return result


def parse_s12_trajectory(payload: Mapping[str, Any], *, path: Path | None = None) -> S12Trajectory:
    schema = payload.get("schema")
    if schema not in S12_TRAJECTORY_SCHEMAS:
        raise ValueError("S1.2 trajectory file has an unsupported schema")
    if payload.get("pose_convention") != "camera_to_world":
        raise ValueError("S1.2 trajectory must explicitly use camera_to_world poses")
    if payload.get("translation_unit") != "mm":
        raise ValueError("S1.2 trajectory translation_unit must be mm")
    rows = payload.get("poses")
    if not isinstance(rows, list) or not rows:
        raise ValueError("S1.2 trajectory requires a non-empty poses array")
    records: list[S12PoseRecord] = []
    seen: set[int] = set()
    origins: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("S1.2 trajectory pose entries must be objects")
        frame_id = row.get("frame_id")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("S1.2 trajectory pose has an invalid frame_id")
        if frame_id in seen:
            raise ValueError(f"S1.2 trajectory contains duplicate frame_id {frame_id}")
        pose_status = _required_string(row, "pose_status", frame_id=frame_id)
        pose_kind = _required_string(row, "pose_kind", frame_id=frame_id)
        pose_origin = _required_string(row, "pose_origin", frame_id=frame_id)
        tracking_state = _required_string(row, "tracking_state", frame_id=frame_id)
        matrix_value = row.get("camera_to_world")
        matrix = None if matrix_value is None else validate_se3(matrix_value, frame_id=frame_id)
        if pose_status.lower() in _TRACKED_STATES and matrix is None:
            raise ValueError(f"S1.2 tracked pose frame {frame_id} has no camera_to_world")
        records.append(
            S12PoseRecord(
                frame_id=frame_id,
                timestamp_us=_timestamp_us(
                    row, frame_id=frame_id, timestamp_unit=payload.get("timestamp_unit")
                ),
                pose_status=pose_status,
                pose_kind=pose_kind,
                pose_origin=pose_origin,
                camera_to_world=matrix,
                tracking_state=tracking_state,
            )
        )
        seen.add(frame_id)
        origins.add(pose_origin)
    if len(origins) != 1:
        raise ValueError("S1.2 trajectory mixes pose origins")
    if [record.timestamp_us for record in records] != sorted(
        record.timestamp_us for record in records
    ):
        raise ValueError("S1.2 trajectory poses must be chronological")
    return S12Trajectory(
        path=(path or Path("<memory>")),
        schema=str(schema),
        pose_origin=next(iter(origins)),
        records=tuple(records),
    )


def load_s12_trajectory(path: str | Path) -> S12Trajectory:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid S1.2 trajectory: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("S1.2 trajectory root must be an object")
    return parse_s12_trajectory(payload, path=resolved)


def _relative_euler_deg(reference: np.ndarray, pose: np.ndarray) -> tuple[float, float, float]:
    rotation = reference[:3, :3].T @ pose[:3, :3]
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    return tuple(float(math.degrees(value)) for value in (roll, pitch, yaw))


def qualify_s12_poses(
    frames: Sequence[RGBDFrame],
    trajectory: S12Trajectory,
    config: S12TrajectoryConfig,
) -> tuple[S12PoseQualification, ...]:
    by_id = trajectory.by_frame_id()
    direct = trajectory.direct_records
    reference = direct[0].camera_to_world if direct else None
    results: list[S12PoseQualification] = []
    for frame in frames:
        record = by_id.get(frame.frame_id)
        reasons: list[str] = []
        timestamp_delta: float | None = None
        angles: tuple[float | None, float | None, float | None] = (None, None, None)
        if record is None:
            reasons.append("missing_trajectory_record")
        elif not record.is_direct:
            reasons.append("not_direct_orb_pose")
        else:
            if frame.timestamp_us is None:
                reasons.append("missing_frame_timestamp")
            else:
                timestamp_delta = abs(record.timestamp_us - frame.timestamp_us) / 1000.0
                if timestamp_delta > config.maximum_timestamp_delta_ms:
                    reasons.append("timestamp_mismatch")
            if config.audit_rotation and reference is not None:
                assert record.camera_to_world is not None
                angles = _relative_euler_deg(reference, record.camera_to_world)
                limits = (
                    config.maximum_relative_roll_deg,
                    config.maximum_relative_pitch_deg,
                    config.maximum_relative_yaw_deg,
                )
                names = ("roll_limit", "pitch_limit", "yaw_limit")
                reasons.extend(name for value, limit, name in zip(angles, limits, names) if abs(value) > limit)
        results.append(
            S12PoseQualification(
                frame_id=frame.frame_id,
                timestamp_us=int(frame.timestamp_us or 0),
                direct_pose=bool(record and record.is_direct),
                timestamp_delta_ms=timestamp_delta,
                roll_deg=angles[0],
                pitch_deg=angles[1],
                yaw_deg=angles[2],
                qualified=not reasons,
                rejection_reasons=tuple(reasons),
                record=record,
            )
        )
    return tuple(results)


__all__ = [
    "S12PoseQualification",
    "S12PoseRecord",
    "S12Trajectory",
    "load_s12_trajectory",
    "parse_s12_trajectory",
    "qualify_s12_poses",
    "validate_se3",
]
