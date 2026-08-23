"""Explicit, non-interpolating trajectory sources for S1.3."""

from __future__ import annotations

import json
import hashlib
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .video_s12_pose import parse_s12_trajectory, validate_se3
from .video_s13_session import S13Session


@dataclass(frozen=True)
class S13Trajectory:
    source: str
    source_path: Path | None
    poses_by_frame_id: Mapping[int, np.ndarray]
    pose_origin_by_frame_id: Mapping[int, str]
    islands: tuple[tuple[int, ...], ...]
    audit: Mapping[str, Any]
    pose_epoch_by_frame_id: Mapping[int, str] = field(default_factory=dict)


def _islands(
    frame_ids: tuple[int, ...], posed: set[int], epochs: Mapping[int, str]
) -> tuple[tuple[int, ...], ...]:
    result: list[tuple[int, ...]] = []
    current: list[int] = []
    current_epoch: str | None = None
    for frame_id in frame_ids:
        if frame_id in posed:
            epoch = epochs.get(frame_id, "default")
            if current and epoch != current_epoch:
                result.append(tuple(current))
                current = []
            current.append(frame_id)
            current_epoch = epoch
    if current:
        result.append(tuple(current))
    return tuple(result)


def _parse_payload(
    payload: Mapping[str, Any], *, path: Path, session: S13Session
) -> tuple[dict[int, np.ndarray], dict[int, str], dict[int, str], str]:
    schema = payload.get("schema")
    poses: dict[int, np.ndarray] = {}
    origins: dict[int, str] = {}
    epochs: dict[int, str] = {}
    if schema in {
        "gemini305-orbslam3-trajectory/v1",
        "gemini305-orbslam3-trajectory/v2",
        "gemini305-video-s12-trajectory/v1",
    }:
        declared_session = payload.get("session")
        if not isinstance(declared_session, str) or Path(declared_session).expanduser().resolve() != session.root:
            raise ValueError("S1.3 direct trajectory is not bound to the current session")
        trajectory = parse_s12_trajectory(payload, path=path)
        session_frames = session.frame_by_id
        for record in trajectory.records:
            if record.is_direct and record.camera_to_world is not None:
                frame = session_frames.get(record.frame_id)
                if frame is None or abs(record.timestamp_us - frame.timestamp_us) > 5_000:
                    raise ValueError(f"S1.3 direct pose timestamp mismatch for frame {record.frame_id}")
                poses[record.frame_id] = record.camera_to_world
                origins[record.frame_id] = record.pose_origin
                epochs[record.frame_id] = "default"
        rows = payload.get("poses")
        if isinstance(rows, list):
            epoch_counter = 0
            for row in rows:
                if not isinstance(row, Mapping) or not isinstance(row.get("frame_id"), int):
                    continue
                frame_id = int(row["frame_id"])
                if row.get("reset") is True:
                    epoch_counter += 1
                explicit = row.get("epoch_id", row.get("reset_id"))
                epochs[frame_id] = str(explicit) if explicit is not None else f"default:{epoch_counter}"
        contract = "explicit_direct_pose_records"
    else:
        if schema == "gemini305-video-experiment-trajectory-cache/v1":
            from .video_trajectory_cache import session_input_sha256

            if payload.get("input_sha256") != session_input_sha256(session.root):
                raise ValueError("S1.3 trajectory cache does not match the current session")
            if payload.get("cache_origin") != "published_verified_real_orbslam3_video_report":
                raise ValueError("S1.3 trajectory cache origin is not verified")
            origin = "direct_orb_verified_cache"
        elif schema == "gemini305-online-orbslam3-trajectory/v1":
            from .video_trajectory_cache import session_input_sha256

            if payload.get("input_sha256") != session_input_sha256(session.root):
                raise ValueError("S1.3 online trajectory does not match the current session")
            _verify_online_sources(payload, session)
            origin = "direct_orb_capture_bound_online"
        else:
            raise ValueError("S1.3 trajectory has an unsupported schema")
        orb = payload.get("orbslam3", payload)
        if not isinstance(orb, Mapping) or orb.get("pose_convention") != "camera_to_world" or orb.get("translation_unit") != "mm":
            raise ValueError("S1.3 trajectory requires camera_to_world poses in mm")
        ids, matrices = orb.get("tracked_frame_ids"), orb.get("camera_to_world")
        if not isinstance(ids, list) or not isinstance(matrices, list) or len(ids) != len(matrices):
            raise ValueError("S1.3 trajectory pose arrays are inconsistent")
        for frame_id, matrix in zip(ids, matrices, strict=True):
            if isinstance(frame_id, bool) or not isinstance(frame_id, int):
                raise ValueError("S1.3 trajectory frame id is invalid")
            if frame_id in poses:
                raise ValueError("S1.3 trajectory frame ids are not unique")
            poses[frame_id] = validate_se3(matrix, frame_id=frame_id)
            origins[frame_id] = origin
            epochs[frame_id] = "default"
        epoch_ids = orb.get("epoch_ids")
        if isinstance(epoch_ids, list) and len(epoch_ids) == len(ids):
            epochs.update({int(frame_id): str(epoch) for frame_id, epoch in zip(ids, epoch_ids, strict=True)})
        contract = "explicit_direct_pose_arrays"
    allowed = {frame.frame_id for frame in session.frames}
    unknown = sorted(set(poses) - allowed)
    if unknown:
        raise ValueError(f"S1.3 trajectory contains frames outside this session: {unknown[:8]}")
    return poses, origins, epochs, contract


def load_s13_trajectory(
    session: S13Session,
    *,
    trajectory_cache: Path | None,
    reuse_online_trajectory: bool,
    run_offline_orb: bool,
    ignore_pose: bool,
    config_path: Path | None = None,
) -> S13Trajectory:
    selected = sum((trajectory_cache is not None, reuse_online_trajectory, run_offline_orb, ignore_pose))
    if selected != 1:
        raise ValueError(
            "S1.3 requires exactly one explicit trajectory source: --trajectory-cache, "
            "--reuse-online-trajectory, --run-offline-orb, or --ignore-pose"
        )
    if ignore_pose:
        return S13Trajectory(
            source="ignore_pose", source_path=None, poses_by_frame_id={}, pose_origin_by_frame_id={}, islands=(),
            audit={"source": "ignore_pose", "direct_pose_count": 0, "pose_supported": False},
        )
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if trajectory_cache is not None:
        source, path = "trajectory_cache", trajectory_cache.expanduser().resolve()
    elif reuse_online_trajectory:
        source, path = "online_trajectory", session.root / "online_orbslam3_trajectory.json"
    else:
        from .export_orbslam3_trajectory import export_trajectory

        temporary = tempfile.TemporaryDirectory(prefix="g305-s13-offline-orb-")
        path = Path(temporary.name) / "orbslam3_trajectory.json"
        export_trajectory(session.root, path, config_path=config_path)
        source = "offline_orbslam3"
    try:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid S1.3 explicit trajectory: {path}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("S1.3 trajectory root must be an object")
        poses, origins, epochs, contract = _parse_payload(payload, path=path, session=session)
    finally:
        if temporary is not None:
            temporary.cleanup()
    ordered_ids = tuple(frame.frame_id for frame in session.frames)
    islands = _islands(ordered_ids, set(poses), epochs)
    return S13Trajectory(
        source=source,
        source_path=None if source == "offline_orbslam3" else path,
        poses_by_frame_id=poses,
        pose_origin_by_frame_id=origins,
        islands=islands,
        audit={
            "source": source,
            "record_contract": contract,
            "direct_pose_count": len(poses),
            "session_rgb_count": len(session.frames),
            "pose_supported": bool(poses),
            "complete_pose_coverage": len(poses) == len(session.frames),
            "pose_islands": [list(island) for island in islands],
            "interpolated_pose_count": 0,
            "extrapolated_pose_count": 0,
        },
        pose_epoch_by_frame_id=epochs,
    )


__all__ = ["S13Trajectory", "load_s13_trajectory"]
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_online_sources(payload: Mapping[str, Any], session: S13Session) -> None:
    if payload.get("capture_origin") != "writer_committed_files" or session.strict_video is None:
        raise ValueError("S1.3 online trajectory lacks strict capture provenance")
    rows = payload.get("source_file_sha256")
    if not isinstance(rows, list):
        raise ValueError("S1.3 online trajectory lacks committed source hashes")
    by_id = {
        int(row["frame_id"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("frame_id"), int)
    }
    strict_by_id = {frame.frame_id: frame for frame in session.strict_video.rgbd.frames}
    for frame in session.frames:
        recorded, strict_frame = by_id.get(frame.frame_id), strict_by_id.get(frame.frame_id)
        if recorded is None or strict_frame is None:
            raise ValueError("S1.3 online trajectory source coverage is incomplete")
        if (
            recorded.get("color_sha256") != _sha256(frame.color_path)
            or recorded.get("aligned_depth_sha256") != _sha256(strict_frame.aligned_depth_path)
        ):
            raise ValueError("S1.3 online trajectory source hashes do not match this session")
