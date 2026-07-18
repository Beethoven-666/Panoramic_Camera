"""WSL bridge for using ORB-SLAM3 RGB-D as the global trajectory tracker.

The Gemini capture format is deliberately not passed straight to the TUM
example: its colour images have a calibrated rational distortion model and
its depth PNG values can use a session-specific unit.  This adapter creates a
temporary, *undistorted* RGB-D TUM sequence, derives ``DepthMapFactor`` from
the session metadata, launches the user's WSL ORB-SLAM3 installation, and
turns the emitted TUM poses into project-standard camera-to-world millimetre
SE(3) matrices.

ORB-SLAM3 remains a separate GPLv3 executable.  Nothing from its source or
library is imported by the Python package, and the bridge only communicates
through its documented RGB-D command-line input and TUM trajectory output.
"""

from __future__ import annotations

import math
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .session import CameraIntrinsics, RGBDFrame


_ORB_SLAM3_NATIVE_ABORT_RETURN_CODE = 134
_ORB_SLAM3_SEGFAULT_RETURN_CODE = 139
_ORB_SLAM3_MAX_EXECUTION_ATTEMPTS = 2
_ORB_SLAM3_RETRYABLE_NATIVE_FAILURES = {
    _ORB_SLAM3_NATIVE_ABORT_RETURN_CODE: (
        "returncode_134_native_heap_abort_fresh_staging_retry"
    ),
    _ORB_SLAM3_SEGFAULT_RETURN_CODE: (
        "returncode_139_native_sigsegv_fresh_staging_retry"
    ),
}
_ORB_SLAM3_NATIVE_FAILURE_SIGNALS = {
    _ORB_SLAM3_NATIVE_ABORT_RETURN_CODE: 6,
    _ORB_SLAM3_SEGFAULT_RETURN_CODE: 11,
}


class ORBSLAM3Error(RuntimeError):
    """ORB-SLAM3 could not produce a usable metric RGB-D trajectory."""

    def __init__(
        self,
        message: str,
        *,
        attempt_audit: Sequence[Mapping[str, object]] = (),
    ) -> None:
        super().__init__(message)
        # This audit deliberately contains only scalar process facts.  It is
        # safe to include in a report without retaining an RGB-D staging path,
        # command, image name, or partial pose from a failed native process.
        self.attempt_audit = tuple(dict(row) for row in attempt_audit)


@dataclass(frozen=True)
class ORBSLAM3Config:
    """Configuration for an externally installed WSL ORB-SLAM3 runtime."""

    enabled: bool = True
    wsl_executable: str = "wsl.exe"
    root: str = "~/Projects/ORB_SLAM3_WS/ORB_SLAM3"
    executable: str = "Examples/RGB-D/rgbd_tum"
    vocabulary: str = "Vocabulary/ORBvoc.txt"
    timeout_seconds: float = 600.0
    minimum_tracked_fraction: float = 0.95
    feature_count: int = 1800
    fast_threshold: int = 12
    minimum_fast_threshold: int = 5

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None = None
    ) -> "ORBSLAM3Config":
        if value is None:
            return cls()
        payload = dict(value)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"Unknown orbslam3_rgbd configuration keys: {unknown}")
        config = cls(**payload)
        if not config.enabled:
            raise ORBSLAM3Error("ORB-SLAM3 RGB-D backend is disabled")
        if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0.0:
            raise ValueError("ORB-SLAM3 timeout_seconds must be positive")
        if not 0.0 < config.minimum_tracked_fraction <= 1.0:
            raise ValueError("ORB-SLAM3 minimum_tracked_fraction must be in (0, 1]")
        if config.feature_count < 500:
            raise ValueError("ORB-SLAM3 feature_count must be at least 500")
        if not 1 <= config.minimum_fast_threshold <= config.fast_threshold <= 255:
            raise ValueError("ORB-SLAM3 FAST thresholds are invalid")
        return config


@dataclass(frozen=True)
class ORBSLAM3Trajectory:
    """Metric, rebased camera poses returned by one ORB-SLAM3 execution."""

    poses_by_frame_id: dict[int, np.ndarray]
    tracked_frame_ids: tuple[int, ...]
    work_dir: Path
    command: tuple[str, ...]
    stdout_path: Path
    stderr_path: Path
    settings_path: Path
    association_path: Path
    trajectory_path: Path
    config: ORBSLAM3Config
    attempt_audit: tuple[dict[str, object], ...] = ()

    @property
    def tracked_fraction(self) -> float:
        return len(self.tracked_frame_ids) / max(1, len(self.poses_by_frame_id))

    def as_dict(self, *, input_frame_count: int) -> dict[str, object]:
        return {
            "backend": "orbslam3_rgbd_wsl",
            "input_frame_count": input_frame_count,
            "tracked_frame_count": len(self.tracked_frame_ids),
            "tracked_fraction": len(self.tracked_frame_ids) / max(1, input_frame_count),
            "tracked_frame_ids": list(self.tracked_frame_ids),
            "work_dir": str(self.work_dir),
            "settings_path": str(self.settings_path),
            "association_path": str(self.association_path),
            "trajectory_path": str(self.trajectory_path),
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "command": list(self.command),
            "config": asdict(self.config),
            "execution_attempts": [dict(row) for row in self.attempt_audit],
            "pose_convention": "camera_to_world",
            "translation_unit": "mm",
        }


@dataclass(frozen=True)
class _StagedORBSLAM3Attempt:
    """One private, self-contained RGB-D staging attempt."""

    stage_dir: Path
    sequence_dir: Path
    association_path: Path
    settings_path: Path
    trajectory_path: Path
    timestamps: tuple[float, ...]
    command: tuple[str, ...]


@dataclass(frozen=True)
class ORBSLAM3PoseGraphOptimizer:
    """Expose verified ORB-SLAM3 poses through the RGB-D pose-graph interface.

    The short-baseline Open3D edges remain the measured local constraints.  This
    object supplies only the global camera poses returned by ORB-SLAM3, so the
    normal pose-graph code can still validate graph connectivity and compute
    every RGB-D edge residual against that global trajectory.
    """

    trajectory: ORBSLAM3Trajectory
    name: str = "orbslam3_rgbd_wsl"

    def optimize_pose_graph(
        self,
        *,
        node_ids: Sequence[int],
        initial_camera_to_world: Sequence[np.ndarray],
        edges: Sequence[Any],
        config: Any,
    ) -> tuple[np.ndarray, ...]:
        del initial_camera_to_world, edges, config
        missing = [int(node_id) for node_id in node_ids if int(node_id) not in self.trajectory.poses_by_frame_id]
        if missing:
            raise ORBSLAM3Error(
                "ORB-SLAM3 did not provide a real pose for required RGB-D nodes: "
                f"{missing}"
            )
        return tuple(
            np.asarray(self.trajectory.poses_by_frame_id[int(node_id)], dtype=np.float64)
            for node_id in node_ids
        )


def _encode_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".png"
    if image.dtype != np.uint8 and not (image.dtype == np.uint16 and image.ndim == 2):
        raise ValueError(f"Unsupported ORB-SLAM3 staging image type: {image.dtype}")
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise OSError(f"Could not encode ORB-SLAM3 staging image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _decode_image(path: Path, flags: int, *, label: str) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, flags) if encoded.size else None
    if image is None:
        raise OSError(f"Could not decode {label}: {path}")
    return image


def _undistortion_maps(intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray] | None:
    distortion = np.asarray(intrinsics.distortion, dtype=np.float64)
    if distortion.size == 0 or not np.any(distortion):
        return None
    return cv2.initUndistortRectifyMap(
        intrinsics.matrix,
        distortion,
        None,
        intrinsics.matrix,
        (intrinsics.width, intrinsics.height),
        cv2.CV_32FC1,
    )


def _stage_rgbd_sequence(
    frames: Sequence[RGBDFrame],
    intrinsics: CameraIntrinsics,
    stage_dir: Path,
) -> Path:
    """Write calibrated undistorted PNG inputs for ORB-SLAM3's pinhole model."""

    sequence_dir = stage_dir / "sequence"
    maps = _undistortion_maps(intrinsics)
    for frame in frames:
        color = _decode_image(frame.color_path, cv2.IMREAD_COLOR, label="colour image")
        depth = _decode_image(
            frame.aligned_depth_path, cv2.IMREAD_UNCHANGED, label="aligned depth image"
        )
        if color.shape[:2] != (intrinsics.height, intrinsics.width):
            raise ORBSLAM3Error(
                f"Frame {frame.frame_id} colour dimensions do not match calibration"
            )
        if depth.dtype != np.uint16 or depth.shape != color.shape[:2]:
            raise ORBSLAM3Error(
                f"Frame {frame.frame_id} depth is not a colour-aligned uint16 PNG"
            )
        if maps is not None:
            map_x, map_y = maps
            color = cv2.remap(
                color,
                map_x,
                map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            depth = cv2.remap(
                depth,
                map_x,
                map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        stem = f"{frame.frame_id:08d}.png"
        _encode_image(sequence_dir / "color" / stem, color)
        _encode_image(sequence_dir / "depth" / stem, depth)
    return sequence_dir


def _timestamps_seconds(frames: Sequence[RGBDFrame]) -> list[float]:
    values: list[float] = []
    for frame in frames:
        if frame.timestamp_us is None or frame.timestamp_us < 0:
            raise ORBSLAM3Error(
                f"Frame {frame.frame_id} lacks a valid colour timestamp for ORB-SLAM3"
            )
        values.append(float(frame.timestamp_us) / 1_000_000.0)
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ORBSLAM3Error("ORB-SLAM3 input timestamps must be strictly increasing")
    return values


def _write_association(frames: Sequence[RGBDFrame], path: Path) -> list[float]:
    timestamps = _timestamps_seconds(frames)
    lines = [
        f"{timestamp:.6f} color/{frame.frame_id:08d}.png "
        f"{timestamp:.6f} depth/{frame.frame_id:08d}.png"
        for timestamp, frame in zip(timestamps, frames, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return timestamps


def _write_settings(
    frames: Sequence[RGBDFrame],
    intrinsics: CameraIntrinsics,
    path: Path,
    config: ORBSLAM3Config,
) -> float:
    depth_scales = np.asarray(
        [frame.depth_scale_mm_per_unit for frame in frames], dtype=np.float64
    )
    if not np.all(np.isfinite(depth_scales)) or np.any(depth_scales <= 0.0):
        raise ORBSLAM3Error("ORB-SLAM3 input has an invalid depth scale")
    if not np.allclose(depth_scales, depth_scales[0], rtol=0.0, atol=1e-9):
        raise ORBSLAM3Error(
            "ORB-SLAM3 bridge requires a single depth unit across one scan"
        )
    # ORB-SLAM3 computes metres as ``uint16_value / DepthMapFactor``.
    depth_map_factor = 1000.0 / float(depth_scales[0])
    timestamps = _timestamps_seconds(frames)
    intervals = np.diff(np.asarray(timestamps, dtype=np.float64))
    fps = int(np.clip(np.rint(1.0 / np.median(intervals)), 1, 120))
    lines = [
        "%YAML:1.0",
        'File.version: "1.0"',
        'Camera.type: "PinHole"',
        f"Camera1.fx: {intrinsics.fx:.12g}",
        f"Camera1.fy: {intrinsics.fy:.12g}",
        f"Camera1.cx: {intrinsics.cx:.12g}",
        f"Camera1.cy: {intrinsics.cy:.12g}",
        # Inputs are explicitly undistorted before staging.  Several
        # ORB-SLAM3 RGB-D builds nevertheless dereference these five classic
        # OpenCV keys while loading a PinHole camera; omitting them merely
        # prints "optional parameter" diagnostics on some builds but then
        # segfaults on others.  Declare the calibrated staged model honestly
        # as zero-distortion instead of allowing a missing-key fallback.
        "Camera1.k1: 0.0",
        "Camera1.k2: 0.0",
        "Camera1.p1: 0.0",
        "Camera1.p2: 0.0",
        "Camera1.k3: 0.0",
        f"Camera.width: {intrinsics.width}",
        f"Camera.height: {intrinsics.height}",
        f"Camera.fps: {fps}",
        # cv::imread supplies BGR and staged colour images are deliberately BGR.
        "Camera.RGB: 0",
        "Stereo.ThDepth: 40.0",
        "Stereo.b: 0.05",
        f"RGBD.DepthMapFactor: {depth_map_factor:.12g}",
        f"ORBextractor.nFeatures: {config.feature_count}",
        "ORBextractor.scaleFactor: 1.2",
        "ORBextractor.nLevels: 8",
        f"ORBextractor.iniThFAST: {config.fast_threshold}",
        f"ORBextractor.minThFAST: {config.minimum_fast_threshold}",
        "Viewer.KeyFrameSize: 0.05",
        "Viewer.KeyFrameLineWidth: 1.0",
        "Viewer.GraphLineWidth: 1.0",
        "Viewer.PointSize: 2.0",
        "Viewer.CameraSize: 0.08",
        "Viewer.CameraLineWidth: 3.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8",
        "Viewer.ViewpointF: 500.0",
        # The installed ORB-SLAM3 Settings implementation passes this value
        # directly into LocalMapping without a default when the optional key
        # is absent.  Declare the neutral no-far-point-filter setting
        # explicitly so a staged real RGB-D run cannot inherit uninitialised
        # native memory.
        "System.thFarPoints: 0.0",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return depth_map_factor


def _run_checked(
    command: Sequence[str], *, timeout_seconds: float, label: str
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise ORBSLAM3Error(f"Could not start {label}: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise ORBSLAM3Error(
            f"{label} exceeded {timeout_seconds:.0f} seconds"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "no process output").strip()
        raise ORBSLAM3Error(f"{label} failed ({completed.returncode}): {detail[-1200:]}")
    return completed


def _resolve_wsl_path(config: ORBSLAM3Config, path: str) -> str:
    # ``readlink -f`` is used only for a user-configured Linux installation
    # path; shlex prevents a path containing whitespace or shell punctuation
    # from altering the command.
    if path == "~":
        expression = 'readlink -f -- "$HOME"'
    elif path.startswith("~/"):
        expression = 'readlink -f -- "$HOME"/' + shlex.quote(path[2:])
    else:
        expression = f"readlink -f -- {shlex.quote(path)}"
    result = _run_checked(
        [
            config.wsl_executable,
            "-e",
            "bash",
            "-lc",
            expression,
        ],
        timeout_seconds=20.0,
        label="WSL path resolution",
    )
    resolved = result.stdout.strip()
    if not resolved.startswith("/"):
        raise ORBSLAM3Error(f"WSL could not resolve ORB-SLAM3 path: {path}")
    return resolved


def _windows_path_to_wsl(config: ORBSLAM3Config, path: Path) -> str:
    result = _run_checked(
        [config.wsl_executable, "-e", "wslpath", "-a", str(path)],
        timeout_seconds=20.0,
        label="WSL path conversion",
    )
    resolved = result.stdout.strip()
    if not resolved.startswith("/"):
        raise ORBSLAM3Error(f"WSL could not convert staging path: {path}")
    return resolved


def _join_wsl_path(root: str, value: str) -> str:
    """Join a configured Linux-relative path without Windows ``Path`` rules."""

    if value.startswith("/"):
        return value
    return root.rstrip("/") + "/" + value.lstrip("/")


def _quaternion_to_rotation(
    qx: float, qy: float, qz: float, qw: float
) -> np.ndarray:
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ORBSLAM3Error("ORB-SLAM3 trajectory contains a zero quaternion")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _read_tum_trajectory(
    path: Path,
    frames: Sequence[RGBDFrame],
    timestamps: Sequence[float],
) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise ORBSLAM3Error(f"ORB-SLAM3 did not write CameraTrajectory.txt: {path}")
    rows: list[tuple[float, np.ndarray]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ORBSLAM3Error(
                f"ORB-SLAM3 trajectory line {line_number} has {len(fields)} fields, expected 8"
            )
        try:
            timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, fields)
        except ValueError as exc:
            raise ORBSLAM3Error(
                f"ORB-SLAM3 trajectory line {line_number} is not numeric"
            ) from exc
        values = np.asarray([timestamp, tx, ty, tz, qx, qy, qz, qw], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ORBSLAM3Error(
                f"ORB-SLAM3 trajectory line {line_number} is non-finite"
            )
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = _quaternion_to_rotation(qx, qy, qz, qw)
        # TUM trajectories use metres; the project contract uses millimetres.
        pose[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float64) * 1000.0
        rows.append((timestamp, pose))
    if not rows:
        raise ORBSLAM3Error("ORB-SLAM3 wrote an empty CameraTrajectory.txt")

    source_timestamps = np.asarray(timestamps, dtype=np.float64)
    used_indices: set[int] = set()
    poses: dict[int, np.ndarray] = {}
    # The TUM writer rounds timestamps to six decimals.  The capture timestamps
    # have microsecond precision, so 2 microseconds safely distinguishes frames
    # even for the fastest supported RGB-D stream.
    tolerance_seconds = 2.1e-6
    for timestamp, pose in rows:
        index = int(np.argmin(np.abs(source_timestamps - timestamp)))
        if abs(float(source_timestamps[index] - timestamp)) > tolerance_seconds:
            raise ORBSLAM3Error(
                "ORB-SLAM3 emitted a trajectory timestamp outside the staged sequence"
            )
        if index in used_indices:
            raise ORBSLAM3Error("ORB-SLAM3 emitted duplicate trajectory timestamps")
        used_indices.add(index)
        poses[frames[index].frame_id] = pose
    if not poses:
        raise ORBSLAM3Error("ORB-SLAM3 did not track any staged RGB-D frame")

    first_id = frames[0].frame_id
    if first_id not in poses:
        first_id = min(poses, key=lambda value: next(
            index for index, frame in enumerate(frames) if frame.frame_id == value
        ))
    rebase = np.linalg.inv(poses[first_id])
    return {frame_id: rebase @ pose for frame_id, pose in poses.items()}


def _stage_orbslam3_attempt(
    frames: Sequence[RGBDFrame],
    intrinsics: CameraIntrinsics,
    work_dir: Path,
    *,
    config: ORBSLAM3Config,
    executable_wsl: str,
    vocabulary_wsl: str,
) -> _StagedORBSLAM3Attempt:
    """Create a fresh, complete TUM input tree for one native execution.

    A native SIGSEGV must never be retried against a partially written sequence
    or a leftover trajectory.  Each attempt therefore receives a unique staging
    root and remaps every source RGB-D frame again before ORB-SLAM3 starts.
    """

    work_dir.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(
        tempfile.mkdtemp(prefix=".orbslam3_rgbd-attempt-", dir=str(work_dir))
    )
    sequence_dir = _stage_rgbd_sequence(frames, intrinsics, stage_dir)
    association_path = stage_dir / "association.txt"
    timestamps = tuple(_write_association(frames, association_path))
    settings_path = stage_dir / "gemini305_rgbd.yaml"
    _write_settings(frames, intrinsics, settings_path, config)
    trajectory_path = stage_dir / "CameraTrajectory.txt"

    stage_wsl = _windows_path_to_wsl(config, stage_dir)
    sequence_wsl = _windows_path_to_wsl(config, sequence_dir)
    association_wsl = _windows_path_to_wsl(config, association_path)
    settings_wsl = _windows_path_to_wsl(config, settings_path)
    command = (
        config.wsl_executable,
        "--cd",
        stage_wsl,
        "-e",
        "env",
        "PANGOLIN_WINDOW_URI=headless://",
        executable_wsl,
        vocabulary_wsl,
        settings_wsl,
        sequence_wsl,
        association_wsl,
    )
    return _StagedORBSLAM3Attempt(
        stage_dir=stage_dir,
        sequence_dir=sequence_dir,
        association_path=association_path,
        settings_path=settings_path,
        trajectory_path=trajectory_path,
        timestamps=timestamps,
        command=command,
    )


def _run_orbslam3_process(
    command: Sequence[str],
    *,
    stage_dir: Path,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], float]:
    """Launch one ORB-SLAM3 process without interpreting its result."""

    # Keep the private staging root explicit at this seam.  It makes it
    # impossible for a test or future launcher change to accidentally reuse a
    # previous attempt's trajectory while reporting a fresh command.
    del stage_dir
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        raise ORBSLAM3Error("Could not start wsl.exe for ORB-SLAM3") from exc
    except subprocess.TimeoutExpired as exc:
        raise ORBSLAM3Error(
            f"ORB-SLAM3 exceeded {timeout_seconds:.0f} seconds"
        ) from exc
    return completed, time.perf_counter() - started


def _attempt_audit_row(
    *,
    attempt_index: int,
    completed: subprocess.CompletedProcess[str],
    elapsed_seconds: float,
    accepted: bool,
    retry_reason: str | None,
) -> dict[str, object]:
    """Build a compact process audit without exposing temporary paths."""

    returncode = int(completed.returncode)
    return {
        "attempt_index": attempt_index,
        "returncode": returncode,
        "signal": _ORB_SLAM3_NATIVE_FAILURE_SIGNALS.get(returncode),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "accepted": accepted,
        "retry_reason": retry_reason,
    }


def run_orbslam3_rgbd(
    frames: Sequence[RGBDFrame],
    intrinsics: CameraIntrinsics,
    work_dir: str | Path,
    *,
    config: ORBSLAM3Config | Mapping[str, Any] | None = None,
) -> ORBSLAM3Trajectory:
    """Run ORB-SLAM3 RGB-D and return every genuinely tracked camera pose.

    The bridge never manufactures a pose for an untracked frame.  Its caller
    must either select only the returned frame ids or reject the incomplete
    trajectory according to ``minimum_tracked_fraction``.
    """

    selected_config = (
        config
        if isinstance(config, ORBSLAM3Config)
        else ORBSLAM3Config.from_mapping(config)
    )
    if len(frames) < 2:
        raise ORBSLAM3Error("ORB-SLAM3 RGB-D requires at least two frames")
    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise ORBSLAM3Error("ORB-SLAM3 RGB-D input contains duplicate frame ids")

    root_wsl = _resolve_wsl_path(selected_config, selected_config.root)
    executable_wsl = _resolve_wsl_path(
        selected_config, _join_wsl_path(root_wsl, selected_config.executable)
    )
    vocabulary_wsl = _resolve_wsl_path(
        selected_config, _join_wsl_path(root_wsl, selected_config.vocabulary)
    )
    for candidate, label in ((executable_wsl, "executable"), (vocabulary_wsl, "vocabulary")):
        _run_checked(
            [selected_config.wsl_executable, "-e", "test", "-f", candidate],
            timeout_seconds=20.0,
            label=f"ORB-SLAM3 {label} check",
        )

    work_root = Path(work_dir).expanduser().resolve()
    attempt_audit: list[dict[str, object]] = []
    for attempt_index in range(1, _ORB_SLAM3_MAX_EXECUTION_ATTEMPTS + 1):
        staged = _stage_orbslam3_attempt(
            frames,
            intrinsics,
            work_root,
            config=selected_config,
            executable_wsl=executable_wsl,
            vocabulary_wsl=vocabulary_wsl,
        )
        completed, elapsed_seconds = _run_orbslam3_process(
            staged.command,
            stage_dir=staged.stage_dir,
            timeout_seconds=selected_config.timeout_seconds,
        )
        stdout_path = staged.stage_dir / "orbslam3.stdout.txt"
        stderr_path = staged.stage_dir / "orbslam3.stderr.txt"
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")

        retry_reason = (
            _ORB_SLAM3_RETRYABLE_NATIVE_FAILURES.get(int(completed.returncode))
            if attempt_index < _ORB_SLAM3_MAX_EXECUTION_ATTEMPTS
            else None
        )
        attempt_audit.append(
            _attempt_audit_row(
                attempt_index=attempt_index,
                completed=completed,
                elapsed_seconds=elapsed_seconds,
                accepted=False,
                retry_reason=retry_reason,
            )
        )
        if completed.returncode != 0:
            if retry_reason is not None:
                continue
            detail = (completed.stderr or completed.stdout or "no process output").strip()
            raise ORBSLAM3Error(
                f"ORB-SLAM3 RGB-D failed ({completed.returncode}): {detail[-1200:]}",
                attempt_audit=attempt_audit,
            )

        # A normal process exit does not justify a retry.  Missing, duplicate,
        # or insufficient poses remain a structural failure of the sole real
        # trajectory attempt and must not be concealed by another run.
        try:
            poses = _read_tum_trajectory(
                staged.trajectory_path, frames, staged.timestamps
            )
            tracked_ids = tuple(
                frame.frame_id for frame in frames if frame.frame_id in poses
            )
            tracked_fraction = len(tracked_ids) / len(frames)
            if tracked_fraction < selected_config.minimum_tracked_fraction:
                raise ORBSLAM3Error(
                    "ORB-SLAM3 tracked only "
                    f"{len(tracked_ids)}/{len(frames)} frames "
                    f"({tracked_fraction:.1%}), below the required "
                    f"{selected_config.minimum_tracked_fraction:.1%}"
                )
        except ORBSLAM3Error as exc:
            raise ORBSLAM3Error(
                str(exc), attempt_audit=attempt_audit
            ) from exc

        attempt_audit[-1]["accepted"] = True
        return ORBSLAM3Trajectory(
            poses_by_frame_id=poses,
            tracked_frame_ids=tracked_ids,
            work_dir=staged.stage_dir,
            command=staged.command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            settings_path=staged.settings_path,
            association_path=staged.association_path,
            trajectory_path=staged.trajectory_path,
            config=selected_config,
            attempt_audit=tuple(attempt_audit),
        )

    raise AssertionError("ORB-SLAM3 retry loop ended without a result")
