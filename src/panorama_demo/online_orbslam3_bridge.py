"""Persistent stdin bridge for the Gemini 305 ORB-SLAM3 stream runner.

The native ``rgbd_g305_stream_headless`` executable deliberately exposes a
very small protocol:

``FRAME <id> <timestamp-seconds> <undistorted-colour> <undistorted-depth>``
``FINALIZE``

It does *not* publish provisional poses.  ORB-SLAM3 local mapping and loop
closure may revise an earlier pose, so this module accepts a trajectory only
after ``FINALIZE`` has caused the native process to exit successfully and its
final TUM file can be parsed.  Callers must stage calibrated, undistorted RGB
and aligned uint16 depth PNGs before submitting their WSL paths.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Mapping, TextIO

from .orbslam3_bridge import (
    ORBSLAM3Config,
    ORBSLAM3Error,
    ORBSLAM3Trajectory,
    _join_wsl_path,
    _native_process_detail,
    _read_tum_trajectory,
    _resolve_wsl_path,
    _run_checked,
    _windows_path_to_wsl,
)
from .session import CameraIntrinsics, RGBDFrame


_READY = "G305_STREAM_READY"
_ACCEPTED_PREFIX = "G305_STREAM_ACCEPTED "
_FINALIZED_PREFIX = "G305_STREAM_FINALIZED "


@dataclass(frozen=True)
class OnlineORBSLAM3Launch:
    """Resolved paths and command for one private persistent runner."""

    work_dir: Path
    settings_path: Path
    trajectory_path: Path
    protocol_path: Path
    command: tuple[str, ...]
    config: ORBSLAM3Config


def _write_stream_settings(
    *,
    intrinsics: CameraIntrinsics,
    depth_scale_mm_per_unit: float,
    tracking_fps: int,
    path: Path,
    config: ORBSLAM3Config,
) -> None:
    """Write the same calibrated pinhole settings as the batch bridge.

    Unlike a batch association, a persistent runner must be ready before all
    frames exist.  Its depth unit and tracking cadence are capture properties,
    so they are supplied explicitly rather than inferred from a frame list.
    """

    if not math.isfinite(depth_scale_mm_per_unit) or depth_scale_mm_per_unit <= 0.0:
        raise ORBSLAM3Error("Online ORB-SLAM3 input has an invalid depth scale")
    if not isinstance(tracking_fps, int) or not 1 <= tracking_fps <= 120:
        raise ValueError("online ORB-SLAM3 tracking_fps must be an integer in [1, 120]")
    depth_map_factor = 1000.0 / float(depth_scale_mm_per_unit)
    lines = [
        "%YAML:1.0",
        'File.version: "1.0"',
        'Camera.type: "PinHole"',
        f"Camera1.fx: {intrinsics.fx:.12g}",
        f"Camera1.fy: {intrinsics.fy:.12g}",
        f"Camera1.cx: {intrinsics.cx:.12g}",
        f"Camera1.cy: {intrinsics.cy:.12g}",
        # Submitted images have already undergone the one calibrated inverse
        # remap required by the project contract.
        "Camera1.k1: 0.0",
        "Camera1.k2: 0.0",
        "Camera1.p1: 0.0",
        "Camera1.p2: 0.0",
        "Camera1.k3: 0.0",
        f"Camera.width: {intrinsics.width}",
        f"Camera.height: {intrinsics.height}",
        f"Camera.fps: {tracking_fps}",
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
        "Viewer.CameraLineWidth: 1.0",
        "Viewer.ViewpointX: 0.0",
        "Viewer.ViewpointY: -0.7",
        "Viewer.ViewpointZ: -1.8",
        "Viewer.ViewpointF: 500.0",
        "System.thFarPoints: 0.0",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_online_orbslam3_runner(
    *,
    intrinsics: CameraIntrinsics,
    depth_scale_mm_per_unit: float,
    tracking_fps: int,
    work_dir: str | Path,
    config: ORBSLAM3Config | Mapping[str, Any] | None = None,
) -> OnlineORBSLAM3Launch:
    """Create private runner files and resolve the persistent WSL command.

    This performs no image decode/remap and does not start ORB-SLAM3.  The
    capture owner stages each accepted source frame and submits it afterwards.
    """

    selected_config = (
        config if isinstance(config, ORBSLAM3Config) else ORBSLAM3Config.from_mapping(config)
    )
    root_wsl = _resolve_wsl_path(selected_config, selected_config.root)
    executable_wsl = _resolve_wsl_path(
        selected_config, _join_wsl_path(root_wsl, selected_config.stream_executable)
    )
    vocabulary_wsl = _resolve_wsl_path(
        selected_config, _join_wsl_path(root_wsl, selected_config.vocabulary)
    )
    for candidate, label in ((executable_wsl, "stream executable"), (vocabulary_wsl, "vocabulary")):
        _run_checked(
            [selected_config.wsl_executable, "-e", "test", "-f", candidate],
            timeout_seconds=20.0,
            label=f"ORB-SLAM3 {label} check",
        )

    root = Path(work_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    private_dir = Path(tempfile.mkdtemp(prefix=".online_orbslam3_rgbd-", dir=str(root)))
    settings_path = private_dir / "gemini305_rgbd.yaml"
    _write_stream_settings(
        intrinsics=intrinsics,
        depth_scale_mm_per_unit=depth_scale_mm_per_unit,
        tracking_fps=tracking_fps,
        path=settings_path,
        config=selected_config,
    )
    trajectory_path = private_dir / "CameraTrajectory.txt"
    protocol_path = private_dir / "stream_protocol.txt"
    private_wsl = _windows_path_to_wsl(selected_config, private_dir)
    settings_wsl = _windows_path_to_wsl(selected_config, settings_path)
    trajectory_wsl = _windows_path_to_wsl(selected_config, trajectory_path)
    command = (
        selected_config.wsl_executable,
        "--cd",
        private_wsl,
        "-e",
        executable_wsl,
        vocabulary_wsl,
        settings_wsl,
        trajectory_wsl,
    )
    return OnlineORBSLAM3Launch(
        work_dir=private_dir,
        settings_path=settings_path,
        trajectory_path=trajectory_path,
        protocol_path=protocol_path,
        command=command,
        config=selected_config,
    )


def _readline_with_timeout(stream: TextIO, *, timeout_seconds: float, label: str) -> str:
    """Read one protocol response without allowing a native startup hang."""

    result: Queue[str] = Queue(maxsize=1)

    def read() -> None:
        result.put(stream.readline())

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        return result.get(timeout=timeout_seconds)
    except Empty as exc:
        raise ORBSLAM3Error(f"ORB-SLAM3 stream timed out waiting for {label}") from exc


def _validate_wsl_staged_path(path: str, *, label: str) -> str:
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"online ORB-SLAM3 {label} path must be an absolute WSL path")
    if any(character.isspace() or ord(character) < 32 for character in path):
        raise ValueError(f"online ORB-SLAM3 {label} path cannot contain whitespace")
    return path


class PersistentORBSLAM3Runner:
    """One persistent native ORB-SLAM3 process with a fail-closed protocol.

    ``submit`` accepts a real capture frame plus paths to its already staged
    calibrated RGB-D inputs.  The frame itself is retained only for final TUM
    timestamp mapping; no provisional pose is exposed.
    """

    def __init__(self, launch: OnlineORBSLAM3Launch) -> None:
        self.launch = launch
        self._process: subprocess.Popen[str] | None = None
        self._frames: list[RGBDFrame] = []
        self._timestamps: list[float] = []
        self._protocol_lines: list[str] = []
        self._stdout_lines: list[str] = []
        self._state = "new"
        self._started_at: float | None = None

    @property
    def started(self) -> bool:
        return self._state in {"running", "finalizing", "finalized"}

    @property
    def finalized(self) -> bool:
        return self._state == "finalized"

    @property
    def submitted_frame_ids(self) -> tuple[int, ...]:
        return tuple(frame.frame_id for frame in self._frames)

    def start(self) -> None:
        if self._state != "new":
            raise RuntimeError("online ORB-SLAM3 runner has already been started")
        try:
            process = subprocess.Popen(
                list(self.launch.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=os.environ.copy(),
            )
        except OSError as exc:
            self._state = "failed"
            raise ORBSLAM3Error("Could not start WSL ORB-SLAM3 stream runner") from exc
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            self._state = "failed"
            raise ORBSLAM3Error("ORB-SLAM3 stream runner lacks standard I/O pipes")
        self._process = process
        self._started_at = time.perf_counter()
        try:
            ready = self._read_protocol_line(_READY, label="G305_STREAM_READY")
        except Exception:
            self.abort()
            raise
        if ready != _READY:
            self.abort()
            raise ORBSLAM3Error(
                "ORB-SLAM3 stream runner did not acknowledge startup: " + (ready or "EOF")
            )
        self._state = "running"

    def _write_command(self, line: str) -> None:
        if self._process is None or self._process.stdin is None:
            raise ORBSLAM3Error("ORB-SLAM3 stream runner has no writable stdin")
        try:
            self._process.stdin.write(line + "\n")
            self._process.stdin.flush()
        except OSError as exc:
            self._state = "failed"
            raise ORBSLAM3Error("Could not submit command to ORB-SLAM3 stream runner") from exc
        self._protocol_lines.append(line)

    def _read_protocol_line(self, expected: str, *, label: str) -> str:
        """Ignore normal ORB startup logs until the explicit protocol token."""

        assert self._process is not None and self._process.stdout is not None
        deadline = time.monotonic() + self.launch.config.timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise ORBSLAM3Error(f"ORB-SLAM3 stream timed out waiting for {label}")
            raw = _readline_with_timeout(
                self._process.stdout, timeout_seconds=remaining, label=label
            )
            response = raw.strip()
            self._stdout_lines.append(response)
            if response == expected:
                return response
            # ``ORB_SLAM3::System`` prints banner separators as blank lines;
            # an empty ``readline`` is the only actual pipe EOF.
            if raw == "":
                raise ORBSLAM3Error(
                    f"ORB-SLAM3 stream ended before {label}; last output: "
                    + (self._stdout_lines[-2] if len(self._stdout_lines) > 1 else "EOF")
                )

    def submit(
        self,
        frame: RGBDFrame,
        *,
        staged_color_path_wsl: str,
        staged_depth_path_wsl: str,
    ) -> None:
        """Submit one chronological, real frame to the native tracker."""

        if self._state != "running":
            raise RuntimeError("online ORB-SLAM3 runner is not accepting frames")
        if frame.timestamp_us is None or frame.timestamp_us < 0:
            raise ORBSLAM3Error(f"Frame {frame.frame_id} lacks a valid colour timestamp")
        frame_id = int(frame.frame_id)
        timestamp = float(frame.timestamp_us) / 1_000_000.0
        if not math.isfinite(timestamp):
            raise ORBSLAM3Error(f"Frame {frame.frame_id} timestamp is non-finite")
        if self._frames and (
            frame_id <= self._frames[-1].frame_id or timestamp <= self._timestamps[-1]
        ):
            raise ORBSLAM3Error("Online ORB-SLAM3 FRAME commands must be strictly monotonic")
        color_path = _validate_wsl_staged_path(staged_color_path_wsl, label="colour")
        depth_path = _validate_wsl_staged_path(staged_depth_path_wsl, label="depth")
        line = f"FRAME {frame_id} {timestamp:.6f} {color_path} {depth_path}"
        self._write_command(line)
        expected = f"{_ACCEPTED_PREFIX}{frame_id}"
        try:
            self._read_protocol_line(expected, label=f"FRAME {frame_id} acknowledgement")
        except Exception:
            self._state = "failed"
            raise
        self._frames.append(frame)
        self._timestamps.append(timestamp)

    def _attempt_audit(self, *, returncode: int, accepted: bool) -> tuple[dict[str, object], ...]:
        elapsed = 0.0 if self._started_at is None else time.perf_counter() - self._started_at
        return (
            {
                "attempt_index": 1,
                "returncode": int(returncode),
                "signal": None,
                "elapsed_seconds": round(float(elapsed), 3),
                "accepted": accepted,
                "retry_reason": None,
                "protocol": "persistent_stdin_finalize_only",
            },
        )

    def finalize(self) -> ORBSLAM3Trajectory:
        """Shutdown, wait for exit, then parse the only authoritative TUM file."""

        if self._state != "running":
            raise RuntimeError("online ORB-SLAM3 runner cannot be finalized in its current state")
        if len(self._frames) < 2:
            raise ORBSLAM3Error("Online ORB-SLAM3 requires at least two submitted frames")
        self._state = "finalizing"
        self._write_command("FINALIZE")
        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        try:
            self._process.stdin.close()
            expected = f"{_FINALIZED_PREFIX}{self._frames[-1].frame_id}"
            self._read_protocol_line(expected, label="FINALIZE acknowledgement")
            returncode = self._process.wait(timeout=self.launch.config.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self.abort()
            raise ORBSLAM3Error(
                f"ORB-SLAM3 stream exceeded {self.launch.config.timeout_seconds:.0f} seconds"
            ) from exc
        except Exception:
            self._state = "failed"
            raise

        remaining_stdout = self._process.stdout.read()
        stderr = self._process.stderr.read() if self._process.stderr is not None else ""
        stdout = "\n".join((*self._stdout_lines, remaining_stdout.rstrip("\n"))).rstrip("\n")
        stdout_path = self.launch.work_dir / "orbslam3.stream.stdout.txt"
        stderr_path = self.launch.work_dir / "orbslam3.stream.stderr.txt"
        self.launch.protocol_path.write_text("\n".join(self._protocol_lines) + "\n", encoding="utf-8")
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        if returncode != 0:
            completed = subprocess.CompletedProcess(
                list(self.launch.command), returncode, stdout=stdout, stderr=stderr
            )
            self._state = "failed"
            raise ORBSLAM3Error(
                f"ORB-SLAM3 stream failed ({returncode}): {_native_process_detail(completed)[-2400:]}",
                attempt_audit=self._attempt_audit(returncode=returncode, accepted=False),
            )

        # This is intentionally after ``wait``.  A TUM file observed before
        # native Shutdown has no formal meaning and must never escape the bridge.
        try:
            poses = _read_tum_trajectory(
                self.launch.trajectory_path, self._frames, self._timestamps
            )
            tracked_ids = tuple(frame.frame_id for frame in self._frames if frame.frame_id in poses)
            tracked_fraction = len(tracked_ids) / len(self._frames)
            if tracked_fraction < self.launch.config.minimum_tracked_fraction:
                raise ORBSLAM3Error(
                    "ORB-SLAM3 stream tracked only "
                    f"{len(tracked_ids)}/{len(self._frames)} frames ({tracked_fraction:.1%}), "
                    f"below the required {self.launch.config.minimum_tracked_fraction:.1%}"
                )
        except ORBSLAM3Error as exc:
            self._state = "failed"
            raise ORBSLAM3Error(
                str(exc), attempt_audit=self._attempt_audit(returncode=returncode, accepted=False)
            ) from exc
        self._state = "finalized"
        return ORBSLAM3Trajectory(
            poses_by_frame_id=poses,
            tracked_frame_ids=tracked_ids,
            work_dir=self.launch.work_dir,
            command=self.launch.command,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            settings_path=self.launch.settings_path,
            association_path=self.launch.protocol_path,
            trajectory_path=self.launch.trajectory_path,
            config=self.launch.config,
            attempt_audit=self._attempt_audit(returncode=returncode, accepted=True),
        )

    def abort(self) -> None:
        """Best-effort stop for an unfinalized native process; no pose is kept."""

        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if self._state != "finalized":
            self._state = "failed"


def start_online_orbslam3_runner(**kwargs: Any) -> PersistentORBSLAM3Runner:
    """Prepare and start a stream runner in one explicit operation."""

    runner = PersistentORBSLAM3Runner(prepare_online_orbslam3_runner(**kwargs))
    runner.start()
    return runner
