"""Asynchronous capture-time staging for the persistent ORB-SLAM3 runner.

Only writer-committed RGB-D files enter this worker.  It never invents a
frame or a pose: 60 FPS sources may be analysed for motion, while this worker
tracks a deterministic real, timestamp-spaced subset for the video product.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cuda_backend import remap as accelerated_remap
from .online_orbslam3_bridge import start_online_orbslam3_runner
from .orbslam3_bridge import ORBSLAM3Config, ORBSLAM3Error, _undistortion_maps, _windows_path_to_wsl
from .session import CameraIntrinsics, RGBDFrame


@dataclass(frozen=True)
class OnlineORBSource:
    frame: RGBDFrame
    color_sha256: str
    aligned_depth_sha256: str


class OnlineORBTracker:
    """Bounded worker which overlaps remap/tracking with continuous capture."""

    def __init__(
        self,
        *,
        intrinsics: CameraIntrinsics,
        tracking_fps: float,
        work_dir: Path,
        config: dict[str, Any] | None,
        queue_size: int = 256,
    ) -> None:
        if not np.isfinite(tracking_fps) or tracking_fps <= 0.0:
            raise ValueError("online ORB tracking FPS must be finite and positive")
        self.intrinsics = intrinsics
        self.tracking_fps = float(tracking_fps)
        self.work_dir = work_dir
        self.config = ORBSLAM3Config.from_mapping(config)
        self._queue: queue.Queue[OnlineORBSource | None] = queue.Queue(maxsize=queue_size)
        self._thread = threading.Thread(target=self._run, name="online-orbslam3", daemon=False)
        self._lock = threading.Lock()
        self._error: str | None = None
        self._result: dict[str, Any] | None = None
        self._last_committed: OnlineORBSource | None = None
        self._last_scheduled_timestamp: int | None = None
        self._closed = False
        self._thread.start()

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    def submit_committed(self, source: OnlineORBSource) -> None:
        """Called by the disk writer after its atomic RGB and depth writes.

        Never raises into the writer.  A queue overflow makes the online
        trajectory unavailable rather than silently dropping a real source.
        """

        with self._lock:
            if self._closed or self._error is not None:
                return
            self._last_committed = source
            timestamp = source.frame.timestamp_us
            if timestamp is None:
                self._error = "online ORB source lacks a timestamp"
                return
            if (
                self._last_scheduled_timestamp is not None
                and timestamp - self._last_scheduled_timestamp
                < 1_000_000.0 / self.tracking_fps
            ):
                return
            self._last_scheduled_timestamp = timestamp
        try:
            self._queue.put_nowait(source)
        except queue.Full:
            with self._lock:
                self._error = "online ORB queue overflow; no partial trajectory is publishable"

    def close(self) -> dict[str, Any] | None:
        with self._lock:
            if self._closed:
                return self._result
            self._closed = True
            endpoint = self._last_committed
            endpoint_already_scheduled = (
                endpoint is not None
                and self._last_scheduled_timestamp == endpoint.frame.timestamp_us
            )
        # A startup/native failure can end the worker while the writer is
        # still enqueueing.  Do not block shutdown trying to append a sentinel
        # behind an abandoned full queue.
        if self._thread.is_alive():
            if endpoint is not None and not endpoint_already_scheduled:
                self._queue.put(endpoint)
            self._queue.put(None)
        self._thread.join()
        with self._lock:
            return self._result

    def _set_error(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _decode(path: Path, flags: int, label: str) -> np.ndarray:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, flags) if data.size else None
        if image is None:
            raise ORBSLAM3Error(f"Could not decode committed {label}: {path}")
        return image

    @staticmethod
    def _write_png(path: Path, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise ORBSLAM3Error(f"Could not encode online ORB staging image: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded.tobytes())

    def _stage_and_submit(self, runner: Any, source: OnlineORBSource, maps: tuple[np.ndarray, np.ndarray] | None) -> None:
        color = self._decode(source.frame.color_path, cv2.IMREAD_COLOR, "colour")
        depth = self._decode(source.frame.aligned_depth_path, cv2.IMREAD_UNCHANGED, "aligned depth")
        if color.shape != (self.intrinsics.height, self.intrinsics.width, 3):
            raise ORBSLAM3Error(f"Online ORB frame {source.frame.frame_id} colour dimensions differ from calibration")
        if depth.dtype != np.uint16 or depth.shape != color.shape[:2] or not np.any(depth > 0):
            raise ORBSLAM3Error(f"Online ORB frame {source.frame.frame_id} depth is not valid aligned uint16")
        if maps is not None:
            color = accelerated_remap(color, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            depth = accelerated_remap(depth, maps[0], maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        stem = f"{source.frame.frame_id:08d}.png"
        color_path = runner.launch.work_dir / "sequence" / "color" / stem
        depth_path = runner.launch.work_dir / "sequence" / "depth" / stem
        self._write_png(color_path, color)
        self._write_png(depth_path, depth)
        runner.submit(
            source.frame,
            staged_color_path_wsl=_windows_path_to_wsl(self.config, color_path),
            staged_depth_path_wsl=_windows_path_to_wsl(self.config, depth_path),
        )

    def _run(self) -> None:
        runner = None
        selected: list[OnlineORBSource] = []
        try:
            first = self._queue.get()
            if first is None:
                return
            runner = start_online_orbslam3_runner(
                intrinsics=self.intrinsics,
                depth_scale_mm_per_unit=first.frame.depth_scale_mm_per_unit,
                tracking_fps=max(1, int(round(self.tracking_fps))),
                work_dir=self.work_dir,
                config=self.config,
            )
            maps = _undistortion_maps(self.intrinsics)
            source: OnlineORBSource | None = first
            while source is not None:
                self._stage_and_submit(runner, source, maps)
                selected.append(source)
                source = self._queue.get()
            trajectory = runner.finalize()
            selected_ids = [item.frame.frame_id for item in selected]
            if list(trajectory.tracked_frame_ids) != selected_ids:
                raise ORBSLAM3Error("Online ORB final trajectory does not cover every submitted real frame")
            with self._lock:
                self._result = {
                    "schema": "gemini305-online-orbslam3-trajectory/v1",
                    "capture_origin": "writer_committed_files",
                    "tracking_fps": self.tracking_fps,
                    "tracked_frame_ids": selected_ids,
                    "camera_to_world": [trajectory.poses_by_frame_id[frame_id].tolist() for frame_id in selected_ids],
                    "attempts": [dict(row) for row in trajectory.attempt_audit],
                    "pose_convention": "camera_to_world",
                    "translation_unit": "mm",
                    "source_file_sha256": [
                        {"frame_id": item.frame.frame_id, "color_sha256": item.color_sha256, "aligned_depth_sha256": item.aligned_depth_sha256}
                        for item in selected
                    ],
                }
        except BaseException as exc:
            self._set_error(exc)
            if runner is not None:
                runner.abort()
