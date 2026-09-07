from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import cv2
import numpy as np

from .config import load_config
from .session import CameraIntrinsics, RGBDFrame
from .video_online_state import (
    CAPTURE_FRAME_VALIDATION_SCHEMA,
    OnlineScanAccumulator,
    write_online_state,
)


COLOR_EXPOSURE_UNIT_US = 100
MAX_FORMAL_COLOR_EXPOSURE_US = 800
AUTO_EXPOSURE_STARTUP_TRANSITION_MAX_FRAME_SETS = 8
DEFAULT_TRIGGER_TO_IMAGE_DELAY_US = 8_000
DEFAULT_TRIGGER_OUT_DELAY_US = 7_000
GEMINI305_COLOR_FORMAT_PRIORITY = ("RGB", "BGR", "YUYV", "MJPG")
GEMINI305_FRAME_RATES = (5, 10, 15, 20, 30, 60)


class CaptureCancelledError(RuntimeError):
    """A caller cancelled camera discovery or an active continuous capture."""


class _VideoCapturePreflightError(RuntimeError):
    """A video stream failed before any formal frame could be accepted."""


class _VideoCameraControlUnavailableError(_VideoCapturePreflightError):
    """The camera stopped answering control requests before streaming began."""


class _VideoWarmupNoFramesError(_VideoCapturePreflightError):
    """A started video pipeline yielded no complete warmup frames."""


def _is_retryable_video_transport_error(exc: BaseException) -> bool:
    """Identify observed USB/IP camera transport failures, including causes."""

    current: BaseException | None = exc
    seen: set[int] = set()
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        messages.append(str(current).lower())
        current = current.__cause__ or current.__context__
    detail = " ".join(messages)
    return any(
        marker in detail
        for marker in (
            "openusbdevice failed",
            "device is deactivated/disconnected",
            "setxu failed",
            "device response size(0)",
            "connection reset by peer",
        )
    )


CSV_FIELDS = [
    "frame_id",
    "color_index",
    "depth_index",
    "color_device_timestamp_us",
    "depth_device_timestamp_us",
    "sync_delta_us",
    "color_system_timestamp_us",
    "depth_system_timestamp_us",
    "host_timestamp_ns",
    "color_exposure",
    "depth_exposure",
    "color_gain",
    "depth_gain",
    "color_auto_white_balance",
    "color_white_balance",
    "color_frame_number",
    "depth_frame_number",
    "color_sensor_timestamp_raw",
    "depth_sensor_timestamp_raw",
    "depth_scale_mm_per_unit",
    "color_path",
    "aligned_depth_path",
    "raw_depth_path",
    "queue_depth",
]


@dataclass
class FramePacket:
    frame_id: int
    color_bgr: np.ndarray
    aligned_depth: np.ndarray
    raw_depth: np.ndarray | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LiveSessionInfo:
    root: Path
    capture_started_monotonic_ns: int
    calibration: Mapping[str, object]


@dataclass(frozen=True)
class LiveFramePacket:
    frame_id: int
    color_bgr: np.ndarray
    aligned_depth_mm: np.ndarray
    metadata: Mapping[str, object]
    accepted_monotonic_ns: int
    writer_queue_fraction: float
    writer_queue_drops: int


@dataclass(frozen=True)
class CaptureResult:
    session_root: Path
    received_frames: int
    written_frames: int
    queue_drops: int
    write_errors: int
    max_queue_depth: int
    capture_started_monotonic_ns: int
    capture_stopped_monotonic_ns: int


class CaptureObserver(Protocol):
    def on_session_ready(self, session: LiveSessionInfo) -> None: ...

    def on_frame_accepted(self, packet: LiveFramePacket) -> None: ...

    def on_frame_committed(self, frame: "WrittenRGBDFrame") -> None: ...

    def on_capture_stopping(self) -> None: ...

    def on_capture_closed(self, result: CaptureResult) -> None: ...


@dataclass
class WriterStats:
    submitted: int = 0
    written: int = 0
    queue_drops: int = 0
    write_errors: int = 0
    max_queue_depth: int = 0
    errors: list[str] = field(default_factory=list)


class SessionWriterError(RuntimeError):
    """Raised in the capture thread after the disk writer has failed."""


def _atomic_encode(
    path: Path,
    extension: str,
    image: np.ndarray,
    params: list[int],
    *,
    durable: bool = True,
) -> str:
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise IOError(f"OpenCV could not encode {path}")
    payload = encoded.tobytes()
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _parse_color_intrinsics_for_online_orb(calibration: dict[str, Any]) -> CameraIntrinsics:
    """Read the just-written capture calibration without trusting a later reload."""

    intrinsic = calibration.get("color_intrinsic")
    distortion = calibration.get("color_distortion")
    if not isinstance(intrinsic, dict) or not isinstance(distortion, dict):
        raise ValueError("capture calibration lacks color intrinsic/distortion")
    keys = ("width", "height", "fx", "fy", "cx", "cy")
    if any(key not in intrinsic for key in keys):
        raise ValueError("capture color intrinsic is incomplete")
    values = {key: float(intrinsic[key]) for key in keys}
    if not np.isfinite(list(values.values())).all() or values["width"] <= 0 or values["height"] <= 0:
        raise ValueError("capture color intrinsic is non-finite")
    coefficients = tuple(float(distortion.get(key, 0.0)) for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"))
    if not np.isfinite(coefficients).all():
        raise ValueError("capture color distortion is non-finite")
    return CameraIntrinsics(
        int(values["width"]), int(values["height"]), values["fx"], values["fy"],
        values["cx"], values["cy"], coefficients,
    )


@dataclass(frozen=True)
class WrittenRGBDFrame:
    """Exact source-file digests produced by the single writer thread."""

    frame_id: int
    color_path: Path
    aligned_depth_path: Path
    color_sha256: str
    aligned_depth_sha256: str
    timestamp_us: int
    depth_scale_mm_per_unit: float
    frames_csv_row_index: int
    commit_monotonic_ns: int


class SessionWriter:
    def __init__(
        self,
        root: Path,
        queue_size: int,
        jpeg_quality: int,
        depth_png_compression: int,
        save_raw_depth: bool,
        durable_per_frame: bool = True,
        csv_flush_interval: int = 1,
        on_written: Callable[[WrittenRGBDFrame], None] | None = None,
    ) -> None:
        self.root = root
        self.color_dir = root / "color"
        self.aligned_depth_dir = root / "depth_aligned"
        self.raw_depth_dir = root / "depth_raw"
        for directory in (self.color_dir, self.aligned_depth_dir):
            directory.mkdir(parents=True, exist_ok=True)
        if save_raw_depth:
            self.raw_depth_dir.mkdir(parents=True, exist_ok=True)
        self.save_raw_depth = save_raw_depth
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self.depth_png_compression = int(np.clip(depth_png_compression, 0, 9))
        self.durable_per_frame = bool(durable_per_frame)
        self.csv_flush_interval = int(csv_flush_interval)
        self.on_written = on_written
        if self.csv_flush_interval < 1:
            raise ValueError("csv_flush_interval must be positive")
        self.queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=queue_size)
        self.stats = WriterStats()
        self.written_rgbd_frames: list[WrittenRGBDFrame] = []
        self._failed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rgbd-writer", daemon=False)
        self._thread.start()

    def submit(self, packet: FramePacket) -> bool:
        self.raise_if_failed()
        self.stats.submitted += 1
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.stats.queue_drops += 1
            return False
        self.stats.max_queue_depth = max(self.stats.max_queue_depth, self.queue.qsize())
        return True

    def close(self) -> None:
        if self._thread.is_alive():
            self.queue.put(None)
        self._thread.join()

    def raise_if_failed(self) -> None:
        if self._failed.is_set():
            detail = self.stats.errors[0] if self.stats.errors else "unknown writer failure"
            raise SessionWriterError(f"RGB-D session writer failed: {detail}")

    def _record_failure(self, message: str) -> None:
        if self._failed.is_set():
            return
        self.stats.write_errors += 1
        self.stats.errors.append(message)
        self._failed.set()
        print(f"\nWriter error: {message}", file=sys.stderr)

    @staticmethod
    def _discard_incomplete_frame(paths: tuple[Path, ...]) -> None:
        for path in paths:
            for candidate in (path, path.with_suffix(path.suffix + ".partial")):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    pass

    def _run(self) -> None:
        csv_path = self.root / "frames.csv"
        try:
            handle = csv_path.open("w", encoding="utf-8", newline="")
        except Exception as exc:
            self._record_failure(f"frames.csv initialization: {exc}")
            return
        with handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            try:
                writer.writeheader()
            except Exception as exc:
                self._record_failure(f"frames.csv initialization: {exc}")
                return
            while True:
                packet = self.queue.get()
                if packet is None:
                    self.queue.task_done()
                    break
                if self._failed.is_set():
                    self.queue.task_done()
                    continue
                try:
                    stem = f"{packet.frame_id:08d}"
                    color_relative = Path("color") / f"{stem}.jpg"
                    aligned_relative = Path("depth_aligned") / f"{stem}.png"
                    raw_relative = Path("depth_raw") / f"{stem}.png"
                    frame_paths = (
                        self.root / color_relative,
                        self.root / aligned_relative,
                        self.root / raw_relative,
                    )
                    color_sha256 = _atomic_encode(
                        self.root / color_relative,
                        ".jpg",
                        packet.color_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                        durable=self.durable_per_frame,
                    )
                    aligned_depth_sha256 = _atomic_encode(
                        self.root / aligned_relative,
                        ".png",
                        packet.aligned_depth,
                        [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression],
                        durable=self.durable_per_frame,
                    )
                    raw_value = ""
                    if self.save_raw_depth and packet.raw_depth is not None:
                        _atomic_encode(
                            self.root / raw_relative,
                            ".png",
                            packet.raw_depth,
                            [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression],
                            durable=self.durable_per_frame,
                        )
                        raw_value = raw_relative.as_posix()
                    row = {key: packet.metadata.get(key, "") for key in CSV_FIELDS}
                    row.update(
                        {
                            "frame_id": packet.frame_id,
                            "color_path": color_relative.as_posix(),
                            "aligned_depth_path": aligned_relative.as_posix(),
                            "raw_depth_path": raw_value,
                            "queue_depth": self.queue.qsize(),
                        }
                    )
                    writer.writerow(row)
                    if (self.stats.written + 1) % self.csv_flush_interval == 0:
                        handle.flush()
                    written_frame = WrittenRGBDFrame(
                            frame_id=packet.frame_id,
                            color_path=self.root / color_relative,
                            aligned_depth_path=self.root / aligned_relative,
                            color_sha256=color_sha256,
                            aligned_depth_sha256=aligned_depth_sha256,
                            timestamp_us=int(packet.metadata.get("color_device_timestamp_us", packet.frame_id)),
                            depth_scale_mm_per_unit=float(packet.metadata.get("depth_scale_mm_per_unit", 1.0)),
                            frames_csv_row_index=self.stats.written,
                            commit_monotonic_ns=time.monotonic_ns(),
                        )
                    self.written_rgbd_frames.append(written_frame)
                    # Keep expensive downstream processing off this writer
                    # thread: the callback is limited to a bounded enqueue of
                    # immutable, already committed source-file facts.
                    if self.on_written is not None:
                        self.on_written(written_frame)
                    self.stats.written += 1
                except Exception as exc:
                    self._discard_incomplete_frame(frame_paths)
                    self._record_failure(f"frame {packet.frame_id}: {exc}")
                finally:
                    self.queue.task_done()
            try:
                handle.flush()
                if self.durable_per_frame:
                    os.fsync(handle.fileno())
            except Exception as exc:
                self._record_failure(f"frames.csv finalization: {exc}")


def _frame_to_bgr(frame: Any, sdk: Any) -> np.ndarray:
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    format_name = getattr(fmt, "name", str(fmt)).upper()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if format_name == "RGB":
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if format_name == "BGR":
        return data.reshape(height, width, 3).copy()
    if format_name == "RGBA":
        return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_RGBA2BGR)
    if format_name == "BGRA":
        return cv2.cvtColor(data.reshape(height, width, 4), cv2.COLOR_BGRA2BGR)
    if format_name == "Y8":
        return cv2.cvtColor(data.reshape(height, width), cv2.COLOR_GRAY2BGR)
    if format_name == "Y16":
        gray16 = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
            height, width
        )
        gray8 = np.right_shift(gray16, 8).astype(np.uint8)
        return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    if format_name == "YUYV":
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if format_name == "UYVY":
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
    if format_name == "MJPG":
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode MJPG color frame")
        return image
    if format_name == "NV12":
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
    if format_name == "NV21":
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21)
    if format_name == "I420":
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_I420)
    raise ValueError(f"Unsupported color format: {fmt}")


def _depth_array(frame: Any) -> np.ndarray:
    if frame is None:
        raise ValueError("Missing depth frame")
    return np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(
        frame.get_height(), frame.get_width()
    ).copy()


def _enum_name(value: Any) -> str:
    return getattr(value, "name", str(value))


def _profile_dict(profile: Any) -> dict[str, Any]:
    return {
        "width": int(profile.get_width()),
        "height": int(profile.get_height()),
        "fps": int(profile.get_fps()),
        "format": _enum_name(profile.get_format()),
    }


def _available_profiles(profile_list: Any) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for index in range(profile_list.get_count()):
        profile = profile_list.get_stream_profile_by_index(index)
        if all(hasattr(profile, method) for method in ("get_width", "get_height", "get_fps", "get_format")):
            profiles.append(_profile_dict(profile))
    return profiles


def _choose_profile(
    profile_list: Any,
    width: int,
    height: int,
    fps: int,
    formats: list[Any],
    label: str,
) -> Any:
    for fmt in formats:
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            continue
    available = _available_profiles(profile_list)
    raise RuntimeError(
        f"No exact {label} profile for {width}x{height}@{fps}. Available profiles: {available}"
    )


def _metadata(frame: Any, metadata_type: Any) -> int | None:
    try:
        if frame is not None and frame.has_metadata(metadata_type):
            return int(frame.get_metadata_value(metadata_type))
    except Exception:
        pass
    return None


def _metadata_by_name(frame: Any, metadata_types: Any, name: str) -> int | None:
    """Return optional frame metadata without assuming every SDK enum is present."""

    try:
        metadata_type = getattr(metadata_types, name)
    except AttributeError:
        return None
    return _metadata(frame, metadata_type)


def _color_control_metadata(frame: Any, metadata_types: Any) -> dict[str, int | None]:
    """Read the raw color-control metadata used to audit a locked capture."""

    return {
        "color_exposure": _metadata_by_name(frame, metadata_types, "EXPOSURE"),
        "color_gain": _metadata_by_name(frame, metadata_types, "GAIN"),
        "color_auto_white_balance": _metadata_by_name(
            frame, metadata_types, "AUTO_WHITE_BALANCE"
        ),
        "color_white_balance": _metadata_by_name(
            frame, metadata_types, "WHITE_BALANCE"
        ),
    }


def _calibration_to_dict(camera_param: Any) -> dict[str, Any]:
    def intrinsic(value: Any) -> dict[str, Any]:
        return {
            "width": int(value.width),
            "height": int(value.height),
            "fx": float(value.fx),
            "fy": float(value.fy),
            "cx": float(value.cx),
            "cy": float(value.cy),
        }

    def distortion(value: Any) -> dict[str, Any]:
        return {
            "k1": float(value.k1),
            "k2": float(value.k2),
            "k3": float(value.k3),
            "k4": float(getattr(value, "k4", 0.0)),
            "k5": float(getattr(value, "k5", 0.0)),
            "k6": float(getattr(value, "k6", 0.0)),
            "p1": float(value.p1),
            "p2": float(value.p2),
        }

    return {
        "depth_alignment": {
            "enabled": True,
            "aligned_to": "color",
            "method": "software",
            "producer": "pyorbbecsdk.AlignFilter(COLOR_STREAM)",
        },
        "depth_intrinsic": intrinsic(camera_param.depth_intrinsic),
        "color_intrinsic": intrinsic(camera_param.rgb_intrinsic),
        "depth_distortion": distortion(camera_param.depth_distortion),
        "color_distortion": distortion(camera_param.rgb_distortion),
        "depth_to_color": {
            "rotation_row_major": np.asarray(
                camera_param.transform.rot, dtype=np.float64
            ).reshape(-1).tolist(),
            "translation_mm": np.asarray(
                camera_param.transform.transform, dtype=np.float64
            ).reshape(-1).tolist(),
        },
    }


def _set_int_property(device: Any, sdk: Any, property_name: str, value: int) -> int | None:
    try:
        prop = getattr(sdk.OBPropertyID, property_name)
        value_range = device.get_int_property_range(prop)
        bounded = int(np.clip(value, value_range.min, value_range.max))
        step = max(1, int(value_range.step))
        bounded = int(value_range.min + round((bounded - value_range.min) / step) * step)
        bounded = int(np.clip(bounded, value_range.min, value_range.max))
        device.set_int_property(prop, bounded)
        return int(device.get_int_property(prop))
    except Exception as exc:
        print(f"Property warning ({property_name}): {exc}", file=sys.stderr)
        return None


def _set_bool_property(device: Any, sdk: Any, property_name: str, value: bool) -> bool | None:
    try:
        prop = getattr(sdk.OBPropertyID, property_name)
        device.set_bool_property(prop, value)
        return bool(device.get_bool_property(prop))
    except Exception as exc:
        print(f"Property warning ({property_name}): {exc}", file=sys.stderr)
        return None


def _strict_color_control_property(
    device: Any, sdk: Any, property_name: str, label: str
) -> Any:
    """Return a writable color-control property or fail a formal lock closed."""

    try:
        prop = getattr(sdk.OBPropertyID, property_name)
        permission = sdk.OBPermissionType.PERMISSION_WRITE
        supported = bool(device.is_property_supported(prop, permission))
    except Exception as exc:
        raise RuntimeError(
            "Post-warmup color-control lock requires SDK support for "
            f"{label} ({property_name})"
        ) from exc
    if not supported:
        raise RuntimeError(
            "Post-warmup color-control lock requires writable "
            f"{label} ({property_name})"
        )
    return prop


def _strict_read_bool_property(device: Any, prop: Any, label: str) -> bool:
    try:
        return bool(device.get_bool_property(prop))
    except Exception as exc:
        raise RuntimeError(
            f"Post-warmup color-control lock could not read {label}"
        ) from exc


def _strict_read_int_property(device: Any, prop: Any, label: str) -> int:
    try:
        return int(device.get_int_property(prop))
    except Exception as exc:
        raise RuntimeError(
            f"Post-warmup color-control lock could not read {label}"
        ) from exc


def _strict_set_bool_property(device: Any, prop: Any, value: bool, label: str) -> None:
    try:
        device.set_bool_property(prop, bool(value))
    except Exception as exc:
        raise RuntimeError(
            f"Post-warmup color-control lock could not set {label}"
        ) from exc
    applied = _strict_read_bool_property(device, prop, label)
    if applied is not bool(value):
        raise RuntimeError(
            "Post-warmup color-control lock read back "
            f"{label}={applied}, expected {value}"
        )


def _strict_set_int_property(device: Any, prop: Any, value: int, label: str) -> None:
    try:
        value_range = device.get_int_property_range(prop)
        minimum = int(value_range.min)
        maximum = int(value_range.max)
        step = max(1, int(value_range.step))
    except Exception as exc:
        raise RuntimeError(
            "Post-warmup color-control lock could not read the "
            f"{label} range"
        ) from exc
    if value < minimum or value > maximum or (value - minimum) % step:
        raise RuntimeError(
            "Post-warmup color-control lock cannot preserve "
            f"{label}={value} in device range {minimum}..{maximum} step {step}"
        )
    try:
        device.set_int_property(prop, value)
    except Exception as exc:
        raise RuntimeError(
            f"Post-warmup color-control lock could not set {label}"
        ) from exc
    applied = _strict_read_int_property(device, prop, label)
    if applied != value:
        raise RuntimeError(
            "Post-warmup color-control lock read back "
            f"{label}={applied}, expected {value}"
        )


def _lock_color_controls_after_warmup(
    device: Any,
    sdk: Any,
    options: dict[str, Any],
    *,
    warmup_frame_sets: int,
) -> dict[str, Any]:
    """Freeze the device's converged AE/gain/AWB values with exact readback.

    The property values are deliberately stored as raw SDK values.  Only color
    exposure has a project-defined conversion to microseconds; gain and white
    balance units vary across device firmware and must not be relabelled.
    """

    auto_exposure = _strict_color_control_property(
        device,
        sdk,
        "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL",
        "color auto exposure",
    )
    exposure = _strict_color_control_property(
        device, sdk, "OB_PROP_COLOR_EXPOSURE_INT", "color exposure"
    )
    gain = _strict_color_control_property(
        device, sdk, "OB_PROP_COLOR_GAIN_INT", "color gain"
    )
    auto_white_balance = _strict_color_control_property(
        device,
        sdk,
        "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL",
        "color auto white balance",
    )
    white_balance = _strict_color_control_property(
        device,
        sdk,
        "OB_PROP_COLOR_WHITE_BALANCE_INT",
        "color white balance",
    )

    before_auto_exposure = _strict_read_bool_property(
        device, auto_exposure, "color auto exposure"
    )
    before_auto_white_balance = _strict_read_bool_property(
        device, auto_white_balance, "color auto white balance"
    )
    exposure_raw = _strict_read_int_property(device, exposure, "color exposure")
    gain_raw = _strict_read_int_property(device, gain, "color gain")
    white_balance_raw = _strict_read_int_property(
        device, white_balance, "color white balance"
    )
    if exposure_raw <= 0:
        raise RuntimeError(
            "Post-warmup color-control lock read a non-positive color exposure"
        )
    if not _uses_color_auto_exposure(options):
        requested_exposure = _color_exposure_units(options.get("color_exposure_us"))
        if exposure_raw != requested_exposure:
            raise RuntimeError(
                "Color exposure changed during warmup before the formal lock: "
                f"read {exposure_raw}, expected {requested_exposure}"
            )
    requested_gain = options.get("color_gain")
    if requested_gain is not None and gain_raw != int(requested_gain):
        raise RuntimeError(
            "Color gain changed during warmup before the formal lock: "
            f"read {gain_raw}, expected {int(requested_gain)}"
        )

    # A device that disregards its capped AE setting must not have its
    # overlong exposure frozen into a deliverable scan.  The formal cap is
    # fixed at 800 us, while a configuration may deliberately tighten it.
    formal_cap_us = _formal_locked_exposure_cap_us(options)
    if formal_cap_us is not None:
        if exposure_raw > _color_exposure_units(formal_cap_us):
            raise RuntimeError(
                "Post-warmup color-control lock read exposure "
                f"{exposure_raw * COLOR_EXPOSURE_UNIT_US} us above the formal "
                f"cap of {formal_cap_us} us"
            )

    # The Orbbec SDK requires AE to be off before setting either exposure or
    # gain, and AWB to be off before setting white balance.
    _strict_set_bool_property(device, auto_exposure, False, "color auto exposure")
    _strict_set_int_property(device, exposure, exposure_raw, "color exposure")
    _strict_set_int_property(device, gain, gain_raw, "color gain")
    _strict_set_bool_property(
        device, auto_white_balance, False, "color auto white balance"
    )
    _strict_set_int_property(
        device, white_balance, white_balance_raw, "color white balance"
    )

    requested_verification_frames = int(
        options.get("post_lock_verified_frames", 2)
    )
    if requested_verification_frames <= 0:
        raise ValueError("post_lock_verified_frames must be positive")
    return {
        "requested": True,
        "completed": False,
        "state": "locked_pending_frame_metadata",
        "lock_scope": "all",
        "warmup_frame_sets": int(warmup_frame_sets),
        "device_before_lock": {
            "auto_exposure": before_auto_exposure,
            "auto_white_balance": before_auto_white_balance,
            "exposure_raw": exposure_raw,
            "gain_raw": gain_raw,
            "white_balance_raw": white_balance_raw,
        },
        "locked_controls": {
            "auto_exposure": False,
            "exposure_raw": exposure_raw,
            "exposure_us": exposure_raw * COLOR_EXPOSURE_UNIT_US,
            "gain_raw": gain_raw,
            "auto_white_balance": False,
            "white_balance_raw": white_balance_raw,
        },
        "readback_verified": True,
        "formal_exposure_cap_us": formal_cap_us,
        "require_frame_metadata": bool(
            options.get("require_locked_control_metadata", True)
        ),
        "post_lock_verified_frames_requested": requested_verification_frames,
        "post_lock_verified_frames": 0,
        "post_lock_discarded_frames": 0,
        "post_lock_incomplete_frame_sets": 0,
        "post_lock_metadata_mismatches": 0,
    }


def _locked_color_metadata_status(
    metadata: dict[str, int | None], lock_audit: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return missing and mismatched metadata labels for a locked source frame."""

    expected = lock_audit["locked_controls"]
    lock_scope = str(lock_audit.get("lock_scope", "all"))
    if lock_scope not in {"white_balance", "exposure_gain", "all"}:
        raise ValueError(f"Unsupported color-control lock scope: {lock_scope!r}")
    expected_values: dict[str, int] = {}
    if lock_scope in {"exposure_gain", "all"}:
        expected_values.update(
            {
                "color_exposure": int(expected["exposure_raw"]),
                "color_gain": int(expected["gain_raw"]),
            }
        )
    if lock_scope in {"white_balance", "all"}:
        expected_values.update(
            {
                "color_auto_white_balance": 0,
                "color_white_balance": int(expected["white_balance_raw"]),
            }
        )
    selected_metadata = {name: metadata.get(name) for name in expected_values}
    missing = [name for name, value in selected_metadata.items() if value is None]
    mismatches = [
        f"{name}={value} (expected {expected_values[name]})"
        for name, value in selected_metadata.items()
        if value is not None and int(value) != expected_values[name]
    ]
    return missing, mismatches


def _require_valid_locked_color_metadata(
    metadata: dict[str, int | None], lock_audit: dict[str, Any]
) -> None:
    """Fail closed when an emitted frame does not prove the locked controls."""

    missing, mismatches = _locked_color_metadata_status(metadata, lock_audit)
    if missing and bool(lock_audit["require_frame_metadata"]):
        raise RuntimeError(
            "Locked color controls require frame metadata for "
            + ", ".join(missing)
        )
    if mismatches:
        raise RuntimeError(
            "Locked color-control metadata changed after warmup: "
            + ", ".join(mismatches)
        )


def _lock_white_balance_after_warmup(
    device: Any,
    sdk: Any,
    options: dict[str, Any],
    *,
    warmup_frame_sets: int,
) -> dict[str, Any]:
    """Freeze converged AWB while video auto-exposure and gain remain active."""

    auto_white_balance = _strict_color_control_property(
        device,
        sdk,
        "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL",
        "color auto white balance",
    )
    white_balance = _strict_color_control_property(
        device,
        sdk,
        "OB_PROP_COLOR_WHITE_BALANCE_INT",
        "color white balance",
    )
    before_auto_white_balance = _strict_read_bool_property(
        device, auto_white_balance, "color auto white balance"
    )
    white_balance_raw = _strict_read_int_property(
        device, white_balance, "color white balance"
    )
    if white_balance_raw <= 0:
        raise RuntimeError(
            "Post-warmup white-balance lock read a non-positive white balance"
        )
    _strict_set_bool_property(
        device, auto_white_balance, False, "color auto white balance"
    )
    _strict_set_int_property(
        device, white_balance, white_balance_raw, "color white balance"
    )
    requested_verification_frames = int(options.get("post_lock_verified_frames", 2))
    if requested_verification_frames <= 0:
        raise ValueError("post_lock_verified_frames must be positive")
    return {
        "requested": True,
        "completed": False,
        "state": "locked_pending_frame_metadata",
        "lock_scope": "white_balance",
        "warmup_frame_sets": int(warmup_frame_sets),
        "device_before_lock": {
            "auto_white_balance": before_auto_white_balance,
            "white_balance_raw": white_balance_raw,
        },
        "locked_controls": {
            "auto_white_balance": False,
            "white_balance_raw": white_balance_raw,
        },
        "readback_verified": True,
        "require_frame_metadata": bool(
            options.get("require_locked_white_balance_metadata", True)
        ),
        "post_lock_verified_frames_requested": requested_verification_frames,
        "post_lock_verified_frames": 0,
        "post_lock_discarded_frames": 0,
        "post_lock_incomplete_frame_sets": 0,
        "post_lock_metadata_mismatches": 0,
    }


def _discard_and_verify_post_lock_frames(
    pipeline: Any,
    metadata_types: Any,
    lock_audit: dict[str, Any],
    *,
    timeout_seconds: float,
    clock: Any = time.monotonic,
) -> None:
    """Discard the color-control transition and require a stable metadata run."""

    if timeout_seconds <= 0.0:
        raise ValueError("post-lock verification timeout must be positive")
    deadline = float(clock()) + float(timeout_seconds)
    requested_verification = int(lock_audit["post_lock_verified_frames_requested"])
    verified_frames = 0
    metadata_verified_frames = 0

    def require_time_remaining() -> None:
        if float(clock()) >= deadline:
            raise _VideoCapturePreflightError(
                "Post-warmup color-control lock did not receive enough complete "
                "RGB-D frames for metadata verification"
            )

    while verified_frames < requested_verification:
        require_time_remaining()
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            require_time_remaining()
            continue
        raw_color = frames.get_color_frame()
        raw_depth = frames.get_depth_frame()
        if raw_color is None or raw_depth is None:
            lock_audit["post_lock_incomplete_frame_sets"] += 1
            require_time_remaining()
            continue
        require_time_remaining()
        lock_audit["post_lock_discarded_frames"] += 1
        controls = _color_control_metadata(raw_color, metadata_types)
        missing, mismatches = _locked_color_metadata_status(controls, lock_audit)
        if missing and bool(lock_audit["require_frame_metadata"]):
            raise RuntimeError(
                "Locked color controls require frame metadata for " + ", ".join(missing)
            )
        if mismatches:
            lock_audit["post_lock_metadata_mismatches"] += 1
            verified_frames = 0
            metadata_verified_frames = 0
            if float(clock()) >= deadline:
                raise RuntimeError(
                    "Post-warmup color-control lock did not stabilize: "
                    + ", ".join(mismatches)
                )
            continue
        if not missing:
            metadata_verified_frames += 1
        verified_frames += 1
    lock_audit.update(
        {
            "completed": True,
            "state": "verified",
            "post_lock_verified_frames": verified_frames,
            "post_lock_metadata_verified_frames": metadata_verified_frames,
        }
    )


def _set_int_property_to_device_max(
    device: Any,
    sdk: Any,
    property_name: str,
    *,
    minimum_exclusive: int | None = None,
) -> tuple[int, int]:
    """Request a device maximum and return its range maximum and applied value.

    Firmware may clamp the generic property range to the maximum supported by
    the active frame rate. That lower readback is valid as long as it actually
    lifts the project cap being replaced.
    """

    try:
        prop = getattr(sdk.OBPropertyID, property_name)
        value_range = device.get_int_property_range(prop)
        minimum = int(value_range.min)
        maximum = int(value_range.max)
        step = max(1, int(value_range.step))
        if maximum <= 0:
            raise ValueError(f"device reported invalid maximum {maximum}")
        device.set_int_property(prop, maximum)
        applied = int(device.get_int_property(prop))
    except Exception as exc:
        raise RuntimeError(
            f"The camera cannot remove the project exposure cap via {property_name}"
        ) from exc
    if (
        applied < minimum
        or applied > maximum
        or (applied - minimum) % step != 0
    ):
        raise RuntimeError(
            "The camera returned an invalid maximum auto-exposure readback: "
            f"device range {minimum}..{maximum} step {step}, applied {applied}"
        )
    if minimum_exclusive is not None and applied <= minimum_exclusive:
        raise RuntimeError(
            "The camera did not lift its auto-exposure limit above the project "
            f"cap: requested device maximum {maximum}, applied {applied}, "
            f"required above {minimum_exclusive}"
        )
    return maximum, applied


def _uses_color_auto_exposure(options: dict[str, Any]) -> bool:
    configured = options.get("color_auto_exposure")
    if configured is None:
        return options.get("color_exposure_us") is None
    return bool(configured)


def _uses_color_auto_white_balance(options: dict[str, Any]) -> bool:
    """Resolve an explicit AWB policy while retaining old manual-WB configs."""

    configured = options.get("color_auto_white_balance")
    if configured is None:
        return options.get("color_white_balance") is None
    return bool(configured)


def _formal_locked_exposure_cap_us(options: dict[str, Any]) -> int | None:
    """Return the fixed formal lock cap, tightened by an explicit AE cap."""

    if bool(options.get("diagnostic_unrestricted_auto_exposure", False)):
        return None
    configured_cap_us = options.get("color_ae_max_exposure_us")
    if configured_cap_us is None:
        return MAX_FORMAL_COLOR_EXPOSURE_US
    value = int(configured_cap_us)
    if value <= 0:
        raise ValueError("color_ae_max_exposure_us must be positive")
    return min(MAX_FORMAL_COLOR_EXPOSURE_US, value)


def _color_exposure_units(exposure_us: object) -> int:
    if exposure_us is None:
        raise ValueError(
            "color_exposure_us is required when color_auto_exposure is false"
        )
    value = int(exposure_us)
    if value <= 0:
        raise ValueError("color_exposure_us must be positive")
    return max(1, int(np.floor(value / COLOR_EXPOSURE_UNIT_US + 0.5)))


def _color_exposure_metadata_violation(
    options: dict[str, Any], exposure_raw: int | None
) -> str | None:
    """Validate streaming metadata against the active auto/manual exposure mode."""

    if exposure_raw is None:
        return None
    measured_units = int(exposure_raw)
    if _uses_color_auto_exposure(options):
        if bool(options.get("diagnostic_unrestricted_auto_exposure", False)):
            return None
        cap_us = options.get("color_ae_max_exposure_us")
        if cap_us is None:
            return None
        cap_units = _color_exposure_units(cap_us)
        if measured_units > cap_units:
            return (
                "Camera exposure exceeded the motion-safe auto-exposure limit; "
                "add lighting or update camera firmware"
            )
        return None
    requested_us = options.get("color_exposure_us")
    if requested_us is None:
        return "Manual color exposure mode has no requested exposure duration"
    requested_units = _color_exposure_units(requested_us)
    if abs(measured_units - requested_units) > 1:
        return (
            "Camera did not maintain the requested manual color exposure "
            f"({requested_units * COLOR_EXPOSURE_UNIT_US} us)"
        )
    return None


def _fall_back_to_fixed_motion_safe_exposure(
    device: Any,
    sdk: Any,
    options: dict[str, Any],
    *,
    gain_raw: int,
) -> dict[str, Any]:
    """Stop AE and lock the safety ceiling plus the last observed auto gain."""

    cap_us = _formal_locked_exposure_cap_us(options)
    if cap_us is None:
        raise RuntimeError(
            "Cannot fall back from uncapped automatic exposure without a safety cap"
        )
    cap_units = _color_exposure_units(cap_us)
    auto_exposure = _strict_color_control_property(
        device,
        sdk,
        "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL",
        "color auto exposure",
    )
    exposure = _strict_color_control_property(
        device, sdk, "OB_PROP_COLOR_EXPOSURE_INT", "color exposure"
    )
    gain = _strict_color_control_property(
        device, sdk, "OB_PROP_COLOR_GAIN_INT", "color gain"
    )
    _strict_set_bool_property(device, auto_exposure, False, "color auto exposure")
    _strict_set_int_property(device, exposure, cap_units, "color exposure")
    _strict_set_int_property(device, gain, int(gain_raw), "color gain")
    options["color_auto_exposure"] = False
    options["color_exposure_us"] = cap_us
    options["color_gain"] = int(gain_raw)
    options["diagnostic_unrestricted_auto_exposure"] = False
    requested_verification_frames = int(options.get("post_lock_verified_frames", 2))
    if requested_verification_frames <= 0:
        raise ValueError("post_lock_verified_frames must be positive")
    return {
        "requested": True,
        "completed": False,
        "state": "fallback_locked_pending_frame_metadata",
        "lock_scope": "exposure_gain",
        "auto_exposure": False,
        "exposure": cap_units,
        "exposure_us": cap_units * COLOR_EXPOSURE_UNIT_US,
        "gain": int(gain_raw),
        "gain_source": "last_valid_complete_warmup_frame_metadata",
        "fallback_reason": "warmup_metadata_exceeded_auto_exposure_cap",
        "fallback_cap_us": cap_us,
        "require_frame_metadata": True,
        "post_lock_verified_frames_requested": requested_verification_frames,
        "post_lock_verified_frames": 0,
        "post_lock_discarded_frames": 0,
        "post_lock_incomplete_frame_sets": 0,
        "post_lock_metadata_mismatches": 0,
        "locked_controls": {
            "auto_exposure": False,
            "exposure_raw": cap_units,
            "exposure_us": cap_units * COLOR_EXPOSURE_UNIT_US,
            "gain_raw": int(gain_raw),
        },
    }


def _warm_up_video_controls(
    device: Any,
    sdk: Any,
    pipeline: Any,
    metadata_types: Any,
    options: dict[str, Any],
    *,
    clock: Callable[[], float] = time.monotonic,
    on_fallback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Warm up on complete RGB-D sets and stabilize a one-time AE fallback."""

    target = int(options.get("warmup_frames", 30))
    if target <= 0:
        raise ValueError("warmup_frames must be positive")
    timeout_seconds = float(options.get("warmup_timeout_seconds", 15))
    if timeout_seconds <= 0.0:
        raise ValueError("warmup_timeout_seconds must be positive")
    deadline = float(clock()) + timeout_seconds
    fallback_allowed = _uses_color_auto_exposure(options)
    startup_transition_limit = (
        AUTO_EXPOSURE_STARTUP_TRANSITION_MAX_FRAME_SETS if fallback_allowed else 0
    )
    startup_transition_discarded = 0
    startup_transition_active = fallback_allowed
    received = 0
    incomplete = 0
    last_valid_auto_gain_raw: int | None = None
    fallback_audit: dict[str, Any] | None = None

    while received < target:
        remaining = deadline - float(clock())
        if remaining <= 0.0:
            error_type = (
                _VideoWarmupNoFramesError
                if received == 0
                else _VideoCapturePreflightError
            )
            raise error_type(
                f"Capture warmup timed out after receiving {received}/{target} "
                "complete RGB-D frame sets. Try MJPG color, a lower resolution, "
                "another USB 3 port, or disable other camera applications."
            )
        wait_ms = max(1, min(1000, int(np.ceil(remaining * 1000.0))))
        frames = pipeline.wait_for_frames(wait_ms)
        if frames is None:
            continue
        raw_color = frames.get_color_frame()
        raw_depth = frames.get_depth_frame()
        if raw_color is None or raw_depth is None:
            incomplete += 1
            continue

        controls = _color_control_metadata(raw_color, metadata_types)
        if fallback_audit is not None:
            _require_valid_locked_color_metadata(controls, fallback_audit)
            received += 1
            continue

        gain_raw = controls["color_gain"]
        if gain_raw is not None and int(gain_raw) >= 0:
            last_valid_auto_gain_raw = int(gain_raw)
        exposure_raw = controls["color_exposure"]
        exposure_violation = _color_exposure_metadata_violation(options, exposure_raw)
        if exposure_violation is None:
            startup_transition_active = False
            received += 1
            continue
        if (
            startup_transition_active
            and exposure_raw is not None
            and startup_transition_discarded < startup_transition_limit
        ):
            # Gemini 305 can emit several complete frames carrying its stale
            # 3000-us startup exposure before the configured AE ceiling takes
            # effect.  These frames are neither warmup evidence nor a reason
            # to disable a correctly converging AE controller.  The bounded
            # allowance ends at the first compliant frame; later violations
            # still take the production fallback immediately.
            startup_transition_discarded += 1
            continue
        startup_transition_active = False
        if not fallback_allowed:
            raise RuntimeError(exposure_violation)
        if last_valid_auto_gain_raw is None:
            raise RuntimeError(
                "Cannot apply the 800-us fallback without valid warmup "
                "color-gain metadata"
            )

        fallback_audit = _fall_back_to_fixed_motion_safe_exposure(
            device,
            sdk,
            options,
            gain_raw=last_valid_auto_gain_raw,
        )
        fallback_audit.update(
            {
                "trigger_exposure_raw": (
                    None if exposure_raw is None else int(exposure_raw)
                ),
                "trigger_exposure_us": (
                    None
                    if exposure_raw is None
                    else int(exposure_raw) * COLOR_EXPOSURE_UNIT_US
                ),
            }
        )
        if on_fallback is not None:
            on_fallback(fallback_audit)
        transition_timeout = min(
            float(options.get("frame_timeout_seconds", 5)),
            max(0.0, deadline - float(clock())),
        )
        if transition_timeout <= 0.0:
            raise _VideoCapturePreflightError(
                f"Capture warmup timed out after receiving {received}/{target} "
                "complete RGB-D frame sets"
            )
        _discard_and_verify_post_lock_frames(
            pipeline,
            metadata_types,
            fallback_audit,
            timeout_seconds=transition_timeout,
            clock=clock,
        )
        if on_fallback is not None:
            on_fallback(fallback_audit)

    return {
        "warmup_frame_sets": received,
        "warmup_incomplete_frame_sets": incomplete,
        "warmup_auto_exposure_startup_discarded_frame_sets": (
            startup_transition_discarded
        ),
        "warmup_auto_exposure_startup_limit_frame_sets": startup_transition_limit,
        "warmup_exposure_fallback": fallback_audit,
    }


def _color_exposure_control_summary(
    options: dict[str, Any], applied: dict[str, Any]
) -> dict[str, Any]:
    requested_auto = _uses_color_auto_exposure(options)
    applied_auto = bool(applied["auto_exposure"])
    unrestricted_diagnostic = bool(
        options.get("diagnostic_unrestricted_auto_exposure", False)
    )
    return {
        "requested_mode": "auto" if requested_auto else "manual",
        "effective_mode": (
            "auto"
            if applied_auto
            else ("manual_fallback" if requested_auto else "manual")
        ),
        "requested_exposure_us": options.get("color_exposure_us"),
        "requested_auto_cap_us": (
            options.get("color_ae_max_exposure_us") if requested_auto else None
        ),
        "applied_exposure_us": applied.get("exposure_us"),
        "applied_auto_cap_us": applied.get("ae_max_exposure_us"),
        "replaced_motion_safe_cap_us": options.get(
            "diagnostic_replaced_auto_cap_us"
        ),
        "requested_device_max_auto_cap_us": applied.get(
            "ae_max_exposure_requested_us"
        ),
        "device_clamped_auto_cap": applied.get("ae_max_exposure_device_clamped"),
        "auto_exposure_policy": (
            "device_max_diagnostic"
            if unrestricted_diagnostic
            else ("motion_safe_cap" if requested_auto else "manual")
        ),
        "diagnostic_only": unrestricted_diagnostic,
    }


def _configure_color(device: Any, sdk: Any, options: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    exposure = options.get("color_exposure_us")
    auto_exposure = _uses_color_auto_exposure(options)
    ae_max_exposure = options.get("color_ae_max_exposure_us")
    unrestricted_diagnostic = bool(
        options.get("diagnostic_unrestricted_auto_exposure", False)
    )
    if unrestricted_diagnostic and (not auto_exposure or exposure is not None):
        raise ValueError(
            "diagnostic_unrestricted_auto_exposure requires automatic color exposure"
        )
    if auto_exposure and exposure is not None:
        raise ValueError(
            "color_exposure_us must be null when color_auto_exposure is true"
        )
    if not auto_exposure and exposure is None:
        raise ValueError(
            "color_exposure_us is required when color_auto_exposure is false"
        )
    requested_manual_units = (
        _color_exposure_units(exposure) if not auto_exposure else None
    )
    effective_auto_exposure = bool(auto_exposure)
    fallback_exposure_us: int | None = None
    if auto_exposure and unrestricted_diagnostic:
        replaced_cap_us = options.get("diagnostic_replaced_auto_cap_us")
        if replaced_cap_us is None:
            raise ValueError(
                "diagnostic unrestricted auto exposure requires the formal "
                "auto-exposure cap it replaces"
            )
        replaced_cap_units = _color_exposure_units(replaced_cap_us)
        requested_cap, applied_cap = _set_int_property_to_device_max(
            device,
            sdk,
            "OB_PROP_COLOR_AE_MAX_EXPOSURE_INT",
            minimum_exclusive=replaced_cap_units,
        )
        applied["ae_max_exposure_requested"] = requested_cap
        applied["ae_max_exposure_requested_us"] = (
            requested_cap * COLOR_EXPOSURE_UNIT_US
        )
        applied["ae_max_exposure"] = applied_cap
        applied["ae_max_exposure_us"] = applied_cap * COLOR_EXPOSURE_UNIT_US
        applied["ae_max_exposure_device_clamped"] = applied_cap < requested_cap
    elif auto_exposure and ae_max_exposure is not None:
        ae_max_exposure_us = int(ae_max_exposure)
        if ae_max_exposure_us <= 0:
            raise ValueError("color_ae_max_exposure_us must be positive")
        cap_units = _color_exposure_units(ae_max_exposure_us)
        applied_cap = _set_int_property(
            device, sdk, "OB_PROP_COLOR_AE_MAX_EXPOSURE_INT", cap_units
        )
        applied["ae_max_exposure"] = applied_cap
        applied["ae_max_exposure_us"] = (
            applied_cap * COLOR_EXPOSURE_UNIT_US
            if applied_cap is not None
            else None
        )
        if applied_cap is None or applied_cap > cap_units:
            # A long unrestricted exposure irreversibly blurs a moving side scan.
            # Fail safe on older firmware by using the requested cap manually.
            effective_auto_exposure = False
            fallback_exposure_us = ae_max_exposure_us
    applied_auto_exposure = _set_bool_property(
        device,
        sdk,
        "OB_PROP_COLOR_AUTO_EXPOSURE_BOOL",
        effective_auto_exposure,
    )
    applied["auto_exposure"] = applied_auto_exposure
    if applied_auto_exposure is None or applied_auto_exposure != effective_auto_exposure:
        raise _VideoCameraControlUnavailableError(
            "The camera did not apply the requested color exposure mode"
        )
    manual_exposure = fallback_exposure_us if fallback_exposure_us is not None else exposure
    if not effective_auto_exposure and manual_exposure is not None:
        exposure_us = int(manual_exposure)
        exposure_units = (
            _color_exposure_units(exposure_us)
            if fallback_exposure_us is not None
            else requested_manual_units
        )
        assert exposure_units is not None
        applied_units = _set_int_property(
            device, sdk, "OB_PROP_COLOR_EXPOSURE_INT", exposure_units
        )
        if applied_units is None:
            raise _VideoCameraControlUnavailableError(
                "The camera did not apply the requested color exposure"
            )
        if fallback_exposure_us is not None and applied_units > exposure_units:
            raise RuntimeError(
                "The camera cannot enforce the motion-safe color exposure limit"
            )
        if fallback_exposure_us is None and applied_units != exposure_units:
            raise RuntimeError(
                "The camera did not apply the requested manual color exposure: "
                f"requested {exposure_units * COLOR_EXPOSURE_UNIT_US} us, "
                f"applied {applied_units * COLOR_EXPOSURE_UNIT_US} us"
            )
        applied["exposure"] = applied_units
        applied["exposure_us"] = (
            applied_units * COLOR_EXPOSURE_UNIT_US
            if applied_units is not None
            else None
        )
    gain = options.get("color_gain")
    if gain is not None:
        applied["gain"] = _set_int_property(device, sdk, "OB_PROP_COLOR_GAIN_INT", int(gain))
    white_balance = options.get("color_white_balance")
    auto_white_balance = _uses_color_auto_white_balance(options)
    if auto_white_balance and white_balance is not None:
        raise ValueError(
            "color_white_balance must be null when color_auto_white_balance is true"
        )
    applied["auto_white_balance"] = _set_bool_property(
        device,
        sdk,
        "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL",
        auto_white_balance,
    )
    if (
        bool(options.get("lock_color_controls_after_warmup", False))
        and applied["auto_white_balance"] is not auto_white_balance
    ):
        raise _VideoCameraControlUnavailableError(
            "The camera did not apply the requested color white-balance mode"
        )
    if white_balance is not None:
        applied["white_balance"] = _set_int_property(
            device, sdk, "OB_PROP_COLOR_WHITE_BALANCE_INT", int(white_balance)
        )
    if options.get("color_anti_flicker", True):
        applied["anti_flicker"] = _set_bool_property(
            device, sdk, "OB_PROP_COLOR_ANTI_FLICKER_BOOL", True
        )
    return applied


def _apply_color_exposure_mode(
    options: dict[str, Any], args: argparse.Namespace
) -> bool:
    """Apply CLI exposure selection and return whether capture is diagnostic-only."""

    if bool(getattr(args, "diagnostic_unrestricted_auto_exposure", False)):
        options["diagnostic_unrestricted_auto_exposure"] = True
    elif bool(getattr(args, "auto_exposure", False)):
        options["diagnostic_unrestricted_auto_exposure"] = False
        options["color_auto_exposure"] = True
        options["color_exposure_us"] = None
        if options.get("color_ae_max_exposure_us") is None:
            options["color_ae_max_exposure_us"] = options.get(
                "diagnostic_replaced_auto_cap_us"
            )
    elif getattr(args, "exposure_us", None) is not None:
        options["diagnostic_unrestricted_auto_exposure"] = False
        options["color_auto_exposure"] = False
        options["color_exposure_us"] = args.exposure_us

    diagnostic_only = bool(
        options.get("diagnostic_unrestricted_auto_exposure", False)
    )
    options["diagnostic_unrestricted_auto_exposure"] = diagnostic_only
    if diagnostic_only:
        options["color_auto_exposure"] = True
        options["color_exposure_us"] = None
        replaced_cap_us = options.get("color_ae_max_exposure_us")
        if replaced_cap_us is None:
            replaced_cap_us = options.get("diagnostic_replaced_auto_cap_us")
        if replaced_cap_us is None:
            raise ValueError(
                "Diagnostic unrestricted auto exposure requires a positive "
                "color_ae_max_exposure_us baseline in the merged configuration"
            )
        _color_exposure_units(replaced_cap_us)
        options["diagnostic_replaced_auto_cap_us"] = replaced_cap_us
        options["color_ae_max_exposure_us"] = None
    elif (
        _uses_color_auto_exposure(options)
        and options.get("color_ae_max_exposure_us") is None
    ):
        raise ValueError(
            "Uncapped color auto exposure is diagnostic-only; use "
            "--diagnostic-unrestricted-auto-exposure"
        )
    else:
        options.pop("diagnostic_replaced_auto_cap_us", None)
    return diagnostic_only


def _sync_mode_name(mode: Any) -> str:
    name = getattr(mode, "name", None)
    return str(name) if name is not None else str(mode).rsplit(".", 1)[-1]


def _sync_config_to_dict(config: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": _sync_mode_name(config.mode),
        "trigger_out_enable": bool(config.trigger_out_enable),
    }
    for name in (
        "color_delay_us",
        "depth_delay_us",
        "trigger_to_image_delay_us",
        "trigger_out_delay_us",
        "frames_per_trigger",
    ):
        try:
            result[name] = int(getattr(config, name))
        except (AttributeError, TypeError, ValueError):
            result[name] = None
    return result


def _configure_external_sync_output(
    device: Any, sdk: Any, options: dict[str, Any]
) -> dict[str, Any]:
    """Configure a continuous sync pulse tied to the active RGB-D frame rate."""

    enabled = bool(options.get("external_sync_output", True))
    if not enabled:
        return {"enabled": False}

    frame_rate_hz = int(options["fps"])
    if frame_rate_hz <= 0:
        raise ValueError("fps must be positive when external_sync_output is enabled")
    image_delay_us = options.get(
        "trigger_to_image_delay_us", DEFAULT_TRIGGER_TO_IMAGE_DELAY_US
    )
    trigger_out_delay_us = options.get(
        "trigger_out_delay_us", DEFAULT_TRIGGER_OUT_DELAY_US
    )
    if (
        isinstance(image_delay_us, bool)
        or not isinstance(image_delay_us, int)
        or image_delay_us < 0
    ):
        raise ValueError("trigger_to_image_delay_us must be a non-negative integer")
    if (
        isinstance(trigger_out_delay_us, bool)
        or not isinstance(trigger_out_delay_us, int)
        or trigger_out_delay_us < 0
    ):
        raise ValueError("trigger_out_delay_us must be a non-negative integer")
    frame_period_us = round(1_000_000 / frame_rate_hz)
    if max(image_delay_us, trigger_out_delay_us) >= frame_period_us:
        raise ValueError(
            "trigger_to_image_delay_us and trigger_out_delay_us must each be "
            f"below the {frame_period_us} us frame period at {frame_rate_hz} FPS"
        )
    try:
        primary_mode = sdk.OBMultiDeviceSyncMode.PRIMARY
        get_sync_config = device.get_multi_device_sync_config
        set_sync_config = device.set_multi_device_sync_config
    except AttributeError as exc:
        raise RuntimeError(
            "The camera SDK does not expose the required external sync output configuration"
        ) from exc

    try:
        requested = get_sync_config()
        requested.mode = primary_mode
        requested.depth_delay_us = image_delay_us
        requested.color_delay_us = image_delay_us
        requested.trigger_to_image_delay_us = image_delay_us
        requested.trigger_out_enable = True
        requested.trigger_out_delay_us = trigger_out_delay_us
        requested.frames_per_trigger = 1
        set_sync_config(requested)
    except Exception as exc:
        raise RuntimeError(
            "The camera did not apply the external sync output configuration"
        ) from exc

    result = _verify_external_sync_output(device, sdk, options)
    result.update(
        {
            "per_frame_readback_verified_frames": 0,
            "expected_frequency_hz": frame_rate_hz,
            "frequency_source": "capture_fps",
        }
    )
    return result


def _verify_external_sync_output(
    device: Any, sdk: Any, options: dict[str, Any]
) -> dict[str, Any]:
    """Read back the continuous-video sync timing without rewriting it."""

    enabled = bool(options.get("external_sync_output", True))
    if not enabled:
        return {"enabled": False}
    image_delay_us = options.get(
        "trigger_to_image_delay_us", DEFAULT_TRIGGER_TO_IMAGE_DELAY_US
    )
    trigger_out_delay_us = options.get(
        "trigger_out_delay_us", DEFAULT_TRIGGER_OUT_DELAY_US
    )
    if (
        isinstance(image_delay_us, bool)
        or not isinstance(image_delay_us, int)
        or image_delay_us < 0
    ):
        raise ValueError("trigger_to_image_delay_us must be a non-negative integer")
    if (
        isinstance(trigger_out_delay_us, bool)
        or not isinstance(trigger_out_delay_us, int)
        or trigger_out_delay_us < 0
    ):
        raise ValueError("trigger_out_delay_us must be a non-negative integer")
    try:
        primary_mode = sdk.OBMultiDeviceSyncMode.PRIMARY
        applied = device.get_multi_device_sync_config()
    except Exception as exc:
        raise RuntimeError(
            "The camera did not return the external sync output configuration"
        ) from exc

    applied_mode = getattr(applied, "mode", None)
    if applied_mode != primary_mode:
        raise RuntimeError(
            "The camera did not enter PRIMARY sync mode: "
            f"applied {_sync_mode_name(applied_mode)}"
        )
    if not bool(getattr(applied, "trigger_out_enable", False)):
        raise RuntimeError("The camera did not enable its external sync trigger output")
    if getattr(applied, "frames_per_trigger", None) != 1:
        raise RuntimeError(
            "The camera did not apply single-frame synchronization: "
            f"frames_per_trigger={getattr(applied, 'frames_per_trigger', None)!r}"
        )
    if getattr(applied, "trigger_out_delay_us", None) != trigger_out_delay_us:
        raise RuntimeError(
            "The camera did not apply Trigger Out delay: "
            f"read {getattr(applied, 'trigger_out_delay_us', None)!r}, "
            f"expected {trigger_out_delay_us}"
        )
    if getattr(applied, "trigger_to_image_delay_us", None) != image_delay_us:
        raise RuntimeError(
            "The camera did not apply trigger-to-image delay: "
            f"read {getattr(applied, 'trigger_to_image_delay_us', None)!r}, "
            f"expected {image_delay_us}"
        )
    for delay_field in ("color_delay_us", "depth_delay_us"):
        actual = getattr(applied, delay_field, None)
        if actual != image_delay_us:
            raise RuntimeError(
                f"The camera did not apply image delay to {delay_field}: "
                f"read {actual!r}, expected {image_delay_us}"
            )

    result = _sync_config_to_dict(applied)
    result.update(
        {
            "enabled": True,
            "readback_verified": True,
        }
    )
    return result


def _disable_external_sync_output_after_capture(
    device: Any,
    sdk: Any,
    external_sync_output: dict[str, Any],
) -> dict[str, Any]:
    """Switch to STANDALONE and prove that the external Trigger Out is off."""

    if external_sync_output.get("enabled") is not True:
        return {
            "requested": False,
            "attempted": False,
            "completed": True,
            "state": "not_requested",
            "readback_verified": False,
        }
    try:
        standalone_mode = sdk.OBMultiDeviceSyncMode.STANDALONE
        current = device.get_multi_device_sync_config()
        before = _sync_config_to_dict(current)
        current.mode = standalone_mode
        current.trigger_out_enable = False
        device.set_multi_device_sync_config(current)
        applied = device.get_multi_device_sync_config()
    except Exception as exc:
        raise RuntimeError(
            "The camera could not disable external Trigger Out after capture"
        ) from exc

    after = _sync_config_to_dict(applied)
    if getattr(applied, "mode", None) != standalone_mode:
        raise RuntimeError(
            "External Trigger Out shutdown did not enter STANDALONE mode: "
            f"applied {after['mode']}"
        )
    if bool(getattr(applied, "trigger_out_enable", True)):
        raise RuntimeError(
            "External Trigger Out shutdown readback still reports trigger_out_enable=true"
        )
    preserved_fields = (
        "color_delay_us",
        "depth_delay_us",
        "trigger_to_image_delay_us",
        "trigger_out_delay_us",
        "frames_per_trigger",
    )
    changed = [
        name for name in preserved_fields if after.get(name) != before.get(name)
    ]
    if changed:
        details = ", ".join(
            f"{name}={after.get(name)!r} (before {before.get(name)!r})"
            for name in changed
        )
        raise RuntimeError(
            "External Trigger Out shutdown changed preserved sync fields: " + details
        )
    return {
        "requested": True,
        "attempted": True,
        "completed": True,
        "state": "verified_off",
        "before": before,
        "after": after,
        "readback_verified": True,
    }


def _shutdown_video_capture_hardware(
    pipeline: Any,
    *,
    pipeline_started: bool,
    device: Any,
    sdk: Any,
    external_sync_output: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Stop streaming, then disable Trigger Out while retaining both errors."""

    errors: list[dict[str, str]] = []
    if pipeline_started:
        try:
            pipeline.stop()
        except Exception as exc:
            errors.append(
                {
                    "step": "pipeline.stop",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    try:
        shutdown = _disable_external_sync_output_after_capture(
            device, sdk, external_sync_output
        )
    except Exception as exc:
        errors.append(
            {
                "step": "external_sync_output.shutdown",
                "type": type(exc).__name__,
                "message": str(exc),
            }
        )
        shutdown = {
            "requested": external_sync_output.get("enabled") is True,
            "attempted": True,
            "completed": False,
            "state": "failed",
            "readback_verified": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "operator_action": "Check the device sync output or power-cycle the camera",
        }
    return shutdown, errors


def _capture_exception_after_shutdown(
    capture_exception: Exception | None,
    shutdown_errors: list[dict[str, str]],
) -> Exception | None:
    """Keep the primary capture error, or promote the first shutdown error."""

    if capture_exception is not None or not shutdown_errors:
        return capture_exception
    first = shutdown_errors[0]
    return RuntimeError(
        f"{first['step']} failed during capture shutdown: {first['message']}"
    )


def _video_capture_clean_shutdown(
    capture_exception: Exception | None,
    writer_stats: Any,
    received_frames: int,
    external_sync_output: dict[str, Any],
) -> bool:
    """Return eligibility-grade capture shutdown state, including Trigger Out."""

    shutdown = external_sync_output.get("shutdown", {})
    if external_sync_output.get("enabled") is True:
        sync_shutdown_ok = (
            shutdown.get("completed") is True
            and shutdown.get("state") == "verified_off"
            and shutdown.get("after", {}).get("trigger_out_enable") is False
        )
    else:
        sync_shutdown_ok = shutdown.get("state") == "not_requested"
    return bool(
        capture_exception is None
        and int(writer_stats.write_errors) == 0
        and list(writer_stats.errors) == []
        and int(writer_stats.queue_drops) == 0
        and int(writer_stats.written) == int(received_frames)
        and sync_shutdown_ok
    )


def _device_info(device: Any) -> dict[str, Any]:
    info = device.get_device_info()
    result: dict[str, Any] = {}
    for key, method in {
        "name": "get_name",
        "serial_number": "get_serial_number",
        "firmware_version": "get_firmware_version",
        "hardware_version": "get_hardware_version",
        "connection_type": "get_connection_type",
        "vid": "get_vid",
        "pid": "get_pid",
    }.items():
        try:
            result[key] = getattr(info, method)()
        except Exception:
            result[key] = None
    return result


def _compose_capture_preview(
    color: np.ndarray,
    depth: np.ndarray,
    scale: float,
    panorama: np.ndarray | None = None,
) -> np.ndarray:
    depth_mm = depth.astype(np.float32) * scale
    valid = (depth_mm >= 50.0) & (depth_mm <= 1000.0)
    depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
    depth_u8[valid] = np.clip((depth_mm[valid] - 50.0) * (255.0 / 950.0), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    if colored.shape[:2] != color.shape[:2]:
        colored = cv2.resize(colored, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    capture_row = np.hstack((color, colored))
    cv2.putText(
        capture_row, "LIVE RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        capture_row, "ALIGNED DEPTH", (color.shape[1] + 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    panorama_height = max(120, color.shape[0] // 2)
    panorama_panel = np.zeros((panorama_height, capture_row.shape[1], 3), dtype=np.uint8)
    if panorama is None:
        message = "LIVE PANORAMA: waiting for stable one-way motion"
        text_size, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        origin = (
            max(12, (panorama_panel.shape[1] - text_size[0]) // 2),
            max(32, (panorama_panel.shape[0] + text_size[1]) // 2),
        )
        cv2.putText(
            panorama_panel, message, origin, cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (180, 180, 180), 2, cv2.LINE_AA,
        )
    else:
        if panorama.dtype != np.uint8 or panorama.ndim != 3 or panorama.shape[2] != 3:
            raise ValueError("Live panorama preview must be an HxWx3 uint8 image")
        factor = min(
            panorama_panel.shape[1] / panorama.shape[1],
            panorama_panel.shape[0] / panorama.shape[0],
        )
        preview = (
            panorama
            if abs(factor - 1.0) < 1e-9
            else cv2.resize(
                panorama, None, fx=factor, fy=factor,
                interpolation=cv2.INTER_AREA if factor < 1.0 else cv2.INTER_LINEAR,
            )
        )
        top = (panorama_panel.shape[0] - preview.shape[0]) // 2
        left = (panorama_panel.shape[1] - preview.shape[1]) // 2
        panorama_panel[top:top + preview.shape[0], left:left + preview.shape[1]] = preview
        cv2.putText(
            panorama_panel, "LIVE PANORAMA (NON-AUTHORITATIVE)", (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    display = np.vstack((capture_row, panorama_panel))
    max_width, max_height = 1600, 900
    factor = min(max_width / display.shape[1], max_height / display.shape[0], 1.0)
    if factor < 1.0:
        display = cv2.resize(display, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    return display


def _preview(
    color: np.ndarray,
    depth: np.ndarray,
    scale: float,
    panorama: np.ndarray | None = None,
) -> int:
    display = _compose_capture_preview(color, depth, scale, panorama)
    cv2.imshow("Gemini 305 RGB-D capture | Q or ESC to stop", display)
    return cv2.waitKey(1) & 0xFF


def _console_key() -> int | None:
    """Read a Windows console key without blocking when preview is disabled."""
    if os.name != "nt":
        return None
    try:
        import msvcrt

        if not msvcrt.kbhit():
            return None
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            return None
        return ord(key) if key else None
    except (ImportError, OSError):
        return None


def _write_manifest(session_root: Path, manifest: dict[str, Any]) -> None:
    (session_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _video_capture_options(capture: dict[str, Any]) -> dict[str, Any]:
    """Return continuous-video controls without inheriting photo settings."""

    video = capture.get("video_mode")
    if not isinstance(video, dict) or not bool(video.get("enabled", False)):
        raise ValueError("capture.video_mode must be enabled for continuous video")
    options = dict(capture)
    options.update(video)
    options.pop("video_mode", None)
    options.pop("photo_mode", None)
    options["color_formats"] = list(GEMINI305_COLOR_FORMAT_PRIORITY)
    options.pop("depth_format", None)
    required = {
        "color_auto_exposure": True,
        "color_exposure_us": None,
        "diagnostic_unrestricted_auto_exposure": False,
        "color_gain": None,
        "color_auto_white_balance": True,
        "color_white_balance": None,
        "lock_color_controls_after_warmup": True,
        "require_locked_control_metadata": True,
        "lock_white_balance_after_warmup": False,
        "require_locked_white_balance_metadata": False,
    }
    for name, expected in required.items():
        if options.get(name) != expected:
            raise ValueError(
                "Continuous video requires automatic exposure, gain, and white "
                "balance warmup followed by a full manual control lock: "
                f"{name} must be {expected!r}"
            )
    for name in ("trigger_out_delay_us", "trigger_to_image_delay_us"):
        delay_us = options.get(name)
        if (
            isinstance(delay_us, bool)
            or not isinstance(delay_us, int)
            or delay_us < 0
        ):
            raise ValueError(
                f"capture.video_mode.{name} must be a non-negative integer"
            )
    if int(options.get("warmup_frames", 30)) <= 0:
        raise ValueError("warmup_frames must be positive")
    return options


def _validate_video_auto_control_args(args: argparse.Namespace) -> None:
    """Prevent CLI overrides from turning continuous video into a manual mode."""

    incompatible: list[str] = []
    if bool(getattr(args, "auto_exposure", False)):
        incompatible.append("--auto-exposure")
    if bool(getattr(args, "diagnostic_unrestricted_auto_exposure", False)):
        incompatible.append("--diagnostic-unrestricted-auto-exposure")
    if getattr(args, "exposure_us", None) is not None:
        incompatible.append("--exposure-us")
    if getattr(args, "gain", None) is not None:
        incompatible.append("--gain")
    if getattr(args, "white_balance", None) is not None:
        incompatible.append("--white-balance")
    if incompatible:
        raise ValueError(
            "Continuous video always uses automatic exposure/shutter, gain, and "
            "white balance; it rejects: "
            + ", ".join(incompatible)
        )


def run_capture(args: argparse.Namespace) -> Path:
    if bool(getattr(args, "photo_mode", False)):
        # Keep the photo controller and SDK state machine lazy. photo_capture
        # imports shared encoding/session helpers from this module.
        from .photo_capture import run_photo_sequence

        return run_photo_sequence(args)

    return run_video_capture(args)


def _discover_video_device(
    sdk: Any,
    *,
    wait_for_camera: bool,
    poll_interval_seconds: float = 0.5,
    cancel_event: threading.Event | None = None,
) -> tuple[Any, Any]:
    """Return the first camera, optionally waiting for USB hot-plug."""

    context = sdk.Context()
    waiting = False
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise CaptureCancelledError("Camera discovery was cancelled")
        try:
            device_list = context.query_devices()
            if device_list.get_count() > 0:
                device = device_list.get_device_by_index(0)
                if waiting:
                    print("Orbbec camera detected; starting live capture.", flush=True)
                return context, device
        except Exception as exc:
            if not wait_for_camera or not _is_retryable_video_transport_error(exc):
                raise
            if not waiting:
                print(
                    "Orbbec camera was enumerated but is not ready; waiting for "
                    "USB transport recovery (press Ctrl+C to cancel)...",
                    flush=True,
                )
                waiting = True
            context = sdk.Context()
        if not wait_for_camera:
            raise RuntimeError("No Orbbec camera found")
        if not waiting:
            print(
                "No Orbbec camera found; waiting for connection "
                "(press Ctrl+C to cancel)...",
                flush=True,
            )
            waiting = True
        if cancel_event is not None:
            if cancel_event.wait(poll_interval_seconds):
                raise CaptureCancelledError("Camera discovery was cancelled")
        else:
            time.sleep(poll_interval_seconds)


def run_video_capture(
    args: argparse.Namespace,
    *,
    observer: CaptureObserver | None = None,
) -> Path:
    """Capture one continuous RGB-D session with an optional non-blocking observer."""

    config_file = load_config(args.config)
    capture_config = config_file.get("capture", {})
    if not isinstance(capture_config, dict):
        raise ValueError("Configuration is missing capture settings")
    stitch_config = config_file.get("stitch", {})
    if not isinstance(stitch_config, dict):
        raise ValueError("Configuration stitch section must be a mapping")
    video_runtime_config = stitch_config.get("video_runtime", {})
    if not isinstance(video_runtime_config, dict):
        raise ValueError("stitch.video_runtime must be a mapping")
    fixed_video_exposure = getattr(args, "video_exposure_us", None)
    if fixed_video_exposure is None:
        _validate_video_auto_control_args(args)
    elif bool(getattr(args, "photo_mode", False)):
        raise ValueError("--video-exposure-us cannot be used with --photo-mode")
    options = _video_capture_options(capture_config)
    online_scan = OnlineScanAccumulator(
        analysis_width=int(stitch_config.get("analysis_width", 320)),
        motion_backend=str(video_runtime_config.get("motion_backend", "dis")),
    )
    if fixed_video_exposure is not None:
        options["color_auto_exposure"] = False
        options["color_exposure_us"] = int(fixed_video_exposure)
        options["diagnostic_unrestricted_auto_exposure"] = False
    for name in ("width", "height", "fps", "warmup_frames", "queue_size"):
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    for argument, option in (
        ("trigger_to_image_delay_us", "trigger_to_image_delay_us"),
        ("trigger_out_delay_us", "trigger_out_delay_us"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            options[option] = value
    for name in ("trigger_out_delay_us", "trigger_to_image_delay_us"):
        value = options.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if int(options.get("warmup_frames", 30)) <= 0:
        raise ValueError("warmup_frames must be positive")
    _apply_color_exposure_mode(options, args)
    if args.gain is not None:
        options["color_gain"] = args.gain
    if args.white_balance is not None:
        options["color_white_balance"] = args.white_balance
        options["color_auto_white_balance"] = False
    options["color_auto_white_balance"] = _uses_color_auto_white_balance(options)
    options["preview"] = bool(options.get("preview", True)) and not args.no_preview
    raw_depth_override = getattr(args, "raw_depth", None)
    options["save_raw_depth"] = (
        bool(options.get("save_raw_depth", False))
        if raw_depth_override is None
        else bool(raw_depth_override)
    )
    align_mode = str(options.get("align", "software")).strip().lower()
    if align_mode != "software":
        raise ValueError(
            f"Unsupported align mode {align_mode!r}; this capture command currently supports 'software'"
        )
    options["align"] = align_mode
    if not _uses_color_auto_exposure(options):
        _color_exposure_units(options.get("color_exposure_us"))
    color_control_lock_requested = bool(
        options.get("lock_color_controls_after_warmup", False)
    )
    white_balance_lock_requested = bool(
        options.get("lock_white_balance_after_warmup", False)
    )
    if color_control_lock_requested and white_balance_lock_requested:
        raise ValueError(
            "Continuous video cannot request both full color-control and "
            "white-balance-only post-warmup locks"
        )
    if (color_control_lock_requested or white_balance_lock_requested) and int(
        options.get("post_lock_verified_frames", 2)
    ) <= 0:
        raise ValueError("post_lock_verified_frames must be positive")

    try:
        import pyorbbecsdk as sdk
    except ImportError as exc:
        raise RuntimeError(
            "pyorbbecsdk2 is not installed. Run: python -m pip install pyorbbecsdk2"
        ) from exc

    wait_for_camera = bool(getattr(args, "wait_for_camera", False))
    cancel_event = getattr(args, "cancel_event", None)
    if cancel_event is not None and not isinstance(cancel_event, threading.Event):
        raise ValueError("cancel_event must be a threading.Event")
    _device_context: Any | None = None
    device: Any | None = None
    if wait_for_camera:
        # Do not create an empty session directory while a live command is
        # merely waiting for the camera to be connected.
        _device_context, device = _discover_video_device(
            sdk,
            wait_for_camera=True,
            cancel_event=cancel_event,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = (args.output / f"run_{timestamp}").resolve()
    session_root.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema": "panorama-demo-session/v2",
        "started_utc": started_utc,
        "clean_shutdown": False,
        "capture_mode": ("continuous_rgbd_video_fixed_exposure" if fixed_video_exposure is not None else "continuous_rgbd_video_auto"),
        "diagnostic_only": True,
        "formal_stitch_allowed": False,
        "product_eligibility": {"photo_panorama": False, "video_panorama": False},
        "capture_options": options,
    }
    _write_manifest(session_root, manifest)

    if device is None:
        _device_context, device = _discover_video_device(
            sdk,
            wait_for_camera=False,
            cancel_event=cancel_event,
        )
    manifest["device"] = _device_info(device)
    try:
        manifest["sdk_version"] = sdk.get_version()
    except Exception:
        manifest["sdk_version"] = None
    try:
        manifest["python_wrapper_version"] = importlib_metadata.version("pyorbbecsdk2")
    except importlib_metadata.PackageNotFoundError:
        manifest["python_wrapper_version"] = None
    try:
        applied_color_properties = _configure_color(device, sdk, options)
    except Exception as exc:
        manifest.update(
            {
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "capture_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        _write_manifest(session_root, manifest)
        raise
    manifest["applied_color_properties"] = applied_color_properties
    manifest["color_exposure_control"] = _color_exposure_control_summary(
        options, applied_color_properties
    )
    external_sync_output: dict[str, Any] = {
        "enabled": bool(options.get("external_sync_output", True)),
        "state": "not_configured",
    }
    manifest["external_sync_output"] = external_sync_output
    _write_manifest(session_root, manifest)

    # Keep profile selection, property writes, and streaming on the same device.
    pipeline = sdk.Pipeline(device)
    stream_config = sdk.Config()
    color_list = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
    depth_list = pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR)
    try:
        color_formats = [getattr(sdk.OBFormat, name) for name in options["color_formats"]]
    except AttributeError as exc:
        raise ValueError(f"Unknown Orbbec color format in {options['color_formats']!r}") from exc
    color_profile = _choose_profile(
        color_list, options["width"], options["height"], options["fps"], color_formats, "color"
    )
    depth_profile = _choose_profile(
        depth_list,
        options["width"],
        options["height"],
        options["fps"],
        [sdk.OBFormat.Y16],
        "depth",
    )
    stream_config.enable_stream(color_profile)
    stream_config.enable_stream(depth_profile)
    stream_config.set_frame_aggregate_output_mode(sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
    manifest["profiles"] = {
        "color": _profile_dict(color_profile),
        "depth": _profile_dict(depth_profile),
    }
    manifest["available_profiles"] = {
        "color": _available_profiles(color_list),
        "depth": _available_profiles(depth_list),
    }

    if options.get("frame_sync", True):
        try:
            pipeline.enable_frame_sync()
            manifest["frame_sync"] = True
        except Exception as exc:
            manifest["frame_sync"] = False
            manifest["frame_sync_warning"] = str(exc)
    align_filter = sdk.AlignFilter(align_to_stream=sdk.OBStreamType.COLOR_STREAM)
    diagnostic_online_orb = bool(getattr(args, "diagnostic_online_orbslam3", False))
    online_orb_tracker: Any | None = None
    online_orb_source_type: Any | None = None
    observer_error: str | None = None
    capture_runtime_audit = {
        "online_orb_tracker_construct_count": 0,
        "capture_orbslam3_process_count": 0,
        "capture_orbslam3_frame_submit_count": 0,
        "capture_open3d_import_count": 0,
        "capture_open3d_tsdf_call_count": 0,
        "capture_3d_process_count": 0,
    }

    def _enqueue_online_orb(item: WrittenRGBDFrame) -> None:
        nonlocal observer_error
        tracker = online_orb_tracker
        if tracker is not None:
            if online_orb_source_type is None:
                raise RuntimeError("Diagnostic online ORB source type is unavailable")
            tracker.submit_committed(
                online_orb_source_type(
                    frame=RGBDFrame(
                        frame_id=item.frame_id,
                        color_path=item.color_path,
                        aligned_depth_path=item.aligned_depth_path,
                        depth_scale_mm_per_unit=item.depth_scale_mm_per_unit,
                        timestamp_us=item.timestamp_us,
                    ),
                    color_sha256=item.color_sha256,
                    aligned_depth_sha256=item.aligned_depth_sha256,
                )
            )
            capture_runtime_audit["capture_orbslam3_frame_submit_count"] += 1
        if observer is not None:
            try:
                observer.on_frame_committed(item)
            except Exception as exc:
                observer_error = f"{type(exc).__name__}: {exc}"

    writer = SessionWriter(
        session_root,
        queue_size=int(options["queue_size"]),
        jpeg_quality=int(options["jpeg_quality"]),
        depth_png_compression=int(options["depth_png_compression"]),
        save_raw_depth=bool(options["save_raw_depth"]),
        # Continuous video is append-only and published only after its final
        # clean-shutdown manifest.  Avoid two fsync calls per 60-FPS frame;
        # the writer flushes the CSV in bounded batches and on close.
        durable_per_frame=False,
        csv_flush_interval=30,
        on_written=_enqueue_online_orb,
    )

    received = 0
    online_scan_error: str | None = None
    timestamp_regressions = 0
    previous_color_timestamp: int | None = None
    started_monotonic = 0.0
    capture_started_monotonic_ns = 0
    capture_stopped_monotonic_ns = 0
    metadata_checked = False
    exposure_violation_run = 0
    warmup_exposure_fallback: dict[str, Any] | None = None
    metadata_types: Any | None = None
    color_control_lock: dict[str, Any] | None = None
    pipeline_started = False
    capture_exception: Exception | None = None
    shutdown_errors: list[dict[str, str]] = []
    try:
        try:
            metadata_types = sdk.OBFrameMetadataType
        except AttributeError as exc:
            if color_control_lock_requested or white_balance_lock_requested:
                raise RuntimeError(
                    "Post-warmup color-control lock requires the SDK frame metadata API"
                ) from exc
            raise
        external_sync_output = _configure_external_sync_output(device, sdk, options)
        manifest["external_sync_output"] = external_sync_output
        _write_manifest(session_root, manifest)
        pipeline.start(stream_config)
        pipeline_started = True

        warmup_requested_exposure_options = dict(options)

        def record_warmup_fallback(audit: dict[str, Any]) -> None:
            manifest["capture_mode"] = "continuous_rgbd_video_fixed_exposure"
            manifest["color_control_policy"] = (
                "auto_warmup_then_800us_fallback_full_manual_lock"
            )
            manifest["warmup_exposure_fallback"] = dict(audit)
            manifest["applied_color_properties"].update(
                {
                    "auto_exposure": audit["auto_exposure"],
                    "exposure": audit["exposure"],
                    "exposure_us": audit["exposure_us"],
                    "gain": audit["gain"],
                }
            )
            manifest["color_exposure_control"] = _color_exposure_control_summary(
                warmup_requested_exposure_options,
                manifest["applied_color_properties"],
            )
            _write_manifest(session_root, manifest)

        manifest["color_control_policy"] = (
            "fixed_exposure_auto_gain_awb_warmup_then_full_manual_lock"
            if fixed_video_exposure is not None
            else "auto_warmup_then_full_manual_lock"
        )
        manifest["warmup_exposure_fallback"] = None
        warmup_result = _warm_up_video_controls(
            device,
            sdk,
            pipeline,
            metadata_types,
            options,
            on_fallback=record_warmup_fallback,
        )
        warmup_received = int(warmup_result["warmup_frame_sets"])
        warmup_exposure_fallback = warmup_result["warmup_exposure_fallback"]
        manifest.update(warmup_result)
        _write_manifest(session_root, manifest)
        if color_control_lock_requested or white_balance_lock_requested:
            manifest["color_control_lock"] = {
                "requested": True,
                "completed": False,
                "state": "locking_after_warmup",
                "warmup_frame_sets": warmup_received,
            }
            _write_manifest(session_root, manifest)
            lock_function = (
                _lock_color_controls_after_warmup
                if color_control_lock_requested
                else _lock_white_balance_after_warmup
            )
            color_control_lock = lock_function(
                device, sdk, options, warmup_frame_sets=warmup_received
            )
            manifest["color_control_lock"] = color_control_lock
            _write_manifest(session_root, manifest)
            _discard_and_verify_post_lock_frames(
                pipeline,
                metadata_types,
                color_control_lock,
                timeout_seconds=float(options.get("frame_timeout_seconds", 5)),
            )
            manifest["color_control_lock"] = color_control_lock
            _write_manifest(session_root, manifest)
        started_monotonic = time.monotonic()
        capture_started_monotonic_ns = time.monotonic_ns()
        last_frame_monotonic = started_monotonic
        manifest["calibration"] = _calibration_to_dict(pipeline.get_camera_param())
        (session_root / "calibration.json").write_text(
            json.dumps(manifest["calibration"], indent=2), encoding="utf-8"
        )
        if observer is not None:
            observer.on_session_ready(LiveSessionInfo(
                root=session_root,
                capture_started_monotonic_ns=capture_started_monotonic_ns,
                calibration=dict(manifest["calibration"]),
            ))
        if diagnostic_online_orb:
            try:
                from .video_online_orb import OnlineORBSource, OnlineORBTracker

                calibration = _parse_color_intrinsics_for_online_orb(manifest["calibration"])
                online_orb_source_type = OnlineORBSource
                online_orb_tracker = OnlineORBTracker(
                    intrinsics=calibration,
                    tracking_fps=float(video_runtime_config.get("tracking_fps", 8.0)),
                    work_dir=session_root / ".online_orbslam3_work",
                    config=dict(stitch_config.get("orbslam3_rgbd", {})),
                )
                capture_runtime_audit["online_orb_tracker_construct_count"] += 1
                capture_runtime_audit["capture_orbslam3_process_count"] += 1
                manifest["online_orbslam3_trajectory"] = {
                    "state": "tracking",
                    "path": "online_orbslam3_trajectory.json",
                    "source_selection": "real_timestamp_spaced_capture_frames_only",
                }
            except Exception as exc:
                manifest["online_orbslam3_trajectory"] = {
                    "state": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
        else:
            manifest["online_orbslam3_trajectory"] = {
                "state": "deferred",
                "reason": "post_2d_publication_only",
            }

        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise CaptureCancelledError("Video capture was cancelled")
            writer.raise_if_failed()
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                if time.monotonic() - last_frame_monotonic >= float(
                    options.get("frame_timeout_seconds", 5)
                ):
                    raise RuntimeError(
                        "No complete RGB-D frame set arrived before the capture timeout"
                    )
                continue
            raw_color = frames.get_color_frame()
            raw_depth = frames.get_depth_frame()
            if raw_color is None or raw_depth is None:
                continue
            last_frame_monotonic = time.monotonic()
            color_timestamp = int(raw_color.get_timestamp_us())
            depth_timestamp = int(raw_depth.get_timestamp_us())
            if previous_color_timestamp is not None and color_timestamp <= previous_color_timestamp:
                timestamp_regressions += 1
            previous_color_timestamp = color_timestamp
            assert metadata_types is not None
            color_controls = _color_control_metadata(raw_color, metadata_types)
            if color_control_lock is not None:
                _require_valid_locked_color_metadata(color_controls, color_control_lock)
            color_exposure = color_controls["color_exposure"]
            exposure_violation = _color_exposure_metadata_violation(
                options, color_exposure
            )
            exposure_violation_run = (
                exposure_violation_run + 1 if exposure_violation is not None else 0
            )
            if warmup_exposure_fallback and exposure_violation is not None:
                raise RuntimeError(exposure_violation)
            if exposure_violation_run >= 3:
                assert exposure_violation is not None
                raise RuntimeError(
                    exposure_violation + " for three consecutive frames"
                )

            if not metadata_checked:
                manifest["metadata_support"] = {
                    "color_frame_number": (
                        _metadata(raw_color, metadata_types.FRAME_NUMBER) is not None
                    ),
                    "depth_frame_number": (
                        _metadata(raw_depth, metadata_types.FRAME_NUMBER) is not None
                    ),
                    "color_sensor_timestamp": (
                        _metadata(raw_color, metadata_types.SENSOR_TIMESTAMP) is not None
                    ),
                    "depth_sensor_timestamp": (
                        _metadata(raw_depth, metadata_types.SENSOR_TIMESTAMP) is not None
                    ),
                    "color_exposure": color_exposure is not None,
                    "depth_exposure": _metadata(raw_depth, metadata_types.EXPOSURE) is not None,
                    "color_gain": color_controls["color_gain"] is not None,
                    "color_auto_white_balance": (
                        color_controls["color_auto_white_balance"] is not None
                    ),
                    "color_white_balance": (
                        color_controls["color_white_balance"] is not None
                    ),
                }
                metadata_checked = True
                if not any(manifest["metadata_support"].values()):
                    platform_hint = (
                        "verify the Orbbec SDK metadata registration on Windows"
                        if os.name == "nt"
                        else "verify the installed Orbbec SDK and non-root udev rules"
                    )
                    print(
                        "Metadata warning: no tested frame metadata is available; "
                        + platform_hint
                        + ".",
                        file=sys.stderr,
                    )

            aligned_frames = align_filter.process(frames)
            if aligned_frames is None:
                continue
            aligned_color = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if aligned_color is None or aligned_depth_frame is None:
                continue

            if external_sync_output.get("enabled") is True:
                _verify_external_sync_output(device, sdk, options)

            color_image = _frame_to_bgr(aligned_color, sdk).copy()
            aligned_depth = _depth_array(aligned_depth_frame)
            raw_depth_array = _depth_array(raw_depth) if options["save_raw_depth"] else None
            depth_scale = float(aligned_depth_frame.get_depth_scale())
            packet_metadata = {
                "color_index": int(raw_color.get_index()),
                "depth_index": int(raw_depth.get_index()),
                "color_device_timestamp_us": color_timestamp,
                "depth_device_timestamp_us": depth_timestamp,
                "sync_delta_us": color_timestamp - depth_timestamp,
                "color_system_timestamp_us": int(raw_color.get_system_timestamp_us()),
                "depth_system_timestamp_us": int(raw_depth.get_system_timestamp_us()),
                "host_timestamp_ns": time.time_ns(),
                "color_exposure": color_exposure,
                "depth_exposure": _metadata(raw_depth, metadata_types.EXPOSURE),
                "color_gain": color_controls["color_gain"],
                "depth_gain": _metadata(raw_depth, metadata_types.GAIN),
                "color_auto_white_balance": color_controls[
                    "color_auto_white_balance"
                ],
                "color_white_balance": color_controls["color_white_balance"],
                "color_frame_number": _metadata(raw_color, metadata_types.FRAME_NUMBER),
                "depth_frame_number": _metadata(raw_depth, metadata_types.FRAME_NUMBER),
                "color_sensor_timestamp_raw": _metadata(
                    raw_color, metadata_types.SENSOR_TIMESTAMP
                ),
                "depth_sensor_timestamp_raw": _metadata(
                    raw_depth, metadata_types.SENSOR_TIMESTAMP
                ),
                "depth_scale_mm_per_unit": depth_scale,
            }
            color_image.setflags(write=False)
            aligned_depth.setflags(write=False)
            if raw_depth_array is not None:
                raw_depth_array.setflags(write=False)
            accepted_monotonic_ns = time.monotonic_ns()
            accepted_for_write = writer.submit(
                FramePacket(received, color_image, aligned_depth, raw_depth_array, packet_metadata)
            )
            if not accepted_for_write:
                raise RuntimeError(
                    "RGB-D session writer queue is full; capture stopped before "
                    "the session could lose additional frames"
                )
            if accepted_for_write and observer is not None:
                try:
                    observer.on_frame_accepted(LiveFramePacket(
                        frame_id=received,
                        color_bgr=color_image,
                        aligned_depth_mm=aligned_depth,
                        metadata=packet_metadata,
                        accepted_monotonic_ns=accepted_monotonic_ns,
                        writer_queue_fraction=(
                            writer.queue.qsize() / max(1, int(options["queue_size"]))
                        ),
                        writer_queue_drops=writer.stats.queue_drops,
                    ))
                except Exception as exc:
                    observer_error = f"{type(exc).__name__}: {exc}"
            if accepted_for_write and observer is None and online_scan_error is None:
                if (
                    color_image.dtype != np.uint8
                    or color_image.shape != (int(options["height"]), int(options["width"]), 3)
                    or aligned_depth.dtype != np.uint16
                    or aligned_depth.shape != color_image.shape[:2]
                    or not np.any(aligned_depth > 0)
                ):
                    raise RuntimeError(
                        "Continuous video writer received an invalid aligned RGB-D frame"
                    )
                # This is deliberately capture-time work: it consumes the
                # already aligned colour buffer before the next frame arrives.
                try:
                    online_scan.add(received, color_image)
                except Exception as exc:
                    # The capture itself remains usable and will fall back to
                    # the offline analyzer, but never publishes a partial
                    # online state as though it were complete.
                    online_scan_error = f"{type(exc).__name__}: {exc}"
            received += 1
            if external_sync_output.get("enabled") is True:
                external_sync_output["per_frame_readback_verified_frames"] += 1

            if options["preview"]:
                panorama_preview = None
                if observer is not None:
                    preview_provider = getattr(observer, "capture_preview_image", None)
                    if callable(preview_provider):
                        panorama_preview = preview_provider()
                key = _preview(
                    color_image,
                    aligned_depth,
                    depth_scale,
                    panorama_preview,
                )
                if key in (ord("q"), ord("Q"), 27):
                    break
            elif _console_key() in (ord("q"), ord("Q"), 27):
                break
            if args.max_frames and received >= args.max_frames:
                break
            if args.duration and time.monotonic() - started_monotonic >= args.duration:
                break
            if received % 30 == 0:
                print(
                    f"\rreceived={received} written={writer.stats.written} "
                    f"queue={writer.queue.qsize()} drops={writer.stats.queue_drops}",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        manifest["stop_reason"] = "keyboard_interrupt"
    except Exception as exc:
        capture_exception = exc
    finally:
        capture_stopped_monotonic_ns = time.monotonic_ns()
        if observer is not None:
            try:
                observer.on_capture_stopping()
            except Exception as exc:
                observer_error = f"{type(exc).__name__}: {exc}"
        hardware_shutdown, hardware_shutdown_errors = _shutdown_video_capture_hardware(
            pipeline,
            pipeline_started=pipeline_started,
            device=device,
            sdk=sdk,
            external_sync_output=external_sync_output,
        )
        shutdown_errors.extend(hardware_shutdown_errors)
        external_sync_output["shutdown"] = hardware_shutdown
        manifest["external_sync_output"] = external_sync_output
        try:
            writer.close()
        except Exception as exc:
            shutdown_errors.append(
                {"step": "writer.close", "type": type(exc).__name__, "message": str(exc)}
            )
        try:
            writer.raise_if_failed()
        except SessionWriterError as exc:
            if capture_exception is None:
                capture_exception = exc
            else:
                shutdown_errors.append(
                    {
                        "step": "writer.raise_if_failed",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        online_orb_result = None
        if online_orb_tracker is not None:
            try:
                online_orb_result = online_orb_tracker.close()
            except Exception as exc:
                shutdown_errors.append(
                    {
                        "step": "online_orb_tracker.close",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        if options["preview"]:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                shutdown_errors.append(
                    {
                        "step": "cv2.destroyAllWindows",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        capture_exception = _capture_exception_after_shutdown(
            capture_exception, shutdown_errors
        )
        print()

    clean_shutdown = _video_capture_clean_shutdown(
        capture_exception, writer.stats, received, external_sync_output
    )
    manifest.update(
        {
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "clean_shutdown": clean_shutdown,
            "received_frames": received,
            "written_frames": writer.stats.written,
            "queue_drops": writer.stats.queue_drops,
            "write_errors": writer.stats.write_errors,
            "writer_errors": writer.stats.errors,
            "max_queue_depth": writer.stats.max_queue_depth,
            "timestamp_regressions": timestamp_regressions,
            "capture_runtime_audit": capture_runtime_audit,
        }
    )
    if capture_exception is not None:
        manifest["capture_error"] = {
            "type": type(capture_exception).__name__,
            "message": str(capture_exception),
        }
    if shutdown_errors:
        manifest["shutdown_errors"] = shutdown_errors
    if observer_error is not None:
        manifest["live_observer_warning"] = observer_error
    manifest["product_eligibility"] = {
        "photo_panorama": False,
        "video_panorama": bool(
            capture_exception is None
            and manifest["clean_shutdown"] is True
            and writer.stats.write_errors == 0
            and writer.stats.errors == []
            and writer.stats.queue_drops == 0
            and writer.stats.written == received
        ),
    }
    if online_orb_tracker is not None:
        if online_orb_result is not None and online_orb_tracker.error is None:
            manifest["online_orbslam3_trajectory"] = {
                "state": "available",
                "path": "online_orbslam3_trajectory.json",
                "schema": online_orb_result["schema"],
                "tracked_frames": len(online_orb_result["tracked_frame_ids"]),
                "source_selection": "real_timestamp_spaced_capture_frames_only",
            }
        else:
            manifest["online_orbslam3_trajectory"] = {
                "state": "unavailable",
                "reason": online_orb_tracker.error or "no online ORB sources were committed",
            }
    online_state_inputs: tuple[
        list[RGBDFrame], list[dict[str, object]], list[Any], list[Any], dict[str, object]
    ] | None = None
    if (
        capture_exception is None
        and writer.stats.write_errors == 0
        and writer.stats.errors == []
        and online_scan_error is None
    ):
        try:
            qualities, motions, segment = online_scan.finish()
            written = list(writer.written_rgbd_frames)
            if list(online_scan.frame_ids) != [item.frame_id for item in written]:
                raise RuntimeError("online scan and disk writer accepted different frame sets")
            frames_for_state = [
                RGBDFrame(
                    frame_id=item.frame_id,
                    color_path=item.color_path,
                    aligned_depth_path=item.aligned_depth_path,
                    depth_scale_mm_per_unit=1.0,
                )
                for item in written
            ]
            file_hashes = [
                {
                    "frame_id": item.frame_id,
                    "color_sha256": item.color_sha256,
                    "aligned_depth_sha256": item.aligned_depth_sha256,
                }
                for item in written
            ]
            online_state_inputs = (
                frames_for_state,
                file_hashes,
                qualities,
                motions,
                segment,
            )
            manifest["online_video_state"] = {
                "schema": "gemini305-video-online-state/v1",
                "path": "online_video_state.json",
                "origin": "capture",
                "state": "available",
                "analysis_input": "aligned_capture_bgr_before_jpeg_encode",
                "frames": len(frames_for_state),
            }
        except Exception as exc:
            online_scan_error = f"{type(exc).__name__}: {exc}"
    if online_state_inputs is None:
        manifest["online_video_state"] = {
            "state": "unavailable",
            "reason": online_scan_error or "capture_or_writer_failed",
        }
    if capture_exception is not None:
        manifest["capture_error"] = {
            "type": type(capture_exception).__name__,
            "message": str(capture_exception),
        }
    _write_manifest(session_root, manifest)
    if online_orb_result is not None and online_orb_tracker is not None and online_orb_tracker.error is None:
        try:
            online_orb_result["input_sha256"] = {
                "manifest": _sha256_file(session_root / "manifest.json"),
                "calibration": _sha256_file(session_root / "calibration.json"),
                "frames_csv": _sha256_file(session_root / "frames.csv"),
            }
            _write_json_atomic(session_root / "online_orbslam3_trajectory.json", online_orb_result)
        except Exception as exc:
            manifest["online_orbslam3_trajectory"] = {
                "state": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            _write_manifest(session_root, manifest)
    if online_state_inputs is not None:
        frames_for_state, file_hashes, qualities, motions, segment = online_state_inputs
        try:
            write_online_state(
                session_root / "online_video_state.json",
                root=session_root,
                frames=frames_for_state,
                qualities=qualities,
                motions=motions,
                segment=segment,
                origin="capture",
                frame_file_sha256=file_hashes,
                capture_frame_validation={
                    "schema": CAPTURE_FRAME_VALIDATION_SCHEMA,
                    "frame_count": len(frames_for_state),
                    "color_dtype": "uint8",
                    "aligned_depth_dtype": "uint16",
                    "color_shape": [int(options["height"]), int(options["width"]), 3],
                    "aligned_depth_shape": [int(options["height"]), int(options["width"])],
                    "source_hash_provenance": "writer_encoded_bytes_before_atomic_replace",
                },
            )
        except Exception as exc:
            manifest["online_video_state"] = {
                "state": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
            _write_manifest(session_root, manifest)
    capture_result = CaptureResult(
        session_root=session_root,
        received_frames=received,
        written_frames=writer.stats.written,
        queue_drops=writer.stats.queue_drops,
        write_errors=writer.stats.write_errors,
        max_queue_depth=writer.stats.max_queue_depth,
        capture_started_monotonic_ns=capture_started_monotonic_ns,
        capture_stopped_monotonic_ns=capture_stopped_monotonic_ns,
    )
    if observer is not None:
        observer.on_capture_closed(capture_result)
    if capture_exception is not None:
        raise capture_exception
    print(f"Session saved to: {session_root}")
    return session_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture synchronized Gemini 305 RGB-D frames")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/captures"))
    parser.add_argument(
        "--photo-mode",
        action="store_true",
        help=(
            "Use no-preview SOFTWARE_TRIGGERING for an RGB-D photo sequence; "
            "--fps may select an exact supported rate"
        ),
    )
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument(
        "--fps",
        type=int,
        choices=GEMINI305_FRAME_RATES,
        help=(
            "Exact RGB-D frame rate; photo mode otherwise chooses the fastest "
            "compatible rate, video mode uses configuration"
        ),
    )
    parser.add_argument(
        "--image-delay-us",
        "--trigger-to-image-delay-us",
        dest="trigger_to_image_delay_us",
        type=int,
        help=(
            "Trigger-to-image delay in microseconds; defaults to "
            f"{DEFAULT_TRIGGER_TO_IMAGE_DELAY_US}"
        ),
    )
    parser.add_argument(
        "--trigger-out-delay-us",
        type=int,
        help=(
            "Trigger Out delay in microseconds; defaults to "
            f"{DEFAULT_TRIGGER_OUT_DELAY_US}"
        ),
    )
    parser.add_argument("--warmup-frames", type=int)
    parser.add_argument("--queue-size", type=int)
    exposure = parser.add_mutually_exclusive_group()
    exposure.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Deprecated for continuous video; automatic controls are always enabled",
    )
    parser.add_argument(
        "--video-exposure-us",
        type=int,
        help=(
            "Fixed continuous-video exposure from startup; gain and white balance "
            "warm up automatically, then exposure, gain, and white balance are "
            "locked before formal frames"
        ),
    )
    exposure.add_argument(
        "--diagnostic-unrestricted-auto-exposure",
        action="store_true",
        help=(
            "Deprecated for continuous video; it already uses unrestricted "
            "automatic exposure and cannot publish an official panorama"
        ),
    )
    exposure.add_argument(
        "--exposure-us",
        "--manual-exposure-us",
        dest="exposure_us",
        type=int,
        help=(
            "Unsupported for continuous video, which always uses automatic "
            "exposure/shutter"
        ),
    )
    parser.add_argument("--gain", type=int)
    parser.add_argument("--white-balance", type=int)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--no-wait-for-camera",
        dest="wait_for_camera",
        action="store_false",
        help="Fail immediately instead of polling until an Orbbec camera is connected",
    )
    parser.add_argument(
        "--diagnostic-online-orbslam3",
        action="store_true",
        help="Diagnostic-only capture-time ORB-SLAM3; forbidden by g305-video-live",
    )
    raw_depth = parser.add_mutually_exclusive_group()
    raw_depth.add_argument(
        "--raw-depth",
        dest="raw_depth",
        action="store_true",
        help="Also save native depth PNGs (higher CPU/disk load)",
    )
    raw_depth.add_argument(
        "--no-raw-depth",
        dest="raw_depth",
        action="store_false",
        help="Save aligned depth only (the default demo mode)",
    )
    parser.set_defaults(raw_depth=None, wait_for_camera=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        run_capture(args)
    except Exception as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
