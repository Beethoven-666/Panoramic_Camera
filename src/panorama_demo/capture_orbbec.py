from __future__ import annotations

import argparse
import csv
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
from typing import Any

import cv2
import numpy as np

from .config import load_config


COLOR_EXPOSURE_UNIT_US = 100


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


@dataclass
class WriterStats:
    submitted: int = 0
    written: int = 0
    queue_drops: int = 0
    write_errors: int = 0
    max_queue_depth: int = 0
    errors: list[str] = field(default_factory=list)


def _atomic_encode(path: Path, extension: str, image: np.ndarray, params: list[int]) -> None:
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise IOError(f"OpenCV could not encode {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        handle.write(encoded.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class SessionWriter:
    def __init__(
        self,
        root: Path,
        queue_size: int,
        jpeg_quality: int,
        depth_png_compression: int,
        save_raw_depth: bool,
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
        self.queue: queue.Queue[FramePacket | None] = queue.Queue(maxsize=queue_size)
        self.stats = WriterStats()
        self._thread = threading.Thread(target=self._run, name="rgbd-writer", daemon=False)
        self._thread.start()

    def submit(self, packet: FramePacket) -> bool:
        self.stats.submitted += 1
        try:
            self.queue.put_nowait(packet)
        except queue.Full:
            self.stats.queue_drops += 1
            return False
        self.stats.max_queue_depth = max(self.stats.max_queue_depth, self.queue.qsize())
        return True

    def close(self) -> None:
        self.queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        csv_path = self.root / "frames.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while True:
                packet = self.queue.get()
                if packet is None:
                    self.queue.task_done()
                    break
                try:
                    stem = f"{packet.frame_id:08d}"
                    color_relative = Path("color") / f"{stem}.jpg"
                    aligned_relative = Path("depth_aligned") / f"{stem}.png"
                    raw_relative = Path("depth_raw") / f"{stem}.png"
                    _atomic_encode(
                        self.root / color_relative,
                        ".jpg",
                        packet.color_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                    _atomic_encode(
                        self.root / aligned_relative,
                        ".png",
                        packet.aligned_depth,
                        [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression],
                    )
                    raw_value = ""
                    if self.save_raw_depth and packet.raw_depth is not None:
                        _atomic_encode(
                            self.root / raw_relative,
                            ".png",
                            packet.raw_depth,
                            [cv2.IMWRITE_PNG_COMPRESSION, self.depth_png_compression],
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
                    handle.flush()
                    self.stats.written += 1
                except Exception as exc:  # keep capture alive, but report every failure
                    self.stats.write_errors += 1
                    message = f"frame {packet.frame_id}: {exc}"
                    self.stats.errors.append(message)
                    print(f"\nWriter error: {message}", file=sys.stderr)
                finally:
                    self.queue.task_done()


def _frame_to_bgr(frame: Any, sdk: Any) -> np.ndarray:
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    if fmt == sdk.OBFormat.RGB:
        return cv2.cvtColor(data.reshape(height, width, 3), cv2.COLOR_RGB2BGR)
    if fmt == sdk.OBFormat.BGR:
        return data.reshape(height, width, 3).copy()
    if fmt == sdk.OBFormat.YUYV:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == sdk.OBFormat.UYVY:
        return cv2.cvtColor(data.reshape(height, width, 2), cv2.COLOR_YUV2BGR_UYVY)
    if fmt == sdk.OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode MJPG color frame")
        return image
    if fmt == sdk.OBFormat.NV12:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV12)
    if fmt == sdk.OBFormat.NV21:
        return cv2.cvtColor(data.reshape(height * 3 // 2, width), cv2.COLOR_YUV2BGR_NV21)
    if fmt == sdk.OBFormat.I420:
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


def _uses_color_auto_exposure(options: dict[str, Any]) -> bool:
    configured = options.get("color_auto_exposure")
    if configured is None:
        return options.get("color_exposure_us") is None
    return bool(configured)


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
        cap_us = options.get("color_ae_max_exposure_us")
        if cap_us is None:
            return None
        cap_units = _color_exposure_units(cap_us)
        if measured_units > cap_units + 1:
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


def _color_exposure_control_summary(
    options: dict[str, Any], applied: dict[str, Any]
) -> dict[str, Any]:
    requested_auto = _uses_color_auto_exposure(options)
    applied_auto = bool(applied["auto_exposure"])
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
    }


def _configure_color(device: Any, sdk: Any, options: dict[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {}
    exposure = options.get("color_exposure_us")
    auto_exposure = _uses_color_auto_exposure(options)
    ae_max_exposure = options.get("color_ae_max_exposure_us")
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
    if auto_exposure and ae_max_exposure is not None:
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
        raise RuntimeError("The camera did not apply the requested color exposure mode")
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
            raise RuntimeError("The camera did not apply the requested color exposure")
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
    if white_balance is not None:
        applied["auto_white_balance"] = _set_bool_property(
            device, sdk, "OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL", False
        )
        applied["white_balance"] = _set_int_property(
            device, sdk, "OB_PROP_COLOR_WHITE_BALANCE_INT", int(white_balance)
        )
    if options.get("color_anti_flicker", True):
        applied["anti_flicker"] = _set_bool_property(
            device, sdk, "OB_PROP_COLOR_ANTI_FLICKER_BOOL", True
        )
    return applied


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


def _preview(color: np.ndarray, depth: np.ndarray, scale: float) -> int:
    depth_mm = depth.astype(np.float32) * scale
    valid = (depth_mm >= 50.0) & (depth_mm <= 1000.0)
    depth_u8 = np.zeros(depth.shape, dtype=np.uint8)
    depth_u8[valid] = np.clip((depth_mm[valid] - 50.0) * (255.0 / 950.0), 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    if colored.shape[:2] != color.shape[:2]:
        colored = cv2.resize(colored, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
    display = np.hstack((color, colored))
    max_width = 1600
    if display.shape[1] > max_width:
        factor = max_width / display.shape[1]
        display = cv2.resize(display, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
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


def run_capture(args: argparse.Namespace) -> Path:
    config_file = load_config(args.config)
    options = dict(config_file.get("capture", {}))
    for name in ("width", "height", "fps", "warmup_frames", "queue_size"):
        value = getattr(args, name, None)
        if value is not None:
            options[name] = value
    if args.auto_exposure:
        options["color_auto_exposure"] = True
        options["color_exposure_us"] = None
    elif args.exposure_us is not None:
        options["color_auto_exposure"] = False
        options["color_exposure_us"] = args.exposure_us
    if args.gain is not None:
        options["color_gain"] = args.gain
    if args.white_balance is not None:
        options["color_white_balance"] = args.white_balance
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

    try:
        import pyorbbecsdk as sdk
    except ImportError as exc:
        raise RuntimeError(
            "pyorbbecsdk2 is not installed. Run: python -m pip install pyorbbecsdk2"
        ) from exc

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = (args.output / f"run_{timestamp}").resolve()
    session_root.mkdir(parents=True, exist_ok=False)
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "schema": "panorama-demo-session/v1",
        "started_utc": started_utc,
        "clean_shutdown": False,
        "capture_options": options,
    }
    _write_manifest(session_root, manifest)

    context = sdk.Context()
    device_list = context.query_devices()
    if device_list.get_count() == 0:
        raise RuntimeError("No Orbbec camera found")
    device = device_list.get_device_by_index(0)
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
    writer = SessionWriter(
        session_root,
        queue_size=int(options["queue_size"]),
        jpeg_quality=int(options["jpeg_quality"]),
        depth_png_compression=int(options["depth_png_compression"]),
        save_raw_depth=bool(options["save_raw_depth"]),
    )

    received = 0
    timestamp_regressions = 0
    previous_color_timestamp: int | None = None
    started_monotonic = 0.0
    metadata_checked = False
    exposure_violation_run = 0
    metadata_types = sdk.OBFrameMetadataType
    pipeline_started = False
    capture_exception: Exception | None = None
    try:
        pipeline.start(stream_config)
        pipeline_started = True
        warmup_received = 0
        warmup_started = time.monotonic()
        while warmup_received < int(options["warmup_frames"]):
            warmup_frames = pipeline.wait_for_frames(1000)
            if warmup_frames is not None:
                warmup_received += 1
            if time.monotonic() - warmup_started >= float(
                options.get("warmup_timeout_seconds", 15)
            ):
                raise RuntimeError(
                    f"Capture warmup timed out after receiving {warmup_received}/"
                    f"{options['warmup_frames']} complete RGB-D frame sets. "
                    "Try MJPG color, a lower resolution, another USB 3 port, or disable "
                    "other camera applications."
                )
        started_monotonic = time.monotonic()
        last_frame_monotonic = started_monotonic
        manifest["calibration"] = _calibration_to_dict(pipeline.get_camera_param())
        (session_root / "calibration.json").write_text(
            json.dumps(manifest["calibration"], indent=2), encoding="utf-8"
        )

        while True:
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
            color_exposure = _metadata(raw_color, metadata_types.EXPOSURE)
            exposure_violation = _color_exposure_metadata_violation(
                options, color_exposure
            )
            exposure_violation_run = (
                exposure_violation_run + 1 if exposure_violation is not None else 0
            )
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
                    "color_exposure": _metadata(raw_color, metadata_types.EXPOSURE) is not None,
                    "depth_exposure": _metadata(raw_depth, metadata_types.EXPOSURE) is not None,
                }
                metadata_checked = True
                if not any(manifest["metadata_support"].values()):
                    print(
                        "Metadata warning: no tested frame metadata is available; "
                        "run the Orbbec Windows metadata registration script.",
                        file=sys.stderr,
                    )

            aligned_frames = align_filter.process(frames)
            if aligned_frames is None:
                continue
            aligned_color = aligned_frames.get_color_frame()
            aligned_depth_frame = aligned_frames.get_depth_frame()
            if aligned_color is None or aligned_depth_frame is None:
                continue

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
                "color_gain": _metadata(raw_color, metadata_types.GAIN),
                "depth_gain": _metadata(raw_depth, metadata_types.GAIN),
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
            writer.submit(
                FramePacket(received, color_image, aligned_depth, raw_depth_array, packet_metadata)
            )
            received += 1

            if options["preview"]:
                key = _preview(color_image, aligned_depth, depth_scale)
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
        if pipeline_started:
            pipeline.stop()
        writer.close()
        cv2.destroyAllWindows()
        print()

    manifest.update(
        {
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "clean_shutdown": capture_exception is None,
            "received_frames": received,
            "written_frames": writer.stats.written,
            "queue_drops": writer.stats.queue_drops,
            "write_errors": writer.stats.write_errors,
            "writer_errors": writer.stats.errors,
            "max_queue_depth": writer.stats.max_queue_depth,
            "timestamp_regressions": timestamp_regressions,
        }
    )
    if capture_exception is not None:
        manifest["capture_error"] = {
            "type": type(capture_exception).__name__,
            "message": str(capture_exception),
        }
    _write_manifest(session_root, manifest)
    if capture_exception is not None:
        raise capture_exception
    print(f"Session saved to: {session_root}")
    return session_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture synchronized Gemini 305 RGB-D frames")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/captures"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--warmup-frames", type=int)
    parser.add_argument("--queue-size", type=int)
    exposure = parser.add_mutually_exclusive_group()
    exposure.add_argument(
        "--auto-exposure",
        action="store_true",
        help="Enable color auto exposure with the configured motion-safe cap",
    )
    exposure.add_argument(
        "--exposure-us",
        "--manual-exposure-us",
        dest="exposure_us",
        type=int,
        help=(
            "Disable color auto exposure and use this duration in microseconds "
            "(Gemini 305 applies 100 us units; moving-sequence delivery still "
            "enforces its configured exposure limit)"
        ),
    )
    parser.add_argument("--gain", type=int)
    parser.add_argument("--white-balance", type=int)
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-preview", action="store_true")
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
    parser.set_defaults(raw_depth=None)
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
