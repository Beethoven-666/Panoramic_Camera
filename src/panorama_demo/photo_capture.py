from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .capture_orbbec import (
    COLOR_EXPOSURE_UNIT_US,
    FramePacket,
    SessionWriter,
    _calibration_to_dict,
    _console_key,
    _depth_array,
    _device_info,
    _enum_name,
    _frame_to_bgr,
    _metadata,
    _profile_dict,
    _write_manifest,
)
from .config import load_config


MAX_FORMAL_PHOTO_EXPOSURE_US = 800
MIN_GATE_SETTLE_MS = 250
FORMAL_PRIME_ATTEMPTS = 8
FORMAL_TRIGGER_OUT_DELAY_US = 17_000
TRIGGER_DELAY_GUARD_US = 2_000
STALE_DRAIN_LIMIT = 4


class PhotoCaptureError(RuntimeError):
    """Fail-closed software-triggered RGB-D photo error."""


class PhotoCaptureBusyError(PhotoCaptureError):
    """Raised when another accepted photo is still in flight."""


@dataclass(frozen=True)
class PhotoCaptureSettings:
    output_root: Path
    width: int = 1280
    height: int = 800
    color_formats: tuple[str, ...] = ("RGB", "BGR", "YUYV", "MJPG")
    exposure_us: int = MAX_FORMAL_PHOTO_EXPOSURE_US
    trigger_out_delay_us: int = FORMAL_TRIGGER_OUT_DELAY_US
    capture_timeout_ms: int = 8_000
    prime_attempts: int = FORMAL_PRIME_ATTEMPTS
    prime_timeout_ms: int = 1_500
    gate_settle_ms: int = 250
    jpeg_quality: int = 95
    depth_png_compression: int = 1

    def validated(self) -> PhotoCaptureSettings:
        for name in (
            "width",
            "height",
            "exposure_us",
            "trigger_out_delay_us",
            "capture_timeout_ms",
            "prime_attempts",
            "prime_timeout_ms",
            "gate_settle_ms",
            "jpeg_quality",
            "depth_png_compression",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("photo width and height must be positive")
        if not self.color_formats or any(not str(item).strip() for item in self.color_formats):
            raise ValueError("color_formats must contain at least one format")
        if self.exposure_us <= 0 or self.exposure_us % COLOR_EXPOSURE_UNIT_US:
            raise ValueError(
                f"exposure_us must use exact {COLOR_EXPOSURE_UNIT_US} us units"
            )
        if self.exposure_us > MAX_FORMAL_PHOTO_EXPOSURE_US:
            raise ValueError(
                "formal photo exposure cannot exceed "
                f"{MAX_FORMAL_PHOTO_EXPOSURE_US} us"
            )
        if self.trigger_out_delay_us != FORMAL_TRIGGER_OUT_DELAY_US:
            raise ValueError(
                "formal photo sequence requires "
                f"trigger_out_delay_us={FORMAL_TRIGGER_OUT_DELAY_US}"
            )
        if self.capture_timeout_ms <= 0:
            raise ValueError("capture_timeout_ms must be positive")
        if self.prime_attempts != FORMAL_PRIME_ATTEMPTS:
            raise ValueError(
                "formal photo sequence requires exactly "
                f"{FORMAL_PRIME_ATTEMPTS} bounded priming attempts"
            )
        if self.prime_timeout_ms <= 0:
            raise ValueError("photo priming timeout must be positive")
        if self.prime_timeout_ms > self.capture_timeout_ms:
            raise ValueError(
                "prime_timeout_ms cannot exceed capture_timeout_ms"
            )
        if self.gate_settle_ms < MIN_GATE_SETTLE_MS:
            raise ValueError(
                f"gate_settle_ms cannot be below {MIN_GATE_SETTLE_MS} ms"
            )
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be within 1..100")
        if not 0 <= self.depth_png_compression <= 9:
            raise ValueError("depth_png_compression must be within 0..9")
        return self


@dataclass(frozen=True)
class PreparedPhotoCapture:
    session_root: Path
    width: int
    height: int
    fps: int
    color_format: str
    depth_format: str
    exposure_us: int
    trigger_out_delay_us: int
    sync_mode: str = "SOFTWARE_TRIGGERING"
    trigger_out_enable: bool = True
    frames_per_trigger: int = 1

    def __str__(self) -> str:
        return (
            f"{self.width}x{self.height}@{self.fps} {self.color_format}, "
            f"session={self.session_root}"
        )


@dataclass(frozen=True)
class PhotoCaptureResult:
    session_root: Path
    frame_id: int
    color_path: Path
    aligned_depth_path: Path
    frames_csv_path: Path
    color_timestamp_us: int
    depth_timestamp_us: int
    depth_scale_mm_per_unit: float


@dataclass(frozen=True)
class _SyncSnapshot:
    mode: Any
    depth_delay_us: int
    color_delay_us: int
    trigger_to_image_delay_us: int
    trigger_out_enable: bool
    trigger_out_delay_us: int
    frames_per_trigger: int


@dataclass(frozen=True)
class _DeviceSnapshot:
    sync: _SyncSnapshot
    auto_capture: bool
    output_gate: bool
    color_auto_exposure: bool
    color_exposure_raw: int


def _profile_entries(profile_list: Any) -> list[Any]:
    profiles: list[Any] = []
    for index in range(profile_list.get_count()):
        profile = profile_list.get_stream_profile_by_index(index)
        converter = getattr(profile, "as_video_stream_profile", None)
        if callable(converter):
            profile = converter()
        profiles.append(profile)
    return profiles


def _available_video_profiles(profile_list: Any) -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    for profile in _profile_entries(profile_list):
        try:
            available.append(_profile_dict(profile))
        except Exception:
            continue
    return available


def select_fastest_rgbd_profiles(
    color_profile_list: Any,
    depth_profile_list: Any,
    *,
    width: int,
    height: int,
    color_formats: tuple[str, ...],
    sdk: Any,
    trigger_out_delay_us: int = 0,
) -> tuple[Any, Any]:
    """Select the fastest common RGB-D FPS compatible with Trigger Out.

    Color format order only breaks ties at the selected common frame rate. No
    resolution fallback is permitted because it would invalidate calibration
    and the formal RGB-D session contract. A higher FPS is skipped when its
    frame period cannot contain the configured Trigger Out delay plus guard.
    """

    format_rank = {
        name.strip().upper(): rank for rank, name in enumerate(color_formats)
    }
    color_candidates: list[tuple[Any, str, int]] = []
    for profile in _profile_entries(color_profile_list):
        try:
            name = _enum_name(profile.get_format()).upper()
            if (
                int(profile.get_width()) == width
                and int(profile.get_height()) == height
                and name in format_rank
            ):
                color_candidates.append((profile, name, int(profile.get_fps())))
        except Exception:
            continue

    depth_candidates: list[tuple[Any, int]] = []
    y16 = sdk.OBFormat.Y16
    for profile in _profile_entries(depth_profile_list):
        try:
            if (
                int(profile.get_width()) == width
                and int(profile.get_height()) == height
                and profile.get_format() == y16
            ):
                depth_candidates.append((profile, int(profile.get_fps())))
        except Exception:
            continue

    common_fps = sorted(
        {item[2] for item in color_candidates}
        & {item[1] for item in depth_candidates},
        reverse=True,
    )
    if not common_fps:
        raise PhotoCaptureError(
            f"No common color/Y16 depth FPS for {width}x{height}. "
            f"Color profiles: {_available_video_profiles(color_profile_list)}; "
            f"depth profiles: {_available_video_profiles(depth_profile_list)}"
        )
    compatible_fps = [
        fps
        for fps in common_fps
        if trigger_out_delay_us
        <= max(0, round(1_000_000 / fps) - TRIGGER_DELAY_GUARD_US)
    ]
    if not compatible_fps:
        raise PhotoCaptureError(
            f"No common color/Y16 depth FPS for {width}x{height} can safely "
            f"accommodate trigger_out_delay_us={trigger_out_delay_us}. "
            f"Common FPS: {common_fps}"
        )
    fastest = compatible_fps[0]
    color_profile = min(
        (item for item in color_candidates if item[2] == fastest),
        key=lambda item: format_rank[item[1]],
    )[0]
    depth_profile = next(
        profile for profile, fps in depth_candidates if fps == fastest
    )
    return color_profile, depth_profile


class SoftwareTriggeredRGBDPhotoController:
    """Own one Gemini 305 no-preview, one-click/one-trigger RGB-D session."""

    def __init__(
        self,
        settings: PhotoCaptureSettings,
        *,
        sdk: ModuleType | Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings.validated()
        self._sdk = sdk
        self._clock = clock
        self._sleep = sleep
        self._operation_lock = threading.RLock()
        self._capture_lock = threading.Lock()
        self._context: Any | None = None
        self._device: Any | None = None
        self._pipeline: Any | None = None
        self._align_filter: Any | None = None
        self._writer: SessionWriter | None = None
        self._writer_closed = False
        self._snapshot: _DeviceSnapshot | None = None
        self._device_restored = False
        self._session_root: Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._prepared: PreparedPhotoCapture | None = None
        self._last_indices: tuple[int, int] | None = None
        self._sequence_valid = False
        self._pipeline_started = False
        self._closed = False
        self._received = 0
        self._timestamp_regressions = 0
        self._previous_color_timestamp: int | None = None
        self._first_capture_started: float | None = None
        self._last_capture_completed: float | None = None
        self._stop_reason: str | None = None
        self._capture_errors: list[dict[str, str]] = []
        self._close_errors: list[str] = []
        self._preparation_failed = False
        self._exposure_raw = self.settings.exposure_us // COLOR_EXPOSURE_UNIT_US

    @property
    def prepared(self) -> bool:
        return self._prepared is not None and not self._closed

    @property
    def session_root(self) -> Path | None:
        return self._session_root

    def _load_sdk(self) -> Any:
        if self._sdk is not None:
            return self._sdk
        try:
            import pyorbbecsdk as sdk
        except ImportError as exc:
            raise PhotoCaptureError(
                "pyorbbecsdk2 is not installed. Run: "
                "python -m pip install pyorbbecsdk2"
            ) from exc
        self._sdk = sdk
        return sdk

    def _property(self, name: str) -> Any:
        sdk = self._load_sdk()
        try:
            return getattr(sdk.OBPropertyID, name)
        except AttributeError as exc:
            raise PhotoCaptureError(
                f"The camera SDK does not expose required property {name}"
            ) from exc

    def _require_write_support(self, prop: Any, label: str) -> None:
        sdk = self._load_sdk()
        assert self._device is not None
        try:
            permission = sdk.OBPermissionType.PERMISSION_WRITE
            supported = bool(self._device.is_property_supported(prop, permission))
        except Exception as exc:
            raise PhotoCaptureError(
                f"Could not verify write support for {label}"
            ) from exc
        if not supported:
            raise PhotoCaptureError(f"The camera cannot write required {label}")

    def _read_bool(self, prop: Any, label: str) -> bool:
        assert self._device is not None
        try:
            return bool(self._device.get_bool_property(prop))
        except Exception as exc:
            raise PhotoCaptureError(f"Could not read {label}") from exc

    def _read_int(self, prop: Any, label: str) -> int:
        assert self._device is not None
        try:
            return int(self._device.get_int_property(prop))
        except Exception as exc:
            raise PhotoCaptureError(f"Could not read {label}") from exc

    def _set_bool(self, prop: Any, value: bool, label: str) -> None:
        assert self._device is not None
        self._require_write_support(prop, label)
        try:
            self._device.set_bool_property(prop, bool(value))
        except Exception as exc:
            raise PhotoCaptureError(f"Could not write {label}") from exc
        applied = self._read_bool(prop, label)
        if applied is not bool(value):
            raise PhotoCaptureError(
                f"The camera read back {label}={applied}, expected {value}"
            )

    def _set_int(self, prop: Any, value: int, label: str) -> None:
        assert self._device is not None
        self._require_write_support(prop, label)
        try:
            value_range = self._device.get_int_property_range(prop)
            minimum = int(value_range.min)
            maximum = int(value_range.max)
            step = max(1, int(value_range.step))
        except Exception as exc:
            raise PhotoCaptureError(f"Could not read the {label} range") from exc
        if value < minimum or value > maximum or (value - minimum) % step:
            raise PhotoCaptureError(
                f"Requested {label}={value} is outside device range "
                f"{minimum}..{maximum} step {step}"
            )
        try:
            self._device.set_int_property(prop, int(value))
        except Exception as exc:
            raise PhotoCaptureError(f"Could not write {label}") from exc
        applied = self._read_int(prop, label)
        if applied != value:
            raise PhotoCaptureError(
                f"The camera read back {label}={applied}, expected {value}"
            )

    def _snapshot_sync(self) -> _SyncSnapshot:
        assert self._device is not None
        try:
            config = self._device.get_multi_device_sync_config()
            return _SyncSnapshot(
                mode=config.mode,
                depth_delay_us=int(config.depth_delay_us),
                color_delay_us=int(config.color_delay_us),
                trigger_to_image_delay_us=int(config.trigger_to_image_delay_us),
                trigger_out_enable=bool(config.trigger_out_enable),
                trigger_out_delay_us=int(config.trigger_out_delay_us),
                frames_per_trigger=int(config.frames_per_trigger),
            )
        except Exception as exc:
            raise PhotoCaptureError(
                "Could not snapshot the camera synchronization configuration"
            ) from exc

    def _snapshot_device(self) -> _DeviceSnapshot:
        auto_capture = self._property("OB_DEVICE_AUTO_CAPTURE_ENABLE_BOOL")
        output_gate = self._property("OB_PROP_SYNC_SIGNAL_TRIGGER_OUT_BOOL")
        auto_exposure = self._property("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL")
        exposure = self._property("OB_PROP_COLOR_EXPOSURE_INT")
        for prop, label in (
            (auto_capture, "timed auto capture"),
            (output_gate, "SBU Trigger Out gate"),
            (auto_exposure, "color auto exposure"),
            (exposure, "color exposure"),
        ):
            self._require_write_support(prop, label)
        return _DeviceSnapshot(
            sync=self._snapshot_sync(),
            auto_capture=self._read_bool(auto_capture, "timed auto capture"),
            output_gate=self._read_bool(output_gate, "SBU Trigger Out gate"),
            color_auto_exposure=self._read_bool(
                auto_exposure, "color auto exposure"
            ),
            color_exposure_raw=self._read_int(exposure, "color exposure"),
        )

    def _set_output_gate(self, value: bool) -> None:
        self._set_bool(
            self._property("OB_PROP_SYNC_SIGNAL_TRIGGER_OUT_BOOL"),
            value,
            "SBU Trigger Out gate",
        )

    def _disable_timed_auto_capture(self) -> None:
        self._set_bool(
            self._property("OB_DEVICE_AUTO_CAPTURE_ENABLE_BOOL"),
            False,
            "timed auto capture",
        )

    def _apply_manual_exposure(self) -> None:
        self._set_bool(
            self._property("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL"),
            False,
            "color auto exposure",
        )
        self._set_int(
            self._property("OB_PROP_COLOR_EXPOSURE_INT"),
            self._exposure_raw,
            "color exposure",
        )

    def _configure_software_trigger(self) -> dict[str, Any]:
        sdk = self._load_sdk()
        assert self._device is not None
        try:
            mode = sdk.OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
            config = self._device.get_multi_device_sync_config()
            config.mode = mode
            config.depth_delay_us = 0
            config.color_delay_us = 0
            config.trigger_to_image_delay_us = 0
            config.trigger_out_enable = True
            config.trigger_out_delay_us = self.settings.trigger_out_delay_us
            config.frames_per_trigger = 1
            self._device.set_multi_device_sync_config(config)
        except Exception as exc:
            raise PhotoCaptureError(
                "Could not apply SOFTWARE_TRIGGERING synchronization"
            ) from exc
        return self._verify_sync_config()

    def _verify_sync_config(self) -> dict[str, Any]:
        sdk = self._load_sdk()
        assert self._device is not None
        try:
            applied = self._device.get_multi_device_sync_config()
        except Exception as exc:
            raise PhotoCaptureError(
                "Could not read back the synchronization configuration"
            ) from exc
        expected_mode = sdk.OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
        required = {
            "mode": (applied.mode, expected_mode),
            "depth_delay_us": (int(applied.depth_delay_us), 0),
            "color_delay_us": (int(applied.color_delay_us), 0),
            "trigger_to_image_delay_us": (
                int(applied.trigger_to_image_delay_us),
                0,
            ),
            "trigger_out_enable": (bool(applied.trigger_out_enable), True),
            "trigger_out_delay_us": (
                int(applied.trigger_out_delay_us),
                self.settings.trigger_out_delay_us,
            ),
            "frames_per_trigger": (int(applied.frames_per_trigger), 1),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in required.items()
            if actual != expected
        ]
        if mismatches:
            raise PhotoCaptureError(
                "Unsafe software-trigger readback: " + ", ".join(mismatches)
            )
        return {
            "mode": _enum_name(applied.mode),
            "trigger_out_enable": True,
            "frames_per_trigger": 1,
            "depth_delay_us": 0,
            "color_delay_us": 0,
            "trigger_to_image_delay_us": 0,
            "trigger_out_delay_us": self.settings.trigger_out_delay_us,
        }

    def _verify_runtime(self, *, expected_gate: bool) -> None:
        self._verify_sync_config()
        if self._read_bool(
            self._property("OB_DEVICE_AUTO_CAPTURE_ENABLE_BOOL"),
            "timed auto capture",
        ):
            raise PhotoCaptureError("Timed auto capture became enabled")
        if self._read_bool(
            self._property("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL"),
            "color auto exposure",
        ):
            raise PhotoCaptureError("Color auto exposure became enabled")
        exposure = self._read_int(
            self._property("OB_PROP_COLOR_EXPOSURE_INT"), "color exposure"
        )
        if exposure != self._exposure_raw:
            raise PhotoCaptureError(
                f"Color exposure changed to {exposure}, expected {self._exposure_raw}"
            )
        gate = self._read_bool(
            self._property("OB_PROP_SYNC_SIGNAL_TRIGGER_OUT_BOOL"),
            "SBU Trigger Out gate",
        )
        if gate is not expected_gate:
            raise PhotoCaptureError(
                f"SBU Trigger Out gate={gate}, expected {expected_gate}"
            )

    def _restore_device(self) -> list[str]:
        """Best-effort restore that never reopens SBU after an unsafe step."""

        if self._device is None or self._snapshot is None:
            return []
        snapshot = self._snapshot
        errors: list[str] = []
        sync_restored = False
        try:
            config = self._device.get_multi_device_sync_config()
            for name in (
                "mode",
                "depth_delay_us",
                "color_delay_us",
                "trigger_to_image_delay_us",
                "trigger_out_enable",
                "trigger_out_delay_us",
                "frames_per_trigger",
            ):
                setattr(config, name, getattr(snapshot.sync, name))
            self._device.set_multi_device_sync_config(config)
            if self._snapshot_sync() != snapshot.sync:
                raise PhotoCaptureError(
                    "The camera did not restore its synchronization configuration"
                )
            sync_restored = True
        except Exception as exc:
            errors.append(f"sync configuration: {exc}")

        auto_capture = self._property("OB_DEVICE_AUTO_CAPTURE_ENABLE_BOOL")
        output_gate = self._property("OB_PROP_SYNC_SIGNAL_TRIGGER_OUT_BOOL")
        auto_exposure = self._property("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL")
        exposure = self._property("OB_PROP_COLOR_EXPOSURE_INT")

        for label, action in (
            (
                "disable color auto exposure for restore",
                lambda: self._set_bool(
                    auto_exposure, False, "color auto exposure"
                ),
            ),
            (
                "restore color exposure",
                lambda: self._set_int(
                    exposure,
                    snapshot.color_exposure_raw,
                    "color exposure",
                ),
            ),
            (
                "restore color auto exposure",
                lambda: self._set_bool(
                    auto_exposure,
                    snapshot.color_auto_exposure,
                    "color auto exposure",
                ),
            ),
        ):
            try:
                action()
            except Exception as exc:
                errors.append(f"{label}: {exc}")

        # Timed capture and the physical output gate are restored last and only
        # when every earlier step was verified. Otherwise force both safe-off.
        if sync_restored and not errors:
            try:
                self._set_bool(
                    auto_capture,
                    snapshot.auto_capture,
                    "timed auto capture",
                )
            except Exception as exc:
                errors.append(f"restore timed auto capture: {exc}")
            if not errors:
                try:
                    self._set_bool(
                        output_gate,
                        snapshot.output_gate,
                        "SBU Trigger Out gate",
                    )
                except Exception as exc:
                    errors.append(f"restore SBU Trigger Out gate: {exc}")

        if errors:
            for label, action in (
                (
                    "force timed auto capture off",
                    lambda: self._set_bool(
                        auto_capture, False, "timed auto capture"
                    ),
                ),
                (
                    "force SBU Trigger Out gate off",
                    lambda: self._set_bool(
                        output_gate, False, "SBU Trigger Out gate"
                    ),
                ),
            ):
                try:
                    action()
                except Exception as exc:
                    errors.append(f"{label}: {exc}")
        return errors

    def _select_profiles(self) -> tuple[Any, Any, list[dict[str, Any]], list[dict[str, Any]]]:
        sdk = self._load_sdk()
        assert self._pipeline is not None
        color_list = self._pipeline.get_stream_profile_list(
            sdk.OBSensorType.COLOR_SENSOR
        )
        depth_list = self._pipeline.get_stream_profile_list(
            sdk.OBSensorType.DEPTH_SENSOR
        )
        color, depth = select_fastest_rgbd_profiles(
            color_list,
            depth_list,
            width=self.settings.width,
            height=self.settings.height,
            color_formats=self.settings.color_formats,
            sdk=sdk,
            trigger_out_delay_us=self.settings.trigger_out_delay_us,
        )
        fps = int(color.get_fps())
        maximum_delay = max(0, round(1_000_000 / fps) - TRIGGER_DELAY_GUARD_US)
        if self.settings.trigger_out_delay_us > maximum_delay:
            raise PhotoCaptureError(
                f"trigger_out_delay_us must be <= {maximum_delay} at {fps} FPS"
            )
        return (
            color,
            depth,
            _available_video_profiles(color_list),
            _available_video_profiles(depth_list),
        )

    @staticmethod
    def _frame_indices(frames: Any) -> tuple[int, int]:
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if color is None or depth is None:
            raise PhotoCaptureError("A complete RGB-D frame set is required")
        try:
            color_index = int(color.get_index())
            depth_index = int(depth.get_index())
        except Exception as exc:
            raise PhotoCaptureError(
                "Frame indices are required to reject stale triggered frames"
            ) from exc
        if color_index < 0 or depth_index < 0:
            raise PhotoCaptureError("Frame indices must be non-negative")
        return color_index, depth_index

    def _prime_pipeline(self) -> tuple[tuple[int, int], float]:
        assert self._device is not None and self._pipeline is not None
        assert self._align_filter is not None
        last_error: Exception | None = None
        for _attempt in range(self.settings.prime_attempts):
            self._verify_runtime(expected_gate=False)
            trigger_started = self._clock()
            try:
                # These bounded startup triggers are never formal captures:
                # the physical SBU gate is verified OFF before every call.
                self._device.trigger_capture()
                wait_started = self._clock()
                frames = self._pipeline.wait_for_frames(
                    self.settings.prime_timeout_ms
                )
                waited = max(0.0, self._clock() - wait_started)
                if frames is None:
                    remaining = self.settings.prime_timeout_ms / 1_000 - waited
                    if remaining > 0.0:
                        self._sleep(remaining)
                    continue
                indices = self._frame_indices(frames)
                aligned = self._align_filter.process(frames)
                if aligned is None:
                    raise PhotoCaptureError(
                        "RGB-D software-trigger priming alignment failed"
                    )
                self._frame_indices(aligned)
                return indices, trigger_started
            except Exception as exc:
                last_error = exc
        detail = f": {last_error}" if last_error is not None else ""
        raise PhotoCaptureError(
            "RGB-D software-trigger priming failed after bounded gate-off "
            f"attempts; no formal trigger was issued{detail}"
        )

    def _wait_for_priming_quiescence(
        self,
        baseline: tuple[int, int],
        *,
        last_trigger_started: float,
    ) -> tuple[int, int]:
        """Hold SBU closed through the full late-response safety window."""

        assert self._pipeline is not None
        deadline = (
            last_trigger_started + self.settings.capture_timeout_ms / 1_000
        )
        late_frames = 0
        maximum_late_frames = self.settings.prime_attempts + STALE_DRAIN_LIMIT
        while True:
            self._verify_runtime(expected_gate=False)
            remaining = deadline - self._clock()
            if remaining <= 0.0:
                break
            # Wait through the whole remaining safety window. A late frame
            # returns early and is drained below; an empty result proves the
            # complete configured response horizon in one SDK call.
            timeout_ms = max(1, math.ceil(remaining * 1_000))
            try:
                wait_started = self._clock()
                frames = self._pipeline.wait_for_frames(timeout_ms)
            except Exception as exc:
                raise PhotoCaptureError(
                    "Could not verify that priming became quiescent while the "
                    "physical output gate was off"
                ) from exc
            waited = max(0.0, self._clock() - wait_started)
            if frames is None:
                unspent_wait = timeout_ms / 1_000 - waited
                if unspent_wait > 0.0:
                    self._sleep(unspent_wait)
                continue
            late_frames += 1
            if late_frames > maximum_late_frames:
                raise PhotoCaptureError(
                    "Priming produced more late RGB-D frames than the bounded "
                    "trigger count permits"
                )
            indices = self._frame_indices(frames)
            baseline = (
                max(baseline[0], indices[0]),
                max(baseline[1], indices[1]),
            )

        guard_us = self.settings.trigger_out_delay_us + TRIGGER_DELAY_GUARD_US
        if guard_us:
            self._sleep(guard_us / 1_000_000)
        self._verify_runtime(expected_gate=False)
        try:
            overflow = self._pipeline.wait_for_frames(1)
        except Exception as exc:
            raise PhotoCaptureError(
                "Final priming queue verification failed while the output gate "
                "was off"
            ) from exc
        if overflow is not None:
            raise PhotoCaptureError(
                "A priming response arrived after the complete late-response "
                "window; the physical output gate will remain off"
            )
        return baseline

    def _new_session_root(self) -> Path:
        parent = self.settings.output_root.expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        stem = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")[:-3]
        for suffix in range(1_000):
            candidate = parent / (stem if suffix == 0 else f"{stem}_{suffix:03d}")
            try:
                candidate.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return candidate
        raise PhotoCaptureError("Could not allocate a unique photo session directory")

    def prepare(self) -> PreparedPhotoCapture:
        with self._operation_lock:
            if self._closed:
                raise PhotoCaptureError("This photo controller is closed")
            if self._prepared is not None:
                raise PhotoCaptureError("Photo mode is already prepared")
            if self._device is not None or self._pipeline is not None:
                raise PhotoCaptureError(
                    "A previous photo cleanup is incomplete; close the controller "
                    "before preparing again"
                )
            sdk = self._load_sdk()
            try:
                self._context = sdk.Context()
                devices = self._context.query_devices()
                if devices.get_count() == 0:
                    raise PhotoCaptureError("No Orbbec camera found")
                self._device = devices.get_device_by_index(0)
                self._snapshot = self._snapshot_device()
                self._device_restored = False

                # Some Gemini 305 firmware re-enables the physical SBU gate
                # when a new multi-device sync config is committed. Close it
                # before the transition and again immediately afterwards. No
                # software trigger is issued in between, and timed automatic
                # capture is already disabled.
                self._set_output_gate(False)
                self._disable_timed_auto_capture()
                sync_config = self._configure_software_trigger()
                self._set_output_gate(False)
                self._apply_manual_exposure()

                self._pipeline = sdk.Pipeline(self._device)
                color_profile, depth_profile, colors, depths = self._select_profiles()
                stream_config = sdk.Config()
                stream_config.enable_stream(color_profile)
                stream_config.enable_stream(depth_profile)
                stream_config.set_frame_aggregate_output_mode(
                    sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
                )
                try:
                    self._pipeline.enable_frame_sync()
                except Exception as exc:
                    raise PhotoCaptureError(
                        "Could not enable synchronized RGB-D frames"
                    ) from exc
                self._align_filter = sdk.AlignFilter(
                    align_to_stream=sdk.OBStreamType.COLOR_STREAM
                )
                self._pipeline.start(stream_config)
                self._pipeline_started = True
                # Some firmware reapplies exposure defaults while streams start.
                self._apply_manual_exposure()
                self._verify_runtime(expected_gate=False)
                if self.settings.gate_settle_ms:
                    self._sleep(self.settings.gate_settle_ms / 1_000)
                primed_indices, last_prime_trigger = self._prime_pipeline()
                self._last_indices = self._wait_for_priming_quiescence(
                    primed_indices,
                    last_trigger_started=last_prime_trigger,
                )

                calibration = _calibration_to_dict(self._pipeline.get_camera_param())
                color_intrinsic = calibration.get("color_intrinsic", {})
                if (
                    int(color_intrinsic.get("width", 0)) != self.settings.width
                    or int(color_intrinsic.get("height", 0)) != self.settings.height
                ):
                    raise PhotoCaptureError(
                        "Camera calibration dimensions do not match the selected color profile"
                    )

                self._session_root = self._new_session_root()
                self._manifest = {
                    "schema": "panorama-demo-session/v1",
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "clean_shutdown": False,
                    "capture_mode": "software_triggered_rgbd_photo_sequence",
                    "diagnostic_only": False,
                    # An open writer/session is never a formal panorama input.
                    # Successful close is the only transition to True.
                    "formal_stitch_allowed": False,
                    "session_state": "open",
                    "no_video_preview": True,
                    "one_formal_trigger_per_capture": True,
                    "sequence_rate_policy": "fastest_unthrottled",
                    "capture_options": {
                        "width": self.settings.width,
                        "height": self.settings.height,
                        "fps": int(color_profile.get_fps()),
                        "color_formats": list(self.settings.color_formats),
                        "align": "software",
                        "frame_sync": True,
                        "color_auto_exposure": False,
                        "color_exposure_us": self.settings.exposure_us,
                        "save_raw_depth": False,
                    },
                    "device": _device_info(self._device),
                    "profiles": {
                        "color": _profile_dict(color_profile),
                        "depth": _profile_dict(depth_profile),
                    },
                    "available_profiles": {"color": colors, "depth": depths},
                    "software_trigger": sync_config,
                    "calibration": calibration,
                }
                try:
                    self._manifest["sdk_version"] = sdk.get_version()
                except Exception:
                    self._manifest["sdk_version"] = None
                try:
                    self._manifest["python_wrapper_version"] = (
                        importlib_metadata.version("pyorbbecsdk2")
                    )
                except importlib_metadata.PackageNotFoundError:
                    self._manifest["python_wrapper_version"] = None
                (self._session_root / "calibration.json").write_text(
                    json.dumps(calibration, indent=2), encoding="utf-8"
                )
                _write_manifest(self._session_root, self._manifest)
                self._writer = SessionWriter(
                    self._session_root,
                    queue_size=1,
                    jpeg_quality=self.settings.jpeg_quality,
                    depth_png_compression=self.settings.depth_png_compression,
                    save_raw_depth=False,
                )
                self._writer_closed = False

                self._set_output_gate(True)
                if self.settings.gate_settle_ms:
                    self._sleep(self.settings.gate_settle_ms / 1_000)
                self._verify_runtime(expected_gate=True)
                self._sequence_valid = True
                self._prepared = PreparedPhotoCapture(
                    session_root=self._session_root,
                    width=self.settings.width,
                    height=self.settings.height,
                    fps=int(color_profile.get_fps()),
                    color_format=_enum_name(color_profile.get_format()),
                    depth_format=_enum_name(depth_profile.get_format()),
                    exposure_us=self.settings.exposure_us,
                    trigger_out_delay_us=self.settings.trigger_out_delay_us,
                )
                return self._prepared
            except BaseException as exc:
                cleanup_errors = self._cleanup_after_prepare_failure(exc)
                message = str(exc)
                if cleanup_errors:
                    message += "; cleanup failed: " + "; ".join(cleanup_errors)
                if isinstance(exc, PhotoCaptureError) and not cleanup_errors:
                    raise
                raise PhotoCaptureError(message) from exc

    def _cleanup_after_prepare_failure(self, exc: BaseException) -> list[str]:
        self._preparation_failed = True
        errors: list[str] = []
        if self._writer is not None and not self._writer_closed:
            try:
                self._writer.close()
                self._writer_closed = True
            except Exception as close_exc:
                errors.append(f"writer: {close_exc}")
        if self._pipeline_started and self._pipeline is not None:
            try:
                self._pipeline.stop()
                self._pipeline_started = False
            except Exception as stop_exc:
                errors.append(f"pipeline: {stop_exc}")
        if not self._pipeline_started:
            try:
                restore_errors = self._restore_device()
            except Exception as restore_exc:
                errors.append(f"device restore: {restore_exc}")
            else:
                if restore_errors:
                    errors.extend(
                        f"device restore: {item}" for item in restore_errors
                    )
                else:
                    self._device_restored = True
        if self._session_root is not None and self._manifest is not None:
            self._manifest.update(
                {
                    "ended_utc": datetime.now(timezone.utc).isoformat(),
                    "clean_shutdown": False,
                    "formal_stitch_allowed": False,
                    "session_state": "failed",
                    "capture_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )
            try:
                _write_manifest(self._session_root, self._manifest)
            except Exception as manifest_exc:
                errors.append(f"manifest: {manifest_exc}")
        self._sequence_valid = False
        self._prepared = None
        if not self._pipeline_started and self._device_restored:
            self._pipeline = None
            self._align_filter = None
            self._device = None
            self._context = None
            self._snapshot = None
        return errors

    def _drain_stale_frames(self) -> tuple[int, int]:
        assert self._pipeline is not None
        baseline = self._last_indices
        if baseline is None:
            raise PhotoCaptureError("Photo priming did not establish frame indices")
        for _ in range(STALE_DRAIN_LIMIT):
            try:
                frames = self._pipeline.wait_for_frames(1)
            except Exception as exc:
                raise PhotoCaptureError(
                    "Could not drain stale RGB-D frames; no trigger was sent"
                ) from exc
            if frames is None:
                break
            indices = self._frame_indices(frames)
            baseline = (
                max(baseline[0], indices[0]),
                max(baseline[1], indices[1]),
            )
        try:
            overflow = self._pipeline.wait_for_frames(1)
        except Exception as exc:
            raise PhotoCaptureError(
                "Could not verify an empty stale RGB-D queue; no trigger was sent"
            ) from exc
        if overflow is not None:
            indices = self._frame_indices(overflow)
            self._last_indices = (
                max(baseline[0], indices[0]),
                max(baseline[1], indices[1]),
            )
            raise PhotoCaptureError(
                "The stale RGB-D queue exceeded its safe drain limit; no trigger "
                "was sent"
            )
        self._last_indices = baseline
        return baseline

    def _wait_for_fresh_frames(self, baseline: tuple[int, int]) -> Any:
        assert self._pipeline is not None
        deadline = self._clock() + self.settings.capture_timeout_ms / 1_000
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise PhotoCaptureError(
                    "Triggered RGB-D frame timed out; the controller will not retrigger"
                )
            timeout_ms = max(1, min(self.settings.capture_timeout_ms, math.ceil(remaining * 1_000)))
            try:
                wait_started = self._clock()
                frames = self._pipeline.wait_for_frames(timeout_ms)
            except Exception as exc:
                raise PhotoCaptureError(
                    "Waiting for the triggered RGB-D frame failed; the controller "
                    "will not retrigger"
                ) from exc
            if frames is None:
                waited = max(0.0, self._clock() - wait_started)
                unspent_wait = timeout_ms / 1_000 - waited
                if unspent_wait > 0.0:
                    self._sleep(unspent_wait)
                continue
            indices = self._frame_indices(frames)
            if indices[0] > baseline[0] and indices[1] > baseline[1]:
                self._last_indices = indices
                return frames

    @staticmethod
    def _required_timestamp(frame: Any, method: str, label: str) -> int:
        try:
            value = int(getattr(frame, method)())
        except Exception as exc:
            raise PhotoCaptureError(f"Missing {label}") from exc
        if value < 0:
            raise PhotoCaptureError(f"{label} must be non-negative")
        return value

    def _packet_from_frames(self, frames: Any, frame_id: int) -> FramePacket:
        sdk = self._load_sdk()
        assert self._align_filter is not None
        raw_color = frames.get_color_frame()
        raw_depth = frames.get_depth_frame()
        if raw_color is None or raw_depth is None:
            raise PhotoCaptureError("Triggered frame set is missing color or depth")
        aligned = self._align_filter.process(frames)
        if aligned is None:
            raise PhotoCaptureError("Software depth alignment returned no frame set")
        aligned_color = aligned.get_color_frame()
        aligned_depth_frame = aligned.get_depth_frame()
        if aligned_color is None or aligned_depth_frame is None:
            raise PhotoCaptureError("Aligned RGB-D frame set is incomplete")
        color_image = _frame_to_bgr(aligned_color, sdk).copy()
        aligned_depth = _depth_array(aligned_depth_frame)
        expected_shape = (self.settings.height, self.settings.width)
        if color_image.shape[:2] != expected_shape or aligned_depth.shape != expected_shape:
            raise PhotoCaptureError(
                "Aligned RGB-D dimensions do not match the selected profile"
            )
        depth_scale = float(aligned_depth_frame.get_depth_scale())
        if not math.isfinite(depth_scale) or depth_scale <= 0:
            raise PhotoCaptureError("Aligned depth scale must be finite and positive")

        color_timestamp = self._required_timestamp(
            raw_color, "get_timestamp_us", "color device timestamp"
        )
        depth_timestamp = self._required_timestamp(
            raw_depth, "get_timestamp_us", "depth device timestamp"
        )
        if (
            self._previous_color_timestamp is not None
            and color_timestamp <= self._previous_color_timestamp
        ):
            self._timestamp_regressions += 1
        self._previous_color_timestamp = color_timestamp
        metadata_types = sdk.OBFrameMetadataType
        color_exposure = _metadata(raw_color, metadata_types.EXPOSURE)
        if color_exposure is None:
            raise PhotoCaptureError(
                "Triggered color exposure metadata is required; the requested "
                "property value cannot be substituted for device frame metadata"
            )
        if color_exposure <= 0 or color_exposure > self._exposure_raw:
            raise PhotoCaptureError(
                "Triggered color exposure metadata exceeds the formal photo limit"
            )
        metadata = {
            "color_index": int(raw_color.get_index()),
            "depth_index": int(raw_depth.get_index()),
            "color_device_timestamp_us": color_timestamp,
            "depth_device_timestamp_us": depth_timestamp,
            "sync_delta_us": color_timestamp - depth_timestamp,
            "color_system_timestamp_us": self._required_timestamp(
                raw_color, "get_system_timestamp_us", "color system timestamp"
            ),
            "depth_system_timestamp_us": self._required_timestamp(
                raw_depth, "get_system_timestamp_us", "depth system timestamp"
            ),
            "host_timestamp_ns": time.time_ns(),
            "color_exposure": color_exposure,
            "depth_exposure": _metadata(raw_depth, metadata_types.EXPOSURE),
            "color_gain": _metadata(raw_color, metadata_types.GAIN),
            "depth_gain": _metadata(raw_depth, metadata_types.GAIN),
            "color_frame_number": _metadata(
                raw_color, metadata_types.FRAME_NUMBER
            ),
            "depth_frame_number": _metadata(
                raw_depth, metadata_types.FRAME_NUMBER
            ),
            "color_sensor_timestamp_raw": _metadata(
                raw_color, metadata_types.SENSOR_TIMESTAMP
            ),
            "depth_sensor_timestamp_raw": _metadata(
                raw_depth, metadata_types.SENSOR_TIMESTAMP
            ),
            "depth_scale_mm_per_unit": depth_scale,
        }
        return FramePacket(frame_id, color_image, aligned_depth, None, metadata)

    def _update_live_manifest(self) -> None:
        if self._session_root is None or self._manifest is None or self._writer is None:
            return
        self._manifest.update(
            {
                "received_frames": self._received,
                "written_frames": self._writer.stats.written,
                "queue_drops": self._writer.stats.queue_drops,
                "write_errors": self._writer.stats.write_errors,
                "writer_errors": list(self._writer.stats.errors),
                "timestamp_regressions": self._timestamp_regressions,
            }
        )
        _write_manifest(self._session_root, self._manifest)

    def _record_capture_error(self, exc: BaseException) -> None:
        if self._manifest is None or self._session_root is None:
            return
        error = {"type": type(exc).__name__, "message": str(exc)}
        self._sequence_valid = False
        self._capture_errors.append(error)
        self._manifest.update(
            {
                "formal_stitch_allowed": False,
                "capture_error": error,
                "capture_errors": list(self._capture_errors),
            }
        )
        try:
            self._update_live_manifest()
        except Exception:
            # Preserve the original capture failure. Final close will attempt to
            # write the complete manifest again and report its own error.
            pass

    def capture_once(self) -> PhotoCaptureResult:
        if not self._capture_lock.acquire(blocking=False):
            raise PhotoCaptureBusyError("Another photo capture is still in progress")
        try:
            with self._operation_lock:
                if not self.prepared or self._pipeline is None or self._device is None:
                    raise PhotoCaptureError("Photo mode is not prepared")
                if not self._sequence_valid:
                    raise PhotoCaptureError(
                        "The previous trigger sequence is uncertain; end the session "
                        "and prepare photo mode again"
                    )
                assert self._writer is not None and self._session_root is not None
                self._verify_runtime(expected_gate=True)
                baseline = self._drain_stale_frames()
                self._verify_runtime(expected_gate=True)
                self._sequence_valid = False

                # Safety invariant: this is the only formal trigger in this
                # accepted call. No exception path below retries it.
                if self._first_capture_started is None:
                    self._first_capture_started = self._clock()
                try:
                    self._device.trigger_capture()
                except Exception as exc:
                    raise PhotoCaptureError(
                        "The formal software trigger call failed; the device may "
                        "still have received it, so the controller will not retry"
                    ) from exc

                frames = self._wait_for_fresh_frames(baseline)
                frame_id = self._received
                packet = self._packet_from_frames(frames, frame_id)
                write_errors_before = self._writer.stats.write_errors
                written_before = self._writer.stats.written
                if not self._writer.submit(packet):
                    raise PhotoCaptureError(
                        "The photo writer queue rejected the triggered RGB-D frame"
                    )
                self._writer.queue.join()
                if (
                    self._writer.stats.write_errors != write_errors_before
                    or self._writer.stats.written != written_before + 1
                ):
                    detail = (
                        self._writer.stats.errors[-1]
                        if self._writer.stats.errors
                        else "unknown write failure"
                    )
                    raise PhotoCaptureError(
                        "The triggered RGB-D photo could not be saved; the controller "
                        f"will not retrigger: {detail}"
                    )
                self._received += 1
                self._last_capture_completed = self._clock()
                self._sequence_valid = True
                self._update_live_manifest()
                stem = f"{frame_id:08d}"
                return PhotoCaptureResult(
                    session_root=self._session_root,
                    frame_id=frame_id,
                    color_path=self._session_root / "color" / f"{stem}.jpg",
                    aligned_depth_path=(
                        self._session_root / "depth_aligned" / f"{stem}.png"
                    ),
                    frames_csv_path=self._session_root / "frames.csv",
                    color_timestamp_us=int(
                        packet.metadata["color_device_timestamp_us"]
                    ),
                    depth_timestamp_us=int(
                        packet.metadata["depth_device_timestamp_us"]
                    ),
                    depth_scale_mm_per_unit=float(
                        packet.metadata["depth_scale_mm_per_unit"]
                    ),
                )
        except BaseException as exc:
            self._record_capture_error(exc)
            raise
        finally:
            self._capture_lock.release()

    def set_stop_reason(self, reason: str) -> None:
        with self._operation_lock:
            value = str(reason).strip()
            if not value:
                raise ValueError("photo sequence stop reason cannot be empty")
            self._stop_reason = value

    def close(self) -> None:
        with self._operation_lock:
            if self._closed:
                return
            self._sequence_valid = False
            self._prepared = None
            errors: list[str] = []
            if self._device is not None and not self._device_restored:
                try:
                    self._set_output_gate(False)
                except Exception as exc:
                    errors.append(f"disable output gate: {exc}")
            if self._pipeline_started and self._pipeline is not None:
                try:
                    self._pipeline.stop()
                    self._pipeline_started = False
                except Exception as exc:
                    errors.append(f"stop pipeline: {exc}")
            if self._writer is not None and not self._writer_closed:
                try:
                    self._writer.close()
                    self._writer_closed = True
                except Exception as exc:
                    errors.append(f"close writer: {exc}")
            if not self._pipeline_started and not self._device_restored:
                try:
                    restore_errors = self._restore_device()
                except Exception as exc:
                    errors.append(f"restore device: {exc}")
                else:
                    if restore_errors:
                        errors.extend(
                            f"restore device: {item}" for item in restore_errors
                        )
                    else:
                        self._device_restored = True

            writer_errors = (
                self._writer.stats.write_errors if self._writer is not None else 0
            )
            hardware_safe = not self._pipeline_started and (
                self._device is None
                or self._snapshot is None
                or self._device_restored
            )
            writer_safe = self._writer is None or self._writer_closed
            resources_safe = hardware_safe and writer_safe
            clean_shutdown = (
                resources_safe
                and not errors
                and writer_errors == 0
                and not self._capture_errors
                and not self._close_errors
                and not self._preparation_failed
            )

            if errors:
                self._close_errors.extend(errors)

            if self._session_root is not None and self._manifest is not None:
                elapsed_seconds = None
                achieved_fps = None
                if (
                    self._first_capture_started is not None
                    and self._last_capture_completed is not None
                ):
                    elapsed_seconds = max(
                        0.0,
                        self._last_capture_completed - self._first_capture_started,
                    )
                    if elapsed_seconds > 0.0 and self._received > 0:
                        achieved_fps = self._received / elapsed_seconds
                self._manifest.update(
                    {
                        "ended_utc": datetime.now(timezone.utc).isoformat(),
                        "clean_shutdown": clean_shutdown,
                        "formal_stitch_allowed": clean_shutdown,
                        "session_state": "closed" if clean_shutdown else "failed",
                        "received_frames": self._received,
                        "written_frames": (
                            self._writer.stats.written if self._writer is not None else 0
                        ),
                        "queue_drops": (
                            self._writer.stats.queue_drops
                            if self._writer is not None
                            else 0
                        ),
                        "write_errors": writer_errors,
                        "writer_errors": (
                            list(self._writer.stats.errors)
                            if self._writer is not None
                            else []
                        ),
                        "timestamp_regressions": self._timestamp_regressions,
                        "capture_errors": list(self._capture_errors),
                        "close_errors": list(self._close_errors),
                        "photo_sequence": {
                            "frames": self._received,
                            "elapsed_seconds": elapsed_seconds,
                            "achieved_fps": achieved_fps,
                            "rate_policy": "fastest_unthrottled",
                            "stop_reason": self._stop_reason,
                        },
                    }
                )
                if self._close_errors and not self._capture_errors:
                    self._manifest["capture_error"] = {
                        "type": "PhotoCaptureCloseError",
                        "message": "; ".join(self._close_errors),
                    }
                try:
                    _write_manifest(self._session_root, self._manifest)
                except Exception as exc:
                    errors.append(f"write final manifest: {exc}")

            if resources_safe:
                self._pipeline = None
                self._align_filter = None
                self._writer = None
                self._device = None
                self._context = None
                self._snapshot = None
                self._closed = True
            if errors or not resources_safe:
                if not errors:
                    errors.append("camera or writer resources remain active")
                raise PhotoCaptureError(
                    "Could not close photo session: " + "; ".join(errors)
                )


def photo_settings_from_config(
    config_path: str | Path | None,
    output_root: Path,
    *,
    width: int | None = None,
    height: int | None = None,
) -> PhotoCaptureSettings:
    config = load_config(config_path)
    capture = config.get("capture")
    if not isinstance(capture, dict):
        raise ValueError("Configuration is missing capture settings")
    photo = capture.get("photo_mode")
    if not isinstance(photo, dict) or not bool(photo.get("enabled", False)):
        raise ValueError("capture.photo_mode must be enabled")
    if not bool(photo.get("fastest_common_fps", False)):
        raise ValueError(
            "capture.photo_mode must select the fastest common RGB-D FPS"
        )
    if bool(capture.get("diagnostic_unrestricted_auto_exposure", False)):
        raise ValueError(
            "Unrestricted diagnostic auto exposure cannot be used by photo mode"
        )
    try:
        settings = PhotoCaptureSettings(
            output_root=Path(output_root),
            width=int(width if width is not None else capture["width"]),
            height=int(height if height is not None else capture["height"]),
            color_formats=tuple(str(item) for item in capture["color_formats"]),
            exposure_us=int(photo["exposure_us"]),
            trigger_out_delay_us=int(photo["trigger_out_delay_us"]),
            capture_timeout_ms=int(photo["capture_timeout_ms"]),
            prime_attempts=int(photo["prime_attempts"]),
            prime_timeout_ms=int(photo["prime_timeout_ms"]),
            gate_settle_ms=int(photo["gate_settle_ms"]),
            jpeg_quality=int(capture["jpeg_quality"]),
            depth_png_compression=int(capture["depth_png_compression"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid capture.photo_mode configuration: {exc}") from exc
    return settings.validated()


def _validate_photo_sequence_args(args: Any) -> None:
    incompatible: list[str] = []
    for name in (
        "fps",
        "warmup_frames",
        "queue_size",
        "exposure_us",
        "gain",
        "white_balance",
    ):
        if getattr(args, name, None) is not None:
            incompatible.append("--" + name.replace("_", "-"))
    if bool(getattr(args, "auto_exposure", False)):
        incompatible.append("--auto-exposure")
    if bool(getattr(args, "diagnostic_unrestricted_auto_exposure", False)):
        incompatible.append("--diagnostic-unrestricted-auto-exposure")
    if getattr(args, "raw_depth", None) is True:
        incompatible.append("--raw-depth")
    if incompatible:
        raise ValueError(
            "Photo-sequence mode uses fixed safe settings and rejects: "
            + ", ".join(incompatible)
        )
    duration = getattr(args, "duration", None)
    max_frames = getattr(args, "max_frames", None)
    if duration is not None and float(duration) <= 0.0:
        raise ValueError("--duration must be positive")
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("--max-frames must be positive")


def run_photo_sequence(
    args: Any,
    *,
    controller_factory: Callable[
        [PhotoCaptureSettings], SoftwareTriggeredRGBDPhotoController
    ] = SoftwareTriggeredRGBDPhotoController,
    key_reader: Callable[[], int | None] = _console_key,
    clock: Callable[[], float] = time.monotonic,
    output: Callable[[str], None] = print,
) -> Path:
    """Capture a fastest-unthrottled, low-frame-rate RGB-D photo sequence."""

    _validate_photo_sequence_args(args)
    settings = photo_settings_from_config(
        getattr(args, "config", None),
        Path(getattr(args, "output", Path("data/captures"))),
        width=getattr(args, "width", None),
        height=getattr(args, "height", None),
    )
    controller = controller_factory(settings)
    prepared: PreparedPhotoCapture | None = None
    primary_error: BaseException | None = None
    close_error: BaseException | None = None
    stop_reason = "unknown"
    frame_count = 0
    started = clock()
    try:
        prepared = controller.prepare()
        output(
            "Photo sequence ready: "
            f"{prepared.width}x{prepared.height}@{prepared.fps} "
            f"{prepared.color_format}; no video preview"
        )
        started = clock()
        while True:
            result = controller.capture_once()
            frame_count += 1
            elapsed = max(clock() - started, 1e-9)
            # Console writes can become the limiting factor in an otherwise
            # unthrottled trigger/capture/write loop.  Keep a lightweight
            # heartbeat without adding per-frame I/O to the hot path.
            if frame_count == 1 or frame_count % 10 == 0:
                output(
                    f"frame={result.frame_id} saved={result.color_path} "
                    f"effective_fps={frame_count / elapsed:.2f}"
                )
            maximum = getattr(args, "max_frames", None)
            duration = getattr(args, "duration", None)
            if maximum is not None and frame_count >= int(maximum):
                stop_reason = "max_frames"
                break
            if duration is not None and elapsed >= float(duration):
                stop_reason = "duration"
                break
            if key_reader() in (ord("q"), ord("Q"), 27):
                stop_reason = "console_key"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    except BaseException as exc:
        primary_error = exc
        stop_reason = "capture_error" if prepared is not None else "prepare_error"
    finally:
        try:
            controller.set_stop_reason(stop_reason)
        except BaseException as exc:
            close_error = exc
        try:
            controller.close()
        except BaseException as exc:
            close_error = exc

    if primary_error is not None:
        if close_error is not None:
            raise PhotoCaptureError(
                f"Photo sequence failed ({primary_error}); cleanup also failed "
                f"({close_error})"
            ) from primary_error
        raise primary_error
    if close_error is not None:
        raise close_error
    if prepared is None:
        raise PhotoCaptureError("Photo sequence did not prepare a session")

    manifest = json.loads(
        (prepared.session_root / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("clean_shutdown") is not True or manifest.get(
        "formal_stitch_allowed"
    ) is not True:
        raise PhotoCaptureError(
            "Photo sequence ended without a clean formal RGB-D session"
        )
    output(f"Photo sequence saved to: {prepared.session_root}")
    return prepared.session_root


__all__ = [
    "PhotoCaptureBusyError",
    "PhotoCaptureError",
    "PhotoCaptureResult",
    "PhotoCaptureSettings",
    "PreparedPhotoCapture",
    "SoftwareTriggeredRGBDPhotoController",
    "photo_settings_from_config",
    "run_photo_sequence",
    "select_fastest_rgbd_profiles",
]
