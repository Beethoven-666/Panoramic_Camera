from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from panorama_demo import capture_orbbec as capture
from panorama_demo.photo_capture import (
    PhotoCaptureBusyError,
    PhotoCaptureError,
    PhotoCaptureSettings,
    PreparedPhotoCapture,
    SoftwareTriggeredRGBDPhotoController,
    run_photo_sequence,
    select_fastest_rgbd_profiles,
)
from panorama_demo.session import load_rgbd_session


class _Profile:
    def __init__(self, width: int, height: int, fmt: str, fps: int) -> None:
        self.width = width
        self.height = height
        self.fmt = fmt
        self.fps = fps

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def get_format(self) -> str:
        return self.fmt

    def get_fps(self) -> int:
        return self.fps


class _ProfileList:
    def __init__(self, profiles: list[_Profile]) -> None:
        self.profiles = profiles

    def get_count(self) -> int:
        return len(self.profiles)

    def get_stream_profile_by_index(self, index: int) -> _Profile:
        return self.profiles[index]


class _BaseProfile:
    """Mirror SDK profile lists that require explicit video downcasting."""

    def __init__(self, video_profile: _Profile) -> None:
        self.video_profile = video_profile

    def as_video_stream_profile(self) -> _Profile:
        return self.video_profile


class _BaseProfileList(_ProfileList):
    def get_stream_profile_by_index(self, index: int) -> _BaseProfile:
        return _BaseProfile(self.profiles[index])


class _Frame:
    def __init__(self, *, color: bool, index: int, exposure: int = 8) -> None:
        self.color = color
        self.index = index
        self.exposure = exposure
        if color:
            self.array = np.full((3, 4, 3), (20, 40, 60), dtype=np.uint8)
        else:
            self.array = np.full((3, 4), 10_000, dtype=np.uint16)

    def get_width(self) -> int:
        return 4

    def get_height(self) -> int:
        return 3

    def get_format(self) -> str:
        return "RGB" if self.color else "Y16"

    def get_data(self) -> bytes:
        return self.array.tobytes()

    def get_index(self) -> int:
        return self.index

    def get_timestamp_us(self) -> int:
        return self.index * 1_000 + (10 if self.color else 12)

    def get_system_timestamp_us(self) -> int:
        return self.index * 1_000 + (20 if self.color else 22)

    def get_depth_scale(self) -> float:
        return 0.1

    def has_metadata(self, metadata: str) -> bool:
        return metadata in {"exposure", "gain", "frame_number", "sensor_timestamp"}

    def get_metadata_value(self, metadata: str) -> int:
        values = {
            "exposure": self.exposure if self.color else 1,
            "gain": 16,
            "frame_number": self.index,
            "sensor_timestamp": self.index * 10,
        }
        return values[metadata]


class _FrameSet:
    def __init__(self, index: int) -> None:
        self.color = _Frame(color=True, index=index)
        self.depth = _Frame(color=False, index=index)

    def get_color_frame(self) -> _Frame:
        return self.color

    def get_depth_frame(self) -> _Frame:
        return self.depth


class _Pipeline:
    def __init__(self, device: _Device, sdk: _SDK) -> None:
        self.device = device
        self.sdk = sdk
        self.started = False
        self.frame_sync = False
        self.delivered_trigger = 0
        self.formal_timeout = False
        self.formal_started = threading.Event()
        self.release_formal: threading.Event | None = None
        self.queued_frames: list[_FrameSet] = []
        self.stop_failures = 0
        self.color_profiles = _ProfileList(
            [
                _Profile(4, 3, "RGB", 30),
                _Profile(4, 3, "RGB", 60),
                _Profile(4, 3, "MJPG", 60),
                _Profile(2, 2, "RGB", 90),
            ]
        )
        self.depth_profiles = _ProfileList(
            [
                _Profile(4, 3, "Y16", 30),
                _Profile(4, 3, "Y16", 60),
                _Profile(2, 2, "Y16", 90),
            ]
        )

    def get_stream_profile_list(self, sensor: str) -> _ProfileList:
        return self.color_profiles if sensor == "color" else self.depth_profiles

    def enable_frame_sync(self) -> None:
        self.frame_sync = True

    def start(self, _config: object) -> None:
        self.started = True

    def stop(self) -> None:
        if self.stop_failures > 0:
            self.stop_failures -= 1
            raise RuntimeError("pipeline stop failed")
        self.started = False

    def wait_for_frames(self, _timeout_ms: int) -> _FrameSet | None:
        if self.queued_frames:
            return self.queued_frames.pop(0)
        if self.device.trigger_count <= self.delivered_trigger:
            return None
        formal = self.device.bool_properties["output_gate"]
        if not formal and self.device.gate_off_timeouts_remaining > 0:
            self.device.gate_off_timeouts_remaining -= 1
            return None
        if formal:
            self.formal_started.set()
            if self.formal_timeout:
                return None
            if self.release_formal is not None:
                assert self.release_formal.wait(timeout=5)
        self.delivered_trigger = self.device.trigger_count
        return _FrameSet(self.delivered_trigger)

    def get_camera_param(self) -> object:
        intrinsic = SimpleNamespace(width=4, height=3, fx=3.0, fy=3.0, cx=2.0, cy=1.5)
        distortion = SimpleNamespace(k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0)
        transform = SimpleNamespace(
            rot=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            transform=[0.0, 0.0, 0.0],
        )
        return SimpleNamespace(
            depth_intrinsic=intrinsic,
            rgb_intrinsic=intrinsic,
            depth_distortion=distortion,
            rgb_distortion=distortion,
            transform=transform,
        )


class _Config:
    def __init__(self) -> None:
        self.profiles: list[_Profile] = []
        self.aggregate_mode: str | None = None

    def enable_stream(self, profile: _Profile) -> None:
        self.profiles.append(profile)

    def set_frame_aggregate_output_mode(self, mode: str) -> None:
        self.aggregate_mode = mode


class _DeviceInfo:
    def get_name(self) -> str:
        return "Fake Gemini 305"

    def get_serial_number(self) -> str:
        return "FAKE123"

    def get_firmware_version(self) -> str:
        return "1.0"

    def get_hardware_version(self) -> str:
        return "1.0"

    def get_connection_type(self) -> str:
        return "USB3"

    def get_vid(self) -> int:
        return 0x2BC5

    def get_pid(self) -> int:
        return 1


class _Device:
    def __init__(self) -> None:
        self.sync = SimpleNamespace(
            mode="STANDALONE",
            depth_delay_us=3,
            color_delay_us=4,
            trigger_to_image_delay_us=5,
            trigger_out_enable=False,
            trigger_out_delay_us=6,
            frames_per_trigger=2,
        )
        self.bool_properties = {
            "auto_capture": True,
            "output_gate": True,
            "auto_exposure": True,
        }
        self.int_properties = {"exposure": 10}
        self.trigger_count = 0
        self.external_pulse_count = 0
        self.gate_off_timeouts_remaining = 0
        self.sync_config_reopens_gate = True
        self.force_frames_per_trigger: int | None = None
        self.bool_write_failures: dict[str, int] = {}

    def get_device_info(self) -> _DeviceInfo:
        return _DeviceInfo()

    def is_property_supported(self, _prop: str, _permission: str) -> bool:
        return True

    def get_multi_device_sync_config(self) -> object:
        return self.sync

    def set_multi_device_sync_config(self, config: object) -> None:
        self.sync = config
        if self.sync_config_reopens_gate:
            self.bool_properties["output_gate"] = bool(
                self.sync.trigger_out_enable
            )
        if self.force_frames_per_trigger is not None:
            self.sync.frames_per_trigger = self.force_frames_per_trigger

    def get_bool_property(self, prop: str) -> bool:
        return self.bool_properties[prop]

    def set_bool_property(self, prop: str, value: bool) -> None:
        failures = self.bool_write_failures.get(prop, 0)
        if failures > 0:
            self.bool_write_failures[prop] = failures - 1
            raise RuntimeError(f"bool write failed: {prop}")
        self.bool_properties[prop] = bool(value)

    def get_int_property(self, prop: str) -> int:
        return self.int_properties[prop]

    def set_int_property(self, prop: str, value: int) -> None:
        self.int_properties[prop] = int(value)

    def get_int_property_range(self, _prop: str) -> object:
        return SimpleNamespace(min=1, max=100, step=1)

    def trigger_capture(self) -> None:
        self.trigger_count += 1
        if self.bool_properties["output_gate"]:
            self.external_pulse_count += 1


class _DeviceList:
    def __init__(self, device: _Device) -> None:
        self.device = device

    def get_count(self) -> int:
        return 1

    def get_device_by_index(self, _index: int) -> _Device:
        return self.device


class _SDK:
    OBPropertyID = SimpleNamespace(
        OB_DEVICE_AUTO_CAPTURE_ENABLE_BOOL="auto_capture",
        OB_PROP_SYNC_SIGNAL_TRIGGER_OUT_BOOL="output_gate",
        OB_PROP_COLOR_AUTO_EXPOSURE_BOOL="auto_exposure",
        OB_PROP_COLOR_EXPOSURE_INT="exposure",
    )
    OBPermissionType = SimpleNamespace(PERMISSION_WRITE="write")
    OBMultiDeviceSyncMode = SimpleNamespace(
        SOFTWARE_TRIGGERING="SOFTWARE_TRIGGERING"
    )
    OBFormat = SimpleNamespace(RGB="RGB", MJPG="MJPG", Y16="Y16")
    OBSensorType = SimpleNamespace(COLOR_SENSOR="color", DEPTH_SENSOR="depth")
    OBFrameAggregateOutputMode = SimpleNamespace(FULL_FRAME_REQUIRE="full")
    OBStreamType = SimpleNamespace(COLOR_STREAM="color")
    OBFrameMetadataType = SimpleNamespace(
        EXPOSURE="exposure",
        GAIN="gain",
        FRAME_NUMBER="frame_number",
        SENSOR_TIMESTAMP="sensor_timestamp",
    )

    def __init__(self) -> None:
        self.device = _Device()
        self.pipeline: _Pipeline | None = None
        self.pipeline_stop_failures_on_create = 0
        self.align_returns_none = False

    def Context(self) -> object:  # noqa: N802 - mirror SDK API
        return SimpleNamespace(query_devices=lambda: _DeviceList(self.device))

    def Pipeline(self, device: _Device) -> _Pipeline:  # noqa: N802
        self.pipeline = _Pipeline(device, self)
        self.pipeline.stop_failures = self.pipeline_stop_failures_on_create
        return self.pipeline

    def Config(self) -> _Config:  # noqa: N802
        return _Config()

    def AlignFilter(self, *, align_to_stream: str) -> object:  # noqa: N802
        assert align_to_stream == "color"
        return SimpleNamespace(
            process=lambda frames: None if self.align_returns_none else frames
        )

    @staticmethod
    def get_version() -> str:
        return "fake-sdk"


def _settings(tmp_path: Path, **overrides: object) -> PhotoCaptureSettings:
    values: dict[str, object] = {
        "output_root": tmp_path,
        "width": 4,
        "height": 3,
        "color_formats": ("RGB", "MJPG"),
        "exposure_us": 800,
        "trigger_out_delay_us": 17000,
        "capture_timeout_ms": 30,
        "prime_attempts": 8,
        "prime_timeout_ms": 1,
        "gate_settle_ms": 250,
        "jpeg_quality": 90,
        "depth_png_compression": 0,
    }
    values.update(overrides)
    return PhotoCaptureSettings(**values)  # type: ignore[arg-type]


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


def _controller(
    tmp_path: Path, sdk: _SDK, **overrides: object
) -> SoftwareTriggeredRGBDPhotoController:
    clock = _FakeClock()
    return SoftwareTriggeredRGBDPhotoController(
        _settings(tmp_path, **overrides),
        sdk=sdk,
        clock=clock,
        sleep=clock.sleep,
    )


class _SequenceController:
    def __init__(
        self,
        settings: PhotoCaptureSettings,
        *,
        fail_on_capture: int | None = None,
    ) -> None:
        self.settings = settings
        self.fail_on_capture = fail_on_capture
        self.capture_calls = 0
        self.close_calls = 0
        self.stop_reason: str | None = None
        self.session_root = settings.output_root / "run_fake_photo_sequence"

    def prepare(self) -> PreparedPhotoCapture:
        self.session_root.mkdir(parents=True)
        (self.session_root / "manifest.json").write_text(
            json.dumps(
                {
                    "clean_shutdown": False,
                    "formal_stitch_allowed": False,
                }
            ),
            encoding="utf-8",
        )
        return PreparedPhotoCapture(
            session_root=self.session_root,
            width=self.settings.width,
            height=self.settings.height,
            fps=60,
            color_format="MJPG",
            depth_format="Y16",
            exposure_us=self.settings.exposure_us,
            trigger_out_delay_us=self.settings.trigger_out_delay_us,
        )

    def capture_once(self) -> object:
        self.capture_calls += 1
        if self.fail_on_capture == self.capture_calls:
            raise PhotoCaptureError("synthetic frame failure")
        return SimpleNamespace(
            frame_id=self.capture_calls - 1,
            color_path=self.session_root / "color" / f"{self.capture_calls:08d}.jpg",
        )

    def set_stop_reason(self, reason: str) -> None:
        self.stop_reason = reason

    def close(self) -> None:
        self.close_calls += 1
        successful = self.fail_on_capture is None
        (self.session_root / "manifest.json").write_text(
            json.dumps(
                {
                    "clean_shutdown": successful,
                    "formal_stitch_allowed": successful,
                }
            ),
            encoding="utf-8",
        )


def _photo_args(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "photo_mode": True,
        "config": None,
        "output": tmp_path,
        "width": None,
        "height": None,
        "fps": None,
        "warmup_frames": None,
        "queue_size": None,
        "exposure_us": None,
        "gain": None,
        "white_balance": None,
        "auto_exposure": False,
        "diagnostic_unrestricted_auto_exposure": False,
        "raw_depth": None,
        "duration": None,
        "max_frames": 3,
        "no_preview": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fastest_profile_uses_highest_common_fps_before_format_order() -> None:
    sdk = _SDK()
    color = _ProfileList(
        [
            _Profile(4, 3, "RGB", 30),
            _Profile(4, 3, "RGB", 60),
            _Profile(4, 3, "MJPG", 60),
            _Profile(4, 3, "MJPG", 90),
        ]
    )
    depth = _ProfileList(
        [_Profile(4, 3, "Y16", 30), _Profile(4, 3, "Y16", 60)]
    )

    selected_color, selected_depth = select_fastest_rgbd_profiles(
        color,
        depth,
        width=4,
        height=3,
        color_formats=("MJPG", "RGB"),
        sdk=sdk,
    )

    assert selected_color.get_fps() == 60
    assert selected_color.get_format() == "MJPG"
    assert selected_depth.get_fps() == 60


def test_fastest_profile_respects_trigger_out_delay_safety() -> None:
    sdk = _SDK()
    color = _ProfileList(
        [_Profile(4, 3, "RGB", 30), _Profile(4, 3, "RGB", 60)]
    )
    depth = _ProfileList(
        [_Profile(4, 3, "Y16", 30), _Profile(4, 3, "Y16", 60)]
    )

    selected_color, selected_depth = select_fastest_rgbd_profiles(
        color,
        depth,
        width=4,
        height=3,
        color_formats=("RGB",),
        sdk=sdk,
        trigger_out_delay_us=17_000,
    )

    assert selected_color.get_fps() == 30
    assert selected_depth.get_fps() == 30


def test_fastest_profile_downcasts_real_sdk_base_stream_profiles() -> None:
    sdk = _SDK()
    color = _BaseProfileList([_Profile(4, 3, "RGB", 60)])
    depth = _BaseProfileList([_Profile(4, 3, "Y16", 60)])

    selected_color, selected_depth = select_fastest_rgbd_profiles(
        color,
        depth,
        width=4,
        height=3,
        color_formats=("RGB",),
        sdk=sdk,
    )

    assert selected_color.get_format() == "RGB"
    assert selected_depth.get_format() == "Y16"


def test_prepare_primes_with_gate_off_and_applies_exact_trigger_contract(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)

    prepared = controller.prepare()

    assert prepared.fps == 30
    assert sdk.device.trigger_count >= 1
    assert sdk.device.external_pulse_count == 0
    assert sdk.device.sync.mode == "SOFTWARE_TRIGGERING"
    assert sdk.device.sync.trigger_out_enable is True
    assert sdk.device.sync.frames_per_trigger == 1
    assert sdk.device.sync.trigger_out_delay_us == 17000
    assert sdk.device.bool_properties["output_gate"] is True
    assert sdk.device.bool_properties["auto_capture"] is False
    controller.close()
    assert sdk.device.sync.mode == "STANDALONE"
    assert sdk.device.sync.frames_per_trigger == 2
    assert sdk.device.bool_properties == {
        "auto_capture": True,
        "output_gate": True,
        "auto_exposure": True,
    }
    assert sdk.device.int_properties["exposure"] == 10


def test_bounded_priming_failure_never_emits_external_pulse(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    sdk.device.gate_off_timeouts_remaining = 8
    controller = _controller(tmp_path, sdk)

    with pytest.raises(PhotoCaptureError, match="bounded gate-off attempts"):
        controller.prepare()

    assert sdk.device.trigger_count == 8
    assert sdk.device.external_pulse_count == 0


def test_rgbd_pipeline_can_consume_first_gate_off_trigger_for_warmup(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    sdk.device.gate_off_timeouts_remaining = 1
    controller = _controller(tmp_path, sdk)

    controller.prepare()

    assert sdk.device.trigger_count == 2
    assert sdk.device.external_pulse_count == 0
    controller.close()


def test_three_formal_frames_issue_exactly_three_triggers_and_pulses(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    controller.prepare()
    triggers_before = sdk.device.trigger_count
    pulses_before = sdk.device.external_pulse_count

    for _ in range(3):
        controller.capture_once()

    assert sdk.device.trigger_count - triggers_before == 3
    assert sdk.device.external_pulse_count - pulses_before == 3
    controller.close()


def test_one_capture_issues_one_external_pulse_and_writes_formal_rgbd_session(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    prepared = controller.prepare()
    triggers_before = sdk.device.trigger_count
    pulses_before = sdk.device.external_pulse_count

    result = controller.capture_once()

    assert sdk.device.trigger_count - triggers_before == 1
    assert sdk.device.external_pulse_count - pulses_before == 1
    assert result.color_path.is_file()
    assert result.aligned_depth_path.is_file()
    assert result.frames_csv_path.is_file()
    assert len(list((prepared.session_root / "color").glob("*.jpg"))) == 1
    assert len(list((prepared.session_root / "depth_aligned").glob("*.png"))) == 1
    open_manifest = json.loads(
        (prepared.session_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert open_manifest["formal_stitch_allowed"] is False
    with pytest.raises(ValueError, match="not cleanly closed"):
        load_rgbd_session(prepared.session_root)
    controller.close()
    session = load_rgbd_session(prepared.session_root)
    assert len(session.frames) == 1
    manifest = json.loads(
        (prepared.session_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clean_shutdown"] is True
    assert manifest["capture_mode"] == "software_triggered_rgbd_photo_sequence"
    assert manifest["one_formal_trigger_per_capture"] is True
    assert manifest["photo_sequence"]["frames"] == 1


def test_timeout_never_retriggers_and_invalidates_sequence(tmp_path: Path) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    controller.prepare()
    assert sdk.pipeline is not None
    sdk.pipeline.formal_timeout = True
    triggers_before = sdk.device.trigger_count

    with pytest.raises(PhotoCaptureError, match="will not retrigger"):
        controller.capture_once()

    assert sdk.device.trigger_count - triggers_before == 1
    with pytest.raises(PhotoCaptureError, match="sequence is uncertain"):
        controller.capture_once()
    assert sdk.device.trigger_count - triggers_before == 1
    session_root = controller.session_root
    assert session_root is not None
    controller.close()
    manifest = json.loads(
        (session_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clean_shutdown"] is False
    assert manifest["formal_stitch_allowed"] is False
    assert len(manifest["capture_errors"]) >= 1


def test_concurrent_capture_is_rejected_without_second_trigger(tmp_path: Path) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk, capture_timeout_ms=2_000)
    controller.prepare()
    assert sdk.pipeline is not None
    release = threading.Event()
    sdk.pipeline.release_formal = release
    errors: list[BaseException] = []

    def first_capture() -> None:
        try:
            controller.capture_once()
        except BaseException as exc:  # pragma: no cover - diagnostic path
            errors.append(exc)

    triggers_before = sdk.device.trigger_count
    thread = threading.Thread(target=first_capture)
    thread.start()
    assert sdk.pipeline.formal_started.wait(timeout=2)
    with pytest.raises(PhotoCaptureBusyError):
        controller.capture_once()
    assert sdk.device.trigger_count - triggers_before == 1
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert sdk.device.trigger_count - triggers_before == 1
    controller.close()


def test_stale_queue_overflow_fails_before_formal_trigger(tmp_path: Path) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    controller.prepare()
    assert sdk.pipeline is not None
    sdk.pipeline.queued_frames.extend(_FrameSet(100 + index) for index in range(5))
    triggers_before = sdk.device.trigger_count

    with pytest.raises(PhotoCaptureError, match="exceeded its safe drain limit"):
        controller.capture_once()

    assert sdk.device.trigger_count == triggers_before
    assert sdk.device.external_pulse_count == 0
    with pytest.raises(PhotoCaptureError, match="sequence is uncertain"):
        controller.capture_once()
    assert sdk.device.trigger_count == triggers_before
    controller.close()


def test_writer_failure_after_trigger_never_retriggers(tmp_path: Path) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    controller.prepare()
    assert controller._writer is not None
    controller._writer.submit = lambda _packet: False  # type: ignore[method-assign]
    triggers_before = sdk.device.trigger_count

    with pytest.raises(PhotoCaptureError, match="writer queue rejected"):
        controller.capture_once()

    assert sdk.device.trigger_count - triggers_before == 1
    with pytest.raises(PhotoCaptureError, match="sequence is uncertain"):
        controller.capture_once()
    assert sdk.device.trigger_count - triggers_before == 1
    controller.close()


def test_missing_frame_exposure_metadata_fails_without_forging_or_retriggering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    controller.prepare()
    monkeypatch.setattr(
        sdk.OBFrameMetadataType,
        "EXPOSURE",
        "unavailable_exposure",
    )
    triggers_before = sdk.device.trigger_count

    with pytest.raises(PhotoCaptureError, match="metadata is required"):
        controller.capture_once()

    assert sdk.device.trigger_count - triggers_before == 1
    assert controller.session_root is not None
    assert not list((controller.session_root / "color").glob("*.jpg"))
    with pytest.raises(PhotoCaptureError, match="sequence is uncertain"):
        controller.capture_once()
    assert sdk.device.trigger_count - triggers_before == 1
    controller.close()


def test_pipeline_stop_failure_keeps_auto_capture_and_output_gate_safe_off(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    prepared = controller.prepare()
    assert sdk.pipeline is not None
    sdk.pipeline.stop_failures = 1

    with pytest.raises(PhotoCaptureError, match="pipeline stop failed"):
        controller.close()

    assert sdk.pipeline.started is True
    assert sdk.device.sync.mode == "SOFTWARE_TRIGGERING"
    assert sdk.device.bool_properties["auto_capture"] is False
    assert sdk.device.bool_properties["output_gate"] is False
    controller.close()
    assert sdk.pipeline.started is False
    assert sdk.device.sync.mode == "STANDALONE"
    assert sdk.device.bool_properties["auto_capture"] is True
    assert sdk.device.bool_properties["output_gate"] is True
    manifest = json.loads(
        (prepared.session_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clean_shutdown"] is False
    assert manifest["formal_stitch_allowed"] is False


def test_prepare_cleanup_does_not_restore_outputs_if_pipeline_stop_fails(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    sdk.pipeline_stop_failures_on_create = 1
    sdk.align_returns_none = True
    controller = _controller(tmp_path, sdk)

    with pytest.raises(PhotoCaptureError, match="pipeline stop failed"):
        controller.prepare()

    assert sdk.pipeline is not None
    assert sdk.pipeline.started is True
    assert sdk.device.sync.mode == "SOFTWARE_TRIGGERING"
    assert sdk.device.bool_properties["auto_capture"] is False
    assert sdk.device.bool_properties["output_gate"] is False
    sdk.align_returns_none = False
    controller.close()
    assert sdk.pipeline.started is False
    assert sdk.device.sync.mode == "STANDALONE"
    assert sdk.device.bool_properties["auto_capture"] is True
    assert sdk.device.bool_properties["output_gate"] is True


def test_partial_restore_failure_is_safe_off_and_close_can_retry(
    tmp_path: Path,
) -> None:
    sdk = _SDK()
    controller = _controller(tmp_path, sdk)
    prepared = controller.prepare()
    sdk.device.bool_write_failures["auto_capture"] = 1

    with pytest.raises(PhotoCaptureError, match="restore timed auto capture"):
        controller.close()

    assert sdk.device.bool_properties["auto_capture"] is False
    assert sdk.device.bool_properties["output_gate"] is False
    controller.close()
    assert sdk.device.bool_properties["auto_capture"] is True
    assert sdk.device.bool_properties["output_gate"] is True
    manifest = json.loads(
        (prepared.session_root / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clean_shutdown"] is False
    assert manifest["formal_stitch_allowed"] is False


def test_unsafe_sync_readback_fails_before_any_trigger(tmp_path: Path) -> None:
    sdk = _SDK()
    sdk.device.force_frames_per_trigger = 2
    controller = _controller(tmp_path, sdk)

    with pytest.raises(PhotoCaptureError, match="frames_per_trigger"):
        controller.prepare()

    assert sdk.device.trigger_count == 0
    assert sdk.device.external_pulse_count == 0


def test_formal_photo_exposure_cannot_be_relaxed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot exceed 800 us"):
        _settings(tmp_path, exposure_us=900).validated()


def test_physical_gate_settle_wait_cannot_be_relaxed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be below 250 ms"):
        _settings(tmp_path, gate_settle_ms=249).validated()


def test_formal_photo_sequence_requires_17000_us_trigger_out_delay(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="trigger_out_delay_us=17000"):
        _settings(tmp_path, trigger_out_delay_us=0).validated()


def test_formal_photo_sequence_locks_bounded_priming_attempts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="8 bounded priming attempts"):
        _settings(tmp_path, prime_attempts=2).validated()


def test_photo_sequence_runs_unthrottled_one_capture_at_a_time(
    tmp_path: Path,
) -> None:
    controllers: list[_SequenceController] = []

    def factory(settings: PhotoCaptureSettings) -> _SequenceController:
        controller = _SequenceController(settings)
        controllers.append(controller)
        return controller

    messages: list[str] = []
    session_root = run_photo_sequence(
        _photo_args(tmp_path, max_frames=3),
        controller_factory=factory,  # type: ignore[arg-type]
        key_reader=lambda: None,
        output=messages.append,
    )

    controller = controllers[0]
    assert session_root == controller.session_root
    assert controller.capture_calls == 3
    assert controller.close_calls == 1
    assert controller.stop_reason == "max_frames"
    assert controller.settings.exposure_us == 800
    assert any("no video preview" in message for message in messages)


def test_photo_sequence_stops_after_duration_on_a_complete_frame(
    tmp_path: Path,
) -> None:
    controllers: list[_SequenceController] = []
    ticks = iter((0.0, 0.0, 0.5, 2.1))

    def factory(settings: PhotoCaptureSettings) -> _SequenceController:
        controller = _SequenceController(settings)
        controllers.append(controller)
        return controller

    run_photo_sequence(
        _photo_args(tmp_path, duration=2.0),
        controller_factory=factory,  # type: ignore[arg-type]
        key_reader=lambda: None,
        clock=lambda: next(ticks),
        output=lambda _message: None,
    )

    assert controllers[0].capture_calls == 2
    assert controllers[0].stop_reason == "duration"


def test_photo_sequence_stops_after_q_without_an_extra_capture(
    tmp_path: Path,
) -> None:
    controllers: list[_SequenceController] = []

    def factory(settings: PhotoCaptureSettings) -> _SequenceController:
        controller = _SequenceController(settings)
        controllers.append(controller)
        return controller

    run_photo_sequence(
        _photo_args(tmp_path),
        controller_factory=factory,  # type: ignore[arg-type]
        key_reader=lambda: ord("q"),
        output=lambda _message: None,
    )

    assert controllers[0].capture_calls == 1
    assert controllers[0].stop_reason == "console_key"


def test_photo_sequence_rejects_manual_fps_override(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--fps"):
        run_photo_sequence(
            _photo_args(tmp_path, fps=30),
            output=lambda _message: None,
        )


def test_photo_sequence_stops_on_first_frame_failure_without_retry(
    tmp_path: Path,
) -> None:
    controllers: list[_SequenceController] = []

    def factory(settings: PhotoCaptureSettings) -> _SequenceController:
        controller = _SequenceController(settings, fail_on_capture=2)
        controllers.append(controller)
        return controller

    with pytest.raises(PhotoCaptureError, match="synthetic frame failure"):
        run_photo_sequence(
            _photo_args(tmp_path, max_frames=5),
            controller_factory=factory,  # type: ignore[arg-type]
            key_reader=lambda: None,
            output=lambda _message: None,
        )

    controller = controllers[0]
    assert controller.capture_calls == 2
    assert controller.close_calls == 1
    assert controller.stop_reason == "capture_error"


def test_g305_capture_parser_and_dispatch_expose_photo_sequence_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = capture.build_parser().parse_args(
        ["--photo-mode", "--max-frames", "2", "--output", str(tmp_path)]
    )
    called: list[object] = []

    def fake_run(received: object) -> Path:
        called.append(received)
        return tmp_path / "run_fake"

    import panorama_demo.photo_capture as photo_module

    monkeypatch.setattr(photo_module, "run_photo_sequence", fake_run)

    result = capture.run_capture(args)

    assert result == tmp_path / "run_fake"
    assert called == [args]
