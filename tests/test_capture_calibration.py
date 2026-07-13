from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import panorama_demo.capture_orbbec as capture
from panorama_demo.capture_orbbec import _calibration_to_dict
from panorama_demo.config import load_config


def _intrinsic() -> SimpleNamespace:
    return SimpleNamespace(width=1280, height=800, fx=600, fy=601, cx=640, cy=400)


def _distortion() -> SimpleNamespace:
    return SimpleNamespace(k1=0, k2=0, k3=0, k4=0, k5=0, k6=0, p1=0, p2=0)


def test_calibration_flattens_sdk_matrix_arrays() -> None:
    camera = SimpleNamespace(
        depth_intrinsic=_intrinsic(),
        rgb_intrinsic=_intrinsic(),
        depth_distortion=_distortion(),
        rgb_distortion=_distortion(),
        transform=SimpleNamespace(
            rot=np.eye(3, dtype=np.float32),
            transform=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        ),
    )
    result = _calibration_to_dict(camera)
    assert result["depth_alignment"] == {
        "enabled": True,
        "aligned_to": "color",
        "method": "software",
        "producer": "pyorbbecsdk.AlignFilter(COLOR_STREAM)",
    }
    assert result["depth_to_color"]["rotation_row_major"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["depth_to_color"]["translation_mm"] == [1.0, 2.0, 3.0]


def test_color_auto_exposure_is_explicitly_enabled(monkeypatch) -> None:
    boolean_calls: list[tuple[str, bool]] = []
    integer_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        capture,
        "_set_bool_property",
        lambda _device, _sdk, name, value: boolean_calls.append((name, value)) or value,
    )
    monkeypatch.setattr(
        capture,
        "_set_int_property",
        lambda _device, _sdk, name, value: integer_calls.append((name, value)) or value,
    )

    applied = capture._configure_color(
        object(),
        object(),
        {
            "color_auto_exposure": True,
            "color_exposure_us": None,
            "color_gain": None,
            "color_white_balance": None,
            "color_anti_flicker": False,
        },
    )

    assert applied["auto_exposure"] is True
    assert ("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", True) in boolean_calls
    assert integer_calls == []


def test_color_auto_exposure_applies_motion_safe_cap(monkeypatch) -> None:
    boolean_calls: list[tuple[str, bool]] = []
    integer_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        capture,
        "_set_bool_property",
        lambda _device, _sdk, name, value: boolean_calls.append((name, value)) or value,
    )
    monkeypatch.setattr(
        capture,
        "_set_int_property",
        lambda _device, _sdk, name, value: integer_calls.append((name, value)) or value,
    )

    applied = capture._configure_color(
        object(),
        object(),
        {
            "color_auto_exposure": True,
            "color_exposure_us": None,
            "color_ae_max_exposure_us": 800,
            "color_gain": None,
            "color_white_balance": None,
            "color_anti_flicker": False,
        },
    )

    assert applied["ae_max_exposure_us"] == 800
    assert applied["auto_exposure"] is True
    assert integer_calls == [("OB_PROP_COLOR_AE_MAX_EXPOSURE_INT", 8)]
    assert ("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", True) in boolean_calls


def test_diagnostic_unrestricted_auto_exposure_uses_verified_device_max(
    monkeypatch,
) -> None:
    property_id = object()
    state = {"value": 8}
    device = SimpleNamespace(
        get_int_property_range=lambda prop: SimpleNamespace(min=1, max=333, step=1),
        set_int_property=lambda prop, value: state.update(value=value),
        get_int_property=lambda prop: state["value"],
    )
    sdk = SimpleNamespace(
        OBPropertyID=SimpleNamespace(OB_PROP_COLOR_AE_MAX_EXPOSURE_INT=property_id)
    )
    monkeypatch.setattr(capture, "_set_bool_property", lambda *_args: True)

    applied = capture._configure_color(
        device,
        sdk,
        {
            "diagnostic_unrestricted_auto_exposure": True,
            "diagnostic_replaced_auto_cap_us": 800,
            "color_auto_exposure": True,
            "color_exposure_us": None,
            "color_ae_max_exposure_us": None,
            "color_gain": None,
            "color_white_balance": None,
            "color_anti_flicker": False,
        },
    )

    assert state["value"] == 333
    assert applied["auto_exposure"] is True
    assert applied["ae_max_exposure_requested"] == 333
    assert applied["ae_max_exposure_requested_us"] == 33_300
    assert applied["ae_max_exposure"] == 333
    assert applied["ae_max_exposure_us"] == 33_300
    assert applied["ae_max_exposure_device_clamped"] is False


def test_diagnostic_unrestricted_auto_exposure_accepts_frame_rate_clamp(
    monkeypatch,
) -> None:
    state = {"value": 8}

    def set_clamped_value(_prop, value: int) -> None:
        state["value"] = min(value, 300)

    device = SimpleNamespace(
        get_int_property_range=lambda _prop: SimpleNamespace(
            min=1, max=1990, step=1
        ),
        set_int_property=set_clamped_value,
        get_int_property=lambda _prop: state["value"],
    )
    sdk = SimpleNamespace(
        OBPropertyID=SimpleNamespace(OB_PROP_COLOR_AE_MAX_EXPOSURE_INT=object())
    )
    monkeypatch.setattr(capture, "_set_bool_property", lambda *_args: True)
    options = {
        "diagnostic_unrestricted_auto_exposure": True,
        "diagnostic_replaced_auto_cap_us": 800,
        "color_auto_exposure": True,
        "color_exposure_us": None,
        "color_ae_max_exposure_us": None,
        "color_gain": None,
        "color_white_balance": None,
        "color_anti_flicker": False,
    }

    applied = capture._configure_color(device, sdk, options)
    summary = capture._color_exposure_control_summary(options, applied)

    assert applied["ae_max_exposure_requested"] == 1990
    assert applied["ae_max_exposure_requested_us"] == 199_000
    assert applied["ae_max_exposure"] == 300
    assert applied["ae_max_exposure_us"] == 30_000
    assert applied["ae_max_exposure_device_clamped"] is True
    assert summary["replaced_motion_safe_cap_us"] == 800
    assert summary["requested_device_max_auto_cap_us"] == 199_000
    assert summary["applied_auto_cap_us"] == 30_000
    assert summary["device_clamped_auto_cap"] is True


def test_diagnostic_unrestricted_auto_exposure_rejects_unverified_device_max(
    monkeypatch,
) -> None:
    device = SimpleNamespace(
        get_int_property_range=lambda _prop: SimpleNamespace(min=1, max=333, step=1),
        set_int_property=lambda _prop, _value: None,
        get_int_property=lambda _prop: 8,
    )
    sdk = SimpleNamespace(
        OBPropertyID=SimpleNamespace(OB_PROP_COLOR_AE_MAX_EXPOSURE_INT=object())
    )
    monkeypatch.setattr(capture, "_set_bool_property", lambda *_args: True)

    with pytest.raises(RuntimeError, match="did not lift"):
        capture._configure_color(
            device,
            sdk,
            {
                "diagnostic_unrestricted_auto_exposure": True,
                "diagnostic_replaced_auto_cap_us": 800,
                "color_auto_exposure": True,
                "color_exposure_us": None,
                "color_ae_max_exposure_us": None,
                "color_gain": None,
                "color_white_balance": None,
                "color_anti_flicker": False,
            },
        )


def test_unsupported_auto_exposure_cap_falls_back_to_fixed_cap(monkeypatch) -> None:
    boolean_calls: list[tuple[str, bool]] = []
    integer_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        capture,
        "_set_bool_property",
        lambda _device, _sdk, name, value: boolean_calls.append((name, value)) or value,
    )

    def set_integer(_device, _sdk, name: str, value: int) -> int | None:
        integer_calls.append((name, value))
        return None if name == "OB_PROP_COLOR_AE_MAX_EXPOSURE_INT" else value

    monkeypatch.setattr(capture, "_set_int_property", set_integer)

    applied = capture._configure_color(
        object(),
        object(),
        {
            "color_auto_exposure": True,
            "color_exposure_us": None,
            "color_ae_max_exposure_us": 800,
            "color_gain": None,
            "color_white_balance": None,
            "color_anti_flicker": False,
        },
    )

    assert applied["auto_exposure"] is False
    assert applied["exposure_us"] == 800
    assert ("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", False) in boolean_calls
    assert integer_calls == [
        ("OB_PROP_COLOR_AE_MAX_EXPOSURE_INT", 8),
        ("OB_PROP_COLOR_EXPOSURE_INT", 8),
    ]


def test_color_configuration_rejects_unverified_exposure_mode(monkeypatch) -> None:
    monkeypatch.setattr(capture, "_set_bool_property", lambda *_args: None)
    monkeypatch.setattr(capture, "_set_int_property", lambda *_args: 8)

    with pytest.raises(RuntimeError, match="did not apply"):
        capture._configure_color(
            object(),
            object(),
            {
                "color_auto_exposure": True,
                "color_exposure_us": None,
                "color_ae_max_exposure_us": 800,
                "color_gain": None,
                "color_white_balance": None,
                "color_anti_flicker": False,
            },
        )


def test_fixed_exposure_disables_auto_exposure(monkeypatch) -> None:
    boolean_calls: list[tuple[str, bool]] = []
    integer_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        capture,
        "_set_bool_property",
        lambda _device, _sdk, name, value: boolean_calls.append((name, value)) or value,
    )
    monkeypatch.setattr(
        capture,
        "_set_int_property",
        lambda _device, _sdk, name, value: integer_calls.append((name, value)) or value,
    )

    applied = capture._configure_color(
        object(),
        object(),
        {
            "color_auto_exposure": False,
            "color_exposure_us": 1000,
            "color_ae_max_exposure_us": 800,
            "color_gain": None,
            "color_white_balance": None,
            "color_anti_flicker": False,
        },
    )

    assert applied == {
        "auto_exposure": False,
        "exposure": 10,
        "exposure_us": 1000,
    }
    assert ("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", False) in boolean_calls
    assert integer_calls == [("OB_PROP_COLOR_EXPOSURE_INT", 10)]


def test_fixed_exposure_rejects_a_clamped_device_value(monkeypatch) -> None:
    monkeypatch.setattr(capture, "_set_bool_property", lambda *_args: False)
    monkeypatch.setattr(capture, "_set_int_property", lambda *_args: 9)

    with pytest.raises(RuntimeError, match="requested 1000 us, applied 900 us"):
        capture._configure_color(
            object(),
            object(),
            {
                "color_auto_exposure": False,
                "color_exposure_us": 1000,
                "color_ae_max_exposure_us": 800,
                "color_gain": None,
                "color_white_balance": None,
                "color_anti_flicker": False,
            },
        )


def test_fixed_exposure_rejects_missing_device_readback(monkeypatch) -> None:
    monkeypatch.setattr(capture, "_set_bool_property", lambda *_args: False)
    monkeypatch.setattr(capture, "_set_int_property", lambda *_args: None)

    with pytest.raises(RuntimeError, match="did not apply the requested"):
        capture._configure_color(
            object(),
            object(),
            {
                "color_auto_exposure": False,
                "color_exposure_us": 1000,
                "color_gain": None,
                "color_white_balance": None,
                "color_anti_flicker": False,
            },
        )


def test_manual_metadata_uses_requested_value_not_auto_cap() -> None:
    options = {
        "color_auto_exposure": False,
        "color_exposure_us": 1500,
        "color_ae_max_exposure_us": 800,
    }

    assert capture._color_exposure_metadata_violation(options, 15) is None
    assert "requested manual" in str(
        capture._color_exposure_metadata_violation(options, 8)
    )


def test_auto_metadata_still_enforces_exposure_cap() -> None:
    options = {
        "color_auto_exposure": True,
        "color_exposure_us": None,
        "color_ae_max_exposure_us": 800,
    }

    assert capture._color_exposure_metadata_violation(options, 9) is None
    assert "auto-exposure limit" in str(
        capture._color_exposure_metadata_violation(options, 10)
    )


def test_diagnostic_unrestricted_auto_exposure_allows_long_metadata() -> None:
    options = {
        "diagnostic_unrestricted_auto_exposure": True,
        "color_auto_exposure": True,
        "color_exposure_us": None,
        "color_ae_max_exposure_us": None,
    }

    assert capture._color_exposure_metadata_violation(options, 301) is None


def test_exposure_quantization_is_consistent_at_half_unit() -> None:
    assert capture._color_exposure_units(850) == 9


@pytest.mark.parametrize(
    ("options", "applied", "effective_mode"),
    [
        (
            {
                "color_auto_exposure": False,
                "color_exposure_us": 1000,
                "color_ae_max_exposure_us": 800,
            },
            {"auto_exposure": False, "exposure_us": 1000},
            "manual",
        ),
        (
            {
                "color_auto_exposure": True,
                "color_exposure_us": None,
                "color_ae_max_exposure_us": 800,
            },
            {"auto_exposure": False, "exposure_us": 800},
            "manual_fallback",
        ),
    ],
)
def test_exposure_manifest_distinguishes_requested_and_effective_modes(
    options: dict[str, object], applied: dict[str, object], effective_mode: str
) -> None:
    summary = capture._color_exposure_control_summary(options, applied)

    assert summary["effective_mode"] == effective_mode
    assert summary["requested_mode"] == (
        "auto" if options["color_auto_exposure"] else "manual"
    )
    if summary["requested_mode"] == "manual":
        assert summary["requested_auto_cap_us"] is None


@pytest.mark.parametrize("flag", ["--exposure-us", "--manual-exposure-us"])
def test_manual_exposure_cli_aliases(flag: str) -> None:
    args = capture.build_parser().parse_args([flag, "900"])

    assert args.exposure_us == 900
    assert args.auto_exposure is False


def test_exposure_cli_modes_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        capture.build_parser().parse_args(
            ["--auto-exposure", "--exposure-us", "900"]
        )

    with pytest.raises(SystemExit):
        capture.build_parser().parse_args(
            ["--diagnostic-unrestricted-auto-exposure", "--auto-exposure"]
        )

    with pytest.raises(SystemExit):
        capture.build_parser().parse_args(
            ["--diagnostic-unrestricted-auto-exposure", "--exposure-us", "900"]
        )


def test_default_cli_keeps_configured_auto_exposure_mode() -> None:
    args = capture.build_parser().parse_args([])

    assert args.auto_exposure is False
    assert args.diagnostic_unrestricted_auto_exposure is False
    assert args.exposure_us is None


def test_diagnostic_unrestricted_cli_resolves_explicit_capture_mode() -> None:
    options = dict(load_config()["capture"])
    args = capture.build_parser().parse_args(
        ["--diagnostic-unrestricted-auto-exposure"]
    )

    diagnostic_only = capture._apply_color_exposure_mode(options, args)

    assert diagnostic_only is True
    assert options["diagnostic_unrestricted_auto_exposure"] is True
    assert options["color_auto_exposure"] is True
    assert options["color_exposure_us"] is None
    assert options["color_ae_max_exposure_us"] is None
    assert options["diagnostic_replaced_auto_cap_us"] == 800


def test_standard_auto_exposure_cli_clears_diagnostic_config() -> None:
    diagnostic_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "capture_unrestricted_auto_exposure.yaml"
    )
    options = dict(load_config(diagnostic_config)["capture"])
    args = capture.build_parser().parse_args(["--auto-exposure"])

    diagnostic_only = capture._apply_color_exposure_mode(options, args)

    assert diagnostic_only is False
    assert options["diagnostic_unrestricted_auto_exposure"] is False
    assert options["color_auto_exposure"] is True
    assert options["color_exposure_us"] is None
    assert options["color_ae_max_exposure_us"] == 800
    assert "diagnostic_replaced_auto_cap_us" not in options


def test_uncapped_auto_exposure_requires_explicit_diagnostic_mode() -> None:
    options = {
        "color_auto_exposure": True,
        "color_exposure_us": None,
        "color_ae_max_exposure_us": None,
    }
    args = capture.build_parser().parse_args([])

    with pytest.raises(ValueError, match="diagnostic-only"):
        capture._apply_color_exposure_mode(options, args)


def test_diagnostic_unrestricted_mode_requires_formal_cap_baseline() -> None:
    options = {
        "diagnostic_unrestricted_auto_exposure": True,
        "color_auto_exposure": True,
        "color_exposure_us": None,
        "color_ae_max_exposure_us": None,
    }
    args = capture.build_parser().parse_args([])

    with pytest.raises(ValueError, match="requires a positive"):
        capture._apply_color_exposure_mode(options, args)


def test_diagnostic_capture_manifest_is_marked_before_camera_discovery(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "captures"
    device_list = SimpleNamespace(get_count=lambda: 0)
    context = SimpleNamespace(query_devices=lambda: device_list)
    monkeypatch.setitem(
        sys.modules,
        "pyorbbecsdk",
        SimpleNamespace(Context=lambda: context),
    )
    args = capture.build_parser().parse_args(
        [
            "--output",
            str(output),
            "--diagnostic-unrestricted-auto-exposure",
        ]
    )

    with pytest.raises(RuntimeError, match="No Orbbec camera"):
        capture.run_capture(args)

    sessions = list(output.glob("run_*"))
    assert len(sessions) == 1
    manifest = json.loads(
        (sessions[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["capture_mode"] == "diagnostic_unrestricted_auto_exposure"
    assert manifest["diagnostic_only"] is True
    assert manifest["formal_stitch_allowed"] is False
    assert manifest["capture_options"]["color_ae_max_exposure_us"] is None
    assert manifest["capture_options"]["diagnostic_replaced_auto_cap_us"] == 800


def test_invalid_manual_exposure_fails_before_session_creation(tmp_path) -> None:
    output = tmp_path / "captures"
    args = capture.build_parser().parse_args(
        ["--output", str(output), "--exposure-us", "0"]
    )

    with pytest.raises(ValueError, match="must be positive"):
        capture.run_capture(args)

    assert not output.exists()


def test_external_sync_output_uses_primary_mode_and_capture_fps() -> None:
    primary_mode = SimpleNamespace(name="PRIMARY")
    standalone_mode = SimpleNamespace(name="STANDALONE")
    config = SimpleNamespace(
        mode=standalone_mode,
        trigger_out_enable=False,
        color_delay_us=11,
        depth_delay_us=12,
        trigger_to_image_delay_us=13,
        trigger_out_delay_us=14,
        frames_per_trigger=2,
    )
    device = SimpleNamespace()
    device.get_multi_device_sync_config = lambda: config
    applied: list[SimpleNamespace] = []
    device.set_multi_device_sync_config = lambda value: applied.append(value)
    sdk = SimpleNamespace(
        OBMultiDeviceSyncMode=SimpleNamespace(PRIMARY=primary_mode)
    )

    result = capture._configure_external_sync_output(
        device, sdk, {"external_sync_output": True, "fps": 30}
    )

    assert applied == [config]
    assert config.mode is primary_mode
    assert config.trigger_out_enable is True
    assert config.trigger_out_delay_us == 0
    assert config.color_delay_us == 11
    assert config.depth_delay_us == 12
    assert config.trigger_to_image_delay_us == 13
    assert config.frames_per_trigger == 2
    assert result["enabled"] is True
    assert result["readback_verified"] is True
    assert result["mode"] == "PRIMARY"
    assert result["expected_frequency_hz"] == 30
    assert result["frequency_source"] == "capture_fps"


def test_external_sync_output_rejects_unverified_readback() -> None:
    primary_mode = SimpleNamespace(name="PRIMARY")
    standalone_mode = SimpleNamespace(name="STANDALONE")
    requested = SimpleNamespace(
        mode=standalone_mode,
        trigger_out_enable=False,
        trigger_out_delay_us=0,
    )
    applied = SimpleNamespace(
        mode=standalone_mode,
        trigger_out_enable=False,
        trigger_out_delay_us=0,
    )
    reads = iter((requested, applied))
    device = SimpleNamespace(
        get_multi_device_sync_config=lambda: next(reads),
        set_multi_device_sync_config=lambda _value: None,
    )
    sdk = SimpleNamespace(
        OBMultiDeviceSyncMode=SimpleNamespace(PRIMARY=primary_mode)
    )

    with pytest.raises(RuntimeError, match="did not enter PRIMARY"):
        capture._configure_external_sync_output(
            device, sdk, {"external_sync_output": True, "fps": 30}
        )


def test_external_sync_output_rejects_disabled_trigger_readback() -> None:
    primary_mode = SimpleNamespace(name="PRIMARY")
    requested = SimpleNamespace(
        mode=SimpleNamespace(name="STANDALONE"),
        trigger_out_enable=False,
        trigger_out_delay_us=0,
    )
    applied = SimpleNamespace(
        mode=primary_mode,
        trigger_out_enable=False,
        trigger_out_delay_us=0,
    )
    reads = iter((requested, applied))
    device = SimpleNamespace(
        get_multi_device_sync_config=lambda: next(reads),
        set_multi_device_sync_config=lambda _value: None,
    )
    sdk = SimpleNamespace(
        OBMultiDeviceSyncMode=SimpleNamespace(PRIMARY=primary_mode)
    )

    with pytest.raises(RuntimeError, match="did not enable"):
        capture._configure_external_sync_output(
            device, sdk, {"external_sync_output": True, "fps": 30}
        )


def test_external_sync_output_rejects_nonzero_output_delay_readback() -> None:
    primary_mode = SimpleNamespace(name="PRIMARY")
    requested = SimpleNamespace(
        mode=SimpleNamespace(name="STANDALONE"),
        trigger_out_enable=False,
        trigger_out_delay_us=7,
    )
    applied = SimpleNamespace(
        mode=primary_mode,
        trigger_out_enable=True,
        trigger_out_delay_us=7,
    )
    reads = iter((requested, applied))
    device = SimpleNamespace(
        get_multi_device_sync_config=lambda: next(reads),
        set_multi_device_sync_config=lambda _value: None,
    )
    sdk = SimpleNamespace(
        OBMultiDeviceSyncMode=SimpleNamespace(PRIMARY=primary_mode)
    )

    with pytest.raises(RuntimeError, match="zero external sync output delay"):
        capture._configure_external_sync_output(
            device, sdk, {"external_sync_output": True, "fps": 30}
        )


def test_external_sync_output_rejects_missing_sdk_api() -> None:
    with pytest.raises(RuntimeError, match="does not expose"):
        capture._configure_external_sync_output(
            object(), object(), {"external_sync_output": True, "fps": 30}
        )


def test_external_sync_output_can_be_disabled_without_touching_device() -> None:
    result = capture._configure_external_sync_output(
        object(), object(), {"external_sync_output": False, "fps": 30}
    )

    assert result == {"enabled": False}


def test_external_sync_failure_is_recorded_before_stream_start(
    tmp_path, monkeypatch
) -> None:
    device = object()
    device_list = SimpleNamespace(
        get_count=lambda: 1,
        get_device_by_index=lambda _index: device,
    )
    context = SimpleNamespace(query_devices=lambda: device_list)
    sdk = SimpleNamespace(Context=lambda: context, get_version=lambda: "test")
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
    monkeypatch.setattr(capture, "_device_info", lambda _device: {"name": "fake"})
    monkeypatch.setattr(capture.importlib_metadata, "version", lambda _name: "test")
    monkeypatch.setattr(
        capture,
        "_configure_color",
        lambda *_args: {"auto_exposure": True},
    )
    monkeypatch.setattr(
        capture,
        "_configure_external_sync_output",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("external sync readback failed")
        ),
    )
    args = capture.build_parser().parse_args(["--output", str(tmp_path)])

    with pytest.raises(RuntimeError, match="external sync readback failed"):
        capture.run_capture(args)

    sessions = list(tmp_path.glob("run_*"))
    assert len(sessions) == 1
    manifest = json.loads(
        (sessions[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["clean_shutdown"] is False
    assert manifest["capture_error"] == {
        "type": "RuntimeError",
        "message": "external sync readback failed",
    }
    assert not (sessions[0] / "frames.csv").exists()


def test_property_configuration_failure_is_recorded_in_manifest(
    tmp_path, monkeypatch
) -> None:
    device = object()
    device_list = SimpleNamespace(
        get_count=lambda: 1,
        get_device_by_index=lambda _index: device,
    )
    context = SimpleNamespace(query_devices=lambda: device_list)
    sdk = SimpleNamespace(Context=lambda: context, get_version=lambda: "test")
    monkeypatch.setitem(sys.modules, "pyorbbecsdk", sdk)
    monkeypatch.setattr(capture, "_device_info", lambda _device: {"name": "fake"})
    monkeypatch.setattr(capture.importlib_metadata, "version", lambda _name: "test")
    monkeypatch.setattr(
        capture,
        "_configure_color",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("manual readback failed")),
    )
    args = capture.build_parser().parse_args(
        ["--output", str(tmp_path), "--exposure-us", "1000"]
    )

    with pytest.raises(RuntimeError, match="manual readback failed"):
        capture.run_capture(args)

    sessions = list(tmp_path.glob("run_*"))
    assert len(sessions) == 1
    manifest = json.loads(
        (sessions[0] / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["capture_error"] == {
        "type": "RuntimeError",
        "message": "manual readback failed",
    }


@pytest.mark.parametrize(
    ("auto_exposure", "exposure_us", "message"),
    [
        (True, 800, "must be null"),
        (False, None, "is required"),
        (False, 0, "must be positive"),
    ],
)
def test_color_exposure_rejects_contradictory_modes_before_device_write(
    monkeypatch,
    auto_exposure: bool,
    exposure_us: int | None,
    message: str,
) -> None:
    boolean_calls: list[tuple[str, bool]] = []
    integer_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        capture,
        "_set_bool_property",
        lambda _device, _sdk, name, value: boolean_calls.append((name, value)),
    )
    monkeypatch.setattr(
        capture,
        "_set_int_property",
        lambda _device, _sdk, name, value: integer_calls.append((name, value)),
    )

    with pytest.raises(ValueError, match=message):
        capture._configure_color(
            object(),
            object(),
            {
                "color_auto_exposure": auto_exposure,
                "color_exposure_us": exposure_us,
                "color_gain": None,
                "color_white_balance": None,
                "color_anti_flicker": False,
            },
        )

    assert boolean_calls == []
    assert integer_calls == []
