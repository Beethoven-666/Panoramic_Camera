from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import panorama_demo.capture_orbbec as capture
from panorama_demo.capture_orbbec import _calibration_to_dict


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


def test_default_cli_keeps_configured_auto_exposure_mode() -> None:
    args = capture.build_parser().parse_args([])

    assert args.auto_exposure is False
    assert args.exposure_us is None


def test_invalid_manual_exposure_fails_before_session_creation(tmp_path) -> None:
    output = tmp_path / "captures"
    args = capture.build_parser().parse_args(
        ["--output", str(output), "--exposure-us", "0"]
    )

    with pytest.raises(ValueError, match="must be positive"):
        capture.run_capture(args)

    assert not output.exists()


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
