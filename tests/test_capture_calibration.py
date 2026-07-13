from __future__ import annotations

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
            "color_exposure_us": 800,
            "color_gain": None,
            "color_white_balance": None,
            "color_anti_flicker": False,
        },
    )

    assert applied == {
        "auto_exposure": False,
        "exposure": 8,
        "exposure_us": 800,
    }
    assert ("OB_PROP_COLOR_AUTO_EXPOSURE_BOOL", False) in boolean_calls
    assert integer_calls == [("OB_PROP_COLOR_EXPOSURE_INT", 8)]


@pytest.mark.parametrize(
    ("auto_exposure", "exposure_us", "message"),
    [
        (True, 800, "must be null"),
        (False, None, "is required"),
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
