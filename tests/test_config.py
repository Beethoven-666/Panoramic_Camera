from __future__ import annotations

from pathlib import Path

import pytest

from panorama_demo.config import load_config


def test_default_capture_uses_motion_capped_auto_exposure() -> None:
    config = load_config()

    assert config["capture"]["color_auto_exposure"] is True
    assert config["capture"]["color_exposure_us"] is None
    assert config["capture"]["color_ae_max_exposure_us"] == 800


@pytest.mark.parametrize(
    ("exposure_yaml", "expected_auto_exposure", "expected_exposure"),
    [
        ("800", False, 800),
        ("null", True, None),
    ],
)
def test_legacy_exposure_override_infers_auto_exposure_mode(
    tmp_path: Path,
    exposure_yaml: str,
    expected_auto_exposure: bool,
    expected_exposure: int | None,
) -> None:
    custom = tmp_path / "capture.yaml"
    custom.write_text(
        f"capture:\n  color_exposure_us: {exposure_yaml}\n",
        encoding="utf-8",
    )

    config = load_config(custom)

    assert config["capture"]["color_auto_exposure"] is expected_auto_exposure
    assert config["capture"]["color_exposure_us"] == expected_exposure


def test_explicit_auto_exposure_mode_is_not_overridden(tmp_path: Path) -> None:
    custom = tmp_path / "capture.yaml"
    custom.write_text(
        "capture:\n"
        "  color_auto_exposure: true\n"
        "  color_exposure_us: 800\n",
        encoding="utf-8",
    )

    config = load_config(custom)

    assert config["capture"]["color_auto_exposure"] is True
    assert config["capture"]["color_exposure_us"] == 800
