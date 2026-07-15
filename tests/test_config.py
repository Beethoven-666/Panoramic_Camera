from __future__ import annotations

from pathlib import Path

import pytest

from panorama_demo.config import load_config
from panorama_demo.stitch_sequence import (
    _central_strip_diagnostic_config,
    _validate_safety_envelope,
)


def test_default_capture_uses_motion_capped_auto_exposure() -> None:
    config = load_config()

    assert config["capture"]["color_auto_exposure"] is True
    assert config["capture"]["color_exposure_us"] is None
    assert config["capture"]["color_ae_max_exposure_us"] == 800
    assert config["capture"]["diagnostic_unrestricted_auto_exposure"] is False
    assert config["capture"]["frame_sync"] is True
    assert config["capture"]["external_sync_output"] is True
    assert config["capture"]["fps"] == 30
    assert config["stitch"]["max_canvas_megapixels"] == 200
    assert config["stitch"]["diagnostic_force"] is False
    assert config["stitch"]["pose_backend"] == "hybrid_orbslam3_rgbd"
    assert config["stitch"]["dense_fusion_backend"] == "tsdf_plane_dense_rgbd"
    assert config["stitch"]["central_strip_diagnostic"] == {
        "enabled": False,
        "reference_scale_mode": "robust_aligned_depth_plane",
        "orientation_mode": "verified_camera_to_world",
        "maximum_central_band_fraction": 0.20,
        "minimum_pair_overlap_pixels": 96,
        "exposure_mode": "global_gain",
        "multiband_levels": 5,
    }
    assert config["stitch"]["pose_graph"]["enabled"] is True
    assert config["stitch"]["rgbd_projection"]["mode"] == (
        "orthographic_side_scan"
    )
    assert config["stitch"]["rgbd_projection"]["minimum_coverage_ratio"] == 0.95
    assert config["stitch"]["scan_seam"]["backend"] == (
        "graphcut_depth_constrained"
    )
    assert config["stitch"]["scan_seam"]["multiband_levels"] == 5
    legacy_formal_keys = {
        "model",
        "device",
        "inference_width",
        "max_points",
        "max_pair_canvas",
        "min_matches",
        "max_unistitch_reprojection_px",
        "allow_magsac_fallback",
        "prefer_magsac_layout",
        "min_magsac_inlier_ratio",
        "sequence_motion_model",
        "translation_anchor_y",
        "save_pair_previews",
    }
    assert legacy_formal_keys.isdisjoint(config["stitch"])


def test_default_rgbd_photo_mode_preserves_single_trigger_safety_contract() -> None:
    photo_mode = load_config()["capture"]["photo_mode"]

    assert photo_mode == {
        "enabled": True,
        "fastest_common_fps": True,
        "exposure_us": 800,
        "trigger_out_delay_us": 17000,
        "capture_timeout_ms": 8000,
        "prime_attempts": 8,
        "prime_timeout_ms": 1500,
        "gate_settle_ms": 250,
    }


def test_unrestricted_auto_exposure_config_is_explicitly_diagnostic() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "capture_unrestricted_auto_exposure.yaml"
    )

    assert config["capture"]["diagnostic_unrestricted_auto_exposure"] is True
    assert config["capture"]["color_auto_exposure"] is True
    assert config["capture"]["color_exposure_us"] is None
    assert config["capture"]["color_ae_max_exposure_us"] is None
    assert config["capture"]["diagnostic_replaced_auto_cap_us"] == 800
    assert config["stitch"]["diagnostic_force"] is True
    assert config["stitch"]["input_quality_gate"] is False
    assert config["stitch"]["scan_seam"]["quality_gate"] is False
    # Process-survival protection is intentionally inherited by diagnostics.
    assert config["stitch"]["max_canvas_megapixels"] == 200


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


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("maximum_motion_exposure_us",), 1201, "1200 us"),
        (("color_exposure_unit_us",), 1, "metadata unit"),
        (("max_canvas_megapixels",), 201, "200 MP"),
        (("rgbd_projection", "minimum_coverage_ratio"), 0.94, "95%"),
        (("rgbd_projection", "max_aggregate_megapixels"), 201, "200 MP"),
        (("rgbd_odometry", "minimum_fitness"), 0.14, "cannot be relaxed"),
        (("pose_quality", "maximum_total_rotation_deg"), 11.0, "cannot be relaxed"),
        (("scan_seam", "quality_gate"), False, "cannot disable"),
    ],
)
def test_formal_stitch_config_cannot_relax_safety_envelope(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = load_config()["stitch"]
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        _validate_safety_envelope(config, diagnostic_force=False)


def test_diagnostic_threshold_bypass_keeps_resource_hard_limits() -> None:
    config = load_config()["stitch"]
    config["maximum_motion_exposure_us"] = 60_000
    config["rgbd_odometry"]["minimum_fitness"] = 0.0
    config["pose_quality"]["maximum_total_rotation_deg"] = 180.0

    _validate_safety_envelope(config, diagnostic_force=True)

    config["rgbd_projection"]["max_aggregate_megapixels"] = 201
    with pytest.raises(ValueError, match="200 MP"):
        _validate_safety_envelope(config, diagnostic_force=True)


def test_central_strip_config_is_closed_and_cannot_enable_formal_path() -> None:
    config = load_config()["stitch"]
    config["central_strip_diagnostic"]["unexpected"] = 1

    with pytest.raises(ValueError, match="Unknown central_strip_diagnostic"):
        _central_strip_diagnostic_config(config, diagnostic_renderer=None)

    config = load_config()["stitch"]
    config["central_strip_diagnostic"]["enabled"] = True
    with pytest.raises(ValueError, match="only be activated"):
        _central_strip_diagnostic_config(config, diagnostic_renderer=None)

    effective = _central_strip_diagnostic_config(
        load_config()["stitch"], diagnostic_renderer=lambda **kwargs: kwargs
    )
    assert effective is not None
    assert effective["enabled"] is True
