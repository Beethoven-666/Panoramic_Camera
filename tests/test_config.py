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
    assert config["capture"]["color_auto_white_balance"] is True
    assert config["capture"]["color_white_balance"] is None
    assert config["capture"]["lock_color_controls_after_warmup"] is True
    assert config["capture"]["post_lock_verified_frames"] == 2
    assert config["capture"]["require_locked_control_metadata"] is True
    assert config["capture"]["frame_sync"] is True
    assert config["capture"]["external_sync_output"] is True
    assert config["capture"]["fps"] == 30
    assert config["stitch"]["max_canvas_megapixels"] == 200
    assert config["stitch"]["diagnostic_force"] is False
    assert config["stitch"]["handoff_fallback_policy"] == {
        "publish_degraded": True,
        "local_apap_flow_enabled": False,
        "manual_review_for_grade_c": True,
    }
    assert config["stitch"]["pose_backend"] == "hybrid_orbslam3_rgbd"
    assert config["stitch"]["sequence_blend_mode"] == (
        "calibrated_rgb_pushbroom"
    )
    assert "dense_fusion_backend" not in config["stitch"]
    assert "rgbd_projection" not in config["stitch"]
    assert config["stitch"]["tsdf_visualization"] == {
        "enabled": True,
        "voxel_length_mm": 5.0,
        "sdf_truncation_mm": 20.0,
        "maximum_depth_mm": 10000.0,
    }
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
    assert config["stitch"]["calibrated_rgb_pushbroom"] == {
        "mode": "calibrated_rgb_pushbroom",
        "maximum_central_band_fraction": 0.20,
        "endpoint_outer_half_fov": True,
        "seam_search_width_pixels": 64,
        "max_canvas_megapixels": 200,
        "max_aggregate_megapixels": 200,
        "max_pose_count": 160,
        "max_resident_frames": 5,
        "minimum_valid_scale_pairs": 3,
        "scale_central_fraction": 0.20,
        "scale_low_gradient_quantile": 0.45,
        "scale_minimum_response": 0.10,
        "scale_max_relative_mad": 0.35,
        "residual_alignment": {
            "backend": "se3_epipolar_hierarchical_rgb",
            "analysis_width": 640,
            "maximum_preview_megapixels": 2.0,
            "maximum_evidence_megapixels": 3.0,
            "held_out_fraction": 0.20,
            "owner_track_consistency": True,
            "background_model": "identity",
            "maximum_residual_displacement_pixels": 8.0,
            "maximum_background_roll_degrees": 0.25,
            "maximum_flow_fb_error_pixels": 1.0,
            "maximum_epipolar_error_pixels": 1.0,
            "component_model": "owner_only",
            "maximum_edge_step_p95_pixels": 1.5,
            "maximum_edge_step_pixels": 3.0,
            "cut_mesh_formal_enabled": False,
        },
        "geometry_assisted_seam": {
            "enabled": True,
            "analysis_corridor_width_pixels": 128,
            "trigger_edge_offset_p95_pixels": 1.0,
            "absolute_depth_tolerance_mm": 20.0,
            "relative_depth_tolerance": 0.02,
            "depth_noise_mm": 0.0,
            "mutual_reprojection_tolerance_pixels": 0.40,
            "edge_guard_radius_pixels": 8,
            "minimum_trigger_boundary_observable_pixels": 32,
            "flow_validation_preview_scale": 0.75,
            "minimum_held_out_flow_validation_pixels": 8,
            "minimum_held_out_strong_edge_validation_pixels": 8,
            "maximum_held_out_flow_fb_error_pixels": 0.75,
            "mesh_cell_pixels": 16,
            "minimum_mutual_correspondences": 30,
            "minimum_active_mesh_cells": 4,
            "maximum_local_displacement_pixels": 8.0,
            "maximum_straight_line_deviation_pixels": 1.0,
            "minimum_actual_rgb_line_length_pixels": 24.0,
            "minimum_actual_rgb_line_support_fraction": 0.80,
            "maximum_actual_rgb_line_segments": 32,
            "actual_rgb_line_inverse_maximum_iterations": 8,
            "actual_rgb_line_inverse_maximum_residual_pixels": 0.05,
            "maximum_held_out_error_pixels": 0.75,
            "maximum_held_out_maximum_error_pixels": 2.0,
            "minimum_held_out_improvement_pixels": 0.05,
            "minimum_held_out_improvement_ratio": 0.30,
        },
    }
    assert config["stitch"]["scan_seam"]["backend"] == (
        "rgb_monotonic_hard_owner_graphcut"
    )


def test_formal_delivery_requires_display_only_tsdf_export() -> None:
    stitch = load_config()["stitch"]
    stitch["tsdf_visualization"] = {"enabled": False}

    with pytest.raises(ValueError, match="tsdf_visualization export"):
        _validate_safety_envelope(stitch, diagnostic_force=False)
    assert stitch["scan_seam"]["multiband_levels"] == 3
    assert stitch["scan_seam"]["exposure_mode"] == (
        "safe_wall_global_linear_rgb"
    )
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
    assert legacy_formal_keys.isdisjoint(stitch)


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
        (
            ("calibrated_rgb_pushbroom", "maximum_central_band_fraction"),
            0.21,
            "cannot exceed 0.20",
        ),
        (
            ("calibrated_rgb_pushbroom", "endpoint_outer_half_fov"),
            False,
            "requires endpoint_outer_half_fov",
        ),
        (
            ("calibrated_rgb_pushbroom", "seam_search_width_pixels"),
            31,
            "32-64",
        ),
        (
            ("calibrated_rgb_pushbroom", "seam_search_width_pixels"),
            65,
            "32-64",
        ),
        (
            ("calibrated_rgb_pushbroom", "max_canvas_megapixels"),
            201,
            "200 MP",
        ),
        (
            ("calibrated_rgb_pushbroom", "max_aggregate_megapixels"),
            201,
            "200 MP",
        ),
        (("calibrated_rgb_pushbroom", "max_resident_frames"), 6, "2-5"),
        (
            ("calibrated_rgb_pushbroom", "scale_minimum_response"),
            0.09,
            "cannot be relaxed below 0.10",
        ),
        (
            ("calibrated_rgb_pushbroom", "scale_max_relative_mad"),
            0.36,
            "cannot exceed 0.35",
        ),
        (
            ("calibrated_rgb_pushbroom", "scale_low_gradient_quantile"),
            0.46,
            "cannot exceed 0.45",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "maximum_residual_displacement_pixels",
            ),
            8.1,
            "residual_alignment",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "maximum_preview_megapixels",
            ),
            2.1,
            "residual_alignment",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "maximum_evidence_megapixels",
            ),
            3.1,
            "residual_alignment",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "component_model",
            ),
            "rigid_normal",
            "residual_alignment",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "backend",
            ),
            "homography",
            "residual_alignment",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "held_out_fraction",
            ),
            0.19,
            "20% held-out",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "owner_track_consistency",
            ),
            False,
            "owner track consistency",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "minimum_trigger_boundary_observable_pixels",
            ),
            31,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "flow_validation_preview_scale",
            ),
            0.49,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "maximum_held_out_flow_fb_error_pixels",
            ),
            0.76,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "minimum_held_out_strong_edge_validation_pixels",
            ),
            7,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "mutual_reprojection_tolerance_pixels",
            ),
            0.41,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "enabled",
            ),
            False,
            "geometry_assisted_seam.enabled",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "analysis_corridor_width_pixels",
            ),
            95,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "mesh_cell_pixels",
            ),
            24,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "edge_guard_radius_pixels",
            ),
            7,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "minimum_mutual_correspondences",
            ),
            29,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "minimum_active_mesh_cells",
            ),
            3,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "absolute_depth_tolerance_mm",
            ),
            20.01,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "relative_depth_tolerance",
            ),
            0.0201,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "depth_noise_mm",
            ),
            0.1,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "maximum_local_displacement_pixels",
            ),
            8.1,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "maximum_straight_line_deviation_pixels",
            ),
            1.01,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "maximum_held_out_error_pixels",
            ),
            0.76,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "geometry_assisted_seam",
                "minimum_held_out_improvement_ratio",
            ),
            0.29,
            "geometry_assisted_seam",
        ),
        (
            (
                "calibrated_rgb_pushbroom",
                "residual_alignment",
                "background_model",
            ),
            "se2",
            "sole global geometry",
        ),
        (("rgbd_odometry", "minimum_fitness"), 0.14, "cannot be relaxed"),
        (("pose_quality", "maximum_total_rotation_deg"), 11.0, "cannot be relaxed"),
        (("scan_seam", "multiband_levels"), 4, "1-3"),
        (("scan_seam", "quality_gate"), False, "cannot disable"),
        (
            ("scan_seam", "exposure_mode"),
            "safe_wall_smooth_gain",
            "safe_wall_global_linear_rgb",
        ),
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

    config["calibrated_rgb_pushbroom"]["max_canvas_megapixels"] = 201
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
