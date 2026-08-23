from __future__ import annotations

from copy import deepcopy

import pytest

from panorama_demo.video_s12_config import load_s12_config


def _config() -> dict[str, object]:
    return {
        "schema": "gemini305-video-s12-standalone/v1",
        "algorithm_id": "S012_standalone_auto_anchor_dense_central_slit_v1",
        "diagnostic_only": True,
        "production_eligible": False,
        "trajectory": {
            "require_direct_orb_pose": True,
            "allow_interpolated_pose": False,
            "allow_extrapolated_pose": False,
            "require_uniform_pose_origin": True,
            "maximum_timestamp_delta_ms": 5.0,
            "audit_rotation": True,
            "maximum_relative_roll_deg": 1.0,
            "maximum_relative_pitch_deg": 1.0,
            "maximum_relative_yaw_deg": 2.0,
            "maximum_source_pruning_rounds": 2,
        },
        "scan": {
            "minimum_pose_count": 3,
            "minimum_scan_displacement_m": 0.1,
            "rail_axis_ransac_iterations": 32,
            "rail_axis_inlier_threshold_m": 0.02,
            "maximum_cross_track_error_m": 0.03,
            "maximum_backward_step_m": 0.002,
            "maximum_pause_span_frames": 30,
            "require_image_motion_direction_agreement": True,
        },
        "motion_graph": {
            "preserve_reliable_measurements": True,
            "apply_median_filter": False,
            "interpolate_unreliable_edges": False,
            "extrapolate_edges": False,
            "minimum_model_inlier_count": 16,
            "minimum_fb_retention_ratio": 0.45,
            "minimum_model_inlier_ratio_retained": 0.45,
            "minimum_effective_inlier_ratio_detected": 0.25,
            "reliability_gate_mode": "split_v1",
            "minimum_inlier_count": 16,
            "minimum_inlier_ratio": 0.45,
            "local_edge": {
                "maximum_frame_gap": 64,
                "lk_window_size": 21,
                "lk_max_level": 3,
            },
            "anchor_long": {
                "use_local_frame_gap": False,
                "maximum_frame_gap": 256,
                "minimum_spacing_px": 64,
                "maximum_spacing_px": 144,
                "require_timestamp_audit": True,
                "minimum_timestamp_gap_ms": 0.001,
                "maximum_timestamp_gap_ms": 10000.0,
                "require_predicted_overlap": True,
                "minimum_predicted_overlap_fraction": 0.5,
                "require_direct_image_match": True,
                "use_provisional_initial_flow": False,
                "initial_flow_source": "endpoint_phase_correlation",
                "feature_domain": "fixed_calibrated_cx_slit",
                "feature_slit_width_px": 48,
                "preprocessing": "sobel_gradient_magnitude",
                "phase_correlation_window": "none",
                "lk_window_size": 91,
                "lk_max_level": 4,
            },
        },
        "anchors": {"maximum_direct_local_path_difference_px": 2.0},
        "horizontal_solver": {},
        "source_selection": {"permit_virtual_rgb_source": False},
        "schedule": {"permit_dynamic_pair_boundary": False, "permit_object_owner": False},
        "open3d_audit": {
            "allow_rgb_generation": False,
            "allow_pose_replacement": False,
            "allow_tsdf": False,
        },
        "render": {
            "transition_width_px": 0,
            "color_gain_enabled": False,
            "rotation_enabled": False,
            "synthetic_hole_fill": False,
            "foreign_source_fill": False,
        },
        "vertical": {
            "enabled_for_stage_a": False,
            "allow_canvas_interpolated_offset": False,
            "allow_pair_local_warp": False,
            "allow_row_dependent_warp": False,
        },
        "tracks": {},
        "acceptance": {},
        "output": {},
    }


def test_s12_config_loads_only_standalone_diagnostic_contract() -> None:
    config = load_s12_config(_config())
    assert config.scan.minimum_pose_count == 3
    assert config.trajectory.maximum_timestamp_delta_ms == 5.0
    assert "baseline" not in config.raw


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("trajectory", "allow_interpolated_pose", True),
        ("motion_graph", "apply_median_filter", True),
        ("source_selection", "permit_virtual_rgb_source", True),
        ("render", "transition_width_px", 1),
        ("vertical", "enabled_for_stage_a", True),
    ],
)
def test_s12_config_rejects_forbidden_stage_a_behavior(
    section: str, key: str, value: object
) -> None:
    document = deepcopy(_config())
    document[section][key] = value  # type: ignore[index]
    with pytest.raises(ValueError, match=key):
        load_s12_config(document)


def test_s12_config_rejects_production_eligibility() -> None:
    document = _config()
    document["production_eligible"] = True
    with pytest.raises(ValueError, match="diagnostic-only"):
        load_s12_config(document)
