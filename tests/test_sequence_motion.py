from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import panorama_demo.stitch_sequence as sequence
from panorama_demo.calibrated_rgb_pushbroom import GeometryAssistedSeamConfig
from panorama_demo.local_apap_flow import LocalAPAPFlowConfig
from panorama_demo.rgbd_odometry import RGBDOdometryConfig
from panorama_demo.session import load_rgbd_session
from panorama_demo.synthetic import generate_sequence


class _MetricTranslationBackend:
    name = "test_metric_rgbd"

    def __init__(self, poses: dict[int, np.ndarray]) -> None:
        self.poses = poses
        self.pairs: list[tuple[int, int]] = []
        self.optimized_node_ids: tuple[int, ...] = ()

    def estimate_pair(self, *, reference, source, intrinsics, config):
        del intrinsics, config
        reference_id = int(reference.frame_id)
        source_id = int(source.frame_id)
        self.pairs.append((reference_id, source_id))
        return {
            "source_to_reference": (
                np.linalg.inv(self.poses[reference_id]) @ self.poses[source_id]
            ),
            "converged": True,
            "fitness": 0.99,
            "rmse_mm": 0.25,
            "information": np.eye(6, dtype=np.float64) * 100.0,
            "backend": self.name,
        }

    def optimize_pose_graph(
        self, *, node_ids, initial_camera_to_world, edges, config
    ):
        del edges, config
        self.optimized_node_ids = tuple(int(value) for value in node_ids)
        return tuple(np.asarray(pose).copy() for pose in initial_camera_to_world)


def _synthetic_session(tmp_path: Path):
    root = generate_sequence(
        tmp_path / "session",
        frame_count=5,
        frame_width=160,
        frame_height=100,
        step=30,
        seed=31,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    poses = {
        int(row["frame_id"]): np.asarray(
            row["matrix_row_major"], dtype=np.float64
        ).reshape(4, 4)
        for row in manifest["known_trajectory"]["poses"]
    }
    return load_rgbd_session(root), poses


def test_formal_sequence_exposes_no_legacy_2d_motion_or_interpolation_api() -> None:
    for name in (
        "regularize_pair_homography",
        "interpolate_translation_transforms",
        "interpolate_motion_guided_transforms",
        "_parse_protected_regions",
    ):
        assert not hasattr(sequence, name)


def test_pose_edge_estimation_connects_only_real_rgbd_pose_nodes(
    tmp_path: Path,
) -> None:
    session, poses = _synthetic_session(tmp_path)
    selected = [session.frames[index] for index in (0, 2, 4)]
    backend = _MetricTranslationBackend(poses)

    edges, optional_failures = sequence._estimate_pose_edges(
        selected,
        session.calibration,
        RGBDOdometryConfig(working_width=160),
        backend=backend,
        nonadjacent_gap=2,
    )

    assert optional_failures == []
    assert backend.pairs == [(0, 2), (2, 4), (0, 4)]
    assert [
        (edge.reference_node_id, edge.source_node_id) for edge in edges
    ] == backend.pairs
    assert all(edge.source_to_reference.shape == (4, 4) for edge in edges)
    np.testing.assert_allclose(
        edges[0].source_to_reference,
        np.linalg.inv(poses[0]) @ poses[2],
    )


def test_formal_pushbroom_receives_exact_optimized_se3_without_depth_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, poses = _synthetic_session(tmp_path)
    backend = _MetricTranslationBackend(poses)
    received: dict[str, object] = {}

    def legacy_projection_must_not_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("formal RGB pushbroom reached legacy depth projection")

    def fake_pushbroom(frames, optimized_poses, calibration, **kwargs):
        received["frame_ids"] = [frame.frame_id for frame in frames]
        received["poses"] = [np.asarray(pose).copy() for pose in optimized_poses]
        received["calibration"] = calibration
        received["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            panorama=np.full((100, 300, 3), 127, dtype=np.uint8),
            metadata={
                "backend": "calibrated_rgb_pushbroom",
                "pixel_source": "calibrated_rgb_source_samples",
                "depth_used_for_output_pixels": False,
                "depth_used_for_local_geometry": False,
                "point_cloud_constructed": False,
                "tsdf_constructed": False,
                "reference_plane_fitted": False,
                "layout": {},
                "rgb_motion_scale": {},
                "residual_alignment": {
                    "backend": "se3_epipolar_hierarchical_rgb",
                    "selected_model": "identity",
                    "preview_remap_count": len(frames),
                    "full_resolution_output_remap_count": len(frames),
                    "per_source_parameters": [
                        {
                            "source_index": index,
                            "translation_x_pixels": 0.0,
                            "translation_y_pixels": 0.0,
                            "roll_degrees": 0.0,
                            "centre_x_pixels": 0.0,
                            "centre_y_pixels": 0.0,
                            "identity": True,
                        }
                        for index, _frame in enumerate(frames)
                    ],
                    "held_out_metrics_before": {},
                    "held_out_metrics_after": {},
                    "component_audit": {},
                    "topology_audit": {"accepted": True},
                    "working_set_audit": {},
                },
                "geometry_assisted_seam": {
                    "enabled": True,
                    "config": GeometryAssistedSeamConfig().as_dict(),
                    "scope": "adjacent_seam_corridors_only",
                    "depth_used_for_output_pixels": False,
                    "depth_used_for_local_geometry": False,
                    "triggered_pair_count": 0,
                    "accepted_pair_count": 0,
                    "hard_owner_fallback_pair_count": 0,
                    "pairs": [
                        {
                            "pair_index": index,
                            "frame_ids": [
                                int(frames[index].frame_id),
                                int(frames[index + 1].frame_id),
                            ],
                            "triggered": False,
                            "corridor_x": None,
                            "warp_source_index": None,
                            "accepted": False,
                            "fallback": "not_needed",
                            "audit": {"reason": "mock"},
                        }
                        for index in range(len(frames) - 1)
                    ],
                },
                "pairs": [
                    {
                        "first_frame_id": int(frames[index].frame_id),
                        "second_frame_id": int(frames[index + 1].frame_id),
                        "graphcut_used": True,
                        "hard_cut_row_count": 0,
                        "foreground_anchor_handoff_continuity": {
                            "policy": (
                                "foreground_owner_only_continuity_audit_no_local_deformation"
                            ),
                            "handoff_count": 0,
                            "continuity_audit_count": 0,
                            "coverage_complete_count": 0,
                            "owner_only_no_deformation_count": 0,
                            "local_deformation_attempted": False,
                            "audits": [],
                        },
                        "geometry_assistance": {
                            "triggered": False,
                            "accepted": False,
                            "fallback": "not_needed",
                            "audit": {"reason": "mock"},
                        },
                    }
                    for index in range(len(frames) - 1)
                ],
                "quality_metrics": {"quality_pass": True},
            },
        )

    monkeypatch.setattr(sequence, "_full_projection_frames", legacy_projection_must_not_run)
    monkeypatch.setattr(sequence, "render_calibrated_rgb_pushbroom", fake_pushbroom)
    args = sequence._parser().parse_args(
        [str(session.root), "--output", str(tmp_path / "output")]
    )

    report = sequence.run(args, odometry_backend=backend)

    assert received["frame_ids"] == list(backend.optimized_node_ids)
    assert received["calibration"] == session.calibration
    for frame_id, optimized in zip(
        received["frame_ids"], received["poses"], strict=True
    ):
        np.testing.assert_allclose(optimized, poses[frame_id])
    kwargs = received["kwargs"]
    assert kwargs["quality_gate"] is False
    assert kwargs["multiband_levels"] == 3
    assert len(kwargs["rgb_motions"]) == len(backend.optimized_node_ids) - 1
    assert report["render_strategy"] == "calibrated_rgb_pushbroom"
    assert report["render"]["depth_used_for_output_pixels"] is False
    render_transforms = json.loads(
        (tmp_path / "output" / "render_transforms.json").read_text(encoding="utf-8")
    )
    alignment = render_transforms["residual_alignment"]
    assert alignment["selected_model"] == "identity"
    assert alignment["per_source_parameters"][1]["translation_x_pixels"] == 0.0
    delivery = json.loads(
        (tmp_path / "output" / "delivery.json").read_text(encoding="utf-8")
    )
    assert delivery["alignment_model"] == "identity"


def test_geometry_compaction_rejects_dense_nested_audits_and_invalid_accepted_meshes() -> None:
    settings = GeometryAssistedSeamConfig()
    accepted_trigger_audit = {
        "trigger_reasons": ["boundary_high_rgb_risk"],
        "preview_risk_policy": (
            "rgb_risk_at_nominal_owner_boundary_plus_or_minus_"
            "2_full_resolution_pixels"
        ),
        "boundary_high_risk_topology_policy": (
            "3x3_closed_8_connected_raw_structural_rgb_risk_seed_component_"
            "covering_nominal_centreline_minimum_72_full_resolution_pixels_"
            "18_rows_and_26_row_span"
        ),
        "preview_risk_preview_scale": 0.75,
        "boundary_risk_component_count": 1,
        "boundary_centreline_touching_risk_component_count": 1,
        "boundary_qualifying_risk_component_count": 1,
        "boundary_high_rgb_risk": True,
        "minimum_boundary_high_risk_component_pixel_count": 41,
        "minimum_boundary_high_risk_component_row_count": 14,
        "minimum_boundary_high_risk_component_row_span": 20,
        "boundary_largest_qualifying_risk_component_pixel_count": 60,
        "boundary_largest_qualifying_risk_component_row_count": 20,
        "boundary_largest_qualifying_risk_component_row_span": 20,
        "preview_hard_cut_row_count": 0,
        "preview_full_height_hard_cut": False,
        "preview_owner_probe_status": "graphcut",
        "preview_owner_probe_graphcut_used": True,
        "common_row_count": 64,
    }

    def render_metadata(audit: dict[str, object], *, accepted: bool) -> dict[str, object]:
        if accepted:
            audit = {**accepted_trigger_audit, **audit}
        return {
            "geometry_assisted_seam": {
                "config": settings.as_dict(),
                "scope": "adjacent_seam_corridors_only",
                "depth_used_for_output_pixels": False,
                "depth_used_for_local_geometry": accepted,
                "triggered_pair_count": int(accepted),
                "accepted_pair_count": int(accepted),
                "hard_owner_fallback_pair_count": 0,
                "pairs": [
                    {
                        "pair_index": 0,
                        "frame_ids": [10, 11],
                        "triggered": accepted,
                        "corridor_x": [20, 148],
                        "warp_source_index": 1 if accepted else None,
                        "accepted": accepted,
                        "fallback": "none" if accepted else "not_needed",
                        "audit": audit,
                    }
                ],
            }
        }

    with pytest.raises(RuntimeError, match="dense data"):
        sequence._compact_geometry_assistance_for_transforms(
            render_metadata(
                {"reason": "mock", "nested": {"depth_mm": [1000.0]}},
                accepted=False,
            ),
            settings,
            [10, 11],
        )

    with pytest.raises(RuntimeError, match="straight-line gate"):
        sequence._compact_geometry_assistance_for_transforms(
            render_metadata(
                {
                    "reason": "accepted",
                    "mesh": {
                        "metric_unit": "full_resolution_pixels",
                        "accepted": True,
                        "maximum_straight_line_deviation_pixels": 1.01,
                        "active_cell_count": 4,
                        "largest_connected_active_cell_count": 4,
                        "straight_line_audit_policy": (
                            "raw_same_layer_active_centrelines_internal_grid_edges_and_cell_diagonals"
                        ),
                    },
                },
                accepted=True,
            ),
            settings,
            [10, 11],
        )

    compact = sequence._compact_geometry_assistance_for_transforms(
        render_metadata(
            {
                "reason": "accepted",
                "mesh": {
                    "metric_unit": "full_resolution_pixels",
                    "accepted": True,
                    "maximum_straight_line_deviation_pixels": 1.0,
                    "active_cell_count": 4,
                    "largest_connected_active_cell_count": 4,
                    "straight_line_audit_policy": (
                        "raw_same_layer_active_centrelines_internal_grid_edges_and_cell_diagonals"
                    ),
                    "correspondence_count": 30,
                    "training_count": 24,
                    "held_out_count": 6,
                    "free_node_count": 1,
                    "held_out_error_p95_before_pixels": 1.20,
                    "held_out_error_p95_after_pixels": 0.60,
                    "held_out_error_max_after_pixels": 1.50,
                    "maximum_displacement_pixels": 2.0,
                    "minimum_jacobian_determinant": 0.80,
                    "maximum_jacobian_determinant": 1.20,
                    "maximum_jacobian_condition": 1.50,
                    "boundary_identity_maximum_error_pixels": 0.0,
                },
                "depth_protected_pixel_count": 12,
                "rgb_transparent_or_reflection_protected_pixel_count": 10,
                "protected_pixel_count": 18,
                "protected_active_overlap_pixel_count": 0,
                "active_non_same_layer_overlap_pixel_count": 0,
                "geometry": {
                    "first_to_second": {
                        "mutual_consistent_pixel_count": 160,
                        "mutual_depth_residual_ratio_p95": 0.45,
                        "mutual_depth_residual_ratio_max": 0.90,
                    },
                    "second_to_first": {
                        "mutual_consistent_pixel_count": 158,
                        "mutual_depth_residual_ratio_p95": 0.40,
                        "mutual_depth_residual_ratio_max": 0.85,
                    },
                    "first_surface_safety": {
                        "policy": (
                            "one_dominant_far_depth_component_mesh_safe_"
                            "near_and_ambiguous_components_hard_owner"
                        ),
                        "mesh_safe_pixel_count": 180,
                        "dominant_component_pixel_count": 180,
                        "hard_owner_pixel_count": 24,
                        "near_foreground_pixel_count": 20,
                        "ambiguous_or_unreliable_pixel_count": 4,
                        "component_count": 2,
                        "material_component_count": 2,
                        "near_foreground_component_count": 1,
                        "depth_anchor_component_count": 2,
                        "dominant_component_fraction": 0.80,
                        "dominant_component_median_depth_mm": 1000.0,
                        "analysis_scope": "adjacent_seam_raw_footprint",
                        "analysis_pixel_count": 224,
                        "base_safe_pixel_count": 204,
                    },
                    "second_surface_safety": {
                        "policy": (
                            "one_dominant_far_depth_component_mesh_safe_"
                            "near_and_ambiguous_components_hard_owner"
                        ),
                        "mesh_safe_pixel_count": 176,
                        "dominant_component_pixel_count": 176,
                        "hard_owner_pixel_count": 28,
                        "near_foreground_pixel_count": 24,
                        "ambiguous_or_unreliable_pixel_count": 4,
                        "component_count": 2,
                        "material_component_count": 2,
                        "near_foreground_component_count": 1,
                        "depth_anchor_component_count": 2,
                        "dominant_component_fraction": 0.78,
                        "dominant_component_median_depth_mm": 1000.0,
                        "analysis_scope": "adjacent_seam_raw_footprint",
                        "analysis_pixel_count": 224,
                        "base_safe_pixel_count": 204,
                    },
                },
                "virtual_background_component": {
                    "policy": (
                        "one_4_connected_bilateral_background_component_"
                        "crossing_nominal_owner_boundary"
                    ),
                    "nominal_boundary_x": 64,
                    "depth_safe_component_count": 1,
                    "boundary_crossing_component_count": 1,
                    "selected_component_label": 1,
                    "selected_component_pixel_count": 128,
                    "nonselected_depth_safe_pixel_count": 0,
                },
                "depth_same_layer_before_rgb_protection_pixel_count": 140,
                "candidate_depth_same_layer_pixel_count": 128,
                "depth_same_layer_pixel_count": 128,
                "rgb_flow_application_pixel_count": 160,
                "rgb_flow_fit_excluded_same_layer_pixel_count": 32,
                "same_layer_pixel_count": 96,
                "mesh_fit_support_pixel_count": 64,
                "mesh_candidate_pixel_count": 64,
                "mesh_active_pixel_count": 64,
                "rgb_transparency_protection": {
                    "policy": "rgb_occlusion_or_any_strong_rgb_structure_dilated_guard",
                    "guard_radius_pixels": 8,
                    "preview_occluded_pixel_count": 2,
                    "preview_strong_rgb_structure_pixel_count": 4,
                    "preview_uncertain_or_rejected_strong_edge_pixel_count": 3,
                    "preview_unsafe_pixel_count": 5,
                    "full_resolution_unguarded_pixel_count": 10,
                    "full_resolution_protected_pixel_count": 42,
                    "full_resolution_tile_pixel_count": 256,
                },
                "final_owner": {
                    "policy": (
                        "final_nominal_owner_boundary_rgb_risk_and_hard_cut_"
                        "closure"
                    ),
                    "nominal_boundary_x": 64,
                    "nominal_boundary_half_width_pixels": 2,
                    "final_nominal_boundary_risk_pixel_count": 0,
                    "final_nominal_boundary_risk_row_count": 0,
                    "final_boundary_common_row_count": 64,
                    "final_common_row_count": 64,
                    "final_hard_cut_row_count": 0,
                    "final_full_height_hard_cut": False,
                    "final_graphcut_used": True,
                    "final_pair_level_hard_owner": False,
                },
                "rgb_flow_fit_support": {
                    "policy": (
                        "training_only_accepted_bidirectional_rgb_flow_and_epipolar_support"
                    ),
                    "preview_scale": 0.50,
                    "metric_unit": "full_resolution_pixels",
                    "maximum_full_resolution_fb_error_pixels": 0.75,
                    "maximum_preview_fb_error_pixels": 0.375,
                    "preview_training_flow_consistent_pixel_count": 16,
                    "preview_application_flow_consistent_pixel_count": 20,
                    "preview_held_out_pixel_count": 8,
                    "full_resolution_flow_consistent_pixel_count": 128,
                    "full_resolution_application_flow_consistent_pixel_count": 160,
                    "full_resolution_tile_pixel_count": 256,
                },
                "rgb_flow_validation": {
                    "accepted": True,
                    "reason": "accepted",
                    "preview_scale": 0.75,
                    "metric_unit": "full_resolution_pixels",
                    "held_out_observable_flow_pixel_count": 8,
                    "held_out_flow_fb_error_p95_pixels": 0.50,
                    "maximum_held_out_flow_fb_error_pixels": 0.75,
                    "held_out_strong_edge": {
                        "accepted": True,
                        "reason": "accepted",
                        "policy": (
                            "same_held_out_flow_epipolar_supported_strong_rgb_edges"
                        ),
                        "metric_unit": "full_resolution_pixels",
                        "preview_scale": 0.75,
                        "held_out_strong_edge_pixel_count": 8,
                        "held_out_strong_edge_p95_before_pixels": 1.10,
                        "held_out_strong_edge_p95_after_pixels": 0.50,
                        "held_out_strong_edge_maximum_after_pixels": 1.50,
                        "held_out_strong_edge_improvement_pixels": 0.60,
                        "held_out_strong_edge_improvement_ratio": (
                            0.60 / 1.10
                        ),
                    },
                    "rgb_actual_line_straightness": {
                        "accepted": True,
                        "observed": True,
                        "reason": "accepted",
                        "metric_unit": "full_resolution_pixels",
                        "policy": (
                            "baseline_second_rgb_dual_source_canny_hough_"
                            "raw_forward_inverse_chord_bend"
                        ),
                        "minimum_actual_rgb_line_length_pixels": 24.0,
                        "minimum_actual_rgb_line_support_fraction": 0.80,
                        "maximum_actual_rgb_line_segments": 32,
                        "maximum_straight_line_deviation_pixels": 1.0,
                        "inverse_maximum_iterations": 8,
                        "inverse_maximum_residual_pixels": 0.05,
                        "raw_hough_segment_count": 2,
                        "deduplicated_hough_segment_count": 1,
                        "eligible_hough_segment_count": 1,
                        "tested_line_run_count": 1,
                        "maximum_line_bend_pixels": 0.75,
                        "p95_line_bend_pixels": 0.60,
                        "maximum_inverse_residual_pixels": 0.03,
                        "p95_inverse_residual_pixels": 0.02,
                    },
                },
            },
            accepted=True,
        ),
        settings,
        [10, 11],
    )
    assert compact["pairs"][0]["audit"]["mesh"]["active_cell_count"] == 4

    def rejected_metadata() -> dict[str, object]:
        return render_metadata(
            json.loads(json.dumps(compact["pairs"][0]["audit"])), accepted=True
        )

    # A local APAP candidate inherits the same depth-layer, RGB protection and
    # final owner-closure audits even when the older RGB-D mesh was rejected.
    # It has its own scalar fit/held-out evidence and must not be accepted by a
    # weaker sidecar branch.
    local_settings = LocalAPAPFlowConfig(enabled=True)
    accepted_apap = rejected_metadata()
    accepted_apap_geometry = accepted_apap["geometry_assisted_seam"]
    accepted_apap_geometry["local_apap_flow"] = local_settings.as_dict()
    accepted_apap_geometry["local_apap_flow_attempted_pair_count"] = 1
    accepted_apap_geometry["local_apap_flow_accepted_pair_count"] = 1
    accepted_apap_audit = accepted_apap_geometry["pairs"][0]["audit"]
    accepted_apap_audit["mesh"]["accepted"] = False
    accepted_apap_audit["mesh_candidate_pixel_count"] = 0
    accepted_apap_audit["mesh_active_pixel_count"] = 0
    accepted_apap_audit["local_deformation_active_pixel_count"] = 64
    accepted_apap_audit["foreground_instance_active_overlap_pixel_count"] = 0
    accepted_apap_audit["local_deformation"] = {
        "enabled": True,
        "attempted": True,
        "accepted": True,
        "method": "apap",
        "dense_evidence_storage": "temporary_only",
        "application_policy": "same_layer_visible_nonprotected_instance_or_background_only",
        "boundary_policy": "outer_corridor_border_identity",
        "correspondence_policy": "bidirectional_rgbd_mutual_virtual_coordinates",
        "analysis_rgb_remap_count": 2,
        "active_pixel_count": 64,
        "correspondence_count": 40,
        "apap_inliers": 40,
        "apap_inlier_ratio": 1.0,
        "active_mesh_cell_count": 4,
        "max_displacement_px": 2.0,
        "jacobian_min": 0.8,
        "local_scale_min": 0.9,
        "local_scale_max": 1.1,
        "held_out_pixel_count": 30,
        "held_out_error_before_p95": 1.0,
        "held_out_error_after_p95": 0.5,
        "held_out_improvement_ratio": 0.5,
    }
    compact_apap = sequence._compact_geometry_assistance_for_transforms(
        accepted_apap, settings, [10, 11], local_settings
    )
    assert compact_apap["local_apap_flow_accepted_pair_count"] == 1
    assert compact_apap["pairs"][0]["audit"]["local_deformation"]["method"] == "apap"

    bad_apap_owner_closure = json.loads(json.dumps(accepted_apap))
    bad_apap_owner_closure["geometry_assisted_seam"]["pairs"][0]["audit"][
        "final_owner"
    ]["final_full_height_hard_cut"] = True
    with pytest.raises(RuntimeError, match="final RGB owner-closure audit"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_apap_owner_closure, settings, [10, 11], local_settings
        )

    # A safe wall is not required to retain a solver-valid Hough line after
    # foreground/transparent structures have been hard-owned.  The intrinsic
    # mesh centreline/edge/diagonal audit remains mandatory, while the absent
    # observed line is preserved as a scalar non-veto result.  An eligible
    # baseline segment can still lack a solver-valid active run.
    unobserved_line = rejected_metadata()
    unobserved_audit = unobserved_line["geometry_assisted_seam"]["pairs"][0][
        "audit"
    ]["rgb_flow_validation"]["rgb_actual_line_straightness"]
    unobserved_audit.update(
        {
            "observed": False,
            "reason": "not_observed_no_solver_valid_line",
            "raw_hough_segment_count": 1,
            "deduplicated_hough_segment_count": 1,
            "eligible_hough_segment_count": 1,
            "tested_line_run_count": 0,
            "maximum_line_bend_pixels": None,
            "p95_line_bend_pixels": None,
            "maximum_inverse_residual_pixels": None,
            "p95_inverse_residual_pixels": None,
        }
    )
    compact_unobserved = sequence._compact_geometry_assistance_for_transforms(
        unobserved_line, settings, [10, 11]
    )
    assert compact_unobserved["pairs"][0]["audit"]["rgb_flow_validation"][
        "rgb_actual_line_straightness"
    ]["observed"] is False

    malformed_unobserved = rejected_metadata()
    malformed_line = malformed_unobserved["geometry_assisted_seam"]["pairs"][0][
        "audit"
    ]["rgb_flow_validation"]["rgb_actual_line_straightness"]
    malformed_line.update(
        {
            "observed": False,
            "reason": "not_observed_no_solver_valid_line",
            "tested_line_run_count": 0,
            "maximum_line_bend_pixels": 0.10,
            "p95_line_bend_pixels": None,
            "maximum_inverse_residual_pixels": None,
            "p95_inverse_residual_pixels": None,
        }
    )
    with pytest.raises(RuntimeError, match="invalid unobserved RGB line audit"):
        sequence._compact_geometry_assistance_for_transforms(
            malformed_unobserved, settings, [10, 11]
        )

    bad_fit_support = rejected_metadata()
    bad_fit_support["geometry_assisted_seam"]["pairs"][0]["audit"][
        "mesh_fit_support_pixel_count"
    ] = 97
    with pytest.raises(RuntimeError, match="invalid virtual background support"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_fit_support, settings, [10, 11]
        )

    bad_application_count = rejected_metadata()
    bad_application_count["geometry_assisted_seam"]["pairs"][0]["audit"][
        "rgb_flow_fit_support"
    ]["full_resolution_application_flow_consistent_pixel_count"] = 159
    with pytest.raises(RuntimeError, match="inconsistent independent flow-fit support"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_application_count, settings, [10, 11]
        )

    bad_partition = rejected_metadata()
    bad_partition["geometry_assisted_seam"]["pairs"][0]["audit"]["mesh"][
        "training_count"
    ] = 25
    with pytest.raises(RuntimeError, match="held-out partition"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_partition, settings, [10, 11]
        )

    bad_strong_edge = rejected_metadata()
    bad_strong_edge["geometry_assisted_seam"]["pairs"][0]["audit"][
        "rgb_flow_validation"
    ]["held_out_strong_edge"]["held_out_strong_edge_improvement_ratio"] = 0.5
    with pytest.raises(RuntimeError, match="strong-edge gate"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_strong_edge, settings, [10, 11]
        )

    bad_actual_line = rejected_metadata()
    bad_actual_line["geometry_assisted_seam"]["pairs"][0]["audit"][
        "rgb_flow_validation"
    ]["rgb_actual_line_straightness"]["maximum_line_bend_pixels"] = 1.01
    with pytest.raises(RuntimeError, match="actual RGB line-straightness gate"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_actual_line, settings, [10, 11]
        )

    bad_final_owner = rejected_metadata()
    bad_final_owner["geometry_assisted_seam"]["pairs"][0]["audit"][
        "final_owner"
    ]["final_full_height_hard_cut"] = True
    with pytest.raises(RuntimeError, match="final RGB owner-closure audit"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_final_owner, settings, [10, 11]
        )

    bad_trigger_topology = rejected_metadata()
    bad_trigger_topology["geometry_assisted_seam"]["pairs"][0]["audit"][
        "boundary_qualifying_risk_component_count"
    ] = 0
    with pytest.raises(RuntimeError, match="structural-risk topology"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_trigger_topology, settings, [10, 11]
        )

    bad_flow = rejected_metadata()
    bad_flow["geometry_assisted_seam"]["pairs"][0]["audit"][
        "rgb_flow_validation"
    ]["held_out_flow_fb_error_p95_pixels"] = -0.1
    with pytest.raises(RuntimeError, match="RGB-flow gate"):
        sequence._compact_geometry_assistance_for_transforms(bad_flow, settings, [10, 11])

    bad_depth_residual = rejected_metadata()
    bad_depth_residual["geometry_assisted_seam"]["pairs"][0]["audit"][
        "geometry"
    ]["first_to_second"]["mutual_depth_residual_ratio_max"] = 1.01
    with pytest.raises(RuntimeError, match="depth-residual gate"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_depth_residual, settings, [10, 11]
        )

    bad_protected_overlap = rejected_metadata()
    bad_protected_overlap["geometry_assisted_seam"]["pairs"][0]["audit"][
        "protected_active_overlap_pixel_count"
    ] = 1
    with pytest.raises(RuntimeError, match="owner-protected component"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_protected_overlap, settings, [10, 11]
        )

    bad_aggregate = rejected_metadata()
    bad_aggregate["geometry_assisted_seam"]["accepted_pair_count"] = 0
    with pytest.raises(RuntimeError, match="accepted aggregate"):
        sequence._compact_geometry_assistance_for_transforms(
            bad_aggregate, settings, [10, 11]
        )

    bad_list = rejected_metadata()
    bad_list["geometry_assisted_seam"]["pairs"][0]["audit"]["trigger_reasons"] = [
        ["not-a-scalar-list"]
    ]
    with pytest.raises(RuntimeError, match="dense list"):
        sequence._compact_geometry_assistance_for_transforms(bad_list, settings, [10, 11])


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"pose_backend": "unistitch"}, "pose_backend"),
        ({"sequence_blend_mode": "feather"}, "calibrated_rgb_pushbroom"),
        (
            {"dense_fusion_backend": "tsdf_plane_dense_rgbd"},
            "TSDF and RGB-D projection",
        ),
        (
            {"rgbd_projection": {"mode": "orthographic_side_scan"}},
            "TSDF and RGB-D projection",
        ),
        (
            {"calibrated_rgb_pushbroom": {"mode": "central_strip"}},
            "Formal renderer mode",
        ),
        (
            {"scan_seam": {"backend": "dp"}},
            "rgb_monotonic_hard_owner_graphcut",
        ),
    ],
)
def test_formal_backend_configuration_rejects_non_pushbroom_paths(
    override: dict[str, object], message: str
) -> None:
    config: dict[str, object] = {
        "pose_backend": "open3d_rgbd",
        "sequence_blend_mode": "calibrated_rgb_pushbroom",
        "calibrated_rgb_pushbroom": {
            "mode": "calibrated_rgb_pushbroom",
        },
        "scan_seam": {"backend": "rgb_monotonic_hard_owner_graphcut"},
    }
    config.update(override)

    with pytest.raises(ValueError, match=message):
        sequence._validate_backend_config(config)
