from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import panorama_demo.calibrated_rgb_pushbroom as pushbroom_module
from panorama_demo.calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomRenderer,
    CalibratedRGBPushbroomConfig,
    GeometryAssistedSeamConfig,
    _boundary_rgb_risk_audit,
    _geometry_trigger_from_preview,
    _held_out_strong_rgb_edge_gate,
    _rgb_transparent_or_reflection_protection_from_preview,
    _lift_rgb_flow_application_and_fit_support_to_geometry_tile,
    _lift_training_rgb_flow_consistency_to_geometry_tile,
    _force_monotonic_component_owners,
    _pair_level_hard_owner_options,
    _pair_level_hard_owner_masks,
    _preflight_failure_pair_index,
    _raw_pixels_to_virtual_coordinates,
    _resolve_pair_level_hard_owners,
    _audit_suppressed_source_frames,
    _select_pair_level_hard_owner,
    _PhotometricEdge,
    _adaptive_multiband_levels,
    _apply_linear_bgr_gain,
    _blend_safe_pair_zone,
    _graphcut_monotonic_owner,
    _solve_global_linear_rgb_gains,
    _srgb_to_linear_bgr,
    build_calibrated_rgb_pushbroom_layout,
    estimate_rgb_motion_pixels_per_mm,
    render_calibrated_rgb_pushbroom,
)
from panorama_demo.rgb_residual_alignment import ProtectedComponentFragment
from panorama_demo.geometry_assisted_local_warp import LocalMeshInverseWarp, TileBounds
from panorama_demo.session import RGBDSession, load_rgbd_session
from panorama_demo.synthetic import generate_sequence


def _make_rgb_pushbroom_input(tmp_path: Path, *, seed: int) -> tuple[RGBDSession, list[np.ndarray]]:
    root = generate_sequence(
        tmp_path / "session",
        frame_count=5,
        frame_width=320,
        frame_height=200,
        # Dense real pose nodes leave a 32--64 px calibrated seam-search
        # corridor inside the unchanged 20% central source-band limit.
        step=16,
        seed=seed,
    )
    # The general renderer fixture isolates calibrated-strip, photometric and
    # ownership contracts from the generator's deliberately dense foreground
    # clutter.  Dedicated tests below cover protected foreground components.
    for index in range(5):
        image = np.full(
            (200, 320, 3),
            (180 + 2 * index, 182 + index, 184),
            dtype=np.uint8,
        )
        assert cv2.imwrite(str(root / "color" / f"{index:08d}.jpg"), image)
    session = load_rgbd_session(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    trajectory = manifest["known_trajectory"]
    assert trajectory["transform"] == "camera_to_world"
    return (
        session,
        [
            np.asarray(row["matrix_row_major"], dtype=np.float64).reshape(4, 4)
            for row in trajectory["poses"]
        ],
    )


def _reliable_adjacent_rgb_motions(frame_count: int) -> list[dict[str, object]]:
    """Synthetic frames advance by 16 RGB pixels at the far background."""

    return [{"dx": 16.0, "reliable": True} for _ in range(frame_count - 1)]


def test_geometry_trigger_requires_local_owner_boundary_evidence() -> None:
    evidence = SimpleNamespace(
        metrics={
            "edge_normal_step_p95_pixels": 4.0,
            "edge_normal_step_max_pixels": 8.0,
            "flow_fb_error_p95_pixels": 3.0,
        }
    )
    settings = GeometryAssistedSeamConfig()
    boundary_contract = {
        "preview_risk_policy": (
            "rgb_risk_at_nominal_owner_boundary_plus_or_minus_"
            "2_full_resolution_pixels"
        ),
        "boundary_high_risk_topology_policy": (
            "3x3_closed_8_connected_raw_structural_rgb_risk_seed_component_covering_"
            "nominal_centreline_minimum_72_full_resolution_pixels_18_rows_"
            "and_26_row_span"
        ),
        "preview_risk_preview_scale": 0.75,
        "boundary_risk_pixel_count": 0,
        "boundary_risk_row_count": 0,
        "boundary_common_row_count": 40,
        "common_row_count": 40,
        "boundary_risk_component_count": 0,
        "boundary_centreline_touching_risk_component_count": 0,
        "boundary_largest_risk_component_pixel_count": 0,
        "boundary_largest_risk_component_row_count": 0,
        "boundary_largest_risk_component_row_span": 0,
        "minimum_boundary_high_risk_component_pixel_count": 41,
        "minimum_boundary_high_risk_component_row_count": 14,
        "minimum_boundary_high_risk_component_row_span": 20,
        "boundary_qualifying_risk_component_count": 0,
        "boundary_largest_qualifying_risk_component_pixel_count": 0,
        "boundary_largest_qualifying_risk_component_row_count": 0,
        "boundary_largest_qualifying_risk_component_row_span": 0,
        "boundary_high_rgb_risk": False,
        "preview_hard_cut_row_count": 0,
        "preview_full_height_hard_cut": False,
        "preview_owner_probe_status": "not_needed_no_boundary_rgb_risk",
        "preview_owner_probe_graphcut_used": False,
    }

    triggered, audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 31,
            "edge_normal_step_p95_pixels": 4.0,
            "edge_normal_step_max_pixels": 8.0,
        },
    )

    assert not triggered
    assert audit["boundary_support_sufficient"] is False
    triggered, audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 32,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 2.1,
        },
    )
    assert triggered
    assert audit["trigger_reasons"] == ["boundary_edge_normal_step_max"]

    isolated_risk_triggered, isolated_risk_audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 0,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 0.0,
            "boundary_risk_pixel_count": 3,
            "boundary_risk_row_count": 2,
            "boundary_risk_component_count": 1,
            "boundary_largest_risk_component_pixel_count": 3,
            "boundary_largest_risk_component_row_count": 2,
            "boundary_largest_risk_component_row_span": 2,
            "preview_owner_probe_status": "not_needed_unqualified_boundary_rgb_risk",
        },
    )
    assert not isolated_risk_triggered
    assert isolated_risk_audit["trigger_reasons"] == []

    risk_triggered, risk_audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 0,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 0.0,
            "boundary_risk_pixel_count": 60,
            "boundary_risk_row_count": 20,
            "boundary_risk_component_count": 1,
            "boundary_centreline_touching_risk_component_count": 1,
            "boundary_largest_risk_component_pixel_count": 60,
            "boundary_largest_risk_component_row_count": 20,
            "boundary_largest_risk_component_row_span": 20,
            "boundary_qualifying_risk_component_count": 1,
            "boundary_largest_qualifying_risk_component_pixel_count": 60,
            "boundary_largest_qualifying_risk_component_row_count": 20,
            "boundary_largest_qualifying_risk_component_row_span": 20,
            "boundary_high_rgb_risk": True,
            "preview_owner_probe_status": "graphcut",
            "preview_owner_probe_graphcut_used": True,
        },
    )
    assert risk_triggered
    assert risk_audit["trigger_reasons"] == ["boundary_high_rgb_risk"]

    hard_cut_triggered, hard_cut_audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 0,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 0.0,
            "boundary_risk_pixel_count": 60,
            "boundary_risk_row_count": 20,
            "boundary_risk_component_count": 1,
            "boundary_centreline_touching_risk_component_count": 1,
            "boundary_largest_risk_component_pixel_count": 60,
            "boundary_largest_risk_component_row_count": 20,
            "boundary_largest_risk_component_row_span": 20,
            "boundary_qualifying_risk_component_count": 1,
            "boundary_largest_qualifying_risk_component_pixel_count": 60,
            "boundary_largest_qualifying_risk_component_row_count": 20,
            "boundary_largest_qualifying_risk_component_row_span": 20,
            "boundary_high_rgb_risk": True,
            "preview_hard_cut_row_count": 40,
            "preview_full_height_hard_cut": True,
            "preview_owner_probe_status": "full_height_hard_cut",
        },
    )
    assert hard_cut_triggered
    assert hard_cut_audit["trigger_reasons"] == [
        "boundary_high_rgb_risk",
        "preview_full_height_hard_cut",
    ]

    with pytest.raises(RuntimeError, match="RGB-risk policy"):
        _geometry_trigger_from_preview(
            evidence,
            settings,
            {"observable_pixel_count": 32},
        )


def test_boundary_risk_trigger_requires_one_material_connected_component() -> None:
    common = np.ones((48, 17), dtype=bool)
    fragmented = np.zeros_like(common)
    fragmented[4:8, 7:11] = True
    fragmented[18:22, 7:11] = True
    fragmented_audit = _boundary_rgb_risk_audit(
        fragmented,
        common,
        nominal_boundary=8,
        half_width_pixels=2,
        preview_scale=1.0,
    )
    # The two small components have 32 pixels in total, but neither persists
    # across the required eight rows.  They stay protected by RGB ownership
    # without spending a depth-assisted local-mesh attempt.
    assert fragmented_audit["boundary_risk_pixel_count"] == 32
    assert fragmented_audit["boundary_risk_component_count"] == 2
    assert fragmented_audit["boundary_qualifying_risk_component_count"] == 0
    assert fragmented_audit["boundary_high_rgb_risk"] is False

    material = np.zeros_like(common)
    material[8:34, 7:10] = True
    material_audit = _boundary_rgb_risk_audit(
        material,
        common,
        nominal_boundary=8,
        half_width_pixels=2,
        preview_scale=1.0,
    )
    assert material_audit["boundary_risk_pixel_count"] == 78
    assert material_audit["boundary_risk_component_count"] == 1
    assert material_audit["boundary_qualifying_risk_component_count"] == 1
    assert material_audit["boundary_high_rgb_risk"] is True

    preview_material = np.zeros_like(common)
    preview_material[8:28, 7:10] = True
    preview_audit = _boundary_rgb_risk_audit(
        preview_material,
        common,
        nominal_boundary=8,
        half_width_pixels=2,
        preview_scale=0.75,
    )
    assert preview_audit["minimum_boundary_high_risk_component_pixel_count"] == 41
    assert preview_audit["minimum_boundary_high_risk_component_row_count"] == 14
    assert preview_audit["minimum_boundary_high_risk_component_row_span"] == 20
    assert preview_audit["boundary_high_rgb_risk"] is True


def test_geometry_mesh_fit_mask_uses_training_only_bidirectional_rgb_flow() -> None:
    """Held-out or locally bad flow can never enable a full-res mesh sample."""

    accepted = np.ones((4, 6), dtype=bool)
    held_out = np.zeros_like(accepted)
    held_out[1, 2] = True
    uncertain = np.zeros_like(accepted)
    uncertain[2, 1] = True
    fb = np.full(accepted.shape, 0.20, dtype=np.float64)
    fb[0, 4] = 0.90
    # 0.50 px in a half-resolution preview is a one-full-resolution-pixel
    # discrepancy.  It must not pass the formal 0.75 px support gate.
    fb[3, 2] = 0.50
    evidence = SimpleNamespace(
        accepted_mask=accepted,
        held_out_mask=held_out,
        observable_mask=np.ones_like(accepted),
        flow_uncertain_mask=uncertain,
        forward_backward_error=fb,
    )

    mask, audit = _lift_training_rgb_flow_consistency_to_geometry_tile(
        evidence,
        preview_origin_x=5.0,
        preview_scale=0.5,
        corridor_x=(10, 22),
        canvas_height=8,
        maximum_full_resolution_fb_error_pixels=0.75,
    )

    assert mask.shape == (8, 12)
    # Each preview pixel maps to its 2x2 full-resolution footprint.  The
    # held-out location, an uncertain location, and a local FB violation are
    # all excluded before mesh fitting, while ordinary training support stays.
    assert not mask[2:4, 4:6].any()
    assert not mask[4:6, 2:4].any()
    assert not mask[0:2, 8:10].any()
    assert not mask[6:8, 4:6].any()
    assert mask[0:2, 0:2].all()
    assert audit["preview_training_flow_consistent_pixel_count"] == 20
    assert audit["full_resolution_flow_consistent_pixel_count"] == int(
        np.count_nonzero(mask)
    )
    assert audit["metric_unit"] == "full_resolution_pixels"
    assert audit["maximum_full_resolution_fb_error_pixels"] == pytest.approx(0.75)
    assert audit["maximum_preview_fb_error_pixels"] == pytest.approx(0.375)


def test_geometry_flow_application_keeps_safe_rgb_holdout_for_validation() -> None:
    """Held-out RGB evidence is withheld from fitting, not from safe sampling."""

    accepted = np.ones((4, 6), dtype=bool)
    held_out = np.zeros_like(accepted)
    held_out[1, 2] = True
    evidence = SimpleNamespace(
        accepted_mask=accepted,
        held_out_mask=held_out,
        observable_mask=np.ones_like(accepted),
        flow_uncertain_mask=np.zeros_like(accepted),
        forward_backward_error=np.full(accepted.shape, 0.20, dtype=np.float64),
    )

    application, fit_support, audit = (
        _lift_rgb_flow_application_and_fit_support_to_geometry_tile(
            evidence,
            preview_origin_x=5.0,
            preview_scale=0.5,
            corridor_x=(10, 22),
            canvas_height=8,
            maximum_full_resolution_fb_error_pixels=0.75,
        )
    )

    # The held-out preview pixel expands to a 2x2 full-resolution footprint.
    # It can receive a validated mesh sample, but it cannot contribute RGB
    # evidence to fitting the mesh nodes.
    assert application[2:4, 4:6].all()
    assert not fit_support[2:4, 4:6].any()
    assert np.all(fit_support <= application)
    assert audit["preview_application_flow_consistent_pixel_count"] == 24
    assert audit["preview_training_flow_consistent_pixel_count"] == 23
    assert audit["full_resolution_application_flow_consistent_pixel_count"] == int(
        np.count_nonzero(application)
    )
    assert audit["full_resolution_flow_consistent_pixel_count"] == int(
        np.count_nonzero(fit_support)
    )


def test_geometry_rgb_transparency_protection_keeps_unsafe_edge_hard_owned() -> None:
    """Visual transparency/reflection evidence is independent of wall depth."""

    shape = (32, 64)
    accepted = np.ones(shape, dtype=bool)
    uncertain = np.zeros(shape, dtype=bool)
    occluded = np.zeros(shape, dtype=bool)
    edge = np.full(shape, np.nan, dtype=np.float64)
    # A flow-uncertain diagonal highlight, a well-tracked strong optical edge,
    # and an independently occluded point stand in for a reflective/transparent
    # foreground object.  The stable edge is deliberately accepted: stable
    # RGB flow alone cannot prove that a transparent visual layer is the wall.
    # The remote white-wall interior has no strong edge and stays blend-eligible.
    diagonal = np.arange(8, 16)
    uncertain[diagonal, diagonal] = True
    accepted[diagonal, diagonal] = False
    edge[diagonal, diagonal] = 1.0
    stable = np.arange(8, 16)
    edge[stable, 48] = 1.0
    occluded[6, 40] = True
    evidence = SimpleNamespace(
        accepted_mask=accepted,
        flow_uncertain_mask=uncertain,
        occluded_mask=occluded,
        edge_normal_step_pixels=edge,
    )

    protected, audit = _rgb_transparent_or_reflection_protection_from_preview(
        evidence,
        preview_origin_x=0.0,
        preview_scale=0.5,
        corridor_x=(0, 128),
        canvas_height=64,
        guard_radius_pixels=8,
    )

    assert protected.shape == (64, 128)
    # The nearest mapping plus 8 px guard covers the unsafe diagonal/object.
    assert protected[20, 20]
    assert protected[24, 96]
    assert protected[12, 80]
    # A sufficiently remote smooth white wall is not converted into a global
    # protection mask merely because another component was unreliable.
    assert not protected[56, 116]
    assert audit["preview_strong_rgb_structure_pixel_count"] == 16
    assert audit["preview_uncertain_or_rejected_strong_edge_pixel_count"] == 8
    assert audit["preview_unsafe_pixel_count"] == 17
    assert audit["full_resolution_protected_pixel_count"] == int(
        np.count_nonzero(protected)
    )
    assert audit["guard_radius_pixels"] == 8


def test_geometry_background_selector_cannot_bridge_a_visual_owner_guard() -> None:
    """The one permitted mesh layer is selected after visual protection."""

    # The raw depth layer is connected, but a strong transparent/reflective
    # RGB guard cuts it at the nominal owner boundary.  Neither resulting
    # island has support on both source sides, so the pair must hard-own rather
    # than silently fitting two mesh islands from the former raw component.
    visual_safe = np.ones((24, 40), dtype=bool)
    visual_safe[:, 19:21] = False
    selected, audit = pushbroom_module._select_single_virtual_background_component(
        visual_safe,
        corridor_x=(100, 140),
        nominal_boundary_x=120,
    )

    assert not selected.any()
    assert audit["depth_safe_component_count"] == 2
    assert audit["boundary_crossing_component_count"] == 0
    assert audit["selected_component_pixel_count"] == 0


def test_geometry_candidate_requires_same_held_out_strong_edge_improvement() -> None:
    held_out = np.ones((3, 3), dtype=bool)
    accepted = np.ones_like(held_out)
    before = SimpleNamespace(
        held_out_mask=held_out,
        accepted_mask=accepted,
        edge_normal_step_pixels=np.full(held_out.shape, 1.40, dtype=np.float64),
    )
    after = SimpleNamespace(
        held_out_mask=held_out.copy(),
        accepted_mask=accepted.copy(),
        edge_normal_step_pixels=np.full(held_out.shape, 0.35, dtype=np.float64),
    )

    accepted_audit = _held_out_strong_rgb_edge_gate(
        before, after, settings=GeometryAssistedSeamConfig()
    )
    assert accepted_audit["accepted"] is True
    assert accepted_audit["held_out_strong_edge_pixel_count"] == 9
    assert accepted_audit["held_out_strong_edge_improvement_ratio"] == pytest.approx(0.75)

    rejected_after = SimpleNamespace(
        held_out_mask=held_out.copy(),
        accepted_mask=accepted.copy(),
        edge_normal_step_pixels=np.full(held_out.shape, 0.90, dtype=np.float64),
    )
    rejected_audit = _held_out_strong_rgb_edge_gate(
        before, rejected_after, settings=GeometryAssistedSeamConfig()
    )
    assert rejected_audit["accepted"] is False
    assert rejected_audit["reason"] == "held_out_strong_edge_p95_exceeded"


def test_geometry_strong_edge_gate_converts_preview_error_to_full_resolution() -> None:
    held_out = np.ones((3, 3), dtype=bool)
    accepted = np.ones_like(held_out)
    before = SimpleNamespace(
        held_out_mask=held_out,
        accepted_mask=accepted,
        edge_normal_step_pixels=np.full(held_out.shape, 1.00, dtype=np.float64),
    )
    after = SimpleNamespace(
        held_out_mask=held_out.copy(),
        accepted_mask=accepted.copy(),
        edge_normal_step_pixels=np.full(held_out.shape, 0.40, dtype=np.float64),
    )

    audit = _held_out_strong_rgb_edge_gate(
        before,
        after,
        settings=GeometryAssistedSeamConfig(),
        preview_scale=0.50,
    )

    assert audit["metric_unit"] == "full_resolution_pixels"
    assert audit["held_out_strong_edge_p95_after_preview_pixels"] == pytest.approx(
        0.40
    )
    assert audit["held_out_strong_edge_p95_after_pixels"] == pytest.approx(0.80)
    assert audit["accepted"] is False
    assert audit["reason"] == "held_out_strong_edge_p95_exceeded"


def test_actual_rgb_line_gate_rejects_visible_mesh_bend_without_candidate_rgb() -> None:
    """A baseline door-frame line sees raw forward-mesh bend in output space."""

    preview = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.line(preview, (24, 3), (24, 44), (255, 255, 255), 2)
    common = np.ones(preview.shape[:2], dtype=bool)
    active = np.ones_like(common)
    bounds = TileBounds(0.0, 0.0, 63.0, 63.0)
    grid = np.asarray((0.0, 32.0, 63.0), dtype=np.float64)
    identity = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.zeros((3, 3), dtype=np.float64),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
    )
    curved = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 1.9, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
    )
    settings = GeometryAssistedSeamConfig()

    identity_audit = pushbroom_module._actual_rgb_line_straightness_gate(
        baseline_first_rgb=preview,
        baseline_second_rgb=preview,
        common_preview=common,
        active_preview=active,
        preview_left=0,
        preview_scale=0.75,
        mesh_warp=identity,
        settings=settings,
    )
    curved_audit = pushbroom_module._actual_rgb_line_straightness_gate(
        baseline_first_rgb=preview,
        baseline_second_rgb=preview,
        common_preview=common,
        active_preview=active,
        preview_left=0,
        preview_scale=0.75,
        mesh_warp=curved,
        settings=settings,
    )

    assert identity_audit["accepted"] is True
    assert identity_audit["observed"] is True
    assert identity_audit["tested_line_run_count"] >= 1
    assert curved_audit["accepted"] is False
    assert curved_audit["observed"] is True
    assert curved_audit["reason"] == "rgb_actual_line_bend_exceeded"
    assert curved_audit["maximum_line_bend_pixels"] > 1.0


def test_actual_rgb_line_gate_records_safe_wall_without_a_line_as_not_observed() -> None:
    """Owner-only structures cannot be required as a mesh-line witness."""

    preview = np.full((48, 48, 3), 128, dtype=np.uint8)
    bounds = TileBounds(0.0, 0.0, 63.0, 63.0)
    grid = np.asarray((0.0, 32.0, 63.0), dtype=np.float64)
    warp = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
    )

    audit = pushbroom_module._actual_rgb_line_straightness_gate(
        baseline_first_rgb=preview,
        baseline_second_rgb=preview,
        common_preview=np.ones(preview.shape[:2], dtype=bool),
        active_preview=np.ones(preview.shape[:2], dtype=bool),
        preview_left=0,
        preview_scale=0.75,
        mesh_warp=warp,
        settings=GeometryAssistedSeamConfig(),
    )

    assert audit["accepted"] is True
    assert audit["observed"] is False
    assert audit["reason"] == "not_observed_no_solver_valid_line"
    assert audit["tested_line_run_count"] == 0
    assert audit["maximum_line_bend_pixels"] is None


def test_geometry_active_audit_tracks_only_actual_safe_nonidentity_pixels() -> None:
    """A nominally active cell cannot make a guarded pixel appear mesh-active."""

    bounds = TileBounds(0.0, 0.0, 63.0, 63.0)
    grid = np.asarray((0.0, 32.0, 63.0), dtype=np.float64)
    same_layer = np.ones((64, 64), dtype=bool)
    same_layer[30:35, 30:35] = False
    warp = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
        same_layer_mask=same_layer,
        same_layer_origin_xy=(0.0, 0.0),
    )

    active = pushbroom_module._mesh_active_mask(
        warp, x0=0, x1=64, height=64
    )

    assert not active[32, 32]
    assert active[48, 48]


def test_geometry_topology_fallback_uses_exactly_one_fully_covering_source() -> None:
    first = np.ones((3, 4), dtype=bool)
    second = np.ones((3, 4), dtype=bool)
    second[0, 0] = False
    assert _select_pair_level_hard_owner(first, second) == 0

    first[1, 1] = False
    assert _select_pair_level_hard_owner(first, second) is None
    assert _preflight_failure_pair_index(
        "pair_local_component_owners_do_not_admit_a_monotonic_seam:17"
    ) == 17
    assert _preflight_failure_pair_index("unrelated_failure") is None


def test_geometry_topology_fallback_preserves_shared_sources_across_a_run() -> None:
    first = np.ones((3, 4), dtype=bool)
    second = np.ones((3, 4), dtype=bool)
    assert _pair_level_hard_owner_options(first, second) == (0, 1)

    resolved = _resolve_pair_level_hard_owners(
        (17, 18, 19, 42),
        {
            17: (0, 1),
            18: (1,),
            19: (0, 1),
            42: (0, 1),
        },
    )

    # The run prefers its second RGB source to retain every shared strip;
    # the isolated corridor prefers its first source.
    assert resolved == {17: 1, 18: 1, 19: 1, 42: 0}


def test_geometry_component_fallback_preserves_safe_pair_coverage() -> None:
    left = ProtectedComponentFragment(
        pair_index=4,
        global_bbox=(10, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        allowed_owners=(0, 1),
        component_label=1,
        preferred_owner=1,
    )
    right = ProtectedComponentFragment(
        pair_index=4,
        global_bbox=(22, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        allowed_owners=(0, 1),
        component_label=2,
        preferred_owner=0,
    )

    result = _force_monotonic_component_owners((left, right))

    assert result is not None
    forced, owners = result
    assert owners == {1: 0, 2: 1}
    assert [fragment.allowed_owners for fragment in forced] == [(0,), (1,)]


def test_geometry_pair_hard_owner_and_source_suppression_are_audited() -> None:
    first = np.ones((2, 3), dtype=bool)
    second = np.ones((2, 3), dtype=bool)
    owner0, owner1, cuts = _pair_level_hard_owner_masks(first, second, 1)
    assert not np.any(owner0)
    assert np.all(owner1)
    assert np.all(cuts == -1)

    suppressed = _audit_suppressed_source_frames(
        (100, 101, 102),
        (5, 0, 4),
        {0},
    )
    assert suppressed == [
        {
            "source_index": 1,
            "frame_id": 101,
            "adjacent_hard_owner_topology_pairs": [0],
            "reason": "fully_covered_by_audited_hard_owner_topology_decision",
        }
    ]
    with pytest.raises(RuntimeError, match="without an audited geometry seam decision"):
        _audit_suppressed_source_frames((100, 101), (1, 0), set())


def test_raw_geometry_coordinates_invert_the_calibrated_rgb_map_subpixel(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=13)
    settings = CalibratedRGBPushbroomConfig()
    scale = estimate_rgb_motion_pixels_per_mm(
        session.frames,
        poses,
        session.calibration,
        settings,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in session.frames],
        poses,
        session.calibration,
        scale,
        settings,
    )
    renderer = CalibratedRGBPushbroomRenderer(layout, session.calibration, poses)
    source_index = 1
    x0 = layout.support_left_x[source_index]
    x1 = min(x0 + 20, layout.support_right_x[source_index])
    tile = renderer.render_local_geometry_map(
        session.frames[source_index], source_index, x0=x0, x1=x1
    )
    yy, xx = np.nonzero(tile.valid_mask)
    raw = np.column_stack((tile.source_map_x[yy, xx], tile.source_map_y[yy, xx]))
    recovered = _raw_pixels_to_virtual_coordinates(
        raw,
        layout=layout,
        calibration=session.calibration,
        camera_to_world=poses[source_index],
        source_index=source_index,
        residual_warp=None,
    )
    expected = np.column_stack((x0 + xx, yy)).astype(np.float64)
    np.testing.assert_allclose(recovered, expected, atol=1e-4)


def test_pushbroom_uses_every_real_frame_once_with_bounded_strip_residency(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=17)
    motions = _reliable_adjacent_rgb_motions(len(session.frames))

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=motions,
    )

    metadata = result.metadata
    metrics = metadata["quality_metrics"]
    assert len(motions) == len(session.frames) - 1
    assert metadata["source_count"] == len(session.frames)
    assert metadata["frame_ids"] == [frame.frame_id for frame in session.frames]
    assert metadata["single_inverse_remap_per_source"] is True
    assert metadata["interpolated_pose_count"] == 0
    assert metrics["source_remap_count"] == len(session.frames)
    assert 2 <= metrics["maximum_resident_strips"] <= 5
    assert len(metadata["rgb_motion_scale"]["samples"]) == len(session.frames) - 1
    assert metadata["layout"]["endpoint_policy"] == "outward_half_fov"
    assert len(metadata["layout"]["endpoint_outer_owner_intervals_x"]) == 2
    assert metadata["layout"]["maximum_source_strip_width"] > 64
    supports = metadata["layout"]["source_support_intervals_x"]
    assert all(right - left <= 64 for left, right in supports[1:-1])
    assert all(count > 0 for count in metadata["source_owner_pixel_counts"])
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )
    assert metrics["endpoint_outer_half_fov_preserved"] is True
    assert metrics["endpoint_outer_half_fov_trimmed_column_counts"] == [0, 0]
    assert metrics["endpoint_outer_half_fov_trimmed_invalid_pixel_counts"] == [0, 0]


def test_pushbroom_keeps_outward_endpoint_coverage_when_virtual_x_is_reversed(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=19)
    reversed_poses = []
    for pose in poses:
        reversed_pose = pose.copy()
        reversed_pose[:3, :3] = np.diag((-1.0, 1.0, -1.0))
        reversed_poses.append(reversed_pose)

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        reversed_poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    layout = result.metadata["layout"]
    metrics = result.metadata["quality_metrics"]
    assert layout["temporal_to_virtual_x_sign"] < 0.0
    assert layout["endpoint_policy"] == "outward_half_fov"
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )


def test_pushbroom_pair_blends_are_narrow_and_never_include_rgb_risk(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=23)
    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    pairs = result.metadata["pairs"]
    metrics = result.metadata["quality_metrics"]
    assert len(pairs) == len(session.frames) - 1
    assert all(2 <= pair["blend_width_pixels"] <= 8 for pair in pairs)
    # Endpoint supports may be shortened by the outward-half-FOV policy, but
    # interior seams retain a 32--64 px search corridor that is independent
    # from the 2--8 px output blend band.
    assert all(2 <= pair["search_corridor_width_pixels"] <= 64 for pair in pairs)
    assert all(
        32 <= pair["search_corridor_width_pixels"] <= 64 for pair in pairs[1:-1]
    )
    assert all(
        pair["search_corridor_width_pixels"] > pair["blend_width_pixels"]
        for pair in pairs
    )
    assert all(pair["blend_zone_risk_pixel_count"] == 0 for pair in pairs)
    assert metrics["blend_zone_risk_pixel_count"] == 0
    assert metrics["blend_zone_risk_fraction"] == 0.0
    assert metrics["blend_zone_fraction"] <= 0.20


def test_pushbroom_crops_from_valid_mask_and_preserves_valid_black_rgb(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=31)
    black = np.zeros(
        (session.calibration.height, session.calibration.width, 3), dtype=np.uint8
    )
    # Preserve a neutral wall rail solely for the mandatory photometric
    # measurement; the large black region remains valid output content.
    black[:40, :, :] = 190
    for frame in session.frames:
        assert cv2.imwrite(str(frame.color_path), black)
        # Rendering after this deletion proves output pixels are not read from
        # the strict session's aligned-depth files.
        frame.aligned_depth_path.unlink()

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    crop = result.metadata["crop"]
    assert result.metadata["depth_used_for_output_pixels"] is False
    assert result.panorama.shape[:2] == (crop["height"], crop["width"])
    assert crop["width"] > 0 and crop["height"] > 0
    assert np.any(np.all(result.panorama == 0, axis=2))


def test_pushbroom_rejects_insufficient_reliable_rgb_motion_scale(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=41)
    invalid_motions = [
        {"dx": 0.0, "reliable": False} for _ in range(len(session.frames) - 1)
    ]

    with pytest.raises(RuntimeError, match="too few reliable adjacent RGB motion"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=invalid_motions,
        )


def test_pushbroom_rejects_unstable_rgb_motion_scale(tmp_path: Path) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=47)
    unstable_motions = [
        {"dx": value, "reliable": True} for value in (60.0, 120.0, 1200.0, 2400.0)
    ]

    with pytest.raises(RuntimeError, match="RGB-motion scale is unstable"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=unstable_motions,
        )


def test_global_linear_rgb_gain_solver_is_joint_per_channel_and_fail_closed() -> None:
    source_count = 5
    # Each BGR channel has its own linear log-gain slope.  The mean-zero gauge
    # is known analytically, and affine gain curves are not biased by the
    # second-difference regularizer.
    slope_bgr = np.array((0.045, -0.025, 0.030), dtype=np.float64)
    expected_log_gains = (
        np.arange(source_count, dtype=np.float64)[:, None] - 2.0
    ) * slope_bgr[None, :]
    edges = [
        _PhotometricEdge(
            log_relation_bgr=slope_bgr.copy(),
            support_pixels=1024,
            mad_bgr=np.full(3, 0.005, dtype=np.float64),
            raw_signed_l_delta=0.0,
        )
        for _ in range(source_count - 1)
    ]

    gains, metrics = _solve_global_linear_rgb_gains(source_count, edges)

    assert metrics["photometric_mode"] == "safe_wall_global_linear_rgb"
    assert metrics["photometric_global_solver"] is True
    np.testing.assert_allclose(np.log(gains), expected_log_gains, atol=1e-8)
    assert not np.allclose(gains[:, 0], gains[:, 1])
    assert not np.allclose(gains[:, 1], gains[:, 2])

    with pytest.raises(RuntimeError, match="No reliable safe white-wall"):
        _solve_global_linear_rgb_gains(source_count, [edges[0], None, *edges[2:]])


def test_linear_rgb_gain_application_is_per_channel_not_gamma_scalar() -> None:
    encoded = np.full((4, 5, 3), (96, 144, 192), dtype=np.uint8)
    gain_bgr = np.array((1.20, 0.82, 1.08), dtype=np.float64)

    corrected = _apply_linear_bgr_gain(encoded, gain_bgr)
    expected_linear = _srgb_to_linear_bgr(encoded) * gain_bgr.reshape(1, 1, 3)
    actual_linear = _srgb_to_linear_bgr(corrected)

    # Encoding quantization is the only allowed discrepancy; a scalar applied
    # to gamma-encoded RGB would fail this linear-light comparison.
    np.testing.assert_allclose(actual_linear, expected_linear, atol=0.005)
    assert not np.all(corrected[:, :, 0] == corrected[:, :, 1])
    assert not np.all(corrected[:, :, 1] == corrected[:, :, 2])


def test_protected_foreground_component_is_owned_wholly_by_one_source() -> None:
    height, width = 32, 64
    first = np.full((height, width, 3), 190, dtype=np.uint8)
    second = np.full((height, width, 3), 190, dtype=np.uint8)
    valid = np.ones((height, width), dtype=bool)
    # A horizontal hose spans the nominal seam.  Its uniform interior is part
    # of the supplied connected protection component, so the GraphCut owner
    # must not split it at the nominal centre.
    protected = np.zeros((height, width), dtype=bool)
    protected[12:20, 18:46] = True
    first[protected] = (20, 20, 20)
    second[protected] = (20, 20, 20)

    owner0, owner1, cuts, _, _, split_count, boundary_guard_count = (
        _graphcut_monotonic_owner(
            first,
            second,
            valid,
            valid,
            protected,
            nominal_boundary=32,
        )
    )

    assert np.all(owner0[protected]) or np.all(owner1[protected])
    assert not np.any(owner0[protected] & owner1[protected])
    assert split_count == 0
    assert boundary_guard_count == 0
    assert np.all(cuts[12:20] < 18) or np.all(cuts[12:20] >= 46)


def test_local_multiband_uses_distinct_owner_masks_and_adaptive_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 12, 20
    first = np.full((height, width, 3), (80, 120, 160), dtype=np.uint8)
    second = np.full((height, width, 3), (120, 160, 200), dtype=np.uint8)
    common = np.ones((height, width), dtype=bool)
    protected = np.zeros((height, width), dtype=bool)
    safe_wall = common.copy()
    owner0 = np.zeros_like(common)
    owner0[:, :8] = True
    owner1 = common & ~owner0
    cuts = np.full(height, 7, dtype=np.int32)
    captured_masks: list[np.ndarray] = []
    captured_images: list[np.ndarray] = []

    class _CapturingBlender:
        def setNumBands(self, _: int) -> None:
            pass

        def prepare(self, _: tuple[int, int, int, int]) -> None:
            pass

        def feed(
            self, image: np.ndarray, mask: np.ndarray, _: tuple[int, int]
        ) -> None:
            captured_images.append(np.asarray(image).copy())
            captured_masks.append(np.asarray(mask).copy())

        def blend(
            self, _: None, __: None
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                captured_images[0],
                np.where(captured_masks[0] | captured_masks[1], 255, 0).astype(
                    np.uint8
                ),
            )

    monkeypatch.setattr(
        pushbroom_module.cv2,
        "detail_MultiBandBlender",
        _CapturingBlender,
    )
    (
        _,
        zone,
        pixels,
        levels,
        mask0_pixels,
        mask1_pixels,
        masks_distinct,
    ) = _blend_safe_pair_zone(
        first,
        second,
        common,
        protected,
        safe_wall,
        owner0,
        owner1,
        cuts,
        blend_width=6,
        levels=3,
    )

    assert pixels == int(np.count_nonzero(zone)) > 0
    assert levels == 3
    assert mask0_pixels > 0 and mask1_pixels > 0
    assert masks_distinct is True
    assert len(captured_masks) == 2
    assert not np.array_equal(captured_masks[0], captured_masks[1])
    assert _adaptive_multiband_levels(2, 3) == 1
    assert _adaptive_multiband_levels(4, 3) == 2
    assert _adaptive_multiband_levels(8, 3) == 3


def test_preview_residual_diagnostics_keep_identity_output_and_remap_counts_separate(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=53)
    motions = _reliable_adjacent_rgb_motions(len(session.frames))
    default_result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=motions,
    )
    identity_result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        config={"residual_alignment": {"background_model": "identity"}},
        rgb_motions=motions,
    )

    # Preview evidence uses a separate low-resolution inverse remap.  It is
    # forbidden from changing a formal source sample while the selected model
    # remains identity.
    assert np.array_equal(default_result.panorama, identity_result.panorama)
    alignment = default_result.metadata["residual_alignment"]
    metrics = default_result.metadata["quality_metrics"]
    assert alignment["selected_model"] == "identity"
    assert alignment["analysis_preview_remap_count"] == len(session.frames)
    assert alignment["full_resolution_output_remap_count"] == len(session.frames)
    assert metrics["analysis_preview_remap_count"] == len(session.frames)
    assert metrics["full_resolution_output_remap_count"] == len(session.frames)
    assert len(alignment["evidence"]) == len(session.frames) - 1
    working = alignment["working_set_audit"]
    assert working["preview_streaming_maximum_resident_previews"] == 2
    assert (
        working["preview_evidence_pixel_count"]
        <= working["preview_evidence_hard_limit_pixels"]
    )
    assert working["preview_evidence_storage"] == "bounded_in_memory_analysis_only"


def test_preview_evidence_budget_fails_before_formal_full_resolution_remaps(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=59)

    with pytest.raises(RuntimeError, match="preview evidence exceeds"):
        render_calibrated_rgb_pushbroom(
            session.frames,
            poses,
            session.calibration,
            rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
            config={
                "residual_alignment": {
                    "maximum_evidence_megapixels": 0.000001,
                }
            },
        )
