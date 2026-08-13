from __future__ import annotations

import numpy as np

from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s12_schedule import build_s012_schedule
from panorama_demo.video_s13_alignment import (
    S13AlignmentConfig,
    S13ApplicationBand,
    estimate_s13_pair_alignment,
    reestimate_s13_final_corridor_alignment,
)
from panorama_demo.video_s13_m5 import (
    build_s13_p2_replay,
    estimate_s13_m5_transactions,
    render_s13_p2_from_raw,
    run_s13_m5,
)
from panorama_demo.video_s13_quality import (
    long_horizontal_structure_metrics,
    long_horizontal_structure_nondegrading,
)
from panorama_demo.video_s13_seam import select_s13_seam
from panorama_demo.video_s13_vertical import estimate_s13_vertical, render_s13_p1_from_raw


def _calibration() -> CameraIntrinsics:
    return CameraIntrinsics(96, 64, 80.0, 80.0, 47.5, 31.5, ())


def _identity_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(96, dtype=np.float32), np.arange(64, dtype=np.float32)
    )
    return u, v, np.ones((64, 96), dtype=bool)


def _correspondences() -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.linspace(18, 76, 12), np.linspace(6, 57, 8))
    reference = np.column_stack((x.ravel(), y.ravel())).astype(np.float64)
    return reference, reference + np.asarray((0.75, 0.25))


def _alignment(**overrides):
    u, v, valid = _identity_grid()
    reference, moving = _correspondences()
    arguments = {
        "pair_index": 0,
        "pair_frame_ids": (10, 11),
        "non_reference_side": "right",
        "p0_source_u": u,
        "p0_source_v": v,
        "p0_valid": valid,
        "source_size": (96, 64),
        "reference_points_xy": reference,
        "non_reference_points_xy": moving,
        "application_band": S13ApplicationBand.straight(
            height=64, left_x=32, right_x=64
        ),
        "accepted_vertical_dy_by_row": np.full(64, 0.25),
        "vertical_accepted": True,
        "quality_evaluator": lambda model, _u, _v, _valid: {
            "non_degrading": model == "C2_subpixel_translation",
            "reason": None if model == "C2_subpixel_translation" else "forced_simpler_fallback",
        },
    }
    arguments.update(overrides)
    return estimate_s13_pair_alignment(**arguments)


def test_alignment_exposes_c0_to_c4_fallback_jacobian_and_identity_taper() -> None:
    alignment = _alignment()
    candidates = {candidate.model: candidate for candidate in alignment.candidates}

    assert tuple(candidates) == (
        "C0_identity",
        "C1_accepted_vertical",
        "C2_subpixel_translation",
        "C3_translation_tiny_rotation",
        "C4_bounded_light_affine",
    )
    assert alignment.fallback_order == (
        "C4_bounded_light_affine",
        "C3_translation_tiny_rotation",
        "C2_subpixel_translation",
        "C1_accepted_vertical",
        "C0_identity",
    )
    assert alignment.selected_model == "C2_subpixel_translation"
    assert alignment.non_reference_side == "right"
    assert alignment.alignment_shoulder[1] - alignment.alignment_shoulder[0] == 96
    assert candidates["C0_identity"].accepted is True
    assert candidates["C3_translation_tiny_rotation"].accepted is False
    assert candidates["C4_bounded_light_affine"].accepted is False

    selected = alignment.selected
    assert selected.audit["candidate_built_from_immutable_p0_grid"] is True
    assert selected.audit["continuous_taper_to_identity"] is True
    assert selected.audit["map_finite"] is True
    assert selected.audit["map_in_source_bounds"] is True
    assert selected.audit["positive_jacobian"] is True
    assert selected.audit["minimum_jacobian"] > 0.0
    assert np.all(selected.target_delta_u[:, 0] == 0.0)
    assert np.allclose(selected.target_delta_u[:, -1], 0.0, atol=1e-6)
    assert np.all(selected.target_delta_v[:, 0] == 0.0)
    assert np.allclose(selected.target_delta_v[:, -1], 0.0, atol=1e-6)


def test_alignment_rejects_estimated_model_with_bad_absolute_held_out_p95() -> None:
    u, v, valid = _identity_grid()
    reference, moving = _correspondences()
    key = np.floor(reference[:, 0]).astype(np.int64) * 17 + np.floor(
        reference[:, 1]
    ).astype(np.int64) * 31
    moved = moving.copy()
    moved[key % 5 == 0, 0] += 3.0
    alignment = estimate_s13_pair_alignment(
        pair_index=0,
        pair_frame_ids=(10, 11),
        non_reference_side="right",
        p0_source_u=u,
        p0_source_v=v,
        p0_valid=valid,
        source_size=(96, 64),
        reference_points_xy=reference,
        non_reference_points_xy=moved,
        application_band=S13ApplicationBand.straight(height=64, left_x=32, right_x=64),
        config=S13AlignmentConfig(maximum_estimated_model_held_out_p95_px=1.5),
    )
    translation = next(
        candidate for candidate in alignment.candidates
        if candidate.model == "C2_subpixel_translation"
    )
    assert translation.accepted is False
    assert translation.failure_reason == "held_out_residual_out_of_bounds"
    assert translation.metrics["residual_p95_px"] > 1.5


def test_final_corridor_geometry_is_reestimated_around_moved_seam_from_p0() -> None:
    seam = 48 + np.rint(2.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, 64))).astype(np.int32)
    u, v, valid = _identity_grid()
    reference, moving = _correspondences()
    alignment = reestimate_s13_final_corridor_alignment(
        final_seam_x_by_row=seam,
        application_half_width_px=4,
        allowed_left_x_by_row=np.full(64, 38, dtype=np.int32),
        allowed_right_x_by_row=np.full(64, 59, dtype=np.int32),
        pair_index=0,
        pair_frame_ids=(10, 11),
        non_reference_side="right",
        p0_source_u=u,
        p0_source_v=v,
        p0_valid=valid,
        source_size=(96, 64),
        reference_points_xy=reference,
        non_reference_points_xy=moving,
        accepted_vertical_dy_by_row=np.full(64, 0.25),
        vertical_accepted=True,
        quality_evaluator=lambda model, _u, _v, _valid: {
            "non_degrading": model == "C2_subpixel_translation"
        },
    )

    assert alignment.reestimated_for_final_seam is True
    assert np.all(alignment.application_band.left_x_by_row <= seam - 4)
    assert np.all(alignment.application_band.right_x_by_row >= seam + 5)
    assert all(
        candidate.audit["candidate_built_from_immutable_p0_grid"] is True
        for candidate in alignment.candidates
    )


def test_seam_dp_has_bounded_unit_steps_and_complete_fallback_chain(monkeypatch) -> None:
    import panorama_demo.video_s13_seam as seam_module

    rng = np.random.default_rng(9)
    left = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    right = np.roll(left, 1, axis=1)
    valid = np.ones((48, 64), dtype=bool)
    result = select_s13_seam(left, right, valid, valid, 32, maximum_shift_px=6)
    assert result.model == "monotone_dp"
    assert np.all(np.isin(np.diff(result.seam_x_by_row), (-1, 0, 1)))
    assert np.max(np.abs(result.seam_x_by_row - 32)) <= 6
    assert result.audit["ordered_non_crossing_search_bounds"] is True
    assert set(result.cost_components) == {
        "color", "gradient_magnitude", "gradient_direction", "double_edge",
        "motion_residual", "thin_strong_structure", "boundary_offset", "slope", "curvature",
    }

    monkeypatch.setattr(
        seam_module, "_dp_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forced DP failure")),
    )
    shifted = select_s13_seam(left, right, valid, valid, 32, maximum_shift_px=6)
    assert shifted.model == "shifted_straight"
    assert shifted.fallback_chain == ("monotone_dp", "shifted_straight")

    monkeypatch.setattr(
        seam_module, "_straight_candidate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forced straight failure")),
    )
    midpoint = select_s13_seam(left, right, valid, valid, 32, maximum_shift_px=6)
    assert midpoint.model == "midpoint_straight"
    assert midpoint.fallback_chain == (
        "monotone_dp", "shifted_straight", "midpoint_straight"
    )
    assert np.all(midpoint.seam_x_by_row == 32)


def test_seam_dp_requires_cost_margin_over_shifted_straight(monkeypatch) -> None:
    import panorama_demo.video_s13_seam as seam_module

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    valid = np.ones((48, 64), dtype=bool)
    dp = np.full(48, 31, dtype=np.int32)
    straight = np.full(48, 32, dtype=np.int32)
    monkeypatch.setattr(seam_module, "_dp_candidate", lambda *_a, **_k: (dp, 99.0))
    monkeypatch.setattr(seam_module, "_straight_candidate", lambda *_a, **_k: (straight, 100.0))

    simple = select_s13_seam(image, image, valid, valid, 32)
    assert simple.model == "shifted_straight"
    assert simple.audit["selection_reason"] == "straight_preferred_without_dp_margin"

    monkeypatch.setattr(seam_module, "_dp_candidate", lambda *_a, **_k: (dp, 95.0))
    curved = select_s13_seam(image, image, valid, valid, 32)
    assert curved.model == "monotone_dp"
    assert curved.audit["dp_relative_improvement"] == 0.05


def test_dp_is_generated_and_ranked_when_shifted_straight_fails(monkeypatch) -> None:
    import panorama_demo.video_s13_seam as seam_module

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    valid = np.ones((48, 64), dtype=bool)
    dp = np.full(48, 31, dtype=np.int32)
    monkeypatch.setattr(seam_module, "_dp_candidate", lambda *_a, **_k: (dp, 80.0))
    monkeypatch.setattr(
        seam_module,
        "_straight_candidate",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("forced straight failure")),
    )

    result = select_s13_seam(image, image, valid, valid, 32)

    assert result.model == "monotone_dp"
    assert result.audit["selection_reason"] == "dp_ranked_by_independent_local_objective"
    assert "monotone_dp" in result.candidate_seams
    assert np.array_equal(result.seam_x_by_row, dp)


def test_long_horizontal_structure_rejects_vertical_step() -> None:
    aligned = np.zeros((64, 96, 3), dtype=np.uint8)
    aligned[30:34] = 255
    stepped = aligned.copy()
    stepped[:, 48:] = 0
    stepped[32:36, 48:] = 255
    seam = np.full(64, 48, dtype=np.int32)

    before = long_horizontal_structure_metrics(aligned, seam)
    after = long_horizontal_structure_metrics(stepped, seam)
    accepted, reason = long_horizontal_structure_nondegrading(before, after)

    assert before["observed"] is True
    assert after["observed"] is True
    assert before["absolute_best_vertical_lag_px"] == 0
    assert after["absolute_best_vertical_lag_px"] >= 1
    assert accepted is False
    assert reason == "horizontal_structure_continuity_degraded"


def test_long_horizontal_structure_rejects_new_displaced_support() -> None:
    blank = np.zeros((64, 96, 3), dtype=np.uint8)
    stepped = blank.copy()
    stepped[28:32, :48] = 255
    stepped[31:35, 48:] = 255
    seam = np.full(64, 48, dtype=np.int32)

    before = long_horizontal_structure_metrics(blank, seam)
    after = long_horizontal_structure_metrics(stepped, seam)
    accepted, reason = long_horizontal_structure_nondegrading(before, after)

    assert before["observed"] is False
    assert after["observed"] is True
    assert after["line_score"] > 0.25
    assert accepted is False
    assert reason == "horizontal_structure_new_misalignment"


def test_symmetric_output_audit_rejects_seam_created_horizontal_step() -> None:
    import panorama_demo.video_s13_m5 as m5_module

    aligned = np.zeros((64, 96, 3), dtype=np.uint8)
    aligned[28:32] = 255
    damaged = aligned.copy()
    damaged[:, 48:] = 0
    damaged[31:35, 48:] = 255
    base = np.full(64, 48, dtype=np.int32)
    candidate = 48 + np.rint(3.0 * np.sin(np.linspace(0, 2 * np.pi, 64))).astype(
        np.int32
    )

    audit = m5_module._audit_rendered_seam_change(
        aligned, damaged, base, candidate
    )

    assert audit["eligible"] is False
    assert audit["comparison_policy"] == (
        "symmetric_before_and_after_owner_paths_with_held_out_shoulders"
    )


def test_owner_seam_audit_uses_actual_before_path_not_midpoint(monkeypatch) -> None:
    import panorama_demo.video_s13_m5 as m5_module

    image = np.zeros((64, 96, 3), dtype=np.uint8)
    shifted_straight = np.full(64, 45, dtype=np.int32)
    dp = 48 + np.rint(2.0 * np.sin(np.linspace(0, 2 * np.pi, 64))).astype(
        np.int32
    )
    observed: dict[str, np.ndarray] = {}

    def recording_audit(before_image, after_image, base_seam, candidate_seam):
        observed["base"] = np.asarray(base_seam).copy()
        observed["candidate"] = np.asarray(candidate_seam).copy()
        return {"eligible": True, "reason": None}

    monkeypatch.setattr(
        m5_module, "_audit_rendered_seam_change", recording_audit
    )
    m5_module._audit_owner_seam_change(
        image, image, shifted_straight, dp
    )

    assert np.array_equal(observed["base"], shifted_straight)
    assert np.array_equal(observed["candidate"], dp)


def _m5_inputs():
    calibration = _calibration()
    schedule = build_s012_schedule(
        (0, 1, 2), (47.5, 55.5, 63.5), calibration,
        hard_internal_width_px=None,
    )
    rng = np.random.default_rng(31)
    base = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    images = {0: base, 1: np.roll(base, 1, axis=1), 2: np.roll(base, 2, axis=1)}
    vertical = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    return calibration, schedule, images, vertical


def test_transactions_cover_every_pair_and_p2_remaps_each_raw_source_once() -> None:
    calibration, schedule, images, vertical = _m5_inputs()
    pairs = estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="a" * 64,
    )

    assert len(pairs) == len(schedule.assignments) - 1
    assert [pair.transaction["transaction_id"] for pair in pairs] == [
        "m5-pair-0000", "m5-pair-0001"
    ]
    for pair in pairs:
        assert pair.transaction["decision"] in {"applied", "rolled_back"}
        assert "before_metrics" in pair.transaction
        assert "candidate_after_metrics" in pair.transaction
        assert "after_metrics" in pair.transaction
        assert pair.transaction["comparison_coordinate_policy"] == (
            "same_owner_geometry_plus_symmetric_base_candidate_seam"
        )
        assert pair.transaction["parent_stage_sha256"] == "a" * 64
        if pair.transaction["decision"] == "applied":
            assert pair.transaction[
                "final_corridor_geometry_reestimated_from_immutable_p0"
            ] is True
            assert pair.alignment is not None
            assert pair.alignment.reestimated_for_final_seam is True
        else:
            assert pair.transaction["fallback_used"] is True
            assert pair.transaction["selected_geometry_model"] == "C0_identity"

    calls: list[int] = []

    def load(frame_id: int) -> np.ndarray:
        calls.append(frame_id)
        return images[frame_id]

    result = render_s13_p2_from_raw(
        schedule, calibration, load, vertical, pairs, final_seams=True,
        selected_hypothesis_ids=(-1, -1, -1),
        placement_methods=("test", "test", "test"),
    )
    assert calls == [0, 1, 2]
    assert result.remap_invocations == 3
    assert set(result.pixel_provenance) >= {
        "owner_frame_id", "owner_source_index", "assignment_index", "source_u", "source_v",
        "valid", "selected_motion_hypothesis_id", "placement_method_code",
        "geometry_transaction_id", "seam_transaction_id", "photometric_transaction_id",
        "blend_transaction_id", "secondary_frame_id", "secondary_source_u",
        "secondary_source_v", "secondary_weight",
    }
    valid = result.pixel_provenance["valid"]
    assert np.array_equal(valid, result.pixel_provenance["owner_frame_id"] >= 0)
    assert np.isfinite(result.pixel_provenance["source_u"][valid]).all()
    assert np.isfinite(result.pixel_provenance["source_v"][valid]).all()
    assert np.all(result.pixel_provenance["secondary_frame_id"] == -1)
    assert np.all(result.pixel_provenance["secondary_weight"] == 0.0)


def test_frozen_source_map_provider_is_shared_by_render_and_replay() -> None:
    calibration, schedule, images, vertical = _m5_inputs()
    pairs = estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="e" * 64,
    )
    calls: list[tuple[int, int, int]] = []

    def provider(source_index: int, x0: int, x1: int):
        calls.append((source_index, x0, x1))
        height = schedule.canvas_height
        width = x1 - x0
        u = np.broadcast_to(
            np.arange(x0, x1, dtype=np.float32)[None, :], (height, width)
        ).copy()
        v = np.broadcast_to(
            np.arange(height, dtype=np.float32)[:, None], (height, width)
        ).copy()
        valid = np.ones((height, width), dtype=bool)
        field = np.full((height, width), source_index, dtype=np.int32)
        return u, v, valid, field

    rendered = render_s13_p2_from_raw(
        schedule, calibration, images.__getitem__, vertical, pairs,
        final_seams=True, map_provider=provider,
    )
    replay = build_s13_p2_replay(
        schedule, calibration, vertical, pairs, map_provider=provider,
    )

    assert np.array_equal(
        rendered.pixel_provenance["component_correction_field_id"][
            rendered.pixel_provenance["owner_source_index"] == 1
        ],
        np.ones(np.count_nonzero(
            rendered.pixel_provenance["owner_source_index"] == 1
        ), dtype=np.int32),
    )
    assert np.all(replay[0].left_component_correction_field_id == 0)
    assert np.all(replay[0].right_component_correction_field_id == 1)
    assert calls


def test_transactions_record_same_geometry_seam_and_independent_seam_audits() -> None:
    calibration, schedule, images, vertical = _m5_inputs()
    pairs = estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="b" * 64,
    )

    for pair in pairs:
        transaction = pair.transaction
        assert transaction["comparison_coordinate_policy"] == (
            "same_owner_geometry_plus_symmetric_base_candidate_seam"
        )
        if transaction["decision"] == "applied":
            assert transaction["geometry_comparison_seam_sha256"] == (
                transaction["selected_seam_sha256"]
            )
            assert transaction["held_out_seam_selection_audits"]
            assert all(
                audit["selected_seam_sha256"]
                for audit in transaction["held_out_seam_selection_audits"]
            )
        else:
            assert transaction["fallback_reason"]


def test_run_m5_reports_same_seam_sequence_selection_policy() -> None:
    calibration, schedule, images, vertical = _m5_inputs()
    parent = render_s13_p1_from_raw(
        schedule, calibration, images.__getitem__, vertical
    ).image

    result = run_s13_m5(
        schedule, calibration, images.__getitem__, vertical, parent,
        parent_stage_sha256="c" * 64,
    )

    assert result.selection_audit["comparison_coordinate_policy"] == (
        "same_owner_geometry_plus_symmetric_base_candidate_seam"
    )
    assert result.selection_audit["applied_unevaluable_indices"] == []
    assert "seam_output_sequence_audit" in result.selection_audit
    assert len(result.selection_audit["seam_output_pair_audits"]) == 2


def test_diagnostic_score_cannot_reject_hard_safe_p2(monkeypatch) -> None:
    import panorama_demo.video_s13_m5 as m5_module

    calibration, schedule, images, vertical = _m5_inputs()
    parent = render_s13_p1_from_raw(schedule, calibration, images.__getitem__, vertical).image
    monkeypatch.setattr(
        m5_module,
        "sequence_structure_decision",
        lambda *_args, **_kwargs: (
            False,
            {"eligible": False, "reason": "aggregate_improvement_not_proven"},
        ),
    )

    result = run_s13_m5(
        schedule, calibration, images.__getitem__, vertical, parent,
        parent_stage_sha256="d" * 64,
    )

    assert result.selected_as_best is False
    assert result.diagnostic_quality["runtime_authority"] is False
    assert result.hard_audit_passed is True


def test_each_evaluated_candidate_reestimates_its_own_corridor(
    monkeypatch,
) -> None:
    import panorama_demo.video_s13_m5 as m5_module

    calibration, schedule, images, vertical = _m5_inputs()
    original = m5_module.reestimate_s13_final_corridor_alignment
    observed: list[str] = []

    def recording_reestimate(*args, **kwargs):
        seam = np.asarray(kwargs["final_seam_x_by_row"], dtype=np.int32)
        observed.append(m5_module._sha_array(seam))
        return original(*args, **kwargs)

    monkeypatch.setenv("G305_S13_M5_AUDIT_ALL_CANDIDATES", "1")
    monkeypatch.setattr(
        m5_module, "reestimate_s13_final_corridor_alignment", recording_reestimate
    )
    pairs = estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="e" * 64,
    )

    evaluated_hashes = [
        evaluation["seam_sha256"]
        for pair in pairs
        for evaluation in pair.transaction["candidate_evaluations"]
        if evaluation["evaluation_status"] != "skipped_after_higher_rank_safe_candidate"
    ]
    assert observed == evaluated_hashes
    assert any(
        evaluation["model_name"] == "monotone_dp"
        for pair in pairs
        for evaluation in pair.transaction["candidate_evaluations"]
    )
