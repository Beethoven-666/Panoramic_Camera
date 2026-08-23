from __future__ import annotations

from dataclasses import replace

import numpy as np

from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s12_schedule import build_s012_schedule
from panorama_demo.video_s13_base_renderer import render_s13_p0
from panorama_demo.video_s13_selection import select_s13_vertical_parent
from panorama_demo.video_s13_quality import sequence_structure_decision
from panorama_demo.video_s13_vertical import estimate_s13_vertical


def _calibration() -> CameraIntrinsics:
    return CameraIntrinsics(96, 64, 80.0, 80.0, 47.5, 31.5, ())


def _schedule(calibration: CameraIntrinsics):
    return build_s012_schedule(
        (0, 1), (calibration.cx, calibration.cx + 8.0), calibration,
        hard_internal_width_px=None,
    )


def _textured_images() -> dict[int, np.ndarray]:
    rng = np.random.default_rng(17)
    left = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    right = np.roll(left, 1, axis=0)
    return {0: left, 1: right}


def _p0(schedule, calibration, images):
    return render_s13_p0(
        schedule,
        calibration,
        images.__getitem__,
        placement_methods=("test", "test"),
    ).image


def test_selection_records_full_resolution_render_for_all_gains_and_identity(
    monkeypatch,
) -> None:
    import panorama_demo.video_s13_selection as selection_module

    calibration = _calibration()
    schedule = _schedule(calibration)
    images = _textured_images()
    solution = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    p0_image = _p0(schedule, calibration, images)

    original_render = selection_module.render_s13_p1_from_raw
    rendered_shapes: list[tuple[int, int, int]] = []

    def recording_render(*args, **kwargs):
        result = original_render(*args, **kwargs)
        rendered_shapes.append(result.image.shape)
        return result

    monkeypatch.setattr(selection_module, "render_s13_p1_from_raw", recording_render)
    selected = select_s13_vertical_parent(
        schedule, calibration, images.__getitem__, solution, p0_image
    )

    candidates = selected.audit["gain_candidates"]
    assert [row["gain"] for row in candidates] == [0.0, 0.25, 0.5, 1.0]
    assert all(row["actual_full_resolution_render_compared"] is True for row in candidates)
    # Each gain renders its global base once; local candidates use only the
    # non-overlapping application bands and the selected result is rendered
    # once after parent selection.
    assert 4 <= len(rendered_shapes) <= 5
    assert set(rendered_shapes) == {(schedule.canvas_height, schedule.canvas_width, 3)}
    assert selected.audit["identity_always_candidate"] is True
    assert selected.audit["p0_metrics"]
    assert selected.audit["p0_mean_score"] is not None


def test_exact_probe_selection_matches_reference_full_selection() -> None:
    calibration = _calibration()
    schedule = _schedule(calibration)
    images = _textured_images()
    solution = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    p0_image = _p0(schedule, calibration, images)

    full = select_s13_vertical_parent(
        schedule, calibration, images.__getitem__, solution, p0_image,
        reference_final_remap=True,
    )
    probe = select_s13_vertical_parent(
        schedule, calibration, images.__getitem__, solution, p0_image,
        exact_seam_probes=True, reference_final_remap=True,
    )

    assert probe.audit["selection_domain"] == "exact_seam_probe"
    assert probe.stage == full.stage
    assert probe.selected_gain == full.selected_gain
    assert probe.audit["candidate_set_median_mad_scores"] == full.audit[
        "candidate_set_median_mad_scores"
    ]
    assert np.array_equal(probe.result.image, full.result.image)


def test_untextured_insufficient_evidence_selects_p0_identity() -> None:
    calibration = _calibration()
    schedule = _schedule(calibration)
    images = {
        0: np.zeros((64, 96, 3), dtype=np.uint8),
        1: np.zeros((64, 96, 3), dtype=np.uint8),
    }
    solution = estimate_s13_vertical(schedule, calibration, images.__getitem__)

    selected = select_s13_vertical_parent(
        schedule, calibration, images.__getitem__, solution,
        _p0(schedule, calibration, images),
    )

    assert selected.stage == "P0_identity"
    assert selected.selected_gain == 0.0
    assert selected.audit["selected_stage"] == "P0_identity"
    assert selected.audit["identity_always_candidate"] is True
    assert all(pair.status == "rolled_back" for pair in selected.solution.pairs)


def test_supported_local_candidate_rolls_back_when_structure_is_not_nondegrading(
    monkeypatch,
) -> None:
    import panorama_demo.video_s13_selection as selection_module

    calibration = _calibration()
    schedule = _schedule(calibration)
    images = _textured_images()
    measured = estimate_s13_vertical(schedule, calibration, images.__getitem__)

    supported_pair = replace(
        measured.pairs[0],
        supported_row_count=calibration.height,
        status="candidate",
        failure_reason=None,
    )
    forced_rows = {
        key: (np.full(calibration.height, 1.5, dtype=np.float32),)
        for key in measured.gain_local_row_residuals
    }
    solution = replace(
        measured,
        pairs=(supported_pair,),
        gain_local_row_residuals=forced_rows,
    )
    monkeypatch.setattr(
        selection_module,
        "structurally_non_degrading",
        lambda _before, _after: (False, "synthetic_structure_degraded"),
    )

    selected = select_s13_vertical_parent(
        schedule, calibration, images.__getitem__, solution,
        _p0(schedule, calibration, images),
    )

    for gain_row in selected.audit["gain_candidates"]:
        decision = gain_row["pair_local_decisions"][0]
        assert decision["decision"] == "rolled_back"
        assert decision["reason"] == "synthetic_structure_degraded"
        assert decision["global_only_metrics"]["evaluable"] is True
        assert decision["global_plus_local_metrics"]["evaluable"] is True
        assert decision["global_only_metrics"]["score"] is not None
        assert decision["global_plus_local_metrics"]["score"] is not None
    assert selected.solution.pairs[0].status == "rolled_back"
    assert np.all(selected.solution.local_row_residuals[0] == 0.0)


def _sequence_metric(*, score: float, double_edge: float = 1.0) -> dict[str, object]:
    return {
        "evaluable": True,
        "reason": None,
        "color_jump": 1.0,
        "gradient_jump": 1.0,
        "double_edge": double_edge,
        "motion_residual": 1.0,
        "thin_structure": 1.0,
        "horizontal_edge_mismatch": 1.0,
        "score": score,
        "radius_px": 4,
    }


def test_sequence_parent_allows_bounded_minor_component_outliers() -> None:
    before = tuple(_sequence_metric(score=10.0) for _ in range(84))
    after = tuple(
        _sequence_metric(score=9.0, double_edge=1.5 if index < 4 else 1.0)
        for index in range(84)
    )

    eligible, audit = sequence_structure_decision(before, after)

    assert eligible is True
    assert audit["component_failure_count"] == 4
    assert audit["component_failure_budget"] == 4
    assert audit["pair_total_score_regression_indices"] == []
    assert audit["aggregate_improvement_fraction"] == 0.1


def test_sequence_parent_rejects_excess_or_catastrophic_component_failures() -> None:
    before = tuple(_sequence_metric(score=10.0) for _ in range(84))
    too_many = tuple(
        _sequence_metric(score=9.0, double_edge=1.5 if index < 5 else 1.0)
        for index in range(84)
    )
    catastrophic = tuple(
        _sequence_metric(score=9.0, double_edge=4.0 if index == 0 else 1.0)
        for index in range(84)
    )

    eligible_many, audit_many = sequence_structure_decision(before, too_many)
    eligible_catastrophic, audit_catastrophic = sequence_structure_decision(
        before, catastrophic
    )

    assert eligible_many is False
    assert audit_many["reason"] == "component_failure_budget_exceeded"
    assert eligible_catastrophic is False
    assert audit_catastrophic["reason"] == "catastrophic_component_regression"


def test_sequence_parent_can_budget_one_noncatastrophic_pair_score_outlier() -> None:
    before = tuple(_sequence_metric(score=10.0) for _ in range(84))
    after = tuple(
        _sequence_metric(score=10.5 if index == 0 else 8.0)
        for index in range(84)
    )

    eligible, audit = sequence_structure_decision(
        before,
        after,
        maximum_component_failure_fraction=0.10,
        maximum_pair_score_failure_fraction=0.02,
    )

    assert eligible is True
    assert audit["pair_total_score_regression_indices"] == [0]
    assert audit["pair_total_score_failure_budget"] == 1
    assert audit["catastrophic_pair_total_score_regressions"] == []
