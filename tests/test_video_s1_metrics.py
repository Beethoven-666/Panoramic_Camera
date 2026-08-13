from __future__ import annotations

import numpy as np

from panorama_demo.video_s1_handoff import HandoffObservation
from panorama_demo.video_s1_metrics import (
    EdgeResidualSample,
    EdgeRidge,
    StageVerticalMetrics,
    compare_stage_b_to_stage_c_per_pair,
    evaluate_heldout_single_copy,
    slope_compensated_residual_samples,
    summarize_vertical_residuals,
)


def _heldout(match_id: str, left_x: float, right_x: float, y: float = 10.0) -> HandoffObservation:
    return HandoffObservation(
        match_id=match_id,
        origin_pair_lineage="0->1",
        left_frame_id=0,
        right_frame_id=1,
        role="heldout",
        left_analysis_x=left_x,
        left_y=y,
        right_analysis_x=right_x,
        right_y=y,
        left_calibrated_x=left_x,
        right_calibrated_x=right_x,
        left_canvas_x=left_x,
        right_canvas_x=right_x,
        forward_backward_error_px=0.0,
        vertical_error_px=0.0,
        texture_score=1.0,
        observation_canvas_dx=right_x - left_x,
    )


def test_heldout_single_copy_reports_duplicate_missing_and_correct_copy() -> None:
    observations = ((
        _heldout("single", 10, 16),  # B=9: right copy only
        _heldout("duplicate", 10, 16),  # B=13: represented in separate call below
        _heldout("missing", 16, 10),
    ),)
    single = evaluate_heldout_single_copy(((observations[0][0],),), (9,))
    duplicate = evaluate_heldout_single_copy(((observations[0][1],),), (13,))
    missing = evaluate_heldout_single_copy(((observations[0][2],),), (13,))

    assert single.single_copy_count == 1
    assert duplicate.duplicate_count == 1
    assert missing.missing_count == 1


def test_single_copy_uses_half_open_owner_and_valid_evaluable_denominator() -> None:
    at_boundary = _heldout("edge", 20, 20)
    invalid = _heldout("invalid", 10, 30, y=99)
    mask = np.ones((32, 64), dtype=bool)
    result = evaluate_heldout_single_copy(
        ((at_boundary, invalid),),
        (20,),
        left_valid_masks=(mask,),
        right_valid_masks=(mask,),
    )

    assert result.total_heldout_count == 2
    assert result.evaluable_count == 1
    assert result.evaluable_fraction == 0.5
    assert result.samples[0].copy_count == 1
    assert result.samples[0].right_copy_present


def test_solver_observation_is_rejected_from_final_acceptance() -> None:
    solver = _heldout("wrong-role", 10, 20)
    solver = HandoffObservation(**{**solver.__dict__, "role": "solver"})

    with np.testing.assert_raises(ValueError):
        evaluate_heldout_single_copy(((solver,),), (15,))


def test_slope_compensation_uses_boundary_intercepts_not_adjacent_integer_rows() -> None:
    left = (EdgeRidge(40.20, 0.25, 80.0, 1.2, 8),)
    right = (EdgeRidge(40.55, 0.25, 75.0, 1.4, 8),)

    samples, double_edges = slope_compensated_residual_samples(left, right)

    assert double_edges == 0
    assert len(samples) == 1
    assert abs(samples[0].subpixel_residual_px - 0.35) < 1e-6
    assert samples[0].legacy_integer_peak_row_gap_px == 1


def test_large_legacy_gap_is_reported_without_silent_eight_pixel_trim() -> None:
    samples = (
        EdgeResidualSample(10.0, 19.5, 9.5, 10, 55.0, 1.5),
        EdgeResidualSample(30.0, 30.2, 0.2, 0, 60.0, 1.2),
    )
    metrics = summarize_vertical_residuals(samples, expected_sample_count=2)

    assert metrics.measurement_sample_count == 2
    assert metrics.subpixel_edge_residual_max == 9.5
    assert metrics.legacy_integer_peak_row_gap_p95 > 9.0
    assert metrics.coverage == 1.0


def test_strong_ridge_correspondence_beyond_eight_pixels_is_not_trimmed() -> None:
    descriptor = tuple(np.linspace(-0.4, 0.4, 13))
    norm = float(np.linalg.norm(descriptor))
    descriptor = tuple(item / norm for item in descriptor)
    left = (EdgeRidge(10.0, 0.0, 80.0, 1.0, 8, 1, descriptor),)
    right = (EdgeRidge(19.5, 0.0, 75.0, 1.0, 8, 1, descriptor),)

    samples, doubles = slope_compensated_residual_samples(left, right)

    assert doubles == 0
    assert len(samples) == 1
    assert samples[0].subpixel_residual_px == 9.5


def test_ridge_pairing_rejects_opposite_polarity_as_double_edges() -> None:
    left = (EdgeRidge(20.0, 0.0, 80.0, 1.0, 8, 1),)
    right = (EdgeRidge(20.1, 0.0, 80.0, 1.0, 8, -1),)

    samples, doubles = slope_compensated_residual_samples(left, right)

    assert not samples
    assert doubles == 2


def test_antiblur_statistics_report_strength_width_and_double_edges() -> None:
    samples = (
        EdgeResidualSample(10, 10.2, 0.2, 0, 80.0, 1.0),
        EdgeResidualSample(20, 20.4, 0.4, 0, 40.0, 3.0),
    )
    metrics = summarize_vertical_residuals(
        samples, expected_sample_count=4, double_edge_count=2
    )

    assert metrics.coverage == 0.5
    assert metrics.edge_peak_strength == 60.0
    assert metrics.edge_width == 2.0
    assert metrics.double_edge_count == 2


def test_stage_report_has_required_stage_b_c_subpixel_and_legacy_keys() -> None:
    before = summarize_vertical_residuals(
        (EdgeResidualSample(10, 11, 1.0, 1, 50, 1),), expected_sample_count=1
    )
    after = summarize_vertical_residuals(
        (EdgeResidualSample(10, 10.25, 0.25, 0, 50, 1),), expected_sample_count=1
    )
    report = StageVerticalMetrics(before, after).as_report()

    assert report["stage_b_subpixel_edge_residual_p95"] == 1.0
    assert report["stage_c_subpixel_edge_residual_p95"] == 0.25
    assert report["stage_c_subpixel_edge_residual_p99"] == 0.25
    assert report["legacy_integer_peak_row_gap_p95"] == 0.0
    assert report["stage_c_measurement_sample_count"] == 1


def test_per_pair_vertical_api_preserves_every_boundary_and_reports_p95() -> None:
    image = np.zeros((64, 48), dtype=np.uint8)
    image[20:, :] = 255
    image[44:, :] = 96
    boundaries = (16, 32)

    result = compare_stage_b_to_stage_c_per_pair(image, image, boundaries)

    assert [item.pair_index for item in result] == [0, 1]
    assert [item.boundary_x for item in result] == [16, 32]
    assert all(item.stage_c.measurement_sample_count >= 2 for item in result)
    assert all(item.stage_c.subpixel_edge_residual_p95 <= 1e-6 for item in result)
    assert all("stage_c_subpixel_edge_residual_p95" in item.as_report() for item in result)


def test_per_pair_vertical_api_keeps_unobservable_pair_record() -> None:
    blank = np.zeros((32, 32), dtype=np.uint8)

    result = compare_stage_b_to_stage_c_per_pair(blank, blank, (16,))

    assert len(result) == 1
    assert result[0].stage_c.measurement_sample_count == 0
    assert np.isnan(result[0].stage_c.subpixel_edge_residual_p95)
    assert result[0].stage_c.coverage == 0.0
