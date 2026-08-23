from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_m51_r2 import S13M51R2Config, S13M51R3Config
from panorama_demo.video_s13_m51_r3_diagnostic import (
    assess_c2e_diagnostic_roi,
    build_c2e_diagnostic_from_pair_transaction,
)
from tests.test_video_s13_m51_r2 import _small_m5_inputs
import panorama_demo.video_s13_m5 as m5_module


def _v6() -> S13M51R3Config:
    return S13M51R3Config(
        enabled=True,
        lk_error_mad_floor=0.25,
        lk_error_absolute_maximum=12.0,
        complete_reassessment_pair_indices=(71,),
    )


def test_v6_reassesses_only_pair_71_without_enabling_micro_rescue() -> None:
    config = _v6()
    assert config.complete_reassessment_pair_indices == (71,)
    assert config.requires_complete_seam_reassessment(71) is True
    assert config.requires_complete_seam_reassessment(70) is False
    assert config.requires_complete_seam_reassessment(72) is False
    assert config.micro_rescue_enabled is False


def test_complete_reassessment_records_s2_s1_s0_even_for_equal_seams() -> None:
    calibration, schedule, images, vertical = _small_m5_inputs()
    config = S13M51R3Config(
        enabled=True,
        lk_error_mad_floor=0.25,
        lk_error_absolute_maximum=12.0,
        complete_reassessment_pair_indices=(0,),
    )
    pair = m5_module.estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="e" * 64, m51_r2_config=config,
    )[0]
    evaluations = pair.transaction["candidate_evaluations"]
    assert pair.transaction["schema"] == (
        "gemini305-video-s13-m5-pair-transaction/v4"
    )
    assert {row["model_code"] for row in evaluations} == {"S2", "S1", "S0"}
    assert all(
        row["evaluation_status"] != "skipped_after_higher_rank_safe_candidate"
        for row in evaluations
    )
    assert len({row["candidate_id"] for row in evaluations}) == 3
    assert pair.transaction["complete_seam_reassessment_diagnostic"] is True
    assert pair.transaction["micro_rescue"] == {
        "enabled": False, "attempted": False, "accepted": False
    }
    assert pair.transaction["c2e"] == {
        "enabled": False, "attempted": False, "accepted": False
    }
    report = build_c2e_diagnostic_from_pair_transaction(pair.transaction)
    assert report["source_pair_transaction_id"] == pair.transaction["transaction_id"]
    assert report["applied_to_p2"] is False
    assert report["all_13_pass"] is False
    assert report["unevaluable_count"] > 0


def test_v6_non_target_pair_keeps_v5_selection_and_seam() -> None:
    calibration, schedule, images, vertical = _small_m5_inputs()
    common = {
        "enabled": True,
        "lk_error_mad_floor": 0.25,
        "lk_error_absolute_maximum": 12.0,
    }
    v5 = m5_module.estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="e" * 64,
        m51_r2_config=S13M51R2Config(**common),
    )[0]
    v6 = m5_module.estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="e" * 64,
        m51_r2_config=S13M51R3Config(**common),
    )[0]

    assert v6.transaction["complete_seam_reassessment_diagnostic"] is False
    assert v6.transaction["selected_candidate_id"] == v5.transaction["selected_candidate_id"]
    assert v6.transaction["selected_geometry_model"] == v5.transaction["selected_geometry_model"]
    assert np.array_equal(v6.seam_x_by_row, v5.seam_x_by_row)


def test_c2e_diagnostic_reports_all_thirteen_passes_without_authority() -> None:
    report = assess_c2e_diagnostic_roi({
        "held_out_edge_p95_px": 0.9,
        "absolute_improvement_px": 0.6,
        "relative_improvement_fraction": 0.31,
        "forward_reverse_lag_discrepancy_px": 0.5,
        "best_second_uniqueness_fraction": 0.10,
        "orientation_difference_degrees": 10.0,
        "support_retention": 0.98,
        "minimum_jacobian": 0.5,
        "micro_displacement_px": 3.0,
        "total_map_displacement_px": 8.0,
        "horizontal_catastrophe_guard_passed": True,
        "halo_regression_px": 0.25,
        "finite_in_bounds_passed": True,
    })
    assert len(report["states"]) == 13
    assert report["all_13_pass"] is True
    assert report["diagnostic_only"] is True
    assert report["runtime_authority"] is False
    assert report["micro_rescue"] == {
        "enabled": False, "attempted": False, "accepted": False
    }
    assert report["c2e"] == {
        "enabled": False, "attempted": False, "accepted": False
    }


def test_c2e_diagnostic_marks_missing_and_bad_evidence_separately() -> None:
    report = assess_c2e_diagnostic_roi({
        "held_out_edge_p95_px": 1.01,
        "finite_in_bounds_passed": True,
    })
    by_name = {row["name"]: row for row in report["states"]}
    assert by_name["held_out_edge_p95_px"]["state"] == "fail"
    assert by_name["finite_in_bounds_passed"]["state"] == "pass"
    assert by_name["minimum_jacobian"]["state"] == "unevaluable"
    assert report["pass_count"] == 1
    assert report["fail_count"] == 1
    assert report["unevaluable_count"] == 11
    assert report["all_13_pass"] is False
