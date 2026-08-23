from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_hard_audit import (
    audit_s13_p2_stage,
    audit_s13_seam_family,
    audit_s13_seam_path,
    long_horizontal_structure_catastrophe_guard,
)


def _horizontal(*, lag: int, zero: float = 0.9, support: int = 80):
    return {
        "observed": True,
        "absolute_best_vertical_lag_px": abs(lag),
        "zero_lag_correlation": zero,
        "support_row_count": support,
    }


def test_wave_beam_is_rejected_by_catastrophe_guard() -> None:
    passed, reason, audit = long_horizontal_structure_catastrophe_guard(
        _horizontal(lag=0), _horizontal(lag=2)
    )

    assert passed is False
    assert reason == "horizontal_structure_absolute_lag_catastrophe"
    assert "horizontal_structure_lag_increase_catastrophe" in audit["failures"]


def test_minor_horizontal_change_is_diagnostic_not_a_hard_failure() -> None:
    passed, reason, audit = long_horizontal_structure_catastrophe_guard(
        _horizontal(lag=0, zero=0.90),
        _horizontal(lag=1, zero=0.71),
    )

    assert passed is True
    assert reason is None
    assert audit["zero_lag_correlation_drop"] < 0.20


def test_seam_path_rejects_nonfinite_nonintegral_steps_and_bounds() -> None:
    valid = audit_s13_seam_path(
        np.asarray([4, 5, 5, 4], dtype=np.int32),
        canvas_height=4,
        base_boundary_x=4,
        search_left_x=2,
        search_right_x=6,
    )
    assert valid["passed"] is True

    bad = audit_s13_seam_path(
        np.asarray([4.0, 5.5, np.nan, 9.0]),
        canvas_height=4,
        base_boundary_x=4,
        search_left_x=2,
        search_right_x=6,
    )
    assert bad["passed"] is False
    assert "seam_coordinates_nonfinite" in bad["failures"]

    stepped = audit_s13_seam_path(
        np.asarray([4, 4, 7, 7]),
        canvas_height=4,
        base_boundary_x=4,
        search_left_x=2,
        search_right_x=6,
    )
    assert "seam_outside_search_bounds" in stepped["failures"]
    assert "seam_row_step_invalid" in stepped["failures"]


def test_seam_family_rejects_crossing_and_zero_width_owner() -> None:
    safe = audit_s13_seam_family(
        np.asarray([[3, 3, 3], [6, 6, 6]]),
        assignment_count=3,
        canvas_height=3,
        canvas_width=9,
    )
    assert safe["passed"] is True
    assert safe["minimum_owner_width_px"] == 3

    collapsed = audit_s13_seam_family(
        np.asarray([[3, 3, 3], [3, 4, 2]]),
        assignment_count=3,
        canvas_height=3,
        canvas_width=9,
    )
    assert collapsed["passed"] is False
    assert "seam_family_crossing_or_zero_width_owner" in collapsed["failures"]


def _p2_fixture():
    height, width = 5, 8
    owner_index = np.zeros((height, width), np.int32)
    owner_index[:, 4:] = 1
    valid = np.ones((height, width), bool)
    owner = np.where(owner_index == 0, 10, 11).astype(np.int32)
    u = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width)).copy()
    v = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width)).copy()
    tx = np.where(owner_index == 0, -1, 0).astype(np.int32)
    provenance = {
        "owner_frame_id": owner,
        "owner_source_index": owner_index,
        "source_u": u,
        "source_v": v,
        "valid": valid,
        "geometry_transaction_id": tx,
        "seam_transaction_id": tx.copy(),
        "secondary_frame_id": np.full((height, width), -1, np.int32),
        "secondary_weight": np.zeros((height, width), np.float32),
    }
    seams = np.full((1, height), 4, np.int32)
    return valid, provenance, seams


def test_p2_stage_accepts_complete_owner_only_provenance() -> None:
    valid, provenance, seams = _p2_fixture()
    audit = audit_s13_p2_stage(
        valid_mask=valid,
        pixel_provenance=provenance,
        seams_x_by_row=seams,
        assignment_frame_ids=(10, 11),
        source_sizes=((8, 5), (8, 5)),
        pair_transaction_count=1,
        expected_support_mask=np.ones_like(valid),
    )

    assert audit["passed"] is True
    assert audit["fatal_failures"] == []


def test_p2_stage_rejects_source_bounds_and_owner_transaction_mismatch() -> None:
    valid, provenance, seams = _p2_fixture()
    provenance["source_u"][2, 6] = 20.0
    provenance["geometry_transaction_id"][1, 6] = -1

    audit = audit_s13_p2_stage(
        valid_mask=valid,
        pixel_provenance=provenance,
        seams_x_by_row=seams,
        assignment_frame_ids=(10, 11),
        source_sizes=((8, 5), (8, 5)),
        pair_transaction_count=1,
        expected_support_mask=np.ones_like(valid),
    )

    assert audit["passed"] is False
    assert "valid_source_coordinates_in_bounds" in audit["fatal_failures"]
    assert "transaction_ids_owner_consistent" in audit["fatal_failures"]


def test_p2_stage_rejects_internal_hole_but_allows_external_invalid_region() -> None:
    valid, provenance, seams = _p2_fixture()
    valid[2, 2] = False
    for name in (
        "owner_frame_id", "owner_source_index", "geometry_transaction_id",
        "seam_transaction_id", "secondary_frame_id",
    ):
        provenance[name][2, 2] = -1
    provenance["valid"][2, 2] = False
    provenance["source_u"][2, 2] = np.nan
    provenance["source_v"][2, 2] = np.nan
    audit = audit_s13_p2_stage(
        valid_mask=valid,
        pixel_provenance=provenance,
        seams_x_by_row=seams,
        assignment_frame_ids=(10, 11),
        source_sizes=((8, 5), (8, 5)),
        pair_transaction_count=1,
        expected_support_mask=np.ones_like(valid),
    )
    assert "internal_holes_present" in audit["fatal_failures"]
    assert audit["internal_holes"]["internal_hole_component_count"] == 1

    valid[0, 0] = False
    for name in (
        "owner_frame_id", "owner_source_index", "geometry_transaction_id",
        "seam_transaction_id", "secondary_frame_id",
    ):
        provenance[name][0, 0] = -1
    provenance["valid"][0, 0] = False
    provenance["source_u"][0, 0] = np.nan
    provenance["source_v"][0, 0] = np.nan
    valid[2, 2] = True
    provenance["owner_frame_id"][2, 2] = 10
    provenance["owner_source_index"][2, 2] = 0
    provenance["geometry_transaction_id"][2, 2] = -1
    provenance["seam_transaction_id"][2, 2] = -1
    provenance["valid"][2, 2] = True
    provenance["source_u"][2, 2] = 2.0
    provenance["source_v"][2, 2] = 2.0
    external = audit_s13_p2_stage(
        valid_mask=valid,
        pixel_provenance=provenance,
        seams_x_by_row=seams,
        assignment_frame_ids=(10, 11),
        source_sizes=((8, 5), (8, 5)),
        pair_transaction_count=1,
        expected_support_mask=np.ones_like(valid),
    )
    assert external["internal_holes"]["passed"] is True


def test_v6_r2_hard_audit_requires_component_correction_field() -> None:
    valid, provenance, seams = _p2_fixture()
    audit = audit_s13_p2_stage(
        valid_mask=valid,
        pixel_provenance=provenance,
        seams_x_by_row=seams,
        assignment_frame_ids=(10, 11),
        source_sizes=((8, 5), (8, 5)),
        pair_transaction_count=1,
        expected_support_mask=np.ones_like(valid),
        require_component_correction_fields=True,
    )

    assert audit["passed"] is False
    assert "pixel_provenance_fields_missing" in audit["fatal_failures"]
