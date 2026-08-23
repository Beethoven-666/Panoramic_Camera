"""Diagnostic-only C2E gate measurements for the S1.3 M5.1-r3 study.

This module does not estimate, construct, select, or apply a C2E map.  It only
classifies already measured ROI evidence against the thirteen proposed gates.
Missing or non-finite evidence is unevaluable, never a pass.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


_NUMERIC_GATES = (
    ("held_out_edge_p95_px", "maximum", 1.0),
    ("absolute_improvement_px", "minimum", 0.5),
    ("relative_improvement_fraction", "minimum", 0.30),
    ("forward_reverse_lag_discrepancy_px", "maximum", 0.5),
    ("best_second_uniqueness_fraction", "minimum", 0.10),
    ("orientation_difference_degrees", "maximum", 10.0),
    ("support_retention", "minimum", 0.98),
    ("minimum_jacobian", "minimum", 0.5),
    ("micro_displacement_px", "maximum", 3.0),
    ("total_map_displacement_px", "maximum", 8.0),
)
_BOOLEAN_GATES = (
    "horizontal_catastrophe_guard_passed",
    "finite_in_bounds_passed",
)


def assess_c2e_diagnostic_roi(measurements: Mapping[str, object]) -> dict[str, object]:
    """Classify the proposed 13 C2E gates without authorizing C2E."""

    rows: list[dict[str, object]] = []
    for name, direction, threshold in _NUMERIC_GATES:
        raw = measurements.get(name)
        if not isinstance(raw, (int, float, np.integer, np.floating)) or not np.isfinite(raw):
            state = "unevaluable"
            value = None
        else:
            value = float(raw)
            passed = value <= threshold if direction == "maximum" else value >= threshold
            state = "pass" if passed else "fail"
        rows.append({
            "name": name, "state": state, "value": value,
            "direction": direction, "threshold": threshold,
        })

    halo = measurements.get("halo_regression_px")
    if not isinstance(halo, (int, float, np.integer, np.floating)) or not np.isfinite(halo):
        halo_state, halo_value = "unevaluable", None
    else:
        halo_value = float(halo)
        halo_state = "pass" if halo_value <= 0.25 else "fail"
    rows.append({
        "name": "halo_regression_px", "state": halo_state,
        "value": halo_value, "direction": "maximum", "threshold": 0.25,
    })
    for name in _BOOLEAN_GATES:
        raw = measurements.get(name)
        state = "pass" if raw is True else "fail" if raw is False else "unevaluable"
        rows.append({
            "name": name, "state": state,
            "value": raw if isinstance(raw, bool) else None,
            "direction": "must_be_true", "threshold": True,
        })
    if len(rows) != 13:
        raise AssertionError("C2E diagnostic must contain exactly thirteen gates")
    pass_count = sum(row["state"] == "pass" for row in rows)
    fail_count = sum(row["state"] == "fail" for row in rows)
    unevaluable_count = sum(row["state"] == "unevaluable" for row in rows)
    disabled = {"enabled": False, "attempted": False, "accepted": False}
    return {
        "schema": "gemini305-video-s13-m51-r3-c2e-diagnostic-roi/v1",
        "diagnostic_only": True,
        "runtime_authority": False,
        "applied_to_p2": False,
        "states": rows,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "unevaluable_count": unevaluable_count,
        "all_13_pass": pass_count == 13,
        "micro_rescue": dict(disabled),
        "c2e": dict(disabled),
    }


def build_c2e_diagnostic_from_pair_transaction(
    transaction: Mapping[str, object],
) -> dict[str, object]:
    """Derive a non-authoritative 13-state report from a real v6 pair audit.

    Existing M5 map and edge audits can prove only a subset of the proposed
    C2E gates.  Candidate-delta improvement, reverse-lag, micro displacement,
    and halo evidence remain unevaluable because C2E is intentionally absent.
    """

    transaction_id = transaction.get("transaction_id")
    if not isinstance(transaction_id, str) or not transaction_id.startswith("m5-pair-"):
        raise ValueError("C2E diagnostic requires one M5 pair transaction")
    selected_model = transaction.get("selected_geometry_model")
    evaluations = transaction.get("candidate_evaluations")
    selected_evaluation: Mapping[str, object] | None = None
    if isinstance(evaluations, list):
        for value in evaluations:
            if not isinstance(value, Mapping):
                continue
            if value.get("geometry_model") == selected_model and value.get(
                "hard_gate_passed"
            ) is True:
                selected_evaluation = value
                break

    measurements: dict[str, object] = {}
    edge = transaction.get("edge_registration")
    if isinstance(edge, Mapping) and edge.get("evaluable") is True:
        # This is real selected-map ROI evidence.  It may assess the absolute
        # residual gate, but cannot claim C2E improvement over that map.
        measurements["held_out_edge_p95_px"] = edge.get(
            "p95_supported_abs_lag_px"
        )
        measurements["best_second_uniqueness_fraction"] = edge.get(
            "minimum_uniqueness_margin"
        )
        measurements["orientation_difference_degrees"] = edge.get(
            "maximum_orientation_difference_degrees"
        )
    if selected_evaluation is not None:
        horizontal = selected_evaluation.get("horizontal_hard_audit")
        if isinstance(horizontal, Mapping) and isinstance(horizontal.get("passed"), bool):
            measurements["horizontal_catastrophe_guard_passed"] = horizontal["passed"]
        failures = selected_evaluation.get("hard_gate_failures")
        if isinstance(failures, list):
            finite_failures = {
                "geometry_map_nonfinite", "geometry_source_out_of_bounds"
            }
            measurements["finite_in_bounds_passed"] = not any(
                value in finite_failures for value in failures
            )
    report = assess_c2e_diagnostic_roi(measurements)
    return {
        **report,
        "source_pair_transaction_id": transaction_id,
        "source_pair_transaction_schema": transaction.get("schema"),
        "source_selected_geometry_model": selected_model,
        "source_measurements": measurements,
        "applied_to_p2": False,
    }


__all__ = [
    "assess_c2e_diagnostic_roi", "build_c2e_diagnostic_from_pair_transaction",
]
