"""Small, JSON-safe stage comparators for the S013 M6.2 shadow path."""

from __future__ import annotations

from typing import Any

import numpy as np


def compare_s13_m62_arrays(
    reference: np.ndarray, candidate: np.ndarray, *, label: str, maximum: float,
) -> dict[str, Any]:
    """Compare a stage and retain only the first useful diagnostic coordinate."""

    expected, observed = np.asarray(reference), np.asarray(candidate)
    result: dict[str, Any] = {"label": label, "shape_equal": expected.shape == observed.shape}
    if expected.shape != observed.shape:
        result.update(passed=False, max_abs=None, first_difference=None)
        return result
    nonfinite = int(np.size(observed) - np.isfinite(observed).sum())
    difference = np.abs(expected.astype(np.float64) - observed.astype(np.float64))
    changed = difference > maximum
    indices = np.argwhere(changed)
    first = None if not len(indices) else [int(value) for value in indices[0]]
    result.update(
        passed=not bool(len(indices)) and nonfinite == 0,
        max_abs=float(np.max(difference, initial=0.0)), nonfinite_count=nonfinite,
        differing_value_count=int(len(indices)), first_difference=first,
    )
    return result


def compare_s13_m62_u8(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Byte comparison with distinct shadow and authority gates."""

    report = compare_s13_m62_arrays(reference, candidate, label="final_u8", maximum=0.0)
    if not report["shape_equal"]:
        report.update(differing_pixel_count=None, differing_channel_count=None,
                      p999_abs_dn=None, shadow_gate_passed=False, authority_gate_passed=False)
        return report
    delta = np.abs(np.asarray(reference, np.int16) - np.asarray(candidate, np.int16))
    changed = delta > 0
    report.update(
        differing_pixel_count=int(np.count_nonzero(np.any(changed, axis=2))),
        differing_channel_count=int(np.count_nonzero(changed)),
        max_abs_dn=int(np.max(delta, initial=0)),
        p999_abs_dn=float(np.quantile(delta, 0.999)),
        shadow_gate_passed=bool(np.max(delta, initial=0) <= 1),
        authority_gate_passed=not bool(np.any(changed)),
    )
    return report


def build_s13_m62_equivalence_report(
    *, corrected_roi: list[dict[str, Any]], owner_linear: dict[str, Any],
    b1_pairs: list[dict[str, Any]], final_linear: dict[str, Any], final_u8: dict[str, Any],
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    """Return compact report fields without retaining source or mask arrays."""

    failed_roi = next((item for item in corrected_roi if not item["comparison"]["passed"]), None)
    failed_b1 = next((item for item in b1_pairs if not item["comparison"]["passed"]), None)
    gate = bool(
        not fallback_reason and not failed_roi and owner_linear["passed"]
        and not failed_b1 and final_linear["passed"] and final_u8["authority_gate_passed"]
    )
    return {
        "requested_mode": "shadow_gpu_b1", "resolved_mode": "cpu_authoritative_hybrid",
        "fallback_reason": fallback_reason,
        "corrected_roi_linear_max_abs": max((
            float("inf") if item["comparison"]["max_abs"] is None else item["comparison"]["max_abs"]
            for item in corrected_roi
        ), default=0.0),
        "owner_linear_max_abs": owner_linear.get("max_abs"),
        "first_differing_pair_index": None if failed_b1 is None else failed_b1["pair_index"],
        "b1_pair_linear_max_abs": max((
            float("inf") if item["comparison"]["max_abs"] is None else item["comparison"]["max_abs"]
            for item in b1_pairs
        ), default=0.0),
        "final_linear_max_abs": final_linear.get("max_abs"),
        "final_u8_differing_pixel_count": final_u8.get("differing_pixel_count"),
        "final_u8_differing_channel_count": final_u8.get("differing_channel_count"),
        "final_u8_max_abs_dn": final_u8.get("max_abs_dn"),
        "final_u8_p999_abs_dn": final_u8.get("p999_abs_dn"),
        "byte_identical": final_u8.get("authority_gate_passed", False),
        "authority_gate_passed": gate,
        "corrected_roi": corrected_roi, "owner_linear": owner_linear,
        "b1_pairs": b1_pairs, "final_linear": final_linear,
    }


__all__ = ["compare_s13_m62_arrays", "compare_s13_m62_u8", "build_s13_m62_equivalence_report"]
