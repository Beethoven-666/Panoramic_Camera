"""JSON-ready report assembly for effective M6.2 runs."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .video_s13_m62_effectiveness import (
    S13M62QualityEvidence, build_s13_m62_effectiveness_report,
)
from .video_s13_m62_evidence import pair_evidence_document


def _compact_metrics(value: object, *, detailed: bool) -> object:
    if not isinstance(value, Mapping):
        return value
    result = dict(value)
    pairs = result.get("pair_metrics")
    if not detailed and isinstance(pairs, (list, tuple)):
        result["pair_metrics"] = sorted(
            (dict(item) for item in pairs if isinstance(item, Mapping)),
            key=lambda item: float(item.get("p95_linear", -1.0)),
            reverse=True,
        )[:10]
    return result


def _compact_candidate_audits(values: object, *, detailed: bool) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for value in values if isinstance(values, (list, tuple)) else ():
        if not isinstance(value, Mapping):
            continue
        row = dict(value)
        for key in ("train", "heldout", "before", "after"):
            if key in row:
                row[key] = _compact_metrics(row[key], detailed=detailed)
        components = row.get("components")
        if isinstance(components, (list, tuple)):
            row["components"] = [
                {
                    **dict(component),
                    "before": _compact_metrics(component.get("before"), detailed=detailed),
                    "after": _compact_metrics(component.get("after"), detailed=detailed),
                }
                for component in components if isinstance(component, Mapping)
            ]
        result.append(row)
    return result


def _sample_baseline_p95(sample: object) -> float:
    left = np.asarray(sample.heldout_left_rgb_linear)
    right = np.asarray(sample.heldout_right_rgb_linear)
    return (
        -1.0
        if not len(left)
        else float(np.quantile(np.linalg.norm(left - right, axis=1), 0.95))
    )


def _selected_audit(plan: object) -> Mapping[str, object]:
    return next(
        (item for item in plan.photometric_solution.candidate_audits if item.get("selected") is True),
        {},
    )


def build_s13_m62_report(
    plan: object,
    *, p2_image: np.ndarray, p3_image: np.ndarray,
    gpu_equivalence: Mapping[str, object], performance: Mapping[str, object],
) -> dict[str, object]:
    selected = _selected_audit(plan)
    baseline = next(
        (item for item in plan.photometric_solution.candidate_audits if item.get("model") == "Q0_identity"),
        {},
    )
    selected_metrics = selected.get("heldout", {})
    baseline_metrics = baseline.get("heldout", {})
    if not isinstance(selected_metrics, Mapping):
        selected_metrics = {}
    if not isinstance(baseline_metrics, Mapping):
        baseline_metrics = {}
    nonidentity = plan.photometric_solution.model_family != "Q0_identity"
    aggregate_improved = bool(
        nonidentity
        and (
            (
                selected.get("aggregate_median_before") is not None
                and selected.get("aggregate_median_after") is not None
                and float(selected["aggregate_median_after"]) < float(selected["aggregate_median_before"])
            )
            or (
                baseline_metrics.get("aggregate_median_linear") is not None
                and selected_metrics.get("aggregate_median_linear") is not None
                and float(selected_metrics["aggregate_median_linear"])
                < float(baseline_metrics["aggregate_median_linear"])
            )
        )
    )
    macro_improved = bool(
        nonidentity
        and (
            (
                selected.get("macro_pair_p95_before") is not None
                and selected.get("macro_pair_p95_after") is not None
                and float(selected["macro_pair_p95_after"]) < float(selected["macro_pair_p95_before"])
            )
            or (
                baseline_metrics.get("macro_pair_p95_linear") is not None
                and selected_metrics.get("macro_pair_p95_linear") is not None
                and float(selected_metrics["macro_pair_p95_linear"])
                < float(baseline_metrics["macro_pair_p95_linear"])
            )
        )
    )
    worst_nonreg = bool(
        nonidentity
        and "worst_pair_regression" not in selected.get("rejection_reasons", ())
    )
    protected_active = sum(
        int(np.count_nonzero((item.secondary_weight > 0.0) & item.protected_mask))
        for item in plan.blend_plans
    )
    active_transactions = [
        candidate
        for item in plan.blend_plans
        if np.any(item.secondary_weight > 0.0)
        for candidate in item.transaction.candidate_audits
        if candidate.get("selected") is True
    ]
    blend_improved = bool(active_transactions) and all(
        isinstance(item.get("immediate_seam_benefit_linear"), (int, float))
        and float(item["immediate_seam_benefit_linear"]) > 0.0
        for item in active_transactions
    )
    evidence_sufficient = bool(
        not plan.evidence.unsupported_cut_pair_indices
        or plan.photometric_solution.model_family != "Q0_identity"
    )
    effectiveness = build_s13_m62_effectiveness_report(
        plan.photometric_solution, plan.blend_plans, p2_image, p3_image,
        quality=S13M62QualityEvidence(
            evidence_sufficient=evidence_sufficient,
            aggregate_residual_improved=aggregate_improved,
            macro_residual_improved=macro_improved,
            worst_pair_nonregression=worst_nonreg,
            blend_residual_improved=blend_improved,
            protected_active_pixel_count=protected_active,
        ),
        fallback_status=(
            "reference_fallback" if gpu_equivalence.get("fallback_reason") is not None
            and performance.get("execution_mode") == "candidate_single_pass" else None
        ),
        fallback_reason=(
            str(gpu_equivalence.get("fallback_reason"))
            if gpu_equivalence.get("fallback_reason") is not None
            and performance.get("execution_mode") == "candidate_single_pass" else None
        ),
    )
    component_count = int(plan.evidence.component_count)
    m63 = plan.m63
    m63_quality = () if m63 is None else m63.quality_rows
    detailed = bool(plan.retain_runtime_details)
    compact_audits = _compact_candidate_audits(
        plan.photometric_solution.candidate_audits, detailed=detailed
    )
    compact_pair_indices = {
        item.pair_index
        for item in sorted(
            plan.evidence.adjacent_samples,
            key=_sample_baseline_p95,
            reverse=True,
        )[:10]
    }
    if m63 is not None:
        compact_pair_indices.update(m63.quality_cut_pair_indices)
    cut_guard_changed = 0
    if m63 is not None:
        difference = np.any(np.asarray(p2_image) != np.asarray(p3_image), axis=2)
        width = int(m63.audit.get("cut_guard_width_px", 12))
        by_pair = {item.pair_index: item for item in plan.p2.replay_pairs}
        cut_guard = np.zeros(difference.shape, bool)
        for pair_index in m63.quality_cut_pair_indices:
            pair = by_pair[pair_index]
            for row, seam_x in enumerate(np.asarray(pair.seam_x_by_row, np.int32)):
                x0 = max(0, int(seam_x) - width)
                x1 = min(difference.shape[1], int(seam_x) + width + 1)
                cut_guard[row, x0:x1] = True
        cut_guard_changed = int(np.count_nonzero(difference & cut_guard))
    ql_rows = (
        [] if m63 is None else list(dict(m63.audit.get("ql", {})).get("pairs", ()))
    )
    if not detailed:
        ql_rows = [row for row in ql_rows if row.get("accepted") is True]
    blend_rows = list(effectiveness["pair_blend_audits"])
    b0_reason_counts: dict[str, int] = {}
    for row in blend_rows:
        reason = row.get("b0_reason")
        if reason is not None:
            key = str(reason)
            b0_reason_counts[key] = b0_reason_counts.get(key, 0) + 1
    if not detailed:
        blend_rows = [row for row in blend_rows if row.get("model") != "B0_owner_only"][:10]
    return {
        "schema": "gemini305-video-s13-m62-report/v3",
        "effectiveness": effectiveness,
        "cpu_oracle": {
            "enabled": performance.get("execution_mode") != "candidate_single_pass",
            "photometric_model": plan.photometric_solution.model_family,
            "blend_plan_count": len(plan.blend_plans),
            "B0_count": sum(item.transaction.model == "B0_owner_only" for item in plan.blend_plans),
            "B1_count": sum(item.transaction.model == "B1_narrow_feather" for item in plan.blend_plans),
            "B2_count": sum(item.transaction.model == "B2_safe_masked_multiband" for item in plan.blend_plans),
            "published_pixel_authority": "cpu",
        },
        "gpu_equivalence": dict(gpu_equivalence),
        "photometric": {
            "model_family": plan.photometric_solution.model_family,
            "source_count": len(plan.photometric_solution.source_parameters),
            "candidate_audits": compact_audits,
            "all_sources_parameterized": bool(selected.get("all_sources_parameterized", True)),
            "raw_fallback_source_count": int(selected.get("raw_fallback_source_count", 0)),
            "out_of_bounds_source_count": int(selected.get("out_of_bounds_source_count", 0)),
            "nonfinite_source_count": int(selected.get("nonfinite_source_count", 0)),
            "partial_identity_fallback_detected": bool(
                selected.get("partial_identity_fallback_detected", False)
            ),
        },
        "evidence": {
            "tier_a_sample_count": plan.evidence.tier_a_sample_count,
            "tier_b_sample_count": plan.evidence.tier_b_sample_count,
            "adjacent_edge_count": plan.evidence.adjacent_edge_count,
            "bridge_edge_count": plan.evidence.bridge_edge_count,
            "component_count": component_count,
            "unsupported_cut_pair_indices": list(plan.evidence.unsupported_cut_pair_indices),
            "evidence_graph_connected": component_count == 1,
            "evidence_component_count": component_count,
            "eligible_adjacent_edge_count": int(sum(
                item.edge_kind == "adjacent" and item.edge_eligible is True
                for item in plan.evidence.solve_samples
            )),
            "eligible_bridge_edge_count": int(sum(
                item.edge_kind != "adjacent" and item.edge_eligible is True
                for item in plan.evidence.solve_samples
            )),
            "quality_cut_edge_count": len(()) if m63 is None else len(m63.quality_cut_pair_indices),
            "quality_cut_pair_indices": [] if m63 is None else list(m63.quality_cut_pair_indices),
            "soft_downweighted_pair_indices": [] if m63 is None else list(m63.soft_downweighted_pair_indices),
            "unsupported_sample_edge_count": len(plan.evidence.unsupported_cut_pair_indices),
            "quality": [
                {
                    "pair_index": item.pair_index,
                    "classification": item.classification,
                    "heldout_sample_count": item.heldout_sample_count,
                    "baseline_median_linear": item.baseline_median_linear,
                    "baseline_p95_linear": item.baseline_p95_linear,
                    "log_luminance_mad": item.log_luminance_mad,
                    "inlier_fraction": item.inlier_fraction,
                    "reasons": list(item.reasons),
                }
                for item in m63_quality
            ],
            "pairs": [
                pair_evidence_document(item)
                for item in plan.evidence.adjacent_samples
                if detailed or item.pair_index in compact_pair_indices
            ],
            "bridges": [
                pair_evidence_document(item)
                for item in sorted(
                    (value for value in plan.evidence.solve_samples if value.edge_kind != "adjacent"),
                    key=_sample_baseline_p95,
                    reverse=True,
                )[:(None if detailed else 10)]
            ],
        },
        "selection": {
            "candidate_audits": compact_audits,
            "macro_before_after": {
                "before": selected.get("macro_pair_p95_before"),
                "after": selected.get("macro_pair_p95_after"),
            },
            "worst_pair_before_after": {
                "before": selected.get("worst_pair_p95_before"),
                "after": selected.get("worst_pair_p95_after"),
            },
            **({} if m63 is None else dict(m63.audit)),
        },
        "component_shadow": dict(plan.component_shadow),
        "q1r": {
            "raw_gain_minimum": selected.get("raw_gain_minimum"),
            "raw_gain_maximum": selected.get("raw_gain_maximum"),
            "selected_gain_minimum": selected.get("selected_gain_minimum"),
            "selected_gain_maximum": selected.get("selected_gain_maximum"),
            "source_fallback_count": int(selected.get("raw_fallback_source_count", 0)),
        },
        "q4c": {
            "component_count": int(selected.get("component_count", 0)),
            "accepted_components": int(selected.get("accepted_component_count", 0)),
            "anchor_sources": list(selected.get("anchor_sources", ())),
            "cut_guard_changed_pixels": cut_guard_changed,
        },
        "ql": {
            "active_pair_count": (
                0 if m63 is None else int(dict(m63.audit.get("ql", {})).get("active_pair_count", 0))
            ),
            "authority": "shadow" if m63 is not None else "disabled",
            "overlap_normalization": "log_domain_clamp",
            "pairs": ql_rows,
        },
        "qt": {
            "selected_gain": 1.0,
            "enabled": False if m63 is None else bool(m63.audit.get("qt_enabled", False)),
        },
        "same_p2_reference": {
            "old_commit": "c3aaa7ddbaf8aac5e00cded71ba9e4f1689b822a",
            "comparison_authority": "benchmarks/S013_M6_3_same_p2",
        },
        "blend": {
            "models": [item.transaction.model for item in plan.blend_plans],
            "active_pair_count": sum(bool(np.any(item.secondary_weight > 0.0)) for item in plan.blend_plans),
            "pairs": blend_rows,
            "b0_reason_counts": b0_reason_counts,
        },
        "safety": {
            "invalid_nonzero": int(np.count_nonzero(p3_image[~plan.valid_mask])),
            "protected_active": protected_active,
            "gpu_evaluated": bool(gpu_equivalence.get("gpu_evaluated", False)),
            "source_partial_fallback": bool(selected.get("partial_identity_fallback_detected", False)),
            "cut_guard_changed_pixels": cut_guard_changed,
        },
        "performance": dict(performance),
        "p3_relation_to_p2": dict(effectiveness["p3_vs_p2"]),
    }


__all__ = ["build_s13_m62_report"]
