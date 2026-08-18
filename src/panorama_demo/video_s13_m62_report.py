"""JSON-ready report assembly for effective M6.2 runs."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .video_s13_m62_effectiveness import (
    S13M62QualityEvidence, build_s13_m62_effectiveness_report,
)
from .video_s13_m62_evidence import pair_evidence_document


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
    nonidentity = plan.photometric_solution.model_family != "Q0_identity"
    aggregate_improved = bool(
        nonidentity
        and selected.get("aggregate_median_before") is not None
        and selected.get("aggregate_median_after") is not None
        and float(selected["aggregate_median_after"]) < float(selected["aggregate_median_before"])
    )
    macro_improved = bool(
        nonidentity
        and selected.get("macro_pair_p95_before") is not None
        and selected.get("macro_pair_p95_after") is not None
        and float(selected["macro_pair_p95_after"]) < float(selected["macro_pair_p95_before"])
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
            "candidate_audits": list(plan.photometric_solution.candidate_audits),
        },
        "evidence": {
            "tier_a_sample_count": plan.evidence.tier_a_sample_count,
            "tier_b_sample_count": plan.evidence.tier_b_sample_count,
            "adjacent_edge_count": plan.evidence.adjacent_edge_count,
            "bridge_edge_count": plan.evidence.bridge_edge_count,
            "component_count": component_count,
            "unsupported_cut_pair_indices": list(plan.evidence.unsupported_cut_pair_indices),
            "pairs": [pair_evidence_document(item) for item in plan.evidence.adjacent_samples],
            "bridges": [pair_evidence_document(item) for item in plan.evidence.solve_samples if item.edge_kind != "adjacent"],
        },
        "selection": {
            "candidate_audits": list(plan.photometric_solution.candidate_audits),
            "macro_before_after": {
                "before": selected.get("macro_pair_p95_before"),
                "after": selected.get("macro_pair_p95_after"),
            },
            "worst_pair_before_after": {
                "before": selected.get("worst_pair_p95_before"),
                "after": selected.get("worst_pair_p95_after"),
            },
        },
        "component_shadow": dict(plan.component_shadow),
        "blend": {
            "models": [item.transaction.model for item in plan.blend_plans],
            "active_pair_count": sum(bool(np.any(item.secondary_weight > 0.0)) for item in plan.blend_plans),
            "pairs": effectiveness["pair_blend_audits"],
        },
        "safety": {
            "invalid_nonzero": int(np.count_nonzero(p3_image[~plan.valid_mask])),
            "protected_active": protected_active,
            "gpu_evaluated": bool(gpu_equivalence.get("gpu_evaluated", False)),
        },
        "performance": dict(performance),
        "p3_relation_to_p2": dict(effectiveness["p3_vs_p2"]),
    }


__all__ = ["build_s13_m62_report"]
