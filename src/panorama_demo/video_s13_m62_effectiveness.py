"""Pure effectiveness classification for the S1.3 M6.2 visual result.

Equivalence to another executor is deliberately outside this module.  This
report answers the narrower question of whether the selected M6 decisions
produced a measured, quality-gated change relative to sealed P2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from .video_s13_blend import S13BlendPlan
from .video_s13_photometric import S13PhotometricSolution


EffectivenessStatus = Literal[
    "applied_global_photometric",
    "applied_component_photometric",
    "applied_blend_only",
    "safe_noop_insufficient_evidence",
    "safe_noop_no_measurable_benefit",
    "reference_fallback",
    "hard_failure",
]
FallbackStatus = Literal["reference_fallback", "hard_failure"]


@dataclass(frozen=True)
class S13M62QualityEvidence:
    """Observed benefit and safety gates used by effectiveness classification."""

    evidence_sufficient: bool = False
    aggregate_residual_improved: bool = False
    macro_residual_improved: bool = False
    worst_pair_nonregression: bool = False
    cut_guard_unchanged: bool = True
    structure_unchanged: bool = True
    blend_residual_improved: bool = False
    protected_active_pixel_count: int = 0


def _is_nonidentity_parameter(gain: Sequence[float], bias: Sequence[float]) -> bool:
    return bool(
        not np.allclose(np.asarray(gain, dtype=np.float64), 1.0, rtol=0.0, atol=1e-12)
        or not np.allclose(np.asarray(bias, dtype=np.float64), 0.0, rtol=0.0, atol=1e-12)
    )


def normalize_s13_m62_b0_reason(reason: str | None) -> str:
    """Normalize implementation-specific B0 fallbacks into report categories."""

    value = "" if reason is None else reason.strip().lower()
    if "unsupported" in value and "cut" in value:
        return "unsupported_cut"
    if "forced" in value or "owner_only" in value:
        return "forced_owner_only"
    if "protected" in value:
        return "protected_structure"
    if "weight" in value or "active" in value:
        return "no_active_weight"
    if "residual" in value:
        return "photometric_residual_too_large"
    if "safe" in value or "evidence" in value or "sample" in value:
        return "insufficient_safe_fraction"
    return "unknown"


def _image_difference(p2_image: np.ndarray, p3_image: np.ndarray) -> dict[str, int | bool]:
    p2 = np.asarray(p2_image)
    p3 = np.asarray(p3_image)
    if p2.shape != p3.shape:
        raise ValueError("S1.3 M6.2 P2 and P3 image shapes disagree")
    if p2.ndim != 3 or p2.shape[2] != 3:
        raise ValueError("S1.3 M6.2 effectiveness requires HxWx3 images")
    if p2.dtype != np.uint8 or p3.dtype != np.uint8:
        raise ValueError("S1.3 M6.2 effectiveness requires uint8 P2 and P3 images")
    absolute = np.abs(p3.astype(np.int16) - p2.astype(np.int16))
    differing_channels = int(np.count_nonzero(absolute))
    differing_pixels = int(np.count_nonzero(np.any(absolute != 0, axis=2)))
    return {
        "differing_pixel_count": differing_pixels,
        "differing_channel_count": differing_channels,
        "max_abs_dn": int(np.max(absolute, initial=0)),
        "equals": differing_channels == 0,
    }


def build_s13_m62_effectiveness_report(
    photometric_solution: S13PhotometricSolution,
    blend_plans: Sequence[S13BlendPlan],
    p2_image: np.ndarray,
    p3_image: np.ndarray,
    *,
    quality: S13M62QualityEvidence = S13M62QualityEvidence(),
    fallback_status: FallbackStatus | None = None,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    """Return a conservative, JSON-ready M6.2 effectiveness report."""

    if quality.protected_active_pixel_count < 0:
        raise ValueError("protected_active_pixel_count must be non-negative")
    if fallback_status is None and fallback_reason is not None:
        raise ValueError("fallback_reason requires fallback_status")

    difference = _image_difference(p2_image, p3_image)
    nonidentity_source_count = sum(
        _is_nonidentity_parameter(item.gain_bgr, item.bias_bgr)
        for item in photometric_solution.source_parameters
    )
    pair_audits: list[dict[str, object]] = []
    active_blend_pair_count = 0
    active_blend_pixel_count = 0
    for plan in blend_plans:
        active_pixels = int(np.count_nonzero(np.asarray(plan.secondary_weight) > 0.0))
        model = plan.transaction.model
        active = model != "B0_owner_only" and active_pixels > 0
        active_blend_pair_count += int(active)
        active_blend_pixel_count += active_pixels if active else 0
        pair_audits.append({
            "pair_index": int(plan.transaction.pair_index),
            "model": model,
            "active_pixel_count": active_pixels if active else 0,
            "b0_reason": (
                normalize_s13_m62_b0_reason(plan.transaction.fallback_reason)
                if not active else None
            ),
        })

    changed = not bool(difference["equals"])
    controlled = quality.cut_guard_unchanged and quality.structure_unchanged
    photometric_benefit = bool(
        quality.aggregate_residual_improved
        and quality.macro_residual_improved
        and quality.worst_pair_nonregression
    )
    blend_benefit = bool(
        quality.blend_residual_improved
        and quality.protected_active_pixel_count == 0
    )
    if fallback_status is not None:
        status: EffectivenessStatus = fallback_status
        selected_reason = fallback_reason or fallback_status
    elif nonidentity_source_count > 0 and changed and controlled and photometric_benefit:
        if "component" in photometric_solution.model_family.lower():
            status = "applied_component_photometric"
        else:
            status = "applied_global_photometric"
        selected_reason = "controlled_photometric_change_with_quality_benefit"
    elif (
        nonidentity_source_count == 0
        and active_blend_pair_count > 0
        and active_blend_pixel_count > 0
        and changed and controlled and blend_benefit
    ):
        status = "applied_blend_only"
        selected_reason = "controlled_active_blend_with_quality_benefit"
    elif not quality.evidence_sufficient:
        status = "safe_noop_insufficient_evidence"
        selected_reason = "insufficient_effectiveness_evidence"
    else:
        status = "safe_noop_no_measurable_benefit"
        selected_reason = "effect_or_quality_benefit_not_measurable"

    return {
        "status": status,
        "selected_reason": selected_reason,
        "selected_photometric_model": photometric_solution.model_family,
        "nonidentity_source_count": int(nonidentity_source_count),
        "active_blend_pair_count": active_blend_pair_count,
        "active_blend_pixel_count": active_blend_pixel_count,
        "protected_active_pixel_count": int(quality.protected_active_pixel_count),
        "p3_equals_p2": bool(difference["equals"]),
        "p3_vs_p2": {
            "differing_pixel_count": difference["differing_pixel_count"],
            "differing_channel_count": difference["differing_channel_count"],
            "max_abs_dn": difference["max_abs_dn"],
        },
        "pair_blend_audits": pair_audits,
    }


__all__ = [
    "EffectivenessStatus",
    "FallbackStatus",
    "S13M62QualityEvidence",
    "build_s13_m62_effectiveness_report",
    "normalize_s13_m62_b0_reason",
]
