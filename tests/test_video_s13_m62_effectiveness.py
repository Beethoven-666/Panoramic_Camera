from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_s13_blend import S13BlendPlan, S13BlendTransaction
from panorama_demo.video_s13_m62_effectiveness import (
    FallbackStatus,
    S13M62QualityEvidence,
    build_s13_m62_effectiveness_report,
    normalize_s13_m62_b0_reason,
)
from panorama_demo.video_s13_photometric import (
    PHOTOMETRIC_BOUNDS_SCHEMA,
    PHOTOMETRIC_SCHEMA,
    S13PhotometricSolution,
    S13SourcePhotometricParameters,
)


def _solution(model: str = "Q0_identity", *, nonidentity: bool = False) -> S13PhotometricSolution:
    parameter = S13SourcePhotometricParameters(
        source_index=0,
        frame_id=10,
        model=model,
        gain_bgr=(1.02, 1.02, 1.02) if nonidentity else (1.0, 1.0, 1.0),
        bias_bgr=(0.0, 0.0, 0.0),
        component_id=0,
        evidence_weight=20.0,
        fallback_reason=None if nonidentity else "identity_candidate",
    )
    return S13PhotometricSolution(
        PHOTOMETRIC_SCHEMA,
        PHOTOMETRIC_BOUNDS_SCHEMA,
        model,
        (parameter,),
        (),
        {},
        {},
        1,
        0.75,
        False,
    )


def _blend(
    model: str = "B0_owner_only", *, active_pixels: int = 0, reason: str | None = None
) -> S13BlendPlan:
    weight = np.zeros((2, 3), np.float32)
    weight.flat[:active_pixels] = 0.25
    transaction = S13BlendTransaction(
        pair_index=4,
        transaction_id=4,
        parent_p2_pair_transaction_sha256="a" * 64,
        model=model,
        total_width_px=2 if active_pixels else 0,
        pyramid_levels=0,
        protected_pixel_count=0,
        blended_pixel_count=active_pixels,
        hard_audit_passed=True,
        fallback_reason=reason,
        candidate_audits=(),
    )
    mask = np.zeros((2, 3), bool)
    return S13BlendPlan(transaction, weight, mask, mask)


def _images() -> tuple[np.ndarray, np.ndarray]:
    p2 = np.zeros((2, 3, 3), np.uint8)
    p3 = p2.copy()
    p3[0, 0] = (1, 2, 0)
    p3[1, 2, 2] = 7
    return p2, p3


def test_global_photometric_requires_controlled_difference_and_all_benefit_gates() -> None:
    p2, p3 = _images()
    report = build_s13_m62_effectiveness_report(
        _solution("Q1_scalar_luminance_gain", nonidentity=True),
        (_blend(),),
        p2,
        p3,
        quality=S13M62QualityEvidence(
            evidence_sufficient=True,
            aggregate_residual_improved=True,
            macro_residual_improved=True,
            worst_pair_nonregression=True,
        ),
    )

    assert report["status"] == "applied_global_photometric"
    assert report["nonidentity_source_count"] == 1
    assert report["p3_equals_p2"] is False
    assert report["p3_vs_p2"] == {
        "differing_pixel_count": 2,
        "differing_channel_count": 3,
        "max_abs_dn": 7,
    }


def test_component_model_is_reported_separately() -> None:
    p2, p3 = _images()
    report = build_s13_m62_effectiveness_report(
        _solution("Q4c_component_boundary_anchored_scalar_gain", nonidentity=True),
        (),
        p2,
        p3,
        quality=S13M62QualityEvidence(
            evidence_sufficient=True,
            aggregate_residual_improved=True,
            macro_residual_improved=True,
            worst_pair_nonregression=True,
        ),
    )
    assert report["status"] == "applied_component_photometric"


def test_blend_only_requires_actual_weighted_pixels_and_benefit() -> None:
    p2, p3 = _images()
    report = build_s13_m62_effectiveness_report(
        _solution(),
        (_blend("B1_narrow_feather", active_pixels=3),),
        p2,
        p3,
        quality=S13M62QualityEvidence(
            evidence_sufficient=True,
            blend_residual_improved=True,
        ),
    )
    assert report["status"] == "applied_blend_only"
    assert report["active_blend_pair_count"] == 1
    assert report["active_blend_pixel_count"] == 3


@pytest.mark.parametrize(
    ("quality", "expected"),
    [
        (S13M62QualityEvidence(), "safe_noop_insufficient_evidence"),
        (
            S13M62QualityEvidence(
                evidence_sufficient=True,
                aggregate_residual_improved=True,
                macro_residual_improved=False,
                worst_pair_nonregression=True,
            ),
            "safe_noop_no_measurable_benefit",
        ),
    ],
)
def test_unproven_nonidentity_is_never_reported_as_applied(
    quality: S13M62QualityEvidence, expected: str
) -> None:
    p2, p3 = _images()
    report = build_s13_m62_effectiveness_report(
        _solution("Q1_scalar_luminance_gain", nonidentity=True), (), p2, p3, quality=quality
    )
    assert report["status"] == expected


def test_exact_p2_return_is_no_measurable_benefit_even_with_quality_flags() -> None:
    p2 = np.zeros((1, 2, 3), np.uint8)
    report = build_s13_m62_effectiveness_report(
        _solution("Q1_scalar_luminance_gain", nonidentity=True),
        (),
        p2,
        p2.copy(),
        quality=S13M62QualityEvidence(
            evidence_sufficient=True,
            aggregate_residual_improved=True,
            macro_residual_improved=True,
            worst_pair_nonregression=True,
        ),
    )
    assert report["status"] == "safe_noop_no_measurable_benefit"
    assert report["p3_equals_p2"] is True
    assert report["p3_vs_p2"] == {
        "differing_pixel_count": 0,
        "differing_channel_count": 0,
        "max_abs_dn": 0,
    }


@pytest.mark.parametrize(
    ("fallback_status", "fallback_reason"),
    [
        ("reference_fallback", "gpu_authority_gate_failed"),
        ("hard_failure", "invalid_owner_topology"),
    ],
)
def test_fallback_status_has_precedence_without_hiding_diff_measurement(
    fallback_status: FallbackStatus, fallback_reason: str
) -> None:
    p2, p3 = _images()
    report = build_s13_m62_effectiveness_report(
        _solution("Q1_scalar_luminance_gain", nonidentity=True),
        (),
        p2,
        p3,
        fallback_status=fallback_status,
        fallback_reason=fallback_reason,
    )
    assert report["status"] == fallback_status
    assert report["selected_reason"] == fallback_reason
    assert report["p3_vs_p2"]["differing_pixel_count"] == 2


def test_b0_reasons_are_normalized_per_pair() -> None:
    p2 = np.zeros((1, 1, 3), np.uint8)
    report = build_s13_m62_effectiveness_report(
        _solution(),
        (_blend(reason="forced_owner_only_rebuild"),),
        p2,
        p2.copy(),
    )
    assert report["pair_blend_audits"] == [{
        "pair_index": 4,
        "model": "B0_owner_only",
        "active_pixel_count": 0,
        "b0_reason": "forced_owner_only",
    }]
    assert normalize_s13_m62_b0_reason("global_protected_structure_overlap") == "protected_structure"
    assert normalize_s13_m62_b0_reason("residual_above_limit") == "photometric_residual_too_large"
    assert normalize_s13_m62_b0_reason(None) == "unknown"


def test_invalid_image_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="uint8"):
        build_s13_m62_effectiveness_report(
            _solution(), (), np.zeros((1, 1, 3), np.float32), np.zeros((1, 1, 3), np.float32)
        )
