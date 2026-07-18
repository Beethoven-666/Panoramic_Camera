from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from panorama_demo.handoff_continuity import (
    HandoffContinuityConfig,
    HandoffDecision,
    HandoffMethod,
    build_handoff_continuity_audit,
    evaluate_handoff_continuity,
    summarize_handoff_methods,
)


def _good_residuals(value: float, count: int = 12) -> np.ndarray:
    return np.full(count, value, dtype=np.float64)


def test_array_builder_accepts_complete_continuous_anchor_and_retains_scalars_only() -> None:
    audit = build_handoff_continuity_audit(
        pair_index=36,
        instance_id="hose_01",
        candidate_anchor_frame_id=35,
        normal_residual_pixels=_good_residuals(0.50),
        centreline_residual_pixels=_good_residuals(0.70),
        direction_delta_degrees=_good_residuals(5.0),
        same_layer_support=np.ones((3, 4), dtype=bool),
        coverage=np.ones((3, 4), dtype=np.uint8),
        delta_e00=_good_residuals(3.0),
        luminance_jump=_good_residuals(1.0),
    )

    assert audit.accepted
    assert audit.decision is HandoffDecision.ANCHOR
    assert audit.same_layer_support_ratio == pytest.approx(1.0)
    assert audit.coverage_ratio == pytest.approx(1.0)
    assert audit.delta_e00_p95 == pytest.approx(3.0)
    assert audit.luminance_jump_p95 == pytest.approx(1.0)
    assert not any(isinstance(value, np.ndarray) for value in asdict(audit).values())
    serialized = audit.as_dict()
    assert serialized["evidence_storage"] == "scalar_only"
    assert serialized["decision"] == "anchor"
    assert all(not isinstance(value, np.ndarray) for value in serialized.values())


def test_array_builder_routes_discontinuous_or_incomplete_anchor_to_fallback() -> None:
    normal = np.array([0.1] * 11 + [3.0], dtype=np.float64)
    audit = build_handoff_continuity_audit(
        pair_index=2,
        instance_id="coupler_3",
        candidate_anchor_frame_id=8,
        normal_residual_pixels=normal,
        centreline_residual_pixels=_good_residuals(1.2),
        direction_delta_degrees=_good_residuals(9.0),
        same_layer_support=np.array([True] * 7 + [False] * 5),
        coverage=np.array([True] * 11 + [False]),
    )

    assert not audit.accepted
    assert audit.decision is HandoffDecision.APAP_FLOW_FALLBACK
    assert "normal_residual_maximum_exceeded" in audit.rejection_reasons
    assert "centreline_residual_p95_exceeded" in audit.rejection_reasons
    assert "direction_delta_p95_exceeded" in audit.rejection_reasons
    assert "same_layer_support_ratio_too_small" in audit.rejection_reasons
    assert "incomplete_anchor_coverage" in audit.rejection_reasons


def test_scalar_evaluator_accepts_renderer_metadata_without_arrays() -> None:
    audit = evaluate_handoff_continuity(
        pair_index=1,
        instance_id=7,
        candidate_anchor_frame_id=12,
        normal_residual_sample_count=24,
        normal_residual_p95_pixels=0.70,
        normal_residual_max_pixels=1.80,
        centreline_residual_sample_count=24,
        centreline_residual_p95_pixels=0.90,
        direction_sample_count=24,
        direction_delta_p95_degrees=7.0,
        same_layer_supported_pixel_count=21,
        same_layer_candidate_pixel_count=24,
        coverage_pixel_count=24,
        coverage_required_pixel_count=24,
    )

    assert audit.accepted
    assert audit.instance_id == "7"
    assert audit.same_layer_support_ratio == pytest.approx(0.875)


def test_scalar_evaluator_returns_a_failed_audit_when_metrics_are_unavailable() -> None:
    audit = evaluate_handoff_continuity(
        pair_index=1,
        instance_id="unmeasured_hose",
        candidate_anchor_frame_id=12,
        normal_residual_sample_count=8,
        normal_residual_p95_pixels=None,
        normal_residual_max_pixels=None,
        centreline_residual_sample_count=8,
        centreline_residual_p95_pixels=None,
        direction_sample_count=8,
        direction_delta_p95_degrees=None,
        same_layer_supported_pixel_count=8,
        same_layer_candidate_pixel_count=8,
        coverage_pixel_count=8,
        coverage_required_pixel_count=8,
    )

    assert not audit.accepted
    assert audit.decision is HandoffDecision.APAP_FLOW_FALLBACK
    assert "normal_residual_metrics_unavailable" in audit.rejection_reasons
    assert "centreline_residual_metric_unavailable" in audit.rejection_reasons
    assert "direction_metric_unavailable" in audit.rejection_reasons


@pytest.mark.parametrize(
    "config",
    (
        {"maximum_normal_residual_p95_pixels": 0.76},
        {"minimum_same_layer_support_ratio": 0.69},
        {"minimum_coverage_ratio": 0.99},
        {"minimum_direction_samples": 7},
        {"unknown": 1},
    ),
)
def test_configuration_cannot_relax_closed_handoff_limits(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HandoffContinuityConfig.from_mapping(config)


def test_array_builder_rejects_unavailable_or_non_boolean_evidence() -> None:
    kwargs = {
        "pair_index": 0,
        "instance_id": "hose",
        "candidate_anchor_frame_id": 1,
        "normal_residual_pixels": _good_residuals(0.1),
        "centreline_residual_pixels": _good_residuals(0.1),
        "direction_delta_degrees": _good_residuals(1.0),
        "same_layer_support": np.ones(12, dtype=bool),
        "coverage": np.ones(12, dtype=bool),
    }
    with pytest.raises(ValueError, match="normal_residual_pixels"):
        build_handoff_continuity_audit(
            **{**kwargs, "normal_residual_pixels": np.array([0.1, np.nan])}
        )
    with pytest.raises(ValueError, match="coverage"):
        build_handoff_continuity_audit(
            **{**kwargs, "coverage": np.array([0.0] * 12, dtype=np.float64)}
        )


def test_summary_counts_only_completed_handoff_methods() -> None:
    summary = summarize_handoff_methods(
        (
            HandoffMethod.ANCHOR,
            "apap",
            {"method": "apap_plus_dense_flow"},
            {"decision": "hard_cut_degraded"},
        )
    )

    assert summary == {"anchor": 1, "apap": 1, "flow_mesh": 1, "hard_cut": 1}
    with pytest.raises(ValueError, match="Unsupported handoff method"):
        summarize_handoff_methods(("apap_flow_fallback",))
