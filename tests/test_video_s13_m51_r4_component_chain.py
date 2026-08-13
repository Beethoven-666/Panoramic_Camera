from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from panorama_demo.video_s13_m51_r2 import S13M51R4Config
from panorama_demo.video_s13_m51_r4_component_chain import (
    S13ComponentApplicationSegment,
    S13ComponentSegmentCandidate,
    S13ComponentSolverError,
    S13EdgeComponentObservation,
    S13SourceComponentCorrection,
    aggregate_s13_runtime_edge_traces,
    audit_s13_component_candidate_map_safety,
    audit_s13_obligation_coverage,
    build_s13_edge_component_chains,
    build_s13_source_component_correction,
    canonical_s13_normal_search_lags,
    canonical_s13_support_sha256,
    compose_s13_source_component_deltas,
    evaluate_s13_component_candidate_quality,
    freeze_s13_baseline_c2e_obligations,
    freeze_s13_source_correction_registry,
    locate_s13_dependency_failure_subset,
    make_s13_edge_component_observation,
    plan_s13_failed_segment_split,
    propagate_s13_edge_component_evidence,
    s13_component_segment_budget_priority,
    select_s13_component_patch_set,
    select_s13_component_segment_split_pair,
    solve_s13_component_segment_source_offsets,
    split_s13_component_application_segment_at_pair,
    split_s13_edge_component_chain,
    source_map_oracle_from_arrays,
    trace_s13_owner_only_component_edge,
)
from panorama_demo.video_s13_m5 import _select_s13_component_gain_candidate
from panorama_demo.video_s13_quality import pair_edge_registration_metrics


@dataclass(frozen=True)
class _Config:
    maximum_pair_gap: int = 1
    minimum_chain_pair_count: int = 2
    minimum_component_y_overlap_fraction: float = 0.5
    maximum_component_normal_difference_degrees: float = 10.0
    maximum_component_endpoint_distance_px: float = 8.0
    maximum_predicted_y_disagreement_px: float = 6.0
    minimum_component_match_margin_fraction: float = 0.1
    minimum_signed_gradient_agreement: float = 0.1
    component_match_weights: dict[str, float] | None = None
    source_offset_regularization: float = 0.05
    solver_huber_delta_px: float = 0.75
    maximum_solver_condition_number: float = 1_000_000.0
    maximum_normalized_solver_residual: float = 3.0
    maximum_source_normal_offset_px: float = 3.0
    normal_search_step_px: float = 0.5
    edge_core_radius_px: int = 2
    normal_taper_radius_px: int = 8
    endpoint_taper_px: int = 12
    maximum_post_edge_p95_px: float = 1.0
    maximum_post_edge_step_px: float = 1.5
    minimum_resolved_improvement_px: float = 0.10
    minimum_absolute_improvement_px: float = 0.5
    minimum_relative_improvement_fraction: float = 0.30
    minimum_break_length_reduction_fraction: float = 0.50
    maximum_non_target_p95_regression_px: float = 0.10
    maximum_non_target_step_regression_px: float = 0.25
    minimum_application_segment_pair_count: int = 1
    split_on_solver_outlier: bool = True
    allow_partial_application: bool = True
    maximum_segment_split_depth: int = 4
    maximum_exact_conflict_group_nodes: int = 12

    def __post_init__(self) -> None:
        if self.component_match_weights is None:
            object.__setattr__(self, "component_match_weights", {
                "y_overlap": 0.30,
                "predicted_y": 0.25,
                "endpoint_distance": 0.20,
                "correlation": 0.15,
                "uniqueness": 0.10,
            })


def _observation(pair: int, component: int, lag: float, *, y0: int = 10) -> S13EdgeComponentObservation:
    return S13EdgeComponentObservation(
        pair_index=pair,
        component_id=component,
        source_indices=(pair, pair + 1),
        global_bbox_xyxy=(20 + pair * 2, y0, 31 + pair * 2, y0 + 20),
        block_indices=(0,),
        normal_x=0.0,
        normal_y=1.0,
        fitted_line_offset=float(y0 + 10),
        forward_best_lag_px=lag,
        reverse_best_lag_px=-lag,
        correlation=0.95,
        uniqueness_fraction=0.4,
        orientation_difference_degrees=0.0,
        mask_sha256=f"{pair + component + 1:064x}",
        evidence_state="actionable" if abs(lag) > 1.0 else "safe_anchor",
        exclusion_reasons=(),
        signed_gradient_polarity=1.0,
        forward_valid_samples=20,
        reverse_valid_samples=20,
        reference_support_samples=20,
    )


def test_canonical_lags_are_exactly_thirteen_half_pixel_states() -> None:
    lags = canonical_s13_normal_search_lags(-3.0, 3.0, 0.5)
    assert lags.dtype == np.float64
    assert lags.tolist() == pytest.approx([x / 2 for x in range(-6, 7)])
    assert not lags.flags.writeable


def test_forward_reverse_observation_fits_line_and_freezes_support() -> None:
    x = np.arange(8, 28, dtype=np.int32)
    y = np.full_like(x, 15)
    support = np.column_stack((x, y))
    left = np.zeros((40, 40), np.float32)
    right = np.zeros_like(left)
    left[y, x] = 10.0
    right[y + 2, x] = 10.0
    observation, evidence = make_s13_edge_component_observation(
        pair_index=3,
        component_id=7,
        source_indices=(3, 4),
        support_xy=support,
        left_magnitude=left,
        right_magnitude=right,
        left_gradient_y=left,
        right_gradient_y=right,
        lags=canonical_s13_normal_search_lags(),
        block_indices=(1,),
    )
    assert observation.forward_best_lag_px == pytest.approx(-2.0)
    assert observation.reverse_best_lag_px == pytest.approx(2.0)
    assert abs(observation.forward_best_lag_px + observation.reverse_best_lag_px) <= 0.5
    assert observation.evidence_state == "actionable"
    assert evidence.support_xy.shape == (20, 2)
    assert evidence.forward_scores.shape == (13,)
    assert evidence.reverse_scores.shape == (13,)
    assert not evidence.support_xy.flags.writeable
    assert not evidence.forward_scores.flags.writeable

    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="inverse-map-sign", observations=(observation,)
    )
    offsets = solve_s13_component_segment_source_offsets(
        segment, config=_Config(), owner_support_by_source={3: 1, 4: 1}
    )
    assert offsets[4] - offsets[3] == pytest.approx(2.0, abs=1e-6)
    # In an inverse map, output feature position is raw_position - delta.
    assert 15.0 - offsets[3] == pytest.approx(17.0 - offsets[4], abs=1e-6)


def _constant_orientation_observation(
    *, right_angle_degrees: float = 0.0, inverted_rows: int = 0
) -> S13EdgeComponentObservation:
    y = np.arange(8, 28, dtype=np.int32)
    x = np.full_like(y, 20)
    support = np.column_stack((x, y))
    magnitude = np.full((40, 40), 10.0, np.float32)
    left_gx = np.full_like(magnitude, 10.0)
    left_gy = np.zeros_like(magnitude)
    angle = np.deg2rad(right_angle_degrees)
    right_gx = np.full_like(magnitude, 10.0 * np.cos(angle))
    right_gy = np.full_like(magnitude, 10.0 * np.sin(angle))
    if inverted_rows:
        right_gx[y[:inverted_rows], :] *= -1.0
        right_gy[y[:inverted_rows], :] *= -1.0
    observation, _evidence = make_s13_edge_component_observation(
        pair_index=0,
        component_id=0,
        source_indices=(0, 1),
        support_xy=support,
        left_magnitude=magnitude,
        right_magnitude=magnitude,
        left_gradient_x=left_gx,
        left_gradient_y=left_gy,
        right_gradient_x=right_gx,
        right_gradient_y=right_gy,
        lags=np.asarray([0.0], np.float64),
        minimum_uniqueness_fraction=-1.0,
    )
    return observation


def test_modulo_pi_orientation_is_independent_from_partial_gradient_sign() -> None:
    observation = _constant_orientation_observation(inverted_rows=4)

    assert observation.orientation_difference_degrees == pytest.approx(0.0)
    assert observation.signed_gradient_agreement == pytest.approx(0.6)
    assert observation.evidence_state == "safe_anchor"


def test_full_contrast_reversal_fails_signed_gate_not_orientation_gate() -> None:
    observation = _constant_orientation_observation(inverted_rows=20)

    assert observation.orientation_difference_degrees == pytest.approx(0.0)
    assert observation.signed_gradient_agreement == pytest.approx(-1.0)
    assert observation.evidence_state == "ambiguous"
    assert observation.exclusion_reasons == ("signed_gradient_disagrees",)


def test_real_orientation_difference_above_ten_degrees_still_fails() -> None:
    observation = _constant_orientation_observation(right_angle_degrees=15.0)

    assert observation.orientation_difference_degrees == pytest.approx(15.0)
    assert observation.signed_gradient_agreement > 0.9
    assert observation.evidence_state == "ambiguous"
    assert observation.exclusion_reasons == ("orientation_difference_exceeded",)


def test_probe_second_edge_geometry_has_one_unique_compatible_link() -> None:
    left = replace(
        _observation(71, 9, 3.0, y0=322),
        global_bbox_xyxy=(1505, 322, 1531, 337),
        normal_x=0.315199,
        normal_y=-0.949025,
        fitted_line_offset=165.470439,
        correlation=0.978577,
        uniqueness_fraction=0.349480,
        signed_gradient_polarity=0.233167,
    )
    right = replace(
        _observation(72, 8, -0.5, y0=321),
        global_bbox_xyxy=(1511, 321, 1537, 346),
        normal_x=0.364541,
        normal_y=-0.931187,
        fitted_line_offset=243.434758,
        correlation=0.981253,
        uniqueness_fraction=0.422243,
        signed_gradient_polarity=0.336166,
    )

    chains = build_s13_edge_component_chains((left, right), config=_Config())

    linked = [chain for chain in chains if len(chain.observations) == 2]
    assert len(linked) == 1
    assert [(row.pair_index, row.component_id) for row in linked[0].observations] == [
        (71, 9), (72, 8),
    ]


def test_match_matrix_is_deterministic_and_keeps_components_8_and_9_separate() -> None:
    observations = (
        _observation(71, 8, 1.5, y0=288),
        _observation(71, 9, 3.0, y0=322),
        _observation(72, 8, 0.5, y0=322),
        _observation(72, 9, 0.5, y0=288),
    )
    first: list[object] = []
    second: list[object] = []

    chains = build_s13_edge_component_chains(
        observations, config=_Config(), match_matrix_sink=first
    )
    build_s13_edge_component_chains(
        tuple(reversed(observations)), config=_Config(), match_matrix_sink=second
    )

    assert first == second
    assert len(first) == 1
    candidates = first[0]["candidates"]
    assert all("hard_gates" in row and "subscores" in row for row in candidates)
    assert all("winner" in row and "reject_reasons" in row for row in candidates)
    winner_ids = {
        (row["left_component_id"], row["right_component_id"])
        for row in candidates if row["winner"] is True
    }
    assert winner_ids == {(8, 9), (9, 8)}
    assert all(len(set(chain.observations)) == 2 for chain in chains)
    assert not any(
        {row.component_id for row in chain.observations} == {8, 9}
        and len({row.pair_index for row in chain.observations}) == 1
        for chain in chains
    )


def test_exact_evidence_propagates_bidirectionally_across_multiple_safe_anchors() -> None:
    rows = {
        pair: ((_observation(pair, 0, 2.0 if pair == 2 else 0.5), object()),)
        for pair in range(9)
    }
    # Physical identity ends at pair 0 and pair 4. Pairs farther away must not
    # pay the exact reverse-search cost merely because they exist.
    rows[0] = ((_observation(0, 0, 0.5, y0=60), object()),)
    rows[4] = ((_observation(4, 0, 0.5, y0=60), object()),)
    calls: list[int] = []

    def load(pair_index: int) -> object:
        calls.append(pair_index)
        return rows[pair_index]

    evidence, audit = propagate_s13_edge_component_evidence(
        {2: rows[2]},
        available_pair_indices=tuple(range(9)),
        pair_evidence_loader=load,
        config=_Config(),
    )

    assert set(calls) == {0, 1, 3, 4}
    assert 5 not in calls and 8 not in calls
    assert {(row[0].pair_index, row[0].component_id) for row in evidence} == {
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
    }
    assert audit["seed_pair_indices"] == [2]
    assert audit["reverse_evidence_loader_call_count"] == 4
    assert audit["propagation_link_count"] == 2


def test_roi_budget_priority_is_deterministic_excess_confidence_then_coverage() -> None:
    rows = [
        S13ComponentApplicationSegment.create(
            parent_chain_id=f"budget-{index}",
            observations=(_observation(index, index, lag),),
        )
        for index, lag in enumerate((1.5, 2.0, 2.0, 2.0))
    ]
    rows[2] = S13ComponentApplicationSegment.create(
        parent_chain_id="budget-confidence",
        observations=(replace(rows[2].observations[0], correlation=0.99),),
    )
    evidence = {
        (row.observations[0].pair_index, row.observations[0].component_id): (
            SimpleNamespace(support_xy=np.column_stack((
                np.arange(5 + index, dtype=np.int32),
                np.zeros(5 + index, dtype=np.int32),
            )))
        )
        for index, row in enumerate(rows)
    }
    ordered = sorted(
        reversed(rows),
        key=lambda row: s13_component_segment_budget_priority(
            row, evidence_by_component=evidence,
            maximum_post_edge_p95_px=1.0,
        ),
    )
    # Largest excess first; tied excess uses confidence before coverage, and
    # the lower-excess row is always last regardless of input order.
    assert ordered[0] is rows[2]
    assert ordered[-1] is rows[0]
    assert [row.segment_id for row in ordered] == [
        row.segment_id for row in sorted(
            rows,
            key=lambda row: s13_component_segment_budget_priority(
                row, evidence_by_component=evidence,
                maximum_post_edge_p95_px=1.0,
            ),
        )
    ]


def _gain_candidate(
    *, decision: str, gain: float, step: float, p95: float = 0.5
) -> S13ComponentSegmentCandidate:
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id=f"gain-{decision}-{gain}-{step}",
        observations=(_observation(0, 0, 2.0),),
    )
    return S13ComponentSegmentCandidate(
        segment, (gain,), gain, (), decision, (),
        {"candidate_metrics": {
            "maximum_step_px": step, "edge_p95_px": p95,
            "break_length_px": 0.0, "double_edge_length_px": 0.0,
        }, "correction_energy": gain},
    )


def test_gain_selection_prefers_resolved_then_uses_frozen_step_tolerance() -> None:
    improved = _gain_candidate(
        decision="improved_unresolved", gain=0.5, step=0.2
    )
    resolved = _gain_candidate(decision="resolved", gain=1.0, step=1.0)
    assert _select_s13_component_gain_candidate((improved, resolved)) is resolved

    small = _gain_candidate(decision="resolved", gain=0.5, step=1.14)
    large = _gain_candidate(decision="resolved", gain=1.0, step=1.0)
    assert _select_s13_component_gain_candidate((large, small)) is small

    outside = _gain_candidate(decision="resolved", gain=0.5, step=1.16)
    assert _select_s13_component_gain_candidate((large, outside)) is large


def test_obligations_are_frozen_before_chain_filtering() -> None:
    observations = (_observation(0, 0, 2.0), _observation(1, 0, 0.5))
    obligations = freeze_s13_baseline_c2e_obligations(observations)
    assert len(obligations) == 2
    assert obligations[0].severe is True
    assert obligations[1].severe is False
    assert obligations[0].scope == "runtime_detected"


def test_obligation_freeze_records_exact_column_trace_and_severity_basis() -> None:
    support = np.asarray([(column, 10 + column // 4) for column in range(16)], np.int32)
    observation = replace(
        _observation(0, 0, 2.0),
        mask_sha256=canonical_s13_support_sha256(support),
    )
    obligations = freeze_s13_baseline_c2e_obligations(
        (observation,),
        support_by_authority={
            (0, 0, observation.mask_sha256): support,
        },
        severe_threshold_px=1.0,
        minimum_evaluable_columns=12,
    )
    metrics = obligations[0].baseline_metrics
    assert obligations[0].severe is True
    assert obligations[0].evaluable is True
    assert metrics["trace_unique_column_count"] == 16
    assert metrics["trace_missing_column_count"] == 0
    assert metrics["severe_reason"] == "absolute_symmetric_normal_lag_exceeded"


def test_obligation_coverage_rejects_pair_source_authority_tamper() -> None:
    sha = "a" * 64
    obligation = {
        "pair_index": 2, "component_id": 7, "support_sha256": sha,
        "scope": "runtime_detected", "baseline_metrics": {},
        "severe": True, "evaluable": True,
    }
    segment = {
        "segment_id": "s", "pair_indices": [2], "source_indices": [2, 4],
        "state": "resolved", "observation_authority": [{
            "pair_index": 2, "component_id": 7, "support_sha256": sha,
        }],
    }
    with pytest.raises(ValueError, match="pair/source authority"):
        audit_s13_obligation_coverage((obligation,), (segment,), ("s",))


def test_chain_matching_keeps_distinct_y_layers_and_segmentation_cuts_ambiguity() -> None:
    observations = (
        _observation(0, 8, 1.5, y0=10),
        _observation(1, 8, 1.5, y0=11),
        _observation(0, 9, 3.0, y0=50),
        _observation(1, 9, 3.0, y0=51),
    )
    chains = build_s13_edge_component_chains(observations, config=_Config())
    assert len(chains) == 2
    assert sorted(chain.pair_indices for chain in chains) == [(0, 1), (0, 1)]
    first = chains[0]
    broken = first.observations[0]
    broken = S13EdgeComponentObservation(
        **{**broken.__dict__, "evidence_state": "ambiguous", "exclusion_reasons": ("test",)}
    )
    first = type(first)(
        chain_id=first.chain_id,
        pair_indices=first.pair_indices,
        source_indices=first.source_indices,
        observations=(broken, *first.observations[1:]),
        canonical_normal_xy=first.canonical_normal_xy,
        global_bbox_xyxy=first.global_bbox_xyxy,
        chain_sha256=first.chain_sha256,
    )
    segments = split_s13_edge_component_chain(first, config=_Config())
    assert len(segments) == 1
    assert segments[0].pair_indices == (1,)
    assert segments[0].left_cut_reason == "ambiguous_observation"


def test_chain_matching_rejects_pair_gap_and_reversed_gradient_polarity() -> None:
    config = replace(_Config(), minimum_chain_pair_count=1)
    gap = build_s13_edge_component_chains(
        (_observation(0, 0, 2.0), _observation(2, 0, 2.0)), config=config
    )
    assert [chain.pair_indices for chain in gap] == [(0,), (2,)]

    reversed_polarity = replace(
        _observation(1, 0, 2.0), signed_gradient_polarity=-1.0
    )
    polarity = build_s13_edge_component_chains(
        (_observation(0, 0, 2.0), reversed_polarity), config=config
    )
    assert {chain.pair_indices for chain in polarity} == {(0,), (1,)}


def test_ten_seam_alternating_sign_chain_solves_and_renders_local_map() -> None:
    observations = tuple(
        _observation(pair, 0, 1.5 if pair % 2 == 0 else -1.5)
        for pair in range(10)
    )
    chains = build_s13_edge_component_chains(observations, config=_Config())
    assert len(chains) == 1
    assert chains[0].pair_indices == tuple(range(10))
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id=chains[0].chain_id, observations=observations
    )
    offsets = solve_s13_component_segment_source_offsets(
        segment,
        config=_Config(),
        owner_support_by_source={source: 1 for source in segment.source_indices},
    )
    assert max(abs(value) for value in offsets.values()) <= 3.0
    support = np.zeros((48, 64), bool)
    support[20, 8:56] = True
    correction = build_s13_source_component_correction(
        segment_id=segment.segment_id,
        source_index=1,
        frame_id=101,
        support_mask=support,
        source_offset_px=offsets[1],
        normal_xy=(0.0, 1.0),
        config=_Config(),
    )
    base = np.zeros((48, 64), np.float32)
    _du, rendered_dv, field = compose_s13_source_component_deltas(
        base_delta_u=base,
        base_delta_v=base,
        domain_xyxy=(0, 0, 64, 48),
        corrections=(correction,),
    )
    assert np.any(rendered_dv != 0.0)
    assert np.any(field == 0)


def test_solver_rejects_offset_limit_and_ill_conditioned_system() -> None:
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="bounds",
        observations=tuple(_observation(pair, 0, 3.0) for pair in range(4)),
    )
    support = {source: 1 for source in segment.source_indices}
    with pytest.raises(ValueError, match="offset"):
        solve_s13_component_segment_source_offsets(
            segment, config=_Config(), owner_support_by_source=support
        )
    with pytest.raises(ValueError, match="condition"):
        solve_s13_component_segment_source_offsets(
            segment,
            config=replace(
                _Config(),
                maximum_source_normal_offset_px=100.0,
                maximum_solver_condition_number=1.0,
            ),
            owner_support_by_source=support,
        )


def test_irls_solver_uses_owner_weighted_zero_mean_gauge() -> None:
    observations = (_observation(0, 0, 2.0), _observation(1, 0, 2.0))
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-a", observations=observations
    )
    offsets = solve_s13_component_segment_source_offsets(
        segment,
        config=_Config(),
        owner_support_by_source={0: 1, 1: 2, 2: 1},
    )
    assert offsets[1] - offsets[0] == pytest.approx(-2.0, abs=1e-6)
    assert offsets[2] - offsets[1] == pytest.approx(-2.0, abs=1e-6)
    assert sum([offsets[0], 2 * offsets[1], offsets[2]]) == pytest.approx(0.0, abs=1e-6)


def test_solver_outlier_failure_carries_deterministic_pair_split_audits() -> None:
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-outlier",
        observations=tuple(
            _observation(pair, 0, lag)
            for pair, lag in enumerate((3.0, -3.0, 3.0, -3.0))
        ),
    )
    config = _Config(maximum_normalized_solver_residual=0.01)

    with pytest.raises(S13ComponentSolverError) as caught:
        solve_s13_component_segment_source_offsets(
            segment,
            config=config,
            owner_support_by_source={source: 1 for source in segment.source_indices},
        )

    audits = caught.value.pair_audits
    assert [row["pair_index"] for row in audits] == [0, 1, 2, 3]
    assert all(row["normalized_solver_residual"] >= 0.0 for row in audits)
    assert any(row["hard_violation_count"] == 1 for row in audits)
    assert select_s13_component_segment_split_pair(audits) in segment.pair_indices


def test_correction_is_local_readonly_and_patch_conflicts_choose_utility() -> None:
    observation = _observation(0, 0, 2.0)
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-a", observations=(observation,)
    )
    support = np.zeros((32, 32), bool)
    support[np.arange(8, 24), np.arange(8, 24)] = True
    correction = build_s13_source_component_correction(
        segment_id=segment.segment_id,
        source_index=1,
        frame_id=101,
        support_mask=support,
        source_offset_px=2.0,
        normal_xy=(0.0, 1.0),
        config=_Config(),
    )
    assert correction.weight.max() == pytest.approx(1.0)
    assert np.all(correction.delta_u == 0.0)
    assert np.all(correction.delta_v[correction.weight == 0] == 0.0)
    assert not correction.delta_v.flags.writeable
    for values in (correction.weight, correction.delta_u, correction.delta_v):
        assert np.allclose(values.astype(np.float64) * 1e6, np.rint(values * 1e6))
    low = S13ComponentSegmentCandidate(
        segment, (2.0,), 1.0, (correction,), "resolved", (),
        {"rescued_severe_seam_count": 1, "worst_seam_absolute_improvement": 0.5,
         "supported_unique_edge_columns": 10, "post_maximum_step": 1.0,
         "correction_energy": 100.0},
    )
    high_segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-b", observations=(observation,)
    )
    high_correction = S13SourceComponentCorrection(
        **{**correction.__dict__, "segment_id": high_segment.segment_id}
    )
    high = S13ComponentSegmentCandidate(
        high_segment, (2.0,), 1.0, (high_correction,), "resolved", (),
        {"rescued_severe_seam_count": 2, "worst_seam_absolute_improvement": 0.5,
         "supported_unique_edge_columns": 10, "post_maximum_step": 1.0,
         "correction_energy": 100.0},
    )
    patch = select_s13_component_patch_set((low, high), config=_Config())
    assert patch.accepted_segment_ids == (high.segment.segment_id,)
    assert low.segment.segment_id in patch.rejected_segment_ids
    registry = freeze_s13_source_correction_registry(patch)
    assert registry.accepted_segment_ids == patch.accepted_segment_ids
    assert dict(registry.field_id_by_segment) == {high.segment.segment_id: 0}
    with pytest.raises(TypeError):
        registry.corrections_by_source[99] = ()


def _conflict_test_candidate(
    name: str,
    *,
    correction_points: tuple[tuple[int, int], ...],
    rescued_seams: int,
    utility_penalty: float = 1.0,
) -> S13ComponentSegmentCandidate:
    unique = sum((index + 1) * ord(value) for index, value in enumerate(name))
    observation = _observation(unique, unique, 2.0)
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id=f"conflict-{name}", observations=(observation,)
    )
    corrections = tuple(
        S13SourceComponentCorrection(
            segment_id=segment.segment_id,
            source_index=source_index,
            frame_id=1000 + source_index,
            x0=x,
            x1=x + 1,
            y0=0,
            y1=1,
            delta_u=np.zeros((1, 1), np.float32),
            delta_v=np.zeros((1, 1), np.float32),
            weight=np.ones((1, 1), np.float32),
            correction_sha256=f"{unique + source_index + x:064x}"[-64:],
        )
        for source_index, x in correction_points
    )
    return S13ComponentSegmentCandidate(
        segment,
        (1.0,),
        1.0,
        corrections,
        "resolved",
        (),
        {
            "rescued_severe_seam_count": rescued_seams,
            "worst_seam_absolute_improvement": 0.0,
            "supported_unique_edge_columns": 0,
            "post_maximum_step": utility_penalty,
            "correction_energy": utility_penalty,
        },
    )


def test_conflicts_are_solved_exactly_per_connected_component() -> None:
    center = _conflict_test_candidate(
        "center", correction_points=((0, 10), (0, 30)), rescued_seams=3
    )
    left = _conflict_test_candidate(
        "left", correction_points=((0, 10),), rescued_seams=2
    )
    right = _conflict_test_candidate(
        "right", correction_points=((0, 30),), rescued_seams=2
    )
    isolated = tuple(
        _conflict_test_candidate(
            f"isolated-{index}",
            correction_points=((index + 1, index),),
            rescued_seams=1,
        )
        for index in range(10)
    )
    candidates = (center, left, right, *isolated)

    forward = select_s13_component_patch_set(candidates, config=_Config())
    reverse = select_s13_component_patch_set(tuple(reversed(candidates)), config=_Config())

    expected = tuple(sorted((left.segment.segment_id, right.segment.segment_id, *(
        row.segment.segment_id for row in isolated
    ))))
    assert len(candidates) == 13
    assert forward.accepted_segment_ids == expected
    assert reverse.accepted_segment_ids == expected
    assert forward.rejected_segment_ids == (center.segment.segment_id,)
    assert reverse.rejected_segment_ids == (center.segment.segment_id,)
    assert forward.audit["conflict_components"] == reverse.audit["conflict_components"]
    methods = [row["selection_method"] for row in forward.audit["conflict_components"]]
    assert methods.count("exact") == 1
    assert methods.count("isolated") == 10
    assert "lexicographic_greedy" not in methods


def test_isolated_candidates_are_all_retained_independent_of_total_count() -> None:
    candidates = tuple(
        _conflict_test_candidate(
            f"weak-isolated-{index}",
            correction_points=((index, index),),
            rescued_seams=0,
            utility_penalty=100.0,
        )
        for index in range(13)
    )

    patch = select_s13_component_patch_set(
        tuple(reversed(candidates)),
        config=_Config(maximum_exact_conflict_group_nodes=2),
    )

    assert patch.accepted_segment_ids == tuple(sorted(
        candidate.segment.segment_id for candidate in candidates
    ))
    assert patch.rejected_segment_ids == ()
    assert all(
        row["selection_method"] == "isolated"
        for row in patch.audit["conflict_components"]
    )


def test_same_source_nonintersecting_segments_are_both_selected() -> None:
    candidates = []
    for component_id, x_slice in ((0, slice(5, 17)), (1, slice(43, 55))):
        observation = _observation(0, component_id, 2.0, y0=10 + 20 * component_id)
        segment = S13ComponentApplicationSegment.create(
            parent_chain_id=f"chain-{component_id}", observations=(observation,)
        )
        support = np.zeros((64, 64), bool)
        support[12 + 20 * component_id, x_slice] = True
        correction = build_s13_source_component_correction(
            segment_id=segment.segment_id,
            source_index=0,
            frame_id=100,
            support_mask=support,
            source_offset_px=1.0,
            normal_xy=(0.0, 1.0),
            config=_Config(),
        )
        candidates.append(S13ComponentSegmentCandidate(
            segment, (1.0,), 1.0, (correction,), "resolved", (),
            {"rescued_severe_seam_count": 1,
             "worst_seam_absolute_improvement": 1.0,
             "supported_unique_edge_columns": 12,
             "post_maximum_step": 1.0,
             "correction_energy": 1.0},
        ))
    patch = select_s13_component_patch_set(tuple(candidates), config=_Config())
    assert set(patch.accepted_segment_ids) == {
        candidate.segment.segment_id for candidate in candidates
    }
    assert len(patch.corrections_by_source[0]) == 2


def test_roi_budget_defers_lower_priority_segments_deterministically() -> None:
    segments = [S13ComponentApplicationSegment.create(
        parent_chain_id=name,
        observations=(_observation(index, index, lag),),
    ) for index, (name, lag) in enumerate((
        ("low", 1.2), ("high", 3.0), ("middle", 2.0)
    ))]
    evidence = {
        (segment.observations[0].pair_index, segment.observations[0].component_id):
        SimpleNamespace(support_xy=np.column_stack((
            np.arange(30 + index * 10), np.zeros(30 + index * 10)
        )))
        for index, segment in enumerate(segments)
    }
    ordered = sorted(
        segments,
        key=lambda segment: s13_component_segment_budget_priority(
            segment,
            evidence_by_component=evidence,
            maximum_post_edge_p95_px=1.0,
        ),
    )
    names = {segment.segment_id: segment.parent_chain_id for segment in segments}
    costs = {"high": 60, "middle": 40, "low": 30}
    used = 0
    selected, deferred = [], []
    for segment in ordered:
        name = names[segment.segment_id]
        cost = costs[name]
        if used + cost > 100:
            deferred.append(name)
        else:
            selected.append(name)
            used += cost
    assert selected == ["high", "middle"]
    assert deferred == ["low"]


def _normal_observation(pair: int, angle_degrees: float, *, negate: bool = False):
    angle = np.deg2rad(angle_degrees)
    nx, ny = float(np.sin(angle)), float(np.cos(angle))
    if negate:
        nx, ny = -nx, -ny
    x0 = 8 + pair * 20
    return replace(
        _observation(pair, 0, 2.0, y0=10),
        global_bbox_xyxy=(x0, 10, x0 + 8, 30),
        normal_x=nx,
        normal_y=ny,
        fitted_line_offset=ny * 20.0 + nx * (x0 + 3.5),
    )


def test_local_normal_field_bends_deterministically_between_fitted_lines() -> None:
    support = np.zeros((40, 48), bool)
    support[20, 8:36] = True
    correction = build_s13_source_component_correction(
        segment_id="curved", source_index=1, frame_id=10,
        support_mask=support, source_offset_px=2.0, normal_xy=(0.0, 1.0),
        config=_Config(),
        local_normal_observations=(
            _normal_observation(0, -8.0), _normal_observation(1, 8.0),
        ),
    )
    active = correction.weight > 0.5
    global_columns = np.indices(active.shape)[1] + correction.x0
    assert float(np.mean(correction.delta_u[active & (global_columns < 18)])) < 0.0
    assert float(np.mean(correction.delta_u[active & (global_columns > 27)])) > 0.0
    assert correction.normal_field_audit["node_count"] == 2
    assert correction.normal_field_audit["maximum_pixel_angle_degrees"] <= 8.0 + 1e-6
    assert not correction.delta_u.flags.writeable


def test_local_normal_field_canonicalizes_opposite_hemisphere_exactly() -> None:
    support = np.zeros((40, 48), bool)
    support[20, 8:36] = True
    kwargs = dict(
        segment_id="sign", source_index=1, frame_id=10,
        support_mask=support, source_offset_px=1.0, normal_xy=(0.0, 1.0),
        config=_Config(),
    )
    positive = build_s13_source_component_correction(
        **kwargs, local_normal_observations=(_normal_observation(0, 8.0),)
    )
    negated = build_s13_source_component_correction(
        **kwargs, local_normal_observations=(_normal_observation(0, 8.0, negate=True),)
    )
    assert np.array_equal(positive.delta_u, negated.delta_u)
    assert np.array_equal(positive.delta_v, negated.delta_v)
    assert positive.correction_sha256 == negated.correction_sha256


def test_local_normal_field_rejects_node_outside_canonical_angle() -> None:
    support = np.zeros((40, 48), bool)
    support[20, 8:36] = True
    with pytest.raises(ValueError, match="normal node.*angle"):
        build_s13_source_component_correction(
            segment_id="angle", source_index=1, frame_id=10,
            support_mask=support, source_offset_px=1.0, normal_xy=(0.0, 1.0),
            config=_Config(), local_normal_observations=(_normal_observation(0, 11.0),),
        )


def test_single_straight_local_normal_is_array_and_hash_exact_compatible() -> None:
    support = np.zeros((32, 32), bool)
    support[16, 6:26] = True
    kwargs = dict(
        segment_id="straight", source_index=1, frame_id=10,
        support_mask=support, source_offset_px=1.25, normal_xy=(0.0, 1.0),
        config=_Config(),
    )
    legacy = build_s13_source_component_correction(**kwargs)
    fitted = build_s13_source_component_correction(
        **kwargs, local_normal_observations=(_normal_observation(0, 0.0),)
    )
    assert np.array_equal(legacy.delta_u, fitted.delta_u)
    assert np.array_equal(legacy.delta_v, fitted.delta_v)
    assert np.array_equal(legacy.weight, fitted.weight)
    assert legacy.correction_sha256 == fitted.correction_sha256


def test_repair_complete_requires_every_exact_frozen_obligation() -> None:
    resolved_observation = _observation(0, 0, 2.0)
    unresolved_same_pair = _observation(0, 1, 2.0, y0=40)
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-a", observations=(resolved_observation,)
    )
    support = np.zeros((80, 80), bool)
    support[10:30, 20:32] = True
    correction = build_s13_source_component_correction(
        segment_id=segment.segment_id,
        source_index=0,
        frame_id=100,
        support_mask=support,
        source_offset_px=1.0,
        normal_xy=(0.0, 1.0),
        config=_Config(),
    )
    candidate = S13ComponentSegmentCandidate(
        segment, (1.0,), 1.0, (correction,), "resolved", (),
        {"rescued_severe_seam_count": 1, "worst_seam_absolute_improvement": 1.0,
         "supported_unique_edge_columns": 12, "post_maximum_step": 1.0,
         "correction_energy": 1.0},
    )
    obligations = freeze_s13_baseline_c2e_obligations(
        (resolved_observation, unresolved_same_pair)
    )

    patch = select_s13_component_patch_set(
        (candidate,), config=_Config(), obligations=obligations
    )

    assert patch.application_state == "partial"
    assert patch.audit["repair_complete"] is False


def test_source_map_oracle_canonicalizes_invalid_and_negative_zero() -> None:
    u = np.asarray([[1.0, -0.0], [3.0, 4.0]], np.float32)
    v = np.asarray([[2.0, 9.0], [-0.0, 5.0]], np.float32)
    valid = np.asarray([[True, False], [False, True]])
    field = np.asarray([[4, 7], [8, -1]], np.int32)
    oracle = source_map_oracle_from_arrays(
        source_index=2,
        domain_xyxy=(10, 20, 12, 22),
        u=u,
        v=v,
        valid=valid,
        field_id=field,
    )
    invalid = oracle.valid == 0
    assert np.all(oracle.u[invalid] == 0.0)
    assert np.all(oracle.v[invalid] == 0.0)
    assert np.all(oracle.field_id[invalid] == -1)
    assert not np.signbit(oracle.u).any()
    assert len(oracle.oracle_sha256) == 64
    assert not oracle.u.flags.writeable


def _candidate_map_safety_inputs() -> dict[str, object]:
    height = width = 10
    base_u = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width)).copy()
    base_v = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None], (height, width)
    ).copy()
    influence = np.zeros((height, width), bool)
    influence[2:8, 2:8] = True
    evidence = np.zeros_like(influence)
    evidence[4:6, 4:6] = True
    candidate_u = base_u.copy()
    candidate_v = base_v.copy()
    candidate_v[influence] += 0.1
    return {
        "base_u": base_u,
        "base_v": base_v,
        "base_valid": np.ones_like(influence),
        "candidate_u": candidate_u,
        "candidate_v": candidate_v,
        "candidate_valid": np.ones_like(influence),
        "owner_mask": np.ones_like(influence),
        "evidence_mask": evidence,
        "influence_mask": influence,
        "source_size": (width, height),
        "minimum_owner_retention": 1.0,
        "minimum_evidence_retention": 0.98,
        "minimum_jacobian": 0.5,
        "maximum_displacement_px": 8.0,
        "maximum_halo_regression_px": 0.25,
    }


def test_candidate_map_safety_accepts_local_exact_bounded_map() -> None:
    audit = audit_s13_component_candidate_map_safety(
        **_candidate_map_safety_inputs()
    )
    assert audit["passed"] is True
    assert audit["segment_boundary_curvature_spike_px"] <= 0.25
    assert audit["formal_owner_support_retention"] == 1.0
    assert audit["evidence_sample_retention"] == 1.0
    assert audit["exterior_map_exact"] is True
    assert audit["candidate_minimum_inverse_map_jacobian"] >= 0.5
    assert audit["minimum_inverse_map_jacobian_ratio"] is not None


@pytest.mark.parametrize(
    ("tamper", "failed_field"),
    [
        ("owner_valid", "formal_owner_valid_exact"),
        ("evidence_valid", "evidence_sample_retention"),
        ("bounds", "affected_maps_in_bounds"),
        ("displacement", "maximum_combined_map_displacement_px"),
        ("jacobian", "candidate_minimum_inverse_map_jacobian"),
        ("halo", "halo_regression_px"),
        ("exterior", "exterior_map_exact"),
    ],
)
def test_candidate_map_safety_fails_closed_on_gate_tamper(
    tamper: str, failed_field: str,
) -> None:
    values = _candidate_map_safety_inputs()
    influence = np.asarray(values["influence_mask"])
    evidence = np.asarray(values["evidence_mask"])
    if tamper == "owner_valid":
        values["candidate_valid"] = np.asarray(values["candidate_valid"]).copy()
        values["candidate_valid"][1, 1] = False
    elif tamper == "evidence_valid":
        values["candidate_valid"] = np.asarray(values["candidate_valid"]).copy()
        values["candidate_valid"][evidence] = False
    elif tamper == "bounds":
        values["candidate_u"] = np.asarray(values["candidate_u"]).copy()
        values["candidate_u"][influence] = 20.0
    elif tamper == "displacement":
        values["candidate_v"] = np.asarray(values["candidate_v"]).copy()
        values["candidate_v"][influence] += 9.0
        values["source_size"] = (100, 100)
    elif tamper == "jacobian":
        values["candidate_u"] = np.asarray(values["candidate_u"]).copy()
        values["candidate_u"][:, 2:8] = values["candidate_u"][:, 2:8][:, ::-1]
        values["maximum_halo_regression_px"] = 20.0
    elif tamper == "halo":
        values["candidate_v"] = np.asarray(values["candidate_v"]).copy()
        values["candidate_v"][influence] += 0.4
    elif tamper == "exterior":
        values["candidate_u"] = np.asarray(values["candidate_u"]).copy()
        values["candidate_u"][0, 0] += 0.01
    audit = audit_s13_component_candidate_map_safety(**values)
    assert audit["passed"] is False
    if failed_field in {"maximum_combined_map_displacement_px", "halo_regression_px"}:
        assert float(audit[failed_field]) > float(
            values[
                "maximum_displacement_px"
                if failed_field.startswith("maximum_combined")
                else "maximum_halo_regression_px"
            ]
        )
    elif failed_field == "candidate_minimum_inverse_map_jacobian":
        assert float(audit[failed_field]) < float(values["minimum_jacobian"])
    elif failed_field == "evidence_sample_retention":
        assert float(audit[failed_field]) < float(values["minimum_evidence_retention"])
    else:
        assert audit[failed_field] is False


def test_rejected_exclusive_domain_stays_exact_under_accepted_patch() -> None:
    values = _candidate_map_safety_inputs()
    accepted = np.asarray(values["influence_mask"])
    rejected_exclusive = np.zeros_like(accepted)
    rejected_exclusive[0:2, 7:10] = True
    audit = audit_s13_component_candidate_map_safety(**values)
    assert audit["passed"] is True
    assert np.array_equal(
        np.asarray(values["candidate_u"])[rejected_exclusive],
        np.asarray(values["base_u"])[rejected_exclusive],
    )
    tampered = dict(values)
    tampered["candidate_u"] = np.asarray(values["candidate_u"]).copy()
    tampered["candidate_u"][rejected_exclusive] += 0.01
    assert audit_s13_component_candidate_map_safety(**tampered)["passed"] is False


def test_delta_composition_adds_one_winner_and_assigns_canonical_field_id() -> None:
    support = np.zeros((24, 24), bool)
    support[10, 5:19] = True
    correction = build_s13_source_component_correction(
        segment_id="segment-z", source_index=0, frame_id=1,
        support_mask=support, source_offset_px=1.0, normal_xy=(0.0, 1.0),
        config=_Config(),
    )
    base_u = np.full((24, 24), 0.25, np.float32)
    base_v = np.full((24, 24), -0.5, np.float32)
    delta_u, delta_v, field = compose_s13_source_component_deltas(
        base_delta_u=base_u,
        base_delta_v=base_v,
        domain_xyxy=(0, 0, 24, 24),
        corrections=(correction,),
    )
    assert np.array_equal(delta_u, base_u)
    assert np.any(delta_v != base_v)
    assert set(np.unique(field)) == {-1, 0}
    assert np.all(field[delta_v == base_v] == -1)
    assert not delta_v.flags.writeable


def test_quality_classification_and_split_key_are_deterministic() -> None:
    evaluation = evaluate_s13_component_candidate_quality(
        baseline={"edge_p95_px": 1.2, "maximum_step_px": 1.4,
                  "break_length_px": 0, "double_edge_length_px": 0,
                  "non_target_p95_px": 0.2},
        candidate={"edge_p95_px": 0.9, "maximum_step_px": 1.4,
                   "break_length_px": 0, "double_edge_length_px": 0,
                   "non_target_p95_px": 0.2},
        hard_gates={"map_finite": True, "jacobian": True},
        config=_Config(),
    )
    assert evaluation.decision == "resolved"
    boundary = evaluate_s13_component_candidate_quality(
        baseline={"edge_p95_px": 2.0, "maximum_step_px": 1.4,
                  "break_length_px": 0, "double_edge_length_px": 0,
                  "non_target_p95_px": 0.2},
        candidate={"edge_p95_px": 0.9, "maximum_step_px": 1.4,
                   "break_length_px": 0, "double_edge_length_px": 0,
                   "non_target_p95_px": 0.2, "search_boundary_hit": True},
        hard_gates={"map_finite": True},
        config=_Config(),
    )
    assert boundary.decision == "improved_unresolved"
    baseline_boundary = evaluate_s13_component_candidate_quality(
        baseline={"edge_p95_px": 2.0, "maximum_step_px": 1.4,
                  "break_length_px": 0, "double_edge_length_px": 0,
                  "non_target_p95_px": 0.2, "search_boundary_hit": True},
        candidate={"edge_p95_px": 0.9, "maximum_step_px": 1.4,
                   "break_length_px": 0, "double_edge_length_px": 0,
                   "non_target_p95_px": 0.2},
        hard_gates={"map_finite": True},
        config=_Config(),
    )
    assert baseline_boundary.decision == "improved_unresolved"
    assert baseline_boundary.audit["baseline_search_boundary_hit"] is True
    split = select_s13_component_segment_split_pair((
        {"pair_index": 8, "hard_violation_count": 1,
         "maximum_normalized_gate_excess": 0.2, "normalized_solver_residual": 1.0,
         "baseline_edge_excess_px": 2.0},
        {"pair_index": 7, "hard_violation_count": 1,
         "maximum_normalized_gate_excess": 0.2, "normalized_solver_residual": 1.0,
         "baseline_edge_excess_px": 2.0},
    ))
    assert split == 7


def test_recursive_split_removes_failed_edge_and_preserves_outer_lineage() -> None:
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-recursive",
        observations=tuple(_observation(pair, 0, 2.0) for pair in range(5)),
        left_cut_reason="outer-left",
        right_cut_reason="outer-right",
    )

    children = split_s13_component_application_segment_at_pair(
        segment,
        split_pair_index=2,
        reason="solver_outlier",
        minimum_pair_count=1,
    )

    assert [child.pair_indices for child in children] == [(0, 1), (3, 4)]
    assert children[0].left_cut_reason == "outer-left"
    assert children[0].right_cut_reason == "solver_outlier"
    assert children[1].left_cut_reason == "solver_outlier"
    assert children[1].right_cut_reason == "outer-right"
    assert all(2 not in child.pair_indices for child in children)
    assert all(child.parent_chain_id == segment.parent_chain_id for child in children)


def test_recursive_split_enforces_minimum_child_pair_count_locally() -> None:
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-minimum",
        observations=tuple(_observation(pair, 0, 2.0) for pair in range(4)),
    )

    children = split_s13_component_application_segment_at_pair(
        segment,
        split_pair_index=1,
        reason="candidate_gate_failed",
        minimum_pair_count=2,
    )

    assert len(children) == 1
    assert children[0].pair_indices == (2, 3)
    assert children[0].left_cut_reason == "candidate_gate_failed"


def test_recursive_split_rejects_invalid_authority() -> None:
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-invalid",
        observations=(_observation(4, 0, 2.0),),
    )
    with pytest.raises(ValueError, match="outside"):
        split_s13_component_application_segment_at_pair(
            segment, split_pair_index=5, reason="invalid"
        )


def test_failed_segment_split_policy_honors_depth_partial_and_solver_controls() -> None:
    segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-policy",
        observations=tuple(_observation(pair, 0, 2.0) for pair in range(5)),
    )
    audits = tuple(
        {
            "pair_index": pair,
            "hard_violation_count": int(pair == 2),
            "maximum_normalized_gate_excess": 0.0,
            "normalized_solver_residual": 0.0,
            "baseline_edge_excess_px": 1.0,
        }
        for pair in range(5)
    )

    split_pair, children = plan_s13_failed_segment_split(
        segment,
        pair_audits=audits,
        reason="solver_outlier",
        split_depth=0,
        failure_kind="solver",
        config=_Config(),
    )
    assert split_pair == 2
    assert [child.pair_indices for child in children] == [(0, 1), (3, 4)]

    disabled = _Config(split_on_solver_outlier=False)
    assert plan_s13_failed_segment_split(
        segment,
        pair_audits=audits,
        reason="solver_outlier",
        split_depth=0,
        failure_kind="solver",
        config=disabled,
    ) == (None, ())
    assert plan_s13_failed_segment_split(
        segment,
        pair_audits=audits,
        reason="quality_gate",
        split_depth=disabled.maximum_segment_split_depth,
        failure_kind="quality",
        config=disabled,
    ) == (None, ())
    no_partial = _Config(allow_partial_application=False)
    assert plan_s13_failed_segment_split(
        segment,
        pair_audits=audits,
        reason="quality_gate",
        split_depth=0,
        failure_kind="quality",
        config=no_partial,
    ) == (None, ())


def _runtime_shifted_line() -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((96, 96, 3), np.uint8)
    cv2.line(left, (28, 8), (68, 88), (255, 255, 255), 2, cv2.LINE_AA)
    transform = np.asarray([[1.0, 0.0, -1.8], [0.0, 1.0, 0.9]], np.float32)
    right = cv2.warpAffine(left, transform, (96, 96), flags=cv2.INTER_LINEAR)
    return left, right


def _runtime_evidence(
    left: np.ndarray, right: np.ndarray,
) -> tuple[dict[str, object], list[tuple[object, object]]]:
    sink: list[tuple[object, object]] = []
    metrics = pair_edge_registration_metrics(
        left,
        right,
        np.ones(left.shape[:2], bool),
        np.ones(right.shape[:2], bool),
        np.full(left.shape[0], left.shape[1] // 2, np.int32),
        config=S13M51R4Config(),
        exact_evidence_sink=sink,
        pair_index=5,
        global_x_offset=100,
    )
    return metrics, sink


def test_runtime_sink_merges_overlapping_blocks_and_uses_canonical_support_hash() -> None:
    left, right = _runtime_shifted_line()
    first_metrics, first = _runtime_evidence(left, right)
    second_metrics, second = _runtime_evidence(left, right)

    assert first_metrics["supported_component_count"] == 1
    assert first_metrics["component_audits"][0]["observation_count"] >= 2
    assert second_metrics["supported_component_count"] == 1
    assert len(first) == len(second) == 1
    first_observation, first_evidence = first[0]
    second_observation, _second_evidence = second[0]
    assert first_observation.component_id == 0
    assert first_observation.mask_sha256 == canonical_s13_support_sha256(
        first_evidence.support_xy
    )
    assert first_observation.global_bbox_xyxy[0] >= 100
    assert (first_observation.normal_x, first_observation.normal_y) == pytest.approx(
        (second_observation.normal_x, second_observation.normal_y), abs=1e-9
    )
    assert first_observation.forward_best_lag_px == second_observation.forward_best_lag_px
    assert first_observation.reverse_best_lag_px == second_observation.reverse_best_lag_px
    assert abs(
        first_observation.forward_best_lag_px
        + first_observation.reverse_best_lag_px
    ) <= 0.5

    next_sink: list[tuple[object, object]] = []
    pair_edge_registration_metrics(
        left,
        right,
        np.ones(left.shape[:2], bool),
        np.ones(right.shape[:2], bool),
        np.full(left.shape[0], left.shape[1] // 2, np.int32),
        config=S13M51R4Config(),
        exact_evidence_sink=next_sink,
        pair_index=6,
        global_x_offset=102,
    )
    chains = build_s13_edge_component_chains(
        (first_observation, next_sink[0][0]), config=S13M51R4Config()
    )
    assert len(chains) == 1
    assert chains[0].pair_indices == (5, 6)


def test_runtime_sink_keeps_ambiguity_local_and_reverse_failure_does_not_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    left = np.zeros((128, 96, 3), np.uint8)
    cv2.line(left, (32, 8), (50, 48), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(left, (43, 8), (61, 48), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(left, (32, 78), (62, 122), (255, 255, 255), 2, cv2.LINE_AA)
    transform = np.asarray([[1.0, 0.0, -1.7], [0.0, 1.0, 1.1]], np.float32)
    right = cv2.warpAffine(left, transform, (96, 128), flags=cv2.INTER_LINEAR)

    metrics, evidence = _runtime_evidence(left, right)
    assert metrics["ambiguous_component_count"] >= 1
    assert any(row[0].evidence_state == "ambiguous" for row in evidence)
    assert any(row[0].evidence_state == "actionable" for row in evidence)

    import panorama_demo.video_s13_m51_r4_component_chain as component_chain

    def reverse_failure(**_kwargs: object) -> object:
        raise ValueError("synthetic reverse failure")

    monkeypatch.setattr(
        component_chain, "make_s13_edge_component_observation", reverse_failure
    )
    failed_metrics, failed_evidence = _runtime_evidence(*_runtime_shifted_line())
    assert failed_metrics["evaluable"] is True
    assert len(failed_evidence) == 1
    failed_observation, failed_arrays = failed_evidence[0]
    assert failed_observation.evidence_state == "unevaluable"
    assert failed_observation.exclusion_reasons == (
        "symmetric_hypothesis_failed:ValueError",
    )
    assert np.all(np.isneginf(failed_arrays.forward_scores))
    assert np.all(np.isneginf(failed_arrays.reverse_scores))
    assert np.all(failed_arrays.forward_support_counts == 0)
    obligations = freeze_s13_baseline_c2e_obligations((failed_observation,))
    assert len(obligations) == 1
    assert obligations[0].evaluable is False


def test_runtime_sink_freezes_detected_component_when_all_lags_are_unevaluable() -> None:
    left, _right = _runtime_shifted_line()
    metrics, evidence = _runtime_evidence(left, np.zeros_like(left))

    assert metrics["evaluable"] is False
    assert len(evidence) == 1
    observation, arrays = evidence[0]
    assert observation.evidence_state == "unevaluable"
    assert np.all(np.isneginf(arrays.forward_scores))
    assert observation.mask_sha256 == canonical_s13_support_sha256(arrays.support_xy)
    obligations = freeze_s13_baseline_c2e_obligations((observation,))
    assert len(obligations) == 1
    assert obligations[0].evaluable is False


def test_runtime_sink_reuses_forward_rows_and_only_samples_reverse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import panorama_demo.video_s13_m51_r4_component_chain as component_chain

    original = component_chain._hypothesis_scores
    calls = 0

    def counted(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(component_chain, "_hypothesis_scores", counted)
    metrics, evidence = _runtime_evidence(*_runtime_shifted_line())

    assert metrics["supported_component_count"] == 1
    assert len(evidence) == 1
    assert evidence[0][0].evidence_state == "actionable"
    assert calls == 1


def test_runtime_forward_cache_carries_unsigned_and_signed_orientation_separately() -> None:
    left, right = _runtime_shifted_line()
    contexts: list[object] = []
    pair_edge_registration_metrics(
        left,
        right,
        np.ones(left.shape[:2], bool),
        np.ones(right.shape[:2], bool),
        np.full(left.shape[0], left.shape[1] // 2, np.int32),
        config=S13M51R4Config(),
        forward_evidence_sink=contexts,
        pair_index=5,
    )

    assert len(contexts) == 1
    rows = [
        row
        for block_rows in contexts[0]["forward_rows_by_block"].values()
        for row in block_rows
    ]
    assert rows
    assert all("orientation_agreement" in row for row in rows)
    assert all("signed_gradient_agreement" in row for row in rows)
    assert all(0.0 <= float(row["orientation_agreement"]) <= 1.0 for row in rows)
    assert all(-1.0 <= float(row["signed_gradient_agreement"]) <= 1.0 for row in rows)


def _runtime_trace(
    image: np.ndarray,
    support: np.ndarray,
    *,
    normal: tuple[float, float] = (0.0, 1.0),
    seam_x: int | None = None,
):
    height = image.shape[0]
    return trace_s13_owner_only_component_edge(
        image,
        np.ones(image.shape[:2], bool),
        support_xy=support,
        normal_xy=normal,
        signed_gradient_polarity=1.0,
        seam_x_by_row=np.full(height, image.shape[1] // 2 if seam_x is None else seam_x),
        search_radius_px=6,
    )


def test_runtime_y_edge_trace_does_not_jump_to_stronger_physical_edge() -> None:
    image = np.zeros((56, 72, 3), np.uint8)
    image[20:] = 70
    image[34:] = 255
    support = np.asarray([(column, 20) for column in range(18, 54)], np.int32)

    trace = _runtime_trace(image, support)

    assert trace.metrics["coverage_fraction"] >= 0.95
    assert np.nanmax(trace.y_edge_px) <= 21
    assert np.nanmin(trace.y_edge_px) >= 18


def test_runtime_y_edge_trace_reports_missing_break_and_double_edge() -> None:
    clean = np.zeros((48, 64, 3), np.uint8)
    clean[18:] = 100
    clean[23:] = 220
    support = np.asarray([(column, 18) for column in range(12, 52)], np.int32)
    doubled = _runtime_trace(clean, support)
    broken_image = clean.copy()
    broken_image[:, 26:36] = 55
    broken = _runtime_trace(broken_image, support)

    assert doubled.metrics["double_edge_length_px"] >= 40
    assert broken.metrics["missing_column_count"] >= 8
    assert broken.metrics["break_length_px"] >= 8


def test_runtime_y_edge_trace_reports_slope_curvature_and_seam_crossing() -> None:
    height, width = 52, 72
    image = np.zeros((height, width, 3), np.uint8)
    points = []
    for column in range(width):
        row = 12 + column // 4
        image[row:, column] = 255
        if 14 <= column < 58:
            points.append((column, row))
    support = np.asarray(points, np.int32)
    slope = 0.25
    norm = float(np.hypot(slope, 1.0))

    trace = _runtime_trace(
        image, support, normal=(-slope / norm, 1.0 / norm), seam_x=36
    )
    aggregate = aggregate_s13_runtime_edge_traces((trace,))

    assert 0.15 <= float(trace.metrics["expected_slope_px_per_column"]) <= 0.35
    assert trace.metrics["second_difference_p95_px"] <= 1.0
    assert trace.metrics["seam_crossing_count"] == 1
    assert aggregate["evaluable_edge_columns"] >= 60
    assert aggregate["edge_p95_px"] <= 1.0


def test_runtime_y_edge_trace_clips_search_at_image_boundary() -> None:
    image = np.zeros((28, 48, 3), np.uint8)
    image[2:] = 255
    support = np.asarray([(column, 2) for column in range(8, 40)], np.int32)

    trace = _runtime_trace(image, support)

    assert trace.metrics["coverage_fraction"] >= 0.95
    assert np.nanmin(trace.y_edge_px) >= 1
    assert np.nanmax(trace.y_edge_px) <= 3


def test_candidate_correlation_and_uniqueness_are_explicit_hard_gates() -> None:
    metrics = {
        "edge_p95_px": 1.5,
        "maximum_step_px": 1.5,
        "break_length_px": 0.0,
        "double_edge_length_px": 0.0,
        "non_target_p95_px": 0.0,
    }

    evaluation = evaluate_s13_component_candidate_quality(
        baseline=metrics,
        candidate={**metrics, "edge_p95_px": 0.5, "maximum_step_px": 0.5},
        hard_gates={
            "candidate_correlation": False,
            "candidate_uniqueness": False,
        },
        config=S13M51R4Config(),
    )

    assert evaluation.decision == "rejected"
    assert evaluation.rejection_reasons == (
        "hard_gate_failed:candidate_correlation",
        "hard_gate_failed:candidate_uniqueness",
    )


def test_dependency_localization_does_not_remove_unrelated_low_utility() -> None:
    supports = {
        "a": np.asarray(((0, 0), (1, 0), (2, 0)), np.int32),
        "b": np.asarray(((1, 0), (2, 0), (3, 0)), np.int32),
        "unrelated": np.asarray(((20, 20), (21, 20)), np.int32),
    }

    def evaluate(subset: tuple[str, ...]):
        failed = {"a", "b"}.issubset(subset)
        return not failed, {"failed": failed}

    removed, audit = locate_s13_dependency_failure_subset(
        ("unrelated", "b", "a"),
        pixel_support_by_candidate=supports,
        evaluate_subset=evaluate,
        utility_by_candidate={"a": (1,), "b": (2,), "unrelated": (-100,)},
    )

    assert removed == "a"
    assert audit["localized_candidate_ids"] == ["a", "b"]
    assert "unrelated" not in audit["minimal_failure_candidate_ids"]


def test_dependency_localization_uses_leave_one_out_for_joint_failure() -> None:
    support = np.asarray(((0, 0), (1, 0)), np.int32)
    calls: list[tuple[str, ...]] = []

    def evaluate(subset: tuple[str, ...]):
        calls.append(subset)
        failed = {"a", "b"}.issubset(subset)
        return not failed, {"failed": failed}

    removed, audit = locate_s13_dependency_failure_subset(
        ("a", "b", "c"),
        pixel_support_by_candidate={"a": support, "b": support, "c": support},
        evaluate_subset=evaluate,
        utility_by_candidate={"a": (4,), "b": (1,), "c": (0,)},
    )

    assert removed == "b"
    assert audit["localization_method"] == "leave_one_out"
    assert audit["minimal_failure_candidate_ids"] == ["a", "b"]
    assert calls == [("b", "c"), ("a", "c"), ("a", "b")]


def test_dependency_localization_ddmin_is_deterministic() -> None:
    support = np.asarray(((0, 0), (1, 0)), np.int32)

    def evaluate(subset: tuple[str, ...]):
        failed = (
            {"a", "b"}.issubset(subset)
            or {"c", "d"}.issubset(subset)
        )
        return not failed, {"failed": failed}

    kwargs = {
        "pixel_support_by_candidate": {
            key: support for key in ("a", "b", "c", "d")
        },
        "evaluate_subset": evaluate,
        "utility_by_candidate": {
            "a": (2,), "b": (1,), "c": (4,), "d": (3,),
        },
    }
    first_removed, first_audit = locate_s13_dependency_failure_subset(
        ("d", "b", "a", "c"), **kwargs
    )
    second_removed, second_audit = locate_s13_dependency_failure_subset(
        ("a", "b", "c", "d"), **kwargs
    )

    assert first_removed == second_removed == "b"
    assert first_audit["localization_method"] == "deterministic_ddmin"
    assert first_audit["minimal_failure_candidate_ids"] == ["a", "b"]
    assert first_audit == second_audit
