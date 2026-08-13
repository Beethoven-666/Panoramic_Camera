from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from panorama_demo.video_s13_m51_r2 import S13M51R4Config
from panorama_demo.video_s13_m51_r4_component_chain import (
    S13ComponentApplicationSegment,
    S13ComponentSegmentCandidate,
    S13EdgeComponentObservation,
    S13SourceComponentCorrection,
    build_s13_edge_component_chains,
    build_s13_source_component_correction,
    canonical_s13_normal_search_lags,
    canonical_s13_support_sha256,
    compose_s13_source_component_deltas,
    evaluate_s13_component_candidate_quality,
    freeze_s13_baseline_c2e_obligations,
    make_s13_edge_component_observation,
    select_s13_component_patch_set,
    select_s13_component_segment_split_pair,
    solve_s13_component_segment_source_offsets,
    split_s13_edge_component_chain,
    source_map_oracle_from_arrays,
)
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


def test_obligations_are_frozen_before_chain_filtering() -> None:
    observations = (_observation(0, 0, 2.0), _observation(1, 0, 0.5))
    obligations = freeze_s13_baseline_c2e_obligations(observations)
    assert len(obligations) == 2
    assert obligations[0].severe is True
    assert obligations[1].severe is False
    assert obligations[0].scope == "runtime_detected"


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
    low = S13ComponentSegmentCandidate(
        segment, (2.0,), 1.0, (correction,), "resolved", (),
        {"rescued_severe_seam_count": 1, "worst_seam_absolute_improvement": 0.5,
         "supported_unique_edge_columns": 10, "post_maximum_step": 1.0,
         "correction_energy": 100.0},
    )
    high_correction = S13SourceComponentCorrection(
        **{**correction.__dict__, "segment_id": "higher"}
    )
    high_segment = S13ComponentApplicationSegment.create(
        parent_chain_id="chain-b", observations=(observation,)
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
    split = select_s13_component_segment_split_pair((
        {"pair_index": 8, "hard_violation_count": 1,
         "maximum_normalized_gate_excess": 0.2, "normalized_solver_residual": 1.0,
         "baseline_edge_excess_px": 2.0},
        {"pair_index": 7, "hard_violation_count": 1,
         "maximum_normalized_gate_excess": 0.2, "normalized_solver_residual": 1.0,
         "baseline_edge_excess_px": 2.0},
    ))
    assert split == 7


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
