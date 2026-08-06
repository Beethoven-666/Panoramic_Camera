from __future__ import annotations

import pytest

from panorama_demo.video_tuning import (
    COARSE_TRIAL_MAXIMUM,
    DevelopmentTrial,
    QualityComponents,
    TrialHardGates,
    ValidationCandidate,
    VideoTuningError,
    assess_development_trial,
    build_deterministic_trial_batch,
    rank_validation_candidates,
    validate_trial_batch,
)


def _quality(value: float = 1.0) -> QualityComponents:
    return QualityComponents(value, value, value, value, value, value)


def _gates(**changes: object) -> TrialHardGates:
    values: dict[str, object] = {
        "mesh_fold_count": 0,
        "object_internal_seam_count": 0,
        "global_max_gain": 1.5,
        "owner_fragment_count": 1,
        "baseline_owner_fragment_count": 1,
        "warm_3m_seconds": 15.0,
        "projected_20m_seconds": 80.0,
        "cuda_oom": False,
        "determinism_delta": 0.0,
        "determinism_limit": 0.0,
        "structural_output_complete": True,
    }
    values.update(changes)
    return TrialHardGates(**values)  # type: ignore[arg-type]


def _trial(index: int = 0, *, phase: str = "coarse") -> DevelopmentTrial:
    return DevelopmentTrial("C4", phase, index, {"mesh_cell": 16})


def test_q_uses_the_frozen_component_weights() -> None:
    quality = QualityComponents(1.0, 0.5, 0.0, 1.0, 1.0, 0.0)
    assert quality.score == pytest.approx(0.30 + 0.125 + 0.10 + 0.10)
    assert quality.as_dict()["Q"] == pytest.approx(quality.score)


def test_trial_batches_are_development_only_and_obey_coarse_and_fine_limits() -> None:
    coarse = [_trial(index) for index in range(12)]
    assert validate_trial_batch(coarse, phase="coarse") == tuple(coarse)
    with pytest.raises(VideoTuningError, match="12 through 16"):
        validate_trial_batch(coarse[:11], phase="coarse")
    fine = [_trial(index, phase="fine") for index in range(24)]
    assert len(validate_trial_batch(fine, phase="fine")) == 24
    with pytest.raises(VideoTuningError, match="1 through 24"):
        validate_trial_batch(fine + [_trial(24, phase="fine")], phase="fine")
    with pytest.raises(VideoTuningError, match="development_only"):
        DevelopmentTrial("C4", "coarse", 0, {}, evaluation_scope="holdout_only")
    assert COARSE_TRIAL_MAXIMUM == 16


def test_deterministic_trial_builder_freezes_order_and_canonical_parameter_key_order() -> None:
    trials = build_deterministic_trial_batch(
        "C4", phase="coarse", parameter_sets=[{"z": 3, "a": 1} for _ in range(12)]
    )
    assert [trial.trial_id for trial in trials] == [f"C4:coarse:{index:02d}" for index in range(12)]
    assert list(trials[0].parameters) == ["a", "z"]


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"mesh_fold_count": 1}, "mesh_fold_detected"),
        ({"object_internal_seam_count": 1}, "object_internal_seam_detected"),
        ({"global_max_gain": 1.5001}, "global_gain_exceeds_1_50"),
        ({"owner_fragment_count": 2}, "owner_fragments_exceed_baseline_allowance"),
        ({"warm_3m_seconds": 15.01}, "warm_3m_exceeds_15_seconds"),
        ({"projected_20m_seconds": 80.01}, "projected_20m_exceeds_80_seconds"),
        ({"cuda_oom": True}, "cuda_oom"),
        ({"determinism_delta": 0.01}, "determinism_limit_exceeded"),
        ({"structural_output_complete": False}, "structural_output_incomplete"),
    ],
)
def test_early_stop_gates_cannot_be_compensated_by_q(change: dict[str, object], reason: str) -> None:
    assessment = assess_development_trial(_trial(), hard_gates=_gates(**change), quality=_quality())
    assert reason in assessment.early_stop_reasons
    assert assessment.quality is None
    assert assessment.quality_score is None
    assert not assessment.eligible_for_fine_or_validation


def test_validation_rank_uses_q_then_documented_tie_breakers_and_excludes_performance_failures() -> None:
    # A is within normalized 2% of B, so it wins on faster 3 m warm time.
    faster_near_tie = ValidationCandidate("A", _quality(0.90), True, 8.0, 5, 400, 2)
    slower_near_tie = ValidationCandidate("B", _quality(0.91), True, 9.0, 2, 100, 0)
    clearly_best_q = ValidationCandidate("C", _quality(0.95), True, 12.0, 9, 900, 5)
    failed = ValidationCandidate("D", _quality(1.0), False, 1.0, 1, 1, 1)
    ranking = rank_validation_candidates([slower_near_tie, failed, faster_near_tie, clearly_best_q])
    assert [candidate.algorithm_id for candidate in ranking.ranked] == ["C", "A", "B"]
    assert ranking.selected is clearly_best_q
    assert [candidate.algorithm_id for candidate in ranking.top_three] == ["C", "A", "B"]
    assert ranking.rejected_algorithm_ids == ("D",)


def test_validation_tie_breaks_module_memory_fallback_then_id_stably() -> None:
    candidates = [
        ValidationCandidate("z", _quality(0.9), True, 8.0, 3, 10, 0),
        ValidationCandidate("y", _quality(0.9), True, 8.0, 2, 999, 9),
        ValidationCandidate("x", _quality(0.9), True, 8.0, 2, 20, 9),
        ValidationCandidate("w", _quality(0.9), True, 8.0, 2, 20, 1),
    ]
    ranking = rank_validation_candidates(candidates)
    assert [candidate.algorithm_id for candidate in ranking.ranked] == ["w", "x", "y", "z"]
    with pytest.raises(VideoTuningError, match="validation_only"):
        ValidationCandidate("holdout", _quality(), True, 1.0, 1, 1, 1, evaluation_scope="holdout_only")
