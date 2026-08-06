"""Deterministic, development-only tuning and validation ranking evidence.

This module deliberately plans and assesses evidence; it never renders a
session, invokes a holdout, or writes a production lock.  Keeping the policy
pure makes the Phase 8 decision reproducible and prevents a convenient tuning
helper from becoming a route around the one-time holdout lifecycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


QUALITY_WEIGHTS: Mapping[str, float] = {
    "line_continuity": 0.30,
    "object_integrity": 0.25,
    "seam_photometric": 0.15,
    "owner_topology": 0.10,
    "detail_preservation": 0.10,
    "flow_mesh_consistency": 0.10,
}
"""Frozen Phase 8 Q weights; components are normalized quality scores [0, 1]."""

COARSE_TRIAL_MINIMUM = 12
COARSE_TRIAL_MAXIMUM = 16
FINE_TRIAL_MAXIMUM = 24
QUALITY_TIE_DELTA = 0.02


class VideoTuningError(ValueError):
    """Evidence does not meet the immutable Phase 8 tuning contract."""


def _finite_non_negative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise VideoTuningError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class QualityComponents:
    """Six normalized quality components used to calculate Q after hard gates."""

    line_continuity: float
    object_integrity: float
    seam_photometric: float
    owner_topology: float
    detail_preservation: float
    flow_mesh_consistency: float

    def __post_init__(self) -> None:
        for name in QUALITY_WEIGHTS:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise VideoTuningError(f"{name} must be a finite normalized score in [0, 1]")

    @property
    def score(self) -> float:
        return sum(QUALITY_WEIGHTS[name] * float(getattr(self, name)) for name in QUALITY_WEIGHTS)

    def as_dict(self) -> dict[str, float]:
        return {**{name: float(getattr(self, name)) for name in QUALITY_WEIGHTS}, "Q": self.score}


@dataclass(frozen=True)
class TrialHardGates:
    """All early-stop observations required for one development trial.

    ``owner_fragment_allowance`` makes the plan's "significantly more than
    baseline" phrase explicit.  Its conservative default permits no increase.
    """

    mesh_fold_count: int
    object_internal_seam_count: int
    global_max_gain: float
    owner_fragment_count: int
    baseline_owner_fragment_count: int
    warm_3m_seconds: float
    projected_20m_seconds: float
    cuda_oom: bool
    determinism_delta: float
    determinism_limit: float
    structural_output_complete: bool
    owner_fragment_allowance: int = 0

    def __post_init__(self) -> None:
        for name in (
            "mesh_fold_count",
            "object_internal_seam_count",
            "owner_fragment_count",
            "baseline_owner_fragment_count",
            "owner_fragment_allowance",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise VideoTuningError(f"{name} must be a non-negative integer")
        for name in (
            "global_max_gain",
            "warm_3m_seconds",
            "projected_20m_seconds",
            "determinism_delta",
            "determinism_limit",
        ):
            _finite_non_negative(getattr(self, name), name=name)
        if not isinstance(self.cuda_oom, bool) or not isinstance(self.structural_output_complete, bool):
            raise VideoTuningError("cuda_oom and structural_output_complete must be booleans")

    def early_stop_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.mesh_fold_count > 0:
            reasons.append("mesh_fold_detected")
        if self.object_internal_seam_count > 0:
            reasons.append("object_internal_seam_detected")
        if self.global_max_gain > 1.50:
            reasons.append("global_gain_exceeds_1_50")
        if self.owner_fragment_count > self.baseline_owner_fragment_count + self.owner_fragment_allowance:
            reasons.append("owner_fragments_exceed_baseline_allowance")
        if self.warm_3m_seconds > 15.0:
            reasons.append("warm_3m_exceeds_15_seconds")
        if self.projected_20m_seconds > 80.0:
            reasons.append("projected_20m_exceeds_80_seconds")
        if self.cuda_oom:
            reasons.append("cuda_oom")
        if self.determinism_delta > self.determinism_limit:
            reasons.append("determinism_limit_exceeded")
        if not self.structural_output_complete:
            reasons.append("structural_output_incomplete")
        return tuple(reasons)


@dataclass(frozen=True)
class DevelopmentTrial:
    """A deterministic Phase 8 trial specification with no render authority."""

    family_id: str
    phase: str
    trial_index: int
    parameters: Mapping[str, object]
    evaluation_scope: str = "development_only"

    def __post_init__(self) -> None:
        if not isinstance(self.family_id, str) or not self.family_id:
            raise VideoTuningError("family_id must be non-empty")
        if self.phase not in {"coarse", "fine"}:
            raise VideoTuningError("phase must be coarse or fine")
        if not isinstance(self.trial_index, int) or self.trial_index < 0:
            raise VideoTuningError("trial_index must be a non-negative integer")
        if self.evaluation_scope != "development_only":
            raise VideoTuningError("tuning trials are development_only; validation and holdout are forbidden")
        if not isinstance(self.parameters, Mapping):
            raise VideoTuningError("parameters must be a mapping")

    @property
    def trial_id(self) -> str:
        return f"{self.family_id}:{self.phase}:{self.trial_index:02d}"


def validate_trial_batch(trials: Sequence[DevelopmentTrial], *, phase: str) -> tuple[DevelopmentTrial, ...]:
    """Validate deterministic trial counts and unique ids before execution."""

    if phase not in {"coarse", "fine"}:
        raise VideoTuningError("phase must be coarse or fine")
    count = len(trials)
    if phase == "coarse" and not COARSE_TRIAL_MINIMUM <= count <= COARSE_TRIAL_MAXIMUM:
        raise VideoTuningError("coarse search requires 12 through 16 development trials")
    if phase == "fine" and not 1 <= count <= FINE_TRIAL_MAXIMUM:
        raise VideoTuningError("fine search requires 1 through 24 development trials")
    if any(trial.phase != phase for trial in trials):
        raise VideoTuningError("trial batch mixes phases")
    identifiers = [trial.trial_id for trial in trials]
    if len(set(identifiers)) != len(identifiers):
        raise VideoTuningError("trial batch contains duplicate deterministic trial ids")
    return tuple(trials)


def build_deterministic_trial_batch(
    family_id: str, *, phase: str, parameter_sets: Sequence[Mapping[str, object]],
) -> tuple[DevelopmentTrial, ...]:
    """Freeze caller-declared parameter sets into index-stable development trials.

    There is intentionally no random sampler here.  A deterministic TPE/Optuna
    driver must persist its proposed parameter sets first, then pass that exact
    ordered sequence to this helper.  That preserves reproducibility without
    making Optuna a formal runtime dependency or giving this module execution
    authority.
    """

    trials = tuple(
        DevelopmentTrial(
            family_id=family_id,
            phase=phase,
            trial_index=index,
            parameters=dict(sorted(parameters.items())),
        )
        for index, parameters in enumerate(parameter_sets)
    )
    return validate_trial_batch(trials, phase=phase)


@dataclass(frozen=True)
class TrialAssessment:
    """Immutable development evidence; Q is absent when an early gate failed."""

    trial: DevelopmentTrial
    hard_gates: TrialHardGates
    quality: QualityComponents | None

    @property
    def early_stop_reasons(self) -> tuple[str, ...]:
        return self.hard_gates.early_stop_reasons()

    @property
    def eligible_for_fine_or_validation(self) -> bool:
        return not self.early_stop_reasons and self.quality is not None

    @property
    def quality_score(self) -> float | None:
        return self.quality.score if self.eligible_for_fine_or_validation else None

    def as_dict(self) -> dict[str, object]:
        return {
            "trial_id": self.trial.trial_id,
            "family_id": self.trial.family_id,
            "phase": self.trial.phase,
            "evaluation_scope": self.trial.evaluation_scope,
            "parameters": dict(self.trial.parameters),
            "early_stop_reasons": list(self.early_stop_reasons),
            "eligible_for_fine_or_validation": self.eligible_for_fine_or_validation,
            "quality": self.quality.as_dict() if self.quality is not None else None,
        }


def assess_development_trial(
    trial: DevelopmentTrial, *, hard_gates: TrialHardGates, quality: QualityComponents | None,
) -> TrialAssessment:
    """Record a trial without letting Q compensate for a hard-gate failure."""

    reasons = hard_gates.early_stop_reasons()
    if reasons and quality is not None:
        # Keeping the raw components would invite accidental score-based revival.
        quality = None
    return TrialAssessment(trial=trial, hard_gates=hard_gates, quality=quality)


@dataclass(frozen=True)
class ValidationCandidate:
    """Validation-only ranking evidence for a candidate that passed hard gates."""

    algorithm_id: str
    quality: QualityComponents
    performance_gate_passed: bool
    warm_3m_seconds: float
    module_count: int
    peak_gpu_memory_bytes: int
    fallback_complexity: int
    evaluation_scope: str = "validation_only"

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm_id, str) or not self.algorithm_id:
            raise VideoTuningError("algorithm_id must be non-empty")
        if self.evaluation_scope != "validation_only":
            raise VideoTuningError("candidate ranking only accepts validation_only evidence")
        if not isinstance(self.performance_gate_passed, bool):
            raise VideoTuningError("performance_gate_passed must be boolean")
        _finite_non_negative(self.warm_3m_seconds, name="warm_3m_seconds")
        for name in ("module_count", "peak_gpu_memory_bytes", "fallback_complexity"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise VideoTuningError(f"{name} must be a non-negative integer")

    @property
    def quality_score(self) -> float:
        return self.quality.score


@dataclass(frozen=True)
class ValidationRanking:
    """Ranked validation evidence; this is not a production selection or lock."""

    ranked: tuple[ValidationCandidate, ...]
    rejected_algorithm_ids: tuple[str, ...]

    @property
    def selected(self) -> ValidationCandidate | None:
        return self.ranked[0] if self.ranked else None

    @property
    def top_three(self) -> tuple[ValidationCandidate, ...]:
        return self.ranked[:3]


def _prefer_within_quality_tie(first: ValidationCandidate, second: ValidationCandidate) -> ValidationCandidate:
    """Apply the documented tie ordering, ending in stable algorithm identity."""

    key_first = (
        first.warm_3m_seconds,
        first.module_count,
        first.peak_gpu_memory_bytes,
        first.fallback_complexity,
        first.algorithm_id,
    )
    key_second = (
        second.warm_3m_seconds,
        second.module_count,
        second.peak_gpu_memory_bytes,
        second.fallback_complexity,
        second.algorithm_id,
    )
    return first if key_first <= key_second else second


def rank_validation_candidates(candidates: Sequence[ValidationCandidate]) -> ValidationRanking:
    """Rank A-quality validation evidence with the plan's <2% Q tie policy.

    ``QUALITY_TIE_DELTA`` is an absolute 0.02 difference because Q is normalized
    to [0, 1].  Candidates that fail performance remain explicitly rejected;
    no quality value can restore them.
    """

    unique_ids = [candidate.algorithm_id for candidate in candidates]
    if len(set(unique_ids)) != len(unique_ids):
        raise VideoTuningError("validation ranking requires unique algorithm ids")
    passing = [candidate for candidate in candidates if candidate.performance_gate_passed]
    rejected = tuple(candidate.algorithm_id for candidate in candidates if not candidate.performance_gate_passed)
    # Insertion sort makes the near-tie comparison deterministic and auditable.
    ranked: list[ValidationCandidate] = []
    for candidate in passing:
        inserted = False
        for index, current in enumerate(ranked):
            difference = abs(candidate.quality_score - current.quality_score)
            if difference < QUALITY_TIE_DELTA:
                preferred = _prefer_within_quality_tie(candidate, current)
            else:
                preferred = candidate if candidate.quality_score > current.quality_score else current
            if preferred is candidate:
                ranked.insert(index, candidate)
                inserted = True
                break
        if not inserted:
            ranked.append(candidate)
    return ValidationRanking(ranked=tuple(ranked), rejected_algorithm_ids=rejected)


__all__ = [
    "COARSE_TRIAL_MAXIMUM",
    "COARSE_TRIAL_MINIMUM",
    "FINE_TRIAL_MAXIMUM",
    "QUALITY_TIE_DELTA",
    "QUALITY_WEIGHTS",
    "DevelopmentTrial",
    "QualityComponents",
    "TrialAssessment",
    "TrialHardGates",
    "ValidationCandidate",
    "ValidationRanking",
    "VideoTuningError",
    "assess_development_trial",
    "build_deterministic_trial_batch",
    "rank_validation_candidates",
    "validate_trial_batch",
]
