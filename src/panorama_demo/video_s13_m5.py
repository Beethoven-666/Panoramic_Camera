"""M5 geometry/seam transactions and owner-only P2 rendering for S1.3."""

from __future__ import annotations

import hashlib
import json
import math
import os
from contextvars import ContextVar
import time
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .calibrated_remap import accelerated_remap, undistortion_maps
from .session import CameraIntrinsics
from .video_s12_schedule import S012Schedule, validate_s012_schedule
from .video_s13_alignment import (
    S13AlignmentCandidate,
    S13ApplicationBand,
    S13PairAlignment,
    estimate_s13_pair_alignment,
    reestimate_s13_final_corridor_alignment,
)
from .video_s13_quality import (
    SeamStructureFeatures,
    append_s13_exact_component_evidence_from_forward_context,
    prepare_seam_structure,
    long_horizontal_structure_metrics,
    long_horizontal_structure_nondegrading,
    edge_registration_visual_suspect,
    pair_edge_registration_metrics,
    seam_structure_metrics,
    sequence_structure_decision,
    structurally_non_degrading,
    symmetric_seam_structure_metrics,
)
from .video_s13_seam import (
    S13SeamCandidate,
    S13SeamResult,
    build_s13_seam_candidates,
    rank_s13_seam_candidates,
    select_s13_seam,
)
from .video_s13_replay import S13P2ReplayPair
from .video_s13_m51_r2 import S13M51R2Config, S13M51R3Config, S13M51R4Config
from .video_s13_vertical import S13VerticalSolution
from .video_s13_m51_r4_component_chain import (
    aggregate_s13_runtime_edge_traces,
    audit_s13_component_candidate_map_safety,
    audit_s13_obligation_coverage,
    SourceCorrectionRegistry,
    SourceMapOracle,
    build_s13_edge_component_chains,
    build_s13_source_component_correction,
    evaluate_s13_component_candidate_quality,
    freeze_s13_baseline_c2e_obligations,
    freeze_s13_source_correction_registry,
    make_s13_edge_component_observation,
    locate_s13_dependency_failure_subset,
    plan_s13_failed_segment_split,
    propagate_s13_edge_component_evidence,
    s13_component_segment_budget_priority,
    S13ComponentSegmentCandidate,
    S13ComponentQualityEvaluation,
    S13ComponentPatchSet,
    S13ComponentSolverError,
    select_s13_component_patch_set,
    solve_s13_component_segment_source_offsets,
    source_map_oracle_from_arrays,
    split_s13_edge_component_chain,
    trace_s13_owner_only_component_edge,
    compose_s13_source_component_deltas,
)
from .video_s13_hard_audit import long_horizontal_structure_catastrophe_guard


S13SourceMapProvider = Callable[
    [int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]
S13SampledSourceProvider = Callable[
    [int, int, int], tuple[np.ndarray, np.ndarray, bool]
]


_C2E_STRUCTURAL_APPLICATION_GATES = frozenset({
    "offset_bounds",
    "owner_support",
    "formal_owner_retention",
    "evidence_retention",
    "final_inverse_map",
    "individual_segment_gates",
    "exclusive_correction_fields",
    "composite_map_safety",
})


def _evaluate_s13_runtime_component_candidate(
    *,
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    hard_gates: Mapping[str, bool],
    config: S13M51R4Config,
) -> S13ComponentQualityEvaluation:
    """Apply the configured C2E policy without weakening map safety.

    ``all_structurally_safe`` is an explicit manual visual-acceptance policy.
    It records the normal automatic quality decision, but only finite/bounds,
    owner-retention, evidence-retention and final inverse-map gates retain veto
    authority.  Correlation, uniqueness, trace coverage and measured visual
    improvement remain audit evidence rather than application gates.
    """

    try:
        automatic = evaluate_s13_component_candidate_quality(
            baseline=baseline,
            candidate=candidate,
            hard_gates=hard_gates,
            config=config,
        )
    except ValueError as exc:
        if config.application_policy != "all_structurally_safe":
            raise
        automatic = S13ComponentQualityEvaluation(
            "rejected",
            ("automatic_quality_unevaluable",),
            {
                "hard_gate_failures": (),
                "automatic_quality_error_type": type(exc).__name__,
                "automatic_quality_error": str(exc),
            },
        )
    if config.application_policy != "all_structurally_safe":
        return automatic
    failed_structural = tuple(sorted(
        name
        for name in _C2E_STRUCTURAL_APPLICATION_GATES
        if name in hard_gates and hard_gates.get(name) is not True
    ))
    audit = {
        **dict(automatic.audit),
        "application_policy": "all_structurally_safe",
        "automatic_quality_decision": automatic.decision,
        "automatic_quality_rejection_reasons": list(automatic.rejection_reasons),
        "quality_gates_runtime_authority": False,
        "structural_gate_failures": list(failed_structural),
    }
    if failed_structural:
        return S13ComponentQualityEvaluation(
            "rejected",
            tuple(f"hard_gate_failed:{name}" for name in failed_structural),
            audit,
        )
    return S13ComponentQualityEvaluation("resolved", (), audit)


def _component_trace_metrics(
    observations: Sequence[object], *, config: S13M51R4Config
) -> dict[str, float | bool]:
    lags = np.asarray([
        0.5 * (
            float(row.forward_best_lag_px) - float(row.reverse_best_lag_px)
        )
        for row in observations
    ], dtype=np.float64)
    if lags.size == 0 or not np.isfinite(lags).all():
        raise ValueError("component ROI has no finite edge trace")
    break_length = 0.0
    double_edge_length = 0.0
    guard_steps = []
    for row, lag in zip(observations, lags, strict=True):
        x0, y0, x1, y1 = row.global_bbox_xyxy
        trace_length = float(max(x1 - x0, y1 - y0))
        if (
            int(row.forward_valid_samples) < 2
            or int(row.reverse_valid_samples) < 2
            or float(row.correlation) < config.minimum_c2e_correlation
        ):
            break_length += trace_length
        if float(row.uniqueness_fraction) < config.minimum_c2e_uniqueness_fraction:
            double_edge_length += trace_length
        if abs(float(lag)) <= config.maximum_post_edge_p95_px:
            guard_steps.append(abs(float(lag)))
    absolute = np.abs(lags)
    return {
        "edge_p50_px": float(np.percentile(absolute, 50.0)),
        "edge_p95_px": float(np.percentile(absolute, 95.0)),
        "maximum_step_px": float(np.max(absolute)),
        "break_length_px": break_length,
        "double_edge_length_px": double_edge_length,
        "non_target_p95_px": (
            float(np.percentile(guard_steps, 95.0)) if guard_steps else 0.0
        ),
        "search_boundary_hit": any(row.search_boundary_hit for row in observations),
    }


def _select_s13_component_gain_candidate(
    candidates: Sequence[S13ComponentSegmentCandidate],
) -> S13ComponentSegmentCandidate | None:
    """Apply the frozen per-segment lexicographic gain selection policy."""

    passing = [
        row for row in candidates
        if row.decision in {"resolved", "improved_unresolved"}
    ]
    if not passing:
        return None
    decision_rank = {"resolved": 0, "improved_unresolved": 1}
    best_decision = min(decision_rank[row.decision] for row in passing)
    eligible = [row for row in passing if decision_rank[row.decision] == best_decision]
    best_step = min(
        float(row.audit["candidate_metrics"]["maximum_step_px"]) for row in eligible
    )
    near_step = [
        row for row in eligible
        if float(row.audit["candidate_metrics"]["maximum_step_px"]) <= best_step + 0.15
    ]
    best_p95 = min(
        float(row.audit["candidate_metrics"]["edge_p95_px"]) for row in near_step
    )
    near_p95 = [
        row for row in near_step
        if float(row.audit["candidate_metrics"]["edge_p95_px"]) <= best_p95 + 0.15
    ]
    best_break = min(
        float(row.audit["candidate_metrics"]["break_length_px"])
        + float(row.audit["candidate_metrics"]["double_edge_length_px"])
        for row in near_p95
    )
    near_break = [
        row for row in near_p95
        if float(row.audit["candidate_metrics"]["break_length_px"])
        + float(row.audit["candidate_metrics"]["double_edge_length_px"])
        == best_break
    ]
    return min(
        near_break,
        key=lambda row: (float(row.gain), float(row.audit["correction_energy"])),
    )


@dataclass(frozen=True)
class S13M5Pair:
    transaction: Mapping[str, object]
    seam_x_by_row: np.ndarray
    alignment: S13PairAlignment | None
    component_evidence: tuple[tuple[object, object], ...] = ()
    component_forward_probe: tuple[object, ...] | None = None
    component_propagation_audit: Mapping[str, object] | None = None
    # Kept in memory for the immediate M6 C2E decision only.  These are the
    # already generated compact corridor seam paths, never a disk replay.
    c2e_seam_candidates: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class S13P2Result:
    image: np.ndarray
    valid_mask: np.ndarray
    pixel_provenance: dict[str, np.ndarray]
    remap_invocations: int
    decoded_frame_ids: tuple[int, ...]
    expected_support_mask: np.ndarray
    actual_remap_invocations: int = 0


@dataclass(frozen=True)
class S13M5Result:
    pairs: tuple[S13M5Pair, ...]
    replay_pairs: tuple[S13P2ReplayPair, ...]
    geometry_result: S13P2Result
    final_result: S13P2Result
    seam_overlay: np.ndarray
    hard_audit_passed: bool
    hard_audit: Mapping[str, object]
    diagnostic_quality: Mapping[str, object]
    selected_as_best: bool
    before_mean_score: float | None
    after_mean_score: float | None
    selection_audit: Mapping[str, object]
    performance: Mapping[str, float]
    source_map_oracles: tuple[SourceMapOracle, ...] = ()
    base_source_map_oracles: tuple[SourceMapOracle, ...] = ()
    component_chain_audit: Mapping[str, object] | None = None
    component_patch_set: S13ComponentPatchSet | None = None
    source_correction_registry: SourceCorrectionRegistry | None = None
    estimation_result: S13M5EstimationResult | None = None


@dataclass(frozen=True)
class S13M51R4EvidenceContext:
    raw_rgb_by_frame: Mapping[int, np.ndarray]
    component_forward_probes: Mapping[int, tuple[object, ...]]
    stage_order: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.raw_rgb_by_frame, MappingProxyType):
            raise TypeError("S1.3 R4 raw cache authority must be immutable")
        if not isinstance(self.component_forward_probes, MappingProxyType):
            raise TypeError("S1.3 R4 Sobel/evidence cache authority must be immutable")
        if any(image.flags.writeable for image in self.raw_rgb_by_frame.values()):
            raise ValueError("S1.3 R4 raw cache views must be read-only")
        if self.stage_order != (
            "pair_estimation", "topology_repair", "component_rescue",
            "source_map_oracle_freeze", "formal_render",
        ):
            raise ValueError("S1.3 R4 estimation/render phase order is invalid")


@dataclass(frozen=True)
class S13M5EstimationResult:
    pairs: tuple[S13M5Pair, ...]
    source_correction_registry: SourceCorrectionRegistry
    component_patch_set: S13ComponentPatchSet
    component_chain_audit: Mapping[str, object]
    evidence_context: S13M51R4EvidenceContext
    base_source_map_oracles: tuple[SourceMapOracle, ...]
    final_source_map_oracles: tuple[SourceMapOracle, ...]
    performance_counters: Mapping[str, int | float]
    expected_support_mask: np.ndarray
    expected_support_audit: Mapping[str, object]


@dataclass(frozen=True)
class S13PairCorrespondences:
    reference_xy: np.ndarray
    moving_xy: np.ndarray
    audit: Mapping[str, object]

    def __iter__(self):
        """Keep legacy internal two-value unpacking while exposing the audit."""

        yield self.reference_xy
        yield self.moving_xy


@dataclass(frozen=True)
class S13M5PairInput:
    """Read-only base-corridor state for one independently evaluable pair."""

    pair_index: int
    left_frame_id: int
    right_frame_id: int
    corridor_x0: int
    corridor_x1: int
    left_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    right_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    def __post_init__(self) -> None:
        if self.corridor_x1 <= self.corridor_x0:
            raise ValueError("S1.3 M5 pair corridor must be non-empty")
        expected = (len(self.left_maps), len(self.right_maps))
        if expected != (4, 4):
            raise ValueError("S1.3 M5 pair input requires four map arrays per side")
        shape = (self.left_maps[0].shape, self.right_maps[0].shape)
        if shape[0] != shape[1] or shape[0][1] != self.corridor_x1 - self.corridor_x0:
            raise ValueError("S1.3 M5 pair input map shape disagrees with its corridor")


def _freeze_m5_pair_input(
    *, pair_index: int, left_frame_id: int, right_frame_id: int, corridor_x0: int,
    corridor_x1: int, left_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    right_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> S13M5PairInput:
    """Make the maps explicit read-only inputs before any CPU pair evaluation."""

    for array in (*left_maps, *right_maps):
        np.asarray(array).setflags(write=False)
    return S13M5PairInput(
        pair_index, left_frame_id, right_frame_id, corridor_x0, corridor_x1, left_maps, right_maps,
    )


_RUNTIME_UNSEALED = ContextVar("s13_m5_runtime_unsealed", default=False)
_RUNTIME_RESIDENT_REMAP = ContextVar("s13_m5_resident_remap", default=None)
_RUNTIME_RESIDENT_BATCH = ContextVar("s13_m5_resident_batch", default=None)


def set_s13_m5_runtime_unsealed(enabled: bool):
    """Disable legacy transaction digests for the no-artifact fast path."""
    return _RUNTIME_UNSEALED.set(bool(enabled))


def reset_s13_m5_runtime_unsealed(token: object) -> None:
    _RUNTIME_UNSEALED.reset(token)


def set_s13_m5_resident_remap(remap: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None):
    """Bind the CUDA-resident compact-corridor sampler for one in-memory run."""
    return _RUNTIME_RESIDENT_REMAP.set(remap)


def reset_s13_m5_resident_remap(token: object) -> None:
    _RUNTIME_RESIDENT_REMAP.reset(token)


def set_s13_m5_resident_batch(
    sampler: Callable[[Sequence[tuple[int, np.ndarray, np.ndarray, np.ndarray]]], tuple[np.ndarray, ...]] | None,
):
    """Bind one bounded CUDA batch sampler for independent base corridors."""
    return _RUNTIME_RESIDENT_BATCH.set(sampler)


def reset_s13_m5_resident_batch(token: object) -> None:
    _RUNTIME_RESIDENT_BATCH.reset(token)


def _sha_array(*arrays: np.ndarray) -> str:
    if _RUNTIME_UNSEALED.get():
        return "runtime-unsealed"
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sha_json(value: Mapping[str, object]) -> str:
    if _RUNTIME_UNSEALED.get():
        return "runtime-unsealed"
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


_V5_TRANSACTION_FLOAT_DECIMAL_PLACES = 6


def _canonicalize_v5_transaction_value(
    value: object,
    *,
    path: str = "$",
    nonfinite_as_none: bool = False,
) -> object:
    """Return the frozen JSON value representation used by M5.1-r2/v5.

    Candidate selection consumes the full-precision in-memory metrics before
    this function is called.  Quantization therefore affects only the sealed
    audit representation and its digest, not pixels, thresholds, or ranking.
    """

    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_v5_transaction_value(
                item,
                path=f"{path}.{key}",
                nonfinite_as_none=nonfinite_as_none,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonicalize_v5_transaction_value(
                item,
                path=f"{path}[{index}]",
                nonfinite_as_none=nonfinite_as_none,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            if nonfinite_as_none:
                return None
            raise ValueError(
                f"S1.3 v5 transaction floats must be finite at {path}"
            )
        rounded = round(number, _V5_TRANSACTION_FLOAT_DECIMAL_PLACES)
        return 0.0 if rounded == 0.0 else rounded
    return value


def _finalize_v5_transaction(value: Mapping[str, object]) -> dict[str, object]:
    """Canonicalize a v5 transaction and bind its normalized content hash."""

    without_digest = dict(value)
    without_digest.pop("result_stage_sha256", None)
    normalized = _canonicalize_v5_transaction_value(without_digest)
    if not isinstance(normalized, dict):  # pragma: no cover - Mapping guarantees this
        raise TypeError("S1.3 v5 transaction must be a mapping")
    normalized["result_stage_sha256"] = _sha_json(normalized)
    return normalized


def _successor_transaction_schema(config: S13M51R2Config | None) -> str:
    if isinstance(config, S13M51R4Config):
        return "gemini305-video-s13-m5-pair-transaction/v5"
    if isinstance(config, S13M51R3Config):
        return "gemini305-video-s13-m5-pair-transaction/v4"
    if config is not None and config.enabled:
        return "gemini305-video-s13-m5-pair-transaction/v3"
    return "gemini305-video-s13-m5-pair-transaction/v2"


def _edge_visual_suspect_for_pair(
    edge_registration: Mapping[str, object],
    seam_local_lk_p95_px: float | None,
    *,
    config: S13M51R2Config,
    component_local_authority: bool,
) -> bool:
    """Keep v5 selection authority outside the explicitly scoped v6 pair."""

    if isinstance(config, S13M51R3Config) and not component_local_authority:
        legacy = edge_registration.get("legacy_pair_decision")
        if not isinstance(legacy, Mapping) or legacy.get("evaluable") is not True:
            return False
        if legacy.get("multiple_layer_or_ambiguous") is True:
            return False

        def exceeds(value: object, threshold: float) -> bool:
            return (
                isinstance(value, (int, float))
                and math.isfinite(float(value))
                and float(value) > threshold
            )

        return bool(
            exceeds(seam_local_lk_p95_px, config.suspect_lk_p95_px)
            or exceeds(
                legacy.get("median_supported_abs_lag_px"),
                config.suspect_edge_median_px,
            )
            or exceeds(
                legacy.get("p95_supported_abs_lag_px"),
                config.suspect_edge_p95_px,
            )
        )
    return edge_registration_visual_suspect(
        edge_registration,
        seam_local_lk_p95_px=seam_local_lk_p95_px,
        config=config,
    )


def _base_calibrated_map(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    source_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assignment = schedule.assignments[source_index]
    height, width = schedule.canvas_height, schedule.canvas_width
    canvas_x = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], (height, width))
    canvas_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width))
    calibrated_x = float(calibration.cx) + canvas_x - float(assignment.center_x)
    calibrated_y = canvas_y
    inverse = undistortion_maps(calibration)
    if inverse is None:
        source_u, source_v = calibrated_x, calibrated_y
    else:
        source_u = cv2.remap(inverse[0], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        source_v = cv2.remap(inverse[1], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (
        np.isfinite(source_u) & np.isfinite(source_v)
        & (source_u >= 0.0) & (source_u <= int(calibration.width) - 1)
        & (source_v >= 0.0) & (source_v <= int(calibration.height) - 1)
    )
    return source_u.astype(np.float32), source_v.astype(np.float32), valid


def _map_crop(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    source_index: int,
    x0: int,
    x1: int,
    global_vertical_offset: float,
    candidate: S13AlignmentCandidate | None,
    component_corrections: tuple[object, ...] = (),
    component_field_ids: Mapping[str, int] | None = None,
    vertical_parent: S13VerticalSolution | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assignment = schedule.assignments[source_index]
    height = schedule.canvas_height
    canvas_x = np.broadcast_to(np.arange(x0, x1, dtype=np.float32)[None, :], (height, x1 - x0))
    canvas_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], canvas_x.shape)
    delta_u = np.zeros(canvas_x.shape, dtype=np.float32)
    delta_v = np.zeros(canvas_x.shape, dtype=np.float32)
    if vertical_parent is not None and source_index > 0:
        if len(vertical_parent.local_row_residuals) != len(schedule.assignments) - 1:
            raise ValueError("S1.3 P1 parent residuals do not cover every source handoff")
        parent_pair = vertical_parent.pairs[source_index - 1]
        parent_width = min(
            int(parent_pair.application_right_x - parent_pair.application_left_x),
            int(assignment.width),
        )
        parent_x0 = int(assignment.left_x)
        parent_x1 = parent_x0 + max(0, parent_width)
        overlap0, overlap1 = max(x0, parent_x0), min(x1, parent_x1)
        if overlap1 > overlap0:
            taper = np.linspace(
                1.0, 0.0, parent_width, endpoint=True, dtype=np.float32
            )
            rows = np.asarray(
                vertical_parent.local_row_residuals[source_index - 1],
                dtype=np.float32,
            )
            if rows.shape != (height,) or not np.isfinite(rows).all():
                raise ValueError("S1.3 P1 parent residual is not a finite H-vector")
            target = np.s_[:, overlap0 - x0:overlap1 - x0]
            parent_columns = np.s_[overlap0 - parent_x0:overlap1 - parent_x0]
            delta_v[target] = -rows[:, None] * taper[parent_columns][None, :]
    if candidate is not None and candidate.model != "C0_identity":
        overlap0, overlap1 = max(x0, candidate.x0), min(x1, candidate.x1)
        if overlap1 > overlap0:
            target = np.s_[:, overlap0 - x0:overlap1 - x0]
            source = np.s_[:, overlap0 - candidate.x0:overlap1 - candidate.x0]
            delta_u[target] = candidate.target_delta_u[source]
            delta_v[target] = candidate.target_delta_v[source]
    field_id = np.full(canvas_x.shape, -1, dtype=np.int32)
    if component_corrections:
        delta_u, delta_v, field_id = compose_s13_source_component_deltas(
            base_delta_u=delta_u,
            base_delta_v=delta_v,
            domain_xyxy=(x0, 0, x1, height),
            corrections=component_corrections,
            field_ids=component_field_ids,
        )
    calibrated_x = float(calibration.cx) + canvas_x - float(assignment.center_x) + delta_u
    calibrated_y = canvas_y - float(global_vertical_offset) + delta_v
    inverse = undistortion_maps(calibration)
    if inverse is None:
        source_u, source_v = calibrated_x, calibrated_y
    else:
        source_u = cv2.remap(inverse[0], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
        source_v = cv2.remap(inverse[1], calibrated_x, calibrated_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    valid = (
        np.isfinite(source_u) & np.isfinite(source_v)
        & (source_u >= 0.0) & (source_u <= int(calibration.width) - 1)
        & (source_v >= 0.0) & (source_v <= int(calibration.height) - 1)
    )
    return source_u.astype(np.float32), source_v.astype(np.float32), valid, np.asarray(
        field_id, dtype=np.int32
    )


def _sample_crop(
    raw: np.ndarray,
    maps: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    u, v, valid = maps[:3]
    resident_remap = _RUNTIME_RESIDENT_REMAP.get()
    sampled = (
        resident_remap(raw, u, v)
        if resident_remap is not None else accelerated_remap(
            raw, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
    )
    return sampled, valid


def _sample_m5_base_pair(
    pair: S13M5PairInput,
    raw_loader: Callable[[int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample the independent left/right base corridors in one bounded batch."""

    batch = _RUNTIME_RESIDENT_BATCH.get()
    if batch is None:
        left, left_valid = _sample_crop(raw_loader(pair.left_frame_id), pair.left_maps)
        right, right_valid = _sample_crop(raw_loader(pair.right_frame_id), pair.right_maps)
        return left, left_valid, right, right_valid
    left, right = batch((
        (pair.left_frame_id, raw_loader(pair.left_frame_id), pair.left_maps[0], pair.left_maps[1]),
        (pair.right_frame_id, raw_loader(pair.right_frame_id), pair.right_maps[0], pair.right_maps[1]),
    ))
    return left, pair.left_maps[2], right, pair.right_maps[2]


def _sample_m5_right_candidate(
    *,
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    source_index: int,
    x0: int,
    x1: int,
    global_vertical_offset: float,
    candidate: S13AlignmentCandidate,
    vertical_parent: S13VerticalSolution | None,
    raw_image: np.ndarray,
    base_maps: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    base_image: np.ndarray,
    base_valid: np.ndarray,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
    np.ndarray,
    bool,
]:
    """Reuse the exact base sample for the explicitly map-neutral C0 model."""

    if candidate.model == "C0_identity":
        return base_maps, base_image, base_valid, True
    maps = _map_crop(
        schedule,
        calibration,
        source_index,
        x0,
        x1,
        global_vertical_offset,
        candidate,
        vertical_parent=vertical_parent,
    )
    sampled, valid = _sample_crop(raw_image, maps)
    return maps, sampled, valid, False


def _pair_correspondences(
    left: np.ndarray,
    right: np.ndarray,
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    *,
    x_offset: int,
    config: S13M51R2Config | None = None,
) -> S13PairCorrespondences:
    settings = config or S13M51R2Config()
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    mask = (left_valid & right_valid).astype(np.uint8) * 255
    points = cv2.goodFeaturesToTrack(left_gray, 400, 0.01, 4.0, mask=mask, blockSize=5)
    empty = np.empty((0, 2), np.float64)
    empty_audit: dict[str, object] = {
        "detected_count": 0,
        "status_count": 0,
        "finite_count": 0,
        "target_in_bounds_count": 0,
        "target_valid_count": 0,
        "error_filtered_count": 0,
        "final_count": 0,
        "error_median": None,
        "error_mad": None,
        "error_limit": None,
        "raw_error_count": 0,
        "raw_error_median": None,
        "raw_error_mad": None,
        "raw_error_p50": None,
        "raw_error_p90": None,
        "raw_error_p95": None,
        "raw_error_p99": None,
        "raw_error_maximum": None,
        "filter_enabled": bool(settings.enabled),
        "legacy_decision_preserved": not settings.enabled,
        "gftt_call_count": 1,
        "forward_pyr_lk_call_count": 0,
        "backward_pyr_lk_call_count": 0,
    }
    if points is None:
        return S13PairCorrespondences(empty.copy(), empty.copy(), empty_audit)
    moved, status, error = cv2.calcOpticalFlowPyrLK(
        left_gray, right_gray, points, None, winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    empty_audit["detected_count"] = int(np.asarray(points).reshape(-1, 2).shape[0])
    empty_audit["forward_pyr_lk_call_count"] = 1
    if moved is None or status is None:
        return S13PairCorrespondences(empty.copy(), empty.copy(), empty_audit)
    source = np.asarray(points).reshape(-1, 2).astype(np.float64)
    target = np.asarray(moved).reshape(-1, 2).astype(np.float64)
    status_values = np.asarray(status).reshape(-1) > 0
    if len(source) != len(target) or len(source) != len(status_values):
        raise ValueError("S1.3 M5 PyrLK result count disagrees with detected points")
    finite_values = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    legacy_keep = status_values & finite_values

    target_in_bounds = np.zeros(len(source), dtype=bool)
    target_in_bounds[legacy_keep] = (
        (target[legacy_keep, 0] >= 0.0)
        & (target[legacy_keep, 0] <= right_valid.shape[1] - 1.0)
        & (target[legacy_keep, 1] >= 0.0)
        & (target[legacy_keep, 1] <= right_valid.shape[0] - 1.0)
    )
    target_valid = np.zeros(len(source), dtype=bool)
    indexable = legacy_keep & target_in_bounds
    if np.any(indexable):
        # Finite/in-bounds is deliberately established before round/cast/index.
        tx = np.rint(target[indexable, 0]).astype(np.int32)
        ty = np.rint(target[indexable, 1]).astype(np.int32)
        target_valid[indexable] = np.asarray(right_valid, dtype=bool)[ty, tx]
    hard_keep = legacy_keep & target_in_bounds & target_valid

    error_values = None if error is None else np.asarray(error).reshape(-1).astype(np.float64)
    if error_values is not None and len(error_values) != len(source):
        raise ValueError("S1.3 M5 PyrLK error count disagrees with detected points")
    raw_error = (
        np.empty(0, dtype=np.float64)
        if error_values is None
        else error_values[legacy_keep & np.isfinite(error_values)]
    )
    distribution: dict[str, object] = {
        "raw_error_count": int(raw_error.size),
        "raw_error_median": None,
        "raw_error_mad": None,
        "raw_error_p50": None,
        "raw_error_p90": None,
        "raw_error_p95": None,
        "raw_error_p99": None,
        "raw_error_maximum": None,
    }
    if raw_error.size and settings.collect_lk_error_statistics:
        raw_median = float(np.median(raw_error))
        distribution.update({
            "raw_error_median": raw_median,
            "raw_error_mad": float(np.median(np.abs(raw_error - raw_median))),
            "raw_error_p50": float(np.percentile(raw_error, 50.0)),
            "raw_error_p90": float(np.percentile(raw_error, 90.0)),
            "raw_error_p95": float(np.percentile(raw_error, 95.0)),
            "raw_error_p99": float(np.percentile(raw_error, 99.0)),
            "raw_error_maximum": float(np.max(raw_error)),
        })

    final_keep = legacy_keep.copy()
    error_median = error_mad = error_limit = None
    error_filtered_count = int(hard_keep.sum())
    if settings.enabled:
        final_keep = hard_keep.copy()
        valid_error = (
            np.empty(0, dtype=np.float64)
            if error_values is None
            else error_values[hard_keep & np.isfinite(error_values)]
        )
        if valid_error.size >= settings.minimum_correspondence_count_for_error_filter:
            error_median = float(np.median(valid_error))
            error_mad = float(np.median(np.abs(valid_error - error_median)))
            sigma = 1.4826 * error_mad
            assert settings.lk_error_mad_floor is not None
            assert settings.lk_error_absolute_maximum is not None
            robust_limit = error_median + settings.lk_error_sigma_multiplier * max(
                sigma, settings.lk_error_mad_floor
            )
            error_limit = float(min(settings.lk_error_absolute_maximum, robust_limit))
            error_keep = np.isfinite(error_values) & (error_values <= error_limit)  # type: ignore[operator]
            final_keep &= error_keep
            error_filtered_count = int(final_keep.sum())

    selected_source = source[final_keep].copy()
    selected_target = target[final_keep].copy()
    selected_source[:, 0] += int(x_offset)
    selected_target[:, 0] += int(x_offset)
    audit: dict[str, object] = {
        "detected_count": int(len(source)),
        "status_count": int(status_values.sum()),
        "finite_count": int(legacy_keep.sum()),
        "target_in_bounds_count": int(indexable.sum()),
        "target_valid_count": int(hard_keep.sum()),
        "error_filtered_count": error_filtered_count,
        "final_count": int(final_keep.sum()),
        "error_median": error_median,
        "error_mad": error_mad,
        "error_limit": error_limit,
        **distribution,
        "filter_enabled": bool(settings.enabled),
        "legacy_decision_preserved": not settings.enabled,
        "gftt_call_count": 1,
        "forward_pyr_lk_call_count": 1,
        "backward_pyr_lk_call_count": 0,
    }
    return S13PairCorrespondences(selected_source, selected_target, audit)


def _pair_domain(schedule: S012Schedule, pair_index: int, half_width: int = 48) -> tuple[int, int]:
    boundary = schedule.boundaries[pair_index + 1]
    return max(0, boundary - half_width), min(schedule.canvas_width, boundary + half_width)


def _allowed_application_bounds(schedule: S012Schedule, pair_index: int) -> tuple[int, int]:
    boundary = schedule.boundaries[pair_index + 1]
    previous = schedule.boundaries[pair_index] if pair_index > 0 else 0
    following = schedule.boundaries[pair_index + 2] if pair_index + 2 < len(schedule.boundaries) else schedule.canvas_width
    domain_left = int(math.ceil(0.5 * (previous + boundary)))
    domain_right = int(math.floor(0.5 * (boundary + following))) + 1
    left = max(domain_left, boundary - 8)
    right = min(domain_right, boundary + 9)
    if right - left < 3:
        left, right = max(0, boundary - 1), min(schedule.canvas_width, boundary + 2)
    return left, right


def _fallback_pair(
    schedule: S012Schedule,
    pair_index: int,
    parent_sha: str,
    frame_ids: tuple[int, int],
    reason: str,
    parent_result_sha256: str | None = None,
    p0_ancestor_completion_sha256: str | None = None,
    before: Mapping[str, object] | None = None,
    after: Mapping[str, object] | None = None,
    correspondence_filter: Mapping[str, object] | None = None,
    m51_r2_config: S13M51R2Config | None = None,
    topology_repair_used: bool = False,
) -> S13M5Pair:
    seam = np.full(schedule.canvas_height, schedule.boundaries[pair_index + 1], dtype=np.int32)
    core: dict[str, object] = {
        "schema": _successor_transaction_schema(m51_r2_config),
        "transaction_id": f"m5-pair-{pair_index:04d}",
        "parent_stage": "P1",
        "parent_stage_sha256": parent_sha,
        "parent_completion_sha256": parent_sha,
        "parent_result_sha256": parent_result_sha256,
        "p0_ancestor_completion_sha256": p0_ancestor_completion_sha256,
        "pair_frame_ids": list(frame_ids),
        "motion_hypothesis_id": -1,
        "support_mask_sha256": None,
        "geometry_model": "C1_accepted_vertical_parent",
        "seam_model": "S0_midpoint_straight",
        "map_delta_sha256": None,
        "decision": "rolled_back",
        "selection_policy": "minimum_local_objective_among_hard_safe_candidates",
        "selected_candidate_id": f"pair-{pair_index:04d}-S0-fallback",
        "selected_seam_model": "midpoint_straight",
        "selected_geometry_model": "C0_identity",
        "pair_hard_audit_passed": True,
        "fallback_used": True,
        "fallback_reason": reason,
        "candidate_generation": [],
        "candidate_evaluations": [],
        "before_metrics": dict(before) if before is not None else {"evaluable": False, "reason": reason, "score": None},
        "candidate_after_metrics": (
            dict(after) if after is not None else {"evaluable": False, "reason": reason, "score": None}
        ),
        "after_metrics": (
            dict(before) if before is not None else {"evaluable": False, "reason": reason, "score": None}
        ),
        "rollback_reason": reason,
        "final_corridor_geometry_reestimated_from_immutable_p0": False,
        "comparison_coordinate_policy": (
            "same_owner_geometry_plus_symmetric_base_candidate_seam"
        ),
    }
    if correspondence_filter is not None:
        core["correspondence_filter"] = dict(correspondence_filter)
    if m51_r2_config is not None and m51_r2_config.enabled:
        core.update({
            "correspondence_filter": dict(correspondence_filter or {}),
            "seam_local_evidence": {
                "requested_half_widths_px": list(m51_r2_config.evidence_half_widths_px),
                "selected_half_width_px": None, "raw_count": 0,
                "selected_count": 0, "minimum_required_count": 36,
                "sufficient_for_train_held_out": False,
                "full_shoulder_fallback_used": False,
                "mode": "seam_local_insufficient",
            },
            "edge_registration": {
                "evaluable": False, "visual_suspect": False,
                "supported_block_count": 0, "supported_component_count": 0,
                "median_supported_abs_lag_px": None,
                "p95_supported_abs_lag_px": None, "ambiguous": False,
            },
            "selection_continued_for_structure": False,
            "selected_seam_rank": 0, "selected_geometry_rank": 0,
            "seam_fallback_used": True, "geometry_fallback_used": True,
            "topology_repair_used": bool(topology_repair_used),
            "unresolved_oblique_structure": False,
            "unexpected_exception_fallback": False,
            "micro_rescue": {"enabled": False, "attempted": False, "accepted": False},
            "c2e": {"enabled": False, "attempted": False, "accepted": False},
            "complete_seam_reassessment_diagnostic": False,
        })
    if m51_r2_config is not None and m51_r2_config.enabled:
        core = _finalize_v5_transaction(core)
    else:
        core["result_stage_sha256"] = _sha_json(core)
    return S13M5Pair(core, seam, None, c2e_seam_candidates=(seam.copy(),))


def _compose_pair_preview(left: np.ndarray, right: np.ndarray, seam_local: np.ndarray) -> np.ndarray:
    height, width = left.shape[:2]
    x = np.arange(width, dtype=np.int32)[None, :]
    owner_right = x >= seam_local[:, None]
    return np.where(owner_right[..., None], right, left).astype(np.uint8)


def _audit_rendered_seam_change(
    before_image: np.ndarray,
    after_image: np.ndarray,
    before_seam_x_by_row: np.ndarray,
    after_seam_x_by_row: np.ndarray,
    *,
    before_features: SeamStructureFeatures | None = None,
    after_features: SeamStructureFeatures | None = None,
) -> dict[str, object]:
    """Audit two rendered owner topologies on both paths and held-out shoulders."""

    before_seam = np.asarray(before_seam_x_by_row, dtype=np.int32)
    after_seam = np.asarray(after_seam_x_by_row, dtype=np.int32)
    cached_before = before_features or prepare_seam_structure(before_image)
    cached_after = after_features or prepare_seam_structure(after_image)
    if cached_before is None or cached_after is None:
        raise ValueError("S1.3 seam audit images cannot build structure features")
    symmetric_before, symmetric_after = symmetric_seam_structure_metrics(
        cached_before,
        cached_after,
        before_seam,
        after_seam,
    )
    symmetric_ok, symmetric_reason = structurally_non_degrading(
        symmetric_before, symmetric_after
    )
    path_audits: dict[str, object] = {}
    paths_ok = True
    first_reason: str | None = None
    for name, path in (("before_owner", before_seam), ("after_owner", after_seam)):
        before_metrics = seam_structure_metrics(cached_before, path)
        after_metrics = seam_structure_metrics(cached_after, path)
        structure_ok, structure_reason = structurally_non_degrading(
            before_metrics, after_metrics
        )
        before_horizontal = long_horizontal_structure_metrics(cached_before, path)
        after_horizontal = long_horizontal_structure_metrics(cached_after, path)
        horizontal_ok, horizontal_reason = long_horizontal_structure_nondegrading(
            before_horizontal, after_horizontal
        )
        path_ok = bool(structure_ok and horizontal_ok)
        if not path_ok and first_reason is None:
            first_reason = structure_reason if not structure_ok else horizontal_reason
        paths_ok = paths_ok and path_ok
        path_audits[name] = {
            "eligible": path_ok,
            "reason": None if path_ok else (
                structure_reason if not structure_ok else horizontal_reason
            ),
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "before_horizontal_continuity": before_horizontal,
            "after_horizontal_continuity": after_horizontal,
        }
    eligible = bool(symmetric_ok and paths_ok)
    reason = None if eligible else (symmetric_reason if not symmetric_ok else first_reason)
    return {
        "eligible": eligible,
        "reason": reason,
        "comparison_policy": (
            "symmetric_before_and_after_owner_paths_with_held_out_shoulders"
        ),
        "before_seam_sha256": _sha_array(before_seam),
        "after_seam_sha256": _sha_array(after_seam),
        "before_image_sha256": _sha_array(before_image),
        "after_image_sha256": _sha_array(after_image),
        "symmetric_before_metrics": symmetric_before,
        "symmetric_after_metrics": symmetric_after,
        "symmetric_structure_eligible": bool(symmetric_ok),
        "symmetric_structure_reason": symmetric_reason,
        "path_audits": path_audits,
    }


def _audit_owner_seam_change(
    left_image: np.ndarray,
    right_image: np.ndarray,
    before_seam_x_by_row: np.ndarray,
    after_seam_x_by_row: np.ndarray,
) -> dict[str, object]:
    """Compose both owner choices from identical pixels before comparing them."""

    before_seam = np.asarray(before_seam_x_by_row, dtype=np.int32)
    after_seam = np.asarray(after_seam_x_by_row, dtype=np.int32)
    before_preview = _compose_pair_preview(left_image, right_image, before_seam)
    after_preview = _compose_pair_preview(left_image, right_image, after_seam)
    audit = _audit_rendered_seam_change(
        before_preview,
        after_preview,
        before_seam,
        after_seam,
    )
    return {
        **audit,
        "before_owner_seam_sha256": _sha_array(before_seam),
        "after_owner_seam_sha256": _sha_array(after_seam),
    }


def _select_held_out_safe_seam(
    left_image: np.ndarray,
    right_image: np.ndarray,
    seam_result: S13SeamResult,
    maximum_model: str,
) -> tuple[str, np.ndarray, dict[str, object]]:
    """Only retain a more complex seam after independent structure evidence."""

    candidates = seam_result.candidate_seams
    base = np.asarray(candidates["midpoint_straight"], dtype=np.int32)
    straight_value = candidates.get("shifted_straight")
    dp_value = candidates.get("monotone_dp")
    straight = None if straight_value is None else np.asarray(straight_value, dtype=np.int32)
    dp = None if dp_value is None else np.asarray(dp_value, dtype=np.int32)
    allowed_rank = {
        "midpoint_straight": 0,
        "shifted_straight": 1,
        "monotone_dp": 2,
    }[maximum_model]

    straight_audit: dict[str, object] | None = None
    straight_safe = False
    if straight is not None and allowed_rank >= 1:
        straight_audit = _audit_owner_seam_change(
            left_image, right_image, base, straight
        )
        straight_safe = bool(straight_audit["eligible"])

    dp_audit: dict[str, object] | None = None
    dp_safe = False
    if dp is not None and straight is not None and allowed_rank >= 2 and straight_safe:
        dp_audit = _audit_owner_seam_change(
            left_image, right_image, straight, dp
        )
        dp_safe = bool(dp_audit["eligible"])

    if dp_safe:
        selected_model, selected_seam = "monotone_dp", dp
        reason = "dp_cost_margin_and_held_out_structure_proved"
    elif straight_safe:
        selected_model, selected_seam = "shifted_straight", straight
        reason = (
            "dp_held_out_structure_not_proved"
            if allowed_rank >= 2 and dp is not None
            else "straight_held_out_structure_proved"
        )
    else:
        selected_model, selected_seam = "midpoint_straight", base
        reason = "straight_held_out_structure_not_proved"
    audit: dict[str, object] = {
        "input_maximum_model": maximum_model,
        "selected_model": selected_model,
        "selection_reason": reason,
        "base_seam_sha256": _sha_array(base),
        "shifted_straight_seam_sha256": None if straight is None else _sha_array(straight),
        "dp_seam_sha256": None if dp is None else _sha_array(dp),
        "selected_seam_sha256": _sha_array(selected_seam),
        "straight_vs_base": straight_audit,
        "dp_vs_straight": dp_audit,
    }
    return selected_model, selected_seam.copy(), audit


def estimate_s13_m5_transactions(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    *,
    parent_stage_sha256: str,
    parent_result_sha256: str | None = None,
    p0_ancestor_completion_sha256: str | None = None,
    maximum_seam_shift_px: int = 8,
    m51_r2_config: S13M51R2Config | None = None,
) -> tuple[S13M5Pair, ...]:
    """Evaluate peer seam candidates independently from immutable P1/P0 grids."""

    validate_s012_schedule(schedule)
    pairs: list[S13M5Pair] = []
    raw_cache: dict[int, np.ndarray] = {}
    successor = m51_r2_config or S13M51R2Config()
    vertical_parent = vertical if isinstance(successor, S13M51R4Config) else None
    instrumentation_requested = m51_r2_config is not None
    audit_all = os.environ.get("G305_S13_M5_AUDIT_ALL_CANDIDATES", "0") == "1"

    def raw(frame_id: int) -> np.ndarray:
        if frame_id not in raw_cache:
            raw_cache[frame_id] = np.asarray(image_loader(frame_id))
        return raw_cache[frame_id]

    for pair_index, (left_assignment, right_assignment) in enumerate(
        zip(schedule.assignments[:-1], schedule.assignments[1:])
    ):
        frame_ids = (left_assignment.frame_id, right_assignment.frame_id)
        correspondence_audit: Mapping[str, object] | None = None
        complete_reassessment = (
            isinstance(successor, S13M51R3Config)
            and successor.requires_complete_seam_reassessment(pair_index)
        )
        try:
            x0, x1 = _pair_domain(schedule, pair_index)
            if x1 - x0 < 12:
                raise ValueError("final_corridor_too_narrow")
            left_maps = _map_crop(
                schedule, calibration, pair_index, x0, x1,
                vertical.global_offsets_px[pair_index], None,
                vertical_parent=vertical_parent,
            )
            right_maps = _map_crop(
                schedule, calibration, pair_index + 1, x0, x1,
                vertical.global_offsets_px[pair_index + 1], None,
                vertical_parent=vertical_parent,
            )
            pair_input = _freeze_m5_pair_input(
                pair_index=pair_index,
                left_frame_id=frame_ids[0],
                right_frame_id=frame_ids[1],
                corridor_x0=x0,
                corridor_x1=x1,
                left_maps=left_maps,
                right_maps=right_maps,
            )
            left_image, left_valid, right_image, right_valid = (
                _sample_m5_base_pair(pair_input, raw)
            )
            correspondence_result = _pair_correspondences(
                left_image, right_image, left_valid, right_valid,
                x_offset=x0, config=successor,
            )
            reference, moving = correspondence_result
            correspondence_audit = correspondence_result.audit
            p0_u, p0_v, p0_valid = _base_calibrated_map(
                schedule, calibration, pair_index + 1
            )
            allowed_left, allowed_right = _allowed_application_bounds(schedule, pair_index)
            base_band = S13ApplicationBand.straight(
                height=schedule.canvas_height, left_x=allowed_left, right_x=allowed_right
            )
            local_vertical = -np.asarray(vertical.local_row_residuals[pair_index], dtype=np.float64)
            preliminary = estimate_s13_pair_alignment(
                pair_index=pair_index, pair_frame_ids=frame_ids, non_reference_side="right",
                p0_source_u=p0_u, p0_source_v=p0_v, p0_valid=p0_valid,
                source_size=(int(calibration.width), int(calibration.height)),
                reference_points_xy=reference, non_reference_points_xy=moving,
                application_band=base_band, accepted_vertical_dy_by_row=local_vertical,
                alignment_shoulder=(x0, x1),
                vertical_accepted=bool(np.any(local_vertical != 0.0)),
            )
            _preliminary_maps, preliminary_right, preliminary_valid, _ = (
                _sample_m5_right_candidate(
                    schedule=schedule,
                    calibration=calibration,
                    source_index=pair_index + 1,
                    x0=x0,
                    x1=x1,
                    global_vertical_offset=vertical.global_offsets_px[pair_index + 1],
                    candidate=preliminary.selected,
                    vertical_parent=vertical_parent,
                    raw_image=raw(frame_ids[1]),
                    base_maps=right_maps,
                    base_image=right_image,
                    base_valid=right_valid,
                )
            )
            base_local = schedule.boundaries[pair_index + 1] - x0
            previous = schedule.boundaries[pair_index] - x0 if pair_index > 0 else None
            following = (
                schedule.boundaries[pair_index + 2] - x0
                if pair_index + 2 < len(schedule.boundaries) else None
            )
            seam_result = select_s13_seam(
                left_image, preliminary_right, left_valid, preliminary_valid, base_local,
                maximum_shift_px=maximum_seam_shift_px,
                previous_boundary_x=previous, next_boundary_x=following,
            )
            candidates, generation_statuses = build_s13_seam_candidates(pair_index, seam_result)
            ordered = rank_s13_seam_candidates(candidates)
            allowed_left_rows = np.full(schedule.canvas_height, allowed_left, dtype=np.int32)
            allowed_right_rows = np.full(schedule.canvas_height, allowed_right, dtype=np.int32)
            evaluations: list[dict[str, object]] = []
            selected_candidate: S13SeamCandidate | None = None
            selected_alignment: S13PairAlignment | None = None
            selected_before: Mapping[str, object] | None = None
            selected_after: Mapping[str, object] | None = None
            selected_before_horizontal: Mapping[str, object] | None = None
            selected_after_horizontal: Mapping[str, object] | None = None
            selected_edge_registration: Mapping[str, object] = {
                "evaluable": False, "visual_suspect": False,
                "supported_block_count": 0, "supported_component_count": 0,
                "median_supported_abs_lag_px": None,
                "p95_supported_abs_lag_px": None, "ambiguous": False,
            }
            selected_component_evidence: tuple[tuple[object, object], ...] = ()
            selected_component_forward_probe: tuple[object, ...] | None = None
            selected_seam_rank = 0
            selected_geometry_rank = 0
            selection_continued_for_structure = False
            unresolved_oblique_structure = False
            hard_safe_baseline: tuple[object, ...] | None = None
            horizontal_lag_baseline: tuple[float, tuple[object, ...]] | None = None
            left_edge_features_cached = None
            for seam_rank, candidate in enumerate(ordered):
                if selected_candidate is not None and not audit_all and not complete_reassessment:
                    evaluations.append({
                        "candidate_id": candidate.candidate_id,
                        "model_code": candidate.model_code,
                        "model_name": candidate.model_name,
                        "generation_status": "generated",
                        "evaluation_status": "skipped_after_higher_rank_safe_candidate",
                        "local_objective": candidate.local_objective,
                        "hard_gate_passed": None,
                        "hard_gate_failures": [],
                        "generation_audit": dict(candidate.generation_audit),
                    })
                    continue
                seam_local = np.asarray(candidate.seam_x_by_row, dtype=np.int32)
                seam_global = seam_local + x0
                failures: list[str] = []
                if seam_local.shape != (schedule.canvas_height,):
                    failures.append("seam_shape_invalid")
                if not np.isfinite(seam_local).all():
                    failures.append("seam_nonfinite")
                if np.any(np.abs(np.diff(seam_local)) > 1):
                    failures.append("seam_row_step_invalid")
                search_left = int(seam_result.audit["search_left_x"])
                search_right = int(seam_result.audit["search_right_x"])
                if np.any((seam_local < search_left) | (seam_local > search_right)):
                    failures.append("seam_out_of_search_bounds")
                alignment: S13PairAlignment | None = None
                before_metrics: Mapping[str, object] = {"evaluable": False, "score": None}
                after_metrics: Mapping[str, object] = {"evaluable": False, "score": None}
                before_horizontal: Mapping[str, object] = {"observed": False}
                after_horizontal: Mapping[str, object] = {"observed": False}
                geometry_candidates: list[dict[str, object]] = []
                map_delta_sha: str | None = None
                corridor_bounds: list[int] | None = None
                edge_registration: Mapping[str, object] = {
                    "evaluable": False, "visual_suspect": False,
                    "supported_block_count": 0, "supported_component_count": 0,
                    "median_supported_abs_lag_px": None,
                    "p95_supported_abs_lag_px": None, "ambiguous": False,
                }
                visual_suspect = False
                candidate_component_evidence: list[tuple[object, object]] = []
                candidate_forward_evidence: list[Mapping[str, object]] = []
                candidate_component_forward_probe: tuple[object, ...] | None = None
                geometry_rank = 0
                if not failures:
                    alignment = reestimate_s13_final_corridor_alignment(
                        final_seam_x_by_row=seam_global,
                        application_half_width_px=max(2, min(8, (allowed_right - allowed_left - 1) // 2)),
                        allowed_left_x_by_row=allowed_left_rows,
                        allowed_right_x_by_row=allowed_right_rows,
                        pair_index=pair_index, pair_frame_ids=frame_ids, non_reference_side="right",
                        p0_source_u=p0_u, p0_source_v=p0_v, p0_valid=p0_valid,
                        source_size=(int(calibration.width), int(calibration.height)),
                        reference_points_xy=reference, non_reference_points_xy=moving,
                        accepted_vertical_dy_by_row=local_vertical, alignment_shoulder=(x0, x1),
                        vertical_accepted=bool(np.any(local_vertical != 0.0)),
                        m51_r2_config=successor,
                    )
                    selected_map = alignment.selected
                    geometry_candidates = [
                        {"model": item.model, "accepted": item.accepted,
                         "failure_reason": item.failure_reason, "metrics": dict(item.metrics),
                         "audit": dict(item.audit)} for item in alignment.candidates
                    ]
                    for key, reason in (
                        ("map_finite", "geometry_map_nonfinite"),
                        ("map_in_source_bounds", "geometry_source_out_of_bounds"),
                        ("positive_jacobian", "geometry_nonpositive_jacobian"),
                    ):
                        if selected_map.audit.get(key) is not True:
                            failures.append(reason)
                    if float(selected_map.audit.get("support_retention", 0.0)) < 0.95:
                        failures.append("geometry_support_retention_insufficient")
                    if float(selected_map.audit.get("maximum_map_displacement_px", math.inf)) > 8.0:
                        failures.append("geometry_displacement_exceeded")
                    _final_maps, final_right, final_valid, final_sample_cache_hit = (
                        _sample_m5_right_candidate(
                            schedule=schedule,
                            calibration=calibration,
                            source_index=pair_index + 1,
                            x0=x0,
                            x1=x1,
                            global_vertical_offset=(
                                vertical.global_offsets_px[pair_index + 1]
                            ),
                            candidate=selected_map,
                            vertical_parent=vertical_parent,
                            raw_image=raw(frame_ids[1]),
                            base_maps=right_maps,
                            base_image=right_image,
                            base_valid=right_valid,
                        )
                    )
                    before_preview = _compose_pair_preview(left_image, right_image, seam_local)
                    after_preview = (
                        before_preview
                        if final_sample_cache_hit
                        else _compose_pair_preview(left_image, final_right, seam_local)
                    )
                    before_features = prepare_seam_structure(before_preview)
                    after_features = (
                        before_features
                        if final_sample_cache_hit
                        else prepare_seam_structure(after_preview)
                    )
                    if before_features is None or after_features is None:
                        raise ValueError("pair preview feature construction failed")
                    before_metrics = seam_structure_metrics(before_features, seam_local)
                    after_metrics = seam_structure_metrics(after_features, seam_local)
                    before_horizontal = long_horizontal_structure_metrics(before_features, seam_local)
                    after_horizontal = long_horizontal_structure_metrics(
                        after_features, seam_local
                    )
                    # Only catastrophic horizontal damage is a runtime gate.
                    horizontal_safe, horizontal_reason, horizontal_audit = (
                        long_horizontal_structure_catastrophe_guard(before_horizontal, after_horizontal)
                    )
                    if not horizontal_safe:
                        failures.append(str(horizontal_reason or "horizontal_structure_catastrophe"))
                    if successor.enabled and not failures:
                        if left_edge_features_cached is None:
                            left_edge_features_cached = prepare_seam_structure(left_image)
                        left_edge_features = left_edge_features_cached
                        final_edge_features = prepare_seam_structure(final_right)
                        if left_edge_features is None or final_edge_features is None:
                            raise ValueError("pair edge feature construction failed")
                        edge_registration = pair_edge_registration_metrics(
                            left_edge_features, final_edge_features, left_valid, final_valid, seam_local,
                            config=successor,
                            pair_index=pair_index,
                            global_x_offset=x0,
                            forward_evidence_sink=(
                                candidate_forward_evidence
                                if isinstance(successor, S13M51R4Config) else None
                            ),
                        )
                        lk_p95_value = selected_map.metrics.get("residual_p95_px")
                        lk_p95 = (
                            float(lk_p95_value)
                            if isinstance(lk_p95_value, (int, float)) else None
                        )
                        visual_suspect = _edge_visual_suspect_for_pair(
                            edge_registration,
                            seam_local_lk_p95_px=lk_p95,
                            config=successor,
                            component_local_authority=complete_reassessment,
                        )
                        edge_registration = {
                            **dict(edge_registration), "visual_suspect": visual_suspect,
                        }

                        if isinstance(successor, S13M51R4Config):
                            if len(candidate_forward_evidence) != 1:
                                raise ValueError(
                                    "S1.3 C2E forward evidence context is missing"
                                )
                            candidate_component_forward_probe = (
                                candidate_forward_evidence[0], left_edge_features,
                                final_edge_features, int(x0),
                            )
                        if visual_suspect:
                            if isinstance(successor, S13M51R4Config):
                                # The ordinary forward block audit above is the
                                # trigger. Only unresolved suspect pairs pay
                                # for exact physical-component reverse evidence;
                                # cached maps and Sobel features are reused.
                                candidate_component_evidence.clear()
                                if len(candidate_forward_evidence) != 1:
                                    raise ValueError(
                                        "S1.3 C2E forward evidence context is missing"
                                    )
                                append_s13_exact_component_evidence_from_forward_context(
                                    candidate_forward_evidence[0],
                                    left_features=left_edge_features,
                                    right_features=final_edge_features,
                                    config=successor,
                                    exact_evidence_sink=candidate_component_evidence,
                                    pair_index=pair_index,
                                    global_x_offset=x0,
                                )
                        if visual_suspect:
                            selection_continued_for_structure = True
                            if hard_safe_baseline is None:
                                hard_safe_baseline = (
                                    candidate, alignment, before_metrics, after_metrics,
                                    before_horizontal, after_horizontal, edge_registration,
                                    seam_rank, 0, tuple(candidate_component_evidence),
                                    candidate_component_forward_probe,
                                )
                            clean_alternatives: list[
                                tuple[float, int, int, object, object, object, object]
                            ] = []
                            simplicity = {
                                "C0_identity": 0, "C1_accepted_vertical": 1,
                                "C2_subpixel_translation": 2,
                                "C3_translation_tiny_rotation": 3,
                                "C4_bounded_light_affine": 4,
                            }
                            for alternate_index, alternate in enumerate(alignment.candidates):
                                if not alternate.accepted or alternate_index == alignment.selected_candidate_index:
                                    continue
                                if (
                                    alternate.audit.get("map_finite") is not True
                                    or alternate.audit.get("map_in_source_bounds") is not True
                                    or alternate.audit.get("positive_jacobian") is not True
                                    or float(alternate.audit.get("support_retention", 0.0)) < 0.95
                                    or float(alternate.audit.get("maximum_map_displacement_px", math.inf)) > 8.0
                                ):
                                    continue
                                alternate_maps = _map_crop(
                                    schedule, calibration, pair_index + 1, x0, x1,
                                    vertical.global_offsets_px[pair_index + 1], alternate,
                                    vertical_parent=vertical_parent,
                                )
                                alternate_right, alternate_valid = _sample_crop(
                                    raw(frame_ids[1]), alternate_maps
                                )
                                alternate_preview = _compose_pair_preview(
                                    left_image, alternate_right, seam_local
                                )
                                alternate_preview_features = prepare_seam_structure(alternate_preview)
                                alternate_edge_features = prepare_seam_structure(alternate_right)
                                if alternate_preview_features is None or alternate_edge_features is None:
                                    raise ValueError("alternate pair feature construction failed")
                                alternate_after = seam_structure_metrics(
                                    alternate_preview_features, seam_local
                                )
                                alternate_horizontal = long_horizontal_structure_metrics(
                                    alternate_preview_features, seam_local
                                )
                                alternate_safe, _alternate_reason, _alternate_audit = (
                                    long_horizontal_structure_catastrophe_guard(
                                        before_horizontal, alternate_horizontal
                                    )
                                )
                                if not alternate_safe:
                                    continue
                                alternate_forward_evidence: list[Mapping[str, object]] = []
                                alternate_edge = pair_edge_registration_metrics(
                                    left_edge_features, alternate_edge_features, left_valid,
                                    alternate_valid, seam_local, config=successor,
                                    pair_index=pair_index,
                                    global_x_offset=x0,
                                    forward_evidence_sink=alternate_forward_evidence,
                                )
                                alt_lk_value = alternate.metrics.get("residual_p95_px")
                                alt_lk = (
                                    float(alt_lk_value)
                                    if isinstance(alt_lk_value, (int, float)) else None
                                )
                                if _edge_visual_suspect_for_pair(
                                    alternate_edge,
                                    seam_local_lk_p95_px=alt_lk,
                                    config=successor,
                                    component_local_authority=complete_reassessment,
                                ):
                                    continue
                                edge_p95_value = alternate_edge.get("p95_supported_abs_lag_px")
                                edge_p95 = (
                                    float(edge_p95_value)
                                    if isinstance(edge_p95_value, (int, float)) else math.inf
                                )
                                clean_alternatives.append((
                                    edge_p95, simplicity.get(alternate.model, 99),
                                    alternate_index, alternate_after,
                                    (alternate_horizontal, alternate_edge),
                                    (),
                                    (
                                        alternate_forward_evidence[0],
                                        left_edge_features,
                                        alternate_edge_features,
                                        int(x0),
                                    ) if len(alternate_forward_evidence) == 1 else None,
                                ))
                            if clean_alternatives:
                                minimum_edge = min(item[0] for item in clean_alternatives)
                                equivalent = [
                                    item for item in clean_alternatives
                                    if item[0] <= minimum_edge + successor.equivalent_edge_p95_tolerance_px
                                ]
                                chosen = min(equivalent, key=lambda item: item[1])
                                geometry_rank = int(chosen[2])
                                alignment = replace(
                                    alignment,
                                    selected_model=alignment.candidates[geometry_rank].model,
                                    selected_candidate_index=geometry_rank,
                                )
                                selected_map = alignment.selected
                                after_metrics = chosen[3]  # type: ignore[assignment]
                                alternate_details = chosen[4]
                                after_horizontal = alternate_details[0]  # type: ignore[index,assignment]
                                edge_registration = {
                                    **dict(alternate_details[1]),  # type: ignore[arg-type,index]
                                    "visual_suspect": False,
                                }
                                candidate_component_evidence = list(chosen[5])
                                candidate_component_forward_probe = chosen[6]
                                visual_suspect = False
                    map_delta_sha = _sha_array(selected_map.target_delta_u, selected_map.target_delta_v)
                    corridor_bounds = [
                        int(alignment.application_band.left_x_by_row.min()),
                        int(alignment.application_band.right_x_by_row.max()),
                    ]
                else:
                    horizontal_audit = {"passed": False, "not_evaluated": True}
                hard_safe = not failures
                passed = hard_safe
                if successor.enabled and hard_safe and visual_suspect:
                    passed = False
                horizontal_lag_value = after_horizontal.get(
                    "absolute_best_vertical_lag_px"
                )
                horizontal_lag = (
                    float(horizontal_lag_value)
                    if (
                        after_horizontal.get("observed") is True
                        and isinstance(horizontal_lag_value, (int, float))
                        and np.isfinite(float(horizontal_lag_value))
                    )
                    else None
                )
                continue_for_horizontal_zero_lag = bool(
                    passed
                    and selected_candidate is None
                    and isinstance(successor, S13M51R4Config)
                    and horizontal_lag is not None
                    and horizontal_lag > 0.0
                )
                evaluation = {
                    "candidate_id": candidate.candidate_id,
                    "model_code": candidate.model_code,
                    "model_name": candidate.model_name,
                    "generation_status": "generated",
                    "evaluation_status": (
                        "horizontal_zero_lag_search_continued"
                        if continue_for_horizontal_zero_lag
                        else "selected" if passed and selected_candidate is None
                        else "hard_safe_not_selected" if passed
                        else "visual_suspect_continued" if hard_safe and visual_suspect
                        else "rejected_hard_gate"
                    ),
                    "local_objective": candidate.local_objective,
                    "seam_sha256": _sha_array(seam_global),
                    "corridor_bounds": corridor_bounds,
                    "geometry_input_grid_sha256": _sha_array(p0_u, p0_v, p0_valid),
                    "geometry_model": None if alignment is None else alignment.selected.model,
                    "geometry_candidates": geometry_candidates,
                    "map_delta_sha256": map_delta_sha,
                    "horizontal_hard_audit": horizontal_audit,
                    "edge_registration": dict(edge_registration),
                    "visual_suspect": bool(visual_suspect),
                    "hard_gate_passed": hard_safe,
                    "hard_gate_failures": failures,
                    "diagnostic_metrics": {"before": before_metrics, "after": after_metrics},
                    "generation_audit": dict(candidate.generation_audit),
                }
                evaluations.append(evaluation)
                if continue_for_horizontal_zero_lag:
                    selection_continued_for_structure = True
                    candidate_state: tuple[object, ...] = (
                        candidate, alignment, before_metrics, after_metrics,
                        before_horizontal, after_horizontal, edge_registration,
                        seam_rank, geometry_rank, tuple(candidate_component_evidence),
                        candidate_component_forward_probe,
                    )
                    if (
                        horizontal_lag_baseline is None
                        or horizontal_lag < horizontal_lag_baseline[0]
                    ):
                        horizontal_lag_baseline = (horizontal_lag, candidate_state)
                elif passed and selected_candidate is None:
                    selected_candidate = candidate
                    selected_alignment = alignment
                    selected_before, selected_after = before_metrics, after_metrics
                    selected_before_horizontal, selected_after_horizontal = before_horizontal, after_horizontal
                    selected_edge_registration = edge_registration
                    selected_seam_rank = seam_rank
                    selected_geometry_rank = geometry_rank
                    selected_component_evidence = tuple(candidate_component_evidence)
                    selected_component_forward_probe = candidate_component_forward_probe
            if selected_candidate is None and horizontal_lag_baseline is not None:
                (
                    selected_candidate, selected_alignment, selected_before, selected_after,
                    selected_before_horizontal, selected_after_horizontal,
                    selected_edge_registration, selected_seam_rank, selected_geometry_rank,
                    selected_component_evidence, selected_component_forward_probe,
                ) = horizontal_lag_baseline[1]
                for row in evaluations:
                    if row.get("candidate_id") == selected_candidate.candidate_id:
                        row["evaluation_status"] = (
                            "selected_best_available_horizontal_lag"
                        )
                        break
            if selected_candidate is None and hard_safe_baseline is not None:
                (
                    selected_candidate, selected_alignment, selected_before, selected_after,
                    selected_before_horizontal, selected_after_horizontal,
                    selected_edge_registration, selected_seam_rank, selected_geometry_rank,
                    selected_component_evidence, selected_component_forward_probe,
                ) = hard_safe_baseline
                unresolved_oblique_structure = True
                for row in evaluations:
                    if row.get("candidate_id") == selected_candidate.candidate_id:
                        row["evaluation_status"] = (
                            "selected_unresolved_hard_safe_baseline"
                        )
                        break
            if selected_candidate is None or selected_alignment is None:
                fallback = _fallback_pair(
                    schedule, pair_index, parent_stage_sha256, frame_ids,
                    "all_generated_candidates_failed_declared_hard_gates",
                    parent_result_sha256, p0_ancestor_completion_sha256,
                    correspondence_filter=correspondence_audit,
                    m51_r2_config=successor,
                )
                transaction = {
                    **dict(fallback.transaction),
                    "candidate_generation": [dict(item) for item in generation_statuses],
                    "candidate_evaluations": evaluations,
                }
                transaction = _finalize_v5_transaction(transaction)
                pairs.append(S13M5Pair(
                    transaction, fallback.seam_x_by_row, None,
                    c2e_seam_candidates=(fallback.seam_x_by_row.copy(),),
                ))
                continue
            selected_seam = np.asarray(selected_candidate.seam_x_by_row, dtype=np.int32) + x0
            selected_map = selected_alignment.selected
            core: dict[str, object] = {
                "schema": _successor_transaction_schema(successor),
                "transaction_id": f"m5-pair-{pair_index:04d}",
                "parent_stage": "P1",
                "parent_stage_sha256": parent_stage_sha256,
                "parent_completion_sha256": parent_stage_sha256,
                "parent_result_sha256": parent_result_sha256,
                "p0_ancestor_completion_sha256": p0_ancestor_completion_sha256,
                "pair_frame_ids": list(frame_ids),
                "candidate_generation": [dict(item) for item in generation_statuses],
                "candidate_evaluations": evaluations,
                "selection_policy": (
                    "minimum_local_objective_with_supported_edge_residual_veto"
                    if successor.enabled
                    else "minimum_local_objective_among_hard_safe_candidates"
                ),
                "selected_candidate_id": selected_candidate.candidate_id,
                "selected_seam_model": selected_candidate.model_name,
                "selected_geometry_model": selected_map.model,
                "pair_hard_audit_passed": True,
                "fallback_used": selected_candidate.model_code == "S0" and selected_map.model == "C0_identity",
                "fallback_reason": (
                    "complex_candidates_failed_or_ranked_later" if selected_candidate.model_code == "S0" else None
                ),
                "motion_hypothesis_id": -1,
                "support_mask_sha256": _sha_array(left_valid & right_valid),
                "geometry_model": selected_map.model,
                "seam_model": selected_candidate.model_name,
                "map_delta_sha256": _sha_array(selected_map.target_delta_u, selected_map.target_delta_v),
                "decision": "applied",
                "before_metrics": dict(selected_before or {}),
                "candidate_after_metrics": dict(selected_after or {}),
                "after_metrics": dict(selected_after or {}),
                "before_horizontal_continuity": dict(selected_before_horizontal or {}),
                "candidate_after_horizontal_continuity": dict(selected_after_horizontal or {}),
                "after_horizontal_continuity": dict(selected_after_horizontal or {}),
                "rollback_reason": None,
                "comparison_coordinate_policy": "same_owner_geometry_plus_symmetric_base_candidate_seam",
                "geometry_comparison_seam_sha256": _sha_array(selected_seam),
                "selected_seam_sha256": _sha_array(selected_seam),
                "result_seam_sha256": _sha_array(selected_seam),
                "held_out_seam_selection_audits": [
                    {
                        "candidate_id": item.get("candidate_id"),
                        "evaluation_status": item.get("evaluation_status"),
                        "selected_seam_sha256": item.get("seam_sha256"),
                        "hard_gate_passed": item.get("hard_gate_passed"),
                        "hard_gate_failures": item.get("hard_gate_failures", []),
                    }
                    for item in evaluations
                    if item.get("evaluation_status") != "skipped_after_higher_rank_safe_candidate"
                ],
                "final_corridor_geometry_reestimated_from_immutable_p0": True,
                "geometry_candidates": evaluations,
                "seam_audit": dict(seam_result.audit),
                "seam_cost_components": dict(
                    seam_result.candidate_cost_components[selected_candidate.model_name]
                ),
                "seam_fallback_chain": list(seam_result.fallback_chain),
            }
            if instrumentation_requested:
                core["correspondence_filter"] = dict(correspondence_result.audit)
            if successor.enabled:
                transaction_edge = dict(selected_edge_registration)
                transaction_edge.setdefault(
                    "ambiguous",
                    bool(transaction_edge.get("multiple_layer_or_ambiguous", False)),
                )
                core.update({
                    "seam_local_evidence": dict(
                        selected_alignment.seam_local_evidence or {}
                    ),
                    "edge_registration": transaction_edge,
                    "selection_continued_for_structure": bool(
                        selection_continued_for_structure
                    ),
                    "selected_seam_rank": int(selected_seam_rank),
                    "selected_geometry_rank": int(selected_geometry_rank),
                    "seam_fallback_used": bool(selected_seam_rank > 0),
                    "geometry_fallback_used": bool(
                        selected_map.model in {"C0_identity", "C1_accepted_vertical"}
                    ),
                    "topology_repair_used": False,
                    "unresolved_oblique_structure": bool(unresolved_oblique_structure),
                    "unexpected_exception_fallback": False,
                    "micro_rescue": {
                        "enabled": False, "attempted": False, "accepted": False,
                    },
                    "c2e": {
                        "enabled": False, "attempted": False, "accepted": False,
                    },
                    "complete_seam_reassessment_diagnostic": bool(
                        complete_reassessment
                    ),
                })
            if successor.enabled:
                core = _finalize_v5_transaction(core)
            else:
                core["result_stage_sha256"] = _sha_json(core)
            pairs.append(S13M5Pair(
                core, selected_seam, selected_alignment, selected_component_evidence,
                selected_component_forward_probe,
                c2e_seam_candidates=tuple(
                    np.asarray(candidate.seam_x_by_row, dtype=np.int32).copy() + x0
                    for candidate in ordered
                ),
            ))
        except Exception as exc:
            if successor.enabled:
                raise
            fallback = _fallback_pair(
                schedule, pair_index, parent_stage_sha256, frame_ids,
                f"{type(exc).__name__}:{exc}",
                parent_result_sha256,
                p0_ancestor_completion_sha256,
                correspondence_filter=(
                    correspondence_audit if instrumentation_requested else None
                ),
                m51_r2_config=m51_r2_config,
            )
            pairs.append(fallback)
    # One deterministic topology repair pass.  A local candidate may stay in
    # its own ordered search domain yet meet its neighbour on a row; only the
    # involved pairs fall back to their immutable P1 midpoint grids.
    if len(pairs) > 1:
        seams = np.stack([np.asarray(pair.seam_x_by_row, dtype=np.int32) for pair in pairs])
        crossing = np.flatnonzero(np.any(np.diff(seams, axis=0) < 1, axis=1))
        repair_indices = sorted({int(index) for value in crossing for index in (value, value + 1)})
        for pair_index in repair_indices:
            left = schedule.assignments[pair_index]
            right = schedule.assignments[pair_index + 1]
            original = pairs[pair_index]
            fallback = _fallback_pair(
                schedule, pair_index, parent_stage_sha256,
                (left.frame_id, right.frame_id),
                "seam_family_crossing_deterministic_midpoint_identity_repair",
                parent_result_sha256,
                p0_ancestor_completion_sha256,
                correspondence_filter=(
                    original.transaction.get("correspondence_filter")
                    if instrumentation_requested else None
                ),
                m51_r2_config=m51_r2_config,
                topology_repair_used=True,
            )
            transaction = {
                **dict(fallback.transaction),
                "candidate_generation": original.transaction.get("candidate_generation", []),
                "candidate_evaluations": original.transaction.get("candidate_evaluations", []),
            }
            if instrumentation_requested and "correspondence_filter" in original.transaction:
                transaction["correspondence_filter"] = original.transaction[
                    "correspondence_filter"
                ]
            if successor.enabled:
                transaction = _finalize_v5_transaction(transaction)
            else:
                transaction["result_stage_sha256"] = _sha_json(transaction)
            pairs[pair_index] = S13M5Pair(
                transaction, fallback.seam_x_by_row, None,
                c2e_seam_candidates=(fallback.seam_x_by_row.copy(),),
            )
    if isinstance(successor, S13M51R4Config):
        seed_by_pair = {
            pair_index: pair.component_evidence
            for pair_index, pair in enumerate(pairs)
            if pair.component_evidence
        }
        if seed_by_pair:
            def load_exact_pair(pair_index: int):
                probe = pairs[pair_index].component_forward_probe
                if probe is None:
                    return ()
                context, left_features, right_features, global_x_offset = probe
                sink: list[tuple[object, object]] = []
                append_s13_exact_component_evidence_from_forward_context(
                    context,
                    left_features=left_features,
                    right_features=right_features,
                    config=successor,
                    exact_evidence_sink=sink,
                    pair_index=pair_index,
                    global_x_offset=int(global_x_offset),
                )
                return tuple(sink)

            propagated, propagation_audit = propagate_s13_edge_component_evidence(
                seed_by_pair,
                available_pair_indices=tuple(
                    index for index, pair in enumerate(pairs)
                    if pair.component_forward_probe is not None
                ),
                pair_evidence_loader=load_exact_pair,
                config=successor,
            )
            evidence_by_pair: dict[int, list[tuple[object, object]]] = {}
            for row in propagated:
                evidence_by_pair.setdefault(int(row[0].pair_index), []).append(row)
            pairs = [
                replace(
                    pair,
                    component_evidence=tuple(evidence_by_pair.get(index, ())),
                    component_propagation_audit=(
                        propagation_audit if index == 0 else None
                    ),
                )
                for index, pair in enumerate(pairs)
            ]
    return tuple(pairs)


def _seams_array(schedule: S012Schedule, pairs: Sequence[S13M5Pair], *, final: bool) -> np.ndarray:
    seams = []
    for index, pair in enumerate(pairs):
        if final:
            seams.append(np.asarray(pair.seam_x_by_row, dtype=np.int32))
        else:
            seams.append(np.full(schedule.canvas_height, schedule.boundaries[index + 1], dtype=np.int32))
    if not seams:
        return np.empty((0, schedule.canvas_height), dtype=np.int32)
    result = np.stack(seams)
    if np.any(np.diff(result, axis=0) < 0):
        raise ValueError("S1.3 M5 adjacent seams cross")
    if np.any(np.abs(result - np.asarray(schedule.boundaries[1:-1])[:, None]) > 8):
        raise ValueError("S1.3 M5 seam exceeds maximum shift")
    return result


def build_s13_p2_replay(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    vertical: S13VerticalSolution,
    pairs: Sequence[S13M5Pair],
    *,
    corridor_half_width_px: int = 8,
    map_provider: S13SourceMapProvider | None = None,
) -> tuple[S13P2ReplayPair, ...]:
    """Serialize selected P2 sampling without re-estimating geometry or seams."""

    if not 4 <= int(corridor_half_width_px) <= 16:
        raise ValueError("S1.3 P2 replay half-width must be in [4, 16]")
    seams = _seams_array(schedule, pairs, final=True)
    replay: list[S13P2ReplayPair] = []
    for pair_index, pair in enumerate(pairs):
        seam = seams[pair_index]
        x0 = max(0, int(seam.min()) - int(corridor_half_width_px))
        x1 = min(schedule.canvas_width, int(seam.max()) + int(corridor_half_width_px) + 1)
        left_candidate = (
            pairs[pair_index - 1].alignment.selected
            if pair_index > 0 and pairs[pair_index - 1].alignment is not None
            else None
        )
        right_candidate = pair.alignment.selected if pair.alignment is not None else None
        if map_provider is None:
            left_base = _map_crop(
                schedule, calibration, pair_index, x0, x1,
                vertical.global_offsets_px[pair_index], left_candidate,
            )
            right_base = _map_crop(
                schedule, calibration, pair_index + 1, x0, x1,
                vertical.global_offsets_px[pair_index + 1], right_candidate,
            )
            left_maps = left_base
            right_maps = right_base
        else:
            left_maps = map_provider(pair_index, x0, x1)
            right_maps = map_provider(pair_index + 1, x0, x1)
        canvas_x = np.arange(x0, x1, dtype=np.int32)[None, :]
        owner_right = canvas_x >= seam[:, None]
        replay.append(S13P2ReplayPair(
            pair_index=pair_index,
            left_source_index=pair_index,
            right_source_index=pair_index + 1,
            left_frame_id=int(schedule.assignments[pair_index].frame_id),
            right_frame_id=int(schedule.assignments[pair_index + 1].frame_id),
            corridor_x0=x0,
            corridor_x1=x1,
            seam_x_by_row=seam.copy(),
            left_source_u=np.asarray(left_maps[0], dtype=np.float32),
            left_source_v=np.asarray(left_maps[1], dtype=np.float32),
            left_valid=np.asarray(left_maps[2], dtype=bool),
            right_source_u=np.asarray(right_maps[0], dtype=np.float32),
            right_source_v=np.asarray(right_maps[1], dtype=np.float32),
            right_valid=np.asarray(right_maps[2], dtype=bool),
            primary_owner_right_mask=owner_right,
            geometry_transaction_numeric_id=pair_index,
            seam_transaction_numeric_id=pair_index,
            parent_pair_transaction_sha256="",
            left_component_correction_field_id=(
                np.asarray(left_maps[3], dtype=np.int32) if map_provider is not None else None
            ),
            right_component_correction_field_id=(
                np.asarray(right_maps[3], dtype=np.int32) if map_provider is not None else None
            ),
        ))
    return tuple(replay)


def plan_s13_m5_render_domains(
    schedule: S012Schedule,
    pairs: Sequence[S13M5Pair],
) -> Mapping[int, tuple[int, int]]:
    """Plan the union of the fixed and selected owner topologies."""

    width, height = schedule.canvas_width, schedule.canvas_height
    canvas = np.arange(width, dtype=np.int32)[None, :]
    domains: dict[int, list[tuple[int, int]]] = {
        index: [] for index in range(len(schedule.assignments))
    }
    for final in (False, True):
        owner = np.zeros((height, width), np.int32)
        for seam in _seams_array(schedule, pairs, final=final):
            owner += canvas >= seam[:, None]
        for source in domains:
            columns = np.flatnonzero(np.any(owner == source, axis=0))
            if columns.size:
                domains[source].append((int(columns[0]), int(columns[-1]) + 1))
    return MappingProxyType({
        source: (min(row[0] for row in rows), max(row[1] for row in rows))
        for source, rows in domains.items() if rows
    })


def plan_s13_m5_oracle_domains(
    schedule: S012Schedule,
    pairs: Sequence[S13M5Pair],
    registry: SourceCorrectionRegistry,
    *,
    corridor_half_width_px: int = 8,
    bilinear_halo_px: int = 1,
) -> Mapping[int, tuple[int, int]]:
    """Plan the union of both owner topologies, replay, and correction halos."""

    width = schedule.canvas_width
    domains: dict[int, list[tuple[int, int]]] = {
        source: [domain]
        for source, domain in plan_s13_m5_render_domains(schedule, pairs).items()
    }
    for source in range(len(schedule.assignments)):
        domains.setdefault(source, [])
    for pair_index, pair in enumerate(pairs):
        seam = np.asarray(pair.seam_x_by_row, np.int32)
        x0 = max(0, int(seam.min()) - corridor_half_width_px)
        x1 = min(width, int(seam.max()) + corridor_half_width_px + 1)
        domains[pair_index].append((x0, x1))
        domains[pair_index + 1].append((x0, x1))
    for source, corrections in registry.corrections_by_source.items():
        for correction in corrections:
            active_columns = np.flatnonzero(np.any(correction.weight > 0.0, axis=0))
            if active_columns.size:
                domains[int(source)].append((
                    max(0, correction.x0 + int(active_columns[0]) - bilinear_halo_px),
                    min(width, correction.x0 + int(active_columns[-1]) + 1 + bilinear_halo_px),
                ))
    return MappingProxyType({
        source: (min(row[0] for row in rows), max(row[1] for row in rows))
        for source, rows in domains.items() if rows
    })


def source_map_oracle_provider(
    oracles: Sequence[SourceMapOracle],
) -> S13SourceMapProvider:
    """Return a strict slice-only provider; requests outside authority fail."""

    by_source = {row.source_index: row for row in oracles}

    def provide(source_index: int, x0: int, x1: int):
        oracle = by_source.get(source_index)
        if oracle is None:
            raise ValueError("source-map oracle authority is missing")
        ox0, oy0, ox1, oy1 = oracle.domain_xyxy
        if oy0 != 0 or x0 < ox0 or x1 > ox1 or x0 >= x1:
            raise ValueError("source-map oracle request exceeds frozen domain")
        local = np.s_[:, x0 - ox0:x1 - ox0]
        return (
            oracle.u[local], oracle.v[local], oracle.valid[local] != 0,
            oracle.field_id[local],
        )

    return provide


def _build_s13_expected_support_mask_exact(
    schedule: S012Schedule,
    provider: S13SourceMapProvider,
) -> tuple[np.ndarray, Mapping[str, int | float]]:
    """Build the unchanged full-canvas support union exactly once."""

    started = time.perf_counter()
    height, width = schedule.canvas_height, schedule.canvas_width
    mask = np.zeros((height, width), dtype=bool)
    provider_call_count = 0
    for source_index in range(len(schedule.assignments)):
        maps = provider(source_index, 0, width)
        valid = np.asarray(maps[2], dtype=bool)
        if valid.shape != (height, width):
            raise ValueError("S1.3 M5 expected support provider shape changed")
        mask |= valid
        provider_call_count += 1
    mask.setflags(write=False)
    return mask, MappingProxyType({
        "build_seconds": time.perf_counter() - started,
        "build_count": 1,
        "provider_call_count": provider_call_count,
        "requested_pixel_count": provider_call_count * height * width,
        "formal_geometry_rebuild_count": 0,
        "formal_final_rebuild_count": 0,
        "mask_true_pixel_count": int(np.count_nonzero(mask)),
    })


def _estimate_s13_m5_pre_render(
    *,
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    vertical: S13VerticalSolution,
    pairs: tuple[S13M5Pair, ...],
    raw_cache: Mapping[int, np.ndarray],
    registry: SourceCorrectionRegistry,
    patch_set: S13ComponentPatchSet,
    component_audit: Mapping[str, object],
    final_map_provider: S13SourceMapProvider,
    performance_counters: Mapping[str, int | float],
) -> tuple[S13M5EstimationResult, S13SourceMapProvider]:
    """Freeze all R4 authorities before any formal P2 render."""

    domains = plan_s13_m5_oracle_domains(schedule, pairs, registry)
    base_oracles, final_oracles = [], []
    for source, (x0, x1) in domains.items():
        candidate = (
            pairs[source - 1].alignment.selected
            if source > 0 and pairs[source - 1].alignment is not None else None
        )
        base = _map_crop(
            schedule, calibration, source, x0, x1,
            vertical.global_offsets_px[source], candidate,
            vertical_parent=vertical,
        )
        final = final_map_provider(source, x0, x1)
        base_oracles.append(source_map_oracle_from_arrays(
            source_index=source, domain_xyxy=(x0, 0, x1, schedule.canvas_height),
            u=base[0], v=base[1], valid=base[2],
            field_id=np.full(base[2].shape, -1, np.int32),
        ))
        final_oracles.append(source_map_oracle_from_arrays(
            source_index=source, domain_xyxy=(x0, 0, x1, schedule.canvas_height),
            u=final[0], v=final[1], valid=final[2], field_id=final[3],
        ))

    def expected_support(source: int, x0: int, x1: int):
        candidate = (
            pairs[source - 1].alignment.selected
            if source > 0 and pairs[source - 1].alignment is not None else None
        )
        maps = _map_crop(
            schedule, calibration, source, x0, x1,
            vertical.global_offsets_px[source], candidate,
            vertical_parent=vertical,
        )
        return maps[0], maps[1], maps[2], np.full(maps[2].shape, -1, np.int32)

    probes = MappingProxyType({
        index: pair.component_forward_probe for index, pair in enumerate(pairs)
        if pair.component_forward_probe is not None
    })
    context = S13M51R4EvidenceContext(
        raw_rgb_by_frame=MappingProxyType(dict(raw_cache)),
        component_forward_probes=probes,
        stage_order=(
            "pair_estimation", "topology_repair", "component_rescue",
            "source_map_oracle_freeze", "formal_render",
        ),
    )
    expected_support_mask, expected_support_audit = (
        _build_s13_expected_support_mask_exact(schedule, expected_support)
    )
    result = S13M5EstimationResult(
        pairs=pairs, source_correction_registry=registry,
        component_patch_set=patch_set,
        component_chain_audit=MappingProxyType(dict(component_audit)),
        evidence_context=context,
        base_source_map_oracles=tuple(base_oracles),
        final_source_map_oracles=tuple(final_oracles),
        performance_counters=MappingProxyType(dict(performance_counters)),
        expected_support_mask=expected_support_mask,
        expected_support_audit=expected_support_audit,
    )
    return result, source_map_oracle_provider(final_oracles)


def _finalize_s13_m5_render(
    *,
    estimate: S13M5EstimationResult,
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    vertical: S13VerticalSolution,
    selected_hypothesis_ids: tuple[int, ...] | None,
    placement_methods: tuple[str, ...] | None,
    map_provider: S13SourceMapProvider,
    expected_support_mask: np.ndarray,
    final_image_composer: Callable[[tuple[int, ...], np.ndarray, dict[str, np.ndarray]], np.ndarray] | None = None,
) -> tuple[S13P2Result, S13P2Result, tuple[S13P2ReplayPair, ...]]:
    """Consume only frozen pre-render authority for both formal renders/replay."""

    def raw(frame_id: int) -> np.ndarray:
        return estimate.evidence_context.raw_rgb_by_frame[frame_id]

    oracle_by_source = {
        row.source_index: row for row in estimate.final_source_map_oracles
    }
    sampled_cache: dict[int, tuple[int, int, np.ndarray, np.ndarray]] = {}
    def sampled_source(source_index: int, x0: int, x1: int):
        cached = sampled_cache.get(source_index)
        cache_miss = cached is None
        if cached is None:
            oracle = oracle_by_source.get(source_index)
            if oracle is None:
                raise ValueError("sampled source cache authority is missing")
            ox0, oy0, ox1, oy1 = oracle.domain_xyxy
            if oy0 != 0 or oy1 != schedule.canvas_height:
                raise ValueError("sampled source cache domain is not full-height")
            assignment = schedule.assignments[source_index]
            source = np.asarray(raw(int(assignment.frame_id)))
            maps = map_provider(source_index, ox0, ox1)
            sampled, valid = _sample_crop(source, maps[:3])
            sampled = np.asarray(sampled).view()
            valid = np.asarray(valid, dtype=bool).view()
            sampled.flags.writeable = False
            valid.flags.writeable = False
            cached = (ox0, ox1, sampled, valid)
            sampled_cache[source_index] = cached
        ox0, ox1, sampled, valid = cached
        if x0 < ox0 or x1 > ox1 or x0 >= x1:
            raise ValueError("sampled source request exceeds frozen domain")
        local = np.s_[:, x0 - ox0:x1 - ox0]
        return sampled[local], valid[local], cache_miss

    geometry = render_s13_p2_from_raw(
        schedule, calibration, raw, vertical, estimate.pairs, final_seams=False,
        selected_hypothesis_ids=selected_hypothesis_ids,
        placement_methods=placement_methods, map_provider=map_provider,
        expected_support_mask=expected_support_mask,
        sampled_source_provider=sampled_source,
    )
    final = render_s13_p2_from_raw(
        schedule, calibration, raw, vertical, estimate.pairs, final_seams=True,
        selected_hypothesis_ids=selected_hypothesis_ids,
        placement_methods=placement_methods, map_provider=map_provider,
        expected_support_mask=expected_support_mask,
        sampled_source_provider=sampled_source,
        image_composer=final_image_composer,
    )
    replay = build_s13_p2_replay(
        schedule, calibration, vertical, estimate.pairs, map_provider=map_provider
    )
    return geometry, final, replay


def _ordinary_s13_sampled_source_provider(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    pairs: Sequence[S13M5Pair],
) -> tuple[S13SourceMapProvider, S13SampledSourceProvider, np.ndarray]:
    """Cache union maps/samples and the topology-independent support audit."""

    domains = plan_s13_m5_render_domains(schedule, pairs)
    map_cache: dict[
        int, tuple[int, int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    cache: dict[int, tuple[int, int, np.ndarray, np.ndarray]] = {}

    def provide_maps(source_index: int, x0: int, x1: int):
        cached = map_cache.get(source_index)
        if cached is None:
            domain = domains.get(source_index)
            if domain is None:
                raise ValueError("ordinary source map cache authority is missing")
            ox0, ox1 = domain
            candidate = (
                pairs[source_index - 1].alignment.selected
                if source_index > 0
                and pairs[source_index - 1].alignment is not None
                else None
            )
            maps = _map_crop(
                schedule, calibration, source_index, ox0, ox1,
                vertical.global_offsets_px[source_index], candidate,
            )
            cached = (ox0, ox1, maps)
            map_cache[source_index] = cached
        ox0, ox1, maps = cached
        if x0 < ox0 or x1 > ox1 or x0 >= x1:
            raise ValueError("ordinary source map request exceeds owner union")
        local = np.s_[:, x0 - ox0:x1 - ox0]
        return tuple(np.asarray(value)[local] for value in maps)

    def provide(source_index: int, x0: int, x1: int):
        cached = cache.get(source_index)
        cache_miss = cached is None
        if cached is None:
            domain = domains.get(source_index)
            if domain is None:
                raise ValueError("ordinary sampled source cache authority is missing")
            ox0, ox1 = domain
            maps = provide_maps(source_index, ox0, ox1)
            assignment = schedule.assignments[source_index]
            source = np.asarray(image_loader(int(assignment.frame_id)))
            if source.shape != (
                schedule.canvas_height, int(calibration.width), 3
            ) or source.dtype != np.uint8:
                raise ValueError("S1.3 M5 raw RGB source shape/type changed")
            sampled, valid = _sample_crop(source, maps[:3])
            sampled = np.asarray(sampled).view()
            valid = np.asarray(valid, dtype=bool).view()
            sampled.flags.writeable = False
            valid.flags.writeable = False
            cached = (ox0, ox1, sampled, valid)
            cache[source_index] = cached
        ox0, ox1, sampled, valid = cached
        if x0 < ox0 or x1 > ox1 or x0 >= x1:
            raise ValueError("ordinary sampled source request exceeds owner union")
        local = np.s_[:, x0 - ox0:x1 - ox0]
        return sampled[local], valid[local], cache_miss

    expected_support = np.zeros(
        (schedule.canvas_height, schedule.canvas_width), dtype=bool
    )
    for source_index in range(len(schedule.assignments)):
        assignment = schedule.assignments[source_index]
        candidate = (
            pairs[source_index - 1].alignment.selected
            if source_index > 0 and pairs[source_index - 1].alignment is not None
            else None
        )
        # Outside this calibrated-x interval cv.remap can only read the
        # constant -1 border, so no source-valid support can exist.  Keep a
        # ten-pixel halo: the hard gate permits an 8 px candidate displacement
        # and cv.remap needs its interpolation margin.  This avoids building a
        # mostly-invalid full-canvas map without excluding shifted support.
        support_x0 = max(
            0,
            int(math.floor(float(assignment.center_x) - float(calibration.cx))) - 10,
        )
        support_x1 = min(
            schedule.canvas_width,
            int(
                math.ceil(
                    float(assignment.center_x)
                    - float(calibration.cx)
                    + float(calibration.width)
                )
            )
            + 10,
        )
        support = _map_crop(
            schedule,
            calibration,
            source_index,
            support_x0,
            support_x1,
            vertical.global_offsets_px[source_index],
            candidate,
        )[2]
        expected_support[:, support_x0:support_x1] |= support
    expected_support.flags.writeable = False
    return provide_maps, provide, expected_support


def render_s13_p2_from_raw(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    pairs: Sequence[S13M5Pair],
    *,
    final_seams: bool,
    selected_hypothesis_ids: tuple[int, ...] | None = None,
    placement_methods: tuple[str, ...] | None = None,
    map_provider: S13SourceMapProvider | None = None,
    base_map_provider: S13SourceMapProvider | None = None,
    expected_support_provider: S13SourceMapProvider | None = None,
    expected_support_mask: np.ndarray | None = None,
    sampled_source_provider: S13SampledSourceProvider | None = None,
    image_composer: Callable[[tuple[int, ...], np.ndarray, dict[str, np.ndarray]], np.ndarray] | None = None,
) -> S13P2Result:
    """Formally remap each real contributor once from raw RGB for this P2 asset."""

    validate_s012_schedule(schedule)
    if len(pairs) != len(schedule.assignments) - 1:
        raise ValueError("S1.3 M5 pair transactions do not cover all render pairs")
    seams = _seams_array(schedule, pairs, final=final_seams)
    height, width = schedule.canvas_height, schedule.canvas_width
    owner_index = np.zeros((height, width), dtype=np.int32)
    for seam in seams:
        owner_index += np.arange(width, dtype=np.int32)[None, :] >= seam[:, None]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    valid_full = np.zeros((height, width), dtype=bool)
    owner = np.full((height, width), -1, dtype=np.int32)
    assignment_full = np.full((height, width), -1, dtype=np.int32)
    source_u_full = np.full((height, width), np.nan, dtype=np.float32)
    source_v_full = np.full((height, width), np.nan, dtype=np.float32)
    geometry_tx = np.full((height, width), -1, dtype=np.int32)
    seam_tx = np.full((height, width), -1, dtype=np.int32)
    hypothesis = np.full((height, width), -1, dtype=np.int32)
    placement = np.full((height, width), -1, dtype=np.int16)
    component_field = np.full((height, width), -1, dtype=np.int32)
    decoded: list[int] = []
    actual_remap_invocations = 0
    if expected_support_mask is None:
        expected_support = np.zeros((height, width), dtype=bool)
    else:
        expected_support = np.asarray(expected_support_mask, dtype=bool)
        if expected_support.shape != (height, width):
            raise ValueError("S1.3 M5 expected support cache shape changed")
        expected_support.setflags(write=False)
    for source_index, assignment in enumerate(schedule.assignments):
        candidate = None
        if source_index > 0 and pairs[source_index - 1].alignment is not None:
            candidate = pairs[source_index - 1].alignment.selected
        if expected_support_mask is None:
            support_provider = expected_support_provider or map_provider
            if support_provider is None:
                expected_maps = _map_crop(
                    schedule, calibration, source_index, 0, width,
                    vertical.global_offsets_px[source_index], candidate,
                )
            else:
                expected_maps = support_provider(source_index, 0, width)[:3]
            expected_support |= expected_maps[2]
        mask = owner_index == source_index
        if not np.any(mask):
            continue
        columns = np.flatnonzero(np.any(mask, axis=0))
        x0, x1 = int(columns[0]), int(columns[-1]) + 1
        if map_provider is None and base_map_provider is None:
            base_maps = _map_crop(
                schedule, calibration, source_index, x0, x1,
                vertical.global_offsets_px[source_index], candidate,
            )
            maps = base_maps
        elif map_provider is None:
            maps = base_map_provider(source_index, x0, x1)
        else:
            maps = map_provider(source_index, x0, x1)
        decoded.append(assignment.frame_id)
        if sampled_source_provider is not None:
            sampled, cached_valid, cache_miss = sampled_source_provider(
                source_index, x0, x1
            )
            map_valid = maps[2]
            if not np.array_equal(cached_valid, map_valid):
                raise ValueError("sampled source cache valid support changed")
            actual_remap_invocations += int(cache_miss)
        elif image_composer is None:
            raw = np.asarray(image_loader(assignment.frame_id))
            if raw.shape != (height, int(calibration.width), 3) or raw.dtype != np.uint8:
                raise ValueError("S1.3 M5 raw RGB source shape/type changed")
            sampled, map_valid = _sample_crop(raw, maps[:3])
            actual_remap_invocations += 1
        else:
            map_valid = maps[2]
        owned = mask[:, x0:x1] & map_valid
        roi = np.s_[:, x0:x1]
        if image_composer is None:
            output[roi][owned] = sampled[owned]
        valid_full[roi][owned] = True
        owner[roi][owned] = assignment.frame_id
        assignment_full[roi][owned] = source_index
        source_u_full[roi][owned] = maps[0][owned]
        source_v_full[roi][owned] = maps[1][owned]
        component_field[roi][owned] = np.asarray(maps[3], dtype=np.int32)[owned]
        if source_index > 0:
            pair_index = source_index - 1
            geometry_tx[roi][owned] = pair_index
            seam_tx[roi][owned] = pair_index
        if selected_hypothesis_ids is not None:
            hypothesis[roi][owned] = int(selected_hypothesis_ids[source_index])
        placement[roi][owned] = source_index
    if len(decoded) != len(set(decoded)):
        raise ValueError("S1.3 M5 decoded one contributor more than once")
    pixel = {
        "owner_frame_id": owner,
        "owner_source_index": assignment_full.copy(),
        "assignment_index": assignment_full,
        "source_u": source_u_full,
        "source_v": source_v_full,
        "valid": valid_full,
        "selected_motion_hypothesis_id": hypothesis,
        "placement_method_code": placement,
        "placement_method_names": np.asarray(placement_methods or ()),
        "geometry_transaction_id": geometry_tx,
        "seam_transaction_id": seam_tx,
        "photometric_transaction_id": np.full((height, width), -1, np.int32),
        "blend_transaction_id": np.full((height, width), -1, np.int32),
        "secondary_frame_id": np.full((height, width), -1, np.int32),
        "secondary_source_u": np.full((height, width), np.nan, np.float32),
        "secondary_source_v": np.full((height, width), np.nan, np.float32),
        "secondary_weight": np.zeros((height, width), np.float32),
    }
    if map_provider is not None:
        pixel["component_correction_field_id"] = component_field
    if not np.array_equal(valid_full, owner >= 0):
        raise ValueError("S1.3 M5 valid and owner topology disagree")
    finite = valid_full & (~np.isfinite(source_u_full) | ~np.isfinite(source_v_full))
    if np.any(finite):
        raise ValueError("S1.3 M5 valid source provenance is nonfinite")
    if image_composer is not None:
        output = np.asarray(image_composer(tuple(decoded), valid_full, pixel))
        if output.shape != (height, width, 3) or output.dtype != np.uint8:
            raise ValueError("S1.3 M5 resident P2 composer returned an invalid image")
    return S13P2Result(
        output, valid_full, pixel, len(decoded), tuple(decoded), expected_support,
        actual_remap_invocations,
    )


def _overlay_seams(image: np.ndarray, pairs: Sequence[S13M5Pair]) -> np.ndarray:
    overlay = image.copy()
    for pair in pairs:
        seam = np.asarray(pair.seam_x_by_row, dtype=np.int32)
        rows = np.arange(len(seam), dtype=np.int32)
        valid = (seam >= 0) & (seam < image.shape[1])
        overlay[rows[valid], seam[valid]] = (0, 0, 255)
    return overlay


def render_s13_component_roi_from_raw(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    pairs: Sequence[S13M5Pair],
    roi_xyxy: tuple[int, int, int, int],
    *,
    registry: SourceCorrectionRegistry | None,
    field_ids: Mapping[str, int] | None = None,
    preserve_vertical_parent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one owner-only validation ROI without a full-canvas render."""

    x0, y0, x1, y1 = roi_xyxy
    if not (0 <= x0 < x1 <= schedule.canvas_width):
        raise ValueError("S1.3 component ROI x-domain is invalid")
    if not (0 <= y0 < y1 <= schedule.canvas_height):
        raise ValueError("S1.3 component ROI y-domain is invalid")
    seams = _seams_array(schedule, pairs, final=True)[:, y0:y1]
    columns = np.arange(x0, x1, dtype=np.int32)[None, :]
    owner = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
    for seam in seams:
        owner += columns >= seam[:, None]
    output = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    valid_output = np.zeros(owner.shape, dtype=bool)
    for source_index in np.unique(owner):
        source = int(source_index)
        alignment = None
        if source > 0 and pairs[source - 1].alignment is not None:
            alignment = pairs[source - 1].alignment.selected
        corrections = tuple(
            () if registry is None
            else registry.corrections_by_source.get(source, ())
        )
        maps = _map_crop(
            schedule, calibration, source, x0, x1,
            vertical.global_offsets_px[source], alignment, corrections,
            field_ids,
            vertical_parent=vertical if preserve_vertical_parent else None,
        )
        sampled, valid = _sample_crop(
            np.asarray(image_loader(int(schedule.assignments[source].frame_id))),
            tuple(row[y0:y1] for row in maps),
        )
        selected = (owner == source) & valid
        output[selected] = sampled[selected]
        valid_output[selected] = True
    return output, valid_output


def run_s13_m5(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    vertical: S13VerticalSolution,
    vertical_parent_image: np.ndarray,
    *,
    parent_stage_sha256: str,
    parent_result_sha256: str | None = None,
    p0_ancestor_completion_sha256: str | None = None,
    selected_hypothesis_ids: tuple[int, ...] | None = None,
    placement_methods: tuple[str, ...] | None = None,
    m51_r2_config: S13M51R2Config | None = None,
    final_image_composer: Callable[[tuple[int, ...], np.ndarray, dict[str, np.ndarray]], np.ndarray] | None = None,
) -> S13M5Result:
    started = time.perf_counter()
    tick = time.perf_counter()
    raw_cache: dict[int, np.ndarray] = {}

    def cached_image_loader(frame_id: int) -> np.ndarray:
        if frame_id not in raw_cache:
            # The caller's frame store owns the contiguous BGR source for the
            # whole run.  M5 never mutates it, so a read-only view preserves
            # the component-chain immutability contract without a second
            # full-resolution BGR allocation/copy per contributor.
            shared = np.ascontiguousarray(image_loader(frame_id)).view()
            shared.flags.writeable = False
            raw_cache[frame_id] = shared
        return raw_cache[frame_id]

    pairs = estimate_s13_m5_transactions(
        schedule, calibration, cached_image_loader, vertical,
        parent_stage_sha256=parent_stage_sha256,
        parent_result_sha256=parent_result_sha256,
        p0_ancestor_completion_sha256=p0_ancestor_completion_sha256,
        m51_r2_config=m51_r2_config,
    )
    component_chain_audit: dict[str, object] | None = None
    component_chain_seconds = 0.0
    component_patch_set: S13ComponentPatchSet | None = None
    source_correction_registry: SourceCorrectionRegistry | None = None
    component_field_ids: dict[str, int] = {}
    geometry_seconds = 0.0
    roi_candidate_pixels = 0
    roi_preview_count = 0
    component_linear_solve_count = 0
    component_gain_candidate_count = 0
    component_segment_candidate_count = 0
    component_split_count = 0
    p2_full_resolution_render_count = 0
    estimation_result: S13M5EstimationResult | None = None
    if isinstance(m51_r2_config, S13M51R4Config):
        component_started = time.perf_counter()
        geometry_seconds = component_started - tick
        observations = [
            row[0] for pair in pairs for row in pair.component_evidence
        ]
        evidence_by_component = {
            (observation.pair_index, observation.component_id): evidence
            for pair in pairs for observation, evidence in pair.component_evidence
        }
        support_by_authority = {
            (observation.pair_index, observation.component_id, observation.mask_sha256): (
                evidence.support_xy
            )
            for pair in pairs for observation, evidence in pair.component_evidence
        }
        obligations = freeze_s13_baseline_c2e_obligations(
            observations,
            support_by_authority=support_by_authority,
            severe_threshold_px=m51_r2_config.maximum_post_edge_p95_px,
            minimum_evaluable_columns=m51_r2_config.minimum_evaluable_edge_columns,
        )
        component_match_matrices: list[Mapping[str, object]] = []
        all_chains = build_s13_edge_component_chains(
            observations, config=m51_r2_config,
            match_matrix_sink=component_match_matrices,
        )
        chains = tuple(
            chain for chain in all_chains
            if len(set(chain.pair_indices)) >= m51_r2_config.minimum_chain_pair_count
            and any(
                item.evidence_state == "actionable"
                and abs(item.forward_best_lag_px) > 1.0
                for item in chain.observations
            )
        )
        segments = tuple(
            segment
            for chain in chains
            for segment in split_s13_edge_component_chain(chain, config=m51_r2_config)
        )
        solved_rows: list[dict[str, object]] = []
        segment_candidates: list[S13ComponentSegmentCandidate] = []
        def measure_candidate_roi(
            segment_observations: Sequence[object],
            corrections: Sequence[object],
        ) -> tuple[
            dict[str, float | bool], tuple[object, ...], dict[str, object]
        ]:
            nonlocal roi_candidate_pixels, roi_preview_count
            measured = []
            baseline_traces = []
            candidate_traces = []
            source_map_audits: list[dict[str, object]] = []
            horizontal_audits: list[dict[str, object]] = []
            field_ids = {
                row.segment_id: index
                for index, row in enumerate(sorted(
                    corrections, key=lambda item: (item.segment_id, item.source_index)
                ))
            }
            halo = int(math.ceil(
                m51_r2_config.normal_search_maximum_px
                + m51_r2_config.normal_taper_radius_px
            ))
            for observation in segment_observations:
                evidence = evidence_by_component.get(
                    (observation.pair_index, observation.component_id)
                )
                if evidence is None:
                    raise ValueError("segment ROI evidence is missing")
                bx0, by0, bx1, by1 = observation.global_bbox_xyxy
                x0 = max(0, bx0 - halo)
                x1 = min(schedule.canvas_width, bx1 + halo)
                y0 = max(0, by0 - halo)
                y1 = min(schedule.canvas_height, by1 + halo)
                if x1 <= x0 or y1 <= y0:
                    raise ValueError("segment ROI domain is empty")
                source_features = []
                base_source_images = []
                candidate_source_images = []
                base_source_valid = []
                candidate_source_valid = []
                for source_index in observation.source_indices:
                    alignment = None
                    if source_index > 0 and pairs[source_index - 1].alignment is not None:
                        alignment = pairs[source_index - 1].alignment.selected
                    source_corrections = tuple(
                        row for row in corrections if row.source_index == source_index
                    )
                    base_maps = _map_crop(
                        schedule, calibration, source_index, x0, x1,
                        vertical.global_offsets_px[source_index], alignment,
                        vertical_parent=vertical,
                    )
                    maps = _map_crop(
                        schedule, calibration, source_index, x0, x1,
                        vertical.global_offsets_px[source_index], alignment,
                        source_corrections, field_ids, vertical_parent=vertical,
                    )
                    sliced_base_maps = tuple(row[y0:y1] for row in base_maps)
                    sliced_maps = tuple(row[y0:y1] for row in maps)
                    raw_source = cached_image_loader(
                        int(schedule.assignments[source_index].frame_id)
                    )
                    base_sampled, base_valid = _sample_crop(
                        raw_source, sliced_base_maps
                    )
                    sampled, candidate_valid = _sample_crop(raw_source, sliced_maps)
                    features = prepare_seam_structure(sampled)
                    if features is None:
                        raise ValueError("segment ROI feature construction failed")
                    source_features.append(features)
                    base_source_images.append(base_sampled)
                    candidate_source_images.append(sampled)
                    base_source_valid.append(base_valid)
                    candidate_source_valid.append(candidate_valid)

                    evidence_mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
                    support_xy = np.asarray(evidence.support_xy, dtype=np.int32)
                    local_x = support_xy[:, 0] - x0
                    local_y = support_xy[:, 1] - y0
                    inside = (
                        (local_x >= 0) & (local_x < x1 - x0)
                        & (local_y >= 0) & (local_y < y1 - y0)
                    )
                    evidence_mask[local_y[inside], local_x[inside]] = True
                    influence_mask = np.zeros_like(evidence_mask)
                    for correction in source_corrections:
                        overlap_x0 = max(x0, int(correction.x0))
                        overlap_x1 = min(x1, int(correction.x1))
                        overlap_y0 = max(y0, int(correction.y0))
                        overlap_y1 = min(y1, int(correction.y1))
                        if overlap_x1 <= overlap_x0 or overlap_y1 <= overlap_y0:
                            continue
                        target = np.s_[
                            overlap_y0 - y0:overlap_y1 - y0,
                            overlap_x0 - x0:overlap_x1 - x0,
                        ]
                        source = np.s_[
                            overlap_y0 - int(correction.y0):overlap_y1 - int(correction.y0),
                            overlap_x0 - int(correction.x0):overlap_x1 - int(correction.x0),
                        ]
                        influence_mask[target] |= correction.weight[source] > 0.0
                    owner_mask = (
                        gauge_owner[y0:y1, x0:x1] == int(source_index)
                    )
                    source_audit = audit_s13_component_candidate_map_safety(
                        base_u=sliced_base_maps[0],
                        base_v=sliced_base_maps[1],
                        base_valid=sliced_base_maps[2],
                        candidate_u=sliced_maps[0],
                        candidate_v=sliced_maps[1],
                        candidate_valid=sliced_maps[2],
                        owner_mask=owner_mask,
                        evidence_mask=evidence_mask,
                        influence_mask=influence_mask,
                        source_size=(int(calibration.width), int(calibration.height)),
                        minimum_owner_retention=(
                            m51_r2_config.minimum_formal_owner_support_retention
                        ),
                        minimum_evidence_retention=(
                            m51_r2_config.minimum_evidence_sample_retention
                        ),
                        minimum_jacobian=m51_r2_config.minimum_jacobian,
                        maximum_displacement_px=(
                            m51_r2_config.maximum_combined_map_displacement_px
                        ),
                        maximum_halo_regression_px=(
                            m51_r2_config.maximum_halo_regression_px
                        ),
                    )
                    source_map_audits.append({
                        "pair_index": int(observation.pair_index),
                        "component_id": int(observation.component_id),
                        "source_index": int(source_index),
                        **dict(source_audit),
                    })
                    roi_candidate_pixels += int((x1 - x0) * (y1 - y0))
                if roi_candidate_pixels > m51_r2_config.maximum_total_roi_candidate_pixels:
                    raise ValueError("component ROI candidate pixel budget exceeded")
                support = np.asarray(evidence.support_xy, dtype=np.int32).copy()
                support[:, 0] -= x0
                support[:, 1] -= y0
                left_features, right_features = source_features
                measured_observation, _measured_evidence = (
                    make_s13_edge_component_observation(
                        pair_index=observation.pair_index,
                        component_id=observation.component_id,
                        source_indices=observation.source_indices,
                        support_xy=support,
                        left_magnitude=left_features.gradient,
                        right_magnitude=right_features.gradient,
                        left_gradient_x=left_features.gradient_x,
                        left_gradient_y=left_features.gradient_y,
                        right_gradient_x=right_features.gradient_x,
                        right_gradient_y=right_features.gradient_y,
                        minimum_correlation=m51_r2_config.minimum_c2e_correlation,
                        minimum_uniqueness_fraction=(
                            m51_r2_config.minimum_c2e_uniqueness_fraction
                        ),
                        maximum_forward_reverse_discrepancy_px=(
                            m51_r2_config.maximum_forward_reverse_discrepancy_px
                        ),
                    )
                )
                measured.append(measured_observation)
                seam_local = (
                    np.asarray(
                        pairs[int(observation.pair_index)].seam_x_by_row[y0:y1],
                        dtype=np.int32,
                    ) - x0
                )
                before_preview = _compose_pair_preview(
                    base_source_images[0], base_source_images[1], seam_local
                )
                after_preview = _compose_pair_preview(
                    candidate_source_images[0], candidate_source_images[1], seam_local
                )
                preview_columns = np.arange(x1 - x0, dtype=np.int32)[None, :]
                owner_right = preview_columns >= seam_local[:, None]
                baseline_preview_valid = np.where(
                    owner_right, base_source_valid[1], base_source_valid[0]
                )
                candidate_preview_valid = np.where(
                    owner_right, candidate_source_valid[1], candidate_source_valid[0]
                )
                # Trace only this compact observation ROI. Exact support,
                # normal, and polarity are frozen from baseline evidence and
                # shared by the baseline/candidate owner-only previews.
                trace_support = np.asarray(evidence.support_xy, dtype=np.int32).copy()
                trace_support[:, 0] -= x0
                trace_support[:, 1] -= y0
                trace_normal = (
                    float(observation.normal_x), float(observation.normal_y)
                )
                baseline_traces.append(trace_s13_owner_only_component_edge(
                    before_preview,
                    baseline_preview_valid,
                    support_xy=trace_support,
                    normal_xy=trace_normal,
                    signed_gradient_polarity=float(
                        observation.signed_gradient_polarity
                    ),
                    seam_x_by_row=seam_local,
                    search_radius_px=max(1, min(6, halo)),
                ))
                candidate_traces.append(trace_s13_owner_only_component_edge(
                    after_preview,
                    candidate_preview_valid,
                    support_xy=trace_support,
                    normal_xy=trace_normal,
                    signed_gradient_polarity=float(
                        observation.signed_gradient_polarity
                    ),
                    seam_x_by_row=seam_local,
                    search_radius_px=max(1, min(6, halo)),
                ))
                before_features = prepare_seam_structure(before_preview)
                after_features = prepare_seam_structure(after_preview)
                if before_features is None or after_features is None:
                    raise ValueError("segment horizontal preview feature construction failed")
                before_horizontal = long_horizontal_structure_metrics(
                    before_features, seam_local
                )
                after_horizontal = long_horizontal_structure_metrics(
                    after_features, seam_local
                )
                horizontal_safe, horizontal_reason, horizontal_audit = (
                    long_horizontal_structure_catastrophe_guard(
                        before_horizontal, after_horizontal
                    )
                )
                horizontal_nondegrading, horizontal_nondegrading_reason = (
                    long_horizontal_structure_nondegrading(
                        before_horizontal, after_horizontal
                    )
                )
                preexisting_catastrophe_nondegrading = bool(
                    not horizontal_safe and horizontal_nondegrading
                )
                horizontal_safe = bool(
                    horizontal_safe or preexisting_catastrophe_nondegrading
                )
                if preexisting_catastrophe_nondegrading:
                    horizontal_reason = None
                horizontal_audits.append({
                    "pair_index": int(observation.pair_index),
                    "component_id": int(observation.component_id),
                    "passed": horizontal_safe,
                    "reason": horizontal_reason,
                    "audit": horizontal_audit,
                    "nondegrading_passed": horizontal_nondegrading,
                    "nondegrading_reason": horizontal_nondegrading_reason,
                    "preexisting_catastrophe_nondegrading": (
                        preexisting_catastrophe_nondegrading
                    ),
                    "before": before_horizontal,
                    "after": after_horizontal,
                })
            roi_preview_count += 1
            safety = {
                "passed": bool(
                    source_map_audits
                    and all(row["passed"] is True for row in source_map_audits)
                    and all(row["passed"] is True for row in horizontal_audits)
                ),
                "source_map_audits": source_map_audits,
                "horizontal_audits": horizontal_audits,
                "formal_owner_support_retention": min(
                    float(row["formal_owner_support_retention"])
                    for row in source_map_audits
                ),
                "evidence_sample_retention": min(
                    float(row["evidence_sample_retention"])
                    for row in source_map_audits
                ),
                "minimum_final_inverse_map_jacobian": min(
                    float(row["candidate_minimum_inverse_map_jacobian"])
                    for row in source_map_audits
                ),
                "minimum_inverse_map_jacobian_ratio": min(
                    (
                        float(row["minimum_inverse_map_jacobian_ratio"])
                        for row in source_map_audits
                        if row["minimum_inverse_map_jacobian_ratio"] is not None
                    ),
                    default=None,
                ),
                "maximum_combined_map_displacement_px": max(
                    float(row["maximum_combined_map_displacement_px"])
                    for row in source_map_audits
                ),
                "maximum_halo_regression_px": max(
                    float(row["halo_regression_px"])
                    for row in source_map_audits
                ),
                "maximum_segment_boundary_step_px": max(
                    float(row["segment_boundary_step_px"])
                    for row in source_map_audits
                ),
                "maximum_segment_boundary_curvature_spike_px": max(
                    float(row["segment_boundary_curvature_spike_px"])
                    for row in source_map_audits
                ),
            }
            return (
                {
                    "baseline_runtime_trace": dict(
                        aggregate_s13_runtime_edge_traces(
                            baseline_traces,
                            search_boundary_hit=any(
                                row.search_boundary_hit for row in segment_observations
                            ),
                        )
                    ),
                    **dict(aggregate_s13_runtime_edge_traces(
                        candidate_traces,
                        search_boundary_hit=any(
                            row.search_boundary_hit for row in measured
                        ),
                    )),
                },
                tuple(measured),
                safety,
            )

        chain_by_id = {chain.chain_id: chain for chain in chains}
        final_seams_for_gauge = _seams_array(schedule, pairs, final=True)
        gauge_owner = np.zeros(
            (schedule.canvas_height, schedule.canvas_width), dtype=np.int32
        )
        for seam in final_seams_for_gauge:
            gauge_owner += np.arange(schedule.canvas_width)[None, :] >= seam[:, None]
        def split_pair_audits(
            segment: object,
            *,
            candidate_observations: Sequence[object] = (),
            offsets: Mapping[int, float] | None = None,
            hard_failure: bool = False,
        ) -> tuple[dict[str, object], ...]:
            candidate_by_pair = {
                int(row.pair_index): row for row in candidate_observations
            }
            audits: list[dict[str, object]] = []
            for observation in segment.observations:
                pair_index = int(observation.pair_index)
                candidate = candidate_by_pair.get(pair_index)
                candidate_lag = (
                    abs(0.5 * (
                        float(candidate.forward_best_lag_px)
                        - float(candidate.reverse_best_lag_px)
                    ))
                    if candidate is not None else abs(0.5 * (
                        float(observation.forward_best_lag_px)
                        - float(observation.reverse_best_lag_px)
                    ))
                )
                hard_violations = int(hard_failure)
                gate_excesses = [
                    max(
                        0.0,
                        candidate_lag - m51_r2_config.maximum_post_edge_p95_px,
                    ) / max(abs(m51_r2_config.maximum_post_edge_p95_px), 1e-6)
                ]
                if candidate is not None:
                    if candidate.evidence_state in {"ambiguous", "unevaluable"}:
                        hard_violations += 1
                    discrepancy = abs(
                        float(candidate.forward_best_lag_px)
                        + float(candidate.reverse_best_lag_px)
                    )
                    gate_excesses.append(max(
                        0.0,
                        discrepancy
                        - m51_r2_config.maximum_forward_reverse_discrepancy_px,
                    ) / max(
                        abs(m51_r2_config.maximum_forward_reverse_discrepancy_px),
                        1e-6,
                    ))
                    gate_excesses.append(max(
                        0.0,
                        m51_r2_config.minimum_c2e_correlation
                        - float(candidate.correlation),
                    ) / max(abs(m51_r2_config.minimum_c2e_correlation), 1e-6))
                    gate_excesses.append(max(
                        0.0,
                        m51_r2_config.minimum_c2e_uniqueness_fraction
                        - float(candidate.uniqueness_fraction),
                    ) / max(
                        abs(m51_r2_config.minimum_c2e_uniqueness_fraction),
                        1e-6,
                    ))
                normalized_residual = 0.0
                if offsets is not None:
                    left_source, right_source = observation.source_indices
                    residual = (
                        float(offsets[right_source])
                        - float(offsets[left_source])
                        + 0.5 * (
                            float(observation.forward_best_lag_px)
                            - float(observation.reverse_best_lag_px)
                        )
                    )
                    normalized_residual = abs(residual) / max(
                        m51_r2_config.normal_search_step_px, 1e-6
                    )
                audits.append({
                    "pair_index": pair_index,
                    "hard_violation_count": hard_violations,
                    "maximum_normalized_gate_excess": max(gate_excesses),
                    "normalized_solver_residual": normalized_residual,
                    "baseline_edge_excess_px": max(
                        0.0,
                        abs(0.5 * (
                            float(observation.forward_best_lag_px)
                            - float(observation.reverse_best_lag_px)
                        )) - m51_r2_config.maximum_post_edge_p95_px,
                    ),
                })
            return tuple(audits)

        def segment_budget_priority(segment: object) -> tuple[object, ...]:
            return s13_component_segment_budget_priority(
                segment,
                evidence_by_component=evidence_by_component,
                maximum_post_edge_p95_px=(
                    m51_r2_config.maximum_post_edge_p95_px
                ),
            )

        def segment_roi_candidate_cost(segment: object) -> int:
            halo = int(math.ceil(
                m51_r2_config.normal_search_maximum_px
                + m51_r2_config.normal_taper_radius_px
            ))
            per_gain = 0
            for row in segment.observations:
                bx0, by0, bx1, by1 = row.global_bbox_xyxy
                width = max(0, min(schedule.canvas_width, bx1 + halo) - max(0, bx0 - halo))
                height = max(0, min(schedule.canvas_height, by1 + halo) - max(0, by0 - halo))
                per_gain += width * height * len(row.source_indices)
            return per_gain * len(m51_r2_config.correction_gain_candidates)

        pending_segments = [
            (segment, 0, None, segment.segment_id) for segment in segments
        ]
        pending_segments.sort(key=lambda row: segment_budget_priority(row[0]))
        split_lineage: list[dict[str, object]] = []
        while pending_segments:
            segment, split_depth, parent_segment_id, root_segment_id = (
                pending_segments.pop(0)
            )
            component_segment_candidate_count += 1
            selected: S13ComponentSegmentCandidate | None = None
            candidate_row: S13ComponentSegmentCandidate | None = None
            failure_candidate_audit: dict[str, object] = {}
            failure_reason: str | None = None
            failure_pair_audits: tuple[dict[str, object], ...] = ()
            estimated_roi_cost = segment_roi_candidate_cost(segment)
            if (
                roi_candidate_pixels + estimated_roi_cost
                > m51_r2_config.maximum_total_roi_candidate_pixels
            ):
                failure_reason = "candidate pixel budget exceeded before evaluation"
            try:
                if failure_reason is not None:
                    raise ValueError(failure_reason)
                x0, y0, x1, y1 = segment.global_bbox_xyxy
                owner_support = {
                    source: int(np.count_nonzero(
                        gauge_owner[y0:y1, x0:x1] == source
                    ))
                    for source in segment.source_indices
                }
                component_linear_solve_count += 1
                offsets = solve_s13_component_segment_source_offsets(
                    segment, config=m51_r2_config,
                    owner_support_by_source=owner_support,
                )
                chain = chain_by_id[segment.parent_chain_id]
                lags = np.asarray([
                    0.5 * (row.forward_best_lag_px - row.reverse_best_lag_px)
                    for row in segment.observations
                ], dtype=np.float64)
                baseline_metrics = _component_trace_metrics(
                    segment.observations, config=m51_r2_config
                )
                passing_gain_candidates: list[S13ComponentSegmentCandidate] = []
                for gain in m51_r2_config.correction_gain_candidates:
                    component_gain_candidate_count += 1
                    corrections = []
                    for source in segment.source_indices:
                        support_mask = np.zeros(
                            (schedule.canvas_height, schedule.canvas_width), dtype=bool
                        )
                        for observation in segment.observations:
                            evidence = evidence_by_component.get(
                                (observation.pair_index, observation.component_id)
                            )
                            if evidence is None or source not in observation.source_indices:
                                continue
                            xy = np.asarray(evidence.support_xy, dtype=np.int32)
                            support_mask[xy[:, 1], xy[:, 0]] = True
                        if not np.any(support_mask):
                            raise ValueError("segment_source_support_missing")
                        corrections.append(build_s13_source_component_correction(
                            segment_id=segment.segment_id,
                            source_index=source,
                            frame_id=int(schedule.assignments[source].frame_id),
                            support_mask=support_mask,
                            source_offset_px=float(offsets[source]) * float(gain),
                            normal_xy=chain.canonical_normal_xy,
                            config=m51_r2_config,
                            local_normal_observations=tuple(
                                observation for observation in segment.observations
                                if source in observation.source_indices
                            ),
                        ))
                    candidate_metrics, measured_observations, candidate_safety = (
                        measure_candidate_roi(
                        segment.observations, corrections
                        )
                    )
                    runtime_baseline = dict(
                        candidate_metrics.pop("baseline_runtime_trace")
                    )
                    baseline_metrics = runtime_baseline
                    quality = _evaluate_s13_runtime_component_candidate(
                        baseline=baseline_metrics,
                        candidate=candidate_metrics,
                        hard_gates={
                            "forward_reverse": all(
                                abs(row.forward_best_lag_px + row.reverse_best_lag_px)
                                <= m51_r2_config.maximum_forward_reverse_discrepancy_px
                                for row in segment.observations
                            ),
                            "offset_bounds": all(
                                abs(float(value) * float(gain))
                                <= m51_r2_config.maximum_source_normal_offset_px
                                for value in offsets.values()
                            ),
                            "owner_support": all(value > 0 for value in owner_support.values()),
                            "formal_owner_retention": (
                                candidate_safety["formal_owner_support_retention"]
                                >= m51_r2_config.minimum_formal_owner_support_retention
                            ),
                            "candidate_correlation": all(
                                float(row.correlation)
                                >= m51_r2_config.minimum_c2e_correlation
                                for row in measured_observations
                            ),
                            "candidate_uniqueness": all(
                                float(row.uniqueness_fraction)
                                >= m51_r2_config.minimum_c2e_uniqueness_fraction
                                for row in measured_observations
                            ),
                            "runtime_trace_columns": (
                                int(candidate_metrics["evaluable_edge_columns"])
                                >= m51_r2_config.minimum_evaluable_edge_columns
                            ),
                            "evidence_retention": (
                                candidate_safety["evidence_sample_retention"]
                                >= m51_r2_config.minimum_evidence_sample_retention
                            ),
                            "final_inverse_map": candidate_safety["passed"] is True,
                        },
                        config=m51_r2_config,
                    )
                    audit = {
                        **dict(quality.audit),
                        "rescued_severe_seam_count": sum(abs(value) > 1.0 for value in lags),
                        "worst_seam_absolute_improvement": float(
                            np.max(np.abs(lags))
                            - float(candidate_metrics["maximum_step_px"])
                        ),
                        "supported_unique_edge_columns": int(
                            sum(row.reference_support_samples for row in segment.observations)
                        ),
                        "post_maximum_step": candidate_metrics["maximum_step_px"],
                        "correction_energy": float(sum(
                            np.sum(row.delta_u.astype(np.float64) ** 2 + row.delta_v.astype(np.float64) ** 2)
                            for row in corrections
                        )),
                        "baseline_metrics": baseline_metrics,
                        "candidate_metrics": candidate_metrics,
                        "candidate_map_safety": candidate_safety,
                        "local_normal_field_audits": [
                            dict(row.normal_field_audit) for row in corrections
                        ],
                    }
                    candidate_row = S13ComponentSegmentCandidate(
                        segment=segment,
                        source_offsets_px=tuple(float(offsets[source]) for source in segment.source_indices),
                        gain=float(gain),
                        corrections=tuple(corrections),
                        decision=quality.decision,
                        rejection_reasons=quality.rejection_reasons,
                        audit=audit,
                    )
                    if candidate_row.decision in {"resolved", "improved_unresolved"}:
                        passing_gain_candidates.append(candidate_row)
                selected = _select_s13_component_gain_candidate(
                    passing_gain_candidates
                )
                if selected is None:
                    if candidate_row is None:
                        raise ValueError("segment_has_no_gain_candidate")
                    failure_reason = ";".join(candidate_row.rejection_reasons) or (
                        "candidate_quality_rejected"
                    )
                    failure_candidate_audit = dict(candidate_row.audit)
                    failure_pair_audits = split_pair_audits(
                        segment,
                        candidate_observations=measured_observations,
                        offsets=offsets,
                    )
                else:
                    segment_candidates.append(selected)
                    solved_rows.append({
                    "segment_id": segment.segment_id,
                    "parent_segment_id": parent_segment_id,
                    "root_segment_id": root_segment_id,
                    "parent_chain_id": segment.parent_chain_id,
                    "split_depth": split_depth,
                    "left_cut_reason": segment.left_cut_reason,
                    "right_cut_reason": segment.right_cut_reason,
                    "pair_indices": list(segment.pair_indices),
                    "source_indices": list(segment.source_indices),
                    "observation_authority": [
                        {
                            "pair_index": row.pair_index,
                            "component_id": row.component_id,
                            "support_sha256": row.mask_sha256,
                        }
                        for row in segment.observations
                    ],
                    "source_offsets_px": {str(key): value for key, value in offsets.items()},
                    "gain": selected.gain,
                    "state": selected.decision,
                    "reason": None if not selected.rejection_reasons else list(selected.rejection_reasons),
                    "audit": dict(selected.audit),
                })
            except ValueError as exc:
                failure_reason = f"solver_rejected:{exc}"
                failure_pair_audits = tuple(
                    dict(row) for row in exc.pair_audits
                ) if isinstance(exc, S13ComponentSolverError) else split_pair_audits(
                    segment, hard_failure=True,
                )

            if selected is not None:
                continue
            budget_deferred = bool(
                failure_reason
                and "candidate pixel budget exceeded" in failure_reason
            )
            children = ()
            split_pair_index: int | None = None
            if not budget_deferred:
                failure_kind = (
                    "solver"
                    if str(failure_reason).startswith("solver_rejected:")
                    else "quality"
                )
                split_pair_index, children = plan_s13_failed_segment_split(
                    segment,
                    pair_audits=failure_pair_audits,
                    reason=str(failure_reason or "candidate_gate_failed"),
                    split_depth=split_depth,
                    failure_kind=failure_kind,
                    config=m51_r2_config,
                )
            if children:
                component_split_count += 1
                child_ids = [child.segment_id for child in children]
                split_row = {
                    "segment_id": segment.segment_id,
                    "parent_segment_id": parent_segment_id,
                    "root_segment_id": root_segment_id,
                    "parent_chain_id": segment.parent_chain_id,
                    "split_depth": split_depth,
                    "left_cut_reason": segment.left_cut_reason,
                    "right_cut_reason": segment.right_cut_reason,
                    "pair_indices": list(segment.pair_indices),
                    "source_indices": list(segment.source_indices),
                    "state": "split",
                    "reason": failure_reason,
                    "split_pair_index": split_pair_index,
                    "child_segment_ids": child_ids,
                    "split_pair_audits": list(failure_pair_audits),
                    "audit": failure_candidate_audit,
                }
                solved_rows.append(split_row)
                split_lineage.append(split_row)
                pending_segments.extend(
                    (child, split_depth + 1, segment.segment_id, root_segment_id)
                    for child in children
                )
                pending_segments.sort(key=lambda row: segment_budget_priority(row[0]))
                continue

            decision = "budget_deferred" if budget_deferred else "rejected"
            terminal = S13ComponentSegmentCandidate(
                segment=segment,
                source_offsets_px=tuple(0.0 for _source in segment.source_indices),
                gain=1.0,
                corrections=(),
                decision=decision,
                rejection_reasons=(str(failure_reason or "segment_rejected"),),
                audit={
                    "split_depth": split_depth,
                    "split_exhausted": not budget_deferred,
                    "split_pair_audits": list(failure_pair_audits),
                },
            )
            segment_candidates.append(terminal)
            solved_rows.append({
                "segment_id": segment.segment_id,
                "parent_segment_id": parent_segment_id,
                "root_segment_id": root_segment_id,
                "parent_chain_id": segment.parent_chain_id,
                "split_depth": split_depth,
                "left_cut_reason": segment.left_cut_reason,
                "right_cut_reason": segment.right_cut_reason,
                "pair_indices": list(segment.pair_indices),
                "source_indices": list(segment.source_indices),
                "observation_authority": [
                    {
                        "pair_index": row.pair_index,
                        "component_id": row.component_id,
                        "support_sha256": row.mask_sha256,
                    }
                    for row in segment.observations
                ],
                "state": decision,
                "reason": failure_reason,
                "split_pair_audits": list(failure_pair_audits),
                "audit": failure_candidate_audit,
            })
        dependency_group_audits: list[dict[str, object]] = []
        dependency_failure_lineage: list[dict[str, object]] = []
        while True:
            component_patch_set = select_s13_component_patch_set(
                segment_candidates, config=m51_r2_config, obligations=obligations
            )
            accepted_by_id = {
                row.segment.segment_id: row for row in segment_candidates
                if row.segment.segment_id in component_patch_set.accepted_segment_ids
            }
            pending_ids = set(accepted_by_id)
            dependency_groups: list[tuple[str, ...]] = []
            while pending_ids:
                seed = min(pending_ids)
                group = {seed}
                frontier = [seed]
                pending_ids.remove(seed)
                while frontier:
                    current = accepted_by_id[frontier.pop()]
                    current_sources = set(current.segment.source_indices)
                    current_pairs = {
                        pair
                        for source in current_sources
                        for pair in (source - 1, source)
                        if 0 <= pair < len(pairs)
                    }
                    linked = []
                    for other_id in sorted(pending_ids):
                        other = accepted_by_id[other_id]
                        other_sources = set(other.segment.source_indices)
                        other_pairs = {
                            pair
                            for source in other_sources
                            for pair in (source - 1, source)
                            if 0 <= pair < len(pairs)
                        }
                        if current_sources & other_sources or current_pairs & other_pairs:
                            linked.append(other_id)
                    for other_id in linked:
                        pending_ids.remove(other_id)
                        group.add(other_id)
                        frontier.append(other_id)
                dependency_groups.append(tuple(sorted(group)))
            dependency_group_audits = []
            failed_group: tuple[str, ...] | None = None
            for group_index, group_ids in enumerate(dependency_groups):
                group_candidates = [accepted_by_id[row] for row in group_ids]
                group_observations = tuple(
                    observation for candidate in group_candidates
                    for observation in candidate.segment.observations
                )
                group_corrections = tuple(
                    correction for candidate in group_candidates
                    for correction in candidate.corrections
                )
                candidate, _measured_group_observations, group_safety = measure_candidate_roi(
                    group_observations, group_corrections
                )
                baseline = dict(candidate.pop("baseline_runtime_trace"))
                quality = _evaluate_s13_runtime_component_candidate(
                    baseline=baseline,
                    candidate=candidate,
                    hard_gates={
                        "individual_segment_gates": all(
                            row.decision in {"resolved", "improved_unresolved"}
                            for row in group_candidates
                        ),
                        "exclusive_correction_fields": True,
                        "composite_map_safety": group_safety["passed"] is True,
                        "candidate_correlation": all(
                            float(row.correlation)
                            >= m51_r2_config.minimum_c2e_correlation
                            for row in _measured_group_observations
                        ),
                        "candidate_uniqueness": all(
                            float(row.uniqueness_fraction)
                            >= m51_r2_config.minimum_c2e_uniqueness_fraction
                            for row in _measured_group_observations
                        ),
                    },
                    config=m51_r2_config,
                )
                passed = quality.decision in {"resolved", "improved_unresolved"}
                dependency_group_audits.append({
                    "group_id": f"c2e-dependency-{group_index:04d}",
                    "segment_ids": list(group_ids),
                    "source_indices": sorted({
                        source for row in group_candidates
                        for source in row.segment.source_indices
                    }),
                    "affected_pair_indices": sorted({
                        pair
                        for row in group_candidates
                        for source in row.segment.source_indices
                        for pair in (source - 1, source)
                        if 0 <= pair < len(pairs)
                    }),
                    "baseline_metrics": baseline,
                    "candidate_metrics": candidate,
                    "candidate_map_safety": group_safety,
                    "decision": quality.decision,
                    "rejection_reasons": list(quality.rejection_reasons),
                    "passed": passed,
                })
                if not passed:
                    failed_group = group_ids
                    break
            if failed_group is None:
                break
            failed_candidates = {
                row: accepted_by_id[row] for row in failed_group
            }
            utility_by_candidate = {
                segment_id: (
                    float(row.audit.get("rescued_severe_seam_count", 0)),
                    float(row.audit.get("worst_seam_absolute_improvement", 0.0)),
                    float(row.audit.get("supported_unique_edge_columns", 0)),
                    -float(row.audit.get("post_maximum_step", math.inf)),
                    -float(row.audit.get("correction_energy", math.inf)),
                )
                for segment_id, row in failed_candidates.items()
            }
            pixel_support_by_candidate: dict[str, np.ndarray] = {}
            for segment_id, row in failed_candidates.items():
                supports: list[np.ndarray] = []
                for correction in row.corrections:
                    yy, xx = np.nonzero(np.asarray(correction.weight) > 0.0)
                    supports.append(np.column_stack((
                        xx.astype(np.int32) + int(correction.x0),
                        yy.astype(np.int32) + int(correction.y0),
                    )))
                pixel_support_by_candidate[segment_id] = (
                    np.unique(np.concatenate(supports), axis=0)
                    if supports else np.empty((0, 2), np.int32)
                )
            subset_cache: dict[tuple[str, ...], tuple[bool, Mapping[str, object]]] = {}

            def evaluate_dependency_subset(
                subset_ids: tuple[str, ...],
            ) -> tuple[bool, Mapping[str, object]]:
                cached = subset_cache.get(subset_ids)
                if cached is not None:
                    return cached
                subset = [failed_candidates[row] for row in subset_ids]
                subset_observations = tuple(
                    observation for candidate in subset
                    for observation in candidate.segment.observations
                )
                subset_corrections = tuple(
                    correction for candidate in subset
                    for correction in candidate.corrections
                )
                candidate, measured, safety = measure_candidate_roi(
                    subset_observations, subset_corrections
                )
                baseline = dict(candidate.pop("baseline_runtime_trace"))
                evaluation = _evaluate_s13_runtime_component_candidate(
                    baseline=baseline,
                    candidate=candidate,
                    hard_gates={
                        "individual_segment_gates": all(
                            row.decision in {"resolved", "improved_unresolved"}
                            for row in subset
                        ),
                        "exclusive_correction_fields": True,
                        "composite_map_safety": safety["passed"] is True,
                        "candidate_correlation": all(
                            float(row.correlation)
                            >= m51_r2_config.minimum_c2e_correlation
                            for row in measured
                        ),
                        "candidate_uniqueness": all(
                            float(row.uniqueness_fraction)
                            >= m51_r2_config.minimum_c2e_uniqueness_fraction
                            for row in measured
                        ),
                    },
                    config=m51_r2_config,
                )
                passed = evaluation.decision in {
                    "resolved", "improved_unresolved"
                }
                result = (passed, {
                    "decision": evaluation.decision,
                    "rejection_reasons": list(evaluation.rejection_reasons),
                    "baseline_metrics": baseline,
                    "candidate_metrics": candidate,
                    "candidate_map_safety": safety,
                })
                subset_cache[subset_ids] = result
                return result

            removal_id, failure_localization = (
                locate_s13_dependency_failure_subset(
                    failed_group,
                    pixel_support_by_candidate=pixel_support_by_candidate,
                    evaluate_subset=evaluate_dependency_subset,
                    utility_by_candidate=utility_by_candidate,
                )
            )
            dependency_failure_lineage.append({
                "failed_group_segment_ids": list(failed_group),
                "removed_segment_id": removal_id,
                **dict(failure_localization),
            })
            for row in dependency_group_audits:
                if tuple(row.get("segment_ids", [])) == tuple(failed_group):
                    row["failure_localization"] = dict(failure_localization)
                    break
            segment_candidates = [
                replace(
                    row,
                    decision="rejected",
                    rejection_reasons=tuple(row.rejection_reasons)
                    + ("dependency_group_composite_gate_failed",),
                ) if row.segment.segment_id == removal_id else row
                for row in segment_candidates
            ]
            for solved in solved_rows:
                if solved.get("segment_id") == removal_id:
                    solved["state"] = "rejected"
                    solved["reason"] = ["dependency_group_composite_gate_failed"]
                    solved["dependency_failure_localization"] = dict(
                        failure_localization
                    )
        if split_lineage:
            component_patch_set = replace(
                component_patch_set,
                unresolved_regions=(
                    *component_patch_set.unresolved_regions,
                    *(
                        {
                            "segment_id": row["segment_id"],
                            "state": "split",
                            "split_pair_index": row["split_pair_index"],
                            "child_segment_ids": list(row["child_segment_ids"]),
                            "reason": row["reason"],
                        }
                        for row in split_lineage
                    ),
                ),
            )
        source_correction_registry = freeze_s13_source_correction_registry(
            component_patch_set
        )
        component_field_ids = dict(source_correction_registry.field_id_by_segment)
        hypothesis_rows = []
        def finite_score_rows(values: np.ndarray) -> list[float | None]:
            return [float(value) if np.isfinite(value) else None for value in values]

        for pair in pairs:
            for observation, evidence in pair.component_evidence:
                hypothesis_rows.append({
                    "pair_index": observation.pair_index,
                    "component_id": observation.component_id,
                    "bbox_xyxy": list(observation.global_bbox_xyxy),
                    "evidence_state": observation.evidence_state,
                    "forward_best_lag_px": observation.forward_best_lag_px,
                    "reverse_best_lag_px": observation.reverse_best_lag_px,
                    "correlation": observation.correlation,
                    "uniqueness_fraction": observation.uniqueness_fraction,
                    "orientation_difference_degrees": (
                        observation.orientation_difference_degrees
                    ),
                    "signed_gradient_polarity": observation.signed_gradient_polarity,
                    "signed_gradient_agreement": observation.signed_gradient_agreement,
                    "normal_xy": [observation.normal_x, observation.normal_y],
                    "fitted_line_offset": observation.fitted_line_offset,
                    "exclusion_reasons": list(observation.exclusion_reasons),
                    "search_boundary_hit": observation.search_boundary_hit,
                    "support_sha256": observation.mask_sha256,
                    "forward_scores": finite_score_rows(evidence.forward_scores),
                    "reverse_scores": finite_score_rows(evidence.reverse_scores),
                    "forward_support_counts": np.asarray(
                        evidence.forward_support_counts
                    ).tolist(),
                    "reverse_support_counts": np.asarray(
                        evidence.reverse_support_counts
                    ).tolist(),
                })
        component_chain_audit = {
            "schema": "gemini305-video-s13-component-chain-c2e-hard-audit/v1",
            "application_state": component_patch_set.application_state,
            "repair_complete": bool(component_patch_set.audit.get("repair_complete", False)),
            "observation_count": len(observations),
            "chain_count": len(chains),
            "raw_partition_chain_count": len(all_chains),
            "segment_count": len(solved_rows),
            "baseline_c2e_obligations": [
                {
                    "obligation_id": row.obligation_id,
                    "pair_index": row.pair_index,
                    "component_id": row.component_id,
                    "bbox_xyxy": list(row.global_bbox_xyxy),
                    "support_sha256": row.support_sha256,
                    "baseline_metrics": dict(row.baseline_metrics),
                    "severe": row.severe,
                    "evaluable": row.evaluable,
                    "scope": row.scope,
                }
                for row in obligations
            ],
            "chains": [{
                "chain_id": row.chain_id,
                "pair_indices": list(row.pair_indices),
                "source_indices": list(row.source_indices),
                "component_ids": [item.component_id for item in row.observations],
                "chain_sha256": row.chain_sha256,
            } for row in chains],
            "segments": solved_rows,
            "split_lineage": split_lineage,
            "accepted_segment_ids": list(component_patch_set.accepted_segment_ids),
            "rejected_segment_ids": list(component_patch_set.rejected_segment_ids),
            "deferred_segment_ids": list(component_patch_set.deferred_segment_ids),
            "unresolved_regions": [
                dict(row) for row in component_patch_set.unresolved_regions
            ],
            "field_id_table": component_field_ids,
            "dependency_groups": dependency_group_audits,
            "dependency_failure_lineage": dependency_failure_lineage,
            "forward_reverse_hypotheses": hypothesis_rows,
            "component_match_matrices": component_match_matrices,
            "exact_evidence_propagation": next((
                dict(pair.component_propagation_audit)
                for pair in pairs
                if pair.component_propagation_audit is not None
            ), {
                "policy": "visual_suspect_seed_bidirectional_lazy_safe_anchor_propagation",
                "seed_pair_indices": [], "probed_pair_indices": [],
                "reverse_evidence_loader_call_count": 0,
                "exact_pair_count": 0, "exact_observation_count": 0,
                "propagation_link_count": 0, "propagation_links": [],
                "match_matrices": [],
            }),
            "fatal_failures": [],
            "passed": True,
            "roi_candidate_pixels": roi_candidate_pixels,
            "roi_preview_count": roi_preview_count,
        }
        component_chain_seconds = time.perf_counter() - component_started
    else:
        geometry_seconds = time.perf_counter() - tick
    frozen_map_cache: dict[
        tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    def frozen_map_provider(source_index: int, x0: int, x1: int):
        cache_key = (source_index, x0, x1)
        cached = frozen_map_cache.get(cache_key)
        if cached is not None:
            return cached
        candidate = None
        if source_index > 0 and pairs[source_index - 1].alignment is not None:
            candidate = pairs[source_index - 1].alignment.selected
        maps = _map_crop(
            schedule, calibration, source_index, x0, x1,
            vertical.global_offsets_px[source_index], candidate,
            tuple(
                () if component_patch_set is None
                else source_correction_registry.corrections_by_source.get(source_index, ())
            ),
            component_field_ids,
            vertical_parent=vertical,
        )
        # Freeze the exact provider result. Crop-origin ULP differences are
        # canonicalized once below by taking formal-owner values from the
        # final provenance and replay slices from that oracle; the render hot
        # path must not promote every full map to float64 merely for hashing.
        frozen = maps
        frozen_map_cache[cache_key] = frozen
        return frozen

    tick = time.perf_counter()
    formal_provider = (
        frozen_map_provider if isinstance(m51_r2_config, S13M51R4Config) else None
    )
    if formal_provider is not None:
        if (
            source_correction_registry is None
            or component_patch_set is None
            or component_chain_audit is None
        ):
            raise ValueError("S1.3 R4 pre-render estimation authority is incomplete")
        estimation_result, formal_provider = (
            _estimate_s13_m5_pre_render(
                schedule=schedule, calibration=calibration, vertical=vertical,
                pairs=tuple(pairs), raw_cache=raw_cache,
                registry=source_correction_registry, patch_set=component_patch_set,
                component_audit=component_chain_audit,
                final_map_provider=formal_provider,
                performance_counters={
                "component_linear_solve_count": component_linear_solve_count,
                "component_gain_candidate_count": component_gain_candidate_count,
                "component_segment_candidate_count": component_segment_candidate_count,
                "component_split_count": component_split_count,
                "roi_candidate_pixels": roi_candidate_pixels,
                "roi_preview_count": roi_preview_count,
                },
            )
        )
        source_map_oracles = list(estimation_result.final_source_map_oracles)
        geometry, final, replay_pairs = _finalize_s13_m5_render(
            estimate=estimation_result, schedule=schedule, calibration=calibration,
            vertical=vertical, selected_hypothesis_ids=selected_hypothesis_ids,
            placement_methods=placement_methods, map_provider=formal_provider,
            expected_support_mask=estimation_result.expected_support_mask,
            final_image_composer=final_image_composer,
        )
        expected_support_audit = dict(estimation_result.expected_support_audit)
        p2_full_resolution_render_count += 2
    else:
        expected_support_audit = {
            "build_seconds": 0.0,
            "build_count": 1,
            "provider_call_count": len(schedule.assignments),
            "requested_pixel_count": (
                len(schedule.assignments) * schedule.canvas_height * schedule.canvas_width
            ),
            "formal_geometry_rebuild_count": 0,
            "formal_final_rebuild_count": 0,
            "mask_true_pixel_count": 0,
        }
        source_map_oracles = []
        (
            ordinary_map_provider,
            ordinary_sampled_source,
            ordinary_expected_support,
        ) = _ordinary_s13_sampled_source_provider(
            schedule, calibration, cached_image_loader, vertical, pairs
        )
        expected_support_audit["mask_true_pixel_count"] = int(
            np.count_nonzero(ordinary_expected_support)
        )
        geometry = render_s13_p2_from_raw(
            schedule, calibration, cached_image_loader, vertical, pairs,
            final_seams=False, selected_hypothesis_ids=selected_hypothesis_ids,
            placement_methods=placement_methods,
            base_map_provider=ordinary_map_provider,
            expected_support_mask=ordinary_expected_support,
            sampled_source_provider=ordinary_sampled_source,
        )
        final = render_s13_p2_from_raw(
            schedule, calibration, cached_image_loader, vertical, pairs,
            final_seams=True, selected_hypothesis_ids=selected_hypothesis_ids,
            placement_methods=placement_methods,
            base_map_provider=ordinary_map_provider,
            expected_support_mask=ordinary_expected_support,
            sampled_source_provider=ordinary_sampled_source,
            image_composer=final_image_composer,
        )
        p2_full_resolution_render_count += 2
        replay_pairs = build_s13_p2_replay(schedule, calibration, vertical, pairs)
    if component_chain_audit is not None:
        exterior_mismatch = 0
        valid_mismatch = 0
        minimum_jacobian = math.inf
        minimum_jacobian_ratio = math.inf
        maximum_combined_displacement = 0.0
        source_out_of_bounds = 0
        changed_map_pixels = 0
        labelled_map_pixels = 0
        field_authority_valid = True
        correction_domain_valid = True
        correction_arrays_finite = True
        maximum_component_offset = 0.0
        resolved_field_overlap_count = 0
        known_field_ids = set(component_field_ids.values())
        base_oracle_by_source = {
            row.source_index: row
            for row in estimation_result.base_source_map_oracles
        } if estimation_result is not None else {}
        for oracle in source_map_oracles:
            x0, _y0, x1, _y1 = oracle.domain_xyxy
            base_oracle = base_oracle_by_source.get(oracle.source_index)
            if base_oracle is None or base_oracle.domain_xyxy != oracle.domain_xyxy:
                raise ValueError("S1.3 R4 base/final source-map authority is incomplete")
            base_u, base_v = base_oracle.u, base_oracle.v
            base_valid = base_oracle.valid != 0
            exterior = oracle.field_id < 0
            changed = (
                (oracle.u.view(np.uint32) != base_u.view(np.uint32))
                | (oracle.v.view(np.uint32) != base_v.view(np.uint32))
            )
            changed_map_pixels += int(np.count_nonzero((oracle.valid != 0) & changed))
            labelled_map_pixels += int(np.count_nonzero(oracle.field_id >= 0))
            exterior_mismatch += int(np.count_nonzero(
                exterior & ((oracle.u != base_u) | (oracle.v != base_v))
            ))
            valid_mismatch += int(np.count_nonzero((oracle.valid != 0) != base_valid))
            valid = oracle.valid != 0
            source_out_of_bounds += int(np.count_nonzero(
                valid & (
                    (oracle.u < 0.0) | (oracle.u > float(calibration.width - 1))
                    | (oracle.v < 0.0) | (oracle.v > float(calibration.height - 1))
                )
            ))
            interior_valid = valid.copy()
            interior_valid[1:, :] &= valid[:-1, :]
            interior_valid[:-1, :] &= valid[1:, :]
            interior_valid[:, 1:] &= valid[:, :-1]
            interior_valid[:, :-1] &= valid[:, 1:]
            correction_domain = interior_valid & (oracle.field_id >= 0)
            field_authority_valid = field_authority_valid and set(
                np.unique(oracle.field_id).tolist()
            ).issubset({-1, *known_field_ids})
            if np.any(correction_domain):
                displacement = np.hypot(
                    oracle.u.astype(np.float64) - base_u.astype(np.float64),
                    oracle.v.astype(np.float64) - base_v.astype(np.float64),
                )
                maximum_combined_displacement = max(
                    maximum_combined_displacement,
                    float(np.max(displacement[correction_domain])),
                )
                du_dy, du_dx = np.gradient(oracle.u.astype(np.float64))
                dv_dy, dv_dx = np.gradient(oracle.v.astype(np.float64))
                determinant = du_dx * dv_dy - du_dy * dv_dx
                base_du_dy, base_du_dx = np.gradient(base_u.astype(np.float64))
                base_dv_dy, base_dv_dx = np.gradient(base_v.astype(np.float64))
                base_determinant = (
                    base_du_dx * base_dv_dy - base_du_dy * base_dv_dx
                )
                evaluable = correction_domain & np.isfinite(determinant)
                if np.any(evaluable):
                    minimum_jacobian = min(
                        minimum_jacobian, float(np.min(determinant[evaluable]))
                    )
                ratio_evaluable = evaluable & (base_determinant > 0.0)
                if np.any(ratio_evaluable):
                    minimum_jacobian_ratio = min(
                        minimum_jacobian_ratio,
                        float(np.min(
                            determinant[ratio_evaluable]
                            / base_determinant[ratio_evaluable]
                        )),
                    )
        if source_correction_registry is not None:
            for corrections in source_correction_registry.corrections_by_source.values():
                occupancy = np.zeros(
                    (schedule.canvas_height, schedule.canvas_width), dtype=np.uint16
                )
                for correction in corrections:
                    active = correction.weight > 0.0
                    correction_arrays_finite = correction_arrays_finite and bool(
                        np.isfinite(correction.weight).all()
                        and np.isfinite(correction.delta_u).all()
                        and np.isfinite(correction.delta_v).all()
                    )
                    correction_domain_valid = correction_domain_valid and bool(
                        np.count_nonzero(correction.delta_u[~active]) == 0
                        and np.count_nonzero(correction.delta_v[~active]) == 0
                    )
                    maximum_component_offset = max(
                        maximum_component_offset,
                        float(np.max(np.hypot(
                            correction.delta_u.astype(np.float64),
                            correction.delta_v.astype(np.float64),
                        ))) if np.any(active) else 0.0,
                    )
                    occupancy[
                        correction.y0:correction.y1, correction.x0:correction.x1
                    ] += active.astype(np.uint16)
                resolved_field_overlap_count += int(np.count_nonzero(occupancy > 1))
        obligation_coverage = audit_s13_obligation_coverage(
            component_chain_audit.get("baseline_c2e_obligations", []),
            component_chain_audit.get("segments", []),
            component_chain_audit.get("accepted_segment_ids", []),
        )
        component_failures = []
        if exterior_mismatch:
            component_failures.append("omega_out_exterior_map_changed")
        if valid_mismatch:
            component_failures.append("formal_owner_valid_support_changed")
        if source_out_of_bounds:
            component_failures.append("source_map_out_of_bounds")
        if minimum_jacobian < m51_r2_config.minimum_jacobian:
            component_failures.append("minimum_jacobian_failed")
        if maximum_combined_displacement > (
            m51_r2_config.maximum_combined_map_displacement_px + 1e-6
        ):
            component_failures.append("combined_map_displacement_exceeded")
        if not correction_arrays_finite:
            component_failures.append("component_correction_nonfinite")
        if not correction_domain_valid:
            component_failures.append("component_correction_outside_omega_map")
        if maximum_component_offset > m51_r2_config.maximum_source_normal_offset_px + 1e-6:
            component_failures.append("maximum_component_offset_exceeded")
        if resolved_field_overlap_count:
            component_failures.append("resolved_component_field_overlap")
        if obligation_coverage["repair_complete"] is not component_chain_audit.get(
            "repair_complete"
        ):
            component_failures.append("repair_complete_authority_mismatch")
        formal_logical_source_render_count = (
            geometry.remap_invocations + final.remap_invocations
        )
        formal_remap_invocations = (
            geometry.actual_remap_invocations + final.actual_remap_invocations
        )
        extra_full_resolution_render_count = max(
            0, p2_full_resolution_render_count - 2
        )
        if p2_full_resolution_render_count != 2:
            component_failures.append("p2_full_resolution_render_count_invalid")
        if formal_logical_source_render_count != 2 * len(schedule.assignments):
            component_failures.append("formal_logical_source_render_count_invalid")
        if formal_remap_invocations != len(schedule.assignments):
            component_failures.append("formal_raw_rgb_remap_invocation_count_invalid")
        provenance_fields = final.pixel_provenance[
            "component_correction_field_id"
        ]
        field_authority_valid = field_authority_valid and set(
            np.unique(provenance_fields).tolist()
        ).issubset({-1, *known_field_ids})
        if not field_authority_valid:
            component_failures.append("component_field_authority_invalid")
        dependency_groups = component_chain_audit.get("dependency_groups", [])
        grouped_segments = {
            str(segment_id) for group in dependency_groups
            for segment_id in group.get("segment_ids", [])
            if group.get("passed") is True
        }
        if grouped_segments != set(component_patch_set.accepted_segment_ids):
            component_failures.append("dependency_group_coverage_invalid")
        component_chain_audit.update({
            "omega_out_exterior_uv_mismatch_count": exterior_mismatch,
            "valid_support_mismatch_count": valid_mismatch,
            "source_out_of_bounds_pixel_count": source_out_of_bounds,
            "minimum_final_inverse_map_jacobian": (
                None if not math.isfinite(minimum_jacobian) else minimum_jacobian
            ),
            "minimum_final_to_base_jacobian_ratio": (
                None if not math.isfinite(minimum_jacobian_ratio)
                else minimum_jacobian_ratio
            ),
            "maximum_combined_map_displacement_px": maximum_combined_displacement,
            "changed_map_pixel_count": changed_map_pixels,
            "labelled_map_pixel_count": labelled_map_pixels,
            "field_authority_valid": field_authority_valid,
            "correction_arrays_finite": correction_arrays_finite,
            "correction_domain_valid": correction_domain_valid,
            "maximum_component_offset_px": maximum_component_offset,
            "resolved_field_overlap_pixel_count": resolved_field_overlap_count,
            "obligation_coverage": dict(obligation_coverage),
            "p2_full_resolution_render_count": p2_full_resolution_render_count,
            "extra_full_resolution_render_count": extra_full_resolution_render_count,
            "formal_logical_source_render_count": formal_logical_source_render_count,
            "formal_raw_rgb_remap_invocations": formal_remap_invocations,
            "fatal_failures": component_failures,
            "passed": not component_failures,
        })
        normalized_component_audit = _canonicalize_v5_transaction_value(
            component_chain_audit,
            nonfinite_as_none=True,
        )
        if not isinstance(normalized_component_audit, dict):
            raise TypeError("S1.3 component-chain audit must be a mapping")
        component_chain_audit = normalized_component_audit
    seam_seconds = time.perf_counter() - tick
    geometry_features = prepare_seam_structure(geometry.image)
    final_features = prepare_seam_structure(final.image)
    if geometry_features is None or final_features is None:
        raise ValueError("S1.3 P2 full-canvas feature construction failed")
    # Geometry is compared on one owner topology.  Seam ownership is then
    # independently compared against the fixed-boundary render on both the
    # base and candidate paths, so neither half can authorize the other.
    before_metrics: list[Mapping[str, object]] = []
    after_metrics: list[Mapping[str, object]] = []
    audited_pair_indices: list[int] = []
    neutral_unevaluable_rollback_indices: list[int] = []
    applied_unevaluable_indices: list[int] = []
    horizontal_failure_indices: list[int] = []
    for pair_index, pair in enumerate(pairs):
        transaction = pair.transaction
        before = transaction.get("before_metrics")
        after = transaction.get("after_metrics")
        if (
            isinstance(before, Mapping)
            and isinstance(after, Mapping)
            and before.get("evaluable") is True
            and after.get("evaluable") is True
        ):
            before_metrics.append(before)
            after_metrics.append(after)
            audited_pair_indices.append(pair_index)
        elif transaction.get("decision") == "applied":
            applied_unevaluable_indices.append(pair_index)
        else:
            neutral_unevaluable_rollback_indices.append(pair_index)
        before_horizontal = transaction.get("before_horizontal_continuity")
        after_horizontal = transaction.get("after_horizontal_continuity")
        if isinstance(before_horizontal, Mapping) and isinstance(after_horizontal, Mapping):
            horizontal_ok, _reason = long_horizontal_structure_nondegrading(
                before_horizontal, after_horizontal
            )
            if not horizontal_ok:
                horizontal_failure_indices.append(pair_index)

    geometry_sequence_selected, geometry_selection_audit = sequence_structure_decision(
        before_metrics, after_metrics
    ) if before_metrics else (False, {"eligible": False, "reason": "no_evaluable_pair_transactions"})

    seam_output_pair_audits: list[dict[str, object]] = []
    seam_output_before: list[Mapping[str, object]] = []
    seam_output_after: list[Mapping[str, object]] = []
    seam_output_failure_indices: list[int] = []
    for pair_index, pair in enumerate(pairs):
        base_seam = np.full(
            schedule.canvas_height,
            schedule.boundaries[pair_index + 1],
            dtype=np.int32,
        )
        audit = _audit_rendered_seam_change(
            geometry.image,
            final.image,
            base_seam,
            np.asarray(pair.seam_x_by_row, dtype=np.int32),
            before_features=geometry_features,
            after_features=final_features,
        )
        seam_output_pair_audits.append({"pair_index": pair_index, **audit})
        symmetric_before = audit["symmetric_before_metrics"]
        symmetric_after = audit["symmetric_after_metrics"]
        if isinstance(symmetric_before, Mapping) and isinstance(symmetric_after, Mapping):
            seam_output_before.append(symmetric_before)
            seam_output_after.append(symmetric_after)
        if audit["eligible"] is not True:
            seam_output_failure_indices.append(pair_index)
    seam_output_sequence_selected, seam_output_sequence_audit = (
        sequence_structure_decision(
            seam_output_before,
            seam_output_after,
            maximum_component_failure_fraction=0.0,
            maximum_pair_score_failure_fraction=0.0,
            minimum_mean_improvement_fraction=0.0,
        )
        if seam_output_before
        else (False, {"eligible": False, "reason": "no_evaluable_seam_output_pairs"})
    )

    legacy_selected = bool(
        geometry_sequence_selected
        and not applied_unevaluable_indices
        and not horizontal_failure_indices
        and seam_output_sequence_selected
        and not seam_output_failure_indices
    )
    selection_reason = None
    if not geometry_sequence_selected:
        selection_reason = geometry_selection_audit.get("reason")
    elif applied_unevaluable_indices:
        selection_reason = "applied_geometry_transaction_unevaluable"
    elif horizontal_failure_indices:
        selection_reason = "geometry_horizontal_continuity_failure"
    elif not seam_output_sequence_selected:
        selection_reason = seam_output_sequence_audit.get("reason")
    elif seam_output_failure_indices:
        selection_reason = "seam_output_pair_non_degradation_not_proven"

    selection_audit = {
        **dict(geometry_selection_audit),
        "eligible": legacy_selected,
        "reason": selection_reason,
        "comparison_coordinate_policy": (
            "same_owner_geometry_plus_symmetric_base_candidate_seam"
        ),
        "geometry_sequence_eligible": bool(geometry_sequence_selected),
        "geometry_sequence_audit": dict(geometry_selection_audit),
        "audited_pair_indices": audited_pair_indices,
        "neutral_unevaluable_rollback_indices": neutral_unevaluable_rollback_indices,
        "applied_unevaluable_indices": applied_unevaluable_indices,
        "horizontal_continuity_failure_indices": horizontal_failure_indices,
        "seam_output_sequence_eligible": bool(seam_output_sequence_selected),
        "seam_output_sequence_audit": dict(seam_output_sequence_audit),
        "seam_output_failure_indices": seam_output_failure_indices,
        "seam_output_pair_audits": seam_output_pair_audits,
    }
    before_mean_value = selection_audit.get("before_mean_score")
    after_mean_value = selection_audit.get("after_mean_score")
    before_mean = float(before_mean_value) if isinstance(before_mean_value, (int, float)) else None
    after_mean = float(after_mean_value) if isinstance(after_mean_value, (int, float)) else None
    from .video_s13_hard_audit import audit_s13_p2_stage

    seams = _seams_array(schedule, pairs, final=True)
    hard_audit = audit_s13_p2_stage(
        valid_mask=final.valid_mask,
        pixel_provenance=final.pixel_provenance,
        seams_x_by_row=seams,
        assignment_frame_ids=tuple(item.frame_id for item in schedule.assignments),
        source_sizes=tuple(
            (int(calibration.width), int(calibration.height)) for _item in schedule.assignments
        ),
        pair_transaction_count=len(pairs),
        expected_support_mask=final.expected_support_mask,
        require_component_correction_fields=isinstance(m51_r2_config, S13M51R4Config),
    )
    if component_chain_audit is not None:
        topology = hard_audit.get("owner_and_provenance", {})
        seam_family = hard_audit.get("seam_family", {})
        topology_authority = {
            "owner_valid_topology_unchanged": bool(
                topology.get("valid_owner_consistent") is True
                and topology.get("provenance_valid_consistent") is True
                and topology.get("owner_source_index_valid") is True
                and topology.get("owner_frame_consistent") is True
            ),
            "base_geometry_provenance_valid": bool(
                topology.get("transaction_ids_valid") is True
                and topology.get("transaction_ids_owner_consistent") is True
            ),
            "secondary_provenance_unchanged": topology.get("secondary_owner_only") is True,
            "seam_topology_valid": seam_family.get("passed") is True,
        }
        component_failures = list(component_chain_audit.get("fatal_failures", []))
        component_failures.extend(
            name for name, passed in topology_authority.items() if not passed
        )
        component_chain_audit = {
            **dict(component_chain_audit),
            **topology_authority,
            "fatal_failures": component_failures,
            "passed": not component_failures,
        }
    if component_chain_audit is not None and component_chain_audit.get("passed") is not True:
        hard_audit = {
            **dict(hard_audit),
            "passed": False,
            "fatal_failures": [
                *list(hard_audit.get("fatal_failures", [])),
                "component_chain_c2e_hard_audit_failed",
            ],
        }
    hard_audit = {
        **dict(hard_audit),
        **({"component_chain_c2e": dict(component_chain_audit)} if component_chain_audit is not None else {}),
        "parent_verification_deferred_to_stage_sealer": True,
        "pair_fallback_count": sum(
            bool(pair.transaction.get("fallback_used"))
            or pair.transaction.get("decision") == "rolled_back"
            for pair in pairs
        ),
        "horizontal_catastrophe_summary": {
            "rejected_candidate_count": sum(
                str(failure).startswith("horizontal_structure_")
                for pair in pairs
                for evaluation in pair.transaction.get("candidate_evaluations", [])
                if isinstance(evaluation, Mapping)
                for failure in evaluation.get("hard_gate_failures", [])
            ),
        },
    }
    diagnostic_quality = {
        "schema": "gemini305-video-s13-m5-diagnostic-quality/v2",
        "diagnostic_only": True,
        "runtime_authority": False,
        "legacy_minimum_improvement_fraction": 0.005,
        "legacy_policy_result": legacy_selected,
        "before_mean_score": before_mean,
        "after_mean_score": after_mean,
        "relative_change": (
            None if before_mean is None or after_mean is None or abs(before_mean) < 1e-12
            else (after_mean - before_mean) / abs(before_mean)
        ),
        "legacy_selection_audit": selection_audit,
    }
    correspondence_rows = [
        pair.transaction.get("correspondence_filter", {}) for pair in pairs
    ]
    evaluated_rows = [
        evaluation
        for pair in pairs
        for evaluation in pair.transaction.get("candidate_evaluations", [])
        if isinstance(evaluation, Mapping)
        and evaluation.get("evaluation_status") != "skipped_after_higher_rank_safe_candidate"
    ]
    if component_chain_audit is not None:
        normalized_component_audit = _canonicalize_v5_transaction_value(
            component_chain_audit
        )
        if not isinstance(normalized_component_audit, dict):
            raise TypeError("S1.3 component-chain audit must be a mapping")
        component_chain_audit = normalized_component_audit
    return S13M5Result(
        pairs=pairs,
        replay_pairs=replay_pairs,
        geometry_result=geometry,
        final_result=final,
        seam_overlay=_overlay_seams(final.image, pairs),
        hard_audit_passed=hard_audit.get("passed") is True,
        hard_audit=hard_audit,
        diagnostic_quality=diagnostic_quality,
        selected_as_best=legacy_selected,
        before_mean_score=before_mean,
        after_mean_score=after_mean,
        selection_audit=selection_audit,
        performance={
            "geometry": geometry_seconds,
            "seam_and_p2_render": seam_seconds,
            "total_m5": time.perf_counter() - started,
            "m5_expected_support": dict(expected_support_audit),
            "gftt_call_count": sum(int(row.get("gftt_call_count", 0)) for row in correspondence_rows if isinstance(row, Mapping)),
            "forward_pyr_lk_call_count": sum(int(row.get("forward_pyr_lk_call_count", 0)) for row in correspondence_rows if isinstance(row, Mapping)),
            "backward_pyr_lk_call_count": 0,
            "full_canvas_feature_build_count": int(
                geometry_features is not None
            ) + int(final_features is not None),
            "pair_feature_build_count": sum(2 for _row in evaluated_rows),
            "risk_pair_count": sum(bool(pair.transaction.get("selection_continued_for_structure")) for pair in pairs),
            "extra_seam_candidate_evaluation_count": sum(max(0, sum(1 for row in pair.transaction.get("candidate_evaluations", []) if isinstance(row, Mapping) and row.get("evaluation_status") != "skipped_after_higher_rank_safe_candidate") - 1) for pair in pairs),
            "extra_geometry_roi_sample_count": sum(max(0, int(pair.transaction.get("selected_geometry_rank", 0))) for pair in pairs),
            "micro_rescue_attempt_count": 0,
            "p2_full_resolution_render_count": p2_full_resolution_render_count,
            "extra_full_resolution_render_count": 0,
            "formal_logical_source_render_count": (
                geometry.remap_invocations + final.remap_invocations
            ),
            "formal_raw_rgb_remap_invocations": (
                geometry.actual_remap_invocations + final.actual_remap_invocations
            ),
            "sampled_source_cache_hit_count": (
                geometry.remap_invocations + final.remap_invocations
                - geometry.actual_remap_invocations - final.actual_remap_invocations
            ),
            "depth_call_count": 0,
            "dis_call_count": 0,
            "open3d_call_count": 0,
            "orbslam3_call_count_in_m5": 0,
            "m4_geometry_reestimation_count": 0,
            "gain_enumeration_count": component_gain_candidate_count,
            "component_chain_detection_count": int(isinstance(m51_r2_config, S13M51R4Config)),
            "component_chain_candidate_count": int(
                0 if component_chain_audit is None else component_chain_audit["chain_count"]
            ),
            "component_chain_accepted_count": int(
                0 if component_patch_set is None
                else len(component_patch_set.accepted_segment_ids)
            ),
            "component_segment_candidate_count": int(
                component_segment_candidate_count
            ),
            "component_segment_resolved_count": sum(
                row.get("state") == "resolved"
                for row in (() if component_chain_audit is None else component_chain_audit["segments"])
            ),
            "component_segment_improved_unresolved_count": sum(
                row.get("state") == "improved_unresolved"
                for row in (() if component_chain_audit is None else component_chain_audit["segments"])
            ),
            "component_segment_rejected_count": int(
                0 if component_chain_audit is None else sum(
                    row.get("state") == "rejected"
                    for row in component_chain_audit["segments"]
                )
            ),
            "component_segment_split_count": component_split_count,
            "component_dependency_group_count": int(
                0 if component_chain_audit is None
                else len(component_chain_audit.get("dependency_groups", []))
            ),
            "component_observation_count": int(
                0 if component_chain_audit is None else component_chain_audit["observation_count"]
            ),
            "forward_edge_hypothesis_count": int(
                0 if component_chain_audit is None else sum(
                    len(row.get("forward_scores", ()))
                    for row in component_chain_audit.get(
                        "forward_reverse_hypotheses", ()
                    )
                )
            ),
            "reverse_edge_hypothesis_count": int(
                0 if component_chain_audit is None else sum(
                    len(row.get("reverse_scores", ()))
                    for row in component_chain_audit.get(
                        "forward_reverse_hypotheses", ()
                    )
                )
            ),
            "chain_linear_solve_count": component_linear_solve_count,
            "component_gain_candidate_count": component_gain_candidate_count,
            "component_propagation_pair_count": int(
                0 if component_chain_audit is None else len(
                    component_chain_audit.get(
                        "exact_evidence_propagation", {}
                    ).get("probed_pair_indices", ())
                )
            ),
            "chain_roi_preview_count": roi_preview_count,
            "component_correction_pixel_count": int(
                0 if source_correction_registry is None else sum(
                    np.count_nonzero(correction.weight > 0.0)
                    for corrections in source_correction_registry.corrections_by_source.values()
                    for correction in corrections
                )
            ),
            "component_chain_seconds": component_chain_seconds,
        },
        source_map_oracles=tuple(source_map_oracles),
        base_source_map_oracles=(
            () if estimation_result is None
            else estimation_result.base_source_map_oracles
        ),
        component_chain_audit=component_chain_audit,
        component_patch_set=component_patch_set,
        source_correction_registry=source_correction_registry,
        estimation_result=estimation_result,
    )


__all__ = [
    "S13M51R4EvidenceContext", "S13M5EstimationResult", "S13M5Pair",
    "S13M5Result", "S13P2Result", "S13PairCorrespondences",
    "build_s13_p2_replay",
    "estimate_s13_m5_transactions", "render_s13_p2_from_raw", "run_s13_m5",
    "reset_s13_m5_resident_batch", "reset_s13_m5_resident_remap", "reset_s13_m5_runtime_unsealed",
    "set_s13_m5_resident_batch", "set_s13_m5_resident_remap", "set_s13_m5_runtime_unsealed",
]
