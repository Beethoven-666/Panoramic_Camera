"""Fail-closed candidate selection and one-time holdout state management."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .video_recovery import canonical_sha256


SELECTION_SCHEMA = "gemini305-video-algorithm-selection/v3"
EVALUATOR_SCHEMA_V2 = "gemini305-video-offline-visual-evaluation/v2"


class VideoAlgorithmSelectionError(RuntimeError):
    """Candidate evidence cannot safely establish a production selection."""


_V2_COMPONENTS_BY_CANDIDATE: dict[str, tuple[str, ...]] = {
    "C1_constrained_owner": ("c1_constrained_owner",),
    "C2_dis_rgb_mesh": ("c1_constrained_owner", "c2_dis_mesh"),
    "C3_raft_rgb_mesh": ("c1_constrained_owner", "c3_raft_mesh"),
    "C4_raft_rgbd_layered_mesh": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh",
    ),
    "C5_object_lock": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh", "c5_object_owner_lock",
    ),
    "C6_multiband": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh", "c5_object_owner_lock", "c6_safe_multiband",
    ),
    "C7_photometric_graph": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh", "c5_object_owner_lock", "c6_safe_multiband", "c7_global_photometric",
    ),
    "C8_multilabel_window": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh", "c5_object_owner_lock", "c6_safe_multiband", "c7_global_photometric", "c8_local_multilabel_owner",
    ),
    "C9_positive_jacobian_line_mesh": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh",
        "c9_line_preserving_layered_mesh",
    ),
    "C10_depth_conditioned_multi_perspective_layout": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh",
        "c10_depth_conditioned_layout",
    ),
    "C13_robust_photometric_bundle": (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh",
        "c5_object_owner_lock", "c6_safe_multiband", "c7_global_photometric",
        "c8_local_multilabel_owner", "c13_robust_photometric_bundle",
    ),
}


def _read_json(path: Path) -> dict[str, object] | None:
    """Read an optional measurement sidecar without trusting malformed data."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _measurement_evidence_reasons(report_path: Path) -> list[str]:
    """Require immutable annotation and read-only visual evidence for selection.

    The renderer grades remain authoritative for publishing a candidate
    experiment.  This independent check merely prevents an A-labelled report
    from becoming production when its fixed source annotations are stale,
    omitted, or fail a hard visual gate.
    """

    reasons: list[str] = []
    progress = _read_json(report_path.parent / "video_annotation_source_progress_audit.json")
    if progress is None:
        reasons.append("annotation_source_progress_audit_missing_or_invalid")
    elif (
        progress.get("schema") != "gemini305-video-annotation-source-progress-audit/v1"
        or progress.get("measurement_only") is not True
        or progress.get("verified") is not True
        or progress.get("selection_eligible") is not True
    ):
        reasons.append("annotation_source_progress_not_verified")

    evaluation = _read_json(report_path.parent / "visual_metrics.json")
    if evaluation is None:
        reasons.append("offline_visual_measurement_missing_or_invalid")
        return reasons
    if (
        evaluation.get("schema") != EVALUATOR_SCHEMA_V2
        or evaluation.get("measurement_only") is not True
        or evaluation.get("automatic_grade_promotion_allowed") is not False
        or evaluation.get("projection_available") is not True
    ):
        reasons.append("offline_visual_measurement_not_eligible")
        return reasons
    for group in ("object_integrity", "line_continuity", "safe_background"):
        entries = evaluation.get(group)
        if not isinstance(entries, dict) or not entries:
            reasons.append(f"{group}_measurement_missing")
            continue
        if any(
            not isinstance(value, dict)
            or value.get("status") != "evaluated"
            for value in entries.values()
        ):
            reasons.append(f"{group}_measurement_not_evaluable")
    return reasons


def _visual_hard_gate_reasons(report_path: Path) -> list[str]:
    """Keep actual visual failures separate from missing measurement evidence."""

    evaluation = _read_json(report_path.parent / "visual_metrics.json")
    if evaluation is None or evaluation.get("schema") != EVALUATOR_SCHEMA_V2:
        return []
    reasons: list[str] = []
    for group in ("object_integrity", "line_continuity", "safe_background"):
        entries = evaluation.get(group)
        if isinstance(entries, dict) and entries and any(
            isinstance(value, dict)
            and value.get("status") == "evaluated"
            and value.get("hard_gate_pass") is not True
            for value in entries.values()
        ):
            reasons.append(f"{group}_hard_gate_not_passed")
    return reasons


def _finite_number(value: object, *, default: float = math.inf) -> float:
    """Return a safe ranking value; missing measurements always lose a tie."""

    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return default


def _ranking_quality(evaluation: Mapping[str, Any]) -> float:
    """Derive a deterministic quality margin from the immutable v2 evidence.

    Only already-passing measured quantities contribute.  The score is a
    ranking tie-breaker, never a replacement for a hard gate.
    """

    margins: list[float] = []
    for entry in (evaluation.get("object_integrity") or {}).values():
        if isinstance(entry, Mapping) and entry.get("status") == "evaluated":
            # Object gates are discrete; a passed object contributes its full
            # score without pretending a non-existent continuous metric.
            margins.append(1.0)
    for entry in (evaluation.get("line_continuity") or {}).values():
        if isinstance(entry, Mapping) and entry.get("status") == "evaluated":
            step = _finite_number(entry.get("line_step_p95_px"))
            angle = _finite_number(entry.get("line_orientation_delta_p95_degrees"))
            margins.extend((max(0.0, 1.0 - step / 1.0), max(0.0, 1.0 - angle / 3.0)))
    for entry in (evaluation.get("safe_background") or {}).values():
        if isinstance(entry, Mapping) and entry.get("status") == "evaluated":
            delta = _finite_number(entry.get("delta_e00_p95"))
            brightness = _finite_number(entry.get("brightness_step_p95_percent"))
            margins.extend((max(0.0, 1.0 - delta / 3.0), max(0.0, 1.0 - brightness / 2.0)))
    return sum(margins) / len(margins) if margins else float("-inf")


def _performance_ranking_values(report: Mapping[str, Any]) -> tuple[float, float]:
    """Find recorded warm time and VRAM without inventing missing evidence."""

    performance = report.get("performance")
    if not isinstance(performance, Mapping):
        performance = {}
    benchmark = report.get("benchmark")
    if not isinstance(benchmark, Mapping):
        benchmark = {}
    warm = _finite_number(benchmark.get("warm_median_seconds"), default=_finite_number(performance.get("warm_median_seconds")))
    renderer = report.get("renderer")
    runtime = renderer.get("gpu_runtime") if isinstance(renderer, Mapping) and isinstance(renderer.get("gpu_runtime"), Mapping) else {}
    peak = _finite_number(
        benchmark.get("peak_vram_bytes"),
        default=_finite_number(performance.get("peak_vram_bytes"), default=_finite_number(runtime.get("peak_reserved_bytes"))),
    )
    return warm, peak


def _quality_gate_lock(lock_path: str | Path | None) -> dict[str, object] | None:
    if lock_path is None:
        return None
    path = Path(lock_path).expanduser().resolve()
    lock = _read_json(path)
    if lock is None or lock.get("schema") != "gemini305-video-quality-gate-lock/v1":
        raise VideoAlgorithmSelectionError(f"Invalid quality gate lock: {path}")
    claimed = lock.get("sha256")
    unhashed = {key: value for key, value in lock.items() if key != "sha256"}
    if not isinstance(claimed, str) or claimed != canonical_sha256(unhashed):
        raise VideoAlgorithmSelectionError(f"Quality gate lock hash mismatch: {path}")
    return {"path": str(path), "version": lock.get("quality_gate_version"), "sha256": claimed}


@dataclass(frozen=True)
class CandidateSelection:
    algorithm_id: str
    eligible: bool
    reasons: tuple[str, ...]
    report_path: Path
    implementation_validity: tuple[str, ...] = ()
    measurement_validity: tuple[str, ...] = ()
    hard_gate_validity: tuple[str, ...] = ()
    performance_validity: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "report_path": str(self.report_path),
            "implementation_validity": list(self.implementation_validity),
            "measurement_validity": list(self.measurement_validity),
            "hard_gate_validity": list(self.hard_gate_validity),
            "performance_validity": list(self.performance_validity),
        }


def assess_validation_candidate(report_path: str | Path) -> CandidateSelection:
    """Accept only validation-scoped, A/A/A evidence for a future freeze."""

    path = Path(report_path).expanduser().resolve()
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoAlgorithmSelectionError(f"Invalid candidate report: {path}") from exc
    algorithm = report.get("algorithm")
    if not isinstance(algorithm, dict) or not isinstance(algorithm.get("algorithm_id"), str):
        raise VideoAlgorithmSelectionError("Candidate report lacks immutable algorithm identity")
    implementation_reasons: list[str] = []
    if algorithm.get("role") != "candidate":
        implementation_reasons.append("report_role_is_not_candidate")
    if algorithm.get("fallback_used") is not False:
        implementation_reasons.append("fallback_used")
    if algorithm.get("execution_backend") != "video_visual_renderer_v2_cuda":
        implementation_reasons.append("candidate_not_executed_by_v2_cuda_renderer")
    required_components = _V2_COMPONENTS_BY_CANDIDATE.get(str(algorithm.get("algorithm_id")))
    if required_components is not None:
        declared = algorithm.get("required_components")
        if not isinstance(declared, list) or tuple(declared) != required_components:
            implementation_reasons.append("v2_required_component_lineage_missing_or_mismatched")
        component_execution = algorithm.get("component_execution")
        if not isinstance(component_execution, dict):
            implementation_reasons.append("v2_component_execution_audit_missing")
        else:
            for component in required_components:
                record = component_execution.get(component)
                if (
                    not isinstance(record, dict)
                    or record.get("required") is not True
                    or record.get("initialized") is not True
                    or record.get("applied_to_output") is not True
                    or not isinstance(record.get("applied_output_pixel_count"), int)
                    or int(record["applied_output_pixel_count"]) <= 0
                ):
                    implementation_reasons.append(f"v2_component_not_applied_to_output:{component}")
        if algorithm.get("candidate_run_state") == "invalid_component_execution":
            implementation_reasons.append("candidate_run_state_invalid_component_execution")
        if algorithm.get("component_execution_selection_eligible") is False:
            implementation_reasons.append("candidate_component_execution_not_selection_eligible")
    hard_gate_reasons: list[str] = []
    scope = report.get("evaluation_scope")
    if scope != "validation_only":
        hard_gate_reasons.append("not_validation_only")
    performance_reasons: list[str] = []
    grades = report.get("grades")
    if not isinstance(grades, dict):
        hard_gate_reasons.append("grades_missing")
    else:
        for grade in ("structural", "visual", "performance", "overall"):
            if grades.get(grade) != "A":
                (performance_reasons if grade == "performance" else hard_gate_reasons).append(f"{grade}_grade_not_A")
    renderer = report.get("renderer")
    metrics = renderer.get("quality_metrics") if isinstance(renderer, dict) else None
    if isinstance(metrics, dict) and metrics.get("candidate_mesh_evidence_output_warp_applied") is False:
        implementation_reasons.append("mesh_evidence_not_applied_to_output")
    measurement_reasons = _measurement_evidence_reasons(path)
    hard_gate_reasons.extend(_visual_hard_gate_reasons(path))
    reasons = implementation_reasons + measurement_reasons + hard_gate_reasons + performance_reasons
    return CandidateSelection(
        algorithm_id=algorithm["algorithm_id"],
        eligible=not reasons,
        reasons=tuple(reasons),
        report_path=path,
        implementation_validity=tuple(implementation_reasons),
        measurement_validity=tuple(measurement_reasons),
        hard_gate_validity=tuple(hard_gate_reasons),
        performance_validity=tuple(performance_reasons),
    )


def write_validation_selection(
    report_paths: Iterable[str | Path], *, output: str | Path, quality_gate_lock: str | Path | None = None
) -> dict[str, object]:
    """Write the v3 deterministic recovery selection without using holdout."""

    assessments = [assess_validation_candidate(path) for path in report_paths]
    if not assessments:
        raise VideoAlgorithmSelectionError("At least one validation report is required")
    lock = _quality_gate_lock(quality_gate_lock)
    ranked: list[tuple[tuple[float, float, float, int, str], CandidateSelection, dict[str, object]]] = []
    for item in assessments:
        report = _read_json(item.report_path) or {}
        evaluation = _read_json(item.report_path.parent / "visual_metrics.json") or {}
        quality = _ranking_quality(evaluation)
        warm, peak = _performance_ranking_values(report)
        required = report.get("algorithm", {}).get("required_components", []) if isinstance(report.get("algorithm"), Mapping) else []
        required_count = len(required) if isinstance(required, list) else math.inf
        score = {"quality_score": quality, "warm_median_seconds": warm, "peak_vram_bytes": peak, "required_component_count": required_count}
        ranked.append(((-quality, warm, peak, required_count, item.algorithm_id), item, score))
    ranked.sort(key=lambda value: value[0])
    rank_by_path = {item.report_path: (index + 1, score) for index, (_, item, score) in enumerate(ranked) if item.eligible}
    candidates = []
    for item in assessments:
        data = item.as_dict()
        rank = rank_by_path.get(item.report_path)
        data["ranking"] = {"rank": rank[0], **rank[1]} if rank is not None else None
        candidates.append(data)
    eligible = [item for _, item, _ in ranked if item.eligible]
    result = {
        "schema": SELECTION_SCHEMA,
        "selection_stage": "validation",
        "measurement_schema": EVALUATOR_SCHEMA_V2,
        "quality_gate_lock": lock,
        "candidates": candidates,
        "eligible_candidate_count": len(eligible),
        "selected_algorithm_id": eligible[0].algorithm_id if eligible else None,
        "selection_status": "ready_for_first_holdout" if eligible else "not_selectable",
        "holdout_not_run": True,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


__all__ = [
    "CandidateSelection",
    "VideoAlgorithmSelectionError",
    "assess_validation_candidate",
    "write_validation_selection",
]
