"""Fail-closed candidate selection and one-time holdout state management."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .video_candidate_manifest import (
    VIDEO_CANDIDATE_MANIFEST_SCHEMA,
    canonical_candidate_manifest_sha256,
)
from .video_recovery import canonical_sha256


SELECTION_SCHEMA = "gemini305-video-algorithm-selection/v3"
EVALUATOR_SCHEMA_V2 = "gemini305-video-offline-visual-evaluation/v2"
BENCHMARK_SCHEMA = "gemini305-video-benchmark-result/v2"
PERFORMANCE_BENCHMARK_KIND = "full_3m_summary_minimal"


class VideoAlgorithmSelectionError(RuntimeError):
    """Candidate evidence cannot safely establish a production selection."""


def _read_json(path: Path) -> dict[str, object] | None:
    """Read an optional measurement sidecar without trusting malformed data."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _component_names(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ) or len(set(value)) != len(value):
        return None
    return tuple(value)


def _locked_component_contract(
    algorithm: Mapping[str, Any],
) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None, list[str]]:
    """Load the candidate's immutable manifest and compare every identity.

    There is intentionally no candidate-name table here.  A report is only
    meaningful when its recorded config and manifest still resolve to the
    exact same locked component declarations.
    """

    reasons: list[str] = []
    config_sha = algorithm.get("config_sha256")
    manifest_sha = algorithm.get("candidate_manifest_sha256")
    manifest_name = algorithm.get("candidate_manifest_path")
    candidate_id = algorithm.get("algorithm_id")
    if not all(isinstance(value, str) and value for value in (
        config_sha, manifest_sha, manifest_name, candidate_id,
    )):
        return None, None, ["candidate_manifest_identity_missing"]
    manifest = _read_json(Path(manifest_name).expanduser())
    if manifest is None or manifest.get("schema") != VIDEO_CANDIDATE_MANIFEST_SCHEMA:
        return None, None, ["candidate_manifest_missing_or_invalid"]
    claimed_manifest_sha = manifest.get("manifest_sha256")
    unhashed = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        not isinstance(claimed_manifest_sha, str)
        or claimed_manifest_sha != canonical_candidate_manifest_sha256(unhashed)
        or claimed_manifest_sha != manifest_sha
    ):
        reasons.append("candidate_manifest_sha_mismatch")
    candidates = manifest.get("candidates")
    entry = candidates.get(candidate_id) if isinstance(candidates, Mapping) else None
    if not isinstance(entry, Mapping):
        return None, None, [*reasons, "candidate_manifest_entry_missing"]
    if entry.get("config_sha256") != config_sha:
        reasons.append("candidate_config_sha_mismatch")
    evidence = _component_names(entry.get("required_evidence_components"))
    output = _component_names(entry.get("required_output_components"))
    replacements = entry.get("replaces_output_components", [])
    if evidence is None or output is None or not isinstance(replacements, list):
        return None, None, [*reasons, "candidate_manifest_component_contract_invalid"]
    if set(evidence) & set(output) or set(replacements) & set(output):
        reasons.append("candidate_manifest_component_contract_invalid")
    if _component_names(algorithm.get("required_evidence_components")) != evidence:
        reasons.append("report_evidence_component_lineage_missing_or_mismatched")
    if _component_names(algorithm.get("required_output_components")) != output:
        reasons.append("report_output_component_lineage_missing_or_mismatched")
    # Compatibility field is still emitted by specs, but cannot reinterpret
    # evidence as a pixel-output obligation.
    if _component_names(algorithm.get("required_components")) != output:
        reasons.append("report_required_components_missing_or_mismatched")
    if algorithm.get("replaces_output_components") != replacements:
        reasons.append("report_replacement_component_lineage_missing_or_mismatched")
    return evidence, output, reasons


def _component_contract_reasons(
    algorithm: Mapping[str, Any],
) -> list[str]:
    evidence_components, output_components, reasons = _locked_component_contract(algorithm)
    if evidence_components is None or output_components is None:
        return reasons
    evidence = algorithm.get("component_evidence")
    if not isinstance(evidence, Mapping):
        reasons.append("candidate_evidence_component_audit_missing")
    else:
        for component in evidence_components:
            record = evidence.get(component)
            if (
                not isinstance(record, Mapping)
                or record.get("valid") is not True
                or not isinstance(record.get("sample_count"), int)
                or int(record["sample_count"]) <= 0
                or record.get("audit_passed") is not True
            ):
                reasons.append(f"candidate_evidence_component_not_valid:{component}")
    execution = algorithm.get("component_execution")
    if not isinstance(execution, Mapping):
        reasons.append("candidate_output_component_execution_audit_missing")
    else:
        for component in output_components:
            record = execution.get(component)
            if (
                not isinstance(record, Mapping)
                or record.get("required") is not True
                or record.get("initialized") is not True
                or record.get("applied_to_output") is not True
                or not isinstance(record.get("applied_output_pixel_count"), int)
                or int(record["applied_output_pixel_count"]) <= 0
            ):
                reasons.append(f"candidate_output_component_not_applied:{component}")
    if algorithm.get("candidate_run_state") == "invalid_component_execution":
        reasons.append("candidate_run_state_invalid_component_execution")
    if algorithm.get("component_execution_selection_eligible") is not True:
        reasons.append("candidate_component_execution_not_selection_eligible")
    return reasons


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


def _performance_benchmark(
    report_path: Path, algorithm: Mapping[str, Any]
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """Load only full 3 m, summary/minimal benchmark performance evidence.

    Per-run validation reports deliberately carry performance ``NE``.  They
    are useful for visual evidence but cannot become a timing shortcut.  This
    resolver therefore rejects embedded/report-local timing and accepts a
    benchmark summary only when it explicitly records the report as one of
    its measured runs.
    """

    report_resolved = str(report_path.resolve())
    for parent in (report_path.parent, *report_path.parents):
        candidate = _read_json(parent / "benchmark.json")
        if candidate is None:
            continue
        if candidate.get("schema") != BENCHMARK_SCHEMA:
            continue
        run_paths = {
            str(Path(str(row.get("video_report", ""))).expanduser().resolve())
            for row in candidate.get("runs", [])
            if isinstance(row, Mapping) and isinstance(row.get("video_report"), str)
        }
        if report_resolved not in run_paths:
            continue
        reasons: list[str] = []
        if candidate.get("benchmark_kind") != PERFORMANCE_BENCHMARK_KIND:
            reasons.append("performance_benchmark_is_not_full_3m_summary_minimal")
        observability = candidate.get("observability")
        if observability != {"report_level": "summary", "artifact_level": "minimal"}:
            reasons.append("performance_benchmark_is_not_summary_minimal")
        if candidate.get("evaluation_scope") != "validation_only":
            reasons.append("performance_benchmark_not_validation_only")
        benchmark_algorithm = candidate.get("algorithm")
        if not isinstance(benchmark_algorithm, Mapping) or (
            benchmark_algorithm.get("algorithm_id") != algorithm.get("algorithm_id")
            or benchmark_algorithm.get("config_sha256") != algorithm.get("config_sha256")
        ):
            reasons.append("performance_benchmark_algorithm_identity_mismatch")
        gate = candidate.get("performance_gate")
        if not isinstance(gate, Mapping) or gate.get("status") != "passed":
            reasons.append("performance_benchmark_gate_not_passed")
        return candidate, reasons
    return None, ["performance_benchmark_missing_or_invalid"]


def _performance_ranking_values(benchmark: Mapping[str, Any] | None) -> tuple[float, float]:
    """Rank only by complete benchmark evidence, never report-local timings."""

    if not isinstance(benchmark, Mapping):
        return math.inf, math.inf
    warm = _finite_number(benchmark.get("warm_median_seconds"))
    peak = _finite_number(benchmark.get("peak_vram_bytes"))
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
    if algorithm.get("working_tree_dirty") is not False:
        implementation_reasons.append("candidate_working_tree_dirty_or_unrecorded")
    source_commit = algorithm.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        implementation_reasons.append("candidate_runtime_source_commit_missing_or_invalid")
    implementation_reasons.extend(_component_contract_reasons(algorithm))
    hard_gate_reasons: list[str] = []
    scope = report.get("evaluation_scope")
    if scope != "validation_only":
        hard_gate_reasons.append("not_validation_only")
    performance_reasons: list[str] = []
    grades = report.get("grades")
    if not isinstance(grades, dict):
        hard_gate_reasons.append("grades_missing")
    else:
        for grade in ("structural", "visual", "overall"):
            if grades.get(grade) != "A":
                hard_gate_reasons.append(f"{grade}_grade_not_A")
    renderer = report.get("renderer")
    metrics = renderer.get("quality_metrics") if isinstance(renderer, dict) else None
    if isinstance(metrics, dict) and metrics.get("candidate_mesh_evidence_output_warp_applied") is False:
        implementation_reasons.append("mesh_evidence_not_applied_to_output")
    measurement_reasons = _measurement_evidence_reasons(path)
    hard_gate_reasons.extend(_visual_hard_gate_reasons(path))
    _, benchmark_reasons = _performance_benchmark(path, algorithm)
    performance_reasons.extend(benchmark_reasons)
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
        algorithm = report.get("algorithm") if isinstance(report.get("algorithm"), Mapping) else {}
        benchmark, _ = _performance_benchmark(item.report_path, algorithm)
        warm, peak = _performance_ranking_values(benchmark)
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
