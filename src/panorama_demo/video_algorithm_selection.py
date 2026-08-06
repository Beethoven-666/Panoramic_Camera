"""Fail-closed candidate selection and one-time holdout state management."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
        evaluation.get("schema") != "gemini305-video-offline-visual-evaluation/v1"
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
            or value.get("hard_gate_pass") is not True
            for value in entries.values()
        ):
            reasons.append(f"{group}_hard_gate_not_passed")
    return reasons


@dataclass(frozen=True)
class CandidateSelection:
    algorithm_id: str
    eligible: bool
    reasons: tuple[str, ...]
    report_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "algorithm_id": self.algorithm_id,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "report_path": str(self.report_path),
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
    reasons: list[str] = []
    if algorithm.get("role") != "candidate":
        reasons.append("report_role_is_not_candidate")
    if algorithm.get("fallback_used") is not False:
        reasons.append("fallback_used")
    if algorithm.get("execution_backend") != "video_visual_renderer_v2_cuda":
        reasons.append("candidate_not_executed_by_v2_cuda_renderer")
    required_components = _V2_COMPONENTS_BY_CANDIDATE.get(str(algorithm.get("algorithm_id")))
    if required_components is not None:
        executed = algorithm.get("executed_candidate_components")
        if not isinstance(executed, dict):
            reasons.append("v2_executed_component_audit_missing")
        else:
            for component in required_components:
                if executed.get(component) is not True:
                    reasons.append(f"v2_component_not_executed:{component}")
    scope = report.get("evaluation_scope")
    if scope != "validation_only":
        reasons.append("not_validation_only")
    grades = report.get("grades")
    if not isinstance(grades, dict):
        reasons.append("grades_missing")
    else:
        for grade in ("structural", "visual", "performance", "overall"):
            if grades.get(grade) != "A":
                reasons.append(f"{grade}_grade_not_A")
    renderer = report.get("renderer")
    metrics = renderer.get("quality_metrics") if isinstance(renderer, dict) else None
    if isinstance(metrics, dict) and metrics.get("candidate_mesh_evidence_output_warp_applied") is False:
        reasons.append("mesh_evidence_not_applied_to_output")
    reasons.extend(_measurement_evidence_reasons(path))
    return CandidateSelection(
        algorithm_id=algorithm["algorithm_id"],
        eligible=not reasons,
        reasons=tuple(reasons),
        report_path=path,
    )


def write_validation_selection(
    report_paths: Iterable[str | Path], *, output: str | Path
) -> dict[str, object]:
    """Write an immutable-style report without silently selecting a loser."""

    assessments = [assess_validation_candidate(path) for path in report_paths]
    if not assessments:
        raise VideoAlgorithmSelectionError("At least one validation report is required")
    eligible = [item for item in assessments if item.eligible]
    result = {
        "schema": "gemini305-video-algorithm-selection/v1",
        "selection_stage": "validation",
        "candidates": [item.as_dict() for item in assessments],
        "selected_algorithm_id": eligible[0].algorithm_id if len(eligible) == 1 else None,
        "selection_status": "ready_for_first_holdout" if len(eligible) == 1 else "not_selectable",
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
