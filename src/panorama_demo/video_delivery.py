"""Atomic independent publication for a video two-dimensional delivery."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIDEO_DELIVERY_SCHEMA = "gemini305-video-panorama-delivery/v2"
VIDEO_REPORT_SCHEMA = "gemini305-video-panorama-report/v2"

_PUBLISHED_STATES = frozenset({"published", "published_degraded"})
# ``NE`` is intentionally limited to non-performance-evaluable validation or
# audit deliveries.  It prevents a read-only evidence run from being reused
# as a production timing claim.
_GRADE_VALUES = frozenset({"A", "B", "C", "NE"})
_ALGORITHM_ROLES = frozenset({"baseline", "candidate", "production"})


def _require_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Video report {name} must be a non-empty string")
    return value


def _require_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Video report {name} must be an object")
    return value


def _without_legacy_presets(value: Any) -> Any:
    """Copy JSON-compatible report data while removing retired preset metadata."""
    if isinstance(value, dict):
        return {
            str(key): _without_legacy_presets(item)
            for key, item in value.items()
            if key != "preset"
        }
    if isinstance(value, list):
        return [_without_legacy_presets(item) for item in value]
    if isinstance(value, tuple):
        return [_without_legacy_presets(item) for item in value]
    return value


def _delivery_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Validate the v2 publication contract and derive its marker fields."""
    delivery_state = report.get("delivery_state")
    if delivery_state not in _PUBLISHED_STATES:
        raise ValueError("Video report delivery_state must be published or published_degraded")

    algorithm = _require_mapping(report.get("algorithm"), "algorithm")
    role = algorithm.get("role")
    if role not in _ALGORITHM_ROLES:
        raise ValueError("Video report algorithm.role is unsupported")
    algorithm_id = _require_nonempty_string(algorithm.get("algorithm_id"), "algorithm.algorithm_id")
    _require_nonempty_string(algorithm.get("implementation_id"), "algorithm.implementation_id")
    fallback_used = algorithm.get("fallback_used")
    if not isinstance(fallback_used, bool):
        raise ValueError("Video report algorithm.fallback_used must be a boolean")

    observability = _require_mapping(report.get("observability"), "observability")
    report_level = _require_nonempty_string(observability.get("report_level"), "observability.report_level")
    artifact_level = _require_nonempty_string(observability.get("artifact_level"), "observability.artifact_level")

    grades = _require_mapping(report.get("grades"), "grades")
    grade_names = {
        "structural": "structural_grade",
        "visual": "visual_grade",
        "performance": "performance_grade",
        "overall": "overall_grade",
    }
    delivery_grades: dict[str, str] = {}
    for source_name, delivery_name in grade_names.items():
        grade = grades.get(source_name)
        if grade not in _GRADE_VALUES:
            raise ValueError(f"Video report grades.{source_name} must be A, B, or C")
        delivery_grades[delivery_name] = grade

    manual_review = report.get("manual_review_required", False)
    if not isinstance(manual_review, bool):
        raise ValueError("Video report manual_review_required must be a boolean")
    if fallback_used and (delivery_grades["overall_grade"] != "C" or not manual_review):
        raise ValueError("A video baseline fallback must be grade C and require manual review")

    return {
        "delivery_state": delivery_state,
        "algorithm_id": algorithm_id,
        "algorithm_role": role,
        "fallback_used": fallback_used,
        **delivery_grades,
        "report_level": report_level,
        "artifact_level": artifact_level,
        "manual_review_required": manual_review,
    }


def invalidate_video_delivery(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "video_delivery.json", "video_report.json", "video_failure.json",
        "video_panorama.jpg", "video_panorama.png", "video_pixel_provenance.npz",
        "video_annotation_projection.json", "video_annotation_projection_masks.npz",
        "video_source_progress_evidence.json", "video_annotation_source_progress_audit.json",
        "visual_metrics.json", "video_timing.json",
        "central_strips", ".central_strips.pending",
        "central_strips_owner_only", ".central_strips_owner_only.pending",
        # A fresh 2-D delivery must not appear to be paired with a mesh made
        # from a previous source session or prior video rendering run.
        "video_3d_delivery.json", "video_3d_failure.json",
        "video_tsdf_mesh.glb", "video_tsdf_mesh_mobile.glb",
        "video_tsdf_mesh_viewer.html",
    ):
        path = output / name
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


def write_video_failure(output: Path, input_path: Path, exc: Exception) -> None:
    invalidate_video_delivery(output)
    pending = output / ".video_failure.pending.json"
    payload: dict[str, Any] = {
        "schema": "gemini305-video-panorama-failure/v1",
        "input": str(input_path),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "deliverable_published": False,
    }
    diagnostics = getattr(exc, "diagnostics", None)
    if isinstance(diagnostics, dict) and diagnostics:
        # Candidate-only gates may expose structured, non-pixel diagnostics
        # which are indispensable for a fail-closed recovery decision.
        payload["diagnostics"] = diagnostics
    attempts = getattr(exc, "attempt_audit", ())
    if isinstance(attempts, (list, tuple)):
        compact_attempts = [dict(row) for row in attempts if isinstance(row, dict)]
        if compact_attempts:
            payload["orbslam3_execution_attempts"] = compact_attempts
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(pending, output / "video_failure.json")


def write_invalid_candidate_experiment(output: Path, report: dict[str, Any]) -> dict[str, Any]:
    """Persist a fail-closed candidate report without publishing a delivery.

    An output component that never reached a final pixel is an invalid
    experiment, not a degraded panorama.  Retaining only its report keeps the
    failure auditable while making it impossible to mistake it for a usable
    2-D delivery.
    """

    algorithm = _require_mapping(report.get("algorithm"), "algorithm")
    grades = _require_mapping(report.get("grades"), "grades")
    if (
        report.get("delivery_state") != "experiment_invalid"
        or algorithm.get("role") != "candidate"
        or algorithm.get("candidate_run_state") != "invalid_component_execution"
        or grades.get("implementation") != "F"
        or grades.get("overall") != "F"
        or report.get("strict_quality_pass") is not False
    ):
        raise ValueError("Invalid candidate report does not satisfy the F experiment contract")
    output.mkdir(parents=True, exist_ok=True)
    report = _without_legacy_presets(report)
    report["schema"] = VIDEO_REPORT_SCHEMA
    report_path = output / ".video_report.pending.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(report_path, output / "video_report.json")
    return report


def publish_video_2d(
    output: Path,
    panorama: np.ndarray,
    owner: np.ndarray,
    report: dict[str, Any],
    *,
    pending_central_strips: Path | None = None,
    pending_central_strips_owner_only: Path | None = None,
) -> dict[str, Any]:
    if panorama.dtype != np.uint8 or panorama.ndim != 3 or panorama.shape[2] != 3:
        raise ValueError("Video panorama must be an 8-bit BGR image")
    if owner.shape != panorama.shape[:2]:
        raise ValueError("Video owner map shape does not match panorama")
    report = _without_legacy_presets(report)
    delivery_fields = _delivery_fields(report)
    if pending_central_strips is not None:
        export = report.get("central_strip_export")
        if not isinstance(export, dict) or not pending_central_strips.is_dir():
            raise RuntimeError("Video delivery lacks staged central-strip images")
        export["path"] = str(output / "central_strips")
    if pending_central_strips_owner_only is not None:
        owner_only_export = report.get("central_strip_owner_only_export")
        if not isinstance(owner_only_export, dict) or not pending_central_strips_owner_only.is_dir():
            raise RuntimeError("Video delivery lacks staged owner-only central-strip images")
        owner_only_export["path"] = str(output / "central_strips_owner_only")
    pending_jpg, pending_png = output / ".video_panorama.pending.jpg", output / ".video_panorama.pending.png"
    if not cv2.imwrite(str(pending_jpg), panorama, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("Could not encode video panorama JPEG")
    if not cv2.imwrite(str(pending_png), panorama):
        raise OSError("Could not encode video panorama PNG")
    pending_prov = output / ".video_provenance.pending.npz"
    np.savez_compressed(pending_prov, owner_frame_id=owner.astype(np.int32, copy=False))
    report["schema"] = VIDEO_REPORT_SCHEMA
    report["panorama"] = str(output / "video_panorama.jpg")
    report["provenance"] = str(output / "video_pixel_provenance.npz")
    report_path = output / ".video_report.pending.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    delivery = {
        "schema": VIDEO_DELIVERY_SCHEMA,
        **delivery_fields,
        "report": "video_report.json",
    }
    performance = report.get("performance")
    if isinstance(performance, dict):
        for key in (
            "primary_post_capture_seconds",
            "maximum_post_seconds",
            "within_post_capture_budget",
        ):
            if key in performance:
                delivery[key] = performance[key]
    if pending_central_strips is not None:
        delivery["central_strip_export"] = report["central_strip_export"]
    if pending_central_strips_owner_only is not None:
        delivery["central_strip_owner_only_export"] = report["central_strip_owner_only_export"]
    delivery_path = output / ".video_delivery.pending.json"
    delivery_path.write_text(json.dumps(delivery, indent=2), encoding="utf-8")
    os.replace(pending_jpg, output / "video_panorama.jpg")
    os.replace(pending_png, output / "video_panorama.png")
    os.replace(pending_prov, output / "video_pixel_provenance.npz")
    if pending_central_strips is not None:
        os.replace(pending_central_strips, output / "central_strips")
    if pending_central_strips_owner_only is not None:
        os.replace(pending_central_strips_owner_only, output / "central_strips_owner_only")
    os.replace(report_path, output / "video_report.json")
    os.replace(delivery_path, output / "video_delivery.json")
    return report
