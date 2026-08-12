"""Explicit offline review authority for S013 M8."""

from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .video_s13_bundle import atomic_write_json, sha256_file
from .video_s13_evaluation import _json, review_stage_binding, verify_review_bundle


REVIEW_RECORD_SCHEMA = "gemini305-video-s13-review-record/v1"
CURRENT_REVIEWED_SCHEMA = "gemini305-video-s13-current-reviewed/v2"
REVIEW_METHODS = {"manual", "offline_dataset"}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def review_generation(
    acceptance: Path,
    bundle: Path,
    branch: str,
    *,
    stage: str,
    reviewer: str,
    review_method: str,
    note: str,
) -> dict[str, Any]:
    reviewer = reviewer.strip()
    review_method = review_method.strip()
    if not reviewer or review_method not in REVIEW_METHODS:
        raise ValueError(
            "M8 reviewer is required and review method must be manual or offline_dataset"
        )
    if stage not in {"P2", "P3", "P4"}:
        raise ValueError("M8 reviewed stage must be P2, P3, or real P4")
    verify_review_bundle(bundle)
    form = _json(bundle / "review_form.json")
    conclusion = form["stage_conclusions"].get(stage, {})
    if conclusion.get("decision") != "accepted":
        raise ValueError("M8 stage is not accepted by the frozen review form")
    binding = review_stage_binding(acceptance, branch, stage)
    reviewed_at_utc = _utc()
    record = {
        "schema": REVIEW_RECORD_SCHEMA,
        "branch": branch,
        "generation_id": binding["generation_id"],
        "stage": stage,
        "reviewer": reviewer,
        "review_method": review_method,
        "note": note,
        "reviewed_at_utc": reviewed_at_utc,
        "bundle_manifest_sha256": sha256_file(bundle / "bundle_manifest.json"),
        "completion": binding["completion"],
        "completion_sha256": binding["completion_sha256"],
        "result": binding["result"],
        "result_sha256": binding["result_sha256"],
        "hard_audit": binding["hard_audit"],
        "hard_audit_sha256": binding["hard_audit_sha256"],
        "automatic_quality_remains_blocked": True,
        "diagnostic_only": True,
        "production_eligible": False,
    }
    atomic_write_json(bundle / "review_record.json", record)
    pointer = {
        **record,
        "schema": CURRENT_REVIEWED_SCHEMA,
        "review_record": str((bundle / "review_record.json").resolve()),
        "review_record_sha256": sha256_file(bundle / "review_record.json"),
    }
    out = acceptance / "runs_v4" / branch / "formal/current_reviewed.json"
    atomic_write_json(out, pointer)
    return pointer
