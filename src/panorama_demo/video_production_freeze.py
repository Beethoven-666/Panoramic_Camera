"""Fail-closed evidence lifecycle for freezing a video production lock.

This module is intentionally separate from both the public renderer and the
development runner.  It can record *one* first-holdout result and can create a
production configuration/lock only from that immutable evidence.  It never
runs a renderer, chooses parameters, or turns a failed holdout into a pass.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import yaml

from .video_algorithm import (
    VIDEO_ALGORITHM_CONFIG_SCHEMA,
    VideoAlgorithmSpec,
    build_algorithm_spec,
    load_algorithm_config,
)
from .video_algorithm_lock import VIDEO_ALGORITHM_LOCK_SCHEMA, verify_algorithm_lock
from .video_algorithm_selection import _measurement_evidence_reasons


HOLDOUT_STATE_SCHEMA = "gemini305-video-first-holdout-state/v1"


class VideoProductionFreezeError(RuntimeError):
    """Production evidence is absent, ambiguous, consumed, or incompatible."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: str | Path, *, label: str) -> tuple[Path, dict[str, object]]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoProductionFreezeError(f"Invalid {label}: {resolved}") from exc
    if not isinstance(payload, dict):
        raise VideoProductionFreezeError(f"{label} must contain a JSON object: {resolved}")
    return resolved, payload


def _atomic_write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create evidence and reject replacement of prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise VideoProductionFreezeError(f"Refusing to overwrite immutable evidence: {path}")
    pending = path.with_name(f".{path.name}.pending")
    if pending.exists():
        raise VideoProductionFreezeError(f"Pending immutable evidence already exists: {pending}")
    try:
        pending.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Exclusive creation reserves the destination even if two callers race.
        with pending.open("rb") as source, path.open("xb") as destination:
            destination.write(source.read())
    except FileExistsError as exc:
        raise VideoProductionFreezeError(f"Refusing to overwrite immutable evidence: {path}") from exc
    finally:
        pending.unlink(missing_ok=True)


def _selection_algorithm(selection_path: Path, selection: Mapping[str, object]) -> str:
    if selection.get("schema") != "gemini305-video-algorithm-selection/v1":
        raise VideoProductionFreezeError("Validation selection schema is invalid")
    if selection.get("selection_stage") != "validation":
        raise VideoProductionFreezeError("Production freeze requires a validation selection")
    if selection.get("selection_status") != "ready_for_first_holdout":
        raise VideoProductionFreezeError("Validation selection is not ready for first holdout")
    if selection.get("holdout_not_run") is not True:
        raise VideoProductionFreezeError("Validation selection does not represent an unused holdout")
    algorithm_id = selection.get("selected_algorithm_id")
    candidates = selection.get("candidates")
    if not isinstance(algorithm_id, str) or not algorithm_id:
        raise VideoProductionFreezeError("Validation selection lacks exactly one selected algorithm")
    if not isinstance(candidates, list):
        raise VideoProductionFreezeError("Validation selection candidates are invalid")
    selected = [item for item in candidates if isinstance(item, dict) and item.get("algorithm_id") == algorithm_id]
    if len(selected) != 1 or selected[0].get("eligible") is not True:
        raise VideoProductionFreezeError("Selected validation candidate is not eligible")
    return algorithm_id


def _all_a_grades(report: Mapping[str, object]) -> bool:
    grades = report.get("grades")
    return isinstance(grades, dict) and all(
        grades.get(name) == "A" for name in ("structural", "visual", "performance", "overall")
    )


def _validate_holdout_report(report_path: Path, report: Mapping[str, object], *, algorithm_id: str) -> bool:
    algorithm = report.get("algorithm")
    if not isinstance(algorithm, dict):
        raise VideoProductionFreezeError("Holdout report lacks algorithm identity")
    if algorithm.get("role") != "candidate" or algorithm.get("algorithm_id") != algorithm_id:
        raise VideoProductionFreezeError("Holdout report does not match the selected candidate")
    if algorithm.get("fallback_used") is not False:
        raise VideoProductionFreezeError("Holdout report used a fallback")
    if algorithm.get("execution_backend") != "video_visual_renderer_v2_cuda":
        raise VideoProductionFreezeError(
            "Holdout report was not executed by the v2 CUDA renderer"
        )
    if report.get("evaluation_scope") != "holdout_only":
        raise VideoProductionFreezeError("First holdout evidence must be scoped exactly to holdout_only")
    reasons = _measurement_evidence_reasons(report_path)
    if reasons:
        raise VideoProductionFreezeError(
            "Holdout report has ineligible measurement evidence: " + ", ".join(reasons)
        )
    return _all_a_grades(report)


def record_first_holdout(
    *,
    validation_selection: str | Path,
    holdout_report: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Seal exactly one first holdout attempt, including a failed attempt.

    A failed first holdout is deliberately recorded with ``first_holdout_pass``
    false.  It consumes the first-holdout slot and therefore cannot later be
    silently replaced by a more favourable result.
    """

    selection_path, selection = _read_object(validation_selection, label="validation selection")
    algorithm_id = _selection_algorithm(selection_path, selection)
    report_path, report = _read_object(holdout_report, label="holdout report")
    passed = _validate_holdout_report(report_path, report, algorithm_id=algorithm_id)
    algorithm = dict(report["algorithm"])
    state: dict[str, object] = {
        "schema": HOLDOUT_STATE_SCHEMA,
        "first_holdout_consumed": True,
        "first_holdout_pass": passed,
        "selected_algorithm_id": algorithm_id,
        "selection_sha256": _sha256_file(selection_path),
        "holdout_report_sha256": _sha256_file(report_path),
        "algorithm": {
            key: algorithm.get(key)
            for key in ("algorithm_id", "config_sha256", "source_commit", "model_sha256")
        },
    }
    target = Path(output).expanduser().resolve()
    _atomic_write_json_new(target, state)
    return state


def _read_valid_holdout_state(
    path: str | Path,
    *,
    selection_path: Path,
    selected_algorithm_id: str,
) -> tuple[Path, dict[str, object]]:
    state_path, state = _read_object(path, label="first holdout state")
    if state.get("schema") != HOLDOUT_STATE_SCHEMA or state.get("first_holdout_consumed") is not True:
        raise VideoProductionFreezeError("First holdout state is not sealed")
    if state.get("first_holdout_pass") is not True:
        raise VideoProductionFreezeError("First holdout did not pass; production cannot be frozen")
    if state.get("selected_algorithm_id") != selected_algorithm_id:
        raise VideoProductionFreezeError("First holdout state selected a different algorithm")
    if state.get("selection_sha256") != _sha256_file(selection_path):
        raise VideoProductionFreezeError("First holdout state does not match validation selection bytes")
    if not isinstance(state.get("holdout_report_sha256"), str):
        raise VideoProductionFreezeError("First holdout state lacks report hash")
    return state_path, state


def _validate_dataset_lock(path: str | Path) -> tuple[Path, str]:
    lock_path, payload = _read_object(path, label="dataset lock")
    if payload.get("schema") != "gemini305-video-dataset-lock/v1":
        raise VideoProductionFreezeError("Production requires a valid immutable dataset lock")
    return lock_path, _sha256_file(lock_path)


def _production_document(candidate_spec: VideoAlgorithmSpec, *, production_algorithm_id: str) -> dict[str, object]:
    source = load_algorithm_config(candidate_spec.config_path)
    document = dict(source)
    for key in ("candidate_id", "parent_candidate_id", "changed_components", "config_sha256"):
        document.pop(key, None)
    document["config_schema"] = VIDEO_ALGORITHM_CONFIG_SCHEMA
    document["role"] = "production"
    document["algorithm_id"] = production_algorithm_id
    # Production may fall back only through the explicit structural-failure
    # path in the public pipeline; the config must declare that capability.
    document["allow_baseline_fallback"] = True
    return document


def freeze_production(
    *,
    validation_selection: str | Path,
    holdout_state: str | Path,
    candidate_config: str | Path,
    dataset_lock: str | Path,
    production_config: str | Path,
    production_lock: str | Path,
    production_algorithm_id: str = "production_v1",
) -> dict[str, object]:
    """Create a production YAML and lock from sealed selection/holdout evidence.

    Both output paths must be new.  In particular this function never "updates"
    a production lock after data, metrics, or code have changed.
    """

    if not isinstance(production_algorithm_id, str) or not production_algorithm_id:
        raise VideoProductionFreezeError("production_algorithm_id must be non-empty")
    selection_path, selection = _read_object(validation_selection, label="validation selection")
    selected_algorithm_id = _selection_algorithm(selection_path, selection)
    state_path, state = _read_valid_holdout_state(
        holdout_state, selection_path=selection_path, selected_algorithm_id=selected_algorithm_id
    )
    _, dataset_sha256 = _validate_dataset_lock(dataset_lock)
    try:
        candidate_spec = build_algorithm_spec(candidate_config, expected_role="candidate")
    except Exception as exc:
        raise VideoProductionFreezeError(f"Selected candidate config is invalid: {exc}") from exc
    if candidate_spec.algorithm_id != selected_algorithm_id:
        raise VideoProductionFreezeError("Candidate config does not match selected algorithm")
    state_algorithm = state.get("algorithm")
    if not isinstance(state_algorithm, dict):
        raise VideoProductionFreezeError("First holdout state lacks immutable algorithm identity")
    for field, observed in (
        ("algorithm_id", candidate_spec.algorithm_id),
        ("config_sha256", candidate_spec.config_sha256),
        ("source_commit", candidate_spec.source_commit),
        ("model_sha256", candidate_spec.model_sha256),
    ):
        if state_algorithm.get(field) != observed:
            raise VideoProductionFreezeError(f"Candidate config no longer matches first holdout {field}")

    config_target = Path(production_config).expanduser().resolve()
    lock_target = Path(production_lock).expanduser().resolve()
    if config_target == lock_target:
        raise VideoProductionFreezeError("Production config and lock paths must differ")
    if config_target.exists() or lock_target.exists():
        raise VideoProductionFreezeError("Refusing to overwrite an existing production config or lock")
    config_target.parent.mkdir(parents=True, exist_ok=True)
    document = _production_document(candidate_spec, production_algorithm_id=production_algorithm_id)
    pending_config = config_target.with_name(f".{config_target.name}.pending")
    if pending_config.exists():
        raise VideoProductionFreezeError(f"Pending production config already exists: {pending_config}")
    try:
        pending_config.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
        with pending_config.open("rb") as source, config_target.open("xb") as destination:
            destination.write(source.read())
    except FileExistsError as exc:
        raise VideoProductionFreezeError("Refusing to overwrite an existing production config") from exc
    finally:
        pending_config.unlink(missing_ok=True)
    try:
        production_spec = build_algorithm_spec(config_target, expected_role="production")
        relative_config = os.path.relpath(config_target, lock_target.parent).replace("\\", "/")
        lock: dict[str, object] = {
            "schema": VIDEO_ALGORITHM_LOCK_SCHEMA,
            "role": "production",
            "algorithm_id": production_spec.algorithm_id,
            "config_path": relative_config,
            "config_sha256": production_spec.config_sha256,
            "source_commit": production_spec.source_commit,
            "model_sha256": production_spec.model_sha256,
            "dataset_lock_sha256": dataset_sha256,
            "freeze_evidence": {
                "selected_candidate_algorithm_id": selected_algorithm_id,
                "validation_selection_sha256": _sha256_file(selection_path),
                "first_holdout_state_sha256": _sha256_file(state_path),
                "first_holdout_report_sha256": state["holdout_report_sha256"],
            },
        }
        _atomic_write_json_new(lock_target, lock)
    except Exception:
        # The config is evidence only if its paired lock was created.  Its
        # removal is a narrowly-scoped rollback of this function's new file.
        config_target.unlink(missing_ok=True)
        raise
    verify_algorithm_lock(lock_target, expected_role="production")
    return lock


__all__ = [
    "HOLDOUT_STATE_SCHEMA",
    "VideoProductionFreezeError",
    "freeze_production",
    "record_first_holdout",
]
