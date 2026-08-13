"""Fail-closed recovery checkpoint for video algorithm selection.

This module deliberately has no path that reserves holdout, writes a
production lock, or mutates the frozen session.  It snapshots the blocked
validation state so recovery changes can be audited against the exact state
that preceded them.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .video_runtime_environment import atomic_write_json, capture_runtime_environment


QUALITY_GATE_LOCK: dict[str, float | int] = {
    "object_internal_seam_count_max": 0,
    "object_owner_count_max": 1,
    "object_maximum_handoffs": 1,
    "line_step_p95_max_px": 1.0,
    "line_orientation_p95_max_deg": 3.0,
    "safe_background_delta_e00_p95_max": 3.0,
    "safe_background_brightness_step_max_percent": 2.0,
    "global_gain_max": 1.35,
    "global_bias_abs_max": 0.08,
    "mesh_minimum_jacobian": 0.05,
    "mesh_minimum_scale": 0.70,
    "mesh_maximum_scale": 1.40,
    "warm_median_max_seconds": 8.0,
    "warm_maximum_max_seconds": 9.0,
    "cold_max_seconds": 12.0,
    "predicted_20m_conservative_max_seconds": 55.0,
}


class VideoRecoveryError(RuntimeError):
    """A blocked state cannot safely be checkpointed."""


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible lock value deterministically."""

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoRecoveryError(f"Invalid required recovery JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VideoRecoveryError(f"Required recovery JSON is not an object: {path}")
    return value


def _atomic_write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Atomically publish a compact, deterministic candidate matrix."""

    fields = (
        "algorithm_id",
        "eligible",
        "report_path",
        "selection_reasons",
    )
    payload_path = path.with_suffix(path.suffix + ".pending")
    with payload_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload_path.replace(path)


def checkpoint_blocked_selection(
    benchmark_root: str | Path,
    *,
    commit: str,
    test_result: str,
) -> dict[str, object]:
    """Write R0 evidence without altering selection, holdout, or production.

    The caller supplies the Git commit and the already-observed test result so
    this function does not inspect or alter repository state beyond its
    dedicated ``recovery/`` artifacts.
    """

    root = Path(benchmark_root).expanduser().resolve()
    selection_path = root / "algorithm_selection_v2_current.json"
    holdout_path = root / "holdout_state.json"
    selection = _read_json(selection_path)
    holdout = _read_json(holdout_path)
    if selection.get("selection_status") != "not_selectable" or selection.get("selected_algorithm_id") is not None:
        raise VideoRecoveryError("R0 checkpoint requires an unselected blocked validation state")
    if holdout.get("first_holdout_attempted") is not False or holdout.get("production_frozen") is not False:
        raise VideoRecoveryError("R0 checkpoint refuses to overwrite a consumed holdout or production state")
    if any(root.rglob("production.lock.json")):
        raise VideoRecoveryError("R0 checkpoint refuses a benchmark tree that already has a production lock")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        raise VideoRecoveryError("Blocked selection lacks a candidate list")
    recovery = root / "recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    quality_lock = {
        "schema": "gemini305-video-quality-gate-lock/v1",
        "quality_gate_version": "quality-gates-v1-unchanged",
        "gates": QUALITY_GATE_LOCK,
    }
    quality_lock["sha256"] = canonical_sha256(quality_lock)
    atomic_write_json(root / "quality_gate_lock.json", quality_lock)
    snapshot = {
        "schema": "gemini305-video-selection-recovery-blocked-snapshot/v1",
        "source_selection_path": str(selection_path),
        "source_selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(),
        "selection": selection,
        "holdout_state": holdout,
        "commit": commit,
        "quality_gate_lock_sha256": quality_lock["sha256"],
        "holdout_not_reserved": True,
        "production_lock_created": False,
    }
    atomic_write_json(recovery / "blocked_selection_snapshot.json", snapshot)
    matrix: list[dict[str, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise VideoRecoveryError("Blocked selection candidate entry is invalid")
        reasons = candidate.get("reasons", [])
        matrix.append(
            {
                "algorithm_id": str(candidate.get("algorithm_id", "")),
                "eligible": str(candidate.get("eligible") is True).lower(),
                "report_path": str(candidate.get("report_path", "")),
                "selection_reasons": ";".join(str(reason) for reason in reasons) if isinstance(reasons, list) else "invalid_reasons",
            }
        )
    _atomic_write_csv(recovery / "blocked_candidate_matrix.csv", matrix)
    (recovery / "blocked_test_result.txt").write_text(test_result.rstrip() + "\n", encoding="utf-8")
    environment = {
        "schema": "gemini305-video-selection-recovery-environment/v1",
        "commit": commit,
        "runtime": capture_runtime_environment(),
        "quality_gate_lock_sha256": quality_lock["sha256"],
    }
    atomic_write_json(recovery / "blocked_environment.json", environment)
    return {
        "recovery_root": str(recovery),
        "quality_gate_lock_sha256": quality_lock["sha256"],
        "candidate_count": len(matrix),
        "holdout_not_reserved": True,
        "production_lock_created": False,
    }


__all__ = ["QUALITY_GATE_LOCK", "VideoRecoveryError", "canonical_sha256", "checkpoint_blocked_selection"]
