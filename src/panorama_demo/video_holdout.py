"""One-time, fail-closed first-holdout lifecycle and 20 m script generation.

The development CLI deliberately has no holdout option.  This module keeps
the first blind evaluation separate, consuming an atomically-created ledger
*before* any holdout pixels are rendered.  A failed or interrupted first run
therefore cannot be silently retried and represented as the original blind
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
from pathlib import Path
from .video_algorithm_lock import VideoAlgorithmLock, verify_algorithm_lock
from .video_dataset_lock import verify_dataset_lock
from .video_runtime_environment import atomic_write_json


HOLDOUT_STATE_SCHEMA = "gemini305-video-first-holdout-state/v1"


class VideoHoldoutError(RuntimeError):
    """A proposed first holdout is not safely eligible to run."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_selection(path: Path) -> tuple[dict[str, object], str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoHoldoutError(f"Invalid validation selection: {path}") from exc
    if not isinstance(document, dict):
        raise VideoHoldoutError("Validation selection must be a JSON object")
    if document.get("schema") != "gemini305-video-algorithm-selection/v1":
        raise VideoHoldoutError("Unsupported validation selection schema")
    if document.get("selection_status") != "ready_for_first_holdout":
        raise VideoHoldoutError("Validation selection is not ready for first holdout")
    if document.get("holdout_not_run") is not True:
        raise VideoHoldoutError("Validation selection does not assert holdout_not_run")
    selected = document.get("selected_algorithm_id")
    if not isinstance(selected, str) or not selected:
        raise VideoHoldoutError("Validation selection lacks exactly one selected algorithm")
    return document, _sha256_file(path)


def _state_payload(
    *, selection: dict[str, object], selection_path: Path, selection_sha256: str,
    candidate_lock: Path, lock: VideoAlgorithmLock, dataset_lock: Path,
    attempt_token: str,
) -> dict[str, object]:
    return {
        "schema": HOLDOUT_STATE_SCHEMA,
        "status": "reserved",
        "first_holdout_consumed": True,
        "first_holdout_pass": None,
        "attempt_token": attempt_token,
        "selected_algorithm_id": selection["selected_algorithm_id"],
        "validation_selection_path": str(selection_path),
        "validation_selection_sha256": selection_sha256,
        "candidate_lock_path": str(candidate_lock),
        "candidate_lock_sha256": _sha256_file(candidate_lock),
        "dataset_lock_path": str(dataset_lock),
        "dataset_lock_sha256": _sha256_file(dataset_lock),
        "algorithm": {
            "algorithm_id": lock.algorithm_id,
            "config_sha256": lock.config_sha256,
            "source_commit": lock.source_commit,
            "model_sha256": dict(lock.model_sha256),
        },
    }


def reserve_first_holdout(
    *, session: str | Path, dataset_lock: str | Path, selection_path: str | Path,
    candidate_lock: str | Path, state_path: str | Path,
) -> dict[str, object]:
    """Atomically consume the one permitted first holdout before rendering.

    ``state_path`` must not already exist.  ``x`` mode is used deliberately:
    replacement/cleanup of a failed state would permit a contaminated retry.
    """

    session_path = Path(session).expanduser().resolve()
    root = session_path if session_path.is_dir() else session_path.parent
    dataset = Path(dataset_lock).expanduser().resolve()
    selection_file = Path(selection_path).expanduser().resolve()
    lock_file = Path(candidate_lock).expanduser().resolve()
    state = Path(state_path).expanduser().resolve()
    if state.exists():
        raise VideoHoldoutError(
            f"First holdout has already been reserved or completed: {state}"
        )
    verify_dataset_lock(root, dataset)
    selection, selection_sha256 = _read_selection(selection_file)
    spec = verify_algorithm_lock(lock_file, expected_role="candidate")
    if spec.algorithm_id != selection["selected_algorithm_id"]:
        raise VideoHoldoutError("Candidate lock does not match selected validation algorithm")
    # Read after validation so the state binds the verified immutable lock.
    lock = VideoAlgorithmLock(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        config_path=spec.config_path,
        config_sha256=spec.config_sha256,
        source_commit=spec.source_commit,
        model_sha256=dict(spec.model_sha256),
        dataset_lock_sha256=None,
    )
    payload = _state_payload(
        selection=selection, selection_path=selection_file,
        selection_sha256=selection_sha256, candidate_lock=lock_file, lock=lock,
        dataset_lock=dataset, attempt_token=secrets.token_hex(24),
    )
    state.parent.mkdir(parents=True, exist_ok=True)
    try:
        with state.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise VideoHoldoutError(
            f"First holdout has already been reserved or completed: {state}"
        ) from exc
    return payload


def complete_first_holdout(
    state_path: str | Path, *, attempt_token: str, passed: bool,
    report_path: str | Path | None = None, error: str | None = None,
) -> dict[str, object]:
    """Record the sole first-holdout outcome; completed states are immutable."""

    state = Path(state_path).expanduser().resolve()
    try:
        current = json.loads(state.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoHoldoutError(f"Invalid first holdout state: {state}") from exc
    if not isinstance(current, dict) or current.get("schema") != HOLDOUT_STATE_SCHEMA:
        raise VideoHoldoutError("Unsupported first holdout state")
    if current.get("status") != "reserved" or current.get("attempt_token") != attempt_token:
        raise VideoHoldoutError("First holdout state is not owned by this reserved attempt")
    current["status"] = "passed" if passed else "failed"
    current["first_holdout_pass"] = bool(passed)
    current["report_path"] = str(Path(report_path).expanduser().resolve()) if report_path else None
    current["error"] = error
    atomic_write_json(state, current)
    return current


def _production_script(*, production_lock: Path, lock_sha256: str, algorithm_id: str) -> str:
    lock_literal = str(production_lock).replace("'", "''")
    return f'''# Generated only after a verified immutable production lock.\n# This script runs a real 20 m capture result; it does not claim the result in advance.\n[CmdletBinding()]\nparam(\n    [Parameter(Mandatory = $true)] [ValidateNotNullOrEmpty()] [string] $Session,\n    [Parameter(Mandatory = $true)] [ValidateNotNullOrEmpty()] [string] $Output,\n    [string] $VideoPanorama = 'D:\\Panoramic_Camera\\.conda\\Scripts\\g305-video-panorama.exe'\n)\n$ErrorActionPreference = 'Stop'\n$ExpectedLockSha256 = '{lock_sha256}'\n$ProductionLock = '{lock_literal}'\n$ExpectedAlgorithmId = '{algorithm_id}'\nif (-not (Test-Path -LiteralPath $Session -PathType Container)) {{ throw "Session does not exist: $Session" }}\nif (-not (Test-Path -LiteralPath $ProductionLock -PathType Leaf)) {{ throw "Production lock does not exist: $ProductionLock" }}\n$actualLock = (Get-FileHash -LiteralPath $ProductionLock -Algorithm SHA256).Hash.ToLowerInvariant()\nif ($actualLock -ne $ExpectedLockSha256) {{ throw 'Production lock changed after this script was generated.' }}\nif (-not (Test-Path -LiteralPath $VideoPanorama -PathType Leaf)) {{ throw "g305-video-panorama executable does not exist: $VideoPanorama" }}\n& $VideoPanorama $Session --maximum-post-seconds 60 --output $Output\nif ($LASTEXITCODE -ne 0) {{ throw "g305-video-panorama failed with exit code $LASTEXITCODE" }}\n$delivery = Join-Path $Output 'video_delivery.json'\nif (-not (Test-Path -LiteralPath $delivery -PathType Leaf)) {{ throw 'No video_delivery.json was published.' }}\n$published = Get-Content -LiteralPath $delivery -Raw | ConvertFrom-Json\nif ($published.algorithm.algorithm_id -ne $ExpectedAlgorithmId) {{ throw 'Published algorithm identity differs from the frozen production lock.' }}\nWrite-Host "20 m run published with production algorithm $ExpectedAlgorithmId. Inspect video_delivery.json and video_report.json; this run is the real acceptance result."\n'''


def write_user_20m_test_script(
    output: str | Path, *, production_lock: str | Path,
) -> Path:
    """Generate a user-facing script only after validating a production lock."""

    lock_file = Path(production_lock).expanduser().resolve()
    spec = verify_algorithm_lock(lock_file, expected_role="production")
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _production_script(
            production_lock=lock_file, lock_sha256=_sha256_file(lock_file),
            algorithm_id=spec.algorithm_id,
        ),
        encoding="utf-8",
        newline="\r\n",
    )
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the production-gated user 20 m test script")
    parser.add_argument("--production-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        output = write_user_20m_test_script(args.output, production_lock=args.production_lock)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(output)


__all__ = [
    "HOLDOUT_STATE_SCHEMA",
    "VideoHoldoutError",
    "complete_first_holdout",
    "reserve_first_holdout",
    "write_user_20m_test_script",
]
