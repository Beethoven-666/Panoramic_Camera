"""Immutable-input locking for approved and diagnostic video experiment sessions.

The lock deliberately records bytes, not decoded pixels.  This makes an
accidental re-encode, replacement, or partial capture copy fail before a
benchmark can be compared with a previous result.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


APPROVED_RUN_NAME = "run_20260804_162340"
DIAGNOSTIC_DEVELOPMENT_RUN_NAME = "run_20260806_153033"
V6_TRACKING_GATE_RUN_NAME = "run_20260807_140140"
CONTROL_FILE_SHA256 = {
    "manifest.json": "11e52a86126b7a4445806bb7b8b82abd507d35f90e3c94797e5008d87af89cb0",
    "calibration.json": "9e19b8dc506b27834b4fa0166294deecb1c23d93e3b7bb93184b3aa8c5691330",
    "frames.csv": "f27d7dd4b675193a3846fa70fd1e8461da7898568b300c6e1e4ea190e1fcb42d",
}
V6_TRACKING_GATE_CONTROL_FILE_SHA256 = {
    "manifest.json": "0e82fa48b51703f228ee1922c1bfd2b7eebd74584d4deb7bfd872cf7242d07d0",
    "calibration.json": "9e19b8dc506b27834b4fa0166294deecb1c23d93e3b7bb93184b3aa8c5691330",
    "frames.csv": "ea9784b6c47e9d5b6b9898d82512a139e425cbab412d84dd55299037c7109380",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_root(value: Path) -> Path:
    value = value.expanduser().resolve()
    return value if value.is_dir() else value.parent


def _source_paths(root: Path) -> list[Path]:
    """Return capture source files in stable order and exclude generated state."""

    allowed_suffixes = {".jpg", ".jpeg", ".png"}
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and (path.parent.name in {"color", "rgb", "depth_aligned", "raw_depth"}
             or "aligned" in path.parent.name.lower())
    ]
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


@dataclass(frozen=True)
class DatasetLock:
    schema: str
    session: str
    control_sha256: dict[str, str]
    source_sha256: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "session": self.session,
            "control_sha256": dict(self.control_sha256),
            "source_sha256": dict(self.source_sha256),
        }


def create_dataset_lock(session: Path) -> DatasetLock:
    root = _session_root(session)
    if root.name != APPROVED_RUN_NAME:
        raise ValueError(f"Experiments only accept the locked session {APPROVED_RUN_NAME}")
    controls = {name: sha256_file(root / name) for name in CONTROL_FILE_SHA256}
    mismatched = [name for name, expected in CONTROL_FILE_SHA256.items() if controls[name] != expected]
    if mismatched:
        raise ValueError(f"Approved session control hash mismatch: {', '.join(mismatched)}")
    sources = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _source_paths(root)
    }
    if not sources:
        raise ValueError("Approved session has no RGB-D source images to lock")
    return DatasetLock(
        schema="gemini305-video-dataset-lock/v1",
        session=str(root),
        control_sha256=controls,
        source_sha256=sources,
    )


def _create_development_dataset_lock(session: Path) -> DatasetLock:
    """Lock the explicitly authorised diagnostic capture by bytes.

    This is deliberately a different lock schema and file name from the
    approved-session lock.  It is valid only for candidate development work;
    consequently it can neither replace the approved dataset lock nor be
    consumed by the first-holdout or production paths.
    """

    root = _session_root(session)
    if root.name != DIAGNOSTIC_DEVELOPMENT_RUN_NAME:
        raise ValueError(
            "Development experiments only accept the diagnostic session "
            f"{DIAGNOSTIC_DEVELOPMENT_RUN_NAME}"
        )
    controls = {
        name: sha256_file(root / name)
        for name in ("manifest.json", "calibration.json", "frames.csv")
    }
    sources = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _source_paths(root)
    }
    if not sources:
        raise ValueError("Diagnostic session has no RGB-D source images to lock")
    return DatasetLock(
        schema="gemini305-video-development-dataset-lock/v1",
        session=str(root),
        control_sha256=controls,
        source_sha256=sources,
    )


def create_v6_tracking_gate_dataset_lock(session: Path) -> DatasetLock:
    """Freeze the v6 FAST primary bytes for direct-ORB feasibility work only."""

    root = _session_root(session)
    if root.name != V6_TRACKING_GATE_RUN_NAME:
        raise ValueError(
            "v6 direct-ORB tracking gate only accepts the FAST primary session "
            f"{V6_TRACKING_GATE_RUN_NAME}"
        )
    controls = {
        name: sha256_file(root / name) for name in V6_TRACKING_GATE_CONTROL_FILE_SHA256
    }
    mismatched = [
        name
        for name, expected in V6_TRACKING_GATE_CONTROL_FILE_SHA256.items()
        if controls[name] != expected
    ]
    if mismatched:
        raise ValueError(
            "v6 FAST primary control hash mismatch: " + ", ".join(mismatched)
        )
    sources = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in _source_paths(root)
    }
    if not sources:
        raise ValueError("v6 FAST primary has no RGB-D source images to lock")
    return DatasetLock(
        schema="gemini305-video-v6-tracking-gate-dataset-lock/v1",
        session=str(root),
        control_sha256=controls,
        source_sha256=sources,
    )


def v6_tracking_gate_dataset_lock_path(benchmark_root: Path) -> Path:
    return benchmark_root / "v6_tracking_gate_dataset_lock.json"


def write_or_verify_v6_tracking_gate_dataset_lock(
    session: Path, benchmark_root: Path
) -> DatasetLock:
    """Create once and then byte-verify the isolated v6 tracking-gate input."""

    expected = create_v6_tracking_gate_dataset_lock(session)
    path = v6_tracking_gate_dataset_lock_path(benchmark_root)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid v6 tracking-gate dataset lock: {path}") from exc
        if saved != expected.as_dict():
            raise ValueError("v6 tracking-gate dataset lock mismatch; input has changed")
        return expected
    path.write_text(json.dumps(expected.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return expected


def write_dataset_lock(session: Path, benchmark_root: Path) -> DatasetLock:
    lock = create_dataset_lock(session)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    (benchmark_root / "dataset_lock.json").write_text(
        json.dumps(lock.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (benchmark_root / "source_files_sha256.json").write_text(
        json.dumps(lock.source_sha256, indent=2, sort_keys=True), encoding="utf-8"
    )
    return lock


def development_dataset_lock_path(benchmark_root: Path) -> Path:
    """Return the candidate-only lock location for a diagnostic benchmark."""

    return benchmark_root / "development_dataset_lock.json"


def write_development_dataset_lock(session: Path, benchmark_root: Path) -> DatasetLock:
    """Create the separate immutable lock for the authorised diagnostic session."""

    lock = _create_development_dataset_lock(session)
    benchmark_root.mkdir(parents=True, exist_ok=True)
    development_dataset_lock_path(benchmark_root).write_text(
        json.dumps(lock.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (benchmark_root / "development_source_files_sha256.json").write_text(
        json.dumps(lock.source_sha256, indent=2, sort_keys=True), encoding="utf-8"
    )
    return lock


def verify_dataset_lock(session: Path, lock_path: Path) -> DatasetLock:
    try:
        saved = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset lock: {lock_path}") from exc
    current = create_dataset_lock(session)
    if saved != current.as_dict():
        raise ValueError("Dataset lock mismatch; experiment input has changed")
    return current


def _is_diagnostic_development_session(session: Path) -> bool:
    return _session_root(session).name == DIAGNOSTIC_DEVELOPMENT_RUN_NAME


def require_candidate_role_for_diagnostic_session(session: Path, role: str) -> None:
    """Prevent the new capture from entering baseline or production workflows."""

    if _is_diagnostic_development_session(session) and role != "candidate":
        raise ValueError(
            "The diagnostic development session is candidate-only; it cannot run "
            f"as {role}"
        )


def write_or_verify_experiment_dataset_lock(
    session: Path, benchmark_root: Path, *, role: str
) -> DatasetLock:
    """Lock an experiment input without broadening the approved-session lock."""

    require_candidate_role_for_diagnostic_session(session, role)
    if _is_diagnostic_development_session(session):
        lock_path = development_dataset_lock_path(benchmark_root)
        if lock_path.is_file():
            return verify_development_dataset_lock(session, lock_path)
        return write_development_dataset_lock(session, benchmark_root)
    lock_path = benchmark_root / "dataset_lock.json"
    if lock_path.is_file():
        return verify_dataset_lock(session, lock_path)
    return write_dataset_lock(session, benchmark_root)


def verify_development_dataset_lock(session: Path, lock_path: Path) -> DatasetLock:
    try:
        saved = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid development dataset lock: {lock_path}") from exc
    current = _create_development_dataset_lock(session)
    if saved != current.as_dict():
        raise ValueError("Development dataset lock mismatch; experiment input has changed")
    return current


def verify_experiment_dataset_lock(session: Path, benchmark_root: Path, *, role: str) -> DatasetLock:
    """Verify the correct per-session lock without accepting it as a holdout lock."""

    require_candidate_role_for_diagnostic_session(session, role)
    if _is_diagnostic_development_session(session):
        return verify_development_dataset_lock(session, development_dataset_lock_path(benchmark_root))
    return verify_dataset_lock(session, benchmark_root / "dataset_lock.json")
