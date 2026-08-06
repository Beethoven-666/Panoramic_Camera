"""Immutable-input locking for the single approved video experiment session.

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
CONTROL_FILE_SHA256 = {
    "manifest.json": "11e52a86126b7a4445806bb7b8b82abd507d35f90e3c94797e5008d87af89cb0",
    "calibration.json": "9e19b8dc506b27834b4fa0166294deecb1c23d93e3b7bb93184b3aa8c5691330",
    "frames.csv": "f27d7dd4b675193a3846fa70fd1e8461da7898568b300c6e1e4ea190e1fcb42d",
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


def verify_dataset_lock(session: Path, lock_path: Path) -> DatasetLock:
    try:
        saved = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid dataset lock: {lock_path}") from exc
    current = create_dataset_lock(session)
    if saved != current.as_dict():
        raise ValueError("Dataset lock mismatch; experiment input has changed")
    return current
