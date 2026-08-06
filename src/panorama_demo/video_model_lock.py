"""Verified local model assets for candidate-only video algorithms.

An algorithm declaration names a model by digest; this module ties that name
to a versioned manifest and, when execution is requested, to the exact local
bytes.  It deliberately does not fetch anything from the network and is never
used by the public production entry point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .paths import PROJECT_ROOT
from .video_raft_runtime import sha256_file


VIDEO_MODEL_LOCK_SCHEMA = "gemini305-video-model-lock/v1"


class VideoModelLockError(RuntimeError):
    """A candidate model manifest or local file is not immutable evidence."""


@dataclass(frozen=True)
class VideoModelLock:
    model_id: str
    path: Path
    sha256: str
    implementation: str
    license: str
    license_url: str
    candidate_only: bool


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise VideoModelLockError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise VideoModelLockError(f"{label} must be a SHA-256 hex digest") from exc
    return value.lower()


def _string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise VideoModelLockError(f"Model lock requires non-empty {key!r}")
    return value


def read_video_model_lock(path: str | Path) -> VideoModelLock:
    """Read a committed candidate model manifest without reading its weights."""

    manifest = Path(path).expanduser().resolve()
    if not manifest.is_file():
        raise VideoModelLockError(f"Candidate model lock does not exist: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoModelLockError(f"Candidate model lock is invalid JSON: {manifest}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != VIDEO_MODEL_LOCK_SCHEMA:
        raise VideoModelLockError(f"Unsupported candidate model lock schema: {manifest}")
    candidate_only = payload.get("candidate_only")
    if candidate_only is not True:
        raise VideoModelLockError("Candidate model lock must explicitly set candidate_only=true")
    path_ref = _string(payload, "file")
    model_path = (PROJECT_ROOT / path_ref).resolve()
    return VideoModelLock(
        model_id=_string(payload, "model_id"),
        path=model_path,
        sha256=_sha256(payload.get("sha256"), label="model lock sha256"),
        implementation=_string(payload, "implementation"),
        license=_string(payload, "license"),
        license_url=_string(payload, "license_url"),
        candidate_only=True,
    )


def verify_candidate_models(
    model_sha256: Mapping[str, str], *, require_files: bool = True
) -> tuple[VideoModelLock, ...]:
    """Resolve every declared model to a committed manifest and exact bytes.

    Empty model maps are valid.  Unknown IDs, a hash mismatch, absent local
    assets, and changed bytes all stop candidate execution before rendering.
    """

    locks: list[VideoModelLock] = []
    for model_id, expected in sorted(model_sha256.items()):
        manifest_root = PROJECT_ROOT / "configs" / "video_algorithms"
        matching = []
        for manifest in manifest_root.glob("*.model.json"):
            try:
                lock_candidate = read_video_model_lock(manifest)
            except VideoModelLockError:
                raise
            if lock_candidate.model_id == model_id:
                matching.append(lock_candidate)
        if not matching:
            raise VideoModelLockError(f"Candidate model manifest is not registered: {model_id}")
        if len(matching) != 1:
            raise VideoModelLockError(f"Candidate model manifest is ambiguous: {model_id}")
        lock = matching[0]
        if lock.model_id != model_id:
            raise VideoModelLockError(
                f"Candidate model manifest id mismatch: expected {model_id}, received {lock.model_id}"
            )
        expected_sha = _sha256(expected, label=f"candidate model_sha256.{model_id}")
        if lock.sha256 != expected_sha:
            raise VideoModelLockError(
                f"Candidate model lock SHA-256 mismatch for {model_id}"
            )
        if require_files:
            if not lock.path.is_file():
                raise VideoModelLockError(f"Candidate model file is required: {lock.path}")
            actual = sha256_file(lock.path)
            if actual != lock.sha256:
                raise VideoModelLockError(
                    f"Candidate model file SHA-256 mismatch for {model_id}: {actual}"
                )
        locks.append(lock)
    return tuple(locks)


__all__ = [
    "VIDEO_MODEL_LOCK_SCHEMA",
    "VideoModelLock",
    "VideoModelLockError",
    "read_video_model_lock",
    "verify_candidate_models",
]
