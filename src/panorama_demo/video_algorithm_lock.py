"""Read and verify immutable baseline/production algorithm locks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .video_algorithm import (
    AlgorithmRole,
    VideoAlgorithmConfigurationError,
    VideoAlgorithmSpec,
    build_algorithm_spec,
)


VIDEO_ALGORITHM_LOCK_SCHEMA = "gemini305-video-algorithm-lock/v1"


class VideoAlgorithmLockError(RuntimeError):
    """An immutable lock is absent, malformed, or no longer matches its config."""


@dataclass(frozen=True)
class VideoAlgorithmLock:
    role: AlgorithmRole
    algorithm_id: str
    config_path: Path
    config_sha256: str
    source_commit: str
    model_sha256: dict[str, str]
    dataset_lock_sha256: str | None


def _lock_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise VideoAlgorithmLockError(f"Algorithm lock requires non-empty {key!r}")
    return value


def read_algorithm_lock(path: str | Path, *, expected_role: AlgorithmRole | None = None) -> VideoAlgorithmLock:
    lock_path = Path(path).expanduser().resolve()
    if not lock_path.is_file():
        raise VideoAlgorithmLockError(f"Algorithm lock does not exist: {lock_path}")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VideoAlgorithmLockError(f"Invalid algorithm lock JSON: {lock_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != VIDEO_ALGORITHM_LOCK_SCHEMA:
        raise VideoAlgorithmLockError(f"Unsupported algorithm lock schema: {lock_path}")
    role = payload.get("role")
    if role not in ("baseline", "candidate", "production"):
        raise VideoAlgorithmLockError("Algorithm lock role is invalid")
    if expected_role is not None and role != expected_role:
        raise VideoAlgorithmLockError(f"Expected {expected_role} lock, received {role}")
    config_ref = _lock_string(payload, "config_path")
    config_path = (lock_path.parent / config_ref).resolve()
    model_sha256 = payload.get("model_sha256", {})
    if not isinstance(model_sha256, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in model_sha256.items()
    ):
        raise VideoAlgorithmLockError("model_sha256 must be a mapping of strings")
    dataset_hash = payload.get("dataset_lock_sha256")
    if dataset_hash is not None and not isinstance(dataset_hash, str):
        raise VideoAlgorithmLockError("dataset_lock_sha256 must be a string or null")
    return VideoAlgorithmLock(
        role=role,
        algorithm_id=_lock_string(payload, "algorithm_id"),
        config_path=config_path,
        config_sha256=_lock_string(payload, "config_sha256"),
        source_commit=_lock_string(payload, "source_commit"),
        model_sha256=dict(model_sha256),
        dataset_lock_sha256=dataset_hash,
    )


def verify_algorithm_lock(path: str | Path, *, expected_role: AlgorithmRole | None = None) -> VideoAlgorithmSpec:
    """Return a spec only when its config matches every lock identity field."""

    lock = read_algorithm_lock(path, expected_role=expected_role)
    try:
        spec = build_algorithm_spec(lock.config_path, expected_role=lock.role)
    except VideoAlgorithmConfigurationError as exc:
        raise VideoAlgorithmLockError(f"Locked algorithm config is invalid: {exc}") from exc
    mismatches = {
        "algorithm_id": (lock.algorithm_id, spec.algorithm_id),
        "config_sha256": (lock.config_sha256, spec.config_sha256),
        "source_commit": (lock.source_commit, spec.source_commit),
        "model_sha256": (lock.model_sha256, spec.model_sha256),
    }
    failed = [name for name, (expected, observed) in mismatches.items() if expected != observed]
    if failed:
        raise VideoAlgorithmLockError("Algorithm lock mismatch: " + ", ".join(failed))
    if lock.role == "production" and not lock.dataset_lock_sha256:
        raise VideoAlgorithmLockError("Production lock requires dataset_lock_sha256")
    return spec
