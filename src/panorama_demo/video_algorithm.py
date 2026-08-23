"""Immutable video algorithm specifications and candidate-config validation.

This module intentionally has no dependency on the renderer.  It is the
single schema boundary between a development candidate YAML and the pipeline
which consumes it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


AlgorithmRole = Literal["baseline", "candidate", "production"]
VIDEO_ALGORITHM_CONFIG_SCHEMA = "gemini305-video-algorithm/v1"
VIDEO_CANDIDATE_CONFIG_SCHEMA = "gemini305-video-candidate/v1"


class VideoAlgorithmConfigurationError(ValueError):
    """An algorithm declaration is incomplete, unsafe, or inconsistent."""


def candidate_runtime_git_identity() -> tuple[str, bool]:
    """Return the real repository identity for a candidate experiment.

    Candidate YAML is deliberately not allowed to carry a long-lived commit
    assertion.  A dirty tree is still recorded, but is made ineligible by the
    selector rather than being silently treated as a reproducible result.
    """

    repository = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True,
            text=True, check=True, timeout=3,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain=v1"], cwd=repository, capture_output=True,
            text=True, check=True, timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise VideoAlgorithmConfigurationError(
            "Cannot determine candidate runtime git identity"
        ) from exc
    if len(head) != 40:
        raise VideoAlgorithmConfigurationError("Candidate runtime git HEAD is invalid")
    return head, bool(status.strip())


@dataclass(frozen=True)
class VideoAlgorithmSpec:
    """Identity and immutable provenance of one algorithm invocation."""

    role: AlgorithmRole
    algorithm_id: str
    implementation_id: str
    config_path: Path
    config_sha256: str
    source_commit: str
    model_sha256: dict[str, str]
    allow_baseline_fallback: bool
    required_components: tuple[str, ...] = ()
    working_tree_dirty: bool = False
    candidate_manifest_path: Path | None = None
    candidate_manifest_sha256: str | None = None
    required_evidence_components: tuple[str, ...] = ()
    required_output_components: tuple[str, ...] = ()
    replaces_output_components: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "algorithm_id": self.algorithm_id,
            "implementation_id": self.implementation_id,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "source_commit": self.source_commit,
            "model_sha256": dict(self.model_sha256),
            "allow_baseline_fallback": self.allow_baseline_fallback,
            "required_components": list(self.required_components),
            "working_tree_dirty": self.working_tree_dirty,
            "candidate_manifest_path": (
                str(self.candidate_manifest_path) if self.candidate_manifest_path else None
            ),
            "candidate_manifest_sha256": self.candidate_manifest_sha256,
            "required_evidence_components": list(self.required_evidence_components),
            "required_output_components": list(self.required_output_components),
            "replaces_output_components": list(self.replaces_output_components),
        }


def _require_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise VideoAlgorithmConfigurationError(f"Algorithm config requires non-empty {key!r}")
    return value


def _validate_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise VideoAlgorithmConfigurationError(f"{field} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise VideoAlgorithmConfigurationError(f"{field} must be a SHA-256 hex digest") from exc
    return value.lower()


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash a config while excluding its self-referential ``config_sha256``.

    Candidate YAMLs carry their own digest for auditability.  Excluding that
    one field makes the declaration verifiable without a hash fixed point and
    makes whitespace/key ordering irrelevant.
    """

    canonical = dict(config)
    canonical.pop("config_sha256", None)
    try:
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VideoAlgorithmConfigurationError("Algorithm config is not JSON-canonicalizable") from exc
    return hashlib.sha256(encoded).hexdigest()


def load_algorithm_config(path: str | Path) -> dict[str, Any]:
    """Load one standalone YAML algorithm declaration without merging defaults."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise VideoAlgorithmConfigurationError(f"Algorithm config does not exist: {config_path}")
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VideoAlgorithmConfigurationError(f"Invalid algorithm YAML: {config_path}") from exc
    if not isinstance(document, dict):
        raise VideoAlgorithmConfigurationError("Algorithm config root must be a mapping")
    return document


def _validate_common_config(config: Mapping[str, Any]) -> AlgorithmRole:
    role = config.get("role")
    if role not in ("baseline", "candidate", "production"):
        raise VideoAlgorithmConfigurationError("role must be baseline, candidate, or production")
    _require_string(config, "algorithm_id")
    _require_string(config, "implementation_id")
    if role != "candidate":
        _require_string(config, "source_commit")
    model_hashes = config.get("model_sha256", {})
    if not isinstance(model_hashes, dict) or not all(
        isinstance(name, str) and _validate_sha256(digest, field=f"model_sha256.{name}")
        for name, digest in model_hashes.items()
    ):
        raise VideoAlgorithmConfigurationError("model_sha256 must map model names to SHA-256 digests")
    return role


def build_algorithm_spec(path: str | Path, *, expected_role: AlgorithmRole | None = None) -> VideoAlgorithmSpec:
    """Validate ``path`` and return its immutable public identity.

    This never merges ``demo.yaml``: a baseline must remain reproducible even
    after shared runtime defaults are refactored.
    """

    config_path = Path(path).expanduser().resolve()
    config = load_algorithm_config(config_path)
    role = _validate_common_config(config)
    if expected_role is not None and role != expected_role:
        raise VideoAlgorithmConfigurationError(
            f"Expected {expected_role} config, received {role}: {config_path}"
        )
    schema = config.get("config_schema")
    contract = None
    working_tree_dirty = False
    source_commit: str
    if role == "candidate":
        if schema != VIDEO_CANDIDATE_CONFIG_SCHEMA:
            raise VideoAlgorithmConfigurationError(
                f"Candidate config_schema must be {VIDEO_CANDIDATE_CONFIG_SCHEMA!r}"
            )
        candidate_id = _require_string(config, "candidate_id")
        if candidate_id != config["algorithm_id"]:
            raise VideoAlgorithmConfigurationError("candidate_id must equal algorithm_id")
        if "parent_candidate_id" not in config or (
            config["parent_candidate_id"] is not None
            and not isinstance(config["parent_candidate_id"], str)
        ):
            raise VideoAlgorithmConfigurationError("candidate requires parent_candidate_id")
        changed = config.get("changed_components")
        if not isinstance(changed, list) or not all(isinstance(item, str) and item for item in changed):
            raise VideoAlgorithmConfigurationError("candidate requires changed_components strings")
        # ``required_components`` was an ambiguous old declaration.  The
        # locked manifest separates evidence from pixels that must actually
        # reach the final output.
        declared_hash = _validate_sha256(config.get("config_sha256"), field="config_sha256")
        actual_hash = canonical_config_sha256(config)
        if declared_hash != actual_hash:
            raise VideoAlgorithmConfigurationError(
                f"Candidate config_sha256 mismatch for {config_path}: {declared_hash} != {actual_hash}"
            )
        from .video_candidate_manifest import CandidateManifestError, load_candidate_component_contract

        try:
            contract = load_candidate_component_contract(config_path, config)
        except CandidateManifestError as exc:
            raise VideoAlgorithmConfigurationError(str(exc)) from exc
        source_commit, working_tree_dirty = candidate_runtime_git_identity()
    elif schema != VIDEO_ALGORITHM_CONFIG_SCHEMA:
        raise VideoAlgorithmConfigurationError(
            f"Algorithm config_schema must be {VIDEO_ALGORITHM_CONFIG_SCHEMA!r}"
        )
    else:
        source_commit = _require_string(config, "source_commit")
    allow_fallback = config.get("allow_baseline_fallback", False)
    if not isinstance(allow_fallback, bool):
        raise VideoAlgorithmConfigurationError("allow_baseline_fallback must be a bool")
    return VideoAlgorithmSpec(
        role=role,
        algorithm_id=_require_string(config, "algorithm_id"),
        implementation_id=_require_string(config, "implementation_id"),
        config_path=config_path,
        config_sha256=canonical_config_sha256(config),
        source_commit=source_commit,
        model_sha256={str(key): str(value).lower() for key, value in config.get("model_sha256", {}).items()},
        allow_baseline_fallback=allow_fallback,
        required_components=(contract.required_output_components if contract else tuple(config.get("required_components", ()))),
        working_tree_dirty=working_tree_dirty,
        candidate_manifest_path=contract.manifest_path if contract else None,
        candidate_manifest_sha256=contract.manifest_sha256 if contract else None,
        required_evidence_components=contract.required_evidence_components if contract else (),
        required_output_components=contract.required_output_components if contract else (),
        replaces_output_components=contract.replaces_output_components if contract else (),
    )
