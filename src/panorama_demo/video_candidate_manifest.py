"""Locked, config-derived contracts for candidate experiments.

The manifest is deliberately separate from a mutable candidate YAML.  A
candidate run records both hashes, while this module verifies that the two
documents say exactly the same thing about component lineage.  This prevents a
selection rule from silently acquiring a second, hard-coded candidate table.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping



class CandidateManifestError(ValueError):
    """A locked candidate manifest cannot establish an execution contract."""


VIDEO_CANDIDATE_MANIFEST_SCHEMA = "gemini305-video-candidate-manifest/v1"
_MANIFEST_NAME = "candidate_manifest.json"


@dataclass(frozen=True)
class CandidateComponentContract:
    """The immutable execution obligations of a single candidate."""

    candidate_id: str
    config_sha256: str
    manifest_path: Path
    manifest_sha256: str
    required_evidence_components: tuple[str, ...]
    required_output_components: tuple[str, ...]
    replaces_output_components: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "config_sha256": self.config_sha256,
            "candidate_manifest_path": str(self.manifest_path),
            "candidate_manifest_sha256": self.manifest_sha256,
            "required_evidence_components": list(self.required_evidence_components),
            "required_output_components": list(self.required_output_components),
            "replaces_output_components": list(self.replaces_output_components),
        }


def canonical_candidate_manifest_sha256(payload: Mapping[str, object]) -> str:
    """Hash a manifest while excluding its self-referential digest."""

    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    try:
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CandidateManifestError("Candidate manifest is not JSON-canonicalizable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_config_sha256(config: Mapping[str, object]) -> str:
    canonical = dict(config)
    canonical.pop("config_sha256", None)
    try:
        encoded = json.dumps(
            canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CandidateManifestError("Candidate config is not JSON-canonicalizable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _unique_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item for item in value
    ) or len(set(value)) != len(value):
        raise CandidateManifestError(
            f"candidate {field} must be a non-empty unique string list"
        )
    return tuple(value)


def candidate_manifest_path(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().resolve().parent / _MANIFEST_NAME


def load_candidate_component_contract(
    config_path: str | Path, config: Mapping[str, Any]
) -> CandidateComponentContract:
    """Read and verify the locked entry matching a candidate YAML.

    The manifest may not introduce, remove, or reinterpret any component
    declaration in the YAML.  Conversely, selection only trusts the manifest
    hash recorded in the final report, never an in-memory candidate table.
    """

    path = candidate_manifest_path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CandidateManifestError(
            f"Candidate manifest is missing or invalid: {path}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("schema") != VIDEO_CANDIDATE_MANIFEST_SCHEMA:
        raise CandidateManifestError(f"Unsupported candidate manifest: {path}")
    claimed = payload.get("manifest_sha256")
    unhashed = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual = canonical_candidate_manifest_sha256(unhashed)
    if not isinstance(claimed, str) or claimed != actual:
        raise CandidateManifestError(f"Candidate manifest SHA mismatch: {path}")
    candidate_id = config.get("candidate_id")
    entries = payload.get("candidates")
    if not isinstance(candidate_id, str) or not isinstance(entries, dict):
        raise CandidateManifestError(f"Candidate manifest is malformed: {path}")
    entry = entries.get(candidate_id)
    if not isinstance(entry, dict):
        raise CandidateManifestError(
            f"Candidate manifest has no entry for {candidate_id!r}: {path}"
        )
    config_sha = _canonical_config_sha256(config)
    if entry.get("config_sha256") != config_sha:
        raise CandidateManifestError(
            f"Candidate manifest config SHA mismatch for {candidate_id!r}"
        )
    evidence = _unique_strings(
        entry.get("required_evidence_components"), field="required_evidence_components"
    )
    output = _unique_strings(
        entry.get("required_output_components"), field="required_output_components"
    )
    replaces_value = entry.get("replaces_output_components", [])
    if not isinstance(replaces_value, list) or not all(
        isinstance(item, str) and item for item in replaces_value
    ) or len(set(replaces_value)) != len(replaces_value):
        raise CandidateManifestError(
            "candidate replaces_output_components must be a unique string list"
        )
    replaces = tuple(replaces_value)
    for field, expected in (
        ("required_evidence_components", evidence),
        ("required_output_components", output),
        ("replaces_output_components", replaces),
    ):
        value = config.get(field)
        if not isinstance(value, list) or tuple(value) != expected:
            raise CandidateManifestError(
                f"Candidate manifest {field} disagrees with candidate YAML for {candidate_id!r}"
            )
    if set(evidence) & set(output):
        raise CandidateManifestError(
            "candidate evidence and output components must be disjoint"
        )
    if set(replaces) & set(output):
        raise CandidateManifestError(
            "candidate cannot require a replaced output component as output"
        )
    return CandidateComponentContract(
        candidate_id=candidate_id,
        config_sha256=config_sha,
        manifest_path=path,
        manifest_sha256=claimed,
        required_evidence_components=evidence,
        required_output_components=output,
        replaces_output_components=replaces,
    )


__all__ = [
    "CandidateComponentContract",
    "CandidateManifestError",
    "VIDEO_CANDIDATE_MANIFEST_SCHEMA",
    "candidate_manifest_path",
    "canonical_candidate_manifest_sha256",
    "load_candidate_component_contract",
]
