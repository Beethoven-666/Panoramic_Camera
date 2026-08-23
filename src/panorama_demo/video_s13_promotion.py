"""Auditable promotion boundary for the exact S013 Visual Continuity V11.

The candidate declaration remains immutable.  A production declaration may
change lifecycle, publication, and observation policy, but its pixel-producing
contract must remain byte-for-byte equivalent after canonicalisation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .video_algorithm import canonical_config_sha256, load_algorithm_config
from .video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)


S13_V11_CANDIDATE_CONFIG_SHA256 = (
    "a03f443aaf72463afc1f06507fe2bfc888d9211f4c4a62634bbfb29fa0f2dab0"
)
S13_V11_CANDIDATE_SOURCE_COMMIT = "0cc93e7cff231ffc2370847034aa89a727023269"
S13_V11_CANDIDATE_RELATIVE_PATH = (
    "configs/video_candidates/s013/"
    "S013_output_first_progressive_dense_central_slit_v4_cuda_m63_"
    "visual_continuity_v11.yaml"
)

_COMPONENT_NAME = "s013_output_first_progressive_dense_central_slit"
_TOP_LEVEL_NON_PIXEL_FIELDS = frozenset(
    {
        "config_schema",
        "role",
        "source_commit",
        "promoted_from",
        "config_sha256",
        "diagnostic_only",
        "production_eligible",
        "production_lock_eligible",
    }
)
_COMPONENT_NON_PIXEL_FIELDS = frozenset(
    {"diagnostic_only", "production_eligible", "production_lock_eligible", "output"}
)
# These are the closed observation/encoding controls present in V11.  Keeping
# unknown runtime keys in the canonical contract makes future compute-affecting
# additions fail closed instead of being silently classified as telemetry.
_RUNTIME_NON_PIXEL_FIELDS = frozenset(
    {
        "stage_image_policy",
        "output_image_format",
        "png_compression",
        "write_jpg",
        "write_masks",
        "write_overlays",
        "write_provenance",
        "write_replay_assets",
        "write_transactions",
        "write_performance_json",
        "write_quality_report",
        "write_reproducibility",
        "write_completion",
        "write_pointer",
        "sha_validation",
        "stage_sealing",
        "promoted_stage_reverify",
        "independent_raw_rgb_replay",
        "asynchronous_stage_writer",
        "stage_writer_workers",
        "stage_writer_queue_size",
    }
)


class S13V11PromotionError(ValueError):
    """The proposed production declaration is not an exact V11 promotion."""


def _require_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise S13V11PromotionError(f"S013 V11 promotion requires mapping {field!r}")
    return value


def canonical_s13_pixel_contract(document: Mapping[str, object]) -> dict[str, object]:
    """Return the V11 pixel contract with only allowed non-pixel fields removed.

    The function deliberately uses a closed field list.  In particular it
    retains runtime backend, motion, source selection, owner, geometry, seam,
    photometric, blend, C2E, truth, and quality declarations.
    """

    if not isinstance(document, Mapping):
        raise S13V11PromotionError("S013 V11 document must be a mapping")
    canonical: dict[str, Any] = copy.deepcopy(dict(document))
    for key in _TOP_LEVEL_NON_PIXEL_FIELDS:
        canonical.pop(key, None)

    components = _require_mapping(canonical.get("components"), field="components")
    component = dict(
        _require_mapping(components.get(_COMPONENT_NAME), field=f"components.{_COMPONENT_NAME}")
    )
    for key in _COMPONENT_NON_PIXEL_FIELDS:
        component.pop(key, None)
    runtime = dict(_require_mapping(component.get("runtime"), field="component.runtime"))
    for key in _RUNTIME_NON_PIXEL_FIELDS:
        runtime.pop(key, None)
    if runtime:
        component["runtime"] = runtime
    else:
        component.pop("runtime", None)
    canonical_components = dict(components)
    canonical_components[_COMPONENT_NAME] = component
    canonical["components"] = canonical_components
    return canonical


def s13_pixel_contract_sha256(document: Mapping[str, object]) -> str:
    """Hash the JSON-canonical form of :func:`canonical_s13_pixel_contract`."""

    try:
        encoded = json.dumps(
            canonical_s13_pixel_contract(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise S13V11PromotionError("S013 V11 pixel contract is not canonicalizable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _verify_exact_identity(document: Mapping[str, object], *, role: str) -> None:
    if document.get("role") != role:
        raise S13V11PromotionError(f"S013 V11 {role} role is not exact")
    if document.get("algorithm_id") != S13_VISUAL_CONTINUITY_ALGORITHM_ID:
        raise S13V11PromotionError("S013 V11 algorithm_id is not exact")
    if document.get("implementation_id") != S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID:
        raise S13V11PromotionError("S013 V11 implementation_id is not exact")
    if document.get("allow_baseline_fallback") is not False:
        raise S13V11PromotionError("S013 V11 forbids baseline fallback")


def verify_s13_v11_promotion(candidate: Path, production: Path) -> None:
    """Fail unless ``production`` is a lifecycle-only promotion of exact V11."""

    candidate_document = load_algorithm_config(candidate)
    production_document = load_algorithm_config(production)
    _verify_exact_identity(candidate_document, role="candidate")
    _verify_exact_identity(production_document, role="production")
    if candidate_document.get("config_schema") != "gemini305-video-candidate/v1":
        raise S13V11PromotionError("S013 V11 candidate schema is not exact")
    if production_document.get("config_schema") != "gemini305-video-algorithm/v1":
        raise S13V11PromotionError("S013 V11 production schema is not exact")
    if candidate_document.get("candidate_id") != S13_VISUAL_CONTINUITY_ALGORITHM_ID:
        raise S13V11PromotionError("S013 V11 candidate_id is not exact")
    if candidate_document.get("config_sha256") != S13_V11_CANDIDATE_CONFIG_SHA256:
        raise S13V11PromotionError("S013 V11 candidate config_sha256 changed")
    if canonical_config_sha256(candidate_document) != S13_V11_CANDIDATE_CONFIG_SHA256:
        raise S13V11PromotionError("S013 V11 candidate content changed")

    candidate_component = _require_mapping(
        _require_mapping(candidate_document.get("components"), field="candidate.components").get(
            _COMPONENT_NAME
        ),
        field=f"candidate.components.{_COMPONENT_NAME}",
    )
    production_component = _require_mapping(
        _require_mapping(
            production_document.get("components"), field="production.components"
        ).get(_COMPONENT_NAME),
        field=f"production.components.{_COMPONENT_NAME}",
    )
    if any(
        candidate_component.get(key) is not expected
        for key, expected in (
            ("diagnostic_only", True),
            ("production_eligible", False),
            ("production_lock_eligible", False),
        )
    ):
        raise S13V11PromotionError("S013 V11 candidate lifecycle flags changed")
    if any(
        production_component.get(key) is not expected
        for key, expected in (
            ("diagnostic_only", False),
            ("production_eligible", True),
            ("production_lock_eligible", True),
        )
    ):
        raise S13V11PromotionError("S013 V11 production lifecycle flags are not eligible")
    candidate_output = _require_mapping(candidate_component.get("output"), field="candidate.output")
    production_output = _require_mapping(
        production_component.get("output"), field="production.output"
    )
    if candidate_output.get("write_production_delivery") is not False:
        raise S13V11PromotionError("S013 V11 candidate publication flag changed")
    if production_output.get("write_production_delivery") is not True:
        raise S13V11PromotionError("S013 V11 production delivery must be enabled")

    promoted_from = _require_mapping(
        production_document.get("promoted_from"), field="promoted_from"
    )
    expected_provenance = {
        "candidate_path": S13_V11_CANDIDATE_RELATIVE_PATH,
        "candidate_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        "candidate_config_sha256": S13_V11_CANDIDATE_CONFIG_SHA256,
        "candidate_source_commit": S13_V11_CANDIDATE_SOURCE_COMMIT,
    }
    if dict(promoted_from) != expected_provenance:
        raise S13V11PromotionError("S013 V11 promoted_from provenance is not exact")
    source_commit = production_document.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise S13V11PromotionError("S013 V11 production source_commit must be a Git SHA")
    try:
        int(source_commit, 16)
    except ValueError as exc:
        raise S13V11PromotionError(
            "S013 V11 production source_commit must be a Git SHA"
        ) from exc

    candidate_contract = canonical_s13_pixel_contract(candidate_document)
    production_contract = canonical_s13_pixel_contract(production_document)
    if candidate_contract != production_contract:
        raise S13V11PromotionError("S013 V11 production pixel contract differs from candidate")


__all__ = [
    "S13_V11_CANDIDATE_CONFIG_SHA256",
    "S13_V11_CANDIDATE_RELATIVE_PATH",
    "S13_V11_CANDIDATE_SOURCE_COMMIT",
    "S13V11PromotionError",
    "canonical_s13_pixel_contract",
    "s13_pixel_contract_sha256",
    "verify_s13_v11_promotion",
]
