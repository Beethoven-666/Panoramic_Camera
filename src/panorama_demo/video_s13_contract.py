"""Fail-closed contract for the isolated S1.3 output-first candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .video_algorithm import load_algorithm_config
from .video_s13_m61_config import (
    M61_ALGORITHM_ID,
    M61_CONTRACT_SCHEMA,
    M61_IMPLEMENTATION_ID,
)


S13_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v3"
S13_IMPLEMENTATION_ID = "s013_output_first_progressive_dense_central_slit_preview"
S13_CONTRACT_SCHEMA = "gemini305-video-s13-output-first/v3"


@dataclass(frozen=True)
class S13Config:
    path: Path
    document: Mapping[str, Any]
    component: Mapping[str, Any]

    @property
    def analysis_width_px(self) -> int:
        return int(self.component["motion"]["analysis_width_px"])

    @property
    def normal_target_advance_px(self) -> float:
        return float(self.component["schedule"]["normal_target_advance_px"])

    @property
    def risky_target_advance_px(self) -> float:
        return float(self.component["schedule"]["risky_target_advance_px"])


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"S1.3 {name} must be a mapping")
    return value


def _walk_keys(value: object, prefix: str = "") -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.append(name.lower())
            keys.extend(_walk_keys(child, name))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child, prefix))
    return tuple(keys)


def validate_s13_document(document: Mapping[str, Any], *, path: Path) -> S13Config:
    if document.get("config_schema") != "gemini305-video-candidate/v1":
        raise ValueError("S1.3 requires gemini305-video-candidate/v1")
    if document.get("role") != "candidate":
        raise ValueError("S1.3 is candidate-only")
    if document.get("candidate_id") != S13_ALGORITHM_ID or document.get("algorithm_id") != S13_ALGORITHM_ID:
        raise ValueError("S1.3 candidate identity is not exact")
    if document.get("implementation_id") != S13_IMPLEMENTATION_ID:
        raise ValueError("S1.3 implementation identity is not exact")
    if document.get("allow_baseline_fallback") is not False:
        raise ValueError("S1.3 forbids baseline fallback")
    components = _mapping(document.get("components"), "components")
    component = _mapping(
        components.get("s013_output_first_progressive_dense_central_slit"),
        "component",
    )
    if component.get("contract_schema") != S13_CONTRACT_SCHEMA:
        raise ValueError("S1.3 contract schema is invalid")
    if (
        component.get("diagnostic_only") is not True
        or component.get("production_eligible") is not False
        or component.get("production_lock_eligible") is not False
    ):
        raise ValueError("S1.3 must remain diagnostic-only and production-ineligible")
    truth = _mapping(component.get("truth"), "truth")
    required_truth = {
        "require_real_rgb_source": True,
        "allow_interpolated_pose": False,
        "allow_extrapolated_pose": False,
        "allow_virtual_rgb_source": False,
        "allow_foreign_fill": False,
        "allow_synthetic_color_fill": False,
    }
    for key, expected in required_truth.items():
        if truth.get(key) is not expected:
            raise ValueError(f"S1.3 truth.{key} is fixed at {expected!r}")
    base = _mapping(component.get("base"), "base")
    if base.get("publish_before_optimization") is not True or int(base.get("transition_width_px", -1)) != 0:
        raise ValueError("S1.3 P0 must be owner-only and published before optimization")
    if any(base.get(key) is not False for key in ("color_gain_enabled", "local_warp_enabled", "blend_enabled")):
        raise ValueError("S1.3 P0 forbids gain, warp, and blend")
    depth = _mapping(component.get("depth_assist"), "depth_assist")
    if depth.get("enabled") is not False or depth.get("allow_color_generation") is not False:
        raise ValueError("S1.3 M0-M2 forbids depth assist and depth color")
    motion = _mapping(component.get("motion"), "motion")
    if motion.get("fixed_kmeans_forbidden") is not True or motion.get("direct_local_delta_used_as_gate") is not False:
        raise ValueError("S1.3 forbids fixed-K clustering and direct/local gates")
    for key in _walk_keys(document):
        compact = key.replace("-", "_")
        if "within_2px" in compact or (
            "direct" in compact and "local" in compact and "maximum" in compact and "difference" in compact
        ):
            raise ValueError("S1.3 configuration contains a forbidden 2 px direct/local gate")
    output = _mapping(component.get("output"), "output")
    if output.get("write_production_delivery") is not False:
        raise ValueError("S1.3 cannot write production delivery")
    forward = _mapping(component.get("forward_pipeline"), "forward_pipeline")
    if list(forward.get("stage_order", ())) != ["P0", "P1", "P2", "P3"]:
        raise ValueError("S1.3 M6 stage order must stop at P3")
    if (
        forward.get("default_stop_after") != "P3"
        or forward.get("forbid_parent_reselection") is not True
        or forward.get("current_preview_runtime_authority") is not False
    ):
        raise ValueError("S1.3 M6 forward pointer contract is invalid")
    replay = _mapping(component.get("p2_replay"), "p2_replay")
    if (
        replay.get("enabled") is not True
        or replay.get("completion_schema") != "gemini305-video-s13-p2-completion/v3"
        or int(replay.get("maximum_secondary_corridor_width_px", -1)) != 8
        or replay.get("require_transaction_hash_match") is not True
    ):
        raise ValueError("S1.3 M6 P2 replay contract is invalid")
    photometric = _mapping(component.get("photometric"), "photometric")
    if list(photometric.get("model_candidates", ())) != [
        "identity", "scalar_luminance_gain", "rgb_diagonal_gain", "bounded_rgb_gain_bias"
    ]:
        raise ValueError("S1.3 M6 photometric candidates must preserve Q0-Q3 order")
    if (
        photometric.get("color_domain") != "linear_srgb"
        or photometric.get("safe_background_only") is not True
        or photometric.get("train_heldout_split") is not True
        or photometric.get("low_frequency_luminance_field") is not False
        or photometric.get("failure_policy") != "identity"
    ):
        raise ValueError("S1.3 M6 photometric safety contract is invalid")
    blend = _mapping(component.get("blend"), "blend")
    if (
        blend.get("enabled") is not True
        or blend.get("safe_background_only") is not True
        or int(blend.get("maximum_total_width_px", -1)) != 8
        or int(blend.get("maximum_levels", -1)) != 2
        or int(blend.get("maximum_color_contributors_per_pixel", -1)) != 2
        or float(blend.get("protected_structure_weight", -1)) != 0.0
        or blend.get("failure_policy") != "owner_only"
    ):
        raise ValueError("S1.3 M6 blend safety contract is invalid")
    return S13Config(path=path, document=document, component=component)


def load_s13_config(path: str | Path) -> S13Config:
    resolved = Path(path).expanduser().resolve()
    return validate_s13_document(load_algorithm_config(resolved), path=resolved)


def is_s13_identity(*, algorithm_id: str, implementation_id: str, role: str) -> bool:
    return role == "candidate" and algorithm_id == S13_ALGORITHM_ID and implementation_id == S13_IMPLEMENTATION_ID


def is_s13_m61_identity(*, algorithm_id: str, implementation_id: str, role: str) -> bool:
    """Recognize M6.1 without weakening the historical v3 identity."""

    return (
        role == "candidate"
        and algorithm_id == M61_ALGORITHM_ID
        and implementation_id == M61_IMPLEMENTATION_ID
    )


__all__ = [
    "S13_ALGORITHM_ID",
    "S13_CONTRACT_SCHEMA",
    "S13_IMPLEMENTATION_ID",
    "M61_ALGORITHM_ID",
    "M61_CONTRACT_SCHEMA",
    "M61_IMPLEMENTATION_ID",
    "S13Config",
    "is_s13_identity",
    "is_s13_m61_identity",
    "load_s13_config",
    "validate_s13_document",
]
