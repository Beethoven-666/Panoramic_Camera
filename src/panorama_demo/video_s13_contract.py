"""Fail-closed contract for the isolated S1.3 output-first candidate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .video_algorithm import load_algorithm_config
from .video_s13_m61_config import (
    FORMAL_M6_ALGORITHM_ID,
    FORMAL_M6_IMPLEMENTATION_ID,
    M61_ALGORITHM_ID,
    M61_CONTRACT_SCHEMA,
    M61_IMPLEMENTATION_ID,
    M61_P2_COMPLETION_SCHEMA,
)


S13_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v3"
S13_IMPLEMENTATION_ID = "s013_output_first_progressive_dense_central_slit_preview"
S13_CONTRACT_SCHEMA = "gemini305-video-s13-output-first/v3"
S13_FORMAL_M6_ALGORITHM_ID = FORMAL_M6_ALGORITHM_ID
S13_FORMAL_M6_IMPLEMENTATION_ID = FORMAL_M6_IMPLEMENTATION_ID
S13_M51_R2_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v5"
S13_M51_R2_IMPLEMENTATION_ID = "s013_output_first_progressive_dense_central_slit_m51_r2"
S13_M51_R2_CONTRACT_SCHEMA = "gemini305-video-s13-output-first/v5"
S13_M51_R2_P2_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v5"
S13_M51_R3_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v6"
S13_M51_R3_IMPLEMENTATION_ID = (
    "s013_output_first_progressive_dense_central_slit_m51_r3_component_local_ambiguity"
)
S13_M51_R3_CONTRACT_SCHEMA = "gemini305-video-s13-output-first/v6"
S13_M51_R3_P2_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v6"

_S13_COMPONENT_NAME = "s013_output_first_progressive_dense_central_slit"


@dataclass(frozen=True)
class S13IdentityDescriptor:
    contract_schema: str
    p2_completion_schema: str
    p2_only: bool
    requires_m61_bootstrap: bool
    m51_r2_enabled: bool
    m51_r3_enabled: bool
    m6_eligible: bool
    required_output_component: str
    default_stop_after: str
    stage_order: tuple[str, ...]


_S13_IDENTITY_CONTRACTS = {
    (S13_ALGORITHM_ID, S13_IMPLEMENTATION_ID): S13IdentityDescriptor(
        S13_CONTRACT_SCHEMA, "gemini305-video-s13-p2-completion/v3", False,
        False, False, False, True, "s013_p0_owner_only", "P3", ("P0", "P1", "P2", "P3"),
    ),
    (S13_FORMAL_M6_ALGORITHM_ID, S13_FORMAL_M6_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3", "P4"),
    ),
    (S13_M51_R2_ALGORITHM_ID, S13_M51_R2_IMPLEMENTATION_ID): S13IdentityDescriptor(
        S13_M51_R2_CONTRACT_SCHEMA, S13_M51_R2_P2_COMPLETION_SCHEMA, True,
        False, True, False, False, "s013_p2_v5", "P2", ("P0", "P1", "P2"),
    ),
    (S13_M51_R3_ALGORITHM_ID, S13_M51_R3_IMPLEMENTATION_ID): S13IdentityDescriptor(
        S13_M51_R3_CONTRACT_SCHEMA, S13_M51_R3_P2_COMPLETION_SCHEMA, True,
        False, True, True, False, "s013_p2_v6", "P2", ("P0", "P1", "P2"),
    ),
}


@dataclass(frozen=True)
class S13Config:
    path: Path
    document: Mapping[str, Any]
    component: Mapping[str, Any]
    identity: S13IdentityDescriptor

    @property
    def p2_completion_schema(self) -> str:
        return self.identity.p2_completion_schema

    @property
    def p2_only(self) -> bool:
        return self.identity.p2_only

    @property
    def requires_m61_bootstrap(self) -> bool:
        return self.identity.requires_m61_bootstrap

    @property
    def m51_r2_enabled(self) -> bool:
        return self.identity.m51_r2_enabled

    @property
    def m51_r3_enabled(self) -> bool:
        return self.identity.m51_r3_enabled

    @property
    def m6_eligible(self) -> bool:
        return self.identity.m6_eligible

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
    algorithm_id = document.get("algorithm_id")
    if document.get("candidate_id") != algorithm_id:
        raise ValueError("S1.3 candidate identity is not exact")
    identity_contract = _S13_IDENTITY_CONTRACTS.get(
        (algorithm_id, document.get("implementation_id"))
    )
    if identity_contract is None:
        raise ValueError("S1.3 implementation identity is not exact")
    contract_schema = identity_contract.contract_schema
    p2_completion_schema = identity_contract.p2_completion_schema
    requires_m61_bootstrap = identity_contract.requires_m61_bootstrap
    if document.get("allow_baseline_fallback") is not False:
        raise ValueError("S1.3 forbids baseline fallback")
    components = _mapping(document.get("components"), "components")
    component = _mapping(
        components.get(_S13_COMPONENT_NAME),
        "component",
    )
    if component.get("contract_schema") != contract_schema:
        raise ValueError("S1.3 contract schema is invalid")
    if requires_m61_bootstrap:
        _mapping(document.get("m61_bootstrap"), "M6.1 bootstrap")
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
    expected_stage_order = list(identity_contract.stage_order)
    if list(forward.get("stage_order", ())) != expected_stage_order:
        raise ValueError("S1.3 forward stage order is invalid")
    if (
        forward.get("default_stop_after") != identity_contract.default_stop_after
        or forward.get("forbid_parent_reselection") is not True
        or forward.get("current_preview_runtime_authority") is not False
    ):
        raise ValueError("S1.3 forward pointer contract is invalid")
    if requires_m61_bootstrap and (
        forward.get("allow_resume_from_sealed_stage") is not True
        or forward.get("p4_requires_certified_handoff_and_actual_winner") is not True
    ):
        raise ValueError("S1.3 M7 must require a sealed handoff and an actual winner")
    replay = _mapping(component.get("p2_replay"), "p2_replay")
    if (
        replay.get("enabled") is not True
        or replay.get("completion_schema") != p2_completion_schema
        or int(replay.get("maximum_secondary_corridor_width_px", -1)) != 8
        or replay.get("require_transaction_hash_match") is not True
    ):
        raise ValueError("S1.3 M6 P2 replay contract is invalid")
    if identity_contract.p2_only:
        if "m61_bootstrap" in document:
            raise ValueError("S1.3 P2-only identity forbids an M6.1 bootstrap")
        if document.get("required_output_components") != [
            identity_contract.required_output_component
        ]:
            raise ValueError("S1.3 P2-only identity has an invalid required output")
        m51_r2 = _mapping(component.get("m51_r2"), "M5.1-r2 config")
        if m51_r2.get("enabled") is not True:
            raise ValueError("S1.3 P2-only successor requires M5.1-r2 to be enabled")
        if identity_contract.m51_r3_enabled:
            m51_r3 = _mapping(component.get("m51_r3"), "M5.1-r3 config")
            if dict(m51_r3) != {
                "enabled": True,
                "reassessment_pair_indices": [71],
                "micro_rescue_enabled": False,
                "c2e_enabled": False,
            }:
                raise ValueError("S1.3 v6 M5.1-r3 reassessment contract is invalid")
        serialized = repr(document).lower()
        if "quality_thresholds_m61" in serialized or "threshold_approval" in serialized:
            raise ValueError("S1.3 P2-only identity forbids old M6 threshold/approval bindings")
    else:
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
        expected_blend_width = 2 if requires_m61_bootstrap else 8
        expected_blend_levels = 1 if requires_m61_bootstrap else 2
        if (
            blend.get("enabled") is not True
            or blend.get("safe_background_only") is not True
            or int(blend.get("maximum_total_width_px", -1)) != expected_blend_width
            or int(blend.get("maximum_levels", -1)) != expected_blend_levels
            or int(blend.get("maximum_color_contributors_per_pixel", -1)) != 2
            or float(blend.get("protected_structure_weight", -1)) != 0.0
            or blend.get("failure_policy") != "owner_only"
        ):
            raise ValueError("S1.3 M6 blend safety contract is invalid")
    if requires_m61_bootstrap:
        if (
            list(blend.get("model_candidates", ()))
            != ["B0_owner_only", "B1_narrow_feather_2px"]
            or list(blend.get("ineligible_models", ()))
            != ["B2_safe_masked_multiband", "B3", "B4"]
            or blend.get("adaptive_levels") is not False
        ):
            raise ValueError("S1.3 formal M6 permits only B0 or total-width-2 B1")
        repair = _mapping(component.get("repair"), "repair")
        if (
            repair.get("enabled") is not True
            or repair.get("authorization_schema") != "gemini305-video-s13-m7-handoff/v2"
            or list(repair.get("allowed_stage_exits", ()))
            != ["completed_target", "completed_manual_forward"]
            or repair.get("manual_forward_authorization_schema")
            != "gemini305-video-s13-m7-manual-forward-authorization/v1"
            or repair.get("manual_forward_requires_explicit_user_authorization") is not True
            or repair.get("manual_forward_preserve_automatic_quality_records") is not True
            or repair.get("manual_forward_core_only") is not True
            or list(repair.get("allowed_component_classes", ()))
            != ["protected", "geometry", "seam", "owner"]
            or list(repair.get("core_candidates", ())) != [
                "R0_keep_p3", "R1_b1_2px_to_b0_owner_only",
                "R2_downgrade_seam_reestimate_geometry",
                "R3_downgrade_geometry_reselect_seam",
            ]
            or repair.get("no_handoff_or_no_winner_policy") != "no_p4_keep_p3"
        ):
            raise ValueError("S1.3 M7 authorization/Core contract is invalid")
        required_true = (
            "completed_target_requires_m7_handoff_eligible",
            "systemic_photometric_component_forbidden",
            "adjacent_or_merged_seam_component_forbidden",
            "q_parameters_frozen", "thresholds_frozen", "source_set_frozen",
        )
        required_false = (
            "q4_enabled", "low_frequency_field_enabled", "expanded_feather_enabled",
            "multiband_enabled", "cross_pair_joint_repair_enabled", "frame_insertion_enabled",
            "graphcut_enabled", "depth_risk_enabled", "bounded_mesh_enabled",
        )
        if any(repair.get(key) is not True for key in required_true) or any(
            repair.get(key) is not False for key in required_false
        ):
            raise ValueError("S1.3 M7 frozen/disabled safety contract is invalid")
    return S13Config(
        path=path, document=document, component=component, identity=identity_contract
    )


def load_s13_config(path: str | Path) -> S13Config:
    resolved = Path(path).expanduser().resolve()
    return validate_s13_document(load_algorithm_config(resolved), path=resolved)


def is_s13_identity(*, algorithm_id: str, implementation_id: str, role: str) -> bool:
    return role == "candidate" and (algorithm_id, implementation_id) in _S13_IDENTITY_CONTRACTS


def claims_s13_document(document: Mapping[str, Any]) -> bool:
    """Recognize exact or malformed full-chain S1.3 claims before dispatch.

    A config which names either supported identity half, schema, or component
    must pass the strict S1.3 validator; it may not fall through to a legacy
    candidate renderer after mutating the other half of its identity.
    """

    algorithms = {identity[0] for identity in _S13_IDENTITY_CONTRACTS}
    implementations = {identity[1] for identity in _S13_IDENTITY_CONTRACTS}
    values = {
        document.get("candidate_id"),
        document.get("algorithm_id"),
        document.get("implementation_id"),
    }
    if values & (algorithms | implementations):
        return True
    components = document.get("components")
    if isinstance(components, Mapping) and _S13_COMPONENT_NAME in components:
        return True
    component = components.get(_S13_COMPONENT_NAME) if isinstance(components, Mapping) else None
    return isinstance(component, Mapping) and component.get("contract_schema") in {
        contract.contract_schema for contract in _S13_IDENTITY_CONTRACTS.values()
    }


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
    "S13_FORMAL_M6_ALGORITHM_ID",
    "S13_FORMAL_M6_IMPLEMENTATION_ID",
    "S13_M51_R2_ALGORITHM_ID",
    "S13_M51_R2_IMPLEMENTATION_ID",
    "S13_M51_R2_CONTRACT_SCHEMA",
    "S13_M51_R2_P2_COMPLETION_SCHEMA",
    "S13_M51_R3_ALGORITHM_ID",
    "S13_M51_R3_IMPLEMENTATION_ID",
    "S13_M51_R3_CONTRACT_SCHEMA",
    "S13_M51_R3_P2_COMPLETION_SCHEMA",
    "M61_ALGORITHM_ID",
    "M61_CONTRACT_SCHEMA",
    "M61_IMPLEMENTATION_ID",
    "S13Config",
    "S13IdentityDescriptor",
    "claims_s13_document",
    "is_s13_identity",
    "is_s13_m61_identity",
    "load_s13_config",
    "validate_s13_document",
]
