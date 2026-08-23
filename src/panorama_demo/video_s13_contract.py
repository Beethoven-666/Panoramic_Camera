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
S13_M51_R4_ALGORITHM_ID = S13_M51_R3_ALGORITHM_ID
S13_M51_R4_IMPLEMENTATION_ID = (
    "s013_output_first_progressive_dense_central_slit_m51_r4_component_chain_c2e"
)
S13_M51_R4_CONTRACT_SCHEMA = "gemini305-video-s13-output-first/v6-r2"
S13_M51_R4_P2_COMPLETION_SCHEMA = "gemini305-video-s13-p2-completion/v6-r2"
S13_CUDA_RESIDENT_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v4_cuda_resident_v1"
S13_CUDA_RESIDENT_IMPLEMENTATION_ID = "s013_output_first_progressive_dense_central_slit_m61_cuda_resident_v1"
S13_CUDA_RESIDENT_V2_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v4_cuda_resident_v2"
S13_CUDA_RESIDENT_V2_IMPLEMENTATION_ID = "s013_output_first_progressive_dense_central_slit_m61_cuda_resident_v2"
S13_CUDA_STRUCTURAL_V3_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v4_cuda_structural_equivalent_v3"
S13_CUDA_STRUCTURAL_V3_IMPLEMENTATION_ID = "s013_m61_cuda_guarded_probe_structural_equivalent_v3"
S13_M62_ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v4_cuda_m62_cpu_equivalent_v5"
S13_M62_IMPLEMENTATION_ID = "s013_m62_cpu_oracle_gpu_staged_equivalence_v5"
S13_M62_EFFECTIVE_ALGORITHM_ID = (
    "S013_output_first_progressive_dense_central_slit_v4_cuda_m62_effective_v6"
)
S13_M62_EFFECTIVE_IMPLEMENTATION_ID = (
    "s013_m62_effective_photometric_single_pass_cuda_v6"
)
S13_M63_ALGORITHM_ID = (
    "S013_output_first_progressive_dense_central_slit_v4_cuda_m63_robust_photometric_v7"
)
S13_M63_IMPLEMENTATION_ID = "s013_m63_robust_centered_photometric_quality_cut_v7"
S13_FIRST_ROUND_PERF_ALGORITHM_ID = (
    "S013_output_first_progressive_dense_central_slit_v4_cuda_m63_first_round_perf_v8"
)
S13_FIRST_ROUND_PERF_IMPLEMENTATION_ID = (
    "s013_m63_preflight_rgb_handoff_v8"
)
S13_SECOND_ROUND_PERF_ALGORITHM_ID = (
    "S013_output_first_progressive_dense_central_slit_v4_cuda_m63_second_round_perf_v9"
)
S13_SECOND_ROUND_PERF_IMPLEMENTATION_ID = (
    "s013_m63_deferred_step4_m5_final_authority_base_atlas_v9"
)
S13_THIRD_ROUND_PERF_ALGORITHM_ID = (
    "S013_output_first_progressive_dense_central_slit_v4_cuda_m63_third_round_perf_v10"
)
S13_THIRD_ROUND_PERF_IMPLEMENTATION_ID = "s013_m63_compact_p0_map_v10"
S13_VISUAL_CONTINUITY_ALGORITHM_ID = (
    "S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11"
)
S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID = (
    "s013_m5_disjoint_seam_m63_pair_guard_v11"
)

_S13_COMPONENT_NAME = "s013_output_first_progressive_dense_central_slit"


@dataclass(frozen=True)
class S13IdentityDescriptor:
    contract_schema: str
    p2_completion_schema: str
    p2_only: bool
    requires_m61_bootstrap: bool
    m51_r2_enabled: bool
    m51_r3_enabled: bool
    m51_r4_enabled: bool
    m6_eligible: bool
    required_output_component: str
    default_stop_after: str
    stage_order: tuple[str, ...]
    runtime_backend: str = "numpy_reference"
    m62_equivalence: bool = False
    m62_effective: bool = False
    m63_robust: bool = False


_S13_IDENTITY_CONTRACTS = {
    (S13_ALGORITHM_ID, S13_IMPLEMENTATION_ID): S13IdentityDescriptor(
        S13_CONTRACT_SCHEMA, "gemini305-video-s13-p2-completion/v3", False,
        False, False, False, False, True, "s013_p0_owner_only", "P3", ("P0", "P1", "P2", "P3"),
    ),
    (S13_FORMAL_M6_ALGORITHM_ID, S13_FORMAL_M6_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3"),
    ),
    (S13_CUDA_RESIDENT_ALGORITHM_ID, S13_CUDA_RESIDENT_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3"),
        "cupy_cuda_resident",
    ),
    (S13_CUDA_RESIDENT_V2_ALGORITHM_ID, S13_CUDA_RESIDENT_V2_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3"),
        "cupy_cuda_resident_v2",
    ),
    (S13_CUDA_STRUCTURAL_V3_ALGORITHM_ID, S13_CUDA_STRUCTURAL_V3_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3"),
        "cupy_cuda_structural_equivalent_v3",
    ),
    (S13_M62_ALGORITHM_ID, S13_M62_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3"),
        "cupy_cuda_m62_cpu_equivalent_v5", True,
    ),
    (S13_M62_EFFECTIVE_ALGORITHM_ID, S13_M62_EFFECTIVE_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3"),
        "cupy_cuda_m62_effective_v6", True, True,
    ),
    (S13_M63_ALGORITHM_ID, S13_M63_IMPLEMENTATION_ID): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3", ("P0", "P1", "P2", "P3"),
        "cupy_cuda_m63_robust_v7", True, True, True,
    ),
    (
        S13_FIRST_ROUND_PERF_ALGORITHM_ID,
        S13_FIRST_ROUND_PERF_IMPLEMENTATION_ID,
    ): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3",
        ("P0", "P1", "P2", "P3"),
        "cupy_cuda_m63_robust_v7", True, True, True,
    ),
    (
        S13_SECOND_ROUND_PERF_ALGORITHM_ID,
        S13_SECOND_ROUND_PERF_IMPLEMENTATION_ID,
    ): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3",
        ("P0", "P1", "P2", "P3"),
        "cupy_cuda_m63_robust_v7", True, True, True,
    ),
    (
        S13_THIRD_ROUND_PERF_ALGORITHM_ID,
        S13_THIRD_ROUND_PERF_IMPLEMENTATION_ID,
    ): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3",
        ("P0", "P1", "P2", "P3"),
        "cupy_cuda_m63_robust_v7", True, True, True,
    ),
    (
        S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
    ): S13IdentityDescriptor(
        M61_CONTRACT_SCHEMA, M61_P2_COMPLETION_SCHEMA, False,
        True, False, False, False, True, "s013_m61_p3", "P3",
        ("P0", "P1", "P2", "P3"),
        "cupy_cuda_m63_robust_v7", True, True, True,
    ),
    (S13_M51_R2_ALGORITHM_ID, S13_M51_R2_IMPLEMENTATION_ID): S13IdentityDescriptor(
        S13_M51_R2_CONTRACT_SCHEMA, S13_M51_R2_P2_COMPLETION_SCHEMA, True,
        False, True, False, False, False, "s013_p2_v5", "P2", ("P0", "P1", "P2"),
    ),
    (S13_M51_R3_ALGORITHM_ID, S13_M51_R3_IMPLEMENTATION_ID): S13IdentityDescriptor(
        S13_M51_R3_CONTRACT_SCHEMA, S13_M51_R3_P2_COMPLETION_SCHEMA, True,
        False, True, True, False, False, "s013_p2_v6", "P2", ("P0", "P1", "P2"),
    ),
    (S13_M51_R4_ALGORITHM_ID, S13_M51_R4_IMPLEMENTATION_ID): S13IdentityDescriptor(
        S13_M51_R4_CONTRACT_SCHEMA, S13_M51_R4_P2_COMPLETION_SCHEMA, True,
        False, True, True, True, False, "s013_p2_v6", "P2", ("P0", "P1", "P2"),
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
    def m51_r4_enabled(self) -> bool:
        return self.identity.m51_r4_enabled

    @property
    def m6_eligible(self) -> bool:
        return self.identity.m6_eligible

    @property
    def runtime_backend(self) -> str:
        return self.identity.runtime_backend

    @property
    def m62_equivalence(self) -> bool:
        return self.identity.m62_equivalence

    @property
    def m62_effective(self) -> bool:
        return self.identity.m62_effective

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
    fast_formal = requires_m61_bootstrap
    compact_p0 = document.get("m5_compact_p0_map")
    if compact_p0 is not None:
        compact_p0 = _mapping(compact_p0, "M5 compact P0 map")
        if (
            compact_p0.get("enabled") is not True
            or compact_p0.get("mode") != "compact_exact_window"
            or compact_p0.get("full_reference_fallback") is not True
        ):
            raise ValueError("S1.3 compact P0 map contract is invalid")
    evidence_sampling = document.get("m62_evidence_sampling")
    if evidence_sampling is not None:
        evidence_sampling = _mapping(evidence_sampling, "M6.2 evidence sampling")
        if (
            evidence_sampling.get("mode") not in {"pair_reference", "source_packed_cv2"}
            or evidence_sampling.get("reference_mode_available") is not True
            or evidence_sampling.get("exact_sample_shadow_in_formal") is not False
        ):
            raise ValueError("S1.3 M6.2 evidence sampling contract is invalid")
    if document.get("allow_baseline_fallback") is not False:
        raise ValueError("S1.3 forbids baseline fallback")
    components = _mapping(document.get("components"), "components")
    component = _mapping(
        components.get(_S13_COMPONENT_NAME),
        "component",
    )
    declared_backend = component.get("runtime_backend", "numpy_reference")
    if declared_backend != identity_contract.runtime_backend:
        raise ValueError("S1.3 runtime backend does not match registered identity")
    if component.get("contract_schema") != contract_schema:
        raise ValueError("S1.3 contract schema is invalid")
    if requires_m61_bootstrap:
        _mapping(document.get("m61_bootstrap"), "M6.1 bootstrap")
    if identity_contract.runtime_backend == "cupy_cuda_structural_equivalent_v3":
        equivalence = _mapping(document.get("cuda_structural_equivalence"), "CUDA v3 equivalence")
        if equivalence.get("mode") != "structural_exact_rgb_tolerant":
            raise ValueError("S1.3 CUDA v3 requires structural-exact RGB-tolerant mode")
        probes = _mapping(equivalence.get("probes"), "CUDA v3 probes")
        if probes.get("mode") != "guarded_tolerant" or probes.get("ambiguous_action") != "full_reference_fallback":
            raise ValueError("S1.3 CUDA v3 probe fallback contract is invalid")
    if identity_contract.m62_equivalence:
        equivalence = _mapping(document.get("m62_equivalence"), "M6.2 equivalence")
        if equivalence.get("cpu_oracle_enabled") is not True:
            raise ValueError("S1.3 M6.2 requires a CPU oracle")
        publish = _mapping(equivalence.get("publish_mode"), "M6.2 publish mode")
        if publish.get("default") != "cpu_authoritative_hybrid":
            raise ValueError("S1.3 M6.2 must publish CPU-authoritative P3")
    if identity_contract.m62_effective:
        execution = _mapping(document.get("m62_execution"), "M6.2 execution")
        if (
            execution.get("mode") != "candidate_single_pass"
            or list(execution.get("available_modes", ()))
            != ["reference", "shadow_audit", "candidate_single_pass", "parity_test"]
            or execution.get("legacy_reference_comparison") is not False
            or execution.get("gpu_shadow_every_run") is not False
            or execution.get("audit_reference_mode_available") is not True
            or execution.get("q0_b0_direct_return") is not True
        ):
            raise ValueError("S1.3 effective M6.2 execution contract is invalid")
        effectiveness = _mapping(document.get("m62_effectiveness"), "M6.2 effectiveness")
        if any(
            effectiveness.get(key) is not True
            for key in (
                "require_explicit_status",
                "report_candidate_rejections",
                "report_pair_ineligibility",
                "report_p3_vs_p2",
                "report_macro_seam_metrics",
            )
        ):
            raise ValueError("S1.3 effective M6.2 reporting contract is invalid")
        if identity_contract.m63_robust:
            m63 = _mapping(document.get("m63_photometric"), "M6.3 photometric")
            if list(m63.get("candidates", ())) != [
                "Q0_identity",
                "Q1R_robust_centered_scalar_gain",
                "Q4c_quality_cut_component_scalar_gain",
                "QL_pair_tapered_scalar_gain",
                "QT_global_tone_gain",
            ]:
                raise ValueError("S1.3 M6.3 candidate order is invalid")
            if m63.get("q2_rgb_gain_enabled") is not False or m63.get("q3_gain_bias_enabled") is not False:
                raise ValueError("S1.3 M6.3 forbids Q2/Q3 authority")
            q1r = _mapping(m63.get("q1r"), "M6.3 Q1R")
            if (
                q1r.get("enabled") is not True
                or q1r.get("huber_irls") is not True
                or q1r.get("gauge") != "weighted_zero_mean_log_gain"
                or q1r.get("source_level_fallback_forbidden") is not True
                or float(q1r.get("minimum_gain", -1)) != 0.92
                or float(q1r.get("maximum_gain", -1)) != 1.08
            ):
                raise ValueError("S1.3 M6.3 Q1R contract is invalid")
            q4c = _mapping(m63.get("q4c"), "M6.3 Q4c")
            if (
                q4c.get("enabled") is not True
                or q4c.get("component_atomic") is not True
                or q4c.get("boundary_anchor_identity") is not True
                or q4c.get("cut_blend_model") != "B0_owner_only"
                or q4c.get("cut_guard_must_equal_p2") is not True
            ):
                raise ValueError("S1.3 M6.3 Q4c contract is invalid")
        m61_selection = _mapping(
            _mapping(document.get("m61_bootstrap"), "M6.1 bootstrap").get("selection"),
            "M6.1 selection",
        )
        if dict(m61_selection) != {
            "aggregate_p95_nonreg_absolute_tolerance_linear": 0.0005,
            "aggregate_p95_nonreg_relative_tolerance": 0.02,
            "minimum_actionable_macro_p95_benefit_fraction": 0.05,
            "minimum_aggregate_median_benefit_fraction": 0.01,
            "minimum_composite_benefit_fraction": 0.03,
            "require_macro_and_micro_nonregression": True,
            "selection_score_mde_linear": 0.0005,
            "worst_pair_p95_nonreg_absolute_tolerance_linear": 0.0005,
            "worst_pair_p95_nonreg_relative_tolerance": 0.02,
        }:
            raise ValueError("S1.3 effective M6.2 must retain the M6.1 quality selection gate")
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
    if fast_formal and forward.get("allow_resume_from_sealed_stage") is not False:
        raise ValueError("S1.3 fast formal pipeline forbids disk resume")
    replay = _mapping(component.get("p2_replay"), "p2_replay")
    if fast_formal:
        if (
            replay.get("enabled") is not False
            or replay.get("completion_schema") is not None
            or replay.get("require_transaction_hash_match") is not False
        ):
            raise ValueError("S1.3 fast formal pipeline forbids P2 disk replay")
    elif (
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
        if not identity_contract.m51_r4_enabled and "m51_r4" in component:
            raise ValueError("S1.3 pre-r4 identity forbids an M5.1-r4 configuration")
        if identity_contract.m51_r4_enabled:
            m51_r4 = _mapping(component.get("m51_r4"), "M5.1-r4 config")
            from .video_s13_m51_r2 import S13M51R4Config

            arguments = dict(m51_r2)
            arguments.update(dict(m51_r4))
            arguments.update(
                complete_reassessment_pair_indices=tuple(
                    int(value) for value in m51_r3["reassessment_pair_indices"]
                ),
                component_local_ambiguity_enabled=True,
            )
            try:
                S13M51R4Config(**arguments)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("S1.3 v6-r2 M5.1-r4 contract is invalid") from exc
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
        if identity_contract.m62_effective:
            evidence = _mapping(component.get("photometric_evidence"), "photometric evidence")
            tier_a = _mapping(evidence.get("tier_a"), "photometric Tier A")
            tier_b = _mapping(evidence.get("tier_b"), "photometric Tier B")
            bridge = _mapping(evidence.get("bridge"), "photometric bridge")
            if (
                tier_a.get("enabled") is not True
                or list(tier_a.get("uses", ())) != ["solve", "blend", "heldout"]
                or int(tier_a.get("minimum_train_samples", -1)) != 128
                or tier_b.get("enabled") is not True
                or list(tier_b.get("uses", ())) != ["solve"]
                or tier_b.get("blend_eligible") is not False
                or int(tier_b.get("minimum_combined_train_samples", -1)) != 192
                or bridge.get("enabled") is not True
                or int(bridge.get("source_index_gap", -1)) != 2
                or int(bridge.get("minimum_overlap_width_px", -1)) != 32
                or int(bridge.get("minimum_safe_samples", -1)) != 256
                or float(bridge.get("minimum_inlier_fraction", -1.0)) != 0.7
                or bridge.get("photometric_relation_only") is not True
                or bridge.get("failure_policy") != "omit_edge"
            ):
                raise ValueError("S1.3 effective M6.2 evidence contract is invalid")
            component_model = _mapping(
                component.get("photometric_component_model"),
                "photometric component model",
            )
            if (
                component_model.get("enabled") is not True
                or component_model.get("model")
                != "Q4c_component_boundary_anchored_scalar_gain"
                or component_model.get("authority") != "shadow"
                or component_model.get("scalar_luminance_only") is not True
                or float(component_model.get("bias", -1.0)) != 0.0
                or float(component_model.get("minimum_gain", -1.0)) != 0.94
                or float(component_model.get("maximum_gain", -1.0)) != 1.06
                or component_model.get("component_atomic") is not True
                or component_model.get("unsupported_cut_anchor_identity") is not True
                or component_model.get("unsupported_cut_blend_model") != "B0_owner_only"
                or int(component_model.get("cut_guard_width_px", -1)) != 12
            ):
                raise ValueError("S1.3 effective M6.2 component model contract is invalid")
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
        m6 = _mapping(component.get("m6"), "M6 config")
        auto_c2e = _mapping(m6.get("auto_c2e"), "M6 automatic C2E")
        if (
            repair.get("enabled") is not True
            or repair.get("automatic") is not True
            or list(repair.get("allowed_component_classes", ()))
            != ["protected", "geometry", "seam", "owner"]
            or list(repair.get("core_candidates", ())) != [
                "C0_keep_standard", "C1_force_owner_only",
                "C2_downgrade_seam", "C3_downgrade_geometry",
            ]
            or repair.get("no_handoff_or_no_winner_policy") != "keep_p3"
            or auto_c2e.get("enabled") is not True
            or auto_c2e.get("authorization_required") is not False
            or auto_c2e.get("policy") != "all_structurally_safe"
            or int(auto_c2e.get("full_resolution_render_count", -1)) != 1
        ):
            raise ValueError("S1.3 M6 automatic C2E contract is invalid")
        required_true = (
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
            raise ValueError("S1.3 M7 optional repair must remain frozen/disabled")
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
    "S13_M51_R4_ALGORITHM_ID",
    "S13_M51_R4_IMPLEMENTATION_ID",
    "S13_M51_R4_CONTRACT_SCHEMA",
    "S13_M51_R4_P2_COMPLETION_SCHEMA",
    "S13_CUDA_RESIDENT_ALGORITHM_ID",
    "S13_CUDA_RESIDENT_IMPLEMENTATION_ID",
    "S13_M62_ALGORITHM_ID",
    "S13_M62_IMPLEMENTATION_ID",
    "S13_M62_EFFECTIVE_ALGORITHM_ID",
    "S13_M62_EFFECTIVE_IMPLEMENTATION_ID",
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
