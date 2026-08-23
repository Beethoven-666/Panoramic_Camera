from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_contract import (
    S13_FIRST_ROUND_PERF_ALGORITHM_ID,
    S13_FIRST_ROUND_PERF_IMPLEMENTATION_ID,
    S13_ALGORITHM_ID,
    S13_FORMAL_M6_ALGORITHM_ID,
    S13_FORMAL_M6_IMPLEMENTATION_ID,
    S13_M62_EFFECTIVE_ALGORITHM_ID,
    S13_M62_EFFECTIVE_IMPLEMENTATION_ID,
    claims_s13_document,
    is_s13_identity,
    load_s13_config,
    validate_s13_document,
)
from panorama_demo.video_s13_motion import descriptive_delta_risk


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v3.yaml"
FORMAL_M6_CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4.yaml"
CUDA_CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_resident_v1.yaml"
CUDA_V2_CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_resident_v2.yaml"
CUDA_V3_CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_structural_equivalent_v3.yaml"
M62_EFFECTIVE_CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_m62_effective_v6.yaml"
FIRST_ROUND_PERF_CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_m63_first_round_perf_v8.yaml"


def test_first_round_performance_successor_is_distinct_and_hash_bound() -> None:
    config = load_s13_config(FIRST_ROUND_PERF_CONFIG)
    spec = build_algorithm_spec(FIRST_ROUND_PERF_CONFIG, expected_role="candidate")

    assert spec.algorithm_id == S13_FIRST_ROUND_PERF_ALGORITHM_ID
    assert spec.implementation_id == S13_FIRST_ROUND_PERF_IMPLEMENTATION_ID
    assert config.runtime_backend == "cupy_cuda_m63_robust_v7"
    assert config.document["parent_candidate_id"] == (
        "S013_output_first_progressive_dense_central_slit_v4_cuda_m63_robust_photometric_v7"
    )
    assert is_s13_identity(
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
        role=spec.role,
    )


def test_s13_config_and_sibling_manifest_are_isolated_and_hash_bound() -> None:
    config = load_s13_config(CONFIG)
    spec = build_algorithm_spec(CONFIG, expected_role="candidate")
    assert spec.algorithm_id == S13_ALGORITHM_ID
    assert spec.candidate_manifest_path == CONFIG.parent / "candidate_manifest.json"
    assert spec.allow_baseline_fallback is False
    assert config.component["base"]["transition_width_px"] == 0
    assert config.component["depth_assist"]["enabled"] is False
    assert config.component["output"]["write_production_delivery"] is False


def test_formal_m6_identity_adds_m61_to_the_full_chain_without_mutating_v3() -> None:
    config = load_s13_config(FORMAL_M6_CONFIG)
    spec = build_algorithm_spec(FORMAL_M6_CONFIG, expected_role="candidate")
    assert S13_ALGORITHM_ID == "S013_output_first_progressive_dense_central_slit_v3"
    assert spec.algorithm_id == S13_FORMAL_M6_ALGORITHM_ID
    assert spec.implementation_id == S13_FORMAL_M6_IMPLEMENTATION_ID
    assert spec.candidate_manifest_path == FORMAL_M6_CONFIG.parent / "candidate_manifest.json"
    assert spec.required_evidence_components == (
        "real_rgb_sources",
        "rgb_motion_telemetry",
        "s013_m61_threshold_approval",
    )
    assert spec.required_output_components == ("s013_m61_p3",)
    assert config.component["contract_schema"] == "gemini305-video-s13-output-first/v4"
    assert config.document["m61_bootstrap"]["blend"]["eligible_models"] == [
        "B0_owner_only",
        "B1_narrow_feather_2px",
    ]
    assert is_s13_identity(
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
        role=spec.role,
    )


def test_cuda_successor_has_distinct_registered_runtime_backend() -> None:
    config = load_s13_config(CUDA_CONFIG)
    assert config.runtime_backend == "cupy_cuda_resident"
    assert config.document["candidate_id"] != load_s13_config(FORMAL_M6_CONFIG).document["candidate_id"]


def test_cuda_v2_successor_is_independently_registered_and_keeps_fast_contract() -> None:
    config = load_s13_config(CUDA_V2_CONFIG)
    spec = build_algorithm_spec(CUDA_V2_CONFIG, expected_role="candidate")
    assert config.runtime_backend == "cupy_cuda_resident_v2"
    assert spec.algorithm_id == "S013_output_first_progressive_dense_central_slit_v4_cuda_resident_v2"
    assert config.document["parent_candidate_id"] == (
        "S013_output_first_progressive_dense_central_slit_v4_cuda_resident_v1"
    )
    assert config.component["runtime"]["stage_writer_queue_size"] == 4


def test_cuda_v3_isolated_structural_equivalence_identity_is_hash_bound() -> None:
    config = load_s13_config(CUDA_V3_CONFIG)
    spec = build_algorithm_spec(CUDA_V3_CONFIG, expected_role="candidate")
    assert config.runtime_backend == "cupy_cuda_structural_equivalent_v3"
    assert spec.algorithm_id.endswith("structural_equivalent_v3")
    assert config.document["cuda_structural_equivalence"]["probes"]["ambiguous_action"] == "full_reference_fallback"


def test_m62_effective_successor_has_single_pass_identity_and_safe_component_model() -> None:
    config = load_s13_config(M62_EFFECTIVE_CONFIG)
    spec = build_algorithm_spec(M62_EFFECTIVE_CONFIG, expected_role="candidate")
    component = config.component

    assert spec.algorithm_id == S13_M62_EFFECTIVE_ALGORITHM_ID
    assert spec.implementation_id == S13_M62_EFFECTIVE_IMPLEMENTATION_ID
    assert config.runtime_backend == "cupy_cuda_m62_effective_v6"
    assert config.m62_effective is True
    assert config.document["m62_execution"]["mode"] == "candidate_single_pass"
    assert config.document["m62_execution"]["available_modes"] == [
        "reference",
        "shadow_audit",
        "candidate_single_pass",
        "parity_test",
    ]
    assert component["photometric_evidence"]["tier_b"]["blend_eligible"] is False
    assert component["photometric_evidence"]["bridge"]["photometric_relation_only"] is True
    assert component["photometric_component_model"]["model"] == (
        "Q4c_component_boundary_anchored_scalar_gain"
    )
    assert component["photometric_component_model"]["authority"] == "shadow"
    assert component["repair"]["q4_enabled"] is False
    assert component["output"]["write_production_delivery"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("m62_execution", "mode"), "parity_test", "execution"),
        (("m62_execution", "gpu_shadow_every_run"), True, "execution"),
        (
            (
                "m61_bootstrap",
                "selection",
                "minimum_actionable_macro_p95_benefit_fraction",
            ),
            0.0,
            "quality selection",
        ),
        (
            (
                "components",
                "s013_output_first_progressive_dense_central_slit",
                "photometric_evidence",
                "tier_b",
                "blend_eligible",
            ),
            True,
            "evidence",
        ),
        (
            (
                "components",
                "s013_output_first_progressive_dense_central_slit",
                "photometric_component_model",
                "model",
            ),
            "Q4_repair",
            "component model",
        ),
        (
            (
                "components",
                "s013_output_first_progressive_dense_central_slit",
                "repair",
                "q4_enabled",
            ),
            True,
            "repair",
        ),
    ],
)
def test_m62_effective_contract_rejects_execution_or_q4_scope_drift(
    path: tuple[str, ...], value: object, message: str
) -> None:
    document = yaml.safe_load(M62_EFFECTIVE_CONFIG.read_text(encoding="utf-8"))
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        validate_s13_document(document, path=M62_EFFECTIVE_CONFIG)


def test_malformed_formal_m6_claim_is_recognized_and_rejected() -> None:
    document = yaml.safe_load(FORMAL_M6_CONFIG.read_text(encoding="utf-8"))
    document["implementation_id"] = "wrong_implementation"
    assert claims_s13_document(document)
    with pytest.raises(ValueError, match="identity"):
        validate_s13_document(document, path=FORMAL_M6_CONFIG)

    document = yaml.safe_load(FORMAL_M6_CONFIG.read_text(encoding="utf-8"))
    document.pop("m61_bootstrap")
    with pytest.raises(ValueError, match="bootstrap"):
        validate_s13_document(document, path=FORMAL_M6_CONFIG)


def test_s13_does_not_modify_shared_candidate_manifest() -> None:
    shared = ROOT / "configs/video_candidates/candidate_manifest.json"
    assert hashlib.sha256(shared.read_bytes()).hexdigest() == "0c1460236de1a6afb11fff34df0ebb18bcea237011c6c0d56d24b461363160b6"
    baseline = ROOT / "configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json"
    assert hashlib.sha256(baseline.read_bytes()).hexdigest() == "0010fb37f67cd4e390e67e3d6cf6192eec3adebd98ceecfafc223b23056e89f9"
    assert not (ROOT / "configs/video_algorithms/production.lock.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["components"]["s013_output_first_progressive_dense_central_slit"]["truth"].__setitem__("allow_interpolated_pose", True),
        lambda d: d["components"]["s013_output_first_progressive_dense_central_slit"]["base"].__setitem__("transition_width_px", 2),
        lambda d: d["components"]["s013_output_first_progressive_dense_central_slit"].__setitem__("maximum_direct_local_path_difference_px", 2.0),
        lambda d: d["components"]["s013_output_first_progressive_dense_central_slit"]["output"].__setitem__("write_production_delivery", True),
    ],
)
def test_s13_contract_rejects_forbidden_behavior(mutation) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    modified = copy.deepcopy(document)
    mutation(modified)
    with pytest.raises(ValueError):
        validate_s13_document(modified, path=CONFIG)


def test_direct_local_delta_is_telemetry_for_slow_known_value() -> None:
    audit = descriptive_delta_risk(7.43485)
    assert audit == {"direct_local_delta_px": 7.43485, "risk": True, "structural_gate": False}
