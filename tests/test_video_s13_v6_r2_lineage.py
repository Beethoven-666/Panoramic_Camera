from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from panorama_demo.video_algorithm import build_algorithm_spec, canonical_config_sha256
from panorama_demo.video_candidate_manifest import canonical_candidate_manifest_sha256
from panorama_demo.video_s13_contract import (
    S13_M51_R3_ALGORITHM_ID,
    S13_M51_R3_IMPLEMENTATION_ID,
    S13_M51_R4_CONTRACT_SCHEMA,
    S13_M51_R4_IMPLEMENTATION_ID,
    S13_M51_R4_P2_COMPLETION_SCHEMA,
    is_s13_identity,
    load_s13_config,
    validate_s13_document,
)
from panorama_demo.video_s13_m51_r2 import S13M51R4Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/video_candidates/s013"
V6_CONFIG = CONFIG_ROOT / "S013_output_first_progressive_dense_central_slit_v6.yaml"
V6_R2_CONFIG_SHA256 = "5ccaf952fe94d4c42626f3ec9e9555666c023a5ad654885ac4d9def2c96b3dde"
V6_R2_LOCAL_MANIFEST_SHA256 = (
    "df3116358f90320a69873d841e65593e0535b18b58adf34da7792e83da0e3d29"
)


def test_v6_r2_identity_and_local_manifest_are_canonically_bound() -> None:
    config = load_s13_config(V6_CONFIG)
    spec = build_algorithm_spec(V6_CONFIG, expected_role="candidate")
    manifest = json.loads((CONFIG_ROOT / "candidate_manifest.json").read_text(encoding="utf-8"))

    assert config.document["algorithm_id"] == S13_M51_R3_ALGORITHM_ID
    assert config.document["implementation_id"] == S13_M51_R4_IMPLEMENTATION_ID
    assert config.component["contract_schema"] == S13_M51_R4_CONTRACT_SCHEMA
    assert config.p2_completion_schema == S13_M51_R4_P2_COMPLETION_SCHEMA
    assert config.p2_only is True
    assert config.m51_r2_enabled is True
    assert config.m51_r3_enabled is True
    assert config.m51_r4_enabled is True
    assert config.m6_eligible is False
    assert config.component["m51_r4"]["application_policy"] == (
        "all_structurally_safe"
    )
    assert config.document["required_output_components"] == ["s013_p2_v6"]
    assert config.component["p2_replay"]["completion_schema"] == (
        S13_M51_R4_P2_COMPLETION_SCHEMA
    )
    assert config.document["config_sha256"] == canonical_config_sha256(config.document)
    assert config.document["config_sha256"] == V6_R2_CONFIG_SHA256
    assert manifest["candidates"][S13_M51_R3_ALGORITHM_ID]["config_sha256"] == (
        config.document["config_sha256"]
    )
    assert manifest["manifest_sha256"] == canonical_candidate_manifest_sha256(manifest)
    assert manifest["manifest_sha256"] == V6_R2_LOCAL_MANIFEST_SHA256
    assert spec.config_sha256 == config.document["config_sha256"]
    assert spec.candidate_manifest_sha256 == manifest["manifest_sha256"]


def test_v6_r2_keeps_the_v6_r1_identity_registered() -> None:
    assert is_s13_identity(
        algorithm_id=S13_M51_R3_ALGORITHM_ID,
        implementation_id=S13_M51_R3_IMPLEMENTATION_ID,
        role="candidate",
    )


def test_v6_r1_identity_cannot_silently_carry_r4_authority() -> None:
    document = yaml.safe_load(V6_CONFIG.read_text(encoding="utf-8"))
    component = document["components"]["s013_output_first_progressive_dense_central_slit"]
    document["implementation_id"] = S13_M51_R3_IMPLEMENTATION_ID
    component["contract_schema"] = "gemini305-video-s13-output-first/v6"
    component["p2_replay"]["completion_schema"] = "gemini305-video-s13-p2-completion/v6"

    with pytest.raises(ValueError, match="M5.1-r4"):
        validate_s13_document(document, path=V6_CONFIG)

    component.pop("m51_r4")
    legacy = validate_s13_document(document, path=V6_CONFIG)
    assert legacy.m51_r3_enabled is True
    assert legacy.m51_r4_enabled is False
    assert is_s13_identity(
        algorithm_id=S13_M51_R3_ALGORITHM_ID,
        implementation_id=S13_M51_R4_IMPLEMENTATION_ID,
        role="candidate",
    )


def test_m51_r4_defaults_freeze_thirteen_states_and_weights() -> None:
    config = S13M51R4Config()

    assert config.normal_search_states_px == (
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
    )
    assert dict(config.component_match_weights) == {
        "y_overlap": 0.30,
        "predicted_y": 0.25,
        "endpoint_distance": 0.20,
        "correlation": 0.15,
        "uniqueness": 0.10,
    }
    assert config.correction_gain_candidates == (1.0, 0.75, 0.5)
    with pytest.raises(TypeError):
        config.component_match_weights["y_overlap"] = 1.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        config.maximum_pair_gap = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides",
    (
        {"component_match_weights": {"y_overlap": 1.0}},
        {
            "component_match_weights": {
                "y_overlap": 0.31,
                "predicted_y": 0.25,
                "endpoint_distance": 0.20,
                "correlation": 0.15,
                "uniqueness": 0.10,
            }
        },
        {"normal_search_step_px": 0.4},
        {"normal_search_minimum_px": -2.5},
        {"minimum_formal_owner_support_retention": 0.999999},
        {"correction_gain_candidates": (1.0, 0.5, 0.75)},
        {"application_policy": "apply_even_invalid_maps"},
        {"minimum_application_segment_pair_count": 3, "minimum_chain_pair_count": 2},
        {"maximum_combined_map_displacement_px": 2.0},
        {"minimum_jacobian": float("nan")},
    ),
)
def test_m51_r4_rejects_invalid_ranges_and_relationships(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        S13M51R4Config(**overrides)


def test_m51_r4_threshold_boundaries_are_inclusive() -> None:
    config = S13M51R4Config(
        minimum_component_y_overlap_fraction=0.0,
        minimum_component_match_margin_fraction=0.0,
        minimum_c2e_correlation=1.0,
        minimum_c2e_uniqueness_fraction=0.0,
        maximum_forward_reverse_discrepancy_px=0.0,
        minimum_evidence_sample_retention=1.0,
    )

    assert config.minimum_component_y_overlap_fraction == 0.0
    assert config.minimum_c2e_correlation == 1.0
    assert config.maximum_forward_reverse_discrepancy_px == 0.0
