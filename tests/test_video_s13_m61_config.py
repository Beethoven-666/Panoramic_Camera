from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from panorama_demo.video_s13_contract import is_s13_identity, is_s13_m61_identity
from panorama_demo.video_s13_m61_config import (
    M61_ALGORITHM_ID,
    M61_IMPLEMENTATION_ID,
    M61_LUMINANCE_FIELD_SCHEMA,
    M61_QUALITY_THRESHOLDS_SCHEMA,
    M61_SCHEMA_REGISTRY,
    M61_THRESHOLD_APPROVAL_SCHEMA,
    S13LuminanceFieldDisabledConfig,
    S13BlendConfig,
    S13PhotometricEvidenceConfig,
    S13PhotometricSelectionConfig,
    S13PhotometricSolverConfig,
    S13VisualQualityConfig,
    canonical_sha256,
    load_s13_m61_effective_config,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    evidence = S13PhotometricEvidenceConfig()
    evidence_sha = evidence.canonical_sha256
    branches = ["fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose"]
    approval = {
        "schema": M61_THRESHOLD_APPROVAL_SCHEMA,
        "proposed_threshold_sha256": "1" * 64,
        "baseline_quality_manifest_sha256": "2" * 64,
        "mde_calibration_sha256": "3" * 64,
        "four_p2_parents": [
            {
                "branch": branch,
                "run_id": f"run-{index}",
                "p2_completion_sha256": f"{index + 4:x}" * 64,
                "p2_canonical_image_sha256": f"{index + 8:x}" * 64,
            }
            for index, branch in enumerate(branches)
        ],
        "decision": "approved",
        "approver": "maintainer",
        "approved_at": "2026-08-12T00:00:00Z",
        "reason": "approved fixed Q0+B0 calibration",
    }
    approval_sha = canonical_sha256(approval)
    threshold = {
        "schema": M61_QUALITY_THRESHOLDS_SCHEMA,
        "status": "final",
        "photometric_evidence_config_sha256": evidence_sha,
        "mde": {"aggregate_p95_linear": 0.0005},
        "metrics": {},
        "baseline": {},
        "actionability": {},
        "proposed_threshold_sha256": approval["proposed_threshold_sha256"],
        "threshold_approval_sha256": approval_sha,
    }
    threshold_sha = canonical_sha256(threshold)
    _write(tmp_path / "quality.json", threshold)
    _write(tmp_path / "approval.json", approval)
    candidate = {
        "candidate_id": M61_ALGORITHM_ID,
        "algorithm_id": M61_ALGORITHM_ID,
        "implementation_id": M61_IMPLEMENTATION_ID,
        "m61_bootstrap": {
            "quality_thresholds_path": "quality.json",
            "quality_thresholds_sha256": threshold_sha,
            "threshold_approval_path": "approval.json",
            "threshold_approval_sha256": approval_sha,
            "photometric_evidence_config_sha256": evidence_sha,
            "solver": asdict(S13PhotometricSolverConfig()),
            "selection": asdict(S13PhotometricSelectionConfig()),
            "blend": asdict(S13BlendConfig()),
            "visual_quality": asdict(S13VisualQualityConfig()),
            "luminance_field": asdict(S13LuminanceFieldDisabledConfig()),
        },
    }
    candidate["config_sha256"] = canonical_sha256(candidate)
    path = tmp_path / "candidate.yaml"
    path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    return path, asdict(evidence)


def test_m61_identity_is_additive_and_v3_identity_is_unchanged() -> None:
    assert is_s13_m61_identity(
        algorithm_id=M61_ALGORITHM_ID, implementation_id=M61_IMPLEMENTATION_ID, role="candidate"
    )
    assert not is_s13_identity(
        algorithm_id=M61_ALGORITHM_ID, implementation_id=M61_IMPLEMENTATION_ID, role="candidate"
    )
    assert M61_SCHEMA_REGISTRY["contract"] == "gemini305-video-s13-output-first/v4"
    assert M61_SCHEMA_REGISTRY["p3_hard_audit"].endswith("/v2")


def test_evidence_config_is_strict_finite_and_hash_stable() -> None:
    config = S13PhotometricEvidenceConfig.from_mapping({})
    assert config.diagnostic_shoulder_per_side_px == 64
    assert config.canonical_sha256 == S13PhotometricEvidenceConfig.from_mapping(
        copy.deepcopy(asdict(config))
    ).canonical_sha256
    with pytest.raises(ValueError, match="unknown keys"):
        S13PhotometricEvidenceConfig.from_mapping({"unused": 1})
    with pytest.raises(ValueError, match="must be in"):
        S13PhotometricEvidenceConfig.from_mapping({"train_fraction": float("nan")})


def test_q4_contract_rejects_nonzero_or_solver_invocation() -> None:
    valid = asdict(S13LuminanceFieldDisabledConfig())
    assert valid["schema"] == M61_LUMINANCE_FIELD_SCHEMA
    with pytest.raises(ValueError, match="disabled zero"):
        S13LuminanceFieldDisabledConfig.from_mapping({**valid, "canonical_field_value": 0.01})
    with pytest.raises(ValueError, match="disabled zero"):
        S13LuminanceFieldDisabledConfig.from_mapping({**valid, "solver": "irls"})


def test_solver_selection_blend_and_quality_are_strict() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        S13PhotometricSolverConfig.from_mapping({"dead_parameter": 1})
    with pytest.raises(ValueError, match="finite"):
        S13PhotometricSolverConfig.from_mapping({"huber_delta": float("nan")})
    with pytest.raises(ValueError, match="gain bounds"):
        S13PhotometricSolverConfig.from_mapping({"minimum_gain": 1.0})
    with pytest.raises(ValueError, match="only B0 and B1"):
        S13BlendConfig.from_mapping({"eligible_models": ["B0_owner_only", "B2"]})
    with pytest.raises(ValueError, match="color metric"):
        S13VisualQualityConfig.from_mapping({"color_metric": "delta76"})


def test_aggregate_p95_uses_exact_two_percent_or_point_0005_formula() -> None:
    selection = S13PhotometricSelectionConfig.from_mapping({})
    assert selection.aggregate_p95_nonregression_passes(0.01, 0.0105)
    assert not selection.aggregate_p95_nonregression_passes(0.01, 0.010501)
    assert selection.aggregate_p95_nonregression_passes(0.10, 0.102)
    assert not selection.aggregate_p95_nonregression_passes(0.10, 0.102001)
    with pytest.raises(ValueError, match="fixed at 2%"):
        S13PhotometricSelectionConfig.from_mapping(
            {"aggregate_p95_nonreg_absolute_tolerance_linear": 0.0015}
        )


def test_effective_config_requires_final_threshold_and_approved_witness(tmp_path: Path) -> None:
    candidate, evidence = _fixture(tmp_path)
    effective = load_s13_m61_effective_config(
        candidate, photometric_evidence_config=evidence
    )
    assert effective.algorithm_id == M61_ALGORITHM_ID
    assert len(effective.effective_config_sha256) == 64

    approval_path = tmp_path / "approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["decision"] = "rejected"
    _write(approval_path, approval)
    with pytest.raises(ValueError, match="missing or rejected"):
        load_s13_m61_effective_config(candidate, photometric_evidence_config=evidence)


def test_effective_config_rejects_path_escape_and_threshold_tamper(tmp_path: Path) -> None:
    candidate_path, evidence = _fixture(tmp_path)
    candidate = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    candidate["m61_bootstrap"]["quality_thresholds_path"] = "../quality.json"
    candidate["config_sha256"] = canonical_sha256(
        {key: value for key, value in candidate.items() if key != "config_sha256"}
    )
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        load_s13_m61_effective_config(candidate_path, photometric_evidence_config=evidence)

    candidate_path, evidence = _fixture(tmp_path / "tamper")
    threshold_path = candidate_path.parent / "quality.json"
    threshold = json.loads(threshold_path.read_text(encoding="utf-8"))
    threshold["mde"]["aggregate_p95_linear"] = 0.0015
    _write(threshold_path, threshold)
    with pytest.raises(ValueError, match="threshold SHA disagrees"):
        load_s13_m61_effective_config(candidate_path, photometric_evidence_config=evidence)
