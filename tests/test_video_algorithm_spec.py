from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
import yaml

from panorama_demo.video_algorithm import (
    VIDEO_CANDIDATE_CONFIG_SCHEMA,
    VideoAlgorithmConfigurationError,
    build_algorithm_spec,
    canonical_config_sha256,
)
from panorama_demo.video_observability import ObservabilitySpec


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_spec_exposes_immutable_identity():
    path = ROOT / "configs" / "video_candidates" / "C4_raft_rgbd_layered_mesh.yaml"
    spec = build_algorithm_spec(path, expected_role="candidate")

    assert spec.role == "candidate"
    assert spec.algorithm_id == "C4_raft_rgbd_layered_mesh"
    assert spec.config_sha256 == "7cd2c090032270284c41167991c44846f6d44bb44ae93869f4d5df6d76968e3d"
    assert spec.allow_baseline_fallback is False
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    actual_dirty = bool(subprocess.run(
        ["git", "status", "--porcelain=v1"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip())
    assert spec.source_commit == actual_head
    assert spec.working_tree_dirty is actual_dirty


def test_candidate_self_hash_excludes_only_its_self_referential_field(tmp_path):
    source = ROOT / "configs" / "video_candidates" / "C1_constrained_owner.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["config_sha256"] = "0" * 64
    path = tmp_path / "tampered.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(VideoAlgorithmConfigurationError, match="config_sha256 mismatch"):
        build_algorithm_spec(path)

    config["config_sha256"] = canonical_config_sha256(config)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = yaml.safe_load(
        (ROOT / "configs" / "video_candidates" / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    (tmp_path / "candidate_manifest.json").write_text(
        __import__("json").dumps(manifest), encoding="utf-8"
    )
    assert build_algorithm_spec(path).config_sha256 == config["config_sha256"]
    assert config["config_schema"] == VIDEO_CANDIDATE_CONFIG_SCHEMA


def test_observability_is_algorithm_independent_and_audit_requires_full_report():
    assert ObservabilitySpec().as_dict() == {
        "report_level": "summary",
        "artifact_level": "minimal",
    }
    assert ObservabilitySpec(report_level="full", artifact_level="audit").artifact_level == "audit"
    with pytest.raises(ValueError, match="requires report_level=full"):
        ObservabilitySpec(report_level="summary", artifact_level="audit")
