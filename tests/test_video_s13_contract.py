from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_contract import (
    S13_ALGORITHM_ID,
    load_s13_config,
    validate_s13_document,
)
from panorama_demo.video_s13_motion import descriptive_delta_risk


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v3.yaml"


def test_s13_config_and_sibling_manifest_are_isolated_and_hash_bound() -> None:
    config = load_s13_config(CONFIG)
    spec = build_algorithm_spec(CONFIG, expected_role="candidate")
    assert spec.algorithm_id == S13_ALGORITHM_ID
    assert spec.candidate_manifest_path == CONFIG.parent / "candidate_manifest.json"
    assert spec.allow_baseline_fallback is False
    assert config.component["base"]["transition_width_px"] == 0
    assert config.component["depth_assist"]["enabled"] is False
    assert config.component["output"]["write_production_delivery"] is False


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
