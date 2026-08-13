from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np
import yaml

from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_bundle import verify_stage
from panorama_demo.video_s13_contract import load_s13_config, validate_s13_document
from panorama_demo.video_s13_experiment import run_s13_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/video_candidates/s013"
V4_CONFIG = CONFIG_ROOT / "S013_output_first_progressive_dense_central_slit_v4.yaml"
V5_CONFIG = CONFIG_ROOT / "S013_output_first_progressive_dense_central_slit_v5.yaml"
V6_CONFIG = CONFIG_ROOT / "S013_output_first_progressive_dense_central_slit_v6.yaml"
P2_V6_SCHEMA = "gemini305-video-s13-p2-completion/v6-r2"
V5_FILE_SHA256 = "7a30885c031b2c2ad07f6bca315de3e2df1776b7f42ae9e5fce5ff990028b975"
V5_CONFIG_SHA256 = "3766a1f322fe3671381b1b617a19c81c45b3c5b91839da47f6db5e21f5bce813"


def _video_session(tmp_path: Path) -> Path:
    root = generate_sequence(
        tmp_path / "video", frame_count=4, frame_width=96, frame_height=64, step=8
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "capture_mode": "continuous_rgbd_video_fixed_exposure",
            "product_eligibility": {
                "photo_panorama": False,
                "video_panorama": True,
            },
            "writer_errors": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _run_v6(
    session: Path,
    output: Path,
    *,
    run_m6: bool | None = None,
    resume_generation: Path | None = None,
) -> dict[str, object]:
    return run_s13_experiment(
        input_path=session,
        output=output,
        candidate_config=V6_CONFIG,
        algorithm_spec=build_algorithm_spec(V6_CONFIG, expected_role="candidate"),
        trajectory_cache=None,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=True,
        config_path=None,
        run_m6=run_m6,
        resume_generation=resume_generation,
    )


def test_v6_identity_is_an_isolated_p2_only_successor() -> None:
    config = load_s13_config(V6_CONFIG)

    assert config.document["parent_candidate_id"] == (
        "S013_output_first_progressive_dense_central_slit_v5"
    )
    assert config.document["implementation_id"] == (
        "s013_output_first_progressive_dense_central_slit_m51_r4_component_chain_c2e"
    )
    assert config.document["required_output_components"] == ["s013_p2_v6"]
    assert config.p2_completion_schema == P2_V6_SCHEMA
    assert config.p2_only is True
    assert config.m6_eligible is False
    assert config.component["m51_r3"] == {
        "enabled": True,
        "reassessment_pair_indices": [71],
        "micro_rescue_enabled": False,
        "c2e_enabled": False,
    }
    assert config.component["forward_pipeline"]["stage_order"] == ["P0", "P1", "P2"]
    assert config.component["forward_pipeline"]["default_stop_after"] == "P2"
    assert "m61_bootstrap" not in config.document


def test_v6_does_not_mutate_v5_config_or_manifest_entry() -> None:
    assert hashlib.sha256(V5_CONFIG.read_bytes()).hexdigest() == V5_FILE_SHA256
    document = yaml.safe_load(V5_CONFIG.read_text(encoding="utf-8"))
    assert document["config_sha256"] == V5_CONFIG_SHA256
    manifest = json.loads((CONFIG_ROOT / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidates"][document["candidate_id"]]["config_sha256"] == V5_CONFIG_SHA256


@pytest.mark.parametrize(
    "binding",
    (
        {"quality_thresholds_path": "configs/video_candidates/s013/quality_thresholds_m61_v2.json"},
        {"threshold_approval_path": "artifacts/S013_M6_1_metrics_baseline_v2/threshold_approval.json"},
    ),
)
def test_v6_rejects_old_m6_threshold_or_approval_lineage(binding: dict[str, str]) -> None:
    document = yaml.safe_load(V6_CONFIG.read_text(encoding="utf-8"))
    document["m61_bootstrap"] = binding
    with pytest.raises(ValueError, match="bootstrap|threshold/approval"):
        validate_s13_document(document, path=V6_CONFIG)


def test_v6_defaults_to_sealed_p2_and_rejects_m6_or_resume(tmp_path: Path) -> None:
    session = _video_session(tmp_path)
    explicit_output = tmp_path / "explicit"
    with pytest.raises(ValueError, match="P2-only"):
        _run_v6(session, explicit_output, run_m6=True)
    assert not explicit_output.exists()

    output = tmp_path / "out"
    report = _run_v6(session, output)
    generation = Path(str(report["generation"]))
    completion = verify_stage(
        generation / "P2", completion_name="P2_completion.json", schema=P2_V6_SCHEMA
    )
    transactions = json.loads(
        (generation / "P2/pair_transactions.json").read_text(encoding="utf-8")
    )
    assert transactions["schema"] == (
        "gemini305-video-s13-m5-pair-transactions/v4"
    )
    assert (generation / "P2/component_chain_transactions/manifest.json").is_file()
    assert (generation / "P2/source_corrections/manifest.json").is_file()
    assert (generation / "P2/source_maps/manifest.json").is_file()
    assert completion["p2_replay_schema"] == "gemini305-video-s13-p2-replay/v2"
    assert completion["application_state"] in {"complete", "partial", "none"}
    assert completion["repair_complete"] is False
    with np.load(generation / "P2/p2_pixel_provenance.npz", allow_pickle=False) as stored:
        assert "component_correction_field_id" in stored.files
    assert report["final_stage"] == "P2"
    assert completion["m6_eligible"] is False
    assert completion["m6_blocked_reason"] == (
        "successor_p2_requires_fresh_four_branch_threshold_lineage"
    )
    assert not (generation / "P3").exists()
    assert not (generation / "M6").exists()

    sealed_p2 = (generation / "P2/P2_completion.json").read_bytes()
    with pytest.raises(ValueError, match="P2-only"):
        _run_v6(session, output, resume_generation=generation)
    with pytest.raises(ValueError, match="algorithm/config binding changed"):
        run_s13_experiment(
            input_path=session,
            output=output,
            candidate_config=V4_CONFIG,
            algorithm_spec=build_algorithm_spec(V4_CONFIG, expected_role="candidate"),
            trajectory_cache=None,
            reuse_online_trajectory=False,
            run_offline_orb=False,
            ignore_pose=True,
            config_path=None,
            resume_generation=generation,
        )
    assert (generation / "P2/P2_completion.json").read_bytes() == sealed_p2
    assert not (generation / "P3").exists()
