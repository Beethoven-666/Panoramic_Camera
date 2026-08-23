from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_bundle import verify_stage
from panorama_demo.video_s13_contract import load_s13_config, validate_s13_document
from panorama_demo.video_s13_experiment import run_s13_experiment


ROOT = Path(__file__).resolve().parents[1]
V5_CONFIG = (
    ROOT
    / "configs/video_candidates/s013/"
    "S013_output_first_progressive_dense_central_slit_v5.yaml"
)
V4_CONFIG = (
    ROOT
    / "configs/video_candidates/s013/"
    "S013_output_first_progressive_dense_central_slit_v4.yaml"
)
P2_V5_SCHEMA = "gemini305-video-s13-p2-completion/v5"


def _assert_canonical_v5_transaction(transaction: dict[str, object]) -> None:
    content = dict(transaction)
    digest = content.pop("result_stage_sha256")
    assert digest == hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    pending = [content]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, float):
            assert value == round(value, 6)
            assert value != 0.0 or str(value) == "0.0"


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


def _run_v5(
    session: Path,
    output: Path,
    *,
    run_m6: bool | None = None,
    resume_generation: Path | None = None,
) -> dict[str, object]:
    return run_s13_experiment(
        input_path=session,
        output=output,
        candidate_config=V5_CONFIG,
        algorithm_spec=build_algorithm_spec(V5_CONFIG, expected_role="candidate"),
        trajectory_cache=None,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=True,
        config_path=None,
        run_m6=run_m6,
        resume_generation=resume_generation,
    )


def test_v5_identity_is_p2_only_and_has_no_m61_bootstrap() -> None:
    config = load_s13_config(V5_CONFIG)

    assert config.p2_completion_schema == P2_V5_SCHEMA
    assert config.p2_only is True
    assert config.requires_m61_bootstrap is False
    assert config.m51_r2_enabled is True
    assert config.m6_eligible is False
    assert config.component["forward_pipeline"] == {
        "stage_order": ["P0", "P1", "P2"],
        "default_stop_after": "P2",
        "forbid_parent_reselection": True,
        "current_preview_runtime_authority": False,
    }
    assert "m61_bootstrap" not in config.document
    assert config.document["required_output_components"] == ["s013_p2_v5"]


@pytest.mark.parametrize(
    "binding",
    (
        {
            "quality_thresholds_path": "configs/video_candidates/s013/quality_thresholds_m61_v2.json"
        },
        {
            "threshold_approval_path": "artifacts/S013_M6_1_metrics_baseline_v2/threshold_approval.json"
        },
    ),
)
def test_v5_rejects_old_m6_threshold_or_approval_lineage(binding: dict[str, str]) -> None:
    document = yaml.safe_load(V5_CONFIG.read_text(encoding="utf-8"))
    document["m61_bootstrap"] = binding
    with pytest.raises(ValueError, match="bootstrap|threshold/approval"):
        validate_s13_document(document, path=V5_CONFIG)


def test_v5_default_stops_at_hash_sealed_p2(tmp_path: Path) -> None:
    output = tmp_path / "out"
    report = _run_v5(_video_session(tmp_path), output)
    generation = Path(str(report["generation"]))
    completion = verify_stage(
        generation / "P2",
        completion_name="P2_completion.json",
        schema=P2_V5_SCHEMA,
    )

    assert report["final_stage"] == "P2"
    assert completion["m6_eligible"] is False
    assert completion["m6_blocked_reason"] == (
        "successor_p2_requires_fresh_four_branch_threshold_lineage"
    )
    assert "native_p2_v4" not in completion
    pointer = json.loads((output / "current_latest.json").read_text(encoding="utf-8"))
    assert pointer["stage"] == "P2"
    assert not (generation / "P3").exists()
    assert not (generation / "M6").exists()
    assert not list(output.rglob("delivery.json"))
    assert not list(output.rglob("video_delivery.json"))

    transaction_manifest = json.loads(
        (generation / "P2/pair_transactions.json").read_text(encoding="utf-8")
    )
    written_transactions = []
    for transaction_path in sorted(
        (generation / "P2/pair_transactions").glob("pair_*.json")
    ):
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
        _assert_canonical_v5_transaction(transaction)
        written_transactions.append(transaction)
    assert transaction_manifest["pairs"] == written_transactions


def test_v5_explicit_m6_and_resume_fail_closed(tmp_path: Path) -> None:
    session = _video_session(tmp_path)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="P2-only"):
        _run_v5(session, output, run_m6=True)

    report = _run_v5(session, output)
    generation = Path(str(report["generation"]))
    sealed_p2 = (generation / "P2/P2_completion.json").read_bytes()
    with pytest.raises(ValueError, match="P2-only"):
        _run_v5(session, output, resume_generation=generation)
    assert (generation / "P2/P2_completion.json").read_bytes() == sealed_p2
    assert not (generation / "P3").exists()

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
