from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import yaml

from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_bundle import verify_stage
from panorama_demo.video_s13_contract import load_s13_config, validate_s13_document
from panorama_demo.video_s13_experiment import (
    _ExternalBoxObservationSeed,
    _box_target_coverage_summary,
    _component_obligation_outcomes,
    _external_box_physical_tracks,
    _seeded_edge_trace_metrics,
    _segment_authority_support,
    _strong_edge_trace_metrics,
    _verify_segment_support_asset,
    run_s13_experiment,
)
from panorama_demo.video_s13_bundle import sha256_file, write_npz
from panorama_demo.video_s13_m51_r4_component_chain import (
    canonical_s13_support_sha256,
)


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
    manual_c2e_forward_m7: bool = False,
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
        manual_c2e_forward_m7=manual_c2e_forward_m7,
    )


def test_v6_explicit_manual_c2e_forward_seals_m1_through_m7(tmp_path) -> None:
    session = _video_session(tmp_path)
    report = _run_v6(
        session,
        tmp_path / "manual_m7",
        manual_c2e_forward_m7=True,
    )
    generation = Path(str(report["generation"]))

    assert report["final_stage"] == "P4"
    assert report["optimizer_state"] == "m7_sealed"
    assert report["m5"]["state"] == "P2_sealed"
    assert report["m6"]["state"] == "P3_sealed"
    assert report["m7"]["state"] == "P4_sealed"
    manifest = json.loads(
        (generation / "generation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["implemented_milestones"] == [
        "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7",
    ]
    completion = verify_stage(
        generation / "P4",
        completion_name="P4_completion.json",
        schema="gemini305-video-s13-p4-manual-c2e-completion/v1",
    )
    assert completion["selected_candidate"] == "R0_keep_p3"
    assert completion["new_pixel_generation"] is False


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
    performance = json.loads(
        (generation / "P2/performance.json").read_text(encoding="utf-8")
    )
    component_manifest = json.loads(
        (generation / "P2/component_chain_transactions/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "component_match_matrices" in component_manifest
    assert "exact_evidence_propagation" in component_manifest
    assert len(component_manifest["decision_payload_stable_sha256"]) == 64
    source_map_manifest = json.loads(
        (generation / "P2/source_maps/manifest.json").read_text(encoding="utf-8")
    )
    assert performance["p2_full_resolution_render_count"] == 2
    assert performance["formal_logical_source_render_count"] == 2 * len(
        source_map_manifest["sources"]
    )
    assert performance["formal_raw_rgb_remap_invocations"] == len(
        source_map_manifest["sources"]
    )
    assert performance["sampled_source_cache_hit_count"] == len(
        source_map_manifest["sources"]
    )
    assert performance["component_segment_split_count"] == len(
        component_manifest["split_lineage"]
    )
    assert performance["forward_edge_hypothesis_count"] == sum(
        len(row["forward_scores"])
        for row in component_manifest["forward_reverse_hypotheses"]
    )
    assert performance["reverse_edge_hypothesis_count"] == sum(
        len(row["reverse_scores"])
        for row in component_manifest["forward_reverse_hypotheses"]
    )
    assert performance["component_propagation_pair_count"] == len(
        component_manifest["exact_evidence_propagation"]["probed_pair_indices"]
    )
    assert performance["gain_enumeration_count"] == performance[
        "component_gain_candidate_count"
    ]
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


def test_clean_v6_semantic_rejection_does_not_publish_a_sealed_p2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _video_session(tmp_path)
    output = tmp_path / "out"
    spec = replace(
        build_algorithm_spec(V6_CONFIG, expected_role="candidate"),
        working_tree_dirty=False,
    )

    monkeypatch.setattr(
        "panorama_demo.video_s13_experiment.verify_s13_v6_r2_p2",
        lambda _path: (_ for _ in ()).throw(ValueError("semantic rejection")),
    )
    report = run_s13_experiment(
        input_path=session,
        output=output,
        candidate_config=V6_CONFIG,
        algorithm_spec=spec,
        trajectory_cache=None,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=True,
        config_path=None,
    )

    generation = Path(str(report["generation"]))
    assert report["m5"]["state"] == "failed_parent_preserved"
    assert report["m5"]["error"] == "semantic rejection"
    assert not (generation / "P2").exists()
    latest = json.loads((output / "current_latest.json").read_text(encoding="utf-8"))
    assert latest["stage"] == "P1"


def test_external_box_trace_measures_steps_breaks_and_double_edges() -> None:
    clean = np.zeros((48, 64, 3), dtype=np.uint8)
    for column in range(clean.shape[1]):
        row = 12 + column // 4
        clean[row:, column] = 255
    valid = np.ones(clean.shape[:2], dtype=bool)
    clean_metrics, overlay = _strong_edge_trace_metrics(clean, valid)
    assert clean_metrics["edge_step_p95_px"] <= 1.0
    assert clean_metrics["maximum_local_step_px"] <= 1.0
    assert clean_metrics["break_length_px"] == 0
    assert overlay.shape == clean.shape

    broken = clean.copy()
    broken[:, 20:28] = 127
    broken_metrics, _ = _strong_edge_trace_metrics(broken, valid)
    assert broken_metrics["break_length_px"] >= 6


def test_external_box_seeded_trace_does_not_jump_between_physical_edges() -> None:
    image = np.zeros((80, 96, 3), dtype=np.uint8)
    upper_support: list[tuple[int, int]] = []
    lower_support: list[tuple[int, int]] = []
    for column in range(image.shape[1]):
        upper = 12 + column // 8
        lower = 48 + column // 12
        image[upper:, column] = 96
        image[lower:, column] = 255
        if 32 <= column < 64:
            upper_support.append((column, upper))
            lower_support.append((column, lower))
    valid = np.ones(image.shape[:2], dtype=bool)
    tracks = (
        {
            "track_id": "upper",
            "support_xy": np.asarray(upper_support, dtype=np.int32),
            "observation_authority": [],
            "obligation_ids": ["upper-obligation"],
        },
        {
            "track_id": "lower",
            "support_xy": np.asarray(lower_support, dtype=np.int32),
            "observation_authority": [],
            "obligation_ids": ["lower-obligation"],
        },
    )

    metrics, overlay = _seeded_edge_trace_metrics(image, valid, tracks)

    assert metrics["track_count"] == 2
    assert metrics["coverage_fraction"] >= 0.95
    assert metrics["edge_step_p95_px"] <= 1.0
    assert metrics["maximum_local_step_px"] <= 1.0
    assert all(row["maximum_local_step_px"] <= 1.0 for row in metrics["tracks"])
    assert overlay.shape == image.shape


def test_external_box_physical_tracks_keep_nearby_components_separate() -> None:
    def seed(pair: int, component: int, row: int) -> _ExternalBoxObservationSeed:
        support = np.asarray(
            [(column, row + (column - 20) // 8) for column in range(20, 44)],
            dtype=np.int32,
        )
        return _ExternalBoxObservationSeed(
            pair_index=pair,
            component_id=component,
            support_sha256=f"{pair}-{component}",
            support_xy=support,
            obligation_id=f"obligation-{pair}-{component}",
            severe=True,
            evaluable=True,
        )

    tracks = _external_box_physical_tracks(
        (seed(71, 8, 15), seed(71, 9, 45), seed(72, 7, 15), seed(72, 8, 45)),
        minimum_y_overlap_fraction=0.5,
        maximum_predicted_y_difference_px=6.0,
        maximum_normal_difference_degrees=10.0,
    )

    assert len(tracks) == 2
    authorities = sorted(
        sorted((row["pair_index"], row["component_id"])
               for row in track["observation_authority"])
        for track in tracks
    )
    assert authorities == [[(71, 8), (72, 7)], [(71, 9), (72, 8)]]


def test_box_completion_requires_resolved_repair_but_not_nonsevere_anchor() -> None:
    severe = {
        "obligation_id": "severe", "pair_index": 71, "component_id": 8,
        "support_sha256": "severe-sha", "severe": True, "evaluable": True,
    }
    anchor = {
        "obligation_id": "anchor", "pair_index": 72, "component_id": 8,
        "support_sha256": "anchor-sha", "severe": False, "evaluable": True,
    }
    component_doc = {
        "accepted_segment_ids": ["resolved"],
        "segments": [{
            "segment_id": "resolved",
            "state": "resolved",
            "observation_authority": [{
                "pair_index": 71, "component_id": 8,
                "support_sha256": "severe-sha",
            }],
        }],
    }

    summary = _box_target_coverage_summary(component_doc, (severe, anchor))

    assert summary["box_target_obligation_ids"] == ["severe", "anchor"]
    assert summary["box_target_repair_obligation_ids"] == ["severe"]
    assert summary["box_target_anchor_obligation_ids"] == ["anchor"]
    assert summary["repair_obligations_covered"] is True


def test_box_completion_does_not_accept_improved_unresolved_segment() -> None:
    obligation = {
        "obligation_id": "severe", "pair_index": 71, "component_id": 9,
        "support_sha256": "severe-sha", "severe": True, "evaluable": True,
    }
    component_doc = {
        "accepted_segment_ids": ["partial"],
        "segments": [{
            "segment_id": "partial",
            "state": "improved_unresolved",
            "observation_authority": [{
                "pair_index": 71, "component_id": 9,
                "support_sha256": "severe-sha",
            }],
        }],
    }

    summary = _box_target_coverage_summary(component_doc, (obligation,))

    assert summary["repair_obligations_covered"] is False
    assert summary["box_target_unresolved_repair_obligation_ids"] == ["severe"]


def test_segment_support_asset_reverifies_exact_observation_authority(
    tmp_path: Path,
) -> None:
    selected_support = np.asarray(((10, 20), (11, 20), (12, 21)), np.int32)
    sibling_support = np.asarray(((40, 50), (41, 50), (42, 51)), np.int32)
    selected_sha = canonical_s13_support_sha256(selected_support)
    sibling_sha = canonical_s13_support_sha256(sibling_support)
    selected = SimpleNamespace(
        pair_index=71, component_id=8, mask_sha256=selected_sha
    )
    sibling = SimpleNamespace(
        pair_index=71, component_id=9, mask_sha256=sibling_sha
    )
    pair = SimpleNamespace(component_evidence=(
        (selected, SimpleNamespace(support_xy=selected_support)),
        (sibling, SimpleNamespace(support_xy=sibling_support)),
    ))
    authority = ({
        "pair_index": 71,
        "component_id": 8,
        "support_sha256": selected_sha,
    },)

    support, sealed_authority = _segment_authority_support((pair,), authority)

    assert np.array_equal(support, selected_support)
    assert not np.any(np.all(support[:, None] == sibling_support[None, :], axis=2))
    transaction_root = tmp_path / "component_chain_transactions"
    transaction_root.mkdir()
    observation_asset = "observation_pair_0071_component_0008.npz"
    write_npz(transaction_root / observation_asset, {"support_xy": selected_support})
    segment_asset = "s_exact.npz"
    write_npz(transaction_root / segment_asset, {
        "support_xy": support,
        "authority_pair_indices": np.asarray((71,), np.int32),
        "authority_component_ids": np.asarray((8,), np.int32),
        "authority_support_sha256": np.asarray((selected_sha,)),
    })
    segment_document = {
        "asset": segment_asset,
        "asset_sha256": sha256_file(transaction_root / segment_asset),
        "observation_authority": list(sealed_authority),
    }
    evidence_assets = ({
        "pair_index": 71,
        "component_id": 8,
        "support_sha256": selected_sha,
        "asset": observation_asset,
        "asset_sha256": sha256_file(transaction_root / observation_asset),
    },)

    verification = _verify_segment_support_asset(
        transaction_root, segment_document, evidence_assets
    )

    assert verification == {
        "passed": True, "authority_count": 1, "support_sample_count": 3,
    }


def test_component_manifest_explains_every_obligation_outcome() -> None:
    obligations = [
        {
            "obligation_id": "resolved", "pair_index": 1, "component_id": 2,
            "support_sha256": "r", "severe": True, "evaluable": True,
        },
        {
            "obligation_id": "anchor", "pair_index": 1, "component_id": 3,
            "support_sha256": "a", "severe": False, "evaluable": True,
        },
        {
            "obligation_id": "missing", "pair_index": 1, "component_id": 4,
            "support_sha256": "m", "severe": True, "evaluable": True,
        },
    ]
    audit = {
        "baseline_c2e_obligations": obligations,
        "accepted_segment_ids": ["winner"],
        "rejected_segment_ids": [],
        "deferred_segment_ids": [],
        "segments": [{
            "segment_id": "winner", "state": "resolved",
            "observation_authority": [{
                "pair_index": 1, "component_id": 2, "support_sha256": "r",
            }],
        }],
    }

    outcomes = _component_obligation_outcomes(audit)

    assert [row["obligation_id"] for row in outcomes] == [
        "resolved", "anchor", "missing",
    ]
    assert [row["outcome"] for row in outcomes] == [
        "resolved", "safe_anchor", "uncovered",
    ]
    assert outcomes[2]["reason"] == "no_segment_contains_exact_authority"
