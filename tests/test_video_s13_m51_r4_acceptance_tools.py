from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts.summarize_s13_m51_r4_validation import summarize_acceptance
from scripts.verify_s13_m51_r4_reproducible_seal import (
    NORMALIZATION_POINTERS,
    normalize_json,
    summarize_paired_timings,
    verify,
)


BRANCHES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")
BOX_ASSETS = (
    "current_roi.png",
    "candidate_roi.png",
    "current_vs_candidate.png",
    "edge_trace_overlay.png",
    "component_masks_overlay.png",
    "pair_0068_0078_metrics.json",
    "component_chain_audit.json",
    "source_offsets.json",
    "forward_reverse_hypotheses.json",
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _npz(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=np.asarray([value, value + 1], np.int32))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rebind_p2(p2: Path) -> None:
    component = p2 / "component_chain_transactions/manifest.json"
    correction = p2 / "source_corrections/manifest.json"
    maps = p2 / "source_maps/manifest.json"
    pairs = p2 / "pair_transactions.json"
    replay = p2 / "p2_replay_manifest.json"
    component_sha = _sha(component)
    value = json.loads(correction.read_text(encoding="utf-8"))
    value["parent_component_transaction_manifest_sha256"] = component_sha
    _json(correction, value)
    correction_sha = _sha(correction)
    value = json.loads(maps.read_text(encoding="utf-8"))
    value["parent_source_correction_manifest_sha256"] = correction_sha
    _json(maps, value)
    maps_sha = _sha(maps)
    value = json.loads(pairs.read_text(encoding="utf-8"))
    value.update(
        {
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": correction_sha,
            "source_map_oracle_manifest_sha256": maps_sha,
        }
    )
    for row in value.get("pairs", []):
        row["component_chain_c2e"].update(
            {
                "component_transaction_manifest_sha256": component_sha,
                "source_correction_manifest_sha256": correction_sha,
                "source_map_oracle_manifest_sha256": maps_sha,
            }
        )
    _json(pairs, value)
    pairs_sha = _sha(pairs)
    value = json.loads(replay.read_text(encoding="utf-8"))
    value.update(
        {
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": correction_sha,
            "source_map_oracle_manifest_sha256": maps_sha,
            "aggregate_pair_transaction_manifest_sha256": pairs_sha,
        }
    )
    _json(replay, value)
    completion = p2 / "P2_completion.json"
    value = json.loads(completion.read_text(encoding="utf-8"))
    value.update(
        {
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": correction_sha,
            "source_map_oracle_manifest_sha256": maps_sha,
            "aggregate_pair_transaction_manifest_sha256": pairs_sha,
            "p2_replay_manifest_sha256": _sha(replay),
        }
    )
    for relative in (
        "component_chain_transactions/manifest.json",
        "source_corrections/manifest.json",
        "source_maps/manifest.json",
        "pair_transactions.json",
        "p2_replay_manifest.json",
    ):
        value["assets_sha256"][relative] = _sha(p2 / relative)
    _json(completion, value)


def _cell(root: Path, round_name: str, branch: str, *, run_number: int) -> None:
    run = root / round_name / branch
    generation_id = f"generation-{round_name}-{branch}"
    generation = run / "generations" / generation_id
    p2 = generation / "P2"
    generation_manifest = {
        "schema": "gemini305-video-s13-generation/v1",
        "generation_id": generation_id,
        "algorithm": {
            "algorithm_id": "S013_output_first_progressive_dense_central_slit_v6",
            "implementation_id": (
                "s013_output_first_progressive_dense_central_slit_m51_r4_component_chain_c2e"
            ),
            "source_commit": "b" * 40,
            "working_tree_dirty": False,
            "config_sha256": "c" * 64,
            "candidate_manifest_sha256": "d" * 64,
        },
    }
    _json(generation / "generation_manifest.json", generation_manifest)
    _json(
        run / "current_latest.json",
        {
            "stage": "P2",
            "generation": f"generations/{generation_id}",
            "generation_id": generation_id,
        },
    )
    (p2 / "geometry_and_seam_panorama_owner_only.png").parent.mkdir(
        parents=True, exist_ok=True
    )
    (p2 / "geometry_and_seam_panorama_owner_only.png").write_bytes(
        b"same png " + branch.encode()
    )
    _npz(p2 / "p2_seams.npz", len(branch))
    p2.mkdir(parents=True, exist_ok=True)
    for directory in ("source_corrections", "source_maps", "pair_replay"):
        (p2 / directory).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p2 / "p2_pixel_provenance.npz",
        component_correction_field_id=np.asarray([0, -1], np.int32),
    )
    np.savez_compressed(
        p2 / "source_corrections/source_0001.npz",
        field_id=np.asarray([0], np.int32),
        delta_u=np.asarray([0.5], np.float32),
        delta_v=np.asarray([0.25], np.float32),
    )
    np.savez_compressed(
        p2 / "source_maps/source_0001.npz",
        field_id=np.asarray([0, -1], np.int32),
        u=np.asarray([1.0, 2.0], np.float32),
        v=np.asarray([1.0, 2.0], np.float32),
        valid=np.asarray([1, 1], np.uint8),
    )
    np.savez_compressed(
        p2 / "pair_replay/pair_0000.npz",
        left_component_correction_field_id=np.asarray([0, -1], np.int32),
        right_component_correction_field_id=np.asarray([-1, 0], np.int32),
    )
    stable = {
        "schema": "gemini305-video-s13-component-chain-transactions/v1",
        "decision_payload_stable_sha256": "a" * 64,
        "application_state": "partial",
        "accepted_segment_ids": ["segment-a"],
        "rejected_segment_ids": [],
        "deferred_segment_ids": [],
    }
    _json(p2 / "component_chain_transactions/manifest.json", stable)
    correction_asset = p2 / "source_corrections/source_0001.npz"
    source_correction_manifest = {
        "schema": "gemini305-video-s13-source-corrections/v1",
        "parent_component_transaction_manifest_sha256": _sha(
            p2 / "component_chain_transactions/manifest.json"
        ),
        "application_state": "partial",
        "contributors": ["segment-a"],
        "field_id_table": [
            {"field_id": 0, "segment_id": "segment-a", "source_index": 1}
        ],
        "sources": [
            {
                "source_index": 1,
                "asset": "source_0001.npz",
                "asset_sha256": _sha(correction_asset),
            }
        ],
    }
    _json(p2 / "source_corrections/manifest.json", source_correction_manifest)
    source_map_asset = p2 / "source_maps/source_0001.npz"
    _json(
        p2 / "source_maps/manifest.json",
        {
            "schema": "gemini305-video-s13-source-map-oracles/v1",
            "parent_source_correction_manifest_sha256": _sha(
                p2 / "source_corrections/manifest.json"
            ),
            "source_count": 1,
            "sources": [
                {
                    "source_index": 1,
                    "asset": "source_0001.npz",
                    "asset_sha256": _sha(source_map_asset),
                }
            ],
        },
    )
    component_sha = _sha(p2 / "component_chain_transactions/manifest.json")
    correction_sha = _sha(p2 / "source_corrections/manifest.json")
    map_sha = _sha(p2 / "source_maps/manifest.json")
    _json(
        p2 / "pair_transactions.json",
        {
            "schema": "gemini305-video-s13-m5-pair-transactions/v4",
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": correction_sha,
            "source_map_oracle_manifest_sha256": map_sha,
            "pairs": [
                {
                    "pair_index": 0,
                    "component_chain_c2e": {
                        "schema": "gemini305-video-s13-component-chain-c2e-pair-ref/v1",
                        "application_state": "partial",
                        "component_transaction_manifest_sha256": component_sha,
                        "source_correction_manifest_sha256": correction_sha,
                        "source_map_oracle_manifest_sha256": map_sha,
                        "refs": [
                            {
                                "segment_id": "segment-a",
                                "field_id": 0,
                                "role": "rescued",
                                "state": "improved_unresolved",
                            }
                        ],
                    },
                }
            ],
            "run": run_number,
        },
    )
    pair_sha = _sha(p2 / "pair_transactions.json")
    _json(
        p2 / "p2_replay_manifest.json",
        {
            "schema": "gemini305-video-s13-p2-replay/v2",
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": correction_sha,
            "source_map_oracle_manifest_sha256": map_sha,
            "aggregate_pair_transaction_manifest_sha256": pair_sha,
            "pairs": [
                {
                    "pair_index": 0,
                    "asset": "pair_replay/pair_0000.npz",
                    "relevant_segment_ids": ["segment-a"],
                }
            ],
            "run": run_number,
        },
    )
    _json(
        p2 / "hard_audit.json",
        {"passed": True, "schema": "gemini305-video-s13-hard-audit/v1"},
    )
    _json(
        p2 / "performance.json",
        {
            "total_m5": 21.0 + run_number / 10,
            "component_chain_seconds": 0.4 + run_number / 100,
            "p2_full_resolution_render_count": 2,
            "extra_full_resolution_render_count": 0,
            "backward_pyr_lk_call_count": 0,
        },
    )
    box = run / "validation/box_component_chain"
    for name in BOX_ASSETS:
        path = box / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            _json(path, {"asset": name, "box_target_repair_complete": False})
        else:
            path.write_bytes(b"image")
    assets = {
        path.relative_to(p2).as_posix(): _sha(path)
        for path in p2.rglob("*")
        if path.is_file()
    }
    _json(
        p2 / "P2_completion.json",
        {
            "schema": "gemini305-video-s13-p2-completion/v6-r2",
            "stage": "P2",
            "m6_eligible": False,
            "hard_audit_passed": True,
            "application_state": "partial",
            "repair_complete": False,
            "generation_id": generation_id,
            "generation_manifest_sha256": _sha(generation / "generation_manifest.json"),
            "algorithm_id": generation_manifest["algorithm"]["algorithm_id"],
            "implementation_id": generation_manifest["algorithm"]["implementation_id"],
            "contract_schema": "gemini305-video-s13-output-first/v6-r2",
            "config_sha256": generation_manifest["algorithm"]["config_sha256"],
            "candidate_manifest_sha256": generation_manifest["algorithm"][
                "candidate_manifest_sha256"
            ],
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": correction_sha,
            "source_map_oracle_manifest_sha256": map_sha,
            "aggregate_pair_transaction_manifest_sha256": pair_sha,
            "p2_replay_manifest_sha256": _sha(p2 / "p2_replay_manifest.json"),
            "provenance_schema": "gemini305-video-s13-p2-provenance/v6-r2",
            "applied_segment_ids": ["segment-a"],
            "resolved_segment_ids": [],
            "improved_segment_ids": ["segment-a"],
            "rejected_segment_ids": [],
            "deferred_segment_ids": [],
            "assets_sha256": assets,
        },
    )


def _acceptance(tmp_path: Path) -> Path:
    root = tmp_path / "acceptance"
    for round_index, round_name in enumerate(("round_a", "round_b")):
        for branch in BRANCHES:
            _cell(root, round_name, branch, run_number=round_index)
    _json(
        root / "paired_timing_runs.json",
        {
            "schema": "gemini305-video-s13-m51-r4-paired-timing-runs/v1",
            "pairs": [
                {
                    "pair_id": f"pair-{index}",
                    "order": (
                        "v6-r1_then_v6-r2" if index % 2 == 0 else "v6-r2_then_v6-r1"
                    ),
                    "baseline_seconds": 20.0,
                    "candidate_seconds": 20.5,
                }
                for index in range(5)
            ],
        },
    )
    return root


def test_normalization_uses_only_the_frozen_exact_json_pointers() -> None:
    assert NORMALIZATION_POINTERS == (
        "/generation_id",
        "/generation_manifest_sha256",
        "/parent_completion_sha256",
        "/created_at_utc",
        "/output_root",
        "/timing",
        "/performance",
    )
    value = {
        "generation_id": "dynamic",
        "created_at_utc": "dynamic",
        "nested": {"generation_id": "must remain"},
        "payload": 3,
    }
    assert normalize_json(value) == {
        "nested": {"generation_id": "must remain"},
        "payload": 3,
    }


def test_verify_builds_four_by_two_cell_seal_and_three_comparison_classes(
    tmp_path: Path,
) -> None:
    seal = verify(_acceptance(tmp_path))
    assert seal["schema"] == "gemini305-video-s13-m51-r4-reproducible-seal/v1"
    assert seal["normalization"]["schema"] == (
        "gemini305-video-s13-reproducibility-normalization/v1"
    )
    assert len(seal["cells"]) == 8
    assert seal["exact_comparison"]["passed"] is True
    assert seal["run_specific_lineage"]["passed"] is True
    assert seal["timing_comparison"]["paired_summary"]["pair_count"] == 5
    assert seal["timing_comparison"]["paired_summary"]["median_passed"] is True
    assert all(cell["box_asset_audit"]["passed"] for cell in seal["cells"])


def test_verify_rejects_nonallowlisted_payload_drift_or_missing_box_asset(
    tmp_path: Path,
) -> None:
    root = _acceptance(tmp_path)
    manifest = next(
        (root / "round_b/fast_direct/generations").glob(
            "*/P2/component_chain_transactions/manifest.json"
        )
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["selection_policy"] = "nonallowlisted-drift"
    _json(manifest, value)
    _rebind_p2(manifest.parents[1])
    with pytest.raises(ValueError, match="exact reproducibility"):
        verify(root)

    root = _acceptance(tmp_path / "missing")
    (root / "round_b/fast_direct/validation/box_component_chain/current_roi.png").unlink()
    with pytest.raises(ValueError, match="box component-chain assets"):
        verify(root)


def test_paired_timing_summary_applies_median_and_maximum_gates() -> None:
    summary = summarize_paired_timings(
        [
            {
                "pair_id": str(index),
                "order": (
                    "v6-r1_then_v6-r2" if index % 2 == 0 else "v6-r2_then_v6-r1"
                ),
                "baseline_seconds": 20.0,
                "candidate_seconds": candidate,
            }
            for index, candidate in enumerate((20.4, 20.5, 20.6, 20.7, 22.1))
        ]
    )
    assert summary["median_delta_seconds"] == pytest.approx(0.6)
    assert summary["median_limit_seconds"] == pytest.approx(1.0)
    assert summary["median_passed"] is True
    assert summary["maximum_passed"] is False


def test_verify_rejects_a_run_specific_asset_that_breaks_its_own_lineage(
    tmp_path: Path,
) -> None:
    root = _acceptance(tmp_path)
    manifest = next(
        (root / "round_a/slow_direct/generations").glob(
            "*/P2/source_corrections/manifest.json"
        )
    )
    _json(manifest, {"schema": "gemini305-video-s13-source-corrections/v1", "tampered": True})
    with pytest.raises(ValueError, match="lineage|asset hash mismatch"):
        verify(root)


def test_verify_refuses_to_seal_when_the_paired_timing_gate_fails(
    tmp_path: Path,
) -> None:
    root = _acceptance(tmp_path)
    path = root / "paired_timing_runs.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["pairs"][0]["candidate_seconds"] = 23.0
    _json(path, value)
    with pytest.raises(ValueError, match="paired timing performance gate"):
        verify(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("dirty", "clean committed worktree"),
        ("completion_binding", "explicit lineage"),
        ("dag_parent", "manifest DAG"),
        ("accepted_authority", "accepted correction authority"),
    ),
)
def test_verify_rejects_invalid_clean_lineage_dag_or_accepted_authority(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = _acceptance(tmp_path)
    run = root / "round_a/fast_direct"
    generation = next((run / "generations").iterdir())
    p2 = generation / "P2"
    if mutation == "dirty":
        path = generation / "generation_manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["algorithm"]["working_tree_dirty"] = True
        _json(path, value)
    elif mutation == "completion_binding":
        path = p2 / "P2_completion.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value.pop("source_map_oracle_manifest_sha256")
        _json(path, value)
    elif mutation == "dag_parent":
        path = p2 / "source_maps/manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["parent_source_correction_manifest_sha256"] = "0" * 64
        _json(path, value)
        completion = p2 / "P2_completion.json"
        completion_value = json.loads(completion.read_text(encoding="utf-8"))
        changed_sha = _sha(path)
        completion_value["source_map_oracle_manifest_sha256"] = changed_sha
        completion_value["assets_sha256"]["source_maps/manifest.json"] = changed_sha
        _json(completion, completion_value)
    else:
        path = p2 / "source_corrections/manifest.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["field_id_table"] = []
        value["contributors"] = []
        _json(path, value)
        _rebind_p2(p2)
    with pytest.raises(ValueError, match=message):
        verify(root)


def test_acceptance_summary_reports_branch_states_and_writes_compact_assets(
    tmp_path: Path,
) -> None:
    root = _acceptance(tmp_path)
    output = tmp_path / "summary"
    written = summarize_acceptance(root, output)
    assert {path.name for path in written} == {
        "acceptance_summary.json",
        "paired_timing_summary.json",
        "reproducible_seal.json",
    }
    summary = json.loads((output / "acceptance_summary.json").read_text())
    assert summary["schema"] == "gemini305-video-s13-m51-r4-acceptance-summary/v1"
    assert summary["matrix"] == {"branch_count": 4, "round_count": 2, "cell_count": 8}
    assert summary["all_p2_hard_passed"] is True
    assert summary["all_box_assets_present"] is True
    with pytest.raises(FileExistsError, match="new absent"):
        summarize_acceptance(root, output)
