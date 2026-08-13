from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from panorama_demo.video_s13_m51_r4_component_chain import (
    source_map_oracle_from_arrays,
)
from panorama_demo.video_s13_v6_r2_verifier import (
    canonical_pair_transaction_sha256,
    canonical_source_map_slice_sha256,
    verify_s13_v6_r2_p2,
)


ALGORITHM_ID = "S013_output_first_progressive_dense_central_slit_v6"
IMPLEMENTATION_ID = (
    "s013_output_first_progressive_dense_central_slit_m51_r4_component_chain_c2e"
)
CONFIG_SHA = "1" * 64
MANIFEST_SHA = "2" * 64
SOURCE_COMMIT = "3" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _refresh_completion_assets(p2: Path) -> None:
    completion_path = p2 / "P2_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["assets_sha256"] = {
        path.relative_to(p2).as_posix(): _sha(path)
        for path in sorted(p2.rglob("*"))
        if path.is_file() and path != completion_path
    }
    _json(completion_path, completion)


def _build_fixture(tmp_path: Path) -> Path:
    generation = tmp_path / "generation"
    p2 = generation / "P2"
    p2.mkdir(parents=True)
    generation_manifest = {
        "schema": "gemini305-video-s13-generation/v1",
        "generation_id": "test-generation",
        "algorithm": {
            "algorithm_id": ALGORITHM_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "config_sha256": CONFIG_SHA,
            "candidate_manifest_sha256": MANIFEST_SHA,
            "source_commit": SOURCE_COMMIT,
            "working_tree_dirty": False,
        },
    }
    _json(generation / "generation_manifest.json", generation_manifest)

    height, width = 2, 4
    canvas_u = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    canvas_v = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None], (height, width)
    )
    valid = np.ones((height, width), dtype=np.uint8)
    labels = np.full((height, width), -1, dtype=np.int32)
    oracles = [
        source_map_oracle_from_arrays(
            source_index=index,
            domain_xyxy=(0, 0, width, height),
            u=canvas_u,
            v=canvas_v,
            valid=valid,
            field_id=labels,
        )
        for index in range(2)
    ]
    source_rows = []
    for oracle in oracles:
        asset = f"source_maps/source_{oracle.source_index:04d}.npz"
        _npz(
            p2 / asset,
            source_index=np.asarray(oracle.source_index, np.int32),
            domain_xyxy=np.asarray(oracle.domain_xyxy, np.int32),
            u=oracle.u,
            v=oracle.v,
            valid=oracle.valid,
            field_id=oracle.field_id,
        )
        source_rows.append(
            {
                "source_index": oracle.source_index,
                "domain_xyxy": list(oracle.domain_xyxy),
                "asset": asset,
                "asset_sha256": _sha(p2 / asset),
                "oracle_sha256": oracle.oracle_sha256,
            }
        )

    component = {
        "schema": "gemini305-video-s13-component-chain-transactions/v1",
        "application_state": "none",
        "repair_complete": False,
        "segments": [],
        "accepted_segment_ids": [],
        "rejected_segment_ids": [],
        "deferred_segment_ids": [],
    }
    _json(p2 / "component_chain_transactions/manifest.json", component)
    component_sha = _sha(p2 / "component_chain_transactions/manifest.json")
    corrections = {
        "schema": "gemini305-video-s13-source-corrections/v1",
        "parent_component_transaction_manifest_sha256": component_sha,
        "application_state": "none",
        "field_id_table": [],
        "sources": [
            {
                "source_index": row["source_index"],
                "source_map_oracle_sha256": row["oracle_sha256"],
            }
            for row in source_rows
        ],
    }
    _json(p2 / "source_corrections/manifest.json", corrections)
    corrections_sha = _sha(p2 / "source_corrections/manifest.json")
    source_maps = {
        "schema": "gemini305-video-s13-source-map-oracles/v1",
        "parent_source_correction_manifest_sha256": corrections_sha,
        "source_count": 2,
        "sources": source_rows,
    }
    _json(p2 / "source_maps/manifest.json", source_maps)
    source_maps_sha = _sha(p2 / "source_maps/manifest.json")

    pair = {
        "schema": "gemini305-video-s13-m5-pair-transaction/v5",
        "transaction_id": "m5-pair-0000",
        "pair_frame_ids": [10, 11],
        "decision": "rolled_back",
        "component_chain_c2e": {
            "schema": "gemini305-video-s13-component-chain-c2e-pair-ref/v1",
            "application_state": "none",
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": corrections_sha,
            "source_map_oracle_manifest_sha256": source_maps_sha,
            "left_source_map_oracle_sha256": oracles[0].oracle_sha256,
            "right_source_map_oracle_sha256": oracles[1].oracle_sha256,
            "refs": [],
        },
    }
    pair["result_stage_sha256"] = canonical_pair_transaction_sha256(pair)
    pair_asset = "pair_transactions/pair_0000.json"
    _json(p2 / pair_asset, pair)
    pair_sha = _sha(p2 / pair_asset)
    aggregate = {
        "schema": "gemini305-video-s13-m5-pair-transactions/v4",
        "all_pairs_reported": True,
        "component_transaction_manifest_sha256": component_sha,
        "source_correction_manifest_sha256": corrections_sha,
        "source_map_oracle_manifest_sha256": source_maps_sha,
        "pair_transaction_assets": [
            {"pair_index": 0, "asset": pair_asset, "sha256": pair_sha}
        ],
        "pairs": [pair],
    }
    _json(p2 / "pair_transactions.json", aggregate)
    aggregate_sha = _sha(p2 / "pair_transactions.json")

    slice_bbox = (0, 0, width, height)
    left_slice_sha = canonical_source_map_slice_sha256(oracles[0], slice_bbox)
    right_slice_sha = canonical_source_map_slice_sha256(oracles[1], slice_bbox)
    replay_asset = "pair_replay/pair_0000.npz"
    owner_right = np.broadcast_to(
        np.arange(width, dtype=np.int32)[None, :] >= 2, (height, width)
    )
    _npz(
        p2 / replay_asset,
        pair_index=np.asarray(0, np.int32),
        left_source_index=np.asarray(0, np.int32),
        right_source_index=np.asarray(1, np.int32),
        left_frame_id=np.asarray(10, np.int32),
        right_frame_id=np.asarray(11, np.int32),
        corridor_x0=np.asarray(0, np.int32),
        corridor_x1=np.asarray(width, np.int32),
        seam_x_by_row=np.full(height, 2, np.int32),
        left_source_u=oracles[0].u,
        left_source_v=oracles[0].v,
        left_valid=oracles[0].valid.astype(bool),
        right_source_u=oracles[1].u,
        right_source_v=oracles[1].v,
        right_valid=oracles[1].valid.astype(bool),
        left_component_correction_field_id=oracles[0].field_id,
        right_component_correction_field_id=oracles[1].field_id,
        primary_owner_right_mask=owner_right,
        geometry_transaction_numeric_id=np.asarray(0, np.int32),
        seam_transaction_numeric_id=np.asarray(0, np.int32),
        parent_pair_transaction_sha256=np.asarray(pair_sha),
    )
    replay = {
        "schema": "gemini305-video-s13-p2-replay/v2",
        "pair_count": 1,
        "source_count": 2,
        "component_transaction_manifest_sha256": component_sha,
        "source_correction_manifest_sha256": corrections_sha,
        "source_map_oracle_manifest_sha256": source_maps_sha,
        "aggregate_pair_transaction_manifest_sha256": aggregate_sha,
        "pairs": [
            {
                "pair_index": 0,
                "asset": replay_asset,
                "parent_pair_transaction": pair_asset,
                "parent_pair_transaction_sha256": pair_sha,
                "left_source_map_oracle_sha256": oracles[0].oracle_sha256,
                "right_source_map_oracle_sha256": oracles[1].oracle_sha256,
                "left_source_map_slice_sha256": left_slice_sha,
                "right_source_map_slice_sha256": right_slice_sha,
                "relevant_segment_transaction_sha256": [],
            }
        ],
    }
    _json(p2 / "p2_replay_manifest.json", replay)
    replay_sha = _sha(p2 / "p2_replay_manifest.json")

    owner_source = np.broadcast_to(
        np.where(np.arange(width) < 2, 0, 1)[None, :], (height, width)
    ).astype(np.int32)
    provenance = {
        "owner_source_index": owner_source,
        "owner_frame_id": np.where(owner_source == 0, 10, 11).astype(np.int32),
        "source_u": canvas_u,
        "source_v": canvas_v,
        "valid": np.ones((height, width), dtype=bool),
        "component_correction_field_id": labels,
    }
    _npz(p2 / "p2_pixel_provenance.npz", **provenance)
    _json(p2 / "hard_audit.json", {"passed": True})

    completion = {
        "schema": "gemini305-video-s13-p2-completion/v6-r2",
        "sealed": True,
        "stage": "P2",
        "generation_id": "test-generation",
        "algorithm_id": ALGORITHM_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "contract_schema": "gemini305-video-s13-output-first/v6-r2",
        "config_sha256": CONFIG_SHA,
        "candidate_manifest_sha256": MANIFEST_SHA,
        "generation_manifest_sha256": _sha(generation / "generation_manifest.json"),
        "source_commit": SOURCE_COMMIT,
        "working_tree_dirty": False,
        "hard_audit_passed": True,
        "m6_eligible": False,
        "application_state": "none",
        "repair_complete": False,
        "provenance_schema": "gemini305-video-s13-p2-provenance/v6-r2",
        "p2_replay_schema": "gemini305-video-s13-p2-replay/v2",
        "component_transaction_manifest_sha256": component_sha,
        "source_correction_manifest_sha256": corrections_sha,
        "source_map_oracle_manifest_sha256": source_maps_sha,
        "aggregate_pair_transaction_manifest_sha256": aggregate_sha,
        "p2_replay_manifest_sha256": replay_sha,
    }
    _json(p2 / "P2_completion.json", completion)
    _refresh_completion_assets(p2)
    return p2


def test_v6_r2_semantic_verifier_accepts_complete_noop_lineage(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    result = verify_s13_v6_r2_p2(p2)
    assert result["passed"] is True
    assert result["pair_count"] == 1
    assert result["source_count"] == 2


def test_v6_r2_verifier_requires_completion_identity(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    completion = json.loads((p2 / "P2_completion.json").read_text(encoding="utf-8"))
    completion.pop("implementation_id")
    _json(p2 / "P2_completion.json", completion)
    with pytest.raises(ValueError, match="identity"):
        verify_s13_v6_r2_p2(p2)


def test_v6_r2_verifier_rejects_application_state_lineage_tamper(
    tmp_path: Path,
) -> None:
    p2 = _build_fixture(tmp_path)
    completion_path = p2 / "P2_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["application_state"] = "complete"
    completion_path.write_text(json.dumps(completion), encoding="utf-8")

    with pytest.raises(ValueError, match="application state lineage"):
        verify_s13_v6_r2_p2(p2)


def test_v6_r2_verifier_rejects_pair_digest_tamper(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    pair_path = p2 / "pair_transactions/pair_0000.json"
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    pair["decision"] = "applied"
    _json(pair_path, pair)
    aggregate_path = p2 / "pair_transactions.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["pair_transaction_assets"][0]["sha256"] = _sha(pair_path)
    aggregate["pairs"][0] = pair
    _json(aggregate_path, aggregate)
    aggregate_sha = _sha(aggregate_path)
    replay_path = p2 / "p2_replay_manifest.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["aggregate_pair_transaction_manifest_sha256"] = aggregate_sha
    _json(replay_path, replay)
    completion_path = p2 / "P2_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["aggregate_pair_transaction_manifest_sha256"] = aggregate_sha
    completion["p2_replay_manifest_sha256"] = _sha(replay_path)
    _json(completion_path, completion)
    _refresh_completion_assets(p2)
    with pytest.raises(ValueError, match="digest"):
        verify_s13_v6_r2_p2(p2)


def test_v6_r2_verifier_rejects_aggregate_pair_sha_tamper(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    aggregate_path = p2 / "pair_transactions.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["pair_transaction_assets"][0]["sha256"] = "0" * 64
    _json(aggregate_path, aggregate)
    completion = json.loads((p2 / "P2_completion.json").read_text(encoding="utf-8"))
    completion["aggregate_pair_transaction_manifest_sha256"] = _sha(aggregate_path)
    _json(p2 / "P2_completion.json", completion)
    _refresh_completion_assets(p2)
    with pytest.raises(ValueError, match="pair transaction asset"):
        verify_s13_v6_r2_p2(p2)


def test_v6_r2_verifier_rejects_replay_oracle_tamper(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    replay_path = p2 / "pair_replay/pair_0000.npz"
    with np.load(replay_path, allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name], copy=True) for name in stored.files}
    arrays["left_source_u"][0, 0] += 1.0
    _npz(replay_path, **arrays)
    _refresh_completion_assets(p2)
    with pytest.raises(ValueError, match="oracle"):
        verify_s13_v6_r2_p2(p2)


def test_v6_r2_verifier_rejects_unknown_provenance_field_id(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    provenance_path = p2 / "p2_pixel_provenance.npz"
    with np.load(provenance_path, allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name], copy=True) for name in stored.files}
    arrays["component_correction_field_id"][0, 0] = 99
    _npz(provenance_path, **arrays)
    _refresh_completion_assets(p2)
    with pytest.raises(ValueError, match="field table"):
        verify_s13_v6_r2_p2(p2)


def test_v6_r2_verifier_rejects_reverse_dag_binding(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    component_path = p2 / "component_chain_transactions/manifest.json"
    component = json.loads(component_path.read_text(encoding="utf-8"))
    component["audit"] = {"p2_replay_manifest_sha256": "0" * 64}
    _json(component_path, component)
    component_sha = _sha(component_path)
    corrections_path = p2 / "source_corrections/manifest.json"
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    corrections["parent_component_transaction_manifest_sha256"] = component_sha
    _json(corrections_path, corrections)
    corrections_sha = _sha(corrections_path)
    source_maps_path = p2 / "source_maps/manifest.json"
    source_maps = json.loads(source_maps_path.read_text(encoding="utf-8"))
    source_maps["parent_source_correction_manifest_sha256"] = corrections_sha
    _json(source_maps_path, source_maps)
    source_maps_sha = _sha(source_maps_path)
    for path in (p2 / "pair_transactions.json", p2 / "p2_replay_manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["component_transaction_manifest_sha256"] = component_sha
        payload["source_correction_manifest_sha256"] = corrections_sha
        payload["source_map_oracle_manifest_sha256"] = source_maps_sha
        if "pairs" in payload and path.name == "pair_transactions.json":
            payload["pairs"][0]["component_chain_c2e"].update(
                {
                    "component_transaction_manifest_sha256": component_sha,
                    "source_correction_manifest_sha256": corrections_sha,
                    "source_map_oracle_manifest_sha256": source_maps_sha,
                }
            )
        _json(path, payload)
    completion_path = p2 / "P2_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.update(
        {
            "component_transaction_manifest_sha256": component_sha,
            "source_correction_manifest_sha256": corrections_sha,
            "source_map_oracle_manifest_sha256": source_maps_sha,
            "aggregate_pair_transaction_manifest_sha256": _sha(
                p2 / "pair_transactions.json"
            ),
            "p2_replay_manifest_sha256": _sha(p2 / "p2_replay_manifest.json"),
        }
    )
    _json(completion_path, completion)
    _refresh_completion_assets(p2)
    with pytest.raises(ValueError, match="reverse DAG"):
        verify_s13_v6_r2_p2(p2)


def test_v6_r2_verifier_rejects_replay_slice_sha_tamper(tmp_path: Path) -> None:
    p2 = _build_fixture(tmp_path)
    replay_path = p2 / "p2_replay_manifest.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["pairs"][0]["left_source_map_slice_sha256"] = "0" * 64
    _json(replay_path, replay)
    completion_path = p2 / "P2_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["p2_replay_manifest_sha256"] = _sha(replay_path)
    _json(completion_path, completion)
    _refresh_completion_assets(p2)
    with pytest.raises(ValueError, match="slice SHA"):
        verify_s13_v6_r2_p2(p2)
