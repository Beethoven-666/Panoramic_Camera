from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from panorama_demo.video_s13_m61_evidence import (
    GRAPH_TOPOLOGY_SCHEMA,
    S13PhotometricEvidenceConfig,
    _split_roles,
    build_s13_m61_evidence,
)
from panorama_demo.video_s13_replay import S13P2ReplayPair, S13VerifiedP2


def _verified(tmp_path: Path) -> S13VerifiedP2:
    height, width = 32, 48
    owner = np.broadcast_to((np.arange(width) >= 24).astype(np.int32), (height, width)).copy()
    valid = np.ones((height, width), bool)
    provenance = {
        "owner_source_index": owner,
        "owner_frame_id": np.where(owner == 0, 10, 11).astype(np.int32),
        "source_u": np.broadcast_to(np.arange(width, dtype=np.float32), valid.shape).copy(),
        "source_v": np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], valid.shape).copy(),
        "valid": valid,
    }
    pair_shape = (height, 16)
    x = np.broadcast_to(np.arange(16, 32, dtype=np.float32), pair_shape).copy()
    y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], pair_shape).copy()
    pair = S13P2ReplayPair(
        0, 0, 1, 10, 11, 16, 32, np.full(height, 24, np.int32),
        x, y, np.ones(pair_shape, bool), x, y, np.ones(pair_shape, bool),
        x >= 24, 0, 0, "a" * 64,
    )
    assets = {
        "pair_replay/pair_0000.npz": "b" * 64,
        "p2_pixel_provenance.npz": "c" * 64,
        "pair_transactions.json": "d" * 64,
        "p2_replay_manifest.json": "e" * 64,
    }
    return S13VerifiedP2(
        tmp_path, {"source_count": 2, "generation_id": "g", "assets_sha256": assets,
                   "parent_completion_sha256": "f" * 64,
                   "result_asset_sha256": "1" * 64}, "2" * 64,
        np.full((height, width, 3), 80, np.uint8), valid, provenance, ({},), (pair,), {},
    )


def test_prepare_evidence_uses_only_intersected_sealed_support(tmp_path: Path) -> None:
    p2 = _verified(tmp_path)
    images = {10: np.full((32, 48, 3), 80, np.uint8),
              11: np.full((32, 48, 3), 82, np.uint8)}
    config = S13PhotometricEvidenceConfig(
        topology_minimum_train_samples=1, topology_minimum_train_tiles=1,
        topology_minimum_vertical_blocks=1,
    )
    manifest = build_s13_m61_evidence(
        p2, tmp_path / "out", branch="fast_direct", run_id="run", generation_id="g",
        frame_image_loader=images.__getitem__, raw_rgb_sha256={10: "3" * 64, 11: "4" * 64},
        config=config,
    )
    with np.load(tmp_path / "out/photometric_replay/pair_0000.npz", allow_pickle=False) as stored:
        support = stored["common_audited_support"]
        assert support.shape[1] == 48  # 64 px shoulder is clipped to the canvas.
        assert np.count_nonzero(support) == 32 * 16
        assert not np.any(support[:, :16])
        assert not np.any(support[:, 32:])
        assert not np.any(stored["train"] & stored["validation"])
        assert not np.any(stored["train"] & stored["excluded_guard"])
        assert np.any(stored["train"] & stored["common_audited_support"])
        # Output-quality coverage spans the canonical 64 px shoulder even
        # where the old narrow replay has no bilateral UV.  Solver evidence
        # remains strictly confined to that audited intersection.
        assert np.any(stored["quality_output_safe"] & ~support)
        assert not np.any(stored["solver_matched_safe"] & ~support)
    topology = json.loads(
        (tmp_path / "out/photometric_replay/graph_topology/topology.json").read_text()
    )
    assert topology["schema"] == GRAPH_TOPOLOGY_SCHEMA
    assert topology["train_only"] is True
    assert topology["contains_candidate_parameters"] is False
    assert manifest["candidate_solver_invocations"] == 0


def test_canvas_tile_split_is_global_and_leaves_train_interior() -> None:
    config = S13PhotometricEvidenceConfig()
    y = np.broadcast_to(np.arange(160, dtype=np.int32)[:, None], (160, 160))
    x = np.broadcast_to(np.arange(160, dtype=np.int32), (160, 160))
    whole = _split_roles(x, y, config)
    crop = _split_roles(x[24:136, 31:129], y[24:136, 31:129], config)
    for whole_role, crop_role in zip(whole, crop, strict=True):
        assert np.array_equal(whole_role[24:136, 31:129], crop_role)
    assert np.any(whole[0])
    assert np.any(whole[1])
    assert np.any(whole[2])


def test_evidence_config_rejects_unknown_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="unknown"):
        S13PhotometricEvidenceConfig.from_mapping({"unused": 1})
    with pytest.raises(ValueError, match="finite"):
        S13PhotometricEvidenceConfig.from_mapping({"safe_gradient_max": float("nan")})
    with pytest.raises(ValueError, match="16 px"):
        S13PhotometricEvidenceConfig.from_mapping({"tile_size_px": 8})


def test_single_source_component_remains_identity_boundary(tmp_path: Path) -> None:
    p2 = _verified(tmp_path)
    pair = p2.replay_pairs[0]
    broken = S13P2ReplayPair(
        pair.pair_index, pair.left_source_index, pair.right_source_index,
        pair.left_frame_id, pair.right_frame_id, pair.corridor_x0, pair.corridor_x1,
        pair.seam_x_by_row, pair.left_source_u, pair.left_source_v,
        np.zeros_like(pair.left_valid), pair.right_source_u, pair.right_source_v,
        pair.right_valid, pair.primary_owner_right_mask, 0, 0, pair.parent_pair_transaction_sha256,
    )
    p2 = S13VerifiedP2(
        p2.root, p2.completion, p2.completion_sha256, p2.result_image, p2.valid_mask,
        p2.provenance, p2.transactions, (broken,), p2.immutable_sha256,
    )
    images = {10: np.full((32, 48, 3), 80, np.uint8),
              11: np.full((32, 48, 3), 82, np.uint8)}
    build_s13_m61_evidence(
        p2, tmp_path / "out", branch="b", run_id="r", generation_id="g",
        frame_image_loader=images.__getitem__, raw_rgb_sha256={10: "3" * 64, 11: "4" * 64},
        config=S13PhotometricEvidenceConfig(),
    )
    topology = json.loads(
        (tmp_path / "out/photometric_replay/graph_topology/topology.json").read_text()
    )
    assert [item["single_source_identity_boundary"] for item in topology["components"]] == [True, True]
