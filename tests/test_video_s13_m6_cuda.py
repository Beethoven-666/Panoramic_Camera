from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.video_s13_m6 import _formal_source_maps
from panorama_demo.video_s13_m6_cuda import build_s13_m6_source_rois
from panorama_demo.video_s13_replay import S13P2ReplayPair, S13VerifiedP2


def _p2(tmp_path: Path) -> S13VerifiedP2:
    shape = (4, 12)
    owner = np.broadcast_to((np.arange(shape[1]) >= 6).astype(np.int32), shape).copy()
    valid = np.ones(shape, bool)
    pair_shape = (4, 8)
    pair = S13P2ReplayPair(
        0, 0, 1, 10, 11, 2, 10, np.full(4, 6, np.int32),
        np.broadcast_to(np.arange(2, 10, dtype=np.float32), pair_shape).copy(),
        np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], pair_shape).copy(),
        np.ones(pair_shape, bool),
        np.broadcast_to(np.arange(2, 10, dtype=np.float32), pair_shape).copy(),
        np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], pair_shape).copy(),
        np.ones(pair_shape, bool),
        np.broadcast_to(np.arange(2, 10)[None, :] >= 6, pair_shape), 0, 0, "a" * 64,
    )
    provenance = {
        "owner_source_index": owner,
        "source_u": np.broadcast_to(np.arange(12, dtype=np.float32), shape).copy(),
        "source_v": np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], shape).copy(),
    }
    return S13VerifiedP2(
        tmp_path, {"source_count": 2}, "b" * 64,
        np.zeros((*shape, 3), np.uint8), valid, provenance, ({},), (pair,), {},
    )


def test_m6_compact_source_rois_match_full_source_maps(tmp_path: Path) -> None:
    p2 = _p2(tmp_path)
    rois = build_s13_m6_source_rois(p2, (10, 11))
    assert [(roi.x0, roi.x1) for roi in rois] == [(0, 10), (2, 12)]
    for roi in rois:
        full_u, full_v, full_mapped = _formal_source_maps(p2, roi.source_index)
        local = np.s_[:, roi.x0:roi.x1]
        assert np.array_equal(roi.map_u, full_u[local])
        assert np.array_equal(roi.map_v, full_v[local])
        assert np.array_equal(roi.mapped, full_mapped[local])
        expected_owner = p2.valid_mask[local] & (
            p2.provenance["owner_source_index"][local] == roi.source_index
        )
        assert np.array_equal(roi.owner_mask, expected_owner)


def test_m6_compact_source_rois_reject_conflicting_replay_maps(tmp_path: Path) -> None:
    p2 = _p2(tmp_path)
    pair = p2.replay_pairs[0]
    conflicting = S13P2ReplayPair(
        pair.pair_index, pair.left_source_index, pair.right_source_index,
        pair.left_frame_id, pair.right_frame_id, pair.corridor_x0, pair.corridor_x1,
        pair.seam_x_by_row, pair.left_source_u + 0.1, pair.left_source_v, pair.left_valid,
        pair.right_source_u, pair.right_source_v, pair.right_valid,
        pair.primary_owner_right_mask, pair.geometry_transaction_numeric_id,
        pair.seam_transaction_numeric_id, pair.parent_pair_transaction_sha256,
    )
    broken = S13VerifiedP2(
        p2.root, p2.completion, p2.completion_sha256, p2.result_image, p2.valid_mask,
        p2.provenance, p2.transactions, (pair, conflicting), p2.immutable_sha256,
    )
    try:
        build_s13_m6_source_rois(broken, (10, 11))
    except ValueError as exc:
        assert "conflicting source sampling" in str(exc)
    else:
        raise AssertionError("conflicting compact replay maps must fail")
