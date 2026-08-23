from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_c2e import select_s13_fast_c2e
from panorama_demo.video_s13_m5 import S13M5Pair
from panorama_demo.video_s13_replay import S13P2ReplayPair


def _pair() -> S13P2ReplayPair:
    shape = (4, 6)
    columns = np.arange(2, 8, dtype=np.float32)
    return S13P2ReplayPair(
        0, 0, 1, 10, 11, 2, 8, np.full(4, 5, np.int32),
        np.broadcast_to(columns, shape).copy(),
        np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], shape).copy(),
        np.ones(shape, bool),
        np.broadcast_to(columns, shape).copy(),
        np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], shape).copy(),
        np.ones(shape, bool),
        np.broadcast_to(columns[None, :] >= 5, shape), 0, 0, "a" * 64,
    )


def test_fast_c2e_enters_all_pairs_but_keeps_c0_without_risk() -> None:
    pair = _pair()
    images = {10: np.full((4, 10, 3), 20, np.uint8), 11: np.full((4, 10, 3), 20, np.uint8)}
    m5 = S13M5Pair({"selection_continued_for_structure": False}, pair.seam_x_by_row, None)
    selected, labels, owner_only = select_s13_fast_c2e((m5,), (pair,), images.__getitem__)
    assert labels == ("C0_keep_standard",)
    assert owner_only == frozenset()
    assert selected[0] is pair


def test_fast_c2e_uses_pair_local_c1_for_unresolved_structure() -> None:
    pair = _pair()
    images = {10: np.full((4, 10, 3), 20, np.uint8), 11: np.full((4, 10, 3), 20, np.uint8)}
    m5 = S13M5Pair({"unresolved_oblique_structure": True}, pair.seam_x_by_row, None)
    selected, labels, owner_only = select_s13_fast_c2e((m5,), (pair,), images.__getitem__)
    assert selected == (pair,)
    assert labels == ("C1_owner_only",)
    assert owner_only == frozenset({0})


def test_fast_c2e_rejects_nonmonotone_cached_seam() -> None:
    pair = _pair()
    images = {10: np.full((4, 10, 3), 20, np.uint8), 11: np.full((4, 10, 3), 20, np.uint8)}
    invalid = np.asarray([3, 6, 3, 6], dtype=np.int32)
    m5 = S13M5Pair(
        {"selection_continued_for_structure": True}, pair.seam_x_by_row, None,
        c2e_seam_candidates=(invalid,),
    )
    selected, labels, _ = select_s13_fast_c2e((m5,), (pair,), images.__getitem__)
    assert selected == (pair,)
    assert labels == ("C0_keep_standard",)
