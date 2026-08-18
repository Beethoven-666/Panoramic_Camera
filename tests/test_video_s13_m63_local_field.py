from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_m63_local_field import (
    build_s13_m63_local_log_gain_fields,
    evaluate_s13_m63_ql_shadow,
)
from panorama_demo.video_s13_replay import S13P2ReplayPair
from test_video_s13_m63_solver import _sample


def _pair(index: int, left: int, right: int, seam: int) -> S13P2ReplayPair:
    shape = (2, 8)
    zeros = np.zeros(shape, np.float32)
    valid = np.ones(shape, bool)
    return S13P2ReplayPair(
        pair_index=index,
        left_source_index=left,
        right_source_index=right,
        left_frame_id=left,
        right_frame_id=right,
        corridor_x0=0,
        corridor_x1=8,
        seam_x_by_row=np.full(2, seam, np.int32),
        left_source_u=zeros,
        left_source_v=zeros,
        left_valid=valid,
        right_source_u=zeros,
        right_source_v=zeros,
        right_valid=valid,
        primary_owner_right_mask=np.zeros(shape, bool),
        geometry_transaction_numeric_id=0,
        seam_transaction_numeric_id=0,
        parent_pair_transaction_sha256="a" * 64,
    )


def test_ql_shadow_accepts_stable_positive_and_negative_relations():
    samples = (
        _sample(0, 0, 1, 0.02, 0.0),
        _sample(1, 1, 2, 0.0, 0.02),
    )
    result = evaluate_s13_m63_ql_shadow(samples)
    assert result["active_pair_count"] == 2
    assert result["pairs"][0]["left_log_correction"] < 0.0
    assert result["pairs"][1]["left_log_correction"] > 0.0


def test_ql_cosine_field_is_continuous_clamped_and_order_independent():
    pairs = (_pair(0, 0, 1, 4), _pair(1, 1, 2, 4))
    relations = {0: 0.06, 1: -0.06}
    first = build_s13_m63_local_log_gain_fields(
        3, pairs, canvas_shape=(2, 9), pair_relations=relations, support_width_px=4
    )
    second = build_s13_m63_local_log_gain_fields(
        3, tuple(reversed(pairs)), canvas_shape=(2, 9), pair_relations=relations, support_width_px=4
    )
    for left, right in zip(first, second, strict=True):
        assert np.array_equal(left, right)
        assert np.max(np.abs(left)) <= 0.04
    assert first[0][0, 0] == 0.0
    assert first[2][0, 8] == 0.0
    assert np.max(np.abs(np.diff(first[1][0]))) < 0.04


def test_ql_excludes_protected_or_quality_cut_pair_from_shadow():
    sample = _sample(5, 0, 1, 0.02, 0.0)
    result = evaluate_s13_m63_ql_shadow(
        (sample,), excluded_pair_indices=frozenset({5})
    )
    assert result["active_pair_count"] == 0
    assert result["pairs"] == []
