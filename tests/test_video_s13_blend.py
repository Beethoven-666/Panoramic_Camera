from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_blend import (
    S13BlendConfig,
    adaptive_s13_multiband_levels,
    select_s13_blend_plans,
)
from panorama_demo.video_s13_photometric import S13PhotometricSampleSet
from panorama_demo.video_s13_replay import S13P2ReplayPair


def _pair() -> S13P2ReplayPair:
    shape = (8, 8)
    return S13P2ReplayPair(
        pair_index=0, left_source_index=0, right_source_index=1,
        left_frame_id=10, right_frame_id=11, corridor_x0=2, corridor_x1=10,
        seam_x_by_row=np.full(8, 6, np.int32),
        left_source_u=np.ones(shape, np.float32), left_source_v=np.ones(shape, np.float32),
        left_valid=np.ones(shape, bool), right_source_u=np.ones(shape, np.float32),
        right_source_v=np.ones(shape, np.float32), right_valid=np.ones(shape, bool),
        primary_owner_right_mask=np.broadcast_to(np.arange(2, 10)[None, :] >= 6, shape),
        geometry_transaction_numeric_id=0, seam_transaction_numeric_id=0,
        parent_pair_transaction_sha256="a" * 64,
    )


def _sample(protected: np.ndarray) -> S13PhotometricSampleSet:
    safe = ~protected
    empty = np.empty((0, 3), np.float32)
    xy = np.empty((0, 2), np.int32)
    return S13PhotometricSampleSet(
        pair_index=0, left_source_index=0, right_source_index=1,
        train_left_rgb_linear=empty, train_right_rgb_linear=empty,
        heldout_left_rgb_linear=empty, heldout_right_rgb_linear=empty,
        train_canvas_xy=xy, heldout_canvas_xy=xy,
        safe_mask=safe, protected_mask=protected, common_mask=np.ones_like(safe),
        safe_mask_sha256="a" * 64, protected_mask_sha256="b" * 64,
    )


def test_adaptive_levels_follow_m6_envelope() -> None:
    assert adaptive_s13_multiband_levels(2) == 0
    assert adaptive_s13_multiband_levels(4) == 1
    assert adaptive_s13_multiband_levels(8) == 2


def test_protected_pixels_are_owner_only() -> None:
    pair = _pair()
    protected = np.zeros((8, 8), bool)
    protected[:, 3:5] = True
    corrected = {0: np.full((8, 12, 3), 0.3, np.float32), 1: np.full((8, 12, 3), 0.31, np.float32)}
    plans, _ = select_s13_blend_plans(
        [pair], [_sample(protected)], corrected, canvas_shape=(8, 12),
        config=S13BlendConfig(minimum_safe_fraction_for_feather=0.0),
    )
    assert np.all(plans[0].secondary_weight[protected] == 0.0)
    assert np.max(plans[0].secondary_weight, initial=0.0) < 0.5


def test_force_owner_only_selects_b0() -> None:
    pair = _pair()
    corrected = {0: np.full((8, 12, 3), 0.3, np.float32), 1: np.full((8, 12, 3), 0.3, np.float32)}
    plans, _ = select_s13_blend_plans(
        [pair], [_sample(np.zeros((8, 8), bool))], corrected,
        canvas_shape=(8, 12), force_owner_only=True,
    )
    assert plans[0].transaction.model == "B0_owner_only"
    assert not np.any(plans[0].secondary_weight)
