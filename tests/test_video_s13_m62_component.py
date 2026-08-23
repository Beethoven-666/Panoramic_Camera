from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_m62_component import evaluate_s13_m62_q4c_shadow
from panorama_demo.video_s13_photometric import S13PhotometricSampleSet


def _edge(index: int, left_index: int, right_index: int, gain: float) -> S13PhotometricSampleSet:
    rng = np.random.default_rng(index + 20)
    left = rng.uniform(0.1, 0.6, (512, 3)).astype(np.float32)
    right = left / gain
    mask = np.ones((8, 64), bool)
    xy = np.stack((np.arange(512) % 64, np.arange(512) // 64), axis=1).astype(np.int32)
    return S13PhotometricSampleSet(
        pair_index=index, left_source_index=left_index, right_source_index=right_index,
        train_left_rgb_linear=left, train_right_rgb_linear=right,
        heldout_left_rgb_linear=left.copy(), heldout_right_rgb_linear=right.copy(),
        train_canvas_xy=xy, heldout_canvas_xy=xy.copy(), safe_mask=mask,
        protected_mask=np.zeros_like(mask), common_mask=mask,
        safe_mask_sha256="", protected_mask_sha256="", edge_eligible=True,
    )


def test_q4c_is_shadow_only_and_unsupported_cut_sources_are_exact_identity() -> None:
    report = evaluate_s13_m62_q4c_shadow(
        4, [_edge(0, 0, 1, 1.04), _edge(2, 2, 3, 1.03)], [1]
    )
    assert report["authority"] == "shadow"
    assert report["unsupported_cut_boundary_sources_identity"] is True
    assert report["cut_guard_changed_pixel_count"] == 0
    assert report["gain_minimum"] >= 0.94
    assert report["gain_maximum"] <= 1.06


def test_q4c_rolls_back_component_when_gain_is_out_of_bounds() -> None:
    report = evaluate_s13_m62_q4c_shadow(2, [_edge(0, 0, 1, 1.4)], [])
    assert report["nonidentity_source_count"] == 0
    assert report["components"][0]["accepted"] is False
    assert "component_gain_out_of_bounds" in report["components"][0]["rejection_reasons"]
