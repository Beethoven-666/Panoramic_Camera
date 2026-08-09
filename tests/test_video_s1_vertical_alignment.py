from __future__ import annotations

import cv2
import numpy as np

import panorama_demo.video_s1_vertical_alignment as alignment
from panorama_demo.video_s1_vertical_alignment import (
    VerticalAlignmentConfig,
    VerticalCandidate,
    align_vertical_pair,
    fit_ransac_soft_prior,
    fit_smooth_vertical_curve,
    generate_vertical_candidates,
    precompute_feature_cache,
    precompute_source_features,
    select_local_gains,
    solve_candidate_path,
    solve_vertical_curve,
)


def _textured_pair(*, dy: int = 2, height: int = 192, width: int = 240):
    rng = np.random.default_rng(14)
    left = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(8, height, 16):
        cv2.line(left, (0, y), (width - 1, y), (80 + y % 120, 130, 200), 2)
    for x in (31, 87, 153, 211):
        cv2.line(left, (x, 0), (x, height - 1), (220, 90 + x % 100, 45), 3)
    left = cv2.add(left, rng.integers(0, 25, left.shape, dtype=np.uint8))
    right = cv2.warpAffine(
        left,
        np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, float(dy)]]),
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return left, right


def _candidate(window: int, dy: float, cost: float, confidence: float = 0.8, *, missing=False):
    return VerticalCandidate(
        window_index=window,
        center_y=16.0 + 16.0 * window,
        dy_px=dy,
        raw_cost=cost,
        normalized_cost=cost,
        luminance_cost=cost,
        vertical_gradient_cost=cost,
        color_cost=cost,
        valid_fraction=0.9,
        texture_score=0.8,
        left_right_error_px=0.0 if not missing else float("inf"),
        multi_x_consensus_error_px=0.0 if not missing else float("inf"),
        confidence=confidence if not missing else 0.0,
        is_missing=missing,
    )


def test_feature_cache_computes_each_real_source_once(monkeypatch):
    image, _ = _textured_pair(height=64, width=96)
    calls = 0
    original = alignment.precompute_source_features

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(alignment, "precompute_source_features", counted)
    cache = precompute_feature_cache({10: image, 11: image, 12: image}, horizontal_scale=0.5)

    assert calls == 3
    assert set(cache) == {10, 11, 12}
    assert cache[10].gray.shape == (64, 48)
    assert cache[10].lab.shape == (64, 48, 3)
    assert cache[10].valid_mask.dtype == np.bool_


def test_top_k_nms_multi_subwindow_and_lr_confidence_recover_vertical_shift():
    left_image, right_image = _textured_pair(dy=2)
    config = VerticalAlignmentConfig(
        search_radius_y_px=8,
        window_height_px=32,
        window_step_px=16,
        top_k_candidates=5,
        candidate_nms_radius_px=2,
    )
    left = precompute_source_features(left_image, horizontal_scale=0.5)
    right = precompute_source_features(right_image, horizontal_scale=0.5)

    windows = generate_vertical_candidates(left, right, config)
    real = [item for item in windows[len(windows) // 2] if not item.is_missing]

    assert any(abs(item.dy_px - 2.0) < 0.25 for item in real)
    integer_peaks = sorted(round(item.dy_px) for item in real)
    assert all(second - first > 2 for first, second in zip(integer_peaks, integer_peaks[1:]))
    best = max(real, key=lambda item: item.confidence)
    assert abs(best.dy_px - 2.0) < 0.25
    assert best.left_right_error_px < 0.25
    assert best.multi_x_consensus_error_px < 0.25
    assert best.confidence > 0.1


def test_textureless_or_invalid_windows_keep_explicit_missing_and_zero_curve():
    image = np.full((96, 160, 3), 127, dtype=np.uint8)
    invalid = np.zeros((96, 160), dtype=bool)
    left = precompute_source_features(image, invalid, horizontal_scale=0.5)
    right = precompute_source_features(image, invalid, horizontal_scale=0.5)

    result = align_vertical_pair(left, right, VerticalAlignmentConfig(search_radius_y_px=4))

    assert result.pair_status == "s1_zero_correction"
    assert result.candidates_by_window
    assert all(len(window) == 1 and window[0].is_missing for window in result.candidates_by_window)
    assert np.array_equal(result.curve.dy_applied, np.zeros(96, dtype=np.float32))


def test_ransac_is_soft_and_dp_prefers_continuous_path_over_periodic_distractor():
    config = VerticalAlignmentConfig(
        ransac_minimum_points=6,
        dp_first_order_weight=1.0,
        dp_second_order_weight=0.8,
        dp_missing_cost=1.25,
    )
    windows = []
    for index in range(9):
        correct = 1.0 + 0.08 * index
        distractor = correct + (12.0 if index % 2 else -12.0)
        windows.append(
            (
                _candidate(index, correct, 0.18, 0.8),
                _candidate(index, distractor, 0.05, 0.45),
                _candidate(index, 0.0, config.dp_missing_cost, missing=True),
            )
        )

    prior = fit_ransac_soft_prior(windows, config)
    selected = solve_candidate_path(windows, prior, config)

    assert prior.available
    assert prior.inlier_count >= 6
    assert all(not item.is_missing for item in selected)
    assert np.max(np.abs(np.diff([item.dy_px for item in selected]))) < 0.2


def test_dp_uses_local_missing_without_changing_other_windows():
    config = VerticalAlignmentConfig(dp_missing_cost=0.4)
    windows = []
    for index in range(7):
        if index == 3:
            windows.append((_candidate(index, 0.0, config.dp_missing_cost, missing=True),))
        else:
            windows.append(
                (
                    _candidate(index, 2.0, 0.05, 0.9),
                    _candidate(index, 0.0, config.dp_missing_cost, missing=True),
                )
            )

    selected, _, curve = solve_vertical_curve(windows, 128, config)

    assert selected[3].is_missing
    assert all(not item.is_missing for index, item in enumerate(selected) if index != 3)
    assert abs(float(np.median(curve.dy_smoothed[24:104])) - 2.0) < 0.35
    assert np.isfinite(curve.dy_smoothed).all()


def test_smoothing_returns_to_zero_and_limits_local_slope():
    config = VerticalAlignmentConfig(
        smoothing_control_spacing_px=32,
        maximum_local_slope=0.08,
    )
    selected = (
        _candidate(0, 4.0, 0.1, 1.0),
        VerticalCandidate(**{**_candidate(1, 4.0, 0.1, 1.0).__dict__, "center_y": 48.0}),
    )

    curve = fit_smooth_vertical_curve(selected, 256, config)

    assert abs(float(curve.dy_smoothed[-1])) < abs(float(curve.dy_smoothed[48]))
    assert np.max(np.abs(np.diff(curve.dy_smoothed))) <= config.maximum_local_slope + 1e-5
    assert np.isfinite(curve.dy_smoothed).all()


def test_gain_selection_chooses_correction_and_full_alignment_is_stable():
    rng = np.random.default_rng(7)
    left_image = cv2.GaussianBlur(
        rng.integers(0, 256, (192, 240, 3), dtype=np.uint8), (3, 3), 0
    )
    right_image = cv2.warpAffine(
        left_image,
        np.float32([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]]),
        (240, 192),
        borderMode=cv2.BORDER_CONSTANT,
    )
    config = VerticalAlignmentConfig(
        search_radius_y_px=5,
        gain_smoothness_weight=0.02,
    )
    left = precompute_source_features(left_image, horizontal_scale=0.5)
    right = precompute_source_features(right_image, horizontal_scale=0.5)
    result = align_vertical_pair(left, right, config, select_gains=False)

    row_gain, block_gains = select_local_gains(left, right, result.curve, config)

    assert result.pair_status == "s1_ok"
    assert abs(float(np.median(result.curve.dy_smoothed[32:-32])) - 2.0) < 0.25
    assert block_gains and np.median(block_gains) >= 0.5
    assert np.isfinite(row_gain).all()

