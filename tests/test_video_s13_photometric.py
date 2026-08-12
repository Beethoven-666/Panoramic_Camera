from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_photometric import (
    S13PhotometricConfig,
    S13PhotometricSampleSet,
    solve_s13_photometric,
)


def _samples(left: np.ndarray, right: np.ndarray) -> S13PhotometricSampleSet:
    count = len(left)
    mask = np.ones((1, count), dtype=bool)
    coordinates = np.stack((np.arange(count), np.zeros(count)), axis=1).astype(np.int32)
    return S13PhotometricSampleSet(
        pair_index=0,
        left_source_index=0,
        right_source_index=1,
        train_left_rgb_linear=left,
        train_right_rgb_linear=right,
        heldout_left_rgb_linear=left.copy(),
        heldout_right_rgb_linear=right.copy(),
        train_canvas_xy=coordinates,
        heldout_canvas_xy=coordinates.copy(),
        safe_mask=mask,
        protected_mask=np.zeros_like(mask),
        common_mask=mask,
        safe_mask_sha256="a" * 64,
        protected_mask_sha256="b" * 64,
    )


def test_identity_is_always_available_and_selected_for_equal_sources() -> None:
    rng = np.random.default_rng(4)
    values = rng.uniform(0.1, 0.7, (256, 3)).astype(np.float32)
    solution = solve_s13_photometric(
        [_samples(values, values)], frame_ids=[10, 11],
        config=S13PhotometricConfig(minimum_pair_sample_count=32),
    )
    assert solution.model_family == "Q0_identity"
    assert np.allclose([p.gain_bgr for p in solution.source_parameters], 1.0)


def test_known_scalar_exposure_is_solved_globally_in_linear_light() -> None:
    rng = np.random.default_rng(5)
    left = rng.uniform(0.1, 0.6, (512, 3)).astype(np.float32)
    right = left / 1.15
    solution = solve_s13_photometric(
        [_samples(left, right)], frame_ids=[10, 11],
        config=S13PhotometricConfig(minimum_pair_sample_count=32),
    )
    assert solution.model_family in {"Q1_scalar_luminance_gain", "Q2_rgb_diagonal_gain"}
    gains = np.asarray([p.gain_bgr for p in solution.source_parameters])
    assert np.max(gains) <= 1.25
    assert np.min(gains) >= 0.80


def test_disconnected_source_falls_back_to_identity() -> None:
    values = np.full((128, 3), 0.3, np.float32)
    solution = solve_s13_photometric(
        [_samples(values, values / 1.1)], frame_ids=[10, 11, 12],
        config=S13PhotometricConfig(minimum_pair_sample_count=32),
    )
    isolated = solution.source_parameters[2]
    assert isolated.gain_bgr == (1.0, 1.0, 1.0)
    assert isolated.bias_bgr == (0.0, 0.0, 0.0)


def test_out_of_bounds_candidate_does_not_remove_q0_fallback() -> None:
    left = np.full((128, 3), 0.8, np.float32)
    right = np.full((128, 3), 0.2, np.float32)
    solution = solve_s13_photometric(
        [_samples(left, right)], frame_ids=[10, 11],
        config=S13PhotometricConfig(minimum_pair_sample_count=32),
    )
    assert solution.model_family == "Q0_identity"
    assert solution.candidate_audits[0]["selected"] is True
