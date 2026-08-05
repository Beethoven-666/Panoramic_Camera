from __future__ import annotations

import numpy as np

from panorama_demo.video_photometric import (
    AdjacentBGRAOverlap,
    VideoPhotometricConfig,
    VideoPhotometricCorrection,
    apply_video_photometric_correction,
    solve_video_global_photometric,
)


def _linear_to_bgra(linear: np.ndarray, *, alpha: int = 255) -> np.ndarray:
    values = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    bgr = np.rint(np.clip(encoded * 255.0, 0.0, 255.0)).astype(np.uint8)
    return np.dstack((bgr, np.full(bgr.shape[:2], alpha, dtype=np.uint8)))


def _overlap(
    left_index: int,
    right_index: int,
    left: np.ndarray,
    right: np.ndarray,
) -> AdjacentBGRAOverlap:
    valid = np.ones(left.shape[:2], dtype=bool)
    return AdjacentBGRAOverlap(left_index, right_index, left, right, valid, valid)


def test_global_gain_bias_solver_composes_real_source_corrections() -> None:
    rng = np.random.default_rng(44)
    source0_linear = rng.uniform(0.08, 0.72, size=(72, 88, 3)).astype(np.float32)
    relation01_gain = np.array((1.12, 0.91, 1.06), dtype=np.float32)
    relation01_bias = np.array((0.018, -0.012, 0.009), dtype=np.float32)
    source1_linear = (source0_linear - relation01_bias) / relation01_gain
    relation12_gain = np.array((0.95, 1.08, 0.93), dtype=np.float32)
    relation12_bias = np.array((-0.006, 0.011, -0.008), dtype=np.float32)
    source2_linear = (source1_linear - relation12_bias) / relation12_gain

    result = solve_video_global_photometric(
        3,
        (
            _overlap(0, 1, _linear_to_bgra(source0_linear), _linear_to_bgra(source1_linear)),
            _overlap(1, 2, _linear_to_bgra(source1_linear), _linear_to_bgra(source2_linear)),
        ),
    )

    assert result.accepted
    np.testing.assert_allclose(result.corrections[0].gain_bgr, 1.0, atol=1e-10)
    np.testing.assert_allclose(result.corrections[0].bias_bgr, 0.0, atol=1e-10)
    np.testing.assert_allclose(
        result.corrections[1].gain_bgr, relation01_gain, atol=0.025
    )
    np.testing.assert_allclose(
        result.corrections[1].bias_bgr, relation01_bias, atol=0.012
    )
    np.testing.assert_allclose(
        result.corrections[2].gain_bgr,
        relation01_gain * relation12_gain,
        atol=0.035,
    )
    np.testing.assert_allclose(
        result.corrections[2].bias_bgr,
        relation01_gain * relation12_bias + relation01_bias,
        atol=0.018,
    )
    assert result.audit["model"] == "gain_bias"
    assert result.audit["fail_closed_identity"] is False
    assert result.audit["training_pixel_count"] > 0
    assert result.audit["held_out_pixel_count"] > 0
    assert result.audit["held_out_pair_residual_p95_max_linear"] <= 0.035
    assert len(result.audit["pairs"]) == 2
    assert result.audit["pairs"][0]["training_pixels"] > 0
    assert result.audit["pairs"][0]["held_out_pixels"] > 0


def test_gain_only_solver_does_not_introduce_bias() -> None:
    rng = np.random.default_rng(8)
    source0_linear = rng.uniform(0.10, 0.70, size=(64, 80, 3)).astype(np.float32)
    relation_gain = np.array((1.14, 0.88, 1.05), dtype=np.float32)
    source1_linear = source0_linear / relation_gain

    result = solve_video_global_photometric(
        2,
        (_overlap(0, 1, _linear_to_bgra(source0_linear), _linear_to_bgra(source1_linear)),),
        config=VideoPhotometricConfig(model="gain_only"),
    )

    assert result.accepted
    np.testing.assert_allclose(result.corrections[1].gain_bgr, relation_gain, atol=0.025)
    np.testing.assert_allclose(result.corrections[1].bias_bgr, 0.0, atol=1e-12)


def test_insufficient_support_fails_closed_to_identity_for_every_source() -> None:
    image = np.full((10, 12, 4), 128, dtype=np.uint8)
    image[:, :, 3] = 255
    result = solve_video_global_photometric(
        2,
        (_overlap(0, 1, image, image.copy()),),
        config=VideoPhotometricConfig(minimum_support_pixels=128),
    )

    assert not result.accepted
    assert result.audit["fail_closed_identity"] is True
    assert "rejected_pair_0_1" in str(result.audit["rejection_reason"])
    for correction in result.corrections:
        np.testing.assert_array_equal(correction.gain_bgr, np.ones(3))
        np.testing.assert_array_equal(correction.bias_bgr, np.zeros(3))


def test_out_of_range_pair_gain_fails_closed_without_partial_transform() -> None:
    rng = np.random.default_rng(99)
    source0_linear = rng.uniform(0.08, 0.40, size=(64, 80, 3)).astype(np.float32)
    source1_linear = source0_linear / 1.75
    result = solve_video_global_photometric(
        2,
        (_overlap(0, 1, _linear_to_bgra(source0_linear), _linear_to_bgra(source1_linear)),),
    )

    assert not result.accepted
    assert "pair_gain_out_of_bounds" in str(result.audit["rejection_reason"])
    np.testing.assert_array_equal(result.corrections[1].gain_bgr, np.ones(3))


def test_held_out_spatial_tiles_reject_relation_that_only_fits_training() -> None:
    """A correction cannot pass by fitting the exact pixels it was trained on."""

    rng = np.random.default_rng(108)
    left_linear = rng.uniform(0.12, 0.52, size=(80, 96, 3)).astype(np.float32)
    gain = np.array((1.08, 0.93, 1.04), dtype=np.float32)
    right_linear = left_linear / gain
    config = VideoPhotometricConfig(
        held_out_tile_side_pixels=8,
        held_out_tile_modulus=5,
        held_out_tile_remainder=0,
        minimum_training_pixels=256,
        minimum_held_out_pixels=128,
    )
    rows, columns = np.indices(left_linear.shape[:2])
    held_out = ((rows // 8 + columns // 8) % 5) == 0
    # Corrupt whole held-out tiles in linear light.  The fit sees only the
    # remaining tiles, while the independent validation must fail.
    left_linear[held_out] = np.clip(left_linear[held_out] + 0.16, 0.0, 1.0)

    result = solve_video_global_photometric(
        2,
        (_overlap(0, 1, _linear_to_bgra(left_linear), _linear_to_bgra(right_linear)),),
        config=config,
    )

    assert not result.accepted
    assert "held_out_residual_out_of_bounds" in str(result.audit["rejection_reason"])
    pair = result.audit["pairs"][0]
    assert pair["training_pixels"] > pair["held_out_pixels"] > 0
    assert pair["held_out_residual_p95_linear"] > config.maximum_held_out_residual_p95
    np.testing.assert_array_equal(result.corrections[1].bias_bgr, np.zeros(3))


def test_apply_correction_works_in_linear_light_and_preserves_alpha() -> None:
    linear = np.full((4, 5, 3), (0.20, 0.31, 0.42), dtype=np.float32)
    image = _linear_to_bgra(linear)
    image[0, 0, 3] = 0
    correction = VideoPhotometricCorrection(
        gain_bgr=np.array((1.10, 0.90, 1.05), dtype=np.float64),
        bias_bgr=np.array((0.01, -0.02, 0.0), dtype=np.float64),
    )

    output = apply_video_photometric_correction(image, correction)

    np.testing.assert_array_equal(output[:, :, 3], image[:, :, 3])
    expected = _linear_to_bgra(
        linear * correction.gain_bgr.reshape(1, 1, 3)
        + correction.bias_bgr.reshape(1, 1, 3)
    )
    np.testing.assert_allclose(output[:, :, :3], expected[:, :, :3], atol=1)
