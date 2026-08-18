from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_m63_tone import select_s13_m63_global_tone


def test_global_tone_uses_one_uniform_gain_and_preserves_invalid_zero():
    image = np.full((4, 5, 3), 0.25, np.float32)
    valid = np.ones((4, 5), bool)
    valid[0, 0] = False
    result = select_s13_m63_global_tone(image, valid)
    assert result.gain == 1.12
    assert np.allclose(result.image[valid], 0.28)
    assert np.count_nonzero(result.image[~valid]) == 0
    ratios = result.image[valid] / image[valid]
    assert np.allclose(ratios, result.gain)


def test_global_tone_rejects_highlight_and_per_channel_clipping():
    image = np.full((10, 10, 3), 0.94, np.float32)
    image[..., 2] = 0.99
    valid = np.ones((10, 10), bool)
    result = select_s13_m63_global_tone(image, valid)
    assert result.gain == 1.0
    assert np.array_equal(result.image, image)


def test_global_tone_p995_gate_can_select_a_smaller_candidate():
    image = np.full((10, 10, 3), 0.88, np.float32)
    valid = np.ones((10, 10), bool)
    result = select_s13_m63_global_tone(image, valid)
    assert result.gain in (1.02, 1.04, 1.06, 1.08, 1.10)
    assert float(np.quantile(result.image[valid].mean(axis=1), 0.995)) <= 0.98
