from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_m63_quality_cut import classify_s13_m63_pairs
from test_video_s13_m63_solver import _sample


def test_quality_cut_uses_population_and_not_fixed_pair_indices():
    levels = np.linspace(0.0, 0.02, 6)
    samples = [
        _sample(index, index, index + 1, levels[index], levels[index + 1])
        for index in range(5)
    ]
    samples[1] = _sample(1, 1, 2, 0.0, 0.0, outlier=True)
    rows = classify_s13_m63_pairs(samples)
    assert rows[1].classification == "photometric_quality_cut"
    assert rows[0].classification == "trusted"
    assert rows[-1].classification == "trusted"


def test_forced_candidate_regression_becomes_a_bounded_hard_cut():
    samples = [_sample(7, 0, 1, 0.0, 0.005)]
    rows = classify_s13_m63_pairs(samples, forced_hard_pair_indices=frozenset({7}))
    assert rows[0].classification == "photometric_quality_cut"
    assert "candidate_regression" in rows[0].reasons
