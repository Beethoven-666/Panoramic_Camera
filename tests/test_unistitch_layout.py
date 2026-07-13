from __future__ import annotations

import pytest

from panorama_demo.unistitch_adapter import AlignmentError, _choose_layout_method


def _choose(**overrides):
    values = {
        "unistitch_median": 8.0,
        "magsac_median": 2.0,
        "unistitch_median_on_magsac_inliers": 4.0,
        "magsac_inliers": 60,
        "magsac_inlier_ratio": 0.6,
        "min_matches": 40,
        "min_magsac_inlier_ratio": 0.5,
        "max_unistitch_reprojection_px": 20.0,
        "allow_magsac_fallback": True,
        "prefer_magsac_layout": True,
    }
    values.update(overrides)
    return _choose_layout_method(**values)


def test_layout_prefers_magsac_only_on_sufficient_common_support() -> None:
    assert _choose() == ("magsac_preferred", 2.0)
    assert _choose(magsac_inlier_ratio=0.4) == ("unistitch_global", 8.0)


def test_layout_compares_residuals_on_the_same_support() -> None:
    assert _choose(
        unistitch_median=12.0,
        unistitch_median_on_magsac_inliers=1.0,
    ) == ("unistitch_global", 12.0)


def test_layout_uses_magsac_as_fallback_for_invalid_unistitch() -> None:
    assert _choose(unistitch_median=25.0) == ("magsac_preferred", 2.0)
    assert _choose(
        unistitch_median=25.0,
        prefer_magsac_layout=False,
    ) == ("magsac_fallback", 2.0)


def test_diagnostic_threshold_accepts_best_magsac_below_official_ratio() -> None:
    assert _choose(
        unistitch_median=50.72,
        magsac_inlier_ratio=0.487,
        min_magsac_inlier_ratio=0.0,
        max_unistitch_reprojection_px=1_000_000.0,
    ) == ("magsac_preferred", 2.0)


def test_strict_layout_rejects_invalid_unistitch_without_running_fallback() -> None:
    with pytest.raises(AlignmentError, match="failed validation"):
        _choose(
            unistitch_median=25.0,
            allow_magsac_fallback=False,
        )
