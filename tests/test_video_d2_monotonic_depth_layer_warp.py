from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_d2_monotonic_depth_layer_warp import (
    D2LayerWarp, D2MonotonicDepthLayerWarpError, audit_d2_monotonic_depth_layer_warp,
)


def _layers(*, scale: float = 1.0, vertical: float = 0.5) -> tuple[D2LayerWarp, ...]:
    x = np.array([0.0, 32.0, 64.0, 96.0], dtype=np.float64)
    return tuple(D2LayerWarp(layer, x, x * scale, np.full_like(x, vertical)) for layer in ("far", "mid", "near"))


def test_monotonic_warp_has_positive_jacobian_and_a_real_output_change() -> None:
    audit = audit_d2_monotonic_depth_layer_warp(_layers(scale=1.1), actual_output_warp_pixel_count=321)
    assert audit["actual_output_warp_pixel_count"] == 321
    assert all(entry["minimum_jacobian"] >= 0.05 for entry in audit["layers"].values())


@pytest.mark.parametrize("scale", [0.69, 1.41])
def test_monotonic_warp_rejects_scale_outside_closed_bound(scale: float) -> None:
    with pytest.raises(D2MonotonicDepthLayerWarpError, match="scale"):
        audit_d2_monotonic_depth_layer_warp(_layers(scale=scale), actual_output_warp_pixel_count=1)


def test_monotonic_warp_rejects_fold_and_zero_actual_output() -> None:
    folded = list(_layers())
    folded[1] = D2LayerWarp("mid", np.array([0.0, 32.0, 64.0]), np.array([0.0, 36.0, 35.0]), np.zeros(3))
    with pytest.raises(D2MonotonicDepthLayerWarpError, match="increments"):
        audit_d2_monotonic_depth_layer_warp(folded, actual_output_warp_pixel_count=1)
    with pytest.raises(D2MonotonicDepthLayerWarpError, match="actual_output"):
        audit_d2_monotonic_depth_layer_warp(_layers(), actual_output_warp_pixel_count=0)


def test_monotonic_warp_rejects_unbounded_vertical_residual() -> None:
    with pytest.raises(D2MonotonicDepthLayerWarpError, match="vertical residual"):
        audit_d2_monotonic_depth_layer_warp(_layers(vertical=1.51), actual_output_warp_pixel_count=1)
