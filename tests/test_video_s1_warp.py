from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_s1_warp import (
    audit_vertical_offsets,
    compose_final_inverse_map,
    compose_pair_vertical_warp,
    cosine_fade_weights,
    cosine_shoulder_weights,
    sample_vertical_correction,
    symmetric_pair_vertical_offsets,
    validate_inverse_map,
)


def test_cosine_shoulder_is_maximum_at_seam_and_zero_at_centre() -> None:
    x = np.arange(0.0, 21.0)
    left = cosine_shoulder_weights(x, boundary_x=10.0, shoulder_width_px=4.0, side="left")
    right = cosine_shoulder_weights(x, boundary_x=10.0, shoulder_width_px=6.0, side="right")
    assert left[10] == pytest.approx(1.0)
    assert right[10] == pytest.approx(1.0)
    assert left[6] == pytest.approx(0.0)
    assert right[16] == pytest.approx(0.0)
    assert np.all(left[:6] == 0.0) and np.all(left[11:] == 0.0)
    assert np.all(right[:10] == 0.0) and np.all(right[17:] == 0.0)
    assert np.array_equal(
        left,
        cosine_fade_weights(x, boundary_x=10.0, shoulder_width_px=4.0, side="left"),
    )


def test_vertical_curve_is_local_and_pair_warp_is_symmetric() -> None:
    x = np.arange(0.0, 21.0)
    y = np.arange(0.0, 11.0)
    curve_y = np.array([2.0, 5.0, 8.0])
    dy = np.array([2.0, 2.0, 2.0])
    sampled = sample_vertical_correction(y, curve_y, dy, gain=0.5)
    assert np.all(sampled[:2] == 0.0) and np.all(sampled[9:] == 0.0)
    assert sampled[5] == pytest.approx(1.0)

    left, right = symmetric_pair_vertical_offsets(
        x,
        y,
        boundary_x=10.0,
        left_shoulder_px=4.0,
        right_shoulder_px=4.0,
        curve_y=curve_y,
        dy_px=dy,
        gain=0.5,
    )
    assert left[5, 10] == pytest.approx(-0.5)
    assert right[5, 10] == pytest.approx(0.5)
    assert np.all(left[:, :6] == 0.0)
    assert np.all(right[:, 15:] == 0.0)
    assert np.all(left[:, 11:] == 0.0)
    assert np.all(right[:, :10] == 0.0)


def test_safety_audit_rejects_nonfinite_and_excessive_local_slope() -> None:
    safe = np.tile(np.linspace(0.0, 0.14, 8)[:, None], (1, 3))
    report = audit_vertical_offsets(safe, maximum_local_slope=0.08)
    assert report.safe
    assert report.minimum_mapping_step > 0.0

    steep = safe.copy()
    steep[4:, :] += 0.2
    report = audit_vertical_offsets(steep, maximum_local_slope=0.08)
    assert report.monotonic
    assert not report.within_slope_limit
    assert not report.safe

    nonfinite = safe.copy()
    nonfinite[2, 1] = np.nan
    assert not audit_vertical_offsets(nonfinite).safe


def test_inverse_map_composition_is_pure_finite_and_bounds_aware() -> None:
    height, width = 8, 5
    base_x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width)).copy()
    base_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width)).copy()
    offset = np.zeros((height, width), dtype=np.float64)
    offset[:, -1] = np.linspace(0.0, 0.07, height)
    original_y = base_y.copy()

    result_x, result_y, valid, audit = compose_final_inverse_map(
        base_x,
        base_y,
        offset,
        source_height=height,
        maximum_local_slope=0.08,
    )
    assert audit.safe
    assert np.array_equal(base_y, original_y)
    assert np.array_equal(result_x, base_x)
    assert np.isfinite(result_x).all() and np.isfinite(result_y).all()
    assert np.allclose(result_y[:, 0], base_y[:, 0])
    assert not valid[-1, -1]


def test_inverse_map_composition_rejects_unsafe_warp_before_remap() -> None:
    base = np.zeros((4, 3), dtype=np.float32)
    unsafe = np.zeros_like(base)
    unsafe[2:, :] = -2.0
    with pytest.raises(ValueError, match="safety audit"):
        compose_final_inverse_map(base, base, unsafe)


def test_compose_pair_api_accepts_vertical_curve_shape_without_mutation() -> None:
    base_y = np.broadcast_to(np.arange(9.0)[:, None], (9, 11)).copy()
    original = base_y.copy()
    result = compose_pair_vertical_warp(
        base_y,
        boundary_x=5.0,
        curve={"y": [1.0, 4.0, 7.0], "dy_applied": [0.1, 0.1, 0.1]},
        side="left",
        shoulder_width_px=3.0,
    )
    assert np.array_equal(base_y, original)
    assert result[4, 5] == pytest.approx(base_y[4, 5] + 0.05)
    assert np.allclose(result[:, :2], base_y[:, :2])
    assert np.allclose(result[:, 6:], base_y[:, 6:])


def test_validate_inverse_map_checks_only_owned_valid_samples() -> None:
    map_x = np.zeros((3, 4), dtype=np.float32)
    map_y = np.zeros((3, 4), dtype=np.float32)
    map_y[0, 0] = np.nan
    valid = np.ones((3, 4), dtype=bool)
    valid[0, 0] = False
    detached = validate_inverse_map(
        map_x, map_y, valid_mask=valid, source_width=4, source_height=3
    )
    assert np.array_equal(detached, valid)
    map_x[1, 1] = 4.0
    with pytest.raises(ValueError, match="outside"):
        validate_inverse_map(
            map_x, map_y, valid_mask=valid, source_width=4, source_height=3
        )
