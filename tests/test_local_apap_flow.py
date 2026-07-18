from __future__ import annotations

import cv2
import numpy as np
import pytest

from panorama_demo.local_apap_flow import (
    LocalAPAPFlowConfig,
    LocalAPAPFlowInverseWarp,
    fit_local_apap_plus_dense_flow,
)


def _translated_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    height, width = 96, 128
    random = np.random.default_rng(91)
    target = cv2.GaussianBlur(
        random.integers(0, 256, size=(height, width, 3), dtype=np.uint8),
        (0, 0),
        1.2,
    )
    source = cv2.warpAffine(
        target,
        np.float32([[1.0, 0.0, 3.0], [0.0, 1.0, 1.0]]),
        (width, height),
        borderMode=cv2.BORDER_REFLECT,
    )
    same_layer = np.zeros((height, width), dtype=bool)
    same_layer[8:-8, 8:-8] = True
    target_points = np.asarray(
        [(x, y) for y in range(12, height - 12, 12) for x in range(12, width - 12, 12)],
        dtype=np.float64,
    )
    source_points = target_points + np.asarray((3.0, 1.0), dtype=np.float64)
    return source, target, same_layer, (source_points, target_points)


def test_local_apap_candidate_is_bounded_to_the_same_layer_interior() -> None:
    source, target, same_layer, correspondences = _translated_pair()
    protected = np.zeros_like(same_layer)
    protected[30:40, 50:60] = True
    application = same_layer & ~protected

    result = fit_local_apap_plus_dense_flow(
        source,
        target,
        same_layer_mask=same_layer,
        application_mask=application,
        protected_mask=protected,
        correspondences=correspondences,
        config=LocalAPAPFlowConfig(enabled=True),
    )

    assert result.accepted
    assert result.method in {"apap", "apap_plus_dense_flow"}
    assert not np.any(result.active_mask & protected)
    assert np.all(result.active_mask <= application)
    assert result.inverse_map_x is not None
    assert result.inverse_map_y is not None
    audit = result.as_dict()
    assert audit["active_pixel_count"] == int(np.count_nonzero(result.active_mask))
    assert audit["held_out_improvement_ratio"] >= 0.30
    assert "inverse_map_x" not in audit
    assert "active_mask" not in audit


def test_disabled_or_insufficient_candidate_requests_a_hard_cut() -> None:
    source, target, same_layer, correspondences = _translated_pair()

    disabled = fit_local_apap_plus_dense_flow(
        source,
        target,
        same_layer_mask=same_layer,
        correspondences=correspondences,
    )
    assert not disabled.accepted
    assert disabled.method == "hard_cut_degraded"
    assert disabled.audit["reason"] == "local_apap_flow_disabled"

    insufficient = fit_local_apap_plus_dense_flow(
        source,
        target,
        same_layer_mask=same_layer,
        correspondences=(correspondences[0][:8], correspondences[1][:8]),
        config=LocalAPAPFlowConfig(enabled=True),
    )
    assert not insufficient.accepted
    assert insufficient.method == "hard_cut_degraded"
    assert insufficient.audit["reason"] == "insufficient_correspondences"


def test_candidate_rejects_mask_escape_and_relaxed_configuration() -> None:
    source, target, same_layer, correspondences = _translated_pair()
    application = same_layer.copy()
    application[0, 0] = True

    with pytest.raises(ValueError, match="same-layer subset"):
        fit_local_apap_plus_dense_flow(
            source,
            target,
            same_layer_mask=same_layer,
            application_mask=application,
            correspondences=correspondences,
            config=LocalAPAPFlowConfig(enabled=True),
        )
    with pytest.raises(ValueError, match="maximum_displacement"):
        LocalAPAPFlowConfig.from_mapping({"maximum_displacement_pixels": 8.1})
    with pytest.raises(ValueError, match="minimum_local_scale"):
        LocalAPAPFlowConfig.from_mapping({"minimum_local_scale": 0.79})
    with pytest.raises(ValueError, match="mesh_cell_pixels must be an integer"):
        LocalAPAPFlowConfig.from_mapping({"mesh_cell_pixels": 16.5})


def test_candidate_refuses_correspondences_whose_source_side_is_protected() -> None:
    source, target, same_layer, correspondences = _translated_pair()
    protected = np.zeros_like(same_layer)
    source_points, target_points = correspondences
    protected[
        np.rint(source_points[:, 1]).astype(np.int32),
        np.rint(source_points[:, 0]).astype(np.int32),
    ] = True

    result = fit_local_apap_plus_dense_flow(
        source,
        target,
        same_layer_mask=same_layer,
        application_mask=same_layer & ~protected,
        protected_mask=protected,
        correspondences=(source_points, target_points),
        config=LocalAPAPFlowConfig(enabled=True),
    )

    assert not result.accepted
    assert result.audit["reason"] == "insufficient_correspondences"


def test_inverse_warp_only_maps_verified_safe_local_samples() -> None:
    """The renderer adapter must never interpolate through an unsafe cell."""

    height, width = 8, 10
    yy, xx = np.indices((height, width), dtype=np.float32)
    inverse_x = xx.copy()
    inverse_y = yy.copy()
    active = np.zeros((height, width), dtype=bool)
    active[1:7, 1:9] = True

    # A valid output sample at local (2, 2) samples safe local source (3, 2).
    inverse_x[2, 2] = 3.0
    # Local (3, 3) is itself eligible but its source is a protected/inactive
    # cell.  It must retain identity rather than sampling the protected pixel.
    inverse_x[3, 3] = 4.0
    active[3, 4] = False
    # A protected output cell is likewise inactive even if its map is nonidentity.
    inverse_x[3, 4] = 5.0
    # An otherwise active output whose map leaves the corridor must also be
    # identity: an out-of-bounds value is not a usable RGB source sample.
    inverse_x[3, 5] = float(width + 1)

    warp = LocalAPAPFlowInverseWarp(
        corridor_x0=100,
        inverse_map_x=inverse_x,
        inverse_map_y=inverse_y,
        active_mask=active,
    )
    query_x = np.asarray(
        [99.0, 102.0, 103.0, 104.0, 105.0, 105.0], dtype=np.float64
    )
    query_y = np.asarray([2.0, 2.0, 3.0, 3.0, 7.0, 3.0], dtype=np.float64)

    mapped_x, mapped_y = warp.inverse_virtual_coordinates(query_x, query_y)

    # Outside the corridor, the safe active sample, unsafe-source sample,
    # protected output, outside the vertical corridor, and out-of-bounds
    # source sampling, respectively.
    np.testing.assert_array_equal(
        mapped_x,
        np.asarray([99.0, 103.0, 103.0, 104.0, 105.0, 105.0], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        mapped_y,
        np.asarray([2.0, 2.0, 3.0, 3.0, 7.0, 3.0], dtype=np.float64),
    )
