from __future__ import annotations

import cv2
import numpy as np
import pytest

import panorama_demo.render as render_module
from panorama_demo.render import (
    compute_canvas,
    largest_valid_rectangle,
    render_panorama,
    render_scan_panorama,
)


IDENTITY = np.eye(3, dtype=np.float64)


def _solid(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def _translation(x: float, y: float = 0.0) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def test_compute_canvas_for_identity_transform() -> None:
    image = _solid(4, 6, (0, 0, 0))

    info = compute_canvas([image], [IDENTITY], max_megapixels=1.0)

    assert (info.width, info.height) == (6, 4)
    assert (info.min_x, info.min_y, info.max_x, info.max_y) == (0.0, 0.0, 6.0, 4.0)
    np.testing.assert_array_equal(info.translation, IDENTITY)
    assert info.as_dict() == {
        "width": 6,
        "height": 4,
        "bounds": [0.0, 0.0, 6.0, 4.0],
        "translation": IDENTITY.tolist(),
    }


def test_compute_canvas_handles_negative_world_coordinates() -> None:
    image = _solid(4, 6, (0, 0, 0))

    info = compute_canvas(
        [image, image], [IDENTITY, _translation(-3.0, -2.0)], max_megapixels=1.0
    )

    assert (info.width, info.height) == (9, 6)
    assert (info.min_x, info.min_y, info.max_x, info.max_y) == (-3.0, -2.0, 6.0, 4.0)
    np.testing.assert_array_equal(info.translation, _translation(3.0, 2.0))


@pytest.mark.parametrize(
    "images,transforms",
    [([], []), ([_solid(2, 2, (0, 0, 0))], []), ([], [IDENTITY])],
)
def test_compute_canvas_requires_matching_nonempty_inputs(
    images: list[np.ndarray], transforms: list[np.ndarray]
) -> None:
    with pytest.raises(ValueError, match="non-empty and have equal length"):
        compute_canvas(images, transforms, max_megapixels=1.0)


@pytest.mark.parametrize(
    "bad_transform",
    [
        np.zeros((2, 3), dtype=np.float64),
        np.array([[1.0, 0.0, np.nan], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_compute_canvas_rejects_invalid_homographies(bad_transform: np.ndarray) -> None:
    image = _solid(2, 2, (0, 0, 0))

    with pytest.raises(ValueError):
        compute_canvas([image], [bad_transform], max_megapixels=1.0)


def test_compute_canvas_enforces_megapixel_limit() -> None:
    image = _solid(100, 100, (0, 0, 0))

    with pytest.raises(MemoryError, match="above the 0.0 MP demo limit"):
        compute_canvas([image], [IDENTITY], max_megapixels=0.009)


def test_render_single_image_round_trips_without_clipping() -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)

    panorama, info = render_panorama(
        [image], [IDENTITY], max_megapixels=1.0, feather_pixels=8
    )

    assert panorama.shape == image.shape
    assert (info.width, info.height) == (32, 24)
    np.testing.assert_allclose(panorama, image, atol=1)


def test_render_blends_black_pixels_as_valid_image_content() -> None:
    black = _solid(20, 20, (0, 0, 0))
    colored = _solid(20, 20, (200, 100, 50))

    panorama, _ = render_panorama(
        [black, colored], [IDENTITY, IDENTITY], max_megapixels=1.0
    )

    np.testing.assert_allclose(panorama[10, 10], np.array([100, 50, 25]), atol=1)


def test_render_expands_canvas_and_preserves_nonoverlap_regions() -> None:
    left = _solid(8, 10, (255, 0, 0))
    right = _solid(8, 10, (0, 255, 0))

    panorama, info = render_panorama(
        [left, right], [IDENTITY, _translation(6.0)], max_megapixels=1.0
    )

    assert panorama.shape == (8, 16, 3)
    assert info.width == 16
    np.testing.assert_allclose(panorama[4, 2], np.array([255, 0, 0]), atol=1)
    np.testing.assert_allclose(panorama[4, 13], np.array([0, 255, 0]), atol=1)
    assert panorama[4, 7, 0] > 0
    assert panorama[4, 7, 1] > 0


def test_render_rejects_non_bgr_image() -> None:
    grayscale = np.zeros((5, 7), dtype=np.uint8)

    with pytest.raises(ValueError, match="expects BGR uint8 images"):
        render_panorama([grayscale], [IDENTITY], max_megapixels=1.0)


def test_largest_valid_rectangle_uses_explicit_mask() -> None:
    mask = np.zeros((5, 8), dtype=np.uint8)
    mask[:, 2:7] = 255
    mask[1:4, 1] = 255

    crop = largest_valid_rectangle(mask)

    assert (crop.x, crop.y, crop.width, crop.height) == (2, 0, 5, 5)


def test_scan_renderer_round_trips_one_frame_and_reports_crop() -> None:
    rng = np.random.default_rng(7)
    image = rng.integers(10, 246, size=(32, 48, 3), dtype=np.uint8)

    panorama, info = render_scan_panorama(
        [image], [IDENTITY], max_megapixels=1.0, multiband_levels=2
    )

    assert panorama.shape == image.shape
    assert info.crop.as_dict() == {"x": 0, "y": 0, "width": 48, "height": 32}
    assert info.quality_metrics["crop_height_ratio"] == 1.0
    assert info.quality_metrics["crop_width_ratio"] == 1.0
    np.testing.assert_allclose(panorama, image, atol=2)


def test_scan_renderer_measures_single_frame_crop_before_quality_pass() -> None:
    image = _solid(100, 140, (40, 80, 120))
    affine = cv2.getRotationMatrix2D((70.0, 50.0), 20.0, 1.0)
    transform = np.vstack((affine, [0.0, 0.0, 1.0]))

    _, info = render_scan_panorama(
        [image], [transform], max_megapixels=1.0, quality_gate=False
    )

    assert info.quality_metrics["quality_pass"] is False
    assert info.quality_metrics["crop_height_ratio"] < 0.90
    with pytest.raises(RuntimeError, match="source height remains"):
        render_scan_panorama(
            [image], [transform], max_megapixels=1.0, quality_gate=True
        )


def test_scan_renderer_crops_shifted_solid_frames_without_voids() -> None:
    left = _solid(32, 48, (30, 60, 90))
    right = _solid(32, 48, (90, 60, 30))

    panorama, info = render_scan_panorama(
        [left, right],
        [IDENTITY, _translation(20.0, 4.0)],
        max_megapixels=1.0,
        seam_margin=8,
        multiband_levels=2,
        auto_foreground=False,
        quality_gate=False,
    )

    assert panorama.shape == (28, 68, 3)
    assert info.crop.as_dict() == {"x": 0, "y": 4, "width": 68, "height": 28}
    assert np.all(panorama > 0)


def test_scan_renderer_does_not_interpolate_black_at_fractional_edges() -> None:
    white = _solid(40, 60, (255, 255, 255))

    panorama, info = render_scan_panorama(
        [white, white],
        [IDENTITY, _translation(20.4, 0.35)],
        max_megapixels=1.0,
        seam_margin=12,
        multiband_levels=2,
        exposure_mode="none",
        quality_gate=True,
    )

    assert info.quality_metrics["quality_pass"] is True
    assert int(panorama.min()) >= 250


def test_scan_renderer_enforces_aggregate_working_set_limit() -> None:
    image = _solid(100, 100, (30, 60, 90))

    with pytest.raises(MemoryError, match="aggregate MP"):
        render_scan_panorama(
            [image, image],
            [IDENTITY, IDENTITY],
            max_megapixels=0.015,
            multiband_levels=2,
        )


def test_scan_renderer_does_not_fall_back_to_overlap_averaging(monkeypatch) -> None:
    class BrokenSeamFinder:
        def find(self, *_args):
            raise cv2.error("synthetic graph-cut failure")

    monkeypatch.setattr(
        cv2,
        "detail_GraphCutSeamFinder",
        lambda _cost: BrokenSeamFinder(),
    )
    left = _solid(32, 48, (30, 60, 90))
    right = _solid(32, 48, (90, 60, 30))

    with pytest.raises(RuntimeError, match="refusing to fall back"):
        render_scan_panorama(
            [left, right],
            [IDENTITY, _translation(20.0, 4.0)],
            max_megapixels=1.0,
            seam_margin=8,
            multiband_levels=2,
            auto_foreground=False,
        )


def test_scan_renderer_reports_global_gain_and_protected_regions() -> None:
    dark = _solid(32, 48, (80, 80, 80))
    bright = _solid(32, 48, (180, 180, 180))

    panorama, info = render_scan_panorama(
        [dark, bright],
        [IDENTITY, _translation(20.0, 4.0)],
        max_megapixels=1.0,
        seam_margin=8,
        multiband_levels=2,
        exposure_mode="global_gain",
        seam_mask_sigma=1.0,
        protected_regions=[(1, 22, 6, 80, 40)],
        auto_foreground=False,
        quality_gate=False,
    )

    assert panorama.shape == (28, 68, 3)
    assert info.exposure_mode == "global_gain"
    assert info.seam_mask_sigma == 1.0
    assert info.protected_regions == ((1, 22, 6, 68, 36),)
    assert not np.allclose(info.color_gains[0], info.color_gains[1])


def test_scan_renderer_rejects_protected_region_outside_canvas() -> None:
    image = _solid(16, 24, (30, 60, 90))

    with pytest.raises(ValueError, match="does not intersect the scan canvas"):
        render_scan_panorama(
            [image, image],
            [IDENTITY, _translation(4.0)],
            max_megapixels=1.0,
            protected_regions=[(0, 100, 0, 110, 10)],
        )


def test_scan_renderer_auto_protects_depth_disagreement_crossed_by_seam() -> None:
    rng = np.random.default_rng(19)
    background = rng.integers(150, 210, size=(80, 120, 3), dtype=np.uint8)
    first = background.copy()
    second = background.copy()
    first[28:68, 42:72] = (20, 20, 220)
    second[28:68, 54:84] = (20, 20, 220)
    depth0 = np.full((80, 120), 2000.0, dtype=np.float32)
    depth1 = depth0.copy()
    depth0[28:68, 42:72] = 500.0
    depth1[28:68, 54:84] = 500.0

    panorama, info = render_scan_panorama(
        [first, second],
        [IDENTITY, IDENTITY],
        max_megapixels=1.0,
        seam_margin=24,
        multiband_levels=2,
        exposure_mode="none",
        seam_mask_sigma=1.0,
        depth_maps_mm=[depth0, depth1],
        auto_foreground=True,
        quality_gate=False,
    )

    hsv = cv2.cvtColor(panorama, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 10) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 120)
    _, _, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8))
    largest = stats[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]

    assert info.automatic_protected_regions == ()
    assert info.quality_metrics["automatic_dp_seam_count"] == 1
    assert info.quality_metrics["strict_owner_partition"] is True
    assert info.quality_metrics["final_owner_boundary_pair_count"] == 1
    assert info.quality_metrics["unowned_valid_pixel_count"] == 0
    assert info.quality_metrics["multiple_owner_pixel_count"] == 0
    assert largest[cv2.CC_STAT_WIDTH] <= 34


def test_monotonic_seam_uses_depth_when_rgb_has_no_disagreement() -> None:
    image = _solid(72, 112, (90, 120, 150))
    depth0 = np.full((72, 112), 2000.0, dtype=np.float32)
    depth1 = depth0.copy()
    depth0[20:60, 38:62] = 500.0
    depth1[20:60, 50:74] = 500.0

    _, info = render_scan_panorama(
        [image, image],
        [IDENTITY, IDENTITY],
        max_megapixels=1.0,
        seam_margin=24,
        multiband_levels=2,
        exposure_mode="none",
        depth_maps_mm=[depth0, depth1],
        quality_gate=False,
    )

    assert info.quality_metrics["automatic_dp_seam_count"] == 1
    assert info.quality_metrics["final_owner_boundary_pair_count"] == 1
    assert info.quality_metrics["seam_risk_fraction"] == 0.0


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        (
            "nonadjacent_owner_boundary_pixel_count",
            4,
            "non-adjacent frame boundary",
        ),
        ("final_owner_boundary_pair_count", 0, "boundaries disappeared"),
    ],
)
def test_quality_gate_rejects_invalid_final_owner_adjacency(
    monkeypatch, metric: str, value: int, message: str
) -> None:
    original = render_module._evaluate_final_owner_boundaries

    def unsafe_metrics(*args, **kwargs):
        metrics = original(*args, **kwargs)
        metrics[metric] = value
        return metrics

    monkeypatch.setattr(
        render_module, "_evaluate_final_owner_boundaries", unsafe_metrics
    )
    image = _solid(72, 112, (90, 120, 150))

    with pytest.raises(RuntimeError, match=message):
        render_scan_panorama(
            [image, image],
            [IDENTITY, _translation(36)],
            max_megapixels=1.0,
            seam_margin=24,
            multiband_levels=2,
            exposure_mode="none",
            quality_gate=True,
        )


def test_owner_topology_detects_nonadjacent_contact_without_source_overlap() -> None:
    shape = (10, 12)
    images = [_solid(*shape, (80, 100, 120)) for _ in range(3)]
    owners = [np.zeros(shape, dtype=np.uint8) for _ in range(3)]
    owners[0][:, :4] = 255
    owners[2][:, 4:8] = 255
    owners[1][:, 8:] = 255
    valid = [owner.copy() for owner in owners]

    metrics = render_module._evaluate_final_owner_boundaries(
        images,
        valid,
        [None, None, None],
        owners,
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([0, 1, 2], dtype=np.int64),
        blend_guard_pixels=8,
    )

    assert metrics["strict_owner_partition"] is True
    assert metrics["nonadjacent_owner_boundary_pixel_count"] > 0
