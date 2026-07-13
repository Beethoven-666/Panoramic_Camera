from __future__ import annotations

import cv2
import numpy as np
import pytest

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
    np.testing.assert_allclose(panorama, image, atol=2)


def test_scan_renderer_crops_shifted_solid_frames_without_voids() -> None:
    left = _solid(32, 48, (30, 60, 90))
    right = _solid(32, 48, (90, 60, 30))

    panorama, info = render_scan_panorama(
        [left, right],
        [IDENTITY, _translation(20.0, 4.0)],
        max_megapixels=1.0,
        seam_margin=8,
        multiband_levels=2,
    )

    assert panorama.shape == (28, 68, 3)
    assert info.crop.as_dict() == {"x": 0, "y": 4, "width": 68, "height": 28}
    assert np.all(panorama > 0)


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
