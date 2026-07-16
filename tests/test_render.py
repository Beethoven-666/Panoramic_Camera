from __future__ import annotations

import cv2
import numpy as np
import pytest

import panorama_demo.render as render_module
from panorama_demo.render import (
    PrewarpedScanSource,
    compute_canvas,
    largest_valid_rectangle,
    render_panorama,
    render_prewarped_scan_panorama,
    render_projected_scan_panorama,
    render_scan_panorama,
)
from panorama_demo.rgbd_projection import ProjectedRGBDSource


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


def _projected_source(
    frame_id: int,
    image: np.ndarray,
    valid_mask: np.ndarray,
    surface_depth_mm: np.ndarray,
    center_x: float,
    *,
    sampling_ratio: float = 1.0,
) -> ProjectedRGBDSource:
    x, y, width, height = cv2.boundingRect(valid_mask)
    return ProjectedRGBDSource(
        frame_id=frame_id,
        warped_rgb=image.copy(),
        valid_mask=valid_mask.copy(),
        surface_depth_mm=np.asarray(surface_depth_mm, dtype=np.float32).copy(),
        surface_depth_valid_mask=valid_mask.copy(),
        camera_depth_mm=np.asarray(surface_depth_mm, dtype=np.float32).copy(),
        camera_depth_valid_mask=valid_mask.copy(),
        projected_center_xy=(float(center_x), image.shape[0] * 0.5),
        valid_bbox=(x, y, width, height),
        projected_height_px=height,
        sampling_stats={"projected_sampling_ratio": sampling_ratio},
        camera_center_xy=(float(center_x), image.shape[0] * 0.5),
    )


def _prewarped_source(
    image: np.ndarray,
    color_valid_mask: np.ndarray,
    center_x: float,
    *,
    surface_depth_valid_mask: np.ndarray | None = None,
    camera_depth_valid_mask: np.ndarray | None = None,
) -> PrewarpedScanSource:
    shape = image.shape[:2]
    surface_valid = (
        np.zeros(shape, dtype=np.uint8)
        if surface_depth_valid_mask is None
        else np.asarray(surface_depth_valid_mask, dtype=np.uint8).copy()
    )
    camera_valid = (
        np.zeros(shape, dtype=np.uint8)
        if camera_depth_valid_mask is None
        else np.asarray(camera_depth_valid_mask, dtype=np.uint8).copy()
    )
    surface_depth = np.zeros(shape, dtype=np.float32)
    camera_depth = np.zeros(shape, dtype=np.float32)
    surface_depth[surface_valid > 0] = 2000.0
    camera_depth[camera_valid > 0] = 2000.0
    return PrewarpedScanSource(
        rgb=image.copy(),
        color_valid_mask=np.asarray(color_valid_mask, dtype=np.uint8).copy(),
        surface_depth_mm=surface_depth,
        surface_depth_valid_mask=surface_valid,
        camera_depth_mm=camera_depth,
        camera_depth_valid_mask=camera_valid,
        projected_center_xy=(float(center_x), image.shape[0] * 0.5),
    )


def _full_projected_sources(
    *,
    count: int = 2,
    shape: tuple[int, int] = (40, 80),
    color: tuple[int, int, int] = (90, 120, 150),
) -> list[ProjectedRGBDSource]:
    height, width = shape
    image = _solid(height, width, color)
    valid = np.full(shape, 255, dtype=np.uint8)
    depth = np.full(shape, 1000.0, dtype=np.float32)
    return [
        _projected_source(
            index,
            image,
            valid,
            depth,
            (index + 0.5) * width / count,
        )
        for index in range(count)
    ]


class _DeterministicGraphCut:
    def __init__(self, mode: str = "split", captured: list | None = None) -> None:
        self.mode = mode
        self.captured = captured

    def find(self, _images, _corners, masks):
        allowed0 = np.asarray(masks[0]) > 0
        allowed1 = np.asarray(masks[1]) > 0
        if self.captured is not None:
            self.captured.append((allowed0.copy(), allowed1.copy()))
        common = allowed0 & allowed1
        only0 = allowed0 & ~allowed1
        only1 = allowed1 & ~allowed0
        columns = np.arange(allowed0.shape[1])[None, :]
        split = allowed0.shape[1] // 2
        owner0 = only0 | (common & (columns < split))
        owner1 = only1 | (common & (columns >= split))
        if self.mode == "first":
            owner0 = allowed0.copy()
            owner1 = allowed1 & ~owner0
        elif self.mode == "second":
            owner1 = allowed1.copy()
            owner0 = allowed0 & ~owner1
        elif self.mode in {"hole", "overlap"}:
            row, column = np.argwhere(common)[len(np.argwhere(common)) // 2]
            if self.mode == "hole":
                owner0[row, column] = False
                owner1[row, column] = False
            else:
                owner0[row, column] = True
                owner1[row, column] = True
        return [
            np.where(owner0, 255, 0).astype(np.uint8),
            np.where(owner1, 255, 0).astype(np.uint8),
        ]


class _DeterministicMultiBand:
    def __init__(self, *, zero_weight_wedge: bool = False) -> None:
        self.zero_weight_wedge = zero_weight_wedge

    def setNumBands(self, _levels: int) -> None:
        pass

    def prepare(self, rectangle: tuple[int, int, int, int]) -> None:
        _, _, width, height = rectangle
        self.image = np.zeros((height, width, 3), dtype=np.int16)
        self.mask = np.zeros((height, width), dtype=np.uint8)

    def feed(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        corner: tuple[int, int],
    ) -> None:
        x, y = corner
        height, width = mask.shape
        target = self.image[y : y + height, x : x + width]
        selected = mask > 0
        target[selected] = image[selected]
        self.mask[y : y + height, x : x + width][selected] = 255

    def blend(self, _destination, _destination_mask):
        output_mask = self.mask.copy()
        if self.zero_weight_wedge:
            row, column = np.argwhere(output_mask > 0)[0]
            output_mask[row, column] = 0
        return self.image.copy(), output_mask


def _install_projected_cv_stubs(
    monkeypatch,
    *,
    graphcut_mode: str = "split",
    captured: list | None = None,
    zero_weight_wedge: bool = False,
) -> None:
    monkeypatch.setattr(
        cv2,
        "detail_GraphCutSeamFinder",
        lambda _cost: _DeterministicGraphCut(graphcut_mode, captured),
    )
    monkeypatch.setattr(
        cv2,
        "detail_MultiBandBlender",
        lambda: _DeterministicMultiBand(zero_weight_wedge=zero_weight_wedge),
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


def test_projected_graphcut_success_still_runs_strict_owner_audit(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)
    original = render_module._evaluate_final_owner_boundaries
    audited: list[dict[str, float | int | bool]] = []

    def record_audit(*args, **kwargs):
        metrics = original(*args, **kwargs)
        audited.append(metrics.copy())
        return metrics

    monkeypatch.setattr(
        render_module, "_evaluate_final_owner_boundaries", record_audit
    )

    panorama, info = render_projected_scan_panorama(
        _full_projected_sources(),
        exposure_mode="none",
        multiband_levels=1,
    )

    owner_count = np.sum(np.stack(info.owner_masks) > 0, axis=0)
    assert panorama.shape == (40, 80, 3)
    assert len(audited) == 1
    assert audited[0]["strict_owner_partition"] is True
    assert audited[0]["final_owner_boundary_pair_count"] == 1
    assert np.all(owner_count == 1)
    assert info.quality_metrics["quality_pass"] is True


def test_projected_graphcut_error_has_no_dp_or_average_fallback(monkeypatch) -> None:
    class BrokenGraphCut:
        def find(self, *_args):
            raise cv2.error("synthetic projected GraphCut failure")

    def forbidden_fallback(*_args, **_kwargs):
        raise AssertionError("a forbidden fallback was called")

    monkeypatch.setattr(
        cv2, "detail_GraphCutSeamFinder", lambda _cost: BrokenGraphCut()
    )
    monkeypatch.setattr(render_module, "_monotonic_pair_seam", forbidden_fallback)
    monkeypatch.setattr(
        render_module, "_normalized_multiband_blend", forbidden_fallback
    )
    monkeypatch.setattr(
        render_module,
        "_opencv_multiband_owner_boundary_blend",
        forbidden_fallback,
    )

    with pytest.raises(
        RuntimeError,
        match="no DP, feather, or averaging fallback is allowed",
    ):
        render_projected_scan_panorama(
            _full_projected_sources(),
            exposure_mode="none",
            multiband_levels=1,
        )


def test_projected_high_risk_component_is_hard_owned_before_graphcut(
    monkeypatch,
) -> None:
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    _install_projected_cv_stubs(monkeypatch, captured=captured)
    shape = (64, 96)
    image = _solid(*shape, (100, 100, 100))
    valid = np.full(shape, 255, dtype=np.uint8)
    depth0 = np.full(shape, 1000.0, dtype=np.float32)
    depth1 = depth0.copy()
    depth1[24:36, 16:28] = 1300.0
    sources = [
        _projected_source(
            0, image, valid, depth0, 24.0, sampling_ratio=0.5
        ),
        _projected_source(
            1, image, valid, depth1, 72.0, sampling_ratio=1.0
        ),
    ]

    _, info = render_projected_scan_panorama(
        sources,
        exposure_mode="none",
        multiband_levels=1,
        quality_gate=False,
    )

    assert len(captured) == 1
    assert not captured[0][0][30, 22]
    assert captured[0][1][30, 22]
    assert info.hard_owner_component_count == 1
    assert info.owner_masks[0][30, 22] == 0
    assert info.owner_masks[1][30, 22] == 255


def test_world_surface_depth_conflict_is_hard_risk_in_textured_scene() -> None:
    shape = (48, 72)
    yy, xx = np.indices(shape)
    checker = (((xx // 4) + (yy // 4)) % 2 * 255).astype(np.uint8)
    image = cv2.merge((checker, 255 - checker, checker))
    valid = np.full(shape, 255, dtype=np.uint8)
    first = np.full(shape, 1000.0, dtype=np.float32)
    second = first.copy()
    second[14:34, 24:40] += 400.0

    risk, _, _ = render_module._pair_risk_mask(
        image,
        image.copy(),
        valid,
        valid,
        first,
        second,
        depth_valid0=valid,
        depth_valid1=valid,
        world_surface_depth=True,
    )

    assert np.all(risk[18:30, 28:36] > 0)


def test_world_surface_depth_risk_is_world_origin_invariant() -> None:
    shape = (40, 60)
    image = _solid(*shape, (80, 120, 160))
    valid = np.full(shape, 255, dtype=np.uint8)
    first = np.full(shape, 1000.0, dtype=np.float32)
    second = first.copy()
    second[10:30, 20:40] += 200.0

    def risk_at_offset(offset: float) -> np.ndarray:
        risk, _, _ = render_module._pair_risk_mask(
            image,
            image,
            valid,
            valid,
            first + offset,
            second + offset,
            depth_valid0=valid,
            depth_valid1=valid,
            world_surface_depth=True,
        )
        return risk

    np.testing.assert_array_equal(risk_at_offset(0.0), risk_at_offset(10_000.0))


def test_projected_near_foreground_requires_a_real_range_disagreement() -> None:
    shape = (40, 60)
    image = _solid(*shape, (80, 120, 160))
    valid = np.full(shape, 255, dtype=np.uint8)
    surface = np.full(shape, 2500.0, dtype=np.float32)
    camera_depth = np.full(shape, 1500.0, dtype=np.float32)
    camera_depth[12:28, 22:38] = 500.0

    risk, _, _ = render_module._pair_risk_mask(
        image,
        image,
        valid,
        valid,
        surface,
        surface,
        depth_valid0=valid,
        depth_valid1=valid,
        camera_depth0=camera_depth,
        camera_depth1=camera_depth,
        camera_depth_valid0=valid,
        camera_depth_valid1=valid,
        world_surface_depth=True,
    )

    assert not np.any(risk)

    second_camera_depth = camera_depth.copy()
    second_camera_depth[12:28, 22:38] += 120.0
    risk, _, _ = render_module._pair_risk_mask(
        image,
        image,
        valid,
        valid,
        surface,
        surface,
        depth_valid0=valid,
        depth_valid1=valid,
        camera_depth0=camera_depth,
        camera_depth1=second_camera_depth,
        camera_depth_valid0=valid,
        camera_depth_valid1=valid,
        world_surface_depth=True,
    )

    assert np.all(risk[16:24, 26:34] > 0)


def test_projected_high_risk_band_crossing_corridor_gets_a_single_owner(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)
    shape = (64, 96)
    image = _solid(*shape, (100, 100, 100))
    valid = np.full(shape, 255, dtype=np.uint8)
    depth0 = np.full(shape, 1000.0, dtype=np.float32)
    depth1 = depth0.copy()
    depth1[28:36, :] = 1300.0
    sources = [
        _projected_source(0, image, valid, depth0, 24.0),
        _projected_source(1, image, valid, depth1, 72.0),
    ]

    _, info = render_projected_scan_panorama(
        sources,
        exposure_mode="none",
        multiband_levels=1,
        quality_gate=False,
    )

    assert info.hard_owner_component_count >= 1


@pytest.mark.parametrize("mode", ["hole", "overlap"])
def test_projected_graphcut_rejects_owner_holes_and_overlaps(
    monkeypatch, mode: str
) -> None:
    _install_projected_cv_stubs(monkeypatch, graphcut_mode=mode)

    with pytest.raises(RuntimeError, match="hole or multiple owners"):
        render_projected_scan_panorama(
            _full_projected_sources(),
            exposure_mode="none",
            multiband_levels=1,
        )


def test_projected_three_frame_local_ribbons_prevent_nonadjacent_contact(monkeypatch) -> None:
    modes = iter(("first", "second"))
    monkeypatch.setattr(
        cv2,
        "detail_GraphCutSeamFinder",
        lambda _cost: _DeterministicGraphCut(next(modes)),
    )

    _, info = render_projected_scan_panorama(
        _full_projected_sources(count=3, shape=(40, 90)),
        exposure_mode="none",
        multiband_levels=1,
    )

    assert info.quality_metrics["nonadjacent_owner_boundary_pixel_count"] == 0


def test_projected_owner_boundary_requires_real_common_coverage(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)
    shape = (40, 80)
    image = _solid(*shape, (90, 120, 150))
    depth = np.full(shape, 1000.0, dtype=np.float32)
    left_valid = np.zeros(shape, dtype=np.uint8)
    right_valid = np.zeros(shape, dtype=np.uint8)
    left_valid[:, :40] = 255
    right_valid[:, 40:] = 255
    sources = [
        _projected_source(0, image, left_valid, depth, 20.0),
        _projected_source(1, image, right_valid, depth, 60.0),
    ]

    with pytest.raises(RuntimeError, match="lacks real pair overlap"):
        render_projected_scan_panorama(
            sources,
            exposure_mode="none",
            multiband_levels=1,
        )


def test_projected_missing_adjacent_owner_boundary_fails(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch, graphcut_mode="first")

    with pytest.raises(RuntimeError, match="adjacent owner boundaries are missing"):
        render_projected_scan_panorama(
            _full_projected_sources(),
            exposure_mode="none",
            multiband_levels=1,
        )


def test_projected_multiband_zero_weight_wedge_fails(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch, zero_weight_wedge=True)

    with pytest.raises(RuntimeError, match="zero-weight owner wedge"):
        render_projected_scan_panorama(
            _full_projected_sources(),
            exposure_mode="none",
            multiband_levels=1,
        )


def test_projected_multiband_partitions_overlapping_adjacent_zones() -> None:
    shape = (24, 18)
    images = [_solid(*shape, (100, 120, 140)) for _ in range(3)]
    valid = [np.full(shape, 255, dtype=np.uint8) for _ in range(3)]
    owners = [np.zeros(shape, dtype=np.uint8) for _ in range(3)]
    owners[0][:, :6] = 255
    owners[1][:, 6:12] = 255
    owners[2][:, 12:] = 255
    depths = [np.full(shape, 1000.0, dtype=np.float32) for _ in range(3)]

    blended, output_mask, blend_pixels, _ = (
        render_module._opencv_multiband_owner_boundary_blend(
            images,
            valid,
            owners,
            depths,
            valid,
            depths,
            valid,
            np.asarray([0, 1, 2]),
            levels=1,
            radius=7,
        )
    )

    assert np.all(output_mask == 255)
    assert blend_pixels > 0
    assert blended.shape == (*shape, 3)


def test_projected_pairwise_multiband_blocks_nonadjacent_low_frequency_color() -> None:
    shape = (64, 256)
    images = [
        _solid(*shape, (0, 40, 80)),
        _solid(*shape, (0, 100, 140)),
        _solid(*shape, (255, 160, 200)),
    ]
    valid = [np.full(shape, 255, dtype=np.uint8) for _ in range(3)]
    owners = [np.zeros(shape, dtype=np.uint8) for _ in range(3)]
    owners[0][:, :80] = 255
    owners[1][:, 80:176] = 255
    owners[2][:, 176:] = 255
    depths = [np.full(shape, 1000.0, dtype=np.float32) for _ in range(3)]

    blended, output_mask, _, _ = (
        render_module._opencv_multiband_owner_boundary_blend(
            images,
            valid,
            owners,
            depths,
            valid,
            depths,
            valid,
            np.asarray([0, 1, 2]),
            levels=4,
            radius=16,
        )
    )

    assert np.all(output_mask == 255)
    assert np.max(blended[:, 64:96, 0]) == 0


def test_projected_actual_blend_zone_risk_is_quality_gated(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)

    def report_risky_blend(images, _valid, owner_masks, *_args):
        owner_union = np.logical_or.reduce([mask > 0 for mask in owner_masks])
        output = np.zeros_like(images[0])
        for image, owner in zip(images, owner_masks, strict=True):
            output[owner > 0] = image[owner > 0]
        return output, owner_union.astype(np.uint8) * 255, 64, 0.25

    monkeypatch.setattr(
        render_module,
        "_opencv_multiband_owner_boundary_blend",
        report_risky_blend,
    )

    with pytest.raises(RuntimeError, match="actual MultiBand zone reaches RGB-D risk"):
        render_projected_scan_panorama(
            _full_projected_sources(),
            exposure_mode="none",
            multiband_levels=1,
        )


def test_projected_renderer_treats_black_pixels_as_valid_content(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)

    panorama, info = render_projected_scan_panorama(
        _full_projected_sources(color=(0, 0, 0)),
        exposure_mode="none",
        multiband_levels=1,
    )

    assert panorama.shape == (40, 80, 3)
    assert np.count_nonzero(panorama) == 0
    assert info.crop.as_dict() == {"x": 0, "y": 0, "width": 80, "height": 40}
    assert info.quality_metrics["strict_owner_partition"] is True
    assert info.quality_metrics["multiband_output_mask_complete"] is True


def test_prewarped_renderer_allows_colour_without_measured_depth(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)
    shape = (40, 80)
    image = _solid(*shape, (0, 0, 0))
    colour_valid = np.full(shape, 255, dtype=np.uint8)
    sources = [
        _prewarped_source(image, colour_valid, 20.0),
        _prewarped_source(image, colour_valid, 60.0),
    ]
    true_overlap = np.zeros(shape, dtype=np.uint8)
    true_overlap[:, 36:44] = 255

    panorama, info = render_prewarped_scan_panorama(
        sources,
        source_order=(0, 1),
        owner_boundaries_x=(40.0,),
        max_megapixels=1.0,
        exposure_mode="none",
        multiband_levels=1,
        pair_overlap_masks=(true_overlap,),
    )

    assert panorama.shape == (*shape, 3)
    assert np.count_nonzero(panorama) == 0
    assert info.quality_metrics["strict_owner_partition"] is True
    assert info.quality_metrics["multiband_output_mask_complete"] is True


def test_projected_wrapper_retains_strict_colour_depth_contract() -> None:
    sources = _full_projected_sources()
    sources[0].surface_depth_valid_mask[0, 0] = 0

    with pytest.raises(ValueError, match="world-depth validity must agree"):
        render_projected_scan_panorama(sources, exposure_mode="none")


def test_prewarped_renderer_rejects_reverse_order_and_nonmidpoint_boundary() -> None:
    shape = (40, 80)
    image = _solid(*shape, (90, 120, 150))
    colour_valid = np.full(shape, 255, dtype=np.uint8)
    sources = [
        _prewarped_source(image, colour_valid, 20.0),
        _prewarped_source(image, colour_valid, 60.0),
    ]

    with pytest.raises(RuntimeError, match="not strictly increasing"):
        render_prewarped_scan_panorama(
            sources,
            source_order=(1, 0),
            owner_boundaries_x=(40.0,),
            max_megapixels=1.0,
        )
    with pytest.raises(RuntimeError, match="adjacent-centre midpoints"):
        render_prewarped_scan_panorama(
            sources,
            source_order=(0, 1),
            owner_boundaries_x=(39.0,),
            max_megapixels=1.0,
        )


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        (
            "nonadjacent_owner_boundary_pixel_count",
            1,
            "non-adjacent projected owners touch",
        ),
        (
            "unsupported_owner_boundary_pixel_count",
            1,
            "owner boundary lacks real pair overlap",
        ),
        (
            "final_owner_boundary_pair_count",
            0,
            "adjacent owner boundaries are missing",
        ),
    ],
)
def test_prewarped_owner_topology_is_never_quality_gate_bypassable(
    monkeypatch, metric: str, value: int, message: str
) -> None:
    _install_projected_cv_stubs(monkeypatch)
    original = render_module._evaluate_final_owner_boundaries

    def invalid_metrics(*args, **kwargs):
        metrics = original(*args, **kwargs)
        metrics[metric] = value
        return metrics

    monkeypatch.setattr(render_module, "_evaluate_final_owner_boundaries", invalid_metrics)
    shape = (40, 80)
    image = _solid(*shape, (90, 120, 150))
    colour_valid = np.full(shape, 255, dtype=np.uint8)
    sources = [
        _prewarped_source(image, colour_valid, 20.0),
        _prewarped_source(image, colour_valid, 60.0),
    ]

    with pytest.raises(RuntimeError, match=message):
        render_prewarped_scan_panorama(
            sources,
            source_order=(0, 1),
            owner_boundaries_x=(40.0,),
            max_megapixels=1.0,
            exposure_mode="none",
            multiband_levels=1,
            quality_gate=False,
        )


def test_prewarped_empty_owner_is_never_quality_gate_bypassable(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch, graphcut_mode="first")
    shape = (40, 80)
    image = _solid(*shape, (90, 120, 150))
    colour_valid = np.full(shape, 255, dtype=np.uint8)
    sources = [
        _prewarped_source(image, colour_valid, 20.0),
        _prewarped_source(image, colour_valid, 60.0),
    ]

    with pytest.raises(RuntimeError, match="sources lost all ownership"):
        render_prewarped_scan_panorama(
            sources,
            source_order=(0, 1),
            owner_boundaries_x=(40.0,),
            max_megapixels=1.0,
            exposure_mode="none",
            multiband_levels=1,
            quality_gate=False,
        )


def test_safe_wall_exposure_uses_only_low_residual_background() -> None:
    shape = (64, 128)
    first = _solid(*shape, (80, 80, 80))
    second = _solid(*shape, (120, 120, 120))
    # A flat foreground patch has little local gradient in its interior, so
    # this verifies the residual trimming rather than merely edge exclusion.
    first[18:46, 16:52] = (20, 180, 20)
    second[18:46, 16:52] = (180, 20, 180)
    valid = [np.full(shape, 255, dtype=np.uint8) for _ in range(2)]

    corrected, gains, metrics = render_module._safe_wall_exposure_compensation(
        [first, second],
        valid,
        "safe_wall_smooth_gain",
        order=np.asarray([0, 1]),
        pair_overlap_masks=(np.full(shape, 255, dtype=np.uint8),),
    )

    wall = np.ones(shape, dtype=bool)
    wall[18:46, 16:52] = False
    np.testing.assert_allclose(
        corrected[0][wall].mean(axis=0), corrected[1][wall].mean(axis=0), atol=2.0
    )
    assert gains[1, 0] / gains[0, 0] == pytest.approx(2.0 / 3.0, abs=0.05)
    assert 0 < metrics["safe_exposure_support_pixel_count"] < shape[0] * shape[1]


def test_rgb_disparity_risk_helper_uses_no_depth_inputs() -> None:
    shape = (24, 32)
    image = _solid(*shape, (0, 0, 0))
    valid = np.full(shape, 255, dtype=np.uint8)
    supplied = np.zeros(shape, dtype=np.uint8)
    supplied[10:14, 14:18] = 255

    risk = render_module.compute_rgb_disparity_risk_mask(
        image, image, valid, valid, supplied_risk_mask=supplied
    )

    assert risk.dtype == np.uint8
    assert np.all(risk[10:14, 14:18] == 255)


def test_prewarped_rgb_risk_is_hard_owned_and_never_multiband(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)

    class ForbiddenMultiBand:
        def __init__(self) -> None:
            raise AssertionError("risk/guard pixels must not enter MultiBand")

    monkeypatch.setattr(cv2, "detail_MultiBandBlender", ForbiddenMultiBand)
    shape = (40, 80)
    colour_valid = np.full(shape, 255, dtype=np.uint8)
    first_colour = (20, 40, 60)
    second_colour = (200, 180, 160)
    sources = [
        _prewarped_source(_solid(*shape, first_colour), colour_valid, 20.0),
        _prewarped_source(_solid(*shape, second_colour), colour_valid, 60.0),
    ]
    overlap = np.zeros(shape, dtype=np.uint8)
    overlap[:, 36:44] = 255
    risk = overlap.copy()

    panorama, info = render_prewarped_scan_panorama(
        sources,
        source_order=(0, 1),
        owner_boundaries_x=(40.0,),
        pair_overlap_masks=(overlap,),
        pair_rgb_risk_masks=(risk,),
        exposure_mode="none",
        multiband_levels=5,
        quality_gate=False,
    )

    assert np.all(panorama[:, 36:44] == np.asarray(first_colour))
    assert info.blend_zone_pixel_count == 0
    assert info.quality_metrics["blend_zone_risk_fraction"] == 0.0
    assert info.multiband_levels == 3


def test_pair_blend_radius_tracks_narrower_owner_and_caps_at_eight() -> None:
    shape = (16, 120)

    def owner(width: int) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        mask[:, :width] = 255
        return mask

    assert render_module._pair_blend_radius_from_masks(owner(11), owner(70)) == 2
    assert render_module._pair_blend_radius_from_masks(owner(34), owner(70)) == 6
    assert render_module._pair_blend_radius_from_masks(owner(80), owner(100)) == 8
    assert render_module._effective_narrow_multiband_levels(9) == 3


def test_prewarped_crop_uses_largest_fully_valid_rectangle(monkeypatch) -> None:
    _install_projected_cv_stubs(monkeypatch)
    shape = (40, 80)
    colour_valid = np.full(shape, 255, dtype=np.uint8)
    # The bounding box is the whole canvas, but its top-left notch is invalid.
    # The renderer must crop to an all-valid rectangle rather than return a
    # black/invalid corner just because RGB happens to be non-black elsewhere.
    colour_valid[:10, :20] = 0
    image = _solid(*shape, (40, 80, 120))
    sources = [
        _prewarped_source(image, colour_valid, 20.0),
        _prewarped_source(image, colour_valid, 60.0),
    ]
    overlap = colour_valid.copy()

    panorama, info = render_prewarped_scan_panorama(
        sources,
        source_order=(0, 1),
        owner_boundaries_x=(40.0,),
        pair_overlap_masks=(overlap,),
        exposure_mode="none",
        multiband_levels=1,
        quality_gate=False,
    )

    assert info.crop.as_dict() == {"x": 0, "y": 10, "width": 80, "height": 30}
    assert panorama.shape == (30, 80, 3)
