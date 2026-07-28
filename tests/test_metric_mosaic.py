from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.metric_mosaic import (
    METRIC_WORKING_BYTES_PER_CANVAS_PIXEL,
    MetricMosaicConfig,
    _projected_source_confidence,
    _temporal_depth_tolerance,
    render_metric_mosaic,
)
from panorama_demo.rgbd_projection import (
    PinholeIntrinsics,
    RGBDProjectionFrame,
    estimate_projection_canvas,
    project_rgbd_source,
    project_rgbd_source_compact,
)
from panorama_demo.session import CameraIntrinsics, RGBDFrame


def _write_frame(
    root: Path,
    frame_id: int,
    color: np.ndarray,
    depth: np.ndarray,
) -> RGBDFrame:
    color_path = root / f"{frame_id:04d}.png"
    depth_path = root / f"{frame_id:04d}.depth.png"
    assert cv2.imwrite(str(color_path), color)
    assert cv2.imwrite(str(depth_path), depth.astype(np.uint16))
    return RGBDFrame(
        frame_id=frame_id,
        color_path=color_path,
        aligned_depth_path=depth_path,
        depth_scale_mm_per_unit=1.0,
        timestamp_us=frame_id,
        color_exposure_raw=8,
        color_gain=16,
    )


def _intrinsics(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=100.0,
        fy=100.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
        distortion=(0.0,) * 8,
    )


def _pose(x_mm: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x_mm
    return pose


def test_metric_mosaic_has_aligned_depth_confidence_and_owner(
    tmp_path: Path,
) -> None:
    height, width = 7, 11
    depth = np.full((height, width), 1000, dtype=np.uint16)
    black = np.zeros((height, width, 3), dtype=np.uint8)
    bright = np.full((height, width, 3), (20, 80, 180), dtype=np.uint8)
    frames = [
        _write_frame(tmp_path, 10, black, depth),
        _write_frame(tmp_path, 11, bright, depth),
    ]

    result = render_metric_mosaic(
        frames,
        [_pose(0.0), _pose(20.0)],
        _intrinsics(width, height),
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 1500.0,
            "preview_width": 64,
            "maximum_canvas_megapixels": 1.0,
        },
    )

    result.validate()
    valid = result.valid_mask
    assert result.metadata["coordinate_system"]["pixel_size_mm"] == 2.0
    assert np.all(np.isfinite(result.depth_mm[valid]))
    assert np.all(np.isnan(result.depth_mm[~valid]))
    assert np.all(result.confidence_u16[valid] > 0)
    assert set(np.unique(result.owner_frame_id[valid])) <= {10, 11}
    # Black source pixels remain valid because validity never derives from RGB.
    assert np.any(np.all(result.image_bgr[valid] == 0, axis=1))


def test_metric_mosaic_world_zbuffer_keeps_nearer_surface(
    tmp_path: Path,
) -> None:
    height, width = 5, 9
    far_depth = np.full((height, width), 1000, dtype=np.uint16)
    near_depth = np.full((height, width), 600, dtype=np.uint16)
    far_color = np.full((height, width, 3), (200, 20, 20), dtype=np.uint8)
    near_color = np.full((height, width, 3), (20, 200, 20), dtype=np.uint8)
    frames = [
        _write_frame(tmp_path, 20, far_color, far_depth),
        _write_frame(tmp_path, 21, near_color, near_depth),
    ]

    result = render_metric_mosaic(
        frames,
        [_pose(0.0), _pose(0.0)],
        _intrinsics(width, height),
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 1500.0,
            "preview_width": 64,
            "maximum_canvas_megapixels": 1.0,
        },
    )

    near_owned = result.owner_frame_id == 21
    assert np.any(near_owned)
    assert np.all(result.depth_mm[near_owned] == pytest.approx(600.0))
    assert np.all(result.image_bgr[near_owned] == (20, 200, 20))


def test_metric_mosaic_fixes_near_objects_at_known_world_positions(
    tmp_path: Path,
) -> None:
    height, width = 21, 41
    intrinsics = _intrinsics(width, height)
    # A 24 mm camera translation is exactly four source pixels at 600 mm
    # range with fx=100.  The two views therefore observe the same two near
    # objects at different source pixels but at identical world positions.
    camera_x = (0.0, 24.0)
    object_world_x = (0.0, 60.0)
    object_colors = ((0, 0, 255), (0, 255, 0))
    frames: list[RGBDFrame] = []
    poses: list[np.ndarray] = []
    for frame_index, translation_x in enumerate(camera_x):
        depth = np.full((height, width), 1000, dtype=np.uint16)
        color = np.full((height, width, 3), 32, dtype=np.uint8)
        for world_x, bgr in zip(
            object_world_x, object_colors, strict=True
        ):
            source_x = int(round(
                intrinsics.cx
                + (world_x - translation_x)
                * intrinsics.fx
                / 600.0
            ))
            depth[9:12, source_x - 1 : source_x + 2] = 600
            color[9:12, source_x - 1 : source_x + 2] = bgr
        frames.append(
            _write_frame(
                tmp_path,
                70 + frame_index,
                color,
                depth,
            )
        )
        poses.append(_pose(translation_x))

    result = render_metric_mosaic(
        frames,
        poses,
        intrinsics,
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 1500.0,
            "preview_width": 64,
            "maximum_canvas_megapixels": 2.0,
        },
    )

    coordinate = result.metadata["coordinate_system"]
    pixel_size = float(coordinate["pixel_size_mm"])
    scan_origin = float(coordinate["crop_scan_origin_mm"])
    recovered_x: list[float] = []
    for bgr in object_colors:
        selected = np.all(
            result.image_bgr == np.asarray(bgr, dtype=np.uint8),
            axis=2,
        )
        assert np.any(selected)
        rows, columns = np.nonzero(selected)
        assert set(np.unique(result.owner_frame_id[selected])) <= {70, 71}
        assert np.all(result.depth_mm[selected] == pytest.approx(600.0))
        recovered_x.append(
            float(np.median(scan_origin + columns * pixel_size))
        )

    assert recovered_x[0] == pytest.approx(object_world_x[0], abs=2.0)
    assert recovered_x[1] == pytest.approx(object_world_x[1], abs=2.0)
    assert recovered_x[1] - recovered_x[0] == pytest.approx(60.0, abs=2.0)
    assert result.metadata["single_owner_valid_pixel_count"] == int(
        np.count_nonzero(result.valid_mask)
    )
    assert result.metadata["unowned_valid_pixel_count"] == 0


def test_metric_mosaic_formal_resolution_cannot_be_relaxed() -> None:
    with pytest.raises(ValueError, match="fixed at 2 mm/pixel"):
        MetricMosaicConfig.from_mapping({"millimetres_per_pixel": 1.5})


def test_metric_strict_completion_uses_accepted_surface_support_not_bbox_fill(
    tmp_path: Path,
) -> None:
    height = width = 9
    depth = np.full((height, width), 3000, dtype=np.uint16)
    color = np.arange(
        height * width * 3,
        dtype=np.uint8,
    ).reshape(height, width, 3)
    frames = [
        _write_frame(tmp_path, frame_id, color, depth)
        for frame_id in (80, 81)
    ]
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=1000.0,
        fy=1000.0,
        cx=4.0,
        cy=4.0,
        distortion=(0.0,) * 8,
    )

    result = render_metric_mosaic(
        frames,
        [_pose(0.0), _pose(2.0)],
        intrinsics,
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 3000.0,
            "preview_width": 64,
            "maximum_canvas_megapixels": 1.0,
        },
    )

    audit = result.metadata["surface_footprint_audit"]
    assert result.metadata["strict_v1_metric_complete"] is True
    assert result.metadata["strict_incomplete_reasons"] == []
    assert audit["accepted_continuous_surface_support_ratio"] > 0.01
    assert audit["footprint_rasterized_pixel_count"] > 0
    assert audit["morphological_hole_fill_used"] is False
    # The conservative frustum bbox intentionally includes unobserved cells;
    # strict metric quality is therefore not a rectangular fill-ratio test.
    assert result.metadata["invalid_pixel_count"] > 0


def test_metric_low_continuous_support_is_quality_incomplete_not_structural_failure(
    tmp_path: Path,
) -> None:
    height, width = 7, 11
    depth = np.full((height, width), 1000, dtype=np.uint16)
    color = np.full((height, width, 3), 90, dtype=np.uint8)
    frames = [
        _write_frame(tmp_path, frame_id, color, depth)
        for frame_id in (90, 91)
    ]
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=50.0,
        fy=50.0,
        cx=5.0,
        cy=3.0,
        distortion=(0.0,) * 8,
    )

    result = render_metric_mosaic(
        frames,
        [_pose(0.0), _pose(20.0)],
        intrinsics,
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 1500.0,
            "preview_width": 64,
            "maximum_canvas_megapixels": 1.0,
        },
    )

    result.validate()
    assert result.metadata["strict_v1_metric_complete"] is False
    assert result.metadata["strict_incomplete_reasons"]
    assert (
        result.metadata["surface_footprint_audit"][
            "accepted_continuous_surface_support_ratio"
        ]
        < 0.01
    )


def test_metric_working_budget_includes_delivery_crop_copies() -> None:
    assert METRIC_WORKING_BYTES_PER_CANVAS_PIXEL == 72


def test_metric_reuses_one_undistortion_map_for_all_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 7, 11
    depth = np.full((height, width), 1000, dtype=np.uint16)
    color = np.full((height, width, 3), 80, dtype=np.uint8)
    frames = [
        _write_frame(tmp_path, frame_id, color, depth)
        for frame_id in (41, 42, 43)
    ]
    base = _intrinsics(width, height)
    intrinsics = CameraIntrinsics(
        width=base.width,
        height=base.height,
        fx=base.fx,
        fy=base.fy,
        cx=base.cx,
        cy=base.cy,
        distortion=(0.01, -0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    real_init = cv2.initUndistortRectifyMap
    calls = 0

    def counted_init(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_init(*args, **kwargs)

    monkeypatch.setattr(cv2, "initUndistortRectifyMap", counted_init)
    render_metric_mosaic(
        frames,
        [_pose(0.0), _pose(10.0), _pose(20.0)],
        intrinsics,
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 1500.0,
            "preview_width": 64,
            "maximum_canvas_megapixels": 1.0,
        },
    )

    assert calls == 1


def test_temporal_tolerance_uses_local_camera_range_not_world_origin() -> None:
    config = MetricMosaicConfig(
        temporal_absolute_tolerance_mm=20.0,
        temporal_relative_tolerance=0.02,
    )
    tolerance = _temporal_depth_tolerance(
        np.asarray([500.0, 2000.0], np.float32),
        np.asarray([600.0, 1800.0], np.float32),
        config,
    )
    np.testing.assert_allclose(tolerance, [20.0, 40.0])


def _metric_merge_state(height: int, width: int) -> dict[str, np.ndarray]:
    return {
        "image": np.zeros((height, width, 3), dtype=np.uint8),
        "depth": np.full((height, width), np.inf, dtype=np.float32),
        "camera_depth": np.full((height, width), np.nan, dtype=np.float32),
        "confidence": np.zeros((height, width), dtype=np.float32),
        "owner": np.full((height, width), -1, dtype=np.int32),
        "support": np.zeros((height, width), dtype=np.uint16),
        "conflict": np.zeros((height, width), dtype=np.uint16),
        "edge": np.zeros((height, width), dtype=bool),
    }


def _merge_metric_source_reference(
    state: dict[str, np.ndarray],
    source: object,
    config: MetricMosaicConfig,
    region: tuple[slice, slice],
) -> None:
    image = state["image"][region]
    depth = state["depth"][region]
    camera_depth = state["camera_depth"][region]
    confidence = state["confidence"][region]
    owner = state["owner"][region]
    support = state["support"][region]
    conflict = state["conflict"][region]
    edge = state["edge"][region]
    source_confidence, source_edge = _projected_source_confidence(
        source.camera_depth_mm,
        source.valid_mask,
        config,
    )
    candidate = source_confidence > 0.0
    tolerance = _temporal_depth_tolerance(
        source.camera_depth_mm,
        camera_depth,
        config,
    )
    empty = candidate & (owner < 0)
    overlap = candidate & (owner >= 0)
    delta = source.surface_depth_mm - depth
    same_layer = overlap & (np.abs(delta) <= tolerance)
    nearer = overlap & (delta < -tolerance)
    farther = overlap & (delta > tolerance)
    support[same_layer] = np.minimum(
        support[same_layer].astype(np.uint32) + 1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    conflict[nearer | farther] = np.minimum(
        conflict[nearer | farther].astype(np.uint32) + 1,
        np.iinfo(np.uint16).max,
    ).astype(np.uint16)
    take = empty | nearer | (
        same_layer & (source_confidence > confidence + np.float32(1e-6))
    )
    image[take] = source.warped_rgb[take]
    depth[take] = source.surface_depth_mm[take]
    camera_depth[take] = source.camera_depth_mm[take]
    confidence[take] = source_confidence[take]
    owner[take] = int(source.frame_id)
    edge[take] = source_edge[take]
    support[empty | nearer] = 1


def test_compact_metric_temporal_merge_matches_full_canvas_per_pixel() -> None:
    height, width = 9, 17
    camera = PinholeIntrinsics(
        width=width,
        height=height,
        fx=160.0,
        fy=160.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
    )
    colors = [
        np.full((height, width, 3), value, dtype=np.uint8)
        for value in ((20, 40, 60), (80, 100, 120), (140, 160, 180))
    ]
    depths = [
        np.full((height, width), 1000.0, dtype=np.float32),
        np.full((height, width), 1008.0, dtype=np.float32),
        np.full((height, width), 700.0, dtype=np.float32),
    ]
    depths[1][2:5, 5:9] = 0.0
    depths[2][:, :8] = 0.0
    projection_frames = [
        RGBDProjectionFrame(index + 30, color, depth, _pose(12.0 * index))
        for index, (color, depth) in enumerate(zip(colors, depths, strict=True))
    ]
    canvas = estimate_projection_canvas(
        projection_frames,
        camera,
        max_canvas_megapixels=1.0,
        millimetres_per_pixel=2.0,
        maximum_depth_mm=1500.0,
        maximum_resident_sources=1,
    )
    config = MetricMosaicConfig(
        minimum_depth_mm=100.0,
        maximum_depth_mm=1500.0,
        preview_width=64,
        maximum_canvas_megapixels=1.0,
    )
    legacy_state = _metric_merge_state(canvas.height, canvas.width)
    compact_state = _metric_merge_state(canvas.height, canvas.width)

    for frame in projection_frames:
        full = project_rgbd_source(
            frame,
            camera,
            canvas,
            chunk_rows=3,
            maximum_depth_mm=1500.0,
        )
        compact = project_rgbd_source_compact(
            frame,
            camera,
            canvas,
            chunk_rows=3,
            maximum_depth_mm=1500.0,
        )
        _merge_metric_source_reference(
            legacy_state,
            full,
            config,
            (slice(0, canvas.height), slice(0, canvas.width)),
        )
        _merge_metric_source_reference(
            compact_state,
            compact,
            config,
            compact.canvas_slices,
        )

    for name in legacy_state:
        np.testing.assert_array_equal(compact_state[name], legacy_state[name])
