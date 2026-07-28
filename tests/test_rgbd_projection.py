from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.rgbd_projection import (
    PinholeIntrinsics,
    ProjectedRGBDSource,
    RGBDProjectionFrame,
    RGBDProjectionResult,
    estimate_projection_canvas,
    estimate_side_scan_footprints,
    expand_compact_rgbd_source,
    project_rgbd_source,
    project_rgbd_source_compact,
    project_selected_rgbd_sources,
)


def _intrinsics(width: int, height: int, focal: float = 1000.0) -> PinholeIntrinsics:
    return PinholeIntrinsics(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
    )


def _pose(x_mm: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x_mm
    return pose


def _frame(
    frame_id: int,
    rgb: np.ndarray,
    depth_mm: np.ndarray,
    x_mm: float = 0.0,
) -> RGBDProjectionFrame:
    return RGBDProjectionFrame(frame_id, rgb, depth_mm, _pose(x_mm))


def _canvas_pixel(result, world_point_mm: tuple[float, float, float]) -> tuple[int, int]:
    point = result.canvas.world_to_canvas(np.asarray(world_point_mm, dtype=np.float64))
    return int(np.rint(point[0])), int(np.rint(point[1]))


def test_black_rgb_is_valid_projected_content() -> None:
    height, width = 5, 7
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1000.0, dtype=np.float32)

    result = project_selected_rgbd_sources(
        [_frame(7, rgb, depth)], _intrinsics(width, height), max_canvas_megapixels=1.0
    )
    source = result.sources[0]

    assert np.count_nonzero(source.valid_mask) == height * width
    assert np.all(source.warped_rgb[source.valid_mask > 0] == 0)
    np.testing.assert_array_equal(source.valid_mask, source.surface_depth_valid_mask)
    np.testing.assert_array_equal(source.valid_mask, source.camera_depth_valid_mask)
    assert np.all(source.surface_depth_mm[source.valid_mask > 0] == 1000.0)
    assert np.all(source.camera_depth_mm[source.valid_mask > 0] == 1000.0)
    assert source.sampling_stats["point_centres_preserved"] is True
    assert source.sampling_stats["point_splat_only"] is False
    assert source.sampling_stats["nearest_measured_rgb_only"] is True


def test_invalid_depth_hole_is_not_fabricated() -> None:
    height = width = 7
    rgb = np.full((height, width, 3), 180, dtype=np.uint8)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    depth[height // 2, width // 2] = 0.0

    result = project_selected_rgbd_sources(
        [_frame(0, rgb, depth)], _intrinsics(width, height), max_canvas_megapixels=1.0
    )
    x, y = _canvas_pixel(result, (0.0, 0.0, 1000.0))
    source = result.sources[0]

    assert source.valid_mask[y, x] == 0
    assert source.surface_depth_valid_mask[y, x] == 0
    assert source.surface_depth_mm[y, x] == 0.0
    assert source.camera_depth_valid_mask[y, x] == 0
    assert source.camera_depth_mm[y, x] == 0.0
    assert source.sampling_stats["selected_zbuffer_pixel_count"] == width * height - 1


def test_per_source_zbuffer_keeps_nearest_world_surface() -> None:
    height, width = 3, 5
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.zeros((height, width), dtype=np.float32)
    # u=1,z=1000 and u=2,z=500 both reconstruct to world x=1 mm.
    depth[1, 1] = 1000.0
    depth[1, 2] = 500.0
    rgb[1, 1] = (20, 40, 60)
    rgb[1, 2] = (90, 110, 130)
    camera = PinholeIntrinsics(width, height, 1000.0, 1000.0, 0.0, 1.0)

    result = project_selected_rgbd_sources(
        [_frame(0, rgb, depth)], camera, max_canvas_megapixels=1.0
    )
    x, y = _canvas_pixel(result, (1.0, 0.0, 500.0))
    source = result.sources[0]

    np.testing.assert_array_equal(source.warped_rgb[y, x], (90, 110, 130))
    assert source.surface_depth_mm[y, x] == 500.0
    assert source.sampling_stats["zbuffer_collision_count"] == 1


def test_metric_reprojection_aligns_near_and_far_layers_across_translation() -> None:
    height, width = 5, 21
    camera = PinholeIntrinsics(width, height, 100.0, 100.0, 10.0, 2.0)
    first_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    second_rgb = np.zeros_like(first_rgb)
    first_depth = np.zeros((height, width), dtype=np.float32)
    second_depth = np.zeros_like(first_depth)
    # World points: near=(0,0,500), far=(40,0,1000).  A camera translated
    # +10 mm observes depth-dependent pixel shifts of two and one pixels.
    first_depth[2, 10] = 500.0
    second_depth[2, 8] = 500.0
    first_depth[2, 14] = 1000.0
    second_depth[2, 13] = 1000.0
    first_rgb[2, 10] = second_rgb[2, 8] = (220, 30, 20)
    first_rgb[2, 14] = second_rgb[2, 13] = (20, 30, 220)

    result = project_selected_rgbd_sources(
        [_frame(0, first_rgb, first_depth), _frame(1, second_rgb, second_depth, 10.0)],
        camera,
        max_canvas_megapixels=1.0,
    )
    near_x, y = _canvas_pixel(result, (0.0, 0.0, 500.0))
    far_x, far_y = _canvas_pixel(result, (40.0, 0.0, 1000.0))

    assert near_x != far_x
    assert y == far_y
    for source in result.sources:
        assert source.valid_mask[y, near_x] == 255
        assert source.valid_mask[y, far_x] == 255
        assert source.surface_depth_mm[y, near_x] == 500.0
        assert source.surface_depth_mm[y, far_x] == 1000.0


def test_depth_discontinuity_is_counted_and_never_crossed() -> None:
    height, width = 4, 8
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    depth = np.full((height, width), 2000.0, dtype=np.float32)
    depth[:, :4] = 500.0

    result = project_selected_rgbd_sources(
        [_frame(0, rgb, depth)], _intrinsics(width, height), max_canvas_megapixels=1.0
    )
    source = result.sources[0]

    assert source.sampling_stats["depth_discontinuity_edge_count"] == height
    assert source.sampling_stats["point_splat_only"] is False
    assert source.sampling_stats["rejected_depth_edge_sample_count"] > 0
    assert source.sampling_stats["surface_footprint_continuity_gate_used"] is True


def test_sparse_footprints_report_metric_scan_coverage_without_warps() -> None:
    height, width = 6, 10
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    estimate = estimate_side_scan_footprints(
        [_frame(10, rgb, depth), _frame(11, rgb, depth, 100.0)],
        _intrinsics(width, height),
        working_width=5,
    )

    assert len(estimate.footprints) == 2
    assert estimate.footprints[1].camera_center_scan_x_mm == pytest.approx(100.0)
    assert estimate.footprints[1].scan_x_interval_mm[0] == pytest.approx(
        estimate.footprints[0].scan_x_interval_mm[0] + 100.0
    )
    assert "warped_rgb" not in estimate.as_dict()["footprints"][0]


@pytest.mark.parametrize(
    "bad_pose",
    [
        np.full((4, 4), np.nan),
        np.diag([2.0, 1.0, 1.0, 1.0]),
        np.diag([-1.0, 1.0, 1.0, 1.0]),
        np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]],
            dtype=np.float64,
        ),
    ],
)
def test_projection_rejects_non_finite_or_non_se3_pose(bad_pose: np.ndarray) -> None:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.full((3, 4), 1000.0, dtype=np.float32)
    frame = RGBDProjectionFrame(0, rgb, depth, bad_pose)

    with pytest.raises(ValueError, match="camera_to_world"):
        estimate_projection_canvas([frame], _intrinsics(4, 3), max_canvas_megapixels=1.0)


def test_projection_rejects_pose_translation_in_implicit_open3d_metres() -> None:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    depth = np.full((3, 4), 1000.0, dtype=np.float32)
    frame = RGBDProjectionFrame(
        0, rgb, depth, np.eye(4, dtype=np.float64), camera_to_world_unit="m"
    )

    with pytest.raises(ValueError, match="explicitly in mm"):
        estimate_projection_canvas([frame], _intrinsics(4, 3), max_canvas_megapixels=1.0)


def test_projection_rejects_canvas_and_aggregate_working_set_before_allocation() -> None:
    height = width = 100
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    frames = [_frame(0, rgb, depth), _frame(1, rgb, depth)]
    camera = _intrinsics(width, height)

    with pytest.raises(MemoryError, match="Orthographic canvas"):
        estimate_projection_canvas(frames, camera, max_canvas_megapixels=0.005)
    with pytest.raises(MemoryError, match="aggregate working set"):
        estimate_projection_canvas(
            frames,
            camera,
            max_canvas_megapixels=1.0,
            max_aggregate_megapixels=0.015,
        )


def test_metric_canvas_honours_fixed_two_millimetre_grid() -> None:
    height, width = 5, 9
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    result = project_selected_rgbd_sources(
        [_frame(0, rgb, depth)],
        _intrinsics(width, height),
        max_canvas_megapixels=1.0,
        millimetres_per_pixel=2.0,
    )

    assert result.canvas.pixels_per_mm == pytest.approx(0.5)
    assert result.canvas.as_dict()["millimetres_per_pixel"] == pytest.approx(2.0)


def test_streaming_projection_budget_counts_only_resident_sources() -> None:
    height = width = 100
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    frames = [_frame(index, rgb, depth, 5.0 * index) for index in range(8)]

    canvas = estimate_projection_canvas(
        frames,
        _intrinsics(width, height),
        max_canvas_megapixels=1.0,
        max_aggregate_megapixels=0.02,
        maximum_resident_sources=1,
    )

    assert canvas.aggregate_megapixels == pytest.approx(
        canvas.canvas_megapixels
    )


def test_compact_projection_expands_pixel_exactly_to_full_canvas_api() -> None:
    height, width = 9, 15
    camera = _intrinsics(width, height, focal=180.0)
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    depth[:, :3] = 650.0
    depth[2:5, 7:10] = 0.0
    frame = _frame(17, rgb, depth, 35.0)
    canvas = estimate_projection_canvas(
        [frame],
        camera,
        max_canvas_megapixels=1.0,
        millimetres_per_pixel=2.0,
        maximum_resident_sources=1,
    )

    legacy_full = project_rgbd_source(frame, camera, canvas, chunk_rows=3)
    compact = project_rgbd_source_compact(frame, camera, canvas, chunk_rows=3)
    expanded = expand_compact_rgbd_source(compact, canvas)

    assert compact.warped_rgb.shape[:2] == (
        compact.valid_bbox[3] - compact.valid_bbox[1],
        compact.valid_bbox[2] - compact.valid_bbox[0],
    )
    assert compact.warped_rgb.shape[0] <= canvas.height
    assert compact.warped_rgb.shape[1] <= canvas.width
    for name in (
        "warped_rgb",
        "valid_mask",
        "surface_depth_mm",
        "surface_depth_valid_mask",
        "camera_depth_mm",
        "camera_depth_valid_mask",
    ):
        np.testing.assert_array_equal(
            getattr(expanded, name),
            getattr(legacy_full, name),
        )
    assert expanded.as_dict() == legacy_full.as_dict()


def test_configured_depth_frustum_contains_full_resolution_extrema() -> None:
    height, width = 9, 15
    camera = _intrinsics(width, height, focal=180.0)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    preview_depth = np.full((height, width), 900.0, dtype=np.float32)
    full_depth = preview_depth.copy()
    full_depth[0, 0] = 3000.0
    preview = _frame(3, rgb, preview_depth, 20_000.0)
    full = _frame(3, rgb, full_depth, 20_000.0)

    canvas = estimate_projection_canvas(
        [preview],
        camera,
        max_canvas_megapixels=1.0,
        maximum_depth_mm=3000.0,
        millimetres_per_pixel=2.0,
        maximum_resident_sources=1,
    )
    projected = project_rgbd_source_compact(
        full,
        camera,
        canvas,
        chunk_rows=3,
        maximum_depth_mm=3000.0,
    )

    assert projected.sampling_stats["selected_zbuffer_pixel_count"] > 0
    assert projected.sampling_stats["out_of_canvas_sample_count"] == 0
    assert projected.valid_bbox[0] >= 0
    assert projected.valid_bbox[2] <= canvas.width


def _full_depth_plane(
    *,
    depth_mm: np.ndarray,
    rgb: np.ndarray | None = None,
    fx: float = 1000.0,
    fy: float = 1000.0,
    cx: float | None = None,
    cy: float | None = None,
    millimetres_per_pixel: float = 2.0,
    maximum_depth_mm: float = 3000.0,
) -> tuple[RGBDProjectionResult, ProjectedRGBDSource]:
    height, width = depth_mm.shape
    camera = PinholeIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=(width - 1) * 0.5 if cx is None else cx,
        cy=(height - 1) * 0.5 if cy is None else cy,
    )
    color = (
        np.zeros((height, width, 3), dtype=np.uint8)
        if rgb is None
        else np.asarray(rgb, dtype=np.uint8)
    )
    result = project_selected_rgbd_sources(
        [_frame(0, color, np.asarray(depth_mm, dtype=np.float32))],
        camera,
        max_canvas_megapixels=2.0,
        maximum_depth_mm=maximum_depth_mm,
        millimetres_per_pixel=millimetres_per_pixel,
    )
    return result, result.sources[0]


def test_continuous_far_plane_rasterizes_measured_pixel_footprints() -> None:
    height = width = 9
    depth = np.full((height, width), 3000.0, dtype=np.float32)
    result, source = _full_depth_plane(depth_mm=depth)

    point_centres = np.zeros(source.valid_mask.shape, dtype=bool)
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    camera_points = np.stack(
        (
            (columns - 4.0) * depth / 1000.0,
            (rows - 4.0) * depth / 1000.0,
            depth,
        ),
        axis=-1,
    )
    projected = result.canvas.world_to_canvas(camera_points.reshape(-1, 3))
    point_x = np.rint(projected[:, 0]).astype(np.int64)
    point_y = np.rint(projected[:, 1]).astype(np.int64)
    point_centres[point_y, point_x] = True

    footprint_only = (source.valid_mask > 0) & ~point_centres
    assert np.any(footprint_only)
    assert source.sampling_stats["measured_center_candidate_count"] == 81
    assert source.sampling_stats["continuous_surface_sample_count"] == 49
    assert source.sampling_stats["footprint_rasterized_pixel_count"] == int(
        np.count_nonzero(footprint_only)
    )
    assert source.sampling_stats["morphological_hole_fill_used"] is False


def test_footprint_rgb_and_depth_are_nearest_real_source_samples() -> None:
    height = width = 9
    depth = np.full((height, width), 3000.0, dtype=np.float32)
    sample_id = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    rgb = np.stack(
        (
            sample_id,
            np.bitwise_xor(sample_id, np.uint8(0x5A)),
            np.bitwise_xor(sample_id, np.uint8(0xC3)),
        ),
        axis=-1,
    )
    result, source = _full_depth_plane(depth_mm=depth, rgb=rgb)

    source_palette = {tuple(value) for value in rgb.reshape(-1, 3).tolist()}
    output_palette = {
        tuple(value)
        for value in source.warped_rgb[source.valid_mask > 0].tolist()
    }
    assert output_palette <= source_palette
    assert set(np.unique(source.surface_depth_mm[source.valid_mask > 0])) == {
        3000.0
    }
    assert set(np.unique(source.camera_depth_mm[source.valid_mask > 0])) == {
        3000.0
    }

    # At equal surface depth, the exact winning colour must be the measured
    # source centre nearest to the target grid point; no bilinear colour can
    # satisfy this identity for the unique source palette.
    rows, columns = np.indices(depth.shape, dtype=np.float64)
    camera_points = np.stack(
        (
            (columns - 4.0) * depth / 1000.0,
            (rows - 4.0) * depth / 1000.0,
            depth,
        ),
        axis=-1,
    )
    projected = result.canvas.world_to_canvas(camera_points.reshape(-1, 3))
    centre_mask = np.zeros(source.valid_mask.shape, dtype=bool)
    centre_mask[
        np.rint(projected[:, 1]).astype(np.int64),
        np.rint(projected[:, 0]).astype(np.int64),
    ] = True
    footprint_y, footprint_x = np.argwhere(
        (source.valid_mask > 0) & ~centre_mask
    )[0]
    distances = np.square(projected[:, 0] - footprint_x) + np.square(
        projected[:, 1] - footprint_y
    )
    expected = rgb.reshape(-1, 3)[int(np.argmin(distances))]
    np.testing.assert_array_equal(
        source.warped_rgb[footprint_y, footprint_x],
        expected,
    )


def test_surface_footprints_do_not_cross_depth_step() -> None:
    height, width = 9, 10
    depth = np.full((height, width), 2000.0, dtype=np.float32)
    depth[:, :5] = 500.0
    result, source = _full_depth_plane(
        depth_mm=depth,
        fx=1000.0,
        fy=1000.0,
        cx=4.5,
        cy=4.0,
        millimetres_per_pixel=0.25,
        maximum_depth_mm=2000.0,
    )
    y = int(
        np.rint(
            result.canvas.world_to_canvas(np.array([0.0, 0.0, 1000.0]))[1]
        )
    )
    for world_x in (0.0, 0.25, 0.5, 0.75):
        x = int(
            np.rint(
                result.canvas.world_to_canvas(
                    np.array([world_x, 0.0, 1000.0])
                )[0]
            )
        )
        assert source.valid_mask[y, x] == 0
    assert source.sampling_stats["depth_discontinuity_edge_count"] == height
    assert source.sampling_stats["rejected_depth_edge_sample_count"] > 0
    assert source.sampling_stats["footprint_rasterized_pixel_count"] > 0


def test_surface_footprints_preserve_invalid_depth_hole() -> None:
    height = width = 9
    depth = np.full((height, width), 3000.0, dtype=np.float32)
    depth[4, 4] = 0.0
    result, source = _full_depth_plane(depth_mm=depth)
    x, y = np.rint(
        result.canvas.world_to_canvas(np.array([0.0, 0.0, 3000.0]))
    ).astype(np.int64)

    assert source.valid_mask[y, x] == 0
    assert source.surface_depth_valid_mask[y, x] == 0
    assert source.camera_depth_valid_mask[y, x] == 0
    assert source.surface_depth_mm[y, x] == 0.0
    assert source.camera_depth_mm[y, x] == 0.0
    assert (
        source.sampling_stats["rejected_invalid_neighbourhood_sample_count"]
        >= 9
    )
    assert source.sampling_stats["footprint_rasterized_pixel_count"] > 0


def test_depth_continuous_projected_fold_is_point_only() -> None:
    height = width = 7
    depth = np.tile(
        1000.0 - 10.0 * np.arange(width, dtype=np.float32),
        (height, 1),
    )
    _, source = _full_depth_plane(
        depth_mm=depth,
        fx=100.0,
        fy=100.0,
        cx=-100.0,
        cy=3.0,
        millimetres_per_pixel=10.0,
        maximum_depth_mm=1200.0,
    )

    assert source.sampling_stats["rejected_depth_edge_sample_count"] == 0
    assert source.sampling_stats["rejected_fold_sample_count"] == 25
    assert source.sampling_stats["continuous_surface_sample_count"] == 0
    assert source.sampling_stats["footprint_candidate_count"] == 0
    assert source.sampling_stats["footprint_rasterized_pixel_count"] == 0
    assert source.sampling_stats["point_centres_preserved"] is True


def test_slanted_surface_footprints_stay_within_one_metric_grid_cell() -> None:
    height = width = 9
    rows, columns = np.indices((height, width), dtype=np.float32)
    depth = 2960.0 + 2.0 * columns + rows
    sample_id = np.arange(height * width, dtype=np.uint8).reshape(height, width)
    rgb = np.stack((sample_id, sample_id, sample_id), axis=-1)
    result, source = _full_depth_plane(depth_mm=depth, rgb=rgb)
    assert source.sampling_stats["footprint_rasterized_pixel_count"] > 0

    scan = np.asarray(result.canvas.scan_axis)
    down = -np.asarray(result.canvas.up_axis)
    min_scan, min_down, _, _ = result.canvas.world_bounds
    valid_y, valid_x = np.nonzero(source.valid_mask)
    for canvas_y, canvas_x in zip(valid_y, valid_x, strict=True):
        source_id = int(source.warped_rgb[canvas_y, canvas_x, 0])
        source_y, source_x = divmod(source_id, width)
        z = float(depth[source_y, source_x])
        measured = np.array(
            [
                (source_x - 4.0) * z / 1000.0,
                (source_y - 4.0) * z / 1000.0,
                z,
            ]
        )
        grid_scan = min_scan + canvas_x * 2.0
        grid_down = min_down + canvas_y * 2.0
        assert abs(grid_scan - float(measured @ scan)) <= 2.0 + 1e-6
        assert abs(grid_down - float(measured @ down)) <= 2.0 + 1e-6
        assert source.surface_depth_mm[canvas_y, canvas_x] == pytest.approx(
            float(measured @ np.asarray(result.canvas.normal_axis))
        )
