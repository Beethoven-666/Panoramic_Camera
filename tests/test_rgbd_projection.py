from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.rgbd_projection import (
    PinholeIntrinsics,
    RGBDProjectionFrame,
    estimate_projection_canvas,
    estimate_side_scan_footprints,
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
    assert source.sampling_stats["point_splat_only"] is True


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


def test_depth_discontinuity_is_counted_but_never_triangulated() -> None:
    height, width = 4, 8
    rgb = np.full((height, width, 3), 100, dtype=np.uint8)
    depth = np.full((height, width), 2000.0, dtype=np.float32)
    depth[:, :4] = 500.0

    result = project_selected_rgbd_sources(
        [_frame(0, rgb, depth)], _intrinsics(width, height), max_canvas_megapixels=1.0
    )
    source = result.sources[0]

    assert source.sampling_stats["depth_discontinuity_edge_count"] == height
    assert source.sampling_stats["point_splat_only"] is True
    # No triangle rasterizer can create more owned cells than input samples.
    assert np.count_nonzero(source.valid_mask) <= height * width


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
