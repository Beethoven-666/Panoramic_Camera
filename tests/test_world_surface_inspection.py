from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from panorama_demo.session import CameraIntrinsics, RGBDFrame
from panorama_demo.world_surface_inspection import (
    _AutomaticInstanceObservation,
    _automatic_observation_consistency,
    AutomaticInstanceConfig,
    WorldSurfaceInspectionConfig,
    render_world_surface_inspection,
)


def _frame(
    root: Path,
    frame_id: int,
    image: np.ndarray,
    depth: np.ndarray,
) -> RGBDFrame:
    image_path = root / f"{frame_id}.png"
    depth_path = root / f"{frame_id}.depth.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(depth_path), depth.astype(np.uint16))
    return RGBDFrame(
        frame_id=frame_id,
        color_path=image_path,
        aligned_depth_path=depth_path,
        depth_scale_mm_per_unit=1.0,
    )


def _pose(x_mm: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x_mm
    return pose


def test_world_surface_uses_real_rgb_and_one_owner_per_pixel(
    tmp_path: Path,
) -> None:
    height, width = 48, 64
    depth = np.full((height, width), 500, dtype=np.uint16)
    first = np.full((height, width, 3), (20, 40, 80), dtype=np.uint8)
    second = np.full((height, width, 3), (120, 160, 200), dtype=np.uint8)
    frames = [
        _frame(tmp_path, 3, first, depth),
        _frame(tmp_path, 7, second, depth),
    ]
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=80.0,
        fy=80.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
        distortion=(0.0,) * 8,
    )

    result = render_world_surface_inspection(
        frames,
        [_pose(0.0), _pose(30.0)],
        intrinsics,
        config=WorldSurfaceInspectionConfig(
            minimum_depth_mm=100.0,
            maximum_depth_mm=1000.0,
            depth_mesh_cell_size_pixels=2,
            maximum_canvas_megapixels=1.0,
        ),
    )

    result.validate()
    valid_colours = np.unique(result.image_bgr[result.valid_mask], axis=0)
    allowed = {tuple(first[0, 0]), tuple(second[0, 0])}
    assert {tuple(value) for value in valid_colours} <= allowed
    assert set(np.unique(result.owner_frame_id[result.valid_mask])) <= {3, 7}
    assert result.metadata["real_pose_count"] == 2
    assert result.metadata["all_real_poses_consumed"] is True
    assert result.metadata["no_rgb_interpolation"] is True
    assert result.metadata["no_hole_fill"] is True
    assert result.metadata["no_tsdf"] is True
    assert result.metadata["positive_jacobian_required"] is True
    assert result.metadata["complete_3x3_depth_support_required"] is True
    assert all(
        source["mesh"]["rejected_jacobian_cell_count"] == 0
        for source in result.metadata["sources"]
    )


def test_world_surface_rejects_cells_across_depth_boundary(
    tmp_path: Path,
) -> None:
    height, width = 48, 64
    depth = np.full((height, width), 500, dtype=np.uint16)
    depth[:, width // 2 :] = 800
    image = np.full((height, width, 3), (10, 70, 130), dtype=np.uint8)
    frames = [
        _frame(tmp_path, 11, image, depth),
        _frame(tmp_path, 12, image, depth),
    ]
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=80.0,
        fy=80.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
        distortion=(0.0,) * 8,
    )

    result = render_world_surface_inspection(
        frames,
        [_pose(0.0), _pose(30.0)],
        intrinsics,
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 1000.0,
            "depth_mesh_cell_size_pixels": 2,
            "maximum_canvas_megapixels": 1.0,
        },
    )

    rejected = sum(
        int(source["depth_edge_rejected_pixel_count"])
        for source in result.metadata["sources"]
    )
    rejected_cells = sum(
        int(source["mesh"]["rejected_invalid_or_boundary_cell_count"])
        + int(source["mesh"]["rejected_discontinuous_cell_count"])
        for source in result.metadata["sources"]
    )
    assert rejected > 0
    assert rejected_cells > 0
    assert result.metadata["depth_boundary_crossing_allowed"] is False


def test_world_surface_component_lock_uses_one_reprojected_rgb_owner(
    tmp_path: Path,
) -> None:
    height, width = 48, 64
    first_depth = np.full((height, width), 800, dtype=np.uint16)
    second_depth = first_depth.copy()
    first_depth[14:36, 20:42] = 400
    # Same compact world object after a +20 mm camera translation.
    second_depth[14:36, 16:38] = 400
    first = np.full((height, width, 3), (15, 45, 90), dtype=np.uint8)
    second = np.full((height, width, 3), (120, 180, 220), dtype=np.uint8)
    frames = [
        _frame(tmp_path, 21, first, first_depth),
        _frame(tmp_path, 22, second, second_depth),
    ]
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=80.0,
        fy=80.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
        distortion=(0.0,) * 8,
    )

    result = render_world_surface_inspection(
        frames,
        [_pose(0.0), _pose(20.0)],
        intrinsics,
        config={
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 1000.0,
            "depth_mesh_cell_size_pixels": 2,
            "maximum_canvas_megapixels": 1.0,
            "component_minimum_pixels": 40,
            "component_minimum_single_owner_coverage": 0.40,
        },
    )

    audit = result.metadata["component_owner_lock"]
    accepted = [
        component
        for component in audit["components"]
        if component["accepted"]
    ]
    assert accepted
    assert audit["one_real_frame_per_accepted_component"] is True
    assert audit["no_cross_owner_fill"] is True
    for component in accepted:
        assert component["owner_count_after"] == 1
        label = int(component["component_id"])
        mask = result.component_label == label
        locked_owners = np.unique(
            result.component_locked_owner_frame_id[
                mask & result.component_locked_valid_mask
            ]
        )
        assert locked_owners.tolist() == [component["selected_frame_id"]]


def test_automatic_instance_requires_two_view_world_and_target_consistency() -> None:
    first_mask = np.zeros((40, 60), dtype=bool)
    second_mask = np.zeros_like(first_mask)
    first_mask[10:30, 20:40] = True
    second_mask[11:31, 21:41] = True

    def observation(
        frame_id: int,
        mask: np.ndarray,
        centroid_x: float,
    ) -> _AutomaticInstanceObservation:
        return _AutomaticInstanceObservation(
            frame_id=frame_id,
            panel_index=frame_id,
            source_mask=np.zeros((40, 60), dtype=bool),
            target_mask=mask,
            source_seed_pixel_count=100,
            source_mask_pixel_count=400,
            target_pixel_count=int(np.count_nonzero(mask)),
            source_bbox_xywh=(10, 10, 20, 20),
            target_bbox_xywh=_bbox(mask),
            world_min_sdn_mm=(centroid_x - 20.0, -20.0, 350.0),
            world_max_sdn_mm=(centroid_x + 20.0, 20.0, 390.0),
            world_centroid_sdn_mm=(centroid_x, 0.0, 370.0),
            median_target_depth_mm=370.0,
            centrality=0.9,
            sharpness=100.0,
            audit={},
        )

    first = observation(1, first_mask, 100.0)
    consistent = observation(2, second_mask, 105.0)
    inconsistent = observation(3, second_mask, 300.0)
    config = AutomaticInstanceConfig()

    passed, audit = _automatic_observation_consistency(
        first, consistent, config
    )
    assert passed is True
    assert audit["pass"] is True
    rejected, rejected_audit = _automatic_observation_consistency(
        first, inconsistent, config
    )
    assert rejected is False
    assert rejected_audit["world_centroid_distance_mm"] > (
        config.maximum_world_centroid_distance_mm
    )


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    yy, xx = np.nonzero(mask)
    return (
        int(xx.min()),
        int(yy.min()),
        int(xx.max() - xx.min() + 1),
        int(yy.max() - yy.min() + 1),
    )
