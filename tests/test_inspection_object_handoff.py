from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.inspection_object_handoff import (
    ObjectHandoffSource,
    build_object_owner_interval,
    fit_complete_object_owner,
    project_complete_object_owner_from_rgbd,
    select_automatic_complete_object_owner,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
)
from panorama_demo.session import CameraIntrinsics


def test_object_owner_interval_covers_shifted_copies_contiguously() -> None:
    shape = (30, 80)
    first = np.zeros(shape, dtype=bool)
    second = np.zeros(shape, dtype=bool)
    first[10:20, 25:40] = True
    second[10:20, 38:55] = True
    result = build_object_owner_interval(
        panel_index=2,
        view_dependent_footprints=(first, second),
        selected_panel_valid_mask=np.ones(shape, dtype=bool),
        horizontal_guard_pixels=2,
        vertical_guard_pixels=1,
    )
    for row in np.flatnonzero(np.any(result.lock_mask, axis=1)):
        columns = np.flatnonzero(result.lock_mask[row])
        assert np.all(result.lock_mask[row, columns[0] : columns[-1] + 1])
    assert np.all(result.lock_mask[result.union_footprint])
    assert result.audit["row_contiguous"] is True
    assert result.audit["single_owner"] is True


def test_object_owner_interval_fails_when_selected_view_is_incomplete() -> None:
    footprint = np.zeros((20, 50), dtype=bool)
    footprint[5:15, 15:35] = True
    valid = np.ones(footprint.shape, dtype=bool)
    valid[:, 30:] = False
    with pytest.raises(RuntimeError, match="complete handoff interval"):
        build_object_owner_interval(
            panel_index=1,
            view_dependent_footprints=(footprint,),
            selected_panel_valid_mask=valid,
        )


def test_complete_object_owner_uses_held_out_rgbd_correspondences() -> None:
    height, width = 80, 100
    source = np.zeros((height, width, 3), dtype=np.uint8)
    source[20:55, 30:65] = (17, 93, 211)
    source_mask = np.zeros((height, width), dtype=bool)
    source_mask[20:55, 30:65] = True

    # target = source + (12, 7), expressed as target-to-source maps.
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx - 12.0
    map_y = yy - 7.0
    valid = (
        (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    result = fit_complete_object_owner(
        source_image_bgr=source,
        source_object_mask=source_mask,
        mesh_map_x=map_x,
        mesh_map_y=map_y,
        mesh_valid_mask=valid,
        target_corner_x=0,
        target_shape=(height, width),
        frame_id=9,
        panel_index=2,
        minimum_correspondences=80,
    )
    expected = np.zeros_like(source_mask)
    expected[27:62, 42:77] = True
    assert np.array_equal(result.target_mask, expected)
    assert np.all(result.target_image_bgr[result.target_mask] == (17, 93, 211))
    assert result.audit["held_out_p95_pixels"] < 1e-4
    assert result.audit["rgb_generated"] is False
    assert result.audit["pose_modified"] is False
    assert result.audit["blend_used"] is False


def test_complete_object_owner_rejects_nonlocal_mesh() -> None:
    source = np.zeros((60, 80, 3), dtype=np.uint8)
    mask = np.zeros(source.shape[:2], dtype=bool)
    mask[10:45, 20:55] = True
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    # Alternating target rows cannot be represented by one local similarity.
    map_x = xx + np.where((yy.astype(np.int32) % 2) == 0, 0.0, 20.0)
    map_y = yy
    with pytest.raises(RuntimeError):
        fit_complete_object_owner(
            source_image_bgr=source,
            source_object_mask=mask,
            mesh_map_x=map_x,
            mesh_map_y=map_y,
            mesh_valid_mask=np.ones(mask.shape, dtype=bool),
            target_corner_x=0,
            target_shape=mask.shape,
            frame_id=1,
            panel_index=0,
            minimum_correspondences=80,
        )


def test_complete_object_owner_directly_projects_real_depth_cells() -> None:
    height, width = 50, 70
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[12:38, 20:50] = (21, 107, 203)
    depth = np.full((height, width), 500.0, dtype=np.float32)
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[12:38, 20:50] = True
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=50.0,
        fy=50.0,
        cx=35.0,
        cy=25.0,
        distortion=(),
    )
    layout = InspectionMultiviewLayout(
        width=width,
        height=height,
        reference_depth_mm=500.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=(
            VirtualPerspectivePanel(
                panel_index=0,
                anchor_scan_mm=0.0,
                canvas_offset_x=0.0,
                center_world_mm=(0.0, 0.0, 0.0),
            ),
        ),
        panel_step_mm=500.0,
        canvas_megapixels=width * height / 1_000_000.0,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = 20.0
    result = project_complete_object_owner_from_rgbd(
        source_image_bgr=image,
        source_depth_mm=depth,
        source_reliable_depth=np.ones(depth.shape, dtype=bool),
        source_object_mask=object_mask,
        camera_to_world=pose,
        layout=layout,
        intrinsics=intrinsics,
        frame_id=3,
        panel_index=0,
        minimum_cells=64,
    )
    target_y, target_x = np.nonzero(result.target_mask)
    assert int(np.min(target_x)) >= 22
    assert int(np.max(target_x)) <= 52
    assert np.all(
        result.target_image_bgr[result.target_mask] == (21, 107, 203)
    )
    assert result.audit["direct_world_projection"] is True
    assert result.audit["fitted_display_warp"] is False


def _automatic_handoff_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[ObjectHandoffSource, ...],
    InspectionMultiviewLayout,
    CameraIntrinsics,
]:
    height, width = 72, 96
    component = np.zeros((height, width), dtype=bool)
    component[20:52, 32:64] = True
    baseline_owner = np.full((height, width), -1, dtype=np.int32)
    baseline_owner[component & (np.indices(component.shape)[1] < 48)] = 3
    baseline_owner[component & (np.indices(component.shape)[1] >= 48)] = 7
    yy, xx = np.indices(component.shape, dtype=np.float32)
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=70.0,
        fy=70.0,
        cx=48.0,
        cy=36.0,
        distortion=(),
    )
    layout = InspectionMultiviewLayout(
        width=width,
        height=height,
        reference_depth_mm=500.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=(
            VirtualPerspectivePanel(
                panel_index=0,
                anchor_scan_mm=0.0,
                canvas_offset_x=0.0,
                center_world_mm=(0.0, 0.0, 0.0),
            ),
        ),
        panel_step_mm=500.0,
        canvas_megapixels=width * height / 1_000_000.0,
    )
    sources: list[ObjectHandoffSource] = []
    for frame_id, colour in ((3, (31, 101, 211)), (7, (33, 103, 209))):
        image = np.full((height, width, 3), 8, dtype=np.uint8)
        image[component] = colour
        depth = np.full((height, width), 900.0, dtype=np.float32)
        depth[component] = 500.0
        sources.append(
            ObjectHandoffSource(
                frame_id=frame_id,
                panel_index=0,
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=np.ones(component.shape, dtype=bool),
                camera_to_world=np.eye(4, dtype=np.float64),
                mesh_corner_x=0,
                mesh_map_x=xx.copy(),
                mesh_map_y=yy.copy(),
                mesh_valid_mask=component.copy(),
                mesh_relative_depth_mm=np.where(
                    component, 500.0, np.nan
                ).astype(np.float32),
            )
        )
    return component, baseline_owner, tuple(sources), layout, intrinsics


def test_automatic_object_handoff_requires_two_consistent_real_views() -> None:
    component, baseline_owner, sources, layout, intrinsics = (
        _automatic_handoff_fixture()
    )
    result = select_automatic_complete_object_owner(
        target_component_mask=component,
        baseline_owner_frame_id=baseline_owner,
        sources=sources,
        target_panel_index=0,
        layout=layout,
        intrinsics=intrinsics,
        minimum_seed_pixels=32,
    )
    assert result.owner.frame_id in {3, 7}
    assert result.audit["accepted_source_count"] == 2
    assert result.audit["manual_bbox_used"] is False
    assert result.audit["manual_frame_id_used"] is False
    assert result.audit["silent_fallback_allowed"] is False
    assert (
        result.audit[
            "selected_cross_view_footprint_union_coverage_ratio"
        ]
        == pytest.approx(1.0)
    )
    assert np.count_nonzero(result.owner.target_mask & component) > 900


def test_automatic_object_handoff_rejects_single_owner_component() -> None:
    component, baseline_owner, sources, layout, intrinsics = (
        _automatic_handoff_fixture()
    )
    baseline_owner[component] = 3
    with pytest.raises(RuntimeError, match="baseline multi-owner"):
        select_automatic_complete_object_owner(
            target_component_mask=component,
            baseline_owner_frame_id=baseline_owner,
            sources=sources,
            target_panel_index=0,
            layout=layout,
            intrinsics=intrinsics,
            minimum_seed_pixels=32,
        )
