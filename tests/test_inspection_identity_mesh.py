from __future__ import annotations

import numpy as np

from panorama_demo.inspection_identity_mesh import (
    InspectionIdentityMeshConfig,
    InspectionIdentityMeshSource,
    composite_inspection_identity_owners,
)
from panorama_demo.inspection_multiview import (
    InspectionForegroundIdentityOwner,
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
)
from panorama_demo.session import CameraIntrinsics


def _fixture():
    height, width = 80, 128
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=100.0,
        fy=100.0,
        cx=64.0,
        cy=40.0,
        distortion=(),
    )
    layout = InspectionMultiviewLayout(
        width=width,
        height=height,
        reference_depth_mm=1000.0,
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
        panel_step_mm=100.0,
        canvas_megapixels=height * width / 1_000_000.0,
    )
    image = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.indices((height, width))
    image[..., 0] = xx
    image[..., 1] = yy
    image[..., 2] = 200
    source_mask = np.zeros((height, width), dtype=bool)
    source_mask[20:53, 40:85] = True
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    depth[source_mask] = 600.0
    target = source_mask.copy()
    owner = InspectionForegroundIdentityOwner(
        group_id=7,
        structure_id=0,
        structure_kind="synthetic_object",
        identity_track_id=3,
        panel_index=0,
        frame_id=22,
        source_index=0,
        source_mask=source_mask,
        target_footprint=target,
        measured_depth_coverage_ratio=1.0,
        projected_in_bounds_ratio=1.0,
    )
    source = InspectionIdentityMeshSource(
        panel_index=0,
        frame_id=22,
        image_bgr=image,
        depth_mm=depth,
        reliable_depth=np.ones((height, width), dtype=bool),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    return intrinsics, layout, image, owner, source


def test_identity_owner_uses_true_rgbd_inverse_mesh_and_one_rgb_source():
    intrinsics, layout, image, owner, source = _fixture()
    shape = (layout.height, layout.width)
    output_image = np.full((*shape, 3), 17, dtype=np.uint8)
    output_depth = np.full(shape, 1000.0, dtype=np.float32)
    output_confidence = np.full(shape, 0.1, dtype=np.float32)
    output_owner = np.full(shape, 11, dtype=np.int32)
    reliable = np.zeros(shape, dtype=bool)
    overlay = np.zeros(shape, dtype=bool)

    audit = composite_inspection_identity_owners(
        owners=(owner,),
        sources_by_frame_id={22: source},
        layout=layout,
        intrinsics=intrinsics,
        output_image=output_image,
        output_depth=output_depth,
        output_confidence=output_confidence,
        output_owner=output_owner,
        output_reliable_depth=reliable,
        output_overlay_mask=overlay,
        config=InspectionIdentityMeshConfig(
            cell_size_pixels=4,
            maximum_fill_distance_pixels=2.0,
        ),
    )

    assert audit["component_count"] == 1
    assert audit["rgb_generated"] is False
    assert audit["rgb_alpha_blended"] is False
    assert audit["pose_modified"] is False
    assert np.all(output_owner[owner.target_footprint] == 22)
    assert np.all(reliable[owner.target_footprint])
    assert np.all(overlay[owner.target_footprint])
    assert np.allclose(output_depth[owner.target_footprint], 600.0)
    assert np.array_equal(
        output_image[owner.target_footprint],
        image[owner.source_mask],
    )
    assert np.all(output_owner[~owner.target_footprint] == 11)


def test_identity_owner_fails_closed_when_depth_hole_is_too_large():
    intrinsics, layout, _, owner, source = _fixture()
    reliable = source.reliable_depth.copy()
    reliable[24:49, 48:77] = False
    source = InspectionIdentityMeshSource(
        panel_index=source.panel_index,
        frame_id=source.frame_id,
        image_bgr=source.image_bgr,
        depth_mm=source.depth_mm,
        reliable_depth=reliable,
        camera_to_world=source.camera_to_world,
    )
    shape = (layout.height, layout.width)
    try:
        composite_inspection_identity_owners(
            owners=(owner,),
            sources_by_frame_id={22: source},
            layout=layout,
            intrinsics=intrinsics,
            output_image=np.zeros((*shape, 3), dtype=np.uint8),
            output_depth=np.full(shape, 1000.0, dtype=np.float32),
            output_confidence=np.zeros(shape, dtype=np.float32),
            output_owner=np.zeros(shape, dtype=np.int32),
            output_reliable_depth=np.zeros(shape, dtype=bool),
            output_overlay_mask=np.zeros(shape, dtype=bool),
            config=InspectionIdentityMeshConfig(
                cell_size_pixels=4,
                maximum_fill_distance_pixels=2.0,
            ),
        )
    except RuntimeError as error:
        assert (
            "direct inverse-mesh support ratio" in str(error)
            or "fill fraction exceeds" in str(error)
            or "depth hole exceeds" in str(error)
        )
    else:
        raise AssertionError("An oversized identity depth hole was accepted")


def test_cross_panel_identity_owner_uses_target_scene_z_buffer() -> None:
    intrinsics, base_layout, image, base_owner, source = _fixture()
    width = intrinsics.width
    layout = InspectionMultiviewLayout(
        width=2 * width,
        height=base_layout.height,
        reference_depth_mm=base_layout.reference_depth_mm,
        scan_axis=base_layout.scan_axis,
        down_axis=base_layout.down_axis,
        normal_axis=base_layout.normal_axis,
        panels=(
            base_layout.panels[0],
            VirtualPerspectivePanel(
                panel_index=1,
                anchor_scan_mm=0.0,
                canvas_offset_x=float(width),
                center_world_mm=(0.0, 0.0, 0.0),
            ),
        ),
        panel_step_mm=100.0,
        canvas_megapixels=2
        * width
        * base_layout.height
        / 1_000_000.0,
    )
    target_footprint = np.zeros(
        (layout.height, layout.width), dtype=bool
    )
    target_footprint[:, width:] = base_owner.target_footprint
    owner = InspectionForegroundIdentityOwner(
        group_id=base_owner.group_id,
        structure_id=base_owner.structure_id,
        structure_kind=base_owner.structure_kind,
        identity_track_id=base_owner.identity_track_id,
        panel_index=0,
        target_panel_index=1,
        frame_id=base_owner.frame_id,
        source_index=base_owner.source_index,
        source_mask=base_owner.source_mask,
        target_footprint=target_footprint,
        measured_depth_coverage_ratio=1.0,
        projected_in_bounds_ratio=1.0,
    )
    target_source = InspectionIdentityMeshSource(
        panel_index=1,
        frame_id=23,
        image_bgr=np.zeros_like(image),
        depth_mm=np.full(image.shape[:2], 1000.0, dtype=np.float32),
        reliable_depth=np.ones(image.shape[:2], dtype=bool),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    shape = (layout.height, layout.width)
    output_owner = np.full(shape, 11, dtype=np.int32)

    audit = composite_inspection_identity_owners(
        owners=(owner,),
        sources_by_frame_id={22: source, 23: target_source},
        layout=layout,
        intrinsics=intrinsics,
        output_image=np.zeros((*shape, 3), dtype=np.uint8),
        output_depth=np.full(shape, 1000.0, dtype=np.float32),
        output_confidence=np.zeros(shape, dtype=np.float32),
        output_owner=output_owner,
        output_reliable_depth=np.zeros(shape, dtype=bool),
        output_overlay_mask=np.zeros(shape, dtype=bool),
        config=InspectionIdentityMeshConfig(),
    )

    component = audit["components"][0]
    assert component["target_scene_z_buffer_required"] is True
    assert component["target_scene_missing_pixel_count"] == 0
    assert component["target_scene_occluded_pixel_count"] == 0
    assert np.all(output_owner[target_footprint] == 22)


def test_cross_panel_identity_owner_rejects_nearer_target_occluder() -> None:
    intrinsics, base_layout, image, base_owner, source = _fixture()
    width = intrinsics.width
    layout = InspectionMultiviewLayout(
        width=2 * width,
        height=base_layout.height,
        reference_depth_mm=base_layout.reference_depth_mm,
        scan_axis=base_layout.scan_axis,
        down_axis=base_layout.down_axis,
        normal_axis=base_layout.normal_axis,
        panels=(
            base_layout.panels[0],
            VirtualPerspectivePanel(
                panel_index=1,
                anchor_scan_mm=0.0,
                canvas_offset_x=float(width),
                center_world_mm=(0.0, 0.0, 0.0),
            ),
        ),
        panel_step_mm=100.0,
        canvas_megapixels=2
        * width
        * base_layout.height
        / 1_000_000.0,
    )
    target_footprint = np.zeros(
        (layout.height, layout.width), dtype=bool
    )
    target_footprint[:, width:] = base_owner.target_footprint
    owner = InspectionForegroundIdentityOwner(
        group_id=base_owner.group_id,
        structure_id=base_owner.structure_id,
        structure_kind=base_owner.structure_kind,
        identity_track_id=base_owner.identity_track_id,
        panel_index=0,
        target_panel_index=1,
        frame_id=base_owner.frame_id,
        source_index=base_owner.source_index,
        source_mask=base_owner.source_mask,
        target_footprint=target_footprint,
        measured_depth_coverage_ratio=1.0,
        projected_in_bounds_ratio=1.0,
    )
    # A nearer complete target view must conservatively occlude the object.
    # Using a complete plane keeps this test about depth ordering rather than
    # sparse projected support at the artificial depth discontinuity.
    target_depth = np.full(image.shape[:2], 400.0, dtype=np.float32)
    target_source = InspectionIdentityMeshSource(
        panel_index=1,
        frame_id=23,
        image_bgr=np.zeros_like(image),
        depth_mm=target_depth,
        reliable_depth=np.ones(image.shape[:2], dtype=bool),
        camera_to_world=np.eye(4, dtype=np.float64),
    )
    shape = (layout.height, layout.width)

    try:
        composite_inspection_identity_owners(
            owners=(owner,),
            sources_by_frame_id={22: source, 23: target_source},
            layout=layout,
            intrinsics=intrinsics,
            output_image=np.zeros((*shape, 3), dtype=np.uint8),
            output_depth=np.full(shape, 1000.0, dtype=np.float32),
            output_confidence=np.zeros(shape, dtype=np.float32),
            output_owner=np.zeros(shape, dtype=np.int32),
            output_reliable_depth=np.zeros(shape, dtype=bool),
            output_overlay_mask=np.zeros(shape, dtype=bool),
            config=InspectionIdentityMeshConfig(),
        )
    except RuntimeError as error:
        assert "fully occluded" in str(error)
    else:
        raise AssertionError("A nearer target RGB-D occluder was ignored")
