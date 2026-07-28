from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.inspection_multiview import (
    _apply_continuous_canvas_exposure_curve,
    _background_owner_boundary_audit,
    _build_depth_mesh_panel_remap,
    _composite_reference_panel,
    _composite_locked_foreground_mesh_rgb,
    _DepthMeshPanelRemap,
    _enforce_foreground_components_single_owner,
    _owner_topology_audit,
    _prepare_pre_seam_hard_owner_intervals,
    _ReferencePanelRaster,
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    InspectionPreSeamHardOwnerInterval,
    VirtualPerspectivePanel,
    estimate_inspection_layout,
    estimate_inspection_working_set,
    project_world_points_to_panels,
    render_inspection_multiview,
)
from panorama_demo.session import CameraIntrinsics, RGBDFrame


def _intrinsics(width: int = 80, height: int = 60) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=70.0,
        fy=70.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
        distortion=(),
    )


def _layout() -> InspectionMultiviewLayout:
    return InspectionMultiviewLayout(
        width=150,
        height=60,
        reference_depth_mm=1000.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=(
            VirtualPerspectivePanel(0, 0.0, 0.0, (0.0, 0.0, 0.0)),
            VirtualPerspectivePanel(1, 1000.0, 70.0, (1000.0, 0.0, 0.0)),
        ),
        panel_step_mm=1000.0,
        canvas_megapixels=0.009,
    )


def _pre_seam_test_rasters() -> tuple[
    tuple[_ReferencePanelRaster, ...], tuple[int, int]
]:
    shape = (20, 80)
    rasters = (
        _ReferencePanelRaster(
            panel_index=0,
            frame_id=10,
            corner_x=0,
            image_bgr=np.zeros((shape[0], 60, 3), dtype=np.uint8),
            valid_mask=np.ones((shape[0], 60), dtype=bool),
            protected_mask=np.zeros((shape[0], 60), dtype=bool),
            confidence=np.ones((shape[0], 60), dtype=np.float32),
        ),
        _ReferencePanelRaster(
            panel_index=1,
            frame_id=11,
            corner_x=20,
            image_bgr=np.zeros((shape[0], 60, 3), dtype=np.uint8),
            valid_mask=np.ones((shape[0], 60), dtype=bool),
            protected_mask=np.zeros((shape[0], 60), dtype=bool),
            confidence=np.ones((shape[0], 60), dtype=np.float32),
        ),
    )
    return rasters, shape


def _pre_seam_interval(
    *,
    track_id: int,
    panel_index: int,
    frame_id: int,
    shape: tuple[int, int],
    x0: int,
    x1: int,
) -> InspectionPreSeamHardOwnerInterval:
    lock = np.zeros(shape, dtype=bool)
    lock[5:15, x0:x1] = True
    footprint = np.zeros(shape, dtype=bool)
    footprint[7:13, x0 + 1 : x1 - 1] = True
    return InspectionPreSeamHardOwnerInterval(
        track_id=track_id,
        panel_index=panel_index,
        frame_id=frame_id,
        lock_mask=lock,
        union_footprint=footprint,
    )


def test_pre_seam_hard_owner_intervals_validate_and_merge() -> None:
    rasters, shape = _pre_seam_test_rasters()
    interval = _pre_seam_interval(
        track_id=4,
        panel_index=0,
        frame_id=10,
        shape=shape,
        x0=30,
        x1=46,
    )

    locked, guard, audit = _prepare_pre_seam_hard_owner_intervals(
        (interval,), rasters, shape
    )

    assert np.all(locked[interval.lock_mask] == 0)
    assert np.array_equal(guard, interval.lock_mask)
    assert audit["used"] is True
    assert audit["interval_count"] == 1
    assert audit["different_panel_conflict_pixel_count"] == 0
    assert audit["intervals"][0]["row_contiguous"] is True


def test_pre_seam_spatial_lock_is_separate_from_rgb_and_blend_masks() -> None:
    rasters, shape = _pre_seam_test_rasters()
    base = _pre_seam_interval(
        track_id=5,
        panel_index=0,
        frame_id=10,
        shape=shape,
        x0=30,
        x1=46,
    )
    transfer = np.zeros(shape, dtype=bool)
    transfer[8:12, 35:40] = True
    owner_only = np.zeros(shape, dtype=bool)
    owner_only[7:13, 34:41] = True
    interval = InspectionPreSeamHardOwnerInterval(
        track_id=base.track_id,
        panel_index=base.panel_index,
        frame_id=base.frame_id,
        lock_mask=base.lock_mask,
        union_footprint=base.union_footprint,
        rgb_transfer_mask=transfer,
        owner_only_mask=owner_only,
    )

    locked, guard, audit = _prepare_pre_seam_hard_owner_intervals(
        (interval,), rasters, shape
    )

    assert np.all(locked[base.lock_mask] == 0)
    assert np.array_equal(guard, owner_only)
    assert audit["locked_pixel_count"] == int(
        np.count_nonzero(base.lock_mask)
    )
    assert audit["owner_only_pixel_count"] == int(
        np.count_nonzero(owner_only)
    )
    assert audit["intervals"][0]["rgb_transfer_pixel_count"] == int(
        np.count_nonzero(transfer)
    )


def test_deferred_true_depth_owner_guards_without_locking_background_panel() -> None:
    rasters, shape = _pre_seam_test_rasters()
    base = _pre_seam_interval(
        track_id=6,
        panel_index=0,
        frame_id=99,
        shape=shape,
        x0=30,
        x1=46,
    )
    interval = InspectionPreSeamHardOwnerInterval(
        track_id=base.track_id,
        panel_index=base.panel_index,
        frame_id=base.frame_id,
        lock_mask=base.lock_mask,
        union_footprint=base.union_footprint,
        rgb_source_panel_index=1,
        rgb_transfer_mask=base.union_footprint,
        owner_only_mask=base.lock_mask,
        deferred_true_depth_identity_overlay=True,
    )

    locked, guard, audit = _prepare_pre_seam_hard_owner_intervals(
        (interval,), rasters, shape
    )

    assert np.all(locked == -1)
    assert np.array_equal(guard, base.lock_mask)
    assert audit["locked_pixel_count"] == 0
    row = audit["intervals"][0]
    assert row["background_spatial_panel_lock_applied"] is False
    assert row["background_panel_owner_decoupled_from_rgb_owner"] is True


def test_panel_native_object_guards_can_overlap_across_background_panels() -> None:
    rasters, shape = _pre_seam_test_rasters()
    first_base = _pre_seam_interval(
        track_id=7,
        panel_index=0,
        frame_id=10,
        shape=shape,
        x0=30,
        x1=46,
    )
    second_base = _pre_seam_interval(
        track_id=8,
        panel_index=1,
        frame_id=11,
        shape=shape,
        x0=40,
        x1=54,
    )
    first_transfer = np.zeros(shape, dtype=bool)
    first_transfer[7:13, 32:39] = True
    second_transfer = np.zeros(shape, dtype=bool)
    second_transfer[7:13, 47:52] = True
    first = InspectionPreSeamHardOwnerInterval(
        track_id=first_base.track_id,
        panel_index=first_base.panel_index,
        frame_id=first_base.frame_id,
        lock_mask=first_base.lock_mask,
        union_footprint=first_transfer,
        rgb_transfer_mask=first_transfer,
        owner_only_mask=first_base.lock_mask,
        background_panel_lock_required=False,
    )
    second = InspectionPreSeamHardOwnerInterval(
        track_id=second_base.track_id,
        panel_index=second_base.panel_index,
        frame_id=second_base.frame_id,
        lock_mask=second_base.lock_mask,
        union_footprint=second_transfer,
        rgb_transfer_mask=second_transfer,
        owner_only_mask=second_base.lock_mask,
        background_panel_lock_required=False,
    )

    locked, guard, audit = _prepare_pre_seam_hard_owner_intervals(
        (first, second), rasters, shape
    )

    assert np.any(first.lock_mask & second.lock_mask)
    assert not np.any(first_transfer & second_transfer)
    assert np.all(locked == -1)
    assert np.array_equal(guard, first.lock_mask | second.lock_mask)
    assert audit["locked_pixel_count"] == 0
    assert audit["different_panel_conflict_pixel_count"] == 0
    assert audit["different_rgb_owner_transfer_conflict_pixel_count"] == 0
    assert all(
        row["background_spatial_panel_lock_applied"] is False
        for row in audit["intervals"]
    )
    assert all(
        row["background_panel_owner_decoupled_from_rgb_owner"] is True
        for row in audit["intervals"]
    )


def test_panel_native_objects_reject_true_rgb_footprint_owner_overlap() -> None:
    rasters, shape = _pre_seam_test_rasters()
    first_base = _pre_seam_interval(
        track_id=9,
        panel_index=0,
        frame_id=10,
        shape=shape,
        x0=30,
        x1=46,
    )
    second_base = _pre_seam_interval(
        track_id=10,
        panel_index=1,
        frame_id=11,
        shape=shape,
        x0=38,
        x1=52,
    )
    first = InspectionPreSeamHardOwnerInterval(
        track_id=first_base.track_id,
        panel_index=first_base.panel_index,
        frame_id=first_base.frame_id,
        lock_mask=first_base.lock_mask,
        union_footprint=first_base.union_footprint,
        rgb_transfer_mask=first_base.union_footprint,
        owner_only_mask=first_base.lock_mask,
        background_panel_lock_required=False,
    )
    second = InspectionPreSeamHardOwnerInterval(
        track_id=second_base.track_id,
        panel_index=second_base.panel_index,
        frame_id=second_base.frame_id,
        lock_mask=second_base.lock_mask,
        union_footprint=second_base.union_footprint,
        rgb_transfer_mask=second_base.union_footprint,
        owner_only_mask=second_base.lock_mask,
        background_panel_lock_required=False,
    )

    with pytest.raises(RuntimeError, match="different real RGB owners"):
        _prepare_pre_seam_hard_owner_intervals(
            (first, second), rasters, shape
        )


def test_pre_seam_hard_owner_intervals_reject_conflicting_panels() -> None:
    rasters, shape = _pre_seam_test_rasters()
    first = _pre_seam_interval(
        track_id=1,
        panel_index=0,
        frame_id=10,
        shape=shape,
        x0=30,
        x1=46,
    )
    second = _pre_seam_interval(
        track_id=2,
        panel_index=1,
        frame_id=11,
        shape=shape,
        x0=38,
        x1=52,
    )

    with pytest.raises(RuntimeError, match="different panels"):
        _prepare_pre_seam_hard_owner_intervals(
            (first, second), rasters, shape
        )


def test_pre_seam_hard_owner_interval_rejects_frame_mismatch() -> None:
    rasters, shape = _pre_seam_test_rasters()
    interval = _pre_seam_interval(
        track_id=3,
        panel_index=0,
        frame_id=11,
        shape=shape,
        x0=30,
        x1=46,
    )

    with pytest.raises(RuntimeError, match="panel/frame"):
        _prepare_pre_seam_hard_owner_intervals(
            (interval,), rasters, shape
        )


def test_pre_seam_hard_owner_interval_rejects_invalid_panel_coverage() -> None:
    rasters, shape = _pre_seam_test_rasters()
    interval = _pre_seam_interval(
        track_id=5,
        panel_index=0,
        frame_id=10,
        shape=shape,
        x0=56,
        x1=64,
    )

    with pytest.raises(RuntimeError, match="complete valid coverage"):
        _prepare_pre_seam_hard_owner_intervals(
            (interval,), rasters, shape
        )


def _twenty_metre_160_panel_layout() -> InspectionMultiviewLayout:
    panel_count = 160
    scan_span_mm = 20_000.0
    canvas_width = 18_080
    canvas_height = 800
    panel_step_mm = scan_span_mm / (panel_count - 1)
    canvas_step = (canvas_width - 1280) / (panel_count - 1)
    return InspectionMultiviewLayout(
        width=canvas_width,
        height=canvas_height,
        reference_depth_mm=730.4,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=index,
                anchor_scan_mm=panel_step_mm * index,
                canvas_offset_x=canvas_step * index,
                center_world_mm=(panel_step_mm * index, 0.0, 0.0),
            )
            for index in range(panel_count)
        ),
        panel_step_mm=panel_step_mm,
        canvas_megapixels=(
            canvas_width * canvas_height / 1_000_000.0
        ),
    )


def test_reference_panel_lazy_maps_preserve_all_composite_pixels() -> None:
    intrinsics = _intrinsics()
    layout = _layout()
    yy, xx = np.indices((intrinsics.height, intrinsics.width))
    source_image = np.stack(
        (
            (xx * 3) % 256,
            (yy * 5) % 256,
            (xx + yy * 2) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    protected = np.zeros(source_image.shape[:2], dtype=bool)
    protected[14:27, 31:46] = True

    def composite(
        *, retain_reference_maps: bool
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        _ReferencePanelRaster,
    ]:
        output_image = np.zeros(
            (layout.height, layout.width, 3), dtype=np.uint8
        )
        output_depth = np.full(
            (layout.height, layout.width), np.inf, dtype=np.float32
        )
        output_confidence = np.zeros(
            (layout.height, layout.width), dtype=np.float32
        )
        output_owner = np.full(
            (layout.height, layout.width), -1, dtype=np.int32
        )
        output_reliable = np.zeros(
            (layout.height, layout.width), dtype=bool
        )
        _, raster = _composite_reference_panel(
            output_image=output_image,
            output_depth=output_depth,
            output_confidence=output_confidence,
            output_owner=output_owner,
            output_reliable_depth=output_reliable,
            source_image=source_image,
            source_protected_mask=protected,
            source_pose=np.eye(4),
            frame_id=7,
            panel_index=0,
            layout=layout,
            intrinsics=intrinsics,
            retain_reference_maps=retain_reference_maps,
        )
        return (
            output_image,
            output_depth,
            output_confidence,
            output_owner,
            output_reliable,
            raster,
        )

    eager = composite(retain_reference_maps=True)
    lazy = composite(retain_reference_maps=False)
    for eager_value, lazy_value in zip(
        eager[:5], lazy[:5], strict=True
    ):
        assert np.array_equal(eager_value, lazy_value)
    for attribute in (
        "image_bgr",
        "valid_mask",
        "protected_mask",
        "confidence",
    ):
        assert np.array_equal(
            getattr(eager[5], attribute),
            getattr(lazy[5], attribute),
        )
    assert eager[5].reference_map_x is not None
    assert eager[5].reference_map_y is not None
    assert lazy[5].reference_map_x is None
    assert lazy[5].reference_map_y is None


def test_resource_estimator_accepts_corridor_local_20m_160_panel_plan() -> None:
    estimate = estimate_inspection_working_set(
        _twenty_metre_160_panel_layout(),
        _intrinsics(width=1280, height=800),
        config=InspectionMultiviewConfig(
            maximum_working_bytes=4_000_000_000
        ),
    )

    audit = estimate.as_dict()
    assert estimate.model == "corridor_local_adjacent_pair_streaming/v1"
    assert estimate.panel_count == 160
    assert estimate.canvas_pixel_count == 18_080 * 800
    assert estimate.estimated_peak_bytes < 4_000_000_000
    assert audit["estimate_role"] == (
        "required_corridor_local_target_contract"
    )
    assert audit["runtime_peak_measured"] is False
    assert audit["maximum_resident_adjacent_panels"] == 2
    assert audit["maximum_resident_pair_corridors"] == 1
    assert audit["per_panel_full_canvas_array_count"] == 0
    assert audit["per_pair_full_canvas_array_count"] == 0
    assert audit["panel_canvas_product_bytes"] == 0
    assert audit["pair_canvas_product_bytes"] == 0
    assert audit["within_budget"] is True


@pytest.mark.parametrize(
    ("per_panel_count", "per_pair_count"),
    ((1, 0), (0, 1)),
)
def test_resource_estimator_rejects_scaled_full_canvas_residency(
    per_panel_count: int,
    per_pair_count: int,
) -> None:
    with pytest.raises(MemoryError, match="full-canvas resident arrays"):
        estimate_inspection_working_set(
            _twenty_metre_160_panel_layout(),
            _intrinsics(width=1280, height=800),
            per_panel_full_canvas_array_count=per_panel_count,
            per_pair_full_canvas_array_count=per_pair_count,
        )


def test_resource_estimator_rejects_corridor_local_plan_over_budget() -> None:
    with pytest.raises(MemoryError, match="exceeds its byte budget"):
        estimate_inspection_working_set(
            _twenty_metre_160_panel_layout(),
            _intrinsics(width=1280, height=800),
            config=InspectionMultiviewConfig(
                maximum_working_bytes=100_000_000
            ),
        )


def test_continuous_photometric_curve_applies_same_gain_to_owner_guard() -> None:
    height, width = 48, 160
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = (92, 100, 108)
    image[:, width // 2 :] = (122, 111, 101)
    safe_background = np.ones((height, width), dtype=bool)
    safe_background[12:36, 72:88] = False
    valid = np.ones((height, width), dtype=bool)
    before_guard = image[~safe_background].copy()
    before_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    before_delta = float(np.mean(np.linalg.norm(
        before_lab[:, width // 2 - 1] - before_lab[:, width // 2],
        axis=1,
    )))

    corrected, audit = _apply_continuous_canvas_exposure_curve(
        image, safe_background, valid
    )

    after_lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB).astype(np.float32)
    after_delta = float(np.mean(np.linalg.norm(
        after_lab[:, width // 2 - 1] - after_lab[:, width // 2],
        axis=1,
    )))
    assert audit["applied"] is True
    assert audit["method"].startswith("neutral_safe_background_estimated")
    assert not np.array_equal(corrected[~safe_background], before_guard)
    assert audit["corrected_pixel_count"] == int(np.count_nonzero(valid))
    assert audit["reference_rgb_used"] is False
    assert audit["column_varying_gain_used"] is False
    assert audit["maximum_adjacent_column_gain_delta"] == 0.0
    assert audit["minimum_gain"] == pytest.approx(audit["maximum_gain"])
    assert after_delta >= before_delta


def test_owner_boundary_audit_excludes_owner_only_guard() -> None:
    height, width = 32, 80
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :40] = (70, 70, 70)
    image[:, 40:] = (180, 180, 180)
    owner = np.empty((height, width), dtype=np.int32)
    owner[:, :40] = 10
    owner[:, 40:] = 20
    valid = np.ones((height, width), dtype=bool)
    foreground = np.zeros_like(valid)
    config = InspectionMultiviewConfig(
        maximum_background_owner_boundary_lab_p95=30.0
    )

    unguarded = _background_owner_boundary_audit(
        image, owner, valid, foreground, config
    )
    owner_only_guard = np.zeros_like(valid)
    owner_only_guard[:, 39:41] = True
    guarded = _background_owner_boundary_audit(
        image,
        owner,
        valid,
        foreground,
        config,
        owner_only_guard_mask=owner_only_guard,
    )

    assert unguarded["pass"] is False
    assert unguarded["pairs"][0]["left_frame_id"] == 10
    assert unguarded["pairs"][0]["right_frame_id"] == 20
    assert guarded["pass"] is True
    assert guarded["sample_count"] == 0
    assert guarded["owner_only_guard_pixel_count"] == 64


def test_reference_plane_is_panel_invariant_and_near_surface_keeps_parallax() -> None:
    intrinsics = _intrinsics()
    layout = _layout()
    background = np.asarray([[500.0, 0.0, 1000.0]])
    x, _, _, _ = project_world_points_to_panels(
        background, layout, intrinsics
    )
    expected = intrinsics.cx + intrinsics.fx * 0.5
    assert x[0] == pytest.approx(expected)

    # Evaluate the same point explicitly in both panel equations.  It aligns
    # on D0, whereas a near point retains a nonzero disparity.
    def panel_x(point: np.ndarray, panel: VirtualPerspectivePanel) -> float:
        relative = point - np.asarray(panel.center_world_mm)
        return (
            panel.canvas_offset_x
            + intrinsics.cx
            + intrinsics.fx * relative[0] / relative[2]
        )

    assert panel_x(background[0], layout.panels[0]) == pytest.approx(
        panel_x(background[0], layout.panels[1])
    )
    near = np.asarray([500.0, 0.0, 500.0])
    assert abs(
        panel_x(near, layout.panels[0])
        - panel_x(near, layout.panels[1])
    ) > 1.0


def test_inverse_depth_mesh_is_continuous_and_has_no_point_splat_holes() -> None:
    intrinsics = _intrinsics()
    layout = _layout()
    depth = np.full(
        (intrinsics.height, intrinsics.width), 500.0, dtype=np.float32
    )
    config = InspectionMultiviewConfig(
        minimum_depth_mm=200.0,
        maximum_depth_mm=1500.0,
        depth_mesh_cell_size_pixels=8,
        depth_mesh_boundary_margin_pixels=1,
    )
    mesh = _build_depth_mesh_panel_remap(
        source_depth_mm=depth,
        source_solver_valid=np.ones(depth.shape, dtype=bool),
        source_pose=np.eye(4),
        panel_index=0,
        layout=layout,
        intrinsics=intrinsics,
        config=config,
    )

    # Every boundary-safe target pixel is covered by a continuous inverse
    # map.  A rounded forward point splat under a non-integral transform would
    # leave interior holes; the local mesh does not.
    assert np.all(mesh.valid_mask[1:-1, 1:-1])
    assert mesh.audit["accepted_cell_count"] > 0
    assert mesh.audit["accepted_triangle_count"] == (
        mesh.audit["accepted_cell_count"] * 2
    )
    assert mesh.audit["minimum_accepted_jacobian"] > 0.0
    assert mesh.audit["rgb_generated"] is False
    assert mesh.audit["pose_modified"] is False
    target_y, target_x = np.indices(mesh.valid_mask.shape, dtype=np.float32)
    assert np.max(
        np.abs(mesh.map_x[mesh.valid_mask] - target_x[mesh.valid_mask])
    ) < 1e-4
    assert np.max(
        np.abs(mesh.map_y[mesh.valid_mask] - target_y[mesh.valid_mask])
    ) < 1e-4
    assert np.all(np.isfinite(mesh.relative_depth_mm[mesh.valid_mask]))
    assert np.max(
        np.abs(mesh.relative_depth_mm[mesh.valid_mask] - 500.0)
    ) < 1e-4


def test_inverse_depth_mesh_preserves_near_parallax_and_rejects_boundaries() -> None:
    intrinsics = _intrinsics()
    layout = _layout()
    depth = np.full(
        (intrinsics.height, intrinsics.width), 500.0, dtype=np.float32
    )
    solver_valid = np.ones(depth.shape, dtype=bool)
    # A protected depth boundary invalidates every cell touching it instead
    # of allowing an inverse triangle to bridge two surfaces.
    solver_valid[:, 38:42] = False
    config = InspectionMultiviewConfig(
        minimum_depth_mm=200.0,
        maximum_depth_mm=1500.0,
        depth_mesh_cell_size_pixels=4,
        depth_mesh_boundary_margin_pixels=1,
    )
    pose_a = np.eye(4)
    pose_b = np.eye(4)
    pose_b[0, 3] = 100.0
    mesh_a = _build_depth_mesh_panel_remap(
        source_depth_mm=depth,
        source_solver_valid=solver_valid,
        source_pose=pose_a,
        panel_index=0,
        layout=layout,
        intrinsics=intrinsics,
        config=config,
    )
    mesh_b = _build_depth_mesh_panel_remap(
        source_depth_mm=depth,
        source_solver_valid=solver_valid,
        source_pose=pose_b,
        panel_index=0,
        layout=layout,
        intrinsics=intrinsics,
        config=config,
    )

    common = mesh_a.valid_mask & mesh_b.valid_mask
    assert np.any(common)
    # The same target pixel samples a source coordinate shifted by the real
    # 100 mm camera baseline: fx * baseline / depth = 14 pixels.
    disparity = mesh_a.map_x[common] - mesh_b.map_x[common]
    assert float(np.median(disparity)) == pytest.approx(14.0, abs=1e-4)
    assert mesh_a.audit["rejected_invalid_or_boundary_cell_count"] > 0
    assert np.all(np.isfinite(mesh_a.map_x[mesh_a.valid_mask]))
    assert np.all(np.isfinite(mesh_a.map_y[mesh_a.valid_mask]))
    assert np.all(np.isnan(mesh_a.map_x[~mesh_a.valid_mask]))
    assert np.all(np.isnan(mesh_a.map_y[~mesh_a.valid_mask]))


def test_rendered_foreground_component_gets_one_fully_covering_owner() -> None:
    layout = _layout()
    shape = (layout.height, 80)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    valid = np.ones(shape, dtype=bool)
    confidence = np.ones(shape, dtype=np.float32)
    rasters = [
        _ReferencePanelRaster(
            panel_index=0,
            frame_id=10,
            corner_x=0,
            image_bgr=image,
            valid_mask=valid,
            protected_mask=np.zeros(shape, dtype=bool),
            confidence=confidence,
        ),
        _ReferencePanelRaster(
            panel_index=1,
            frame_id=11,
            corner_x=70,
            image_bgr=image,
            valid_mask=valid,
            protected_mask=np.zeros(shape, dtype=bool),
            confidence=confidence,
        ),
    ]
    mesh_masks = [
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=bool),
    ]
    # One connected rendered object lies wholly in the panels' real RGB
    # overlap.  The second source has more accepted depth support and must own
    # the complete object, including pixels where only its reference RGB is
    # available.
    foreground = np.zeros((layout.height, layout.width), dtype=bool)
    foreground[20:30, 72:78] = True
    mesh_masks[0][20:30, 72:75] = True
    mesh_masks[1][20:30, 2:8] = True
    source_images = [
        np.full((*shape, 3), (20, 30, 40), dtype=np.uint8),
        np.full((*shape, 3), (80, 90, 100), dtype=np.uint8),
    ]
    meshes = [
        _DepthMeshPanelRemap(
            corner_x=corner,
            map_x=np.zeros(shape, dtype=np.float32),
            map_y=np.zeros(shape, dtype=np.float32),
            relative_depth_mm=np.full(shape, 500.0, dtype=np.float32),
            valid_mask=mask,
            audit={},
        )
        for corner, mask in zip((0, 70), mesh_masks, strict=True)
    ]
    output_image = np.zeros(
        (layout.height, layout.width, 3), dtype=np.uint8
    )
    output_depth = np.full(
        (layout.height, layout.width), 1000.0, dtype=np.float32
    )
    output_confidence = np.full(
        (layout.height, layout.width), 0.1, dtype=np.float32
    )
    output_owner = np.full(
        (layout.height, layout.width), 10, dtype=np.int32
    )
    output_owner[:, 75:] = 11
    output_reliable = foreground.copy()
    output_depth[foreground] = 500.0

    audit = _enforce_foreground_components_single_owner(
        output_image=output_image,
        output_depth=output_depth,
        output_confidence=output_confidence,
        output_owner=output_owner,
        output_reliable_depth=output_reliable,
        reference_rasters=rasters,
        depth_mesh_candidates=[
            (mesh, source_image, confidence, frame_id)
            for mesh, source_image, frame_id in zip(
                meshes, source_images, (10, 11), strict=True
            )
        ],
        reference_depth_mm=layout.reference_depth_mm,
        config=InspectionMultiviewConfig(
            minimum_depth_mm=200.0,
            maximum_depth_mm=1500.0,
        ),
    )

    assert audit["all_components_assigned"] is True
    assert audit["component_count"] == 1
    assert audit["replaced_component_count"] == 1
    assert np.array_equal(
        np.unique(output_owner[foreground]), np.asarray([11])
    )
    # Replacement RGB comes from the caller-provided panel raster (the
    # exposure-compensated photometric domain), never the raw mesh source.
    assert np.all(output_image[foreground] == 0)
    assert audit["components"][0]["row_owner_topology_preserved"] is True


def test_final_owner_topology_audit_rejects_backward_repeated_island() -> None:
    monotonic = np.asarray(
        [[10, 10, 11, 11, 12, 12]], dtype=np.int32
    )
    valid = np.ones(monotonic.shape, dtype=bool)
    good = _owner_topology_audit(monotonic, valid, (10, 11, 12))
    assert good["pass"] is True
    assert good["backward_transition_count"] == 0
    assert good["repeated_owner_island_count"] == 0

    repeated = np.asarray(
        [[10, 10, 11, 11, 10, 12]], dtype=np.int32
    )
    bad = _owner_topology_audit(repeated, valid, (10, 11, 12))
    assert bad["pass"] is False
    assert bad["backward_transition_count"] == 1
    assert bad["repeated_owner_island_count"] == 1
    assert bad["backward_transition_example_rows"] == [0]


def test_locked_foreground_rgb_uses_inverse_mesh_and_fills_only_map() -> None:
    shape = (12, 20)
    yy, xx = np.indices(shape, dtype=np.float32)
    component = np.zeros(shape, dtype=bool)
    component[3:9, 6:14] = True
    mesh_valid = component.copy()
    mesh_valid[5:7, 9:11] = False
    mesh = _DepthMeshPanelRemap(
        corner_x=0,
        map_x=np.where(mesh_valid, xx, np.nan).astype(np.float32),
        map_y=np.where(mesh_valid, yy, np.nan).astype(np.float32),
        relative_depth_mm=np.where(
            mesh_valid, 500.0, np.inf
        ).astype(np.float32),
        valid_mask=mesh_valid,
        audit={},
    )
    source = np.full((*shape, 3), (31, 63, 127), dtype=np.uint8)
    confidence = np.ones(shape, dtype=np.float32)
    raster = _ReferencePanelRaster(
        panel_index=0,
        frame_id=10,
        corner_x=0,
        image_bgr=np.zeros_like(source),
        valid_mask=np.ones(shape, dtype=bool),
        protected_mask=np.zeros(shape, dtype=bool),
        confidence=confidence,
    )
    locked_panel = np.full(shape, -1, dtype=np.int16)
    locked_panel[component] = 0
    locked_labels = np.zeros(shape, dtype=np.int32)
    locked_labels[component] = 1
    output_image = np.zeros_like(source)
    output_confidence = np.zeros(shape, dtype=np.float32)
    output_owner = np.full(shape, -1, dtype=np.int32)
    output_owner[component] = 10

    audit = _composite_locked_foreground_mesh_rgb(
        locked_panel_index=locked_panel,
        locked_component_labels=locked_labels,
        reference_rasters=[raster],
        depth_mesh_candidates=[(mesh, source, confidence, 10)],
        compensated_source_images=[source],
        output_image=output_image,
        output_confidence=output_confidence,
        output_owner=output_owner,
        config=InspectionMultiviewConfig(),
    )

    assert np.all(output_image[component] == np.asarray([31, 63, 127]))
    assert audit["all_components_inverse_sampled"] is True
    assert audit["same_layer_map_fill_pixel_count"] == 4
    assert audit["reference_plane_rgb_fallback_pixel_count"] == 0


def _write_frame(
    root: Path,
    frame_id: int,
    color: np.ndarray,
    depth: np.ndarray,
) -> RGBDFrame:
    color_path = root / f"color_{frame_id}.png"
    depth_path = root / f"depth_{frame_id}.png"
    assert cv2.imwrite(str(color_path), color)
    assert cv2.imwrite(str(depth_path), depth)
    return RGBDFrame(
        frame_id=frame_id,
        color_path=color_path,
        aligned_depth_path=depth_path,
        depth_scale_mm_per_unit=1.0,
    )


def test_default_render_does_not_retain_unused_reference_maps(
    tmp_path: Path,
) -> None:
    intrinsics = _intrinsics(width=128)
    yy, xx = np.indices((intrinsics.height, intrinsics.width))
    first_color = np.stack(
        (
            (xx * 2) % 256,
            (yy * 3) % 256,
            np.full_like(xx, 96),
        ),
        axis=2,
    ).astype(np.uint8)
    second_color = np.roll(first_color, 4, axis=1)
    depth = np.full(
        (intrinsics.height, intrinsics.width), 1000, np.uint16
    )
    frames = [
        _write_frame(tmp_path, 20, first_color, depth),
        _write_frame(tmp_path, 21, second_color, depth),
    ]
    poses = [np.eye(4), np.eye(4)]
    poses[1] = poses[1].copy()
    poses[1][0, 3] = 120.0

    result = render_inspection_multiview(
        frames,
        poses,
        intrinsics,
        config=InspectionMultiviewConfig(
            minimum_depth_mm=200.0,
            maximum_depth_mm=1500.0,
            preview_stride=2,
            chunk_rows=20,
            foreground_world_anchor_enabled=False,
        ),
    )

    audit = result.metadata["reference_inverse_maps"]
    assert audit["foreground_world_anchor_enabled"] is False
    assert audit["reference_panel_count"] >= 2
    assert audit["retained_panel_count"] == 0
    assert audit["retained_panel_indices"] == []
    assert audit["retained_bytes"] == 0
    assert audit["lazy_recomputed_panel_count"] == 0
    assert audit["unused_map_retention_count"] == 0
    assert audit["depth_mesh_source_image_policy"] == (
        "reference_panel_placeholder_no_rgb_read_in_write_rgb_false_path"
    )


def test_pre_seam_hard_owner_interval_is_guarded_and_crop_preserved(
    tmp_path: Path,
) -> None:
    intrinsics = _intrinsics(width=128)
    yy, xx = np.indices((intrinsics.height, intrinsics.width))
    first_color = np.stack(
        (
            (xx * 2) % 256,
            (yy * 3) % 256,
            np.full_like(xx, 96),
        ),
        axis=2,
    ).astype(np.uint8)
    second_color = np.roll(first_color, 4, axis=1)
    depth = np.full(
        (intrinsics.height, intrinsics.width), 1000, np.uint16
    )
    frames = [
        _write_frame(tmp_path, 30, first_color, depth),
        _write_frame(tmp_path, 31, second_color, depth),
    ]
    poses = [np.eye(4), np.eye(4)]
    poses[1] = poses[1].copy()
    poses[1][0, 3] = 120.0
    config = InspectionMultiviewConfig(
        minimum_depth_mm=200.0,
        maximum_depth_mm=1500.0,
        preview_stride=2,
        chunk_rows=20,
        foreground_world_anchor_enabled=False,
    )
    baseline = render_inspection_multiview(
        frames, poses, intrinsics, config=config
    )
    explicit_empty = render_inspection_multiview(
        frames,
        poses,
        intrinsics,
        pre_seam_hard_owner_intervals=(),
        config=config,
    )
    assert np.array_equal(explicit_empty.image_bgr, baseline.image_bgr)
    assert np.array_equal(
        explicit_empty.owner_frame_id, baseline.owner_frame_id
    )

    layout = estimate_inspection_layout(
        frames, poses, intrinsics, config=config
    )
    seam = baseline.metadata["background_seam_audit"][
        "panel_chain_seams"
    ][0]
    nominal_x = int(round(float(seam["nominal_x"])))
    lock = np.zeros((layout.height, layout.width), dtype=bool)
    lock[20:40, nominal_x - 4 : nominal_x + 5] = True
    footprint = np.zeros_like(lock)
    footprint[22:38, nominal_x - 2 : nominal_x + 3] = True
    selected = baseline.metadata["selected_panel_sources"][0]
    interval = InspectionPreSeamHardOwnerInterval(
        track_id=99,
        panel_index=int(selected["panel_index"]),
        frame_id=int(selected["frame_id"]),
        lock_mask=lock,
        union_footprint=footprint,
    )

    result = render_inspection_multiview(
        frames,
        poses,
        intrinsics,
        pre_seam_hard_owner_intervals=(interval,),
        config=config,
    )

    audit = result.metadata["background_seam_audit"][
        "pre_seam_hard_owner_intervals"
    ]
    assert audit["used"] is True
    assert audit["solver_locked_owner_mismatch_pixel_count"] == 0
    assert audit["final_owner_mismatch_pixel_count"] == 0
    assert audit["multiband_intersection_pixel_count"] == 0
    assert audit["dis_flow_intersection_pixel_count"] == 0
    assert audit["crop_preserved_all_locked_pixels"] is True
    assert audit["post_crop_owner_mismatch_pixel_count"] == 0
    assert "pre_seam_single_panel_hard_owner_interval_used" in (
        result.metadata["strict_incomplete_reasons"]
    )
    assert result.metadata["strict_v1_inspection_complete"] is False


def test_full_fov_rgbd_reprojection_and_nearest_surface_owner(tmp_path: Path) -> None:
    intrinsics = _intrinsics(width=128)
    yy, xx = np.indices((intrinsics.height, intrinsics.width))
    first_color = np.stack(
        (
            (xx * 3) % 256,
            (yy * 4) % 256,
            np.full_like(xx, 80),
        ),
        axis=2,
    ).astype(np.uint8)
    second_color = np.zeros_like(first_color)
    second_color[..., 1] = 220
    far = np.full((intrinsics.height, intrinsics.width), 1000, np.uint16)
    near = far.copy()
    near[16:48, 56:88] = 500
    frames = [
        _write_frame(tmp_path, 10, first_color, far),
        _write_frame(tmp_path, 11, second_color, near),
    ]
    poses = [np.eye(4), np.eye(4)]
    poses[1] = poses[1].copy()
    poses[1][0, 3] = 120.0

    result = render_inspection_multiview(
        frames,
        poses,
        intrinsics,
        config=InspectionMultiviewConfig(
            minimum_depth_mm=200.0,
            maximum_depth_mm=1500.0,
            preview_stride=2,
            chunk_rows=20,
            foreground_world_anchor_enabled=True,
        ),
    )

    assert result.metadata["fixed_strip_pushbroom"] is False
    assert result.metadata["metric_raster_used_for_rgb"] is False
    reference_maps = result.metadata["reference_inverse_maps"]
    assert reference_maps["foreground_world_anchor_enabled"] is True
    assert reference_maps["retained_panel_count"] == (
        reference_maps["reference_panel_count"]
    )
    assert reference_maps["retained_bytes"] > 0
    assert reference_maps["depth_mesh_source_image_policy"] == (
        "original_rgb_retained_for_enabled_world_anchor"
    )
    resource = result.metadata["resource_estimate"]
    assert resource["within_budget"] is True
    assert resource["panel_canvas_product_bytes"] == 0
    assert resource["pair_canvas_product_bytes"] == 0
    audits = result.metadata["source_audits"]
    assert all(item["full_width_source_sampling"] for item in audits)
    assert all(not item["central_twenty_percent_only"] for item in audits)
    assert all(
        item["foreground_sampling_model"]
        == "single_full_fov_panel_rgb_with_depth_mesh_visibility"
        for item in audits
    )
    assert any(
        item["reliable_foreground_geometry_pixel_count"] > 0
        for item in audits
    )
    assert 11 in np.unique(result.owner_frame_id[result.valid_mask])
    assert float(np.nanmin(result.relative_depth_mm)) < 750.0
    seam = result.metadata["background_seam_audit"]
    assert seam["exposure_compensation_used"] is True
    assert seam["protected_pixel_count"] > 0
    assert seam["protected_blend_intersection_pixel_count"] == 0
    compact_storage = seam["compact_evidence_storage"]
    assert compact_storage["model"] == (
        "panel_local_and_pair_corridor/v1"
    )
    assert compact_storage["per_panel_full_canvas_array_count"] == 0
    assert compact_storage["per_pair_full_canvas_array_count"] == 0
    assert compact_storage["pair_cost_bytes"] == (
        (len(audits) - 1)
        * resource["canvas_height"]
        * InspectionMultiviewConfig().chain_seam_corridor_width_pixels
        * 4
    )
    adaptive = seam["panel_chain_topology"][
        "adaptive_boundary_selection"
    ]
    assert adaptive["corridors_nonoverlapping"] is True
    assert adaptive["risk_is_seam_forbidden"] is False
    assert adaptive["risk_usage"] == (
        "adaptive_nominal_boundary_selection; foreground_"
        "components_use_explicit_single_panel_owner_locks"
    )
    assert adaptive["pair_count"] == 1
    assert adaptive["risk_pixel_counts"][0] > 0
    assert "full_depth_mesh_union" in adaptive["risk_sources"]
    assert "left_right_reference_protected_masks" in adaptive["risk_sources"]
    component_locks = seam["foreground_component_locks"]
    assert component_locks["candidate_chain_lock_component_count"] >= (
        component_locks["retained_chain_lock_component_count"]
    )
    assert component_locks["rejected_chain_lock_component_count"] == len(
        component_locks["rejected_chain_lock_candidates"]
    )
    assert component_locks["retained_chain_locks_applied"] is (
        component_locks["retained_chain_lock_component_count"] > 0
    )
    assert result.metadata["foreground_owner_continuity_summary"][
        "foreground_blend_pixel_count"
    ] == 0
    assert result.metadata["foreground_owner_continuity_summary"][
        "all_components_single_owner"
    ] is True
    assert result.metadata["foreground_component_assignment"][
        "all_components_assigned"
    ] is True
    assert result.metadata["foreground_component_assignment"][
        "rgb_photometric_domain"
    ].startswith("same_opencv_channels")
    object_anchor = result.metadata[
        "foreground_component_assignment"
    ]["object_world_anchor"]
    assert object_anchor["all_tracks_visible"] is True
    assert object_anchor["blend_pixel_count"] == 0
    assert object_anchor["track_count"] >= 1
    category_counts = result.metadata[
        "foreground_component_assignment"
    ]["category_counts"]
    assert category_counts["reliable_mesh_component_count"] == (
        object_anchor["track_count"]
    )
    assert (
        category_counts["invalid_depth_owner_only_component_count"]
        == seam["invalid_depth_owner_only_locks"]["component_count"]
    )
    assert seam["invalid_depth_owner_only_locks"][
        "reference_plane_rgb_allowed"
    ] is True
    assert seam["invalid_depth_owner_only_locks"][
        "multiband_allowed"
    ] is False
    assert result.metadata["foreground_component_assignment"][
        "assignment_stage"
    ] == (
        "rgbd_world_track_before_seam_then_depth_ordered_hard_"
        "object_overlay_after_monotone_background_chain"
    )
    assert result.metadata["foreground_component_assignment"][
        "post_composition_foreground_overlay_component_count"
    ] == object_anchor["track_count"]
    assert seam["continuous_canvas_exposure"]["application_order"] == (
        "after_all_foreground_component_owner_replacements"
    )
    assert seam["continuous_canvas_exposure"][
        "applied_uniformly_to_background_and_foreground"
    ] is True
    assert seam["continuous_canvas_exposure"][
        "foreground_rgb_preserved_from_selected_real_owner"
    ] is True
    assert seam["continuous_canvas_exposure"][
        "owner_only_guard_pixel_count"
    ] > 0
    assert seam["continuous_canvas_exposure"][
        "corrected_owner_only_guard_intersection_pixel_count"
    ] > 0
    assert seam["owner_boundary_visual_audit"][
        "owner_only_guard_pixel_count"
    ] > 0
    assert result.metadata["owner_topology_audit"]["pass"] is True
    assert result.metadata["owner_topology_audit"][
        "all_rows_monotonic"
    ] is True
    assert result.metadata["owner_topology_audit"][
        "no_repeated_owner_islands"
    ] is True
    assert result.metadata["owner_topology_audit"]["scope"] == (
        "background_pixels_excluding_rgbd_object_overlay_and_"
        "audited_reference_footprint_replacements"
    )
    assert result.metadata["owner_topology_audit"][
        "foreground_owner_islands_allowed"
    ] is True


def test_resource_limit_fails_before_canvas_allocation(tmp_path: Path) -> None:
    intrinsics = _intrinsics()
    color = np.zeros((intrinsics.height, intrinsics.width, 3), np.uint8)
    depth = np.full((intrinsics.height, intrinsics.width), 1000, np.uint16)
    frames = [
        _write_frame(tmp_path, 0, color, depth),
        _write_frame(tmp_path, 1, color, depth),
    ]
    poses = [np.eye(4), np.eye(4)]
    poses[1] = poses[1].copy()
    poses[1][0, 3] = 1_000_000.0

    with pytest.raises(MemoryError, match="resource limit"):
        render_inspection_multiview(
            frames,
            poses,
            intrinsics,
            config=InspectionMultiviewConfig(
                maximum_canvas_megapixels=0.01
            ),
        )
