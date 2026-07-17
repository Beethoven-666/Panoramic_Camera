from dataclasses import replace

import numpy as np
import pytest
import cv2
import panorama_demo.calibrated_rgb_pushbroom as pushbroom_module
import panorama_demo.geometry_assisted_local_warp as geometry_module

from panorama_demo.geometry_assisted_local_warp import (
    GeometryAssistConfig,
    GeometryIntrinsics,
    LayerLabel,
    LocalMeshInverseWarp,
    LocalMeshWarpConfig,
    LocalWarpConfig,
    TileBounds,
    analyze_adjacent_rgbd_pair,
    depth_edge_guard,
    depth_tolerance_mm,
    extract_signed_occlusion_foreground_components,
    fit_local_inverse_warp,
    fit_local_mesh_inverse_warp,
    match_bidirectional_signed_occlusion_instances,
    mutually_consistent_correspondences,
    sample_aligned_depth_nearest,
    solve_active_mesh_forward_inverse,
)


def _intrinsics(width: int, height: int, focal: float = 100.0) -> GeometryIntrinsics:
    return GeometryIntrinsics(
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


def test_bidirectional_reprojection_keeps_real_se3_translation_and_mutual_matches() -> None:
    height, width = 7, 31
    camera = _intrinsics(width, height)
    first = np.full((height, width), 1000.0, dtype=np.float32)
    second = np.full_like(first, 1000.0)

    result = analyze_adjacent_rgbd_pair(first, second, camera, _pose(), _pose(10.0))
    forward = result.first_to_second

    # A point at the first camera centre moves one native pixel left in a
    # camera translated +10 mm with f=100 and z=1000 mm.
    y, x = height // 2, width // 2
    assert forward.target_x[y, x] == pytest.approx(x - 1.0)
    assert forward.target_y[y, x] == pytest.approx(y)
    assert forward.depth_consistent[y, x]
    assert forward.mutual_consistent[y, x]
    assert forward.labels[y, x] == LayerLabel.CONSISTENT
    source, target = mutually_consistent_correspondences(forward)
    assert len(source) > 0
    assert np.any(np.all(source == (x, y), axis=1))
    assert not hasattr(result, "rgb")


def test_round_trip_depth_miss_is_protected_even_when_one_way_depth_matches() -> None:
    """A depth-consistent splat cannot enter a mesh without a round trip."""

    height, width = 7, 101
    camera = _intrinsics(width, height)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    result = analyze_adjacent_rgbd_pair(
        depth,
        depth,
        camera,
        _pose(),
        _pose(0.1),
        config=GeometryAssistConfig(
            edge_guard_radius_pixels=0,
            mutual_pixel_tolerance=0.4,
        ),
    )
    forward = result.first_to_second
    one_way_only = (
        forward.depth_consistent
        & forward.zbuffer_visible
        & ~forward.mutual_consistent
    )

    assert np.any(one_way_only)
    assert np.all(forward.protected_mask[one_way_only])
    source, _target = mutually_consistent_correspondences(forward)
    assert len(source) < int(np.count_nonzero(forward.depth_consistent))


def test_nearest_target_camera_zbuffer_wins_collision_deterministically() -> None:
    height, width = 3, 21
    camera = _intrinsics(width, height)
    first = np.zeros((height, width), dtype=np.float32)
    second = np.zeros_like(first)
    # Both first pixels project to second pixel (8, 1).  The 500 mm sample is
    # nearer in the *target camera* and must win instead of being averaged.
    first[1, 10] = 500.0
    first[1, 9] = 1000.0
    second[1, 8] = 500.0

    result = analyze_adjacent_rgbd_pair(first, second, camera, _pose(), _pose(10.0))
    forward = result.first_to_second

    assert forward.zbuffer_visible[1, 10]
    assert not forward.zbuffer_visible[1, 9]
    assert forward.labels[1, 10] == LayerLabel.CONSISTENT
    assert forward.labels[1, 9] == LayerLabel.OCCLUDED
    assert forward.zbuffer_depth_mm[1, 8] == pytest.approx(500.0)
    assert forward.zbuffer_source_index[1, 8] == 1 * width + 10


def test_depth_mismatch_marks_nearer_source_as_foreground_not_same_surface() -> None:
    height, width = 3, 21
    camera = _intrinsics(width, height)
    first = np.zeros((height, width), dtype=np.float32)
    second = np.zeros_like(first)
    first[1, 10] = 500.0
    # The reprojected first point lands at second (8, 1), but this target sees
    # a farther 1000 mm surface.  It is a first-source foreground occluder.
    second[1, 8] = 1000.0

    result = analyze_adjacent_rgbd_pair(first, second, camera, _pose(), _pose(10.0))
    forward = result.first_to_second

    assert forward.source_foreground[1, 10]
    assert not forward.depth_consistent[1, 10]
    assert forward.labels[1, 10] == LayerLabel.FOREGROUND
    assert forward.protected_mask[1, 10]


def test_mutually_visible_near_depth_component_is_still_hard_owned() -> None:
    """A thick handle/hose cannot become mesh-safe just by being visible twice."""

    height = width = 101
    camera = _intrinsics(width, height)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    # A 500 mm object is fully visible in both identical views.  Its inner
    # region lies outside the 8 px edge guard, so this tests layer ownership
    # rather than the pre-existing discontinuity dilation.
    depth[30:71, 30:71] = 500.0
    result = analyze_adjacent_rgbd_pair(
        depth,
        depth,
        camera,
        _pose(),
        _pose(),
        config=GeometryAssistConfig(edge_guard_radius_pixels=8),
    )
    safety = result.first_surface_safety

    assert result.first_to_second.mutual_consistent[50, 50]
    assert safety.near_foreground_mask[50, 50]
    assert safety.hard_owner_mask[50, 50]
    assert not safety.mesh_safe_mask[50, 50]
    assert safety.mesh_safe_mask[10, 10]
    assert safety.near_foreground_component_count == 1
    assert result.audit.first_surface_safety["near_foreground_pixel_count"] == 529


def _direct_signed_foreground(
    base: object, component_mask: np.ndarray
) -> object:
    """Mark one synthetic direct foreground region without changing geometry."""

    reprojection = base
    labels = np.asarray(reprojection.labels, dtype=np.uint8).copy()
    labels[component_mask] = int(LayerLabel.FOREGROUND)
    residual_ratio = np.asarray(reprojection.depth_residual_ratio, dtype=np.float32).copy()
    residual_ratio[component_mask] = 2.0
    return replace(
        reprojection,
        source_foreground=np.asarray(component_mask, dtype=bool),
        labels=labels,
        depth_residual_ratio=residual_ratio,
    )


def test_signed_occlusion_instances_require_direct_foreground_and_virtual_overlap() -> None:
    """A foreground anchor is direct bilateral occlusion evidence, not a mesh layer."""

    height = width = 101
    camera = _intrinsics(width, height)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    base = analyze_adjacent_rgbd_pair(depth, depth, camera, _pose(), _pose())
    component = np.zeros_like(depth, dtype=bool)
    component[30:71, 30:71] = True
    first = extract_signed_occlusion_foreground_components(
        _direct_signed_foreground(base.first_to_second, component)
    )
    second = extract_signed_occlusion_foreground_components(
        _direct_signed_foreground(base.second_to_first, component)
    )

    instances = match_bidirectional_signed_occlusion_instances(
        first, second, first.labels, second.labels
    )

    assert first.component_count == second.component_count == 1
    assert first.eroded_core_pixel_count == second.eroded_core_pixel_count == 1521
    assert len(instances.matches) == 1
    match = instances.matches[0]
    assert match.first_component_pixel_count == 1681
    assert match.second_component_pixel_count == 1681
    assert match.virtual_overlap_pixel_count == 1681
    assert match.virtual_overlap_row_count == 41
    assert instances.first_instance_labels[50, 50] == match.instance_id
    assert instances.second_instance_labels[50, 50] == match.instance_id


def test_mutual_same_layer_background_cannot_be_promoted_to_a_foreground_instance() -> None:
    """The former seed-and-grow route may never bridge mutual wall pixels."""

    height = width = 101
    camera = _intrinsics(width, height)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    base = analyze_adjacent_rgbd_pair(depth, depth, camera, _pose(), _pose())
    # The whole image is mutual, z-buffer-visible and depth-consistent wall,
    # but it carries no signed foreground label.
    assert base.first_to_second.mutual_consistent[50, 50]
    assert base.first_to_second.depth_consistent[50, 50]
    first = extract_signed_occlusion_foreground_components(base.first_to_second)
    second = extract_signed_occlusion_foreground_components(base.second_to_first)

    instances = match_bidirectional_signed_occlusion_instances(
        first, second, first.labels, second.labels
    )

    assert first.component_count == second.component_count == 0
    assert instances.matches == ()
    assert not np.any(instances.first_instance_labels)
    assert not np.any(instances.second_instance_labels)


def test_signed_occlusion_instance_rejects_two_equally_plausible_partners() -> None:
    """A split foreground label cannot select an arbitrary RGB owner."""

    height = width = 101
    camera = _intrinsics(width, height)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    base = analyze_adjacent_rgbd_pair(depth, depth, camera, _pose(), _pose())
    first_mask = np.zeros_like(depth, dtype=bool)
    first_mask[20:50, 20:50] = True
    second_mask = np.zeros_like(depth, dtype=bool)
    second_mask[20:50, 20:35] = True
    second_mask[20:50, 40:55] = True
    first = extract_signed_occlusion_foreground_components(
        _direct_signed_foreground(base.first_to_second, first_mask)
    )
    second = extract_signed_occlusion_foreground_components(
        _direct_signed_foreground(base.second_to_first, second_mask)
    )
    assert first.component_count == 1
    assert second.component_count == 2

    first_virtual = np.zeros_like(first.labels)
    second_virtual = np.zeros_like(second.labels)
    first_virtual[20:50, 20:50] = 1
    second_virtual[20:50, 20:35] = 1
    second_virtual[20:50, 35:50] = 2
    instances = match_bidirectional_signed_occlusion_instances(
        first, second, first_virtual, second_virtual
    )

    assert instances.matches == ()
    assert instances.rejected_component_count == 3
    assert not np.any(instances.first_instance_labels)
    assert not np.any(instances.second_instance_labels)


def test_tiny_far_depth_sliver_cannot_disqualify_a_dominant_safe_wall() -> None:
    """The background anchor ignores a distant sub-10%-area depth fragment."""

    height = width = 101
    camera = _intrinsics(width, height)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    depth[5:17, 5:17] = 1600.0
    result = analyze_adjacent_rgbd_pair(
        depth,
        depth,
        camera,
        _pose(),
        _pose(),
        config=GeometryAssistConfig(edge_guard_radius_pixels=0),
    )
    safety = result.first_surface_safety

    assert safety.mesh_safe_mask[50, 50]
    assert safety.hard_owner_mask[10, 10]
    assert safety.depth_anchor_component_count == 1
    assert result.audit.first_surface_safety["dominant_component_median_depth_mm"] == (
        pytest.approx(1000.0)
    )


def test_safe_planar_geometry_chain_can_accept_a_default_local_mesh() -> None:
    """A real RGB-D wall can reach the bounded inverse-mesh solver safely.

    This intentionally stops before RGB preview/Hough validation, which belongs
    to the renderer.  It proves the geometry half of the formal path is
    satisfiable without loosening any depth, SE(3), mesh, or held-out limits:
    bidirectional reprojection supplies one safe wall component, then an
    independently bounded local inverse sampling residual is accepted.  The
    residual is a test-only sampling discrepancy, never a changed pose/depth.
    """

    height, width = 128, 192
    camera = _intrinsics(width, height)
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    geometry = analyze_adjacent_rgbd_pair(
        depth,
        depth,
        camera,
        _pose(),
        _pose(10.0),
        config=GeometryAssistConfig(
            mutual_pixel_tolerance=0.40,
            edge_guard_radius_pixels=8,
        ),
    )
    forward = geometry.first_to_second
    reverse = geometry.second_to_first
    source, target = mutually_consistent_correspondences(forward)
    source_x = source[:, 0].astype(np.int64)
    source_y = source[:, 1].astype(np.int64)
    target_x = np.rint(target[:, 0]).astype(np.int64)
    target_y = np.rint(target[:, 1]).astype(np.int64)
    bilateral_safe = (
        geometry.first_surface_safety.mesh_safe_mask[source_y, source_x]
        & geometry.second_surface_safety.mesh_safe_mask[target_y, target_x]
        & ~forward.protected_mask[source_y, source_x]
        & ~reverse.protected_mask[target_y, target_x]
    )
    same_layer = np.zeros((height, width), dtype=bool)
    same_layer[source_y[bilateral_safe], source_x[bilateral_safe]] = True
    selected, component_audit = (
        pushbroom_module._select_single_virtual_background_component(
            same_layer,
            corridor_x=(0, width),
            nominal_boundary_x=width // 2,
        )
    )

    assert geometry.first_surface_safety.dominant_component_fraction == pytest.approx(1.0)
    assert geometry.second_surface_safety.dominant_component_fraction == pytest.approx(1.0)
    assert component_audit["boundary_crossing_component_count"] == 1
    assert component_audit["nonselected_depth_safe_pixel_count"] == 0

    fit_support = (
        bilateral_safe
        & (source[:, 0] >= 24.0)
        & (source[:, 0] <= 144.0)
        & (source[:, 1] >= 24.0)
        & (source[:, 1] <= 112.0)
    )
    output = source[fit_support]
    # In the shared virtual tile, the second raw pixel has a one-pixel pose
    # translation relative to the first.  Undoing that exact SE(3) shift gives
    # the same common coordinate before the deliberately small local residual.
    baseline_sample = target[fit_support] + np.asarray((1.0, 0.0))
    np.testing.assert_allclose(baseline_sample, output)
    sample = baseline_sample + np.column_stack(
        (
            0.48 + 0.002 * (output[:, 0] - 50.0),
            -0.32 + 0.0016 * (output[:, 1] - 50.0),
        )
    )
    fit = fit_local_mesh_inverse_warp(
        output,
        sample,
        TileBounds(0.0, 0.0, float(width - 1), float(height - 1)),
        same_layer_mask=selected,
        config=LocalMeshWarpConfig(minimum_correspondences=30, held_out_seed=9),
    )

    assert fit.warp is not None
    assert fit.audit.accepted
    assert fit.audit.active_cell_count >= 4
    assert fit.audit.held_out_error_p95_after_pixels <= 0.75
    assert fit.audit.held_out_error_max_after_pixels <= 2.0
    assert fit.audit.maximum_displacement_pixels <= 8.0
    assert fit.audit.maximum_straight_line_deviation_pixels <= 1.0


def test_depth_edge_guard_marks_both_sides_and_hole_boundaries() -> None:
    depth = np.full((8, 10), 1000.0, dtype=np.float32)
    depth[:, :5] = 500.0
    depth[4, 2] = 0.0
    edge, guard = depth_edge_guard(
        depth, config=GeometryAssistConfig(edge_guard_radius_pixels=1)
    )

    assert np.all(edge[:, 4])
    assert np.all(edge[:, 5])
    assert edge[4, 1] or edge[4, 3]
    assert np.all(guard[:, 4])
    assert np.all(guard[:, 5])
    assert guard[4, 2]


def test_depth_tolerance_combines_absolute_relative_and_noise_terms() -> None:
    config = GeometryAssistConfig(depth_noise_mm=11.0, depth_sigma_multiplier=3.0)
    tolerance = depth_tolerance_mm(np.asarray((500.0, 2000.0)), config)

    np.testing.assert_allclose(tolerance, (33.0, 40.0))


def test_local_mesh_config_keeps_the_non_relaxable_structural_limits() -> None:
    with pytest.raises(ValueError, match="minimum_active_cells"):
        LocalMeshWarpConfig(minimum_active_cells=3).validate()
    with pytest.raises(ValueError, match="maximum_straight_line_deviation_pixels"):
        LocalMeshWarpConfig(maximum_straight_line_deviation_pixels=1.01).validate()


def test_mesh_node_shared_only_by_disconnected_cells_is_pinned() -> None:
    """A diagonal owner/depth gap cannot communicate mesh displacement."""

    active = np.asarray(((True, False), (False, True)), dtype=bool)
    active_nodes, free_nodes, component_labels = geometry_module._mesh_free_nodes(active)

    assert active_nodes[1, 1]
    assert not free_nodes[1, 1]
    assert component_labels[1, 1] == 0


def test_sample_aligned_depth_nearest_handles_an_arbitrary_narrow_2d_map() -> None:
    raw_depth = np.asarray(
        ((0.0, 101.0, 102.0, 103.0), (201.0, 202.0, 0.0, 204.0), (301.0, 302.0, 303.0, 304.0)),
        dtype=np.float32,
    )
    # This is a 2 x 3 strip, not an image-sized inverse map.  Values at .49
    # and .51 prove nearest-neighbour selection rather than depth blending.
    map_x = np.asarray(((1.49, 1.51, -0.1), (3.0, 2.0, np.nan)), dtype=np.float32)
    map_y = np.asarray(((0.0, 0.0, 1.0), (1.0, 1.0, 2.0)), dtype=np.float32)

    sampled = sample_aligned_depth_nearest(raw_depth, map_x, map_y)

    np.testing.assert_array_equal(
        sampled.valid_mask,
        np.asarray(((True, True, False), (True, False, False))),
    )
    np.testing.assert_allclose(
        sampled.depth_mm,
        np.asarray(((101.0, 102.0, 0.0), (204.0, 0.0, 0.0)), dtype=np.float32),
    )


def test_sample_aligned_depth_nearest_honours_raw_valid_mask() -> None:
    raw_depth = np.full((3, 4), 750.0, dtype=np.float32)
    raw_valid = np.ones(raw_depth.shape, dtype=bool)
    raw_valid[1, 2] = False
    map_x = np.asarray(((2.0, 1.0), (2.0, 3.0)), dtype=np.float32)
    map_y = np.asarray(((1.0, 1.0), (0.0, 2.0)), dtype=np.float32)

    sampled = sample_aligned_depth_nearest(
        raw_depth, map_x, map_y, valid_mask=raw_valid
    )

    np.testing.assert_array_equal(
        sampled.valid_mask,
        np.asarray(((False, True), (True, True))),
    )
    assert sampled.depth_mm[0, 0] == 0.0


def test_bounded_local_inverse_warp_accepts_held_out_improvement_and_tapers_boundary() -> None:
    values = np.asarray((20.0, 35.0, 50.0, 65.0, 80.0))
    grid_x, grid_y = np.meshgrid(values, values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source = output + np.asarray((2.0, -1.0))
    bounds = TileBounds(0.0, 0.0, 100.0, 100.0)

    result = fit_local_inverse_warp(
        output,
        source,
        bounds,
        config=LocalWarpConfig(minimum_correspondences=12, held_out_seed=17),
    )

    assert result.warp is not None
    assert result.audit.accepted
    assert result.audit.held_out_error_p95_after_pixels == pytest.approx(0.0, abs=1e-7)
    assert result.audit.held_out_error_max_after_pixels == pytest.approx(0.0, abs=1e-7)
    sample_x, sample_y = result.warp.inverse_coordinates(50.0, 50.0)
    assert sample_x == pytest.approx(52.0)
    assert sample_y == pytest.approx(49.0)
    # A local correction cannot move pixels on its own outer boundary.
    edge_x, edge_y = result.warp.inverse_coordinates(0.0, 50.0)
    assert edge_x == pytest.approx(0.0)
    assert edge_y == pytest.approx(50.0)


def test_local_inverse_warp_rejects_excessive_displacement() -> None:
    values = np.asarray((20.0, 35.0, 50.0, 65.0, 80.0))
    grid_x, grid_y = np.meshgrid(values, values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source = output + np.asarray((10.0, 0.0))

    result = fit_local_inverse_warp(
        output,
        source,
        TileBounds(0.0, 0.0, 100.0, 100.0),
        config=LocalWarpConfig(minimum_correspondences=12, held_out_seed=3),
    )

    assert result.warp is None
    assert result.audit.reason == "maximum_displacement_exceeded"


@pytest.mark.parametrize("grid_spacing", (16, 32))
def test_local_mesh_inverse_warp_is_held_out_audited_and_boundary_identity(
    grid_spacing: int,
) -> None:
    # Keep support away from the immutable outer mesh ring.  This verifies the
    # 16 px and 32 px mesh variants can meet the formal 30% held-out
    # improvement threshold without relying on a boundary-moving warp.
    x_values = np.arange(24.0, 145.0, 4.0)
    y_values = np.arange(24.0, 113.0, 4.0)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    # A small spatially varying, same-layer inverse displacement exercises a
    # real grid rather than merely treating the mesh as one global translation.
    # It remains below the formal one-pixel newly-introduced-bend limit for
    # both allowed grid spacings.
    source = output + np.column_stack(
        (
            0.48 + 0.002 * (output[:, 0] - 50.0),
            -0.32 + 0.0016 * (output[:, 1] - 50.0),
        )
    )
    result = fit_local_mesh_inverse_warp(
        output,
        source,
        TileBounds(0.0, 0.0, 160.0, 128.0),
        config=LocalMeshWarpConfig(
            grid_spacing_pixels=grid_spacing,
            minimum_correspondences=48,
            held_out_seed=9,
            maximum_held_out_error_pixels=1.5,
        ),
    )

    assert result.warp is not None
    assert result.audit.accepted
    assert result.audit.held_out_error_p95_after_pixels < (
        result.audit.held_out_error_p95_before_pixels
    )
    assert result.audit.held_out_error_max_after_pixels <= 2.0
    assert (
        (result.audit.held_out_error_p95_before_pixels - result.audit.held_out_error_p95_after_pixels)
        / result.audit.held_out_error_p95_before_pixels
        >= 0.30
    )
    assert result.audit.boundary_identity_maximum_error_pixels == pytest.approx(0.0)
    assert result.audit.maximum_straight_line_deviation_pixels <= 1.0
    center_x, center_y = result.warp.inverse_virtual_coordinates(50.0, 50.0)
    assert center_x == pytest.approx(50.48, abs=0.35)
    assert center_y == pytest.approx(49.68, abs=0.35)
    edge_x, edge_y = result.warp.inverse_virtual_coordinates(0.0, 50.0)
    assert edge_x == pytest.approx(0.0)
    assert edge_y == pytest.approx(50.0)


def test_mesh_straight_line_audit_rejects_bending_despite_positive_jacobian() -> None:
    bounds = TileBounds(0.0, 0.0, 32.0, 32.0)
    grid = np.asarray((0.0, 16.0, 32.0))
    curved = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 1.9, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
    )
    assert geometry_module._mesh_straight_line_deviation(curved) > 1.0
    minimum_det, maximum_det, maximum_condition, _ = geometry_module._mesh_jacobian_audit(
        curved
    )
    assert minimum_det >= 0.70
    assert maximum_det <= 1.30
    assert maximum_condition <= 2.0

    # This field previously passed because centre-lines alone measured only
    # 0.95 px.  The internal vertical grid edge x=16 bends 1.9 px, so fitting
    # the exact same-layer correspondences must now reject it without a test
    # monkeypatch.
    x_values = np.arange(1.0, 32.0, 2.0)
    y_values = np.arange(1.0, 32.0, 2.0)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source_x, source_y = curved.inverse_virtual_coordinates(output[:, 0], output[:, 1])
    source = np.column_stack((source_x, source_y))

    result = fit_local_mesh_inverse_warp(
        output,
        source,
        bounds,
        config=LocalMeshWarpConfig(
            grid_spacing_pixels=16,
            minimum_correspondences=48,
            held_out_seed=9,
        ),
    )

    assert result.warp is None
    assert result.audit.reason == "mesh_straight_line_deviation_exceeded"
    assert result.audit.maximum_straight_line_deviation_pixels > 1.0


def test_raw_active_mesh_forward_inverse_never_uses_identity_fallback() -> None:
    """The observed-RGB gate gets only unique raw active-cell solutions."""

    bounds = TileBounds(0.0, 0.0, 63.0, 63.0)
    grid = np.asarray((0.0, 32.0, 63.0))
    same_layer = np.ones((64, 64), dtype=bool)
    same_layer[32, 32] = False
    warp = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.2, 0.0), (0.1, 0.8, 0.1), (0.0, 0.2, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.asarray(
            ((0.0, 0.0, 0.0), (-0.1, 0.2, -0.1), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        active_cells=np.ones((2, 2), dtype=bool),
        same_layer_mask=same_layer,
        same_layer_origin_xy=(0.0, 0.0),
    )
    q_truth = np.asarray(((12.0, 18.0), (48.0, 43.0)), dtype=np.float64)
    p_x, p_y = warp.inverse_virtual_coordinates(q_truth[:, 0], q_truth[:, 1])
    solved = solve_active_mesh_forward_inverse(
        warp, np.column_stack((p_x, p_y))
    )

    assert solved.valid_mask.all()
    np.testing.assert_allclose(solved.output_points_xy, q_truth, atol=0.05)
    assert np.nanmax(solved.residual_pixels) <= 0.05

    # The raw solver refuses a protected output/sample rather than returning
    # the renderer's identity fallback for it.
    protected_p_x, protected_p_y = warp.inverse_virtual_coordinates(32.0, 32.0)
    protected = solve_active_mesh_forward_inverse(
        warp, np.asarray([(protected_p_x, protected_p_y)], dtype=np.float64)
    )
    assert not protected.valid_mask[0]


def test_local_mesh_requires_one_connected_four_cell_same_layer_region() -> None:
    values = np.linspace(17.0, 31.0, 8)
    grid_x, grid_y = np.meshgrid(values, values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    result = fit_local_mesh_inverse_warp(
        output,
        output + np.asarray((0.4, -0.2)),
        TileBounds(0.0, 0.0, 48.0, 48.0),
        config=LocalMeshWarpConfig(minimum_correspondences=48),
    )

    assert result.warp is None
    assert result.audit.reason == "insufficient_active_mesh_cells"
    assert result.audit.active_cell_count == 1
    assert result.audit.largest_connected_active_cell_count == 1


def test_local_mesh_warp_keeps_protected_same_layer_hole_at_identity() -> None:
    x_values = np.arange(24.0, 201.0, 4.0)
    y_values = np.arange(24.0, 169.0, 4.0)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source = output + np.asarray((1.2, -0.8))
    same_layer = np.ones((193, 225), dtype=bool)
    # This is a depth/occlusion-protected square.  A mesh cell touching it is
    # inactive and sampling inside it must remain exactly identity.
    same_layer[84:101, 100:117] = False
    result = fit_local_mesh_inverse_warp(
        output,
        source,
        TileBounds(0.0, 0.0, 224.0, 192.0),
        same_layer_mask=same_layer,
        config=LocalMeshWarpConfig(
            minimum_correspondences=48,
            held_out_seed=11,
            maximum_jacobian_determinant=1.5,
        ),
    )

    assert result.warp is not None
    assert result.audit.active_cell_count > 0
    protected_x, protected_y = result.warp.inverse_virtual_coordinates(104.0, 88.0)
    assert protected_x == pytest.approx(104.0)
    assert protected_y == pytest.approx(88.0)
    # Source-side protection is equally strict: an otherwise safe point whose
    # proposed inverse sample would land in the guarded square must fall back
    # to its own original RGB coordinate instead of crossing that layer.
    source_guard_x, source_guard_y = result.warp.inverse_virtual_coordinates(99.0, 88.0)
    assert source_guard_x == pytest.approx(99.0)
    assert source_guard_y == pytest.approx(88.0)
    active_x, active_y = result.warp.inverse_virtual_coordinates(160.0, 96.0)
    assert abs(active_x - 160.0) > 0.05 or abs(active_y - 96.0) > 0.05


def test_flow_backed_background_mesh_never_moves_a_foreground_hose_guard() -> None:
    """A fine diagonal foreground stays owner-only beside an accepted wall mesh.

    The correspondences model a previously validated RGB-flow/background
    support set.  The hose itself and its 10 px guard are removed from both
    the application and fit domain, so a successful background fit must still
    have no actual non-identity samples in that foreground footprint.
    """

    height, width = 193, 225
    protected_u8 = np.zeros((height, width), dtype=np.uint8)
    cv2.line(protected_u8, (70, 25), (145, 170), 255, 8)
    protected = cv2.dilate(
        protected_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)),
    ) > 0
    background = ~protected
    x_values = np.arange(24.0, 201.0, 4.0)
    y_values = np.arange(24.0, 169.0, 4.0)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source = output + np.column_stack(
        (
            0.70 + 0.001 * (output[:, 0] - 110.0),
            -0.42 + 0.001 * (output[:, 1] - 90.0),
        )
    )

    result = fit_local_mesh_inverse_warp(
        output,
        source,
        TileBounds(0.0, 0.0, float(width - 1), float(height - 1)),
        same_layer_mask=background,
        # The supplied fit support deliberately represents flow-validated
        # background only; a foreground flow observation cannot opt the hose
        # back into the inverse map.
        fit_support_mask=background,
        config=LocalMeshWarpConfig(
            minimum_correspondences=48,
            held_out_seed=11,
            maximum_jacobian_determinant=1.5,
        ),
    )

    assert result.warp is not None
    assert result.audit.accepted
    xx, yy = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
        indexing="xy",
    )
    mapped_x, mapped_y = result.warp.inverse_virtual_coordinates(xx, yy)
    active = np.hypot(mapped_x - xx, mapped_y - yy) > 1e-9
    assert np.any(active)
    # This is the renderer-level invariant in raster form: an accepted local
    # background mesh cannot tug a hose, its metal coupling, or the guard
    # reserved for a single RGB owner.
    assert not np.any(active & protected)


def test_local_mesh_uses_sparse_fit_support_without_freezing_safe_holdout_pixels() -> None:
    """RGB-held-out support validates a safe mesh but never fits its nodes."""

    x_values = np.arange(24.0, 145.0, 8.0)
    y_values = np.arange(24.0, 113.0, 8.0)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source = output + np.column_stack(
        (
            0.48 + 0.002 * (output[:, 0] - 50.0),
            -0.32 + 0.0016 * (output[:, 1] - 50.0),
        )
    )
    safety = np.ones((129, 161), dtype=bool)
    fit_support = np.zeros_like(safety)
    # The stricter fit-support contract validates both the virtual output and
    # its proposed source sample.  Mark only sparse correspondence footprints
    # in each domain, leaving the rest of the safe wall available to a
    # validated field but unavailable for node fitting.
    for points in (output, source):
        xx = np.rint(points[:, 0]).astype(np.int64)
        yy = np.rint(points[:, 1]).astype(np.int64)
        fit_support[yy, xx] = True
    result = fit_local_mesh_inverse_warp(
        output,
        source,
        TileBounds(0.0, 0.0, 160.0, 128.0),
        same_layer_mask=safety,
        fit_support_mask=fit_support,
        config=LocalMeshWarpConfig(minimum_correspondences=48, held_out_seed=9),
    )

    assert result.warp is not None
    assert result.audit.accepted
    # This point is depth/RGB-safe but deliberately absent from the fit mask.
    # It must use the validated field rather than becoming an identity hole.
    assert not fit_support[50, 52]
    mapped_x, mapped_y = result.warp.inverse_virtual_coordinates(52.0, 50.0)
    assert abs(mapped_x - 52.0) > 0.05 or abs(mapped_y - 50.0) > 0.05


def test_local_mesh_inverse_warp_rejects_unsafe_large_or_fold_like_field() -> None:
    values = np.arange(8.0, 93.0, 4.0)
    grid_x, grid_y = np.meshgrid(values, values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source = output + np.asarray((10.0, 0.0))
    result = fit_local_mesh_inverse_warp(
        output,
        source,
        TileBounds(0.0, 0.0, 100.0, 100.0),
        config=LocalMeshWarpConfig(
            minimum_correspondences=48,
            held_out_seed=4,
            maximum_jacobian_determinant=2.0,
        ),
    )

    assert result.warp is None
    assert result.audit.reason in {
        "mesh_jacobian_determinant_out_of_bounds",
        "mesh_jacobian_condition_out_of_bounds",
        "mesh_straight_line_deviation_exceeded",
        "mesh_maximum_displacement_exceeded",
    }


def test_local_mesh_rejects_a_candidate_without_the_required_held_out_improvement_ratio() -> None:
    values = np.arange(8.0, 93.0, 4.0)
    grid_x, grid_y = np.meshgrid(values, values, indexing="xy")
    output = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    source = output + np.column_stack(
        (
            0.48 + 0.002 * (output[:, 0] - 50.0),
            -0.32 + 0.0016 * (output[:, 1] - 50.0),
        )
    )

    result = fit_local_mesh_inverse_warp(
        output,
        source,
        TileBounds(0.0, 0.0, 100.0, 100.0),
        config=LocalMeshWarpConfig(
                grid_spacing_pixels=32,
                minimum_correspondences=48,
                held_out_seed=9,
                maximum_held_out_error_pixels=1.5,
                minimum_held_out_improvement_ratio=0.99,
            ),
    )

    assert result.warp is None
    assert result.audit.reason == "mesh_held_out_improvement_ratio_too_small"


def test_geometry_pair_rejects_non_se3_pose() -> None:
    camera = _intrinsics(7, 5)
    depth = np.full((5, 7), 1000.0, dtype=np.float32)
    bad_pose = _pose()
    bad_pose[0, 0] = 2.0

    with pytest.raises(ValueError, match="camera_to_world"):
        analyze_adjacent_rgbd_pair(depth, depth, camera, bad_pose, _pose())
