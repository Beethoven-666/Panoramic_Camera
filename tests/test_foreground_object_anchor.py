from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from panorama_demo.foreground_object_anchor import (
    ForegroundAnchorSource,
    _split_oversized_world_cluster,
    plan_foreground_object_anchors,
)
from panorama_demo.session import CameraIntrinsics


@dataclass(frozen=True)
class _Panel:
    anchor_scan_mm: float
    canvas_offset_x: float
    center_world_mm: tuple[float, float, float]


@dataclass(frozen=True)
class _Layout:
    width: int
    height: int
    reference_depth_mm: float
    scan_axis: tuple[float, float, float]
    down_axis: tuple[float, float, float]
    normal_axis: tuple[float, float, float]
    panels: tuple[_Panel, ...]


def test_oversized_world_cluster_splits_distinct_normal_layers_only() -> None:
    # The scan/down footprint is one compact object-sized region, while
    # residual depth-edge samples have connected two physically different
    # normal layers.  The layers must not become one movable RGB owner.
    world_basis = np.asarray(
        [
            [10.0, 20.0, 500.0],
            [30.0, 40.0, 520.0],
            [12.0, 22.0, 870.0],
            [32.0, 42.0, 890.0],
        ],
        dtype=np.float64,
    )

    groups = _split_oversized_world_cluster(
        np.arange(4, dtype=np.intp),
        world_basis,
    )

    assert [group.tolist() for group in groups] == [[0, 1], [2, 3]]
    assert all(
        float(np.ptp(world_basis[group, 2])) <= 300.0 for group in groups
    )


def test_compact_world_cluster_is_not_cut_by_arbitrary_bin_boundaries() -> None:
    world_basis = np.asarray(
        [
            [299.0, 299.0, 699.0],
            [301.0, 301.0, 701.0],
        ],
        dtype=np.float64,
    )
    indices = np.asarray([0, 1], dtype=np.intp)

    groups = _split_oversized_world_cluster(indices, world_basis)

    assert len(groups) == 1
    assert np.array_equal(groups[0], indices)


def test_compact_world_object_uses_two_views_and_one_selected_rgb_owner() -> None:
    width, height = 120, 90
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=100.0,
        fy=100.0,
        cx=60.0,
        cy=45.0,
        distortion=(),
    )
    camera_x = (-100.0, 0.0, 100.0)
    panels = tuple(
        _Panel(
            anchor_scan_mm=value,
            canvas_offset_x=40.0 + index * 120.0,
            center_world_mm=(value, 0.0, 0.0),
        )
        for index, value in enumerate(camera_x)
    )
    layout = _Layout(
        width=360,
        height=90,
        reference_depth_mm=1000.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=panels,
    )
    sources: list[ForegroundAnchorSource] = []
    yy, xx = np.indices((height, width), dtype=np.float32)
    for index, tx in enumerate(camera_x):
        depth = np.full((height, width), 1000.0, dtype=np.float32)
        # The same 180x108 mm planar object is observed from each translated
        # camera. Its source rectangle moves by the real perspective amount.
        center_u = intrinsics.cx - intrinsics.fx * tx / 600.0
        x0, x1 = int(round(center_u - 15)), int(round(center_u + 15))
        y0, y1 = 36, 55
        depth[y0:y1, max(0, x0):min(width, x1)] = 600.0
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.asarray(xx, dtype=np.uint8)
        image[..., 1] = np.asarray(yy * 2, dtype=np.uint8)
        image[..., 2] = 80 + index * 20
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = tx
        sources.append(
            ForegroundAnchorSource(
                source_index=index,
                panel_index=index,
                frame_id=100 + index,
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=np.ones(depth.shape, dtype=bool),
                camera_to_world=pose,
                reference_map_x=xx.copy(),
                reference_map_y=yy.copy(),
            )
        )

    plan = plan_foreground_object_anchors(
        sources,
        layout,
        intrinsics,
        minimum_component_pixels=120,
    )

    assert plan.audit["near_depth_margin_mm"] == 40.0
    assert plan.audit["canvas_boundary_margin_pixels"] == 16
    assert plan.audit["track_count"] >= 1
    anchor = plan.anchors[0]
    assert len(anchor.world_support_source_indices) >= 2
    assert len(anchor.observation_ids) >= 2
    assert anchor.selected.fit_inlier_ratio >= 0.40
    assert anchor.selected.fit_rmse_pixels <= 5.0
    assert plan.audit["tracks"][0]["world_support_source_count"] >= 2
    tracked = [
        plan.observations[index] for index in anchor.observation_ids
    ]
    target_centres = np.asarray(
        [
            (
                item.target_bbox_xywh[0] + item.target_bbox_xywh[2] * 0.5,
                item.target_bbox_xywh[1] + item.target_bbox_xywh[3] * 0.5,
            )
            for item in tracked
        ],
        dtype=np.float64,
    )
    assert np.ptp(target_centres[:, 0]) <= 2.0
    assert np.ptp(target_centres[:, 1]) <= 2.0
    assert np.count_nonzero(plan.target_mask) > 0
