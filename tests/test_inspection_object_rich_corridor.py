from __future__ import annotations

from dataclasses import replace

import numpy as np

from panorama_demo.inspection_object_rich_corridor import (
    extract_object_rich_corridor,
    interval_pair_metrics,
    track_object_rich_corridors,
)
from panorama_demo.inspection_ocr_panel import extract_ocr_seeded_panel
from panorama_demo.session import CameraIntrinsics


def _synthetic_corridor():
    height, width = 300, 800
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    image[110:210, 60:360] = (225, 225, 225)
    image[145:170, 150:260] = (20, 210, 230)
    image[105:210, 395:515] = (55, 60, 65)
    image[145:205, 575:635] = (20, 20, 20)
    depth = np.full((height, width), 900.0, dtype=np.float32)
    depth[110:210, 60:360] = 600.0
    depth[105:210, 395:515] = 620.0
    depth[145:205, 575:635] = 610.0
    reliable = np.ones((height, width), dtype=bool)
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=600.0,
        fy=600.0,
        cx=400.0,
        cy=150.0,
        distortion=(),
    )
    ocr = np.asarray(
        [[150, 145], [260, 145], [260, 170], [150, 170]],
        dtype=np.float32,
    )
    panel, _ = extract_ocr_seeded_panel(
        frame_id=1,
        source_index=0,
        image_bgr=image,
        depth_mm=depth,
        reliable_depth=reliable,
        ocr_polygon_xy=ocr,
        camera_to_world=np.eye(4, dtype=np.float64),
        intrinsics=intrinsics,
    )
    assert panel is not None
    corridor, audit = extract_object_rich_corridor(
        panel=panel,
        image_bgr=image,
        depth_mm=depth,
        reliable_depth=reliable,
        geometric_valid=np.ones(depth.shape, dtype=bool),
        camera_to_world=np.eye(4, dtype=np.float64),
        intrinsics=intrinsics,
        reference_depth_mm=900.0,
        scan_axis_world=(1.0, 0.0, 0.0),
    )
    assert audit["pass"] is True
    assert corridor is not None
    return corridor


def test_extract_object_rich_corridor_grows_by_fixed_x_gap() -> None:
    corridor = _synthetic_corridor()
    assert len(corridor.structures) >= 2
    assert corridor.inverse_map_coverage_ratio == 1.0
    assert corridor.right_endpoint_x > corridor.panel.bbox_xywh[0]


def test_corridor_world_range_tracks_across_two_views() -> None:
    first = _synthetic_corridor()
    second = replace(
        first,
        frame_id=2,
        source_index=1,
        relative_scan_range_mm=(
            first.relative_scan_range_mm[0] + 2.0,
            first.relative_scan_range_mm[1] + 2.0,
        ),
    )
    assert track_object_rich_corridors((first, second)) == ((0, 1),)
    iou, coverage = interval_pair_metrics(
        first.relative_scan_range_mm,
        second.relative_scan_range_mm,
    )
    assert iou >= 0.50
    assert coverage >= 0.75
