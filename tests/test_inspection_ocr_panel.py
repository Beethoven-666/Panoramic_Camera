from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from panorama_demo.inspection_ocr_panel import (
    StableObjectTrackEvidence,
    audit_object_rich_interval,
    audit_relative_world_geometry,
    extract_ocr_seeded_panel,
    select_object_rich_neighbor_tracks,
    track_ocr_seeded_panels,
)
from panorama_demo.session import CameraIntrinsics


def _synthetic_panel():
    height, width = 300, 500
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    image[100:220, 70:430] = (220, 220, 220)
    image[145:175, 195:305] = (30, 210, 230)
    cv2.putText(
        image,
        "WAVE",
        (205, 168),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    depth = np.full((height, width), 900.0, dtype=np.float32)
    depth[100:220, 70:430] = 600.0
    reliable = np.ones((height, width), dtype=bool)
    ocr = np.asarray(
        [[195, 145], [305, 145], [305, 175], [195, 175]],
        dtype=np.float32,
    )
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=400.0,
        fy=400.0,
        cx=250.0,
        cy=150.0,
        distortion=(),
    )
    panel, audit = extract_ocr_seeded_panel(
        frame_id=1,
        source_index=0,
        image_bgr=image,
        depth_mm=depth,
        reliable_depth=reliable,
        ocr_polygon_xy=ocr,
        camera_to_world=np.eye(4, dtype=np.float64),
        intrinsics=intrinsics,
    )
    assert audit["pass"] is True
    assert panel is not None
    return panel


def test_extract_ocr_seeded_panel_uses_connected_white_same_layer() -> None:
    panel = _synthetic_panel()
    assert panel.rectangularity >= 0.65
    assert panel.audit["ocr_coverage_ratio"] >= 0.90
    assert panel.world_extent_pca_mm[0] >= 80.0


def test_panel_track_requires_consistent_world_structure() -> None:
    first = _synthetic_panel()
    second = replace(first, frame_id=2, source_index=1)
    inconsistent = replace(
        first,
        frame_id=3,
        source_index=2,
        world_centroid_mm=(400.0, 0.0, 600.0),
    )
    tracks = track_ocr_seeded_panels((first, second, inconsistent))
    assert tracks == ((0, 1),)


def test_object_rich_selection_does_not_depend_on_track_id() -> None:
    evidence = (
        StableObjectTrackEvidence(
            track_id=900,
            observation_count=38,
            selected_panel_observation_count=3,
            common_frame_ids=(20, 26),
            median_lab_l=70.0,
            clarity_variance=200.0,
            minimum_depth_coverage_ratio=0.95,
            adjacent_to_panel=True,
        ),
        StableObjectTrackEvidence(
            track_id=3,
            observation_count=25,
            selected_panel_observation_count=3,
            common_frame_ids=(20, 26),
            median_lab_l=55.0,
            clarity_variance=300.0,
            minimum_depth_coverage_ratio=0.96,
            adjacent_to_panel=True,
        ),
        StableObjectTrackEvidence(
            track_id=0,
            observation_count=10,
            selected_panel_observation_count=3,
            common_frame_ids=(20, 26),
            median_lab_l=40.0,
            clarity_variance=500.0,
            minimum_depth_coverage_ratio=0.98,
            adjacent_to_panel=True,
        ),
    )
    assert select_object_rich_neighbor_tracks(evidence) == (900, 3)


def test_object_rich_interval_requires_complete_contiguous_coverage() -> None:
    accepted = audit_object_rich_interval(
        projected_x_spans=((100, 650), (640, 820), (850, 920)),
        projected_in_bounds_ratios=(0.99, 0.98, 0.97),
        depth_coverage_ratios=(0.96, 0.94, 0.91),
        source_width_pixels=1280,
    )
    assert accepted["pass"] is True
    rejected = audit_object_rich_interval(
        projected_x_spans=((100, 650), (900, 1000), (1100, 1180)),
        projected_in_bounds_ratios=(0.99, 0.98, 0.97),
        depth_coverage_ratios=(0.96, 0.94, 0.91),
        source_width_pixels=1280,
    )
    assert rejected["pass"] is False


def test_relative_world_geometry_requires_two_consistent_views() -> None:
    accepted = audit_relative_world_geometry(
        {
            8: (0.0, 0.0, 0.0),
            20: (10.0, 0.0, 0.0),
        },
        {
            8: (200.0, 20.0, 0.0),
            20: (211.0, 21.0, 0.0),
        },
    )
    assert accepted["pass"] is True
    rejected = audit_relative_world_geometry(
        {
            8: (0.0, 0.0, 0.0),
            20: (10.0, 0.0, 0.0),
        },
        {
            8: (200.0, 20.0, 0.0),
            20: (410.0, 21.0, 0.0),
        },
    )
    assert rejected["pass"] is False
