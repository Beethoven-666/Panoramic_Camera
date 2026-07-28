from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from panorama_demo.inspection_multiview import (
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
)
from panorama_demo.inspection_world_coverage import (
    InspectionWorldCoverageConfig,
    audit_inspection_world_coverage,
)
from panorama_demo.session import CameraIntrinsics, RGBDFrame


def _frame(tmp_path: Path, frame_id: int, depth_mm: int) -> RGBDFrame:
    depth_path = tmp_path / f"depth_{frame_id}.png"
    assert cv2.imwrite(
        str(depth_path), np.full((6, 8), depth_mm, dtype=np.uint16)
    )
    return RGBDFrame(
        frame_id=frame_id,
        color_path=tmp_path / f"unused_{frame_id}.png",
        aligned_depth_path=depth_path,
        depth_scale_mm_per_unit=1.0,
    )


def _layout() -> InspectionMultiviewLayout:
    panels = tuple(
        VirtualPerspectivePanel(
            panel_index=index,
            anchor_scan_mm=0.0,
            canvas_offset_x=0.0,
            center_world_mm=(0.0, 0.0, 0.0),
        )
        for index in range(2)
    )
    return InspectionMultiviewLayout(
        width=8,
        height=6,
        reference_depth_mm=1000.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=panels,
        panel_step_mm=0.0,
        canvas_megapixels=48 / 1_000_000.0,
    )


def _audit(
    frames: list[RGBDFrame], owner: np.ndarray
) -> dict[str, object]:
    intrinsics = CameraIntrinsics(
        width=8,
        height=6,
        fx=100.0,
        fy=100.0,
        cx=3.5,
        cy=2.5,
        distortion=(0.0,) * 8,
    )
    return audit_inspection_world_coverage(
        frames=frames,
        poses=[np.eye(4, dtype=np.float64) for _ in frames],
        intrinsics=intrinsics,
        layout=_layout(),
        owner_frame_id=owner,
        crop_xywh=(0, 0, 8, 6),
        selected_panel_sources=[
            {"panel_index": 0, "source_position": 0, "frame_id": 10},
            {"panel_index": 1, "source_position": 1, "frame_id": 11},
        ],
        config=InspectionWorldCoverageConfig(
            sample_stride=1,
            voxel_size_mm=4.0,
            match_radius_voxels=1,
            minimum_cell_voxels=1,
        ),
    )


def test_world_coverage_matches_near_surface_reached_by_rgb_owner(
    tmp_path: Path,
) -> None:
    frames = [_frame(tmp_path, 10, 600), _frame(tmp_path, 11, 600)]
    owner = np.full((6, 8), 10, dtype=np.int32)

    audit = _audit(frames, owner)

    assert audit["colour_or_geometry_mutation"] is False
    assert audit["multiview_observed_world_voxel_count"] > 0
    assert audit["multiview_world_coverage_ratio"] == 1.0


def test_world_coverage_reports_near_surface_missing_from_final_owner(
    tmp_path: Path,
) -> None:
    frames = [_frame(tmp_path, 10, 1000), _frame(tmp_path, 11, 600)]
    owner = np.full((6, 8), 10, dtype=np.int32)

    audit = _audit(frames, owner)

    assert audit["observed_world_voxel_count"] > 0
    assert audit["represented_world_voxel_count"] == 0
    assert audit["observed_world_coverage_ratio"] == 0.0
