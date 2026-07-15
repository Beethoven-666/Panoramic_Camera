from __future__ import annotations

import json
from pathlib import Path
import struct

import cv2
import numpy as np

from panorama_demo.dense_fusion import (
    DenseFusionConfig,
    _crop_dense_result,
    _depth_point_foreground_overlay,
    _mesh_to_glb,
    _supported_depth_mask,
    _tsdf_local_holes,
)
from panorama_demo.rgbd_projection import PinholeIntrinsics, ProjectionCanvas
from panorama_demo.session import RGBDFrame


def _write_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    assert success
    path.write_bytes(encoded.tobytes())


def _canvas() -> ProjectionCanvas:
    return ProjectionCanvas(
        width=20,
        height=20,
        world_bounds=(-10.0, -10.0, 10.0, 10.0),
        pixels_per_mm=1.0,
        scan_axis=(1.0, 0.0, 0.0),
        up_axis=(0.0, -1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        maximum_depth_mm=1000.0,
        source_count=2,
        canvas_megapixels=0.001,
        aggregate_megapixels=0.002,
    )


def _frame(tmp_path: Path, frame_id: int, color: tuple[int, int, int], depth: int) -> RGBDFrame:
    color_path = tmp_path / f"color_{frame_id}.png"
    depth_path = tmp_path / f"depth_{frame_id}.png"
    _write_png(color_path, np.full((3, 3, 3), color, dtype=np.uint8))
    _write_png(depth_path, np.full((3, 3), depth, dtype=np.uint16))
    return RGBDFrame(
        frame_id=frame_id,
        color_path=color_path,
        aligned_depth_path=depth_path,
        depth_scale_mm_per_unit=1.0,
        timestamp_us=frame_id + 1,
    )


def test_depth_point_foreground_zbuffer_prefers_camera_side_measurement(
    tmp_path: Path,
) -> None:
    far = _frame(tmp_path, 1, (0, 0, 255), 20)
    near = _frame(tmp_path, 2, (0, 255, 0), 10)
    intrinsics = PinholeIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0)
    config = DenseFusionConfig(
        foreground_offset_mm=20.0,
        foreground_neighbor_depth_tolerance_mm=5.0,
    )
    identity = np.eye(4, dtype=np.float64)

    image, mask, _, audit = _depth_point_foreground_overlay(
        (far, near),
        (identity, identity),
        intrinsics,
        _canvas(),
        100.0,
        1.0,
        None,
        config,
    )

    assert audit["foreground_zbuffer_pixel_count"] > 0
    assert mask[10, 10]
    # BGR green from the nearer real depth point wins over the red point.
    assert image[10, 10].tolist() == [0, 255, 0]


def test_supported_depth_mask_removes_isolated_depth_fly_point() -> None:
    depth = np.full((5, 5), 500.0, dtype=np.float32)
    depth[2, 2] = 100.0
    valid = np.ones(depth.shape, dtype=bool)

    supported = _supported_depth_mask(
        depth, valid, tolerance_mm=20.0, minimum_neighbors=1
    )

    assert not supported[2, 2]
    assert supported[0, 0]


def test_safe_crop_uses_visibility_mask_not_black_rgb_content() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    visible = np.zeros((8, 10), dtype=bool)
    visible[1:7, 1:9] = True
    foreground = np.zeros_like(visible)
    foreground[3, 4] = True
    config = DenseFusionConfig(mask_close_kernel_size=1, mask_erode_pixels=0)

    cropped, cropped_visible, cropped_foreground, crop, metadata = _crop_dense_result(
        image, visible, foreground, config=config
    )

    assert crop == (1, 1, 8, 6)
    assert cropped.shape == (6, 8, 3)
    assert cropped_visible.all()
    assert cropped_foreground[2, 3]
    assert metadata["strategy"] == "maximum_safe_inscribed_rectangle"


def test_depth_fallback_is_limited_to_holes_enclosed_by_tsdf_geometry() -> None:
    tsdf = np.zeros((11, 11), dtype=bool)
    tsdf[2:9, 2:9] = True
    tsdf[4:7, 4:7] = False

    holes = _tsdf_local_holes(tsdf, 5)

    assert holes[5, 5]
    assert not holes[0, 0]
    assert not holes[1, 5]


class _TriangleMesh:
    vertices = np.array(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    triangles = np.array(((0, 1, 2),), dtype=np.int32)
    vertex_colors = np.array(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    vertex_normals = np.array(
        ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )

    def has_vertex_normals(self) -> bool:
        return True

    def compute_vertex_normals(self) -> None:
        raise AssertionError("The test mesh already provides normals")


def test_tsdf_mesh_export_is_a_standard_coloured_glb() -> None:
    glb = _mesh_to_glb(_TriangleMesh())

    magic, version, total_length = struct.unpack("<4sII", glb[:12])
    json_length, json_type = struct.unpack("<II", glb[12:20])
    payload = json.loads(glb[20 : 20 + json_length])

    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(glb)
    assert json_type == 0x4E4F534A
    assert payload["asset"]["version"] == "2.0"
    assert payload["nodes"][0]["rotation"] == [1.0, 0.0, 0.0, 0.0]
    primitive = payload["meshes"][0]["primitives"][0]
    assert primitive["attributes"] == {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}
    assert primitive["indices"] == 3
