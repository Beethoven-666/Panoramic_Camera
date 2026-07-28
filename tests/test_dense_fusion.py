from __future__ import annotations

import json
from pathlib import Path
import struct
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import panorama_demo.dense_fusion as dense_fusion
from panorama_demo.dense_fusion import (
    DenseFusionConfig,
    DisplayOnlyTSDFUniqueBlockEstimate,
    _crop_dense_result,
    _depth_point_foreground_overlay,
    _integrate_tsdf_tensor_cuda,
    _integrate_tsdf_with_audit,
    _mesh_to_glb,
    _supported_depth_mask,
    _tsdf_local_holes,
    estimate_display_only_tsdf_unique_blocks,
    export_tsdf_mesh,
    plan_display_only_tsdf_cuda_capacity,
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


def test_cuda_capacity_plan_preserves_default_five_mm_short_scan() -> None:
    plan = plan_display_only_tsdf_cuda_capacity(
        target_gpu_byte_budget=2_500_000_000,
        requested_voxel_length_mm=5.0,
        block_resolution=16,
        estimated_unique_blocks=18_000,
    )

    assert plan.planned_voxel_length_mm == 5.0
    assert plan.block_capacity == 22_500
    assert plan.raw_capacity_bytes == 22_500 * (16**3) * 20
    assert plan.capacity_utilization == 0.8
    assert plan.preflight_pass is True
    assert plan.adaptive_voxel_used is False


def test_cuda_capacity_plan_adapts_long_display_scan_to_eight_mm() -> None:
    plan = plan_display_only_tsdf_cuda_capacity(
        target_gpu_byte_budget=2_500_000_000,
        requested_voxel_length_mm=5.0,
        block_resolution=16,
        estimated_unique_blocks=60_000,
    )

    assert plan.planned_voxel_length_mm == 8.0
    assert plan.planned_estimated_unique_blocks == 23_438
    assert plan.block_capacity == 29_298
    assert plan.block_capacity <= 30_000
    assert plan.raw_capacity_bytes <= plan.target_gpu_byte_budget
    assert plan.capacity_utilization <= plan.maximum_capacity_utilization
    assert plan.preflight_pass is True
    assert plan.adaptive_voxel_used is True
    assert plan.display_only is True
    assert plan.participates_in_panorama is False
    assert plan.rgb_owner_feedback_permitted is False
    assert plan.cpu_fallback_permitted is False


def test_cuda_capacity_plan_fails_gate_instead_of_overcommitting() -> None:
    plan = plan_display_only_tsdf_cuda_capacity(
        target_gpu_byte_budget=2_500_000_000,
        requested_voxel_length_mm=5.0,
        block_resolution=16,
        estimated_unique_blocks=100_000,
    )

    assert plan.planned_voxel_length_mm == 8.0
    assert plan.block_capacity == 30_000
    assert plan.raw_capacity_bytes == 30_000 * (16**3) * 20
    assert plan.utilization_gate_pass is False
    assert plan.preflight_pass is False
    assert plan.cpu_fallback_permitted is False


def test_cuda_capacity_plan_validates_budget_and_resolution() -> None:
    with np.testing.assert_raises_regex(ValueError, "cannot hold one"):
        plan_display_only_tsdf_cuda_capacity(
            target_gpu_byte_budget=1,
            requested_voxel_length_mm=5.0,
            block_resolution=16,
            estimated_unique_blocks=1,
        )
    with np.testing.assert_raises_regex(ValueError, "positive integer"):
        plan_display_only_tsdf_cuda_capacity(
            target_gpu_byte_budget=2_500_000_000,
            requested_voxel_length_mm=5.0,
            block_resolution=0,
            estimated_unique_blocks=1,
        )


def test_display_tsdf_prefer_does_not_silently_fallback_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_open3d = SimpleNamespace(
        _build_config={"BUILD_CUDA_MODULE": False},
        core=SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setattr(dense_fusion, "_require_open3d", lambda: fake_open3d)
    monkeypatch.setattr(
        dense_fusion,
        "cuda_status",
        lambda: SimpleNamespace(mode="prefer"),
    )

    def forbidden_cpu(*args: object, **kwargs: object) -> object:
        raise AssertionError("implicit CPU TSDF fallback was invoked")

    monkeypatch.setattr(
        dense_fusion,
        "_integrate_tsdf_legacy_cpu",
        forbidden_cpu,
    )
    intrinsics = PinholeIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0)
    with pytest.raises(RuntimeError, match="implicit CPU fallback is forbidden"):
        _integrate_tsdf_with_audit(
            [object()],
            [np.eye(4, dtype=np.float64)],
            intrinsics,
            None,
            DenseFusionConfig(),
        )


def test_unique_block_estimator_samples_three_world_layers_read_only(
    tmp_path: Path,
) -> None:
    frame = _frame(tmp_path, 5, (10, 20, 30), 1000)
    intrinsics = PinholeIntrinsics(3, 3, 1000.0, 1000.0, 0.0, 0.0)

    estimate = estimate_display_only_tsdf_unique_blocks(
        [frame],
        [np.eye(4, dtype=np.float64)],
        intrinsics,
        None,
        voxel_length_mm=5.0,
        block_resolution=16,
        sdf_truncation_mm=100.0,
        maximum_depth_mm=2000.0,
        depth_stride=8,
        safety_factor=1.85,
    )

    assert estimate.layer_offsets_mm == (-100.0, 0.0, 100.0)
    assert estimate.sampled_valid_depth_pixel_count == 1
    assert estimate.sampled_layer_coordinate_count == 3
    assert estimate.raw_unique_block_count == 3
    assert estimate.safety_estimated_unique_blocks == 6
    assert estimate.read_only is True
    assert estimate.participates_in_panorama is False
    assert estimate.rgb_owner_feedback_permitted is False


def test_capacity_plan_uses_measured_eight_mm_estimate() -> None:
    plan = plan_display_only_tsdf_cuda_capacity(
        target_gpu_byte_budget=2_500_000_000,
        requested_voxel_length_mm=5.0,
        block_resolution=16,
        estimated_unique_blocks=51_551,
        estimated_unique_blocks_at_maximum_voxel=21_177,
        maximum_block_capacity=30_000,
        maximum_adaptive_voxel_length_mm=8.0,
    )

    assert plan.planned_voxel_length_mm == 8.0
    assert plan.planned_estimated_unique_blocks == 21_177
    assert plan.block_capacity == 26_472
    assert plan.preflight_pass is True


def test_failed_capacity_plan_stops_before_open3d_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_plan = plan_display_only_tsdf_cuda_capacity(
        target_gpu_byte_budget=2_500_000_000,
        requested_voxel_length_mm=5.0,
        block_resolution=16,
        estimated_unique_blocks=100_000,
    )

    def forbidden_open3d() -> object:
        raise AssertionError("Open3D was loaded before capacity preflight")

    monkeypatch.setattr(dense_fusion, "_require_open3d", forbidden_open3d)
    with pytest.raises(MemoryError, match="before VoxelBlockGrid allocation"):
        _integrate_tsdf_with_audit(
            [object()],
            [np.eye(4, dtype=np.float64)],
            PinholeIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0),
            None,
            DenseFusionConfig(voxel_length_mm=8.0),
            cuda_capacity_plan=failed_plan,
        )


def test_cuda_integrator_allocates_planned_voxel_and_capacity(
    tmp_path: Path,
) -> None:
    zero_depth_frame = _frame(tmp_path, 6, (10, 20, 30), 0)
    plan = plan_display_only_tsdf_cuda_capacity(
        target_gpu_byte_budget=2_500_000_000,
        requested_voxel_length_mm=5.0,
        block_resolution=16,
        estimated_unique_blocks=18_000,
    )
    allocation: dict[str, object] = {}

    class FakeVolume:
        def __init__(self, **kwargs: object) -> None:
            allocation.update(kwargs)

        def hashmap(self) -> object:
            return SimpleNamespace(size=lambda: 14_046)

        def extract_triangle_mesh(self, **kwargs: object) -> object:
            return SimpleNamespace(to_legacy=lambda: _TriangleMesh())

    fake_open3d = SimpleNamespace(
        core=SimpleNamespace(
            Device=lambda value: value,
            Tensor=lambda value, **kwargs: np.asarray(value),
            float32="float32",
            float64="float64",
            cuda=SimpleNamespace(
                synchronize=lambda: None,
                release_cache=lambda: None,
            ),
        ),
        t=SimpleNamespace(
            geometry=SimpleNamespace(
                VoxelBlockGrid=lambda **kwargs: FakeVolume(**kwargs),
            )
        ),
    )
    _, audit = _integrate_tsdf_tensor_cuda(
        fake_open3d,
        [zero_depth_frame],
        [np.eye(4, dtype=np.float64)],
        PinholeIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0),
        None,
        DenseFusionConfig(voxel_length_mm=5.0),
        capacity_plan=plan,
    )

    assert allocation["voxel_size"] == 0.005
    assert allocation["block_resolution"] == 16
    assert allocation["block_count"] == 22_500
    assert audit["active_block_count"] == 14_046
    assert audit["capacity_fraction"] == pytest.approx(14_046 / 22_500)


def test_formal_export_audits_estimate_and_capacity_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimates: list[float] = []

    def fake_estimate(
        frames: object,
        poses: object,
        intrinsics: object,
        maps: object,
        *,
        voxel_length_mm: float,
        block_resolution: int,
        sdf_truncation_mm: float,
        maximum_depth_mm: float,
        depth_stride: int,
        safety_factor: float,
    ) -> DisplayOnlyTSDFUniqueBlockEstimate:
        estimates.append(voxel_length_mm)
        raw = 8632 if voxel_length_mm == 5.0 else 11447
        return DisplayOnlyTSDFUniqueBlockEstimate(
            voxel_length_mm=voxel_length_mm,
            block_resolution=block_resolution,
            block_side_length_mm=voxel_length_mm * block_resolution,
            depth_stride=depth_stride,
            sdf_truncation_mm=sdf_truncation_mm,
            layer_offsets_mm=(
                -sdf_truncation_mm,
                0.0,
                sdf_truncation_mm,
            ),
            frame_count=1,
            frames_with_valid_depth=1,
            sampled_valid_depth_pixel_count=100,
            sampled_layer_coordinate_count=300,
            per_frame_unique_block_sum=raw,
            raw_unique_block_count=raw,
            safety_factor=safety_factor,
            safety_estimated_unique_blocks=int(np.ceil(raw * safety_factor)),
        )

    received: dict[str, object] = {}

    def fake_integrate(*args: object, **kwargs: object) -> tuple[object, dict]:
        received.update(kwargs)
        return _TriangleMesh(), {"backend": "fake_cuda"}

    monkeypatch.setattr(
        dense_fusion,
        "cuda_status",
        lambda: SimpleNamespace(mode="prefer"),
    )
    monkeypatch.setattr(
        dense_fusion,
        "estimate_display_only_tsdf_unique_blocks",
        fake_estimate,
    )
    monkeypatch.setattr(
        dense_fusion,
        "_integrate_tsdf_with_audit",
        fake_integrate,
    )
    _, audit = export_tsdf_mesh(
        [object()],
        [np.eye(4, dtype=np.float64)],
        PinholeIntrinsics(3, 3, 1.0, 1.0, 1.0, 1.0),
    )

    assert estimates == [5.0]
    plan = received["cuda_capacity_plan"]
    assert plan.planned_voxel_length_mm == 5.0
    assert plan.block_capacity == 19_963
    assert audit["capacity_preflight"]["preflight_pass"] is True
    assert audit["unique_block_estimate"]["requested_voxel"][
        "raw_unique_block_count"
    ] == 8632
    assert audit["configuration"]["requested_voxel_length_mm"] == 5.0
    assert audit["configuration"]["planned_voxel_length_mm"] == 5.0
    assert audit["display_only"] is True
    assert audit["participates_in_panorama"] is False
