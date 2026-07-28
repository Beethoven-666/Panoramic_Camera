"""Fail-closed smoke test for the project's Open3D CUDA wheel."""

from __future__ import annotations

import gc
import importlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

cuda_backend = importlib.import_module("panorama_demo.cuda_backend")
cuda_backend.configure_cuda_dll_search_path()

o3d = importlib.import_module("open3d")


def main() -> None:
    build_config = dict(getattr(o3d, "_build_config", {}))
    if not build_config.get("BUILD_CUDA_MODULE", False):
        raise RuntimeError("Open3D was built without BUILD_CUDA_MODULE")
    if not o3d.core.cuda.is_available():
        raise RuntimeError("Open3D CUDA runtime/device probe failed")

    device = o3d.core.Device("CUDA:0")
    probe = o3d.core.Tensor(
        [2.0, 3.0], dtype=o3d.core.float32, device=device
    )
    squared = (probe * probe).cpu().numpy()
    np.testing.assert_array_equal(squared, np.array([4.0, 9.0], np.float32))

    odometry_height, odometry_width = 120, 160
    yy, xx = np.mgrid[:odometry_height, :odometry_width]
    odometry_color = np.stack(
        (
            (xx * 3 + yy) % 256,
            (yy * 5 + xx) % 256,
            (xx * 7 + yy * 11) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    odometry_depth = (
        900.0 + 0.8 * xx + 0.5 * yy + 30.0 * np.sin(xx / 13.0)
    ).astype(np.float32)
    odometry_rgbd = o3d.t.geometry.RGBDImage(
        o3d.t.geometry.Image(
            o3d.core.Tensor(odometry_color, device=device)
        ),
        o3d.t.geometry.Image(
            o3d.core.Tensor(odometry_depth, device=device)
        ),
    )
    odometry_intrinsic = o3d.core.Tensor(
        [
            [140.0, 0.0, odometry_width / 2],
            [0.0, 140.0, odometry_height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=o3d.core.float64,
        device=device,
    )
    odometry_identity = o3d.core.Tensor.eye(
        4, dtype=o3d.core.float64, device=device
    )
    odometry_criteria = [
        o3d.t.pipelines.odometry.OdometryConvergenceCriteria(2)
        for _ in range(3)
    ]
    odometry = o3d.t.pipelines.odometry.rgbd_odometry_multi_scale(
        odometry_rgbd,
        odometry_rgbd,
        odometry_intrinsic,
        odometry_identity,
        1000.0,
        3.0,
        odometry_criteria,
        o3d.t.pipelines.odometry.Method.Hybrid,
    )
    np.testing.assert_allclose(
        odometry.transformation.numpy(), np.eye(4), atol=1e-6
    )
    if float(odometry.fitness) < 0.99:
        raise RuntimeError("Open3D Tensor CUDA RGB-D odometry smoke test failed")

    height, width = 48, 64
    depth = np.full((height, width), 1000, dtype=np.uint16)
    color = np.zeros((height, width, 3), dtype=np.uint8)
    color[..., 1] = 192
    depth_image = o3d.t.geometry.Image(o3d.core.Tensor(depth, device=device))
    color_image = o3d.t.geometry.Image(o3d.core.Tensor(color, device=device))
    intrinsic = o3d.core.Tensor(
        [[60.0, 0.0, width / 2], [0.0, 60.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=o3d.core.float64,
    )
    extrinsic = o3d.core.Tensor.eye(4, dtype=o3d.core.float64)
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(
            o3d.core.float32,
            o3d.core.float32,
            o3d.core.float32,
        ),
        attr_channels=((1,), (1,), (3,)),
        voxel_size=0.02,
        block_resolution=16,
        block_count=1000,
        device=device,
    )
    blocks = volume.compute_unique_block_coordinates(
        depth_image,
        intrinsic,
        extrinsic,
        depth_scale=1000.0,
        depth_max=3.0,
        trunc_voxel_multiplier=4.0,
    )
    # Two identical observations make the project's weight_threshold=1.0
    # extraction contract observable in this minimal synthetic scene.
    for _ in range(2):
        volume.integrate(
            blocks,
            depth_image,
            color_image,
            intrinsic,
            extrinsic,
            depth_scale=1000.0,
            depth_max=3.0,
            trunc_voxel_multiplier=4.0,
        )
    o3d.core.cuda.synchronize()
    active_blocks = int(volume.hashmap().size())
    mesh = volume.extract_triangle_mesh(weight_threshold=1.0)
    mesh_vertices = int(mesh.vertex.positions.shape[0])
    mesh_triangles = int(mesh.triangle.indices.shape[0])
    if active_blocks <= 0 or mesh_vertices <= 0:
        raise RuntimeError("Open3D CUDA VoxelBlockGrid smoke test was empty")
    odometry_fitness = float(odometry.fitness)
    odometry_rmse = float(odometry.inlier_rmse)
    del (
        mesh,
        volume,
        blocks,
        depth_image,
        color_image,
        odometry,
        odometry_rgbd,
        probe,
    )
    gc.collect()
    o3d.core.cuda.synchronize()
    o3d.core.cuda.release_cache()

    print(
        json.dumps(
            {
                "open3d_version": o3d.__version__,
                "build_cuda_module": True,
                "cuda_available": True,
                "device_count": int(o3d.core.cuda.device_count()),
                "device": str(device),
                "odometry_backend": "open3d_tensor_cuda_rgbd",
                "odometry_fitness": odometry_fitness,
                "odometry_rmse_m": odometry_rmse,
                "active_blocks": active_blocks,
                "mesh_vertices": mesh_vertices,
                "mesh_triangles": mesh_triangles,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
