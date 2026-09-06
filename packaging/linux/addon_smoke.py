"""Small synthetic CUDA TSDF integration; not physical or H0 qualification."""
import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert o3d.__version__ == "0.19.0+1e7b17438"
    assert o3d.core.cuda.is_available()
    device = o3d.core.Device("CUDA:0")
    volume = o3d.t.geometry.VoxelBlockGrid(attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32,) * 3, attr_channels=((1,), (1,), (3,)),
        voxel_size=0.02, block_resolution=8, block_count=1024, device=device)
    intrinsic = o3d.core.Tensor([[50.,0,31.5],[0,50.,23.5],[0,0,1]], dtype=o3d.core.float64)
    extrinsic = o3d.core.Tensor(np.eye(4), dtype=o3d.core.float64)
    depth = o3d.t.geometry.Image(o3d.core.Tensor(np.full((48,64),1000,np.uint16), device=device))
    color = o3d.t.geometry.Image(o3d.core.Tensor(np.full((48,64,3),128,np.uint8), device=device))
    blocks = volume.compute_unique_block_coordinates(depth, intrinsic, extrinsic,
        depth_scale=1000., depth_max=3., trunc_voxel_multiplier=4.)
    for _ in range(3):
        volume.integrate(blocks, depth, color, intrinsic, extrinsic,
            depth_scale=1000., depth_max=3., trunc_voxel_multiplier=4.)
    mesh = volume.extract_triangle_mesh(weight_threshold=1.).to_legacy()
    assert len(mesh.vertices) >= 3 and len(mesh.triangles) >= 1
    report = {"status":"PASS", "open3d_version":o3d.__version__, "backend":"open3d_tensor_cuda_tsdf",
              "vertices":len(mesh.vertices), "triangles":len(mesh.triangles),
              "software_ready":False,"hardware_qualified":False,"release_ready":False}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
