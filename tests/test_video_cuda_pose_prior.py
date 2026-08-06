from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_pose_prior import cuda_pose_inverse_grid_from_target_depth


def _torch():
    return importlib.import_module("torch")


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_orb_rgbd_inverse_grid_prior_uses_real_relative_pose_and_depth_gate():
    torch = _torch()
    device = torch.device("cuda:0")
    height = width = 16
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    target_grid = torch.stack((2.0 * xx / float(width - 1) - 1.0, 2.0 * yy / float(height - 1) - 1.0), dim=-1)
    source_depth = torch.full((height, width), 1000.0, device=device)
    target_depth = torch.full((height, width), 1000.0, device=device)
    source_pose = torch.eye(4, device=device)
    target_pose = torch.eye(4, device=device)
    # Target camera is 10 mm right of source.  At 1000 mm with fx=100 this
    # maps each target pixel to source x + 1; it is an actual SE(3) prior, not
    # a layout-only scalar shift.
    target_pose[0, 3] = 10.0

    result = cuda_pose_inverse_grid_from_target_depth(
        torch,
        target_inverse_grid_xy=target_grid,
        source_depth_mm=source_depth,
        target_depth_mm=target_depth,
        source_camera_to_world=source_pose,
        target_camera_to_world=target_pose,
        fx=100.0,
        fy=100.0,
        cx=0.0,
        cy=0.0,
    )

    mapped_x = (result.inverse_grid_xy[..., 0] + 1.0) * float(width - 1) * 0.5
    assert result.inverse_grid_xy.device.type == "cuda"
    assert float(mapped_x[8, 8].item()) == pytest.approx(9.0, abs=1.0e-4)
    assert bool(result.safe_mask[8, 8].item()) is True
    # The right border would map outside the real source and must not be used.
    assert bool(result.safe_mask[8, -1].item()) is False
    assert result.audit["uses_real_orb_camera_to_world"] is True
    assert result.audit["uses_real_aligned_depth"] is True
    assert result.audit["creates_panorama_depth"] is False
    assert result.audit["host_transfer_count"] == 0
