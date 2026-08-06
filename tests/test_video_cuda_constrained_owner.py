from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_constrained_owner import (
    CudaConstrainedOwnerError,
    constrained_curved_hard_owner,
)


def _torch():
    return importlib.import_module("torch")


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_c1_curved_owner_keeps_single_real_owner_and_protection():
    torch = _torch()
    device = torch.device("cuda:0")
    height, width = 12, 24
    cost = torch.ones((height, width), device=device)
    # Prefer a gently moving seam, still bounded by the configured step.
    for row in range(height):
        cost[row, 9 + row // 3] = 0.0
    first = torch.ones((height, width), dtype=torch.bool, device=device)
    second = torch.ones((height, width), dtype=torch.bool, device=device)
    protected = torch.zeros((height, width), dtype=torch.bool, device=device)
    protected[:, 7] = True
    protected_owner = torch.full((height, width), 10, dtype=torch.int32, device=device)

    result = constrained_curved_hard_owner(
        torch,
        seam_cost=cost,
        first_valid_mask=first,
        second_valid_mask=second,
        first_frame_id=10,
        second_frame_id=20,
        corridor_x=(4, 20),
        protected_mask=protected,
        protected_owner_frame_id=protected_owner,
        maximum_row_step_pixels=2,
    )

    assert result.owner_frame_id.device.type == "cuda"
    assert result.seam_x_by_row.device.type == "cuda"
    assert bool(torch.all((result.owner_frame_id == 10) | (result.owner_frame_id == 20)).item())
    assert bool(torch.all(result.owner_frame_id[:, 7] == 10).item())
    assert result.audit["strict_single_owner"] is True
    assert result.audit["maximum_observed_row_step_pixels"] <= 2
    assert result.audit["host_transfer_count"] == 0


def test_cuda_c1_rejects_non_cuda_cost_tensor():
    torch = _torch()
    cost = torch.ones((8, 16))
    valid = torch.ones((8, 16), dtype=torch.bool)

    with pytest.raises(CudaConstrainedOwnerError, match="CUDA-resident"):
        constrained_curved_hard_owner(
            torch,
            seam_cost=cost,
            first_valid_mask=valid,
            second_valid_mask=valid,
            first_frame_id=1,
            second_frame_id=2,
            corridor_x=(4, 12),
        )
