from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_multilabel_owner import (
    CudaMultilabelOwnerError,
    optimise_cuda_c8_local_multilabel_owner,
)


def _torch():
    return importlib.import_module("torch")


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_c8_owner_is_device_resident_monotone_and_preserves_real_lock() -> None:
    torch = _torch()
    device = torch.device("cuda:0")
    source_count, height, width = 3, 5, 15
    valid = torch.ones((source_count, height, width), dtype=torch.bool, device=device)
    cost = torch.full((source_count, height, width), 5.0, device=device)
    cost[0, :, :5] = 0.0
    cost[1, :, 5:10] = 0.0
    cost[2, :, 10:] = 0.0
    protected = torch.full((height, width), -1, dtype=torch.int32, device=device)
    protected[:, 7] = 20

    result = optimise_cuda_c8_local_multilabel_owner(
        torch,
        unary_cost=cost,
        source_valid_mask=valid,
        source_frame_ids=(10, 20, 30),
        protected_owner_frame_id=protected,
    )

    assert result.owner_frame_id.device.type == "cuda"
    assert result.valid_mask.device.type == "cuda"
    assert bool(torch.all(result.owner_frame_id[:, 7] == 20).item())
    owner_index = torch.where(
        result.owner_frame_id == 10,
        0,
        torch.where(result.owner_frame_id == 20, 1, 2),
    )
    assert bool(torch.all(owner_index[:, 1:] >= owner_index[:, :-1]).item())
    assert result.audit["strict_single_owner"] is True
    assert result.audit["temporal_order_constraint"] is True
    assert result.audit["host_transfer_count"] == 0
    assert result.audit["creates_colour"] is False
    assert result.audit["creates_pose"] is False


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_c8_invalid_gap_resets_order_and_backwards_locks_fail_closed() -> None:
    torch = _torch()
    device = torch.device("cuda:0")
    valid = torch.ones((2, 2, 8), dtype=torch.bool, device=device)
    valid[:, :, 3:5] = False
    cost = torch.zeros((2, 2, 8), device=device)
    protected = torch.full((2, 8), -1, dtype=torch.int32, device=device)
    protected[:, 1] = 20
    protected[:, 6] = 10
    # The locks would be backwards if they were in one island, but the source
    # support gap gives each real local island its own temporal chain.
    result = optimise_cuda_c8_local_multilabel_owner(
        torch,
        unary_cost=cost,
        source_valid_mask=valid,
        source_frame_ids=(10, 20),
        protected_owner_frame_id=protected,
    )
    assert bool(torch.all(result.owner_frame_id[:, 3:5] == -1).item())
    assert bool(torch.all(result.owner_frame_id[:, 1] == 20).item())
    assert bool(torch.all(result.owner_frame_id[:, 6] == 10).item())

    impossible = torch.full((2, 8), -1, dtype=torch.int32, device=device)
    impossible[:, 1] = 20
    impossible[:, 2] = 10
    with pytest.raises(CudaMultilabelOwnerError, match="temporally/order infeasible"):
        optimise_cuda_c8_local_multilabel_owner(
            torch,
            unary_cost=torch.zeros((2, 2, 8), device=device),
            source_valid_mask=torch.ones((2, 2, 8), dtype=torch.bool, device=device),
            source_frame_ids=(10, 20),
            protected_owner_frame_id=impossible,
        )


def test_cuda_c8_rejects_cpu_or_nonchronological_input() -> None:
    torch = _torch()
    with pytest.raises(CudaMultilabelOwnerError, match="CUDA-resident"):
        optimise_cuda_c8_local_multilabel_owner(
            torch,
            unary_cost=torch.zeros((2, 2, 8)),
            source_valid_mask=torch.ones((2, 2, 8), dtype=torch.bool),
            source_frame_ids=(10, 20),
        )
