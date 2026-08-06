from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_mesh import fit_cuda_local_mesh


def _torch():
    return importlib.import_module("torch")


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_mesh_uses_disjoint_held_out_support_and_zeroes_boundaries():
    torch = _torch()
    device = torch.device("cuda:0")
    flow = torch.zeros((16, 16, 2), device=device)
    flow[2:-2, 2:-2, 0] = 0.5
    safe = torch.zeros((16, 16), dtype=torch.bool, device=device)
    safe[2:-2, 2:-2] = True
    train = safe.clone()
    train[::2] = False
    held_out = safe & ~train
    protected = torch.zeros_like(safe)

    result = fit_cuda_local_mesh(
        torch,
        flow_xy=flow,
        training_mask=train,
        held_out_mask=held_out,
        safe_mask=safe,
        protected_mask=protected,
    )

    assert result.offset_xy.device.type == "cuda"
    assert result.audit["train_held_out_disjoint"] is True
    assert result.audit["boundary_identity_maximum_error_px"] == 0.0
    assert result.audit["host_transfer_count"] == 0
