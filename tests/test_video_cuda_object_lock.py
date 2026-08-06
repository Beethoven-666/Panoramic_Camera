from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_object_lock import cuda_depth_object_protection, lock_cuda_protected_owner


def _torch():
    return importlib.import_module("torch")


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_object_lock_only_pins_an_aligned_depth_protected_domain_with_one_genuine_owner():
    torch = _torch()
    device = torch.device("cuda:0")
    depth = torch.full((12, 16), 700.0, device=device)
    # A genuine depth discontinuity, not a semantic/manual mask, creates the
    # sole C5 protection domain.
    depth[:, 8:] = 1100.0
    protected, audit = cuda_depth_object_protection(
        torch, first_depth_mm=depth, second_depth_mm=depth, depth_edge_guard_pixels=1
    )
    first = torch.ones_like(protected)
    second = torch.ones_like(protected)
    owner = torch.where(torch.arange(16, device=device)[None, :] < 8, 10, 20).expand(12, -1).to(torch.int32).clone()

    locked = lock_cuda_protected_owner(
        torch, owner_frame_id=owner, first_valid_mask=first, second_valid_mask=second,
        protected_mask=protected, first_frame_id=10, second_frame_id=20, preferred_owner_frame_id=20,
    )

    assert audit["host_transfer_count"] == 0
    assert audit["object_protected_pixel_count"] == 0
    assert audit["depth_edge_protected_pixel_count"] > 0
    assert audit["protection_input"] == "aligned_depth_only"
    assert locked.audit["accepted"] is True
    assert bool(torch.all(locked.owner_frame_id[protected] == 20).item())
    assert locked.audit["host_transfer_count"] == 0


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_object_lock_rejects_split_protection_without_one_source_coverage():
    torch = _torch()
    device = torch.device("cuda:0")
    valid = torch.ones((8, 12), dtype=torch.bool, device=device)
    first, second = valid.clone(), valid.clone()
    first[:, 8:] = False
    second[:, :4] = False
    protected = valid.clone()
    owner = torch.where(torch.arange(12, device=device)[None, :] < 6, 10, 20).expand(8, -1).to(torch.int32).clone()

    locked = lock_cuda_protected_owner(
        torch, owner_frame_id=owner, first_valid_mask=first, second_valid_mask=second,
        protected_mask=protected, first_frame_id=10, second_frame_id=20,
    )

    assert locked.audit["accepted"] is False
    assert locked.audit["reason"] == "no_single_genuine_source_covers_protection"
