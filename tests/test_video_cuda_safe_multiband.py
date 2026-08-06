from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_safe_multiband import (
    CudaSafeMultiBandError,
    blend_cuda_safe_multiband,
)


def _torch():
    return importlib.import_module("torch")


def test_cuda_safe_multiband_rejects_non_c6_bandwidth_before_any_device_work():
    with pytest.raises(CudaSafeMultiBandError, match="band_pixels"):
        blend_cuda_safe_multiband(
            _torch(),
            first_bgr=None,
            second_bgr=None,
            owner_frame_id=None,
            first_frame_id=1,
            second_frame_id=2,
            safe_background_mask=None,
            protected_mask=None,
            risk_mask=None,
            band_pixels=15,
        )


def test_cuda_safe_multiband_rejects_non_c6_pyramid_level_before_any_device_work():
    with pytest.raises(CudaSafeMultiBandError, match="levels"):
        blend_cuda_safe_multiband(
            _torch(),
            first_bgr=None,
            second_bgr=None,
            owner_frame_id=None,
            first_frame_id=1,
            second_frame_id=2,
            safe_background_mask=None,
            protected_mask=None,
            risk_mask=None,
            levels=2,
        )


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_safe_multiband_blends_only_safe_boundary_and_preserves_owner():
    torch = _torch()
    device = torch.device("cuda:0")
    first = torch.zeros((3, 12, 16), dtype=torch.uint8, device=device)
    first[0] = 20
    second = torch.zeros((3, 12, 16), dtype=torch.uint8, device=device)
    second[2] = 220
    owner = torch.full((12, 16), 7, dtype=torch.int32, device=device)
    owner[:, 8:] = 9
    safe = torch.zeros((12, 16), dtype=torch.bool, device=device)
    safe[2:10, 5:11] = True
    protected = torch.zeros_like(safe)
    risk = torch.zeros_like(safe)

    result = blend_cuda_safe_multiband(
        torch,
        first_bgr=first,
        second_bgr=second,
        owner_frame_id=owner,
        first_frame_id=7,
        second_frame_id=9,
        safe_background_mask=safe,
        protected_mask=protected,
        risk_mask=risk,
        band_pixels=16,
        levels=3,
    )

    assert result.bgr.device.type == "cuda"
    assert result.blend_mask.device.type == "cuda"
    assert result.owner_frame_id is owner
    assert torch.equal(result.owner_frame_id, owner)
    assert bool((result.blend_mask & ~safe).any().item()) is False
    assert bool(result.blend_mask.any().item()) is True
    assert torch.equal(result.bgr[:, 0, 0], first[:, 0, 0])
    assert torch.equal(result.bgr[:, 0, 15], second[:, 0, 15])
    assert result.audit["dense_host_transfer_count"] == 0
    assert result.audit["owner_map_preserved"] is True
    assert result.audit["effective_levels"] == 3


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_safe_multiband_rejects_risk_or_protected_safe_overlap():
    torch = _torch()
    device = torch.device("cuda:0")
    first = torch.zeros((3, 4, 4), dtype=torch.uint8, device=device)
    second = torch.ones((3, 4, 4), dtype=torch.uint8, device=device)
    owner = torch.full((4, 4), 3, dtype=torch.int32, device=device)
    owner[:, 2:] = 5
    safe = torch.ones((4, 4), dtype=torch.bool, device=device)
    protected = torch.zeros_like(safe)
    protected[1, 1] = True

    with pytest.raises(CudaSafeMultiBandError, match="protected or risk"):
        blend_cuda_safe_multiband(
            torch,
            first_bgr=first,
            second_bgr=second,
            owner_frame_id=owner,
            first_frame_id=3,
            second_frame_id=5,
            safe_background_mask=safe,
            protected_mask=protected,
            risk_mask=torch.zeros_like(safe),
            band_pixels=16,
        )


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_safe_multiband_rejects_an_unprovided_owner_source():
    torch = _torch()
    device = torch.device("cuda:0")
    image = torch.zeros((3, 4, 4), dtype=torch.uint8, device=device)
    owner = torch.full((4, 4), 3, dtype=torch.int32, device=device)
    owner[:, 2:] = 5
    owner[0, 0] = 11
    mask = torch.zeros((4, 4), dtype=torch.bool, device=device)

    with pytest.raises(CudaSafeMultiBandError, match="unrelated real frame"):
        blend_cuda_safe_multiband(
            torch,
            first_bgr=image,
            second_bgr=image,
            owner_frame_id=owner,
            first_frame_id=3,
            second_frame_id=5,
            safe_background_mask=mask,
            protected_mask=mask,
            risk_mask=mask,
            band_pixels=16,
        )
