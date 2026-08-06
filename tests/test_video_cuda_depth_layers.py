from __future__ import annotations

import importlib

import pytest

from panorama_demo.video_cuda_depth_layers import cuda_same_layer_safe_mask


def _torch():
    return importlib.import_module("torch")


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_depth_layer_mask_accepts_only_same_layer_bidirectional_safe_pixels():
    torch = _torch()
    device = torch.device("cuda:0")
    first = torch.full((8, 8), 1000.0, device=device)
    second = torch.full((8, 8), 1000.0, device=device)
    forward = torch.zeros((8, 8, 2), device=device)
    backward = torch.zeros((8, 8, 2), device=device)
    second[3, 3] = 1400.0
    forward[2, 2, 0] = 3.0

    safe, audit = cuda_same_layer_safe_mask(
        torch,
        first_depth_mm=first,
        second_depth_mm=second,
        forward_flow_xy=forward,
        backward_flow_xy=backward,
    )

    assert safe.device.type == "cuda"
    assert bool(safe[3, 3].item()) is False
    assert bool(safe[2, 2].item()) is False
    assert audit["same_layer_safe_pixel_count"] < 64
    assert audit["host_transfer_count"] == 0
