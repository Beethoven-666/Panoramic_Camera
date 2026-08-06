from __future__ import annotations

import importlib

import numpy as np
import pytest

from panorama_demo.video_cuda_renderer import (
    CudaRenderSource,
    TorchCudaCandidateTileRenderer,
    TorchCudaVideoRendererError,
    calibrated_inverse_grid,
    compose_inverse_grid,
)
from panorama_demo.video_gpu_runtime import ResidentVideoFrameCache, VideoGpuRuntimeConfig


def _torch():
    return importlib.import_module("torch")


def test_cuda_renderer_rejects_cpu_cache():
    cache = ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="off"))

    with pytest.raises(TorchCudaVideoRendererError, match="active CUDA"):
        TorchCudaCandidateTileRenderer(cache)


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_renderer_rejects_raft_on_a_different_device_before_inference():
    cache = ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="required"))
    frame = cache.upload(
        frame_id=1,
        timestamp_us=0,
        color_u8=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_mm=np.full((8, 8), 600, dtype=np.uint16),
        pose_prior=np.eye(4, dtype=np.float32),
    )
    other = cache.upload(
        frame_id=2,
        timestamp_us=1,
        color_u8=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_mm=np.full((8, 8), 600, dtype=np.uint16),
        pose_prior=np.eye(4, dtype=np.float32),
    )

    class WrongDeviceRaft:
        device = "cuda:99"

        def estimate_pair_tensors(self, *_: object, **__: object) -> object:
            raise AssertionError("device check must happen before model inference")

    with pytest.raises(TorchCudaVideoRendererError, match="one CUDA device"):
        TorchCudaCandidateTileRenderer(cache).estimate_raft_flow(
            WrongDeviceRaft(), source=frame, target=other
        )
    cache.close()


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_renderer_uses_one_composed_grid_sample_and_final_only_download():
    torch = _torch()
    cache = ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="required", maximum_resident_frames=2))
    rgb0 = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb0[..., 0] = 10
    rgb1 = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb1[..., 2] = 200
    depth = np.full((4, 4), 700, dtype=np.uint16)
    pose = np.eye(4, dtype=np.float32)
    first = cache.upload(frame_id=7, timestamp_us=10, color_u8=rgb0, depth_mm=depth, pose_prior=pose)
    second = cache.upload(frame_id=9, timestamp_us=20, color_u8=rgb1, depth_mm=depth, pose_prior=pose)
    grid = calibrated_inverse_grid(cache, height=4, width=4, fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    composed = compose_inverse_grid(grid, residual_mesh_offset_xy=torch.zeros_like(grid))
    owner = torch.tensor(
        [[7, 7, 9, 9], [7, 7, 9, 9], [7, 7, 9, 9], [7, 7, 9, 9]],
        dtype=torch.int32,
        device=cache.device,
    )
    result = TorchCudaCandidateTileRenderer(cache).render_hard_owner(
        [CudaRenderSource(7, first, composed), CudaRenderSource(9, second, composed)], owner
    )

    panorama, provenance, audit = TorchCudaCandidateTileRenderer(cache).finalize(result)

    assert panorama.shape == (4, 4, 3)
    assert np.array_equal(provenance, owner.cpu().numpy())
    # RGB sources cross the CUDA delivery boundary as OpenCV BGR.  A red
    # source therefore appears in channel 2, and a blue source in channel 0.
    assert np.all(panorama[:, :2, 2] == 10)
    assert np.all(panorama[:, 2:, 0] == 200)
    assert audit["grid_sample_count"] == 2
    assert audit["gpu_runtime"]["intermediate_d2h_count"] == 0
    assert audit["gpu_runtime"]["final_d2h_copy_count"] == 2
    cache.close()


@pytest.mark.skipif(not _torch().cuda.is_available(), reason="requires actual CUDA")
def test_cuda_renderer_rejects_owner_without_a_real_source_sample():
    torch = _torch()
    cache = ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="required"))
    rgb = np.full((4, 4, 3), 50, dtype=np.uint8)
    frame = cache.upload(
        frame_id=3,
        timestamp_us=0,
        color_u8=rgb,
        depth_mm=np.full((4, 4), 600, dtype=np.uint16),
        pose_prior=np.eye(4, dtype=np.float32),
    )
    grid = calibrated_inverse_grid(cache, height=4, width=4, fx=2.0, fy=2.0, cx=1.5, cy=1.5)
    owner = torch.full((4, 4), 42, dtype=torch.int32, device=cache.device)

    with pytest.raises(TorchCudaVideoRendererError, match="lacks a valid genuine"):
        TorchCudaCandidateTileRenderer(cache).render_hard_owner(
            [CudaRenderSource(3, frame, grid)], owner
        )
    cache.close()
