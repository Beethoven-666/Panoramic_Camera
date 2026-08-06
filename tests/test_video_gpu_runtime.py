"""Focused contract tests for the candidate-only resident GPU frame cache."""

from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_gpu_runtime import (
    GpuVideoFrame,
    ResidentVideoFrameCache,
    VideoGpuRuntimeConfig,
    VideoGpuRuntimeError,
)


torch = pytest.importorskip("torch")


def _source(frame_id: int) -> dict[str, object]:
    color = np.full((3, 5, 3), frame_id, dtype=np.uint8)
    depth = np.full((3, 5), 1000, dtype=np.uint16)
    depth[0, 0] = 0
    return {
        "frame_id": frame_id,
        "timestamp_us": frame_id * 1_000,
        "color_u8": color,
        "depth_mm": depth,
        "pose_prior": np.eye(4, dtype=np.float32),
    }


def test_cpu_fallback_is_explicit_and_keeps_real_source_tensors_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="prefer"))
    frame = cache.upload(**_source(10))

    assert isinstance(frame, GpuVideoFrame)
    assert frame.color_u8.shape == (3, 3, 5)
    assert frame.color_u8.dtype == torch.uint8
    assert frame.color_linear.dtype == torch.float32
    assert frame.depth_mm.dtype == torch.float32
    assert frame.depth_valid.dtype == torch.bool
    assert frame.depth_valid[0, 0].item() is False
    assert frame.color_u8.device.type == "cpu"
    audit = cache.audit()
    assert audit["execution_backend"] == "torch_cpu_fallback"
    assert audit["cuda_active"] is False
    assert audit["cuda_fallback_reason"] == "cuda_unavailable"
    assert audit["h2d_frame_upload_count"] == 0
    assert audit["per_source_h2d_upload_count"] == {10: 0}
    assert audit["per_source_h2d_bytes"] == {10: 0}
    assert audit["h2d_total_bytes"] == 0
    assert audit["gpu_memory"]["available"] is False
    assert audit["streams"] == {
        "upload": False,
        "compute": False,
        "output": False,
        "overlap_enabled": False,
    }


def test_frame_can_be_observed_twice_but_never_reuploaded_after_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = ResidentVideoFrameCache(
        VideoGpuRuntimeConfig(maximum_resident_frames=1, cuda_mode="off")
    )
    first = cache.upload(**_source(1))
    assert cache.upload(**_source(1)) is first
    cache.upload(**_source(2))
    assert cache.resident_frame_ids == (2,)
    with pytest.raises(VideoGpuRuntimeError, match="re-upload"):
        cache.upload(**_source(1))
    audit = cache.audit()
    assert audit["logical_frame_upload_count"] == 2
    assert audit["h2d_frame_upload_count"] == 0
    assert audit["eviction_count"] == 1


def test_required_cuda_fails_closed_and_final_download_boundary_is_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(VideoGpuRuntimeError, match="CUDA is required"):
        ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="required"))

    cache = ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="off"))
    frame = cache.upload(**_source(3))
    with pytest.raises(VideoGpuRuntimeError, match="only final"):
        cache.copy_final_to_cpu(frame.color_linear, artifact="intermediate")  # type: ignore[arg-type]
    copied = cache.copy_final_to_cpu(frame.color_linear, artifact="panorama")
    assert copied.device.type == "cpu"
    assert cache.audit()["final_d2h_copy_count"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA Torch runtime")
def test_cuda_runtime_audits_exact_transfers_events_memory_and_final_device_guard() -> None:
    cache = ResidentVideoFrameCache(
        VideoGpuRuntimeConfig(maximum_resident_frames=2, cuda_mode="required")
    )
    frame = cache.upload(**_source(7))
    assert frame.color_u8.device.type == "cuda"
    assert frame.color_u8.device.index == 0

    # A CPU tensor cannot be smuggled through the CUDA runtime's final D2H
    # boundary, even when it otherwise looks like a Torch final artifact.
    with pytest.raises(VideoGpuRuntimeError, match="selected device"):
        cache.copy_final_to_cpu(torch.zeros((1,), device="cpu"), artifact="scalar_audit")
    # Output is likewise forbidden until it has a concrete compute event to
    # wait on; this prevents a stream audit from claiming a dependency it did
    # not actually establish.
    with pytest.raises(VideoGpuRuntimeError, match="compute event"):
        cache.copy_final_to_cpu(frame.color_linear, artifact="panorama")

    with cache.compute_context():
        final = frame.color_linear.mul(1.0)
    copied = cache.copy_final_to_cpu(final, artifact="panorama")
    assert copied.device.type == "cpu"

    audit = cache.audit()
    expected_h2d_bytes = (3 * 5 * 3) + (3 * 5 * 4) + (4 * 4 * 4)
    assert audit["execution_backend"] == "torch_cuda_resident"
    assert audit["per_source_h2d_upload_count"] == {7: 1}
    assert audit["per_source_h2d_bytes"] == {7: expected_h2d_bytes}
    assert audit["h2d_tensor_copy_count"] == 3
    assert audit["h2d_total_bytes"] == expected_h2d_bytes
    assert audit["final_d2h_copy_count"] == 1
    assert audit["final_d2h_bytes"] == final.numel() * final.element_size()
    assert audit["stream_events"] == {
        "dependency_mode": "cuda_event_chain",
        "upload_event_record_count": 1,
        "compute_wait_upload_event_count": 1,
        "compute_event_record_count": 1,
        "output_wait_compute_event_count": 1,
        "output_event_record_count": 1,
        "host_synchronization_count": 1,
        "host_synchronization_scope": "final_d2h_only",
    }
    memory = audit["gpu_memory"]
    assert memory["available"] is True
    assert memory["scope"] == "process_device_since_runtime_initialization_reset"
    assert memory["peak_allocated_bytes"] >= memory["baseline_allocated_bytes"]
    assert memory["peak_increment_allocated_bytes"] >= 0


def test_invalid_source_shape_and_closed_cache_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cache = ResidentVideoFrameCache(VideoGpuRuntimeConfig(cuda_mode="off"))
    source = _source(4)
    source["depth_mm"] = np.zeros((2, 5), dtype=np.uint16)
    with pytest.raises(VideoGpuRuntimeError, match="matching color"):
        cache.upload(**source)  # type: ignore[arg-type]
    cache.close()
    with pytest.raises(VideoGpuRuntimeError, match="closed"):
        cache.upload(**_source(5))
