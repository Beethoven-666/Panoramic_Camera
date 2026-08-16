from __future__ import annotations

import numpy as np
import pytest
import cv2

from panorama_demo.cuda_backend import cuda_status
from panorama_demo.video_s13_cuda_runtime import DeviceStageImage, S13CudaRuntime


def test_cuda_runtime_requires_required_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "off")
    cuda_status(refresh=True)
    with pytest.raises(RuntimeError, match="G305_CUDA=required"):
        S13CudaRuntime()


def test_cuda_runtime_preloads_once_and_accounts_transfers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "required")
    if not cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    runtime = S13CudaRuntime(memory_fraction=0.65)
    try:
        source = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)
        runtime.preload_sources({7: source})
        runtime.preload_sources({7: source})
        assert np.array_equal(runtime.download_roi(runtime.source(7), (0, 4, 0, 6)), source)
        downloaded = runtime.download_stage_once(DeviceStageImage("P0", runtime.device_copy(source)))
        assert np.array_equal(downloaded, source)
        report = runtime.report()
        assert report["source_upload_count_by_frame"] == {7: 1}
        assert report["full_stage_downloads"]["P0"] == 1
        assert report["designated_fallback_count"] == 0
    finally:
        runtime.close()


def test_cuda_runtime_fixed_point_linear_remap_matches_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "required")
    if not cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    rng = np.random.default_rng(20260816)
    source = rng.integers(0, 256, (47, 61, 3), dtype=np.uint8)
    yy, xx = np.indices((43, 59), dtype=np.float32)
    map_u = xx - 1.17 + rng.uniform(-0.49, 0.49, xx.shape).astype(np.float32)
    map_v = yy + 0.31 + rng.uniform(-0.49, 0.49, yy.shape).astype(np.float32)
    fixed, fractions = cv2.convertMaps(map_u, map_v, cv2.CV_16SC2)
    expected = cv2.remap(source, fixed, fractions, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    runtime = S13CudaRuntime()
    try:
        runtime.preload_sources({5: source})
        actual = runtime.download_roi(runtime.remap_linear_exact(5, map_u, map_v), (0, map_u.shape[0], 0, map_u.shape[1]))
        np.testing.assert_array_equal(actual, expected)
    finally:
        runtime.close()


def test_cuda_runtime_float_linear_remap_matches_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "required")
    if not cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    rng = np.random.default_rng(20260817)
    source = rng.integers(0, 256, (47, 61, 3), dtype=np.uint8)
    yy, xx = np.indices((43, 59), dtype=np.float32)
    map_u = xx - 1.17 + rng.uniform(-0.49, 0.49, xx.shape).astype(np.float32)
    map_v = yy + 0.31 + rng.uniform(-0.49, 0.49, yy.shape).astype(np.float32)
    expected = cv2.remap(source, map_u, map_v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    runtime = S13CudaRuntime()
    try:
        runtime.preload_sources({5: source})
        np.testing.assert_array_equal(runtime.remap_host_source(source, map_u, map_v), expected)
        assert runtime.report()["source_upload_count_by_frame"] == {5: 1}
    finally:
        runtime.close()


def test_cuda_runtime_resolves_each_preloaded_source_without_aliasing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "required")
    if not cuda_status(refresh=True).available:
        pytest.skip("CUDA runtime is unavailable")
    first = np.full((8, 9, 3), 17, np.uint8)
    second = np.full((8, 9, 3), 203, np.uint8)
    yy, xx = np.indices((8, 9), dtype=np.float32)
    runtime = S13CudaRuntime()
    try:
        runtime.preload_sources({1: first, 2: second})
        np.testing.assert_array_equal(runtime.remap_host_source(first, xx, yy), first)
        np.testing.assert_array_equal(runtime.remap_host_source(second, xx, yy), second)
    finally:
        runtime.close()
