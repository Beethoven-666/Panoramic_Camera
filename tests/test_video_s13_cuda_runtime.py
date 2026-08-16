from __future__ import annotations

import numpy as np
import pytest

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
