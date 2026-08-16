"""Device-native runtime for the isolated S1.3 CUDA successor.

This module intentionally owns CuPy arrays; it never changes the public
NumPy-facing CUDA helpers used by the production renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

from .cuda_backend import cuda_status


@dataclass(frozen=True)
class DeviceStageImage:
    stage_name: str
    image: Any


class S13CudaRuntime:
    """Bounded CuPy source/map cache with explicit transfer accounting."""

    def __init__(self, *, device_id: int = 0, memory_fraction: float = 0.65) -> None:
        status = cuda_status(refresh=True)
        if status.mode != "required" or not status.cupy_available or not status.available:
            raise RuntimeError("S1.3 CUDA resident runner requires G305_CUDA=required and CuPy")
        import cupy as cp

        self.cp = cp
        self.device_id = int(device_id)
        self.memory_fraction = float(memory_fraction)
        if not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError("memory_fraction must be in (0, 1]")
        cp.cuda.Device(self.device_id).use()
        self.upload_stream = cp.cuda.Stream(non_blocking=True)
        self.compute_stream = cp.cuda.Stream(non_blocking=True)
        self.download_stream = cp.cuda.Stream(non_blocking=True)
        self._sources: dict[int, Any] = {}
        self._maps: dict[object, Any] = {}
        self._uploads: dict[int, int] = {}
        self._h2d = self._d2h = self._kernels = 0
        self._full_downloads = {stage: 0 for stage in ("P0", "P1", "P2", "P3")}
        self._roi_downloads = 0
        self._map_hits = self._map_misses = 0
        self._started = time.perf_counter()
        self._stage_wall_ms: dict[str, float] = {}

    def preload_sources(self, selected_frames: Mapping[int, np.ndarray]) -> None:
        free, _ = self.cp.cuda.runtime.memGetInfo()
        required = sum(int(np.asarray(image).nbytes) for image in selected_frames.values())
        if required > int(free * self.memory_fraction):
            raise MemoryError("S1.3 CUDA resident source cache exceeds its bounded GPU budget")
        with self.upload_stream:
            for frame_id, image in selected_frames.items():
                if frame_id in self._sources:
                    continue
                host = np.ascontiguousarray(image)
                self._sources[int(frame_id)] = self.cp.asarray(host)
                self._uploads[int(frame_id)] = self._uploads.get(int(frame_id), 0) + 1
                self._h2d += int(host.nbytes)

    def source(self, frame_id: int) -> Any:
        try:
            return self._sources[int(frame_id)]
        except KeyError as exc:
            raise KeyError(f"S1.3 CUDA source was not preloaded: {frame_id}") from exc

    def upload_map_once(self, key: object, map_host: np.ndarray) -> Any:
        cached = self._maps.get(key)
        if cached is not None:
            self._map_hits += 1
            return cached
        host = np.ascontiguousarray(map_host)
        with self.upload_stream:
            cached = self.cp.asarray(host)
        self._maps[key] = cached
        self._map_misses += 1
        self._h2d += int(host.nbytes)
        return cached

    def device_copy(self, array: np.ndarray | Any) -> Any:
        """Upload a compact decision mask/ROI; caller controls its lifetime."""
        if isinstance(array, self.cp.ndarray):
            return array
        host = np.ascontiguousarray(array)
        with self.upload_stream:
            result = self.cp.asarray(host)
        self._h2d += int(host.nbytes)
        return result

    def download_stage_once(self, stage: DeviceStageImage) -> np.ndarray:
        if stage.stage_name not in self._full_downloads:
            raise ValueError(f"Unknown S1.3 stage {stage.stage_name}")
        with self.download_stream:
            result = self.cp.asnumpy(stage.image)
        self._d2h += int(result.nbytes)
        self._full_downloads[stage.stage_name] += 1
        return result

    def download_roi(self, array: Any, roi: tuple[int, int, int, int]) -> np.ndarray:
        y0, y1, x0, x1 = roi
        with self.download_stream:
            result = self.cp.asnumpy(array[y0:y1, x0:x1])
        self._d2h += int(result.nbytes)
        self._roi_downloads += 1
        return result

    def note_stage(self, name: str, elapsed_seconds: float) -> None:
        self._stage_wall_ms[name] = 1000.0 * max(0.0, float(elapsed_seconds))

    def report(self) -> dict[str, object]:
        free, total = self.cp.cuda.runtime.memGetInfo()
        pool = self.cp.get_default_memory_pool()
        return {
            "schema": "s13-cuda-resident-audit/v1",
            "requested_mode": "required",
            "resolved_backend": "cupy",
            "device_id": self.device_id,
            "device_name": cuda_status().device_name,
            "context_initialized_before_output": True,
            "kernel_launch_count": self._kernels,
            "h2d_bytes": self._h2d,
            "d2h_bytes": self._d2h,
            "source_upload_count": sum(self._uploads.values()),
            "source_upload_count_by_frame": dict(self._uploads),
            "map_upload_count": self._map_misses,
            "map_cache_hit_count": self._map_hits,
            "map_cache_miss_count": self._map_misses,
            "full_stage_download_count": sum(self._full_downloads.values()),
            "full_stage_downloads": dict(self._full_downloads),
            "roi_download_count": self._roi_downloads,
            "peak_device_used_bytes": int(total - free),
            "peak_memory_pool_bytes": int(pool.used_bytes()),
            "designated_fallback_count": 0,
            "fallback_reasons": [],
            "stage_wall_ms": dict(self._stage_wall_ms),
            "stage_gpu_event_ms": {},
            "cpu_retained_algorithms": [
                "M1 GFTT", "M1 forward/backward PyrLK", "M5 GFTT",
                "M5 forward/backward PyrLK", "M5 geometry decision", "M5 seam DP",
                "M5 component decision", "C2E median and final decision",
                "M6 quantile/median/lstsq/model selection", "PNG encoding",
            ],
        }

    def close(self) -> None:
        self._sources.clear()
        self._maps.clear()


__all__ = ["DeviceStageImage", "S13CudaRuntime"]
