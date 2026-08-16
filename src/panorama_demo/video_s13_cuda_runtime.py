"""Device-native runtime for the isolated S1.3 CUDA successor.

This module intentionally owns CuPy arrays; it never changes the public
NumPy-facing CUDA helpers used by the production renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import cv2
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
        self._upload_ready = cp.cuda.Event()
        self._compute_ready = cp.cuda.Event()
        self._sources: dict[int, Any] = {}
        self._source_ids: dict[tuple[int, tuple[int, ...], tuple[int, ...], str], int] = {}
        self._maps: dict[object, Any] = {}
        self._map_host_refs: dict[object, np.ndarray] = {}
        self._uploads: dict[int, int] = {}
        self._h2d = self._d2h = self._kernels = 0
        self._full_downloads = {stage: 0 for stage in ("P0", "P1", "P2", "P3")}
        self._roi_downloads = 0
        self._map_hits = self._map_misses = 0
        self._started = time.perf_counter()
        self._stage_wall_ms: dict[str, float] = {}
        self._linear_kernel: Any | None = None
        self._float_linear_kernel: Any | None = None

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
                self._source_ids[self._host_source_key(image)] = int(frame_id)
                self._uploads[int(frame_id)] = self._uploads.get(int(frame_id), 0) + 1
                self._h2d += int(host.nbytes)
            self._upload_ready.record(self.upload_stream)

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
            self._upload_ready.record(self.upload_stream)
        self._maps[key] = cached
        # Address keys are valid only while the host allocation remains alive.
        # Holding this compact map reference prevents NumPy from reusing its
        # pointer for another map later in the same deterministic run.
        self._map_host_refs[key] = host
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
            self._upload_ready.record(self.upload_stream)
        self._h2d += int(host.nbytes)
        return result

    def remap_linear_exact(self, source_frame_id: int, map_u: np.ndarray, map_v: np.ndarray) -> Any:
        """Return a device BGR remap using OpenCV's 1/32 fixed-point map form.

        ``cv2.convertMaps`` is intentionally kept on the CPU: it establishes
        OpenCV's fixed-point remap representation. The device kernel then
        applies the same four integer bilinear weights without creating a
        NumPy image boundary. Callers must opt into this fixed-map semantic;
        the legacy float-map renderer remains the reference authority.
        """
        source = self.source(source_frame_id)
        if source.dtype != self.cp.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise TypeError("S1.3 exact device remap requires HxWx3 uint8 source")
        fixed, fractions = cv2.convertMaps(
            np.ascontiguousarray(map_u, dtype=np.float32),
            np.ascontiguousarray(map_v, dtype=np.float32), cv2.CV_16SC2,
        )
        fixed_gpu = self.upload_map_once(("linear-fixed", source_frame_id, self._host_array_key(fixed)), fixed)
        fraction_gpu = self.upload_map_once(("linear-frac", source_frame_id, self._host_array_key(fractions)), fractions)
        output = self.cp.empty((*fractions.shape, 3), dtype=self.cp.uint8)
        kernel = self._exact_linear_kernel()
        count = int(fractions.size)
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            kernel(
                ((count + 255) // 256,), (256,),
                (source, np.int32(source.shape[0]), np.int32(source.shape[1]), fixed_gpu,
                 fraction_gpu, np.int32(fractions.shape[0]), np.int32(fractions.shape[1]), output),
            )
            self._compute_ready.record(self.compute_stream)
        self._kernels += 1
        return output

    def remap_linear_float(self, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray) -> Any:
        """Float-map OpenCV-compatible remap from an already-resident source.

        The map upload is cached and the source is resolved by object identity,
        so a contributor is not uploaded again at each S1.3 stage boundary.
        """
        try:
            source_frame_id = self._source_ids[self._host_source_key(source)]
        except KeyError as exc:
            raise ValueError("S1.3 CUDA remap received a non-resident source image") from exc
        source_gpu = self.source(source_frame_id)
        map_u = np.ascontiguousarray(map_u, dtype=np.float32)
        map_v = np.ascontiguousarray(map_v, dtype=np.float32)
        if map_u.shape != map_v.shape or map_u.ndim != 2:
            raise ValueError("S1.3 CUDA remap maps must be equally-shaped 2-D arrays")
        map_u_gpu = self.upload_map_once(("float-u", self._host_array_key(map_u)), map_u)
        map_v_gpu = self.upload_map_once(("float-v", self._host_array_key(map_v)), map_v)
        output = self.cp.empty((*map_u.shape, 3), dtype=self.cp.uint8)
        kernel = self._float_exact_linear_kernel()
        count = int(map_u.size)
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            kernel(
                ((count + 255) // 256,), (256,),
                (source_gpu, np.int32(source_gpu.shape[0]), np.int32(source_gpu.shape[1]),
                 map_u_gpu, map_v_gpu, np.int32(map_u.shape[0]), np.int32(map_u.shape[1]), output),
            )
            self._compute_ready.record(self.compute_stream)
        self._kernels += 1
        return output

    def remap_host_source(
        self, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray,
        interpolation: int = cv2.INTER_LINEAR, *, borderMode: int = cv2.BORDER_CONSTANT,
        borderValue: object = 0,
    ) -> np.ndarray:
        if interpolation != cv2.INTER_LINEAR or borderMode != cv2.BORDER_CONSTANT or borderValue != 0:
            raise ValueError("S1.3 resident remap only supports linear constant-zero RGB sampling")
        device = self.remap_linear_float(source, map_u, map_v)
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
            result = self.cp.asnumpy(device)
        self._d2h += int(result.nbytes)
        return result

    def remap_resident_frame(
        self, frame_id: int, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray,
    ) -> np.ndarray:
        if self._host_source_key(source) not in self._source_ids:
            raise ValueError("S1.3 resident frame source identity changed")
        if self._source_ids[self._host_source_key(source)] != int(frame_id):
            raise ValueError("S1.3 resident frame id/source identity mismatch")
        # ``preload_sources`` owns the immutable contributor and records this
        # exact host allocation.  Re-summing a full device image here forced a
        # stream synchronization for every compact remap, defeating residency
        # without increasing the supported-use safety of the cache.
        device = self.remap_linear_float(source, map_u, map_v)
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
            result = self.cp.asnumpy(device)
        self._d2h += int(result.nbytes)
        return result

    def remap_resident_frame_device(
        self, frame_id: int, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray,
    ) -> Any:
        if self._source_ids.get(self._host_source_key(source)) != int(frame_id):
            raise ValueError("S1.3 resident frame id/source identity mismatch")
        return self.remap_linear_float(source, map_u, map_v)

    def remap_resident_frame_linear(
        self, frame_id: int, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray,
    ) -> np.ndarray:
        """Remap an already-resident source and decode sRGB before one D2H."""
        if self._source_ids.get(self._host_source_key(source)) != int(frame_id):
            raise ValueError("S1.3 resident frame id/source identity mismatch")
        sampled = self.remap_linear_float(source, map_u, map_v)
        with self.compute_stream:
            encoded = sampled.astype(self.cp.float32) / self.cp.float32(255.0)
            linear = self.cp.where(
                encoded <= self.cp.float32(0.04045),
                encoded / self.cp.float32(12.92),
                self.cp.power(
                    (encoded + self.cp.float32(0.055)) / self.cp.float32(1.055),
                    self.cp.float32(2.4),
                ),
            ).astype(self.cp.float32)
            self._compute_ready.record(self.compute_stream)
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
            result = self.cp.asnumpy(linear)
        self._kernels += 1
        self._d2h += int(result.nbytes)
        return result

    def remap_resident_frame_corrected_linear(
        self, frame_id: int, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray,
        gain_bgr: tuple[float, float, float], bias_bgr: tuple[float, float, float],
        mapped: np.ndarray,
    ) -> np.ndarray:
        """Apply a selected M6 affine correction before the sole linear D2H."""
        if self._source_ids.get(self._host_source_key(source)) != int(frame_id):
            raise ValueError("S1.3 resident frame id/source identity mismatch")
        sampled = self.remap_linear_float(source, map_u, map_v)
        gain = self.device_copy(np.asarray(gain_bgr, dtype=np.float32).reshape(1, 1, 3))
        bias = self.device_copy(np.asarray(bias_bgr, dtype=np.float32).reshape(1, 1, 3))
        mapped_gpu = self.device_copy(np.asarray(mapped, dtype=bool))
        with self.compute_stream:
            encoded = sampled.astype(self.cp.float32) / self.cp.float32(255.0)
            linear = self.cp.where(
                encoded <= self.cp.float32(0.04045),
                encoded / self.cp.float32(12.92),
                self.cp.power(
                    (encoded + self.cp.float32(0.055)) / self.cp.float32(1.055),
                    self.cp.float32(2.4),
                ),
            ).astype(self.cp.float32)
            corrected = self.cp.clip(linear * gain + bias, 0.0, 1.0)
            corrected[~mapped_gpu] = self.cp.float32(0.0)
            self._compute_ready.record(self.compute_stream)
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
            result = self.cp.asnumpy(corrected)
        self._kernels += 1
        self._d2h += int(result.nbytes)
        return result

    def new_stage_canvas(self, height: int, width: int) -> Any:
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            canvas = self.cp.zeros((int(height), int(width), 3), dtype=self.cp.uint8)
            self._compute_ready.record(self.compute_stream)
        return canvas

    def compose_owner_roi(self, canvas: Any, x0: int, x1: int, sampled: Any, valid: np.ndarray) -> None:
        valid_gpu = self.device_copy(np.asarray(valid, dtype=bool))
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            roi = canvas[:, int(x0):int(x1)]
            roi[valid_gpu] = sampled[valid_gpu]
            self._compute_ready.record(self.compute_stream)

    def download_stage_canvas(self, stage_name: str, canvas: Any) -> np.ndarray:
        return self.download_stage_once(DeviceStageImage(stage_name, canvas))

    def _exact_linear_kernel(self) -> Any:
        if self._linear_kernel is not None:
            return self._linear_kernel
        self._linear_kernel = self.cp.RawKernel(
            r'''extern "C" __global__
            void s13_linear(const unsigned char* src, int src_h, int src_w,
                            const short* xy, const unsigned short* frac,
                            int out_h, int out_w, unsigned char* out) {
                int p = blockDim.x * blockIdx.x + threadIdx.x;
                int count = out_h * out_w;
                if (p >= count) return;
                int x0 = (int)xy[2 * p];
                int y0 = (int)xy[2 * p + 1];
                int f = (int)frac[p];
                int fx = f & 31;
                int fy = f >> 5;
                int w00 = (32 - fx) * (32 - fy);
                int w01 = fx * (32 - fy);
                int w10 = (32 - fx) * fy;
                int w11 = fx * fy;
                for (int c = 0; c < 3; ++c) {
                    int sum = 0;
                    int sx[4] = {x0, x0 + 1, x0, x0 + 1};
                    int sy[4] = {y0, y0, y0 + 1, y0 + 1};
                    int ww[4] = {w00, w01, w10, w11};
                    for (int i = 0; i < 4; ++i) {
                        if (sx[i] >= 0 && sx[i] < src_w && sy[i] >= 0 && sy[i] < src_h)
                            sum += ww[i] * (int)src[(sy[i] * src_w + sx[i]) * 3 + c];
                    }
                    out[p * 3 + c] = (unsigned char)((sum + 512) >> 10);
                }
            }''',
            "s13_linear",
        )
        return self._linear_kernel

    @staticmethod
    def _host_source_key(source: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...], str]:
        array = np.asarray(source)
        return (
            int(array.__array_interface__["data"][0]), tuple(array.shape),
            tuple(array.strides), array.dtype.str,
        )

    @staticmethod
    def _host_array_key(array: np.ndarray) -> tuple[int, tuple[int, ...], tuple[int, ...], str]:
        """Address-based key for immutable per-run maps without copying them."""
        value = np.asarray(array)
        return (
            int(value.__array_interface__["data"][0]), tuple(value.shape),
            tuple(value.strides), value.dtype.str,
        )

    def _float_exact_linear_kernel(self) -> Any:
        if self._float_linear_kernel is not None:
            return self._float_linear_kernel
        self._float_linear_kernel = self.cp.RawKernel(
            r'''extern "C" __global__
            void s13_float_linear(const unsigned char* src, int src_h, int src_w,
                                  const float* mx, const float* my,
                                  int out_h, int out_w, unsigned char* out) {
                int p = blockDim.x * blockIdx.x + threadIdx.x;
                int count = out_h * out_w;
                if (p >= count) return;
                double x = (double)mx[p], y = (double)my[p];
                for (int c = 0; c < 3; ++c) {
                    double value = 0.0;
                    if (isfinite(x) && isfinite(y)) {
                        int x0 = (int)floorf((float)x), y0 = (int)floorf((float)y);
                        double ax = x - (double)x0, ay = y - (double)y0;
                        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
                            int sx = x0 + dx, sy = y0 + dy;
                            if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
                                double wx = dx ? ax : 1.0 - ax;
                                double wy = dy ? ay : 1.0 - ay;
                                value += (double)src[(sy * src_w + sx) * 3 + c] * wx * wy;
                            }
                        }
                    }
                    int rounded = (int)nearbyintf((float)value);
                    out[p * 3 + c] = (unsigned char)min(255, max(0, rounded));
                }
            }''', "s13_float_linear",
        )
        return self._float_linear_kernel

    def download_stage_once(self, stage: DeviceStageImage) -> np.ndarray:
        if stage.stage_name not in self._full_downloads:
            raise ValueError(f"Unknown S1.3 stage {stage.stage_name}")
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
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
        self._source_ids.clear()
        self._maps.clear()
        self._map_host_refs.clear()


__all__ = ["DeviceStageImage", "S13CudaRuntime"]
