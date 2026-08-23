"""Device-native runtime for the isolated S1.3 CUDA successor.

This module intentionally owns CuPy arrays; it never changes the public
NumPy-facing CUDA helpers used by the production renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .cuda_backend import cuda_status


@dataclass(frozen=True)
class DeviceStageImage:
    stage_name: str
    image: Any


@dataclass(frozen=True)
class S13DeviceRemapRequest:
    """One exact, bounded resident-source remap request."""

    request_id: str
    frame_id: int
    map_u: np.ndarray
    map_v: np.ndarray
    map_cache_key: object | None = None


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
        self._semantic_map_hits = self._semantic_map_misses = 0
        self._corridor_batches = self._corridor_tiles = 0
        self._corridor_d2h_count = self._corridor_d2h_bytes = 0
        self._event_ranges: dict[str, list[tuple[Any, Any]]] = {}
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

    def _request_map_key(self, request: S13DeviceRemapRequest, axis: str) -> object:
        if request.map_cache_key is None:
            return ("batch", axis, request.frame_id, self._host_array_key(
                request.map_u if axis == "u" else request.map_v
            ))
        return ("semantic", axis, request.frame_id, request.map_cache_key)

    def remap_batch_to_host(
        self, requests: Sequence[S13DeviceRemapRequest], *, batch_size: int = 16,
        event_range: str = "M5_CORRIDOR_BATCH",
    ) -> tuple[np.ndarray, ...]:
        """Exact remaps grouped by shape with one host transfer per bounded batch.

        The host-facing result order is always the request order.  No worker is
        allowed to touch this method; it owns the CUDA stream and its explicit
        synchronization boundary.
        """
        if not requests:
            return ()
        if batch_size not in (8, 16, 32):
            raise ValueError("S1.3 remap batch size must be one of 8, 16, or 32")
        indexed = list(enumerate(requests))
        results: list[np.ndarray | None] = [None] * len(indexed)
        buckets: dict[tuple[int, int], list[tuple[int, S13DeviceRemapRequest]]] = {}
        for index, request in indexed:
            u, v = np.asarray(request.map_u), np.asarray(request.map_v)
            if u.ndim != 2 or u.shape != v.shape:
                raise ValueError("S1.3 batch remap maps must be equally-shaped 2-D arrays")
            buckets.setdefault(tuple(u.shape), []).append((index, request))
        for items in buckets.values():
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                device_tiles: list[Any] = []
                event_start, event_end = self.cp.cuda.Event(), self.cp.cuda.Event()
                with self.compute_stream:
                    self.compute_stream.record(event_start)
                    for _index, request in chunk:
                        u_key = self._request_map_key(request, "u")
                        v_key = self._request_map_key(request, "v")
                        if request.map_cache_key is not None:
                            if u_key in self._maps and v_key in self._maps:
                                self._semantic_map_hits += 1
                            else:
                                self._semantic_map_misses += 1
                        device_tiles.append(self._remap_frame_linear_float(
                            request.frame_id, request.map_u, request.map_v, u_key, v_key
                        ))
                    atlas = self.cp.stack(device_tiles, axis=0)
                    self.compute_stream.record(event_end)
                    self._compute_ready.record(self.compute_stream)
                with self.download_stream:
                    self.download_stream.wait_event(self._compute_ready)
                    host_atlas = self.cp.asnumpy(atlas)
                self._d2h += int(host_atlas.nbytes)
                self._corridor_batches += 1
                self._corridor_tiles += len(chunk)
                self._corridor_d2h_count += 1
                self._corridor_d2h_bytes += int(host_atlas.nbytes)
                self._event_ranges.setdefault(event_range, []).append((event_start, event_end))
                for tile_index, (index, _request) in enumerate(chunk):
                    results[index] = host_atlas[tile_index]
        return tuple(result for result in results if result is not None)

    remap_probe_batch_to_host = remap_batch_to_host

    def _remap_frame_linear_float(
        self, frame_id: int, map_u: np.ndarray, map_v: np.ndarray, u_key: object, v_key: object,
    ) -> Any:
        source_gpu = self.source(frame_id)
        map_u = np.ascontiguousarray(map_u, dtype=np.float32)
        map_v = np.ascontiguousarray(map_v, dtype=np.float32)
        map_u_gpu = self.upload_map_once(u_key, map_u)
        map_v_gpu = self.upload_map_once(v_key, map_v)
        output = self.cp.empty((*map_u.shape, 3), dtype=self.cp.uint8)
        kernel = self._float_exact_linear_kernel()
        count = int(map_u.size)
        self.compute_stream.wait_event(self._upload_ready)
        kernel(
            ((count + 255) // 256,), (256,),
            (source_gpu, np.int32(source_gpu.shape[0]), np.int32(source_gpu.shape[1]),
             map_u_gpu, map_v_gpu, np.int32(map_u.shape[0]), np.int32(map_u.shape[1]), output),
        )
        self._kernels += 1
        return output

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

    def remap_resident_frame_exact_device(
        self, frame_id: int, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray,
    ) -> Any:
        """OpenCV fixed-map-equivalent resident remap for pixel-sealed stages."""
        if self._source_ids.get(self._host_source_key(source)) != int(frame_id):
            raise ValueError("S1.3 resident frame id/source identity mismatch")
        return self.remap_linear_exact(frame_id, map_u, map_v)

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
            # The compact affine parameters and mask were enqueued after the
            # remap's earlier upload event.  Join their newest upload point
            # before consuming them, otherwise a warm GPU can race P3.
            self.compute_stream.wait_event(self._upload_ready)
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

    def remap_resident_frame_corrected_linear_device(
        self, frame_id: int, source: np.ndarray, map_u: np.ndarray, map_v: np.ndarray,
        gain_bgr: tuple[float, float, float], bias_bgr: tuple[float, float, float],
        mapped: np.ndarray,
    ) -> Any:
        """Return one compact corrected linear tile without a source D2H."""
        if self._source_ids.get(self._host_source_key(source)) != int(frame_id):
            raise ValueError("S1.3 resident frame id/source identity mismatch")
        sampled = self.remap_linear_float(source, map_u, map_v)
        gain = self.device_copy(np.asarray(gain_bgr, dtype=np.float32).reshape(1, 1, 3))
        bias = self.device_copy(np.asarray(bias_bgr, dtype=np.float32).reshape(1, 1, 3))
        mapped_gpu = self.device_copy(np.asarray(mapped, dtype=bool))
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
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
        self._kernels += 1
        return corrected

    def new_linear_canvas(self, height: int, width: int) -> Any:
        """Allocate the M6 owner canvas on the resident device stream."""
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            canvas = self.cp.zeros((int(height), int(width), 3), dtype=self.cp.float32)
            self._compute_ready.record(self.compute_stream)
        return canvas

    def compose_linear_owner_roi(
        self, canvas: Any, x0: int, x1: int, corrected: Any, owner_mask: np.ndarray,
    ) -> None:
        """Write exact owner pixels from a compact corrected source tile."""
        owner_gpu = self.device_copy(np.asarray(owner_mask, dtype=bool))
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            target = canvas[:, int(x0):int(x1)]
            target[owner_gpu] = corrected[owner_gpu]
            self._compute_ready.record(self.compute_stream)

    def download_linear_canvas(self, canvas: Any) -> np.ndarray:
        """Download the single composed M6 owner canvas."""
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
            result = self.cp.asnumpy(canvas)
        self._d2h += int(result.nbytes)
        return result

    def apply_b1_secondary_weight_roi_device(
        self, *, canvas_device: Any, x0: int, x1: int, left_device: Any,
        right_device: Any, primary_owner_right_mask: np.ndarray,
        secondary_weight: np.ndarray, protected_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the M6.2 B1 primary/secondary semantics and return its ROI.

        The input gates are intentionally host-side so malformed decision
        arrays fail before a kernel can touch a published owner canvas.
        """
        left, right = np.asarray(left_device), np.asarray(right_device)
        owner_right, weight = np.asarray(primary_owner_right_mask, bool), np.asarray(secondary_weight, np.float32)
        if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
            raise ValueError("S1.3 M6.2 B1 left/right tiles must be equal HxWx3 arrays")
        if owner_right.shape != left.shape[:2] or weight.shape != left.shape[:2]:
            raise ValueError("S1.3 M6.2 B1 mask/weight shape does not match corridor")
        if int(x1) - int(x0) != left.shape[1]:
            raise ValueError("S1.3 M6.2 B1 ROI x0/x1 does not match corridor width")
        if not np.all(np.isfinite(weight)) or np.any(weight < 0.0) or np.any(weight > 0.499):
            raise ValueError("S1.3 M6.2 B1 secondary weight must be finite within [0, 0.499]")
        active = weight > 0.0
        if protected_mask is not None:
            protected = np.asarray(protected_mask, bool)
            if protected.shape != weight.shape:
                raise ValueError("S1.3 M6.2 B1 protected mask shape does not match corridor")
            if np.any(active & protected):
                raise ValueError("S1.3 M6.2 B1 protected pixels cannot be active")
        canvas = canvas_device if isinstance(canvas_device, self.cp.ndarray) else self.device_copy(canvas_device)
        left_gpu, right_gpu = self.device_copy(left.astype(np.float32)), self.device_copy(right.astype(np.float32))
        owner_gpu, weight_gpu = self.device_copy(owner_right), self.device_copy(weight)
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            primary = self.cp.where(owner_gpu[..., None], right_gpu, left_gpu)
            secondary = self.cp.where(owner_gpu[..., None], left_gpu, right_gpu)
            result = primary * (self.cp.float32(1.0) - weight_gpu[..., None]) + secondary * weight_gpu[..., None]
            target = canvas[:, int(x0):int(x1)]
            target[weight_gpu > 0.0] = result[weight_gpu > 0.0]
            self._compute_ready.record(self.compute_stream)
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
            host = self.cp.asnumpy(result)
        self._kernels += 1
        self._d2h += int(host.nbytes)
        return host

    def encode_linear_srgb_shadow(self, linear: np.ndarray) -> np.ndarray:
        """Encode a final linear canvas on CUDA for M6.2 shadow comparison."""
        host = np.ascontiguousarray(linear, dtype=np.float32)
        device = self.device_copy(host)
        with self.compute_stream:
            self.compute_stream.wait_event(self._upload_ready)
            encoded = self.cp.where(
                device <= self.cp.float32(0.0031308),
                device * self.cp.float32(12.92),
                self.cp.float32(1.055) * self.cp.power(device, self.cp.float32(1.0 / 2.4)) - self.cp.float32(0.055),
            )
            result = self.cp.rint(self.cp.clip(encoded * self.cp.float32(255.0), 0.0, 255.0)).astype(self.cp.uint8)
            self._compute_ready.record(self.compute_stream)
        with self.download_stream:
            self.download_stream.wait_event(self._compute_ready)
            output = self.cp.asnumpy(result)
        self._kernels += 1
        self._d2h += int(output.nbytes)
        return output

    def download_compact_tiles(
        self, tiles: Sequence[Any], *, batch_size: int = 16, event_range: str = "M6_SAMPLE",
    ) -> tuple[np.ndarray, ...]:
        """Batch equally shaped compact device tiles into bounded D2H atlases."""
        if not tiles:
            return ()
        if batch_size not in (8, 16, 32):
            raise ValueError("S1.3 compact tile batch size must be one of 8, 16, or 32")
        indexed = list(enumerate(tiles))
        results: list[np.ndarray | None] = [None] * len(indexed)
        buckets: dict[tuple[int, ...], list[tuple[int, Any]]] = {}
        for index, tile in indexed:
            buckets.setdefault(tuple(tile.shape), []).append((index, tile))
        for items in buckets.values():
            for start in range(0, len(items), batch_size):
                chunk = items[start:start + batch_size]
                event_start, event_end = self.cp.cuda.Event(), self.cp.cuda.Event()
                with self.compute_stream:
                    self.compute_stream.record(event_start)
                    atlas = self.cp.stack([tile for _index, tile in chunk], axis=0)
                    self.compute_stream.record(event_end)
                    self._compute_ready.record(self.compute_stream)
                with self.download_stream:
                    self.download_stream.wait_event(self._compute_ready)
                    host_atlas = self.cp.asnumpy(atlas)
                self._d2h += int(host_atlas.nbytes)
                self._corridor_batches += 1
                self._corridor_tiles += len(chunk)
                self._corridor_d2h_count += 1
                self._corridor_d2h_bytes += int(host_atlas.nbytes)
                self._event_ranges.setdefault(event_range, []).append((event_start, event_end))
                for tile_index, (index, _tile) in enumerate(chunk):
                    results[index] = host_atlas[tile_index]
        return tuple(result for result in results if result is not None)

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
            self.download_stream.wait_event(self._compute_ready)
            result = self.cp.asnumpy(array[y0:y1, x0:x1])
        self._d2h += int(result.nbytes)
        self._roi_downloads += 1
        return result

    def note_stage(self, name: str, elapsed_seconds: float) -> None:
        self._stage_wall_ms[name] = 1000.0 * max(0.0, float(elapsed_seconds))

    def report(self) -> dict[str, object]:
        free, total = self.cp.cuda.runtime.memGetInfo()
        pool = self.cp.get_default_memory_pool()
        events: dict[str, float] = {}
        for name, ranges in self._event_ranges.items():
            events[name] = float(sum(self.cp.cuda.get_elapsed_time(start, end) for start, end in ranges))
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
            "source_preload_count": sum(self._uploads.values()),
            "source_preload_bytes": sum(int(value.nbytes) for value in self._sources.values()),
            "source_cache_hit_count": 0,
            "source_cache_miss_count": sum(self._uploads.values()),
            "map_semantic_cache_hit_count": self._semantic_map_hits,
            "map_semantic_cache_miss_count": self._semantic_map_misses,
            "corridor_batch_count": self._corridor_batches,
            "corridor_tile_count": self._corridor_tiles,
            "corridor_d2h_count": self._corridor_d2h_count,
            "corridor_d2h_bytes": self._corridor_d2h_bytes,
            "cuda_event_ms": events,
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
            "stage_gpu_event_ms": events,
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


__all__ = ["DeviceStageImage", "S13CudaRuntime", "S13DeviceRemapRequest"]
