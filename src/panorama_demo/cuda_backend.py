"""Optional CUDA primitives used by the RGB-D pipeline.

The project deliberately keeps CUDA behind a small boundary:

* ``G305_CUDA=auto`` (default) uses CUDA when a supported device/runtime exists.
* ``G305_CUDA=off`` forces the reference CPU implementation.
* ``G305_CUDA=required`` fails closed if CUDA cannot be initialized.

OpenCV CUDA is preferred when the installed OpenCV build provides it.  The
official Python wheels are commonly CPU-only, so a CuPy implementation covers
the high-volume inverse-remap and pinhole geometry paths without introducing
Torch into the formal pipeline.  All public functions return NumPy arrays and
therefore do not leak device ownership into trajectory or delivery auditing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from threading import Lock
import time
from typing import Any

import cv2
import numpy as np


_VALID_MODES = {"auto", "off", "required"}
_MINIMUM_AUTO_BYTES = 64 * 1024
_STATUS_LOCK = Lock()
_STATUS: "CudaStatus | None" = None
_CUPY: Any | None = None
_REMAP_KERNELS: dict[str, Any] = {}
_AUTO_DECISIONS: dict[tuple[object, ...], str] = {}
_COUNTERS = {
    "opencv_cuda_calls": 0,
    "cupy_calls": 0,
    "cpu_calls": 0,
    "host_to_device_bytes": 0,
    "device_to_host_bytes": 0,
}


@dataclass(frozen=True)
class CudaStatus:
    mode: str
    available: bool
    backend: str
    device_count: int
    device_name: str | None
    opencv_cuda_available: bool
    cupy_available: bool
    reason: str | None

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["counters"] = dict(_COUNTERS)
        result["auto_decisions"] = {
            repr(key): value for key, value in _AUTO_DECISIONS.items()
        }
        return result


def _mode() -> str:
    value = os.environ.get("G305_CUDA", "auto").strip().lower()
    if value not in _VALID_MODES:
        raise RuntimeError(
            f"G305_CUDA must be one of {sorted(_VALID_MODES)}, got {value!r}"
        )
    return value


def _detect_status() -> CudaStatus:
    global _CUPY
    mode = _mode()
    if mode == "off":
        return CudaStatus(
            mode=mode,
            available=False,
            backend="cpu",
            device_count=0,
            device_name=None,
            opencv_cuda_available=False,
            cupy_available=False,
            reason="disabled_by_G305_CUDA",
        )
    cv_count = 0
    try:
        cv_count = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        cv_count = 0
    cupy_count = 0
    cupy_name: str | None = None
    cupy_reason: str | None = None
    try:
        import cupy as cp

        cupy_count = int(cp.cuda.runtime.getDeviceCount())
        if cupy_count:
            raw_name = cp.cuda.runtime.getDeviceProperties(0).get("name")
            cupy_name = (
                raw_name.decode("utf-8", errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            # Force context creation now so "required" cannot fail later after
            # formal output staging has begun.
            cp.cuda.Device(0).use()
            cp.zeros(1, dtype=cp.uint8).sum().get()
            _CUPY = cp
    except Exception as exc:  # pragma: no cover - depends on local CUDA runtime
        cupy_reason = f"{type(exc).__name__}: {exc}"
        _CUPY = None
        cupy_count = 0
    available = cv_count > 0 or cupy_count > 0
    if mode == "required" and not available:
        raise RuntimeError(
            "CUDA was required but neither a CUDA-enabled OpenCV build nor "
            f"a working CuPy device is available ({cupy_reason or 'no device'})"
        )
    backend = "opencv_cuda" if cv_count > 0 else ("cupy" if cupy_count > 0 else "cpu")
    return CudaStatus(
        mode=mode,
        available=available,
        backend=backend,
        device_count=max(cv_count, cupy_count),
        device_name=cupy_name,
        opencv_cuda_available=cv_count > 0,
        cupy_available=cupy_count > 0,
        reason=None if available else (cupy_reason or "no_cuda_runtime"),
    )


def cuda_status(*, refresh: bool = False) -> CudaStatus:
    global _STATUS
    with _STATUS_LOCK:
        if refresh or _STATUS is None or _STATUS.mode != _mode():
            _STATUS = _detect_status()
        return _STATUS


def cuda_metadata() -> dict[str, object]:
    """Return scalar-only runtime provenance suitable for JSON reports."""

    return cuda_status().as_dict()


def _use_cuda(nbytes: int) -> bool:
    status = cuda_status()
    if not status.available:
        return False
    return status.mode == "required" or int(nbytes) >= _MINIMUM_AUTO_BYTES


def _opencv_cuda_remap(
    source: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_mode: int,
    border_value: object,
) -> np.ndarray:
    source_gpu = cv2.cuda_GpuMat()
    map_x_gpu = cv2.cuda_GpuMat()
    map_y_gpu = cv2.cuda_GpuMat()
    source_gpu.upload(source)
    map_x_gpu.upload(map_x)
    map_y_gpu.upload(map_y)
    result_gpu = cv2.cuda.remap(
        source_gpu,
        map_x_gpu,
        map_y_gpu,
        interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    )
    _COUNTERS["opencv_cuda_calls"] += 1
    _COUNTERS["host_to_device_bytes"] += (
        int(source.nbytes) + int(map_x.nbytes) + int(map_y.nbytes)
    )
    result = result_gpu.download()
    _COUNTERS["device_to_host_bytes"] += int(result.nbytes)
    return result


def _cupy_remap_kernel(dtype: np.dtype[Any]) -> Any:
    key = np.dtype(dtype).str
    if key in _REMAP_KERNELS:
        return _REMAP_KERNELS[key]
    if np.dtype(dtype) == np.dtype(np.uint8):
        scalar, output, convert = (
            "unsigned char",
            "unsigned char",
            "out[out_i] = (unsigned char)min(255.0f, max(0.0f, nearbyintf(value)));",
        )
    elif np.dtype(dtype) == np.dtype(np.float32):
        scalar, output, convert = "float", "float", "out[out_i] = value;"
    else:
        raise TypeError(f"CuPy remap does not support {dtype}")
    source = f"""
    extern "C" __global__
    void remap(const {scalar}* src, const int src_h, const int src_w,
               const int channels, const float* mx, const float* my,
               const int out_h, const int out_w, const int linear,
               const float border, {output}* out) {{
        int pixel = blockDim.x * blockIdx.x + threadIdx.x;
        int count = out_h * out_w;
        if (pixel >= count) return;
        float x = mx[pixel];
        float y = my[pixel];
        for (int c = 0; c < channels; ++c) {{
            int out_i = pixel * channels + c;
            float value = border;
            if (isfinite(x) && isfinite(y)) {{
                if (!linear) {{
                    int ix = (int)nearbyintf(x);
                    int iy = (int)nearbyintf(y);
                    if (ix >= 0 && ix < src_w && iy >= 0 && iy < src_h)
                        value = (float)src[(iy * src_w + ix) * channels + c];
                }} else {{
                    int x0 = (int)floorf(x);
                    int y0 = (int)floorf(y);
                    float ax = x - (float)x0;
                    float ay = y - (float)y0;
                    value = 0.0f;
                    for (int dy = 0; dy < 2; ++dy) {{
                        int sy = y0 + dy;
                        float wy = dy ? ay : 1.0f - ay;
                        for (int dx = 0; dx < 2; ++dx) {{
                            int sx = x0 + dx;
                            float wx = dx ? ax : 1.0f - ax;
                            float sample = border;
                            if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h)
                                sample = (float)src[(sy * src_w + sx) * channels + c];
                            value += sample * wx * wy;
                        }}
                    }}
                }}
            }}
            {convert}
        }}
    }}
    """
    kernel = _CUPY.RawKernel(source, "remap")
    _REMAP_KERNELS[key] = kernel
    return kernel


def _cupy_remap(
    source: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    border_value: object,
) -> np.ndarray:
    cp = _CUPY
    if cp is None:  # pragma: no cover - guarded by status
        raise RuntimeError("CuPy CUDA backend is unavailable")
    src = np.ascontiguousarray(source)
    mx = np.ascontiguousarray(map_x, dtype=np.float32)
    my = np.ascontiguousarray(map_y, dtype=np.float32)
    channels = 1 if src.ndim == 2 else int(src.shape[2])
    if channels not in {1, 3, 4}:
        raise TypeError("CUDA remap supports one, three, or four channels")
    if isinstance(border_value, tuple):
        nonzero = any(float(item) != 0.0 for item in border_value)
        border = float(border_value[0]) if not nonzero else np.nan
    else:
        border = float(border_value)
    if not np.isfinite(border) and not (
        src.dtype == np.dtype(np.float32) and np.isnan(border)
    ):
        raise TypeError("CUDA remap requires a scalar/equal-channel border value")
    src_gpu, mx_gpu, my_gpu = cp.asarray(src), cp.asarray(mx), cp.asarray(my)
    shape = mx.shape if channels == 1 else (*mx.shape, channels)
    out_gpu = cp.empty(shape, dtype=src.dtype)
    count = int(mx.size)
    kernel = _cupy_remap_kernel(src.dtype)
    kernel(
        ((count + 255) // 256,),
        (256,),
        (
            src_gpu,
            np.int32(src.shape[0]),
            np.int32(src.shape[1]),
            np.int32(channels),
            mx_gpu,
            my_gpu,
            np.int32(mx.shape[0]),
            np.int32(mx.shape[1]),
            np.int32(interpolation == cv2.INTER_LINEAR),
            np.float32(border),
            out_gpu,
        ),
    )
    result = cp.asnumpy(out_gpu)
    _COUNTERS["cupy_calls"] += 1
    _COUNTERS["host_to_device_bytes"] += int(src.nbytes + mx.nbytes + my.nbytes)
    _COUNTERS["device_to_host_bytes"] += int(result.nbytes)
    return result


def remap(
    source: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    interpolation: int,
    *,
    borderMode: int = cv2.BORDER_CONSTANT,
    borderValue: object = 0,
) -> np.ndarray:
    """CUDA-accelerated ``cv2.remap`` subset with a reference CPU fallback."""

    src = np.asarray(source)
    mx = np.asarray(map_x, dtype=np.float32)
    my = np.asarray(map_y, dtype=np.float32)
    supported = (
        src.dtype in {np.dtype(np.uint8), np.dtype(np.float32)}
        and src.ndim in {2, 3}
        and mx.ndim == 2
        and mx.shape == my.shape
        and interpolation in {cv2.INTER_NEAREST, cv2.INTER_LINEAR}
        and borderMode == cv2.BORDER_CONSTANT
    )
    status = cuda_status()
    if supported and _use_cuda(src.nbytes + mx.nbytes + my.nbytes):
        decision_key = (
            "remap",
            src.dtype.str,
            src.ndim,
            1 if src.ndim == 2 else int(src.shape[2]),
            mx.shape,
            interpolation,
        )
        if status.mode == "auto" and _AUTO_DECISIONS.get(decision_key) == "cpu":
            _COUNTERS["cpu_calls"] += 1
            return cv2.remap(
                src,
                mx,
                my,
                interpolation,
                borderMode=borderMode,
                borderValue=borderValue,
            )
        gpu_result: np.ndarray | None = None
        gpu_elapsed = float("inf")
        gpu_error: Exception | None = None
        started = time.perf_counter()
        if status.opencv_cuda_available:
            try:
                gpu_result = _opencv_cuda_remap(
                    np.ascontiguousarray(src),
                    mx,
                    my,
                    interpolation,
                    borderMode,
                    borderValue,
                )
                gpu_elapsed = time.perf_counter() - started
            except cv2.error as exc:
                gpu_error = exc
                if status.mode == "required" and not status.cupy_available:
                    raise
        if gpu_result is None and status.cupy_available:
            started = time.perf_counter()
            try:
                gpu_result = _cupy_remap(src, mx, my, interpolation, borderValue)
                gpu_elapsed = time.perf_counter() - started
            except (TypeError, ValueError) as exc:
                gpu_error = exc
                if status.mode == "required":
                    raise
        if gpu_result is not None and status.mode == "required":
            return gpu_result
        if gpu_result is not None and status.mode == "auto":
            started = time.perf_counter()
            cpu_result = cv2.remap(
                src,
                mx,
                my,
                interpolation,
                borderMode=borderMode,
                borderValue=borderValue,
            )
            cpu_elapsed = time.perf_counter() - started
            if np.issubdtype(cpu_result.dtype, np.integer):
                parity = bool(
                    np.max(
                        np.abs(
                            cpu_result.astype(np.int64)
                            - gpu_result.astype(np.int64)
                        ),
                        initial=0,
                    )
                    <= (1 if interpolation == cv2.INTER_LINEAR else 0)
                )
            else:
                parity = bool(
                    np.allclose(
                        cpu_result,
                        gpu_result,
                        rtol=1e-5,
                        atol=1e-5,
                        equal_nan=True,
                    )
                )
            selected = (
                "cuda"
                if parity and gpu_elapsed < cpu_elapsed * 0.95
                else "cpu"
            )
            _AUTO_DECISIONS[decision_key] = selected
            if selected == "cuda":
                return gpu_result
            _COUNTERS["cpu_calls"] += 1
            return cpu_result
        if status.mode == "required" and gpu_error is not None:
            raise RuntimeError("CUDA remap failed") from gpu_error
    _COUNTERS["cpu_calls"] += 1
    return cv2.remap(
        src,
        mx,
        my,
        interpolation,
        borderMode=borderMode,
        borderValue=borderValue,
    )


def pinhole_unproject(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Unproject a pinhole pixel batch on CUDA when it is large enough."""

    uu = np.asarray(u, dtype=np.float64).reshape(-1)
    vv = np.asarray(v, dtype=np.float64).reshape(-1)
    zz = np.asarray(depth, dtype=np.float64).reshape(-1)
    if _CUPY is not None and _use_cuda(uu.nbytes + vv.nbytes + zz.nbytes):
        cp = _CUPY
        u_gpu, v_gpu, z_gpu = cp.asarray(uu), cp.asarray(vv), cp.asarray(zz)
        result = cp.stack(
            ((u_gpu - cx) * z_gpu / fx, (v_gpu - cy) * z_gpu / fy, z_gpu),
            axis=1,
        )
        _COUNTERS["cupy_calls"] += 1
        _COUNTERS["host_to_device_bytes"] += int(uu.nbytes + vv.nbytes + zz.nbytes)
        output = cp.asnumpy(result)
        _COUNTERS["device_to_host_bytes"] += int(output.nbytes)
        return output
    _COUNTERS["cpu_calls"] += 1
    return np.column_stack(((uu - cx) * zz / fx, (vv - cy) * zz / fy, zz))


def pinhole_project(
    points: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a pinhole 3-D batch on CUDA when it is large enough."""

    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if _CUPY is not None and _use_cuda(values.nbytes):
        cp = _CUPY
        points_gpu = cp.asarray(values)
        positive_z = cp.maximum(points_gpu[:, 2], 1e-12)
        x_gpu = fx * points_gpu[:, 0] / positive_z + cx
        y_gpu = fy * points_gpu[:, 1] / positive_z + cy
        x, y = cp.asnumpy(x_gpu), cp.asnumpy(y_gpu)
        _COUNTERS["cupy_calls"] += 1
        _COUNTERS["host_to_device_bytes"] += int(values.nbytes)
        _COUNTERS["device_to_host_bytes"] += int(x.nbytes + y.nbytes)
        return x, y
    _COUNTERS["cpu_calls"] += 1
    positive_z = np.maximum(values[:, 2], 1e-12)
    return (
        fx * values[:, 0] / positive_z + cx,
        fy * values[:, 1] / positive_z + cy,
    )


def transform_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Apply ``points @ rotation.T + translation`` on CUDA for large batches."""

    values = np.asarray(points, dtype=np.float64)
    rot = np.asarray(rotation, dtype=np.float64)
    offset = np.asarray(translation, dtype=np.float64)
    if _CUPY is not None and _use_cuda(values.nbytes):
        cp = _CUPY
        output_gpu = cp.asarray(values) @ cp.asarray(rot).T + cp.asarray(offset)
        output = cp.asnumpy(output_gpu)
        _COUNTERS["cupy_calls"] += 1
        _COUNTERS["host_to_device_bytes"] += int(
            values.nbytes + rot.nbytes + offset.nbytes
        )
        _COUNTERS["device_to_host_bytes"] += int(output.nbytes)
        return output
    _COUNTERS["cpu_calls"] += 1
    return values @ rot.T + offset
