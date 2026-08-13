"""Optional CUDA primitives used by the RGB-D pipeline.

The project deliberately keeps CUDA behind a small boundary:

* ``G305_CUDA=prefer`` (default) uses CUDA whenever a supported implementation
  exists and falls back to the identical CPU operation only when necessary.
* ``G305_CUDA=auto`` benchmarks equivalent CPU/CUDA remaps per shape.
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
import sys
from threading import Lock
import time
from typing import Any

import cv2
import numpy as np


_VALID_MODES = {"prefer", "auto", "off", "required"}
_MINIMUM_AUTO_BYTES = 64 * 1024
_STATUS_LOCK = Lock()
_STATUS: "CudaStatus | None" = None
_CUPY: Any | None = None
_REMAP_KERNELS: dict[str, Any] = {}
_AUTO_DECISIONS: dict[tuple[object, ...], str] = {}
_FALLBACKS: list[dict[str, str]] = []
_CUDA_DLL_HANDLES: list[Any] = []
_CUDA_DLL_DIRECTORY: str | None = None
_CUDNN_DLL_DIRECTORY: str | None = None
_COUNTERS = {
    "open3d_cuda_calls": 0,
    "opencv_cuda_calls": 0,
    "cupy_calls": 0,
    "cpu_calls": 0,
    "host_to_device_bytes": 0,
    "device_to_host_bytes": 0,
    # Wall-clock operation accounting includes transfers and the required
    # host/device synchronisation at this public NumPy boundary.  It is not
    # presented as kernel-only CUDA time.
    "gpu_remap_wall_seconds": 0.0,
    "cpu_remap_wall_seconds": 0.0,
}


def record_open3d_cuda_call(
    *,
    host_to_device_bytes: int = 0,
    device_to_host_bytes: int = 0,
) -> None:
    """Record an audited Open3D Tensor CUDA operation.

    Open3D owns its device buffers, so it cannot use the CuPy/OpenCV wrappers
    below.  Keeping its calls in the same audit makes the final report reflect
    the complete CUDA path instead of only custom kernels.
    """

    _COUNTERS["open3d_cuda_calls"] += 1
    _COUNTERS["host_to_device_bytes"] += max(0, int(host_to_device_bytes))
    _COUNTERS["device_to_host_bytes"] += max(0, int(device_to_host_bytes))


def configure_cuda_dll_search_path() -> str | None:
    """Make the validated CUDA 12 runtime visible to Windows Python DLLs."""

    global _CUDA_DLL_DIRECTORY
    if os.name != "nt":
        return None
    configured = os.environ.get("G305_CUDA_TOOLKIT")
    candidates = [
        configured,
        r"D:\open3d_cuda_build\cuda128-toolkit\Library",
        os.environ.get("CUDA_PATH_V12_8"),
        os.environ.get("CUDA_PATH"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = os.path.abspath(os.path.expandvars(candidate))
        bin_dir = root if os.path.basename(root).lower() == "bin" else os.path.join(
            root, "bin"
        )
        if not os.path.isfile(os.path.join(bin_dir, "cudart64_12.dll")):
            continue
        current_path = os.environ.get("PATH", "")
        entries = current_path.split(os.pathsep) if current_path else []
        if os.path.normcase(bin_dir) not in {
            os.path.normcase(item) for item in entries
        }:
            os.environ["PATH"] = bin_dir + os.pathsep + current_path
        add_directory = getattr(os, "add_dll_directory", None)
        if callable(add_directory):
            _CUDA_DLL_HANDLES.append(add_directory(bin_dir))
        _CUDA_DLL_DIRECTORY = bin_dir
        return bin_dir
    return None


def configure_cudnn_dll_search_path() -> str | None:
    """Expose the CUDA-DNN runtime required by ONNX Runtime on Windows."""

    global _CUDNN_DLL_DIRECTORY
    if os.name != "nt":
        return None
    configured = os.environ.get("G305_CUDNN_DIR")
    candidates = [
        configured,
        os.path.join(
            sys.prefix, "Lib", "site-packages", "torch", "lib"
        ),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        directory = os.path.abspath(os.path.expandvars(candidate))
        if not os.path.isfile(os.path.join(directory, "cudnn64_9.dll")):
            continue
        current_path = os.environ.get("PATH", "")
        entries = current_path.split(os.pathsep) if current_path else []
        if os.path.normcase(directory) not in {
            os.path.normcase(item) for item in entries
        }:
            os.environ["PATH"] = directory + os.pathsep + current_path
        add_directory = getattr(os, "add_dll_directory", None)
        if callable(add_directory):
            _CUDA_DLL_HANDLES.append(add_directory(directory))
        _CUDNN_DLL_DIRECTORY = directory
        return directory
    return None


configure_cuda_dll_search_path()
configure_cudnn_dll_search_path()


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
        result["fallbacks"] = list(_FALLBACKS)
        result["cuda_dll_directory"] = _CUDA_DLL_DIRECTORY
        result["cudnn_dll_directory"] = _CUDNN_DLL_DIRECTORY
        return result


def _mode() -> str:
    value = os.environ.get("G305_CUDA", "prefer").strip().lower()
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


def reset_cuda_audit() -> None:
    """Reset per-run counters without destroying the initialized CUDA context."""

    for key in _COUNTERS:
        _COUNTERS[key] = 0
    _AUTO_DECISIONS.clear()
    _FALLBACKS.clear()


def _record_remap_timing(*, device: str, elapsed_seconds: float) -> None:
    key = "gpu_remap_wall_seconds" if device == "gpu" else "cpu_remap_wall_seconds"
    _COUNTERS[key] += max(0.0, float(elapsed_seconds))


def _use_cuda(nbytes: int) -> bool:
    status = cuda_status()
    if not status.available:
        return False
    if status.mode in {"prefer", "required"}:
        return True
    return int(nbytes) >= _MINIMUM_AUTO_BYTES


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
    elif np.dtype(dtype) == np.dtype(np.uint16):
        scalar, output, convert = (
            "unsigned short",
            "unsigned short",
            (
                "out[out_i] = (unsigned short)min(65535.0f, "
                "max(0.0f, nearbyintf(value)));"
            ),
        )
    else:
        raise TypeError(f"CuPy remap does not support {dtype}")
    source = f"""
    extern "C" __global__
    void remap(const {scalar}* src, const int src_h, const int src_w,
               const int channels, const float* mx, const float* my,
               const int out_h, const int out_w, const int linear,
               const int replicate, const float border, {output}* out) {{
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
                    if (replicate) {{
                        ix = min(src_w - 1, max(0, ix));
                        iy = min(src_h - 1, max(0, iy));
                    }}
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
                            if (replicate) {{
                                sx = min(src_w - 1, max(0, sx));
                                sy = min(src_h - 1, max(0, sy));
                            }}
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
    border_mode: int,
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
            np.int32(border_mode == cv2.BORDER_REPLICATE),
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
        src.dtype
        in {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)}
        and src.ndim in {2, 3}
        and mx.ndim == 2
        and mx.shape == my.shape
        and interpolation in {cv2.INTER_NEAREST, cv2.INTER_LINEAR}
        and borderMode in {cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE}
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
            borderMode,
        )
        if status.mode == "auto" and _AUTO_DECISIONS.get(decision_key) == "cpu":
            _COUNTERS["cpu_calls"] += 1
            started = time.perf_counter()
            result = cv2.remap(
                src,
                mx,
                my,
                interpolation,
                borderMode=borderMode,
                borderValue=borderValue,
            )
            _record_remap_timing(device="cpu", elapsed_seconds=time.perf_counter() - started)
            return result
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
            except Exception as exc:
                gpu_error = exc
                if status.mode == "required" and not status.cupy_available:
                    raise
        if gpu_result is None and status.cupy_available:
            started = time.perf_counter()
            try:
                gpu_result = _cupy_remap(
                    src,
                    mx,
                    my,
                    interpolation,
                    borderMode,
                    borderValue,
                )
                gpu_elapsed = time.perf_counter() - started
            except Exception as exc:
                gpu_error = exc
                if status.mode == "required":
                    raise
        if gpu_result is not None and status.mode == "required":
            _record_remap_timing(device="gpu", elapsed_seconds=gpu_elapsed)
            return gpu_result
        if gpu_result is not None and status.mode == "prefer":
            _record_remap_timing(device="gpu", elapsed_seconds=gpu_elapsed)
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
            _record_remap_timing(device="cpu", elapsed_seconds=cpu_elapsed)
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
                _record_remap_timing(device="gpu", elapsed_seconds=gpu_elapsed)
                return gpu_result
            _COUNTERS["cpu_calls"] += 1
            return cpu_result
        if status.mode == "required" and gpu_error is not None:
            raise RuntimeError("CUDA remap failed") from gpu_error
        if gpu_error is not None:
            _FALLBACKS.append(
                {
                    "operation": "remap",
                    "stage": "runtime",
                    "reason": f"{type(gpu_error).__name__}: {gpu_error}",
                }
            )
    elif status.mode == "required":
        raise RuntimeError(
            "CUDA remap was required but the source dtype, interpolation, "
            "border mode, or map shape is unsupported"
        )
    _COUNTERS["cpu_calls"] += 1
    started = time.perf_counter()
    result = cv2.remap(
        src,
        mx,
        my,
        interpolation,
        borderMode=borderMode,
        borderValue=borderValue,
    )
    _record_remap_timing(device="cpu", elapsed_seconds=time.perf_counter() - started)
    return result


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
    status = cuda_status()
    use_cuda = _use_cuda(uu.nbytes + vv.nbytes + zz.nbytes)
    if use_cuda and status.cupy_available and _CUPY is not None:
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
    if use_cuda and status.mode == "required":
        raise RuntimeError("CUDA pinhole unprojection requires a working CuPy device")
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
    status = cuda_status()
    use_cuda = _use_cuda(values.nbytes)
    if use_cuda and status.cupy_available and _CUPY is not None:
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
    if use_cuda and status.mode == "required":
        raise RuntimeError("CUDA pinhole projection requires a working CuPy device")
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
    status = cuda_status()
    use_cuda = _use_cuda(values.nbytes)
    if use_cuda and status.cupy_available and _CUPY is not None:
        cp = _CUPY
        output_gpu = cp.asarray(values) @ cp.asarray(rot).T + cp.asarray(offset)
        output = cp.asnumpy(output_gpu)
        _COUNTERS["cupy_calls"] += 1
        _COUNTERS["host_to_device_bytes"] += int(
            values.nbytes + rot.nbytes + offset.nbytes
        )
        _COUNTERS["device_to_host_bytes"] += int(output.nbytes)
        return output
    if use_cuda and status.mode == "required":
        raise RuntimeError("CUDA point transforms require a working CuPy device")
    _COUNTERS["cpu_calls"] += 1
    return values @ rot.T + offset


def srgb_to_linear_bgr(image: np.ndarray) -> np.ndarray:
    """Decode uint8/float BGR into float32 linear light on CUDA when possible."""

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("sRGB input must be an HxWx3 BGR array")
    status = cuda_status()
    use_cuda = _use_cuda(values.nbytes)
    if use_cuda and status.cupy_available and _CUPY is not None:
        cp = _CUPY
        encoded = cp.asarray(values, dtype=cp.float32) / cp.float32(255.0)
        result_gpu = cp.where(
            encoded <= cp.float32(0.04045),
            encoded / cp.float32(12.92),
            cp.power(
                (encoded + cp.float32(0.055)) / cp.float32(1.055),
                cp.float32(2.4),
            ),
        ).astype(cp.float32)
        result = cp.asnumpy(result_gpu)
        _COUNTERS["cupy_calls"] += 1
        _COUNTERS["host_to_device_bytes"] += int(values.nbytes)
        _COUNTERS["device_to_host_bytes"] += int(result.nbytes)
        return result
    if use_cuda and status.mode == "required":
        raise RuntimeError("CUDA sRGB decoding requires a working CuPy device")
    _COUNTERS["cpu_calls"] += 1
    encoded = np.asarray(values, dtype=np.float32) / 255.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb_bgr(linear: np.ndarray) -> np.ndarray:
    """Encode finite linear BGR samples to uint8 sRGB on CUDA when possible."""

    values = np.asarray(linear, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("Linear RGB input must be an HxWx3 array")
    status = cuda_status()
    use_cuda = _use_cuda(values.nbytes)
    if use_cuda and status.cupy_available and _CUPY is not None:
        cp = _CUPY
        linear_gpu = cp.clip(cp.asarray(values), 0.0, 1.0)
        encoded_gpu = cp.where(
            linear_gpu <= cp.float32(0.0031308),
            linear_gpu * cp.float32(12.92),
            cp.float32(1.055)
            * cp.power(linear_gpu, cp.float32(1.0 / 2.4))
            - cp.float32(0.055),
        )
        result_gpu = cp.rint(cp.clip(encoded_gpu * 255.0, 0.0, 255.0)).astype(
            cp.uint8
        )
        result = cp.asnumpy(result_gpu)
        _COUNTERS["cupy_calls"] += 1
        _COUNTERS["host_to_device_bytes"] += int(values.nbytes)
        _COUNTERS["device_to_host_bytes"] += int(result.nbytes)
        return result
    if use_cuda and status.mode == "required":
        raise RuntimeError("CUDA sRGB encoding requires a working CuPy device")
    _COUNTERS["cpu_calls"] += 1
    clipped = np.clip(values, 0.0, 1.0)
    encoded = np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded * 255.0, 0.0, 255.0)).astype(np.uint8)
