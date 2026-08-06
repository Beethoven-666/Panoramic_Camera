"""Candidate-only PyTorch resident-frame runtime for video experiments.

This module is deliberately a *foundation*, not a renderer.  It owns the
single logical host-to-device upload of a real RGB-D source and exposes the
three CUDA streams needed by a future tiled renderer.  In particular, it does
not synthesize source frames, poses, colours, or pixel owners, and it is not
imported by the photo pipeline or the public production entry point.

When CUDA is not available, the same contract can be exercised on CPU for
tests and development.  The audit calls that path ``torch_cpu_fallback`` and
sets every CUDA/H2D counter to zero; callers must never report it as CUDA.
"""

from __future__ import annotations

import importlib
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Iterator, Literal

import numpy as np


class VideoGpuRuntimeError(RuntimeError):
    """Raised when the resident candidate-frame contract would be violated."""


@dataclass(frozen=True)
class GpuVideoFrame:
    """One real RGB-D source resident in the selected Torch device.

    ``color_u8`` remains the decoded source colour.  ``color_linear`` is
    derived on that same device, so its creation does not constitute a second
    host-to-device upload.  Depth is always in millimetres and ``pose_prior``
    is only a supplied prior; neither field is a generated source or pose.
    """

    frame_id: int
    timestamp_us: int
    color_u8: Any
    color_linear: Any
    depth_mm: Any
    depth_valid: Any
    pose_prior: Any
    # Immutable source-frame annotation raster, if an experimental object-lock
    # route supplied one.  It crosses the device boundary together with this
    # real source exactly once and has no colour or pose authority.
    object_mask: Any | None = None


@dataclass(frozen=True)
class VideoGpuRuntimeConfig:
    """Immutable runtime policy for one candidate experiment.

    ``maximum_resident_frames`` is a per-run resource bound.  Most local
    routes use 1--5 frames, while an audited global photometric graph may
    derive a larger bound from its complete real-source sequence.  A caller
    that needs an evicted source again must plan before release: re-uploading
    an already seen source is rejected to preserve the one logical
    H2D-upload-per-source invariant.
    """

    maximum_resident_frames: int = 5
    cuda_mode: Literal["prefer", "required", "off"] = "prefer"
    cuda_device: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.maximum_resident_frames, int)
            or isinstance(self.maximum_resident_frames, bool)
            or self.maximum_resident_frames < 1
        ):
            raise ValueError("maximum_resident_frames must be a positive integer")
        if self.cuda_mode not in {"prefer", "required", "off"}:
            raise ValueError("cuda_mode must be one of: prefer, required, off")
        if (
            not isinstance(self.cuda_device, int)
            or isinstance(self.cuda_device, bool)
            or self.cuda_device < 0
        ):
            raise ValueError("cuda_device must be a non-negative integer")


@dataclass(frozen=True)
class VideoGpuStreams:
    """The upload, compute, and output streams of an actual CUDA runtime.

    CPU fallback intentionally has ``None`` for all three fields.  This makes
    it impossible for a consumer of the audit to mistake a synchronous CPU
    run for CUDA stream overlap.
    """

    upload_stream: Any | None
    compute_stream: Any | None
    output_stream: Any | None


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise VideoGpuRuntimeError(
            "Candidate GPU runtime requires the optional torch runtime"
        ) from exc


def _real_rgb(value: np.ndarray, *, label: str) -> np.ndarray:
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise VideoGpuRuntimeError(f"{label} must be a decoded uint8 HxWx3 RGB source frame")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise VideoGpuRuntimeError(f"{label} cannot be empty")
    return np.ascontiguousarray(image)


def _real_depth(value: np.ndarray, *, height: int, width: int) -> np.ndarray:
    depth = np.asarray(value)
    if depth.ndim != 2 or depth.shape != (height, width):
        raise VideoGpuRuntimeError("depth_mm must be a HxW array matching color_u8")
    if depth.dtype not in (np.dtype(np.uint16), np.dtype(np.float32), np.dtype(np.float64)):
        raise VideoGpuRuntimeError("depth_mm must use uint16, float32, or float64 millimetres")
    numeric = np.ascontiguousarray(depth, dtype=np.float32)
    if np.isneginf(numeric).any():
        raise VideoGpuRuntimeError("depth_mm cannot contain negative infinity")
    return numeric


def _real_object_mask(value: np.ndarray | None, *, height: int, width: int) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.ndim != 2 or mask.shape != (height, width) or mask.dtype != np.bool_:
        raise VideoGpuRuntimeError("object_mask must be a bool HxW source-frame annotation raster")
    return np.ascontiguousarray(mask)


def _pose_matrix(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value)
    if pose.shape != (4, 4) or pose.dtype.kind not in {"f", "i", "u"}:
        raise VideoGpuRuntimeError("pose_prior must be a finite numeric 4x4 matrix")
    pose = np.ascontiguousarray(pose, dtype=np.float32)
    if not np.isfinite(pose).all():
        raise VideoGpuRuntimeError("pose_prior must be finite")
    return pose


class ResidentVideoFrameCache:
    """Bounded candidate-only cache with audited single logical uploads.

    The cache uses LRU eviction only for sources that have already completed
    their local window.  An evicted frame remains in the seen-source ledger,
    so attempting to upload it a second time fails closed instead of silently
    violating the one-upload invariant.
    """

    def __init__(
        self,
        config: VideoGpuRuntimeConfig = VideoGpuRuntimeConfig(),
        *,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config
        self._torch = torch_module or _load_torch()
        self._device, self._execution_backend, self._fallback_reason = self._select_device()
        self._streams = self._create_streams()
        self._frames: OrderedDict[int, GpuVideoFrame] = OrderedDict()
        self._uploaded_frame_ids: set[int] = set()
        self._per_source_h2d_uploads: dict[int, int] = {}
        self._per_source_h2d_bytes: dict[int, int] = {}
        self._logical_frame_upload_count = 0
        self._cpu_materialization_count = 0
        self._h2d_tensor_copy_count = 0
        self._h2d_total_bytes = 0
        self._eviction_count = 0
        self._final_d2h_copy_count = 0
        self._final_d2h_bytes = 0
        self._upload_event: Any | None = None
        self._compute_event: Any | None = None
        self._output_event: Any | None = None
        self._upload_event_record_count = 0
        self._compute_event_record_count = 0
        self._output_event_record_count = 0
        self._compute_wait_upload_event_count = 0
        self._output_wait_compute_event_count = 0
        self._final_output_stream_synchronize_count = 0
        self._gpu_memory_audit = self._start_gpu_memory_audit()
        self._closed = False

    def _select_device(self) -> tuple[Any, str, str | None]:
        cuda_available = False
        if self.config.cuda_mode != "off":
            try:
                cuda_available = bool(self._torch.cuda.is_available())
            except Exception:
                cuda_available = False
        if cuda_available:
            try:
                device_count = int(self._torch.cuda.device_count())
            except Exception as exc:
                raise VideoGpuRuntimeError("Unable to enumerate CUDA devices") from exc
            if self.config.cuda_device >= device_count:
                raise VideoGpuRuntimeError(
                    f"Requested CUDA device {self.config.cuda_device} is unavailable"
                )
            return self._torch.device(f"cuda:{self.config.cuda_device}"), "torch_cuda_resident", None
        if self.config.cuda_mode == "required":
            raise VideoGpuRuntimeError("CUDA is required for this candidate GPU runtime")
        reason = "cuda_disabled_by_policy" if self.config.cuda_mode == "off" else "cuda_unavailable"
        return self._torch.device("cpu"), "torch_cpu_fallback", reason

    def _create_streams(self) -> VideoGpuStreams:
        if self.cuda_active:
            try:
                return VideoGpuStreams(
                    upload_stream=self._torch.cuda.Stream(device=self._device),
                    compute_stream=self._torch.cuda.Stream(device=self._device),
                    output_stream=self._torch.cuda.Stream(device=self._device),
                )
            except Exception as exc:
                raise VideoGpuRuntimeError("Unable to create candidate CUDA streams") from exc
        return VideoGpuStreams(None, None, None)

    def _start_gpu_memory_audit(self) -> dict[str, object]:
        """Start a per-runtime peak-memory measurement when CUDA exposes it.

        CUDA peak counters are process/device scoped.  Resetting them once at
        construction lets the audit state exactly what its peak represents;
        if an unusual Torch build does not expose these counters, the runtime
        records that fact instead of inventing a memory value.
        """

        audit: dict[str, object] = {
            "available": False,
            "scope": "not_applicable",
            "baseline_allocated_bytes": None,
            "baseline_reserved_bytes": None,
            "current_allocated_bytes": None,
            "current_reserved_bytes": None,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "peak_increment_allocated_bytes": None,
            "peak_increment_reserved_bytes": None,
        }
        if not self.cuda_active:
            return audit
        try:
            with self._cuda_device_context():
                baseline_allocated = int(self._torch.cuda.memory_allocated(self._device))
                baseline_reserved = int(self._torch.cuda.memory_reserved(self._device))
                self._torch.cuda.reset_peak_memory_stats(self._device)
            audit.update(
                {
                    "available": True,
                    "scope": "process_device_since_runtime_initialization_reset",
                    "baseline_allocated_bytes": baseline_allocated,
                    "baseline_reserved_bytes": baseline_reserved,
                }
            )
        except Exception as exc:
            audit["unavailable_reason"] = type(exc).__name__
        return audit

    def _gpu_memory_audit_snapshot(self) -> dict[str, object]:
        audit = dict(self._gpu_memory_audit)
        if not bool(audit["available"]):
            return audit
        try:
            with self._cuda_device_context():
                current_allocated = int(self._torch.cuda.memory_allocated(self._device))
                current_reserved = int(self._torch.cuda.memory_reserved(self._device))
                peak_allocated = int(self._torch.cuda.max_memory_allocated(self._device))
                peak_reserved = int(self._torch.cuda.max_memory_reserved(self._device))
            baseline_allocated = int(audit["baseline_allocated_bytes"])
            baseline_reserved = int(audit["baseline_reserved_bytes"])
            audit.update(
                {
                    "current_allocated_bytes": current_allocated,
                    "current_reserved_bytes": current_reserved,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "peak_increment_allocated_bytes": max(
                        0, peak_allocated - baseline_allocated
                    ),
                    "peak_increment_reserved_bytes": max(
                        0, peak_reserved - baseline_reserved
                    ),
                }
            )
        except Exception as exc:
            audit["available"] = False
            audit["scope"] = "unavailable_after_runtime_initialization"
            audit["unavailable_reason"] = type(exc).__name__
        return audit

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def cuda_active(self) -> bool:
        return self._execution_backend == "torch_cuda_resident"

    @property
    def torch_module(self) -> Any:
        """Expose the selected Torch runtime without leaking private state."""

        return self._torch

    @property
    def streams(self) -> VideoGpuStreams:
        return self._streams

    @property
    def resident_frame_ids(self) -> tuple[int, ...]:
        return tuple(self._frames.keys())

    def get(self, frame_id: int) -> GpuVideoFrame:
        self._assert_open()
        try:
            frame = self._frames.pop(frame_id)
        except KeyError as exc:
            raise VideoGpuRuntimeError(f"frame {frame_id} is not resident") from exc
        self._frames[frame_id] = frame
        return frame

    def upload(
        self,
        *,
        frame_id: int,
        timestamp_us: int,
        color_u8: np.ndarray,
        depth_mm: np.ndarray,
        pose_prior: np.ndarray,
        object_mask: np.ndarray | None = None,
    ) -> GpuVideoFrame:
        """Upload a new concrete source once and return its resident tensors."""

        self._assert_open()
        self._validate_ids(frame_id, timestamp_us)
        if frame_id in self._frames:
            return self.get(frame_id)
        if frame_id in self._uploaded_frame_ids:
            raise VideoGpuRuntimeError(
                f"frame {frame_id} was already evicted; re-upload would exceed H2D <= 1"
            )
        rgb = _real_rgb(color_u8, label="color_u8")
        depth = _real_depth(depth_mm, height=rgb.shape[0], width=rgb.shape[1])
        objects = _real_object_mask(object_mask, height=rgb.shape[0], width=rgb.shape[1])
        pose = _pose_matrix(pose_prior)
        h2d_bytes_before = self._h2d_total_bytes
        with self.upload_context():
            frame = self._make_frame(frame_id, timestamp_us, rgb, depth, pose, objects)
        self._uploaded_frame_ids.add(frame_id)
        self._per_source_h2d_uploads[frame_id] = 1 if self.cuda_active else 0
        self._per_source_h2d_bytes[frame_id] = self._h2d_total_bytes - h2d_bytes_before
        self._logical_frame_upload_count += 1
        if not self.cuda_active:
            self._cpu_materialization_count += 1
        self._frames[frame_id] = frame
        self._evict_if_needed()
        return frame

    def _make_frame(
        self,
        frame_id: int,
        timestamp_us: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        object_mask: np.ndarray | None,
    ) -> GpuVideoFrame:
        torch = self._torch
        # A source first crosses the device boundary as source tensors.  The
        # linear RGB and validity mask are then derived on-device, never by a
        # second NumPy round-trip.
        color = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
        raw_depth = torch.from_numpy(depth)
        raw_pose = torch.from_numpy(pose)
        raw_object_mask = None if object_mask is None else torch.from_numpy(object_mask)
        if self.cuda_active:
            color = self._to_device_once(color)
            raw_depth = self._to_device_once(raw_depth)
            raw_pose = self._to_device_once(raw_pose)
            if raw_object_mask is not None:
                raw_object_mask = self._to_device_once(raw_object_mask)
        color = color.contiguous()
        depth_mm = raw_depth.to(dtype=torch.float32).contiguous()
        depth_valid = torch.isfinite(depth_mm) & (depth_mm > 0.0)
        rgb_float = color.to(dtype=torch.float32).div(255.0)
        color_linear = torch.where(
            rgb_float <= 0.04045,
            rgb_float / 12.92,
            torch.pow((rgb_float + 0.055) / 1.055, 2.4),
        ).contiguous()
        return GpuVideoFrame(
            frame_id=frame_id,
            timestamp_us=timestamp_us,
            color_u8=color,
            color_linear=color_linear,
            depth_mm=depth_mm,
            depth_valid=depth_valid,
            pose_prior=raw_pose.to(dtype=torch.float32).contiguous(),
            object_mask=None if raw_object_mask is None else raw_object_mask.bool().contiguous(),
        )

    def _to_device_once(self, tensor: Any) -> Any:
        if tensor.device.type != "cpu":
            raise VideoGpuRuntimeError("candidate source upload must originate from CPU memory")
        byte_count = int(tensor.numel() * tensor.element_size())
        try:
            # Pinning is optional at this boundary; a platform that cannot
            # pin remains correct and is still audited as the same source
            # transaction rather than falling back to CPU invisibly.
            if not tensor.is_pinned():
                tensor = tensor.pin_memory()
            uploaded = tensor.to(self._device, non_blocking=True)
        except Exception as exc:
            raise VideoGpuRuntimeError("Unable to upload candidate source to CUDA") from exc
        if uploaded.device.type != "cuda" or uploaded.device.index != self.config.cuda_device:
            raise VideoGpuRuntimeError("candidate source upload reached an unexpected CUDA device")
        self._h2d_tensor_copy_count += 1
        self._h2d_total_bytes += byte_count
        return uploaded

    def _evict_if_needed(self) -> None:
        while len(self._frames) > self.config.maximum_resident_frames:
            self._frames.popitem(last=False)
            self._eviction_count += 1

    def release(self, frame_id: int) -> None:
        """Release one completed window source without permitting re-upload."""

        self._assert_open()
        self._frames.pop(frame_id, None)

    @contextmanager
    def upload_context(self) -> Iterator[None]:
        with self._stream_context(self._streams.upload_stream):
            yield
            if self.cuda_active:
                assert self._streams.upload_stream is not None
                self._upload_event = self._record_event(self._streams.upload_stream)
                self._upload_event_record_count += 1

    @contextmanager
    def compute_context(self) -> Iterator[None]:
        if self.cuda_active:
            assert self._streams.compute_stream is not None
            if self._upload_event is None:
                raise VideoGpuRuntimeError(
                    "compute requires a successfully recorded upload event"
                )
        with self._stream_context(self._streams.compute_stream):
            if self.cuda_active:
                assert self._streams.compute_stream is not None
                assert self._upload_event is not None
                # Link this compute window to the last completed upload point,
                # rather than waiting for the mutable upload stream itself.
                # Upload can therefore continue past this event while compute
                # runs, without a host-side synchronization.
                self._streams.compute_stream.wait_event(self._upload_event)
                self._compute_wait_upload_event_count += 1
            yield
            if self.cuda_active:
                assert self._streams.compute_stream is not None
                self._compute_event = self._record_event(self._streams.compute_stream)
                self._compute_event_record_count += 1

    @contextmanager
    def output_context(self) -> Iterator[None]:
        if self.cuda_active:
            assert self._streams.compute_stream is not None
            assert self._streams.output_stream is not None
            if self._compute_event is None:
                raise VideoGpuRuntimeError(
                    "output requires a successfully recorded compute event"
                )
        with self._stream_context(self._streams.output_stream):
            if self.cuda_active:
                assert self._streams.output_stream is not None
                assert self._compute_event is not None
                self._streams.output_stream.wait_event(self._compute_event)
                self._output_wait_compute_event_count += 1
            yield
            if self.cuda_active:
                assert self._streams.output_stream is not None
                self._output_event = self._record_event(self._streams.output_stream)
                self._output_event_record_count += 1

    def _record_event(self, stream: Any) -> Any:
        try:
            event = self._torch.cuda.Event()
            event.record(stream)
            return event
        except Exception as exc:
            raise VideoGpuRuntimeError("Unable to record candidate CUDA stream event") from exc

    @contextmanager
    def _cuda_device_context(self) -> Iterator[None]:
        if not self.cuda_active:
            with nullcontext():
                yield
            return
        with self._torch.cuda.device(self._device):
            yield

    @contextmanager
    def _stream_context(self, stream: Any | None) -> Iterator[None]:
        if stream is None:
            with nullcontext():
                yield
        else:
            with self._cuda_device_context():
                with self._torch.cuda.stream(stream):
                    yield

    def copy_final_to_cpu(
        self,
        tensor: Any,
        *,
        artifact: Literal["panorama", "provenance", "scalar_audit"],
    ) -> Any:
        """Permit the only controlled device-to-host boundary.

        Rendering intermediates are intentionally not accepted here.  CPU
        fallback returns a detached CPU tensor without incrementing D2H; that
        distinction appears in the audit rather than being presented as a GPU
        download.
        """

        self._assert_open()
        if artifact not in {"panorama", "provenance", "scalar_audit"}:
            raise VideoGpuRuntimeError("only final panorama/provenance/scalar audit downloads are allowed")
        if not hasattr(tensor, "detach") or not hasattr(tensor, "device"):
            raise VideoGpuRuntimeError("copy_final_to_cpu requires a Torch tensor")
        detached = tensor.detach()
        if self.cuda_active:
            if (
                detached.device.type != "cuda"
                or detached.device.index != self.config.cuda_device
            ):
                raise VideoGpuRuntimeError(
                    "final CUDA download must originate on this runtime's selected device"
                )
            with self.output_context():
                result = detached.to("cpu")
            # A final output synchronisation is a permitted explicit boundary.
            assert self._streams.output_stream is not None
            self._streams.output_stream.synchronize()
            self._final_output_stream_synchronize_count += 1
            self._final_d2h_copy_count += 1
            self._final_d2h_bytes += int(result.numel() * result.element_size())
            return result
        if detached.device.type != "cpu":
            raise VideoGpuRuntimeError(
                "CPU fallback cannot download a tensor from an unowned CUDA device"
            )
        return detached.cpu()

    def audit(self) -> dict[str, object]:
        """Return measured cache/transfer facts without claiming unavailable CUDA."""

        return {
            "schema": "gemini305-video-gpu-runtime/v2",
            "candidate_only": True,
            "execution_backend": self._execution_backend,
            "cuda_active": self.cuda_active,
            "cuda_fallback_reason": self._fallback_reason,
            "device": self.device,
            "streams": {
                "upload": self.cuda_active,
                "compute": self.cuda_active,
                "output": self.cuda_active,
                "overlap_enabled": self.cuda_active,
            },
            "maximum_resident_frames": self.config.maximum_resident_frames,
            "resident_frame_ids": list(self.resident_frame_ids),
            "resident_frame_count": len(self._frames),
            "eviction_count": self._eviction_count,
            "logical_frame_upload_count": self._logical_frame_upload_count,
            "per_source_h2d_upload_count": dict(sorted(self._per_source_h2d_uploads.items())),
            "per_source_h2d_bytes": dict(sorted(self._per_source_h2d_bytes.items())),
            "h2d_frame_upload_count": sum(self._per_source_h2d_uploads.values()),
            "h2d_tensor_copy_count": self._h2d_tensor_copy_count,
            "h2d_total_bytes": self._h2d_total_bytes,
            "cpu_materialization_count": self._cpu_materialization_count,
            "intermediate_d2h_count": 0,
            "final_d2h_copy_count": self._final_d2h_copy_count,
            "final_d2h_bytes": self._final_d2h_bytes,
            "stream_events": {
                "dependency_mode": "cuda_event_chain" if self.cuda_active else "none",
                "upload_event_record_count": self._upload_event_record_count,
                "compute_wait_upload_event_count": self._compute_wait_upload_event_count,
                "compute_event_record_count": self._compute_event_record_count,
                "output_wait_compute_event_count": self._output_wait_compute_event_count,
                "output_event_record_count": self._output_event_record_count,
                "host_synchronization_count": self._final_output_stream_synchronize_count,
                "host_synchronization_scope": "final_d2h_only",
            },
            "gpu_memory": self._gpu_memory_audit_snapshot(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._frames.clear()
        self._closed = True

    def _assert_open(self) -> None:
        if self._closed:
            raise VideoGpuRuntimeError("resident frame cache is closed")

    @staticmethod
    def _validate_ids(frame_id: int, timestamp_us: int) -> None:
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise VideoGpuRuntimeError("frame_id must be a non-negative integer")
        if isinstance(timestamp_us, bool) or not isinstance(timestamp_us, int) or timestamp_us < 0:
            raise VideoGpuRuntimeError("timestamp_us must be a non-negative integer")


__all__ = [
    "GpuVideoFrame",
    "ResidentVideoFrameCache",
    "VideoGpuRuntimeConfig",
    "VideoGpuRuntimeError",
    "VideoGpuStreams",
]
