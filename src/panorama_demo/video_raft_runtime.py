"""Candidate-only, local-weight torchvision RAFT-small optical-flow runtime.

This module deliberately has no downloader and no link to a public video
entrypoint.  A caller must pass an *explicit existing local file* and its
expected SHA-256.  Consequently a missing, replaced, or unverifiable model is
an error, never a request to torchvision to fetch a checkpoint and never a
zero-flow fallback.

The runtime is intentionally limited to adjacent, real source-frame evidence.
It returns a flow field and provenance/audit data only; it cannot create a
render source, a pose, or output panorama colour.
"""

from __future__ import annotations

import hashlib
import importlib
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


_SHA256_HEX_LENGTH = 64


class RAFTSmallRuntimeError(RuntimeError):
    """Raised when candidate RAFT evidence cannot be obtained safely."""


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise RAFTSmallRuntimeError("RAFT-small weights SHA-256 must be 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RAFTSmallRuntimeError(
            "RAFT-small weights SHA-256 must be 64 hexadecimal characters"
        ) from exc
    return value.lower()


def sha256_file(path: str | Path) -> str:
    """Return the byte-for-byte SHA-256 of a local weight file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_raft_small_weights(path: str | Path, expected_sha256: str) -> Path:
    """Verify a supplied local RAFT checkpoint and return its resolved path.

    This is purposely a file-only boundary: URLs, torchvision weight enums and
    cache paths are not accepted.  A candidate must record the resulting
    digest in its immutable algorithm identity before it can use the runtime.
    """

    expected = _validate_sha256(expected_sha256)
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise RAFTSmallRuntimeError(
            f"RAFT-small local weights are required and were not found: {candidate}"
        )
    resolved = candidate.resolve()
    actual = sha256_file(resolved)
    if actual != expected:
        raise RAFTSmallRuntimeError(
            "RAFT-small local weights SHA-256 mismatch: "
            f"expected {expected}, received {actual} ({resolved})"
        )
    return resolved


@dataclass(frozen=True)
class RAFTSmallRuntimeConfig:
    """Immutable local model identity for one candidate run."""

    weights_path: Path
    weights_sha256: str
    cuda_device: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.cuda_device, int) or isinstance(self.cuda_device, bool) or self.cuda_device < 0:
            raise ValueError("cuda_device must be a non-negative integer")
        object.__setattr__(self, "weights_path", Path(self.weights_path).expanduser())
        object.__setattr__(self, "weights_sha256", _validate_sha256(self.weights_sha256))


@dataclass(frozen=True)
class RAFTSmallFlowAudit:
    """The complete local-model and inference provenance of one adjacent pair."""

    source_frame_id: int
    target_frame_id: int
    weights_path: str
    weights_sha256: str
    device: str
    precision: str
    input_height: int
    input_width: int
    padded_height: int
    padded_width: int
    output_height: int
    output_width: int
    finite: bool
    model: str = "torchvision_raft_small"
    downloaded: bool = False
    inference_wall_seconds: float = 0.0
    inference_device_seconds: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "device": self.device,
            "precision": self.precision,
            "input_shape": [self.input_height, self.input_width],
            "padded_shape": [self.padded_height, self.padded_width],
            "output_shape": [self.output_height, self.output_width],
            "flow_finite": self.finite,
            "downloaded": self.downloaded,
            "inference_wall_seconds": self.inference_wall_seconds,
            "inference_device_seconds": self.inference_device_seconds,
        }


@dataclass(frozen=True)
class RAFTSmallFlowResult:
    """Dense source-to-target displacement in source-image pixel coordinates."""

    flow_xy: np.ndarray
    audit: RAFTSmallFlowAudit


@dataclass(frozen=True)
class RAFTSmallTensorFlowAudit:
    """Audit for an already-resident tensor inference with no host download.

    ``flow_finite`` is deliberately not checked here: converting a CUDA
    reduction to ``bool`` would synchronise/download a scalar.  The caller
    may perform that small final audit at its explicitly permitted boundary.
    """

    source_frame_id: int
    target_frame_id: int
    weights_path: str
    weights_sha256: str
    device: str
    precision: str
    input_height: int
    input_width: int
    padded_height: int
    padded_width: int
    output_height: int
    output_width: int
    model: str = "torchvision_raft_small"
    downloaded: bool = False
    output_residency: str = "device_tensor"
    host_transfer_count: int = 0
    flow_finite: None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "source_frame_id": self.source_frame_id,
            "target_frame_id": self.target_frame_id,
            "weights_path": self.weights_path,
            "weights_sha256": self.weights_sha256,
            "device": self.device,
            "precision": self.precision,
            "input_shape": [self.input_height, self.input_width],
            "padded_shape": [self.padded_height, self.padded_width],
            "output_shape": [self.output_height, self.output_width],
            "flow_finite": self.flow_finite,
            "flow_finite_audit": "deferred_no_d2h",
            "output_residency": self.output_residency,
            "host_transfer_count": self.host_transfer_count,
            "downloaded": self.downloaded,
        }


@dataclass(frozen=True)
class RAFTSmallTensorFlowResult:
    """Adjacent flow retained on the Torch runtime device as HxWx2 float32."""

    flow_xy: Any
    audit: RAFTSmallTensorFlowAudit


def _load_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RAFTSmallRuntimeError(
            "Candidate RAFT-small requires the optional torch/torchvision runtime"
        ) from exc


def _torchvision_raft_small_factory() -> Callable[..., Any]:
    try:
        module = importlib.import_module("torchvision.models.optical_flow")
        return getattr(module, "raft_small")
    except (ImportError, AttributeError) as exc:
        raise RAFTSmallRuntimeError(
            "Candidate RAFT-small requires torchvision.models.optical_flow.raft_small"
        ) from exc


def _checkpoint_state_dict(payload: object) -> Mapping[str, object]:
    """Accept the two non-executable state-dict container forms we support."""

    if not isinstance(payload, Mapping):
        raise RAFTSmallRuntimeError("RAFT-small checkpoint must contain a state dictionary")
    nested = payload.get("state_dict")
    state = nested if isinstance(nested, Mapping) else payload
    if not state or not all(isinstance(key, str) for key in state):
        raise RAFTSmallRuntimeError("RAFT-small checkpoint state dictionary is invalid")
    return state


def _coerce_real_rgb_frame(image: np.ndarray, *, label: str) -> np.ndarray:
    """Validate a concrete decoded RGB frame before it reaches the model."""

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise RAFTSmallRuntimeError(f"{label} must be a decoded uint8 HxWx3 RGB source frame")
    if array.shape[0] < 8 or array.shape[1] < 8:
        raise RAFTSmallRuntimeError(f"{label} is too small for RAFT-small inference")
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def _padded_extent(length: int, multiple: int = 8) -> int:
    return ((length + multiple - 1) // multiple) * multiple


class TorchvisionRAFTSmallRuntime:
    """Load a verified local torchvision RAFT-small checkpoint once per run.

    Passing ``model_factory`` is an intentional test seam.  Production callers
    use the torchvision factory and always pass ``weights=None`` so torchvision
    is unable to download an enum checkpoint.
    """

    def __init__(
        self,
        config: RAFTSmallRuntimeConfig,
        *,
        model_factory: Callable[..., Any] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config
        self.weights_path = verify_local_raft_small_weights(
            config.weights_path, config.weights_sha256
        )
        self._torch = torch_module or _load_torch()
        self._device, self._precision = self._select_device()
        factory = model_factory or _torchvision_raft_small_factory()
        try:
            # ``None`` is critical: torchvision weight enums otherwise use a
            # URL/cache downloader that is forbidden for candidate execution.
            model = factory(weights=None, progress=False)
        except TypeError:
            # The narrow seam keeps a simple mock factory convenient while the
            # real torchvision factory always receives the explicit None above.
            model = factory()
        except Exception as exc:
            raise RAFTSmallRuntimeError("Unable to construct torchvision RAFT-small") from exc
        self._model = model
        self._load_verified_state_dict()
        try:
            self._model.to(self._device)
            self._model.eval()
        except Exception as exc:
            raise RAFTSmallRuntimeError("Unable to initialize RAFT-small on the selected device") from exc

    def _select_device(self) -> tuple[Any, str]:
        try:
            cuda_available = bool(self._torch.cuda.is_available())
        except Exception:
            cuda_available = False
        if cuda_available:
            try:
                count = int(self._torch.cuda.device_count())
            except Exception as exc:
                raise RAFTSmallRuntimeError("Unable to enumerate CUDA devices for RAFT-small") from exc
            if self.config.cuda_device >= count:
                raise RAFTSmallRuntimeError(
                    f"Requested CUDA device {self.config.cuda_device} is unavailable for RAFT-small"
                )
            return self._torch.device(f"cuda:{self.config.cuda_device}"), "float16"
        return self._torch.device("cpu"), "float32"

    def _load_verified_state_dict(self) -> None:
        try:
            # ``weights_only`` rejects arbitrary pickle code.  It is available
            # in supported torch versions; the fallback is deliberately absent.
            payload = self._torch.load(
                self.weights_path, map_location="cpu", weights_only=True
            )
            state = _checkpoint_state_dict(payload)
            result = self._model.load_state_dict(state, strict=True)
            missing = tuple(getattr(result, "missing_keys", ()))
            unexpected = tuple(getattr(result, "unexpected_keys", ()))
            if missing or unexpected:
                raise RAFTSmallRuntimeError(
                    "RAFT-small checkpoint does not exactly match torchvision RAFT-small"
                )
        except RAFTSmallRuntimeError:
            raise
        except Exception as exc:
            raise RAFTSmallRuntimeError(
                "Unable to load verified RAFT-small local checkpoint"
            ) from exc

    @property
    def device(self) -> str:
        return str(self._device)

    @staticmethod
    def _validate_pair_ids(source_frame_id: int, target_frame_id: int) -> None:
        if isinstance(source_frame_id, bool) or isinstance(target_frame_id, bool):
            raise RAFTSmallRuntimeError("RAFT-small frame IDs must be integer source-frame IDs")
        if not isinstance(source_frame_id, int) or not isinstance(target_frame_id, int):
            raise RAFTSmallRuntimeError("RAFT-small frame IDs must be integers")
        if source_frame_id == target_frame_id:
            raise RAFTSmallRuntimeError("RAFT-small requires two distinct adjacent source-frame IDs")

    def estimate_pair(
        self,
        source_rgb: np.ndarray,
        target_rgb: np.ndarray,
        *,
        source_frame_id: int,
        target_frame_id: int,
    ) -> RAFTSmallFlowResult:
        """Infer adjacent source→target flow without modifying either source.

        Frame IDs are mandatory provenance, and matching IDs are rejected: a
        caller must not manufacture a pair by feeding one real source twice.
        Adjacentness itself is established by the source-selection/ORB chain;
        this low-level runtime records the concrete pair it was given.
        """

        self._validate_pair_ids(source_frame_id, target_frame_id)
        source = _coerce_real_rgb_frame(source_rgb, label="source_rgb")
        target = _coerce_real_rgb_frame(target_rgb, label="target_rgb")
        if source.shape != target.shape:
            raise RAFTSmallRuntimeError("RAFT-small adjacent source frames must have the same RGB shape")
        height, width = source.shape[:2]
        padded_height, padded_width = _padded_extent(height), _padded_extent(width)
        source_tensor = self._frame_tensor(source, padded_height, padded_width)
        target_tensor = self._frame_tensor(target, padded_height, padded_width)
        started = time.perf_counter()
        device_elapsed: float | None = None
        try:
            inference_context = self._torch.inference_mode()
            autocast_context = (
                self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
                if self._precision == "float16"
                else nullcontext()
            )
            start_event = end_event = None
            if self._precision == "float16":
                start_event = self._torch.cuda.Event(enable_timing=True)
                end_event = self._torch.cuda.Event(enable_timing=True)
                start_event.record()
            with inference_context, autocast_context:
                output = self._model(source_tensor, target_tensor)
                prediction = output[-1] if isinstance(output, (tuple, list)) else output
            if end_event is not None and start_event is not None:
                end_event.record()
                end_event.synchronize()
                device_elapsed = float(start_event.elapsed_time(end_event)) / 1000.0
            flow = prediction.detach().float().cpu().numpy()
        except Exception as exc:
            raise RAFTSmallRuntimeError("RAFT-small inference failed for the adjacent source pair") from exc
        if flow.ndim != 4 or flow.shape[0] != 1 or flow.shape[1] != 2:
            raise RAFTSmallRuntimeError("RAFT-small returned an invalid flow tensor shape")
        if flow.shape[2:] != (padded_height, padded_width):
            raise RAFTSmallRuntimeError("RAFT-small returned flow at an unexpected resolution")
        flow_xy = np.ascontiguousarray(flow[0, :, :height, :width].transpose(1, 2, 0), dtype=np.float32)
        finite = bool(np.isfinite(flow_xy).all())
        if not finite:
            raise RAFTSmallRuntimeError("RAFT-small produced non-finite flow; candidate pair is rejected")
        audit = RAFTSmallFlowAudit(
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
            weights_path=str(self.weights_path),
            weights_sha256=self.config.weights_sha256,
            device=self.device,
            precision=self._precision,
            input_height=height,
            input_width=width,
            padded_height=padded_height,
            padded_width=padded_width,
            output_height=height,
            output_width=width,
            finite=True,
            inference_wall_seconds=time.perf_counter() - started,
            inference_device_seconds=device_elapsed,
        )
        return RAFTSmallFlowResult(flow_xy=flow_xy, audit=audit)

    def estimate_pair_tensors(
        self,
        source_rgb: Any,
        target_rgb: Any,
        *,
        source_frame_id: int,
        target_frame_id: int,
    ) -> RAFTSmallTensorFlowResult:
        """Infer flow from already-resident CHW source tensors without D2H.

        Inputs must be concrete Torch tensors on this runtime's device, with
        shape ``3xHxW`` and equal dimensions.  ``uint8`` tensors represent
        decoded sRGB in ``[0, 255]``; floating tensors represent decoded sRGB
        normalised to ``[0, 1]``.  No validation reduction is converted to a
        host scalar, and this method never invokes ``cpu()``, ``numpy()``, or
        a device transfer.  Thus it is suitable only after a caller has used
        the candidate resident-frame cache to upload real sources once.
        """

        self._validate_pair_ids(source_frame_id, target_frame_id)
        source = self._resident_tensor(source_rgb, label="source_rgb")
        target = self._resident_tensor(target_rgb, label="target_rgb")
        if tuple(source.shape) != tuple(target.shape):
            raise RAFTSmallRuntimeError("RAFT-small adjacent source tensors must have the same RGB shape")
        height, width = int(source.shape[1]), int(source.shape[2])
        padded_height, padded_width = _padded_extent(height), _padded_extent(width)
        source = self._normalise_resident_tensor(source, padded_height, padded_width)
        target = self._normalise_resident_tensor(target, padded_height, padded_width)
        try:
            inference_context = self._torch.inference_mode()
            autocast_context = (
                self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
                if self._precision == "float16"
                else nullcontext()
            )
            with inference_context, autocast_context:
                output = self._model(source, target)
                prediction = output[-1] if isinstance(output, (tuple, list)) else output
            flow = prediction.detach().float()
        except Exception as exc:
            raise RAFTSmallRuntimeError("RAFT-small tensor inference failed for the adjacent source pair") from exc
        if flow.ndim != 4 or tuple(flow.shape[:2]) != (1, 2):
            raise RAFTSmallRuntimeError("RAFT-small returned an invalid tensor flow shape")
        if tuple(flow.shape[2:]) != (padded_height, padded_width):
            raise RAFTSmallRuntimeError("RAFT-small returned tensor flow at an unexpected resolution")
        # Keep flow in the source's image coordinates and on the model device.
        # ``contiguous`` is a device-local layout operation, not a host copy.
        flow_xy = flow[0, :, :height, :width].permute(1, 2, 0).contiguous()
        audit = RAFTSmallTensorFlowAudit(
            source_frame_id=source_frame_id,
            target_frame_id=target_frame_id,
            weights_path=str(self.weights_path),
            weights_sha256=self.config.weights_sha256,
            device=self.device,
            precision=self._precision,
            input_height=height,
            input_width=width,
            padded_height=padded_height,
            padded_width=padded_width,
            output_height=height,
            output_width=width,
        )
        return RAFTSmallTensorFlowResult(flow_xy=flow_xy, audit=audit)

    def _resident_tensor(self, tensor: Any, *, label: str) -> Any:
        if not isinstance(tensor, self._torch.Tensor):
            raise RAFTSmallRuntimeError(f"{label} must be an already-resident Torch tensor")
        if tensor.device != self._device:
            raise RAFTSmallRuntimeError(
                f"{label} must already reside on {self._device}; tensor uploads are not allowed here"
            )
        if tensor.ndim != 3 or int(tensor.shape[0]) != 3:
            raise RAFTSmallRuntimeError(f"{label} must be a CHW tensor with exactly three RGB channels")
        if int(tensor.shape[1]) < 8 or int(tensor.shape[2]) < 8:
            raise RAFTSmallRuntimeError(f"{label} is too small for RAFT-small inference")
        allowed = {self._torch.uint8, self._torch.float16, self._torch.float32, self._torch.float64}
        if tensor.dtype not in allowed:
            raise RAFTSmallRuntimeError(f"{label} must use uint8, float16, float32, or float64")
        return tensor.detach().contiguous()

    def _normalise_resident_tensor(self, tensor: Any, padded_height: int, padded_width: int) -> Any:
        if tensor.dtype == self._torch.uint8:
            normalised = tensor.to(dtype=self._torch.float32).div(255.0)
        else:
            # Float input is intentionally assumed to be a caller-validated
            # [0, 1] decoded sRGB tensor.  Checking min/max on CUDA would add
            # a synchronising scalar download, forbidden at this boundary.
            normalised = tensor.to(dtype=self._torch.float32)
        normalised = normalised.mul(2.0).sub(1.0).unsqueeze(0)
        height, width = int(tensor.shape[1]), int(tensor.shape[2])
        if (height, width) != (padded_height, padded_width):
            padding = (0, padded_width - width, 0, padded_height - height)
            normalised = self._torch.nn.functional.pad(normalised, padding, mode="replicate")
        return normalised.contiguous()

    def _frame_tensor(self, rgb: np.ndarray, padded_height: int, padded_width: int) -> Any:
        tensor = self._torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(device=self._device, dtype=self._torch.float32, non_blocking=True)
        # Torchvision's RAFT preprocessing maps [0, 1] RGB to [-1, 1].
        tensor = tensor.div(255.0).mul(2.0).sub(1.0)
        height, width = rgb.shape[:2]
        if (height, width) != (padded_height, padded_width):
            pad = (0, padded_width - width, 0, padded_height - height)
            tensor = self._torch.nn.functional.pad(tensor, pad, mode="replicate")
        return tensor


__all__ = [
    "RAFTSmallFlowAudit",
    "RAFTSmallFlowResult",
    "RAFTSmallTensorFlowAudit",
    "RAFTSmallTensorFlowResult",
    "RAFTSmallRuntimeConfig",
    "RAFTSmallRuntimeError",
    "TorchvisionRAFTSmallRuntime",
    "sha256_file",
    "verify_local_raft_small_weights",
]
