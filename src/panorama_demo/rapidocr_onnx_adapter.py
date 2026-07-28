"""Optional fail-closed RapidOCR ONNX adapter.

RapidOCR is imported only when the adapter is explicitly constructed.  The
wrapper supplies preprocessing and postprocessing, while all three ONNX
sessions are replaced with explicitly constructed, profiled CUDA-first
sessions.  Shape/control nodes may use CPU, but a compute-heavy node on CPU or
an inference without observed CUDA execution is rejected before results are
returned.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np

from .cuda_backend import (
    configure_cuda_dll_search_path,
    configure_cudnn_dll_search_path,
)


class RapidOCROnnxError(RuntimeError):
    """Base error for the optional OCR adapter."""


class RapidOCRProviderError(RapidOCROnnxError):
    """Raised when actual CUDA execution cannot be proven."""


@dataclass(frozen=True)
class RapidOCRModels:
    detection: Path
    classification: Path
    recognition: Path

    def validated(self) -> "RapidOCRModels":
        values = {
            "detection": Path(self.detection).expanduser().resolve(),
            "classification": Path(self.classification).expanduser().resolve(),
            "recognition": Path(self.recognition).expanduser().resolve(),
        }
        for name, path in values.items():
            if not path.is_file() or path.suffix.lower() != ".onnx":
                raise FileNotFoundError(
                    f"RapidOCR {name} ONNX model was not found: {path}"
                )
        return RapidOCRModels(**values)


@dataclass(frozen=True)
class RapidOCRRuntime:
    provider: str = "CUDAExecutionProvider"
    device_id: int = 0
    profile_directory: Path | None = None
    allow_shape_control_cpu: bool = True

    def validate(self) -> None:
        if self.provider != "CUDAExecutionProvider":
            raise ValueError(
                "RapidOCR adapter currently requires CUDAExecutionProvider"
            )
        if int(self.device_id) < 0:
            raise ValueError("RapidOCR CUDA device_id must be non-negative")


@dataclass(frozen=True)
class RapidOCRDetection:
    polygon_xy: np.ndarray
    text: str
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "polygon_xy": self.polygon_xy.tolist(),
            "text": self.text,
            "score": float(self.score),
        }


_HEAVY_COMPUTE_OPERATORS = frozenset(
    {
        "Conv",
        "FusedConv",
        "MatMul",
        "Gemm",
        "Attention",
        "LSTM",
        "GRU",
    }
)
_SHAPE_CONTROL_CPU_OPERATORS = frozenset(
    {
        "Cast",
        "Concat",
        "ConstantOfShape",
        "Equal",
        "Expand",
        "Gather",
        "GatherElements",
        "GatherND",
        "Greater",
        "Identity",
        "Less",
        "NonZero",
        "Range",
        "Reshape",
        "Shape",
        "Size",
        "Slice",
        "Squeeze",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    return " ".join(text.split())


def _normalized_quad(value: object, width: int, height: int) -> np.ndarray:
    polygon = np.asarray(value, dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("RapidOCR polygon must be a finite 4x2 quadrilateral")
    polygon[:, 0] = np.clip(polygon[:, 0], 0.0, float(width - 1))
    polygon[:, 1] = np.clip(polygon[:, 1], 0.0, float(height - 1))
    center = np.mean(polygon, axis=0)
    angles = np.arctan2(
        polygon[:, 1] - center[1], polygon[:, 0] - center[0]
    )
    polygon = polygon[np.argsort(angles)]
    start = int(np.argmin(np.sum(polygon, axis=1)))
    polygon = np.roll(polygon, -start, axis=0)
    first = polygon[1] - polygon[0]
    second = polygon[2] - polygon[0]
    area_twice = float(
        first[0] * second[1] - first[1] * second[0]
    )
    if abs(area_twice) < 1e-3:
        raise ValueError("RapidOCR polygon is degenerate")
    return np.ascontiguousarray(polygon, dtype=np.float32)


def normalize_rapidocr_result(
    raw_result: object,
    *,
    image_width: int,
    image_height: int,
) -> tuple[RapidOCRDetection, ...]:
    """Normalize RapidOCR rows into deterministic polygon/text/score values."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("OCR image dimensions must be positive")
    detections: list[RapidOCRDetection] = []
    for row in raw_result or []:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            continue
        if len(row) != 3:
            continue
        text = _normalized_text(row[1])
        try:
            score = float(row[2])
            polygon = _normalized_quad(
                row[0], image_width, image_height
            )
        except (TypeError, ValueError):
            continue
        if not text or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            continue
        detections.append(
            RapidOCRDetection(
                polygon_xy=polygon,
                text=text,
                score=score,
            )
        )
    detections.sort(
        key=lambda item: (
            float(np.min(item.polygon_xy[:, 1])),
            float(np.min(item.polygon_xy[:, 0])),
            item.text,
            -item.score,
        )
    )
    return tuple(detections)


def summarize_onnxruntime_profile(
    profile_path: str | Path,
) -> dict[str, object]:
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    provider_events: dict[str, int] = {}
    provider_duration_us: dict[str, int] = {}
    operator_providers: dict[str, set[str]] = {}
    for event in events:
        args = event.get("args") or {}
        provider = args.get("provider")
        if not provider:
            continue
        provider = str(provider)
        provider_events[provider] = provider_events.get(provider, 0) + 1
        provider_duration_us[provider] = (
            provider_duration_us.get(provider, 0)
            + int(event.get("dur") or 0)
        )
        operator = str(args.get("op_name") or event.get("name") or "unknown")
        operator_providers.setdefault(operator, set()).add(provider)
    return {
        "provider_node_events": provider_events,
        "provider_duration_us": provider_duration_us,
        "operator_providers": {
            operator: sorted(providers)
            for operator, providers in sorted(operator_providers.items())
        },
    }


def audit_profile_cuda_execution(
    profiles: Mapping[str, Mapping[str, object]],
    *,
    allow_shape_control_cpu: bool,
) -> dict[str, object]:
    """Fail closed unless executed heavy operators are CUDA-only."""

    stages: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    executed_stage_count = 0
    cuda_stage_count = 0
    heavy_cuda_event_kinds = 0
    for stage, profile in profiles.items():
        provider_events = {
            str(key): int(value)
            for key, value in dict(
                profile.get("provider_node_events", {})
            ).items()
        }
        operator_providers = {
            str(key): [str(provider) for provider in value]
            for key, value in dict(
                profile.get("operator_providers", {})
            ).items()
        }
        executed = sum(provider_events.values()) > 0
        if executed:
            executed_stage_count += 1
        cuda_events = provider_events.get("CUDAExecutionProvider", 0)
        if cuda_events > 0:
            cuda_stage_count += 1
        heavy_cpu: list[str] = []
        heavy_cuda: list[str] = []
        unexpected_cpu: list[str] = []
        for operator, providers in operator_providers.items():
            if operator in _HEAVY_COMPUTE_OPERATORS:
                if "CUDAExecutionProvider" in providers:
                    heavy_cuda.append(operator)
                if "CPUExecutionProvider" in providers:
                    heavy_cpu.append(operator)
            if (
                "CPUExecutionProvider" in providers
                and operator not in _SHAPE_CONTROL_CPU_OPERATORS
                and operator not in _HEAVY_COMPUTE_OPERATORS
            ):
                unexpected_cpu.append(operator)
        heavy_cuda_event_kinds += len(heavy_cuda)
        stage_failures: list[str] = []
        if executed and cuda_events <= 0:
            stage_failures.append("executed_without_cuda_node")
        if heavy_cpu:
            stage_failures.append("heavy_compute_operator_on_cpu")
        if unexpected_cpu:
            stage_failures.append("non_shape_control_operator_on_cpu")
        if not allow_shape_control_cpu and any(
            "CPUExecutionProvider" in providers
            for providers in operator_providers.values()
        ):
            stage_failures.append("all_cpu_operators_disabled")
        stages[stage] = {
            "executed": executed,
            "provider_node_events": provider_events,
            "heavy_cuda_operators": sorted(heavy_cuda),
            "heavy_cpu_operators": sorted(heavy_cpu),
            "unexpected_cpu_operators": sorted(unexpected_cpu),
            "failures": stage_failures,
            "pass": not stage_failures,
        }
        failures.extend(f"{stage}:{value}" for value in stage_failures)
    if executed_stage_count == 0:
        failures.append("no_onnx_stage_executed")
    if heavy_cuda_event_kinds == 0:
        failures.append("no_heavy_compute_operator_observed_on_cuda")
    passed = not failures
    return {
        "pass": passed,
        "executed_stage_count": executed_stage_count,
        "cuda_stage_count": cuda_stage_count,
        "heavy_cuda_operator_kind_count": heavy_cuda_event_kinds,
        "allow_shape_control_cpu": bool(allow_shape_control_cpu),
        "stages": stages,
        "failures": failures,
    }


class RapidOCROnnxAdapter:
    """Explicit-model CUDA-first adapter around RapidOCR preprocessing."""

    def __init__(
        self,
        models: RapidOCRModels,
        runtime: RapidOCRRuntime,
    ) -> None:
        self.models = models.validated()
        runtime.validate()
        configure_cuda_dll_search_path()
        configure_cudnn_dll_search_path()
        self.runtime = runtime
        try:
            import onnxruntime as ort
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RapidOCROnnxError(
                "rapidocr-onnxruntime and onnxruntime-gpu are optional; "
                "install them in an isolated diagnostic environment"
            ) from exc
        available = tuple(str(value) for value in ort.get_available_providers())
        if runtime.provider not in available:
            raise RapidOCRProviderError(
                "CUDAExecutionProvider is unavailable; refusing silent "
                f"whole-image CPU OCR. Available providers: {available}"
            )
        try:
            engine = RapidOCR(
                det_model_path=str(self.models.detection),
                cls_model_path=str(self.models.classification),
                rec_model_path=str(self.models.recognition),
                det_use_cuda=True,
                cls_use_cuda=True,
                rec_use_cuda=True,
            )
        except Exception as exc:
            raise RapidOCROnnxError(
                f"RapidOCR explicit-model construction failed: {exc}"
            ) from exc
        bindings = self._runtime_bindings(engine)
        profile_directory = (
            Path(runtime.profile_directory).expanduser().resolve()
            if runtime.profile_directory is not None
            else None
        )
        if profile_directory is not None:
            profile_directory.mkdir(parents=True, exist_ok=True)
        sessions: dict[str, Any] = {}
        providers: dict[str, tuple[str, ...]] = {}
        provider_spec: list[Any] = [
            (
                runtime.provider,
                {
                    "device_id": str(runtime.device_id),
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": "1",
                },
            ),
            "CPUExecutionProvider",
        ]
        for stage, (binding, model_path) in bindings.items():
            options = ort.SessionOptions()
            options.enable_profiling = True
            if profile_directory is not None:
                options.profile_file_prefix = str(
                    profile_directory / f"rapidocr_{stage}"
                )
            try:
                session = ort.InferenceSession(
                    str(model_path),
                    sess_options=options,
                    providers=provider_spec,
                )
            except Exception as exc:
                raise RapidOCRProviderError(
                    f"RapidOCR {stage} CUDA session creation failed: {exc}"
                ) from exc
            active = tuple(str(value) for value in session.get_providers())
            if not active or active[0] != runtime.provider:
                raise RapidOCRProviderError(
                    f"RapidOCR {stage} silently changed provider order: {active}"
                )
            binding.session = session
            sessions[stage] = session
            providers[stage] = active
        self._engine = engine
        self._sessions = sessions
        self.active_providers = providers
        self.execution_audit: dict[str, object] | None = None
        self.profile_paths: dict[str, Path] = {}

    def _runtime_bindings(
        self, engine: object
    ) -> dict[str, tuple[object, Path]]:
        try:
            det = engine.text_det.infer
            cls = engine.text_cls.infer
            rec = engine.text_rec.session
            for value in (det, cls, rec):
                if not hasattr(value, "session"):
                    raise AttributeError("missing session")
        except AttributeError as exc:
            raise RapidOCRProviderError(
                "RapidOCR wrapper internals do not expose replaceable ONNX "
                "sessions; provider control cannot be guaranteed"
            ) from exc
        return {
            "detection": (det, self.models.detection),
            "classification": (cls, self.models.classification),
            "recognition": (rec, self.models.recognition),
        }

    def _verify_first_execution(self) -> None:
        profiles: dict[str, dict[str, object]] = {}
        for stage, session in self._sessions.items():
            try:
                raw_path = session.end_profiling()
            except Exception as exc:
                raise RapidOCRProviderError(
                    f"RapidOCR {stage} execution profile is unavailable: {exc}"
                ) from exc
            if not raw_path:
                raise RapidOCRProviderError(
                    f"RapidOCR {stage} did not produce an execution profile"
                )
            path = Path(raw_path).resolve()
            self.profile_paths[stage] = path
            profiles[stage] = summarize_onnxruntime_profile(path)
        audit = audit_profile_cuda_execution(
            profiles,
            allow_shape_control_cpu=self.runtime.allow_shape_control_cpu,
        )
        audit.update(
            {
                "active_providers": {
                    key: list(value)
                    for key, value in self.active_providers.items()
                },
                "profile_paths": {
                    key: str(value)
                    for key, value in self.profile_paths.items()
                },
            }
        )
        self.execution_audit = audit
        if audit["pass"] is not True:
            raise RapidOCRProviderError(
                "RapidOCR actual execution did not satisfy CUDA policy: "
                f"{audit['failures']}"
            )

    def predict(
        self, image_bgr: np.ndarray
    ) -> tuple[RapidOCRDetection, ...]:
        image = np.asarray(image_bgr)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError("RapidOCR input must be a non-empty HxWx3 uint8 BGR")
        try:
            raw_result, _ = self._engine(np.ascontiguousarray(image))
        except Exception as exc:
            detail = str(exc)
            if any(
                marker in detail
                for marker in (
                    "CUDAExecutionProvider",
                    "CudaKernel",
                    "cuDNN",
                    "cudnn",
                )
            ):
                raise RapidOCRProviderError(
                    f"RapidOCR CUDA execution failed: {detail}"
                ) from exc
            raise RapidOCROnnxError(f"RapidOCR inference failed: {exc}") from exc
        if self.execution_audit is None:
            self._verify_first_execution()
        return normalize_rapidocr_result(
            raw_result,
            image_width=image.shape[1],
            image_height=image.shape[0],
        )

    def audit(self) -> dict[str, object]:
        return {
            "schema": "rapidocr-onnx-adapter-runtime/v1",
            "models": {
                "detection": str(self.models.detection),
                "classification": str(self.models.classification),
                "recognition": str(self.models.recognition),
                "sha256": {
                    "detection": _sha256(self.models.detection),
                    "classification": _sha256(
                        self.models.classification
                    ),
                    "recognition": _sha256(self.models.recognition),
                },
            },
            "requested_provider": self.runtime.provider,
            "device_id": int(self.runtime.device_id),
            "active_providers": {
                key: list(value)
                for key, value in self.active_providers.items()
            },
            "execution_verified": self.execution_audit is not None,
            "execution": self.execution_audit,
            "silent_whole_image_cpu_fallback_allowed": False,
            "shape_control_cpu_allowed": bool(
                self.runtime.allow_shape_control_cpu
            ),
        }


__all__ = [
    "RapidOCRDetection",
    "RapidOCRModels",
    "RapidOCROnnxAdapter",
    "RapidOCROnnxError",
    "RapidOCRProviderError",
    "RapidOCRRuntime",
    "audit_profile_cuda_execution",
    "normalize_rapidocr_result",
    "summarize_onnxruntime_profile",
]
