"""Project-contained FastSAM-s ONNX polygon proposal inference.

This module intentionally has no dependency on Ultralytics, Torch, or
Torchvision.  It implements the small amount of YOLOv8 segmentation
post-processing needed by the exported FastSAM-s model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .cuda_backend import (
    configure_cuda_dll_search_path,
    configure_cudnn_dll_search_path,
)


@dataclass(frozen=True)
class FastSAMOnnxConfig:
    image_size: int = 1024
    stride: int = 32
    confidence_threshold: float = 0.25
    nms_iou_threshold: float = 0.90
    max_detections: int = 300


@dataclass(frozen=True)
class LetterboxTransform:
    original_height: int
    original_width: int
    input_height: int
    input_width: int
    gain: float
    left: int
    top: int


@dataclass(frozen=True)
class FastSAMPolygonProposal:
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    polygon_xy: np.ndarray
    mask: np.ndarray


class FastSAMOnnxError(RuntimeError):
    """Raised when the ONNX contract or requested execution mode is invalid."""


def _letterbox(
    image_bgr: np.ndarray,
    *,
    size: int,
    stride: int,
    dynamic_minimum_rectangle: bool,
) -> tuple[np.ndarray, LetterboxTransform]:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("FastSAM input must be a BGR HxWx3 image")
    h, w = image_bgr.shape[:2]
    gain = min(size / h, size / w)
    resized_w, resized_h = round(w * gain), round(h * gain)
    pad_w, pad_h = size - resized_w, size - resized_h
    if dynamic_minimum_rectangle:
        pad_w %= stride
        pad_h %= stride
    pad_w /= 2
    pad_h /= 2
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    if (resized_w, resized_h) != (w, h):
        resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image_bgr
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    transform = LetterboxTransform(
        original_height=h,
        original_width=w,
        input_height=int(padded.shape[0]),
        input_width=int(padded.shape[1]),
        gain=float(gain),
        left=int(left),
        top=int(top),
    )
    rgb_chw = padded[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb_chw, dtype=np.float32)[None] / np.float32(255.0)
    return tensor, transform


def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    boxes = np.empty_like(boxes_xywh, dtype=np.float32)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    return boxes


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, max_detections: int) -> np.ndarray:
    """Greedy, class-agnostic NMS with torchvision-compatible ordering."""
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size and len(kept) < max_detections:
        index = int(order[0])
        kept.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[index, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[index, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[index, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[index, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = np.maximum(0.0, boxes[index, 2] - boxes[index, 0]) * np.maximum(
            0.0, boxes[index, 3] - boxes[index, 1]
        )
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        union = area_i + area_rest - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[iou <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def _scale_boxes_to_original(boxes: np.ndarray, transform: LetterboxTransform) -> np.ndarray:
    scaled = boxes.astype(np.float32, copy=True)
    scaled[:, (0, 2)] -= transform.left
    scaled[:, (1, 3)] -= transform.top
    scaled /= np.float32(transform.gain)
    scaled[:, (0, 2)] = np.clip(scaled[:, (0, 2)], 0, transform.original_width)
    scaled[:, (1, 3)] = np.clip(scaled[:, (1, 3)], 0, transform.original_height)
    return scaled


def _merge_multi_segment(segments: Sequence[np.ndarray]) -> np.ndarray:
    """Match the contour joining used by YOLO segmentation text export."""
    work = [np.asarray(segment, dtype=np.float32).reshape(-1, 2) for segment in segments]
    if len(work) == 1:
        return work[0]
    index_list: list[list[int]] = [[] for _ in work]
    for i in range(1, len(work)):
        distances = ((work[i - 1][:, None] - work[i][None]) ** 2).sum(axis=2)
        idx1, idx2 = np.unravel_index(int(np.argmin(distances)), distances.shape)
        index_list[i - 1].append(int(idx1))
        index_list[i].append(int(idx2))
    joined: list[np.ndarray] = []
    for round_index in range(2):
        if round_index == 0:
            for i, indices in enumerate(index_list):
                if len(indices) == 2 and indices[0] > indices[1]:
                    indices = indices[::-1]
                    work[i] = work[i][::-1]
                work[i] = np.roll(work[i], -indices[0], axis=0)
                work[i] = np.concatenate((work[i], work[i][:1]), axis=0)
                if i in {0, len(index_list) - 1}:
                    joined.append(work[i])
                else:
                    relative = indices[1] - indices[0]
                    joined.append(work[i][: relative + 1])
        else:
            for i in range(len(index_list) - 1, -1, -1):
                if i not in {0, len(index_list) - 1}:
                    distance = abs(index_list[i][1] - index_list[i][0])
                    joined.append(work[i][distance:])
    return np.concatenate(joined, axis=0)


def _mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    contours = cv2.findContours(
        np.ascontiguousarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )[0]
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    segments = [contour.reshape(-1, 2) for contour in contours]
    return _merge_multi_segment(segments).astype(np.float32, copy=False)


def _resize_mask_logits(logits_hwc: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize arbitrary mask-channel counts without exceeding OpenCV's channel limit."""
    if logits_hwc.ndim != 3:
        raise ValueError("mask logits must have HWC shape")
    chunks: list[np.ndarray] = []
    for start in range(0, logits_hwc.shape[2], 64):
        resized = cv2.resize(
            np.ascontiguousarray(logits_hwc[:, :, start : start + 64]),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        if resized.ndim == 2:
            resized = resized[:, :, None]
        chunks.append(resized)
    if not chunks:
        return np.zeros((height, width, 0), dtype=logits_hwc.dtype)
    return np.concatenate(chunks, axis=2)


class FastSAMOnnxRunner:
    """Run a dynamic FastSAM-s ONNX export with CUDA-first execution."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device_id: int = 0,
        allow_cpu_diagnostic_fallback: bool = False,
        config: FastSAMOnnxConfig | None = None,
        enable_profiling: bool = False,
        profile_directory: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"FastSAM ONNX model not found: {self.model_path}")
        self.config = config or FastSAMOnnxConfig()
        configure_cuda_dll_search_path()
        configure_cudnn_dll_search_path()
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise FastSAMOnnxError("onnxruntime-gpu is required for FastSAM ONNX inference") from exc

        available = list(ort.get_available_providers())
        options = ort.SessionOptions()
        options.enable_profiling = bool(enable_profiling)
        if profile_directory is not None:
            directory = Path(profile_directory).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            options.profile_file_prefix = str(directory / "fastsam")
        self.diagnostic_cpu_fallback_used = False
        self.execution_mode = "cuda_priority"
        if "CUDAExecutionProvider" not in available:
            if not allow_cpu_diagnostic_fallback:
                raise FastSAMOnnxError(
                    "CUDAExecutionProvider is unavailable; refusing silent whole-graph CPU fallback"
                )
            providers: list[Any] = ["CPUExecutionProvider"]
            self.diagnostic_cpu_fallback_used = True
            self.execution_mode = "explicit_cpu_diagnostic_fallback"
        else:
            providers = [("CUDAExecutionProvider", {"device_id": str(device_id)})]
        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=providers,
            )
        except Exception as exc:
            if not allow_cpu_diagnostic_fallback:
                raise FastSAMOnnxError(f"CUDA FastSAM session creation failed: {exc}") from exc
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self.diagnostic_cpu_fallback_used = True
            self.execution_mode = "explicit_cpu_diagnostic_fallback"

        self.active_providers = tuple(self._session.get_providers())
        if self.execution_mode == "cuda_priority" and (
            not self.active_providers or self.active_providers[0] != "CUDAExecutionProvider"
        ):
            raise FastSAMOnnxError(
                "CUDA provider registration silently fell back; refusing whole-graph CPU inference"
            )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 2:
            raise FastSAMOnnxError("expected one image input and two FastSAM segmentation outputs")
        self._input_name = inputs[0].name
        self._output_names = [output.name for output in outputs]
        shape = inputs[0].shape
        self._dynamic_spatial = not (
            len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int)
        )

    def predict(self, image_bgr: np.ndarray) -> list[FastSAMPolygonProposal]:
        tensor, transform = _letterbox(
            image_bgr,
            size=self.config.image_size,
            stride=self.config.stride,
            dynamic_minimum_rectangle=self._dynamic_spatial,
        )
        prediction, prototypes = self._session.run(
            self._output_names,
            {self._input_name: tensor},
        )
        return self.decode_outputs(prediction, prototypes, transform)

    def decode_outputs(
        self,
        prediction: np.ndarray,
        prototypes: np.ndarray,
        transform: LetterboxTransform,
    ) -> list[FastSAMPolygonProposal]:
        pred = np.asarray(prediction)
        proto = np.asarray(prototypes)
        if pred.ndim != 3 or pred.shape[0] != 1:
            raise FastSAMOnnxError(f"unexpected prediction shape: {pred.shape}")
        if proto.ndim != 4 or proto.shape[0] != 1:
            raise FastSAMOnnxError(f"unexpected prototype shape: {proto.shape}")
        rows = pred[0].T
        mask_channels = int(proto.shape[1])
        class_channels = int(rows.shape[1] - 4 - mask_channels)
        if class_channels != 1:
            raise FastSAMOnnxError(
                f"FastSAM-s export must have one class; observed {class_channels}"
            )
        scores = rows[:, 4]
        candidates = scores > self.config.confidence_threshold
        rows = rows[candidates]
        scores = scores[candidates]
        if not rows.size:
            return []
        boxes_input = _xywh_to_xyxy(rows[:, :4])
        keep = _nms(
            boxes_input,
            scores,
            self.config.nms_iou_threshold,
            self.config.max_detections,
        )
        boxes_input = boxes_input[keep]
        scores = scores[keep]
        coefficients = rows[keep, 5:]
        boxes_original = _scale_boxes_to_original(boxes_input, transform)

        c, proto_h, proto_w = proto[0].shape
        logits = coefficients.astype(np.float32) @ proto[0].reshape(c, -1).astype(np.float32)
        logits_hwc = logits.reshape(-1, proto_h, proto_w).transpose(1, 2, 0)
        gain = min(proto_h / transform.original_height, proto_w / transform.original_width)
        pad_w = (proto_w - round(transform.original_width * gain)) / 2
        pad_h = (proto_h - round(transform.original_height * gain)) / 2
        top, left = round(pad_h - 0.1), round(pad_w - 0.1)
        bottom = proto_h - round(pad_h + 0.1)
        right = proto_w - round(pad_w + 0.1)
        cropped_logits = logits_hwc[top:bottom, left:right]
        native_logits = _resize_mask_logits(
            cropped_logits,
            transform.original_width,
            transform.original_height,
        )

        proposals: list[FastSAMPolygonProposal] = []
        for index, (score, box) in enumerate(zip(scores, boxes_original)):
            mask = np.asarray(native_logits[:, :, index] > 0.0, dtype=np.uint8)
            x1 = int(np.ceil(box[0]))
            y1 = int(np.ceil(box[1]))
            x2 = int(np.ceil(box[2]))
            y2 = int(np.ceil(box[3]))
            x1, x2 = max(0, x1), min(transform.original_width, x2)
            y1, y2 = max(0, y1), min(transform.original_height, y2)
            if x1 > 0:
                mask[:, :x1] = 0
            if x2 < transform.original_width:
                mask[:, x2:] = 0
            if y1 > 0:
                mask[:y1] = 0
            if y2 < transform.original_height:
                mask[y2:] = 0
            polygon = _mask_to_polygon(mask)
            if polygon.size == 0:
                continue
            proposals.append(
                FastSAMPolygonProposal(
                    score=float(score),
                    bbox_xyxy=tuple(float(value) for value in box),
                    polygon_xy=polygon,
                    mask=mask.astype(bool, copy=False),
                )
            )
        return proposals

    def end_profiling(self) -> Path | None:
        if not hasattr(self._session, "end_profiling"):
            return None
        path = self._session.end_profiling()
        return Path(path) if path else None


def summarize_onnxruntime_profile(profile_path: str | Path) -> dict[str, Any]:
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    by_provider: dict[str, dict[str, Any]] = {}
    operator_providers: dict[str, set[str]] = {}
    for event in events:
        args = event.get("args") or {}
        provider = args.get("provider")
        if not provider:
            continue
        bucket = by_provider.setdefault(provider, {"node_events": 0, "duration_us": 0})
        bucket["node_events"] += 1
        bucket["duration_us"] += int(event.get("dur") or 0)
        operator = str(args.get("op_name") or event.get("name") or "unknown")
        operator_providers.setdefault(operator, set()).add(provider)
    return {
        "providers": by_provider,
        "operator_providers": {
            operator: sorted(providers) for operator, providers in sorted(operator_providers.items())
        },
    }
