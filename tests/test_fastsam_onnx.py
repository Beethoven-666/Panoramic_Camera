from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from panorama_demo.fastsam_onnx import (
    LetterboxTransform,
    _letterbox,
    _mask_to_polygon,
    _nms,
    _resize_mask_logits,
)


def test_module_does_not_import_forbidden_frameworks() -> None:
    source = (Path(__file__).parents[1] / "src" / "panorama_demo" / "fastsam_onnx.py").read_text(
        encoding="utf-8"
    )
    compact = source.replace(" ", "")
    assert "importultralytics" not in compact
    assert "importtorch" not in compact
    assert "importtorchvision" not in compact


def test_dynamic_letterbox_uses_minimum_stride_rectangle() -> None:
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    tensor, transform = _letterbox(
        image,
        size=1024,
        stride=32,
        dynamic_minimum_rectangle=True,
    )
    assert tensor.shape == (1, 3, 640, 1024)
    assert transform == LetterboxTransform(800, 1280, 640, 1024, 0.8, 0, 0)
    assert tensor.dtype == np.float32


def test_nms_retains_overlapping_boxes_at_high_historical_threshold() -> None:
    boxes = np.asarray([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], np.float32)
    assert _nms(boxes, scores, 0.9, 300).tolist() == [0, 1, 2]
    assert _nms(boxes, scores, 0.5, 300).tolist() == [0, 2]


def test_mask_polygon_is_external_simple_contour() -> None:
    mask = np.zeros((20, 30), np.uint8)
    cv2.rectangle(mask, (3, 4), (15, 12), 1, thickness=-1)
    polygon = _mask_to_polygon(mask)
    assert polygon.shape == (4, 2)
    assert set(map(tuple, polygon.astype(int))) == {(3, 4), (3, 12), (15, 12), (15, 4)}


def test_mask_resize_supports_more_than_128_proposals() -> None:
    logits = np.zeros((8, 12, 137), dtype=np.float32)
    logits[2:6, 3:9] = 1.0
    resized = _resize_mask_logits(logits, 24, 16)
    assert resized.shape == (16, 24, 137)
    assert np.allclose(resized[:, :, 0], resized[:, :, 136])
