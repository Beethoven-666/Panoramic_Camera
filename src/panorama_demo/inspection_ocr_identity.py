"""Read-only OCR anchors for associating text with complete FastSAM masks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class OCRTextDetection:
    polygon_xy: np.ndarray
    text: str
    confidence: float


def normalize_ocr_text(text: str) -> str:
    """Normalize OCR text without introducing semantic substitutions."""

    return re.sub(r"[^A-Z0-9]", "", text.upper())


def normalized_edit_similarity(first: str, second: str) -> float:
    """Return normalized Levenshtein similarity in ``[0, 1]``."""

    lhs = normalize_ocr_text(first)
    rhs = normalize_ocr_text(second)
    if not lhs and not rhs:
        return 1.0
    if not lhs or not rhs:
        return 0.0
    previous = list(range(len(rhs) + 1))
    for row, left_character in enumerate(lhs, start=1):
        current = [row]
        for column, right_character in enumerate(rhs, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + int(left_character != right_character),
                )
            )
        previous = current
    return float(1.0 - previous[-1] / max(len(lhs), len(rhs)))


def audit_waveshare_text(
    detection: OCRTextDetection,
    *,
    minimum_confidence: float = 0.45,
    minimum_similarity: float = 0.70,
) -> dict[str, object]:
    """Apply a fixed fuzzy identity gate to one OCR detection."""

    normalized = normalize_ocr_text(detection.text)
    similarity = normalized_edit_similarity(normalized, "WAVESHARE")
    accepted = bool(
        np.isfinite(detection.confidence)
        and detection.confidence >= minimum_confidence
        and 7 <= len(normalized) <= 14
        and (
            "WAVESHARE" in normalized
            or similarity >= minimum_similarity
        )
    )
    return {
        "raw_text": detection.text,
        "normalized_text": normalized,
        "confidence": float(detection.confidence),
        "target_similarity": similarity,
        "pass": accepted,
    }


def _polygon_mask(
    polygon_xy: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [np.rint(polygon_xy).astype(np.int32)],
        1,
    )
    return mask.astype(bool)


def audit_complete_white_mask(
    *,
    ocr_polygon_xy: np.ndarray,
    candidate_polygon_xy: np.ndarray,
    lab_image: np.ndarray,
    minimum_text_coverage: float = 0.80,
) -> dict[str, object]:
    """Audit whether a FastSAM mask encloses a complete white text box."""

    image = np.asarray(lab_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Lab image must have shape HxWx3")
    height, width = image.shape[:2]
    text_polygon = np.asarray(ocr_polygon_xy, dtype=np.float32)
    candidate_polygon = np.asarray(
        candidate_polygon_xy, dtype=np.float32
    )
    if (
        text_polygon.shape != (4, 2)
        or candidate_polygon.ndim != 2
        or candidate_polygon.shape[0] < 3
        or candidate_polygon.shape[1] != 2
    ):
        return {"pass": False, "reason": "invalid_polygon"}

    text_mask = _polygon_mask(text_polygon, (height, width))
    candidate_mask = _polygon_mask(candidate_polygon, (height, width))
    text_area = int(np.count_nonzero(text_mask))
    candidate_area = int(np.count_nonzero(candidate_mask))
    if text_area <= 0 or candidate_area <= 0:
        return {"pass": False, "reason": "empty_polygon"}
    coverage = float(
        np.count_nonzero(text_mask & candidate_mask) / text_area
    )
    text_x, text_y, text_width, text_height = cv2.boundingRect(
        np.rint(text_polygon).astype(np.int32)
    )
    box_x, box_y, box_width, box_height = cv2.boundingRect(
        np.rint(candidate_polygon).astype(np.int32)
    )
    area_ratio = float(candidate_area / text_area)
    width_ratio = float(box_width / max(1, text_width))
    height_ratio = float(box_height / max(1, text_height))
    vertex_distances = [
        float(
            cv2.pointPolygonTest(
                candidate_polygon,
                (float(point[0]), float(point[1])),
                True,
            )
        )
        for point in text_polygon
    ]
    horizontal_margin = min(
        text_x - box_x,
        box_x + box_width - (text_x + text_width),
    )
    vertical_margin = min(
        text_y - box_y,
        box_y + box_height - (text_y + text_height),
    )
    measured_lab = image[candidate_mask]
    median_lab = np.median(measured_lab, axis=0)
    chroma_distance = float(
        np.linalg.norm(median_lab[1:].astype(np.float64) - 128.0)
    )
    touches_image_border = bool(
        box_x <= 0
        or box_y <= 0
        or box_x + box_width >= width
        or box_y + box_height >= height
    )
    accepted = bool(
        coverage >= minimum_text_coverage
        and min(vertex_distances) >= -2.0
        and 4.0 <= area_ratio <= 90.0
        and 1.25 <= width_ratio <= 8.0
        and 1.50 <= height_ratio <= 12.0
        and horizontal_margin >= max(2, int(0.03 * box_width))
        and vertical_margin >= max(3, int(0.05 * box_height))
        and float(median_lab[0]) >= 145.0
        and chroma_distance <= 28.0
        and not touches_image_border
    )
    score = float(
        coverage
        + 0.20 * min(1.0, min(vertex_distances) / 12.0)
        + 0.10 * min(1.0, area_ratio / 12.0)
    )
    return {
        "text_coverage_ratio": coverage,
        "candidate_to_text_area_ratio": area_ratio,
        "candidate_to_text_width_ratio": width_ratio,
        "candidate_to_text_height_ratio": height_ratio,
        "minimum_text_vertex_margin_pixels": min(vertex_distances),
        "minimum_horizontal_bbox_margin_pixels": int(horizontal_margin),
        "minimum_vertical_bbox_margin_pixels": int(vertical_margin),
        "median_lab": [float(value) for value in median_lab],
        "median_lab_chroma_distance": chroma_distance,
        "touches_image_border": touches_image_border,
        "score": score,
        "pass": accepted,
    }


def select_unambiguous_mask_association(
    audits: Sequence[dict[str, object]],
    *,
    ambiguity_margin: float = 0.05,
) -> int | None:
    """Select one passing mask only when the best score is unambiguous."""

    passing = sorted(
        (
            (float(row["score"]), index)
            for index, row in enumerate(audits)
            if bool(row.get("pass", False))
        ),
        reverse=True,
    )
    if not passing:
        return None
    if (
        len(passing) > 1
        and passing[0][0] - passing[1][0] < ambiguity_margin
    ):
        return None
    return int(passing[0][1])


__all__ = [
    "OCRTextDetection",
    "audit_complete_white_mask",
    "audit_waveshare_text",
    "normalize_ocr_text",
    "normalized_edit_similarity",
    "select_unambiguous_mask_association",
]
