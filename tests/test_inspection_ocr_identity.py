from __future__ import annotations

import cv2
import numpy as np

from panorama_demo.inspection_ocr_identity import (
    OCRTextDetection,
    audit_complete_white_mask,
    audit_waveshare_text,
    normalize_ocr_text,
    normalized_edit_similarity,
    select_unambiguous_mask_association,
)


def test_waveshare_text_gate_accepts_fixed_fuzzy_identity() -> None:
    detection = OCRTextDetection(
        polygon_xy=np.asarray(
            [[20, 30], [100, 30], [100, 45], [20, 45]],
            dtype=np.float32,
        ),
        text="WAVESHARF",
        confidence=0.91,
    )
    audit = audit_waveshare_text(detection)
    assert normalize_ocr_text("Wave-share!") == "WAVESHARE"
    assert normalized_edit_similarity("WAVESHARF", "WAVESHARE") > 0.8
    assert audit["pass"] is True
    assert audit["normalized_text"] == "WAVESHARF"


def test_complete_white_mask_requires_enclosed_text_and_white_surface() -> None:
    lab = np.full((120, 180, 3), (210, 128, 128), dtype=np.uint8)
    text = np.asarray(
        [[60, 50], [120, 50], [120, 65], [60, 65]],
        dtype=np.float32,
    )
    complete = np.asarray(
        [[35, 25], [145, 25], [145, 95], [35, 95]],
        dtype=np.int32,
    )
    audit = audit_complete_white_mask(
        ocr_polygon_xy=text,
        candidate_polygon_xy=complete,
        lab_image=lab,
    )
    assert audit["pass"] is True

    partial = np.asarray(
        [[70, 45], [125, 45], [125, 70], [70, 70]],
        dtype=np.int32,
    )
    partial_audit = audit_complete_white_mask(
        ocr_polygon_xy=text,
        candidate_polygon_xy=partial,
        lab_image=lab,
    )
    assert partial_audit["pass"] is False


def test_mask_selection_rejects_near_tie() -> None:
    assert (
        select_unambiguous_mask_association(
            (
                {"pass": True, "score": 1.20},
                {"pass": True, "score": 1.17},
            )
        )
        is None
    )
    assert (
        select_unambiguous_mask_association(
            (
                {"pass": True, "score": 1.20},
                {"pass": True, "score": 1.10},
            )
        )
        == 0
    )


def test_dark_candidate_fails_white_surface_gate() -> None:
    bgr = np.full((100, 160, 3), 20, dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    audit = audit_complete_white_mask(
        ocr_polygon_xy=np.asarray(
            [[55, 42], [105, 42], [105, 55], [55, 55]],
            dtype=np.float32,
        ),
        candidate_polygon_xy=np.asarray(
            [[30, 20], [130, 20], [130, 80], [30, 80]],
            dtype=np.int32,
        ),
        lab_image=lab,
    )
    assert audit["pass"] is False
