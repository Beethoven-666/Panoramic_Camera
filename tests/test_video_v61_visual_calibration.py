from __future__ import annotations

import csv
import json

import cv2
import numpy as np

from panorama_demo.video_v61_visual_calibration import (
    Phase13Pair,
    build_phase13_review_package,
    frozen_phase13_split,
    render_phase13_pair,
)
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _write_real_rgb_pair(root, old: np.ndarray, new: np.ndarray) -> Phase13Pair:
    (root / "color").mkdir(parents=True)
    for frame_id, image in ((1, old), (2, new)):
        assert cv2.imwrite(str(root / "color" / f"{frame_id:08d}.jpg"), image)
    with (root / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame_id", "color_path"))
        writer.writeheader()
        writer.writerows(({"frame_id": 1, "color_path": "color/00000001.jpg"}, {"frame_id": 2, "color_path": "color/00000002.jpg"}))
    return Phase13Pair("synthetic-real-capture", root, 1, 2)


def _evidence(old: np.ndarray, new: np.ndarray, overlap: np.ndarray) -> VideoDISPairEvidence:
    height, width = overlap.shape
    zeros = np.zeros((height, width), np.float32)
    flow = np.zeros((height, width, 2), np.float32)
    return VideoDISPairEvidence(
        flow_forward=flow, flow_backward=-flow, fb_error=zeros, rgb_residual=zeros,
        gradient_residual=zeros, occlusion_risk_mask=np.zeros((height, width), bool),
        correspondence_confidence=np.ones((height, width), np.float32), reliable_mask=np.asarray(overlap, bool),
        sampled_new_bgra=np.dstack((new, np.full((height, width), 255, np.uint8))),
    )


def test_phase13_graphcut_is_diagnostic_and_never_enters_hard_protection(tmp_path) -> None:
    old = np.zeros((480, 200, 3), np.uint8)
    cv2.rectangle(old, (30, 20), (80, 460), (160, 160, 160), 2)
    pair = _write_real_rgb_pair(tmp_path / "capture", old, old.copy())

    result, crop, owner = render_phase13_pair(pair, evidence_factory=_evidence)

    assert result.graphcut_called
    assert result.graphcut_accepted
    assert result.generated
    assert result.graphcut_guard_intersection_pixels == 0
    assert result.blend_guard_intersection_pixels == 0
    assert crop is not None and crop.shape == (480, 160, 3)
    assert owner is not None and set(np.unique(owner)).issubset({0, 255})
    assert result.core_metrics["core_pixels_each_side"] >= 0
    assert "not_evaluable" in result.core_metrics
    assert "line_continuity_break_suspect_count" in result.context_metrics


def test_phase13_annotation_manifest_is_blinded_and_has_required_human_fields(tmp_path, monkeypatch) -> None:
    old = np.zeros((480, 200, 3), np.uint8)
    cv2.rectangle(old, (30, 20), (80, 460), (160, 160, 160), 2)
    capture_root = tmp_path / "captures"
    pair = _write_real_rgb_pair(capture_root / "run", old, old.copy())
    monkeypatch.setattr("panorama_demo.video_v61_visual_calibration.phase13_pairs", lambda _root: (pair,))

    ledger = build_phase13_review_package(capture_root, tmp_path / "review")
    manifest = json.loads((tmp_path / "review" / "annotation_manifest.json").read_text(encoding="utf-8"))

    assert ledger["graphcut_call_count"] == 1
    assert len(manifest["samples"]) == 1
    sample = manifest["samples"][0]
    assert "graphcut" not in json.dumps(sample).lower()
    assert "failure" not in json.dumps(sample).lower()
    for key in ("seam_visible", "gradient_break_visible", "double_edge_or_ghost", "line_break", "confidence"):
        assert key in sample and sample[key] is None


def test_phase13_split_is_frozen_before_rendering_and_contains_holdout() -> None:
    split = frozen_phase13_split("data/captures/video")
    assert split["schema"] == "video-v61-phase13-split-lock/v1"
    assert {entry["split"] for entry in split["pairs"]} == {"calibration", "held_out"}
