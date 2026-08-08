from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from panorama_demo.video_visual_renderer import VideoDISPairEvidence
from panorama_demo.video_v61_blocker_poc import (
    V61BlockerPocSpec,
    run_v61_blocker_poc,
    run_v61_poc_pair,
)


def _translated_pair(shift: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = 480, 200
    old = np.zeros((height, width, 3), np.uint8)
    cv2.rectangle(old, (45, 40), (120, 450), (255, 255, 255), 3)
    cv2.line(old, (20, 30), (180, 460), (180, 180, 180), 2)
    matrix = np.float32(((1, 0, shift), (0, 1, 0)))
    new = cv2.warpAffine(old, matrix, (width, height))
    return old, new


def _exact_translation_evidence(dx: float):
    def factory(old: np.ndarray, new: np.ndarray, overlap: np.ndarray) -> VideoDISPairEvidence:
        height, width = overlap.shape
        flow = np.zeros((height, width, 2), np.float32)
        flow[..., 0] = dx
        zeros = np.zeros((height, width), np.float32)
        return VideoDISPairEvidence(
            flow_forward=flow,
            flow_backward=-flow,
            fb_error=zeros,
            rgb_residual=zeros,
            gradient_residual=zeros,
            occlusion_risk_mask=np.zeros((height, width), bool),
            correspondence_confidence=np.ones((height, width), np.float32),
            reliable_mask=np.asarray(overlap, bool),
            sampled_new_bgra=np.dstack((new, np.full((height, width), 255, np.uint8))),
        )

    return factory


def test_v61_poc_uses_one_real_pair_evidence_then_graphcut_and_narrow_blend() -> None:
    old, new = _translated_pair(1)
    result = run_v61_poc_pair(
        old, new, left_frame_id=10, right_frame_id=11,
        evidence_factory=_exact_translation_evidence(1.0),
    )

    assert result.alignment_accepted
    assert result.pre_seam_pass
    assert result.graphcut_called
    assert result.graphcut_accepted
    assert result.blend_band_pixel_count >= 0
    assert result.double_edge_count is not None
    assert result.ghost_count is not None


def test_v61_poc_preseam_failure_does_not_call_graphcut() -> None:
    old, new = _translated_pair(16)
    result = run_v61_poc_pair(old, new, left_frame_id=10, right_frame_id=11)

    assert not result.pre_seam_pass
    assert not result.graphcut_called


def test_v61_coarse_placement_does_not_consume_the_four_pixel_residual_budget() -> None:
    old, new = _translated_pair(16)
    result = run_v61_poc_pair(
        old, new, left_frame_id=10, right_frame_id=11,
        evidence_factory=_exact_translation_evidence(0.0),
    )

    assert abs(result.coarse_dx_px or 0.0) > 12.0
    assert result.residual_max_displacement_px is not None
    assert result.residual_max_displacement_px <= 4.0


def test_v61_exposes_not_evaluable_metrics_instead_of_ambiguous_nulls() -> None:
    blank = np.zeros((480, 200, 3), np.uint8)
    result = run_v61_poc_pair(blank, blank, left_frame_id=10, right_frame_id=11)

    assert not result.pre_seam_pass
    assert "edge_residual_p95_px" in result.not_evaluable_metrics
    assert "double_edge_count" in result.not_evaluable_metrics


def test_v61_phase_one_baseline_lock_hashes_the_original_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "benchmarks" / "v61_blocker_poc_v1.lock.json").read_text(encoding="utf-8"))
    evidence = root / str(lock["poc_json"]["path"])

    if not evidence.is_file():
        pytest.skip("local Phase 1 evidence is not available in this checkout")
    actual = hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert actual == lock["poc_json"]["sha256"]


def test_v61_real_frozen_slow_pair_remains_a_strict_phase_one_one_regression() -> None:
    root = Path(__file__).resolve().parents[1]
    session = root / "data" / "captures" / "video" / "run_20260806_153033"
    if not session.is_dir():
        pytest.skip("frozen Phase 1.1 session is not available in this checkout")
    old = cv2.imread(str(session / "color" / "00000087.jpg"), cv2.IMREAD_COLOR)
    new = cv2.imread(str(session / "color" / "00000088.jpg"), cv2.IMREAD_COLOR)
    assert old is not None and new is not None

    result = run_v61_poc_pair(old, new, left_frame_id=87, right_frame_id=88)

    assert result.coarse_dx_px is not None
    assert result.alignment_accepted
    assert result.residual_max_displacement_px is not None
    assert result.residual_max_displacement_px <= 4.0
    assert not result.pre_seam_pass
    assert result.rejection_reason == "pre_seam_geometry_gate_failed"
    assert "double_edge_count" in result.not_evaluable_metrics


def test_v61_poc_reads_only_real_capture_rgb_frames(tmp_path) -> None:
    root = tmp_path / "run"
    (root / "color").mkdir(parents=True)
    old, middle = _translated_pair(1)
    _, right = _translated_pair(2)
    for frame_id, image in ((1, old), (2, middle), (3, right)):
        assert cv2.imwrite(str(root / "color" / f"{frame_id:08d}.jpg"), image)
    with (root / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame_id", "color_path"))
        writer.writeheader()
        for frame_id in (1, 2, 3):
            writer.writerow({"frame_id": frame_id, "color_path": f"color/{frame_id:08d}.jpg"})

    result = run_v61_blocker_poc(V61BlockerPocSpec("synthetic", root, 1, (2,), 3))

    assert result.name == "synthetic"
    assert result.baseline.left_frame_id == 1
    assert [pair.left_frame_id for pair in result.densified_pairs] == [1, 2]
    assert [pair.right_frame_id for pair in result.densified_pairs] == [2, 3]
    assert result.baseline_runtime_ms > 0.0
