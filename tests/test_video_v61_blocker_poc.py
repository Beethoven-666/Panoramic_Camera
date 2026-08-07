from __future__ import annotations

import csv

import cv2
import numpy as np

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
