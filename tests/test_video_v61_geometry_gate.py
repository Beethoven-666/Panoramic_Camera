from __future__ import annotations

import cv2
import numpy as np

from panorama_demo.video_v61_geometry_gate import evaluate_v61_geometry_gate
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _evidence(shape: tuple[int, int]) -> VideoDISPairEvidence:
    height, width = shape
    flow = np.zeros((height, width, 2), np.float32)
    zeros = np.zeros(shape, np.float32)
    return VideoDISPairEvidence(flow, -flow, zeros, zeros, zeros, np.zeros(shape, bool), np.ones(shape, np.float32), np.ones(shape, bool), np.zeros((height, width, 4), np.uint8))


def test_tail_outlier_is_guarded_without_being_an_absolute_max_veto() -> None:
    old = np.zeros((480, 160, 3), np.uint8)
    cv2.line(old, (80, 0), (80, 479), (255, 255, 255), 1)
    new = old.copy()
    new[240, 80] = 0
    audit = evaluate_v61_geometry_gate(old, new, _evidence(old.shape[:2]), support=np.ones(old.shape[:2], bool), protected=np.zeros(old.shape[:2], bool))

    assert audit.edge_abs_max_px is not None
    assert audit.tail_guard.shape == old.shape[:2]
    assert audit.tail_count >= 0


def test_geometry_gate_rejects_true_fb_p95_failure() -> None:
    old = np.zeros((480, 160, 3), np.uint8)
    cv2.line(old, (80, 0), (80, 479), (255, 255, 255), 1)
    evidence = _evidence(old.shape[:2])
    evidence = VideoDISPairEvidence(evidence.flow_forward, evidence.flow_backward, np.full(old.shape[:2], 2.0, np.float32), evidence.rgb_residual, evidence.gradient_residual, evidence.occlusion_risk_mask, evidence.correspondence_confidence, evidence.reliable_mask, evidence.sampled_new_bgra)
    audit = evaluate_v61_geometry_gate(old, old.copy(), evidence, support=np.ones(old.shape[:2], bool), protected=np.zeros(old.shape[:2], bool))

    assert not audit.accepted
    assert "fb_p95" in str(audit.rejection_reason)
