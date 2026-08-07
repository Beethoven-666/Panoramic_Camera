from __future__ import annotations

import numpy as np

from panorama_demo.video_object_mask import build_video_object_masks
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _evidence(shape: tuple[int, int], *, rectangle: bool = True) -> VideoDISPairEvidence:
    height, width = shape
    forward = np.zeros((height, width, 2), np.float32)
    backward = np.zeros_like(forward)
    if rectangle:
        forward[20:45, 18:46, 0] = 4.0
        backward[20:45, 22:50, 0] = -4.0
    else:
        forward[20:45, 18:23, 0] = 4.0
        backward[20:45, 22:27, 0] = -4.0
    zeros = np.zeros(shape, np.float32)
    return VideoDISPairEvidence(
        forward, backward, zeros, zeros, zeros, np.zeros(shape, bool), zeros,
        np.ones(shape, bool), np.zeros((*shape, 4), np.uint8),
    )


def test_motion_residual_rectangle_gets_context_collar_and_homography_eligibility() -> None:
    result = build_video_object_masks(_evidence((80, 96)), strong_protection=np.zeros((80, 96), bool))

    assert result.candidate_mask[30, 30]
    assert result.protected_mask.sum() > result.candidate_mask.sum()
    assert result.homography_mask[30, 30]
    assert result.components[0].stable_across_pair
    assert result.components[0].rectangular


def test_thin_non_rectangle_is_protected_but_cannot_enable_homography() -> None:
    result = build_video_object_masks(_evidence((80, 96), rectangle=False), strong_protection=np.zeros((80, 96), bool))

    assert result.candidate_mask.any()
    assert result.protected_mask.any()
    assert not result.homography_mask.any()
