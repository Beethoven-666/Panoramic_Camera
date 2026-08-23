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
    assert len(result.component_masks) == 1
    assert np.array_equal(result.component_masks[0], result.candidate_mask)


def test_thin_non_rectangle_is_protected_but_cannot_enable_homography() -> None:
    result = build_video_object_masks(_evidence((80, 96), rectangle=False), strong_protection=np.zeros((80, 96), bool))

    assert result.candidate_mask.any()
    assert result.protected_mask.any()
    assert not result.homography_mask.any()


def test_strong_line_crossing_keeps_one_object_component_but_blocks_homography() -> None:
    protection = np.zeros((80, 96), bool)
    protection[20:45, 31] = True

    result = build_video_object_masks(_evidence((80, 96)), strong_protection=protection)

    assert result.candidate_mask[30, 30]
    assert result.candidate_mask[30, 32]
    assert len(result.components) == 1
    assert result.components[0].rectangular
    assert not result.components[0].homography_eligible
    assert not result.homography_mask.any()
