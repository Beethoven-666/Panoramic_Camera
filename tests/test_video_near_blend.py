from __future__ import annotations

import numpy as np

from panorama_demo.video_hard_guards import build_video_hard_guards
from panorama_demo.video_near_blend import apply_near_multiband, build_near_blend_eligible_mask
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _evidence(shape: tuple[int, int]) -> VideoDISPairEvidence:
    zeros = np.zeros(shape, np.float32)
    return VideoDISPairEvidence(np.zeros((*shape, 2), np.float32), np.zeros((*shape, 2), np.float32), zeros, zeros, zeros, np.zeros(shape, bool), np.ones(shape, np.float32), np.ones(shape, bool), np.zeros((*shape, 4), np.uint8))


def test_near_blend_is_limited_to_safe_interior_and_never_enters_guard() -> None:
    old = np.full((32, 48, 3), 30, np.uint8)
    new = np.full_like(old, 180)
    evidence = _evidence(old.shape[:2])
    guards = build_video_hard_guards(old, new, evidence, edge_guard_radius_px=0)
    eligible = build_near_blend_eligible_mask(np.ones(old.shape[:2], bool), np.ones(old.shape[:2], bool), evidence, guards)
    owner_new = np.zeros(old.shape[:2], bool)
    owner_new[:, 24:] = True
    owner = old.copy()
    owner[owner_new] = new[owner_new]
    output, band, audit = apply_near_multiband(old, new, owner, owner_new, eligible, guards)

    assert audit.applied
    assert audit.multiband_levels <= 3
    assert not np.any(band & guards.protected)
    assert np.array_equal(output[~band], owner[~band])


def test_occlusion_or_rgb_failure_cannot_become_near_blend_eligible() -> None:
    image = np.zeros((16, 24, 3), np.uint8)
    evidence = _evidence(image.shape[:2])
    evidence.occlusion_risk_mask[5:8, 6:9] = True
    evidence.rgb_residual[2:4, 2:4] = 99.0
    guards = build_video_hard_guards(image, image, evidence)
    eligible = build_near_blend_eligible_mask(np.ones(image.shape[:2], bool), np.ones(image.shape[:2], bool), evidence, guards)

    assert not eligible[5:8, 6:9].any()
    assert not eligible[2:4, 2:4].any()
