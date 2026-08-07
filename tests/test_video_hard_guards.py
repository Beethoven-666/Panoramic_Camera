from __future__ import annotations

import numpy as np

from panorama_demo.video_hard_guards import audit_guard_owner_intersection, build_video_hard_guards
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _evidence(shape: tuple[int, int]) -> VideoDISPairEvidence:
    height, width = shape
    zeros = np.zeros(shape, np.float32)
    occlusion = np.zeros(shape, bool)
    occlusion[10:15, 20:25] = True
    return VideoDISPairEvidence(np.zeros((height, width, 2), np.float32), np.zeros((height, width, 2), np.float32), zeros, zeros, zeros, occlusion, zeros, ~occlusion, np.zeros((height, width, 4), np.uint8))


def test_hard_guards_protect_line_object_boundary_thin_and_occlusion_without_depth() -> None:
    old = np.zeros((48, 64, 3), np.uint8)
    new = old.copy()
    old[:, 30:32] = 255
    object_mask = np.zeros(old.shape[:2], bool)
    object_mask[20:35, 40:55] = True
    guards = build_video_hard_guards(old, new, _evidence(old.shape[:2]), object_mask=object_mask)

    assert guards.audit.line_guard_pixels > 0
    assert guards.audit.thin_structure_pixels > 0
    assert guards.audit.object_outer_boundary_pixels > 0
    assert guards.audit.occlusion_risk_pixels == 25
    assert audit_guard_owner_intersection(guards.hard_owner_new.copy(), guards) == 0
    labels = guards.hard_owner_new.copy()
    labels[guards.hard_owner_old] = True
    assert audit_guard_owner_intersection(labels, guards) == guards.audit.hard_owner_old_pixels
