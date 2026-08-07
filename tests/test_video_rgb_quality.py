from __future__ import annotations

import numpy as np

from panorama_demo.video_graphcut_seam import VideoGraphCutAudit
from panorama_demo.video_rgb_quality import assess_video_rgb_quality


def _audit(seam: tuple[int, ...]) -> VideoGraphCutAudit:
    return VideoGraphCutAudit(True, False, seam, max((abs(b - a) for a, b in zip(seam, seam[1:])), default=0), 0, 0, True, True, None)


def test_rgb_quality_accepts_strict_owner_and_zero_step_seam() -> None:
    bgr = np.zeros((12, 20, 3), np.uint8)
    owner = np.zeros((12, 20), np.int32)
    audit = assess_video_rgb_quality(bgr, owner, np.ones(owner.shape, bool), (_audit((10,) * 12),))

    assert audit.strict_quality_pass
    assert audit.failure_reasons == ()


def test_rgb_quality_rejects_staircase_and_invalid_owner_topology() -> None:
    bgr = np.zeros((12, 20, 3), np.uint8)
    owner = np.zeros((12, 20), np.int32)
    owner[0, 0] = -1
    audit = assess_video_rgb_quality(bgr, owner, np.ones(owner.shape, bool), (_audit(tuple(range(12))),))

    assert not audit.strict_quality_pass
    assert "owner_topology" in audit.failure_reasons
    assert "seam_step_p95" in audit.failure_reasons
