from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_blend import S13BlendPlan, S13BlendTransaction
from panorama_demo.video_s13_photometric import (
    PHOTOMETRIC_BOUNDS_SCHEMA,
    PHOTOMETRIC_SCHEMA,
    S13PhotometricSolution,
)
from panorama_demo.video_s13_replay import S13P2ReplayPair
from panorama_demo.video_s13_visual_quality import build_s13_p3_diagnostic_quality


def test_diagnostic_rejects_new_worst_seam_jump_without_runtime_authority() -> None:
    height, width = 8, 12
    pair = S13P2ReplayPair(
        0, 0, 1, 10, 11, 2, 10, np.full(height, 6, np.int32),
        np.ones((height, 8), np.float32), np.ones((height, 8), np.float32),
        np.ones((height, 8), bool), np.ones((height, 8), np.float32),
        np.ones((height, 8), np.float32), np.ones((height, 8), bool),
        np.broadcast_to(np.arange(2, 10)[None, :] >= 6, (height, 8)),
        0, 0, "a" * 64,
    )
    plan = S13BlendPlan(
        S13BlendTransaction(0, 0, "a" * 64, "B0_owner_only", 0, 0, 0, 0, True, None, ()),
        np.zeros((height, 8), np.float32), np.ones((height, 8), bool),
        np.zeros((height, 8), bool),
    )
    p2 = np.full((height, width, 3), 80, np.uint8)
    after = p2.copy()
    after[:, 6:] = 120
    solution = S13PhotometricSolution(
        PHOTOMETRIC_SCHEMA, PHOTOMETRIC_BOUNDS_SCHEMA, "Q0_identity", (), (), {}, {}, 1, 0.75, False
    )
    report = build_s13_p3_diagnostic_quality(p2, after, after, [pair], [plan], solution)
    assert report["quality_pass"] is False
    assert report["runtime_authority"] is False
