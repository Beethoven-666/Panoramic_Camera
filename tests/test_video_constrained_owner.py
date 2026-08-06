from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_constrained_owner import (
    ConstrainedOwnerConfig,
    PairRisk,
    assert_c1_real_source_owners,
    assess_c1_pair_risk,
    render_c1_constrained_hard_owner_pair,
    select_c1_risk_aware_keyframes,
)
from panorama_demo.video_visual_renderer import VideoVisualSource


def _source(frame_id: int, image: np.ndarray) -> VideoVisualSource:
    return VideoVisualSource(frame_id=frame_id, bgra=image.astype(np.uint8))


def test_c1_selects_only_real_ids_at_12_and_5_pixel_risk_steps() -> None:
    ids = (10, 11, 12, 13, 14, 15, 16)
    risks = (False, PairRisk(True, 100, 40.0, 0.0, "high_luma_residual"), False, False, False, False)
    plan = select_c1_risk_aware_keyframes(ids, (3, 3, 3, 3, 3, 3), risks)
    assert plan.source_indices == (0, 2, 6)
    assert plan.source_frame_ids == (10, 12, 16)
    assert plan.risky_edge_count == 1
    assert plan.as_dict()["real_source_frames_only"] is True
    assert plan.as_dict()["interpolated_poses"] is False


def test_c1_pair_is_corridor_scoped_second_order_and_verbatim_real_source() -> None:
    height, width = 48, 160
    first = np.full((height, width, 4), (30, 40, 50, 255), dtype=np.uint8)
    second = np.full_like(first, (160, 150, 140, 255))
    # The line runs through the nominal middle.  The seam must account for it,
    # while every rendered sample remains an exact source pixel.
    for row in range(height):
        first[row, 80:84, :3] = 255
        second[row, 80:84, :3] = 255
    result = render_c1_constrained_hard_owner_pair(
        _source(4, first), _source(9, second),
        config=ConstrainedOwnerConfig(seam_corridor_width_pixels=48, maximum_row_step_pixels=3),
    )
    audit = result.audit
    assert audit.corridor_x == (56, 104)
    assert len(audit.seam_x_by_row) == height
    assert audit.line_constraint_pixel_count > 0
    assert max(abs(audit.seam_x_by_row[row] - audit.seam_x_by_row[row - 1]) for row in range(1, height)) <= 3
    assert set(np.unique(result.owner_frame_id)) == {4, 9}
    assert_c1_real_source_owners(result, (_source(4, first), _source(9, second)))


def test_c1_cleanup_removes_tiny_nonfixed_owner_island_without_inventing_source() -> None:
    height, width = 32, 96
    first = np.full((height, width, 4), (10, 20, 30, 255), dtype=np.uint8)
    second = np.full_like(first, (90, 80, 70, 255))
    # A tiny fixed first-owner island within the incoming side is protected;
    # it remains literal first-source data.  Cleanup must never rewrite fixed
    # evidence, and must still preserve strict owner/source topology.
    fixed = np.full((height, width), -1, dtype=np.int32)
    fixed[10:12, 70:72] = 1
    result = render_c1_constrained_hard_owner_pair(
        _source(1, first), _source(2, second),
        config=ConstrainedOwnerConfig(seam_corridor_width_pixels=64, owner_cleanup_minimum_pixels=8),
        fixed_owner_frame_id=fixed,
    )
    assert np.all(result.owner_frame_id[10:12, 70:72] == 1)
    assert_c1_real_source_owners(result, (_source(1, first), _source(2, second)))


def test_c1_risk_rejects_disjoint_pair_and_bad_fixed_owner() -> None:
    first = np.zeros((8, 16, 4), dtype=np.uint8)
    second = np.zeros_like(first)
    first[:, :4, 3] = 255
    second[:, 12:, 3] = 255
    left, right = _source(1, first), _source(2, second)
    assert assess_c1_pair_risk(left, right).reason == "no_common_real_source_support"
    with pytest.raises(ValueError, match="only name"):
        render_c1_constrained_hard_owner_pair(left, right, fixed_owner_frame_id=np.full((8, 16), 7))


def test_c1_rejects_nonmonotonic_ids_and_invalid_config() -> None:
    with pytest.raises(ValueError, match="chronological"):
        select_c1_risk_aware_keyframes((2, 1), (1.0,), (False,))
    with pytest.raises(ValueError, match="risk step"):
        ConstrainedOwnerConfig(risk_keyframe_step_pixels=13.0)
