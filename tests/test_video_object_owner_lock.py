from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_object_owner_lock import (
    ObjectOwnerLockConfig,
    assert_object_owner_lock,
    depth_connected_object_candidates,
    enforce_object_owner_lock,
    plan_object_owner_lock,
)
from panorama_demo.video_visual_renderer import VideoVisualSource


def _source(frame_id: int, *, alpha: np.ndarray | None = None, depth: np.ndarray | None = None) -> VideoVisualSource:
    shape = (20, 30)
    image = np.zeros((*shape, 4), dtype=np.uint8)
    image[..., :3] = (11 + frame_id, 33, 55)
    image[..., 3] = 255 if alpha is None else alpha
    return VideoVisualSource(frame_id=frame_id, bgra=image, depth_mm=depth)


def test_object_owner_lock_expands_object_and_pins_every_protected_pixel_to_one_real_source() -> None:
    first, second = _source(101), _source(102)
    object_mask = np.zeros((20, 30), dtype=bool)
    object_mask[8:11, 12:15] = True
    plan = plan_object_owner_lock(
        first, second, object_mask=object_mask,
        config=ObjectOwnerLockConfig(object_guard_pixels=2, depth_edge_guard_pixels=0),
    )
    assert plan.accepted
    assert plan.audit.locked_owner_frame_id == 101
    assert np.count_nonzero(plan.protected_object_mask) > np.count_nonzero(object_mask)
    proposed = np.full((20, 30), 102, dtype=np.int32)
    actual = enforce_object_owner_lock(proposed, plan)
    assert np.all(actual[plan.protected_mask] == 101)
    assert np.all(actual[~plan.protected_mask] == 102)
    assert_object_owner_lock(actual, plan)
    audit = plan.audit.as_dict()
    assert audit["creates_colour"] is False
    assert audit["creates_pose"] is False
    assert audit["interpolates_source_frames"] is False


def test_object_owner_lock_derives_conservative_depth_edge_protection_without_object_mask() -> None:
    depth_a = np.full((20, 30), 1000, dtype=np.uint16)
    depth_b = depth_a.copy()
    depth_a[:, 15:] = 1600
    first, second = _source(4, depth=depth_a), _source(5, depth=depth_b)
    plan = plan_object_owner_lock(
        first, second,
        config=ObjectOwnerLockConfig(object_guard_pixels=0, depth_edge_guard_pixels=2),
    )
    assert plan.accepted
    assert plan.audit.object_input_pixel_count == 0
    assert plan.audit.protected_depth_edge_pixel_count > 0
    assert plan.protected_depth_edge_mask[:, 15].any()
    assert np.all(plan.owner_frame_id[plan.protected_mask] == 4)


def test_o1_depth_connected_components_only_keep_common_near_foreground() -> None:
    first_depth = np.full((20, 30), 1600, dtype=np.uint16)
    second_depth = first_depth.copy()
    first_depth[5:14, 8:18] = 700
    second_depth[5:14, 8:18] = 720
    first, second = _source(4, depth=first_depth), _source(5, depth=second_depth)
    result = depth_connected_object_candidates(
        first, second, minimum_component_pixels=32, near_depth_quantile=0.30
    )
    assert result.component_count == 1
    assert result.candidate_pixel_count == 90
    assert np.all(result.mask[5:14, 8:18])
    assert not result.mask[0, 0]
    audit = result.as_dict()
    assert audit["uses_rgb"] is False
    assert audit["uses_only_aligned_depth"] is True


def test_object_owner_lock_rejects_partial_source_coverage_instead_of_splitting_an_object() -> None:
    alpha = np.full((20, 30), 255, dtype=np.uint8)
    alpha[:, 14:] = 0
    first, second = _source(7, alpha=alpha), _source(8)
    object_mask = np.zeros((20, 30), dtype=bool)
    object_mask[4:16, 10:20] = True
    plan = plan_object_owner_lock(
        first, second, object_mask=object_mask,
        config=ObjectOwnerLockConfig(object_guard_pixels=0),
    )
    assert not plan.accepted
    assert plan.audit.rejection_reason == "selected_real_owner_does_not_cover_protected_domain"
    with pytest.raises(ValueError, match="rejected"):
        enforce_object_owner_lock(np.full((20, 30), 8, dtype=np.int32), plan)


def test_object_owner_lock_counts_only_one_explicit_adjacent_handoff_and_fails_closed_after_limit() -> None:
    first, second = _source(20), _source(21)
    object_mask = np.zeros((20, 30), dtype=bool)
    object_mask[5:10, 5:10] = True
    transferred = plan_object_owner_lock(
        first, second, object_mask=object_mask, previous_owner_frame_id=20,
        previous_handoff_count=0, preferred_owner_frame_id=21,
        config=ObjectOwnerLockConfig(maximum_handoffs=1),
    )
    assert transferred.audit.locked_owner_frame_id == 21
    assert transferred.audit.resulting_handoff_count == 1
    # The explicit handoff limit remains fail-closed even if an invalid history
    # is supplied through a future pair planner.
    rejected = plan_object_owner_lock(
        first, second, object_mask=object_mask, previous_owner_frame_id=20,
        previous_handoff_count=1, preferred_owner_frame_id=21,
        config=ObjectOwnerLockConfig(maximum_handoffs=1),
    )
    assert not rejected.accepted
    assert rejected.audit.rejection_reason == "maximum_object_handoffs_exceeded"


def test_object_owner_lock_rejects_nonreal_history_or_mask_shape() -> None:
    first, second = _source(1), _source(2)
    with pytest.raises(ValueError, match="adjacent real source"):
        plan_object_owner_lock(first, second, previous_owner_frame_id=99)
    with pytest.raises(ValueError, match="match"):
        plan_object_owner_lock(first, second, object_mask=np.zeros((2, 3), dtype=bool))


def test_object_owner_lock_limits_depth_protection_to_a_real_common_corridor() -> None:
    depth = np.full((20, 30), 1000, dtype=np.uint16)
    depth[:, 15:] = 1600
    first, second = _source(11, depth=depth), _source(12, depth=depth)
    corridor = np.zeros((20, 30), dtype=bool)
    corridor[:, 10:20] = True
    plan = plan_object_owner_lock(
        first,
        second,
        constraint_mask=corridor,
        config=ObjectOwnerLockConfig(object_guard_pixels=0, depth_edge_guard_pixels=0),
    )
    assert plan.accepted
    assert not np.any(plan.protected_mask[:, :10])
    assert not np.any(plan.protected_mask[:, 20:])
