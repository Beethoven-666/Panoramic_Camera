from __future__ import annotations

import pytest

from panorama_demo.video_object_patch_planning import (
    VideoDirectSourceSupport,
    VideoObjectRegion,
    VideoTrackingSourcePlan,
    plan_object_patches,
    replan_wide_object_patches,
)


def _sources(*supports: tuple[float, float]) -> tuple[VideoDirectSourceSupport, ...]:
    return tuple(
        VideoDirectSourceSupport(frame_id=index, support_x=support)
        for index, support in enumerate(supports, start=10)
    )


def test_n_req_uses_complete_object_and_context_not_fixed_strip_width() -> None:
    plan = plan_object_patches(
        VideoObjectRegion("box", (90.0, 250.0), collar_px=10),
        _sources((0.0, 120.0), (100.0, 210.0), (190.0, 300.0)),
    )

    assert plan.initial_n_req == 3
    assert plan.final_replanned_n_req == 3
    assert plan.category == "very_wide"
    assert plan.geometry_patch_count == 3
    assert plan.redundant_geometry_patch_count == 0
    assert plan.small_fragment_count == 0
    assert plan.patch_island_count == 0


def test_oversized_object_replans_with_the_next_direct_orb_tracking_candidate() -> None:
    region = VideoObjectRegion("shelf", (90.0, 410.0), collar_px=10)
    candidates = {
        "T0": VideoTrackingSourcePlan("T0", _sources((0, 120), (100, 210), (190, 300), (280, 440))),
        "T1": VideoTrackingSourcePlan("T1", _sources((0, 120), (100, 210), (190, 300), (280, 440))),
    }

    candidate, plan = replan_wide_object_patches(region, candidates)

    assert candidate == "T1"
    assert plan.initial_n_req == 4
    assert plan.final_replanned_n_req == 4
    assert plan.category == "oversized"
    assert plan.replan_reason == "initial_N_req_gt_3_reselected_denser_direct_orb_tracking"


def test_plan_rejects_non_direct_orb_and_noncontinuous_coverage() -> None:
    with pytest.raises(ValueError, match="non-direct-ORB"):
        VideoDirectSourceSupport(1, (0.0, 10.0), direct_orb=False)
    with pytest.raises(RuntimeError, match="no continuous"):
        plan_object_patches(
            VideoObjectRegion("gap", (5.0, 30.0), collar_px=0),
            _sources((0.0, 10.0), (20.0, 40.0)),
        )
