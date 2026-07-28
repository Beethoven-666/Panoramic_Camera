from __future__ import annotations

import numpy as np

from panorama_demo.dis_track_direct_handoff import (
    DirectHandoffConfig,
    DirectProjectedObservation,
    evaluate_direct_track,
    target_mask_shape_audit,
)


def _observation(
    candidate_id: int,
    frame_id: int,
    mask: np.ndarray,
    *,
    clarity: float = 10.0,
) -> DirectProjectedObservation:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask] = (10, 80, 220)
    return DirectProjectedObservation(
        candidate_id=candidate_id,
        frame_id=frame_id,
        source_panel_index=candidate_id,
        target_panel_index=1,
        target_mask=np.ascontiguousarray(mask),
        target_image_bgr=image,
        source_depth_coverage_ratio=0.98,
        clarity=clarity,
        projection_audit={
            "direct_world_projection": True,
            "fitted_display_warp": False,
            "rgb_generated": False,
            "pose_modified": False,
            "blend_used": False,
        },
    )


def test_two_consistent_direct_masks_choose_one_real_owner() -> None:
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 15:45] = True
    first = _observation(0, 10, mask, clarity=20.0)
    second = _observation(1, 20, mask, clarity=40.0)
    decision = evaluate_direct_track(7, [first, second])
    assert decision.accepted is True
    assert decision.selected_observation is second
    assert decision.audit["consistent_projection_count"] == 2
    assert decision.audit["selected_target_union_coverage_ratio"] == 1.0
    assert decision.audit["translation_used"] is False
    assert decision.audit["affine_used"] is False
    assert decision.audit["fitted_warp_used"] is False
    assert decision.audit["generated_rgb_used"] is False


def test_inconsistent_target_masks_fail_iou_gate() -> None:
    first_mask = np.zeros((40, 80), dtype=bool)
    second_mask = np.zeros_like(first_mask)
    first_mask[10:30, 5:25] = True
    second_mask[10:30, 55:75] = True
    decision = evaluate_direct_track(
        8,
        [
            _observation(0, 10, first_mask),
            _observation(1, 20, second_mask),
        ],
    )
    assert decision.accepted is False
    assert (
        decision.audit["reason"]
        == "no_two_selected_panel_target_masks_are_iou_consistent"
    )


def test_target_holes_are_audited_without_filling() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[10:30, 10:30] = False
    before = mask.copy()
    audit = target_mask_shape_audit(mask)
    assert audit["internal_hole_pixel_count"] == 400
    assert audit["internal_hole_ratio"] > 0.10
    assert np.array_equal(mask, before)
    config = DirectHandoffConfig(maximum_target_internal_hole_ratio=0.10)
    decision = evaluate_direct_track(
        9,
        [
            _observation(0, 10, mask),
            _observation(1, 20, mask),
        ],
        config=config,
    )
    assert decision.accepted is False
    assert (
        decision.audit["reason"]
        == "fewer_than_two_fixed_gate_direct_projections"
    )


def test_fixed_gates_require_two_selected_panel_observations() -> None:
    config = DirectHandoffConfig()
    config.validate()
    assert config.minimum_projection_count == 2
    assert config.minimum_pair_target_iou == 0.50
    assert config.minimum_selected_union_coverage_ratio == 0.90
