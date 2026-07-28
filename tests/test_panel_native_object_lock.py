from __future__ import annotations

import numpy as np

from panorama_demo.inspection_fastsam_track import FastSAMRGBDCandidate
from panorama_demo.panel_native_object_lock import (
    PanelNativeLockConfig,
    baseline_pair_costs,
    map_mask_through_existing_inverse,
    observation_identity_audit,
)


def _candidate(candidate_id: int, frame_id: int) -> FastSAMRGBDCandidate:
    return FastSAMRGBDCandidate(
        candidate_id=candidate_id,
        source_index=candidate_id,
        frame_id=frame_id,
        polygon_xy=np.asarray(
            [[4, 4], [7, 4], [7, 7], [4, 7]], dtype=np.int32
        ),
        bbox_xywh=(4, 4, 4, 4),
        source_area_pixels=16,
        depth_coverage_ratio=1.0,
        world_voxel_hashes=frozenset(range(20)),
        world_dilated_voxel_hashes=frozenset(range(20)),
        world_centroid_mm=(100.0 + candidate_id, 20.0, 400.0),
        world_spans_mm=(40.0, 40.0, 20.0),
        median_lab=(120.0, 128.0, 128.0),
        aspect_ratio=1.0,
        solidity=1.0,
    )


def _mapped(candidate_id: int, frame_id: int, panel_index: int):
    yy, xx = np.indices((12, 12), dtype=np.float32)
    source_mask = np.zeros((12, 12), dtype=bool)
    source_mask[4:8, 4:8] = True
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    image[source_mask] = (20, 80, 200)
    observation, audit = map_mask_through_existing_inverse(
        candidate=_candidate(candidate_id, frame_id),
        panel_index=panel_index,
        frame_id=frame_id,
        source_mask=source_mask,
        source_image_bgr=image,
        inverse_map_x=xx,
        inverse_map_y=yy,
        inverse_valid_mask=np.ones((12, 12), dtype=bool),
        corner_x=0,
        canvas_shape=(12, 12),
    )
    assert observation is not None
    assert audit["accepted"] is True
    return observation


def test_whole_mask_uses_existing_inverse_map_and_real_rgb_only() -> None:
    observation = _mapped(0, 10, 0)
    assert np.array_equal(observation.target_mask, observation.source_mask)
    assert np.all(
        observation.target_image_bgr[observation.target_mask]
        == np.asarray([20, 80, 200], dtype=np.uint8)
    )
    assert observation.audit["translation_used"] is False
    assert observation.audit["affine_used"] is False
    assert observation.audit["new_warp_used"] is False
    assert observation.audit["fill_used"] is False
    assert observation.audit["generated_color_used"] is False


def test_incomplete_existing_inverse_map_fails_whole_mask_gate() -> None:
    yy, xx = np.indices((12, 12), dtype=np.float32)
    source_mask = np.zeros((12, 12), dtype=bool)
    source_mask[4:8, 4:8] = True
    valid = np.ones((12, 12), dtype=bool)
    valid[:, 6:] = False
    observation, audit = map_mask_through_existing_inverse(
        candidate=_candidate(0, 10),
        panel_index=0,
        frame_id=10,
        source_mask=source_mask,
        source_image_bgr=np.zeros((12, 12, 3), dtype=np.uint8),
        inverse_map_x=xx,
        inverse_map_y=yy,
        inverse_valid_mask=valid,
        corner_x=0,
        canvas_shape=(12, 12),
    )
    assert observation is None
    assert (
        audit["rejection_reason"]
        == "whole_source_mask_not_represented_by_existing_inverse_map"
    )


def test_identity_requires_two_views_and_keeps_masks_immutable() -> None:
    first = _mapped(0, 10, 0)
    second = _mapped(1, 20, 1)
    first_before = first.target_mask.copy()
    second_before = second.target_mask.copy()
    audit = observation_identity_audit(first, second)
    assert audit["pass"] is True
    assert audit["rgbd_world_role"] == "identity_and_merge_split_rejection_only"
    assert np.array_equal(first.target_mask, first_before)
    assert np.array_equal(second.target_mask, second_before)


def test_baseline_pair_costs_prefer_existing_closed_boundary() -> None:
    owner = np.zeros((5, 30), dtype=np.int16)
    owner[:, 15:] = 1
    costs = baseline_pair_costs(
        owner, [15.0], corridor_width_pixels=20
    )
    local = costs[0].values
    assert costs[0].corner_x == 5
    assert np.all(np.argmin(local, axis=1) + costs[0].corner_x == 15)
    assert np.all(local[:, 10] == 0.0)


def test_fixed_first_gate_configuration_requires_two_views() -> None:
    config = PanelNativeLockConfig()
    config.validate()
    assert config.minimum_view_count == 2
    assert config.minimum_inverse_source_coverage_ratio == 0.90
