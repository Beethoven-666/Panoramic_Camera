from __future__ import annotations

import numpy as np

from panorama_demo.video_s1_config import S1SourceSelectionConfig, S11SourceSelectionConfig
from panorama_demo.video_s1_layout import (
    S1MotionEdge,
    S1PairLineage,
    build_source_layout,
    cumulative_monotonic_progress,
    inherit_s01_v1_exact,
    insert_local_real_source_rescue,
    progress_analysis_to_full_resolution,
    select_local_real_source_rescue,
    select_dynamic_real_sources,
    validate_owner_partition,
    validate_refined_layout_invariants,
)


CONFIG = S1SourceSelectionConfig()


def _edges(values, *, unreliable=()):
    return tuple(S1MotionEdge(dx=value, reliable=index not in unreliable) for index, value in enumerate(values))


def test_monotonic_progress_retains_reverse_jitter_without_moving_canvas_backwards() -> None:
    progress = cumulative_monotonic_progress(_edges((4.0, 5.0, -2.0, 6.0)), direction=1)

    np.testing.assert_allclose(progress, (0.0, 4.0, 9.0, 9.0, 15.0))


def test_dynamic_selection_is_ordered_real_unique_and_keeps_endpoints() -> None:
    frame_ids = tuple(range(10, 20))
    selected = select_dynamic_real_sources(frame_ids, _edges((3.0,) * 9), CONFIG, direction=1)

    assert selected[0] == 0 and selected[-1] == len(frame_ids) - 1
    assert selected == tuple(sorted(set(selected)))
    assert [frame_ids[index] for index in selected] == [10, 14, 18, 19]


def test_fast_or_unreliable_motion_selects_adjacent_real_sources_more_densely() -> None:
    frame_ids = tuple(range(8))
    fast = select_dynamic_real_sources(frame_ids, _edges((25.0,) * 7), CONFIG, direction=1)
    risky = select_dynamic_real_sources(
        frame_ids, _edges((3.0,) * 7, unreliable=(2, 3)), CONFIG, direction=1
    )

    assert fast == tuple(range(8))
    assert 3 in risky and 4 in risky
    assert all(frame_ids[index] == index for index in fast + risky)


def test_nearby_source_candidate_prefers_clearer_real_frame() -> None:
    qualities = [
        {"sharpness": 10.0, "texture_coverage": 0.2, "dark_ratio": 0.0, "saturated_ratio": 0.0}
        for _ in range(6)
    ]
    qualities[3] = {"sharpness": 300.0, "texture_coverage": 0.8, "dark_ratio": 0.0, "saturated_ratio": 0.0}
    selected = select_dynamic_real_sources(
        tuple(range(6)), _edges((3.5,) * 5), CONFIG, direction=1,
        source_qualities=qualities,
    )

    assert 3 in selected
    assert selected == tuple(sorted(set(selected)))


def test_owner_midpoints_form_one_complete_partition_and_shoulders_may_overlap() -> None:
    frame_ids = (100, 101, 102, 103, 104)
    source_indices = (0, 2, 4)
    centers = (100.0, 112.0, 128.0)
    supports = ((0, 200), (12, 212), (28, 228))
    layouts = build_source_layout(frame_ids, source_indices, centers, supports, CONFIG)

    validate_owner_partition(layouts)
    assert [(item.owner_left_x, item.owner_right_x) for item in layouts] == [
        (0, 106), (106, 120), (120, 228)
    ]
    assert layouts[0].frame_id == 100 and layouts[-1].frame_id == 104
    assert layouts[0].owner_left_x == layouts[0].support_left_x
    assert layouts[-1].owner_right_x == layouts[-1].support_right_x
    assert layouts[0].match_right_x > layouts[1].match_left_x
    assert layouts[1].measured_progress_from_previous_px == 12.0


def test_narrow_support_reduces_shoulder_without_invalidating_s0_owner() -> None:
    layouts = build_source_layout(
        (7, 8), (0, 1), (50.0, 62.0), ((0, 60), (55, 112)), CONFIG
    )

    validate_owner_partition(layouts)
    boundary = layouts[0].owner_right_x
    assert boundary == layouts[1].owner_left_x
    assert layouts[0].match_right_x - boundary < CONFIG.shoulder_minimum_px
    assert layouts[0].owner_left_x < layouts[0].owner_right_x


def test_s11_exact_inheritance_copies_every_s01_layout_field_and_canvas() -> None:
    baseline = build_source_layout(
        (100, 101, 102, 103, 104),
        (0, 2, 4),
        (100.0, 112.0, 128.0),
        ((0, 200), (12, 212), (28, 228)),
        CONFIG,
    )
    inherited = inherit_s01_v1_exact(
        [item.__dict__ for item in baseline], tuple(range(100, 105)), expected_canvas_width=228
    )

    assert inherited == baseline
    assert inherited is not baseline
    assert inherited[0].owner_left_x == inherited[0].support_left_x
    assert inherited[-1].owner_right_x == inherited[-1].support_right_x


def test_s11_exact_inheritance_rejects_relayout_or_endpoint_changes() -> None:
    baseline = build_source_layout(
        (10, 11, 12), (0, 1, 2), (10.0, 22.0, 34.0), ((0, 50), (12, 62), (24, 74)), CONFIG
    )
    changed = [dict(item.__dict__) for item in baseline]
    changed[0]["owner_left_x"] = 1
    with np.testing.assert_raises_regex(ValueError, "complete left endpoint"):
        inherit_s01_v1_exact(changed, (10, 11, 12))

    changed = [dict(item.__dict__) for item in baseline]
    changed[0]["owner_right_x"] += 1
    changed[1]["owner_left_x"] += 1
    with np.testing.assert_raises_regex(ValueError, "exact midpoint"):
        inherit_s01_v1_exact(changed, (10, 11, 12))


def test_rescue_uses_full_resolution_progress_and_stable_candidate_ranking() -> None:
    config = S11SourceSelectionConfig()
    progress = (0.0, 3.0, 5.0, 7.0, 10.0)
    np.testing.assert_allclose(
        progress_analysis_to_full_resolution(progress, calibrated_width=640, analysis_width=320),
        (0.0, 6.0, 10.0, 14.0, 20.0),
    )
    rescue = select_local_real_source_rescue(
        (10, 11, 12, 13, 14),
        progress,
        0,
        4,
        config,
        calibrated_width=640,
        analysis_width=320,
        source_quality_penalties=(0.0, 0.1, 0.5, 0.0, 0.0),
    )

    assert rescue is not None
    assert rescue.frame_index == 2
    assert rescue.frame_id == 12
    assert rescue.origin_pair_lineage.frame_ids == (10, 14)
    assert rescue.maximum_child_step_fullres_px == 10.0
    assert rescue.improvement_fullres_px == 10.0
    assert insert_local_real_source_rescue((0, 4), rescue) == (0, 2, 4)


def test_rescue_requires_real_rgb_improvement_and_obeys_lineage_limits() -> None:
    config = S11SourceSelectionConfig()
    common = dict(
        frame_ids=(20, 21, 22, 23),
        progress_analysis_px=(0.0, 1.0, 2.0, 3.0),
        left_source_index=0,
        right_source_index=3,
        config=config,
        calibrated_width=320,
        analysis_width=320,
        origin_pair_lineage=S1PairLineage(20, 23),
    )
    assert select_local_real_source_rescue(**common, rgb_available=(True, False, False, True)) is None
    assert select_local_real_source_rescue(**common, refinement_round=3) is None
    assert select_local_real_source_rescue(
        **common, prior_inserted_frame_indices=(1, 2)
    ) is None


def test_two_round_rescue_preserves_original_endpoints_and_origin_lineage() -> None:
    config = S11SourceSelectionConfig()
    frame_ids = tuple(range(30, 37))
    progress = tuple(float(index * 4) for index in range(7))
    lineage = S1PairLineage(30, 36)
    first = select_local_real_source_rescue(
        frame_ids, progress, 0, 6, config,
        calibrated_width=640, analysis_width=320, origin_pair_lineage=lineage,
    )
    assert first is not None
    sources = insert_local_real_source_rescue((0, 6), first)
    second = select_local_real_source_rescue(
        frame_ids, progress, first.frame_index, 6, config,
        calibrated_width=640, analysis_width=320, origin_pair_lineage=lineage,
        refinement_round=2, prior_inserted_frame_indices=(first.frame_index,),
    )
    assert second is not None
    sources = insert_local_real_source_rescue(sources, second)

    assert sources[0] == 0 and sources[-1] == 6
    assert first.origin_pair_lineage == second.origin_pair_lineage == lineage
    assert len(sources) == 4


def test_refined_layout_validation_rejects_canvas_extent_drift() -> None:
    initial = build_source_layout(
        (10, 11, 12), (0, 2), (20.0, 44.0), ((0, 64), (24, 88)), CONFIG
    )
    refined = build_source_layout(
        (10, 11, 12), (0, 1, 2), (20.0, 32.0, 44.0),
        ((0, 64), (12, 76), (24, 88)), CONFIG,
    )
    validate_refined_layout_invariants(initial, refined)

    drifted = build_source_layout(
        (10, 11, 12), (0, 1, 2), (20.0, 32.0, 44.0),
        ((0, 64), (12, 76), (24, 89)), CONFIG,
    )
    with np.testing.assert_raises_regex(ValueError, "canvas extent"):
        validate_refined_layout_invariants(initial, drifted)
