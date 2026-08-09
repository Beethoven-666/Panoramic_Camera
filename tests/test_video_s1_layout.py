from __future__ import annotations

import numpy as np

from panorama_demo.video_s1_config import S1SourceSelectionConfig
from panorama_demo.video_s1_layout import (
    S1MotionEdge,
    build_source_layout,
    cumulative_monotonic_progress,
    select_dynamic_real_sources,
    validate_owner_partition,
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
