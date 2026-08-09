from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from panorama_demo.video_s1_handoff import (
    BoundaryCandidate,
    HandoffCoordinateTransform,
    HandoffObservation,
    analysis_to_calibrated_x,
    attach_aligned_depth_risk,
    attach_background_residual,
    build_forbidden_intervals,
    estimate_background_canvas_dx,
    extract_calibrated_handoff_observations,
    generate_boundary_candidates,
    solve_global_straight_boundaries,
    split_solver_heldout,
)


def _observation(
    match_id: str,
    left_x: float,
    right_x: float,
    y: float,
    *,
    dx: float | None = None,
    role: str = "solver",
) -> HandoffObservation:
    canvas_dx = right_x - left_x if dx is None else dx
    return HandoffObservation(
        match_id=match_id,
        origin_pair_lineage="477->501",
        left_frame_id=477,
        right_frame_id=501,
        role=role,
        left_analysis_x=left_x,
        left_y=y,
        right_analysis_x=right_x,
        right_y=y,
        left_calibrated_x=left_x,
        right_calibrated_x=right_x,
        left_canvas_x=left_x,
        right_canvas_x=right_x,
        forward_backward_error_px=0.1,
        vertical_error_px=0.0,
        texture_score=2.0,
        observation_canvas_dx=canvas_dx,
    )


def test_analysis_calibrated_canvas_pixel_center_conversion() -> None:
    transform = HandoffCoordinateTransform(424, 848, 37.0)

    assert analysis_to_calibrated_x(0.0, 424, 848) == 0.5
    assert transform.analysis_to_calibrated_x(10.0) == 20.5
    assert transform.calibrated_to_analysis_x(20.5) == 10.0
    assert transform.analysis_to_canvas_x(10.0) == 57.5


def test_calibrated_gftt_lk_forward_backward_produces_stable_canvas_evidence() -> None:
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, (96, 160), dtype=np.uint8)
    right = cv2.warpAffine(
        left,
        np.float32([[1, 0, 4], [0, 1, 0]]),
        (160, 96),
        borderMode=cv2.BORDER_REFLECT,
    )
    valid = np.ones(left.shape, dtype=bool)
    left_transform = HandoffCoordinateTransform(160, 320, 100.0)
    # Four A pixels equal eight C pixels; support cancels background motion.
    right_transform = HandoffCoordinateTransform(160, 320, 92.0)

    first = extract_calibrated_handoff_observations(
        left,
        right,
        valid,
        valid,
        left_transform,
        right_transform,
        calibration_id="cal-1",
        origin_pair_lineage="0->1",
        left_frame_id=10,
        right_frame_id=11,
        corridor_canvas=(120, 380),
        maximum_vertical_match_error_px=1.0,
    )
    second = extract_calibrated_handoff_observations(
        left,
        right,
        valid,
        valid,
        left_transform,
        right_transform,
        calibration_id="cal-1",
        origin_pair_lineage="0->1",
        left_frame_id=10,
        right_frame_id=11,
        corridor_canvas=(120, 380),
        maximum_vertical_match_error_px=1.0,
    )

    assert len(first) >= 12
    assert [item.match_id for item in first] == [item.match_id for item in second]
    assert np.median([abs(item.observation_canvas_dx) for item in first]) < 0.25
    assert max(item.forward_backward_error_px for item in first) <= 0.75


def test_split_is_stable_and_heldout_never_enters_background_solver() -> None:
    observations = tuple(_observation(f"{index:04x}" + "0" * 12, 20, 20, index) for index in range(40))
    first = split_solver_heldout(observations, heldout_fraction=0.25)
    second = split_solver_heldout(tuple(reversed(observations)), heldout_fraction=0.25)

    assert [item.match_id for item in first.solver] == [item.match_id for item in second.solver]
    assert [item.match_id for item in first.heldout] == [item.match_id for item in second.heldout]
    assert not set(item.match_id for item in first.solver) & set(item.match_id for item in first.heldout)
    assert all(item.role == "solver" for item in first.solver)
    assert all(item.role == "heldout" for item in first.heldout)
    with np.testing.assert_raises(ValueError):
        estimate_background_canvas_dx(first.heldout)


def test_background_residual_and_multi_point_cluster_create_hard_interval() -> None:
    background = tuple(_observation(f"b{index}", 10 + index, 10 + index, index * 7) for index in range(8))
    foreground = tuple(
        _observation(f"f{index}", 50 + 0.2 * index, 56 + 0.2 * index, 10 + index * 9)
        for index in range(4)
    )
    all_observations = background + foreground
    background_dx = estimate_background_canvas_dx(all_observations)
    residuals = attach_background_residual(all_observations, background_dx)
    intervals = build_forbidden_intervals(residuals, guard_px=0.0)

    assert abs(background_dx) < 0.01
    assert len(intervals) == 1
    assert intervals[0].match_count == 4
    assert intervals[0].median_relative_dx_px == 6.0
    assert intervals[0].left_x <= 50 and intervals[0].right_x >= 56


def test_single_point_and_unclustered_strong_edges_stay_soft() -> None:
    isolated = attach_background_residual((_observation("one", 50, 58, 20),), 0.0)

    assert build_forbidden_intervals(isolated) == ()


def test_soft_crossing_cost_protects_zero_residual_projection_interval() -> None:
    # A zero background-relative residual is not a zero copy-risk: the two raw
    # canvas projections still straddle every boundary in [10, 14).
    observation = replace(
        _observation("background", 10, 14, 20),
        relative_dx=0.0,
    )
    candidates = generate_boundary_candidates(
        midpoint_x=12,
        allowed_left_x=9,
        allowed_right_x=15,
        forbidden_intervals=(),
        soft_observations=(observation,),
    )
    solution = solve_global_straight_boundaries(
        (candidates,),
        midpoint_boundaries=(12,),
        first_owner_left_x=0,
        last_owner_right_x=30,
    )

    assert not 10 <= solution.boundaries[0] < 14
    assert next(item for item in candidates if item.x == 12).soft_crossing_cost == 1.0

    with np.testing.assert_raises(ValueError):
        generate_boundary_candidates(
            midpoint_x=12,
            allowed_left_x=9,
            allowed_right_x=15,
            forbidden_intervals=(),
            soft_observations=(replace(observation, role="heldout"),),
        )


def test_aligned_depth_only_weights_existing_rgb_observations() -> None:
    observations = (
        _observation("edge", 5, 5, 5),
        _observation("inconsistent", 12, 12, 12),
        _observation("missing", 19, 19, 19),
    )
    left = np.full((24, 24), 1000.0, dtype=np.float32)
    right = np.full((24, 24), 1000.0, dtype=np.float32)
    left[5, 6] = 1100.0
    right[11:14, 11:14] = 1100.0
    left[19, 19] = 0.0
    right[19, 19] = 0.0

    weighted = attach_aligned_depth_risk(observations, left, right)

    assert len(weighted) == len(observations)
    assert weighted[0].left_depth_edge
    assert weighted[0].depth_risk_weight == 2.0
    assert not weighted[1].left_depth_edge
    assert not weighted[1].right_depth_edge
    assert weighted[1].depth_risk_weight == 0.5
    assert weighted[2].left_depth_mm is None
    assert weighted[2].right_depth_mm is None
    assert weighted[2].depth_risk_weight == 1.0
    # Depth cannot independently make a hard interval; the RGB residual gate
    # remains authoritative even for an edge-weighted observation.
    zero_residual = tuple(replace(item, relative_dx=0.0) for item in weighted)
    assert build_forbidden_intervals(
        zero_residual,
        minimum_match_count=1,
        minimum_vertical_span_px=0.0,
    ) == ()


def test_forbidden_interval_is_half_open_and_safe_line_shifts_from_midpoint() -> None:
    cluster = tuple(
        replace(
            _observation(f"f{index}", 98, 104, index * 10),
            relative_dx=6.0,
        )
        for index in range(4)
    )
    interval = build_forbidden_intervals(cluster, guard_px=0.0)[0]
    assert interval.contains(interval.left_x)
    assert interval.contains((interval.left_x + interval.right_x) / 2)
    assert not interval.contains(interval.right_x)

    candidates = generate_boundary_candidates(
        midpoint_x=101,
        allowed_left_x=90,
        allowed_right_x=110,
        forbidden_intervals=(interval,),
    )
    solution = solve_global_straight_boundaries(
        (candidates,),
        midpoint_boundaries=(101,),
        first_owner_left_x=0,
        last_owner_right_x=200,
    )

    assert solution.feasible
    assert not interval.contains(solution.boundaries[0])
    assert solution.decisions[0].classification == "straight_boundary_shifted"
    assert not solution.decisions[0].needs_s2


def test_safe_midpoint_is_retained_and_unobservable_midpoint_requests_review_not_s2() -> None:
    candidates = generate_boundary_candidates(
        midpoint_x=100,
        allowed_left_x=90,
        allowed_right_x=110,
        forbidden_intervals=(),
    )
    safe = solve_global_straight_boundaries(
        (candidates,), midpoint_boundaries=(100,),
        first_owner_left_x=0, last_owner_right_x=200,
    )
    unknown = solve_global_straight_boundaries(
        (candidates,), midpoint_boundaries=(100,), observable=(False,),
        first_owner_left_x=0, last_owner_right_x=200,
    )

    assert safe.boundaries == (100,)
    assert safe.decisions[0].classification == "midpoint_safe"
    assert unknown.boundaries == (100,)
    assert unknown.decisions[0].classification == "unobservable_midpoint"
    assert unknown.decisions[0].manual_review_required
    assert not unknown.decisions[0].needs_s2


def test_global_dp_enforces_monotonic_minimum_owner_width() -> None:
    candidate_sets = (
        (
            BoundaryCandidate(20, 0.0, False),
            BoundaryCandidate(30, 0.2, False),
        ),
        (
            BoundaryCandidate(25, 0.0, False),
            BoundaryCandidate(40, 0.2, False),
        ),
    )
    solution = solve_global_straight_boundaries(
        candidate_sets,
        midpoint_boundaries=(20, 25),
        first_owner_left_x=0,
        last_owner_right_x=60,
        minimum_internal_owner_width_px=9,
    )

    assert solution.feasible
    assert solution.boundaries == (20, 40)
    assert np.diff(solution.boundaries)[0] >= 9
