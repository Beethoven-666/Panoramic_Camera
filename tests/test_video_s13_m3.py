from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s13_motion import (
    S13MotionEdge,
    S13MotionHypothesis,
    S13MotionObservation,
    S13Progress,
    _extract_hypotheses,
)
from panorama_demo.video_s13_progress import (
    build_s13_m3_layout,
    select_s13_m3_primary_scan,
    selected_hypothesis_ids_for_spatial_sources,
)
from panorama_demo.video_s13_schedule import plan_s13_m3_schedule
from panorama_demo.video_s13_session import S13RenderFrame
from panorama_demo.video_s13_trajectory import S13Trajectory, _islands


def _frame(frame_id: int) -> S13RenderFrame:
    return S13RenderFrame(frame_id, Path(f"{frame_id}.png"), frame_id * 1000, 100, 40)


def _edge(left: int, right: int, hypotheses: tuple[S13MotionHypothesis, ...]) -> S13MotionEdge:
    return S13MotionEdge(
        left, right, 1, 8.0, 0.0, 12.0, 20, 8.0, 0.0, 1.0, 8.0, "grid_lk", False, (),
        motion_hypotheses=hypotheses,
    )


def _hypothesis(hypothesis_id: int, advance: float, cell: str, confidence: float = 0.9) -> S13MotionHypothesis:
    return S13MotionHypothesis(
        hypothesis_id, advance, 0.0, 10.0, 0.1, 0.5, 0.5, 1.0, 0.8, 0.5, 0.5,
        f"signature-{cell}", "grid_lk_weighted_histogram", confidence, (cell,),
    )


def test_motion_hypotheses_are_data_driven_zero_to_three() -> None:
    assert _extract_hypotheses((), selected_advance_px=None, image_width_px=100, image_height_px=40) == ()
    observations = tuple(
        S13MotionObservation(float(index * 4), float(index % 5 * 7), 4.0 if index < 12 else 11.0,
                             0.1, 1.0, f"{index % 5}:{index // 5}")
        for index in range(24)
    )
    hypotheses = _extract_hypotheses(
        observations, selected_advance_px=4.0, image_width_px=100, image_height_px=40
    )
    assert 1 <= len(hypotheses) <= 3
    assert all(item.support_signature_sha256 and item.support_cells for item in hypotheses)


def test_whole_chain_dp_prefers_coherent_support_lineage() -> None:
    frames = tuple(_frame(index) for index in range(4))
    edges = tuple(
        _edge(index, index + 1, (_hypothesis(0, 8.0, "a", 0.8), _hypothesis(1, 20.0, f"x{index}", 0.81)))
        for index in range(3)
    )
    trajectory = S13Trajectory("ignore_pose", None, {}, {}, (), {})
    layout = build_s13_m3_layout(frames, edges, trajectory)
    assert layout.layout_level == "L0_coherent_rgb"
    assert [step.selected_hypothesis_id for step in layout.lineage] == [0, 0, 0]
    assert layout.metric_scale_claim is False
    assert layout.pose_supported is False


def test_source_u_cap_rescues_real_intermediate_frames() -> None:
    calibration = CameraIntrinsics(100, 40, 80.0, 80.0, 49.5, 19.5, ())
    progress = S13Progress(tuple(range(9)), tuple(float(index * 30) for index in range(9)),
                           ("origin",) + ("motion_hypothesis",) * 8, True, True, True)
    plan = plan_s13_m3_schedule(progress, (), calibration, normal_target_advance_px=90.0,
                                risky_target_advance_px=90.0, source_u_fraction_cap=0.20)
    assert plan.rescued_frame_ids
    assert all(assignment.width <= plan.source_u_cap_px for schedule in plan.schedules
               for assignment in schedule.assignments[1:-1])


def test_pose_only_layout_is_normalized_nonmetric() -> None:
    frames = tuple(_frame(index) for index in range(3))
    poses = {}
    for index in range(3):
        pose = np.eye(4)
        pose[0, 3] = index * 1000.0
        poses[index] = pose
    trajectory = S13Trajectory("cache", None, poses, {index: "direct" for index in poses}, ((0, 1, 2),), {})
    layout = build_s13_m3_layout(frames, (), trajectory)
    assert layout.layout_level == "L2_normalized_pose"
    assert layout.metric_scale_claim is False
    assert layout.pixels_per_meter is None


def test_extreme_gaps_use_explicit_panel_fallback() -> None:
    calibration = CameraIntrinsics(100, 40, 80.0, 80.0, 49.5, 19.5, ())
    progress = S13Progress((0, 1, 2, 3), (0.0, 1000.0, 2000.0, 3000.0),
                           ("origin", "motion_hypothesis", "motion_hypothesis", "motion_hypothesis"),
                           True, True, True)
    plan = plan_s13_m3_schedule(progress, (), calibration, normal_target_advance_px=1.0,
                                risky_target_advance_px=1.0, source_u_fraction_cap=0.20)
    assert len(plan.schedules) > 1
    assert plan.panel_gap_frame_ids
    assert plan.cap_satisfied is True


def test_pose_islands_split_only_on_explicit_epoch_not_missing_pose() -> None:
    assert _islands((0, 1, 2, 3), {0, 2, 3}, {0: "a", 2: "a", 3: "b"}) == ((0, 2), (3,))


def test_l3_temporal_slit_is_nonspatial() -> None:
    frames = tuple(_frame(index) for index in range(4))
    layout = build_s13_m3_layout(frames, (), S13Trajectory("ignore_pose", None, {}, {}, (), {}))
    assert layout.layout_level == "L3_temporal_slit"
    assert layout.progress.spatial is False
    assert set(layout.progress.centers_x) == {0.0}
    assert all(method == "F7_temporal_order" for method in layout.progress.placement_methods[1:])


def test_primary_scan_trims_stationary_lead_and_tail_after_full_session_gate_fails() -> None:
    frames = tuple(_frame(index) for index in range(14))
    values = (0.0,) * 5 + (8.0,) * 4 + (0.0,) * 4
    edges = tuple(
        S13MotionEdge(
            index,
            index + 1,
            1,
            value,
            0.0,
            12.0,
            20,
            value,
            0.0,
            1.0,
            value,
            "grid_lk",
            False,
            (),
            motion_hypotheses=(
                () if abs(value) < 0.25 else (_hypothesis(index, value, "a"),)
            ),
        )
        for index, value in enumerate(values)
    )

    selected = select_s13_m3_primary_scan(
        frames,
        edges,
        S13Trajectory("ignore_pose", None, {}, {}, (), {}),
        image_width=100,
    )

    assert selected.layout.progress.spatial is True
    assert tuple(frame.frame_id for frame in selected.frames) == (5, 6, 7, 8, 9)
    assert selected.audit["mode"] == "trimmed_primary_one_way"
    assert selected.audit["leading_discarded_frame_count"] == 5
    assert selected.audit["trailing_discarded_frame_count"] == 4
    assert selected.audit["displacement_px"] == 32.0


def test_primary_scan_does_not_recover_static_session() -> None:
    frames = tuple(_frame(index) for index in range(8))
    edges = tuple(
        S13MotionEdge(
            index, index + 1, 1, 0.0, 0.0, 12.0, 20, 0.0, 0.0, 1.0,
            0.0, "grid_lk", False, (),
        )
        for index in range(7)
    )

    selected = select_s13_m3_primary_scan(
        frames,
        edges,
        S13Trajectory("ignore_pose", None, {}, {}, (), {}),
        image_width=100,
    )

    assert selected.layout.progress.spatial is False
    assert selected.audit["mode"] == "full_session_no_recoverable_segment"


def test_primary_scan_keeps_an_already_spatial_full_session_unchanged() -> None:
    frames = tuple(_frame(index) for index in range(5))
    edges = tuple(
        _edge(index, index + 1, (_hypothesis(index, 8.0, "a"),))
        for index in range(4)
    )
    trajectory = S13Trajectory("ignore_pose", None, {}, {}, (), {})
    expected = build_s13_m3_layout(frames, edges, trajectory)

    selected = select_s13_m3_primary_scan(
        frames,
        edges,
        trajectory,
        image_width=100,
    )

    assert selected.frames == frames
    assert selected.edges == edges
    assert selected.layout.progress == expected.progress
    assert selected.audit["mode"] == "full_session"
    assert selected.audit["leading_discarded_frame_count"] == 0
    assert selected.audit["trailing_discarded_frame_count"] == 0


def test_signed_reverse_run_creates_panel_segment_break() -> None:
    frames = tuple(_frame(index) for index in range(6))
    values = (8.0, 9.0, 8.0, -7.0, -8.0)
    edges = tuple(
        S13MotionEdge(
            index, index + 1, 1, value, 0.0, 12.0, 20, value, 0.0, 1.0, value,
            "grid_lk", False, (), motion_hypotheses=(_hypothesis(0, value, "a"),),
        )
        for index, value in enumerate(values)
    )
    layout = build_s13_m3_layout(frames, edges, S13Trajectory("ignore_pose", None, {}, {}, (), {}))
    assert layout.canonical_scan_direction == 1
    assert layout.segment_break_pairs == ((3, 4),)
    assert layout.progress.centers_x[-1] == layout.progress.centers_x[3]
    plan = plan_s13_m3_schedule(
        layout.progress, edges, CameraIntrinsics(100, 40, 80.0, 80.0, 49.5, 19.5, ()),
        segment_break_pairs=layout.segment_break_pairs,
    )
    assert len(plan.schedules) == 2


def test_negative_scan_orders_real_sources_right_to_left_before_p0() -> None:
    frames = tuple(_frame(index) for index in range(4))
    edges = tuple(
        S13MotionEdge(
            index, index + 1, 1, -8.0, 0.0, 12.0, 20, -8.0, 0.0, 1.0,
            -8.0, "grid_lk", False, (),
            motion_hypotheses=(_hypothesis(index, -8.0, "a"),),
        )
        for index in range(3)
    )
    layout = build_s13_m3_layout(
        frames, edges, S13Trajectory("ignore_pose", None, {}, {}, (), {})
    )
    assert layout.canonical_scan_direction == -1
    assert layout.progress.frame_ids == (0, 1, 2, 3)
    assert layout.progress.centers_x == (0.0, 8.0, 16.0, 24.0)

    calibration = CameraIntrinsics(100, 40, 80.0, 80.0, 49.5, 19.5, ())
    plan = plan_s13_m3_schedule(
        layout.progress,
        edges,
        calibration,
        normal_target_advance_px=1.0,
        risky_target_advance_px=1.0,
        canonical_scan_direction=layout.canonical_scan_direction,
    )
    selection = plan.selections[0]
    schedule = plan.schedules[0]
    assert selection.frame_ids == (3, 2, 1, 0)
    assert tuple(item.frame_id for item in schedule.assignments) == (3, 2, 1, 0)
    assert tuple(item.center_x for item in schedule.assignments) == (49.5, 57.5, 65.5, 73.5)
    assert selection.risk_by_frame_id == {3: False, 2: False, 1: False, 0: False}
    assert selected_hypothesis_ids_for_spatial_sources(
        layout, selection.frame_ids
    ) == (-1, 2, 1, 0)


def test_sparse_direct_pose_uses_cumulative_rgb_between_endpoints() -> None:
    frames = tuple(_frame(index) for index in range(7))
    edges = tuple(_edge(index, index + 1, (_hypothesis(0, 8.0, "a"),)) for index in range(6))
    poses = {}
    for index in (0, 2, 4, 6):
        pose = np.eye(4)
        pose[0, 3] = index * 100.0
        poses[index] = pose
    trajectory = S13Trajectory(
        "cache", None, poses, {index: "direct" for index in poses}, ((0, 2, 4, 6),), {},
        {index: "epoch-a" for index in poses},
    )
    layout = build_s13_m3_layout(frames, edges, trajectory)
    assert layout.pose_calibration_pairs == 3
    assert layout.pixels_per_meter == 80.0
    assert layout.audit["pose_interpolation_used"] is False
    assert layout.audit["pose_calibration_reason"] is None
