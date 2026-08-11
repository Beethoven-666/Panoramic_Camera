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
from panorama_demo.video_s13_progress import build_s13_m3_layout
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
