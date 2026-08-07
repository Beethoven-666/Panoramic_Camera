from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo.quality import FrameQuality, MotionEstimate
from panorama_demo.session import RGBDFrame
from panorama_demo.video_motion_resampler import (
    MotionResamplingConfig,
    VideoRenderPlan,
    compose_selected_motions,
    insert_v6_real_rescue_sources,
    select_render_keyframes,
)
from panorama_demo.video_panorama import (
    _select_real_orb_tracking_indices,
    run_direct_orb_tracking_gate,
)
from panorama_demo.video_source_selection import (
    VideoFrontalityConfig,
    assess_video_source_frontality,
    off_axis_angle_degrees,
    plan_frontality_owner_spans,
)
from panorama_demo.session import CameraIntrinsics


def _frame(index: int) -> RGBDFrame:
    return RGBDFrame(
        frame_id=index,
        color_path=Path(f"color/{index}.jpg"),
        aligned_depth_path=Path(f"depth_aligned/{index}.png"),
        depth_scale_mm_per_unit=0.1,
        timestamp_us=index * 16_667,
        color_exposure_raw=1,
        color_gain=16,
    )


def _quality() -> FrameQuality:
    return FrameQuality(8.0, 25.0, 0.6, 0.05, 0.01, 0.4, 8.0)


def test_motion_resampler_retains_real_endpoints_and_reduces_dense_motion() -> None:
    frames = tuple(_frame(index) for index in range(11))
    motions = [MotionEstimate(2.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast") for _ in range(10)]

    plan = select_render_keyframes(
        frames,
        motions,
        full_resolution_scale=1.0,
        frame_width=848,
        qualities=[_quality() for _ in frames],
    )

    assert plan.source_indices[0] == 0
    assert plan.source_indices[-1] == 10
    assert len(plan.frames) < len(frames)
    assert [frame.frame_id for frame in plan.frames] == list(plan.source_indices)
    assert plan.as_dict()["interpolated_poses"] is False


def test_motion_resampler_keeps_risky_edges_denser_and_composes_real_motion() -> None:
    frames = tuple(_frame(index) for index in range(7))
    motions = [
        MotionEstimate(4.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast"),
        MotionEstimate(4.0, 7.0, 0, 0.0, 0.0, "dis_unreliable"),
        MotionEstimate(4.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast"),
        MotionEstimate(4.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast"),
        MotionEstimate(4.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast"),
        MotionEstimate(4.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast"),
    ]
    plan = select_render_keyframes(
        frames,
        motions,
        full_resolution_scale=1.0,
        frame_width=848,
        qualities=[_quality() for _ in frames],
        config=MotionResamplingConfig(normal_target_step_pixels=16.0, risk_target_step_pixels=8.0),
    )

    assert plan.high_risk_edge_count == 1
    combined = compose_selected_motions(motions, plan.source_indices)
    assert len(combined) == len(plan.frames) - 1
    assert sum(item.dx for item in combined) == 24.0
    assert all(item.method == "composed_video_motion" for item in combined)


def test_v6_rescue_inserts_only_existing_direct_orb_midpoints_for_oversized_gaps() -> None:
    frames = tuple(_frame(index) for index in range(5))
    motions = [MotionEstimate(5.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast") for _ in range(4)]
    plan = select_render_keyframes(
        frames,
        motions,
        full_resolution_scale=1.0,
        frame_width=848,
        qualities=[_quality() for _ in frames],
        config=MotionResamplingConfig(normal_target_step_pixels=8.0, risk_target_step_pixels=5.0),
    )

    rescued = insert_v6_real_rescue_sources(
        frames, motions, plan, full_resolution_scale=1.0, maximum_step_pixels=8.0,
    )

    assert rescued.source_indices == (0, 1, 2, 3, 4)
    assert rescued.rescue_source_indices == (1, 3)
    assert rescued.rescue_source_frame_ids == (1, 3)
    assert rescued.as_dict()["interpolated_poses"] is False


def test_v6_rescue_spends_its_limited_budget_on_the_largest_measured_gap() -> None:
    frames = tuple(_frame(index) for index in range(7))
    motions = [
        MotionEstimate(value, 0.0, 80, 0.9, 0.7, "dis_ultrafast")
        for value in (5.0, 5.0, 15.0, 15.0, 15.0, 15.0)
    ]
    plan = VideoRenderPlan(
        frames=(frames[0], frames[2], frames[6]), source_indices=(0, 2, 6),
        scan_direction=1, high_risk_edge_count=0, normal_target_step_pixels=8.0,
        risk_target_step_pixels=5.0,
    )

    rescued = insert_v6_real_rescue_sources(
        frames, motions, plan, full_resolution_scale=1.0, maximum_step_pixels=8.0,
        maximum_rescues=1,
    )

    assert rescued.rescue_source_indices == (4,)
    assert rescued.source_indices == (0, 2, 4, 6)


def test_motion_composition_allows_only_certified_internal_anchor_spans() -> None:
    motions = [MotionEstimate(1.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast") for _ in range(6)]
    try:
        compose_selected_motions(motions, (1, 3, 5))
    except ValueError as exc:
        assert "scan endpoints" in str(exc)
    else:
        raise AssertionError("ordinary source selection accepted missing endpoints")

    combined = compose_selected_motions(
        motions, (1, 3, 5), require_scan_endpoints=False
    )
    assert [item.dx for item in combined] == [2.0, 2.0]


def test_motion_resampling_config_rejects_relaxed_order() -> None:
    try:
        MotionResamplingConfig.from_mapping({"normal_target_step_pixels": 2.0})
    except ValueError as exc:
        assert "minimum <= risk <= normal" in str(exc)
    else:
        raise AssertionError("Invalid resampling order was accepted")


def test_direct_orb_fps_selection_uses_only_ordered_real_frames() -> None:
    frames = tuple(_frame(index) for index in range(25))

    selected_8 = _select_real_orb_tracking_indices(frames, target_fps=8.0)
    selected_12 = _select_real_orb_tracking_indices(frames, target_fps=12.0)
    selected_16 = _select_real_orb_tracking_indices(frames, target_fps=16.0)

    assert selected_8[0] == selected_12[0] == selected_16[0] == 0
    assert selected_8[-1] == selected_12[-1] == selected_16[-1] == 24
    assert len(selected_8) < len(selected_12) < len(selected_16)
    assert all(tuple(sorted(set(indices))) == indices for indices in (selected_8, selected_12, selected_16))


def test_direct_orb_tracking_gate_selects_lowest_complete_candidate(monkeypatch, tmp_path) -> None:
    import panorama_demo.video_panorama as video_panorama

    frames = tuple(_frame(index) for index in range(13))
    qualities = [_quality() for _ in frames]
    motions = [MotionEstimate(1.0, 0.0, 80, 0.9, 0.7, "dis_ultrafast") for _ in range(12)]
    session = SimpleNamespace(
        rgbd=SimpleNamespace(
            frames=frames,
            calibration=CameraIntrinsics(
                width=848, height=480, fx=600.0, fy=600.0,
                cx=423.5, cy=239.5, distortion=(),
            ),
            root=tmp_path,
        )
    )
    monkeypatch.setattr(video_panorama, "load_config", lambda _path: {"stitch": {}})
    monkeypatch.setattr(video_panorama, "load_video_session", lambda *_args, **_kwargs: session)
    monkeypatch.setattr(
        video_panorama,
        "analyse_video_scan",
        lambda *_args, **_kwargs: (
            qualities,
            motions,
            {"start_index": 0, "end_index": 12, "scan_direction": 1},
        ),
    )

    attempts = 0

    def _run_orb(candidate_frames, *_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        ids = tuple(frame.frame_id for frame in candidate_frames)
        if attempts == 1:
            raise RuntimeError("T0 lost tracking")
        return SimpleNamespace(
            tracked_frame_ids=ids,
            poses_by_frame_id={frame_id: np.eye(4) for frame_id in ids},
            attempt_audit=(),
        )

    monkeypatch.setattr(video_panorama, "run_orbslam3_rgbd", _run_orb)
    report = run_direct_orb_tracking_gate(
        tmp_path,
        tmp_path / "gate",
        fps_candidates=(8.0, 12.0, 16.0),
        fast_orbslam3_config={"feature_count": 1000},
    )

    assert report["selected_tracking_candidate_id"] == "T1"
    assert report["full_direct_orb_chain_available"] is True
    assert report["interpolated_poses"] is False
    assert report["tracking_candidates"][0]["status"] == "direct_chain_failed"
    assert report["tracking_candidates"][1]["direct_orb_pose_coverage"] == 1.0
    assert report["tracking_candidates"][1]["frontality_coverage_pass"] is True


def test_frontality_spans_are_calibrated_dynamic_ranges_not_fixed_strip_widths() -> None:
    calibration = CameraIntrinsics(
        width=848, height=480, fx=600.0, fy=600.0, cx=423.5, cy=239.5, distortion=(),
    )
    records = assess_video_source_frontality(
        tuple(_frame(index) for index in range(3)), calibration
    )

    assert len(records) == 3
    assert records[0].near_target_span[1] - records[0].near_target_span[0] < (
        records[0].general_target_span[1] - records[0].general_target_span[0]
    )
    assert records[0].general_target_span[1] - records[0].general_target_span[0] < (
        records[0].general_hard_span[1] - records[0].general_hard_span[0]
    )
    assert records[0].as_dict()["valid_frontality_span"] == list(records[0].general_target_span)
    assert off_axis_angle_degrees(calibration, calibration.cx) == 0.0
    assert off_axis_angle_degrees(calibration, 0.0) < -6.0
    with pytest.raises(ValueError, match="target <= hard"):
        VideoFrontalityConfig(near_target_degrees=5.0, near_hard_degrees=4.0)


def test_frontality_owner_plan_uses_real_centres_and_dynamic_intervals() -> None:
    calibration = CameraIntrinsics(
        width=848, height=480, fx=600.0, fy=600.0, cx=423.5, cy=239.5, distortion=(),
    )
    records = assess_video_source_frontality(tuple(_frame(index) for index in range(3)), calibration)
    plan = plan_frontality_owner_spans(records, calibration, (0.0, 40.0, 115.0))

    assert plan.source_centres_x == (0.0, 40.0, 115.0)
    assert plan.owner_intervals_x[0][1] == 20.0
    assert plan.owner_intervals_x[1][1] == 77.5
    assert plan.as_dict()["fixed_owner_pixel_width"] is None

    with pytest.raises(ValueError, match="no common hard-frontality"):
        plan_frontality_owner_spans(records, calibration, (0.0, 500.0, 1_000.0))
