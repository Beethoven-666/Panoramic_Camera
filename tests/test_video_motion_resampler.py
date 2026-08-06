from __future__ import annotations

from pathlib import Path

from panorama_demo.quality import FrameQuality, MotionEstimate
from panorama_demo.session import RGBDFrame
from panorama_demo.video_motion_resampler import (
    MotionResamplingConfig,
    compose_selected_motions,
    select_render_keyframes,
)


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
