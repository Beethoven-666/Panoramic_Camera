from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import panorama_demo.video_s1_renderer as renderer_module

from panorama_demo.quality import FrameQuality, MotionEstimate
from panorama_demo.session import CameraIntrinsics, RGBDFrame
from panorama_demo.video_s1_config import VideoS11ExperimentConfig, load_s1_config
from panorama_demo.video_s1_layout import S1SourceLayout
from panorama_demo.video_s1_handoff import (
    BoundaryCandidate,
    ForbiddenInterval,
    HandoffEvidenceSplit,
    HandoffObservation,
)
from panorama_demo.video_s1_layout import S1PairLineage
from panorama_demo.video_s1_renderer import (
    S11StageProvenance,
    build_calibrated_analysis_depth,
    compose_strict_owner_panorama,
    evaluate_rendered_heldout_single_copy,
    render_video_s11,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_s1_config(
    ROOT / "configs/video_candidates/S011_object_safe_straight_handoff_v1.yaml"
)
assert isinstance(CONFIG, VideoS11ExperimentConfig)


def _quality() -> FrameQuality:
    return FrameQuality(100.0, 30.0, 0.8, 0.0, 0.0, 40.0, 20.0)


def _frame(tmp_path: Path, frame_id: int, image: np.ndarray) -> RGBDFrame:
    color = tmp_path / f"color_{frame_id}.png"
    depth = tmp_path / f"depth_{frame_id}.png"
    assert cv2.imwrite(str(color), image)
    assert cv2.imwrite(str(depth), np.full(image.shape[:2], 1000, dtype=np.uint16))
    return RGBDFrame(frame_id, color, depth, 1.0, frame_id * 1000, 5, 1)


def _baseline() -> dict[str, object]:
    layouts = (
        S1SourceLayout(10, 0, 212.0, 0, 222, 0, 424, 0, 286, 0.0),
        S1SourceLayout(12, 2, 232.0, 222, 444, 20, 444, 158, 444, 20.0),
    )
    return {
        "expected": {
            "canvas_width": 444,
            "session": {"calibration_sha256": "synthetic-calibration"},
        },
        "layout": {
            "sources": [
                {
                    "frame_id": item.frame_id,
                    "source_index": item.source_index,
                    "canvas_center_x": item.canvas_center_x,
                    "owner_left_x": item.owner_left_x,
                    "owner_right_x": item.owner_right_x,
                    "support_left_x": item.support_left_x,
                    "support_right_x": item.support_right_x,
                    "match_left_x": item.match_left_x,
                    "match_right_x": item.match_right_x,
                    "measured_progress_from_previous_px": (
                        item.measured_progress_from_previous_px
                    ),
                    "source_quality": dict(item.source_quality),
                }
                for item in layouts
            ]
        },
    }


def test_s11_reuses_frozen_sources_and_remaps_each_final_source_once(
    tmp_path: Path,
) -> None:
    height, width = 64, 424
    rng = np.random.default_rng(24)
    first = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    middle = np.roll(first, -10, axis=1)
    last = np.roll(first, -20, axis=1)
    frames = (
        _frame(tmp_path, 10, first),
        _frame(tmp_path, 11, middle),
        _frame(tmp_path, 12, last),
    )
    calibration = CameraIntrinsics(
        width, height, 330.0, 330.0, width / 2, height / 2, (0.0,) * 8
    )
    result = render_video_s11(
        frames,
        calibration,
        (
            MotionEstimate(10.0, 0.0, 100, 0.9, 0.8, "features"),
            MotionEstimate(10.0, 0.0, 100, 0.9, 0.8, "features"),
        ),
        (_quality(), _quality(), _quality()),
        scan_direction=1,
        analysis_width=424,
        config=CONFIG,
        baseline=_baseline(),
    )

    assert result.selected_frame_indices == (0, 2)
    assert result.rescue_events == ()
    assert result.remap_invocations == len(result.final_layouts) == 2
    assert result.stage_a_nominal_midpoint_owner_only.shape == (height, 444, 3)
    assert result.stage_b_handoff_only.shape == result.stage_c_final.shape
    assert result.pair_results[0].pair_status == "midpoint_safe"
    for provenance in (
        result.stage_a_provenance,
        result.stage_b_provenance,
        result.stage_c_provenance,
    ):
        assert np.all(provenance.geometric_owner_frame_id >= 0)
        assert np.array_equal(
            provenance.final_valid, provenance.declared_owner_sample_valid
        )
        assert np.all(provenance.final_owner_frame_id[~provenance.final_valid] == -1)


def test_strict_owner_keeps_declared_invalid_pixels_unfilled() -> None:
    layouts = (
        S1SourceLayout(7, 0, 2.0, 0, 2, 0, 4, 0, 4, 0.0),
        S1SourceLayout(9, 1, 4.0, 2, 6, 2, 6, 2, 6, 2.0),
    )
    first = np.full((3, 4, 3), 20, dtype=np.uint8)
    second = np.full((3, 4, 3), 80, dtype=np.uint8)
    first_valid = np.ones((3, 4), dtype=bool)
    first_valid[1, 1] = False

    panorama, provenance = compose_strict_owner_panorama(
        ((0, first, first_valid), (2, second, np.ones((3, 4), dtype=bool))),
        layouts,
        canvas_width=6,
    )

    assert provenance.geometric_owner_frame_id[1, 1] == 7
    assert not provenance.final_valid[1, 1]
    assert provenance.final_owner_frame_id[1, 1] == -1
    assert np.array_equal(panorama[1, 1], np.zeros(3, dtype=np.uint8))


def test_calibrated_depth_keeps_missing_depth_invalid() -> None:
    calibration = CameraIntrinsics(64, 32, 50.0, 50.0, 31.5, 15.5, (0.0,) * 8)
    depth = np.full((32, 64), 1000.0, dtype=np.float32)
    depth[:, 28:36] = 0.0

    result = build_calibrated_analysis_depth(
        depth, calibration, analysis_width=32
    )

    assert result.shape == (32, 32)
    assert np.any(result == 0.0)
    assert np.all(result >= 0.0)


def test_rendered_heldout_audits_actual_owner_and_validity() -> None:
    owner = np.full((4, 8), -1, dtype=np.int64)
    owner[:, :4] = 10
    owner[:, 4:] = 12
    valid = np.ones((4, 8), dtype=bool)
    valid[2, 6] = False
    owner[2, 6] = -1
    provenance = S11StageProvenance(owner.copy(), valid.copy(), owner, valid)
    observation = HandoffObservation(
        match_id="heldout",
        origin_pair_lineage="10->12",
        left_frame_id=10,
        right_frame_id=12,
        role="heldout",
        left_analysis_x=2.0,
        left_y=2.0,
        right_analysis_x=6.0,
        right_y=2.0,
        left_calibrated_x=2.0,
        right_calibrated_x=6.0,
        left_canvas_x=2.0,
        right_canvas_x=6.0,
        forward_backward_error_px=0.1,
        vertical_error_px=0.0,
        texture_score=1.0,
        observation_canvas_dx=4.0,
    )

    metrics = evaluate_rendered_heldout_single_copy(
        (observation,), left_frame_id=10, right_frame_id=12,
        boundary_x=4,
        provenance=provenance,
    )

    assert metrics["evaluable_count"] == 1
    assert metrics["single_copy_count"] == 1
    assert metrics["duplicate_count"] == 0
    assert metrics["missing_count"] == 0


def test_s11_rescues_only_a_hard_risk_pair_with_no_safe_straight_line(
    tmp_path: Path, monkeypatch
) -> None:
    height, width = 64, 424
    rng = np.random.default_rng(91)
    image = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    frames = tuple(_frame(tmp_path, frame_id, image) for frame_id in (10, 11, 12))
    calibration = CameraIntrinsics(
        width, height, 330.0, 330.0, width / 2, height / 2, (0.0,) * 8
    )
    solver = tuple(
        HandoffObservation(
            match_id=f"{index:016x}",
            origin_pair_lineage="10->12",
            left_frame_id=10,
            right_frame_id=12,
            role="solver",
            left_analysis_x=200.0,
            left_y=float(index * 3),
            right_analysis_x=206.0,
            right_y=float(index * 3),
            left_calibrated_x=200.0,
            right_calibrated_x=206.0,
            left_canvas_x=216.0,
            right_canvas_x=228.0,
            forward_backward_error_px=0.1,
            vertical_error_px=0.0,
            texture_score=1.0,
            observation_canvas_dx=12.0,
            relative_dx=6.0,
        )
        for index in range(12)
    )
    heldout = (
        HandoffObservation(
            **{
                **solver[0].__dict__,
                "match_id": "origin-heldout",
                "role": "heldout",
            }
        ),
    )

    def fake_evidence(layouts, initial_layouts, *_args, **_kwargs):
        if len(layouts) == 2:
            midpoint = layouts[0].owner_right_x
            interval = ForbiddenInterval(
                midpoint - 64,
                midpoint + 64,
                0,
                tuple(item.match_id for item in solver),
                len(solver),
                33.0,
                6.0,
                1,
            )
            candidates = tuple(
                BoundaryCandidate(x, float(abs(x - midpoint)), True)
                for x in range(midpoint - 48, midpoint + 49)
            )
            return (
                renderer_module._S11PairEvidence(
                    0,
                    S1PairLineage(10, 12),
                    HandoffEvidenceSplit(solver, heldout),
                    (interval,),
                    candidates,
                    True,
                ),
            )
        pairs = []
        for pair_index, (left, right) in enumerate(zip(layouts, layouts[1:])):
            midpoint = left.owner_right_x
            pairs.append(renderer_module._S11PairEvidence(
                pair_index,
                S1PairLineage(10, 12),
                HandoffEvidenceSplit((), ()),
                (),
                (BoundaryCandidate(midpoint, 0.0, False),),
                False,
            ))
        return tuple(pairs)

    monkeypatch.setattr(renderer_module, "_s11_pair_evidence", fake_evidence)
    result = render_video_s11(
        frames,
        calibration,
        (
            MotionEstimate(10.0, 0.0, 100, 0.9, 0.8, "features"),
            MotionEstimate(10.0, 0.0, 100, 0.9, 0.8, "features"),
        ),
        (_quality(), _quality(), _quality()),
        scan_direction=1,
        analysis_width=424,
        config=CONFIG,
        baseline=_baseline(),
    )

    assert result.selected_frame_indices == (0, 1, 2)
    assert len(result.rescue_events) == 1
    assert result.rescue_events[0]["frame_id"] == 11
    assert result.rescue_events[0]["origin_heldout_denominator"] == 1
    assert "origin_stage_c_heldout_metrics" in result.rescue_events[0]
    assert result.remap_invocations == 3
    assert result.initial_layouts[0].frame_id == result.final_layouts[0].frame_id == 10
    assert result.initial_layouts[-1].frame_id == result.final_layouts[-1].frame_id == 12
