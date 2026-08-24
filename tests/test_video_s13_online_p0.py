from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s13_base_renderer import render_s13_p0
from panorama_demo.video_s13_fast_pipeline import _measure_fast_motion
from panorama_demo.video_s13_online_p0 import (
    S13OnlineP0Engine,
    S13ScheduleSemanticAssignment,
    semantic_assignments,
)
from panorama_demo.video_s13_online_checkpoint import S13OnlineCheckpointChain
from panorama_demo.video_s13_progress import (
    build_s13_m3_layout,
    selected_hypothesis_ids_for_spatial_sources,
)
from panorama_demo.video_s13_schedule import plan_s13_m3_schedule
from panorama_demo.video_s13_session import S13RenderFrame, read_s13_rgb
from panorama_demo.video_s13_trajectory import S13Trajectory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ignore_pose() -> S13Trajectory:
    return S13Trajectory("ignore", None, {}, {}, (), {})


def test_online_committed_shadow_final_schedule_matches_batch(tmp_path: Path) -> None:
    calibration = CameraIntrinsics(
        width=64, height=32, fx=50.0, fy=50.0, cx=31.5, cy=15.5, distortion=(),
    )
    color = tmp_path / "color"
    color.mkdir()
    scene = np.random.default_rng(132).integers(
        0, 256, size=(32, 112, 3), dtype=np.uint8,
    )
    frames = []
    committed = []
    engine = S13OnlineP0Engine(
        session_root=tmp_path,
        calibration=calibration,
        analysis_width_px=424,
    )
    for frame_id, position in enumerate(range(0, 49, 8)):
        path = color / f"{frame_id:06d}.jpg"
        assert cv2.imwrite(str(path), scene[:, position:position + 64])
        frame = S13RenderFrame(frame_id, path, frame_id * 1_000, 64, 32)
        frames.append(frame)
        identity = SimpleNamespace(
            frame_id=frame_id,
            color_path=path,
            color_sha256=_sha256(path),
            timestamp_us=frame.timestamp_us,
        )
        committed.append(identity)
        engine.append_committed(identity)

    motion, _profile, _execution = _measure_fast_motion(
        tuple(frames),
        analysis_width_px=424,
        prepared_analysis=None,
        prepared_gradients=None,
        policy="deferred_step4",
    )
    layout = build_s13_m3_layout(tuple(frames), motion, _ignore_pose())
    plan = plan_s13_m3_schedule(
        layout.progress,
        motion,
        calibration,
        normal_target_advance_px=8.0,
        risky_target_advance_px=5.0,
        segment_break_pairs=layout.segment_break_pairs,
        canonical_scan_direction=layout.canonical_scan_direction,
    )
    selection, schedule = plan.selections[0], plan.schedules[0]
    hypothesis_ids = selected_hypothesis_ids_for_spatial_sources(
        layout, selection.frame_ids
    )
    online = engine.snapshot()

    assert online.frontiers.committed_frame_index == len(frames) - 1
    assert online.semantic_assignments == semantic_assignments(
        schedule, selection, hypothesis_ids
    )
    batch_p0 = render_s13_p0(
        schedule,
        calibration,
        lambda frame_id: read_s13_rgb(frames[frame_id]),
        placement_methods=selection.placement_methods,
        selected_hypothesis_ids=hypothesis_ids,
    )
    assert online.current_p0 is not None
    assert np.array_equal(online.current_p0.image, batch_p0.image)
    assert np.array_equal(online.current_p0.valid_mask, batch_p0.valid_mask)
    for name in (
        "owner_frame_id", "owner_source_index", "assignment_index",
        "source_u", "source_v", "selected_motion_hypothesis_id",
    ):
        assert np.array_equal(
            online.current_p0.pixel_provenance[name],
            batch_p0.pixel_provenance[name],
            equal_nan=True,
        )
    for name in ("manifest.json", "calibration.json", "frames.csv"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    frozen = engine.freeze(
        committed_frames=tuple(committed),
        production_config_sha256="a" * 64,
    )

    assert frozen.reuse_level == "frozen_p0_authority_v1"
    assert frozen.full_m0_m3_recomputed is False
    assert frozen.finalized_tail_source_count <= 4
    assert frozen.rollback_checkpoint_used is False
    assert np.array_equal(frozen.continuation.p0.image, batch_p0.image)


def test_checkpoint_chain_rolls_back_to_last_consistent_assignment() -> None:
    semantic = tuple(
        S13ScheduleSemanticAssignment(index, index, index, index + 1, "m", index)
        for index in range(3)
    )
    renders = tuple(
        SimpleNamespace(
            assignment_index=index,
            image_roi=np.full((2, 1, 3), index, dtype=np.uint8),
            valid_roi=np.ones((2, 1), dtype=bool),
        )
        for index in range(3)
    )
    chain = S13OnlineCheckpointChain()
    chain.advance(semantic, renders, 3)
    changed = (
        semantic[0],
        S13ScheduleSemanticAssignment(10, 10, 10, 11, "changed", 10),
        semantic[2],
    )

    assert chain.longest_consistent_prefix(changed, renders) == 1
