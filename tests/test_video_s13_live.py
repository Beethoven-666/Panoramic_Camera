from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import cv2

from panorama_demo.capture_orbbec import (
    CaptureResult,
    LiveFramePacket,
    LiveSessionInfo,
    WrittenRGBDFrame,
    _compose_capture_preview,
    build_parser,
)
from panorama_demo.video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)
from panorama_demo.video_s13_live import S13V11LiveObserver


def _wait(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def _packet(frame_id: int, accepted_ns: int, *, writer_fraction: float = 0.0) -> LiveFramePacket:
    image = np.full((120, 212, 3), 30 + frame_id, dtype=np.uint8)
    depth = np.full((120, 212), 1000, dtype=np.uint16)
    image.setflags(write=False)
    depth.setflags(write=False)
    return LiveFramePacket(
        frame_id=frame_id,
        color_bgr=image,
        aligned_depth_mm=depth,
        metadata={},
        accepted_monotonic_ns=accepted_ns,
        writer_queue_fraction=writer_fraction,
        writer_queue_drops=0,
    )


def _observer(tmp_path: Path) -> S13V11LiveObserver:
    observer = S13V11LiveObserver(
        production_config_sha256="a" * 64,
        preview_hz=1000.0,
        motion_estimator=lambda _left, _right: (8.0, True),
    )
    observer.on_session_ready(LiveSessionInfo(
        root=tmp_path,
        capture_started_monotonic_ns=1,
        calibration={},
    ))
    return observer


def test_preview_waits_for_both_08_seconds_and_32_analysis_pixels(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    base = time.monotonic_ns()
    for frame_id in range(4):
        observer.on_frame_accepted(_packet(frame_id, base + frame_id * 200_000_000))
    _wait(lambda: observer.snapshot().analysed_frames == 4)

    before = observer.snapshot()
    assert before.cumulative_forward_424_px == 24.0
    assert before.stable_motion_seconds == 0.6
    assert before.preview_updates == 0
    assert not (tmp_path / "live_preview.jpg").exists()

    observer.on_frame_accepted(_packet(4, base + 800_000_000))
    _wait(lambda: observer.snapshot().preview_updates == 1)
    after = observer.snapshot()
    assert after.stable_motion_seconds == 0.8
    assert after.cumulative_forward_424_px == 32.0
    assert after.direction_consistency == 1.0
    assert after.reliable_motion_fraction == 1.0
    assert after.preview_queue_peak == 1
    assert (tmp_path / "live_preview.jpg").is_file()
    panorama = observer.capture_preview_image()
    assert panorama is not None
    assert panorama.dtype == np.uint8
    assert panorama.ndim == 3 and panorama.shape[2] == 3
    assert panorama.shape == (120, 424, 3)
    assert panorama.flags.writeable is False
    saved = cv2.imread(str(tmp_path / "live_preview.jpg"), cv2.IMREAD_COLOR)
    assert saved is not None
    assert saved.shape[1] > saved.shape[0]
    assert np.all(saved > 0)
    state = json.loads((tmp_path / "live_preview_state.json").read_text(encoding="utf-8"))
    assert {
        "schema", "authority", "algorithm_id", "preview_generation",
        "latest_frame_id", "stable_source_count", "mutable_source_count",
        "stage_visualization", "capture_active",
    } <= state.keys()
    assert state["schema"] == "gemini305-s013-v11-live-preview/v1"
    assert state == {
        **state,
        "authority": "non_authoritative_live_preview",
        "algorithm_id": S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        "preview_generation": 1,
        "latest_frame_id": 4,
        "stable_source_count": 5,
        "mutable_source_count": 0,
        "stage_visualization": "s013_incremental_hard_owner_preview/v1",
        "capture_active": True,
    }
    observer.on_capture_stopping()
    stopped = json.loads((tmp_path / "live_preview_state.json").read_text(encoding="utf-8"))
    assert stopped["capture_active"] is False
    assert stopped["mutable_source_count"] == 0


def test_capture_window_places_live_panorama_below_rgbd_preview() -> None:
    color = np.full((120, 212, 3), 40, dtype=np.uint8)
    depth = np.full((120, 212), 500, dtype=np.uint16)
    panorama = np.full((60, 180, 3), (10, 80, 160), dtype=np.uint8)

    display = _compose_capture_preview(color, depth, 1.0, panorama)

    assert display.shape == (240, 424, 3)
    panorama_region = display[120:]
    assert np.any(np.all(panorama_region == (10, 80, 160), axis=2))


def test_motion_preview_normalizes_both_scan_directions_to_world_left_to_right() -> None:
    physical_left = np.full((12, 24, 3), (10, 20, 30), dtype=np.uint8)
    physical_middle = np.full((12, 24, 3), (40, 50, 60), dtype=np.uint8)
    physical_right = np.full((12, 24, 3), (70, 80, 90), dtype=np.uint8)

    left_to_right = S13V11LiveObserver._render_motion_preview((
        (0.0, physical_left),
        (8.0, physical_middle),
        (16.0, physical_right),
    ))
    right_to_left = S13V11LiveObserver._render_motion_preview((
        (0.0, physical_right),
        (-8.0, physical_middle),
        (-16.0, physical_left),
    ))

    assert np.array_equal(left_to_right, right_to_left)
    assert np.array_equal(left_to_right[:, 0], physical_left[:, 0])
    assert np.array_equal(left_to_right[:, -1], physical_right[:, -1])


def test_capture_window_shows_waiting_panel_before_panorama_is_ready() -> None:
    color = np.zeros((120, 212, 3), dtype=np.uint8)
    depth = np.full((120, 212), 500, dtype=np.uint16)

    display = _compose_capture_preview(color, depth, 1.0)

    assert display.shape == (240, 424, 3)
    assert np.any(display[120:] > 0)


def test_direction_change_resets_stable_motion_start(tmp_path: Path) -> None:
    motions = deque([8.0, 8.0, -8.0, -8.0, -8.0, -8.0])
    observer = S13V11LiveObserver(
        production_config_sha256="a" * 64,
        preview_hz=1000.0,
        motion_estimator=lambda _left, _right: (motions.popleft(), True),
    )
    observer.on_session_ready(LiveSessionInfo(tmp_path, 1, {}))
    base = time.monotonic_ns()
    for frame_id in range(6):
        observer.on_frame_accepted(_packet(frame_id, base + frame_id * 200_000_000))
    _wait(lambda: observer.snapshot().analysed_frames == 6)
    assert observer.snapshot().preview_updates == 0
    assert observer.snapshot().stable_motion_seconds == 0.6

    observer.on_frame_accepted(_packet(6, base + 1_200_000_000))
    _wait(lambda: observer.snapshot().preview_updates == 1)
    assert observer.snapshot().stable_motion_seconds == 0.8
    assert observer.snapshot().cumulative_forward_424_px == 32.0
    observer.on_capture_stopping()


def test_motion_gap_over_04_seconds_starts_a_new_stable_segment(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    base = time.monotonic_ns()
    offsets_ms = [0, 200, 400, 900, 1100, 1300, 1500]
    for frame_id, offset_ms in enumerate(offsets_ms):
        observer.on_frame_accepted(_packet(frame_id, base + offset_ms * 1_000_000))
    _wait(lambda: observer.snapshot().analysed_frames == len(offsets_ms))
    before = observer.snapshot()
    assert before.preview_updates == 0
    assert before.stable_motion_seconds == 0.6
    assert before.cumulative_forward_424_px == 24.0

    observer.on_frame_accepted(_packet(7, base + 1_700_000_000))
    _wait(lambda: observer.snapshot().preview_updates == 1)
    after = observer.snapshot()
    assert after.stable_motion_seconds == 0.8
    assert after.cumulative_forward_424_px == 32.0
    observer.on_capture_stopping()


def test_preview_failure_is_atomic_and_does_not_stop_analysis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    observer = _observer(tmp_path)
    monkeypatch.setattr("panorama_demo.video_s13_live.cv2.imencode", lambda *_args: (False, None))
    base = time.monotonic_ns()
    for frame_id in range(5):
        observer.on_frame_accepted(_packet(frame_id, base + frame_id * 200_000_000))
    _wait(lambda: (tmp_path / "live_preview_failure.json").is_file())
    failure = json.loads((tmp_path / "live_preview_failure.json").read_text(encoding="utf-8"))
    assert failure["capture_continues"] is True
    assert not (tmp_path / ".live_preview_failure.pending.json").exists()

    for frame_id in range(5, 8):
        observer.on_frame_accepted(_packet(frame_id, base + frame_id * 200_000_000))
    _wait(lambda: observer.snapshot().analysed_frames == 8)
    snapshot = observer.snapshot()
    assert snapshot.preview_updates == 0
    assert snapshot.preview_failures == 1
    observer.on_capture_stopping()


def test_writer_pressure_delays_preview_and_stop_cancels_future_updates(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    base = time.monotonic_ns()
    for frame_id in range(6):
        observer.on_frame_accepted(_packet(
            frame_id,
            base + frame_id * 200_000_000,
            writer_fraction=0.5 if frame_id >= 4 else 0.0,
        ))
    _wait(lambda: observer.snapshot().analysed_frames == 6)
    assert observer.snapshot().preview_updates == 0

    observer.on_frame_accepted(_packet(6, base + 1_200_000_000))
    _wait(lambda: observer.snapshot().preview_updates == 1)
    observer.on_capture_stopping()
    snapshot = observer.snapshot()
    observer.on_frame_accepted(_packet(7, base + 1_400_000_000))
    time.sleep(0.05)
    assert observer.snapshot().preview_updates == snapshot.preview_updates
    assert observer.snapshot().stopped is True


def test_closed_capture_freezes_complete_validated_inputs_only_handoff(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    base = time.monotonic_ns()
    for frame_id in range(2):
        observer.on_frame_accepted(_packet(frame_id, base + frame_id * 200_000_000))
        observer.on_frame_committed(WrittenRGBDFrame(
            frame_id=frame_id,
            color_path=tmp_path / f"{frame_id}.jpg",
            aligned_depth_path=tmp_path / f"{frame_id}.png",
            color_sha256=str(frame_id) * 64,
            aligned_depth_sha256=str(frame_id + 1) * 64,
            timestamp_us=frame_id,
            depth_scale_mm_per_unit=1.0,
            frames_csv_row_index=frame_id,
            commit_monotonic_ns=base + frame_id,
        ))
    observer.on_capture_stopping()
    observer.on_capture_closed(CaptureResult(
        session_root=tmp_path,
        received_frames=2,
        written_frames=2,
        queue_drops=0,
        write_errors=0,
        max_queue_depth=1,
        capture_started_monotonic_ns=base,
        capture_stopped_monotonic_ns=base + 500_000_000,
    ))

    handoff = observer.freeze_handoff()
    assert handoff.algorithm_id == S13_VISUAL_CONTINUITY_ALGORITHM_ID
    assert handoff.implementation_id == S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID
    assert handoff.production_config_sha256 == "a" * 64
    assert handoff.reuse_level == "validated_inputs_only"
    assert handoff.online_shadow is None
    assert handoff.frozen_p0_authority is None
    assert not hasattr(observer, "_shadow_thread")
    assert handoff.online_2d_metrics["shadow_processed_committed_frames"] == 0
    assert handoff.online_2d_metrics["full_m0_m3_recomputed"] is True
    assert [item.frame_id for item in handoff.committed_frames] == [0, 1]
    assert handoff.capture_stopped_monotonic_ns > handoff.capture_started_monotonic_ns


def test_capture_default_has_no_diagnostic_online_orb() -> None:
    args = build_parser().parse_args([])
    assert args.diagnostic_online_orbslam3 is False
