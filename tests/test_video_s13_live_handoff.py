from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import psutil
import pytest

from panorama_demo.capture_orbbec import CaptureResult, LiveFramePacket, LiveSessionInfo, WrittenRGBDFrame
from panorama_demo.video_s13_committed_ledger import CommittedLedger
from panorama_demo.video_s13_live import CommittedFrameIdentity, S13V11LiveObserver
from panorama_demo.video_s13_motion import S13MotionAccumulator, S13PreparedMotionFrame


def test_live_authority_uses_committed_jpeg_not_preview_bgr(tmp_path):
    calibration = {"color_intrinsic": {"width": 64, "height": 32, "fx": 50, "fy": 50, "cx": 31.5, "cy": 15.5},
                   "color_distortion": {}}
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "frames.csv").write_text("frames")
    scene = np.random.default_rng(132).integers(0, 256, (32, 112, 3), np.uint8)
    observer = S13V11LiveObserver(production_config_sha256="a" * 64)
    started = time.monotonic_ns()
    try:
        observer.on_session_ready(LiveSessionInfo(tmp_path, started, calibration))
        for frame_id, position in enumerate(range(0, 49, 8)):
            color, depth = tmp_path / f"{frame_id}.jpg", tmp_path / f"{frame_id}.png"
            cv2.imwrite(str(color), scene[:, position:position + 64])
            cv2.imwrite(str(depth), np.ones((32, 64), np.uint16))
            observer.on_frame_accepted(LiveFramePacket(frame_id, np.zeros((32, 64, 3), np.uint8),
                                                      np.ones((32, 64), np.uint16), {},
                                                      started + frame_id * 200_000_000, 0, 0))
            observer.on_frame_committed(WrittenRGBDFrame(frame_id, color, depth,
                hashlib.sha256(color.read_bytes()).hexdigest(), hashlib.sha256(depth.read_bytes()).hexdigest(),
                frame_id * 1000, 1, frame_id, started + frame_id * 200_000_000))
        observer.on_capture_stopping()
        observer.on_capture_closed(CaptureResult(tmp_path, 7, 7, 0, 0, 1, started, time.monotonic_ns()))
        handoff = observer.freeze_handoff()
        assert handoff.reuse_level == "frozen_p0_authority_v1"
        assert np.count_nonzero(handoff.frozen_p0_authority.continuation.p0.image) > 0
        assert handoff.frozen_p0_authority.p0_prefix_reused_fraction > 0
        assert len(observer._authority_engine._raw_by_frame_id) <= 8
        assert len(observer._authority_engine._motion._prepared) <= 4
        assert (tmp_path / "checkpoints/online-p0/current.json").is_file()
        from panorama_demo.video_algorithm import VideoAlgorithmSpec
        from panorama_demo.video_s13_production import run_s13_v11_production

        corrupt = replace(handoff, frozen_p0_authority=replace(
            handoff.frozen_p0_authority, p0_owner_sha256="0" * 64))
        spec = VideoAlgorithmSpec(role="production", algorithm_id=handoff.algorithm_id,
                                 implementation_id=handoff.implementation_id,
                                 config_path=tmp_path / "config.yaml", config_sha256="a" * 64,
                                 source_commit="b" * 40, model_sha256={}, allow_baseline_fallback=False)
        seen = []

        def capture_recompute(**kwargs):
            seen.append(kwargs["live_handoff"])
            raise RuntimeError("recompute boundary reached")

        with pytest.raises(RuntimeError, match="recompute boundary reached"):
            run_s13_v11_production(session_path=tmp_path, output=tmp_path / "recompute",
                                  algorithm_spec=spec, live_handoff=corrupt,
                                  authority_runner=capture_recompute)
        assert seen[0].reuse_level == "validated_inputs_only"
        assert seen[0].online_2d_metrics["full_m0_m3_recomputed"] is True
        assert seen[0].algorithm_id == handoff.algorithm_id
        assert (tmp_path / "recompute/online_checkpoint_rollback.json").is_file()
    finally:
        observer.on_capture_stopping()
        observer.close_authority()


def test_thirty_thousand_ledger_entries_keep_image_cache_and_handles_bounded(tmp_path, monkeypatch):
    from panorama_demo import video_s13_motion
    monkeypatch.setattr(video_s13_motion, "measure_s13_motion_edge", lambda *_a, **_kw: None)
    ledger = CommittedLedger(tmp_path / "ledger.jsonl", CommittedFrameIdentity)
    motion = S13MotionAccumulator()
    prepared = S13PreparedMotionFrame(0, "", "", np.zeros((32, 64), np.uint8), 1,
                                      np.zeros((32, 64), np.float32), np.empty((0, 1, 2), np.float32), "")
    process = psutil.Process()
    handles_before = process.num_handles() if hasattr(process, "num_handles") else process.num_fds()
    threads_before = threading.active_count()
    for index in range(30_000):
        identity = CommittedFrameIdentity(index, Path("color.jpg"), Path("depth.png"), "a" * 64,
                                          "b" * 64, index, index, index)
        ledger.append(identity)
        motion.append(replace(prepared, frame_id=index, gray_424=prepared.gray_424.copy()))
    assert len(ledger) == 30_000
    assert ledger[29_999].frame_id == 29_999
    assert len(ledger._offsets) * ledger._offsets.itemsize == 240_000
    assert len(motion._prepared) == 4
    handles_after = process.num_handles() if hasattr(process, "num_handles") else process.num_fds()
    assert handles_after <= handles_before + 2
    assert threading.active_count() == threads_before
