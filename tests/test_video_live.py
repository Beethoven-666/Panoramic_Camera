from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import panorama_demo.video_live as video_live
from panorama_demo.capture_orbbec import (
    CaptureResult,
    LiveFramePacket,
    LiveSessionInfo,
    WrittenRGBDFrame,
)
from panorama_demo.video_algorithm import VideoAlgorithmSpec
from panorama_demo.video_s13_contract import (
    S13_VISUAL_CONTINUITY_ALGORITHM_ID,
    S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
)


def _spec(tmp_path: Path) -> VideoAlgorithmSpec:
    return VideoAlgorithmSpec(
        role="production",
        algorithm_id=S13_VISUAL_CONTINUITY_ALGORITHM_ID,
        implementation_id=S13_VISUAL_CONTINUITY_IMPLEMENTATION_ID,
        config_path=tmp_path / "production.yaml",
        config_sha256="a" * 64,
        source_commit="b" * 40,
        model_sha256={},
        allow_baseline_fallback=False,
    )


def test_live_command_closes_preview_freezes_handoff_and_only_then_runs_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = tmp_path / "capture" / "run_test"
    session.mkdir(parents=True)
    spec = _spec(tmp_path)
    events: list[str] = []
    captured: dict[str, object] = {}

    def fake_capture(_args: object, *, observer: object) -> Path:
        events.append("capture")
        observer.on_session_ready(LiveSessionInfo(session, 1, {}))
        color = np.zeros((8, 16, 3), dtype=np.uint8)
        depth = np.ones((8, 16), dtype=np.uint16)
        observer.on_frame_accepted(LiveFramePacket(0, color, depth, {}, 2, 0.0, 0))
        observer.on_frame_committed(WrittenRGBDFrame(
            frame_id=0,
            color_path=session / "color.jpg",
            aligned_depth_path=session / "depth.png",
            color_sha256="c" * 64,
            aligned_depth_sha256="d" * 64,
            timestamp_us=0,
            depth_scale_mm_per_unit=1.0,
            frames_csv_row_index=0,
            commit_monotonic_ns=3,
        ))
        observer.on_capture_stopping()
        observer.on_capture_closed(CaptureResult(session, 1, 1, 0, 0, 1, 1, 4))
        manifest = {
            "clean_shutdown": True,
            "product_eligibility": {"video_panorama": True},
            "received_frames": 1,
            "written_frames": 1,
            "queue_drops": 0,
            "write_errors": 0,
            "max_queue_depth": 1,
            "capture_runtime_audit": {name: 0 for name in video_live._CAPTURE_ZERO_COUNTERS},
        }
        (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return session

    def fake_production(**kwargs: object) -> dict[str, object]:
        events.append("production")
        captured.update(kwargs)
        assert kwargs["live_handoff"].reuse_level == "validated_inputs_only"
        assert kwargs["live_handoff"].committed_frames[0].frame_id == 0
        return {"delivery_state": "published"}

    monkeypatch.setattr(video_live, "_lock_paths", lambda _path: ({}, tmp_path / "base", tmp_path / "prod"))
    monkeypatch.setattr(video_live, "resolve_video_algorithm", lambda *_args, **_kwargs: spec)
    monkeypatch.setattr(video_live, "run_video_capture", fake_capture)
    monkeypatch.setattr(video_live, "run_video_algorithm", fake_production)
    args = video_live.build_parser().parse_args([
        "--output", str(tmp_path / "capture"),
        "--panorama-output", str(tmp_path / "panorama"),
        "--no-preview",
    ])

    result = video_live.run(args)

    assert events == ["capture", "production"]
    assert captured["defer_3d"] is True
    assert result["capture"]["capture_runtime_audit"] == {
        name: 0 for name in video_live._CAPTURE_ZERO_COUNTERS
    }
    report = json.loads((session / "live_capture_report.json").read_text(encoding="utf-8"))
    assert report["handoff_reuse_level"] == "validated_inputs_only"
    assert report["preview_authority"] == "non_authoritative_live_preview"


def test_live_command_rejects_diagnostic_orb_before_camera_use(tmp_path: Path) -> None:
    args = video_live.build_parser().parse_args([
        "--output", str(tmp_path),
        "--diagnostic-online-orbslam3",
    ])
    with pytest.raises(ValueError, match="forbids capture-time ORB"):
        video_live.run(args)


def test_live_command_rejects_photo_mode() -> None:
    args = video_live.build_parser().parse_args(["--photo-mode"])
    with pytest.raises(ValueError, match="continuous RGB-D"):
        video_live.run(args)


@pytest.mark.parametrize("name", video_live._CAPTURE_ZERO_COUNTERS)
def test_live_capture_heavy_processing_counters_are_fail_closed(name: str) -> None:
    audit = {key: 0 for key in video_live._CAPTURE_ZERO_COUNTERS}
    audit[name] = 1
    with pytest.raises(RuntimeError, match="forbidden heavy processing"):
        video_live._require_capture_zero(audit)
