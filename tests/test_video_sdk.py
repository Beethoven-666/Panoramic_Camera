from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from panorama_demo import (
    Gemini305VideoSDK,
    PanoramaProcessingError,
    VideoJobState,
    VideoPanoramaResult,
    VideoSDKConfig,
)
from panorama_demo import video_sdk_3d, video_live, video_pipeline
from panorama_demo.sdk_state import CaptureError, ThreeDProcessingError
from panorama_demo.capture_orbbec import (
    _VideoCameraControlUnavailableError,
    _VideoCapturePreflightError,
    _VideoWarmupNoFramesError,
)


@pytest.fixture(autouse=True)
def isolated_sdk_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests provision the shared lock directory as the Linux installer does."""
    original = VideoSDKConfig.__post_init__
    lock_root = tmp_path / "locks"
    lock_root.mkdir()

    def initialize(self: VideoSDKConfig) -> None:
        object.__setattr__(self, "camera_lock_root", lock_root)
        object.__setattr__(self, "state_root", tmp_path / "state")
        original(self)

    monkeypatch.setattr(VideoSDKConfig, "__post_init__", initialize)


def _published_2d(session: Path, output: Path) -> None:
    session.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("video_panorama.png", "video_panorama.jpg", "video_pixel_provenance.npz"):
        (output / name).write_bytes(b"artifact")
    (output / "video_report.json").write_text(
        json.dumps({"trajectory": {"source": "ignore_pose"}, "grades": {"overall": "A"}}),
        encoding="utf-8",
    )
    (output / "video_timing.json").write_text(
        json.dumps({"final_2d": {"capture_stop_to_p3_published_seconds": 1.25}}),
        encoding="utf-8",
    )
    (output / "video_delivery.json").write_text(
        json.dumps({"delivery_state": "published", "manual_review_required": False,
                    "primary_post_capture_seconds": 1.25}), encoding="utf-8",
    )


def test_process_session_reuses_locked_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, output = tmp_path / "session", tmp_path / "output"
    seen: dict[str, object] = {}

    def fake_run(**kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        _published_2d(session, output)
        return {}

    monkeypatch.setattr(video_pipeline, "run_video_algorithm", fake_run)
    result = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).process_session(session, output)
    assert result.authority == "ignore_pose_s013_v11_production"
    assert result.primary_post_capture_seconds == 1.25
    assert seen["role"] == "production"
    assert seen["defer_3d"] is True
    assert Path(seen["config_path"]).is_file() is False
    assert "candidate_config" not in seen


def test_capture_job_reuses_video_live_and_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    capture, session, output = tmp_path / "captures", tmp_path / "captures/run_1", tmp_path / "output"

    def fake_live(args: object) -> dict[str, object]:
        assert getattr(args, "wait_for_camera") is True
        assert isinstance(getattr(args, "cancel_event"), threading.Event)
        assert getattr(args, "post_3d") is False
        _published_2d(session, output)
        return {"session": str(session), "delivery": {}, "capture": {}}

    monkeypatch.setattr(video_live, "run", fake_live)
    job = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).capture_and_process(
        capture, output, duration_seconds=0.1
    )
    result = job.result(5)
    assert job.state is VideoJobState.SUCCEEDED
    assert result.session == session.resolve()


def test_capture_job_retries_zero_frame_warmup_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "captures"
    session = capture_root / "run_2"
    output = tmp_path / "output"
    calls = 0

    def fake_live(_args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _VideoWarmupNoFramesError("receiving 0/30 complete RGB-D")
        _published_2d(session, output)
        return {"session": str(session), "delivery": {}, "capture": {}}

    monkeypatch.setattr(video_live, "run", fake_live)
    job = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).capture_and_process(
        capture_root, output, duration_seconds=0.1
    )

    result = job.result(5)
    assert calls == 3
    assert result.session == session.resolve()


def test_capture_job_retries_camera_control_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "captures"
    session = capture_root / "run_3"
    output = tmp_path / "output"
    calls = 0

    def fake_live(_args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _VideoCameraControlUnavailableError("setXu failed")
        _published_2d(session, output)
        return {"session": str(session), "delivery": {}, "capture": {}}

    monkeypatch.setattr(video_live, "run", fake_live)
    job = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).capture_and_process(
        capture_root, output, duration_seconds=0.1
    )

    assert job.result(5).session == session.resolve()
    assert calls == 3


def test_capture_job_does_not_retry_preflight_when_wait_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_live(_args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise _VideoCameraControlUnavailableError("setXu failed")

    monkeypatch.setattr(video_live, "run", fake_live)
    job = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).capture_and_process(
        tmp_path / "captures",
        tmp_path / "output",
        duration_seconds=0.1,
        wait_for_camera=False,
    )

    with pytest.raises(CaptureError, match="setXu failed"):
        job.result(5)
    assert calls == 1


def test_capture_job_cancels_while_waiting_to_retry_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted = threading.Event()

    def fake_live(_args: object) -> dict[str, object]:
        attempted.set()
        raise _VideoCameraControlUnavailableError("setXu failed")

    monkeypatch.setattr(video_live, "run", fake_live)
    job = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).capture_and_process(
        tmp_path / "captures", tmp_path / "output", duration_seconds=0.1
    )

    assert attempted.wait(2)
    assert job.cancel() is True
    with pytest.raises(PanoramaProcessingError, match="setXu failed"):
        job.result(2)
    assert job.state is VideoJobState.CANCELLED


def test_capture_job_does_not_retry_other_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fake_live(_args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("partial capture failed")

    monkeypatch.setattr(video_live, "run", fake_live)
    job = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).capture_and_process(
        tmp_path / "captures", tmp_path / "output", duration_seconds=0.1
    )

    with pytest.raises(PanoramaProcessingError, match="partial capture failed"):
        job.result(5)
    assert calls == 1


def test_capture_job_retries_post_lock_preflight_until_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root = tmp_path / "captures"
    session = capture_root / "run_2"
    output = tmp_path / "output"
    calls = 0

    def fake_live(_args: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _VideoCapturePreflightError("post-lock frames unavailable")
        _published_2d(session, output)
        return {"session": str(session), "delivery": {}, "capture": {}}

    monkeypatch.setattr(video_live, "run", fake_live)
    job = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off")).capture_and_process(
        capture_root, output, duration_seconds=0.1
    )

    assert job.result(5).session == session.resolve()
    assert calls == 3


def test_job_cancellation_is_cooperative() -> None:
    started = threading.Event()

    def target(cancel: threading.Event) -> VideoPanoramaResult:
        started.set()
        cancel.wait(5)
        raise RuntimeError("cancelled")

    from panorama_demo.video_sdk import VideoProcessingJob

    job = VideoProcessingJob(target)
    assert started.wait(1)
    assert job.cancel() is True
    with pytest.raises(PanoramaProcessingError, match="cancelled"):
        job.result(2)
    assert job.state is VideoJobState.CANCELLED


def test_post_3d_failure_preserves_typed_2d_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session, two_d = tmp_path / "session", tmp_path / "two_d"
    _published_2d(session, two_d)
    original = (two_d / "video_delivery.json").read_bytes()

    def fail(**_kwargs: object) -> dict[str, object]:
        raise ThreeDProcessingError("orb failure; 2-D was preserved")

    monkeypatch.setattr(video_sdk_3d, "supervise_post_3d", fail)
    sdk = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off"))
    with pytest.raises(ThreeDProcessingError, match="2-D was preserved"):
        sdk.run_post_3d(session, two_d)
    assert (two_d / "video_delivery.json").read_bytes() == original


def test_video_result_rejects_preview_or_non_ignore_pose(tmp_path: Path) -> None:
    session, output = tmp_path / "session", tmp_path / "output"
    _published_2d(session, output)
    report = json.loads((output / "video_report.json").read_text(encoding="utf-8"))
    report["trajectory"]["source"] = "online_preview"
    (output / "video_report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(PanoramaProcessingError, match="ignore-pose"):
        VideoPanoramaResult.load(session, output)


@pytest.mark.parametrize("stop", ["transport", "low_disk"])
def test_committed_frame_preflight_error_never_reconnects_and_returns_emergency(tmp_path, monkeypatch, stop):
    import csv
    from types import SimpleNamespace
    import cv2
    import numpy as np
    from panorama_demo.commit_journal import CommitJournal, file_sha256

    calls = 0

    def failed_capture(args):
        nonlocal calls
        calls += 1
        session = Path(args.output) / "run_1"
        session.mkdir(parents=True)
        (session / "color").mkdir()
        (session / "depth_aligned").mkdir()
        # Regression disqualifies formal output even after prefix salvage.
        (session / "manifest.json").write_text(json.dumps({
            "schema": "panorama-demo-session/v2", "clean_shutdown": False,
            "capture_mode": "continuous_rgbd_video_auto", "received_frames": 1,
            "timestamp_regressions": 1 if stop == "transport" else 0}))
        (session / "calibration.json").write_text("{}")
        rgb, depth = session / "color/1.jpg", session / "depth_aligned/1.png"
        cv2.imwrite(str(rgb), np.full((24, 40, 3), 90, np.uint8))
        cv2.imwrite(str(depth), np.full((24, 40), 500, np.uint16))
        args.capture_progress("WARMING_UP", session_root=session, camera_serial="TEST")
        args.capture_progress("CAPTURING", session_root=session)
        journal = CommitJournal(session)
        with (session / "frames.csv").open("w", newline="") as handle:
            row = {"frame_id": 1, "raw_depth_path": "", "color_exposure": 8}
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
            journal.record(SimpleNamespace(frame_id=1, timestamp_us=10, color_path=rgb,
                aligned_depth_path=depth, color_sha256=file_sha256(rgb),
                aligned_depth_sha256=file_sha256(depth)), row, handle)
        journal.close()
        if stop == "low_disk":
            args.disk_guard.space = lambda _: (1024**3, 4096)
            assert not args.disk_guard.check(1, force=True)
            args.capture_progress("STOPPING", committed_frames=1, stop_reason="LOW_DISK")
            raise RuntimeError("Only one frame available for finalization")
        raise _VideoCapturePreflightError("transport failed after commit")

    monkeypatch.setattr(video_live, "run", failed_capture)
    sdk = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off"))
    job = sdk.start_capture(tmp_path / "owned")
    result = job.result(5)
    assert calls == 1
    assert result.formal_2d_published is False
    assert result.emergency_2d_path.is_file()
    assert job.state.value == "COMPLETED_WITH_WARNINGS"
    assert not (result.output_dir / "video_delivery.json").exists()
    assert (job._record.root / "session/run_1/color/1.jpg").is_file()
    other = Gemini305VideoSDK(sdk.config)
    assert other.get_job(job.job_id).result().emergency_2d_path == result.emergency_2d_path


def test_second_client_can_observe_but_cannot_cancel_controller(tmp_path, monkeypatch):
    from panorama_demo import SDKBusyError
    entered = threading.Event()

    def waiting(args):
        entered.set()
        args.cancel_event.wait(5)
        raise RuntimeError("cancelled")

    monkeypatch.setattr(video_live, "run", waiting)
    sdk = Gemini305VideoSDK(VideoSDKConfig(cuda_mode="off"))
    job = sdk.start_capture(tmp_path / "owned")
    assert entered.wait(2)
    other = Gemini305VideoSDK(sdk.config)
    try:
        assert other.get_status(job.job_id)["state"] == "WAITING_FOR_CAMERA"
        with pytest.raises(SDKBusyError):
            other.stop_capture(job.job_id)
        with pytest.raises(SDKBusyError):
            other.start_capture(tmp_path / "second")
    finally:
        sdk.release_control()
    assert job.state is VideoJobState.CANCELLED
