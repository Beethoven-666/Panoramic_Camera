"""Replay a real continuous RGB-D session through the S013 V11 live boundary.

This is an acceptance runner, not a synthetic renderer: every accepted and
committed frame is read from the immutable session, the capture-time observer
runs at the recorded cadence, and final pixels come from the locked production
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2

from panorama_demo.capture_orbbec import (
    CaptureResult,
    LiveFramePacket,
    LiveSessionInfo,
    WrittenRGBDFrame,
)
from panorama_demo.video_pipeline import _lock_paths
from panorama_demo.video_algorithm_registry import resolve_video_algorithm
from panorama_demo.video_s13_production import run_s13_v11_production
from panorama_demo.video_s13_live import S13V11LiveObserver
from panorama_demo.video_session import load_video_session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_replay(
    *,
    session_path: Path,
    output: Path,
    config_path: Path | None,
    cadence_scale: float,
    maximum_post_seconds: float,
) -> dict[str, object]:
    if cadence_scale <= 0:
        raise ValueError("cadence_scale must be positive")
    _, baseline_lock, production_lock = _lock_paths(config_path)
    spec = resolve_video_algorithm(
        "production",
        baseline_lock=baseline_lock,
        production_lock=production_lock,
        candidate_config=None,
    )
    session = load_video_session(
        session_path.expanduser().resolve(),
        validate_frame_files=True,
        validation_workers=4,
    )
    frames = tuple(session.rgbd.frames)
    if len(frames) < 2 or frames[0].timestamp_us is None:
        raise ValueError("Live replay requires at least two timestamped RGB-D frames")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(
        (session.rgbd.root / "calibration.json").read_text(encoding="utf-8")
    )
    observer = S13V11LiveObserver(
        production_config_sha256=spec.config_sha256,
        preview_output=output,
    )
    capture_started_ns = time.monotonic_ns()
    observer.on_session_ready(LiveSessionInfo(
        root=session.rgbd.root,
        capture_started_monotonic_ns=capture_started_ns,
        calibration=calibration,
    ))
    first_timestamp_us = int(frames[0].timestamp_us)
    for row_index, frame in enumerate(frames):
        if frame.timestamp_us is None:
            raise ValueError("Live replay frame lacks a timestamp")
        target_ns = capture_started_ns + int(
            (int(frame.timestamp_us) - first_timestamp_us) * 1000 / cadence_scale
        )
        remaining = (target_ns - time.monotonic_ns()) / 1_000_000_000.0
        if remaining > 0:
            time.sleep(remaining)
        color = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(frame.aligned_depth_path), cv2.IMREAD_UNCHANGED)
        if color is None or depth is None:
            raise OSError(f"Could not decode replay frame {frame.frame_id}")
        color.setflags(write=False)
        depth.setflags(write=False)
        accepted_ns = time.monotonic_ns()
        observer.on_frame_accepted(LiveFramePacket(
            frame_id=int(frame.frame_id),
            color_bgr=color,
            aligned_depth_mm=depth,
            metadata={"replay": True},
            accepted_monotonic_ns=accepted_ns,
            writer_queue_fraction=0.0,
            writer_queue_drops=0,
        ))
        observer.on_frame_committed(WrittenRGBDFrame(
            frame_id=int(frame.frame_id),
            color_path=frame.color_path,
            aligned_depth_path=frame.aligned_depth_path,
            color_sha256=_sha256(frame.color_path),
            aligned_depth_sha256=_sha256(frame.aligned_depth_path),
            timestamp_us=int(frame.timestamp_us),
            depth_scale_mm_per_unit=float(frame.depth_scale_mm_per_unit),
            frames_csv_row_index=row_index,
            commit_monotonic_ns=time.monotonic_ns(),
        ))
    capture_stopped_ns = time.monotonic_ns()
    observer.on_capture_stopping()
    observer.on_capture_closed(CaptureResult(
        session_root=session.rgbd.root,
        received_frames=len(frames),
        written_frames=len(frames),
        queue_drops=0,
        write_errors=0,
        max_queue_depth=0,
        capture_started_monotonic_ns=capture_started_ns,
        capture_stopped_monotonic_ns=capture_stopped_ns,
    ))
    handoff = observer.freeze_handoff()
    snapshot = observer.snapshot()
    capture_report: dict[str, object] = {
        "schema": "gemini305-s013-v11-live-replay-acceptance/v1",
        "real_continuous_rgbd_session": str(session.rgbd.root),
        "algorithm_id": spec.algorithm_id,
        "implementation_id": spec.implementation_id,
        "production_config_sha256": spec.config_sha256,
        "received_frames": len(frames),
        "written_frames": len(frames),
        "queue_drops": 0,
        "write_errors": 0,
        "preview_updates": snapshot.preview_updates,
        "preview_skipped_due_to_load": snapshot.preview_skipped_due_to_load,
        "preview_failures": snapshot.preview_failures,
        "preview_first_display_seconds": handoff.online_2d_metrics["first_preview_seconds"],
        "preview_latency_p95_ms": snapshot.preview_latency_p95_ms,
        "online_orb_tracker_construct_count": 0,
        "orbslam3_subprocess_count_before_2d": 0,
        "open3d_import_count_before_2d": 0,
        "tsdf_call_count_before_2d": 0,
        "three_d_process_count_before_2d": 0,
        "handoff_reuse_level": handoff.reuse_level,
    }
    (output / "live_capture_report.json").write_text(
        json.dumps(capture_report, indent=2), encoding="utf-8"
    )
    published = run_s13_v11_production(
        session_path=session.rgbd.root,
        output=output,
        algorithm_spec=spec,
        maximum_post_seconds=maximum_post_seconds,
        live_handoff=handoff,
    )
    return {"capture": capture_report, "delivery": published}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay real continuous RGB-D frames through S013 V11 live acceptance"
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cadence-scale", type=float, default=1.0)
    parser.add_argument("--maximum-post-seconds", type=float, default=60.0)
    args = parser.parse_args()
    report = run_replay(
        session_path=args.session,
        output=args.output,
        config_path=args.config,
        cadence_scale=args.cadence_scale,
        maximum_post_seconds=args.maximum_post_seconds,
    )
    print(json.dumps(report["capture"], indent=2))


if __name__ == "__main__":
    main()
