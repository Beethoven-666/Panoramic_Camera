"""Post-capture ORB-SLAM3 and final TSDF delivery in an independent process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from .video_3d_launcher import write_isolated_3d_failure

TRAJECTORY_SCHEMA = "gemini305-video-post-capture-orbslam3-trajectory/v1"
TRAJECTORY_LOCK_SCHEMA = "gemini305-video-post-capture-orbslam3-lock/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    try:
        pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _tracking_frames(frames: tuple[Any, ...], *, target_fps: float) -> tuple[Any, ...]:
    if len(frames) < 2 or not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Post-capture 3-D requires two frames and a positive tracking FPS")
    interval_us = 1_000_000.0 / float(target_fps)
    selected = [frames[0]]
    previous = frames[0].timestamp_us
    if previous is None:
        raise ValueError("Post-capture ORB source lacks a timestamp")
    for frame in frames[1:]:
        if frame.timestamp_us is None:
            raise ValueError("Post-capture ORB source lacks a timestamp")
        if float(frame.timestamp_us - previous) >= interval_us:
            selected.append(frame)
            previous = frame.timestamp_us
    if selected[-1].frame_id != frames[-1].frame_id:
        selected.append(frames[-1])
    return tuple(selected)


def _write_trajectory(
    output_3d: Path,
    *,
    session_root: Path,
    tracked_frames: tuple[Any, ...],
    trajectory: Any,
) -> Path:
    requested_ids = [int(frame.frame_id) for frame in tracked_frames]
    tracked_ids = [int(value) for value in trajectory.tracked_frame_ids]
    if tracked_ids != requested_ids or set(trajectory.poses_by_frame_id) != set(requested_ids):
        raise RuntimeError("Post-capture ORB-SLAM3 did not return the complete genuine chain")
    payload: dict[str, object] = {
        "schema": TRAJECTORY_SCHEMA,
        "pose_convention": "camera_to_world",
        "translation_unit": "mm",
        "tracked_frame_ids": tracked_ids,
        "camera_to_world": [
            np.asarray(trajectory.poses_by_frame_id[frame_id], dtype=np.float64).tolist()
            for frame_id in tracked_ids
        ],
        "attempts": [dict(row) for row in trajectory.attempt_audit],
        "interpolated_pose_count": 0,
        "extrapolated_pose_count": 0,
    }
    trajectory_path = output_3d / "orbslam3_trajectory.json"
    _atomic_json(trajectory_path, payload)
    lock = {
        "schema": TRAJECTORY_LOCK_SCHEMA,
        "trajectory": trajectory_path.name,
        "trajectory_sha256": _sha256(trajectory_path),
        "session_input_sha256": {
            name: _sha256(session_root / filename)
            for name, filename in (
                ("manifest", "manifest.json"),
                ("calibration", "calibration.json"),
                ("frames_csv", "frames.csv"),
            )
        },
    }
    _atomic_json(output_3d / "orbslam3_trajectory.lock.json", lock)
    return trajectory_path


def run_post_capture_3d(
    *,
    session_path: Path,
    two_d_output: Path,
    output_3d: Path,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Generate the only ORB trajectory and final 3-D artifacts after 2-D."""

    session_root = session_path.expanduser().resolve()
    session_root = session_root if session_root.is_dir() else session_root.parent
    two_d_root = two_d_output.expanduser().resolve()
    destination = output_3d.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    delivery_path = two_d_root / "video_delivery.json"
    if not delivery_path.is_file():
        raise ValueError("Post-capture 3-D requires a published 2-D delivery")
    two_d_sha = _sha256(delivery_path)
    started_ns = time.monotonic_ns()
    timing: dict[str, object] = {
        "schema": "gemini305-video-post-capture-3d-timing/v1",
        "process_started_monotonic_ns": started_ns,
        "preview_generated": False,
        "isolation": {
            "two_d_delivery_sha256": two_d_sha,
            "runs_in_child_process": True,
            "orb_is_post_capture_only": True,
            "preview_generated": False,
        },
    }
    try:
        # These are child-only imports.  The 2-D parent imports only the pure
        # stdlib launcher and therefore cannot initialize ORB/Open3D/TSDF state.
        from .orbslam3_bridge import run_orbslam3_rgbd
        from .video_3d import _invalidate_3d, _load_2d_report, publish_video_3d
        from .video_session import load_video_session

        _invalidate_3d(destination)
        for name in ("orbslam3_trajectory.json", "orbslam3_trajectory.lock.json"):
            (destination / name).unlink(missing_ok=True)
        _load_2d_report(two_d_root)
        video = load_video_session(session_root, validate_frame_files=True, validation_workers=4)
        stitch = dict(config.get("stitch", {}))
        settings = dict(stitch.get("video_panorama", {}))
        tracking_fps = float(settings.get("fast_orb_target_fps", 8.0))
        tracked_frames = _tracking_frames(tuple(video.rgbd.frames), target_fps=tracking_fps)
        orb_config = dict(stitch.get("orbslam3_rgbd", {}))
        fast_orb = settings.get("fast_orbslam3_rgbd")
        if isinstance(fast_orb, Mapping):
            orb_config.update(dict(fast_orb))
        orb_config["minimum_tracked_fraction"] = 1.0
        timing["orb_started_monotonic_ns"] = time.monotonic_ns()
        with tempfile.TemporaryDirectory(prefix="g305-video-post-3d-orb-") as work:
            trajectory = run_orbslam3_rgbd(
                tracked_frames,
                video.rgbd.calibration,
                work,
                config=orb_config,
            )
        timing["orb_completed_monotonic_ns"] = time.monotonic_ns()
        trajectory_path = _write_trajectory(
            destination,
            session_root=session_root,
            tracked_frames=tracked_frames,
            trajectory=trajectory,
        )
        timing["final_3d_started_monotonic_ns"] = time.monotonic_ns()
        result = publish_video_3d(
            destination,
            input_path=session_root,
            trajectory_path=trajectory_path,
            source_frame_ids=[int(frame.frame_id) for frame in tracked_frames],
            config=config,
            two_d_output=two_d_root,
        )
        timing["final_3d_completed_monotonic_ns"] = time.monotonic_ns()
        timing["status"] = "published"
        return result
    except Exception as exc:
        timing["failed_monotonic_ns"] = time.monotonic_ns()
        timing["status"] = "failed"
        timing["error_type"] = type(exc).__name__
        write_isolated_3d_failure(
            destination, two_d_output=two_d_root, exc=exc, stage="post_capture_3d"
        )
        if _sha256(delivery_path) != two_d_sha:
            raise RuntimeError("Post-capture 3-D changed the published 2-D delivery") from exc
        raise
    finally:
        timing["process_finished_monotonic_ns"] = time.monotonic_ns()
        _atomic_json(destination / "video_3d_timing.json", timing)


def main() -> None:
    from .config import load_config

    parser = argparse.ArgumentParser(
        description="Run post-capture ORB-SLAM3 and publish final video 3-D artifacts"
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--two-d-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        run_post_capture_3d(
            session_path=args.session,
            two_d_output=args.two_d_output,
            output_3d=args.output,
            config=load_config(args.config),
        )
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()


__all__ = ["run_post_capture_3d"]
