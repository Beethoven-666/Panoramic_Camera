from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo.video_panorama import (
    _cached_trajectory,
    _contained_online_tracking_indices,
)
from panorama_demo.video_trajectory_cache import (
    TRAJECTORY_CACHE_SCHEMA,
    freeze_verified_trajectory,
    session_input_sha256,
)


def _published_report(session: Path, report_directory: Path) -> Path:
    report_directory.mkdir()
    for name, value in {
        "manifest.json": "manifest\n",
        "calibration.json": "calibration\n",
        "frames.csv": "frames\n",
    }.items():
        (session / name).write_text(value, encoding="utf-8")
    hashes = session_input_sha256(session)
    poses = [np.eye(4).tolist(), np.array(((1, 0, 0, 10), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))).tolist()]
    report = {
        "schema": "gemini305-video-panorama-report/v2",
        "delivery_state": "published",
        "grades": {"structural": "A"},
        "input_sha256": hashes,
        "motion_analysis_frame_ids": [10, 11, 12],
        "orb_tracking_frame_ids": [10, 12],
        "orb_tracking_source": "real_time_spaced_video_frames_only",
        "orbslam3": {
            "tracked_frame_ids": [10, 12],
            "camera_to_world": poses,
            "attempts": [{"attempt": 1, "status": "ok"}],
            "pose_convention": "camera_to_world",
            "translation_unit": "mm",
        },
    }
    path = report_directory / "video_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    (report_directory / "video_delivery.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-video-panorama-delivery/v2",
                "delivery_state": "published",
                "report": "video_report.json",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_freeze_verified_report_creates_cache_accepted_by_video_renderer(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    report_path = _published_report(session, tmp_path / "published")
    cache = tmp_path / "trajectory_cache.json"

    frozen = freeze_verified_trajectory(
        input_path=session, report_path=report_path, output_path=cache
    )

    assert frozen["created"] is True
    payload = json.loads(cache.read_text(encoding="utf-8"))
    assert payload["schema"] == TRAJECTORY_CACHE_SCHEMA
    assert payload["orbslam3"]["camera_to_world"][1][0][3] == 10
    poses, ids, attempts = _cached_trajectory(
        cache,
        frames=(SimpleNamespace(frame_id=10), SimpleNamespace(frame_id=12)),
        input_sha256=session_input_sha256(session),
        require_capture_provenance=False,
    )
    assert ids == (10, 12)
    assert poses[12][0, 3] == pytest.approx(10.0)
    assert attempts == [{"attempt": 1, "status": "ok"}]
    assert freeze_verified_trajectory(
        input_path=session, report_path=report_path, output_path=cache
    )["created"] is False


def test_freeze_rejects_input_hash_drift_and_malformed_real_pose(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    report_path = _published_report(session, tmp_path / "published")
    (session / "manifest.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input hashes"):
        freeze_verified_trajectory(
            input_path=session, report_path=report_path, output_path=tmp_path / "cache.json"
        )

    # Restore the original input and corrupt the source report pose.  The
    # freezer must not launder a non-rigid matrix into a reusable cache.
    (session / "manifest.json").write_text("manifest\n", encoding="utf-8")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["orbslam3"]["camera_to_world"][1][0][0] = 2.0
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not rigid"):
        freeze_verified_trajectory(
            input_path=session, report_path=report_path, output_path=tmp_path / "cache.json"
        )


def test_cached_experiment_trajectory_rejects_changed_frames_csv(tmp_path: Path) -> None:
    """A cache must not silently cross a changed real-source file mapping."""

    session = tmp_path / "session"
    session.mkdir()
    report_path = _published_report(session, tmp_path / "published")
    cache = tmp_path / "trajectory_cache.json"
    freeze_verified_trajectory(input_path=session, report_path=report_path, output_path=cache)

    # The frame ids can remain plausible while their source paths or ordering
    # change, so the cache loader must compare the complete input-lock triplet
    # rather than only calibration/manifest metadata.
    (session / "frames.csv").write_text("changed real source mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input hashes"):
        _cached_trajectory(
            cache,
            frames=(SimpleNamespace(frame_id=10), SimpleNamespace(frame_id=12)),
            input_sha256=session_input_sha256(session),
            require_capture_provenance=False,
        )


def test_progress_slice_reuses_contained_online_anchors_without_rephasing(tmp_path: Path) -> None:
    """A slice beginning between 8 FPS anchors must retain cache-identical IDs."""

    session = tmp_path / "session"
    session.mkdir()
    for name, value in {
        "manifest.json": "manifest\n",
        "calibration.json": "calibration\n",
        "frames.csv": "frames\n",
    }.items():
        (session / name).write_text(value, encoding="utf-8")

    frames = []
    for frame_id in range(101, 109):
        color = session / f"color_{frame_id}.png"
        depth = session / f"depth_{frame_id}.png"
        color.write_bytes(f"color-{frame_id}".encode())
        depth.write_bytes(f"depth-{frame_id}".encode())
        frames.append(
            SimpleNamespace(
                frame_id=frame_id, color_path=color, aligned_depth_path=depth
            )
        )

    # The full capture's certified chain was [100, 103, 106, 109].  The
    # restricted real-source interval starts at 101 (not an anchor), so a
    # fresh target-FPS selection would phase differently.  It must instead
    # select only the contained real cached anchors [103, 106].
    indices = _contained_online_tracking_indices(tuple(frames), (100, 103, 106, 109))
    assert indices == (2, 5)
    selected = tuple(frames[index] for index in indices)

    def digest(path: Path) -> str:
        import hashlib

        return hashlib.sha256(path.read_bytes()).hexdigest()

    cache = tmp_path / "online_orbslam3_trajectory.json"
    cache.write_text(
        json.dumps(
            {
                "schema": "gemini305-online-orbslam3-trajectory/v1",
                "capture_origin": "writer_committed_files",
                "input_sha256": session_input_sha256(session),
                "source_file_sha256": [
                    {
                        "frame_id": frame.frame_id,
                        "color_sha256": digest(frame.color_path),
                        "aligned_depth_sha256": digest(frame.aligned_depth_path),
                    }
                    for frame in selected
                ],
                "orbslam3": {
                    "tracked_frame_ids": [103, 106],
                    "camera_to_world": [np.eye(4).tolist(), np.eye(4).tolist()],
                    "attempts": [],
                },
            }
        ),
        encoding="utf-8",
    )
    poses, ids, _ = _cached_trajectory(
        cache,
        frames=selected,
        input_sha256=session_input_sha256(session),
        require_capture_provenance=True,
    )
    assert ids == (103, 106)
    assert set(poses) == {103, 106}
