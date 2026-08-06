from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo.video_panorama import _cached_trajectory
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
