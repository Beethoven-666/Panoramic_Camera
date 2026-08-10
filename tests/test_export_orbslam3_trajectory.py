from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import panorama_demo.export_orbslam3_trajectory as exporter
from panorama_demo.synthetic import generate_sequence


def test_exporter_writes_only_complete_real_orbslam3_pose_payload(
    tmp_path: Path, monkeypatch
) -> None:
    session_root = generate_sequence(
        tmp_path / "session", frame_count=3, frame_width=64, frame_height=48, step=8
    )

    def fake_run(frames, _calibration, _work, *, config):
        stdout_path = Path(_work) / "orbslam3.stdout.txt"
        stderr_path = Path(_work) / "orbslam3.stderr.txt"
        stdout_path.write_text("tracked all frames\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        poses = {frame.frame_id: np.eye(4, dtype=np.float64) for frame in frames}
        poses[frames[1].frame_id][0, 3] = 10.0
        return SimpleNamespace(
            tracked_frame_ids=tuple(frame.frame_id for frame in frames),
            poses_by_frame_id=poses,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            as_dict=lambda *, input_frame_count: {
                "backend": "orbslam3_rgbd_wsl",
                "input_frame_count": input_frame_count,
                "tracked_frame_count": input_frame_count,
                "tracked_fraction": 1.0,
                "pose_convention": "camera_to_world",
                "translation_unit": "mm",
                "config_enabled": config.enabled,
            },
        )

    monkeypatch.setattr(exporter, "run_orbslam3_rgbd", fake_run)
    output = tmp_path / "trajectory.json"
    payload = exporter.export_trajectory(session_root, output)

    assert output.is_file()
    assert payload["schema"] == "gemini305-orbslam3-trajectory/v2"
    assert payload["complete_tracking_required"] is True
    assert [row["frame_id"] for row in payload["poses"]] == [0, 1, 2]
    assert payload["poses"][1]["camera_to_world"][0][3] == 10.0
    assert payload["poses"][1]["pose_kind"] == "direct_orbslam3"
    assert payload["poses"][1]["pose_status"] == "valid"
    assert payload["poses"][1]["pose_origin"] == "offline_orbslam3_rgbd_full_session"
    assert payload["poses"][1]["tracking_state"] == "tracked"
    assert isinstance(payload["poses"][1]["timestamp_us"], int)
    assert output.with_suffix(".stdout.txt").read_text(encoding="utf-8") == "tracked all frames\n"
    assert output.with_suffix(".stderr.txt").is_file()
    audit = json.loads(output.with_suffix(".audit.json").read_text(encoding="utf-8"))
    assert audit["tracked_frame_count"] == 3
    assert audit["untracked_frame_ids"] == []
    assert len(audit["trajectory_sha256"]) == 64
    assert not list(tmp_path.glob(".trajectory.json.pending"))


def test_exporter_rejects_partial_tracking_without_writing_output(
    tmp_path: Path, monkeypatch
) -> None:
    session_root = generate_sequence(
        tmp_path / "session", frame_count=3, frame_width=64, frame_height=48, step=8
    )

    def fake_run(frames, *_args, **_kwargs):
        return SimpleNamespace(
            tracked_frame_ids=(frames[0].frame_id, frames[1].frame_id),
            poses_by_frame_id={},
        )

    monkeypatch.setattr(exporter, "run_orbslam3_rgbd", fake_run)
    output = tmp_path / "trajectory.json"
    try:
        exporter.export_trajectory(session_root, output)
    except RuntimeError as exc:
        assert "did not track every" in str(exc)
    else:
        raise AssertionError("partial trajectory unexpectedly exported")
    assert not output.exists()
