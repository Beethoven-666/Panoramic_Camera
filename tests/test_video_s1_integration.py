from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import panorama_demo.video_s1_experiment as experiment
from panorama_demo.quality import FrameQuality, MotionEstimate
from panorama_demo.session import CameraIntrinsics, RGBDFrame, RGBDSession
from panorama_demo.video_session import VideoSession


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_s01_baseline_verifier_rejects_artifact_drift(tmp_path: Path) -> None:
    session = tmp_path / "run_test"
    baseline = tmp_path / "baseline"
    session.mkdir()
    baseline.mkdir()
    for name, payload in (
        ("manifest.json", b"{}"),
        ("calibration.json", b"{}"),
        ("frames.csv", b"frame_id\n"),
    ):
        (session / name).write_bytes(payload)
    report = {"algorithm_id": "S01_output_first_vertical_alignment_v1", "source_selection": {"source_frame_ids": [1, 2]}}
    (baseline / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (baseline / "source_layout.json").write_text('{"sources": []}', encoding="utf-8")
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    assert cv2.imwrite(str(baseline / "s0_panorama_owner_only.png"), image)
    assert cv2.imwrite(str(baseline / "s1_panorama_owner_only.png"), image)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "schema": "gemini305-video-s1-baseline-lock/v1",
        "runs": {
            "run_test": {
                "session": {
                    "manifest_sha256": _sha(session / "manifest.json"),
                    "calibration_sha256": _sha(session / "calibration.json"),
                    "frames_csv_sha256": _sha(session / "frames.csv"),
                },
                "canvas_width": 6,
                "canvas_height": 4,
                "source_frame_ids": [1, 2],
                "artifacts": {
                    "report_sha256": _sha(baseline / "report.json"),
                    "source_layout_sha256": _sha(baseline / "source_layout.json"),
                    "s0_owner_only_sha256": _sha(baseline / "s0_panorama_owner_only.png"),
                    "s1_owner_only_sha256": _sha(baseline / "s1_panorama_owner_only.png"),
                },
            }
        },
    }), encoding="utf-8")

    verified = experiment.verify_s01_baseline(session, baseline, lock_path=lock)
    assert verified["run_name"] == "run_test"
    (baseline / "report.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        experiment.verify_s01_baseline(session, baseline, lock_path=lock)


def test_experiment_writes_complete_s0_s1_when_pair_alignment_degrades(tmp_path, monkeypatch) -> None:
    height, width = 96, 160
    frames = []
    for frame_id in (1, 2):
        color = tmp_path / f"c{frame_id}.png"
        depth = tmp_path / f"d{frame_id}.png"
        assert cv2.imwrite(str(color), np.full((height, width, 3), 127, np.uint8))
        assert cv2.imwrite(str(depth), np.full((height, width), 1000, np.uint16))
        frames.append(RGBDFrame(frame_id, color, depth, 1.0, frame_id * 1000, 5, 1))
    calibration = CameraIntrinsics(width, height, 130.0, 130.0, width / 2, height / 2, (0.0,) * 8)
    rgbd = RGBDSession(tmp_path, calibration, tuple(frames), {"capture_mode": "continuous_rgbd_video_auto"})
    video = VideoSession(rgbd, "continuous_rgbd_video_auto", True, True)
    quality = FrameQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(experiment, "load_video_session", lambda *args, **kwargs: video)
    monkeypatch.setattr(
        experiment,
        "analyse_video_scan",
        lambda *args, **kwargs: (
            [quality, quality],
            [MotionEstimate(10.0, 0.0, 30, 0.8, 0.5, "features")],
            {"start_index": 0, "end_index": 1, "scan_direction": 1},
        ),
    )
    output = tmp_path / "out"

    report = experiment.run(
        tmp_path,
        ROOT / "configs/video_candidates/S01_output_first_vertical_alignment_v1.yaml",
        output,
    )

    assert report["complete_panorama_generated"] is True
    assert report["fatal_quality_gate"] is False
    assert report["s1"]["pair_zero_correction_count"] == 1
    for name in (
        "s0_panorama_owner_only.png", "s0_panorama.png", "s1_panorama_owner_only.png",
        "s1_panorama.png", "owner_map.png", "valid_mask.png", "source_layout.json",
        "source_selection.csv", "overlap_summary.csv", "pair_summary.csv",
        "performance.json", "report.json", "config_snapshot.yaml",
    ):
        assert (output / name).is_file(), name
