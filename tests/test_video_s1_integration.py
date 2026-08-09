from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

import panorama_demo.video_s1_experiment as experiment
from panorama_demo.quality import FrameQuality, MotionEstimate
from panorama_demo.session import CameraIntrinsics, RGBDFrame, RGBDSession
from panorama_demo.video_session import VideoSession


ROOT = Path(__file__).resolve().parents[1]


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
