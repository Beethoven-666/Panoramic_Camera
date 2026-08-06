from __future__ import annotations

import argparse
import csv
import json

import cv2
import numpy as np
import pytest

from panorama_demo.video_benchmark import _write_leaderboard, _write_visual_metrics_if_annotated, run


def test_leaderboard_update_is_idempotent_and_preserves_fail_closed_status(tmp_path):
    path = tmp_path / "leaderboard.csv"
    summary = {
        "algorithm": {"algorithm_id": "C4", "config_sha256": "a" * 64},
        "run_count": 3,
        "warm_median_seconds": 11.0,
        "warm_max_seconds": 12.0,
        "gate_status": "failed",
        "result_sha256": "b" * 64,
    }
    _write_leaderboard(path, summary)
    _write_leaderboard(path, summary)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["gate_status"] == "failed"
    assert rows[0]["result_sha256"] == "b" * 64


def test_candidate_benchmark_requires_an_immutable_non_holdout_interval(tmp_path):
    args = argparse.Namespace(
        repeat=1, algorithm="candidate", candidate_config=tmp_path / "candidate.yaml",
        progress_range=None, split=None, session=tmp_path / "session", output=tmp_path / "output", config=None,
    )
    with pytest.raises(ValueError, match="immutable --split"):
        run(args)


def test_annotation_metrics_are_read_only_and_cannot_promote_the_grade(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    image = np.full((10, 12, 3), 30, dtype=np.uint8)
    owner = np.full((10, 12), 7, dtype=np.int32)
    assert cv2.imwrite(str(output / "video_panorama.png"), image)
    assert cv2.imwrite(str(output / "video_panorama.jpg"), image)
    np.savez_compressed(output / "video_pixel_provenance.npz", owner_frame_id=owner)
    report = {
        "algorithm": {"role": "candidate", "algorithm_id": "C4", "implementation_id": "test", "config_sha256": "a" * 64, "source_commit": "b", "model_sha256": {}},
        "grades": {"structural": "A", "visual": "C", "performance": "A", "overall": "C"},
        "evaluation_scope": "validation_only", "input_sha256": {},
    }
    (output / "video_report.json").write_text(json.dumps(report), encoding="utf-8")
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "objects.json").write_text(json.dumps({
        "schema": "gemini305-video-source-annotations/v1",
        "source_frames": {"7": {"color_path": "color/00000007.jpg", "scan_progress": 0.2}},
        "objects": [{"id": "object", "frame_id": 7, "polygon": [[0, 0], [1, 0], [1, 1]]}],
        "lines": [{"id": "line", "frame_id": 7, "points": [[0, 0], [1, 0]]}],
        "safe_background": [{"id": "background", "frame_id": 7, "polygon": [[0, 0], [1, 0], [1, 1]]}],
    }), encoding="utf-8")
    image_before = (output / "video_panorama.png").read_bytes()
    owner_before = (output / "video_pixel_provenance.npz").read_bytes()
    evidence = _write_visual_metrics_if_annotated(output, tmp_path)
    assert evidence is not None
    assert evidence["automatic_grade_promotion_allowed"] is False
    assert evidence["hard_gate_pass"] is False
    assert (output / "video_panorama.png").read_bytes() == image_before
    assert (output / "video_pixel_provenance.npz").read_bytes() == owner_before
