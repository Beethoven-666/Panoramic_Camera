from __future__ import annotations

import json
import csv
from types import SimpleNamespace

from panorama_demo.commit_journal import CommitJournal, file_sha256
from panorama_demo.video_recovery import salvage_prefix

import pytest

from panorama_demo.video_recovery import VideoRecoveryError, checkpoint_blocked_selection


def _blocked_root(tmp_path):
    root = tmp_path / "run_20260804_162340"
    root.mkdir()
    (root / "algorithm_selection_v2_current.json").write_text(
        json.dumps(
            {
                "selection_status": "not_selectable",
                "selected_algorithm_id": None,
                "candidates": [
                    {
                        "algorithm_id": "C1_constrained_owner",
                        "eligible": False,
                        "reasons": ["line_continuity_hard_gate_not_passed"],
                        "report_path": "candidate/video_report.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "holdout_state.json").write_text(
        json.dumps({"first_holdout_attempted": False, "production_frozen": False}), encoding="utf-8"
    )
    return root


def test_checkpoint_preserves_blocked_selection_and_does_not_create_production_lock(tmp_path):
    root = _blocked_root(tmp_path)

    result = checkpoint_blocked_selection(root, commit="a" * 40, test_result="1080 passed, 2 skipped")

    recovery = root / "recovery"
    assert result["candidate_count"] == 1
    assert result["holdout_not_reserved"] is True
    assert (recovery / "blocked_selection_snapshot.json").is_file()
    assert (recovery / "blocked_candidate_matrix.csv").is_file()
    assert (recovery / "blocked_test_result.txt").read_text(encoding="utf-8") == "1080 passed, 2 skipped\n"
    assert (root / "quality_gate_lock.json").is_file()
    assert not list(root.rglob("production.lock.json"))


def test_checkpoint_refuses_consumed_holdout_or_existing_production_lock(tmp_path):
    root = _blocked_root(tmp_path)
    (root / "holdout_state.json").write_text(
        json.dumps({"first_holdout_attempted": True, "production_frozen": False}), encoding="utf-8"
    )

    with pytest.raises(VideoRecoveryError, match="consumed holdout"):
        checkpoint_blocked_selection(root, commit="a" * 40, test_result="test")


def test_recovery_preserves_original_failure_and_excludes_unconfirmed_tail(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "color").mkdir()
    (source / "depth_aligned").mkdir()
    original = {"schema": "panorama-demo-session/v2", "clean_shutdown": False,
                "capture_mode": "continuous_rgbd_video_auto", "received_frames": 2,
                "timestamp_regressions": 1, "capture_error": {"type": "Disconnected"}}
    (source / "manifest.json").write_text(json.dumps(original))
    (source / "calibration.json").write_text("{}")
    journal = CommitJournal(source)
    color, depth = source / "color/1.jpg", source / "depth_aligned/1.png"
    color.write_bytes(b"committed color")
    depth.write_bytes(b"committed depth")
    with (source / "frames.csv").open("w", newline="") as handle:
        row = {"frame_id": 1, "raw_depth_path": ""}
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        journal.record(SimpleNamespace(frame_id=1, timestamp_us=10, color_path=color,
                                       aligned_depth_path=depth, color_sha256=file_sha256(color),
                                       aligned_depth_sha256=file_sha256(depth)), row, handle)
    journal.close()
    (source / "color/2.jpg").write_bytes(b"unconfirmed")
    recovered, rows, report = salvage_prefix(source, tmp_path / "recovered")
    assert len(rows) == 1
    assert not (recovered / "color/2.jpg").exists()
    assert report["truncated_frame_count"] == 1
    assert report["formal_eligible"] is False
    assert json.loads((source / "manifest.json").read_text()) == original
    manifest = json.loads((recovered / "manifest.json").read_text())
    assert manifest["clean_shutdown"] is True
    assert manifest["recovery"]["original_clean_shutdown"] is False
    assert manifest["product_eligibility"]["video_panorama"] is False
