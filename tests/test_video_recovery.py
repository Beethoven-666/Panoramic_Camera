from __future__ import annotations

import json

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
