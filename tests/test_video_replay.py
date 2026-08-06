from __future__ import annotations

import json

import pytest

from panorama_demo.video_replay import _scan_facts_sha256


def test_replay_facts_digest_ignores_destination_only_and_detects_motion_change(tmp_path):
    base = {
        "input_sha256": {"frames": "a"}, "frame_file_sha256": [], "qualities": [],
        "motions": [{"dx": 1.0}], "scan_segment": {"start_index": 0}, "origin": "offline_prepare",
    }
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    first.write_text(json.dumps(base), encoding="utf-8")
    altered = dict(base)
    altered["origin"] = "capture"
    second.write_text(json.dumps(altered), encoding="utf-8")
    assert _scan_facts_sha256(first) == _scan_facts_sha256(second)
    altered["motions"] = [{"dx": 2.0}]
    second.write_text(json.dumps(altered), encoding="utf-8")
    assert _scan_facts_sha256(first) != _scan_facts_sha256(second)


def test_replay_facts_digest_rejects_invalid_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid online state"):
        _scan_facts_sha256(path)
