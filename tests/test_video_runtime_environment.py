from __future__ import annotations

import json

from panorama_demo.video_runtime_environment import (
    DETERMINISTIC_RESULT_SCHEMA,
    atomic_write_json,
    deterministic_result_payload,
)


def _report() -> dict[str, object]:
    return {
        "algorithm": {
            "role": "candidate",
            "algorithm_id": "C4",
            "implementation_id": "test",
            "config_sha256": "a" * 64,
            "source_commit": "b" * 40,
            "model_sha256": {"raft": "c" * 64},
        },
        "grades": {"structural": "A", "visual": "C", "performance": "A", "overall": "C"},
        "evaluation_scope": "validation_only",
        "input_sha256": {"frames": "d" * 64},
        "performance": {"post_capture_seconds": 9.1},
    }


def test_deterministic_result_ignores_timing_and_is_content_addressed(tmp_path):
    for name, data in (
        ("video_panorama.png", b"png"),
        ("video_panorama.jpg", b"jpg"),
        ("video_pixel_provenance.npz", b"npz"),
    ):
        (tmp_path / name).write_bytes(data)
    first = deterministic_result_payload(tmp_path, _report())
    changed = _report()
    changed["performance"] = {"post_capture_seconds": 999.9}
    second = deterministic_result_payload(tmp_path, changed)
    assert first["schema"] == DETERMINISTIC_RESULT_SCHEMA
    assert first["result_sha256"] == second["result_sha256"]
    (tmp_path / "video_panorama.png").write_bytes(b"changed")
    assert deterministic_result_payload(tmp_path, _report())["result_sha256"] != first["result_sha256"]


def test_atomic_json_is_canonical_and_leaves_no_pending_file(tmp_path):
    path = tmp_path / "environment.json"
    atomic_write_json(path, {"z": 1, "a": {"b": True}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": {"b": True}, "z": 1}
    assert not (tmp_path / ".environment.json.pending").exists()
