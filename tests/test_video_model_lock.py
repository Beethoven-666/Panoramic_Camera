from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import panorama_demo.video_model_lock as model_lock
from panorama_demo.video_model_lock import VideoModelLockError, verify_candidate_models


ROOT = Path(__file__).resolve().parents[1]
RAFT_ID = "torchvision_raft_small_C_T_V2"
RAFT_SHA = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"


def test_raft_manifest_matches_the_candidate_declarations_without_requiring_weights():
    locks = verify_candidate_models({RAFT_ID: RAFT_SHA}, require_files=False)
    assert len(locks) == 1
    assert locks[0].candidate_only is True
    assert locks[0].license == "BSD-3-Clause"


def test_model_verification_rejects_changed_local_bytes(monkeypatch, tmp_path):
    weights = tmp_path / "model.pth"
    weights.write_bytes(b"expected bytes")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest_dir = tmp_path / "configs" / "video_algorithms"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / f"{RAFT_ID}.model.json"
    manifest.write_text(
        json.dumps({
            "schema": model_lock.VIDEO_MODEL_LOCK_SCHEMA,
            "model_id": RAFT_ID,
            "file": str(weights),
            "sha256": digest,
            "implementation": "test.raft_small",
            "license": "BSD-3-Clause",
            "license_url": "https://example.invalid/license",
            "candidate_only": True,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(model_lock, "PROJECT_ROOT", tmp_path)
    verify_candidate_models({RAFT_ID: digest})
    weights.write_bytes(b"changed bytes")
    with pytest.raises(VideoModelLockError, match="file SHA-256 mismatch"):
        verify_candidate_models({RAFT_ID: digest})
