from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from panorama_demo.video_algorithm import candidate_runtime_git_identity, canonical_config_sha256
from panorama_demo.video_candidate_manifest import canonical_candidate_manifest_sha256
from panorama_demo.video_holdout import (
    VideoHoldoutError,
    complete_first_holdout,
    reserve_first_holdout,
    write_user_20m_test_script,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_lock(tmp_path: Path, *, role: str = "candidate") -> Path:
    runtime_head, _ = candidate_runtime_git_identity()
    document: dict[str, object] = {
        "config_schema": "gemini305-video-candidate/v1" if role == "candidate" else "gemini305-video-algorithm/v1",
        "role": role,
        "candidate_id": "C8" if role == "candidate" else None,
        "parent_candidate_id": "C7" if role == "candidate" else None,
        "algorithm_id": "C8",
        "implementation_id": "video_visual_renderer_v2",
        "source_commit": runtime_head if role == "candidate" else "a" * 40,
        "model_sha256": {},
        "allow_baseline_fallback": False,
        "changed_components": ["seam"] if role == "candidate" else None,
        "required_evidence_components": ["orb_anchor_trajectory"] if role == "candidate" else None,
        "required_output_components": ["final_owner"] if role == "candidate" else None,
        "replaces_output_components": [] if role == "candidate" else None,
    }
    document = {key: value for key, value in document.items() if value is not None}
    if role == "candidate":
        document["config_sha256"] = canonical_config_sha256(document)
    config = tmp_path / f"{role}.yaml"
    config.write_text(yaml.safe_dump(document, sort_keys=True), encoding="utf-8")
    if role == "candidate":
        manifest = {
            "schema": "gemini305-video-candidate-manifest/v1",
            "candidates": {
                "C8": {
                    "config_sha256": document["config_sha256"],
                    "required_evidence_components": document["required_evidence_components"],
                    "required_output_components": document["required_output_components"],
                    "replaces_output_components": document["replaces_output_components"],
                }
            },
        }
        manifest["manifest_sha256"] = canonical_candidate_manifest_sha256(manifest)
        (tmp_path / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    lock = tmp_path / f"{role}.lock.json"
    lock.write_text(json.dumps({
        "schema": "gemini305-video-algorithm-lock/v1", "role": role,
        "algorithm_id": "C8", "config_path": config.name,
        "config_sha256": canonical_config_sha256(document),
        "source_commit": runtime_head if role == "candidate" else "a" * 40,
        "model_sha256": {}, "dataset_lock_sha256": "b" * 64 if role == "production" else None,
    }), encoding="utf-8")
    return lock


def test_first_holdout_uses_atomic_ledger_and_cannot_be_repeated(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    dataset = tmp_path / "dataset_lock.json"
    dataset.write_text("{}", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema": "gemini305-video-algorithm-selection/v1",
        "selection_status": "ready_for_first_holdout", "holdout_not_run": True,
        "selected_algorithm_id": "C8",
    }), encoding="utf-8")
    lock = _candidate_lock(tmp_path)
    monkeypatch.setattr("panorama_demo.video_holdout.verify_dataset_lock", lambda *_: None)
    state = tmp_path / "holdout_state.json"

    reserved = reserve_first_holdout(
        session=session, dataset_lock=dataset, selection_path=selection,
        candidate_lock=lock, state_path=state,
    )
    assert reserved["status"] == "reserved"
    assert reserved["first_holdout_consumed"] is True
    with pytest.raises(VideoHoldoutError, match="already been reserved"):
        reserve_first_holdout(
            session=session, dataset_lock=dataset, selection_path=selection,
            candidate_lock=lock, state_path=state,
        )
    completed = complete_first_holdout(
        state, attempt_token=str(reserved["attempt_token"]), passed=False, error="quality gate failed"
    )
    assert completed["status"] == "failed"
    assert completed["first_holdout_pass"] is False
    with pytest.raises(VideoHoldoutError, match="not owned"):
        complete_first_holdout(state, attempt_token=str(reserved["attempt_token"]), passed=True)


def test_holdout_refuses_lock_that_does_not_match_selected_candidate(tmp_path, monkeypatch):
    session = tmp_path / "session"
    session.mkdir()
    dataset = tmp_path / "dataset_lock.json"
    dataset.write_text("{}", encoding="utf-8")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "schema": "gemini305-video-algorithm-selection/v1",
        "selection_status": "ready_for_first_holdout", "holdout_not_run": True,
        "selected_algorithm_id": "wrong",
    }), encoding="utf-8")
    monkeypatch.setattr("panorama_demo.video_holdout.verify_dataset_lock", lambda *_: None)
    with pytest.raises(VideoHoldoutError, match="does not match"):
        reserve_first_holdout(
            session=session, dataset_lock=dataset, selection_path=selection,
            candidate_lock=_candidate_lock(tmp_path), state_path=tmp_path / "holdout_state.json",
        )


def test_user_20m_script_requires_valid_production_lock(tmp_path):
    target = tmp_path / "user_20m_test.ps1"
    with pytest.raises(Exception, match="does not exist"):
        write_user_20m_test_script(target, production_lock=tmp_path / "missing.lock.json")

    lock = _candidate_lock(tmp_path, role="production")
    written = write_user_20m_test_script(target, production_lock=lock)
    script = written.read_text(encoding="utf-8")
    assert _hash(lock) in script
    assert "--maximum-post-seconds 60" in script
    assert "does not claim the result in advance" in script
