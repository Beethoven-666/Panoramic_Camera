from __future__ import annotations

import json
from pathlib import Path

import pytest

from panorama_demo.video_s13_bundle import (
    atomic_write_json,
    seal_p0,
    seal_stage,
    sha256_file,
    update_current_latest,
    update_current_reviewed,
    verify_stage,
)


P0_SCHEMA = "gemini305-video-s13-p0-completion/v1"
P1_SCHEMA = "gemini305-video-s13-p1-vertical-completion/v2"
P2_SCHEMA = "gemini305-video-s13-p2-completion/v2"


def _asset(stage: Path, name: str, value: bytes) -> None:
    stage.mkdir(parents=True, exist_ok=True)
    (stage / name).write_bytes(value)


def test_current_latest_advances_by_seal_and_hard_audit_only(tmp_path: Path) -> None:
    root = tmp_path / "out"
    generation = root / "generations/g1"
    p0 = generation / "P0"
    _asset(p0, "base.png", b"p0")
    seal_p0(p0, generation_id="g1", metadata={"result_asset": "base.png"})
    latest0 = update_current_latest(
        root, generation, stage="P0", completion_name="P0_completion.json",
        completion_schema=P0_SCHEMA, result_asset="base.png",
    )
    assert latest0["stage"] == "P0"

    p0_sha = sha256_file(p0 / "P0_completion.json")
    p1 = generation / "P1"
    _asset(p1, "vertical.png", b"p1")
    seal_stage(
        p1, completion_name="P1_completion.json", schema=P1_SCHEMA,
        metadata={
            "generation_id": "g1", "stage": "P1", "hard_audit_passed": True,
            "p0_parent_sha256": p0_sha, "result_asset": "vertical.png",
            "result_asset_sha256": sha256_file(p1 / "vertical.png"),
        },
    )
    latest1 = update_current_latest(
        root, generation, stage="P1", completion_name="P1_completion.json",
        completion_schema=P1_SCHEMA, result_asset="vertical.png",
    )
    assert latest1["stage"] == "P1"
    assert latest1["parent_stage"] == "P0"
    assert latest1["parent_completion_sha256"] == p0_sha

    p1_sha = sha256_file(p1 / "P1_completion.json")
    p2 = generation / "P2"
    _asset(p2, "final.png", b"p2")
    seal_stage(
        p2, completion_name="P2_completion.json", schema=P2_SCHEMA,
        metadata={
            "generation_id": "g1", "stage": "P2", "hard_audit_passed": True,
            "parent_stage": "P1", "parent_completion_sha256": p1_sha,
            "result_asset": "final.png", "result_asset_sha256": sha256_file(p2 / "final.png"),
            "selected_as_best": False,
        },
    )
    latest2 = update_current_latest(
        root, generation, stage="P2", completion_name="P2_completion.json",
        completion_schema=P2_SCHEMA, result_asset="final.png",
    )
    assert latest2["stage"] == "P2"
    assert latest2["parent_completion_sha256"] == p1_sha
    assert json.loads((root / "current_latest.json").read_text(encoding="utf-8")) == latest2


def test_bad_stage_or_result_hash_preserves_current_latest(tmp_path: Path) -> None:
    root = tmp_path / "out"
    generation = root / "generations/g1"
    p0 = generation / "P0"
    _asset(p0, "base.png", b"good")
    seal_p0(p0, generation_id="g1", metadata={})
    before = update_current_latest(
        root, generation, stage="P0", completion_name="P0_completion.json",
        completion_schema=P0_SCHEMA, result_asset="base.png",
    )
    (p0 / "base.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        update_current_latest(
            root, generation, stage="P0", completion_name="P0_completion.json",
            completion_schema=P0_SCHEMA, result_asset="base.png",
        )
    assert json.loads((root / "current_latest.json").read_text(encoding="utf-8")) == before


def test_verify_stage_rejects_unsafe_asset_paths(tmp_path: Path) -> None:
    stage = tmp_path / "P1"
    stage.mkdir()
    atomic_write_json(stage / "P1_completion.json", {
        "schema": P1_SCHEMA, "sealed": True,
        "assets_sha256": {"../escape.bin": "0" * 64},
    })
    with pytest.raises(ValueError, match="path is unsafe"):
        verify_stage(stage, completion_name="P1_completion.json", schema=P1_SCHEMA)


def test_pending_generation_cannot_be_published(tmp_path: Path) -> None:
    root = tmp_path / "out"
    generation = root / "generations/.g1.token.pending"
    p0 = generation / "P0"
    _asset(p0, "base.png", b"p0")
    seal_p0(p0, generation_id="g1", metadata={})
    with pytest.raises(ValueError, match="pending"):
        update_current_latest(
            root, generation, stage="P0", completion_name="P0_completion.json",
            completion_schema=P0_SCHEMA, result_asset="base.png",
        )


def test_reviewed_pointer_changes_only_on_explicit_review(tmp_path: Path) -> None:
    root = tmp_path / "out"
    generation = root / "generations/g1"
    p0 = generation / "P0"
    _asset(p0, "base.png", b"p0")
    seal_p0(p0, generation_id="g1", metadata={})
    update_current_latest(
        root, generation, stage="P0", completion_name="P0_completion.json",
        completion_schema=P0_SCHEMA, result_asset="base.png",
    )
    assert not (root / "current_reviewed.json").exists()
    reviewed = update_current_reviewed(
        root, generation, stage="P0", completion_name="P0_completion.json",
        completion_schema=P0_SCHEMA, result_asset="base.png",
        review_method="manual", reviewer="tester", note="visual check",
        write_legacy_preview=True,
    )
    assert reviewed["reviewer"] == "tester"
    preview = json.loads((root / "current_preview.json").read_text(encoding="utf-8"))
    assert preview["deprecated"] is True
    assert preview["runtime_authority"] is False
