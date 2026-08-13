from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import panorama_demo.video_s13_evaluation as evaluation
from panorama_demo.video_s13_evaluation import (
    BRANCHES,
    BUNDLE_SCHEMA,
    CURRENT_REVIEWED_SCHEMA,
    DAG_SCHEMA,
    INTEGRITY_SCHEMA,
    LOCK_SCHEMA,
    REVIEW_RECORD_SCHEMA,
    verify_evaluation_lock,
    verify_review_bundle,
    verify_transaction_dag,
)
from panorama_demo.video_s13_review import review_generation


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


def test_branch_stage_roots_prefers_current_forward_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = tmp_path / "forward" / "generation"
    for stage in ("P0", "P1", "P2"):
        (generation / stage).mkdir(parents=True)
    formal = tmp_path / "acceptance" / "runs_v4" / "fast_direct" / "formal"
    monkeypatch.setattr(
        evaluation,
        "_verify_branch",
        lambda _acceptance, _branch: {
            "generation_id": "new-generation",
            "p2_root": str(generation / "P2"),
            "run_root": str(formal),
        },
    )

    roots = evaluation.branch_stage_roots(tmp_path, tmp_path / "acceptance", "fast_direct")

    assert roots["P0"] == generation / "P0"
    assert roots["P1"] == generation / "P1"
    assert roots["P2"] == (generation / "P2").resolve()


def _dag(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    asset = _write(tmp_path / "dag-asset.json", "{}")
    names = (
        "P0_completion",
        "P1_completion",
        "P1_vertical_solution",
        "P2_completion",
        "P3_photometric_solution",
        "P3_completion",
        "M6_automatic_quality",
        "M6_blocked_report",
        "M6_manual_forward",
        "M7_handoff",
        "M7_preflight",
        "M7_no_winner",
        "M7_known_out_of_scope",
    )
    nodes = []
    for index, name in enumerate(names):
        nodes.append(
            {
                "node_id": name,
                "node_type": "evidence",
                "stage": "M8",
                "asset": str(asset.resolve()),
                "sha256": _sha(asset),
                "parent_node_ids": [] if index == 0 else [names[index - 1]],
                "generation_id": "g1",
                "transaction_id": None,
            }
        )
    document: dict[str, object] = {
        "schema": DAG_SCHEMA,
        "generation_id": "g1",
        "p4_state": "not_run_no_real_winner",
        "expected_counts": {
            "p2_pair_transactions": 0,
            "p2_replays": 0,
            "p3_blend_transactions": 0,
            "m7_components": 0,
        },
        "nodes": nodes,
    }
    path = _write(tmp_path / "transaction_dag.json", document)
    return path, document


def test_dag_rejects_cycle_dangling_mutation_and_missing_required_node(
    tmp_path: Path,
) -> None:
    path, document = _dag(tmp_path)
    verify_transaction_dag(path)
    nodes = document["nodes"]
    assert isinstance(nodes, list)
    nodes[0]["parent_node_ids"] = ["missing"]
    _write(path, document)
    with pytest.raises(ValueError, match="dangling"):
        verify_transaction_dag(path)
    nodes[0]["parent_node_ids"] = [nodes[-1]["node_id"]]
    _write(path, document)
    with pytest.raises(ValueError, match="cycle"):
        verify_transaction_dag(path)
    nodes[0]["parent_node_ids"] = []
    removed = nodes.pop()
    _write(path, document)
    with pytest.raises(ValueError, match="required evidence"):
        verify_transaction_dag(path)
    nodes.append(removed)
    _write(path, document)
    Path(nodes[0]["asset"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="asset changed"):
        verify_transaction_dag(path)


def test_review_bundle_verifies_every_manifest_asset(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    dag, _ = _dag(bundle)
    _write(
        bundle / "stage_integrity.json",
        {
            "schema": INTEGRITY_SCHEMA,
            "passed": True,
            "no_new_canonical_pixels": True,
            "production_eligible": False,
        },
    )
    payload = _write(bundle / "full/P3.png", b"pixels")
    assets = {
        item.relative_to(bundle).as_posix(): _sha(item)
        for item in bundle.rglob("*")
        if item.is_file()
    }
    _write(
        bundle / "bundle_manifest.json",
        {"schema": BUNDLE_SCHEMA, "assets_sha256": assets},
    )
    verify_review_bundle(bundle)
    payload.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="asset changed"):
        verify_review_bundle(bundle)


def test_review_rejects_empty_reviewer_and_unknown_method(tmp_path: Path) -> None:
    for reviewer, method in (("", "manual"), ("reviewer", "frozen_policy")):
        with pytest.raises(ValueError, match="review method"):
            review_generation(
                tmp_path,
                tmp_path,
                "fast_direct",
                stage="P3",
                reviewer=reviewer,
                review_method=method,
                note="",
            )


@pytest.mark.parametrize("stage", ("P2", "P3", "P4"))
def test_review_pointer_supports_all_contract_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    bundle = tmp_path / "evaluation/run"
    _write(
        bundle / "review_form.json",
        {"stage_conclusions": {stage: {"decision": "accepted"}}},
    )
    manifest = _write(bundle / "bundle_manifest.json", {"schema": BUNDLE_SCHEMA})
    completion = _write(tmp_path / f"{stage}_completion.json", "completion")
    result = _write(tmp_path / f"{stage}.png", b"pixels")
    audit = _write(tmp_path / f"{stage}_audit.json", "audit")
    binding = {
        "generation_id": "g1",
        "completion": str(completion.resolve()),
        "completion_sha256": _sha(completion),
        "result": str(result.resolve()),
        "result_sha256": _sha(result),
        "hard_audit": str(audit.resolve()),
        "hard_audit_sha256": _sha(audit),
    }
    monkeypatch.setattr(
        "panorama_demo.video_s13_review.verify_review_bundle", lambda _path: {}
    )
    monkeypatch.setattr(
        "panorama_demo.video_s13_review.review_stage_binding",
        lambda *_args: binding,
    )
    pointer = review_generation(
        tmp_path,
        bundle,
        "fast_direct",
        stage=stage,
        reviewer="reviewer",
        review_method="offline_dataset",
        note="accepted",
    )
    assert pointer["schema"] == CURRENT_REVIEWED_SCHEMA
    assert pointer["stage"] == stage
    assert pointer["reviewed_at_utc"]
    assert pointer["bundle_manifest_sha256"] == _sha(manifest)
    record = json.loads((bundle / "review_record.json").read_text(encoding="utf-8"))
    assert record["schema"] == REVIEW_RECORD_SCHEMA


def _minimal_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], dict[str, object]]:
    repo = tmp_path / "repo"
    package_lock = _write(repo / "pyproject.toml", "lock")
    snapshot: dict[str, object] = {
        "repository": str(repo.resolve()),
        "branch": "test",
        "commit": "abc",
        "dirty": False,
        "tracked_diff_sha256": "clean",
        "relevant_worktree_files": {},
        "package_lock": {"path": "pyproject.toml", "sha256": _sha(package_lock)},
        "environment": {"python": "test"},
    }
    monkeypatch.setattr(
        "panorama_demo.video_s13_evaluation.code_snapshot", lambda _repo: snapshot
    )
    monkeypatch.setattr(
        "panorama_demo.video_s13_evaluation.verify_review_bundle", lambda _path: {}
    )
    config = _write(repo / "config.yaml", "config")
    manifest = _write(repo / "manifest.json", "manifest")
    evidence = {
        name: _binding(_write(tmp_path / "evidence" / f"{name}.json", name))
        for name in (
            "blocked_report",
            "manual_forward_authorization",
            "m7_handoff",
            "m7_preflight",
            "m7_core_comparisons",
            "m7_validation_summary",
            "known_limitations",
            "validation_summary",
            "test_summary",
        )
    }
    runs = []
    for branch in BRANCHES:
        base = tmp_path / branch
        review_record = _write(
            base / "review_record.json",
            {
                "schema": REVIEW_RECORD_SCHEMA,
                "stage": "P3",
                "generation_id": "g1",
                "review_method": "manual",
                "reviewer": "reviewer",
                "reviewed_at_utc": "2026-01-01T00:00:00Z",
                "completion_sha256": "completion",
                "result_sha256": "result",
                "hard_audit_sha256": "audit",
            },
        )
        pointer_document = {
            **json.loads(review_record.read_text(encoding="utf-8")),
            "schema": CURRENT_REVIEWED_SCHEMA,
            "review_record_sha256": _sha(review_record),
        }
        pointer = _write(base / "current_reviewed.json", pointer_document)
        bundle_manifest = _write(base / "bundle_manifest.json", "bundle")
        dag = _write(base / "transaction_dag.json", "dag")
        integrity = _write(base / "stage_integrity.json", "integrity")
        completions = {
            stage: _binding(_write(base / f"{stage}_completion.json", stage))
            for stage in ("P0", "P1", "P2", "P3")
        }
        session = base / "session"
        for name in (
            "manifest.json",
            "frames.csv",
            "calibration.json",
            "orbslam3_trajectory.json",
            "rgb.png",
        ):
            _write(session / name, name)
        runs.append(
            {
                "branch": branch,
                "generation_id": "g1",
                "reviewed_stage": "P3",
                "current_reviewed": _binding(pointer),
                "review_record": _binding(review_record),
                "bundle_manifest": _binding(bundle_manifest),
                "transaction_dag": _binding(dag),
                "stage_integrity": _binding(integrity),
                "stage_completions": completions,
                "data": {
                    "session_root": str(session.resolve()),
                    "manifest_sha256": _sha(session / "manifest.json"),
                    "frames_csv_sha256": _sha(session / "frames.csv"),
                    "calibration_sha256": _sha(session / "calibration.json"),
                    "trajectory_cache_sha256": _sha(
                        session / "orbslam3_trajectory.json"
                    ),
                    "rgb_files": [{"path": "rgb.png", "sha256": _sha(session / "rgb.png")}],
                },
            }
        )
    lock: dict[str, object] = {
        "schema": LOCK_SCHEMA,
        "diagnostic_only": True,
        "production_eligible": False,
        "production_lock_eligible": False,
        "code": snapshot,
        "algorithm": {
            "config_path": str(config.resolve()),
            "config_sha256": _sha(config),
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": _sha(manifest),
        },
        "evidence": evidence,
        "runs": runs,
        "execution": {
            "environment_variables": {
                key: value
                for key, value in sorted(os.environ.items())
                if key.startswith("G305_")
            }
        },
    }
    path = _write(tmp_path / "evaluation.lock.json", lock)
    return path, lock, snapshot


def test_lock_detects_code_config_trajectory_and_evidence_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, lock, snapshot = _minimal_lock(tmp_path, monkeypatch)
    verify_evaluation_lock(path)
    snapshot["commit"] = "changed"
    with pytest.raises(ValueError, match="code snapshot"):
        verify_evaluation_lock(path)
    snapshot["commit"] = "abc"
    Path(lock["algorithm"]["config_path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="algorithm"):
        verify_evaluation_lock(path)
    Path(lock["algorithm"]["config_path"]).write_text("config", encoding="utf-8")
    session = Path(lock["runs"][0]["data"]["session_root"])
    (session / "orbslam3_trajectory.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="session input"):
        verify_evaluation_lock(path)
    (session / "orbslam3_trajectory.json").write_text(
        "orbslam3_trajectory.json", encoding="utf-8"
    )
    evidence = Path(lock["evidence"]["test_summary"]["path"])
    evidence.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation evidence"):
        verify_evaluation_lock(path)


def test_lock_rejects_incomplete_four_run_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, lock, _ = _minimal_lock(tmp_path, monkeypatch)
    lock["runs"].pop()
    _write(path, lock)
    with pytest.raises(ValueError, match="contract invalid"):
        verify_evaluation_lock(path)
