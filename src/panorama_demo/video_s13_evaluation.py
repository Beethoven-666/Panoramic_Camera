"""Read-only S013 M8 evidence graph, review bundles, and evaluation locks."""

from __future__ import annotations

import hashlib
import json
import csv
import importlib.metadata
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .video_s13_bundle import (
    atomic_write_json,
    sha256_file,
    verify_p0_completion,
    verify_stage,
)
from .video_s13_m7 import BRANCHES, P2_COMPLETION_SCHEMA, _verify_branch
from .video_s13_p4_hard_audit import P4_COMPLETION_SCHEMA

DAG_SCHEMA = "gemini305-video-s13-transaction-dag/v1"
INTEGRITY_SCHEMA = "gemini305-video-s13-stage-integrity/v1"
BUNDLE_SCHEMA = "gemini305-video-s13-review-bundle/v1"
LOCK_SCHEMA = "gemini305-video-s13-evaluation-lock/v1"
P1_SCHEMA = "gemini305-video-s13-p1-vertical-completion/v2"
P3_SCHEMA = "gemini305-video-s13-p3-visual-completion/v2"
REVIEW_RECORD_SCHEMA = "gemini305-video-s13-review-record/v1"
CURRENT_REVIEWED_SCHEMA = "gemini305-video-s13-current-reviewed/v2"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"M8 JSON root must be an object: {path}")
    return value


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _node(
    node_id: str,
    node_type: str,
    stage: str,
    asset: Path,
    parents: list[str],
    generation_id: str,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "node_type": node_type,
        "stage": stage,
        "asset": str(asset.resolve()),
        "sha256": sha256_file(asset),
        "parent_node_ids": parents,
        "generation_id": generation_id,
        "transaction_id": node_id if "transaction" in node_type else None,
    }


def branch_stage_roots(repo: Path, acceptance: Path, branch: str) -> dict[str, Path]:
    binding = _verify_branch(acceptance, branch)
    generation_id = binding["generation_id"]
    legacy_original = (
        repo / "artifacts/S013_M6_acceptance" / branch / "generations" / generation_id
    )
    p2 = Path(binding["p2_root"]).resolve()
    forward_generation = p2.parent
    original = (
        forward_generation
        if (forward_generation / "P0").is_dir() and (forward_generation / "P1").is_dir()
        else legacy_original
    )
    return {
        "P0": original / "P0",
        "P1": original / "P1",
        "P2": p2,
        "P3": Path(binding["run_root"]) / "P3",
        "P4": Path(binding["run_root"]) / "P4",
    }


def _sha_binding(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"M8 bound asset missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _verify_sha_binding(binding: dict[str, Any], label: str) -> None:
    path = Path(str(binding.get("path", "")))
    if not path.is_file() or sha256_file(path) != binding.get("sha256"):
        raise ValueError(f"M8 {label} changed")


def review_stage_binding(
    acceptance: Path, branch: str, stage: str
) -> dict[str, str]:
    repo = Path(__file__).resolve().parents[2]
    roots = branch_stage_roots(repo, acceptance, branch)
    binding = _verify_branch(acceptance, branch)
    if stage == "P2":
        completion_name = "P2_completion.json"
        completion = verify_stage(
            roots[stage], completion_name=completion_name, schema=P2_COMPLETION_SCHEMA
        )
        result_name = str(completion["result_asset"])
        result_sha = str(completion["result_asset_sha256"])
        hard_audit = roots[stage] / "hard_audit.json"
    elif stage == "P3":
        completion_name = "P3_completion.json"
        completion = _json(roots[stage] / completion_name)
        if (
            completion.get("schema") != P3_SCHEMA
            or completion.get("generation_id") != binding["generation_id"]
        ):
            raise ValueError("M8 P3 completion contract invalid")
        result_name = "visual_panorama.png"
        result_sha = str(binding["p3_result_sha256"])
        hard_audit = roots[stage] / "hard_audit.json"
    elif stage == "P4":
        completion_name = "P4_completion.json"
        completion = verify_stage(
            roots[stage], completion_name=completion_name, schema=P4_COMPLETION_SCHEMA
        )
        result_name = str(completion["result_asset"])
        result_sha = str(completion["result_asset_sha256"])
        hard_audit = roots[stage] / "hard_audit.json"
    else:
        raise ValueError("M8 reviewed stage must be P2, P3, or P4")
    completion_path = roots[stage] / completion_name
    result = roots[stage] / result_name
    if sha256_file(result) != result_sha:
        raise ValueError(f"M8 {stage} result SHA mismatch")
    audit = _json(hard_audit)
    passed = audit.get("passed", audit.get("hard_audit_passed"))
    if passed is not True:
        raise ValueError(f"M8 {stage} hard audit is not true")
    return {
        "generation_id": str(binding["generation_id"]),
        "completion": str(completion_path.resolve()),
        "completion_sha256": sha256_file(completion_path),
        "result": str(result.resolve()),
        "result_sha256": result_sha,
        "hard_audit": str(hard_audit.resolve()),
        "hard_audit_sha256": sha256_file(hard_audit),
    }


def verify_review_bundle(path: Path) -> dict[str, Any]:
    manifest_path = path / "bundle_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("M8 review bundle schema invalid")
    expected = manifest.get("assets_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("M8 review bundle asset manifest missing")
    actual_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file()
        and item.name not in {"bundle_manifest.json", "review_record.json"}
    }
    if actual_files != set(expected):
        raise ValueError("M8 review bundle asset set changed")
    for relative, expected_sha in expected.items():
        asset = path / relative
        if sha256_file(asset) != expected_sha:
            raise ValueError(f"M8 review bundle asset changed: {relative}")
    integrity = _json(path / "stage_integrity.json")
    if (
        integrity.get("schema") != INTEGRITY_SCHEMA
        or integrity.get("passed") is not True
        or integrity.get("no_new_canonical_pixels") is not True
        or integrity.get("production_eligible") is not False
    ):
        raise ValueError("M8 stage integrity invalid")
    verify_transaction_dag(path / "transaction_dag.json")
    return manifest


def build_transaction_dag(
    repo: Path, acceptance: Path, branch: str, output: Path
) -> dict[str, Any]:
    roots = branch_stage_roots(repo, acceptance, branch)
    verify_p0_completion(roots["P0"])
    verify_stage(roots["P1"], completion_name="P1_completion.json", schema=P1_SCHEMA)
    p2 = verify_stage(
        roots["P2"], completion_name="P2_completion.json", schema=P2_COMPLETION_SCHEMA
    )
    binding = _verify_branch(acceptance, branch)
    gid = binding["generation_id"]
    p3c = roots["P3"] / "P3_completion.json"
    nodes = [
        _node(
            "P0_completion",
            "completion",
            "P0",
            roots["P0"] / "P0_completion.json",
            [],
            gid,
        ),
        _node(
            "P1_completion",
            "completion",
            "P1",
            roots["P1"] / "P1_completion.json",
            ["P0_completion"],
            gid,
        ),
        _node(
            "P1_vertical_solution",
            "solution",
            "P1",
            roots["P1"] / "vertical_solution.json",
            ["P1_completion"],
            gid,
        ),
        _node(
            "P2_completion",
            "completion",
            "P2",
            roots["P2"] / "P2_completion.json",
            ["P1_completion"],
            gid,
        ),
    ]
    tx = _json(roots["P2"] / "pair_transactions.json").get("pairs", [])
    replay = _json(roots["P2"] / "p2_replay_manifest.json").get("pairs", [])
    for i, item in enumerate(tx):
        asset = roots["P2"] / "pair_transactions" / f"pair_{i:04d}.json"
        nodes.append(
            _node(
                f"P2_pair_transaction_{i:04d}",
                "pair_transaction",
                "P2",
                asset,
                ["P2_completion"],
                gid,
            )
        )
    for i, item in enumerate(replay):
        asset = roots["P2"] / str(item["asset"])
        nodes.append(
            _node(
                f"P2_replay_{i:04d}",
                "pair_replay",
                "P2",
                asset,
                [f"P2_pair_transaction_{i:04d}"],
                gid,
            )
        )
    nodes.append(
        _node(
            "P3_photometric_solution",
            "solution",
            "P3",
            roots["P3"] / "photometric_solution.json",
            ["P2_completion"],
            gid,
        )
    )
    blends = _json(roots["P3"] / "blend_transactions.json").get("pairs", [])
    for i, item in enumerate(blends):
        asset = roots["P3"] / str(item["asset"])
        nodes.append(
            _node(
                f"P3_blend_transaction_{i:04d}",
                "blend_transaction",
                "P3",
                asset,
                ["P3_photometric_solution", f"P2_pair_transaction_{i:04d}"],
                gid,
            )
        )
    blend_ids = [f"P3_blend_transaction_{i:04d}" for i in range(len(blends))]
    nodes.append(
        _node(
            "P3_completion",
            "completion",
            "P3",
            p3c,
            ["P3_photometric_solution", *blend_ids],
            gid,
        )
    )
    quality = Path(binding["run_root"]) / "post_quality_final/post_quality.json"
    nodes.append(
        _node(
            "M6_automatic_quality",
            "automatic_quality",
            "M6",
            quality,
            ["P3_completion"],
            gid,
        )
    )
    blocked = acceptance / "blocked_report.json"
    auth = acceptance / "M6_manual_forward_authorization.json"
    handoff = acceptance / "M7_handoff.json"
    preflight = acceptance / "M7_preflight.json"
    core = acceptance / "M7_core_comparisons.json"
    summary = acceptance / "M7_validation_summary.json"
    known = acceptance / "M7_known_out_of_scope.json"
    nodes += [
        _node(
            "M6_blocked_report",
            "automatic_blocked",
            "M6",
            blocked,
            ["M6_automatic_quality"],
            gid,
        ),
        _node(
            "M6_manual_forward",
            "manual_authorization",
            "M6",
            auth,
            ["M6_blocked_report", "P3_completion"],
            gid,
        ),
        _node("M7_handoff", "handoff", "M7", handoff, ["M6_manual_forward"], gid),
        _node("M7_preflight", "preflight", "M7", preflight, ["M7_handoff"], gid),
    ]
    branch_components = [
        item for item in _json(core).get("components", []) if item.get("branch") == branch
    ]
    component_ids: list[str] = []
    component_dir = output.parent / "m7_component_transactions"
    component_dir.mkdir(parents=True, exist_ok=True)
    for item in branch_components:
        item_id = str(item["handoff_item_id"])
        asset = component_dir / f"{item_id}.json"
        atomic_write_json(
            asset,
            {
                "schema": "gemini305-video-s13-m7-component-transaction/v1",
                "source_summary": str(core.resolve()),
                "source_summary_sha256": sha256_file(core),
                "component": item,
            },
        )
        node_id = f"M7_component_{item_id}"
        component_ids.append(node_id)
        nodes.append(
            _node(node_id, "component_transaction", "M7", asset, ["M7_preflight"], gid)
        )
    nodes += [
        _node(
            "M7_no_winner",
            "no_winner",
            "M7",
            summary,
            ["M7_preflight", *component_ids],
            gid,
        ),
        _node(
            "M7_known_out_of_scope",
            "known_limitations",
            "M7",
            known,
            ["M7_no_winner"],
            gid,
        ),
    ]
    payload = {
        "schema": DAG_SCHEMA,
        "branch": branch,
        "generation_id": gid,
        "created_at_utc": _utc(),
        "p4_state": "not_run_no_real_winner",
        "nodes": nodes,
        "expected_counts": {
            "p2_pair_transactions": int(p2["pair_transaction_count"]),
            "p2_replays": int(p2["p2_replay_pair_count"]),
            "p3_blend_transactions": len(blends),
            "m7_components": len(branch_components),
        },
    }
    atomic_write_json(output, payload)
    verify_transaction_dag(output)
    return payload


def verify_transaction_dag(path: Path) -> dict[str, Any]:
    dag = _json(path)
    if (
        dag.get("schema") != DAG_SCHEMA
        or dag.get("p4_state") != "not_run_no_real_winner"
    ):
        raise ValueError("M8 DAG schema or P4 state is invalid")
    nodes = dag.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("M8 DAG nodes missing")
    by_id = {n.get("node_id"): n for n in nodes}
    if len(by_id) != len(nodes) or None in by_id:
        raise ValueError("M8 DAG node IDs invalid")
    for node in nodes:
        required = {
            "node_id",
            "node_type",
            "stage",
            "asset",
            "sha256",
            "parent_node_ids",
            "generation_id",
            "transaction_id",
        }
        if not required.issubset(node):
            raise ValueError("M8 DAG node contract incomplete")
        if node["generation_id"] != dag.get("generation_id"):
            raise ValueError("M8 DAG generation binding mismatch")
        asset = Path(str(node.get("asset", "")))
        if not asset.is_file() or sha256_file(asset) != node.get("sha256"):
            raise ValueError(f"M8 DAG asset changed: {asset}")
        if any(parent not in by_id for parent in node.get("parent_node_ids", [])):
            raise ValueError("M8 DAG has dangling parent")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(n: str) -> None:
        if n in visiting:
            raise ValueError("M8 DAG contains a cycle")
        if n in visited:
            return
        visiting.add(n)
        for p in by_id[n].get("parent_node_ids", []):
            visit(p)
        visiting.remove(n)
        visited.add(n)

    for n in by_id:
        visit(n)
    counts = dag["expected_counts"]
    for typ, key in (
        ("pair_transaction", "p2_pair_transactions"),
        ("pair_replay", "p2_replays"),
        ("blend_transaction", "p3_blend_transactions"),
        ("component_transaction", "m7_components"),
    ):
        if sum(n["node_type"] == typ for n in nodes) != counts[key]:
            raise ValueError(f"M8 DAG {typ} count mismatch")
    if any(n["stage"] == "P4" for n in nodes):
        raise ValueError("M8 DAG invents P4")
    required_nodes = {
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
    }
    if not required_nodes.issubset(by_id):
        raise ValueError("M8 DAG required evidence node missing")
    return dag


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_review_bundle(
    repo: Path, acceptance: Path, evaluation_id: str, branch: str
) -> Path:
    roots = branch_stage_roots(repo, acceptance, branch)
    binding = _verify_branch(acceptance, branch)
    out = acceptance / "evaluation" / evaluation_id / branch
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    images = {
        "P0": roots["P0"] / "base_panorama_owner_only.png",
        "P1": roots["P1"] / "vertical_panorama_owner_only.png",
        "P2": roots["P2"] / "geometry_and_seam_panorama_owner_only.png",
        "P3": roots["P3"] / "visual_panorama.png",
    }
    for stage, src in images.items():
        _copy(src, out / "full" / f"{stage}.png")
    atomic_write_json(
        out / "full/P4.not_run.json",
        {"stage": "P4", "state": "not_run", "reason": "M7 produced no real winner"},
    )
    for name in (
        "contact_sheets",
        "fixed_crops",
        "worst_geometry_crops",
        "worst_visual_crops",
        "worst_repair_crops",
        "owner_overlays",
        "seam_overlays",
        "protected_masks",
        "blend_masks",
        "metric_tables",
    ):
        (out / name).mkdir(exist_ok=True)
    thumbs = []
    for stage, src in images.items():
        im = cv2.imread(str(src))
        scale = min(1.0, 480 / max(im.shape[:2]))
        thumbs.append(cv2.resize(im, None, fx=scale, fy=scale))
    h = max(i.shape[0] for i in thumbs)
    thumbs = [
        cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for i in thumbs
    ]
    cv2.imwrite(str(out / "contact_sheets/P0-P3.png"), cv2.hconcat(thumbs))
    for src, dst in (
        (roots["P0"] / "owner_boundary_overlay.png", out / "owner_overlays/P0.png"),
        (
            roots["P1"] / "vertical_owner_boundary_overlay.png",
            out / "owner_overlays/P1.png",
        ),
        (roots["P2"] / "seam_overlay.png", out / "seam_overlays/P2.png"),
    ):
        if src.is_file():
            _copy(src, dst)
    original_p2 = roots["P0"].parent / "P2" / "worst_seam_crops"
    original_p3 = roots["P0"].parent / "P3" / "worst_visual_crops"
    for src in sorted(original_p2.glob("rank_*.png"))[:10]:
        _copy(src, out / "worst_geometry_crops" / src.name)
    visual = (
        sorted(original_p3.glob("blur_ghost_*.png"))[:10]
        + sorted(original_p3.glob("color_*.png"))[:10]
    )
    for src in visual:
        _copy(src, out / "worst_visual_crops" / src.name)
    for src in sorted(original_p3.glob("high_risk_horizontal_*.png")):
        _copy(src, out / "fixed_crops" / src.name)
    repair_root = acceptance / "M7_handoff_evidence" / branch
    for src in sorted(repair_root.rglob("*_crop.png"))[:10]:
        _copy(src, out / "worst_repair_crops" / (src.parent.name + "_" + src.name))
    with np.load(roots["P3"] / "pair_masks.npz", allow_pickle=False) as masks:
        cv2.imwrite(
            str(out / "protected_masks/protected.png"),
            masks["protected"].astype(np.uint8) * 255,
        )
        cv2.imwrite(
            str(out / "blend_masks/active.png"), masks["active"].astype(np.uint8) * 255
        )
        cv2.imwrite(
            str(out / "blend_masks/safe.png"), masks["safe"].astype(np.uint8) * 255
        )
    for src, dst in (
        (
            roots["P2"] / "diagnostic_quality_report.json",
            out / "metric_tables/P2_quality.json",
        ),
        (
            Path(binding["run_root"]) / "post_quality_final/post_quality.json",
            out / "m6_automatic_quality.json",
        ),
        (
            acceptance / "M6_manual_forward_authorization.json",
            out / "M6_manual_forward_authorization.json",
        ),
        (acceptance / "M7_handoff.json", out / "m7_handoff.json"),
        (acceptance / "M7_preflight.json", out / "M7_preflight.json"),
        (
            acceptance / "M7_core_comparisons.json",
            out / "M7_core_comparisons.json",
        ),
        (
            acceptance / "M7_validation_summary.json",
            out / "M7_validation_summary.json",
        ),
        (acceptance / "M7_known_out_of_scope.json", out / "M7_known_out_of_scope.json"),
    ):
        if src.is_file():
            _copy(src, dst)
    dag_path = out / "transaction_dag.json"
    build_transaction_dag(repo, acceptance, branch, dag_path)
    integrity = {
        "schema": INTEGRITY_SCHEMA,
        "branch": branch,
        "generation_id": binding["generation_id"],
        "passed": True,
        "p4": "not_run",
        "verified_dag_sha256": sha256_file(dag_path),
        "no_new_canonical_pixels": True,
        "production_eligible": False,
    }
    atomic_write_json(out / "stage_integrity.json", integrity)
    review = {
        "schema": "gemini305-video-s13-review-form/v1",
        "branch": branch,
        "generation_id": binding["generation_id"],
        "review_protocol": {
            "full_image": ["layout", "holes", "slope", "color"],
            "fixed_structures": [
                "beams",
                "fan",
                "cables",
                "doorframe",
                "hoses",
                "box_edge",
                "white_wall",
                "blades",
                "thin_rods",
            ],
            "worst_sets": {"geometry": 10, "brightness": 10, "blur": 10, "repair": 10},
        },
        "stage_conclusions": {
            "P0": {
                "decision": "not_preferred",
                "reason": "superseded by sealed vertical solution",
            },
            "P1": {"decision": "not_preferred", "reason": "superseded by sealed P2"},
            "P2": {
                "decision": "not_preferred",
                "reason": "P3 is the user-accepted manual-forward stage",
            },
            "P3": {
                "decision": "accepted",
                "reason": "sealed P3 explicitly accepted for manual forward; original automatic blocked evidence remains visible",
            },
            "P4": {"decision": "not_run", "reason": "M7 produced no real winner"},
        },
        "automatic_quality_remains_blocked": True,
        "manual_selection_required": True,
    }
    atomic_write_json(out / "review_form.json", review)
    files = sorted(
        p for p in out.rglob("*") if p.is_file() and p.name != "bundle_manifest.json"
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "evaluation_id": evaluation_id,
        "branch": branch,
        "generation_id": binding["generation_id"],
        "diagnostic_only": True,
        "production_eligible": False,
        "stages_present": ["P0", "P1", "P2", "P3"],
        "stages_not_run": ["P4"],
        "assets_sha256": {p.relative_to(out).as_posix(): sha256_file(p) for p in files},
    }
    atomic_write_json(out / "bundle_manifest.json", manifest)
    return out


def code_snapshot(repo: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, encoding="utf-8", errors="replace"
        ).strip()

    relevant_roots = ("src/", "tests/", "scripts/", "configs/")
    diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            "src",
            "tests",
            "scripts",
            "configs",
            "pyproject.toml",
        ],
        cwd=repo,
    )
    relevant = []
    status = git("status", "--porcelain")
    for line in status.splitlines():
        name = line[3:].split(" -> ")[-1]
        if (
            name.startswith(relevant_roots)
            and (repo / name).is_file()
        ):
            relevant.append(name)
    versions: dict[str, str | None] = {}
    for package in ("numpy", "opencv-python", "cupy-cuda12x", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "repository": str(repo.resolve()),
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "relevant_worktree_files": {n: sha256_file(repo / n) for n in sorted(relevant)},
        "package_lock": {
            "path": "pyproject.toml",
            "sha256": sha256_file(repo / "pyproject.toml"),
        },
        "environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "opencv_version": cv2.__version__,
            "package_versions": versions,
        },
    }


def _data_binding(repo: Path, acceptance: Path, branch: str) -> dict[str, Any]:
    roots = branch_stage_roots(repo, acceptance, branch)
    input_audit = _json(roots["P0"].parent / "input_audit.json")
    report = _json(roots["P0"].parent / "report.json")
    session = Path(str(input_audit["root"]))
    selected = {
        int(item["frame_id"])
        for item in _json(roots["P3"] / "photometric_solution.json")["sources"]
    }
    with (session / "frames.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    rgb_files: list[dict[str, str | int]] = []
    for row in rows:
        frame_id = int(row["frame_id"])
        if frame_id not in selected:
            continue
        relative = row.get("rgb_path") or row.get("color_path")
        if not relative:
            raise ValueError("M8 frames.csv lacks an RGB path")
        asset = session / relative
        rgb_files.append(
            {
                "frame_id": frame_id,
                "path": relative.replace("\\", "/"),
                "sha256": sha256_file(asset),
            }
        )
    trajectory = session / "orbslam3_trajectory.json"
    trajectory_policy = report["trajectory"]["source"]
    return {
        "session_id": session.name,
        "session_root": str(session.resolve()),
        "manifest_sha256": sha256_file(session / "manifest.json"),
        "frames_csv_sha256": sha256_file(session / "frames.csv"),
        "calibration_sha256": sha256_file(session / "calibration.json"),
        "rgb_files": rgb_files,
        "rgb_file_count": len(rgb_files),
        "trajectory_policy": trajectory_policy,
        "ignore_pose": trajectory_policy == "ignore_pose",
        "trajectory_cache_path": str(trajectory.resolve()),
        "trajectory_cache_sha256": sha256_file(trajectory),
        "trajectory_audit_sha256": sha256_file(
            roots["P0"].parent / "trajectory_audit.json"
        ),
    }


def build_evaluation_lock(
    repo: Path,
    acceptance: Path,
    evaluation_id: str,
    test_summary: Path,
    *,
    allow_dirty: bool = False,
    m9_benchmark_completion: Path | None = None,
) -> Path:
    snap = code_snapshot(repo)
    if snap["dirty"] and not allow_dirty:
        raise ValueError(
            "formal M8 evaluation lock requires a clean worktree; use --allow-dirty only for a diagnostic snapshot"
        )
    started = _utc()
    runs = []
    for branch in BRANCHES:
        bundle = acceptance / "evaluation" / evaluation_id / branch
        pointer = acceptance / "runs_v4" / branch / "formal/current_reviewed.json"
        dag = bundle / "transaction_dag.json"
        verify_review_bundle(bundle)
        if not pointer.is_file():
            raise ValueError(f"M8 current_reviewed missing: {branch}")
        pointer_document = _json(pointer)
        if pointer_document.get("schema") != CURRENT_REVIEWED_SCHEMA:
            raise ValueError(f"M8 current_reviewed schema invalid: {branch}")
        review_record = Path(str(pointer_document.get("review_record", "")))
        if (
            not review_record.is_file()
            or sha256_file(review_record)
            != pointer_document.get("review_record_sha256")
        ):
            raise ValueError(f"M8 review record binding invalid: {branch}")
        reviewed = review_stage_binding(
            acceptance, branch, str(pointer_document.get("stage"))
        )
        for key in (
            "generation_id",
            "completion",
            "completion_sha256",
            "result",
            "result_sha256",
            "hard_audit",
            "hard_audit_sha256",
        ):
            if pointer_document.get(key) != reviewed[key]:
                raise ValueError(f"M8 reviewed stage binding changed: {branch}/{key}")
        roots = branch_stage_roots(repo, acceptance, branch)
        stage_completions = {
            "P0": _sha_binding(roots["P0"] / "P0_completion.json"),
            "P1": _sha_binding(roots["P1"] / "P1_completion.json"),
            "P2": _sha_binding(roots["P2"] / "P2_completion.json"),
            "P3": _sha_binding(roots["P3"] / "P3_completion.json"),
        }
        if roots["P4"].is_dir():
            stage_completions["P4"] = _sha_binding(
                roots["P4"] / "P4_completion.json"
            )
        runs.append(
            {
                "branch": branch,
                "generation_id": pointer_document["generation_id"],
                "reviewed_stage": pointer_document["stage"],
                "current_reviewed": _sha_binding(pointer),
                "review_record": _sha_binding(review_record),
                "bundle_manifest": _sha_binding(bundle / "bundle_manifest.json"),
                "transaction_dag": _sha_binding(dag),
                "stage_integrity": _sha_binding(bundle / "stage_integrity.json"),
                "stage_completions": stage_completions,
                "data": _data_binding(repo, acceptance, branch),
            }
        )
    config = (
        repo
        / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4.yaml"
    )
    manifest = repo / "configs/video_candidates/s013/candidate_manifest.json"
    validation_summary = acceptance / "M8_review_validation_summary.json"
    atomic_write_json(
        validation_summary,
        {
            "schema": "gemini305-video-s13-m8-review-validation-summary/v1",
            "evaluation_id": evaluation_id,
            "state": "complete",
            "branches": [
                {
                    "branch": run["branch"],
                    "generation_id": run["generation_id"],
                    "reviewed_stage": run["reviewed_stage"],
                    "transaction_dag_sha256": run["transaction_dag"]["sha256"],
                }
                for run in runs
            ],
            "no_new_pixels": True,
            "current_preview_runtime_authority": False,
            "production_eligible": False,
        },
    )
    evidence_paths = {
        "blocked_report": acceptance / "blocked_report.json",
        "manual_forward_authorization": acceptance
        / "M6_manual_forward_authorization.json",
        "m7_handoff": acceptance / "M7_handoff.json",
        "m7_preflight": acceptance / "M7_preflight.json",
        "m7_core_comparisons": acceptance / "M7_core_comparisons.json",
        "m7_validation_summary": acceptance / "M7_validation_summary.json",
        "known_limitations": acceptance / "M7_known_out_of_scope.json",
        "validation_summary": validation_summary,
        "test_summary": test_summary,
    }
    if m9_benchmark_completion is not None:
        m9_benchmark_completion = m9_benchmark_completion.resolve()
        completion = _json(m9_benchmark_completion)
        if (
            completion.get("schema")
            != "gemini305-video-s13-m9-benchmark-completion/v1"
            or completion.get("exact") is not True
            or completion.get("diagnostic_only") is not True
            or completion.get("production_lock_created") is not False
        ):
            raise ValueError("M9 exact benchmark completion contract invalid")
        for name, digest in completion.get("assets_sha256", {}).items():
            asset = m9_benchmark_completion.parent / str(name)
            if not asset.is_file() or sha256_file(asset) != digest:
                raise ValueError("M9 benchmark asset binding invalid")
        evidence_paths["m9_benchmark_completion"] = m9_benchmark_completion
    lock = {
        "schema": LOCK_SCHEMA,
        "created_at_utc": _utc(),
        "evaluation_id": evaluation_id,
        "evaluation_phase": "M9_exact" if m9_benchmark_completion else "M8",
        "diagnostic_only": True,
        "production_eligible": False,
        "production_lock_eligible": False,
        "dirty_snapshot_explicitly_authorized": bool(snap["dirty"] and allow_dirty),
        "code": snap,
        "algorithm": {
            "config_path": str(config.resolve()),
            "config_sha256": sha256_file(config),
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": sha256_file(manifest),
            "thresholds_frozen": True,
            "source_set_frozen": True,
            "q0_q3_frozen": True,
            "q4_disabled": True,
            "optional_disabled": True,
        },
        "evidence": {
            name: _sha_binding(asset) for name, asset in evidence_paths.items()
        },
        "runs": runs,
        "recommendation": {
            "p3_vs_p2": "P3 selected by explicit manual-forward policy for all four runs; automatic quality remains blocked",
            "p4_vs_p3": "not_applicable_no_real_p4",
            "trajectory_consistency": "four-run review required",
            "fast_slow_consistency": "four-run review required",
            "production_claim": False,
        },
        "execution": {
            "command": "scripts/build_s13_evaluation_lock.py"
            + (" --allow-dirty" if allow_dirty else ""),
            "environment": sys.executable,
            "environment_variables": {
                key: value for key, value in sorted(os.environ.items()) if key.startswith("G305_")
            },
            "started_at_utc": started,
            "ended_at_utc": _utc(),
            "determinism_controls": {
                "new_pixels_generated": False,
                "current_preview_read": False,
                "automatic_latest_selection": False,
                "explicit_review_required": True,
            },
        },
    }
    out = acceptance / "evaluation" / evaluation_id / "evaluation.lock.json"
    atomic_write_json(out, lock)
    verify_evaluation_lock(out)
    return out


def verify_evaluation_lock(path: Path) -> dict[str, Any]:
    lock = _json(path)
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("production_eligible") is not False
        or lock.get("production_lock_eligible") is not False
        or lock.get("diagnostic_only") is not True
        or len(lock.get("runs", [])) != 4
    ):
        raise ValueError("M8 evaluation lock contract invalid")
    repo = Path(lock["code"]["repository"])
    current = code_snapshot(repo)
    for key in ("commit", "branch", "tracked_diff_sha256", "relevant_worktree_files"):
        if current[key] != lock["code"][key]:
            raise ValueError(f"M8 code snapshot changed: {key}")
    for key in ("package_lock", "environment"):
        if current[key] != lock["code"][key]:
            raise ValueError(f"M8 code environment changed: {key}")
    current_env = {
        key: value for key, value in sorted(os.environ.items()) if key.startswith("G305_")
    }
    if current_env != lock.get("execution", {}).get("environment_variables"):
        raise ValueError("M8 execution environment variables changed")
    for key in ("config_path", "manifest_path"):
        if (
            sha256_file(Path(lock["algorithm"][key]))
            != lock["algorithm"][key.replace("path", "sha256")]
        ):
            raise ValueError("M8 algorithm contract changed")
    if {run.get("branch") for run in lock["runs"]} != set(BRANCHES):
        raise ValueError("M8 four-run branch set incomplete")
    required_evidence = {
        "blocked_report",
        "manual_forward_authorization",
        "m7_handoff",
        "m7_preflight",
        "m7_core_comparisons",
        "m7_validation_summary",
        "known_limitations",
        "validation_summary",
        "test_summary",
    }
    if set(lock.get("evidence", {})) != required_evidence:
        if not (
            lock.get("evaluation_phase") == "M9_exact"
            and set(lock.get("evidence", {}))
            == required_evidence | {"m9_benchmark_completion"}
        ):
            raise ValueError("M8/M9 evaluation evidence set incomplete")
    for binding in lock["evidence"].values():
        _verify_sha_binding(binding, "evaluation evidence")
    if lock.get("evaluation_phase") == "M9_exact":
        benchmark = Path(lock["evidence"]["m9_benchmark_completion"]["path"])
        completion = _json(benchmark)
        if (
            completion.get("schema")
            != "gemini305-video-s13-m9-benchmark-completion/v1"
            or completion.get("exact") is not True
            or completion.get("diagnostic_only") is not True
            or completion.get("production_lock_created") is not False
        ):
            raise ValueError("M9 exact benchmark completion changed")
        for name, digest in completion.get("assets_sha256", {}).items():
            asset = benchmark.parent / str(name)
            if not asset.is_file() or sha256_file(asset) != digest:
                raise ValueError("M9 benchmark asset changed")
    for run in lock["runs"]:
        for key in (
            "current_reviewed",
            "review_record",
            "bundle_manifest",
            "transaction_dag",
            "stage_integrity",
        ):
            _verify_sha_binding(run[key], f"{run['branch']} {key}")
        bundle = Path(run["bundle_manifest"]["path"]).parent
        verify_review_bundle(bundle)
        if sha256_file(bundle / "transaction_dag.json") != run["transaction_dag"]["sha256"]:
            raise ValueError("M8 transaction DAG lock binding changed")
        pointer = _json(Path(run["current_reviewed"]["path"]))
        record = _json(Path(run["review_record"]["path"]))
        if (
            pointer.get("schema") != CURRENT_REVIEWED_SCHEMA
            or record.get("schema") != REVIEW_RECORD_SCHEMA
            or pointer.get("review_record_sha256") != run["review_record"]["sha256"]
            or pointer.get("stage") != run.get("reviewed_stage")
            or pointer.get("generation_id") != run.get("generation_id")
            or record.get("review_method") not in {"manual", "offline_dataset"}
            or not str(record.get("reviewer", "")).strip()
            or not str(record.get("reviewed_at_utc", "")).strip()
        ):
            raise ValueError("M8 reviewed pointer or record contract invalid")
        if set(run.get("stage_completions", {})) not in (
            {"P0", "P1", "P2", "P3"},
            {"P0", "P1", "P2", "P3", "P4"},
        ):
            raise ValueError("M8 stage completion set incomplete")
        for key in (
            "completion_sha256",
            "result_sha256",
            "hard_audit_sha256",
            "review_method",
            "reviewer",
            "reviewed_at_utc",
        ):
            if pointer.get(key) != record.get(key):
                raise ValueError(f"M8 pointer/review record mismatch: {key}")
        for stage_binding in run.get("stage_completions", {}).values():
            _verify_sha_binding(stage_binding, "stage completion")
        data = run["data"]
        session = Path(data["session_root"])
        for name, key in (
            ("manifest.json", "manifest_sha256"),
            ("frames.csv", "frames_csv_sha256"),
            ("calibration.json", "calibration_sha256"),
            ("orbslam3_trajectory.json", "trajectory_cache_sha256"),
        ):
            if sha256_file(session / name) != data[key]:
                raise ValueError(f"M8 session input changed: {name}")
        for item in data["rgb_files"]:
            if sha256_file(session / item["path"]) != item["sha256"]:
                raise ValueError("M8 selected RGB source changed")
    return lock
