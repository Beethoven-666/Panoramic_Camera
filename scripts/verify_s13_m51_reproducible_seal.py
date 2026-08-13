"""Verify and summarize two four-branch S013 M5.1-r2 P2-only runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BRANCHES = ("fast_direct", "fast_ignore_pose", "slow_direct", "slow_ignore_pose")
P2_SCHEMA = "gemini305-video-s13-p2-completion/v5"
SEAL_SCHEMA = "gemini305-video-s13-m51-reproducible-seal/v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generation_p2(run: Path) -> tuple[Path, dict[str, Any]]:
    pointer = _read(run / "current_latest.json")
    if pointer.get("stage") != "P2":
        raise ValueError(f"current_latest is not P2: {run}")
    generation = run / str(pointer["generation"])
    for forbidden in ("P3", "M6", "delivery.json", "video_delivery.json"):
        if (generation / forbidden).exists() or (run / forbidden).exists():
            raise ValueError(f"forbidden post-P2 output exists: {run / forbidden}")
    completion = _read(generation / "P2" / "P2_completion.json")
    if (
        completion.get("schema") != P2_SCHEMA
        or completion.get("m6_eligible") is not False
        or completion.get("stage") != "P2"
    ):
        raise ValueError(f"P2-only completion contract changed: {generation}")
    return generation / "P2", pointer


def _edge_decision(value: object) -> dict[str, object]:
    edge = value if isinstance(value, Mapping) else {}
    return {
        key: edge.get(key)
        for key in (
            "evaluable",
            "reason",
            "supported_block_count",
            "supported_component_count",
            "multiple_layer_or_ambiguous",
            "ambiguous",
            "visual_suspect",
        )
    }


def _candidate_decision(value: object) -> dict[str, object]:
    row = value if isinstance(value, Mapping) else {}
    return {
        key: row.get(key)
        for key in (
            "candidate_id",
            "model_code",
            "model_name",
            "generation_status",
            "evaluation_status",
            "seam_sha256",
            "geometry_model",
            "map_delta_sha256",
            "visual_suspect",
            "hard_gate_passed",
            "hard_gate_failures",
        )
    }


def _transaction_decision(value: Mapping[str, object]) -> dict[str, object]:
    evidence = value.get("seam_local_evidence")
    evidence_row = evidence if isinstance(evidence, Mapping) else {}
    return {
        key: value.get(key)
        for key in (
            "schema",
            "transaction_id",
            "pair_frame_ids",
            "selection_policy",
            "selected_candidate_id",
            "selected_seam_model",
            "selected_geometry_model",
            "selected_seam_rank",
            "selected_geometry_rank",
            "pair_hard_audit_passed",
            "fallback_used",
            "fallback_reason",
            "seam_fallback_used",
            "geometry_fallback_used",
            "topology_repair_used",
            "decision",
            "support_mask_sha256",
            "map_delta_sha256",
            "geometry_comparison_seam_sha256",
            "selected_seam_sha256",
            "result_seam_sha256",
            "selection_continued_for_structure",
            "unresolved_oblique_structure",
            "unexpected_exception_fallback",
            "micro_rescue",
        )
    } | {
        "candidate_generation": value.get("candidate_generation", []),
        "candidate_evaluations": [
            _candidate_decision(item)
            for item in value.get("candidate_evaluations", [])
        ],
        "seam_local_evidence": {
            key: evidence_row.get(key)
            for key in (
                "mode",
                "counts_by_half_width_px",
                "selected_count",
                "selected_half_width_px",
                "sufficient_for_train_held_out",
                "full_shoulder_fallback_used",
            )
        },
        "edge_registration_decision": _edge_decision(value.get("edge_registration")),
    }


def _decision_payload(p2: Path) -> list[dict[str, object]]:
    manifest = _read(p2 / "pair_transactions.json")
    rows = manifest.get("pairs")
    if not isinstance(rows, list):
        raise ValueError(f"pair transaction manifest is invalid: {p2}")
    return [_transaction_decision(row) for row in rows if isinstance(row, Mapping)]


def _npz_equal(left: Path, right: Path, *, omit: frozenset[str] = frozenset()) -> bool:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        keys_a = set(a.files) - omit
        keys_b = set(b.files) - omit
        return keys_a == keys_b and all(
            np.array_equal(
                a[key],
                b[key],
                equal_nan=(a[key].dtype.kind in {"f", "c"}),
            )
            for key in keys_a
        )


def _replay_equal(left: Path, right: Path) -> bool:
    left_files = sorted((left / "pair_replay").glob("pair_*.npz"))
    right_files = sorted((right / "pair_replay").glob("pair_*.npz"))
    if [item.name for item in left_files] != [item.name for item in right_files]:
        return False
    omitted = frozenset({"parent_pair_transaction_sha256"})
    return all(
        _npz_equal(a, b, omit=omitted)
        for a, b in zip(left_files, right_files, strict=True)
    )


def verify(root: Path) -> dict[str, object]:
    branches: dict[str, object] = {}
    for branch in BRANCHES:
        left, left_pointer = _generation_p2(root / "run_a" / branch)
        right, right_pointer = _generation_p2(root / "run_b" / branch)
        left_decisions = _decision_payload(left)
        right_decisions = _decision_payload(right)
        checks = {
            "canonical_p2_png_exact": _sha_file(
                left / "geometry_and_seam_panorama_owner_only.png"
            ) == _sha_file(right / "geometry_and_seam_panorama_owner_only.png"),
            "seam_arrays_exact": _npz_equal(left / "p2_seams.npz", right / "p2_seams.npz"),
            "provenance_arrays_exact": _npz_equal(
                left / "p2_pixel_provenance.npz",
                right / "p2_pixel_provenance.npz",
            ),
            "pair_selection_payload_exact": left_decisions == right_decisions,
            "replay_pair_content_exact": _replay_equal(left, right),
        }
        if not all(checks.values()):
            failed = [key for key, passed in checks.items() if not passed]
            raise ValueError(f"{branch} reproducibility failed: {failed}")
        branches[branch] = {
            "run_a_generation": left_pointer["generation_id"],
            "run_b_generation": right_pointer["generation_id"],
            "pair_count": len(left_decisions),
            "decision_payload_sha256": _sha_json(left_decisions),
            "canonical_p2_png_sha256": _sha_file(
                left / "geometry_and_seam_panorama_owner_only.png"
            ),
            "seam_arrays_sha256": _sha_file(left / "p2_seams.npz"),
            "checks": checks,
        }
    return {
        "schema": SEAL_SCHEMA,
        "sealed": True,
        "diagnostic_only": True,
        "production_eligible": False,
        "m6_eligible": False,
        "comparison_policy": {
            "generation_specific_lineage_retained_but_not_compared": [
                "P0/P1 completion SHA",
                "pair result_stage_sha256",
                "replay parent_pair_transaction_sha256",
                "generation id/path/time",
            ],
            "diagnostic_float_audits_not_selection_payload": True,
            "nan_arrays_compared_with_equal_nan": True,
        },
        "branches": branches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = verify(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
