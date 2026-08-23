"""Metrics-only calibration for the S013 M6.1 approval boundary.

This module deliberately has no renderer or candidate-solver dependency.  It
consumes four sealed P2-v4 ``baseline_input.json`` files, measures the identity
Q0/B0 baseline, and writes only the proposed (never final) threshold assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


BASELINE_INPUT_SCHEMA = "gemini305-video-s13-m61-calibration-input/v1"
BASELINE_QUALITY_SCHEMA = "gemini305-video-s13-m61-quality-baseline/v1"
EVALUABILITY_SCHEMA = "gemini305-video-s13-m61-evaluability/v1"
MDE_SCHEMA = "gemini305-video-s13-m61-mde-calibration/v1"
THRESHOLD_SCHEMA = "gemini305-video-s13-m61-quality-thresholds/v2"
EXPECTED_BRANCHES = (
    "fast_direct",
    "fast_ignore_pose",
    "slow_direct",
    "slow_ignore_pose",
)
METRIC_NAMES = (
    "aggregate_linear_residual_median",
    "aggregate_linear_residual_p95",
    "pair_linear_residual_p95",
    "delta_e00",
    "immediate_seam",
    "gradient_ghost",
    "owner_plateau",
    "clipping_fraction",
)
FORBIDDEN_OUTPUT_NAMES = {
    "quality_thresholds_m61_v2.json",
    "candidate_manifest.json",
    "effective_config.json",
    "P3",
}


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def load_baseline_input(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != BASELINE_INPUT_SCHEMA:
        raise ValueError(f"{path}: unsupported baseline input schema")
    if payload.get("sealed") is not True:
        raise ValueError(f"{path}: P2-v4 baseline input is not sealed")
    if payload.get("q0_b0_only", True) is not True or payload.get("q_solver_invocations", 0) != 0:
        raise ValueError(f"{path}: calibration input crossed the Q0+B0 preparation boundary")
    if payload.get("photometric_model") != "Q0_identity" or payload.get("blend_model") != "B0_owner_only":
        raise ValueError(f"{path}: calibration permits Q0_identity+B0_owner_only only")
    branch = payload.get("branch")
    if branch not in EXPECTED_BRANCHES:
        raise ValueError(f"{path}: unexpected branch {branch!r}")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
        raise ValueError(f"{path}: run_id is required")
    for field in ("p2_completion_sha256", "p2_canonical_image_sha256", "graph_topology_sha256"):
        _sha(payload.get(field), f"{path}:{field}")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path}: pairs must be a non-empty array")
    seen: set[int] = set()
    for row in pairs:
        if not isinstance(row, dict) or isinstance(row.get("pair_index"), bool) or not isinstance(row.get("pair_index"), int):
            raise ValueError(f"{path}: each pair requires integer pair_index")
        pair_index = row["pair_index"]
        if pair_index < 0 or pair_index in seen:
            raise ValueError(f"{path}: duplicate/invalid pair_index {pair_index}")
        seen.add(pair_index)
        classification = row.get("classification", row.get("quality_class"))
        if classification not in {"quality_safe_evaluable", "protected_only", "quality_insufficient_evidence"}:
            raise ValueError(f"{path}: invalid pair classification")
        row.setdefault("classification", classification)
        metrics = row.get("metrics", row.get("statistics"))
        if not isinstance(metrics, dict):
            raise ValueError(f"{path}: pair {pair_index} metrics are required")
        row.setdefault("metrics", metrics)
        for name, value in metrics.items():
            if name not in METRIC_NAMES:
                raise ValueError(f"{path}: unknown metric {name!r}")
            _finite_number(value, f"pair {pair_index}:{name}")
        replicates = row.get("calibration_replicates", {})
        if not isinstance(replicates, dict):
            raise ValueError(f"{path}: calibration_replicates must be an object")
        for name, values in replicates.items():
            if name not in METRIC_NAMES or not isinstance(values, list):
                raise ValueError(f"{path}: invalid calibration replicate {name!r}")
            for value in values:
                _finite_number(value, f"pair {pair_index}:{name}:replicate")
    return payload


def _quantile(values: Sequence[float], q: float) -> float | None:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else None


def _pair_evaluability(row: Mapping[str, Any]) -> dict[str, Any]:
    classification = str(row["classification"])
    samples = int(row.get("validation_sample_count", 0))
    tiles = int(row.get("independent_validation_tile_count", 0))
    blocks = int(row.get("vertical_block_count", 0))
    overlap = int(row.get("train_validation_overlap_count", 0))
    measured = classification == "quality_safe_evaluable"
    gate = measured and samples >= 128 and tiles >= 8 and blocks >= 4 and overlap == 0
    return {
        "pair_index": int(row["pair_index"]),
        "classification": classification,
        "validation_sample_count": samples,
        "independent_validation_tile_count": tiles,
        "vertical_block_count": blocks,
        "train_validation_overlap_count": overlap,
        "high_confidence": gate and samples >= 256,
        "evaluable": gate,
        "reason": None if gate else (classification if not measured else "coverage_below_frozen_provisional_gate"),
    }


def calibrate_m61_quality(inputs: Sequence[tuple[Path, Mapping[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Build the four proposed calibration assets without rendering or solving."""
    if len(inputs) != 4:
        raise ValueError("exactly four branch-specific sealed P2-v4 inputs are required")
    by_branch = {str(payload["branch"]): (path, payload) for path, payload in inputs}
    if tuple(sorted(by_branch)) != tuple(sorted(EXPECTED_BRANCHES)) or len(by_branch) != 4:
        raise ValueError("baseline inputs must contain each fixed branch exactly once")

    branch_reports: list[dict[str, Any]] = []
    eval_reports: list[dict[str, Any]] = []
    replicate_deltas: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    parents: list[dict[str, Any]] = []
    all_metric_values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for branch in EXPECTED_BRANCHES:
        path, payload = by_branch[branch]
        pair_rows = []
        branch_eval = []
        for row in sorted(payload["pairs"], key=lambda item: int(item["pair_index"])):
            evaluation = _pair_evaluability(row)
            branch_eval.append(evaluation)
            metrics = {name: _finite_number(value, name) for name, value in sorted(row["metrics"].items())}
            pair_rows.append({"pair_index": row["pair_index"], "classification": row["classification"], "metrics": metrics})
            if evaluation["evaluable"]:
                for name, value in metrics.items():
                    all_metric_values[name].append(value)
                    for replicate in row.get("calibration_replicates", {}).get(name, []):
                        replicate_deltas[name].append(abs(_finite_number(replicate, name) - value))
        safe_count = sum(row["classification"] == "quality_safe_evaluable" for row in branch_eval)
        insufficient_count = sum(row["classification"] == "quality_insufficient_evidence" for row in branch_eval)
        branch_reports.append({
            "branch": branch,
            "run_id": payload["run_id"],
            "baseline_input_sha256": _sha_file(path),
            "graph_topology_sha256": payload["graph_topology_sha256"],
            "photometric_model": "Q0_identity",
            "blend_model": "B0_owner_only",
            "pairs": pair_rows,
        })
        eval_reports.append({
            "branch": branch,
            "pairs": branch_eval,
            "safe_pair_count": safe_count,
            "quality_insufficient_evidence_safe_pair_count": insufficient_count,
            "all_safe_pairs_evaluable": safe_count > 0 and insufficient_count == 0 and all(
                row["evaluable"] for row in branch_eval if row["classification"] == "quality_safe_evaluable"
            ),
        })
        parents.append({
            "branch": branch,
            "run_id": payload["run_id"],
            "p2_completion_sha256": payload["p2_completion_sha256"],
            "p2_canonical_image_sha256": payload["p2_canonical_image_sha256"],
            "graph_topology_sha256": payload["graph_topology_sha256"],
        })

    baseline = {
        "schema": BASELINE_QUALITY_SCHEMA,
        "calibration_role": "metrics_only",
        "candidate_execution_allowed": False,
        "photometric_model": "Q0_identity",
        "blend_model": "B0_owner_only",
        "branches": branch_reports,
        "aggregate": {name: {"median": _quantile(values, 0.5), "p95": _quantile(values, 0.95)} for name, values in all_metric_values.items()},
    }
    evaluability = {
        "schema": EVALUABILITY_SCHEMA,
        "train_validation_overlap_required": 0,
        "provisional_gates": {"validation_samples": 128, "independent_tiles": 8, "vertical_blocks": 4, "high_confidence_samples": 256},
        "branches": eval_reports,
    }
    # Explicit Step-2 replicates take precedence.  The deterministic fallback is
    # a leave-one-pair/tile-fold proxy over the sealed validation measurements;
    # it measures absolute perturbation around the fixed median and never looks
    # at a non-identity candidate.
    mde_samples: dict[str, list[float]] = {}
    for name in METRIC_NAMES:
        samples = replicate_deltas[name]
        if not samples:
            values = all_metric_values[name]
            center = _quantile(values, 0.5) or 0.0
            samples = [abs(value - center) for value in values]
        mde_samples[name] = samples
    mde_values = {name: max(_quantile(values, 0.95) or 0.0, 1e-12) for name, values in mde_samples.items()}
    mde = {
        "schema": MDE_SCHEMA,
        "method": "absolute-delta-p95/block-jackknife-tile-fold-synthetic-known-transform/v1",
        "metrics": {
            name: {
                "absolute_mde": value,
                "replicate_count": len(mde_samples[name]),
                "sample_source": "explicit_calibration_replicates" if replicate_deltas[name] else "sealed_pair_tile_fold_proxy",
            }
            for name, value in mde_values.items()
        },
    }
    proposed = {
        "schema": THRESHOLD_SCHEMA,
        "status": "proposed",
        "approval_required": True,
        "candidate_execution_allowed": False,
        "allowed_photometric_models": ["Q0_identity"],
        "allowed_blend_models": ["B0_owner_only"],
        "aggregate_validation_p95_non_regression": {"relative_tolerance": 0.02, "absolute_tolerance_linear": 0.0005, "formula": "after <= before + max(0.02 * before, 0.0005)"},
        "mde_schema": MDE_SCHEMA,
        "canonical_mde_values": mde_values,
        "four_p2_parents": parents,
        "graph_topology_sha256_by_branch": {row["branch"]: row["graph_topology_sha256"] for row in parents},
        "baseline_quality_sha256": canonical_sha256(baseline),
        "baseline_quality_manifest_sha256": canonical_sha256(baseline),
        "evaluability_sha256": canonical_sha256(evaluability),
        "mde_calibration_sha256": canonical_sha256(mde),
        "next_required_action": "obtain_explicit_human_approval_and_write_append_only_threshold_approval_witness",
    }
    return {"baseline": baseline, "evaluability": evaluability, "mde": mde, "proposed": proposed}


def write_proposed_calibration(output_dir: Path, assets: Mapping[str, Mapping[str, Any]]) -> tuple[Path, ...]:
    """Atomically write only the Step-3 pre-approval assets into a fresh directory."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"calibration output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_atlases").mkdir()
    names = {
        "baseline": "quality_metrics_baseline.json",
        "evaluability": "evaluability_by_pair.json",
        "mde": "mde_calibration.json",
        "proposed": "quality_thresholds_m61_v2.proposed.json",
    }
    written: list[Path] = []
    try:
        for key, filename in names.items():
            path = output_dir / filename
            pending = output_dir / f".{filename}.pending"
            pending.write_bytes(canonical_json_bytes(dict(assets[key])))
            os.replace(pending, path)
            written.append(path)
    except Exception:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    if any((output_dir / name).exists() for name in FORBIDDEN_OUTPUT_NAMES):
        raise RuntimeError("calibration crossed the pre-approval output boundary")
    return tuple(written)


__all__ = [
    "BASELINE_INPUT_SCHEMA",
    "EXPECTED_BRANCHES",
    "calibrate_m61_quality",
    "canonical_sha256",
    "load_baseline_input",
    "write_proposed_calibration",
]
