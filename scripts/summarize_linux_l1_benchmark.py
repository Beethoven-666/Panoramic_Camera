#!/usr/bin/env python3
"""Fail-closed comparison of existing Windows and Linux L1 benchmark suites."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from panorama_demo.sdk_acceptance import (
    read, require, verify_2d_contract, verify_2d_equivalence, verify_3d,
    verify_native_orb, verify_doctor,
)


def _load_root(root: Path | None) -> dict[str, Any]:
    suites: dict[str, Any] = {}
    if root is None:
        return suites
    for suite_path in root.rglob("suite.json"):
        identity = json.loads(suite_path.read_text(encoding="utf-8"))
        samples = []
        for sample_path in sorted(suite_path.parent.glob("run_*/sample.json")):
            sample = json.loads(sample_path.read_text(encoding="utf-8"))
            for field in ("status", "reason_code", "wall_seconds", "artifacts"):
                if field not in sample:
                    raise ValueError(f"Missing {field}: {sample_path}")
            samples.append(sample)
        key = f"{identity['operation']}:{identity.get('entry', 'cli')}"
        if key in suites:
            raise ValueError(f"Duplicate suite: {key}")
        suites[key] = {"identity": identity, "samples": samples}
        suites[key]["root"] = suite_path.parent
    return suites


def _statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) < 5:
        return {
            "status": "FAIL",
            "reason_code": "INSUFFICIENT_SAMPLES",
            "sample_count": len(samples),
        }
    successful = [sample for sample in samples if sample["status"] == "PASS"]
    values = [float(sample["wall_seconds"]) for sample in successful]
    return {
        "status": "PASS" if len(successful) == len(samples) else "FAIL",
        "reason_code": "ALL_RUNS_PASSED"
        if len(successful) == len(samples)
        else "RUN_FAILURE",
        "sample_count": len(samples),
        "success_count": len(successful),
        "success_rate": len(successful) / len(samples),
        "wall_seconds_median": statistics.median(values) if values else None,
        "wall_seconds_p95": float(np.percentile(values, 95, method="linear"))
        if values
        else None,
    }


def _performance(samples, key):
    def stats(values):
        return ({"status": "PASS", "median": float(np.median(values)),
                 "p95": float(np.percentile(values, 95)), "maximum": float(np.max(values))}
                if values else {"status": "NOT_EXECUTED", "median": None, "p95": None, "maximum": None})
    orb, tsdf, glb = [], [], []
    for sample in samples:
        output = Path(sample["product_output"])
        if key == "orb:cli":
            trajectory = read(output / "orbslam3_trajectory.json")
            orb.append(sum(a["elapsed_seconds"] for a in trajectory["execution_attempts"]))
        if key == "post-3d:sdk":
            timing = read(output / "video_3d_timing.json")
            orb.append((timing["orb_completed_monotonic_ns"] - timing["orb_started_monotonic_ns"]) / 1e9)
            dense = read(output / "video_3d_delivery.json")["audit"].get("timing_seconds", {})
            if "tsdf_integration" in dense:
                tsdf.append(dense["tsdf_integration"])
            if "projective_texture_and_glb" in dense:
                glb.append(dense["projective_texture_and_glb"])
    rss = [s["rss"]["peak_process_tree_rss_bytes"] for s in samples
           if s.get("rss", {}).get("status") == "PASS"]
    gpu = [s["gpu"]["peak_gpu_memory_mib"] for s in samples if s.get("gpu", {}).get("status") == "PASS"]
    return {"orb_seconds": stats(orb), "tsdf_seconds": stats(tsdf),
            "projective_texture_and_glb_seconds": stats(glb),
            "peak_process_tree_rss_bytes": max(rss) if rss else None,
            "rss_sampling": "PASS" if len(rss) == len(samples) else "NOT_EXECUTED",
            "peak_gpu_memory_mib": max(gpu) if gpu else None,
            "gpu_sampling": "PASS" if len(gpu) == len(samples) else "NOT_EXECUTED",
            "gpu_utilization_samples": [s.get("gpu", {}).get("utilization_percent", []) for s in samples],
            "gpu_memory_scope": "whole_device"}


def summarize(windows_root: Path | None, linux_root: Path, expected_binding=None) -> dict[str, Any]:
    windows, linux = _load_root(windows_root), _load_root(linux_root)
    required_linux = {"frozen-2d:cli", "frozen-2d:sdk", "orb:cli", "post-3d:sdk"}
    missing = sorted(required_linux - linux.keys())
    status, reason = (
        ("FAIL", "REQUIRED_SUITE_MISSING") if missing else ("PASS", "SUITES_COMPLETE")
    )
    suites: dict[str, Any] = {}
    for platform_name, values in (("windows", windows), ("linux", linux)):
        suites[platform_name] = {
            key: _statistics(value["samples"]) for key, value in values.items()
        }
    if any(suites["linux"][key]["status"] != "PASS" for key in required_linux & linux.keys()):
        status, reason = "FAIL", "LINUX_SUITE_FAILED"
    evidence_errors = []
    if not missing:
        for key in sorted(required_linux):
            suite = linux[key]
            root = suite["root"]
            try:
                warmups = list(root.glob("warmup_*/sample.json"))
                require(len(warmups) == 1 and read(warmups[0])["status"] == "PASS",
                        "Missing successful single warmup")
                require({p.name for p in root.glob("run_*")} == {f"run_{i:02}" for i in range(1, 6)},
                        "Need exactly five retained formal runs")
                binding = read(root / "binding.json")
                if expected_binding is not None:
                    require(binding == expected_binding, "Suite binding mismatch")
                verify_doctor(root / "doctor.json", profile="full_software")
                contracts = []
                for run in [warmups[0].parent, *sorted(root.glob("run_*"))]:
                    sample = read(run / "sample.json")
                    require(sample["status"] == "PASS", "Failed run: " + run.name)
                    output = Path(sample["product_output"])
                    require(output.is_relative_to(run.resolve()), "Run output points outside its own retained directory")
                    if key.startswith("frozen-2d:"):
                        contract = verify_2d_contract(output, binding)
                        contracts.append(contract)
                        if key == "frozen-2d:cli":
                            peer_run = linux["frozen-2d:sdk"]["root"] / run.name
                            peer = Path(read(peer_run / "sample.json")["product_output"])
                            verify_2d_equivalence(output, peer, binding)
                    elif key == "orb:cli":
                        contracts.append(verify_native_orb(output, binding))
                    elif key == "post-3d:sdk":
                        contracts.append(verify_3d(output, expected_binding=binding))
                suites["linux"][key]["verified_run_count"] = len(contracts)
                suites["linux"][key]["performance"] = _performance(suite["samples"], key)
                if key.startswith("frozen-2d:"):
                    values = [c["capture_stop_to_p3_published_seconds"] for c in contracts[1:]]
                    suites["linux"][key].update(
                        capture_stop_to_p3_published_seconds_median=statistics.median(values),
                        capture_stop_to_p3_published_seconds_p95=float(np.percentile(values, 95)))
                if key == "orb:cli":
                    suites["linux"][key]["repeatability"] = orb_repeatability(contracts[1:])
                if key == "post-3d:sdk":
                    verify_3d(root / "failure-isolation/2d/3d", failure=True, expected_binding=binding)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                evidence_errors.append({"suite": key, "error": str(exc)})
        if evidence_errors:
            status, reason = "FAIL", "RAW_EVIDENCE_INVALID"
    return {
        "schema": "gemini305-linux-l1-benchmark-summary/v1",
        "status": status,
        "reason_code": reason,
        "missing_required_suites": missing,
        "suites": suites,
        "evidence_errors": evidence_errors,
        "metric_boundaries": {
            "two_d_user_wait": "capture_stop_to_p3_published_seconds",
            "orb_tsdf_3d_reported_separately": True,
        },
    }


def orb_repeatability(contracts):
    """Report numerical variation without inventing a cross-run acceptance threshold."""
    ids = [set(c["tracked_frame_ids"]) for c in contracts]
    endpoints = [np.asarray(c["camera_to_world"])[-1] for c in contracts]
    translations, rotations = [], []
    increments = []
    for contract in contracts:
        poses = np.asarray(contract["camera_to_world"])
        increments.extend(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).tolist())
    for left, right in zip(endpoints, endpoints[1:]):
        delta = np.linalg.inv(left) @ right
        translations.append(float(np.linalg.norm(delta[:3, 3])))
        rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))))
    def stats(values):
        return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
                "maximum": float(np.max(values))}
    return {"status": "PASS", "cross_run_pose_equality_required": False,
            "tracked_frame_counts": [len(i) for i in ids],
            "tracked_frame_intersection_fraction": len(set.intersection(*ids)) / len(set.union(*ids)),
            "adjacent_translation_mm": stats(increments),
            "endpoint_translation_mm": stats(translations), "endpoint_rotation_degrees": stats(rotations)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-root", "--optional-historical-windows-root", type=Path)
    parser.add_argument("--linux-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.windows_root, args.linux_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output.parent / "performance-summary.json").write_text(json.dumps(
        result["suites"]["linux"], indent=2), encoding="utf-8")
    repeatability = result["suites"]["linux"].get("orb:cli", {}).get("repeatability",
        {"status": "NOT_EXECUTED"})
    (args.output.parent / "orb-repeatability.json").write_text(json.dumps(repeatability, indent=2), encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
