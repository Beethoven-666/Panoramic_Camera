#!/usr/bin/env python3
"""Fail-closed comparison of existing Windows and Linux L1 benchmark suites."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
from panorama_demo.sdk_acceptance import read, verify_2d, verify_3d, verify_doctor


def _load_root(root: Path) -> dict[str, Any]:
    suites: dict[str, Any] = {}
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


def summarize(windows_root: Path, linux_root: Path) -> dict[str, Any]:
    windows, linux = _load_root(windows_root), _load_root(linux_root)
    required_linux = {"frozen-2d:cli", "frozen-2d:sdk", "post-3d:sdk"}
    missing = sorted(required_linux - linux.keys())
    status, reason = (
        ("FAIL", "REQUIRED_SUITE_MISSING") if missing else ("PASS", "SUITES_COMPLETE")
    )
    suites: dict[str, Any] = {}
    for platform_name, values in (("windows", windows), ("linux", linux)):
        suites[platform_name] = {
            key: _statistics(value["samples"]) for key, value in values.items()
        }
    if any(item["status"] != "PASS" for item in suites["linux"].values() if item):
        status, reason = "FAIL", "LINUX_SUITE_FAILED"
    evidence_errors = []
    if not missing:
        for key, suite in linux.items():
            root = suite["root"]
            try:
                warmups = list(root.glob("warmup_*/sample.json"))
                if not warmups or any(read(p)["status"] != "PASS" for p in warmups):
                    raise ValueError("Missing successful warmup")
                verify_doctor(root / "doctor.json")
                for run in sorted(root.glob("run_*")):
                    if key.startswith("frozen-2d:"):
                        peer = (
                            linux[
                                "frozen-2d:sdk"
                                if key.endswith(":cli")
                                else "frozen-2d:cli"
                            ]["root"]
                            / run.name
                            / "2d"
                        )
                        verify_2d(run / "2d", peer)
                    elif key == "post-3d:sdk":
                        verify_3d(run / "3d")
                if key == "post-3d:sdk":
                    verify_3d(root / "failure-isolation/3d", failure=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows-root", type=Path, required=True)
    parser.add_argument("--linux-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.windows_root, args.linux_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
