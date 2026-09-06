#!/usr/bin/env python3
"""Sole acceptance-status writer. Phase 2 tests this tool without issuing qualification."""

import argparse
import json
from pathlib import Path

from panorama_demo.sdk_acceptance import (
    binding,
    native_host,
    read,
    require,
    verify_2d,
    verify_3d,
    verify_binding,
    verify_capture_contracts,
    verify_doctor,
)


def aggregate(kind, root, expected):
    verify_binding(read(root / "binding.json"), expected)
    if kind == "software":
        from summarize_linux_l1_benchmark import summarize

        result = summarize(root / "windows", root / "linux")
        require(
            result["status"] == "PASS",
            "L1 raw suites incomplete: " + str(result.get("evidence_errors")),
        )
        for suite in (root / "linux").rglob("suite.json"):
            verify_binding(read(suite.parent / "binding.json"), expected)
    else:
        require(native_host(), "WSL and non-native hosts cannot issue H0")
        result = read(root / "sdk_h0_acceptance.json")
        require(result.get("platform") == "native_linux", "H0 evidence is not native")
        records = result["records"]
        formal = [r for r in records if not r["warmup"]]
        warmup = [r for r in records if r["warmup"]]
        require(
            len(formal) >= 5
            and len(warmup) >= 1
            and all(r["status"] == "PASS" for r in records),
            "H0 requires successful 1+5 runs",
        )
        require(result["cli_crosscheck"]["status"] == "PASS", "CLI cross-check missing")
        verify_capture_contracts(result["capture_contracts"])
        verify_doctor(root / "doctor.json")
        for record in records:
            verify_2d(
                Path(record["output"]),
                Path(result["cli_crosscheck"]["detail"]["root"]) / record["label"],
            )
        verify_3d(Path(result["post_3d_success_root"]))
        verify_3d(Path(result["post_3d_failure_root"]), failure=True)
    return {
        "schema": "gemini305-sdk-acceptance-status/v2",
        "status": "PASS",
        "binding": expected,
        "raw_run_directories": [str(root.resolve())],
        "software_ready": kind == "software",
        "hardware_qualified": kind == "native",
        "release_ready": False,
        "evidence": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["software", "native"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--project-wheel", type=Path, required=True)
    parser.add_argument("--bundle-checksums", type=Path, required=True)
    parser.add_argument("--runtime-variant", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    expected = binding(
        args.source_commit,
        args.project_wheel,
        args.bundle_checksums,
        args.runtime_variant,
        args.platform,
    )
    result = aggregate(args.kind, args.raw_root, expected)
    name = (
        "software_acceptance_status.json"
        if args.kind == "software"
        else "native_acceptance_status.json"
    )
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with (args.output_directory / name).open("x") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
