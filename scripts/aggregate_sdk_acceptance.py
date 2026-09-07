#!/usr/bin/env python3
"""Sole acceptance-status writer. Phase 2 tests this tool without issuing qualification."""

import argparse
import json
from pathlib import Path

from panorama_demo.sdk_acceptance import (
    native_host,
    read,
    require,
    verify_2d,
    verify_3d,
    verify_binding,
    verify_capture_contracts,
    verify_doctor,
    sha256,
)


def aggregate(kind, root, expected, candidate_index=None):
    verify_binding(read(root / "binding.json"), expected)
    if kind == "software":
        from create_sdk_acceptance_binding import revalidate_binding
        from summarize_linux_l1_benchmark import summarize

        require(candidate_index is not None, "Frozen candidate index required")
        freeze = read(Path(candidate_index).parent.parent / "phase3_bundle_freeze.json")
        require(freeze.get("status") == "PASS" and freeze.get("bundle_frozen") is True
                and freeze.get("milestone") == "RC_BUNDLE_FROZEN"
                and freeze.get("candidate_index_sha256") == sha256(candidate_index)
                and freeze.get("build") == "build_a", "Phase 3 build_a RC is not frozen")
        revalidate_binding(expected, candidate_index)
        verify_doctor(root / "doctor-full-software.json", profile="full_software")
        result = summarize(None, root / "linux", expected)
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
        "schema": "gemini305-sdk-software-acceptance/v3" if kind == "software" else "gemini305-sdk-acceptance-status/v2",
        "status": "PASS",
        "binding": expected,
        "raw_run_directories": [str(root.resolve())],
        "software_ready": kind == "software",
        "hardware_qualified": kind == "native",
        "release_ready": False,
        "signature_status": "UNSIGNED",
        "milestone": "SDK_SOFTWARE_READY" if kind == "software" else "SDK_HARDWARE_QUALIFIED",
        "sdk_cli_exact_equivalence": "PASS" if kind == "software" else None,
        "native_orb": "PASS" if kind == "software" else None,
        "post_3d": "PASS",
        "post_3d_failure_isolation": "PASS",
        "wsl_usbip_supplemental": "NOT_EXECUTED" if not (root / "wsl-usbip").exists() else "SEE_RAW_SUPPLEMENTAL",
        "evidence": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["software", "native"], required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    expected = read(args.binding)
    result = aggregate(args.kind, args.raw_root, expected, args.candidate_index)
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
