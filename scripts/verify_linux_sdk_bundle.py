#!/usr/bin/env python3
"""Read-only verification of bundle contents; never issues SDK qualification."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def verify(root):
    errors = []
    covered = set()
    for line in (root / "checksums.sha256").read_text().splitlines():
        sha, relative = line.split("  ", 1)
        path = root / relative
        path.resolve().relative_to(root.resolve())
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            errors.append("CHECKSUM_MISMATCH:" + relative)
        covered.add(relative)
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if (
            path.is_file()
            and relative not in covered
            and relative
            not in {
                "checksums.sha256",
                "checksums.sha256.asc",
                "checksums.sha256.TEST_ONLY.asc",
            }
        ):
            errors.append("UNDECLARED_FILE:" + relative)
    manifest = json.loads((root / "manifests/bundle-manifest.json").read_text())
    for wheel in (root / "wheels").glob("*.whl"):
        if manifest["kind"] == "base" and wheel.name.lower().startswith("open3d-"):
            errors.append("OPEN3D_IN_BASE")
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if Path(name).name in {
                    "ORBvoc.txt",
                    "ORBvoc.bin",
                    "rgbd_tum_headless",
                    "rgbd_g305_stream_headless",
                    "libORB_SLAM3.so",
                }:
                    errors.append("ORB_PUBLIC_DISTRIBUTION_BLOCKED:" + name)
    for name in (
        "runtime-variant-manifest.json",
        "dependency-manifest.json",
        "elf-dependency-report.json",
        "sbom.spdx.json",
    ):
        json.loads((root / "manifests" / name).read_text())
    if any(
        manifest.get(key) is not False
        for key in ("software_ready", "hardware_qualified", "release_ready")
    ):
        errors.append("DEVELOPMENT_BUNDLE_CANNOT_ISSUE_QUALIFICATION")
    return {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "source_commit": manifest["source_commit"],
        "bundle_checksums_sha256": hashlib.sha256(
            (root / "checksums.sha256").read_bytes()
        ).hexdigest(),
        "software_ready": False,
        "hardware_qualified": False,
        "release_ready": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.bundle)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    raise SystemExit(result["status"] != "PASS")
