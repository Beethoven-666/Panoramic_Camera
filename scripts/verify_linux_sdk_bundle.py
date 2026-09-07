#!/usr/bin/env python3
"""Read-only verification of bundle contents; never issues SDK qualification."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import sys

sys.dont_write_bytecode = True


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def verify(root):
    errors = []
    covered = set()
    for line in (root / "checksums.sha256").read_text().splitlines():
        sha, relative = line.split("  ", 1)
        path = root / relative
        path.resolve().relative_to(root.resolve())
        if relative in covered or not path.is_file() or digest(path) != sha:
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
    if manifest.get("content_checksums_file"):
        if digest(root / manifest["content_checksums_file"]) != manifest["content_checksums_sha256"]:
            errors.append("PAYLOAD_CHECKSUM_INDEX_MISMATCH")
    blocked = {"ORBvoc.txt", "ORBvoc.bin", "rgbd_tum_headless",
               "rgbd_g305_stream_headless", "libORB_SLAM3.so"}
    for path in root.rglob("*"):
        if path.name in blocked:
            errors.append("ORB_PUBLIC_DISTRIBUTION_BLOCKED:" + path.name)
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
    if manifest.get("signature_status") != "UNSIGNED":
        errors.append("CANDIDATE_MUST_REMAIN_UNSIGNED")
    return {
        "status": "FAIL" if errors else "PASS",
        "errors": errors,
        "source_commit": manifest["source_commit"],
        "content_checksums_sha256": digest(root / "checksums.sha256"),
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
