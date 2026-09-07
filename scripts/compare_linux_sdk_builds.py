#!/usr/bin/env python3
"""Compare independent builds' wheel bytes and three immutable archives."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare(build_a, build_b):
    if build_a.resolve() == build_b.resolve():
        raise ValueError("Independent build directories are required")
    a, b = [json.loads((root / "release-candidate-index.json").read_text()) for root in (build_a, build_b)]
    errors = []
    for key in ("source_commit", "sdk_version", "runtime_variant", "production_lock", "project_wheel_sha256"):
        if a[key] != b[key]:
            errors.append("IDENTITY_MISMATCH:" + key)
    wheels = [root / "project-wheel" / index["project_wheel_filename"] for root, index in ((build_a, a), (build_b, b))]
    wheel_sha = [sha(p) for p in wheels]
    if wheel_sha[0] != wheel_sha[1] or wheel_sha != [a["project_wheel_sha256"], b["project_wheel_sha256"]]:
        errors.append("PROJECT_WHEEL_MISMATCH")
    with zipfile.ZipFile(wheels[0]) as za, zipfile.ZipFile(wheels[1]) as zb:
        if za.namelist() != zb.namelist() or any(za.read(name) != zb.read(name) for name in za.namelist()):
            errors.append("WHEEL_METADATA_RECORD_OR_RESOURCES_MISMATCH")
    artifacts = {}
    for kind in ("base", "three_d_addon", "source_compliance"):
        values = []
        for root, index in ((build_a, a), (build_b, b)):
            entry = index["artifacts"][kind]
            path = root / entry["filename"]
            actual = sha(path)
            if actual != entry["archive_sha256"] or path.stat().st_size != entry["size"]:
                errors.append("INDEX_ARCHIVE_MISMATCH:" + kind)
            values.append(actual)
        if values[0] != values[1]:
            errors.append("ARCHIVE_NOT_REPRODUCIBLE:" + kind)
        artifacts[kind] = {"build_a_sha256": values[0], "build_b_sha256": values[1], "equal": values[0] == values[1]}
    return {"status": "FAIL" if errors else "PASS", "errors": errors,
            "source_commit": a["source_commit"], "project_wheel_sha256": wheel_sha,
            "artifacts": artifacts, "software_ready": False, "hardware_qualified": False, "release_ready": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-a", required=True, type=Path)
    parser.add_argument("--build-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.build_a, args.build_b)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))
    raise SystemExit(result["status"] != "PASS")
