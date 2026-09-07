#!/usr/bin/env python3
"""Verify the indexed immutable archive before extracting into a new directory."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

sys.dont_write_bytecode = True
ORB_NAMES = {"ORBvoc.txt", "ORBvoc.bin", "rgbd_tum_headless",
             "rgbd_g305_stream_headless", "libORB_SLAM3.so"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def open_tar(path):
    try:
        import zstandard
    except ImportError:
        process = subprocess.Popen(["zstd", "-q", "-d", "-c", str(path)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                yield archive
        finally:
            process.stdout.close()
            if process.wait() != 0:
                raise ValueError("ZSTD_DECOMPRESSION_FAILED")
    else:
        with (path.open("rb") as raw,
              zstandard.ZstdDecompressor().stream_reader(raw) as stream,
              tarfile.open(fileobj=stream, mode="r|") as archive):
            yield archive


def validate_member(member, expected_top):
    path = PurePosixPath(member.name)
    if (path.is_absolute() or not path.parts or ".." in path.parts or
            "\\" in member.name or ":" in path.parts[0] or path.parts[0] != expected_top):
        raise ValueError("UNSAFE_ARCHIVE_PATH:" + member.name)
    # The builder emits regular files and directories only, so links/devices are not valid RC content.
    if not (member.isfile() or member.isdir()):
        raise ValueError("UNSUPPORTED_ARCHIVE_MEMBER:" + member.name)


def verify_contents(root, kind):
    covered = set()
    for line in (root / "checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        path.resolve().relative_to(root.resolve())
        if relative in covered or not path.is_file() or sha(path) != expected:
            raise ValueError("CHECKSUM_MISMATCH_OR_DUPLICATE:" + relative)
        covered.add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != covered | {"checksums.sha256"}:
        raise ValueError("UNDECLARED_FILE_OR_CHECKSUM_ENTRY")
    if kind == "source_compliance":
        return
    manifest = json.loads((root / "manifests/bundle-manifest.json").read_text())
    if manifest["kind"] != {"base": "base", "three_d_addon": "3d-addon"}[kind]:
        raise ValueError("BUNDLE_KIND_MISMATCH")
    if manifest.get("signature_status") != "UNSIGNED" or any(
            manifest.get(key) is not False for key in ("software_ready", "hardware_qualified", "release_ready")):
        raise ValueError("CANDIDATE_CANNOT_ISSUE_QUALIFICATION")
    if sha(root / manifest["content_checksums_file"]) != manifest["content_checksums_sha256"]:
        raise ValueError("PAYLOAD_CHECKSUM_INDEX_MISMATCH")
    for path in root.rglob("*"):
        if path.name in ORB_NAMES:
            raise ValueError("ORB_PUBLIC_DISTRIBUTION_BLOCKED:" + path.name)
    for wheel in (root / "wheels").glob("*.whl"):
        if kind == "base" and wheel.name.lower().startswith("open3d-"):
            raise ValueError("OPEN3D_IN_BASE")
        with zipfile.ZipFile(wheel) as archive:
            if any(PurePosixPath(name).name in ORB_NAMES for name in archive.namelist()):
                raise ValueError("ORB_PUBLIC_DISTRIBUTION_BLOCKED:" + wheel.name)


def verify(index_path, archive_path, kind, destination=None):
    index = json.loads(index_path.read_text())
    item = index["artifacts"][kind]
    if (archive_path.name != item["filename"] or archive_path.stat().st_size != item["size"]
            or sha(archive_path) != item["archive_sha256"]):
        raise ValueError("ARCHIVE_IDENTITY_MISMATCH")
    version, variant = index["sdk_version"], index["runtime_variant"]
    expected_top = (f"gemini305-sdk-source-compliance-{version}" if kind == "source_compliance"
                    else f"gemini305-sdk-{'base' if kind == 'base' else '3d-addon'}-{version}-{variant}")
    names = set()
    # Complete path audit before creating the requested destination or extracting any member.
    with open_tar(archive_path) as archive:
        for member in archive:
            validate_member(member, expected_top)
            if member.name in names:
                raise ValueError("DUPLICATE_ARCHIVE_MEMBER:" + member.name)
            names.add(member.name)
    if not names:
        raise ValueError("EMPTY_ARCHIVE")
    with tempfile.TemporaryDirectory(prefix="g305-verify-") as temporary:
        stage = Path(temporary)
        with open_tar(archive_path) as archive:
            for member in archive:
                target = stage / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output)
                    target.chmod(member.mode)
        root = stage / expected_top
        verify_contents(root, kind)
        content_sha = sha(root / "checksums.sha256")
        if kind != "source_compliance":
            manifest = json.loads((root / "manifests/bundle-manifest.json").read_text())
            for key in ("source_commit", "sdk_version", "runtime_variant", "project_wheel_sha256", "production_lock"):
                if manifest[key] != index[key]:
                    raise ValueError("INDEX_CONTENT_IDENTITY_MISMATCH:" + key)
            if content_sha != item["content_checksums_sha256"]:
                raise ValueError("CONTENT_CHECKSUM_INDEX_MISMATCH")
        if destination is not None:
            destination.mkdir(parents=True, exist_ok=False)
            shutil.move(str(root), str(destination / expected_top))
    return {"status": "PASS", "kind": kind, "archive_sha256": item["archive_sha256"],
            "content_checksums_sha256": content_sha, "top_directory": expected_top,
            "extracted_root": str(destination / expected_top) if destination is not None else None,
            "signature_status": "UNSIGNED", "software_ready": False,
            "hardware_qualified": False, "release_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-index", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("base", "three_d_addon", "source_compliance"))
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.candidate_index, args.archive, args.kind, args.extract_to)
    except (ValueError, KeyError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        result = {"status": "FAIL", "error": str(exc), "software_ready": False,
                  "hardware_qualified": False, "release_ready": False}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))
    return result["status"] != "PASS"


if __name__ == "__main__":
    raise SystemExit(main())
