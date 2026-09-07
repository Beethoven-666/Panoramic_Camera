#!/usr/bin/env python3
"""Build an independent unsigned H0 tar.zst from exact committed Git blobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout


def allowed(name):
    return ((name.startswith("qualification/native_h0/") and name.endswith((".py", ".json")))
            or (name.startswith("scripts/") and "/" not in name[8:] and "native_h0" in name and name.endswith(".py"))
            or name == "docs/NATIVE_UBUNTU_H0.md")


def build(repo, output_directory, commit="HEAD"):
    repo, output = Path(repo).resolve(), Path(output_directory).resolve()
    resolved = git(repo, "rev-parse", "--verify", commit + "^{commit}").decode().strip()
    subject = "2abb132043e7c5ae1805a2ed1dd91516ebf2f874"
    require_ancestor = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", subject, resolved],
                                     capture_output=True)
    if require_ancestor.returncode != 0 or resolved == subject:
        raise ValueError("Tools must be independently committed above the frozen candidate")
    changed = git(repo, "diff", "--name-only", subject, resolved).decode().splitlines()
    if any(not (allowed(name) or name.startswith("tests/test_native_h0_")) for name in changed):
        raise ValueError("SDK/runtime/config change requires rc2 and phases 3-5")
    names = [name for name in git(repo, "ls-tree", "-r", "--name-only", resolved).decode().splitlines() if allowed(name)]
    required = {"qualification/native_h0/binding.py", "qualification/native_h0/aggregator.py",
                "scripts/create_native_h0_binding.py", "scripts/aggregate_native_h0_acceptance.py"}
    if not required.issubset(names):
        raise ValueError("H0 tool commit is incomplete")
    payload = {name: git(repo, "show", resolved + ":" + name) for name in names}
    epoch = int(git(repo, "show", "-s", "--format=%ct", resolved).decode().strip())
    manifest = {"schema": "gemini305-native-h0-tools/v1", "h0_tool_commit": resolved,
                "subject_source_commit": subject, "requires_python": ">=3.10,<3.11",
                "candidate_schema": "gemini305-linux-sdk-release-candidate/v2",
                "native_acceptance_schema": "gemini305-sdk-native-acceptance/v3",
                "build_time": datetime.fromtimestamp(epoch, timezone.utc).isoformat(),
                "signature_status": "UNSIGNED",
                "files": {name: hashlib.sha256(value).hexdigest() for name, value in sorted(payload.items())}}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    payload["native-h0-tools-manifest.json"] = manifest_bytes
    output.mkdir(parents=True, exist_ok=False)
    archive = output / ("gemini305-native-h0-tools-" + resolved + ".tar.zst")
    with archive.open("xb") as stream:
        process = subprocess.Popen(["zstd", "-q", "-6", "-T1", "-c"], stdin=subprocess.PIPE,
                                   stdout=stream, stderr=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name, value in sorted(payload.items()):
                    entry = tarfile.TarInfo(name)
                    entry.size, entry.mtime, entry.mode = len(value), epoch, 0o644
                    tar.addfile(entry, io.BytesIO(value))
            process.stdin.close()
            stderr = process.stderr.read().decode(errors="replace")
            if process.wait() != 0:
                raise ValueError("H0 tools zstd compression failed: " + stderr)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdin.close()
            process.stderr.close()
    manifest_path = output / "native-h0-tools-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    checksum = output / "native-h0-tools-checksums.sha256"
    checksum.write_text("".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n"
                                for path in (archive, manifest_path)), encoding="utf-8", newline="\n")
    return {"h0_tool_commit": resolved, "archive": str(archive), "manifest": str(manifest_path),
            "checksums": str(checksum), "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--commit", default="HEAD")
    args = parser.parse_args()
    print(json.dumps(build(args.repo, args.output_directory, args.commit), indent=2))


if __name__ == "__main__":
    main()
