#!/usr/bin/env python3
"""Build development portability bundles from a clean source export and exact wheels."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
import zipfile

VARIANT = "ubuntu22.04-x86_64-py310-sm120"
VERSION = "0.3.0rc1"
OPEN3D_COMMIT = "1e7b17438687a0b0c1e5a7187321ac7044afe275"
ORB_COMMIT = "4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4"
PANGOLIN_COMMIT = "aff6883c83f3fd7e8268a9715e84266c42e2efe3"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def source_identity(root):
    if (root / ".git").exists():
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip():
            raise ValueError("Refusing a dirty build")
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    else:
        value = (root / ".g305-source-commit").read_text().strip()
    if not re.fullmatch("[0-9a-f]{40}", value):
        raise ValueError("Source commit must identify a clean git archive")
    return value


def locked_wheels(lock, available):
    selected = []
    for line in lock.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        requirement, digest = line.split(" --hash=sha256:")
        name, version = requirement.split("==")
        normalized = name.lower().replace("-", "_")
        matches = [
            p
            for p in available
            if p.name.split("-")[0].lower().replace("-", "_") == normalized
            and p.name.split("-")[1] == version
        ]
        if len(matches) != 1 or sha(matches[0]) != digest:
            raise ValueError("Exact wheel absent, duplicate or corrupt: " + requirement)
        selected.append(matches[0])
    return selected


def inspect_wheel(path, licenses):
    elfs = []
    with zipfile.ZipFile(path) as archive:
        metadata = email.message_from_bytes(
            archive.read(
                next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            )
        )
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            if "license" in name.lower() or "notice" in name.lower():
                target = licenses / path.stem / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
            with archive.open(name) as handle:
                header = handle.read(4)
            if header != b"\x7fELF":
                continue
            with tempfile.NamedTemporaryFile() as handle:
                handle.write(archive.read(name))
                handle.flush()
                dynamic = subprocess.check_output(
                    ["readelf", "-d", handle.name], text=True
                )
                file_type = subprocess.check_output(
                    ["file", "-b", handle.name], text=True
                ).strip()
            rpaths = re.findall(r"\((?:RPATH|RUNPATH)\).*?\[(.*?)\]", dynamic)
            if any("/home/zyh/build" in p or "/tmp/" in p for p in rpaths):
                raise ValueError("Wheel ELF retains build RPATH: " + name)
            elfs.append(
                {
                    "wheel": path.name,
                    "member": name,
                    "file": file_type,
                    "rpath": rpaths,
                    "needed": re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic),
                    "readelf_dynamic": dynamic,
                }
            )
        return {
            "name": metadata["Name"],
            "version": metadata["Version"],
            "wheel": path.name,
            "sha256": sha(path),
            "size": path.stat().st_size,
            "requires_dist": metadata.get_all("Requires-Dist", []),
            "license_expression": metadata.get("License-Expression") or "NOASSERTION",
            "upstream_license_metadata": metadata.get("License"),
            "elf": elfs,
        }


def spdx(dependencies, patches, commit, kind):
    packages = []
    for index, dep in enumerate(dependencies):
        packages.append(
            {
                "SPDXID": f"SPDXRef-Package-{index}",
                "name": dep["name"],
                "versionInfo": dep["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": dep["license_expression"],
                "checksums": [{"algorithm": "SHA256", "checksumValue": dep["sha256"]}],
                "packageFileName": "wheels/" + dep["wheel"],
            }
        )
    for name, version in [
        ("Open3D-source", OPEN3D_COMMIT),
        ("ORB-SLAM3-external-source", ORB_COMMIT),
        ("Pangolin-external-source", PANGOLIN_COMMIT),
    ]:
        packages.append(
            {
                "SPDXID": "SPDXRef-" + name,
                "name": name,
                "versionInfo": version,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseDeclared": "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "comment": "Source identity only; external ORB/Pangolin binaries are not bundled.",
            }
        )
    files = []
    for dep in dependencies:
        for item in dep["elf"]:
            files.append(
                {
                    "SPDXID": f"SPDXRef-ELF-{len(files)}",
                    "fileName": dep["wheel"] + "!/" + item["member"],
                    "fileTypes": ["BINARY"],
                    "licenseConcluded": "NOASSERTION",
                    "copyrightText": "NOASSERTION",
                }
            )
    for patch in patches:
        files.append(
            {
                "SPDXID": f"SPDXRef-Patch-{len(files)}",
                "fileName": patch["path"],
                "checksums": [
                    {"algorithm": "SHA256", "checksumValue": patch["sha256"]}
                ],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"gemini305-{kind}-{VERSION}",
        "documentNamespace": f"https://spdx.org/spdxdocs/gemini305-{commit}-{kind}",
        "creationInfo": {
            "created": "2026-09-07T00:00:00Z",
            "creators": ["Tool: build_linux_sdk_bundles.py"],
        },
        "packages": packages,
        "files": files,
    }


def sign(checksums, key, test_only=False):
    if not key:
        return
    # Only an existing key id from the caller's GPG keyring is accepted. No key is generated or copied.
    if Path(key).is_file():
        raise ValueError(
            "Import private keys outside this tool; pass an existing GPG key id"
        )
    subprocess.run(
        [
            "gpg",
            "--batch",
            "--local-user",
            key,
            "--armor",
            "--detach-sign",
            "--output",
            str(checksums) + (".TEST_ONLY.asc" if test_only else ".asc"),
            str(checksums),
        ],
        check=True,
    )


def bundle(args, kind, source_commit, wheel):
    root = args.output / f"gemini305-sdk-{kind}-{VERSION}-{VARIANT}"
    root.mkdir(parents=True, exist_ok=False)
    for name in ("wheels", "requirements", "config", "manifests", "docs", "licenses"):
        (root / name).mkdir()
    for path in (args.source / "packaging/linux").iterdir():
        if path.is_file():
            shutil.copy2(path, root / path.name)
        elif path.name == "orbbec-official":
            shutil.copytree(path, root / path.name)
    for path in (args.source / "docs").glob("*.md"):
        shutil.copy2(path, root / "docs" / path.name)
    for name in ("README.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md"):
        if (args.source / name).exists():
            shutil.copy2(args.source / name, root / name)
    for name in (
        "demo.yaml",
        "video_algorithms/s013_visual_continuity_v11_production.lock.json",
        "video_algorithms/s013_visual_continuity_v11_production.yaml",
    ):
        path = args.source / "configs" / name
        if path.exists():
            target = root / "config" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    lock_names = (
        ["addon-lock-py310", "open3d-lock-py310"]
        if kind == "3d-addon"
        else ["linux-runtime-lock-py310", "orbbec-wrapper-lock-py310"]
    )
    available = list(args.wheelhouse.rglob("*.whl")) + [args.open3d_wheel]
    selected = []
    for name in lock_names:
        lock = args.source / "requirements" / (name + ".txt")
        selected.extend(locked_wheels(lock, available))
        shutil.copy2(lock, root / "requirements" / lock.name)
    if kind == "base":
        selected.append(wheel)
        (root / "requirements/project-lock.txt").write_text(
            f"gemini305-rgbd-panorama=={VERSION} --hash=sha256:{sha(wheel)}\n"
        )
    dependencies = []
    for path in selected:
        shutil.copy2(path, root / "wheels" / path.name)
        dependencies.append(inspect_wheel(path, root / "licenses"))
    if kind == "base" and any(d["name"].lower() == "open3d" for d in dependencies):
        raise ValueError("Open3D leaked into base")
    patches = [
        {"path": str(p.relative_to(args.source)), "sha256": sha(p)}
        for p in (args.source / "scripts/patches").glob("*.patch")
    ]
    variant = json.loads(args.runtime_manifest.read_text())
    variant.update(
        {
            "runtime_variant": VARIANT,
            "ubuntu": "22.04",
            "architecture": "x86_64",
            "python_abi": "cp310",
            "requires_python": ">=3.10,<3.11",
            "glibc_minimum": "2.35",
            "gpu_compute_capability": "12.0",
            "orb": {
                "source_commit": ORB_COMMIT,
                "pangolin_source_commit": PANGOLIN_COMMIT,
                "external_only": True,
                "distribution_license_status": "BLOCKED",
                "bundled_binary": False,
                "bundled_vocabulary": False,
            },
        }
    )
    write_json(root / "manifests/runtime-variant-manifest.json", variant)
    write_json(root / "manifests/dependency-manifest.json", dependencies)
    write_json(
        root / "manifests/elf-dependency-report.json",
        {
            "scope": "all bundled wheel ELF dynamic tables",
            "elf": [e for d in dependencies for e in d["elf"]],
            "system_dependencies": variant["system_dependencies"],
            "runtime_resolution_validation": "clean-prefix doctor and CLI/2D/addon smokes; ORB scanned separately",
        },
    )
    write_json(
        root / "manifests/sbom.spdx.json",
        spdx(dependencies, patches, source_commit, kind),
    )
    identity = json.loads(
        (
            args.source
            / "configs/video_algorithms/s013_visual_continuity_v11_production.lock.json"
        ).read_text()
    )
    write_json(
        root / "manifests/bundle-manifest.json",
        {
            "schema": "gemini305-linux-runtime-bundle/v1",
            "kind": kind,
            "sdk_version": VERSION,
            "source_commit": source_commit,
            "runtime_variant": VARIANT,
            "project_wheel_sha256": sha(wheel),
            "production_lock": identity,
            "development_portability_only": True,
            "software_ready": False,
            "hardware_qualified": False,
            "release_ready": False,
            "project_license": "BLOCKED_OWNER_TEXT_NOT_PROVIDED",
            "signature_status": "TEST_ONLY"
            if args.signing_key and args.test_only_signature
            else ("DETACHED_UNQUALIFIED" if args.signing_key else "UNSIGNED"),
            "files": [
                {"path": p.relative_to(root).as_posix(), "size": p.stat().st_size}
                for p in sorted(root.rglob("*"))
                if p.is_file()
            ],
        },
    )
    (root / "checksums.sha256").write_text(
        "".join(
            sha(p) + "  " + p.relative_to(root).as_posix() + "\n"
            for p in sorted(root.rglob("*"))
            if p.is_file()
        )
    )
    sign(root / "checksums.sha256", args.signing_key, args.test_only_signature)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "wheelhouse", "open3d-wheel", "runtime-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--signing-key")
    parser.add_argument("--test-only-signature", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Build requires CPython 3.10")
    source_commit = source_identity(args.source)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "build.log").open("xb") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(args.output / "project-wheel"),
                str(args.source),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    (wheel,) = (args.output / "project-wheel").glob("*.whl")
    roots = [bundle(args, kind, source_commit, wheel) for kind in ("base", "3d-addon")]
    import zstandard

    archive = args.output / f"gemini305-sdk-source-compliance-{VERSION}.tar.zst"
    with (
        archive.open("xb") as handle,
        zstandard.ZstdCompressor(level=6).stream_writer(handle) as stream,
        tarfile.open(fileobj=stream, mode="w|") as tar,
    ):
        for name in (
            "src",
            "configs",
            "scripts",
            "packaging",
            "requirements",
            "docs",
            "pyproject.toml",
            "setup.py",
            "build_support.py",
            "MANIFEST.in",
            "README.md",
            "THIRD_PARTY_NOTICES.md",
            "CHANGELOG.md",
            ".g305-source-commit",
        ):
            path = args.source / name
            if path.exists():
                tar.add(path, arcname="source/" + name)
        for root in roots:
            tar.add(root / "licenses", arcname=root.name + "/licenses")
            tar.add(
                root / "manifests/sbom.spdx.json", arcname=root.name + "/sbom.spdx.json"
            )
    write_json(
        args.output / "build-result.json",
        {
            "source_commit": source_commit,
            "bundles": [str(p) for p in roots],
            "project_wheel": str(wheel),
            "project_wheel_sha256": sha(wheel),
            "source_compliance_archive": str(archive),
            "source_compliance_sha256": sha(archive),
            "software_ready": False,
            "hardware_qualified": False,
            "release_ready": False,
        },
    )


if __name__ == "__main__":
    main()
