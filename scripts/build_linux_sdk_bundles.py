#!/usr/bin/env python3
"""Build development portability bundles from a clean source export and exact wheels."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import importlib.metadata
import os
import platform
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
ACCEPTANCE_TOOLS = (
    "run_linux_l1_benchmark.py", "summarize_linux_l1_benchmark.py",
    "aggregate_sdk_acceptance.py", "verify_linux_sdk_bundle.py",
    "verify_linux_sdk_archive.py", "create_sdk_acceptance_binding.py",
    "compare_linux_sdk_builds.py", "verify_s13_v11_live_equivalence.py",
)


def checksum_text(root, excluded=()):
    return "".join(sha(p) + "  " + p.relative_to(root).as_posix() + "\n"
                   for p in sorted(root.rglob("*"))
                   if p.is_file() and p.relative_to(root).as_posix() not in excluded)


def deterministic_archive(archive, entries, epoch):
    """One-thread zstd and canonical tar metadata; never overwrite an artifact."""
    import zstandard

    with (archive.open("xb") as handle,
          zstandard.ZstdCompressor(level=6, threads=0).stream_writer(handle) as stream,
          tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as tar):
        for path, name in sorted(entries, key=lambda pair: pair[1]):
            info = tar.gettarinfo(str(path), arcname=name)
            if not (info.isfile() or info.isdir()):
                raise ValueError("Only regular files and directories are distributed: " + name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = epoch
            info.mode = 0o755 if info.isdir() or path.suffix == ".sh" else 0o644
            info.pax_headers = {}
            if info.isfile():
                with path.open("rb") as source:
                    tar.addfile(info, source)
            else:
                tar.addfile(info)


def tree_entries(root, top):
    return [(root, top)] + [(p, top + "/" + p.relative_to(root).as_posix())
                           for p in root.rglob("*")
                           if "__pycache__" not in p.parts and
                           not any(x.endswith(".egg-info") for x in p.parts)]


def validate_descriptor(args):
    descriptor = json.loads((args.source / "packaging/linux/release-candidate.json").read_text())
    version_source = (args.source / "src/panorama_demo/version.py").read_text()
    if re.search(r'__version__\s*=\s*["\']([^"\']+)', version_source)[1] != descriptor["sdk_version"]:
        raise ValueError("Descriptor and package version mismatch")
    lock = json.loads((args.source / "configs/video_algorithms/s013_visual_continuity_v11_production.lock.json").read_text())
    config = (args.source / "configs/video_algorithms/s013_visual_continuity_v11_production.yaml").read_text()
    for key in ("algorithm_id", "config_sha256"):
        if lock[key] != descriptor["production_" + key]:
            raise ValueError("Production lock differs from candidate descriptor")
    if ("implementation_id: " + descriptor["production_implementation_id"]) not in config:
        raise ValueError("Production implementation differs from candidate descriptor")
    # Run the existing canonical config validator from the exported source without importing the SDK.
    import yaml
    parsed = yaml.safe_load(config)
    canonical = {key: value for key, value in parsed.items() if key != "config_sha256"}
    if hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest() != lock["config_sha256"]:
        raise ValueError("Production canonical config SHA mismatch")
    runtime = json.loads(args.runtime_manifest.read_text())
    for key in ("runtime_variant", "python_abi", "ubuntu", "architecture", "glibc_minimum", "gpu_compute_capability"):
        if runtime.get(key) != descriptor[key]:
            raise ValueError("Runtime manifest mismatch: " + key)
    for key in ("version", "source_commit", "wheel_sha256"):
        if runtime["open3d"][key] != descriptor["open3d_" + key]:
            raise ValueError("Private Open3D identity mismatch: " + key)
    if (runtime["orb"]["source_commit"] != descriptor["orb_source_commit"] or
            runtime["orb"]["pangolin_source_commit"] != descriptor["pangolin_source_commit"] or
            runtime["orb"].get("external_only") is not True):
        raise ValueError("External ORB Runtime identity mismatch")
    if sha(args.open3d_wheel) != descriptor["open3d_wheel_sha256"]:
        raise ValueError("Private Open3D wheel SHA mismatch")
    locked_wheels(args.source / "requirements/open3d-lock-py310.txt", [args.open3d_wheel])
    if "-cp310-cp310-" not in args.open3d_wheel.name:
        raise ValueError("Open3D wheel Python ABI mismatch")
    if descriptor["signature_status"] != "UNSIGNED" or descriptor["orb_distribution"] != "external_only":
        raise ValueError("This candidate must remain UNSIGNED with external ORB")
    return descriptor


def build_environment(args, commit, epoch):
    lock = args.source / "requirements/linux-build-lock-py310.txt"
    wheels = locked_wheels(lock, list(args.build_wheelhouse.glob("*.whl")))
    versions = {}
    for line in lock.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split(" --hash=")[0].split("==")
        if importlib.metadata.version(name) != version:
            raise ValueError("Build environment does not match exact build lock: " + name)
        versions[name] = version
    return {"python_executable": sys.executable, "python_version": sys.version,
            "versions": versions, "build_lock_sha256": sha(lock),
            "build_wheels": {p.name: sha(p) for p in wheels},
            "source_date_epoch": epoch, "source_commit": commit,
            "os": platform.platform(), "glibc": platform.libc_ver(),
            "architecture": platform.machine(), "builder_sha256": sha(Path(__file__))}


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
                    "sha256": hashlib.sha256(archive.read(name)).hexdigest(),
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
                    "checksums": [
                        {"algorithm": "SHA256", "checksumValue": item["sha256"]}
                    ],
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
    if kind == "base":
        (root / "acceptance-tools").mkdir()
        for name in ACCEPTANCE_TOOLS:
            shutil.copyfile(args.source / "scripts" / name, root / "acceptance-tools" / name)
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
        {"path": p.relative_to(args.source).as_posix(), "sha256": sha(p)}
        for p in sorted((args.source / "scripts/patches").glob("*.patch"))
    ]
    if kind == "base":
        patches.extend({"path": "acceptance-tools/" + name, "sha256": sha(root / "acceptance-tools" / name)}
                       for name in ACCEPTANCE_TOOLS)
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
    (root / "payload-checksums.sha256").write_text(checksum_text(root), encoding="utf-8", newline="\n")
    write_json(
        root / "manifests/bundle-manifest.json",
        {
            "schema": "gemini305-linux-runtime-bundle/v1",
            "kind": kind,
            "sdk_version": VERSION,
            "source_commit": source_commit,
            "runtime_variant": VARIANT,
            "project_wheel_sha256": sha(wheel),
            "content_checksums_sha256": sha(root / "payload-checksums.sha256"),
            "content_checksums_file": "payload-checksums.sha256",
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
    (root / "checksums.sha256").write_text(checksum_text(root), encoding="utf-8", newline="\n")
    sign(root / "checksums.sha256", args.signing_key, args.test_only_signature)
    return root


def main():
    global VERSION, VARIANT, OPEN3D_COMMIT, ORB_COMMIT, PANGOLIN_COMMIT
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "wheelhouse", "build-wheelhouse", "open3d-wheel", "runtime-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--signing-key")
    parser.add_argument("--test-only-signature", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 10):
        raise SystemExit("Build requires CPython 3.10")
    if args.signing_key or args.test_only_signature:
        raise ValueError("Phase 3 candidates must remain UNSIGNED")
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    if platform.system() != "Linux" or str(args.source).startswith("/mnt/") or str(args.output).startswith("/mnt/"):
        raise ValueError("Build requires a Linux ext4 checkout and output")
    source_commit = source_identity(args.source)
    epoch = int(os.environ["SOURCE_DATE_EPOCH"])
    if epoch < 315532800:
        raise ValueError("SOURCE_DATE_EPOCH must be ZIP-compatible (1980 or later)")
    descriptor = validate_descriptor(args)
    VERSION, VARIANT = descriptor["sdk_version"], descriptor["runtime_variant"]
    OPEN3D_COMMIT = descriptor["open3d_source_commit"]
    ORB_COMMIT = descriptor["orb_source_commit"]
    PANGOLIN_COMMIT = descriptor["pangolin_source_commit"]
    environment = build_environment(args, source_commit, epoch)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "build-environment.json", environment)
    env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    env.update(PYTHONHASHSEED="0", TZ="UTC")
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
            env=env,
        )
    (wheel,) = (args.output / "project-wheel").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        metadata = email.message_from_bytes(archive.read(next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))))
        if metadata["Version"] != VERSION or not wheel.name.endswith("-py3-none-any.whl"):
            raise ValueError("Built wheel metadata/version/ABI mismatch")
    roots = [bundle(args, kind, source_commit, wheel) for kind in ("base", "3d-addon")]
    artifacts = {}
    for key, root in zip(("base", "three_d_addon"), roots):
        archive_path = args.output / (root.name + ".tar.zst")
        deterministic_archive(archive_path, tree_entries(root, root.name), epoch)
        artifacts[key] = {"filename": archive_path.name, "size": archive_path.stat().st_size,
                          "archive_sha256": sha(archive_path),
                          "content_checksums_sha256": sha(root / "checksums.sha256")}
    archive = args.output / f"gemini305-sdk-source-compliance-{VERSION}.tar.zst"
    # A clean git export is the compliance source, including tests and all frozen tools.
    with tempfile.TemporaryDirectory(prefix="g305-compliance-") as temporary:
        stage = Path(temporary) / archive.name.removesuffix(".tar.zst")
        source = stage / "source"
        source.mkdir(parents=True)
        for name in (
            "src", "configs", "scripts", "packaging", "requirements", "tests",
            "artifacts/S013_M6_1_metrics_baseline_v2/threshold_approval.json",
            "docs", "pyproject.toml", "setup.py", "build_support.py", "MANIFEST.in",
            "README.md", "THIRD_PARTY_NOTICES.md", "CHANGELOG.md", "AGENTS.md",
        ):
            path = args.source / name
            if path.exists():
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if path.is_dir():
                    shutil.copytree(path, target, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
                else:
                    shutil.copyfile(path, target)
        (source / ".g305-source-commit").write_text(source_commit + "\n")
        for root in roots:
            shutil.copytree(root / "licenses", stage / root.name / "licenses")
            shutil.copyfile(root / "manifests/sbom.spdx.json", stage / root.name / "sbom.spdx.json")
        (stage / "checksums.sha256").write_text(checksum_text(stage), encoding="utf-8", newline="\n")
        deterministic_archive(archive, tree_entries(stage, stage.name), epoch)
    artifacts["source_compliance"] = {"filename": archive.name, "size": archive.stat().st_size,
                                      "archive_sha256": sha(archive)}
    wrapper = next(p for p in (roots[0] / "wheels").glob("pyorbbecsdk2-*.whl"))
    index = {"schema": "gemini305-linux-sdk-release-candidate/v2",
             "candidate_id": f"{epoch}-{source_commit[:12]}", "source_commit": source_commit,
             "sdk_version": VERSION, "runtime_variant": VARIANT,
             "project_wheel_filename": wheel.name, "project_wheel_size": wheel.stat().st_size,
             "project_wheel_sha256": sha(wheel), "artifacts": artifacts,
             "open3d_wheel_sha256": sha(args.open3d_wheel),
             "orbbec_wrapper_wheel_sha256": sha(wrapper),
             "production_lock": json.loads((roots[0] / "manifests/bundle-manifest.json").read_text())["production_lock"],
             "signature_status": "UNSIGNED", "bundle_frozen": False,
             "milestone": "RC_BUNDLE_BUILT_PENDING_VERIFICATION",
             "software_ready": False, "hardware_qualified": False, "release_ready": False}
    write_json(args.output / "release-candidate-index.json", index)
    shutil.copyfile(args.source / "scripts/verify_linux_sdk_archive.py", args.output / "verify_linux_sdk_archive.py")
    release_files = [args.output / item["filename"] for item in artifacts.values()]
    release_files += [args.output / "release-candidate-index.json", args.output / "verify_linux_sdk_archive.py"]
    (args.output / "release-candidate-checksums.sha256").write_text(
        "".join(sha(p) + "  " + p.name + "\n" for p in sorted(release_files)), encoding="utf-8", newline="\n")
    write_json(
        args.output / "build-result.json",
        {
            "source_commit": source_commit,
            "bundles": [str(p) for p in roots],
            "project_wheel": str(wheel),
            "project_wheel_sha256": sha(wheel),
            "source_compliance_archive": str(archive),
            "source_compliance_sha256": sha(archive),
            "artifacts": artifacts,
            "signature_status": "UNSIGNED",
            "software_ready": False,
            "hardware_qualified": False,
            "release_ready": False,
        },
    )


if __name__ == "__main__":
    main()
