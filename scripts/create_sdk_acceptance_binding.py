#!/usr/bin/env python3
"""Bind immutable candidate archives and actual replay/runtime inputs, without Git."""

import argparse
import json
import hashlib
import sys
import zipfile
from importlib import metadata
from pathlib import Path

from panorama_demo.sdk_acceptance import read, require, sha256, verify_orb_runtime


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def check_contents(root, expected_sha):
    root = Path(root)
    require(sha256(root / "checksums.sha256") == expected_sha, "Content checksum identity changed")
    covered = set()
    for line in (root / "checksums.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        require(sha256(root / relative) == digest, "Extracted content changed: " + relative)
        covered.add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == covered | {"checksums.sha256"}, "Extracted content has undeclared/missing files")


def session_identity(root):
    root = Path(root).resolve()
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}
    require(all(name in files for name in ("manifest.json", "calibration.json", "frames.csv")),
            "Frozen session is incomplete")
    return {"root": str(root), "files": files, "input_sha256": {
        "manifest": files["manifest.json"], "calibration": files["calibration.json"],
        "frames_csv": files["frames.csv"]}}


def installed_project_identity(wheel, source_commit):
    distribution = metadata.distribution("gemini305-rgbd-panorama")
    prefix = Path(sys.prefix).resolve()
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            path = Path(distribution.locate_file(name)).resolve()
            require(path.is_relative_to(prefix), "SDK imported outside installed prefix")
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            require(sha256(path) == expected, "Installed project wheel member changed: " + name)
            members[name] = expected
    source = Path(distribution.locate_file("panorama_demo/_runtime/source_commit.txt"))
    require(source.read_text().strip() == source_commit, "Installed SDK source commit mismatch")
    return {"prefix": str(prefix), "version": distribution.version, "source_commit": source_commit,
            "wheel_members_sha256": members}


def create_binding(candidate_index, archives, orb_manifest, frozen_session, platform_label,
                   *, base_root=None, addon_root=None):
    index = read(candidate_index)
    require(index.get("schema") == "gemini305-linux-sdk-release-candidate/v2",
            "Candidate index must bind real archive SHA")
    require(index.get("signature_status") == "UNSIGNED"
            and all(index.get(k) is False for k in ("software_ready", "hardware_qualified", "release_ready")),
            "Candidate qualification or signature is invalid")
    require(set(archives) == {"base", "three_d_addon", "source_compliance"}, "Three archives required")
    artifacts = {}
    for kind, value in archives.items():
        path = Path(value).resolve()
        actual, expected = file_identity(path), index["artifacts"][kind]
        require(path.name == expected["filename"] and actual["size"] == expected["size"]
                and actual["sha256"] == expected["archive_sha256"], "Archive binding mismatch: " + kind)
        artifacts[kind] = actual
    roots = {"base": Path(base_root) if base_root else Path(archives["base"]).with_suffix("").with_suffix(""),
             "three_d_addon": Path(addon_root) if addon_root else Path(archives["three_d_addon"]).with_suffix("").with_suffix("")}
    for kind, root in roots.items():
        check_contents(root, index["artifacts"][kind]["content_checksums_sha256"])
        manifest = read(root / "manifests/bundle-manifest.json")
        require(manifest["source_commit"] == index["source_commit"]
                and manifest["runtime_variant"] == index["runtime_variant"]
                and manifest["production_lock"] == index["production_lock"], "Bundle/source identity mismatch")
        require(all(manifest.get(k) is False for k in ("software_ready", "hardware_qualified", "release_ready")),
                "Frozen bundle may not issue qualification")
    wheel = roots["base"] / "wheels" / index["project_wheel_filename"]
    require(sha256(wheel) == index["project_wheel_sha256"], "Project wheel changed")
    require(read(roots["three_d_addon"] / "manifests/bundle-manifest.json")["project_wheel_sha256"] == sha256(wheel),
            "Addon references a different base project wheel")
    (open3d,) = (roots["three_d_addon"] / "wheels").glob("open3d-*.whl")
    require(sha256(open3d) == index["open3d_wheel_sha256"]
            and "0.19.0+1e7b17438" in open3d.name, "Private CUDA Open3D wheel identity mismatch")
    orb_manifest = Path(orb_manifest).resolve()
    portable_path = orb_manifest.parent / "orb-runtime-manifest.json"
    original, portable = read(orb_manifest), read(portable_path)
    files = dict(portable["files"])
    files["Vocabulary/ORBvoc.txt"] = original["artifacts"]["Vocabulary/ORBvoc.txt"]
    orb = {"manifest_path": str(orb_manifest), "manifest_sha256": sha256(orb_manifest),
           "portability_manifest_path": str(portable_path), "portability_manifest_sha256": sha256(portable_path),
           "root": str(orb_manifest.parent), "orbslam3_commit": original["orbslam3_commit"], "files": files}
    verify_orb_runtime(orb)
    lock = roots["base"] / "config/video_algorithms/s013_visual_continuity_v11_production.lock.json"
    require(read(lock) == index["production_lock"], "Production lock changed")
    return {"schema": "gemini305-sdk-acceptance-binding/v3", "candidate_index": file_identity(candidate_index),
            "source_commit": index["source_commit"], "runtime_variant": index["runtime_variant"],
            "platform": platform_label, "artifacts": artifacts,
            "extracted_roots": {k: str(v.resolve()) for k, v in roots.items()},
            "project_wheel": file_identity(wheel), "project_wheel_sha256": sha256(wheel),
            "installed_project": installed_project_identity(wheel, index["source_commit"]),
            "open3d_wheel": file_identity(open3d), "orb_runtime": orb,
            "production_lock": file_identity(lock), "production_identity": index["production_lock"],
            "frozen_session": session_identity(frozen_session), "signature_status": "UNSIGNED"}


def revalidate_binding(expected, candidate_index):
    actual = create_binding(candidate_index, {k: v["path"] for k, v in expected["artifacts"].items()},
        expected["orb_runtime"]["manifest_path"], expected["frozen_session"]["root"], expected["platform"],
        base_root=expected["extracted_roots"]["base"], addon_root=expected["extracted_roots"]["three_d_addon"])
    require(actual == expected, "Frozen binding changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate-index", "base-archive", "addon-archive", "source-compliance-archive",
                 "orb-runtime-manifest", "frozen-session", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--addon-root", type=Path)
    parser.add_argument("--platform-label", required=True)
    args = parser.parse_args()
    result = create_binding(args.candidate_index, {"base": args.base_archive,
        "three_d_addon": args.addon_archive, "source_compliance": args.source_compliance_archive},
        args.orb_runtime_manifest, args.frozen_session, args.platform_label,
        base_root=args.base_root, addon_root=args.addon_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)


if __name__ == "__main__":
    main()
