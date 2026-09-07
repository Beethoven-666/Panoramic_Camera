"""Rebind immutable phase 3/4 subjects to independently committed H0 tools.

No panorama_demo import occurs before all installed project wheel members have
been compared with the frozen wheel. Cross-host comparisons ignore only paths.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
import importlib.metadata as metadata
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import tarfile
import zipfile

from .common import file_identity, read, require, sha256

SUBJECT_COMMIT = "2abb132043e7c5ae1805a2ed1dd91516ebf2f874"
RUNTIME_VARIANT = "ubuntu22.04-x86_64-py310-sm120"
ARCHIVE_SHA = {
    "base": "5e488e8f7145a9aec9da13998a01e507dfac818b9f04d2b1bbbcc72df4eb18e0",
    "three_d_addon": "59db7a1a90178afb2605763510762f3e917707476501f72971ac4594b5fad0c4",
    "source_compliance": "bcc60c197b65f145480f6d89d5d415fc444881a97ef7d607512f61c868cc93b5",
}
PROJECT_WHEEL_SHA = "c99b58a389821a6146cc77866c1c631fdc81875f07f940a2edb4b1724ea00cc8"


def check_contents(root, expected_sha):
    root = Path(root).resolve()
    checksums = root / "checksums.sha256"
    require(sha256(checksums) == expected_sha, "Content checksum identity changed")
    covered = set()
    for line in checksums.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        require(relative not in covered, "Duplicate content checksum: " + relative)
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Content path escapes bundle")
        require(sha256(path) == digest, "Extracted content changed: " + relative)
        covered.add(relative)
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    require(actual == covered | {"checksums.sha256"}, "Extracted bundle has undeclared/missing files")


def installed_wheel_identity(wheel, distribution_name, *, allowed_prefix=None):
    distribution = metadata.distribution(distribution_name)
    prefix = Path(allowed_prefix or sys.prefix).resolve()
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        import json
        require(not json.loads(direct_url).get("dir_info", {}).get("editable"), "Editable install forbidden")
    members = {}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                continue
            path = Path(distribution.locate_file(name)).resolve()
            require(path.is_relative_to(prefix), "Wheel member is outside installed prefix: " + name)
            expected = hashlib.sha256(archive.read(name)).hexdigest()
            require(sha256(path) == expected, "Installed wheel member changed: " + name)
            members[name] = expected
    return {"prefix": str(prefix), "version": distribution.version, "wheel_members_sha256": members}


def installed_addon_identity(candidate, base_identity):
    """Inspect the installer's separate addon venv and supported base .pth link."""
    import importlib.util
    import json

    spec = importlib.util.find_spec("panorama_demo")
    require(spec and Path(spec.origin).resolve().is_relative_to(Path(base_identity["prefix"])),
            "SDK resolver must come from the verified base installation")
    from panorama_demo.sdk_runtime import addon_python
    interpreter = addon_python()
    require(interpreter is not None, "Frozen 3-D addon is not installed")
    worker = Path(__file__).resolve().parents[2] / "scripts/native_h0_addon_identity.py"
    completed = subprocess.run([str(interpreter), str(worker),
        "--open3d-wheel", candidate["open3d_wheel"]["path"],
        "--project-wheel", candidate["project_wheel"]["path"],
        "--base-prefix", base_identity["prefix"]], capture_output=True, text=True, check=False)
    require(completed.returncode == 0, "Installed addon member validation failed: " + completed.stderr)
    result = json.loads(completed.stdout)
    require(result["installed_project"] == base_identity, "Addon sees a different base SDK")
    result["addon_python"] = str(interpreter)
    marker = Path(sys.prefix).parent / "addon/addon-install-manifest.json"
    if marker.exists():
        result["install_manifest"] = file_identity(marker)
    return result


def orb_identity(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    portable_path = manifest_path.parent / "orb-runtime-manifest.json"
    original, portable = read(manifest_path), read(portable_path)
    require(original["orbslam3_commit"] == portable["orbslam3_commit"], "ORB source commits differ")
    files = dict(portable["files"])
    files["Vocabulary/ORBvoc.txt"] = original["artifacts"]["Vocabulary/ORBvoc.txt"]
    require("Examples/RGB-D/rgbd_tum_headless" in files, "ORB executable is not bound")
    for name, record in files.items():
        path = (manifest_path.parent / name).resolve()
        require(path.is_relative_to(manifest_path.parent), "ORB member escaped Runtime")
        require(sha256(path) == record["sha256"], "ORB Runtime member changed: " + name)
    executable = manifest_path.parent / "Examples/RGB-D/rgbd_tum_headless"
    with executable.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", "ORB executable is not native ELF")
    return {"manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "portability_manifest_path": str(portable_path), "portability_manifest_sha256": sha256(portable_path),
            "root": str(manifest_path.parent), "orbslam3_commit": original["orbslam3_commit"], "files": files}


def verify_candidate(candidate_index, phase3_freeze, archives, base_root, addon_root, orb_manifest):
    """Read-only identity audit, usable on a staging host; issues no qualification."""
    index = read(candidate_index)
    require(index.get("schema") == "gemini305-linux-sdk-release-candidate/v2", "Wrong candidate schema")
    require(index.get("source_commit") == SUBJECT_COMMIT and index.get("sdk_version") == "0.3.0rc1"
            and index.get("runtime_variant") == RUNTIME_VARIANT, "Wrong candidate identity")
    require(index.get("signature_status") == "UNSIGNED"
            and all(index.get(k) is False for k in ("software_ready", "hardware_qualified", "release_ready")),
            "Candidate must remain unsigned and unqualified")
    freeze = read(phase3_freeze)
    require(freeze.get("status") == "PASS" and freeze.get("bundle_frozen") is True
            and freeze.get("milestone") == "RC_BUNDLE_FROZEN" and freeze.get("build") == "build_a"
            and freeze.get("candidate_index_sha256") == sha256(candidate_index)
            and freeze.get("source_commit") == SUBJECT_COMMIT
            and freeze.get("project_wheel_sha256") == PROJECT_WHEEL_SHA
            and freeze.get("archives") == index["artifacts"], "Phase 3 freeze mismatch")
    require(set(archives) == set(ARCHIVE_SHA), "All three frozen archives are required")
    identities = {}
    for kind, path in archives.items():
        item = file_identity(path)
        expected = index["artifacts"][kind]
        require(item["sha256"] == ARCHIVE_SHA[kind] == expected["archive_sha256"]
                and item["size"] == expected["size"] and Path(path).name == expected["filename"],
                "Frozen archive mismatch: " + kind)
        identities[kind] = item
    roots = {"base": Path(base_root).resolve(), "three_d_addon": Path(addon_root).resolve()}
    for kind, root in roots.items():
        check_contents(root, index["artifacts"][kind]["content_checksums_sha256"])
        manifest = read(root / "manifests/bundle-manifest.json")
        require(manifest["source_commit"] == SUBJECT_COMMIT and manifest["runtime_variant"] == RUNTIME_VARIANT
                and manifest["production_lock"] == index["production_lock"], "Extracted manifest identity mismatch")
        require(all(manifest.get(k) is False for k in ("software_ready", "hardware_qualified", "release_ready")),
                "Frozen bundle issued qualification")
    wheel = roots["base"] / "wheels" / index["project_wheel_filename"]
    require(sha256(wheel) == PROJECT_WHEEL_SHA == index["project_wheel_sha256"], "Project wheel mismatch")
    require(read(roots["three_d_addon"] / "manifests/bundle-manifest.json")["project_wheel_sha256"] == PROJECT_WHEEL_SHA,
            "Addon targets another project wheel")
    open3d = list((roots["three_d_addon"] / "wheels").glob("open3d-*.whl"))
    require(len(open3d) == 1, "Exactly one private Open3D wheel required")
    require("0.19.0+1e7b17438" in open3d[0].name and sha256(open3d[0]) == index["open3d_wheel_sha256"],
            "Private Open3D wheel mismatch")
    lock = roots["base"] / "config/video_algorithms/s013_visual_continuity_v11_production.lock.json"
    require(read(lock) == index["production_lock"], "Production lock mismatch")
    return {"candidate_index": file_identity(candidate_index), "phase3_freeze": file_identity(phase3_freeze),
            "artifacts": identities, "extracted_roots": {k: str(v) for k, v in roots.items()},
            "project_wheel": file_identity(wheel), "open3d_wheel": file_identity(open3d[0]),
            "orb_runtime": orb_identity(orb_manifest), "production_lock": file_identity(lock),
            "production_identity": index["production_lock"]}


def verify_software(software_acceptance, candidate):
    report = read(software_acceptance)
    require(report.get("schema") == "gemini305-sdk-software-acceptance/v3"
            and report.get("status") == "PASS" and report.get("milestone") == "SDK_SOFTWARE_READY"
            and report.get("software_ready") is True and report.get("hardware_qualified") is False
            and report.get("release_ready") is False and report.get("signature_status") == "UNSIGNED",
            "Phase 4 software qualification is absent or invalid")
    old = report["binding"]
    require(old.get("schema") == "gemini305-sdk-acceptance-binding/v3"
            and old.get("source_commit") == SUBJECT_COMMIT and old.get("runtime_variant") == RUNTIME_VARIANT,
            "Software binding candidate mismatch")
    for kind, item in candidate["artifacts"].items():
        require(old["artifacts"][kind]["sha256"] == item["sha256"], "Software archive mismatch: " + kind)
    for key in ("candidate_index", "project_wheel", "open3d_wheel", "production_lock"):
        require(old[key]["sha256"] == candidate[key]["sha256"], "Software identity mismatch: " + key)
    require(old.get("project_wheel_sha256") == PROJECT_WHEEL_SHA
            and old["production_identity"] == candidate["production_identity"], "Software production mismatch")
    for key in ("manifest_sha256", "portability_manifest_sha256", "orbslam3_commit", "files"):
        require(old["orb_runtime"][key] == candidate["orb_runtime"][key], "Software ORB mismatch: " + key)
    return file_identity(software_acceptance)


def stage4_reference_identity(reference_2d, software_acceptance):
    """Bind a relocated, retained formal phase 4 sample to its original suite."""
    reference = Path(reference_2d).resolve()
    sample_root, suite_root = reference.parent, reference.parent.parent
    require(reference.name == "2d" and sample_root.name in {f"run_{i:02d}" for i in range(1, 6)}
            and suite_root.name in {"cli-2d", "sdk-2d"} and suite_root.parent.name == "linux",
            "Reference must be a retained formal phase 4 2-D sample")
    software_root = suite_root.parent.parent
    software = read(software_acceptance)
    require(read(software_root / "binding.json") == software["binding"]
            and read(suite_root / "binding.json") == software["binding"], "Reference phase 4 binding mismatch")
    suite, sample = read(suite_root / "suite.json"), read(sample_root / "sample.json")
    require(suite["binding"] == software["binding"] and suite["operation"] == "frozen-2d"
            and suite["runs"] == 5 and suite["warmups"] == 1, "Reference was not in accepted phase 4 suite")
    original_roots = software["raw_run_directories"]
    require(len(original_roots) == 1 and sample["product_output"] ==
            str(PurePosixPath(original_roots[0]) / reference.relative_to(software_root).as_posix()),
            "Reference sample does not match software qualification raw root")
    require(sample["exit_code"] == 0 and sample["warmup"] is False, "Reference run did not complete")
    for name in ("video_delivery.json", "video_report.json", "video_timing.json"):
        require(read(reference / name) == sample["artifacts"][name], "Phase 4 reference output changed: " + name)
    names = ("video_delivery.json", "video_report.json", "video_timing.json", "video_panorama.png",
             "video_panorama.jpg", "video_pixel_provenance.npz", "process_audit.json")
    return {"root": str(reference), "files": {name: sha256(reference / name) for name in names},
            "software_acceptance": file_identity(software_acceptance),
            "sample": file_identity(sample_root / "sample.json"), "suite": file_identity(suite_root / "suite.json")}


@contextmanager
def decompressed_archive(path):
    """Use the offline host's zstd utility; no new Python/runtime dependency."""
    process = subprocess.Popen(["zstd", "-q", "-d", "-c", str(path)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        yield process.stdout
        process.stdout.close()
        stderr = process.stderr.read().decode(errors="replace")
        require(process.wait() == 0, "H0 tools zstd decompression failed: " + stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def verify_tools(archive_path, manifest_path, tools_root=None):
    """Verify archive bytes against manifest members and deployed tool contents."""

    manifest = read(manifest_path)
    commit = manifest.get("h0_tool_commit", "")
    require(manifest.get("schema") == "gemini305-native-h0-tools/v1" and len(commit) == 40
            and all(c in "0123456789abcdef" for c in commit) and commit != SUBJECT_COMMIT,
            "Independent H0 tool commit required")
    require(manifest.get("signature_status") == "UNSIGNED" and manifest.get("requires_python") == ">=3.10,<3.11"
            and manifest.get("native_acceptance_schema") == "gemini305-sdk-native-acceptance/v3",
            "H0 tool manifest contract mismatch")
    members = manifest["files"]
    require(members and all(name.startswith(("qualification/native_h0/", "scripts/"))
                           or name == "docs/NATIVE_UBUNTU_H0.md" for name in members), "Unexpected tool payload")
    actual = {}
    with decompressed_archive(archive_path) as decoded:
        with tarfile.open(fileobj=decoded, mode="r|") as tar:
            for member in tar:
                require(member.isfile() and member.name not in actual, "Non-file or duplicate tools entry")
                actual[member.name] = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
    expected = dict(members)
    expected["native-h0-tools-manifest.json"] = sha256(manifest_path)
    require(actual == expected, "H0 archive member inventory changed")
    root = Path(tools_root).resolve() if tools_root else Path(__file__).resolve().parents[2]
    for name, digest in members.items():
        require(sha256(root / name) == digest, "Deployed H0 tool changed: " + name)
    checksums = Path(archive_path).parent / "native-h0-tools-checksums.sha256"
    declared = {}
    for line in checksums.read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(name not in declared, "Duplicate tool archive checksum")
        declared[name] = digest
    require(declared == {Path(archive_path).name: sha256(archive_path), Path(manifest_path).name: sha256(manifest_path)},
            "H0 external archive/manifest checksums mismatch")
    return {"commit": commit, "archive": file_identity(archive_path), "archive_sha256": sha256(archive_path),
            "manifest": file_identity(manifest_path), "manifest_sha256": sha256(manifest_path),
            "checksums": file_identity(checksums), "root": str(root)}


def create_binding(candidate_index, software_acceptance, phase3_freeze, archives, base_root, addon_root,
                   orb_manifest, frozen_session, tools_archive, tools_manifest,
                   host_inventory=None, camera_inventory=None, *, h0_id, usb_inventory=None):
    require(platform.system() == "Linux" and platform.python_implementation() == "CPython"
            and sys.version_info[:2] == (3, 10) and os.geteuid() != 0, "Native non-root CPython 3.10 required")
    require(not os.environ.get("PYTHONPATH") and not os.environ.get("PYTHONHOME"), "PYTHONPATH/PYTHONHOME forbidden")
    release = platform.freedesktop_os_release()
    virtual = subprocess.run(["systemd-detect-virt"], capture_output=True, text=True, check=False)
    require(release.get("ID") == "ubuntu" and release.get("VERSION_ID") == "22.04"
            and platform.machine() == "x86_64" and virtual.stdout.strip() == "none"
            and virtual.returncode == 1 and "microsoft" not in platform.release().lower(),
            "Bare-metal Ubuntu 22.04 x86_64 required")
    require(isinstance(h0_id, str) and bool(h0_id.strip()), "Unique H0 ID required")
    candidate = verify_candidate(candidate_index, phase3_freeze, archives, base_root, addon_root, orb_manifest)
    software = verify_software(software_acceptance, candidate)
    installed = installed_wheel_identity(candidate["project_wheel"]["path"], "gemini305-rgbd-panorama")
    require(installed["version"] == "0.3.0rc1", "Installed SDK version changed")
    tool = verify_tools(tools_archive, tools_manifest)
    installed_open3d = installed_addon_identity(candidate, installed)
    # Safe only after immutable installed member validation above.
    from panorama_demo.sdk_acceptance import verify_orb_runtime
    verify_orb_runtime(candidate["orb_runtime"])
    inventories = {name: file_identity(path) for name, path in
                   (("host", host_inventory), ("camera", camera_inventory), ("usb", usb_inventory)) if path is not None}
    host = read(host_inventory) if host_inventory else {}
    if host:
        require(host.get("machine_id") == Path("/etc/machine-id").read_text().strip(), "H0 evidence belongs to another host")
    camera = read(camera_inventory) if camera_inventory else {}
    usb = read(usb_inventory) if usb_inventory else {}
    serial = camera.get("device", {}).get("serial_number")
    matched_usb = [item for item in usb.get("devices", []) if item.get("serial") == serial]
    files = {p.relative_to(frozen_session).as_posix(): sha256(p)
             for p in sorted(Path(frozen_session).rglob("*")) if p.is_file()}
    require(files == read(software_acceptance)["binding"]["frozen_session"]["files"], "Native replay input changed")
    return {"schema": "gemini305-native-h0-binding/v1", "h0_id": h0_id,
        "subject_source_commit": SUBJECT_COMMIT, "h0_tool_commit": tool["commit"], "subject": {
        "source_commit": SUBJECT_COMMIT, "sdk_version": "0.3.0rc1", "runtime_variant": RUNTIME_VARIANT,
        "base_archive_sha256": ARCHIVE_SHA["base"], "addon_archive_sha256": ARCHIVE_SHA["three_d_addon"],
        "source_archive_sha256": ARCHIVE_SHA["source_compliance"], "project_wheel_sha256": PROJECT_WHEEL_SHA,
        "candidate_index_sha256": candidate["candidate_index"]["sha256"],
        "software_acceptance_status_sha256": software["sha256"]}, "qualification_tool": tool,
        "runtime": {"open3d_wheel_sha256": candidate["open3d_wheel"]["sha256"],
                    "orb_runtime_manifest_sha256": candidate["orb_runtime"]["manifest_sha256"],
                    "production_lock_sha256": candidate["production_lock"]["sha256"]},
        "candidate": candidate, "software_acceptance": software, "installed_project": installed,
        "installed_open3d": installed_open3d,
        "frozen_session": {"root": str(Path(frozen_session).resolve()), "files": files},
        "host": host, "camera": camera,
        "inventory_inputs": inventories, "camera_serial": serial,
        "usb_port_path": matched_usb[0].get("port_path") if len(matched_usb) == 1 else None,
        "udev_rule_sha256": camera.get("udev_rule", {}).get("sha256"),
        "platform": "NATIVE_UBUNTU_22_04", "signature_status": "UNSIGNED"}


def revalidate_binding(expected, candidate_index, software_acceptance):
    require(expected.get("schema") == "gemini305-native-h0-binding/v1", "Native binding v1 required")
    candidate, tool = expected["candidate"], expected["qualification_tool"]
    inventories = expected.get("inventory_inputs", {})
    paths = {name: record["path"] for name, record in inventories.items()}
    actual = create_binding(candidate_index, software_acceptance, candidate["phase3_freeze"]["path"],
        {k: v["path"] for k, v in candidate["artifacts"].items()}, candidate["extracted_roots"]["base"],
        candidate["extracted_roots"]["three_d_addon"], candidate["orb_runtime"]["manifest_path"],
        Path(expected["frozen_session"]["root"]), tool["archive"]["path"], tool["manifest"]["path"],
        paths.get("host"), paths.get("camera"), h0_id=expected["h0_id"], usb_inventory=paths.get("usb"))
    require(actual == expected, "Native binding or its raw subject/tool inputs changed")
    return actual


def finalize_binding(raw_root):
    """Seal host/camera identity after the bootstrap's real recorded replay."""
    from .campaign import Campaign, verify_operation
    from .inventory import validate_inventory
    from .common import write_new

    root = Path(raw_root).resolve()
    bootstrap = read(root / "bootstrap_binding.json")
    Campaign(root, bootstrap)
    verify_operation(root, bootstrap, "preflight/native-replay")
    require(not bootstrap.get("inventory_inputs"), "Bootstrap must precede camera inventory")
    candidate, tool = bootstrap["candidate"], bootstrap["qualification_tool"]
    actual = create_binding(candidate["candidate_index"]["path"], bootstrap["software_acceptance"]["path"],
        candidate["phase3_freeze"]["path"], {k: v["path"] for k, v in candidate["artifacts"].items()},
        candidate["extracted_roots"]["base"], candidate["extracted_roots"]["three_d_addon"],
        candidate["orb_runtime"]["manifest_path"], Path(bootstrap["frozen_session"]["root"]),
        tool["archive"]["path"], tool["manifest"]["path"], root / "preflight/host_inventory.json",
        root / "preflight/camera_inventory.json", h0_id=bootstrap["h0_id"], usb_inventory=root / "preflight/usb_inventory.json")
    for key in ("subject", "qualification_tool", "candidate", "runtime", "installed_project", "installed_open3d", "frozen_session"):
        require(actual[key] == bootstrap[key], "Bootstrap subject/runtime changed before inventory: " + key)
    result = validate_inventory(root, actual)
    require(not result["errors"], "Cannot finalize native identity: " + str(result["errors"]))
    write_new(root / "native_h0_binding.json", actual)
    return actual
