"""External identity gates reject changed raw inputs and old qualification schemas."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualification.native_h0 import binding
from qualification.native_h0.common import sha256


@pytest.mark.parametrize("change,accepted", [
    ({}, True),
    ({"virtualization": {"stdout": "none", "returncode": 1}}, False),
    ({"virtualization": {"stdout": "kvm", "returncode": 0}}, False),
    ({"virtualization": {"stdout": "vmware", "returncode": 1}}, False),
    ({"virtualization": {"status": "NOT_EXECUTED"}}, False),
    ({"kernel": "microsoft-standard-WSL2"}, False),
    ({"container_markers": ["/.dockerenv"]}, False),
    ({"os_release": {"ID": "ubuntu", "VERSION_ID": "24.04"}}, False),
    ({"effective_uid": 0}, False),
    ({"python_version": [3, 12, 1]}, False),
])
def test_binding_execution_host_accepts_only_vmware_profile(monkeypatch, change, accepted):
    host = {"system": "Linux", "architecture": "x86_64", "python_implementation": "CPython",
            "python_version": [3, 10, 12], "effective_uid": 1000,
            "os_release": {"ID": "ubuntu", "VERSION_ID": "22.04"},
            "virtualization": {"stdout": "vmware\n", "returncode": 0},
            "kernel": "5.15.0-generic", "container_markers": []}
    host.update(change)
    monkeypatch.setattr(binding, "collect_host", lambda: host)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delenv("PYTHONHOME", raising=False)
    if accepted:
        assert binding.verify_execution_host() == host
    else:
        with pytest.raises(ValueError, match="required"):
            binding.verify_execution_host()


def test_old_baremetal_binding_requires_fresh_vmware_campaign():
    with pytest.raises(ValueError, match="Fresh VMware"):
        binding.revalidate_binding({"schema": "gemini305-native-h0-binding/v1",
                                    "platform": "NATIVE_UBUNTU_22_04"}, "unused", "unused")


def test_old_tools_manifest_cannot_be_reused_for_vmware(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "gemini305-native-h0-tools/v1", "h0_tool_commit": "a" * 40,
        "requires_python": ">=3.10,<3.11", "native_acceptance_schema": "gemini305-sdk-native-acceptance/v3",
        "signature_status": "UNSIGNED"}))
    with pytest.raises(ValueError, match="manifest contract"):
        binding.verify_tools(tmp_path / "unused", manifest)


def test_extracted_content_mutation_and_extra_file_rejected(tmp_path):
    content = tmp_path / "runtime.txt"
    content.write_text("frozen")
    checksums = tmp_path / "checksums.sha256"
    checksums.write_text(sha256(content) + "  runtime.txt\n")
    expected = sha256(checksums)
    binding.check_contents(tmp_path, expected)
    content.write_text("changed")
    with pytest.raises(ValueError, match="content changed"):
        binding.check_contents(tmp_path, expected)
    content.write_text("frozen")
    (tmp_path / "extra.txt").write_text("unexpected runtime")
    with pytest.raises(ValueError, match="undeclared"):
        binding.check_contents(tmp_path, expected)


def software_fixture(tmp_path):
    candidate = {"artifacts": {kind: {"sha256": value} for kind, value in binding.ARCHIVE_SHA.items()},
                 "production_identity": {"algorithm": "locked"},
                 "orb_runtime": {"manifest_sha256": "manifest", "portability_manifest_sha256": "portable",
                                 "orbslam3_commit": "orb", "files": {"executable": {"sha256": "elf"}}}}
    for key in ("candidate_index", "project_wheel", "open3d_wheel", "production_lock"):
        candidate[key] = {"sha256": key}
    old = dict(candidate, schema="gemini305-sdk-acceptance-binding/v3", source_commit=binding.SUBJECT_COMMIT,
               runtime_variant=binding.RUNTIME_VARIANT, project_wheel_sha256=binding.PROJECT_WHEEL_SHA)
    report = {"schema": "gemini305-sdk-software-acceptance/v3", "status": "PASS", "milestone": "SDK_SOFTWARE_READY",
              "software_ready": True, "hardware_qualified": False, "release_ready": False,
              "signature_status": "UNSIGNED", "binding": old}
    path = tmp_path / "software.json"
    path.write_text(json.dumps(report))
    return candidate, report, path


@pytest.mark.parametrize("key", ["candidate_index", "project_wheel", "open3d_wheel", "production_lock"])
def test_software_rejects_other_candidate_identity(tmp_path, key):
    candidate, report, path = software_fixture(tmp_path)
    binding.verify_software(path, candidate)
    report["binding"][key] = {"sha256": "different"}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Software identity mismatch"):
        binding.verify_software(path, candidate)


@pytest.mark.parametrize("key", ["manifest_sha256", "portability_manifest_sha256", "orbslam3_commit", "files"])
def test_software_rejects_other_orb_runtime(tmp_path, key):
    candidate, report, path = software_fixture(tmp_path)
    report = json.loads(json.dumps(report))
    report["binding"]["orb_runtime"][key] = "different"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Software ORB mismatch"):
        binding.verify_software(path, candidate)


def test_software_rejects_legacy_native_status(tmp_path):
    candidate, report, path = software_fixture(tmp_path)
    report["schema"] = "gemini305-sdk-acceptance-status/v2"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="qualification"):
        binding.verify_software(path, candidate)


def tools_fixture(tmp_path):
    if not shutil.which("zstd"):
        pytest.skip("tools archive verification requires the offline host's zstd utility")
    tools = tmp_path / "tools"
    tools.mkdir()
    name = "qualification/native_h0/example.py"
    source = tools / name
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# committed tool\n")
    manifest = {"schema": "gemini305-native-h0-tools/v1", "h0_tool_commit": "a" * 40,
                "requires_python": ">=3.10,<3.11", "native_acceptance_schema": "gemini305-sdk-native-acceptance/v3",
                "qualification_environment": "VMWARE_UBUNTU_22_04", "bare_metal_qualified": False,
                "signature_status": "UNSIGNED", "files": {name: sha256(source)}}
    manifest_path = tmp_path / "native-h0-tools-manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    archive = tmp_path / "tools.tar.zst"
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as tar:
        for entry_name, value in ((name, source.read_bytes()), (manifest_path.name, manifest_path.read_bytes())):
            entry = tarfile.TarInfo(entry_name)
            entry.size = len(value)
            tar.addfile(entry, io.BytesIO(value))
    result = subprocess.run(["zstd", "-q", "-c"], input=raw.getvalue(), capture_output=True, check=True)
    archive.write_bytes(result.stdout)
    (tmp_path / "native-h0-tools-checksums.sha256").write_text(
        "".join(sha256(p) + "  " + p.name + "\n" for p in (archive, manifest_path)))
    return archive, manifest_path, tools, source


def test_tools_archive_and_deployed_payload_both_checked(tmp_path):
    archive, manifest, tools, source = tools_fixture(tmp_path)
    result = binding.verify_tools(archive, manifest, tools)
    assert result["commit"] != binding.SUBJECT_COMMIT
    source.write_text("# modified after extraction")
    with pytest.raises(ValueError, match="Deployed H0 tool changed"):
        binding.verify_tools(archive, manifest, tools)


def test_tools_external_checksum_changed_rejected(tmp_path):
    archive, manifest, tools, source = tools_fixture(tmp_path)
    checksums = tmp_path / "native-h0-tools-checksums.sha256"
    checksums.write_text(checksums.read_text().replace(sha256(archive), hashlib.sha256(b"wrong").hexdigest()))
    with pytest.raises(ValueError, match="external archive/manifest"):
        binding.verify_tools(archive, manifest, tools)


def test_stage4_reference_relocation_and_stale_report(tmp_path):
    software_root = tmp_path / "relocated-software"
    suite = software_root / "linux/cli-2d"
    reference = suite / "run_01/2d"
    reference.mkdir(parents=True)
    old_binding = {"source_commit": binding.SUBJECT_COMMIT}
    for path in (software_root / "binding.json", suite / "binding.json"):
        path.write_text(json.dumps(old_binding))
    (suite / "suite.json").write_text(json.dumps({"binding": old_binding, "operation": "frozen-2d", "runs": 5, "warmups": 1}))
    json_files = ("video_report.json", "video_delivery.json", "video_timing.json")
    artifacts = {name: {"original": name} for name in json_files}
    for name, value in artifacts.items():
        (reference / name).write_text(json.dumps(value))
    for name in ("video_panorama.png", "video_panorama.jpg", "video_pixel_provenance.npz", "process_audit.json"):
        (reference / name).write_bytes(b"original")
    sample = {"product_output": "/old/software/linux/cli-2d/run_01/2d", "exit_code": 0, "warmup": False, "artifacts": artifacts}
    (reference.parent / "sample.json").write_text(json.dumps(sample))
    software = tmp_path / "software.json"
    software.write_text(json.dumps({"binding": old_binding, "raw_run_directories": ["/old/software"]}))
    result = binding.stage4_reference_identity(reference, software)
    assert len(result["files"]) == 7
    (reference / "video_report.json").write_text('{"changed": true}')
    with pytest.raises(ValueError, match="reference output changed"):
        binding.stage4_reference_identity(reference, software)


@pytest.mark.parametrize("mutation", [None, "pth", "base-member", "open3d-member"])
def test_installer_separate_addon_uses_verified_base_pth(tmp_path, monkeypatch, mutation):
    module_path = Path(__file__).resolve().parents[1] / "scripts/native_h0_addon_identity.py"
    spec = importlib.util.spec_from_file_location("h0_addon_identity_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base, addon = tmp_path / "base/venv", tmp_path / "base/addon/venv"
    base_lib, addon_lib = base / "site-packages", addon / "site-packages"
    project_name, open3d_name = "panorama_demo/__init__.py", "open3d/__init__.py"
    for root, name in ((base_lib, project_name), (addon_lib, open3d_name)):
        (root / name).parent.mkdir(parents=True)
        (root / name).write_bytes(b"frozen")
    project_wheel, open3d_wheel = tmp_path / "project.whl", tmp_path / "open3d.whl"
    for wheel, name in ((project_wheel, project_name), (open3d_wheel, open3d_name)):
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(name, b"frozen")
    link = addon_lib / "g305_base.pth"
    link.write_text(str(base_lib.resolve()) + "\n")
    def distribution(name):
        root = base_lib if name == "gemini305-rgbd-panorama" else addon_lib
        return SimpleNamespace(version="test", read_text=lambda key: None, locate_file=lambda name: root / name)
    monkeypatch.setattr(binding.metadata, "distribution", distribution)
    monkeypatch.setattr(sys, "prefix", str(addon))
    monkeypatch.setattr(module.sysconfig, "get_paths", lambda: {"purelib": str(addon_lib)})
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(base_lib / project_name)))
    if mutation == "pth":
        link.write_text(str(tmp_path / "unrelated-checkout"))
    elif mutation == "base-member":
        (base_lib / project_name).write_bytes(b"modified")
    elif mutation == "open3d-member":
        (addon_lib / open3d_name).write_bytes(b"modified")
    if mutation:
        with pytest.raises(ValueError, match="changed|pth"):
            module.inspect(open3d_wheel, project_wheel, base)
    else:
        result = module.inspect(open3d_wheel, project_wheel, base)
        assert result["prefix"] == str(addon.resolve())
        assert result["installed_project"]["prefix"] == str(base.resolve())
        assert result["base_pth"]["sha256"] == sha256(link)
