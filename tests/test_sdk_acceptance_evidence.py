import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from panorama_demo import sdk_acceptance as acceptance


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def production(root):
    helper = load(
        "acceptance_fixture",
        Path(__file__).with_name("test_video_s13_live_acceptance.py"),
    )
    helper._publish(root)
    report = acceptance.read(root / "video_report.json")
    report["algorithm"]["config_sha256"] = (
        "3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc"
    )
    report["trajectory"] = {"direct_pose_count": 0, "pose_supported": False}
    (root / "video_report.json").write_text(json.dumps(report))
    (root / "process_audit.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-2d-process-audit/v1",
                "orb_calls": 0,
                "open3d_imports": 0,
                "three_d_calls": 0,
                "online_orb_constructions": 0,
                "completed": True,
            }
        )
    )


def test_canonical_evidence_reads_actual_owner_and_timing(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    production(left)
    production(right)
    result = acceptance.verify_2d(left, right)
    assert result["capture_stop_to_p3_published_seconds"] == 1.2
    path = right / "video_pixel_provenance.npz"
    with np.load(path) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["owner_frame_id"][0, 0] = 20
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="equivalence"):
        acceptance.verify_2d(left, right)


def test_changed_bundle_cannot_inherit_acceptance(tmp_path):
    wheel = tmp_path / "wheel"
    wheel.write_bytes(b"wheel")
    checksums = tmp_path / "checksums"
    checksums.write_bytes(b"first")
    old = acceptance.binding("a" * 40, wheel, checksums, "sm120", "native_linux")
    checksums.write_bytes(b"changed")
    new = acceptance.binding("a" * 40, wheel, checksums, "sm120", "native_linux")
    with pytest.raises(ValueError, match="binding mismatch"):
        acceptance.verify_binding({"binding": old}, new)


def test_extracted_candidate_tamper_and_extra_files_are_rejected(tmp_path):
    helper = load("binding_test", Path(__file__).resolve().parents[1] / "scripts/create_sdk_acceptance_binding.py")
    content = tmp_path / "payload"
    content.write_bytes(b"original")
    checksum = tmp_path / "checksums.sha256"
    checksum.write_text(acceptance.sha256(content) + "  payload\n")
    expected = acceptance.sha256(checksum)
    helper.check_contents(tmp_path, expected)
    content.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="content changed"):
        helper.check_contents(tmp_path, expected)
    content.write_bytes(b"original")
    (tmp_path / "unlisted.py").write_text("extra")
    with pytest.raises(ValueError, match="undeclared"):
        helper.check_contents(tmp_path, expected)


def test_doctor_pass_cannot_issue_software_or_h0(tmp_path, monkeypatch):
    aggregator = load(
        "aggregate_test",
        Path(__file__).resolve().parents[1] / "scripts/aggregate_sdk_acceptance.py",
    )
    (tmp_path / "binding.json").write_text(json.dumps({"binding": {}}))
    (tmp_path / "doctor.json").write_text(
        json.dumps(
            {"checks": {"camera": {"status": "PASS", "required_for_base": True}}}
        )
    )
    monkeypatch.setattr(aggregator, "native_host", lambda: False)
    with pytest.raises(ValueError, match="WSL"):
        aggregator.aggregate("native", tmp_path, {})
    # A camera/doctor record has no raw capture or canonical product evidence.
    with pytest.raises(ValueError, match="artifact is missing"):
        acceptance.verify_2d(tmp_path)


def test_2d_contract_is_independent_and_equivalence_rejects_self(tmp_path):
    tmp_path = tmp_path / "output"
    production(tmp_path)
    assert acceptance.verify_2d_contract(tmp_path)["status"] == "PASS"
    with pytest.raises(ValueError, match="self-comparison"):
        acceptance.verify_2d_equivalence(tmp_path, tmp_path / ".")
    report = acceptance.read(tmp_path / "video_report.json")
    del report["stage_pixel_sha256"]["P1"]
    (tmp_path / "video_report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="P0-P3"):
        acceptance.verify_2d_contract(tmp_path)


def test_full_software_doctor_accepts_no_camera_but_requires_addon(tmp_path):
    checks = {name: {"status": "PASS", "required_for_base": name == "python"}
              for name in ("python", "orbbec_wrapper", "open3d_cuda", "orbslam3_external_runtime")}
    checks["camera_device"] = {"status": "NOT_EXECUTED", "reason_code": "NO_CAMERA"}
    path = tmp_path / "doctor.json"
    path.write_text(json.dumps({"checks": checks}))
    assert acceptance.verify_doctor(path, "full_software")["checks"]["camera_device"]["reason_code"] == "NO_CAMERA"
    checks["open3d_cuda"]["status"] = "BLOCKED"
    path.write_text(json.dumps({"checks": checks}))
    with pytest.raises(ValueError, match="open3d_cuda"):
        acceptance.verify_doctor(path, "full_software")


def orb_fixture(root):
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = 20
    payload = {"backend": "orbslam3_rgbd_native_linux", "pose_convention": "camera_to_world",
        "translation_unit": "mm", "execution_attempts": [{"accepted": True, "returncode": 0}],
        "config": {"runtime_kind": "native_linux", "minimum_tracked_fraction": 0.95},
        "tracked_frame_ids": [1, 2], "input_frame_count": 2, "tracked_frame_count": 2,
        "tracked_fraction": 1.0, "poses": [{"frame_id": i + 1, "timestamp_us": (i + 1) * 1000000,
            "pose_kind": "direct_orbslam3", "pose_status": "valid", "camera_to_world": p.tolist()}
            for i, p in enumerate(poses)],
        "stdout_file": "stdout.txt", "stderr_file": "stderr.txt", "tum_file": "trajectory.txt",
        "association_file": "association.txt"}
    for name in ("stdout.txt", "stderr.txt", "association.txt"):
        (root / name).write_text("")
    (root / "trajectory.txt").write_text("1 0 0 0 0 0 0 1\n2 .02 0 0 0 0 0 1\n")
    (root / "orbslam3_trajectory.json").write_text(json.dumps(payload))
    return payload


@pytest.mark.parametrize("defect", ["nonrigid", "timestamp", "native", "synthetic", "tum"])
def test_native_orb_rejects_real_contract_defects(tmp_path, defect):
    payload = orb_fixture(tmp_path)
    assert acceptance.verify_native_orb(tmp_path)["tracked_poses"] == 2
    if defect == "nonrigid":
        # Determinant is still one: orthogonality must also be checked.
        payload["poses"][1]["camera_to_world"][0][1] = 0.1
    elif defect == "timestamp":
        payload["poses"][1]["timestamp_us"] = 1000000
    elif defect == "native":
        payload["backend"] = "orbslam3_rgbd_windows_wsl_legacy"
    elif defect == "synthetic":
        payload["poses"][1]["pose_kind"] = "interpolated"
    else:
        payload["poses"][1]["camera_to_world"][0][3] = 21
    (tmp_path / "orbslam3_trajectory.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        acceptance.verify_native_orb(tmp_path)


def test_process_audit_detects_heavy_call_and_preserves_failure(tmp_path):
    # The acceptance guard deliberately requires a fresh process. Other full-suite
    # tests exercise real Open3D, so verify this boundary in its actual process model.
    code = """
import sys
from pathlib import Path
from panorama_demo.sdk_acceptance import audit_2d_process
try:
    with audit_2d_process(Path(sys.argv[1])):
        __import__('open3d')
except RuntimeError as exc:
    assert 'Open3D' in str(exc)
else:
    raise AssertionError('heavy import accepted')
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, check=True)
    report = acceptance.read(tmp_path / "process_audit.json")
    assert report["open3d_imports"] == 1 and report["completed"] is False
