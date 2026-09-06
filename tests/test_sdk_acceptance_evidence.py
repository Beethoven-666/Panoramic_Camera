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
    with pytest.raises(OSError):
        acceptance.verify_2d(tmp_path)


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
