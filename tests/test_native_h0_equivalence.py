import hashlib
import json

import numpy as np
import pytest

from qualification.native_h0.equivalence import verify_native_equivalence
from test_video_s13_live_acceptance import _publish


def _fixture(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    inputs = {}
    for key, name in (("manifest", "manifest.json"), ("calibration", "calibration.json"), ("frames_csv", "frames.csv")):
        (session / name).write_text("captured " + key)
        inputs[key] = hashlib.sha256((session / name).read_bytes()).hexdigest()
    roots = [tmp_path / name for name in ("sdk", "cli")]
    for root in roots:
        _publish(root)
        path = root / "video_report.json"
        report = json.loads(path.read_text())
        report["input_sha256"] = inputs
        report["algorithm"]["config_sha256"] = "3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc"
        report["trajectory"] = {"direct_pose_count": 0, "pose_supported": False}
        path.write_text(json.dumps(report))
        (root / "process_audit.json").write_text(json.dumps({
            "schema": "gemini305-2d-process-audit/v1", "completed": True,
            "orb_calls": 0, "open3d_imports": 0, "three_d_calls": 0, "online_orb_constructions": 0}))
    return session, roots[0], roots[1]


def test_exact_native_equivalence_passes(tmp_path):
    session, sdk, cli = _fixture(tmp_path)
    report = verify_native_equivalence(sdk, cli, session_root=session)
    assert report["status"] == "PASS"
    assert report["full_seam_transactions_exact"] is True


@pytest.mark.parametrize("failure", ["seam_field", "dtype", "stage", "input", "grade", "manual", "emergency"])
def test_all_native_exact_gates(tmp_path, failure):
    session, sdk, cli = _fixture(tmp_path)
    path = cli / "video_report.json"
    report = json.loads(path.read_text())
    if failure == "seam_field":
        # Existing canonical comparator omits this full transaction field.
        report["pair_seam_decisions"][0]["m5_transaction"]["audit_cost"] = 999
    elif failure == "dtype":
        archive = cli / "video_pixel_provenance.npz"
        with np.load(archive) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["owner_frame_id"] = arrays["owner_frame_id"].astype(np.int64)
        np.savez_compressed(archive, **arrays)
    elif failure == "stage":
        report["stage_pixel_sha256"]["P1"] = "f" * 64
    elif failure == "input":
        (session / "manifest.json").write_text("different session")
    elif failure == "grade":
        report["grades"]["overall"] = "C"
    elif failure == "manual":
        report["manual_review_required"] = True
    else:
        (cli / "emergency_delivery.json").write_text("{}")
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        verify_native_equivalence(sdk, cli, session_root=session, require_standard_quality=True)


def test_no_self_comparison(tmp_path):
    session, sdk, cli = _fixture(tmp_path)
    with pytest.raises(ValueError, match="self-comparison"):
        verify_native_equivalence(sdk, sdk, session_root=session)
