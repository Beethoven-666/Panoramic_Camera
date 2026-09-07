"""Tool integration tests with fake SDK boundaries; never hardware qualification."""

from contextlib import nullcontext
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from qualification.native_h0 import campaign, endurance_campaign
from panorama_demo.sdk_state import RetentionPolicy


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _binding():
    return {"h0_id": "test-only", "subject": {"source_commit": "a" * 40},
            "qualification_tool": {"commit": "b" * 40}, "camera_serial": "serial",
            "software_acceptance": {"path": "software.json"}, "frozen_session": {"root": "frozen"},
            "candidate": {"orb_runtime": {"root": "orb-runtime"}}}


def test_cli_error_writes_failure_and_restores_argv(tmp_path, monkeypatch):
    import panorama_demo.sdk_acceptance as acceptance
    import panorama_demo.video_panorama as cli
    monkeypatch.setattr(acceptance, "audit_2d_process", lambda *_: nullcontext())
    seen = []

    def failed_main():
        seen.extend(campaign.sys.argv)
        raise SystemExit("ERROR: actual CLI failure")

    monkeypatch.setattr(cli, "main", failed_main)
    original = campaign.sys.argv
    run = campaign.Campaign(tmp_path / "h0", _binding(), create=True)
    with pytest.raises(SystemExit):
        with run.operation("cli-crosscheck/run_01") as root:
            campaign.cli_replay(tmp_path / "session", root / "2d")
    assert campaign.sys.argv is original
    assert "--defer-3d" in seen
    assert json.loads((root / "operation_end.json").read_text())["status"] == "FAIL"
    campaign.verify_seal(root)


def test_capture_once_uses_public_sdk_config_and_exact_parameters(tmp_path, monkeypatch):
    import panorama_demo.sdk_acceptance as acceptance
    import qualification.native_h0.observer as observer
    monkeypatch.setattr(acceptance, "audit_2d_process", lambda *_: nullcontext())
    monkeypatch.setattr(observer, "NativeCaptureObserver", lambda *_: nullcontext())
    calls = []

    def capture_and_process(**kwargs):
        calls.append(kwargs)
        result = SimpleNamespace(session=tmp_path / "sessions/new", output_dir=tmp_path / "2d",
            job_id="job", formal_2d_published=True, overall_grade="A", manual_review_required=False)
        return SimpleNamespace(result=lambda: result, state=SimpleNamespace(value="COMPLETED"))

    sdk = SimpleNamespace(config=SimpleNamespace(preview_enabled=False,
                          retention_policy=RetentionPolicy.DELETE_AFTER_ALL_SUCCESS),
                          capture_and_process=capture_and_process)
    record = campaign.capture_once(sdk, tmp_path, exposure_us=None, duration_seconds=7200)
    assert record["preview_enabled"] is False  # Never fabricate requested capability.
    assert record["retention_policy"] == "delete_after_all_success"
    assert calls == [{"capture_root": tmp_path / "sessions", "output_dir": tmp_path / "2d",
                     "width": 848, "height": 480, "fps": 60, "duration_seconds": 7200,
                     "video_exposure_us": None, "preview_window": False, "wait_for_camera": True,
                     "maximum_post_seconds": 60}]


def test_native_sdk_config_matches_frozen_public_api(tmp_path):
    config = campaign.sdk_config(_binding(), tmp_path)
    assert config.preview_enabled is True
    assert config.retention_policy is RetentionPolicy.KEEP
    assert config.orb_runtime_kind == "native_linux"
    assert config.cuda_mode.value == "required"


def _campaign_fixture(tmp_path, monkeypatch):
    import panorama_demo.sdk_acceptance as acceptance
    import qualification.native_h0.capture_verifier as capture
    import qualification.native_h0.preview_verifier as preview
    import qualification.native_h0.equivalence as equality
    import qualification.native_h0.binding as binding_module
    binding = _binding()
    root = tmp_path / "h0"
    root.mkdir()
    monkeypatch.setattr(campaign, "Campaign", lambda *_: None)
    monkeypatch.setattr(campaign, "verify_operation", lambda r, b, name: Path(r) / name)
    monkeypatch.setattr(campaign, "verify_usb_snapshots", lambda *_: None)
    monkeypatch.setattr(binding_module, "stage4_reference_identity", lambda *_: {"test": True})
    monkeypatch.setattr(capture, "verify_native_capture_session", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(preview, "verify_native_preview", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(equality, "verify_native_equivalence", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(acceptance, "verify_doctor", lambda *a, **k: None)
    monkeypatch.setattr(acceptance, "verify_2d", lambda *a, **k: None)

    def three_d(path, **kwargs):
        if not (Path(path) / "actual-execution-test-marker").exists():
            raise ValueError("Missing real 3-D evidence")
        if not kwargs.get("failure"):
            expected_session = "frozen" if "native-replay" in Path(path).parts else str(root / "sdk-physical/run_05/session")
            assert kwargs["expected_binding"] == {"orb_runtime": binding["candidate"]["orb_runtime"],
                                                   "frozen_session": {"root": expected_session}}

    monkeypatch.setattr(acceptance, "verify_3d", three_d)
    _write(root / "profile.json", {"width": 848, "height": 480, "fps": 60, "max_sync_delta_us": 1000})
    replay = root / "preflight/native-replay"
    _write(replay / "replay_result.json", {"reference_2d": "reference"})
    _write(replay / "stage4_reference_identity.json", {"test": True})
    _write(replay / "post-3d/success/2d/3d/actual-execution-test-marker", {})
    reviews = {}
    for label in campaign.LABELS:
        path = root / "sdk-physical" / label
        output, session = path / "2d", path / "session"
        _write(path / "capture_result.json", {"session": str(session), "output": str(output),
            "duration_seconds": 3, "video_exposure_us": 100, "preview_enabled": True,
            "preview_window": False, "retention_policy": "keep", "formal_2d_published": True})
        _write(output / "video_report.json", {"grades": {"overall": "A"}})
        _write(output / "video_delivery.json", {"manual_review_required": False})
        _write(output / "video_timing.json", {"final_2d": {"capture_stop_to_p3_published_seconds": 1}})
        _write(output / "video_panorama.png", {})
        reviews[label] = {"accepted": True, "png_sha256": campaign.sha256(output / "video_panorama.png"),
                          "reviewed_utc": "2026-09-07T00:00:00Z"}
    for name in ("capture-contracts/fixed", "capture-contracts/auto", "capture-contracts/photo", "cli-live", "profile-smoke"):
        _write(root / name / "capture_result.json", {"session": str(root / name / "session")})
    _write(root / "post-3d/post_3d.json", {"session": str(root / "sdk-physical/run_05/session")})
    for name in ("success", "failure-isolation"):
        _write(root / "post-3d" / name / "2d/3d/actual-execution-test-marker", {})
    del reviews["warmup_01"]
    _write(root / "scene/reference.png", {})
    _write(root / "scene/h0_scene.json", {"h0_id": "test-only", "operator": "test",
        "rail_or_vehicle": "fixture", "movement_distance_m": 1, "target_speed_m_s": 1,
        "actual_speed_record": "fixture", "scan_direction": "left_to_right",
        "nearest_object_distance_m": .5, "lighting": "fixture", "camera_mount": "fixture",
        "usb_cable_port": "fixture", "sync_tolerance_approval": "fixture",
        "reference_images": ["scene/reference.png"]})
    _write(root / "scene/h0_visual_review.json", {"h0_id": "test-only", "operator": "test", "runs": reviews})
    return root, binding


@pytest.mark.parametrize("failure", ["performance_missing", "performance_failed", "quality", "3d", "none"])
def test_campaign_revalidates_performance_quality_and_3d(tmp_path, monkeypatch, failure):
    root, binding = _campaign_fixture(tmp_path, monkeypatch)
    output = root / "sdk-physical/run_05/2d"
    if failure == "performance_missing":
        (output / "video_timing.json").unlink()
    elif failure == "performance_failed":
        _write(output / "video_timing.json", {"final_2d": {"capture_stop_to_p3_published_seconds": 61}})
    elif failure == "quality":
        _write(output / "video_report.json", {"grades": {"overall": "C"}})
    elif failure == "3d":
        (root / "post-3d/success/2d/3d/actual-execution-test-marker").unlink()
    report = campaign.validate_campaign(root, binding)
    assert report["status"] == ("PASS" if failure == "none" else "FAIL")


@pytest.mark.parametrize("failure", ["short_device_span", "missing_2d", "reuse", "none"])
def test_endurance_reads_current_p0_contract_and_rejects_early_stop(tmp_path, monkeypatch, failure):
    import panorama_demo
    import panorama_demo.sdk_acceptance as acceptance
    config = SimpleNamespace()
    monkeypatch.setattr(endurance_campaign, "sdk_config", lambda *_: config)
    monkeypatch.setattr(panorama_demo, "Gemini305VideoSDK", lambda *_: SimpleNamespace(release_control=lambda: None))
    session, output = tmp_path / "session", tmp_path / "2d"
    session.mkdir()
    _write(output / "video_report.json", {"live_handoff": {
        "p0_prefix_reused_fraction": 0.1 if failure == "reuse" else 0.95,
        "full_m0_m3_recomputed": False}})
    with (session / "frames.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["color_device_timestamp_us"])
        writer.writerows([[0], [int((100 if failure == "short_device_span" else 1800) * 1e6)]])
    monkeypatch.setattr(endurance_campaign, "capture_once", lambda *a, **k: {
        "session": str(session), "output": str(output), "formal_2d_published": True})
    monkeypatch.setattr(endurance_campaign, "read_observation", lambda *_: iter([
        {"kind": "capture_started", "capture_started_monotonic_ns": 0},
        {"kind": "capture_stopped", "monotonic_ns": int(1900e9)}]))
    monkeypatch.setattr(endurance_campaign, "verify_native_capture_session", lambda *a, **k: {"status": "PASS"})
    monkeypatch.setattr(endurance_campaign, "verify_native_preview", lambda *a, **k: None)
    monkeypatch.setattr(endurance_campaign, "ensure_capacity", lambda *a: {"free_bytes": 10**15})
    monkeypatch.setattr(endurance_campaign, "command", lambda *_: {"returncode": 0, "stdout": "-- No entries --"})

    def formal_2d(*args):
        if failure == "missing_2d":
            raise ValueError("Formal 2D missing")

    monkeypatch.setattr(acceptance, "verify_2d", formal_2d)
    calibration = tmp_path / "calibration"
    _write(calibration / "storage_rate.json", {"h0_id": "test-only", "total_committed_bytes_per_second": 1})
    if failure == "none":
        result = endurance_campaign.run_endurance_session(tmp_path, _binding(), "continuous-30m",
                                                         {"max_sync_delta_us": 1000}, calibration)
        assert result["p0_reuse"]["p0_prefix_reused_fraction"] == .95
    else:
        with pytest.raises(ValueError):
            endurance_campaign.run_endurance_session(tmp_path, _binding(), "continuous-30m",
                                                    {"max_sync_delta_us": 1000}, calibration)


@pytest.mark.parametrize("failure", ["slow", "port", "reconnect", "missing", "none"])
def test_operation_usb_snapshots_are_bound_and_stable(tmp_path, failure):
    binding = {**_binding(), "usb_port_path": "1-2"}
    device = {"serial": "serial", "speed": "5000", "port_path": "1-2", "busnum": "1",
              "devnum": "2", "sysfs_realpath": "/sys/devices/pci/usb1/1-2"}
    for name in ("usb_before.json", "usb_after.json"):
        value = dict(device)
        if name == "usb_after.json":
            if failure == "slow":
                value["speed"] = "480"
            elif failure == "port":
                value["port_path"] = "1-3"
            elif failure == "reconnect":
                value["devnum"] = "3"
            elif failure == "missing":
                continue
        _write(tmp_path / name, {**campaign.binding_identity(binding), "devices": [value]})
    if failure == "none":
        campaign.verify_usb_snapshots(tmp_path, binding)
    else:
        with pytest.raises((ValueError, OSError)):
            campaign.verify_usb_snapshots(tmp_path, binding)
