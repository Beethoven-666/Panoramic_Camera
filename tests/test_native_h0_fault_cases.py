import json

import pytest

from qualification.native_h0.fault_cases import FAULT_CASES, fault_errors, operator_checkpoint, validate_fault_campaign, validate_loopback_mount


def test_no_wait_observed_absence_error_and_release():
    record = {"execution_kind":"native_physical", "wait_for_camera":False,
              "error_code":"CAMERA_NOT_FOUND", "elapsed_seconds":.3}
    events = [{"event":"device_absent", "monotonic_seconds":1},
              {"event":"lease_released", "monotonic_seconds":1.3}]
    assert fault_errors("no-wait", record, events) == []
    record["formal_delivery"] = "fake.json"
    assert fault_errors("no-wait", record, events)


@pytest.mark.parametrize("case", FAULT_CASES)
def test_pass_boolean_without_actual_events_never_qualifies(case):
    assert fault_errors(case, {"status":"PASS", "passed":True}, [])


def test_operator_abort_and_confirmation_are_only_actions(tmp_path):
    path = tmp_path / "events.jsonl"
    with pytest.raises(RuntimeError):
        operator_checkpoint(path, "Unplug", confirm=lambda _: "ABORT")
    event = json.loads(path.read_text())
    assert event["event"] == "operator_checkpoint"
    assert event["answer"] == "ABORT"


def test_system_disk_never_allowed_for_lowdisk(monkeypatch, tmp_path):
    monkeypatch.setattr("qualification.native_h0.fault_cases.command", lambda _: {"stdout":
        '{"filesystems":[{"source":"/dev/nvme0n1p1","target":"/","fstype":"ext4"}]}'} )
    with pytest.raises(RuntimeError, match="loopback"):
        validate_loopback_mount(tmp_path)


def test_all_eight_missing_faults_block(tmp_path):
    assert len(validate_fault_campaign(tmp_path, {})["errors"]) == 8


@pytest.mark.parametrize("busy", [True, False])
def test_lease_child_uses_public_sdk_capture_not_primitive_lock(monkeypatch, tmp_path, busy):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import panorama_demo
    from panorama_demo.sdk_state import SDKBusyError
    from qualification.native_h0.fault_cases import lease_probe
    calls = []
    session = tmp_path / "session"
    session.mkdir()
    (session / "manifest.json").write_text('{"written_frames":180}')
    class FakeSDK:
        def __init__(self, config):
            pass
        def start_capture(self, *args, **kwargs):
            calls.append((args, kwargs))
            if busy:
                raise SDKBusyError("controlled")
            result = SimpleNamespace(session=session, output_dir=tmp_path / "2d", formal_2d_published=True)
            return SimpleNamespace(result=lambda: result, state=SimpleNamespace(value="COMPLETED"))
        def release_control(self):
            calls.append("released")
    monkeypatch.setattr(panorama_demo, "Gemini305VideoSDK", FakeSDK)
    monkeypatch.setattr("qualification.native_h0.observer.NativeCaptureObserver", lambda _: nullcontext())
    result = lease_probe(tmp_path, object(), "serial")
    assert calls[0][1]["fps"] == 60
    assert calls[-1] == "released"
    assert result["sdk_method"] == "start_capture"
    assert result["event"] == ("lease_busy" if busy else "lease_acquired")
