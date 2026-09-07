import csv
import json

import cv2
import numpy as np
import pytest

from qualification.native_h0.capture_verifier import verify_native_capture_session


def _fixture(tmp_path, monkeypatch):
    # Strict loader has its own candidate tests; these fixtures exercise the
    # additional H0 gates, decoding real RGB/Y16 files without any camera claim.
    import panorama_demo.video_session
    monkeypatch.setattr(panorama_demo.video_session, "load_video_session", lambda *a, **k: None)
    root = tmp_path / "session"
    root.mkdir()
    (root / "color").mkdir()
    (root / "depth_aligned").mkdir()
    shutdown = {"completed": True, "readback_verified": True,
                "after": {"mode": "STANDALONE", "trigger_out_enable": False}}
    manifest = {"schema": "panorama-demo-session/v2", "capture_mode": "continuous_rgbd_video_fixed_exposure",
                "device": {"name": "Gemini 305", "serial_number": "SERIAL", "firmware_version": "1.0.70"},
                "python_wrapper_version": "2.1.2+g305.1", "sdk_version": "2.9.3",
                "capture_options": {"trigger_to_image_delay_us": 8000, "trigger_out_delay_us": 7000},
                "profiles": {kind: {"width": 848, "height": 480, "fps": 60, "format": fmt}
                             for kind, fmt in (("color", "RGB"), ("depth", "Y16"))},
                "written_frames": 3, "received_frames": 3, "queue_drops": 0, "write_errors": 0,
                "timestamp_regressions": 0, "writer_errors": [], "clean_shutdown": True,
                "product_eligibility": {"photo_panorama": False, "video_panorama": True},
                "capture_runtime_audit": {key: 0 for key in (
                    "online_orb_tracker_construct_count", "capture_orbslam3_process_count",
                    "capture_orbslam3_frame_submit_count", "capture_open3d_import_count",
                    "capture_open3d_tsdf_call_count", "capture_3d_process_count")},
                "external_sync_output": {"per_frame_readback_verified_frames": 3, "shutdown": shutdown},
                "color_exposure_control": {"requested_exposure_us": 100},
                "color_control_lock": {"completed": True, "readback_verified": True, "warmup_frame_sets": 3, "locked_controls": {
                    "auto_exposure": False, "auto_white_balance": False, "exposure_raw": 1,
                    "gain_raw": 10, "white_balance_raw": 5000}}}
    rows, events = [], [{"kind": "lease_acquired", "serial": "SERIAL", "held": True, "monotonic_ns": 1}]
    for index in range(3):
        color, depth = f"color/{index}.jpg", f"depth_aligned/{index}.png"
        cv2.imwrite(str(root / color), np.zeros((480, 848, 3), np.uint8))
        cv2.imwrite(str(root / depth), np.ones((480, 848), np.uint16))
        row = {"frame_id": index, "color_device_timestamp_us": 100000 + index * 16667,
               "depth_device_timestamp_us": 100000 + index * 16667, "sync_delta_us": 0,
               "color_exposure": 1, "color_gain": 10, "color_white_balance": 5000,
               "color_path": color, "aligned_depth_path": depth}
        rows.append(row)
        events.append({"kind": "sync_readback", "monotonic_ns": 100 + 2 * index,
                       "readback": {"mode": "PRIMARY", "trigger_out_enable": True,
                                    "trigger_to_image_delay_us": 8000, "trigger_out_delay_us": 7000,
                                    "color_delay_us": 8000, "depth_delay_us": 8000}})
        events.append({"kind": "frame_submitted", "monotonic_ns": 101 + 2 * index,
                       "session": str(root), "frame_id": index, "accepted": True, "metadata": row.copy()})
    events.extend([{"kind": "hardware_shutdown", "monotonic_ns": 200, "pipeline_was_started": True,
                    "errors": [], "shutdown": shutdown},
                   {"kind": "writer_closed", "monotonic_ns": 250, "written_frames": 3,
                    "queue_drops": 0, "write_errors": 0, "writer_errors": [], "queue_depth": 0,
                    "worker_alive": False},
                   {"kind": "lease_released", "serial": "SERIAL", "held": False, "monotonic_ns": 300}])
    return root, manifest, rows, events


def _write(tmp_path, root, manifest, rows, events):
    (root / "manifest.json").write_text(json.dumps(manifest))
    with (root / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = tmp_path / "observation.json"
    path.with_suffix(".jsonl").write_text("".join(json.dumps({**e, "sequence": i}) + "\n"
                                                   for i, e in enumerate(events, 1)))
    path.write_text(json.dumps({"schema": "gemini305-native-h0-observation/v1", "completed": True,
                              "event_count": len(events), "events_path": str(path.with_suffix(".jsonl")),
                              "observer_errors": []}))
    return path


def test_native_capture_actual_rows_and_files(tmp_path, monkeypatch):
    args = _fixture(tmp_path, monkeypatch)
    observation = _write(tmp_path, *args)
    result = verify_native_capture_session(args[0], expected_serial="SERIAL", max_sync_delta_us=1000,
                                          observation_path=observation)
    assert result["status"] == "PASS"
    assert result["effective_fps"] == pytest.approx(60, abs=0.01)


@pytest.mark.parametrize("failure", ["sync_readback", "queue_drops", "missing_counter", "false_counter",
                                     "fps", "timestamp", "sync_delta", "exposure", "frame_id",
                                     "file", "cleanup", "lease", "reconnect", "observed_frame"])
def test_native_capture_rejects_incomplete_actual_evidence(tmp_path, monkeypatch, failure):
    root, manifest, rows, events = _fixture(tmp_path, monkeypatch)
    if failure == "sync_readback":
        manifest["external_sync_output"]["per_frame_readback_verified_frames"] = 2
    elif failure == "queue_drops":
        manifest["queue_drops"] = 1
    elif failure == "missing_counter":
        del manifest["write_errors"]
    elif failure == "false_counter":
        manifest["write_errors"] = False
    elif failure == "fps":
        rows[-1]["color_device_timestamp_us"] = rows[-1]["depth_device_timestamp_us"] = 200000
    elif failure == "timestamp":
        rows[-1]["color_device_timestamp_us"] = rows[-1]["depth_device_timestamp_us"] = 1
    elif failure == "sync_delta":
        rows[0]["sync_delta_us"] = 1001
    elif failure == "exposure":
        rows[0]["color_exposure"] = 8
    elif failure == "frame_id":
        rows[1]["frame_id"] = 0
    elif failure == "file":
        (root / rows[0]["aligned_depth_path"]).write_bytes(b"broken")
    elif failure == "cleanup":
        events[-3]["errors"] = [{"step": "pipeline.stop"}]
    elif failure == "lease":
        events[-1]["held"] = True
    elif failure == "reconnect":
        events.insert(3, {"kind": "device_discovered", "monotonic_ns": 102})
    else:
        events[2]["metadata"]["color_exposure"] = 2
    path = _write(tmp_path, root, manifest, rows, events)
    with pytest.raises(ValueError):
        verify_native_capture_session(root, expected_serial="SERIAL", max_sync_delta_us=1000,
                                      observation_path=path)


def test_observer_keeps_actual_writer_and_lease_semantics(tmp_path):
    from panorama_demo.camera_lease import CameraLease
    from panorama_demo.capture_orbbec import FramePacket, SessionWriter
    from qualification.native_h0.observer import NativeCaptureObserver, read_observation

    root = tmp_path / "writer"
    root.mkdir()
    original_submit = SessionWriter.submit
    path = tmp_path / "observed.json"
    with NativeCaptureObserver(path) as observer:
        with CameraLease(tmp_path, "test-serial", "unit-test"):
            writer = SessionWriter(root, 4, 90, 1, False, durable_per_frame=False)
            assert writer.submit(FramePacket(0, np.zeros((2, 2, 3), np.uint8),
                                            np.ones((2, 2), np.uint16), None,
                                            {"color_device_timestamp_us": 10, "depth_scale_mm_per_unit": 1}))
            writer.close()
            snapshot = observer.snapshot_runtime()
            assert snapshot["committed_frame_count"] == 1
            assert snapshot["lease_owned"] is True
            assert snapshot["capture_active"] is False
    assert SessionWriter.submit is original_submit
    events = list(read_observation(path))
    assert any(e["kind"] == "writer_closed" and e["worker_alive"] is False for e in events)
    assert any(e["kind"] == "lease_released" and e["held"] is False for e in events)


def test_observer_preserves_typed_exception_and_restores_hooks(tmp_path):
    from panorama_demo.camera_lease import CameraLease
    from panorama_demo.sdk_state import CaptureError
    from qualification.native_h0.observer import NativeCaptureObserver, read_observation

    original = CameraLease.acquire
    path = tmp_path / "failed.json"
    with pytest.raises(CaptureError):
        with NativeCaptureObserver(path):
            CameraLease(tmp_path / "absent-lock-root", "serial", "unit-test").acquire()
    assert CameraLease.acquire is original
    events = list(read_observation(path))
    assert any(e["kind"] == "boundary_error" and e["error_code"] == "CAMERA_LOCK_ROOT_UNAVAILABLE"
               for e in events)


def test_auto_contract_requires_unrequested_exposure_and_full_lock(tmp_path, monkeypatch):
    root, manifest, rows, events = _fixture(tmp_path, monkeypatch)
    manifest["capture_mode"] = "continuous_rgbd_video_auto"
    manifest["color_control_policy"] = "auto_warmup_then_full_manual_lock"
    path = _write(tmp_path, root, manifest, rows, events)
    assert verify_native_capture_session(root, mode="auto", requested_exposure_us=None,
                                        expected_serial="SERIAL", max_sync_delta_us=1000,
                                        observation_path=path)["status"] == "PASS"
    with pytest.raises(ValueError, match="must not request"):
        verify_native_capture_session(root, mode="auto", requested_exposure_us=100,
                                      expected_serial="SERIAL", max_sync_delta_us=1000,
                                      observation_path=path)


def test_photo_contract_uses_triggered_pair_and_actual_shutdown(tmp_path, monkeypatch):
    import panorama_demo.session
    monkeypatch.setattr(panorama_demo.session, "load_rgbd_session", lambda *a, **k: None)
    root, manifest, rows, events = _fixture(tmp_path, monkeypatch)
    manifest.update(schema="panorama-demo-session/v1", capture_mode="software_triggered_rgbd_photo_sequence",
                    formal_stitch_allowed=True, one_formal_trigger_per_capture=True,
                    software_trigger={"per_frame_readback_verified_frames": 3})
    for event in events:
        if event["kind"] == "sync_readback":
            event["readback"]["mode"] = "SOFTWARE_TRIGGERING"
    events = [event for event in events if event["kind"] != "hardware_shutdown"]
    events[-1:-1] = [{"kind": "photo_pair_completed", "frame_id": i, "monotonic_ns": 260+i} for i in range(3)] + [
        {"kind": "photo_closed", "monotonic_ns": 290, "pipeline_stopped": True, "writer_closed": True,
         "device_restored": True, "physical_output_gate": False, "sync_readback": {"trigger_out_enable": False}}]
    path = _write(tmp_path, root, manifest, rows, events)
    assert verify_native_capture_session(root, mode="photo", expected_serial="SERIAL", max_sync_delta_us=1000,
                                        observation_path=path)["status"] == "PASS"
    events[-2]["physical_output_gate"] = True
    path = _write(tmp_path, root, manifest, rows, events)
    with pytest.raises(ValueError, match="Photo hardware"):
        verify_native_capture_session(root, mode="photo", expected_serial="SERIAL", max_sync_delta_us=1000,
                                      observation_path=path)
