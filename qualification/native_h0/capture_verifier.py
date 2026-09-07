"""Native capture gates over actual manifests, CSV rows, images and observations."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .observer import read_observation


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Missing numeric " + label) from exc
    _require(not isinstance(value, bool) and math.isfinite(result), "Invalid " + label)
    return result


def verify_native_capture_session(session_root: Path, *, expected_serial: str,
                                  max_sync_delta_us: float, mode: str = "fixed",
                                  requested_exposure_us: int | None = 100,
                                  observation_path: Path | None = None) -> dict:
    """Verify one normal run; fault salvage has a separate, explicit contract.

    max_sync_delta_us is a pre-run profile decision, not a fitted threshold.
    Photo mode verifies its software-trigger contract, not continuous 60 FPS.
    """
    import cv2
    import numpy as np

    root = Path(session_root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _require(mode in {"fixed", "auto", "photo"}, "Unknown capture contract")
    _require(expected_serial and isinstance(expected_serial, str), "Missing bound camera serial")
    threshold = _number(max_sync_delta_us, "pre-run sync threshold")
    _require(0 <= threshold < 1_000_000 / 60, "Sync threshold must exclude a different 60 Hz frame")
    device = manifest.get("device", {})
    _require(device.get("serial_number") == expected_serial, "Capture camera serial mismatch")
    _require(device.get("name") in {"Gemini 305", "Orbbec Gemini 305"}, "Wrong Gemini 305 identity")
    _require(device.get("firmware_version") == "1.0.70", "Wrong camera firmware")
    _require(manifest.get("python_wrapper_version") == "2.1.2+g305.1", "Wrong Orbbec wrapper")
    _require(str(manifest.get("sdk_version")) == "2.9.3", "Wrong native SDK version")
    photo = mode == "photo"
    _require(manifest.get("schema") == f"panorama-demo-session/v{1 if photo else 2}", "Wrong capture schema")
    expected_mode = {"fixed": "continuous_rgbd_video_fixed_exposure",
                     "auto": "continuous_rgbd_video_auto",
                     "photo": "software_triggered_rgbd_photo_sequence"}[mode]
    _require(manifest.get("capture_mode") == expected_mode, "Wrong capture mode")
    for kind in ("color", "depth"):
        profile = manifest.get("profiles", {}).get(kind, {})
        _require(all(type(profile.get(key)) is int and profile[key] == value
                     for key, value in (("width", 848), ("height", 480), ("fps", 60))),
                 "Wrong actual " + kind + " profile")
        _require(profile.get("format") in ({"Y16"} if kind == "depth" else {"RGB", "BGR", "YUYV", "MJPG"}),
                 "Unsupported actual " + kind + " format")
    count = manifest.get("written_frames")
    _require(type(count) is int and count > 0, "Missing written frame count")
    _require(type(manifest.get("received_frames")) is int and manifest["received_frames"] == count,
             "Received/written frame count differs")
    for field in ("queue_drops", "write_errors", "timestamp_regressions"):
        _require(type(manifest.get(field)) is int and manifest[field] == 0, "Nonzero or missing " + field)
    _require(manifest.get("writer_errors") == [], "Writer error evidence incomplete")
    _require(manifest.get("clean_shutdown") is True, "Capture did not close cleanly")
    _require(not manifest.get("capture_error") and not manifest.get("shutdown_errors"), "Capture/shutdown error")
    with (root / "frames.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == count, "CSV count differs from written_frames")
    ids = [int(row["frame_id"]) for row in rows]
    _require(ids == list(range(count)), "Frame IDs must start at zero and be consecutive")
    color_times, depth_times = [], []
    observed_max_delta = 0.0
    for row in rows:
        color_time = _number(row.get("color_device_timestamp_us"), "color timestamp")
        depth_time = _number(row.get("depth_device_timestamp_us"), "depth timestamp")
        _require(color_time >= 0 and depth_time >= 0, "Negative device timestamp")
        delta = _number(row.get("sync_delta_us"), "sync delta")
        _require(delta == color_time - depth_time and abs(delta) <= threshold, "RGB-D synchronization exceeded")
        observed_max_delta = max(observed_max_delta, abs(delta))
        color_times.append(color_time)
        depth_times.append(depth_time)
        exposure = _number(row.get("color_exposure"), "exposure") * 100
        _require(0 < exposure <= 800, "Exposure outside formal capture bound")
        _require(_number(row.get("color_gain"), "gain") >= 0, "Invalid gain metadata")
        if mode == "fixed":
            _require(exposure == requested_exposure_us, "Fixed exposure does not match request")
        for key, shape, dtype in (("color_path", (480, 848, 3), np.uint8),
                                  ("aligned_depth_path", (480, 848), np.uint16)):
            value = row.get(key)
            _require(isinstance(value, str) and bool(value), "Missing " + key)
            path = (root / value).resolve()
            _require(path.is_relative_to(root), "Frame file is outside captured session")
            if key == "aligned_depth_path":
                _require(path.parent == root / "depth_aligned", "Depth does not come from depth_aligned")
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            _require(image is not None and image.shape == shape and image.dtype == dtype,
                     "Invalid RGB-D frame file: " + str(path))
    for timestamps in (color_times, depth_times):
        _require(all(b > a for a, b in zip(timestamps, timestamps[1:])), "Device timestamp regression")
    fps = None
    if not photo:
        audit = manifest.get("capture_runtime_audit", {})
        for key in ("online_orb_tracker_construct_count", "capture_orbslam3_process_count",
                    "capture_orbslam3_frame_submit_count", "capture_open3d_import_count",
                    "capture_open3d_tsdf_call_count", "capture_3d_process_count"):
            _require(type(audit.get(key)) is int and audit[key] == 0,
                     "Heavy Runtime capture audit missing or nonzero: " + key)
        _require(count >= 2, "Insufficient timestamps for actual FPS")
        fps = (count - 1) * 1_000_000 / (color_times[-1] - color_times[0])
        depth_fps = (count - 1) * 1_000_000 / (depth_times[-1] - depth_times[0])
        _require(57 <= fps <= 63 and 57 <= depth_fps <= 63, "Actual FPS outside 57..63")
        _require(manifest.get("product_eligibility") == {"photo_panorama": False, "video_panorama": True},
                 "Video product is not eligible")
        sync = manifest.get("external_sync_output", {})
        _require(sync.get("per_frame_readback_verified_frames") == count, "Incomplete per-frame sync readback")
        shutdown = sync.get("shutdown", {})
        _require(shutdown.get("completed") is True and shutdown.get("readback_verified") is True
                 and shutdown.get("after", {}).get("mode") == "STANDALONE"
                 and shutdown.get("after", {}).get("trigger_out_enable") is False,
                 "Missing final STANDALONE and Trigger Out readback")
        if mode == "fixed":
            _require(type(requested_exposure_us) is int and 0 < requested_exposure_us <= 800,
                     "Invalid fixed exposure request")
            _require(manifest.get("color_exposure_control", {}).get("requested_exposure_us") == requested_exposure_us,
                     "Fixed exposure request differs from manifest")
        else:
            _require(requested_exposure_us is None, "Auto contract must not request fixed exposure")
            _require(manifest.get("color_control_policy") == "auto_warmup_then_full_manual_lock", "Missing auto warmup policy")
        lock = manifest.get("color_control_lock", {})
        controls = lock.get("locked_controls", {})
        _require(lock.get("completed") is True and lock.get("readback_verified") is True
                 and controls.get("auto_exposure") is False and controls.get("auto_white_balance") is False,
                 "Color controls were not locked and read back")
        _require(type(lock.get("warmup_frame_sets")) is int and lock["warmup_frame_sets"] > 0,
                 "Missing actual color-control warmup frames")
        for row in rows:
            _require(_number(row.get("color_exposure"), "exposure") == controls.get("exposure_raw")
                     and _number(row.get("color_gain"), "gain") == controls.get("gain_raw")
                     and _number(row.get("color_white_balance"), "white balance") == controls.get("white_balance_raw"),
                     "Formal frame metadata differs from locked controls")
        from panorama_demo.video_session import load_video_session
        load_video_session(root, validate_frame_files=False)  # Files decoded above.
    else:
        _require(manifest.get("formal_stitch_allowed") is True
                 and manifest.get("one_formal_trigger_per_capture") is True,
                 "Photo formal trigger contract missing")
        _require(manifest.get("software_trigger", {}).get("per_frame_readback_verified_frames") == count,
                 "Incomplete photo readback")
        from panorama_demo.session import load_rgbd_session
        load_rgbd_session(root, validate_frame_files=False)

    _require(observation_path is not None, "Native capture needs current-run observation")
    # Stream all frame events; retain only compact control/Preview records.
    controls_observed = []
    submitted_count = 0
    first_submit_ns = last_submit_ns = None
    last_sync = None
    for event in read_observation(observation_path):
        if event["kind"] == "sync_readback":
            last_sync = event
        elif event["kind"] == "frame_submitted":
            _require(last_sync is not None and (last_submit_ns is None or last_sync["monotonic_ns"] > last_submit_ns),
                     "Formal frame lacks a fresh observed sync readback")
            readback = last_sync.get("readback", {})
            options = manifest.get("capture_options", {})
            _require(readback.get("mode") == ("SOFTWARE_TRIGGERING" if photo else "PRIMARY")
                     and readback.get("trigger_out_enable") is True, "Wrong observed sync mode/gate")
            for key in ("trigger_to_image_delay_us", "trigger_out_delay_us"):
                _require(type(options.get(key)) is int and readback.get(key) == options[key],
                         "Observed per-frame sync delay differs: " + key)
            _require(readback.get("color_delay_us") == options["trigger_to_image_delay_us"]
                     and readback.get("depth_delay_us") == options["trigger_to_image_delay_us"],
                     "Observed RGB-D image delays differ")
            _require(Path(event.get("session", "")).resolve() == root, "Observer belongs to another session")
            _require(event.get("accepted") is True and event.get("frame_id") == submitted_count,
                     "Observed submit sequence differs from CSV")
            _require(submitted_count < count, "Observed extra frame submit")
            metadata = event.get("metadata", {})
            row = rows[submitted_count]
            for key in ("color_device_timestamp_us", "depth_device_timestamp_us", "color_exposure", "color_gain"):
                _require(_number(row[key], key) == metadata.get(key), "Observed frame differs from CSV: " + key)
            submitted_count += 1
            if first_submit_ns is None:
                first_submit_ns = event["monotonic_ns"]
            last_submit_ns = event["monotonic_ns"]
        elif event["kind"] in {"device_discovered", "lease_acquired", "lease_released",
                                "photo_pair_completed", "photo_closed", "hardware_shutdown", "writer_closed"}:
            controls_observed.append(event)
    _require(submitted_count == count, "Observed/committed frame count differs")
    _require(not any(e["kind"] == "device_discovered" and e["monotonic_ns"] > first_submit_ns
                     for e in controls_observed), "Device reconnected after first formal frame")
    acquired = [e for e in controls_observed if e["kind"] == "lease_acquired" and e.get("serial") == expected_serial]
    released = [e for e in controls_observed if e["kind"] == "lease_released" and e.get("serial") == expected_serial]
    _require(acquired and released and acquired[0].get("held") is True and released[-1].get("held") is False,
             "Missing actual camera lease acquire/release")
    _require(acquired[0]["monotonic_ns"] < first_submit_ns and released[0]["monotonic_ns"] > last_submit_ns,
             "Camera lease did not span all formal frames")
    if photo:
        pairs = [e.get("frame_id") for e in controls_observed if e["kind"] == "photo_pair_completed"]
        _require(pairs == ids, "Photo trigger calls did not each complete one pair")
        closed = [e for e in controls_observed if e["kind"] == "photo_closed"]
        _require(closed and closed[-1].get("pipeline_stopped") is True
                 and closed[-1].get("writer_closed") is True and closed[-1].get("device_restored") is True
                 and closed[-1].get("physical_output_gate") is False
                 and closed[-1].get("sync_readback", {}).get("trigger_out_enable") is False,
                 "Photo hardware close/readback incomplete")
    else:
        closed = [e for e in controls_observed if e["kind"] == "hardware_shutdown"]
        _require(len(closed) == 1 and closed[0].get("pipeline_was_started") is True
                 and closed[0].get("errors") == [] and closed[0].get("shutdown") == shutdown,
                 "Pipeline stop/shutdown was not observed")
    _require(released[0]["monotonic_ns"] > closed[-1]["monotonic_ns"], "Camera lease released before hardware close")
    drained = [e for e in controls_observed if e["kind"] == "writer_closed"]
    _require(drained and drained[-1].get("worker_alive") is False and drained[-1].get("queue_depth") == 0
             and drained[-1].get("written_frames") == count and drained[-1].get("queue_drops") == 0
             and drained[-1].get("write_errors") == 0 and drained[-1].get("writer_errors") == [],
             "Writer drain did not complete with all frames")
    _require(released[0]["monotonic_ns"] > drained[-1]["monotonic_ns"], "Camera lease released before writer drain")
    return {"schema": "gemini305-native-h0-capture-verification/v1", "status": "PASS",
            "session": str(root), "mode": mode, "written_frames": count,
            "effective_fps": fps, "maximum_observed_sync_delta_us": observed_max_delta,
            "maximum_allowed_sync_delta_us": threshold, "queue_drops": 0,
            "write_errors": 0, "timestamp_regressions": 0}
