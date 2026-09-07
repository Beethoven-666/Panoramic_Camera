"""Real fresh 60s/30m/2h capture actions; never substitutes repeated short runs."""
from __future__ import annotations

import re
import csv
import time
from pathlib import Path

from .campaign import binding_identity, capture_once, relative, sdk_config
from .capture_verifier import verify_native_capture_session
from .common import read, require, write_new
from .endurance import ensure_capacity, measured_storage_rate
from .inventory import command
from .observer import read_observation
from .preview_verifier import verify_native_preview

PROFILES = {"storage-calibration-60s": (60, 100), "continuous-30m": (1800, 100),
            "continuous-2h": (7200, None)}


def run_endurance_session(root, binding, name, profile, calibration=None):
    from panorama_demo import Gemini305VideoSDK
    root = Path(root)
    require(name in PROFILES, "Unknown native endurance suite")
    duration, exposure = PROFILES[name]
    capacity = None
    if duration > 60:
        require(calibration is not None, "Real 60-second storage calibration is required")
        measured = read(Path(calibration) / "storage_rate.json")
        require(measured["h0_id"] == binding["h0_id"], "Storage calibration belongs to another H0")
        capacity = ensure_capacity(root, measured["total_committed_bytes_per_second"], duration)
        write_new(root / "capacity_before.json", capacity)
    sdk = Gemini305VideoSDK(sdk_config(binding, root))
    since = int(time.time())
    try:
        result = capture_once(sdk, root, duration_seconds=duration, exposure_us=exposure, telemetry=True)
    finally:
        sdk.release_control()
    intervals = {}
    for event in read_observation(root / "capture_observation.json"):
        if event["kind"] == "capture_started":
            intervals["start"] = event["capture_started_monotonic_ns"]
        elif event["kind"] == "capture_stopped":
            intervals["stop"] = event["monotonic_ns"]
    actual_seconds = (intervals["stop"] - intervals["start"]) / 1e9
    require(actual_seconds >= duration, "Continuous capture ended before the required duration")
    # Preview/thread draining can extend the observer stop callback. Confirm
    # continuous acquisition duration using the actual device timestamps too.
    with (Path(result["session"]) / "frames.csv").open(newline="", encoding="utf-8") as stream:
        timestamps = [int(row["color_device_timestamp_us"]) for row in csv.DictReader(stream)]
    require(len(timestamps) >= 2, "Endurance device timestamps missing")
    device_span_seconds = (timestamps[-1] - timestamps[0]) / 1_000_000
    require(device_span_seconds >= duration - 2 / 57,
            "Endurance device-time span is shorter than requested capture duration")
    verified = verify_native_capture_session(result["session"], expected_serial=binding["camera_serial"],
        max_sync_delta_us=profile["max_sync_delta_us"], mode="auto" if exposure is None else "fixed",
        requested_exposure_us=exposure, observation_path=root / "capture_observation.json")
    verify_native_preview(root, observation_path=root / "capture_observation.json")
    from panorama_demo.sdk_acceptance import verify_2d
    require(result["formal_2d_published"] is True, "Emergency output cannot qualify endurance")
    verify_2d(result["output"])
    report = read(Path(result["output"]) / "video_report.json")
    reuse = report.get("live_handoff", report.get("frozen_p0_reuse", {}))
    if duration > 60:
        require(type(reuse.get("p0_prefix_reused_fraction")) in (int, float)
                and .90 <= reuse["p0_prefix_reused_fraction"] <= 1
                and reuse.get("full_m0_m3_recomputed") is False,
                "Endurance frozen P0 reuse gate failed")
    kernel = command(["journalctl", "-k", "--since", f"@{since}", "--no-pager", "-o", "short-iso"])
    write_new(root / "kernel_events.json", kernel)
    kernel_errors = [line for line in kernel.get("stdout", "").splitlines()
                     if re.search(r"out of memory|oom-kill|NVRM: Xid|USB.*reset|thermal.*shutdown", line, re.I)]
    observation = {**binding_identity(binding), "execution_kind": "native_physical",
        "profile": {"width": 848, "height": 480, "fps": 60, "preview_enabled": True,
                    "preview_window": False, "retention_policy": "keep"},
        "requested_duration_seconds": duration, "video_exposure_us": exposure,
        "actual_capture_seconds": actual_seconds, "capture_interval": intervals,
        "device_timestamp_span_seconds": device_span_seconds,
        "session_relative_path": relative(root, result["session"]),
        "timing_relative_path": relative(root, Path(result["output"]) / "video_timing.json"),
        "free_bytes_before": capacity["free_bytes"] if capacity else None,
        "capture_verification": {"errors": [], "raw_result": verified},
        "predeclared_resource_limits": {"rss_bytes": 3 * 1024**3, "gpu_memory_bytes": 3 * 1024**3},
        "p0_reuse": reuse,
        "kernel_events_checked": kernel.get("returncode") == 0 and bool(kernel.get("stdout", "").strip())
            and "No journal files" not in kernel.get("stderr", ""),
        "kernel_errors": kernel_errors, "max_sync_delta_us": profile["max_sync_delta_us"]}
    write_new(root / "observations.json", observation)
    if duration == 60:
        write_new(root / "storage_rate.json", {**binding_identity(binding),
            **measured_storage_rate(result["session"], actual_seconds)})
    return observation
