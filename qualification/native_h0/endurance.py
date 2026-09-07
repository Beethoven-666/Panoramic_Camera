"""Native resource sampling and duration/capacity qualification from raw samples."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time

from .inventory import bound_errors, command, read_json

GIB = 1024**3
TELEMETRY_FIELDS = (
    "rss_bytes", "gpu_memory_bytes", "gpu_utilization_percent", "gpu_temperature_c",
    "thermal_throttle_reason", "cpu_load", "free_memory_bytes", "swap_used_bytes",
    "disk_free_bytes", "write_bytes_per_second", "writer_queue_depth", "preview_queue_depth",
    "committed_frame_count", "checkpoint_count", "thread_count", "file_descriptor_count",
    "sealed_prefix_length", "mutable_tail_length", "lease_owned",
)


def required_capacity(bytes_per_second, duration_seconds):
    if not math.isfinite(bytes_per_second) or bytes_per_second <= 0 or duration_seconds <= 0:
        raise ValueError("Positive measured write rate and duration required")
    return math.ceil(bytes_per_second * duration_seconds * 1.25 + 20 * GIB)


def ensure_capacity(storage_root, bytes_per_second, duration_seconds):
    required = required_capacity(bytes_per_second, duration_seconds)
    free = shutil.disk_usage(storage_root).free
    if free < required:
        raise RuntimeError(f"Endurance blocked: {free} free bytes < {required} required")
    return {"free_bytes": free, "required_bytes": required, "duration_seconds": duration_seconds,
            "measured_bytes_per_second": bytes_per_second}


def measured_storage_rate(session, duration_seconds):
    if duration_seconds < 60:
        raise ValueError("Storage calibration must span at least 60 actual capture seconds")
    totals = {"rgb_bytes": 0, "aligned_depth_bytes": 0, "csv_log_bytes": 0, "other_bytes": 0}
    for path in Path(session).rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(session).parts
        key = ("rgb_bytes" if "color" in parts or "rgb" in parts else
               "aligned_depth_bytes" if "depth_aligned" in parts else
               "csv_log_bytes" if path.suffix in {".csv", ".log", ".json", ".jsonl"} else "other_bytes")
        totals[key] += path.stat().st_size
    return {**totals, "actual_capture_seconds": duration_seconds,
            "total_committed_bytes_per_second": sum(totals.values()) / duration_seconds,
            **{key.replace("bytes", "bytes_per_second"): value / duration_seconds for key, value in totals.items()}}


def sample_process_tree(pid, storage_root, runtime_sample=None):
    """Unavailable counters remain explicit missing values; never impute zeros."""
    row = {"monotonic_seconds": time.monotonic(), "unix_seconds": time.time()}
    try:
        import psutil
        parent = psutil.Process(pid)
        processes = [parent] + parent.children(recursive=True)
        row.update(rss_bytes=sum(p.memory_info().rss for p in processes),
                   thread_count=sum(p.num_threads() for p in processes),
                   file_descriptor_count=sum(p.num_fds() for p in processes),
                   cpu_load=os.getloadavg()[0], free_memory_bytes=psutil.virtual_memory().available,
                   swap_used_bytes=psutil.swap_memory().used,
                   disk_free_bytes=shutil.disk_usage(storage_root).free,
                   process_ids=[p.pid for p in processes])
        gpu = command(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"])
        if gpu.get("returncode") == 0:
            process_ids = set(row["process_ids"])
            row["gpu_memory_bytes"] = sum(int(line.split(",")[1].strip()) * 1024**2
                for line in gpu["stdout"].splitlines() if line.strip() and int(line.split(",")[0]) in process_ids)
        metrics = command(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader,nounits"])
        if metrics.get("returncode") == 0:
            values = metrics["stdout"].splitlines()[0].split(",")
            row.update(gpu_utilization_percent=float(values[0]), gpu_temperature_c=float(values[1]), thermal_throttle_reason=values[2].strip())
    except Exception as exc:
        row["sample_error"] = str(exc)
    if runtime_sample:
        try:
            row.update(runtime_sample())
        except Exception as exc:
            row["runtime_sample_error"] = f"{type(exc).__name__}: {exc}"
    for field in TELEMETRY_FIELDS:
        row.setdefault(field, {"status": "NOT_EXECUTED", "reason": "Counter not observed"})
    return row


class TelemetryRecorder:
    def __init__(self, output, pid, storage_root, runtime_sample=None):
        self.output, self.pid, self.storage_root = Path(output), pid, storage_root
        self.runtime_sample = runtime_sample
        self.stop_event = threading.Event()
        self.thread = None

    def __enter__(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.stream = (self.output / "telemetry.jsonl").open("x", encoding="utf-8")
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            row = sample_process_tree(self.pid, self.storage_root, self.runtime_sample)
            self.stream.write(json.dumps(row) + "\n")
            self.stream.flush()
            self.stop_event.wait(max(0., 1. - (time.monotonic() - started)))

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.stream.close()
        rows = [json.loads(line) for line in (self.output / "telemetry.jsonl").read_text().splitlines()]
        with (self.output / "resource_timeseries.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, ["monotonic_seconds", *TELEMETRY_FIELDS], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summary = {"samples": len(rows), "missing_fields": sorted({k for r in rows for k in TELEMETRY_FIELDS if isinstance(r[k], dict)}),
                   "peaks": {k: max((r[k] for r in rows if isinstance(r[k], (int, float))), default=None)
                             for k in ("rss_bytes", "gpu_memory_bytes", "thread_count", "file_descriptor_count")}}
        (self.output / "telemetry_summary.json").write_text(json.dumps(summary, indent=2))


def telemetry_errors(rows, duration, limits):
    errors = []
    if len(rows) < duration * .98:
        errors.append("Missing 1 Hz telemetry coverage")
    for row in rows:
        if any(field not in row or isinstance(row[field], dict) for field in TELEMETRY_FIELDS):
            errors.append("Telemetry counter NOT_EXECUTED or missing")
            break
    if not rows or errors:
        return errors or ["Telemetry absent"]
    times = [r["monotonic_seconds"] for r in rows]
    if times[-1] - times[0] < duration - 2 or any(b - a <= 0 or b - a > 2.5 for a, b in zip(times, times[1:])):
        errors.append("Telemetry has a duration/monotonicity/gap failure")
    for field in ("rss_bytes", "gpu_memory_bytes"):
        limit = limits.get(field, 3 * GIB)
        if limit > 3 * GIB:
            errors.append("Higher resource limit requires a new approved support profile before campaign")
        if any(not isinstance(r[field], (int, float)) or not math.isfinite(r[field]) or r[field] > limit for r in rows):
            errors.append(f"{field}: resource limit exceeded or unmeasured")
    for field in ("thread_count", "file_descriptor_count"):
        tail = [r[field] for r in rows[len(rows) // 2:]]
        if len(tail) > 10 and tail[-1] > tail[0] + 5 and all(b >= a for a, b in zip(tail, tail[1:])):
            errors.append(f"{field}: sustained growth")
    if any(r["lease_owned"] is not True for r in rows):
        errors.append("Camera lease lost or unmeasured")
    if any(r["preview_queue_depth"] > 1 for r in rows):
        errors.append("Preview latest-only queue exceeded")
    return errors


def validate_endurance(root, binding):
    evidence, errors = {}, []
    for name, duration in (("storage-calibration-60s", 60), ("continuous-30m", 1800), ("continuous-2h", 7200)):
        path = Path(root) / "endurance" / name
        try:
            from .campaign import verify_operation
            from .observer import read_observation
            verify_operation(root, binding, "endurance/" + name)
            record = read_json(path / "observations.json")
            errors.extend(bound_errors(record, binding, name))
            evidence[name] = record
            if record.get("execution_kind") != "native_physical" or record.get("actual_capture_seconds", 0) < duration:
                errors.append(f"{name}: actual physical duration missing")
            if record.get("profile") != {"width": 848, "height": 480, "fps": 60, "preview_enabled": True, "preview_window": False, "retention_policy": "keep"}:
                errors.append(f"{name}: profile mismatch")
            session = (path / record["session_relative_path"]).resolve()
            session.relative_to(Path(root).resolve())
            observation = path / "capture_observation.json"
            intervals = {}
            for event in read_observation(observation):
                if event["kind"] == "capture_started":
                    intervals["start"] = event["capture_started_monotonic_ns"]
                elif event["kind"] == "capture_stopped":
                    intervals["stop"] = event["monotonic_ns"]
            actual_seconds = (intervals["stop"] - intervals["start"]) / 1e9
            if actual_seconds < duration or actual_seconds != record["actual_capture_seconds"]:
                errors.append(f"{name}: actual observer capture interval failed")
            with (session / "frames.csv").open(newline="", encoding="utf-8-sig") as stream:
                stamps = [int(row["color_device_timestamp_us"]) for row in csv.DictReader(stream)]
            if len(stamps) < 2 or (stamps[-1] - stamps[0]) / 1e6 < duration - 2 / 57:
                errors.append(f"{name}: actual device timestamp span shorter than full duration")
            manifest = read_json(session / "manifest.json")
            if manifest.get("clean_shutdown") is not True:
                errors.append(f"{name}: clean shutdown missing")
            from .capture_verifier import verify_native_capture_session
            from .preview_verifier import verify_native_preview
            profile = read_json(Path(root) / "profile.json")
            verify_native_capture_session(session, expected_serial=binding["camera_serial"],
                max_sync_delta_us=profile["max_sync_delta_us"], mode="auto" if duration == 7200 else "fixed",
                requested_exposure_us=None if duration == 7200 else 100, observation_path=observation)
            verify_native_preview(path, observation_path=observation)
            from panorama_demo.sdk_acceptance import verify_2d
            two_d = (path / record["timing_relative_path"]).parent
            verify_2d(two_d)
            if name == "storage-calibration-60s":
                rate = measured_storage_rate(session, actual_seconds)
                evidence[name]["measured_rate"] = rate
                continue
            rows = [json.loads(line) for line in (path / "telemetry.jsonl").read_text().splitlines()]
            rows = [row for row in rows if row.get("capture_active") is True]
            errors.extend(f"{name}: {e}" for e in telemetry_errors(rows, duration, record.get("predeclared_resource_limits", {})))
            calibration = evidence.get("storage-calibration-60s", {}).get("measured_rate", {})
            rate = calibration.get("total_committed_bytes_per_second", 0)
            if not rate or record.get("free_bytes_before", 0) < required_capacity(rate, duration):
                errors.append(f"{name}: calibrated capacity gate missing or failed")
            timing = read_json(path / record["timing_relative_path"])
            seconds = timing.get("final_2d", {}).get("capture_stop_to_p3_published_seconds")
            if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 <= seconds <= 60:
                errors.append(f"{name}: final_2d stop-to-P3 missing/exceeded")
            report = read_json((path / record["timing_relative_path"]).parent / "video_report.json")
            reuse = report.get("live_handoff", report.get("frozen_p0_reuse", {}))
            if (type(reuse.get("p0_prefix_reused_fraction")) not in (int, float)
                    or not .90 <= reuse["p0_prefix_reused_fraction"] <= 1
                    or reuse.get("full_m0_m3_recomputed") is not False):
                errors.append(f"{name}: P0 reuse gate failed")
            kernel = read_json(path / "kernel_events.json")
            kernel_errors = [line for line in kernel.get("stdout", "").splitlines()
                if re.search(r"out of memory|oom-kill|NVRM: Xid|USB.*reset|thermal.*shutdown", line, re.I)]
            if (kernel.get("returncode") != 0 or not kernel.get("stdout", "").strip()
                    or "No journal files" in kernel.get("stderr", "") or kernel_errors):
                errors.append(f"{name}: kernel OOM/Xid/USB/thermal evidence missing/failed")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{name}: NOT_EXECUTED/invalid evidence: {exc}")
    return {"errors": errors, "evidence": evidence}
