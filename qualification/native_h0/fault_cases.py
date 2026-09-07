"""Physical fault checkpoints and independent checks of native raw observations.

Operator confirmation records an action only; it never substitutes for the
device/SDK events observed by the instrumented campaign process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import threading
import time

from .inventory import bound_errors, command, read_json, _text

FAULT_CASES = ("no-wait", "wait-cancel", "pre-formal-reconnect", "post-formal-salvage",
               "early-emergency", "low-disk", "camera-lease", "sigterm")
DEVICE_LOSS = {"CAMERA_DISCONNECTED", "FRAME_TIMEOUT", "USB_TRANSPORT_ERROR"}


class USBPortMonitor:
    """Observe physical sysfs port transitions independently at 20 Hz."""
    def __init__(self, port_path, serial, emit):
        self.path = Path("/sys/bus/usb/devices") / port_path
        self.serial, self.emit = serial, emit
        self.done = threading.Event()

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        previous = None
        while not self.done.is_set():
            current = (self.path.exists(), _text(self.path / "serial"), _text(self.path / "speed"))
            if current != previous:
                if previous is None:
                    kind = "usb_initial_state"
                elif previous[0] and not current[0]:
                    kind = "device_disconnected"
                elif not previous[0] and current[0]:
                    kind = "device_reconnected"
                else:
                    kind = "usb_identity_or_speed_changed"
                self.emit(kind, sysfs_path=str(self.path), present=current[0], serial=current[1],
                          speed_mbps=current[2], expected_serial=self.serial)
                previous = current
            self.done.wait(.05)

    def __exit__(self, *_):
        self.done.set()
        self.thread.join()


def operator_checkpoint(events_path, prompt, *, confirm=input):
    requested = time.monotonic()
    if confirm is input and os.name == "posix":
        # Worker stdout is a retained log; the physical operator needs the prompt.
        with open("/dev/tty", "w") as terminal:
            terminal.write(prompt + " [type DONE on stdin after physical action, or ABORT]:\n")
            terminal.flush()
    answer = confirm(prompt + " [type DONE after physical action, or ABORT]: ")
    event = {"event": "operator_checkpoint", "prompt": prompt,
             "requested_monotonic": requested, "monotonic_seconds": time.monotonic(),
             "unix_seconds": time.time(), "answer": answer}
    with Path(events_path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    if answer.strip() != "DONE":
        raise RuntimeError("Operator aborted physical fault case")
    return event


def lease_probe(root, config, serial, job_root=None):
    from panorama_demo import Gemini305VideoSDK
    from panorama_demo.sdk_state import SDKBusyError
    from .observer import NativeCaptureObserver
    result = {"pid": os.getpid(), "serial": serial, "monotonic_seconds": time.monotonic()}
    sdk = Gemini305VideoSDK(config)
    try:
        with NativeCaptureObserver(Path(root) / "capture_observation.json"):
            job = sdk.start_capture(Path(root) / "sdk-capture", duration_seconds=3,
                                    width=848, height=480, fps=60, video_exposure_us=100,
                                    preview_window=False, wait_for_camera=False)
            published = job.result()
            manifest = read_json(published.session / "manifest.json")
            result.update(event="lease_acquired", released=True,
                sdk_method="start_capture", session=str(published.session), output=str(published.output_dir),
                captured_frames=manifest.get("written_frames"), terminal_state=job.state.value,
                formal_2d_published=published.formal_2d_published)
    except SDKBusyError as exc:
        result.update(event="lease_busy", error_code=exc.error_code, error_type=type(exc).__name__, sdk_method="start_capture")
        if job_root:
            from panorama_demo.video_sdk import VideoProcessingJob
            readonly = VideoProcessingJob.load(Path(job_root))
            result["readonly_state"] = readonly.state.value
            preview_root = Path(job_root) / "preview"
            if ((preview_root / "live_preview.jpg").is_file()
                    and read_json(preview_root / "live_preview_state.json").get("authority") == "non_authoritative_live_preview"):
                result["readonly_preview_observed"] = True
    finally:
        sdk.release_control()
    with (Path(root) / "lease_probe.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def _normalise_observer(events):
    result = []
    for raw in events:
        event = {**raw, "event": raw.get("kind", raw.get("event")),
                 "monotonic_seconds": raw.get("monotonic_seconds", raw.get("monotonic_ns", 0) / 1e9)}
        result.append(event)
        if (event["event"] == "writer_closed" and event.get("queue_depth") == 0
                and event.get("worker_alive") is False and event.get("write_errors") == 0
                and event.get("writer_errors") == []):
            result.append({**event, "event": "writer_drained"})
        if event["event"] == "boundary_error" and event.get("boundary") == "acquire" and event.get("error_type") == "SDKBusyError":
            result.append({**event, "event": "lease_busy"})
        if event["event"] == "job_state":
            payload = event.get("payload", {})
            reason = payload.get("stop_reason")
            if reason == "LOW_DISK":
                result.append({**event, "event": "low_disk_stop"})
        if event["event"] == "hardware_shutdown":
            shutdown = event.get("shutdown", {})
            if (not event.get("errors") and shutdown.get("state") == "verified_off"
                    and shutdown.get("readback_verified") is True
                    and shutdown.get("after", {}).get("trigger_out_enable") is False):
                result.append({**event, "event": "trigger_out_disabled_readback"})
            elif shutdown.get("completed") is False:
                transitions = [e for e in result if e["event"] in {"device_disconnected", "device_reconnected"}]
                if transitions and transitions[-1]["event"] == "device_disconnected":
                    result.append({**event, "event": "trigger_out_readback_device_lost"})
    return sorted(result, key=lambda e: e["monotonic_seconds"])


def run_fault_case(case, root, binding, config, *, loopback_root=None, owned_session=False):
    """Run one fresh physical SDK job with explicit unplug/checkpoint control.

Raw observation shortcomings remain verifier failures. Operator acknowledgements
are never converted to device events or counted as successful fault injection.
"""
    if case not in FAULT_CASES:
        raise ValueError("Unknown physical fault case")
    from panorama_demo import Gemini305VideoSDK
    from .campaign import binding_identity, relative, run_worker
    from .observer import NativeCaptureObserver, read_observation
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    action_path = root / "operator_events.jsonl"
    mount = validate_loopback_mount(loopback_root) if case == "low-disk" else None
    if case in {"no-wait", "wait-cancel"}:
        operator_checkpoint(action_path, "Disconnect the Gemini 305 now")
    else:
        operator_checkpoint(action_path, "Connect the bound Gemini 305 and prepare the standard moving scene")
    if case == "sigterm":
        return _run_sigterm_cli(root, binding, action_path)
    sdk = Gemini305VideoSDK(config)
    result = error = None
    started = time.monotonic()
    cancelled = None
    with NativeCaptureObserver(root / "capture_observation.json") as observer, USBPortMonitor(
            binding["usb_port_path"], binding["camera_serial"], observer._event):
        # Actual SDK enumeration, not operator assertion, proves camera absence.
        if case in {"no-wait", "wait-cancel"}:
            import pyorbbecsdk as ob
            if ob.Context().query_devices().get_count() != 0:
                raise RuntimeError("Camera must actually be absent for this case")
            observer._event("device_absent")
        from panorama_demo import capture_orbbec
        original_discovery = capture_orbbec._discover_video_device
        if case == "pre-formal-reconnect":
            injected = False
            def discovery_checkpoint(*args, **kwargs):
                nonlocal injected
                found = original_discovery(*args, **kwargs)
                if not injected:
                    injected = True
                    operator_checkpoint(action_path, "Discovery is paused before formal frames. Disconnect Gemini 305")
                    operator_checkpoint(action_path, "Reconnect Gemini 305 now; SDK will continue its real pre-formal recovery")
                return found
            capture_orbbec._discover_video_device = discovery_checkpoint
        capture_root = Path(loopback_root) / f"{binding['h0_id']}-capture" if mount else root / "sessions"
        if mount and capture_root.exists():
            raise FileExistsError(capture_root)
        capture_options = dict(width=848, height=480, fps=60, duration_seconds=120 if case != "no-wait" else 3,
            video_exposure_us=100, preview_window=False, wait_for_camera=case != "no-wait")
        if owned_session:
            capture_root = root
            job = sdk.start_capture(root, **capture_options)
        else:
            job = sdk.capture_and_process(capture_root=capture_root, output_dir=root / "2d", **capture_options)
        try:
            if case == "wait-cancel":
                if not job.wait(.5):
                    cancelled = time.monotonic()
                    observer._event("job_cancel_called", accepted=job.cancel())
            elif case in {"post-formal-salvage", "early-emergency", "camera-lease"}:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    snapshot = observer.snapshot_runtime()
                    frames = snapshot.get("committed_frame_count", 0)
                    if (isinstance(frames, int) and frames >= (120 if case in {"post-formal-salvage", "camera-lease"} else 1)
                            and (case != "camera-lease" or sdk.get_preview(job.job_id))):
                        break
                    if job.wait(.05):
                        raise RuntimeError("Capture ended before physical fault checkpoint")
                else:
                    raise TimeoutError("No durable formal prefix for fault injection")
                if case == "camera-lease":
                    probe = root / "contending-process"
                    probe.mkdir()
                    run_worker("lease-probe", probe, binding, serial=binding["camera_serial"], job_root=str(job._record.root))
                    event = read_json(probe / "lease_probe.json")
                    observer._event(event.pop("event"), **event)
                    if event.get("readonly_preview_observed"):
                        observer._event("readonly_preview_observed", pid=event["pid"], state=event["readonly_state"])
                    job.cancel()
                else:
                    operator_checkpoint(action_path, "Disconnect Gemini 305 now; do not reconnect until this job has ended")
            result = job.result(timeout=200)
        except Exception as exc:
            error = exc
        finally:
            if not job.wait(0):
                job.cancel()
                job.wait(90)
            sdk.release_control()
            capture_orbbec._discover_video_device = original_discovery
        payload = dict(job._record.payload)
        if case == "camera-lease":
            probe = root / "released-process"
            probe.mkdir()
            run_worker("lease-probe", probe, binding, serial=binding["camera_serial"])
            event = read_json(probe / "lease_probe.json")
            observer._event(event.pop("event"), **event)
            if event.get("released"):
                observer._event("lease_released", pid=event["pid"])
    ended = time.monotonic()
    record = {**binding_identity(binding), "execution_kind": "native_physical", "wait_for_camera": case != "no-wait",
              "elapsed_seconds": ended - started, "terminal_state": job.state.value,
              "stop_reason": payload.get("stop_reason"), "error_code": getattr(error, "error_code", None) or payload.get("failure", {}).get("error_code"),
              "cancel_latency_seconds": None if cancelled is None else ended - cancelled,
              "retention_policy": config.retention_policy.value, "approved_retry_limit": 3,
              "loopback_mount": mount, "created_sessions": [str(p.parent) for p in capture_root.rglob("manifest.json")
                  if read_json(p).get("written_frames", 0) > 0],
              "all_session_manifests": [str(p) for p in capture_root.rglob("manifest.json")]}
    if result:
        session_evidence = result.session
        if mount:
            session_evidence = root / "retained-session"
            shutil.copytree(result.session, session_evidence)
            record["loopback_original_session"] = str(result.session)
        record["session_relative_path"] = os.path.relpath(session_evidence, root)
        for name, filename in (("formal_delivery", "video_delivery.json"), ("emergency_delivery", "emergency_delivery.json")):
            path = result.output_dir / filename
            record[name] = relative(root, path) if path.is_file() else None
    raw = list(read_observation(root / "capture_observation.json"))
    raw.extend(_events(action_path) if action_path.exists() else [])
    with (root / "events.jsonl").open("x", encoding="utf-8") as stream:
        for event in _normalise_observer(raw):
            stream.write(json.dumps(event) + "\n")
    with (root / "observations.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    return record


def _run_sigterm_cli(root, binding, action_path):
    """Send real SIGTERM to this dedicated worker running the candidate CLI."""
    import sys
    from panorama_demo.video_live import main
    from panorama_demo.camera_lease import CameraLease
    from panorama_demo.sdk_acceptance import audit_2d_process
    from .campaign import binding_identity
    from .observer import NativeCaptureObserver, read_observation
    with NativeCaptureObserver(root / "capture_observation.json") as observer:
        done = threading.Event()
        def stop_after_commit():
            while not done.wait(.05):
                count = observer.snapshot_runtime().get("committed_frame_count")
                if isinstance(count, int) and count >= 120:
                    observer._event("sigterm_sent", signal=int(signal.SIGTERM), committed_frames=count)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
        monitor = threading.Thread(target=stop_after_commit, daemon=True)
        monitor.start()
        previous = sys.argv
        try:
            sys.argv = ["g305-video-live", "--output", str(root / "sessions"), "--panorama-output", str(root / "2d"),
                        "--width", "848", "--height", "480", "--fps", "60", "--duration", "120",
                        "--video-exposure-us", "100", "--no-preview"]
            with CameraLease(Path("/var/lock/gemini305-sdk"), "discovery", "h0-sigterm"), CameraLease(
                    Path("/var/lock/gemini305-sdk"), binding["camera_serial"], "h0-sigterm"), audit_2d_process(root / "2d"):
                main()
        finally:
            sys.argv = previous
            done.set()
            monitor.join()
    manifests = list((root / "sessions").rglob("manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("Signal case did not produce one fresh session")
    manifest = read_json(manifests[0])
    record = {**binding_identity(binding), "execution_kind": "native_physical",
              "session_relative_path": str(manifests[0].parent.relative_to(root)),
              "stop_reason": manifest.get("stop_reason"), "cli_returncode": 0,
              "clean_shutdown": manifest.get("clean_shutdown"),
              "formal_delivery": "2d/video_delivery.json" if (root / "2d/video_delivery.json").exists() else None,
              "emergency_delivery": "2d/emergency_delivery.json" if (root / "2d/emergency_delivery.json").exists() else None}
    raw = _normalise_observer(list(read_observation(root / "capture_observation.json")))
    (root / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in raw))
    (root / "observations.json").write_text(json.dumps(record, indent=2))
    return record


def validate_loopback_mount(path):
    """Require a pre-mounted dedicated loop device; never allocate/fill a system disk."""
    result = command(["findmnt", "-J", "-T", str(Path(path).resolve()), "-o", "SOURCE,TARGET,FSTYPE"])
    try:
        filesystem = json.loads(result["stdout"])["filesystems"][0]
        if (filesystem["fstype"] != "ext4" or not filesystem["source"].startswith("/dev/loop")
                or Path(filesystem["target"]).resolve() != Path(path).resolve()
                or filesystem["target"] == "/"):
            raise ValueError("not a dedicated loopback ext4 mount")
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Low disk physical case blocked: {exc}") from exc
    return filesystem


def send_safe_sigterm(process, events_path, committed_frames):
    if os.name != "posix" or committed_frames < 1 or process.poll() is not None:
        raise RuntimeError("SIGTERM requires a running native child after a committed formal frame")
    event = {"event": "sigterm_sent", "pid": process.pid, "committed_frames": committed_frames,
             "monotonic_seconds": time.monotonic(), "signal": signal.SIGTERM}
    process.send_signal(signal.SIGTERM)
    with Path(events_path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")
    return event


def _events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fault_errors(case, record, events):
    errors = []
    def require(ok, message):
        if not ok:
            errors.append(f"{case}: {message}")
    require(case in FAULT_CASES, "unknown case")
    require(record.get("execution_kind") == "native_physical", "physical execution missing")
    require(bool(events), "raw events missing")
    timestamps = [e.get("monotonic_seconds") for e in events]
    require(all(isinstance(t, (int, float)) for t in timestamps)
            and all(b >= a for a, b in zip(timestamps, timestamps[1:]) if a is not None and b is not None), "event clock invalid")
    names = [e.get("event") for e in events]
    def by_name(name):
        return [e for e in events if e.get("event") == name]
    commits = by_name("frame_committed")
    require("lease_released" in names, "lease release not observed")
    if commits:
        first = commits[0]["monotonic_seconds"]
        require(not any(e.get("event") in {"device_reconnected", "pipeline_restarted"}
                        and e["monotonic_seconds"] > first for e in events), "reconnect after first formal frame prohibited")
        ids = [e.get("frame_id") for e in commits]
        require(all(isinstance(i, int) for i in ids) and len(set(ids)) == len(ids), "invalid committed frame IDs")
    if case in {"no-wait", "wait-cancel"}:
        require(not commits and not record.get("formal_delivery") and not record.get("emergency_delivery"), "no-data case produced frames/delivery")
        require("device_absent" in names, "camera absence not observed")
        if case == "no-wait":
            require(record.get("wait_for_camera") is False and record.get("error_code") in {"CAMERA_NOT_FOUND", "CAMERA_UNAVAILABLE", "CAMERA_NOT_CONNECTED", "CAPTURE_ERROR"}, "stable no-wait error missing")
            require(0 <= record.get("elapsed_seconds", 999) <= 5, "no-wait return exceeded 5 seconds")
        else:
            require(record.get("terminal_state") == "CANCELLED_NO_DATA" and "job_cancel_called" in names,
                    "job.cancel()/CANCELLED_NO_DATA missing")
            require(0 <= record.get("cancel_latency_seconds", 999) <= 5, "cancel latency exceeded 5 seconds")
    elif case == "pre-formal-reconnect":
        reconnects = by_name("device_reconnected")
        require(bool(reconnects) and bool(by_name("device_disconnected")), "actual disconnect/reconnect absent")
        require(len(reconnects) <= record.get("approved_retry_limit", 0), "retry bound missing/exceeded")
        require(len(record.get("created_sessions", [])) == 1 and bool(commits), "exactly one final session required")
        require(not any(e.get("warmup") is True for e in commits), "failed warmup committed")
        require(bool(record.get("formal_delivery")) and not record.get("emergency_delivery"), "normal recovery P3 absent")
    elif case in {"post-formal-salvage", "early-emergency"}:
        losses = by_name("device_disconnected")
        require(bool(commits) and bool(losses) and losses[0]["monotonic_seconds"] >= commits[0]["monotonic_seconds"], "post-formal physical disconnect absent")
        require(record.get("terminal_state") == "COMPLETED_WITH_WARNINGS" and record.get("stop_reason") in DEVICE_LOSS, "loss warning/stop reason missing")
        require("writer_drained" in names and record.get("retention_policy") == "keep", "durable prefix drain/retention absent")
        require("trigger_out_disabled_readback" in names or "trigger_out_readback_device_lost" in names, "final Trigger Out outcome absent")
        if case == "post-formal-salvage":
            require(bool(record.get("formal_delivery")) and not record.get("emergency_delivery"), "safe prefix did not produce formal P3")
        else:
            require(not record.get("formal_delivery") and bool(record.get("emergency_delivery")), "early loss must produce independent emergency")
    elif case == "low-disk":
        fs = record.get("loopback_mount", {})
        require(str(fs.get("source", "")).startswith("/dev/loop") and fs.get("fstype") == "ext4"
                and fs.get("target") not in {None, "/"}, "dedicated loopback ext4 missing")
        require(record.get("stop_reason") == "LOW_DISK" and record.get("terminal_state") == "COMPLETED_WITH_WARNINGS", "LOW_DISK controlled warning stop missing")
        require("low_disk_stop" in names and "writer_drained" in names and "enospc" not in names and "write_error" not in names, "low disk did not stop before write failure")
        require("post_3d_started" not in names and bool(record.get("formal_delivery") or record.get("emergency_delivery")), "loadable output/3-D skip missing")
    elif case == "camera-lease":
        acquired, rejected = by_name("lease_acquired"), by_name("lease_busy")
        require(len({e.get("pid") for e in acquired}) >= 2 and bool(rejected), "two real processes and retry acquisition required")
        require(all(e.get("error_code") in {"CAMERA_BUSY", "JOB_CONTROL_NOT_OWNED"} for e in rejected), "second process did not return typed busy error")
        require(any(e.get("sdk_method") == "start_capture" and e.get("captured_frames", 0) > 0
                    and e.get("formal_2d_published") is True for e in acquired), "second public SDK capture after release missing")
        owners = set()
        for event in events:
            if event.get("event") == "lease_acquired":
                owners.add(event.get("pid"))
                require(len(owners) == 1, "two concurrent camera writers")
            elif event.get("event") == "lease_released":
                owners.discard(event.get("pid"))
        require(not owners, "camera ownership remains after fault")
        require("readonly_preview_observed" in names, "second process readonly status/preview missing")
    elif case == "sigterm":
        require(bool(commits) and bool(by_name("sigterm_sent")), "SIGTERM after committed frame missing")
        require("writer_drained" in names and "trigger_out_disabled_readback" in names,
                "SIGTERM drain/output gate close missing")
        require(record.get("stop_reason") == "SIGTERM" and (
            record.get("terminal_state") in {"COMPLETED", "COMPLETED_WITH_WARNINGS"}
            or record.get("cli_returncode") == 0 and record.get("clean_shutdown") is True), "safe terminal signal state absent")
        require(bool(record.get("formal_delivery") or record.get("emergency_delivery")), "signal-safe output missing")
    return errors


def validate_fault_campaign(root, binding):
    evidence, errors = {}, []
    for case in FAULT_CASES:
        path = Path(root) / "faults" / case
        try:
            from .campaign import verify_operation
            verify_operation(root, binding, "faults/" + case)
            record = read_json(path / "observations.json")
            errors.extend(bound_errors(record, binding, case))
            evidence[case] = record
            errors.extend(fault_errors(case, record, _events(path / "events.jsonl")))
            for kind in ("formal_delivery", "emergency_delivery"):
                if record.get(kind):
                    delivery = (path / record[kind]).resolve()
                    delivery.relative_to(Path(root).resolve())
                    value = read_json(delivery)
                    if kind == "formal_delivery":
                        if delivery.name != "video_delivery.json" or "emergency" in str(value.get("schema", "")):
                            errors.append(f"{case}: formal delivery invalid")
                        from panorama_demo.video_sdk import VideoPanoramaResult
                        session = (path / record["session_relative_path"]).resolve()
                        session.relative_to(Path(root).resolve())
                        VideoPanoramaResult.load(session, delivery.parent)
                    else:
                        if value.get("schema") != "gemini305-sdk-emergency-2d/v1":
                            errors.append(f"{case}: emergency schema invalid")
                        from panorama_demo.video_sdk import VideoPanoramaResult
                        session = (path / record["session_relative_path"]).resolve()
                        session.relative_to(Path(root).resolve())
                        VideoPanoramaResult.load(session, delivery.parent)
        except (OSError, ValueError, KeyError, TypeError, ImportError, AttributeError) as exc:
            errors.append(f"{case}: NOT_EXECUTED/invalid raw evidence: {exc}")
    return {"errors": errors, "evidence": evidence}
