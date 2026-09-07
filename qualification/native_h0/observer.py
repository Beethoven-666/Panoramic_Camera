"""External, non-mutating observations of the frozen candidate's Python boundaries.

Install around the SDK call or the CLI main function in the *same process*.
The installed candidate functions are always called unmodified and their return
values/exceptions are preserved. JSONL is streamed so endurance does not retain
one Python object per frame. These are observations, never qualification flags.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
import threading
import time
import weakref


class NativeCaptureObserver:
    def __init__(self, observation_path: Path):
        self.path = Path(observation_path).resolve()
        self.events_path = self.path.with_suffix(".jsonl")
        self._patches = []
        self._lock = threading.Lock()
        self._errors = []
        self._sequence = 0
        self._writer = None
        self._preview = None
        self._leases = {}
        self._capture_active = False

    def _event(self, kind, **data):
        with self._lock:
            self._sequence += 1
            record = {"sequence": self._sequence, "kind": kind,
                      "monotonic_ns": time.monotonic_ns(), "pid": os.getpid(), **data}
            try:
                self._stream.write(json.dumps(record, default=str, allow_nan=False) + "\n")
                self._stream.flush()
            except Exception as exc:
                self._errors.append(f"{type(exc).__name__}: {exc}")

    def _wrap(self, owner, name, after):
        original = getattr(owner, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self._event("boundary_error", boundary=name, error_type=type(exc).__name__,
                            error_code=getattr(exc, "error_code", None), message=str(exc))
                raise
            try:
                after(result, args, kwargs)
            except Exception as exc:
                self._errors.append(f"{name}: {type(exc).__name__}: {exc}")
            return result

        self._patches.append((owner, name, original))
        setattr(owner, name, observed)

    def __enter__(self):
        from panorama_demo import capture_orbbec as capture
        from panorama_demo.camera_lease import CameraLease
        from panorama_demo.photo_capture import SoftwareTriggeredRGBDPhotoController as Photo
        from panorama_demo.video_s13_live import S13V11LiveObserver as Preview
        from panorama_demo.commit_journal import CommitJournal
        from panorama_demo.sdk_state import JobRecord

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Both files are exclusive: stale evidence must never be inherited.
        self._summary_stream = self.path.open("x", encoding="utf-8")
        try:
            self._stream = self.events_path.open("x", encoding="utf-8")
        except Exception:
            self._summary_stream.close()
            raise
        self._event("observer_started")
        def writer_created(result, args, kwargs):
            self._writer = weakref.ref(args[0])
            self._event("writer_created", session=str(args[0].root.resolve()))

        self._wrap(capture.SessionWriter, "__init__", writer_created)
        self._wrap(capture.SessionWriter, "close", lambda r, a, k: self._event(
            "writer_closed", session=str(a[0].root.resolve()), written_frames=a[0].stats.written,
            queue_drops=a[0].stats.queue_drops, write_errors=a[0].stats.write_errors,
            writer_errors=a[0].stats.errors, queue_depth=a[0].queue.qsize(),
            worker_alive=a[0]._thread.is_alive()))
        self._wrap(CommitJournal, "record", lambda r, a, k: self._event(
            "frame_committed", session=str(a[0].root), frame_id=a[1].frame_id,
            committed_count=a[0].count, durable_count=(None if a[0].last_checkpoint is None
                                                      else a[0].last_checkpoint["count"])))
        self._wrap(CommitJournal, "checkpoint", lambda r, a, k: self._event(
            "durable_checkpoint", session=str(a[0].root), checkpoint=a[0].last_checkpoint))
        self._wrap(JobRecord, "_persist", lambda r, a, k: self._event(
            "job_state", job_root=str(a[0].root), event=a[1], payload=dict(a[0].payload)))
        self._wrap(capture, "_discover_video_device",
                   lambda r, a, k: self._event("device_discovered", device=capture._device_info(r[1])))
        self._wrap(capture, "_verify_external_sync_output",
                   lambda r, a, k: self._event("sync_readback", readback=r))
        self._wrap(capture.SessionWriter, "submit", lambda r, a, k: self._event(
            "frame_submitted", session=str(a[0].root.resolve()), frame_id=a[1].frame_id,
            accepted=r, metadata=a[1].metadata, queue_depth=a[0].queue.qsize(),
            queue_capacity=a[0].queue.maxsize))
        self._wrap(capture, "_shutdown_video_capture_hardware", lambda r, a, k: self._event(
            "hardware_shutdown", pipeline_was_started=k["pipeline_started"],
            shutdown=r[0], errors=r[1]))
        def lease_acquired(result, args, kwargs):
            lease = args[0]
            self._leases[lease.serial] = weakref.ref(lease)
            self._event("lease_acquired", serial=lease.serial, held=lease.held, path=str(lease.path))

        self._wrap(CameraLease, "acquire", lease_acquired)
        self._wrap(CameraLease, "release", lambda r, a, k: self._event(
            "lease_released", serial=a[0].serial, held=a[0].held, path=str(a[0].path)))
        def preview_ready(result, args, kwargs):
            self._preview = weakref.ref(args[0])
            self._capture_active = True
            self._event("capture_started", session=str(args[1].root.resolve()),
                        capture_started_monotonic_ns=args[1].capture_started_monotonic_ns)

        self._wrap(Preview, "on_session_ready", preview_ready)
        def capture_stopped(result, args, kwargs):
            self._capture_active = False
            self._event("capture_stopped")

        self._wrap(Preview, "on_capture_stopping", capture_stopped)
        self._wrap(Preview, "_request_preview", lambda r, a, k: self._event(
            "preview_request", frame_id=a[1].frame_id,
            writer_queue_fraction=a[1].writer_queue_fraction,
            writer_queue_drops=a[1].writer_queue_drops,
            preview_queue_depth=a[0]._preview_queue.qsize(),
            preview_queue_capacity=a[0]._preview_queue.maxsize))
        self._wrap(Photo, "capture_once", lambda r, a, k: self._event(
            "photo_pair_completed", frame_id=r.frame_id))
        self._wrap(Photo, "_verify_sync_config", lambda r, a, k: self._event(
            "sync_readback", readback=r))

        def photo_closed(result, args, kwargs):
            controller = args[0]
            record = {"pipeline_stopped": not controller._pipeline_started,
                      "writer_closed": controller._writer_closed,
                      "device_restored": controller._device_restored}
            if controller._device is not None:
                record["sync_readback"] = capture._sync_config_to_dict(
                    controller._device.get_multi_device_sync_config())
                record["physical_output_gate"] = controller._read_bool(
                    controller._property("OB_PROP_SYNC_SIGNAL_TRIGGER_OUT_BOOL"), "H0 output gate readback")
            self._event("photo_closed", **record)

        self._wrap(Photo, "close", photo_closed)

        def replaced(result, args, kwargs):
            target = Path(args[1])
            if target.name == "live_preview_state.json":
                state = json.loads(target.read_text(encoding="utf-8"))
                self._event("preview_state_published", path=str(target.resolve()), state=state,
                            image_exists=(target.parent / "live_preview.jpg").is_file())

        self._wrap(os, "replace", replaced)
        return self

    def snapshot_runtime(self):
        """Read active writer/P0 state. Missing measurements stay explicit."""
        missing = {"status": "NOT_EXECUTED", "reason": "component not active"}
        result = {key: dict(missing) for key in (
            "writer_queue_depth", "preview_queue_depth", "committed_frame_count", "checkpoint_count",
            "sealed_prefix_length", "mutable_tail_length", "lease_owned", "write_bytes_per_second")}
        result["capture_active"] = self._capture_active
        writer = None if self._writer is None else self._writer()
        if writer is not None:
            result.update(writer_queue_depth=writer.queue.qsize(), committed_frame_count=writer.stats.written)
        preview = None if self._preview is None else self._preview()
        if preview is not None:
            result["preview_queue_depth"] = preview._preview_queue.qsize()
            engine = preview._authority_engine
            if engine is not None:
                snapshot = engine.snapshot()
                if snapshot is not None:
                    frontier = snapshot.frontiers
                    result.update(sealed_prefix_length=frontier.sealed_source_count,
                                  mutable_tail_length=frontier.selected_source_count - frontier.sealed_source_count,
                                  checkpoint_count=len(engine._checkpoint_chain.checkpoints))
            elif preview._shadow_failure_reason is None:
                result.update(checkpoint_count=0, sealed_prefix_length=0, mutable_tail_length=0)
        leases = [ref() for serial, ref in self._leases.items() if serial != "discovery"]
        if leases:
            result["lease_owned"] = all(lease is not None and lease.held for lease in leases)
        # Storage throughput belongs to the external process/disk sampler. This
        # method does not scan a growing two-hour session once per second.
        return result

    def __exit__(self, exc_type, exc, traceback):
        for owner, name, original in reversed(self._patches):
            setattr(owner, name, original)
        self._event("observer_stopped", operation_error=None if exc is None else type(exc).__name__)
        self._stream.close()
        json.dump({"schema": "gemini305-native-h0-observation/v1", "completed": True,
                   "operation_error": None if exc is None else type(exc).__name__,
                   "events_path": str(self.events_path), "event_count": self._sequence,
                   "observer_errors": self._errors}, self._summary_stream, indent=2)
        self._summary_stream.close()


def read_observation(path: Path):
    path = Path(path).resolve()
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (summary.get("schema") != "gemini305-native-h0-observation/v1"
            or summary.get("completed") is not True or summary.get("observer_errors") != []):
        raise ValueError("Incomplete native observation")
    events_path = path.with_suffix(".jsonl")
    if Path(summary["events_path"]).resolve() != events_path:
        raise ValueError("Observation event path mismatch")
    count = 0
    with events_path.open(encoding="utf-8") as stream:
        for count, line in enumerate(stream, 1):
            record = json.loads(line)
            if record.get("sequence") != count or type(record.get("monotonic_ns")) is not int:
                raise ValueError("Observation event sequence is incomplete")
            yield record
    if count != summary.get("event_count") or count < 2:
        raise ValueError("Observation event count mismatch")
