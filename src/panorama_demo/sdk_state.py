"""Durable SDK job states and structured errors; no camera or image imports."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

from .version import __version__


class SDKError(RuntimeError):
    default_code = "SDK_ERROR"

    def __init__(self, message: str, *, error_code: str | None = None,
                 stage: str = "CREATED", recoverable: bool = False,
                 artifact_path: str | Path | None = None,
                 cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.error_code = error_code or self.default_code
        self.stage = stage
        self.recoverable = recoverable
        self.artifact_path = None if artifact_path is None else str(artifact_path)
        self.cause_type = None if cause is None else type(cause).__name__

    def as_dict(self) -> dict[str, object]:
        return {"error_code": self.error_code, "stage": self.stage,
                "recoverable": self.recoverable, "message": str(self),
                "artifact_path": self.artifact_path, "cause_type": self.cause_type}


class SDKBusyError(SDKError):
    default_code = "CAMERA_BUSY"


class CaptureError(SDKError):
    default_code = "CAPTURE_ERROR"


class ThreeDProcessingError(SDKError):
    default_code = "THREE_D_PROCESSING_ERROR"


class JobState(str, Enum):
    CREATED = "CREATED"
    WAITING_FOR_CAMERA = "WAITING_FOR_CAMERA"
    WARMING_UP = "WARMING_UP"
    CAPTURING = "CAPTURING"
    STOPPING = "STOPPING"
    FINALIZING_2D = "FINALIZING_2D"
    PUBLISHED_2D = "PUBLISHED_2D"
    PROCESSING_3D = "PROCESSING_3D"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"
    CANCELLED_NO_DATA = "CANCELLED_NO_DATA"
    # Existing video SDK names retain their terminal meanings.
    RUNNING = "CAPTURING"
    SUCCEEDED = "COMPLETED"
    CANCELLED = "CANCELLED_NO_DATA"


class StopReason(str, Enum):
    USER_REQUEST = "USER_REQUEST"
    DURATION_REACHED = "DURATION_REACHED"
    MAX_FRAMES_REACHED = "MAX_FRAMES_REACHED"
    LOW_DISK = "LOW_DISK"
    SIGINT = "SIGINT"
    SIGTERM = "SIGTERM"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    FRAME_TIMEOUT = "FRAME_TIMEOUT"
    USB_TRANSPORT_ERROR = "USB_TRANSPORT_ERROR"
    WRITER_ERROR = "WRITER_ERROR"
    ONLINE_CHECKPOINT_FAILED = "ONLINE_CHECKPOINT_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    TIMESTAMP_REGRESSION = "TIMESTAMP_REGRESSION"


class CompletionState(str, Enum):
    FORMAL_2D_PUBLISHED = "FORMAL_2D_PUBLISHED"
    EMERGENCY_2D_PUBLISHED = "EMERGENCY_2D_PUBLISHED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED_NO_VALID_FRAME = "FAILED_NO_VALID_FRAME"
    CANCELLED_NO_DATA = "CANCELLED_NO_DATA"


class RetentionPolicy(str, Enum):
    DELETE_AFTER_ALL_SUCCESS = "delete_after_all_success"
    KEEP = "keep"


TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.COMPLETED_WITH_WARNINGS,
                            JobState.FAILED, JobState.CANCELLED_NO_DATA})
_TRANSITIONS = {
    JobState.CREATED: {JobState.WAITING_FOR_CAMERA, JobState.WARMING_UP,
                       JobState.FINALIZING_2D, JobState.CANCELLED_NO_DATA},
    JobState.WAITING_FOR_CAMERA: {JobState.WARMING_UP, JobState.STOPPING,
                                  JobState.CANCELLED_NO_DATA},
    JobState.WARMING_UP: {JobState.CAPTURING, JobState.WAITING_FOR_CAMERA,
                          JobState.STOPPING, JobState.CANCELLED_NO_DATA},
    JobState.CAPTURING: {JobState.STOPPING},
    JobState.STOPPING: {JobState.FINALIZING_2D, JobState.CANCELLED_NO_DATA, JobState.WAITING_FOR_CAMERA},
    JobState.FINALIZING_2D: {JobState.PUBLISHED_2D, JobState.COMPLETED_WITH_WARNINGS},
    JobState.PUBLISHED_2D: {JobState.PROCESSING_3D, JobState.COMPLETED,
                           JobState.COMPLETED_WITH_WARNINGS},
    JobState.PROCESSING_3D: {JobState.COMPLETED, JobState.COMPLETED_WITH_WARNINGS},
}


def atomic_json(path: Path, payload: Mapping[str, object], *, durable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    os.replace(pending, path)
    if durable and os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class JobRecord:
    """One controller writes; any client can reload the atomic snapshot."""

    def __init__(self, root: Path, payload: dict[str, Any]) -> None:
        self.root = root.resolve()
        self.payload = payload
        self._lock = threading.RLock()

    @classmethod
    def create(cls, root: Path, *, source_commit: str, config_sha: str,
               owns_session: bool) -> "JobRecord":
        root.mkdir(parents=True, exist_ok=False)
        for name in ("preview", "2d", "3d", "checkpoints", "failures", "cleanup"):
            (root / name).mkdir()
        record = cls(root, {"schema": "gemini305-sdk-job/v1", "job_id": root.name,
                           "state": JobState.CREATED.value, "sdk_version": __version__,
                           "source_commit": source_commit, "production_config_sha256": config_sha,
                           "pid": os.getpid(), "camera_serial": None,
                           "owns_session": owns_session, "stop_reason": None,
                           "completion_state": None, "result": None, "warnings": []})
        record._persist("created")
        return record

    @classmethod
    def load(cls, root: Path) -> "JobRecord":
        payload = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if payload.get("schema") != "gemini305-sdk-job/v1":
            raise ValueError("Unsupported job state schema")
        JobState(payload["state"])
        return cls(root, payload)

    @property
    def state(self) -> JobState:
        return JobState(self.payload["state"])

    def _persist(self, event: str) -> None:
        self.payload["updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.payload["updated_monotonic_ns"] = time.monotonic_ns()
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": event, **self.payload}) + "\n")
            handle.flush()
        atomic_json(self.root / "state.json", self.payload)

    def update(self, **fields: Any) -> None:
        if "state" in fields:
            raise ValueError("Use transition() to change job state")
        with self._lock:
            self.payload.update(fields)
            self._persist("updated")

    def transition(self, target: JobState, **fields: Any) -> None:
        target = JobState(target)
        with self._lock:
            current = self.state
            if (current is JobState.STOPPING and target is JobState.WAITING_FOR_CAMERA
                    and self.payload.get("committed_frames", 0) != 0):
                raise ValueError("Cannot reconnect after a committed frame")
            if current in TERMINAL_STATES or (
                target not in _TRANSITIONS.get(current, set()) and target != JobState.FAILED
            ):
                raise ValueError(f"Invalid job transition: {current.value} -> {target.value}")
            self.payload.update(fields)
            self.payload["state"] = target.value
            self._persist("transition")

    def request_stop(self, reason: StopReason) -> None:
        with self._lock:
            if self.payload["stop_reason"] is None:
                self.payload["stop_reason"] = StopReason(reason).value
                atomic_json(self.root / "control.json", {"stop_reason": reason.value})
                self._persist("stop_requested")
