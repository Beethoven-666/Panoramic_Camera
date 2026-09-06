"""Kernel-owned process lease, held through camera and writer shutdown."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import IO

from .sdk_state import CaptureError, SDKBusyError
from .version import __version__


class CameraLease:
    def __init__(self, root: Path, serial: str, job_id: str) -> None:
        if not serial or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for c in serial):
            raise ValueError("Camera serial must be a nonempty filesystem-safe SDK serial")
        self.root, self.serial, self.job_id = Path(root).expanduser().resolve(), serial, job_id
        self.path = self.root / f"camera-{serial}.lock"
        self._handle: IO[str] | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> "CameraLease":
        if self.held:
            return self
        if not self.root.is_dir():
            raise CaptureError("Camera lock root must be provisioned explicitly",
                               error_code="CAMERA_LOCK_ROOT_UNAVAILABLE", artifact_path=self.root)
        try:
            handle = self.path.open("a+", encoding="utf-8")
        except OSError as exc:
            raise CaptureError("Camera lock root is not writable",
                               error_code="CAMERA_LOCK_ROOT_UNAVAILABLE", cause=exc) from exc
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                # Windows development uses a kernel byte lock on the same file.
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            handle.close()
            raise SDKBusyError("Camera is controlled by another process", recoverable=True,
                               artifact_path=self.path, cause=exc) from exc
        self._handle = handle
        try:
            handle.seek(0)
            handle.truncate()
            json.dump({"pid": os.getpid(), "job_id": self.job_id,
                       "sdk_version": __version__, "camera_serial": self.serial,
                       "started_utc": datetime.now(timezone.utc).isoformat()}, handle)
            handle.flush()
        except BaseException:
            self.release()
            raise
        return self

    def release(self) -> None:
        if self._handle is not None:
            # Never unlink a lock file: waiters must lock the same inode.
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "CameraLease":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()
