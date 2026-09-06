"""Bounded-rate disk checks driven by committed bytes, before accepting frames."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Callable

from .sdk_state import CaptureError


def disk_space(path: Path) -> tuple[int, int | None]:
    ancestor = path.resolve()
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    inodes = os.statvfs(ancestor).f_favail if hasattr(os, "statvfs") else None
    return free, inodes


class DiskGuard:
    def __init__(self, capture_root: Path, output_root: Path, *, reserve_gib: float = 10,
                 minimum_free_capture_seconds: float = 120,
                 final_2d_scratch_bytes: int = 512 * 1024**2,
                 manifest_log_bytes: int = 16 * 1024**2,
                 post_3d_bytes: int = 0,
                 space: Callable[[Path], tuple[int, int | None]] = disk_space,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if reserve_gib <= 0 or minimum_free_capture_seconds <= 0:
            raise ValueError("Disk reserve and capture runway must be positive")
        self.capture_root, self.output_root = Path(capture_root), Path(output_root)
        self.reserve = int(reserve_gib * 1024**3)
        self.runway = minimum_free_capture_seconds
        self.scratch = final_2d_scratch_bytes + manifest_log_bytes
        self.post_3d_bytes = post_3d_bytes
        self.space, self.clock = space, clock
        self._samples: deque[tuple[float, int]] = deque(maxlen=600)
        self._lock = threading.Lock()
        self._last_check = float("-inf")
        self._last_frame = -1
        self.last_report: dict[str, object] = {}

    def committed(self, byte_count: int) -> None:
        with self._lock:
            self._samples.append((self.clock(), int(byte_count)))

    def required_reserve(self) -> int:
        with self._lock:
            samples = tuple(self._samples)
        rate = 40 * 1024**2  # Conservative initial RGB-D write rate until measured.
        if len(samples) > 1 and samples[-1][0] > samples[0][0]:
            rate = sum(item[1] for item in samples[1:]) / (samples[-1][0] - samples[0][0])
        return max(self.reserve, int(rate * self.runway) + self.scratch)

    def check(self, frame_count: int, *, force: bool = False, preflight: bool = False) -> bool:
        now = self.clock()
        if not force and now - self._last_check < 1 and frame_count - self._last_frame < 60:
            return bool(self.last_report.get("capture_allowed", True))
        self._last_check, self._last_frame = now, frame_count
        required = self.required_reserve()
        capture_free, capture_inodes = self.space(self.capture_root)
        output_free, output_inodes = self.space(self.output_root)
        output_required = self.scratch + (self.post_3d_bytes if preflight else 0)
        allowed = (capture_free > required and output_free > max(self.reserve, output_required)
                   and (capture_inodes is None or capture_inodes > 1024)
                   and (output_inodes is None or output_inodes > 128))
        self.last_report = {"capture_allowed": allowed, "capture_free_bytes": capture_free,
                            "output_free_bytes": output_free, "capture_free_inodes": capture_inodes,
                            "output_free_inodes": output_inodes, "stop_threshold_bytes": required,
                            "warning_threshold_bytes": int(required * 1.25),
                            "warning": capture_free <= required * 1.25,
                            "checked_monotonic": now, "committed_frame_count": frame_count}
        if preflight and not allowed:
            raise CaptureError("Insufficient disk reserve for capture and finalization",
                               error_code="LOW_DISK", recoverable=True)
        return allowed

    def preflight(self) -> None:
        self.check(0, force=True, preflight=True)

    def three_d_allowed(self) -> bool:
        free, inodes = self.space(self.output_root)
        return free > self.reserve + self.post_3d_bytes and (inodes is None or inodes > 1024)
