"""Append-only disk ledger and batch-fsynced recoverable RGB-D boundaries."""

from __future__ import annotations

import hashlib
import csv
import json
import os
from pathlib import Path
import time
from typing import Any, IO

from .sdk_state import atomic_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    return {name: manifest.get(name) for name in
            ("schema", "started_utc", "capture_mode", "device", "capture_options")}


class CommitJournal:
    def __init__(self, root: Path, *, interval_frames: int = 30, interval_seconds: float = 1) -> None:
        self.root = root.resolve()
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir(exist_ok=True)
        self.path = self.checkpoints / "committed.jsonl"
        self._handle = self.path.open("xb")
        self.interval_frames, self.interval_seconds = interval_frames, interval_seconds
        self.count = 0
        self._digest = hashlib.sha256()
        self._pending_paths: list[Path] = []
        self._last_boundary = time.monotonic()
        self._last_id: int | None = None
        self._last_timestamp: int | None = None
        self._safe = True
        self.last_checkpoint: dict[str, Any] | None = None

    def record(self, frame: Any, row: dict[str, Any], csv_handle: IO[str]) -> None:
        timestamp, frame_id = int(frame.timestamp_us), int(frame.frame_id)
        if (not self._safe or (self._last_id is not None and frame_id != self._last_id + 1)
                or (self._last_timestamp is not None and timestamp <= self._last_timestamp)):
            self._safe = False
            return
        self._last_id, self._last_timestamp = frame_id, timestamp
        payload = {"frame_id": frame_id, "row_index": self.count, "row": row,
                   "color_path": frame.color_path.relative_to(self.root).as_posix(),
                   "aligned_depth_path": frame.aligned_depth_path.relative_to(self.root).as_posix(),
                   "color_sha256": frame.color_sha256,
                   "aligned_depth_sha256": frame.aligned_depth_sha256,
                   "timestamp_us": timestamp}
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        self._handle.write(encoded)
        self._digest.update(encoded)
        self.count += 1
        self._pending_paths.extend((frame.color_path, frame.aligned_depth_path))
        if self.count == 1 or self.count % self.interval_frames == 0 or time.monotonic() - self._last_boundary >= self.interval_seconds:
            self.checkpoint(csv_handle)

    def checkpoint(self, csv_handle: IO[str]) -> None:
        if not self.count:
            return
        calibration = self.root / "calibration.json"
        manifest = self.root / "manifest.json"
        if not calibration.is_file() or not manifest.is_file():
            return
        csv_handle.flush()
        os.fsync(csv_handle.fileno())
        for path in [*self._pending_paths, calibration, manifest]:
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
        if os.name == "posix":
            for directory in (self.root, self.root / "color", self.root / "depth_aligned"):
                descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        document = json.loads(manifest.read_text(encoding="utf-8"))
        payload = {"schema": "gemini305-durable-commit/v1", "count": self.count,
                   "ledger_bytes": self._handle.tell(), "ledger_sha256": self._digest.hexdigest(),
                   "calibration_sha256": file_sha256(calibration),
                   "manifest_identity": manifest_identity(document),
                   "last_frame_id": self._last_id, "last_timestamp_us": self._last_timestamp,
                   "created_monotonic_ns": time.monotonic_ns()}
        if self.last_checkpoint is not None and self.last_checkpoint["count"] != self.count:
            atomic_json(self.checkpoints / "previous_commit.json", self.last_checkpoint, durable=True)
        atomic_json(self.checkpoints / "current_commit.json", payload, durable=True)
        self.last_checkpoint = payload
        self._pending_paths.clear()
        self._last_boundary = time.monotonic()

    def close(self) -> None:
        self._handle.close()


def read_durable_prefix(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    failures = []
    for name in ("current_commit.json", "previous_commit.json"):
        try:
            checkpoint = json.loads((root / "checkpoints" / name).read_text(encoding="utf-8"))
            if checkpoint["schema"] != "gemini305-durable-commit/v1":
                raise ValueError("Unsupported durable checkpoint")
            if file_sha256(root / "calibration.json") != checkpoint["calibration_sha256"]:
                raise ValueError("Calibration changed")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest_identity(manifest) != checkpoint["manifest_identity"]:
                raise ValueError("Capture manifest identity changed")
            with (root / "checkpoints" / "committed.jsonl").open("rb") as handle:
                data = handle.read(int(checkpoint["ledger_bytes"]))
            if hashlib.sha256(data).hexdigest() != checkpoint["ledger_sha256"]:
                raise ValueError("Durable ledger changed")
            rows = [json.loads(line) for line in data.splitlines()]
            if len(rows) != checkpoint["count"] or not rows:
                raise ValueError("Durable ledger count mismatch")
            with (root / "frames.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in rows:
                    actual = next(reader, None)
                    expected = {key: "" if value is None else str(value)
                                for key, value in row["row"].items()}
                    if actual != expected:
                        raise ValueError("Committed CSV prefix differs from durable ledger")
            for index, row in enumerate(rows):
                if row["row_index"] != index or (index and row["frame_id"] != rows[index - 1]["frame_id"] + 1):
                    raise ValueError("Durable ledger is not contiguous")
                if index and row["timestamp_us"] <= rows[index - 1]["timestamp_us"]:
                    raise ValueError("Durable ledger timestamp regressed")
                for field in ("color", "aligned_depth"):
                    path = (root / row[f"{field}_path"]).resolve()
                    path.relative_to(root)
                    if file_sha256(path) != row[f"{field}_sha256"]:
                        raise ValueError(f"Committed {field} bytes changed at {row['frame_id']}")
            return rows, {**checkpoint, "checkpoint_used": name, "rollback_reasons": failures}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(f"{name}: {exc}")
    raise ValueError("No valid durable committed prefix: " + "; ".join(failures))
