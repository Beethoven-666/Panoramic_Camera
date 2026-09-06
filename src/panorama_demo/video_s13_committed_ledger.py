"""Small-offset-index disk sequence for capture-time committed identities."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
import threading
from typing import Any, Callable


class CommittedLedger(Sequence):
    def __init__(self, path: Path, factory: Callable[..., Any]) -> None:
        self.path, self.factory = path, factory
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=False)
        self._offsets = array("Q")
        self._lock = threading.Lock()

    def append(self, identity: Any) -> None:
        payload = asdict(identity)
        for field in ("color_path", "aligned_depth_path"):
            payload[field] = str(payload[field])
        encoded = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        with self._lock, self.path.open("ab") as handle:
            offset = handle.tell()
            handle.write(encoded)
            handle.flush()
            self._offsets.append(offset)

    def __len__(self) -> int:
        with self._lock:
            return len(self._offsets)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        with self._lock:
            offset = self._offsets[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            payload = json.loads(handle.readline())
        for field in ("color_path", "aligned_depth_path"):
            payload[field] = Path(payload[field])
        return self.factory(**payload)
