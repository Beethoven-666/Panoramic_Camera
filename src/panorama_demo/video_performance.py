"""Small, dependency-free performance accounting for video delivery."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class VideoPerformanceProfiler:
    """Record serial named stages from the post-capture delivery path."""

    started_at: float = field(default_factory=time.perf_counter)
    stage_seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not name or name in self.stage_seconds:
            raise ValueError(f"Performance stage must be unique and non-empty: {name!r}")
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] = time.perf_counter() - started

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def as_dict(self, *, maximum_post_seconds: float | None) -> dict[str, object]:
        elapsed = self.elapsed_seconds
        return {
            "post_capture_seconds": elapsed,
            "stage_seconds": dict(self.stage_seconds),
            "maximum_post_seconds": maximum_post_seconds,
            "within_post_capture_budget": (
                None if maximum_post_seconds is None else elapsed <= maximum_post_seconds
            ),
        }
