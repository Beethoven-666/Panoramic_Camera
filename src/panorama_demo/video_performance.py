"""Timing accounting which keeps primary delivery separate from audits.

The delivery SLA is deliberately about the primary 2-D result.  Read-only
audit exports and offline annotation measurement are valuable evidence, but
must neither make a primary delivery miss its SLA nor make a validation run
look like it has production performance evidence.
"""

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
    audit_export_seconds: float = 0.0
    offline_evaluation_seconds: float = 0.0

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not name or name in self.stage_seconds:
            raise ValueError(f"Performance stage must be unique and non-empty: {name!r}")
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] = time.perf_counter() - started

    @contextmanager
    def audit_export(self) -> Iterator[None]:
        """Account a read-only archive export outside the primary SLA."""

        started = time.perf_counter()
        try:
            yield
        finally:
            self.audit_export_seconds += time.perf_counter() - started

    @contextmanager
    def offline_evaluation(self) -> Iterator[None]:
        """Account post-publication measurement outside the primary SLA."""

        started = time.perf_counter()
        try:
            yield
        finally:
            self.offline_evaluation_seconds += time.perf_counter() - started

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    def as_dict(self, *, maximum_post_seconds: float | None) -> dict[str, object]:
        elapsed = self.elapsed_seconds
        primary = max(0.0, elapsed - self.audit_export_seconds - self.offline_evaluation_seconds)
        return {
            "primary_post_capture_seconds": primary,
            "audit_export_seconds": self.audit_export_seconds,
            "offline_evaluation_seconds": self.offline_evaluation_seconds,
            "stage_seconds": dict(self.stage_seconds),
            "maximum_post_seconds": maximum_post_seconds,
            "within_post_capture_budget": (
                None if maximum_post_seconds is None else primary <= maximum_post_seconds
            ),
        }
