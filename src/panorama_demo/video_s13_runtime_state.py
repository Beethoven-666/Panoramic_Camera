"""In-memory parent chain for the fast S1.3 P0--P3 pipeline.

Stage images are snapshots for people.  They are deliberately not an input to
the next stage: the result object is the only parent authority within a run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


class S13Stage(IntEnum):
    """The four image-producing stages in their only legal order."""

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3


@dataclass(slots=True)
class S13StageResult:
    """One in-memory stage result and its immutable lineage metadata."""

    run_id: str
    stage: S13Stage
    revision: int
    parent_revision: int | None
    image: np.ndarray
    valid: np.ndarray
    runtime_data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class S13RuntimeContext:
    """Own the current result and reject invalid or cross-run transitions."""

    run_id: str
    current: S13StageResult | None = None

    def require(self, expected: S13Stage) -> S13StageResult:
        current = self.current
        if current is None or current.stage != expected:
            actual = None if current is None else current.stage.name
            raise RuntimeError(f"Expected {expected.name}, got {actual}")
        return current

    @staticmethod
    def _validate_result(result: S13StageResult) -> None:
        image = np.asarray(result.image)
        valid = np.asarray(result.valid)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("S1.3 stage image must be uint8 BGR")
        if valid.ndim != 2 or valid.shape != image.shape[:2] or valid.dtype != np.bool_:
            raise ValueError("S1.3 stage valid mask must match the image")

    def initialize_p0(self, result: S13StageResult) -> S13StageResult:
        if self.current is not None:
            raise RuntimeError("P0 has already been initialized")
        self._validate_result(result)
        if result.run_id != self.run_id:
            raise RuntimeError("P0 belongs to another run")
        if result.stage != S13Stage.P0 or result.revision != 0:
            raise RuntimeError("Initial stage must be revision-zero P0")
        if result.parent_revision is not None:
            raise RuntimeError("P0 must not have a parent revision")
        self.current = result
        return result

    def commit(
        self, *, expected_parent: S13Stage, candidate: S13StageResult
    ) -> S13StageResult:
        parent = self.require(expected_parent)
        self._validate_result(candidate)
        if candidate.run_id != self.run_id:
            raise RuntimeError("Candidate belongs to another run")
        if candidate.parent_revision != parent.revision:
            raise RuntimeError("Candidate was not built from current parent")
        if candidate.revision != parent.revision + 1:
            raise RuntimeError("Candidate revision is not monotonic")
        if candidate.stage != S13Stage(parent.stage + 1):
            raise RuntimeError("Stage transition skipped or regressed")
        self.current = candidate
        return candidate


__all__ = ["S13RuntimeContext", "S13Stage", "S13StageResult"]
