"""Auditable disabled source-u luminance-field contract for S013 M6.1.

M6.1 deliberately has no non-zero spatial photometric solver.  Keeping the
zero payload explicit prevents a later caller from smuggling an unversioned
field into the formal renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


LUMINANCE_FIELD_SCHEMA = "gemini305-video-s13-source-u-luminance-field/v1"


@dataclass(frozen=True)
class S13LuminanceFieldDisabledConfig:
    enabled: bool = False
    solver: str = "disabled"

    def __post_init__(self) -> None:
        if self.enabled is not False or self.solver != "disabled":
            raise ValueError("S013 M6.1 permits only the disabled Q4 contract")


def canonical_zero_luminance_fields(
    frame_ids: Sequence[int], *, sample_count: int = 0
) -> dict[str, np.ndarray]:
    """Return the sole legal Q4 payload, ordered by real source/frame."""

    if isinstance(sample_count, bool) or int(sample_count) != sample_count or sample_count < 0:
        raise ValueError("S013 luminance-field sample_count must be a non-negative integer")
    ids = np.asarray(tuple(int(value) for value in frame_ids), dtype=np.int32)
    if ids.ndim != 1 or len(set(ids.tolist())) != ids.size:
        raise ValueError("S013 luminance-field frame order must contain unique real frames")
    return {
        "frame_id": ids,
        "source_u_field": np.zeros((ids.size, int(sample_count)), dtype=np.float32),
    }


def validate_disabled_luminance_fields(
    arrays: dict[str, np.ndarray], *, frame_ids: Sequence[int]
) -> None:
    if set(arrays) != {"frame_id", "source_u_field"}:
        raise ValueError("S013 luminance-field payload fields disagree")
    expected_ids = np.asarray(tuple(int(value) for value in frame_ids), dtype=np.int32)
    actual_ids = np.asarray(arrays["frame_id"])
    field = np.asarray(arrays["source_u_field"])
    if actual_ids.dtype != np.int32 or not np.array_equal(actual_ids, expected_ids):
        raise ValueError("S013 luminance-field source/frame order disagrees")
    if field.dtype != np.float32 or field.ndim != 2 or field.shape[0] != expected_ids.size:
        raise ValueError("S013 luminance-field zero payload shape/type disagrees")
    if np.any(field != 0.0) or not np.isfinite(field).all():
        raise ValueError("S013 M6.1 rejects every non-zero luminance field")


def luminance_field_document(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    field = np.asarray(arrays["source_u_field"])
    return {
        "schema": LUMINANCE_FIELD_SCHEMA,
        "q4_state": "disabled",
        "solver_invocations": 0,
        "canonical_zero": True,
        "field_dtype": str(field.dtype),
        "field_shape": list(field.shape),
    }


def solve_nonzero_luminance_field(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("non-zero Q4 is not authorized for S013 M6.1")


__all__ = [
    "LUMINANCE_FIELD_SCHEMA", "S13LuminanceFieldDisabledConfig",
    "canonical_zero_luminance_fields", "luminance_field_document",
    "solve_nonzero_luminance_field", "validate_disabled_luminance_fields",
]
