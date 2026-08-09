"""Pure helpers for the S01 local vertical inverse-map correction.

The functions in this module only alter the vertical coordinate of an
already-built calibrated inverse map.  They neither decode nor resample RGB;
the caller can therefore compose every accepted pair correction first and do
the sole full-resolution remap afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class VerticalWarpAudit:
    finite: bool
    monotonic: bool
    within_slope_limit: bool
    maximum_abs_slope: float
    minimum_mapping_step: float

    @property
    def safe(self) -> bool:
        return self.finite and self.monotonic and self.within_slope_limit

    def as_dict(self) -> dict[str, bool | float]:
        return {
            "finite": self.finite,
            "monotonic": self.monotonic,
            "within_slope_limit": self.within_slope_limit,
            "maximum_abs_slope": self.maximum_abs_slope,
            "minimum_mapping_step": self.minimum_mapping_step,
            "safe": self.safe,
        }


def cosine_fade_weights(
    canvas_x: np.ndarray,
    *,
    boundary_x: float,
    shoulder_width_px: float,
    side: Literal["left", "right"],
) -> np.ndarray:
    """Return a seam-local cosine fade, one at the boundary and zero away.

    ``left`` covers ``[boundary-width, boundary]`` and ``right`` covers
    ``[boundary, boundary+width]``.  Values outside that shoulder are exactly
    zero, which keeps the central owner area unchanged.
    """

    x = np.asarray(canvas_x, dtype=np.float64)
    if x.ndim not in {1, 2} or not np.isfinite(x).all():
        raise ValueError("canvas_x must be a finite one- or two-dimensional array")
    if not np.isfinite(boundary_x):
        raise ValueError("boundary_x must be finite")
    if not np.isfinite(shoulder_width_px) or shoulder_width_px <= 0.0:
        raise ValueError("shoulder_width_px must be finite and positive")
    if side == "left":
        distance = boundary_x - x
    elif side == "right":
        distance = x - boundary_x
    else:
        raise ValueError("side must be 'left' or 'right'")
    inside = (distance >= 0.0) & (distance <= shoulder_width_px)
    phase = np.clip(distance / shoulder_width_px, 0.0, 1.0)
    weights = 0.5 * (1.0 + np.cos(np.pi * phase))
    return np.where(inside, weights, 0.0)


def cosine_shoulder_weights(
    canvas_x: np.ndarray,
    *,
    boundary_x: float,
    shoulder_width_px: float,
    side: Literal["left", "right"],
) -> np.ndarray:
    """Backward-compatible descriptive alias for :func:`cosine_fade_weights`."""

    return cosine_fade_weights(
        canvas_x,
        boundary_x=boundary_x,
        shoulder_width_px=shoulder_width_px,
        side=side,
    )


def sample_vertical_correction(
    canvas_y: np.ndarray,
    curve_y: np.ndarray,
    dy_px: np.ndarray,
    *,
    gain: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Sample a 1-D ``dy(y)`` curve, returning zero outside its support."""

    query = np.asarray(canvas_y, dtype=np.float64)
    y = np.asarray(curve_y, dtype=np.float64)
    dy = np.asarray(dy_px, dtype=np.float64)
    if query.ndim != 1 or y.ndim != 1 or dy.ndim != 1 or y.size != dy.size:
        raise ValueError("canvas_y, curve_y and dy_px must be aligned one-dimensional arrays")
    if y.size < 2 or not np.isfinite(query).all() or not np.isfinite(y).all() or not np.isfinite(dy).all():
        raise ValueError("Vertical correction coordinates must be finite and contain two points")
    if np.any(np.diff(y) <= 0.0):
        raise ValueError("curve_y must be strictly increasing")
    gain_array = np.asarray(gain, dtype=np.float64)
    if gain_array.ndim == 0:
        if not np.isfinite(gain_array) or not 0.0 <= float(gain_array) <= 1.0:
            raise ValueError("gain must be finite and in [0, 1]")
        controlled = dy * float(gain_array)
    else:
        if gain_array.shape != dy.shape or not np.isfinite(gain_array).all():
            raise ValueError("Per-control gain must be finite and match dy_px")
        if np.any((gain_array < 0.0) | (gain_array > 1.0)):
            raise ValueError("Per-control gain must be in [0, 1]")
        controlled = dy * gain_array
    return np.interp(query, y, controlled, left=0.0, right=0.0)


def symmetric_pair_vertical_offsets(
    canvas_x: np.ndarray,
    canvas_y: np.ndarray,
    *,
    boundary_x: float,
    left_shoulder_px: float,
    right_shoulder_px: float,
    curve_y: np.ndarray,
    dy_px: np.ndarray,
    gain: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the two half-corrections for one adjacent-source boundary.

    Positive ``dy_px`` means the matching right-image feature is on a lower
    row.  The left source therefore samples half a row upward and the right
    source half a row downward in inverse-map coordinates.  Each result is
    nonzero only on its own side of the owner boundary.
    """

    x = np.asarray(canvas_x, dtype=np.float64)
    y = np.asarray(canvas_y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("canvas_x and canvas_y must be one-dimensional grids")
    vertical = sample_vertical_correction(y, curve_y, dy_px, gain=gain)
    left_weight = cosine_shoulder_weights(
        x, boundary_x=boundary_x, shoulder_width_px=left_shoulder_px, side="left"
    )
    right_weight = cosine_shoulder_weights(
        x, boundary_x=boundary_x, shoulder_width_px=right_shoulder_px, side="right"
    )
    left = -0.5 * vertical[:, None] * left_weight[None, :]
    right = 0.5 * vertical[:, None] * right_weight[None, :]
    return left, right


def _curve_arrays(curve: object) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(curve, Mapping):
        y = curve.get("y")
        dy = curve.get("dy_applied", curve.get("dy_smoothed", curve.get("dy_px")))
    elif isinstance(curve, tuple) and len(curve) == 2:
        y, dy = curve
    else:
        y = getattr(curve, "y", None)
        dy = getattr(
            curve,
            "dy_applied",
            getattr(curve, "dy_smoothed", getattr(curve, "dy_px", None)),
        )
    if y is None or dy is None:
        raise TypeError("curve must expose y and dy_applied/dy_smoothed, or be a (y, dy) tuple")
    return np.asarray(y, dtype=np.float64), np.asarray(dy, dtype=np.float64)


def compose_pair_vertical_warp(
    map_y: np.ndarray,
    *,
    boundary_x: float,
    curve: object,
    side: Literal["left", "right"],
    shoulder_width_px: float,
    canvas_x: np.ndarray | None = None,
    gain: float | np.ndarray = 1.0,
    symmetric_half_correction: bool = True,
    maximum_local_slope: float = 0.08,
) -> np.ndarray:
    """Compose one side of one pair curve into an existing vertical map.

    ``curve`` may be a ``VerticalCurve``-like object/mapping or a ``(y, dy)``
    tuple.  The returned array is new; ``map_y`` is never modified.  A left
    source takes the positive half correction and a right source the negative
    half correction.
    """

    base = np.asarray(map_y)
    if base.ndim != 2 or not np.isfinite(base).all():
        raise ValueError("map_y must be a finite two-dimensional inverse map")
    x = (
        np.arange(base.shape[1], dtype=np.float64)
        if canvas_x is None
        else np.asarray(canvas_x, dtype=np.float64)
    )
    if x.shape != (base.shape[1],):
        raise ValueError("canvas_x must contain one coordinate per inverse-map column")
    curve_y, dy_px = _curve_arrays(curve)
    vertical = sample_vertical_correction(
        np.arange(base.shape[0], dtype=np.float64), curve_y, dy_px, gain=gain
    )
    weight = cosine_fade_weights(
        x,
        boundary_x=boundary_x,
        shoulder_width_px=shoulder_width_px,
        side=side,
    )
    sign = 1.0 if side == "left" else -1.0
    scale = 0.5 if symmetric_half_correction else 1.0
    offset = sign * scale * vertical[:, None] * weight[None, :]
    audit = audit_vertical_offsets(offset, maximum_local_slope=maximum_local_slope)
    if not audit.safe:
        raise ValueError("Pair vertical correction failed finite, monotonic, or slope safety audit")
    return np.ascontiguousarray(base.astype(np.float64, copy=False) + offset)


def audit_vertical_offsets(
    vertical_offset: np.ndarray, *, maximum_local_slope: float = 0.08
) -> VerticalWarpAudit:
    """Audit ``source_y = canvas_y + offset`` along every image column."""

    offset = np.asarray(vertical_offset, dtype=np.float64)
    if offset.ndim != 2 or offset.shape[0] < 2:
        raise ValueError("vertical_offset must be a two-dimensional image with at least two rows")
    if not np.isfinite(maximum_local_slope) or not 0.0 < maximum_local_slope < 1.0:
        raise ValueError("maximum_local_slope must be finite and in (0, 1)")
    finite = bool(np.isfinite(offset).all())
    if not finite:
        return VerticalWarpAudit(False, False, False, float("inf"), float("-inf"))
    slopes = np.diff(offset, axis=0)
    maximum = float(np.max(np.abs(slopes))) if slopes.size else 0.0
    minimum_step = float(np.min(1.0 + slopes)) if slopes.size else 1.0
    return VerticalWarpAudit(
        finite=True,
        monotonic=minimum_step > 0.0,
        within_slope_limit=maximum <= maximum_local_slope + 1e-12,
        maximum_abs_slope=maximum,
        minimum_mapping_step=minimum_step,
    )


def compose_final_inverse_map(
    base_inverse_x: np.ndarray,
    base_inverse_y: np.ndarray,
    vertical_offset: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    source_height: int | None = None,
    maximum_local_slope: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, VerticalWarpAudit]:
    """Compose a safe S1 offset into a base map without resampling pixels."""

    map_x = np.asarray(base_inverse_x)
    map_y = np.asarray(base_inverse_y)
    offset = np.asarray(vertical_offset, dtype=np.float64)
    if map_x.shape != map_y.shape or map_x.shape != offset.shape or map_x.ndim != 2:
        raise ValueError("Base inverse maps and vertical_offset must have the same 2-D shape")
    audit = audit_vertical_offsets(offset, maximum_local_slope=maximum_local_slope)
    if not audit.safe:
        raise ValueError("Vertical correction failed finite, monotonic, or slope safety audit")
    if valid_mask is None:
        valid = np.ones(map_x.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).copy()
        if valid.shape != map_x.shape:
            raise ValueError("valid_mask must match the inverse-map shape")
    base_finite = np.isfinite(map_x) & np.isfinite(map_y)
    if np.any(valid & ~base_finite):
        raise ValueError("A valid base inverse-map sample is not finite")
    composed_y = map_y.astype(np.float64, copy=False) + offset
    valid &= base_finite & np.isfinite(composed_y)
    if source_height is not None:
        if isinstance(source_height, bool) or int(source_height) < 2:
            raise ValueError("source_height must be at least two")
        valid &= (composed_y >= 0.0) & (composed_y <= int(source_height) - 1)
    # Keep the returned maps finite even where the validity mask excludes a
    # sample, simplifying downstream cv2.remap and diagnostics.
    result_x = np.where(base_finite, map_x, -1.0).astype(np.float32, copy=False)
    result_y = np.where(np.isfinite(composed_y), composed_y, -1.0).astype(np.float32, copy=False)
    return (
        np.ascontiguousarray(result_x),
        np.ascontiguousarray(result_y),
        np.ascontiguousarray(valid),
        audit,
    )


def validate_inverse_map(
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    source_width: int | None = None,
    source_height: int | None = None,
) -> np.ndarray:
    """Validate finite/bounded samples and return a detached boolean mask."""

    x = np.asarray(map_x)
    y = np.asarray(map_y)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("Inverse maps must be matching two-dimensional arrays")
    if valid_mask is None:
        valid = np.ones(x.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).copy()
        if valid.shape != x.shape:
            raise ValueError("valid_mask must match inverse maps")
    if np.any(valid & (~np.isfinite(x) | ~np.isfinite(y))):
        raise ValueError("Valid inverse-map coordinates must be finite")
    if source_width is not None:
        if isinstance(source_width, bool) or int(source_width) < 2:
            raise ValueError("source_width must be at least two")
        if np.any(valid & ((x < 0.0) | (x > int(source_width) - 1))):
            raise ValueError("Valid inverse-map x coordinate is outside the source")
    if source_height is not None:
        if isinstance(source_height, bool) or int(source_height) < 2:
            raise ValueError("source_height must be at least two")
        if np.any(valid & ((y < 0.0) | (y > int(source_height) - 1))):
            raise ValueError("Valid inverse-map y coordinate is outside the source")
    return np.ascontiguousarray(valid)
