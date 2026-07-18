"""Bounded APAP-plus-flow candidates for one local RGB handoff corridor.

This module intentionally does *not* alter a camera pose, compose a global
panorama transform, create colour, or persist dense evidence.  It evaluates a
single calibrated adjacent-pair corridor and may return an inverse sampling
map only for caller-supplied same-layer, visible, non-protected pixels.  The
caller remains responsible for binding that map to one RGB owner and for
keeping it out of every MultiBand region.

The implementation has two stages.  A locally weighted DLT grid supplies an
APAP-style coarse inverse field; a bidirectional Farneback residual is then
composed only where its forward/backward check, held-out photometric check,
Jacobian, scale, bounds, and displacement audits all pass.  Any failed check
returns a scalar-only hard-cut recommendation instead of a partially trusted
warp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np


_MINIMUM_CORRIDOR_WIDTH = 96
_MAXIMUM_CORRIDOR_WIDTH = 160


def _finite(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value: object, name: str) -> int:
    """Accept only finite integral scalar limits, never silent truncation."""

    number = _finite(value, name)
    integer = int(number)
    if not math.isclose(number, float(integer), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be an integer")
    return integer


@dataclass(frozen=True)
class LocalAPAPFlowConfig:
    """Closed safety envelope for one local APAP-plus-flow candidate."""

    enabled: bool = False
    minimum_correspondences: int = 40
    minimum_inlier_ratio: float = 0.60
    mesh_cell_pixels: int = 16
    minimum_active_mesh_cells: int = 4
    maximum_displacement_pixels: float = 8.0
    minimum_local_scale: float = 0.80
    maximum_local_scale: float = 1.25
    maximum_flow_fb_p95_pixels: float = 1.0
    maximum_flow_fb_pixels: float = 3.0
    held_out_fraction: float = 0.20
    minimum_held_out_pixels: int = 30
    minimum_held_out_improvement_ratio: float = 0.30

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "LocalAPAPFlowConfig":
        supplied = {} if value is None else dict(value)
        allowed = {
            "enabled",
            "minimum_correspondences",
            "minimum_inlier_ratio",
            "mesh_cell_pixels",
            "minimum_active_mesh_cells",
            "maximum_displacement_pixels",
            "minimum_local_scale",
            "maximum_local_scale",
            "maximum_flow_fb_p95_pixels",
            "maximum_flow_fb_pixels",
            "held_out_fraction",
            "minimum_held_out_pixels",
            "minimum_held_out_improvement_ratio",
        }
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(
                "Unknown local APAP/flow configuration keys: " + ", ".join(unknown)
            )
        try:
            result = cls(**supplied)
        except TypeError as exc:
            raise ValueError("Invalid local APAP/flow configuration") from exc
        result.validate()
        return result

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("local APAP/flow enabled must be a boolean")
        if _integer(self.minimum_correspondences, "minimum_correspondences") < 40:
            raise ValueError("local APAP/flow requires at least 40 correspondences")
        if _integer(self.mesh_cell_pixels, "mesh_cell_pixels") not in {16, 32}:
            raise ValueError("local APAP/flow mesh_cell_pixels must be 16 or 32")
        if _integer(self.minimum_active_mesh_cells, "minimum_active_mesh_cells") < 4:
            raise ValueError("local APAP/flow requires at least four active cells")
        if _integer(self.minimum_held_out_pixels, "minimum_held_out_pixels") < 30:
            raise ValueError("local APAP/flow requires at least 30 held-out pixels")
        values = {
            "minimum_inlier_ratio": self.minimum_inlier_ratio,
            "maximum_displacement_pixels": self.maximum_displacement_pixels,
            "minimum_local_scale": self.minimum_local_scale,
            "maximum_local_scale": self.maximum_local_scale,
            "maximum_flow_fb_p95_pixels": self.maximum_flow_fb_p95_pixels,
            "maximum_flow_fb_pixels": self.maximum_flow_fb_pixels,
            "held_out_fraction": self.held_out_fraction,
            "minimum_held_out_improvement_ratio": self.minimum_held_out_improvement_ratio,
        }
        numbers = {name: _finite(value, name) for name, value in values.items()}
        if not 0.60 <= numbers["minimum_inlier_ratio"] <= 1.0:
            raise ValueError("minimum_inlier_ratio must be in [0.60, 1]")
        if not 0.0 < numbers["maximum_displacement_pixels"] <= 8.0:
            raise ValueError("maximum_displacement_pixels must be in (0, 8]")
        if not math.isclose(numbers["minimum_local_scale"], 0.80, abs_tol=1e-12):
            raise ValueError("minimum_local_scale must equal 0.80")
        if not math.isclose(numbers["maximum_local_scale"], 1.25, abs_tol=1e-12):
            raise ValueError("maximum_local_scale must equal 1.25")
        if not 0.0 < numbers["maximum_flow_fb_p95_pixels"] <= 1.0:
            raise ValueError("maximum_flow_fb_p95_pixels must be in (0, 1]")
        if not 0.0 < numbers["maximum_flow_fb_pixels"] <= 3.0:
            raise ValueError("maximum_flow_fb_pixels must be in (0, 3]")
        if numbers["maximum_flow_fb_p95_pixels"] > numbers["maximum_flow_fb_pixels"]:
            raise ValueError("flow FB P95 limit cannot exceed its maximum limit")
        if not math.isclose(numbers["held_out_fraction"], 0.20, abs_tol=1e-12):
            raise ValueError("held_out_fraction must equal 0.20")
        if not 0.30 <= numbers["minimum_held_out_improvement_ratio"] <= 1.0:
            raise ValueError("minimum_held_out_improvement_ratio must be in [0.30, 1]")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "enabled": bool(self.enabled),
            "minimum_correspondences": int(self.minimum_correspondences),
            "minimum_inlier_ratio": float(self.minimum_inlier_ratio),
            "mesh_cell_pixels": int(self.mesh_cell_pixels),
            "minimum_active_mesh_cells": int(self.minimum_active_mesh_cells),
            "maximum_displacement_pixels": float(self.maximum_displacement_pixels),
            "minimum_local_scale": float(self.minimum_local_scale),
            "maximum_local_scale": float(self.maximum_local_scale),
            "maximum_flow_fb_p95_pixels": float(self.maximum_flow_fb_p95_pixels),
            "maximum_flow_fb_pixels": float(self.maximum_flow_fb_pixels),
            "held_out_fraction": float(self.held_out_fraction),
            "minimum_held_out_pixels": int(self.minimum_held_out_pixels),
            "minimum_held_out_improvement_ratio": float(
                self.minimum_held_out_improvement_ratio
            ),
        }


@dataclass(frozen=True)
class LocalAPAPFlowResult:
    """One local candidate; maps are present only after every audit succeeds."""

    accepted: bool
    method: str
    inverse_map_x: np.ndarray | None
    inverse_map_y: np.ndarray | None
    active_mask: np.ndarray
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        active = np.asarray(self.active_mask, dtype=bool)
        if active.ndim != 2:
            raise ValueError("local APAP/flow active mask must be two-dimensional")
        if self.accepted:
            if self.method not in {"apap", "apap_plus_dense_flow"}:
                raise ValueError("accepted local candidate has the wrong method")
            if self.inverse_map_x is None or self.inverse_map_y is None:
                raise ValueError("accepted local candidate requires inverse maps")
            map_x = np.asarray(self.inverse_map_x, dtype=np.float32)
            map_y = np.asarray(self.inverse_map_y, dtype=np.float32)
            if map_x.shape != active.shape or map_y.shape != active.shape:
                raise ValueError("local APAP/flow map and active-mask shapes differ")
            if not np.isfinite(map_x[active]).all() or not np.isfinite(map_y[active]).all():
                raise ValueError("accepted local APAP/flow map is non-finite")
            object.__setattr__(self, "inverse_map_x", np.ascontiguousarray(map_x))
            object.__setattr__(self, "inverse_map_y", np.ascontiguousarray(map_y))
        elif self.method != "hard_cut_degraded":
            raise ValueError("rejected local candidate must request hard_cut_degraded")
        object.__setattr__(self, "active_mask", np.ascontiguousarray(active))
        object.__setattr__(self, "audit", dict(self.audit))

    def as_dict(self) -> dict[str, object]:
        """Return bounded audit data, never maps or masks."""

        return {
            "accepted": bool(self.accepted),
            "method": self.method,
            "active_pixel_count": int(np.count_nonzero(self.active_mask)),
            "dense_evidence_storage": "temporary_only",
            **dict(self.audit),
        }


@dataclass(frozen=True)
class LocalAPAPFlowInverseWarp:
    """One accepted local map exposed through the renderer warp protocol.

    The maps use corridor-local *virtual* coordinates: an output coordinate
    in the first strip maps to the sampling coordinate in the second strip.
    This is deliberately not a pose, a global homography, or a raw-image map.
    The small adapter only evaluates pixels explicitly accepted by
    :class:`LocalAPAPFlowResult`; every other query is exactly identity.
    """

    corridor_x0: int
    inverse_map_x: np.ndarray
    inverse_map_y: np.ndarray
    active_mask: np.ndarray

    def __post_init__(self) -> None:
        x0 = int(self.corridor_x0)
        if x0 != self.corridor_x0:
            raise ValueError("local APAP/flow corridor x origin must be an integer")
        map_x = np.asarray(self.inverse_map_x, dtype=np.float32)
        map_y = np.asarray(self.inverse_map_y, dtype=np.float32)
        active = np.asarray(self.active_mask, dtype=bool)
        if (
            map_x.ndim != 2
            or map_y.shape != map_x.shape
            or active.shape != map_x.shape
            or not map_x.size
            or not np.isfinite(map_x[active]).all()
            or not np.isfinite(map_y[active]).all()
        ):
            raise ValueError("local APAP/flow inverse warp is malformed")
        object.__setattr__(self, "corridor_x0", x0)
        object.__setattr__(self, "inverse_map_x", np.ascontiguousarray(map_x))
        object.__setattr__(self, "inverse_map_y", np.ascontiguousarray(map_y))
        object.__setattr__(self, "active_mask", np.ascontiguousarray(active))

    @property
    def is_identity(self) -> bool:
        return False

    def inverse_virtual_coordinates(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map only verified integer output samples; retain identity otherwise.

        Formal full-resolution output coordinates are integer pixel centres.
        Nearest-cell lookup therefore preserves the exact per-pixel safety
        audit instead of interpolating through an unknown/protected neighbour.
        Preview-only fractional coordinates also use this conservative
        identity-or-verified-sample rule.
        """

        x_values, y_values = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        mapped_x = np.asarray(x_values, dtype=np.float64).copy()
        mapped_y = np.asarray(y_values, dtype=np.float64).copy()
        flat_x = x_values.reshape(-1)
        flat_y = y_values.reshape(-1)
        local_x = np.rint(flat_x - float(self.corridor_x0)).astype(np.int64)
        local_y = np.rint(flat_y).astype(np.int64)
        height, width = self.active_mask.shape
        inside = (
            np.isfinite(flat_x)
            & np.isfinite(flat_y)
            & (local_x >= 0)
            & (local_x < width)
            & (local_y >= 0)
            & (local_y < height)
        )
        if np.any(inside):
            positions = np.flatnonzero(inside)
            ix = local_x[positions]
            iy = local_y[positions]
            output_active = self.active_mask[iy, ix]
            if np.any(output_active):
                active_positions = positions[output_active]
                active_x = ix[output_active]
                active_y = iy[output_active]
                source_x = self.inverse_map_x[active_y, active_x]
                source_y = self.inverse_map_y[active_y, active_x]
                source_ix = np.rint(source_x).astype(np.int64)
                source_iy = np.rint(source_y).astype(np.int64)
                source_inside = (
                    np.isfinite(source_x)
                    & np.isfinite(source_y)
                    & (source_ix >= 0)
                    & (source_ix < width)
                    & (source_iy >= 0)
                    & (source_iy < height)
                )
                safe = np.zeros(active_positions.shape, dtype=bool)
                if np.any(source_inside):
                    safe[source_inside] = self.active_mask[
                        source_iy[source_inside], source_ix[source_inside]
                    ]
                if np.any(safe):
                    accepted_positions = active_positions[safe]
                    mapped_x.reshape(-1)[accepted_positions] = (
                        float(self.corridor_x0) + source_x[safe]
                    )
                    mapped_y.reshape(-1)[accepted_positions] = source_y[safe]
        return mapped_x, mapped_y


def _bgr_gray(value: object, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be a BGR uint8 image")
    if image.shape[0] < 8 or image.shape[1] < 8:
        raise ValueError(f"{name} is too small for a local APAP/flow audit")
    return cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_BGR2GRAY)


def _mask(value: object, shape: tuple[int, int], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=bool)
    if result.shape != shape:
        raise ValueError(f"{name} must match the local corridor")
    return np.ascontiguousarray(result)


def _correspondence_arrays(
    correspondences: tuple[object, object] | None,
    source_gray: np.ndarray,
    target_gray: np.ndarray,
    support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if correspondences is not None:
        if not isinstance(correspondences, tuple) or len(correspondences) != 2:
            raise ValueError("correspondences must be a (source_points, target_points) tuple")
        source_points = np.asarray(correspondences[0], dtype=np.float64)
        target_points = np.asarray(correspondences[1], dtype=np.float64)
    else:
        mask_u8 = np.where(support, 255, 0).astype(np.uint8)
        detector = cv2.ORB_create(nfeatures=1200, fastThreshold=8)
        key0, descriptor0 = detector.detectAndCompute(source_gray, mask_u8)
        key1, descriptor1 = detector.detectAndCompute(target_gray, mask_u8)
        if descriptor0 is None or descriptor1 is None or not key0 or not key1:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
            descriptor0, descriptor1
        )
        matches = sorted(matches, key=lambda match: (match.distance, match.queryIdx))
        source_points = np.asarray(
            [key0[match.queryIdx].pt for match in matches], dtype=np.float64
        )
        target_points = np.asarray(
            [key1[match.trainIdx].pt for match in matches], dtype=np.float64
        )
    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("correspondence points must be N x 2 arrays")
    if source_points.shape != target_points.shape or source_points.shape[1:] != (2,):
        raise ValueError("source and target correspondence arrays must both be N x 2")
    finite = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(axis=1)
    h, w = support.shape
    in_bounds = (
        (source_points[:, 0] >= 0.0)
        & (source_points[:, 0] < w)
        & (source_points[:, 1] >= 0.0)
        & (source_points[:, 1] < h)
        & (target_points[:, 0] >= 0.0)
        & (target_points[:, 0] < w)
        & (target_points[:, 1] >= 0.0)
        & (target_points[:, 1] < h)
    )
    if np.any(finite & in_bounds):
        source_x = np.rint(source_points[:, 0]).astype(np.int32, copy=False)
        source_y = np.rint(source_points[:, 1]).astype(np.int32, copy=False)
        rounded_x = np.rint(target_points[:, 0]).astype(np.int32, copy=False)
        rounded_y = np.rint(target_points[:, 1]).astype(np.int32, copy=False)
        source_x = np.clip(source_x, 0, w - 1)
        source_y = np.clip(source_y, 0, h - 1)
        rounded_x = np.clip(rounded_x, 0, w - 1)
        rounded_y = np.clip(rounded_y, 0, h - 1)
        in_support = support[rounded_y, rounded_x] & support[source_y, source_x]
    else:
        in_support = np.zeros(len(source_points), dtype=bool)
    keep = finite & in_bounds & in_support
    return (
        np.ascontiguousarray(source_points[keep]),
        np.ascontiguousarray(target_points[keep]),
    )


def _weighted_homography(
    target_points: np.ndarray, source_points: np.ndarray, weights: np.ndarray
) -> np.ndarray | None:
    """Solve local target-to-source DLT without introducing a global transform."""

    keep = np.asarray(weights, dtype=np.float64) > 1e-4
    if int(np.count_nonzero(keep)) < 4:
        return None
    target = np.asarray(target_points[keep], dtype=np.float64)
    source = np.asarray(source_points[keep], dtype=np.float64)
    root_weight = np.sqrt(np.asarray(weights[keep], dtype=np.float64))
    x, y = target[:, 0], target[:, 1]
    u, v = source[:, 0], source[:, 1]
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    rows0 = np.column_stack((-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u))
    rows1 = np.column_stack((zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v))
    matrix = np.empty((target.shape[0] * 2, 9), dtype=np.float64)
    matrix[0::2] = rows0 * root_weight[:, None]
    matrix[1::2] = rows1 * root_weight[:, None]
    try:
        _, singular, vectors = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    # Exact correspondences intentionally leave the homogeneous null-space
    # singular value near zero.  Degeneracy is instead exposed by the next
    # singular value (for example, collinear support).
    if not np.isfinite(singular).all() or singular.size < 2 or singular[-2] <= 1e-10:
        return None
    homography = vectors[-1].reshape(3, 3)
    if not np.isfinite(homography).all() or abs(float(homography[2, 2])) <= 1e-12:
        return None
    return homography / homography[2, 2]


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray | None:
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    mapped = homogeneous @ np.asarray(homography, dtype=np.float64).T
    denominator = mapped[:, 2]
    if np.any(~np.isfinite(mapped)) or np.any(np.abs(denominator) <= 1e-9):
        return None
    return mapped[:, :2] / denominator[:, None]


def _grid_coordinates(length: int, spacing: int) -> np.ndarray:
    values = list(range(0, int(length), int(spacing)))
    if values[-1] != int(length) - 1:
        values.append(int(length) - 1)
    return np.asarray(values, dtype=np.float64)


def _interpolate_grid(
    grid_x: np.ndarray, grid_y: np.ndarray, values: np.ndarray, height: int, width: int
) -> np.ndarray:
    yy, xx = np.indices((height, width), dtype=np.float64)
    ix = np.clip(np.searchsorted(grid_x, xx, side="right") - 1, 0, len(grid_x) - 2)
    iy = np.clip(np.searchsorted(grid_y, yy, side="right") - 1, 0, len(grid_y) - 2)
    dx = np.maximum(grid_x[ix + 1] - grid_x[ix], 1e-12)
    dy = np.maximum(grid_y[iy + 1] - grid_y[iy], 1e-12)
    tx = (xx - grid_x[ix]) / dx
    ty = (yy - grid_y[iy]) / dy
    v00 = values[iy, ix]
    v01 = values[iy, ix + 1]
    v10 = values[iy + 1, ix]
    v11 = values[iy + 1, ix + 1]
    return (1.0 - ty) * ((1.0 - tx) * v00 + tx * v01) + ty * (
        (1.0 - tx) * v10 + tx * v11
    )


def _coarse_apap_inverse_map(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    height: int,
    width: int,
    settings: LocalAPAPFlowConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    grid_x = _grid_coordinates(width, int(settings.mesh_cell_pixels))
    grid_y = _grid_coordinates(height, int(settings.mesh_cell_pixels))
    node_x = np.empty((len(grid_y), len(grid_x)), dtype=np.float64)
    node_y = np.empty_like(node_x)
    sigma_squared = float(max(width, height) * 0.40) ** 2
    for row, y in enumerate(grid_y):
        for column, x in enumerate(grid_x):
            if row in {0, len(grid_y) - 1} or column in {0, len(grid_x) - 1}:
                node_x[row, column] = x
                node_y[row, column] = y
                continue
            distance_squared = np.sum((target_points - (x, y)) ** 2, axis=1)
            weights = np.exp(-distance_squared / max(sigma_squared, 1e-12))
            homography = _weighted_homography(target_points, source_points, weights)
            if homography is None:
                return None
            mapped = _project(homography, np.asarray([[x, y]], dtype=np.float64))
            if mapped is None:
                return None
            node_x[row, column], node_y[row, column] = mapped[0]
    map_x = _interpolate_grid(grid_x, grid_y, node_x, height, width)
    map_y = _interpolate_grid(grid_x, grid_y, node_y, height, width)
    yy, xx = np.indices((height, width), dtype=np.float64)
    # The entire outer border, not merely grid nodes, is a fixed identity
    # boundary.  This prevents an accepted candidate from dragging the corridor
    # edge or a neighbouring safe-wall source coordinate.
    for array, identity in ((map_x, xx), (map_y, yy)):
        array[0, :] = identity[0, :]
        array[-1, :] = identity[-1, :]
        array[:, 0] = identity[:, 0]
        array[:, -1] = identity[:, -1]
    return np.asarray(map_x, dtype=np.float32), np.asarray(map_y, dtype=np.float32)


def _sample_vector_field(field: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    values = [
        cv2.remap(
            np.asarray(field[:, :, channel], dtype=np.float32),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )
        for channel in range(field.shape[2])
    ]
    return np.dstack(values)


def _sample_boolean_mask(
    mask: np.ndarray, map_x: np.ndarray, map_y: np.ndarray
) -> np.ndarray:
    """Nearest-sample an eligibility mask without treating out-of-bounds as safe."""

    sampled = cv2.remap(
        np.asarray(mask, dtype=np.uint8),
        np.asarray(map_x, dtype=np.float32),
        np.asarray(map_y, dtype=np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.asarray(sampled, dtype=bool)


def _map_jacobian(map_x: np.ndarray, map_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx_dy, dx_dx = np.gradient(np.asarray(map_x, dtype=np.float64))
    dy_dy, dy_dx = np.gradient(np.asarray(map_y, dtype=np.float64))
    determinant = dx_dx * dy_dy - dx_dy * dy_dx
    squared_trace = dx_dx**2 + dx_dy**2 + dy_dx**2 + dy_dy**2
    discriminant = np.maximum(squared_trace**2 - 4.0 * determinant**2, 0.0)
    sigma_max = np.sqrt(np.maximum((squared_trace + np.sqrt(discriminant)) / 2.0, 0.0))
    sigma_min = np.sqrt(np.maximum((squared_trace - np.sqrt(discriminant)) / 2.0, 0.0))
    return determinant, sigma_min, sigma_max


def _candidate_cells(mask: np.ndarray, spacing: int) -> int:
    height, width = mask.shape
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return 0
    columns = np.clip(xx // int(spacing), 0, max(0, (width - 1) // int(spacing)))
    rows = np.clip(yy // int(spacing), 0, max(0, (height - 1) // int(spacing)))
    return int(np.unique(rows.astype(np.int64) * max(1, width) + columns).size)


def _reject(shape: tuple[int, int], reason: str, **audit: object) -> LocalAPAPFlowResult:
    return LocalAPAPFlowResult(
        accepted=False,
        method="hard_cut_degraded",
        inverse_map_x=None,
        inverse_map_y=None,
        active_mask=np.zeros(shape, dtype=bool),
        audit={"accepted": False, "reason": reason, **audit},
    )


def _accepted_apap(
    map_x: np.ndarray,
    map_y: np.ndarray,
    active: np.ndarray,
    **audit: object,
) -> LocalAPAPFlowResult:
    """Return a fully audited APAP-only result after flow was not beneficial."""

    return LocalAPAPFlowResult(
        accepted=True,
        method="apap",
        inverse_map_x=map_x,
        inverse_map_y=map_y,
        active_mask=active,
        audit={"accepted": True, "method": "apap", **audit},
    )


def fit_local_apap_plus_dense_flow(
    source_bgr: object,
    target_bgr: object,
    *,
    same_layer_mask: object,
    application_mask: object | None = None,
    protected_mask: object | None = None,
    correspondences: tuple[object, object] | None = None,
    config: LocalAPAPFlowConfig | Mapping[str, object] | None = None,
) -> LocalAPAPFlowResult:
    """Fit a local, audited inverse map or return a hard-cut recommendation.

    ``same_layer_mask`` must already encode mutual depth visibility and source
    identity.  ``application_mask`` may only narrow it; ``protected_mask`` is
    always excluded.  This function never infers permissions from RGB alone.
    Correspondences are source-to-target ``(x, y)`` points; when omitted, ORB
    is used only to create a local candidate that still faces every gate.
    """

    settings = (
        LocalAPAPFlowConfig.from_mapping(config)
        if isinstance(config, Mapping)
        else (LocalAPAPFlowConfig() if config is None else config)
    )
    if not isinstance(settings, LocalAPAPFlowConfig):
        raise TypeError("config must be LocalAPAPFlowConfig or a mapping")
    settings.validate()
    source_gray = _bgr_gray(source_bgr, "source_bgr")
    target_gray = _bgr_gray(target_bgr, "target_bgr")
    if source_gray.shape != target_gray.shape:
        raise ValueError("source_bgr and target_bgr must have the same shape")
    height, width = source_gray.shape
    if not _MINIMUM_CORRIDOR_WIDTH <= width <= _MAXIMUM_CORRIDOR_WIDTH:
        raise ValueError("local APAP/flow corridor width must be in [96, 160]")
    same_layer = _mask(same_layer_mask, (height, width), "same_layer_mask")
    requested = (
        same_layer
        if application_mask is None
        else _mask(application_mask, (height, width), "application_mask")
    )
    protected = (
        np.zeros((height, width), dtype=bool)
        if protected_mask is None
        else _mask(protected_mask, (height, width), "protected_mask")
    )
    if np.any(requested & ~same_layer):
        raise ValueError("application_mask must be a same-layer subset")
    if np.any(requested & protected):
        raise ValueError("application_mask cannot enter protected pixels")
    # A fixed one-pixel identity border is part of the non-folding contract.
    requested = np.ascontiguousarray(requested.copy())
    requested[[0, -1], :] = False
    requested[:, [0, -1]] = False
    # The APAP grid fades from an identity border.  Do not apply a field in
    # that transition cell: retaining the calibrated owner there is safer
    # than allowing a legitimate translation to look like a local scale.
    active_support = np.ascontiguousarray(requested.copy())
    margin = int(settings.mesh_cell_pixels)
    if height > margin * 2 and width > margin * 2:
        active_support[:margin, :] = False
        active_support[-margin:, :] = False
        active_support[:, :margin] = False
        active_support[:, -margin:] = False
    if not settings.enabled:
        return _reject((height, width), "local_apap_flow_disabled")
    if not np.any(requested):
        return _reject((height, width), "no_same_layer_application_support")
    if not np.any(active_support):
        return _reject((height, width), "no_interior_same_layer_application_support")
    source_points, target_points = _correspondence_arrays(
        correspondences, source_gray, target_gray, requested
    )
    match_count = int(len(source_points))
    if match_count < int(settings.minimum_correspondences):
        return _reject(
            (height, width),
            "insufficient_correspondences",
            correspondence_count=match_count,
        )
    homography, inlier_mask = cv2.findHomography(
        np.asarray(source_points, dtype=np.float32),
        np.asarray(target_points, dtype=np.float32),
        cv2.RANSAC,
        2.0,
    )
    if homography is None or inlier_mask is None:
        return _reject((height, width), "apap_ransac_failed", correspondence_count=match_count)
    inliers = np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
    inlier_count = int(np.count_nonzero(inliers))
    inlier_ratio = float(inlier_count / max(1, match_count))
    if inlier_count < int(settings.minimum_correspondences) or inlier_ratio < float(
        settings.minimum_inlier_ratio
    ):
        return _reject(
            (height, width),
            "insufficient_apap_inliers",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
            apap_inlier_ratio=inlier_ratio,
        )
    source_points = source_points[inliers]
    target_points = target_points[inliers]
    correspondence_mask = np.zeros((height, width), dtype=bool)
    rounded_x = np.clip(np.rint(target_points[:, 0]).astype(np.int32), 0, width - 1)
    rounded_y = np.clip(np.rint(target_points[:, 1]).astype(np.int32), 0, height - 1)
    correspondence_mask[rounded_y, rounded_x] = True
    active_cells = _candidate_cells(correspondence_mask, int(settings.mesh_cell_pixels))
    if active_cells < int(settings.minimum_active_mesh_cells):
        return _reject(
            (height, width),
            "insufficient_apap_cell_coverage",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
            active_mesh_cell_count=active_cells,
        )
    coarse = _coarse_apap_inverse_map(
        source_points,
        target_points,
        height=height,
        width=width,
        settings=settings,
    )
    if coarse is None:
        return _reject(
            (height, width),
            "local_apap_dlt_failed",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
    )
    base_x, base_y = coarse
    yy, xx = np.indices((height, width), dtype=np.float32)
    base_bounds = (
        (base_x >= 0.0)
        & (base_x <= width - 1)
        & (base_y >= 0.0)
        & (base_y <= height - 1)
    )
    if not np.all(base_bounds[active_support]):
        return _reject(
            (height, width),
            "apap_inverse_sampling_out_of_bounds",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
        )
    # The source-side sampling point has exactly the same eligibility
    # contract as the output point.  A map that starts on the verified layer
    # but samples through an occlusion edge, depth hole, or protected feature
    # is made identity at that output point; the caller's hard owner remains
    # responsible there.  This permits a protected hole inside an otherwise
    # safe same-layer instance without permitting a single unsafe sample.
    source_safe = _sample_boolean_mask(same_layer & ~protected, base_x, base_y)
    active_support = np.ascontiguousarray(active_support & source_safe)
    if not np.any(active_support):
        return _reject(
            (height, width),
            "apap_inverse_sampling_has_no_same_layer_source_support",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
        )
    base_displacement = np.hypot(base_x - xx, base_y - yy)
    base_maximum_displacement = float(np.max(base_displacement[active_support]))
    if base_maximum_displacement > float(settings.maximum_displacement_pixels):
        return _reject(
            (height, width),
            "apap_maximum_displacement_exceeded",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
            max_displacement_px=base_maximum_displacement,
        )
    base_jacobian, base_scale_min, base_scale_max = _map_jacobian(base_x, base_y)
    base_jacobian_values = np.asarray(base_jacobian[active_support], dtype=np.float64)
    base_smallest_scale = float(np.min(base_scale_min[active_support]))
    base_largest_scale = float(np.max(base_scale_max[active_support]))
    if (
        not np.isfinite(base_jacobian_values).all()
        or np.any(base_jacobian_values <= 0.0)
        or base_smallest_scale < float(settings.minimum_local_scale)
        or base_largest_scale > float(settings.maximum_local_scale)
    ):
        return _reject(
            (height, width),
            "invalid_apap_jacobian_or_local_scale",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
            jacobian_min=float(np.nanmin(base_jacobian_values)),
            local_scale_min=base_smallest_scale,
            local_scale_max=base_largest_scale,
        )
    base_held_out = active_support & (
        (
            xx.astype(np.int64) * 73856093
            + yy.astype(np.int64) * 19349663
        )
        % 5
        == 0
    )
    base_held_out_count = int(np.count_nonzero(base_held_out))
    if base_held_out_count < int(settings.minimum_held_out_pixels):
        return _reject(
            (height, width),
            "insufficient_held_out_apap_support",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
            held_out_pixel_count=base_held_out_count,
        )
    base_sample = cv2.remap(
        source_gray,
        base_x,
        base_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    base_before = np.abs(
        target_gray.astype(np.float32) - source_gray.astype(np.float32)
    )
    base_after = np.abs(target_gray.astype(np.float32) - base_sample.astype(np.float32))
    base_before_p95 = float(np.percentile(base_before[base_held_out], 95.0))
    base_after_p95 = float(np.percentile(base_after[base_held_out], 95.0))
    base_improvement = float(
        (base_before_p95 - base_after_p95) / max(base_before_p95, 1e-6)
    )
    if base_improvement < float(settings.minimum_held_out_improvement_ratio):
        return _reject(
            (height, width),
            "apap_held_out_photometric_improvement_insufficient",
            correspondence_count=match_count,
            apap_inliers=inlier_count,
            held_out_pixel_count=base_held_out_count,
            held_out_error_before_p95=base_before_p95,
            held_out_error_after_p95=base_after_p95,
            held_out_improvement_ratio=base_improvement,
        )
    base_audit = {
        "correspondence_count": match_count,
        "apap_inliers": inlier_count,
        "apap_inlier_ratio": inlier_ratio,
        "active_mesh_cell_count": active_cells,
        "source_safe_application_pixel_count": int(np.count_nonzero(active_support)),
        "max_displacement_px": base_maximum_displacement,
        "jacobian_min": float(np.min(base_jacobian_values)),
        "local_scale_min": base_smallest_scale,
        "local_scale_max": base_largest_scale,
        "held_out_pixel_count": base_held_out_count,
        "held_out_error_before_p95": base_before_p95,
        "held_out_error_after_p95": base_after_p95,
        "held_out_improvement_ratio": base_improvement,
        "application_policy": "same_layer_visible_nonprotected_instance_or_background_only",
        "boundary_policy": "outer_corridor_border_identity",
    }
    warped_source = cv2.remap(
        source_gray,
        base_x,
        base_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    backward = cv2.calcOpticalFlowFarneback(
        target_gray,
        warped_source,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    forward = cv2.calcOpticalFlowFarneback(
        warped_source,
        target_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    residual_x = xx + np.asarray(backward[:, :, 0], dtype=np.float32)
    residual_y = yy + np.asarray(backward[:, :, 1], dtype=np.float32)
    inside_residual = (
        (residual_x >= 0.0)
        & (residual_x <= width - 1)
        & (residual_y >= 0.0)
        & (residual_y <= height - 1)
    )
    sampled_forward = _sample_vector_field(forward, residual_x, residual_y)
    fb_error = np.linalg.norm(np.asarray(backward, dtype=np.float32) + sampled_forward, axis=2)
    composed_x = cv2.remap(
        base_x,
        residual_x,
        residual_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    composed_y = cv2.remap(
        base_y,
        residual_x,
        residual_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    flow_valid = (
        active_support
        & inside_residual
        & np.isfinite(fb_error)
        & (fb_error <= float(settings.maximum_flow_fb_pixels))
        & np.isfinite(composed_x)
        & np.isfinite(composed_y)
        & _sample_boolean_mask(same_layer & ~protected, composed_x, composed_y)
    )
    if not np.any(flow_valid):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="no_forward_backward_consistent_flow",
        )
    flow_values = np.asarray(fb_error[flow_valid], dtype=np.float64)
    flow_p95 = float(np.percentile(flow_values, 95.0))
    flow_max = float(np.max(flow_values))
    if flow_p95 > float(settings.maximum_flow_fb_p95_pixels):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="flow_forward_backward_p95_exceeded",
            flow_fb_p95_px=flow_p95,
            flow_fb_max_px=flow_max,
        )
    # Do not splice a dense residual into a subset of an otherwise accepted
    # APAP field.  Such a seam inside the active region is not audited by a
    # per-pixel FB metric and could introduce a fold or a visible step.  The
    # fully valid APAP field remains an approved B-grade candidate instead.
    if not np.array_equal(flow_valid, active_support):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="flow_does_not_cover_entire_apap_application_region",
            dense_flow_candidate_pixel_count=int(np.count_nonzero(flow_valid)),
        )
    map_x = np.asarray(base_x.copy(), dtype=np.float32)
    map_y = np.asarray(base_y.copy(), dtype=np.float32)
    map_x[flow_valid] = composed_x[flow_valid]
    map_y[flow_valid] = composed_y[flow_valid]
    map_x[[0, -1], :] = xx[[0, -1], :]
    map_x[:, [0, -1]] = xx[:, [0, -1]]
    map_y[[0, -1], :] = yy[[0, -1], :]
    map_y[:, [0, -1]] = yy[:, [0, -1]]
    bounds = (
        (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    if not np.all(bounds[flow_valid]):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="inverse_sampling_out_of_bounds",
        )
    displacement = np.hypot(map_x - xx, map_y - yy)
    maximum_displacement = float(np.max(displacement[active_support]))
    if maximum_displacement > float(settings.maximum_displacement_pixels):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="maximum_displacement_exceeded",
            dense_flow_max_displacement_px=maximum_displacement,
        )
    jacobian, scale_min, scale_max = _map_jacobian(map_x, map_y)
    jacobian_values = np.asarray(jacobian[active_support], dtype=np.float64)
    smallest_scale = float(np.min(scale_min[active_support]))
    largest_scale = float(np.max(scale_max[active_support]))
    if (
        not np.isfinite(jacobian_values).all()
        or np.any(jacobian_values <= 0.0)
        or smallest_scale < float(settings.minimum_local_scale)
        or largest_scale > float(settings.maximum_local_scale)
    ):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="invalid_jacobian_or_local_scale",
            dense_flow_jacobian_min=float(np.nanmin(jacobian_values)),
            dense_flow_local_scale_min=smallest_scale,
            dense_flow_local_scale_max=largest_scale,
        )
    held_out = active_support & (
        ((xx.astype(np.int64) * 73856093 + yy.astype(np.int64) * 19349663) % 5)
        == 0
    )
    held_out_count = int(np.count_nonzero(held_out))
    if held_out_count < int(settings.minimum_held_out_pixels):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="insufficient_held_out_flow_support",
            dense_flow_held_out_pixel_count=held_out_count,
        )
    final_sample = cv2.remap(
        source_gray,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # The held-out comparison is against the uncorrected handoff, not only
    # against the already-improved APAP field.  A valid APAP coarse field may
    # leave essentially zero dense residual; that is a success, not a reason
    # to force a visible hard cut.
    before = np.abs(target_gray.astype(np.float32) - source_gray.astype(np.float32))
    after = np.abs(target_gray.astype(np.float32) - final_sample.astype(np.float32))
    before_p95 = float(np.percentile(before[held_out], 95.0))
    after_p95 = float(np.percentile(after[held_out], 95.0))
    improvement = float((before_p95 - after_p95) / max(before_p95, 1e-6))
    if improvement < float(settings.minimum_held_out_improvement_ratio):
        return _accepted_apap(
            base_x,
            base_y,
            active_support,
            **base_audit,
            dense_flow_status="held_out_photometric_improvement_insufficient",
            dense_flow_held_out_pixel_count=held_out_count,
            dense_flow_held_out_error_after_p95=after_p95,
            dense_flow_held_out_improvement_ratio=improvement,
        )
    return LocalAPAPFlowResult(
        accepted=True,
        method="apap_plus_dense_flow",
        inverse_map_x=map_x,
        inverse_map_y=map_y,
        active_mask=active_support,
        audit={
            "accepted": True,
            "method": "apap_plus_dense_flow",
            "correspondence_count": match_count,
            "apap_inliers": inlier_count,
            "apap_inlier_ratio": inlier_ratio,
            "active_mesh_cell_count": active_cells,
            "flow_fb_p95_px": flow_p95,
            "flow_fb_max_px": flow_max,
            "dense_flow_active_pixel_count": int(np.count_nonzero(flow_valid)),
            "max_displacement_px": maximum_displacement,
            "jacobian_min": float(np.min(jacobian_values)),
            "local_scale_min": smallest_scale,
            "local_scale_max": largest_scale,
            "held_out_pixel_count": held_out_count,
            "held_out_error_before_p95": before_p95,
            "held_out_error_after_p95": after_p95,
            "held_out_improvement_ratio": improvement,
            "application_policy": "same_layer_visible_nonprotected_instance_or_background_only",
            "boundary_policy": "outer_corridor_border_identity",
        },
    )


__all__ = [
    "LocalAPAPFlowConfig",
    "LocalAPAPFlowInverseWarp",
    "LocalAPAPFlowResult",
    "fit_local_apap_plus_dense_flow",
]
