"""Shared calibrated inverse-remapping primitives.

The RGB-D pipeline stores aligned colour/depth frames in the camera's native
pixel coordinates.  Consumers that need an undistorted pinhole image or a
different calibrated output surface must use one inverse map, rather than
resampling a full-resolution image several times.  This module intentionally
does not know about a particular panorama backend; it only implements the
colour-camera calibration contract used by both metric projection paths.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class CalibratedIntrinsics(Protocol):
    """Minimal colour calibration needed for an OpenCV inverse map."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Sequence[float]


def camera_matrix(intrinsics: CalibratedIntrinsics) -> np.ndarray:
    """Return a validated 3×3 colour-camera matrix."""

    values = np.asarray(
        [
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
        ],
        dtype=np.float64,
    )
    if (
        int(intrinsics.width) <= 0
        or int(intrinsics.height) <= 0
        or not np.isfinite(values).all()
        or values[0] <= 0.0
        or values[1] <= 0.0
    ):
        raise ValueError("Colour intrinsics must have finite positive dimensions and focal lengths")
    return np.array(
        [[values[0], 0.0, values[2]], [0.0, values[1], values[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def distortion_coefficients(intrinsics: CalibratedIntrinsics) -> np.ndarray | None:
    """Return OpenCV distortion coefficients, or ``None`` for pinhole input."""

    values = tuple(float(value) for value in intrinsics.distortion)
    if len(values) not in {0, 4, 5, 8, 12, 14}:
        raise ValueError("OpenCV distortion must contain 0, 4, 5, 8, 12, or 14 values")
    if values and not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError("Colour distortion coefficients must be finite")
    if not values or not np.any(np.asarray(values, dtype=np.float64)):
        return None
    return np.asarray(values, dtype=np.float64)


def undistortion_maps(
    intrinsics: CalibratedIntrinsics,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build raw-source maps for an undistorted image in the same pinhole K."""

    matrix = camera_matrix(intrinsics)
    distortion = distortion_coefficients(intrinsics)
    if distortion is None:
        return None
    return cv2.initUndistortRectifyMap(
        matrix,
        distortion,
        None,
        matrix,
        (int(intrinsics.width), int(intrinsics.height)),
        cv2.CV_32FC1,
    )


def undistort_depth_with_validity(
    depth_mm: np.ndarray,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort aligned depth once with nearest-neighbour sampling.

    The returned mask describes geometric map validity only.  It deliberately
    does not turn an invalid depth value into an invalid *colour* pixel; callers
    that need colour and measured-depth validity must retain separate masks.
    """

    depth = np.asarray(depth_mm)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.number):
        raise ValueError("Aligned depth must be a numeric two-dimensional array")
    if maps is None:
        return np.asarray(depth, dtype=np.float32), np.ones(depth.shape, dtype=bool)
    map_x, map_y = maps
    if map_x.shape != depth.shape or map_y.shape != depth.shape:
        raise ValueError("Undistortion maps must match aligned depth dimensions")
    result = cv2.remap(
        np.asarray(depth, dtype=np.float32),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= depth.shape[1] - 1)
        & (map_y >= 0.0)
        & (map_y <= depth.shape[0] - 1)
    )
    return result, valid


def camera_points_to_source_pixels(
    camera_points_mm: np.ndarray,
    intrinsics: CalibratedIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map calibrated camera points to one raw colour-image inverse map.

    ``camera_points_mm`` are points in the source camera coordinate system.
    The output is raw-source ``map_x``, ``map_y`` plus a finite positive-z
    validity mask.  It is safe to pass the maps directly to ``cv2.remap``;
    callers retain the returned mask so black RGB content is never confused
    with an out-of-bounds sample.
    """

    points = np.asarray(camera_points_mm, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("Camera points must end in a three-vector")
    matrix = camera_matrix(intrinsics)
    flat = points.reshape(-1, 3)
    finite_positive = np.isfinite(flat).all(axis=1) & (flat[:, 2] > 1e-9)
    map_x = np.full(flat.shape[0], -1.0, dtype=np.float32)
    map_y = np.full(flat.shape[0], -1.0, dtype=np.float32)
    if np.any(finite_positive):
        usable = flat[finite_positive]
        distortion = distortion_coefficients(intrinsics)
        if distortion is None:
            source_x = matrix[0, 0] * usable[:, 0] / usable[:, 2] + matrix[0, 2]
            source_y = matrix[1, 1] * usable[:, 1] / usable[:, 2] + matrix[1, 2]
            projected = np.stack((source_x, source_y), axis=1)
        else:
            projected, _ = cv2.projectPoints(
                usable.reshape(-1, 1, 3),
                np.zeros((3, 1), dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                matrix,
                distortion,
            )
            projected = projected.reshape(-1, 2)
        map_x[finite_positive] = projected[:, 0].astype(np.float32)
        map_y[finite_positive] = projected[:, 1].astype(np.float32)
    shape = points.shape[:-1]
    return (
        map_x.reshape(shape),
        map_y.reshape(shape),
        finite_positive.reshape(shape),
    )
