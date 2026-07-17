"""Pure geometry primitives for conservative local RGB-D seam assistance.

This module intentionally does *not* render an image, read a session, change a
camera pose, or build a global projection.  It is a narrow, adjacent-pair
building block for a future renderer:

* aligned depth is unprojected with calibrated intrinsics;
* points move between real ``camera_to_world`` poses in millimetres;
* a per-direction point z-buffer establishes visible, mutually consistent
  surfaces and conservative occlusion labels;
* depth discontinuities become a hard guard; and
* callers can fit either a bounded affine diagnostic warp or a 16/32 px
  identity-pinned local inverse mesh per already segmented surface layer.

The only image-like inputs are depth arrays.  In particular, the results never
contain RGB pixels or synthesize colour: a renderer must still sample every
published pixel from an original RGB source.  The helpers are deliberately
small and side-effect free so they can be audited before they are wired into a
formal path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Mapping, Protocol, Sequence, runtime_checkable

import cv2
import numpy as np


_SE3_ATOL = 1e-5


@runtime_checkable
class IntrinsicsLike(Protocol):
    """Minimal calibrated colour/depth-aligned pinhole interface."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Sequence[float]


@dataclass(frozen=True)
class GeometryIntrinsics:
    """Pinhole intrinsics in the native aligned-colour pixel domain."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...] = ()

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray(
            ((self.fx, 0.0, self.cx), (0.0, self.fy, self.cy), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )


@dataclass(frozen=True)
class SampledDepth:
    """Nearest-neighbour depth sampled in an arbitrary strip/tile domain.

    Invalid results are explicitly represented by ``valid_mask=False`` and a
    zero depth value.  The zero is never a manufactured measurement; callers
    must honour the mask before unprojection.
    """

    depth_mm: np.ndarray
    valid_mask: np.ndarray


def _value(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _coerce_distortion(value: object) -> tuple[float, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        names = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        return tuple(float(value.get(name, 0.0)) for name in names)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return tuple(float(item) for item in array)


def coerce_intrinsics(value: IntrinsicsLike | Mapping[str, object]) -> GeometryIntrinsics:
    """Validate a calibrated aligned-colour pinhole description."""

    try:
        intrinsics = GeometryIntrinsics(
            width=int(_value(value, "width")),
            height=int(_value(value, "height")),
            fx=float(_value(value, "fx")),
            fy=float(_value(value, "fy")),
            cx=float(_value(value, "cx")),
            cy=float(_value(value, "cy")),
            distortion=_coerce_distortion(_value(value, "distortion", ())),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid geometry-assist intrinsics") from exc
    numeric = np.asarray(
        [
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            *intrinsics.distortion,
        ],
        dtype=np.float64,
    )
    if intrinsics.width <= 0 or intrinsics.height <= 0:
        raise ValueError("Geometry-assist intrinsic dimensions must be positive")
    if not np.isfinite(numeric).all() or intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("Geometry-assist intrinsics must be finite with positive focal lengths")
    if len(intrinsics.distortion) not in {0, 4, 5, 8, 12, 14}:
        raise ValueError("Geometry-assist distortion must contain 0, 4, 5, 8, 12, or 14 values")
    return intrinsics


def validate_camera_to_world(camera_to_world: np.ndarray) -> np.ndarray:
    """Return a finite rigid camera-to-world transform in millimetres."""

    pose = np.asarray(camera_to_world, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("camera_to_world must be a finite 4x4 SE(3) matrix")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=_SE3_ATOL):
        raise ValueError("camera_to_world has an invalid homogeneous final row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_SE3_ATOL):
        raise ValueError("camera_to_world rotation is not orthonormal")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=_SE3_ATOL):
        raise ValueError("camera_to_world rotation determinant is not +1")
    return pose.copy()


@dataclass(frozen=True)
class GeometryAssistConfig:
    """Non-bypassable local-pair geometry limits.

    ``depth_noise_mm`` is a conservative calibrated one-sigma noise bound.  A
    caller with a range-dependent noise model may pass a safely upper-bounded
    value for the analysed tile; the tolerance remains the maximum of absolute,
    relative, and sigma-derived terms.
    """

    absolute_depth_tolerance_mm: float = 20.0
    relative_depth_tolerance: float = 0.02
    depth_noise_mm: float = 0.0
    depth_sigma_multiplier: float = 3.0
    edge_absolute_depth_mm: float = 20.0
    edge_relative_depth: float = 0.02
    edge_guard_radius_pixels: int = 8
    mutual_pixel_tolerance: float = 1.0
    maximum_depth_mm: float | None = None

    def validate(self) -> None:
        finite_positive = (
            "absolute_depth_tolerance_mm",
            "relative_depth_tolerance",
            "depth_sigma_multiplier",
            "edge_absolute_depth_mm",
            "edge_relative_depth",
            "mutual_pixel_tolerance",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        noise = float(self.depth_noise_mm)
        if not math.isfinite(noise) or noise < 0.0:
            raise ValueError("depth_noise_mm must be finite and non-negative")
        if not isinstance(self.edge_guard_radius_pixels, (int, np.integer)) or (
            self.edge_guard_radius_pixels < 0 or self.edge_guard_radius_pixels > 64
        ):
            raise ValueError("edge_guard_radius_pixels must be an integer in [0, 64]")
        if self.maximum_depth_mm is not None:
            maximum = float(self.maximum_depth_mm)
            if not math.isfinite(maximum) or maximum <= 0.0:
                raise ValueError("maximum_depth_mm must be finite and positive")


def _coerce_config(value: GeometryAssistConfig | None) -> GeometryAssistConfig:
    config = GeometryAssistConfig() if value is None else value
    if not isinstance(config, GeometryAssistConfig):
        raise TypeError("config must be a GeometryAssistConfig")
    config.validate()
    return config


def depth_tolerance_mm(
    depth_mm: np.ndarray | float, config: GeometryAssistConfig | None = None
) -> np.ndarray:
    """Return the conservative depth-consistency tolerance in millimetres."""

    settings = _coerce_config(config)
    depth = np.asarray(depth_mm, dtype=np.float64)
    return np.maximum.reduce(
        (
            np.full(depth.shape, settings.absolute_depth_tolerance_mm, dtype=np.float64),
            np.abs(depth) * settings.relative_depth_tolerance,
            np.full(
                depth.shape,
                settings.depth_noise_mm * settings.depth_sigma_multiplier,
                dtype=np.float64,
            ),
        )
    )


def _validate_depth(
    value: np.ndarray,
    intrinsics: GeometryIntrinsics,
    name: str,
    supplied_valid: np.ndarray | None,
    config: GeometryAssistConfig,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(value, dtype=np.float64)
    expected = (intrinsics.height, intrinsics.width)
    if depth.shape != expected:
        raise ValueError(f"{name} must match calibrated aligned-colour shape {expected}")
    if not np.isfinite(depth[np.isfinite(depth)]).all():  # pragma: no cover - defensive
        raise ValueError(f"{name} contains unsupported non-numeric values")
    finite = np.isfinite(depth)
    if np.any(depth[finite] < 0.0):
        raise ValueError(f"{name} cannot contain negative depth")
    valid = finite & (depth > 0.0)
    if config.maximum_depth_mm is not None:
        valid &= depth <= float(config.maximum_depth_mm)
    if supplied_valid is not None:
        mask = np.asarray(supplied_valid)
        if mask.shape != expected or mask.dtype not in {np.dtype(bool), np.dtype(np.uint8)}:
            raise ValueError(f"{name} valid mask must be bool/uint8 and match depth")
        valid &= mask.astype(bool)
    return depth, valid


def sample_aligned_depth_nearest(
    raw_depth_mm: np.ndarray,
    raw_map_x: np.ndarray,
    raw_map_y: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> SampledDepth:
    """Sample aligned depth through a raw-colour inverse map with no blending.

    ``raw_map_x`` and ``raw_map_y`` may have any matching two-dimensional
    narrow-strip shape.  They address the full raw aligned-depth image, exactly
    like an RGB inverse map.  Depth is sampled by nearest neighbour; a map that
    is non-finite, out of bounds, points to zero/non-finite depth, or is masked
    invalid yields an explicit invalid output sample.  This function has no RGB
    argument or I/O and is therefore safe to use while constructing geometry
    evidence before a renderer samples colour.
    """

    raw_depth = np.asarray(raw_depth_mm, dtype=np.float64)
    if raw_depth.ndim != 2:
        raise ValueError("raw_depth_mm must be a two-dimensional aligned-depth array")
    map_x = np.asarray(raw_map_x, dtype=np.float64)
    map_y = np.asarray(raw_map_y, dtype=np.float64)
    if map_x.ndim != 2 or map_x.shape != map_y.shape:
        raise ValueError("raw_map_x and raw_map_y must be matching two-dimensional arrays")
    source_valid = np.isfinite(raw_depth) & (raw_depth > 0.0)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask)
        if supplied.shape != raw_depth.shape or supplied.dtype not in {
            np.dtype(bool),
            np.dtype(np.uint8),
        }:
            raise ValueError("valid_mask must be bool/uint8 and match raw_depth_mm")
        source_valid &= supplied.astype(bool)
    sampled = np.zeros(map_x.shape, dtype=np.float32)
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= raw_depth.shape[1] - 1)
        & (map_y >= 0.0)
        & (map_y <= raw_depth.shape[0] - 1)
    )
    if np.any(valid):
        x = np.rint(map_x[valid]).astype(np.int64)
        y = np.rint(map_y[valid]).astype(np.int64)
        # A floating coordinate inside the continuous image domain can round
        # onto the first index beyond an even-sized image only in pathological
        # precision cases.  Recheck after rounding rather than clipping it.
        rounded_inside = (
            (x >= 0)
            & (x < raw_depth.shape[1])
            & (y >= 0)
            & (y < raw_depth.shape[0])
        )
        valid_positions = np.flatnonzero(valid)
        accepted_positions = valid_positions[rounded_inside]
        accepted_x = x[rounded_inside]
        accepted_y = y[rounded_inside]
        accepted_depth_valid = source_valid[accepted_y, accepted_x]
        final_positions = accepted_positions[accepted_depth_valid]
        if final_positions.size:
            flat_sampled = sampled.reshape(-1)
            flat_sampled[final_positions] = raw_depth[
                accepted_y[accepted_depth_valid], accepted_x[accepted_depth_valid]
            ].astype(np.float32)
            sampled = flat_sampled.reshape(sampled.shape)
        final_valid = np.zeros(map_x.size, dtype=bool)
        final_valid[final_positions] = True
        valid = final_valid.reshape(map_x.shape)
    return SampledDepth(depth_mm=sampled, valid_mask=valid)


def _unproject_pixels(
    u: np.ndarray,
    v: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: GeometryIntrinsics,
) -> np.ndarray:
    """Unproject distorted aligned-colour pixels into camera coordinates."""

    z = np.asarray(depth_mm, dtype=np.float64).reshape(-1)
    pixels = np.column_stack(
        (np.asarray(u, dtype=np.float64).reshape(-1), np.asarray(v, dtype=np.float64).reshape(-1))
    )
    if intrinsics.distortion:
        normalised = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2),
            intrinsics.matrix,
            np.asarray(intrinsics.distortion, dtype=np.float64),
        ).reshape(-1, 2)
        x = normalised[:, 0] * z
        y = normalised[:, 1] * z
    else:
        x = (pixels[:, 0] - intrinsics.cx) * z / intrinsics.fx
        y = (pixels[:, 1] - intrinsics.cy) * z / intrinsics.fy
    return np.column_stack((x, y, z))


def _project_camera_points(
    points_camera_mm: np.ndarray, intrinsics: GeometryIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Project positive-camera-z points to native aligned-colour pixels."""

    points = np.asarray(points_camera_mm, dtype=np.float64).reshape(-1, 3)
    if intrinsics.distortion:
        pixels, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            intrinsics.matrix,
            np.asarray(intrinsics.distortion, dtype=np.float64),
        )
        projected = pixels.reshape(-1, 2)
        return projected[:, 0], projected[:, 1]
    positive_z = np.maximum(points[:, 2], 1e-12)
    return (
        intrinsics.fx * points[:, 0] / positive_z + intrinsics.cx,
        intrinsics.fy * points[:, 1] / positive_z + intrinsics.cy,
    )


def _to_world(points_camera: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points_camera @ pose[:3, :3].T + pose[:3, 3]


def _to_camera(points_world: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (points_world - pose[:3, 3]) @ pose[:3, :3]


def depth_edge_guard(
    depth_mm: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    config: GeometryAssistConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(edge_mask, dilated_guard)`` without crossing invalid depth.

    A depth edge is placed on both samples of a discontinuous horizontal or
    vertical pair.  The guard also marks the valid side of a depth hole, so a
    local deformation cannot interpolate a visible surface into unknown depth.
    """

    settings = _coerce_config(config)
    depth = np.asarray(depth_mm, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_edge_guard requires a two-dimensional depth array")
    valid = np.isfinite(depth) & (depth > 0.0)
    if settings.maximum_depth_mm is not None:
        valid &= depth <= float(settings.maximum_depth_mm)
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.shape != depth.shape or mask.dtype not in {np.dtype(bool), np.dtype(np.uint8)}:
            raise ValueError("depth_edge_guard valid_mask must be bool/uint8 and match depth")
        valid &= mask.astype(bool)

    edge = np.zeros(depth.shape, dtype=bool)
    if depth.shape[1] > 1:
        left, right = depth[:, :-1], depth[:, 1:]
        left_valid, right_valid = valid[:, :-1], valid[:, 1:]
        threshold = np.maximum(
            settings.edge_absolute_depth_mm,
            np.minimum(left, right) * settings.edge_relative_depth,
        )
        discontinuity = left_valid & right_valid & (np.abs(left - right) > threshold)
        hole_boundary = left_valid ^ right_valid
        marked = discontinuity | hole_boundary
        edge[:, :-1] |= marked
        edge[:, 1:] |= marked
    if depth.shape[0] > 1:
        top, bottom = depth[:-1, :], depth[1:, :]
        top_valid, bottom_valid = valid[:-1, :], valid[1:, :]
        threshold = np.maximum(
            settings.edge_absolute_depth_mm,
            np.minimum(top, bottom) * settings.edge_relative_depth,
        )
        discontinuity = top_valid & bottom_valid & (np.abs(top - bottom) > threshold)
        hole_boundary = top_valid ^ bottom_valid
        marked = discontinuity | hole_boundary
        edge[:-1, :] |= marked
        edge[1:, :] |= marked

    if settings.edge_guard_radius_pixels == 0:
        return edge, edge.copy()
    radius = int(settings.edge_guard_radius_pixels)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    guard = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    return edge, guard


class LayerLabel(IntEnum):
    """Visibility classification of a source sample in the opposite camera."""

    UNKNOWN = 0
    CONSISTENT = 1
    FOREGROUND = 2
    OCCLUDED = 3
    DISOCCLUDED = 4


@dataclass(frozen=True)
class DirectedReprojection:
    """One depth source reprojected into an adjacent real camera.

    All source-shaped masks live in the source image coordinate system.  The
    z-buffer arrays live in the target coordinate system and are useful for
    audits only; no RGB is stored or chosen here.
    """

    source_valid: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    projected_depth_mm: np.ndarray
    target_depth_mm: np.ndarray
    depth_residual_ratio: np.ndarray
    in_target_bounds: np.ndarray
    zbuffer_visible: np.ndarray
    target_valid: np.ndarray
    depth_consistent: np.ndarray
    source_foreground: np.ndarray
    source_occluded: np.ndarray
    target_missing: np.ndarray
    source_depth_edge: np.ndarray
    source_edge_guard: np.ndarray
    target_edge_guard: np.ndarray
    mutual_consistent: np.ndarray
    labels: np.ndarray
    zbuffer_depth_mm: np.ndarray
    zbuffer_source_index: np.ndarray

    @property
    def protected_mask(self) -> np.ndarray:
        """Pixels unsafe for cross-source blending or a cross-layer warp."""

        return (
            self.source_edge_guard
            | self.target_edge_guard
            | (self.labels != int(LayerLabel.CONSISTENT))
            # A one-way depth comparison is not enough for a local inverse
            # field.  Quantisation, z-buffer collisions and a target-side
            # round-trip miss can all leave a sample labelled CONSISTENT in
            # this direction while the corresponding source coordinate does
            # not return within the configured bilateral tolerance.  Keep
            # that sample hard-owned rather than allowing a mesh cell to span
            # it merely because other fit samples were mutual.
            | ~self.mutual_consistent
        )

    def as_dict(self) -> dict[str, int | float | None]:
        mutual_ratio = np.asarray(self.depth_residual_ratio, dtype=np.float64)[
            np.asarray(self.mutual_consistent, dtype=bool)
        ]
        mutual_ratio = mutual_ratio[np.isfinite(mutual_ratio)]
        return {
            "source_valid_pixel_count": int(np.count_nonzero(self.source_valid)),
            "in_target_bounds_pixel_count": int(np.count_nonzero(self.in_target_bounds)),
            "zbuffer_visible_pixel_count": int(np.count_nonzero(self.zbuffer_visible)),
            "depth_consistent_pixel_count": int(np.count_nonzero(self.depth_consistent)),
            "mutual_consistent_pixel_count": int(
                np.count_nonzero(self.mutual_consistent)
            ),
            "mutual_depth_residual_ratio_p95": (
                float(np.percentile(mutual_ratio, 95.0))
                if mutual_ratio.size
                else None
            ),
            "mutual_depth_residual_ratio_max": (
                float(np.max(mutual_ratio)) if mutual_ratio.size else None
            ),
            "foreground_pixel_count": int(np.count_nonzero(self.source_foreground)),
            "occluded_pixel_count": int(np.count_nonzero(self.source_occluded)),
            "disoccluded_pixel_count": int(np.count_nonzero(self.target_missing)),
            "depth_edge_pixel_count": int(np.count_nonzero(self.source_depth_edge)),
            "depth_edge_guard_pixel_count": int(np.count_nonzero(self.source_edge_guard)),
            "zbuffer_selected_pixel_count": int(
                np.count_nonzero(np.isfinite(self.zbuffer_depth_mm))
            ),
        }


@dataclass(frozen=True)
class DirectedSurfaceSafety:
    """Depth-layer safety for one source image, without any RGB data.

    A mutually visible source point may still belong to a fully visible
    foreground object.  This classification makes that distinction explicit:
    only one dominant far/background component remains mesh-safe; near and
    ambiguous components are owner-only even when both cameras observe them.
    """

    mesh_safe_mask: np.ndarray = field(repr=False)
    near_foreground_mask: np.ndarray = field(repr=False)
    ambiguous_or_unreliable_mask: np.ndarray = field(repr=False)
    hard_owner_mask: np.ndarray = field(repr=False)
    component_count: int
    material_component_count: int
    dominant_component_pixel_count: int
    dominant_component_fraction: float
    dominant_component_median_depth_mm: float | None
    near_foreground_component_count: int
    depth_anchor_component_count: int
    analysis_scope: str
    analysis_pixel_count: int
    base_safe_pixel_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": (
                "one_dominant_far_depth_component_mesh_safe_"
                "near_and_ambiguous_components_hard_owner"
            ),
            "component_count": int(self.component_count),
            "material_component_count": int(self.material_component_count),
            "dominant_component_pixel_count": int(
                self.dominant_component_pixel_count
            ),
            "dominant_component_fraction": float(self.dominant_component_fraction),
            "dominant_component_median_depth_mm": (
                float(self.dominant_component_median_depth_mm)
                if self.dominant_component_median_depth_mm is not None
                else None
            ),
            "mesh_safe_pixel_count": int(np.count_nonzero(self.mesh_safe_mask)),
            "near_foreground_component_count": int(
                self.near_foreground_component_count
            ),
            "depth_anchor_component_count": int(self.depth_anchor_component_count),
            "near_foreground_pixel_count": int(
                np.count_nonzero(self.near_foreground_mask)
            ),
            "ambiguous_or_unreliable_pixel_count": int(
                np.count_nonzero(self.ambiguous_or_unreliable_mask)
            ),
            "hard_owner_pixel_count": int(np.count_nonzero(self.hard_owner_mask)),
            "analysis_scope": self.analysis_scope,
            "analysis_pixel_count": int(self.analysis_pixel_count),
            "base_safe_pixel_count": int(self.base_safe_pixel_count),
        }


@dataclass(frozen=True)
class PairGeometryAudit:
    """Small scalar audit for a complete adjacent RGB-D pair."""

    first_to_second: dict[str, int | float | None]
    second_to_first: dict[str, int | float | None]
    first_depth_edge_guard_pixel_count: int
    second_depth_edge_guard_pixel_count: int
    first_surface_safety: dict[str, object]
    second_surface_safety: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "first_to_second": dict(self.first_to_second),
            "second_to_first": dict(self.second_to_first),
            "first_depth_edge_guard_pixel_count": self.first_depth_edge_guard_pixel_count,
            "second_depth_edge_guard_pixel_count": self.second_depth_edge_guard_pixel_count,
            "first_surface_safety": dict(self.first_surface_safety),
            "second_surface_safety": dict(self.second_surface_safety),
        }


@dataclass(frozen=True)
class AdjacentPairGeometry:
    """Bidirectional geometry evidence for exactly one adjacent source pair."""

    first_to_second: DirectedReprojection
    second_to_first: DirectedReprojection
    first_surface_safety: DirectedSurfaceSafety
    second_surface_safety: DirectedSurfaceSafety
    audit: PairGeometryAudit


@dataclass(frozen=True)
class ForegroundInstanceMatch:
    """One uniquely associated signed-occlusion instance across a pair.

    Both source components are direct signed occlusion observations: their
    source point is visibly in front of the target depth in its own direction.
    The association itself is measured only in the calibrated narrow virtual
    corridor where both raw components appear.  This is evidence for an RGB
    owner decision only; it contains neither RGB samples nor a warp and never
    changes a camera pose.
    """

    instance_id: int
    first_component_label: int
    second_component_label: int
    first_component_pixel_count: int
    second_component_pixel_count: int
    first_virtual_pixel_count: int
    second_virtual_pixel_count: int
    virtual_overlap_pixel_count: int
    virtual_overlap_row_count: int

    def as_dict(self) -> dict[str, int]:
        return {
            "instance_id": int(self.instance_id),
            "first_component_label": int(self.first_component_label),
            "second_component_label": int(self.second_component_label),
            "first_component_pixel_count": int(self.first_component_pixel_count),
            "second_component_pixel_count": int(self.second_component_pixel_count),
            "first_virtual_pixel_count": int(self.first_virtual_pixel_count),
            "second_virtual_pixel_count": int(self.second_virtual_pixel_count),
            "virtual_overlap_pixel_count": int(self.virtual_overlap_pixel_count),
            "virtual_overlap_row_count": int(self.virtual_overlap_row_count),
        }


@dataclass(frozen=True)
class SignedOcclusionForegroundComponents:
    """Material raw components of one directed signed foreground occlusion.

    The label image remains in the native aligned-colour coordinate system.
    It contains only direct ``FOREGROUND`` observations with a positive signed
    depth residual; it deliberately excludes mutually-consistent same-layer
    samples, background/occluded samples, and any inferred grown region.
    """

    labels: np.ndarray = field(repr=False)
    component_pixel_counts: Mapping[int, int]
    signed_occlusion_pixel_count: int
    eroded_core_pixel_count: int
    rejected_component_count: int

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 2 or labels.dtype.kind not in {"i", "u"}:
            raise ValueError("Signed occlusion labels must be a two-dimensional integer image")
        counts = {int(label): int(count) for label, count in self.component_pixel_counts.items()}
        if any(label <= 0 or count <= 0 for label, count in counts.items()):
            raise ValueError("Signed occlusion component labels/counts must be positive")
        if set(np.unique(labels).tolist()) - {0, *counts}:
            raise ValueError("Signed occlusion labels contain an unknown component")
        for value in (
            self.signed_occlusion_pixel_count,
            self.eroded_core_pixel_count,
            self.rejected_component_count,
        ):
            if int(value) < 0:
                raise ValueError("Signed occlusion counts must be non-negative")
        object.__setattr__(self, "labels", np.ascontiguousarray(labels, dtype=np.int32))
        object.__setattr__(self, "component_pixel_counts", counts)
        object.__setattr__(
            self,
            "signed_occlusion_pixel_count",
            int(self.signed_occlusion_pixel_count),
        )
        object.__setattr__(self, "eroded_core_pixel_count", int(self.eroded_core_pixel_count))
        object.__setattr__(
            self,
            "rejected_component_count",
            int(self.rejected_component_count),
        )

    @property
    def component_count(self) -> int:
        return len(self.component_pixel_counts)


@dataclass(frozen=True)
class BidirectionalForegroundInstances:
    """Matched owner-only signed-occlusion labels for an adjacent RGB-D pair.

    ``first_instance_labels`` and ``second_instance_labels`` share one small
    instance-id namespace.  A nonzero value is present only after both
    directions have a material direct foreground occlusion and their
    calibrated virtual footprints identify one another uniquely.  Split,
    merge, hole, same-layer, and ambiguous components are absent rather than
    guessed.
    """

    first_instance_labels: np.ndarray = field(repr=False)
    second_instance_labels: np.ndarray = field(repr=False)
    matches: tuple[ForegroundInstanceMatch, ...]
    rejected_component_count: int
    first_signed_occlusion_pixel_count: int = 0
    second_signed_occlusion_pixel_count: int = 0
    first_eroded_occlusion_core_pixel_count: int = 0
    second_eroded_occlusion_core_pixel_count: int = 0

    def __post_init__(self) -> None:
        first = np.asarray(self.first_instance_labels)
        second = np.asarray(self.second_instance_labels)
        if (
            first.ndim != 2
            or second.ndim != 2
            or first.shape != second.shape
            or first.dtype.kind not in {"i", "u"}
            or second.dtype.kind not in {"i", "u"}
        ):
            raise ValueError("Foreground instance labels must be matching integer images")
        if int(self.rejected_component_count) < 0:
            raise ValueError("Foreground rejected component count must be non-negative")
        for value in (
            self.first_signed_occlusion_pixel_count,
            self.second_signed_occlusion_pixel_count,
            self.first_eroded_occlusion_core_pixel_count,
            self.second_eroded_occlusion_core_pixel_count,
        ):
            if int(value) < 0:
                raise ValueError("Foreground signed-occlusion counts must be non-negative")
        instance_ids = {match.instance_id for match in self.matches}
        if any(int(value) <= 0 for value in instance_ids):
            raise ValueError("Foreground instance ids must be positive")
        if len(instance_ids) != len(self.matches):
            raise ValueError("Foreground instance ids must be unique")
        permitted = {0, *instance_ids}
        if (
            not set(np.unique(first).tolist()).issubset(permitted)
            or not set(np.unique(second).tolist()).issubset(permitted)
        ):
            raise ValueError("Foreground instance labels contain an unknown id")
        object.__setattr__(self, "first_instance_labels", np.ascontiguousarray(first))
        object.__setattr__(self, "second_instance_labels", np.ascontiguousarray(second))
        object.__setattr__(self, "rejected_component_count", int(self.rejected_component_count))
        object.__setattr__(
            self,
            "first_signed_occlusion_pixel_count",
            int(self.first_signed_occlusion_pixel_count),
        )
        object.__setattr__(
            self,
            "second_signed_occlusion_pixel_count",
            int(self.second_signed_occlusion_pixel_count),
        )
        object.__setattr__(
            self,
            "first_eroded_occlusion_core_pixel_count",
            int(self.first_eroded_occlusion_core_pixel_count),
        )
        object.__setattr__(
            self,
            "second_eroded_occlusion_core_pixel_count",
            int(self.second_eroded_occlusion_core_pixel_count),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": (
                "reciprocal_unique_signed_occlusion_virtual_instances_owner_only"
            ),
            "matched_instance_count": len(self.matches),
            "rejected_component_count": int(self.rejected_component_count),
            "first_signed_occlusion_pixel_count": int(
                self.first_signed_occlusion_pixel_count
            ),
            "second_signed_occlusion_pixel_count": int(
                self.second_signed_occlusion_pixel_count
            ),
            "first_eroded_occlusion_core_pixel_count": int(
                self.first_eroded_occlusion_core_pixel_count
            ),
            "second_eroded_occlusion_core_pixel_count": int(
                self.second_eroded_occlusion_core_pixel_count
            ),
            "instances": [match.as_dict() for match in self.matches],
        }


def _empty_directed(
    shape: tuple[int, int],
    source_valid: np.ndarray,
    source_depth_edge: np.ndarray,
    source_guard: np.ndarray,
) -> DirectedReprojection:
    nan = np.full(shape, np.nan, dtype=np.float32)
    false = np.zeros(shape, dtype=bool)
    labels = np.full(shape, int(LayerLabel.UNKNOWN), dtype=np.uint8)
    return DirectedReprojection(
        source_valid=np.asarray(source_valid, dtype=bool),
        target_x=nan.copy(),
        target_y=nan.copy(),
        projected_depth_mm=nan.copy(),
        target_depth_mm=nan.copy(),
        depth_residual_ratio=nan.copy(),
        in_target_bounds=false.copy(),
        zbuffer_visible=false.copy(),
        target_valid=false.copy(),
        depth_consistent=false.copy(),
        source_foreground=false.copy(),
        source_occluded=false.copy(),
        target_missing=false.copy(),
        source_depth_edge=np.asarray(source_depth_edge, dtype=bool),
        source_edge_guard=np.asarray(source_guard, dtype=bool),
        target_edge_guard=false.copy(),
        mutual_consistent=false.copy(),
        labels=labels,
        zbuffer_depth_mm=nan.copy(),
        zbuffer_source_index=np.full(shape, -1, dtype=np.int32),
    )


def _zbuffer_winners(
    target_shape: tuple[int, int],
    source_flat_indices: np.ndarray,
    target_u: np.ndarray,
    target_v: np.ndarray,
    target_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose nearest point splats per target pixel without interpolation."""

    height, width = target_shape
    winners = np.zeros(source_flat_indices.size, dtype=bool)
    zbuffer_depth = np.full(target_shape, np.inf, dtype=np.float32)
    zbuffer_source = np.full(target_shape, -1, dtype=np.int32)
    if not source_flat_indices.size:
        return winners, zbuffer_depth, zbuffer_source
    target_x = np.rint(target_u).astype(np.int64)
    target_y = np.rint(target_v).astype(np.int64)
    flattened_target = target_y * width + target_x
    # Sorting by target, then z, then original source index makes equal-depth
    # ties deterministic and never blends two samples.
    order = np.lexsort((source_flat_indices, target_depth, flattened_target))
    ordered_target = flattened_target[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = ordered_target[1:] != ordered_target[:-1]
    selected_positions = order[first]
    winners[selected_positions] = True
    selected_x = target_x[selected_positions]
    selected_y = target_y[selected_positions]
    zbuffer_depth[selected_y, selected_x] = target_depth[selected_positions].astype(
        np.float32
    )
    zbuffer_source[selected_y, selected_x] = source_flat_indices[selected_positions]
    return winners, zbuffer_depth, zbuffer_source


def _directed_reprojection(
    source_depth: np.ndarray,
    source_valid: np.ndarray,
    target_depth: np.ndarray,
    target_valid: np.ndarray,
    intrinsics: GeometryIntrinsics,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    source_edge: np.ndarray,
    source_guard: np.ndarray,
    target_guard: np.ndarray,
    config: GeometryAssistConfig,
) -> DirectedReprojection:
    """Reproject one source depth map and classify it against a target map."""

    shape = source_depth.shape
    result = _empty_directed(shape, source_valid, source_edge, source_guard)
    flat_source = np.flatnonzero(source_valid)
    if not flat_source.size:
        return result
    source_y, source_x = np.unravel_index(flat_source, shape)
    source_z = source_depth[source_y, source_x]
    source_points = _unproject_pixels(source_x, source_y, source_z, intrinsics)
    world_points = _to_world(source_points, source_pose)
    target_points = _to_camera(world_points, target_pose)
    positive_z = target_points[:, 2] > 1e-6
    target_u = np.full(flat_source.shape, np.nan, dtype=np.float64)
    target_v = np.full(flat_source.shape, np.nan, dtype=np.float64)
    if np.any(positive_z):
        projected_u, projected_v = _project_camera_points(
            target_points[positive_z], intrinsics
        )
        target_u[positive_z] = projected_u
        target_v[positive_z] = projected_v
    inside = (
        positive_z
        & np.isfinite(target_u)
        & np.isfinite(target_v)
        & (target_u >= 0.0)
        & (target_u <= intrinsics.width - 1)
        & (target_v >= 0.0)
        & (target_v <= intrinsics.height - 1)
    )

    result_target_x = result.target_x.copy()
    result_target_y = result.target_y.copy()
    result_projected_depth = result.projected_depth_mm.copy()
    result_inside = result.in_target_bounds.copy()
    result_target_x[source_y, source_x] = target_u.astype(np.float32)
    result_target_y[source_y, source_x] = target_v.astype(np.float32)
    result_projected_depth[source_y, source_x] = target_points[:, 2].astype(np.float32)
    result_inside[source_y, source_x] = inside

    candidate_positions = np.flatnonzero(inside)
    selected_candidate, zbuffer_depth, zbuffer_source = _zbuffer_winners(
        target_depth.shape,
        flat_source[candidate_positions],
        target_u[candidate_positions],
        target_v[candidate_positions],
        target_points[candidate_positions, 2],
    )
    zbuffer_visible = result.zbuffer_visible.copy()
    if candidate_positions.size:
        winner_positions = candidate_positions[selected_candidate]
        zbuffer_visible[source_y[winner_positions], source_x[winner_positions]] = True

    sampled_target_depth = result.target_depth_mm.copy()
    residual_ratio = result.depth_residual_ratio.copy()
    sampled_target_valid = result.target_valid.copy()
    sampled_target_guard = result.target_edge_guard.copy()
    if candidate_positions.size:
        target_ix = np.rint(target_u[candidate_positions]).astype(np.int64)
        target_iy = np.rint(target_v[candidate_positions]).astype(np.int64)
        source_candidate_y = source_y[candidate_positions]
        source_candidate_x = source_x[candidate_positions]
        sampled_target_depth[source_candidate_y, source_candidate_x] = target_depth[
            target_iy, target_ix
        ].astype(np.float32)
        sampled_target_valid[source_candidate_y, source_candidate_x] = target_valid[
            target_iy, target_ix
        ]
        sampled_target_guard[source_candidate_y, source_candidate_x] = target_guard[
            target_iy, target_ix
        ]

    target_missing = result.target_missing.copy()
    source_foreground = result.source_foreground.copy()
    source_occluded = result.source_occluded.copy()
    consistent = result.depth_consistent.copy()
    labels = result.labels.copy()
    # A source point outside the neighbour's field of view is a disocclusion,
    # not a zero-error correspondence.  It cannot participate in a local warp.
    labels[source_valid & ~result_inside] = int(LayerLabel.DISOCCLUDED)
    target_missing[source_valid & ~result_inside] = True
    nonvisible = source_valid & result_inside & ~zbuffer_visible
    labels[nonvisible] = int(LayerLabel.OCCLUDED)
    source_occluded[nonvisible] = True
    visible = source_valid & result_inside & zbuffer_visible
    missing_target = visible & ~sampled_target_valid
    labels[missing_target] = int(LayerLabel.DISOCCLUDED)
    target_missing[missing_target] = True
    comparable = visible & sampled_target_valid
    if np.any(comparable):
        difference = sampled_target_depth[comparable].astype(np.float64) - result_projected_depth[
            comparable
        ].astype(np.float64)
        reference_depth = np.maximum(
            sampled_target_depth[comparable].astype(np.float64),
            result_projected_depth[comparable].astype(np.float64),
        )
        tolerance = depth_tolerance_mm(reference_depth, config)
        residual_ratio_values = np.abs(difference) / tolerance
        equal = np.abs(difference) <= tolerance
        foreground = difference > tolerance
        hidden = difference < -tolerance
        comparable_indices = np.flatnonzero(comparable)
        flat_labels = labels.reshape(-1)
        flat_consistent = consistent.reshape(-1)
        flat_foreground = source_foreground.reshape(-1)
        flat_occluded = source_occluded.reshape(-1)
        flat_residual_ratio = residual_ratio.reshape(-1)
        flat_residual_ratio[comparable_indices] = residual_ratio_values.astype(
            np.float32
        )
        flat_labels[comparable_indices[equal]] = int(LayerLabel.CONSISTENT)
        flat_consistent[comparable_indices[equal]] = True
        flat_labels[comparable_indices[foreground]] = int(LayerLabel.FOREGROUND)
        flat_foreground[comparable_indices[foreground]] = True
        flat_labels[comparable_indices[hidden]] = int(LayerLabel.OCCLUDED)
        flat_occluded[comparable_indices[hidden]] = True
        labels = flat_labels.reshape(shape)
        consistent = flat_consistent.reshape(shape)
        source_foreground = flat_foreground.reshape(shape)
        source_occluded = flat_occluded.reshape(shape)
        residual_ratio = flat_residual_ratio.reshape(shape)

    return DirectedReprojection(
        source_valid=np.asarray(source_valid, dtype=bool),
        target_x=result_target_x,
        target_y=result_target_y,
        projected_depth_mm=result_projected_depth,
        target_depth_mm=sampled_target_depth,
        depth_residual_ratio=residual_ratio,
        in_target_bounds=result_inside,
        zbuffer_visible=zbuffer_visible,
        target_valid=sampled_target_valid,
        depth_consistent=consistent,
        source_foreground=source_foreground,
        source_occluded=source_occluded,
        target_missing=target_missing,
        source_depth_edge=np.asarray(source_edge, dtype=bool),
        source_edge_guard=np.asarray(source_guard, dtype=bool),
        target_edge_guard=sampled_target_guard,
        mutual_consistent=np.zeros(shape, dtype=bool),
        labels=labels,
        zbuffer_depth_mm=zbuffer_depth,
        zbuffer_source_index=zbuffer_source,
    )


def _mutual_consistency(
    forward: DirectedReprojection,
    backward: DirectedReprojection,
    tolerance_pixels: float,
) -> np.ndarray:
    """Require a forward visible match to return to its original source pixel."""

    candidate = forward.depth_consistent & forward.zbuffer_visible
    result = np.zeros(candidate.shape, dtype=bool)
    if not np.any(candidate):
        return result
    source_y, source_x = np.nonzero(candidate)
    target_x = np.rint(forward.target_x[source_y, source_x]).astype(np.int64)
    target_y = np.rint(forward.target_y[source_y, source_x]).astype(np.int64)
    reverse_good = (
        backward.depth_consistent[target_y, target_x]
        & backward.zbuffer_visible[target_y, target_x]
    )
    returned_x = backward.target_x[target_y, target_x]
    returned_y = backward.target_y[target_y, target_x]
    error = np.hypot(returned_x - source_x, returned_y - source_y)
    accepted = reverse_good & np.isfinite(error) & (error <= tolerance_pixels)
    result[source_y[accepted], source_x[accepted]] = True
    return result


_MINIMUM_MATERIAL_SURFACE_COMPONENT_PIXELS = 64
_MINIMUM_DOMINANT_BACKGROUND_FRACTION = 0.50
_MAXIMUM_CONFIDENT_FOREGROUND_FRACTION = 0.45
# A tiny distant hole/sliver must not redefine the wall merely because it is
# numerically the deepest valid depth in the corridor. Only components with
# this minimum share of local safe support may anchor the background depth;
# the eventual mesh still requires one >=50% component.
_MINIMUM_DEPTH_ANCHOR_COMPONENT_FRACTION = 0.10


def classify_directed_surface_safety(
    reprojection: DirectedReprojection,
    source_depth_mm: np.ndarray,
    *,
    analysis_mask: np.ndarray | None = None,
    analysis_scope: str | None = None,
    config: GeometryAssistConfig | None = None,
) -> DirectedSurfaceSafety:
    """Classify one directed depth image into mesh-safe and owner-only layers.

    A complete foreground object can be mutually visible in both cameras, so
    bilateral reprojection alone does not make it a safe background surface.
    This deliberately conservative first-stage classifier retains at most one
    material, far-depth connected component as a mesh-safe background.  Any
    nearer component, small/fragmented component, invalid depth, or component
    whose background evidence is ambiguous becomes a hard-owner region.

    The classifier consumes only depth/reprojection masks.  It neither reads
    RGB nor estimates a plane or a camera motion.  The optional analysis mask
    lets a seam planner classify only the native pixels represented by its
    96--160 px calibrated corridor; unrelated depth clutter elsewhere in a
    full source image must not decide whether that corridor has one safe wall.
    """

    settings = _coerce_config(config)
    depth = np.asarray(source_depth_mm, dtype=np.float64)
    shape = np.asarray(reprojection.source_valid, dtype=bool).shape
    if depth.shape != shape or depth.ndim != 2:
        raise ValueError("Surface-safety depth must match the directed source shape")
    fields = (
        reprojection.mutual_consistent,
        reprojection.labels,
        reprojection.source_edge_guard,
        reprojection.target_edge_guard,
    )
    if any(np.asarray(value).shape != shape for value in fields):
        raise ValueError("Directed surface-safety masks must share one source shape")
    if analysis_mask is None:
        analysis = np.ones(shape, dtype=bool)
        scope = "full_source"
    else:
        analysis = np.asarray(analysis_mask, dtype=bool)
        if analysis.shape != shape or analysis.ndim != 2:
            raise ValueError("Surface-safety analysis mask must match the source shape")
        scope = "explicit_source_footprint"
    if analysis_scope is not None:
        if not isinstance(analysis_scope, str) or not analysis_scope:
            raise ValueError("Surface-safety analysis scope must be a non-empty string")
        scope = analysis_scope
    analysis_count = int(np.count_nonzero(analysis))
    depth_valid = np.isfinite(depth) & (depth > 0.0)
    source_valid = np.asarray(reprojection.source_valid, dtype=bool) & depth_valid
    base_safe = (
        source_valid
        & np.asarray(reprojection.mutual_consistent, dtype=bool)
        & (np.asarray(reprojection.labels) == int(LayerLabel.CONSISTENT))
        & ~np.asarray(reprojection.source_edge_guard, dtype=bool)
        & ~np.asarray(reprojection.target_edge_guard, dtype=bool)
        & analysis
    )
    base_count = int(np.count_nonzero(base_safe))
    mesh_safe = np.zeros(shape, dtype=bool)
    near_foreground = np.zeros(shape, dtype=bool)
    if not base_count:
        ambiguous = np.ones(shape, dtype=bool)
        return DirectedSurfaceSafety(
            mesh_safe_mask=mesh_safe,
            near_foreground_mask=near_foreground,
            ambiguous_or_unreliable_mask=ambiguous,
            hard_owner_mask=ambiguous.copy(),
            component_count=0,
            material_component_count=0,
            dominant_component_pixel_count=0,
            dominant_component_fraction=0.0,
            dominant_component_median_depth_mm=None,
            near_foreground_component_count=0,
            depth_anchor_component_count=0,
            analysis_scope=scope,
            analysis_pixel_count=analysis_count,
            base_safe_pixel_count=0,
        )

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        base_safe.astype(np.uint8), connectivity=4
    )
    records: list[tuple[int, int, float]] = []
    for label in range(1, int(component_count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < _MINIMUM_MATERIAL_SURFACE_COMPONENT_PIXELS:
            continue
        values = depth[labels == label]
        values = values[np.isfinite(values) & (values > 0.0)]
        if not values.size:
            continue
        records.append((label, area, float(np.median(values))))

    dominant_label: int | None = None
    dominant_area = 0
    dominant_depth: float | None = None
    depth_anchor_records = [
        record
        for record in records
        if float(record[1] / base_count) >= _MINIMUM_DEPTH_ANCHOR_COMPONENT_FRACTION
    ]
    if depth_anchor_records:
        deepest_depth = max(record[2] for record in depth_anchor_records)
        deepest_tolerance = float(
            depth_tolerance_mm(np.asarray((deepest_depth,), dtype=np.float64), settings)[0]
        )
        far_candidates = [
            record
            for record in records
            if record[2] >= deepest_depth - deepest_tolerance
        ]
        # Prefer material coverage first, then a farther median and a stable
        # label tie-break.  A small distant sliver cannot become the declared
        # background because the fraction gate below rejects it.
        label, area, median_depth = max(
            far_candidates,
            key=lambda record: (record[1], record[2], -record[0]),
        )
        fraction = float(area / base_count)
        if fraction >= _MINIMUM_DOMINANT_BACKGROUND_FRACTION:
            dominant_label = int(label)
            dominant_area = int(area)
            dominant_depth = float(median_depth)
            mesh_safe = labels == dominant_label

    near_component_count = 0
    if dominant_depth is not None:
        for label, area, median_depth in records:
            if label == dominant_label:
                continue
            tolerance = float(
                depth_tolerance_mm(
                    np.asarray((max(median_depth, dominant_depth),), dtype=np.float64),
                    settings,
                )[0]
            )
            if (
                float(area / base_count) <= _MAXIMUM_CONFIDENT_FOREGROUND_FRACTION
                and median_depth + tolerance < dominant_depth
            ):
                near_foreground |= labels == label
                near_component_count += 1

    # Every non-background source position, including a depth hole or an
    # under-supported/small component, is owner-only.  This is intentionally
    # stricter than just guarding a depth edge: a thick hose or handle stays
    # with one original RGB source throughout its interior.
    ambiguous = ~mesh_safe & ~near_foreground
    return DirectedSurfaceSafety(
        mesh_safe_mask=np.ascontiguousarray(mesh_safe),
        near_foreground_mask=np.ascontiguousarray(near_foreground),
        ambiguous_or_unreliable_mask=np.ascontiguousarray(ambiguous),
        hard_owner_mask=np.ascontiguousarray(~mesh_safe),
        component_count=int(component_count - 1),
        material_component_count=len(records),
        dominant_component_pixel_count=dominant_area,
        dominant_component_fraction=(float(dominant_area / base_count) if base_count else 0.0),
        dominant_component_median_depth_mm=dominant_depth,
        near_foreground_component_count=near_component_count,
        depth_anchor_component_count=len(depth_anchor_records),
        analysis_scope=scope,
        analysis_pixel_count=analysis_count,
        base_safe_pixel_count=base_count,
    )


def analyze_adjacent_rgbd_pair(
    first_depth_mm: np.ndarray,
    second_depth_mm: np.ndarray,
    intrinsics: IntrinsicsLike | Mapping[str, object],
    first_camera_to_world: np.ndarray,
    second_camera_to_world: np.ndarray,
    *,
    first_valid_mask: np.ndarray | None = None,
    second_valid_mask: np.ndarray | None = None,
    first_analysis_mask: np.ndarray | None = None,
    second_analysis_mask: np.ndarray | None = None,
    analysis_scope: str | None = None,
    config: GeometryAssistConfig | None = None,
) -> AdjacentPairGeometry:
    """Build pure bidirectional geometry evidence for one adjacent RGB-D pair.

    The inputs must be aligned depth images in the calibrated colour domain and
    real camera-to-world poses whose translations are millimetres.  The result
    contains only geometry and masks; it deliberately has no RGB argument.
    """

    settings = _coerce_config(config)
    camera = coerce_intrinsics(intrinsics)
    first_pose = validate_camera_to_world(first_camera_to_world)
    second_pose = validate_camera_to_world(second_camera_to_world)
    first_depth, first_valid = _validate_depth(
        first_depth_mm, camera, "first_depth_mm", first_valid_mask, settings
    )
    second_depth, second_valid = _validate_depth(
        second_depth_mm, camera, "second_depth_mm", second_valid_mask, settings
    )
    first_edge, first_guard = depth_edge_guard(
        first_depth, valid_mask=first_valid, config=settings
    )
    second_edge, second_guard = depth_edge_guard(
        second_depth, valid_mask=second_valid, config=settings
    )
    first_to_second = _directed_reprojection(
        first_depth,
        first_valid,
        second_depth,
        second_valid,
        camera,
        first_pose,
        second_pose,
        first_edge,
        first_guard,
        second_guard,
        settings,
    )
    second_to_first = _directed_reprojection(
        second_depth,
        second_valid,
        first_depth,
        first_valid,
        camera,
        second_pose,
        first_pose,
        second_edge,
        second_guard,
        first_guard,
        settings,
    )
    first_mutual = _mutual_consistency(
        first_to_second, second_to_first, settings.mutual_pixel_tolerance
    )
    second_mutual = _mutual_consistency(
        second_to_first, first_to_second, settings.mutual_pixel_tolerance
    )
    first_to_second = replace(first_to_second, mutual_consistent=first_mutual)
    second_to_first = replace(second_to_first, mutual_consistent=second_mutual)
    first_surface_safety = classify_directed_surface_safety(
        first_to_second,
        first_depth,
        analysis_mask=first_analysis_mask,
        analysis_scope=analysis_scope,
        config=settings,
    )
    second_surface_safety = classify_directed_surface_safety(
        second_to_first,
        second_depth,
        analysis_mask=second_analysis_mask,
        analysis_scope=analysis_scope,
        config=settings,
    )
    audit = PairGeometryAudit(
        first_to_second=first_to_second.as_dict(),
        second_to_first=second_to_first.as_dict(),
        first_depth_edge_guard_pixel_count=int(np.count_nonzero(first_guard)),
        second_depth_edge_guard_pixel_count=int(np.count_nonzero(second_guard)),
        first_surface_safety=first_surface_safety.as_dict(),
        second_surface_safety=second_surface_safety.as_dict(),
    )
    return AdjacentPairGeometry(
        first_to_second,
        second_to_first,
        first_surface_safety,
        second_surface_safety,
        audit,
    )


_MINIMUM_SIGNED_OCCLUSION_COMPONENT_PIXELS = 64
_MINIMUM_SIGNED_OCCLUSION_ERODED_CORE_PIXELS = 32
_MINIMUM_SIGNED_OCCLUSION_MEDIAN_RESIDUAL_RATIO = 1.25
_MINIMUM_VIRTUAL_INSTANCE_OVERLAP_PIXELS = 12
_MINIMUM_VIRTUAL_INSTANCE_OVERLAP_ROWS = 8
_MINIMUM_VIRTUAL_INSTANCE_OVERLAP_FRACTION = 0.05


def extract_signed_occlusion_foreground_components(
    reprojection: DirectedReprojection,
) -> SignedOcclusionForegroundComponents:
    """Return material direct foreground-occlusion components in one source.

    A component is not inferred from nearby same-layer support.  Every output
    sample itself must be a z-buffer-visible source point which is in front of
    valid target depth by at least one formal depth tolerance.  The 3x3 eroded
    core is a quality proof only; the un-eroded direct observation remains the
    eventual owner-only label so the seam guard is never widened by matching.
    """

    fields = (
        reprojection.source_valid,
        reprojection.in_target_bounds,
        reprojection.zbuffer_visible,
        reprojection.target_valid,
        reprojection.source_foreground,
        reprojection.labels,
        reprojection.target_x,
        reprojection.target_y,
        reprojection.depth_residual_ratio,
    )
    shape = np.asarray(reprojection.source_valid, dtype=bool).shape
    if any(np.asarray(field).shape != shape for field in fields):
        raise ValueError("Signed foreground reprojection fields must share one source shape")
    residual_ratio = np.asarray(reprojection.depth_residual_ratio, dtype=np.float64)
    direct_foreground = (
        np.asarray(reprojection.source_valid, dtype=bool)
        & np.asarray(reprojection.in_target_bounds, dtype=bool)
        & np.asarray(reprojection.zbuffer_visible, dtype=bool)
        & np.asarray(reprojection.target_valid, dtype=bool)
        & np.asarray(reprojection.source_foreground, dtype=bool)
        & (np.asarray(reprojection.labels) == int(LayerLabel.FOREGROUND))
        & np.isfinite(reprojection.target_x)
        & np.isfinite(reprojection.target_y)
        & np.isfinite(residual_ratio)
        & (residual_ratio >= 1.0)
    )
    component_count, provisional, stats, _ = cv2.connectedComponentsWithStats(
        direct_foreground.astype(np.uint8), connectivity=8
    )
    labels = np.zeros(shape, dtype=np.int32)
    areas: dict[int, int] = {}
    eroded_core_count = 0
    rejected = 0
    next_label = 1
    kernel = np.ones((3, 3), dtype=np.uint8)
    for provisional_label in range(1, int(component_count)):
        component = provisional == provisional_label
        area = int(stats[provisional_label, cv2.CC_STAT_AREA])
        if area < _MINIMUM_SIGNED_OCCLUSION_COMPONENT_PIXELS:
            rejected += 1
            continue
        core = cv2.erode(component.astype(np.uint8), kernel, iterations=1) > 0
        core_count = int(np.count_nonzero(core))
        if core_count < _MINIMUM_SIGNED_OCCLUSION_ERODED_CORE_PIXELS:
            rejected += 1
            continue
        component_residual = residual_ratio[component]
        component_residual = component_residual[np.isfinite(component_residual)]
        if (
            not component_residual.size
            or float(np.median(component_residual))
            < _MINIMUM_SIGNED_OCCLUSION_MEDIAN_RESIDUAL_RATIO
        ):
            rejected += 1
            continue
        labels[component] = next_label
        areas[next_label] = area
        eroded_core_count += core_count
        next_label += 1
    return SignedOcclusionForegroundComponents(
        labels=labels,
        component_pixel_counts=areas,
        signed_occlusion_pixel_count=int(np.count_nonzero(direct_foreground)),
        eroded_core_pixel_count=eroded_core_count,
        rejected_component_count=rejected,
    )


def _positive_label_pixel_counts(labels: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(labels, dtype=np.int32), return_counts=True)
    return {
        int(value): int(count)
        for value, count in zip(values, counts, strict=True)
        if int(value) > 0
    }


def _unique_dominant_virtual_partner(
    candidates: Sequence[tuple[int, int]],
) -> int | None:
    """Return one component only when it dominates every alternate overlap."""

    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda item: (-int(item[1]), int(item[0])))
    if len(ranked) > 1 and int(ranked[0][1]) < 2 * int(ranked[1][1]):
        return None
    return int(ranked[0][0])


def match_bidirectional_signed_occlusion_instances(
    first_components: SignedOcclusionForegroundComponents,
    second_components: SignedOcclusionForegroundComponents,
    first_virtual_component_labels: np.ndarray,
    second_virtual_component_labels: np.ndarray,
    *,
    minimum_overlap_pixels: int = _MINIMUM_VIRTUAL_INSTANCE_OVERLAP_PIXELS,
    minimum_overlap_rows: int = _MINIMUM_VIRTUAL_INSTANCE_OVERLAP_ROWS,
    minimum_overlap_fraction: float = _MINIMUM_VIRTUAL_INSTANCE_OVERLAP_FRACTION,
) -> BidirectionalForegroundInstances:
    """Associate only uniquely overlapping direct foreground observations.

    The two directions are deliberately *not* joined through a depth-consistent
    correspondence: foreground occlusion and mutual same-layer evidence are
    semantically disjoint.  Instead, each direction first proves its own
    signed foreground component, then the calibrated 96--160 px virtual
    corridor must show one exact, dominant component overlap.  No dilation or
    optical-flow displacement participates in this identity proof.
    """

    if int(minimum_overlap_pixels) < 1 or int(minimum_overlap_rows) < 1:
        raise ValueError("Signed-occlusion overlap gates must be positive")
    if not 0.0 < float(minimum_overlap_fraction) <= 1.0:
        raise ValueError("Signed-occlusion overlap fraction must be in (0, 1]")

    first_labels = np.asarray(first_components.labels, dtype=np.int32)
    second_labels = np.asarray(second_components.labels, dtype=np.int32)
    if first_labels.shape != second_labels.shape:
        raise ValueError("Adjacent signed-occlusion labels must share one native shape")
    first_virtual = np.asarray(first_virtual_component_labels, dtype=np.int32)
    second_virtual = np.asarray(second_virtual_component_labels, dtype=np.int32)
    if (
        first_virtual.ndim != 2
        or second_virtual.ndim != 2
        or first_virtual.shape != second_virtual.shape
    ):
        raise ValueError("Signed-occlusion virtual labels must share one corridor shape")
    if set(np.unique(first_virtual).tolist()) - {0, *first_components.component_pixel_counts}:
        raise ValueError("First virtual labels contain an unknown signed foreground component")
    if set(np.unique(second_virtual).tolist()) - {0, *second_components.component_pixel_counts}:
        raise ValueError("Second virtual labels contain an unknown signed foreground component")

    first_virtual_counts = _positive_label_pixel_counts(first_virtual)
    second_virtual_counts = _positive_label_pixel_counts(second_virtual)
    overlap = (first_virtual > 0) & (second_virtual > 0)
    overlap_counts: dict[tuple[int, int], int] = {}
    overlap_rows: dict[tuple[int, int], int] = {}
    if np.any(overlap):
        rows, _columns = np.nonzero(overlap)
        pairs = np.column_stack((first_virtual[overlap], second_virtual[overlap]))
        unique_pairs, inverse, counts = np.unique(
            pairs, axis=0, return_inverse=True, return_counts=True
        )
        for index, (pair, count) in enumerate(zip(unique_pairs, counts, strict=True)):
            key = (int(pair[0]), int(pair[1]))
            overlap_counts[key] = int(count)
            overlap_rows[key] = int(np.unique(rows[inverse == index]).size)

    first_partners: dict[int, list[tuple[int, int]]] = {
        label: [] for label in first_components.component_pixel_counts
    }
    second_partners: dict[int, list[tuple[int, int]]] = {
        label: [] for label in second_components.component_pixel_counts
    }
    for (first_label, second_label), count in overlap_counts.items():
        first_partners[first_label].append((second_label, count))
        second_partners[second_label].append((first_label, count))
    first_dominant = {
        label: _unique_dominant_virtual_partner(partners)
        for label, partners in first_partners.items()
    }
    second_dominant = {
        label: _unique_dominant_virtual_partner(partners)
        for label, partners in second_partners.items()
    }

    first_instance_labels = np.zeros_like(first_labels, dtype=np.int32)
    second_instance_labels = np.zeros_like(second_labels, dtype=np.int32)
    matches: list[ForegroundInstanceMatch] = []
    next_instance_id = 1
    for (first_label, second_label), overlap_count in sorted(overlap_counts.items()):
        first_virtual_count = first_virtual_counts.get(first_label, 0)
        second_virtual_count = second_virtual_counts.get(second_label, 0)
        minimum_overlap = max(
            int(minimum_overlap_pixels),
            int(
                math.ceil(
                    float(minimum_overlap_fraction)
                    * min(first_virtual_count, second_virtual_count)
                )
            ),
        )
        if (
            overlap_count < minimum_overlap
            or overlap_rows[(first_label, second_label)]
            < int(minimum_overlap_rows)
            or first_dominant.get(first_label) != second_label
            or second_dominant.get(second_label) != first_label
        ):
            continue
        first_instance_labels[first_labels == first_label] = next_instance_id
        second_instance_labels[second_labels == second_label] = next_instance_id
        matches.append(
            ForegroundInstanceMatch(
                instance_id=next_instance_id,
                first_component_label=first_label,
                second_component_label=second_label,
                first_component_pixel_count=first_components.component_pixel_counts[
                    first_label
                ],
                second_component_pixel_count=second_components.component_pixel_counts[
                    second_label
                ],
                first_virtual_pixel_count=first_virtual_count,
                second_virtual_pixel_count=second_virtual_count,
                virtual_overlap_pixel_count=overlap_count,
                virtual_overlap_row_count=overlap_rows[(first_label, second_label)],
            )
        )
        next_instance_id += 1
    rejected_component_count = (
        len(first_components.component_pixel_counts)
        + len(second_components.component_pixel_counts)
        - 2 * len(matches)
    )
    return BidirectionalForegroundInstances(
        first_instance_labels=first_instance_labels,
        second_instance_labels=second_instance_labels,
        matches=tuple(matches),
        rejected_component_count=rejected_component_count,
        first_signed_occlusion_pixel_count=(
            first_components.signed_occlusion_pixel_count
        ),
        second_signed_occlusion_pixel_count=(
            second_components.signed_occlusion_pixel_count
        ),
        first_eroded_occlusion_core_pixel_count=(
            first_components.eroded_core_pixel_count
        ),
        second_eroded_occlusion_core_pixel_count=(
            second_components.eroded_core_pixel_count
        ),
    )


def mutually_consistent_correspondences(
    reprojection: DirectedReprojection,
    *,
    exclude_protected: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source/target pixel pairs suitable for one same-layer warp fit.

    The pairs remain in each source's raw aligned-colour coordinate system.  A
    future strip renderer converts them into its local virtual coordinates
    before calling :func:`fit_local_inverse_warp`.
    """

    mask = np.asarray(reprojection.mutual_consistent, dtype=bool)
    if exclude_protected:
        mask &= ~reprojection.protected_mask
    y, x = np.nonzero(mask)
    source = np.column_stack((x, y)).astype(np.float64, copy=False)
    target = np.column_stack(
        (reprojection.target_x[y, x], reprojection.target_y[y, x])
    ).astype(np.float64, copy=False)
    return source, target


@dataclass(frozen=True)
class TileBounds:
    """Half-open local virtual tile bounds for a geometry-assisted inverse warp."""

    x0: float
    y0: float
    x1: float
    y1: float

    def validate(self) -> None:
        values = np.asarray((self.x0, self.y0, self.x1, self.y1), dtype=np.float64)
        if not np.isfinite(values).all() or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("TileBounds must be finite with x1>x0 and y1>y0")

    def contains(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float64)
        return (
            (points[:, 0] >= self.x0)
            & (points[:, 0] <= self.x1)
            & (points[:, 1] >= self.y0)
            & (points[:, 1] <= self.y1)
        )


@dataclass(frozen=True)
class LocalWarpConfig:
    """Safety limits for one layer-local affine inverse sampling warp."""

    minimum_correspondences: int = 24
    held_out_fraction: float = 0.20
    held_out_seed: int = 20301117
    robust_iterations: int = 4
    huber_delta_pixels: float = 1.0
    boundary_taper_pixels: float = 8.0
    maximum_displacement_pixels: float = 8.0
    minimum_jacobian_determinant: float = 0.70
    maximum_jacobian_determinant: float = 1.30
    maximum_jacobian_condition: float = 2.0
    maximum_held_out_error_pixels: float = 0.75
    maximum_held_out_maximum_error_pixels: float = 2.0
    minimum_held_out_improvement_pixels: float = 0.05
    minimum_held_out_improvement_ratio: float = 0.30

    def validate(self) -> None:
        if not isinstance(self.minimum_correspondences, (int, np.integer)) or (
            self.minimum_correspondences < 6
        ):
            raise ValueError("minimum_correspondences must be an integer of at least six")
        if not 0.05 <= float(self.held_out_fraction) < 0.5:
            raise ValueError("held_out_fraction must be in [0.05, 0.5)")
        if not isinstance(self.held_out_seed, (int, np.integer)):
            raise ValueError("held_out_seed must be an integer")
        if not isinstance(self.robust_iterations, (int, np.integer)) or self.robust_iterations < 1:
            raise ValueError("robust_iterations must be a positive integer")
        for name in (
            "huber_delta_pixels",
            "boundary_taper_pixels",
            "maximum_displacement_pixels",
            "minimum_jacobian_determinant",
            "maximum_jacobian_determinant",
            "maximum_jacobian_condition",
            "maximum_held_out_error_pixels",
            "maximum_held_out_maximum_error_pixels",
            "minimum_held_out_improvement_pixels",
            "minimum_held_out_improvement_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_jacobian_determinant >= self.maximum_jacobian_determinant:
            raise ValueError("Jacobian determinant range must be ordered")
        if self.maximum_held_out_error_pixels > self.maximum_held_out_maximum_error_pixels:
            raise ValueError(
                "maximum_held_out_error_pixels cannot exceed maximum_held_out_maximum_error_pixels"
            )
        if not 0.0 < float(self.minimum_held_out_improvement_ratio) <= 1.0:
            raise ValueError("minimum_held_out_improvement_ratio must be in (0, 1]")


@dataclass(frozen=True)
class LocalWarpAudit:
    """Audit trail for accepting or rejecting a local inverse warp."""

    accepted: bool
    reason: str
    correspondence_count: int
    training_count: int
    held_out_count: int
    train_error_p95_before_pixels: float | None
    train_error_p95_after_pixels: float | None
    held_out_error_p95_before_pixels: float | None
    held_out_error_p95_after_pixels: float | None
    held_out_error_max_before_pixels: float | None
    held_out_error_max_after_pixels: float | None
    maximum_displacement_pixels: float | None
    displacement_p95_pixels: float | None
    jacobian_determinant: float | None
    jacobian_condition: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_unit": "full_resolution_pixels",
            "accepted": self.accepted,
            "reason": self.reason,
            "correspondence_count": self.correspondence_count,
            "training_count": self.training_count,
            "held_out_count": self.held_out_count,
            "train_error_p95_before_pixels": self.train_error_p95_before_pixels,
            "train_error_p95_after_pixels": self.train_error_p95_after_pixels,
            "held_out_error_p95_before_pixels": self.held_out_error_p95_before_pixels,
            "held_out_error_p95_after_pixels": self.held_out_error_p95_after_pixels,
            "held_out_error_max_before_pixels": self.held_out_error_max_before_pixels,
            "held_out_error_max_after_pixels": self.held_out_error_max_after_pixels,
            "maximum_displacement_pixels": self.maximum_displacement_pixels,
            "displacement_p95_pixels": self.displacement_p95_pixels,
            "jacobian_determinant": self.jacobian_determinant,
            "jacobian_condition": self.jacobian_condition,
        }


@dataclass(frozen=True)
class LocalInverseWarp:
    """Bounded affine inverse map tapered exactly to identity at tile edges.

    ``linear`` and ``offset`` map a requested output coordinate ``q`` to a
    source sampling coordinate.  They do not describe a camera pose.  The
    taper prevents an accepted local correction from changing any coordinate at
    or outside the tile boundary.
    """

    bounds: TileBounds
    linear: np.ndarray
    offset: np.ndarray
    boundary_taper_pixels: float

    def inverse_coordinates(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map local output coordinates to source sampling coordinates."""

        self.bounds.validate()
        x_values, y_values = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        points = np.stack((x_values, y_values), axis=-1)
        affine = points @ np.asarray(self.linear, dtype=np.float64).T + np.asarray(
            self.offset, dtype=np.float64
        )
        taper = _tile_taper(points, self.bounds, self.boundary_taper_pixels)
        mapped = points + taper[..., None] * (affine - points)
        return mapped[..., 0], mapped[..., 1]


@dataclass(frozen=True)
class LocalWarpFitResult:
    """Accepted warp or fail-closed audit with ``warp`` set to ``None``."""

    warp: LocalInverseWarp | None
    audit: LocalWarpAudit


def _tile_taper(points: np.ndarray, bounds: TileBounds, radius: float) -> np.ndarray:
    """C1 taper that is exactly zero on/outside every tile boundary."""

    if radius <= 0.0:
        inside = bounds.contains(np.asarray(points, dtype=np.float64).reshape(-1, 2))
        return inside.reshape(np.asarray(points).shape[:-1]).astype(np.float64)
    x, y = points[..., 0], points[..., 1]
    distance = np.minimum.reduce((x - bounds.x0, bounds.x1 - x, y - bounds.y0, bounds.y1 - y))
    normalised = np.clip(distance / radius, 0.0, 1.0)
    return normalised * normalised * (3.0 - 2.0 * normalised)


def _p95(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, 95.0)) if finite.size else None


def _empty_warp_audit(reason: str, count: int = 0) -> LocalWarpFitResult:
    return LocalWarpFitResult(
        warp=None,
        audit=LocalWarpAudit(
            accepted=False,
            reason=reason,
            correspondence_count=count,
            training_count=0,
            held_out_count=0,
            train_error_p95_before_pixels=None,
            train_error_p95_after_pixels=None,
            held_out_error_p95_before_pixels=None,
            held_out_error_p95_after_pixels=None,
            held_out_error_max_before_pixels=None,
            held_out_error_max_after_pixels=None,
            maximum_displacement_pixels=None,
            displacement_p95_pixels=None,
            jacobian_determinant=None,
            jacobian_condition=None,
        ),
    )


def _fit_affine_irls(
    output_points: np.ndarray,
    sample_points: np.ndarray,
    config: LocalWarpConfig,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Robustly fit ``sample = output @ linear.T + offset``."""

    design = np.column_stack((output_points, np.ones(len(output_points), dtype=np.float64)))
    weights = np.ones(len(output_points), dtype=np.float64)
    coefficients: np.ndarray | None = None
    condition = math.inf
    for _ in range(int(config.robust_iterations)):
        sqrt_weights = np.sqrt(weights)[:, None]
        weighted_design = design * sqrt_weights
        normal = weighted_design.T @ weighted_design
        condition = float(np.linalg.cond(normal))
        if not math.isfinite(condition) or condition > 1e10:
            return None
        try:
            coefficients, _, rank, _ = np.linalg.lstsq(
                weighted_design, sample_points * sqrt_weights, rcond=None
            )
        except np.linalg.LinAlgError:
            return None
        if rank != 3 or not np.isfinite(coefficients).all():
            return None
        predicted = design @ coefficients
        residual = np.linalg.norm(predicted - sample_points, axis=1)
        scale = max(0.05, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        threshold = max(float(config.huber_delta_pixels), 1.5 * scale)
        weights = np.minimum(1.0, threshold / np.maximum(residual, 1e-9))
    assert coefficients is not None
    return coefficients[:2, :].T, coefficients[2, :], condition


def _warp_grid_displacements(
    warp: LocalInverseWarp, bounds: TileBounds
) -> np.ndarray:
    """Sample enough of a tile to audit a tapered affine displacement."""

    xs = np.linspace(bounds.x0, bounds.x1, 9, dtype=np.float64)
    ys = np.linspace(bounds.y0, bounds.y1, 9, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    mapped_x, mapped_y = warp.inverse_coordinates(grid_x, grid_y)
    return np.hypot(mapped_x - grid_x, mapped_y - grid_y).reshape(-1)


def fit_local_inverse_warp(
    output_points_xy: np.ndarray,
    source_sample_points_xy: np.ndarray,
    bounds: TileBounds,
    *,
    config: LocalWarpConfig | None = None,
) -> LocalWarpFitResult:
    """Fit and audit one same-layer local inverse affine warp.

    ``output_points_xy`` are coordinates in a caller's canonical local tile;
    ``source_sample_points_xy`` are the coordinates at which the source must be
    sampled to depict those same 3-D points.  Geometry callers should feed only
    mutually depth-consistent, non-guarded points from one surface layer.

    The result is fail-closed: any insufficient support, unstable affine,
    excessive displacement, fold-like Jacobian, or lack of held-out improvement
    returns ``warp=None``.
    """

    settings = LocalWarpConfig() if config is None else config
    if not isinstance(settings, LocalWarpConfig):
        raise TypeError("config must be a LocalWarpConfig")
    settings.validate()
    bounds.validate()
    output = np.asarray(output_points_xy, dtype=np.float64)
    source = np.asarray(source_sample_points_xy, dtype=np.float64)
    if output.ndim != 2 or source.ndim != 2 or output.shape[1:] != (2,) or source.shape != output.shape:
        raise ValueError("Warp correspondences must be matching N x 2 arrays")
    finite = np.isfinite(output).all(axis=1) & np.isfinite(source).all(axis=1)
    inside = bounds.contains(output) if output.size else np.empty(0, dtype=bool)
    output = output[finite & inside]
    source = source[finite & inside]
    count = len(output)
    if count < int(settings.minimum_correspondences):
        return _empty_warp_audit("insufficient_same_layer_correspondences", count)
    # The held-out subset is deterministic but shuffled, avoiding a scan-order
    # bias that would otherwise reserve only one tile side for validation.
    rng = np.random.default_rng(int(settings.held_out_seed))
    order = rng.permutation(count)
    held_out_count = max(1, int(round(count * float(settings.held_out_fraction))))
    held_out_indices = order[:held_out_count]
    training_indices = order[held_out_count:]
    if len(training_indices) < int(settings.minimum_correspondences):
        return _empty_warp_audit("insufficient_training_correspondences", count)
    fitted = _fit_affine_irls(output[training_indices], source[training_indices], settings)
    if fitted is None:
        return _empty_warp_audit("ill_conditioned_or_rank_deficient_affine", count)
    linear, offset, condition = fitted
    determinant = float(np.linalg.det(linear))
    linear_condition = float(np.linalg.cond(linear))
    candidate = LocalInverseWarp(
        bounds=bounds,
        linear=np.asarray(linear, dtype=np.float64),
        offset=np.asarray(offset, dtype=np.float64),
        boundary_taper_pixels=float(settings.boundary_taper_pixels),
    )
    train_x, train_y = candidate.inverse_coordinates(
        output[training_indices, 0], output[training_indices, 1]
    )
    held_x, held_y = candidate.inverse_coordinates(
        output[held_out_indices, 0], output[held_out_indices, 1]
    )
    train_after = np.hypot(
        train_x - source[training_indices, 0], train_y - source[training_indices, 1]
    )
    held_after = np.hypot(
        held_x - source[held_out_indices, 0], held_y - source[held_out_indices, 1]
    )
    train_before = np.linalg.norm(output[training_indices] - source[training_indices], axis=1)
    held_before = np.linalg.norm(output[held_out_indices] - source[held_out_indices], axis=1)
    displacement = _warp_grid_displacements(candidate, bounds)
    maximum_displacement = float(np.max(displacement))
    displacement_p95 = _p95(displacement)
    audit = LocalWarpAudit(
        accepted=False,
        reason="rejected",
        correspondence_count=count,
        training_count=len(training_indices),
        held_out_count=len(held_out_indices),
        train_error_p95_before_pixels=_p95(train_before),
        train_error_p95_after_pixels=_p95(train_after),
        held_out_error_p95_before_pixels=_p95(held_before),
        held_out_error_p95_after_pixels=_p95(held_after),
        held_out_error_max_before_pixels=(
            float(np.max(held_before)) if held_before.size else None
        ),
        held_out_error_max_after_pixels=(
            float(np.max(held_after)) if held_after.size else None
        ),
        maximum_displacement_pixels=maximum_displacement,
        displacement_p95_pixels=displacement_p95,
        jacobian_determinant=determinant,
        jacobian_condition=linear_condition,
    )
    if not math.isfinite(determinant) or not math.isfinite(linear_condition):
        return LocalWarpFitResult(None, replace(audit, reason="non_finite_jacobian"))
    if not (
        float(settings.minimum_jacobian_determinant)
        <= determinant
        <= float(settings.maximum_jacobian_determinant)
    ):
        return LocalWarpFitResult(None, replace(audit, reason="jacobian_determinant_out_of_bounds"))
    if linear_condition > float(settings.maximum_jacobian_condition):
        return LocalWarpFitResult(None, replace(audit, reason="jacobian_condition_out_of_bounds"))
    if maximum_displacement > float(settings.maximum_displacement_pixels):
        return LocalWarpFitResult(None, replace(audit, reason="maximum_displacement_exceeded"))
    held_before_p95 = audit.held_out_error_p95_before_pixels
    held_after_p95 = audit.held_out_error_p95_after_pixels
    if held_before_p95 is None or held_after_p95 is None:
        return LocalWarpFitResult(None, replace(audit, reason="no_held_out_support"))
    if held_after_p95 > float(settings.maximum_held_out_error_pixels):
        return LocalWarpFitResult(None, replace(audit, reason="held_out_error_exceeded"))
    held_after_max = audit.held_out_error_max_after_pixels
    if (
        held_after_max is None
        or held_after_max > float(settings.maximum_held_out_maximum_error_pixels)
    ):
        return LocalWarpFitResult(
            None, replace(audit, reason="held_out_maximum_error_exceeded")
        )
    if held_before_p95 - held_after_p95 < float(settings.minimum_held_out_improvement_pixels):
        return LocalWarpFitResult(None, replace(audit, reason="no_material_held_out_improvement"))
    improvement_ratio = (held_before_p95 - held_after_p95) / max(held_before_p95, 1e-9)
    if improvement_ratio < float(settings.minimum_held_out_improvement_ratio):
        return LocalWarpFitResult(
            None, replace(audit, reason="held_out_improvement_ratio_too_small")
        )
    return LocalWarpFitResult(candidate, replace(audit, accepted=True, reason="accepted"))


@dataclass(frozen=True)
class LocalMeshWarpConfig:
    """Limits for a regular 16/32 px same-layer inverse deformation mesh.

    The mesh is pinned to identity at the analysis-tile boundary.  Inside the
    tile, every protected/unknown sample remains pointwise identity at both
    its output and mapped source coordinate; inactive cells are never
    evaluated.  This preserves hard depth-layer ownership without throwing
    away all mesh degrees of freedom next to a conservative 8--12 px guard.
    """

    grid_spacing_pixels: int = 16
    minimum_correspondences: int = 48
    minimum_samples_per_active_cell: int = 4
    # A one-cell field has too little spatial support to distinguish a local
    # correction from an arbitrary residual.  The formal seam contract
    # requires at least four same-layer cells, even for diagnostic runs.
    minimum_active_cells: int = 4
    held_out_fraction: float = 0.20
    held_out_seed: int = 20301117
    robust_iterations: int = 4
    huber_delta_pixels: float = 1.0
    smoothness_weight: float = 0.25
    identity_weight: float = 0.02
    # Sparse invalid depth samples can fragment an otherwise safe wall into
    # sub-cell holes.  The evaluated warp checks the exact same-layer mask at
    # both output and source coordinates, so protected samples remain identity
    # even when a predominantly safe cell is fitted.
    minimum_active_fraction: float = 0.50
    maximum_fit_correspondences: int = 4096
    maximum_displacement_pixels: float = 8.0
    minimum_jacobian_determinant: float = 0.70
    maximum_jacobian_determinant: float = 1.30
    maximum_jacobian_condition: float = 2.0
    # A fitted same-layer mesh may improve correspondence error while bending
    # a doorway or skirting-board edge.  Measure the maximum departure of
    # sampled horizontal, vertical and cell-diagonal strokes from their mapped
    # chord and keep it at or below the formal one-pixel limit.
    maximum_straight_line_deviation_pixels: float = 1.0
    # Formal defaults intentionally match the renderer's closed envelope.
    maximum_held_out_error_pixels: float = 0.75
    maximum_held_out_maximum_error_pixels: float = 2.0
    minimum_held_out_improvement_pixels: float = 0.05
    minimum_held_out_improvement_ratio: float = 0.30

    def validate(self) -> None:
        if int(self.grid_spacing_pixels) not in {16, 32}:
            raise ValueError("grid_spacing_pixels must be exactly 16 or 32")
        for name, minimum in (
            ("minimum_correspondences", 12),
            ("minimum_samples_per_active_cell", 1),
            ("minimum_active_cells", 4),
            ("robust_iterations", 1),
            ("maximum_fit_correspondences", 48),
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if not isinstance(self.held_out_seed, (int, np.integer)):
            raise ValueError("held_out_seed must be an integer")
        if not 0.05 <= float(self.held_out_fraction) < 0.5:
            raise ValueError("held_out_fraction must be in [0.05, 0.5)")
        if not 0.50 <= float(self.minimum_active_fraction) <= 1.0:
            raise ValueError("minimum_active_fraction must be in [0.50, 1]")
        for name in (
            "huber_delta_pixels",
            "smoothness_weight",
            "identity_weight",
            "maximum_displacement_pixels",
            "minimum_jacobian_determinant",
            "maximum_jacobian_determinant",
            "maximum_jacobian_condition",
            "maximum_straight_line_deviation_pixels",
            "maximum_held_out_error_pixels",
            "maximum_held_out_maximum_error_pixels",
            "minimum_held_out_improvement_pixels",
            "minimum_held_out_improvement_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_jacobian_determinant >= self.maximum_jacobian_determinant:
            raise ValueError("Jacobian determinant range must be ordered")
        if float(self.maximum_straight_line_deviation_pixels) > 1.0:
            raise ValueError(
                "maximum_straight_line_deviation_pixels cannot exceed 1"
            )
        if self.maximum_held_out_error_pixels > self.maximum_held_out_maximum_error_pixels:
            raise ValueError(
                "maximum_held_out_error_pixels cannot exceed maximum_held_out_maximum_error_pixels"
            )
        if not 0.0 < float(self.minimum_held_out_improvement_ratio) <= 1.0:
            raise ValueError("minimum_held_out_improvement_ratio must be in (0, 1]")


@dataclass(frozen=True)
class LocalMeshWarpAudit:
    """Auditable acceptance decision for a regular local inverse mesh."""

    accepted: bool
    reason: str
    correspondence_count: int
    training_count: int
    held_out_count: int
    active_cell_count: int
    largest_connected_active_cell_count: int
    free_node_count: int
    train_error_p95_before_pixels: float | None
    train_error_p95_after_pixels: float | None
    held_out_error_p95_before_pixels: float | None
    held_out_error_p95_after_pixels: float | None
    held_out_error_max_before_pixels: float | None
    held_out_error_max_after_pixels: float | None
    maximum_displacement_pixels: float | None
    displacement_p95_pixels: float | None
    minimum_jacobian_determinant: float | None
    maximum_jacobian_determinant: float | None
    maximum_jacobian_condition: float | None
    maximum_straight_line_deviation_pixels: float | None
    boundary_identity_maximum_error_pixels: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "metric_unit": "full_resolution_pixels",
            "accepted": self.accepted,
            "reason": self.reason,
            "correspondence_count": self.correspondence_count,
            "training_count": self.training_count,
            "held_out_count": self.held_out_count,
            "active_cell_count": self.active_cell_count,
            "largest_connected_active_cell_count": self.largest_connected_active_cell_count,
            "free_node_count": self.free_node_count,
            "train_error_p95_before_pixels": self.train_error_p95_before_pixels,
            "train_error_p95_after_pixels": self.train_error_p95_after_pixels,
            "held_out_error_p95_before_pixels": self.held_out_error_p95_before_pixels,
            "held_out_error_p95_after_pixels": self.held_out_error_p95_after_pixels,
            "held_out_error_max_before_pixels": self.held_out_error_max_before_pixels,
            "held_out_error_max_after_pixels": self.held_out_error_max_after_pixels,
            "maximum_displacement_pixels": self.maximum_displacement_pixels,
            "displacement_p95_pixels": self.displacement_p95_pixels,
            "minimum_jacobian_determinant": self.minimum_jacobian_determinant,
            "maximum_jacobian_determinant": self.maximum_jacobian_determinant,
            "maximum_jacobian_condition": self.maximum_jacobian_condition,
            "maximum_straight_line_deviation_pixels": self.maximum_straight_line_deviation_pixels,
            "straight_line_audit_policy": _MESH_STRAIGHT_LINE_AUDIT_POLICY,
            "boundary_identity_maximum_error_pixels": self.boundary_identity_maximum_error_pixels,
        }


@dataclass(frozen=True)
class LocalMeshInverseWarp:
    """Piecewise-bilinear inverse mesh with pointwise protected identity."""

    bounds: TileBounds
    grid_x: np.ndarray
    grid_y: np.ndarray
    inverse_dx: np.ndarray
    inverse_dy: np.ndarray
    active_cells: np.ndarray
    same_layer_mask: np.ndarray | None = field(default=None, repr=False)
    same_layer_origin_xy: tuple[float, float] | None = None

    def inverse_coordinates(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return inverse source coordinates, preserving inactive/boundary identity."""

        self.bounds.validate()
        x_values, y_values = np.broadcast_arrays(
            np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        )
        points = np.stack((x_values, y_values), axis=-1)
        flat = points.reshape(-1, 2)
        dx, dy, active = _evaluate_mesh_displacement(
            flat,
            self.bounds,
            np.asarray(self.grid_x, dtype=np.float64),
            np.asarray(self.grid_y, dtype=np.float64),
            np.asarray(self.inverse_dx, dtype=np.float64),
            np.asarray(self.inverse_dy, dtype=np.float64),
            np.asarray(self.active_cells, dtype=bool),
        )
        mapped = flat.copy()
        apply = active.copy()
        if self.same_layer_mask is not None:
            mask, origin = _coerce_same_layer_mask(
                self.same_layer_mask, self.bounds, self.same_layer_origin_xy
            )
            # A protected output point remains exactly identity even when its
            # surrounding 16/32 px cell has enough safe support to fit.
            apply &= _mask_at_points(mask, origin, flat)
        mapped[apply, 0] += dx[apply]
        mapped[apply, 1] += dy[apply]
        if self.same_layer_mask is not None and np.any(apply):
            mask, origin = _coerce_same_layer_mask(
                self.same_layer_mask, self.bounds, self.same_layer_origin_xy
            )
            # Likewise, never sample through a protected/unknown source-side
            # virtual coordinate.  The caller's hard-owner seam takes over.
            target_safe = _mask_at_points(mask, origin, mapped)
            rejected = apply & ~target_safe
            mapped[rejected] = flat[rejected]
        mapped = mapped.reshape(points.shape)
        return mapped[..., 0], mapped[..., 1]

    def inverse_virtual_coordinates(
        self, x: np.ndarray | float, y: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alias compatible with the existing composite inverse-map protocol."""

        return self.inverse_coordinates(x, y)


@dataclass(frozen=True)
class ActiveMeshForwardInverseResult:
    """Raw forward-inverse solutions used only by visual straight-line audit.

    ``LocalMeshInverseWarp`` maps final virtual output coordinates ``q`` to a
    second-source sample coordinate ``p=M(q)``.  The normal renderer must use
    its protective identity fallback, but that fallback would conceal a fold
    or an unresolved line sample during validation.  This result therefore
    exposes only explicitly active, same-layer, unique raw solutions of
    ``M(q)=p``.  It never participates in output sampling.
    """

    output_points_xy: np.ndarray = field(repr=False)
    valid_mask: np.ndarray = field(repr=False)
    residual_pixels: np.ndarray = field(repr=False)
    candidate_cell_counts: np.ndarray = field(repr=False)
    solution_counts: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class LocalMeshWarpFitResult:
    """An accepted mesh or a fail-closed mesh audit with ``warp=None``."""

    warp: LocalMeshInverseWarp | None
    audit: LocalMeshWarpAudit


def _mesh_axis(start: float, stop: float, spacing: int) -> np.ndarray:
    values = list(np.arange(start, stop, float(spacing), dtype=np.float64))
    if not values or not math.isclose(values[0], start, abs_tol=1e-9):
        values.insert(0, float(start))
    if not math.isclose(values[-1], stop, abs_tol=1e-9):
        values.append(float(stop))
    return np.asarray(values, dtype=np.float64)


def _coerce_same_layer_mask(
    same_layer_mask: np.ndarray | None,
    bounds: TileBounds,
    origin_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Validate a tile-aligned same-surface mask without inventing support."""

    origin = (bounds.x0, bounds.y0) if origin_xy is None else origin_xy
    if len(origin) != 2 or not np.isfinite(origin).all():
        raise ValueError("same_layer_origin_xy must contain two finite coordinates")
    if same_layer_mask is None:
        width = max(1, int(math.ceil(bounds.x1 - bounds.x0)) + 1)
        height = max(1, int(math.ceil(bounds.y1 - bounds.y0)) + 1)
        return np.ones((height, width), dtype=bool), (float(origin[0]), float(origin[1]))
    mask = np.asarray(same_layer_mask)
    if mask.ndim != 2 or mask.dtype not in {np.dtype(bool), np.dtype(np.uint8)}:
        raise ValueError("same_layer_mask must be a two-dimensional bool/uint8 mask")
    if not mask.size:
        raise ValueError("same_layer_mask cannot be empty")
    return mask.astype(bool), (float(origin[0]), float(origin[1]))


def _mask_at_points(
    mask: np.ndarray, origin: tuple[float, float], points: np.ndarray
) -> np.ndarray:
    checked = np.asarray(points, dtype=np.float64)
    if checked.ndim != 2 or checked.shape[1] != 2:
        raise ValueError("Mask sample points must have shape (N, 2)")
    result = np.zeros(len(checked), dtype=bool)
    finite = np.isfinite(checked).all(axis=1)
    if not np.any(finite):
        return result
    positions = np.flatnonzero(finite)
    ix = np.rint(checked[positions, 0] - origin[0]).astype(np.int64)
    iy = np.rint(checked[positions, 1] - origin[1]).astype(np.int64)
    inside = (ix >= 0) & (ix < mask.shape[1]) & (iy >= 0) & (iy < mask.shape[0])
    result[positions[inside]] = mask[iy[inside], ix[inside]]
    return result


def _mesh_cell_coordinates(
    points: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find bilinear cell indices and weights for in-bounds tile coordinates."""

    ix = np.searchsorted(grid_x, points[:, 0], side="right") - 1
    iy = np.searchsorted(grid_y, points[:, 1], side="right") - 1
    ix = np.clip(ix, 0, len(grid_x) - 2)
    iy = np.clip(iy, 0, len(grid_y) - 2)
    cell_width = grid_x[ix + 1] - grid_x[ix]
    cell_height = grid_y[iy + 1] - grid_y[iy]
    tx = np.clip((points[:, 0] - grid_x[ix]) / cell_width, 0.0, 1.0)
    ty = np.clip((points[:, 1] - grid_y[iy]) / cell_height, 0.0, 1.0)
    weights = np.column_stack(
        ((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty)
    )
    return ix, iy, tx, ty, weights


def _raw_active_mesh_forward_and_jacobian(
    warp: LocalMeshInverseWarp,
    *,
    row: int,
    column: int,
    point_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate raw ``p=M(q)`` and its analytic Jacobian in one active cell."""

    q = np.asarray(point_xy, dtype=np.float64)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError("Raw mesh forward point must be one finite (x, y) pair")
    x0, x1 = float(warp.grid_x[column]), float(warp.grid_x[column + 1])
    y0, y1 = float(warp.grid_y[row]), float(warp.grid_y[row + 1])
    width, height = x1 - x0, y1 - y0
    if width <= 0.0 or height <= 0.0:
        raise RuntimeError("Raw mesh cell has non-positive dimensions")
    tx = float(np.clip((q[0] - x0) / width, 0.0, 1.0))
    ty = float(np.clip((q[1] - y0) / height, 0.0, 1.0))

    def bilinear(nodes: np.ndarray) -> float:
        n00 = float(nodes[row, column])
        n10 = float(nodes[row, column + 1])
        n01 = float(nodes[row + 1, column])
        n11 = float(nodes[row + 1, column + 1])
        return float(
            (1.0 - tx) * (1.0 - ty) * n00
            + tx * (1.0 - ty) * n10
            + (1.0 - tx) * ty * n01
            + tx * ty * n11
        )

    def derivatives(nodes: np.ndarray) -> tuple[float, float]:
        n00 = float(nodes[row, column])
        n10 = float(nodes[row, column + 1])
        n01 = float(nodes[row + 1, column])
        n11 = float(nodes[row + 1, column + 1])
        derivative_x = ((1.0 - ty) * (n10 - n00) + ty * (n11 - n01)) / width
        derivative_y = ((1.0 - tx) * (n01 - n00) + tx * (n11 - n10)) / height
        return float(derivative_x), float(derivative_y)

    dx = bilinear(np.asarray(warp.inverse_dx, dtype=np.float64))
    dy = bilinear(np.asarray(warp.inverse_dy, dtype=np.float64))
    ddx_dx, ddx_dy = derivatives(np.asarray(warp.inverse_dx, dtype=np.float64))
    ddy_dx, ddy_dy = derivatives(np.asarray(warp.inverse_dy, dtype=np.float64))
    return (
        q + np.asarray((dx, dy), dtype=np.float64),
        np.asarray(
            ((1.0 + ddx_dx, ddx_dy), (ddy_dx, 1.0 + ddy_dy)),
            dtype=np.float64,
        ),
    )


def solve_active_mesh_forward_inverse(
    warp: LocalMeshInverseWarp,
    source_points_xy: np.ndarray,
    *,
    maximum_iterations: int = 8,
    maximum_residual_pixels: float = 0.05,
) -> ActiveMeshForwardInverseResult:
    """Solve ``M(q)=p`` only in unique active same-layer mesh cells.

    This deliberately bypasses :meth:`LocalMeshInverseWarp.inverse_coordinates`
    because its identity fallback is correct for rendering but invalid for a
    geometry audit.  Each candidate cell is selected from its raw forward
    quadrilateral bounding box, then solved by cell-constrained analytic
    Newton iteration.  Missing, non-positive-Jacobian, protected, ambiguous,
    or non-converged samples remain invalid rather than falling back to
    identity.
    """

    points = np.asarray(source_points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Raw mesh forward-inverse input must have shape (N, 2)")
    iterations = int(maximum_iterations)
    tolerance = float(maximum_residual_pixels)
    if iterations <= 0 or not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Raw mesh forward-inverse settings are invalid")
    warp.bounds.validate()
    active_cells = np.asarray(warp.active_cells, dtype=bool)
    expected_shape = (len(warp.grid_y) - 1, len(warp.grid_x) - 1)
    if active_cells.shape != expected_shape:
        raise RuntimeError("Raw mesh active-cell layout is malformed")
    source_safe: np.ndarray | None = None
    same_layer: np.ndarray | None = None
    origin: tuple[float, float] | None = None
    if warp.same_layer_mask is not None:
        same_layer, origin = _coerce_same_layer_mask(
            warp.same_layer_mask, warp.bounds, warp.same_layer_origin_xy
        )
        source_safe = _mask_at_points(same_layer, origin, points)
    output = np.full(points.shape, np.nan, dtype=np.float64)
    valid = np.zeros(len(points), dtype=bool)
    residuals = np.full(len(points), np.nan, dtype=np.float64)
    candidate_counts = np.zeros(len(points), dtype=np.int32)
    solution_counts = np.zeros(len(points), dtype=np.int32)
    active_indices = tuple(zip(*np.nonzero(active_cells), strict=True))
    if not active_indices:
        return ActiveMeshForwardInverseResult(
            output, valid, residuals, candidate_counts, solution_counts
        )
    candidate_bboxes: list[tuple[int, int, float, float, float, float]] = []
    for row, column in active_indices:
        corners = np.asarray(
            (
                (warp.grid_x[column], warp.grid_y[row]),
                (warp.grid_x[column + 1], warp.grid_y[row]),
                (warp.grid_x[column], warp.grid_y[row + 1]),
                (warp.grid_x[column + 1], warp.grid_y[row + 1]),
            ),
            dtype=np.float64,
        )
        mapped = np.asarray(
            [
                _raw_active_mesh_forward_and_jacobian(
                    warp, row=row, column=column, point_xy=corner
                )[0]
                for corner in corners
            ],
            dtype=np.float64,
        )
        if not np.isfinite(mapped).all():
            continue
        candidate_bboxes.append(
            (
                int(row),
                int(column),
                float(np.min(mapped[:, 0])),
                float(np.max(mapped[:, 0])),
                float(np.min(mapped[:, 1])),
                float(np.max(mapped[:, 1])),
            )
        )
    boundary_epsilon = 1e-9
    for point_index, source in enumerate(points):
        if not np.isfinite(source).all() or (
            source_safe is not None and not bool(source_safe[point_index])
        ):
            continue
        solutions: list[tuple[np.ndarray, float]] = []
        for row, column, min_x, max_x, min_y, max_y in candidate_bboxes:
            if (
                source[0] < min_x - tolerance
                or source[0] > max_x + tolerance
                or source[1] < min_y - tolerance
                or source[1] > max_y + tolerance
            ):
                continue
            candidate_counts[point_index] += 1
            x0, x1 = float(warp.grid_x[column]), float(warp.grid_x[column + 1])
            y0, y1 = float(warp.grid_y[row]), float(warp.grid_y[row + 1])
            q = np.clip(source, (x0, y0), (x1, y1)).astype(np.float64)
            converged = False
            for _ in range(iterations):
                mapped, jacobian = _raw_active_mesh_forward_and_jacobian(
                    warp, row=row, column=column, point_xy=q
                )
                difference = mapped - source
                determinant = float(np.linalg.det(jacobian))
                if (
                    not np.isfinite(jacobian).all()
                    or not math.isfinite(determinant)
                    or determinant <= 0.0
                ):
                    break
                residual = float(np.linalg.norm(difference))
                if residual <= tolerance:
                    converged = True
                    break
                try:
                    step = np.linalg.solve(jacobian, difference)
                except np.linalg.LinAlgError:
                    break
                if not np.isfinite(step).all():
                    break
                q = np.clip(q - step, (x0, y0), (x1, y1))
            mapped, jacobian = _raw_active_mesh_forward_and_jacobian(
                warp, row=row, column=column, point_xy=q
            )
            residual = float(np.linalg.norm(mapped - source))
            determinant = float(np.linalg.det(jacobian))
            if not (
                converged
                and math.isfinite(residual)
                and residual <= tolerance
                and math.isfinite(determinant)
                and determinant > 0.0
                and x0 - boundary_epsilon <= q[0] <= x1 + boundary_epsilon
                and y0 - boundary_epsilon <= q[1] <= y1 + boundary_epsilon
            ):
                continue
            if same_layer is not None and origin is not None and not bool(
                _mask_at_points(same_layer, origin, q.reshape(1, 2))[0]
            ):
                continue
            if all(np.linalg.norm(q - prior[0]) > 1e-4 for prior in solutions):
                solutions.append((q, residual))
        solution_counts[point_index] = len(solutions)
        if len(solutions) == 1:
            output[point_index] = solutions[0][0]
            residuals[point_index] = solutions[0][1]
            valid[point_index] = True
    return ActiveMeshForwardInverseResult(
        output,
        valid,
        residuals,
        candidate_counts,
        solution_counts,
    )


def _mesh_active_cells_from_mask(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    mask: np.ndarray,
    origin: tuple[float, float],
    minimum_fraction: float,
) -> np.ndarray:
    active = np.zeros((len(grid_y) - 1, len(grid_x) - 1), dtype=bool)
    for row in range(active.shape[0]):
        for column in range(active.shape[1]):
            x0 = max(0, int(math.ceil(grid_x[column] - origin[0])))
            x1 = min(mask.shape[1] - 1, int(math.floor(grid_x[column + 1] - origin[0])))
            y0 = max(0, int(math.ceil(grid_y[row] - origin[1])))
            y1 = min(mask.shape[0] - 1, int(math.floor(grid_y[row + 1] - origin[1])))
            if x1 < x0 or y1 < y0:
                continue
            active[row, column] = float(np.mean(mask[y0 : y1 + 1, x0 : x1 + 1])) >= minimum_fraction
    return active


def _retain_supported_active_mesh_components(
    active_cells: np.ndarray, minimum_cells: int
) -> tuple[np.ndarray, int]:
    """Keep only 4-connected same-layer mesh components with enough support."""

    active = np.asarray(active_cells, dtype=bool)
    retained = np.zeros_like(active)
    visited = np.zeros_like(active)
    largest = 0
    rows, columns = active.shape
    for start_row, start_column in zip(*np.nonzero(active), strict=True):
        if visited[start_row, start_column]:
            continue
        stack = [(int(start_row), int(start_column))]
        visited[start_row, start_column] = True
        component: list[tuple[int, int]] = []
        while stack:
            row, column = stack.pop()
            component.append((row, column))
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= next_row < rows
                    and 0 <= next_column < columns
                    and active[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
        largest = max(largest, len(component))
        if len(component) >= minimum_cells:
            for row, column in component:
                retained[row, column] = True
    return retained, largest


def _mesh_free_nodes(
    active_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return active/free mesh nodes, pinning only the analysis-tile edge.

    Protected samples are enforced as identity pointwise by
    :meth:`LocalMeshInverseWarp.inverse_coordinates`; pinning every node next
    to a protected guard would turn a fragmented but safely observed wall into
    a near-rigid mesh and defeat the local correction.  Inactive cells remain
    identity because the evaluator never enters them, while all four outer
    tile edges are exactly zero-displacement anchors.
    """

    rows, columns = active_cells.shape
    active_nodes = np.zeros((rows + 1, columns + 1), dtype=bool)
    component_labels = np.zeros_like(active_cells, dtype=np.int32)
    next_component = 0
    for start_row, start_column in zip(*np.nonzero(active_cells), strict=True):
        if component_labels[start_row, start_column]:
            continue
        next_component += 1
        stack = [(int(start_row), int(start_column))]
        component_labels[start_row, start_column] = next_component
        while stack:
            row, column = stack.pop()
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= next_row < rows
                    and 0 <= next_column < columns
                    and active_cells[next_row, next_column]
                    and component_labels[next_row, next_column] == 0
                ):
                    component_labels[next_row, next_column] = next_component
                    stack.append((next_row, next_column))
    node_component = np.zeros((rows + 1, columns + 1), dtype=np.int32)
    shared_between_components = np.zeros_like(active_nodes)
    for row in range(rows):
        for column in range(columns):
            if not active_cells[row, column]:
                continue
            active_nodes[row : row + 2, column : column + 2] = True
            label = component_labels[row, column]
            for node_row, node_column in (
                (row, column),
                (row, column + 1),
                (row + 1, column),
                (row + 1, column + 1),
            ):
                current = node_component[node_row, node_column]
                if current == 0:
                    node_component[node_row, node_column] = label
                elif current != label:
                    shared_between_components[node_row, node_column] = True
    # Two 4-disconnected cells can meet at a single vertex.  Sharing a free
    # vertex would let the least-squares system communicate displacement
    # across an owner-protected/depth-unknown gap, so that vertex is pinned.
    node_component[shared_between_components] = 0
    fixed = np.zeros_like(active_nodes)
    fixed[[0, -1], :] = True
    fixed[:, [0, -1]] = True
    return active_nodes, active_nodes & ~fixed & ~shared_between_components, node_component


def _evaluate_mesh_displacement(
    points: np.ndarray,
    bounds: TileBounds,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    dx_nodes: np.ndarray,
    dy_nodes: np.ndarray,
    active_cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a mesh only in active cells; everywhere else is identity."""

    points = np.asarray(points, dtype=np.float64)
    count = len(points)
    dx = np.zeros(count, dtype=np.float64)
    dy = np.zeros(count, dtype=np.float64)
    inside = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= bounds.x0)
        & (points[:, 0] <= bounds.x1)
        & (points[:, 1] >= bounds.y0)
        & (points[:, 1] <= bounds.y1)
    )
    if not np.any(inside):
        return dx, dy, inside
    positions = np.flatnonzero(inside)
    ix, iy, _, _, weights = _mesh_cell_coordinates(points[positions], grid_x, grid_y)
    active = active_cells[iy, ix]
    active_positions = positions[active]
    if not active_positions.size:
        return dx, dy, np.zeros(count, dtype=bool)
    active_ix, active_iy, _, _, active_weights = _mesh_cell_coordinates(
        points[active_positions], grid_x, grid_y
    )
    dx_values = (
        active_weights[:, 0] * dx_nodes[active_iy, active_ix]
        + active_weights[:, 1] * dx_nodes[active_iy, active_ix + 1]
        + active_weights[:, 2] * dx_nodes[active_iy + 1, active_ix]
        + active_weights[:, 3] * dx_nodes[active_iy + 1, active_ix + 1]
    )
    dy_values = (
        active_weights[:, 0] * dy_nodes[active_iy, active_ix]
        + active_weights[:, 1] * dy_nodes[active_iy, active_ix + 1]
        + active_weights[:, 2] * dy_nodes[active_iy + 1, active_ix]
        + active_weights[:, 3] * dy_nodes[active_iy + 1, active_ix + 1]
    )
    dx[active_positions] = dx_values
    dy[active_positions] = dy_values
    active_mask = np.zeros(count, dtype=bool)
    active_mask[active_positions] = True
    return dx, dy, active_mask


def _empty_mesh_audit(
    reason: str,
    count: int = 0,
    *,
    active_cell_count: int = 0,
    largest_connected_active_cell_count: int = 0,
    free_node_count: int = 0,
) -> LocalMeshWarpFitResult:
    return LocalMeshWarpFitResult(
        warp=None,
        audit=LocalMeshWarpAudit(
            accepted=False,
            reason=reason,
            correspondence_count=count,
            training_count=0,
            held_out_count=0,
            active_cell_count=int(active_cell_count),
            largest_connected_active_cell_count=int(
                largest_connected_active_cell_count
            ),
            free_node_count=int(free_node_count),
            train_error_p95_before_pixels=None,
            train_error_p95_after_pixels=None,
            held_out_error_p95_before_pixels=None,
            held_out_error_p95_after_pixels=None,
            held_out_error_max_before_pixels=None,
            held_out_error_max_after_pixels=None,
            maximum_displacement_pixels=None,
            displacement_p95_pixels=None,
            minimum_jacobian_determinant=None,
            maximum_jacobian_determinant=None,
            maximum_jacobian_condition=None,
            maximum_straight_line_deviation_pixels=None,
            boundary_identity_maximum_error_pixels=None,
        ),
    )


def _mesh_node_indices(
    ix: np.ndarray, iy: np.ndarray, grid_width: int
) -> np.ndarray:
    return np.column_stack(
        (
            iy * grid_width + ix,
            iy * grid_width + ix + 1,
            (iy + 1) * grid_width + ix,
            (iy + 1) * grid_width + ix + 1,
        )
    )


def _solve_mesh_nodes(
    output: np.ndarray,
    desired_displacement: np.ndarray,
    node_indices: np.ndarray,
    node_weights: np.ndarray,
    node_to_free: np.ndarray,
    free_nodes: np.ndarray,
    node_component: np.ndarray,
    robust_weights: np.ndarray,
    config: LocalMeshWarpConfig,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Solve a compact normal system with four non-zero weights per sample."""

    count = int(np.count_nonzero(free_nodes))
    normal = np.eye(count, dtype=np.float64) * float(config.identity_weight)
    right_x = np.zeros(count, dtype=np.float64)
    right_y = np.zeros(count, dtype=np.float64)
    for row, source_node_ids in enumerate(node_indices):
        free_indices = node_to_free[source_node_ids]
        usable = free_indices >= 0
        if not np.any(usable):
            continue
        indices = free_indices[usable]
        weights = node_weights[row, usable]
        weight = float(robust_weights[row])
        normal[np.ix_(indices, indices)] += weight * np.outer(weights, weights)
        right_x[indices] += weight * weights * desired_displacement[row, 0]
        right_y[indices] += weight * weights * desired_displacement[row, 1]
    # Smooth only between free nodes.  Protected samples still remain
    # pointwise identity at evaluation time; this regulariser cannot create a
    # colour/depth value or make an inactive cell active.
    free_grid = node_to_free.reshape(free_nodes.shape)
    component_grid = np.asarray(node_component, dtype=np.int32)
    if component_grid.shape != free_nodes.shape:
        raise RuntimeError("Mesh node component labels are malformed")
    smoothness = float(config.smoothness_weight)
    for first, second, first_components, second_components in (
        (
            free_grid[:, :-1],
            free_grid[:, 1:],
            component_grid[:, :-1],
            component_grid[:, 1:],
        ),
        (
            free_grid[:-1, :],
            free_grid[1:, :],
            component_grid[:-1, :],
            component_grid[1:, :],
        ),
    ):
        pairs = np.column_stack((first.reshape(-1), second.reshape(-1)))
        component_pairs = np.column_stack(
            (first_components.reshape(-1), second_components.reshape(-1))
        )
        pairs = pairs[
            (pairs[:, 0] >= 0)
            & (pairs[:, 1] >= 0)
            & (component_pairs[:, 0] > 0)
            & (component_pairs[:, 0] == component_pairs[:, 1])
        ]
        for left, right in pairs:
            normal[left, left] += smoothness
            normal[right, right] += smoothness
            normal[left, right] -= smoothness
            normal[right, left] -= smoothness
    condition = float(np.linalg.cond(normal))
    if not math.isfinite(condition) or condition > 1e10:
        return None
    try:
        solution_x = np.linalg.solve(normal, right_x)
        solution_y = np.linalg.solve(normal, right_y)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(solution_x).all() or not np.isfinite(solution_y).all():
        return None
    return solution_x, solution_y, condition


def _mesh_jacobian_audit(
    warp: LocalMeshInverseWarp,
) -> tuple[float, float, float, np.ndarray]:
    """Measure Jacobians and displacement over active cells and their centres."""

    determinants: list[float] = []
    conditions: list[float] = []
    sample_points: list[tuple[float, float]] = []
    for row, column in zip(*np.nonzero(warp.active_cells), strict=True):
        width = warp.grid_x[column + 1] - warp.grid_x[column]
        height = warp.grid_y[row + 1] - warp.grid_y[row]
        dx00 = warp.inverse_dx[row, column]
        dx10 = warp.inverse_dx[row, column + 1]
        dx01 = warp.inverse_dx[row + 1, column]
        dx11 = warp.inverse_dx[row + 1, column + 1]
        dy00 = warp.inverse_dy[row, column]
        dy10 = warp.inverse_dy[row, column + 1]
        dy01 = warp.inverse_dy[row + 1, column]
        dy11 = warp.inverse_dy[row + 1, column + 1]
        for tx, ty in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.5, 0.5)):
            ddx_dx = ((1.0 - ty) * (dx10 - dx00) + ty * (dx11 - dx01)) / width
            ddx_dy = ((1.0 - tx) * (dx01 - dx00) + tx * (dx11 - dx10)) / height
            ddy_dx = ((1.0 - ty) * (dy10 - dy00) + ty * (dy11 - dy01)) / width
            ddy_dy = ((1.0 - tx) * (dy01 - dy00) + tx * (dy11 - dy10)) / height
            jacobian = np.asarray(
                ((1.0 + ddx_dx, ddx_dy), (ddy_dx, 1.0 + ddy_dy)), dtype=np.float64
            )
            determinants.append(float(np.linalg.det(jacobian)))
            conditions.append(float(np.linalg.cond(jacobian)))
            sample_points.append(
                (
                    warp.grid_x[column] + tx * width,
                    warp.grid_y[row] + ty * height,
                )
            )
    points = np.asarray(sample_points, dtype=np.float64)
    mapped_x, mapped_y = warp.inverse_coordinates(points[:, 0], points[:, 1])
    displacement = np.hypot(mapped_x - points[:, 0], mapped_y - points[:, 1])
    return (
        min(determinants, default=1.0),
        max(determinants, default=1.0),
        max(conditions, default=1.0),
        displacement,
    )


def _contiguous_true_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive runs of true cells in a one-dimensional mask."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, enabled in enumerate(values):
        if enabled and start is None:
            start = index
        elif not enabled and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return tuple(runs)


_MESH_STRAIGHT_LINE_AUDIT_POLICY = (
    "raw_same_layer_active_centrelines_internal_grid_edges_and_cell_diagonals"
)


def _line_samples_for_cells(
    axis: np.ndarray, begin: int, end: int
) -> np.ndarray:
    """Sample every mesh-cell endpoint of an inclusive one-dimensional run."""

    values: list[np.ndarray] = []
    for cell in range(begin, end + 1):
        samples = np.linspace(axis[cell], axis[cell + 1], 5, dtype=np.float64)
        values.append(samples if not values else samples[1:])
    return np.concatenate(values) if values else np.empty(0, dtype=np.float64)


def _mapped_chord_deviation(points: np.ndarray) -> float:
    """Return the maximum perpendicular departure from a mapped line chord."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] != 2:
        return 0.0
    start = values[0]
    end = values[-1]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if not math.isfinite(length) or length <= 1e-9:
        return 0.0
    offsets = values[1:-1] - start
    deviation = np.abs(
        direction[0] * offsets[:, 1] - direction[1] * offsets[:, 0]
    ) / length
    return float(np.max(deviation)) if deviation.size else 0.0


def _mesh_straight_line_deviation(warp: LocalMeshInverseWarp) -> float:
    """Bound mesh-induced curvature of straight same-layer image strokes.

    Horizontal and vertical centre-lines catch bends of doorframes and
    skirting boards across a run of cells.  Internal grid edges catch the
    otherwise missed case where the edge itself passes through a displaced mesh
    node.  Both diagonals of every active cell catch the bilinear cross term
    that can bend an oblique edge inside one cell.  The audit evaluates the raw
    mesh before ``inverse_coordinates`` applies protected identity fallback:
    protected/unknown depth is excluded instead of being misread as a safe
    line continuation.
    """

    active = np.asarray(warp.active_cells, dtype=bool)
    if active.ndim != 2 or not np.any(active):
        return 0.0
    maximum = 0.0
    mask, origin = _coerce_same_layer_mask(
        warp.same_layer_mask, warp.bounds, warp.same_layer_origin_xy
    )

    def measure(points: np.ndarray) -> None:
        nonlocal maximum
        samples = np.asarray(points, dtype=np.float64)
        if samples.shape[0] < 3:
            return
        dx, dy, active_samples = _evaluate_mesh_displacement(
            samples,
            warp.bounds,
            np.asarray(warp.grid_x, dtype=np.float64),
            np.asarray(warp.grid_y, dtype=np.float64),
            np.asarray(warp.inverse_dx, dtype=np.float64),
            np.asarray(warp.inverse_dy, dtype=np.float64),
            active,
        )
        mapped = samples.copy()
        mapped[:, 0] += dx
        mapped[:, 1] += dy
        output_safe = _mask_at_points(mask, origin, samples)
        source_safe = _mask_at_points(mask, origin, mapped)
        safe = (
            active_samples
            & output_safe
            & source_safe
            & np.isfinite(mapped).all(axis=1)
        )
        for begin, end in _contiguous_true_runs(safe):
            if end - begin + 1 >= 3:
                maximum = max(maximum, _mapped_chord_deviation(mapped[begin : end + 1]))

    # Centre-lines across connected active runs sample the two line families
    # most likely to contain a doorframe or a skirting-board edge.
    for row in range(active.shape[0]):
        centre_y = 0.5 * (warp.grid_y[row] + warp.grid_y[row + 1])
        for begin, end in _contiguous_true_runs(active[row]):
            x_values = _line_samples_for_cells(warp.grid_x, begin, end)
            measure(np.column_stack((x_values, np.full_like(x_values, centre_y))))
    for column in range(active.shape[1]):
        centre_x = 0.5 * (warp.grid_x[column] + warp.grid_x[column + 1])
        for begin, end in _contiguous_true_runs(active[:, column]):
            y_values = _line_samples_for_cells(warp.grid_y, begin, end)
            measure(np.column_stack((np.full_like(y_values, centre_x), y_values)))

    # A line such as a doorframe can lie exactly on an internal mesh edge.
    # Require support on both sides so an identity-protected edge is never
    # treated as the continuation of a same-layer line.
    for row in range(1, active.shape[0]):
        shared = active[row - 1] & active[row]
        for begin, end in _contiguous_true_runs(shared):
            x_values = _line_samples_for_cells(warp.grid_x, begin, end)
            measure(
                np.column_stack(
                    (x_values, np.full_like(x_values, warp.grid_y[row]))
                )
            )
    for column in range(1, active.shape[1]):
        shared = active[:, column - 1] & active[:, column]
        for begin, end in _contiguous_true_runs(shared):
            y_values = _line_samples_for_cells(warp.grid_y, begin, end)
            measure(
                np.column_stack(
                    (np.full_like(y_values, warp.grid_x[column]), y_values)
                )
            )

    # A bilinear cell maps its two diagonals non-linearly if its cross term is
    # non-zero.  Audit both directions even when no horizontal/vertical line
    # happens to traverse the local seam.
    parameter = np.linspace(0.0, 1.0, 9)
    for row, column in zip(*np.nonzero(active), strict=True):
        x0, x1 = warp.grid_x[column], warp.grid_x[column + 1]
        y0, y1 = warp.grid_y[row], warp.grid_y[row + 1]
        measure(
            np.column_stack((x0 + parameter * (x1 - x0), y0 + parameter * (y1 - y0)))
        )
        measure(
            np.column_stack((x0 + parameter * (x1 - x0), y1 - parameter * (y1 - y0)))
        )
    return float(maximum)


def _stratified_mesh_fit_indices(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    maximum_count: int,
    minimum_per_cell: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bound fitting work without silently dropping an active cell's support."""

    count = len(cell_x)
    if count <= maximum_count:
        return np.arange(count, dtype=np.int64)
    cells, inverse = np.unique(
        np.column_stack((cell_y, cell_x)), axis=0, return_inverse=True
    )
    mandatory: list[np.ndarray] = []
    for cell_index in range(len(cells)):
        members = np.flatnonzero(inverse == cell_index)
        take = min(len(members), minimum_per_cell)
        mandatory.append(rng.choice(members, size=take, replace=False))
    selected = np.unique(np.concatenate(mandatory)) if mandatory else np.empty(0, dtype=np.int64)
    # The configured 4096-point default exceeds 4 samples for every possible
    # 16 px cell of the bounded 160 x 800 corridor.  If a caller lowers the
    # cap below this structural minimum, preserve coverage rather than fitting
    # an unobserved cell from smoothness alone.
    if len(selected) >= maximum_count:
        return np.sort(selected)
    remaining = np.setdiff1d(np.arange(count, dtype=np.int64), selected, assume_unique=False)
    extra = rng.choice(remaining, size=maximum_count - len(selected), replace=False)
    return np.sort(np.concatenate((selected, extra)))


def fit_local_mesh_inverse_warp(
    output_points_xy: np.ndarray,
    source_sample_points_xy: np.ndarray,
    bounds: TileBounds,
    *,
    same_layer_mask: np.ndarray | None = None,
    same_layer_origin_xy: tuple[float, float] | None = None,
    fit_support_mask: np.ndarray | None = None,
    fit_support_origin_xy: tuple[float, float] | None = None,
    config: LocalMeshWarpConfig | None = None,
) -> LocalMeshWarpFitResult:
    """Fit a bounded 16/32 px inverse mesh from one verified surface layer.

    The caller must first obtain geometry-only correspondences and supply the
    common local virtual coordinates of the same physical layer.  The
    ``same_layer_mask`` is the application/safety domain: cells touching an
    inactive region are identity-pinned and no bilinear interpolation crosses
    it.  ``fit_support_mask`` may be a strict subset used only to select
    independent RGB-flow-supported correspondences.  Separating those masks
    lets a held-out but otherwise safe RGB pixel validate a field without
    letting its RGB evidence fit that field.  The mesh is accepted only after
    deterministic held-out improvement and Jacobian, displacement, and
    boundary audits succeed.
    """

    settings = LocalMeshWarpConfig() if config is None else config
    if not isinstance(settings, LocalMeshWarpConfig):
        raise TypeError("config must be a LocalMeshWarpConfig")
    settings.validate()
    bounds.validate()
    output = np.asarray(output_points_xy, dtype=np.float64)
    source = np.asarray(source_sample_points_xy, dtype=np.float64)
    if output.ndim != 2 or source.ndim != 2 or output.shape[1:] != (2,) or source.shape != output.shape:
        raise ValueError("Mesh correspondences must be matching N x 2 arrays")
    mask, mask_origin = _coerce_same_layer_mask(
        same_layer_mask, bounds, same_layer_origin_xy
    )
    if fit_support_mask is None:
        fit_mask, fit_mask_origin = mask, mask_origin
    else:
        fit_mask, fit_mask_origin = _coerce_same_layer_mask(
            fit_support_mask, bounds, fit_support_origin_xy
        )
    finite = np.isfinite(output).all(axis=1) & np.isfinite(source).all(axis=1)
    in_bounds = bounds.contains(output) if output.size else np.empty(0, dtype=bool)
    in_layer = _mask_at_points(mask, mask_origin, output) if output.size else np.empty(0, dtype=bool)
    source_in_layer = (
        _mask_at_points(mask, mask_origin, source)
        if output.size
        else np.empty(0, dtype=bool)
    )
    in_fit_support = (
        _mask_at_points(fit_mask, fit_mask_origin, output)
        & _mask_at_points(fit_mask, fit_mask_origin, source)
        if output.size
        else np.empty(0, dtype=bool)
    )
    usable_points = finite & in_bounds & in_layer & source_in_layer & in_fit_support
    output = output[usable_points]
    source = source[usable_points]
    if len(output) < int(settings.minimum_correspondences):
        return _empty_mesh_audit("insufficient_same_layer_correspondences", len(output))
    grid_x = _mesh_axis(bounds.x0, bounds.x1, int(settings.grid_spacing_pixels))
    grid_y = _mesh_axis(bounds.y0, bounds.y1, int(settings.grid_spacing_pixels))
    active_cells = _mesh_active_cells_from_mask(
        grid_x,
        grid_y,
        mask,
        mask_origin,
        float(settings.minimum_active_fraction),
    )
    ix, iy, _, _, point_weights = _mesh_cell_coordinates(output, grid_x, grid_y)
    cell_counts = np.zeros(active_cells.shape, dtype=np.int32)
    np.add.at(cell_counts, (iy, ix), 1)
    active_cells &= cell_counts >= int(settings.minimum_samples_per_active_cell)
    candidate_active_cell_count = int(np.count_nonzero(active_cells))
    active_cells, largest_active_component_cell_count = (
        _retain_supported_active_mesh_components(
            active_cells, int(settings.minimum_active_cells)
        )
    )
    active_cell_count = int(np.count_nonzero(active_cells))
    if active_cell_count < int(settings.minimum_active_cells):
        return _empty_mesh_audit(
            "insufficient_active_mesh_cells",
            len(output),
            active_cell_count=candidate_active_cell_count,
            largest_connected_active_cell_count=largest_active_component_cell_count,
        )
    in_active_cell = active_cells[iy, ix]
    output = output[in_active_cell]
    source = source[in_active_cell]
    ix = ix[in_active_cell]
    iy = iy[in_active_cell]
    point_weights = point_weights[in_active_cell]
    if len(output) < int(settings.minimum_correspondences):
        return _empty_mesh_audit("insufficient_supported_active_cell_correspondences", len(output))
    active_nodes, free_nodes, node_component = _mesh_free_nodes(active_cells)
    free_node_count = int(np.count_nonzero(free_nodes))
    if not np.any(active_cells) or free_node_count == 0:
        return _empty_mesh_audit("no_interior_same_layer_mesh_support", len(output))
    node_to_free = np.full(active_nodes.size, -1, dtype=np.int32)
    node_to_free[np.flatnonzero(free_nodes)] = np.arange(free_node_count, dtype=np.int32)
    node_indices = _mesh_node_indices(ix, iy, len(grid_x))
    has_free_basis = np.any(node_to_free[node_indices] >= 0, axis=1)
    output = output[has_free_basis]
    source = source[has_free_basis]
    node_indices = node_indices[has_free_basis]
    point_weights = point_weights[has_free_basis]
    ix = ix[has_free_basis]
    iy = iy[has_free_basis]
    count = len(output)
    if count < int(settings.minimum_correspondences):
        return _empty_mesh_audit("insufficient_interior_mesh_correspondences", count)
    rng = np.random.default_rng(int(settings.held_out_seed))
    if count > int(settings.maximum_fit_correspondences):
        selected = _stratified_mesh_fit_indices(
            ix,
            iy,
            int(settings.maximum_fit_correspondences),
            int(settings.minimum_samples_per_active_cell),
            rng,
        )
        output = output[selected]
        source = source[selected]
        node_indices = node_indices[selected]
        point_weights = point_weights[selected]
        ix = ix[selected]
        iy = iy[selected]
        count = len(output)
    order = rng.permutation(count)
    held_out_count = max(1, int(round(count * float(settings.held_out_fraction))))
    held_out_indices = order[:held_out_count]
    training_indices = order[held_out_count:]
    if len(training_indices) < int(settings.minimum_correspondences):
        return _empty_mesh_audit("insufficient_training_correspondences", count)
    desired = source - output
    robust_weights = np.ones(len(training_indices), dtype=np.float64)
    solution: tuple[np.ndarray, np.ndarray, float] | None = None
    for _ in range(int(settings.robust_iterations)):
        solution = _solve_mesh_nodes(
            output[training_indices],
            desired[training_indices],
            node_indices[training_indices],
            point_weights[training_indices],
            node_to_free,
            free_nodes,
            node_component,
            robust_weights,
            settings,
        )
        if solution is None:
            return _empty_mesh_audit("ill_conditioned_mesh_solver", count)
        solution_x, solution_y, _ = solution
        dx_nodes = np.zeros(active_nodes.shape, dtype=np.float64)
        dy_nodes = np.zeros(active_nodes.shape, dtype=np.float64)
        dx_nodes.reshape(-1)[np.flatnonzero(free_nodes)] = solution_x
        dy_nodes.reshape(-1)[np.flatnonzero(free_nodes)] = solution_y
        temporary = LocalMeshInverseWarp(
            bounds=bounds,
            grid_x=grid_x,
            grid_y=grid_y,
            inverse_dx=dx_nodes,
            inverse_dy=dy_nodes,
            active_cells=active_cells,
            same_layer_mask=mask,
            same_layer_origin_xy=mask_origin,
        )
        fitted_x, fitted_y = temporary.inverse_coordinates(
            output[training_indices, 0], output[training_indices, 1]
        )
        residual = np.hypot(
            fitted_x - source[training_indices, 0],
            fitted_y - source[training_indices, 1],
        )
        scale = max(0.05, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        threshold = max(float(settings.huber_delta_pixels), 1.5 * scale)
        robust_weights = np.minimum(1.0, threshold / np.maximum(residual, 1e-9))
    assert solution is not None
    solution_x, solution_y, _ = solution
    dx_nodes = np.zeros(active_nodes.shape, dtype=np.float64)
    dy_nodes = np.zeros(active_nodes.shape, dtype=np.float64)
    dx_nodes.reshape(-1)[np.flatnonzero(free_nodes)] = solution_x
    dy_nodes.reshape(-1)[np.flatnonzero(free_nodes)] = solution_y
    candidate = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid_x,
        grid_y=grid_y,
        inverse_dx=dx_nodes,
        inverse_dy=dy_nodes,
        active_cells=active_cells,
        same_layer_mask=mask,
        same_layer_origin_xy=mask_origin,
    )
    train_x, train_y = candidate.inverse_coordinates(
        output[training_indices, 0], output[training_indices, 1]
    )
    held_x, held_y = candidate.inverse_coordinates(
        output[held_out_indices, 0], output[held_out_indices, 1]
    )
    train_after = np.hypot(
        train_x - source[training_indices, 0], train_y - source[training_indices, 1]
    )
    held_after = np.hypot(
        held_x - source[held_out_indices, 0], held_y - source[held_out_indices, 1]
    )
    train_before = np.linalg.norm(output[training_indices] - source[training_indices], axis=1)
    held_before = np.linalg.norm(output[held_out_indices] - source[held_out_indices], axis=1)
    minimum_det, maximum_det, maximum_condition, displacement = _mesh_jacobian_audit(candidate)
    maximum_straight_line_deviation = _mesh_straight_line_deviation(candidate)
    maximum_displacement = float(np.max(displacement)) if displacement.size else 0.0
    fixed_nodes = ~free_nodes
    fixed_displacement = np.hypot(dx_nodes[fixed_nodes], dy_nodes[fixed_nodes])
    boundary_identity_error = (
        float(np.max(fixed_displacement)) if fixed_displacement.size else 0.0
    )
    audit = LocalMeshWarpAudit(
        accepted=False,
        reason="rejected",
        correspondence_count=count,
        training_count=len(training_indices),
        held_out_count=len(held_out_indices),
        active_cell_count=active_cell_count,
        largest_connected_active_cell_count=largest_active_component_cell_count,
        free_node_count=free_node_count,
        train_error_p95_before_pixels=_p95(train_before),
        train_error_p95_after_pixels=_p95(train_after),
        held_out_error_p95_before_pixels=_p95(held_before),
        held_out_error_p95_after_pixels=_p95(held_after),
        held_out_error_max_before_pixels=(
            float(np.max(held_before)) if held_before.size else None
        ),
        held_out_error_max_after_pixels=(
            float(np.max(held_after)) if held_after.size else None
        ),
        maximum_displacement_pixels=maximum_displacement,
        displacement_p95_pixels=_p95(displacement),
        minimum_jacobian_determinant=minimum_det,
        maximum_jacobian_determinant=maximum_det,
        maximum_jacobian_condition=maximum_condition,
        maximum_straight_line_deviation_pixels=maximum_straight_line_deviation,
        boundary_identity_maximum_error_pixels=boundary_identity_error,
    )
    if not (
        float(settings.minimum_jacobian_determinant)
        <= minimum_det
        <= maximum_det
        <= float(settings.maximum_jacobian_determinant)
    ):
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_jacobian_determinant_out_of_bounds"))
    if not math.isfinite(maximum_condition) or maximum_condition > float(
        settings.maximum_jacobian_condition
    ):
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_jacobian_condition_out_of_bounds"))
    if (
        not math.isfinite(maximum_straight_line_deviation)
        or maximum_straight_line_deviation
        > float(settings.maximum_straight_line_deviation_pixels)
    ):
        return LocalMeshWarpFitResult(
            None, replace(audit, reason="mesh_straight_line_deviation_exceeded")
        )
    if maximum_displacement > float(settings.maximum_displacement_pixels):
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_maximum_displacement_exceeded"))
    if boundary_identity_error > 1e-9:
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_boundary_not_identity"))
    before_p95 = audit.held_out_error_p95_before_pixels
    after_p95 = audit.held_out_error_p95_after_pixels
    if before_p95 is None or after_p95 is None:
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_no_held_out_support"))
    if after_p95 > float(settings.maximum_held_out_error_pixels):
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_held_out_error_exceeded"))
    after_max = audit.held_out_error_max_after_pixels
    if (
        after_max is None
        or after_max > float(settings.maximum_held_out_maximum_error_pixels)
    ):
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_held_out_maximum_error_exceeded"))
    if before_p95 - after_p95 < float(settings.minimum_held_out_improvement_pixels):
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_no_material_held_out_improvement"))
    improvement_ratio = (before_p95 - after_p95) / max(before_p95, 1e-9)
    if improvement_ratio < float(settings.minimum_held_out_improvement_ratio):
        return LocalMeshWarpFitResult(None, replace(audit, reason="mesh_held_out_improvement_ratio_too_small"))
    return LocalMeshWarpFitResult(candidate, replace(audit, accepted=True, reason="accepted"))


__all__ = [
    "AdjacentPairGeometry",
    "ActiveMeshForwardInverseResult",
    "BidirectionalForegroundInstances",
    "DirectedReprojection",
    "DirectedSurfaceSafety",
    "ForegroundInstanceMatch",
    "GeometryAssistConfig",
    "GeometryIntrinsics",
    "IntrinsicsLike",
    "LayerLabel",
    "LocalInverseWarp",
    "LocalMeshInverseWarp",
    "LocalMeshWarpAudit",
    "LocalMeshWarpConfig",
    "LocalMeshWarpFitResult",
    "LocalWarpAudit",
    "LocalWarpConfig",
    "LocalWarpFitResult",
    "PairGeometryAudit",
    "SampledDepth",
    "SignedOcclusionForegroundComponents",
    "TileBounds",
    "analyze_adjacent_rgbd_pair",
    "coerce_intrinsics",
    "classify_directed_surface_safety",
    "depth_edge_guard",
    "depth_tolerance_mm",
    "extract_signed_occlusion_foreground_components",
    "fit_local_inverse_warp",
    "fit_local_mesh_inverse_warp",
    "match_bidirectional_signed_occlusion_instances",
    "mutually_consistent_correspondences",
    "sample_aligned_depth_nearest",
    "solve_active_mesh_forward_inverse",
    "validate_camera_to_world",
]
