"""Diagnostic-only constrained foreground inverse deformation.

The formal RGB pushbroom renderer deliberately treats foreground instances as
hard-owner content.  This module is an isolated experimental analysis helper:
it can fit and apply one small *inverse* mesh inside an already established
foreground instance, but it never receives or changes a pose, creates pixels,
blends sources, or writes an artifact.  Its caller is responsible for the
separate two-file diagnostic publication route.

Dense maps and masks are kept only in memory.  :meth:`ForegroundDeformationResult.as_dict`
returns scalar audit evidence only, so it is safe to embed in a diagnostic
report.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import cv2
import numpy as np

from .cuda_backend import remap as accelerated_remap
_IDENTITY_EPSILON = 1e-5


@dataclass(frozen=True)
class ForegroundDeformationExperimentConfig:
    """Closed limits for the diagnostic foreground deformation experiment.

    YAML intentionally exposes only ``enabled``.  Every numerical bound lives
    here so the formal configuration cannot be loosened accidentally before a
    separate policy and delivery schema are approved.
    """

    enabled: bool = False
    analysis_corridor_width_pixels: int = 128
    mesh_cell_pixels: int = 16
    maximum_displacement_pixels: float = 2.0
    minimum_track_association_score: float = 0.90
    minimum_correspondences: int = 48
    minimum_held_out_strong_edge_pixels: int = 30
    maximum_flow_fb_error_pixels: float = 0.75
    maximum_held_out_error_pixels: float = 0.75
    maximum_held_out_maximum_error_pixels: float = 2.0
    minimum_held_out_improvement_ratio: float = 0.30
    minimum_local_scale: float = 0.95
    maximum_local_scale: float = 1.05

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "ForegroundDeformationExperimentConfig":
        supplied = {} if value is None else dict(value)
        unknown = sorted(set(supplied) - {"enabled"})
        if unknown:
            raise ValueError(
                "foreground_deformation_experiment only accepts enabled; "
                "the experimental safety limits are fixed"
            )
        enabled = supplied.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError("foreground_deformation_experiment.enabled must be a boolean")
        result = cls(enabled=bool(enabled))
        result.validate()
        return result

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("foreground deformation enabled must be a boolean")
        if not 96 <= int(self.analysis_corridor_width_pixels) <= 160:
            raise ValueError("foreground deformation corridor must be in [96, 160]")
        if int(self.mesh_cell_pixels) not in {16, 32}:
            raise ValueError("foreground deformation mesh cells must be 16 or 32 px")
        finite_positive = (
            "maximum_displacement_pixels",
            "minimum_track_association_score",
            "maximum_flow_fb_error_pixels",
            "maximum_held_out_error_pixels",
            "maximum_held_out_maximum_error_pixels",
            "minimum_held_out_improvement_ratio",
            "minimum_local_scale",
            "maximum_local_scale",
        )
        for name in finite_positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"foreground deformation {name} must be finite and positive")
        if int(self.minimum_correspondences) < 48:
            raise ValueError("foreground deformation needs at least 48 correspondences")
        if int(self.minimum_held_out_strong_edge_pixels) < 30:
            raise ValueError("foreground deformation needs at least 30 held-out strong edges")
        if self.maximum_displacement_pixels > 2.0:
            raise ValueError("foreground deformation maximum displacement cannot exceed 2 px")
        if self.maximum_held_out_error_pixels > 0.75:
            raise ValueError("foreground deformation held-out P95 cannot exceed 0.75 px")
        if self.maximum_held_out_maximum_error_pixels > 2.0:
            raise ValueError("foreground deformation held-out maximum cannot exceed 2 px")
        if self.minimum_held_out_improvement_ratio < 0.30:
            raise ValueError("foreground deformation needs at least 30% held-out improvement")
        if self.minimum_local_scale < 0.95 or self.maximum_local_scale > 1.05:
            raise ValueError("foreground deformation local scale must stay within [0.95, 1.05]")
        if self.minimum_local_scale > self.maximum_local_scale:
            raise ValueError("foreground deformation local scale bounds are unordered")

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "analysis_corridor_width_pixels": int(self.analysis_corridor_width_pixels),
            "mesh_cell_pixels": int(self.mesh_cell_pixels),
            "maximum_displacement_pixels": float(self.maximum_displacement_pixels),
            "minimum_track_association_score": float(self.minimum_track_association_score),
            "minimum_correspondences": int(self.minimum_correspondences),
            "minimum_held_out_strong_edge_pixels": int(self.minimum_held_out_strong_edge_pixels),
            "maximum_flow_fb_error_pixels": float(self.maximum_flow_fb_error_pixels),
            "maximum_held_out_error_pixels": float(self.maximum_held_out_error_pixels),
            "maximum_held_out_maximum_error_pixels": float(
                self.maximum_held_out_maximum_error_pixels
            ),
            "minimum_held_out_improvement_ratio": float(
                self.minimum_held_out_improvement_ratio
            ),
            "local_scale_range": [
                float(self.minimum_local_scale),
                float(self.maximum_local_scale),
            ],
        }


@dataclass(frozen=True)
class ForegroundTrackEvidence:
    """The non-image evidence needed before a foreground mesh may be tried."""

    track_id: int
    association_score: float
    one_to_one: bool
    no_split_merge: bool
    complete_source_coverage: bool
    bidirectional_visibility: bool
    contour_correspondence: bool
    centreline_correspondence: bool
    no_real_joint: bool
    no_object_endpoint: bool
    no_occlusion_or_disocclusion: bool
    native_resolution: bool

    def validate(self) -> None:
        if int(self.track_id) < 0:
            raise ValueError("foreground deformation track_id must be non-negative")
        if not math.isfinite(float(self.association_score)) or not 0.0 <= float(
            self.association_score
        ) <= 1.0:
            raise ValueError("foreground deformation track score must be in [0, 1]")
        for name in (
            "one_to_one",
            "no_split_merge",
            "complete_source_coverage",
            "bidirectional_visibility",
            "contour_correspondence",
            "centreline_correspondence",
            "no_real_joint",
            "no_object_endpoint",
            "no_occlusion_or_disocclusion",
            "native_resolution",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"foreground deformation evidence {name} must be a boolean")

    def as_dict(self) -> dict[str, object]:
        return {
            "track_id": int(self.track_id),
            "association_score": float(self.association_score),
            "one_to_one": bool(self.one_to_one),
            "no_split_merge": bool(self.no_split_merge),
            "complete_source_coverage": bool(self.complete_source_coverage),
            "bidirectional_visibility": bool(self.bidirectional_visibility),
            "contour_correspondence": bool(self.contour_correspondence),
            "centreline_correspondence": bool(self.centreline_correspondence),
            "no_real_joint": bool(self.no_real_joint),
            "no_object_endpoint": bool(self.no_object_endpoint),
            "no_occlusion_or_disocclusion": bool(
                self.no_occlusion_or_disocclusion
            ),
            "native_resolution": bool(self.native_resolution),
        }


@dataclass(frozen=True)
class ForegroundDeformationResult:
    """In-memory diagnostic deformation and a scalar-only acceptance audit."""

    accepted: bool
    reason: str
    warped_source_bgr: np.ndarray
    active_mask: np.ndarray
    inverse_map_x: np.ndarray
    inverse_map_y: np.ndarray
    audit: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return dict(self.audit)


def _as_bgr(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be a BGR uint8 image")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError(f"{name} cannot be empty")
    return np.ascontiguousarray(image)


def _as_mask(value: np.ndarray | None, shape: tuple[int, int], name: str) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(value)
    if mask.shape != shape or mask.ndim != 2:
        raise ValueError(f"{name} must match the foreground image shape")
    if mask.dtype not in {np.dtype(bool), np.dtype(np.uint8)}:
        raise ValueError(f"{name} must be bool or uint8")
    return np.ascontiguousarray(mask.astype(bool))


def _identity_maps(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    return x, y


def _p95(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return None if finite.size == 0 else float(np.percentile(finite, 95.0))


def _scalar(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if math.isfinite(float(value)) else None


def _empty_result(
    *,
    source: np.ndarray,
    reason: str,
    track: ForegroundTrackEvidence,
    config: ForegroundDeformationExperimentConfig,
    foreground_mask: np.ndarray,
    protected_mask: np.ndarray,
    extra: Mapping[str, object] | None = None,
) -> ForegroundDeformationResult:
    map_x, map_y = _identity_maps(source.shape[:2])
    audit: dict[str, object] = {
        "policy": "foreground_local_inverse_mesh_diagnostic_v1",
        "candidate": False,
        "accepted": False,
        "reason": reason,
        "track": track.as_dict(),
        "config": config.as_dict(),
        "foreground_instance_pixel_count": int(np.count_nonzero(foreground_mask)),
        "protected_pixel_count": int(np.count_nonzero(protected_mask)),
        "active_pixel_count": 0,
        "maximum_displacement_pixels": 0.0,
        "outside_instance_maximum_displacement_pixels": 0.0,
        "pose_rewrite_detected": False,
        "color_generation_detected": False,
        "owner_policy": "single_source_hard_owner_only",
        "alpha_blend_pixel_count": 0,
        "multiband_pixel_count": 0,
        "global_flow_or_apap_used": False,
    }
    if extra:
        audit.update(dict(extra))
    return ForegroundDeformationResult(
        accepted=False,
        reason=reason,
        warped_source_bgr=source.copy(),
        active_mask=np.zeros(source.shape[:2], dtype=bool),
        inverse_map_x=map_x,
        inverse_map_y=map_y,
        audit=audit,
    )


def _flow_pair(
    reference: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    common = dict(
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    return (
        cv2.calcOpticalFlowFarneback(first_gray, second_gray, None, **common),
        cv2.calcOpticalFlowFarneback(second_gray, first_gray, None, **common),
    )


def _sample_vector_field(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    first = accelerated_remap(
        np.asarray(field[:, :, 0], dtype=np.float32),
        x.astype(np.float32),
        y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    second = accelerated_remap(
        np.asarray(field[:, :, 1], dtype=np.float32),
        x.astype(np.float32),
        y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    return np.dstack((first, second)).astype(np.float64)


def _nearest_mask(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sampled = accelerated_remap(
        np.asarray(mask, dtype=np.uint8),
        x.astype(np.float32),
        y.astype(np.float32),
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.asarray(sampled, dtype=bool)


def _strong_edges(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    candidates = magnitude[mask]
    if candidates.size == 0:
        return np.zeros(mask.shape, dtype=bool)
    # Native image gradients below this value are too weak to support a
    # sub-pixel seam observation.  The percentile keeps the rule deterministic
    # across a dark hose and a bright one.
    threshold = max(12.0, float(np.percentile(candidates, 60.0)))
    return np.ascontiguousarray(mask & (magnitude >= threshold))


def _deterministic_held_out(mask: np.ndarray) -> np.ndarray:
    """Reserve roughly one fifth of strong-edge samples without random state."""

    y, x = np.indices(mask.shape)
    return np.ascontiguousarray(mask & (((11 * x + 7 * y) % 5) == 0))


def _mesh_scale_audit(
    map_x: np.ndarray, map_y: np.ndarray, active: np.ndarray
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return singular-scale and determinant extrema on local active quads."""

    if active.shape[0] < 2 or active.shape[1] < 2:
        return None, None, None, None
    quad = (
        active[:-1, :-1]
        & active[:-1, 1:]
        & active[1:, :-1]
        & active[1:, 1:]
    )
    if not np.any(quad):
        return None, None, None, None
    dx_x = map_x[:-1, 1:] - map_x[:-1, :-1]
    dx_y = map_y[:-1, 1:] - map_y[:-1, :-1]
    dy_x = map_x[1:, :-1] - map_x[:-1, :-1]
    dy_y = map_y[1:, :-1] - map_y[:-1, :-1]
    selected = np.flatnonzero(quad)
    matrices = np.stack(
        (
            np.column_stack((dx_x.ravel()[selected], dy_x.ravel()[selected])),
            np.column_stack((dx_y.ravel()[selected], dy_y.ravel()[selected])),
        ),
        axis=1,
    )
    singular = np.linalg.svd(matrices, compute_uv=False)
    determinants = np.linalg.det(matrices)
    return (
        _scalar(float(np.min(singular))),
        _scalar(float(np.max(singular))),
        _scalar(float(np.min(determinants))),
        _scalar(float(np.max(determinants))),
    )


def _track_reason(
    track: ForegroundTrackEvidence, config: ForegroundDeformationExperimentConfig
) -> str | None:
    if track.association_score < config.minimum_track_association_score:
        return "track_association_score_below_threshold"
    if not track.one_to_one or not track.no_split_merge:
        return "track_split_merge_or_nonunique_association"
    if not track.complete_source_coverage:
        return "incomplete_dual_source_foreground_coverage"
    if not track.bidirectional_visibility:
        return "missing_bidirectional_foreground_visibility"
    if not track.contour_correspondence or not track.centreline_correspondence:
        return "foreground_contour_or_centreline_correspondence_rejected"
    if not track.no_real_joint:
        return "physical_joint_or_connector_not_excluded"
    if not track.no_object_endpoint:
        return "object_endpoint_not_excluded"
    if not track.no_occlusion_or_disocclusion:
        return "occlusion_or_disocclusion_not_excluded"
    if not track.native_resolution:
        return "non_native_or_upsampled_foreground_evidence"
    return None


def _active_mesh_cell_audit(
    mask: np.ndarray, cell_pixels: int
) -> tuple[int, int]:
    """Count 4-connected 16/32 px mesh cells supported by one instance."""

    y, x = np.nonzero(mask)
    if not len(x):
        return 0, 0
    cells = {
        (int(row // cell_pixels), int(column // cell_pixels))
        for row, column in zip(y, x, strict=True)
    }
    largest = 0
    remaining = set(cells)
    while remaining:
        seed = remaining.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            row, column = frontier.pop()
            for neighbour in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    frontier.append(neighbour)
        largest = max(largest, len(component))
    return len(cells), largest


def _crossing_instance_mask(
    foreground: np.ndarray, *, owner_boundary_x: float
) -> np.ndarray | None:
    """Return the one connected instance that genuinely spans an owner edge.

    A foreground label can contain several disconnected pieces after a narrow
    corridor crop.  An experiment may only touch the piece with support on
    both sides of the actual hard-owner boundary; a nearby object must not be
    promoted merely because it happens to be in the same label image.
    """

    labels_count, labels = cv2.connectedComponents(
        np.ascontiguousarray(foreground.astype(np.uint8)), connectivity=4
    )
    columns = np.arange(foreground.shape[1], dtype=np.float64)[None, :]
    crossing_labels: list[int] = []
    for label in range(1, int(labels_count)):
        component = labels == label
        if np.any(component & (columns < owner_boundary_x)) and np.any(
            component & (columns >= owner_boundary_x)
        ):
            crossing_labels.append(label)
    if len(crossing_labels) != 1:
        return None
    return np.ascontiguousarray(labels == crossing_labels[0])


def _contour_and_centreline_support(
    foreground: np.ndarray,
    *,
    flow_safe: np.ndarray,
    fb_error: np.ndarray,
    maximum_fb_error: float,
    owner_boundary_x: float,
    band_half_width: int,
) -> tuple[dict[str, object], str | None]:
    """Verify native contour and centreline correspondence near the seam."""

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(foreground.astype(np.uint8), kernel, iterations=1).astype(bool)
    contour = foreground & ~eroded
    distance = cv2.distanceTransform(foreground.astype(np.uint8), cv2.DIST_L2, 3)
    positive_distance = distance[foreground]
    if positive_distance.size == 0:
        return {"contour_count": 0, "centreline_count": 0}, "empty_foreground_instance"
    centreline_threshold = max(1.0, float(np.percentile(positive_distance, 80.0)))
    centreline = foreground & (distance >= centreline_threshold)
    columns = np.arange(foreground.shape[1], dtype=np.float64)[None, :]
    seam_band = np.abs(columns - float(owner_boundary_x)) <= float(band_half_width)

    def summary(mask: np.ndarray) -> tuple[int, float | None]:
        supported = mask & seam_band & flow_safe
        values = fb_error[supported]
        return int(np.count_nonzero(supported)), _p95(values)

    contour_count, contour_p95 = summary(contour)
    centreline_count, centreline_p95 = summary(centreline)
    required = 8
    audit: dict[str, object] = {
        "contour_correspondence_count": contour_count,
        "contour_flow_fb_error_p95_pixels": contour_p95,
        "centreline_correspondence_count": centreline_count,
        "centreline_flow_fb_error_p95_pixels": centreline_p95,
        "correspondence_band_half_width_pixels": int(band_half_width),
    }
    if contour_count < required or centreline_count < required:
        return audit, "insufficient_foreground_contour_or_centreline_correspondence"
    if (
        contour_p95 is None
        or centreline_p95 is None
        or contour_p95 > maximum_fb_error
        or centreline_p95 > maximum_fb_error
    ):
        return audit, "foreground_contour_or_centreline_fb_gate_rejected"
    return audit, None


def attempt_foreground_deformation(
    reference_bgr: np.ndarray,
    source_bgr: np.ndarray,
    foreground_internal_mask: np.ndarray,
    track: ForegroundTrackEvidence,
    *,
    config: ForegroundDeformationExperimentConfig | None = None,
    reference_valid_mask: np.ndarray | None = None,
    source_valid_mask: np.ndarray | None = None,
    source_foreground_mask: np.ndarray | None = None,
    protected_mask: np.ndarray | None = None,
    owner_boundary_x: float | None = None,
) -> ForegroundDeformationResult:
    """Try one fully-audited foreground-only inverse mesh.

    ``reference_bgr`` supplies output coordinates and ``source_bgr`` is the
    only possible colour owner.  The returned image is always a copy of the
    source image outside the accepted instance interior.  Any failed gate
    returns identity maps and an unchanged source image.
    """

    settings = ForegroundDeformationExperimentConfig() if config is None else config
    if not isinstance(settings, ForegroundDeformationExperimentConfig):
        raise TypeError("foreground deformation config has the wrong type")
    settings.validate()
    reference = _as_bgr(reference_bgr, "reference_bgr")
    source = _as_bgr(source_bgr, "source_bgr")
    if reference.shape != source.shape:
        raise ValueError("foreground deformation source images must have equal shape")
    track.validate()
    shape = reference.shape[:2]
    foreground = _as_mask(foreground_internal_mask, shape, "foreground_internal_mask")
    reference_valid = _as_mask(reference_valid_mask, shape, "reference_valid_mask")
    source_valid = _as_mask(source_valid_mask, shape, "source_valid_mask")
    protected = (
        _as_mask(protected_mask, shape, "protected_mask")
        if protected_mask is not None
        else np.zeros(shape, dtype=bool)
    )
    boundary_x = (
        float(shape[1]) / 2.0
        if owner_boundary_x is None
        else float(owner_boundary_x)
    )
    if not math.isfinite(boundary_x) or not 0.0 < boundary_x < float(shape[1]):
        raise ValueError("foreground deformation owner boundary must be inside its corridor")
    if not 96 <= int(shape[1]) <= 160:
        raise ValueError(
            "foreground deformation may only receive a 96-160 px adjacent corridor"
        )

    if not settings.enabled:
        return _empty_result(
            source=source,
            reason="foreground_deformation_experiment_disabled",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
        )
    if source_foreground_mask is None:
        return _empty_result(
            source=source,
            reason="missing_complete_source_foreground_coverage_evidence",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
        )
    source_foreground = _as_mask(
        source_foreground_mask, shape, "source_foreground_mask"
    )
    track_reason = _track_reason(track, settings)
    if track_reason is not None:
        return _empty_result(
            source=source,
            reason=track_reason,
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
        )
    if not np.any(foreground):
        return _empty_result(
            source=source,
            reason="empty_foreground_instance",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
        )
    crossing_foreground = _crossing_instance_mask(
        foreground, owner_boundary_x=boundary_x
    )
    if crossing_foreground is None:
        return _empty_result(
            source=source,
            reason="foreground_instance_does_not_uniquely_cross_owner_boundary",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={"owner_boundary_x": boundary_x},
        )
    foreground = crossing_foreground
    if np.any(foreground & protected):
        return _empty_result(
            source=source,
            reason="foreground_instance_intersects_protected_domain",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
        )

    # The outer contour is explicitly pinned to identity.  This gives the
    # object boundary and all background/protection a pointwise zero-displacement
    # certificate even when an inner 16/32 px mesh cell is active.
    interior = cv2.erode(
        foreground.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    ).astype(bool)
    interior &= reference_valid & ~protected
    if int(np.count_nonzero(interior)) < settings.minimum_correspondences:
        return _empty_result(
            source=source,
            reason="insufficient_foreground_instance_interior",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={"foreground_interior_pixel_count": int(np.count_nonzero(interior))},
        )

    flow_forward, flow_backward = _flow_pair(reference, source)
    grid_x, grid_y = _identity_maps(shape)
    sample_x = grid_x.astype(np.float64) + np.asarray(flow_forward[:, :, 0], dtype=np.float64)
    sample_y = grid_y.astype(np.float64) + np.asarray(flow_forward[:, :, 1], dtype=np.float64)
    backward_at_forward = _sample_vector_field(flow_backward, sample_x, sample_y)
    forward = np.asarray(flow_forward, dtype=np.float64)
    fb_error = np.hypot(
        forward[:, :, 0] + backward_at_forward[:, :, 0],
        forward[:, :, 1] + backward_at_forward[:, :, 1],
    )
    source_inside = (
        np.isfinite(sample_x)
        & np.isfinite(sample_y)
        & (sample_x >= 0.0)
        & (sample_x <= float(shape[1] - 1))
        & (sample_y >= 0.0)
        & (sample_y <= float(shape[0] - 1))
    )
    source_covered = _nearest_mask(source_valid, sample_x, sample_y) & _nearest_mask(
        source_foreground, sample_x, sample_y
    )
    foreground_flow_safe = (
        foreground
        & reference_valid
        & ~protected
        & source_inside
        & source_covered
        & np.isfinite(fb_error)
    )
    flow_safe = interior & foreground_flow_safe
    correspondence_audit, correspondence_reason = _contour_and_centreline_support(
        foreground,
        flow_safe=foreground_flow_safe,
        fb_error=fb_error,
        maximum_fb_error=settings.maximum_flow_fb_error_pixels,
        owner_boundary_x=boundary_x,
        band_half_width=int(settings.mesh_cell_pixels),
    )
    if correspondence_reason is not None:
        return _empty_result(
            source=source,
            reason=correspondence_reason,
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "foreground_interior_pixel_count": int(np.count_nonzero(interior)),
                "owner_boundary_x": boundary_x,
                "correspondence": correspondence_audit,
            },
        )
    strong_edges = _strong_edges(reference, interior)
    raw_error = np.hypot(forward[:, :, 0], forward[:, :, 1])
    columns = np.arange(shape[1], dtype=np.float64)[None, :]
    seam_band = np.abs(columns - boundary_x) <= float(settings.mesh_cell_pixels)
    # The residual trigger is deliberately limited to native RGB strong edges
    # straddling the actual hard-owner boundary.  A mismatch elsewhere on the
    # same object is not a licence to deform this seam corridor.
    seam_strong_edges = strong_edges & flow_safe & seam_band
    # A strong edge with a zero native vector is valid evidence *against* a
    # seam mismatch, but it is not a displacement correspondence to fit.
    # Keep the measured subset deterministic before splitting it into
    # train/held-out pixels.
    observed_strong_edges = seam_strong_edges & (raw_error > 0.10)
    if int(np.count_nonzero(observed_strong_edges)) < (
        settings.minimum_held_out_strong_edge_pixels
    ):
        return _empty_result(
            source=source,
            reason="no_measurable_foreground_seam_residual",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "foreground_interior_pixel_count": int(np.count_nonzero(interior)),
                "seam_strong_edge_count": int(np.count_nonzero(seam_strong_edges)),
                "observed_seam_strong_edge_count": int(
                    np.count_nonzero(observed_strong_edges)
                ),
                "native_residual_required": "p95_gt_0.75_or_max_gt_2.0",
                "owner_boundary_x": boundary_x,
                "correspondence": correspondence_audit,
            },
        )
    held_out = _deterministic_held_out(observed_strong_edges)
    held_out_count = int(np.count_nonzero(held_out))
    raw_p95 = _p95(raw_error[held_out])
    raw_maximum = (
        float(np.max(raw_error[held_out])) if held_out_count else None
    )
    if held_out_count < settings.minimum_held_out_strong_edge_pixels:
        return _empty_result(
            source=source,
            reason="insufficient_held_out_strong_foreground_edges",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "foreground_interior_pixel_count": int(np.count_nonzero(interior)),
                "held_out_strong_edge_count": held_out_count,
                "held_out_error_p95_before_pixels": raw_p95,
                "held_out_error_max_before_pixels": raw_maximum,
                "owner_boundary_x": boundary_x,
                "correspondence": correspondence_audit,
            },
        )
    held_out_fb_p95 = _p95(fb_error[held_out])
    held_out_fb_maximum = (
        float(np.max(fb_error[held_out])) if held_out_count else None
    )
    if (
        held_out_fb_p95 is None
        or held_out_fb_maximum is None
        or held_out_fb_p95 > settings.maximum_flow_fb_error_pixels
        or held_out_fb_maximum > settings.maximum_flow_fb_error_pixels
    ):
        return _empty_result(
            source=source,
            reason="held_out_foreground_bidirectional_flow_gate_rejected",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "held_out_strong_edge_count": held_out_count,
                "held_out_flow_fb_error_p95_pixels": held_out_fb_p95,
                "held_out_flow_fb_error_maximum_pixels": held_out_fb_maximum,
                "owner_boundary_x": boundary_x,
                "correspondence": correspondence_audit,
            },
        )
    if (
        (raw_p95 is None or raw_p95 <= settings.maximum_held_out_error_pixels)
        and (raw_maximum is None or raw_maximum <= settings.maximum_held_out_maximum_error_pixels)
    ):
        return _empty_result(
            source=source,
            reason="no_measurable_foreground_seam_residual",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "foreground_interior_pixel_count": int(np.count_nonzero(interior)),
                "held_out_strong_edge_count": held_out_count,
                "held_out_error_p95_before_pixels": raw_p95,
                "held_out_error_max_before_pixels": raw_maximum,
                "native_residual_required": "p95_gt_0.75_or_max_gt_2.0",
                "owner_boundary_x": boundary_x,
                "correspondence": correspondence_audit,
            },
        )

    correspondence_mask = flow_safe & (fb_error <= settings.maximum_flow_fb_error_pixels)
    if int(np.count_nonzero(correspondence_mask)) < settings.minimum_correspondences:
        return _empty_result(
            source=source,
            reason="insufficient_bidirectional_foreground_correspondences",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "held_out_strong_edge_count": held_out_count,
                "bidirectional_correspondence_count": int(np.count_nonzero(correspondence_mask)),
                "flow_fb_error_p95_pixels": _p95(fb_error[flow_safe]),
                "held_out_error_p95_before_pixels": raw_p95,
                "held_out_error_max_before_pixels": raw_maximum,
                "owner_boundary_x": boundary_x,
                "correspondence": correspondence_audit,
            },
        )

    # This first experimental branch intentionally fits the most constrained
    # useful member of a 16/32 px local mesh family: every active mesh node
    # receives the same robust *local* inverse displacement.  It is not a
    # global transform (the field is only applied inside this one instance),
    # and its node grid, held-out partition, source-domain test and scale audit
    # are retained so a future non-rigid node solver cannot silently weaken the
    # acceptance contract.  A non-uniform residual simply fails closed today.
    # Flat hose interiors carry no displacement observation.  Fit only native
    # strong RGB structure; the remaining interior is an application domain,
    # not invented evidence.  The deterministic held-out edge subset is never
    # consulted by this median fit.
    training = observed_strong_edges & correspondence_mask & ~held_out
    training_count = int(np.count_nonzero(training))
    if training_count < settings.minimum_correspondences:
        return _empty_result(
            source=source,
            reason="insufficient_training_foreground_correspondences",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "held_out_strong_edge_count": held_out_count,
                "bidirectional_correspondence_count": int(np.count_nonzero(correspondence_mask)),
                "training_correspondence_count": training_count,
            },
        )
    displacement_x = float(np.median(forward[:, :, 0][training]))
    displacement_y = float(np.median(forward[:, :, 1][training]))
    displacement_norm = float(math.hypot(displacement_x, displacement_y))
    training_residual = np.hypot(
        forward[:, :, 0][training] - displacement_x,
        forward[:, :, 1][training] - displacement_y,
    )
    fit_audit = {
        "method": "regular_local_translation_inverse_mesh",
        "mesh_cell_pixels": int(settings.mesh_cell_pixels),
        "active_cell_count": None,
        "largest_connected_active_cell_count": None,
        "correspondence_count": int(np.count_nonzero(correspondence_mask)),
        "training_count": training_count,
        "held_out_count": held_out_count,
        "training_error_p95_before_pixels": _p95(raw_error[training]),
        "training_error_p95_after_pixels": _p95(training_residual),
        "inverse_translation_pixels": [displacement_x, displacement_y],
        "maximum_displacement_pixels": displacement_norm,
    }
    if displacement_norm > settings.maximum_displacement_pixels:
        return _empty_result(
            source=source,
            reason="foreground_mesh_maximum_displacement_exceeded",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={"mesh": fit_audit},
        )

    # Pin the contour to identity, then ramp the fitted local mesh field over
    # enough pixels that the worst permitted 2 px displacement cannot exceed
    # the 0.95--1.05 local-scale envelope.  The control lattice is still
    # 16/32 px; this distance-based taper gives every outer mesh node an exact
    # zero boundary condition instead of hiding a discontinuity at the mask.
    distance_to_boundary = cv2.distanceTransform(
        foreground.astype(np.uint8), cv2.DIST_L2, 3
    )
    allowed_scale_gradient = min(
        settings.maximum_local_scale - 1.0,
        1.0 - settings.minimum_local_scale,
    ) * 0.80
    transition_width = max(
        int(settings.mesh_cell_pixels),
        int(math.ceil(displacement_norm / max(allowed_scale_gradient, 1e-6))),
    )
    boundary_pinned_distance = np.maximum(distance_to_boundary - 1.0, 0.0)
    taper = np.clip(boundary_pinned_distance / float(transition_width), 0.0, 1.0)
    taper = np.ascontiguousarray(taper * interior)
    full_strength = interior & (taper >= 0.95)
    active_cell_count, largest_active_component = _active_mesh_cell_audit(
        full_strength, int(settings.mesh_cell_pixels)
    )
    validation_held_out = held_out & full_strength
    validation_held_out_count = int(np.count_nonzero(validation_held_out))
    fit_audit.update(
        {
            "active_cell_count": active_cell_count,
            "largest_connected_active_cell_count": largest_active_component,
            "boundary_pinned_transition_width_pixels": transition_width,
            "mesh_held_out_count": validation_held_out_count,
        }
    )
    if largest_active_component < 4:
        return _empty_result(
            source=source,
            reason="insufficient_connected_foreground_mesh_cells",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "active_mesh_cell_count": active_cell_count,
                "largest_connected_active_mesh_cell_count": largest_active_component,
                "boundary_pinned_transition_width_pixels": transition_width,
            },
        )
    if validation_held_out_count < settings.minimum_held_out_strong_edge_pixels:
        return _empty_result(
            source=source,
            reason="insufficient_held_out_strong_edges_inside_foreground_mesh",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "held_out_strong_edge_count": held_out_count,
                "mesh_held_out_strong_edge_count": validation_held_out_count,
                "boundary_pinned_transition_width_pixels": transition_width,
            },
        )
    proposed_x = grid_x.astype(np.float64) + displacement_x * taper
    proposed_y = grid_y.astype(np.float64) + displacement_y * taper
    inverse_x, inverse_y = _identity_maps(shape)
    inverse_x[interior] = np.asarray(proposed_x, dtype=np.float32)[interior]
    inverse_y[interior] = np.asarray(proposed_y, dtype=np.float32)[interior]
    displacement = np.hypot(
        inverse_x.astype(np.float64) - grid_x,
        inverse_y.astype(np.float64) - grid_y,
    )
    active = interior & (displacement > _IDENTITY_EPSILON)
    mapped_source_safe = _nearest_mask(source_valid, inverse_x, inverse_y) & _nearest_mask(
        source_foreground, inverse_x, inverse_y
    )
    active &= mapped_source_safe
    # A selected mesh cannot silently keep a deformation whose output source
    # fell into a foreground hole.  It must fall back as a whole, not trim the
    # unsafe subset into a partially trusted field.
    if np.any(interior & ~mapped_source_safe & (displacement > _IDENTITY_EPSILON)):
        return _empty_result(
            source=source,
            reason="foreground_mesh_maps_outside_complete_source_coverage",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={"mesh": fit_audit},
        )
    scale_min, scale_max, det_min, det_max = _mesh_scale_audit(
        inverse_x, inverse_y, active
    )
    if (
        scale_min is None
        or scale_max is None
        or det_min is None
        or det_max is None
        or det_min <= 0.0
        or scale_min < settings.minimum_local_scale
        or scale_max > settings.maximum_local_scale
    ):
        return _empty_result(
            source=source,
            reason="foreground_mesh_scale_or_jacobian_rejected",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "mesh": fit_audit,
                "local_scale_min": scale_min,
                "local_scale_max": scale_max,
                "local_jacobian_determinant_min": det_min,
                "local_jacobian_determinant_max": det_max,
            },
        )
    held_indices = np.flatnonzero(validation_held_out)
    mapped_error = np.hypot(
        inverse_x.ravel()[held_indices].astype(np.float64)
        - sample_x.ravel()[held_indices],
        inverse_y.ravel()[held_indices].astype(np.float64)
        - sample_y.ravel()[held_indices],
    )
    held_after_p95 = _p95(mapped_error)
    held_after_max = float(np.max(mapped_error)) if mapped_error.size else None
    validation_raw_p95 = _p95(raw_error[validation_held_out])
    validation_raw_maximum = (
        float(np.max(raw_error[validation_held_out]))
        if validation_held_out_count
        else None
    )
    improvement_ratio = (
        (float(validation_raw_p95) - float(held_after_p95))
        / max(float(validation_raw_p95), 1e-9)
        if validation_raw_p95 is not None and held_after_p95 is not None
        else None
    )
    if (
        held_after_p95 is None
        or held_after_max is None
        or held_after_p95 > settings.maximum_held_out_error_pixels
        or held_after_max > settings.maximum_held_out_maximum_error_pixels
        or improvement_ratio is None
        or improvement_ratio < settings.minimum_held_out_improvement_ratio
    ):
        return _empty_result(
            source=source,
            reason="foreground_held_out_edge_gate_rejected",
            track=track,
            config=settings,
            foreground_mask=foreground,
            protected_mask=protected,
            extra={
                "mesh": fit_audit,
                "held_out_strong_edge_count": validation_held_out_count,
                "held_out_error_p95_before_pixels": validation_raw_p95,
                "held_out_error_max_before_pixels": validation_raw_maximum,
                "held_out_error_p95_after_pixels": held_after_p95,
                "held_out_error_max_after_pixels": held_after_max,
                "held_out_improvement_ratio": improvement_ratio,
            },
        )

    remapped = accelerated_remap(
        source,
        inverse_x.astype(np.float32),
        inverse_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    warped = source.copy()
    warped[active] = remapped[active]
    outside = ~foreground
    outside_displacement = float(np.max(displacement[outside])) if np.any(outside) else 0.0
    protected_displacement = (
        float(np.max(displacement[protected])) if np.any(protected) else 0.0
    )
    if outside_displacement > _IDENTITY_EPSILON or protected_displacement > _IDENTITY_EPSILON:
        raise RuntimeError("foreground deformation escaped its instance interior")
    if np.any(active & (~foreground | protected)):
        raise RuntimeError("foreground deformation active mask escaped its protected domain")
    audit = {
        "policy": "foreground_local_inverse_mesh_diagnostic_v1",
        "candidate": True,
        "accepted": True,
        "reason": "accepted",
        "track": track.as_dict(),
        "config": settings.as_dict(),
        "foreground_instance_pixel_count": int(np.count_nonzero(foreground)),
        "foreground_interior_pixel_count": int(np.count_nonzero(interior)),
        "protected_pixel_count": int(np.count_nonzero(protected)),
        "active_pixel_count": int(np.count_nonzero(active)),
        "bidirectional_correspondence_count": int(np.count_nonzero(correspondence_mask)),
        "flow_fb_error_p95_pixels": _p95(fb_error[flow_safe]),
        "held_out_strong_edge_count": held_out_count,
        "mesh_held_out_strong_edge_count": validation_held_out_count,
        "held_out_error_p95_before_pixels": validation_raw_p95,
        "held_out_error_max_before_pixels": validation_raw_maximum,
        "held_out_error_p95_after_pixels": held_after_p95,
        "held_out_error_max_after_pixels": held_after_max,
        "held_out_improvement_ratio": improvement_ratio,
        "maximum_displacement_pixels": float(np.max(displacement[active])) if np.any(active) else 0.0,
        "displacement_p95_pixels": _p95(displacement[active]),
        "outside_instance_maximum_displacement_pixels": outside_displacement,
        "protected_maximum_displacement_pixels": protected_displacement,
        "local_scale_min": scale_min,
        "local_scale_max": scale_max,
        "local_jacobian_determinant_min": det_min,
        "local_jacobian_determinant_max": det_max,
        # Singular-value bounds apply to every active local quad, so they are
        # the conservative width/length/centreline preservation proxy.
        "contour_scale_min": scale_min,
        "contour_scale_max": scale_max,
        "centreline_scale_min": scale_min,
        "centreline_scale_max": scale_max,
        "owner_boundary_x": boundary_x,
        "boundary_pinned_transition_width_pixels": transition_width,
        "boundary_pinned_zero_displacement": bool(
            not np.any((foreground & ~interior) & (displacement > _IDENTITY_EPSILON))
        ),
        "correspondence": correspondence_audit,
        "mesh": fit_audit,
        "pose_rewrite_detected": False,
        "color_generation_detected": False,
        "owner_policy": "single_source_hard_owner_only",
        "alpha_blend_pixel_count": 0,
        "multiband_pixel_count": 0,
        "global_flow_or_apap_used": False,
    }
    return ForegroundDeformationResult(
        accepted=True,
        reason="accepted",
        warped_source_bgr=warped,
        active_mask=np.ascontiguousarray(active),
        inverse_map_x=inverse_x,
        inverse_map_y=inverse_y,
        audit=audit,
    )


__all__ = [
    "ForegroundDeformationExperimentConfig",
    "ForegroundDeformationResult",
    "ForegroundTrackEvidence",
    "attempt_foreground_deformation",
]
