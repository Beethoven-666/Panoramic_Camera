"""Candidate-only local flow/depth evidence and safe inverse mesh sampling.

The mesh is fitted solely from two already placed real sources.  When its
strict audit passes, callers may sample *the first real source* through its
inverse map in the verified same-layer region.  The helper never invents an
RGB frame, changes ownership, or changes pose; an absent/rejected mesh has no
output and therefore leaves the caller's hard owner untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .geometry_assisted_local_warp import (
    GeometryAssistConfig,
    LocalMeshWarpAudit,
    LocalMeshWarpConfig,
    LocalMeshWarpFitResult,
    TileBounds,
    depth_edge_guard,
    fit_local_mesh_inverse_warp,
)
from .video_visual_renderer import VideoVisualSource


FlowEstimator = Callable[[VideoVisualSource, VideoVisualSource], np.ndarray]


@dataclass(frozen=True)
class LocalMeshEvidenceConfig:
    """Closed C2--C4 evidence limits for one adjacent, placed source pair."""

    flow_backend: str = "dis"
    corridor_width_pixels: int = 96
    correspondence_spacing_pixels: int = 4
    forward_backward_maximum_error_pixels: float = 1.0
    maximum_flow_magnitude_pixels: float = 8.0
    require_depth_safety: bool = False
    depth_absolute_tolerance_mm: float = 20.0
    depth_relative_tolerance: float = 0.02
    depth_edge_guard_pixels: int = 3
    support_border_pixels: int = 16
    mesh: LocalMeshWarpConfig = LocalMeshWarpConfig()

    def __post_init__(self) -> None:
        if self.flow_backend not in {"dis", "raft"}:
            raise ValueError("flow_backend must be 'dis' or 'raft'")
        if not 96 <= int(self.corridor_width_pixels) <= 160:
            raise ValueError("corridor_width_pixels must be in [96, 160]")
        if not 1 <= int(self.correspondence_spacing_pixels) <= 16:
            raise ValueError("correspondence_spacing_pixels must be in [1, 16]")
        if not 0.0 < float(self.forward_backward_maximum_error_pixels) <= 8.0:
            raise ValueError("forward_backward_maximum_error_pixels must be in (0, 8]")
        if not 0.0 < float(self.maximum_flow_magnitude_pixels) <= 8.0:
            raise ValueError("maximum_flow_magnitude_pixels must be in (0, 8]")
        if not 0.0 < float(self.depth_absolute_tolerance_mm):
            raise ValueError("depth_absolute_tolerance_mm must be positive")
        if not 0.0 < float(self.depth_relative_tolerance) <= 1.0:
            raise ValueError("depth_relative_tolerance must be in (0, 1]")
        if not 0 <= int(self.depth_edge_guard_pixels) <= 16:
            raise ValueError("depth_edge_guard_pixels must be in [0, 16]")
        if not 0 <= int(self.support_border_pixels) <= 32:
            raise ValueError("support_border_pixels must be in [0, 32]")
        if not isinstance(self.mesh, LocalMeshWarpConfig):
            raise TypeError("mesh must be a LocalMeshWarpConfig")
        self.mesh.validate()


@dataclass(frozen=True)
class LocalMeshEvidenceAudit:
    """Auditable input and decision evidence for a candidate mesh attempt."""

    first_frame_id: int
    second_frame_id: int
    flow_backend: str
    corridor_bounds_xyxy: tuple[int, int, int, int] | None
    overlap_pixel_count: int
    forward_backward_reliable_pixel_count: int
    depth_safe_pixel_count: int | None
    safe_pixel_count: int
    correspondence_candidate_count: int
    correspondence_count: int
    rejection_reason: str | None
    mesh: LocalMeshWarpAudit | None
    depth_safety_required: bool
    method: str = "candidate_local_flow_depth_mesh_evidence/v1"

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "first_frame_id": self.first_frame_id,
            "second_frame_id": self.second_frame_id,
            "flow_backend": self.flow_backend,
            "corridor_bounds_xyxy": list(self.corridor_bounds_xyxy)
            if self.corridor_bounds_xyxy is not None
            else None,
            "overlap_pixel_count": self.overlap_pixel_count,
            "forward_backward_reliable_pixel_count": self.forward_backward_reliable_pixel_count,
            "depth_safe_pixel_count": self.depth_safe_pixel_count,
            "safe_pixel_count": self.safe_pixel_count,
            "correspondence_candidate_count": self.correspondence_candidate_count,
            "correspondence_count": self.correspondence_count,
            "depth_safety_required": self.depth_safety_required,
            "accepted": bool(self.mesh is not None and self.mesh.accepted),
            "rejection_reason": self.rejection_reason,
            "mesh": self.mesh.as_dict() if self.mesh is not None else None,
            "creates_colour": False,
            "creates_owner": False,
            "creates_pose": False,
            "real_adjacent_sources_only": True,
        }


@dataclass(frozen=True)
class LocalMeshEvidenceResult:
    """Mesh fit plus its safe domain; a ``None`` warp is a hard-owner fallback."""

    fit: LocalMeshWarpFitResult | None
    same_layer_mask: np.ndarray
    output_points_xy: np.ndarray
    source_points_xy: np.ndarray
    audit: LocalMeshEvidenceAudit

    @property
    def accepted(self) -> bool:
        return bool(self.fit is not None and self.fit.warp is not None and self.fit.audit.accepted)


@dataclass(frozen=True)
class LocalMeshOutputSampling:
    """Literal samples from the first source through an accepted inverse mesh."""

    bgr: np.ndarray
    applied_mask: np.ndarray

    @property
    def applied_pixel_count(self) -> int:
        return int(np.count_nonzero(self.applied_mask))


def _valid(source: VideoVisualSource) -> np.ndarray:
    return np.asarray(source.bgra)[..., 3] > 0


def _corridor(overlap: np.ndarray, width: int) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    columns = np.flatnonzero(np.any(overlap, axis=0))
    rows = np.flatnonzero(np.any(overlap, axis=1))
    if not columns.size or not rows.size:
        return np.zeros_like(overlap, dtype=bool), None
    left, right = int(columns[0]), int(columns[-1]) + 1
    actual_width = min(width, right - left)
    center = (left + right) // 2
    x0 = max(left, center - actual_width // 2)
    x1 = min(right, x0 + actual_width)
    x0 = max(left, x1 - actual_width)
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x_grid = np.arange(overlap.shape[1])[None, :]
    return overlap & (x_grid >= x0) & (x_grid < x1), (x0, y0, x1, y1)


def _dis_flow(first: VideoVisualSource, second: VideoVisualSource) -> np.ndarray:
    first_gray = cv2.cvtColor(np.asarray(first.bgra), cv2.COLOR_BGRA2GRAY)
    second_gray = cv2.cvtColor(np.asarray(second.bgra), cv2.COLOR_BGRA2GRAY)
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST).calc(
        first_gray, second_gray, None
    )


def _coerce_flow(value: np.ndarray, shape: tuple[int, int], label: str) -> np.ndarray:
    flow = np.asarray(value, dtype=np.float32)
    if flow.shape != (*shape, 2) or not np.isfinite(flow).all():
        raise ValueError(f"{label} must be finite HxWx2 flow matching the placed source canvas")
    return np.ascontiguousarray(flow)


def _sample_vector(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        field,
        x.astype(np.float32),
        y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(np.nan, np.nan),
    )


def _depth_safe_mask(
    first: VideoVisualSource,
    second: VideoVisualSource,
    *,
    flow: np.ndarray,
    settings: LocalMeshEvidenceConfig,
) -> np.ndarray:
    """Return safe first-source pixels using depth only, never RGB or pose."""

    if first.depth_mm is None or second.depth_mm is None:
        raise ValueError("depth safety was required but one adjacent real source has no aligned depth")
    first_depth = np.asarray(first.depth_mm, dtype=np.float32)
    second_depth = np.asarray(second.depth_mm, dtype=np.float32)
    shape = first_depth.shape
    first_valid = np.isfinite(first_depth) & (first_depth > 0.0)
    second_valid = np.isfinite(second_depth) & (second_depth > 0.0)
    # Reuse the formal depth-edge guard; with only placed-image flow this is a
    # local safety filter, not a pose/reprojection substitute.
    guard_config = GeometryAssistConfig(
        absolute_depth_tolerance_mm=float(settings.depth_absolute_tolerance_mm),
        relative_depth_tolerance=float(settings.depth_relative_tolerance),
        edge_absolute_depth_mm=float(settings.depth_absolute_tolerance_mm),
        edge_relative_depth=float(settings.depth_relative_tolerance),
        edge_guard_radius_pixels=int(settings.depth_edge_guard_pixels),
    )
    _, first_guard = depth_edge_guard(
        first_depth, valid_mask=first_valid, config=guard_config
    )
    _, second_guard = depth_edge_guard(
        second_depth, valid_mask=second_valid, config=guard_config
    )
    yy, xx = np.indices(shape, dtype=np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    sampled_depth = cv2.remap(
        second_depth, map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    sampled_valid = cv2.remap(
        second_valid.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    sampled_guard = cv2.remap(
        second_guard.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=1,
    ).astype(bool)
    tolerance = np.maximum(
        float(settings.depth_absolute_tolerance_mm),
        np.maximum(first_depth, sampled_depth) * float(settings.depth_relative_tolerance),
    )
    return (
        first_valid
        & sampled_valid
        & ~first_guard
        & ~sampled_guard
        & (np.abs(first_depth - sampled_depth) <= tolerance)
    )


def _empty(
    first: VideoVisualSource,
    second: VideoVisualSource,
    settings: LocalMeshEvidenceConfig,
    *,
    reason: str,
    corridor: tuple[int, int, int, int] | None,
    overlap_count: int,
    reliable_count: int = 0,
    depth_safe_count: int | None = None,
    safe_count: int = 0,
    candidate_count: int = 0,
) -> LocalMeshEvidenceResult:
    shape = np.asarray(first.bgra).shape[:2]
    return LocalMeshEvidenceResult(
        fit=None,
        same_layer_mask=np.zeros(shape, dtype=bool),
        output_points_xy=np.empty((0, 2), dtype=np.float64),
        source_points_xy=np.empty((0, 2), dtype=np.float64),
        audit=LocalMeshEvidenceAudit(
            first.frame_id, second.frame_id, settings.flow_backend, corridor, overlap_count,
            reliable_count, depth_safe_count, safe_count, candidate_count, 0, reason, None,
            settings.require_depth_safety,
        ),
    )


def assess_candidate_local_mesh_evidence(
    first: VideoVisualSource,
    second: VideoVisualSource,
    *,
    config: LocalMeshEvidenceConfig | None = None,
    flow_estimator: FlowEstimator | None = None,
    additional_safety_mask: np.ndarray | None = None,
) -> LocalMeshEvidenceResult:
    """Audit a local mesh from two real placed sources without modifying either.

    ``flow_estimator`` is required for ``raft`` and is also the test seam for
    deterministic synthetic flow.  It must return source-to-target flow for
    the exact supplied adjacent pair; the reverse call is made independently
    for forward/backward consistency.  The function never fills a failed mesh
    with identity samples: ``fit is None`` means retain a hard owner.
    """

    settings = config or LocalMeshEvidenceConfig()
    if first.frame_id == second.frame_id:
        raise ValueError("local mesh evidence requires two distinct real source frame ids")
    first_image, second_image = np.asarray(first.bgra), np.asarray(second.bgra)
    if first_image.shape != second_image.shape:
        raise ValueError("adjacent local mesh sources must share one placed BGRA canvas")
    if settings.flow_backend == "raft" and flow_estimator is None:
        raise ValueError("RAFT local mesh evidence requires an explicit verified flow estimator")
    overlap = _valid(first) & _valid(second)
    corridor_mask, corridor = _corridor(overlap, int(settings.corridor_width_pixels))
    overlap_count = int(np.count_nonzero(overlap))
    if corridor is None:
        return _empty(first, second, settings, reason="no_common_real_source_support", corridor=None, overlap_count=0)
    estimate = flow_estimator or _dis_flow
    forward = _coerce_flow(estimate(first, second), overlap.shape, "forward flow")
    backward = _coerce_flow(estimate(second, first), overlap.shape, "backward flow")
    yy, xx = np.indices(overlap.shape, dtype=np.float32)
    mapped_x, mapped_y = xx + forward[..., 0], yy + forward[..., 1]
    sampled_backward = _sample_vector(backward, mapped_x, mapped_y)
    forward_magnitude = np.linalg.norm(forward, axis=2)
    fb_error = np.linalg.norm(forward + sampled_backward, axis=2)
    target_valid = cv2.remap(
        _valid(second).astype(np.uint8), mapped_x, mapped_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    reliable = (
        corridor_mask
        & target_valid
        & np.isfinite(fb_error)
        & (fb_error <= float(settings.forward_backward_maximum_error_pixels))
        & np.isfinite(forward_magnitude)
        & (forward_magnitude <= float(settings.maximum_flow_magnitude_pixels))
    )
    reliable_count = int(np.count_nonzero(reliable))
    if not reliable_count:
        return _empty(first, second, settings, reason="no_forward_backward_reliable_flow", corridor=corridor, overlap_count=overlap_count)

    depth_safe_count: int | None = None
    safety = np.ones(overlap.shape, dtype=bool)
    if settings.require_depth_safety:
        try:
            depth_safety = _depth_safe_mask(first, second, flow=forward, settings=settings)
        except ValueError as exc:
            return _empty(first, second, settings, reason=str(exc), corridor=corridor, overlap_count=overlap_count, reliable_count=reliable_count)
        depth_safe_count = int(np.count_nonzero(depth_safety & corridor_mask))
        safety &= depth_safety
    if additional_safety_mask is not None:
        supplied = np.asarray(additional_safety_mask, dtype=bool)
        if supplied.shape != overlap.shape:
            raise ValueError("additional_safety_mask must match the placed source canvas")
        safety &= supplied

    x0, y0, x1, y1 = corridor
    border = int(settings.support_border_pixels)
    if x1 - x0 <= 2 * border or y1 - y0 <= 2 * border:
        return _empty(first, second, settings, reason="corridor_too_small_for_boundary_pinned_mesh", corridor=corridor, overlap_count=overlap_count, reliable_count=reliable_count, depth_safe_count=depth_safe_count)
    inner = np.zeros(overlap.shape, dtype=bool)
    inner[y0 + border : y1 - border, x0 + border : x1 - border] = True
    same_layer = reliable & safety & inner
    safe_count = int(np.count_nonzero(same_layer))
    step = int(settings.correspondence_spacing_pixels)
    candidates = same_layer & (np.mod(np.arange(overlap.shape[0])[:, None] - y0, step) == 0) & (np.mod(np.arange(overlap.shape[1])[None, :] - x0, step) == 0)
    y, x = np.nonzero(candidates)
    source = np.column_stack((x.astype(np.float64), y.astype(np.float64)))
    output = source + forward[y, x].astype(np.float64)
    bounds = TileBounds(float(x0), float(y0), float(x1 - 1), float(y1 - 1))
    in_bounds = bounds.contains(output)
    output, source = output[in_bounds], source[in_bounds]
    # The fitter checks the layer at both q and p.  Marking only the conservative
    # interior safety domain prevents it from crossing alpha/depth protections.
    fit = fit_local_mesh_inverse_warp(
        output,
        source,
        bounds,
        same_layer_mask=same_layer,
        same_layer_origin_xy=(0.0, 0.0),
        fit_support_mask=same_layer,
        fit_support_origin_xy=(0.0, 0.0),
        config=settings.mesh,
    )
    rejection = None if fit.audit.accepted else fit.audit.reason
    return LocalMeshEvidenceResult(
        fit=fit,
        same_layer_mask=np.ascontiguousarray(same_layer),
        output_points_xy=np.ascontiguousarray(output),
        source_points_xy=np.ascontiguousarray(source),
        audit=LocalMeshEvidenceAudit(
            first.frame_id, second.frame_id, settings.flow_backend, corridor, overlap_count,
            reliable_count, depth_safe_count, safe_count, int(np.count_nonzero(candidates)),
            int(len(output)), rejection, fit.audit, settings.require_depth_safety,
        ),
    )


def sample_accepted_mesh_from_first_source(
    first: VideoVisualSource,
    evidence: LocalMeshEvidenceResult,
    *,
    owner_mask: np.ndarray,
) -> LocalMeshOutputSampling:
    """Inverse-sample only real first-source pixels in the accepted safe layer.

    ``owner_mask`` is supplied by the caller's hard seam and ensures mesh
    sampling cannot alter provenance.  We require a non-identity accepted
    inverse displacement plus fully opaque source alpha, so the output never
    fills a hole or turns transparency into RGB content.
    """

    image = np.asarray(first.bgra)
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("first source must be HxWx4 BGRA")
    requested = np.asarray(owner_mask, dtype=bool)
    if requested.shape != image.shape[:2]:
        raise ValueError("owner_mask must match the placed source canvas")
    result = np.zeros(image.shape[:2], dtype=bool)
    bgr = np.asarray(image[..., :3]).copy()
    if not evidence.accepted or evidence.fit is None or evidence.fit.warp is None:
        return LocalMeshOutputSampling(bgr=bgr, applied_mask=result)
    safe = np.asarray(evidence.same_layer_mask, dtype=bool)
    if safe.shape != result.shape:
        raise ValueError("mesh same-layer mask must match the placed source canvas")
    yy, xx = np.indices(result.shape, dtype=np.float64)
    mapped_x, mapped_y = evidence.fit.warp.inverse_coordinates(xx, yy)
    displacement = np.hypot(mapped_x - xx, mapped_y - yy)
    sampled = cv2.remap(
        image,
        mapped_x.astype(np.float32),
        mapped_y.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    # A successful mesh's own inverse_coordinates implementation protects both
    # output and input layer membership.  The explicit opaque check prevents
    # interpolation across alpha support regardless of OpenCV border behaviour.
    result = requested & safe & (displacement > 1e-3) & (sampled[..., 3] == 255)
    bgr[result] = sampled[..., :3][result]
    return LocalMeshOutputSampling(bgr=np.ascontiguousarray(bgr), applied_mask=result)


__all__ = [
    "FlowEstimator",
    "LocalMeshEvidenceAudit",
    "LocalMeshEvidenceConfig",
    "LocalMeshEvidenceResult",
    "LocalMeshOutputSampling",
    "assess_candidate_local_mesh_evidence",
    "sample_accepted_mesh_from_first_source",
]
