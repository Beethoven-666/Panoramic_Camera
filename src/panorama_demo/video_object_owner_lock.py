"""Candidate-only C5 foreground-object owner constraints.

This module is deliberately a planning and provenance utility.  It consumes
two *already placed* real sources plus an optional object mask, derives a
conservative protected domain, and pins that domain to one real source frame.
It never samples, blends, or returns RGB pixels; it also has no pose, session,
or filesystem dependency.  A renderer can apply the returned owner map to its
own hard-owner decision, but must continue to copy colour from the selected
real source verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .video_visual_renderer import VideoVisualSource


_INVALID_OWNER = -1


@dataclass(frozen=True)
class ObjectOwnerLockConfig:
    """Closed C5 limits for one adjacent pair's foreground-owner decision."""

    object_guard_pixels: int = 3
    depth_edge_guard_pixels: int = 3
    depth_absolute_tolerance_mm: float = 20.0
    depth_relative_tolerance: float = 0.02
    maximum_handoffs: int = 1

    def __post_init__(self) -> None:
        if not 0 <= int(self.object_guard_pixels) <= 32:
            raise ValueError("object_guard_pixels must be in [0, 32]")
        if not 0 <= int(self.depth_edge_guard_pixels) <= 32:
            raise ValueError("depth_edge_guard_pixels must be in [0, 32]")
        if not float(self.depth_absolute_tolerance_mm) > 0.0:
            raise ValueError("depth_absolute_tolerance_mm must be positive")
        if not 0.0 < float(self.depth_relative_tolerance) <= 1.0:
            raise ValueError("depth_relative_tolerance must be in (0, 1]")
        if not 0 <= int(self.maximum_handoffs) <= 1:
            raise ValueError("maximum_handoffs must be either 0 or 1")


@dataclass(frozen=True)
class ObjectOwnerLockAudit:
    """Auditable proof that C5 only constrained real source ownership."""

    first_frame_id: int
    second_frame_id: int
    locked_owner_frame_id: int | None
    previous_owner_frame_id: int | None
    previous_handoff_count: int
    resulting_handoff_count: int
    maximum_handoffs: int
    object_input_pixel_count: int
    protected_object_pixel_count: int
    protected_depth_edge_pixel_count: int
    protected_pixel_count: int
    selected_owner_coverage_pixel_count: int
    accepted: bool
    rejection_reason: str | None
    method: str = "candidate_object_owner_lock/v1"

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "first_frame_id": self.first_frame_id,
            "second_frame_id": self.second_frame_id,
            "locked_owner_frame_id": self.locked_owner_frame_id,
            "previous_owner_frame_id": self.previous_owner_frame_id,
            "previous_handoff_count": self.previous_handoff_count,
            "resulting_handoff_count": self.resulting_handoff_count,
            "maximum_handoffs": self.maximum_handoffs,
            "object_input_pixel_count": self.object_input_pixel_count,
            "protected_object_pixel_count": self.protected_object_pixel_count,
            "protected_depth_edge_pixel_count": self.protected_depth_edge_pixel_count,
            "protected_pixel_count": self.protected_pixel_count,
            "selected_owner_coverage_pixel_count": self.selected_owner_coverage_pixel_count,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "creates_colour": False,
            "creates_pose": False,
            "interpolates_source_frames": False,
            "real_adjacent_sources_only": True,
            "single_real_owner": self.accepted,
        }


@dataclass(frozen=True)
class ObjectOwnerLockPlan:
    """One protected pair domain and the owner it is pinned to, if accepted."""

    protected_object_mask: np.ndarray
    protected_depth_edge_mask: np.ndarray
    protected_mask: np.ndarray
    owner_frame_id: np.ndarray
    audit: ObjectOwnerLockAudit

    @property
    def accepted(self) -> bool:
        return self.audit.accepted


@dataclass(frozen=True)
class DepthConnectedObjectMask:
    """Conservative O1 foreground candidates from placed aligned depth only."""

    mask: np.ndarray
    candidate_pixel_count: int
    component_count: int
    depth_threshold_mm: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "candidate_o1_depth_connected_components/v1",
            "candidate_pixel_count": self.candidate_pixel_count,
            "component_count": self.component_count,
            "depth_threshold_mm": self.depth_threshold_mm,
            "uses_rgb": False,
            "uses_pose": False,
            "uses_only_aligned_depth": True,
        }


def _valid(source: VideoVisualSource) -> np.ndarray:
    return np.asarray(source.bgra)[..., 3] > 0


def _expand(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0 or not np.any(mask):
        return np.asarray(mask, dtype=bool).copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    return cv2.dilate(np.asarray(mask, dtype=np.uint8), kernel).astype(bool)


def _coerce_object_mask(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError("object_mask must match the placed source canvas")
    if mask.ndim != 2:
        raise ValueError("object_mask must be a 2-D mask")
    if mask.dtype != bool and not np.issubdtype(mask.dtype, np.number):
        raise ValueError("object_mask must be boolean or numeric")
    if np.issubdtype(mask.dtype, np.floating) and not np.isfinite(mask).all():
        raise ValueError("object_mask must be finite")
    return np.asarray(mask != 0, dtype=bool)


def _depth_edge_mask(source: VideoVisualSource, config: ObjectOwnerLockConfig) -> np.ndarray:
    """Mark depth holes and discontinuities; no reprojection or pose is used."""

    shape = np.asarray(source.bgra).shape[:2]
    if source.depth_mm is None:
        return np.zeros(shape, dtype=bool)
    depth = np.asarray(source.depth_mm, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    edge = ~valid
    tolerance = np.maximum(
        float(config.depth_absolute_tolerance_mm),
        np.abs(depth) * float(config.depth_relative_tolerance),
    )
    horizontal = valid[:, 1:] & valid[:, :-1] & (
        np.abs(depth[:, 1:] - depth[:, :-1])
        > np.maximum(tolerance[:, 1:], tolerance[:, :-1])
    )
    vertical = valid[1:, :] & valid[:-1, :] & (
        np.abs(depth[1:, :] - depth[:-1, :])
        > np.maximum(tolerance[1:, :], tolerance[:-1, :])
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return _expand(edge, int(config.depth_edge_guard_pixels))


def depth_connected_object_candidates(
    first: VideoVisualSource,
    second: VideoVisualSource,
    *,
    minimum_component_pixels: int = 64,
    near_depth_quantile: float = 0.30,
) -> DepthConnectedObjectMask:
    """Find bounded near-depth components in common real-source support.

    This is the C5 O1 option: no RGB segmentation, learned model, pose or
    synthetic correspondence is involved.  A component must be near relative
    to the pair's common finite depth and connected at the placed canvas.  It
    is merely a *candidate* for a later all-real-owner coverage check.
    """

    if int(minimum_component_pixels) < 1:
        raise ValueError("minimum_component_pixels must be positive")
    if not 0.0 < float(near_depth_quantile) < 1.0:
        raise ValueError("near_depth_quantile must be in (0, 1)")
    shape = np.asarray(first.bgra).shape[:2]
    empty = np.zeros(shape, dtype=bool)
    if first.depth_mm is None or second.depth_mm is None:
        return DepthConnectedObjectMask(empty, 0, 0, None)
    first_depth = np.asarray(first.depth_mm, dtype=np.float32)
    second_depth = np.asarray(second.depth_mm, dtype=np.float32)
    if first_depth.shape != shape or second_depth.shape != shape:
        raise ValueError("depth connected components require matching placed depth maps")
    common = _valid(first) & _valid(second)
    finite = common & np.isfinite(first_depth) & np.isfinite(second_depth) & (first_depth > 0) & (second_depth > 0)
    if int(np.count_nonzero(finite)) < int(minimum_component_pixels):
        return DepthConnectedObjectMask(empty, 0, 0, None)
    # Both observations must be locally near.  Require a visible depth-layer
    # gap before accepting a quantile: otherwise one broad planar wall would
    # become a spurious "object" merely because it occupies the near tail.
    values = np.sort(np.maximum(first_depth[finite], second_depth[finite]))
    difference = np.diff(values)
    layer_floor = np.maximum(20.0, values[:-1] * 0.02)
    allowed_rank = max(1, int(np.floor(values.size * float(near_depth_quantile))))
    candidates = np.flatnonzero((difference > layer_floor) & (np.arange(difference.size) < allowed_rank))
    if not candidates.size:
        return DepthConnectedObjectMask(empty, 0, 0, None)
    boundary = int(candidates[np.argmax(difference[candidates])])
    threshold = float((values[boundary] + values[boundary + 1]) * 0.5)
    candidate = finite & (first_depth <= threshold) & (second_depth <= threshold)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(candidate)
    component_count = 0
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) < int(minimum_component_pixels):
            continue
        kept[labels == label] = True
        component_count += 1
    return DepthConnectedObjectMask(
        np.ascontiguousarray(kept), int(np.count_nonzero(kept)), component_count, threshold
    )


def _empty_plan(
    first: VideoVisualSource,
    second: VideoVisualSource,
    *,
    config: ObjectOwnerLockConfig,
    protected_object: np.ndarray,
    protected_depth: np.ndarray,
    previous_owner_frame_id: int | None,
    previous_handoff_count: int,
    reason: str,
) -> ObjectOwnerLockPlan:
    shape = np.asarray(first.bgra).shape[:2]
    protected = protected_object | protected_depth
    return ObjectOwnerLockPlan(
        protected_object_mask=protected_object,
        protected_depth_edge_mask=protected_depth,
        protected_mask=protected,
        owner_frame_id=np.full(shape, _INVALID_OWNER, dtype=np.int32),
        audit=ObjectOwnerLockAudit(
            first_frame_id=first.frame_id,
            second_frame_id=second.frame_id,
            locked_owner_frame_id=None,
            previous_owner_frame_id=previous_owner_frame_id,
            previous_handoff_count=previous_handoff_count,
            resulting_handoff_count=previous_handoff_count,
            maximum_handoffs=int(config.maximum_handoffs),
            object_input_pixel_count=0,
            protected_object_pixel_count=int(np.count_nonzero(protected_object)),
            protected_depth_edge_pixel_count=int(np.count_nonzero(protected_depth)),
            protected_pixel_count=int(np.count_nonzero(protected)),
            selected_owner_coverage_pixel_count=0,
            accepted=False,
            rejection_reason=reason,
        ),
    )


def plan_object_owner_lock(
    first: VideoVisualSource,
    second: VideoVisualSource,
    *,
    object_mask: np.ndarray | None = None,
    constraint_mask: np.ndarray | None = None,
    previous_owner_frame_id: int | None = None,
    previous_handoff_count: int = 0,
    preferred_owner_frame_id: int | None = None,
    config: ObjectOwnerLockConfig | None = None,
) -> ObjectOwnerLockPlan:
    """Create a fail-closed single-owner constraint for an adjacent real pair.

    A protected pixel is only locked when *one* chosen source covers the whole
    protected domain.  If that cannot be proven, the plan is rejected instead
    of splitting an object across sources.  ``previous_owner_frame_id`` makes
    the one permitted owner transfer explicit and auditable.
    """

    settings = config or ObjectOwnerLockConfig()
    if first.frame_id == second.frame_id:
        raise ValueError("object owner lock requires two distinct real source frame ids")
    first_image, second_image = np.asarray(first.bgra), np.asarray(second.bgra)
    if first_image.shape != second_image.shape:
        raise ValueError("adjacent object-lock sources must share one placed BGRA canvas")
    if previous_handoff_count < 0:
        raise ValueError("previous_handoff_count must be non-negative")
    source_ids = (int(first.frame_id), int(second.frame_id))
    if previous_owner_frame_id is not None and int(previous_owner_frame_id) not in source_ids:
        raise ValueError("previous_owner_frame_id must name one adjacent real source")
    if preferred_owner_frame_id is not None and int(preferred_owner_frame_id) not in source_ids:
        raise ValueError("preferred_owner_frame_id must name one adjacent real source")

    shape = first_image.shape[:2]
    support = _valid(first) | _valid(second)
    if constraint_mask is not None:
        constraint = _coerce_object_mask(constraint_mask, shape)
        support &= constraint
    raw_object = _coerce_object_mask(object_mask, shape)
    protected_object = _expand(raw_object, int(settings.object_guard_pixels)) & support
    protected_depth = (_depth_edge_mask(first, settings) | _depth_edge_mask(second, settings)) & support
    protected = protected_object | protected_depth
    if not np.any(protected):
        return _empty_plan(
            first, second, config=settings, protected_object=protected_object,
            protected_depth=protected_depth, previous_owner_frame_id=previous_owner_frame_id,
            previous_handoff_count=previous_handoff_count, reason="no_object_or_depth_edge_protection",
        )

    # Retaining an established owner is the default.  A different preferred
    # owner is an explicit, auditable handoff request; defaulting to the older
    # source when there is no history avoids a hidden colour/quality score.
    chosen = (
        int(preferred_owner_frame_id)
        if preferred_owner_frame_id is not None
        else int(previous_owner_frame_id)
        if previous_owner_frame_id is not None
        else int(first.frame_id)
    )
    chosen_valid = _valid(first) if chosen == first.frame_id else _valid(second)
    coverage = int(np.count_nonzero(chosen_valid & protected))
    if coverage != int(np.count_nonzero(protected)):
        return _empty_plan(
            first, second, config=settings, protected_object=protected_object,
            protected_depth=protected_depth, previous_owner_frame_id=previous_owner_frame_id,
            previous_handoff_count=previous_handoff_count,
            reason="selected_real_owner_does_not_cover_protected_domain",
        )
    resulting_handoffs = int(previous_handoff_count) + int(
        previous_owner_frame_id is not None and int(previous_owner_frame_id) != chosen
    )
    if resulting_handoffs > int(settings.maximum_handoffs):
        return _empty_plan(
            first, second, config=settings, protected_object=protected_object,
            protected_depth=protected_depth, previous_owner_frame_id=previous_owner_frame_id,
            previous_handoff_count=previous_handoff_count, reason="maximum_object_handoffs_exceeded",
        )
    owner = np.full(shape, _INVALID_OWNER, dtype=np.int32)
    owner[protected] = chosen
    return ObjectOwnerLockPlan(
        protected_object_mask=np.ascontiguousarray(protected_object),
        protected_depth_edge_mask=np.ascontiguousarray(protected_depth),
        protected_mask=np.ascontiguousarray(protected),
        owner_frame_id=owner,
        audit=ObjectOwnerLockAudit(
            first_frame_id=first.frame_id,
            second_frame_id=second.frame_id,
            locked_owner_frame_id=chosen,
            previous_owner_frame_id=previous_owner_frame_id,
            previous_handoff_count=int(previous_handoff_count),
            resulting_handoff_count=resulting_handoffs,
            maximum_handoffs=int(settings.maximum_handoffs),
            object_input_pixel_count=int(np.count_nonzero(raw_object)),
            protected_object_pixel_count=int(np.count_nonzero(protected_object)),
            protected_depth_edge_pixel_count=int(np.count_nonzero(protected_depth)),
            protected_pixel_count=int(np.count_nonzero(protected)),
            selected_owner_coverage_pixel_count=coverage,
            accepted=True,
            rejection_reason=None,
        ),
    )


def enforce_object_owner_lock(
    proposed_owner_frame_id: np.ndarray,
    plan: ObjectOwnerLockPlan,
) -> np.ndarray:
    """Return an owner-only constraint application without touching colour.

    Only protected pixels are overwritten.  An invalid/rejected plan cannot be
    silently applied, and all constrained owner values remain a real adjacent
    source id recorded by the plan.
    """

    if not plan.accepted or plan.audit.locked_owner_frame_id is None:
        raise ValueError("cannot enforce a rejected object owner-lock plan")
    proposed = np.asarray(proposed_owner_frame_id)
    if proposed.shape != plan.protected_mask.shape or proposed.ndim != 2:
        raise ValueError("proposed_owner_frame_id must match the placed source canvas")
    if not np.issubdtype(proposed.dtype, np.integer):
        raise ValueError("proposed_owner_frame_id must contain integer frame ids")
    result = np.asarray(proposed, dtype=np.int32).copy()
    result[plan.protected_mask] = int(plan.audit.locked_owner_frame_id)
    assert_object_owner_lock(result, plan)
    return result


def assert_object_owner_lock(owner_frame_id: np.ndarray, plan: ObjectOwnerLockPlan) -> None:
    """Raise unless every protected pixel has exactly the approved real owner."""

    if not plan.accepted or plan.audit.locked_owner_frame_id is None:
        raise ValueError("object owner-lock plan was not accepted")
    owner = np.asarray(owner_frame_id)
    if owner.shape != plan.protected_mask.shape or owner.ndim != 2:
        raise ValueError("owner_frame_id must match the placed source canvas")
    if not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("owner_frame_id must contain integer frame ids")
    if not np.all(owner[plan.protected_mask] == int(plan.audit.locked_owner_frame_id)):
        raise ValueError("protected object/depth-edge pixels must retain one real owner")


__all__ = [
    "ObjectOwnerLockAudit",
    "ObjectOwnerLockConfig",
    "ObjectOwnerLockPlan",
    "DepthConnectedObjectMask",
    "assert_object_owner_lock",
    "depth_connected_object_candidates",
    "enforce_object_owner_lock",
    "plan_object_owner_lock",
]
