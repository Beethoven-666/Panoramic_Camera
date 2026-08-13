"""Direct-ORB object coverage and wide-patch planning for video v6.

Object regions are explicit inputs from a later RGB object/line guard.  This
module does not segment, interpolate, or manufacture sources: it simply finds
the smallest continuous run of already direct-ORB sources whose calibrated
hard-frontality support covers an object plus its context collar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class VideoObjectPlanningConfig:
    """Frozen source/rescue limits and development context-collar defaults."""

    base_source_hard_maximum: int = 44
    rescue_per_seam: int = 1
    rescue_session: int = 4
    final_sources_maximum: int = 48
    large_object_collar_px: tuple[int, int] = (8, 20)
    regular_object_collar_px: tuple[int, int] = (6, 16)
    thin_object_collar_px: tuple[int, int] = (4, 10)

    def __post_init__(self) -> None:
        if not (1 <= self.base_source_hard_maximum <= self.final_sources_maximum):
            raise ValueError("base source maximum must be positive and no greater than final maximum")
        if self.rescue_per_seam != 1 or not 0 <= self.rescue_session <= 4:
            raise ValueError("v6 permits one rescue per seam and at most four per session")
        for collar in (self.large_object_collar_px, self.regular_object_collar_px, self.thin_object_collar_px):
            if len(collar) != 2 or not 0 < collar[0] <= collar[1]:
                raise ValueError("object context collar must be an increasing positive [min, max] range")


@dataclass(frozen=True)
class VideoObjectRegion:
    """A guard-produced canvas interval; no implicit semantic segmentation."""

    object_id: str
    span_x: tuple[float, float]
    collar_px: int

    def __post_init__(self) -> None:
        left, right = (float(value) for value in self.span_x)
        if not self.object_id or not np.isfinite((left, right)).all() or right <= left:
            raise ValueError("object region needs a non-empty finite canvas span")
        if self.collar_px < 0:
            raise ValueError("object context collar cannot be negative")

    @property
    def protected_span_x(self) -> tuple[float, float]:
        return float(self.span_x[0] - self.collar_px), float(self.span_x[1] + self.collar_px)


@dataclass(frozen=True)
class VideoDirectSourceSupport:
    """A true source's output-canvas hard-frontality support interval."""

    frame_id: int
    support_x: tuple[float, float]
    direct_orb: bool = True

    def __post_init__(self) -> None:
        left, right = (float(value) for value in self.support_x)
        if not self.direct_orb:
            raise ValueError("object planning cannot accept a non-direct-ORB source")
        if not np.isfinite((left, right)).all() or right <= left:
            raise ValueError("source support must be a non-empty finite interval")


@dataclass(frozen=True)
class VideoObjectPatchPlan:
    """The minimal necessary continuous source run for one protected object."""

    object_id: str
    requested_span_x: tuple[float, float]
    source_frame_ids: tuple[int, ...]
    initial_n_req: int
    final_replanned_n_req: int
    category: str
    geometry_patch_count: int
    approved_rescue_frame_ids: tuple[int, ...]
    redundant_geometry_patch_count: int
    small_fragment_count: int
    patch_island_count: int
    replan_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "requested_span_x": list(self.requested_span_x),
            "source_frame_ids": list(self.source_frame_ids),
            "initial_N_req": self.initial_n_req,
            "final_replanned_N_req": self.final_replanned_n_req,
            "category": self.category,
            "geometry_patch_count": self.geometry_patch_count,
            "approved_rescue_frame_ids": list(self.approved_rescue_frame_ids),
            "redundant_geometry_patch_count": self.redundant_geometry_patch_count,
            "small_fragment_count": self.small_fragment_count,
            "patch_island_count": self.patch_island_count,
            "replan_reason": self.replan_reason,
        }


@dataclass(frozen=True)
class VideoTrackingSourcePlan:
    """One T0/T1/T2 direct-ORB candidate's real source support."""

    tracking_candidate: str
    sources: tuple[VideoDirectSourceSupport, ...]

    def __post_init__(self) -> None:
        if self.tracking_candidate not in {"T0", "T1", "T2"}:
            raise ValueError("tracking candidate must be T0, T1, or T2")
        ids = tuple(source.frame_id for source in self.sources)
        if len(ids) < 1 or ids != tuple(sorted(set(ids))):
            raise ValueError("tracking sources must be unique and chronological")
        starts = [source.support_x[0] for source in self.sources]
        if any(right <= left for left, right in zip(starts, starts[1:])):
            raise ValueError("source supports must retain direct-ORB chronology")


def _category(n_req: int) -> str:
    return {1: "compact", 2: "wide", 3: "very_wide"}.get(n_req, "oversized")


def _minimal_continuous_cover(
    sources: Sequence[VideoDirectSourceSupport], protected_span: tuple[float, float]
) -> tuple[VideoDirectSourceSupport, ...] | None:
    """Return the smallest chronological support run with no coverage hole."""

    left, right = protected_span
    best: tuple[VideoDirectSourceSupport, ...] | None = None
    for start in range(len(sources)):
        if sources[start].support_x[0] > left:
            continue
        covered_right = sources[start].support_x[1]
        for end in range(start, len(sources)):
            if end > start:
                source = sources[end]
                if source.support_x[0] > covered_right:
                    break
                covered_right = max(covered_right, source.support_x[1])
            if covered_right >= right:
                candidate = tuple(sources[start : end + 1])
                if best is None or len(candidate) < len(best):
                    best = candidate
                break
    return best


def plan_object_patches(
    region: VideoObjectRegion, sources: Sequence[VideoDirectSourceSupport], *, initial_n_req: int | None = None,
    approved_rescue_frame_ids: Sequence[int] = (), config: VideoObjectPlanningConfig | None = None,
    replan_reason: str | None = None,
) -> VideoObjectPatchPlan:
    """Compute N_req without a fixed strip width or fragmented patch fallback."""

    settings = config or VideoObjectPlanningConfig()
    sequence = tuple(sources)
    if len(sequence) > settings.base_source_hard_maximum:
        raise ValueError("base direct source count exceeds v6 maximum")
    cover = _minimal_continuous_cover(sequence, region.protected_span_x)
    if cover is None:
        raise RuntimeError("no continuous direct-ORB frontality cover for protected object span")
    rescues = tuple(int(frame_id) for frame_id in approved_rescue_frame_ids)
    if len(rescues) > settings.rescue_per_seam or len(set(rescues)) != len(rescues):
        raise ValueError("object plan permits at most one distinct rescue source per seam")
    if any(frame_id not in {source.frame_id for source in sequence} for frame_id in rescues):
        raise ValueError("approved rescue source must be an existing direct-ORB source")
    n_req = len(cover)
    initial = n_req if initial_n_req is None else int(initial_n_req)
    if initial < 1:
        raise ValueError("initial N_req must be positive")
    patch_count = n_req + len(rescues)
    return VideoObjectPatchPlan(
        object_id=region.object_id, requested_span_x=region.protected_span_x,
        source_frame_ids=tuple(source.frame_id for source in cover), initial_n_req=initial,
        final_replanned_n_req=n_req, category=_category(n_req), geometry_patch_count=patch_count,
        approved_rescue_frame_ids=rescues, redundant_geometry_patch_count=0,
        small_fragment_count=0, patch_island_count=0, replan_reason=replan_reason,
    )


def replan_wide_object_patches(
    region: VideoObjectRegion, candidates: Mapping[str, VideoTrackingSourcePlan], *,
    config: VideoObjectPlanningConfig | None = None,
) -> tuple[str, VideoObjectPatchPlan]:
    """Use T0 first, then denser T1/T2 only when N_req is oversized.

    The output never inserts a source itself.  A caller may separately approve
    one already-direct source as a rescue after GraphCut cannot reroute safely.
    """

    settings = config or VideoObjectPlanningConfig()
    plans = {name: candidates[name] for name in ("T0", "T1", "T2") if name in candidates}
    if "T0" not in plans:
        raise ValueError("wide-object replanning requires the T0 direct-ORB plan")
    initial = plan_object_patches(region, plans["T0"].sources, config=settings)
    if initial.initial_n_req <= 3:
        return "T0", initial
    last_error: Exception | None = None
    for name in ("T1", "T2"):
        if name not in plans:
            continue
        try:
            replanned = plan_object_patches(
                region, plans[name].sources, initial_n_req=initial.initial_n_req, config=settings,
                replan_reason="initial_N_req_gt_3_reselected_denser_direct_orb_tracking",
            )
            return name, replanned
        except RuntimeError as error:
            last_error = error
    if last_error is not None:
        raise RuntimeError("no denser direct-ORB plan continuously covers oversized object") from last_error
    # A genuinely wide object may remain oversized.  This is allowed only as a
    # contiguous large-patch run, never as many small fragments.
    return "T0", VideoObjectPatchPlan(
        **{**initial.__dict__, "replan_reason": "N_req_gt_3_no_denser_tracking_candidate_available"}
    )


__all__ = [
    "VideoDirectSourceSupport", "VideoObjectPatchPlan", "VideoObjectPlanningConfig", "VideoObjectRegion",
    "VideoTrackingSourcePlan", "plan_object_patches", "replan_wide_object_patches",
]
