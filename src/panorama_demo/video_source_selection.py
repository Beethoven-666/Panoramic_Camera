"""Frontality evidence for real video render-source selection.

This module never creates a frame or a pose.  It converts the calibrated
colour intrinsics into per-source spans which the v6 owner planner may use;
the planner, rather than a fixed centre-strip constant, decides each actual
owner extent.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, degrees, radians, tan
from typing import Sequence

import numpy as np

from .session import CameraIntrinsics, RGBDFrame


@dataclass(frozen=True)
class VideoFrontalityConfig:
    """Frozen v6 off-axis limits for near and general owner support."""

    near_target_degrees: float = 4.0
    near_hard_degrees: float = 7.0
    general_target_degrees: float = 6.0
    general_hard_degrees: float = 10.0

    def __post_init__(self) -> None:
        values = (
            self.near_target_degrees,
            self.near_hard_degrees,
            self.general_target_degrees,
            self.general_hard_degrees,
        )
        if not np.isfinite(values).all() or any(value <= 0.0 or value >= 89.0 for value in values):
            raise ValueError("frontality angles must be finite values in (0, 89) degrees")
        if self.near_target_degrees > self.near_hard_degrees:
            raise ValueError("near frontality requires target <= hard")
        if self.general_target_degrees > self.general_hard_degrees:
            raise ValueError("general frontality requires target <= hard")


@dataclass(frozen=True)
class VideoSourceFrontality:
    """Calibrated source columns valid at each v6 off-axis limit."""

    frame_id: int
    near_target_span: tuple[int, int]
    near_hard_span: tuple[int, int]
    general_target_span: tuple[int, int]
    general_hard_span: tuple[int, int]
    frontality_score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "near_target_span": list(self.near_target_span),
            "near_hard_span": list(self.near_hard_span),
            "general_target_span": list(self.general_target_span),
            "general_hard_span": list(self.general_hard_span),
            # The normal planner starts in the target span; use the hard span
            # only as an explicit warning/reroute boundary.
            "valid_frontality_span": list(self.general_target_span),
            "frontality_score": self.frontality_score,
        }


@dataclass(frozen=True)
class FrontalityOwnerPlan:
    """Monotone owner intervals derived from real source centres and FOV."""

    source_frame_ids: tuple[int, ...]
    source_centres_x: tuple[float, ...]
    owner_intervals_x: tuple[tuple[float, float], ...]
    target_span_exceeded_frame_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_frame_ids": list(self.source_frame_ids),
            "source_centres_x": list(self.source_centres_x),
            "owner_intervals_x": [list(value) for value in self.owner_intervals_x],
            "target_span_exceeded_frame_ids": list(self.target_span_exceeded_frame_ids),
            "fixed_owner_pixel_width": None,
            "monotone_real_source_owner_plan": True,
        }


def _span_for_angle(calibration: CameraIntrinsics, maximum_degrees: float) -> tuple[int, int]:
    half_width = float(calibration.fx) * tan(radians(maximum_degrees))
    left = max(0, int(np.ceil(float(calibration.cx) - half_width)))
    right = min(calibration.width, int(np.floor(float(calibration.cx) + half_width)) + 1)
    if right <= left:
        raise ValueError("frontality span has no calibrated source columns")
    return left, right


def off_axis_angle_degrees(calibration: CameraIntrinsics, x: float) -> float:
    """Return ``atan((x-cx)/fx)`` in degrees for one source column."""

    if not np.isfinite(x):
        raise ValueError("source column must be finite")
    return degrees(atan((float(x) - float(calibration.cx)) / float(calibration.fx)))


def assess_video_source_frontality(
    frames: Sequence[RGBDFrame],
    calibration: CameraIntrinsics,
    *,
    config: VideoFrontalityConfig | None = None,
) -> tuple[VideoSourceFrontality, ...]:
    """Return one identically calibrated, real-source span record per frame."""

    sources = tuple(frames)
    if len(sources) < 2:
        raise ValueError("Video source frontality requires at least two real frames")
    ids = tuple(int(frame.frame_id) for frame in sources)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("Video source frame ids must be unique and chronological")
    settings = config or VideoFrontalityConfig()
    near_target = _span_for_angle(calibration, settings.near_target_degrees)
    near_hard = _span_for_angle(calibration, settings.near_hard_degrees)
    general_target = _span_for_angle(calibration, settings.general_target_degrees)
    general_hard = _span_for_angle(calibration, settings.general_hard_degrees)
    target_width = general_target[1] - general_target[0]
    hard_width = general_hard[1] - general_hard[0]
    score = target_width / hard_width
    return tuple(
        VideoSourceFrontality(
            frame_id=frame_id,
            near_target_span=near_target,
            near_hard_span=near_hard,
            general_target_span=general_target,
            general_hard_span=general_hard,
            frontality_score=score,
        )
        for frame_id in ids
    )


def plan_frontality_owner_spans(
    frontality: Sequence[VideoSourceFrontality],
    calibration: CameraIntrinsics,
    source_centres_x: Sequence[float],
) -> FrontalityOwnerPlan:
    """Allocate dynamic monotone owners inside each source's hard FOV span.

    ``source_centres_x`` is the real-source scan layout in output pixels.  No
    synthetic position is introduced: the caller derives it from measured
    adjacent RGB progress and direct ORB camera centres.
    """

    records = tuple(frontality)
    centres = tuple(float(value) for value in source_centres_x)
    if len(records) < 2 or len(records) != len(centres):
        raise ValueError("frontality owner plan requires aligned real source records")
    if not np.isfinite(centres).all() or any(right <= left for left, right in zip(centres, centres[1:])):
        raise ValueError("frontality owner centres must be finite and strictly increasing")
    hard_left = np.asarray(
        [centre + record.general_hard_span[0] - calibration.cx for centre, record in zip(centres, records, strict=True)],
        dtype=np.float64,
    )
    hard_right = np.asarray(
        [centre + record.general_hard_span[1] - calibration.cx for centre, record in zip(centres, records, strict=True)],
        dtype=np.float64,
    )
    boundaries = [float(hard_left[0])]
    for index in range(len(records) - 1):
        overlap_left = max(float(hard_left[index]), float(hard_left[index + 1]))
        overlap_right = min(float(hard_right[index]), float(hard_right[index + 1]))
        if overlap_right <= overlap_left:
            raise ValueError(
                "adjacent real sources have no common hard-frontality support; "
                "select a denser direct-ORB source"
            )
        midpoint = 0.5 * (centres[index] + centres[index + 1])
        boundaries.append(float(np.clip(midpoint, overlap_left, overlap_right)))
    boundaries.append(float(hard_right[-1]))
    intervals = tuple(
        (boundaries[index], boundaries[index + 1]) for index in range(len(records))
    )
    if any(right <= left for left, right in intervals):
        raise ValueError("frontality owner interval collapsed")
    exceeded: list[int] = []
    for record, centre, (left, right) in zip(records, centres, intervals, strict=True):
        target_left = centre + record.general_target_span[0] - calibration.cx
        target_right = centre + record.general_target_span[1] - calibration.cx
        if left < target_left or right > target_right:
            exceeded.append(record.frame_id)
    return FrontalityOwnerPlan(
        source_frame_ids=tuple(record.frame_id for record in records),
        source_centres_x=centres,
        owner_intervals_x=intervals,
        target_span_exceeded_frame_ids=tuple(exceeded),
    )


def select_video_render_sources(frames: tuple[RGBDFrame, ...]) -> tuple[RGBDFrame, ...]:
    """Keep real ORB-tracked nodes chronological; owner widths stay dynamic."""

    if len(frames) < 2:
        raise ValueError("Video render requires at least two real source frames")
    ids = [frame.frame_id for frame in frames]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Video source frame ids must be unique and chronological")
    return frames
