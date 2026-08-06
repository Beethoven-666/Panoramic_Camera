"""Motion-driven selection of real video render sources.

The selector deliberately never creates a frame or a pose.  It only reduces a
60 FPS capture to a chronological subset of actual RGB-D frames after the
complete ORB-SLAM3 chain has been obtained.  This keeps the video contract
intact while avoiding the cost of rendering nearly-identical views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .quality import FrameQuality, MotionEstimate
from .session import RGBDFrame


@dataclass(frozen=True)
class MotionResamplingConfig:
    """Bounds for selecting real render frames from measured image motion."""

    minimum_step_pixels: float = 3.0
    normal_target_step_pixels: float = 16.0
    risk_target_step_pixels: float = 8.0
    maximum_step_pixels: float = 24.0
    emergency_step_pixels: float = 30.0

    @classmethod
    def from_mapping(cls, value: object | None) -> "MotionResamplingConfig":
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("motion_resampling must be a mapping")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown motion_resampling keys: {unknown}")
        config = cls(**value)
        values = (
            config.minimum_step_pixels,
            config.risk_target_step_pixels,
            config.normal_target_step_pixels,
            config.maximum_step_pixels,
            config.emergency_step_pixels,
        )
        if not np.isfinite(values).all() or config.minimum_step_pixels <= 0.0:
            raise ValueError("motion_resampling steps must be finite and positive")
        if not (
            config.minimum_step_pixels <= config.risk_target_step_pixels
            <= config.normal_target_step_pixels <= config.maximum_step_pixels
            <= config.emergency_step_pixels
        ):
            raise ValueError(
                "motion_resampling requires minimum <= risk <= normal <= maximum <= emergency"
            )
        return config


@dataclass(frozen=True)
class VideoRenderPlan:
    """A deterministic, provenance-preserving render-source selection."""

    frames: tuple[RGBDFrame, ...]
    source_indices: tuple[int, ...]
    scan_direction: int
    high_risk_edge_count: int
    normal_target_step_pixels: float
    risk_target_step_pixels: float

    def as_dict(self) -> dict[str, object]:
        return {
            "source_frame_ids": [frame.frame_id for frame in self.frames],
            "source_indices": list(self.source_indices),
            "source_count": len(self.frames),
            "scan_direction": self.scan_direction,
            "high_risk_edge_count": self.high_risk_edge_count,
            "normal_target_step_pixels": self.normal_target_step_pixels,
            "risk_target_step_pixels": self.risk_target_step_pixels,
            "real_source_frames_only": True,
            "interpolated_poses": False,
        }


def _scan_direction(motions: Sequence[MotionEstimate]) -> int:
    values = np.asarray(
        [motion.dx for motion in motions if motion.reliable and abs(motion.dx) > 1e-6],
        dtype=np.float64,
    )
    if not values.size:
        values = np.asarray([motion.dx for motion in motions], dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Video motion must contain finite horizontal displacement")
    return 1 if float(np.median(values)) >= 0.0 else -1


def _edge_is_risky(
    motion: MotionEstimate,
    *,
    full_resolution_scale: float,
    frame_width: int,
    left_quality: FrameQuality | None,
    right_quality: FrameQuality | None,
) -> bool:
    horizontal = abs(float(motion.dx)) * full_resolution_scale
    vertical = abs(float(motion.dy)) * full_resolution_scale
    if not motion.reliable or vertical > max(3.0, 0.18 * horizontal + 2.0):
        return True
    if horizontal > 0.10 * frame_width:
        return True
    if left_quality is None or right_quality is None:
        return False
    # Strong exposure/texture changes are kept denser so a later audited seam
    # has a real neighbouring source instead of a synthesized intermediate.
    return (
        abs(left_quality.dark_ratio - right_quality.dark_ratio) > 0.08
        or abs(left_quality.saturated_ratio - right_quality.saturated_ratio) > 0.06
        or min(left_quality.texture_coverage, right_quality.texture_coverage) < 0.15
    )


def select_render_keyframes(
    frames: Sequence[RGBDFrame],
    motions: Sequence[MotionEstimate],
    *,
    full_resolution_scale: float,
    frame_width: int,
    qualities: Sequence[FrameQuality] | None = None,
    config: MotionResamplingConfig | None = None,
) -> VideoRenderPlan:
    """Choose a real-source keyframe set using cumulative image progress.

    A source is emitted whenever accumulated reliable progress reaches the
    normal target, or the denser risk target for an unreliable/high-parallax
    edge.  The first and last source are always retained; this avoids holes at
    scan endpoints while preserving chronological provenance.
    """

    sources = tuple(frames)
    if len(sources) < 2 or len(motions) != len(sources) - 1:
        raise ValueError("Render keyframe selection requires N frames and N-1 motions")
    ids = [frame.frame_id for frame in sources]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Video render frames must have unique chronological ids")
    if not np.isfinite(full_resolution_scale) or full_resolution_scale <= 0.0:
        raise ValueError("full_resolution_scale must be finite and positive")
    if frame_width <= 0:
        raise ValueError("frame_width must be positive")
    if qualities is not None and len(qualities) != len(sources):
        raise ValueError("Render qualities must cover every source frame")
    settings = config or MotionResamplingConfig()
    settings = MotionResamplingConfig.from_mapping(settings.__dict__)
    direction = _scan_direction(motions)
    selected = [0]
    accumulated = 0.0
    risk_edges = 0
    for edge_index, motion in enumerate(motions):
        left_quality = qualities[edge_index] if qualities is not None else None
        right_quality = qualities[edge_index + 1] if qualities is not None else None
        risky = _edge_is_risky(
            motion,
            full_resolution_scale=full_resolution_scale,
            frame_width=frame_width,
            left_quality=left_quality,
            right_quality=right_quality,
        )
        if risky:
            risk_edges += 1
        progress = max(0.0, direction * float(motion.dx)) * full_resolution_scale
        accumulated += progress
        target = settings.risk_target_step_pixels if risky else settings.normal_target_step_pixels
        force = progress >= settings.maximum_step_pixels or accumulated >= settings.emergency_step_pixels
        if edge_index + 1 > selected[-1] and (accumulated >= target or force):
            selected.append(edge_index + 1)
            accumulated = 0.0
    if selected[-1] != len(sources) - 1:
        selected.append(len(sources) - 1)
    indices = tuple(selected)
    return VideoRenderPlan(
        frames=tuple(sources[index] for index in indices),
        source_indices=indices,
        scan_direction=direction,
        high_risk_edge_count=risk_edges,
        normal_target_step_pixels=settings.normal_target_step_pixels,
        risk_target_step_pixels=settings.risk_target_step_pixels,
    )


def compose_selected_motions(
    motions: Sequence[MotionEstimate], source_indices: Sequence[int], *, require_scan_endpoints: bool = True
) -> list[MotionEstimate]:
    """Compose measured intermediate motion for selected real-frame pairs.

    This supplies layout evidence only.  It does not create a pose, a frame,
    or a two-dimensional trajectory.
    """

    indices = tuple(int(index) for index in source_indices)
    if len(indices) < 2 or indices != tuple(sorted(set(indices))):
        raise ValueError("Selected source indices must be unique and increasing")
    if require_scan_endpoints and (indices[0] != 0 or indices[-1] != len(motions)):
        raise ValueError("Selected source indices must retain both scan endpoints")
    combined: list[MotionEstimate] = []
    for first, second in zip(indices, indices[1:]):
        span = motions[first:second]
        if not span:
            raise ValueError("Selected source pair has no measured motion")
        combined.append(
            MotionEstimate(
                dx=float(sum(item.dx for item in span)),
                dy=float(sum(item.dy for item in span)),
                matches=int(sum(item.matches for item in span)),
                inlier_ratio=float(np.mean([item.inlier_ratio for item in span])),
                grid_coverage=float(np.mean([item.grid_coverage for item in span])),
                method="composed_video_motion",
            )
        )
    return combined
