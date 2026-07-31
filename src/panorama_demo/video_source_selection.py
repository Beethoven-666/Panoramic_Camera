"""Real-source selection for video rendering; no poses are synthesized."""

from __future__ import annotations

from .session import RGBDFrame


def select_video_render_sources(frames: tuple[RGBDFrame, ...]) -> tuple[RGBDFrame, ...]:
    """Keep every real ORB-tracked node in chronological scan order.

    The tiled renderer is responsible for bounded memory.  Dropping a node
    here would silently turn a video scan into a sparse photo sequence.
    """
    if len(frames) < 2:
        raise ValueError("Video render requires at least two real source frames")
    ids = [frame.frame_id for frame in frames]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ValueError("Video source frame ids must be unique and chronological")
    return frames
