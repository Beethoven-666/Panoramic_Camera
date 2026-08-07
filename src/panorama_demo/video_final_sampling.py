"""One-pass raw-RGB inverse-grid sampling for the v6 video candidate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoSamplingSource:
    """One true raw RGB source and its precomputed output-to-source inverse grid."""
    frame_id: int
    raw_bgr: np.ndarray
    inverse_x: np.ndarray
    inverse_y: np.ndarray
    valid_mask: np.ndarray

    def __post_init__(self) -> None:
        image = np.asarray(self.raw_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("final sampling source must be raw uint8 BGR")
        shape = np.asarray(self.valid_mask).shape
        if len(shape) != 2 or np.asarray(self.inverse_x).shape != shape or np.asarray(self.inverse_y).shape != shape:
            raise ValueError("inverse grids and source valid mask must share an output canvas")
        if not np.isfinite(np.asarray(self.inverse_x)[np.asarray(self.valid_mask, bool)]).all() or not np.isfinite(np.asarray(self.inverse_y)[np.asarray(self.valid_mask, bool)]).all():
            raise ValueError("valid inverse-grid cells must be finite")


@dataclass(frozen=True)
class VideoFinalSamplingAudit:
    source_frame_ids: tuple[int, ...]
    source_inverse_remap_call_count: tuple[int, ...]
    valid_pixel_count: int
    strict_owner_partition: bool
    exactly_one_raw_rgb_sampling_per_source: bool


@dataclass(frozen=True)
class VideoFinalSamplingResult:
    bgr: np.ndarray
    valid_mask: np.ndarray
    owner_frame_id: np.ndarray
    audit: VideoFinalSamplingAudit


def render_video_final_once(sources: Iterable[VideoSamplingSource], owner_frame_id: np.ndarray) -> VideoFinalSamplingResult:
    """Sample every true source once, then copy only its owned output pixels."""
    ordered = tuple(sources)
    if not ordered:
        raise ValueError("final sampling requires at least one real source")
    frame_ids = tuple(int(source.frame_id) for source in ordered)
    if frame_ids != tuple(sorted(set(frame_ids))):
        raise ValueError("final sampling source ids must be unique and chronological")
    owner = np.asarray(owner_frame_id, dtype=np.int32)
    shape = ordered[0].valid_mask.shape
    if owner.shape != shape or any(source.valid_mask.shape != shape for source in ordered):
        raise ValueError("owner map and every source grid must share one output canvas")
    valid = owner >= 0
    known = np.isin(owner[valid], frame_ids)
    if not np.all(known):
        raise ValueError("final owner map contains an unknown source id")
    sampled: list[np.ndarray] = []
    calls: list[int] = []
    for source in ordered:
        # The only raw-RGB resampling operation in this function.  Its result
        # is thereafter copied by owner id, never remapped again.
        sampled.append(cv2.remap(source.raw_bgr, source.inverse_x.astype(np.float32), source.inverse_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)))
        calls.append(1)
    output = np.zeros((*shape, 3), dtype=np.uint8)
    topology = True
    for source, remapped in zip(ordered, sampled, strict=True):
        selected = owner == source.frame_id
        if np.any(selected & ~source.valid_mask):
            topology = False
        output[selected] = remapped[selected]
    if np.any(~valid & (owner >= 0)):
        topology = False
    return VideoFinalSamplingResult(output, valid, owner.copy(), VideoFinalSamplingAudit(frame_ids, tuple(calls), int(valid.sum()), topology, all(value == 1 for value in calls)))


__all__ = ["VideoFinalSamplingAudit", "VideoFinalSamplingResult", "VideoSamplingSource", "render_video_final_once"]
