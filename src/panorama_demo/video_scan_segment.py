"""Selection of one contiguous, direction-consistent video scan segment."""

from __future__ import annotations

import cv2

from .quality import FrameQuality, MotionEstimate, analyze_frame_quality, estimate_translation, resize_for_analysis, select_primary_scan_segment
from .session import RGBDFrame


def analyse_video_scan(frames: tuple[RGBDFrame, ...], *, analysis_width: int = 320) -> tuple[list[FrameQuality], list[MotionEstimate], dict[str, object]]:
    if len(frames) < 2:
        raise ValueError("Video panorama requires at least two RGB-D frames")
    previews = []
    qualities = []
    for frame in frames:
        image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read video colour frame: {frame.color_path}")
        preview = resize_for_analysis(image, analysis_width)
        previews.append(preview)
        qualities.append(analyze_frame_quality(preview))
    motions = [estimate_translation(left, right) for left, right in zip(previews, previews[1:])]
    segment = select_primary_scan_segment(motions, image_width=previews[0].shape[1])
    if segment.end_index - segment.start_index + 1 < 2:
        raise ValueError("Video has no usable contiguous directional scan segment")
    return qualities, motions, segment.as_dict()
