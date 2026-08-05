"""Selection of one contiguous, direction-consistent video scan segment."""

from __future__ import annotations

import cv2
import numpy as np

from .quality import FrameQuality, MotionEstimate, analyze_frame_quality, estimate_translation, resize_for_analysis, select_primary_scan_segment
from .session import RGBDFrame


def _estimate_dis_translation(reference, source) -> MotionEstimate:
    """Cheap dense source-to-reference motion for the video fast path."""

    if reference.shape != source.shape:
        raise ValueError("Video motion images must have identical shape")
    first = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    flow = dis.calc(first, second, None)
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    finite = bool(np.isfinite(flow).all()) and (
        float(np.max(magnitude)) <= float(reference.shape[1]) * 0.30
    )
    valid = np.isfinite(flow).all(axis=2) & (magnitude <= float(reference.shape[1]) * 0.15)
    if not finite or int(valid.sum()) < max(32, valid.size // 10):
        return MotionEstimate(0.0, 0.0, 0, 0.0, 0.0, "dis_unreliable")
    values = flow[valid]
    # DIS estimates reference -> source.  The scan code expects source ->
    # reference, matching the existing feature/phase estimator convention.
    dx, dy = -np.median(values, axis=0)
    coverage = float(valid.mean())
    return MotionEstimate(
        dx=float(dx),
        dy=float(dy),
        matches=int(valid.sum()),
        inlier_ratio=coverage,
        grid_coverage=coverage,
        method="dis_ultrafast",
    )


def estimate_video_motion(
    reference: np.ndarray,
    source: np.ndarray,
    *,
    motion_backend: str = "dis",
) -> MotionEstimate:
    """Estimate one capture-order motion edge for online or offline scan analysis."""

    if motion_backend == "dis":
        return _estimate_dis_translation(reference, source)
    if motion_backend == "feature":
        return estimate_translation(reference, source)
    raise ValueError("video motion_backend must be 'dis' or 'feature'")


def analyse_video_scan(
    frames: tuple[RGBDFrame, ...],
    *,
    analysis_width: int = 320,
    motion_backend: str = "dis",
) -> tuple[list[FrameQuality], list[MotionEstimate], dict[str, object]]:
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
    motions = [
        estimate_video_motion(left, right, motion_backend=motion_backend)
        for left, right in zip(previews, previews[1:])
    ]
    try:
        segment = select_primary_scan_segment(motions, image_width=previews[0].shape[1])
    except RuntimeError:
        if motion_backend != "dis":
            raise
        # DIS is the fast default, but a large startup jump or a textureless
        # diagnostic clip can make dense flow unobservable.  Preserve the
        # historical robust estimator only for this bounded fallback.
        motions = [
            estimate_translation(left, right)
            for left, right in zip(previews, previews[1:])
        ]
        segment = select_primary_scan_segment(motions, image_width=previews[0].shape[1])
    if segment.end_index - segment.start_index + 1 < 2:
        raise ValueError("Video has no usable contiguous directional scan segment")
    return qualities, motions, segment.as_dict()
