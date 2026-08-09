"""Output-first S0/S1 renderer isolated from every production video renderer."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import cv2
import numpy as np

from .quality import FrameQuality, MotionEstimate
from .session import CameraIntrinsics, RGBDFrame
from .video_s1_config import VideoS1ExperimentConfig
from .video_s1_layout import (
    S1MotionEdge,
    S1PairOverlap,
    S1SourceLayout,
    build_monotonic_progress,
    build_source_layouts,
    select_dynamic_sources,
    validate_owner_partition,
)
from .video_s1_vertical_alignment import (
    SourceFeatures,
    VerticalAlignmentResult,
    align_vertical_pair,
    precompute_feature_cache,
)
from .video_s1_warp import audit_vertical_offsets


@dataclass(frozen=True)
class S1PairResult:
    overlap: S1PairOverlap
    alignment: VerticalAlignmentResult
    elapsed_seconds: float


@dataclass(frozen=True)
class VideoS1RenderResult:
    s0_owner_only: np.ndarray
    s0_panorama: np.ndarray
    s1_owner_only: np.ndarray
    s1_panorama: np.ndarray
    owner_map: np.ndarray
    valid_mask: np.ndarray
    layouts: tuple[S1SourceLayout, ...]
    selected_frame_indices: tuple[int, ...]
    pair_results: tuple[S1PairResult, ...]
    remap_invocations: int
    stage_seconds: Mapping[str, float]


def _motion_edges(motions: Sequence[MotionEstimate], qualities: Sequence[FrameQuality]) -> tuple[S1MotionEdge, ...]:
    return tuple(
        S1MotionEdge(
            dx=float(item.dx),
            dy=float(item.dy),
            inlier_ratio=float(item.inlier_ratio),
            grid_coverage=float(item.grid_coverage),
            texture_coverage=float(min(qualities[index].texture_coverage, qualities[index + 1].texture_coverage)),
            reliable=bool(item.reliable),
        )
        for index, item in enumerate(motions)
    )


def _quality_dict(quality: FrameQuality) -> dict[str, float]:
    return {key: float(value) for key, value in quality.as_dict().items()}


def build_experiment_layout(
    frames: Sequence[RGBDFrame],
    motions: Sequence[MotionEstimate],
    qualities: Sequence[FrameQuality],
    *,
    scan_direction: int,
    analysis_width: int,
    image_width: int,
    config: VideoS1ExperimentConfig,
) -> tuple[tuple[int, ...], tuple[S1SourceLayout, ...]]:
    if len(frames) < 2 or len(motions) != len(frames) - 1 or len(qualities) != len(frames):
        raise ValueError("S01 layout inputs are not aligned")
    edges = _motion_edges(motions, qualities)
    indices = select_dynamic_sources(
        [frame.frame_id for frame in frames],
        edges,
        config.source_selection,
        scan_direction=scan_direction,
        source_qualities=[_quality_dict(item) for item in qualities],
    )
    progress = build_monotonic_progress(edges, scan_direction)
    scale = float(image_width) / float(analysis_width)
    selected_progress = progress[np.asarray(indices)] * scale
    centres = selected_progress + 0.5 * image_width
    supports = [(int(round(value - image_width * 0.5)), int(round(value + image_width * 0.5))) for value in centres]
    canvas_origin = min(left for left, _ in supports)
    centres = centres - canvas_origin
    supports = [(left - canvas_origin, right - canvas_origin) for left, right in supports]
    source_qualities = [_quality_dict(qualities[index]) for index in indices]
    # Shoulder sizes are full-resolution values; source steps were measured at
    # analysis scale and explicitly lifted above.
    layouts = build_source_layouts(
        [frame.frame_id for frame in frames], indices, centres, supports,
        config.source_selection, source_qualities=source_qualities,
    )
    validate_owner_partition(layouts)
    return tuple(indices), layouts


def _slice_features(features: SourceFeatures, left: int, right: int) -> SourceFeatures:
    if right <= left:
        raise ValueError("Feature overlap is empty")
    fields = {
        name: np.ascontiguousarray(getattr(features, name)[:, left:right])
        for name in (
            "gray", "lab", "gradient_x", "gradient_y", "canny",
            "structure_tensor_score", "sharpness", "valid_mask",
        )
    }
    return SourceFeatures(**fields, horizontal_scale=features.horizontal_scale)


def extract_pair_overlap(
    pair_index: int,
    left_layout: S1SourceLayout,
    right_layout: S1SourceLayout,
    left_features: SourceFeatures,
    right_features: SourceFeatures,
    *,
    minimum_overlap_px: int,
) -> tuple[S1PairOverlap, SourceFeatures | None, SourceFeatures | None]:
    roi_left = max(left_layout.match_left_x, right_layout.match_left_x)
    roi_right = min(left_layout.match_right_x, right_layout.match_right_x)
    width_full = max(0, roi_right - roi_left)
    scale = left_features.horizontal_scale
    left_x0 = int(round((roi_left - left_layout.support_left_x) * scale))
    left_x1 = int(round((roi_right - left_layout.support_left_x) * scale))
    right_x0 = int(round((roi_left - right_layout.support_left_x) * scale))
    right_x1 = int(round((roi_right - right_layout.support_left_x) * scale))
    common_width = min(left_x1 - left_x0, right_x1 - right_x0)
    height = left_features.gray.shape[0]
    if common_width > 0:
        left_x0 = int(np.clip(left_x0, 0, left_features.gray.shape[1] - common_width))
        right_x0 = int(np.clip(right_x0, 0, right_features.gray.shape[1] - common_width))
        left_crop = _slice_features(left_features, left_x0, left_x0 + common_width)
        right_crop = _slice_features(right_features, right_x0, right_x0 + common_width)
        common = left_crop.valid_mask & right_crop.valid_mask
        row_widths = np.sum(common, axis=1).astype(np.float64) / max(scale, 1e-9)
    else:
        left_crop = right_crop = None
        common = np.zeros((height, 0), dtype=bool)
        row_widths = np.zeros(height, dtype=np.float64)
    positive = row_widths[row_widths > 0]
    percentiles = np.percentile(positive, [5, 50, 95]) if positive.size else np.zeros(3)
    evaluable = float(np.mean(row_widths >= minimum_overlap_px)) if row_widths.size else 0.0
    overlap = S1PairOverlap(
        pair_index=pair_index,
        left_frame_id=left_layout.frame_id,
        right_frame_id=right_layout.frame_id,
        owner_boundary_x=left_layout.owner_right_x,
        roi_left_x=roi_left,
        roi_right_x=roi_right,
        per_row_valid_left_x=np.full(height, roi_left, dtype=np.int32),
        per_row_valid_right_x=np.rint(roi_left + row_widths).astype(np.int32),
        common_valid_mask=common,
        overlap_width_p5=float(percentiles[0]),
        overlap_width_p50=float(percentiles[1]),
        overlap_width_p95=float(percentiles[2]),
        s1_evaluable_fraction=evaluable,
    )
    if width_full < minimum_overlap_px or evaluable < 0.25:
        return overlap, None, None
    return overlap, left_crop, right_crop


def _zero_alignment(height: int, reason: str) -> VerticalAlignmentResult:
    # align_vertical_pair owns the exact zero-curve representation and its
    # fail-local status, so use an intentionally unobservable feature pair.
    blank = np.zeros((height, 64, 3), dtype=np.uint8)
    features = precompute_feature_cache({0: blank, 1: blank}, horizontal_scale=0.5)
    result = align_vertical_pair(features[0], features[1], select_gains=False)
    return replace(result, pair_status="s1_insufficient_overlap", diagnostics={"degraded_reason": reason})


def _undistortion_maps(calibration: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray]:
    distortion = np.asarray(calibration.distortion, dtype=np.float64)
    return cv2.initUndistortRectifyMap(
        calibration.matrix, distortion, None, calibration.matrix,
        (calibration.width, calibration.height), cv2.CV_32FC1,
    )


def _pair_offsets_for_source(
    layout_index: int,
    layouts: Sequence[S1SourceLayout],
    pairs: Sequence[S1PairResult],
    *,
    height: int,
    width: int,
    maximum_local_slope: float,
    warp_support_left_px: int,
    warp_support_right_px: int,
) -> np.ndarray:
    x = np.arange(width, dtype=np.float64) + layouts[layout_index].support_left_x
    offsets = np.zeros((height, width), dtype=np.float64)
    for pair_index in (layout_index - 1, layout_index):
        if pair_index < 0 or pair_index >= len(pairs):
            continue
        pair = pairs[pair_index]
        curve = pair.alignment.curve
        dy = np.interp(np.arange(height), curve.y, curve.dy_applied, left=0.0, right=0.0)
        boundary = pair.overlap.owner_boundary_x
        if layout_index == pair_index:
            distance = boundary - x
            shoulder = max(1, min(warp_support_left_px, boundary - layouts[layout_index].support_left_x))
            weight = np.where(
                (distance >= 0) & (distance <= shoulder),
                0.5 * (1.0 + np.cos(np.pi * np.clip(distance / shoulder, 0, 1))), 0.0,
            )
            offsets -= 0.5 * dy[:, None] * weight[None, :]
        else:
            distance = x - boundary
            shoulder = max(1, min(warp_support_right_px, layouts[layout_index].support_right_x - boundary))
            weight = np.where(
                (distance >= 0) & (distance <= shoulder),
                0.5 * (1.0 + np.cos(np.pi * np.clip(distance / shoulder, 0, 1))), 0.0,
            )
            offsets += 0.5 * dy[:, None] * weight[None, :]
    audit = audit_vertical_offsets(offsets, maximum_local_slope=maximum_local_slope)
    return offsets if audit.safe else np.zeros_like(offsets)


def _remap_source_once(
    image: np.ndarray,
    undistort_x: np.ndarray,
    undistort_y: np.ndarray,
    vertical_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    grid_x = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], (height, width))
    grid_y = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width))
    corrected_y = (grid_y + vertical_offset).astype(np.float32)
    s1_x = cv2.remap(undistort_x, grid_x, corrected_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    s1_y = cv2.remap(undistort_y, grid_x, corrected_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    joint_x = np.concatenate((undistort_x, s1_x), axis=1)
    joint_y = np.concatenate((undistort_y, s1_y), axis=1)
    joint = cv2.remap(image, joint_x, joint_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = (joint_x >= 0) & (joint_x <= width - 1) & (joint_y >= 0) & (joint_y <= height - 1)
    return joint[:, :width], valid[:, :width], joint[:, width:], valid[:, width:]


def compose_owner_panorama(
    contributions: Sequence[tuple[int, np.ndarray, np.ndarray]],
    layouts: Sequence[S1SourceLayout],
    *,
    canvas_width: int,
    fill_from_adjacent: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not contributions or len(contributions) != len(layouts):
        raise ValueError("Owner composition requires one contribution per source")
    height = contributions[0][1].shape[0]
    panorama = np.zeros((height, canvas_width, 3), dtype=np.uint8)
    valid = np.zeros((height, canvas_width), dtype=bool)
    owner = np.full((height, canvas_width), -1, dtype=np.int32)
    for source_index, ((x0, image, mask), layout) in enumerate(zip(contributions, layouts)):
        left = max(layout.owner_left_x, x0)
        right = min(layout.owner_right_x, x0 + image.shape[1], canvas_width)
        if right <= left:
            continue
        local = slice(left - x0, right - x0)
        target = slice(left, right)
        accepted = mask[:, local]
        panorama[:, target][accepted] = image[:, local][accepted]
        valid[:, target][accepted] = True
        owner[:, target][accepted] = source_index
    if fill_from_adjacent and not np.all(valid):
        for source_index, (x0, image, mask) in enumerate(contributions):
            left = max(0, x0)
            right = min(canvas_width, x0 + image.shape[1])
            if right <= left:
                continue
            local = slice(left - x0, right - x0)
            target = slice(left, right)
            accepted = (~valid[:, target]) & mask[:, local]
            panorama[:, target][accepted] = image[:, local][accepted]
            valid[:, target][accepted] = True
            owner[:, target][accepted] = source_index
    return panorama, valid, owner


def _narrow_transition(
    owner_only: np.ndarray,
    contributions: Sequence[tuple[int, np.ndarray, np.ndarray]],
    layouts: Sequence[S1SourceLayout],
    width: int,
) -> np.ndarray:
    result = owner_only.copy()
    if width <= 0:
        return result
    for pair_index, (left_layout, right_layout) in enumerate(zip(layouts, layouts[1:])):
        boundary = left_layout.owner_right_x
        left_x0, left_image, left_valid = contributions[pair_index]
        right_x0, right_image, right_valid = contributions[pair_index + 1]
        for x in range(max(0, boundary - width + 1), min(result.shape[1], boundary + width)):
            lx, rx = x - left_x0, x - right_x0
            if not (0 <= lx < left_image.shape[1] and 0 <= rx < right_image.shape[1]):
                continue
            common = left_valid[:, lx] & right_valid[:, rx]
            alpha = float(np.clip((x - (boundary - width)) / max(1, 2 * width), 0.0, 1.0))
            blended = cv2.addWeighted(left_image[:, lx], 1.0 - alpha, right_image[:, rx], alpha, 0.0)
            result[:, x][common] = blended[common]
    return result


def render_video_s1(
    frames: Sequence[RGBDFrame],
    calibration: CameraIntrinsics,
    motions: Sequence[MotionEstimate],
    qualities: Sequence[FrameQuality],
    *,
    scan_direction: int,
    analysis_width: int,
    config: VideoS1ExperimentConfig,
) -> VideoS1RenderResult:
    stage: dict[str, float] = {}

    def timed(name: str):
        class Timer:
            def __enter__(self): self.started = time.perf_counter()
            def __exit__(self, *_): stage[name] = time.perf_counter() - self.started
        return Timer()

    with timed("source_selection"):
        indices, layouts = build_experiment_layout(
            frames, motions, qualities, scan_direction=scan_direction,
            analysis_width=analysis_width, image_width=calibration.width, config=config,
        )
    with timed("feature_precompute"):
        images: dict[int, np.ndarray] = {}
        for index in indices:
            image = cv2.imread(str(frames[index].color_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read selected S01 source: {frames[index].color_path}")
            if image.shape[:2] != (calibration.height, calibration.width):
                raise ValueError("Selected S01 source does not match calibration dimensions")
            images[frames[index].frame_id] = image
        feature_cache = precompute_feature_cache(
            images, horizontal_scale=config.s1.horizontal_scale
        )
    pair_results: list[S1PairResult] = []
    with timed("candidate_generation"):
        def align_pair(pair_index: int) -> S1PairResult:
            left_layout, right_layout = layouts[pair_index], layouts[pair_index + 1]
            started = time.perf_counter()
            overlap, left_crop, right_crop = extract_pair_overlap(
                pair_index, left_layout, right_layout,
                feature_cache[left_layout.frame_id], feature_cache[right_layout.frame_id],
                minimum_overlap_px=config.source_selection.minimum_s1_overlap_px,
            )
            if left_crop is None or right_crop is None:
                alignment = _zero_alignment(calibration.height, "common overlap is below the S1 minimum")
            else:
                alignment = align_vertical_pair(left_crop, right_crop, config.s1)
            return S1PairResult(overlap, alignment, time.perf_counter() - started)

        # OpenCV kernels already use their own bounded native workers.  Running
        # several pair jobs concurrently oversubscribes those kernels on the
        # target Windows host, so keep pair ordering deterministic here.
        pair_results = [align_pair(index) for index in range(max(0, len(layouts) - 1))]
    with timed("inverse_map_build"):
        undistort_x, undistort_y = _undistortion_maps(calibration)
    s0_contributions: list[tuple[int, np.ndarray, np.ndarray]] = []
    s1_contributions: list[tuple[int, np.ndarray, np.ndarray]] = []
    remap_invocations = 0
    with timed("final_full_resolution_remap"):
        for layout_index, layout in enumerate(layouts):
            offset = _pair_offsets_for_source(
                layout_index, layouts, pair_results, height=calibration.height,
                width=calibration.width, maximum_local_slope=config.s1.maximum_local_slope,
                warp_support_left_px=config.s1.warp_support_left_px,
                warp_support_right_px=config.s1.warp_support_right_px,
            )
            s0_image, s0_valid, s1_image, s1_valid = _remap_source_once(
                images[layout.frame_id], undistort_x, undistort_y, offset
            )
            remap_invocations += 1
            s0_contributions.append((layout.support_left_x, s0_image, s0_valid))
            s1_contributions.append((layout.support_left_x, s1_image, s1_valid))
    canvas_width = layouts[-1].support_right_x
    with timed("s0_render"):
        s0_owner, _, _ = compose_owner_panorama(
            s0_contributions, layouts, canvas_width=canvas_width,
            fill_from_adjacent=config.s0.fill_invalid_owner_from_adjacent_real_source,
        )
        s0 = _narrow_transition(s0_owner, s0_contributions, layouts, config.s0.transition_width_px)
    with timed("s1_compose"):
        s1_owner, valid, owner = compose_owner_panorama(
            s1_contributions, layouts, canvas_width=canvas_width,
            fill_from_adjacent=config.s0.fill_invalid_owner_from_adjacent_real_source,
        )
        s1 = _narrow_transition(s1_owner, s1_contributions, layouts, config.s0.transition_width_px)
    return VideoS1RenderResult(
        s0_owner, s0, s1_owner, s1, owner, valid, layouts, indices,
        tuple(pair_results), remap_invocations, stage,
    )


def serialise_layout(layout: S1SourceLayout) -> dict[str, object]:
    payload = asdict(layout)
    payload["source_quality"] = dict(layout.source_quality)
    return payload
