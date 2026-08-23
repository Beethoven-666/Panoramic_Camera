"""Output-first S0/S1 renderer isolated from every production video renderer."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Mapping, Sequence

import cv2
import numpy as np

from .quality import FrameQuality, MotionEstimate
from .session import CameraIntrinsics, RGBDFrame, read_aligned_depth_mm
from .video_s1_config import VideoS11ExperimentConfig, VideoS1ExperimentConfig
from .video_s1_handoff import (
    BoundaryCandidate,
    ForbiddenInterval,
    HandoffCoordinateTransform,
    HandoffEvidenceSplit,
    HandoffObservation,
    attach_aligned_depth_risk,
    attach_background_residual,
    build_forbidden_intervals,
    estimate_background_canvas_dx,
    extract_calibrated_handoff_observations,
    generate_boundary_candidates,
    solve_global_straight_boundaries,
    split_solver_heldout,
)
from .video_s1_layout import (
    S1MotionEdge,
    S1PairOverlap,
    S1SourceRescue,
    S1SourceLayout,
    build_monotonic_progress,
    build_source_layouts,
    insert_local_real_source_rescue,
    S1PairLineage,
    select_local_real_source_rescue,
    select_dynamic_sources,
    validate_refined_layout_invariants,
    validate_owner_partition,
)
from .video_s1_metrics import (
    PairStageVerticalMetrics,
    compare_stage_b_to_stage_c_per_pair,
    measure_boundary_edge_residuals,
)
from .video_s1_vertical_alignment import (
    SourceFeatures,
    VerticalAlignmentResult,
    align_vertical_pair,
    precompute_feature_cache,
)
from .video_s1_warp import (
    allocate_non_overlapping_warp_supports,
    audit_vertical_offsets,
    compose_pair_warp_transaction,
)


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


@dataclass(frozen=True)
class S11StageProvenance:
    geometric_owner_frame_id: np.ndarray
    declared_owner_sample_valid: np.ndarray
    final_owner_frame_id: np.ndarray
    final_valid: np.ndarray


@dataclass(frozen=True)
class S11PairResult:
    pair_index: int
    origin_pair_lineage: tuple[int, int]
    left_frame_id: int
    right_frame_id: int
    midpoint_x: int
    boundary_x: int
    pair_status: str
    solver_classification: str
    reference_source: str
    needs_s2: bool
    manual_review_required: bool
    capture_limited: bool
    observations: tuple[object, ...]
    forbidden_intervals: tuple[object, ...]
    heldout_metrics: Mapping[str, object]
    alignment: VerticalAlignmentResult
    pair_commit_status: str
    elapsed_seconds: float


@dataclass(frozen=True)
class VideoS11RenderResult:
    stage_a_nominal_midpoint_owner_only: np.ndarray
    stage_b_handoff_only: np.ndarray
    stage_c_final: np.ndarray
    stage_a_provenance: S11StageProvenance
    stage_b_provenance: S11StageProvenance
    stage_c_provenance: S11StageProvenance
    initial_layouts: tuple[S1SourceLayout, ...]
    final_layouts: tuple[S1SourceLayout, ...]
    midpoint_layouts: tuple[S1SourceLayout, ...]
    selected_frame_indices: tuple[int, ...]
    pair_results: tuple[S11PairResult, ...]
    remap_invocations: int
    stage_seconds: Mapping[str, float]
    rescue_events: tuple[Mapping[str, object], ...]
    vertical_metrics: tuple[PairStageVerticalMetrics, ...] = ()


@dataclass(frozen=True)
class _S11AnalysisSource:
    image: np.ndarray
    valid: np.ndarray
    gray: np.ndarray
    rgb_edge_columns: np.ndarray
    depth_mm: np.ndarray | None
    depth_edge_columns: np.ndarray | None


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


def build_calibrated_analysis_source(
    image: np.ndarray,
    calibration: CameraIntrinsics,
    *,
    analysis_width: int,
    undistortion_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample raw R directly into horizontally reduced calibrated A coordinates."""

    if image.shape[:2] != (calibration.height, calibration.width):
        raise ValueError("S1.1 analysis source does not match calibration")
    if analysis_width < 32 or analysis_width > calibration.width:
        raise ValueError("S1.1 handoff analysis width is invalid")
    undistort_x, undistort_y = undistortion_maps or _undistortion_maps(calibration)
    x_c = (
        (np.arange(analysis_width, dtype=np.float32) + 0.5)
        * (float(calibration.width) / float(analysis_width))
        - 0.5
    )
    grid_x = np.broadcast_to(x_c[None, :], (calibration.height, analysis_width))
    grid_y = np.broadcast_to(
        np.arange(calibration.height, dtype=np.float32)[:, None],
        (calibration.height, analysis_width),
    )
    raw_x = cv2.remap(
        undistort_x, grid_x, grid_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1,
    )
    raw_y = cv2.remap(
        undistort_y, grid_x, grid_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1,
    )
    valid = (
        (raw_x >= 0.0)
        & (raw_x <= calibration.width - 1)
        & (raw_y >= 0.0)
        & (raw_y <= calibration.height - 1)
    )
    calibrated = cv2.remap(
        image, raw_x, raw_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return np.ascontiguousarray(calibrated), np.ascontiguousarray(valid)


def build_calibrated_analysis_depth(
    depth_mm: np.ndarray,
    calibration: CameraIntrinsics,
    *,
    analysis_width: int,
    undistortion_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Sample aligned millimetre depth into A without inventing depth values."""

    if depth_mm.shape != (calibration.height, calibration.width):
        raise ValueError("S1.1 aligned depth does not match calibration")
    if analysis_width < 32 or analysis_width > calibration.width:
        raise ValueError("S1.1 handoff analysis width is invalid")
    undistort_x, undistort_y = undistortion_maps or _undistortion_maps(calibration)
    x_c = (
        (np.arange(analysis_width, dtype=np.float32) + 0.5)
        * (float(calibration.width) / float(analysis_width))
        - 0.5
    )
    grid_x = np.broadcast_to(x_c[None, :], (calibration.height, analysis_width))
    grid_y = np.broadcast_to(
        np.arange(calibration.height, dtype=np.float32)[:, None],
        (calibration.height, analysis_width),
    )
    raw_x = cv2.remap(
        undistort_x, grid_x, grid_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1,
    )
    raw_y = cv2.remap(
        undistort_y, grid_x, grid_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1,
    )
    valid = (
        (raw_x >= 0.0)
        & (raw_x <= calibration.width - 1)
        & (raw_y >= 0.0)
        & (raw_y <= calibration.height - 1)
    )
    calibrated = cv2.remap(
        np.asarray(depth_mm, dtype=np.float32), raw_x, raw_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    calibrated[~valid | ~np.isfinite(calibrated) | (calibrated <= 0.0)] = 0.0
    return np.ascontiguousarray(calibrated)


def _normalised_edge_columns(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a robust per-column vertical-edge crossing cost in [0, 1]."""

    values = np.asarray(image, dtype=np.float32)
    gradient = np.abs(cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3))
    if gradient.ndim == 3:
        gradient = np.max(gradient, axis=2)
    masked = np.where(valid, gradient, 0.0)
    columns = np.percentile(masked, 90.0, axis=0)
    columns[~np.any(valid, axis=0)] = 0.0
    scale = float(np.percentile(columns[columns > 0.0], 95.0)) if np.any(columns > 0.0) else 0.0
    if scale <= 0.0:
        return np.zeros(columns.shape, dtype=np.float64)
    return np.ascontiguousarray(np.clip(columns / scale, 0.0, 1.0), dtype=np.float64)


def _normalised_depth_edge_columns(depth_mm: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_mm) & (depth_mm > 0.0)
    left = depth_mm[:, :-1]
    right = depth_mm[:, 1:]
    both = valid[:, :-1] & valid[:, 1:]
    gate = np.maximum(20.0, 0.02 * np.maximum(left, right))
    edge = both & (np.abs(right - left) > gate)
    columns = np.zeros(depth_mm.shape[1], dtype=np.float64)
    if edge.size:
        columns[1:] = np.mean(edge, axis=0, dtype=np.float64)
    scale = float(np.percentile(columns[columns > 0.0], 95.0)) if np.any(columns > 0.0) else 0.0
    if scale > 0.0:
        columns = np.clip(columns / scale, 0.0, 1.0)
    return np.ascontiguousarray(columns)


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


def compose_strict_owner_panorama(
    contributions: Sequence[tuple[int, np.ndarray, np.ndarray]],
    layouts: Sequence[S1SourceLayout],
    *,
    canvas_width: int,
) -> tuple[np.ndarray, S11StageProvenance]:
    """Compose one hard owner with no foreign fill and explicit provenance."""

    if not contributions or len(contributions) != len(layouts):
        raise ValueError("Strict owner composition requires one contribution per layout")
    height = contributions[0][1].shape[0]
    panorama = np.zeros((height, canvas_width, 3), dtype=np.uint8)
    geometric = np.full((height, canvas_width), -1, dtype=np.int64)
    declared_valid = np.zeros((height, canvas_width), dtype=bool)
    final_owner = np.full((height, canvas_width), -1, dtype=np.int64)
    for (x0, image, source_valid), layout in zip(contributions, layouts):
        if image.shape[:2] != source_valid.shape or image.shape[0] != height:
            raise ValueError("Strict owner contribution shapes do not match")
        left = max(0, layout.owner_left_x)
        right = min(canvas_width, layout.owner_right_x)
        if right <= left:
            raise ValueError("Strict owner interval is empty")
        geometric[:, left:right] = int(layout.frame_id)
        local_left, local_right = left - x0, right - x0
        inside = (
            np.arange(local_left, local_right, dtype=np.int64) >= 0
        ) & (
            np.arange(local_left, local_right, dtype=np.int64) < image.shape[1]
        )
        if not np.all(inside):
            continue
        valid = source_valid[:, local_left:local_right]
        declared_valid[:, left:right] = valid
        target = panorama[:, left:right]
        target[valid] = image[:, local_left:local_right][valid]
        target_owner = final_owner[:, left:right]
        target_owner[valid] = int(layout.frame_id)
    if np.any(geometric < 0):
        raise ValueError("Geometric owner contains a gap")
    final_valid = final_owner >= 0
    if not np.array_equal(final_valid, declared_valid):
        raise ValueError("Strict owner validity and provenance disagree")
    provenance = S11StageProvenance(
        geometric_owner_frame_id=np.ascontiguousarray(geometric),
        declared_owner_sample_valid=np.ascontiguousarray(declared_valid),
        final_owner_frame_id=np.ascontiguousarray(final_owner),
        final_valid=np.ascontiguousarray(final_valid),
    )
    return panorama, provenance


def evaluate_rendered_heldout_single_copy(
    observations: Sequence[HandoffObservation],
    *,
    left_frame_id: int,
    right_frame_id: int,
    boundary_x: int,
    provenance: S11StageProvenance,
) -> dict[str, object]:
    """Audit held-out copies against the pixels actually present in one stage."""

    owner = np.asarray(provenance.final_owner_frame_id)
    valid = np.asarray(provenance.final_valid, dtype=bool)
    if owner.shape != valid.shape or owner.ndim != 2:
        raise ValueError("Rendered heldout provenance shape is invalid")
    samples: list[dict[str, object]] = []
    for item in observations:
        if item.role != "heldout":
            raise ValueError("Rendered single-copy audit accepts heldout observations only")
        left_y, left_x = int(np.rint(item.left_y)), int(np.rint(item.left_canvas_x))
        right_y, right_x = int(np.rint(item.right_y)), int(np.rint(item.right_canvas_x))
        left_inside = 0 <= left_y < owner.shape[0] and 0 <= left_x < owner.shape[1]
        right_inside = 0 <= right_y < owner.shape[0] and 0 <= right_x < owner.shape[1]
        evaluable = left_inside and right_inside
        # Copy semantics follow the pair's half-open straight cut. Provenance
        # supplies the actual rendered validity at each projection; the final
        # owner may legitimately be a later rescue/adjacent real source when a
        # narrow cell is refined, so requiring the original endpoint frame ID
        # would misclassify a present projection as missing.
        left_present = bool(
            evaluable
            and valid[left_y, left_x]
            and item.left_canvas_x < boundary_x
        )
        right_present = bool(
            evaluable
            and valid[right_y, right_x]
            and item.right_canvas_x >= boundary_x
        )
        copy_count = int(left_present) + int(right_present) if evaluable else 0
        samples.append({
            "match_id": item.match_id,
            "copy_count": copy_count,
            "evaluable": evaluable,
            "left_copy_present": left_present,
            "right_copy_present": right_present,
            "left_endpoint_frame_id": int(left_frame_id),
            "right_endpoint_frame_id": int(right_frame_id),
        })
    evaluable_samples = [item for item in samples if item["evaluable"]]
    duplicate = sum(int(item["copy_count"]) >= 2 for item in evaluable_samples)
    missing = sum(int(item["copy_count"]) == 0 for item in evaluable_samples)
    single = sum(int(item["copy_count"]) == 1 for item in evaluable_samples)
    total = len(samples)
    evaluable_count = len(evaluable_samples)
    denominator = max(1, evaluable_count)
    return {
        "total_heldout_count": total,
        "evaluable_count": evaluable_count,
        "evaluable_fraction": float(evaluable_count / total) if total else 0.0,
        "duplicate_count": duplicate,
        "missing_count": missing,
        "single_copy_count": single,
        "duplicate_fraction": float(duplicate / denominator) if evaluable_count else 0.0,
        "missing_fraction": float(missing / denominator) if evaluable_count else 0.0,
        "samples": samples,
    }


def layouts_with_boundaries(
    layouts: Sequence[S1SourceLayout],
    boundaries: Sequence[int],
    *,
    corridor_half_width_px: int,
) -> tuple[S1SourceLayout, ...]:
    """Return the same sources/supports with a new monotonic hard-owner partition."""

    if len(boundaries) != len(layouts) - 1 or corridor_half_width_px < 1:
        raise ValueError("Boundary count or corridor width is invalid")
    all_boundaries = [layouts[0].support_left_x, *(int(value) for value in boundaries), layouts[-1].support_right_x]
    if any(right <= left for left, right in zip(all_boundaries, all_boundaries[1:])):
        raise ValueError("Straight boundaries must be strictly increasing")
    result: list[S1SourceLayout] = []
    for index, layout in enumerate(layouts):
        owner_left, owner_right = all_boundaries[index], all_boundaries[index + 1]
        if owner_left < layout.support_left_x or owner_right > layout.support_right_x:
            raise ValueError("Straight owner interval leaves its source support")
        result.append(replace(
            layout,
            owner_left_x=owner_left,
            owner_right_x=owner_right,
            match_left_x=max(layout.support_left_x, owner_left - corridor_half_width_px),
            match_right_x=min(layout.support_right_x, owner_right + corridor_half_width_px),
        ))
    validate_owner_partition(result)
    return tuple(result)


def rebuild_refined_midpoint_layouts(
    initial_layouts: Sequence[S1SourceLayout],
    selected_frame_indices: Sequence[int],
    frame_ids: Sequence[int],
    progress_fullres_px: Sequence[float],
    qualities: Sequence[FrameQuality],
    *,
    image_width: int,
    shoulder_px: int = 64,
) -> tuple[S1SourceLayout, ...]:
    """Insert local real sources while preserving every initial S01 anchor exactly."""

    selected = tuple(int(value) for value in selected_frame_indices)
    if tuple(sorted(set(selected))) != selected or not initial_layouts:
        raise ValueError("Refined real-source indices must be strictly increasing")
    if selected[0] != initial_layouts[0].source_index or selected[-1] != initial_layouts[-1].source_index:
        raise ValueError("Refined layout cannot change the S01 endpoint sources")
    progress = np.asarray(progress_fullres_px, dtype=np.float64)
    if len(frame_ids) != len(progress) or len(qualities) != len(frame_ids):
        raise ValueError("Refined layout evidence must align with all real frames")
    anchors = {item.source_index: item for item in initial_layouts}
    anchor_indices = np.asarray(sorted(anchors), dtype=np.int64)
    centers: list[float] = []
    for index in selected:
        if index in anchors:
            centers.append(float(anchors[index].canvas_center_x))
            continue
        right_position = int(np.searchsorted(anchor_indices, index, side="right"))
        if right_position == 0 or right_position >= len(anchor_indices):
            raise ValueError("A rescued source is not bracketed by S01 anchors")
        left_index = int(anchor_indices[right_position - 1])
        right_index = int(anchor_indices[right_position])
        denominator = progress[right_index] - progress[left_index]
        fraction = 0.5 if denominator <= 1e-9 else (progress[index] - progress[left_index]) / denominator
        centers.append(float(
            anchors[left_index].canvas_center_x
            + np.clip(fraction, 0.0, 1.0)
            * (anchors[right_index].canvas_center_x - anchors[left_index].canvas_center_x)
        ))
    boundaries = [initial_layouts[0].support_left_x]
    boundaries.extend(int(np.floor((left + right) * 0.5 + 0.5)) for left, right in zip(centers, centers[1:]))
    boundaries.append(initial_layouts[-1].support_right_x)
    result: list[S1SourceLayout] = []
    for position, (index, center) in enumerate(zip(selected, centers)):
        if index in anchors:
            support_left, support_right = anchors[index].support_left_x, anchors[index].support_right_x
        else:
            support_left = int(np.floor(center - 0.5 * image_width + 0.5))
            support_right = support_left + image_width
        owner_left, owner_right = boundaries[position], boundaries[position + 1]
        if owner_left < support_left or owner_right > support_right:
            raise ValueError("Refined source support does not contain its midpoint owner")
        result.append(S1SourceLayout(
            frame_id=int(frame_ids[index]),
            source_index=index,
            canvas_center_x=center,
            owner_left_x=owner_left,
            owner_right_x=owner_right,
            support_left_x=support_left,
            support_right_x=support_right,
            match_left_x=max(support_left, owner_left - shoulder_px),
            match_right_x=min(support_right, owner_right + shoulder_px),
            measured_progress_from_previous_px=(0.0 if position == 0 else center - centers[position - 1]),
            source_quality=_quality_dict(qualities[index]),
        ))
    validate_owner_partition(result)
    if result[0].support_left_x != initial_layouts[0].support_left_x or result[-1].support_right_x != initial_layouts[-1].support_right_x:
        raise ValueError("Refined layout changed the S01 canvas extent")
    return tuple(result)


@dataclass(frozen=True)
class _S11PairEvidence:
    pair_index: int
    lineage: S1PairLineage
    split: HandoffEvidenceSplit
    forbidden_intervals: tuple[ForbiddenInterval, ...]
    candidates: tuple[BoundaryCandidate, ...]
    observable: bool


def _s11_baseline_layouts(
    baseline: Mapping[str, object], frames: Sequence[RGBDFrame]
) -> tuple[S1SourceLayout, ...]:
    layout = baseline.get("layout")
    expected = baseline.get("expected")
    if not isinstance(layout, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("Verified S01 baseline is missing layout or expected metadata")
    sources = layout.get("sources")
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("Verified S01 baseline has no source layout")
    canvas_width = expected.get("canvas_width")
    if isinstance(canvas_width, bool) or not isinstance(canvas_width, int):
        raise ValueError("Verified S01 baseline has no integer canvas width")
    from .video_s1_layout import inherit_s01_v1_exact

    return inherit_s01_v1_exact(
        sources,
        [frame.frame_id for frame in frames],
        expected_canvas_width=canvas_width,
    )


def _s11_lineage_for_pair(
    left: S1SourceLayout,
    right: S1SourceLayout,
    initial_layouts: Sequence[S1SourceLayout],
) -> S1PairLineage:
    for origin_left, origin_right in zip(initial_layouts, initial_layouts[1:]):
        if (
            origin_left.source_index <= left.source_index
            and right.source_index <= origin_right.source_index
        ):
            return S1PairLineage(origin_left.frame_id, origin_right.frame_id)
    raise ValueError("Refined pair is not contained by one original S01 lineage")


def _s11_pair_evidence(
    layouts: Sequence[S1SourceLayout],
    initial_layouts: Sequence[S1SourceLayout],
    analysis_cache: Mapping[int, _S11AnalysisSource],
    calibration: CameraIntrinsics,
    config: VideoS11ExperimentConfig,
    *,
    calibration_id: str,
) -> tuple[_S11PairEvidence, ...]:
    result: list[_S11PairEvidence] = []
    half_corridor = config.handoff.match_corridor_width_px * 0.5
    for pair_index, (left, right) in enumerate(zip(layouts, layouts[1:])):
        lineage = _s11_lineage_for_pair(left, right, initial_layouts)
        left_source = analysis_cache[left.frame_id]
        right_source = analysis_cache[right.frame_id]
        left_gray, left_valid = left_source.gray, left_source.valid
        right_gray, right_valid = right_source.gray, right_source.valid
        midpoint = left.owner_right_x
        common_left = max(left.support_left_x, right.support_left_x)
        common_right = min(left.support_right_x, right.support_right_x)
        corridor = (
            max(
                float(common_left),
                float(left.owner_left_x),
                midpoint - half_corridor,
            ),
            min(
                float(common_right),
                float(right.owner_right_x),
                midpoint + half_corridor,
            ),
        )
        if corridor[1] <= corridor[0]:
            observations: tuple[HandoffObservation, ...] = ()
        else:
            observations = extract_calibrated_handoff_observations(
                left_gray,
                right_gray,
                left_valid,
                right_valid,
                HandoffCoordinateTransform(
                    config.handoff.analysis_width_px,
                    calibration.width,
                    left.support_left_x,
                ),
                HandoffCoordinateTransform(
                    config.handoff.analysis_width_px,
                    calibration.width,
                    right.support_left_x,
                ),
                calibration_id=calibration_id,
                origin_pair_lineage=f"{lineage.origin_left_frame_id}->{lineage.origin_right_frame_id}",
                left_frame_id=left.frame_id,
                right_frame_id=right.frame_id,
                corridor_canvas=corridor,
                lk_forward_backward_max_px=config.handoff.lk_forward_backward_max_px,
                maximum_vertical_match_error_px=config.handoff.maximum_vertical_match_error_px,
            )
            # Pair evidence is local to the two adjacent owner cells.  A match
            # whose projection lands in a third source's owner is not evidence
            # about this handoff and must not enter either solver or heldout.
            pair_left = float(left.owner_left_x)
            pair_right = float(right.owner_right_x)
            observations = tuple(
                item for item in observations
                if pair_left <= item.left_canvas_x < pair_right
                and pair_left <= item.right_canvas_x < pair_right
            )
        split = split_solver_heldout(observations) if observations else HandoffEvidenceSplit((), ())
        if (
            config.handoff.use_aligned_depth_for_risk
            and left_source.depth_mm is not None
            and right_source.depth_mm is not None
        ):
            split = HandoffEvidenceSplit(
                attach_aligned_depth_risk(
                    split.solver, left_source.depth_mm, right_source.depth_mm
                ),
                attach_aligned_depth_risk(
                    split.heldout, left_source.depth_mm, right_source.depth_mm
                ),
            )
        if split.solver:
            background_dx = estimate_background_canvas_dx(split.solver)
            split = HandoffEvidenceSplit(
                attach_background_residual(split.solver, background_dx),
                attach_background_residual(split.heldout, background_dx),
            )
        intervals = build_forbidden_intervals(
            split.solver,
            minimum_match_count=config.handoff.minimum_protected_match_count,
            minimum_vertical_span_px=config.handoff.minimum_protected_vertical_span_px,
            minimum_interval_width_px=config.handoff.protected_interval_min_width_px,
            minimum_relative_dx_px=config.handoff.minimum_background_relative_dx_px,
            cluster_neighbor_x_px=config.handoff.cluster_neighbor_x_px,
            cluster_neighbor_y_px=config.handoff.cluster_neighbor_y_px,
            cluster_relative_dx_tolerance_px=(
                config.handoff.cluster_relative_dx_tolerance_px
            ),
            merge_gap_px=config.handoff.cluster_merge_gap_px,
            guard_px=config.handoff.conflict_guard_fullres_px,
        )
        allowed_left = max(common_left, left.owner_left_x + 1)
        allowed_right = min(common_right, right.owner_right_x - 1)
        candidate_x = np.arange(allowed_left, allowed_right + 1, dtype=np.float64)

        def sample_columns(layout: S1SourceLayout,
                           columns: np.ndarray | None) -> np.ndarray:
            if columns is None or candidate_x.size == 0:
                return np.zeros(candidate_x.size, dtype=np.float64)
            calibrated_x = candidate_x - float(layout.support_left_x)
            analysis_x = (
                (calibrated_x + 0.5)
                * (float(config.handoff.analysis_width_px) / float(calibration.width))
                - 0.5
            )
            return np.interp(
                analysis_x,
                np.arange(columns.size, dtype=np.float64),
                columns,
                left=0.0,
                right=0.0,
            )

        rgb_edge_cost = np.maximum(
            sample_columns(left, left_source.rgb_edge_columns),
            sample_columns(right, right_source.rgb_edge_columns),
        )
        depth_edge_cost = None
        if config.handoff.use_aligned_depth_for_risk:
            depth_edge_cost = np.maximum(
                sample_columns(left, left_source.depth_edge_columns),
                sample_columns(right, right_source.depth_edge_columns),
            )
        candidates = generate_boundary_candidates(
            midpoint_x=midpoint,
            allowed_left_x=allowed_left,
            allowed_right_x=allowed_right,
            forbidden_intervals=intervals,
            soft_observations=split.solver,
            maximum_shift_px=config.handoff.maximum_boundary_shift_px,
            rgb_edge_cost=rgb_edge_cost,
            depth_edge_cost=depth_edge_cost,
        )
        result.append(_S11PairEvidence(
            pair_index=pair_index,
            lineage=lineage,
            split=split,
            forbidden_intervals=intervals,
            candidates=candidates,
            observable=len(split.solver) >= config.handoff.minimum_match_count,
        ))
    return tuple(result)


def _s11_reference_source(
    boundary_x: int, intervals: Sequence[ForbiddenInterval]
) -> str:
    if not intervals:
        return "none"
    reference: set[str] = set()
    for interval in intervals:
        if boundary_x >= interval.right_x:
            reference.add("left")
        elif boundary_x <= interval.left_x:
            reference.add("right")
        else:
            reference.update(("left", "right"))
    if len(reference) > 1:
        return "both"
    return next(iter(reference))


def _s11_zero_alignment(
    height: int, config: VideoS11ExperimentConfig, reason: str
) -> VerticalAlignmentResult:
    blank = np.zeros((height, 64, 3), dtype=np.uint8)
    features = precompute_feature_cache({0: blank, 1: blank}, horizontal_scale=1.0)
    result = align_vertical_pair(features[0], features[1], config.s1, select_gains=False)
    return replace(
        result,
        pair_status="s11_zero_unobservable",
        diagnostics={"degraded_reason": reason},
    )


def _analysis_pair_edge_p95(
    left_source: _S11AnalysisSource,
    right_source: _S11AnalysisSource,
    left_layout: S1SourceLayout,
    right_layout: S1SourceLayout,
    boundary_x: int,
    *,
    calibrated_width: int,
    row_dy: np.ndarray | None = None,
    reference_source: str = "none",
) -> float:
    """Measure the Stage-B straight seam in cached calibrated A coordinates."""

    shoulder = 16
    height, analysis_width = left_source.gray.shape

    if reference_source == "left":
        left_scale, right_scale = 0.0, 1.0
    elif reference_source == "right":
        left_scale, right_scale = -1.0, 0.0
    else:
        left_scale, right_scale = -0.5, 0.5

    def sample(source: _S11AnalysisSource, layout: S1SourceLayout,
               canvas_x: np.ndarray, dy_scale: float) -> tuple[np.ndarray, np.ndarray]:
        calibrated_x = canvas_x - float(layout.support_left_x)
        analysis_x = (
            (calibrated_x + 0.5)
            * (float(analysis_width) / float(calibrated_width))
            - 0.5
        ).astype(np.float32)
        map_x = np.broadcast_to(analysis_x[None, :], (height, analysis_x.size))
        map_y = np.broadcast_to(
            np.arange(height, dtype=np.float32)[:, None], map_x.shape
        ).copy()
        if row_dy is not None:
            map_y += np.asarray(row_dy, dtype=np.float32)[:, None] * dy_scale
        image = cv2.remap(
            source.image, map_x, map_y, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        valid = cv2.remap(
            source.valid.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)
        return image, valid

    left_x = np.arange(boundary_x - shoulder, boundary_x, dtype=np.float64)
    right_x = np.arange(boundary_x, boundary_x + shoulder, dtype=np.float64)
    left_image, left_valid = sample(left_source, left_layout, left_x, left_scale)
    right_image, right_valid = sample(right_source, right_layout, right_x, right_scale)
    image = np.concatenate((left_image, right_image), axis=1)
    valid = np.concatenate((left_valid, right_valid), axis=1)
    metrics = measure_boundary_edge_residuals(
        image, (shoulder,), valid_mask=valid, shoulder_width_px=8
    )
    return float(metrics.subpixel_edge_residual_p95)


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


def render_video_s11(
    frames: Sequence[RGBDFrame],
    calibration: CameraIntrinsics,
    motions: Sequence[MotionEstimate],
    qualities: Sequence[FrameQuality],
    *,
    scan_direction: int,
    analysis_width: int,
    config: VideoS11ExperimentConfig,
    baseline: Mapping[str, object],
) -> VideoS11RenderResult:
    """Render isolated S1.1 Stage A/B/C from one verified, frozen S01 baseline."""

    if len(frames) < 2 or len(motions) != len(frames) - 1 or len(qualities) != len(frames):
        raise ValueError("S1.1 inputs are not aligned")
    if scan_direction not in {-1, 1}:
        raise ValueError("S1.1 scan direction must be -1 or 1")
    stage: dict[str, float] = {}

    def timed(name: str):
        class Timer:
            def __enter__(self):
                self.started = time.perf_counter()

            def __exit__(self, *_):
                stage[name] = time.perf_counter() - self.started

        return Timer()

    with timed("baseline_inheritance"):
        initial_layouts = _s11_baseline_layouts(baseline, frames)
        selected_indices = tuple(item.source_index for item in initial_layouts)
        edges = _motion_edges(motions, qualities)
        progress_analysis = build_monotonic_progress(edges, scan_direction)
        progress_fullres = (
            progress_analysis * (float(calibration.width) / float(analysis_width))
        )
    expected = baseline["expected"]
    assert isinstance(expected, Mapping)
    session_hashes = expected.get("session", {})
    calibration_id = (
        str(session_hashes.get("calibration_sha256", "unknown"))
        if isinstance(session_hashes, Mapping)
        else "unknown"
    )
    undistort_maps = _undistortion_maps(calibration)
    raw_images: dict[int, np.ndarray] = {}
    analysis_cache: dict[int, _S11AnalysisSource] = {}

    def cache_source(source_index: int) -> None:
        frame = frames[source_index]
        if frame.frame_id in analysis_cache:
            return
        image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read selected S1.1 source: {frame.color_path}")
        if image.shape[:2] != (calibration.height, calibration.width):
            raise ValueError("Selected S1.1 source does not match calibration dimensions")
        raw_images[frame.frame_id] = image
        calibrated_image, calibrated_valid = build_calibrated_analysis_source(
            image,
            calibration,
            analysis_width=config.handoff.analysis_width_px,
            undistortion_maps=undistort_maps,
        )
        gray = cv2.cvtColor(calibrated_image, cv2.COLOR_BGR2GRAY)
        depth_analysis = None
        depth_edge_columns = None
        if config.handoff.use_aligned_depth_for_risk:
            depth_analysis = build_calibrated_analysis_depth(
                read_aligned_depth_mm(frame),
                calibration,
                analysis_width=config.handoff.analysis_width_px,
                undistortion_maps=undistort_maps,
            )
            depth_edge_columns = _normalised_depth_edge_columns(depth_analysis)
        analysis_cache[frame.frame_id] = _S11AnalysisSource(
            image=calibrated_image,
            valid=calibrated_valid,
            gray=gray,
            rgb_edge_columns=_normalised_edge_columns(gray, calibrated_valid),
            depth_mm=depth_analysis,
            depth_edge_columns=depth_edge_columns,
        )

    rescue_events: list[Mapping[str, object]] = []
    origin_heldout_by_lineage: dict[
        tuple[int, int], tuple[HandoffObservation, ...]
    ] = {}
    inserted_by_lineage: dict[tuple[int, int], list[int]] = {}
    capture_limited_lineages: set[tuple[int, int]] = set()
    candidate_generation_seconds = 0.0

    def build_pair_evidence(
        layouts: Sequence[S1SourceLayout],
    ) -> tuple[_S11PairEvidence, ...]:
        nonlocal candidate_generation_seconds
        started = time.perf_counter()
        result = _s11_pair_evidence(
            layouts,
            initial_layouts,
            analysis_cache,
            calibration,
            config,
            calibration_id=calibration_id,
        )
        candidate_generation_seconds += time.perf_counter() - started
        return result

    with timed("handoff_and_rescue"):
        midpoint_layouts = rebuild_refined_midpoint_layouts(
            initial_layouts,
            selected_indices,
            [frame.frame_id for frame in frames],
            progress_fullres,
            qualities,
            image_width=calibration.width,
            shoulder_px=config.handoff.match_corridor_width_px // 2,
        )
        for index in selected_indices:
            cache_source(index)
        for refinement_round in range(1, config.source_selection.maximum_refinement_rounds + 1):
            evidence = build_pair_evidence(midpoint_layouts)
            round_rescues: list[
                tuple[S1SourceRescue, tuple[HandoffObservation, ...]]
            ] = []
            used_lineages: set[tuple[int, int]] = set()
            for pair in evidence:
                midpoint = midpoint_layouts[pair.pair_index].owner_right_x
                lineage_key = pair.lineage.frame_ids
                midpoint_hard = any(
                    interval.contains(midpoint) for interval in pair.forbidden_intervals
                )
                has_safe = any(
                    not candidate.hard_forbidden_crossing for candidate in pair.candidates
                )
                if (
                    not pair.observable
                    or not midpoint_hard
                    or has_safe
                    or lineage_key in used_lineages
                ):
                    continue
                left_layout = midpoint_layouts[pair.pair_index]
                right_layout = midpoint_layouts[pair.pair_index + 1]
                rescue = select_local_real_source_rescue(
                    [frame.frame_id for frame in frames],
                    progress_analysis,
                    left_layout.source_index,
                    right_layout.source_index,
                    config.source_selection,
                    calibrated_width=calibration.width,
                    analysis_width=analysis_width,
                    origin_pair_lineage=pair.lineage,
                    refinement_round=refinement_round,
                    prior_inserted_frame_indices=inserted_by_lineage.get(lineage_key, ()),
                    rgb_available=[frame.color_path.is_file() for frame in frames],
                    source_quality_penalties=[
                        max(0.0, 1.0 - float(item.texture_coverage)) for item in qualities
                    ],
                )
                if rescue is None:
                    capture_limited_lineages.add(lineage_key)
                    continue
                provisional_indices = selected_indices
                for accepted, _origin_heldout in round_rescues:
                    provisional_indices = insert_local_real_source_rescue(
                        provisional_indices, accepted
                    )
                provisional_indices = insert_local_real_source_rescue(
                    provisional_indices, rescue
                )
                provisional_layouts = rebuild_refined_midpoint_layouts(
                    initial_layouts,
                    provisional_indices,
                    [frame.frame_id for frame in frames],
                    progress_fullres,
                    qualities,
                    image_width=calibration.width,
                    shoulder_px=config.handoff.match_corridor_width_px // 2,
                )
                if any(
                    layout.owner_right_x - layout.owner_left_x
                    < config.handoff.minimum_internal_owner_width_px
                    for layout in provisional_layouts[1:-1]
                ):
                    # A rescue that makes the required straight-owner topology
                    # infeasible is not safe. Keep the frozen parent and expose
                    # the lineage as capture-limited instead.
                    capture_limited_lineages.add(lineage_key)
                    continue
                round_rescues.append((
                    rescue,
                    tuple(pair.split.heldout),
                ))
                used_lineages.add(lineage_key)
            if not round_rescues:
                break
            for rescue_value, origin_heldout in round_rescues:
                # The tuple is deliberately retained in the audit event even
                # though child-pair evidence is rebuilt from scratch.  Rescue
                # must never make the origin held-out denominator disappear.
                rescue = rescue_value
                selected_indices = insert_local_real_source_rescue(selected_indices, rescue)
                inserted_by_lineage.setdefault(rescue.origin_pair_lineage.frame_ids, []).append(
                    rescue.frame_index
                )
                origin_heldout_by_lineage.setdefault(
                    rescue.origin_pair_lineage.frame_ids, origin_heldout
                )
                rescue_events.append({
                    **asdict(rescue),
                    "origin_midpoint_x": next(
                        left.owner_right_x
                        for left, right in zip(initial_layouts, initial_layouts[1:])
                        if (
                            left.frame_id == rescue.origin_pair_lineage.origin_left_frame_id
                            and right.frame_id
                            == rescue.origin_pair_lineage.origin_right_frame_id
                        )
                    ),
                    "origin_heldout_denominator": len(origin_heldout),
                    "origin_heldout_match_ids": [
                        item.match_id for item in origin_heldout
                    ],
                })
                cache_source(rescue.frame_index)
            midpoint_layouts = rebuild_refined_midpoint_layouts(
                initial_layouts,
                selected_indices,
                [frame.frame_id for frame in frames],
                progress_fullres,
                qualities,
                image_width=calibration.width,
                shoulder_px=config.handoff.match_corridor_width_px // 2,
            )
            validate_refined_layout_invariants(initial_layouts, midpoint_layouts)
        evidence = build_pair_evidence(midpoint_layouts)
        final_unsafe_lineages = {
            item.lineage.frame_ids
            for item in evidence
            if item.observable
            and any(
                interval.contains(midpoint_layouts[item.pair_index].owner_right_x)
                for interval in item.forbidden_intervals
            )
            and not any(
                not candidate.hard_forbidden_crossing for candidate in item.candidates
            )
        }
        capture_limited_lineages.intersection_update(final_unsafe_lineages)
    stage["handoff_candidate_generation"] = candidate_generation_seconds
    with timed("straight_boundary_dp"):
        solution = solve_global_straight_boundaries(
            [item.candidates for item in evidence],
            midpoint_boundaries=[item.owner_right_x for item in midpoint_layouts[:-1]],
            first_owner_left_x=midpoint_layouts[0].owner_left_x,
            last_owner_right_x=midpoint_layouts[-1].owner_right_x,
            minimum_internal_owner_width_px=(
                config.handoff.minimum_internal_owner_width_px
            ),
            observable=[item.observable for item in evidence],
        )
        final_layouts = layouts_with_boundaries(
            midpoint_layouts,
            solution.boundaries,
            corridor_half_width_px=config.handoff.match_corridor_width_px // 2,
        )
    feature_cache: dict[int, SourceFeatures] = {}
    analysis_scale = config.handoff.analysis_width_px / calibration.width
    with timed("vertical_features"):
        for layout in final_layouts:
            image = analysis_cache[layout.frame_id].image
            features = precompute_feature_cache(
                {layout.frame_id: image}, horizontal_scale=1.0
            )[layout.frame_id]
            feature_cache[layout.frame_id] = replace(
                features, horizontal_scale=analysis_scale
            )
    height, width = calibration.height, calibration.width
    identity_y = np.asarray(undistort_maps[1], dtype=np.float64)
    calibration_valid = (
        np.isfinite(undistort_maps[0])
        & np.isfinite(undistort_maps[1])
        & (undistort_maps[0] >= 0.0)
        & (undistort_maps[0] <= width - 1)
        & (undistort_maps[1] >= 0.0)
        & (undistort_maps[1] <= height - 1)
    )
    source_maps = [np.ascontiguousarray(identity_y.copy()) for _ in final_layouts]
    # Construct the explicit identity/degraded result once. Re-running the
    # complete matcher on a blank pair for every skipped source pair dominated
    # real post-capture time while producing the same zero curve.
    zero_vertical_alignment = _s11_zero_alignment(
        height, config, "cached S1.1 identity vertical result"
    )

    def zero_vertical(reason: str) -> VerticalAlignmentResult:
        return replace(
            zero_vertical_alignment,
            diagnostics={"degraded_reason": reason},
        )

    support_widths = [
        allocate_non_overlapping_warp_supports(
            layout.owner_right_x - layout.owner_left_x,
            has_left_pair=index > 0,
            has_right_pair=index + 1 < len(final_layouts),
            owner_fraction=config.s1.warp_support_owner_fraction,
            minimum_px=config.s1.warp_support_minimum_px,
            maximum_px=config.s1.warp_support_maximum_px,
        )
        for index, layout in enumerate(final_layouts)
    ]
    pending_pairs: list[tuple[object, ...]] = []

    def prepare_pair_vertical(
        pair_index: int,
    ) -> tuple[S1PairOverlap, VerticalAlignmentResult, float]:
        started = time.perf_counter()
        left_layout, right_layout = final_layouts[pair_index : pair_index + 2]
        pair_evidence = evidence[pair_index]
        decision = solution.decisions[pair_index]
        reference = _s11_reference_source(
            decision.boundary_x, pair_evidence.forbidden_intervals
        )
        vertical_errors = np.asarray(
            [
                abs(item.vertical_error_px)
                for item in (
                    *pair_evidence.split.solver,
                    *pair_evidence.split.heldout,
                )
            ],
            dtype=np.float64,
        )
        vertical_error_p95 = (
            float(np.percentile(vertical_errors, 95))
            if vertical_errors.size
            else 0.0
        )
        stage_b_edge_p95 = _analysis_pair_edge_p95(
            analysis_cache[left_layout.frame_id],
            analysis_cache[right_layout.frame_id],
            left_layout,
            right_layout,
            decision.boundary_x,
            calibrated_width=calibration.width,
        )
        vertical_required = (
            (
                vertical_error_p95 >= config.s1.fine_vertical_trigger_p95_px
                or (
                    np.isfinite(stage_b_edge_p95)
                    and stage_b_edge_p95
                    > config.s1.fine_vertical_trigger_p95_px
                )
            )
            and reference != "both"
        )
        overlap, left_crop, right_crop = extract_pair_overlap(
            pair_index,
            left_layout,
            right_layout,
            feature_cache[left_layout.frame_id],
            feature_cache[right_layout.frame_id],
            minimum_overlap_px=16,
        )
        if not vertical_required:
            alignment = zero_vertical(
                "Stage B sparse vertical residual does not require fine vertical",
            )
        elif left_crop is None or right_crop is None:
            alignment = zero_vertical("vertical overlap is insufficient")
        else:
            # MotionEstimate.dy is the source-to-next image displacement.
            # The inverse sampling candidate asks where the right image is
            # sampled for a left row, hence the search prior has opposite sign.
            motion_dy = -float(
                sum(
                    item.dy
                    for item in motions[
                        left_layout.source_index : right_layout.source_index
                    ]
                )
            ) * (float(calibration.width) / float(analysis_width))
            alignment = align_vertical_pair(
                left_crop,
                right_crop,
                config.s1,
                select_gains=False,
                risk_pair=bool(pair_evidence.forbidden_intervals),
                motion_dy_px=motion_dy,
            )
            corrected_edge_p95 = _analysis_pair_edge_p95(
                analysis_cache[left_layout.frame_id],
                analysis_cache[right_layout.frame_id],
                left_layout,
                right_layout,
                decision.boundary_x,
                calibrated_width=calibration.width,
                row_dy=np.asarray(alignment.curve.dy_applied, dtype=np.float64),
                reference_source=reference,
            )
            # The 424 px analysis image is evidence for deciding whether fine
            # alignment is worth attempting, but it is not the acceptance
            # domain.  In real captures it can merge two full-resolution
            # ridges or suppress a narrow edge.  Keep its before/after values
            # in diagnostics and let the downstream full-resolution Stage B/C
            # transaction gate make the sole commit/rollback decision.
        corrected_edge_p95 = (
            _analysis_pair_edge_p95(
                analysis_cache[left_layout.frame_id],
                analysis_cache[right_layout.frame_id],
                left_layout,
                right_layout,
                decision.boundary_x,
                calibrated_width=calibration.width,
                row_dy=np.asarray(alignment.curve.dy_applied, dtype=np.float64),
                reference_source=reference,
            )
            if vertical_required
            else stage_b_edge_p95
        )
        alignment = replace(
            alignment,
            diagnostics={
                **dict(alignment.diagnostics),
                "stage_b_analysis_edge_residual_p95_px": stage_b_edge_p95,
                "corrected_analysis_edge_residual_p95_px": corrected_edge_p95,
                "sparse_vertical_error_p95_px": vertical_error_p95,
            },
        )
        return overlap, alignment, time.perf_counter() - started

    with timed("vertical_pair_transactions"):
        with ThreadPoolExecutor(
            # Batched cost volumes are memory-bandwidth bound on the target;
            # concurrent real pairs regress wall time substantially.
            max_workers=1
        ) as executor:
            vertical_futures = [
                executor.submit(prepare_pair_vertical, pair_index)
                for pair_index in range(len(evidence))
            ]
        for pair_index, (left_layout, right_layout) in enumerate(
            zip(final_layouts, final_layouts[1:])
        ):
            started = time.perf_counter()
            pair_evidence = evidence[pair_index]
            decision = solution.decisions[pair_index]
            overlap, alignment, alignment_elapsed = vertical_futures[pair_index].result()
            reference = _s11_reference_source(
                decision.boundary_x, pair_evidence.forbidden_intervals
            )
            transaction = compose_pair_warp_transaction(
                source_maps[pair_index],
                source_maps[pair_index + 1],
                boundary_x=decision.boundary_x,
                curve=alignment.curve,
                reference_source=reference,
                left_support_px=support_widths[pair_index][1],
                right_support_px=support_widths[pair_index + 1][0],
                left_canvas_x=(
                    np.arange(width, dtype=np.float64) + left_layout.support_left_x
                ),
                right_canvas_x=(
                    np.arange(width, dtype=np.float64) + right_layout.support_left_x
                ),
                left_valid_mask=calibration_valid,
                right_valid_mask=calibration_valid,
                left_source_height=height,
                right_source_height=height,
                maximum_local_slope=config.s1.maximum_local_slope,
            )
            source_maps[pair_index] = transaction.left_map_y
            source_maps[pair_index + 1] = transaction.right_map_y
            lineage_key = pair_evidence.lineage.frame_ids
            capture_limited = lineage_key in capture_limited_lineages
            pending_pairs.append((
                pair_index,
                pair_evidence,
                decision,
                overlap,
                alignment,
                reference,
                transaction.pair_commit_status,
                capture_limited,
                alignment_elapsed + time.perf_counter() - started,
            ))
    stage["candidate_generation"] = (
        candidate_generation_seconds
        + stage.get("vertical_pair_transactions", 0.0)
    )
    identity_contributions: list[tuple[int, np.ndarray, np.ndarray]] = []
    corrected_contributions: list[tuple[int, np.ndarray, np.ndarray]] = []
    remap_invocations = 0
    with timed("joint_full_resolution_remap"):
        for index, layout in enumerate(final_layouts):
            vertical_offset = source_maps[index] - identity_y
            identity_image, identity_valid, corrected_image, corrected_valid = _remap_source_once(
                raw_images[layout.frame_id],
                undistort_maps[0],
                undistort_maps[1],
                vertical_offset,
            )
            remap_invocations += 1
            identity_contributions.append(
                (layout.support_left_x, identity_image, identity_valid)
            )
            corrected_contributions.append(
                (layout.support_left_x, corrected_image, corrected_valid)
            )
    canvas_width = final_layouts[-1].support_right_x
    with timed("stage_composition"):
        stage_a, provenance_a = compose_strict_owner_panorama(
            identity_contributions, midpoint_layouts, canvas_width=canvas_width
        )
        stage_b, provenance_b = compose_strict_owner_panorama(
            identity_contributions, final_layouts, canvas_width=canvas_width
        )
        stage_c, provenance_c = compose_strict_owner_panorama(
            corrected_contributions, final_layouts, canvas_width=canvas_width
        )
        final_vertical_metrics = compare_stage_b_to_stage_c_per_pair(
            stage_b,
            stage_c,
            [item.owner_right_x for item in final_layouts[:-1]],
            stage_b_valid=provenance_b.final_valid,
            stage_c_valid=provenance_c.final_valid,
        )
    if rescue_events:
        audited_events: list[Mapping[str, object]] = []
        for event in rescue_events:
            lineage_value = event["origin_pair_lineage"]
            assert isinstance(lineage_value, Mapping)
            lineage = (
                int(lineage_value["origin_left_frame_id"]),
                int(lineage_value["origin_right_frame_id"]),
            )
            origin_heldout = origin_heldout_by_lineage.get(lineage, ())
            origin_boundary = int(event["origin_midpoint_x"])
            audited_events.append({
                **event,
                "origin_stage_b_heldout_metrics": (
                    evaluate_rendered_heldout_single_copy(
                        origin_heldout,
                        left_frame_id=lineage[0],
                        right_frame_id=lineage[1],
                        boundary_x=origin_boundary,
                        provenance=provenance_b,
                    )
                ),
                "origin_stage_c_heldout_metrics": (
                    evaluate_rendered_heldout_single_copy(
                        origin_heldout,
                        left_frame_id=lineage[0],
                        right_frame_id=lineage[1],
                        boundary_x=origin_boundary,
                        provenance=provenance_c,
                    )
                ),
            })
        rescue_events = audited_events
    pair_results: list[S11PairResult] = []
    with timed("heldout_metrics"):
        for pending in pending_pairs:
            (
                pair_index,
                pair_evidence,
                decision,
                _overlap,
                alignment,
                reference,
                pair_commit_status,
                capture_limited,
                elapsed,
            ) = pending
            assert isinstance(pair_evidence, _S11PairEvidence)
            left_frame_id = final_layouts[pair_index].frame_id
            right_frame_id = final_layouts[pair_index + 1].frame_id
            stage_b_metrics = evaluate_rendered_heldout_single_copy(
                pair_evidence.split.heldout,
                left_frame_id=left_frame_id,
                right_frame_id=right_frame_id,
                boundary_x=decision.boundary_x,
                provenance=provenance_b,
            )
            stage_c_metrics = evaluate_rendered_heldout_single_copy(
                pair_evidence.split.heldout,
                left_frame_id=left_frame_id,
                right_frame_id=right_frame_id,
                boundary_x=decision.boundary_x,
                provenance=provenance_c,
            )
            metrics = {
                **stage_c_metrics,
                "stage_b": stage_b_metrics,
                "stage_c": stage_c_metrics,
            }
            lineage = pair_evidence.lineage.frame_ids
            heldout_rejected = bool(
                int(stage_c_metrics["evaluable_count"]) > 0
                and (
                    int(stage_c_metrics["duplicate_count"]) > 0
                    or int(stage_c_metrics["missing_count"]) > 0
                )
            )
            base_needs_s2 = bool(decision.needs_s2 or capture_limited)
            if heldout_rejected:
                pair_status = "heldout_rejected"
            elif lineage in inserted_by_lineage and not base_needs_s2:
                pair_status = "source_refined_then_safe"
            elif decision.needs_s2 and decision.boundary_x == decision.midpoint_x:
                pair_status = "unsafe_fallback_midpoint"
            else:
                pair_status = (
                    "capture_limited" if capture_limited else decision.classification
                )
            pair_results.append(S11PairResult(
                pair_index=pair_index,
                origin_pair_lineage=lineage,
                left_frame_id=left_frame_id,
                right_frame_id=right_frame_id,
                midpoint_x=midpoint_layouts[pair_index].owner_right_x,
                boundary_x=decision.boundary_x,
                pair_status=pair_status,
                solver_classification=decision.classification,
                reference_source=str(reference),
                needs_s2=bool(base_needs_s2 or heldout_rejected),
                manual_review_required=bool(
                    decision.manual_review_required
                    or capture_limited
                    or heldout_rejected
                ),
                capture_limited=bool(capture_limited),
                observations=tuple(
                    (*pair_evidence.split.solver, *pair_evidence.split.heldout)
                ),
                forbidden_intervals=pair_evidence.forbidden_intervals,
                heldout_metrics=metrics,
                alignment=alignment,
                pair_commit_status=(
                    str(pair_commit_status)
                ),
                elapsed_seconds=float(elapsed),
            ))
    return VideoS11RenderResult(
        stage_a_nominal_midpoint_owner_only=stage_a,
        stage_b_handoff_only=stage_b,
        stage_c_final=stage_c,
        stage_a_provenance=provenance_a,
        stage_b_provenance=provenance_b,
        stage_c_provenance=provenance_c,
        initial_layouts=initial_layouts,
        final_layouts=final_layouts,
        midpoint_layouts=midpoint_layouts,
        selected_frame_indices=selected_indices,
        pair_results=tuple(pair_results),
        remap_invocations=remap_invocations,
        stage_seconds=stage,
        rescue_events=tuple(rescue_events),
        vertical_metrics=final_vertical_metrics,
    )


def serialise_layout(layout: S1SourceLayout) -> dict[str, object]:
    payload = asdict(layout)
    payload["source_quality"] = dict(layout.source_quality)
    return payload
