"""End-to-end v6 candidate renderer for one adjacent real-source pair."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomConfig,
    CalibratedRGBPushbroomRenderer,
    CalibratedRGBPushbroomResult,
    build_calibrated_rgb_pushbroom_layout,
    estimate_rgb_motion_pixels_per_mm,
)
from .video_final_sampling import VideoSamplingSource, sample_video_sources_once
from .video_graphcut_seam import VideoGraphCutAudit, solve_video_graphcut_seam
from .video_hard_guards import audit_guard_owner_intersection, build_video_hard_guards
from .video_local_alignment import VideoLocalAlignmentConfig, fit_background_alignment
from .video_near_blend import apply_near_multiband, build_near_blend_eligible_mask
from .video_rgb_quality import VideoRGBQualityAudit, assess_video_rgb_quality
from .video_visual_renderer import VideoDISPairEvidence, video_dis_pair_evidence


@dataclass(frozen=True)
class VideoV6PairRenderResult:
    bgr: np.ndarray
    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    dis_evidence: VideoDISPairEvidence
    graphcut_audit: VideoGraphCutAudit
    quality: VideoRGBQualityAudit
    near_blend_pixel_count: int
    source_sampling_call_count: int


@dataclass(frozen=True)
class VideoV6RenderResult:
    bgr: np.ndarray
    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    graphcut_audits: tuple[VideoGraphCutAudit, ...]
    quality: VideoRGBQualityAudit
    source_sampling_call_count: int
    expanded_real_owner_pair_frame_ids: tuple[tuple[int, int], ...]


def build_v6_sampling_sources(
    frames: tuple[object, ...], poses: tuple[np.ndarray, ...], calibration: object, *,
    pushbroom_config: dict[str, object], rgb_motions: list[object], motion_pixels_to_full_resolution: float,
) -> tuple[VideoSamplingSource, ...]:
    """Build full-canvas inverse grids without invoking the legacy compositor."""
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    scale = estimate_rgb_motion_pixels_per_mm(
        frames, poses, calibration, settings, rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in frames], poses, calibration, scale, settings
    )
    renderer = CalibratedRGBPushbroomRenderer(layout, calibration, poses)
    height, width = layout.canvas_height, layout.canvas_width
    grid_x = np.arange(width, dtype=np.float64)
    grid_y = np.arange(height, dtype=np.float64)
    results: list[VideoSamplingSource] = []
    for index, frame in enumerate(frames):
        map_x, map_y, valid = renderer._inverse_map(index, grid_x, grid_y)
        support = np.zeros((height, width), dtype=bool)
        support[:, layout.support_left_x[index] : layout.support_right_x[index]] = True
        image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read v6 raw RGB source {frame.color_path}")
        results.append(VideoSamplingSource(int(frame.frame_id), image, map_x, map_y, valid & support))
    return tuple(results)


def _preview_bgr(source: VideoSamplingSource, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Analysis-only low-resolution calibrated sampling; never output RGB."""
    map_x = source.inverse_x[::factor, ::factor].astype(np.float32)
    map_y = source.inverse_y[::factor, ::factor].astype(np.float32)
    image = cv2.remap(source.raw_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return image, np.asarray(source.valid_mask, bool)[::factor, ::factor]


def _apply_output_matrix_to_grid(
    source: VideoSamplingSource, matrix: np.ndarray, support: np.ndarray,
) -> VideoSamplingSource:
    """Compose an audited output transform into an inverse grid before raw sampling."""
    height, width = source.valid_mask.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(height * width, np.float32)))
    projected = points @ np.asarray(matrix, np.float64).T
    target_x = (projected[:, 0] / projected[:, 2]).reshape((height, width)).astype(np.float32)
    target_y = (projected[:, 1] / projected[:, 2]).reshape((height, width)).astype(np.float32)
    adjusted_x = cv2.remap(source.inverse_x.astype(np.float32), target_x, target_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
    adjusted_y = cv2.remap(source.inverse_y.astype(np.float32), target_x, target_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
    adjusted_valid = cv2.remap(source.valid_mask.astype(np.uint8), target_x, target_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    mask = np.asarray(support, bool)
    return VideoSamplingSource(
        source.frame_id, source.raw_bgr,
        np.where(mask, adjusted_x, source.inverse_x), np.where(mask, adjusted_y, source.inverse_y),
        np.where(mask, adjusted_valid, source.valid_mask),
    )


def apply_v6_background_alignment_to_grids(
    sources: tuple[VideoSamplingSource, ...], *, return_audits: bool = False,
) -> tuple[VideoSamplingSource, ...] | tuple[tuple[VideoSamplingSource, ...], tuple[dict[str, object], ...]]:
    """Apply only accepted RGB/DIS background models before the final sample."""
    adjusted = list(sources)
    audits: list[dict[str, object]] = []
    scale = 4
    for index in range(1, len(adjusted)):
        old_preview, old_valid = _preview_bgr(adjusted[index - 1], scale)
        new_preview, new_valid = _preview_bgr(adjusted[index], scale)
        old_bgra = np.dstack((old_preview, old_valid.astype(np.uint8) * 255))
        new_bgra = np.dstack((new_preview, new_valid.astype(np.uint8) * 255))
        evidence = video_dis_pair_evidence(old_bgra, new_bgra, old_valid & new_valid)
        if evidence is None:
            audits.append({"old_frame_id": adjusted[index - 1].frame_id, "new_frame_id": adjusted[index].frame_id, "accepted": False, "reason": "no_preview_dis_evidence"})
            continue
        # Fit in preview coordinates with equivalently scaled hard limits;
        # the accepted matrix is lifted to the full output grid exactly once.
        alignment_config = VideoLocalAlignmentConfig(
            background_displacement_target_px=6.0 / scale,
            background_displacement_hard_px=10.0 / scale,
            background_held_out_fb_target_px=1.25 / scale,
            background_held_out_fb_hard_px=2.0 / scale,
        )
        alignment = fit_background_alignment(evidence, config=alignment_config)
        if not alignment.audit.accepted or alignment.matrix is None:
            audits.append({"old_frame_id": adjusted[index - 1].frame_id, "new_frame_id": adjusted[index].frame_id, "accepted": False, "model": alignment.audit.selected_model, "reason": alignment.audit.rejection_reason})
            continue
        scaling = np.diag((float(scale), float(scale), 1.0))
        matrix = scaling @ alignment.matrix @ np.linalg.inv(scaling)
        support = adjusted[index - 1].valid_mask & adjusted[index].valid_mask
        adjusted[index] = _apply_output_matrix_to_grid(adjusted[index], matrix, support)
        audits.append({"old_frame_id": adjusted[index - 1].frame_id, "new_frame_id": adjusted[index].frame_id, "accepted": True, "model": alignment.audit.selected_model, "warning": alignment.audit.large_alignment_warning})
    outcome = tuple(adjusted)
    return (outcome, tuple(audits)) if return_audits else outcome


def render_video_v6_candidate(
    frames: tuple[object, ...], poses: tuple[np.ndarray, ...], calibration: object, *,
    pushbroom_config: dict[str, object], rgb_motions: list[object], motion_pixels_to_full_resolution: float,
) -> CalibratedRGBPushbroomResult:
    """Return the legacy result container while executing only the v6 path."""
    sources = build_v6_sampling_sources(
        frames, poses, calibration, pushbroom_config=pushbroom_config, rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    aligned_sources, alignment_audits = apply_v6_background_alignment_to_grids(sources, return_audits=True)
    result = render_video_v6_real_sources(aligned_sources)
    metadata = {
        "schema": "video-v6-rgb-only-graphcut/v1",
        "renderer": "v6_real_source_graphcut_once_sampling",
        "quality_metrics": {
            "quality_pass": result.quality.strict_quality_pass and not result.expanded_real_owner_pair_frame_ids,
            "strict_quality_pass": result.quality.strict_quality_pass and not result.expanded_real_owner_pair_frame_ids,
            "failure_reasons": list(result.quality.failure_reasons),
            "seam_step_p95_px": result.quality.seam_step_p95_px,
            "seam_step_abs_max_px": result.quality.seam_step_abs_max_px,
            "double_edge_count": result.quality.double_edge_count,
            "ghost_count": result.quality.ghost_count,
        },
        "v6_pair_graphcut": [
            {
                "graphcut_called": audit.graphcut_called,
                "accepted": audit.accepted,
                "maximum_adjacent_row_step_px": audit.maximum_adjacent_row_step_px,
                "owner_island_count": audit.owner_island_count,
                "small_fragment_count": audit.small_fragment_count,
            }
            for audit in result.graphcut_audits
        ],
        "raw_rgb_once_sampling": {
            "source_frame_ids": [source.frame_id for source in sources],
            "source_sampling_call_count": result.source_sampling_call_count,
            "exactly_once": True,
        },
        "expanded_real_owner_pair_frame_ids": [list(pair) for pair in result.expanded_real_owner_pair_frame_ids],
        "background_alignment": list(alignment_audits),
    }
    return CalibratedRGBPushbroomResult(result.bgr, metadata, owner_frame_id=result.owner_frame_id)


def render_video_v6_real_pair(
    old_source: VideoSamplingSource, new_source: VideoSamplingSource, *, object_mask: np.ndarray | None = None,
) -> VideoV6PairRenderResult:
    """Render one final pair through every v6 stage, without a legacy fallback."""
    (old_record, old_bgr), (new_record, new_bgr) = sample_video_sources_once((old_source, new_source))
    old_valid, new_valid = np.asarray(old_record.valid_mask, bool), np.asarray(new_record.valid_mask, bool)
    overlap = old_valid & new_valid
    rows, columns = np.where(overlap)
    if rows.size == 0:
        raise RuntimeError("v6 pair has no real common support for GraphCut")
    # GraphCut's frozen domain is always the complete calibrated 480px image
    # height.  Invalid top/bottom cells remain masked; shrinking to observed
    # overlap would silently violate that domain when pose levelling clips a
    # few rows.
    full_left, full_right = int(columns.min()), int(columns.max()) + 1
    corridor_width = max(96, min(160, full_right - full_left))
    centre = (full_left + full_right) // 2
    left = max(0, min(old_valid.shape[1] - corridor_width, centre - corridor_width // 2))
    right = left + corridor_width
    top, bottom = 0, old_valid.shape[0]
    old_crop, new_crop = old_bgr[top:bottom, left:right], new_bgr[top:bottom, left:right]
    old_crop_valid, new_crop_valid = old_valid[top:bottom, left:right], new_valid[top:bottom, left:right]
    old_bgra = np.dstack((old_crop, old_crop_valid.astype(np.uint8) * 255))
    new_bgra = np.dstack((new_crop, new_crop_valid.astype(np.uint8) * 255))
    evidence = video_dis_pair_evidence(old_bgra, new_bgra, old_crop_valid & new_crop_valid)
    if evidence is None:
        raise RuntimeError("v6 pair did not produce its required F/B DIS evidence")
    object_crop = None if object_mask is None else np.asarray(object_mask, bool)[top:bottom, left:right]
    guards = build_video_hard_guards(old_crop, new_crop, evidence, object_mask=object_crop)
    graphcut = solve_video_graphcut_seam(old_crop, new_crop, old_crop_valid, new_crop_valid, hard_owner_old=guards.hard_owner_old, hard_owner_new=guards.hard_owner_new)
    graphcut = replace(graphcut, audit=replace(graphcut.audit, canvas_x_offset=left))
    if not graphcut.audit.accepted or audit_guard_owner_intersection(graphcut.choose_new, guards):
        raise RuntimeError("v6 GraphCut pair failed topology or hard-guard ownership")
    owner = np.full(old_valid.shape, -1, np.int32)
    owner[old_valid] = int(old_source.frame_id)
    crop_owner = owner[top:bottom, left:right]
    crop_owner[graphcut.choose_new] = int(new_source.frame_id)
    owner[top:bottom, left:right] = crop_owner
    owner[new_valid & ~old_valid] = int(new_source.frame_id)
    base = old_bgr.copy()
    base[owner == int(new_source.frame_id)] = new_bgr[owner == int(new_source.frame_id)]
    eligible = build_near_blend_eligible_mask(old_crop_valid, new_crop_valid, evidence, guards)
    blended_crop, _, blend_audit = apply_near_multiband(old_crop, new_crop, base[top:bottom, left:right], graphcut.choose_new, eligible, guards)
    base[top:bottom, left:right] = blended_crop
    valid = owner >= 0
    quality = assess_video_rgb_quality(base, owner, valid, (graphcut.audit,))
    return VideoV6PairRenderResult(base, owner, valid, evidence, graphcut.audit, quality, blend_audit.band_pixel_count, 2)


def render_video_v6_real_sources(sources: tuple[VideoSamplingSource, ...]) -> VideoV6RenderResult:
    """Compose a chronological v6 source chain after exactly one sampling per source."""
    if len(sources) < 2:
        raise ValueError("v6 renderer requires at least two chronological real sources")
    sampled = sample_video_sources_once(sources)
    first, first_bgr = sampled[0]
    owner = np.full(first.valid_mask.shape, -1, np.int32)
    owner[first.valid_mask] = int(first.frame_id)
    output = first_bgr.copy()
    audits: list[VideoGraphCutAudit] = []
    expanded_owner_pairs: list[tuple[int, int]] = []
    for (old_source, old_bgr), (new_source, new_bgr) in zip(sampled[:-1], sampled[1:], strict=True):
        old_valid, new_valid = np.asarray(old_source.valid_mask, bool), np.asarray(new_source.valid_mask, bool)
        overlap = old_valid & new_valid
        rows, columns = np.where(overlap)
        if rows.size == 0:
            raise RuntimeError("adjacent v6 real sources have no common GraphCut support")
        full_left, full_right = int(columns.min()), int(columns.max()) + 1
        corridor_width = max(96, min(160, full_right - full_left))
        centre = (full_left + full_right) // 2
        left = max(0, min(old_valid.shape[1] - corridor_width, centre - corridor_width // 2))
        right = left + corridor_width
        top, bottom = 0, old_valid.shape[0]
        old_crop, new_crop = old_bgr[top:bottom, left:right], new_bgr[top:bottom, left:right]
        old_crop_valid, new_crop_valid = old_valid[top:bottom, left:right], new_valid[top:bottom, left:right]
        evidence = video_dis_pair_evidence(np.dstack((old_crop, old_crop_valid.astype(np.uint8) * 255)), np.dstack((new_crop, new_crop_valid.astype(np.uint8) * 255)), old_crop_valid & new_crop_valid)
        if evidence is None:
            raise RuntimeError("adjacent v6 pair did not produce required F/B DIS evidence")
        guards = build_video_hard_guards(old_crop, new_crop, evidence)
        graphcut = solve_video_graphcut_seam(old_crop, new_crop, old_crop_valid, new_crop_valid, hard_owner_old=guards.hard_owner_old, hard_owner_new=guards.hard_owner_new)
        graphcut = replace(graphcut, audit=replace(graphcut.audit, canvas_x_offset=left))
        guard_violation = audit_guard_owner_intersection(graphcut.choose_new, guards)
        if guard_violation:
            raise RuntimeError(
                "v6 GraphCut chain pair failed "
                f"{old_source.frame_id}->{new_source.frame_id}: "
                f"reason={graphcut.audit.rejection_reason}, "
                f"row_step={graphcut.audit.maximum_adjacent_row_step_px}, "
                f"islands={graphcut.audit.owner_island_count}, "
                f"small_fragments={graphcut.audit.small_fragment_count}, "
                f"guard_violations={guard_violation}"
            )
        if not graphcut.audit.accepted:
            # Required recovery stage 2: retain the failed GraphCut evidence,
            # then extend one existing real source across this corridor.  This
            # is explicitly degraded, never a synthetic seam success or DP
            # replacement; a later planner may instead insert one direct-ORB
            # rescue source.
            graphcut.choose_new[:] = False
            expanded_owner_pairs.append((int(old_source.frame_id), int(new_source.frame_id)))
        crop_owner = owner[top:bottom, left:right]
        crop_owner[graphcut.choose_new] = int(new_source.frame_id)
        owner[top:bottom, left:right] = crop_owner
        owner[new_valid & ~old_valid] = int(new_source.frame_id)
        output[owner == int(new_source.frame_id)] = new_bgr[owner == int(new_source.frame_id)]
        eligible = build_near_blend_eligible_mask(old_crop_valid, new_crop_valid, evidence, guards)
        blended, _, _ = apply_near_multiband(old_crop, new_crop, output[top:bottom, left:right], graphcut.choose_new, eligible, guards)
        output[top:bottom, left:right] = blended
        audits.append(graphcut.audit)
    valid = owner >= 0
    effective_audits = tuple(audit for audit in audits if audit.accepted)
    return VideoV6RenderResult(output, owner, valid, tuple(audits), assess_video_rgb_quality(output, owner, valid, effective_audits), len(sampled), tuple(expanded_owner_pairs))


__all__ = ["VideoV6PairRenderResult", "VideoV6RenderResult", "apply_v6_background_alignment_to_grids", "build_v6_sampling_sources", "render_video_v6_candidate", "render_video_v6_real_pair", "render_video_v6_real_sources"]
