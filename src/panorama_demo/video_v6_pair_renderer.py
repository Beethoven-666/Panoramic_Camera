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
from .video_local_alignment import (
    VideoLocalAlignmentConfig,
    fit_background_alignment,
    fit_near_protected_alignment,
)
from .video_near_blend import apply_near_multiband, build_near_blend_eligible_mask
from .video_object_mask import VideoObjectMaskConfig, VideoObjectMaskResult, build_video_object_masks
from .video_photometric import (
    AdjacentBGRAOverlap,
    apply_video_photometric_correction,
    solve_video_global_photometric,
)
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
    photometric_audit: dict[str, object]
    prepared_pairs: tuple[object, ...]


@dataclass(frozen=True)
class _PreparedV6Pair:
    old_source: VideoSamplingSource
    new_source: VideoSamplingSource
    left: int
    right: int
    old_valid: np.ndarray
    new_valid: np.ndarray
    old_crop_valid: np.ndarray
    new_crop_valid: np.ndarray
    evidence: VideoDISPairEvidence
    guards: object
    photometric_protection: np.ndarray
    graphcut_audit: VideoGraphCutAudit
    choose_new: np.ndarray
    owner_expanded: bool
    near_ladder_audit: object
    object_masks: VideoObjectMaskResult


def _safe_photometric_background(pair: _PreparedV6Pair) -> np.ndarray:
    """Return common, low-residual background samples for a pair fit."""

    return (
        pair.old_crop_valid & pair.new_crop_valid & pair.evidence.reliable_mask
        & np.isfinite(pair.evidence.fb_error) & (pair.evidence.fb_error <= 0.75)
        & np.isfinite(pair.evidence.rgb_residual) & (pair.evidence.rgb_residual <= 20.0)
        & ~pair.evidence.occlusion_risk_mask & ~pair.photometric_protection
    )


def _photometric_matched_right(
    right_bgr: np.ndarray, right_valid: np.ndarray, evidence: VideoDISPairEvidence,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample right evidence at cached forward-DIS correspondences only."""

    height, width = right_valid.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    flow = np.asarray(evidence.flow_forward, dtype=np.float32)
    if flow.shape != (height, width, 2):
        raise ValueError("photometric DIS flow must match the right evidence crop")
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    matched_bgr = cv2.remap(
        np.asarray(right_bgr), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    matched_valid = cv2.remap(
        np.asarray(right_valid, dtype=np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    return matched_bgr, matched_valid


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
            reason = alignment.audit.rejection_reason
            if alignment.audit.accepted and alignment.mesh_displacement is not None:
                reason = "bounded_mesh_lacks_full_resolution_line_and_error_qualification"
            audits.append({"old_frame_id": adjusted[index - 1].frame_id, "new_frame_id": adjusted[index].frame_id, "accepted": False, "model": alignment.audit.selected_model, "reason": reason})
            continue
        support = adjusted[index - 1].valid_mask & adjusted[index].valid_mask
        scaling = np.diag((float(scale), float(scale), 1.0))
        matrix = scaling @ alignment.matrix @ np.linalg.inv(scaling)
        adjusted[index] = _apply_output_matrix_to_grid(adjusted[index], matrix, support)
        audits.append({"old_frame_id": adjusted[index - 1].frame_id, "new_frame_id": adjusted[index].frame_id, "accepted": True, "model": alignment.audit.selected_model, "warning": alignment.audit.large_alignment_warning})
    outcome = tuple(adjusted)
    return (outcome, tuple(audits)) if return_audits else outcome


def _preview_near_alignment_config(scale: int) -> VideoLocalAlignmentConfig:
    return VideoLocalAlignmentConfig(
        near_translation_target_px=3.0 / scale,
        near_translation_hard_px=6.0 / scale,
        near_homography_corner_displacement_hard_px=6.0 / scale,
        near_homography_held_out_fb_p95_max_px=1.0 / scale,
        near_homography_held_out_fb_abs_max_px=2.0 / scale,
    )


def apply_v6_near_alignment_to_grids(
    sources: tuple[VideoSamplingSource, ...], *, return_audits: bool = False,
) -> tuple[VideoSamplingSource, ...] | tuple[
    tuple[VideoSamplingSource, ...], dict[tuple[int, int], tuple[np.ndarray, np.ndarray]], tuple[dict[str, object], ...],
]:
    """Apply only audited object-local matrices to final inverse grids."""

    adjusted = list(sources)
    owner_masks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    audits: list[dict[str, object]] = []
    scale = 4
    for index in range(1, len(adjusted)):
        old_preview, old_valid = _preview_bgr(adjusted[index - 1], scale)
        new_preview, new_valid = _preview_bgr(adjusted[index], scale)
        evidence = video_dis_pair_evidence(
            np.dstack((old_preview, old_valid.astype(np.uint8) * 255)),
            np.dstack((new_preview, new_valid.astype(np.uint8) * 255)), old_valid & new_valid,
        )
        old_id, new_id = adjusted[index - 1].frame_id, adjusted[index].frame_id
        if evidence is None:
            audits.append({"old_frame_id": old_id, "new_frame_id": new_id, "accepted": False, "reason": "no_preview_dis_evidence"})
            continue
        base_guards = build_video_hard_guards(
            old_preview, new_preview, evidence, old_valid=old_valid, new_valid=new_valid,
        )
        objects = build_video_object_masks(
            evidence, strong_protection=base_guards.protected,
            config=VideoObjectMaskConfig(minimum_component_pixels=16, residual_floor_px=1.5 / scale),
        )
        support = old_valid & new_valid & objects.candidate_mask & ~base_guards.protected
        near = fit_near_protected_alignment(
            evidence, support=support, plane_verified=bool(objects.homography_mask.any()),
            config=_preview_near_alignment_config(scale),
        )
        if not near.audit.accepted or near.matrix is None or near.audit.selected_model == "identity":
            audits.append({"old_frame_id": old_id, "new_frame_id": new_id, "accepted": False, "model": near.audit.selected_model, "reason": near.audit.rejection_reason})
            continue
        scaling = np.diag((float(scale), float(scale), 1.0))
        full_matrix = scaling @ near.matrix @ np.linalg.inv(scaling)
        candidate = cv2.resize(
            objects.candidate_mask.astype(np.uint8),
            (adjusted[index].valid_mask.shape[1], adjusted[index].valid_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        protected = cv2.resize(
            objects.protected_mask.astype(np.uint8),
            (adjusted[index].valid_mask.shape[1], adjusted[index].valid_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        adjusted[index] = _apply_output_matrix_to_grid(adjusted[index], full_matrix, candidate)
        owner_masks[(int(old_id), int(new_id))] = (candidate, protected)
        audits.append({"old_frame_id": old_id, "new_frame_id": new_id, "accepted": True, "model": near.audit.selected_model, "homography_eligible": bool(objects.homography_mask.any())})
    outcome = tuple(adjusted)
    return (outcome, owner_masks, tuple(audits)) if return_audits else outcome


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
    near_sources, near_owner_masks, near_alignment_audits = apply_v6_near_alignment_to_grids(
        aligned_sources, return_audits=True,
    )
    result = render_video_v6_real_sources(near_sources, near_owner_masks=near_owner_masks)
    effective_observations = iter(result.quality.seam_observations)
    pair_metadata: list[dict[str, object]] = []
    for old_source, new_source, audit, prepared in zip(
        sources[:-1], sources[1:], result.graphcut_audits, result.prepared_pairs, strict=True,
    ):
        observation = next(effective_observations, None) if audit.accepted else None
        pair_metadata.append({
            "old_frame_id": old_source.frame_id,
            "new_frame_id": new_source.frame_id,
            "graphcut_called": audit.graphcut_called,
            "accepted": audit.accepted,
            "rejection_reason": audit.rejection_reason,
            "valid_pixel_exactly_one_owner": audit.valid_pixel_exactly_one_owner,
            "rescue_corridor_used": audit.rescue_corridor_used,
            "canvas_x_offset": audit.canvas_x_offset,
            "maximum_adjacent_row_step_px": audit.maximum_adjacent_row_step_px,
            "owner_island_count": audit.owner_island_count,
            "small_fragment_count": audit.small_fragment_count,
            "double_edge_count": None if observation is None else observation.double_edge_count,
            "ghost_count": None if observation is None else observation.ghost_count,
            "evaluated_seam_rows": None if observation is None else observation.evaluated_row_count,
            "near_ladder": {
                "selected_model": prepared.near_ladder_audit.selected_model,
                "accepted": prepared.near_ladder_audit.accepted,
                "rejection_reason": prepared.near_ladder_audit.rejection_reason,
                "held_out_residual_p95_px": prepared.near_ladder_audit.held_out_residual_p95_px,
                "held_out_residual_abs_max_px": prepared.near_ladder_audit.held_out_residual_abs_max_px,
                "maximum_displacement_px": prepared.near_ladder_audit.maximum_displacement_px,
            },
            "object_mask": {
                "residual_threshold_px": prepared.object_masks.residual_threshold_px,
                "candidate_pixel_count": int(prepared.object_masks.candidate_mask.sum()),
                "protected_pixel_count": int(prepared.object_masks.protected_mask.sum()),
                "homography_pixel_count": int(prepared.object_masks.homography_mask.sum()),
                "components": [
                    {
                        "area_pixels": component.area_pixels,
                        "bounding_box_xywh": list(component.bounding_box_xywh),
                        "collar_px": component.collar_px,
                        "stable_across_pair": component.stable_across_pair,
                        "rectangular": component.rectangular,
                        "homography_eligible": component.homography_eligible,
                    }
                    for component in prepared.object_masks.components
                ],
            },
        })
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
        "v6_pair_graphcut": pair_metadata,
        "raw_rgb_once_sampling": {
            "source_frame_ids": [source.frame_id for source in sources],
            "source_sampling_call_count": result.source_sampling_call_count,
            "exactly_once": True,
        },
        "expanded_real_owner_pair_frame_ids": [list(pair) for pair in result.expanded_real_owner_pair_frame_ids],
        "background_alignment": list(alignment_audits),
        "near_protected_alignment": list(near_alignment_audits),
        "video_global_photometric": result.photometric_audit,
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
    guards = build_video_hard_guards(
        old_crop, new_crop, evidence, object_mask=object_crop,
        old_valid=old_crop_valid, new_valid=new_crop_valid,
    )
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


def render_video_v6_real_sources(
    sources: tuple[VideoSamplingSource, ...], *,
    near_owner_masks: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None = None,
) -> VideoV6RenderResult:
    """Compose a chronological v6 source chain after exactly one sampling per source."""
    if len(sources) < 2:
        raise ValueError("v6 renderer requires at least two chronological real sources")
    sampled = sample_video_sources_once(sources)
    prepared_pairs: list[_PreparedV6Pair] = []
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
        base_guards = build_video_hard_guards(
            old_crop, new_crop, evidence, old_valid=old_crop_valid, new_valid=new_crop_valid,
        )
        object_masks = build_video_object_masks(
            evidence, strong_protection=base_guards.protected,
        )
        preferred_new = None
        stored = (near_owner_masks or {}).get((int(old_source.frame_id), int(new_source.frame_id)))
        protected_object = object_masks.protected_mask
        if stored is not None:
            stored_candidate, stored_protected = stored
            preferred_new = np.asarray(stored_candidate, bool)[:, left:right]
            protected_object |= np.asarray(stored_protected, bool)[:, left:right]
        guards = build_video_hard_guards(
            old_crop, new_crop, evidence, object_mask=protected_object, prefer_new_mask=preferred_new,
            old_valid=old_crop_valid, new_valid=new_crop_valid,
        )
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
        near_audit = fit_near_protected_alignment(
            evidence,
            support=old_crop_valid & new_crop_valid & object_masks.candidate_mask & ~base_guards.protected,
            plane_verified=bool(object_masks.homography_mask.any()),
        ).audit
        prepared_pairs.append(_PreparedV6Pair(
            old_source, new_source, left, right, old_valid, new_valid,
            old_crop_valid, new_crop_valid, evidence, guards, base_guards.protected, graphcut.audit,
            graphcut.choose_new.copy(), not graphcut.audit.accepted, near_audit, object_masks,
        ))
        audits.append(graphcut.audit)

    overlaps: list[AdjacentBGRAOverlap] = []
    for index, pair in enumerate(prepared_pairs):
        old_bgr = sampled[index][1]
        new_bgr = sampled[index + 1][1]
        old_crop = old_bgr[:, pair.left:pair.right]
        new_crop = new_bgr[:, pair.left:pair.right]
        matched_new_crop, matched_new_valid = _photometric_matched_right(
            new_crop, pair.new_crop_valid, pair.evidence,
        )
        safe_background = _safe_photometric_background(pair) & matched_new_valid
        overlaps.append(AdjacentBGRAOverlap(
            index, index + 1,
            np.dstack((old_crop, pair.old_crop_valid.astype(np.uint8) * 255)),
            np.dstack((matched_new_crop, matched_new_valid.astype(np.uint8) * 255)),
            safe_background, safe_background,
        ))
    photometric = solve_video_global_photometric(len(sampled), overlaps)
    corrected: list[np.ndarray] = []
    for index, (source, bgr) in enumerate(sampled):
        if not photometric.accepted:
            corrected.append(bgr)
            continue
        bgra = np.dstack((bgr, np.asarray(source.valid_mask, bool).astype(np.uint8) * 255))
        corrected.append(apply_video_photometric_correction(bgra, photometric.corrections[index])[:, :, :3])

    first = sampled[0][0]
    owner = np.full(first.valid_mask.shape, -1, np.int32)
    owner[first.valid_mask] = int(first.frame_id)
    output = corrected[0].copy()
    for index, pair in enumerate(prepared_pairs):
        new_bgr = corrected[index + 1]
        crop_owner = owner[:, pair.left:pair.right]
        crop_owner[pair.choose_new] = int(pair.new_source.frame_id)
        owner[:, pair.left:pair.right] = crop_owner
        owner[pair.new_valid & ~pair.old_valid] = int(pair.new_source.frame_id)
        output[owner == int(pair.new_source.frame_id)] = new_bgr[owner == int(pair.new_source.frame_id)]
        old_crop = corrected[index][:, pair.left:pair.right]
        new_crop = new_bgr[:, pair.left:pair.right]
        eligible = build_near_blend_eligible_mask(pair.old_crop_valid, pair.new_crop_valid, pair.evidence, pair.guards)
        blended, _, _ = apply_near_multiband(old_crop, new_crop, output[:, pair.left:pair.right], pair.choose_new, eligible, pair.guards)
        output[:, pair.left:pair.right] = blended
    valid = owner >= 0
    effective_audits = tuple(audit for audit in audits if audit.accepted)
    return VideoV6RenderResult(
        output, owner, valid, tuple(audits),
        assess_video_rgb_quality(output, owner, valid, effective_audits), len(sampled),
        tuple(expanded_owner_pairs), photometric.audit, tuple(prepared_pairs),
    )


__all__ = ["VideoV6PairRenderResult", "VideoV6RenderResult", "apply_v6_background_alignment_to_grids", "build_v6_sampling_sources", "render_video_v6_candidate", "render_video_v6_real_pair", "render_video_v6_real_sources"]
