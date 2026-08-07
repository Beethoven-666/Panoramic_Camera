"""End-to-end v6 candidate renderer for one adjacent real-source pair."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .video_final_sampling import VideoSamplingSource, sample_video_sources_once
from .video_graphcut_seam import VideoGraphCutAudit, solve_video_graphcut_seam
from .video_hard_guards import audit_guard_owner_intersection, build_video_hard_guards
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
    top, bottom, left, right = int(rows.min()), int(rows.max()) + 1, int(columns.min()), int(columns.max()) + 1
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
    for (old_source, old_bgr), (new_source, new_bgr) in zip(sampled[:-1], sampled[1:], strict=True):
        old_valid, new_valid = np.asarray(old_source.valid_mask, bool), np.asarray(new_source.valid_mask, bool)
        overlap = old_valid & new_valid
        rows, columns = np.where(overlap)
        if rows.size == 0:
            raise RuntimeError("adjacent v6 real sources have no common GraphCut support")
        top, bottom, left, right = int(rows.min()), int(rows.max()) + 1, int(columns.min()), int(columns.max()) + 1
        old_crop, new_crop = old_bgr[top:bottom, left:right], new_bgr[top:bottom, left:right]
        old_crop_valid, new_crop_valid = old_valid[top:bottom, left:right], new_valid[top:bottom, left:right]
        evidence = video_dis_pair_evidence(np.dstack((old_crop, old_crop_valid.astype(np.uint8) * 255)), np.dstack((new_crop, new_crop_valid.astype(np.uint8) * 255)), old_crop_valid & new_crop_valid)
        if evidence is None:
            raise RuntimeError("adjacent v6 pair did not produce required F/B DIS evidence")
        guards = build_video_hard_guards(old_crop, new_crop, evidence)
        graphcut = solve_video_graphcut_seam(old_crop, new_crop, old_crop_valid, new_crop_valid, hard_owner_old=guards.hard_owner_old, hard_owner_new=guards.hard_owner_new)
        if not graphcut.audit.accepted or audit_guard_owner_intersection(graphcut.choose_new, guards):
            raise RuntimeError("v6 GraphCut chain pair failed topology or hard guards")
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
    return VideoV6RenderResult(output, owner, valid, tuple(audits), assess_video_rgb_quality(output, owner, valid, audits), len(sampled))


__all__ = ["VideoV6PairRenderResult", "VideoV6RenderResult", "render_video_v6_real_pair", "render_video_v6_real_sources"]
