"""Complete-canvas, fail-closed v6.1 tail-guarded candidate renderer.

This is an experiment-only renderer.  It consumes the same direct-ORB source
grids used by v6; source provenance/Open3D adjacency are enforced by the video
orchestrator before this module is reached.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from .calibrated_rgb_pushbroom import CalibratedRGBPushbroomResult
from .video_final_sampling import VideoSamplingSource, sample_video_sources_once
from .video_graphcut_seam import VideoGraphCutAudit, solve_video_graphcut_seam
from .video_hard_guards import VideoHardGuards, audit_guard_owner_intersection, build_video_hard_guards
from .video_local_alignment import fit_near_protected_alignment
from .video_near_blend import VideoNearBlendConfig, apply_near_multiband, build_near_blend_eligible_mask
from .video_rgb_quality import assess_video_rgb_quality
from .video_v61_geometry_gate import V61GeometryAudit, evaluate_v61_geometry_gate
from .video_v6_pair_renderer import build_v6_sampling_sources
from .video_visual_renderer import video_dis_pair_evidence


@dataclass(frozen=True)
class V61PairState:
    old_frame_id: int
    new_frame_id: int
    gate_state: str
    geometry: V61GeometryAudit | None
    alignment_accepted: bool
    alignment_model: str
    graphcut_called: bool
    graphcut_accepted: bool
    fallback_reason: str | None
    blend_pixel_count: int
    actual_old_owner_pixels: int
    actual_new_owner_pixels: int
    not_evaluable_reason: str | None


def _with_tail_guard(guards: VideoHardGuards, tail: np.ndarray) -> VideoHardGuards:
    protected = np.asarray(guards.protected, bool) | np.asarray(tail, bool)
    # A tail sample has no reliable colour correspondence.  Keep its old real
    # owner (unless old is invalid, which is impossible inside common support).
    return replace(guards, protected=protected, hard_owner_old=np.asarray(guards.hard_owner_old, bool) | tail)


def _real_internal_handoff(audit: VideoGraphCutAudit, choose_new: np.ndarray, width: int) -> bool:
    seam = tuple(int(value) for value in audit.seam_x_by_row)
    return bool(
        np.any(choose_new) and np.any(~choose_new)
        and len(seam) == choose_new.shape[0]
        and all(4 <= value <= width - 4 for value in seam)
    )


def render_video_v61_real_sources(sources: tuple[VideoSamplingSource, ...]) -> CalibratedRGBPushbroomResult:
    """Render N direct real sources; every pair ends graphcut/degraded/F.

    Structural source violations raise (the pipeline turns them into F).  A
    geometric or GraphCut failure instead expands the incoming real source as
    a hard owner and marks the result C-compatible with zero blend pixels.
    """
    if len(sources) < 2:
        raise ValueError("v6.1 requires at least two direct real sources")
    ids = tuple(int(source.frame_id) for source in sources)
    if ids != tuple(sorted(set(ids))):
        raise RuntimeError("v6.1 structural_failure: source ids must be unique and chronological")
    sampled = sample_video_sources_once(sources)
    owner = np.full(sources[0].valid_mask.shape, -1, np.int32)
    owner[np.asarray(sources[0].valid_mask, bool)] = ids[0]
    output = sampled[0][1].copy()
    states: list[V61PairState] = []
    quality_audits: list[VideoGraphCutAudit] = []
    for index, ((old_source, old_bgr), (new_source, new_bgr)) in enumerate(zip(sampled[:-1], sampled[1:], strict=True)):
        old_valid, new_valid = np.asarray(old_source.valid_mask, bool), np.asarray(new_source.valid_mask, bool)
        common = old_valid & new_valid
        rows, columns = np.where(common)
        if rows.size == 0:
            # A disjoint source is a valid real owner extension, not a pose or
            # topology failure.  No GraphCut is meaningful.
            owner[new_valid] = int(new_source.frame_id)
            output[new_valid] = new_bgr[new_valid]
            states.append(V61PairState(int(old_source.frame_id), int(new_source.frame_id), "hard_owner_degraded", None, False, "not_evaluable", False, False, "no_common_real_support", 0, 0, int(new_valid.sum()), "no_common_real_support"))
            continue
        width = min(160, int(columns.max()) - int(columns.min()) + 1)
        if width < 96:
            owner[new_valid] = int(new_source.frame_id)
            output[new_valid] = new_bgr[new_valid]
            states.append(V61PairState(int(old_source.frame_id), int(new_source.frame_id), "hard_owner_degraded", None, False, "not_evaluable", False, False, "corridor_below_96px", 0, 0, int(new_valid.sum()), "graphcut_corridor_unavailable"))
            continue
        left = max(0, min(old_valid.shape[1] - width, (int(columns.min()) + int(columns.max()) + 1 - width) // 2))
        right = left + width
        old_crop, new_crop = old_bgr[:, left:right], new_bgr[:, left:right]
        old_crop_valid, new_crop_valid = old_valid[:, left:right], new_valid[:, left:right]
        evidence = video_dis_pair_evidence(np.dstack((old_crop, old_crop_valid.astype(np.uint8) * 255)), np.dstack((new_crop, new_crop_valid.astype(np.uint8) * 255)), old_crop_valid & new_crop_valid)
        if evidence is None:
            owner[new_valid] = int(new_source.frame_id)
            output[new_valid] = new_bgr[new_valid]
            states.append(V61PairState(int(old_source.frame_id), int(new_source.frame_id), "hard_owner_degraded", None, False, "not_evaluable", False, False, "no_fb_dis_evidence", 0, 0, int(new_valid.sum()), "no_fb_dis_evidence"))
            continue
        base_guards = build_video_hard_guards(old_crop, new_crop, evidence, old_valid=old_crop_valid, new_valid=new_crop_valid)
        geometry = evaluate_v61_geometry_gate(old_crop, new_crop, evidence, support=old_crop_valid & new_crop_valid, protected=base_guards.protected)
        guards = _with_tail_guard(base_guards, geometry.tail_guard)
        alignment = fit_near_protected_alignment(evidence, support=old_crop_valid & new_crop_valid & ~guards.protected, plane_verified=False)
        if not geometry.accepted or not alignment.audit.accepted:
            reason = geometry.rejection_reason if not geometry.accepted else alignment.audit.rejection_reason
            owner[new_valid] = int(new_source.frame_id)
            output[new_valid] = new_bgr[new_valid]
            states.append(V61PairState(int(old_source.frame_id), int(new_source.frame_id), "hard_owner_degraded", geometry, alignment.audit.accepted, alignment.audit.selected_model, False, False, reason, 0, 0, int(new_valid.sum()), "geometry_or_alignment_gate"))
            continue
        graphcut = solve_video_graphcut_seam(old_crop, new_crop, old_crop_valid, new_crop_valid, hard_owner_old=guards.hard_owner_old, hard_owner_new=guards.hard_owner_new)
        graphcut = replace(graphcut, audit=replace(graphcut.audit, canvas_x_offset=left))
        safe_handoff = graphcut.audit.accepted and not audit_guard_owner_intersection(graphcut.choose_new, guards) and _real_internal_handoff(graphcut.audit, graphcut.choose_new, width)
        if not safe_handoff:
            owner[new_valid] = int(new_source.frame_id)
            output[new_valid] = new_bgr[new_valid]
            states.append(V61PairState(int(old_source.frame_id), int(new_source.frame_id), "hard_owner_degraded", geometry, True, alignment.audit.selected_model, True, False, graphcut.audit.rejection_reason or "no_real_internal_two_owner_handoff", 0, 0, int(new_valid.sum()), "graphcut_not_accepted"))
            continue
        crop_owner = owner[:, left:right]
        crop_owner[graphcut.choose_new] = int(new_source.frame_id)
        owner[:, left:right] = crop_owner
        owner[new_valid & ~old_valid] = int(new_source.frame_id)
        output[owner == int(new_source.frame_id)] = new_bgr[owner == int(new_source.frame_id)]
        eligible = build_near_blend_eligible_mask(old_crop_valid, new_crop_valid, evidence, guards)
        blended, _band, blend = apply_near_multiband(old_crop, new_crop, output[:, left:right], graphcut.choose_new, eligible, guards, config=VideoNearBlendConfig(near_width_px=2))
        output[:, left:right] = blended
        old_count, new_count = int(np.count_nonzero(~graphcut.choose_new & (old_crop_valid & new_crop_valid))), int(np.count_nonzero(graphcut.choose_new & (old_crop_valid & new_crop_valid)))
        states.append(V61PairState(int(old_source.frame_id), int(new_source.frame_id), "graphcut_accepted", geometry, True, alignment.audit.selected_model, True, True, None, blend.band_pixel_count, old_count, new_count, None))
        quality_audits.append(graphcut.audit)
    valid = owner >= 0
    if np.any(valid & ~np.isin(owner, ids)):
        raise RuntimeError("v6.1 structural_failure: owner referenced an illegal source")
    quality = assess_video_rgb_quality(output, owner, valid, tuple(quality_audits))
    degraded = any(state.gate_state == "hard_owner_degraded" for state in states)
    metadata = {"schema": "video-v61-tail-guarded-full-panorama/v1", "renderer": "V61_tail_guarded_full_panorama", "candidate_only": True, "pair_states": [asdict(state) for state in states], "quality_metrics": {"strict_quality_pass": bool(quality.strict_quality_pass and not degraded), "grade": "C" if degraded else "B", "manual_review_required": degraded, "seam_step_p95_px": quality.seam_step_p95_px, "double_edge_count": quality.double_edge_count, "ghost_count": quality.ghost_count}, "raw_rgb_once_sampling": {"source_frame_ids": list(ids), "source_sampling_call_count": len(sources), "exactly_once": True}}
    return CalibratedRGBPushbroomResult(output, metadata, owner_frame_id=owner)


def render_video_v61_candidate(frames: tuple[object, ...], poses: tuple[np.ndarray, ...], calibration: object, *, pushbroom_config: dict[str, object], rgb_motions: list[object], motion_pixels_to_full_resolution: float, **_ignored: object) -> CalibratedRGBPushbroomResult:
    sources = build_v6_sampling_sources(frames, poses, calibration, pushbroom_config=pushbroom_config, rgb_motions=rgb_motions, motion_pixels_to_full_resolution=motion_pixels_to_full_resolution)
    return render_video_v61_real_sources(sources)


__all__ = ["V61PairState", "render_video_v61_candidate", "render_video_v61_real_sources"]
