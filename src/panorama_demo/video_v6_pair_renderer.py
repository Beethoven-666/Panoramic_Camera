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
from .video_object_patch_planning import (
    VideoDirectSourceSupport,
    VideoObjectRegion,
    plan_object_patches,
)
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


def _photometric_background_audit(pair: _PreparedV6Pair) -> dict[str, int]:
    """Count each fixed photometric eligibility stage for report evidence."""

    common = pair.old_crop_valid & pair.new_crop_valid
    reliable = common & pair.evidence.reliable_mask
    fb_ok = reliable & np.isfinite(pair.evidence.fb_error) & (pair.evidence.fb_error <= 0.75)
    rgb_ok = fb_ok & np.isfinite(pair.evidence.rgb_residual) & (pair.evidence.rgb_residual <= 20.0)
    no_occlusion = rgb_ok & ~pair.evidence.occlusion_risk_mask
    safe = no_occlusion & ~pair.photometric_protection
    return {
        "common_valid_pixels": int(np.count_nonzero(common)),
        "reliable_pixels": int(np.count_nonzero(reliable)),
        "fb_target_pixels": int(np.count_nonzero(fb_ok)),
        "rgb_residual_pixels": int(np.count_nonzero(rgb_ok)),
        "nonoccluded_pixels": int(np.count_nonzero(no_occlusion)),
        "safe_background_pixels": int(np.count_nonzero(safe)),
    }


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


def _crop_dis_evidence(
    evidence: VideoDISPairEvidence, left: int, right: int,
) -> VideoDISPairEvidence:
    """Slice one already-computed F/B DIS observation into a GraphCut corridor."""

    if not 0 <= left < right <= evidence.fb_error.shape[1]:
        raise ValueError("DIS corridor slice is outside its cached evidence")
    return replace(
        evidence,
        flow_forward=evidence.flow_forward[:, left:right],
        flow_backward=evidence.flow_backward[:, left:right],
        fb_error=evidence.fb_error[:, left:right],
        rgb_residual=evidence.rgb_residual[:, left:right],
        gradient_residual=evidence.gradient_residual[:, left:right],
        occlusion_risk_mask=evidence.occlusion_risk_mask[:, left:right],
        correspondence_confidence=evidence.correspondence_confidence[:, left:right],
        reliable_mask=evidence.reliable_mask[:, left:right],
        sampled_new_bgra=evidence.sampled_new_bgra[:, left:right],
    )


def _low_structure_corridor_left(
    old_bgr: np.ndarray, new_bgr: np.ndarray, common_valid: np.ndarray, *,
    overlap_left: int, overlap_right: int, width: int, image_width: int,
) -> int:
    """Pick one legal GraphCut corridor using RGB structure only.

    This runs before the pair's single F/B DIS observation.  It neither warps
    RGB nor emits a pose: it merely avoids placing the permitted GraphCut
    search window on the densest Canny structure when an equally real common
    support window is available.
    """

    if width <= 0 or width > image_width or not 0 <= overlap_left < overlap_right <= image_width:
        raise ValueError("v6 structure corridor is outside the legal canvas")
    old_gray = cv2.cvtColor(np.asarray(old_bgr), cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(np.asarray(new_bgr), cv2.COLOR_BGR2GRAY)
    structure = cv2.Canny(old_gray, 80, 160) > 0
    structure |= cv2.Canny(new_gray, 80, 160) > 0
    valid = np.asarray(common_valid, bool)
    if valid.shape != structure.shape:
        raise ValueError("v6 structure corridor valid mask has wrong shape")
    centre_left = max(0, min(
        image_width - width, overlap_left + (overlap_right - overlap_left - width) // 2,
    ))
    minimum_left = max(0, overlap_left - width + 1)
    maximum_left = min(image_width - width, overlap_right - 1)
    candidates = set(range(minimum_left, maximum_left + 1, 4))
    candidates.add(centre_left)
    best: tuple[float, int] | None = None
    for left in sorted(candidates):
        right = left + width
        support = valid[:, left:right]
        if not support.any():
            continue
        support_count = int(np.count_nonzero(support))
        density = float(np.count_nonzero(structure[:, left:right] & support)) / float(support_count)
        # Retain a very small central preference so a nearly equal low-texture
        # corridor does not drift to a projection boundary.
        coverage_penalty = 0.10 * (1.0 - support_count / float(width * support.shape[0]))
        score = density + coverage_penalty + 0.01 * abs(left - centre_left) / max(1, width)
        candidate = (score, left)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("v6 structure corridor has no real common support")
    return best[1]


def _hard_frontality_supports(
    sources: tuple[VideoSamplingSource, ...], hard_spans: dict[int, tuple[int, int]],
) -> tuple[VideoDirectSourceSupport, ...]:
    """Map raw calibrated hard-frontality columns into output-canvas spans."""

    supports: list[VideoDirectSourceSupport] = []
    for source in sources:
        span = hard_spans.get(int(source.frame_id))
        if span is None:
            raise ValueError(f"v6 source {source.frame_id} has no hard-frontality span")
        raw_left, raw_right = (int(value) for value in span)
        if raw_right <= raw_left:
            raise ValueError("hard-frontality span must be non-empty")
        valid = np.asarray(source.valid_mask, bool)
        raw_x = np.asarray(source.inverse_x, np.float32)
        support = valid & np.isfinite(raw_x) & (raw_x >= raw_left) & (raw_x < raw_right)
        columns = np.flatnonzero(np.any(support, axis=0))
        if not columns.size:
            raise RuntimeError(f"v6 source {source.frame_id} has no canvas hard-frontality support")
        left, right = int(columns[0]), int(columns[-1]) + 1
        if not np.all(np.any(support, axis=0)[left:right]):
            raise RuntimeError(f"v6 source {source.frame_id} hard-frontality support is not continuous")
        supports.append(VideoDirectSourceSupport(int(source.frame_id), (float(left), float(right))))
    return tuple(supports)


def _object_patch_plans(
    prepared_pairs: tuple[object, ...], supports: tuple[VideoDirectSourceSupport, ...],
) -> list[dict[str, object]]:
    """Plan the minimal continuous hard-frontality cover for every object region."""

    plans: list[dict[str, object]] = []
    for pair in prepared_pairs:
        for component in pair.object_masks.components:
            x, _y, width, _height = component.bounding_box_xywh
            region = VideoObjectRegion(
                f"{pair.old_source.frame_id}_{pair.new_source.frame_id}_{component.label}",
                (float(pair.left + x), float(pair.left + x + width)), component.collar_px,
            )
            try:
                plan = plan_object_patches(region, supports)
            except RuntimeError as error:
                plans.append({
                    "object_id": region.object_id,
                    "requested_span_x": list(region.protected_span_x),
                    "accepted": False,
                    "reason": str(error),
                })
                continue
            payload = plan.as_dict()
            payload["accepted"] = True
            plans.append(payload)
    return plans


def _compact_object_owner_preference(
    object_masks: VideoObjectMaskResult, *, old_frame_id: int, new_frame_id: int,
    canvas_left: int, supports: tuple[VideoDirectSourceSupport, ...] | None,
) -> np.ndarray:
    """Prefer the new real source only for a complete compact object cover.

    The default for every protected object is the chronological old hard owner.
    A switch is allowed only when the object's *entire* connected component
    plus its context collar has a single, continuous hard-frontality cover and
    that cover is this adjacent new source.  Wide, unstable, cable/fan, and
    otherwise unplannable regions deliberately remain hard-owner-only.
    """

    preferred = np.zeros_like(object_masks.candidate_mask, dtype=bool)
    if supports is None:
        return preferred
    if len(object_masks.components) != len(object_masks.component_masks):
        raise RuntimeError("v6 object audit/mask component cardinality mismatch")
    canvas_right = canvas_left + object_masks.candidate_mask.shape[1]
    for audit, component in zip(object_masks.components, object_masks.component_masks, strict=True):
        x, _y, width, _height = audit.bounding_box_xywh
        object_left = canvas_left + x
        object_right = canvas_left + x + width
        region = VideoObjectRegion(
            f"{old_frame_id}_{new_frame_id}_{audit.label}",
            # The GraphCut corridor is the complete observable domain for
            # this decision.  A collar that reaches its edge is deliberately
            # clipped there rather than demanding invisible pixels beyond it.
            (float(max(canvas_left, object_left - audit.collar_px)),
             float(min(canvas_right, object_right + audit.collar_px))),
            0,
        )
        try:
            plan = plan_object_patches(region, supports)
        except RuntimeError:
            continue
        if plan.final_replanned_n_req != 1 or plan.source_frame_ids != (new_frame_id,):
            continue
        collar_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (audit.collar_px * 2 + 1, audit.collar_px * 2 + 1),
        )
        # Prefer the same all-connected object and automatic collar that is
        # protected from GraphCut.  Source validity remains enforced by the
        # hard-guard builder.
        component_with_collar = cv2.dilate(
            np.asarray(component, np.uint8), collar_kernel,
        ).astype(bool)
        # A free-standing new-owner island would require two extra seams around
        # the object and has already proven prone to GraphCut topology failure
        # on the frozen T1 data.  Only allow the compact source switch when it
        # is connected to the corridor boundary, so it extends the new-source
        # region through a single monotone seam rather than creating an island.
        if not (component_with_collar[:, 0].any() or component_with_collar[:, -1].any()):
            continue
        preferred |= component_with_collar
    return preferred


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


def _apply_output_mesh_to_grid(
    source: VideoSamplingSource, mesh: np.ndarray, support: np.ndarray, *, preview_scale: int,
) -> VideoSamplingSource:
    """Compose a bounded preview mesh into a final inverse grid once.

    The mesh is correspondence evidence, not a free RGB flow warp.  Its
    displacement is lifted into the output coordinate system and is applied
    only to the independently safe background support supplied by the caller.
    Protected lines, thin structures and occlusions retain their original
    inverse-grid coordinates exactly.
    """

    height, width = source.valid_mask.shape
    field = cv2.resize(
        np.asarray(mesh, np.float32), (width, height), interpolation=cv2.INTER_LINEAR,
    ) * float(preview_scale)
    yy, xx = np.indices((height, width), dtype=np.float32)
    target_x = xx + field[..., 0]
    target_y = yy + field[..., 1]
    adjusted_x = cv2.remap(
        source.inverse_x.astype(np.float32), target_x, target_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0,
    )
    adjusted_y = cv2.remap(
        source.inverse_y.astype(np.float32), target_x, target_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0,
    )
    adjusted_valid = cv2.remap(
        source.valid_mask.astype(np.uint8), target_x, target_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ) > 0
    mask = np.asarray(support, bool)
    if mask.shape != source.valid_mask.shape:
        raise ValueError("mesh safe support must match the final inverse grid")
    return VideoSamplingSource(
        source.frame_id, source.raw_bgr,
        np.where(mask, adjusted_x, source.inverse_x), np.where(mask, adjusted_y, source.inverse_y),
        np.where(mask, adjusted_valid, source.valid_mask),
    )


def _lift_preview_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Lift a preview-only audit mask without interpolating its protection."""

    height, width = shape
    return cv2.resize(
        np.asarray(mask, np.uint8), (width, height), interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


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
        guards = build_video_hard_guards(
            old_preview, new_preview, evidence, old_valid=old_valid, new_valid=new_valid,
        )
        safe_preview = old_valid & new_valid & ~guards.protected
        # Fit in preview coordinates with equivalently scaled hard limits;
        # the accepted matrix is lifted to the full output grid exactly once.
        alignment_config = VideoLocalAlignmentConfig(
            background_displacement_target_px=6.0 / scale,
            background_displacement_hard_px=10.0 / scale,
            background_held_out_fb_target_px=1.25 / scale,
            background_held_out_fb_hard_px=2.0 / scale,
        )
        alignment = fit_background_alignment(evidence, support=safe_preview, config=alignment_config)
        if not alignment.audit.accepted:
            reason = alignment.audit.rejection_reason
            audits.append({"old_frame_id": adjusted[index - 1].frame_id, "new_frame_id": adjusted[index].frame_id, "accepted": False, "model": alignment.audit.selected_model, "reason": reason})
            continue
        support = _lift_preview_mask(safe_preview, adjusted[index].valid_mask.shape)
        scaling = np.diag((float(scale), float(scale), 1.0))
        if alignment.matrix is not None:
            matrix = scaling @ alignment.matrix @ np.linalg.inv(scaling)
            adjusted[index] = _apply_output_matrix_to_grid(adjusted[index], matrix, support)
        elif alignment.mesh_displacement is not None:
            adjusted[index] = _apply_output_mesh_to_grid(
                adjusted[index], alignment.mesh_displacement, support, preview_scale=scale,
            )
        else:
            audits.append({"old_frame_id": adjusted[index - 1].frame_id, "new_frame_id": adjusted[index].frame_id, "accepted": False, "model": alignment.audit.selected_model, "reason": "accepted_alignment_has_no_grid_model"})
            continue
        audits.append({
            "old_frame_id": adjusted[index - 1].frame_id,
            "new_frame_id": adjusted[index].frame_id,
            "accepted": True,
            "model": alignment.audit.selected_model,
            "warning": alignment.audit.large_alignment_warning,
            "protected_grid_warp_pixels": 0,
            "safe_background_grid_warp_pixels": int(support.sum()),
        })
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
    frontality_hard_spans: dict[int, tuple[int, int]] | None = None,
) -> CalibratedRGBPushbroomResult:
    """Return the legacy result container while executing only the v6 path."""
    sources = build_v6_sampling_sources(
        frames, poses, calibration, pushbroom_config=pushbroom_config, rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    hard_frontality = None if frontality_hard_spans is None else _hard_frontality_supports(
        sources, frontality_hard_spans,
    )
    aligned_sources, alignment_audits = apply_v6_background_alignment_to_grids(sources, return_audits=True)
    near_sources, near_owner_masks, near_alignment_audits = apply_v6_near_alignment_to_grids(
        aligned_sources, return_audits=True,
    )
    result = render_video_v6_real_sources(
        near_sources, near_owner_masks=near_owner_masks, frontality_supports=hard_frontality,
    )
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
            "photometric_background": _photometric_background_audit(prepared),
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
        "object_patch_plans": (
            [] if hard_frontality is None else _object_patch_plans(result.prepared_pairs, hard_frontality)
        ),
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
    frontality_supports: tuple[VideoDirectSourceSupport, ...] | None = None,
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
        # Calculate F/B DIS once on the maximum permitted rescue corridor.
        # A normal 96--160px GraphCut may then fail topology and consume one
        # 192px retry by slicing this same evidence, never by recomputing flow.
        rescue_width = max(96, min(192, full_right - full_left))
        corridor_width = min(160, rescue_width)
        rescue_left = _low_structure_corridor_left(
            old_bgr, new_bgr, overlap,
            overlap_left=full_left, overlap_right=full_right, width=rescue_width,
            image_width=old_valid.shape[1],
        )
        rescue_right = rescue_left + rescue_width
        left = rescue_left + (rescue_width - corridor_width) // 2
        right = left + corridor_width
        top, bottom = 0, old_valid.shape[0]
        rescue_old_crop = old_bgr[top:bottom, rescue_left:rescue_right]
        rescue_new_crop = new_bgr[top:bottom, rescue_left:rescue_right]
        rescue_old_valid = old_valid[top:bottom, rescue_left:rescue_right]
        rescue_new_valid = new_valid[top:bottom, rescue_left:rescue_right]
        rescue_evidence = video_dis_pair_evidence(
            np.dstack((rescue_old_crop, rescue_old_valid.astype(np.uint8) * 255)),
            np.dstack((rescue_new_crop, rescue_new_valid.astype(np.uint8) * 255)),
            rescue_old_valid & rescue_new_valid,
        )
        if rescue_evidence is None:
            raise RuntimeError("adjacent v6 pair did not produce required F/B DIS evidence")

        def build_corridor(corridor_left: int, corridor_right: int):
            relative_left, relative_right = corridor_left - rescue_left, corridor_right - rescue_left
            old_crop = rescue_old_crop[:, relative_left:relative_right]
            new_crop = rescue_new_crop[:, relative_left:relative_right]
            old_crop_valid = rescue_old_valid[:, relative_left:relative_right]
            new_crop_valid = rescue_new_valid[:, relative_left:relative_right]
            evidence = _crop_dis_evidence(rescue_evidence, relative_left, relative_right)
            base_guards = build_video_hard_guards(
                old_crop, new_crop, evidence, old_valid=old_crop_valid, new_valid=new_crop_valid,
            )
            object_masks = build_video_object_masks(
                evidence, strong_protection=base_guards.protected,
            )
            preferred_new = _compact_object_owner_preference(
                object_masks, old_frame_id=int(old_source.frame_id), new_frame_id=int(new_source.frame_id),
                canvas_left=corridor_left, supports=frontality_supports,
            )
            stored = (near_owner_masks or {}).get((int(old_source.frame_id), int(new_source.frame_id)))
            protected_object = object_masks.protected_mask
            if stored is not None:
                stored_candidate, stored_protected = stored
                preferred_new |= np.asarray(stored_candidate, bool)[:, corridor_left:corridor_right]
                protected_object |= np.asarray(stored_protected, bool)[:, corridor_left:corridor_right]
            guards = build_video_hard_guards(
                old_crop, new_crop, evidence, object_mask=protected_object, prefer_new_mask=preferred_new,
                old_valid=old_crop_valid, new_valid=new_crop_valid,
            )
            graphcut = solve_video_graphcut_seam(
                old_crop, new_crop, old_crop_valid, new_crop_valid,
                hard_owner_old=guards.hard_owner_old, hard_owner_new=guards.hard_owner_new,
            )
            graphcut = replace(graphcut, audit=replace(graphcut.audit, canvas_x_offset=corridor_left))
            return old_crop, new_crop, old_crop_valid, new_crop_valid, evidence, base_guards, object_masks, guards, graphcut

        old_crop, new_crop, old_crop_valid, new_crop_valid, evidence, base_guards, object_masks, guards, graphcut = build_corridor(left, right)
        if not graphcut.audit.accepted and rescue_width > corridor_width:
            left, right = rescue_left, rescue_right
            old_crop, new_crop, old_crop_valid, new_crop_valid, evidence, base_guards, object_masks, guards, graphcut = build_corridor(left, right)
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
