"""Isolated v6.1 real-RGB narrow-strip blocker proof of concept.

This module deliberately does *not* build a panorama, read a pose, read
depth, or publish an artifact.  It compares one existing wide-baseline RGB
pair with the same interval split by an actual capture RGB frame.  The POC is
the Phase-1 evidence gate before the v6.1 anchor/render-strip architecture is
allowed to change the video pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import cv2
import numpy as np

from .video_graphcut_seam import VideoGraphCutAudit, solve_video_graphcut_seam
from .video_hard_guards import build_video_hard_guards
from .video_local_alignment import VideoAlignmentAudit, VideoLocalAlignmentConfig, fit_near_protected_alignment
from .video_near_blend import VideoNearBlendConfig, apply_near_multiband, build_near_blend_eligible_mask
from .video_rgb_quality import assess_video_rgb_quality
from .video_visual_renderer import VideoDISPairEvidence, video_dis_pair_evidence


@dataclass(frozen=True)
class V61BlockerPocConfig:
    """Frozen Phase-1 bounds for one central 480px pair corridor."""

    corridor_width_px: int = 160
    minimum_reliable_pixels: int = 128
    fb_p95_hard_px: float = 1.25
    # Phase 1.3 reconciles this current diagnostic default with the declared
    # Phase 1.2 contract.  The historical Phase 1.2 source is separately
    # content-addressed by ``v61_blocker_poc_v12.lock.json`` and is untouched.
    edge_residual_p95_hard_px: float = 0.75
    edge_residual_abs_hard_px: float = 1.25
    edge_normal_search_px: int = 8
    edge_normal_sample_step_px: float = 0.125
    edge_correspondence_band_px: float = 2.0
    minimum_matched_edge_fraction: float = 0.50
    blend_width_px: int = 2
    allow_graphcut: bool = False

    def __post_init__(self) -> None:
        if not 96 <= self.corridor_width_px <= 160:
            raise ValueError("POC corridor must be within the v6.1 96--160px range")
        if self.minimum_reliable_pixels < 64:
            raise ValueError("POC needs at least 64 reliable DIS pixels")
        if (
            self.fb_p95_hard_px <= 0.0
            or self.edge_residual_p95_hard_px <= 0.0
            or self.edge_residual_abs_hard_px <= 0.0
        ):
            raise ValueError("POC residual limits must be positive")
        if not 1 <= self.edge_normal_search_px <= 16:
            raise ValueError("edge-normal search must be in [1, 16]px")
        if not 0.05 <= self.edge_normal_sample_step_px <= 0.5:
            raise ValueError("edge-normal sample step must be in [0.05, 0.5]px")
        if not 0.5 <= self.edge_correspondence_band_px <= 4.0:
            raise ValueError("edge correspondence band must be in [0.5, 4]px")
        if not 0.0 < self.minimum_matched_edge_fraction <= 1.0:
            raise ValueError("minimum matched-edge fraction must be in (0, 1]")
        if not 2 <= self.blend_width_px <= 4:
            raise ValueError("POC blend must stay within the v6.1 2--4px range")


@dataclass(frozen=True)
class V61BlockerPocSpec:
    """One A--M--B real-capture experiment; ``M`` has no pose field."""

    name: str
    session_root: Path
    left_frame_id: int
    middle_frame_ids: tuple[int, ...]
    right_frame_id: int

    def __post_init__(self) -> None:
        ids = (self.left_frame_id, *self.middle_frame_ids, self.right_frame_id)
        if len(ids) < 3 or tuple(sorted(ids)) != ids or len(set(ids)) != len(ids):
            raise ValueError("POC frame ids must be distinct and chronological A--M--B capture frames")


@dataclass(frozen=True)
class V61PocPairMetrics:
    left_frame_id: int
    right_frame_id: int
    fb_residual_p95_px: float | None
    edge_residual_p95_px: float | None
    edge_residual_abs_max_px: float | None
    line_step_p95_px: float | None
    line_step_abs_max_px: float | None
    double_edge_count: int | None
    ghost_count: int | None
    reliable_pixel_count: int
    alignment_model: str
    alignment_accepted: bool
    pre_seam_pass: bool
    graphcut_called: bool
    graphcut_accepted: bool
    blend_band_pixel_count: int
    runtime_ms: float
    rejection_reason: str | None
    coarse_dx_px: float | None = None
    coarse_dy_px: float | None = None
    coarse_response: float | None = None
    residual_max_displacement_px: float | None = None
    matched_edge_sample_count: int = 0
    edge_sample_count: int = 0
    edge_match_fraction: float | None = None
    not_evaluable_metrics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = tuple(
            name
            for name, value in (
                ("fb_residual_p95_px", self.fb_residual_p95_px),
                ("edge_residual_p95_px", self.edge_residual_p95_px),
                ("edge_residual_abs_max_px", self.edge_residual_abs_max_px),
                ("line_step_p95_px", self.line_step_p95_px),
                ("line_step_abs_max_px", self.line_step_abs_max_px),
                ("double_edge_count", self.double_edge_count),
                ("ghost_count", self.ghost_count),
            )
            if value is None
        )
        object.__setattr__(self, "not_evaluable_metrics", missing)


@dataclass(frozen=True)
class V61BlockerPocResult:
    name: str
    baseline: V61PocPairMetrics
    densified_pairs: tuple[V61PocPairMetrics, ...]
    baseline_double_edge_count: int | None
    baseline_ghost_count: int | None
    densified_double_edge_count: int | None
    densified_ghost_count: int | None
    baseline_runtime_ms: float
    densified_runtime_ms: float
    visual_metrics_non_worse: bool
    visual_metrics_improved: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


EvidenceFactory = Callable[[np.ndarray, np.ndarray, np.ndarray], VideoDISPairEvidence | None]


def _p95(values: np.ndarray) -> float | None:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    return None if not finite.size else float(np.percentile(finite, 95.0))


def _alignment_config() -> VideoLocalAlignmentConfig:
    """Use only the v6.1 identity/translation/rotation/affine ladder bounds."""

    return VideoLocalAlignmentConfig(
        near_translation_target_px=2.0,
        near_translation_hard_px=4.0,
        near_rotation_target_deg=0.75,
        near_rotation_hard_deg=1.5,
        near_affine_scale_min=0.97,
        near_affine_scale_max=1.03,
        near_affine_anisotropic_ratio_max=1.03,
        near_affine_shear_abs_max=0.03,
        near_homography_corner_displacement_hard_px=6.0,
        near_homography_scale_min=0.97,
        near_homography_scale_max=1.03,
        near_homography_line_orientation_change_max_deg=1.5,
        near_homography_held_out_fb_p95_max_px=1.0,
        near_homography_held_out_fb_abs_max_px=2.0,
    )


def _central_corridor(image: np.ndarray, width: int) -> np.ndarray:
    height, image_width = image.shape[:2]
    if height != 480 or image_width < width:
        raise ValueError("POC requires a 480px-tall RGB frame wider than its corridor")
    left = (image_width - width) // 2
    return np.ascontiguousarray(image[:, left : left + width])


def _coarse_strip_placement(left_bgr: np.ndarray, right_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Estimate only the unconstrained full-frame strip translation.

    This is a placement prior, not a residual correction and not a pose.  The
    v6.1 4px limit therefore applies only to the later alignment matrix.
    """

    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    window = cv2.createHanningWindow((left_gray.shape[1], left_gray.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(left_gray, right_gray, window)
    if not np.isfinite((dx, dy, response)).all():
        raise RuntimeError("coarse strip placement returned a non-finite translation")
    matrix = np.array(((1.0, 0.0, dx), (0.0, 1.0, dy), (0.0, 0.0, 1.0)), dtype=np.float64)
    return matrix, float(dx), float(dy), float(response)


def _aligned_new_image(new_bgr: np.ndarray, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample new at the old-coordinate targets; matrix maps old -> new."""

    height, width = new_bgr.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(height * width, np.float32)))
    projected = points @ np.asarray(matrix, np.float64).T
    map_x = (projected[:, 0] / projected[:, 2]).reshape((height, width)).astype(np.float32)
    map_y = (projected[:, 1] / projected[:, 2]).reshape((height, width)).astype(np.float32)
    aligned = cv2.remap(new_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = cv2.remap(
        np.full((height, width), 255, np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(bool)
    return aligned, valid


def _output_coordinate_flows(evidence: VideoDISPairEvidence, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express coarse-coordinate F/B DIS in the residual-corrected output grid."""

    height, width = evidence.flow_forward.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    points = np.column_stack((xx.ravel(), yy.ravel(), np.ones(height * width, np.float32)))
    inverse = np.linalg.inv(np.asarray(matrix, np.float64))
    forward_coarse = np.column_stack((
        (xx + evidence.flow_forward[..., 0]).ravel(),
        (yy + evidence.flow_forward[..., 1]).ravel(),
        np.ones(height * width, np.float32),
    ))
    forward_final = forward_coarse @ inverse.T
    forward_final = forward_final[:, :2] / forward_final[:, 2:3]
    forward = (forward_final - points[:, :2]).reshape((height, width, 2)).astype(np.float32)
    coarse_at_final = points @ np.asarray(matrix, np.float64).T
    coarse_at_final = coarse_at_final[:, :2] / coarse_at_final[:, 2:3]
    coarse_x = coarse_at_final[:, 0].reshape((height, width)).astype(np.float32)
    coarse_y = coarse_at_final[:, 1].reshape((height, width)).astype(np.float32)
    backward_sampled = cv2.remap(
        evidence.flow_backward, coarse_x, coarse_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(np.nan, np.nan),
    ).reshape((-1, 2))
    backward_target = coarse_at_final[:, :2] + backward_sampled
    backward = (backward_target - points[:, :2]).reshape((height, width, 2)).astype(np.float32)
    return forward, backward


def _edge_normal_residual(
    source_bgr: np.ndarray, target_bgr: np.ndarray, support: np.ndarray, expected_flow: np.ndarray, *,
    search_px: int, sample_step_px: float, correspondence_band_px: float,
) -> tuple[np.ndarray, int]:
    """Return sub-pixel, orientation-matched distances along source-edge normals."""

    source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY)
    source_edge = cv2.Canny(source_gray, 80, 160) > 0
    target_edge = cv2.Canny(target_gray, 80, 160) > 0
    source_edge &= np.asarray(support, bool)
    target_edge &= np.asarray(support, bool)
    yy, xx = np.nonzero(source_edge)
    if not xx.size or not target_edge.any():
        return np.empty(0, np.float32), 0
    stride = max(1, int(np.ceil(xx.size / 4096)))
    yy, xx = yy[::stride].astype(np.float32), xx[::stride].astype(np.float32)
    integer_y, integer_x = yy.astype(int), xx.astype(int)
    grad_x = cv2.Sobel(source_gray, cv2.CV_32F, 1, 0)[integer_y, integer_x]
    grad_y = cv2.Sobel(source_gray, cv2.CV_32F, 0, 1)[integer_y, integer_x]
    expected = np.asarray(expected_flow, np.float32)[integer_y, integer_x]
    magnitude = np.hypot(grad_x, grad_y)
    usable = magnitude > 1e-6
    if not usable.any():
        return np.empty(0, np.float32), int(xx.size)
    yy, xx, grad_x, grad_y, magnitude, expected = (
        value[usable] for value in (yy, xx, grad_x, grad_y, magnitude, expected)
    )
    normal_x, normal_y = grad_x / magnitude, grad_y / magnitude
    offsets = np.arange(-search_px, search_px + 0.5 * sample_step_px, sample_step_px, dtype=np.float32)
    target_candidate = cv2.dilate(target_edge.astype(np.uint8), np.ones((3, 3), np.uint8))
    target_grad_x = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0)
    target_grad_y = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1)
    scores = np.full((len(xx), len(offsets)), -np.inf, np.float32)
    for index, offset in enumerate(offsets):
        map_x, map_y = xx + offset * normal_x, yy + offset * normal_y
        candidate = cv2.remap(target_candidate, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).reshape(-1) > 0
        sampled_x = cv2.remap(target_grad_x, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(-1)
        sampled_y = cv2.remap(target_grad_y, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).reshape(-1)
        sampled_magnitude = np.hypot(sampled_x, sampled_y)
        orientation = np.zeros_like(sampled_magnitude)
        valid_orientation = sampled_magnitude > 1e-6
        orientation[valid_orientation] = np.abs(
            sampled_x[valid_orientation] * normal_x[valid_orientation]
            + sampled_y[valid_orientation] * normal_y[valid_orientation]
        ) / sampled_magnitude[valid_orientation]
        correspondence = (map_x - (xx + expected[:, 0])) ** 2 + (map_y - (yy + expected[:, 1])) ** 2
        eligible = (
            candidate
            & (orientation >= np.cos(np.deg2rad(30.0)))
            & (correspondence <= correspondence_band_px**2)
        )
        # Match the *nearest* compatible target edge.  Gradient strength only
        # breaks ties; selecting the strongest edge would jump across a nearby
        # parallel contour and falsely report the search boundary as residual.
        scores[:, index] = np.where(
            eligible,
            -abs(float(offset)) + 1e-6 * sampled_magnitude * orientation,
            -np.inf,
        )
    best_index = np.argmax(scores, axis=1)
    best_score = scores[np.arange(len(xx)), best_index]
    matched = np.isfinite(best_score) & (best_index > 0) & (best_index < len(offsets) - 1)
    if not matched.any():
        return np.empty(0, np.float32), int(len(xx))
    selected = best_index[matched]
    residual = offsets[selected]
    return np.abs(residual).astype(np.float32), int(len(xx))


def _matched_edge_normal_metrics(
    old_bgr: np.ndarray, aligned_new_bgr: np.ndarray, support: np.ndarray, forward_flow: np.ndarray,
    backward_flow: np.ndarray, *, search_px: int, sample_step_px: float, correspondence_band_px: float,
    minimum_fraction: float,
) -> tuple[float | None, float | None, int, int]:
    """Symmetric matched edge-normal P95 and absolute maximum after placement."""

    forward, forward_count = _edge_normal_residual(
        old_bgr, aligned_new_bgr, support, forward_flow, search_px=search_px,
        sample_step_px=sample_step_px, correspondence_band_px=correspondence_band_px,
    )
    backward, backward_count = _edge_normal_residual(
        aligned_new_bgr, old_bgr, support, backward_flow, search_px=search_px,
        sample_step_px=sample_step_px, correspondence_band_px=correspondence_band_px,
    )
    values = np.concatenate((forward, backward))
    count = forward_count + backward_count
    if not values.size or count == 0 or len(values) / count < minimum_fraction:
        return None, None, int(len(values)), count
    return _p95(values), float(np.max(values)), int(len(values)), count


def _null_metrics(
    left_id: int,
    right_id: int,
    *,
    model: str,
    accepted: bool,
    reliable: int,
    fb_p95: float | None,
    coarse_dx: float | None,
    coarse_dy: float | None,
    coarse_response: float | None,
    runtime_ms: float,
    reason: str,
) -> V61PocPairMetrics:
    return V61PocPairMetrics(
        left_frame_id=left_id,
        right_frame_id=right_id,
        fb_residual_p95_px=fb_p95,
        edge_residual_p95_px=None,
        edge_residual_abs_max_px=None,
        line_step_p95_px=None,
        line_step_abs_max_px=None,
        double_edge_count=None,
        ghost_count=None,
        reliable_pixel_count=reliable,
        alignment_model=model,
        alignment_accepted=accepted,
        pre_seam_pass=False,
        graphcut_called=False,
        graphcut_accepted=False,
        blend_band_pixel_count=0,
        runtime_ms=runtime_ms,
        rejection_reason=reason,
        coarse_dx_px=coarse_dx,
        coarse_dy_px=coarse_dy,
        coarse_response=coarse_response,
    )


def run_v61_poc_pair(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    *,
    left_frame_id: int,
    right_frame_id: int,
    config: V61BlockerPocConfig | None = None,
    evidence_factory: EvidenceFactory | None = None,
) -> V61PocPairMetrics:
    """Run one post-placement F/B DIS observation then bounded geometry and GraphCut.

    No GraphCut call is made when the geometry gate fails.  The function is
    intentionally pair-local: it has no pose or strip-placement output.
    """

    settings = config or V61BlockerPocConfig()
    started = perf_counter()
    old_full, new_raw = (np.asarray(image) for image in (left_bgr, right_bgr))
    if old_full.shape != new_raw.shape or old_full.ndim != 3 or old_full.shape[2] != 3:
        raise ValueError("POC pair requires same-shape BGR capture frames")
    coarse_matrix, coarse_dx, coarse_dy, coarse_response = _coarse_strip_placement(old_full, new_raw)
    # First placement always samples from the complete original RGB frame;
    # crop only after it is in the coarse shared output coordinate system.
    coarse_new_full, coarse_valid_full = _aligned_new_image(new_raw, coarse_matrix)
    old = _central_corridor(old_full, settings.corridor_width_px)
    new = _central_corridor(coarse_new_full, settings.corridor_width_px)
    valid = _central_corridor(np.ones(old_full.shape[:2], dtype=np.uint8), settings.corridor_width_px).astype(bool)
    new_valid = _central_corridor(coarse_valid_full.astype(np.uint8), settings.corridor_width_px).astype(bool)
    overlap_before_residual = valid & new_valid
    factory = evidence_factory or (
        lambda left, right, overlap: video_dis_pair_evidence(
            np.dstack((left, overlap.astype(np.uint8) * 255)),
            np.dstack((right, overlap.astype(np.uint8) * 255)),
            overlap,
        )
    )
    # Evidence is owned by the pair stage; callbacks must not be able to
    # mutate the canonical all-real support mask used by later audit stages.
    evidence = factory(old, new, overlap_before_residual.copy())
    def elapsed() -> float:
        return (perf_counter() - started) * 1000.0
    if evidence is None:
        return _null_metrics(
            left_frame_id, right_frame_id, model="hard_owner_only", accepted=False, reliable=0,
            fb_p95=None, coarse_dx=coarse_dx, coarse_dy=coarse_dy, coarse_response=coarse_response,
            runtime_ms=elapsed(), reason="no_fb_dis_evidence",
        )
    reliable = np.asarray(evidence.reliable_mask, bool) & overlap_before_residual
    reliable_count = int(reliable.sum())
    # Guard construction currently uses in-place mask operators internally;
    # retain the full real pair support for the later geometry/GraphCut audit.
    guards = build_video_hard_guards(
        old, new, evidence, old_valid=valid.copy(), new_valid=new_valid.copy(),
    )
    support = reliable & ~np.asarray(guards.protected, bool)
    alignment = fit_near_protected_alignment(evidence, support=support, plane_verified=False, config=_alignment_config())
    audit: VideoAlignmentAudit = alignment.audit
    fb_p95 = _p95(np.asarray(evidence.fb_error)[reliable])
    if not audit.accepted or alignment.matrix is None:
        return _null_metrics(
            left_frame_id, right_frame_id, model=audit.selected_model, accepted=audit.accepted,
            reliable=reliable_count, fb_p95=fb_p95, coarse_dx=coarse_dx, coarse_dy=coarse_dy,
            coarse_response=coarse_response, runtime_ms=elapsed(),
            reason=audit.rejection_reason or "no_bounded_alignment",
        )
    crop_left = (old_full.shape[1] - settings.corridor_width_px) // 2
    crop = np.array(((1.0, 0.0, crop_left), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    combined = coarse_matrix @ crop @ alignment.matrix @ np.linalg.inv(crop)
    final_new_full, final_valid_full = _aligned_new_image(new_raw, combined)
    aligned_new = _central_corridor(final_new_full, settings.corridor_width_px)
    aligned_valid = _central_corridor(final_valid_full.astype(np.uint8), settings.corridor_width_px).astype(bool)
    overlap = valid & aligned_valid
    output_forward, output_backward = _output_coordinate_flows(evidence, alignment.matrix)
    edge_p95, edge_abs, matched_edge_count, edge_sample_count = _matched_edge_normal_metrics(
        old, aligned_new, overlap, output_forward, output_backward, search_px=settings.edge_normal_search_px,
        sample_step_px=settings.edge_normal_sample_step_px,
        correspondence_band_px=settings.edge_correspondence_band_px,
        minimum_fraction=settings.minimum_matched_edge_fraction,
    )
    pre_pass = bool(
        reliable_count >= settings.minimum_reliable_pixels
        and fb_p95 is not None and fb_p95 <= settings.fb_p95_hard_px
        and edge_p95 is not None and edge_p95 <= settings.edge_residual_p95_hard_px
        and edge_abs is not None and edge_abs <= settings.edge_residual_abs_hard_px
    )
    if not pre_pass:
        return V61PocPairMetrics(
            left_frame_id, right_frame_id, fb_p95, edge_p95, edge_abs, None, None, None, None,
            reliable_count, audit.selected_model, True, False, False, False, 0, elapsed(), "pre_seam_geometry_gate_failed",
            coarse_dx, coarse_dy, coarse_response, audit.maximum_displacement_px,
            matched_edge_count, edge_sample_count,
            (None if edge_sample_count == 0 else matched_edge_count / edge_sample_count),
        )
    if not settings.allow_graphcut:
        return V61PocPairMetrics(
            left_frame_id, right_frame_id, fb_p95, edge_p95, edge_abs, None, None, None, None,
            reliable_count, audit.selected_model, True, True, False, False, 0, elapsed(), "phase_1_2_graphcut_disabled",
            coarse_dx, coarse_dy, coarse_response, audit.maximum_displacement_px,
            matched_edge_count, edge_sample_count,
            (None if edge_sample_count == 0 else matched_edge_count / edge_sample_count),
        )
    aligned_guards = build_video_hard_guards(
        old, aligned_new, evidence, old_valid=valid.copy(), new_valid=aligned_valid.copy(),
    )
    graphcut = solve_video_graphcut_seam(
        old, aligned_new, valid, aligned_valid,
        hard_owner_old=aligned_guards.hard_owner_old,
        hard_owner_new=aligned_guards.hard_owner_new,
    )
    graph_audit: VideoGraphCutAudit = graphcut.audit
    if not graph_audit.accepted:
        return V61PocPairMetrics(
            left_frame_id, right_frame_id, fb_p95, edge_p95, edge_abs, None, graph_audit.maximum_adjacent_row_step_px,
            None, None, reliable_count, audit.selected_model, True, True, True, False, 0, elapsed(), graph_audit.rejection_reason,
            coarse_dx, coarse_dy, coarse_response, audit.maximum_displacement_px,
            matched_edge_count, edge_sample_count,
            (None if edge_sample_count == 0 else matched_edge_count / edge_sample_count),
        )
    owner = np.full(valid.shape, int(left_frame_id), np.int32)
    owner[graphcut.choose_new] = int(right_frame_id)
    output = old.copy()
    output[graphcut.choose_new] = aligned_new[graphcut.choose_new]
    eligible = build_near_blend_eligible_mask(valid, aligned_valid, evidence, aligned_guards)
    output, _band, blend = apply_near_multiband(
        old, aligned_new, output, graphcut.choose_new, eligible, aligned_guards,
        config=VideoNearBlendConfig(near_width_px=settings.blend_width_px),
    )
    quality = assess_video_rgb_quality(output, owner, valid, (graph_audit,))
    return V61PocPairMetrics(
        left_frame_id, right_frame_id, fb_p95, edge_p95, edge_abs, quality.seam_step_p95_px,
        quality.seam_step_abs_max_px, quality.double_edge_count, quality.ghost_count,
        reliable_count, audit.selected_model, True, True, True, True, blend.band_pixel_count,
        elapsed(), None, coarse_dx, coarse_dy, coarse_response, audit.maximum_displacement_px,
        matched_edge_count, edge_sample_count,
        (None if edge_sample_count == 0 else matched_edge_count / edge_sample_count),
    )


def _capture_rgb_paths(session_root: Path) -> dict[int, Path]:
    root = Path(session_root).resolve()
    frames_csv = root / "frames.csv"
    if not frames_csv.is_file():
        raise FileNotFoundError(f"POC session has no frames.csv: {frames_csv}")
    paths: dict[int, Path] = {}
    with frames_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = int(row["frame_id"])
            candidate = (root / row["color_path"]).resolve()
            if root not in candidate.parents or not candidate.is_file():
                raise FileNotFoundError(f"POC RGB frame is missing or outside its session: {frame_id}")
            paths[frame_id] = candidate
    return paths


def _read_rgb(paths: dict[int, Path], frame_id: int) -> np.ndarray:
    path = paths.get(frame_id)
    if path is None:
        raise ValueError(f"POC frame {frame_id} is not a real capture frame")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode POC RGB frame {path}")
    return image


def _sum_metric(values: Sequence[V61PocPairMetrics], attribute: str) -> int | None:
    result = [getattr(value, attribute) for value in values]
    return None if any(value is None for value in result) else int(sum(int(value) for value in result))


def run_v61_blocker_poc(
    spec: V61BlockerPocSpec, *, config: V61BlockerPocConfig | None = None,
    evidence_factory: EvidenceFactory | None = None,
) -> V61BlockerPocResult:
    """Compare the old A--B seam with A--M--B real-RGB densification."""

    paths = _capture_rgb_paths(spec.session_root)
    frames = {frame_id: _read_rgb(paths, frame_id) for frame_id in (spec.left_frame_id, *spec.middle_frame_ids, spec.right_frame_id)}
    baseline = run_v61_poc_pair(
        frames[spec.left_frame_id], frames[spec.right_frame_id], left_frame_id=spec.left_frame_id,
        right_frame_id=spec.right_frame_id, config=config, evidence_factory=evidence_factory,
    )
    ids = (spec.left_frame_id, *spec.middle_frame_ids, spec.right_frame_id)
    densified = tuple(
        run_v61_poc_pair(frames[left], frames[right], left_frame_id=left, right_frame_id=right, config=config, evidence_factory=evidence_factory)
        for left, right in zip(ids[:-1], ids[1:], strict=True)
    )
    baseline_double, baseline_ghost = baseline.double_edge_count, baseline.ghost_count
    dense_double = _sum_metric(densified, "double_edge_count")
    dense_ghost = _sum_metric(densified, "ghost_count")
    comparable = None not in (baseline_double, baseline_ghost, dense_double, dense_ghost)
    non_worse = bool(comparable and dense_double <= baseline_double and dense_ghost <= baseline_ghost)
    improved = bool(non_worse and (dense_double < baseline_double or dense_ghost < baseline_ghost))
    return V61BlockerPocResult(
        spec.name, baseline, densified, baseline_double, baseline_ghost, dense_double, dense_ghost,
        baseline.runtime_ms, float(sum(pair.runtime_ms for pair in densified)), non_worse, improved,
    )


def default_v61_blocker_specs(root: Path) -> tuple[V61BlockerPocSpec, ...]:
    """The three primary Phase-1 blocker seams frozen in the v6 audit."""

    return (
        V61BlockerPocSpec("140140_63_to_66_dense", root / "run_20260807_140140", 63, (64, 65), 66),
        V61BlockerPocSpec("162340_196_to_202_dense", root / "run_20260804_162340", 196, (197, 199, 200), 202),
        V61BlockerPocSpec("153033_87_to_95_dense", root / "run_20260806_153033", 87, (88, 89, 90, 91, 92, 93, 94), 95),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run isolated v6.1 A--M--B real-RGB blocker POCs")
    parser.add_argument("--captures-root", type=Path, required=True, help="Directory containing frozen run_* sessions")
    parser.add_argument("--output", type=Path, help="Optional JSON evidence path; never a production artifact")
    args = parser.parse_args(argv)
    results = [result.as_dict() for result in (run_v61_blocker_poc(spec) for spec in default_v61_blocker_specs(args.captures_root))]
    payload = {
        "schema": "video-v61-blocker-poc/v1.2",
        "phase": "1.2_edge_measurement_calibration",
        "isolated_candidate_only": True,
        "results": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
