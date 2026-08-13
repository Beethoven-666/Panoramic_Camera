"""M6 orchestration: replay sealed P2 and produce hard-safe visual P3."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Mapping

import cv2
import numpy as np

from .cuda_backend import remap as accelerated_remap
from .video_s13_blend import (
    S13BlendConfig,
    S13BlendPlan,
    apply_s13_blend_plan,
    select_s13_blend_plans,
)
from .video_s13_photometric import (
    S13PhotometricConfig,
    S13PhotometricSampleSet,
    S13PhotometricSolution,
    apply_s13_photometric_linear,
    extract_s13_photometric_samples,
    linear_to_srgb_bgr,
    solve_s13_photometric,
    srgb_to_linear_bgr,
)
from .video_s13_replay import S13VerifiedP2
from .video_s13_visual_quality import build_s13_p3_diagnostic_quality


@dataclass(frozen=True)
class S13P3Result:
    photometric_owner_only: np.ndarray
    visual_panorama: np.ndarray
    valid_mask: np.ndarray
    pixel_provenance: Mapping[str, np.ndarray]
    photometric_solution: S13PhotometricSolution
    photometric_samples: tuple[S13PhotometricSampleSet, ...]
    blend_plans: tuple[S13BlendPlan, ...]
    protected_structure_mask: np.ndarray
    safe_blend_mask: np.ndarray
    blend_weight_map: np.ndarray
    photometric_training_mask: np.ndarray
    photometric_heldout_mask: np.ndarray
    diagnostic_quality: Mapping[str, object]
    performance: Mapping[str, object]


def _source_frame_ids(p2: S13VerifiedP2) -> tuple[int, ...]:
    source_count = int(p2.completion["source_count"])
    frame_ids = [-1] * source_count
    for pair in p2.replay_pairs:
        frame_ids[pair.left_source_index] = pair.left_frame_id
        frame_ids[pair.right_source_index] = pair.right_frame_id
    if any(value < 0 for value in frame_ids) or len(set(frame_ids)) != source_count:
        raise ValueError("S1.3 P2 replay source/frame mapping is incomplete")
    return tuple(frame_ids)


def _formal_source_maps(
    p2: S13VerifiedP2, source_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = p2.valid_mask.shape
    map_u = np.full(shape, -1.0, dtype=np.float32)
    map_v = np.full(shape, -1.0, dtype=np.float32)
    mapped = np.zeros(shape, dtype=bool)
    owner = np.asarray(p2.provenance["owner_source_index"], dtype=np.int32)
    primary = p2.valid_mask & (owner == source_index)
    map_u[primary] = np.asarray(p2.provenance["source_u"], np.float32)[primary]
    map_v[primary] = np.asarray(p2.provenance["source_v"], np.float32)[primary]
    mapped |= primary
    for pair in p2.replay_pairs:
        if source_index not in (pair.left_source_index, pair.right_source_index):
            continue
        if source_index == pair.left_source_index:
            u, v, valid = pair.left_source_u, pair.left_source_v, pair.left_valid
        else:
            u, v, valid = pair.right_source_u, pair.right_source_v, pair.right_valid
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        existing = mapped[roi] & valid
        if np.any(existing):
            if (
                np.max(np.abs(map_u[roi][existing] - u[existing]), initial=0.0) > 1e-3
                or np.max(np.abs(map_v[roi][existing] - v[existing]), initial=0.0) > 1e-3
            ):
                raise ValueError("S1.3 P2 replay gives conflicting source sampling")
        map_u[roi][valid] = u[valid]
        map_v[roi][valid] = v[valid]
        mapped[roi] |= valid
    return map_u, map_v, mapped


def _formal_remap_sources(
    p2: S13VerifiedP2,
    raw_by_frame: Mapping[int, np.ndarray],
    solution: S13PhotometricSolution,
) -> tuple[dict[int, np.ndarray], int, int, int]:
    corrected: dict[int, np.ndarray] = {}
    peak_bytes = 0
    for parameter in solution.source_parameters:
        raw = np.asarray(raw_by_frame[parameter.frame_id])
        map_u, map_v, mapped = _formal_source_maps(p2, parameter.source_index)
        sampled = accelerated_remap(
            raw, map_u, map_v, cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        linear = srgb_to_linear_bgr(sampled)
        adjusted = apply_s13_photometric_linear(linear, parameter)
        adjusted[~mapped] = 0.0
        corrected[parameter.source_index] = adjusted
        peak_bytes = max(peak_bytes, raw.nbytes + map_u.nbytes + map_v.nbytes + sampled.nbytes + adjusted.nbytes)
    return corrected, len(solution.source_parameters), len(solution.source_parameters), peak_bytes


def _compose_owner_only(
    p2: S13VerifiedP2, corrected: Mapping[int, np.ndarray]
) -> np.ndarray:
    owner = np.asarray(p2.provenance["owner_source_index"], dtype=np.int32)
    output = np.zeros((*p2.valid_mask.shape, 3), dtype=np.float32)
    for source_index, source in corrected.items():
        mask = p2.valid_mask & (owner == source_index)
        output[mask] = source[mask]
    return output


def _p3_provenance(
    p2: S13VerifiedP2,
    plans: tuple[S13BlendPlan, ...],
) -> dict[str, np.ndarray]:
    provenance = {name: np.asarray(value).copy() for name, value in p2.provenance.items()}
    shape = p2.valid_mask.shape
    primary_source = np.asarray(provenance["owner_source_index"], dtype=np.int32)
    provenance["photometric_transaction_id"] = np.where(
        p2.valid_mask, primary_source, -1
    ).astype(np.int32)
    provenance["blend_transaction_id"] = np.full(shape, -1, np.int32)
    provenance["secondary_frame_id"] = np.full(shape, -1, np.int32)
    provenance["secondary_source_index"] = np.full(shape, -1, np.int32)
    provenance["secondary_source_u"] = np.full(shape, np.nan, np.float32)
    provenance["secondary_source_v"] = np.full(shape, np.nan, np.float32)
    provenance["secondary_weight"] = np.zeros(shape, np.float32)
    for pair, plan in zip(p2.replay_pairs, plans, strict=True):
        weight = np.asarray(plan.secondary_weight, np.float32)
        active = weight > 0.0
        if not np.any(active):
            continue
        primary_right = pair.primary_owner_right_mask
        secondary_frame = np.where(primary_right, pair.left_frame_id, pair.right_frame_id)
        secondary_source = np.where(
            primary_right, pair.left_source_index, pair.right_source_index
        )
        secondary_u = np.where(primary_right, pair.left_source_u, pair.right_source_u)
        secondary_v = np.where(primary_right, pair.left_source_v, pair.right_source_v)
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        provenance["blend_transaction_id"][roi][active] = plan.transaction.transaction_id
        provenance["secondary_frame_id"][roi][active] = secondary_frame[active]
        provenance["secondary_source_index"][roi][active] = secondary_source[active]
        provenance["secondary_source_u"][roi][active] = secondary_u[active]
        provenance["secondary_source_v"][roi][active] = secondary_v[active]
        provenance["secondary_weight"][roi][active] = weight[active]
    return provenance


def run_s13_m6(
    p2: S13VerifiedP2,
    image_loader: Callable[[int], np.ndarray],
    *,
    photometric_config: S13PhotometricConfig = S13PhotometricConfig(),
    blend_config: S13BlendConfig = S13BlendConfig(),
    force_identity_owner_only: bool = False,
) -> S13P3Result:
    """Replay P2 once from raw RGB; M4/M5/trajectory/depth are never called."""

    started = time.perf_counter()
    tick = time.perf_counter()
    samples, masks, raw_cache = extract_s13_photometric_samples(
        p2.replay_pairs, image_loader, canvas_shape=p2.valid_mask.shape,
        config=photometric_config,
    )
    sample_seconds = time.perf_counter() - tick
    frame_ids = _source_frame_ids(p2)
    tick = time.perf_counter()
    solution = solve_s13_photometric(
        samples, frame_ids=frame_ids, config=photometric_config,
        force_identity=force_identity_owner_only,
    )
    solve_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    corrected, decode_count, remap_count, peak_bytes = _formal_remap_sources(
        p2, raw_cache, solution
    )
    remap_seconds = time.perf_counter() - tick
    owner_linear = _compose_owner_only(p2, corrected)
    tick = time.perf_counter()
    plans, blend_masks = select_s13_blend_plans(
        p2.replay_pairs, samples, corrected, canvas_shape=p2.valid_mask.shape,
        config=blend_config, force_owner_only=force_identity_owner_only,
    )
    blend_analysis_seconds = time.perf_counter() - tick
    final_linear = owner_linear.copy()
    for pair, plan in zip(p2.replay_pairs, plans, strict=True):
        if not np.any(plan.secondary_weight > 0.0):
            continue
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        pair_result = apply_s13_blend_plan(
            corrected[pair.left_source_index][roi],
            corrected[pair.right_source_index][roi],
            pair,
            plan,
        )
        active = plan.secondary_weight > 0.0
        final_linear[roi][active] = pair_result[active]
    provenance = _p3_provenance(p2, plans)
    owner_u8 = linear_to_srgb_bgr(owner_linear)
    final_u8 = linear_to_srgb_bgr(final_linear)
    owner_u8[~p2.valid_mask] = 0
    final_u8[~p2.valid_mask] = 0
    diagnostic = build_s13_p3_diagnostic_quality(
        p2.result_image, owner_u8, final_u8, p2.replay_pairs, plans, solution
    )
    performance: dict[str, object] = {
        "p2_verify_and_load": 0.0,
        "replay_verify": 0.0,
        "photometric_sample_extraction": sample_seconds,
        "photometric_solve": solve_seconds,
        "luminance_field_solve": 0.0,
        "blend_candidate_analysis": blend_analysis_seconds,
        "full_resolution_decode": 0.0,
        "full_resolution_remap": remap_seconds,
        "photometric_owner_compose": 0.0,
        "blend_compose": 0.0,
        "p3_hard_audit": 0.0,
        "artifact_export": 0.0,
        "total_m6": time.perf_counter() - started,
        "m4_reestimated_in_m6": 0,
        "m5_reestimated_in_m6": 0,
        "geometry_reestimated_in_m6": 0,
        "seam_reestimated_in_m6": 0,
        "geometry_candidate_count_in_m6": 0,
        "seam_candidate_count_in_m6": 0,
        "trajectory_estimation_invocations": 0,
        "open3d_invocations": 0,
        "depth_invocations": 0,
        "tsdf_invocations": 0,
        "p3_full_resolution_candidate_render_count": 1,
        "formal_raw_rgb_unique_sources": decode_count,
        "formal_raw_rgb_remap_invocations": remap_count,
        "full_resolution_render_count": 1,
        "peak_memory_bytes_estimate": peak_bytes + owner_linear.nbytes + final_linear.nbytes,
        "fallback_rebuild_identity_owner_only": force_identity_owner_only,
    }
    return S13P3Result(
        photometric_owner_only=owner_u8,
        visual_panorama=final_u8,
        valid_mask=p2.valid_mask.copy(),
        pixel_provenance=provenance,
        photometric_solution=solution,
        photometric_samples=samples,
        blend_plans=plans,
        protected_structure_mask=np.asarray(blend_masks["protected"], bool),
        safe_blend_mask=np.asarray(blend_masks["safe"], bool),
        blend_weight_map=np.asarray(blend_masks["secondary_weight"], np.float32),
        photometric_training_mask=np.asarray(masks["train"], bool),
        photometric_heldout_mask=np.asarray(masks["heldout"], bool),
        diagnostic_quality=diagnostic,
        performance=performance,
    )


__all__ = ["S13P3Result", "run_s13_m6"]
