"""CPU reference executor for one immutable S013 M6.2 plan."""

from __future__ import annotations

import time
from typing import Mapping

import numpy as np

from .video_s13_blend import apply_s13_blend_plan
from .video_s13_m6 import S13P3Result, _p3_provenance
from .video_s13_photometric import (
    apply_s13_photometric_linear, linear_to_srgb_bgr, srgb_to_linear_bgr,
)
from .video_s13_m62_plan import S13M62ExecutionPlan
from .cuda_backend import remap as accelerated_remap
import cv2


def execute_s13_m62_cpu_reference(
    plan: S13M62ExecutionPlan, *, retain_runtime_details: bool = False,
) -> S13P3Result:
    """Execute exactly the decisions frozen in ``plan`` on CPU arrays."""

    started = time.perf_counter()
    corrected = plan.corrected_rois
    if (
        plan.photometric_solution.model_family == "Q0_identity"
        and all(item.transaction.model == "B0_owner_only" for item in plan.blend_plans)
    ):
        image = np.asarray(plan.p2_image).copy()
        return S13P3Result(
            photometric_owner_only=image.copy(), visual_panorama=image,
            valid_mask=plan.valid_mask.copy(), pixel_provenance={},
            photometric_solution=plan.photometric_solution, photometric_samples=plan.photometric_samples,
            blend_plans=plan.blend_plans, protected_structure_mask=np.zeros(plan.valid_mask.shape, bool),
            safe_blend_mask=np.zeros(plan.valid_mask.shape, bool), blend_weight_map=np.zeros(plan.valid_mask.shape, np.float32),
            photometric_training_mask=np.asarray(plan.sample_masks["train"], bool),
            photometric_heldout_mask=np.asarray(plan.sample_masks["heldout"], bool), diagnostic_quality={},
            performance={"total_m6": time.perf_counter() - started, "published_pixel_authority": "cpu", "q0_b0_direct_p2": True},
        )
    owner_linear = _compose_owner_only_roi(plan, corrected)

    def pair_provider(pair: object) -> tuple[np.ndarray, np.ndarray]:
        return _pair_tiles(plan, corrected, pair)

    plans = plan.blend_plans
    blend_masks = _blend_masks(plan)
    final_linear = owner_linear.copy()
    for pair, blend_plan in zip(plan.p2.replay_pairs, plans, strict=True):
        active = np.asarray(blend_plan.secondary_weight) > 0.0
        if np.any(active):
            roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
            final_linear[roi][active] = apply_s13_blend_plan(*pair_provider(pair), pair, blend_plan)[active]
    owner_u8, final_u8 = linear_to_srgb_bgr(owner_linear), linear_to_srgb_bgr(final_linear)
    owner_u8[~plan.valid_mask] = 0
    final_u8[~plan.valid_mask] = 0
    return S13P3Result(
        photometric_owner_only=owner_u8, visual_panorama=final_u8, valid_mask=plan.valid_mask.copy(),
        pixel_provenance=_p3_provenance(plan.p2, plans) if retain_runtime_details else {},
        photometric_solution=plan.photometric_solution, photometric_samples=plan.photometric_samples,
        blend_plans=plans, protected_structure_mask=np.asarray(blend_masks["protected"], bool),
        safe_blend_mask=np.asarray(blend_masks["safe"], bool),
        blend_weight_map=np.asarray(blend_masks["secondary_weight"], np.float32),
        photometric_training_mask=np.asarray(plan.sample_masks["train"], bool),
        photometric_heldout_mask=np.asarray(plan.sample_masks["heldout"], bool), diagnostic_quality={},
        performance={"total_m6": time.perf_counter() - started, "published_pixel_authority": "cpu"},
    )


def build_s13_m62_cpu_corrected_rois(plan: S13M62ExecutionPlan) -> dict[int, np.ndarray]:
    """Pixel-only corrected ROI work shared with the GPU parity harness."""
    corrected: dict[int, np.ndarray] = {}
    for parameter, roi in zip(plan.photometric_solution.source_parameters, plan.source_rois, strict=True):
        sampled = accelerated_remap(plan.raw_by_frame[parameter.frame_id], roi.map_u, roi.map_v,
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        linear = srgb_to_linear_bgr(sampled)
        adjusted = apply_s13_photometric_linear(linear, parameter)
        adjusted[~roi.mapped] = 0.0
        corrected[parameter.source_index] = adjusted
    return corrected


def _blend_masks(plan: S13M62ExecutionPlan) -> dict[str, np.ndarray]:
    safe = np.zeros(plan.valid_mask.shape, bool)
    protected = np.zeros(plan.valid_mask.shape, bool)
    weight = np.zeros(plan.valid_mask.shape, np.float32)
    for pair, blend_plan in zip(plan.p2.replay_pairs, plan.blend_plans, strict=True):
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        safe[roi] |= blend_plan.safe_mask
        protected[roi] |= blend_plan.protected_mask
        weight[roi] = blend_plan.secondary_weight
    return {"safe": safe, "protected": protected, "secondary_weight": weight}


def _compose_owner_only_roi(plan: S13M62ExecutionPlan, corrected: Mapping[int, np.ndarray]) -> np.ndarray:
    output = np.zeros((*plan.valid_mask.shape, 3), np.float32)
    for roi in plan.source_rois:
        output[:, roi.x0:roi.x1][roi.owner_mask] = corrected[roi.source_index][roi.owner_mask]
    return output


def _pair_tiles(plan: S13M62ExecutionPlan, corrected: Mapping[int, np.ndarray], pair: object) -> tuple[np.ndarray, np.ndarray]:
    by_source = {roi.source_index: roi for roi in plan.source_rois}
    def tile(source: int) -> np.ndarray:
        roi = by_source[source]
        return corrected[source][:, pair.corridor_x0 - roi.x0:pair.corridor_x1 - roi.x0]
    return tile(pair.left_source_index), tile(pair.right_source_index)


__all__ = ["execute_s13_m62_cpu_reference", "build_s13_m62_cpu_corrected_rois", "_compose_owner_only_roi", "_pair_tiles"]
