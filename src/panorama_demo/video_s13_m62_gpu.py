"""GPU B1 shadow executor.  It consumes a frozen M6.2 plan only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .video_s13_blend import apply_s13_blend_plan
from .video_s13_m62_equivalence import compare_s13_m62_arrays, compare_s13_m62_u8, build_s13_m62_equivalence_report
from .video_s13_m62_plan import S13M62ExecutionPlan
from .video_s13_m62_reference import _compose_owner_only_roi, _pair_tiles
from .video_s13_photometric import linear_to_srgb_bgr


@dataclass(frozen=True)
class S13M62GpuShadowResult:
    equivalence: dict[str, Any]
    final_linear: np.ndarray | None
    final_u8: np.ndarray | None


def execute_s13_m62_gpu_shadow(
    plan: S13M62ExecutionPlan, *, cpu_reference: object, cuda_runtime: object | None,
) -> S13M62GpuShadowResult:
    """Apply only frozen B1 arrays on GPU, while CPU remains publication authority."""

    if cuda_runtime is None or not hasattr(cuda_runtime, "apply_b1_secondary_weight_roi_device"):
        return S13M62GpuShadowResult(
            build_s13_m62_equivalence_report(corrected_roi=[], owner_linear={"passed": False, "max_abs": None},
                b1_pairs=[], final_linear={"passed": False, "max_abs": None},
                final_u8={"authority_gate_passed": False}, fallback_reason="cuda_runtime_unavailable"), None, None)
    # The reference is publication authority; CUDA performs the same frozen
    # corrected ROI operations and is compared before it can reach B1.
    corrected: dict[int, np.ndarray] = {}
    corrected_reports: list[dict[str, Any]] = []
    for parameter, roi in zip(plan.photometric_solution.source_parameters, plan.source_rois, strict=True):
        gpu_roi = cuda_runtime.remap_resident_frame_corrected_linear(
            parameter.frame_id, plan.raw_by_frame[parameter.frame_id], roi.map_u, roi.map_v,
            parameter.gain_bgr, parameter.bias_bgr, roi.mapped,
        )
        corrected[roi.source_index] = np.asarray(gpu_roi, np.float32)
        corrected_reports.append({"source_index": roi.source_index, "comparison": compare_s13_m62_arrays(
            plan.corrected_rois[roi.source_index], corrected[roi.source_index], label="corrected_roi", maximum=1e-6)})
    owner = _compose_owner_only_roi(plan, corrected)
    expected_owner = _compose_owner_only_roi(plan, plan.corrected_rois)
    canvas_device = cuda_runtime.device_copy(owner)
    expected_canvas = expected_owner.copy()
    owner_report = compare_s13_m62_arrays(expected_owner, owner, label="owner_linear", maximum=1e-6)
    b1_reports: list[dict[str, Any]] = []
    for pair, blend_plan in zip(plan.p2.replay_pairs, plan.blend_plans, strict=True):
        active = np.asarray(blend_plan.secondary_weight, np.float32) > 0.0
        if not np.any(active):
            continue
        left, right = _pair_tiles(plan, corrected, pair)
        expected_left, expected_right = _pair_tiles(plan, plan.corrected_rois, pair)
        gpu_pair = cuda_runtime.apply_b1_secondary_weight_roi_device(
            canvas_device=canvas_device, x0=pair.corridor_x0, x1=pair.corridor_x1,
            left_device=left, right_device=right, primary_owner_right_mask=pair.primary_owner_right_mask,
            secondary_weight=blend_plan.secondary_weight,
            protected_mask=np.asarray(blend_plan.protected_mask, bool),
        )
        expected = apply_s13_blend_plan(expected_left, expected_right, pair, blend_plan)
        expected_canvas[:, pair.corridor_x0:pair.corridor_x1][active] = expected[active]
        b1_reports.append({"pair_index": pair.pair_index, "comparison": compare_s13_m62_arrays(
            expected[active], np.asarray(gpu_pair)[active], label="b1_pair_linear", maximum=1e-6)})
    canvas = cuda_runtime.download_linear_canvas(canvas_device)
    final_linear_report = compare_s13_m62_arrays(expected_canvas, canvas,
        label="final_linear", maximum=1e-6)
    final_u8 = (
        cuda_runtime.encode_linear_srgb_shadow(canvas)
        if hasattr(cuda_runtime, "encode_linear_srgb_shadow") else linear_to_srgb_bgr(canvas)
    )
    u8_report = compare_s13_m62_u8(cpu_reference.visual_panorama, final_u8)
    report = build_s13_m62_equivalence_report(
        corrected_roi=corrected_reports, owner_linear=owner_report, b1_pairs=b1_reports,
        final_linear=final_linear_report, final_u8=u8_report)
    report["gpu_evaluated"] = True
    return S13M62GpuShadowResult(report, canvas, final_u8)
__all__ = ["S13M62GpuShadowResult", "execute_s13_m62_gpu_shadow"]
