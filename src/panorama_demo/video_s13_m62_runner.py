"""Safe M6.2 staged-equivalence entry point used by the fast candidate path."""

from __future__ import annotations

from typing import Callable, Any

import numpy as np

from .video_s13_m62_equivalence import compare_s13_m62_u8
from .video_s13_m62_gpu import execute_s13_m62_gpu_shadow
from .video_s13_m62_plan import build_s13_m62_execution_plan
from .video_s13_m62_reference import execute_s13_m62_cpu_reference
from .video_s13_blend import S13BlendConfig
from .video_s13_photometric import S13PhotometricConfig
from .video_s13_replay import S13VerifiedP2
from .video_s13_m6 import run_s13_m6


def run_s13_m62_cpu_authoritative(
    p2: S13VerifiedP2, image_loader: Callable[[int], np.ndarray], *, cuda_runtime: object | None = None,
    force_owner_only_pair_indices: frozenset[int] = frozenset(), retain_runtime_details: bool = False,
    photometric_config: S13PhotometricConfig = S13PhotometricConfig(),
    blend_config: S13BlendConfig = S13BlendConfig(),
) -> tuple[object, dict[str, Any]]:
    """Publish CPU P3 and, when available, run a non-authoritative GPU shadow."""

    plan = build_s13_m62_execution_plan(
        p2, image_loader, photometric_config=photometric_config, blend_config=blend_config,
        force_owner_only_pair_indices=force_owner_only_pair_indices,
    )
    reference = execute_s13_m62_cpu_reference(plan, retain_runtime_details=retain_runtime_details)
    if any(item.transaction.model == "B2_safe_masked_multiband" for item in plan.blend_plans):
        equivalence: dict[str, Any] = {
            "requested_mode": "shadow_gpu_b1", "resolved_mode": "cpu_authoritative_hybrid",
            "fallback_reason": "b2_cpu_reference_fallback", "gpu_evaluated": False,
            "authority_gate_passed": False, "byte_identical": False,
        }
    else:
        try:
            shadow = execute_s13_m62_gpu_shadow(plan, cpu_reference=reference, cuda_runtime=cuda_runtime)
            equivalence = shadow.equivalence
        except Exception as exc:  # Pixel authority remains the CPU reference.
            equivalence = {
                "requested_mode": "shadow_gpu_b1", "resolved_mode": "cpu_authoritative_hybrid",
                "fallback_reason": f"gpu_shadow_failed:{type(exc).__name__}",
                "gpu_evaluated": False, "authority_gate_passed": False, "byte_identical": False,
            }
    legacy = run_s13_m6(
        p2, image_loader, photometric_config=photometric_config, blend_config=blend_config,
        force_owner_only_pair_indices=force_owner_only_pair_indices,
        retain_runtime_details=False,
    )
    legacy_comparison = compare_s13_m62_u8(legacy.visual_panorama, reference.visual_panorama)
    if not legacy_comparison["authority_gate_passed"]:
        raise ValueError("S1.3 M6.2 CPU executor diverged from legacy CPU M6 reference")
    published = compare_s13_m62_u8(legacy.visual_panorama, reference.visual_panorama)
    equivalence["legacy_cpu_reference_comparison"] = legacy_comparison
    equivalence["published_p3_cpu_comparison"] = published
    equivalence["published_pixel_authority"] = "cpu"
    performance = dict(reference.performance)
    performance.update({"m62_cpu_oracle": True, "m62_gpu_shadow_attempted": cuda_runtime is not None})
    object.__setattr__(reference, "performance", performance)
    return reference, {"plan": plan, "gpu_equivalence": equivalence}


__all__ = ["run_s13_m62_cpu_authoritative"]
