"""Execution modes for the effective M6.2 decision and pixel pipeline."""

from __future__ import annotations

import time
from typing import Any, Callable, Literal

import numpy as np

from .video_s13_blend import S13BlendConfig
from .video_s13_m6 import run_s13_m6
from .video_s13_m62_equivalence import compare_s13_m62_u8
from .video_s13_m62_evidence import S13M62EvidenceSamplingMode
from .video_s13_m62_gpu import execute_s13_m62_gpu_shadow
from .video_s13_m62_plan import build_s13_m62_execution_plan
from .video_s13_m62_reference import execute_s13_m62_cpu_reference
from .video_s13_photometric import S13PhotometricConfig
from .video_s13_m63_solver import S13M63Config
from .video_s13_replay import S13VerifiedP2


S13M62ExecutionMode = Literal[
    "reference", "shadow_audit", "candidate_single_pass", "parity_test"
]
_MODES = frozenset({"reference", "shadow_audit", "candidate_single_pass", "parity_test"})


def run_s13_m62(
    p2: S13VerifiedP2,
    image_loader: Callable[[int], np.ndarray],
    *,
    execution_mode: S13M62ExecutionMode = "candidate_single_pass",
    cuda_runtime: object | None = None,
    force_owner_only_pair_indices: frozenset[int] = frozenset(),
    retain_runtime_details: bool = False,
    photometric_config: S13PhotometricConfig = S13PhotometricConfig(),
    blend_config: S13BlendConfig = S13BlendConfig(),
    m63_config: S13M63Config = S13M63Config(),
    evidence_sampling_mode: S13M62EvidenceSamplingMode = "pair_reference",
    packed_reference_fallback: bool = True,
) -> tuple[object, dict[str, Any]]:
    """Run exactly the work authorized by ``execution_mode``."""

    if execution_mode not in _MODES:
        raise ValueError(f"unsupported S1.3 M6.2 execution mode: {execution_mode}")
    started = time.perf_counter()
    plan = build_s13_m62_execution_plan(
        p2, image_loader, photometric_config=photometric_config,
        blend_config=blend_config,
        force_owner_only_pair_indices=force_owner_only_pair_indices,
        retain_runtime_details=retain_runtime_details,
        m63_config=m63_config,
        evidence_sampling_mode=evidence_sampling_mode,
        packed_reference_fallback=packed_reference_fallback,
    )
    reference = execute_s13_m62_cpu_reference(
        plan, retain_runtime_details=retain_runtime_details
    )
    shadow_count = 0
    legacy_count = 0
    equivalence: dict[str, Any] = {
        "requested_mode": execution_mode,
        "resolved_mode": "cpu_authoritative_single_pass",
        "gpu_evaluated": False,
        "authority_gate_passed": True,
        "byte_identical": True,
        "published_pixel_authority": "cpu",
    }
    if execution_mode in {"shadow_audit", "parity_test"}:
        shadow_count = 1
        if any(item.transaction.model == "B2_safe_masked_multiband" for item in plan.blend_plans):
            equivalence.update({
                "fallback_reason": "b2_cpu_reference_fallback",
                "gpu_evaluated": False,
                "authority_gate_passed": False,
                "byte_identical": False,
            })
        else:
            try:
                shadow = execute_s13_m62_gpu_shadow(
                    plan, cpu_reference=reference, cuda_runtime=cuda_runtime
                )
                equivalence = shadow.equivalence
                equivalence.update({
                    "requested_mode": execution_mode,
                    "resolved_mode": "cpu_authoritative_shadow_audit",
                    "published_pixel_authority": "cpu",
                })
            except Exception as exc:  # CPU stays authoritative.
                equivalence.update({
                    "fallback_reason": f"gpu_shadow_failed:{type(exc).__name__}",
                    "gpu_evaluated": False,
                    "authority_gate_passed": False,
                    "byte_identical": False,
                })
    if execution_mode == "parity_test":
        legacy_count = 1
        legacy = run_s13_m6(
            p2, image_loader, photometric_config=photometric_config,
            blend_config=blend_config,
            force_owner_only_pair_indices=force_owner_only_pair_indices,
            retain_runtime_details=False,
        )
        comparison = compare_s13_m62_u8(legacy.visual_panorama, reference.visual_panorama)
        equivalence["legacy_cpu_reference_comparison"] = comparison
        if plan.q0_b0_direct_return and not comparison["authority_gate_passed"]:
            raise ValueError("S1.3 M6.2 Q0/B0 direct return diverged from legacy CPU M6")
    performance = dict(reference.performance)
    performance.update({
        "execution_mode": execution_mode,
        "decision_plan_seconds": plan.decision_plan_seconds,
        "pixel_executor_seconds": float(reference.performance.get("total_m6", 0.0)),
        "legacy_m6_call_count": legacy_count,
        "gpu_shadow_call_count": shadow_count,
        "pixel_executor_count": int(reference.performance.get("pixel_executor_count", 0)),
        "legacy_m6_executed": bool(legacy_count),
        "gpu_shadow_executed": bool(shadow_count),
        "total_m6": time.perf_counter() - started,
        "published_pixel_authority": "cpu",
        **plan.decision_timings,
    })
    object.__setattr__(reference, "performance", performance)
    return reference, {"plan": plan, "gpu_equivalence": equivalence}


def run_s13_m62_cpu_authoritative(
    p2: S13VerifiedP2, image_loader: Callable[[int], np.ndarray], *,
    cuda_runtime: object | None = None,
    force_owner_only_pair_indices: frozenset[int] = frozenset(),
    retain_runtime_details: bool = False,
    photometric_config: S13PhotometricConfig = S13PhotometricConfig(),
    blend_config: S13BlendConfig = S13BlendConfig(),
    execution_mode: S13M62ExecutionMode = "parity_test",
    m63_config: S13M63Config = S13M63Config(),
    evidence_sampling_mode: S13M62EvidenceSamplingMode = "pair_reference",
    packed_reference_fallback: bool = True,
) -> tuple[object, dict[str, Any]]:
    """Compatibility wrapper; callers should pass an explicit mode."""
    return run_s13_m62(
        p2, image_loader, execution_mode=execution_mode, cuda_runtime=cuda_runtime,
        force_owner_only_pair_indices=force_owner_only_pair_indices,
        retain_runtime_details=retain_runtime_details,
        photometric_config=photometric_config, blend_config=blend_config,
        m63_config=m63_config,
        evidence_sampling_mode=evidence_sampling_mode,
        packed_reference_fallback=packed_reference_fallback,
    )


__all__ = ["S13M62ExecutionMode", "run_s13_m62", "run_s13_m62_cpu_authoritative"]
