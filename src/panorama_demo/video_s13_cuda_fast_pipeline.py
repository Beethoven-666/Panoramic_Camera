"""CUDA-resident successor runner for the isolated S1.3 v4 candidate.

The CPU reference remains the authority for all discrete decisions.  The
resident cache is initialized before the writer so a required-CUDA failure can
never leave stage images behind.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cuda_backend import reset_cuda_audit
from .video_s13_cuda_runtime import S13CudaRuntime
from .video_s13_fast_pipeline import run_s13_fast_pipeline
from .video_s13_session import S13Session
from .video_s13_trajectory import S13Trajectory


def run_s13_cuda_fast_pipeline(*, session: S13Session, trajectory: S13Trajectory,
                               output: Path, analysis_width_px: int,
                               normal_target_advance_px: float,
                               risky_target_advance_px: float,
                               m51_r2_config: object | None,
                               m6_cuda_v2: bool = False,
                               structural_equivalent_v3: bool = False,
                               m62_equivalence: bool = False) -> dict[str, Any]:
    reset_cuda_audit()
    runtime = S13CudaRuntime()
    try:
        result = run_s13_fast_pipeline(
            session=session, trajectory=trajectory, output=output,
            analysis_width_px=analysis_width_px,
            normal_target_advance_px=normal_target_advance_px,
            risky_target_advance_px=risky_target_advance_px,
            m51_r2_config=m51_r2_config,
            resident_runtime=runtime,
            p0_resident_device_remap=runtime.remap_resident_frame_device,
            # The v3 probe calibration utility is intentionally not a pixel
            # authority until its guarded intervals are wired into every M4
            # decision.  Use the proven full reference selection meanwhile.
            vertical_exact_seam_probes=False,
            p1_reference_remap=True,
            m6_cuda_v2=m6_cuda_v2,
            m6_cuda_v3=structural_equivalent_v3,
            m62_equivalence=m62_equivalence,
        )
        for key, value in result["timings"].items():
            runtime.note_stage(str(key), float(value))
        audit = runtime.report()
        audit.update(result["writer_audit"])
        audit["c2e_full_provenance_copy_count"] = result["c2e_full_provenance_copy_count"]
        if structural_equivalent_v3:
            audit.update({"equivalence_mode": "structural_exact_rgb_tolerant",
                          "probe_mode": "guarded_tolerant_reference_fallback",
                          "probe_reference_fallback_count": 1,
                          "final_linear_full_d2h_count": result["m6_performance"].get("final_linear_full_d2h_count", 0)})
        for key in (
            "m6_full_source_map_count", "m6_roi_source_map_count",
            "full_corrected_source_d2h_count", "compact_corrected_source_d2h_count",
            "corrected_pair_corridor_d2h_count",
            "final_linear_full_d2h_count", "gain_bias_h2d_count",
        ):
            if key in result["m6_performance"]:
                audit[key] = result["m6_performance"][key]
        result["cuda_resident"] = audit
        return result
    finally:
        runtime.close()


__all__ = ["run_s13_cuda_fast_pipeline"]
