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
                               m51_r2_config: object | None) -> dict[str, Any]:
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
        )
        for key, value in result["timings"].items():
            runtime.note_stage(str(key), float(value))
        result["cuda_resident"] = runtime.report()
        return result
    finally:
        runtime.close()


__all__ = ["run_s13_cuda_fast_pipeline"]
