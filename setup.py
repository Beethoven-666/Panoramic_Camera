from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).resolve().parent
RUNTIME_RESOURCES = (
    "configs/demo.yaml",
    "configs/video_algorithms/s013_visual_continuity_v11_production.yaml",
    "configs/video_algorithms/s013_visual_continuity_v11_production.lock.json",
    "configs/video_algorithms/baseline_legacy_fast_b07b561.yaml",
    "configs/video_algorithms/baseline_legacy_fast_b07b561.lock.json",
    "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4_cuda_m63_visual_continuity_v11.yaml",
    "configs/video_candidates/s013/quality_thresholds_m61_v2.json",
    "artifacts/S013_M6_1_metrics_baseline_v2/threshold_approval.json",
)


class build_py(_build_py):
    """Copy immutable Runtime inputs into the wheel without editing bytes."""

    def run(self) -> None:
        super().run()
        package_root = Path(self.build_lib) / "panorama_demo" / "_runtime"
        for relative in RUNTIME_RESOURCES:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(f"Required Runtime resource is missing: {source}")
            destination = package_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


setup(cmdclass={"build_py": build_py})
