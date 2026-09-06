"""Setuptools resource copy hook; no independent package metadata."""
from pathlib import Path
import shutil
import subprocess

from setuptools.command.build_py import build_py

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


def source_commit() -> str:
    if (ROOT / ".git").exists():
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
            raise RuntimeError("Dirty SDK builds are prohibited")
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    marker = ROOT / ".g305-source-commit"
    value = marker.read_text(encoding="ascii").strip()
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise RuntimeError("Source archive lacks its verified commit identity")
    return value


class BuildPy(build_py):
    def run(self) -> None:
        commit = source_commit()
        super().run()
        package = Path(self.build_lib) / "panorama_demo" / "_runtime"
        for relative in RUNTIME_RESOURCES:
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        (package / "source_commit.txt").write_text(commit + "\n", encoding="ascii")
