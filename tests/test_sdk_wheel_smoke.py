from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

from panorama_demo.paths import PROJECT_ROOT


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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_wheel_contains_exact_runtime_resource_bytes(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--no-build-isolation", "--ignore-requires-python",
         "--wheel-dir", str(tmp_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = list(tmp_path.glob("gemini305_rgbd_panorama-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        metadata = archive.read(next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))).decode()
        assert "Requires-Python: <3.11,>=3.10" in metadata
        assert "Requires-Dist: open3d" not in metadata
        assert "Requires-Dist: opencv-python-headless" in metadata
        assert len(archive.read("panorama_demo/_runtime/source_commit.txt").strip()) == 40
        for relative in RUNTIME_RESOURCES:
            packaged = archive.read(f"panorama_demo/_runtime/{relative}")
            source = (PROJECT_ROOT / relative).read_bytes()
            assert _sha256(packaged) == _sha256(source), relative
