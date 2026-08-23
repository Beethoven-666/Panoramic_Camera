from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from panorama_demo.video_s13_motion import (
    descriptive_delta_risk,
)
from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_bundle import verify_p0_completion
from panorama_demo.video_s13_experiment import run_s13_experiment


ROOT = Path(__file__).resolve().parents[1]
REAL_ROOT = ROOT / "data/captures/video"
RUN_REAL = os.environ.get("G305_RUN_REAL_S13_REGRESSION") == "1"
CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v3.yaml"


def _run_real(session: Path, output: Path) -> dict:
    return run_s13_experiment(
        input_path=session,
        output=output,
        candidate_config=CONFIG,
        algorithm_spec=build_algorithm_spec(CONFIG, expected_role="candidate"),
        trajectory_cache=None,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=True,
        config_path=None,
    )


@pytest.mark.skipif(not RUN_REAL, reason="set G305_RUN_REAL_S13_REGRESSION=1 for real RGB regression")
def test_s13_real_slow_rgb_progress_is_not_blocked_by_legacy_delta(tmp_path: Path) -> None:
    report = _run_real(REAL_ROOT / "run_20260806_153033", tmp_path / "slow")
    layout = json.loads((Path(report["generation"]) / "P0/base_layout.json").read_text(encoding="utf-8"))
    legacy = descriptive_delta_risk(7.43485)
    assert report["render_state"] == "base_generated"
    assert report["direct_local_delta_is_fatal"] is False
    assert legacy["risk"] is True
    assert legacy["structural_gate"] is False
    assert layout["layout_level"] == "L0_coherent_rgb"
    assert sum(step["selected_hypothesis_id"] >= 0 for step in layout["coherent_lineage"]) >= 54
    assert layout["centers_x"][54] - layout["centers_x"][0] < 8.0
    assert not set(range(1, 55)).intersection(layout["selected_frame_ids"])
    assert layout["observed_pause_expansion_count"] == 0
    assert layout["observed_pause_expansion_px"] == 0.0
    assert layout["canvas_width"] < 1800
    assert verify_p0_completion(Path(report["completion"]).parent)["sealed"] is True


@pytest.mark.skipif(not RUN_REAL, reason="set G305_RUN_REAL_S13_REGRESSION=1 for real RGB regression")
def test_s13_real_fast_disconnected_telemetry_still_has_complete_progress(tmp_path: Path) -> None:
    report = _run_real(REAL_ROOT / "run_20260807_140140", tmp_path / "fast")
    layout = json.loads((Path(report["generation"]) / "P0/base_layout.json").read_text(encoding="utf-8"))
    assert report["render_state"] == "base_generated"
    assert report["motion_graph_connected_telemetry"] is False
    assert report["motion_graph_disconnection_is_fatal"] is False
    assert layout["layout_level"] in {"L0_coherent_rgb", "L1_rgb_pose_soft"}
    assert sum(step["selected_hypothesis_id"] >= 0 for step in layout["coherent_lineage"]) >= 29
    assert layout["centers_x"][29] - layout["centers_x"][0] < 8.0
    assert not set(range(1, 30)).intersection(layout["selected_frame_ids"])
    assert layout["observed_pause_expansion_count"] == 0
    assert layout["observed_pause_expansion_px"] == 0.0
    assert layout["canvas_width"] < 2150
    assert Path(report["current_base"]).is_file()
    assert verify_p0_completion(Path(report["completion"]).parent)["sealed"] is True
