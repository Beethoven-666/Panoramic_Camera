from __future__ import annotations

from panorama_demo.video_v6_development_matrix import build_v6_development_matrix


def _report(*, quality: bool = True, seconds: float = 9.0) -> dict[str, object]:
    return {
        "algorithm": {"algorithm_id": "V6_rgb_only_graphcut"},
        "renderer": {
            "raw_rgb_once_sampling": {
                "exactly_once": True, "source_frame_ids": [10, 20], "source_sampling_call_count": 2,
            },
            "quality_metrics": {"strict_quality_pass": quality},
        },
        "source_frame_ids": [10, 20],
        "open3d_edges": [{"reference_node_id": 10, "source_node_id": 20}],
        "untracked_motion_analysis_frames_rendered": False,
        "performance": {"primary_post_capture_seconds": seconds},
    }


def test_v6_matrix_requires_all_frozen_evidence_and_keeps_production_false() -> None:
    reports = {
        "FAST_PRIMARY_DEVELOPMENT": _report(),
        "FAST_PRESSURE_REGRESSION": _report(),
        "SLOW_DEVELOPMENT_CONTROL": _report(),
    }

    matrix = build_v6_development_matrix(reports)

    assert matrix["development_matrix_pass"] is True
    assert matrix["production_3m_pass"] is False
    assert matrix["datasets"]["FAST_PRIMARY_DEVELOPMENT"]["current_dataset_candidate_pass"] is True


def test_v6_matrix_exposes_visual_and_timing_failures_without_weakening_them() -> None:
    reports = {
        "FAST_PRIMARY_DEVELOPMENT": _report(),
        "FAST_PRESSURE_REGRESSION": _report(quality=False),
        "SLOW_DEVELOPMENT_CONTROL": _report(seconds=20.1),
    }

    matrix = build_v6_development_matrix(reports)

    assert matrix["development_matrix_pass"] is False
    assert "strict_visual_gate_failed" in matrix["datasets"]["FAST_PRESSURE_REGRESSION"]["failure_reasons"]
    assert "candidate_hard_20_second_limit_exceeded" in matrix["datasets"]["SLOW_DEVELOPMENT_CONTROL"]["failure_reasons"]
