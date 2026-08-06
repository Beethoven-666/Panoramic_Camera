from __future__ import annotations

import json

from panorama_demo.video_algorithm_selection import assess_validation_candidate, write_validation_selection


def _report(*, scope: str = "validation_only", mesh_applied: bool = True) -> dict[str, object]:
    return {
        "algorithm": {
            "role": "candidate",
            "algorithm_id": "C8",
            "fallback_used": False,
            "execution_backend": "video_visual_renderer_v2_cuda",
            "executed_candidate_components": {
                "c1_constrained_owner": True,
                "c2_dis_mesh": True,
                "c3_raft_mesh": True,
                "c4_depth_layered_mesh": True,
                "c5_object_owner_lock": True,
                "c6_safe_multiband": True,
                "c7_global_photometric": True,
                "c8_local_multilabel_owner": True,
            },
        },
        "evaluation_scope": scope,
        "grades": {"structural": "A", "visual": "A", "performance": "A", "overall": "A"},
        "renderer": {"quality_metrics": {"candidate_mesh_evidence_output_warp_applied": mesh_applied}},
    }


def test_selection_requires_validation_only_a_grades_and_output_mesh(tmp_path):
    candidate = tmp_path / "report.json"
    candidate.write_text(json.dumps(_report()), encoding="utf-8")
    (tmp_path / "video_annotation_source_progress_audit.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-video-annotation-source-progress-audit/v1",
                "measurement_only": True,
                "verified": True,
                "selection_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "visual_metrics.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-video-offline-visual-evaluation/v1",
                "measurement_only": True,
                "automatic_grade_promotion_allowed": False,
                "projection_available": True,
                "object_integrity": {"object": {"status": "evaluated", "hard_gate_pass": True}},
                "line_continuity": {"line": {"status": "evaluated", "hard_gate_pass": True}},
                "safe_background": {"background": {"status": "evaluated", "hard_gate_pass": True}},
            }
        ),
        encoding="utf-8",
    )
    assessed = assess_validation_candidate(candidate)
    assert assessed.eligible
    result = write_validation_selection([candidate], output=tmp_path / "selection.json")
    assert result["selected_algorithm_id"] == "C8"

    candidate.write_text(json.dumps(_report(scope="full_scan", mesh_applied=False)), encoding="utf-8")
    assessed = assess_validation_candidate(candidate)
    assert not assessed.eligible
    assert "not_validation_only" in assessed.reasons
    assert "mesh_evidence_not_applied_to_output" in assessed.reasons


def test_selection_rejects_missing_or_unevaluable_measurement_sidecars(tmp_path):
    candidate = tmp_path / "report.json"
    candidate.write_text(json.dumps(_report()), encoding="utf-8")

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "annotation_source_progress_audit_missing_or_invalid" in assessed.reasons
    assert "offline_visual_measurement_missing_or_invalid" in assessed.reasons


def test_selection_rejects_legacy_candidate_bridge_even_when_other_evidence_is_a(tmp_path):
    candidate = tmp_path / "report.json"
    candidate.write_text(
        json.dumps(
            {
                **_report(),
                "algorithm": {
                    "role": "candidate",
                    "algorithm_id": "C8",
                    "fallback_used": False,
                    "execution_backend": "legacy_candidate_experiment_bridge",
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "video_annotation_source_progress_audit.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-video-annotation-source-progress-audit/v1",
                "measurement_only": True,
                "verified": True,
                "selection_eligible": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "visual_metrics.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-video-offline-visual-evaluation/v1",
                "measurement_only": True,
                "automatic_grade_promotion_allowed": False,
                "projection_available": True,
                "object_integrity": {"object": {"status": "evaluated", "hard_gate_pass": True}},
                "line_continuity": {"line": {"status": "evaluated", "hard_gate_pass": True}},
                "safe_background": {"background": {"status": "evaluated", "hard_gate_pass": True}},
            }
        ),
        encoding="utf-8",
    )

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "candidate_not_executed_by_v2_cuda_renderer" in assessed.reasons


def test_selection_requires_the_exact_declared_v2_component_lineage(tmp_path):
    candidate = tmp_path / "report.json"
    report = _report()
    report["algorithm"]["algorithm_id"] = "C4_raft_rgbd_layered_mesh"
    report["algorithm"]["executed_candidate_components"] = {
        "c1_constrained_owner": True,
        "c3_raft_mesh": True,
        "c4_depth_layered_mesh": False,
    }
    candidate.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "video_annotation_source_progress_audit.json").write_text(
        json.dumps({"schema": "gemini305-video-annotation-source-progress-audit/v1", "measurement_only": True, "verified": True, "selection_eligible": True}),
        encoding="utf-8",
    )
    (tmp_path / "visual_metrics.json").write_text(
        json.dumps({
            "schema": "gemini305-video-offline-visual-evaluation/v1", "measurement_only": True,
            "automatic_grade_promotion_allowed": False, "projection_available": True,
            "object_integrity": {"object": {"status": "evaluated", "hard_gate_pass": True}},
            "line_continuity": {"line": {"status": "evaluated", "hard_gate_pass": True}},
            "safe_background": {"background": {"status": "evaluated", "hard_gate_pass": True}},
        }),
        encoding="utf-8",
    )

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "v2_component_not_executed:c4_depth_layered_mesh" in assessed.reasons
