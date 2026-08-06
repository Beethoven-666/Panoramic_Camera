from __future__ import annotations

import json

from panorama_demo.video_algorithm_selection import assess_validation_candidate, write_validation_selection
from panorama_demo.video_recovery import QUALITY_GATE_LOCK, canonical_sha256


def _report(*, scope: str = "validation_only", mesh_applied: bool = True) -> dict[str, object]:
    components = (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh",
        "c5_object_owner_lock", "c6_safe_multiband", "c7_global_photometric",
        "c8_local_multilabel_owner",
    )
    return {
        "algorithm": {
            "role": "candidate",
            "algorithm_id": "C8_multilabel_window",
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
            "required_components": list(components),
            "component_execution": {
                component: {
                    "required": True, "initialized": True,
                    "attempted_pair_count": 1, "accepted_pair_count": 1,
                    "applied_pair_count": 1, "applied_output_pixel_count": 8,
                    "maximum_applied_displacement_px": 1.0,
                    "fallback_pair_count": 0, "applied_to_output": True,
                    "rejection_reasons": {},
                }
                for component in components
            },
            "candidate_run_state": "completed",
            "component_execution_selection_eligible": True,
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
                "schema": "gemini305-video-offline-visual-evaluation/v2",
                "measurement_only": True,
                "automatic_grade_promotion_allowed": False,
                "projection_available": True,
                "object_integrity": {"object": {"status": "evaluated", "hard_gate_pass": True}},
                "line_continuity": {"line": {"status": "evaluated", "hard_gate_pass": True, "line_step_p95_px": 0.1, "line_orientation_delta_p95_degrees": 0.2}},
                "safe_background": {"background": {"status": "evaluated", "hard_gate_pass": True, "delta_e00_p95": 0.1, "brightness_step_p95_percent": 0.1}},
            }
        ),
        encoding="utf-8",
    )
    assessed = assess_validation_candidate(candidate)
    assert assessed.eligible
    result = write_validation_selection([candidate], output=tmp_path / "selection.json")
    assert result["selected_algorithm_id"] == "C8_multilabel_window"

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
                "schema": "gemini305-video-offline-visual-evaluation/v2",
                "measurement_only": True,
                "automatic_grade_promotion_allowed": False,
                "projection_available": True,
                "object_integrity": {"object": {"status": "evaluated", "hard_gate_pass": True}},
                "line_continuity": {"line": {"status": "evaluated", "hard_gate_pass": True, "line_step_p95_px": 0.1, "line_orientation_delta_p95_degrees": 0.2}},
                "safe_background": {"background": {"status": "evaluated", "hard_gate_pass": True, "delta_e00_p95": 0.1, "brightness_step_p95_percent": 0.1}},
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
    report["algorithm"]["required_components"] = [
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh"
    ]
    report["algorithm"]["component_execution"] = {
        "c1_constrained_owner": {
            "required": True, "initialized": True, "applied_to_output": True,
            "applied_output_pixel_count": 2,
        },
        "c3_raft_mesh": {
            "required": True, "initialized": True, "applied_to_output": True,
            "applied_output_pixel_count": 2,
        },
        "c4_depth_layered_mesh": {
            "required": True, "initialized": True, "applied_to_output": False,
            "applied_output_pixel_count": 0,
        },
    }
    candidate.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "video_annotation_source_progress_audit.json").write_text(
        json.dumps({"schema": "gemini305-video-annotation-source-progress-audit/v1", "measurement_only": True, "verified": True, "selection_eligible": True}),
        encoding="utf-8",
    )
    (tmp_path / "visual_metrics.json").write_text(
        json.dumps({
            "schema": "gemini305-video-offline-visual-evaluation/v2", "measurement_only": True,
            "automatic_grade_promotion_allowed": False, "projection_available": True,
            "object_integrity": {"object": {"status": "evaluated", "hard_gate_pass": True}},
            "line_continuity": {"line": {"status": "evaluated", "hard_gate_pass": True, "line_step_p95_px": 0.1, "line_orientation_delta_p95_degrees": 0.2}},
            "safe_background": {"background": {"status": "evaluated", "hard_gate_pass": True, "delta_e00_p95": 0.1, "brightness_step_p95_percent": 0.1}},
        }),
        encoding="utf-8",
    )

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "v2_component_not_applied_to_output:c4_depth_layered_mesh" in assessed.reasons


def test_selection_rejects_bool_only_component_claim_without_final_output_lineage(tmp_path):
    candidate = tmp_path / "report.json"
    report = _report()
    report["algorithm"].pop("component_execution")
    candidate.write_text(json.dumps(report), encoding="utf-8")

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "v2_component_execution_audit_missing" in assessed.reasons


def test_v3_selection_ranks_all_eligible_candidates_with_the_locked_gates(tmp_path):
    def write_candidate(name: str, algorithm_id: str, components: list[str], *, quality: float, warm: float, vram: int):
        root = tmp_path / name
        root.mkdir()
        report = _report()
        report["algorithm"]["algorithm_id"] = algorithm_id
        report["algorithm"]["required_components"] = components
        report["algorithm"]["component_execution"] = {
            component: {"required": True, "initialized": True, "applied_to_output": True, "applied_output_pixel_count": 9}
            for component in components
        }
        report["benchmark"] = {"warm_median_seconds": warm, "peak_vram_bytes": vram}
        (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
        (root / "video_annotation_source_progress_audit.json").write_text(json.dumps({
            "schema": "gemini305-video-annotation-source-progress-audit/v1", "measurement_only": True,
            "verified": True, "selection_eligible": True,
        }), encoding="utf-8")
        # Lower observed errors mean a higher deterministic quality margin.
        (root / "visual_metrics.json").write_text(json.dumps({
            "schema": "gemini305-video-offline-visual-evaluation/v2", "measurement_only": True,
            "automatic_grade_promotion_allowed": False, "projection_available": True,
            "object_integrity": {"o": {"status": "evaluated", "hard_gate_pass": True}},
            "line_continuity": {"l": {"status": "evaluated", "hard_gate_pass": True, "line_step_p95_px": quality, "line_orientation_delta_p95_degrees": quality}},
            "safe_background": {"b": {"status": "evaluated", "hard_gate_pass": True, "delta_e00_p95": quality, "brightness_step_p95_percent": quality}},
        }), encoding="utf-8")
        return root / "report.json"

    first = write_candidate("first", "C1_constrained_owner", ["c1_constrained_owner"], quality=0.5, warm=2.0, vram=100)
    second = write_candidate("second", "C2_dis_rgb_mesh", ["c1_constrained_owner", "c2_dis_mesh"], quality=0.1, warm=7.0, vram=900)
    lock = {"schema": "gemini305-video-quality-gate-lock/v1", "quality_gate_version": "quality-gates-v1-unchanged", "gates": QUALITY_GATE_LOCK}
    lock["sha256"] = canonical_sha256(lock)
    lock_path = tmp_path / "quality_gate_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    result = write_validation_selection([first, second], output=tmp_path / "selection.json", quality_gate_lock=lock_path)

    assert result["schema"] == "gemini305-video-algorithm-selection/v3"
    assert result["selected_algorithm_id"] == "C2_dis_rgb_mesh"
    assert result["selection_status"] == "ready_for_first_holdout"
    assert result["eligible_candidate_count"] == 2
    assert result["quality_gate_lock"]["sha256"] == lock["sha256"]
    ranked = {candidate["algorithm_id"]: candidate["ranking"]["rank"] for candidate in result["candidates"]}
    assert ranked == {"C1_constrained_owner": 2, "C2_dis_rgb_mesh": 1}
