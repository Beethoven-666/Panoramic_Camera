from __future__ import annotations

import json
from pathlib import Path

from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_algorithm_selection import assess_validation_candidate, write_validation_selection
from panorama_demo.video_delivery import write_invalid_candidate_experiment
from panorama_demo.video_recovery import QUALITY_GATE_LOCK, canonical_sha256


ROOT = Path(__file__).resolve().parents[1]


def _report(
    *, scope: str = "validation_only", mesh_applied: bool = True,
    candidate_id: str = "C8_multilabel_window",
) -> dict[str, object]:
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / f"{candidate_id}.yaml",
        expected_role="candidate",
    )
    components = spec.required_output_components
    return {
        "algorithm": {
            **spec.as_dict(),
            "role": "candidate",
            "fallback_used": False,
            "working_tree_dirty": False,
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
            "component_evidence": {
                component: {"valid": True, "sample_count": 8, "audit_passed": True}
                for component in spec.required_evidence_components
            },
            "candidate_run_state": "completed",
            "component_execution_selection_eligible": True,
        },
        "evaluation_scope": scope,
        "grades": {"structural": "A", "visual": "A", "performance": "A", "overall": "A"},
        "renderer": {"quality_metrics": {"candidate_mesh_evidence_output_warp_applied": mesh_applied}},
    }


def _write_full_3m_minimal_benchmark(root, report_path, *, warm: float = 2.0, vram: int = 100) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    (root / "benchmark.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-video-benchmark-result/v2",
                "benchmark_kind": "full_3m_summary_minimal",
                "observability": {"report_level": "summary", "artifact_level": "minimal"},
                "evaluation_scope": "validation_only",
                "algorithm": report["algorithm"],
                "runs": [{"video_report": str(report_path)}],
                "warm_median_seconds": warm,
                "peak_vram_bytes": vram,
                "performance_gate": {"status": "passed"},
            }
        ),
        encoding="utf-8",
    )


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
    _write_full_3m_minimal_benchmark(tmp_path, candidate)
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


def test_selection_requires_a_full_3m_summary_minimal_benchmark_not_report_timing(tmp_path):
    candidate = tmp_path / "report.json"
    report = _report()
    # Validation has deliberately not evaluated performance.  A complete
    # benchmark, not this grade or a report-local elapsed value, is the only
    # admissible performance evidence.
    report["grades"]["performance"] = "NE"
    report["performance"] = {"primary_post_capture_seconds": 0.001}
    candidate.write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "video_annotation_source_progress_audit.json").write_text(json.dumps({
        "schema": "gemini305-video-annotation-source-progress-audit/v1", "measurement_only": True,
        "verified": True, "selection_eligible": True,
    }), encoding="utf-8")
    (tmp_path / "visual_metrics.json").write_text(json.dumps({
        "schema": "gemini305-video-offline-visual-evaluation/v2", "measurement_only": True,
        "automatic_grade_promotion_allowed": False, "projection_available": True,
        "object_integrity": {"o": {"status": "evaluated", "hard_gate_pass": True}},
        "line_continuity": {"l": {"status": "evaluated", "hard_gate_pass": True, "line_step_p95_px": 0.1, "line_orientation_delta_p95_degrees": 0.1}},
        "safe_background": {"b": {"status": "evaluated", "hard_gate_pass": True, "delta_e00_p95": 0.1, "brightness_step_p95_percent": 0.1}},
    }), encoding="utf-8")

    assert "performance_benchmark_missing_or_invalid" in assess_validation_candidate(candidate).reasons

    _write_full_3m_minimal_benchmark(tmp_path, candidate)
    assert assess_validation_candidate(candidate).eligible

    summary = json.loads((tmp_path / "benchmark.json").read_text(encoding="utf-8"))
    summary["observability"]["artifact_level"] = "audit"
    (tmp_path / "benchmark.json").write_text(json.dumps(summary), encoding="utf-8")
    assert "performance_benchmark_is_not_summary_minimal" in assess_validation_candidate(candidate).reasons


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
    _write_full_3m_minimal_benchmark(tmp_path, candidate)

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "candidate_not_executed_by_v2_cuda_renderer" in assessed.reasons


def test_selection_requires_the_exact_declared_v2_component_lineage(tmp_path):
    candidate = tmp_path / "report.json"
    report = _report(candidate_id="C4_raft_rgbd_layered_mesh")
    report["algorithm"]["executed_candidate_components"] = {
        "c1_constrained_owner": True,
        "c3_raft_mesh": True,
        "c4_depth_layered_mesh": False,
    }
    report["algorithm"]["component_execution"] = {
        "c1_constrained_owner": {
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
    _write_full_3m_minimal_benchmark(tmp_path, candidate)

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "candidate_output_component_not_applied:c4_depth_layered_mesh" in assessed.reasons


def test_selection_rejects_bool_only_component_claim_without_final_output_lineage(tmp_path):
    candidate = tmp_path / "report.json"
    report = _report()
    report["algorithm"].pop("component_execution")
    candidate.write_text(json.dumps(report), encoding="utf-8")

    assessed = assess_validation_candidate(candidate)

    assert not assessed.eligible
    assert "candidate_output_component_execution_audit_missing" in assessed.reasons


def test_selection_requires_locked_manifest_evidence_and_clean_runtime_provenance(tmp_path):
    candidate = tmp_path / "report.json"
    report = _report(candidate_id="C4_raft_rgbd_layered_mesh")
    report["algorithm"]["working_tree_dirty"] = True
    # C4 replaces C3 output: C3 must remain audited evidence, rather than a
    # fabricated requirement that it affected the final pixels.
    report["algorithm"]["component_execution"].pop("c3_raft_mesh", None)
    candidate.write_text(json.dumps(report), encoding="utf-8")

    assessed = assess_validation_candidate(candidate)

    assert "candidate_working_tree_dirty_or_unrecorded" in assessed.reasons
    assert not any("c3_raft_mesh" in reason and "output" in reason for reason in assessed.reasons)


def test_invalid_component_execution_persists_only_an_f_experiment_report(tmp_path):
    report = _report()
    report["algorithm"]["candidate_run_state"] = "invalid_component_execution"
    report.update({
        "delivery_state": "experiment_invalid",
        "strict_quality_pass": False,
        "grades": {"implementation": "F", "overall": "F"},
    })

    written = write_invalid_candidate_experiment(tmp_path, report)

    assert written["delivery_state"] == "experiment_invalid"
    assert json.loads((tmp_path / "video_report.json").read_text(encoding="utf-8"))["grades"]["overall"] == "F"
    assert not (tmp_path / "video_delivery.json").exists()


def test_v3_selection_ranks_all_eligible_candidates_with_the_locked_gates(tmp_path):
    def write_candidate(name: str, algorithm_id: str, components: list[str], *, quality: float, warm: float, vram: int):
        root = tmp_path / name
        root.mkdir()
        report = _report(candidate_id=algorithm_id)
        report["algorithm"]["required_components"] = components
        report["algorithm"]["component_execution"] = {
            component: {"required": True, "initialized": True, "applied_to_output": True, "applied_output_pixel_count": 9}
            for component in components
        }
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
        _write_full_3m_minimal_benchmark(root, root / "report.json", warm=warm, vram=vram)
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
