from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from panorama_demo.video_algorithm import canonical_config_sha256
from panorama_demo.video_algorithm_lock import verify_algorithm_lock
from panorama_demo.video_algorithm_registry import VideoAlgorithmRegistry, VideoAlgorithmRegistryError
from panorama_demo.video_algorithm_selection import write_validation_selection
from panorama_demo.video_production_freeze import (
    VideoProductionFreezeError,
    freeze_production,
    record_first_holdout,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE_LOCK = ROOT / "configs" / "video_algorithms" / "baseline_legacy_fast_b07b561.lock.json"
CANDIDATE = ROOT / "configs" / "video_candidates" / "C1_constrained_owner.yaml"


def test_baseline_lock_freezes_legacy_fast_configuration():
    spec = verify_algorithm_lock(BASELINE_LOCK, expected_role="baseline")

    assert spec.algorithm_id == "legacy_fast_b07b561"
    assert spec.implementation_id == "legacy_visual_seam"
    assert spec.config_sha256 == "15b26b47bf5a25d78073abb1be49bc6c69437a6250c9505dcb79af857e82c29b"


def test_registry_never_allows_mutable_candidate_config_for_baseline_or_production(tmp_path):
    registry = VideoAlgorithmRegistry(
        baseline_lock=BASELINE_LOCK,
        production_lock=tmp_path / "production.lock.json",
    )

    with pytest.raises(VideoAlgorithmRegistryError, match="baseline does not accept candidate_config"):
        registry.resolve("baseline", candidate_config=CANDIDATE)
    with pytest.raises(VideoAlgorithmRegistryError, match="production does not accept candidate_config"):
        registry.resolve("production", candidate_config=CANDIDATE)
    with pytest.raises(VideoAlgorithmRegistryError, match="does not exist"):
        registry.resolve("production")


def test_candidate_never_gets_an_automatic_baseline_fallback(tmp_path):
    registry = VideoAlgorithmRegistry(
        baseline_lock=BASELINE_LOCK,
        production_lock=tmp_path / "production.lock.json",
    )
    assert registry.resolve("candidate", candidate_config=CANDIDATE).allow_baseline_fallback is False
    with pytest.raises(VideoAlgorithmRegistryError, match="candidate requires candidate_config"):
        registry.resolve("candidate")


def _write_candidate(path: Path) -> Path:
    payload = {
        "config_schema": "gemini305-video-candidate/v1",
        "role": "candidate",
        "candidate_id": "C_final",
        "parent_candidate_id": "C0_baseline_reference",
        "algorithm_id": "C_final",
        "implementation_id": "video_visual_renderer_v2",
        "source_commit": "a" * 40,
        "model_sha256": {},
        "allow_baseline_fallback": False,
        "changed_components": ["strict_owner"],
    }
    payload["config_sha256"] = canonical_config_sha256(payload)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _write_measurement_sidecars(directory: Path) -> None:
    (directory / "video_annotation_source_progress_audit.json").write_text(
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
    (directory / "visual_metrics.json").write_text(
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


def _write_report(path: Path, *, scope: str, grades: str = "A") -> Path:
    path.write_text(
        json.dumps(
            {
                "algorithm": {
                    "role": "candidate",
                    "algorithm_id": "C_final",
                    "implementation_id": "video_visual_renderer_v2",
                    "execution_backend": "video_visual_renderer_v2_cuda",
                    "config_sha256": None,
                    "source_commit": "a" * 40,
                    "model_sha256": {},
                    "fallback_used": False,
                },
                "evaluation_scope": scope,
                "grades": {name: grades for name in ("structural", "visual", "performance", "overall")},
                "renderer": {"quality_metrics": {"candidate_mesh_evidence_output_warp_applied": True}},
            }
        ),
        encoding="utf-8",
    )
    _write_measurement_sidecars(path.parent)
    return path


def _prepare_selection_and_holdout(tmp_path: Path, *, holdout_grade: str = "A") -> tuple[Path, Path, Path]:
    candidate = _write_candidate(tmp_path / "candidate.yaml")
    config = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    validation = _write_report(tmp_path / "validation.json", scope="validation_only")
    report = json.loads(validation.read_text(encoding="utf-8"))
    report["algorithm"]["config_sha256"] = config["config_sha256"]
    validation.write_text(json.dumps(report), encoding="utf-8")
    selection = tmp_path / "selection.json"
    write_validation_selection([validation], output=selection)
    holdout_dir = tmp_path / "holdout"
    holdout_dir.mkdir()
    holdout = _write_report(holdout_dir / "report.json", scope="holdout_only", grades=holdout_grade)
    report = json.loads(holdout.read_text(encoding="utf-8"))
    report["algorithm"]["config_sha256"] = config["config_sha256"]
    holdout.write_text(json.dumps(report), encoding="utf-8")
    return candidate, selection, holdout


def test_production_freeze_requires_selected_candidate_and_one_passing_holdout(tmp_path):
    candidate, selection, holdout = _prepare_selection_and_holdout(tmp_path)
    holdout_state = tmp_path / "holdout_state.json"
    state = record_first_holdout(
        validation_selection=selection, holdout_report=holdout, output=holdout_state
    )
    assert state["first_holdout_pass"] is True
    dataset_lock = tmp_path / "dataset_lock.json"
    dataset_lock.write_text(json.dumps({"schema": "gemini305-video-dataset-lock/v1"}), encoding="utf-8")
    config_target = tmp_path / "production_v1.yaml"
    lock_target = tmp_path / "production.lock.json"

    lock = freeze_production(
        validation_selection=selection,
        holdout_state=holdout_state,
        candidate_config=candidate,
        dataset_lock=dataset_lock,
        production_config=config_target,
        production_lock=lock_target,
    )

    assert lock["role"] == "production"
    assert lock["freeze_evidence"]["selected_candidate_algorithm_id"] == "C_final"
    spec = verify_algorithm_lock(lock_target, expected_role="production")
    assert spec.algorithm_id == "production_v1"
    assert spec.allow_baseline_fallback is True


def test_first_holdout_is_consumed_even_when_it_fails_and_cannot_freeze(tmp_path):
    candidate, selection, holdout = _prepare_selection_and_holdout(tmp_path, holdout_grade="C")
    holdout_state = tmp_path / "holdout_state.json"
    state = record_first_holdout(
        validation_selection=selection, holdout_report=holdout, output=holdout_state
    )
    assert state["first_holdout_pass"] is False
    with pytest.raises(VideoProductionFreezeError, match="overwrite immutable evidence"):
        record_first_holdout(
            validation_selection=selection, holdout_report=holdout, output=holdout_state
        )
    dataset_lock = tmp_path / "dataset_lock.json"
    dataset_lock.write_text(json.dumps({"schema": "gemini305-video-dataset-lock/v1"}), encoding="utf-8")
    with pytest.raises(VideoProductionFreezeError, match="did not pass"):
        freeze_production(
            validation_selection=selection,
            holdout_state=holdout_state,
            candidate_config=candidate,
            dataset_lock=dataset_lock,
            production_config=tmp_path / "production.yaml",
            production_lock=tmp_path / "production.lock.json",
        )


def test_first_holdout_rejects_a_legacy_bridge_report(tmp_path):
    _, selection, holdout = _prepare_selection_and_holdout(tmp_path)
    report = json.loads(holdout.read_text(encoding="utf-8"))
    report["algorithm"]["execution_backend"] = "legacy_candidate_experiment_bridge"
    holdout.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(VideoProductionFreezeError, match="not executed by the v2 CUDA renderer"):
        record_first_holdout(
            validation_selection=selection,
            holdout_report=holdout,
            output=tmp_path / "holdout_state.json",
        )


def test_production_freeze_rejects_candidate_mutated_after_holdout(tmp_path):
    candidate, selection, holdout = _prepare_selection_and_holdout(tmp_path)
    state_path = tmp_path / "holdout_state.json"
    record_first_holdout(validation_selection=selection, holdout_report=holdout, output=state_path)
    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    payload["changed_components"] = ["mutated"]
    payload["config_sha256"] = canonical_config_sha256(payload)
    candidate.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    dataset_lock = tmp_path / "dataset_lock.json"
    dataset_lock.write_text(json.dumps({"schema": "gemini305-video-dataset-lock/v1"}), encoding="utf-8")

    with pytest.raises(VideoProductionFreezeError, match="no longer matches first holdout config_sha256"):
        freeze_production(
            validation_selection=selection,
            holdout_state=state_path,
            candidate_config=candidate,
            dataset_lock=dataset_lock,
            production_config=tmp_path / "production.yaml",
            production_lock=tmp_path / "production.lock.json",
        )
