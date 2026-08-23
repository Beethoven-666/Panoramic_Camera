from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from panorama_demo.video_dataset_lock import (
    DIAGNOSTIC_DEVELOPMENT_RUN_NAME,
    V6_PRIMARY_DEVELOPMENT_RUN_NAME,
    create_dataset_lock,
    development_dataset_lock_path,
    verify_experiment_dataset_lock,
    write_or_verify_experiment_dataset_lock,
    write_or_verify_v6_tracking_gate_dataset_lock,
)
from panorama_demo.video_experiment import _benchmark_root, run
from panorama_demo.video_pipeline import run_video_algorithm
from panorama_demo.video_split import write_or_verify_split


def _diagnostic_session(tmp_path: Path) -> Path:
    root = tmp_path / DIAGNOSTIC_DEVELOPMENT_RUN_NAME
    (root / "color").mkdir(parents=True)
    (root / "depth_aligned").mkdir()
    for name in ("manifest.json", "calibration.json", "frames.csv"):
        (root / name).write_text("{}\n", encoding="utf-8")
    (root / "color" / "00000000.jpg").write_bytes(b"rgb")
    (root / "depth_aligned" / "00000000.png").write_bytes(b"depth")
    return root


def test_diagnostic_session_uses_a_separate_candidate_only_lock(tmp_path):
    session = _diagnostic_session(tmp_path)
    benchmark_root = tmp_path / "benchmarks" / session.name

    lock = write_or_verify_experiment_dataset_lock(session, benchmark_root, role="candidate")

    assert lock.schema == "gemini305-video-development-dataset-lock/v1"
    assert development_dataset_lock_path(benchmark_root).is_file()
    assert not (benchmark_root / "dataset_lock.json").exists()
    assert verify_experiment_dataset_lock(session, benchmark_root, role="candidate") == lock
    with pytest.raises(ValueError, match="candidate-only"):
        write_or_verify_experiment_dataset_lock(session, benchmark_root, role="baseline")
    # The old holdout/production lock verifier remains intentionally unable to
    # accept the diagnostic capture.
    with pytest.raises(ValueError, match="locked session"):
        create_dataset_lock(session)


def test_fast_primary_can_run_the_v6_candidate_matrix_but_not_production(tmp_path):
    session = _diagnostic_session(tmp_path)
    session.rename(session.with_name(V6_PRIMARY_DEVELOPMENT_RUN_NAME))
    session = session.with_name(V6_PRIMARY_DEVELOPMENT_RUN_NAME)
    benchmark_root = tmp_path / "benchmarks" / session.name

    lock = write_or_verify_experiment_dataset_lock(session, benchmark_root, role="candidate")

    assert lock.schema == "gemini305-video-development-dataset-lock/v1"
    assert verify_experiment_dataset_lock(session, benchmark_root, role="candidate") == lock
    with pytest.raises(ValueError, match="candidate-only"):
        write_or_verify_experiment_dataset_lock(session, benchmark_root, role="production")


def test_v6_tracking_gate_uses_only_the_frozen_fast_primary_bytes(tmp_path, monkeypatch):
    import panorama_demo.video_dataset_lock as dataset_lock

    session = tmp_path / dataset_lock.V6_TRACKING_GATE_RUN_NAME
    (session / "color").mkdir(parents=True)
    (session / "depth_aligned").mkdir()
    for name in ("manifest.json", "calibration.json", "frames.csv"):
        (session / name).write_text("{}\n", encoding="utf-8")
    (session / "color" / "00000000.jpg").write_bytes(b"rgb")
    (session / "depth_aligned" / "00000000.png").write_bytes(b"depth")
    expected_controls = {
        name: dataset_lock.sha256_file(session / name)
        for name in ("manifest.json", "calibration.json", "frames.csv")
    }
    monkeypatch.setattr(dataset_lock, "V6_TRACKING_GATE_CONTROL_FILE_SHA256", expected_controls)
    benchmark_root = tmp_path / "benchmarks" / session.name

    lock = write_or_verify_v6_tracking_gate_dataset_lock(session, benchmark_root)

    assert lock.schema == "gemini305-video-v6-tracking-gate-dataset-lock/v1"
    assert write_or_verify_v6_tracking_gate_dataset_lock(session, benchmark_root) == lock
    (session / "frames.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="control hash mismatch"):
        write_or_verify_v6_tracking_gate_dataset_lock(session, benchmark_root)


def test_diagnostic_benchmark_root_and_split_are_per_session_and_immutable(tmp_path):
    session = _diagnostic_session(tmp_path)
    assert _benchmark_root(session) == Path("benchmarks") / session.name
    split_path = tmp_path / "benchmarks" / session.name / "split_definition.json"
    initial = write_or_verify_split(split_path)
    split_path.write_text(json.dumps({"schema": "mutated"}), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        write_or_verify_split(split_path)
    assert initial["schema"] == "gemini305-video-split/v1"


def test_experiment_rejects_diagnostic_baseline_before_pipeline(tmp_path):
    session = _diagnostic_session(tmp_path)
    args = argparse.Namespace(
        input=session,
        output=tmp_path / "output",
        algorithm="baseline",
        candidate_config=None,
        report_level="summary",
        artifact_level="minimal",
        maximum_post_seconds=None,
        defer_3d=True,
        reuse_online_trajectory=False,
        trajectory_cache=None,
        config=None,
        progress_range=None,
        split=None,
    )
    with pytest.raises(ValueError, match="candidate-only"):
        run(args)


def test_pipeline_rejects_diagnostic_session_for_public_production(tmp_path):
    session = _diagnostic_session(tmp_path)
    with pytest.raises(ValueError, match="candidate-only"):
        run_video_algorithm(
            input_path=session,
            output=tmp_path / "output",
            role="production",
        )


def test_diagnostic_candidate_run_creates_only_development_lock(tmp_path, monkeypatch):
    session = _diagnostic_session(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    args = argparse.Namespace(
        input=session,
        output=output,
        algorithm="candidate",
        candidate_config=tmp_path / "candidate.yaml",
        report_level="summary",
        artifact_level="minimal",
        maximum_post_seconds=None,
        defer_3d=True,
        reuse_online_trajectory=True,
        trajectory_cache=None,
        config=None,
        progress_range=(0.30, 0.48),
        split="validation",
    )
    monkeypatch.setattr("panorama_demo.video_experiment.load_config", lambda _: {"stitch": {}})
    benchmark_root = tmp_path / "benchmarks" / session.name
    monkeypatch.setattr("panorama_demo.video_experiment._benchmark_root", lambda _: benchmark_root)
    seen: dict[str, object] = {}

    def _run_video_algorithm(**kwargs):
        seen.update(kwargs)
        return {"panorama": "diagnostic.png"}

    monkeypatch.setattr("panorama_demo.video_experiment.run_video_algorithm", _run_video_algorithm)

    run(args)

    assert development_dataset_lock_path(benchmark_root).is_file()
    assert not (benchmark_root / "dataset_lock.json").exists()
    assert (benchmark_root / "split_definition.json").is_file()
    assert seen["reuse_online_trajectory"] is True
