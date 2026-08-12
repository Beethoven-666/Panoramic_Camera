from __future__ import annotations

import argparse
import json
from pathlib import Path

import panorama_demo.video_experiment as video_experiment
import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v3.yaml"
FORMAL_M6_CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4.yaml"


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "input": tmp_path / "input",
        "output": tmp_path / "output",
        "algorithm": "candidate",
        "candidate_config": CONFIG,
        "report_level": "summary",
        "artifact_level": "minimal",
        "maximum_post_seconds": None,
        "defer_3d": True,
        "reuse_online_trajectory": False,
        "trajectory_cache": None,
        "run_offline_orb": False,
        "ignore_pose": True,
        "simulate_optimizer_all_fail": False,
        "config": None,
        "tracking_gate_only": False,
        "tracking_fps_candidates": (8.0, 12.0, 16.0),
        "progress_range": None,
        "split": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("candidate_config", (CONFIG, FORMAL_M6_CONFIG))
def test_s13_dispatch_bypasses_shared_pipeline_and_allows_full_scan(
    tmp_path: Path, monkeypatch, candidate_config: Path
) -> None:
    called: dict[str, object] = {}

    def fail_shared(*_args, **_kwargs):
        raise AssertionError("shared production/candidate pipeline was called")

    def fake_s13(**kwargs):
        called.update(kwargs)
        kwargs["output"].mkdir(parents=True)
        return {"schema": "gemini305-video-s13-output-first-report/v1", "panorama": "p0.png"}

    monkeypatch.setattr(video_experiment, "run_video_algorithm", fail_shared)
    import panorama_demo.video_s13_experiment as s13_experiment

    monkeypatch.setattr(s13_experiment, "run_s13_experiment", fake_s13)
    report = video_experiment.run(_args(tmp_path, candidate_config=candidate_config))
    assert report["panorama"] == "p0.png"
    assert called["ignore_pose"] is True
    assert called["algorithm_spec"].algorithm_id.startswith("S013_")


def test_s13_trajectory_source_must_be_explicit_and_unique(tmp_path: Path, monkeypatch) -> None:
    def fake_s13(**kwargs):
        selected = sum((kwargs["trajectory_cache"] is not None, kwargs["reuse_online_trajectory"], kwargs["run_offline_orb"], kwargs["ignore_pose"]))
        if selected != 1:
            raise ValueError("exactly one explicit trajectory source")
        kwargs["output"].mkdir(parents=True)
        return {"panorama": "p0.png"}

    import panorama_demo.video_s13_experiment as s13_experiment

    monkeypatch.setattr(s13_experiment, "run_s13_experiment", fake_s13)
    for args in (
        _args(tmp_path, ignore_pose=False),
        _args(tmp_path, ignore_pose=True, run_offline_orb=True),
    ):
        try:
            video_experiment.run(args)
        except ValueError as exc:
            assert "exactly one" in str(exc)
        else:
            raise AssertionError("invalid S1.3 trajectory selection passed")


@pytest.mark.parametrize(
    ("source", "implementation"),
    (
        (CONFIG, "s013_output_first_progressive_dense_central_slit_preview"),
        (FORMAL_M6_CONFIG, "s013_output_first_progressive_dense_central_slit_m61_v2"),
    ),
)
def test_malformed_s13_claim_fails_closed_instead_of_falling_to_legacy(
    tmp_path: Path, monkeypatch, source: Path, implementation: str
) -> None:
    document = source.read_text(encoding="utf-8").replace(
        f"implementation_id: {implementation}", "implementation_id: wrong_implementation"
    )
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(document, encoding="utf-8")
    (tmp_path / "candidate_manifest.json").write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(video_experiment, "run_video_algorithm", lambda **_kwargs: (_ for _ in ()).throw(AssertionError("legacy fallback")))
    try:
        video_experiment.run(_args(tmp_path, candidate_config=candidate))
    except ValueError:
        pass
    else:
        raise AssertionError("malformed S1.3 claim did not fail closed")
