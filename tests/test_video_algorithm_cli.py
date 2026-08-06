from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from panorama_demo.video_experiment import run
from panorama_demo.video_pipeline import production_parser


def test_public_video_cli_does_not_advertise_retired_presets():
    help_text = production_parser().format_help()
    assert "--preset" not in help_text
    assert "--algorithm" not in help_text


def test_experiment_cli_supports_only_explicit_verified_trajectory_cache():
    from panorama_demo.video_experiment import _parser

    help_text = _parser().format_help()
    assert "--trajectory-cache" in help_text
    assert "--reuse-online-trajectory" in help_text


def test_experiment_rejects_online_trajectory_reuse_for_baseline(tmp_path):
    args = argparse.Namespace(
        input=Path("data/captures/video/run_20260804_162340"), output=tmp_path,
        algorithm="baseline", candidate_config=None,
        report_level="summary", artifact_level="minimal", maximum_post_seconds=None,
        defer_3d=True, reuse_online_trajectory=True, config=None,
        progress_range=None, split=None,
    )
    with pytest.raises(ValueError, match="candidate-only"):
        run(args)


def test_experiment_requires_a_candidate_config(tmp_path):
    args = argparse.Namespace(
        input=Path("data/captures/video/run_20260804_162340"),
        output=tmp_path,
        algorithm="candidate",
        candidate_config=None,
        report_level="summary",
        artifact_level="minimal",
        maximum_post_seconds=None,
        defer_3d=True,
        config=None,
        progress_range=None,
        split=None,
    )
    with pytest.raises(ValueError, match="candidate requires"):
        run(args)


def test_audit_artifacts_require_full_report():
    from panorama_demo.video_observability import ObservabilitySpec

    with pytest.raises(ValueError, match="requires report_level=full"):
        ObservabilitySpec.from_values(report_level="summary", artifact_level="audit")


def test_experiment_cannot_label_an_arbitrary_or_holdout_range_as_validation(tmp_path):
    args = argparse.Namespace(
        input=Path("data/captures/video/run_20260804_162340"), output=tmp_path,
        algorithm="candidate", candidate_config=Path("configs/video_candidates/C1_constrained_owner.yaml"),
        report_level="summary", artifact_level="minimal", maximum_post_seconds=None,
        defer_3d=True, config=None, progress_range=(0.84, 1.0), split="validation",
    )
    with pytest.raises(ValueError, match="immutable interval"):
        run(args)


def test_candidate_experiment_cannot_run_an_exploratory_full_scan(tmp_path):
    args = argparse.Namespace(
        input=Path("data/captures/video/run_20260804_162340"), output=tmp_path,
        algorithm="candidate", candidate_config=Path("configs/video_candidates/C1_constrained_owner.yaml"),
        report_level="summary", artifact_level="minimal", maximum_post_seconds=None,
        defer_3d=True, config=None, progress_range=None, split=None,
    )
    with pytest.raises(ValueError, match="require an immutable"):
        run(args)
