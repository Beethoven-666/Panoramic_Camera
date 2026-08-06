from __future__ import annotations

import json
from pathlib import Path

import pytest

from panorama_demo.video_performance_projection import (
    FixedRunPerformanceError,
    fixed_run_performance_projection,
    run_final_fixed_run_performance_evidence,
    run_fixed_prefix_measurements,
    write_fixed_run_performance_projection,
)


def test_projection_uses_maximum_incremental_slope_and_is_not_a_real_20m_claim():
    result = fixed_run_performance_projection({0.25: 2.0, 0.50: 3.0, 0.75: 4.0, 1.0: 5.0})
    assert result["claim"] == "fixed-run performance projection"
    assert result["does_not_prove_real_20m"] is True
    assert result["fit"]["maximum_incremental_slope_seconds_per_m"] == pytest.approx(4.0 / 3.0)
    assert result["status"] == "passed"


def test_projection_fails_closed_when_conservative_gate_is_missed(tmp_path):
    result = write_fixed_run_performance_projection(
        tmp_path / "scaling_projection.json", {0.25: 15.0, 0.50: 25.0, 0.75: 40.0, 1.0: 60.0}
    )
    assert result["status"] == "failed"
    assert result["gates"]["conservative_20m_le_55s"] is False
    assert json.loads((tmp_path / "scaling_projection.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_projection_requires_the_fixed_prefix_set():
    with pytest.raises(ValueError, match="exactly"):
        fixed_run_performance_projection({0.25: 1.0, 0.50: 2.0})


def _write_measurement_report(output: Path, *, seconds: float, performance_grade: str = "A") -> None:
    output.mkdir(parents=True)
    (output / "video_report.json").write_text(
        json.dumps(
            {
                "algorithm": {
                    "role": "production",
                    "algorithm_id": "production_v1",
                    "implementation_id": "video_visual_renderer_v2",
                    "config_sha256": "a" * 64,
                    "source_commit": "b" * 40,
                    "model_sha256": {"raft": "c" * 64},
                },
                "performance": {"post_capture_seconds": seconds},
                "grades": {"performance": performance_grade},
            }
        ),
        encoding="utf-8",
    )


def test_fixed_prefix_runner_executes_exactly_three_of_each_and_uses_worst_repeat(tmp_path):
    requests = []

    def runner(request):
        requests.append(request)
        # Repeat three is deliberately slower and must be used for the
        # extrapolation rather than an optimistic median.
        _write_measurement_report(
            request.output,
            seconds={0.25: 1.0, 0.50: 2.0, 0.75: 3.0, 1.0: 4.0}[request.progress] + 0.1 * request.repeat,
        )

    result = run_fixed_prefix_measurements(runner, output=tmp_path / "prefix")

    assert len(requests) == 12
    assert {(request.label, request.repeat) for request in requests} == {
        (label, repeat) for label in ("P25", "P50", "P75", "P100") for repeat in (1, 2, 3)
    }
    p25 = next(item for item in result["prefix_summary"] if item["label"] == "P25")
    assert p25["projection_seconds"] == pytest.approx(1.3)
    assert result["does_not_run_holdout"] is True
    assert result["does_not_mutate_dataset"] is True
    assert result["does_not_create_production_lock"] is True
    assert (tmp_path / "prefix" / "fixed_prefix_measurements.json").is_file()
    assert (tmp_path / "prefix" / "scaling_projection.json").is_file()


def test_final_schedule_includes_one_cold_five_warm_and_all_prefix_repeats(tmp_path):
    requests = []

    def runner(request):
        requests.append(request)
        seconds = 4.0 if request.run_kind in {"cold", "warm"} else request.progress * 4.0
        _write_measurement_report(request.output, seconds=seconds)

    result = run_final_fixed_run_performance_evidence(runner, output=tmp_path / "final")

    assert len(requests) == 18
    assert [(request.run_kind, request.repeat) for request in requests[:6]] == [
        ("cold", 1),
        ("warm", 1),
        ("warm", 2),
        ("warm", 3),
        ("warm", 4),
        ("warm", 5),
    ]
    assert len([request for request in requests if request.run_kind == "prefix" and request.label == "P100"]) == 3
    assert result["gates"]["exact_cold_repeats"] is True
    assert result["gates"]["exact_warm_repeats"] is True
    assert result["gates"]["exact_prefix_repeats"] is True
    assert result["gates"]["performance_projection_passed"] is True
    assert result["status"] == "passed"
    assert (tmp_path / "final" / "final_performance_evidence.json").is_file()


def test_fixed_prefix_runner_rejects_non_a_performance_and_existing_evidence(tmp_path):
    with pytest.raises(FixedRunPerformanceError, match="performance grade"):
        run_fixed_prefix_measurements(
            lambda request: _write_measurement_report(request.output, seconds=1.0, performance_grade="C"),
            output=tmp_path / "bad",
        )

    output = tmp_path / "existing"
    output.mkdir()
    (output / "fixed_prefix_measurements.json").write_text("{}", encoding="utf-8")
    called = False

    def never_called(request):
        nonlocal called
        called = True

    with pytest.raises(FixedRunPerformanceError, match="overwrite"):
        run_fixed_prefix_measurements(never_called, output=output)
    assert called is False
