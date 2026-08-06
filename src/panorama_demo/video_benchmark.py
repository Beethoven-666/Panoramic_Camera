"""Repeatable, evidence-first benchmark driver for locked video experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .video_dataset_lock import verify_dataset_lock
from .video_experiment import run as run_experiment
from .video_offline_evaluation import evaluate_delivery_artifacts, write_offline_evaluation
from .video_runtime_environment import (
    atomic_write_json,
    capture_runtime_environment,
    deterministic_result_payload,
    write_deterministic_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a locked video algorithm")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument(
        "--progress-range", metavar=("START", "END"), type=float, nargs=2,
        help="Required immutable development/validation interval for candidates.",
    )
    parser.add_argument("--split", choices=("development", "validation"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--config", type=Path)
    return parser


def _benchmark_root(output: Path, *, session_root: Path) -> Path:
    """Locate the fixed benchmark root rather than a nested experiment dir."""

    expected_name = session_root.name
    for parent in (output, *output.parents):
        if parent.name == expected_name and parent.parent.name == "benchmarks":
            return parent
    return Path("benchmarks") / expected_name


def _all_grades_a(report: dict[str, Any]) -> bool:
    grades = report.get("grades")
    return isinstance(grades, dict) and all(
        grades.get(name) == "A" for name in ("structural", "visual", "performance", "overall")
    )


def _write_leaderboard(path: Path, summary: dict[str, Any]) -> None:
    """Idempotently update the leaderboard keyed by immutable result evidence."""

    fieldnames = (
        "algorithm_id", "config_sha256", "run_count", "warm_median_seconds",
        "warm_max_seconds", "gate_status", "result_sha256",
    )
    rows: list[dict[str, str]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    algorithm = dict(summary.get("algorithm", {}))
    row = {
        "algorithm_id": str(algorithm.get("algorithm_id", "")),
        "config_sha256": str(algorithm.get("config_sha256", "")),
        "run_count": str(summary["run_count"]),
        "warm_median_seconds": str(summary["warm_median_seconds"]),
        "warm_max_seconds": str(summary["warm_max_seconds"]),
        "gate_status": str(summary["gate_status"]),
        "result_sha256": str(summary["result_sha256"]),
    }
    key = (row["algorithm_id"], row["config_sha256"])
    rows = [old for old in rows if (old.get("algorithm_id"), old.get("config_sha256")) != key]
    rows.append(row)
    rows.sort(key=lambda value: (value.get("algorithm_id", ""), value.get("config_sha256", "")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_visual_metrics_if_annotated(output: Path, benchmark_root: Path) -> dict[str, object] | None:
    """Measure published artifacts only; sidecars can never promote a grade."""

    annotations = benchmark_root / "annotations" / "objects.json"
    if not annotations.is_file():
        return None
    before = deterministic_result_payload(output, json.loads((output / "video_report.json").read_text(encoding="utf-8")))
    projection = output / "video_annotation_projection.json"
    evaluation = evaluate_delivery_artifacts(
        output, annotations_path=annotations, projection_path=projection if projection.is_file() else None
    )
    if evaluation.get("automatic_grade_promotion_allowed") is not False:
        raise RuntimeError("Offline visual evaluation must remain measurement-only")
    sidecar = write_offline_evaluation(output / "visual_metrics.json", evaluation)
    after = deterministic_result_payload(output, json.loads((output / "video_report.json").read_text(encoding="utf-8")))
    if after["primary_artifacts"] != before["primary_artifacts"]:
        raise RuntimeError("Offline visual evaluation modified a primary video delivery artifact")
    hard_gate_pass = (
        evaluation.get("schema") == "gemini305-video-offline-visual-evaluation/v1"
        and evaluation.get("measurement_only") is True
        and evaluation.get("projection_available") is True
        and all(
            isinstance(entries := evaluation.get(group), dict)
            and bool(entries)
            and all(
                isinstance(entry, dict)
                and entry.get("status") == "evaluated"
                and entry.get("hard_gate_pass") is True
                for entry in entries.values()
            )
            for group in ("object_integrity", "line_continuity", "safe_background")
        )
    )
    return {
        "path": sidecar.name,
        "measurement_only": True,
        "automatic_grade_promotion_allowed": False,
        # This never upgrades a renderer grade.  It is a fail-closed
        # eligibility condition for a candidate benchmark: a missing,
        # unevaluable, or failed fixed validation measurement cannot appear
        # selectable merely because the renderer's internal objective grade
        # was A.
        "hard_gate_pass": hard_gate_pass,
        "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    if args.algorithm == "candidate" and args.candidate_config is None:
        raise ValueError("candidate benchmark requires --candidate-config")
    if args.algorithm == "baseline" and args.candidate_config is not None:
        raise ValueError("baseline benchmark does not accept --candidate-config")
    progress_range = getattr(args, "progress_range", None)
    split = getattr(args, "split", None)
    if (progress_range is None) != (split is None):
        raise ValueError("--progress-range and --split must be provided together")
    if args.algorithm == "candidate" and progress_range is None:
        raise ValueError("candidate benchmark requires immutable --split and --progress-range")
    session = args.session.expanduser().resolve()
    root = session if session.is_dir() else session.parent
    benchmark_root = _benchmark_root(args.output.expanduser().resolve(), session_root=root)
    verify_dataset_lock(root, benchmark_root / "dataset_lock.json")
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output / "environment.json", capture_runtime_environment())
    runs: list[dict[str, Any]] = []
    for index in range(args.repeat):
        output = args.output.expanduser().resolve() / f"run_{index + 1:02d}"
        experiment_args = argparse.Namespace(
            input=session,
            output=output,
            algorithm=args.algorithm,
            candidate_config=args.candidate_config,
            report_level="summary",
            artifact_level="minimal",
            maximum_post_seconds=None,
            defer_3d=True,
            config=args.config,
            progress_range=progress_range,
            split=split,
        )
        report = run_experiment(experiment_args)
        deterministic = write_deterministic_result(output, report)
        visual_metrics = _write_visual_metrics_if_annotated(output, benchmark_root)
        performance = dict(report.get("performance", {}))
        runs.append(
            {
                "run": index + 1,
                "run_kind": "cold" if index == 0 else "warm",
                "overall_grade": dict(report.get("grades", {})).get("overall"),
                "post_capture_seconds": performance.get("post_capture_seconds"),
                "stage_seconds": dict(performance.get("stage_seconds", {})),
                "algorithm_id": dict(report.get("algorithm", {})).get("algorithm_id"),
                "config_sha256": dict(report.get("algorithm", {})).get("config_sha256"),
                "result_sha256": deterministic["result_sha256"],
                "visual_metrics": visual_metrics,
                "output": str(output),
            }
        )
    warm_seconds = [
        float(row["post_capture_seconds"])
        for row in runs[1:]
        if isinstance(row["post_capture_seconds"], (int, float))
    ]
    seconds = [float(row["post_capture_seconds"]) for row in runs if isinstance(row["post_capture_seconds"], (int, float))]
    first_report_path = Path(str(runs[0]["output"])) / "video_report.json"
    first_report = json.loads(first_report_path.read_text(encoding="utf-8"))
    aggregate_core: dict[str, object] = {
        "schema": "gemini305-video-benchmark-result/v1",
        "algorithm": dict(first_report.get("algorithm", {})),
        "evaluation_scope": first_report.get("evaluation_scope"),
        "run_result_sha256": [str(row["result_sha256"]) for row in runs],
        "visual_metrics": [row["visual_metrics"] for row in runs],
        "run_count": len(runs),
    }
    aggregate_core["result_sha256"] = hashlib.sha256(
        json.dumps(aggregate_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    renderer_grades_a = _all_grades_a(first_report) and all(row["overall_grade"] == "A" for row in runs)
    measurement_gates_pass = (
        args.algorithm != "candidate"
        or all(
            isinstance(row["visual_metrics"], dict)
            and row["visual_metrics"].get("hard_gate_pass") is True
            for row in runs
        )
    )
    all_a = renderer_grades_a and measurement_gates_pass
    summary: dict[str, Any] = {
        **aggregate_core,
        "runs": runs,
        "run_count": len(runs),
        "cold_seconds": seconds[0] if seconds else None,
        "warm_median_seconds": statistics.median(warm_seconds) if warm_seconds else None,
        "warm_max_seconds": max(warm_seconds) if warm_seconds else None,
        "gate_status": "passed" if all_a else "failed",
        "eligible_for_selection": bool(all_a),
    }
    atomic_write_json(args.output / "result.json", aggregate_core)
    atomic_write_json(args.output / "performance.json", {
        "schema": "gemini305-video-benchmark-performance/v1",
        "claim": "fixed-run benchmark evidence",
        "cold_seconds": summary["cold_seconds"],
        "warm_median_seconds": summary["warm_median_seconds"],
        "warm_max_seconds": summary["warm_max_seconds"],
        "runs": runs,
        "gate_status": summary["gate_status"],
        "renderer_grades_a": renderer_grades_a,
        "fixed_validation_measurement_gates_pass": measurement_gates_pass,
    })
    atomic_write_json(args.output / "benchmark.json", summary)
    leaderboard = benchmark_root / "leaderboard.csv"
    _write_leaderboard(leaderboard, summary)
    return summary


def main() -> None:
    args = _parser().parse_args()
    try:
        summary = run(args)
    except Exception as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(summary, indent=2))
