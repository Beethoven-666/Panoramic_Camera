"""Read-only v6 frozen-dataset candidate matrix evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from .video_runtime_environment import atomic_write_json


REQUIRED_DATASETS = (
    "FAST_PRIMARY_DEVELOPMENT",
    "FAST_PRESSURE_REGRESSION",
    "SLOW_DEVELOPMENT_CONTROL",
)


def _load_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"v6 matrix report must be an object: {path}")
    return payload


def assess_v6_candidate_report(report: Mapping[str, object]) -> dict[str, object]:
    """Assess one already-rendered report without rerendering or mutating it."""

    algorithm = report.get("algorithm")
    renderer = report.get("renderer")
    source_ids = report.get("source_frame_ids")
    edges = report.get("open3d_edges")
    performance = report.get("performance")
    if not isinstance(algorithm, Mapping) or not isinstance(renderer, Mapping):
        raise ValueError("v6 matrix report lacks algorithm or renderer evidence")
    if not isinstance(source_ids, list) or len(source_ids) < 2 or not all(isinstance(value, int) for value in source_ids):
        raise ValueError("v6 matrix report lacks chronological real source IDs")
    if source_ids != sorted(set(source_ids)):
        raise ValueError("v6 matrix source IDs are not unique chronological real sources")
    if not isinstance(edges, list) or not isinstance(performance, Mapping):
        raise ValueError("v6 matrix report lacks Open3D or performance evidence")
    sampling = renderer.get("raw_rgb_once_sampling")
    quality = renderer.get("quality_metrics")
    if not isinstance(sampling, Mapping) or not isinstance(quality, Mapping):
        raise ValueError("v6 matrix renderer lacks sampling or quality evidence")
    failures: list[str] = []
    if not str(algorithm.get("algorithm_id", "")).startswith("V6_rgb_only_graphcut"):
        failures.append("not_v6_graphcut_candidate")
    if sampling.get("exactly_once") is not True or sampling.get("source_frame_ids") != source_ids:
        failures.append("raw_rgb_not_sampled_once_from_final_real_sources")
    if int(sampling.get("source_sampling_call_count", -1)) != len(source_ids):
        failures.append("source_sampling_call_count_mismatch")
    if len(edges) != len(source_ids) - 1:
        failures.append("open3d_edge_count_mismatch")
    if report.get("untracked_motion_analysis_frames_rendered") is not False:
        failures.append("untracked_motion_frame_rendered")
    if quality.get("strict_quality_pass") is not True:
        failures.append("strict_visual_gate_failed")
    seconds = performance.get("primary_post_capture_seconds")
    if not isinstance(seconds, (int, float)) or seconds < 0:
        failures.append("missing_primary_timing")
    elif float(seconds) > 20.0:
        failures.append("candidate_hard_20_second_limit_exceeded")
    return {
        "algorithm_id": algorithm.get("algorithm_id"),
        "source_frame_ids": list(source_ids),
        "source_count": len(source_ids),
        "open3d_edge_count": len(edges),
        "source_sampling_call_count": sampling.get("source_sampling_call_count"),
        "primary_post_capture_seconds": seconds,
        "strict_quality_pass": quality.get("strict_quality_pass"),
        "current_dataset_candidate_pass": not failures,
        "failure_reasons": failures,
    }


def build_v6_development_matrix(reports: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """Build the three-dataset development verdict from frozen run reports."""

    missing = [name for name in REQUIRED_DATASETS if name not in reports]
    if missing:
        raise ValueError(f"v6 development matrix missing datasets: {missing}")
    rows = {name: assess_v6_candidate_report(reports[name]) for name in REQUIRED_DATASETS}
    return {
        "schema": "gemini305-video-v6-development-matrix/v1",
        "required_datasets": list(REQUIRED_DATASETS),
        "datasets": rows,
        "development_matrix_pass": all(
            bool(row["current_dataset_candidate_pass"]) for row in rows.values()
        ),
        "production_3m_pass": False,
        "production_3m_realtime_pass": False,
        "twenty_metre_validated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build read-only v6 frozen-dataset evidence matrix")
    parser.add_argument("--fast-primary", required=True, type=Path)
    parser.add_argument("--fast-pressure", required=True, type=Path)
    parser.add_argument("--slow-control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    matrix = build_v6_development_matrix({
        "FAST_PRIMARY_DEVELOPMENT": _load_report(args.fast_primary),
        "FAST_PRESSURE_REGRESSION": _load_report(args.fast_pressure),
        "SLOW_DEVELOPMENT_CONTROL": _load_report(args.slow_control),
    })
    atomic_write_json(args.output, matrix)
    print(f"V6 development matrix: {args.output}")


if __name__ == "__main__":
    main()

