"""CLI for the isolated S01 output-first vertical-alignment experiment."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .video_s1_config import S1_ALGORITHM_ID, load_s1_config
from .video_s1_diagnostics import (
    curve_plot,
    owner_boundary_overlay,
    owner_map_preview,
    write_csv,
    write_image,
    write_json,
    write_yaml,
)
from .video_s1_layout import S1MotionEdge, build_monotonic_progress
from .video_s1_renderer import VideoS1RenderResult, render_video_s1, serialise_layout
from .video_scan_segment import analyse_video_scan
from .video_session import load_video_session


REPORT_SCHEMA = "gemini305-video-s1-experiment-report/v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated S01 output-first local vertical-alignment experiment"
    )
    parser.add_argument("input", type=Path, help="Strict continuous RGB-D video session")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "video_candidates" / f"{S1_ALGORITHM_ID}.yaml"


def _finite(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_finite(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _percentiles(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return 0.0, 0.0, 0.0
    result = np.percentile(array, [5, 50, 95])
    return float(result[0]), float(result[1]), float(result[2])


def _boundary_step_samples(image: np.ndarray, boundaries: Iterable[int]) -> list[float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gradient = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    samples: list[float] = []
    half_width = 12
    for boundary in boundaries:
        if boundary < half_width or boundary + half_width > image.shape[1]:
            continue
        left = np.mean(np.abs(gradient[:, boundary - half_width : boundary]), axis=1)
        right = np.mean(np.abs(gradient[:, boundary : boundary + half_width]), axis=1)
        threshold = max(0.08, float(np.percentile(np.concatenate((left, right)), 75)))
        left_peaks = np.flatnonzero(
            (left >= threshold)
            & (left >= np.r_[left[0], left[:-1]])
            & (left >= np.r_[left[1:], left[-1]])
        )
        right_peaks = np.flatnonzero(
            (right >= threshold)
            & (right >= np.r_[right[0], right[:-1]])
            & (right >= np.r_[right[1:], right[-1]])
        )
        if not left_peaks.size or not right_peaks.size:
            continue
        for left_row in left_peaks:
            distances = np.abs(right_peaks - left_row)
            nearest_index = int(np.argmin(distances))
            if distances[nearest_index] > 8:
                continue
            right_row = int(right_peaks[nearest_index])
            # Mutual-nearest matching prevents one strong repeated shelf line
            # from explaining several weaker edges on the other side.
            if int(left_peaks[np.argmin(np.abs(left_peaks - right_row))]) != int(left_row):
                continue
            strength_ratio = min(left[left_row], right[right_row]) / max(left[left_row], right[right_row], 1e-9)
            if strength_ratio >= 0.35:
                samples.append(abs(float(right_row - left_row)))
    return samples


def _quality_comparison(result: VideoS1RenderResult) -> dict[str, float | int]:
    boundaries = [item.owner_right_x for item in result.layouts[:-1]]
    before = _boundary_step_samples(result.s0_owner_only, boundaries)
    after = _boundary_step_samples(result.s1_owner_only, boundaries)
    _, before_p50, before_p95 = _percentiles(before)
    _, after_p50, after_p95 = _percentiles(after)
    improvement = 0.0 if before_p95 <= 1e-9 else (before_p95 - after_p95) / before_p95
    before_edges = cv2.Canny(result.s0_owner_only, 60, 150)
    after_edges = cv2.Canny(result.s1_owner_only, 60, 150)
    corridor = np.zeros(before_edges.shape, dtype=bool)
    for boundary in boundaries:
        corridor[:, max(0, boundary - 2) : min(corridor.shape[1], boundary + 3)] = True
    return {
        "edge_step_before_p50_px": before_p50,
        "edge_step_before_p95_px": before_p95,
        "edge_step_after_p50_px": after_p50,
        "edge_step_after_p95_px": after_p95,
        "relative_p95_improvement": float(improvement),
        "double_edge_change": int(np.sum((after_edges > 0) & corridor) - np.sum((before_edges > 0) & corridor)),
        "measurement_sample_count_before": len(before),
        "measurement_sample_count_after": len(after),
    }


def _report(
    input_path: Path,
    output: Path,
    result: VideoS1RenderResult,
    stage_seconds: dict[str, float],
    *,
    frame_count: int,
    maximum_workers: int,
    diagnostic_errors: list[str],
) -> dict[str, object]:
    source_steps = [layout.measured_progress_from_previous_px for layout in result.layouts[1:]]
    step_p5, step_p50, step_p95 = _percentiles(source_steps)
    overlaps = [pair.overlap.overlap_width_p50 for pair in result.pair_results]
    overlap_p5, overlap_p50, overlap_p95 = _percentiles(overlaps)
    applied = np.concatenate([pair.alignment.curve.dy_applied for pair in result.pair_results]) if result.pair_results else np.zeros(0)
    gains = np.concatenate([pair.alignment.curve.gain for pair in result.pair_results]) if result.pair_results else np.zeros(0)
    dy_abs = np.abs(applied)
    statuses = [pair.alignment.pair_status for pair in result.pair_results]
    observations = sum(
        sum(not item.is_missing for item in pair.alignment.selected_candidates)
        for pair in result.pair_results
    )
    missing = sum(
        sum(item.is_missing for item in pair.alignment.selected_candidates)
        for pair in result.pair_results
    )
    def gain_fraction(value: float) -> float:
        return float(np.mean(np.isclose(gains, value))) if gains.size else 0.0
    slow_pairs = sorted(
        ({"pair_index": pair.overlap.pair_index, "seconds": pair.elapsed_seconds} for pair in result.pair_results),
        key=lambda item: item["seconds"], reverse=True,
    )[:10]
    return {
        "schema": REPORT_SCHEMA,
        "algorithm_id": S1_ALGORITHM_ID,
        "input": str(input_path.resolve()),
        "output": str(output.resolve()),
        "source_count": len(result.layouts),
        "pair_count": len(result.pair_results),
        "s0_generated": True,
        "s1_generated": True,
        "complete_panorama_generated": bool(np.any(result.valid_mask)),
        "source_selection": {
            "method": "cumulative_canvas_motion",
            "fixed_every_n_frames": False,
            "source_frame_ids": [layout.frame_id for layout in result.layouts],
            "step_px_p5": step_p5,
            "step_px_p50": step_p50,
            "step_px_p95": step_p95,
            "adjacent_frame_rescue_count": sum(
                right.source_index - left.source_index == 1
                for left, right in zip(result.layouts, result.layouts[1:])
            ),
        },
        "overlap": {
            "width_p5": overlap_p5,
            "width_p50": overlap_p50,
            "width_p95": overlap_p95,
            "pair_below_s1_minimum_count": sum(status == "s1_insufficient_overlap" for status in statuses),
        },
        "s1": {
            "pair_full_correction_count": sum(status == "s1_ok" and np.all(pair.alignment.curve.gain > 0) for status, pair in zip(statuses, result.pair_results)),
            "pair_partial_correction_count": sum(status == "s1_ok" and np.any(pair.alignment.curve.gain == 0) and np.any(pair.alignment.curve.gain > 0) for status, pair in zip(statuses, result.pair_results)),
            "pair_zero_correction_count": sum(not np.any(pair.alignment.curve.dy_applied) for pair in result.pair_results),
            "window_observation_count": observations,
            "window_missing_count": missing,
            "gain_1_fraction": gain_fraction(1.0),
            "gain_05_fraction": gain_fraction(0.5),
            "gain_025_fraction": gain_fraction(0.25),
            "gain_0_fraction": gain_fraction(0.0),
            "dy_abs_p50_px": float(np.percentile(dy_abs, 50)) if dy_abs.size else 0.0,
            "dy_abs_p95_px": float(np.percentile(dy_abs, 95)) if dy_abs.size else 0.0,
        },
        "quality_comparison": _quality_comparison(result),
        "performance": {
            "stage_seconds": stage_seconds,
            "raw_frame_count": frame_count,
            "source_count": len(result.layouts),
            "pair_count": len(result.pair_results),
            "slowest_pairs": slow_pairs,
            "full_resolution_remap_invocations": result.remap_invocations,
            "each_source_remapped_once": result.remap_invocations == len(result.layouts),
            "cpu_workers": maximum_workers,
            "gpu_used_only_for_remap": False,
        },
        "diagnostic_errors": diagnostic_errors,
        "fatal_quality_gate": False,
    }


def _pair_diagnostics(output: Path, result: VideoS1RenderResult) -> list[str]:
    errors: list[str] = []
    for pair in result.pair_results:
        pair_dir = output / "pair_diagnostics" / f"pair_{pair.overlap.pair_index:04d}"
        try:
            alignment = pair.alignment
            report = {
                "pair_index": pair.overlap.pair_index,
                "left_frame_id": pair.overlap.left_frame_id,
                "right_frame_id": pair.overlap.right_frame_id,
                "pair_status": alignment.pair_status,
                "overlap": {
                    key: _finite(value)
                    for key, value in asdict(pair.overlap).items()
                    if key not in {"common_valid_mask", "per_row_valid_left_x", "per_row_valid_right_x"}
                },
                "diagnostics": _finite(dict(alignment.diagnostics)),
                "ransac": _finite(asdict(alignment.ransac_prior)),
                "gain_blocks": list(alignment.gain_block_values),
                "elapsed_seconds": pair.elapsed_seconds,
            }
            write_json(pair_dir / "report.json", report)
            candidate_rows = []
            for window in alignment.candidates_by_window:
                candidate_rows.extend(_finite(asdict(candidate)) for candidate in window)
            fields = list(asdict(alignment.candidates_by_window[0][0]).keys()) if alignment.candidates_by_window else ["window_index", "center_y", "dy_px", "is_missing"]
            write_csv(pair_dir / "topk_candidates.csv", candidate_rows, fields)
            curve = alignment.curve
            write_csv(
                pair_dir / "dy_curve.csv",
                ({"y": y, "dy_raw": raw, "dy_smoothed": smooth, "confidence": confidence, "gain": gain, "dy_applied": applied}
                 for y, raw, smooth, confidence, gain, applied in zip(curve.y, curve.dy_raw, curve.dy_smoothed, curve.confidence, curve.gain, curve.dy_applied)),
                ["y", "dy_raw", "dy_smoothed", "confidence", "gain", "dy_applied"],
            )
            write_csv(
                pair_dir / "dy_raw.csv",
                ({"window_index": index, "center_y": item.center_y, "dy_px": item.dy_px, "confidence": item.confidence, "missing": item.is_missing}
                 for index, item in enumerate(alignment.selected_candidates)),
                ["window_index", "center_y", "dy_px", "confidence", "missing"],
            )
            write_image(pair_dir / "dy_curve_overlay.png", curve_plot(result.s1_panorama.shape[0], curve.y, curve.dy_raw, curve.dy_smoothed))
            confidence = np.clip(curve.confidence[:, None] * 255.0, 0, 255).astype(np.uint8)
            gain = np.clip(curve.gain[:, None] * 255.0, 0, 255).astype(np.uint8)
            write_image(pair_dir / "confidence.png", np.repeat(confidence, 64, axis=1))
            write_image(pair_dir / "gain.png", np.repeat(gain, 64, axis=1))
            boundary = pair.overlap.owner_boundary_x
            x0, x1 = max(0, boundary - 96), min(result.s0_panorama.shape[1], boundary + 96)
            before = result.s0_panorama[:, x0:x1]
            after = result.s1_panorama[:, x0:x1]
            write_image(pair_dir / "overlap_left.png", before)
            write_image(pair_dir / "overlap_right.png", after)
            write_image(pair_dir / "before_boundary_crop.png", before)
            write_image(pair_dir / "after_boundary_crop.png", after)
            write_image(pair_dir / "candidate_cost.png", curve_plot(result.s1_panorama.shape[0], curve.y, curve.dy_raw, curve.dy_applied))
            before_abs = np.abs(curve.dy_smoothed[curve.valid_observation_mask])
            after_abs = np.abs((curve.dy_smoothed - curve.dy_applied)[curve.valid_observation_mask])
            write_json(pair_dir / "edge_residual_comparison.json", {
                "before_p95_px": float(np.percentile(before_abs, 95)) if before_abs.size else 0.0,
                "after_p95_px": float(np.percentile(after_abs, 95)) if after_abs.size else 0.0,
            })
        except (OSError, ValueError, cv2.error) as exc:
            errors.append(f"pair_{pair.overlap.pair_index:04d}: {exc}")
    return errors


def run(input_path: Path, config_path: Path, output: Path) -> dict[str, object]:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    config = load_s1_config(config_path)
    stage_seconds: dict[str, float] = {}
    stage_started = time.perf_counter()
    session = load_video_session(
        input_path, validate_frame_files=True,
        validation_workers=config.scan.maximum_workers,
    )
    stage_seconds["session_load"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    qualities, motions, segment = analyse_video_scan(
        session.rgbd.frames,
        analysis_width=config.scan.analysis_width,
        motion_backend="feature",
        workers=config.scan.maximum_workers,
    )
    stage_seconds["scan_analysis"] = time.perf_counter() - stage_started
    start_index = int(segment["start_index"])
    end_index = int(segment["end_index"])
    frames = session.rgbd.frames[start_index : end_index + 1]
    segment_qualities = qualities[start_index : end_index + 1]
    segment_motions = motions[start_index:end_index]
    result = render_video_s1(
        frames, session.rgbd.calibration, segment_motions, segment_qualities,
        scan_direction=int(segment["scan_direction"]),
        analysis_width=config.scan.analysis_width, config=config,
    )
    stage_seconds.update(result.stage_seconds)

    # These six files are the non-optional image result.  Diagnostics remain
    # best effort and are deliberately exported afterwards.
    write_image(output / "s0_panorama_owner_only.png", result.s0_owner_only)
    write_image(output / "s0_panorama.png", result.s0_panorama)
    write_image(output / "s1_panorama_owner_only.png", result.s1_owner_only)
    write_image(output / "s1_panorama.png", result.s1_panorama)
    write_image(output / "owner_map.png", owner_map_preview(result.owner_map))
    write_image(output / "valid_mask.png", result.valid_mask.astype(np.uint8) * 255)

    diagnostic_started = time.perf_counter()
    diagnostic_errors = _pair_diagnostics(output, result) if config.output.write_pair_diagnostics else []
    try:
        write_json(output / "source_layout.json", {"sources": [serialise_layout(item) for item in result.layouts]})
        write_csv(
            output / "source_selection.csv",
            ({"source_index": item.source_index, "frame_id": item.frame_id, "canvas_center_x": item.canvas_center_x,
              "measured_progress_from_previous_px": item.measured_progress_from_previous_px} for item in result.layouts),
            ["source_index", "frame_id", "canvas_center_x", "measured_progress_from_previous_px"],
        )
        write_csv(
            output / "overlap_summary.csv",
            ({"pair_index": pair.overlap.pair_index, "left_frame_id": pair.overlap.left_frame_id,
              "right_frame_id": pair.overlap.right_frame_id, "width_p5": pair.overlap.overlap_width_p5,
              "width_p50": pair.overlap.overlap_width_p50, "width_p95": pair.overlap.overlap_width_p95,
              "s1_evaluable_fraction": pair.overlap.s1_evaluable_fraction} for pair in result.pair_results),
            ["pair_index", "left_frame_id", "right_frame_id", "width_p5", "width_p50", "width_p95", "s1_evaluable_fraction"],
        )
        write_csv(
            output / "pair_summary.csv",
            ({"pair_index": pair.overlap.pair_index, "status": pair.alignment.pair_status,
              "elapsed_seconds": pair.elapsed_seconds, "maximum_abs_dy_px": float(np.max(np.abs(pair.alignment.curve.dy_applied)))} for pair in result.pair_results),
            ["pair_index", "status", "elapsed_seconds", "maximum_abs_dy_px"],
        )
        write_yaml(output / "config_snapshot.yaml", config.as_dict())
        write_image(output / "debug" / "owner_boundary_overlay.png", owner_boundary_overlay(result.s1_panorama, [item.owner_right_x for item in result.layouts[:-1]]))
        progress_edges = [S1MotionEdge(float(item.dx), reliable=bool(item.reliable)) for item in segment_motions]
        progress = build_monotonic_progress(progress_edges, int(segment["scan_direction"]))
        write_csv(output / "debug" / "cumulative_motion.csv", ({"frame_offset": index, "progress_px": value} for index, value in enumerate(progress)), ["frame_offset", "progress_px"])
        centres_preview = owner_boundary_overlay(result.s1_panorama, [int(round(item.canvas_center_x)) for item in result.layouts])
        write_image(output / "debug" / "source_centres.png", centres_preview)
    except (OSError, ValueError, cv2.error) as exc:
        diagnostic_errors.append(f"global diagnostics: {exc}")
    stage_seconds["diagnostic_export"] = time.perf_counter() - diagnostic_started
    stage_seconds["total"] = time.perf_counter() - started
    report = _report(
        input_path, output, result, stage_seconds,
        frame_count=len(frames), maximum_workers=config.scan.maximum_workers,
        diagnostic_errors=diagnostic_errors,
    )
    write_json(output / "performance.json", report["performance"])
    write_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run(args.input, args.config or _default_config(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
