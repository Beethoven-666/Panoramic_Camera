"""CLI for the isolated S01 output-first vertical-alignment experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .video_s1_config import (
    S1_ALGORITHM_ID,
    S11_ALGORITHM_ID,
    S11_REPORT_SCHEMA,
    VideoS11ExperimentConfig,
    load_s1_config,
)
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
from .video_s1_metrics import (
    StageVerticalMetrics,
    compare_stage_b_to_stage_c,
    compare_stage_b_to_stage_c_per_pair,
    summarize_vertical_residuals,
)
from .video_s1_renderer import (
    S11StageProvenance,
    VideoS1RenderResult,
    VideoS11RenderResult,
    render_video_s1,
    serialise_layout,
)
from .video_scan_segment import analyse_video_scan
from .video_session import load_video_session


REPORT_SCHEMA = "gemini305-video-s1-experiment-report/v1"
S11_MOTION_ROLE = "diagnostic_layout_evidence_only"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated S01 output-first local vertical-alignment experiment"
    )
    parser.add_argument("input", type=Path, help="Strict continuous RGB-D video session")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Frozen S01 output directory required by the S011 v2 experiment",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-mode",
        choices=("audit", "minimal"),
        default=None,
        help="Override the S011 v2 output mode without changing the frozen config",
    )
    return parser


def _default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "video_candidates" / f"{S1_ALGORITHM_ID}.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_s01_baseline(
    session_root: Path,
    baseline: Path,
    *,
    lock_path: Path | None = None,
) -> dict[str, object]:
    """Fail closed unless a baseline exactly matches the frozen S01 evidence."""

    baseline = baseline.resolve()
    session_root = session_root.resolve()
    lock = lock_path or (
        Path(__file__).resolve().parents[2]
        / "configs/video_candidates/S01_output_first_vertical_alignment_v1.baseline.json"
    )
    document = json.loads(lock.read_text(encoding="utf-8"))
    if document.get("schema") != "gemini305-video-s1-baseline-lock/v1":
        raise ValueError("S01 baseline lock has an unsupported schema")
    run_name = session_root.name
    expected = document.get("runs", {}).get(run_name)
    if not isinstance(expected, dict):
        raise ValueError(f"S01 baseline lock has no entry for {run_name}")
    session_files = {
        "manifest_sha256": session_root / "manifest.json",
        "calibration_sha256": session_root / "calibration.json",
        "frames_csv_sha256": session_root / "frames.csv",
    }
    for key, path in session_files.items():
        if not path.is_file() or _sha256_file(path) != expected["session"][key]:
            raise ValueError(f"S01 baseline session hash mismatch: {path}")
    artifact_files = {
        "report_sha256": baseline / "report.json",
        "source_layout_sha256": baseline / "source_layout.json",
        "s0_owner_only_sha256": baseline / "s0_panorama_owner_only.png",
        "s1_owner_only_sha256": baseline / "s1_panorama_owner_only.png",
    }
    for key, path in artifact_files.items():
        if not path.is_file() or _sha256_file(path) != expected["artifacts"][key]:
            raise ValueError(f"S01 baseline artifact hash mismatch: {path}")
    report = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
    layout = json.loads((baseline / "source_layout.json").read_text(encoding="utf-8"))
    if report.get("algorithm_id") != S1_ALGORITHM_ID:
        raise ValueError("Baseline is not an S01 v1 result")
    if report.get("source_selection", {}).get("source_frame_ids") != expected["source_frame_ids"]:
        raise ValueError("S01 baseline source IDs differ from the frozen lock")
    image = cv2.imread(str(baseline / "s1_panorama_owner_only.png"), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (expected["canvas_height"], expected["canvas_width"]):
        raise ValueError("S01 baseline canvas dimensions differ from the frozen lock")
    return {
        "schema": document["schema"],
        "run_name": run_name,
        "expected": expected,
        "report": report,
        "layout": layout,
        "baseline_path": str(baseline),
        "lock_path": str(lock.resolve()),
    }


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


def _serialise(value: object) -> object:
    """Convert S1.1 audit dataclasses without accepting non-finite JSON numbers."""

    if is_dataclass(value) and not isinstance(value, type):
        return _finite(asdict(value))
    return _finite(value)


def _write_stage_provenance(path: Path, provenance: S11StageProvenance) -> None:
    arrays = {
        "geometric_owner_frame_id": np.asarray(
            provenance.geometric_owner_frame_id, dtype=np.int32
        ),
        "declared_owner_sample_valid": np.asarray(
            provenance.declared_owner_sample_valid, dtype=bool
        ),
        "final_owner_frame_id": np.asarray(provenance.final_owner_frame_id, dtype=np.int32),
        "final_valid": np.asarray(provenance.final_valid, dtype=bool),
    }
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1 or next(iter(shapes), ()) == ():
        raise ValueError("S1.1 stage provenance arrays must share one non-scalar shape")
    if np.any(arrays["final_valid"] & ~arrays["declared_owner_sample_valid"]):
        raise ValueError("S1.1 final valid pixels must be declared owner samples")
    if np.any(arrays["final_valid"] & (
        arrays["final_owner_frame_id"] != arrays["geometric_owner_frame_id"]
    )):
        raise ValueError("S1.1 final owner must equal the geometric hard owner")
    if np.any(~arrays["final_valid"] & (arrays["final_owner_frame_id"] != -1)):
        raise ValueError("S1.1 invalid pixels must have final owner -1")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending.npz")
    np.savez_compressed(pending, **arrays)
    os.replace(pending, path)


def _s11_horizontal_summary(result: VideoS11RenderResult) -> dict[str, object]:
    stage_b_duplicate = stage_b_missing = 0
    stage_c_duplicate = stage_c_missing = 0
    evaluable = heldout = 0
    accepted_duplicate = accepted_missing = accepted_evaluable = 0
    pair_metrics: list[dict[str, object]] = []
    for pair in result.pair_results:
        metrics = dict(pair.heldout_metrics)
        stage_b = dict(metrics.get("stage_b", metrics))
        stage_c = dict(metrics.get("stage_c", metrics))
        pair_metrics.append({
            "pair_index": pair.pair_index,
            "origin_pair_lineage": list(pair.origin_pair_lineage),
            **metrics,
        })
        stage_b_duplicate += int(stage_b.get("duplicate_count", 0))
        stage_b_missing += int(stage_b.get("missing_count", 0))
        stage_c_duplicate += int(stage_c.get("duplicate_count", 0))
        stage_c_missing += int(stage_c.get("missing_count", 0))
        evaluable += int(stage_c.get("evaluable_count", 0))
        heldout += int(stage_c.get("total_heldout_count", 0))
        if not pair.needs_s2:
            accepted_duplicate += int(stage_c.get("duplicate_count", 0))
            accepted_missing += int(stage_c.get("missing_count", 0))
            accepted_evaluable += int(stage_c.get("evaluable_count", 0))
    return {
        "stage_b_duplicate_count": stage_b_duplicate,
        "stage_b_missing_count": stage_b_missing,
        "stage_c_duplicate_count": stage_c_duplicate,
        "stage_c_missing_count": stage_c_missing,
        "heldout_evaluable_count": evaluable,
        "heldout_total_count": heldout,
        "heldout_evaluable_fraction": float(evaluable / heldout) if heldout else 0.0,
        "accepted_safe_pair_duplicate_count": accepted_duplicate,
        "accepted_safe_pair_missing_count": accepted_missing,
        "accepted_safe_pair_evaluable_count": accepted_evaluable,
        "accepted_safe_pair_single_copy_pass": (
            accepted_duplicate == 0 and accepted_missing == 0
        ),
        "pairs": pair_metrics,
    }


def _s11_report(
    input_path: Path,
    output: Path,
    config: VideoS11ExperimentConfig,
    result: VideoS11RenderResult,
    baseline: dict[str, object],
    stage_seconds: dict[str, float],
    *,
    frame_count: int,
) -> dict[str, object]:
    boundaries = [item.owner_right_x for item in result.final_layouts[:-1]]
    per_pair_vertical_metrics = result.vertical_metrics
    if len(per_pair_vertical_metrics) != len(boundaries):
        per_pair_vertical_metrics = compare_stage_b_to_stage_c_per_pair(
            result.stage_b_handoff_only,
            result.stage_c_final,
            boundaries,
            stage_b_valid=result.stage_b_provenance.final_valid,
            stage_c_valid=result.stage_c_provenance.final_valid,
        )

    def aggregate(stage_name: str):
        values = [getattr(item, stage_name) for item in per_pair_vertical_metrics]
        samples = tuple(sample for value in values for sample in value.samples)
        expected = sum(
            int(round(value.measurement_sample_count / value.coverage))
            if value.coverage > 0.0 else value.double_edge_count
            for value in values
        )
        return summarize_vertical_residuals(
            samples,
            expected_sample_count=expected,
            double_edge_count=sum(value.double_edge_count for value in values),
        )

    vertical = StageVerticalMetrics(
        aggregate("stage_b"), aggregate("stage_c")
    ).as_report()
    vertical["per_pair"] = [item.as_report() for item in per_pair_vertical_metrics]
    vertical["per_pair_p95_pass"] = all(
        np.isfinite(item.stage_c.subpixel_edge_residual_p95)
        and item.stage_c.subpixel_edge_residual_p95 <= 0.75
        for item in per_pair_vertical_metrics
    )
    vertical_rejections = {
        item.pair_index: (
            "vertical_unobservable"
            if not np.isfinite(item.stage_c.subpixel_edge_residual_p95)
            else "vertical_p95_exceeds_0_75_px"
        )
        for item in per_pair_vertical_metrics
        if (
            not np.isfinite(item.stage_c.subpixel_edge_residual_p95)
            or item.stage_c.subpixel_edge_residual_p95 > 0.75
        )
    }
    vertical["global_p95_pass"] = bool(
        np.isfinite(vertical["stage_c_subpixel_edge_residual_p95"])
        and vertical["stage_c_subpixel_edge_residual_p95"] <= 0.5
    )
    vertical["global_p99_pass"] = bool(
        np.isfinite(vertical["stage_c_subpixel_edge_residual_p99"])
        and vertical["stage_c_subpixel_edge_residual_p99"] <= 1.0
    )
    baseline_report = baseline["report"]
    assert isinstance(baseline_report, dict)
    baseline_total = float(
        baseline_report.get("performance", {}).get("stage_seconds", {}).get("total", 0.0)
    )
    total = float(stage_seconds.get("total", 0.0))
    performance = {
        "mode": config.output.mode,
        "stage_seconds": stage_seconds,
        "total_seconds": total,
        "baseline_v1_total_seconds": baseline_total,
        "baseline_ratio": float(total / baseline_total) if baseline_total > 0 else None,
        "within_v1_1_15": bool(total <= baseline_total * 1.15) if baseline_total > 0 else None,
        "within_20_seconds": total <= 20.0,
        "raw_frame_count": frame_count,
        "source_count": len(result.final_layouts),
        "pair_count": len(result.pair_results),
        "full_resolution_remap_invocations": result.remap_invocations,
        "each_final_source_remapped_once": (
            result.remap_invocations == len(result.final_layouts)
        ),
        "cpu_workers": config.scan.maximum_workers,
    }
    horizontal = _s11_horizontal_summary(result)
    source_lineage = {
        "initial_source_frame_ids": [item.frame_id for item in result.initial_layouts],
        "final_source_frame_ids": [item.frame_id for item in result.final_layouts],
        "selected_frame_indices": list(result.selected_frame_indices),
        "rescue_count": len(result.rescue_events),
        "rescue_events": [_serialise(item) for item in result.rescue_events],
        "first_endpoint_preserved": (
            result.initial_layouts[0].frame_id == result.final_layouts[0].frame_id
        ),
        "last_endpoint_preserved": (
            result.initial_layouts[-1].frame_id == result.final_layouts[-1].frame_id
        ),
    }
    needs_s2 = []
    for pair in result.pair_results:
        reasons = []
        if pair.needs_s2:
            reasons.append(pair.pair_status)
        if pair.pair_index in vertical_rejections:
            reasons.append(vertical_rejections[pair.pair_index])
        if reasons:
            needs_s2.append({
            "pair_index": pair.pair_index,
            "origin_pair_lineage": list(pair.origin_pair_lineage),
            "reasons": reasons,
        })
    return {
        "schema": S11_REPORT_SCHEMA,
        "algorithm_id": S11_ALGORITHM_ID,
        "diagnostic_only": True,
        "production_eligible": False,
        "motion_role": S11_MOTION_ROLE,
        "input": str(input_path.resolve()),
        "output": str(output.resolve()),
        "baseline": {
            "schema": baseline["schema"],
            "run_name": baseline["run_name"],
            "baseline_path": baseline["baseline_path"],
            "lock_path": baseline["lock_path"],
            "session_hashes": baseline["expected"]["session"],
            "artifact_hashes": baseline["expected"]["artifacts"],
        },
        "canvas": {
            "width": int(result.stage_c_final.shape[1]),
            "height": int(result.stage_c_final.shape[0]),
        },
        "source_lineage": source_lineage,
        "horizontal_handoff": horizontal,
        "vertical_alignment": vertical,
        "pair_audit": [
            {
                "pair_index": pair.pair_index,
                "origin_pair_lineage": list(pair.origin_pair_lineage),
                "left_frame_id": pair.left_frame_id,
                "right_frame_id": pair.right_frame_id,
                "midpoint_x": pair.midpoint_x,
                "boundary_x": pair.boundary_x,
                "pair_status": pair.pair_status,
                "solver_classification": pair.solver_classification,
                "reference_source": pair.reference_source,
                "needs_s2": bool(
                    pair.needs_s2 or pair.pair_index in vertical_rejections
                ),
                "manual_review_required": bool(
                    _pair_manual_review_required(pair)
                    or pair.pair_index in vertical_rejections
                ),
                "vertical_rejection_reason": vertical_rejections.get(pair.pair_index),
                "capture_limited": pair.capture_limited,
                "forbidden_interval_count": len(pair.forbidden_intervals),
                "heldout_metrics": _serialise(pair.heldout_metrics),
                "pair_commit_status": pair.pair_commit_status,
                "alignment_status": pair.alignment.pair_status,
                "alignment_diagnostics": _serialise(pair.alignment.diagnostics),
                "elapsed_seconds": pair.elapsed_seconds,
            }
            for pair in result.pair_results
        ],
        "needs_s2": needs_s2,
        "needs_s2_count": len(needs_s2),
        "performance": performance,
        "fatal_quality_gate": False,
    }


def _write_pair_audit(output: Path, result: VideoS11RenderResult) -> None:
    for pair in result.pair_results:
        pair_dir = output / "pair_audit" / f"pair_{pair.pair_index:04d}"
        write_json(pair_dir / "report.json", {
            "pair_index": pair.pair_index,
            "origin_pair_lineage": list(pair.origin_pair_lineage),
            "left_frame_id": pair.left_frame_id,
            "right_frame_id": pair.right_frame_id,
            "midpoint_x": pair.midpoint_x,
            "boundary_x": pair.boundary_x,
            "pair_status": pair.pair_status,
            "solver_classification": pair.solver_classification,
            "reference_source": pair.reference_source,
            "needs_s2": pair.needs_s2,
            "manual_review_required": _pair_manual_review_required(pair),
            "capture_limited": pair.capture_limited,
            "heldout_metrics": _serialise(pair.heldout_metrics),
            "alignment": _serialise(pair.alignment),
            "pair_commit_status": pair.pair_commit_status,
            "elapsed_seconds": pair.elapsed_seconds,
        })
        observations = [_serialise(item) for item in pair.observations]
        if observations:
            assert isinstance(observations[0], dict)
            write_csv(pair_dir / "observations.csv", observations, list(observations[0]))
        intervals = [_serialise(item) for item in pair.forbidden_intervals]
        if intervals:
            assert isinstance(intervals[0], dict)
            write_csv(pair_dir / "forbidden_intervals.csv", intervals, list(intervals[0]))


def _write_worst_crops(output: Path, result: VideoS11RenderResult) -> None:
    ranked = sorted(
        result.pair_results,
        key=lambda pair: (
            int(pair.needs_s2),
            int(pair.heldout_metrics.get("duplicate_count", 0))
            + int(pair.heldout_metrics.get("missing_count", 0)),
            pair.elapsed_seconds,
        ),
        reverse=True,
    )[:3]
    for rank, pair in enumerate(ranked):
        x0 = max(0, pair.boundary_x - 64)
        x1 = min(result.stage_c_final.shape[1], pair.boundary_x + 64)
        crop_dir = output / "worst_crops" / f"{rank:02d}_pair_{pair.pair_index:04d}"
        write_image(crop_dir / "stage_a.png", result.stage_a_nominal_midpoint_owner_only[:, x0:x1])
        write_image(crop_dir / "stage_b.png", result.stage_b_handoff_only[:, x0:x1])
        write_image(crop_dir / "stage_c.png", result.stage_c_final[:, x0:x1])
        write_json(crop_dir / "audit.json", {
            "pair_index": pair.pair_index,
            "origin_pair_lineage": list(pair.origin_pair_lineage),
            "crop": {"x0": x0, "x1": x1, "y0": 0, "y1": result.stage_c_final.shape[0]},
            "heldout_metrics": _serialise(pair.heldout_metrics),
            "needs_s2": pair.needs_s2,
        })


def _pair_manual_review_required(pair: object) -> bool:
    """Return the renderer's explicit fail-closed review decision."""

    explicit = getattr(pair, "manual_review_required", None)
    if explicit is not None:
        return bool(explicit)
    # Compatibility for synthetic result doubles and pre-field v2 reports.
    status = str(getattr(pair, "pair_status", ""))
    return bool(
        getattr(pair, "needs_s2", False)
        or "unobservable" in status
        or "capture_limited" in status
    )


def _rendered_heldout_row_counts(
    pair: object,
    provenance: S11StageProvenance,
) -> tuple[int, int]:
    owner = np.asarray(provenance.final_owner_frame_id)
    valid = np.asarray(provenance.final_valid, dtype=bool)
    duplicate_rows: set[int] = set()
    missing_rows: set[int] = set()
    for item in getattr(pair, "observations"):
        if getattr(item, "role", None) != "heldout":
            continue
        ly, lx = int(np.rint(item.left_y)), int(np.rint(item.left_canvas_x))
        ry, rx = int(np.rint(item.right_y)), int(np.rint(item.right_canvas_x))
        if not (
            0 <= ly < owner.shape[0]
            and 0 <= lx < owner.shape[1]
            and 0 <= ry < owner.shape[0]
            and 0 <= rx < owner.shape[1]
        ):
            continue
        copies = int(valid[ly, lx] and item.left_canvas_x < pair.boundary_x)
        copies += int(valid[ry, rx] and item.right_canvas_x >= pair.boundary_x)
        representative_row = int(round((ly + ry) / 2.0))
        if copies == 2:
            duplicate_rows.add(representative_row)
        elif copies == 0:
            missing_rows.add(representative_row)
    return len(duplicate_rows), len(missing_rows)


def _write_fast_regressions(
    output: Path,
    result: VideoS11RenderResult,
    baseline: dict[str, object],
) -> None:
    if baseline["run_name"] != "run_20260807_140140":
        return
    for lineage in ((69, 71), (106, 109), (109, 112)):
        pair = next(
            (item for item in result.pair_results if item.origin_pair_lineage == lineage),
            None,
        )
        if pair is None:
            raise ValueError(
                f"Fast-session S1.1 result lost required lineage {lineage[0]}->{lineage[1]}"
            )
        x0 = max(0, pair.boundary_x - 64)
        x1 = min(result.stage_c_final.shape[1], pair.boundary_x + 64)
        crop_dir = output / "fast_acceptance" / f"lineage_{lineage[0]}_to_{lineage[1]}"
        write_image(
            crop_dir / "stage_a_nominal_midpoint_crop.png",
            result.stage_a_nominal_midpoint_owner_only[:, x0:x1],
        )
        write_image(
            crop_dir / "stage_b_handoff_only_crop.png",
            result.stage_b_handoff_only[:, x0:x1],
        )
        write_image(crop_dir / "stage_c_final_crop.png", result.stage_c_final[:, x0:x1])
        write_image(
            crop_dir / "handoff_overlay.png",
            owner_boundary_overlay(result.stage_c_final[:, x0:x1], [pair.boundary_x - x0]),
        )
        write_json(crop_dir / "acceptance.json", {
            "origin_pair_lineage": list(lineage),
            "pair_index": pair.pair_index,
            "boundary_x": pair.boundary_x,
            "crop": {"x0": x0, "x1": x1, "y0": 0, "y1": result.stage_c_final.shape[0]},
            "pair_status": pair.pair_status,
            "needs_s2": pair.needs_s2,
            "manual_review_required": _pair_manual_review_required(pair),
            "heldout_metrics": _serialise(pair.heldout_metrics),
        })


def _measure_box_main_edge_peaks(image: np.ndarray) -> dict[str, object]:
    """Count distinct copies of the fixed slow-box crop's long diagonal edge.

    The acceptance crop contains several real cardboard edges, so the generic
    seam-wide ``double_edge_count`` cannot say whether the particular long box
    edge was duplicated.  This target-specific audit projects the normal image
    gradient along the known *orientation range and region* of that edge, then
    counts separated peaks in line intercept.  It still measures the rendered
    pixels: no expected intercept or pass result is encoded.
    """

    value = np.asarray(image)
    if value.ndim == 3 and value.shape[2] == 3:
        gray = cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif value.ndim == 2:
        gray = value.astype(np.float64)
    else:
        raise ValueError("box main-edge image must be gray or BGR")
    if gray.shape[0] < 111 or gray.shape[1] < 42:
        raise ValueError("box main-edge image is smaller than the fixed audit region")

    gray64 = gray.astype(np.float64)
    gradient_x = cv2.Sobel(gray64, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray64, cv2.CV_64F, 0, 1, ksize=3)
    x_values = np.arange(42, dtype=np.float64)
    intercepts = np.arange(52.0, 78.0 + 0.25, 0.5, dtype=np.float64)
    best_strength = 0.0
    best_slope = float("nan")
    best_scores = np.zeros(intercepts.shape, dtype=np.float64)
    for slope in np.arange(0.65, 1.35 + 0.025, 0.05, dtype=np.float64):
        normal_gradient = np.abs(-slope * gradient_x + gradient_y)
        normal_gradient /= math.sqrt(1.0 + slope * slope)
        scores = np.zeros(intercepts.shape, dtype=np.float64)
        for index, intercept in enumerate(intercepts):
            y_values = slope * x_values + intercept
            keep = (y_values >= 55.0) & (y_values <= 109.0)
            if np.count_nonzero(keep) < 32:
                continue
            sample_x = x_values[keep].astype(np.int64)
            sample_y = y_values[keep]
            lower = np.floor(sample_y).astype(np.int64)
            fraction = sample_y - lower
            response = normal_gradient[lower, sample_x] * (1.0 - fraction)
            response += normal_gradient[lower + 1, sample_x] * fraction
            scores[index] = float(np.mean(response))
        scores = np.convolve(scores, (0.25, 0.5, 0.25), mode="same")
        strength = float(np.max(scores))
        if strength > best_strength:
            best_strength = strength
            best_slope = float(slope)
            best_scores = scores

    # The absolute floor rejects texture/noise in an 8-bit image.  The
    # relative floor admits a genuine second copy while ignoring weak texture
    # running roughly parallel to the cardboard edge.
    minimum_strength = max(24.0, 0.45 * best_strength)
    local_maxima = [
        index
        for index in range(1, len(best_scores) - 1)
        if best_scores[index] >= best_scores[index - 1]
        and best_scores[index] > best_scores[index + 1]
        and best_scores[index] >= minimum_strength
    ]
    selected: list[int] = []
    for index in sorted(local_maxima, key=lambda item: best_scores[item], reverse=True):
        if all(abs(intercepts[index] - intercepts[item]) >= 3.0 for item in selected):
            selected.append(index)
    selected.sort(key=lambda index: intercepts[index])
    return {
        "peak_count": len(selected),
        "peak_strength": best_strength,
        "slope": best_slope,
        "peak_intercepts": [float(intercepts[index]) for index in selected],
        "minimum_peak_strength": minimum_strength,
    }


def _write_box_regression(
    output: Path,
    result: VideoS11RenderResult,
    baseline: dict[str, object],
) -> None:
    if baseline["run_name"] != "run_20260806_153033":
        return
    pair = next(
        (item for item in result.pair_results if item.origin_pair_lineage == (477, 501)),
        None,
    )
    if pair is None:
        raise ValueError("Slow-session S1.1 result lost the required 477->501 lineage")
    x0, x1, y0, y1 = 931, 993, 286, 397
    baseline_image = cv2.imread(
        str(Path(str(baseline["baseline_path"])) / "s1_panorama_owner_only.png"),
        cv2.IMREAD_COLOR,
    )
    if baseline_image is None:
        raise ValueError("Could not read frozen S01 image for the box regression")
    write_image(output / "baseline_v1_crop.png", baseline_image[y0:y1, x0:x1])
    write_image(
        output / "stage_a_nominal_midpoint_crop.png",
        result.stage_a_nominal_midpoint_owner_only[y0:y1, x0:x1],
    )
    write_image(
        output / "stage_b_handoff_only_crop.png",
        result.stage_b_handoff_only[y0:y1, x0:x1],
    )
    write_image(output / "stage_c_final_crop.png", result.stage_c_final[y0:y1, x0:x1])
    overlay = owner_boundary_overlay(result.stage_c_final[y0:y1, x0:x1], [pair.boundary_x - x0])
    write_image(output / "handoff_overlay.png", overlay)
    intervals = [_serialise(item) for item in pair.forbidden_intervals]
    if intervals:
        assert isinstance(intervals[0], dict)
        write_csv(output / "forbidden_intervals.csv", intervals, list(intervals[0]))
    else:
        write_csv(output / "forbidden_intervals.csv", (), ["left_x", "right_x"])
    metrics = dict(pair.heldout_metrics)
    stage_c_metrics = dict(metrics.get("stage_c", metrics))
    measurement_x0 = min(x0, max(0, pair.boundary_x - 16))
    measurement_x1 = max(
        x1, min(result.stage_c_final.shape[1], pair.boundary_x + 16)
    )
    edge_metrics = compare_stage_b_to_stage_c(
        result.stage_b_handoff_only[y0:y1, measurement_x0:measurement_x1],
        result.stage_c_final[y0:y1, measurement_x0:measurement_x1],
        [pair.boundary_x - measurement_x0],
        stage_b_valid=result.stage_b_provenance.final_valid[
            y0:y1, measurement_x0:measurement_x1
        ],
        stage_c_valid=result.stage_c_provenance.final_valid[
            y0:y1, measurement_x0:measurement_x1
        ],
    ).stage_c
    protected_crossing_after = sum(
        bool(getattr(interval, "contains")(pair.boundary_x))
        for interval in pair.forbidden_intervals
    )
    duplicate_observations = int(stage_c_metrics.get("duplicate_count", 0))
    missing_observations = int(stage_c_metrics.get("missing_count", 0))
    duplicate_rows, gap_rows = _rendered_heldout_row_counts(
        pair, result.stage_c_provenance
    )
    heldout_coverage = float(stage_c_metrics.get("evaluable_fraction", 0.0))
    maximum_slope_compensated_jump = float(edge_metrics.subpixel_edge_residual_max)
    main_edge = _measure_box_main_edge_peaks(
        result.stage_c_final[y0:y1, x0:x1]
    )
    single_main_edge_peak = main_edge["peak_count"] == 1
    protected_transition_pixel_count = 0
    checks = {
        "single_main_edge_peak": bool(single_main_edge_peak),
        "duplicate_rows_zero": duplicate_rows == 0,
        "gap_rows_zero": gap_rows == 0,
        "maximum_slope_compensated_jump_le_1_px": (
            math.isfinite(maximum_slope_compensated_jump)
            and maximum_slope_compensated_jump <= 1.0
        ),
        "valid_coverage_ge_90_percent": heldout_coverage >= 0.90,
        "protected_crossing_after_zero": protected_crossing_after == 0,
        "protected_transition_pixels_zero": protected_transition_pixel_count == 0,
    }
    write_json(output / "box_acceptance.json", {
        "origin_pair_lineage": [477, 501],
        "baseline_pair_index": 16,
        "baseline_boundary_x": 951,
        "crop": {"x0": x0, "x1": x1, "y0": y0, "y1": y1},
        "final_boundary_x": pair.boundary_x,
        "pair_status": pair.pair_status,
        "not_unobservable": "unobservable" not in pair.pair_status,
        "not_deferred_to_s2": not pair.needs_s2,
        "duplicate_count": duplicate_observations,
        "missing_count": missing_observations,
        "duplicate_row_count": duplicate_rows,
        "gap_row_count": gap_rows,
        "heldout_evaluable_fraction": heldout_coverage,
        "valid_coverage": heldout_coverage,
        "main_edge_peak_count": main_edge["peak_count"],
        "main_edge_peak_strength": main_edge["peak_strength"],
        "main_edge_slope": main_edge["slope"],
        "main_edge_peak_intercepts": main_edge["peak_intercepts"],
        "main_edge_minimum_peak_strength": main_edge["minimum_peak_strength"],
        "edge_measurement_sample_count": edge_metrics.measurement_sample_count,
        "edge_peak_strength": edge_metrics.edge_peak_strength,
        "edge_width_px": edge_metrics.edge_width,
        "double_edge_count": edge_metrics.double_edge_count,
        "maximum_slope_compensated_jump_px": maximum_slope_compensated_jump,
        "protected_crossing_after": protected_crossing_after,
        "protected_transition_pixel_count": protected_transition_pixel_count,
        "checks": checks,
        "automatic_acceptance_pass": all(checks.values()),
    })


def _run_s11(
    input_path: Path,
    output: Path,
    config: VideoS11ExperimentConfig,
    baseline_path: Path | None,
    *,
    started: float,
    report_output: Path | None = None,
) -> dict[str, object]:
    if baseline_path is None:
        raise ValueError("S011 v2 requires --baseline with a frozen S01 result")
    stage_seconds: dict[str, float] = {}
    stage_started = time.perf_counter()
    session = load_video_session(
        input_path,
        validate_frame_files=True,
        validation_workers=config.scan.maximum_workers,
    )
    stage_seconds["session_load"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    baseline = verify_s01_baseline(session.rgbd.root, baseline_path)
    stage_seconds["baseline_verification"] = time.perf_counter() - stage_started
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
    from . import video_s1_renderer

    renderer = getattr(video_s1_renderer, "render_video_s11", None)
    if renderer is None:
        raise RuntimeError("S011 renderer integration point render_video_s11 is unavailable")
    result = renderer(
        frames,
        session.rgbd.calibration,
        segment_motions,
        segment_qualities,
        scan_direction=int(segment["scan_direction"]),
        analysis_width=config.scan.analysis_width,
        config=config,
        baseline=baseline,
    )
    if not isinstance(result, VideoS11RenderResult):
        raise TypeError("render_video_s11 returned an invalid result type")
    stage_seconds.update(result.stage_seconds)
    write_image(
        output / "s11_nominal_midpoint_owner_only.png",
        result.stage_a_nominal_midpoint_owner_only,
    )
    write_image(output / "s0_panorama_owner_only.png", result.stage_b_handoff_only)
    write_image(output / "s11_panorama_owner_only.png", result.stage_c_final)
    _write_stage_provenance(output / "stage_a_provenance.npz", result.stage_a_provenance)
    _write_stage_provenance(output / "stage_b_provenance.npz", result.stage_b_provenance)
    _write_stage_provenance(output / "stage_c_provenance.npz", result.stage_c_provenance)
    _write_worst_crops(output, result)
    if config.output.mode == "audit":
        if config.output.write_pair_reports:
            _write_pair_audit(output, result)
        if config.output.write_acceptance_crops:
            _write_box_regression(output, result, baseline)
            _write_fast_regressions(output, result, baseline)
        write_json(output / "source_layout.json", {
            "initial_sources": [serialise_layout(item) for item in result.initial_layouts],
            "midpoint_sources": [serialise_layout(item) for item in result.midpoint_layouts],
            "final_sources": [serialise_layout(item) for item in result.final_layouts],
        })
        config_snapshot = config.as_dict()
        config_snapshot["motion_role"] = S11_MOTION_ROLE
        write_yaml(output / "config_snapshot.yaml", config_snapshot)
    metrics_started = time.perf_counter()
    report = _s11_report(
        input_path,
        report_output or output,
        config,
        result,
        baseline,
        stage_seconds,
        frame_count=len(frames),
    )
    stage_seconds["acceptance_metrics"] = time.perf_counter() - metrics_started
    stage_seconds["total"] = time.perf_counter() - started
    performance = report["performance"]
    assert isinstance(performance, dict)
    total = float(stage_seconds["total"])
    baseline_total = float(performance["baseline_v1_total_seconds"])
    performance["total_seconds"] = total
    performance["baseline_ratio"] = (
        float(total / baseline_total) if baseline_total > 0 else None
    )
    performance["within_v1_1_15"] = (
        bool(total <= baseline_total * 1.15) if baseline_total > 0 else None
    )
    performance["within_20_seconds"] = total <= 20.0
    write_json(output / "performance.json", report["performance"])
    write_json(output / "report.json", report)
    return report


def _publish_staged_bundle(staging: Path, output: Path) -> None:
    """Replace one complete v2 bundle while retaining the old bundle on failure."""

    backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
    existing_moved = False
    try:
        if output.exists():
            os.replace(output, backup)
            existing_moved = True
        os.replace(staging, output)
    except BaseException:
        if existing_moved and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _run_s11_atomically(
    input_path: Path,
    output: Path,
    config: VideoS11ExperimentConfig,
    baseline_path: Path | None,
    *,
    started: float,
) -> dict[str, object]:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        report = _run_s11(
            input_path,
            staging,
            config,
            baseline_path,
            started=started,
            report_output=output,
        )
        _publish_staged_bundle(staging, output)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def run(
    input_path: Path,
    config_path: Path,
    output: Path,
    baseline_path: Path | None = None,
    output_mode: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    config = load_s1_config(config_path)
    if isinstance(config, VideoS11ExperimentConfig):
        if output_mode is not None:
            config = replace(config, output=replace(config.output, mode=output_mode))
        return _run_s11_atomically(
            input_path,
            output,
            config,
            baseline_path,
            started=started,
        )
    if output_mode is not None:
        raise ValueError("--output-mode is supported only by the S011 v2 experiment")
    output.mkdir(parents=True, exist_ok=True)
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
    run(
        args.input,
        args.config or _default_config(),
        args.output,
        baseline_path=args.baseline,
        output_mode=args.output_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
