"""Rendered relative selection of the S1.3 vertical parent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping

import numpy as np

from .session import CameraIntrinsics
from .video_s12_schedule import S012Schedule
from .video_s13_quality import (
    METRIC_NAMES,
    prepare_seam_structure,
    seam_structure_metrics,
    sequence_structure_decision,
    structurally_non_degrading,
)
from .video_s13_vertical import (
    S13P1Result,
    S13VerticalSolution,
    render_s13_p1_from_raw,
    vertical_candidate_solution,
)


@dataclass(frozen=True)
class S13VerticalSelection:
    stage: str
    selected_gain: float
    solution: S13VerticalSolution
    result: S13P1Result
    audit: Mapping[str, object]


def _pair_metrics(image: np.ndarray, schedule: S012Schedule) -> tuple[dict[str, object], ...]:
    features = prepare_seam_structure(image)
    if features is None:
        return tuple(
            seam_structure_metrics(image, np.full(schedule.canvas_height, boundary, dtype=np.int32))
            for boundary in schedule.boundaries[1:-1]
        )
    return tuple(
        seam_structure_metrics(features, np.full(schedule.canvas_height, boundary, dtype=np.int32))
        for boundary in schedule.boundaries[1:-1]
    )


def _mean_score(metrics: tuple[dict[str, object], ...]) -> float | None:
    scores = [float(item["score"]) for item in metrics if item.get("evaluable") is True]
    return None if not scores else float(np.mean(scores))


def _candidate_set_scores(
    p0_metrics: tuple[dict[str, object], ...],
    rendered: Mapping[float, tuple[S13VerticalSolution, S13P1Result, tuple[dict[str, object], ...]]],
) -> dict[str, float | None]:
    candidates = {"P0_identity": p0_metrics, **{str(gain): value[2] for gain, value in rendered.items()}}
    totals: dict[str, list[float]] = {name: [] for name in candidates}
    for pair_index in range(len(p0_metrics)):
        for metric_name in (*METRIC_NAMES, "score"):
            sample: list[tuple[str, float]] = []
            for name, metrics in candidates.items():
                value = metrics[pair_index].get(metric_name)
                if metrics[pair_index].get("evaluable") is True and isinstance(value, (int, float)):
                    sample.append((name, float(value)))
            if len(sample) < 2:
                continue
            values = np.asarray([value for _name, value in sample], dtype=np.float64)
            median = float(np.median(values))
            mad = max(1e-6, float(np.median(np.abs(values - median))) * 1.4826)
            for name, value in sample:
                totals[name].append((value - median) / mad)
    return {name: (float(np.mean(values)) if values else None) for name, values in totals.items()}


def select_s13_vertical_parent(
    schedule: S012Schedule,
    calibration: CameraIntrinsics,
    image_loader: Callable[[int], np.ndarray],
    base_solution: S13VerticalSolution,
    p0_image: np.ndarray,
) -> S13VerticalSelection:
    """Compare P0 and every requested gain using actual rendered structure."""

    p0_metrics = _pair_metrics(p0_image, schedule)
    p0_score = _mean_score(p0_metrics)
    rows: list[dict[str, object]] = []
    rendered: dict[float, tuple[S13VerticalSolution, S13P1Result, tuple[dict[str, object], ...]]] = {}
    gains = tuple(float(value) for value in base_solution.audit.get("gain_candidates", (0.0, 0.25, 0.5, 1.0)))
    for gain in gains:
        global_solution = vertical_candidate_solution(
            base_solution, gain, accepted_local_pairs=tuple(False for _ in base_solution.pairs)
        )
        global_result = render_s13_p1_from_raw(schedule, calibration, image_loader, global_solution)
        global_metrics = _pair_metrics(global_result.image, schedule)
        full_local_solution = vertical_candidate_solution(
            base_solution, gain, accepted_local_pairs=tuple(True for _ in base_solution.pairs)
        )
        full_local_result = render_s13_p1_from_raw(schedule, calibration, image_loader, full_local_solution)
        full_local_metrics = _pair_metrics(full_local_result.image, schedule)
        accepted: list[bool] = []
        pair_rows: list[dict[str, object]] = []
        for pair, before, after in zip(base_solution.pairs, global_metrics, full_local_metrics, strict=True):
            nondegrading, reason = structurally_non_degrading(before, after)
            has_candidate = bool(np.any(base_solution.gain_local_row_residuals[str(gain)][pair.pair_index] != 0.0))
            applied = bool(has_candidate and nondegrading)
            accepted.append(applied)
            pair_rows.append({
                "pair_index": pair.pair_index,
                "global_only_metrics": before,
                "global_plus_local_metrics": after,
                "decision": "applied" if applied else "rolled_back",
                "reason": None if applied else (reason or "zero_local_candidate"),
            })
        selected_solution = vertical_candidate_solution(base_solution, gain, accepted_local_pairs=tuple(accepted))
        selected_result = render_s13_p1_from_raw(schedule, calibration, image_loader, selected_solution)
        selected_metrics = _pair_metrics(selected_result.image, schedule)
        rendered[gain] = (selected_solution, selected_result, selected_metrics)
        rows.append({
            "gain": gain,
            "global_only_score": _mean_score(global_metrics),
            "selected_local_score": _mean_score(selected_metrics),
            "pair_local_decisions": pair_rows,
            "actual_full_resolution_render_compared": True,
        })

    normalized_scores = _candidate_set_scores(p0_metrics, rendered)
    for row in rows:
        row["candidate_set_median_mad_score"] = normalized_scores.get(str(row["gain"]))
    # Identity remains a real candidate.  Local row residuals are still
    # accepted strictly pair-by-pair above.  The sequence parent uses a robust
    # aggregate gate: bounded component outliers cannot veto a clearly better
    # long sequence, while every pair's total score remains protected.
    eligible: list[tuple[float, float]] = []
    for row in rows:
        gain = float(row["gain"])
        _solution, _result, metrics = rendered[gain]
        score = _mean_score(metrics)
        sequence_eligible, sequence_audit = sequence_structure_decision(
            p0_metrics,
            metrics,
            maximum_component_failure_fraction=0.10,
            maximum_pair_score_failure_fraction=0.02,
        )
        row["sequence_eligibility"] = sequence_audit
        if sequence_eligible and score is not None and p0_score is not None:
            eligible.append((gain, score))
    selected_gain = 0.0
    stage = "P0_identity"
    if eligible and p0_score is not None:
        # Gain, then local residuals, are increasing complexity.  Near ties
        # deliberately choose the smaller gain; P0 wins without >=0.5% proof.
        best_gain, _best_score = min(eligible, key=lambda item: (item[1], item[0]))
        best_score = _mean_score(rendered[best_gain][2])
        if best_score is not None and best_score < p0_score * 0.995:
            selected_gain, stage = best_gain, "vertical_parent"
    solution, result, selected_metrics = rendered[selected_gain]
    if stage == "P0_identity":
        solution = vertical_candidate_solution(
            base_solution, 0.0, accepted_local_pairs=tuple(False for _ in base_solution.pairs)
        )
        result = render_s13_p1_from_raw(schedule, calibration, image_loader, solution)
        selected_metrics = _pair_metrics(result.image, schedule)
    return S13VerticalSelection(
        stage=stage,
        selected_gain=float(selected_gain),
        solution=solution,
        result=result,
        audit={
            "schema": "gemini305-video-s13-vertical-selection/v1",
            "identity_always_candidate": True,
            "selection_policy": "rendered_pair_relative_structure_non_degradation_then_simplicity",
            "sequence_parent_policy": (
                "bounded_pair_and_component_outlier_budgets_plus_hard_catastrophic_guards_and_mean_improvement"
            ),
            "p0_metrics": [dict(item) for item in p0_metrics],
            "p0_mean_score": p0_score,
            "candidate_set_median_mad_scores": normalized_scores,
            "gain_candidates": rows,
            "selected_stage": stage,
            "selected_gain": float(selected_gain),
            "selected_metrics": [dict(item) for item in selected_metrics],
            "selected_pairs": [asdict(pair) for pair in solution.pairs],
        },
    )


__all__ = ["S13VerticalSelection", "select_s13_vertical_parent"]
