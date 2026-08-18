"""Centered robust scalar photometric candidates for the M6.3 successor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .video_s13_m63_quality_cut import S13M63PairQuality, classify_s13_m63_pairs
from .video_s13_m63_local_field import evaluate_s13_m63_ql_shadow
from .video_s13_photometric import (
    PHOTOMETRIC_BOUNDS_SCHEMA,
    PHOTOMETRIC_SCHEMA,
    S13PhotometricSampleSet,
    S13PhotometricSolution,
    S13SourcePhotometricParameters,
)


Q1R_MODEL = "Q1R_robust_centered_scalar_gain"
Q4C_MODEL = "Q4c_quality_cut_component_scalar_gain"


@dataclass(frozen=True)
class S13M63Config:
    enabled: bool = False
    irls_iterations: int = 8
    huber_delta: float = 1.5
    identity_regularization: float = 0.02
    first_difference_regularization: float = 0.04
    second_difference_regularization: float = 0.03
    minimum_gain: float = 0.92
    maximum_gain: float = 1.08
    shrink_candidates: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.125)
    soft_absolute_pair_p95_linear: float = 0.035
    hard_absolute_pair_p95_linear: float = 0.060
    soft_population_mad_multiplier: float = 3.0
    hard_population_mad_multiplier: float = 6.0
    soft_minimum_inlier_fraction: float = 0.85
    hard_minimum_inlier_fraction: float = 0.75
    soft_maximum_log_luminance_mad: float = 0.040
    hard_maximum_log_luminance_mad: float = 0.060
    minimum_heldout_samples: int = 192
    minimum_aggregate_median_benefit_fraction: float = 0.005
    maximum_aggregate_p95_regression_fraction: float = 0.01
    maximum_macro_p95_regression_fraction: float = 0.01
    minimum_pair_nonregression_fraction: float = 0.70
    minimum_pair_improvement_fraction: float = 0.60
    maximum_trusted_worst_pair_p95_regression_fraction: float = 0.05
    maximum_trusted_worst_pair_p95_regression_absolute: float = 0.001
    q4c_enabled: bool = True
    q4c_minimum_gain: float = 0.94
    q4c_maximum_gain: float = 1.06
    q4c_minimum_component_median_benefit_fraction: float = 0.02
    q4c_minimum_component_macro_p95_benefit_fraction: float = 0.02
    cut_guard_width_px: int = 12
    ql_enabled: bool = True
    qt_enabled: bool = False

    @classmethod
    def from_document(cls, document: Mapping[str, object] | None) -> "S13M63Config":
        if not document:
            return cls()
        q1r = dict(document.get("q1r", {}))
        quality = dict(document.get("quality_cut", {}))
        selection = dict(document.get("selection", {}))
        q4c = dict(document.get("q4c", {}))
        ql = dict(document.get("ql", {}))
        qt = dict(document.get("qt", {}))
        return cls(
            enabled=True,
            irls_iterations=int(q1r.get("irls_iterations", 8)),
            huber_delta=float(q1r.get("huber_delta", 1.5)),
            identity_regularization=float(q1r.get("identity_regularization", 0.02)),
            first_difference_regularization=float(q1r.get("first_difference_regularization", 0.04)),
            second_difference_regularization=float(q1r.get("second_difference_regularization", 0.03)),
            minimum_gain=float(q1r.get("minimum_gain", 0.92)),
            maximum_gain=float(q1r.get("maximum_gain", 1.08)),
            shrink_candidates=tuple(float(value) for value in q1r.get("shrink_candidates", (1, .75, .5, .25, .125))),
            soft_absolute_pair_p95_linear=float(quality.get("soft_absolute_pair_p95_linear", 0.035)),
            hard_absolute_pair_p95_linear=float(quality.get("hard_absolute_pair_p95_linear", 0.060)),
            soft_population_mad_multiplier=float(quality.get("soft_population_mad_multiplier", 3.0)),
            hard_population_mad_multiplier=float(quality.get("hard_population_mad_multiplier", 6.0)),
            soft_minimum_inlier_fraction=float(quality.get("soft_minimum_inlier_fraction", 0.85)),
            hard_minimum_inlier_fraction=float(quality.get("hard_minimum_inlier_fraction", 0.75)),
            soft_maximum_log_luminance_mad=float(quality.get("soft_maximum_log_luminance_mad", 0.040)),
            hard_maximum_log_luminance_mad=float(quality.get("hard_maximum_log_luminance_mad", 0.060)),
            minimum_heldout_samples=int(quality.get("minimum_heldout_samples", 192)),
            minimum_aggregate_median_benefit_fraction=float(selection.get("minimum_aggregate_median_benefit_fraction", 0.005)),
            maximum_aggregate_p95_regression_fraction=float(selection.get("maximum_aggregate_p95_regression_fraction", 0.01)),
            maximum_macro_p95_regression_fraction=float(selection.get("maximum_macro_p95_regression_fraction", 0.01)),
            minimum_pair_nonregression_fraction=float(selection.get("minimum_pair_nonregression_fraction", 0.70)),
            minimum_pair_improvement_fraction=float(selection.get("minimum_pair_improvement_fraction", 0.60)),
            maximum_trusted_worst_pair_p95_regression_fraction=float(selection.get("maximum_trusted_worst_pair_p95_regression_fraction", 0.05)),
            maximum_trusted_worst_pair_p95_regression_absolute=float(selection.get("maximum_trusted_worst_pair_p95_regression_absolute", 0.001)),
            q4c_enabled=bool(q4c.get("enabled", True)),
            q4c_minimum_gain=float(q4c.get("minimum_gain", 0.94)),
            q4c_maximum_gain=float(q4c.get("maximum_gain", 1.06)),
            q4c_minimum_component_median_benefit_fraction=float(q4c.get("minimum_component_median_benefit_fraction", 0.02)),
            q4c_minimum_component_macro_p95_benefit_fraction=float(q4c.get("minimum_component_macro_p95_benefit_fraction", 0.02)),
            cut_guard_width_px=int(q4c.get("cut_guard_width_px", 12)),
            ql_enabled=bool(ql.get("enabled", True)),
            qt_enabled=bool(qt.get("enabled", False)),
        )


@dataclass(frozen=True)
class S13M63SolveResult:
    solution: S13PhotometricSolution
    quality_rows: tuple[S13M63PairQuality, ...]
    quality_cut_pair_indices: tuple[int, ...]
    soft_downweighted_pair_indices: tuple[int, ...]
    trusted_pair_indices: tuple[int, ...]
    audit: Mapping[str, object]


def _crosses_cut(sample: S13PhotometricSampleSet, cuts: set[int]) -> bool:
    low = min(sample.left_source_index, sample.right_source_index)
    high = max(sample.left_source_index, sample.right_source_index)
    return any(low <= cut < high for cut in cuts)


def _components(source_count: int, samples: Sequence[S13PhotometricSampleSet]) -> list[list[int]]:
    adjacency = [set() for _ in range(source_count)]
    for sample in samples:
        adjacency[sample.left_source_index].add(sample.right_source_index)
        adjacency[sample.right_source_index].add(sample.left_source_index)
    remaining = set(range(source_count))
    result: list[list[int]] = []
    while remaining:
        stack = [min(remaining)]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            component.append(node)
            stack.extend(sorted(adjacency[node], reverse=True))
        result.append(sorted(component))
    return result


def _relations(
    samples: Sequence[S13PhotometricSampleSet],
    soft_pair_indices: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    source_count = 1 + max(
        max(item.left_source_index, item.right_source_index) for item in samples
    )
    for item in samples:
        left = np.maximum(item.train_left_rgb_linear.mean(axis=1), 1e-6)
        right = np.maximum(item.train_right_rgb_linear.mean(axis=1), 1e-6)
        relation = np.log(left) - np.log(right)
        row = np.zeros(source_count, np.float64)
        row[item.left_source_index] = -1.0
        row[item.right_source_index] = 1.0
        rows.append(row)
        targets.append(float(np.median(relation)))
        confidence = 0.25 if item.pair_index in soft_pair_indices else 1.0
        weights.append(confidence * np.sqrt(max(1, len(relation))))
    return np.stack(rows), np.asarray(targets), np.asarray(weights)


def _regularization_rows(source_count: int, config: S13M63Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for index in range(source_count):
        row = np.zeros(source_count, np.float64)
        row[index] = 1.0
        rows.append(row)
        targets.append(0.0)
        weights.append(np.sqrt(config.identity_regularization))
    for index in range(source_count - 1):
        row = np.zeros(source_count, np.float64)
        row[index:index + 2] = (-1.0, 1.0)
        rows.append(row)
        targets.append(0.0)
        weights.append(np.sqrt(config.first_difference_regularization))
    for index in range(source_count - 2):
        row = np.zeros(source_count, np.float64)
        row[index:index + 3] = (1.0, -2.0, 1.0)
        rows.append(row)
        targets.append(0.0)
        weights.append(np.sqrt(config.second_difference_regularization))
    return np.stack(rows), np.asarray(targets), np.asarray(weights)


def solve_s13_m63_centered_logs(
    source_count: int,
    samples: Sequence[S13PhotometricSampleSet],
    config: S13M63Config,
    *,
    soft_pair_indices: frozenset[int] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """Huber IRLS with an exact weighted-zero-mean log-gain gauge."""

    pair_rows, targets, base_weights = _relations(samples, soft_pair_indices)
    if pair_rows.shape[1] != source_count:
        pair_rows = np.pad(pair_rows, ((0, 0), (0, source_count - pair_rows.shape[1])))
    reg_rows, reg_targets, reg_weights = _regularization_rows(source_count, config)
    evidence = np.zeros(source_count, np.float64)
    for item in samples:
        value = float(len(item.train_left_rgb_linear))
        evidence[item.left_source_index] += value
        evidence[item.right_source_index] += value
    gauge = np.maximum(evidence, 1.0)
    gauge /= gauge.sum()
    solved = np.zeros(source_count, np.float64)
    robust_weights = np.ones(len(targets), np.float64)
    for _ in range(config.irls_iterations):
        weights = base_weights * np.sqrt(robust_weights)
        matrix = np.vstack((pair_rows * weights[:, None], reg_rows * reg_weights[:, None]))
        target = np.concatenate((targets * weights, reg_targets * reg_weights))
        normal = matrix.T @ matrix
        rhs = matrix.T @ target
        kkt = np.zeros((source_count + 1, source_count + 1), np.float64)
        kkt[:source_count, :source_count] = normal
        kkt[:source_count, source_count] = gauge
        kkt[source_count, :source_count] = gauge
        solved = np.linalg.solve(kkt, np.concatenate((rhs, [0.0])))[:source_count]
        residual = pair_rows @ solved - targets
        center = float(np.median(residual))
        scale = max(1.4826 * float(np.median(np.abs(residual - center))), 1e-6)
        normalized = np.abs(residual) / scale
        robust_weights = np.where(
            normalized <= config.huber_delta,
            1.0,
            config.huber_delta / np.maximum(normalized, 1e-12),
        )
    return solved, evidence


def _solve_anchored_component_logs(
    source_count: int,
    nodes: Sequence[int],
    edges: Sequence[S13PhotometricSampleSet],
    anchors: set[int],
    config: S13M63Config,
) -> np.ndarray:
    """Robustly solve free component nodes with cut anchors eliminated exactly."""

    free = [node for node in nodes if node not in anchors]
    result = np.zeros(source_count, np.float64)
    if not free:
        return result
    pair_rows, targets, base_weights = _relations(edges)
    if pair_rows.shape[1] < source_count:
        pair_rows = np.pad(pair_rows, ((0, 0), (0, source_count - pair_rows.shape[1])))
    columns = np.asarray(free, np.intp)
    pair_matrix = pair_rows[:, columns]
    regularization, reg_targets, reg_weights = _regularization_rows(source_count, config)
    # Do not couple across a quality cut: keep only rows whose nonzero nodes
    # are entirely inside this component.
    node_set = set(nodes)
    keep = np.asarray([
        set(np.flatnonzero(row)).issubset(node_set) for row in regularization
    ], bool)
    reg_matrix = regularization[keep][:, columns]
    reg_targets = reg_targets[keep]
    reg_weights = reg_weights[keep]
    solved = np.zeros(len(free), np.float64)
    robust = np.ones(len(targets), np.float64)
    for _ in range(config.irls_iterations):
        weights = base_weights * np.sqrt(robust)
        matrix = np.vstack((pair_matrix * weights[:, None], reg_matrix * reg_weights[:, None]))
        target = np.concatenate((targets * weights, reg_targets * reg_weights))
        solved = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residual = pair_matrix @ solved - targets
        center = float(np.median(residual))
        scale = max(1.4826 * float(np.median(np.abs(residual - center))), 1e-6)
        normalized = np.abs(residual) / scale
        robust = np.where(
            normalized <= config.huber_delta,
            1.0,
            config.huber_delta / np.maximum(normalized, 1e-12),
        )
    result[columns] = solved
    return result


def _metrics(samples: Sequence[S13PhotometricSampleSet], gains: np.ndarray) -> dict[str, object]:
    pair_rows: list[dict[str, float | int]] = []
    residuals: list[np.ndarray] = []
    for item in samples:
        left = item.heldout_left_rgb_linear * gains[item.left_source_index]
        right = item.heldout_right_rgb_linear * gains[item.right_source_index]
        if not len(left):
            continue
        values = np.linalg.norm(left - right, axis=1)
        residuals.append(values)
        pair_rows.append({
            "pair_index": int(item.pair_index),
            "median_linear": float(np.median(values)),
            "p95_linear": float(np.quantile(values, 0.95)),
            "sample_count": int(len(values)),
        })
    if not residuals:
        return {"evaluable": False, "pair_metrics": ()}
    all_values = np.concatenate(residuals)
    p95 = np.asarray([float(item["p95_linear"]) for item in pair_rows], np.float64)
    return {
        "evaluable": True,
        "aggregate_median_linear": float(np.median(all_values)),
        "aggregate_p95_linear": float(np.quantile(all_values, 0.95)),
        "macro_pair_p95_linear": float(np.mean(p95)),
        "worst_pair_p95_linear": float(np.max(p95)),
        "pair_metrics": tuple(pair_rows),
        "sample_count": int(len(all_values)),
    }


def _selection_reasons(
    baseline: Mapping[str, object], candidate: Mapping[str, object], config: S13M63Config,
) -> tuple[list[str], dict[str, float]]:
    if baseline.get("evaluable") is not True or candidate.get("evaluable") is not True:
        return ["validation_unevaluable"], {}
    before_median = float(baseline["aggregate_median_linear"])
    after_median = float(candidate["aggregate_median_linear"])
    before_p95 = float(baseline["aggregate_p95_linear"])
    after_p95 = float(candidate["aggregate_p95_linear"])
    before_macro = float(baseline["macro_pair_p95_linear"])
    after_macro = float(candidate["macro_pair_p95_linear"])
    before_by_pair = {int(item["pair_index"]): item for item in baseline["pair_metrics"]}
    after_by_pair = {int(item["pair_index"]): item for item in candidate["pair_metrics"]}
    common = sorted(set(before_by_pair) & set(after_by_pair))
    improvement = float(np.mean([
        float(after_by_pair[key]["median_linear"]) < float(before_by_pair[key]["median_linear"])
        for key in common
    ])) if common else 0.0
    nonregression = float(np.mean([
        float(after_by_pair[key]["p95_linear"])
        <= float(before_by_pair[key]["p95_linear"]) * (1.0 + config.maximum_aggregate_p95_regression_fraction)
        for key in common
    ])) if common else 0.0
    worst_before = float(baseline["worst_pair_p95_linear"])
    worst_after = float(candidate["worst_pair_p95_linear"])
    worst_change = worst_after - worst_before
    reasons: list[str] = []
    if before_median - after_median < before_median * config.minimum_aggregate_median_benefit_fraction:
        reasons.append("aggregate_benefit_too_small")
    if after_p95 > before_p95 * (1.0 + config.maximum_aggregate_p95_regression_fraction):
        reasons.append("aggregate_p95_regression")
    if after_macro > before_macro * (1.0 + config.maximum_macro_p95_regression_fraction):
        reasons.append("macro_p95_regression")
    if improvement < config.minimum_pair_improvement_fraction:
        reasons.append("pair_improvement_fraction_too_small")
    if nonregression < config.minimum_pair_nonregression_fraction:
        reasons.append("pair_nonregression_fraction_too_small")
    if worst_change > min(
        worst_before * config.maximum_trusted_worst_pair_p95_regression_fraction,
        config.maximum_trusted_worst_pair_p95_regression_absolute,
    ):
        reasons.append("trusted_worst_pair_regression")
    return reasons, {
        "pair_improvement_fraction": improvement,
        "pair_nonregression_fraction": nonregression,
        "trusted_worst_pair_change": worst_change,
    }


def _solution(
    model: str,
    frame_ids: Sequence[int],
    gains: np.ndarray,
    evidence: np.ndarray,
    candidate_audits: Sequence[Mapping[str, object]],
    metrics: Mapping[str, object],
    component_ids: Sequence[int] | None = None,
) -> S13PhotometricSolution:
    ids = list(range(len(frame_ids))) if component_ids is None else component_ids
    parameters = tuple(S13SourcePhotometricParameters(
        source_index=index,
        frame_id=int(frame_id),
        model=model,
        gain_bgr=(float(gains[index]),) * 3,
        bias_bgr=(0.0, 0.0, 0.0),
        component_id=int(ids[index]),
        evidence_weight=float(evidence[index]),
        fallback_reason=None,
    ) for index, frame_id in enumerate(frame_ids))
    return S13PhotometricSolution(
        schema=PHOTOMETRIC_SCHEMA,
        bounds_schema=PHOTOMETRIC_BOUNDS_SCHEMA,
        model_family=model,
        source_parameters=parameters,
        candidate_audits=tuple(candidate_audits),
        train_audit={},
        heldout_audit=metrics,
        split_seed=20260804,
        train_fraction=0.75,
        low_frequency_luminance_field_enabled=False,
    )


def solve_s13_m63_photometric(
    solve_samples: Sequence[S13PhotometricSampleSet],
    adjacent_samples: Sequence[S13PhotometricSampleSet],
    *,
    frame_ids: Sequence[int],
    config: S13M63Config,
    force_identity: bool = False,
) -> S13M63SolveResult:
    source_count = len(frame_ids)
    forced_regression: set[int] = set()
    quality_rows: tuple[S13M63PairQuality, ...] = ()
    raw_logs: np.ndarray | None = None
    evidence = np.zeros(source_count, np.float64)
    for refit in range(2):
        quality_rows = classify_s13_m63_pairs(
            adjacent_samples,
            soft_absolute_pair_p95_linear=config.soft_absolute_pair_p95_linear,
            hard_absolute_pair_p95_linear=config.hard_absolute_pair_p95_linear,
            soft_population_mad_multiplier=config.soft_population_mad_multiplier,
            hard_population_mad_multiplier=config.hard_population_mad_multiplier,
            soft_minimum_inlier_fraction=config.soft_minimum_inlier_fraction,
            hard_minimum_inlier_fraction=config.hard_minimum_inlier_fraction,
            soft_maximum_log_luminance_mad=config.soft_maximum_log_luminance_mad,
            hard_maximum_log_luminance_mad=config.hard_maximum_log_luminance_mad,
            minimum_heldout_samples=config.minimum_heldout_samples,
            forced_hard_pair_indices=frozenset(forced_regression),
        )
        cuts = {item.pair_index for item in quality_rows if item.classification == "photometric_quality_cut"}
        soft = {item.pair_index for item in quality_rows if item.classification == "soft_low_confidence"}
        eligible = []
        for item in solve_samples:
            if item.edge_eligible is not True or _crosses_cut(item, cuts):
                continue
            eligible.append(item)
        components = _components(source_count, eligible)
        raw_logs = None
        if not force_identity and len(components) == 1 and eligible:
            raw_logs, evidence = solve_s13_m63_centered_logs(
                source_count,
                eligible,
                config,
                soft_pair_indices=frozenset(soft),
            )
            raw_gain = np.exp(raw_logs)
            raw_metrics = _metrics(
                [item for item in adjacent_samples if item.pair_index not in cuts], raw_gain
            )
            baseline = _metrics(
                [item for item in adjacent_samples if item.pair_index not in cuts],
                np.ones(source_count),
            )
            before_by = {int(item["pair_index"]): item for item in baseline.get("pair_metrics", ())}
            after_by = {int(item["pair_index"]): item for item in raw_metrics.get("pair_metrics", ())}
            catastrophic = {
                key for key in set(before_by) & set(after_by)
                if float(after_by[key]["p95_linear"])
                > float(before_by[key]["p95_linear"]) * 1.10 + 0.001
            }
            new = catastrophic - cuts - forced_regression
            if new and refit == 0:
                forced_regression.update(new)
                continue
        break
    cuts = tuple(item.pair_index for item in quality_rows if item.classification == "photometric_quality_cut")
    soft = tuple(item.pair_index for item in quality_rows if item.classification == "soft_low_confidence")
    trusted = tuple(item.pair_index for item in quality_rows if item.classification != "photometric_quality_cut")
    trusted_samples = [item for item in adjacent_samples if item.pair_index in trusted]
    baseline = _metrics(trusted_samples, np.ones(source_count))
    audits: list[dict[str, object]] = [{
        "model": "Q0_identity",
        "selected": False,
        "evidence_graph_connected": len(_components(source_count, [
            item for item in solve_samples if item.edge_eligible is True
        ])) == 1,
        "all_sources_parameterized": True,
        "raw_fallback_source_count": 0,
        "out_of_bounds_source_count": 0,
        "nonfinite_source_count": 0,
        "partial_identity_fallback_detected": False,
        "rejection_reasons": [],
        "heldout": baseline,
    }]
    selected_model = "Q0_identity"
    selected_gains = np.ones(source_count, np.float64)
    selected_alpha: float | None = None
    if raw_logs is not None and np.isfinite(raw_logs).all() and not force_identity:
        for alpha in config.shrink_candidates:
            gains = np.exp(alpha * raw_logs)
            metrics = _metrics(trusted_samples, gains)
            reasons, fractions = _selection_reasons(baseline, metrics, config)
            out_of_bounds = int(np.count_nonzero(
                (gains < config.minimum_gain) | (gains > config.maximum_gain)
            ))
            if out_of_bounds:
                reasons.append("gain_out_of_bounds")
            audit = {
                "model": Q1R_MODEL,
                "alpha": alpha,
                "selected": False,
                "evidence_graph_connected": True,
                "all_sources_parameterized": True,
                "raw_fallback_source_count": 0,
                "out_of_bounds_source_count": out_of_bounds,
                "nonfinite_source_count": 0,
                "partial_identity_fallback_detected": False,
                "raw_gain_minimum": float(np.min(np.exp(raw_logs))),
                "raw_gain_maximum": float(np.max(np.exp(raw_logs))),
                "selected_gain_minimum": float(np.min(gains)),
                "selected_gain_maximum": float(np.max(gains)),
                "rejection_reasons": reasons,
                "heldout": metrics,
                **fractions,
            }
            audits.append(audit)
            if not reasons and selected_model == "Q0_identity":
                selected_model = Q1R_MODEL
                selected_gains = gains
                selected_alpha = alpha
                audit["selected"] = True
    # Quality cuts intentionally split the graph.  Q4c solves each component
    # atomically while fixing every source adjacent to a cut at identity.
    q4c_rows: list[dict[str, object]] = []
    if selected_model == "Q0_identity" and cuts and config.q4c_enabled and not force_identity:
        cut_set = set(cuts)
        eligible = [
            item for item in solve_samples
            if item.edge_eligible is True and not _crosses_cut(item, cut_set)
        ]
        components = _components(source_count, eligible)
        anchors = {value for cut in cuts for value in (cut, cut + 1)}
        q4_gains = np.ones(source_count, np.float64)
        component_ids = [-1] * source_count
        accepted = 0
        for component_id, nodes in enumerate(components):
            for node in nodes:
                component_ids[node] = component_id
            node_set = set(nodes)
            edges = [
                item for item in eligible
                if item.left_source_index in node_set and item.right_source_index in node_set
            ]
            component_samples = [
                item for item in trusted_samples
                if item.left_source_index in node_set and item.right_source_index in node_set
            ]
            reasons: list[str] = []
            candidate = np.ones(source_count, np.float64)
            free = [node for node in nodes if node not in anchors]
            if not free or not edges:
                reasons.append("insufficient_component_evidence")
            else:
                raw_component_logs = _solve_anchored_component_logs(
                    source_count, nodes, edges, anchors, config
                )
                columns = np.asarray(free, np.intp)
                if not np.isfinite(raw_component_logs[columns]).all():
                    reasons.append("component_nonfinite")
            before = _metrics(component_samples, np.ones(source_count))
            after = _metrics(component_samples, candidate)
            if before.get("evaluable") is not True:
                reasons.append("component_heldout_unevaluable")
            elif not reasons:
                candidate_reasons: list[str] = []
                for alpha in config.shrink_candidates:
                    trial = np.ones(source_count, np.float64)
                    trial[columns] = np.exp(alpha * raw_component_logs[columns])
                    trial_after = _metrics(component_samples, trial)
                    trial_reasons: list[str] = []
                    if np.any(trial[columns] < config.q4c_minimum_gain) or np.any(
                        trial[columns] > config.q4c_maximum_gain
                    ):
                        trial_reasons.append("component_gain_out_of_bounds")
                    if trial_after.get("evaluable") is not True:
                        trial_reasons.append("component_heldout_unevaluable")
                    else:
                        before_median = float(before["aggregate_median_linear"])
                        before_macro = float(before["macro_pair_p95_linear"])
                        if before_median - float(trial_after["aggregate_median_linear"]) < (
                            before_median * config.q4c_minimum_component_median_benefit_fraction
                        ):
                            trial_reasons.append("component_aggregate_benefit_too_small")
                        if before_macro - float(trial_after["macro_pair_p95_linear"]) < (
                            before_macro * config.q4c_minimum_component_macro_p95_benefit_fraction
                        ):
                            trial_reasons.append("component_macro_benefit_too_small")
                        if float(trial_after["worst_pair_p95_linear"]) > float(before["worst_pair_p95_linear"]) * 1.05 + 0.001:
                            trial_reasons.append("component_worst_pair_regression")
                    candidate_reasons = trial_reasons
                    if not trial_reasons:
                        candidate = trial
                        after = trial_after
                        break
                else:
                    reasons.extend(candidate_reasons)
            if not reasons:
                q4_gains[nodes] = candidate[nodes]
                accepted += 1
            q4c_rows.append({
                "component_id": component_id,
                "source_indices": nodes,
                "anchor_sources": sorted(node_set & anchors),
                "accepted": not reasons,
                "rejection_reasons": reasons,
                "gain_minimum": float(np.min(candidate[nodes], initial=1.0)),
                "gain_maximum": float(np.max(candidate[nodes], initial=1.0)),
                "before": before,
                "after": after,
            })
        q4_metrics = _metrics(trusted_samples, q4_gains)
        q4_reasons, fractions = _selection_reasons(baseline, q4_metrics, config)
        if accepted == 0:
            q4_reasons.append("no_accepted_component")
        q4_audit = {
            "model": Q4C_MODEL,
            "selected": not q4_reasons,
            "component_count": len(components),
            "accepted_component_count": accepted,
            "anchor_sources": sorted(anchors),
            "all_sources_parameterized": True,
            "raw_fallback_source_count": 0,
            "out_of_bounds_source_count": 0,
            "nonfinite_source_count": 0,
            "partial_identity_fallback_detected": False,
            "rejection_reasons": q4_reasons,
            "heldout": q4_metrics,
            "components": q4c_rows,
            **fractions,
        }
        audits.append(q4_audit)
        if not q4_reasons:
            selected_model = Q4C_MODEL
            selected_gains = q4_gains
            selected_alpha = 1.0
    if selected_model == "Q0_identity":
        audits[0]["selected"] = True
    selected_metrics = _metrics(trusted_samples, selected_gains)
    selected_audit = next(
        (item for item in audits if item.get("selected") is True), audits[0]
    )
    solution = _solution(
        selected_model,
        frame_ids,
        selected_gains,
        evidence,
        audits,
        selected_metrics,
        component_ids=(component_ids if selected_model == Q4C_MODEL else None),
    )
    ql_shadow = (
        evaluate_s13_m63_ql_shadow(
            adjacent_samples, excluded_pair_indices=frozenset(cuts)
        )
        if config.ql_enabled
        else {"authority": "disabled", "active_pair_count": 0, "pairs": []}
    )
    return S13M63SolveResult(
        solution=solution,
        quality_rows=quality_rows,
        quality_cut_pair_indices=tuple(sorted(cuts)),
        soft_downweighted_pair_indices=tuple(sorted(soft)),
        trusted_pair_indices=tuple(sorted(trusted)),
        audit={
            "selected_model": selected_model,
            "selected_alpha": selected_alpha,
            "source_fallback_count": 0,
            "quality_cut_pair_indices": sorted(cuts),
            "soft_downweighted_pair_indices": sorted(soft),
            "trusted_pair_count": len(trusted),
            "cut_guard_width_px": config.cut_guard_width_px,
            "pair_improvement_fraction": selected_audit.get("pair_improvement_fraction"),
            "pair_nonregression_fraction": selected_audit.get("pair_nonregression_fraction"),
            "aggregate_metrics": {
                "before": baseline,
                "after": selected_metrics,
            },
            "trusted_worst_pair_change": selected_audit.get("trusted_worst_pair_change"),
            "rejection_reasons": list(selected_audit.get("rejection_reasons", ())),
            "q4c_components": q4c_rows,
            "ql": ql_shadow,
            "ql_authority": ql_shadow["authority"],
            "qt_enabled": config.qt_enabled,
        },
    )


__all__ = [
    "Q1R_MODEL",
    "Q4C_MODEL",
    "S13M63Config",
    "S13M63SolveResult",
    "solve_s13_m63_centered_logs",
    "solve_s13_m63_photometric",
]
