"""Robust component photometric solve and pre-render selection for S013 M6.1.

The graph is an input, never an output: this module verifies and consumes the
sealed train-only topology and cannot add/remove an edge or split a component
after seeing validation data.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .video_s13_m61_config import (
    M61_GRAPH_TOPOLOGY_SCHEMA,
    M61_PHOTOMETRIC_SOLUTION_SCHEMA,
    S13PhotometricSelectionConfig,
    S13PhotometricSolverConfig,
)


MODEL_ORDER = (
    "Q0_identity",
    "Q1_scalar_luminance_gain",
    "Q2_rgb_diagonal_gain",
    "Q3_bounded_rgb_gain_bias",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"S013 M6.1 {label} is missing or invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"S013 M6.1 {label} must be an object")
    return value


@dataclass(frozen=True)
class S13M61TopologyEdge:
    pair_index: int
    left_source_index: int
    right_source_index: int
    accepted: bool


@dataclass(frozen=True)
class S13M61TopologyComponent:
    component_index: int
    source_indices: tuple[int, ...]


@dataclass(frozen=True)
class S13M61SealedTopology:
    root: Path
    topology_sha256: str
    evidence_config_sha256: str
    source_count: int
    edges: tuple[S13M61TopologyEdge, ...]
    components: tuple[S13M61TopologyComponent, ...]


@dataclass(frozen=True)
class S13M61PairEvidence:
    pair_index: int
    left_source_index: int
    right_source_index: int
    train_left_rgb_linear: np.ndarray
    train_right_rgb_linear: np.ndarray
    validation_left_rgb_linear: np.ndarray
    validation_right_rgb_linear: np.ndarray
    actionable: bool = True

    def __post_init__(self) -> None:
        for name in (
            "train_left_rgb_linear", "train_right_rgb_linear",
            "validation_left_rgb_linear", "validation_right_rgb_linear",
        ):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
                raise ValueError(f"S013 M6.1 {name} must be finite Nx3 linear RGB")
            if np.any(value < 0.0) or np.any(value > 1.0):
                raise ValueError(f"S013 M6.1 {name} is outside linear RGB")
        if self.train_left_rgb_linear.shape != self.train_right_rgb_linear.shape:
            raise ValueError("S013 M6.1 train evidence shapes disagree")
        if self.validation_left_rgb_linear.shape != self.validation_right_rgb_linear.shape:
            raise ValueError("S013 M6.1 validation evidence shapes disagree")


@dataclass(frozen=True)
class S13M61ComponentCandidate:
    component_index: int
    model: str
    source_indices: tuple[int, ...]
    gains_rgb: np.ndarray
    biases_rgb: np.ndarray
    accepted: bool
    rejection_reasons: tuple[str, ...]
    metrics: Mapping[str, object]
    rank: int
    condition_number: float
    irls_iterations: int


@dataclass(frozen=True)
class S13M61PhotometricPlan:
    schema: str
    topology_sha256: str
    source_count: int
    gains_rgb: np.ndarray
    biases_rgb: np.ndarray
    selected_model_by_component: tuple[str, ...]
    component_candidates: tuple[tuple[S13M61ComponentCandidate, ...], ...]
    identity_fallback_inside_solved_component_count: int


def load_sealed_graph_topology(
    topology_root: str | Path, *, expected_evidence_config_sha256: str
) -> S13M61SealedTopology:
    """Verify a sealed topology byte-for-byte without recomputing membership."""

    root = Path(topology_root).expanduser().resolve()
    manifest = _object(root / "manifest.json", "graph topology manifest")
    if set(manifest) != {
        "schema", "topology", "topology_sha256", "train_only",
        "photometric_evidence_config_sha256",
    }:
        raise ValueError("S013 M6.1 graph topology manifest fields disagree")
    relative = Path(str(manifest["topology"]))
    if relative.is_absolute() or relative != Path("topology.json"):
        raise ValueError("S013 M6.1 graph topology path is unsafe")
    topology_path = root / relative
    actual_sha = _sha256_file(topology_path)
    if manifest.get("schema") != M61_GRAPH_TOPOLOGY_SCHEMA or manifest.get("train_only") is not True:
        raise ValueError("S013 M6.1 graph topology manifest is not train-only v1")
    if manifest.get("topology_sha256") != actual_sha:
        raise ValueError("S013 M6.1 graph topology SHA mismatch")
    if manifest.get("photometric_evidence_config_sha256") != expected_evidence_config_sha256:
        raise ValueError("S013 M6.1 graph topology evidence config SHA mismatch")
    document = _object(topology_path, "graph topology")
    required = {
        "schema", "branch", "run_id", "generation_id", "train_only",
        "photometric_evidence_config_sha256", "edges", "components",
        "component_count", "source_count", "contains_candidate_parameters",
    }
    if set(document) != required:
        raise ValueError("S013 M6.1 graph topology fields disagree")
    if (
        document.get("schema") != M61_GRAPH_TOPOLOGY_SCHEMA
        or document.get("train_only") is not True
        or document.get("contains_candidate_parameters") is not False
        or document.get("photometric_evidence_config_sha256") != expected_evidence_config_sha256
    ):
        raise ValueError("S013 M6.1 graph topology contract is invalid")
    source_count = document.get("source_count")
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 1:
        raise ValueError("S013 M6.1 graph topology source count is invalid")
    raw_edges, raw_components = document.get("edges"), document.get("components")
    if not isinstance(raw_edges, list) or not isinstance(raw_components, list):
        raise ValueError("S013 M6.1 graph topology membership is invalid")
    edges: list[S13M61TopologyEdge] = []
    pair_indices: set[int] = set()
    for value in raw_edges:
        if not isinstance(value, Mapping):
            raise ValueError("S013 M6.1 graph topology edge is invalid")
        required_edge = {
            "pair_index", "left_source_index", "right_source_index", "accepted",
            "train_safe_sample_count", "train_tile_count", "vertical_block_count",
            "acceptance_reason",
        }
        if set(value) != required_edge:
            raise ValueError("S013 M6.1 graph topology edge fields disagree")
        pair = int(value["pair_index"])
        left, right = int(value["left_source_index"]), int(value["right_source_index"])
        accepted = value["accepted"]
        if pair in pair_indices or not 0 <= left < source_count or not 0 <= right < source_count:
            raise ValueError("S013 M6.1 graph topology edge identity is invalid")
        if accepted not in (True, False):
            raise ValueError("S013 M6.1 graph topology edge decision is invalid")
        pair_indices.add(pair)
        edges.append(S13M61TopologyEdge(pair, left, right, bool(accepted)))
    components: list[S13M61TopologyComponent] = []
    seen: set[int] = set()
    for expected_index, value in enumerate(raw_components):
        if not isinstance(value, Mapping) or set(value) != {
            "component_index", "source_indices", "single_source_identity_boundary"
        }:
            raise ValueError("S013 M6.1 graph topology component fields disagree")
        members = value["source_indices"]
        if value["component_index"] != expected_index or not isinstance(members, list) or not members:
            raise ValueError("S013 M6.1 graph topology component order is invalid")
        member_tuple = tuple(int(item) for item in members)
        if member_tuple != tuple(sorted(member_tuple)) or seen.intersection(member_tuple):
            raise ValueError("S013 M6.1 graph topology component membership is invalid")
        if value["single_source_identity_boundary"] is not (len(member_tuple) == 1):
            raise ValueError("S013 M6.1 graph topology singleton declaration disagrees")
        seen.update(member_tuple)
        components.append(S13M61TopologyComponent(expected_index, member_tuple))
    if seen != set(range(source_count)) or document["component_count"] != len(components):
        raise ValueError("S013 M6.1 graph topology does not cover each source exactly once")
    component_by_source = {
        source: component.component_index for component in components for source in component.source_indices
    }
    if any(
        edge.accepted
        and component_by_source[edge.left_source_index] != component_by_source[edge.right_source_index]
        for edge in edges
    ):
        raise ValueError("S013 M6.1 accepted topology edge crosses frozen components")
    return S13M61SealedTopology(
        root=root, topology_sha256=actual_sha,
        evidence_config_sha256=expected_evidence_config_sha256,
        source_count=source_count, edges=tuple(edges), components=tuple(components),
    )


def _huber_irls(
    matrix: np.ndarray, target: np.ndarray, config: S13PhotometricSolverConfig,
) -> tuple[np.ndarray, int, float, int]:
    if matrix.ndim != 2 or target.ndim != 1 or matrix.shape[0] != target.size:
        raise ValueError("S013 M6.1 solver matrix shape is invalid")
    weights = np.ones(target.size, np.float64)
    solution = np.zeros(matrix.shape[1], np.float64)
    iteration_count = 0
    for iteration in range(config.irls_iterations):
        weighted = np.sqrt(weights)
        design = matrix * weighted[:, None]
        values = target * weighted
        normal = design.T @ design
        rhs = design.T @ values
        try:
            candidate = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("rank_or_condition_failed") from exc
        eigenvalues = np.linalg.eigvalsh(normal)
        largest = float(eigenvalues[-1]) if eigenvalues.size else 0.0
        cutoff = max(config.rank_tolerance * largest, config.rank_tolerance ** 2)
        rank = int(np.count_nonzero(eigenvalues > cutoff))
        smallest = float(eigenvalues[0]) if eigenvalues.size else 0.0
        condition = math.sqrt(largest / max(smallest, config.rank_tolerance ** 2))
        residual = matrix @ candidate - target
        scale = max(
            config.minimum_robust_scale_linear,
            float(np.median(np.abs(residual))) * 1.4826,
        )
        normalized = np.abs(residual) / scale
        new_weights = np.ones_like(normalized)
        outlier = normalized > config.huber_delta
        new_weights[outlier] = config.huber_delta / normalized[outlier]
        iteration_count = iteration + 1
        if np.max(np.abs(candidate - solution), initial=0.0) <= config.convergence_tolerance:
            solution = candidate
            weights = new_weights
            break
        solution, weights = candidate, new_weights
    inlier_fraction = float(np.mean(weights >= 0.999)) if weights.size else 0.0
    if rank < matrix.shape[1] or condition > config.maximum_condition_number:
        raise ValueError("rank_or_condition_failed")
    if inlier_fraction < config.minimum_inlier_fraction:
        raise ValueError("robust_inlier_fraction_failed")
    return solution, int(rank), condition, iteration_count


def _component_evidence(
    component: S13M61TopologyComponent,
    topology: S13M61SealedTopology,
    evidence_by_pair: Mapping[int, S13M61PairEvidence],
) -> tuple[S13M61PairEvidence, ...]:
    accepted = {
        edge.pair_index: edge for edge in topology.edges
        if edge.accepted and edge.left_source_index in component.source_indices
    }
    rows: list[S13M61PairEvidence] = []
    for pair_index in sorted(accepted):
        if pair_index not in evidence_by_pair:
            raise ValueError("S013 M6.1 accepted topology edge lacks sealed evidence")
        item = evidence_by_pair[pair_index]
        edge = accepted[pair_index]
        if (
            item.pair_index != pair_index
            or item.left_source_index != edge.left_source_index
            or item.right_source_index != edge.right_source_index
        ):
            raise ValueError("S013 M6.1 evidence identity disagrees with sealed topology")
        rows.append(item)
    extra = set(evidence_by_pair).intersection(
        edge.pair_index for edge in topology.edges if not edge.accepted
    )
    if extra:
        # Rejected evidence may exist on disk, but callers must not present it
        # to the solver as an eligible relation.
        raise ValueError("S013 M6.1 rejected topology edge reached the solver")
    return tuple(rows)


def _identity(component: S13M61TopologyComponent) -> tuple[np.ndarray, np.ndarray]:
    return np.ones((len(component.source_indices), 3)), np.zeros((len(component.source_indices), 3))


def _stratified_indices(sample_count: int, maximum: int) -> np.ndarray:
    """Choose a deterministic, domain-spanning subset without a dense random draw."""

    if sample_count <= maximum:
        return np.arange(sample_count, dtype=np.int64)
    # Evidence is stored in canvas row-major order.  Equal-width bins retain
    # the full vertical/shoulder extent and avoid the old flattened-prefix
    # bias while bounding the IRLS working set.
    edges = np.linspace(0, sample_count, maximum + 1, dtype=np.int64)
    return ((edges[:-1] + edges[1:] - 1) // 2).astype(np.int64)


def _solve_gain(
    component: S13M61TopologyComponent,
    pairs: Sequence[S13M61PairEvidence],
    config: S13PhotometricSolverConfig,
    *,
    scalar: bool,
) -> tuple[np.ndarray, np.ndarray, int, float, int]:
    nodes = component.source_indices
    local = {source: index for index, source in enumerate(nodes)}
    channels = 1 if scalar else 3
    equations: list[np.ndarray] = []
    targets: list[float] = []
    maximum_per_pair = 32
    for pair in pairs:
        left = np.clip(pair.train_left_rgb_linear, 1e-6, 1.0)
        right = np.clip(pair.train_right_rgb_linear, 1e-6, 1.0)
        selected = _stratified_indices(len(left), maximum_per_pair)
        relations = (
            np.log(np.mean(left[selected], axis=1)) - np.log(np.mean(right[selected], axis=1))
        )[:, None] if scalar else np.log(left[selected]) - np.log(right[selected])
        for relation in relations:
            for channel in range(channels):
                row = np.zeros(len(nodes) * channels, np.float64)
                row[local[pair.left_source_index] * channels + channel] = -1.0
                row[local[pair.right_source_index] * channels + channel] = 1.0
                equations.append(row)
                targets.append(float(relation[channel]))
    regularization = math.sqrt(config.identity_regularization)
    for variable in range(len(nodes) * channels):
        row = np.zeros(len(nodes) * channels, np.float64)
        row[variable] = regularization
        equations.append(row)
        targets.append(0.0)
    if not pairs:
        raise ValueError("component_has_no_accepted_edges")
    solved, rank, condition, iterations = _huber_irls(
        np.stack(equations), np.asarray(targets), config,
    )
    gains = np.exp(solved.reshape(len(nodes), channels))
    if scalar:
        gains = np.repeat(gains, 3, axis=1)
    biases = np.zeros_like(gains)
    return gains, biases, rank, condition, iterations


def _solve_affine(
    component: S13M61TopologyComponent,
    pairs: Sequence[S13M61PairEvidence],
    config: S13PhotometricSolverConfig,
) -> tuple[np.ndarray, np.ndarray, int, float, int]:
    nodes = component.source_indices
    local = {source: index for index, source in enumerate(nodes)}
    variable_count = len(nodes) * 6
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for pair in pairs:
        selected = _stratified_indices(len(pair.train_left_rgb_linear), 16)
        for left, right in zip(
            pair.train_left_rgb_linear[selected], pair.train_right_rgb_linear[selected], strict=True
        ):
            for channel in range(3):
                row = np.zeros(variable_count, np.float64)
                li, ri = local[pair.left_source_index], local[pair.right_source_index]
                row[6 * li + channel] = float(left[channel])
                row[6 * li + 3 + channel] = 1.0
                row[6 * ri + channel] = -float(right[channel])
                row[6 * ri + 3 + channel] = -1.0
                rows.append(row)
                targets.append(0.0)
    for node in range(len(nodes)):
        for channel in range(3):
            gain = np.zeros(variable_count, np.float64)
            gain[6 * node + channel] = math.sqrt(config.identity_regularization)
            rows.append(gain)
            targets.append(math.sqrt(config.identity_regularization))
            bias = np.zeros(variable_count, np.float64)
            bias[6 * node + 3 + channel] = math.sqrt(config.identity_regularization)
            rows.append(bias)
            targets.append(0.0)
    if not pairs:
        raise ValueError("component_has_no_accepted_edges")
    solved, rank, condition, iterations = _huber_irls(
        np.stack(rows), np.asarray(targets), config,
    )
    parameters = solved.reshape(len(nodes), 6)
    return parameters[:, :3], parameters[:, 3:], rank, condition, iterations


def _residual_metrics(
    pairs: Sequence[S13M61PairEvidence],
    nodes: tuple[int, ...],
    gains: np.ndarray,
    biases: np.ndarray,
) -> dict[str, object]:
    local = {source: index for index, source in enumerate(nodes)}
    all_values: list[np.ndarray] = []
    pair_rows: list[dict[str, object]] = []
    for pair in pairs:
        left = np.clip(
            pair.validation_left_rgb_linear * gains[local[pair.left_source_index]]
            + biases[local[pair.left_source_index]], 0.0, 1.0,
        )
        right = np.clip(
            pair.validation_right_rgb_linear * gains[local[pair.right_source_index]]
            + biases[local[pair.right_source_index]], 0.0, 1.0,
        )
        values = np.linalg.norm(left - right, axis=1)
        if not values.size:
            continue
        all_values.append(values)
        pair_rows.append({
            "pair_index": pair.pair_index,
            "sample_count": int(values.size),
            "median_linear": float(np.median(values)),
            "p95_linear": float(np.quantile(values, 0.95)),
            "actionable": bool(pair.actionable),
        })
    if not all_values:
        return {"evaluable": False, "pair_metrics": tuple()}
    aggregate = np.concatenate(all_values)
    p95s = np.asarray([row["p95_linear"] for row in pair_rows], np.float64)
    actionable = np.asarray([row["actionable"] for row in pair_rows], bool)
    return {
        "evaluable": True,
        "aggregate_median_linear": float(np.median(aggregate)),
        "aggregate_p95_linear": float(np.quantile(aggregate, 0.95)),
        "macro_pair_p95_linear": float(np.mean(p95s)),
        "actionable_macro_pair_p95_linear": (
            float(np.mean(p95s[actionable])) if np.any(actionable) else None
        ),
        "worst_pair_p95_linear": float(np.max(p95s)),
        "pair_metrics": tuple(pair_rows),
    }


def _selection_reasons(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    config: S13PhotometricSelectionConfig,
) -> list[str]:
    if baseline.get("evaluable") is not True or candidate.get("evaluable") is not True:
        return ["validation_unevaluable"]
    reasons: list[str] = []
    before_p95 = float(baseline["aggregate_p95_linear"])
    after_p95 = float(candidate["aggregate_p95_linear"])
    if not config.aggregate_p95_nonregression_passes(before_p95, after_p95):
        reasons.append("aggregate_p95_regression")
    pair_allowance = max(
        config.worst_pair_p95_nonreg_relative_tolerance
        * float(baseline["worst_pair_p95_linear"]),
        config.worst_pair_p95_nonreg_absolute_tolerance_linear,
    )
    if float(candidate["worst_pair_p95_linear"]) > float(baseline["worst_pair_p95_linear"]) + pair_allowance:
        reasons.append("worst_pair_p95_regression")
    if float(candidate["macro_pair_p95_linear"]) > float(baseline["macro_pair_p95_linear"]) + max(
        0.02 * float(baseline["macro_pair_p95_linear"]),
        config.aggregate_p95_nonreg_absolute_tolerance_linear,
    ):
        reasons.append("macro_p95_regression")
    median_improvement = float(baseline["aggregate_median_linear"]) - float(
        candidate["aggregate_median_linear"]
    )
    median_required = max(
        config.minimum_aggregate_median_benefit_fraction
        * float(baseline["aggregate_median_linear"]),
        config.selection_score_mde_linear,
    )
    before_actionable = baseline.get("actionable_macro_pair_p95_linear")
    after_actionable = candidate.get("actionable_macro_pair_p95_linear")
    actionable_benefit = False
    if isinstance(before_actionable, (int, float)) and isinstance(after_actionable, (int, float)):
        actionable_benefit = float(before_actionable) - float(after_actionable) >= max(
            config.minimum_actionable_macro_p95_benefit_fraction * float(before_actionable),
            config.selection_score_mde_linear,
        )
    if median_improvement < median_required and not actionable_benefit:
        reasons.append("minimum_benefit_not_met")
    return reasons


def solve_and_select_s13_m61_photometric(
    topology: S13M61SealedTopology,
    evidence: Sequence[S13M61PairEvidence],
    *,
    solver_config: S13PhotometricSolverConfig,
    selection_config: S13PhotometricSelectionConfig,
) -> S13M61PhotometricPlan:
    """Select one model per frozen component before any formal render."""

    evidence_by_pair: dict[int, S13M61PairEvidence] = {}
    for item in evidence:
        if item.pair_index in evidence_by_pair:
            raise ValueError("S013 M6.1 duplicate pair evidence")
        evidence_by_pair[item.pair_index] = item
    full_gains = np.ones((topology.source_count, 3), np.float64)
    full_biases = np.zeros((topology.source_count, 3), np.float64)
    selected_models: list[str] = []
    audit_groups: list[tuple[S13M61ComponentCandidate, ...]] = []
    for component in topology.components:
        pairs = _component_evidence(component, topology, evidence_by_pair)
        identity_gain, identity_bias = _identity(component)
        baseline_metrics = _residual_metrics(
            pairs, component.source_indices, identity_gain, identity_bias
        )
        candidates: list[S13M61ComponentCandidate] = [S13M61ComponentCandidate(
            component.component_index, "Q0_identity", component.source_indices,
            identity_gain, identity_bias, True, (), baseline_metrics,
            rank=0, condition_number=1.0, irls_iterations=0,
        )]
        if len(component.source_indices) > 1:
            for model in MODEL_ORDER[1:]:
                reasons: list[str] = []
                try:
                    if model == "Q1_scalar_luminance_gain":
                        gains, biases, rank, condition, iterations = _solve_gain(
                            component, pairs, solver_config, scalar=True
                        )
                    elif model == "Q2_rgb_diagonal_gain":
                        gains, biases, rank, condition, iterations = _solve_gain(
                            component, pairs, solver_config, scalar=False
                        )
                    else:
                        gains, biases, rank, condition, iterations = _solve_affine(
                            component, pairs, solver_config
                        )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    gains, biases = identity_gain.copy(), identity_bias.copy()
                    rank, condition, iterations = 0, math.inf, 0
                    reasons.append(str(exc))
                if (
                    not np.isfinite(gains).all() or not np.isfinite(biases).all()
                    or np.any(gains < solver_config.minimum_gain)
                    or np.any(gains > solver_config.maximum_gain)
                    or np.any(np.abs(biases) > solver_config.maximum_absolute_bias_linear)
                ):
                    reasons.append("component_parameter_bounds_failed")
                metrics = _residual_metrics(pairs, component.source_indices, gains, biases)
                if not reasons:
                    reasons.extend(_selection_reasons(baseline_metrics, metrics, selection_config))
                candidates.append(S13M61ComponentCandidate(
                    component.component_index, model, component.source_indices,
                    gains, biases, not reasons, tuple(reasons), metrics,
                    rank=rank, condition_number=condition, irls_iterations=iterations,
                ))
        # Prefer the simplest candidate within the selection-score MDE.  A
        # rejected complex candidate never causes per-source fallback: the
        # entire frozen component remains on its previous simpler model.
        eligible = [candidate for candidate in candidates if candidate.accepted]
        winner = eligible[0]
        for candidate in eligible[1:]:
            old = float(winner.metrics.get("aggregate_median_linear", math.inf))
            new = float(candidate.metrics.get("aggregate_median_linear", math.inf))
            if new < old - selection_config.selection_score_mde_linear:
                winner = candidate
        for local, source in enumerate(component.source_indices):
            full_gains[source] = winner.gains_rgb[local]
            full_biases[source] = winner.biases_rgb[local]
        selected_models.append(winner.model)
        audit_groups.append(tuple(candidates))
    return S13M61PhotometricPlan(
        schema=M61_PHOTOMETRIC_SOLUTION_SCHEMA,
        topology_sha256=topology.topology_sha256,
        source_count=topology.source_count,
        gains_rgb=full_gains,
        biases_rgb=full_biases,
        selected_model_by_component=tuple(selected_models),
        component_candidates=tuple(audit_groups),
        identity_fallback_inside_solved_component_count=0,
    )


def evaluate_precomputed_candidate_metrics(
    baseline: Mapping[str, object], candidate: Mapping[str, object],
    *, selection_config: S13PhotometricSelectionConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Public deterministic gate used by calibration and mutation tests."""

    reasons = tuple(_selection_reasons(baseline, candidate, selection_config))
    return not reasons, reasons


__all__ = [
    "MODEL_ORDER", "S13M61ComponentCandidate", "S13M61PairEvidence",
    "S13M61PhotometricPlan", "S13M61SealedTopology", "S13M61TopologyComponent",
    "S13M61TopologyEdge", "evaluate_precomputed_candidate_metrics",
    "load_sealed_graph_topology", "solve_and_select_s13_m61_photometric",
]
