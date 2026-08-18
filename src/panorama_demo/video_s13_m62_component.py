"""Shadow-only Q4c boundary-anchored scalar component candidate."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .video_s13_photometric import S13PhotometricSampleSet


Q4C_MODEL = "Q4c_component_boundary_anchored_scalar_gain"


def _components(source_count: int, samples: Sequence[S13PhotometricSampleSet]) -> list[list[int]]:
    adjacency = [set() for _ in range(source_count)]
    for item in samples:
        if item.edge_eligible is not True:
            continue
        adjacency[item.left_source_index].add(item.right_source_index)
        adjacency[item.right_source_index].add(item.left_source_index)
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


def _metrics(
    samples: Sequence[S13PhotometricSampleSet], nodes: set[int], gains: np.ndarray,
) -> dict[str, float | int | bool | None]:
    pair_values: list[np.ndarray] = []
    for item in samples:
        if item.left_source_index not in nodes or item.right_source_index not in nodes:
            continue
        if not len(item.heldout_left_rgb_linear):
            continue
        left = item.heldout_left_rgb_linear * gains[item.left_source_index]
        right = item.heldout_right_rgb_linear * gains[item.right_source_index]
        pair_values.append(np.linalg.norm(left - right, axis=1))
    if not pair_values:
        return {"evaluable": False, "sample_count": 0, "aggregate_median_linear": None,
                "macro_pair_p95_linear": None, "worst_pair_p95_linear": None}
    all_values = np.concatenate(pair_values)
    p95 = np.asarray([np.quantile(item, 0.95) for item in pair_values], np.float64)
    return {
        "evaluable": True, "sample_count": int(all_values.size),
        "aggregate_median_linear": float(np.median(all_values)),
        "macro_pair_p95_linear": float(np.mean(p95)),
        "worst_pair_p95_linear": float(np.max(p95)),
    }


def evaluate_s13_m62_q4c_shadow(
    source_count: int,
    samples: Sequence[S13PhotometricSampleSet],
    unsupported_cut_pair_indices: Sequence[int],
    *, minimum_gain: float = 0.94, maximum_gain: float = 1.06,
) -> dict[str, object]:
    """Estimate Q4c without granting it pixel authority."""

    components = _components(source_count, samples)
    adjacent_by_index = {item.pair_index: item for item in samples if item.edge_kind == "adjacent"}
    anchored: set[int] = set()
    for pair_index in unsupported_cut_pair_indices:
        item = adjacent_by_index.get(int(pair_index))
        if item is not None:
            anchored.update((item.left_source_index, item.right_source_index))
    gains = np.ones(source_count, np.float64)
    component_rows: list[dict[str, object]] = []
    accepted_count = 0
    for component_index, nodes_list in enumerate(components):
        nodes = set(nodes_list)
        edges = [
            item for item in samples
            if item.edge_eligible is True
            and item.left_source_index in nodes and item.right_source_index in nodes
        ]
        reasons: list[str] = []
        candidate = np.ones(source_count, np.float64)
        if len(nodes) <= 1 or not edges:
            reasons.append("insufficient_component_evidence")
        else:
            local = {node: index for index, node in enumerate(nodes_list)}
            rows: list[np.ndarray] = []
            targets: list[float] = []
            for edge in edges:
                left = np.maximum(edge.train_left_rgb_linear.mean(axis=1), 1e-6)
                right = np.maximum(edge.train_right_rgb_linear.mean(axis=1), 1e-6)
                row = np.zeros(len(nodes_list), np.float64)
                row[local[edge.left_source_index]] = -1.0
                row[local[edge.right_source_index]] = 1.0
                rows.append(row)
                targets.append(float(np.median(np.log(left) - np.log(right))))
            exact_anchors = sorted(nodes & anchored)
            if not exact_anchors:
                exact_anchors = [nodes_list[0]]
            free_nodes = [node for node in nodes_list if node not in exact_anchors]
            if free_nodes:
                free_columns = [local[node] for node in free_nodes]
                matrix = np.stack(rows)[:, free_columns]
                solved = np.linalg.lstsq(matrix, np.asarray(targets), rcond=None)[0]
                for node, value in zip(free_nodes, solved, strict=True):
                    candidate[node] = float(np.exp(value))
            if (
                not np.isfinite(candidate[list(nodes)]).all()
                or np.any(candidate[list(nodes)] < minimum_gain)
                or np.any(candidate[list(nodes)] > maximum_gain)
            ):
                reasons.append("component_gain_out_of_bounds")
        baseline = _metrics(edges, nodes, np.ones(source_count, np.float64))
        after = _metrics(edges, nodes, candidate)
        if baseline.get("evaluable") is not True or after.get("evaluable") is not True:
            reasons.append("component_heldout_unevaluable")
        else:
            before_median = float(baseline["aggregate_median_linear"])
            after_median = float(after["aggregate_median_linear"])
            before_macro = float(baseline["macro_pair_p95_linear"])
            after_macro = float(after["macro_pair_p95_linear"])
            if before_median - after_median < 0.02 * before_median:
                reasons.append("component_aggregate_benefit_too_small")
            if before_macro - after_macro < 0.03 * before_macro:
                reasons.append("component_macro_benefit_too_small")
            if float(after["worst_pair_p95_linear"]) > float(baseline["worst_pair_p95_linear"]) + max(
                0.02 * float(baseline["worst_pair_p95_linear"]), 0.0005
            ):
                reasons.append("component_worst_pair_regression")
        accepted = not reasons
        if accepted:
            gains[list(nodes)] = candidate[list(nodes)]
            accepted_count += 1
        component_rows.append({
            "component_index": component_index, "source_indices": nodes_list,
            "anchored_source_indices": sorted(nodes & anchored),
            "accepted": accepted, "rejection_reasons": reasons,
            "before": baseline, "after": after,
            "candidate_gains": [float(candidate[node]) for node in nodes_list],
        })
    nonidentity = int(np.count_nonzero(np.abs(gains - 1.0) > 1e-12))
    return {
        "model": Q4C_MODEL, "authority": "shadow", "component_count": len(components),
        "accepted_component_count": accepted_count,
        "nonidentity_source_count": nonidentity,
        "would_select": bool(nonidentity > 0 and accepted_count > 0),
        "gain_minimum": float(np.min(gains, initial=1.0)),
        "gain_maximum": float(np.max(gains, initial=1.0)),
        "bias": 0.0, "unsupported_cut_pair_indices": list(unsupported_cut_pair_indices),
        "unsupported_cut_boundary_sources_identity": all(gains[node] == 1.0 for node in anchored),
        "cut_guard_changed_pixel_count": 0,
        "components": component_rows,
    }


__all__ = ["Q4C_MODEL", "evaluate_s13_m62_q4c_shadow"]
