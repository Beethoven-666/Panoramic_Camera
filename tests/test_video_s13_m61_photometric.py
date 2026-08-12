from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from panorama_demo.video_s13_m61_config import (
    M61_GRAPH_TOPOLOGY_SCHEMA,
    S13PhotometricSelectionConfig,
    S13PhotometricSolverConfig,
)
from panorama_demo.video_s13_m61_photometric import (
    S13M61PairEvidence,
    evaluate_precomputed_candidate_metrics,
    load_sealed_graph_topology,
    solve_and_select_s13_m61_photometric,
)


def _write_topology(tmp_path: Path, components: list[list[int]], edges: list[tuple[int, int, bool]]) -> Path:
    evidence_sha = "a" * 64
    topology = {
        "schema": M61_GRAPH_TOPOLOGY_SCHEMA,
        "branch": "synthetic", "run_id": "r", "generation_id": "g",
        "train_only": True, "photometric_evidence_config_sha256": evidence_sha,
        "edges": [
            {
                "pair_index": index, "left_source_index": left,
                "right_source_index": left + 1, "accepted": accepted,
                "train_safe_sample_count": 512, "train_tile_count": 8,
                "vertical_block_count": 4,
                "acceptance_reason": "accepted" if accepted else "insufficient_train_evidence",
            }
            for index, (left, _right, accepted) in enumerate(edges)
        ],
        "components": [
            {"component_index": index, "source_indices": members,
             "single_source_identity_boundary": len(members) == 1}
            for index, members in enumerate(components)
        ],
        "component_count": len(components), "source_count": sum(map(len, components)),
        "contains_candidate_parameters": False,
    }
    root = tmp_path / "graph_topology"
    root.mkdir()
    path = root / "topology.json"
    path.write_text(json.dumps(topology, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps({
        "schema": M61_GRAPH_TOPOLOGY_SCHEMA, "topology": "topology.json",
        "topology_sha256": digest, "train_only": True,
        "photometric_evidence_config_sha256": evidence_sha,
    }), encoding="utf-8")
    return root


def _pair(index: int, left: int, right: int, gain: float = 1.12) -> S13M61PairEvidence:
    rng = np.random.default_rng(index + 10)
    train = rng.uniform(0.08, 0.65, (600, 3)).astype(np.float64)
    validation = rng.uniform(0.08, 0.65, (300, 3)).astype(np.float64)
    return S13M61PairEvidence(
        index, left, right, train, train / gain, validation, validation / gain, True
    )


def test_sealed_topology_rejects_tamper_and_does_not_reselect_edges(tmp_path: Path) -> None:
    root = _write_topology(tmp_path, [[0, 1]], [(0, 1, True)])
    topology = load_sealed_graph_topology(root, expected_evidence_config_sha256="a" * 64)
    assert topology.components[0].source_indices == (0, 1)
    document = json.loads((root / "topology.json").read_text(encoding="utf-8"))
    document["edges"][0]["accepted"] = False
    (root / "topology.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_sealed_graph_topology(root, expected_evidence_config_sha256="a" * 64)


def test_component_solve_is_bounded_and_never_creates_internal_identity_boundary(tmp_path: Path) -> None:
    root = _write_topology(tmp_path, [[0, 1, 2]], [(0, 1, True), (1, 2, True)])
    topology = load_sealed_graph_topology(root, expected_evidence_config_sha256="a" * 64)
    plan = solve_and_select_s13_m61_photometric(
        topology, [_pair(0, 0, 1), _pair(1, 1, 2)],
        solver_config=S13PhotometricSolverConfig(),
        selection_config=S13PhotometricSelectionConfig(),
    )
    assert plan.identity_fallback_inside_solved_component_count == 0
    assert plan.selected_model_by_component[0] in {
        "Q1_scalar_luminance_gain", "Q2_rgb_diagonal_gain", "Q3_bounded_rgb_gain_bias"
    }
    assert np.all(plan.gains_rgb >= 0.8) and np.all(plan.gains_rgb <= 1.25)


def test_out_of_bounds_source_rolls_back_whole_component(tmp_path: Path) -> None:
    root = _write_topology(tmp_path, [[0, 1]], [(0, 1, True)])
    topology = load_sealed_graph_topology(root, expected_evidence_config_sha256="a" * 64)
    evidence = _pair(0, 0, 1, gain=2.0)
    plan = solve_and_select_s13_m61_photometric(
        topology, [evidence], solver_config=S13PhotometricSolverConfig(),
        selection_config=S13PhotometricSelectionConfig(),
    )
    assert plan.selected_model_by_component == ("Q0_identity",)
    assert np.array_equal(plan.gains_rgb, np.ones((2, 3)))


def _metrics(median: float, p95: float, worst: float, macro: float | None = None) -> dict[str, object]:
    return {
        "evaluable": True, "aggregate_median_linear": median,
        "aggregate_p95_linear": p95, "macro_pair_p95_linear": p95 if macro is None else macro,
        "actionable_macro_pair_p95_linear": p95 if macro is None else macro,
        "worst_pair_p95_linear": worst, "pair_metrics": (),
    }


def test_median_good_p95_bad_and_legacy_slow_ignore_q2_are_rejected() -> None:
    selection = S13PhotometricSelectionConfig()
    accepted, reasons = evaluate_precomputed_candidate_metrics(
        _metrics(0.020, 0.01702, 0.020),
        _metrics(0.018, 0.018559, 0.027),
        selection_config=selection,
    )
    assert not accepted
    assert "aggregate_p95_regression" in reasons
    assert "worst_pair_p95_regression" in reasons

    accepted, reasons = evaluate_precomputed_candidate_metrics(
        _metrics(0.020, 0.020, 0.021),
        _metrics(0.018, 0.023, 0.021),
        selection_config=selection,
    )
    assert not accepted and "aggregate_p95_regression" in reasons


def test_simpler_model_wins_inside_score_mde(tmp_path: Path) -> None:
    root = _write_topology(tmp_path, [[0, 1]], [(0, 1, True)])
    topology = load_sealed_graph_topology(root, expected_evidence_config_sha256="a" * 64)
    pair = _pair(0, 0, 1, gain=1.08)
    plan = solve_and_select_s13_m61_photometric(
        topology, [pair], solver_config=S13PhotometricSolverConfig(),
        selection_config=S13PhotometricSelectionConfig(selection_score_mde_linear=0.01),
    )
    assert plan.selected_model_by_component == ("Q1_scalar_luminance_gain",)
