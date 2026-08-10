from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_s12_anchor import (
    S12AnchorConfig,
    S12AnchorEvidence,
    select_automatic_anchors,
)
from panorama_demo.video_s12_motion_graph import (
    S12MotionEdge,
    S12MotionGraph,
    S12MotionGraphConfig,
    S12MotionGraphError,
    S12MotionNode,
)


def _node(index: int, quality: float = 0.9) -> S12MotionNode:
    return S12MotionNode(index, 1000 + index, np.zeros((32, 64), dtype=np.uint8), index * 0.01, quality)


def _edge(left: int, right: int, *, confidence: float = 0.9, reliable: bool = True, kind: str | None = None) -> S12MotionEdge:
    edge_kind = kind or ("adjacent" if right - left == 1 else "skip")
    return S12MotionEdge(
        left, right, 1000 + left, 1000 + right, 16.0 * (right - left), 0.0,
        50, 0.9, 0.8, 0.8, 0.1, 0.0, (right - left) * 0.01, True,
        confidence, reliable, () if reliable else ("rejected",), edge_kind, True,
    )


def _graph(count: int = 15) -> S12MotionGraph:
    nodes = tuple(_node(index) for index in range(count))
    edges = [_edge(index, index + 1) for index in range(count - 1)]
    edges += [_edge(index, index + 2) for index in range(count - 2)]
    edges += [_edge(index, index + 4) for index in range(count - 4)]
    return S12MotionGraph(nodes, tuple(edges))


def test_two_pass_anchor_selection_is_deterministic_and_uses_direct_long_edges() -> None:
    graph = _graph()
    calls: list[tuple[int, int]] = []

    def estimator(left: int, right: int) -> S12MotionEdge:
        calls.append((left, right))
        return _edge(left, right, confidence=0.92, kind="anchor_long")

    selection = select_automatic_anchors(
        graph, [S12AnchorEvidence() for _ in graph.nodes], calibrated_cx=212.0,
        long_edge_estimator=estimator,
    )

    assert selection.anchor_nodes[0] == 0
    assert selection.anchor_nodes[-1] == len(graph.nodes) - 1
    assert selection.anchor_nodes == (0, 6, 14)
    assert all(edge.kind == "anchor_long" and edge.direct_image_match for edge in selection.long_edges)
    assert all(audit.local_path_edge_indices for audit in selection.interval_audits)
    assert set((edge.left_node, edge.right_node) for edge in selection.long_edges).issubset(set(calls))
    assert selection.provisional_solution.centers_px[0] == 212.0


def test_anchor_priority_uses_long_edge_confidence_before_quality_and_distance() -> None:
    graph = _graph(15)
    nodes = list(graph.nodes)
    nodes[5] = _node(5, quality=1.0)
    graph = S12MotionGraph(tuple(nodes), graph.edges)

    def estimator(left: int, right: int) -> S12MotionEdge:
        confidence = 0.95 if right == 6 else 0.75
        return _edge(left, right, confidence=confidence, kind="anchor_long")

    selection = select_automatic_anchors(
        graph, [S12AnchorEvidence() for _ in graph.nodes], calibrated_cx=100.0,
        long_edge_estimator=estimator,
    )

    assert selection.anchor_nodes[1] == 6


def test_anchor_rejects_missing_direct_long_edge_even_when_local_graph_is_connected() -> None:
    graph = _graph(8)

    def estimator(left: int, right: int) -> S12MotionEdge:
        return _edge(left, right, confidence=0.99, reliable=False, kind="anchor_long")

    with pytest.raises(S12MotionGraphError, match="no reliable direct anchor edge"):
        select_automatic_anchors(
            graph, [S12AnchorEvidence() for _ in graph.nodes], calibrated_cx=100.0,
            long_edge_estimator=estimator,
        )


def test_anchor_candidate_must_have_direct_pose_quality_attitude_local_and_skip_evidence() -> None:
    graph = _graph(15)
    evidence = [S12AnchorEvidence() for _ in graph.nodes]
    evidence[6] = S12AnchorEvidence(direct_pose_qualified=False)

    def estimator(left: int, right: int) -> S12MotionEdge:
        return _edge(left, right, confidence=0.9, kind="anchor_long")

    selection = select_automatic_anchors(
        graph, evidence, calibrated_cx=100.0, long_edge_estimator=estimator,
    )

    assert selection.anchor_nodes[1] == 5


def test_anchor_final_solver_audits_direct_edge_against_independent_local_path() -> None:
    graph = _graph(8)

    def estimator(left: int, right: int) -> S12MotionEdge:
        return _edge(left, right, confidence=0.9, kind="anchor_long")

    selection = select_automatic_anchors(
        graph, [S12AnchorEvidence() for _ in graph.nodes], calibrated_cx=212.0,
        long_edge_estimator=estimator,
        solver_config=S12MotionGraphConfig(maximum_reliable_edge_adjustment_px=0.1),
        anchor_config=S12AnchorConfig(),
    )

    assert all(audit.direct_path_difference_px == pytest.approx(0.0) for audit in selection.interval_audits)
    assert selection.final_solution.maximum_edge_adjustment_px < 1e-9
