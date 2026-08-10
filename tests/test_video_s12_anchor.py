from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from panorama_demo.video_s12_anchor import (
    S12AnchorConfig,
    S12AnchorEvidence,
    build_anchor_evidence,
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


def _edge(
    left: int,
    right: int,
    *,
    confidence: float = 0.9,
    reliable: bool = True,
    kind: str | None = None,
    dx: float | None = None,
    rejection_reasons: tuple[str, ...] | None = None,
) -> S12MotionEdge:
    edge_kind = kind or ("adjacent" if right - left == 1 else "skip")
    return S12MotionEdge(
        left, right, 1000 + left, 1000 + right,
        16.0 * (right - left) if dx is None else dx, 0.0,
        50, 0.9, 0.8, 0.8, 0.1, 0.0, (right - left) * 0.01, True,
        confidence, reliable,
        () if reliable else (rejection_reasons or ("rejected",)), edge_kind, True,
    )


def _graph(count: int = 15) -> S12MotionGraph:
    nodes = tuple(_node(index) for index in range(count))
    edges = [_edge(index, index + 1) for index in range(count - 1)]
    edges += [_edge(index, index + 2) for index in range(count - 2)]
    edges += [_edge(index, index + 4) for index in range(count - 4)]
    return S12MotionGraph(nodes, tuple(edges))


def _evidence(
    graph: S12MotionGraph,
    *,
    direct_false: tuple[int, ...] = (),
    attitude_false: tuple[int, ...] = (),
    paused: tuple[int, ...] = (),
    backward: tuple[int, ...] = (),
) -> tuple[S12AnchorEvidence, ...]:
    count = len(graph.nodes)
    return build_anchor_evidence(
        graph,
        direct_pose_qualified=[index not in direct_false for index in range(count)],
        attitude_qualified=[index not in attitude_false for index in range(count)],
        paused=[index in paused for index in range(count)],
        backward=[index in backward for index in range(count)],
    )


def test_two_pass_anchor_selection_is_deterministic_and_uses_direct_long_edges() -> None:
    graph = _graph()
    calls: list[tuple[int, int]] = []

    def estimator(left: int, right: int) -> S12MotionEdge:
        calls.append((left, right))
        return _edge(left, right, confidence=0.92, kind="anchor_long")

    selection = select_automatic_anchors(
        graph, _evidence(graph), calibrated_cx=212.0,
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
        graph, _evidence(graph), calibrated_cx=100.0,
        long_edge_estimator=estimator,
    )

    assert selection.anchor_nodes[1] == 6


def test_anchor_rejects_missing_direct_long_edge_even_when_local_graph_is_connected() -> None:
    graph = _graph(8)

    def estimator(left: int, right: int) -> S12MotionEdge:
        return _edge(left, right, confidence=0.99, reliable=False, kind="anchor_long")

    with pytest.raises(S12MotionGraphError, match="no reliable direct anchor edge"):
        select_automatic_anchors(
            graph, _evidence(graph), calibrated_cx=100.0,
            long_edge_estimator=estimator,
        )


def test_anchor_candidate_must_have_direct_pose_quality_and_attitude_evidence() -> None:
    graph = _graph(15)
    evidence = _evidence(graph, direct_false=(6,))

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
        graph, _evidence(graph), calibrated_cx=212.0,
        long_edge_estimator=estimator,
        solver_config=S12MotionGraphConfig(maximum_reliable_edge_adjustment_px=0.1),
        anchor_config=S12AnchorConfig(),
    )

    assert all(audit.direct_path_difference_px == pytest.approx(0.0) for audit in selection.interval_audits)
    assert selection.final_solution.maximum_edge_adjustment_px < 1e-9


@pytest.mark.parametrize("qualification", ["paused", "backward"])
def test_paused_or_backward_node_cannot_be_anchor(qualification: str) -> None:
    graph = _graph(15)
    kwargs = {qualification: (6,)}

    selection = select_automatic_anchors(
        graph,
        _evidence(graph, **kwargs),
        calibrated_cx=100.0,
        long_edge_estimator=lambda left, right: _edge(
            left, right, confidence=0.9, kind="anchor_long"
        ),
    )

    assert 6 not in selection.anchor_nodes


def test_single_parallax_failure_does_not_mark_node_as_severe() -> None:
    graph = _graph(15)
    edges = list(graph.edges)
    edges.append(
        _edge(
            6,
            10,
            reliable=False,
            rejection_reasons=("excessive_parallax_clusters",),
        )
    )
    graph = S12MotionGraph(graph.nodes, tuple(edges))

    evidence = _evidence(graph)

    assert not evidence[6].severe_motion_clusters
    assert len(evidence[6].severe_motion_cluster_incident_edge_indices) == 1
    selection = select_automatic_anchors(
        graph,
        evidence,
        calibrated_cx=100.0,
        long_edge_estimator=lambda left, right: _edge(
            left, right, confidence=0.9, kind="anchor_long"
        ),
    )
    assert 6 in selection.anchor_nodes


def test_multiple_independent_parallax_incidents_mark_node_as_severe() -> None:
    graph = _graph(15)
    edges = list(graph.edges)
    edges.extend(
        (
            _edge(
                2,
                6,
                reliable=False,
                rejection_reasons=("excessive_parallax_clusters",),
            ),
            _edge(
                6,
                10,
                reliable=False,
                rejection_reasons=("excessive_parallax_clusters",),
            ),
        )
    )
    graph = S12MotionGraph(graph.nodes, tuple(edges))

    evidence = _evidence(graph)

    assert evidence[6].severe_motion_clusters
    assert "severe_motion_clusters" in evidence[6].rejection_reasons
    assert (
        "incident_edge:excessive_parallax_clusters"
        in evidence[6].rejection_reasons
    )
    selection = select_automatic_anchors(
        graph,
        evidence,
        calibrated_cx=100.0,
        long_edge_estimator=lambda left, right: _edge(
            left, right, confidence=0.9, kind="anchor_long"
        ),
    )
    assert 6 not in selection.anchor_nodes


def test_reliable_spatial_primary_prevents_secondary_only_node_rejection() -> None:
    graph = _graph(15)
    reliable_primary = replace(
        graph.edges[5], dominant_cluster_count=50, secondary_cluster_count=10
    )
    base_edges = (reliable_primary,) + graph.edges[:5] + graph.edges[6:]
    graph = S12MotionGraph(
        graph.nodes,
        base_edges
        + (
            _edge(
                2,
                6,
                reliable=False,
                rejection_reasons=("excessive_parallax_clusters",),
            ),
            _edge(
                6,
                10,
                reliable=False,
                rejection_reasons=("excessive_parallax_clusters",),
            ),
        ),
    )

    evidence = _evidence(graph)

    assert not evidence[6].severe_motion_clusters
    assert len(evidence[6].severe_motion_cluster_incident_edge_indices) == 2


def test_direct_long_and_independent_adjacent_only_path_are_accepted() -> None:
    nodes = tuple(_node(index) for index in range(8))
    graph = S12MotionGraph(
        nodes,
        tuple(_edge(index, index + 1) for index in range(len(nodes) - 1)),
    )

    selection = select_automatic_anchors(
        graph,
        _evidence(graph),
        calibrated_cx=100.0,
        long_edge_estimator=lambda left, right: _edge(
            left, right, confidence=0.9, kind="anchor_long"
        ),
    )

    assert selection.anchor_nodes == (0, 7)
    assert set(selection.interval_audits[0].local_path_composition) == {"adjacent"}


def test_direct_local_path_difference_is_a_hard_gate() -> None:
    graph = _graph(8)

    with pytest.raises(S12MotionGraphError, match="differs from independent local path"):
        select_automatic_anchors(
            graph,
            _evidence(graph),
            calibrated_cx=100.0,
            long_edge_estimator=lambda left, right: _edge(
                left, right, confidence=0.99, kind="anchor_long", dx=114.1
            ),
            anchor_config=S12AnchorConfig(
                maximum_direct_local_path_difference_px=2.0
            ),
        )


def test_interval_audit_reports_analysis_full_scale_and_weakest_local_edge() -> None:
    nodes = tuple(_node(index) for index in range(8))
    edges = [
        _edge(index, index + 1, confidence=0.71 if index == 3 else 0.9)
        for index in range(len(nodes) - 1)
    ]
    graph = S12MotionGraph(nodes, tuple(edges))

    selection = select_automatic_anchors(
        graph,
        _evidence(graph),
        calibrated_cx=100.0,
        long_edge_estimator=lambda left, right: _edge(
            left, right, confidence=0.91, kind="anchor_long", dx=113.0
        ),
        full_resolution_scale=4.0,
    )
    audit = selection.interval_audits[0]

    assert audit.direct_path_difference_px == pytest.approx(1.0)
    assert audit.direct_path_difference_full_resolution_px == pytest.approx(4.0)
    assert audit.direct_edge_confidence == pytest.approx(0.91)
    assert audit.local_path_weakest_confidence == pytest.approx(0.71)
    assert set(audit.local_path_composition) == {"adjacent"}


def test_endpoints_require_only_real_one_sided_local_support() -> None:
    nodes = tuple(_node(index) for index in range(5))
    graph = S12MotionGraph(
        nodes,
        tuple(_edge(index, index + 1) for index in range(len(nodes) - 1)),
    )
    evidence = _evidence(graph)

    assert evidence[0].reliable_local_left_count == 0
    assert evidence[0].reliable_local_right_count == 1
    assert evidence[-1].reliable_local_left_count == 1
    assert evidence[-1].reliable_local_right_count == 0
    selection = select_automatic_anchors(
        graph,
        evidence,
        calibrated_cx=100.0,
        long_edge_estimator=lambda left, right: _edge(
            left, right, confidence=0.9, kind="anchor_long"
        ),
    )
    assert selection.anchor_nodes == (0, 4)


def test_anchor_selection_is_deterministic_for_noncanonical_frame_ids() -> None:
    graph = _graph(15)

    def run() -> tuple[int, ...]:
        return select_automatic_anchors(
            graph,
            _evidence(graph),
            calibrated_cx=100.0,
            long_edge_estimator=lambda left, right: _edge(
                left, right, confidence=0.9, kind="anchor_long"
            ),
        ).anchor_nodes

    assert run() == run()
    assert all(node.frame_id >= 1000 for node in graph.nodes)
