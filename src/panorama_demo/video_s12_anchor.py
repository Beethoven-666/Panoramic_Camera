"""Deterministic two-pass automatic anchors for the S1.2 motion graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .video_s12_motion_graph import (
    S12HorizontalSolution,
    S12MotionEdge,
    S12MotionGraph,
    S12MotionGraphConfig,
    S12MotionGraphError,
    solve_horizontal_centers,
)


@dataclass(frozen=True)
class S12AnchorConfig:
    target_spacing_px: float = 96.0
    minimum_spacing_px: float = 64.0
    maximum_spacing_px: float = 144.0
    target_search_radius_px: float = 24.0
    minimum_anchor_edge_confidence: float = 0.70
    minimum_anchor_image_quality: float = 0.50

    def validated(self) -> "S12AnchorConfig":
        values = np.asarray(tuple(self.__dict__.values()), dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("anchor limits must be finite and positive")
        if not self.minimum_spacing_px <= self.target_spacing_px <= self.maximum_spacing_px:
            raise ValueError("anchor target spacing must be within its allowed range")
        if self.target_search_radius_px > self.maximum_spacing_px - self.minimum_spacing_px:
            raise ValueError("anchor target search radius is wider than the spacing range")
        if self.minimum_anchor_edge_confidence > 1.0 or self.minimum_anchor_image_quality > 1.0:
            raise ValueError("anchor confidence/quality must not exceed one")
        return self


@dataclass(frozen=True)
class S12AnchorEvidence:
    direct_pose_qualified: bool = True
    attitude_qualified: bool = True
    paused_or_backward: bool = False
    severe_motion_clusters: bool = False


@dataclass(frozen=True)
class S12AnchorIntervalAudit:
    left_node: int
    right_node: int
    direct_edge: S12MotionEdge
    local_path_edge_indices: tuple[int, ...]
    local_path_dx_px: float
    direct_path_difference_px: float


@dataclass(frozen=True)
class S12AnchorSelection:
    anchor_nodes: tuple[int, ...]
    long_edges: tuple[S12MotionEdge, ...]
    provisional_solution: S12HorizontalSolution
    final_solution: S12HorizontalSolution
    interval_audits: tuple[S12AnchorIntervalAudit, ...]

    def as_dict(self) -> dict[str, object]:
        frame_ids = (
            [self.long_edges[0].left_frame_id]
            + [edge.right_frame_id for edge in self.long_edges]
            if self.long_edges else []
        )
        return {
            "anchor_nodes": list(self.anchor_nodes),
            "anchor_frame_ids": frame_ids,
            "automatic": True,
            "external_absolute_coordinates": False,
            "direct_long_edge_count": len(self.long_edges),
            "two_independent_paths_each_interval": all(
                bool(item.local_path_edge_indices) and item.direct_edge.reliable
                for item in self.interval_audits
            ),
            "intervals": [
                {
                    "left_node": item.left_node,
                    "right_node": item.right_node,
                    "direct_edge_confidence": item.direct_edge.confidence,
                    "local_path_edge_indices": list(item.local_path_edge_indices),
                    "local_path_dx_px": item.local_path_dx_px,
                    "direct_path_difference_px": item.direct_path_difference_px,
                }
                for item in self.interval_audits
            ],
        }


LongEdgeEstimator = Callable[[int, int], S12MotionEdge]


def _local_admissible(graph: S12MotionGraph, node: int, evidence: S12AnchorEvidence, minimum_quality: float) -> bool:
    if not (
        evidence.direct_pose_qualified
        and evidence.attitude_qualified
        and not evidence.paused_or_backward
        and not evidence.severe_motion_clusters
        and graph.nodes[node].image_quality >= minimum_quality
    ):
        return False
    adjacent = [
        edge for edge in graph.reliable_edges
        if edge.kind == "adjacent" and node in (edge.left_node, edge.right_node)
    ]
    skip = [
        edge for edge in graph.reliable_edges
        if edge.kind == "skip" and node in (edge.left_node, edge.right_node)
    ]
    required_adjacent = 1 if node in {0, len(graph.nodes) - 1} else 2
    return len(adjacent) >= required_adjacent and bool(skip)


def _local_path(
    graph: S12MotionGraph, left: int, right: int
) -> tuple[tuple[int, ...], float] | None:
    """Deterministic shortest reliable local path, excluding anchor-long edges."""

    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in graph.nodes]
    for edge_index, edge in enumerate(graph.edges):
        if not edge.reliable or edge.kind == "anchor_long":
            continue
        adjacency[edge.left_node].append((edge.right_node, edge_index, edge.measured_dx_px))
        adjacency[edge.right_node].append((edge.left_node, edge_index, -edge.measured_dx_px))
    queue: list[tuple[int, tuple[int, ...], float]] = [(left, (), 0.0)]
    visited = {left}
    while queue:
        node, path, dx = queue.pop(0)
        if node == right:
            return path, dx
        for neighbour, edge_index, signed_dx in sorted(adjacency[node]):
            if neighbour not in visited and left <= neighbour <= right:
                visited.add(neighbour)
                queue.append((neighbour, path + (edge_index,), dx + signed_dx))
    return None


def select_automatic_anchors(
    graph: S12MotionGraph,
    evidence: Sequence[S12AnchorEvidence],
    *,
    calibrated_cx: float,
    long_edge_estimator: LongEdgeEstimator,
    anchor_config: S12AnchorConfig | None = None,
    solver_config: S12MotionGraphConfig | None = None,
) -> S12AnchorSelection:
    """Select anchors from provisional progress, then add direct long edges.

    The estimator is invoked for actual endpoint images.  A caller cannot pass
    composed local motion as a long edge because :class:`S12MotionEdge`
    requires ``kind='anchor_long'`` and ``direct_image_match=True``.
    """

    settings = (anchor_config or S12AnchorConfig()).validated()
    if len(evidence) != len(graph.nodes):
        raise ValueError("anchor evidence must cover every motion graph node")
    graph.assert_reliable_connected()
    provisional = solve_horizontal_centers(
        graph, calibrated_cx=calibrated_cx, config=solver_config
    )
    centers = provisional.centers_px
    eligible = {
        index for index, item in enumerate(evidence)
        if _local_admissible(graph, index, item, settings.minimum_anchor_image_quality)
    }
    endpoints = {0, len(graph.nodes) - 1}
    if not endpoints.issubset(eligible):
        raise S12MotionGraphError(
            "first/last scan sources do not satisfy mandatory anchor qualification"
        )
    anchors = [0]
    long_edges: list[S12MotionEdge] = []
    audits: list[S12AnchorIntervalAudit] = []
    last = len(graph.nodes) - 1
    while anchors[-1] != last:
        left = anchors[-1]
        remaining = centers[last] - centers[left]
        if remaining <= settings.maximum_spacing_px:
            candidates = [last]
        else:
            candidates = [
                node for node in sorted(eligible)
                if node > left
                and settings.minimum_spacing_px <= centers[node] - centers[left] <= settings.maximum_spacing_px
                and abs((centers[node] - centers[left]) - settings.target_spacing_px)
                    <= settings.target_search_radius_px
            ]
            if not candidates:
                # Deterministic shortening fallback still uses a real eligible
                # source and a direct match; it never interpolates an anchor.
                candidates = [
                    node for node in sorted(eligible)
                    if node > left
                    and settings.minimum_spacing_px <= centers[node] - centers[left] <= settings.maximum_spacing_px
                ]
        assessed: list[tuple[tuple[object, ...], int, S12MotionEdge, tuple[int, ...], float]] = []
        for node in candidates:
            path = _local_path(graph, left, node)
            if path is None:
                continue
            direct = long_edge_estimator(left, node)
            if (
                direct.left_node != left or direct.right_node != node
                or direct.kind != "anchor_long" or not direct.direct_image_match
            ):
                raise S12MotionGraphError("anchor estimator did not return its direct endpoint edge")
            path_indices, path_dx = path
            weakest = min(graph.edges[index].confidence for index in path_indices)
            rank = (
                not (direct.reliable and direct.confidence >= settings.minimum_anchor_edge_confidence),
                not bool(path_indices),
                -direct.confidence,
                -weakest,
                -graph.nodes[node].image_quality,
                abs((centers[node] - centers[left]) - settings.target_spacing_px),
                graph.nodes[node].frame_id,
            )
            assessed.append((rank, node, direct, path_indices, path_dx))
        assessed.sort(key=lambda item: item[0])
        chosen = next(
            (item for item in assessed if item[2].reliable and item[2].confidence >= settings.minimum_anchor_edge_confidence),
            None,
        )
        if chosen is None:
            raise S12MotionGraphError(
                f"no reliable direct anchor edge and independent local path after node {left}"
            )
        _rank, node, direct, path_indices, path_dx = chosen
        anchors.append(node)
        long_edges.append(direct)
        audits.append(
            S12AnchorIntervalAudit(
                left, node, direct, path_indices, path_dx,
                float(direct.measured_dx_px - path_dx),
            )
        )
    augmented = graph.with_edges(long_edges)
    final = solve_horizontal_centers(
        augmented, calibrated_cx=calibrated_cx, config=solver_config
    )
    return S12AnchorSelection(
        tuple(anchors), tuple(long_edges), provisional, final, tuple(audits)
    )
