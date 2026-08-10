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
    maximum_direct_local_path_difference_px: float = 2.0

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
    direct_pose_qualified: bool
    attitude_qualified: bool
    paused: bool
    backward: bool
    severe_motion_clusters: bool
    image_quality: float
    reliable_local_edge_count: int
    reliable_local_left_count: int
    reliable_local_right_count: int
    reliable_local_edge_indices: tuple[int, ...]
    severe_motion_cluster_incident_edge_indices: tuple[int, ...]
    rejection_reasons: tuple[str, ...]

    @property
    def paused_or_backward(self) -> bool:
        return self.paused or self.backward

    def as_dict(self) -> dict[str, object]:
        return {
            name: list(value) if isinstance(value, tuple) else value
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class S12AnchorIntervalAudit:
    left_node: int
    right_node: int
    direct_edge: S12MotionEdge
    local_path_edge_indices: tuple[int, ...]
    local_path_dx_px: float
    direct_path_difference_px: float
    direct_path_difference_full_resolution_px: float
    direct_edge_confidence: float
    local_path_weakest_confidence: float
    local_path_composition: tuple[str, ...]


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
                    "local_path_composition": list(item.local_path_composition),
                    "local_path_weakest_confidence": item.local_path_weakest_confidence,
                    "local_path_dx_px": item.local_path_dx_px,
                    "direct_path_difference_px": item.direct_path_difference_px,
                    "direct_path_difference_analysis_px": item.direct_path_difference_px,
                    "direct_path_difference_full_resolution_px": (
                        item.direct_path_difference_full_resolution_px
                    ),
                }
                for item in self.interval_audits
            ],
        }


LongEdgeEstimator = Callable[[int, int], S12MotionEdge]


def _severe_motion_cluster_incidents(
    graph: S12MotionGraph, node: int
) -> tuple[int, ...]:
    """Return independent incident edges reporting severe multi-motion evidence.

    A single failed edge is deliberately only diagnostic evidence.  Two or
    more distinct real-image incident edges must agree before the node is
    classified as severe.  ``getattr`` keeps this usable while the expanded
    motion-edge diagnostics are introduced.
    """

    incidents: list[int] = []
    neighbours: set[int] = set()
    for edge_index, edge in enumerate(graph.edges):
        if node not in (edge.left_node, edge.right_node):
            continue
        reasons = set(getattr(edge, "rejection_reasons", ()))
        if "excessive_parallax_clusters" not in reasons:
            continue
        neighbour = edge.right_node if edge.left_node == node else edge.left_node
        if neighbour in neighbours:
            continue
        neighbours.add(neighbour)
        incidents.append(edge_index)
    return tuple(incidents)


def _has_spatially_covered_primary_motion(graph: S12MotionGraph, node: int) -> bool:
    """Whether direct local evidence preserves a qualified primary motion."""

    return any(
        edge.reliable
        and edge.kind in {"adjacent", "skip"}
        and node in (edge.left_node, edge.right_node)
        and edge.grid_coverage > 0.0
        and edge.dominant_cluster_count > 0
        and edge.dominant_cluster_count >= edge.secondary_cluster_count
        for edge in graph.edges
    )


def build_anchor_evidence(
    graph: S12MotionGraph,
    *,
    direct_pose_qualified: Sequence[bool],
    attitude_qualified: Sequence[bool],
    paused: Sequence[bool],
    backward: Sequence[bool],
    image_quality: Sequence[float] | None = None,
    minimum_severe_motion_cluster_incidents: int = 2,
) -> tuple[S12AnchorEvidence, ...]:
    """Build node evidence from trajectory, scan, image, and direct edges.

    The four qualification sequences are intentionally mandatory: callers
    cannot silently manufacture permissive evidence for every node.
    """

    count = len(graph.nodes)
    inputs = (direct_pose_qualified, attitude_qualified, paused, backward)
    if any(len(values) != count for values in inputs):
        raise ValueError("anchor qualification evidence must cover every node")
    if minimum_severe_motion_cluster_incidents < 2:
        raise ValueError("severe multi-motion classification requires at least two incidents")
    qualities = (
        tuple(float(node.image_quality) for node in graph.nodes)
        if image_quality is None
        else tuple(float(value) for value in image_quality)
    )
    if len(qualities) != count or not np.isfinite(qualities).all():
        raise ValueError("anchor image quality must be finite and cover every node")
    if any(not 0.0 <= value <= 1.0 for value in qualities):
        raise ValueError("anchor image quality must be within zero and one")

    result: list[S12AnchorEvidence] = []
    for node in range(count):
        reliable_indices = tuple(
            edge_index
            for edge_index, edge in enumerate(graph.edges)
            if edge.reliable
            and edge.kind in {"adjacent", "skip"}
            and node in (edge.left_node, edge.right_node)
        )
        left_count = sum(graph.edges[index].right_node == node for index in reliable_indices)
        right_count = sum(graph.edges[index].left_node == node for index in reliable_indices)
        severe_incidents = _severe_motion_cluster_incidents(graph, node)
        severe_motion_clusters = bool(
            len(severe_incidents) >= minimum_severe_motion_cluster_incidents
            and not _has_spatially_covered_primary_motion(graph, node)
        )
        reasons: list[str] = []
        if not direct_pose_qualified[node]:
            reasons.append("direct_pose_not_qualified")
        if not attitude_qualified[node]:
            reasons.append("attitude_not_qualified")
        if paused[node]:
            reasons.append("paused")
        if backward[node]:
            reasons.append("backward")
        if severe_motion_clusters:
            reasons.append("severe_motion_clusters")
        if not reliable_indices:
            reasons.append("no_reliable_local_support")
        elif node > 0 and left_count == 0:
            reasons.append("missing_reliable_local_left_support")
        elif node < count - 1 and right_count == 0:
            reasons.append("missing_reliable_local_right_support")
        incident_reasons = {
            reason
            for edge in graph.edges
            if node in (edge.left_node, edge.right_node)
            for reason in getattr(edge, "rejection_reasons", ())
        }
        reasons.extend(f"incident_edge:{reason}" for reason in sorted(incident_reasons))
        result.append(
            S12AnchorEvidence(
                direct_pose_qualified=bool(direct_pose_qualified[node]),
                attitude_qualified=bool(attitude_qualified[node]),
                paused=bool(paused[node]),
                backward=bool(backward[node]),
                severe_motion_clusters=severe_motion_clusters,
                image_quality=qualities[node],
                reliable_local_edge_count=len(reliable_indices),
                reliable_local_left_count=left_count,
                reliable_local_right_count=right_count,
                reliable_local_edge_indices=reliable_indices,
                severe_motion_cluster_incident_edge_indices=severe_incidents,
                rejection_reasons=tuple(reasons),
            )
        )
    return tuple(result)


def _local_admissible(
    graph: S12MotionGraph,
    node: int,
    evidence: S12AnchorEvidence,
    minimum_quality: float,
) -> bool:
    if not (
        evidence.direct_pose_qualified
        and evidence.attitude_qualified
        and not evidence.paused_or_backward
        and not evidence.severe_motion_clusters
        and evidence.image_quality >= minimum_quality
    ):
        return False
    if evidence.reliable_local_edge_count <= 0:
        return False
    if node == 0:
        return evidence.reliable_local_right_count > 0
    if node == len(graph.nodes) - 1:
        return evidence.reliable_local_left_count > 0
    return (
        evidence.reliable_local_left_count > 0
        and evidence.reliable_local_right_count > 0
    )


def _local_path(
    graph: S12MotionGraph, left: int, right: int
) -> tuple[tuple[int, ...], float] | None:
    """Deterministic shortest reliable local path, excluding anchor-long edges."""

    adjacency: list[list[tuple[int, int, float]]] = [[] for _ in graph.nodes]
    for edge_index, edge in enumerate(graph.edges):
        if not edge.reliable or edge.kind not in {"adjacent", "skip"}:
            continue
        adjacency[edge.left_node].append((edge.right_node, edge_index, edge.measured_dx_px))
    paths: dict[int, list[tuple[tuple[int, ...], float, float]]] = {
        left: [((), 0.0, 1.0)]
    }
    for node in range(left, right):
        for path, dx, weakest in paths.get(node, ()):
            for neighbour, edge_index, signed_dx in sorted(adjacency[node]):
                if node < neighbour <= right:
                    edge = graph.edges[edge_index]
                    paths.setdefault(neighbour, []).append(
                        (path + (edge_index,), dx + signed_dx, min(weakest, edge.confidence))
                    )
        for neighbour, _edge_index, _signed_dx in adjacency[node]:
            candidates = paths.get(neighbour)
            if candidates and len(candidates) > 256:
                candidates.sort(
                    key=lambda item: (
                        -item[2],
                        len(item[0]),
                        item[0],
                    )
                )
                paths[neighbour] = candidates[:256]
    candidates = paths.get(right, [])
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -item[2],
            len(item[0]),
            item[0],
        )
    )
    return candidates[0][0], candidates[0][1]


def select_automatic_anchors(
    graph: S12MotionGraph,
    evidence: Sequence[S12AnchorEvidence],
    *,
    calibrated_cx: float,
    long_edge_estimator: LongEdgeEstimator,
    anchor_config: S12AnchorConfig | None = None,
    solver_config: S12MotionGraphConfig | None = None,
    full_resolution_scale: float = 1.0,
) -> S12AnchorSelection:
    """Select anchors from provisional progress, then add direct long edges.

    The estimator is invoked for actual endpoint images.  A caller cannot pass
    composed local motion as a long edge because :class:`S12MotionEdge`
    requires ``kind='anchor_long'`` and ``direct_image_match=True``.
    """

    settings = (anchor_config or S12AnchorConfig()).validated()
    if not np.isfinite(full_resolution_scale) or full_resolution_scale <= 0.0:
        raise ValueError("full_resolution_scale must be finite and positive")
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
                and settings.minimum_spacing_px
                    <= centers[node] - centers[left] <= settings.maximum_spacing_px
            ]
        assessed: list[
            tuple[tuple[object, ...], int, S12MotionEdge, tuple[int, ...], float, float]
        ] = []
        for node in candidates:
            path = _local_path(graph, left, node)
            if path is None:
                continue
            try:
                direct = long_edge_estimator(left, node)
            except (ValueError, S12MotionGraphError):
                # Candidate-level structural rejection is not permission to
                # abandon the other real endpoint images in the search band.
                continue
            if (
                direct.left_node != left or direct.right_node != node
                or direct.kind != "anchor_long" or not direct.direct_image_match
            ):
                raise S12MotionGraphError("anchor estimator did not return its direct endpoint edge")
            path_indices, path_dx = path
            weakest = min(graph.edges[index].confidence for index in path_indices)
            difference = abs(float(direct.measured_dx_px - path_dx))
            rank = (
                not (direct.reliable and direct.confidence >= settings.minimum_anchor_edge_confidence),
                not bool(path_indices),
                difference > settings.maximum_direct_local_path_difference_px,
                -direct.confidence,
                -weakest,
                -evidence[node].image_quality,
                abs((centers[node] - centers[left]) - settings.target_spacing_px),
                graph.nodes[node].frame_id,
            )
            assessed.append((rank, node, direct, path_indices, path_dx, difference))
        assessed.sort(key=lambda item: item[0])
        chosen = next(
            (
                item
                for item in assessed
                if item[2].reliable
                and item[2].confidence >= settings.minimum_anchor_edge_confidence
                and item[5] <= settings.maximum_direct_local_path_difference_px
            ),
            None,
        )
        if chosen is None:
            reliable_differences = [
                item[5]
                for item in assessed
                if item[2].reliable
                and item[2].confidence >= settings.minimum_anchor_edge_confidence
            ]
            if reliable_differences:
                raise S12MotionGraphError(
                    "direct anchor edge differs from independent local path by "
                    f"{min(reliable_differences):.6g}px, exceeding "
                    f"{settings.maximum_direct_local_path_difference_px:.6g}px "
                    f"after node {left}"
                )
            raise S12MotionGraphError(
                f"no reliable direct anchor edge and independent local path after node {left}"
            )
        _rank, node, direct, path_indices, path_dx, difference = chosen
        anchors.append(node)
        long_edges.append(direct)
        audits.append(
            S12AnchorIntervalAudit(
                left,
                node,
                direct,
                path_indices,
                path_dx,
                difference,
                difference * full_resolution_scale,
                direct.confidence,
                min(graph.edges[index].confidence for index in path_indices),
                tuple(graph.edges[index].kind for index in path_indices),
            )
        )
    augmented = graph.with_edges(long_edges)
    final = solve_horizontal_centers(
        augmented, calibrated_cx=calibrated_cx, config=solver_config
    )
    return S12AnchorSelection(
        tuple(anchors), tuple(long_edges), provisional, final, tuple(audits)
    )
