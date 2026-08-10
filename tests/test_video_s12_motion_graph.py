from __future__ import annotations

import cv2
import numpy as np
import pytest

from panorama_demo.video_s12_motion_graph import (
    S12MotionEdge,
    S12MotionGraph,
    S12MotionGraphConfig,
    S12MotionGraphError,
    S12MotionNode,
    build_motion_graph,
    calibrate_analysis_image,
    estimate_direct_motion_edge,
    solve_horizontal_centers,
)


def _image(seed: int = 12012) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((240, 424), dtype=np.uint8)
    for x, y, value in zip(
        rng.integers(8, 416, 500),
        rng.integers(8, 232, 500),
        rng.integers(80, 256, 500),
        strict=True,
    ):
        cv2.circle(image, (int(x), int(y)), 2, int(value), -1)
    return image


def _node(index: int, image: np.ndarray | None = None) -> S12MotionNode:
    return S12MotionNode(index, 100 + index, _image() if image is None else image, index * 0.01)


def _edge(left: int, right: int, dx: float, *, reliable: bool = True, kind: str = "adjacent") -> S12MotionEdge:
    return S12MotionEdge(
        left, right, 100 + left, 100 + right, dx, 0.0, 64, 0.9, 0.8, 0.8,
        0.1, 0.0, (right - left) * 0.01, True, 0.9, reliable,
        () if reliable else ("fixture_rejected",), kind, True,
    )


def test_direct_edge_uses_two_calibrated_real_images_and_preserves_measurement() -> None:
    left_image = _image()
    transform = np.float32([[1.0, 0.0, 5.25], [0.0, 1.0, -1.5]])
    right_image = cv2.warpAffine(left_image, transform, (424, 240))
    config = S12MotionGraphConfig(
        minimum_inlier_count=12,
        minimum_grid_coverage=0.15,
        minimum_vertical_span_fraction=0.30,
    )

    edge = estimate_direct_motion_edge(
        _node(0, left_image), _node(1, right_image),
        expected_image_motion_sign=1, config=config,
    )

    assert edge.direct_image_match is True
    assert edge.reliable is True
    assert edge.measured_dx_px == pytest.approx(5.25, abs=0.15)
    assert edge.measured_dy_px == pytest.approx(-1.5, abs=0.15)
    assert edge.rejection_reasons == ()


def test_calibration_returns_explicit_scaled_target_domain() -> None:
    bgr = cv2.cvtColor(_image(), cv2.COLOR_GRAY2BGR)
    matrix = np.asarray([[400.0, 0.0, 212.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])

    gray, target = calibrate_analysis_image(
        bgr, matrix, np.zeros(5), analysis_width_px=212,
    )

    assert gray.shape == (120, 212)
    assert target[0, 0] == pytest.approx(200.0)
    assert target[0, 2] == pytest.approx(106.0)


def test_graph_fails_closed_when_reliable_direct_edges_do_not_connect_all_nodes() -> None:
    nodes = tuple(_node(index) for index in range(3))
    graph = S12MotionGraph(nodes, (_edge(0, 1, 4.0), _edge(1, 2, 4.0, reliable=False)))

    with pytest.raises(S12MotionGraphError, match="disconnected"):
        graph.assert_reliable_connected()


def test_first_pass_can_return_disconnected_direct_evidence_for_scan_filtering() -> None:
    blank = np.zeros((240, 424), dtype=np.uint8)
    nodes = tuple(_node(index, blank) for index in range(4))
    graph = build_motion_graph(
        nodes,
        expected_image_motion_sign=1,
        config=S12MotionGraphConfig(),
        require_connected=False,
    )
    assert graph.reliable_edges == ()


def test_positive_increment_irls_uses_raw_adjacent_skip_and_long_measurements() -> None:
    nodes = tuple(_node(index) for index in range(4))
    edges = (
        _edge(0, 1, 3.0), _edge(1, 2, 5.0), _edge(2, 3, 2.0),
        _edge(0, 2, 8.0, kind="skip"), _edge(1, 3, 7.0, kind="skip"),
        _edge(0, 3, 10.0, kind="anchor_long"),
    )
    graph = S12MotionGraph(nodes, edges)

    result = solve_horizontal_centers(graph, calibrated_cx=212.0)

    assert result.increments_px == pytest.approx((3.0, 5.0, 2.0), abs=1e-9)
    assert result.centers_px == pytest.approx((212.0, 215.0, 220.0, 222.0), abs=1e-9)
    assert result.reliable_edges == edges
    assert result.as_dict()["normalization_applied"] is False
    assert result.as_dict()["measurement_filter_applied"] is False


def test_active_set_enforces_minimum_increment_without_silent_clipping() -> None:
    nodes = tuple(_node(index) for index in range(3))
    graph = S12MotionGraph(nodes, (_edge(0, 1, 0.05), _edge(1, 2, 1.0), _edge(0, 2, 1.05, kind="skip")))
    config = S12MotionGraphConfig(maximum_reliable_edge_adjustment_px=1.0)

    result = solve_horizontal_centers(graph, calibrated_cx=10.0, config=config)

    assert result.increments_px[0] == pytest.approx(0.2)
    assert result.active_lower_bound_indices == (0,)
    assert result.edge_residuals_px[0] > 0.0


def test_builder_measures_every_candidate_pair_directly() -> None:
    base = _image()
    images = [cv2.warpAffine(base, np.float32([[1, 0, 2 * i], [0, 1, 0]]), (424, 240)) for i in range(5)]
    nodes = tuple(_node(index, image) for index, image in enumerate(images))
    config = S12MotionGraphConfig(
        minimum_inlier_count=12, minimum_grid_coverage=0.15,
        minimum_vertical_span_fraction=0.30,
    )

    graph = build_motion_graph(nodes, expected_image_motion_sign=1, config=config)

    assert len(graph.edges) == 4 + 3 + 1
    assert all(edge.direct_image_match for edge in graph.edges)
    assert {(edge.right_node - edge.left_node) for edge in graph.edges} == {1, 2, 4}
