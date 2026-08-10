from __future__ import annotations

import cv2
import numpy as np
import pytest
import panorama_demo.video_s12_motion_graph as motion_graph_module

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


def _node(
    index: int,
    image: np.ndarray | None = None,
    *,
    frame_id: int | None = None,
    timestamp_ms: float | None = None,
) -> S12MotionNode:
    return S12MotionNode(
        index,
        100 + index if frame_id is None else frame_id,
        _image() if image is None else image,
        index * 0.01,
        timestamp_ms=timestamp_ms,
    )


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


def test_local_edge_over_64_frame_gap_remains_rejected() -> None:
    with pytest.raises(ValueError, match="local motion endpoints exceed maximum frame gap"):
        estimate_direct_motion_edge(
            _node(0, frame_id=10),
            _node(1, frame_id=75),
            kind="skip",
            expected_image_motion_sign=1,
        )


def test_anchor_long_over_64_frames_uses_independent_direct_match() -> None:
    image = _image()
    shifted = cv2.warpAffine(
        image, np.float32([[1.0, 0.0, 5.0], [0.0, 1.0, 0.0]]), (424, 240)
    )
    edge = estimate_direct_motion_edge(
        _node(0, image, frame_id=54, timestamp_ms=1000.0),
        _node(1, shifted, frame_id=170, timestamp_ms=2000.0),
        kind="anchor_long",
        expected_image_motion_sign=1,
        provisional_spacing_px=64.0,
        config=S12MotionGraphConfig(
            minimum_inlier_count=12,
            minimum_grid_coverage=0.15,
            minimum_vertical_span_fraction=0.30,
        ),
    )

    assert edge.frame_gap == 116
    assert edge.kind == "anchor_long"
    assert edge.direct_image_match is True
    assert edge.reliable is True
    assert edge.measured_dx_px == pytest.approx(5.0, abs=0.2)


def test_anchor_long_cannot_be_fabricated_from_composed_local_motion() -> None:
    with pytest.raises(ValueError, match="forbids composed or synthetic"):
        S12MotionEdge(
            0, 2, 10, 170, 64.0, 0.0, 32, 0.5, 0.5, 0.5,
            0.1, 0.1, 0.1, True, 0.8, True, (), "anchor_long", False,
        )


def test_anchor_long_rejects_anomalous_timestamp_gap() -> None:
    edge = estimate_direct_motion_edge(
        _node(0, frame_id=54, timestamp_ms=1000.0),
        _node(1, frame_id=170, timestamp_ms=1000.0),
        kind="anchor_long",
        expected_image_motion_sign=1,
        provisional_spacing_px=64.0,
    )

    assert edge.reliable is False
    assert "anchor_long_timestamp_gap_out_of_range" in edge.rejection_reasons
    assert edge.timestamp_gap_ms == 0.0


def test_anchor_long_rejects_insufficient_predicted_overlap() -> None:
    edge = estimate_direct_motion_edge(
        _node(0, frame_id=54, timestamp_ms=1000.0),
        _node(1, frame_id=170, timestamp_ms=2000.0),
        kind="anchor_long",
        expected_image_motion_sign=1,
        provisional_spacing_px=144.0,
        config=S12MotionGraphConfig(
            anchor_long_minimum_predicted_overlap_fraction=0.80,
        ),
    )

    assert edge.predicted_overlap_fraction == pytest.approx(1.0 - 144.0 / 424.0)
    assert "anchor_long_predicted_overlap_insufficient" in edge.rejection_reasons


def test_anchor_long_audits_provisional_spacing_and_pose_direction() -> None:
    left = _node(0, frame_id=54, timestamp_ms=1000.0)
    right = S12MotionNode(
        1, 170, _image(), -0.01, timestamp_ms=2000.0,
    )
    edge = estimate_direct_motion_edge(
        left,
        right,
        kind="anchor_long",
        expected_image_motion_sign=1,
        provisional_spacing_px=32.0,
    )

    assert "anchor_long_provisional_spacing_out_of_range" in edge.rejection_reasons
    assert "pose_direction_disagreement" in edge.rejection_reasons
    assert edge.pose_direction_agreement is False


def test_local_and_anchor_long_use_independently_configured_lk_matchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, int], int]] = []

    def failed_lk(
        *_args: object, **kwargs: object,
    ) -> tuple[None, None, None]:
        calls.append((kwargs["winSize"], kwargs["maxLevel"]))
        return None, None, None

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", failed_lk)
    config = S12MotionGraphConfig(
        local_edge_lk_window_size=23,
        local_edge_lk_max_level=2,
        anchor_long_lk_window_size=33,
        anchor_long_lk_max_level=5,
    )
    estimate_direct_motion_edge(
        _node(0), _node(1), expected_image_motion_sign=1, config=config,
    )
    estimate_direct_motion_edge(
        _node(0, frame_id=54, timestamp_ms=1000.0),
        _node(1, frame_id=170, timestamp_ms=2000.0),
        kind="anchor_long",
        expected_image_motion_sign=1,
        provisional_spacing_px=64.0,
        config=config,
    )

    assert calls == [((23, 23), 2), ((33, 33), 5)]


def test_diagnostic_counts_ratios_residuals_and_motion_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.asarray(
        [[[10.0 + 18.0 * (index // 2), 20.0 + 35.0 * (index % 2)]] for index in range(20)],
        dtype=np.float32,
    )
    forward = points.copy()
    forward[:10, 0, 0] += 5.0
    forward[10:, 0, 0] += 9.0
    backward = points.copy()
    backward[15:17, 0, 0] += 2.0
    status_forward = np.zeros((20, 1), dtype=np.uint8)
    status_forward[:18] = 1
    status_backward = np.zeros((20, 1), dtype=np.uint8)
    status_backward[:17] = 1
    calls = 0

    def fake_lk(*_args: object, **_kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return forward.copy(), status_forward.copy(), None
        return backward.copy(), status_backward.copy(), None

    monkeypatch.setattr(motion_graph_module, "_grid_features", lambda _gray: points.copy())
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    edge = estimate_direct_motion_edge(
        _node(0),
        _node(1),
        expected_image_motion_sign=1,
            config=S12MotionGraphConfig(
                minimum_inlier_count=4,
                minimum_model_inlier_count=4,
                minimum_grid_coverage=0.01,
            minimum_vertical_span_fraction=0.01,
        ),
    )

    assert edge.detected_feature_count == 20
    assert edge.forward_lk_valid_count == 18
    assert edge.backward_lk_valid_count == 17
    assert edge.fb_retained_count == 15
    assert edge.model_inlier_count == 10
    assert edge.dominant_cluster_count == 10
    assert edge.secondary_cluster_count == 5
    assert edge.fb_retention_ratio == pytest.approx(15 / 20)
    assert edge.model_inlier_ratio_retained == pytest.approx(10 / 15)
    assert edge.effective_inlier_ratio_detected == pytest.approx(10 / 20)
    assert edge.inlier_ratio == edge.effective_inlier_ratio_detected
    assert edge.parallax_cluster_ratio == pytest.approx(5 / 15)
    assert edge.forward_backward_p50_px == 0.0
    assert edge.forward_backward_p95_px == 0.0
    assert edge.model_residual_p50_px == pytest.approx(0.0)
    assert edge.model_residual_p95_px == pytest.approx(4.0)
    assert edge.model_residual_max_px == pytest.approx(4.0)
    assert edge.reliable is True
    assert edge.measured_dx_px == pytest.approx(5.0)


@pytest.mark.parametrize("kind", ["adjacent", "anchor_long"])
def test_split_effective_support_gate_is_identical_for_local_and_direct_long(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    points = np.asarray(
        [[[12.0 + 18.0 * (index // 2), 20.0 + 35.0 * (index % 2)]] for index in range(20)],
        dtype=np.float32,
    )
    forward = points.copy()
    forward[:10, 0, 0] += 5.0
    forward[10:, 0, 0] += 9.0
    calls = 0

    def fake_lk(*_args: object, **_kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return forward.copy(), np.ones((20, 1), dtype=np.uint8), None
        return points.copy(), np.ones((20, 1), dtype=np.uint8), None

    monkeypatch.setattr(motion_graph_module, "_grid_features", lambda _gray: points.copy())
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_lk)
    edge = estimate_direct_motion_edge(
        _node(0, timestamp_ms=1000.0),
        _node(1, timestamp_ms=1100.0),
        kind=kind,  # type: ignore[arg-type]
        expected_image_motion_sign=1,
        provisional_spacing_px=64.0 if kind == "anchor_long" else None,
        config=S12MotionGraphConfig(
            minimum_model_inlier_count=4,
            minimum_effective_inlier_ratio_detected=0.75,
            minimum_grid_coverage=0.01,
            minimum_vertical_span_fraction=0.01,
            maximum_parallax_cluster_ratio=0.99,
            anchor_long_minimum_spacing_px=1.0,
        ),
    )

    assert edge.reliable is False
    assert "insufficient_effective_inlier_ratio_detected" in edge.rejection_reasons
