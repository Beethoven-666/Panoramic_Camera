"""Direct calibrated-image motion graph and horizontal solver for S1.2.

The module deliberately accepts images, never composed motion.  Every edge
therefore identifies the two real frames which were matched.  Reliable
measurements are kept verbatim; rejection removes an edge from the solve
instead of filtering, interpolating, extrapolating, clipping, or normalising
it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np


class S12MotionGraphError(RuntimeError):
    """Structural failure in direct motion evidence or graph geometry."""


@dataclass(frozen=True)
class S12MotionGraphConfig:
    analysis_width_px: int = 424
    lk_forward_backward_max_px: float = 0.75
    ransac_residual_px: float = 1.5
    minimum_inlier_count: int = 16
    minimum_inlier_ratio: float = 0.45
    minimum_model_inlier_count: int = 16
    minimum_fb_retention_ratio: float = 0.45
    minimum_model_inlier_ratio_retained: float = 0.45
    minimum_effective_inlier_ratio_detected: float = 0.25
    minimum_grid_coverage: float = 0.25
    minimum_vertical_span_fraction: float = 0.40
    maximum_parallax_cluster_ratio: float = 0.45
    minimum_edge_confidence: float = 0.0
    adjacent_candidate_steps: tuple[int, ...] = (1,)
    skip_candidate_steps: tuple[int, ...] = (2, 4)
    local_edge_maximum_frame_gap: int = 64
    local_edge_lk_window_size: int = 21
    local_edge_lk_max_level: int = 3
    anchor_long_use_local_frame_gap: bool = False
    anchor_long_maximum_frame_gap: int = 256
    anchor_long_minimum_spacing_px: float = 64.0
    anchor_long_maximum_spacing_px: float = 144.0
    anchor_long_require_timestamp_audit: bool = True
    anchor_long_minimum_timestamp_gap_ms: float = 0.001
    anchor_long_maximum_timestamp_gap_ms: float = 10_000.0
    anchor_long_require_predicted_overlap: bool = True
    anchor_long_minimum_predicted_overlap_fraction: float = 0.50
    anchor_long_require_direct_image_match: bool = True
    anchor_long_use_provisional_initial_flow: bool = False
    anchor_long_initial_flow_source: str = "none"
    anchor_long_feature_domain: str = "full_calibrated_frame"
    anchor_long_feature_slit_width_px: int = 48
    anchor_long_preprocessing: str = "gray"
    anchor_long_lk_window_size: int = 31
    anchor_long_lk_max_level: int = 4
    # Deprecated input-only alias retained for old callers. New runner wiring
    # must use ``local_edge_maximum_frame_gap`` explicitly.
    maximum_direct_edge_frame_gap: int | None = None
    huber_delta_px: float = 1.5
    irls_iterations: int = 4
    minimum_increment_px: float = 0.20
    maximum_reliable_edge_adjustment_px: float = 2.0

    def validated(self) -> "S12MotionGraphConfig":
        if self.analysis_width_px < 64 or self.analysis_width_px > 4096:
            raise ValueError("analysis_width_px must be within 64..4096")
        finite = np.asarray(
            (
                self.lk_forward_backward_max_px,
                self.ransac_residual_px,
                self.minimum_inlier_ratio,
                self.minimum_grid_coverage,
                self.minimum_vertical_span_fraction,
                self.maximum_parallax_cluster_ratio,
                self.minimum_edge_confidence,
                self.huber_delta_px,
                self.minimum_increment_px,
                self.maximum_reliable_edge_adjustment_px,
                self.anchor_long_minimum_spacing_px,
                self.anchor_long_maximum_spacing_px,
                self.anchor_long_minimum_timestamp_gap_ms,
                self.anchor_long_maximum_timestamp_gap_ms,
                self.anchor_long_minimum_predicted_overlap_fraction,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(finite).all() or np.any(finite[:6] <= 0.0) or np.any(finite[7:] <= 0.0):
            raise ValueError("S1.2 motion graph limits must be finite and positive")
        if self.minimum_edge_confidence < 0.0:
            raise ValueError("minimum_edge_confidence must be non-negative")
        if any(value > 1.0 for value in finite[2:7]) or not 0.0 < self.anchor_long_minimum_predicted_overlap_fraction <= 1.0:
            raise ValueError("S1.2 motion graph ratios/confidence must not exceed one")
        if self.minimum_inlier_count < 4:
            raise ValueError("minimum_inlier_count must be at least four")
        split_ratios = (
            self.minimum_fb_retention_ratio,
            self.minimum_model_inlier_ratio_retained,
            self.minimum_effective_inlier_ratio_detected,
        )
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in split_ratios):
            raise ValueError("split motion-graph ratios must be finite and within zero and one")
        if self.minimum_model_inlier_count < 4:
            raise ValueError("minimum_model_inlier_count must be at least four")
        if not 1 <= self.irls_iterations <= 32:
            raise ValueError("irls_iterations must be within 1..32")
        steps = self.adjacent_candidate_steps + self.skip_candidate_steps
        if not steps or any(isinstance(step, bool) or step <= 0 for step in steps):
            raise ValueError("motion graph candidate steps must be positive integers")
        if len(set(steps)) != len(steps):
            raise ValueError("motion graph candidate steps must be unique")
        local_frame_gap = self.effective_local_edge_maximum_frame_gap
        if local_frame_gap < max(steps):
            raise ValueError("local edge maximum frame gap is smaller than a candidate step")
        if self.anchor_long_maximum_frame_gap < 1:
            raise ValueError("anchor_long_maximum_frame_gap must be positive")
        if self.anchor_long_minimum_spacing_px > self.anchor_long_maximum_spacing_px:
            raise ValueError("anchor_long spacing limits are reversed")
        if self.anchor_long_minimum_timestamp_gap_ms > self.anchor_long_maximum_timestamp_gap_ms:
            raise ValueError("anchor_long timestamp limits are reversed")
        for name, value in (
            ("local_edge_lk_window_size", self.local_edge_lk_window_size),
            ("anchor_long_lk_window_size", self.anchor_long_lk_window_size),
        ):
            if value < 3 or value % 2 == 0:
                raise ValueError(f"{name} must be an odd integer of at least three")
        if not 0 <= self.local_edge_lk_max_level <= 8 or not 0 <= self.anchor_long_lk_max_level <= 8:
            raise ValueError("LK maximum pyramid level must be within 0..8")
        if self.anchor_long_feature_domain not in {
            "full_calibrated_frame", "fixed_calibrated_cx_slit"
        }:
            raise ValueError("unknown anchor_long feature domain")
        if self.anchor_long_preprocessing not in {"gray", "sobel_gradient_magnitude"}:
            raise ValueError("unknown anchor_long preprocessing")
        if self.anchor_long_initial_flow_source not in {"none", "endpoint_phase_correlation"}:
            raise ValueError("unknown anchor_long initial-flow source")
        if self.anchor_long_use_provisional_initial_flow and self.anchor_long_initial_flow_source != "none":
            raise ValueError("anchor_long initial-flow sources are mutually exclusive")
        if not 8 <= self.anchor_long_feature_slit_width_px <= self.analysis_width_px:
            raise ValueError("anchor_long feature slit width is invalid")
        return self

    @property
    def effective_local_edge_maximum_frame_gap(self) -> int:
        return (
            self.local_edge_maximum_frame_gap
            if self.maximum_direct_edge_frame_gap is None
            else self.maximum_direct_edge_frame_gap
        )


@dataclass(frozen=True)
class S12MotionNode:
    node_index: int
    frame_id: int
    calibrated_gray: np.ndarray
    pose_progress_m: float
    image_quality: float = 1.0
    timestamp_ms: float | None = None
    calibrated_cx: float | None = None

    def validated(self) -> "S12MotionNode":
        gray = np.asarray(self.calibrated_gray)
        if gray.ndim != 2 or gray.dtype != np.uint8 or min(gray.shape) < 16:
            raise ValueError("calibrated_gray must be a non-trivial uint8 image")
        if self.node_index < 0 or self.frame_id < 0:
            raise ValueError("S1.2 node and frame ids must be non-negative")
        if not math.isfinite(self.pose_progress_m) or not math.isfinite(self.image_quality):
            raise ValueError("S1.2 node metrics must be finite")
        if not 0.0 <= self.image_quality <= 1.0:
            raise ValueError("image_quality must be within zero and one")
        if self.timestamp_ms is not None and (
            not math.isfinite(self.timestamp_ms) or self.timestamp_ms < 0.0
        ):
            raise ValueError("timestamp_ms must be finite and non-negative when supplied")
        if self.calibrated_cx is not None and (
            not math.isfinite(self.calibrated_cx)
            or not 0.0 <= self.calibrated_cx < gray.shape[1]
        ):
            raise ValueError("calibrated_cx must be inside the analysis image")
        return replace(self, calibrated_gray=np.ascontiguousarray(gray))


@dataclass(frozen=True)
class S12MotionEdge:
    left_node: int
    right_node: int
    left_frame_id: int
    right_frame_id: int
    measured_dx_px: float
    measured_dy_px: float
    inlier_count: int
    inlier_ratio: float
    grid_coverage: float
    vertical_span_fraction: float
    forward_backward_p95_px: float
    parallax_cluster_ratio: float
    pose_progress_m: float
    pose_direction_agreement: bool
    confidence: float
    reliable: bool
    rejection_reasons: tuple[str, ...]
    kind: Literal["adjacent", "skip", "anchor_long"] = "adjacent"
    direct_image_match: bool = True
    detected_feature_count: int = 0
    forward_lk_valid_count: int = 0
    backward_lk_valid_count: int = 0
    fb_retained_count: int = 0
    model_inlier_count: int = 0
    fb_retention_ratio: float = 0.0
    model_inlier_ratio_retained: float = 0.0
    effective_inlier_ratio_detected: float = 0.0
    forward_backward_p50_px: float = 0.0
    model_residual_p50_px: float = 0.0
    model_residual_p95_px: float = 0.0
    model_residual_max_px: float = 0.0
    dominant_cluster_count: int = 0
    secondary_cluster_count: int = 0
    frame_gap: int = 0
    timestamp_gap_ms: float = 0.0
    predicted_overlap_fraction: float = 0.0
    provisional_spacing_px: float = 0.0
    feature_domain: str = "full_calibrated_frame"
    preprocessing: str = "gray"
    lk_window_size: int = 0
    lk_max_level: int = 0
    used_provisional_initial_flow: bool = False
    grid_domain_cell_count: int = 48
    grid_covered_cell_count: int = 0
    grid_covered_cell_ids: tuple[str, ...] = ()
    grid_rows: int = 6
    grid_columns: int = 8
    feature_calibrated_cx: float = 0.0
    feature_slit_left_px: int = 0
    feature_slit_right_px: int = 0
    feature_slit_width_px: int = 0
    initial_flow_source: str = "none"
    initial_flow_dx_px: float = 0.0
    initial_flow_dy_px: float = 0.0
    initial_flow_response: float = 0.0

    def __post_init__(self) -> None:
        if self.left_node < 0 or self.right_node <= self.left_node:
            raise ValueError("motion edge nodes must be chronological and distinct")
        if self.left_frame_id < 0 or self.right_frame_id <= self.left_frame_id:
            raise ValueError("motion edge frame ids must be chronological and distinct")
        values = np.asarray(
            (
                self.measured_dx_px,
                self.measured_dy_px,
                self.inlier_ratio,
                self.grid_coverage,
                self.vertical_span_fraction,
                self.forward_backward_p95_px,
                self.parallax_cluster_ratio,
                self.pose_progress_m,
                self.confidence,
                self.fb_retention_ratio,
                self.model_inlier_ratio_retained,
                self.effective_inlier_ratio_detected,
                self.forward_backward_p50_px,
                self.model_residual_p50_px,
                self.model_residual_p95_px,
                self.model_residual_max_px,
                self.timestamp_gap_ms,
                self.predicted_overlap_fraction,
                self.provisional_spacing_px,
                self.initial_flow_dx_px,
                self.initial_flow_dy_px,
                self.initial_flow_response,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or self.inlier_count < 0:
            raise ValueError("motion edge evidence must be finite and non-negative")
        if any(not 0.0 <= value <= 1.0 for value in values[[2, 3, 4, 6, 8]]):
            raise ValueError("motion edge ratios/confidence must be within zero and one")
        if self.kind not in {"adjacent", "skip", "anchor_long"}:
            raise ValueError("unknown S1.2 motion edge kind")
        if self.feature_domain not in {"full_calibrated_frame", "fixed_calibrated_cx_slit"}:
            raise ValueError("unknown S1.2 feature domain")
        if self.preprocessing not in {"gray", "sobel_gradient_magnitude"}:
            raise ValueError("unknown S1.2 motion preprocessing")
        if self.initial_flow_source not in {"none", "endpoint_phase_correlation"}:
            raise ValueError("unknown S1.2 initial-flow source")
        if self.lk_window_size < 0 or self.lk_max_level < 0:
            raise ValueError("motion edge LK audit values must be non-negative")
        if self.grid_covered_cell_count != len(self.grid_covered_cell_ids):
            raise ValueError("motion edge covered-cell audit is inconsistent")
        if not self.direct_image_match:
            raise ValueError("S1.2 graph forbids composed or synthetic motion edges")
        if self.reliable and self.rejection_reasons:
            raise ValueError("a reliable motion edge cannot have rejection reasons")
        counts = (
            self.detected_feature_count,
            self.forward_lk_valid_count,
            self.backward_lk_valid_count,
            self.fb_retained_count,
            self.model_inlier_count,
            self.dominant_cluster_count,
            self.secondary_cluster_count,
            self.frame_gap,
            self.grid_domain_cell_count,
            self.grid_covered_cell_count,
            self.grid_rows,
            self.grid_columns,
            self.feature_slit_left_px,
            self.feature_slit_right_px,
            self.feature_slit_width_px,
        )
        if any(value < 0 for value in counts):
            raise ValueError("motion edge diagnostic counts must be non-negative")
        if any(
            not 0.0 <= value <= 1.0
            for value in (
                self.fb_retention_ratio,
                self.model_inlier_ratio_retained,
                self.effective_inlier_ratio_detected,
                self.predicted_overlap_fraction,
            )
        ):
            raise ValueError("motion edge diagnostic ratios must be within zero and one")

    def as_dict(self) -> dict[str, object]:
        return {
            name: (list(value) if isinstance(value, tuple) else value)
            for name, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class S12MotionGraph:
    nodes: tuple[S12MotionNode, ...]
    edges: tuple[S12MotionEdge, ...]
    coordinate_domain: str = "calibrated_analysis"

    def __post_init__(self) -> None:
        if self.coordinate_domain != "calibrated_analysis":
            raise ValueError("S1.2 motion graph must remain in calibrated analysis coordinates")
        if len(self.nodes) < 2:
            raise ValueError("S1.2 motion graph requires at least two nodes")
        if tuple(node.node_index for node in self.nodes) != tuple(range(len(self.nodes))):
            raise ValueError("S1.2 motion graph node indices must be dense and chronological")
        frame_ids = tuple(node.frame_id for node in self.nodes)
        if any(right <= left for left, right in zip(frame_ids, frame_ids[1:])):
            raise ValueError("S1.2 motion graph frame ids must be strictly chronological")
        for edge in self.edges:
            if edge.right_node >= len(self.nodes):
                raise ValueError("motion edge refers to a missing node")
            if (edge.left_frame_id, edge.right_frame_id) != (
                self.nodes[edge.left_node].frame_id,
                self.nodes[edge.right_node].frame_id,
            ):
                raise ValueError("motion edge frame provenance does not match its nodes")

    @property
    def reliable_edges(self) -> tuple[S12MotionEdge, ...]:
        return tuple(edge for edge in self.edges if edge.reliable)

    def assert_reliable_connected(self) -> None:
        adjacency = [set() for _ in self.nodes]
        for edge in self.reliable_edges:
            adjacency[edge.left_node].add(edge.right_node)
            adjacency[edge.right_node].add(edge.left_node)
        visited = {0}
        pending = [0]
        while pending:
            for neighbour in adjacency[pending.pop()]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    pending.append(neighbour)
        if len(visited) != len(self.nodes):
            missing = sorted(set(range(len(self.nodes))) - visited)
            raise S12MotionGraphError(f"reliable direct motion graph is disconnected: {missing}")

    def with_edges(self, new_edges: Iterable[S12MotionEdge]) -> "S12MotionGraph":
        result = S12MotionGraph(self.nodes, self.edges + tuple(new_edges))
        result.assert_reliable_connected()
        return result


@dataclass(frozen=True)
class S12HorizontalSolution:
    centers_px: tuple[float, ...]
    increments_px: tuple[float, ...]
    edge_residuals_px: tuple[float, ...]
    reliable_edges: tuple[S12MotionEdge, ...]
    active_lower_bound_indices: tuple[int, ...]
    maximum_edge_adjustment_px: float

    def as_dict(self) -> dict[str, object]:
        return {
            "centers_px": list(self.centers_px),
            "increments_px": list(self.increments_px),
            "edge_residuals_px": list(self.edge_residuals_px),
            "active_lower_bound_indices": list(self.active_lower_bound_indices),
            "maximum_edge_adjustment_px": self.maximum_edge_adjustment_px,
            "strictly_increasing": all(
                right > left for left, right in zip(self.centers_px, self.centers_px[1:])
            ),
            "normalization_applied": False,
            "measurement_filter_applied": False,
        }


def calibrate_analysis_image(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    analysis_width_px: int = 424,
) -> tuple[np.ndarray, np.ndarray]:
    """Undistort a raw real RGB image into one explicit calibrated domain.

    Returns ``(gray, target_camera_matrix)``.  The target preserves the full
    calibrated field of view and scales it deterministically to analysis width.
    """

    source = np.asarray(image)
    if source.ndim not in (2, 3) or source.dtype != np.uint8:
        raise ValueError("raw image must be uint8 gray/BGR/BGRA")
    height, width = source.shape[:2]
    if width < 16 or height < 16 or analysis_width_px < 64:
        raise ValueError("invalid calibrated analysis image size")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if matrix.shape != (3, 3) or coefficients.size not in {4, 5, 8, 12, 14}:
        raise ValueError("camera calibration shape is invalid")
    if not np.isfinite(matrix).all() or not np.isfinite(coefficients).all():
        raise ValueError("camera calibration must be finite")
    scale = float(analysis_width_px) / float(width)
    target_height = max(16, int(round(height * scale)))
    target = matrix.copy()
    target[0, :] *= scale
    target[1, :] *= scale
    target[2, :] = (0.0, 0.0, 1.0)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, coefficients, None, target, (analysis_width_px, target_height), cv2.CV_32FC1
    )
    calibrated = cv2.remap(
        source, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    if calibrated.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if calibrated.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        calibrated = cv2.cvtColor(calibrated, code)
    return np.ascontiguousarray(calibrated), target


def _grid_features(
    gray: np.ndarray,
    rows: int = 6,
    columns: int = 8,
    *,
    x_bounds: tuple[int, int] | None = None,
) -> np.ndarray:
    points: list[np.ndarray] = []
    height, width = gray.shape
    for row in range(rows):
        y0, y1 = row * height // rows, (row + 1) * height // rows
        for column in range(columns):
            x0, x1 = column * width // columns, (column + 1) * width // columns
            if x_bounds is not None:
                x0, x1 = max(x0, x_bounds[0]), min(x1, x_bounds[1])
                if x1 - x0 < 8:
                    continue
            cell = gray[y0:y1, x0:x1]
            selected = cv2.goodFeaturesToTrack(
                cell, maxCorners=10, qualityLevel=0.01, minDistance=4, blockSize=5
            )
            if selected is not None:
                selected[:, 0, 0] += x0
                selected[:, 0, 1] += y0
                points.append(selected)
    if not points:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(points).astype(np.float32))


def _sobel_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def estimate_direct_motion_edge(
    left: S12MotionNode,
    right: S12MotionNode,
    *,
    kind: Literal["adjacent", "skip", "anchor_long"] = "adjacent",
    expected_image_motion_sign: int = 1,
    config: S12MotionGraphConfig | None = None,
    provisional_spacing_px: float | None = None,
) -> S12MotionEdge:
    """Directly match two real calibrated images and audit robust translation.

    ``expected_image_motion_sign`` maps observed calibrated-image x motion to
    positive panorama progress.  It must be established from the scan/pose
    direction, not inferred by clipping a negative measurement.
    """

    settings = (config or S12MotionGraphConfig()).validated()
    left, right = left.validated(), right.validated()
    if right.node_index <= left.node_index or right.frame_id <= left.frame_id:
        raise ValueError("direct motion endpoints must be chronological")
    frame_gap = right.frame_id - left.frame_id
    maximum_frame_gap = (
        settings.effective_local_edge_maximum_frame_gap
        if kind != "anchor_long" or settings.anchor_long_use_local_frame_gap
        else settings.anchor_long_maximum_frame_gap
    )
    if frame_gap > maximum_frame_gap:
        edge_name = "local" if kind != "anchor_long" else "anchor_long"
        raise ValueError(f"{edge_name} motion endpoints exceed maximum frame gap")
    if left.calibrated_gray.shape != right.calibrated_gray.shape:
        raise ValueError("direct motion images must share one calibrated domain")
    if left.calibrated_gray.shape[1] != settings.analysis_width_px:
        raise ValueError("direct motion image width does not match the calibrated analysis domain")
    if expected_image_motion_sign not in {-1, 1}:
        raise ValueError("expected_image_motion_sign must be -1 or +1")
    if kind == "anchor_long":
        lk_window_size = settings.anchor_long_lk_window_size
        lk_max_level = settings.anchor_long_lk_max_level
        match_left = left.calibrated_gray
        match_right = right.calibrated_gray
        if settings.anchor_long_preprocessing == "sobel_gradient_magnitude":
            match_left = _sobel_gradient_magnitude(match_left)
            match_right = _sobel_gradient_magnitude(match_right)
        x_bounds = None
        if settings.anchor_long_feature_domain == "fixed_calibrated_cx_slit":
            if left.calibrated_cx is None or right.calibrated_cx is None:
                raise ValueError("fixed calibrated-cx slit requires explicit endpoint calibrated_cx")
            if not math.isclose(left.calibrated_cx, right.calibrated_cx, abs_tol=1e-6):
                raise ValueError("anchor_long endpoint calibrated_cx values must agree")
            cx = float(left.calibrated_cx)
            half = settings.anchor_long_feature_slit_width_px / 2.0
            x_bounds = (int(cx - half), int(cx + half))
            if x_bounds[0] < 0 or x_bounds[1] > left.calibrated_gray.shape[1]:
                raise ValueError("fixed calibrated-cx slit must be fully inside the analysis image")
    else:
        lk_window_size = settings.local_edge_lk_window_size
        lk_max_level = settings.local_edge_lk_max_level
        match_left = left.calibrated_gray
        match_right = right.calibrated_gray
        x_bounds = None
    initial = (
        _grid_features(match_left)
        if x_bounds is None
        else _grid_features(match_left, x_bounds=x_bounds)
    )
    reasons: list[str] = []
    timestamp_gap_ms = (
        float(right.timestamp_ms - left.timestamp_ms)
        if left.timestamp_ms is not None and right.timestamp_ms is not None
        else 0.0
    )
    spacing = float(provisional_spacing_px) if provisional_spacing_px is not None else 0.0
    predicted_overlap = float(np.clip(1.0 - spacing / settings.analysis_width_px, 0.0, 1.0))
    if kind == "anchor_long":
        if settings.anchor_long_require_direct_image_match and (
            left.calibrated_gray is right.calibrated_gray
        ):
            reasons.append("anchor_long_requires_distinct_real_images")
        if settings.anchor_long_require_timestamp_audit:
            if left.timestamp_ms is None or right.timestamp_ms is None:
                reasons.append("anchor_long_timestamp_missing")
            elif not (
                settings.anchor_long_minimum_timestamp_gap_ms
                <= timestamp_gap_ms
                <= settings.anchor_long_maximum_timestamp_gap_ms
            ):
                reasons.append("anchor_long_timestamp_gap_out_of_range")
        if provisional_spacing_px is None or not math.isfinite(spacing):
            reasons.append("anchor_long_provisional_spacing_missing")
            spacing = 0.0
            predicted_overlap = 0.0
        elif not (
            settings.anchor_long_minimum_spacing_px
            <= spacing
            <= settings.anchor_long_maximum_spacing_px
        ):
            reasons.append("anchor_long_provisional_spacing_out_of_range")
        if (
            settings.anchor_long_require_predicted_overlap
            and predicted_overlap < settings.anchor_long_minimum_predicted_overlap_fraction
        ):
            reasons.append("anchor_long_predicted_overlap_insufficient")
    forward_lk_valid_count = 0
    backward_lk_valid_count = 0
    initial_flow_source = "none"
    initial_flow_dx = 0.0
    initial_flow_dy = 0.0
    initial_flow_response = 0.0
    if initial.shape[0] < 4:
        reasons.append("insufficient_detected_features")
        displacement = np.empty((0, 2), dtype=np.float64)
        retained = np.empty((0, 2), dtype=np.float64)
        fb = np.empty(0, dtype=np.float64)
    else:
        forward_initial = None
        lk_flags = 0
        if (
            kind == "anchor_long"
            and settings.anchor_long_initial_flow_source == "endpoint_phase_correlation"
        ):
            shift, response = cv2.phaseCorrelate(
                match_left.astype(np.float32), match_right.astype(np.float32)
            )
            initial_flow_source = "endpoint_phase_correlation"
            initial_flow_dx, initial_flow_dy = float(shift[0]), float(shift[1])
            initial_flow_response = float(response)
            forward_initial = initial.copy()
            forward_initial[:, 0, 0] += initial_flow_dx
            forward_initial[:, 0, 1] += initial_flow_dy
            lk_flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        elif kind == "anchor_long" and settings.anchor_long_use_provisional_initial_flow:
            forward_initial = initial.copy()
            forward_initial[:, 0, 0] += float(expected_image_motion_sign * spacing)
            lk_flags = cv2.OPTFLOW_USE_INITIAL_FLOW
        forward, status_f, _error = cv2.calcOpticalFlowPyrLK(
            match_left, match_right, initial, forward_initial,
            winSize=(lk_window_size, lk_window_size), maxLevel=lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=lk_flags,
        )
        if forward is None or status_f is None:
            reasons.append("forward_lk_failed")
            displacement = np.empty((0, 2), dtype=np.float64)
            retained = np.empty((0, 2), dtype=np.float64)
            fb = np.empty(0, dtype=np.float64)
        else:
            forward_valid = status_f.reshape(-1) > 0
            forward_lk_valid_count = int(np.count_nonzero(forward_valid))
            backward_initial = initial.copy() if lk_flags else None
            backward, status_b, _error = cv2.calcOpticalFlowPyrLK(
                match_right, match_left, forward, backward_initial,
                winSize=(lk_window_size, lk_window_size), maxLevel=lk_max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
                flags=lk_flags,
            )
            if backward is None or status_b is None:
                reasons.append("backward_lk_failed")
                displacement = np.empty((0, 2), dtype=np.float64)
                retained = np.empty((0, 2), dtype=np.float64)
                fb = np.empty(0, dtype=np.float64)
            else:
                backward_valid = status_b.reshape(-1) > 0
                status = forward_valid & backward_valid
                backward_lk_valid_count = int(np.count_nonzero(backward_valid))
                fb_all = np.linalg.norm(backward.reshape(-1, 2) - initial.reshape(-1, 2), axis=1)
                status &= np.isfinite(fb_all) & (fb_all <= settings.lk_forward_backward_max_px)
                retained = initial.reshape(-1, 2)[status].astype(np.float64)
                displacement = (
                    forward.reshape(-1, 2)[status].astype(np.float64) - retained
                )
                fb = fb_all[status]
    if displacement.size:
        # Deterministic translation RANSAC: every retained direct LK vector is
        # a hypothesis.  Stable input order supplies the final tie break.
        distances = np.linalg.norm(
            displacement[:, None, :] - displacement[None, :, :], axis=2
        )
        support = np.count_nonzero(distances <= settings.ransac_residual_px, axis=1)
        hypothesis = int(np.argmax(support))
        inliers = distances[hypothesis] <= settings.ransac_residual_px
        center = np.median(displacement[inliers], axis=0)
        residual = np.linalg.norm(displacement - center, axis=1)
        inliers = residual <= settings.ransac_residual_px
    else:
        center = np.zeros(2, dtype=np.float64)
        inliers = np.zeros(0, dtype=bool)
        residual = np.empty(0, dtype=np.float64)
    inlier_count = int(np.count_nonzero(inliers))
    detected_count = int(initial.shape[0])
    fb_retained_count = int(displacement.shape[0])
    inlier_ratio = float(inlier_count / max(1, initial.shape[0]))
    height, width = left.calibrated_gray.shape
    if x_bounds is None:
        grid_domain_cell_count = 48
    else:
        eligible_columns = sum(
            min((column + 1) * width // 8, x_bounds[1])
            - max(column * width // 8, x_bounds[0])
            >= 8
            for column in range(8)
        )
        grid_domain_cell_count = 6 * eligible_columns
    if inlier_count:
        positions = retained[inliers]
        cells = set(
            zip(
                np.minimum(5, (positions[:, 1] * 6 / height).astype(int)),
                np.minimum(7, (positions[:, 0] * 8 / width).astype(int)),
                strict=True,
            )
        )
        covered_cell_ids = tuple(f"r{row}c{column}" for row, column in sorted(cells))
        grid_coverage = len(cells) / max(1, grid_domain_cell_count)
        vertical_span = float((np.max(positions[:, 1]) - np.min(positions[:, 1])) / height)
        outlier_dx = displacement[~inliers, 0]
        alternate = 0
        for value in outlier_dx:
            alternate = max(alternate, int(np.count_nonzero(np.abs(outlier_dx - value) <= settings.ransac_residual_px)))
        parallax_ratio = float(alternate / max(1, inlier_count + alternate))
    else:
        grid_coverage = vertical_span = 0.0
        covered_cell_ids = ()
        alternate = 0
        parallax_ratio = 1.0
    fb_p50 = float(np.percentile(fb, 50.0)) if fb.size else settings.lk_forward_backward_max_px + 1.0
    fb_p95 = float(np.percentile(fb, 95.0)) if fb.size else settings.lk_forward_backward_max_px + 1.0
    if residual.size:
        residual_p50 = float(np.percentile(residual, 50.0))
        residual_p95 = float(np.percentile(residual, 95.0))
        residual_max = float(np.max(residual))
    else:
        residual_p50 = residual_p95 = residual_max = settings.ransac_residual_px + 1.0
    fb_retention_ratio = float(fb_retained_count / max(1, detected_count))
    model_inlier_ratio_retained = float(inlier_count / max(1, fb_retained_count))
    measured_dx = float(expected_image_motion_sign * center[0])
    measured_dy = float(center[1])
    pose_progress = float(right.pose_progress_m - left.pose_progress_m)
    direction_ok = bool(pose_progress > 0.0 and measured_dx > 0.0)
    gates = (
        (inlier_count >= settings.minimum_model_inlier_count, "insufficient_model_inliers"),
        (fb_retention_ratio >= settings.minimum_fb_retention_ratio, "insufficient_fb_retention_ratio"),
        (
            model_inlier_ratio_retained
            >= settings.minimum_model_inlier_ratio_retained,
            "insufficient_model_inlier_ratio_retained",
        ),
        (
            inlier_ratio >= settings.minimum_effective_inlier_ratio_detected,
            "insufficient_effective_inlier_ratio_detected",
        ),
        (grid_coverage >= settings.minimum_grid_coverage, "insufficient_grid_coverage"),
        (vertical_span >= settings.minimum_vertical_span_fraction, "insufficient_vertical_span"),
        (fb_p95 <= settings.lk_forward_backward_max_px, "forward_backward_error"),
        (parallax_ratio <= settings.maximum_parallax_cluster_ratio, "excessive_parallax_clusters"),
        (direction_ok, "pose_direction_disagreement"),
    )
    reasons.extend(reason for passed, reason in gates if not passed)
    confidence = float(
        min(
            1.0,
            inlier_count / max(1, settings.minimum_model_inlier_count),
            fb_retention_ratio / settings.minimum_fb_retention_ratio,
            model_inlier_ratio_retained
            / settings.minimum_model_inlier_ratio_retained,
            inlier_ratio / settings.minimum_effective_inlier_ratio_detected,
            grid_coverage / settings.minimum_grid_coverage,
            vertical_span / settings.minimum_vertical_span_fraction,
            settings.lk_forward_backward_max_px / max(fb_p95, 1e-9),
            (1.0 - parallax_ratio)
            / max(1e-9, 1.0 - settings.maximum_parallax_cluster_ratio),
        )
    )
    if confidence < settings.minimum_edge_confidence:
        reasons.append("confidence_below_minimum")
    return S12MotionEdge(
        left_node=left.node_index,
        right_node=right.node_index,
        left_frame_id=left.frame_id,
        right_frame_id=right.frame_id,
        measured_dx_px=measured_dx,
        measured_dy_px=measured_dy,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        grid_coverage=grid_coverage,
        vertical_span_fraction=vertical_span,
        forward_backward_p95_px=fb_p95,
        parallax_cluster_ratio=parallax_ratio,
        pose_progress_m=pose_progress,
        pose_direction_agreement=direction_ok,
        confidence=confidence,
        reliable=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        kind=kind,
        direct_image_match=True,
        detected_feature_count=detected_count,
        forward_lk_valid_count=forward_lk_valid_count,
        backward_lk_valid_count=backward_lk_valid_count,
        fb_retained_count=fb_retained_count,
        model_inlier_count=inlier_count,
        fb_retention_ratio=fb_retention_ratio,
        model_inlier_ratio_retained=model_inlier_ratio_retained,
        effective_inlier_ratio_detected=inlier_ratio,
        forward_backward_p50_px=fb_p50,
        model_residual_p50_px=residual_p50,
        model_residual_p95_px=residual_p95,
        model_residual_max_px=residual_max,
        dominant_cluster_count=inlier_count,
        secondary_cluster_count=alternate,
        frame_gap=frame_gap,
        timestamp_gap_ms=timestamp_gap_ms,
        predicted_overlap_fraction=predicted_overlap,
        provisional_spacing_px=spacing,
        feature_domain=(
            settings.anchor_long_feature_domain if kind == "anchor_long" else "full_calibrated_frame"
        ),
        preprocessing=settings.anchor_long_preprocessing if kind == "anchor_long" else "gray",
        lk_window_size=lk_window_size,
        lk_max_level=lk_max_level,
        used_provisional_initial_flow=(
            kind == "anchor_long" and settings.anchor_long_use_provisional_initial_flow
        ),
        grid_domain_cell_count=grid_domain_cell_count,
        grid_covered_cell_count=len(covered_cell_ids),
        grid_covered_cell_ids=covered_cell_ids,
        feature_calibrated_cx=float(left.calibrated_cx or 0.0),
        feature_slit_left_px=x_bounds[0] if x_bounds is not None else 0,
        feature_slit_right_px=x_bounds[1] if x_bounds is not None else 0,
        feature_slit_width_px=(x_bounds[1] - x_bounds[0]) if x_bounds is not None else 0,
        initial_flow_source=initial_flow_source,
        initial_flow_dx_px=initial_flow_dx,
        initial_flow_dy_px=initial_flow_dy,
        initial_flow_response=initial_flow_response,
    )


def build_motion_graph(
    nodes: Sequence[S12MotionNode],
    *,
    expected_image_motion_sign: int,
    config: S12MotionGraphConfig | None = None,
    require_connected: bool = True,
) -> S12MotionGraph:
    """Build adjacent/skip edges by direct matching; never compose an edge.

    ``require_connected=False`` is reserved for the first M1/M2 pass: its
    rejected adjacent edges are fed back into longest-segment selection. The
    final selected graph always uses the default fail-closed connectivity gate.
    """

    settings = (config or S12MotionGraphConfig()).validated()
    checked = tuple(node.validated() for node in nodes)
    graph = S12MotionGraph(checked, ())
    edges: list[S12MotionEdge] = []
    for step in settings.adjacent_candidate_steps + settings.skip_candidate_steps:
        kind: Literal["adjacent", "skip"] = "adjacent" if step in settings.adjacent_candidate_steps else "skip"
        for left_index in range(len(checked) - step):
            edges.append(
                estimate_direct_motion_edge(
                    checked[left_index], checked[left_index + step], kind=kind,
                    expected_image_motion_sign=expected_image_motion_sign, config=settings,
                )
            )
    graph = S12MotionGraph(checked, tuple(edges))
    if require_connected:
        graph.assert_reliable_connected()
    return graph


def _bounded_weighted_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    lower: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    count = design.shape[1]
    active: set[int] = set()
    solution = np.full(count, lower, dtype=np.float64)
    while True:
        free = [index for index in range(count) if index not in active]
        rhs = target.copy()
        if active:
            fixed = sorted(active)
            rhs -= design[:, fixed] @ np.full(len(fixed), lower)
        if free:
            matrix = design[:, free] * np.sqrt(weights)[:, None]
            vector = rhs * np.sqrt(weights)
            candidate, _residual, rank, _singular = np.linalg.lstsq(matrix, vector, rcond=None)
            if rank != len(free) or not np.isfinite(candidate).all():
                raise S12MotionGraphError("horizontal active-set solve is rank deficient")
            solution[free] = candidate
        if active:
            solution[sorted(active)] = lower
        violations = [index for index in free if solution[index] < lower]
        if not violations:
            return solution, tuple(sorted(active))
        active.update(violations)
        if len(active) == count:
            return np.full(count, lower, dtype=np.float64), tuple(range(count))


def solve_horizontal_centers(
    graph: S12MotionGraph,
    *,
    calibrated_cx: float,
    config: S12MotionGraphConfig | None = None,
) -> S12HorizontalSolution:
    """Solve positive adjacent increments with Huber IRLS and an active set."""

    settings = (config or S12MotionGraphConfig()).validated()
    graph.assert_reliable_connected()
    edges = graph.reliable_edges
    if not math.isfinite(calibrated_cx):
        raise ValueError("calibrated_cx must be finite")
    variable_count = len(graph.nodes) - 1
    design = np.zeros((len(edges), variable_count), dtype=np.float64)
    target = np.empty(len(edges), dtype=np.float64)
    base_weight = np.empty(len(edges), dtype=np.float64)
    for row, edge in enumerate(edges):
        design[row, edge.left_node:edge.right_node] = 1.0
        target[row] = edge.measured_dx_px
        base_weight[row] = max(edge.confidence, 1e-6)
    weights = base_weight.copy()
    active: tuple[int, ...] = ()
    increments = np.full(variable_count, settings.minimum_increment_px, dtype=np.float64)
    for _ in range(settings.irls_iterations):
        increments, active = _bounded_weighted_lstsq(
            design, target, weights, settings.minimum_increment_px
        )
        residual = design @ increments - target
        huber = np.ones_like(residual)
        outside = np.abs(residual) > settings.huber_delta_px
        huber[outside] = settings.huber_delta_px / np.abs(residual[outside])
        weights = base_weight * huber
    residual = design @ increments - target
    maximum = float(np.max(np.abs(residual))) if residual.size else math.inf
    if not np.isfinite(increments).all() or np.any(increments < settings.minimum_increment_px - 1e-9):
        raise S12MotionGraphError("horizontal solver violated the positive-increment contract")
    if maximum > settings.maximum_reliable_edge_adjustment_px:
        raise S12MotionGraphError(
            f"reliable direct edge correction {maximum:.6g}px exceeds "
            f"{settings.maximum_reliable_edge_adjustment_px:.6g}px"
        )
    centers = calibrated_cx + np.concatenate(([0.0], np.cumsum(increments)))
    if np.any(np.diff(centers) <= 0.0):
        raise S12MotionGraphError("horizontal centers are not strictly increasing")
    return S12HorizontalSolution(
        tuple(float(value) for value in centers),
        tuple(float(value) for value in increments),
        tuple(float(value) for value in residual), edges, active, maximum,
    )
