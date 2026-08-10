"""Grid-balanced LK and phase-correlation telemetry for S1.3 P0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from .video_s13_session import S13RenderFrame, read_s13_rgb


@dataclass(frozen=True)
class S13MotionEdge:
    source_frame_id: int
    target_frame_id: int
    step: int
    lk_advance_px: float | None
    lk_vertical_px: float | None
    lk_total_weight: float
    lk_observation_count: int
    phase_advance_px: float | None
    phase_vertical_px: float | None
    phase_response: float
    selected_advance_px: float | None
    selected_method: str
    risk: bool
    telemetry_only_reasons: tuple[str, ...]


@dataclass(frozen=True)
class S13Progress:
    frame_ids: tuple[int, ...]
    centers_x: tuple[float, ...]
    placement_methods: tuple[str, ...]
    spatial: bool
    coherent_rgb_motion: bool
    adjacent_reliable_graph_connected: bool


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered_values, ordered_weights = values[order], weights[order]
    midpoint = float(ordered_weights.sum()) * 0.5
    return float(ordered_values[min(int(np.searchsorted(np.cumsum(ordered_weights), midpoint)), len(values) - 1)])


def _analysis_gray(frame: S13RenderFrame, width: int) -> tuple[np.ndarray, float]:
    image = read_s13_rgb(frame)
    scale = min(1.0, float(width) / image.shape[1])
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.0 / scale


def _grid_points(gray: np.ndarray, *, cell_px: int = 28, per_cell: int = 4) -> np.ndarray:
    height, width = gray.shape
    left, right = int(round(width * 0.15)), int(round(width * 0.85))
    selected: list[np.ndarray] = []
    # Detect independently inside every cell. A global detector followed by
    # binning still lets a few strongly textured cells consume maxCorners and
    # is not the grid-balanced LK contract required by S1.3.
    for top in range(0, height, cell_px):
        bottom = min(height, top + cell_px)
        for cell_left in range(left, right, cell_px):
            cell_right = min(right, cell_left + cell_px)
            cell = gray[top:bottom, cell_left:cell_right]
            if min(cell.shape) < 5:
                continue
            points = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=per_cell,
                qualityLevel=0.005,
                minDistance=4,
                blockSize=5,
            )
            if points is None:
                continue
            offset = np.asarray((cell_left, top), dtype=np.float32)
            selected.extend(point + offset for point in points.reshape(-1, 2))
    if not selected:
        return np.empty((0, 1, 2), dtype=np.float32)
    return np.asarray(selected, dtype=np.float32).reshape(-1, 1, 2)


def _lk(left: np.ndarray, right: np.ndarray, output_scale: float) -> tuple[float | None, float | None, float, int]:
    points = _grid_points(left)
    if len(points) == 0:
        return None, None, 0.0, 0
    forward, status_f, error_f = cv2.calcOpticalFlowPyrLK(left, right, points, None, winSize=(21, 21), maxLevel=3)
    if forward is None or status_f is None:
        return None, None, 0.0, 0
    backward, status_b, error_b = cv2.calcOpticalFlowPyrLK(right, left, forward, None, winSize=(21, 21), maxLevel=3)
    if backward is None or status_b is None:
        return None, None, 0.0, 0
    original = points.reshape(-1, 2)
    matched = forward.reshape(-1, 2)
    fb = np.linalg.norm(backward.reshape(-1, 2) - original, axis=1)
    lk_error = 0.5 * (error_f.reshape(-1) + error_b.reshape(-1))
    finite = (
        status_f.reshape(-1).astype(bool) & status_b.reshape(-1).astype(bool)
        & np.isfinite(matched).all(axis=1) & np.isfinite(fb) & np.isfinite(lk_error)
        & (matched[:, 0] >= 0) & (matched[:, 0] < right.shape[1])
        & (matched[:, 1] >= 0) & (matched[:, 1] < right.shape[0])
    )
    if not np.any(finite):
        return None, None, 0.0, 0
    original, matched, fb, lk_error = original[finite], matched[finite], fb[finite], lk_error[finite]
    gradient = cv2.Sobel(left, cv2.CV_32F, 1, 1, ksize=3)
    sample_x = np.clip(np.rint(original[:, 0]).astype(int), 0, left.shape[1] - 1)
    sample_y = np.clip(np.rint(original[:, 1]).astype(int), 0, left.shape[0] - 1)
    texture = np.abs(gradient[sample_y, sample_x]).astype(np.float64)
    texture = np.clip(texture / (np.median(texture) + 1e-6), 0.1, 2.0)
    central = np.clip(1.0 - np.abs(original[:, 0] / max(1, left.shape[1] - 1) - 0.5), 0.25, 1.0)
    weights = texture * central * np.exp(-0.5 * (fb / 1.5) ** 2) * np.exp(-0.5 * (lk_error / 20.0) ** 2)
    usable = np.isfinite(weights) & (weights > 1e-8)
    if not np.any(usable):
        return None, None, 0.0, 0
    displacement = (matched[usable] - original[usable]) * output_scale
    weights = weights[usable]
    return (
        -_weighted_median(displacement[:, 0], weights),
        _weighted_median(displacement[:, 1], weights),
        float(weights.sum()),
        int(len(weights)),
    )


def _phase(left: np.ndarray, right: np.ndarray, output_scale: float) -> tuple[float | None, float | None, float]:
    width = left.shape[1]
    lo, hi = int(round(width * 0.15)), int(round(width * 0.85))
    a, b = left[:, lo:hi].astype(np.float32), right[:, lo:hi].astype(np.float32)
    if a.shape != b.shape or min(a.shape) < 8 or float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        return None, None, 0.0
    shift, response = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F))
    if not np.isfinite(shift).all() or not math.isfinite(response):
        return None, None, 0.0
    return -float(shift[0]) * output_scale, float(shift[1]) * output_scale, float(response)


def measure_s13_motion(
    frames: Sequence[S13RenderFrame], *, analysis_width_px: int = 424, steps: Sequence[int] = (1, 2, 4)
) -> tuple[S13MotionEdge, ...]:
    analysis = [_analysis_gray(frame, analysis_width_px) for frame in frames]
    edges: list[S13MotionEdge] = []
    for step in steps:
        for index in range(len(frames) - step):
            left, scale = analysis[index]
            right, right_scale = analysis[index + step]
            if left.shape != right.shape or not np.isclose(scale, right_scale):
                continue
            lk_x, lk_y, total_weight, count = _lk(left, right, scale)
            phase_x, phase_y, response = _phase(left, right, scale)
            use_lk = lk_x is not None and count >= 8 and total_weight >= 2.0
            selected = lk_x if use_lk else phase_x
            method = "grid_lk" if use_lk else "phase_correlation" if selected is not None else "unavailable"
            disagreement = lk_x is not None and phase_x is not None and abs(lk_x - phase_x) > max(3.0, 0.5 * max(abs(lk_x), abs(phase_x)))
            reasons = []
            if count < 8 or total_weight < 2.0:
                reasons.append("low_lk_support")
            if disagreement:
                reasons.append("lk_phase_disagreement")
            if response < 0.05:
                reasons.append("low_phase_response")
            edges.append(S13MotionEdge(
                source_frame_id=frames[index].frame_id, target_frame_id=frames[index + step].frame_id,
                step=step, lk_advance_px=lk_x, lk_vertical_px=lk_y, lk_total_weight=total_weight,
                lk_observation_count=count, phase_advance_px=phase_x, phase_vertical_px=phase_y,
                phase_response=response, selected_advance_px=selected, selected_method=method,
                risk=bool(reasons), telemetry_only_reasons=tuple(reasons),
            ))
    return tuple(edges)


def build_basic_s13_progress(frames: Sequence[S13RenderFrame], edges: Sequence[S13MotionEdge]) -> S13Progress:
    adjacent_by_pair = {(edge.source_frame_id, edge.target_frame_id): edge for edge in edges if edge.step == 1}
    reliable = [
        abs(float(edge.selected_advance_px)) for edge in edges
        if edge.step == 1 and edge.selected_advance_px is not None and abs(float(edge.selected_advance_px)) >= 0.25
    ]
    session_median = float(np.median(reliable)) if reliable else 1.0
    centers = [0.0]
    methods = ["origin"]
    measured_count = 0
    connected = True
    for left, right in zip(frames[:-1], frames[1:]):
        edge = adjacent_by_pair.get((left.frame_id, right.frame_id))
        if edge is not None and edge.selected_advance_px is not None and abs(edge.selected_advance_px) >= 0.25:
            advance = abs(float(edge.selected_advance_px))
            method = "grid_lk" if edge.selected_method == "grid_lk" else "phase_correlation"
            measured_count += 1
        else:
            advance = session_median
            method = "session_median" if reliable else "temporal_order"
            connected = False
        centers.append(centers[-1] + max(0.0, advance))
        methods.append(method)
    spatial = len(frames) >= 2 and measured_count > 0
    return S13Progress(
        frame_ids=tuple(frame.frame_id for frame in frames), centers_x=tuple(centers), placement_methods=tuple(methods),
        spatial=spatial, coherent_rgb_motion=spatial and measured_count >= max(1, len(frames) // 2),
        adjacent_reliable_graph_connected=connected,
    )


def descriptive_delta_risk(delta_px: float) -> dict[str, object]:
    """Record legacy direct/local deltas as telemetry, never as a P0 gate."""

    return {"direct_local_delta_px": float(delta_px), "risk": abs(float(delta_px)) > 2.0, "structural_gate": False}


__all__ = [
    "S13MotionEdge", "S13Progress", "build_basic_s13_progress", "descriptive_delta_risk", "measure_s13_motion",
]
