"""Grid-balanced LK and phase-correlation telemetry for S1.3 P0."""

from __future__ import annotations

import math
import hashlib
import time
from dataclasses import dataclass
from typing import MutableMapping, Sequence

import cv2
import numpy as np

from .video_s13_session import S13RenderFrame, read_s13_rgb


_MIN_SPATIAL_ADVANCE_PX = 0.25
_MIN_LK_OBSERVATIONS = 8
_MIN_LK_WEIGHT = 2.0
_MIN_PHASE_RESPONSE = 0.05


@dataclass(frozen=True)
class S13MotionObservation:
    x_px: float
    y_px: float
    advance_px: float
    vertical_px: float
    weight: float
    grid_cell: str


@dataclass(frozen=True)
class S13MotionHypothesis:
    hypothesis_id: int
    advance_px: float
    vertical_px: float
    weight: float
    residual_mad_px: float
    central_support: float
    spatial_coverage: float
    largest_component_ratio: float
    vertical_span: float
    centroid_x: float
    centroid_y: float
    support_signature_sha256: str
    source_method: str
    confidence: float
    support_cells: tuple[str, ...] = ()


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
    observations: tuple[S13MotionObservation, ...] = ()
    motion_hypotheses: tuple[S13MotionHypothesis, ...] = ()


@dataclass(frozen=True)
class S13PreparedMotionFrame:
    """One exact M0 input shared by batch and committed-ledger execution."""

    frame_id: int
    source_jpeg_sha256: str
    decoded_pixel_sha256: str
    gray_424: np.ndarray
    output_scale: float
    gradient_424: np.ndarray
    grid_points: np.ndarray
    preprocessing_fingerprint: str


@dataclass(frozen=True)
class S13MotionSnapshot:
    prepared_frames: tuple[S13PreparedMotionFrame, ...]
    edges: tuple[S13MotionEdge, ...]
    step4_computed: bool
    step4_selected: bool


class S13MotionAccumulator:
    """Lossless ordered step-1/2/4 accumulator for committed frames."""

    def __init__(self) -> None:
        self._prepared: list[S13PreparedMotionFrame] = []
        self._edges: list[S13MotionEdge] = []

    def append(self, frame: S13PreparedMotionFrame) -> tuple[S13MotionEdge, ...]:
        if self._prepared and frame.frame_id <= self._prepared[-1].frame_id:
            raise ValueError("S1.3 motion accumulator frame ids must be strictly increasing")
        added: list[S13MotionEdge] = []
        for step in (1, 2, 4):
            if len(self._prepared) < step:
                continue
            edge = measure_s13_motion_edge(self._prepared[-step], frame, step=step)
            if edge is not None:
                self._edges.append(edge)
                added.append(edge)
        self._prepared.append(frame)
        return tuple(added)

    def snapshot(self, *, deferred_step4: bool = True) -> S13MotionSnapshot:
        step4_computed = any(edge.step == 4 for edge in self._edges)
        step4_selected = not deferred_step4 or not reliable_step1_direction_evidence(
            self._edges
        )
        edges = tuple(
            edge for edge in self._edges if edge.step != 4 or step4_selected
        )
        return S13MotionSnapshot(
            prepared_frames=tuple(self._prepared),
            edges=edges,
            step4_computed=step4_computed,
            step4_selected=step4_selected,
        )


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def prepare_s13_motion_frame(
    frame: S13RenderFrame,
    *,
    analysis_width_px: int = 424,
    prepared_analysis: tuple[np.ndarray, float] | None = None,
    prepared_gradient: np.ndarray | None = None,
    include_source_identity: bool = True,
) -> S13PreparedMotionFrame:
    """Prepare a frame once for the exact LK/phase primitive.

    ``prepared_analysis`` is the batch frame-store handoff.  The committed
    online path omits it and therefore decodes the exact JPEG through the same
    ``read_s13_rgb`` implementation.
    """

    gray, output_scale = (
        _analysis_gray(frame, analysis_width_px)
        if prepared_analysis is None else prepared_analysis
    )
    gray = np.asarray(gray)
    gradient = np.asarray(
        cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        if prepared_gradient is None else prepared_gradient
    )
    points = _grid_points(gray)
    source_sha = ""
    if include_source_identity:
        source_sha = hashlib.sha256(frame.color_path.read_bytes()).hexdigest()
    decoded_sha = _array_sha256(gray) if include_source_identity else ""
    fingerprint = (
        hashlib.sha256(
            (
                f"s013-m0/v1|width={int(analysis_width_px)}|scale={float(output_scale):.17g}|"
                f"gray={decoded_sha}|gradient={_array_sha256(gradient)}|"
                f"points={_array_sha256(points)}"
            ).encode("ascii")
        ).hexdigest()
        if include_source_identity else ""
    )
    return S13PreparedMotionFrame(
        frame_id=int(frame.frame_id),
        source_jpeg_sha256=source_sha,
        decoded_pixel_sha256=decoded_sha,
        gray_424=gray,
        output_scale=float(output_scale),
        gradient_424=gradient,
        grid_points=points,
        preprocessing_fingerprint=fingerprint,
    )


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


def _lk(
    left: np.ndarray,
    right: np.ndarray,
    output_scale: float,
    *,
    points: np.ndarray | None = None,
    gradient: np.ndarray | None = None,
) -> tuple[float | None, float | None, float, int, tuple[S13MotionObservation, ...]]:
    points = _grid_points(left) if points is None else points
    if len(points) == 0:
        return None, None, 0.0, 0, ()
    forward, status_f, error_f = cv2.calcOpticalFlowPyrLK(left, right, points, None, winSize=(21, 21), maxLevel=3)
    if forward is None or status_f is None:
        return None, None, 0.0, 0, ()
    backward, status_b, error_b = cv2.calcOpticalFlowPyrLK(right, left, forward, None, winSize=(21, 21), maxLevel=3)
    if backward is None or status_b is None:
        return None, None, 0.0, 0, ()
    original = points.reshape(-1, 2)
    matched = forward.reshape(-1, 2)
    fb = np.linalg.norm(backward.reshape(-1, 2) - original, axis=1)
    with np.errstate(invalid="ignore"):
        error_f_flat = error_f.reshape(-1).astype(np.float64)
        error_b_flat = error_b.reshape(-1).astype(np.float64)
    lk_error = np.full(error_f_flat.shape, np.inf, dtype=np.float64)
    finite_error = np.isfinite(error_f_flat) & np.isfinite(error_b_flat)
    lk_error[finite_error] = 0.5 * (error_f_flat[finite_error] + error_b_flat[finite_error])
    finite = (
        status_f.reshape(-1).astype(bool) & status_b.reshape(-1).astype(bool)
        & np.isfinite(matched).all(axis=1) & np.isfinite(fb) & np.isfinite(lk_error)
        & (matched[:, 0] >= 0) & (matched[:, 0] < right.shape[1])
        & (matched[:, 1] >= 0) & (matched[:, 1] < right.shape[0])
    )
    if not np.any(finite):
        return None, None, 0.0, 0, ()
    original, matched, fb, lk_error = original[finite], matched[finite], fb[finite], lk_error[finite]
    gradient = (
        cv2.Sobel(left, cv2.CV_32F, 1, 1, ksize=3)
        if gradient is None else gradient
    )
    sample_x = np.clip(np.rint(original[:, 0]).astype(int), 0, left.shape[1] - 1)
    sample_y = np.clip(np.rint(original[:, 1]).astype(int), 0, left.shape[0] - 1)
    texture = np.abs(gradient[sample_y, sample_x]).astype(np.float64)
    texture = np.clip(texture / (np.median(texture) + 1e-6), 0.1, 2.0)
    central = np.clip(1.0 - np.abs(original[:, 0] / max(1, left.shape[1] - 1) - 0.5), 0.25, 1.0)
    weights = texture * central * np.exp(-0.5 * (fb / 1.5) ** 2) * np.exp(-0.5 * (lk_error / 20.0) ** 2)
    usable = np.isfinite(weights) & (weights > 1e-8)
    if not np.any(usable):
        return None, None, 0.0, 0, ()
    displacement = (matched[usable] - original[usable]) * output_scale
    positions = original[usable] * output_scale
    weights = weights[usable]
    cell_size = 28.0 * output_scale
    observations = tuple(
        S13MotionObservation(
            x_px=float(position[0]),
            y_px=float(position[1]),
            advance_px=-float(delta[0]),
            vertical_px=float(delta[1]),
            weight=float(weight),
            grid_cell=f"{int(position[0] // cell_size)}:{int(position[1] // cell_size)}",
        )
        for position, delta, weight in zip(positions, displacement, weights)
    )
    return (
        -_weighted_median(displacement[:, 0], weights),
        _weighted_median(displacement[:, 1], weights),
        float(weights.sum()),
        int(len(weights)),
        observations,
    )


def _phase(
    left: np.ndarray,
    right: np.ndarray,
    output_scale: float,
    *,
    hanning: np.ndarray | None = None,
) -> tuple[float | None, float | None, float]:
    width = left.shape[1]
    lo, hi = int(round(width * 0.15)), int(round(width * 0.85))
    a, b = left[:, lo:hi].astype(np.float32), right[:, lo:hi].astype(np.float32)
    if a.shape != b.shape or min(a.shape) < 8 or float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        return None, None, 0.0
    if hanning is None:
        hanning = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a, b, hanning)
    if not np.isfinite(shift).all() or not math.isfinite(response):
        return None, None, 0.0
    return -float(shift[0]) * output_scale, float(shift[1]) * output_scale, float(response)


def _largest_component_ratio(cells: set[str]) -> float:
    if not cells:
        return 0.0
    coordinates = {tuple(int(part) for part in cell.split(":")) for cell in cells}
    largest = 0
    while coordinates:
        pending = [coordinates.pop()]
        size = 0
        while pending:
            x, y = pending.pop()
            size += 1
            for neighbour in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbour in coordinates:
                    coordinates.remove(neighbour)
                    pending.append(neighbour)
        largest = max(largest, size)
    return float(largest) / float(len(cells))


def _extract_hypotheses(
    observations: Sequence[S13MotionObservation],
    *,
    selected_advance_px: float | None,
    image_width_px: float,
    image_height_px: float,
    max_hypotheses: int = 3,
) -> tuple[S13MotionHypothesis, ...]:
    """Extract zero to three weighted local motion modes from the LK support."""

    if len(observations) < _MIN_LK_OBSERVATIONS or max_hypotheses <= 0:
        return ()
    # Preserve the measured sign. Canonical scan direction is a sequence-level
    # decision and must never be re-derived independently for each pair.
    advance = np.asarray([item.advance_px for item in observations], dtype=np.float64)
    vertical = np.asarray([item.vertical_px for item in observations], dtype=np.float64)
    weight = np.asarray([item.weight for item in observations], dtype=np.float64)
    finite = np.isfinite(advance) & np.isfinite(vertical) & np.isfinite(weight) & (weight > 0)
    if int(finite.sum()) < _MIN_LK_OBSERVATIONS or float(weight[finite].sum()) < _MIN_LK_WEIGHT:
        return ()
    indices = np.flatnonzero(finite)
    values = advance[indices]
    bin_px = 0.75
    lo = math.floor(float(values.min()) / bin_px) * bin_px
    hi = math.ceil(float(values.max()) / bin_px) * bin_px + bin_px
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return ()
    bins = np.arange(lo, hi + bin_px * 0.5, bin_px)
    if len(bins) < 2:
        return ()
    histogram, _ = np.histogram(values, bins=bins, weights=weight[indices])
    smooth = np.convolve(histogram.astype(np.float64), np.asarray((0.25, 0.5, 0.25)), mode="same")
    peak_bins = [
        index for index, value in enumerate(smooth)
        if value > 0 and value >= smooth[max(0, index - 1)] and value >= smooth[min(len(smooth) - 1, index + 1)]
    ]
    peak_bins.sort(key=lambda index: (-smooth[index], index))
    hypotheses: list[S13MotionHypothesis] = []
    total_weight = float(weight[indices].sum())
    all_cells = {observations[index].grid_cell for index in indices}
    for peak_bin in peak_bins:
        center = 0.5 * (bins[peak_bin] + bins[peak_bin + 1])
        seed = indices[np.abs(advance[indices] - center) <= 1.25]
        if len(seed) < 3:
            continue
        median = _weighted_median(advance[seed], weight[seed])
        mad = _weighted_median(np.abs(advance[seed] - median), weight[seed])
        support = indices[np.abs(advance[indices] - median) <= max(0.75, 2.5 * mad)]
        if len(support) < 3 or float(weight[support].sum()) < 0.08 * total_weight:
            continue
        refined = _weighted_median(advance[support], weight[support])
        if any(abs(existing.advance_px - refined) < 1.0 for existing in hypotheses):
            continue
        dy = _weighted_median(vertical[support], weight[support])
        residual_mad = _weighted_median(np.abs(advance[support] - refined), weight[support])
        support_weight = float(weight[support].sum())
        support_cells = {observations[index].grid_cell for index in support}
        xs = np.asarray([observations[index].x_px for index in support], dtype=np.float64)
        ys = np.asarray([observations[index].y_px for index in support], dtype=np.float64)
        ws = weight[support]
        signature_payload = "|".join(sorted(support_cells)).encode("utf-8")
        coverage = float(len(support_cells)) / float(max(1, len(all_cells)))
        central = float(ws[(xs >= image_width_px * 0.35) & (xs <= image_width_px * 0.65)].sum()) / support_weight
        vertical_span = float((ys.max() - ys.min()) / max(1.0, image_height_px))
        confidence = min(1.0, support_weight / max(_MIN_LK_WEIGHT, total_weight * 0.35))
        confidence *= min(1.0, coverage / 0.25) * min(1.0, len(support) / 8.0)
        hypotheses.append(S13MotionHypothesis(
            hypothesis_id=len(hypotheses),
            advance_px=float(refined),
            vertical_px=float(dy),
            weight=support_weight,
            residual_mad_px=float(residual_mad),
            central_support=central,
            spatial_coverage=coverage,
            largest_component_ratio=_largest_component_ratio(support_cells),
            vertical_span=vertical_span,
            centroid_x=float(np.average(xs, weights=ws) / max(1.0, image_width_px)),
            centroid_y=float(np.average(ys, weights=ws) / max(1.0, image_height_px)),
            support_signature_sha256=hashlib.sha256(signature_payload).hexdigest(),
            source_method="grid_lk_weighted_histogram",
            confidence=float(confidence),
            support_cells=tuple(sorted(support_cells)),
        ))
        if len(hypotheses) >= max_hypotheses:
            break
    hypotheses.sort(key=lambda item: (-item.confidence, -item.weight, item.advance_px))
    return tuple(
        S13MotionHypothesis(**{**item.__dict__, "hypothesis_id": index})
        for index, item in enumerate(hypotheses)
    )


def measure_s13_motion_edge(
    source: S13PreparedMotionFrame,
    target: S13PreparedMotionFrame,
    *,
    step: int,
    profile: MutableMapping[str, float] | None = None,
    hanning: np.ndarray | None = None,
) -> S13MotionEdge | None:
    """Measure one exact step-1/2/4 edge from shared prepared inputs."""

    edge_started = time.perf_counter()
    left, right = source.gray_424, target.gray_424
    if left.shape != right.shape or not np.isclose(source.output_scale, target.output_scale):
        return None
    lk_started = time.perf_counter()
    lk_x, lk_y, total_weight, count, observations = _lk(
        left,
        right,
        source.output_scale,
        points=source.grid_points,
        gradient=source.gradient_424,
    )
    lk_seconds = time.perf_counter() - lk_started
    lo, hi = int(round(left.shape[1] * 0.15)), int(round(left.shape[1] * 0.85))
    if hanning is None:
        hanning = cv2.createHanningWindow((hi - lo, left.shape[0]), cv2.CV_32F)
    phase_started = time.perf_counter()
    phase_x, phase_y, response = _phase(
        left, right, source.output_scale, hanning=hanning
    )
    phase_seconds = time.perf_counter() - phase_started
    use_lk = lk_x is not None and count >= 8 and total_weight >= 2.0
    selected = lk_x if use_lk else phase_x
    method = (
        "grid_lk" if use_lk
        else "phase_correlation" if selected is not None
        else "unavailable"
    )
    disagreement = (
        lk_x is not None
        and phase_x is not None
        and abs(lk_x - phase_x) > max(3.0, 0.5 * max(abs(lk_x), abs(phase_x)))
    )
    reasons = []
    if count < 8 or total_weight < 2.0:
        reasons.append("low_lk_support")
    if disagreement:
        reasons.append("lk_phase_disagreement")
    if response < 0.05:
        reasons.append("low_phase_response")
    hypothesis_started = time.perf_counter()
    hypotheses = _extract_hypotheses(
        observations,
        selected_advance_px=selected,
        image_width_px=float(left.shape[1]) * source.output_scale,
        image_height_px=float(left.shape[0]) * source.output_scale,
    )
    hypothesis_seconds = time.perf_counter() - hypothesis_started
    if profile is not None:
        profile.update({
            "total_seconds": time.perf_counter() - edge_started,
            "lk_seconds": lk_seconds,
            "phase_seconds": phase_seconds,
            "hypothesis_seconds": hypothesis_seconds,
        })
    return S13MotionEdge(
        source_frame_id=source.frame_id,
        target_frame_id=target.frame_id,
        step=int(step),
        lk_advance_px=lk_x,
        lk_vertical_px=lk_y,
        lk_total_weight=total_weight,
        lk_observation_count=count,
        phase_advance_px=phase_x,
        phase_vertical_px=phase_y,
        phase_response=response,
        selected_advance_px=selected,
        selected_method=method,
        risk=bool(reasons),
        telemetry_only_reasons=tuple(reasons),
        observations=observations,
        motion_hypotheses=hypotheses,
    )


def measure_s13_motion(
    frames: Sequence[S13RenderFrame], *, analysis_width_px: int = 424,
    steps: Sequence[int] = (1, 2, 4),
    prepared_analysis: Sequence[tuple[np.ndarray, float]] | None = None,
    prepared_gradients: Sequence[np.ndarray] | None = None,
    profile: MutableMapping[str, object] | None = None,
) -> tuple[S13MotionEdge, ...]:
    profile_started = time.perf_counter()
    requested_steps = tuple(int(step) for step in steps)
    counters = {
        step: {"edge_count": 0, "total_seconds": 0.0, "lk_seconds": 0.0,
               "phase_seconds": 0.0, "hypothesis_seconds": 0.0}
        for step in requested_steps
    }
    analysis = list(prepared_analysis) if prepared_analysis is not None else None
    if analysis is not None and len(analysis) != len(frames):
        raise ValueError("S1.3 prepared analysis count disagrees with frames")
    # Every source frame participates in up to three step hypotheses.  Feature
    # detection and Sobel are source properties, not pair properties; caching
    # them leaves each LK and phase invocation unchanged.
    gradients = list(prepared_gradients) if prepared_gradients is not None else None
    if gradients is not None and len(gradients) != len(frames):
        raise ValueError("S1.3 prepared gradient count disagrees with frames")
    prepared = tuple(
        prepare_s13_motion_frame(
            frame,
            analysis_width_px=analysis_width_px,
            prepared_analysis=None if analysis is None else analysis[index],
            prepared_gradient=None if gradients is None else gradients[index],
            include_source_identity=False,
        )
        for index, frame in enumerate(frames)
    )
    edges: list[S13MotionEdge] = []
    hanning_by_shape: dict[tuple[int, int], np.ndarray] = {}
    for step in steps:
        for index in range(len(frames) - step):
            edge_profile: dict[str, float] = {}
            left_shape = prepared[index].gray_424.shape
            lo, hi = int(round(left_shape[1] * 0.15)), int(round(left_shape[1] * 0.85))
            phase_shape = (left_shape[0], hi - lo)
            hanning = hanning_by_shape.get(phase_shape)
            if hanning is None:
                hanning = cv2.createHanningWindow(
                    (phase_shape[1], phase_shape[0]), cv2.CV_32F
                )
                hanning_by_shape[phase_shape] = hanning
            edge = measure_s13_motion_edge(
                prepared[index], prepared[index + step], step=int(step),
                profile=edge_profile, hanning=hanning,
            )
            if edge is None:
                continue
            edges.append(edge)
            counters[int(step)]["edge_count"] += 1
            for name in ("total_seconds", "lk_seconds", "phase_seconds", "hypothesis_seconds"):
                counters[int(step)][name] += edge_profile[name]
    if profile is not None:
        bookkeeping_started = time.perf_counter()
        profile.clear()
        for step in (1, 2, 4):
            row = counters.get(step, {})
            profile[f"step{step}_edge_count"] = int(row.get("edge_count", 0))
            for name in ("total_seconds", "lk_seconds", "phase_seconds", "hypothesis_seconds"):
                profile[f"step{step}_{name}"] = float(row.get(name, 0.0))
        profile["requested_steps"] = list(requested_steps)
        profile["profiled_wall_seconds"] = time.perf_counter() - profile_started
        profile["profiling_bookkeeping_seconds"] = time.perf_counter() - bookkeeping_started
    return tuple(edges)


def reliable_step1_direction_evidence(
    edges: Sequence[S13MotionEdge],
) -> tuple[float, ...]:
    """Return the single authoritative reliable step-1 direction evidence."""

    return tuple(
        float(edge.selected_advance_px)
        for edge in edges
        if edge.step == 1
        and edge.selected_advance_px is not None
        and math.isfinite(float(edge.selected_advance_px))
        and abs(float(edge.selected_advance_px)) >= _MIN_SPATIAL_ADVANCE_PX
        and not edge.risk
    )


def _finite_selected_advance(edge: S13MotionEdge | None) -> float | None:
    if edge is None or edge.selected_advance_px is None:
        return None
    value = float(edge.selected_advance_px)
    return value if math.isfinite(value) else None


def _is_confident_near_zero(edge: S13MotionEdge | None) -> bool:
    """Return true only when independent RGB estimators agree on a pause."""

    selected = _finite_selected_advance(edge)
    if edge is None or selected is None or abs(selected) >= _MIN_SPATIAL_ADVANCE_PX or edge.risk:
        return False
    if edge.lk_advance_px is None or edge.phase_advance_px is None:
        return False
    lk = float(edge.lk_advance_px)
    phase = float(edge.phase_advance_px)
    return bool(
        math.isfinite(lk)
        and math.isfinite(phase)
        and edge.lk_observation_count >= _MIN_LK_OBSERVATIONS
        and edge.lk_total_weight >= _MIN_LK_WEIGHT
        and edge.phase_response >= _MIN_PHASE_RESPONSE
        and abs(lk) < _MIN_SPATIAL_ADVANCE_PX
        and abs(phase) < _MIN_SPATIAL_ADVANCE_PX
        and abs(lk - phase) < _MIN_SPATIAL_ADVANCE_PX
    )


def _subpixel_motion_supported_indices(
    frames: Sequence[S13RenderFrame],
    edges: Sequence[S13MotionEdge],
    near_zero_indices: set[int],
) -> set[int]:
    """Use measured step-2/4 displacement to preserve coherent very slow scans."""

    frame_index = {frame.frame_id: index for index, frame in enumerate(frames)}
    supported: set[int] = set()
    for edge in edges:
        if edge.step <= 1 or edge.risk:
            continue
        source = frame_index.get(edge.source_frame_id)
        target = frame_index.get(edge.target_frame_id)
        advance = _finite_selected_advance(edge)
        if (
            source is None
            or target is None
            or target - source != edge.step
            or advance is None
            or abs(advance) < _MIN_SPATIAL_ADVANCE_PX
        ):
            continue
        span = set(range(source, target))
        # A skip edge may validate subpixel motion only when its complete span
        # consists of independently confident near-zero adjacent edges. This
        # prevents a skip edge crossing the start of movement from assigning
        # that later movement to an earlier stationary frame.
        if span and span.issubset(near_zero_indices):
            supported.update(span)
    return supported


def build_basic_s13_progress(frames: Sequence[S13RenderFrame], edges: Sequence[S13MotionEdge]) -> S13Progress:
    adjacent_by_pair = {(edge.source_frame_id, edge.target_frame_id): edge for edge in edges if edge.step == 1}
    reliable = [
        abs(float(edge.selected_advance_px)) for edge in edges
        if (
            edge.step == 1
            and edge.selected_advance_px is not None
            and math.isfinite(float(edge.selected_advance_px))
            and abs(float(edge.selected_advance_px)) >= _MIN_SPATIAL_ADVANCE_PX
            and not edge.risk
        )
    ]
    session_median = float(np.median(reliable)) if reliable else 1.0
    near_zero_indices = {
        index
        for index, (left, right) in enumerate(zip(frames[:-1], frames[1:]))
        if _is_confident_near_zero(adjacent_by_pair.get((left.frame_id, right.frame_id)))
    }
    subpixel_supported = _subpixel_motion_supported_indices(frames, edges, near_zero_indices)
    centers = [0.0]
    methods = ["origin"]
    measured_count = 0
    connected = True
    for index, (left, right) in enumerate(zip(frames[:-1], frames[1:])):
        edge = adjacent_by_pair.get((left.frame_id, right.frame_id))
        observed = _finite_selected_advance(edge)
        if observed is not None and abs(observed) >= _MIN_SPATIAL_ADVANCE_PX:
            advance = abs(observed)
            method = "grid_lk" if edge.selected_method == "grid_lk" else "phase_correlation"
            measured_count += 1
        elif index in near_zero_indices and index in subpixel_supported:
            advance = abs(float(observed))
            base_method = "grid_lk" if edge is not None and edge.selected_method == "grid_lk" else "phase_correlation"
            method = f"{base_method}_subpixel"
            measured_count += 1
        elif index in near_zero_indices:
            # A finite, independently corroborated near-zero edge is evidence
            # that the two real frames are near duplicates. Borrowing the
            # session's moving median here invents distance during a
            # stationary lead-in/out and repeatedly lays the same object
            # across the canvas.
            advance = 0.0
            method = "zero_duplicate"
            connected = False
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
    "S13MotionAccumulator", "S13MotionEdge", "S13MotionSnapshot",
    "S13PreparedMotionFrame", "S13Progress", "build_basic_s13_progress",
    "descriptive_delta_risk", "measure_s13_motion", "measure_s13_motion_edge",
    "prepare_s13_motion_frame", "reliable_step1_direction_evidence",
]
