"""Tail-distribution geometry gate for the isolated v6.1 candidate.

The gate is deliberately pair-local: it observes one cached F/B DIS pass and
does not create RGB, a pose, or a new source.  Edge evidence is symmetric and
orientation compatible.  Rare residual tails become an additional protected
source-to-target band; the absolute maximum is report-only and never vetoes a
pair whose robust P95, coverage, reliable-support, and F/B gates pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .video_visual_renderer import VideoDISPairEvidence


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _bounded_integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class V61GeometryGateConfig:
    """Fail-closed bounds for one 96--160px v6.1 pair corridor."""

    minimum_reliable_pixels: int = 128
    fb_p95_max_px: float = 1.25
    edge_p95_max_px: float = 0.75
    minimum_matched_edge_fraction: float = 0.85
    tail_threshold_px: float = 1.25
    tail_dilation_px: int = 3
    edge_normal_search_px: int = 8
    edge_normal_sample_step_px: float = 0.125
    edge_correspondence_band_px: float = 2.0
    edge_orientation_tolerance_deg: float = 30.0
    maximum_edge_samples_per_direction: int = 4096

    def __post_init__(self) -> None:
        _bounded_integer(
            self.minimum_reliable_pixels,
            "minimum_reliable_pixels",
            128,
            100_000_000,
        )
        fb_limit = _finite_number(self.fb_p95_max_px, "fb_p95_max_px")
        if not 0.0 < fb_limit <= 1.25:
            raise ValueError("fb_p95_max_px must be in (0, 1.25]")
        edge_limit = _finite_number(self.edge_p95_max_px, "edge_p95_max_px")
        if not 0.0 < edge_limit <= 0.75:
            raise ValueError("edge_p95_max_px must be in (0, 0.75]")
        matched_fraction = _finite_number(
            self.minimum_matched_edge_fraction,
            "minimum_matched_edge_fraction",
        )
        if not 0.85 <= matched_fraction <= 1.0:
            raise ValueError("minimum_matched_edge_fraction must be in [0.85, 1]")
        tail_threshold = _finite_number(self.tail_threshold_px, "tail_threshold_px")
        if tail_threshold < 1.25:
            raise ValueError("tail_threshold_px must be at least 1.25")
        _bounded_integer(self.tail_dilation_px, "tail_dilation_px", 1, 8)
        search_px = _bounded_integer(
            self.edge_normal_search_px,
            "edge_normal_search_px",
            1,
            16,
        )
        if tail_threshold >= float(search_px):
            raise ValueError("tail_threshold_px must be below edge_normal_search_px")
        sample_step = _finite_number(
            self.edge_normal_sample_step_px,
            "edge_normal_sample_step_px",
        )
        if not 0.05 <= sample_step <= 0.5:
            raise ValueError("edge_normal_sample_step_px must be in [0.05, 0.5]")
        correspondence_band = _finite_number(
            self.edge_correspondence_band_px,
            "edge_correspondence_band_px",
        )
        if not 0.5 <= correspondence_band <= 4.0:
            raise ValueError("edge_correspondence_band_px must be in [0.5, 4]")
        if correspondence_band > float(search_px):
            raise ValueError("edge_correspondence_band_px cannot exceed edge_normal_search_px")
        orientation = _finite_number(
            self.edge_orientation_tolerance_deg,
            "edge_orientation_tolerance_deg",
        )
        if not 0.0 < orientation <= 45.0:
            raise ValueError("edge_orientation_tolerance_deg must be in (0, 45]")
        _bounded_integer(
            self.maximum_edge_samples_per_direction,
            "maximum_edge_samples_per_direction",
            128,
            65_536,
        )


@dataclass(frozen=True)
class V61GeometryAudit:
    accepted: bool
    rejection_reason: str | None
    reliable_pixel_count: int
    fb_p95_px: float | None
    edge_p95_px: float | None
    edge_p99_px: float | None
    edge_p995_px: float | None
    edge_abs_max_px: float | None
    matched_edge_fraction: float | None
    edge_sample_count: int
    matched_edge_count: int
    forward_edge_sample_count: int
    forward_matched_edge_count: int
    backward_edge_sample_count: int
    backward_matched_edge_count: int
    tail_count: int
    tail_fraction: float | None
    search_boundary_saturation_count: int
    largest_tail_component_pixels: int
    maximum_tail_row_run: int
    tail_occlusion_intersection: int
    tail_low_confidence_intersection: int
    tail_guard_intersection: int
    tail_risk_pixel_count: int
    tail_guard_pixel_count: int
    tail_guard_incremental_pixel_count: int
    tail_guard: np.ndarray
    tail_risk_mask: np.ndarray
    residual_px: np.ndarray
    residual_sample_mask: np.ndarray

    def as_report_dict(self) -> dict[str, bool | int | float | str | None]:
        """Return scalar-only evidence suitable for ``video_report.json``."""

        return {
            "accepted": bool(self.accepted),
            "rejection_reason": self.rejection_reason,
            "reliable_pixel_count": int(self.reliable_pixel_count),
            "fb_p95_px": self.fb_p95_px,
            "edge_p95_px": self.edge_p95_px,
            "edge_p99_px": self.edge_p99_px,
            "edge_p995_px": self.edge_p995_px,
            "edge_abs_max_px": self.edge_abs_max_px,
            "edge_abs_max_gate_role": "telemetry_only",
            "matched_edge_fraction": self.matched_edge_fraction,
            "edge_sample_count": int(self.edge_sample_count),
            "matched_edge_count": int(self.matched_edge_count),
            "forward_edge_sample_count": int(self.forward_edge_sample_count),
            "forward_matched_edge_count": int(self.forward_matched_edge_count),
            "backward_edge_sample_count": int(self.backward_edge_sample_count),
            "backward_matched_edge_count": int(self.backward_matched_edge_count),
            "tail_count": int(self.tail_count),
            "tail_fraction": self.tail_fraction,
            "search_boundary_saturation_count": int(
                self.search_boundary_saturation_count
            ),
            "largest_tail_component_pixels": int(self.largest_tail_component_pixels),
            "maximum_tail_row_run": int(self.maximum_tail_row_run),
            "tail_occlusion_intersection": int(self.tail_occlusion_intersection),
            "tail_low_confidence_intersection": int(
                self.tail_low_confidence_intersection
            ),
            "tail_guard_intersection": int(self.tail_guard_intersection),
            "tail_risk_pixel_count": int(self.tail_risk_pixel_count),
            "tail_guard_pixel_count": int(self.tail_guard_pixel_count),
            "tail_guard_incremental_pixel_count": int(
                self.tail_guard_incremental_pixel_count
            ),
        }

    def report_dict(self) -> dict[str, bool | int | float | str | None]:
        """Compatibility alias for callers that use the shorter name."""

        return self.as_report_dict()


@dataclass(frozen=True)
class _DirectionalEdgeEvidence:
    residual_values: np.ndarray
    residual_px: np.ndarray
    residual_sample_mask: np.ndarray
    tail_risk_mask: np.ndarray
    edge_sample_count: int
    matched_edge_count: int
    tail_count: int
    search_boundary_saturation_count: int


def _p(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return None if not values.size else float(np.percentile(values, percentile))


def _largest_component(mask: np.ndarray) -> int:
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        8,
    )
    return 0 if count <= 1 else int(np.max(stats[1:, cv2.CC_STAT_AREA]))


def _maximum_row_run(mask: np.ndarray) -> int:
    rows = np.any(mask, axis=1)
    best = run = 0
    for value in rows:
        run = run + 1 if value else 0
        best = max(best, run)
    return int(best)


def _empty_direction(shape: tuple[int, int], sample_count: int) -> _DirectionalEdgeEvidence:
    return _DirectionalEdgeEvidence(
        np.empty(0, np.float32),
        np.full(shape, np.nan, np.float32),
        np.zeros(shape, bool),
        np.zeros(shape, bool),
        int(sample_count),
        0,
        0,
        0,
    )


def _directional_edge_normal_evidence(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    support: np.ndarray,
    expected_flow: np.ndarray,
    config: V61GeometryGateConfig,
    *,
    residual_anchor: str,
) -> _DirectionalEdgeEvidence:
    """Measure one orientation-compatible edge-normal direction.

    Candidate target edges are searched along each source-edge normal.  DIS
    is used only to constrain correspondence to the expected target vicinity;
    it is not a pose and it does not alter RGB sampling.  Residual locations
    stay in the shared final-corridor coordinate domain.  Forward values are
    anchored at the old-image source pixel.  Backward values are explicitly
    projected to the selected old-image target endpoint before the two maps
    are merged; a new-image edge raster is never blindly ORed into the old
    residual raster.
    """

    if residual_anchor not in {"source", "target"}:
        raise ValueError("residual_anchor must be source or target")

    source_gray = cv2.cvtColor(
        np.ascontiguousarray(source_bgr),
        cv2.COLOR_BGR2GRAY,
    )
    target_gray = cv2.cvtColor(
        np.ascontiguousarray(target_bgr),
        cv2.COLOR_BGR2GRAY,
    )
    source_edge = (cv2.Canny(source_gray, 80, 160) > 0) & support
    target_edge = (cv2.Canny(target_gray, 80, 160) > 0) & support
    source_y, source_x = np.nonzero(source_edge)
    if not source_x.size:
        return _empty_direction(source_edge.shape, 0)
    stride = max(
        1,
        int(
            np.ceil(
                source_x.size / float(config.maximum_edge_samples_per_direction)
            )
        ),
    )
    source_y = source_y[::stride].astype(np.float32)
    source_x = source_x[::stride].astype(np.float32)
    sample_count = int(source_x.size)
    if not target_edge.any():
        return _empty_direction(source_edge.shape, sample_count)

    integer_y = source_y.astype(np.intp)
    integer_x = source_x.astype(np.intp)
    gradient_x = cv2.Sobel(source_gray, cv2.CV_32F, 1, 0)[integer_y, integer_x]
    gradient_y = cv2.Sobel(source_gray, cv2.CV_32F, 0, 1)[integer_y, integer_x]
    expected = np.asarray(expected_flow, np.float32)[integer_y, integer_x]
    magnitude = np.hypot(gradient_x, gradient_y)
    usable = (
        (magnitude > 1e-6)
        & np.isfinite(expected[:, 0])
        & np.isfinite(expected[:, 1])
    )
    if not usable.any():
        return _empty_direction(source_edge.shape, sample_count)
    source_y, source_x, integer_y, integer_x, gradient_x, gradient_y, magnitude, expected = (
        value[usable]
        for value in (
            source_y,
            source_x,
            integer_y,
            integer_x,
            gradient_x,
            gradient_y,
            magnitude,
            expected,
        )
    )
    normal_x = gradient_x / magnitude
    normal_y = gradient_y / magnitude
    offsets = np.arange(
        -config.edge_normal_search_px,
        config.edge_normal_search_px + 0.5 * config.edge_normal_sample_step_px,
        config.edge_normal_sample_step_px,
        dtype=np.float32,
    )
    target_candidate = cv2.dilate(
        target_edge.astype(np.uint8),
        np.ones((3, 3), np.uint8),
    )
    target_gradient_x = cv2.Sobel(target_gray, cv2.CV_32F, 1, 0)
    target_gradient_y = cv2.Sobel(target_gray, cv2.CV_32F, 0, 1)
    scores = np.full((len(source_x), len(offsets)), -np.inf, np.float32)
    orientation_minimum = math.cos(
        math.radians(config.edge_orientation_tolerance_deg)
    )
    for index, offset in enumerate(offsets):
        map_x = source_x + float(offset) * normal_x
        map_y = source_y + float(offset) * normal_y
        candidate = cv2.remap(
            target_candidate,
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).reshape(-1) > 0
        sampled_x = cv2.remap(
            target_gradient_x,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).reshape(-1)
        sampled_y = cv2.remap(
            target_gradient_y,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        ).reshape(-1)
        sampled_magnitude = np.hypot(sampled_x, sampled_y)
        orientation = np.zeros_like(sampled_magnitude)
        valid_orientation = sampled_magnitude > 1e-6
        orientation[valid_orientation] = np.abs(
            sampled_x[valid_orientation] * normal_x[valid_orientation]
            + sampled_y[valid_orientation] * normal_y[valid_orientation]
        ) / sampled_magnitude[valid_orientation]
        correspondence = (
            map_x - (source_x + expected[:, 0])
        ) ** 2 + (
            map_y - (source_y + expected[:, 1])
        ) ** 2
        eligible = (
            candidate
            & (orientation >= orientation_minimum)
            & (correspondence <= config.edge_correspondence_band_px**2)
        )
        # Prefer the nearest compatible contour.  Gradient strength only
        # breaks exact ties so a stronger parallel contour cannot steal the
        # match and manufacture a large residual.
        scores[:, index] = np.where(
            eligible,
            -abs(float(offset)) + 1e-6 * sampled_magnitude * orientation,
            -np.inf,
        )

    best_index = np.argmax(scores, axis=1)
    best_score = scores[np.arange(len(source_x)), best_index]
    finite_match = np.isfinite(best_score)
    saturated = finite_match & (
        (best_index == 0) | (best_index == len(offsets) - 1)
    )
    matched = finite_match & ~saturated
    if not matched.any():
        empty = _empty_direction(source_edge.shape, sample_count)
        return _DirectionalEdgeEvidence(
            empty.residual_values,
            empty.residual_px,
            empty.residual_sample_mask,
            empty.tail_risk_mask,
            sample_count,
            0,
            0,
            int(np.count_nonzero(saturated)),
        )

    selected_offsets = offsets[best_index[matched]]
    residual_values = np.abs(selected_offsets).astype(np.float32)
    matched_x = source_x[matched]
    matched_y = source_y[matched]
    matched_normal_x = normal_x[matched]
    matched_normal_y = normal_y[matched]
    matched_integer_x = integer_x[matched]
    matched_integer_y = integer_y[matched]
    target_x = matched_x + selected_offsets * matched_normal_x
    target_y = matched_y + selected_offsets * matched_normal_y
    if residual_anchor == "target":
        anchor_x = np.clip(np.rint(target_x).astype(np.intp), 0, source_edge.shape[1] - 1)
        anchor_y = np.clip(np.rint(target_y).astype(np.intp), 0, source_edge.shape[0] - 1)
    else:
        anchor_x = matched_integer_x
        anchor_y = matched_integer_y
    residual_px = np.full(source_edge.shape, np.nan, np.float32)
    # More than one directional sample can project onto the same output pixel.
    # Preserve the worst finite residual at that canonical location.
    for row, column, value in zip(anchor_y, anchor_x, residual_values, strict=True):
        previous = residual_px[row, column]
        if not np.isfinite(previous) or value > previous:
            residual_px[row, column] = value
    residual_sample_mask = np.zeros(source_edge.shape, bool)
    residual_sample_mask[anchor_y, anchor_x] = True

    tail = residual_values > config.tail_threshold_px
    tail_risk = np.zeros(source_edge.shape, np.uint8)
    for source_column, source_row, normal_column, normal_row, offset in zip(
        matched_x[tail],
        matched_y[tail],
        matched_normal_x[tail],
        matched_normal_y[tail],
        selected_offsets[tail],
        strict=True,
    ):
        source_point = (int(round(float(source_column))), int(round(float(source_row))))
        target_point = (
            int(round(float(source_column + offset * normal_column))),
            int(round(float(source_row + offset * normal_row))),
        )
        # This line, rather than only the source-edge pixel, is what protects
        # the source/target ambiguity band.  Running both directions also
        # catches contours visible only from the incoming-source side.
        cv2.line(tail_risk, source_point, target_point, 1, 1)
    return _DirectionalEdgeEvidence(
        residual_values,
        residual_px,
        residual_sample_mask,
        tail_risk.astype(bool) & support,
        sample_count,
        int(np.count_nonzero(matched)),
        int(np.count_nonzero(tail)),
        int(np.count_nonzero(saturated)),
    )


def _validate_inputs(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    evidence: VideoDISPairEvidence,
    support: np.ndarray,
    protected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    old = np.asarray(old_bgr)
    new = np.asarray(new_bgr)
    if (
        old.dtype != np.uint8
        or new.dtype != np.uint8
        or old.ndim != 3
        or new.ndim != 3
        or old.shape != new.shape
        or old.shape[2] != 3
    ):
        raise ValueError("v6.1 geometry gate requires matching uint8 BGR corridors")
    shape = old.shape[:2]
    support_mask = np.asarray(support, bool)
    protected_mask = np.asarray(protected, bool)
    if support_mask.shape != shape or protected_mask.shape != shape:
        raise ValueError("support and protected masks must match the pair corridor")
    expected_shapes = {
        "flow_forward": (shape[0], shape[1], 2),
        "flow_backward": (shape[0], shape[1], 2),
        "fb_error": shape,
        "occlusion_risk_mask": shape,
        "correspondence_confidence": shape,
        "reliable_mask": shape,
    }
    for name, expected_shape in expected_shapes.items():
        if np.asarray(getattr(evidence, name)).shape != expected_shape:
            raise ValueError(f"DIS {name} must match the pair corridor")
    return old, new, support_mask, protected_mask


def _merge_residual_maps(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(first, np.float32).copy()
    second_values = np.asarray(second, np.float32)
    take_second = np.isfinite(second_values) & (
        ~np.isfinite(residual) | (second_values > residual)
    )
    residual[take_second] = second_values[take_second]
    return residual, np.isfinite(residual)


def _collect_matched_edge_evidence(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    evidence: VideoDISPairEvidence,
    support: np.ndarray,
    config: V61GeometryGateConfig,
) -> tuple[_DirectionalEdgeEvidence, _DirectionalEdgeEvidence, np.ndarray, np.ndarray, np.ndarray]:
    forward = _directional_edge_normal_evidence(
        old_bgr,
        new_bgr,
        support,
        np.asarray(evidence.flow_forward, np.float32),
        config,
        residual_anchor="source",
    )
    backward = _directional_edge_normal_evidence(
        new_bgr,
        old_bgr,
        support,
        np.asarray(evidence.flow_backward, np.float32),
        config,
        residual_anchor="target",
    )
    residual, sample_mask = _merge_residual_maps(
        forward.residual_px,
        backward.residual_px,
    )
    tail_risk = forward.tail_risk_mask | backward.tail_risk_mask
    return forward, backward, residual, sample_mask, tail_risk


def matched_edge_residuals(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    evidence: VideoDISPairEvidence,
    support: np.ndarray,
    *,
    config: V61GeometryGateConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return symmetric residual locations in the shared final corridor.

    The sample count is directional (forward plus backward), while the mask is
    a unified corridor raster and therefore intentionally de-duplicates pixels
    sampled in both directions.
    """

    settings = config or V61GeometryGateConfig()
    old, new, support_mask, _protected = _validate_inputs(
        old_bgr,
        new_bgr,
        evidence,
        support,
        np.zeros(np.asarray(old_bgr).shape[:2], bool),
    )
    forward, backward, residual, samples, _tail_risk = _collect_matched_edge_evidence(
        old,
        new,
        evidence,
        support_mask,
        settings,
    )
    return residual, samples, forward.edge_sample_count + backward.edge_sample_count


def evaluate_v61_geometry_gate(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    evidence: VideoDISPairEvidence,
    *,
    support: np.ndarray,
    protected: np.ndarray,
    config: V61GeometryGateConfig | None = None,
) -> V61GeometryAudit:
    """Apply robust hard gates and turn residual tails into protection."""

    settings = config or V61GeometryGateConfig()
    old, new, support_mask, protected_mask = _validate_inputs(
        old_bgr,
        new_bgr,
        evidence,
        support,
        protected,
    )
    reliable = support_mask & np.asarray(evidence.reliable_mask, bool)
    forward, backward, residual, samples, raw_tail_risk = _collect_matched_edge_evidence(
        old,
        new,
        evidence,
        support_mask,
        settings,
    )
    values = np.concatenate((forward.residual_values, backward.residual_values))
    sample_count = forward.edge_sample_count + backward.edge_sample_count
    matched_count = forward.matched_edge_count + backward.matched_edge_count
    fraction = None if sample_count == 0 else float(matched_count) / sample_count
    tail_count = forward.tail_count + backward.tail_count
    tail_fraction = None if matched_count == 0 else float(tail_count) / matched_count

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (settings.tail_dilation_px * 2 + 1,) * 2,
    )
    tail_risk = cv2.dilate(raw_tail_risk.astype(np.uint8), kernel).astype(bool)
    tail_risk &= support_mask
    # ``protected`` already contains the ordinary source and target edge
    # collars.  Emit only the additional tail-specific pixels as tail_guard;
    # their union remains the complete protected source-to-target risk band.
    # This avoids reporting a redundant Canny mask as a v6.1 improvement.
    tail_guard = tail_risk & ~protected_mask

    fb_p95 = _p(np.asarray(evidence.fb_error)[reliable], 95.0)
    edge_p95 = _p(values, 95.0)
    reasons: list[str] = []
    if int(reliable.sum()) < settings.minimum_reliable_pixels:
        reasons.append("minimum_reliable_pixels")
    if fb_p95 is None or fb_p95 > settings.fb_p95_max_px:
        reasons.append("fb_p95")
    if edge_p95 is None or edge_p95 > settings.edge_p95_max_px:
        reasons.append("edge_p95")
    if fraction is None or fraction < settings.minimum_matched_edge_fraction:
        reasons.append("matched_edge_fraction")

    occlusion = np.asarray(evidence.occlusion_risk_mask, bool)
    low_confidence = (
        ~np.asarray(evidence.reliable_mask, bool)
        | ~np.isfinite(np.asarray(evidence.correspondence_confidence, np.float32))
        | (np.asarray(evidence.correspondence_confidence, np.float32) <= 0.0)
    )
    return V61GeometryAudit(
        accepted=not reasons,
        rejection_reason=None if not reasons else ",".join(reasons),
        reliable_pixel_count=int(reliable.sum()),
        fb_p95_px=fb_p95,
        edge_p95_px=edge_p95,
        edge_p99_px=_p(values, 99.0),
        edge_p995_px=_p(values, 99.5),
        edge_abs_max_px=None if not values.size else float(np.max(values)),
        matched_edge_fraction=fraction,
        edge_sample_count=sample_count,
        matched_edge_count=matched_count,
        forward_edge_sample_count=forward.edge_sample_count,
        forward_matched_edge_count=forward.matched_edge_count,
        backward_edge_sample_count=backward.edge_sample_count,
        backward_matched_edge_count=backward.matched_edge_count,
        tail_count=tail_count,
        tail_fraction=tail_fraction,
        search_boundary_saturation_count=(
            forward.search_boundary_saturation_count
            + backward.search_boundary_saturation_count
        ),
        largest_tail_component_pixels=_largest_component(raw_tail_risk),
        maximum_tail_row_run=_maximum_row_run(raw_tail_risk),
        tail_occlusion_intersection=int(np.count_nonzero(tail_risk & occlusion)),
        tail_low_confidence_intersection=int(
            np.count_nonzero(tail_risk & low_confidence)
        ),
        tail_guard_intersection=int(
            np.count_nonzero(tail_risk & protected_mask)
        ),
        tail_risk_pixel_count=int(np.count_nonzero(tail_risk)),
        tail_guard_pixel_count=int(np.count_nonzero(tail_guard)),
        tail_guard_incremental_pixel_count=int(np.count_nonzero(tail_guard)),
        tail_guard=np.ascontiguousarray(tail_guard),
        tail_risk_mask=np.ascontiguousarray(tail_risk),
        residual_px=np.ascontiguousarray(residual),
        residual_sample_mask=np.ascontiguousarray(samples),
    )


__all__ = [
    "V61GeometryAudit",
    "V61GeometryGateConfig",
    "evaluate_v61_geometry_gate",
    "matched_edge_residuals",
]
