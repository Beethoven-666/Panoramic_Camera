"""Tail-distribution geometry gate for the isolated v6.1 candidate.

The gate is deliberately pair-local: it observes one cached F/B DIS pass and
does not create RGB, a pose, or a new source.  Unlike the Phase-1 POC it
records residual locations so rare outliers become hard protected tail guards
rather than an all-pair absolute-maximum veto.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .video_visual_renderer import VideoDISPairEvidence


@dataclass(frozen=True)
class V61GeometryGateConfig:
    minimum_reliable_pixels: int = 128
    fb_p95_max_px: float = 1.25
    edge_p95_max_px: float = 0.75
    minimum_matched_edge_fraction: float = 0.85
    tail_threshold_px: float = 1.25
    tail_dilation_px: int = 3


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
    tail_count: int
    tail_fraction: float | None
    search_boundary_saturation_count: int
    largest_tail_component_pixels: int
    maximum_tail_row_run: int
    tail_occlusion_intersection: int
    tail_low_confidence_intersection: int
    tail_guard_intersection: int
    tail_guard: np.ndarray
    residual_px: np.ndarray
    residual_sample_mask: np.ndarray


def _p(values: np.ndarray, percentile: float) -> float | None:
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    return None if not values.size else float(np.percentile(values, percentile))


def _largest_component(mask: np.ndarray) -> int:
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return 0 if count <= 1 else int(np.max(stats[1:, cv2.CC_STAT_AREA]))


def _maximum_row_run(mask: np.ndarray) -> int:
    rows = np.any(mask, axis=1)
    best = run = 0
    for value in rows:
        run = run + 1 if value else 0
        best = max(best, run)
    return int(best)


def matched_edge_residuals(
    old_bgr: np.ndarray, new_bgr: np.ndarray, evidence: VideoDISPairEvidence, support: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return per-pixel edge residuals and locations in the final corridor.

    DIS supplies only the correspondence centre.  A distance transform over
    target Canny edges measures the nearest edge after that correspondence;
    values at the 8px search boundary are kept as telemetry/saturation.
    """
    old_gray = cv2.cvtColor(np.asarray(old_bgr), cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(np.asarray(new_bgr), cv2.COLOR_BGR2GRAY)
    old_edge = (cv2.Canny(old_gray, 80, 160) > 0) & np.asarray(support, bool)
    new_edge = cv2.Canny(new_gray, 80, 160) > 0
    distance = cv2.distanceTransform((~new_edge).astype(np.uint8), cv2.DIST_L2, 3)
    height, width = old_edge.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    flow = np.asarray(evidence.flow_forward, np.float32)
    mapped = cv2.remap(distance, xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=8.0)
    residual = np.full(old_edge.shape, np.nan, np.float32)
    residual[old_edge] = np.minimum(mapped[old_edge], 8.0)
    return residual, old_edge, int(old_edge.sum())


def evaluate_v61_geometry_gate(
    old_bgr: np.ndarray, new_bgr: np.ndarray, evidence: VideoDISPairEvidence, *,
    support: np.ndarray, protected: np.ndarray, config: V61GeometryGateConfig | None = None,
) -> V61GeometryAudit:
    """Apply robust P95/coverage gates and turn tail locations into a guard."""
    settings = config or V61GeometryGateConfig()
    support, protected = np.asarray(support, bool), np.asarray(protected, bool)
    reliable = support & np.asarray(evidence.reliable_mask, bool)
    residual, samples, sample_count = matched_edge_residuals(old_bgr, new_bgr, evidence, support)
    values = residual[samples]
    values = values[np.isfinite(values)]
    matched = values < 8.0
    matched_count = int(matched.sum())
    fraction = None if sample_count == 0 else matched_count / sample_count
    tail_seed = samples & np.isfinite(residual) & (residual > settings.tail_threshold_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (settings.tail_dilation_px * 2 + 1,) * 2)
    tail_guard = cv2.dilate(tail_seed.astype(np.uint8), kernel).astype(bool) & support
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
    return V61GeometryAudit(
        not reasons, None if not reasons else ",".join(reasons), int(reliable.sum()), fb_p95, edge_p95,
        _p(values, 99.0), _p(values, 99.5), (None if not values.size else float(values.max())), fraction,
        sample_count, matched_count, int(tail_seed.sum()), (None if not sample_count else float(tail_seed.sum()) / sample_count),
        int(np.count_nonzero(samples & (residual >= 8.0))), _largest_component(tail_seed), _maximum_row_run(tail_seed),
        int(np.count_nonzero(tail_seed & np.asarray(evidence.occlusion_risk_mask, bool))),
        int(np.count_nonzero(tail_seed & ~np.asarray(evidence.reliable_mask, bool))),
        int(np.count_nonzero(tail_guard & protected)), tail_guard, residual, samples,
    )


__all__ = ["V61GeometryAudit", "V61GeometryGateConfig", "evaluate_v61_geometry_gate", "matched_edge_residuals"]
