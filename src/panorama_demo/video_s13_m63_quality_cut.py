"""Robust population quality classes for S1.3 M6.3 photometric edges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .video_s13_photometric import S13PhotometricSampleSet


@dataclass(frozen=True)
class S13M63PairQuality:
    pair_index: int
    classification: str
    heldout_sample_count: int
    baseline_median_linear: float | None
    baseline_p95_linear: float | None
    log_luminance_mad: float | None
    inlier_fraction: float | None
    reasons: tuple[str, ...]


def _pair_statistics(sample: S13PhotometricSampleSet) -> dict[str, float | int | None]:
    left = np.asarray(sample.heldout_left_rgb_linear, np.float64)
    right = np.asarray(sample.heldout_right_rgb_linear, np.float64)
    if not len(left):
        return {"count": 0, "median": None, "p95": None, "mad": None, "inlier": None}
    residual = np.linalg.norm(left - right, axis=1)
    relation = np.log(np.maximum(left.mean(axis=1), 1e-6)) - np.log(
        np.maximum(right.mean(axis=1), 1e-6)
    )
    center = float(np.median(relation))
    absolute = np.abs(relation - center)
    mad = float(np.median(absolute))
    scale = max(1.4826 * mad, 1e-6)
    return {
        "count": int(len(left)),
        "median": float(np.median(residual)),
        "p95": float(np.quantile(residual, 0.95)),
        "mad": mad,
        "inlier": float(np.mean(absolute <= 3.0 * scale)),
    }


def classify_s13_m63_pairs(
    samples: Sequence[S13PhotometricSampleSet],
    *,
    soft_absolute_pair_p95_linear: float = 0.035,
    hard_absolute_pair_p95_linear: float = 0.060,
    soft_population_mad_multiplier: float = 3.0,
    hard_population_mad_multiplier: float = 6.0,
    soft_minimum_inlier_fraction: float = 0.85,
    hard_minimum_inlier_fraction: float = 0.75,
    soft_maximum_log_luminance_mad: float = 0.040,
    hard_maximum_log_luminance_mad: float = 0.060,
    minimum_heldout_samples: int = 192,
    forced_hard_pair_indices: frozenset[int] = frozenset(),
) -> tuple[S13M63PairQuality, ...]:
    """Classify adjacent edges from their population, never from fixed indices."""

    adjacent = [item for item in samples if item.edge_kind == "adjacent"]
    statistics = [_pair_statistics(item) for item in adjacent]
    finite_p95 = np.asarray(
        [float(item["p95"]) for item in statistics if item["p95"] is not None], np.float64
    )
    population_center = float(np.median(finite_p95)) if finite_p95.size else 0.0
    population_mad = (
        float(np.median(np.abs(finite_p95 - population_center))) if finite_p95.size else 0.0
    )
    soft_p95 = max(
        soft_absolute_pair_p95_linear,
        population_center + soft_population_mad_multiplier * population_mad,
    )
    hard_p95 = max(
        hard_absolute_pair_p95_linear,
        population_center + hard_population_mad_multiplier * population_mad,
    )
    result: list[S13M63PairQuality] = []
    for sample, stats in zip(adjacent, statistics, strict=True):
        hard: list[str] = []
        soft: list[str] = []
        count = int(stats["count"])
        p95 = stats["p95"]
        mad = stats["mad"]
        inlier = stats["inlier"]
        if sample.pair_index in forced_hard_pair_indices:
            hard.append("candidate_regression")
        if p95 is None or mad is None or inlier is None:
            hard.append("nonfinite_or_missing_evidence")
        else:
            if float(p95) > hard_p95:
                hard.append("baseline_p95_hard")
            elif float(p95) > soft_p95:
                soft.append("baseline_p95_soft")
            if float(mad) > hard_maximum_log_luminance_mad:
                hard.append("log_luminance_mad_hard")
            elif float(mad) > soft_maximum_log_luminance_mad:
                soft.append("log_luminance_mad_soft")
            if float(inlier) < hard_minimum_inlier_fraction:
                hard.append("inlier_fraction_hard")
            elif float(inlier) < soft_minimum_inlier_fraction:
                soft.append("inlier_fraction_soft")
        if count < minimum_heldout_samples:
            hard.append("insufficient_heldout_samples")
        classification = "photometric_quality_cut" if hard else (
            "soft_low_confidence" if soft else "trusted"
        )
        result.append(S13M63PairQuality(
            pair_index=int(sample.pair_index),
            classification=classification,
            heldout_sample_count=count,
            baseline_median_linear=(None if stats["median"] is None else float(stats["median"])),
            baseline_p95_linear=(None if p95 is None else float(p95)),
            log_luminance_mad=(None if mad is None else float(mad)),
            inlier_fraction=(None if inlier is None else float(inlier)),
            reasons=tuple(hard or soft),
        ))
    return tuple(result)


__all__ = ["S13M63PairQuality", "classify_s13_m63_pairs"]
