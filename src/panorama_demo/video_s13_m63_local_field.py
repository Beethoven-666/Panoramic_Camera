"""Continuous pair-tapered scalar correction fields for M6.3 QL."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .video_s13_photometric import S13PhotometricSampleSet
from .video_s13_replay import S13P2ReplayPair


def evaluate_s13_m63_ql_shadow(
    samples: Sequence[S13PhotometricSampleSet],
    *,
    excluded_pair_indices: frozenset[int] = frozenset(),
    maximum_log_gain_absolute: float = 0.04,
    minimum_pair_median_benefit_fraction: float = 0.02,
    maximum_pair_p95_regression_fraction: float = 0.01,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for item in samples:
        if item.edge_kind != "adjacent" or item.pair_index in excluded_pair_indices:
            continue
        left = np.asarray(item.heldout_left_rgb_linear, np.float64)
        right = np.asarray(item.heldout_right_rgb_linear, np.float64)
        if not len(left):
            continue
        left_luma = np.maximum(left.mean(axis=1), 1e-6)
        right_luma = np.maximum(right.mean(axis=1), 1e-6)
        relation = float(np.median(np.log(left_luma) - np.log(right_luma)))
        half = float(np.clip(0.5 * relation, -maximum_log_gain_absolute, maximum_log_gain_absolute))
        before = np.linalg.norm(left - right, axis=1)
        after = np.linalg.norm(left * np.exp(-half) - right * np.exp(half), axis=1)
        before_median = float(np.median(before))
        before_p95 = float(np.quantile(before, 0.95))
        after_median = float(np.median(after))
        after_p95 = float(np.quantile(after, 0.95))
        accepted = bool(
            before_median - after_median
            >= minimum_pair_median_benefit_fraction * before_median
            and after_p95 <= before_p95 * (1.0 + maximum_pair_p95_regression_fraction)
        )
        rows.append({
            "pair_index": int(item.pair_index),
            "relation": relation,
            "left_log_correction": -half,
            "right_log_correction": half,
            "before_median_linear": before_median,
            "after_median_linear": after_median,
            "before_p95_linear": before_p95,
            "after_p95_linear": after_p95,
            "accepted": accepted,
        })
    return {
        "model": "QL_pair_tapered_scalar_gain",
        "authority": "shadow",
        "active_pair_count": sum(item["accepted"] is True for item in rows),
        "pairs": rows,
    }


def build_s13_m63_local_log_gain_fields(
    source_count: int,
    replay_pairs: Sequence[S13P2ReplayPair],
    *,
    canvas_shape: tuple[int, int],
    pair_relations: Mapping[int, float],
    support_width_px: int,
    maximum_log_gain_absolute: float = 0.04,
) -> tuple[np.ndarray, ...]:
    """Compose order-independent cosine fields in log space."""

    if support_width_px <= 0:
        raise ValueError("QL support width must be positive")
    fields = [np.zeros(canvas_shape, np.float32) for _ in range(source_count)]
    for pair in replay_pairs:
        if pair.pair_index not in pair_relations:
            continue
        half = float(np.clip(
            0.5 * pair_relations[pair.pair_index],
            -maximum_log_gain_absolute,
            maximum_log_gain_absolute,
        ))
        for row, seam in enumerate(np.asarray(pair.seam_x_by_row, np.int32)):
            x0 = max(0, int(seam) - support_width_px + 1)
            x1 = min(canvas_shape[1], int(seam) + support_width_px)
            columns = np.arange(x0, x1, dtype=np.int32)
            distance = np.abs(columns - int(seam)).astype(np.float32)
            taper = 0.5 * (1.0 + np.cos(np.pi * distance / float(support_width_px)))
            taper[distance >= support_width_px] = 0.0
            left = columns <= int(seam)
            right = columns >= int(seam)
            fields[pair.left_source_index][row, columns[left]] += -half * taper[left]
            fields[pair.right_source_index][row, columns[right]] += half * taper[right]
    return tuple(
        np.clip(field, -maximum_log_gain_absolute, maximum_log_gain_absolute)
        for field in fields
    )


__all__ = [
    "build_s13_m63_local_log_gain_fields",
    "evaluate_s13_m63_ql_shadow",
]
