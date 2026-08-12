"""Post-render S013 M6.1 quality v2 evaluator.

The evaluator is observational: it has no candidate-selection, rendering, seal,
or pointer authority.  All masks and splits come from sealed P2-v4 pair arrays.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


QUALITY_SCHEMA = "gemini305-video-s13-p3-diagnostic-quality/v2"


def _lab(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("quality images must be uint8 BGR")
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).astype(np.float64)


def ciede2000(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Vectorized Sharma 2005 CIEDE2000 for OpenCV L*a*b* arrays."""
    l1, a1, b1 = np.moveaxis(np.asarray(left, np.float64), -1, 0)
    l2, a2, b2 = np.moveaxis(np.asarray(right, np.float64), -1, 0)
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cbar**7 / (cbar**7 + 25**7)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360)
    hp2 = np.mod(np.degrees(np.arctan2(b2, ap2)), 360)
    hp1 = np.where((ap1 == 0) & (b1 == 0), 0, hp1)
    hp2 = np.where((ap2 == 0) & (b2 == 0), 0, hp2)
    dl, dc = l2 - l1, cp2 - cp1
    dh = hp2 - hp1
    dh = np.where((cp1 * cp2) == 0, 0, dh)
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    d_h = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh / 2))
    lbar, cpbar = (l1 + l2) / 2, (cp1 + cp2) / 2
    hsum = hp1 + hp2
    hdiff = np.abs(hp1 - hp2)
    hpbar = np.where((cp1 * cp2) == 0, hsum, hsum / 2)
    hpbar = np.where(((cp1 * cp2) != 0) & (hdiff > 180) & (hsum < 360), (hsum + 360) / 2, hpbar)
    hpbar = np.where(((cp1 * cp2) != 0) & (hdiff > 180) & (hsum >= 360), (hsum - 360) / 2, hpbar)
    t = (1 - 0.17 * np.cos(np.radians(hpbar - 30)) + 0.24 * np.cos(np.radians(2 * hpbar))
         + 0.32 * np.cos(np.radians(3 * hpbar + 6)) - 0.20 * np.cos(np.radians(4 * hpbar - 63)))
    sl = 1 + 0.015 * (lbar - 50) ** 2 / np.sqrt(20 + (lbar - 50) ** 2)
    sc, sh = 1 + 0.045 * cpbar, 1 + 0.015 * cpbar * t
    rt = -2 * np.sqrt(cpbar**7 / (cpbar**7 + 25**7)) * np.sin(
        np.radians(60 * np.exp(-((hpbar - 275) / 25) ** 2))
    )
    return np.sqrt(np.maximum(0, (dl / sl) ** 2 + (dc / sc) ** 2 + (d_h / sh) ** 2 + rt * (dc / sc) * (d_h / sh)))


def _pair_values(lab: np.ndarray, pair: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(pair["canvas_x"], np.int64)
    y = np.asarray(pair["canvas_y"], np.int64)
    safe = np.asarray(pair.get("quality_output_safe", pair.get("general_safe")), bool)
    validation = np.asarray(pair.get("quality_metric_validation", pair["validation"]), bool)
    protected = np.asarray(pair.get("quality_output_protected", pair.get("protected")), bool)
    seam = np.asarray(pair["seam_x_by_row"], np.int64)
    if x.shape != y.shape or safe.shape != x.shape or validation.shape != x.shape or protected.shape != x.shape:
        raise ValueError("sealed pair array shapes disagree")
    if seam.ndim != 1:
        raise ValueError("seam_x_by_row must be one dimensional")
    eligible = safe & validation & ~protected
    distances = np.abs(x - seam[y])
    left_points: list[tuple[int, int]] = []
    right_points: list[tuple[int, int]] = []
    rows: list[int] = []
    offsets: list[int] = []
    has_frozen_mirror = "quality_mirror_canvas_x" in pair
    mirror_x = np.asarray(pair.get("quality_mirror_canvas_x", x), np.int64)
    for yy, xx, mirrored, distance in zip(
        y[eligible], x[eligible], mirror_x[eligible], distances[eligible], strict=True,
    ):
        row, offset = int(yy), int(distance)
        if has_frozen_mirror:
            left_x, right_x = sorted((int(xx), int(mirrored)))
        else:
            left_x, right_x = int(seam[row] - 1 - offset), int(seam[row] + offset)
        if left_x < 0 or right_x >= lab.shape[1]:
            continue
        left_points.append((row, left_x))
        right_points.append((row, right_x))
        rows.append(row)
        offsets.append(offset)
    if left_points:
        left_index = np.asarray(left_points, np.int64)
        right_index = np.asarray(right_points, np.int64)
        left_lab = lab[left_index[:, 0], left_index[:, 1]]
        right_lab = lab[right_index[:, 0], right_index[:, 1]]
        samples = ciede2000(left_lab, right_lab)
        linear = np.abs(left_lab[:, 0] - right_lab[:, 0]) / 100.0
    else:
        samples = linear = np.empty(0, np.float64)
    return samples, linear, np.asarray(rows, np.int64), np.asarray(offsets, np.int64)


def _pair_report(image: np.ndarray, pair: Mapping[str, np.ndarray], pair_index: int) -> dict[str, Any]:
    values, linear, rows, offsets = _pair_values(_lab(image), pair)
    tile_count = len(set(zip((rows // 16).tolist(), (offsets // 16).tolist(), strict=True)))
    block_values = [float(np.percentile(values[rows // 32 == block], 95)) for block in np.unique(rows // 32)] if values.size else []
    evaluable = bool(values.size >= 128 and tile_count >= 8 and len(block_values) >= 4)
    median = float(np.median(values)) if values.size else None
    p95 = float(np.percentile(values, 95)) if values.size else None
    block_p95 = float(max(block_values)) if block_values else None
    immediate = values[offsets <= 1]
    immediate_p95 = float(np.percentile(immediate, 95)) if immediate.size else None
    return {
        "pair_index": pair_index, "classification": "quality_safe_evaluable" if evaluable else "quality_insufficient_evidence",
        "evaluable": evaluable, "safe_validation_sample_count": int(values.size),
        "independent_tile_count": tile_count, "vertical_block_count": len(block_values),
        "wide_safe_delta_e00_median": median, "wide_safe_delta_e00_p95": p95,
        "safe_pair_block_p95": block_p95, "immediate_seam_delta_e00_p95": immediate_p95,
        "aggregate_validation_p95": float(np.percentile(linear, 95)) if linear.size else None,
        "_wide_values": values,
    }


def _global_metrics(
    p2: np.ndarray, image: np.ndarray, owner_source: np.ndarray, valid: np.ndarray,
    blend_active: np.ndarray | None,
) -> dict[str, Any]:
    p2_lab, lab = _lab(p2), _lab(image)
    clipped = np.any((image <= 1) | (image >= 254), axis=2) & valid
    dark = valid & (p2_lab[..., 0] <= 20)
    dark_delta = lab[..., 0][dark] - p2_lab[..., 0][dark]
    neutral = valid & (np.hypot(p2_lab[..., 1], p2_lab[..., 2]) <= 8)
    chroma = np.hypot(lab[..., 1] - p2_lab[..., 1], lab[..., 2] - p2_lab[..., 2])[neutral]
    owner_colors: dict[int, np.ndarray] = {}
    for source in np.unique(owner_source[valid]):
        if source < 0:
            continue
        own = valid & (owner_source == source)
        if np.count_nonzero(own) >= 16:
            owner_colors[int(source)] = np.median(lab[own], axis=0)
    plateau = [
        float(ciede2000(owner_colors[left], owner_colors[right]))
        for left, right in zip(sorted(owner_colors)[:-1], sorted(owner_colors)[1:], strict=True)
        if right == left + 1
    ]
    x = np.broadcast_to(np.arange(image.shape[1]), valid.shape)[valid].astype(np.float64)
    delta_l = (lab[..., 0] - p2_lab[..., 0])[valid]
    slope = float(abs(np.polyfit(x, delta_l, 1)[0]) * image.shape[1]) if x.size >= 2 and np.ptp(x) else 0.0
    gradient_ratio: float | None = None
    gradient_evaluable = False
    ghost_count = 0
    if blend_active is not None and np.any(blend_active):
        gray0 = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g0 = np.hypot(cv2.Sobel(gray0, cv2.CV_32F, 1, 0), cv2.Sobel(gray0, cv2.CV_32F, 0, 1))[blend_active]
        g1 = np.hypot(cv2.Sobel(gray1, cv2.CV_32F, 1, 0), cv2.Sobel(gray1, cv2.CV_32F, 0, 1))[blend_active]
        strong = g0 >= 2.0
        gradient_evaluable = bool(np.count_nonzero(strong) >= 32)
        if gradient_evaluable:
            gradient_ratio = float(np.median(g1[strong]) / max(np.median(g0[strong]), 1e-6))
            ghost_count = int(np.count_nonzero((g1 > g0 * 1.5 + 4) & (g0 >= 2.0)))
    return {
        "clipping_fraction": float(np.count_nonzero(clipped) / max(np.count_nonzero(valid), 1)),
        "dark_lift_median": float(max(0, np.median(dark_delta))) if dark_delta.size else 0.0,
        "dark_lift_p95": float(max(0, np.percentile(dark_delta, 95))) if dark_delta.size else 0.0,
        "neutral_chroma_drift": float(np.percentile(chroma, 95)) if chroma.size else 0.0,
        "owner_plateau_delta_e00": float(max(plateau, default=0.0)),
        "x_slope_added_drift": slope,
        "b1_gradient_ratio": gradient_ratio,
        "b1_gradient_ratio_evaluable": gradient_evaluable,
        "ghost_new_material_count": ghost_count,
    }


def _aggregate(pair_reports: Sequence[Mapping[str, Any]], global_metrics: Mapping[str, Any], visible_target: float) -> dict[str, Any]:
    evaluable = [row for row in pair_reports if row["evaluable"]]
    medians = [float(row["wide_safe_delta_e00_median"]) for row in evaluable]
    p95s = [float(row["wide_safe_delta_e00_p95"]) for row in evaluable]
    linear_p95s = [float(row["aggregate_validation_p95"]) for row in evaluable]
    micro_values = np.concatenate([np.asarray(row["_wide_values"]) for row in evaluable]) if evaluable else np.empty(0)
    return {
        "wide_safe_delta_e00_median": float(np.median(medians)) if medians else None,
        "wide_safe_delta_e00_p95": float(np.percentile(p95s, 95)) if p95s else None,
        "wide_safe_delta_e00_median_macro": float(np.mean(medians)) if medians else None,
        "wide_safe_delta_e00_median_micro": float(np.median(micro_values)) if micro_values.size else None,
        "wide_safe_delta_e00_p95_macro": float(np.percentile(p95s, 95)) if p95s else None,
        "wide_safe_delta_e00_p95_micro": float(np.percentile(micro_values, 95)) if micro_values.size else None,
        "safe_pair_block_p95_max": max((float(row["safe_pair_block_p95"]) for row in evaluable), default=None),
        "visible_safe_seam_count": sum(
            float(row["safe_pair_block_p95"]) > visible_target for row in evaluable
            if row["safe_pair_block_p95"] is not None
        ),
        "quality_safe_evaluable_pair_count": len(evaluable),
        "quality_insufficient_evidence_pair_count": len(pair_reports) - len(evaluable),
        "aggregate_validation_p95": float(np.percentile(linear_p95s, 95)) if linear_p95s else None,
        **global_metrics,
    }


def _metric_decisions(after: Mapping[str, Any], before: Mapping[str, Any], threshold: Mapping[str, Any], b1_applied: bool) -> tuple[dict[str, Any], bool, bool, bool]:
    decisions: dict[str, Any] = {}
    target_pass, ceiling_pass, nonreg_pass = True, True, True
    pair_count = int(after.get("quality_safe_evaluable_pair_count", 0) + after.get("quality_insufficient_evidence_pair_count", 0))
    for name, spec in threshold["metrics"].items():
        value, baseline = after.get(name), before.get(name)
        applicable = not (name == "b1_gradient_ratio" and not b1_applied)
        evaluable = value is not None or not applicable
        direction = spec["direction"]
        role = spec["quality_role"]
        row: dict[str, Any] = {"value": value, "before": baseline, "applicable": applicable, "evaluable": evaluable, "direction": direction, "quality_role": role}
        if applicable and not evaluable:
            target_pass = ceiling_pass = False
        elif applicable and role == "C_target_ceiling":
            target = spec.get("target")
            ceiling = spec.get("engineering_ceiling")
            if ceiling is None and "engineering_ceiling_rule" in spec:
                ceiling = max(2, math.ceil(0.02 * pair_count))
            row["target"] = target
            row["engineering_ceiling"] = ceiling
            row["target_pass"] = value <= target if direction == "lower_is_better" else value >= target
            row["ceiling_pass"] = value <= ceiling if direction == "lower_is_better" else value >= ceiling
            target_pass &= bool(row["target_pass"])
            ceiling_pass &= bool(row["ceiling_pass"])
        elif applicable:
            relative = float(spec.get("nonreg_relative_tolerance", 0.02))
            absolute = float(spec.get("nonreg_absolute_tolerance", spec.get("nonreg_absolute_tolerance_or_mde", 0.0)))
            tolerance = max(abs(float(baseline or 0)) * relative, absolute)
            row["nonreg_tolerance"] = tolerance
            row["nonreg_pass"] = (value <= baseline + tolerance if direction == "lower_is_better" else value >= baseline - tolerance)
            nonreg_pass &= bool(row["nonreg_pass"])
        decisions[name] = row
    return decisions, target_pass, ceiling_pass, nonreg_pass


def evaluate_m61_post_render_quality(
    p2_image: np.ndarray,
    new_image: np.ndarray,
    old_image: np.ndarray,
    pair_arrays: Sequence[Mapping[str, np.ndarray]],
    owner_source_index: np.ndarray,
    valid_mask: np.ndarray,
    final_threshold: Mapping[str, Any],
    *,
    blend_active_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate the one formal render against frozen roles/targets/ceilings."""
    if final_threshold.get("status") != "final" or final_threshold.get("schema") != "gemini305-video-s13-m61-quality-thresholds/v2":
        raise ValueError("post-render quality requires the approved final threshold v2")
    if p2_image.shape != new_image.shape or new_image.shape != old_image.shape or valid_mask.shape != new_image.shape[:2]:
        raise ValueError("post-render image/mask shapes disagree")
    visible_target = float(final_threshold["metrics"]["wide_safe_delta_e00_p95"]["target"])
    new_pairs = [_pair_report(new_image, pair, index) for index, pair in enumerate(pair_arrays)]
    old_pairs = [_pair_report(old_image, pair, index) for index, pair in enumerate(pair_arrays)]
    baseline_pairs = [_pair_report(p2_image, pair, index) for index, pair in enumerate(pair_arrays)]
    new_global = _global_metrics(p2_image, new_image, np.asarray(owner_source_index), np.asarray(valid_mask, bool), blend_active_mask)
    old_global = _global_metrics(p2_image, old_image, np.asarray(owner_source_index), np.asarray(valid_mask, bool), None)
    baseline_global = _global_metrics(p2_image, p2_image, np.asarray(owner_source_index), np.asarray(valid_mask, bool), None)
    after = _aggregate(new_pairs, new_global, visible_target)
    before = _aggregate(baseline_pairs, baseline_global, visible_target)
    legacy = _aggregate(old_pairs, old_global, visible_target)
    b1_applied = blend_active_mask is not None and bool(np.any(blend_active_mask))
    decisions, targets, ceilings, nonreg = _metric_decisions(after, before, final_threshold, b1_applied)
    insufficient = int(after["quality_insufficient_evidence_pair_count"])
    maximum_fraction = float(final_threshold["actionability"]["required_pair_coverage"]["maximum_insufficient_safe_pair_fraction"])
    coverage_pass = insufficient <= math.floor(maximum_fraction * max(len(pair_arrays), 1))
    regressed = not nonreg or int(after["ghost_new_material_count"]) > int(before["ghost_new_material_count"])
    wide_before, wide_after = before["wide_safe_delta_e00_p95"], after["wide_safe_delta_e00_p95"]
    minimum_benefit = float(final_threshold["actionability"]["quality_minimum_benefit"]["wide_safe_p95_minimum_relative"])
    benefit_pass = wide_before is not None and wide_after is not None and wide_after <= wide_before * (1 - minimum_benefit)
    if not coverage_pass or any(row["applicable"] and not row["evaluable"] for row in decisions.values()):
        state = "unevaluable"
    elif regressed:
        state = "regressed"
    elif targets:
        state = "target"
    elif ceilings and benefit_pass:
        state = "near_target"
    else:
        state = "unresolved"
    atlas = sorted(({
        "pair_index": row["pair_index"], "score": row["wide_safe_delta_e00_p95"],
        "metric": "wide_safe_delta_e00_p95",
    } for row in new_pairs if row["evaluable"]), key=lambda row: (-float(row["score"]), int(row["pair_index"])))[:10]
    public_new_pairs = [{key: value for key, value in row.items() if not key.startswith("_")} for row in new_pairs]
    public_old_pairs = [{key: value for key, value in row.items() if not key.startswith("_")} for row in old_pairs]
    return {
        "schema": QUALITY_SCHEMA, "diagnostic_only": True,
        "candidate_selection_authority": False, "stage_seal_authority": False,
        "contributes_to_engineering_acceptance": True, "sole_engineering_acceptance_authority": False,
        "quality_state": state, "manual_review_required": state in {"near_target", "unresolved", "regressed", "unevaluable"},
        "pair_reports": public_new_pairs, "before_pair_reports": public_old_pairs,
        "aggregate": after, "before_aggregate": before, "legacy_m6_aggregate": legacy,
        "metric_decisions": decisions,
        "all_targets_pass": targets, "all_ceilings_pass": ceilings,
        "all_nonregression_pass": nonreg, "minimum_benefit_pass": benefit_pass,
        "coverage_pass": coverage_pass, "b1_applied": b1_applied,
        "top10_atlas": atlas,
    }


def write_m61_top10_atlas(output_dir: Path, image: np.ndarray, pair_arrays: Sequence[Mapping[str, np.ndarray]], report: Mapping[str, Any]) -> list[Path]:
    """Optionally materialize deterministic crops described by ``top10_atlas``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rank, item in enumerate(report.get("top10_atlas", [])):
        index = int(item["pair_index"])
        pair = pair_arrays[index]
        x, y = np.asarray(pair["canvas_x"]), np.asarray(pair["canvas_y"])
        x0, x1 = max(0, int(x.min())), min(image.shape[1], int(x.max()) + 1)
        y0, y1 = max(0, int(y.min())), min(image.shape[0], int(y.max()) + 1)
        path = output_dir / f"rank_{rank:02d}_pair_{index:04d}.png"
        if not cv2.imwrite(str(path), image[y0:y1, x0:x1]):
            raise OSError(f"failed to write atlas crop {path}")
        written.append(path)
    return written


__all__ = ["QUALITY_SCHEMA", "ciede2000", "evaluate_m61_post_render_quality", "write_m61_top10_atlas"]
