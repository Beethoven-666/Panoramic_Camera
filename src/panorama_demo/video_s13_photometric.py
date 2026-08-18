"""Linear-light, globally solved photometric candidates for S1.3 P3."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .cuda_backend import linear_to_srgb_bgr, srgb_to_linear_bgr
from .video_s13_replay import S13P2ReplayPair


PHOTOMETRIC_SCHEMA = "gemini305-video-s13-photometric-solution/v1"
PHOTOMETRIC_BOUNDS_SCHEMA = "gemini305-video-s13-photometric-bounds/v1"


@dataclass(frozen=True)
class S13PhotometricConfig:
    minimum_linear_intensity: float = 0.02
    maximum_linear_intensity: float = 0.98
    maximum_gradient_percentile: float = 0.80
    maximum_pair_color_residual_percentile: float = 0.80
    minimum_pair_sample_count: int = 128
    maximum_samples_per_pair: int = 8192
    train_fraction: float = 0.75
    deterministic_split_seed: int = 20260804
    minimum_gain: float = 0.80
    maximum_gain: float = 1.25
    maximum_absolute_bias_linear: float = 0.03
    minimum_heldout_improvement_fraction: float = 0.01
    simpler_model_tie_fraction: float = 0.005
    aggregate_p95_nonreg_absolute_tolerance_linear: float = 0.0005
    aggregate_p95_nonreg_relative_tolerance: float = 0.02
    minimum_actionable_macro_p95_benefit_fraction: float = 0.05
    minimum_aggregate_median_benefit_fraction: float = 0.01
    selection_score_mde_linear: float = 0.0005
    worst_pair_p95_nonreg_absolute_tolerance_linear: float = 0.0005
    worst_pair_p95_nonreg_relative_tolerance: float = 0.02
    require_macro_and_micro_nonregression: bool = True
    tier_b_maximum_gradient_percentile: float = 0.90
    tier_b_maximum_residual_percentile: float = 0.90
    hard_protection_percentile: float = 0.95
    minimum_tier_a_plus_b_sample_count: int = 192
    minimum_bridge_sample_count: int = 256
    minimum_evidence_inlier_fraction: float = 0.70
    maximum_log_luminance_mad: float = 0.03
    bridge_minimum_overlap_width_px: int = 32


@dataclass(frozen=True)
class S13PhotometricSampleSet:
    pair_index: int
    left_source_index: int
    right_source_index: int
    train_left_rgb_linear: np.ndarray
    train_right_rgb_linear: np.ndarray
    heldout_left_rgb_linear: np.ndarray
    heldout_right_rgb_linear: np.ndarray
    train_canvas_xy: np.ndarray
    heldout_canvas_xy: np.ndarray
    safe_mask: np.ndarray
    protected_mask: np.ndarray
    common_mask: np.ndarray
    safe_mask_sha256: str
    protected_mask_sha256: str
    tier_a_train_sample_count: int = 0
    tier_a_heldout_sample_count: int = 0
    tier_b_train_sample_count: int = 0
    common_pixel_count: int = 0
    edge_eligible: bool | None = None
    ineligible_reason: str | None = None
    evidence_tier: str = "tier_a"
    robust_scale_linear: float | None = None
    inlier_fraction: float | None = None
    gradient_limit: float | None = None
    residual_limit: float | None = None
    edge_kind: str = "adjacent"
    left_linear_corridor: np.ndarray | None = None
    right_linear_corridor: np.ndarray | None = None


@dataclass(frozen=True)
class S13SourcePhotometricParameters:
    source_index: int
    frame_id: int
    model: str
    gain_bgr: tuple[float, float, float]
    bias_bgr: tuple[float, float, float]
    component_id: int
    evidence_weight: float
    fallback_reason: str | None


@dataclass(frozen=True)
class S13PhotometricSolution:
    schema: str
    bounds_schema: str
    model_family: str
    source_parameters: tuple[S13SourcePhotometricParameters, ...]
    candidate_audits: tuple[Mapping[str, object], ...]
    train_audit: Mapping[str, object]
    heldout_audit: Mapping[str, object]
    split_seed: int
    train_fraction: float
    low_frequency_luminance_field_enabled: bool


def _sha_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _sample_raw(raw: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return cv2.remap(
        raw, np.asarray(u, np.float32), np.asarray(v, np.float32), cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.magnitude(
        cv2.Scharr(gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray, cv2.CV_32F, 0, 1),
    )


def _deterministic_train_mask(
    pair: S13P2ReplayPair, safe: np.ndarray, config: S13PhotometricConfig
) -> np.ndarray:
    rows, columns = np.indices(safe.shape, dtype=np.uint64)
    absolute_x = columns + np.uint64(pair.corridor_x0)
    # Classification is canvas-global rather than pair-local.  Adjacent replay
    # shoulders may overlap; the same canvas sample must never be train for one
    # pair and held-out for its neighbour.
    with np.errstate(over="ignore"):
        mixed = (
            rows * np.uint64(0x9E3779B185EBCA87)
            ^ absolute_x * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ np.uint64(config.deterministic_split_seed)
        )
    bucket = mixed % np.uint64(10000)
    return safe & (bucket < int(round(config.train_fraction * 10000.0)))


def extract_s13_photometric_samples(
    replay_pairs: Sequence[S13P2ReplayPair],
    image_loader: Callable[[int], np.ndarray],
    *,
    canvas_shape: tuple[int, int],
    config: S13PhotometricConfig = S13PhotometricConfig(),
) -> tuple[tuple[S13PhotometricSampleSet, ...], dict[str, np.ndarray], dict[int, np.ndarray]]:
    """Extract deterministic safe-background train/held-out samples."""

    height, width = canvas_shape
    protected_canvas = np.zeros((height, width), dtype=bool)
    safe_canvas = np.zeros((height, width), dtype=bool)
    train_canvas = np.zeros((height, width), dtype=bool)
    heldout_canvas = np.zeros((height, width), dtype=bool)
    raw_cache: dict[int, np.ndarray] = {}
    sample_sets: list[S13PhotometricSampleSet] = []
    for pair in replay_pairs:
        for frame_id in (pair.left_frame_id, pair.right_frame_id):
            if frame_id not in raw_cache:
                raw = np.asarray(image_loader(frame_id))
                if raw.ndim != 3 or raw.shape[2] != 3 or raw.dtype != np.uint8:
                    raise ValueError("S1.3 M6 raw RGB source shape/type changed")
                raw_cache[frame_id] = raw
        left = _sample_raw(raw_cache[pair.left_frame_id], pair.left_source_u, pair.left_source_v)
        right = _sample_raw(raw_cache[pair.right_frame_id], pair.right_source_u, pair.right_source_v)
        common = pair.left_valid & pair.right_valid
        left_linear = srgb_to_linear_bgr(left)
        right_linear = srgb_to_linear_bgr(right)
        intensities = np.concatenate((left_linear, right_linear), axis=2)
        intensity_safe = (
            np.all(intensities >= config.minimum_linear_intensity, axis=2)
            & np.all(intensities <= config.maximum_linear_intensity, axis=2)
        )
        grad_left, grad_right = _gradient(left), _gradient(right)
        residual = np.linalg.norm(left_linear - right_linear, axis=2)
        if np.any(common):
            gradient_limit = float(np.quantile(
                np.concatenate((grad_left[common], grad_right[common])),
                config.maximum_gradient_percentile,
            ))
            residual_limit = float(np.quantile(
                residual[common], config.maximum_pair_color_residual_percentile
            ))
        else:
            gradient_limit = residual_limit = -1.0
        structural_seed = common & (
            (grad_left > gradient_limit) | (grad_right > gradient_limit)
            | (residual > residual_limit)
        )
        # Scharr-Y protects long horizontal structures; the union also covers
        # cables, frames, fan/box outlines, and obvious double edges.
        protected = cv2.dilate(
            structural_seed.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ).astype(bool)
        safe = common & intensity_safe & ~protected
        train = _deterministic_train_mask(pair, safe, config)
        heldout = safe & ~train
        coordinates = np.indices(safe.shape, dtype=np.int32)
        xy = np.stack((coordinates[1] + pair.corridor_x0, coordinates[0]), axis=2)
        train_indices = np.flatnonzero(train)
        heldout_indices = np.flatnonzero(heldout)
        if train_indices.size > config.maximum_samples_per_pair:
            train_indices = train_indices[: config.maximum_samples_per_pair]
        maximum_heldout = max(1, config.maximum_samples_per_pair // 3)
        if heldout_indices.size > maximum_heldout:
            heldout_indices = heldout_indices[:maximum_heldout]
        flat_left, flat_right, flat_xy = (
            left_linear.reshape(-1, 3), right_linear.reshape(-1, 3), xy.reshape(-1, 2)
        )
        sample_sets.append(S13PhotometricSampleSet(
            pair_index=pair.pair_index,
            left_source_index=pair.left_source_index,
            right_source_index=pair.right_source_index,
            train_left_rgb_linear=flat_left[train_indices],
            train_right_rgb_linear=flat_right[train_indices],
            heldout_left_rgb_linear=flat_left[heldout_indices],
            heldout_right_rgb_linear=flat_right[heldout_indices],
            train_canvas_xy=flat_xy[train_indices],
            heldout_canvas_xy=flat_xy[heldout_indices],
            safe_mask=safe,
            protected_mask=protected,
            common_mask=common,
            safe_mask_sha256=_sha_array(safe),
            protected_mask_sha256=_sha_array(protected),
        ))
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        protected_canvas[roi] |= protected
        safe_canvas[roi] |= safe
        train_canvas[roi] |= train
        heldout_canvas[roi] |= heldout
    if np.any(train_canvas & heldout_canvas):
        raise RuntimeError("S1.3 photometric train and held-out masks overlap")
    return tuple(sample_sets), {
        "protected": protected_canvas,
        "safe": safe_canvas,
        "train": train_canvas,
        "heldout": heldout_canvas,
    }, raw_cache


def _components(source_count: int, samples: Sequence[S13PhotometricSampleSet], minimum: int) -> list[list[int]]:
    adjacency = [set() for _ in range(source_count)]
    for sample in samples:
        if (
            sample.edge_eligible is True
            or (sample.edge_eligible is None and len(sample.train_left_rgb_linear) >= minimum)
        ):
            adjacency[sample.left_source_index].add(sample.right_source_index)
            adjacency[sample.right_source_index].add(sample.left_source_index)
    components: list[list[int]] = []
    remaining = set(range(source_count))
    while remaining:
        start = min(remaining)
        stack, component = [start], []
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            component.append(node)
            stack.extend(sorted(adjacency[node], reverse=True))
        components.append(sorted(component))
    return components


def _solve_gain_candidate(
    model: str,
    source_count: int,
    samples: Sequence[S13PhotometricSampleSet],
    config: S13PhotometricConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float], list[str | None]]:
    channels = 1 if model == "Q1_scalar_luminance_gain" else 3
    gains = np.ones((source_count, 3), dtype=np.float64)
    biases = np.zeros((source_count, 3), dtype=np.float64)
    component_ids = [-1] * source_count
    evidence = [0.0] * source_count
    fallback: list[str | None] = [None] * source_count
    components = _components(source_count, samples, config.minimum_pair_sample_count)
    for component_id, nodes in enumerate(components):
        for node in nodes:
            component_ids[node] = component_id
        edges = [
            sample for sample in samples
            if sample.left_source_index in nodes and sample.right_source_index in nodes
            and (
                sample.edge_eligible is True
                or (
                    sample.edge_eligible is None
                    and len(sample.train_left_rgb_linear) >= config.minimum_pair_sample_count
                )
            )
        ]
        if len(nodes) == 1 or not edges:
            fallback[nodes[0]] = "isolated_source_identity"
            continue
        node_to_local = {node: index for index, node in enumerate(nodes)}
        rows: list[np.ndarray] = []
        values: list[np.ndarray] = []
        weights: list[float] = []
        for edge in edges:
            left = np.clip(edge.train_left_rgb_linear, 1e-6, 1.0)
            right = np.clip(edge.train_right_rgb_linear, 1e-6, 1.0)
            if channels == 1:
                relation = np.asarray([np.median(np.log(left.mean(1)) - np.log(right.mean(1)))])
            else:
                relation = np.median(np.log(left) - np.log(right), axis=0)
            row = np.zeros(len(nodes), dtype=np.float64)
            row[node_to_local[edge.left_source_index]] = -1.0
            row[node_to_local[edge.right_source_index]] = 1.0
            rows.append(row)
            values.append(relation)
            weight = float(np.sqrt(len(left)))
            weights.append(weight)
            evidence[edge.left_source_index] += len(left)
            evidence[edge.right_source_index] += len(left)
        anchor = max(nodes, key=lambda node: (evidence[node], -node))
        anchor_row = np.zeros(len(nodes), dtype=np.float64)
        anchor_row[node_to_local[anchor]] = 1.0
        rows.append(anchor_row)
        values.append(np.zeros(channels, dtype=np.float64))
        weights.append(max(weights, default=1.0) * 10.0)
        matrix = np.stack(rows) * np.asarray(weights)[:, None]
        target = np.stack(values) * np.asarray(weights)[:, None]
        logs = np.linalg.lstsq(matrix, target, rcond=None)[0]
        solved = np.exp(logs)
        if channels == 1:
            solved = np.repeat(solved, 3, axis=1)
        for node in nodes:
            value = solved[node_to_local[node]]
            if not np.isfinite(value).all() or np.any(value < config.minimum_gain) or np.any(value > config.maximum_gain):
                fallback[node] = "gain_out_of_bounds_identity"
                continue
            gains[node] = value
    return gains, biases, component_ids, evidence, fallback


def _solve_affine_candidate(
    source_count: int,
    samples: Sequence[S13PhotometricSampleSet],
    config: S13PhotometricConfig,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float], list[str | None]]:
    gains, _zero, component_ids, evidence, fallback = _solve_gain_candidate(
        "Q2_rgb_diagonal_gain", source_count, samples, config
    )
    biases = np.zeros_like(gains)
    components = _components(source_count, samples, config.minimum_pair_sample_count)
    for component_id, nodes in enumerate(components):
        if len(nodes) == 1:
            continue
        node_to_local = {node: index for index, node in enumerate(nodes)}
        edges = [s for s in samples if s.left_source_index in nodes and s.right_source_index in nodes
                 and (s.edge_eligible is True or (
                     s.edge_eligible is None
                     and len(s.train_left_rgb_linear) >= config.minimum_pair_sample_count
                 ))]
        if not edges:
            continue
        anchor = max(nodes, key=lambda node: (evidence[node], -node))
        for channel in range(3):
            rows: list[np.ndarray] = []
            targets: list[float] = []
            for edge in edges:
                stride = max(1, len(edge.train_left_rgb_linear) // 512)
                left = edge.train_left_rgb_linear[::stride, channel]
                right = edge.train_right_rgb_linear[::stride, channel]
                for lv, rv in zip(left, right, strict=True):
                    row = np.zeros(2 * len(nodes), dtype=np.float64)
                    li, ri = node_to_local[edge.left_source_index], node_to_local[edge.right_source_index]
                    row[2 * li] = float(lv)
                    row[2 * li + 1] = 1.0
                    row[2 * ri] = -float(rv)
                    row[2 * ri + 1] = -1.0
                    rows.append(row)
                    targets.append(0.0)
            for local in range(len(nodes)):
                gain_regularizer = np.zeros(2 * len(nodes), dtype=np.float64)
                gain_regularizer[2 * local] = 0.05
                rows.append(gain_regularizer)
                targets.append(0.05)
                bias_regularizer = np.zeros(2 * len(nodes), dtype=np.float64)
                bias_regularizer[2 * local + 1] = 0.2
                rows.append(bias_regularizer)
                targets.append(0.0)
            anchor_local = node_to_local[anchor]
            for offset, target in ((0, 1.0), (1, 0.0)):
                row = np.zeros(2 * len(nodes), dtype=np.float64)
                row[2 * anchor_local + offset] = 100.0
                rows.append(row)
                targets.append(100.0 * target)
            solved = np.linalg.lstsq(np.stack(rows), np.asarray(targets), rcond=None)[0]
            for node in nodes:
                local = node_to_local[node]
                gains[node, channel] = solved[2 * local]
                biases[node, channel] = solved[2 * local + 1]
        for node in nodes:
            if (
                not np.isfinite(gains[node]).all() or not np.isfinite(biases[node]).all()
                or np.any(gains[node] < config.minimum_gain)
                or np.any(gains[node] > config.maximum_gain)
                or np.any(np.abs(biases[node]) > config.maximum_absolute_bias_linear)
            ):
                gains[node] = 1.0
                biases[node] = 0.0
                fallback[node] = "affine_parameters_out_of_bounds_identity"
        for node in nodes:
            component_ids[node] = component_id
    return gains, biases, component_ids, evidence, fallback


def _candidate_metrics(
    samples: Sequence[S13PhotometricSampleSet], gains: np.ndarray, biases: np.ndarray, split: str
) -> dict[str, object]:
    residuals: list[np.ndarray] = []
    pair_rows: list[dict[str, object]] = []
    count = 0
    for sample in samples:
        left = getattr(sample, f"{split}_left_rgb_linear")
        right = getattr(sample, f"{split}_right_rgb_linear")
        if not len(left):
            continue
        corrected_left = left * gains[sample.left_source_index] + biases[sample.left_source_index]
        corrected_right = right * gains[sample.right_source_index] + biases[sample.right_source_index]
        pair_values = np.linalg.norm(corrected_left - corrected_right, axis=1)
        residuals.append(pair_values)
        pair_rows.append({
            "pair_index": int(sample.pair_index),
            "left_source_index": int(sample.left_source_index),
            "right_source_index": int(sample.right_source_index),
            "sample_count": int(pair_values.size),
            "median_linear": float(np.median(pair_values)),
            "p95_linear": float(np.quantile(pair_values, 0.95)),
            "edge_kind": sample.edge_kind,
        })
        count += len(left)
    if not residuals:
        return {"evaluable": False, "reason": "no_safe_samples", "sample_count": 0,
                "residual_median_linear": None, "residual_p95_linear": None,
                "aggregate_median_linear": None, "aggregate_p95_linear": None,
                "macro_pair_median_linear": None, "macro_pair_p95_linear": None,
                "worst_pair_p95_linear": None, "pair_metrics": ()}
    values = np.concatenate(residuals)
    pair_medians = np.asarray([row["median_linear"] for row in pair_rows], np.float64)
    pair_p95s = np.asarray([row["p95_linear"] for row in pair_rows], np.float64)
    aggregate_median = float(np.median(values))
    aggregate_p95 = float(np.quantile(values, 0.95))
    return {
        "evaluable": True,
        "reason": None,
        "sample_count": count,
        "residual_median_linear": aggregate_median,
        "residual_p95_linear": aggregate_p95,
        "aggregate_median_linear": aggregate_median,
        "aggregate_p95_linear": aggregate_p95,
        "macro_pair_median_linear": float(np.median(pair_medians)),
        "macro_pair_p95_linear": float(np.mean(pair_p95s)),
        "worst_pair_p95_linear": float(np.max(pair_p95s)),
        "pair_metrics": tuple(pair_rows),
    }


def _selection_rejection_reasons(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    config: S13PhotometricConfig,
) -> list[str]:
    if baseline.get("evaluable") is not True or candidate.get("evaluable") is not True:
        return ["validation_unevaluable"]
    reasons: list[str] = []
    before_median = float(baseline["aggregate_median_linear"])
    after_median = float(candidate["aggregate_median_linear"])
    required_median = max(
        before_median * config.minimum_aggregate_median_benefit_fraction,
        config.selection_score_mde_linear,
    )
    if before_median - after_median < required_median:
        reasons.append("aggregate_benefit_too_small")
    before_p95 = float(baseline["aggregate_p95_linear"])
    after_p95 = float(candidate["aggregate_p95_linear"])
    aggregate_allowance = max(
        before_p95 * config.aggregate_p95_nonreg_relative_tolerance,
        config.aggregate_p95_nonreg_absolute_tolerance_linear,
    )
    if after_p95 > before_p95 + aggregate_allowance:
        reasons.append("aggregate_p95_regression")
    before_macro = float(baseline["macro_pair_p95_linear"])
    after_macro = float(candidate["macro_pair_p95_linear"])
    required_macro = max(
        before_macro * config.minimum_actionable_macro_p95_benefit_fraction,
        config.selection_score_mde_linear,
    )
    if before_macro - after_macro < required_macro:
        reasons.append("macro_benefit_too_small")
    before_worst = float(baseline["worst_pair_p95_linear"])
    after_worst = float(candidate["worst_pair_p95_linear"])
    worst_allowance = max(
        before_worst * config.worst_pair_p95_nonreg_relative_tolerance,
        config.worst_pair_p95_nonreg_absolute_tolerance_linear,
    )
    if after_worst > before_worst + worst_allowance:
        reasons.append("worst_pair_regression")
    return reasons


def solve_s13_photometric(
    samples: Sequence[S13PhotometricSampleSet],
    *,
    frame_ids: Sequence[int],
    config: S13PhotometricConfig = S13PhotometricConfig(),
    force_identity: bool = False,
) -> S13PhotometricSolution:
    """Try Q0/Q1/Q2/Q3 in order and select by held-out evidence."""

    source_count = len(frame_ids)
    candidate_values: list[tuple[str, np.ndarray, np.ndarray, list[int], list[float], list[str | None]]] = []
    identity_components = list(range(source_count))
    candidate_values.append((
        "Q0_identity", np.ones((source_count, 3)), np.zeros((source_count, 3)),
        identity_components, [0.0] * source_count, ["identity_candidate"] * source_count,
    ))
    if not force_identity:
        for model in ("Q1_scalar_luminance_gain", "Q2_rgb_diagonal_gain"):
            candidate_values.append((model, *_solve_gain_candidate(model, source_count, samples, config)))
        candidate_values.append((
            "Q3_bounded_rgb_gain_bias",
            *_solve_affine_candidate(source_count, samples, config),
        ))
    audits: list[dict[str, object]] = []
    selected_index = 0
    selected_score: float | None = None
    baseline_metrics: Mapping[str, object] | None = None
    for index, (model, gains, biases, components, evidence, fallback) in enumerate(candidate_values):
        train = _candidate_metrics(samples, gains, biases, "train")
        heldout = _candidate_metrics(samples, gains, biases, "heldout")
        hard_safe = bool(
            np.isfinite(gains).all() and np.isfinite(biases).all()
            and np.all(gains >= config.minimum_gain) and np.all(gains <= config.maximum_gain)
            and np.all(np.abs(biases) <= config.maximum_absolute_bias_linear)
        )
        # A component has no photometric evidence tying it to its neighbour.
        # Correcting disconnected components independently creates a visible
        # full-height exposure block at that unsupported boundary.  Q0 remains
        # the only safe whole-canvas model unless every source participates in
        # one evidence-connected solve without an identity fallback.
        globally_connected = bool(
            index == 0
            or (
                source_count > 0
                and len(set(int(value) for value in components)) == 1
                and all(reason is None for reason in fallback)
            )
        )
        score = heldout.get("aggregate_median_linear") if heldout.get("evaluable") is True else None
        if index == 0 and isinstance(score, (float, int)):
            baseline_metrics = heldout
            selected_score = float(score)
        rejection_reasons: list[str] = []
        if index > 0:
            if not hard_safe:
                rejection_reasons.append("parameters_not_hard_safe")
            if not globally_connected:
                rejection_reasons.append("disconnected_evidence_graph")
            if any(reason is not None for reason in fallback):
                rejection_reasons.append("source_fallback")
            if baseline_metrics is None:
                rejection_reasons.append("identity_baseline_unevaluable")
            else:
                rejection_reasons.extend(
                    _selection_rejection_reasons(baseline_metrics, heldout, config)
                )
        if (
            index > 0 and not rejection_reasons
            and isinstance(score, (float, int))
            and selected_score is not None
            and float(score) < selected_score * (1.0 - config.simpler_model_tie_fraction)
        ):
            selected_index, selected_score = index, float(score)
        audits.append({
            "model": model, "hard_safe": hard_safe, "train": train, "heldout": heldout,
            "gain_minimum": float(np.min(gains)), "gain_maximum": float(np.max(gains)),
            "maximum_absolute_bias_linear": float(np.max(np.abs(biases))),
            "identity_fallback_source_count": sum(reason is not None for reason in fallback),
            "globally_connected": globally_connected,
            "component_count": len(set(int(value) for value in components)),
            "fallback_source_count": sum(reason is not None for reason in fallback),
            "train_sample_count": int(train.get("sample_count", 0)),
            "heldout_sample_count": int(heldout.get("sample_count", 0)),
            "aggregate_median_before": None if baseline_metrics is None else baseline_metrics.get("aggregate_median_linear"),
            "aggregate_median_after": heldout.get("aggregate_median_linear"),
            "aggregate_p95_before": None if baseline_metrics is None else baseline_metrics.get("aggregate_p95_linear"),
            "aggregate_p95_after": heldout.get("aggregate_p95_linear"),
            "macro_pair_median_before": None if baseline_metrics is None else baseline_metrics.get("macro_pair_median_linear"),
            "macro_pair_median_after": heldout.get("macro_pair_median_linear"),
            "macro_pair_p95_before": None if baseline_metrics is None else baseline_metrics.get("macro_pair_p95_linear"),
            "macro_pair_p95_after": heldout.get("macro_pair_p95_linear"),
            "worst_pair_p95_before": None if baseline_metrics is None else baseline_metrics.get("worst_pair_p95_linear"),
            "worst_pair_p95_after": heldout.get("worst_pair_p95_linear"),
            "rejection_reasons": rejection_reasons,
            "selected": False,
        })
    audits[selected_index]["selected"] = True
    model, gains, biases, components, evidence, fallback = candidate_values[selected_index]
    parameters = tuple(
        S13SourcePhotometricParameters(
            source_index=index,
            frame_id=int(frame_id),
            model=model if fallback[index] is None else "Q0_identity",
            gain_bgr=tuple(float(value) for value in gains[index]),
            bias_bgr=tuple(float(value) for value in biases[index]),
            component_id=int(components[index]),
            evidence_weight=float(evidence[index]),
            fallback_reason=fallback[index],
        )
        for index, frame_id in enumerate(frame_ids)
    )
    return S13PhotometricSolution(
        schema=PHOTOMETRIC_SCHEMA,
        bounds_schema=PHOTOMETRIC_BOUNDS_SCHEMA,
        model_family=model,
        source_parameters=parameters,
        candidate_audits=tuple(audits),
        train_audit=_candidate_metrics(samples, gains, biases, "train"),
        heldout_audit=_candidate_metrics(samples, gains, biases, "heldout"),
        split_seed=config.deterministic_split_seed,
        train_fraction=config.train_fraction,
        low_frequency_luminance_field_enabled=False,
    )


def apply_s13_photometric_linear(
    linear_bgr: np.ndarray, parameter: S13SourcePhotometricParameters
) -> np.ndarray:
    gain = np.asarray(parameter.gain_bgr, np.float32).reshape(1, 1, 3)
    bias = np.asarray(parameter.bias_bgr, np.float32).reshape(1, 1, 3)
    return np.clip(np.asarray(linear_bgr, np.float32) * gain + bias, 0.0, 1.0)


def photometric_solution_document(solution: S13PhotometricSolution) -> dict[str, object]:
    return {
        "schema": solution.schema,
        "bounds_schema": solution.bounds_schema,
        "model_family": solution.model_family,
        "source_parameters": [
            {
                "source_index": item.source_index, "frame_id": item.frame_id,
                "model": item.model, "gain_bgr": list(item.gain_bgr),
                "bias_bgr": list(item.bias_bgr), "component_id": item.component_id,
                "evidence_weight": item.evidence_weight,
                "fallback_reason": item.fallback_reason,
            }
            for item in solution.source_parameters
        ],
        "candidate_audits": [dict(value) for value in solution.candidate_audits],
        "train_audit": dict(solution.train_audit),
        "heldout_audit": dict(solution.heldout_audit),
        "split_seed": solution.split_seed,
        "train_fraction": solution.train_fraction,
        "low_frequency_luminance_field_enabled": False,
        "diagnostic_only": True,
        "runtime_authority": False,
    }


__all__ = [
    "PHOTOMETRIC_BOUNDS_SCHEMA", "PHOTOMETRIC_SCHEMA", "S13PhotometricConfig",
    "S13PhotometricSampleSet", "S13PhotometricSolution",
    "S13SourcePhotometricParameters", "apply_s13_photometric_linear",
    "extract_s13_photometric_samples", "linear_to_srgb_bgr",
    "photometric_solution_document", "solve_s13_photometric", "srgb_to_linear_bgr",
]
