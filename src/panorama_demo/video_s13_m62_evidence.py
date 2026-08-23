"""Compact two-tier and skip-overlap evidence for the effective M6.2 path."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence

import cv2
import numpy as np

from .cuda_backend import srgb_to_linear_bgr
from .video_s13_m6_cuda import S13M6SourceROI
from .video_s13_photometric import S13PhotometricConfig, S13PhotometricSampleSet
from .video_s13_replay import S13P2ReplayPair


@dataclass(frozen=True)
class S13M62EvidenceBundle:
    adjacent_samples: tuple[S13PhotometricSampleSet, ...]
    solve_samples: tuple[S13PhotometricSampleSet, ...]
    masks: dict[str, np.ndarray]
    raw_by_frame: dict[int, np.ndarray]
    tier_a_sample_count: int
    tier_b_sample_count: int
    adjacent_edge_count: int
    bridge_edge_count: int
    component_count: int
    unsupported_cut_pair_indices: tuple[int, ...]
    performance: dict[str, object]


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


def _split_mask(safe: np.ndarray, x0: int, config: S13PhotometricConfig) -> np.ndarray:
    flat = np.flatnonzero(safe).astype(np.uint64, copy=False)
    rows = flat // np.uint64(safe.shape[1])
    columns = flat % np.uint64(safe.shape[1]) + np.uint64(x0)
    with np.errstate(over="ignore"):
        mixed = (
            rows * np.uint64(0x9E3779B185EBCA87)
            ^ columns * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ np.uint64(config.deterministic_split_seed)
        )
    selected = (mixed % np.uint64(10000)) < int(round(config.train_fraction * 10000.0))
    result = np.zeros(safe.size, dtype=bool)
    result[flat.astype(np.intp, copy=False)] = selected
    return result.reshape(safe.shape)


def _robust_evidence(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> tuple[float | None, float | None]:
    if not np.any(mask):
        return None, None
    left_luma = np.maximum(left.mean(axis=2)[mask], 1e-6)
    right_luma = np.maximum(right.mean(axis=2)[mask], 1e-6)
    relation = np.log(left_luma) - np.log(right_luma)
    center = float(np.median(relation))
    deviations = np.abs(relation - center)
    mad = float(np.median(deviations))
    scale = 1.4826 * mad
    inlier_limit = max(3.0 * scale, 0.01)
    return scale, float(np.mean(deviations <= inlier_limit))


def _sample_set(
    *, pair_index: int, left_source_index: int, right_source_index: int,
    x0: int, left: np.ndarray, right: np.ndarray, common: np.ndarray,
    config: S13PhotometricConfig, edge_kind: str,
    minimum_combined: int,
) -> S13PhotometricSampleSet:
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
        gradients = np.concatenate((grad_left[common], grad_right[common]))
        gradient_a = float(np.quantile(gradients, config.maximum_gradient_percentile))
        gradient_b = float(np.quantile(gradients, config.tier_b_maximum_gradient_percentile))
        gradient_hard = float(np.quantile(gradients, config.hard_protection_percentile))
        residual_a = float(np.quantile(residual[common], config.maximum_pair_color_residual_percentile))
        residual_b = float(np.quantile(residual[common], config.tier_b_maximum_residual_percentile))
        residual_hard = float(np.quantile(residual[common], config.hard_protection_percentile))
    else:
        gradient_a = gradient_b = gradient_hard = -1.0
        residual_a = residual_b = residual_hard = -1.0
    hard_seed = common & (
        (grad_left > gradient_hard) | (grad_right > gradient_hard) | (residual > residual_hard)
    )
    protected = cv2.dilate(
        hard_seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ).astype(bool)
    tier_a = common & intensity_safe & ~protected & (
        (grad_left <= gradient_a) & (grad_right <= gradient_a) & (residual <= residual_a)
    )
    tier_b_all = common & intensity_safe & ~protected & (
        (grad_left <= gradient_b) & (grad_right <= gradient_b) & (residual <= residual_b)
    )
    tier_b = tier_b_all & ~tier_a
    train_a = _split_mask(tier_a, x0, config)
    heldout_a = tier_a & ~train_a
    train_b = _split_mask(tier_b, x0, config)
    solve_train = train_a | train_b
    robust_scale, inlier_fraction = _robust_evidence(left_linear, right_linear, solve_train)
    tier_a_count = int(np.count_nonzero(train_a))
    tier_b_count = int(np.count_nonzero(train_b))
    robust_ok = bool(
        robust_scale is not None and inlier_fraction is not None
        and robust_scale <= config.maximum_log_luminance_mad
        and inlier_fraction >= config.minimum_evidence_inlier_fraction
    )
    edge_eligible = bool(
        tier_a_count >= config.minimum_pair_sample_count
        or (tier_a_count + tier_b_count >= minimum_combined and robust_ok)
    )
    if edge_eligible:
        ineligible_reason = None
    elif tier_a_count + tier_b_count < minimum_combined:
        ineligible_reason = "insufficient_pair_samples"
    elif not robust_ok:
        ineligible_reason = "unstable_tier_b_evidence"
    else:
        ineligible_reason = "unknown"
    flat_left = left_linear.reshape(-1, 3)
    flat_right = right_linear.reshape(-1, 3)
    train_indices = np.flatnonzero(solve_train)
    heldout_indices = np.flatnonzero(heldout_a)
    if train_indices.size > config.maximum_samples_per_pair:
        train_indices = train_indices[:config.maximum_samples_per_pair]
    maximum_heldout = max(1, config.maximum_samples_per_pair // 3)
    if heldout_indices.size > maximum_heldout:
        heldout_indices = heldout_indices[:maximum_heldout]
    train_rows = train_indices // tier_a.shape[1]
    train_columns = train_indices % tier_a.shape[1] + x0
    heldout_rows = heldout_indices // tier_a.shape[1]
    heldout_columns = heldout_indices % tier_a.shape[1] + x0
    return S13PhotometricSampleSet(
        pair_index=pair_index, left_source_index=left_source_index,
        right_source_index=right_source_index,
        train_left_rgb_linear=flat_left[train_indices],
        train_right_rgb_linear=flat_right[train_indices],
        heldout_left_rgb_linear=flat_left[heldout_indices],
        heldout_right_rgb_linear=flat_right[heldout_indices],
        train_canvas_xy=np.stack((train_columns, train_rows), axis=1).astype(np.int32),
        heldout_canvas_xy=np.stack((heldout_columns, heldout_rows), axis=1).astype(np.int32),
        safe_mask=tier_a, protected_mask=protected, common_mask=common,
        safe_mask_sha256="", protected_mask_sha256="",
        tier_a_train_sample_count=tier_a_count,
        tier_a_heldout_sample_count=int(np.count_nonzero(heldout_a)),
        tier_b_train_sample_count=tier_b_count,
        common_pixel_count=int(np.count_nonzero(common)),
        edge_eligible=edge_eligible, ineligible_reason=ineligible_reason,
        evidence_tier=("tier_a" if tier_a_count >= config.minimum_pair_sample_count else "tier_a_plus_b"),
        robust_scale_linear=robust_scale, inlier_fraction=inlier_fraction,
        gradient_limit=gradient_a, residual_limit=residual_a, edge_kind=edge_kind,
        left_linear_corridor=left_linear, right_linear_corridor=right_linear,
    )


def extract_s13_m62_evidence(
    replay_pairs: Sequence[S13P2ReplayPair],
    source_rois: Sequence[S13M6SourceROI],
    image_loader: Callable[[int], np.ndarray],
    *, canvas_shape: tuple[int, int], config: S13PhotometricConfig,
    retain_runtime_details: bool = False,
) -> S13M62EvidenceBundle:
    """Extract Tier A/Tier B adjacent evidence and deterministic i-to-i+2 bridges."""

    raw: dict[int, np.ndarray] = {}
    reference_adjacent_remap_count = 0
    reference_bridge_remap_count = 0
    reference_remap_pixel_count = 0
    reference_remap_seconds = 0.0
    def load(frame_id: int) -> np.ndarray:
        if frame_id not in raw:
            value = np.asarray(image_loader(frame_id))
            if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
                raise ValueError("S1.3 M6 raw RGB source shape/type changed")
            raw[frame_id] = value
        return raw[frame_id]

    adjacent: list[S13PhotometricSampleSet] = []
    height, width = canvas_shape
    masks = {name: np.zeros((height, width), bool) for name in ("safe", "protected", "train", "heldout")}
    for pair in replay_pairs:
        remap_started = time.perf_counter()
        left = _sample_raw(load(pair.left_frame_id), pair.left_source_u, pair.left_source_v)
        reference_remap_seconds += time.perf_counter() - remap_started
        reference_adjacent_remap_count += 1
        reference_remap_pixel_count += int(pair.left_source_u.size)
        remap_started = time.perf_counter()
        right = _sample_raw(load(pair.right_frame_id), pair.right_source_u, pair.right_source_v)
        reference_remap_seconds += time.perf_counter() - remap_started
        reference_adjacent_remap_count += 1
        reference_remap_pixel_count += int(pair.right_source_u.size)
        sample = _sample_set(
            pair_index=pair.pair_index, left_source_index=pair.left_source_index,
            right_source_index=pair.right_source_index, x0=pair.corridor_x0,
            left=left, right=right, common=pair.left_valid & pair.right_valid,
            config=config, edge_kind="adjacent",
            minimum_combined=config.minimum_tier_a_plus_b_sample_count,
        )
        adjacent.append(sample)
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        masks["safe"][roi] |= sample.safe_mask
        masks["protected"][roi] |= sample.protected_mask
        flat_train = sample.train_canvas_xy
        flat_heldout = sample.heldout_canvas_xy
        if flat_train.size:
            masks["train"][flat_train[:, 1], flat_train[:, 0]] = True
        if flat_heldout.size:
            masks["heldout"][flat_heldout[:, 1], flat_heldout[:, 0]] = True

    roi_by_source = {item.source_index: item for item in source_rois}
    bridges: list[S13PhotometricSampleSet] = []
    for left_index in range(max(0, len(source_rois) - 2)):
        right_index = left_index + 2
        left_roi, right_roi = roi_by_source[left_index], roi_by_source[right_index]
        x0, x1 = max(left_roi.x0, right_roi.x0), min(left_roi.x1, right_roi.x1)
        if x1 - x0 < config.bridge_minimum_overlap_width_px:
            continue
        left_slice = np.s_[:, x0 - left_roi.x0:x1 - left_roi.x0]
        right_slice = np.s_[:, x0 - right_roi.x0:x1 - right_roi.x0]
        common = left_roi.mapped[left_slice] & right_roi.mapped[right_slice]
        if not np.any(common):
            continue
        left_map_u, left_map_v = left_roi.map_u[left_slice], left_roi.map_v[left_slice]
        right_map_u, right_map_v = right_roi.map_u[right_slice], right_roi.map_v[right_slice]
        remap_started = time.perf_counter()
        left = _sample_raw(load(left_roi.frame_id), left_map_u, left_map_v)
        reference_remap_seconds += time.perf_counter() - remap_started
        reference_bridge_remap_count += 1
        reference_remap_pixel_count += int(left_map_u.size)
        remap_started = time.perf_counter()
        right = _sample_raw(load(right_roi.frame_id), right_map_u, right_map_v)
        reference_remap_seconds += time.perf_counter() - remap_started
        reference_bridge_remap_count += 1
        reference_remap_pixel_count += int(right_map_u.size)
        sample = _sample_set(
            pair_index=len(replay_pairs) + len(bridges), left_source_index=left_index,
            right_source_index=right_index, x0=x0, left=left, right=right, common=common,
            config=config, edge_kind="bridge_i_plus_2",
            minimum_combined=config.minimum_bridge_sample_count,
        )
        if sample.edge_eligible:
            bridges.append(sample)

    unsupported = tuple(
        int(sample.pair_index) for sample in adjacent if sample.edge_eligible is not True
    )
    if not retain_runtime_details:
        masks = {}
    solve_samples = tuple(adjacent) + tuple(bridges)
    adjacency = [set() for _ in source_rois]
    for item in solve_samples:
        if item.edge_eligible is True:
            adjacency[item.left_source_index].add(item.right_source_index)
            adjacency[item.right_source_index].add(item.left_source_index)
    remaining = set(range(len(source_rois)))
    component_count = 0
    while remaining:
        component_count += 1
        stack = [min(remaining)]
        while stack:
            node = stack.pop()
            if node not in remaining:
                continue
            remaining.remove(node)
            stack.extend(adjacency[node])
    return S13M62EvidenceBundle(
        adjacent_samples=tuple(adjacent), solve_samples=solve_samples, masks=masks,
        raw_by_frame=raw,
        tier_a_sample_count=sum(item.tier_a_train_sample_count for item in solve_samples),
        tier_b_sample_count=sum(item.tier_b_train_sample_count for item in solve_samples),
        adjacent_edge_count=sum(item.edge_eligible is True for item in adjacent),
        bridge_edge_count=len(bridges), component_count=component_count,
        unsupported_cut_pair_indices=unsupported,
        performance={
            "m62_evidence_reference_adjacent_remap_count": reference_adjacent_remap_count,
            "m62_evidence_reference_bridge_remap_count": reference_bridge_remap_count,
            "m62_evidence_reference_remap_pixel_count": reference_remap_pixel_count,
            "m62_evidence_reference_remap_seconds": reference_remap_seconds,
            "m62_evidence_packed_source_count": 0,
            "m62_evidence_packed_remap_count": 0,
            "m62_evidence_packed_remap_pixel_count": 0,
            "m62_evidence_packed_plan_seconds": 0.0,
            "m62_evidence_packed_remap_seconds": 0.0,
            "m62_evidence_unpack_seconds": 0.0,
            "m62_evidence_adjacent_usage_count": reference_adjacent_remap_count,
            "m62_evidence_bridge_usage_count": reference_bridge_remap_count,
            "m62_evidence_sample_mismatch_count": 0,
            "m62_evidence_fallback_count": 0,
            "m62_evidence_fallback_reasons": {},
            "m62_evidence_peak_temporary_bytes": 0,
        },
    )


def pair_evidence_document(sample: S13PhotometricSampleSet) -> dict[str, object]:
    return {
        "pair_index": sample.pair_index,
        "left_source_index": sample.left_source_index,
        "right_source_index": sample.right_source_index,
        "edge_kind": sample.edge_kind,
        "common_pixel_count": sample.common_pixel_count,
        "safe_pixel_count": int(np.count_nonzero(sample.safe_mask)),
        "tier_a_train_sample_count": sample.tier_a_train_sample_count,
        "tier_a_heldout_sample_count": sample.tier_a_heldout_sample_count,
        "tier_b_train_sample_count": sample.tier_b_train_sample_count,
        "edge_eligible": sample.edge_eligible,
        "ineligible_reason": sample.ineligible_reason,
        "evidence_tier": sample.evidence_tier,
        "robust_scale_linear": sample.robust_scale_linear,
        "inlier_fraction": sample.inlier_fraction,
        "gradient_limit": sample.gradient_limit,
        "residual_limit": sample.residual_limit,
    }


__all__ = ["S13M62EvidenceBundle", "extract_s13_m62_evidence", "pair_evidence_document"]
