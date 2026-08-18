"""Two-source, protected narrow blending for the S1.3 P3 visual stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np

from .video_s13_photometric import S13PhotometricSampleSet
from .video_s13_replay import S13P2ReplayPair


BLEND_SCHEMA = "gemini305-video-s13-blend-transactions/v1"


@dataclass(frozen=True)
class S13BlendConfig:
    minimum_total_width_px: int = 1
    maximum_total_width_px: int = 8
    maximum_levels: int = 2
    maximum_safe_residual_for_feather: float = 0.10
    maximum_safe_residual_for_multiband: float = 0.06
    minimum_safe_fraction_for_feather: float = 0.20
    minimum_safe_fraction_for_multiband: float = 0.50
    minimum_immediate_seam_benefit_linear: float = 0.5 / 255.0


@dataclass(frozen=True)
class S13BlendTransaction:
    pair_index: int
    transaction_id: int
    parent_p2_pair_transaction_sha256: str
    model: str
    total_width_px: int
    pyramid_levels: int
    protected_pixel_count: int
    blended_pixel_count: int
    hard_audit_passed: bool
    fallback_reason: str | None
    candidate_audits: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class S13BlendPlan:
    transaction: S13BlendTransaction
    secondary_weight: np.ndarray
    safe_mask: np.ndarray
    protected_mask: np.ndarray


def adaptive_s13_multiband_levels(total_width_px: int, maximum_levels: int = 2) -> int:
    if total_width_px <= 2:
        return 0
    if total_width_px <= 4:
        return min(1, int(maximum_levels))
    return min(2, int(maximum_levels))


def _candidate_weight(
    pair: S13P2ReplayPair,
    safe: np.ndarray,
    protected: np.ndarray,
    *,
    total_width_px: int,
) -> np.ndarray:
    columns = np.arange(pair.corridor_x0, pair.corridor_x1, dtype=np.float32)[None, :]
    signed = columns - pair.seam_x_by_row[:, None].astype(np.float32) + 0.5
    half = max(0.5, total_width_px / 2.0)
    alpha_right = np.clip(0.5 + signed / (2.0 * half), 0.0, 1.0)
    primary_right = pair.primary_owner_right_mask
    secondary = np.where(primary_right, 1.0 - alpha_right, alpha_right).astype(np.float32)
    zone = np.abs(signed) <= half
    secondary[~(zone & safe & ~protected & pair.left_valid & pair.right_valid)] = 0.0
    # The hard owner remains dominant, including the exact seam pixel.
    secondary = np.minimum(secondary, np.float32(0.499))
    return secondary


def _residual(sample: S13PhotometricSampleSet, corrected_left: np.ndarray, corrected_right: np.ndarray) -> float | None:
    mask = sample.safe_mask & sample.common_mask
    if not np.any(mask):
        return None
    values = np.linalg.norm(corrected_left - corrected_right, axis=2)[mask]
    return float(np.median(values)) if values.size else None


def _seam_jump(
    pair: S13P2ReplayPair, left: np.ndarray, right: np.ndarray,
    weight: np.ndarray,
) -> float | None:
    direct = np.where(pair.primary_owner_right_mask[..., None], right, left)
    alpha_right = np.where(
        pair.primary_owner_right_mask, 1.0 - weight, weight
    ).astype(np.float32)
    candidate = left * (1.0 - alpha_right[..., None]) + right * alpha_right[..., None]
    output = direct.copy()
    active = weight > 0.0
    output[active] = candidate[active]
    values: list[float] = []
    for row, seam_x in enumerate(np.asarray(pair.seam_x_by_row, np.int32)):
        column = int(seam_x) - int(pair.corridor_x0)
        if column <= 0 or column >= output.shape[1]:
            continue
        values.append(float(np.linalg.norm(output[row, column] - output[row, column - 1])))
    return float(np.median(values)) if values else None


def select_s13_blend_plans(
    replay_pairs: Sequence[S13P2ReplayPair],
    samples: Sequence[S13PhotometricSampleSet],
    corrected_by_source: Mapping[int, np.ndarray],
    *,
    canvas_shape: tuple[int, int],
    corrected_pair_provider: Callable[[S13P2ReplayPair], tuple[np.ndarray, np.ndarray]] | None = None,
    config: S13BlendConfig = S13BlendConfig(),
    force_owner_only: bool = False,
    force_owner_only_pair_indices: frozenset[int] = frozenset(),
    unsupported_cut_pair_indices: frozenset[int] = frozenset(),
) -> tuple[tuple[S13BlendPlan, ...], dict[str, np.ndarray]]:
    """Try B0, B1, then B2 per pair with global corridor non-overlap."""

    if len(replay_pairs) != len(samples):
        raise ValueError("S1.3 blend replay/sample pair counts disagree")
    used = np.zeros(canvas_shape, dtype=bool)
    safe_canvas = np.zeros(canvas_shape, dtype=bool)
    protected_canvas = np.zeros(canvas_shape, dtype=bool)
    weight_canvas = np.zeros(canvas_shape, dtype=np.float32)
    plans: list[S13BlendPlan] = []
    for pair, sample in zip(replay_pairs, samples, strict=True):
        if corrected_pair_provider is None:
            left = corrected_by_source[pair.left_source_index][:, pair.corridor_x0:pair.corridor_x1]
            right = corrected_by_source[pair.right_source_index][:, pair.corridor_x0:pair.corridor_x1]
        else:
            left, right = corrected_pair_provider(pair)
            expected_shape = sample.safe_mask.shape + (3,)
            if left.shape != expected_shape or right.shape != expected_shape:
                raise ValueError("S1.3 compact corrected pair provider returned the wrong shape")
        residual = _residual(sample, left, right)
        common_count = int(np.count_nonzero(sample.common_mask))
        safe_count = int(np.count_nonzero(sample.safe_mask))
        safe_fraction = safe_count / max(1, common_count)
        candidate_audits: list[dict[str, object]] = [{
            "model": "B0_owner_only", "hard_safe": True, "selected": False,
            "evaluable": True, "reason": None, "blended_pixel_count": 0,
        }]
        model, width, levels, fallback = "B0_owner_only", 0, 0, None
        weight = np.zeros(sample.safe_mask.shape, dtype=np.float32)
        if not force_owner_only and pair.pair_index not in force_owner_only_pair_indices:
            feather_width = 2 if safe_fraction >= 0.35 else 1
            feather = _candidate_weight(
                pair, sample.safe_mask, sample.protected_mask,
                total_width_px=feather_width,
            )
            owner_jump = _seam_jump(pair, left, right, np.zeros_like(feather))
            feather_jump = _seam_jump(pair, left, right, feather)
            feather_benefit = (
                None if owner_jump is None or feather_jump is None
                else owner_jump - feather_jump
            )
            feather_ok = bool(
                residual is not None
                and residual <= config.maximum_safe_residual_for_feather
                and safe_fraction >= config.minimum_safe_fraction_for_feather
                and np.any(feather > 0.0)
                and feather_benefit is not None
                and feather_benefit >= config.minimum_immediate_seam_benefit_linear
            )
            candidate_audits.append({
                "model": "B1_narrow_feather", "hard_safe": feather_ok,
                "selected": False, "evaluable": residual is not None,
                "reason": None if feather_ok else "insufficient_safe_background_or_residual",
                "safe_fraction": safe_fraction, "residual_median_linear": residual,
                "total_width_px": feather_width,
                "blended_pixel_count": int(np.count_nonzero(feather)),
                "immediate_seam_before_linear": owner_jump,
                "immediate_seam_after_linear": feather_jump,
                "immediate_seam_benefit_linear": feather_benefit,
            })
            multiband_width = min(config.maximum_total_width_px, 6 if safe_fraction >= 0.70 else 4)
            multiband = _candidate_weight(
                pair, sample.safe_mask, sample.protected_mask,
                total_width_px=multiband_width,
            )
            multiband_levels = adaptive_s13_multiband_levels(multiband_width, config.maximum_levels)
            multiband_jump = _seam_jump(pair, left, right, multiband)
            multiband_benefit = (
                None if owner_jump is None or multiband_jump is None
                else owner_jump - multiband_jump
            )
            multiband_ok = bool(
                residual is not None
                and residual <= config.maximum_safe_residual_for_multiband
                and safe_fraction >= config.minimum_safe_fraction_for_multiband
                and multiband_levels > 0
                and np.any(multiband > 0.0)
                and multiband_benefit is not None
                and multiband_benefit >= config.minimum_immediate_seam_benefit_linear
            )
            candidate_audits.append({
                "model": "B2_safe_masked_multiband", "hard_safe": multiband_ok,
                "selected": False, "evaluable": residual is not None,
                "reason": None if multiband_ok else "insufficient_safe_background_or_residual",
                "safe_fraction": safe_fraction, "residual_median_linear": residual,
                "total_width_px": multiband_width, "pyramid_levels": multiband_levels,
                "blended_pixel_count": int(np.count_nonzero(multiband)),
                "immediate_seam_before_linear": owner_jump,
                "immediate_seam_after_linear": multiband_jump,
                "immediate_seam_benefit_linear": multiband_benefit,
            })
            if multiband_ok:
                model, width, levels, weight = (
                    "B2_safe_masked_multiband", multiband_width, multiband_levels, multiband
                )
            elif feather_ok:
                model, width, levels, weight = "B1_narrow_feather", feather_width, 0, feather
            else:
                fallback = "no_safe_blend_candidate"
        else:
            fallback = (
                "unsupported_cut_forced_owner_only"
                if pair.pair_index in unsupported_cut_pair_indices
                else "forced_owner_only_rebuild"
            )
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        if np.any(used[roi] & (weight > 0.0)):
            model, width, levels = "B0_owner_only", 0, 0
            weight.fill(0.0)
            fallback = "adjacent_blend_corridor_overlap"
        if np.any(weight[sample.protected_mask] != 0.0):
            raise RuntimeError("S1.3 protected structure reached a blend candidate")
        candidate_audits[0]["selected"] = model == "B0_owner_only"
        for candidate in candidate_audits[1:]:
            candidate["selected"] = candidate["model"] == model
        used[roi] |= weight > 0.0
        safe_canvas[roi] |= sample.safe_mask
        protected_canvas[roi] |= sample.protected_mask
        weight_canvas[roi] = np.maximum(weight_canvas[roi], weight)
        transaction = S13BlendTransaction(
            pair_index=pair.pair_index,
            transaction_id=pair.pair_index,
            parent_p2_pair_transaction_sha256=pair.parent_pair_transaction_sha256,
            model=model,
            total_width_px=width,
            pyramid_levels=levels,
            protected_pixel_count=int(np.count_nonzero(sample.protected_mask)),
            blended_pixel_count=int(np.count_nonzero(weight)),
            hard_audit_passed=True,
            fallback_reason=fallback,
            candidate_audits=tuple(candidate_audits),
        )
        plans.append(S13BlendPlan(transaction, weight, sample.safe_mask, sample.protected_mask))
    # Pair shoulders may overlap.  Protection is canvas-global: a pixel marked
    # risky by either adjacent pair must remain owner-only in every pair.
    globally_safe_plans: list[S13BlendPlan] = []
    used.fill(False)
    weight_canvas.fill(0.0)
    for pair, plan in zip(replay_pairs, plans, strict=True):
        roi = np.s_[:, pair.corridor_x0:pair.corridor_x1]
        weight = plan.secondary_weight.copy()
        weight[protected_canvas[roi]] = 0.0
        active_count = int(np.count_nonzero(weight))
        transaction = plan.transaction
        if transaction.blended_pixel_count and not active_count:
            transaction = S13BlendTransaction(
                pair_index=transaction.pair_index,
                transaction_id=transaction.transaction_id,
                parent_p2_pair_transaction_sha256=transaction.parent_p2_pair_transaction_sha256,
                model="B0_owner_only",
                total_width_px=0,
                pyramid_levels=0,
                protected_pixel_count=transaction.protected_pixel_count,
                blended_pixel_count=0,
                hard_audit_passed=True,
                fallback_reason="global_protected_structure_overlap",
                candidate_audits=transaction.candidate_audits,
            )
        elif active_count != transaction.blended_pixel_count:
            transaction = S13BlendTransaction(
                pair_index=transaction.pair_index,
                transaction_id=transaction.transaction_id,
                parent_p2_pair_transaction_sha256=transaction.parent_p2_pair_transaction_sha256,
                model=transaction.model,
                total_width_px=transaction.total_width_px,
                pyramid_levels=transaction.pyramid_levels,
                protected_pixel_count=transaction.protected_pixel_count,
                blended_pixel_count=active_count,
                hard_audit_passed=True,
                fallback_reason="globally_protected_pixels_removed",
                candidate_audits=transaction.candidate_audits,
            )
        used[roi] |= weight > 0.0
        weight_canvas[roi] = np.maximum(weight_canvas[roi], weight)
        globally_safe_plans.append(S13BlendPlan(
            transaction, weight, plan.safe_mask, plan.protected_mask
        ))
    return tuple(globally_safe_plans), {
        "safe": safe_canvas, "protected": protected_canvas,
        "secondary_weight": weight_canvas, "active": used,
    }


def _multiband_pair(
    left: np.ndarray, right: np.ndarray, alpha_right: np.ndarray, levels: int
) -> np.ndarray:
    """Small deterministic Laplacian blend; levels is strictly at most two."""

    if levels <= 0:
        return left * (1.0 - alpha_right[..., None]) + right * alpha_right[..., None]
    gaussian_left = [np.asarray(left, np.float32)]
    gaussian_right = [np.asarray(right, np.float32)]
    gaussian_alpha = [np.asarray(alpha_right, np.float32)]
    for _ in range(levels):
        gaussian_left.append(cv2.pyrDown(gaussian_left[-1]))
        gaussian_right.append(cv2.pyrDown(gaussian_right[-1]))
        gaussian_alpha.append(cv2.pyrDown(gaussian_alpha[-1]))
    lap_left, lap_right = [], []
    for level in range(levels):
        shape = (gaussian_left[level].shape[1], gaussian_left[level].shape[0])
        lap_left.append(gaussian_left[level] - cv2.pyrUp(gaussian_left[level + 1], dstsize=shape))
        lap_right.append(gaussian_right[level] - cv2.pyrUp(gaussian_right[level + 1], dstsize=shape))
    lap_left.append(gaussian_left[-1])
    lap_right.append(gaussian_right[-1])
    blended = [
        left_level * (1.0 - alpha_level[..., None]) + right_level * alpha_level[..., None]
        for left_level, right_level, alpha_level in zip(
            lap_left, lap_right, gaussian_alpha, strict=True
        )
    ]
    result = blended[-1]
    for level in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended[level].shape[1], blended[level].shape[0]))
        result += blended[level]
    return np.clip(result, 0.0, 1.0)


def apply_s13_blend_plan(
    left: np.ndarray,
    right: np.ndarray,
    pair: S13P2ReplayPair,
    plan: S13BlendPlan,
) -> np.ndarray:
    direct = np.where(pair.primary_owner_right_mask[..., None], right, left)
    weight = np.asarray(plan.secondary_weight, np.float32)
    if not np.any(weight > 0.0):
        return direct.copy()
    alpha_right = np.where(pair.primary_owner_right_mask, 1.0 - weight, weight).astype(np.float32)
    if plan.transaction.model == "B2_safe_masked_multiband":
        candidate = _multiband_pair(left, right, alpha_right, plan.transaction.pyramid_levels)
    else:
        candidate = left * (1.0 - alpha_right[..., None]) + right * alpha_right[..., None]
    result = direct.copy()
    active = weight > 0.0
    result[active] = candidate[active]
    return result


def blend_transaction_document(transaction: S13BlendTransaction) -> dict[str, object]:
    return {
        "schema": "gemini305-video-s13-blend-transaction/v1",
        "pair_index": transaction.pair_index,
        "transaction_id": transaction.transaction_id,
        "parent_p2_pair_transaction_sha256": transaction.parent_p2_pair_transaction_sha256,
        "model": transaction.model,
        "total_width_px": transaction.total_width_px,
        "pyramid_levels": transaction.pyramid_levels,
        "protected_pixel_count": transaction.protected_pixel_count,
        "blended_pixel_count": transaction.blended_pixel_count,
        "hard_audit_passed": transaction.hard_audit_passed,
        "fallback_reason": transaction.fallback_reason,
        "candidate_audits": [dict(value) for value in transaction.candidate_audits],
    }


__all__ = [
    "BLEND_SCHEMA", "S13BlendConfig", "S13BlendPlan", "S13BlendTransaction",
    "adaptive_s13_multiband_levels", "apply_s13_blend_plan",
    "blend_transaction_document", "select_s13_blend_plans",
]
