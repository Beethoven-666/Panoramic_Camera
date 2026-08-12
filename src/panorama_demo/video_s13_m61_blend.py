"""Strict M6.1 owner-only / two-pixel feather planning and composition."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import cv2
import numpy as np


BLEND_SCHEMA = "gemini305-video-s13-blend-transactions/v2"
BLEND_MASK_SCHEMA = "gemini305-video-s13-blend-pair-masks/v1"
ELIGIBLE_MODELS = ("B0_owner_only", "B1_narrow_feather_2px")
INELIGIBLE_MODELS = ("B2_safe_masked_multiband", "B3", "B4")


@dataclass(frozen=True)
class M61BlendCandidate:
    pair_index: int
    seam_x_by_row: np.ndarray
    corridor_x0: int
    corridor_x1: int
    owner_right: np.ndarray
    safe: np.ndarray
    protected: np.ndarray
    common_valid: np.ndarray
    expected_valid: np.ndarray
    left_rgb: np.ndarray
    right_rgb: np.ndarray
    structural_risk: float = 0.0
    evidence_strength: float = 0.0
    minimum_immediate_benefit_linear: float = 0.0
    immediate_benefit_mde_linear: float = 0.0


@dataclass(frozen=True)
class M61BlendTransaction:
    schema: str
    pair_index: int
    model: str
    total_width_px: int
    active_pixel_count: int
    actual_changed_rgb_count: int
    immediate_before_linear: float | None
    immediate_after_linear: float | None
    immediate_benefit_linear: float | None
    selected_before_formal_render: bool
    fallback_reason: str | None
    ineligible_models: tuple[str, ...]


@dataclass(frozen=True)
class M61BlendPlan:
    transaction: M61BlendTransaction
    active: np.ndarray
    safe: np.ndarray
    protected: np.ndarray
    secondary_weight: np.ndarray


def _validate(candidate: M61BlendCandidate) -> tuple[int, int]:
    height = np.asarray(candidate.seam_x_by_row).size
    width = int(candidate.corridor_x1) - int(candidate.corridor_x0)
    shape = (height, width)
    if height <= 0 or width <= 0 or np.asarray(candidate.seam_x_by_row).shape != (height,):
        raise ValueError("M6.1 blend corridor is invalid")
    for name in ("owner_right", "safe", "protected", "common_valid", "expected_valid"):
        if np.asarray(getattr(candidate, name)).shape != shape:
            raise ValueError(f"M6.1 blend mask shape disagrees: {name}")
    for name in ("left_rgb", "right_rgb"):
        array = np.asarray(getattr(candidate, name))
        if array.shape != (*shape, 3) or array.dtype != np.uint8:
            raise ValueError(f"M6.1 blend RGB must be uint8: {name}")
    if np.any(np.asarray(candidate.safe, bool) & np.asarray(candidate.protected, bool)):
        raise ValueError("M6.1 safe/protected masks overlap")
    for value in (
        candidate.structural_risk, candidate.evidence_strength,
        candidate.minimum_immediate_benefit_linear, candidate.immediate_benefit_mde_linear,
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError("M6.1 blend evidence values must be finite and nonnegative")
    return shape


def b1_two_pixel_weights(candidate: M61BlendCandidate) -> np.ndarray:
    """Return the pixel-centre golden B1 weights: exactly 0.25 at x=s-1,s."""

    _validate(candidate)
    columns = np.arange(candidate.corridor_x0, candidate.corridor_x1)[None, :]
    seam = np.asarray(candidate.seam_x_by_row, np.int32)[:, None]
    active = (columns == seam - 1) | (columns == seam)
    eligible = (
        np.asarray(candidate.safe, bool)
        & ~np.asarray(candidate.protected, bool)
        & np.asarray(candidate.common_valid, bool)
        & np.asarray(candidate.expected_valid, bool)
    )
    result = np.zeros(active.shape, np.float32)
    result[active & eligible] = np.float32(0.25)
    return result


def _owner(candidate: M61BlendCandidate) -> np.ndarray:
    return np.where(candidate.owner_right[..., None], candidate.right_rgb, candidate.left_rgb)


def _compose(candidate: M61BlendCandidate, weight: np.ndarray) -> np.ndarray:
    owner = _owner(candidate)
    secondary = np.where(candidate.owner_right[..., None], candidate.left_rgb, candidate.right_rgb)
    result = owner.copy()
    active = weight > 0
    if np.any(active):
        mixed = np.rint(
            owner.astype(np.float32) * (1.0 - weight[..., None])
            + secondary.astype(np.float32) * weight[..., None]
        ).clip(0, 255).astype(np.uint8)
        result[active] = mixed[active]
    return result


def _immediate_residual(candidate: M61BlendCandidate, image: np.ndarray) -> float | None:
    columns = np.arange(candidate.corridor_x0, candidate.corridor_x1)[None, :]
    seam = candidate.seam_x_by_row[:, None]
    left = columns == seam - 1
    right = columns == seam
    usable = left & np.roll(right, -1, axis=1)
    rows, left_columns = np.nonzero(usable)
    if not rows.size:
        return None
    right_columns = left_columns + 1
    residual = np.mean(
        np.abs(image[rows, left_columns].astype(np.float32)
               - image[rows, right_columns].astype(np.float32)), axis=1
    ) / 255.0
    return float(np.median(residual))


def _b0(candidate: M61BlendCandidate, reason: str | None) -> M61BlendPlan:
    shape = _validate(candidate)
    return M61BlendPlan(
        M61BlendTransaction(
            BLEND_SCHEMA, candidate.pair_index, "B0_owner_only", 0, 0, 0,
            _immediate_residual(candidate, _owner(candidate)), None, None, True,
            reason, INELIGIBLE_MODELS,
        ),
        np.zeros(shape, bool), np.asarray(candidate.safe, bool).copy(),
        np.asarray(candidate.protected, bool).copy(), np.zeros(shape, np.float32),
    )


def plan_m61_blends(
    candidates: Sequence[M61BlendCandidate],
) -> tuple[M61BlendPlan, ...]:
    """Select all B0/B1 plans before render, resolving canvas conflicts deterministically."""

    provisional: dict[int, M61BlendPlan] = {}
    ranking: list[tuple[tuple[float, float, float, int], M61BlendCandidate, np.ndarray]] = []
    for candidate in candidates:
        _validate(candidate)
        weight = b1_two_pixel_weights(candidate)
        before = _immediate_residual(candidate, _owner(candidate))
        owner = _owner(candidate)
        composed = _compose(candidate, weight)
        after = _immediate_residual(candidate, composed)
        benefit = None if before is None or after is None else before - after
        required = max(candidate.minimum_immediate_benefit_linear,
                       candidate.immediate_benefit_mde_linear)
        if not np.any(weight):
            provisional[candidate.pair_index] = _b0(candidate, "no_safe_two_pixel_support")
        elif benefit is None or benefit < required:
            provisional[candidate.pair_index] = _b0(candidate, "immediate_benefit_below_mde")
        else:
            gray0 = cv2.cvtColor(owner, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray1 = cv2.cvtColor(composed, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g0 = np.hypot(
                cv2.Sobel(gray0, cv2.CV_32F, 1, 0), cv2.Sobel(gray0, cv2.CV_32F, 0, 1),
            )
            g1 = np.hypot(
                cv2.Sobel(gray1, cv2.CV_32F, 1, 0), cv2.Sobel(gray1, cv2.CV_32F, 0, 1),
            )
            active = weight > 0
            predicted_ghost = active & (g0 >= 2.0) & (g1 > g0 * 1.5 + 4.0)
            if np.any(predicted_ghost):
                provisional[candidate.pair_index] = _b0(candidate, "predicted_ghost_regression")
                continue
            ranking.append(((candidate.structural_risk, -candidate.evidence_strength,
                             -benefit, candidate.pair_index), candidate, weight))
    occupied: set[tuple[int, int]] = set()
    for _key, candidate, weight in sorted(ranking, key=lambda item: item[0]):
        rows, local_columns = np.nonzero(weight > 0)
        pixels = {(int(row), int(column + candidate.corridor_x0))
                  for row, column in zip(rows, local_columns, strict=True)}
        if occupied.intersection(pixels):
            provisional[candidate.pair_index] = _b0(candidate, "global_pair_conflict")
            continue
        occupied.update(pixels)
        before = _immediate_residual(candidate, _owner(candidate))
        after = _immediate_residual(candidate, _compose(candidate, weight))
        benefit = None if before is None or after is None else before - after
        provisional[candidate.pair_index] = M61BlendPlan(
            M61BlendTransaction(
                BLEND_SCHEMA, candidate.pair_index, "B1_narrow_feather_2px", 2,
                int(np.count_nonzero(weight)), 0, before, after, benefit, True, None,
                INELIGIBLE_MODELS,
            ), weight > 0, np.asarray(candidate.safe, bool).copy(),
            np.asarray(candidate.protected, bool).copy(), weight,
        )
    return tuple(provisional[candidate.pair_index] for candidate in candidates)


def compose_and_canonicalize_m61_blend(
    candidate: M61BlendCandidate, plan: M61BlendPlan,
) -> tuple[np.ndarray, M61BlendPlan, Mapping[str, np.ndarray]]:
    """Compose once; a uint8 no-op deterministically becomes a fully rebuilt B0 plan."""

    _validate(candidate)
    if plan.transaction.pair_index != candidate.pair_index:
        raise ValueError("M6.1 blend plan pair disagrees")
    owner = _owner(candidate)
    output = _compose(candidate, plan.secondary_weight)
    changed = np.any(output != owner, axis=2) & plan.active
    changed_count = int(np.count_nonzero(changed))
    if plan.transaction.model == "B1_narrow_feather_2px" and changed_count == 0:
        plan = _b0(candidate, "actual_uint8_noop_canonicalized")
        output = owner
    else:
        plan = replace(
            plan,
            transaction=replace(plan.transaction, actual_changed_rgb_count=changed_count),
        )
    provenance = {
        "blend_transaction_id": np.where(plan.active, candidate.pair_index, -1).astype(np.int32),
        "secondary_weight": np.asarray(plan.secondary_weight, np.float32).copy(),
        "active": np.asarray(plan.active, bool).copy(),
        "safe": np.asarray(plan.safe, bool).copy(),
        "protected": np.asarray(plan.protected, bool).copy(),
    }
    return output, plan, provenance


__all__ = [
    "BLEND_MASK_SCHEMA", "BLEND_SCHEMA", "ELIGIBLE_MODELS", "INELIGIBLE_MODELS",
    "M61BlendCandidate", "M61BlendPlan", "M61BlendTransaction", "b1_two_pixel_weights",
    "compose_and_canonicalize_m61_blend", "plan_m61_blends",
]
