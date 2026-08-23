"""Diagnostic-only visual measurements for S1.3 P3."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .video_s13_blend import S13BlendPlan
from .video_s13_photometric import S13PhotometricSolution
from .video_s13_replay import S13P2ReplayPair


def _seam_jump(image: np.ndarray, pair: S13P2ReplayPair) -> float | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = np.arange(gray.shape[0])
    left = np.clip(pair.seam_x_by_row - 1, 0, gray.shape[1] - 1)
    right = np.clip(pair.seam_x_by_row, 0, gray.shape[1] - 1)
    values = np.abs(gray[rows, left] - gray[rows, right])
    return float(np.median(values)) if values.size else None


def build_s13_p3_diagnostic_quality(
    p2_image: np.ndarray,
    owner_only: np.ndarray,
    final_image: np.ndarray,
    replay_pairs: Sequence[S13P2ReplayPair],
    blend_plans: Sequence[S13BlendPlan],
    photometric: S13PhotometricSolution,
) -> dict[str, object]:
    pair_rows = []
    for pair, plan in zip(replay_pairs, blend_plans, strict=True):
        pair_rows.append({
            "pair_index": pair.pair_index,
            "blend_model": plan.transaction.model,
            "p2_luminance_jump": _seam_jump(p2_image, pair),
            "photometric_owner_luminance_jump": _seam_jump(owner_only, pair),
            "p3_luminance_jump": _seam_jump(final_image, pair),
            "blended_pixel_count": plan.transaction.blended_pixel_count,
            "fallback_reason": plan.transaction.fallback_reason,
        })
    before_laplacian = cv2.Laplacian(p2_image, cv2.CV_32F).var()
    after_laplacian = cv2.Laplacian(final_image, cv2.CV_32F).var()
    before = [row["p2_luminance_jump"] for row in pair_rows if row["p2_luminance_jump"] is not None]
    after = [row["p3_luminance_jump"] for row in pair_rows if row["p3_luminance_jump"] is not None]
    quality_pass = bool(
        before and after
        and float(np.median(after)) <= float(np.median(before)) * 1.02
        and float(np.quantile(after, 0.95)) <= float(np.quantile(before, 0.95)) * 1.02 + 0.25
        and float(np.max(after)) <= float(np.max(before)) + 1.0
        and after_laplacian >= before_laplacian * 0.95
    )
    return {
        "schema": "gemini305-video-s13-p3-diagnostic-quality/v1",
        "diagnostic_only": True,
        "runtime_authority": False,
        "quality_pass": quality_pass,
        "photometric_model": photometric.model_family,
        "pair_reports": pair_rows,
        "median_luminance_jump_before": float(np.median(before)) if before else None,
        "median_luminance_jump_after": float(np.median(after)) if after else None,
        "p95_luminance_jump_before": float(np.quantile(before, 0.95)) if before else None,
        "p95_luminance_jump_after": float(np.quantile(after, 0.95)) if after else None,
        "maximum_luminance_jump_before": float(np.max(before)) if before else None,
        "maximum_luminance_jump_after": float(np.max(after)) if after else None,
        "laplacian_variance_before": float(before_laplacian),
        "laplacian_variance_after": float(after_laplacian),
        "manual_review_required": True,
    }


__all__ = ["build_s13_p3_diagnostic_quality"]
