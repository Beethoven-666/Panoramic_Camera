"""Immediate, corridor-only C2E selection for the fast S1.3 path.

This deliberately consumes M5's in-memory candidates.  It neither reloads a
stage image nor reruns M5; M6 receives the one selected replay set and renders
P3 exactly once.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Sequence

import cv2
import numpy as np

from .video_s13_m5 import S13M5Pair
from .video_s13_replay import S13P2ReplayPair


def _local_residual(pair: S13P2ReplayPair, image_loader: Callable[[int], np.ndarray]) -> float:
    """Score only a two-pixel seam corridor from cached source maps."""
    left = cv2.remap(image_loader(pair.left_frame_id), pair.left_source_u, pair.left_source_v,
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right = cv2.remap(image_loader(pair.right_frame_id), pair.right_source_u, pair.right_source_v,
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    columns = np.arange(pair.corridor_x0, pair.corridor_x1)[None, :]
    near = np.abs(columns - pair.seam_x_by_row[:, None]) <= 1
    valid = near & pair.left_valid & pair.right_valid
    if not np.any(valid):
        return float("inf")
    return float(np.median(np.abs(left.astype(np.int16) - right.astype(np.int16))[valid]))


def _with_seam(pair: S13P2ReplayPair, seam: np.ndarray) -> S13P2ReplayPair:
    seam = np.asarray(seam, dtype=np.int32)
    if seam.shape != pair.seam_x_by_row.shape:
        return pair
    if np.any(seam < pair.corridor_x0) or np.any(seam >= pair.corridor_x1):
        return pair
    columns = np.arange(pair.corridor_x0, pair.corridor_x1)[None, :]
    return replace(pair, seam_x_by_row=seam.copy(), primary_owner_right_mask=columns >= seam[:, None])


def select_s13_fast_c2e(
    pairs: Sequence[S13M5Pair], replay_pairs: Sequence[S13P2ReplayPair],
    image_loader: Callable[[int], np.ndarray],
) -> tuple[tuple[S13P2ReplayPair, ...], tuple[str, ...], frozenset[int]]:
    """Choose C0/C1/C2 locally; C3's audited geometry remains available in M5.

    C1 is selected for an explicit unresolved structural warning and disables
    blending only for that pair.  C2 compares M5's already generated seam
    paths.  Geometry candidates are intentionally not recomputed here: their
    selected M5 source maps are the only maps admitted to the formal render.
    """
    if len(pairs) != len(replay_pairs):
        raise ValueError("S1.3 C2E pair count disagrees with M5 replay")
    selected: list[S13P2ReplayPair] = []
    labels: list[str] = []
    owner_only: set[int] = set()
    for index, (m5, base) in enumerate(zip(pairs, replay_pairs, strict=True)):
        transaction = m5.transaction
        risk = bool(transaction.get("selection_continued_for_structure")) or bool(
            transaction.get("unresolved_oblique_structure")
        )
        # C1 is a local blend-policy candidate; it has no extra raw remap.
        if risk and bool(transaction.get("unresolved_oblique_structure")):
            selected.append(base)
            labels.append("C1_owner_only")
            owner_only.add(index)
            continue
        best, best_score, label = base, _local_residual(base, image_loader), "C0_keep_standard"
        if risk:
            for seam in m5.c2e_seam_candidates:
                candidate = _with_seam(base, seam)
                if candidate is base:
                    continue
                score = _local_residual(candidate, image_loader)
                # Strict improvement avoids an arbitrary candidate churn.
                if score + 1e-6 < best_score:
                    best, best_score, label = candidate, score, "C2_cached_seam"
        selected.append(best)
        labels.append(label)
    return tuple(selected), tuple(labels), frozenset(owner_only)


__all__ = ["select_s13_fast_c2e"]
