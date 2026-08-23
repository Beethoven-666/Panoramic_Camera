"""Uniform global tone candidate kept separate from seam photometric residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class S13M63ToneResult:
    gain: float
    image: np.ndarray
    audit: dict[str, object]


def select_s13_m63_global_tone(
    linear_bgr: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gain_candidates: Sequence[float] = (1.02, 1.04, 1.06, 1.08, 1.10, 1.12),
    maximum_highlight_clip_increment_fraction: float = 0.001,
    maximum_channel_clip_increment_fraction: float = 0.001,
    maximum_valid_p995_linear: float = 0.98,
) -> S13M63ToneResult:
    image = np.asarray(linear_bgr, np.float32)
    valid = np.asarray(valid_mask, bool)
    if valid.shape != image.shape[:2] or not np.any(valid):
        raise ValueError("global tone requires a nonempty matching valid mask")
    values = image[valid]
    baseline_highlight = float(np.mean(np.max(values, axis=1) >= 1.0))
    baseline_channels = np.mean(values >= 1.0, axis=0)
    rows: list[dict[str, object]] = []
    selected = 1.0
    for gain in sorted((float(value) for value in gain_candidates), reverse=True):
        adjusted = values * gain
        p995 = float(np.quantile(adjusted.mean(axis=1), 0.995))
        highlight = float(np.mean(np.max(adjusted, axis=1) >= 1.0))
        channels = np.mean(adjusted >= 1.0, axis=0)
        accepted = bool(
            p995 <= maximum_valid_p995_linear
            and highlight - baseline_highlight <= maximum_highlight_clip_increment_fraction
            and np.all(channels - baseline_channels <= maximum_channel_clip_increment_fraction)
        )
        rows.append({
            "gain": gain,
            "accepted": accepted,
            "valid_p995_linear": p995,
            "highlight_clip_fraction": highlight,
            "channel_clip_fraction_bgr": [float(value) for value in channels],
        })
        if accepted:
            selected = gain
            break
    output = image.copy()
    output[valid] = np.clip(output[valid] * selected, 0.0, 1.0)
    output[~valid] = 0.0
    return S13M63ToneResult(
        gain=selected,
        image=output,
        audit={
            "selected_gain": selected,
            "luminance_before_after": {
                "before_median": float(np.median(values.mean(axis=1))),
                "after_median": float(np.median(output[valid].mean(axis=1))),
            },
            "clipping_before_after": {
                "before_highlight_fraction": baseline_highlight,
                "after_highlight_fraction": float(np.mean(np.max(output[valid], axis=1) >= 1.0)),
            },
            "candidates": rows,
        },
    )


__all__ = ["S13M63ToneResult", "select_s13_m63_global_tone"]
