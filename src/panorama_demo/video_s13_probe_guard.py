"""Conservative interval guards for CUDA probe metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class S13MetricDrift:
    name: str
    maximum_absolute_error: float
    p999_absolute_error: float
    sample_count: int


@dataclass(frozen=True)
class S13ProbeGuardCalibration:
    epsilon_by_metric: Mapping[str, float]
    multiplier: float


def calibrate_probe_guard(full: Mapping[str, float], probe: Mapping[str, float], *, multiplier: float = 2.0, minimum_absolute_epsilon: float = 1e-6) -> S13ProbeGuardCalibration:
    if not np.isfinite(multiplier) or multiplier < 1.0 or minimum_absolute_epsilon <= 0:
        raise ValueError("S013 probe calibration bounds are invalid")
    epsilons: dict[str, float] = {}
    for name in full.keys() & probe.keys():
        left, right = float(full[name]), float(probe[name])
        if np.isfinite(left) and np.isfinite(right):
            epsilons[name] = max(float(minimum_absolute_epsilon), abs(left - right) * float(multiplier))
    return S13ProbeGuardCalibration(epsilons, float(multiplier))


def guarded_ratio_decision(*, before: float | None, after: float | None, epsilon: float | None, ratio: float) -> str:
    """Classify an ``after <= before * ratio`` test without optimistic ties."""
    if any(value is None or not np.isfinite(float(value)) for value in (before, after, epsilon, ratio)) or float(epsilon) < 0:
        return "reference_fallback"
    lower_before, upper_before = float(before) - float(epsilon), float(before) + float(epsilon)
    lower_after, upper_after = float(after) - float(epsilon), float(after) + float(epsilon)
    if upper_after <= lower_before * float(ratio):
        return "pass"
    if lower_after > upper_before * float(ratio):
        return "fail"
    return "reference_fallback"


__all__ = ["S13MetricDrift", "S13ProbeGuardCalibration", "calibrate_probe_guard", "guarded_ratio_decision"]
