"""Native H0 timing gates; only final publication latency is authoritative."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping


def read_publish_seconds(output_root: Path) -> float:
    timing = json.loads((Path(output_root) / "video_timing.json").read_text(encoding="utf-8"))
    value = timing.get("final_2d", {}).get("capture_stop_to_p3_published_seconds")
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Missing finite final_2d.capture_stop_to_p3_published_seconds")
    return float(value)


def summarize_performance(run_roots: Mapping[str, Path]) -> dict:
    """Exactly the five planned formal samples, with linear empirical P95."""
    labels = [f"run_{index:02d}" for index in range(1, 6)]
    if set(run_roots) != set(labels):
        raise ValueError("Performance requires exactly run_01 through run_05")
    roots = [Path(run_roots[label]).resolve() for label in labels]
    if len(set(roots)) != 5:
        raise ValueError("Performance samples must be five distinct outputs")
    values = [read_publish_seconds(root) for root in roots]
    ordered = sorted(values)
    p95 = ordered[3] + 0.8 * (ordered[4] - ordered[3])
    middle = median(values)
    failures = []
    if max(values) > 60:
        failures.append("per_run_over_60_seconds")
    if middle > 15:
        failures.append("median_over_15_seconds")
    if p95 > 20:
        failures.append("p95_over_20_seconds")
    return {"schema": "gemini305-native-h0-performance/v1",
            "status": "FAIL" if failures else "PASS", "errors": failures,
            "metric": "final_2d.capture_stop_to_p3_published_seconds",
            "samples": dict(zip(labels, values)), "median_seconds": middle,
            "p95_seconds": p95, "p95_method": "linear empirical quantile (n-1)*0.95",
            "maximum_seconds": max(values)}
