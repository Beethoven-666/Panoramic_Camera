"""Compact ROI streaming and truthful resource accounting for S013 M6.1."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Mapping

import numpy as np


@dataclass(frozen=True)
class M61StreamingReport:
    formal_raw_rgb_remap_invocations: int
    formal_raw_rgb_unique_sources: int
    maximum_resident_source_rois: int
    peak_live_array_bytes: int
    peak_process_rss_bytes: int
    elapsed_seconds: float
    timing_source: str
    full_canvas_source_cache_created: bool


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        return 0


class M61ROIStreamer:
    """One formal remap per source with an LRU of at most five compact ROIs."""

    def __init__(
        self,
        remap_once: Callable[[int], np.ndarray],
        *,
        maximum_resident_source_rois: int = 5,
        maximum_live_array_bytes: int = 1_500_000_000,
        maximum_process_rss_bytes: int = 2_000_000_000,
    ) -> None:
        if not 1 <= maximum_resident_source_rois <= 5:
            raise ValueError("M6.1 resident ROI cap must be in [1, 5]")
        self._remap_once = remap_once
        self._resident_cap = maximum_resident_source_rois
        self._live_cap = maximum_live_array_bytes
        self._rss_cap = maximum_process_rss_bytes
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._seen: set[int] = set()
        self._live: dict[int, np.ndarray] = {}
        self._peak_live = 0
        self._peak_rss = _rss_bytes()
        self._peak_resident = 0
        self._started = time.perf_counter()

    def _measure(self) -> None:
        live = sum(int(array.nbytes) for array in self._live.values())
        resident = sum(int(array.nbytes) for array in self._cache.values())
        self._peak_live = max(self._peak_live, live + resident)
        self._peak_rss = max(self._peak_rss, _rss_bytes())
        self._peak_resident = max(self._peak_resident, len(self._cache))
        if self._peak_live > self._live_cap:
            raise MemoryError("M6.1 live-array byte cap exceeded")
        if self._peak_rss and self._peak_rss > self._rss_cap:
            raise MemoryError("M6.1 process RSS cap exceeded")

    def source_roi(self, source_index: int) -> np.ndarray:
        if source_index in self._cache:
            self._cache.move_to_end(source_index)
            return self._cache[source_index]
        if source_index in self._seen:
            raise RuntimeError("M6.1 source ROI was evicted before its final use; remap-once violated")
        value = np.asarray(self._remap_once(source_index))
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            raise ValueError("M6.1 formal remap must return compact uint8 BGR ROI")
        self._seen.add(source_index)
        self._cache[source_index] = value
        while len(self._cache) > self._resident_cap:
            self._cache.popitem(last=False)
        self._measure()
        return value

    def release(self, source_index: int) -> None:
        self._cache.pop(source_index, None)
        self._measure()

    @contextmanager
    def live_array(self, array: np.ndarray) -> Iterator[np.ndarray]:
        value = np.asarray(array)
        token = id(value)
        if token in self._live:
            raise ValueError("M6.1 live array registered twice")
        self._live[token] = value
        self._measure()
        try:
            yield value
        finally:
            self._live.pop(token, None)
            self._measure()

    def report(self) -> M61StreamingReport:
        self._measure()
        return M61StreamingReport(
            formal_raw_rgb_remap_invocations=len(self._seen),
            formal_raw_rgb_unique_sources=len(self._seen),
            maximum_resident_source_rois=self._peak_resident,
            peak_live_array_bytes=self._peak_live,
            peak_process_rss_bytes=self._peak_rss,
            elapsed_seconds=float(time.perf_counter() - self._started),
            timing_source="time.perf_counter",
            full_canvas_source_cache_created=False,
        )


def provenance_transaction_arrays(
    shape: tuple[int, int], transactions: Mapping[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Build transaction/weight planes without a source-count×canvas allocation."""

    transaction_id = np.full(shape, -1, np.int32)
    secondary_weight = np.zeros(shape, np.float32)
    for pair_index in sorted(transactions):
        active, weight = transactions[pair_index]
        active = np.asarray(active, bool)
        weight = np.asarray(weight, np.float32)
        if active.shape != shape or weight.shape != shape:
            raise ValueError("M6.1 provenance transaction shape disagrees")
        if np.any(active & (transaction_id >= 0)):
            raise ValueError("M6.1 blend transaction masks overlap")
        if np.any((weight > 0) != active) or np.any(weight < 0) or np.any(weight >= 0.5):
            raise ValueError("M6.1 provenance weight/active contract disagrees")
        transaction_id[active] = int(pair_index)
        secondary_weight[active] = weight[active]
    return {"blend_transaction_id": transaction_id, "secondary_weight": secondary_weight}


__all__ = ["M61ROIStreamer", "M61StreamingReport", "provenance_transaction_arrays"]
