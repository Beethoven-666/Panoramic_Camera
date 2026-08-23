from __future__ import annotations

from functools import wraps
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

import numpy as np
import pytest

from panorama_demo.video_s13_m61_streaming import (
    M61ROIStreamer,
    provenance_transaction_arrays,
)


ROOT = Path(__file__).resolve().parents[1]


def _isolated_rss_process(test: Callable[..., None]) -> Callable[..., None]:
    """Run a default absolute-RSS-cap test in a fresh pytest process."""

    isolated_key = f"{Path(__file__).name}::{test.__name__}"

    @wraps(test)
    def wrapper(*args: object, **kwargs: object) -> None:
        if os.environ.get("G305_PYTEST_RSS_ISOLATED_TEST") == isolated_key:
            test(*args, **kwargs)
            return
        environment = os.environ.copy()
        environment["G305_PYTEST_RSS_ISOLATED_TEST"] = isolated_key
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"{Path(__file__).as_posix()}::{test.__name__}",
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(
                "isolated RSS pytest process failed:\n" + result.stdout,
                pytrace=False,
            )

    wrapper.rss_process_isolated = True  # type: ignore[attr-defined]
    return wrapper


@_isolated_rss_process
def test_compact_roi_streaming_counts_one_remap_and_real_resources() -> None:
    calls: list[int] = []

    def remap(source: int) -> np.ndarray:
        calls.append(source)
        return np.full((8, 12, 3), source, np.uint8)

    stream = M61ROIStreamer(remap, maximum_resident_source_rois=3)
    assert stream.source_roi(0) is stream.source_roi(0)
    stream.source_roi(1)
    stream.source_roi(2)
    with stream.live_array(np.zeros((4, 5), np.float32)):
        time.sleep(0.001)
    report = stream.report()
    assert calls == [0, 1, 2]
    assert report.formal_raw_rgb_remap_invocations == 3
    assert report.formal_raw_rgb_unique_sources == 3
    assert report.maximum_resident_source_rois == 3
    assert report.peak_live_array_bytes >= 3 * 8 * 12 * 3 + 4 * 5 * 4
    assert report.peak_process_rss_bytes > 0
    assert report.elapsed_seconds > 0
    assert report.timing_source == "time.perf_counter"
    assert report.full_canvas_source_cache_created is False


@_isolated_rss_process
def test_evicted_source_cannot_be_remapped_twice() -> None:
    stream = M61ROIStreamer(lambda source: np.zeros((2, 2, 3), np.uint8),
                            maximum_resident_source_rois=1)
    stream.source_roi(0)
    stream.source_roi(1)
    with pytest.raises(RuntimeError, match="remap-once"):
        stream.source_roi(0)


def test_default_rss_cap_streaming_tests_are_process_isolated() -> None:
    assert getattr(
        test_compact_roi_streaming_counts_one_remap_and_real_resources,
        "rss_process_isolated",
        False,
    )
    assert getattr(
        test_evicted_source_cannot_be_remapped_twice,
        "rss_process_isolated",
        False,
    )


def test_live_array_and_rss_caps_fail_closed() -> None:
    stream = M61ROIStreamer(lambda source: np.zeros((2, 2, 3), np.uint8),
                            maximum_live_array_bytes=8)
    with pytest.raises(MemoryError, match="live-array"):
        stream.source_roi(0)
    stream = M61ROIStreamer(lambda source: np.zeros((2, 2, 3), np.uint8),
                            maximum_process_rss_bytes=1)
    with pytest.raises(MemoryError, match="RSS"):
        stream.source_roi(0)


def test_provenance_transactions_reject_overlap_and_bad_weights() -> None:
    shape = (2, 4)
    active0 = np.zeros(shape, bool)
    active0[:, 1] = True
    weight0 = np.zeros(shape, np.float32)
    weight0[active0] = 0.25
    result = provenance_transaction_arrays(shape, {2: (active0, weight0)})
    assert np.all(result["blend_transaction_id"][:, 1] == 2)
    assert np.all(result["secondary_weight"][:, 1] == 0.25)
    with pytest.raises(ValueError, match="overlap"):
        provenance_transaction_arrays(shape, {1: (active0, weight0), 2: (active0, weight0)})
    bad = weight0.copy()
    bad[active0] = 0.5
    with pytest.raises(ValueError, match="weight"):
        provenance_transaction_arrays(shape, {1: (active0, bad)})
