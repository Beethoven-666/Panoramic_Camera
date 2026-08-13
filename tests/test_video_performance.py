from __future__ import annotations

import time

from panorama_demo.video_performance import VideoPerformanceProfiler


def test_primary_delivery_timing_excludes_audit_and_offline_measurement() -> None:
    profiler = VideoPerformanceProfiler()
    with profiler.audit_export():
        time.sleep(0.003)
    with profiler.offline_evaluation():
        time.sleep(0.003)

    result = profiler.as_dict(maximum_post_seconds=1.0)

    assert set(result) >= {
        "primary_post_capture_seconds",
        "audit_export_seconds",
        "offline_evaluation_seconds",
    }
    assert result["audit_export_seconds"] > 0.0
    assert result["offline_evaluation_seconds"] > 0.0
    assert result["primary_post_capture_seconds"] >= 0.0
    assert "post_capture_seconds" not in result
