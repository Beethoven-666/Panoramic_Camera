import json

import pytest

from qualification.native_h0.performance import read_publish_seconds, summarize_performance


def _runs(tmp_path, values):
    roots = {}
    for index, value in enumerate(values, 1):
        root = tmp_path / f"run_{index:02d}"
        root.mkdir()
        (root / "video_timing.json").write_text(json.dumps({"final_2d": {
            "capture_stop_to_p3_published_seconds": value}}))
        roots[root.name] = root
    return roots


def test_exact_samples_and_percentile(tmp_path):
    report = summarize_performance(_runs(tmp_path, [10, 11, 12, 13, 20]))
    assert report["status"] == "PASS"
    assert report["median_seconds"] == 12
    assert report["p95_seconds"] == pytest.approx(18.6)


@pytest.mark.parametrize("values", [[10, 11, 12, 13, 23], [16] * 5, [1, 1, 1, 1, 61]])
def test_each_performance_gate_rejects(tmp_path, values):
    assert summarize_performance(_runs(tmp_path, values))["status"] == "FAIL"


@pytest.mark.parametrize("count", [0, 4, 6])
def test_no_replacement_or_missing_runs(tmp_path, count):
    with pytest.raises(ValueError, match="exactly"):
        summarize_performance(_runs(tmp_path, [1] * count))


@pytest.mark.parametrize("value", [None, True, -1, "1", float("nan"), float("inf")])
def test_missing_or_false_timing_cannot_pass(tmp_path, value):
    roots = _runs(tmp_path, [value])
    with pytest.raises(ValueError, match="finite"):
        read_publish_seconds(roots["run_01"])


def test_legacy_metric_is_not_substitute(tmp_path):
    (tmp_path / "video_timing.json").write_text(json.dumps({"primary_post_capture_seconds": 1}))
    with pytest.raises(ValueError):
        read_publish_seconds(tmp_path)
