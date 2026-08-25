from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _suite(root: Path, operation: str, entry: str, statuses: list[str]) -> None:
    suite = root / f"{operation}_{entry}"
    suite.mkdir(parents=True)
    (suite / "suite.json").write_text(
        json.dumps({"operation": operation, "entry": entry}), encoding="utf-8"
    )
    for index, status in enumerate(statuses, 1):
        run = suite / f"run_{index:02d}"
        run.mkdir()
        (run / "sample.json").write_text(
            json.dumps({"status": status, "reason_code": "X", "wall_seconds": float(index),
                        "artifacts": {"video_timing.json": {"final_2d": {}}}}), encoding="utf-8"
        )


def test_statistics_keeps_failed_samples_and_uses_five_or_more() -> None:
    module = _module("summarize_linux_l1_benchmark")
    samples = [
        {"status": "PASS" if index != 2 else "FAIL", "wall_seconds": float(index)}
        for index in range(1, 7)
    ]
    result = module._statistics(samples)
    assert result["sample_count"] == 6
    assert result["success_count"] == 5
    assert result["success_rate"] == 5 / 6
    assert result["status"] == "FAIL"
    assert result["reason_code"] == "RUN_FAILURE"


def test_summary_fails_closed_for_missing_required_suite(tmp_path: Path) -> None:
    module = _module("summarize_linux_l1_benchmark")
    windows, linux = tmp_path / "windows", tmp_path / "linux"
    windows.mkdir()
    linux.mkdir()
    _suite(linux, "frozen-2d", "cli", ["PASS"] * 5)
    result = module.summarize(windows, linux)
    assert result["status"] == "FAIL"
    assert "frozen-2d:sdk" in result["missing_required_suites"]


def test_summary_keeps_2d_and_3d_suites_separate(tmp_path: Path) -> None:
    module = _module("summarize_linux_l1_benchmark")
    windows, linux = tmp_path / "windows", tmp_path / "linux"
    windows.mkdir()
    linux.mkdir()
    for operation, entry in (("frozen-2d", "cli"), ("frozen-2d", "sdk"), ("post-3d", "sdk")):
        _suite(linux, operation, entry, ["PASS"] * 5)
    result = module.summarize(windows, linux)
    assert result["status"] == "PASS"
    assert set(result["suites"]["linux"]) == {"frozen-2d:cli", "frozen-2d:sdk", "post-3d:sdk"}
    assert result["metric_boundaries"]["orb_tsdf_3d_reported_separately"] is True


def test_missing_sample_field_fails_closed(tmp_path: Path) -> None:
    module = _module("summarize_linux_l1_benchmark")
    root = tmp_path / "root"
    _suite(root, "frozen-2d", "cli", ["PASS"] * 5)
    broken = root / "frozen-2d_cli" / "run_01" / "sample.json"
    broken.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    try:
        module._load_root(root)
    except ValueError as exc:
        assert "Missing reason_code" in str(exc)
    else:
        raise AssertionError("missing sample field was accepted")


def test_rss_sampler_stops_with_process() -> None:
    module = _module("run_linux_l1_benchmark")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.25)"])
    stop = threading.Event()
    target = {}
    thread = threading.Thread(target=module._sample_process, args=(process, stop, target))
    thread.start()
    process.wait()
    stop.set()
    thread.join(2)
    assert not thread.is_alive()
    assert target["rss_sample_count"] >= 1
