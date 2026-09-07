from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
from argparse import Namespace
from pathlib import Path

import pytest


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
            json.dumps(
                {
                    "status": status,
                    "reason_code": "X",
                    "wall_seconds": float(index),
                    "artifacts": {"video_timing.json": {"final_2d": {}}},
                }
            ),
            encoding="utf-8",
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


def test_summary_keeps_suites_separate_but_rejects_exit_status_only(
    tmp_path: Path,
) -> None:
    module = _module("summarize_linux_l1_benchmark")
    windows, linux = tmp_path / "windows", tmp_path / "linux"
    windows.mkdir()
    linux.mkdir()
    for operation, entry in (
        ("frozen-2d", "cli"),
        ("frozen-2d", "sdk"),
        ("orb", "cli"),
        ("post-3d", "sdk"),
    ):
        _suite(linux, operation, entry, ["PASS"] * 5)
    result = module.summarize(windows, linux)
    assert result["status"] == "FAIL"
    assert result["reason_code"] == "RAW_EVIDENCE_INVALID"
    assert set(result["suites"]["linux"]) == {
        "frozen-2d:cli",
        "frozen-2d:sdk",
        "orb:cli",
        "post-3d:sdk",
    }


def test_native_orb_is_required_without_historical_windows(tmp_path):
    module = _module("summarize_linux_l1_benchmark")
    for operation, entry in (("frozen-2d", "cli"), ("frozen-2d", "sdk"), ("post-3d", "sdk")):
        _suite(tmp_path, operation, entry, ["PASS"] * 5)
    result = module.summarize(None, tmp_path)
    assert result["status"] == "FAIL"
    assert result["missing_required_suites"] == ["orb:cli"]
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
    thread = threading.Thread(
        target=module._sample_process, args=(process, stop, target)
    )
    thread.start()
    process.wait()
    stop.set()
    thread.join(2)
    assert not thread.is_alive()
    assert target["rss_sample_count"] >= 1


def test_worker_environment_removes_source_import_overrides(monkeypatch) -> None:
    module = _module("run_linux_l1_benchmark")
    monkeypatch.setenv("PYTHONPATH", "unfrozen-source")
    monkeypatch.setenv("PYTHONHOME", "other-python")
    monkeypatch.setenv("G305_CUDA", "off")
    env = module._clean_environment()
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env
    assert env["G305_CUDA"] == "required"


def test_installed_identity_rejects_source_checkout() -> None:
    module = _module("run_linux_l1_benchmark")
    with pytest.raises(RuntimeError, match="outside installed prefix"):
        module._installed_identity({"source_commit": "a" * 40})


def test_orb_command_binds_native_overlay(tmp_path: Path) -> None:
    module = _module("run_linux_l1_benchmark")
    args = Namespace(operation="orb", session=tmp_path / "session",
                     orb_runtime_root=tmp_path / "external-runtime")
    command, output, _ = module._command(args, tmp_path)
    assert "panorama_demo.export_orbslam3_trajectory" in command
    assert output == tmp_path
    import yaml

    overlay = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
    assert overlay == {"stitch": {"orbslam3_rgbd": {
        "runtime_kind": "native_linux", "root": str(args.orb_runtime_root.resolve())}}}


def test_post3d_command_preserves_reference_and_uses_sdk_output_contract(tmp_path: Path) -> None:
    module = _module("run_linux_l1_benchmark")
    reference = tmp_path / "reference"
    reference.mkdir()
    for name in module.TWO_D_FILES:
        (reference / name).write_bytes(("unit fixture: " + name).encode())
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    args = Namespace(operation="post-3d", entry="sdk", session=tmp_path / "session",
                     orb_runtime_root=tmp_path / "orb", two_d_output=reference)
    command, output, workload = module._command(args, run_dir)
    assert command[-2] == "--worker"
    assert output == run_dir / "2d" / "3d"
    assert Path(workload["two_d_output"]) / "3d" == output
    for name in module.TWO_D_FILES:
        assert (reference / name).read_bytes() == (output.parent / name).read_bytes()


def test_exit_success_without_artifact_contract_cannot_pass(tmp_path: Path, monkeypatch) -> None:
    module = _module("run_linux_l1_benchmark")
    args = Namespace(operation="frozen-2d", runtime_binding=tmp_path / "missing.json")
    monkeypatch.setattr(module, "_command", lambda _a, root: (
        [sys.executable, "-c", "pass"], root / "2d", {}))
    sample = module._run_one(args, tmp_path / "run", warmup=False)
    assert sample["exit_code"] == 0
    assert sample["status"] == "FAIL"
    assert sample["reason_code"] == "ARTIFACT_CONTRACT_FAILED"
