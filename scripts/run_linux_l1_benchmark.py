#!/usr/bin/env python3
"""Isolated L1 benchmark runner for the existing CLI and public video SDK."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil


SCHEMA = "gemini305-linux-l1-benchmark/v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _git_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.splitlines()
    return {"head": head, "status": status}


def _sample_process(process: subprocess.Popen[str], stop: threading.Event, target: dict[str, Any]) -> None:
    peak = 0
    samples = 0
    try:
        root = psutil.Process(process.pid)
        while not stop.wait(0.1):
            members = [root]
            try:
                members.extend(root.children(recursive=True))
            except psutil.Error:
                pass
            rss = 0
            for member in members:
                try:
                    rss += int(member.memory_info().rss)
                except psutil.Error:
                    pass
            peak = max(peak, rss)
            samples += 1
    finally:
        target.update({"peak_process_tree_rss_bytes": peak, "rss_sample_count": samples})


def _sample_gpu(stop: threading.Event, target: dict[str, Any]) -> None:
    memory: list[float] = []
    utilization: list[float] = []
    executable = shutil.which("nvidia-smi")
    if executable is None:
        target.update({"status": "NOT_EXECUTED", "reason_code": "NVIDIA_SMI_MISSING"})
        return
    while not stop.wait(0.2):
        completed = subprocess.run(
            [executable, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if completed.returncode:
            continue
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2:
                try:
                    memory.append(float(fields[0]))
                    utilization.append(float(fields[1]))
                except ValueError:
                    pass
    target.update({
        "status": "PASS" if memory else "NOT_EXECUTED",
        "reason_code": "SAMPLED" if memory else "NO_GPU_SAMPLES",
        "memory_used_mib": memory,
        "utilization_percent": utilization,
    })


def _extract_artifacts(run_dir: Path, product_output: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name in (
        "video_delivery.json", "video_report.json", "video_timing.json",
        "live_capture_report.json", "video_3d_delivery.json", "video_3d_timing.json",
        "orbslam3_trajectory.json", "orbslam3_trajectory.audit.json",
    ):
        candidates = [product_output / name, product_output / "3d" / name, run_dir / name]
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is not None:
            destination = run_dir / "raw" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            artifacts[name] = _json(destination)
    return artifacts


def _worker(arguments: list[str]) -> int:
    payload = json.loads(arguments[0])
    operation = payload["operation"]
    entry = payload.get("entry", "cli")
    if entry != "sdk":
        raise ValueError("Internal worker is SDK-only")
    from panorama_demo import Gemini305VideoSDK

    sdk = Gemini305VideoSDK()
    if operation == "frozen-2d":
        sdk.process_session(payload["session"], payload["output"], defer_3d=True,
                            maximum_post_seconds=payload["maximum_post_seconds"])
    elif operation == "physical-live":
        job = sdk.capture_and_process(
            payload["capture_root"], payload["output"], width=payload["width"],
            height=payload["height"], fps=payload["fps"],
            duration_seconds=payload["duration"], video_exposure_us=payload["video_exposure_us"],
            preview_window=not payload["no_preview"],
            maximum_post_seconds=payload["maximum_post_seconds"],
        )
        result = job.result()
        print(json.dumps({"session": str(result.session), "job_state": job.state.value}))
    elif operation == "post-3d":
        sdk.run_post_3d(payload["session"], payload["two_d_output"], output_dir=payload["output"])
    else:
        raise ValueError(f"Unsupported SDK worker operation: {operation}")
    return 0


def _command(args: argparse.Namespace, run_dir: Path) -> tuple[list[str], Path, dict[str, Any]]:
    output = run_dir / ("3d" if args.operation == "post-3d" else "2d")
    common: dict[str, Any] = {
        "operation": args.operation, "entry": getattr(args, "entry", "cli"),
        "output": str(output), "maximum_post_seconds": getattr(args, "maximum_post_seconds", 60.0),
    }
    if args.operation in {"frozen-2d", "post-3d", "orb"}:
        common["session"] = str(args.session.resolve())
    if args.operation == "post-3d":
        common["two_d_output"] = str(args.two_d_output.resolve())
    if args.operation == "physical-live":
        common.update({name: getattr(args, name) for name in (
            "width", "height", "fps", "duration", "video_exposure_us", "no_preview")})
        common["capture_root"] = str((run_dir / "sessions").resolve())
    if getattr(args, "entry", "cli") == "sdk" and args.operation != "orb":
        return [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(common)], output, common
    if args.operation == "frozen-2d":
        return [sys.executable, "-m", "panorama_demo.video_panorama", str(args.session),
                "--output", str(output), "--maximum-post-seconds", str(args.maximum_post_seconds),
                "--defer-3d"], output, common
    if args.operation == "physical-live":
        command = [sys.executable, "-m", "panorama_demo.video_live", "--output", str(run_dir / "sessions"),
                   "--panorama-output", str(output), "--width", str(args.width), "--height", str(args.height),
                   "--fps", str(args.fps), "--duration", str(args.duration), "--video-exposure-us",
                   str(args.video_exposure_us), "--maximum-post-seconds", str(args.maximum_post_seconds)]
        if args.no_preview:
            command.append("--no-preview")
        return command, output, common
    if args.operation == "post-3d":
        return [sys.executable, "-m", "panorama_demo.video_3d_postprocess", str(args.session),
                "--two-d-output", str(args.two_d_output), "--output", str(output)], output, common
    return [sys.executable, "-m", "panorama_demo.export_orbslam3_trajectory", str(args.session),
            "--output", str(run_dir / "orbslam3_trajectory.json")], run_dir, common


def _run_one(args: argparse.Namespace, run_dir: Path, *, warmup: bool) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    command, output, workload = _command(args, run_dir)
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    stdout_path, stderr_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
    start = time.perf_counter()
    rss: dict[str, Any] = {}
    gpu: dict[str, Any] = {}
    stop = threading.Event()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr, text=True, env=os.environ.copy())
        rss_thread = threading.Thread(target=_sample_process, args=(process, stop, rss), daemon=True)
        gpu_thread = threading.Thread(target=_sample_gpu, args=(stop, gpu), daemon=True)
        rss_thread.start()
        gpu_thread.start()
        exit_code = process.wait()
        stop.set()
        rss_thread.join(5)
        gpu_thread.join(5)
    wall = time.perf_counter() - start
    artifacts = _extract_artifacts(run_dir, output)
    sample = {
        "schema": SCHEMA, "warmup": warmup, "status": "PASS" if exit_code == 0 else "FAIL",
        "reason_code": "COMPLETED" if exit_code == 0 else "PROCESS_EXIT_NONZERO",
        "exit_code": exit_code, "wall_seconds": wall, "rss": rss, "gpu": gpu,
        "workload_request": workload, "artifacts": artifacts,
    }
    (run_dir / "sample.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    return sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("frozen-2d", "physical-live", "post-3d", "orb"):
        item = sub.add_parser(name)
        item.add_argument("--artifact-root", type=Path, required=True)
        item.add_argument("--runs", type=int, default=5)
        item.add_argument("--warmups", type=int, default=1)
        item.add_argument("--platform-label", required=True)
        if name != "orb":
            item.add_argument("--entry", choices=("cli", "sdk"), default="cli")
        if name in {"frozen-2d", "post-3d", "orb"}:
            item.add_argument("--session", type=Path, required=True)
        if name in {"frozen-2d", "physical-live"}:
            item.add_argument("--maximum-post-seconds", type=float, default=60.0)
        if name == "frozen-2d":
            item.add_argument("--cadence-scale", type=float, default=1.0)
        if name == "post-3d":
            item.add_argument("--two-d-output", type=Path, required=True)
        if name == "physical-live":
            item.add_argument("--width", type=int, default=848)
            item.add_argument("--height", type=int, default=480)
            item.add_argument("--fps", type=int, default=60)
            item.add_argument("--duration", type=float, default=3.0)
            item.add_argument("--video-exposure-us", type=int, default=100)
            item.add_argument("--no-preview", action="store_true")
    return parser


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        return _worker(sys.argv[2:])
    args = _parser().parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("runs must be positive and warmups non-negative")
    root = args.artifact_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    identity = {"schema": SCHEMA, "platform_label": args.platform_label,
                "platform": platform.platform(), "git": _git_identity(), "operation": args.operation,
                "entry": getattr(args, "entry", "cli"), "runs": args.runs, "warmups": args.warmups}
    (root / "suite.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    samples = []
    for index in range(args.warmups):
        samples.append(_run_one(args, root / f"warmup_{index + 1:02d}", warmup=True))
    for index in range(args.runs):
        samples.append(_run_one(args, root / f"run_{index + 1:02d}", warmup=False))
    success = sum(sample["status"] == "PASS" for sample in samples if not sample["warmup"])
    summary = {**identity, "status": "PASS" if success == args.runs else "FAIL",
               "reason_code": "ALL_RUNS_PASSED" if success == args.runs else "RUN_FAILURE",
               "successful_runs": success, "required_runs": args.runs}
    (root / "suite_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if success == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
