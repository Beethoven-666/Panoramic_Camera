#!/usr/bin/env python3
"""Isolated L1 benchmark runner for the existing CLI and public video SDK."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_identity(binding: dict[str, Any]) -> dict[str, object]:
    """Qualification imports must come from the offline-installed prefix."""
    spec = importlib.util.find_spec("panorama_demo")
    if spec is None or spec.origin is None:
        raise RuntimeError("Installed SDK is unavailable")
    package = Path(spec.origin).resolve()
    prefix = Path(sys.prefix).resolve()
    if not package.is_relative_to(prefix):
        raise RuntimeError(f"SDK imported outside installed prefix: {package}")
    source_commit = (package.parent / "_runtime/source_commit.txt").read_text(
        encoding="ascii").strip()
    if source_commit != binding.get("source_commit"):
        raise RuntimeError("Installed SDK source commit differs from binding")
    return {"prefix": str(prefix), "package": str(package),
            "source_commit": source_commit, "python": sys.executable}


def _clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["G305_CUDA"] = "required"
    return env


def _sample_process(
    process: subprocess.Popen[str], stop: threading.Event, target: dict[str, Any]
) -> None:
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
    except psutil.Error:
        pass
    finally:
        target.update(
            {"status": "PASS" if samples else "NOT_EXECUTED",
             "peak_process_tree_rss_bytes": peak if samples else None,
             "rss_sample_count": samples}
        )


def _sample_gpu(stop: threading.Event, target: dict[str, Any]) -> None:
    memory: list[float] = []
    utilization: list[float] = []
    executable = shutil.which("nvidia-smi")
    if executable is None and Path("/usr/lib/wsl/lib/nvidia-smi").is_file():
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if executable is None:
        target.update({"status": "NOT_EXECUTED", "reason_code": "NVIDIA_SMI_MISSING"})
        return
    while not stop.wait(0.2):
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False, timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
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
    target.update(
        {
            "status": "PASS" if memory else "NOT_EXECUTED",
            "reason_code": "SAMPLED" if memory else "NO_GPU_SAMPLES",
            "memory_used_mib": memory,
            "utilization_percent": utilization,
            "peak_gpu_memory_mib": max(memory) if memory else None,
            "scope": "whole_device",
        }
    )


def _extract_artifacts(run_dir: Path, product_output: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name in (
        "video_delivery.json",
        "video_report.json",
        "video_timing.json",
        "live_capture_report.json",
        "video_3d_delivery.json",
        "video_3d_timing.json",
        "orbslam3_trajectory.json",
        "orbslam3_trajectory.audit.json",
    ):
        candidates = [
            product_output / name,
            product_output / "3d" / name,
            run_dir / name,
        ]
        source = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if source is not None:
            destination = run_dir / "raw" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            artifacts[name] = _json(destination)
    return artifacts


def _worker(arguments: list[str]) -> int:
    payload = json.loads(arguments[0])
    if payload["operation"] == "frozen-2d":
        from panorama_demo.sdk_acceptance import audit_2d_process

        with audit_2d_process(payload["output"]):
            if payload.get("entry") == "cli":
                from panorama_demo.video_panorama import main

                sys.argv = [
                    "g305-video-panorama",
                    payload["session"],
                    "--output",
                    payload["output"],
                    "--maximum-post-seconds",
                    str(payload["maximum_post_seconds"]),
                    "--defer-3d",
                ]
                main()
            else:
                _sdk_worker(arguments)
        return 0
    return _sdk_worker(arguments)


def _sdk_worker(arguments: list[str]) -> int:
    payload = json.loads(arguments[0])
    operation = payload["operation"]
    entry = payload.get("entry", "cli")
    if entry != "sdk":
        raise ValueError("Internal worker is SDK-only")
    from panorama_demo import Gemini305VideoSDK, VideoSDKConfig

    settings = {"cuda_mode": "required",
                "state_root": str(Path(payload["output"]).parent / ".sdk-state")}
    if payload.get("orb_runtime_root"):
        settings.update(orbslam3_root=payload["orb_runtime_root"],
                        orb_runtime_kind="native_linux")
    if payload.get("failure_injection"):
        settings["orbslam3_executable"] = "Examples/RGB-D/acceptance_missing_executable"
    sdk = Gemini305VideoSDK(VideoSDKConfig(**settings))
    if operation == "frozen-2d":
        sdk.process_session(
            payload["session"],
            payload["output"],
            defer_3d=True,
            maximum_post_seconds=payload["maximum_post_seconds"],
        )
    elif operation == "physical-live":
        job = sdk.capture_and_process(
            payload["capture_root"],
            payload["output"],
            width=payload["width"],
            height=payload["height"],
            fps=payload["fps"],
            duration_seconds=payload["duration"],
            video_exposure_us=payload["video_exposure_us"],
            preview_window=not payload["no_preview"],
            maximum_post_seconds=payload["maximum_post_seconds"],
        )
        result = job.result()
        print(
            json.dumps({"session": str(result.session), "job_state": job.state.value})
        )
    elif operation == "post-3d":
        if payload.get("failure_injection"):
            return _failure_isolation(sdk, payload)
        sdk.run_post_3d(payload["session"], payload["two_d_output"],
                        output_dir=payload["output"])
    else:
        raise ValueError(f"Unsupported SDK worker operation: {operation}")
    return 0


TWO_D_FILES = ("video_delivery.json", "video_panorama.png", "video_panorama.jpg",
               "video_pixel_provenance.npz", "video_report.json", "video_timing.json")


def _copy_two_d(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=False)
    for name in TWO_D_FILES + ("process_audit.json",):
        if name == "process_audit.json" and not (source / name).is_file():
            continue
        shutil.copyfile(source / name, target / name)


def _failure_isolation(sdk: Any, payload: dict[str, Any]) -> int:
    """Call the public API and its real child; only the requested executable is absent."""
    from panorama_demo.sdk_state import ThreeDProcessingError
    from panorama_demo.video_sdk import VideoPanoramaResult

    two_d, output = Path(payload["two_d_output"]), Path(payload["output"])
    before_root = two_d.parent / "before_2d"
    _copy_two_d(two_d, before_root)
    before = {name: _sha256(two_d / name) for name in TWO_D_FILES}
    error = None
    try:
        sdk.run_post_3d(payload["session"], two_d, output_dir=output)
    except ThreeDProcessingError as exc:
        error = exc
    after = {name: _sha256(two_d / name) for name in TWO_D_FILES}
    loaded = VideoPanoramaResult.load(payload["session"], two_d)
    child_logs = list(output.glob("post-3d-*.log"))
    log_text = "\n".join(path.read_text(errors="replace") for path in child_logs)
    timing_path = output / "video_3d_timing.json"
    timing = _json(timing_path) if timing_path.is_file() else {}
    child_executed = (output / "process_boundary.json").is_file()
    injected = ("acceptance_missing_executable" in log_text
                and "orb_started_monotonic_ns" in timing
                and timing.get("status") == "failed")
    passed = (error is not None and before == after and loaded.is_published
              and child_executed and injected
              and (output / "video_3d_failure.json").is_file()
              and not (output / "video_3d_delivery.json").exists())
    evidence = {
        "schema": "gemini305-sdk-post-3d-failure-isolation/v1",
        "status": "PASS" if passed else "FAIL",
        "before_2d": str(before_root), "after_2d": str(two_d),
        "before_sha256": before, "after_sha256": after,
        "typed_error": type(error).__name__ if error else None,
        "error_code": error.error_code if error else None,
        "task_state": "THREE_D_FAILED_2D_PRESERVED" if passed else "FAIL",
        "result_loadable": loaded.is_published,
        "orb_failure_injected": injected, "child_executed": child_executed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "failure_isolation.json").write_text(json.dumps(evidence, indent=2))
    return 0 if passed else 1


def _native_overlay(root: Path, run_dir: Path) -> Path:
    import yaml

    path = run_dir / "native-runtime.yaml"
    path.write_text(yaml.safe_dump({"stitch": {"orbslam3_rgbd": {
        "runtime_kind": "native_linux", "root": str(root.resolve()),
    }}}), encoding="utf-8")
    return path


def _command(
    args: argparse.Namespace, run_dir: Path
) -> tuple[list[str], Path, dict[str, Any]]:
    output = run_dir / ("2d/3d" if args.operation == "post-3d" else "2d")
    common: dict[str, Any] = {
        "operation": args.operation,
        "entry": getattr(args, "entry", "cli"),
        "output": str(output),
        "maximum_post_seconds": getattr(args, "maximum_post_seconds", 60.0),
        "orb_runtime_root": str(args.orb_runtime_root.resolve()) if getattr(args, "orb_runtime_root", None) else None,
    }
    if args.operation in {"frozen-2d", "post-3d", "orb"}:
        common["session"] = str(args.session.resolve())
    if args.operation == "post-3d":
        _copy_two_d(args.two_d_output.resolve(), run_dir / "2d")
        common["two_d_output"] = str((run_dir / "2d").resolve())
        common["failure_injection"] = getattr(args, "failure_injection", False)
    if args.operation == "physical-live":
        common.update(
            {
                name: getattr(args, name)
                for name in (
                    "width",
                    "height",
                    "fps",
                    "duration",
                    "video_exposure_us",
                    "no_preview",
                )
            }
        )
        common["capture_root"] = str((run_dir / "sessions").resolve())
    if args.operation == "frozen-2d" or (
        getattr(args, "entry", "cli") == "sdk" and args.operation != "orb"
    ):
        return (
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                json.dumps(common),
            ],
            output,
            common,
        )
    if args.operation == "physical-live":
        command = [
            sys.executable,
            "-m",
            "panorama_demo.video_live",
            "--output",
            str(run_dir / "sessions"),
            "--panorama-output",
            str(output),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--fps",
            str(args.fps),
            "--duration",
            str(args.duration),
            "--video-exposure-us",
            str(args.video_exposure_us),
            "--maximum-post-seconds",
            str(args.maximum_post_seconds),
        ]
        if args.no_preview:
            command.append("--no-preview")
        return command, output, common
    if args.operation == "post-3d":
        return (
            [
                sys.executable,
                "-m",
                "panorama_demo.video_3d_postprocess",
                str(args.session),
                "--two-d-output",
                common["two_d_output"],
                "--output",
                str(output),
                "--config",
                str(_native_overlay(args.orb_runtime_root, run_dir)),
            ],
            output,
            common,
        )
    return (
        [
            sys.executable,
            "-m",
            "panorama_demo.export_orbslam3_trajectory",
            str(args.session),
            "--output",
            str(run_dir / "orbslam3_trajectory.json"),
            "--config",
            str(_native_overlay(args.orb_runtime_root, run_dir)),
        ],
        run_dir,
        common,
    )


def _run_one(
    args: argparse.Namespace, run_dir: Path, *, warmup: bool
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=False)
    command, output, workload = _command(args, run_dir)
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2), encoding="utf-8"
    )
    stdout_path, stderr_path = run_dir / "stdout.txt", run_dir / "stderr.txt"
    start = time.perf_counter()
    rss: dict[str, Any] = {}
    gpu: dict[str, Any] = {}
    stop = threading.Event()
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command, stdout=stdout, stderr=stderr, text=True,
            env=_clean_environment(), cwd=run_dir,
        )
        rss_thread = threading.Thread(
            target=_sample_process, args=(process, stop, rss), daemon=True
        )
        gpu_thread = threading.Thread(target=_sample_gpu, args=(stop, gpu), daemon=True)
        rss_thread.start()
        gpu_thread.start()
        exit_code = process.wait()
        wall = time.perf_counter() - start
        stop.set()
        rss_thread.join(5)
        gpu_thread.join(5)
    artifact_error = None
    try:
        artifacts = _extract_artifacts(run_dir, output)
    except (OSError, ValueError) as exc:
        artifacts = {}
        artifact_error = f"{type(exc).__name__}: {exc}"
    contract: dict[str, Any] = {"status": "FAIL", "reason": "PROCESS_EXIT_NONZERO"}
    if artifact_error is not None:
        contract["reason"] = artifact_error
    elif exit_code == 0:
        try:
            from panorama_demo.sdk_acceptance import (
                verify_2d_contract, verify_3d, verify_native_orb,
            )

            binding = _json(args.runtime_binding)
            if args.operation in {"frozen-2d", "physical-live"}:
                contract = verify_2d_contract(output, expected_binding=(
                    binding if args.operation == "frozen-2d" else None))
            elif args.operation == "orb":
                contract = verify_native_orb(output, expected_binding=binding)
            else:
                contract = verify_3d(output, expected_binding=binding,
                                     failure=getattr(args, "failure_injection", False))
        except Exception as exc:
            contract = {"status": "FAIL", "reason": f"{type(exc).__name__}: {exc}"}
    passed = exit_code == 0 and contract.get("status") == "PASS"
    sample = {
        "schema": SCHEMA,
        "warmup": warmup,
        "status": "PASS" if passed else "FAIL",
        "reason_code": "CONTRACT_VERIFIED" if passed else (
            "PROCESS_EXIT_NONZERO" if exit_code else "ARTIFACT_CONTRACT_FAILED"),
        "exit_code": exit_code,
        "wall_seconds": wall,
        "rss": rss,
        "gpu": gpu,
        "workload_request": workload,
        "artifacts": artifacts,
        "product_output": str(output),
        "contract": contract,
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
        item.add_argument("--runtime-binding", type=Path, required=True)
        item.add_argument("--orb-runtime-root", type=Path,
                          required=name in {"orb", "post-3d"})
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
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    os.environ["G305_CUDA"] = "required"
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        return _worker(sys.argv[2:])
    args = _parser().parse_args()
    if args.runs < 1 or args.warmups < 0:
        raise SystemExit("runs must be positive and warmups non-negative")
    root = args.artifact_root.resolve()
    binding = _json(args.runtime_binding)
    installed = _installed_identity(binding)
    root.mkdir(parents=True, exist_ok=False)
    identity = {
        "schema": SCHEMA,
        "platform_label": args.platform_label,
        "platform": platform.platform(),
        "installed": installed,
        "binding": binding,
        "operation": args.operation,
        "entry": getattr(args, "entry", "cli"),
        "runs": args.runs,
        "warmups": args.warmups,
    }
    (root / "suite.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")
    shutil.copyfile(args.runtime_binding, root / "binding.json")
    from panorama_demo import Gemini305VideoSDK, VideoSDKConfig

    settings = {"cuda_mode": "required"}
    if args.orb_runtime_root is not None:
        settings.update(orbslam3_root=args.orb_runtime_root, orb_runtime_kind="native_linux")
    (root / "doctor.json").write_text(
        json.dumps(Gemini305VideoSDK(VideoSDKConfig(**settings)).doctor().as_dict(), indent=2)
    )
    samples = []
    for index in range(args.warmups):
        samples.append(_run_one(args, root / f"warmup_{index + 1:02d}", warmup=True))
    for index in range(args.runs):
        samples.append(_run_one(args, root / f"run_{index + 1:02d}", warmup=False))
    success = sum(
        sample["status"] == "PASS" for sample in samples if not sample["warmup"]
    )
    isolation = None
    if args.operation == "post-3d" and args.entry == "sdk":
        args.failure_injection = True
        isolation = _run_one(args, root / "failure-isolation", warmup=False)
    passed = (success == args.runs and all(sample["status"] == "PASS" for sample in samples)
              and (isolation is None or isolation["status"] == "PASS"))
    summary = {
        **identity,
        "status": "PASS" if passed else "FAIL",
        "reason_code": "ALL_RUNS_PASSED" if passed else "RUN_FAILURE",
        "successful_runs": success,
        "required_runs": args.runs,
        "failure_isolation": isolation,
    }
    (root / "suite_result.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
