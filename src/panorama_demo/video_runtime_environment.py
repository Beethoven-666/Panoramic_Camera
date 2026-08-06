"""Reproducibility evidence for the isolated video experiment tools.

This module deliberately does not import a renderer.  It records the runtime
that *called* one, so benchmark and replay evidence cannot affect source
selection, poses, ownership, or pixels.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


RUNTIME_ENVIRONMENT_SCHEMA = "gemini305-video-runtime-environment/v1"
DETERMINISTIC_RESULT_SCHEMA = "gemini305-video-deterministic-result/v1"
_SEED = 20_260_804


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _nvidia_smi() -> dict[str, object] | None:
    """Return driver/device evidence when nvidia-smi is actually installed."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    devices: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            devices.append(
                {"name": fields[0], "driver_version": fields[1], "memory_total_mib": fields[2]}
            )
    return {"devices": devices} if devices else None


def capture_runtime_environment() -> dict[str, object]:
    """Capture dependency and CUDA facts without making CUDA a requirement."""

    packages = {
        name: _package_version(name)
        for name in ("numpy", "opencv-python", "open3d", "cupy-cuda13x", "torch", "torchvision")
    }
    torch_info: dict[str, object] = {"available": False}
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "amp_available": hasattr(torch, "amp"),
            "cuda_graph_available": hasattr(torch.cuda, "CUDAGraph"),
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
        if cuda_available:
            torch_info["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return {
        "schema": RUNTIME_ENVIRONMENT_SCHEMA,
        "seed": _SEED,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "source_commit": _git_commit(),
        "packages": packages,
        "torch": torch_info,
        "nvidia_smi": _nvidia_smi(),
        "requested_cuda_mode": os.environ.get("G305_CUDA"),
    }


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write canonical, stable JSON without leaving a partial evidence file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    try:
        pending.write_bytes(_canonical_bytes(payload) + b"\n")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def deterministic_result_payload(output: Path, report: Mapping[str, Any]) -> dict[str, object]:
    """Return a timing-free, content-addressed result record.

    Timers, temporary paths, and the mutable report/delivery JSON bytes are
    intentionally excluded.  The record is stable iff the primary rendered
    pixels/provenance and locked algorithm identity are stable.
    """

    output = output.expanduser().resolve()
    required = ("video_panorama.png", "video_panorama.jpg", "video_pixel_provenance.npz")
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot record deterministic result; primary artifacts missing: " + ", ".join(missing)
        )
    algorithm = report.get("algorithm")
    grades = report.get("grades")
    if not isinstance(algorithm, Mapping) or not isinstance(grades, Mapping):
        raise ValueError("Benchmark report lacks immutable algorithm identity or grades")
    algorithm_identity = {
        key: algorithm.get(key)
        for key in (
            "role", "algorithm_id", "implementation_id", "config_sha256", "source_commit", "model_sha256"
        )
    }
    artifacts = {
        name: {"bytes": (output / name).stat().st_size, "sha256": _sha256_file(output / name)}
        for name in required
    }
    core: dict[str, object] = {
        "schema": DETERMINISTIC_RESULT_SCHEMA,
        "algorithm": algorithm_identity,
        "evaluation_scope": report.get("evaluation_scope"),
        "input_sha256": report.get("input_sha256"),
        "grades": {key: grades.get(key) for key in ("structural", "visual", "performance", "overall")},
        "primary_artifacts": artifacts,
    }
    core["result_sha256"] = hashlib.sha256(_canonical_bytes(core)).hexdigest()
    return core


def write_deterministic_result(output: Path, report: Mapping[str, Any], *, name: str = "result.json") -> dict[str, object]:
    payload = deterministic_result_payload(output, report)
    atomic_write_json(output / name, payload)
    return payload
