"""Lightweight parent-process launcher for isolated post-capture 3-D."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    try:
        pending.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_isolated_3d_failure(
    output_3d: Path, *, two_d_output: Path, exc: Exception, stage: str
) -> dict[str, object]:
    """Pure-stdlib failure publication which never writes inside the 2-D root."""

    marker = two_d_output / "video_delivery.json"
    marker_sha = _sha256(marker) if marker.is_file() else None
    payload: dict[str, object] = {
        "schema": "gemini305-video-3d-failure/v2",
        "two_d_delivery_preserved": marker_sha is not None,
        "two_d_delivery_sha256": marker_sha,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "stage": stage,
    }
    _atomic_json(output_3d / "video_3d_failure.json", payload)
    after = _sha256(marker) if marker.is_file() else None
    if after != marker_sha:
        raise RuntimeError("3-D failure handling changed the 2-D delivery") from exc
    return payload


def spawn_post_capture_3d(
    *,
    session_path: Path,
    two_d_output: Path,
    config_path: Path | None,
    two_d_delivery_published_monotonic_ns: int,
    two_d_resources_released_monotonic_ns: int,
    popen: Any = subprocess.Popen,
) -> subprocess.Popen[Any]:
    """Record the ordering proof, then create without waiting for the child."""

    output = two_d_output.expanduser().resolve()
    marker = output / "video_delivery.json"
    if not marker.is_file():
        raise ValueError("Cannot spawn post-capture 3-D before 2-D delivery publication")
    spawned_ns = time.monotonic_ns()
    if not (
        spawned_ns > two_d_resources_released_monotonic_ns
        > two_d_delivery_published_monotonic_ns
    ):
        raise RuntimeError("Post-capture 3-D process ordering is invalid")
    timing_path = output / "video_timing.json"
    timing: dict[str, object] = {}
    if timing_path.is_file():
        loaded = json.loads(timing_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            timing.update(loaded)
    timing["isolation"] = {
        "two_d_delivery_published_monotonic_ns": two_d_delivery_published_monotonic_ns,
        "two_d_resources_released_monotonic_ns": two_d_resources_released_monotonic_ns,
        "three_d_process_spawned_monotonic_ns": spawned_ns,
        "three_d_process_boundary": "independent_post_capture_process",
        "parent_waits_for_three_d": False,
    }
    _atomic_json(timing_path, timing)
    command = [
        sys.executable,
        "-m",
        "panorama_demo.video_3d_postprocess",
        str(session_path.expanduser().resolve()),
        "--two-d-output",
        str(output),
        "--output",
        str(output / "3d"),
    ]
    if config_path is not None:
        command.extend(("--config", str(config_path.expanduser().resolve())))
    creationflags = (
        getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
        if os.name == "nt"
        else 0
    )
    try:
        return popen(command, creationflags=creationflags)
    except Exception as exc:
        write_isolated_3d_failure(
            output / "3d",
            two_d_output=output,
            exc=exc,
            stage="process_spawn",
        )
        raise


__all__ = ["spawn_post_capture_3d", "write_isolated_3d_failure"]
