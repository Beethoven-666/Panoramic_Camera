"""Supervise post-3D in a separate process without importing a 3D runtime."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from .commit_journal import file_sha256
from .sdk_state import ThreeDProcessingError, atomic_json
from .video_3d_launcher import write_isolated_3d_failure


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    else:
        import psutil

        children = psutil.Process(process.pid).children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
    process.wait(timeout=10)


def supervise_post_3d(*, session: Path, two_d: Path, output: Path,
                      config_path: Path, timeout_seconds: float,
                      cancel_event: threading.Event | None = None) -> None:
    if output.resolve() != (two_d / "3d").resolve():
        raise ThreeDProcessingError("SDK post-3D output must be the 2-D delivery's 3d directory")
    if (output / "video_3d_delivery.json").exists():
        raise ThreeDProcessingError("A 3-D delivery already exists; refusing to overwrite it")
    names = ("video_delivery.json", "video_panorama.png", "video_pixel_provenance.npz")
    before = {name: file_sha256(two_d / name) for name in names}
    output.mkdir(parents=True, exist_ok=True)
    process = None
    stage = "process_spawn"
    try:
        with (output / f"post-3d-{uuid.uuid4().hex}.log").open("xb") as log:
            # The synchronous 2-D API has returned, releasing its renderer
            # resources. Persist this boundary without changing its delivery.
            released_ns = time.monotonic_ns()
            process = subprocess.Popen([
                sys.executable, "-m", "panorama_demo.video_3d_postprocess", str(session),
                "--two-d-output", str(two_d), "--output", str(output),
                "--config", str(config_path)], stdout=log, stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
            atomic_json(output / "process_boundary.json", {
                "parent_pid": os.getpid(), "child_pid": process.pid,
                "two_d_resources_released_monotonic_ns": released_ns,
                "spawned_monotonic_ns": time.monotonic_ns(), "two_d_sha256": before})
            deadline = time.monotonic() + timeout_seconds
            stage = "process_watchdog"
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    raise ThreeDProcessingError("Post-3D cancelled", error_code="THREE_D_CANCELLED")
                if time.monotonic() >= deadline:
                    raise ThreeDProcessingError("Post-3D watchdog expired", error_code="THREE_D_TIMEOUT")
                try:
                    process.wait(timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode:
                raise ThreeDProcessingError(f"Post-3D child exited {process.returncode}")
    except BaseException as exc:
        if process is not None:
            _terminate_tree(process)
        error = exc if isinstance(exc, Exception) else RuntimeError(str(exc))
        write_isolated_3d_failure(output, two_d_output=two_d, exc=error, stage=stage)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ThreeDProcessingError(
            f"Post-capture 3-D failed; published 2-D was preserved: {exc}",
            error_code=getattr(exc, "error_code", "THREE_D_PROCESSING_ERROR"),
            stage=stage, artifact_path=output / "video_3d_failure.json", cause=error) from exc
    finally:
        if any(file_sha256(two_d / name) != sha for name, sha in before.items()):
            raise ThreeDProcessingError("Post-3D changed a published 2-D artifact",
                                        error_code="TWO_D_ISOLATION_VIOLATION")
