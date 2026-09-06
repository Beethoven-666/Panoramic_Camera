"""SDK capture lifecycle, durable job control and recoverable 2-D finalization."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import threading
import uuid
from typing import Any

from .camera_lease import CameraLease
from .commit_journal import file_sha256, read_durable_prefix
from .disk_guard import DiskGuard
from .paths import runtime_source_commit
from .sdk_state import (CaptureError, CompletionState, JobRecord, JobState, StopReason,
                        atomic_json)


def _transition(record: JobRecord, state: JobState, **fields: Any) -> None:
    if record.state == state:
        if fields:
            record.update(**fields)
    else:
        record.transition(state, **fields)


def _finish_result(sdk: Any, record: JobRecord, result: Any, *, owns_session: bool,
                   cancel_event: threading.Event | None = None,
                   post_3d: bool | None = None) -> Any:
    post_3d = sdk.config.post_3d_enabled if post_3d is None else post_3d
    warnings = list(record.payload.get("warnings", []))
    if result.manual_review_required and "manual_review_required" not in warnings:
        warnings.append("manual_review_required")
    if result.formal_2d_published:
        _transition(record, JobState.PUBLISHED_2D)
    result = replace(result, job_id=record.root.name)
    record.update(result={"session": str(result.session), "output_dir": str(result.output_dir)},
                  completion_state=result.completion_state.value, warnings=warnings)
    three_d_succeeded = False
    if post_3d and result.formal_2d_published:
        _transition(record, JobState.PROCESSING_3D)
        try:
            sdk.run_post_3d(result, cancel_event=cancel_event)
            three_d_succeeded = True
        except Exception as exc:
            warnings.append(f"three_d_failed: {exc}")
            record.update(warnings=warnings)
    if owns_session:
        from .sdk_retention import cleanup_owned_session

        paths = [result.panorama_png, result.panorama_jpg, result.provenance_path,
                 result.delivery_path, result.report_path, result.timing_path]
        if three_d_succeeded:
            paths.extend(path for path in (result.output_dir / "3d").iterdir() if path.is_file())
        outputs = {str(path): file_sha256(path) for path in paths if path is not None and path.is_file()}
        try:
            cleanup_owned_session(record, result.session, policy=sdk.config.retention_policy,
                                  formal_2d_published=result.formal_2d_published,
                                  three_d_requested=post_3d,
                                  three_d_succeeded=three_d_succeeded, output_sha256=outputs)
        except OSError as exc:
            # Cleanup can be resumed; a valid delivery remains valid.
            warnings.append(f"cleanup_pending: {exc}")
            record.update(warnings=warnings)
    target = JobState.COMPLETED_WITH_WARNINGS if warnings or not result.formal_2d_published else JobState.COMPLETED
    _transition(record, target)
    return result


def finalize_prefix(sdk: Any, record: JobRecord, session: Path, output: Path,
                    *, original_error: BaseException | None = None) -> Any:
    from .emergency_2d import publish_emergency_2d
    from .video_recovery import salvage_prefix
    from .video_sdk import VideoPanoramaResult

    if record.state not in {JobState.STOPPING, JobState.FINALIZING_2D}:
        _transition(record, JobState.STOPPING)
    _transition(record, JobState.FINALIZING_2D)
    rows, _ = read_durable_prefix(session)
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    eligible = (manifest.get("clean_shutdown") is True
                and manifest.get("product_eligibility", {}).get("video_panorama") is True)
    if not eligible:
        destination = record.root / "recovered" / uuid.uuid4().hex
        session, rows, recovery = salvage_prefix(session, destination)
        eligible = recovery["formal_eligible"]
        record.update(warnings=[*record.payload.get("warnings", []), "committed_prefix_salvage"])
    if original_error is not None:
        record.update(warnings=[*record.payload.get("warnings", []), str(original_error)])
    if eligible:
        try:
            return sdk._process_session_2d(session, output)
        except Exception as exc:
            atomic_json(record.root / "failures" / f"formal-2d-{uuid.uuid4().hex}.json", {
                "error_code": "PANORAMA_PROCESSING_ERROR", "message": str(exc),
                "cause_type": type(exc).__name__, "session": str(session)})
    report = publish_emergency_2d(session, rows, output,
                                  reason="insufficient_formal_input" if original_error is None else str(original_error))
    record.update(warnings=[*record.payload.get("warnings", []), report["reason"]])
    return VideoPanoramaResult.load(session, output)


def start_capture_job(sdk: Any, output_root: Path, *, capture_root: Path | None = None,
                      explicit_output: Path | None = None, width: int = 848, height: int = 480,
                      fps: int = 60, duration_seconds: float = 3, video_exposure_us: int | None = 100,
                      preview_window: bool = False, wait_for_camera: bool = True,
                      maximum_post_seconds: float = 60, max_frames: int = 0) -> Any:
    from .video_sdk import VideoPanoramaResult, VideoProcessingJob

    job_id = uuid.uuid4().hex
    job_root = output_root.resolve() / "jobs" / job_id
    owns_session = capture_root is None
    capture = job_root / "session" if capture_root is None else capture_root.resolve()
    output = job_root / "2d" if explicit_output is None else explicit_output.resolve()
    guard = DiskGuard(capture, output, reserve_gib=sdk.config.disk_reserve_gib,
                      minimum_free_capture_seconds=sdk.config.minimum_free_capture_seconds,
                      post_3d_bytes=2 * 1024**3 if sdk.config.post_3d_enabled else 0)
    guard.preflight()
    # Discovery itself can open the device in the Orbbec wrapper. Serialize it
    # before touching that wrapper, then also hold the actual serial lease.
    discovery = CameraLease(sdk.config.camera_lock_root, "discovery", job_id).acquire()
    try:
        record = JobRecord.create(job_root, source_commit=runtime_source_commit(),
            config_sha="3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc",
            owns_session=owns_session)
        atomic_json(sdk.config.state_root / f"{job_id}.json", {"job_root": str(job_root)})
    except BaseException:
        discovery.release()
        raise

    def task(cancel_event: threading.Event) -> Any:
        from .capture_orbbec import CaptureCancelledError, _VideoCapturePreflightError
        from .video_live import build_parser, run

        serial_lease = None
        session = None
        site, temporary = sdk._site_config()

        def progress(state: str, **detail: Any) -> None:
            nonlocal serial_lease, session
            if "session_root" in detail:
                session = Path(detail.pop("session_root"))
                detail["session"] = str(session)
            serial = detail.get("camera_serial")
            if serial and serial_lease is None:
                serial_lease = CameraLease(sdk.config.camera_lock_root, str(serial), job_id).acquire()
            stop = detail.pop("stop_reason", None)
            if stop is not None:
                try:
                    record.request_stop(StopReason(stop))
                except ValueError:
                    record.update(warnings=[*record.payload.get("warnings", []), stop])
            _transition(record, JobState(state), **detail)

        try:
            arguments = ["--config", str(site), "--output", str(capture), "--panorama-output", str(output),
                         "--width", str(width), "--height", str(height), "--fps", str(fps),
                         "--duration", str(float(duration_seconds)), "--max-frames", str(max_frames),
                         "--maximum-post-seconds", str(maximum_post_seconds)]
            if video_exposure_us is not None:
                arguments += ["--video-exposure-us", str(video_exposure_us)]
            if not preview_window:
                arguments.append("--no-preview")
            if not wait_for_camera:
                arguments.append("--no-wait-for-camera")
            args = build_parser().parse_args(arguments)
            args.cancel_event, args.disk_guard, args.capture_progress = cancel_event, guard, progress
            args.sdk_preview_enabled = sdk.config.preview_enabled
            args.preview_output = record.root / "preview"
            record.transition(JobState.WAITING_FOR_CAMERA)
            from .sdk import _cuda_policy

            with _cuda_policy(sdk.config.cuda_mode):
                for attempt in range(3):
                    try:
                        published = run(args)
                        session = Path(published["session"])
                        result = VideoPanoramaResult.load(session, output)
                        break
                    except _VideoCapturePreflightError as exc:
                        atomic_json(record.root / "failures" / f"preflight-{attempt}.json",
                                    {"message": str(exc), "cause_type": type(exc).__name__})
                        # No reconnect is permitted once any official frame was
                        # committed, regardless of the low-level exception type.
                        if session is not None and (session / "checkpoints/current_commit.json").exists():
                            result = finalize_prefix(sdk, record, session, output, original_error=exc)
                            break
                        if attempt == 2 or not wait_for_camera:
                            raise CaptureError(str(exc), error_code="CAMERA_PREFLIGHT_FAILED", cause=exc) from exc
                        if cancel_event.wait(1):
                            raise CaptureCancelledError(f"Camera wait cancelled after: {exc}") from exc
                        _transition(record, JobState.WAITING_FOR_CAMERA)
                    except Exception as exc:
                        if session is not None and (session / "checkpoints/current_commit.json").exists():
                            result = finalize_prefix(sdk, record, session, output, original_error=exc)
                            break
                        if cancel_event.is_set():
                            if record.state not in {JobState.WAITING_FOR_CAMERA, JobState.WARMING_UP, JobState.STOPPING}:
                                _transition(record, JobState.STOPPING)
                            _transition(record, JobState.CANCELLED_NO_DATA,
                                        completion_state=CompletionState.CANCELLED_NO_DATA.value)
                        raise
            if record.state != JobState.FINALIZING_2D:
                if record.state != JobState.STOPPING:
                    _transition(record, JobState.STOPPING)
                _transition(record, JobState.FINALIZING_2D)
            if record.payload.get("stop_reason") == StopReason.LOW_DISK.value:
                record.update(warnings=[*record.payload.get("warnings", []), "LOW_DISK"])
            # Both leases remain held until the camera and writer are closed;
            # post-3D is independent and may only begin after this release.
            if serial_lease is not None:
                serial_lease.release()
            discovery.release()
            return _finish_result(sdk, record, result, owns_session=owns_session, cancel_event=cancel_event)
        finally:
            if serial_lease is not None:
                serial_lease.release()
            discovery.release()
            temporary.cleanup()

    job = VideoProcessingJob(task, record=record)
    sdk._jobs[job_id] = job
    return job
