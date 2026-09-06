"""Public in-process SDK facade for the locked continuous-video product."""

from __future__ import annotations

import sys
import json
import time
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .sdk import (
    CudaMode,
    PanoramaProcessingError,
    PanoramaSDKError,
    SDKConfigurationError,
    SDKInputError,
    _cuda_policy,
    _path_argument,
    _read_json_object,
)
from .sdk_doctor import SDKDoctorReport, run_sdk_doctor
from .version import __version__
from .sdk_state import (CompletionState, JobRecord, JobState, RetentionPolicy, SDKBusyError,
                        SDKError, StopReason, TERMINAL_STATES)


def _default_orb_root() -> Path:
    if sys.platform.startswith("linux"):
        return Path.home() / "opt" / "g305-orbslam3"
    return Path.home() / "Projects" / "ORB_SLAM3_WS" / "ORB_SLAM3"


@dataclass(frozen=True)
class VideoSDKConfig:
    """Machine/runtime settings; the locked V11 algorithm is not configurable."""

    config_path: Path | str | None = None
    cuda_mode: CudaMode | str = CudaMode.REQUIRED
    orbslam3_root: Path | str | None = None
    orbslam3_executable: str = "Examples/RGB-D/rgbd_tum_headless"
    orbslam3_stream_executable: str = "Examples/RGB-D/rgbd_g305_stream_headless"
    orbslam3_vocabulary: str = "Vocabulary/ORBvoc.txt"
    orb_runtime_kind: str | None = None
    runtime_profile: str = "cuda_required"
    wait_for_camera: bool = True
    preview_enabled: bool = True
    post_3d_enabled: bool = False
    disk_reserve_gib: float = 10
    minimum_free_capture_seconds: float = 120
    retention_policy: RetentionPolicy | str = RetentionPolicy.DELETE_AFTER_ALL_SUCCESS
    camera_lock_root: Path | str = "/var/lock/gemini305-sdk"
    state_root: Path | str = "~/.local/state/gemini305-sdk"
    three_d_timeout_seconds: float = 600

    def __post_init__(self) -> None:
        if self.runtime_profile != "cuda_required":
            raise SDKConfigurationError("Only the cuda_required production runtime profile is supported")
        if self.disk_reserve_gib <= 0 or self.minimum_free_capture_seconds <= 0 or self.three_d_timeout_seconds <= 0:
            raise SDKConfigurationError("Disk reserve, capture runway and watchdog must be positive")
        for name in ("wait_for_camera", "preview_enabled", "post_3d_enabled"):
            if type(getattr(self, name)) is not bool:
                raise SDKConfigurationError(f"{name} must be boolean")
        try:
            object.__setattr__(self, "retention_policy", RetentionPolicy(self.retention_policy))
        except ValueError as exc:
            raise SDKConfigurationError("Unsupported retention policy") from exc
        for name in ("camera_lock_root", "state_root"):
            object.__setattr__(self, name, _path_argument(getattr(self, name), name=name))
        if self.config_path is not None:
            path = _path_argument(self.config_path, name="config_path")
            if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
                raise SDKConfigurationError("config_path must be an existing YAML file")
            object.__setattr__(self, "config_path", path)
        try:
            mode = CudaMode(self.cuda_mode)
        except (TypeError, ValueError) as exc:
            raise SDKConfigurationError("cuda_mode is unsupported") from exc
        object.__setattr__(self, "cuda_mode", mode)
        root = (
            _default_orb_root()
            if self.orbslam3_root is None
            else _path_argument(self.orbslam3_root, name="orbslam3_root")
        )
        object.__setattr__(self, "orbslam3_root", root)
        runtime_kind = self.orb_runtime_kind or (
            "native_linux" if sys.platform.startswith("linux") else "windows_wsl_legacy"
        )
        if runtime_kind not in {"native_linux", "windows_wsl_legacy"}:
            raise SDKConfigurationError("orb_runtime_kind is unsupported")
        object.__setattr__(self, "orb_runtime_kind", runtime_kind)
        for name in (
            "orbslam3_executable",
            "orbslam3_stream_executable",
            "orbslam3_vocabulary",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
                raise SDKConfigurationError(f"{name} must be a non-empty relative path")


@dataclass(frozen=True)
class VideoPanoramaResult:
    session: Path
    output_dir: Path
    panorama_png: Path
    panorama_jpg: Path | None
    provenance_path: Path
    report_path: Path
    timing_path: Path | None
    delivery_path: Path
    authority: str
    delivery_state: str
    overall_grade: str
    manual_review_required: bool
    primary_post_capture_seconds: float | None
    failure_path: Path | None = None
    formal_2d_published: bool = True
    emergency_2d_path: Path | None = None
    completion_state: CompletionState = CompletionState.FORMAL_2D_PUBLISHED
    job_id: str | None = None

    @property
    def is_published(self) -> bool:
        return self.delivery_state in {"published", "published_degraded"}

    @classmethod
    def load(cls, session: str | Path, output_dir: str | Path) -> "VideoPanoramaResult":
        session_root = _path_argument(session, name="session")
        output = _path_argument(output_dir, name="output_dir")
        if not (output / "video_delivery.json").exists() and (output / "emergency_delivery.json").exists():
            from .commit_journal import file_sha256

            payload = _read_json_object(output / "emergency_delivery.json", label="emergency delivery")
            image = output / "emergency_panorama.png"
            if (payload.get("schema") != "gemini305-sdk-emergency-2d/v1"
                    or payload.get("formal_2d_published") is not False
                    or not image.is_file() or file_sha256(image) != payload.get("png_sha256")):
                raise PanoramaProcessingError("Emergency result is incomplete or changed")
            return cls(session=session_root, output_dir=output, panorama_png=image,
                       panorama_jpg=None, provenance_path=output / "emergency_provenance.npz",
                       report_path=output / "emergency_report.json", timing_path=None,
                       delivery_path=output / "emergency_delivery.json",
                       authority="committed_rgb_emergency_only", delivery_state="emergency_published",
                       overall_grade="NE", manual_review_required=True, primary_post_capture_seconds=None,
                       formal_2d_published=False, emergency_2d_path=image,
                       completion_state=CompletionState.EMERGENCY_2D_PUBLISHED)
        failure = output / "video_failure.json"
        delivery_path = output / "video_delivery.json"
        required = {
            "panorama_png": output / "video_panorama.png",
            "panorama_jpg": output / "video_panorama.jpg",
            "provenance_path": output / "video_pixel_provenance.npz",
            "report_path": output / "video_report.json",
            "timing_path": output / "video_timing.json",
            "delivery_path": delivery_path,
        }
        missing = [path.name for path in required.values() if not path.is_file()]
        if missing:
            suffix = f"; failure={failure}" if failure.is_file() else ""
            raise PanoramaProcessingError(
                f"Video 2-D delivery is incomplete: {', '.join(missing)}{suffix}"
            )
        delivery = _read_json_object(delivery_path, label="video_delivery.json")
        report = _read_json_object(required["report_path"], label="video_report.json")
        timing = _read_json_object(required["timing_path"], label="video_timing.json")
        trajectory = report.get("trajectory")
        grades = report.get("grades")
        if not isinstance(trajectory, Mapping) or trajectory.get("source") != "ignore_pose":
            raise PanoramaProcessingError("Video 2-D report lacks ignore-pose authority")
        if not isinstance(grades, Mapping):
            raise PanoramaProcessingError("Video 2-D report lacks grades")
        state = delivery.get("delivery_state")
        grade = grades.get("overall")
        manual = delivery.get("manual_review_required")
        if state not in {"published", "published_degraded"} or grade not in {"A", "B", "C", "NE"}:
            raise PanoramaProcessingError("Video 2-D delivery state or grade is invalid")
        if type(manual) is not bool:
            raise PanoramaProcessingError("Video 2-D manual review flag is invalid")
        final_2d = timing.get("final_2d")
        elapsed = (final_2d.get("capture_stop_to_p3_published_seconds")
                   if isinstance(final_2d, Mapping) else None)
        if elapsed is not None and (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))):
            raise PanoramaProcessingError("Video 2-D timing is invalid")
        return cls(
            session=session_root,
            output_dir=output,
            authority="ignore_pose_s013_v11_production",
            delivery_state=str(state),
            overall_grade=str(grade),
            manual_review_required=manual,
            primary_post_capture_seconds=None if elapsed is None else float(elapsed),
            failure_path=failure if failure.is_file() else None,
            **required,
        )


@dataclass(frozen=True)
class VideoThreeDResult:
    session: Path
    two_d_output: Path
    output_dir: Path
    delivery_path: Path
    timing_path: Path
    desktop_glb: Path
    mobile_glb: Path
    viewer_path: Path
    delivery_state: str
    failure_path: Path | None = None

    @classmethod
    def load(cls, session: str | Path, two_d_output: str | Path, output_dir: str | Path) -> "VideoThreeDResult":
        session_root = _path_argument(session, name="session")
        two_d = _path_argument(two_d_output, name="two_d_output")
        output = _path_argument(output_dir, name="output_dir")
        failure = output / "video_3d_failure.json"
        delivery = output / "video_3d_delivery.json"
        timing = output / "video_3d_timing.json"
        desktop = output / "video_tsdf_mesh.glb"
        mobile = output / "video_tsdf_mesh_mobile.glb"
        viewer = output / "video_tsdf_mesh_viewer.html"
        missing = [p.name for p in (delivery, timing, desktop, mobile, viewer) if not p.is_file()]
        if missing:
            raise PanoramaProcessingError(
                f"Video 3-D delivery is incomplete: {', '.join(missing)}"
                + (f"; failure={failure}" if failure.is_file() else "")
            )
        payload = _read_json_object(delivery, label="video_3d_delivery.json")
        state = payload.get("delivery_state", "published")
        if not isinstance(state, str):
            raise PanoramaProcessingError("Video 3-D delivery state is invalid")
        return cls(
            session_root, two_d, output, delivery, timing, desktop, mobile, viewer,
            state, failure if failure.is_file() else None,
        )


VideoJobState = JobState


class VideoProcessingJob:
    """One in-process live capture task with cooperative cancellation."""

    def __init__(self, target: Any, *, record: JobRecord | None = None) -> None:
        self._record = record
        self.job_id = None if record is None else record.root.name
        self._read_only = False
        self._cancel_event = threading.Event()
        self._done = threading.Event()
        self._state = VideoJobState.RUNNING
        self._result: VideoPanoramaResult | None = None
        self._error: BaseException | None = None

        def invoke() -> None:
            try:
                self._result = target(self._cancel_event)
                self._state = VideoJobState.SUCCEEDED
            except BaseException as exc:
                self._error = exc
                self._state = (
                    VideoJobState.CANCELLED if self._cancel_event.is_set() else VideoJobState.FAILED
                )
                if self._record is not None and self._record.state not in TERMINAL_STATES:
                    if self._cancel_event.is_set() and self._record.state in {
                        JobState.CREATED, JobState.WAITING_FOR_CAMERA, JobState.WARMING_UP
                    }:
                        self._record.transition(JobState.CANCELLED_NO_DATA,
                            completion_state=CompletionState.CANCELLED_NO_DATA.value)
                    else:
                        self._record.transition(JobState.FAILED, failure={
                            "error_code": getattr(exc, "error_code", "INTERNAL_ERROR"),
                            "message": str(exc), "cause_type": type(exc).__name__},
                            completion_state=CompletionState.FAILED_NO_VALID_FRAME.value)
            finally:
                self._done.set()

        self._thread = threading.Thread(target=invoke, name="g305-video-sdk", daemon=False)
        self._thread.start()

    @property
    def state(self) -> VideoJobState:
        if self._record is not None:
            if self._read_only:
                self._record = JobRecord.load(self._record.root)
            return self._record.state
        return self._state

    def cancel(self) -> bool:
        if self._read_only:
            raise SDKBusyError("This client has read-only access to the job", error_code="JOB_CONTROL_NOT_OWNED")
        if self._done.is_set():
            return False
        self._cancel_event.set()
        if self._record is not None:
            self._record.request_stop(StopReason.USER_REQUEST)
        return True

    def wait(self, timeout: float | None = None) -> bool:
        if self._read_only:
            started = time.monotonic()
            while self.state not in TERMINAL_STATES:
                if timeout is not None and time.monotonic() - started >= timeout:
                    return False
                time.sleep(0.1)
            return True
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> VideoPanoramaResult:
        if not self.wait(timeout):
            raise TimeoutError("Video processing job is still running")
        if self._read_only:
            result = self._record.payload.get("result")
            if result:
                return VideoPanoramaResult.load(result["session"], result["output_dir"])
            raise PanoramaProcessingError(str(self._record.payload.get("failure", self.state.value)))
        if self._error is not None:
            if isinstance(self._error, SDKError):
                raise self._error
            raise PanoramaProcessingError(f"Video processing job failed: {self._error}") from self._error
        assert self._result is not None
        return self._result

    @classmethod
    def load(cls, root: Path) -> "VideoProcessingJob":
        job = cls.__new__(cls)
        job._record = JobRecord.load(root)
        job.job_id = root.name
        job._read_only = True
        return job


class Gemini305VideoSDK:
    """Stable facade for locked S013 V11 2-D and isolated post-capture 3-D."""

    def __init__(self, config: VideoSDKConfig | None = None) -> None:
        if config is not None and not isinstance(config, VideoSDKConfig):
            raise SDKConfigurationError("config must be a VideoSDKConfig or None")
        self._config = config or VideoSDKConfig()
        self._jobs: dict[str, VideoProcessingJob] = {}

    @property
    def version(self) -> str:
        return __version__

    @property
    def config(self) -> VideoSDKConfig:
        return self._config

    def doctor(self) -> SDKDoctorReport:
        return run_sdk_doctor(
            orbslam3_root=self._config.orbslam3_root,
            orbslam3_executable=self._config.orbslam3_executable,
            orbslam3_vocabulary=self._config.orbslam3_vocabulary,
            orb_runtime_kind=self._config.orb_runtime_kind,
            camera_lock_root=self.config.camera_lock_root,
            probe_addon=not any(job.state not in TERMINAL_STATES for job in self._jobs.values()),
        )

    def _site_config(self) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
        from .config import load_config

        config = load_config(self._config.config_path)
        stitch = config.setdefault("stitch", {})
        if not isinstance(stitch, dict):
            raise SDKConfigurationError("stitch config must be a mapping")
        orb = stitch.setdefault("orbslam3_rgbd", {})
        if not isinstance(orb, dict):
            raise SDKConfigurationError("stitch.orbslam3_rgbd must be a mapping")
        orb.update(
            {
                "root": str(self._config.orbslam3_root),
                "runtime_kind": self._config.orb_runtime_kind,
                "executable": self._config.orbslam3_executable,
                "stream_executable": self._config.orbslam3_stream_executable,
                "vocabulary": self._config.orbslam3_vocabulary,
            }
        )
        temporary = tempfile.TemporaryDirectory(prefix="g305-video-sdk-config-")
        path = Path(temporary.name) / "site.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return path, temporary

    def process_session(
        self, session: str | Path, output_dir: str | Path, *, defer_3d: bool = True,
        maximum_post_seconds: float = 60.0,
    ) -> VideoPanoramaResult:
        import uuid
        from .paths import runtime_source_commit
        from .sdk_state import atomic_json
        from .video_sdk_jobs import _finish_result

        input_path = _path_argument(session, name="session")
        output = _path_argument(output_dir, name="output_dir")
        if type(defer_3d) is not bool or maximum_post_seconds <= 0:
            raise SDKInputError("defer_3d must be boolean and maximum_post_seconds positive")
        if any((output / name).exists() for name in ("video_delivery.json", "emergency_delivery.json")):
            raise SDKInputError("Output already contains a delivery; select a new output directory")
        record = JobRecord.create(output.parent / "jobs" / uuid.uuid4().hex,
            source_commit=runtime_source_commit(),
            config_sha="3a49f8cbaa622592eff41b2a9e34cb518c6067d74d15574cf01faeea0cc8fafc",
            owns_session=False)
        atomic_json(self.config.state_root / f"{record.root.name}.json", {"job_root": str(record.root)})
        record.transition(JobState.FINALIZING_2D, session=str(input_path))
        try:
            result = self._process_session_2d(input_path, output, maximum_post_seconds=maximum_post_seconds)
            return _finish_result(self, record, result, owns_session=False, post_3d=not defer_3d)
        except Exception as exc:
            if record.state not in TERMINAL_STATES:
                record.transition(JobState.FAILED, failure={"error_code": getattr(exc, "error_code", "INTERNAL_ERROR"),
                    "message": str(exc), "cause_type": type(exc).__name__})
            raise

    def _process_session_2d(
        self,
        session: str | Path,
        output_dir: str | Path,
        *,
        maximum_post_seconds: float = 60.0,
    ) -> VideoPanoramaResult:
        input_path = _path_argument(session, name="session")
        output = _path_argument(output_dir, name="output_dir")
        site, temporary = self._site_config()
        try:
            from .video_pipeline import run_video_algorithm

            with _cuda_policy(self._config.cuda_mode):
                run_video_algorithm(
                    input_path=input_path,
                    output=output,
                    role="production",
                    config_path=site,
                    maximum_post_seconds=float(maximum_post_seconds),
                    defer_3d=True,
                )
            result = VideoPanoramaResult.load(input_path, output)
            return result
        except PanoramaSDKError:
            raise
        except Exception as exc:
            raise PanoramaProcessingError(f"Video 2-D processing failed: {exc}") from exc
        finally:
            temporary.cleanup()

    def capture_and_process(
        self,
        capture_root: str | Path,
        output_dir: str | Path | None = None,
        *,
        width: int = 848,
        height: int = 480,
        fps: int = 60,
        duration_seconds: float = 3.0,
        video_exposure_us: int | None = 100,
        preview_window: bool = False,
        wait_for_camera: bool = True,
        maximum_post_seconds: float = 60.0,
        max_frames: int = 0,
    ) -> VideoProcessingJob:
        capture = _path_argument(capture_root, name="capture_root")
        output = None if output_dir is None else _path_argument(output_dir, name="output_dir")
        if min(width, height, fps) <= 0 or duration_seconds < 0 or maximum_post_seconds <= 0 or max_frames < 0:
            raise SDKInputError("capture dimensions, fps, duration and post budget must be positive")
        if video_exposure_us is not None and video_exposure_us <= 0:
            raise SDKInputError("video_exposure_us must be positive or None")
        if type(preview_window) is not bool or type(wait_for_camera) is not bool:
            raise SDKInputError("preview_window and wait_for_camera must be booleans")

        from .video_sdk_jobs import start_capture_job

        return start_capture_job(
            self, capture if output is None else output.parent,
            capture_root=None if output is None else capture, explicit_output=output,
            width=width, height=height, fps=fps, duration_seconds=duration_seconds,
            video_exposure_us=video_exposure_us, preview_window=preview_window,
            wait_for_camera=wait_for_camera, maximum_post_seconds=maximum_post_seconds,
            max_frames=max_frames,
        )

    def start_capture(self, output_root: str | Path, **options: Any) -> VideoProcessingJob:
        options.setdefault("duration_seconds", 0)
        options.setdefault("wait_for_camera", self.config.wait_for_camera)
        options.setdefault("preview_window", False)
        return self.capture_and_process(output_root, **options)

    def get_job(self, job_id: str) -> VideoProcessingJob:
        if job_id in self._jobs:
            return self._jobs[job_id]
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise SDKInputError("Invalid SDK job ID")
        index = self.config.state_root / f"{job_id}.json"
        try:
            root = Path(json.loads(index.read_text(encoding="utf-8"))["job_root"])
            return VideoProcessingJob.load(root)
        except (OSError, ValueError, KeyError) as exc:
            raise SDKInputError("SDK job is unavailable") from exc

    def get_status(self, job_id: str) -> dict[str, Any]:
        job = self.get_job(job_id)
        return dict(JobRecord.load(job._record.root).payload)

    def get_preview(self, job_id: str) -> Path | None:
        record = self.get_job(job_id)._record
        for root in (record.root / "preview", record.root / "2d"):
            state = root / "live_preview_state.json"
            image = root / "live_preview.jpg"
            if state.is_file() and image.is_file():
                if json.loads(state.read_text(encoding="utf-8")).get("authority") == "non_authoritative_live_preview":
                    return image
        return None

    def stop_capture(self, job_id: str) -> bool:
        return self.get_job(job_id).cancel()

    def cancel(self, job_id: str) -> bool:
        return self.get_job(job_id).cancel()

    def release_control(self) -> None:
        for job in tuple(self._jobs.values()):
            if job.state not in TERMINAL_STATES:
                job.cancel()
                job.wait()

    def list_recoverable_jobs(self) -> list[dict[str, Any]]:
        if not self.config.state_root.exists():
            return []
        records = []
        for index in self.config.state_root.glob("*.json"):
            try:
                root = Path(json.loads(index.read_text(encoding="utf-8"))["job_root"])
                record = JobRecord.load(root)
                if record.state not in TERMINAL_STATES or record.payload.get("cleanup", {}).get("state") in {"rename_pending", "delete_pending"}:
                    records.append(dict(record.payload))
            except (OSError, ValueError, KeyError):
                continue
        return records

    def recover_job(self, job_id: str) -> VideoPanoramaResult | None:
        from .camera_lease import CameraLease
        from .sdk_retention import resume_cleanup
        from .video_sdk_jobs import _finish_result, finalize_prefix

        record = self.get_job(job_id)._record
        # Recovery changes shared data, so a read-only observer must acquire
        # control after the old process has released it (including crash exit).
        with CameraLease(self.config.camera_lock_root, "discovery", job_id):
            import psutil

            if record.state not in TERMINAL_STATES and psutil.pid_exists(record.payload["pid"]):
                raise SDKBusyError("The recorded job controller is still running",
                                   error_code="JOB_CONTROL_NOT_OWNED")
            if resume_cleanup(record):
                return None
            if record.state in TERMINAL_STATES:
                raise SDKInputError("This job has no unfinished recoverable capture")
            session = record.payload.get("session")
            if session is None:
                raise SDKInputError("This job has no committed session")
            # Preserve the crashed job's event history; recovery is a new job.
            import uuid
            from .paths import runtime_source_commit
            from .sdk_state import atomic_json

            recovered = JobRecord.create(record.root.parent / uuid.uuid4().hex,
                source_commit=runtime_source_commit(), config_sha=record.payload["production_config_sha256"],
                owns_session=False)
            recovered.transition(JobState.FINALIZING_2D, warnings=["crash_recovery"],
                                 recovered_from_job_id=job_id)
            atomic_json(self.config.state_root / f"{recovered.root.name}.json", {"job_root": str(recovered.root)})
            result = finalize_prefix(self, recovered, Path(session), recovered.root / "2d")
            return _finish_result(self, recovered, result, owns_session=False)

    def run_post_3d(
        self,
        session: str | Path | VideoPanoramaResult,
        two_d_output: str | Path | None = None,
        *,
        output_dir: str | Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> VideoThreeDResult:
        if isinstance(session, VideoPanoramaResult):
            result = session
        else:
            if two_d_output is None:
                raise SDKInputError("two_d_output is required for a session path")
            result = VideoPanoramaResult.load(session, two_d_output)
        if not result.formal_2d_published:
            raise SDKInputError("Emergency 2-D cannot authorize post-3D")
        session_root, two_d = result.session, result.output_dir
        output = two_d / "3d" if output_dir is None else _path_argument(output_dir, name="output_dir")
        site, temporary = self._site_config()
        try:
            from .disk_guard import DiskGuard
            from .video_sdk_3d import supervise_post_3d

            DiskGuard(session_root, output, reserve_gib=self.config.disk_reserve_gib,
                      post_3d_bytes=2 * 1024**3).preflight()
            supervise_post_3d(session=session_root, two_d=two_d, output=output,
                              config_path=site, timeout_seconds=self.config.three_d_timeout_seconds,
                              cancel_event=cancel_event)
            return VideoThreeDResult.load(session_root, two_d, output)
        except SDKError:
            raise
        except Exception as exc:
            raise PanoramaProcessingError(
                f"Post-capture 3-D failed; published 2-D was preserved: {exc}"
            ) from exc
        finally:
            temporary.cleanup()


__all__ = [
    "Gemini305VideoSDK",
    "VideoJobState",
    "VideoPanoramaResult",
    "VideoProcessingJob",
    "VideoSDKConfig",
    "VideoThreeDResult",
]
