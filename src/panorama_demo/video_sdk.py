"""Public in-process SDK facade for the locked continuous-video product."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
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

    def __post_init__(self) -> None:
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
    panorama_jpg: Path
    provenance_path: Path
    report_path: Path
    timing_path: Path
    delivery_path: Path
    authority: str
    delivery_state: str
    overall_grade: str
    manual_review_required: bool
    primary_post_capture_seconds: float | None
    failure_path: Path | None = None

    @property
    def is_published(self) -> bool:
        return self.delivery_state in {"published", "published_degraded"}

    @classmethod
    def load(cls, session: str | Path, output_dir: str | Path) -> "VideoPanoramaResult":
        session_root = _path_argument(session, name="session")
        output = _path_argument(output_dir, name="output_dir")
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
        elapsed = delivery.get("primary_post_capture_seconds")
        if elapsed is None:
            final_2d = timing.get("final_2d")
            if isinstance(final_2d, Mapping):
                elapsed = final_2d.get("capture_stop_to_p3_published_seconds")
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


class VideoJobState(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class VideoProcessingJob:
    """One in-process live capture task with cooperative cancellation."""

    def __init__(self, target: Any) -> None:
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
            finally:
                self._done.set()

        self._thread = threading.Thread(target=invoke, name="g305-video-sdk", daemon=False)
        self._thread.start()

    @property
    def state(self) -> VideoJobState:
        return self._state

    def cancel(self) -> bool:
        if self._done.is_set():
            return False
        self._cancel_event.set()
        return True

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def result(self, timeout: float | None = None) -> VideoPanoramaResult:
        if not self._done.wait(timeout):
            raise TimeoutError("Video processing job is still running")
        if self._error is not None:
            if isinstance(self._error, PanoramaSDKError):
                raise self._error
            raise PanoramaProcessingError(f"Video processing job failed: {self._error}") from self._error
        assert self._result is not None
        return self._result


class Gemini305VideoSDK:
    """Stable facade for locked S013 V11 2-D and isolated post-capture 3-D."""

    def __init__(self, config: VideoSDKConfig | None = None) -> None:
        if config is not None and not isinstance(config, VideoSDKConfig):
            raise SDKConfigurationError("config must be a VideoSDKConfig or None")
        self._config = config or VideoSDKConfig()

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
        self,
        session: str | Path,
        output_dir: str | Path,
        *,
        defer_3d: bool = True,
        maximum_post_seconds: float = 60.0,
    ) -> VideoPanoramaResult:
        input_path = _path_argument(session, name="session")
        output = _path_argument(output_dir, name="output_dir")
        if type(defer_3d) is not bool or maximum_post_seconds <= 0:
            raise SDKInputError("defer_3d must be boolean and maximum_post_seconds positive")
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
                    defer_3d=defer_3d,
                )
            return VideoPanoramaResult.load(input_path, output)
        except PanoramaSDKError:
            raise
        except Exception as exc:
            raise PanoramaProcessingError(f"Video 2-D processing failed: {exc}") from exc
        finally:
            temporary.cleanup()

    def capture_and_process(
        self,
        capture_root: str | Path,
        output_dir: str | Path,
        *,
        width: int = 848,
        height: int = 480,
        fps: int = 60,
        duration_seconds: float = 3.0,
        video_exposure_us: int | None = 100,
        preview_window: bool = False,
        wait_for_camera: bool = True,
        maximum_post_seconds: float = 60.0,
    ) -> VideoProcessingJob:
        capture = _path_argument(capture_root, name="capture_root")
        output = _path_argument(output_dir, name="output_dir")
        if min(width, height, fps) <= 0 or duration_seconds <= 0 or maximum_post_seconds <= 0:
            raise SDKInputError("capture dimensions, fps, duration and post budget must be positive")
        if video_exposure_us is not None and video_exposure_us <= 0:
            raise SDKInputError("video_exposure_us must be positive or None")
        if type(preview_window) is not bool or type(wait_for_camera) is not bool:
            raise SDKInputError("preview_window and wait_for_camera must be booleans")

        def task(cancel_event: threading.Event) -> VideoPanoramaResult:
            from .video_live import build_parser, run

            site, temporary = self._site_config()
            try:
                arguments = [
                    "--config", str(site), "--output", str(capture),
                    "--panorama-output", str(output), "--width", str(width),
                    "--height", str(height), "--fps", str(fps),
                    "--duration", str(float(duration_seconds)),
                    "--maximum-post-seconds", str(float(maximum_post_seconds)),
                ]
                if video_exposure_us is not None:
                    arguments.extend(["--video-exposure-us", str(video_exposure_us)])
                if not preview_window:
                    arguments.append("--no-preview")
                if not wait_for_camera:
                    arguments.append("--no-wait-for-camera")
                args = build_parser().parse_args(arguments)
                args.cancel_event = cancel_event
                with _cuda_policy(self._config.cuda_mode):
                    published = run(args)
                return VideoPanoramaResult.load(str(published["session"]), output)
            finally:
                temporary.cleanup()

        return VideoProcessingJob(task)

    def run_post_3d(
        self,
        session: str | Path,
        two_d_output: str | Path,
        *,
        output_dir: str | Path | None = None,
    ) -> VideoThreeDResult:
        session_root = _path_argument(session, name="session")
        two_d = _path_argument(two_d_output, name="two_d_output")
        output = two_d / "3d" if output_dir is None else _path_argument(output_dir, name="output_dir")
        site, temporary = self._site_config()
        try:
            from .config import load_config
            from .video_3d_postprocess import run_post_capture_3d

            run_post_capture_3d(
                session_path=session_root,
                two_d_output=two_d,
                output_3d=output,
                config=load_config(site),
            )
            return VideoThreeDResult.load(session_root, two_d, output)
        except PanoramaSDKError:
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
