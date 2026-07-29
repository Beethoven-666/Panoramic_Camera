"""Stable, small public SDK for the Gemini 305 RGB-D panorama pipeline.

The SDK intentionally exposes only the supported session-validation, demo and
formal-delivery operations.  It does not expose renderer, pose or seam tuning
knobs that would weaken the project's fail-closed delivery contract.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Iterator, Mapping

from .version import __version__


class PanoramaSDKError(RuntimeError):
    """Base exception raised by the public SDK."""


class SDKConfigurationError(PanoramaSDKError, ValueError):
    """The SDK initialization configuration is invalid."""


class SDKInputError(PanoramaSDKError, ValueError):
    """A session, output path or other SDK method argument is invalid."""


class PanoramaProcessingError(PanoramaSDKError):
    """The underlying fail-closed pipeline rejected or could not process a task."""


class CudaMode(str, Enum):
    """CUDA execution policy used for one SDK operation."""

    PREFER = "prefer"
    AUTO = "auto"
    OFF = "off"
    REQUIRED = "required"


_CUDA_OPERATION_LOCK = RLock()


def _path_argument(value: str | Path, *, name: str) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, Path)):
        raise SDKInputError(f"{name} must be a non-empty path string or Path")
    if not str(value).strip():
        raise SDKInputError(f"{name} must not be empty")
    return Path(value).expanduser().resolve()


@dataclass(frozen=True)
class SDKConfig:
    """Configuration accepted when creating :class:`PanoramaSDK`.

    ``config_path`` is an optional YAML overlay merged with ``configs/demo.yaml``
    by the existing formal pipeline.  The overlay may only use values accepted
    by that pipeline's safety validation.
    """

    config_path: Path | str | None = None
    cuda_mode: CudaMode | str = CudaMode.PREFER
    diagnostic_force: bool = False

    def __post_init__(self) -> None:
        if self.config_path is not None:
            path = _path_argument(self.config_path, name="config_path")
            if not path.is_file():
                raise SDKConfigurationError(f"config_path does not exist: {path}")
            if path.suffix.lower() not in {".yaml", ".yml"}:
                raise SDKConfigurationError("config_path must be a YAML file")
            object.__setattr__(self, "config_path", path)
        try:
            mode = CudaMode(self.cuda_mode)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in CudaMode)
            raise SDKConfigurationError(
                f"cuda_mode must be one of: {choices}"
            ) from exc
        object.__setattr__(self, "cuda_mode", mode)
        if type(self.diagnostic_force) is not bool:
            raise SDKConfigurationError("diagnostic_force must be a boolean")


@dataclass(frozen=True)
class SessionSummary:
    """Validated, non-image summary of one strict RGB-D session."""

    root: Path
    frame_count: int
    frame_width: int
    frame_height: int
    depth_alignment: str


@dataclass(frozen=True)
class PanoramaResult:
    """Paths and publication state produced by one SDK panorama operation."""

    output_dir: Path
    panorama_path: Path
    report_path: Path
    delivery_path: Path | None
    delivery_state: str | None
    quality_grade: str | None
    strict_quality_pass: bool | None
    manual_review_required: bool | None
    diagnostic_only: bool

    @property
    def is_published(self) -> bool:
        """Whether a formal ``delivery.json`` was atomically published."""

        return self.delivery_state in {"published", "published_degraded"}

    @classmethod
    def load(cls, output_dir: str | Path) -> "PanoramaResult":
        """Load and validate the public result summary from an output directory."""

        output = _path_argument(output_dir, name="output_dir")
        if not output.is_dir():
            raise SDKInputError(f"output_dir does not exist: {output}")
        delivery_path = output / "delivery.json"
        if delivery_path.is_file():
            payload = _read_json_object(delivery_path, label="delivery.json")
            panorama = output / "panorama.jpg"
            report = output / "report.json"
            if not panorama.is_file() or not report.is_file():
                raise PanoramaProcessingError(
                    "Formal delivery is incomplete: panorama.jpg and report.json are required"
                )
            return cls(
                output_dir=output,
                panorama_path=panorama,
                report_path=report,
                delivery_path=delivery_path,
                delivery_state=_required_string(payload, "delivery_state"),
                quality_grade=_required_string(payload, "quality_grade"),
                strict_quality_pass=_required_bool(payload, "strict_quality_pass"),
                manual_review_required=_required_bool(
                    payload, "manual_review_required"
                ),
                diagnostic_only=False,
            )
        panorama = output / "diagnostic_panorama.jpg"
        report = output / "diagnostic_report.json"
        if panorama.is_file() and report.is_file():
            return cls(
                output_dir=output,
                panorama_path=panorama,
                report_path=report,
                delivery_path=None,
                delivery_state=None,
                quality_grade=None,
                strict_quality_pass=None,
                manual_review_required=None,
                diagnostic_only=True,
            )
        failure = output / "failure.json"
        detail = f" Failure report: {failure}" if failure.is_file() else ""
        raise PanoramaProcessingError(
            "Pipeline did not publish a formal or diagnostic result." + detail
        )


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PanoramaProcessingError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise PanoramaProcessingError(f"{label} must contain a JSON object")
    return payload


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PanoramaProcessingError(f"delivery.json has invalid {key}")
    return value


def _required_bool(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if type(value) is not bool:
        raise PanoramaProcessingError(f"delivery.json has invalid {key}")
    return value


@contextmanager
def _cuda_policy(mode: CudaMode) -> Iterator[None]:
    """Temporarily select one process-wide CUDA policy and restore it safely."""

    with _CUDA_OPERATION_LOCK:
        previous = os.environ.get("G305_CUDA")
        os.environ["G305_CUDA"] = mode.value
        try:
            from . import cuda_backend

            cuda_backend.cuda_status(refresh=True)
            yield
        finally:
            if previous is None:
                os.environ.pop("G305_CUDA", None)
            else:
                os.environ["G305_CUDA"] = previous
            cuda_backend.cuda_status(refresh=True)


class PanoramaSDK:
    """Reusable entry point for strict Gemini 305 RGB-D panorama operations."""

    def __init__(self, config: SDKConfig | None = None) -> None:
        if config is not None and not isinstance(config, SDKConfig):
            raise SDKConfigurationError("config must be an SDKConfig or None")
        self._config = config or SDKConfig()

    @property
    def config(self) -> SDKConfig:
        """The immutable configuration selected for this SDK client."""

        return self._config

    @property
    def version(self) -> str:
        """SDK semantic version."""

        return __version__

    def acceleration_status(self) -> Mapping[str, object]:
        """Return audited CUDA availability for this client's selected mode."""

        try:
            with _cuda_policy(self._config.cuda_mode):
                from .cuda_backend import cuda_metadata

                return cuda_metadata()
        except Exception as exc:
            raise PanoramaProcessingError(
                f"Could not initialize CUDA policy {self._config.cuda_mode.value}: {exc}"
            ) from exc

    def validate_session(self, session: str | Path) -> SessionSummary:
        """Validate a strict RGB-D session before a long-running build."""

        input_path = _path_argument(session, name="session")
        if not input_path.exists():
            raise SDKInputError(f"session does not exist: {input_path}")
        try:
            from .session import load_rgbd_session

            loaded = load_rgbd_session(input_path)
        except Exception as exc:
            raise SDKInputError(f"Invalid RGB-D session: {input_path}: {exc}") from exc
        return SessionSummary(
            root=loaded.root,
            frame_count=len(loaded.frames),
            frame_width=loaded.calibration.width,
            frame_height=loaded.calibration.height,
            depth_alignment=loaded.depth_alignment,
        )

    def capture(
        self,
        output_dir: str | Path,
        *,
        duration_seconds: float | None = None,
        max_frames: int | None = None,
        photo_mode: bool = True,
        preview: bool = False,
    ) -> Path:
        """Capture one synchronized Gemini 305 RGB-D session.

        The SDK defaults to the no-preview, one-trigger-per-frame photo-mode
        state machine. Device-specific exposure, alignment and synchronization
        policy remains in the validated YAML configuration. The returned
        directory is ready for :meth:`validate_session` and :meth:`build`
        after a clean shutdown.
        """

        output = _path_argument(output_dir, name="output_dir")
        if output.exists() and not output.is_dir():
            raise SDKInputError(f"output_dir must be a directory path: {output}")
        if duration_seconds is not None:
            if (
                isinstance(duration_seconds, bool)
                or not isinstance(duration_seconds, (int, float))
                or float(duration_seconds) <= 0.0
            ):
                raise SDKInputError("duration_seconds must be a positive number or None")
        if max_frames is not None:
            if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames <= 0:
                raise SDKInputError("max_frames must be a positive integer or None")
        if type(photo_mode) is not bool or type(preview) is not bool:
            raise SDKInputError("photo_mode and preview must be booleans")
        if photo_mode and preview:
            raise SDKInputError("photo_mode does not support preview")
        args = argparse.Namespace(
            config=self._config.config_path,
            output=output,
            photo_mode=photo_mode,
            width=None,
            height=None,
            fps=None,
            warmup_frames=None,
            queue_size=None,
            auto_exposure=False,
            diagnostic_unrestricted_auto_exposure=False,
            exposure_us=None,
            gain=None,
            white_balance=None,
            duration=(float(duration_seconds) if duration_seconds is not None else None),
            max_frames=max_frames,
            no_preview=not preview,
            raw_depth=None,
        )
        try:
            from .capture_orbbec import run_capture

            return run_capture(args)
        except Exception as exc:
            raise PanoramaProcessingError(f"Gemini 305 capture failed: {exc}") from exc

    def build(self, session: str | Path, output_dir: str | Path) -> PanoramaResult:
        """Run the formal fail-closed panorama pipeline for one RGB-D session."""

        input_path = _path_argument(session, name="session")
        output = _path_argument(output_dir, name="output_dir")
        if output.exists() and not output.is_dir():
            raise SDKInputError(f"output_dir must be a directory path: {output}")
        self.validate_session(input_path)
        args = argparse.Namespace(
            input=input_path,
            output=output,
            config=self._config.config_path,
            render_frame_ids=None,
            diagnostic_force=self._config.diagnostic_force,
        )
        try:
            with _cuda_policy(self._config.cuda_mode):
                from .stitch_sequence import run

                run(args)
            return PanoramaResult.load(output)
        except PanoramaSDKError:
            raise
        except Exception as exc:
            raise PanoramaProcessingError(
                f"Panorama build failed for {input_path}: {exc}"
            ) from exc

    def generate_demo(
        self,
        output_dir: str | Path,
        *,
        frame_count: int = 10,
        frame_width: int = 640,
        frame_height: int = 400,
        step: int = 120,
        scene: str = "plane",
    ) -> Path:
        """Create a deterministic strict RGB-D demo session for integration tests."""

        output = _path_argument(output_dir, name="output_dir")
        if output.exists() and not output.is_dir():
            raise SDKInputError(f"output_dir must be a directory path: {output}")
        for name, value, minimum in (
            ("frame_count", frame_count, 1),
            ("frame_width", frame_width, 16),
            ("frame_height", frame_height, 16),
            ("step", step, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                relation = "at least" if minimum else "non-negative"
                raise SDKInputError(f"{name} must be an integer {relation} {minimum}")
        if not isinstance(scene, str) or not scene.strip():
            raise SDKInputError("scene must be a non-empty string")
        try:
            from .synthetic import generate_sequence

            return generate_sequence(
                output,
                frame_count=frame_count,
                frame_width=frame_width,
                frame_height=frame_height,
                step=step,
                scene=scene,
            )
        except Exception as exc:
            raise SDKInputError(f"Could not generate demo session: {exc}") from exc


def get_sdk_version() -> str:
    """Return the installed SDK semantic version."""

    return __version__


__all__ = [
    "CudaMode",
    "PanoramaProcessingError",
    "PanoramaResult",
    "PanoramaSDK",
    "PanoramaSDKError",
    "SDKConfig",
    "SDKConfigurationError",
    "SDKInputError",
    "SessionSummary",
    "get_sdk_version",
]
