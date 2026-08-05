"""Integrity-checked online scan facts for the video fast delivery path.

An online state records work completed while a capture is still in progress:
strict frame validation, cheap motion and scan segmentation.  Reuse is allowed
only after every source RGB/depth file and the session control files hash to
the recorded values.  It therefore avoids re-decoding unchanged frames after
capture without admitting a changed session.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .quality import FrameQuality, MotionEstimate, analyze_frame_quality, resize_for_analysis, select_primary_scan_segment
from .session import RGBDFrame
from .video_scan_segment import estimate_video_motion


ONLINE_STATE_SCHEMA = "gemini305-video-online-state/v1"
CAPTURE_FRAME_VALIDATION_SCHEMA = "gemini305-online-capture-frame-validation/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_hashes(root: Path) -> dict[str, str]:
    return {
        "manifest": _sha256(root / "manifest.json"),
        "calibration": _sha256(root / "calibration.json"),
        "frames_csv": _sha256(root / "frames.csv"),
    }


@dataclass(frozen=True)
class OnlineVideoState:
    qualities: tuple[FrameQuality, ...]
    motions: tuple[MotionEstimate, ...]
    segment: dict[str, object]
    origin: str
    capture_frame_validation: dict[str, object] | None = None

    @property
    def certifies_strict_frame_files(self) -> bool:
        """Whether capture-time validation may replace a later full decode."""

        return self.origin == "capture" and self.capture_frame_validation is not None


class OnlineScanAccumulator:
    """Accumulate cheap scan facts as accepted RGB-D frames are captured.

    The accumulator intentionally works from the just-aligned capture colour
    image, before JPEG encoding.  The persisted state is later bound to the
    exact encoded RGB-D files by their writer-computed SHA-256 values.
    """

    def __init__(self, *, analysis_width: int = 320, motion_backend: str = "dis") -> None:
        if motion_backend not in {"dis", "feature"}:
            raise ValueError("video motion_backend must be 'dis' or 'feature'")
        self.analysis_width = int(analysis_width)
        self.motion_backend = motion_backend
        self._previous_preview = None
        self._frame_ids: list[int] = []
        self._qualities: list[FrameQuality] = []
        self._motions: list[MotionEstimate] = []

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(self._frame_ids)

    def add(self, frame_id: int, color_bgr) -> None:
        """Record one frame that has been accepted by the disk writer queue."""

        if self._frame_ids and int(frame_id) <= self._frame_ids[-1]:
            raise ValueError("online video frame IDs must be strictly increasing")
        preview = resize_for_analysis(color_bgr, self.analysis_width)
        self._qualities.append(analyze_frame_quality(preview))
        if self._previous_preview is not None:
            self._motions.append(
                estimate_video_motion(
                    self._previous_preview,
                    preview,
                    motion_backend=self.motion_backend,
                )
            )
        self._previous_preview = preview
        self._frame_ids.append(int(frame_id))

    def finish(self) -> tuple[list[FrameQuality], list[MotionEstimate], dict[str, object]]:
        """Return a single primary segment after the capture has ended."""

        if len(self._qualities) < 2 or len(self._motions) != len(self._qualities) - 1:
            raise ValueError("online scan state needs at least two accepted frames")
        segment = select_primary_scan_segment(
            self._motions,
            image_width=int(self._previous_preview.shape[1]),
        )
        if segment.end_index - segment.start_index + 1 < 2:
            raise ValueError("online scan state has no usable directional segment")
        return list(self._qualities), list(self._motions), segment.as_dict()


def write_online_state(
    path: Path,
    *,
    root: Path,
    frames: Sequence[RGBDFrame],
    qualities: Sequence[FrameQuality],
    motions: Sequence[MotionEstimate],
    segment: dict[str, object],
    origin: str,
    frame_file_sha256: Sequence[Mapping[str, object]] | None = None,
    capture_frame_validation: Mapping[str, object] | None = None,
) -> None:
    """Atomically persist a complete, content-bound online state."""

    if origin not in {"capture", "offline_prepare"}:
        raise ValueError("online state origin must be capture or offline_prepare")
    if len(qualities) != len(frames) or len(motions) != len(frames) - 1:
        raise ValueError("online state must cover all frames and adjacent motions")
    if frame_file_sha256 is None:
        file_hashes = [
            {
                "frame_id": int(frame.frame_id),
                "color_sha256": _sha256(frame.color_path),
                "aligned_depth_sha256": _sha256(frame.aligned_depth_path),
            }
            for frame in frames
        ]
    else:
        file_hashes = [dict(record) for record in frame_file_sha256]
        if len(file_hashes) != len(frames) or any(
            record.get("frame_id") != int(frame.frame_id)
            or not isinstance(record.get("color_sha256"), str)
            or not isinstance(record.get("aligned_depth_sha256"), str)
            for record, frame in zip(file_hashes, frames, strict=True)
        ):
            raise ValueError("online state supplied file hashes do not cover its frames")
    if capture_frame_validation is not None:
        validation = dict(capture_frame_validation)
        if (
            origin != "capture"
            or validation.get("schema") != CAPTURE_FRAME_VALIDATION_SCHEMA
            or validation.get("frame_count") != len(frames)
        ):
            raise ValueError("online capture frame validation is invalid")
    else:
        validation = None
    payload = {
        "schema": ONLINE_STATE_SCHEMA,
        "origin": origin,
        "input_sha256": input_hashes(root),
        "frame_file_sha256": file_hashes,
        "qualities": [item.as_dict() for item in qualities],
        "motions": [item.as_dict() for item in motions],
        "scan_segment": dict(segment),
    }
    if validation is not None:
        payload["capture_frame_validation"] = validation
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pending.replace(path)


def _quality(value: object) -> FrameQuality:
    if not isinstance(value, dict):
        raise ValueError("online state quality row is invalid")
    fields = FrameQuality.__dataclass_fields__
    if set(value) != set(fields):
        raise ValueError("online state quality fields are invalid")
    return FrameQuality(**{name: float(value[name]) for name in fields})


def _motion(value: object) -> MotionEstimate:
    if not isinstance(value, dict):
        raise ValueError("online state motion row is invalid")
    required = {"dx", "dy", "matches", "inlier_ratio", "grid_coverage", "method"}
    if not required.issubset(value):
        raise ValueError("online state motion fields are invalid")
    return MotionEstimate(
        dx=float(value["dx"]),
        dy=float(value["dy"]),
        matches=int(value["matches"]),
        inlier_ratio=float(value["inlier_ratio"]),
        grid_coverage=float(value["grid_coverage"]),
        method=str(value["method"]),
    )


def load_online_state(
    path: Path, *, root: Path, frames: Sequence[RGBDFrame]
) -> OnlineVideoState:
    """Verify all source bytes before returning previously computed facts."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid online video state: {path}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != ONLINE_STATE_SCHEMA:
        raise ValueError("Online video state has an unsupported schema")
    origin = payload.get("origin")
    if origin not in {"capture", "offline_prepare"}:
        raise ValueError("Online video state origin is invalid")
    if payload.get("input_sha256") != input_hashes(root):
        raise ValueError("Online video state control-file hashes do not match session")
    records = payload.get("frame_file_sha256")
    if not isinstance(records, list) or len(records) != len(frames):
        raise ValueError("Online video state frame hashes do not cover session")
    for record, frame in zip(records, frames, strict=True):
        if not isinstance(record, dict) or record.get("frame_id") != int(frame.frame_id):
            raise ValueError("Online video state frame IDs do not match session")
        if (
            record.get("color_sha256") != _sha256(frame.color_path)
            or record.get("aligned_depth_sha256") != _sha256(frame.aligned_depth_path)
        ):
            raise ValueError("Online video state frame bytes do not match session")
    qualities = payload.get("qualities")
    motions = payload.get("motions")
    segment = payload.get("scan_segment")
    if not isinstance(qualities, list) or not isinstance(motions, list) or not isinstance(segment, dict):
        raise ValueError("Online video state has incomplete scan facts")
    if len(qualities) != len(frames) or len(motions) != len(frames) - 1:
        raise ValueError("Online video state scan facts do not cover session")
    validation_value = payload.get("capture_frame_validation")
    validation: dict[str, object] | None = None
    if validation_value is not None:
        if (
            not isinstance(validation_value, dict)
            or origin != "capture"
            or validation_value.get("schema") != CAPTURE_FRAME_VALIDATION_SCHEMA
            or validation_value.get("frame_count") != len(frames)
        ):
            raise ValueError("Online capture frame validation is invalid")
        validation = dict(validation_value)
    return OnlineVideoState(
        qualities=tuple(_quality(item) for item in qualities),
        motions=tuple(_motion(item) for item in motions),
        segment=dict(segment),
        origin=origin,
        capture_frame_validation=validation,
    )
