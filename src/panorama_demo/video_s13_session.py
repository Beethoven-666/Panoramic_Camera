"""Strict-session loading with read-only real-RGB salvage for S1.3."""

from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
import cv2
import numpy as np

from .session import CameraIntrinsics
from .video_session import VideoSession, load_video_session


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class S13RenderFrame:
    frame_id: int
    color_path: Path
    timestamp_us: int
    width: int
    height: int


@dataclass(frozen=True)
class S13Session:
    root: Path
    frames: tuple[S13RenderFrame, ...]
    calibration: CameraIntrinsics
    strict_video: VideoSession | None
    strict_error: str | None
    calibration_mode: str
    trust_state: str
    skipped_rgb: tuple[dict[str, object], ...]

    @property
    def frame_by_id(self) -> dict[int, S13RenderFrame]:
        return {frame.frame_id: frame for frame in self.frames}


@dataclass
class S13ValidatedRgbHandoff:
    images_by_frame_id: dict[int, np.ndarray]
    strict_rgb_decode_count: int
    strict_depth_decode_count: int
    fallback_rgb_decode_count: int
    _taken: bool = field(default=False, init=False, repr=False)

    @property
    def retained_count(self) -> int:
        return len(self.images_by_frame_id)

    @property
    def retained_bytes(self) -> int:
        return sum(int(image.nbytes) for image in self.images_by_frame_id.values())

    def take_all(self) -> dict[int, np.ndarray]:
        if self._taken:
            raise RuntimeError("S1.3 validated RGB handoff was already consumed")
        self._taken = True
        images = self.images_by_frame_id
        self.images_by_frame_id = {}
        return images


@dataclass(frozen=True)
class S13SessionLoadBundle:
    session: S13Session
    validated_rgb_handoff: S13ValidatedRgbHandoff
    performance: Mapping[str, object]


def _decode(path: Path) -> np.ndarray | None:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def read_s13_rgb(frame: S13RenderFrame) -> np.ndarray:
    image = _decode(frame.color_path)
    if image is None or image.dtype != np.uint8 or image.shape != (frame.height, frame.width, 3):
        raise ValueError(f"S1.3 RGB source cannot be decoded consistently: {frame.color_path}")
    return image


def _safe_relative(root: Path, value: str) -> Path | None:
    relative = Path(value.strip())
    if not value.strip() or relative.is_absolute():
        return None
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _rows_from_csv(root: Path) -> list[tuple[int, int, Path]]:
    path = root / "frames.csv"
    if not path.is_file():
        return []
    rows: list[tuple[int, int, Path]] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle)):
                try:
                    frame_id = int(row.get("frame_id", ""))
                except (TypeError, ValueError):
                    continue
                color = _safe_relative(root, str(row.get("color_path", "")))
                if frame_id < 0 or color is None:
                    continue
                raw_timestamp = row.get("color_device_timestamp_us") or row.get("timestamp_us")
                try:
                    timestamp = int(raw_timestamp) if raw_timestamp not in (None, "") else index
                except (TypeError, ValueError):
                    timestamp = index
                rows.append((frame_id, max(0, timestamp), color))
    except (OSError, UnicodeError, csv.Error):
        return []
    return rows


def _rows_from_color_dir(root: Path) -> list[tuple[int, int, Path]]:
    directory = root / "color"
    if not directory.is_dir():
        return []
    result: list[tuple[int, int, Path]] = []
    for index, path in enumerate(sorted(item for item in directory.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES)):
        matches = re.findall(r"\d+", path.stem)
        frame_id = int(matches[-1]) if matches else index
        result.append((frame_id, index, path.resolve()))
    return result


def _raw_intrinsics(width: int, height: int) -> CameraIntrinsics:
    focal = float(max(width, height))
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=focal,
        fy=focal,
        cx=(width - 1.0) * 0.5,
        cy=(height - 1.0) * 0.5,
        distortion=(),
    )


def _salvage_calibration(root: Path, *, width: int, height: int) -> CameraIntrinsics | None:
    try:
        payload = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
        intrinsic = payload["color_intrinsic"]
        distortion = payload["color_distortion"]
        values = tuple(float(distortion[key]) for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"))
        calibration = CameraIntrinsics(
            width=int(intrinsic["width"]), height=int(intrinsic["height"]),
            fx=float(intrinsic["fx"]), fy=float(intrinsic["fy"]),
            cx=float(intrinsic["cx"]), cy=float(intrinsic["cy"]), distortion=values,
        )
        if calibration.width != width or calibration.height != height:
            return None
        if not np.isfinite(np.asarray((calibration.fx, calibration.fy, calibration.cx, calibration.cy, *values))).all():
            return None
        return calibration
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def load_s13_session_bundle(
    input_path: str | Path, *, validation_workers: int = 4
) -> S13SessionLoadBundle:
    total_started = time.perf_counter()
    root = Path(input_path).expanduser().resolve()
    root = root if root.is_dir() else root.parent
    strict: VideoSession | None = None
    strict_error: str | None = None
    strict_images: dict[int, np.ndarray] = {}
    strict_started = time.perf_counter()

    def retain_validated(frame, image: np.ndarray) -> None:
        strict_images[int(frame.frame_id)] = image

    try:
        strict = load_video_session(
            root,
            validate_frame_files=True,
            validation_workers=validation_workers,
            validated_color_sink=retain_validated,
        )
        candidates = [
            (frame.frame_id, int(frame.timestamp_us or index), frame.color_path)
            for index, frame in enumerate(strict.rgbd.frames)
        ]
    except Exception as exc:
        strict_images.clear()
        strict_error = f"{type(exc).__name__}: {exc}"
        candidates = _rows_from_csv(root) or _rows_from_color_dir(root)
    strict_seconds = time.perf_counter() - strict_started

    materialize_started = time.perf_counter()
    decoded: list[S13RenderFrame] = []
    skipped: list[dict[str, object]] = []
    handoff_images = strict_images if strict is not None else {}
    expected_shape: tuple[int, int] | None = None
    seen: set[int] = set()
    for frame_id, timestamp, path in candidates:
        image = strict_images.get(frame_id) if strict is not None else _decode(path)
        reason = None
        if frame_id in seen:
            reason = "duplicate_frame_id"
        elif image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            reason = "rgb_decode_failed"
        elif expected_shape is not None and image.shape[:2] != expected_shape:
            reason = "rgb_dimension_mismatch"
        if reason is not None:
            skipped.append({"frame_id": frame_id, "path": str(path), "reason": reason})
            continue
        expected_shape = image.shape[:2]
        if not image.flags.c_contiguous:
            skipped.append({"frame_id": frame_id, "path": str(path), "reason": "rgb_not_c_contiguous"})
            continue
        image.setflags(write=False)
        seen.add(frame_id)
        decoded.append(S13RenderFrame(frame_id, path.resolve(), max(0, timestamp), image.shape[1], image.shape[0]))
        if strict is None:
            handoff_images[frame_id] = image
    decoded.sort(key=lambda item: (item.timestamp_us, item.frame_id))
    if not decoded:
        raise ValueError("S1.3 input_fatal: no decodable real RGB source")
    width, height = decoded[0].width, decoded[0].height
    calibration = strict.rgbd.calibration if strict is not None else _salvage_calibration(root, width=width, height=height)
    calibration_mode = "calibrated_target"
    if calibration is None:
        calibration = _raw_intrinsics(width, height)
        calibration_mode = "raw_rgb_fallback"
    try:
        # Constructing the maps here makes map failure an explicit raw fallback.
        from .calibrated_remap import undistortion_maps

        undistortion_maps(calibration)
    except Exception as exc:
        skipped.append({"frame_id": -1, "path": str(root / "calibration.json"), "reason": f"calibration_map_failed: {exc}"})
        calibration = _raw_intrinsics(width, height)
        calibration_mode = "raw_rgb_fallback"
    trust = "strict_session_rgb_motion" if strict is not None and calibration_mode == "calibrated_target" else "readable_rgb_untrusted_metadata"
    session = S13Session(
        root=root,
        frames=tuple(decoded),
        calibration=calibration,
        strict_video=strict,
        strict_error=strict_error,
        calibration_mode=calibration_mode,
        trust_state=trust,
        skipped_rgb=tuple(skipped),
    )
    validation = {} if strict is None else dict(strict.rgbd.validation_performance)
    materialize_seconds = time.perf_counter() - materialize_started
    handoff = S13ValidatedRgbHandoff(
        images_by_frame_id=handoff_images,
        strict_rgb_decode_count=int(validation.get("strict_rgb_decode_count", 0)),
        strict_depth_decode_count=int(validation.get("strict_depth_decode_count", 0)),
        fallback_rgb_decode_count=len(candidates) if strict is None else 0,
    )
    performance = {
        "total_wall_seconds": time.perf_counter() - total_started,
        "strict_session_load_wall_seconds": strict_seconds,
        "strict_file_validation_wall_seconds": float(
            validation.get("strict_file_validation_wall_seconds", 0.0)
        ),
        "s13_session_materialize_wall_seconds": materialize_seconds,
        "validated_rgb_handoff_wall_seconds": 0.0,
        "strict_rgb_decode_count": handoff.strict_rgb_decode_count,
        "strict_depth_decode_count": handoff.strict_depth_decode_count,
        "s13_fallback_rgb_decode_count": handoff.fallback_rgb_decode_count,
        "validated_rgb_retained_count": handoff.retained_count,
        "validated_rgb_retained_bytes": handoff.retained_bytes,
        "frame_store_post_handoff_decode_count": 0,
    }
    return S13SessionLoadBundle(session, handoff, performance)


def load_s13_session(input_path: str | Path, *, validation_workers: int = 4) -> S13Session:
    return load_s13_session_bundle(
        input_path, validation_workers=validation_workers
    ).session


__all__ = [
    "S13RenderFrame",
    "S13Session",
    "S13SessionLoadBundle",
    "S13ValidatedRgbHandoff",
    "load_s13_session",
    "load_s13_session_bundle",
    "read_s13_rgb",
]
