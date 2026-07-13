from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DISTORTION_KEYS = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")


@dataclass(frozen=True)
class SessionFrame:
    frame_id: int
    color_path: Path
    depth_path: Path | None = None
    timestamp_us: int | None = None
    depth_scale_mm_per_unit: float | None = None
    color_exposure_raw: int | None = None
    color_gain: int | None = None


@dataclass(frozen=True)
class CameraIntrinsics:
    """Validated color-camera calibration used by the RGB-D pipeline."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, ...]

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RGBDFrame:
    """One formal RGB-D source frame.

    The on-disk PNG remains in device units. Use :func:`read_aligned_depth_mm`
    whenever depth pixels enter project code; it performs the sole explicit
    conversion to the project's millimetre convention.
    """

    frame_id: int
    color_path: Path
    aligned_depth_path: Path
    depth_scale_mm_per_unit: float
    timestamp_us: int | None = None
    color_exposure_raw: int | None = None
    color_gain: int | None = None

    @property
    def depth_path(self) -> Path:
        """Compatibility alias; the strict loader never accepts raw depth."""

        return self.aligned_depth_path

    @property
    def exposure(self) -> int | None:
        return self.color_exposure_raw


@dataclass(frozen=True)
class RGBDSession:
    """A fail-closed capture session ready for RGB-D pose estimation."""

    root: Path
    calibration: CameraIntrinsics
    frames: tuple[RGBDFrame, ...]
    manifest: dict[str, Any] | None = None
    depth_alignment: str = "color"


def load_session_manifest(input_path: str | Path) -> dict[str, Any] | None:
    """Load a capture manifest adjacent to a supported session input, if present."""

    path = Path(input_path).expanduser().resolve()
    candidates: list[Path] = []
    if path.is_dir():
        candidates.append(path / "manifest.json")
        if path.name.lower() == "color":
            candidates.append(path.parent / "manifest.json")
    elif path.is_file():
        candidates.append(path.parent / "manifest.json")
        if path.parent.name.lower() == "color":
            candidates.append(path.parent.parent / "manifest.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid session manifest: {candidate}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Session manifest must contain a JSON object: {candidate}")
        return payload
    return None


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label.capitalize()} must contain a JSON object: {path}")
    return payload


def _required_finite_float(
    mapping: dict[str, Any], key: str, *, context: str, positive: bool = False
) -> float:
    if key not in mapping or isinstance(mapping[key], bool):
        raise ValueError(f"{context} is missing numeric {key!r}")
    try:
        value = float(mapping[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has invalid {key!r}") from exc
    if not math.isfinite(value) or (positive and value <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{context} {key!r} must be {qualifier}")
    return value


def _required_positive_int(mapping: dict[str, Any], key: str, *, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} {key!r} must be a positive integer")
    return value


def _parse_color_intrinsics(calibration: dict[str, Any]) -> CameraIntrinsics:
    intrinsic = calibration.get("color_intrinsic")
    if not isinstance(intrinsic, dict):
        raise ValueError("calibration.json is missing color_intrinsic")
    distortion = calibration.get("color_distortion")
    if not isinstance(distortion, dict):
        raise ValueError("calibration.json is missing color_distortion")

    context = "calibration color_intrinsic"
    width = _required_positive_int(intrinsic, "width", context=context)
    height = _required_positive_int(intrinsic, "height", context=context)
    fx = _required_finite_float(intrinsic, "fx", context=context, positive=True)
    fy = _required_finite_float(intrinsic, "fy", context=context, positive=True)
    cx = _required_finite_float(intrinsic, "cx", context=context)
    cy = _required_finite_float(intrinsic, "cy", context=context)
    if not 0.0 <= cx < float(width) or not 0.0 <= cy < float(height):
        raise ValueError("calibration principal point lies outside the color image")

    coefficients = tuple(
        _required_finite_float(
            distortion, key, context="calibration color_distortion"
        )
        for key in DISTORTION_KEYS
    )
    return CameraIntrinsics(width, height, fx, fy, cx, cy, coefficients)


def _declares_color_aligned_depth(
    calibration: dict[str, Any], manifest: dict[str, Any] | None
) -> bool:
    if calibration.get("depth_aligned_to_color") is True:
        return True
    marker = calibration.get("depth_alignment")
    if isinstance(marker, str):
        if marker.strip().lower() in {"color", "to_color", "aligned_to_color"}:
            return True
    elif isinstance(marker, dict):
        target = marker.get("aligned_to", marker.get("target"))
        enabled = marker.get("enabled", marker.get("aligned", True))
        if (
            isinstance(target, str)
            and target.strip().lower() in {"color", "rgb", "color_stream"}
            and enabled is not False
        ):
            return True

    # Legacy sessions produced by this project's v1 capture command predate
    # the calibration marker above.  That capture schema only supported the
    # SDK software AlignFilter targeted explicitly at COLOR_STREAM, so this
    # exact provenance is accepted.  Arbitrary align strings are not.
    capture_options = manifest.get("capture_options") if manifest else None
    align_mode = capture_options.get("align") if isinstance(capture_options, dict) else None
    if (
        manifest is not None
        and manifest.get("schema") == "panorama-demo-session/v1"
        and isinstance(align_mode, str)
        and align_mode.strip().lower() == "software"
    ):
        return True
    return False


def _session_root_and_csv(input_path: str | Path) -> tuple[Path, Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        root = path
        csv_path = root / "frames.csv"
    elif path.is_file() and path.name.lower() == "frames.csv":
        root = path.parent
        csv_path = path
    elif path.exists():
        raise ValueError(
            "Formal RGB-D input must be a session directory or its frames.csv"
        )
    else:
        raise FileNotFoundError(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing frames.csv: {csv_path}")
    return root, csv_path


def _resolve_session_path(
    root: Path, raw_value: str | None, *, field: str, row_number: int
) -> Path:
    value = (raw_value or "").strip()
    if not value:
        raise ValueError(f"frames.csv row {row_number} is missing {field}")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"frames.csv row {row_number} {field} must be relative")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"frames.csv row {row_number} {field} escapes the session root"
        ) from exc
    return resolved


def _decode_image(path: Path, flags: int, *, label: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"Could not read {label}: {path}") from exc
    image = cv2.imdecode(encoded, flags) if encoded.size else None
    if image is None:
        raise ValueError(f"Could not decode {label}: {path}")
    return image


def _validate_frame_files(
    frame: RGBDFrame, calibration: CameraIntrinsics, *, row_number: int
) -> None:
    color = _decode_image(frame.color_path, cv2.IMREAD_COLOR, label="color image")
    expected_shape = (calibration.height, calibration.width)
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError(f"frames.csv row {row_number} color image is not 8-bit RGB")
    if color.shape[:2] != expected_shape:
        raise ValueError(
            f"frames.csv row {row_number} color size {color.shape[1]}x{color.shape[0]} "
            f"does not match calibration {calibration.width}x{calibration.height}"
        )

    depth = _decode_image(
        frame.aligned_depth_path,
        cv2.IMREAD_UNCHANGED,
        label="aligned depth image",
    )
    if frame.aligned_depth_path.suffix.lower() != ".png" or depth.dtype != np.uint16:
        raise ValueError(
            f"frames.csv row {row_number} aligned depth must be a 16-bit PNG"
        )
    if depth.ndim != 2:
        raise ValueError(f"frames.csv row {row_number} aligned depth must be single-channel")
    if depth.shape != expected_shape:
        raise ValueError(
            f"frames.csv row {row_number} aligned depth size "
            f"{depth.shape[1]}x{depth.shape[0]} does not match color/calibration"
        )
    if not np.any(depth > 0):
        raise ValueError(f"frames.csv row {row_number} aligned depth has no valid pixels")
    max_depth_mm = float(depth.max()) * frame.depth_scale_mm_per_unit
    if not math.isfinite(max_depth_mm):
        raise ValueError(f"frames.csv row {row_number} depth conversion is not finite")


def _parse_optional_int(
    row: dict[str, str], key: str, *, row_number: int
) -> int | None:
    value = (row.get(key) or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"frames.csv row {row_number} has invalid {key}") from exc


def load_rgbd_session(input_path: str | Path) -> RGBDSession:
    """Load and fully validate a formal color-aligned RGB-D capture session.

    Unlike :func:`discover_frames`, this entry point never accepts RGB-only
    input and never treats ``depth_path`` or ``raw_depth_path`` as aligned data.
    This contract is intentionally not bypassable by diagnostic quality flags.
    """

    root, csv_path = _session_root_and_csv(input_path)
    calibration_payload = _read_json_object(
        root / "calibration.json", "calibration.json"
    )
    calibration = _parse_color_intrinsics(calibration_payload)
    manifest = load_session_manifest(root)
    if not _declares_color_aligned_depth(calibration_payload, manifest):
        raise ValueError("Session does not explicitly declare depth aligned to color")

    frames: list[RGBDFrame] = []
    frame_ids: set[int] = set()
    color_paths: set[Path] = set()
    depth_paths: set[Path] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {
            "frame_id",
            "color_exposure",
            "color_path",
            "aligned_depth_path",
            "depth_scale_mm_per_unit",
        }
        missing_columns = sorted(required - fieldnames)
        if missing_columns:
            raise ValueError(
                "frames.csv is missing formal RGB-D columns: "
                + ", ".join(missing_columns)
            )
        if not {"color_device_timestamp_us", "timestamp_us"}.intersection(
            fieldnames
        ):
            raise ValueError(
                "frames.csv is missing a formal RGB-D color timestamp column"
            )
        for row_number, row in enumerate(reader, start=2):
            frame_id = _parse_optional_int(row, "frame_id", row_number=row_number)
            if frame_id is None or frame_id < 0:
                raise ValueError(
                    f"frames.csv row {row_number} frame_id must be non-negative"
                )
            if frame_id in frame_ids:
                raise ValueError(f"frames.csv contains duplicate frame_id {frame_id}")

            color_path = _resolve_session_path(
                root, row.get("color_path"), field="color_path", row_number=row_number
            )
            aligned_depth_path = _resolve_session_path(
                root,
                row.get("aligned_depth_path"),
                field="aligned_depth_path",
                row_number=row_number,
            )
            suspicious_parts = {
                part.lower() for part in aligned_depth_path.relative_to(root).parts
            }
            if suspicious_parts.intersection({"depth_raw", "raw_depth"}):
                raise ValueError(
                    f"frames.csv row {row_number} aligned_depth_path points to raw depth"
                )
            relative_depth_parts = aligned_depth_path.relative_to(root).parts
            if (
                len(relative_depth_parts) < 2
                or relative_depth_parts[0].lower() != "depth_aligned"
            ):
                raise ValueError(
                    f"frames.csv row {row_number} aligned_depth_path must use "
                    "the depth_aligned directory"
                )
            raw_depth_value = (row.get("raw_depth_path") or "").strip()
            if raw_depth_value:
                raw_depth_path = _resolve_session_path(
                    root,
                    raw_depth_value,
                    field="raw_depth_path",
                    row_number=row_number,
                )
                if raw_depth_path == aligned_depth_path:
                    raise ValueError(
                        f"frames.csv row {row_number} raw depth cannot masquerade as aligned depth"
                    )

            scale_text = (row.get("depth_scale_mm_per_unit") or "").strip()
            try:
                scale = float(scale_text)
            except ValueError as exc:
                raise ValueError(
                    f"frames.csv row {row_number} has invalid depth scale"
                ) from exc
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"frames.csv row {row_number} depth scale must be finite and positive"
                )

            timestamp = _parse_optional_int(
                row, "color_device_timestamp_us", row_number=row_number
            )
            if timestamp is None:
                timestamp = _parse_optional_int(
                    row, "timestamp_us", row_number=row_number
                )
            if timestamp is None or timestamp < 0:
                raise ValueError(
                    f"frames.csv row {row_number} requires a non-negative color timestamp"
                )
            color_exposure = _parse_optional_int(
                row, "color_exposure", row_number=row_number
            )
            if color_exposure is None or color_exposure <= 0:
                raise ValueError(
                    f"frames.csv row {row_number} requires positive color_exposure metadata"
                )
            frame = RGBDFrame(
                frame_id=frame_id,
                color_path=color_path,
                aligned_depth_path=aligned_depth_path,
                depth_scale_mm_per_unit=scale,
                timestamp_us=timestamp,
                color_exposure_raw=color_exposure,
                color_gain=_parse_optional_int(
                    row, "color_gain", row_number=row_number
                ),
            )
            if color_path in color_paths or aligned_depth_path in depth_paths:
                raise ValueError(
                    f"frames.csv row {row_number} reuses a color or aligned-depth file"
                )
            _validate_frame_files(frame, calibration, row_number=row_number)
            frame_ids.add(frame_id)
            color_paths.add(color_path)
            depth_paths.add(aligned_depth_path)
            frames.append(frame)

    if not frames:
        raise ValueError("Formal RGB-D session contains no frames")
    return RGBDSession(root, calibration, tuple(frames), manifest)


def read_aligned_depth_mm(frame: RGBDFrame) -> np.ndarray:
    """Decode one aligned depth image and convert device units to millimetres."""

    depth = _decode_image(
        frame.aligned_depth_path,
        cv2.IMREAD_UNCHANGED,
        label="aligned depth image",
    )
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(f"Aligned depth is not a single-channel uint16 image: {frame.aligned_depth_path}")
    scale = float(frame.depth_scale_mm_per_unit)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth_scale_mm_per_unit must be finite and positive")
    return depth.astype(np.float32) * np.float32(scale)


def _from_csv(root: Path, csv_path: Path) -> list[SessionFrame]:
    frames: list[SessionFrame] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            color_value = row.get("color_path") or row.get("image_path")
            if not color_value:
                continue
            color = (root / Path(color_value)).resolve()
            depth_value = row.get("aligned_depth_path") or row.get("depth_path")
            depth = (root / Path(depth_value)).resolve() if depth_value else None
            frame_value = row.get("frame_id") or row.get("pair_id") or str(row_index)
            timestamp_value = row.get("color_device_timestamp_us") or row.get("timestamp_us")
            depth_scale_value = row.get("depth_scale_mm_per_unit")
            exposure_value = row.get("color_exposure")
            gain_value = row.get("color_gain")
            frames.append(
                SessionFrame(
                    frame_id=int(frame_value),
                    color_path=color,
                    depth_path=depth,
                    timestamp_us=int(timestamp_value) if timestamp_value else None,
                    depth_scale_mm_per_unit=(
                        float(depth_scale_value) if depth_scale_value else None
                    ),
                    color_exposure_raw=(
                        int(exposure_value) if exposure_value else None
                    ),
                    color_gain=int(gain_value) if gain_value else None,
                )
            )
    return frames


def discover_frames(input_path: str | Path) -> list[SessionFrame]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() == ".csv":
            return _from_csv(path.parent, path)
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return [SessionFrame(frame_id=0, color_path=path)]
        raise ValueError(f"Unsupported input file: {path}")

    if not path.is_dir():
        raise FileNotFoundError(path)

    csv_path = path / "frames.csv"
    if csv_path.exists():
        frames = _from_csv(path, csv_path)
    else:
        color_dir = path / "color"
        scan_dir = color_dir if color_dir.is_dir() else path
        images = sorted(
            item.resolve()
            for item in scan_dir.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
        frames = [SessionFrame(frame_id=index, color_path=item) for index, item in enumerate(images)]

    missing = [str(frame.color_path) for frame in frames if not frame.color_path.exists()]
    if missing:
        raise FileNotFoundError(f"Session references missing color images, first missing: {missing[0]}")
    return frames


def select_frames(
    frames: list[SessionFrame], stride: int = 1, max_frames: int | None = None
) -> list[SessionFrame]:
    if stride < 1:
        raise ValueError("stride must be at least 1")
    selected = frames[::stride]
    if max_frames is not None and max_frames > 0:
        selected = selected[:max_frames]
    return selected
