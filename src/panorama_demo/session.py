from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SessionFrame:
    frame_id: int
    color_path: Path
    depth_path: Path | None = None
    timestamp_us: int | None = None


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
            frames.append(
                SessionFrame(
                    frame_id=int(frame_value),
                    color_path=color,
                    depth_path=depth,
                    timestamp_us=int(timestamp_value) if timestamp_value else None,
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
