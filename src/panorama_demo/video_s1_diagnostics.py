"""Best-effort diagnostics for the isolated S01 video experiment."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _replace_pending(path: Path, pending: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending, path)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    _replace_pending(path, pending)


def write_yaml(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    _replace_pending(path, pending)


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.pending")
    with pending.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    _replace_pending(path, pending)


def write_image(path: Path, image: np.ndarray) -> None:
    if image.size == 0 or image.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise ValueError(f"Unsupported diagnostic image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.stem}.pending{path.suffix}")
    if not cv2.imwrite(str(pending), image):
        raise OSError(f"Could not write image: {path}")
    _replace_pending(path, pending)


def owner_map_preview(owner_map: np.ndarray) -> np.ndarray:
    if owner_map.ndim != 2:
        raise ValueError("owner_map must be two-dimensional")
    preview = np.zeros((*owner_map.shape, 3), dtype=np.uint8)
    valid = owner_map >= 0
    if np.any(valid):
        ids = owner_map[valid].astype(np.uint32)
        preview[valid, 0] = ((ids * 97 + 31) % 223 + 24).astype(np.uint8)
        preview[valid, 1] = ((ids * 57 + 83) % 223 + 24).astype(np.uint8)
        preview[valid, 2] = ((ids * 131 + 17) % 223 + 24).astype(np.uint8)
    return preview


def owner_boundary_overlay(panorama: np.ndarray, boundaries: Sequence[int]) -> np.ndarray:
    overlay = panorama.copy()
    for boundary in boundaries:
        x = int(np.clip(boundary, 0, max(0, overlay.shape[1] - 1)))
        cv2.line(overlay, (x, 0), (x, overlay.shape[0] - 1), (0, 255, 255), 1)
    return overlay


def curve_plot(height: int, y: np.ndarray, raw: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    width = 360
    plot = np.full((max(64, int(height)), width, 3), 255, dtype=np.uint8)
    values = np.concatenate((np.asarray(raw).ravel(), np.asarray(smooth).ravel()))
    finite = values[np.isfinite(values)]
    radius = max(2.0, float(np.max(np.abs(finite))) if finite.size else 2.0)
    centre = width // 2
    cv2.line(plot, (centre, 0), (centre, plot.shape[0] - 1), (190, 190, 190), 1)

    def points(values_: np.ndarray) -> np.ndarray:
        xs = centre + np.asarray(values_, dtype=np.float64) * (0.45 * width / radius)
        ys = np.asarray(y, dtype=np.float64)
        return np.rint(np.column_stack((xs, ys))).astype(np.int32)

    if len(y) > 1:
        cv2.polylines(plot, [points(raw)], False, (80, 80, 220), 1, cv2.LINE_AA)
        cv2.polylines(plot, [points(smooth)], False, (30, 160, 30), 2, cv2.LINE_AA)
    return plot
