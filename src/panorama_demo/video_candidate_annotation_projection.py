"""Candidate-only fixed-annotation projection from calibrated inverse maps.

This is deliberately a measurement adapter, not a renderer.  It accepts the
already-built calibrated inverse coordinate maps, rasterises fixed source
annotations through those maps, and intersects the result with final owner
provenance.  It never reads RGB, changes a sampling grid, or makes an owner
decision.  Callers write the returned payload only after primary publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .video_offline_evaluation import PANORAMA_PROJECTION_SCHEMA


@dataclass(frozen=True)
class CandidateInverseMapSource:
    """One source's already-generated calibrated inverse map on its canvas tile."""

    frame_id: int
    canvas_x0: int
    source_map_x: np.ndarray
    source_map_y: np.ndarray
    valid_mask: np.ndarray
    raw_shape: tuple[int, int]

    def __post_init__(self) -> None:
        maps = (np.asarray(self.source_map_x), np.asarray(self.source_map_y), np.asarray(self.valid_mask))
        if any(value.ndim != 2 for value in maps) or maps[0].shape != maps[1].shape or maps[0].shape != maps[2].shape:
            raise ValueError("Candidate inverse source maps must be matching two-dimensional arrays")
        height, width = self.raw_shape
        if height < 1 or width < 1:
            raise ValueError("Candidate inverse source raw_shape must be positive")


def _coerce_v2_strip(strip: object) -> tuple[int, int, int, int, float]:
    """Read the minimal immutable C1 strip geometry without importing CUDA.

    The annotation adapter deliberately depends on the *planned* v2 geometry,
    not on a rendered owner image.  Keeping this duck-typed avoids a
    renderer-to-measurement import cycle while making the expected source
    identity and scalar layout explicit.
    """

    try:
        frame_id = int(getattr(strip, "frame_id"))
        output_x0 = int(getattr(strip, "output_x0"))
        source_x0 = int(getattr(strip, "source_x0"))
        width = int(getattr(strip, "width"))
        centre_raw = getattr(strip, "source_centre_x")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("v2 projection needs a C1 CUDA source strip") from exc
    centre = float(source_x0) if centre_raw is None else float(centre_raw)
    if frame_id < 0 or output_x0 < 0 or source_x0 < 0 or width < 1 or not np.isfinite(centre):
        raise ValueError("v2 projection received invalid C1 CUDA strip geometry")
    return frame_id, output_x0, source_x0, width, centre


def _c1_corridors(
    strips: Sequence[tuple[int, int, int, int, float]],
    source_shapes: Mapping[int, tuple[int, int]],
    *,
    canvas_width: int,
    corridor_width_pixels: int,
) -> tuple[tuple[int, int], ...]:
    """Mirror ``TorchCudaC1ConstrainedOwnerAlgorithm._pair_corridors``.

    This is geometry only: it neither examines RGB nor changes a C1 seam.
    Computing it independently from final ownership is important because a
    curved seam may assign an annotated source pixels on either side of its
    initial hard-strip boundary.
    """

    if not 8 <= int(corridor_width_pixels) <= 256:
        raise ValueError("v2 C1 projection corridor width must be in [8, 256]")
    corridors: list[tuple[int, int]] = []
    for first, second in zip(strips[:-1], strips[1:], strict=True):
        first_id, first_x0, first_source_x0, first_width, _ = first
        second_id, second_x0, second_source_x0, _, _ = second
        if first_x0 + first_width != second_x0:
            raise ValueError("v2 C1 projection strips must meet at every chronological boundary")
        try:
            first_raw_width = int(source_shapes[first_id][1])
            second_raw_width = int(source_shapes[second_id][1])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("v2 C1 projection lacks a real raw shape for a selected source") from exc
        first_support = (first_x0 - first_source_x0, first_x0 - first_source_x0 + first_raw_width)
        second_support = (second_x0 - second_source_x0, second_x0 - second_source_x0 + second_raw_width)
        shared_left = max(first_support[0], second_support[0], 0)
        shared_right = min(first_support[1], second_support[1], int(canvas_width))
        width = int(corridor_width_pixels)
        if shared_right - shared_left < width:
            raise ValueError("v2 C1 projection pair lacks genuine calibrated common support")
        boundary = first_x0 + first_width
        start = min(max(boundary - width // 2, shared_left), shared_right - width)
        corridors.append((int(start), int(start + width)))
    return tuple(corridors)


def _brown_conrady_inverse_map(
    *,
    canvas_x0: int,
    width: int,
    height: int,
    source_width: int,
    source_height: int,
    source_centre_x: float,
    fx: float,
    fy: float,
    raw_cx: float,
    raw_cy: float,
    distortion: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerical reference for the v2 CUDA ``calibrated_inverse_grid``.

    It uses the exact target coordinate convention from C1's
    ``_render_window``: ``cx = source_centre_x - canvas_x0`` and original
    raw-source ``raw_cx/raw_cy``.  This post-render CPU calculation is a
    read-only measurement representation of the grid already used for the
    one CUDA RGB sample; it never contributes pixels back to the renderer.
    """

    if (
        width < 1
        or height < 1
        or source_width < 2
        or source_height < 2
        or not all(np.isfinite(value) for value in (source_centre_x, fx, fy, raw_cx, raw_cy))
        or fx <= 0.0
        or fy <= 0.0
        or len(distortion) not in (0, 4, 5, 8)
        or not all(np.isfinite(float(value)) for value in distortion)
    ):
        raise ValueError("v2 calibrated inverse projection parameters are invalid")
    coeffs = tuple(float(value) for value in distortion) + (0.0,) * (8 - len(distortion))
    k1, k2, p1, p2, k3, k4, k5, k6 = coeffs
    # Equivalent to the CUDA local target x coordinate after substituting
    # ``cx = source_centre_x - canvas_x0``.  Working in global canvas x
    # prevents a tile-origin mistake from being hidden by the final owner.
    xs = np.arange(int(canvas_x0), int(canvas_x0) + int(width), dtype=np.float64)[None, :]
    ys = np.arange(int(height), dtype=np.float64)[:, None]
    x = (xs - float(source_centre_x)) / float(fx)
    y = (ys - float(raw_cy)) / float(fy)
    radius2 = x * x + y * y
    numerator = 1.0 + k1 * radius2 + k2 * radius2 * radius2 + k3 * radius2 * radius2 * radius2
    denominator = 1.0 + k4 * radius2 + k5 * radius2 * radius2 + k6 * radius2 * radius2 * radius2
    scale = numerator / np.maximum(denominator, 1e-12)
    raw_x = float(fx) * (x * scale + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)) + float(raw_cx)
    raw_y = float(fy) * (y * scale + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y) + float(raw_cy)
    # Preserve the CUDA tile renderer's genuine-border tolerance exactly.
    valid = (
        np.isfinite(raw_x)
        & np.isfinite(raw_y)
        & (raw_x >= -1e-6)
        & (raw_x <= float(source_width - 1) + 1e-6)
        & (raw_y >= -1e-6)
        & (raw_y <= float(source_height - 1) + 1e-6)
    )
    return np.ascontiguousarray(raw_x), np.ascontiguousarray(raw_y), np.ascontiguousarray(valid)


def build_v2_c1_calibrated_inverse_sources(
    *,
    strips: Sequence[object],
    source_shapes: Mapping[int, tuple[int, int]],
    canvas_shape: tuple[int, int],
    calibration: Mapping[str, object],
    annotation_frame_ids: Sequence[int],
    corridor_width_pixels: int = 96,
    include_adjacent_corridors: bool = True,
) -> tuple[CandidateInverseMapSource, ...]:
    """Build C1's exact calibrated inverse maps for fixed annotations only.

    The output windows are the same owner-strip-plus-adjacent-corridor windows
    used by C1's actual CUDA render.  No window is inferred from final owner;
    final provenance is applied later by ``build_candidate_annotation_projection``.
    The narrow function is intentionally reusable by C2--C8: callers that
    introduce an accepted residual grid must supply their composed map rather
    than incorrectly using this nominal C1 reference for altered pixels.
    """

    height, canvas_width = (int(value) for value in canvas_shape)
    if height < 2 or canvas_width < 2:
        raise ValueError("v2 projection canvas shape must be at least 2x2")
    geometry = tuple(_coerce_v2_strip(strip) for strip in strips)
    if not geometry:
        raise ValueError("v2 projection requires one or more selected C1 strips")
    ids = tuple(item[0] for item in geometry)
    if len(set(ids)) != len(ids) or tuple(sorted(ids)) != ids:
        raise ValueError("v2 projection C1 strips must have unique chronological frame ids")
    if geometry[0][1] != 0 or geometry[-1][1] + geometry[-1][3] != canvas_width:
        raise ValueError("v2 projection strips do not cover the rendered canvas")
    for (_, left, _, width, _), (_, right, _, _, _) in zip(geometry[:-1], geometry[1:], strict=True):
        if left + width != right:
            raise ValueError("v2 projection C1 strips are not contiguous")
    for frame_id in ids:
        try:
            raw_height, raw_width = (int(value) for value in source_shapes[frame_id])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("v2 projection lacks a raw shape for a selected source") from exc
        if raw_height != height or raw_width < 2:
            raise ValueError("v2 projection raw source shape differs from its rendered C1 source")
    try:
        fx, fy = float(calibration["fx"]), float(calibration["fy"])
        raw_cx, raw_cy = float(calibration["cx"]), float(calibration["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v2 projection calibration is incomplete") from exc
    distortion_raw = calibration.get("distortion", ())
    if not isinstance(distortion_raw, Sequence) or isinstance(distortion_raw, (str, bytes)):
        raise ValueError("v2 projection distortion must be a numeric sequence")
    annotation_ids = {int(frame_id) for frame_id in annotation_frame_ids}
    corridors = (
        _c1_corridors(
            geometry, source_shapes, canvas_width=canvas_width,
            corridor_width_pixels=int(corridor_width_pixels),
        )
        if include_adjacent_corridors
        else ()
    )
    sources: list[CandidateInverseMapSource] = []
    for index, (frame_id, output_x0, source_x0, width, centre) in enumerate(geometry):
        if frame_id not in annotation_ids:
            continue
        starts, ends = [output_x0], [output_x0 + width]
        if include_adjacent_corridors:
            if index:
                starts.append(corridors[index - 1][0])
            if index < len(corridors):
                ends.append(corridors[index][1])
        raw_height, raw_width = (int(value) for value in source_shapes[frame_id])
        support = (output_x0 - source_x0, output_x0 - source_x0 + raw_width)
        window_x0 = max(min(starts), support[0], 0)
        window_x1 = min(max(ends), support[1], canvas_width)
        if window_x0 > min(starts) or window_x1 < max(ends) or window_x1 - window_x0 < 2:
            raise ValueError("v2 projection C1 source window differs from its genuine CUDA sampling window")
        map_x, map_y, valid = _brown_conrady_inverse_map(
            canvas_x0=window_x0, width=window_x1 - window_x0, height=height,
            source_width=raw_width, source_height=raw_height, source_centre_x=centre,
            fx=fx, fy=fy, raw_cx=raw_cx, raw_cy=raw_cy,
            distortion=tuple(float(value) for value in distortion_raw),
        )
        sources.append(CandidateInverseMapSource(
            frame_id=frame_id, canvas_x0=window_x0, source_map_x=map_x,
            source_map_y=map_y, valid_mask=valid, raw_shape=(raw_height, raw_width),
        ))
    return tuple(sources)


def _raw_mask(shape: tuple[int, int], points: Sequence[Sequence[float]], *, line: bool) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    values = np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)
    if line:
        cv2.line(result, tuple(values[0]), tuple(values[1]), 1, 1)
    else:
        cv2.fillPoly(result, [values], 1)
    return result.astype(bool)


def _project_raw_mask(source: CandidateInverseMapSource, raw_mask: np.ndarray) -> np.ndarray:
    map_x = np.asarray(source.source_map_x, dtype=np.float64)
    map_y = np.asarray(source.source_map_y, dtype=np.float64)
    valid = np.asarray(source.valid_mask, dtype=bool)
    height, width = raw_mask.shape
    usable = valid & np.isfinite(map_x) & np.isfinite(map_y) & (map_x >= 0.0) & (map_x < width) & (map_y >= 0.0) & (map_y < height)
    output = np.zeros(valid.shape, dtype=bool)
    if np.any(usable):
        xs = np.rint(map_x[usable]).astype(np.int32)
        ys = np.rint(map_y[usable]).astype(np.int32)
        output[usable] = raw_mask[ys, xs]
    return output


def _to_final_canvas(
    tile_mask: np.ndarray,
    source: CandidateInverseMapSource,
    *,
    crop_xywh: tuple[int, int, int, int],
    final_owner: np.ndarray,
    horizontal_flip: bool,
) -> np.ndarray:
    crop_x, crop_y, crop_width, crop_height = crop_xywh
    full = np.zeros((crop_height, crop_width), dtype=bool)
    tile_y0 = max(0, crop_y)
    tile_y1 = min(crop_y + crop_height, tile_mask.shape[0])
    tile_x0 = max(crop_x, int(source.canvas_x0))
    tile_x1 = min(crop_x + crop_width, int(source.canvas_x0) + tile_mask.shape[1])
    if tile_y1 > tile_y0 and tile_x1 > tile_x0:
        full[tile_y0 - crop_y : tile_y1 - crop_y, tile_x0 - crop_x : tile_x1 - crop_x] = tile_mask[
            tile_y0:tile_y1, tile_x0 - int(source.canvas_x0) : tile_x1 - int(source.canvas_x0)
        ]
    if horizontal_flip:
        full = full[:, ::-1]
    if full.shape != final_owner.shape:
        raise ValueError("Candidate projection crop does not match final owner map")
    return full & (final_owner == int(source.frame_id))


def _line_points(mask: np.ndarray) -> list[list[float]] | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        return None
    direction = cv2.fitLine(np.column_stack((xs, ys)).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = (float(value) for value in direction)
    projection = (xs - x0) * vx + (ys - y0) * vy
    p0 = np.asarray([x0, y0]) + float(np.min(projection)) * np.asarray([vx, vy])
    p1 = np.asarray([x0, y0]) + float(np.max(projection)) * np.asarray([vx, vy])
    distance = np.abs((xs - x0) * vy - (ys - y0) * vx)
    # Curved/fragmented source-to-panorama line maps are not reduced to a
    # misleading straight reference line.
    if float(np.percentile(distance, 95.0)) > 0.75:
        return None
    return [[float(p0[0]), float(p0[1])], [float(p1[0]), float(p1[1])]]


def build_candidate_annotation_projection(
    annotations: Mapping[str, Any],
    *,
    sources: Sequence[CandidateInverseMapSource],
    final_owner_frame_id: np.ndarray,
    crop_xywh: tuple[int, int, int, int],
    horizontal_flip: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build a measurement-only projection and boolean panorama masks.

    Entries absent from the candidate's real sources, cropped away, or not
    owned by their annotated source are omitted with an audit reason.
    """

    owner = np.asarray(final_owner_frame_id)
    if owner.ndim != 2 or not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("Candidate projection requires a two-dimensional integer final owner map")
    source_by_id = {int(source.frame_id): source for source in sources}
    payload: dict[str, Any] = {
        "schema": PANORAMA_PROJECTION_SCHEMA,
        "measurement_only": True,
        "projection_method": "candidate_calibrated_inverse_map_owner_filtered",
        "panorama_shape": [int(owner.shape[0]), int(owner.shape[1])],
        "objects": [], "lines": [], "safe_background": [], "omitted": [],
    }
    masks: dict[str, np.ndarray] = {}
    for kind in ("objects", "lines", "safe_background"):
        entries = annotations.get(kind, [])
        if not isinstance(entries, list):
            raise ValueError(f"Fixed annotations lack {kind}")
        for entry in entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("frame_id"), int):
                raise ValueError(f"Fixed {kind} annotation is malformed")
            identifier, frame_id = str(entry["id"]), int(entry["frame_id"])
            measurement_group = entry.get("measurement_group")
            if measurement_group is not None and (
                not isinstance(measurement_group, str) or not measurement_group.strip()
            ):
                raise ValueError(f"Fixed {kind} annotation measurement_group must be a non-empty string when present")
            source = source_by_id.get(frame_id)
            if source is None:
                payload["omitted"].append({"id": identifier, "kind": kind, "frame_id": frame_id, "reason": "annotated_source_not_a_candidate_render_source"})
                continue
            points = entry.get("points" if kind == "lines" else "polygon")
            if not isinstance(points, list):
                raise ValueError(f"Fixed {kind} annotation lacks its coordinates")
            local = _project_raw_mask(source, _raw_mask(source.raw_shape, points, line=kind == "lines"))
            projected = _to_final_canvas(local, source, crop_xywh=crop_xywh, final_owner=owner, horizontal_flip=horizontal_flip)
            if not np.any(projected):
                payload["omitted"].append({"id": identifier, "kind": kind, "frame_id": frame_id, "reason": "no_final_owned_projection_pixels"})
                continue
            if kind == "lines":
                fitted = _line_points(projected)
                if fitted is None:
                    payload["omitted"].append({"id": identifier, "kind": kind, "frame_id": frame_id, "reason": "projected_line_not_reliably_straight"})
                    continue
                item: dict[str, Any] = {"id": identifier, "frame_id": frame_id, "points": fitted}
                if measurement_group is not None:
                    item["measurement_group"] = measurement_group
                payload[kind].append(item)
            else:
                key = f"{kind}__{identifier}"
                masks[key] = projected
                item = {"id": identifier, "frame_id": frame_id, "mask_key": key}
                if measurement_group is not None:
                    item["measurement_group"] = measurement_group
                payload[kind].append(item)
    return payload, masks


def write_candidate_annotation_projection_sidecar(
    output_json: str | Path,
    payload: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
) -> tuple[Path, Path]:
    """Atomically write non-primary JSON/NPZ evidence after candidate publish."""

    destination = Path(output_json).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    mask_path = destination.with_name(f"{destination.stem}_masks.npz")
    serialised = dict(payload)
    serialised["mask_artifact"] = mask_path.name
    pending_json = destination.with_name(f".{destination.name}.pending")
    pending_masks = mask_path.with_name(f".{mask_path.stem}.pending.npz")
    try:
        np.savez_compressed(pending_masks, **{key: np.asarray(value, dtype=bool) for key, value in masks.items()})
        pending_json.write_text(json.dumps(serialised, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(pending_masks, mask_path)
        os.replace(pending_json, destination)
    finally:
        pending_json.unlink(missing_ok=True)
        pending_masks.unlink(missing_ok=True)
    return destination, mask_path


__all__ = [
    "CandidateInverseMapSource",
    "build_candidate_annotation_projection",
    "build_v2_c1_calibrated_inverse_sources",
    "write_candidate_annotation_projection_sidecar",
]
