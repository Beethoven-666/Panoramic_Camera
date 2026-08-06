"""Candidate-only fixed-annotation projection from calibrated inverse maps.

This is deliberately a measurement adapter, not a renderer.  It accepts the
already-built calibrated inverse coordinate maps and rasterises fixed source
annotations through those maps.  It deliberately does *not* filter a source
projection by final owner provenance: provenance is the subject of an object
measurement, not permission for its fixed source annotation to exist.  It
never reads RGB, changes a sampling grid, or makes an owner decision. Callers
write the returned payload only after primary publication.
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


def build_v2_full_support_measurement_sources(
    *,
    strips: Sequence[object],
    source_shapes: Mapping[int, tuple[int, int]],
    canvas_shape: tuple[int, int],
    calibration: Mapping[str, object],
    annotation_frame_ids: Sequence[int],
    final_grid_updates: Sequence[Mapping[str, object]] = (),
) -> tuple[CandidateInverseMapSource, ...]:
    """Build post-publication, full-support inverse maps for fixed labels.

    Unlike :func:`build_v2_c1_calibrated_inverse_sources`, this adapter does
    not restrict an annotated source to its final owner strip or a seam
    corridor.  Each annotated real source is represented over the entire
    final canvas and the calibrated inverse map itself supplies the only
    support mask.  Consequently an object such as a fan or cable remains
    measurable even when a later hard-owner decision assigns its projected
    pixels to another source.

    ``strips`` is still required as immutable layout evidence: it binds each
    real source centre to the final canvas.  ``final_grid_updates`` are the
    renderer's already-audited local inverse-grid replacements and are
    applied only after the nominal calibrated map is built.  This function is
    measurement-only; it neither reads RGB nor uses owner provenance.
    """

    height, canvas_width = (int(value) for value in canvas_shape)
    if height < 2 or canvas_width < 2:
        raise ValueError("v2 full-support projection canvas shape must be at least 2x2")
    geometry = tuple(_coerce_v2_strip(strip) for strip in strips)
    if not geometry:
        raise ValueError("v2 full-support projection requires selected source layout")
    ids = tuple(item[0] for item in geometry)
    if len(set(ids)) != len(ids) or tuple(sorted(ids)) != ids:
        raise ValueError("v2 full-support projection strips must have unique chronological frame ids")
    if geometry[0][1] != 0 or geometry[-1][1] + geometry[-1][3] != canvas_width:
        raise ValueError("v2 full-support projection strips do not cover the rendered canvas")
    for (_, left, _, width, _), (_, right, _, _, _) in zip(geometry[:-1], geometry[1:], strict=True):
        if left + width != right:
            raise ValueError("v2 full-support projection strips are not contiguous")
    try:
        fx, fy = float(calibration["fx"]), float(calibration["fy"])
        raw_cx, raw_cy = float(calibration["cx"]), float(calibration["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("v2 full-support projection calibration is incomplete") from exc
    distortion_raw = calibration.get("distortion", ())
    if not isinstance(distortion_raw, Sequence) or isinstance(distortion_raw, (str, bytes)):
        raise ValueError("v2 full-support projection distortion must be a numeric sequence")
    try:
        distortion = tuple(float(value) for value in distortion_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("v2 full-support projection distortion must be numeric") from exc
    annotation_ids = {int(frame_id) for frame_id in annotation_frame_ids}
    sources: list[CandidateInverseMapSource] = []
    for frame_id, _, _, _, centre in geometry:
        if frame_id not in annotation_ids:
            continue
        try:
            raw_height, raw_width = (int(value) for value in source_shapes[frame_id])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("v2 full-support projection lacks a raw shape for a selected source") from exc
        if raw_height != height or raw_width < 2:
            raise ValueError("v2 full-support source shape differs from the rendered real source")
        map_x, map_y, valid = _brown_conrady_inverse_map(
            canvas_x0=0,
            width=canvas_width,
            height=height,
            source_width=raw_width,
            source_height=raw_height,
            source_centre_x=centre,
            fx=fx,
            fy=fy,
            raw_cx=raw_cx,
            raw_cy=raw_cy,
            distortion=distortion,
        )
        sources.append(
            CandidateInverseMapSource(
                frame_id=frame_id,
                canvas_x0=0,
                source_map_x=map_x,
                source_map_y=map_y,
                valid_mask=valid,
                raw_shape=(raw_height, raw_width),
            )
        )
    if not sources and annotation_ids:
        raise ValueError("v2 full-support projection has no annotated selected real source")
    return apply_final_grid_updates(tuple(sources), final_grid_updates)


def apply_final_grid_updates(
    sources: Sequence[CandidateInverseMapSource],
    updates: Sequence[Mapping[str, object]],
) -> tuple[CandidateInverseMapSource, ...]:
    """Apply audited final inverse-grid deltas to measurement source maps.

    C2--C4 may replace a subset of a C1 corridor with an accepted local
    inverse sample.  This function reproduces that *already rendered* final
    grid for annotation projection after primary publication.  It neither
    looks at an annotation while rendering nor uses final ownership to decide
    whether a source annotation exists.
    """

    mutable = {
        int(source.frame_id): [
            np.asarray(source.source_map_x, dtype=np.float64).copy(),
            np.asarray(source.source_map_y, dtype=np.float64).copy(),
            np.asarray(source.valid_mask, dtype=bool).copy(),
            source,
        ]
        for source in sources
    }
    for update in updates:
        if not isinstance(update, Mapping):
            raise ValueError("final measurement grid update must be an object")
        try:
            frame_id = int(update["frame_id"])
            canvas_x0 = int(update["canvas_x0"])
            normalized = np.asarray(update["normalized_grid_xy"], dtype=np.float64)
            applied = np.asarray(update["applied_mask"], dtype=bool)
            raw_shape = tuple(int(value) for value in update["source_shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("final measurement grid update is malformed") from exc
        if len(raw_shape) != 2 or min(raw_shape) < 2 or normalized.ndim != 3 or normalized.shape[-1] != 2:
            raise ValueError("final measurement grid update has invalid source/grid shape")
        if normalized.shape[:2] != applied.shape:
            raise ValueError("final measurement grid update mask does not match its grid")
        target = mutable.get(frame_id)
        if target is None:
            # The annotated fixed sources are a strict subset of candidate
            # sources.  An update for an unannotated source has no evidence
            # to transform and is therefore deliberately ignored.
            continue
        map_x, map_y, valid, source = target
        if tuple(source.raw_shape) != raw_shape:
            raise ValueError("final measurement grid update raw shape differs from its source map")
        # A final joint-owner solve may carry one full-canvas source grid,
        # whereas a fixed annotation source map deliberately retains only the
        # original owner strip plus its bounded corridors.  Intersect the
        # already-rendered grid evidence with that read-only map rather than
        # rejecting a valid update merely because it also covers unannotated
        # canvas columns.  No missing coordinate is extrapolated.
        if normalized.shape[0] != map_x.shape[0]:
            raise ValueError("final measurement grid update height differs from its source map window")
        update_x0, update_x1 = canvas_x0, canvas_x0 + int(normalized.shape[1])
        map_canvas_x0, map_canvas_x1 = int(source.canvas_x0), int(source.canvas_x0) + map_x.shape[1]
        intersect_x0, intersect_x1 = max(update_x0, map_canvas_x0), min(update_x1, map_canvas_x1)
        if intersect_x1 <= intersect_x0:
            continue
        update_left = intersect_x0 - update_x0
        update_right = update_left + (intersect_x1 - intersect_x0)
        local_x0 = intersect_x0 - map_canvas_x0
        local_x1 = local_x0 + (intersect_x1 - intersect_x0)
        normalized = normalized[:, update_left:update_right]
        applied = applied[:, update_left:update_right]
        finite = np.isfinite(normalized).all(axis=-1)
        inside = (
            (normalized[..., 0] >= -1.0 - 1e-6)
            & (normalized[..., 0] <= 1.0 + 1e-6)
            & (normalized[..., 1] >= -1.0 - 1e-6)
            & (normalized[..., 1] <= 1.0 + 1e-6)
        )
        replace = applied & finite & inside
        if not np.any(replace):
            continue
        raw_height, raw_width = raw_shape
        region_x = map_x[:, local_x0:local_x1]
        region_y = map_y[:, local_x0:local_x1]
        region_valid = valid[:, local_x0:local_x1]
        region_x[replace] = (normalized[..., 0][replace] + 1.0) * ((raw_width - 1) * 0.5)
        region_y[replace] = (normalized[..., 1][replace] + 1.0) * ((raw_height - 1) * 0.5)
        region_valid[replace] = True
    return tuple(
        CandidateInverseMapSource(
            frame_id=int(frame_id),
            canvas_x0=int(source.canvas_x0),
            source_map_x=np.ascontiguousarray(map_x),
            source_map_y=np.ascontiguousarray(map_y),
            valid_mask=np.ascontiguousarray(valid),
            raw_shape=source.raw_shape,
        )
        for frame_id, (map_x, map_y, valid, source) in mutable.items()
    )


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
    usable = valid & np.isfinite(map_x) & np.isfinite(map_y)
    output = np.zeros(valid.shape, dtype=bool)
    if np.any(usable):
        xs = np.rint(map_x[usable]).astype(np.int32)
        ys = np.rint(map_y[usable]).astype(np.int32)
        nearest_inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        positions = np.flatnonzero(usable)
        output.flat[positions[nearest_inside]] = raw_mask[ys[nearest_inside], xs[nearest_inside]]
    return output


def _to_final_canvas(
    tile_mask: np.ndarray,
    source: CandidateInverseMapSource,
    *,
    crop_xywh: tuple[int, int, int, int],
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
    return full


def _dense_line_points(mask: np.ndarray) -> list[list[float]] | None:
    """Turn a source-grid projection into an ordered dense output polyline.

    The projected source raster is the nearest-neighbour equivalent of
    ``grid_sample``.  Fitting just its two end points loses a local mesh bend
    and makes the evaluator search the wrong edge.  PCA ordering preserves the
    source-to-output locus while the per-bin median removes raster thickness.
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 8:
        return None
    direction = cv2.fitLine(np.column_stack((xs, ys)).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    vx, vy, x0, y0 = (float(value) for value in direction)
    projection = (xs - x0) * vx + (ys - y0) * vy
    # One representative point per approximately-pixel longitudinal bin.
    bins = np.rint(projection - float(np.min(projection))).astype(np.int32)
    points: list[list[float]] = []
    for value in np.unique(bins):
        selected = bins == value
        points.append([float(np.median(xs[selected])), float(np.median(ys[selected]))])
    return points if len(points) >= 2 else None


def _mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, int]:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return np.asarray([np.nan, np.nan], dtype=np.float64), 0
    return np.asarray([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64), int(xs.size)


def _consensus_masks(masks: Sequence[np.ndarray]) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
    """Build an owner-independent, auditable group consensus mask.

    Pairs must agree by IoU or centre/area geometry.  The consensus itself is
    a one-pixel-dilated intersection where it has useful coverage; otherwise
    a recorded union is retained only after geometric agreement.  This avoids
    silently treating a missing projection as visual failure.
    """
    if not masks:
        return False, None, {"measurement_state": "projection_missing"}
    geometry = []
    for index, mask in enumerate(masks):
        centroid, area = _mask_geometry(mask)
        geometry.append(
            {
                "source_projection_index": int(index),
                "area_pixels": int(area),
                "centroid_xy": [float(centroid[0]), float(centroid[1])],
            }
        )
    if len(masks) == 1:
        return True, masks[0], {
            "measurement_state": "evaluated", "strategy": "single_source", "source_projection_count": 1,
            "source_projection_geometry": geometry,
            "consensus_area_pixels": int(np.count_nonzero(masks[0])),
            "consensus_coverage_of_union": 1.0,
            "consensus_coverage_of_smaller_projection": 1.0,
        }
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = [cv2.dilate(item.astype(np.uint8), kernel, iterations=1).astype(bool) for item in masks]
    intersection = np.logical_and.reduce(dilated)
    union = np.logical_or.reduce(dilated)
    intersection_area, union_area = int(np.count_nonzero(intersection)), int(np.count_nonzero(union))
    iou = float(intersection_area / union_area) if union_area else 0.0
    centres, areas = zip(*(_mask_geometry(item) for item in masks), strict=True)
    centre_distance = max(float(np.linalg.norm(first - second)) for index, first in enumerate(centres) for second in centres[index + 1 :])
    min_area, max_area = min(areas), max(areas)
    area_ratio = float(min_area / max_area) if max_area else 0.0
    pairwise: list[dict[str, Any]] = []
    for first_index, first in enumerate(dilated):
        for second_index, second in enumerate(dilated[first_index + 1 :], start=first_index + 1):
            pair_intersection = int(np.count_nonzero(first & second))
            pair_union = int(np.count_nonzero(first | second))
            first_area = int(areas[first_index])
            second_area = int(areas[second_index])
            pairwise.append(
                {
                    "first_source_projection_index": int(first_index),
                    "second_source_projection_index": int(second_index),
                    "dilated_intersection_area_pixels": pair_intersection,
                    "dilated_union_area_pixels": pair_union,
                    "dilated_iou": float(pair_intersection / pair_union) if pair_union else 0.0,
                    "centroid_distance_px": float(np.linalg.norm(centres[first_index] - centres[second_index])),
                    "area_ratio": float(min(first_area, second_area) / max(first_area, second_area)) if max(first_area, second_area) else 0.0,
                }
            )
    consistent = bool(iou >= 0.70 or (centre_distance <= 3.0 and 0.80 <= area_ratio <= 1.25))
    audit: dict[str, Any] = {
        "measurement_state": "evaluated" if consistent else "projection_inconsistent",
        "source_projection_count": len(masks), "dilated_iou": iou,
        "maximum_center_distance_px": centre_distance, "area_ratio": area_ratio,
        "dilated_intersection_area_pixels": intersection_area,
        "dilated_union_area_pixels": union_area,
        "intersection_coverage_of_smaller_projection": float(intersection_area / min_area) if min_area else 0.0,
        "intersection_coverage_of_union": float(intersection_area / union_area) if union_area else 0.0,
        "source_projection_geometry": geometry,
        "pairwise_projection_agreement": pairwise,
    }
    if not consistent:
        return False, None, audit
    coverage = float(intersection_area / min_area) if min_area else 0.0
    if intersection_area and coverage >= 0.50:
        audit.update({
            "strategy": "dilate_1px_intersection",
            "coverage_of_smaller_projection": coverage,
            "consensus_area_pixels": intersection_area,
            "consensus_coverage_of_union": float(intersection_area / union_area) if union_area else 0.0,
        })
        return True, intersection, audit
    audit.update({
        "strategy": "geometrically_consistent_union",
        "coverage_of_smaller_projection": coverage,
        "consensus_area_pixels": union_area,
        "consensus_coverage_of_union": 1.0,
    })
    return True, union, audit


def build_candidate_annotation_projection(
    annotations: Mapping[str, Any],
    *,
    sources: Sequence[CandidateInverseMapSource],
    final_owner_frame_id: np.ndarray,
    crop_xywh: tuple[int, int, int, int],
    horizontal_flip: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build a measurement-only projection and boolean panorama masks.

    Entries absent from a candidate's real sources or cropped away are omitted
    with an audit reason. ``final_owner_frame_id`` is validated only for final
    panorama shape: it must never erase a fixed source projection.
    """

    owner = np.asarray(final_owner_frame_id)
    if owner.ndim != 2 or not np.issubdtype(owner.dtype, np.integer):
        raise ValueError("Candidate projection requires a two-dimensional integer final owner map")
    crop_x, crop_y, crop_width, crop_height = (int(value) for value in crop_xywh)
    if crop_width < 1 or crop_height < 1 or owner.shape != (crop_height, crop_width):
        raise ValueError("Candidate projection crop does not match final panorama shape")
    source_by_id = {int(source.frame_id): source for source in sources}
    payload: dict[str, Any] = {
        "schema": PANORAMA_PROJECTION_SCHEMA,
        "measurement_only": True,
        "projection_method": "candidate_calibrated_inverse_map_owner_independent_consensus",
        "panorama_shape": [int(owner.shape[0]), int(owner.shape[1])],
        "objects": [], "lines": [], "safe_background": [], "omitted": [], "measurement_groups": [],
    }
    masks: dict[str, np.ndarray] = {}
    for kind in ("objects", "lines", "safe_background"):
        entries = annotations.get(kind, [])
        if not isinstance(entries, list):
            raise ValueError(f"Fixed annotations lack {kind}")
        grouped: dict[str, list[tuple[Mapping[str, Any], CandidateInverseMapSource, np.ndarray]]] = {}
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
            projected = _to_final_canvas(local, source, crop_xywh=crop_xywh, horizontal_flip=horizontal_flip)
            if not np.any(projected):
                payload["omitted"].append({"id": identifier, "kind": kind, "frame_id": frame_id, "reason": "no_panorama_projection_for_fixed_source_annotation"})
                continue
            key = measurement_group if measurement_group is not None else identifier
            grouped.setdefault(key, []).append((entry, source, projected))
        for group_id, members in grouped.items():
            accepted, consensus, audit = _consensus_masks([item[2] for item in members])
            audit.update({
                "kind": kind,
                "measurement_group": group_id,
                "source_annotation_ids": [str(item[0]["id"]) for item in members],
                "source_projection_frame_ids": [int(item[1].frame_id) for item in members],
            })
            payload["measurement_groups"].append(audit)
            if not accepted or consensus is None:
                continue
            consensus_key = f"{kind}__consensus__{group_id}"
            masks[consensus_key] = consensus
            for entry, source, projected in members:
                identifier, frame_id = str(entry["id"]), int(entry["frame_id"])
                source_key = f"{kind}__source_projected__{identifier}"
                masks[source_key] = projected
                item: dict[str, Any] = {"id": identifier, "frame_id": frame_id, "source_projected_mask_key": source_key}
                if kind == "lines":
                    dense = _dense_line_points(consensus)
                    if dense is None:
                        payload["omitted"].append({"id": identifier, "kind": kind, "frame_id": frame_id, "reason": "projected_line_insufficient_dense_samples"})
                        continue
                    item["points"] = dense
                else:
                    item["mask_key"] = consensus_key
                if entry.get("measurement_group") is not None:
                    item["measurement_group"] = entry["measurement_group"]
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
    "apply_final_grid_updates",
    "build_v2_c1_calibrated_inverse_sources",
    "build_v2_full_support_measurement_sources",
    "write_candidate_annotation_projection_sidecar",
]
