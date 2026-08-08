"""Read-only visual evaluation for an already-published video panorama.

The immutable manual annotations describe *source-frame* coordinates.  A
published panorama deliberately contains only its RGB pixels and provenance
owner IDs, not a reverse source-pixel map.  Consequently this module never
pretends that a source polygon can be projected from an owner ID alone.
Object, line, and safe-background measurements require an explicit, immutable
panorama projection sidecar.  Missing projections are reported as
``not_evaluable`` rather than guessed.  Nothing here imports or calls a
renderer, and its output is advisory evidence only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .video_annotations import (
    ANNOTATION_SCHEMA,
    SUPPORTED_ANNOTATION_SCHEMAS,
    VideoAnnotationError,
    load_source_annotations,
)
from .video_observability import owner_boundaries
from .video_visual_metrics import owner_topology_metrics


OFFLINE_EVALUATION_SCHEMA = "gemini305-video-offline-visual-evaluation/v2"
PANORAMA_PROJECTION_SCHEMA = "gemini305-video-panorama-annotation-projection/v2"
_LEGACY_PANORAMA_PROJECTION_SCHEMA = "gemini305-video-panorama-annotation-projection/v1"


class VideoOfflineEvaluationError(ValueError):
    """Raised for malformed read-only measurement inputs."""


def _require_panorama_and_owner(panorama_bgr: np.ndarray, owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    panorama = np.asarray(panorama_bgr)
    owners = np.asarray(owner)
    if panorama.dtype != np.uint8 or panorama.ndim != 3 or panorama.shape[2] != 3:
        raise VideoOfflineEvaluationError("panorama must be an 8-bit BGR image")
    if owners.ndim != 2 or owners.shape != panorama.shape[:2] or not np.issubdtype(owners.dtype, np.integer):
        raise VideoOfflineEvaluationError("owner must be an integer map matching the panorama")
    return panorama, owners.astype(np.int32, copy=False)


def _entry_points(entry: Mapping[str, Any], kind: str) -> list[list[float]]:
    field = "points" if kind == "lines" else "polygon"
    points = entry.get(field)
    minimum = 2 if kind == "lines" else 3
    if not isinstance(points, list) or len(points) < minimum:
        raise VideoOfflineEvaluationError(f"Projection {kind} entry needs {field}")
    out: list[list[float]] = []
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in point)
        ):
            raise VideoOfflineEvaluationError(f"Projection {kind} has an invalid point")
        out.append([float(point[0]), float(point[1])])
    return out


def load_panorama_annotation_projection(
    path: str | Path,
    *,
    annotations: Mapping[str, Any],
    panorama_shape: tuple[int, int],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Load a source-annotation-preserving, panorama-coordinate sidecar.

    It may omit annotations that are outside a split or not visible.  It must
    not invent annotation IDs, alter their source frame, or contain coordinates
    outside the evaluated panorama.
    """

    projection_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(projection_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoOfflineEvaluationError(f"Invalid panorama projection: {projection_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") not in {
        PANORAMA_PROJECTION_SCHEMA, _LEGACY_PANORAMA_PROJECTION_SCHEMA,
    }:
        raise VideoOfflineEvaluationError("Unsupported panorama annotation projection schema")
    if payload.get("measurement_only") is not True:
        raise VideoOfflineEvaluationError("Panorama annotation projection must declare measurement_only=true")
    declared_shape = payload.get("panorama_shape")
    if declared_shape != [int(panorama_shape[0]), int(panorama_shape[1])]:
        raise VideoOfflineEvaluationError("Projection panorama_shape does not match the published panorama")

    source_by_kind: dict[str, dict[str, Mapping[str, Any]]] = {}
    for kind in ("objects", "lines", "safe_background"):
        entries = annotations.get(kind)
        if not isinstance(entries, list):
            raise VideoOfflineEvaluationError(f"Source annotations lack {kind}")
        source_by_kind[kind] = {str(entry.get("id")): entry for entry in entries if isinstance(entry, Mapping)}

    height, width = panorama_shape
    mask_artifact = payload.get("mask_artifact")
    loaded_masks: Mapping[str, np.ndarray] = {}
    if mask_artifact is not None:
        if not isinstance(mask_artifact, str) or Path(mask_artifact).name != mask_artifact:
            raise VideoOfflineEvaluationError("Projection mask_artifact must be a filename in the projection directory")
        try:
            with np.load(projection_path.parent / mask_artifact, allow_pickle=False) as archive:
                loaded_masks = {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError) as exc:
            raise VideoOfflineEvaluationError("Projection mask_artifact is unavailable or malformed") from exc
    result: dict[str, dict[str, dict[str, Any]]] = {kind: {} for kind in source_by_kind}
    # v2 writes group-level projection consensus separately from the source
    # entries.  It is measurement metadata, never a rendering control.
    group_states: dict[str, dict[str, Any]] = {kind: {} for kind in source_by_kind}
    declared_groups = payload.get("measurement_groups", [])
    if declared_groups is not None:
        if not isinstance(declared_groups, list):
            raise VideoOfflineEvaluationError("Projection measurement_groups must be a list")
        for item in declared_groups:
            if not isinstance(item, Mapping):
                raise VideoOfflineEvaluationError("Projection measurement group must be an object")
            kind, identifier, state = item.get("kind"), item.get("measurement_group"), item.get("measurement_state")
            if kind not in group_states or not isinstance(identifier, str) or not identifier:
                raise VideoOfflineEvaluationError("Projection measurement group is malformed")
            if state not in {"evaluated", "projection_inconsistent", "projection_missing"}:
                raise VideoOfflineEvaluationError("Projection measurement group has an invalid state")
            if identifier in group_states[kind]:
                raise VideoOfflineEvaluationError("Projection repeats a measurement group")
            group_states[kind][identifier] = dict(item)
    for kind, known in source_by_kind.items():
        projected_entries = payload.get(kind, [])
        if not isinstance(projected_entries, list):
            raise VideoOfflineEvaluationError(f"Projection {kind} must be a list")
        for projected in projected_entries:
            if not isinstance(projected, Mapping):
                raise VideoOfflineEvaluationError(f"Projection {kind} entry must be an object")
            identifier = projected.get("id")
            if not isinstance(identifier, str) or identifier not in known:
                raise VideoOfflineEvaluationError(f"Projection {kind} references an unknown annotation")
            if identifier in result[kind]:
                raise VideoOfflineEvaluationError(f"Projection repeats annotation {identifier!r}")
            source = known[identifier]
            if projected.get("frame_id") != source.get("frame_id"):
                raise VideoOfflineEvaluationError(f"Projection frame_id differs from fixed annotation {identifier!r}")
            source_measurement_group = source.get("measurement_group")
            projected_measurement_group = projected.get("measurement_group")
            if source_measurement_group != projected_measurement_group:
                raise VideoOfflineEvaluationError(
                    f"Projection measurement_group differs from fixed annotation {identifier!r}"
                )
            if source_measurement_group is not None and (
                not isinstance(source_measurement_group, str) or not source_measurement_group.strip()
            ):
                raise VideoOfflineEvaluationError(
                    f"Fixed annotation {identifier!r} has an invalid measurement_group"
                )
            mask_key = projected.get("mask_key")
            if kind != "lines" and isinstance(mask_key, str):
                mask = loaded_masks.get(mask_key)
                if mask is None or mask.shape != panorama_shape or mask.dtype != bool:
                    raise VideoOfflineEvaluationError(f"Projection {identifier!r} has no matching boolean panorama mask")
                result[kind][identifier] = {
                    "frame_id": int(source["frame_id"]),
                    "mask": mask,
                    "measurement_group": source_measurement_group,
                }
            else:
                points = _entry_points(projected, kind)
                if any(not (0.0 <= x < width and 0.0 <= y < height) for x, y in points):
                    raise VideoOfflineEvaluationError(f"Projection {identifier!r} lies outside panorama bounds")
                result[kind][identifier] = {
                    "frame_id": int(source["frame_id"]),
                    "points": points,
                    "measurement_group": source_measurement_group,
                }
    # Keep metadata out of the three annotation namespaces, preserving the
    # historical mapping API for callers that only consume source entries.
    result["__measurement_groups__"] = group_states
    return result


def _polygon_mask(shape: tuple[int, int], points: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(np.asarray(points, dtype=np.float64)).astype(np.int32)], 1)
    return mask.astype(bool)


def _region_mask(entry: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray:
    mask = entry.get("mask")
    if mask is not None:
        value = np.asarray(mask, dtype=bool)
        if value.shape != shape:
            raise VideoOfflineEvaluationError("Projected annotation mask has the wrong panorama shape")
        return value
    return _polygon_mask(shape, entry["points"])


def _percentile(values: np.ndarray, q: float) -> float | None:
    return float(np.percentile(values, q)) if values.size else None


def _object_metrics(
    owner: np.ndarray,
    mask: np.ndarray,
    *,
    role: str = "compact_foreground_single_owner",
) -> dict[str, Any]:
    valid = mask & (owner >= 0)
    owners = owner[valid]
    if owners.size == 0:
        return {"status": "not_evaluable", "reason": "no_owned_pixels_in_projected_object"}
    ids, counts = np.unique(owners, return_counts=True)
    dominant_index = int(np.argmax(counts))
    # A two-sided owner boundary at the *perimeter* of an annotated object is
    # not an object-internal seam.  Count only discontinuity edges for which
    # both pixels belong to the owned annotated region, then mark both pixels
    # so the connected-component audit retains its historical semantics.
    horizontal = (
        valid[:, :-1]
        & valid[:, 1:]
        & (owner[:, :-1] != owner[:, 1:])
    )
    vertical = (
        valid[:-1, :]
        & valid[1:, :]
        & (owner[:-1, :] != owner[1:, :])
    )
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical
    count, _, _, _ = cv2.connectedComponentsWithStats(boundary.astype(np.uint8), connectivity=8)
    handoffs = [int(np.count_nonzero(row)) for row in horizontal]
    single_owner_required = role != "extended_background_structure"
    seam_gate_pass = bool(max(handoffs, default=0) <= 1 and count <= 1)
    return {
        "status": "evaluated",
        "annotation_role": role,
        "single_owner_required": single_owner_required,
        "valid_pixel_count": int(owners.size),
        "owner_count": int(ids.size),
        "dominant_owner_frame_id": int(ids[dominant_index]),
        "dominant_owner_fraction": float(counts[dominant_index] / owners.size),
        "object_internal_seam_count": int(max(count - 1, 0)),
        "object_internal_seam_pixel_count": int(np.count_nonzero(boundary)),
        "maximum_handoffs": int(max(handoffs, default=0)),
        # Long background structure may have disjoint, separately-owned
        # portions.  It still cannot contain an internal seam or excessive
        # handoffs.  Compact foreground remains exactly owner=1.
        "hard_gate_pass": bool((not single_owner_required or ids.size == 1) and seam_gate_pass),
    }


def _line_arclength_samples(
    points: Sequence[Sequence[float]], *, spacing_px: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return evenly spaced polyline positions and their local tangents.

    Projection v2 can contain a very dense, curved polyline.  Sampling every
    input segment makes the metric depend on that serialization density and,
    more importantly, changes the expected normal at every tiny join.  Work
    in arc length instead: one sample approximately every two panorama pixels
    and a tangent estimated from a two-pixel local chord.  The latter is local
    enough for a curved beam edge while avoiding the unstable one-pixel finite
    differences of a dense projection.
    """

    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] != 2 or not np.all(np.isfinite(raw)):
        raise VideoOfflineEvaluationError("Projected line needs at least two finite points")
    if not np.isfinite(spacing_px) or spacing_px <= 0.0:
        raise ValueError("spacing_px must be finite and positive")

    # Consecutive duplicate vertices have no tangent and otherwise make the
    # cumulative coordinate non-strictly increasing.
    retained = [raw[0]]
    for point in raw[1:]:
        if float(np.hypot(*(point - retained[-1]))) > 1e-6:
            retained.append(point)
    vertices = np.asarray(retained, dtype=np.float64)
    if vertices.shape[0] < 2:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    vectors = np.diff(vertices, axis=0)
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    cumulative = np.concatenate((np.asarray([0.0]), np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total < spacing_px:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    def position_at(distance: np.ndarray) -> np.ndarray:
        clipped = np.clip(distance, 0.0, total)
        index = np.searchsorted(cumulative, clipped, side="right") - 1
        index = np.clip(index, 0, lengths.size - 1)
        fraction = (clipped - cumulative[index]) / lengths[index]
        return vertices[index] + fraction[:, None] * vectors[index]

    stations = np.arange(0.0, total, float(spacing_px), dtype=np.float64)
    if stations.size == 0 or total - stations[-1] > 1e-6:
        stations = np.append(stations, total)
    positions = position_at(stations)
    # Use an arc-length chord centred on each sample (one-sided at endpoints).
    # The window stays in local geometry even when the overall annotation is a
    # long curve.
    half_window = float(spacing_px)
    before = position_at(np.maximum(0.0, stations - half_window))
    after = position_at(np.minimum(total, stations + half_window))
    tangent = after - before
    tangent_norm = np.hypot(tangent[:, 0], tangent[:, 1])
    usable = tangent_norm > 1e-6
    if not np.any(usable):
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return positions[usable], tangent[usable] / tangent_norm[usable, None]


def _line_observations(
    panorama: np.ndarray, points: Sequence[Sequence[float]], *, search_radius: int = 5
) -> dict[str, Any]:
    """Extract one line's read-only edge samples without creating a gate.

    Grouped measurements concatenate observations from their explicitly paired
    source annotations, but never take a finite difference across two
    different projected line segments.
    """

    positions, tangents = _line_arclength_samples(points)
    if positions.size == 0:
        return {
            "status": "not_evaluable", "reason": "projected_line_too_short",
            "sample_count": 0, "offsets": np.empty(0, dtype=np.float64),
            "steps": np.empty(0, dtype=np.float64), "orientation_error": np.empty(0, dtype=np.float64),
        }
    height, width = panorama.shape[:2]
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    offsets: list[float] = []
    orientation_error: list[float] = []
    sample_count = 0
    observed_station_indices: list[int] = []
    for station_index, (position, tangent) in enumerate(zip(positions, tangents, strict=True)):
        normal = np.asarray([-tangent[1], tangent[0]])
        sample_count += 1
        expected = float(np.degrees(np.arctan2(normal[1], normal[0])))
        candidate_offsets = np.arange(-search_radius, search_radius + 1, dtype=np.float64)
        candidates = position[None, :] + candidate_offsets[:, None] * normal[None, :]
        xs = np.rint(candidates[:, 0]).astype(np.int32)
        ys = np.rint(candidates[:, 1]).astype(np.int32)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not np.any(inside):
            continue
        local = magnitude[ys[inside], xs[inside]]
        local_offsets = candidate_offsets[inside]
        # A candidate must first be a plausible instance of the expected
        # normal.  This prevents a nearby vertical edge from masquerading as a
        # poor horizontal-line measurement.  This is deliberately a search
        # constraint, not a softened orientation gate: the accepted edge is
        # still measured against the immutable 3 degree hard gate below.
        order = np.argsort(local)[::-1]
        chosen: tuple[int, float] | None = None
        inside_xs, inside_ys = xs[inside], ys[inside]
        for peak in order:
            if float(local[peak]) <= 0.0:
                break
            x, y = int(inside_xs[peak]), int(inside_ys[peak])
            observed = float(np.degrees(np.arctan2(float(gy[y, x]), float(gx[y, x]))))
            difference = abs((observed - expected + 90.0) % 180.0 - 90.0)
            if difference < 30.0:
                chosen = (int(peak), difference)
                break
        if chosen is None:
            continue
        peak, difference = chosen
        offsets.append(float(local_offsets[peak]))
        orientation_error.append(difference)
        observed_station_indices.append(station_index)
    offset_array = np.asarray(offsets, dtype=np.float64)
    # A missed edge candidate must not manufacture a displacement between
    # distant parts of a line.  Only compare adjoining two-pixel stations.
    indices = np.asarray(observed_station_indices, dtype=np.int32)
    adjacent = np.diff(indices) == 1
    step = np.abs(np.diff(offset_array))[adjacent] if offset_array.size >= 2 else np.empty(0, dtype=np.float64)
    return {
        "status": "evaluated" if offset_array.size >= 3 else "not_evaluable",
        "sample_count": int(sample_count),
        "offsets": offset_array,
        "steps": step,
        "orientation_error": np.asarray(orientation_error, dtype=np.float64),
    }


def _line_metrics_from_observations(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate a single line or an explicit measurement group."""

    offsets = [np.asarray(item["offsets"], dtype=np.float64) for item in observations]
    steps = [np.asarray(item["steps"], dtype=np.float64) for item in observations]
    orientation = [np.asarray(item["orientation_error"], dtype=np.float64) for item in observations]
    offset_array = np.concatenate(offsets) if offsets else np.empty(0, dtype=np.float64)
    step = np.concatenate(steps) if steps else np.empty(0, dtype=np.float64)
    orientation_array = np.concatenate(orientation) if orientation else np.empty(0, dtype=np.float64)
    step_p95 = _percentile(step, 95.0)
    orientation_p95 = _percentile(orientation_array, 95.0)
    return {
        "status": "evaluated" if offset_array.size >= 3 else "not_evaluable",
        "sample_count": int(sum(int(item["sample_count"]) for item in observations)),
        "edge_sample_count": int(offset_array.size),
        "line_step_p95_px": step_p95,
        "line_offset_p95_px": _percentile(np.abs(offset_array), 95.0),
        "line_orientation_delta_p95_degrees": orientation_p95,
        "hard_gate_pass": bool(
            offset_array.size >= 3
            # These exact thresholds are the documented long-straight-line
            # hard gates.  Grouping changes only the source evidence pool;
            # it cannot relax either limit.
            and (step_p95 if step_p95 is not None else float("inf")) < 1.0
            and (orientation_p95 if orientation_p95 is not None else float("inf")) < 3.0
        ),
    }


def _line_metrics(panorama: np.ndarray, points: Sequence[Sequence[float]], *, search_radius: int = 5) -> dict[str, Any]:
    """Measure one projected line using the same aggregation code as groups."""

    observation = _line_observations(panorama, points, search_radius=search_radius)
    if observation["status"] == "not_evaluable" and observation.get("reason") is not None:
        return {"status": "not_evaluable", "reason": observation["reason"]}
    return _line_metrics_from_observations((observation,))


def _delta_e00(lab0: np.ndarray, lab1: np.ndarray) -> np.ndarray:
    """CIEDE2000 on continuous OpenCV Lab values (L 0..100, a/b signed)."""

    first, second = np.asarray(lab0, dtype=np.float64), np.asarray(lab1, dtype=np.float64)
    l1, a1, b1 = first[..., 0], first[..., 1], first[..., 2]
    l2, a2, b2 = second[..., 0], second[..., 1], second[..., 2]
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    mean_c = 0.5 * (c1 + c2)
    g = 0.5 * (1.0 - np.sqrt(mean_c**7 / (mean_c**7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.mod(np.degrees(np.arctan2(b1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(b2, a2p)), 360.0)
    hue_delta = h2p - h1p
    hue_delta = np.where(hue_delta > 180.0, hue_delta - 360.0, hue_delta)
    hue_delta = np.where(hue_delta < -180.0, hue_delta + 360.0, hue_delta)
    hue_delta = np.where((c1p * c2p) == 0.0, 0.0, hue_delta)
    delta_l, delta_c = l2 - l1, c2p - c1p
    delta_h = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(hue_delta / 2.0))
    mean_l, mean_cp = 0.5 * (l1 + l2), 0.5 * (c1p + c2p)
    mean_hp = 0.5 * (h1p + h2p)
    mean_hp = np.where(np.abs(h1p - h2p) > 180.0, mean_hp + 180.0, mean_hp)
    mean_hp = np.where((c1p * c2p) == 0.0, h1p + h2p, mean_hp)
    mean_hp = np.mod(mean_hp, 360.0)
    t = 1.0 - 0.17 * np.cos(np.radians(mean_hp - 30.0)) + 0.24 * np.cos(np.radians(2.0 * mean_hp)) + 0.32 * np.cos(np.radians(3.0 * mean_hp + 6.0)) - 0.20 * np.cos(np.radians(4.0 * mean_hp - 63.0))
    delta_theta = 30.0 * np.exp(-np.square((mean_hp - 275.0) / 25.0))
    rc = 2.0 * np.sqrt(mean_cp**7 / (mean_cp**7 + 25.0**7))
    sl = 1.0 + 0.015 * (mean_l - 50.0) ** 2 / np.sqrt(20.0 + (mean_l - 50.0) ** 2)
    sc, sh = 1.0 + 0.045 * mean_cp, 1.0 + 0.015 * mean_cp * t
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc
    return np.sqrt((delta_l / sl) ** 2 + (delta_c / sc) ** 2 + (delta_h / sh) ** 2 + rt * (delta_c / sc) * (delta_h / sh))


def _safe_background_metrics(panorama: np.ndarray, owner: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    boundary = owner_boundaries(owner)
    lab = cv2.cvtColor(panorama.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    samples0: list[np.ndarray] = []
    samples1: list[np.ndarray] = []
    for first, second in ((mask[:, :-1], mask[:, 1:]), (mask[:-1, :], mask[1:, :])):
        if first.shape[0] == mask.shape[0]:
            owner0, owner1 = owner[:, :-1], owner[:, 1:]
            lab0, lab1 = lab[:, :-1], lab[:, 1:]
        else:
            owner0, owner1 = owner[:-1, :], owner[1:, :]
            lab0, lab1 = lab[:-1, :], lab[1:, :]
        selected = first & second & (owner0 >= 0) & (owner1 >= 0) & (owner0 != owner1)
        if np.any(selected):
            samples0.append(lab0[selected])
            samples1.append(lab1[selected])
    if not samples0:
        return {"status": "not_evaluable", "reason": "no_owner_boundary_in_projected_safe_background"}
    first_lab, second_lab = np.concatenate(samples0), np.concatenate(samples1)
    delta = _delta_e00(first_lab, second_lab)
    brightness = np.abs(first_lab[:, 0] - second_lab[:, 0]) / np.maximum(1.0, 0.5 * (first_lab[:, 0] + second_lab[:, 0])) * 100.0
    delta_p95 = _percentile(delta, 95.0)
    brightness_p95 = _percentile(brightness, 95.0)
    return {
        "status": "evaluated",
        "seam_sample_count": int(delta.size),
        "delta_e00_p95": delta_p95,
        "brightness_step_p95_percent": brightness_p95,
        # ``0.0`` is the best possible observed seam result, not a missing
        # percentile.  Test explicitly for ``None`` so an exact colour match
        # cannot be converted into a false failure by Python's truthiness.
        "hard_gate_pass": bool(
            (delta_p95 if delta_p95 is not None else float("inf")) < 3.0
            and (brightness_p95 if brightness_p95 is not None else float("inf")) < 2.0
        ),
        "boundary_pixels_in_region": int(np.count_nonzero(boundary & mask)),
    }


def _measurement_units(
    *,
    kind: str,
    source_entries: Sequence[Mapping[str, Any]],
    projected: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, str | None, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]], ...]:
    """Collect one-entry units and explicit multi-source measurement groups.

    An omitted projection is normal for a split or an owner-filtered source.
    It does not cause a paired group to borrow coordinates from another source:
    only the members that have an explicit published projection contribute to
    the aggregate mask/line evidence.
    """

    grouped: dict[str, tuple[str | None, list[Mapping[str, Any]], list[Mapping[str, Any]]]] = {}
    for source in source_entries:
        identifier = source.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise VideoOfflineEvaluationError(f"Source annotations have malformed {kind} entry")
        measurement_group = source.get("measurement_group")
        if measurement_group is not None and (
            not isinstance(measurement_group, str) or not measurement_group.strip()
        ):
            raise VideoOfflineEvaluationError(f"Source annotation {identifier!r} has an invalid measurement_group")
        key = measurement_group if measurement_group is not None else identifier
        if key not in grouped:
            grouped[key] = (measurement_group, [], [])
        actual_group, members, projected_members = grouped[key]
        if actual_group != measurement_group:
            raise VideoOfflineEvaluationError(
                f"Measurement group {key!r} collides with an ungrouped annotation id"
            )
        members.append(source)
        item = projected.get(identifier)
        if item is not None:
            projected_members.append(item)
    return tuple(
        (key, group, tuple(members), tuple(projected_members))
        for key, (group, members, projected_members) in grouped.items()
    )


def _measurement_audit(
    *,
    measurement_group: str | None,
    members: Sequence[Mapping[str, Any]],
    projected_members: Sequence[Mapping[str, Any]],
    projected_annotation_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach immutable source/member provenance to a measurement result."""

    member_ids = [str(item["id"]) for item in members]
    projected_frame_ids = [int(item["frame_id"]) for item in projected_members]
    return {
        "measurement_group": measurement_group,
        "source_annotation_ids": member_ids,
        "projected_annotation_ids": list(projected_annotation_ids),
        "source_annotation_count": len(member_ids),
        "projected_member_count": len(projected_members),
        "projected_source_frame_ids": projected_frame_ids,
    }


def _measurement_role(
    *,
    kind: str,
    members: Sequence[Mapping[str, Any]],
    annotation_schema: object,
) -> str:
    """Return one immutable role for a measurement group.

    A group mixing role semantics would silently weaken a gate, so v2 rejects
    it.  v1 carries no role and preserves the historical compact-object rule.
    """

    if annotation_schema == ANNOTATION_SCHEMA:
        return {
            "objects": "compact_foreground_single_owner",
            "lines": "long_line",
            "safe_background": "safe_background",
        }[kind]
    roles = {entry.get("role") for entry in members}
    if len(roles) != 1 or not isinstance(next(iter(roles)), str):
        raise VideoOfflineEvaluationError(f"Measurement group has inconsistent v2 {kind} roles")
    return str(next(iter(roles)))


def evaluate_offline_visual_annotations(
    panorama_bgr: np.ndarray,
    owner: np.ndarray,
    *,
    annotations: Mapping[str, Any],
    projection: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Evaluate rendered artifacts without changing or promoting them.

    ``projection`` is the validated return from
    :func:`load_panorama_annotation_projection`; it is optional so global owner
    topology remains measurable on old deliveries.
    """

    panorama, owners = _require_panorama_and_owner(panorama_bgr, owner)
    if annotations.get("schema") not in SUPPORTED_ANNOTATION_SCHEMAS:
        raise VideoOfflineEvaluationError("Unsupported source annotation schema")
    topology = owner_topology_metrics(owners)
    area = int(owners.size)
    valid = owners >= 0
    topology.update(
        {
            "valid_pixel_count": int(np.count_nonzero(valid)),
            "unowned_pixel_count": int(np.count_nonzero(~valid)),
            "small_fragment_area_limit_pixels": float(area * 0.0005),
            "small_owner_source_count": int(sum(np.count_nonzero(owners == frame_id) < area * 0.0005 for frame_id in np.unique(owners[valid]))),
        }
    )
    result: dict[str, Any] = {
        "schema": OFFLINE_EVALUATION_SCHEMA,
        "measurement_only": True,
        "automatic_grade_promotion_allowed": False,
        "panorama_shape": [int(panorama.shape[0]), int(panorama.shape[1])],
        "owner_topology": topology,
        "object_integrity": {},
        "line_continuity": {},
        "safe_background": {},
        "projection_available": projection is not None,
    }
    for kind, result_key in (
        ("objects", "object_integrity"),
        ("lines", "line_continuity"),
        ("safe_background", "safe_background"),
    ):
        source_entries = annotations.get(kind, [])
        if not isinstance(source_entries, list):
            raise VideoOfflineEvaluationError(f"Source annotations lack {kind}")
        projected = projection.get(kind, {}) if projection is not None else {}
        group_states = projection.get("__measurement_groups__", {}).get(kind, {}) if projection is not None else {}
        typed_sources: list[Mapping[str, Any]] = []
        for source in source_entries:
            if not isinstance(source, Mapping):
                raise VideoOfflineEvaluationError(f"Source annotations have malformed {kind} entry")
            typed_sources.append(source)
        for identifier, measurement_group, members, projected_members in _measurement_units(
            kind=kind, source_entries=typed_sources, projected=projected
        ):
            role = _measurement_role(kind=kind, members=members, annotation_schema=annotations.get("schema"))
            projected_ids = [
                str(source["id"])
                for source in members
                if str(source["id"]) in projected
            ]
            audit = _measurement_audit(
                measurement_group=measurement_group,
                members=members,
                projected_members=projected_members,
                projected_annotation_ids=projected_ids,
            )
            declared_state = group_states.get(identifier if measurement_group is not None else "")
            if isinstance(declared_state, Mapping) and declared_state.get("measurement_state") == "projection_inconsistent":
                result[result_key][identifier] = {
                    "status": "not_evaluable", "reason": "projection_inconsistent", **audit,
                    "annotation_role": role,
                    "projection_consensus": dict(declared_state),
                }
            elif not projected_members:
                result[result_key][identifier] = {
                    "status": "not_evaluable",
                    "reason": "no_panorama_projection_for_fixed_source_annotation",
                    "annotation_role": role,
                    **audit,
                }
            elif kind == "lines":
                observations = [
                    _line_observations(panorama, entry["points"])
                    for entry in projected_members
                ]
                metrics = _line_metrics_from_observations(observations)
                if metrics["status"] != "evaluated":
                    metrics["reason"] = "insufficient_edge_samples_in_projected_measurement_group"
                result[result_key][identifier] = {**audit, "annotation_role": role, **metrics}
            else:
                masks = [_region_mask(entry, owners.shape) for entry in projected_members]
                mask = np.logical_or.reduce(masks)
                metrics = (
                    _object_metrics(owners, mask, role=role)
                    if kind == "objects"
                    else _safe_background_metrics(panorama, owners, mask)
                )
                result[result_key][identifier] = {**audit, "annotation_role": role, **metrics}
    return result


def evaluate_delivery_artifacts(
    delivery: str | Path,
    *,
    annotations_path: str | Path,
    projection_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load a published delivery and evaluate it without writing primary files."""

    root = Path(delivery).expanduser().resolve()
    panorama = cv2.imread(str(root / "video_panorama.png"), cv2.IMREAD_COLOR)
    if panorama is None:
        raise VideoOfflineEvaluationError("Published video_panorama.png is unavailable")
    try:
        with np.load(root / "video_pixel_provenance.npz", allow_pickle=False) as loaded:
            owner = np.asarray(loaded["owner_frame_id"])
    except (OSError, KeyError, ValueError) as exc:
        raise VideoOfflineEvaluationError("Published video provenance is unavailable or malformed") from exc
    annotations = load_source_annotations(annotations_path)
    projection = (
        load_panorama_annotation_projection(projection_path, annotations=annotations, panorama_shape=panorama.shape[:2])
        if projection_path is not None
        else None
    )
    return evaluate_offline_visual_annotations(panorama, owner, annotations=annotations, projection=projection)


def write_offline_evaluation(output: str | Path, evaluation: Mapping[str, Any]) -> Path:
    """Atomically write a sidecar and never modify delivery pixels/provenance."""

    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(f".{destination.name}.pending")
    try:
        pending.write_text(json.dumps(dict(evaluation), indent=2, sort_keys=True), encoding="utf-8")
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only offline video visual evaluation")
    parser.add_argument("--delivery", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--projection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        evaluation = evaluate_delivery_artifacts(args.delivery, annotations_path=args.annotations, projection_path=args.projection)
        output = write_offline_evaluation(args.output, evaluation)
    except (VideoOfflineEvaluationError, VideoAnnotationError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(output)


__all__ = [
    "OFFLINE_EVALUATION_SCHEMA",
    "PANORAMA_PROJECTION_SCHEMA",
    "VideoOfflineEvaluationError",
    "evaluate_delivery_artifacts",
    "evaluate_offline_visual_annotations",
    "load_panorama_annotation_projection",
    "write_offline_evaluation",
]


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
