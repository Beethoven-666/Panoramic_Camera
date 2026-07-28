"""FastSAM contour proposals tracked only by measured RGB-D evidence.

FastSAM supplies a polygon and nothing else.  World position comes from
aligned depth plus immutable camera poses; track identity additionally
requires world-voxel overlap, measured Lab agreement, and contour agreement.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .cuda_backend import pinhole_unproject, transform_points
from .session import CameraIntrinsics


@dataclass(frozen=True)
class FastSAMRGBDCandidate:
    candidate_id: int
    source_index: int
    frame_id: int
    polygon_xy: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    source_area_pixels: int
    depth_coverage_ratio: float
    world_voxel_hashes: frozenset[int]
    world_dilated_voxel_hashes: frozenset[int]
    world_centroid_mm: tuple[float, float, float]
    world_spans_mm: tuple[float, float, float]
    median_lab: tuple[float, float, float]
    aspect_ratio: float
    solidity: float
    # Exact FastSAM topology, stored only inside the candidate bbox.  Polygon
    # is retained for diagnostics, never as the formal binary mask when this
    # payload is available.
    exact_mask_bbox: np.ndarray | None = None


@dataclass(frozen=True)
class FastSAMWorldTrack:
    track_id: int
    candidate_ids: tuple[int, ...]
    source_indices: tuple[int, ...]
    audit: dict[str, object]


def parse_fastsam_polygons(
    path: Path,
    *,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Parse Ultralytics segmentation labels into integer polygons."""

    polygons: list[np.ndarray] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        values = raw_line.split()
        if len(values) < 7 or (len(values) - 1) % 2:
            continue
        coordinates = np.asarray(
            [float(value) for value in values[1:]], dtype=np.float64
        ).reshape(-1, 2)
        if not np.isfinite(coordinates).all():
            continue
        polygon = np.column_stack(
            (
                np.clip(
                    np.rint(coordinates[:, 0] * width),
                    0,
                    width - 1,
                ),
                np.clip(
                    np.rint(coordinates[:, 1] * height),
                    0,
                    height - 1,
                ),
            )
        ).astype(np.int32)
        keep = np.r_[
            True,
            np.any(polygon[1:] != polygon[:-1], axis=1),
        ]
        polygon = polygon[keep]
        if polygon.shape[0] >= 3:
            polygons.append(np.ascontiguousarray(polygon))
    return polygons


def _hash_voxel_keys(keys: np.ndarray) -> frozenset[int]:
    values = np.asarray(keys, dtype=np.int64)
    hashes = (
        values[:, 0] * np.int64(73_856_093)
        ^ values[:, 1] * np.int64(19_349_663)
        ^ values[:, 2] * np.int64(83_492_791)
    )
    return frozenset(int(value) for value in np.unique(hashes))


def _voxel_hashes(
    points: np.ndarray,
    voxel_size_mm: float,
) -> tuple[frozenset[int], frozenset[int]]:
    keys = np.floor(points / voxel_size_mm).astype(np.int64)
    unique_keys = np.unique(keys, axis=0)
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ],
        dtype=np.int64,
    )
    dilated = (
        unique_keys[:, None, :] + offsets[None, :, :]
    ).reshape(-1, 3)
    return _hash_voxel_keys(unique_keys), _hash_voxel_keys(dilated)


def build_fastsam_rgbd_candidate(
    *,
    candidate_id: int,
    source_index: int,
    frame_id: int,
    polygon_xy: np.ndarray,
    image_bgr: np.ndarray,
    lab_image: np.ndarray | None = None,
    depth_mm: np.ndarray,
    reliable_depth: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    reference_depth_mm: float,
    exact_mask_bbox: np.ndarray | None = None,
    sample_stride: int = 6,
    voxel_size_mm: float = 20.0,
) -> FastSAMRGBDCandidate | None:
    """Describe one polygon with measured depth, world voxels and RGB."""

    polygon = np.asarray(polygon_xy, dtype=np.int32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
        return None
    x, y, width, height = cv2.boundingRect(polygon)
    image_height, image_width = depth_mm.shape
    if (
        x <= 0
        or y <= 0
        or x + width >= image_width
        or y + height >= image_height
        or width < 12
        or height < 12
    ):
        return None
    local_polygon = polygon - np.asarray([x, y], dtype=np.int32)
    if exact_mask_bbox is None:
        local_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(local_mask, [local_polygon], 1)
    else:
        exact = np.asarray(exact_mask_bbox, dtype=bool)
        if exact.shape != (height, width):
            raise ValueError(
                "FastSAM exact bbox mask does not match polygon bbox"
            )
        local_mask = np.ascontiguousarray(exact, dtype=np.uint8)
    area = int(np.count_nonzero(local_mask))
    image_area = int(image_width * image_height)
    if area < 300 or area > int(0.12 * image_area):
        return None
    local_reliable = reliable_depth[y : y + height, x : x + width]
    depth_coverage = float(
        np.count_nonzero((local_mask > 0) & local_reliable) / area
    )
    if depth_coverage < 0.85:
        return None
    sample = np.zeros_like(local_mask, dtype=bool)
    sample[::sample_stride, ::sample_stride] = True
    selected = (local_mask > 0) & local_reliable & sample
    yy, xx = np.nonzero(selected)
    if xx.size < 24:
        return None
    xx = xx + x
    yy = yy + y
    selected_depth = depth_mm[yy, xx].astype(np.float64)
    margin = max(35.0, 0.04 * reference_depth_mm)
    if float(np.median(selected_depth)) >= reference_depth_mm - margin:
        return None
    p10, p90 = np.quantile(selected_depth, (0.10, 0.90))
    if p90 - p10 > max(180.0, 0.30 * float(np.median(selected_depth))):
        return None
    camera = pinhole_unproject(
        xx,
        yy,
        selected_depth,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    pose = np.asarray(camera_to_world, dtype=np.float64)
    world = transform_points(camera, pose[:3, :3], pose[:3, 3])
    spans = np.ptp(world, axis=0)
    if np.any(spans > 500.0) or np.count_nonzero(spans > 8.0) < 2:
        return None
    voxel_hashes, dilated_voxel_hashes = _voxel_hashes(
        world, voxel_size_mm
    )
    if len(voxel_hashes) < 8:
        return None
    measured_lab = (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        if lab_image is None
        else np.asarray(lab_image)
    )
    if measured_lab.shape != image_bgr.shape:
        raise ValueError("FastSAM candidate Lab image does not match RGB")
    median_lab = np.median(measured_lab[yy, xx], axis=0)
    contour_area = max(1.0, float(cv2.contourArea(polygon)))
    hull_area = max(
        contour_area, float(cv2.contourArea(cv2.convexHull(polygon)))
    )
    return FastSAMRGBDCandidate(
        candidate_id=int(candidate_id),
        source_index=int(source_index),
        frame_id=int(frame_id),
        polygon_xy=np.ascontiguousarray(polygon),
        bbox_xywh=(int(x), int(y), int(width), int(height)),
        source_area_pixels=area,
        depth_coverage_ratio=depth_coverage,
        world_voxel_hashes=voxel_hashes,
        world_dilated_voxel_hashes=dilated_voxel_hashes,
        world_centroid_mm=tuple(
            float(value) for value in np.median(world, axis=0)
        ),
        world_spans_mm=tuple(float(value) for value in spans),
        median_lab=tuple(float(value) for value in median_lab),
        aspect_ratio=float(width / height),
        solidity=float(contour_area / hull_area),
        exact_mask_bbox=np.ascontiguousarray(local_mask > 0),
    )


@dataclass
class _MutableTrack:
    candidates: list[FastSAMRGBDCandidate]
    voxel_counts: Counter[int]
    last_source_index: int

    def core(self) -> set[int]:
        required = max(1, int(math.ceil(0.30 * len(self.candidates))))
        return {
            value
            for value, count in self.voxel_counts.items()
            if count >= required
        }


def track_fastsam_rgbd_candidates(
    candidates_by_source: Sequence[Sequence[FastSAMRGBDCandidate]],
    *,
    minimum_voxel_overlap_ratio: float = 0.25,
    maximum_lab_delta: float = 30.0,
    maximum_source_gap: int = 12,
) -> tuple[FastSAMWorldTrack, ...]:
    """Greedily track polygons using only world/RGB/contour evidence."""

    tracks: list[_MutableTrack] = []
    for source_candidates in candidates_by_source:
        ordered = sorted(
            source_candidates,
            key=lambda item: (
                item.depth_coverage_ratio,
                len(item.world_voxel_hashes),
                item.source_area_pixels,
                -item.candidate_id,
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        for candidate in ordered:
            best: tuple[float, int] | None = None
            candidate_lab = np.asarray(candidate.median_lab)
            for track_index, track in enumerate(tracks):
                if (
                    track_index in used_tracks
                    or candidate.source_index == track.last_source_index
                    or candidate.source_index - track.last_source_index
                    > maximum_source_gap
                ):
                    continue
                core = track.core()
                intersection = len(candidate.world_voxel_hashes & core)
                overlap = float(
                    intersection
                    / max(1, min(len(candidate.world_voxel_hashes), len(core)))
                )
                if overlap < minimum_voxel_overlap_ratio:
                    continue
                representative = track.candidates[-1]
                if (
                    float(
                        np.linalg.norm(
                            candidate_lab
                            - np.asarray(representative.median_lab)
                        )
                    )
                    > maximum_lab_delta
                ):
                    continue
                area_ratio = max(
                    candidate.source_area_pixels,
                    representative.source_area_pixels,
                ) / max(
                    1,
                    min(
                        candidate.source_area_pixels,
                        representative.source_area_pixels,
                    ),
                )
                aspect_delta = abs(
                    math.log(
                        max(1e-6, candidate.aspect_ratio)
                        / max(1e-6, representative.aspect_ratio)
                    )
                )
                if (
                    area_ratio > 2.50
                    or aspect_delta > 0.80
                    or abs(candidate.solidity - representative.solidity)
                    > 0.35
                ):
                    continue
                score = overlap - 0.002 * float(
                    np.linalg.norm(
                        candidate_lab
                        - np.asarray(representative.median_lab)
                    )
                )
                if best is None or score > best[0]:
                    best = (score, track_index)
            if best is None:
                tracks.append(
                    _MutableTrack(
                        candidates=[candidate],
                        voxel_counts=Counter(candidate.world_voxel_hashes),
                        last_source_index=candidate.source_index,
                    )
                )
                continue
            selected = tracks[best[1]]
            selected.candidates.append(candidate)
            selected.voxel_counts.update(candidate.world_voxel_hashes)
            selected.last_source_index = candidate.source_index
            used_tracks.add(best[1])
    results: list[FastSAMWorldTrack] = []
    for track in tracks:
        sources = sorted(
            {item.source_index for item in track.candidates}
        )
        if len(sources) < 2:
            continue
        core = track.core()
        results.append(
            FastSAMWorldTrack(
                track_id=len(results),
                candidate_ids=tuple(
                    item.candidate_id for item in track.candidates
                ),
                source_indices=tuple(sources),
                audit={
                    "candidate_count": len(track.candidates),
                    "source_count": len(sources),
                    "core_world_voxel_count": len(core),
                    "world_voxel_overlap_gate": float(
                        minimum_voxel_overlap_ratio
                    ),
                    "maximum_lab_delta": float(maximum_lab_delta),
                    "maximum_source_gap": int(maximum_source_gap),
                    "mask_contour_gates": {
                        "maximum_area_ratio": 2.50,
                        "maximum_log_aspect_delta": 0.80,
                        "maximum_solidity_delta": 0.35,
                    },
                },
            )
        )
    return tuple(results)


def polygon_mask(
    candidate: FastSAMRGBDCandidate,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    exact = candidate.exact_mask_bbox
    if exact is None:
        cv2.fillPoly(mask, [candidate.polygon_xy], 1)
    else:
        x, y, width, height = candidate.bbox_xywh
        exact_value = np.asarray(exact, dtype=bool)
        if exact_value.shape != (height, width):
            raise ValueError("FastSAM candidate exact mask bbox is malformed")
        if (
            x < 0
            or y < 0
            or x + width > shape[1]
            or y + height > shape[0]
        ):
            raise ValueError("FastSAM candidate exact mask escapes its frame")
        mask[y : y + height, x : x + width] = exact_value
    return np.ascontiguousarray(mask > 0)


def flow_predict_mask(
    source_mask: np.ndarray,
    backward_flow: np.ndarray,
) -> np.ndarray:
    """Inverse-sample a mask for identity evidence, never for final RGB."""

    mask = np.asarray(source_mask, dtype=np.uint8)
    flow = np.asarray(backward_flow, dtype=np.float32)
    if flow.shape != (*mask.shape, 2):
        raise ValueError("Backward flow does not match the identity mask")
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    predicted = cv2.remap(
        mask,
        xx + flow[..., 0],
        yy + flow[..., 1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.ascontiguousarray(predicted > 0)


def flow_forward_backward_consistency(
    source_mask: np.ndarray,
    forward_flow: np.ndarray,
    backward_flow: np.ndarray,
    *,
    maximum_error_pixels: float = 0.75,
) -> dict[str, object]:
    """Audit sampled source-mask points under forward/backward DIS flow."""

    mask = np.asarray(source_mask, dtype=bool)
    forward = np.asarray(forward_flow, dtype=np.float32)
    backward = np.asarray(backward_flow, dtype=np.float32)
    if (
        forward.shape != (*mask.shape, 2)
        or backward.shape != forward.shape
    ):
        raise ValueError("Forward/backward flow dimensions are invalid")
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return {
            "sample_count": 0,
            "inside_count": 0,
            "consistent_ratio": 0.0,
            "p95_error_pixels": math.inf,
            "maximum_error_pixels": math.inf,
            "pass": False,
        }
    stride = max(1, int(math.ceil(xx.size / 512)))
    xx = xx[::stride].astype(np.float32)
    yy = yy[::stride].astype(np.float32)
    source_x = np.rint(xx).astype(np.int32)
    source_y = np.rint(yy).astype(np.int32)
    predicted_x = xx + forward[source_y, source_x, 0]
    predicted_y = yy + forward[source_y, source_x, 1]
    inside = (
        (predicted_x >= 0.0)
        & (predicted_x <= mask.shape[1] - 1)
        & (predicted_y >= 0.0)
        & (predicted_y <= mask.shape[0] - 1)
    )
    if not np.any(inside):
        return {
            "sample_count": int(xx.size),
            "inside_count": 0,
            "consistent_ratio": 0.0,
            "p95_error_pixels": math.inf,
            "maximum_error_pixels": math.inf,
            "pass": False,
        }
    map_x = predicted_x[inside].reshape(-1, 1).astype(np.float32)
    map_y = predicted_y[inside].reshape(-1, 1).astype(np.float32)
    sampled_backward_x = cv2.remap(
        backward[..., 0],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).reshape(-1)
    sampled_backward_y = cv2.remap(
        backward[..., 1],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    ).reshape(-1)
    error = np.hypot(
        forward[source_y[inside], source_x[inside], 0]
        + sampled_backward_x,
        forward[source_y[inside], source_x[inside], 1]
        + sampled_backward_y,
    )
    finite = np.isfinite(error)
    finite_error = error[finite]
    consistent_ratio = float(
        np.count_nonzero(finite_error <= maximum_error_pixels)
        / max(1, xx.size)
    )
    p95 = (
        float(np.quantile(finite_error, 0.95))
        if finite_error.size
        else math.inf
    )
    maximum = (
        float(np.max(finite_error)) if finite_error.size else math.inf
    )
    return {
        "sample_count": int(xx.size),
        "inside_count": int(np.count_nonzero(inside)),
        "consistent_ratio": consistent_ratio,
        "p95_error_pixels": p95,
        "maximum_error_pixels": maximum,
        "pass": bool(
            consistent_ratio >= 0.80 and p95 <= maximum_error_pixels
        ),
    }


def select_unambiguous_one_to_one_matches(
    valid: np.ndarray,
    score: np.ndarray,
    *,
    ambiguity_margin: float = 0.05,
) -> tuple[tuple[int, int], ...]:
    """Return mutual-best one-to-one matches; merge/split nodes terminate."""

    accepted = np.asarray(valid, dtype=bool)
    values = np.asarray(score, dtype=np.float64)
    if accepted.shape != values.shape or accepted.ndim != 2:
        raise ValueError("Identity match matrix is invalid")
    results: list[tuple[int, int]] = []
    for first in range(accepted.shape[0]):
        targets = np.flatnonzero(accepted[first])
        if targets.size == 0:
            continue
        row_scores = values[first, targets]
        row_order = np.argsort(-row_scores, kind="stable")
        second = int(targets[row_order[0]])
        best_score = float(row_scores[row_order[0]])
        if (
            row_order.size > 1
            and best_score - float(row_scores[row_order[1]])
            < ambiguity_margin
        ):
            continue
        sources = np.flatnonzero(accepted[:, second])
        column_scores = values[sources, second]
        column_order = np.argsort(-column_scores, kind="stable")
        if int(sources[column_order[0]]) != first:
            continue
        if (
            column_order.size > 1
            and best_score - float(column_scores[column_order[1]])
            < ambiguity_margin
        ):
            continue
        if not math.isfinite(float(values[first, second])):
            continue
        results.append((first, second))
    return tuple(results)


__all__ = [
    "FastSAMRGBDCandidate",
    "FastSAMWorldTrack",
    "build_fastsam_rgbd_candidate",
    "flow_forward_backward_consistency",
    "flow_predict_mask",
    "parse_fastsam_polygons",
    "polygon_mask",
    "select_unambiguous_one_to_one_matches",
    "track_fastsam_rgbd_candidates",
]
