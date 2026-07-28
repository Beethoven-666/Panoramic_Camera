"""OCR-seeded RGB-D panel structure extraction for read-only diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from .cuda_backend import pinhole_unproject, transform_points
from .session import CameraIntrinsics


@dataclass(frozen=True)
class OCRSeededPanel:
    frame_id: int
    source_index: int
    mask: np.ndarray
    contour_xy: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    world_points_mm: np.ndarray
    world_centroid_mm: tuple[float, float, float]
    world_extent_pca_mm: tuple[float, float]
    median_lab: tuple[float, float, float]
    aspect_ratio: float
    rectangularity: float
    solidity: float
    clarity_variance: float
    audit: dict[str, object]


@dataclass(frozen=True)
class StableObjectTrackEvidence:
    track_id: int
    observation_count: int
    selected_panel_observation_count: int
    common_frame_ids: tuple[int, ...]
    median_lab_l: float
    clarity_variance: float
    minimum_depth_coverage_ratio: float
    adjacent_to_panel: bool


def _polygon_mask(
    polygon_xy: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [np.rint(polygon_xy).astype(np.int32)],
        1,
    )
    return mask.astype(bool)


def sample_mask_world_points(
    *,
    mask: np.ndarray,
    depth_mm: np.ndarray,
    reliable_depth: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    stride: int = 4,
) -> np.ndarray:
    """Sample measured world points without changing RGB or pose."""

    selected = np.asarray(mask, dtype=bool) & np.asarray(
        reliable_depth, dtype=bool
    )
    sample = np.zeros(selected.shape, dtype=bool)
    sample[::stride, ::stride] = True
    yy, xx = np.nonzero(selected & sample)
    if xx.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    values = np.asarray(depth_mm, dtype=np.float64)[yy, xx]
    valid = np.isfinite(values) & (values > 0.0)
    xx = xx[valid]
    yy = yy[valid]
    values = values[valid]
    camera = pinhole_unproject(
        xx,
        yy,
        values,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    pose = np.asarray(camera_to_world, dtype=np.float64)
    return transform_points(camera, pose[:3, :3], pose[:3, 3])


def _pca_extent(points: np.ndarray) -> tuple[float, float]:
    centered = points - np.median(points, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ axes[:2].T
    spans = np.ptp(coordinates, axis=0)
    ordered = np.sort(spans)[::-1]
    return float(ordered[0]), float(ordered[1])


def extract_ocr_seeded_panel(
    *,
    frame_id: int,
    source_index: int,
    image_bgr: np.ndarray,
    depth_mm: np.ndarray,
    reliable_depth: np.ndarray,
    ocr_polygon_xy: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[OCRSeededPanel | None, dict[str, object]]:
    """Extract one connected white same-layer structure around OCR text."""

    image = np.asarray(image_bgr)
    depth = np.asarray(depth_mm, dtype=np.float32)
    reliable = np.asarray(reliable_depth, dtype=bool)
    polygon = np.asarray(ocr_polygon_xy, dtype=np.float32)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or depth.shape != image.shape[:2]
        or reliable.shape != depth.shape
        or polygon.shape != (4, 2)
    ):
        raise ValueError("OCR panel extraction inputs do not match")
    height, width = depth.shape
    text_x, text_y, text_width, text_height = cv2.boundingRect(
        np.rint(polygon).astype(np.int32)
    )
    margin_x = int(math.ceil(1.75 * text_width))
    margin_top = int(math.ceil(2.5 * text_height))
    margin_bottom = int(math.ceil(3.5 * text_height))
    x0 = max(0, text_x - margin_x)
    x1 = min(width, text_x + text_width + margin_x)
    y0 = max(0, text_y - margin_top)
    y1 = min(height, text_y + text_height + margin_bottom)
    search = np.zeros((height, width), dtype=bool)
    search[y0:y1, x0:x1] = True
    text_mask = _polygon_mask(polygon, (height, width))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    chroma = np.linalg.norm(
        lab[..., 1:].astype(np.float32) - 128.0, axis=2
    )
    white = (lab[..., 0] >= 145) & (chroma <= 28.0)
    ring_radius = max(3, int(round(0.50 * text_height)))
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * ring_radius + 1, 2 * ring_radius + 1),
    )
    ring = (
        cv2.dilate(text_mask.astype(np.uint8), ring_kernel) > 0
    ) & ~text_mask
    initial_seed = ring & search & white & reliable
    seed_depth_values = depth[initial_seed]
    seed_depth_values = seed_depth_values[
        np.isfinite(seed_depth_values) & (seed_depth_values > 0.0)
    ]
    audit: dict[str, object] = {
        "search_bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
        "ocr_bbox_xywh": [
            text_x,
            text_y,
            text_width,
            text_height,
        ],
        "ring_seed_pixel_count": int(seed_depth_values.size),
    }
    if seed_depth_values.size < 30:
        return None, {
            **audit,
            "pass": False,
            "rejection_reason": "insufficient_white_depth_ring_seed",
        }
    seed_depth = float(np.median(seed_depth_values))
    depth_tolerance = max(20.0, 0.02 * seed_depth)
    same_layer = (
        reliable
        & np.isfinite(depth)
        & (np.abs(depth - seed_depth) <= depth_tolerance)
    )
    candidate = (
        search
        & same_layer
        & white
        & (gradient <= 80.0)
    )
    close_size = max(
        3,
        min(15, int(round(0.35 * text_height)) | 1),
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (close_size, close_size)
    )
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_CLOSE,
        close_kernel,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate, connectivity=8
    )
    best_label = 0
    best_overlap = 0
    for label in range(1, count):
        overlap = int(np.count_nonzero((labels == label) & initial_seed))
        if overlap > best_overlap:
            best_label = label
            best_overlap = overlap
    audit.update(
        {
            "seed_depth_mm": seed_depth,
            "same_layer_tolerance_mm": depth_tolerance,
            "component_count": max(0, count - 1),
            "selected_component_ring_seed_overlap": best_overlap,
            "morphological_close_size": close_size,
        }
    )
    if best_label == 0 or best_overlap < 20:
        return None, {
            **audit,
            "pass": False,
            "rejection_reason": "no_connected_same_layer_white_component",
        }
    component = labels == best_label
    text_same_layer = text_mask & same_layer
    structure = component | text_same_layer
    structure = cv2.morphologyEx(
        structure.astype(np.uint8),
        cv2.MORPH_CLOSE,
        close_kernel,
    ) > 0
    contours, _ = cv2.findContours(
        structure.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour = max(contours, key=cv2.contourArea)
    structure = _polygon_mask(contour[:, 0, :], (height, width))
    area = int(np.count_nonzero(structure))
    text_area = max(1, int(np.count_nonzero(text_mask)))
    contour_area = max(1.0, float(cv2.contourArea(contour)))
    rectangle = cv2.minAreaRect(contour)
    rectangle_area = max(
        1.0, float(rectangle[1][0] * rectangle[1][1])
    )
    rectangularity = float(contour_area / rectangle_area)
    hull_area = max(
        contour_area, float(cv2.contourArea(cv2.convexHull(contour)))
    )
    solidity = float(contour_area / hull_area)
    box_x, box_y, box_width, box_height = cv2.boundingRect(contour)
    aspect_ratio = float(
        max(rectangle[1]) / max(1.0, min(rectangle[1]))
    )
    text_coverage = float(
        np.count_nonzero(text_mask & structure) / text_area
    )
    touches_search_border = bool(
        box_x <= x0
        or box_y <= y0
        or box_x + box_width >= x1
        or box_y + box_height >= y1
    )
    structure_depth = depth[structure & reliable]
    depth_coverage = float(
        structure_depth.size / max(1, np.count_nonzero(structure))
    )
    structure_gradient = gradient[structure]
    measured_lab = lab[structure]
    median_lab = np.median(measured_lab, axis=0)
    world = sample_mask_world_points(
        mask=structure,
        depth_mm=depth,
        reliable_depth=reliable,
        camera_to_world=camera_to_world,
        intrinsics=intrinsics,
    )
    extent = (
        _pca_extent(world)
        if world.shape[0] >= 30
        else (0.0, 0.0)
    )
    area_ratio = float(area / text_area)
    clarity = float(
        cv2.Laplacian(gray[box_y : box_y + box_height, box_x : box_x + box_width], cv2.CV_64F).var()
    )
    accepted = bool(
        5.0 <= area_ratio <= 120.0
        and text_coverage >= 0.90
        and rectangularity >= 0.65
        and solidity >= 0.75
        and aspect_ratio >= 2.0
        and not touches_search_border
        and depth_coverage >= 0.85
        and world.shape[0] >= 30
        and extent[0] >= 80.0
        and extent[1] >= 15.0
    )
    audit.update(
        {
            "structure_area_pixels": area,
            "structure_to_text_area_ratio": area_ratio,
            "ocr_coverage_ratio": text_coverage,
            "bbox_xywh": [box_x, box_y, box_width, box_height],
            "rectangularity": rectangularity,
            "solidity": solidity,
            "min_area_rectangle_aspect_ratio": aspect_ratio,
            "touches_search_border": touches_search_border,
            "depth_coverage_ratio": depth_coverage,
            "gradient_median": float(np.median(structure_gradient)),
            "gradient_p95": float(
                np.quantile(structure_gradient, 0.95)
            ),
            "median_lab": [float(value) for value in median_lab],
            "world_sample_count": int(world.shape[0]),
            "world_centroid_mm": (
                [
                    float(value)
                    for value in np.median(world, axis=0)
                ]
                if world.size
                else None
            ),
            "world_extent_pca_mm": list(extent),
            "clarity_laplacian_variance": clarity,
            "pass": accepted,
            "rejection_reason": (
                None if accepted else "panel_structure_gate_failed"
            ),
        }
    )
    if not accepted:
        return None, audit
    return (
        OCRSeededPanel(
            frame_id=int(frame_id),
            source_index=int(source_index),
            mask=np.ascontiguousarray(structure),
            contour_xy=np.ascontiguousarray(contour[:, 0, :]),
            bbox_xywh=(
                int(box_x),
                int(box_y),
                int(box_width),
                int(box_height),
            ),
            world_points_mm=np.ascontiguousarray(world),
            world_centroid_mm=tuple(
                float(value)
                for value in np.median(world, axis=0)
            ),
            world_extent_pca_mm=extent,
            median_lab=tuple(float(value) for value in median_lab),
            aspect_ratio=aspect_ratio,
            rectangularity=rectangularity,
            solidity=solidity,
            clarity_variance=clarity,
            audit=audit,
        ),
        audit,
    )


def track_ocr_seeded_panels(
    candidates: Sequence[OCRSeededPanel],
    *,
    maximum_world_centroid_delta_mm: float = 80.0,
    maximum_extent_ratio: float = 1.40,
    maximum_lab_delta: float = 20.0,
    maximum_log_aspect_delta: float = 0.30,
    maximum_source_gap: int = 12,
) -> tuple[tuple[int, ...], ...]:
    """Build unambiguous temporal tracks from measured panel structures."""

    ordered = sorted(
        enumerate(candidates),
        key=lambda item: (item[1].source_index, item[1].frame_id),
    )
    tracks: list[list[int]] = []
    for candidate_index, candidate in ordered:
        valid: list[tuple[float, int]] = []
        for track_index, track in enumerate(tracks):
            previous = candidates[track[-1]]
            source_gap = candidate.source_index - previous.source_index
            if not 1 <= source_gap <= maximum_source_gap:
                continue
            centroid_delta = float(
                np.linalg.norm(
                    np.asarray(candidate.world_centroid_mm)
                    - np.asarray(previous.world_centroid_mm)
                )
            )
            extent_ratio = max(
                max(candidate.world_extent_pca_mm)
                / max(1e-6, max(previous.world_extent_pca_mm)),
                max(previous.world_extent_pca_mm)
                / max(1e-6, max(candidate.world_extent_pca_mm)),
                min(candidate.world_extent_pca_mm)
                / max(1e-6, min(previous.world_extent_pca_mm)),
                min(previous.world_extent_pca_mm)
                / max(1e-6, min(candidate.world_extent_pca_mm)),
            )
            lab_delta = float(
                np.linalg.norm(
                    np.asarray(candidate.median_lab)
                    - np.asarray(previous.median_lab)
                )
            )
            aspect_delta = abs(
                math.log(
                    candidate.aspect_ratio
                    / max(1e-6, previous.aspect_ratio)
                )
            )
            if (
                centroid_delta <= maximum_world_centroid_delta_mm
                and extent_ratio <= maximum_extent_ratio
                and lab_delta <= maximum_lab_delta
                and aspect_delta <= maximum_log_aspect_delta
            ):
                score = (
                    centroid_delta / maximum_world_centroid_delta_mm
                    + extent_ratio / maximum_extent_ratio
                    + lab_delta / maximum_lab_delta
                )
                valid.append((score, track_index))
        valid.sort()
        if (
            not valid
            or (
                len(valid) > 1
                and valid[1][0] - valid[0][0] < 0.05
            )
        ):
            tracks.append([candidate_index])
        else:
            tracks[valid[0][1]].append(candidate_index)
    return tuple(
        tuple(track) for track in tracks if len(track) >= 2
    )


def select_object_rich_neighbor_tracks(
    tracks: Sequence[StableObjectTrackEvidence],
    *,
    required_track_count: int = 2,
) -> tuple[int, ...]:
    """Select long-lived dark neighbours without using track identifiers."""

    eligible = [
        item
        for item in tracks
        if item.selected_panel_observation_count >= 2
        and len(set(item.common_frame_ids)) >= 2
        and item.median_lab_l <= 150.0
        and item.minimum_depth_coverage_ratio >= 0.85
        and item.adjacent_to_panel
    ]
    eligible.sort(
        key=lambda item: (
            item.observation_count,
            len(set(item.common_frame_ids)),
            item.selected_panel_observation_count,
            item.clarity_variance,
        ),
        reverse=True,
    )
    if len(eligible) < required_track_count:
        return ()
    return tuple(
        int(item.track_id) for item in eligible[:required_track_count]
    )


def audit_object_rich_interval(
    *,
    projected_x_spans: Sequence[tuple[float, float]],
    projected_in_bounds_ratios: Sequence[float],
    depth_coverage_ratios: Sequence[float],
    source_width_pixels: int,
    maximum_gap_pixels: float = 160.0,
) -> dict[str, object]:
    """Audit a pre-seam single-panel interval for several full structures."""

    if (
        len(projected_x_spans) < 3
        or len(projected_x_spans) != len(projected_in_bounds_ratios)
        or len(projected_x_spans) != len(depth_coverage_ratios)
    ):
        return {
            "pass": False,
            "rejection_reason": "object_rich_interval_inputs_incomplete",
        }
    spans = sorted(
        (
            (float(min(first, second)), float(max(first, second)))
            for first, second in projected_x_spans
        ),
        key=lambda item: item[0],
    )
    gaps = [
        max(0.0, spans[index + 1][0] - spans[index][1])
        for index in range(len(spans) - 1)
    ]
    interval_width = float(
        max(item[1] for item in spans)
        - min(item[0] for item in spans)
    )
    minimum_projection = float(min(projected_in_bounds_ratios))
    minimum_depth = float(min(depth_coverage_ratios))
    maximum_gap = float(max(gaps, default=0.0))
    accepted = bool(
        minimum_projection >= 0.90
        and minimum_depth >= 0.85
        and interval_width <= float(source_width_pixels)
        and maximum_gap <= maximum_gap_pixels
    )
    return {
        "ordered_projected_x_spans": [list(item) for item in spans],
        "inter_structure_gaps_pixels": gaps,
        "maximum_inter_structure_gap_pixels": maximum_gap,
        "combined_interval_width_pixels": interval_width,
        "minimum_projected_in_bounds_ratio": minimum_projection,
        "minimum_depth_coverage_ratio": minimum_depth,
        "source_width_pixels": int(source_width_pixels),
        "maximum_allowed_gap_pixels": float(maximum_gap_pixels),
        "pass": accepted,
        "rejection_reason": (
            None
            if accepted
            else "pre_seam_single_panel_interval_gate_failed"
        ),
    }


def audit_relative_world_geometry(
    panel_centroids_by_frame: dict[int, Sequence[float]],
    object_centroids_by_frame: dict[int, Sequence[float]],
    *,
    maximum_relative_vector_deviation_mm: float = 80.0,
) -> dict[str, object]:
    """Audit cross-view relative position using only real world points."""

    common = sorted(
        set(panel_centroids_by_frame) & set(object_centroids_by_frame)
    )
    vectors = np.asarray(
        [
            np.asarray(object_centroids_by_frame[frame_id], dtype=np.float64)
            - np.asarray(panel_centroids_by_frame[frame_id], dtype=np.float64)
            for frame_id in common
        ],
        dtype=np.float64,
    )
    if len(common) < 2 or not np.isfinite(vectors).all():
        return {
            "common_frame_ids": common,
            "relative_vector_count": len(common),
            "pass": False,
            "rejection_reason": "fewer_than_two_finite_relative_world_vectors",
        }
    median = np.median(vectors, axis=0)
    deviations = np.linalg.norm(vectors - median, axis=1)
    maximum_deviation = float(np.max(deviations))
    accepted = bool(
        maximum_deviation <= maximum_relative_vector_deviation_mm
    )
    return {
        "common_frame_ids": common,
        "relative_vector_count": len(common),
        "relative_vectors_mm": vectors.tolist(),
        "median_relative_vector_mm": median.tolist(),
        "maximum_relative_vector_deviation_mm": maximum_deviation,
        "maximum_allowed_relative_vector_deviation_mm": float(
            maximum_relative_vector_deviation_mm
        ),
        "pass": accepted,
        "rejection_reason": (
            None
            if accepted
            else "relative_world_geometry_is_not_cross_view_consistent"
        ),
    }


__all__ = [
    "OCRSeededPanel",
    "StableObjectTrackEvidence",
    "audit_object_rich_interval",
    "audit_relative_world_geometry",
    "extract_ocr_seeded_panel",
    "sample_mask_world_points",
    "select_object_rich_neighbor_tracks",
    "track_ocr_seeded_panels",
]
