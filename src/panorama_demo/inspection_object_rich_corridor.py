"""OCR-panel anchored object-rich corridor detection without instance models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import cv2
import numpy as np

from .inspection_ocr_panel import OCRSeededPanel, sample_mask_world_points
from .session import CameraIntrinsics


@dataclass(frozen=True)
class CorridorStructure:
    structure_id: int
    mask: np.ndarray
    contour_xy: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    area_pixels: int
    median_depth_mm: float
    depth_coverage_ratio: float
    internal_depth_continuity_ratio: float
    median_lab: tuple[float, float, float]
    world_points_mm: np.ndarray


@dataclass(frozen=True)
class ObjectRichCorridor:
    frame_id: int
    source_index: int
    panel: OCRSeededPanel
    structures: tuple[CorridorStructure, ...]
    interval_xyxy: tuple[int, int, int, int]
    left_endpoint_x: int
    right_endpoint_x: int
    left_endpoint_risk_ratio: float
    right_endpoint_risk_ratio: float
    inverse_map_coverage_ratio: float
    relative_scan_range_mm: tuple[float, float]
    clarity_variance: float
    audit: dict[str, object]


def _component_depth_continuity(
    mask: np.ndarray,
    depth_mm: np.ndarray,
) -> float:
    depth = np.asarray(depth_mm, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    ratios = []
    for axis in (0, 1):
        first_mask = np.take(
            selected, range(selected.shape[axis] - 1), axis=axis
        )
        second_mask = np.take(
            selected, range(1, selected.shape[axis]), axis=axis
        )
        pair = first_mask & second_mask
        if not np.any(pair):
            continue
        first_depth = np.take(
            depth, range(depth.shape[axis] - 1), axis=axis
        )
        second_depth = np.take(
            depth, range(1, depth.shape[axis]), axis=axis
        )
        tolerance = np.maximum(
            20.0,
            0.02 * np.minimum(first_depth, second_depth),
        )
        ratios.append(
            np.abs(first_depth[pair] - second_depth[pair])
            <= tolerance[pair]
        )
    if not ratios:
        return 0.0
    values = np.concatenate(ratios)
    return float(np.count_nonzero(values) / values.size)


def _endpoint_column(
    *,
    columns: range,
    y0: int,
    y1: int,
    gradient: np.ndarray,
    depth_mm: np.ndarray,
    reliable_depth: np.ndarray,
    geometric_valid: np.ndarray,
) -> tuple[int | None, float]:
    best: tuple[float, int] | None = None
    depth = np.asarray(depth_mm, dtype=np.float32)
    for x in columns:
        if not 1 <= x < gradient.shape[1] - 1:
            continue
        valid = geometric_valid[y0:y1, x]
        if valid.size == 0 or not np.all(valid):
            continue
        local_gradient = gradient[y0:y1, x]
        left = depth[y0:y1, x - 1]
        right = depth[y0:y1, x + 1]
        depth_valid = (
            reliable_depth[y0:y1, x - 1]
            & reliable_depth[y0:y1, x + 1]
        )
        tolerance = np.maximum(20.0, 0.02 * np.minimum(left, right))
        depth_edge = depth_valid & (np.abs(left - right) > tolerance)
        risky = (local_gradient > 24.0) | depth_edge
        risk_ratio = float(np.count_nonzero(risky) / risky.size)
        score = (
            risk_ratio
            + 0.001
            * float(np.quantile(local_gradient, 0.95))
        )
        if best is None or score < best[0]:
            best = (score, x)
    if best is None:
        return None, 1.0
    selected_x = best[1]
    risky = gradient[y0:y1, selected_x] > 24.0
    left = depth[y0:y1, selected_x - 1]
    right = depth[y0:y1, selected_x + 1]
    depth_valid = (
        reliable_depth[y0:y1, selected_x - 1]
        & reliable_depth[y0:y1, selected_x + 1]
    )
    tolerance = np.maximum(20.0, 0.02 * np.minimum(left, right))
    risky |= depth_valid & (np.abs(left - right) > tolerance)
    return (
        int(selected_x),
        float(np.count_nonzero(risky) / risky.size),
    )


def extract_object_rich_corridor(
    *,
    panel: OCRSeededPanel,
    image_bgr: np.ndarray,
    depth_mm: np.ndarray,
    reliable_depth: np.ndarray,
    geometric_valid: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: CameraIntrinsics,
    reference_depth_mm: float,
    scan_axis_world: Sequence[float],
) -> tuple[ObjectRichCorridor | None, dict[str, object]]:
    """Grow a fixed-gate, single-source corridor from an OCR panel."""

    image = np.asarray(image_bgr)
    depth = np.asarray(depth_mm, dtype=np.float32)
    reliable = np.asarray(reliable_depth, dtype=bool)
    geometric = np.asarray(geometric_valid, dtype=bool)
    if (
        image.shape[:2] != depth.shape
        or reliable.shape != depth.shape
        or geometric.shape != depth.shape
    ):
        raise ValueError("Object-rich corridor rasters are not aligned")
    height, width = depth.shape
    panel_x, panel_y, panel_width, panel_height = panel.bbox_xywh
    search_y0 = max(1, panel_y - int(math.ceil(1.5 * panel_height)))
    search_y1 = min(
        height - 1,
        panel_y + int(math.ceil(2.0 * panel_height)),
    )
    search_x0 = max(
        1, panel_x + int(math.floor(0.45 * panel_width))
    )
    search = np.zeros(depth.shape, dtype=bool)
    search[search_y0:search_y1, search_x0 : width - 1] = True
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    chroma = np.linalg.norm(
        lab[..., 1:].astype(np.float32) - 128.0, axis=2
    )
    foreground_limit = reference_depth_mm - max(
        60.0, 0.08 * reference_depth_mm
    )
    appearance_structure = (
        (gradient >= 24.0)
        | (lab[..., 0] <= 170)
        | (chroma >= 20.0)
    )
    excluded_panel = cv2.dilate(
        panel.mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) > 0
    occupied = (
        search
        & reliable
        & (depth < foreground_limit)
        & appearance_structure
        & ~excluded_panel
    )
    occupied = cv2.morphologyEx(
        occupied.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        occupied, connectivity=8
    )
    candidates: list[CorridorStructure] = []
    rejection_counts: dict[str, int] = {}
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        box_width = int(stats[label, cv2.CC_STAT_WIDTH])
        box_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        component = labels == label
        reason = None
        if area < 300 or box_width < 12 or box_height < 12:
            reason = "too_small"
        elif area > int(0.08 * width * height):
            reason = "too_large"
        elif box_width / max(1, box_height) > 8.0:
            reason = "broad_thin_surface"
        elif x + box_width <= panel_x + int(0.80 * panel_width):
            reason = "not_right_adjacent"
        if reason is not None:
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        coverage = float(
            np.count_nonzero(component & reliable)
            / max(1, np.count_nonzero(component))
        )
        continuity = _component_depth_continuity(component, depth)
        if coverage < 0.85 or continuity < 0.70:
            reason = "depth_support_failed"
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
            continue
        contours, _ = cv2.findContours(
            component.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        contour = max(contours, key=cv2.contourArea)
        world = sample_mask_world_points(
            mask=component,
            depth_mm=depth,
            reliable_depth=reliable,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
        )
        if world.shape[0] < 30:
            rejection_counts["insufficient_world_samples"] = (
                rejection_counts.get("insufficient_world_samples", 0) + 1
            )
            continue
        candidates.append(
            CorridorStructure(
                structure_id=len(candidates),
                mask=np.ascontiguousarray(component),
                contour_xy=np.ascontiguousarray(contour[:, 0, :]),
                bbox_xywh=(x, y, box_width, box_height),
                area_pixels=area,
                median_depth_mm=float(np.median(depth[component])),
                depth_coverage_ratio=coverage,
                internal_depth_continuity_ratio=continuity,
                median_lab=tuple(
                    float(value)
                    for value in np.median(lab[component], axis=0)
                ),
                world_points_mm=np.ascontiguousarray(world),
            )
        )
    ordered = sorted(
        candidates,
        key=lambda item: (item.bbox_xywh[0], -item.area_pixels),
    )
    corridor_right = panel_x + panel_width
    selected: list[CorridorStructure] = []
    for candidate in ordered:
        x, _, box_width, _ = candidate.bbox_xywh
        if x <= corridor_right + 160:
            selected.append(candidate)
            corridor_right = max(corridor_right, x + box_width)
    all_masks = [panel.mask, *(item.mask for item in selected)]
    union = np.logical_or.reduce(all_masks)
    yy, xx = np.nonzero(union)
    audit: dict[str, object] = {
        "frame_id": int(panel.frame_id),
        "search_bbox_xywh": [
            search_x0,
            search_y0,
            width - 1 - search_x0,
            search_y1 - search_y0,
        ],
        "foreground_depth_limit_mm": float(foreground_limit),
        "raw_component_count": max(0, count - 1),
        "candidate_structure_count": len(candidates),
        "selected_structure_count": len(selected),
        "component_rejection_counts": rejection_counts,
    }
    if not selected or not xx.size:
        return None, {
            **audit,
            "pass": False,
            "rejection_reason": "no_right_adjacent_object_rich_structure",
        }
    interval_y0 = max(1, int(np.min(yy)))
    interval_y1 = min(height - 1, int(np.max(yy)) + 1)
    structure_left = int(np.min(xx))
    structure_right = int(np.max(xx)) + 1
    left_x, left_risk = _endpoint_column(
        columns=range(max(1, structure_left - 32), structure_left),
        y0=interval_y0,
        y1=interval_y1,
        gradient=gradient,
        depth_mm=depth,
        reliable_depth=reliable,
        geometric_valid=geometric,
    )
    right_x, right_risk = _endpoint_column(
        columns=range(
            structure_right,
            min(width - 1, structure_right + 33),
        ),
        y0=interval_y0,
        y1=interval_y1,
        gradient=gradient,
        depth_mm=depth,
        reliable_depth=reliable,
        geometric_valid=geometric,
    )
    if left_x is None or right_x is None or left_x >= right_x:
        return None, {
            **audit,
            "pass": False,
            "rejection_reason": "low_risk_interval_endpoints_unavailable",
        }
    inverse_coverage = float(
        np.count_nonzero(
            geometric[interval_y0:interval_y1, left_x : right_x + 1]
        )
        / max(
            1,
            (interval_y1 - interval_y0) * (right_x - left_x + 1),
        )
    )
    scan_axis = np.asarray(scan_axis_world, dtype=np.float64)
    panel_scan = float(
        np.asarray(panel.world_centroid_mm, dtype=np.float64) @ scan_axis
    )
    all_world = np.concatenate(
        [panel.world_points_mm, *(item.world_points_mm for item in selected)]
    )
    relative_scan = all_world @ scan_axis - panel_scan
    scan_range = (
        float(np.quantile(relative_scan, 0.01)),
        float(np.quantile(relative_scan, 0.99)),
    )
    interval_gray = gray[
        interval_y0:interval_y1, left_x : right_x + 1
    ]
    clarity = float(
        cv2.Laplacian(interval_gray, cv2.CV_64F).var()
    )
    accepted = bool(
        left_risk <= 0.15
        and right_risk <= 0.15
        and inverse_coverage >= 1.0
        and right_x - left_x + 1 <= width
    )
    audit.update(
        {
            "selected_structures": [
                {
                    "structure_id": item.structure_id,
                    "bbox_xywh": list(item.bbox_xywh),
                    "area_pixels": item.area_pixels,
                    "median_depth_mm": item.median_depth_mm,
                    "depth_coverage_ratio": item.depth_coverage_ratio,
                    "internal_depth_continuity_ratio": (
                        item.internal_depth_continuity_ratio
                    ),
                    "median_lab": list(item.median_lab),
                    "world_sample_count": int(
                        item.world_points_mm.shape[0]
                    ),
                }
                for item in selected
            ],
            "interval_xyxy": [
                left_x,
                interval_y0,
                right_x + 1,
                interval_y1,
            ],
            "left_endpoint_risk_ratio": left_risk,
            "right_endpoint_risk_ratio": right_risk,
            "inverse_map_coverage_ratio": inverse_coverage,
            "relative_scan_range_mm": list(scan_range),
            "clarity_laplacian_variance": clarity,
            "pass": accepted,
            "rejection_reason": (
                None if accepted else "corridor_endpoint_or_coverage_gate_failed"
            ),
        }
    )
    if not accepted:
        return None, audit
    return (
        ObjectRichCorridor(
            frame_id=int(panel.frame_id),
            source_index=int(panel.source_index),
            panel=panel,
            structures=tuple(selected),
            interval_xyxy=(
                left_x,
                interval_y0,
                right_x + 1,
                interval_y1,
            ),
            left_endpoint_x=left_x,
            right_endpoint_x=right_x,
            left_endpoint_risk_ratio=left_risk,
            right_endpoint_risk_ratio=right_risk,
            inverse_map_coverage_ratio=inverse_coverage,
            relative_scan_range_mm=scan_range,
            clarity_variance=clarity,
            audit=audit,
        ),
        audit,
    )


def interval_pair_metrics(
    first: tuple[float, float],
    second: tuple[float, float],
) -> tuple[float, float]:
    """Return 1D IoU and smaller-range coverage."""

    first_length = max(0.0, first[1] - first[0])
    second_length = max(0.0, second[1] - second[0])
    intersection = max(
        0.0, min(first[1], second[1]) - max(first[0], second[0])
    )
    union = first_length + second_length - intersection
    return (
        float(intersection / union) if union else 0.0,
        float(intersection / min(first_length, second_length))
        if min(first_length, second_length) > 0.0
        else 0.0,
    )


def track_object_rich_corridors(
    corridors: Sequence[ObjectRichCorridor],
    *,
    minimum_range_iou: float = 0.50,
    minimum_smaller_range_coverage: float = 0.75,
    maximum_source_gap: int = 12,
) -> tuple[tuple[int, ...], ...]:
    """Track corridor world ranges with fixed pair gates."""

    ordered = sorted(
        enumerate(corridors),
        key=lambda item: (item[1].source_index, item[1].frame_id),
    )
    tracks: list[list[int]] = []
    for index, corridor in ordered:
        valid: list[tuple[float, int]] = []
        for track_index, track in enumerate(tracks):
            previous = corridors[track[-1]]
            gap = corridor.source_index - previous.source_index
            if not 1 <= gap <= maximum_source_gap:
                continue
            iou, smaller = interval_pair_metrics(
                previous.relative_scan_range_mm,
                corridor.relative_scan_range_mm,
            )
            structure_ratio = max(
                len(previous.structures), len(corridor.structures)
            ) / max(
                1, min(len(previous.structures), len(corridor.structures))
            )
            if (
                iou >= minimum_range_iou
                and smaller >= minimum_smaller_range_coverage
                and structure_ratio <= 2.0
            ):
                valid.append((iou + smaller, track_index))
        valid.sort(reverse=True)
        if (
            not valid
            or (
                len(valid) > 1
                and valid[0][0] - valid[1][0] < 0.05
            )
        ):
            tracks.append([index])
        else:
            tracks[valid[0][1]].append(index)
    return tuple(tuple(track) for track in tracks if len(track) >= 2)


__all__ = [
    "CorridorStructure",
    "ObjectRichCorridor",
    "extract_object_rich_corridor",
    "interval_pair_metrics",
    "track_object_rich_corridors",
]
