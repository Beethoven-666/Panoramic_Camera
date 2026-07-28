"""Bounded panel-native whole-object lock diagnostics.

This module does not participate in the formal renderer.  A source mask may
reach the inspection canvas only through the already audited target-to-source
inverse map of its selected panel.  RGB-D world samples are used solely to
confirm cross-view identity and to reject merge/split ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np

from .inspection_chain_seam import PairCorridorEvidence
from .inspection_fastsam_track import FastSAMRGBDCandidate


@dataclass(frozen=True)
class PanelNativeLockConfig:
    """Fixed first-pass gates for the isolated diagnostic."""

    minimum_view_count: int = 2
    minimum_depth_coverage_ratio: float = 0.90
    minimum_inverse_source_coverage_ratio: float = 0.90
    minimum_dominant_target_component_ratio: float = 0.98
    minimum_world_voxel_overlap_ratio: float = 0.25
    maximum_world_centroid_distance_mm: float = 80.0
    maximum_source_area_ratio: float = 1.80
    minimum_target_mask_iou: float = 0.50
    minimum_target_smaller_mask_coverage: float = 0.70
    merge_split_peer_overlap_ratio: float = 0.20
    maximum_accepted_target_overlap_ratio: float = 0.10

    def validate(self) -> None:
        if self.minimum_view_count < 2:
            raise ValueError("Panel-native lock requires at least two views")
        for name in (
            "minimum_depth_coverage_ratio",
            "minimum_inverse_source_coverage_ratio",
            "minimum_dominant_target_component_ratio",
            "minimum_world_voxel_overlap_ratio",
            "minimum_target_mask_iou",
            "minimum_target_smaller_mask_coverage",
            "merge_split_peer_overlap_ratio",
            "maximum_accepted_target_overlap_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            not math.isfinite(self.maximum_world_centroid_distance_mm)
            or self.maximum_world_centroid_distance_mm <= 0.0
        ):
            raise ValueError(
                "maximum_world_centroid_distance_mm must be positive"
            )
        if (
            not math.isfinite(self.maximum_source_area_ratio)
            or self.maximum_source_area_ratio < 1.0
        ):
            raise ValueError("maximum_source_area_ratio must be at least one")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "PanelNativeLockConfig":
        payload = {} if value is None else dict(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"unknown panel-native lock configuration keys: {unknown}"
            )
        result = cls(**payload)
        result.validate()
        return result


@dataclass(frozen=True)
class PanelNativeObservation:
    """One FastSAM mask sampled only through one existing panel map."""

    candidate: FastSAMRGBDCandidate
    panel_index: int
    frame_id: int
    source_mask: np.ndarray
    target_mask: np.ndarray
    target_image_bgr: np.ndarray
    inverse_source_coverage_ratio: float
    dominant_target_component_ratio: float
    clarity: float
    centrality: float
    audit: dict[str, object]


def mask_overlap_metrics(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    """Return IoU and intersection coverage of the smaller mask."""

    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask overlap inputs must be aligned")
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    smaller = min(int(np.count_nonzero(a)), int(np.count_nonzero(b)))
    return (
        float(intersection / union) if union else 0.0,
        float(intersection / smaller) if smaller else 0.0,
    )


def _dominant_component_ratio(mask: np.ndarray) -> tuple[float, int]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask, dtype=np.uint8), 8
    )
    if count <= 1:
        return 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = int(np.count_nonzero(mask))
    return (
        float(int(np.max(areas)) / total) if total else 0.0,
        int(count - 1),
    )


def map_mask_through_existing_inverse(
    *,
    candidate: FastSAMRGBDCandidate,
    panel_index: int,
    frame_id: int,
    source_mask: np.ndarray,
    source_image_bgr: np.ndarray,
    inverse_map_x: np.ndarray,
    inverse_map_y: np.ndarray,
    inverse_valid_mask: np.ndarray,
    corner_x: int,
    canvas_shape: tuple[int, int],
    config: PanelNativeLockConfig | Mapping[str, object] | None = None,
) -> tuple[PanelNativeObservation | None, dict[str, object]]:
    """Map one complete proposal with the panel's audited inverse map only."""

    selected = (
        config
        if isinstance(config, PanelNativeLockConfig)
        else PanelNativeLockConfig.from_mapping(config)
    )
    selected.validate()
    source = np.asarray(source_mask, dtype=bool)
    image = np.asarray(source_image_bgr, dtype=np.uint8)
    map_x = np.asarray(inverse_map_x, dtype=np.float32)
    map_y = np.asarray(inverse_map_y, dtype=np.float32)
    valid = np.asarray(inverse_valid_mask, dtype=bool)
    if image.shape[:2] != source.shape:
        raise ValueError("source mask and RGB are misaligned")
    if map_x.shape != map_y.shape or map_x.shape != valid.shape:
        raise ValueError("panel inverse-map arrays are misaligned")
    height, width = canvas_shape
    if (
        map_x.shape[0] != height
        or corner_x < 0
        or corner_x + map_x.shape[1] > width
    ):
        raise ValueError("panel inverse map is outside the target canvas")
    safe_x = np.where(valid, map_x, -1.0).astype(np.float32, copy=False)
    safe_y = np.where(valid, map_y, -1.0).astype(np.float32, copy=False)
    local_mask = (
        cv2.remap(
            source.astype(np.uint8),
            safe_x,
            safe_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    ) & valid
    target_count = int(np.count_nonzero(local_mask))
    audit: dict[str, object] = {
        "candidate_id": int(candidate.candidate_id),
        "panel_index": int(panel_index),
        "frame_id": int(frame_id),
        "source_mask_pixel_count": int(np.count_nonzero(source)),
        "target_mask_pixel_count": target_count,
        "mask_mapping": (
            "existing_panel_target_to_source_inverse_map_nearest_only"
        ),
        "translation_used": False,
        "affine_used": False,
        "new_warp_used": False,
        "fill_used": False,
        "generated_color_used": False,
        "accepted": False,
    }
    if (
        candidate.depth_coverage_ratio
        < selected.minimum_depth_coverage_ratio
    ):
        audit["rejection_reason"] = "source_depth_coverage_below_fixed_gate"
        return None, audit
    if target_count == 0:
        audit["rejection_reason"] = "existing_inverse_map_has_no_mask_support"
        return None, audit

    rounded_x = np.rint(map_x[local_mask]).astype(np.int32)
    rounded_y = np.rint(map_y[local_mask]).astype(np.int32)
    inside = (
        (rounded_x >= 0)
        & (rounded_x < source.shape[1])
        & (rounded_y >= 0)
        & (rounded_y < source.shape[0])
    )
    represented = np.zeros(source.shape, dtype=bool)
    represented[rounded_y[inside], rounded_x[inside]] = True
    source_count = int(np.count_nonzero(source))
    source_coverage = float(
        np.count_nonzero(represented & source) / max(1, source_count)
    )
    dominant_ratio, target_component_count = _dominant_component_ratio(
        local_mask
    )
    audit.update(
        {
            "inverse_source_coverage_ratio": source_coverage,
            "dominant_target_component_ratio": dominant_ratio,
            "target_component_count": target_component_count,
        }
    )
    if (
        source_coverage
        < selected.minimum_inverse_source_coverage_ratio
    ):
        audit["rejection_reason"] = (
            "whole_source_mask_not_represented_by_existing_inverse_map"
        )
        return None, audit
    if (
        dominant_ratio
        < selected.minimum_dominant_target_component_ratio
    ):
        audit["rejection_reason"] = (
            "mapped_mask_is_not_one_complete_target_component"
        )
        return None, audit

    target_y, target_x = np.nonzero(local_mask)
    if (
        int(np.min(target_x)) <= 0
        or int(np.max(target_x)) >= map_x.shape[1] - 1
        or int(np.min(target_y)) <= 0
        or int(np.max(target_y)) >= map_x.shape[0] - 1
    ):
        audit["rejection_reason"] = "mapped_mask_touches_panel_map_boundary"
        return None, audit

    sampled = cv2.remap(
        image,
        safe_x,
        safe_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    full_mask = np.zeros(canvas_shape, dtype=bool)
    full_mask[:, corner_x : corner_x + map_x.shape[1]] = local_mask
    full_image = np.zeros((*canvas_shape, 3), dtype=np.uint8)
    full_image[:, corner_x : corner_x + map_x.shape[1]] = sampled

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    clarity = float(np.var(laplacian[source]))
    yy, xx = np.nonzero(source)
    center_x = 0.5 * float(source.shape[1] - 1)
    center_y = 0.5 * float(source.shape[0] - 1)
    radius = np.sqrt(
        ((xx.astype(np.float64) - center_x) / max(1.0, center_x)) ** 2
        + ((yy.astype(np.float64) - center_y) / max(1.0, center_y)) ** 2
    )
    centrality = float(np.clip(1.0 - np.median(radius), 0.0, 1.0))
    audit.update(
        {
            "clarity_laplacian_variance": clarity,
            "mask_centrality": centrality,
            "accepted": True,
            "rejection_reason": None,
        }
    )
    return (
        PanelNativeObservation(
            candidate=candidate,
            panel_index=int(panel_index),
            frame_id=int(frame_id),
            source_mask=np.ascontiguousarray(source),
            target_mask=np.ascontiguousarray(full_mask),
            target_image_bgr=np.ascontiguousarray(full_image),
            inverse_source_coverage_ratio=source_coverage,
            dominant_target_component_ratio=dominant_ratio,
            clarity=clarity,
            centrality=centrality,
            audit=audit,
        ),
        audit,
    )


def observation_identity_audit(
    first: PanelNativeObservation,
    second: PanelNativeObservation,
    *,
    config: PanelNativeLockConfig | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Audit one two-view identity without changing either target mask."""

    selected = (
        config
        if isinstance(config, PanelNativeLockConfig)
        else PanelNativeLockConfig.from_mapping(config)
    )
    selected.validate()
    first_voxels = first.candidate.world_voxel_hashes
    second_voxels = second.candidate.world_voxel_hashes
    intersection = len(first_voxels & second_voxels)
    voxel_overlap = float(
        intersection / max(1, min(len(first_voxels), len(second_voxels)))
    )
    centroid_distance = float(
        np.linalg.norm(
            np.asarray(first.candidate.world_centroid_mm)
            - np.asarray(second.candidate.world_centroid_mm)
        )
    )
    area_ratio = float(
        max(
            first.candidate.source_area_pixels,
            second.candidate.source_area_pixels,
        )
        / max(
            1,
            min(
                first.candidate.source_area_pixels,
                second.candidate.source_area_pixels,
            ),
        )
    )
    target_iou, target_smaller_coverage = mask_overlap_metrics(
        first.target_mask, second.target_mask
    )
    passed = bool(
        first.panel_index != second.panel_index
        and voxel_overlap >= selected.minimum_world_voxel_overlap_ratio
        and centroid_distance
        <= selected.maximum_world_centroid_distance_mm
        and area_ratio <= selected.maximum_source_area_ratio
        and target_iou >= selected.minimum_target_mask_iou
        and target_smaller_coverage
        >= selected.minimum_target_smaller_mask_coverage
    )
    return {
        "first_candidate_id": int(first.candidate.candidate_id),
        "second_candidate_id": int(second.candidate.candidate_id),
        "first_panel_index": int(first.panel_index),
        "second_panel_index": int(second.panel_index),
        "first_frame_id": int(first.frame_id),
        "second_frame_id": int(second.frame_id),
        "world_voxel_overlap_ratio": voxel_overlap,
        "world_centroid_distance_mm": centroid_distance,
        "source_area_ratio": area_ratio,
        "target_mask_iou": target_iou,
        "target_smaller_mask_coverage": target_smaller_coverage,
        "rgbd_world_role": "identity_and_merge_split_rejection_only",
        "pass": passed,
    }


def baseline_pair_costs(
    owner_panel_index: np.ndarray,
    nominal_boundaries_x: Sequence[float],
    *,
    corridor_width_pixels: int,
) -> list[PairCorridorEvidence]:
    """Build fixed costs that preserve the already solved formal chain."""

    owner = np.asarray(owner_panel_index, dtype=np.int16)
    if owner.ndim != 2:
        raise ValueError("baseline owner must be a 2D panel-index raster")
    height, width = owner.shape
    if len(nominal_boundaries_x) <= 0:
        raise ValueError("baseline chain has no adjacent boundaries")
    costs: list[PairCorridorEvidence] = []
    columns = np.arange(width, dtype=np.int32)
    for pair_index, nominal in enumerate(nominal_boundaries_x):
        x0 = int(round(float(nominal))) - corridor_width_pixels // 2
        x0 = min(max(0, x0), width - corridor_width_pixels)
        x1 = x0 + corridor_width_pixels
        local_columns = columns[x0:x1]
        values = np.empty((height, corridor_width_pixels), dtype=np.float32)
        for row in range(height):
            right = np.flatnonzero(
                (owner[row] > pair_index) & (columns >= x0) & (columns < x1)
            )
            if right.size:
                boundary = int(right[0])
            else:
                boundary = int(round(float(nominal)))
            values[row] = np.abs(local_columns - boundary).astype(np.float32)
        costs.append(
            PairCorridorEvidence(
                corner_x=x0,
                values=np.ascontiguousarray(values),
                canvas_width=width,
            )
        )
    return costs


__all__ = [
    "PanelNativeLockConfig",
    "PanelNativeObservation",
    "baseline_pair_costs",
    "map_mask_through_existing_inverse",
    "mask_overlap_metrics",
    "observation_identity_audit",
]
