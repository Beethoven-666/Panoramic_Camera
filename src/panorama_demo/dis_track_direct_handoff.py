"""Fixed-gate decisions for direct RGB-D owners from stable DIS tracks.

The final geometry is supplied by ``project_complete_object_owner_from_rgbd``.
This module only compares its immutable target masks and chooses one real RGB
owner; it never fits, shifts, fills, blends, or modifies a target mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class DirectHandoffConfig:
    minimum_projection_count: int = 2
    minimum_source_depth_coverage_ratio: float = 0.90
    minimum_pair_target_iou: float = 0.50
    minimum_pair_smaller_mask_coverage: float = 0.75
    minimum_selected_union_coverage_ratio: float = 0.90
    minimum_dominant_target_component_ratio: float = 0.95
    maximum_target_internal_hole_ratio: float = 0.10
    maximum_track_overlap_ratio: float = 0.15

    def validate(self) -> None:
        if int(self.minimum_projection_count) < 2:
            raise ValueError("Direct handoff needs at least two projections")
        for name in (
            "minimum_source_depth_coverage_ratio",
            "minimum_pair_target_iou",
            "minimum_pair_smaller_mask_coverage",
            "minimum_selected_union_coverage_ratio",
            "minimum_dominant_target_component_ratio",
            "maximum_target_internal_hole_ratio",
            "maximum_track_overlap_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "DirectHandoffConfig":
        payload = {} if value is None else dict(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"unknown direct-handoff configuration keys: {unknown}"
            )
        result = cls(**payload)
        result.validate()
        return result


@dataclass(frozen=True)
class DirectProjectedObservation:
    candidate_id: int
    frame_id: int
    source_panel_index: int
    target_panel_index: int
    target_mask: np.ndarray
    target_image_bgr: np.ndarray
    source_depth_coverage_ratio: float
    clarity: float
    projection_audit: dict[str, object]


@dataclass(frozen=True)
class DirectTrackDecision:
    accepted: bool
    selected_observation: DirectProjectedObservation | None
    accepted_observations: tuple[DirectProjectedObservation, ...]
    audit: dict[str, object]


def mask_pair_metrics(
    first: np.ndarray, second: np.ndarray
) -> tuple[float, float]:
    a = np.asarray(first, dtype=bool)
    b = np.asarray(second, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Direct target masks must be canvas-aligned")
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    smaller = min(int(np.count_nonzero(a)), int(np.count_nonzero(b)))
    return (
        float(intersection / union) if union else 0.0,
        float(intersection / smaller) if smaller else 0.0,
    )


def target_mask_shape_audit(mask: np.ndarray) -> dict[str, object]:
    """Audit fragmentation and enclosed holes without modifying the mask."""

    selected = np.asarray(mask, dtype=bool)
    count = int(np.count_nonzero(selected))
    if selected.ndim != 2 or count == 0:
        return {
            "target_pixel_count": count,
            "component_count": 0,
            "dominant_component_ratio": 0.0,
            "internal_hole_pixel_count": 0,
            "internal_hole_ratio": 0.0,
        }
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(
        selected.astype(np.uint8), 8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    dominant = int(np.max(areas)) if areas.size else 0
    yy, xx = np.nonzero(selected)
    x0, x1 = int(np.min(xx)), int(np.max(xx)) + 1
    y0, y1 = int(np.min(yy)), int(np.max(yy)) + 1
    local = selected[y0:y1, x0:x1]
    inverse = np.pad(
        (~local).astype(np.uint8), 1, mode="constant", constant_values=1
    )
    outside = np.zeros(
        (inverse.shape[0] + 2, inverse.shape[1] + 2), np.uint8
    )
    flood = inverse.copy()
    cv2.floodFill(flood, outside, (0, 0), 2)
    holes = flood[1:-1, 1:-1] == 1
    hole_count = int(np.count_nonzero(holes))
    return {
        "target_pixel_count": count,
        "component_count": int(component_count - 1),
        "dominant_component_ratio": float(dominant / count),
        "internal_hole_pixel_count": hole_count,
        "internal_hole_ratio": float(hole_count / max(1, count + hole_count)),
    }


def evaluate_direct_track(
    track_id: int,
    observations: Sequence[DirectProjectedObservation],
    *,
    config: DirectHandoffConfig | Mapping[str, object] | None = None,
) -> DirectTrackDecision:
    """Choose one direct owner only when two measured projections agree."""

    selected_config = (
        config
        if isinstance(config, DirectHandoffConfig)
        else DirectHandoffConfig.from_mapping(config)
    )
    selected_config.validate()
    checked: list[DirectProjectedObservation] = []
    observation_audits: list[dict[str, object]] = []
    for observation in observations:
        shape = target_mask_shape_audit(observation.target_mask)
        accepted = bool(
            observation.source_depth_coverage_ratio
            >= selected_config.minimum_source_depth_coverage_ratio
            and shape["dominant_component_ratio"]
            >= selected_config.minimum_dominant_target_component_ratio
            and shape["internal_hole_ratio"]
            <= selected_config.maximum_target_internal_hole_ratio
        )
        observation_audits.append(
            {
                "candidate_id": int(observation.candidate_id),
                "frame_id": int(observation.frame_id),
                "source_panel_index": int(observation.source_panel_index),
                "target_panel_index": int(observation.target_panel_index),
                "source_depth_coverage_ratio": float(
                    observation.source_depth_coverage_ratio
                ),
                "clarity_laplacian_variance": float(observation.clarity),
                "shape": shape,
                "fixed_observation_gate_pass": accepted,
                "projection": dict(observation.projection_audit),
            }
        )
        if accepted:
            checked.append(observation)
    base_audit: dict[str, object] = {
        "track_id": int(track_id),
        "input_projection_count": len(observations),
        "fixed_observation_gate_count": len(checked),
        "observations": observation_audits,
        "accepted": False,
        "direct_world_projection_only": True,
        "translation_used": False,
        "affine_used": False,
        "fitted_warp_used": False,
        "pose_interpolation_used": False,
        "hole_fill_used": False,
        "generated_rgb_used": False,
        "blend_used": False,
    }
    if len(checked) < selected_config.minimum_projection_count:
        return DirectTrackDecision(
            accepted=False,
            selected_observation=None,
            accepted_observations=(),
            audit={
                **base_audit,
                "reason": "fewer_than_two_fixed_gate_direct_projections",
            },
        )

    pair_audits: list[dict[str, object]] = []
    peers: dict[int, list[int]] = {index: [] for index in range(len(checked))}
    for first_index, first in enumerate(checked):
        for second_index in range(first_index + 1, len(checked)):
            second = checked[second_index]
            iou, smaller_coverage = mask_pair_metrics(
                first.target_mask, second.target_mask
            )
            passed = bool(
                iou >= selected_config.minimum_pair_target_iou
                and smaller_coverage
                >= selected_config.minimum_pair_smaller_mask_coverage
            )
            pair_audits.append(
                {
                    "first_candidate_id": int(first.candidate_id),
                    "second_candidate_id": int(second.candidate_id),
                    "first_frame_id": int(first.frame_id),
                    "second_frame_id": int(second.frame_id),
                    "target_mask_iou": iou,
                    "target_smaller_mask_coverage": smaller_coverage,
                    "pass": passed,
                }
            )
            if passed:
                peers[first_index].append(second_index)
                peers[second_index].append(first_index)
    eligible = [index for index, values in peers.items() if values]
    if not eligible:
        return DirectTrackDecision(
            accepted=False,
            selected_observation=None,
            accepted_observations=(),
            audit={
                **base_audit,
                "pair_audits": pair_audits,
                "reason": "no_two_selected_panel_target_masks_are_iou_consistent",
            },
        )

    ranked: list[
        tuple[
            tuple[int, float, float, float, int],
            int,
            tuple[int, ...],
            float,
        ]
    ] = []
    for index in eligible:
        support_indices = (index, *peers[index])
        support = tuple(checked[value] for value in support_indices)
        union = np.logical_or.reduce(
            [item.target_mask for item in support]
        )
        selected_pixels = int(np.count_nonzero(checked[index].target_mask))
        union_pixels = int(np.count_nonzero(union))
        union_coverage = float(selected_pixels / max(1, union_pixels))
        score = (
            len(support_indices),
            union_coverage,
            checked[index].source_depth_coverage_ratio,
            checked[index].clarity,
            -checked[index].frame_id,
        )
        ranked.append((score, index, support_indices, union_coverage))
    _, selected_index, support_indices, union_coverage = max(
        ranked, key=lambda item: item[0]
    )
    selected_observation = checked[selected_index]
    support = tuple(checked[value] for value in support_indices)
    if (
        union_coverage
        < selected_config.minimum_selected_union_coverage_ratio
    ):
        return DirectTrackDecision(
            accepted=False,
            selected_observation=None,
            accepted_observations=(),
            audit={
                **base_audit,
                "pair_audits": pair_audits,
                "selected_candidate_id": int(
                    selected_observation.candidate_id
                ),
                "selected_frame_id": int(selected_observation.frame_id),
                "selected_target_union_coverage_ratio": union_coverage,
                "reason": "single_owner_does_not_cover_consistent_target_union",
            },
        )
    return DirectTrackDecision(
        accepted=True,
        selected_observation=selected_observation,
        accepted_observations=support,
        audit={
            **base_audit,
            "pair_audits": pair_audits,
            "selected_candidate_id": int(
                selected_observation.candidate_id
            ),
            "selected_frame_id": int(selected_observation.frame_id),
            "selected_source_panel_index": int(
                selected_observation.source_panel_index
            ),
            "target_panel_index": int(
                selected_observation.target_panel_index
            ),
            "consistent_projection_count": len(support),
            "consistent_frame_ids": [
                int(item.frame_id) for item in support
            ],
            "selected_target_union_coverage_ratio": union_coverage,
            "single_complete_rgb_owner": True,
            "reason": "accepted_single_direct_rgbd_owner",
            "accepted": True,
        },
    )


__all__ = [
    "DirectHandoffConfig",
    "DirectProjectedObservation",
    "DirectTrackDecision",
    "evaluate_direct_track",
    "mask_pair_metrics",
    "target_mask_shape_audit",
]
