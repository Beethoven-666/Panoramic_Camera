"""Immutable decision authority for the S013 M6.2 CPU/GPU executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

import cv2

from .cuda_backend import remap as accelerated_remap
from .video_s13_blend import S13BlendConfig, S13BlendPlan, select_s13_blend_plans
from .video_s13_m6 import _source_frame_ids
from .video_s13_m6_cuda import S13M6SourceROI, build_s13_m6_source_rois
from .video_s13_photometric import (
    S13PhotometricConfig, S13PhotometricSampleSet, S13PhotometricSolution,
    apply_s13_photometric_linear, extract_s13_photometric_samples,
    solve_s13_photometric, srgb_to_linear_bgr,
)
from .video_s13_replay import S13VerifiedP2


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class S13M62ExecutionPlan:
    """All decisions shared by CPU reference and GPU shadow execution.

    Pixel executors deliberately receive no decision-making callbacks.  This
    prevents CUDA from selecting a different photometric or blend model.
    """

    p2: S13VerifiedP2
    photometric_solution: S13PhotometricSolution
    photometric_samples: tuple[S13PhotometricSampleSet, ...]
    sample_masks: dict[str, np.ndarray]
    raw_by_frame: dict[int, np.ndarray]
    source_rois: tuple[S13M6SourceROI, ...]
    corrected_rois: dict[int, np.ndarray]
    blend_plans: tuple[S13BlendPlan, ...]
    force_owner_only_pair_indices: frozenset[int]
    valid_mask: np.ndarray
    owner_source_index: np.ndarray
    p2_image: np.ndarray
    photometric_config: S13PhotometricConfig
    blend_config: S13BlendConfig


def build_s13_m62_execution_plan(
    p2: S13VerifiedP2,
    image_loader: Callable[[int], np.ndarray],
    *,
    photometric_config: S13PhotometricConfig = S13PhotometricConfig(),
    blend_config: S13BlendConfig = S13BlendConfig(),
    force_identity_owner_only: bool = False,
    force_owner_only_pair_indices: frozenset[int] = frozenset(),
) -> S13M62ExecutionPlan:
    """Build the CPU decision plan once, before either pixel backend runs."""

    samples, masks, raw_cache = extract_s13_photometric_samples(
        p2.replay_pairs, image_loader, canvas_shape=p2.valid_mask.shape,
        config=photometric_config,
    )
    frame_ids = _source_frame_ids(p2)
    solution = solve_s13_photometric(
        samples, frame_ids=frame_ids, config=photometric_config,
        force_identity=force_identity_owner_only,
    )
    source_rois = build_s13_m6_source_rois(p2, frame_ids)
    frozen_masks = {name: _readonly(value) for name, value in masks.items()}
    frozen_raw = {int(frame): _readonly(value) for frame, value in raw_cache.items()}
    corrected: dict[int, np.ndarray] = {}
    for parameter, roi in zip(solution.source_parameters, source_rois, strict=True):
        sampled = accelerated_remap(raw_cache[parameter.frame_id], roi.map_u, roi.map_v,
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        value = apply_s13_photometric_linear(srgb_to_linear_bgr(sampled), parameter)
        value[~roi.mapped] = 0.0
        corrected[parameter.source_index] = _readonly(value)
    roi_by_source = {roi.source_index: roi for roi in source_rois}
    def pair_provider(pair: object) -> tuple[np.ndarray, np.ndarray]:
        def tile(source: int) -> np.ndarray:
            roi = roi_by_source[source]
            return corrected[source][:, pair.corridor_x0 - roi.x0:pair.corridor_x1 - roi.x0]
        return tile(pair.left_source_index), tile(pair.right_source_index)
    plans, _ = select_s13_blend_plans(
        p2.replay_pairs, samples, {}, canvas_shape=p2.valid_mask.shape,
        corrected_pair_provider=pair_provider, config=blend_config,
        force_owner_only=force_identity_owner_only,
        force_owner_only_pair_indices=force_owner_only_pair_indices,
    )
    frozen_plans = tuple(S13BlendPlan(
        plan.transaction, _readonly(plan.secondary_weight), _readonly(plan.safe_mask), _readonly(plan.protected_mask)
    ) for plan in plans)
    return S13M62ExecutionPlan(
        p2=p2, photometric_solution=solution, photometric_samples=tuple(samples),
        sample_masks=frozen_masks, raw_by_frame=frozen_raw, source_rois=source_rois,
        corrected_rois=corrected, blend_plans=frozen_plans,
        force_owner_only_pair_indices=frozenset(int(index) for index in force_owner_only_pair_indices),
        valid_mask=_readonly(p2.valid_mask),
        owner_source_index=_readonly(np.asarray(p2.provenance["owner_source_index"], np.int32)),
        p2_image=_readonly(p2.result_image), photometric_config=photometric_config,
        blend_config=blend_config,
    )


__all__ = ["S13M62ExecutionPlan", "build_s13_m62_execution_plan"]
