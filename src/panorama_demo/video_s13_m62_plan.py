"""Immutable decision authority for effective S013 M6.2 pixel executors."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .video_s13_blend import S13BlendConfig, S13BlendPlan, select_s13_blend_plans
from .video_s13_m6 import _source_frame_ids
from .video_s13_m6_cuda import S13M6SourceROI, build_s13_m6_source_rois
from .video_s13_m62_evidence import S13M62EvidenceBundle, extract_s13_m62_evidence
from .video_s13_m62_component import evaluate_s13_m62_q4c_shadow
from .video_s13_m63_solver import (
    S13M63Config,
    S13M63SolveResult,
    solve_s13_m63_photometric,
)
from .video_s13_photometric import (
    S13PhotometricConfig, S13PhotometricSampleSet, S13PhotometricSolution,
    apply_s13_photometric_linear, solve_s13_photometric,
)
from .video_s13_replay import S13VerifiedP2


def _readonly(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array).copy()
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class S13M62DecisionPlan:
    """CPU-owned decisions; no corrected source pixels are cached here."""

    p2: S13VerifiedP2
    image_loader: Callable[[int], np.ndarray]
    photometric_solution: S13PhotometricSolution
    photometric_samples: tuple[S13PhotometricSampleSet, ...]
    evidence: S13M62EvidenceBundle
    source_rois: tuple[S13M6SourceROI, ...]
    blend_plans: tuple[S13BlendPlan, ...]
    force_owner_only_pair_indices: frozenset[int]
    valid_mask: np.ndarray
    owner_source_index: np.ndarray
    p2_image: np.ndarray
    photometric_config: S13PhotometricConfig
    blend_config: S13BlendConfig
    q0_b0_direct_return: bool
    decision_plan_seconds: float
    component_shadow: dict[str, object]
    m63: S13M63SolveResult | None
    decision_timings: dict[str, object]
    retain_runtime_details: bool


S13M62ExecutionPlan = S13M62DecisionPlan


def build_s13_m62_execution_plan(
    p2: S13VerifiedP2,
    image_loader: Callable[[int], np.ndarray],
    *,
    photometric_config: S13PhotometricConfig = S13PhotometricConfig(),
    blend_config: S13BlendConfig = S13BlendConfig(),
    force_identity_owner_only: bool = False,
    force_owner_only_pair_indices: frozenset[int] = frozenset(),
    retain_runtime_details: bool = False,
    m63_config: S13M63Config = S13M63Config(),
) -> S13M62DecisionPlan:
    """Build evidence, model, and blend decisions without corrected ROI pixels."""

    started = time.perf_counter()
    tick = time.perf_counter()
    frame_ids = _source_frame_ids(p2)
    source_rois = tuple(S13M6SourceROI(
        item.source_index, item.frame_id, item.x0, item.x1,
        _readonly(item.map_u), _readonly(item.map_v), _readonly(item.mapped),
        _readonly(item.owner_mask),
    ) for item in build_s13_m6_source_rois(p2, frame_ids))
    roi_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    evidence = extract_s13_m62_evidence(
        p2.replay_pairs, source_rois, image_loader,
        canvas_shape=p2.valid_mask.shape, config=photometric_config,
        retain_runtime_details=retain_runtime_details,
    )
    evidence_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    m63 = (
        solve_s13_m63_photometric(
            evidence.solve_samples,
            evidence.adjacent_samples,
            frame_ids=frame_ids,
            config=m63_config,
            force_identity=force_identity_owner_only,
        )
        if m63_config.enabled
        else None
    )
    solution = (
        m63.solution
        if m63 is not None
        else solve_s13_photometric(
            evidence.solve_samples, frame_ids=frame_ids, config=photometric_config,
            force_identity=force_identity_owner_only,
        )
    )
    solve_seconds = time.perf_counter() - tick
    parameters = {item.source_index: item for item in solution.source_parameters}

    def pair_provider(pair: object) -> tuple[np.ndarray, np.ndarray]:
        sample = evidence.adjacent_samples[int(pair.pair_index)]
        if sample.left_linear_corridor is None or sample.right_linear_corridor is None:
            raise RuntimeError("S1.3 M6.2 compact pair linear evidence is unavailable")
        return (
            apply_s13_photometric_linear(
                sample.left_linear_corridor, parameters[pair.left_source_index]
            ),
            apply_s13_photometric_linear(
                sample.right_linear_corridor, parameters[pair.right_source_index]
            ),
        )

    unsupported = frozenset(evidence.unsupported_cut_pair_indices) | frozenset(
        () if m63 is None else m63.quality_cut_pair_indices
    )
    forced_pairs = frozenset(force_owner_only_pair_indices) | unsupported
    excluded_canvas_mask = np.zeros(p2.valid_mask.shape, dtype=bool)
    if m63 is not None:
        cut_guard_width = int(m63.audit.get("cut_guard_width_px", 12))
        pairs_by_index = {item.pair_index: item for item in p2.replay_pairs}
        for pair_index in m63.quality_cut_pair_indices:
            pair = pairs_by_index[pair_index]
            for row, seam_x in enumerate(np.asarray(pair.seam_x_by_row, np.int32)):
                x0 = max(0, int(seam_x) - cut_guard_width)
                x1 = min(p2.valid_mask.shape[1], int(seam_x) + cut_guard_width + 1)
                excluded_canvas_mask[row, x0:x1] = True
    tick = time.perf_counter()
    plans, _ = select_s13_blend_plans(
        p2.replay_pairs, evidence.adjacent_samples, {},
        canvas_shape=p2.valid_mask.shape, corrected_pair_provider=pair_provider,
        config=blend_config, force_owner_only=force_identity_owner_only,
        force_owner_only_pair_indices=forced_pairs,
        unsupported_cut_pair_indices=unsupported,
        excluded_canvas_mask=excluded_canvas_mask,
    )
    blend_seconds = time.perf_counter() - tick
    frozen_plans = tuple(S13BlendPlan(
        item.transaction, _readonly(item.secondary_weight), _readonly(item.safe_mask),
        _readonly(item.protected_mask),
    ) for item in plans)
    direct = bool(
        solution.model_family == "Q0_identity"
        and all(item.transaction.model == "B0_owner_only" for item in frozen_plans)
    )
    tick = time.perf_counter()
    component_shadow = ({
        "model": "Q4c_quality_cut_component_scalar_gain",
        "authority": "candidate",
        **dict(m63.audit),
    } if m63 is not None else (
        evaluate_s13_m62_q4c_shadow(
            len(frame_ids), evidence.solve_samples, evidence.unsupported_cut_pair_indices
        )
        if evidence.unsupported_cut_pair_indices
        else {
            "model": "Q4c_component_boundary_anchored_scalar_gain",
            "authority": "shadow", "component_count": 1,
            "accepted_component_count": 0, "nonidentity_source_count": 0,
            "would_select": False, "rejection_reasons": ["global_evidence_connected"],
            "cut_guard_changed_pixel_count": 0,
        }
    ))
    component_seconds = time.perf_counter() - tick
    return S13M62DecisionPlan(
        p2=p2, image_loader=image_loader, photometric_solution=solution,
        photometric_samples=evidence.adjacent_samples, evidence=evidence,
        source_rois=source_rois, blend_plans=frozen_plans,
        force_owner_only_pair_indices=forced_pairs,
        valid_mask=_readonly(p2.valid_mask),
        owner_source_index=_readonly(np.asarray(p2.provenance["owner_source_index"], np.int32)),
        p2_image=_readonly(p2.result_image), photometric_config=photometric_config,
        blend_config=blend_config, q0_b0_direct_return=direct,
        decision_plan_seconds=time.perf_counter() - started,
        component_shadow=component_shadow,
        m63=m63,
        decision_timings={
            "source_roi_maps_seconds": roi_seconds,
            "evidence_extract_seconds": evidence_seconds,
            "photometric_solve_seconds": solve_seconds,
            "blend_decision_seconds": blend_seconds,
            "component_shadow_seconds": component_seconds,
            **evidence.performance,
        },
        retain_runtime_details=retain_runtime_details,
    )


__all__ = ["S13M62DecisionPlan", "S13M62ExecutionPlan", "build_s13_m62_execution_plan"]
