"""Immutable C1--C8 pair-plan derivation for the v2 renderer.

The planner is intentionally free of pixels and models.  It turns a locked
candidate identity plus already-measured pair-risk levels into the exact work
that a v2 renderer must audit.  This prevents a candidate YAML from silently
falling through to a legacy visual-seam path with a different component set.
"""

from __future__ import annotations

from typing import Sequence

from .video_algorithm_contract import PairPlan, VideoAlgorithmContractError


_CANDIDATES = (
    "C1_constrained_owner",
    "C2_dis_rgb_mesh",
    "C3_raft_rgb_mesh",
    "C4_raft_rgbd_layered_mesh",
    "C5_object_lock",
    "C6_multiband",
    "C7_photometric_graph",
    "C8_multilabel_window",
)


def build_v2_candidate_pair_plans(
    candidate_id: str,
    source_frame_ids: Sequence[int],
    risk_levels: Sequence[int],
) -> tuple[PairPlan, ...]:
    """Return every adjacent-real-source operation required by C1--C8.

    Risk only controls the local evidence budget (for example RAFT backward
    on risk corridors); it does not create a source or change the ORB pose
    chain.  All C2+ plans retain C1's real-source hard-owner foundation.
    """

    if candidate_id not in _CANDIDATES:
        raise VideoAlgorithmContractError(f"unsupported v2 candidate: {candidate_id}")
    ids = tuple(int(value) for value in source_frame_ids)
    if len(ids) < 2 or ids != tuple(sorted(ids)) or len(set(ids)) != len(ids):
        raise VideoAlgorithmContractError("v2 pair planner requires unique chronological real source ids")
    if len(risk_levels) != len(ids) - 1 or any(level not in (0, 1, 2, 3) for level in risk_levels):
        raise VideoAlgorithmContractError("v2 pair planner needs an adjacent risk level in [0,3] per real edge")
    number = int(candidate_id[1])
    uses_mesh = number >= 2
    uses_raft = number >= 3
    uses_depth = number >= 4
    uses_object = number >= 5
    uses_multiband = number >= 6
    uses_multilabel = number >= 8
    plans: list[PairPlan] = []
    for left, right, risk in zip(ids[:-1], ids[1:], risk_levels, strict=True):
        flow = "raft_small" if uses_raft else ("dis" if uses_mesh else "none")
        plans.append(
            PairPlan(
                left_frame_id=left,
                right_frame_id=right,
                risk_level=int(risk),
                flow_backend=flow,
                use_raft_backward=bool(uses_raft and risk >= 1),
                use_depth_mesh=uses_depth,
                use_open3d=bool(risk >= 2),
                object_lock_required=uses_object,
                seam_mode="multilabel" if uses_multilabel else "curved_hard_owner",
                blend_mode="safe_multiband" if uses_multiband else "none",
            )
        )
    return tuple(plans)


def candidate_component_audit(candidate_id: str) -> dict[str, object]:
    """Expose immutable component lineage without claiming it ran."""

    if candidate_id not in _CANDIDATES:
        raise VideoAlgorithmContractError(f"unsupported v2 candidate: {candidate_id}")
    number = int(candidate_id[1])
    return {
        "candidate_id": candidate_id,
        "c1_constrained_owner": True,
        "c2_dis_mesh": number >= 2,
        "c3_raft_mesh": number >= 3,
        "c4_depth_layer_safety": number >= 4,
        "c5_object_owner_lock": number >= 5,
        "c6_safe_multiband": number >= 6,
        "c7_global_photometric": number >= 7,
        "c8_local_multilabel_owner": number >= 8,
        "planning_only": True,
    }


__all__ = ["build_v2_candidate_pair_plans", "candidate_component_audit"]
