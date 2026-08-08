"""Process-local synthetic-seam MultiBand visual ablation only.

This hook is intentionally not a candidate or production renderer.  It
creates a visible narrow blend only for diagnosis after the real gate and the
GraphCut topology admission have both been bypassed.
"""
from dataclasses import replace

import numpy as np

from panorama_demo import video_v61_renderer as renderer
from panorama_demo.video_graphcut_seam import VideoGraphCutResult
from panorama_demo.video_near_blend import VideoNearBlendConfig


_evaluate = renderer.evaluate_v61_geometry_gate
_solve = renderer.solve_video_graphcut_seam
_apply = renderer.apply_near_multiband


def _evaluate_without_veto(*args, **kwargs):
    audit = _evaluate(*args, **kwargs)
    if audit.accepted:
        return audit
    return replace(audit, accepted=True, rejection_reason="visual_ablation_bypassed:" + (audit.rejection_reason or "geometry_gate_failed"))


def _solve_with_forced_admission(*args, **kwargs):
    result = _solve(*args, **kwargs)
    return VideoGraphCutResult(
        result.choose_new,
        replace(result.audit, accepted=True, rejection_reason=(
            None if result.audit.accepted else "visual_ablation_forced_graphcut_admission:"
            + (result.audit.rejection_reason or "graphcut_rejected")
        )),
    )


def _all_common_overlap(old_valid, new_valid, evidence, guards, **_kwargs):
    return np.asarray(old_valid, bool) & np.asarray(new_valid, bool)


def _apply_synthetic_midline_multiband(old_bgr, new_bgr, owner_bgr, owner_new, eligible, guards, **_kwargs):
    # A deterministic vertical split is a visual probe, not a semantic owner
    # decision.  The existing MultiBand implementation supplies the 8px band.
    labels = np.zeros_like(np.asarray(owner_new, bool))
    common = np.asarray(eligible, bool)
    columns = np.flatnonzero(np.any(common, axis=0))
    if columns.size:
        midpoint = int(columns[(columns.size - 1) // 2])
        labels[:, midpoint:] = True
    unguarded = replace(guards, protected=np.zeros_like(guards.protected, dtype=bool))
    return _apply(
        old_bgr, new_bgr, owner_bgr, labels, common, unguarded,
        config=VideoNearBlendConfig(near_width_px=8),
    )


renderer.evaluate_v61_geometry_gate = _evaluate_without_veto
renderer.solve_video_graphcut_seam = _solve_with_forced_admission
renderer._real_internal_handoff = lambda *args, **kwargs: True
renderer._old_to_new_monotone = lambda *args, **kwargs: True
renderer.build_near_blend_eligible_mask = _all_common_overlap
renderer.apply_near_multiband = _apply_synthetic_midline_multiband
