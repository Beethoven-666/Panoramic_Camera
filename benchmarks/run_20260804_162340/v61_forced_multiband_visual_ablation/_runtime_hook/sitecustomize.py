"""Process-local forced GraphCut plus MultiBand visual ablation only."""
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
    return replace(
        audit,
        accepted=True,
        rejection_reason="visual_ablation_bypassed:" + (audit.rejection_reason or "geometry_gate_failed"),
    )


def _solve_with_forced_admission(*args, **kwargs):
    result = _solve(*args, **kwargs)
    return VideoGraphCutResult(
        result.choose_new,
        replace(
            result.audit,
            accepted=True,
            rejection_reason=(
                None if result.audit.accepted else "visual_ablation_forced_graphcut_admission:"
                + (result.audit.rejection_reason or "graphcut_rejected")
            ),
        ),
    )


def _forced_safe_overlap(old_valid, new_valid, evidence, guards, **_kwargs):
    # Deliberately bypass only the DIS/rgb eligibility thresholds.  Hard guards
    # remain excluded so the existing blender's guard check still holds.
    return np.asarray(old_valid, bool) & np.asarray(new_valid, bool) & ~np.asarray(guards.protected, bool)


def _apply_wider_forced_band(*args, **kwargs):
    # 8px is the frozen maximum safe narrow-band width; this makes the visual
    # ablation observable without introducing full-canvas blending.
    return _apply(*args, config=VideoNearBlendConfig(near_width_px=8))


renderer.evaluate_v61_geometry_gate = _evaluate_without_veto
renderer.solve_video_graphcut_seam = _solve_with_forced_admission
renderer._real_internal_handoff = lambda *args, **kwargs: True
renderer._old_to_new_monotone = lambda *args, **kwargs: True
renderer.build_near_blend_eligible_mask = _forced_safe_overlap
renderer.apply_near_multiband = _apply_wider_forced_band
