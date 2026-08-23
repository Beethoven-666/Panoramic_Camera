"""Process-local forced-GraphCut/MultiBand visual ablation.

Loaded only through PYTHONPATH for this one isolated visual experiment.  It
records the original geometry failure as a bypass reason, then bypasses the
GraphCut topology admission checks so its labels can reach the existing
safe-background MultiBand stage.  It never edits repository source or config.
"""
from dataclasses import replace

from panorama_demo import video_v61_renderer as renderer
from panorama_demo.video_graphcut_seam import VideoGraphCutResult


_evaluate = renderer.evaluate_v61_geometry_gate
_solve = renderer.solve_video_graphcut_seam


def _evaluate_without_veto(*args, **kwargs):
    audit = _evaluate(*args, **kwargs)
    if audit.accepted:
        return audit
    reason = audit.rejection_reason or "geometry_gate_failed"
    return replace(
        audit,
        accepted=True,
        rejection_reason=f"visual_ablation_bypassed:{reason}",
    )


def _solve_with_forced_admission(*args, **kwargs):
    result = _solve(*args, **kwargs)
    audit = replace(
        result.audit,
        accepted=True,
        rejection_reason=(
            None if result.audit.accepted
            else "visual_ablation_forced_graphcut_admission:"
            + (result.audit.rejection_reason or "graphcut_rejected")
        ),
    )
    return VideoGraphCutResult(result.choose_new, audit)


renderer.evaluate_v61_geometry_gate = _evaluate_without_veto
renderer.solve_video_graphcut_seam = _solve_with_forced_admission
renderer._real_internal_handoff = lambda *args, **kwargs: True
renderer._old_to_new_monotone = lambda *args, **kwargs: True
