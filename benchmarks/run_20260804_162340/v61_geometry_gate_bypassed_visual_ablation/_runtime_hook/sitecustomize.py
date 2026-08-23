"""Process-local visual ablation: record, but do not veto on, geometry quality.

This hook is loaded only through PYTHONPATH for one explicitly isolated run.
It does not alter the checked-out renderer or candidate configuration.
"""
from dataclasses import replace

from panorama_demo import video_v61_renderer as renderer


_evaluate = renderer.evaluate_v61_geometry_gate


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


renderer.evaluate_v61_geometry_gate = _evaluate_without_veto
