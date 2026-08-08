"""Process-local alignment protection-bypass visual ablation only."""
from dataclasses import replace

import numpy as np

from panorama_demo import video_v61_renderer as renderer


_prepare = renderer._prepare_aligned_sources
_build_guards = renderer.build_video_hard_guards


def _unprotected_guards(*args, **kwargs):
    guards = _build_guards(*args, **kwargs)
    empty = np.zeros_like(guards.protected, dtype=bool)
    return replace(guards, protected=empty, hard_owner_old=empty, hard_owner_new=empty)


def _prepare_with_protection_bypassed(*args, **kwargs):
    # The temporary substitution affects only preview alignment support.  The
    # final geometry gate, GraphCut, and MultiBand receive the original guards.
    original = renderer.build_video_hard_guards
    renderer.build_video_hard_guards = _unprotected_guards
    try:
        return _prepare(*args, **kwargs)
    finally:
        renderer.build_video_hard_guards = original


renderer._prepare_aligned_sources = _prepare_with_protection_bypassed
