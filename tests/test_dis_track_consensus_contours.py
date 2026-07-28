from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_dis_track_consensus_contours.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "audit_dis_track_consensus_contours", _PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_consensus_requires_support_from_two_distinct_observations() -> None:
    result = _MODULE.consensus_voxels(
        [
            {(0, 0, 0), (1, 0, 0)},
            {(0, 0, 0), (2, 0, 0)},
            {(0, 0, 0), (2, 0, 0)},
        ]
    )

    assert result == {(0, 0, 0), (2, 0, 0)}


def test_seeded_components_exclude_unseeded_view_dependent_attachment() -> None:
    probable = np.zeros((10, 20), dtype=bool)
    probable[2:8, 2:8] = True
    probable[3:7, 14:18] = True
    seed = np.zeros_like(probable)
    seed[4, 4] = True

    accepted = _MODULE.seeded_components(probable, seed)

    assert np.all(accepted[2:8, 2:8])
    assert not np.any(accepted[:, 14:18])
