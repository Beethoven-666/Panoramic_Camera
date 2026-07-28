from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_dis_track_reference_coverage.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "audit_dis_track_reference_coverage", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_panel_choice_prefers_mapped_support_then_world_proximity() -> None:
    selected = _MODULE.choose_best_panel_result(
        [
            {
                "panel_index": 3,
                "mapped_pixel_count": 90,
                "anchor_distance_mm": 1.0,
            },
            {
                "panel_index": 4,
                "mapped_pixel_count": 100,
                "anchor_distance_mm": 50.0,
            },
            {
                "panel_index": 5,
                "mapped_pixel_count": 100,
                "anchor_distance_mm": 20.0,
            },
        ]
    )

    assert selected["panel_index"] == 5


def test_cross_view_coverage_requires_one_source_to_cover_union() -> None:
    union = np.asarray([1, 2, 3, 4, 5], dtype=np.int32)

    assert _MODULE.footprint_coverage_ratio(
        np.asarray([1, 2, 3, 4, 5]), union
    ) == 1.0
    assert _MODULE.footprint_coverage_ratio(
        np.asarray([1, 2, 3]), union
    ) == 0.6
