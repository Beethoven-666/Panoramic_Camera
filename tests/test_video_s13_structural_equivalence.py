from __future__ import annotations

import numpy as np

from panorama_demo.video_s13_probe_guard import calibrate_probe_guard, guarded_ratio_decision
from panorama_demo.video_s13_structural_equivalence import compare_s13_structural_equivalence


def _stage(image: np.ndarray, owner: int = 0) -> dict[str, object]:
    return {"image": image, "valid_mask": np.ones(image.shape[:2], bool),
            "owner_source_index": np.full(image.shape[:2], owner, np.int32),
            "schedule": (1, 2), "blend_plan": ("B0",)}


def test_identical_stages_are_structurally_and_rgb_equivalent() -> None:
    image = np.full((32, 24, 3), 80, np.uint8)
    reference = {stage: _stage(image) for stage in ("P0", "P1", "P2", "P3")}
    result = compare_s13_structural_equivalence(reference, reference)
    assert result.passed is True
    assert all(row.maximum_abs_dn == 0 for row in result.stages)


def test_rgb_tolerance_does_not_relax_owner_authority() -> None:
    image = np.full((32, 24, 3), 80, np.uint8)
    reference = {stage: _stage(image) for stage in ("P0", "P1", "P2", "P3")}
    candidate = {stage: _stage(image.copy()) for stage in reference}
    candidate["P3"]["image"][0, 0] += 2
    candidate["P2"] = _stage(image.copy(), owner=1)
    result = compare_s13_structural_equivalence(reference, candidate)
    assert result.passed is False
    assert "P2:owner_source_index" in result.structure_failures
    assert next(row for row in result.stages if row.stage_name == "P3").maximum_abs_dn == 2


def test_guarded_probe_never_accepts_an_interval_overlap() -> None:
    calibration = calibrate_probe_guard({"score": 10.0}, {"score": 10.00001})
    assert calibration.epsilon_by_metric["score"] >= 1e-6
    assert guarded_ratio_decision(before=10.0, after=9.0, epsilon=0.01, ratio=0.95) == "pass"
    assert guarded_ratio_decision(before=10.0, after=10.0, epsilon=0.01, ratio=1.0) == "reference_fallback"
    assert guarded_ratio_decision(before=None, after=1.0, epsilon=0.01, ratio=1.0) == "reference_fallback"
