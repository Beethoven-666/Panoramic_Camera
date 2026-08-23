from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.video_s13_m61_post_quality import (
    ciede2000,
    evaluate_m61_post_render_quality,
    write_m61_top10_atlas,
)


def _threshold() -> dict:
    def lower(target, ceiling):
        return {"direction": "lower_is_better", "quality_role": "C_target_ceiling", "required": True, "target": target, "engineering_ceiling": ceiling}
    return {
        "schema": "gemini305-video-s13-m61-quality-thresholds/v2", "status": "final",
        "metrics": {
            "wide_safe_delta_e00_median": lower(1.0, 1.25), "wide_safe_delta_e00_p95": lower(2.0, 2.5),
            "safe_pair_block_p95_max": lower(3.0, 3.5),
            "visible_safe_seam_count": {**lower(2, 2), "engineering_ceiling_rule": "max(2, ceil(0.02 * pair_count))"},
            "owner_plateau_delta_e00": lower(0.5, 1.0), "clipping_fraction": lower(0.0005, 0.001),
            "dark_lift_median": lower(1.0, 1.0), "dark_lift_p95": lower(2.0, 2.0),
            "ghost_new_material_count": lower(0, 0),
            "b1_gradient_ratio": {"direction": "higher_is_better", "quality_role": "C_target_ceiling", "required": True, "target": 0.92, "engineering_ceiling": 0.9},
            "neutral_chroma_drift": {"direction": "lower_is_better", "quality_role": "B_nonreg_only", "required": True, "nonreg_relative_tolerance": 0.02, "nonreg_absolute_tolerance_or_mde": 0.1},
            "x_slope_added_drift": {"direction": "lower_is_better", "quality_role": "B_nonreg_only", "required": True, "nonreg_relative_tolerance": 0.02, "nonreg_absolute_tolerance_or_mde": 0.1},
        },
        "actionability": {"required_pair_coverage": {"maximum_insufficient_safe_pair_fraction": 0.05}, "quality_minimum_benefit": {"wide_safe_p95_minimum_relative": 0.05}},
    }


def _pairs(height=128, width=32):
    x = np.broadcast_to(np.arange(width), (height, width)).copy()
    y = np.broadcast_to(np.arange(height)[:, None], (height, width)).copy()
    return [{"canvas_x": x, "canvas_y": y, "seam_x_by_row": np.full(height, width // 2),
             "quality_output_safe": np.ones_like(x, bool), "quality_output_protected": np.zeros_like(x, bool),
             "validation": np.ones_like(x, bool)}]


def _evaluate(p2, new, old, *, threshold=None, active=None):
    return evaluate_m61_post_render_quality(p2, new, old, _pairs(), np.zeros(p2.shape[:2], np.int32),
                                            np.ones(p2.shape[:2], bool), threshold or _threshold(), blend_active_mask=active)


def test_ciede2000_identity_and_chroma() -> None:
    left = np.array([[50.0, 0.0, 0.0], [50.0, 20.0, 0.0]])
    right = np.array([[50.0, 0.0, 0.0], [50.0, 0.0, 20.0]])
    values = ciede2000(left, right)
    assert values[0] == 0
    assert values[1] > 10


def test_clean_identity_reaches_target_and_has_no_selection_authority() -> None:
    image = np.full((128, 32, 3), 100, np.uint8)
    report = _evaluate(image, image.copy(), image.copy())
    assert report["quality_state"] == "target"
    assert report["candidate_selection_authority"] is False
    assert report["stage_seal_authority"] is False
    assert report["aggregate"]["wide_safe_delta_e00_p95"] == 0


def test_wide_chroma_seam_is_not_hidden_by_equal_luminance() -> None:
    p2 = np.full((128, 32, 3), 100, np.uint8)
    bad = p2.copy()
    bad[:, 16:, 0] = 155
    bad[:, 16:, 2] = 45
    report = _evaluate(p2, bad, p2)
    assert report["quality_state"] in {"unresolved", "regressed"}
    assert report["aggregate"]["wide_safe_delta_e00_p95"] > 2
    assert report["aggregate"]["visible_safe_seam_count"] == 1


def test_insufficient_vertical_coverage_is_unevaluable() -> None:
    image = np.full((32, 32, 3), 100, np.uint8)
    report = evaluate_m61_post_render_quality(image, image, image, _pairs(32), np.zeros((32, 32), np.int32),
                                              np.ones((32, 32), bool), _threshold())
    assert report["quality_state"] == "unevaluable"


def test_b1_low_gradient_is_explicitly_unevaluable() -> None:
    image = np.full((128, 32, 3), 100, np.uint8)
    active = np.zeros((128, 32), bool)
    active[:, 15:17] = True
    report = _evaluate(image, image, image, active=active)
    assert report["metric_decisions"]["b1_gradient_ratio"]["evaluable"] is False
    assert report["quality_state"] == "unevaluable"


def test_nonregression_role_can_produce_regressed_state() -> None:
    p2 = np.full((128, 32, 3), 100, np.uint8)
    old = p2.copy()
    new = p2.copy()
    new[:, :, 0] = np.tile(np.linspace(50, 150, 32, dtype=np.uint8), (128, 1))
    report = _evaluate(p2, new, old)
    assert report["quality_state"] == "regressed"
    assert report["all_nonregression_pass"] is False


def test_owner_plateau_and_local_block_are_reported() -> None:
    p2 = np.full((128, 32, 3), 100, np.uint8)
    new = p2.copy()
    new[48:128, 16:] = (130, 90, 70)
    owner = np.zeros((128, 32), np.int32)
    owner[:, 16:] = 1
    report = evaluate_m61_post_render_quality(
        p2, new, p2, _pairs(), owner, np.ones((128, 32), bool), _threshold()
    )
    assert report["aggregate"]["owner_plateau_delta_e00"] > 0
    assert report["aggregate"]["safe_pair_block_p95_max"] > 0
    assert report["aggregate"]["wide_safe_delta_e00_median_macro"] is not None
    assert report["aggregate"]["wide_safe_delta_e00_median_micro"] is not None


def test_top10_atlas_data_and_optional_crops(tmp_path: Path) -> None:
    image = np.full((128, 32, 3), 100, np.uint8)
    report = _evaluate(image, image, image)
    assert report["top10_atlas"][0]["pair_index"] == 0
    written = write_m61_top10_atlas(tmp_path, image, _pairs(), report)
    assert len(written) == 1 and written[0].is_file()
