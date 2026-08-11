from __future__ import annotations

import numpy as np

from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s12_schedule import build_s012_schedule
from panorama_demo.video_s13_vertical import estimate_s13_vertical, render_s13_p1_from_raw


def _calibration() -> CameraIntrinsics:
    return CameraIntrinsics(96, 64, 80.0, 80.0, 47.5, 31.5, ())


def _textured_image(seed: int, vertical_shift: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)
    if vertical_shift:
        matrix = np.float32([[1, 0, 0], [0, 1, vertical_shift]])
        import cv2

        image = cv2.warpAffine(image, matrix, (96, 64), borderMode=cv2.BORDER_REFLECT)
    return image


def test_vertical_solution_uses_required_shoulder_gain_and_zero_missing_rows() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule((0, 1), (47.5, 55.5), calibration, hard_internal_width_px=None)
    images = {0: np.zeros((64, 96, 3), np.uint8), 1: np.zeros((64, 96, 3), np.uint8)}
    solution = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    assert 64 <= solution.shoulder_width_px <= 128
    assert set(solution.gain_scores) == {"0.0", "0.25", "0.5", "1.0"}
    assert solution.selected_gain in {0.0, 0.25, 0.5, 1.0}
    assert np.all(solution.local_row_residuals[0] == 0.0)
    assert solution.pairs[0].status == "local_zero"


def test_p1_formally_samples_each_raw_contributor_once() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule((0, 1, 2), (47.5, 55.5, 63.5), calibration, hard_internal_width_px=None)
    base = _textured_image(7)
    images = {0: base, 1: _textured_image(7, 2), 2: _textured_image(7, 4)}
    solution = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    calls: list[int] = []

    def load(frame_id: int) -> np.ndarray:
        calls.append(frame_id)
        return images[frame_id]

    result = render_s13_p1_from_raw(schedule, calibration, load, solution)
    assert calls == [0, 1, 2]
    assert result.remap_invocations == 3
    assert np.array_equal(result.valid_mask, result.pixel_provenance["owner_frame_id"] >= 0)
    assert np.all(result.pixel_provenance["secondary_frame_id"] == -1)
    assert np.all(result.pixel_provenance["secondary_weight"] == 0.0)


def test_pair_application_bands_are_nonoverlapping_and_bounded() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule((0, 1, 2, 3), (47.5, 55.5, 63.5, 71.5),
                                   calibration, hard_internal_width_px=None)
    image = _textured_image(3)
    solution = estimate_s13_vertical(schedule, calibration, lambda _frame_id: image)
    bands = [(pair.application_left_x, pair.application_right_x) for pair in solution.pairs]
    assert all(0 <= right - left <= 16 for left, right in bands)
    assert all(right <= next_left for (_left, right), (next_left, _next_right) in zip(bands[:-1], bands[1:]))
    assert solution.audit["translation_rotation_affine_seam_photometric_blend_depth_mesh_enabled"] is False
