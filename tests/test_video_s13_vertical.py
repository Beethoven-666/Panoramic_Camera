from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s12_schedule import build_s012_schedule
from panorama_demo.video_s13_bundle import seal_stage, sha256_file
from panorama_demo.video_s13_vertical import (
    estimate_s13_vertical,
    load_s13_vertical_solution,
    render_s13_p1_local_patch_image,
    render_s13_p1_from_raw,
    save_s13_vertical_solution,
    vertical_candidate_solution,
)


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


def test_two_worker_measurement_is_exactly_equal_to_serial() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule(
        (0, 1, 2, 3), (47.5, 55.5, 63.5, 71.5), calibration,
        hard_internal_width_px=None,
    )
    base = _textured_image(27)
    images = {
        0: base,
        1: _textured_image(27, 1),
        2: _textured_image(27, 2),
        3: _textured_image(27, 3),
    }
    serial = estimate_s13_vertical(
        schedule, calibration, images.__getitem__, measurement_workers=1
    )
    parallel = estimate_s13_vertical(
        schedule, calibration, images.__getitem__, measurement_workers=2
    )

    assert parallel.global_offsets_px == serial.global_offsets_px
    assert parallel.gain_scores == serial.gain_scores
    assert parallel.pairs == serial.pairs
    assert parallel.selected_gain == serial.selected_gain
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            parallel.local_row_residuals, serial.local_row_residuals, strict=True
        )
    )


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


def test_local_patch_image_matches_full_local_render() -> None:
    calibration = _calibration()
    schedule = build_s012_schedule(
        (0, 1, 2), (47.5, 55.5, 63.5), calibration, hard_internal_width_px=None
    )
    base = _textured_image(31)
    images = {0: base, 1: _textured_image(31, 1), 2: _textured_image(31, 2)}
    solution = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    global_solution = vertical_candidate_solution(
        solution, solution.selected_gain, accepted_local_pairs=(False, False)
    )
    global_result = render_s13_p1_from_raw(
        schedule, calibration, images.__getitem__, global_solution
    )
    patched = render_s13_p1_local_patch_image(
        schedule, calibration, images.__getitem__, global_result, solution
    )
    full = render_s13_p1_from_raw(schedule, calibration, images.__getitem__, solution)
    assert np.array_equal(patched, full.image)


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


def test_sealed_p1_solution_round_trip_is_exact(tmp_path: Path) -> None:
    calibration = _calibration()
    schedule = build_s012_schedule((0, 1, 2), (47.5, 55.5, 63.5), calibration,
                                   hard_internal_width_px=None)
    base = _textured_image(19)
    images = {0: base, 1: _textured_image(19, 1), 2: _textured_image(19, 2)}
    solution = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    before = render_s13_p1_from_raw(schedule, calibration, images.__getitem__, solution)
    p1 = tmp_path / "P1"
    document = save_s13_vertical_solution(p1, solution)
    seal_stage(
        p1,
        completion_name="P1_completion.json",
        schema="gemini305-video-s13-p1-vertical-completion/v2",
        metadata={
            "generation_id": "round-trip",
            "stage": "P1",
            "hard_audit_passed": True,
            "selected_global_gain": solution.selected_gain,
            "vertical_solution_json": "vertical_solution.json",
            "vertical_solution_json_sha256": sha256_file(p1 / "vertical_solution.json"),
            "vertical_solution_npz": "vertical_solution.npz",
            "vertical_solution_npz_sha256": document["npz_sha256"],
        },
    )
    loaded = load_s13_vertical_solution(
        p1, expected_completion_sha256=sha256_file(p1 / "P1_completion.json")
    )
    after = render_s13_p1_from_raw(schedule, calibration, images.__getitem__, loaded)
    assert loaded.global_offsets_px == solution.global_offsets_px
    assert loaded.selected_gain == solution.selected_gain
    assert loaded.pairs == solution.pairs
    assert loaded.gain_scores == solution.gain_scores
    assert loaded.shoulder_width_px == solution.shoulder_width_px
    assert loaded.audit == solution.audit
    assert loaded.gain_global_offsets_px == solution.gain_global_offsets_px
    for expected, observed in zip(solution.local_row_residuals, loaded.local_row_residuals, strict=True):
        assert np.array_equal(observed, expected)
    for gain in solution.gain_local_row_residuals:
        for expected, observed in zip(
            solution.gain_local_row_residuals[gain], loaded.gain_local_row_residuals[gain], strict=True
        ):
            assert np.array_equal(observed, expected)
    assert sha256_file(p1 / "vertical_solution.npz") == document["npz_sha256"]
    assert np.array_equal(after.image, before.image)


def test_legacy_p1_without_v2_metadata_is_not_replayed(tmp_path: Path) -> None:
    p1 = tmp_path / "P1"
    p1.mkdir()
    with (p1 / "vertical_solution.npz").open("wb") as handle:
        np.savez_compressed(handle, global_offsets_px=np.zeros(2, dtype=np.float32))
    with np.testing.assert_raises_regex(ValueError, "legacy_p1_not_replayable"):
        load_s13_vertical_solution(p1)
