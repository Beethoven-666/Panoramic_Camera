from __future__ import annotations

import cv2
import numpy as np
import pytest

from panorama_demo.quality import (
    FrameQuality,
    MotionEstimate,
    assess_capture_quality,
    analyze_frame_quality,
    estimate_translation,
    select_layout_from_motion_estimates,
    select_layout_indices_adaptive,
    select_primary_scan_segment,
    select_render_indices_auto,
)


def _quality(*, dark: float = 0.0, saturated: float = 0.0) -> FrameQuality:
    return FrameQuality(5.0, 25.0, 0.5, dark, saturated, 0.4, 6.0)


def test_capture_quality_rejects_long_motion_blurring_exposure() -> None:
    report = assess_capture_quality(
        [_quality(), _quality(), _quality()],
        [8, 301, 301],
        maximum_exposure_us=1200,
    )

    assert report["quality_pass"] is False
    assert report["exposure_p95_us"] > 30_000
    assert "motion-safe" in str(report["failure_reasons"])


def test_capture_quality_accepts_capped_auto_exposure() -> None:
    report = assess_capture_quality(
        [_quality(), _quality()],
        [7, 8],
        maximum_exposure_us=1200,
    )

    assert report["quality_pass"] is True


def test_capture_quality_rejects_uniformly_blurred_input_without_metadata() -> None:
    blurred = FrameQuality(2.0, 6.0, 0.4, 0.0, 0.0, 0.3, 2.5)

    report = assess_capture_quality([blurred] * 5, [None] * 5)

    assert report["quality_pass"] is False
    assert "sharp detail" in str(report["failure_reasons"])


def _textured_strip() -> np.ndarray:
    rng = np.random.default_rng(13)
    image = rng.integers(20, 236, size=(200, 520, 3), dtype=np.uint8)
    for x in range(20, 500, 50):
        cv2.circle(image, (x, 80 + x % 60), 12, (20, 240, 80), 2)
    return image


def test_frame_quality_prefers_sharp_texture_to_blurred_copy() -> None:
    sharp = _textured_strip()[:, :320]
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=4.0)

    sharp_quality = analyze_frame_quality(sharp)
    blurred_quality = analyze_frame_quality(blurred)

    assert sharp_quality.sharpness > blurred_quality.sharpness
    assert sharp_quality.tenengrad > blurred_quality.tenengrad


def test_feature_motion_estimate_tracks_horizontal_scan() -> None:
    strip = _textured_strip()
    reference = strip[:, 0:320]
    source = strip[:, 12:332]

    motion = estimate_translation(reference, source)

    assert motion.reliable
    assert motion.dx == pytest.approx(12.0, abs=1.5)
    assert abs(motion.dy) < 1.5


def test_adaptive_layout_uses_measured_displacement_not_fixed_stride() -> None:
    strip = _textured_strip()
    thumbnails = [strip[:, offset : offset + 320] for offset in range(0, 109, 12)]

    selection, motions = select_layout_indices_adaptive(thumbnails)

    assert selection.indices[0] == 0
    assert selection.indices[-1] == len(thumbnails) - 1
    assert 2 <= len(selection.indices) <= 4
    assert all(motion.reliable for motion in motions)


def test_adaptive_layout_rejects_an_adjacent_overlap_gap() -> None:
    strip = _textured_strip()
    thumbnails = [strip[:, 0:320], strip[:, 100:420]]

    with pytest.raises(RuntimeError, match="moved too far"):
        select_layout_indices_adaptive(thumbnails)


def test_render_selection_rejects_a_stationary_sequence() -> None:
    qualities = [_quality(), _quality()]
    transforms = [np.eye(3), np.eye(3)]

    with pytest.raises(RuntimeError, match="too short"):
        select_render_indices_auto(qualities, transforms, (200, 320, 3))


def test_render_selection_picks_covered_clear_sources() -> None:
    qualities = [_quality() for _ in range(7)]
    transforms = [
        np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for x in range(0, 421, 70)
    ]

    selected, report = select_render_indices_auto(
        qualities, transforms, (200, 320, 3)
    )

    assert len(selected) >= 2
    assert report["coverage_ratio"] >= 0.95


def test_diagnostic_render_selection_bypasses_absolute_quality_filter() -> None:
    qualities = [
        FrameQuality(1.0, 1.0, 0.01, 0.9, 0.0, 0.0, 0.0)
        for _ in range(5)
    ]
    transforms = [
        np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for x in (0.0, 60.0, 120.0, 180.0, 240.0)
    ]

    with pytest.raises(RuntimeError, match="No exposure-safe"):
        select_render_indices_auto(qualities, transforms, (80, 200, 3))

    selected, report = select_render_indices_auto(
        qualities,
        transforms,
        (80, 200, 3),
        quality_gate=False,
    )

    assert len(selected) >= 2
    assert report["quality_gate"] is False
    assert report["coverage_ratio"] >= 0.95


def test_render_selection_does_not_bridge_with_reverse_progress_candidate() -> None:
    positions = np.asarray(
        [0.0, 40.0, 70.0, 69.0, 68.0, 67.0, 100.0, 130.0, 160.0]
    )
    sharpness = [20.0, 1.0, 90.0, 80.0, 70.0, 100.0, 20.0, 30.0, 20.0]
    qualities = [
        FrameQuality(value, 25.0, 0.5, 0.0, 0.0, 0.4, 6.0)
        for value in sharpness
    ]
    transforms = [
        np.array([[1.0, 0.0, x], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for x in positions
    ]

    selected, report = select_render_indices_auto(
        qualities,
        transforms,
        (80, 100, 3),
        maximum_keyframes=5,
        target_spacing_fraction=0.55,
    )

    selected_positions = positions[selected]
    assert len(selected) <= 5
    assert np.all(np.diff(selected_positions) > 0.0)
    assert float(np.max(np.diff(selected_positions))) <= 66.0 + 1e-6
    assert report["coverage_ratio"] >= 0.95
    assert report["maximum_adjacent_spacing_pixels"] <= 66.0 + 1e-6


def test_render_selection_rejects_one_frame_budget() -> None:
    with pytest.raises(ValueError, match="zero or at least two"):
        select_render_indices_auto(
            [_quality(), _quality()],
            [np.eye(3), np.array([[1, 0, 100], [0, 1, 0], [0, 0, 1]])],
            (200, 320, 3),
            maximum_keyframes=1,
        )


def test_scan_segmentation_trims_stops_and_reverse_motion() -> None:
    def motion(dx: float, reliable: bool = True) -> MotionEstimate:
        return MotionEstimate(
            dx=dx,
            dy=0.2,
            matches=40 if reliable else 0,
            inlier_ratio=0.8 if reliable else 0.0,
            grid_coverage=0.5 if reliable else 0.0,
            method="features",
        )

    motions = (
        [motion(0.0)] * 4
        + [motion(12.0)] * 12
        + [motion(0.0)] * 3
        + [motion(-8.0)] * 5
    )

    segment = select_primary_scan_segment(motions, image_width=320)

    assert segment.start_index == 4
    assert segment.end_index == 16
    assert segment.scan_direction == 1


def test_layout_rejects_large_vertical_jump() -> None:
    motions = [
        MotionEstimate(20.0, 1.0, 40, 0.8, 0.5, "features"),
        MotionEstimate(20.0, 90.0, 40, 0.8, 0.5, "features"),
        MotionEstimate(20.0, 1.0, 40, 0.8, 0.5, "features"),
    ]

    with pytest.raises(RuntimeError, match="vertical motion"):
        select_layout_from_motion_estimates(
            motions, frame_count=4, image_width=320
        )


def test_layout_tail_frame_cannot_exceed_frame_budget() -> None:
    motions = [
        MotionEstimate(dx, 0.0, 40, 0.8, 0.5, "features")
        for dx in (20.0, 20.0, 6.0)
    ]

    with pytest.raises(RuntimeError, match="safe frame budget"):
        select_layout_from_motion_estimates(
            motions,
            frame_count=4,
            image_width=100,
            max_selected=3,
        )
