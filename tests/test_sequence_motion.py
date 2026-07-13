from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.session import SessionFrame
from panorama_demo.stitch_sequence import (
    _parse_protected_regions,
    interpolate_translation_transforms,
    regularize_pair_homography,
)


def test_translation_regularization_removes_projective_drift() -> None:
    homography = np.array(
        [[0.97, 0.01, 101.0], [-0.01, 0.99, 2.0], [-0.00008, 0.00001, 1.0]]
    )
    result = regularize_pair_homography(homography, (400, 640, 3), "translation")
    assert np.allclose(result[:2, :2], np.eye(2))
    assert np.allclose(result[2], [0.0, 0.0, 1.0])
    assert 95.0 < result[0, 2] < 110.0
    assert abs(result[1, 2]) < 10.0


def test_similarity_regularization_has_no_projective_terms() -> None:
    angle = np.deg2rad(2.0)
    expected = np.array(
        [
            [np.cos(angle), -np.sin(angle), 80.0],
            [np.sin(angle), np.cos(angle), -3.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = regularize_pair_homography(expected, (400, 640, 3), "similarity")
    assert np.allclose(result, expected, atol=1e-4)


def test_homography_mode_preserves_matrix() -> None:
    matrix = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 2.0], [1e-5, 0.0, 1.0]])
    assert np.allclose(
        regularize_pair_homography(matrix, (100, 200, 3), "homography"), matrix
    )


def test_unknown_motion_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        regularize_pair_homography(np.eye(3), (100, 200, 3), "rigid")


def test_translation_anchor_tracks_the_lower_scan_region() -> None:
    homography = np.array(
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    centered = regularize_pair_homography(
        homography, (800, 1280, 3), "translation", translation_anchor_y=0.5
    )
    lower = regularize_pair_homography(
        homography, (800, 1280, 3), "translation", translation_anchor_y=0.85
    )

    assert centered[0, 2] == pytest.approx(40.0)
    assert lower[0, 2] == pytest.approx(68.0)


def test_interpolate_translation_transforms_for_dense_render_frames(
    tmp_path,
) -> None:
    layout_frames = [
        SessionFrame(10, tmp_path / "10.jpg"),
        SessionFrame(20, tmp_path / "20.jpg"),
    ]
    render_frames = [SessionFrame(15, tmp_path / "15.jpg")]
    transforms = [
        np.eye(3),
        np.array([[1.0, 0.0, -100.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]]),
    ]

    result = interpolate_translation_transforms(
        layout_frames, transforms, render_frames
    )

    np.testing.assert_allclose(
        result[0],
        np.array([[1.0, 0.0, -50.0], [0.0, 1.0, 10.0], [0.0, 0.0, 1.0]]),
    )


def test_parse_protected_regions_maps_frame_ids_to_render_indices(tmp_path) -> None:
    frames = [
        SessionFrame(111, tmp_path / "111.jpg"),
        SessionFrame(188, tmp_path / "188.jpg"),
    ]

    result = _parse_protected_regions(
        ["188:10:20:30:40", "111:1:2:3:4"],
        frames,
    )

    assert result == [(1, 10, 20, 30, 40), (0, 1, 2, 3, 4)]


def test_parse_protected_regions_rejects_non_render_frame(tmp_path) -> None:
    frames = [SessionFrame(111, tmp_path / "111.jpg")]

    with pytest.raises(ValueError, match="is not a render frame"):
        _parse_protected_regions("188:10:20:30:40", frames)
