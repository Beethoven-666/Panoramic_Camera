from __future__ import annotations

import cv2
import numpy as np
import pytest

from panorama_demo.stitch_sequence import regularize_pair_homography


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
