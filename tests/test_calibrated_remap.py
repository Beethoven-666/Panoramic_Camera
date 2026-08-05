from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from panorama_demo.calibrated_remap import camera_points_to_source_pixels


def _intrinsics(distortion: tuple[float, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        width=848,
        height=480,
        fx=408.8442687988281,
        fy=408.7877197265625,
        cx=421.8267517089844,
        cy=236.0321502685547,
        distortion=distortion,
    )


def test_gemini_rational_eight_coefficient_projection_matches_opencv() -> None:
    distortion = (
        -1.1959240436553955,
        0.6872177124023438,
        0.00025656152865849435,
        -0.0003201613435521722,
        -0.11822859197854996,
        -1.1781269311904907,
        0.6627253293991089,
        -0.10863921791315079,
    )
    intrinsics = _intrinsics(distortion)
    rng = np.random.default_rng(11)
    points = rng.normal(size=(37, 53, 3)).astype(np.float64)
    points[..., :2] *= 900.0
    points[..., 2] = rng.uniform(400.0, 5000.0, size=points.shape[:2])

    map_x, map_y, valid = camera_points_to_source_pixels(points, intrinsics)
    matrix = np.asarray(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    expected, _ = cv2.projectPoints(
        points.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        matrix,
        np.asarray(distortion, dtype=np.float64),
    )
    expected = expected.reshape(points.shape[0], points.shape[1], 2)

    assert valid.all()
    np.testing.assert_array_equal(map_x, expected[..., 0].astype(np.float32))
    np.testing.assert_array_equal(map_y, expected[..., 1].astype(np.float32))


def test_non_rational_distortion_uses_the_opencv_compatible_projection() -> None:
    points = np.asarray([[[10.0, 20.0, 1000.0]]], dtype=np.float64)
    intrinsics = _intrinsics((0.01, -0.02, 0.001, -0.001, 0.03))

    map_x, map_y, valid = camera_points_to_source_pixels(points, intrinsics)

    assert valid.tolist() == [[True]]
    assert np.isfinite(map_x).all()
    assert np.isfinite(map_y).all()
