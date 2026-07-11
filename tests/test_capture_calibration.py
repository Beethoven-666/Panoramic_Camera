from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from panorama_demo.capture_orbbec import _calibration_to_dict


def _intrinsic() -> SimpleNamespace:
    return SimpleNamespace(width=1280, height=800, fx=600, fy=601, cx=640, cy=400)


def _distortion() -> SimpleNamespace:
    return SimpleNamespace(k1=0, k2=0, k3=0, k4=0, k5=0, k6=0, p1=0, p2=0)


def test_calibration_flattens_sdk_matrix_arrays() -> None:
    camera = SimpleNamespace(
        depth_intrinsic=_intrinsic(),
        rgb_intrinsic=_intrinsic(),
        depth_distortion=_distortion(),
        rgb_distortion=_distortion(),
        transform=SimpleNamespace(
            rot=np.eye(3, dtype=np.float32),
            transform=np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
        ),
    )
    result = _calibration_to_dict(camera)
    assert result["depth_to_color"]["rotation_row_major"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["depth_to_color"]["translation_mm"] == [1.0, 2.0, 3.0]
