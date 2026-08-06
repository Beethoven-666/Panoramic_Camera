from __future__ import annotations

import numpy as np

from panorama_demo.video_safe_multiband import blend_safe_background_multiband


def test_multiband_is_narrow_safe_and_retains_dominant_real_owner():
    first = np.full((32, 80, 4), (30, 50, 70, 255), dtype=np.uint8)
    second = np.full_like(first, (160, 130, 100, 255))
    owner = np.full((32, 80), 10, dtype=np.int32)
    owner[:, 40:] = 20
    safe = np.ones((32, 80), dtype=bool)
    safe[12:20, 36:45] = False
    result = blend_safe_background_multiband(
        first, second, owner, first_frame_id=10, second_frame_id=20,
        safe_mask=safe, band_pixels=8, levels=3,
    )
    assert result.audit["applied"] is True
    assert np.array_equal(result.dominant_owner_frame_id, owner)
    assert not np.any(result.blend_mask[12:20, 36:45])
    assert result.audit["protected_intersection_pixel_count"] == 0
