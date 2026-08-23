from __future__ import annotations

import numpy as np

from panorama_demo.video_graphcut_seam import solve_video_graphcut_seam


def test_graphcut_uses_the_real_opencv_finder_and_hard_owner_masks(monkeypatch) -> None:
    import panorama_demo.video_graphcut_seam as seam

    calls: list[object] = []

    class _Finder:
        def find(self, _images, _corners, masks):
            calls.append(True)
            masks[0][:, 80:] = 0
            masks[1][:, :80] = 0

    monkeypatch.setattr(seam.cv2.detail, "GraphCutSeamFinder", lambda _cost: _Finder())
    image = np.zeros((480, 120, 3), dtype=np.uint8)
    result = solve_video_graphcut_seam(
        image, image, np.ones(image.shape[:2], bool), np.ones(image.shape[:2], bool),
        hard_owner_old=np.pad(np.ones((480, 4), bool), ((0, 0), (0, 116))),
    )

    assert calls == [True]
    assert result.audit.graphcut_called
    assert result.audit.valid_pixel_exactly_one_owner
    assert not result.choose_new[:, :4].any()


def test_graphcut_rejects_corridor_outside_frozen_v6_geometry() -> None:
    image = np.zeros((479, 120, 3), dtype=np.uint8)
    try:
        solve_video_graphcut_seam(image, image, np.ones(image.shape[:2], bool), np.ones(image.shape[:2], bool))
    except ValueError as error:
        assert "480px" in str(error)
    else:
        raise AssertionError("expected frozen height validation")
