from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from panorama_demo.quality import FrameQuality, MotionEstimate
from panorama_demo.session import CameraIntrinsics, RGBDFrame
from panorama_demo.video_s1_config import load_s1_config
from panorama_demo.video_s1_renderer import compose_owner_panorama, render_video_s1


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_s1_config(ROOT / "configs/video_candidates/S01_output_first_vertical_alignment_v1.yaml")


def _quality() -> FrameQuality:
    return FrameQuality(100.0, 30.0, 0.5, 0.0, 0.0, 40.0, 20.0)


def _frame(tmp_path: Path, frame_id: int, image: np.ndarray) -> RGBDFrame:
    color = tmp_path / f"color_{frame_id}.png"
    depth = tmp_path / f"depth_{frame_id}.png"
    assert cv2.imwrite(str(color), image)
    assert cv2.imwrite(str(depth), np.full(image.shape[:2], 1000, dtype=np.uint16))
    return RGBDFrame(frame_id, color, depth, 1.0, frame_id * 1000, 5, 1)


def test_renderer_uses_same_layout_and_one_joint_remap_per_real_source(tmp_path: Path) -> None:
    height, width = 96, 160
    rng = np.random.default_rng(18)
    left = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    right = cv2.warpAffine(left, np.float32([[1, 0, 0], [0, 1, 2]]), (width, height))
    frames = (_frame(tmp_path, 10, left), _frame(tmp_path, 11, right))
    calibration = CameraIntrinsics(width, height, 130.0, 130.0, width / 2, height / 2, (0.0,) * 8)

    result = render_video_s1(
        frames,
        calibration,
        (MotionEstimate(10.0, 0.0, 100, 0.9, 0.8, "features"),),
        (_quality(), _quality()),
        scan_direction=1,
        analysis_width=160,
        config=CONFIG,
    )

    assert result.remap_invocations == len(result.layouts) == 2
    assert result.s0_panorama.shape == result.s1_panorama.shape == (height, 170, 3)
    assert result.layouts[0].owner_right_x == result.layouts[1].owner_left_x
    assert np.all(result.owner_map[result.valid_mask] >= 0)
    assert result.pair_results[0].alignment.pair_status in {"s1_ok", "s1_zero_correction"}


def test_owner_invalid_pixels_fall_back_only_to_adjacent_real_contribution() -> None:
    from panorama_demo.video_s1_layout import S1SourceLayout

    layouts = (
        S1SourceLayout(1, 0, 2.0, 0, 2, 0, 4, 0, 4, 0.0),
        S1SourceLayout(2, 1, 4.0, 2, 6, 0, 6, 0, 6, 2.0),
    )
    first = np.full((2, 4, 3), 20, np.uint8)
    second = np.full((2, 6, 3), 80, np.uint8)
    first_valid = np.ones((2, 4), bool)
    first_valid[:, 1] = False
    second_valid = np.ones((2, 6), bool)

    panorama, valid, owner = compose_owner_panorama(
        ((0, first, first_valid), (0, second, second_valid)), layouts,
        canvas_width=6, fill_from_adjacent=True,
    )

    assert valid.all()
    assert np.all(panorama[:, 1] == 80)
    assert set(np.unique(owner)) <= {0, 1}
