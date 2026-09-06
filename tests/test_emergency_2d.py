import hashlib
import json

import cv2
import numpy as np
import pytest

from panorama_demo.emergency_2d import publish_emergency_2d


def _frame(root, frame_id):
    (root / "color").mkdir(exist_ok=True)
    (root / "depth_aligned").mkdir(exist_ok=True)
    color, depth = f"color/{frame_id}.jpg", f"depth_aligned/{frame_id}.png"
    cv2.imwrite(str(root / color), np.full((24, 40, 3), 90, np.uint8))
    cv2.imwrite(str(root / depth), np.full((24, 40), 500, np.uint16))
    return {"frame_id": frame_id, "color_path": color, "aligned_depth_path": depth,
            "color_sha256": hashlib.sha256((root / color).read_bytes()).hexdigest(),
            "timestamp_us": frame_id * 1000, "row": {"color_exposure": 8}}


def test_single_frame_is_explicit_emergency_without_formal_delivery(tmp_path):
    row = _frame(tmp_path, 1)
    result = publish_emergency_2d(tmp_path, [row], tmp_path / "out", reason="short_capture")
    assert result["formal_2d_published"] is False
    assert result["completion_state"] == "EMERGENCY_2D_PUBLISHED"
    assert not (tmp_path / "out/video_delivery.json").exists()
    expected = cv2.imread(str(tmp_path / row["color_path"]))
    actual = cv2.imread(result["emergency_2d_path"])
    np.testing.assert_array_equal(actual, expected)
    assert json.loads((tmp_path / "out/emergency_delivery.json").read_text())["schema"] == "gemini305-sdk-emergency-2d/v1"


def test_no_frame_fails_and_constant_frames_choose_earlier_id(tmp_path):
    with pytest.raises(ValueError, match="FAILED_NO_VALID_FRAME"):
        publish_emergency_2d(tmp_path, [], tmp_path / "out", reason="none")
    rows = [_frame(tmp_path, index) for index in (1, 2)]
    result = publish_emergency_2d(tmp_path, rows, tmp_path / "out", reason="no_motion")
    assert result["mode"] == "best_complete_single_frame"
    assert result["source_frame_ids"] == [1]
