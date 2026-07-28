from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import OpenEXR

from panorama_demo.dual_output import (
    dual_output_final_names,
    stage_dual_output,
)
from panorama_demo.metric_mosaic import MetricMosaicResult


def _metric_result() -> MetricMosaicResult:
    valid = np.array(
        [[False, True, True], [True, True, True]],
        dtype=bool,
    )
    depth = np.array(
        [[np.nan, 500.0, 510.0], [520.0, 530.0, 540.0]],
        dtype=np.float32,
    )
    confidence = np.array(
        [[0, 1000, 2000], [3000, 4000, 5000]],
        dtype=np.uint16,
    )
    owner = np.array(
        [[-1, 10, 10], [11, 11, 11]],
        dtype=np.int32,
    )
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[valid] = (20, 40, 60)
    return MetricMosaicResult(
        image_bgr=image,
        depth_mm=depth,
        confidence_u16=confidence,
        owner_frame_id=owner,
        valid_mask=valid,
        metadata={
            "schema": "gemini305-metric-mosaic/v1",
            "coordinate_system": {"pixel_size_mm": 2.0},
            "strict_v1_metric_complete": True,
            "strict_incomplete_reasons": [],
        },
    )


def test_dual_output_round_trips_and_commits_all_files(
    tmp_path: Path,
) -> None:
    inspection = np.full((3, 4, 3), (10, 20, 30), dtype=np.uint8)
    inspection_owner = np.full((3, 4), 12, dtype=np.int32)
    full_extent = np.zeros((5, 6, 4), dtype=np.uint8)
    full_extent[1:4, 1:5, :3] = (10, 20, 30)
    full_extent[1:4, 1:5, 3] = 255
    full_extent_owner = np.full((5, 6), -1, dtype=np.int32)
    full_extent_owner[1:4, 1:5] = 12
    staged = stage_dual_output(
        tmp_path,
        _metric_result(),
        inspection_bgr=inspection,
        inspection_owner_frame_id=inspection_owner,
        inspection_full_extent_bgra=full_extent,
        inspection_full_extent_owner_frame_id=full_extent_owner,
        inspection_metadata={"backend": "test_multiview"},
    )

    assert all(path.is_file() for path in staged.pending_by_final_name.values())
    staged.commit(tmp_path)
    assert all((tmp_path / name).is_file() for name in dual_output_final_names())
    owner = cv2.imread(
        str(tmp_path / "mosaic_owner.png"), cv2.IMREAD_UNCHANGED
    )
    assert owner.dtype == np.uint16
    assert owner.tolist() == [[0, 11, 11], [12, 12, 12]]
    with OpenEXR.File(str(tmp_path / "mosaic_depth.exr")) as source:
        depth = source.channels()["Z"].pixels
    assert depth.dtype == np.float32
    assert np.isnan(depth[0, 0])
    assert depth[1, 2] == 540.0
    metric_meta = json.loads(
        (tmp_path / "mosaic_meta.json").read_text(encoding="utf-8")
    )
    assert metric_meta["coordinate_system"]["pixel_size_mm"] == 2.0
    inspection_meta = json.loads(
        (tmp_path / "inspection_meta.json").read_text(encoding="utf-8")
    )
    assert inspection_meta["unowned_pixel_count"] == 0
    assert inspection_meta["full_extent_transparent_pixel_count"] == 18
