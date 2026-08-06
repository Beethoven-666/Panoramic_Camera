from __future__ import annotations

import numpy as np

from panorama_demo.video_d3_object_first_compositor import (
    compose_d3_object_first_dense_source,
)


def test_d3_copies_a_guarded_object_from_one_real_source_without_blending():
    shape = (48, 64)
    first = np.full((*shape, 3), (10, 20, 30), np.uint8)
    second = np.full((*shape, 3), (90, 100, 110), np.uint8)
    first_depth = np.full(shape, 1000, np.float32)
    second_depth = first_depth.copy()
    # An actual depth discontinuity creates the automatic, non-annotation mask.
    second_depth[20:28, 28:36] = 700
    result = compose_d3_object_first_dense_source(
        panorama_bgr=first.copy(), owner_frame_id=np.full(shape, 11, np.int64),
        first_bgr=first, second_bgr=second, first_depth_mm=first_depth,
        second_depth_mm=second_depth, first_valid=np.ones(shape, bool),
        second_valid=np.ones(shape, bool), raft_forward_xy=np.zeros((*shape, 2), np.float32),
        first_frame_id=11, second_frame_id=12,
    )
    assert result.accepted
    assert result.audit["raft_track_used"] is True
    assert result.audit["object_flow_or_warp"] is False
    assert result.audit["object_multiband"] is False
    changed = result.owner_frame_id == result.audit["selected_owner_frame_id"]
    expected = first if result.audit["selected_owner_frame_id"] == 11 else second
    assert np.array_equal(result.panorama_bgr[changed], expected[changed])


def test_d3_rejects_when_no_single_real_source_has_98_percent_guard_support():
    shape = (48, 64)
    depth = np.full(shape, 1000, np.float32)
    other = depth.copy(); other[20:28, 28:36] = 700
    result = compose_d3_object_first_dense_source(
        panorama_bgr=np.zeros((*shape, 3), np.uint8), owner_frame_id=np.full(shape, 11, np.int64),
        first_bgr=np.zeros((*shape, 3), np.uint8), second_bgr=np.ones((*shape, 3), np.uint8),
        first_depth_mm=depth, second_depth_mm=other,
        first_valid=np.zeros(shape, bool), second_valid=np.zeros(shape, bool),
        raft_forward_xy=np.zeros((*shape, 2), np.float32), first_frame_id=11, second_frame_id=12,
    )
    assert not result.accepted
    assert result.audit["reason"] == "no_depth_connected_foreground_component"
