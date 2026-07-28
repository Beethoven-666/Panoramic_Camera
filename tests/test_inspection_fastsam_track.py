from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.inspection_fastsam_track import (
    build_fastsam_rgbd_candidate,
    flow_forward_backward_consistency,
    flow_predict_mask,
    parse_fastsam_polygons,
    select_unambiguous_one_to_one_matches,
    track_fastsam_rgbd_candidates,
)
from panorama_demo.session import CameraIntrinsics


def test_parse_fastsam_polygon_preserves_contour_order(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "00000000.txt"
    labels.write_text(
        "0 0.2 0.2 0.6 0.2 0.6 0.7 0.2 0.7\n",
        encoding="utf-8",
    )
    polygon = parse_fastsam_polygons(
        labels, width=100, height=80
    )[0]
    assert polygon.tolist() == [
        [20, 16],
        [60, 16],
        [60, 56],
        [20, 56],
    ]


def _candidate(
    candidate_id: int,
    source_index: int,
    colour: tuple[int, int, int],
):
    height, width = 80, 100
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[25:55, 35:65] = colour
    depth = np.full((height, width), 1000.0, dtype=np.float32)
    depth[25:55, 35:65] = 600.0
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=80.0,
        fy=80.0,
        cx=50.0,
        cy=40.0,
        distortion=(),
    )
    result = build_fastsam_rgbd_candidate(
        candidate_id=candidate_id,
        source_index=source_index,
        frame_id=10 + source_index,
        polygon_xy=np.asarray(
            [[35, 25], [64, 25], [64, 54], [35, 54]],
            dtype=np.int32,
        ),
        image_bgr=image,
        depth_mm=depth,
        reliable_depth=np.ones(depth.shape, dtype=bool),
        camera_to_world=np.eye(4, dtype=np.float64),
        intrinsics=intrinsics,
        reference_depth_mm=1000.0,
        sample_stride=3,
    )
    assert result is not None
    return result


def test_fastsam_track_requires_world_rgb_and_contour_agreement() -> None:
    first = _candidate(0, 0, (30, 80, 180))
    second = _candidate(1, 1, (31, 81, 179))
    incompatible = _candidate(2, 1, (220, 220, 20))
    tracks = track_fastsam_rgbd_candidates(
        ((first,), (second, incompatible))
    )
    assert len(tracks) == 1
    assert tracks[0].candidate_ids == (0, 1)
    assert tracks[0].source_indices == (0, 1)


def test_flow_identity_evidence_predicts_mask_and_passes_fb() -> None:
    source = np.zeros((30, 40), dtype=bool)
    source[10:20, 12:22] = True
    forward = np.zeros((30, 40, 2), dtype=np.float32)
    backward = np.zeros_like(forward)
    forward[..., 0] = 1.0
    backward[..., 0] = -1.0
    predicted = flow_predict_mask(source, backward)
    expected = np.zeros_like(source)
    expected[10:20, 13:23] = True
    assert np.array_equal(predicted, expected)
    audit = flow_forward_backward_consistency(
        source, forward, backward
    )
    assert audit["pass"] is True
    assert audit["p95_error_pixels"] == 0.0


def test_one_to_one_match_rejects_merge_split_ambiguity() -> None:
    valid = np.asarray([[True, True], [False, True]])
    score = np.asarray([[0.90, 0.88], [0.0, 0.87]])
    assert select_unambiguous_one_to_one_matches(valid, score) == ()
    valid = np.eye(2, dtype=bool)
    assert select_unambiguous_one_to_one_matches(
        valid, np.eye(2)
    ) == ((0, 0), (1, 1))
