from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from panorama_demo.fastsam_dis_tracking import (
    FastSAMDISFrameInput,
    track_fastsam_dis_frames,
)
from panorama_demo.fastsam_onnx import FastSAMPolygonProposal
from panorama_demo.inspection_fastsam_track import parse_fastsam_polygons
from panorama_demo.inspection_fastsam_track import polygon_mask
from panorama_demo.session import CameraIntrinsics


def _label_proposal(path: Path, frame_id: int) -> FastSAMPolygonProposal:
    label = path / f"{frame_id:08d}.txt"
    label.write_text("0 0.35 0.3125 0.64 0.3125 0.64 0.675 0.35 0.675\n")
    polygon = parse_fastsam_polygons(label, width=100, height=80)[0]
    mask = np.zeros((80, 100), np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    x, y, w, h = cv2.boundingRect(polygon)
    return FastSAMPolygonProposal(
        score=1.0,
        bbox_xyxy=(float(x), float(y), float(x + w), float(y + h)),
        polygon_xy=polygon,
        mask=mask.astype(bool),
    )


def test_label_constructed_proposals_match_fixed_gate_memory_api(tmp_path: Path) -> None:
    intrinsics = CameraIntrinsics(
        width=100,
        height=80,
        fx=80.0,
        fy=80.0,
        cx=50.0,
        cy=40.0,
        distortion=(),
    )
    frames = []
    for source_index, frame_id in enumerate((8, 9, 10)):
        image = np.zeros((80, 100, 3), np.uint8)
        image[25:55, 35:65] = (30, 80, 180)
        depth = np.full((80, 100), 1000.0, np.float32)
        depth[25:55, 35:65] = 600.0
        frames.append(
            FastSAMDISFrameInput(
                frame_id=frame_id,
                image_bgr=image,
                depth_mm=depth,
                camera_to_world=np.eye(4),
                proposals=(_label_proposal(tmp_path, frame_id),),
            )
        )
    result = track_fastsam_dis_frames(
        frames,
        intrinsics=intrinsics,
        reference_depth_mm=1000.0,
        stable_frame_ids=(8, 10),
    )
    assert [len(frame.candidates) for frame in result.frames] == [1, 1, 1]
    assert all(len(frame.identity_masks_preview) == 1 for frame in result.frames)
    assert len(result.tracks) == 1
    assert result.tracks[0].track_id == 0
    assert result.tracks[0].frame_ids == (8, 9, 10)
    assert result.tracks[0].stable_frame_ids == (8, 10)
    assert result.stable_tracks == result.tracks
    assert result.flow_role == "candidate_identity_evidence_only"
    assert result.flow_used_to_warp_rgb_or_position is False


def test_exact_fastsam_mask_hole_survives_candidate_and_preview(
    tmp_path: Path,
) -> None:
    proposal = _label_proposal(tmp_path, 40)
    exact = proposal.mask.copy()
    exact[38:42, 48:52] = False
    proposal = FastSAMPolygonProposal(
        score=proposal.score,
        bbox_xyxy=proposal.bbox_xyxy,
        polygon_xy=proposal.polygon_xy,
        mask=exact,
    )
    intrinsics = CameraIntrinsics(
        width=100,
        height=80,
        fx=80.0,
        fy=80.0,
        cx=50.0,
        cy=40.0,
        distortion=(),
    )
    image = np.zeros((80, 100, 3), np.uint8)
    image[25:55, 35:65] = (30, 80, 180)
    depth = np.full((80, 100), 1000.0, np.float32)
    depth[20:60, 30:70] = 600.0
    frames = tuple(
        FastSAMDISFrameInput(
            frame_id=frame_id,
            image_bgr=image,
            depth_mm=depth,
            camera_to_world=np.eye(4),
            proposals=(proposal,),
        )
        for frame_id in (40, 41, 42)
    )

    result = track_fastsam_dis_frames(
        frames,
        intrinsics=intrinsics,
        reference_depth_mm=1000.0,
    )

    candidate = result.frames[0].candidates[0]
    restored = polygon_mask(candidate, image.shape[:2])
    assert restored[30, 40]
    assert not restored[40, 50]
    preview = result.frames[0].identity_masks_preview[0]
    assert not preview[
        int(round(40 * 0.25)),
        int(round(50 * 0.25)),
    ]
