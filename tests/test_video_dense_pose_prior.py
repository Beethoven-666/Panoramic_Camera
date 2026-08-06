from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from panorama_demo.session import RGBDFrame
from panorama_demo.video_dense_pose_prior import (
    DenseFrameAudit,
    DensePosePriorError,
    ORBPoseAnchor,
    build_dense_real_frame_pose_priors,
)


def _frame(frame_id: int, timestamp_us: int) -> RGBDFrame:
    return RGBDFrame(
        frame_id=frame_id,
        color_path=Path(f"{frame_id}.png"),
        aligned_depth_path=Path(f"{frame_id}_depth.png"),
        depth_scale_mm_per_unit=1.0,
        timestamp_us=timestamp_us,
    )


def _pose(x_mm: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = x_mm
    return result


def _audit(frame_id: int, left: int = 10, right: int = 20) -> DenseFrameAudit:
    return DenseFrameAudit(
        frame_id=frame_id,
        left_anchor_frame_id=left,
        right_anchor_frame_id=right,
        forward_backward_p95_pixels=1.2,
        rgbd_residual_p95_pixels=1.1,
        forward_backward_sample_count=64,
        rgbd_residual_sample_count=64,
    )


def test_dense_pose_priors_keep_two_direct_orb_anchors_and_interpolate_real_middle() -> None:
    priors = build_dense_real_frame_pose_priors(
        (_frame(10, 0), _frame(15, 100_000), _frame(20, 200_000)),
        (ORBPoseAnchor(10, 0, _pose(0.0)), ORBPoseAnchor(20, 200_000, _pose(20.0))),
        audits_by_frame_id={15: _audit(15)},
    )

    assert [item.source_pose_origin for item in priors] == [
        "direct_orb_anchor",
        "interpolated_se3_prior",
        "direct_orb_anchor",
    ]
    assert priors[1].camera_to_world[0, 3] == pytest.approx(10.0)
    assert priors[1].audit["no_extrapolation"] is True
    assert priors[1].audit["dense_evidence"]["accepted"] is True


def test_dense_pose_priors_reject_extrapolation_and_anchor_distance_over_150ms() -> None:
    anchors = (ORBPoseAnchor(10, 0, _pose(0.0)), ORBPoseAnchor(20, 400_000, _pose(20.0)))
    with pytest.raises(DensePosePriorError, match="150 ms"):
        build_dense_real_frame_pose_priors(
            (_frame(10, 0), _frame(15, 200_000), _frame(20, 400_000)),
            anchors,
            audits_by_frame_id={15: _audit(15)},
        )
    with pytest.raises(DensePosePriorError, match="no enclosing"):
        build_dense_real_frame_pose_priors(
            (_frame(5, 0), _frame(10, 100_000), _frame(20, 200_000)),
            (ORBPoseAnchor(10, 100_000, _pose(0.0)), ORBPoseAnchor(20, 200_000, _pose(20.0))),
        )


def test_dense_pose_prior_reports_refined_origin_only_for_bounded_audited_real_frame() -> None:
    delta = _pose(2.0)
    priors = build_dense_real_frame_pose_priors(
        (_frame(10, 0), _frame(15, 100_000), _frame(20, 200_000)),
        (ORBPoseAnchor(10, 0, _pose(0.0)), ORBPoseAnchor(20, 200_000, _pose(20.0))),
        audits_by_frame_id={15: _audit(15)},
        refinements_by_frame_id={15: delta},
    )

    assert priors[1].source_pose_origin == "refined_dense_prior"
    assert priors[1].camera_to_world[0, 3] == pytest.approx(12.0)
    assert priors[1].audit["bounded_refinement"]["translation_mm"] == pytest.approx(2.0)
    with pytest.raises(DensePosePriorError, match="cannot be altered"):
        build_dense_real_frame_pose_priors(
            (_frame(10, 0), _frame(20, 200_000)),
            (ORBPoseAnchor(10, 0, _pose(0.0)), ORBPoseAnchor(20, 200_000, _pose(20.0))),
            refinements_by_frame_id={10: _pose(1.0)},
        )
