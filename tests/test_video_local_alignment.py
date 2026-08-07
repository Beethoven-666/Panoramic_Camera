from __future__ import annotations

import cv2
import numpy as np

from panorama_demo.video_local_alignment import (
    fit_background_alignment,
    fit_near_protected_alignment,
)
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _evidence(flow: np.ndarray) -> VideoDISPairEvidence:
    height, width = flow.shape[:2]
    zeros = np.zeros((height, width), dtype=np.float32)
    reliable = np.ones((height, width), dtype=bool)
    return VideoDISPairEvidence(
        flow_forward=flow.astype(np.float32), flow_backward=-flow.astype(np.float32),
        fb_error=zeros, rgb_residual=zeros, gradient_residual=zeros,
        occlusion_risk_mask=np.zeros((height, width), dtype=bool),
        correspondence_confidence=np.ones((height, width), dtype=np.float32), reliable_mask=reliable,
        sampled_new_bgra=np.zeros((height, width, 4), dtype=np.uint8),
    )


def test_background_alignment_is_rgb_dis_evidence_only_and_never_samples_rgb() -> None:
    flow = np.zeros((40, 64, 2), dtype=np.float32)
    flow[..., 0] = 4.0
    result = fit_background_alignment(_evidence(flow))

    assert result.audit.accepted
    assert result.audit.selected_model == "translation"
    assert result.matrix is not None
    assert result.mesh_displacement is None
    assert result.audit.maximum_displacement_px == 4.0


def test_background_uses_train_only_bounded_mesh_after_global_models_leave_residual() -> None:
    _, x = np.indices((48, 72), dtype=np.float32)
    flow = np.zeros((48, 72, 2), dtype=np.float32)
    flow[..., 0] = 2.0 * np.sin(x / 8.0)

    result = fit_background_alignment(_evidence(flow))

    assert result.audit.accepted
    assert result.audit.selected_model == "bounded_mesh"
    assert result.matrix is None
    assert result.mesh_displacement is not None
    assert result.audit.outer_boundary_zero_displacement
    assert result.audit.positive_jacobian


def test_near_alignment_follows_identity_translation_rotation_affine_order() -> None:
    height, width = 96, 144
    y, x = np.indices((height, width), dtype=np.float32)
    points = np.column_stack((x.ravel(), y.ravel()))
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 2.8, 1.0)
    targets = cv2.transform(points[None, ...], matrix)[0]
    flow = (targets - points).reshape(height, width, 2)

    result = fit_near_protected_alignment(_evidence(flow))

    assert result.audit.accepted
    assert result.audit.selected_model == "rotation"
    assert abs(result.audit.rotation_degrees or 0.0) <= 3.0
    assert result.audit.rejected_models == ("identity", "translation")


def test_near_homography_fails_closed_without_independent_planarity_evidence() -> None:
    height, width = 48, 72
    y, x = np.indices((height, width), dtype=np.float32)
    points = np.column_stack((x.ravel(), y.ravel()))
    homography = np.array(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.004, 0.0, 1.0)), dtype=np.float32)
    targets = cv2.perspectiveTransform(points[None, ...], homography)[0]
    flow = (targets - points).reshape(height, width, 2)

    result = fit_near_protected_alignment(_evidence(flow), plane_verified=False)

    assert not result.audit.accepted
    assert result.matrix is None
    assert "planarity_unverified" in (result.audit.rejection_reason or "")
