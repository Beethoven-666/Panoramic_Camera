from __future__ import annotations

import numpy as np

from panorama_demo.geometry_assisted_local_warp import (
    LocalMeshInverseWarp,
    LocalMeshWarpAudit,
    LocalMeshWarpFitResult,
    TileBounds,
)
from panorama_demo.video_local_mesh_evidence import (
    LocalMeshEvidenceAudit,
    LocalMeshEvidenceResult,
    sample_accepted_mesh_from_first_source,
)
from panorama_demo.video_visual_renderer import VideoVisualSource


def _accepted_evidence(shape: tuple[int, int]) -> LocalMeshEvidenceResult:
    height, width = shape
    mask = np.ones(shape, dtype=bool)
    warp = LocalMeshInverseWarp(
        bounds=TileBounds(0.0, 0.0, float(width - 1), float(height - 1)),
        grid_x=np.array([0.0, float(width - 1)]),
        grid_y=np.array([0.0, float(height - 1)]),
        inverse_dx=np.array([[0.0, -1.0], [0.0, -1.0]]),
        inverse_dy=np.zeros((2, 2)),
        active_cells=np.ones((1, 1), dtype=bool),
        same_layer_mask=mask,
        same_layer_origin_xy=(0.0, 0.0),
    )
    mesh_audit = LocalMeshWarpAudit(
        accepted=True, reason="accepted", correspondence_count=100, training_count=50,
        held_out_count=50, active_cell_count=1, largest_connected_active_cell_count=1,
        free_node_count=1, train_error_p95_before_pixels=1.0,
        train_error_p95_after_pixels=0.1, held_out_error_p95_before_pixels=1.0,
        held_out_error_p95_after_pixels=0.1, held_out_error_max_before_pixels=1.0,
        held_out_error_max_after_pixels=0.1, maximum_displacement_pixels=1.0,
        displacement_p95_pixels=1.0, minimum_jacobian_determinant=1.0,
        maximum_jacobian_determinant=1.0, maximum_jacobian_condition=1.0,
        maximum_straight_line_deviation_pixels=0.0,
        boundary_identity_maximum_error_pixels=0.0,
    )
    audit = LocalMeshEvidenceAudit(
        1, 2, "dis", (0, 0, width, height), height * width, height * width,
        None, height * width, 100, 100, None, mesh_audit, False,
    )
    return LocalMeshEvidenceResult(
        fit=LocalMeshWarpFitResult(warp, mesh_audit), same_layer_mask=mask,
        output_points_xy=np.empty((0, 2)), source_points_xy=np.empty((0, 2)), audit=audit,
    )


def test_accepted_mesh_samples_only_the_existing_first_owner():
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 3] = 255
    image[:, :, 0] = np.arange(8, dtype=np.uint8)
    source = VideoVisualSource(frame_id=1, bgra=image)
    owner = np.zeros((8, 8), dtype=bool)
    owner[:, 2:6] = True
    result = sample_accepted_mesh_from_first_source(source, _accepted_evidence((8, 8)), owner_mask=owner)
    assert result.applied_pixel_count > 0
    assert not np.any(result.applied_mask & ~owner)
    assert np.all(result.bgr[~result.applied_mask] == image[..., :3][~result.applied_mask])
