from __future__ import annotations

import importlib
from pathlib import Path

from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_cuda_long_line import detect_and_track_cuda_long_lines
from panorama_demo.video_cuda_mesh import fit_cuda_coarse_to_fine_local_mesh
from panorama_demo.video_v2_route import is_cuda_c9_positive_jacobian_line_mesh_implementation


def _torch():
    return importlib.import_module("torch")


def test_automatic_long_line_detector_uses_pixels_and_raft_not_annotations():
    torch = _torch()
    bgr = torch.zeros((3, 64, 96), dtype=torch.uint8)
    # A real image-space vertical long edge, with a bidirectionally consistent
    # resident RAFT field.  No annotation input exists in this API.
    bgr[:, :, 30:32] = 255
    forward = torch.zeros((64, 96, 2), dtype=torch.float32)
    forward[..., 0] = 1.0
    evidence = detect_and_track_cuda_long_lines(
        torch, bgr=bgr, forward_xy=forward, backward_xy=-forward,
        safe_mask=torch.ones((64, 96), dtype=torch.bool), minimum_length_px=32,
    )

    assert evidence.audit["annotation_input"] is False
    assert evidence.audit["detected_line_pixel_count"] >= 64
    assert evidence.audit["raft_tracked_line_pixel_count"] >= 64
    assert int(evidence.tracked_mask.sum()) >= 64


def test_coarse_to_fine_mesh_retains_existing_final_grid_constraints():
    torch = _torch()
    height, width = 64, 96
    flow = torch.zeros((height, width, 2), dtype=torch.float32)
    flow[..., 0] = 1.0
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    train = ((xx + yy) & 1) == 0
    result = fit_cuda_coarse_to_fine_local_mesh(
        torch, flow_xy=flow, training_mask=train, held_out_mask=~train,
        safe_mask=torch.ones((height, width), dtype=torch.bool),
        protected_mask=torch.zeros((height, width), dtype=torch.bool),
        protected_boundary_taper_px=8,
    )

    assert result.audit["coarse_to_fine_levels"] == [9, 5]
    assert result.audit["final_grid_constraints_audited"] is True
    assert result.audit["minimum_jacobian"] >= 0.05
    assert result.audit["minimum_local_scale"] >= 0.70
    assert result.audit["maximum_local_scale"] <= 1.40


def test_c9_immutable_candidate_is_registered_as_candidate_only():
    spec = build_algorithm_spec(
        Path("configs/video_candidates/C9_positive_jacobian_line_mesh.yaml")
    )
    assert spec.required_components[-1] == "c9_line_preserving_layered_mesh"
    assert is_cuda_c9_positive_jacobian_line_mesh_implementation(
        role=spec.role, algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    )
    assert not is_cuda_c9_positive_jacobian_line_mesh_implementation(
        role="production", algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    )
