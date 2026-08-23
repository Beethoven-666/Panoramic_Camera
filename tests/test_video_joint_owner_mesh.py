import numpy as np
import pytest

from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_joint_owner_mesh import (
    JointOwnerMeshError,
    optimise_joint_owner_final_grids,
)


def _inputs(*, baseline: int = 10, protected: bool = False):
    k, h, w = 5, 2, 8
    grids = np.zeros((k, h, w, 2), np.float64)
    for index in range(k):
        grids[index, ..., 0] = -0.8 + index * 0.2
    costs = np.ones((k, h, w), np.float64)
    costs[1, :, 3:] = 0.0
    common = dict(
        source_frame_ids=(10, 11, 12, 13, 14), final_grid_xy=grids,
        source_valid_mask=np.ones((k, h, w), bool), rgb_cost=costs,
        raft_confidence=np.ones((k, h, w), np.float64), depth_cost=np.zeros((k, h, w)),
        source_center_cost=np.zeros((k, h, w)), sharpness_cost=np.ones((k, h, w)),
        baseline_owner_frame_id=np.full((h, w), baseline, np.int32),
        seam_protected_mask=np.full((h, w), protected, bool),
        line_protected_mask=np.zeros((h, w), bool), object_protected_mask=np.zeros((h, w), bool),
    )
    return common


def test_c12_emits_genuine_final_grid_for_final_real_source_sampling():
    result = optimise_joint_owner_final_grids(**_inputs())

    assert result.audit["window_frame_count"] == 5
    assert result.audit["renderer_input"] is True
    assert result.audit["selected_grid_required_for_final_sampling"] is True
    assert result.audit["actual_changed_pixel_count"] > 0
    assert np.any(result.owner_frame_id == 11)
    assert np.allclose(result.final_grid_xy[result.owner_frame_id == 11, 0], -0.6)


def test_c12_locks_protected_objects_lines_and_seams_and_rejects_no_change():
    with pytest.raises(JointOwnerMeshError, match="zero actual"):
        optimise_joint_owner_final_grids(**_inputs(protected=True))


def test_c12_rejects_not_a_genuine_five_source_window():
    values = _inputs()
    values["source_frame_ids"] = (10, 11, 12, 13)
    with pytest.raises(JointOwnerMeshError, match="5--7"):
        optimise_joint_owner_final_grids(**values)


def test_c12_is_an_immutable_candidate_only_measurement_component():
    spec = build_algorithm_spec("configs/video_candidates/C12_joint_owner_mesh_window.yaml")

    assert spec.role == "candidate"
    assert spec.required_components[-1] == "c12_joint_owner_final_grid"
