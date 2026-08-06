from __future__ import annotations

import pytest

from panorama_demo.video_algorithm_contract import VideoAlgorithmContractError
from panorama_demo.video_v2_pair_planner import build_v2_candidate_pair_plans, candidate_component_audit


def test_c1_to_c8_pair_plans_are_cumulative_and_real_source_adjacent():
    c1 = build_v2_candidate_pair_plans("C1_constrained_owner", (10, 20, 30), (0, 2))
    c8 = build_v2_candidate_pair_plans("C8_multilabel_window", (10, 20, 30), (0, 2))

    assert [(item.left_frame_id, item.right_frame_id) for item in c8] == [(10, 20), (20, 30)]
    assert c1[0].flow_backend == "none"
    assert c8[0].flow_backend == "raft_small"
    assert c8[0].use_raft_backward is False
    assert c8[1].use_raft_backward is True
    assert c8[1].use_open3d is True
    assert c8[0].use_depth_mesh and c8[0].object_lock_required
    assert c8[0].seam_mode == "multilabel" and c8[0].blend_mode == "safe_multiband"
    assert candidate_component_audit("C8_multilabel_window")["c7_global_photometric"] is True


def test_v2_pair_planner_rejects_nonadjacent_or_unknown_candidate_inputs():
    with pytest.raises(VideoAlgorithmContractError, match="unsupported"):
        build_v2_candidate_pair_plans("C9", (1, 2), (0,))
    with pytest.raises(VideoAlgorithmContractError, match="chronological"):
        build_v2_candidate_pair_plans("C1_constrained_owner", (2, 1), (0,))
    with pytest.raises(VideoAlgorithmContractError, match="risk level"):
        build_v2_candidate_pair_plans("C1_constrained_owner", (1, 2), (4,))
