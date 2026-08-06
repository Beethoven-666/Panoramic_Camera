from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.video_algorithm_contract import (
    PairPlan,
    PreparedVideoAlgorithm,
    VideoAlgorithmContractError,
    VideoAlgorithmResult,
    require_v2_algorithm,
)


def _pair() -> PairPlan:
    return PairPlan(1, 2, 1, "raft_small", True, True, True, True, "multilabel", "safe_multiband")


def test_prepared_contract_requires_every_adjacent_real_pair_and_pose():
    prepared = PreparedVideoAlgorithm(
        source_frame_ids=(1, 2),
        camera_to_world=(np.eye(4), np.eye(4)),
        pair_plans=(_pair(),),
        context_audit={"real_sources_only": True},
    )
    assert prepared.pair_plans[0].left_frame_id == 1
    with pytest.raises(VideoAlgorithmContractError, match="exactly cover"):
        PreparedVideoAlgorithm(
            source_frame_ids=(1, 2), camera_to_world=(np.eye(4), np.eye(4)),
            pair_plans=(PairPlan(2, 3, 0, "none", False, False, True, False, "hard_owner", "none"),),
            context_audit={},
        )


def test_result_contract_rejects_unknown_owner_and_interpolated_pose_claim():
    panorama = np.zeros((3, 4, 3), dtype=np.uint8)
    with pytest.raises(VideoAlgorithmContractError, match="declared real"):
        VideoAlgorithmResult(panorama, np.full((3, 4), 99, dtype=np.int32), (7,), {})
    with pytest.raises(VideoAlgorithmContractError, match="interpolated"):
        VideoAlgorithmResult(panorama, np.full((3, 4), 7, dtype=np.int32), (7,), {"interpolated_pose_count": 1})


def test_v2_algorithm_protocol_requires_both_operations():
    class OnlyPrepare:
        def prepare(self, **_: object) -> object:
            return object()

    with pytest.raises(VideoAlgorithmContractError, match=r"prepare\(\) and render"):
        require_v2_algorithm(OnlyPrepare())
