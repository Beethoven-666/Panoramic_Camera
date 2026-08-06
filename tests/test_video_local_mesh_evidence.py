from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.geometry_assisted_local_warp import LocalMeshWarpConfig
from panorama_demo.video_local_mesh_evidence import (
    LocalMeshEvidenceConfig,
    assess_candidate_local_mesh_evidence,
)
from panorama_demo.video_visual_renderer import VideoVisualSource


def _source(frame_id: int, *, depth: np.ndarray | None = None) -> VideoVisualSource:
    image = np.zeros((128, 160, 4), dtype=np.uint8)
    image[..., :3] = 90
    image[..., 3] = 255
    return VideoVisualSource(frame_id=frame_id, bgra=image, depth_mm=depth)


def _translation_flow(first: VideoVisualSource, second: VideoVisualSource) -> np.ndarray:
    flow = np.zeros((*np.asarray(first.bgra).shape[:2], 2), dtype=np.float32)
    flow[..., 0] = 0.48 if first.frame_id < second.frame_id else -0.48
    flow[..., 1] = -0.28 if first.frame_id < second.frame_id else 0.28
    return flow


def _config(*, depth: bool = False) -> LocalMeshEvidenceConfig:
    return LocalMeshEvidenceConfig(
        flow_backend="raft",
        require_depth_safety=depth,
        mesh=LocalMeshWarpConfig(
            minimum_correspondences=48,
            held_out_seed=19,
            maximum_held_out_error_pixels=1.0,
        ),
    )


def test_candidate_mesh_evidence_accepts_bounded_bidirectional_real_source_flow() -> None:
    result = assess_candidate_local_mesh_evidence(
        _source(10), _source(11), config=_config(), flow_estimator=_translation_flow
    )

    assert result.accepted
    assert result.fit is not None and result.fit.warp is not None
    assert result.audit.mesh is not None and result.audit.mesh.accepted
    assert result.audit.first_frame_id == 10
    assert result.audit.second_frame_id == 11
    assert result.audit.flow_backend == "raft"
    assert result.audit.as_dict()["creates_colour"] is False
    assert result.audit.as_dict()["creates_owner"] is False
    assert result.audit.as_dict()["creates_pose"] is False
    x, y = result.fit.warp.inverse_virtual_coordinates(80.0, 64.0)
    assert x == pytest.approx(79.52, abs=0.35)
    assert y == pytest.approx(64.28, abs=0.35)


def test_candidate_mesh_evidence_rejects_forward_backward_inconsistent_flow() -> None:
    def inconsistent(first: VideoVisualSource, second: VideoVisualSource) -> np.ndarray:
        flow = _translation_flow(first, second)
        if first.frame_id > second.frame_id:
            flow[..., 0] = 2.0
        return flow

    result = assess_candidate_local_mesh_evidence(
        _source(10), _source(11), config=_config(), flow_estimator=inconsistent
    )

    assert not result.accepted
    assert result.fit is None
    assert result.audit.rejection_reason == "no_forward_backward_reliable_flow"


def test_candidate_mesh_evidence_depth_safety_rejects_depth_edge_and_mismatch() -> None:
    first_depth = np.full((128, 160), 1400.0, dtype=np.float32)
    second_depth = np.full((128, 160), 2000.0, dtype=np.float32)
    result = assess_candidate_local_mesh_evidence(
        _source(10, depth=first_depth),
        _source(11, depth=second_depth),
        config=_config(depth=True),
        flow_estimator=_translation_flow,
    )

    assert not result.accepted
    assert result.fit is not None
    assert result.audit.depth_safety_required
    assert result.audit.depth_safe_pixel_count == 0
    assert result.audit.mesh is not None
    assert result.audit.mesh.reason == "insufficient_same_layer_correspondences"


def test_candidate_mesh_evidence_requires_explicit_raft_and_distinct_real_sources() -> None:
    with pytest.raises(ValueError, match="explicit verified flow estimator"):
        assess_candidate_local_mesh_evidence(_source(10), _source(11), config=_config())
    with pytest.raises(ValueError, match="distinct real source frame ids"):
        assess_candidate_local_mesh_evidence(
            _source(10), _source(10), config=_config(), flow_estimator=_translation_flow
        )


def test_candidate_mesh_evidence_never_uses_colour_as_output_even_with_dis() -> None:
    # DIS is an allowed evidence backend, but the result remains a mesh audit;
    # it contains no composed BGRA image or ownership map.
    result = assess_candidate_local_mesh_evidence(
        _source(20), _source(21),
        config=LocalMeshEvidenceConfig(
            flow_backend="dis",
            mesh=LocalMeshWarpConfig(minimum_correspondences=48, held_out_seed=3),
        ),
    )

    assert not hasattr(result, "bgra")
    assert not hasattr(result, "owner_frame_id")
    assert result.audit.as_dict()["real_adjacent_sources_only"] is True
