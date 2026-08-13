from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import panorama_demo.video_s12_metrics as metrics
from panorama_demo.rgbd_odometry import PoseEdge


def test_duplicate_source_selection_preserves_automatic_anchor() -> None:
    selected = metrics.select_unique_sources(
        [10, 11, 12],
        [100.0, 100.25, 108.0],
        anchor_frame_ids=[11, 12],
        image_quality_by_frame={10: 0.9, 11: 0.5, 12: 0.8},
    )
    assert [item.frame_id for item in selected] == [11, 12]
    assert selected[0].center_x == 100.25


def test_source_selection_never_averages_duplicate_centres() -> None:
    selected = metrics.select_unique_sources(
        [1, 2, 3],
        [10.0, 10.2, 20.0],
        anchor_frame_ids=[1, 3],
    )
    assert [item.center_x for item in selected] == [10.0, 20.0]


def test_open3d_audit_compares_edge_to_immutable_orb_pose(monkeypatch) -> None:
    frames = [SimpleNamespace(frame_id=1), SimpleNamespace(frame_id=2)]
    poses = [np.eye(4), np.eye(4)]
    poses[1] = poses[1].copy()
    poses[1][0, 3] = 10.0

    def fake_estimate(*_args, initial_source_to_reference, **_kwargs):
        np.testing.assert_allclose(initial_source_to_reference, poses[1])
        return PoseEdge(
            reference_node_id=1,
            source_node_id=2,
            source_to_reference=poses[1],
            converged=True,
            fitness=1.0,
            rmse_mm=0.1,
            information=np.eye(6),
            reference_valid_depth_ratio=1.0,
            source_valid_depth_ratio=1.0,
            backend="open3d_tensor_cuda_rgbd",
        )

    monkeypatch.setattr(metrics, "estimate_pair_rgbd_odometry", fake_estimate)
    rows = metrics.audit_open3d_source_edges(
        frames, poses, SimpleNamespace(), backend=object()
    )
    assert rows[0]["quality_pass"] is True
    assert rows[0]["backend_pass"] is True
    assert rows[0]["pose_replacement_allowed"] is False
    np.testing.assert_array_equal(poses[1][:3, 3], [10.0, 0.0, 0.0])


def test_open3d_residual_failure_is_fail_closed(monkeypatch) -> None:
    frames = [SimpleNamespace(frame_id=1), SimpleNamespace(frame_id=2)]
    poses = [np.eye(4), np.eye(4)]
    poses[1] = poses[1].copy()
    poses[1][0, 3] = 100.0

    def fake_estimate(*_args, **_kwargs):
        return PoseEdge(
            reference_node_id=1,
            source_node_id=2,
            source_to_reference=np.eye(4),
            converged=True,
            fitness=1.0,
            rmse_mm=0.1,
            information=np.eye(6),
            reference_valid_depth_ratio=1.0,
            source_valid_depth_ratio=1.0,
        )

    monkeypatch.setattr(metrics, "estimate_pair_rgbd_odometry", fake_estimate)
    rows = metrics.audit_open3d_source_edges(
        frames, poses, SimpleNamespace(), backend=object()
    )
    assert rows[0]["quality_pass"] is False
    assert rows[0]["orb_translation_residual_mm"] == pytest.approx(100.0)


def test_open3d_cpu_backend_is_not_accepted_as_formal_stage_a(monkeypatch) -> None:
    frames = [SimpleNamespace(frame_id=1), SimpleNamespace(frame_id=2)]
    poses = [np.eye(4), np.eye(4)]

    def fake_estimate(*_args, **_kwargs):
        return PoseEdge(
            reference_node_id=1,
            source_node_id=2,
            source_to_reference=np.eye(4),
            converged=True,
            fitness=1.0,
            rmse_mm=0.1,
            information=np.eye(6),
            reference_valid_depth_ratio=1.0,
            source_valid_depth_ratio=1.0,
            backend="open3d_rgbd",
        )

    monkeypatch.setattr(metrics, "estimate_pair_rgbd_odometry", fake_estimate)
    row = metrics.audit_open3d_source_edges(
        frames, poses, SimpleNamespace(), backend=object()
    )[0]
    assert row["backend_pass"] is False
    assert row["quality_pass"] is False
