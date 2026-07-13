from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import panorama_demo.stitch_sequence as sequence
from panorama_demo.rgbd_odometry import RGBDOdometryConfig
from panorama_demo.session import load_rgbd_session
from panorama_demo.synthetic import generate_sequence


class _MetricTranslationBackend:
    name = "test_metric_rgbd"

    def __init__(self, poses: dict[int, np.ndarray]) -> None:
        self.poses = poses
        self.pairs: list[tuple[int, int]] = []

    def estimate_pair(self, *, reference, source, intrinsics, config):
        del intrinsics, config
        reference_id = int(reference.frame_id)
        source_id = int(source.frame_id)
        self.pairs.append((reference_id, source_id))
        return {
            "source_to_reference": (
                np.linalg.inv(self.poses[reference_id]) @ self.poses[source_id]
            ),
            "converged": True,
            "fitness": 0.99,
            "rmse_mm": 0.25,
            "information": np.eye(6, dtype=np.float64) * 100.0,
            "backend": self.name,
        }


def _synthetic_session(tmp_path: Path):
    root = generate_sequence(
        tmp_path / "session",
        frame_count=5,
        frame_width=160,
        frame_height=100,
        step=30,
        seed=31,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    poses = {
        int(row["frame_id"]): np.asarray(
            row["matrix_row_major"], dtype=np.float64
        ).reshape(4, 4)
        for row in manifest["known_trajectory"]["poses"]
    }
    return load_rgbd_session(root), poses


def test_formal_sequence_exposes_no_legacy_2d_motion_or_interpolation_api() -> None:
    for name in (
        "regularize_pair_homography",
        "interpolate_translation_transforms",
        "interpolate_motion_guided_transforms",
        "_parse_protected_regions",
    ):
        assert not hasattr(sequence, name)


def test_pose_edge_estimation_connects_only_real_rgbd_pose_nodes(
    tmp_path: Path,
) -> None:
    session, poses = _synthetic_session(tmp_path)
    selected = [session.frames[index] for index in (0, 2, 4)]
    backend = _MetricTranslationBackend(poses)

    edges, optional_failures = sequence._estimate_pose_edges(
        selected,
        session.calibration,
        RGBDOdometryConfig(working_width=160),
        backend=backend,
        nonadjacent_gap=2,
    )

    assert optional_failures == []
    assert backend.pairs == [(0, 2), (2, 4), (0, 4)]
    assert [
        (edge.reference_node_id, edge.source_node_id) for edge in edges
    ] == backend.pairs
    assert all(edge.source_to_reference.shape == (4, 4) for edge in edges)
    np.testing.assert_allclose(
        edges[0].source_to_reference,
        np.linalg.inv(poses[0]) @ poses[2],
    )


def test_full_resolution_projection_sources_keep_exact_optimized_se3(
    tmp_path: Path,
) -> None:
    session, poses = _synthetic_session(tmp_path)
    frames = [session.frames[index] for index in (0, 4)]
    exact_poses = [poses[frame.frame_id] for frame in frames]

    projected = sequence._full_projection_frames(frames, exact_poses)

    assert [frame.frame_id for frame in projected] == [0, 4]
    for source, expected in zip(projected, exact_poses, strict=True):
        assert source.camera_to_world.shape == (4, 4)
        np.testing.assert_allclose(source.camera_to_world, expected)
        assert source.depth_mm.shape == (
            session.calibration.height,
            session.calibration.width,
        )
        assert np.count_nonzero(source.depth_mm) > 0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"pose_backend": "unistitch"}, "pose_backend"),
        ({"sequence_blend_mode": "feather"}, "scan_seam"),
        (
            {"rgbd_projection": {"mode": "homography"}},
            "orthographic_side_scan",
        ),
        (
            {
                "rgbd_projection": {
                    "mode": "orthographic_side_scan",
                    "reject_depth_discontinuity": False,
                }
            },
            "depth-discontinuity",
        ),
        ({"scan_seam": {"backend": "dp"}}, "graphcut_depth_constrained"),
    ],
)
def test_formal_backend_configuration_rejects_legacy_paths(
    override: dict[str, object], message: str
) -> None:
    config: dict[str, object] = {
        "pose_backend": "open3d_rgbd",
        "sequence_blend_mode": "scan_seam",
        "rgbd_projection": {
            "mode": "orthographic_side_scan",
            "reject_depth_discontinuity": True,
        },
        "scan_seam": {"backend": "graphcut_depth_constrained"},
    }
    config.update(override)

    with pytest.raises(ValueError, match=message):
        sequence._validate_backend_config(config)
