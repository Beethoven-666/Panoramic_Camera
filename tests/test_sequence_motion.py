from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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
        self.optimized_node_ids: tuple[int, ...] = ()

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

    def optimize_pose_graph(
        self, *, node_ids, initial_camera_to_world, edges, config
    ):
        del edges, config
        self.optimized_node_ids = tuple(int(value) for value in node_ids)
        return tuple(np.asarray(pose).copy() for pose in initial_camera_to_world)


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


def test_formal_pushbroom_receives_exact_optimized_se3_without_depth_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, poses = _synthetic_session(tmp_path)
    backend = _MetricTranslationBackend(poses)
    received: dict[str, object] = {}

    def legacy_projection_must_not_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("formal RGB pushbroom reached legacy depth projection")

    def fake_pushbroom(frames, optimized_poses, calibration, **kwargs):
        received["frame_ids"] = [frame.frame_id for frame in frames]
        received["poses"] = [np.asarray(pose).copy() for pose in optimized_poses]
        received["calibration"] = calibration
        received["kwargs"] = dict(kwargs)
        return SimpleNamespace(
            panorama=np.full((100, 300, 3), 127, dtype=np.uint8),
            metadata={
                "backend": "calibrated_rgb_pushbroom",
                "pixel_source": "calibrated_rgb_only",
                "depth_used_for_output_pixels": False,
                "point_cloud_constructed": False,
                "tsdf_constructed": False,
                "reference_plane_fitted": False,
                "layout": {},
                "rgb_motion_scale": {},
                "residual_alignment": {
                    "backend": "se3_epipolar_hierarchical_rgb",
                    "selected_model": "background_se2",
                    "preview_remap_count": len(frames),
                    "full_resolution_output_remap_count": len(frames),
                    "per_source_parameters": [
                        {
                            "source_index": index,
                            "translation_x_pixels": 0.25 * index,
                            "translation_y_pixels": 0.0,
                            "roll_degrees": 0.0,
                            "centre_x_pixels": 0.0,
                            "centre_y_pixels": 0.0,
                            "identity": index == 0,
                        }
                        for index, _frame in enumerate(frames)
                    ],
                    "held_out_metrics_before": {},
                    "held_out_metrics_after": {},
                    "component_audit": {},
                    "topology_audit": {"accepted": True},
                    "working_set_audit": {},
                },
                "quality_metrics": {"quality_pass": True},
            },
        )

    monkeypatch.setattr(sequence, "_full_projection_frames", legacy_projection_must_not_run)
    monkeypatch.setattr(sequence, "render_calibrated_rgb_pushbroom", fake_pushbroom)
    args = sequence._parser().parse_args(
        [str(session.root), "--output", str(tmp_path / "output")]
    )

    report = sequence.run(args, odometry_backend=backend)

    assert received["frame_ids"] == list(backend.optimized_node_ids)
    assert received["calibration"] == session.calibration
    for frame_id, optimized in zip(
        received["frame_ids"], received["poses"], strict=True
    ):
        np.testing.assert_allclose(optimized, poses[frame_id])
    kwargs = received["kwargs"]
    assert kwargs["quality_gate"] is True
    assert kwargs["multiband_levels"] == 3
    assert len(kwargs["rgb_motions"]) == len(backend.optimized_node_ids) - 1
    assert report["render_strategy"] == "calibrated_rgb_pushbroom"
    assert report["render"]["depth_used_for_output_pixels"] is False
    render_transforms = json.loads(
        (tmp_path / "output" / "render_transforms.json").read_text(encoding="utf-8")
    )
    alignment = render_transforms["residual_alignment"]
    assert alignment["selected_model"] == "background_se2"
    assert alignment["per_source_parameters"][1]["translation_x_pixels"] == 0.25
    delivery = json.loads(
        (tmp_path / "output" / "delivery.json").read_text(encoding="utf-8")
    )
    assert delivery["alignment_model"] == "background_se2"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"pose_backend": "unistitch"}, "pose_backend"),
        ({"sequence_blend_mode": "feather"}, "calibrated_rgb_pushbroom"),
        (
            {"dense_fusion_backend": "tsdf_plane_dense_rgbd"},
            "TSDF and RGB-D projection",
        ),
        (
            {"rgbd_projection": {"mode": "orthographic_side_scan"}},
            "TSDF and RGB-D projection",
        ),
        (
            {"calibrated_rgb_pushbroom": {"mode": "central_strip"}},
            "Formal renderer mode",
        ),
        (
            {"scan_seam": {"backend": "dp"}},
            "rgb_monotonic_hard_owner_graphcut",
        ),
    ],
)
def test_formal_backend_configuration_rejects_non_pushbroom_paths(
    override: dict[str, object], message: str
) -> None:
    config: dict[str, object] = {
        "pose_backend": "open3d_rgbd",
        "sequence_blend_mode": "calibrated_rgb_pushbroom",
        "calibrated_rgb_pushbroom": {
            "mode": "calibrated_rgb_pushbroom",
        },
        "scan_seam": {"backend": "rgb_monotonic_hard_owner_graphcut"},
    }
    config.update(override)

    with pytest.raises(ValueError, match=message):
        sequence._validate_backend_config(config)
