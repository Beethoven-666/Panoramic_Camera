from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from panorama_demo.central_strip import render_central_strip_diagnostic
import panorama_demo.stitch_sequence as sequence
from panorama_demo.synthetic import generate_sequence


class _KnownTrajectoryRGBDBackend:
    """Deterministic RGB-D backend driven by the synthetic manifest's SE(3)."""

    name = "synthetic_known_rgbd"

    def __init__(
        self,
        session: Path,
        *,
        fitness: float = 0.99,
        rmse_mm: float = 0.5,
    ) -> None:
        manifest = json.loads(
            (session / "manifest.json").read_text(encoding="utf-8")
        )
        trajectory = manifest["known_trajectory"]
        assert trajectory["transform"] == "camera_to_world"
        assert trajectory["translation_unit"] == "millimetres"
        self.poses = {
            int(row["frame_id"]): np.asarray(
                row["matrix_row_major"], dtype=np.float64
            ).reshape(4, 4)
            for row in trajectory["poses"]
        }
        self.fitness = float(fitness)
        self.rmse_mm = float(rmse_mm)
        self.estimated_pairs: list[tuple[int, int]] = []
        self.optimized_node_ids: tuple[int, ...] = ()

    def estimate_pair(self, *, reference, source, intrinsics, config):
        del intrinsics, config
        reference_id = int(reference.frame_id)
        source_id = int(source.frame_id)
        self.estimated_pairs.append((reference_id, source_id))
        source_to_reference = (
            np.linalg.inv(self.poses[reference_id]) @ self.poses[source_id]
        )
        return {
            "source_to_reference": source_to_reference,
            "converged": True,
            "fitness": self.fitness,
            "rmse_mm": self.rmse_mm,
            "information": np.eye(6, dtype=np.float64) * 100.0,
            "backend": self.name,
        }

    def optimize_pose_graph(
        self, *, node_ids, initial_camera_to_world, edges, config
    ):
        del edges, config
        self.optimized_node_ids = tuple(int(value) for value in node_ids)
        # The synthetic measurements are exact, so the propagated initial poses
        # are already the optimum. No image feature or 2-D transform is involved.
        return tuple(np.asarray(pose).copy() for pose in initial_camera_to_world)


def _make_session(tmp_path: Path, *, seed: int) -> Path:
    return generate_sequence(
        tmp_path / "session",
        frame_count=7,
        frame_width=320,
        frame_height=200,
        # The formal owner search requires a 32 px interior overlap.  A 64 px
        # central strip therefore needs a 32 px synthetic camera step rather
        # than the old 60 px almost-touching strips.
        step=32,
        seed=seed,
    )


def test_zero_parameter_rgbd_sequence_publishes_one_complete_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _make_session(tmp_path, seed=19)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    import panorama_demo.dense_fusion as dense_fusion

    def fake_export(frames, poses, intrinsics, *, config):
        assert len(frames) == len(poses) == 7
        assert intrinsics.width == 320
        assert config["enabled"] is True
        return b"glTF-display-only-test-mesh", {
            "backend": "fake_tsdf_display_only",
            "frame_count": len(frames),
            "vertex_count": 3,
            "triangle_count": 1,
            "glb_byte_count": 28,
            "translation_unit": "mm",
            "display_only": True,
            "participates_in_panorama": False,
        }

    monkeypatch.setattr(dense_fusion, "export_tsdf_mesh", fake_export)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    report = sequence.run(args, odometry_backend=backend)

    panorama = cv2.imread(str(output / "panorama.jpg"), cv2.IMREAD_COLOR)
    assert panorama is not None
    assert panorama.shape[1] > 320
    assert panorama.shape[0] >= 180
    assert (output / "delivery.json").is_file()
    assert not (output / "failure.json").exists()
    assert report["schema"] == "gemini305-calibrated-rgb-pushbroom/v9"
    assert report["layout_selection"]["mode"] == "adaptive_rgbd_pose_nodes"
    assert report["render_strategy"] == "calibrated_rgb_pushbroom"
    assert report["render"]["backend"] == "calibrated_rgb_pushbroom"
    assert report["render"]["pixel_source"] == "calibrated_rgb_source_samples"
    assert report["render"]["depth_used_for_output_pixels"] is False
    assert report["render"]["local_geometry_scope"] == "adjacent_seam_corridors_only"
    geometry = report["render"]["geometry_assisted_seam"]
    assert geometry["depth_used_for_output_pixels"] is False
    assert geometry["scope"] == "adjacent_seam_corridors_only"
    assert len(geometry["pairs"]) == len(backend.optimized_node_ids) - 1
    assert report["render"]["point_cloud_constructed"] is False
    assert report["render"]["tsdf_constructed"] is False
    assert report["render"]["reference_plane_fitted"] is False
    assert report["render"]["quality_metrics"]["quality_pass"] is True
    assert report["strict_failure_reasons"] == []
    assert report["pose_quality"]["quality_pass"] is True
    assert report["pose_graph"]["connected"] is True
    assert report["render"]["selection"]["interpolated_pose_count"] == 0
    assert len(backend.optimized_node_ids) >= 2
    assert backend.estimated_pairs
    assert report["render"]["selection"]["mode"] == (
        "calibrated_rgb_pushbroom_all_real_pose_nodes"
    )
    assert report["render"]["frame_ids"] == list(backend.optimized_node_ids)
    assert report["render"]["source_count"] == len(backend.optimized_node_ids)
    metrics = report["render"]["quality_metrics"]
    assert metrics["source_remap_count"] == len(backend.optimized_node_ids)
    assert 2 <= metrics["maximum_resident_strips"] <= 5
    assert all(
        pair["blend_zone_risk_pixel_count"] == 0
        and pair["geometry_blend_zone_pixel_count"] == 0
        and 2 <= pair["blend_width_pixels"] <= 8
        for pair in report["render"]["pairs"]
    )

    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    assert transforms["pose_convention"].startswith("camera_to_world")
    assert transforms["translation_unit"] == "mm"
    assert all(
        np.asarray(node["camera_to_world"]).shape == (4, 4)
        for node in transforms["nodes"]
    )
    render_transforms = json.loads(
        (output / "render_transforms.json").read_text(encoding="utf-8")
    )
    assert render_transforms["schema"] == "calibrated-rgb-pushbroom/v7"
    assert render_transforms["pixel_source"] == "calibrated_rgb_source_samples"
    assert render_transforms["depth_used_for_output_pixels"] is False
    assert [source["frame_id"] for source in render_transforms["sources"]] == list(
        backend.optimized_node_ids
    )
    assert all("aligned_depth_path" not in source for source in render_transforms["sources"])
    compact_alignment = render_transforms["residual_alignment"]
    full_alignment = report["render"]["residual_alignment"]
    assert compact_alignment["backend"] == full_alignment["backend"]
    assert compact_alignment["selected_model"] == full_alignment["selected_model"]
    assert compact_alignment["configuration"]["held_out_fraction"] == 0.20
    assert compact_alignment["topology_audit"]["accepted"] is True
    assert compact_alignment["preview_remap_count"] == len(backend.optimized_node_ids)
    assert compact_alignment["full_resolution_output_remap_count"] == len(
        backend.optimized_node_ids
    )
    assert [
        parameter["frame_id"]
        for parameter in compact_alignment["per_source_parameters"]
    ] == list(backend.optimized_node_ids)
    assert "evidence" not in compact_alignment
    compact_geometry = render_transforms["geometry_assisted_seam"]
    assert compact_geometry["backend"] == "rgbd_bidirectional_visibility_local_inverse_mesh"
    assert compact_geometry["scope"] == "adjacent_seam_corridors_only"
    assert compact_geometry["depth_used_for_output_pixels"] is False
    assert len(compact_geometry["pairs"]) == len(backend.optimized_node_ids) - 1
    assert all(
        "aligned_depth_path" not in pair
        and "depth_mm" not in pair
        and "source_map_x" not in pair
        for pair in compact_geometry["pairs"]
    )
    delivery = json.loads((output / "delivery.json").read_text(encoding="utf-8"))
    assert delivery["quality_pass"] is True
    assert delivery["strict_quality_pass"] is True
    assert delivery["delivery_state"] == "published"
    assert delivery["quality_grade"] in {"A", "B"}
    assert delivery["manual_review_required"] is False
    assert sum(delivery["handoff_fallback_summary"].values()) == len(
        backend.optimized_node_ids
    ) - 1
    assert delivery["pose_backend"] == "open3d_rgbd"
    assert delivery["projection"] == "calibrated_rgb_pushbroom"
    assert delivery["schema"] == "gemini305-panorama-delivery/v9"
    assert delivery["pixel_source"] == "calibrated_rgb_source_samples"
    assert delivery["depth_used_for_output_pixels"] is False
    assert delivery["geometry_assistance_backend"] == compact_geometry["backend"]
    geometry_gate = delivery["geometry_assistance_gate"]
    assert {
        key: geometry_gate[key]
        for key in (
            "minimum_active_mesh_cells",
            "maximum_straight_line_deviation_pixels",
            "rgb_flow_application_policy",
            "rgb_flow_fit_support_policy",
            "actual_rgb_line_observation_policy",
        )
    } == {
        "minimum_active_mesh_cells": 4,
        "maximum_straight_line_deviation_pixels": 1.0,
        "rgb_flow_application_policy": (
            "accepted_bidirectional_rgb_flow_and_epipolar_support_including_held_out"
        ),
        "rgb_flow_fit_support_policy": (
            "training_only_accepted_bidirectional_rgb_flow_and_epipolar_support"
        ),
        "actual_rgb_line_observation_policy": (
            "observed_hough_solver_line_veto_or_not_observed_non_veto"
        ),
    }
    assert geometry_gate["local_apap_flow"]["enabled"] is False
    assert delivery["alignment_backend"] == compact_alignment["backend"]
    assert delivery["alignment_model"] == compact_alignment["selected_model"]
    assert delivery["seam_backend"] == "rgb_monotonic_hard_owner_graphcut"
    assert delivery["blend_backend"] == "safe_wall_local_multiband_narrow_owner_boundary"
    visualization = report["tsdf_visualization"]
    assert visualization["display_only"] is True
    assert visualization["participates_in_panorama"] is False
    assert (output / "tsdf_mesh.glb").read_bytes() == b"glTF-display-only-test-mesh"
    viewer = (output / "tsdf_mesh_viewer.html").read_text(encoding="utf-8")
    assert 'src="tsdf_mesh.glb"' in viewer
    assert "model-viewer" in viewer
    assert delivery["tsdf_visualization"]["display_only"] is True
    for legacy_artifact in (
        "foreground_mask.png",
        "background_exclusion_mask.png",
        "tsdf_foreground_mask.png",
    ):
        assert not (output / legacy_artifact).exists()
    crop = report["render"]["crop"]
    assert panorama.shape[:2] == (crop["height"], crop["width"])
    assert metrics["crop_height_ratio"] >= 0.85
    assert metrics["crop_width_ratio"] >= 0.95


def test_tsdf_export_failure_is_nonblocking_for_valid_rgb_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The display-only mesh cannot invalidate an already-safe RGB delivery."""

    session = _make_session(tmp_path, seed=19)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    import panorama_demo.dense_fusion as dense_fusion

    def fail_export(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("forced display-only TSDF export failure")

    monkeypatch.setattr(dense_fusion, "export_tsdf_mesh", fail_export)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    report = sequence.run(args, odometry_backend=backend)

    expected_visualization = {
        "status": "unavailable",
        "display_only": True,
        "participates_in_panorama": False,
        "reason": "tsdf_export_failed_nonblocking",
        "error_type": "RuntimeError",
    }
    delivery = json.loads((output / "delivery.json").read_text(encoding="utf-8"))
    assert (output / "panorama.jpg").is_file()
    assert not (output / "failure.json").exists()
    assert delivery["delivery_state"] == "published"
    assert delivery["strict_quality_pass"] is True
    assert delivery["tsdf_visualization"] == expected_visualization
    assert report["tsdf_visualization"] == expected_visualization
    assert not (output / "tsdf_mesh.glb").exists()
    assert not (output / "tsdf_mesh_viewer.html").exists()


def test_public_handoff_policy_enables_local_apap_without_a_duplicate_renderer_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public policy is the one formal APAP/flow opt-in surface."""

    session = _make_session(tmp_path, seed=31)
    output = tmp_path / "output"
    config = tmp_path / "policy.yaml"
    config.write_text(
        "stitch:\n"
        "  handoff_fallback_policy:\n"
        "    publish_degraded: true\n"
        "    local_apap_flow_enabled: true\n"
        "    manual_review_for_grade_c: true\n",
        encoding="utf-8",
    )
    backend = _KnownTrajectoryRGBDBackend(session)
    import panorama_demo.dense_fusion as dense_fusion

    monkeypatch.setattr(
        dense_fusion,
        "export_tsdf_mesh",
        lambda *args, **kwargs: (b"glTF-policy-test", {
            "backend": "fake_tsdf_display_only",
            "frame_count": 7,
            "vertex_count": 3,
            "triangle_count": 1,
            "glb_byte_count": 16,
            "translation_unit": "mm",
            "display_only": True,
            "participates_in_panorama": False,
        }),
    )
    args = sequence._parser().parse_args(
        [str(session), "--output", str(output), "--config", str(config)]
    )

    report = sequence.run(args, odometry_backend=backend)

    delivery = json.loads((output / "delivery.json").read_text(encoding="utf-8"))
    assert (output / "panorama.jpg").is_file()
    assert report["render"]["geometry_assisted_seam"]["local_apap_flow"]["enabled"] is True
    assert delivery["geometry_assistance_gate"]["local_apap_flow"]["enabled"] is True


def test_importing_formal_sequence_does_not_load_legacy_model_stack() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import panorama_demo.stitch_sequence; "
                "blocked=('torch', 'kornia', 'lightglue', "
                "'panorama_demo.unistitch_adapter', "
                "'panorama_demo.stitch_common', "
                "'panorama_demo.central_strip', "
                "'panorama_demo.dense_fusion', "
                "'panorama_demo.rgbd_projection'); "
                "loaded=[name for name in sys.modules "
                "if any(name == item or name.startswith(item + '.') "
                "for item in blocked)]; "
                "assert not loaded, loaded"
            ),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr


@pytest.mark.parametrize("activation", ["cli", "config"])
def test_diagnostic_mode_bypasses_quality_thresholds_but_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    activation: str,
) -> None:
    session = _make_session(tmp_path, seed=23)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "capture_mode": "diagnostic_unrestricted_auto_exposure",
            "diagnostic_only": True,
            "formal_stitch_allowed": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = tmp_path / "output"
    # Structurally valid but formally poor odometry must remain usable only in
    # diagnostic mode; graph connectivity and finite SE(3) are still enforced.
    backend = _KnownTrajectoryRGBDBackend(session, fitness=0.0, rmse_mm=500.0)
    original_capture_quality = sequence.assess_capture_quality
    original_render = sequence.render_calibrated_rgb_pushbroom

    def failing_capture_quality(*args, **kwargs):
        result = original_capture_quality(*args, **kwargs)
        result["quality_pass"] = False
        result["failure_reasons"] = ["forced test input failure"]
        return result

    def failing_render(*args, **kwargs):
        assert kwargs["quality_gate"] is False
        result = original_render(*args, **kwargs)
        metadata = dict(result.metadata)
        quality_metrics = dict(metadata["quality_metrics"])
        quality_metrics["quality_pass"] = False
        metadata["quality_metrics"] = quality_metrics
        return SimpleNamespace(panorama=result.panorama, metadata=metadata)

    monkeypatch.setattr(sequence, "assess_capture_quality", failing_capture_quality)
    monkeypatch.setattr(
        sequence, "render_calibrated_rgb_pushbroom", failing_render
    )
    arguments = [str(session), "--output", str(output)]
    if activation == "cli":
        arguments.append("--diagnostic-force")
    else:
        arguments.extend(
            [
                "--config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "configs"
                    / "capture_unrestricted_auto_exposure.yaml"
                ),
            ]
        )
    args = sequence._parser().parse_args(arguments)

    report = sequence.run(args, odometry_backend=backend)

    panorama = cv2.imread(
        str(output / "diagnostic_panorama.jpg"), cv2.IMREAD_COLOR
    )
    assert panorama is not None
    assert sorted(path.name for path in output.iterdir()) == [
        "diagnostic_panorama.jpg",
        "diagnostic_report.json",
    ]
    assert report["diagnostic_only"] is True
    assert report["deliverable_published"] is False
    assert report["input_capture"]["diagnostic_only"] is True
    assert report["input_quality"]["quality_pass"] is False
    assert report["pose_quality"]["quality_pass"] is False
    assert report["render"]["quality_metrics"]["quality_pass"] is False
    assert report["render"]["backend"] == "calibrated_rgb_pushbroom"
    assert report["render"]["depth_used_for_output_pixels"] is False
    assert report["diagnostic_overrides"] == {
        "input_quality_thresholds_bypassed": True,
        "odometry_quality_thresholds_bypassed": True,
        "pose_quality_thresholds_bypassed": True,
        "final_image_quality_thresholds_bypassed": True,
        "calibration_aligned_depth_finite_se3_graph_connectivity_"
        "projection_topology_memory_atomic_safety_required": True,
    }


def test_central_strip_callback_is_diagnostic_only_and_skips_formal_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _make_session(tmp_path, seed=29)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    received: dict[str, object] = {}

    def fake_renderer(**kwargs: object) -> SimpleNamespace:
        received.update(kwargs)
        return SimpleNamespace(
            panorama=np.full((200, 480, 3), 127, dtype=np.uint8),
            metadata={"strip_quality_pass": False, "renderer": "fake-central-strip"},
        )

    def formal_quality_must_not_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("central-strip callback reached formal delivery quality gate")

    monkeypatch.setattr(sequence, "_ensure_publishable_quality", formal_quality_must_not_run)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    report = sequence.run(
        args, odometry_backend=backend, diagnostic_renderer=fake_renderer
    )

    assert set(received) == {
        "plane_frames",
        "plane_poses",
        "render_frames",
        "render_poses",
        "calibration",
        "config",
        "sharpness_scores",
    }
    assert len(received["plane_frames"]) == len(backend.optimized_node_ids)
    assert len(received["render_frames"]) == len(backend.optimized_node_ids)
    assert [frame.frame_id for frame in received["render_frames"]] == list(
        backend.optimized_node_ids
    )
    assert received["config"]["enabled"] is True
    assert report["schema"] == "gemini305-central-strip-diagnostic/v1"
    assert report["diagnostic_only"] is True
    assert report["deliverable_published"] is False
    assert report["central_strip"]["strip_quality_pass"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "diagnostic_panorama.jpg",
        "diagnostic_report.json",
    ]
    assert not (output / "delivery.json").exists()
    assert not (output / "transforms.json").exists()
    assert not (output / "report.json").exists()


def test_central_strip_callback_uses_dense_real_pose_nodes_end_to_end(
    tmp_path: Path,
) -> None:
    """The callback route must not collapse a central strip to FOV endpoints."""

    session = generate_sequence(
        tmp_path / "session",
        frame_count=10,
        frame_width=640,
        frame_height=400,
        step=20,
        seed=41,
    )
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    report = sequence.run(
        args,
        odometry_backend=backend,
        diagnostic_renderer=render_central_strip_diagnostic,
    )

    selection = report["central_strip"]["selection"]
    assert selection["mode"] == "central_strip_real_pose_nodes"
    assert selection["interpolated_pose_count"] == 0
    assert selection["frame_ids"] == list(backend.optimized_node_ids)
    assert len(selection["frame_ids"]) == 10
    assert report["central_strip"]["strip_quality_pass"] is True
    assert cv2.imread(str(output / "diagnostic_panorama.jpg"), cv2.IMREAD_COLOR) is not None
    assert sorted(path.name for path in output.iterdir()) == [
        "diagnostic_panorama.jpg",
        "diagnostic_report.json",
    ]


def test_central_strip_callback_failure_only_publishes_failure(tmp_path: Path) -> None:
    session = _make_session(tmp_path, seed=37)
    output = tmp_path / "output"
    output.mkdir()
    (output / "delivery.json").write_text("stale", encoding="utf-8")
    backend = _KnownTrajectoryRGBDBackend(session)

    def failing_renderer(**kwargs: object) -> SimpleNamespace:
        del kwargs
        raise RuntimeError("forced central-strip renderer failure")

    args = sequence._parser().parse_args([str(session), "--output", str(output)])
    with pytest.raises(RuntimeError, match="forced central-strip renderer failure"):
        sequence.run(
            args, odometry_backend=backend, diagnostic_renderer=failing_renderer
        )

    assert sorted(path.name for path in output.iterdir()) == ["failure.json"]
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert "forced central-strip renderer failure" in failure["message"]
