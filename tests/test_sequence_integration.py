from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from panorama_demo.central_strip import render_central_strip_diagnostic
import panorama_demo.stitch_sequence as sequence
from panorama_demo.synthetic import generate_sequence


def _test_glb(label: str) -> bytes:
    """Return a small, structurally valid glTF 2.0 binary for staging tests."""

    document = json.dumps(
        {"asset": {"version": "2.0", "generator": label}},
        separators=(",", ":"),
    ).encode("utf-8")
    document += b" " * (-len(document) % 4)
    return (
        struct.pack("<III", 0x46546C67, 2, 20 + len(document))
        + struct.pack("<II", len(document), 0x4E4F534A)
        + document
    )


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


def test_odometry_acceleration_audit_reports_measured_cuda_edges() -> None:
    audit = sequence._odometry_acceleration_audit(
        [
            SimpleNamespace(backend="open3d_tensor_cuda_rgbd"),
            SimpleNamespace(backend="open3d_tensor_cuda_rgbd"),
        ]
    )
    assert audit == {
        "selected": "cuda",
        "backend": "open3d_tensor_cuda_rgbd",
        "backends": ["open3d_tensor_cuda_rgbd"],
        "edge_count": 2,
        "cuda_edge_count": 2,
        "reason": "measured_edge_backend_provenance",
    }


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
        return _test_glb("display-only-test"), {
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
    assert report["schema"] == "gemini305-dual-mosaic-report/v11"
    assert report["layout_selection"]["mode"] == "adaptive_rgbd_pose_nodes"
    assert report["render_strategy"] == (
        "trajectory_constrained_depth_aware_multiview_side_scan"
    )
    assert report["mosaicing_method"] == (
        "trajectory_constrained_depth_aware_multi_viewpoint_"
        "side_scan_mosaicing"
    )
    assert report["render"]["backend"] == (
        "overlapping_virtual_perspective_panels_rgbd"
    )
    assert report["render"]["fixed_strip_pushbroom"] is False
    assert report["render"]["ordinary_2d_panorama"] is False
    assert report["render"]["metric_raster_used_for_rgb"] is False
    assert report["render"]["tsdf_used_for_rgb"] is False
    assert report["acceleration"]["cuda_priority"] is True
    assert report["acceleration"]["stages"]["rgb_render_and_geometry"] == (
        report["projection"]["acceleration"]
    )
    assert report["acceleration"]["stages"]["rgbd_odometry"] == {
        "selected": "cpu",
        "backend": "synthetic_known_rgbd",
        "backends": ["synthetic_known_rgbd"],
        "edge_count": 6,
        "cuda_edge_count": 0,
        "reason": "measured_edge_backend_provenance",
    }
    assert report["render"]["pose_interpolation_count"] == 0
    assert report["render"]["real_pose_count"] == len(
        backend.optimized_node_ids
    )
    assert report["render"]["strict_v1_inspection_complete"] is True
    assert report["render"]["strict_incomplete_reasons"] == []
    assert report["metric_mosaic"]["strict_v1_metric_complete"] is True
    assert report["metric_mosaic"]["strict_incomplete_reasons"] == []
    footprint = report["metric_mosaic"]["surface_footprint_audit"]
    assert footprint["point_centres_preserved"] is True
    assert footprint["world_normal_zbuffer_preserved"] is True
    assert footprint["morphological_hole_fill_used"] is False
    assert footprint["invalid_depth_crossing_allowed"] is False
    assert footprint["depth_edge_crossing_allowed"] is False
    assert footprint["fold_crossing_allowed"] is False
    assert footprint["accepted_continuous_surface_support_ratio"] >= (
        footprint["minimum_strict_continuous_surface_support_ratio"]
    )
    assert report["strict_failure_reasons"] == []
    assert report["pose_quality"]["quality_pass"] is True
    assert report["pose_graph"]["connected"] is True
    assert len(backend.optimized_node_ids) >= 2
    assert backend.estimated_pairs
    assert report["render"]["selection"]["policy"] == (
        "all_real_orbslam3_pose_nodes_then_full_fov_panels"
    )
    assert report["render"]["selection"]["pose_frame_count"] == len(
        backend.optimized_node_ids
    )
    assert report["render"]["frame_ids"] == list(backend.optimized_node_ids)
    assert report["render"]["selected_full_fov_source_count"] >= 2
    assert all(
        audit["full_width_source_sampling"] is True
        and audit["central_twenty_percent_only"] is False
        for audit in report["render"]["source_audits"]
    )
    seam = report["render"]["background_seam_audit"]
    topology = seam["panel_chain_topology"]
    assert seam["graphcut_used"] is True
    assert seam["multiband_used"] is True
    assert seam["exposure_compensation_used"] is True
    assert seam["dis_optical_flow_used"] is True
    assert seam["protected_blend_intersection_pixel_count"] == 0
    assert topology["pass"] is True
    assert topology["coverage_closed"] is True
    assert topology["owner_order_monotone"] is True
    assert topology["adjacent_pair_only"] is True
    assert topology["actual_pair_count"] == (
        report["render"]["selected_full_fov_source_count"] - 1
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
    assert render_transforms["schema"] == (
        "trajectory-constrained-rgbd-multiview/v1"
    )
    assert render_transforms["mosaicing_method"] == report["mosaicing_method"]
    assert render_transforms["acceleration"] == report["projection"]["acceleration"]
    assert render_transforms["pixel_source"] == "calibrated_rgb_source_samples"
    assert render_transforms["depth_used_for_output_pixels"] is False
    assert [source["frame_id"] for source in render_transforms["sources"]] == list(
        backend.optimized_node_ids
    )
    assert all("aligned_depth_path" not in source for source in render_transforms["sources"])
    compact_alignment = render_transforms["residual_alignment"]
    assert compact_alignment == {
        "backend": "none",
        "selected_model": "real_se3_virtual_perspective_panels",
        "reason": "V1 inspection does not alter or interpolate real poses",
    }
    compact_geometry = render_transforms["geometry_assisted_seam"]
    assert compact_geometry["backend"] == (
        "depth_confidence_visibility_and_single_owner"
    )
    assert compact_geometry["depth_used_for_output_pixels"] is False
    assert compact_geometry["depth_used_for_local_geometry"] is True
    assert compact_geometry["background_seam_audit"] == seam
    delivery = json.loads((output / "delivery.json").read_text(encoding="utf-8"))
    assert delivery["quality_pass"] is True
    assert delivery["strict_quality_pass"] is True
    assert delivery["delivery_state"] == "published"
    assert delivery["quality_grade"] == "A"
    assert delivery["manual_review_required"] is False
    assert sum(delivery["handoff_fallback_summary"].values()) == len(
        report["render"]["selected_panel_sources"]
    ) - 1
    assert delivery["pose_backend"] == "open3d_rgbd"
    assert delivery["mosaicing_method"] == report["mosaicing_method"]
    assert delivery["acceleration"] == report["acceleration"]
    assert delivery["projection"] == (
        "trajectory_constrained_rgbd_virtual_panels"
    )
    assert delivery["schema"] == "gemini305-panorama-delivery/v11"
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
    assert delivery["seam_backend"] == (
        "opencv_graphcut_guided_monotone_adjacent_chain"
    )
    assert delivery["blend_backend"] == (
        "safe_background_local_multiband_owner_boundary"
    )
    for product_name in (
        "mosaic_metric.png",
        "mosaic_depth.exr",
        "mosaic_confidence.png",
        "mosaic_owner.png",
        "mosaic_meta.json",
        "mosaic_inspection.png",
        "inspection_owner.png",
        "mosaic_inspection_full_extent.png",
        "inspection_full_extent_owner.png",
        "inspection_meta.json",
    ):
        assert (output / product_name).is_file()
    metric_meta = json.loads(
        (output / "mosaic_meta.json").read_text(encoding="utf-8")
    )
    assert metric_meta["schema"] == "gemini305-metric-mosaic/v1"
    assert metric_meta["coordinate_system"]["pixel_size_mm"] == 2.0
    metric_rgb = cv2.imread(
        str(output / "mosaic_metric.png"), cv2.IMREAD_UNCHANGED
    )
    metric_confidence = cv2.imread(
        str(output / "mosaic_confidence.png"), cv2.IMREAD_UNCHANGED
    )
    metric_owner = cv2.imread(
        str(output / "mosaic_owner.png"), cv2.IMREAD_UNCHANGED
    )
    assert metric_rgb is not None
    assert metric_confidence.dtype == np.uint16
    assert metric_owner.dtype == np.uint16
    assert metric_rgb.shape[:2] == metric_confidence.shape == metric_owner.shape
    inspection_rgb = cv2.imread(
        str(output / "mosaic_inspection.png"), cv2.IMREAD_UNCHANGED
    )
    inspection_owner = cv2.imread(
        str(output / "inspection_owner.png"), cv2.IMREAD_UNCHANGED
    )
    inspection_full_extent = cv2.imread(
        str(output / "mosaic_inspection_full_extent.png"),
        cv2.IMREAD_UNCHANGED,
    )
    inspection_full_extent_owner = cv2.imread(
        str(output / "inspection_full_extent_owner.png"),
        cv2.IMREAD_UNCHANGED,
    )
    assert inspection_rgb.shape[:2] == inspection_owner.shape
    assert inspection_full_extent.shape[2] == 4
    assert (
        inspection_full_extent.shape[:2]
        == inspection_full_extent_owner.shape
    )
    assert delivery["products"]["metric"]["pixel_size_mm"] == 2.0
    visualization = report["tsdf_visualization"]
    assert visualization["status"] == "published"
    assert visualization["required_for_delivery"] is True
    assert visualization["display_only"] is True
    assert visualization["participates_in_panorama"] is False
    assert visualization["mesh"] == "tsdf_mesh.glb"
    assert visualization["viewer"] == "tsdf_mesh_viewer.html"
    assert (output / "tsdf_mesh.glb").read_bytes() == _test_glb(
        "display-only-test"
    )
    viewer = (output / "tsdf_mesh_viewer.html").read_text(encoding="utf-8")
    assert 'src="tsdf_mesh.glb"' in viewer
    assert "model-viewer" in viewer
    assert delivery["tsdf_visualization"] == visualization
    foreground_summary = report["foreground_owner_continuity_summary"]
    assert delivery["foreground_owner_continuity_summary"] == foreground_summary
    assert foreground_summary["all_components_single_owner"] is True
    assert foreground_summary["foreground_blend_pixel_count"] == 0
    for legacy_artifact in (
        "foreground_mask.png",
        "background_exclusion_mask.png",
        "tsdf_foreground_mask.png",
    ):
        assert not (output / legacy_artifact).exists()
    crop = report["render"]["crop"]
    assert panorama.shape[:2] == (crop["height"], crop["width"])
    assert report["render"]["invalid_pixel_count"] == 0
    assert {path.name for path in output.iterdir()} == {
        "panorama.jpg",
        "tsdf_mesh.glb",
        "tsdf_mesh_viewer.html",
        "report.json",
        "transforms.json",
        "render_transforms.json",
        "delivery.json",
        "mosaic_metric.png",
        "mosaic_depth.exr",
        "mosaic_confidence.png",
        "mosaic_owner.png",
        "mosaic_meta.json",
        "mosaic_inspection.png",
        "inspection_owner.png",
        "mosaic_inspection_full_extent.png",
        "inspection_full_extent_owner.png",
        "inspection_meta.json",
    }


def test_tsdf_export_failure_fails_closed_and_removes_all_deliverables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing required mesh makes the entire formal attempt F."""

    session = _make_session(tmp_path, seed=19)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    import panorama_demo.dense_fusion as dense_fusion

    def fail_export(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("forced display-only TSDF export failure")

    monkeypatch.setattr(dense_fusion, "export_tsdf_mesh", fail_export)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(RuntimeError, match="forced display-only TSDF export failure"):
        sequence.run(args, odometry_backend=backend)

    assert [path.name for path in output.iterdir()] == ["failure.json"]
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert "forced display-only TSDF export failure" in failure["message"]
    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "tsdf_mesh.glb").exists()
    assert not (output / "tsdf_mesh_viewer.html").exists()


def test_tsdf_viewer_staging_failure_fails_closed_and_removes_all_deliverables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The GLB and viewer are inseparable required formal artifacts."""

    session = _make_session(tmp_path, seed=19)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)

    monkeypatch.setattr(
        sequence,
        "_export_display_only_tsdf_mesh",
        lambda *args, **kwargs: (
            _test_glb("viewer-staging-test"),
            {
                "backend": "fake_tsdf_display_only",
                "display_only": True,
                "participates_in_panorama": False,
            },
        ),
    )

    def fail_viewer(mesh_filename: str) -> str:
        assert mesh_filename == "tsdf_mesh.glb"
        raise OSError("forced TSDF viewer staging failure")

    monkeypatch.setattr(sequence, "_mesh_viewer_html", fail_viewer)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(OSError, match="forced TSDF viewer staging failure"):
        sequence.run(args, odometry_backend=backend)

    assert [path.name for path in output.iterdir()] == ["failure.json"]
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert "forced TSDF viewer staging failure" in failure["message"]
    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "tsdf_mesh.glb").exists()
    assert not (output / "tsdf_mesh_viewer.html").exists()


def test_tsdf_viewer_publish_failure_is_atomic_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed viewer rename removes already-published formal siblings too."""

    session = _make_session(tmp_path, seed=19)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    monkeypatch.setattr(
        sequence,
        "_export_display_only_tsdf_mesh",
        lambda *args, **kwargs: (
            _test_glb("viewer-publish-test"),
            {
                "backend": "fake_tsdf_display_only",
                "display_only": True,
                "participates_in_panorama": False,
            },
        ),
    )
    original_replace = sequence.os.replace

    def fail_viewer_publish(source: object, destination: object) -> None:
        if Path(destination).name == "panorama.jpg":
            assert all(
                (output / name).is_file()
                for name in (
                    ".panorama.pending.jpg",
                    ".tsdf_mesh.pending.glb",
                    ".tsdf_mesh_viewer.pending.html",
                    ".transforms.pending.json",
                    ".render_transforms.pending.json",
                    ".report.pending.json",
                    ".delivery.pending.json",
                )
            )
        if (
            Path(source).name == ".tsdf_mesh_viewer.pending.html"
            and Path(destination).name == "tsdf_mesh_viewer.html"
        ):
            raise OSError("forced TSDF viewer publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(sequence.os, "replace", fail_viewer_publish)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(OSError, match="forced TSDF viewer publication failure"):
        sequence.run(args, odometry_backend=backend)

    assert [path.name for path in output.iterdir()] == ["failure.json"]
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert "forced TSDF viewer publication failure" in failure["message"]
    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "tsdf_mesh.glb").exists()
    assert not (output / "tsdf_mesh_viewer.html").exists()


def test_delivery_marker_publish_failure_is_atomic_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final success marker cannot leave any sibling artifact published."""

    session = _make_session(tmp_path, seed=19)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    monkeypatch.setattr(
        sequence,
        "_export_display_only_tsdf_mesh",
        lambda *args, **kwargs: (
            _test_glb("delivery-marker-publish-test"),
            {
                "backend": "fake_tsdf_display_only",
                "display_only": True,
                "participates_in_panorama": False,
            },
        ),
    )
    original_replace = sequence.os.replace

    def fail_delivery_publish(source: object, destination: object) -> None:
        if (
            Path(source).name == ".delivery.pending.json"
            and Path(destination).name == "delivery.json"
        ):
            raise OSError("forced delivery marker publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(sequence.os, "replace", fail_delivery_publish)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(OSError, match="forced delivery marker publication failure"):
        sequence.run(args, odometry_backend=backend)

    assert [path.name for path in output.iterdir()] == ["failure.json"]


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
        lambda *args, **kwargs: (_test_glb("policy-test"), {
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
    assert report["render_strategy"] == (
        "trajectory_constrained_depth_aware_multiview_side_scan"
    )
    assert "geometry_assisted_seam" not in report["render"]
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
                "'panorama_demo.dense_fusion'); "
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
