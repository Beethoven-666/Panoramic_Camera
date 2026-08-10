from __future__ import annotations

import json
from pathlib import Path

import panorama_demo.video_s12_experiment as experiment
from panorama_demo.synthetic import generate_sequence
import pytest
import yaml


def _video_session_and_trajectory(tmp_path: Path) -> tuple[Path, Path]:
    root = generate_sequence(
        tmp_path / "synthetic_video",
        frame_count=16,
        frame_width=320,
        frame_height=192,
        step=8,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "capture_mode": "continuous_rgbd_video_fixed_exposure",
            "product_eligibility": {
                "photo_panorama": False,
                "video_panorama": True,
            },
            "writer_errors": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    known = manifest["known_trajectory"]["poses"]
    trajectory = {
        "schema": "gemini305-orbslam3-trajectory/v2",
        "pose_convention": "camera_to_world",
        "translation_unit": "mm",
        "poses": [
            {
                "frame_id": row["frame_id"],
                "timestamp_us": row["frame_id"] * 100_000,
                "pose_status": "valid",
                "pose_kind": "direct_orbslam3",
                "pose_origin": "synthetic_direct_orbslam3",
                "tracking_state": "tracked",
                "camera_to_world": [
                    row["matrix_row_major"][offset : offset + 4]
                    for offset in range(0, 16, 4)
                ],
            }
            for row in known
        ],
    }
    trajectory_path = root / "orbslam3_trajectory_v2.json"
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    return root, trajectory_path


def test_s12_cli_has_no_baseline_or_s1_argument() -> None:
    destinations = {action.dest for action in experiment._parser()._actions}
    assert "baseline" not in destinations
    assert "s1" not in destinations
    assert "source_layout_from_s1" not in destinations


def test_stage_b_is_not_implemented_before_review(tmp_path: Path) -> None:
    try:
        experiment.run(
            tmp_path,
            trajectory_path=tmp_path / "trajectory.json",
            output=tmp_path / "out",
            stage="b",
        )
    except ValueError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("Stage B unexpectedly ran")


def test_s12_runs_synthetic_stage_a_without_any_s1_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    root, trajectory = _video_session_and_trajectory(tmp_path)

    def fake_open3d(frames, _poses, _intrinsics, **_kwargs):
        return tuple(
            {
                "reference_node_id": left.frame_id,
                "source_node_id": right.frame_id,
                "backend": "test_open3d_rgbd",
                "quality_pass": True,
            }
            for left, right in zip(frames[:-1], frames[1:])
        )

    monkeypatch.setattr(experiment, "audit_open3d_source_edges", fake_open3d)
    config = yaml.safe_load(experiment._default_config().read_text(encoding="utf-8"))
    config["anchors"]["minimum_anchor_image_quality"] = 0.10
    config["anchors"]["minimum_anchor_edge_confidence"] = 0.10
    config["anchors"]["target_spacing_px"] = 64
    config["anchors"]["minimum_spacing_px"] = 40
    config["anchors"]["maximum_spacing_px"] = 80
    config["anchors"]["target_search_radius_px"] = 20
    config["motion_graph"]["minimum_inlier_ratio"] = 0.05
    config["motion_graph"]["minimum_inlier_count"] = 4
    config["motion_graph"]["minimum_grid_coverage"] = 0.05
    config["motion_graph"]["minimum_vertical_span_fraction"] = 0.10
    config["motion_graph"]["lk_forward_backward_max_px"] = 2.0
    config["motion_graph"]["maximum_parallax_cluster_ratio"] = 0.99
    config_path = tmp_path / "s12.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "stage_a"
    report = experiment.run(
        root,
        trajectory_path=trajectory,
        config_path=config_path,
        output=output,
        stage="a",
    )

    assert report["structural_valid"] is True
    assert report["stage"] == "a"
    assert report["trajectory"]["direct_pose_count"] == 16
    assert report["horizontal"]["global_rescale_applied"] is False
    assert (output / "stage_a_dense_slit.png").is_file()
    assert (output / "owner_frame_id.npz").is_file()
    assert (output / "fixed_regions").is_dir()
    assert not any("s1" in path.name.lower() for path in output.iterdir())


@pytest.mark.parametrize(
    "failure_message",
    ["image write failure", "NPZ write failure", "report write failure"],
)
def test_write_failure_preserves_old_bundle_and_writes_sibling_failure(
    tmp_path: Path, monkeypatch, failure_message: str
) -> None:
    output = tmp_path / "stage_a"
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("keep", encoding="utf-8")

    def fail(*_args, **_kwargs):
        raise RuntimeError(failure_message)

    monkeypatch.setattr(experiment, "_run_stage_a", fail)
    try:
        experiment.run(
            tmp_path,
            trajectory_path=tmp_path / "trajectory.json",
            output=output,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("injected structural failure unexpectedly passed")
    assert marker.read_text(encoding="utf-8") == "keep"
    failure = tmp_path / "stage_a.failure.json"
    assert json.loads(failure.read_text(encoding="utf-8"))["structural_fatal"] is True


def test_atomic_publish_replaces_old_bundle(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".bundle.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    experiment._publish(staging, output)
    assert not (output / "old.txt").exists()
    assert (output / "new.txt").read_text(encoding="utf-8") == "new"


def test_atomic_publish_rename_failure_restores_old_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / ".bundle.staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    real_replace = experiment.os.replace
    failed = False

    def injected_replace(source, destination):
        nonlocal failed
        if Path(source) == staging and Path(destination) == output and not failed:
            failed = True
            raise OSError("injected rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(experiment.os, "replace", injected_replace)
    with pytest.raises(OSError, match="rename failure"):
        experiment._publish(staging, output)
    assert (output / "old.txt").read_text(encoding="utf-8") == "old"
    assert staging.exists()
