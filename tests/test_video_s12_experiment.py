from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
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
    config["anchors"]["maximum_direct_local_path_difference_px"] = 20.0
    config["motion_graph"]["anchor_long"]["minimum_spacing_px"] = 1.0
    config["motion_graph"]["anchor_long"]["maximum_spacing_px"] = 80.0
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
    failure_bundle = tmp_path / "stage_a.failure_bundle"
    assert json.loads((failure_bundle / "failure.json").read_text(encoding="utf-8"))[
        "structural_valid"
    ] is False


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


def test_fixed_region_uses_full_solution_when_reference_is_not_selected(
    tmp_path: Path, monkeypatch
) -> None:
    reference = np.full((80, 120, 3), 127, dtype=np.uint8)
    reference_path = tmp_path / "frame_480.png"
    assert cv2.imwrite(str(reference_path), reference)
    monkeypatch.setattr(
        experiment,
        "_load_validation_regions",
        lambda _run: {
            "box_and_flaps": {
                "reference_frame_id": 480,
                "source_bbox": [40, 20, 70, 50],
            }
        },
    )
    assignment = SimpleNamespace(
        frame_id=481, center_x=90.0, left_x=0, right_x=180, width=180, zero_width=False
    )
    schedule = SimpleNamespace(assignments=(assignment,))
    graph = SimpleNamespace(nodes=(SimpleNamespace(frame_id=480), SimpleNamespace(frame_id=481)))
    render = SimpleNamespace(
        image=np.full((80, 180, 3), 64, dtype=np.uint8),
        owner_frame_id=np.full((80, 180), 481, dtype=np.int32),
        column_frame_id=np.full(180, 481, dtype=np.int32),
    )
    session = SimpleNamespace(
        video=SimpleNamespace(
            rgbd=SimpleNamespace(
                frames=(SimpleNamespace(frame_id=480, color_path=reference_path),)
            )
        )
    )
    rows, failures = experiment._write_region_crops(
        tmp_path / "bundle",
        render,
        schedule,
        graph,
        (100.0, 110.0),
        session,
        SimpleNamespace(cx=60.0),
        "run",
    )
    region = tmp_path / "bundle" / "fixed_regions" / "box_and_flaps"
    assert not failures
    assert rows[0]["available"] is True
    assert (region / "reference.png").is_file()
    assert (region / "panorama.png").is_file()
    assert (region / "owner_overlay.png").is_file()
    provenance = json.loads((region / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["canvas_bbox"] == [80, 20, 110, 50]
    assert provenance["manual_review"]["status"] == "pending"
    assert provenance["lock_eligible"] is False


def _unfinished_report() -> dict:
    return {
        "performance": {
            "algorithm_seconds": 10.0,
            "audit_export_seconds": 6.0,
            "atomic_publish_seconds": 0.0,
            "final_wall_seconds": 0.0,
            "runtime_finalized": False,
            "target_seconds": 20.0,
            "target_pass": False,
        },
        "acceptance": {"pass": True, "structural_fatal": False, "reasons": []},
        "lock_eligible": False,
    }


def test_target_pass_includes_audit_export_and_atomic_publish(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: 21.0)
    report = experiment._finalize_completion(
        output, _unfinished_report(), run_started=0.0, atomic_publish_seconds=3.0
    )
    marker = json.loads(
        (output / "S012_stage_a_completion.json").read_text(encoding="utf-8")
    )
    assert report["performance"]["audit_export_seconds"] == 6.0
    assert marker["atomic_publish_seconds"] == 3.0
    assert marker["final_wall_seconds"] == 21.0
    assert marker["target_pass"] is False
    assert marker["bundle_report_sha256"] == experiment._sha256(output / "report.json")


def test_completion_marker_failure_leaves_lock_ineligible(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    real_atomic = experiment._atomic_write_json

    def fail_marker(path, payload):
        if Path(path).name == "S012_stage_a_completion.json":
            raise OSError("marker failure")
        return real_atomic(path, payload)

    monkeypatch.setattr(experiment, "_atomic_write_json", fail_marker)
    with pytest.raises(OSError, match="marker failure"):
        experiment._finalize_completion(
            output, _unfinished_report(), run_started=0.0, atomic_publish_seconds=1.0
        )
    assert not (output / "S012_stage_a_completion.json").exists()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["lock_eligible"] is False


def test_stage_a_rejects_stage_a_lock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="future Stage B"):
        experiment.run(
            tmp_path,
            trajectory_path=tmp_path / "trajectory.json",
            output=tmp_path / "out",
            stage="a",
            stage_a_lock=tmp_path / "S012_stage_a.lock.json",
        )
