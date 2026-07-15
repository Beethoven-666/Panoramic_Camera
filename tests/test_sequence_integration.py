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
        step=60,
        seed=seed,
    )


def test_zero_parameter_rgbd_sequence_publishes_one_complete_delivery(
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path, seed=19)
    output = tmp_path / "output"
    backend = _KnownTrajectoryRGBDBackend(session)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    report = sequence.run(args, odometry_backend=backend)

    panorama = cv2.imread(str(output / "panorama.jpg"), cv2.IMREAD_COLOR)
    assert panorama is not None
    assert panorama.shape[1] > 320
    assert panorama.shape[0] >= 180
    assert (output / "delivery.json").is_file()
    assert not (output / "failure.json").exists()
    assert report["layout_selection"]["mode"] == "adaptive_rgbd_pose_nodes"
    assert report["render"]["quality_metrics"]["quality_pass"] is True
    assert report["pose_quality"]["quality_pass"] is True
    assert report["pose_graph"]["connected"] is True
    assert report["render"]["selection"]["interpolated_pose_count"] == 0
    assert len(backend.optimized_node_ids) >= 2
    assert backend.estimated_pairs

    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    assert transforms["pose_convention"].startswith("camera_to_world")
    assert transforms["translation_unit"] == "mm"
    assert all(
        np.asarray(node["camera_to_world"]).shape == (4, 4)
        for node in transforms["nodes"]
    )
    delivery = json.loads((output / "delivery.json").read_text(encoding="utf-8"))
    assert delivery["quality_pass"] is True
    assert delivery["pose_backend"] == "open3d_rgbd"
    assert delivery["seam_backend"] == "graphcut_depth_constrained"
    assert np.all(np.max(panorama[[0, -1]], axis=2) > 0)
    assert np.all(np.max(panorama[:, [0, -1]], axis=2) > 0)


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
                "'panorama_demo.central_strip'); "
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
    original_render = sequence.render_projected_scan_panorama

    def failing_capture_quality(*args, **kwargs):
        result = original_capture_quality(*args, **kwargs)
        result["quality_pass"] = False
        result["failure_reasons"] = ["forced test input failure"]
        return result

    def failing_render(*args, **kwargs):
        assert kwargs["quality_gate"] is False
        panorama, info = original_render(*args, **kwargs)
        info.quality_metrics["quality_pass"] = False
        return panorama, info

    monkeypatch.setattr(sequence, "assess_capture_quality", failing_capture_quality)
    monkeypatch.setattr(
        sequence, "render_projected_scan_panorama", failing_render
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
