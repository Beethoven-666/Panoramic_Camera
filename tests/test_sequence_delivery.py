from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import panorama_demo.stitch_sequence as sequence
from panorama_demo.synthetic import generate_sequence


class _DeliveryTestRGBDBackend:
    name = "delivery_test_rgbd"

    def __init__(self, session: Path, *, mode: str = "ok") -> None:
        manifest = json.loads(
            (session / "manifest.json").read_text(encoding="utf-8")
        )
        self.poses = {
            int(row["frame_id"]): np.asarray(
                row["matrix_row_major"], dtype=np.float64
            ).reshape(4, 4)
            for row in manifest["known_trajectory"]["poses"]
        }
        self.mode = mode

    def estimate_pair(self, *, reference, source, intrinsics, config):
        del intrinsics, config
        if self.mode == "odometry_error":
            raise RuntimeError("forced RGB-D odometry failure")
        reference_id = int(reference.frame_id)
        source_id = int(source.frame_id)
        return {
            "source_to_reference": (
                np.linalg.inv(self.poses[reference_id]) @ self.poses[source_id]
            ),
            "converged": self.mode != "disconnected_graph",
            "fitness": 0.99,
            "rmse_mm": 0.5,
            "information": np.eye(6, dtype=np.float64) * 100.0,
            "backend": self.name,
        }

    def optimize_pose_graph(
        self, *, node_ids, initial_camera_to_world, edges, config
    ):
        del node_ids, edges, config
        return tuple(np.asarray(pose).copy() for pose in initial_camera_to_world)


def _make_session(tmp_path: Path, *, seed: int = 41) -> Path:
    return generate_sequence(
        tmp_path / "session",
        frame_count=6,
        frame_width=320,
        frame_height=200,
        step=60,
        seed=seed,
    )


def _write_stale_delivery(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "panorama.jpg",
        "report.json",
        "transforms.json",
        "render_transforms.json",
        "delivery.json",
        "foreground_mask.png",
        "background_exclusion_mask.png",
        "tsdf_foreground_mask.png",
        "foreground_source_id.png",
        "tsdf_mesh.glb",
        "tsdf_mesh_viewer.html",
    ):
        (output / name).write_bytes(b"stale")


def test_failure_report_removes_stale_deliverables(tmp_path: Path) -> None:
    _write_stale_delivery(tmp_path)
    (tmp_path / "diagnostic_panorama.jpg").write_bytes(b"stale")
    (tmp_path / "diagnostic_report.json").write_bytes(b"stale")

    sequence._write_failure_report(
        tmp_path, tmp_path / "input", RuntimeError("bad GraphCut seam")
    )

    assert not (tmp_path / "panorama.jpg").exists()
    assert not (tmp_path / "delivery.json").exists()
    assert not (tmp_path / "diagnostic_panorama.jpg").exists()
    assert not (tmp_path / "diagnostic_report.json").exists()
    for legacy_artifact in (
        "foreground_mask.png",
        "background_exclusion_mask.png",
        "tsdf_foreground_mask.png",
        "foreground_source_id.png",
        "tsdf_mesh.glb",
        "tsdf_mesh_viewer.html",
    ):
        assert not (tmp_path / legacy_artifact).exists()
    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert failure["message"] == "bad GraphCut seam"


def test_clear_delivery_does_not_remove_nondelivery_diagnostics(
    tmp_path: Path,
) -> None:
    diagnostics = tmp_path / "pairs" / "pair.jpg"
    diagnostics.parent.mkdir()
    diagnostics.write_bytes(b"diagnostic")
    (tmp_path / ".report.pending.json").write_bytes(b"partial")

    sequence._clear_delivery_files(tmp_path)

    assert diagnostics.exists()
    assert not (tmp_path / ".report.pending.json").exists()


def test_delivery_marker_is_removed_before_other_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("delivery.json", "panorama.jpg", "report.json"):
        (tmp_path / name).write_bytes(b"stale")
    original_unlink = Path.unlink

    def interrupted_unlink(path: Path, *args, **kwargs) -> None:
        if path.name == "report.json":
            raise OSError("simulated cleanup interruption")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupted_unlink)

    with pytest.raises(OSError, match="cleanup interruption"):
        sequence._clear_delivery_files(tmp_path)

    assert not (tmp_path / "delivery.json").exists()


def test_run_invalidates_delivery_before_configuration_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "delivery.json").write_bytes(b"stale")

    def broken_config(_path: Path | None) -> dict[str, object]:
        assert not (output / "delivery.json").exists()
        raise ValueError("broken configuration")

    monkeypatch.setattr(sequence, "load_config", broken_config)
    args = sequence._parser().parse_args(
        [str(tmp_path / "unused-session"), "--output", str(output)]
    )

    with pytest.raises(ValueError, match="broken configuration"):
        sequence.run(args)

    assert not (output / "delivery.json").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["message"] == "broken configuration"
    assert failure["deliverable_published"] is False


@pytest.mark.parametrize(
    ("capture_pass", "pose_pass", "render_pass", "message"),
    [
        (False, True, True, "input capture quality"),
        (True, False, True, "pose trajectory quality"),
        (True, True, False, "final render quality"),
    ],
)
def test_diagnostic_gate_overrides_cannot_publish(
    capture_pass: bool,
    pose_pass: bool,
    render_pass: bool,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        sequence._ensure_publishable_quality(
            {"quality_pass": capture_pass},
            {"quality_metrics": {"quality_pass": render_pass}},
            {"quality_pass": pose_pass},
        )


def test_manual_render_sources_cannot_reduce_delivery_to_one_frame() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sequence._parse_frame_ids("42")


@pytest.mark.parametrize(
    "legacy_arguments",
    [
        ["--blend-mode", "feather"],
        ["--model", "old.pth"],
        ["--device", "cuda"],
        ["--inference-width", "640"],
        ["--motion-model", "translation"],
        ["--translation-anchor-y", "0.85"],
        ["--strict-unistitch"],
        ["--no-pair-previews"],
    ],
)
def test_cli_does_not_expose_legacy_sequence_algorithm_options(
    legacy_arguments: list[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        sequence._parser().parse_args(["unused-session", *legacy_arguments])


def test_manual_render_frame_ids_cannot_publish_formal_delivery(
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    output = tmp_path / "output"
    _write_stale_delivery(output)
    args = sequence._parser().parse_args(
        [
            str(session),
            "--output",
            str(output),
            "--render-frame-ids",
            "0,1",
        ]
    )

    with pytest.raises(ValueError, match="cannot publish a complete"):
        sequence.run(args, odometry_backend=_DeliveryTestRGBDBackend(session))

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()


def test_diagnostic_capture_requires_force_and_invalidates_stale_delivery(
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
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
    _write_stale_delivery(output)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(RuntimeError, match="diagnostic-only"):
        sequence.run(args, odometry_backend=_DeliveryTestRGBDBackend(session))

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()


@pytest.mark.parametrize("defect", ["calibration", "aligned_depth", "depth_scale"])
def test_diagnostic_force_cannot_bypass_strict_rgbd_session_contract(
    tmp_path: Path,
    defect: str,
) -> None:
    session = _make_session(tmp_path)
    backend = _DeliveryTestRGBDBackend(session)
    if defect == "calibration":
        (session / "calibration.json").unlink()
        expected = "Missing calibration.json"
    elif defect == "aligned_depth":
        with (session / "frames.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            first_row = next(csv.DictReader(handle))
        (session / first_row["aligned_depth_path"]).unlink()
        expected = "Missing aligned depth image"
    else:
        csv_path = session / "frames.csv"
        text = csv_path.read_text(encoding="utf-8")
        text = text.replace(",1.0,", ",0.0,", 1)
        csv_path.write_text(text, encoding="utf-8")
        expected = "depth scale"
    output = tmp_path / "output"
    _write_stale_delivery(output)
    args = sequence._parser().parse_args(
        [str(session), "--output", str(output), "--diagnostic-force"]
    )

    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        sequence.run(args, odometry_backend=backend)

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "diagnostic_panorama.jpg").exists()


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ("odometry", "forced RGB-D odometry failure"),
        ("graph", "disconnected"),
        ("pushbroom", "forced calibrated RGB pushbroom failure"),
    ],
)
def test_rgbd_pipeline_stage_failure_never_leaves_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    message: str,
) -> None:
    session = _make_session(tmp_path)
    output = tmp_path / "output"
    _write_stale_delivery(output)
    mode = {
        "odometry": "odometry_error",
        "graph": "disconnected_graph",
        "pushbroom": "ok",
    }[stage]
    backend = _DeliveryTestRGBDBackend(session, mode=mode)
    if stage == "pushbroom":
        def fail_pushbroom(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("forced calibrated RGB pushbroom failure")

        monkeypatch.setattr(
            sequence, "render_calibrated_rgb_pushbroom", fail_pushbroom
        )
    args = sequence._parser().parse_args(
        [str(session), "--output", str(output), "--diagnostic-force"]
    )

    with pytest.raises(RuntimeError, match=message):
        sequence.run(args, odometry_backend=backend)

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "report.json").exists()
    assert not (output / "diagnostic_panorama.jpg").exists()
    assert not (output / "diagnostic_report.json").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert message in failure["message"]
