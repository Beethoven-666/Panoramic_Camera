from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo import (
    orbslam3_bridge,
    video_3d,
    video_3d_launcher,
    video_3d_postprocess,
    video_session,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published_2d(root: Path, *, input_sha256: dict[str, str] | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "video_report.json").write_text(
        json.dumps({"input_sha256": input_sha256 or {}}), encoding="utf-8"
    )
    (root / "video_delivery.json").write_text(
        json.dumps(
            {
                "schema": "gemini305-video-panorama-delivery/v2",
                "delivery_state": "published",
                "report": "video_report.json",
            }
        ),
        encoding="utf-8",
    )


def _session_files(root: Path) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "calibration.json", "frames.csv"):
        (root / name).write_text(name, encoding="utf-8")
    return {
        key: _sha256(root / name)
        for key, name in (
            ("manifest", "manifest.json"),
            ("calibration", "calibration.json"),
            ("frames_csv", "frames.csv"),
        )
    }


def _fake_video(session_root: Path) -> SimpleNamespace:
    frames = tuple(
        SimpleNamespace(frame_id=index, timestamp_us=index * 150_000)
        for index in range(3)
    )
    calibration = SimpleNamespace(
        width=8,
        height=6,
        fx=10.0,
        fy=10.0,
        cx=4.0,
        cy=3.0,
        distortion=(),
    )
    return SimpleNamespace(
        rgbd=SimpleNamespace(root=session_root, frames=frames, calibration=calibration)
    )


def _trajectory(path: Path, *, session_root: Path) -> Path:
    identity = np.eye(4).tolist()
    path.write_text(
        json.dumps(
            {
                "schema": video_3d_postprocess.TRAJECTORY_SCHEMA,
                "pose_convention": "camera_to_world",
                "translation_unit": "mm",
                "tracked_frame_ids": [0, 1, 2],
                "camera_to_world": [identity, identity, identity],
            }
        ),
        encoding="utf-8",
    )
    (path.parent / "orbslam3_trajectory.lock.json").write_text(
        json.dumps(
            {
                "schema": video_3d_postprocess.TRAJECTORY_LOCK_SCHEMA,
                "trajectory": path.name,
                "trajectory_sha256": _sha256(path),
                "session_input_sha256": {
                    key: _sha256(session_root / name)
                    for key, name in (
                        ("manifest", "manifest.json"),
                        ("calibration", "calibration.json"),
                        ("frames_csv", "frames.csv"),
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_spawn_requires_delivery_and_records_order_before_process_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    two_d = tmp_path / "two_d"
    _published_2d(two_d)
    calls = {"orb": 0, "tsdf": 0, "popen": 0}

    def forbidden_orb(*_args: object, **_kwargs: object) -> None:
        calls["orb"] += 1

    def forbidden_tsdf(*_args: object, **_kwargs: object) -> None:
        calls["tsdf"] += 1

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls["popen"] += 1
        timing = json.loads((two_d / "video_timing.json").read_text(encoding="utf-8"))[
            "isolation"
        ]
        assert timing["three_d_process_spawned_monotonic_ns"] > timing[
            "two_d_delivery_published_monotonic_ns"
        ]
        assert command[1:3] == ["-m", "panorama_demo.video_3d_postprocess"]
        assert command[-2:] == ["--output", str(two_d / "3d")]
        assert "creationflags" in kwargs
        return SimpleNamespace(pid=4321)

    monkeypatch.setattr(orbslam3_bridge, "run_orbslam3_rgbd", forbidden_orb)
    monkeypatch.setattr(video_3d, "export_tsdf_mesh_pair", forbidden_tsdf)
    process = video_3d_launcher.spawn_post_capture_3d(
        session_path=tmp_path / "session",
        two_d_output=two_d,
        config_path=None,
        two_d_delivery_published_monotonic_ns=1,
        two_d_resources_released_monotonic_ns=2,
        popen=fake_popen,
    )

    assert process.pid == 4321
    assert calls == {"orb": 0, "tsdf": 0, "popen": 1}


def test_spawn_before_delivery_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="before 2-D delivery"):
        video_3d_launcher.spawn_post_capture_3d(
            session_path=tmp_path / "session",
            two_d_output=tmp_path / "missing",
            config_path=None,
            two_d_delivery_published_monotonic_ns=1,
            two_d_resources_released_monotonic_ns=2,
            popen=lambda *_args, **_kwargs: None,
        )


def test_spawn_failure_writes_isolated_failure_and_preserves_2d(tmp_path: Path) -> None:
    two_d = tmp_path / "two_d"
    _published_2d(two_d)
    original_sha = _sha256(two_d / "video_delivery.json")

    with pytest.raises(OSError, match="spawn failed"):
        video_3d_launcher.spawn_post_capture_3d(
            session_path=tmp_path / "session",
            two_d_output=two_d,
            config_path=None,
            two_d_delivery_published_monotonic_ns=1,
            two_d_resources_released_monotonic_ns=2,
            popen=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("spawn failed")
            ),
        )

    assert _sha256(two_d / "video_delivery.json") == original_sha
    failure = json.loads((two_d / "3d/video_3d_failure.json").read_text(encoding="utf-8"))
    assert failure["stage"] == "process_spawn"
    assert failure["two_d_delivery_preserved"] is True


def test_postprocess_runs_orb_before_final_3d_without_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    _session_files(session)
    two_d = tmp_path / "two_d"
    _published_2d(two_d)
    output_3d = two_d / "3d"
    calls: list[str] = []
    fake_video = _fake_video(session)

    monkeypatch.setattr(video_session, "load_video_session", lambda *_a, **_k: fake_video)

    def fake_orb(frames: tuple[object, ...], *_args: object, **_kwargs: object) -> object:
        calls.append("orb")
        ids = tuple(frame.frame_id for frame in frames)
        work = Path(_args[1])
        evidence = {}
        for key in ("stdout_path", "stderr_path", "trajectory_path", "association_path"):
            evidence[key] = work / (key + ".txt")
            evidence[key].write_text("unit fixture: " + key)
        return SimpleNamespace(
            tracked_frame_ids=ids,
            poses_by_frame_id={frame_id: np.eye(4) for frame_id in ids},
            attempt_audit=(),
            as_dict=lambda **_kwargs: {"backend": "orbslam3_rgbd_native_linux"},
            **evidence,
        )

    def fake_publish(
        output: Path, *, trajectory_path: Path, source_frame_ids: list[int], **_kwargs: object
    ) -> dict[str, object]:
        calls.append("final_3d")
        assert output == output_3d
        assert trajectory_path.is_file()
        assert source_frame_ids == [0, 1, 2]
        return {"delivery_state": "published", "preview_generated": False}

    monkeypatch.setattr(orbslam3_bridge, "run_orbslam3_rgbd", fake_orb)
    monkeypatch.setattr(video_3d, "publish_video_3d", fake_publish)
    result = video_3d_postprocess.run_post_capture_3d(
        session_path=session,
        two_d_output=two_d,
        output_3d=output_3d,
        config={"stitch": {"video_panorama": {"fast_orb_target_fps": 8.0}}},
    )

    assert calls == ["orb", "final_3d"]
    assert result["preview_generated"] is False
    assert (output_3d / "orbslam3_trajectory.lock.json").is_file()
    assert (output_3d / "video_3d_timing.json").is_file()
    trajectory = json.loads((output_3d / "orbslam3_trajectory.json").read_text())
    assert trajectory["timestamps_us"] == [0, 150000, 300000]
    assert trajectory["backend"] == "orbslam3_rgbd_native_linux"
    for key in ("stdout_file", "stderr_file", "tum_file", "association_file"):
        assert (output_3d / trajectory[key]).is_file()
    assert not tuple(output_3d.glob("*preview*"))


def test_publish_video_3d_writes_only_final_artifacts_under_3d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    input_hashes = _session_files(session)
    two_d = tmp_path / "two_d"
    _published_2d(two_d, input_sha256=input_hashes)
    original_sha = _sha256(two_d / "video_delivery.json")
    output_3d = two_d / "3d"
    output_3d.mkdir()
    trajectory = _trajectory(
        output_3d / "orbslam3_trajectory.json", session_root=session
    )
    monkeypatch.setattr(video_3d, "load_video_session", lambda *_a, **_k: _fake_video(session))
    monkeypatch.setattr(
        video_3d,
        "export_tsdf_mesh_pair",
        lambda *_a, **_k: (b"desktop", b"mobile", {"status": "ok"}),
    )

    result = video_3d.publish_video_3d(
        output_3d,
        input_path=session,
        trajectory_path=trajectory,
        source_frame_ids=[0, 1, 2],
        config={"stitch": {"tsdf_visualization": {}}},
        two_d_output=two_d,
    )

    assert result["preview_generated"] is False
    assert (output_3d / "video_tsdf_mesh.glb").read_bytes() == b"desktop"
    assert (output_3d / "video_tsdf_mesh_mobile.glb").read_bytes() == b"mobile"
    assert (output_3d / "video_tsdf_mesh_viewer.html").is_file()
    assert (output_3d / "video_3d_delivery.json").is_file()
    assert not tuple(output_3d.glob("*preview*"))
    assert _sha256(two_d / "video_delivery.json") == original_sha


@pytest.mark.parametrize("failed_stage", ["tsdf", "mesh_cleanup", "glb_export"])
def test_any_final_3d_stage_failure_preserves_2d_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_stage: str
) -> None:
    session = tmp_path / "session"
    input_hashes = _session_files(session)
    two_d = tmp_path / "two_d"
    _published_2d(two_d, input_sha256=input_hashes)
    two_d_sha = _sha256(two_d / "video_delivery.json")
    output_3d = two_d / "3d"
    output_3d.mkdir()
    trajectory = _trajectory(
        output_3d / "orbslam3_trajectory.json", session_root=session
    )
    monkeypatch.setattr(video_3d, "load_video_session", lambda *_a, **_k: _fake_video(session))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(failed_stage)

    monkeypatch.setattr(video_3d, "export_tsdf_mesh_pair", fail)
    with pytest.raises(RuntimeError, match=failed_stage):
        video_3d.publish_video_3d(
            output_3d,
            input_path=session,
            trajectory_path=trajectory,
            source_frame_ids=[0, 1, 2],
            config={"stitch": {"tsdf_visualization": {}}},
            two_d_output=two_d,
        )

    assert _sha256(two_d / "video_delivery.json") == two_d_sha
    failure = json.loads((output_3d / "video_3d_failure.json").read_text(encoding="utf-8"))
    assert failure["schema"] == "gemini305-video-3d-failure/v2"
    assert failure["two_d_delivery_preserved"] is True
    assert failure["two_d_delivery_sha256"] == two_d_sha


def test_orb_failure_preserves_2d_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path / "session"
    _session_files(session)
    two_d = tmp_path / "two_d"
    _published_2d(two_d)
    two_d_sha = _sha256(two_d / "video_delivery.json")
    monkeypatch.setattr(video_session, "load_video_session", lambda *_a, **_k: _fake_video(session))
    monkeypatch.setattr(
        orbslam3_bridge,
        "run_orbslam3_rgbd",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("orb")),
    )

    with pytest.raises(RuntimeError, match="orb"):
        video_3d_postprocess.run_post_capture_3d(
            session_path=session,
            two_d_output=two_d,
            output_3d=two_d / "3d",
            config={"stitch": {"video_panorama": {"fast_orb_target_fps": 8.0}}},
        )

    assert _sha256(two_d / "video_delivery.json") == two_d_sha
    failure = json.loads((two_d / "3d/video_3d_failure.json").read_text(encoding="utf-8"))
    assert failure["two_d_delivery_preserved"] is True
