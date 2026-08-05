from __future__ import annotations

from pathlib import Path

import pytest

import panorama_demo.online_orbslam3_bridge as online
from panorama_demo.orbslam3_bridge import ORBSLAM3Config, ORBSLAM3Error
from panorama_demo.session import CameraIntrinsics, RGBDFrame


class _FakeStdin:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.closed = False

    def write(self, value: str) -> int:
        self.lines.append(value)
        return len(value)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self.lines = list(lines)

    def readline(self) -> str:
        return self.lines.pop(0) if self.lines else ""

    def read(self) -> str:
        value = "".join(self.lines)
        self.lines.clear()
        return value


class _FakeProcess:
    def __init__(self, *, stdout_lines: list[str], on_wait=None, returncode: int = 0) -> None:
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout(stdout_lines)
        self.stderr = _FakeStdout([])
        self._on_wait = on_wait
        self._returncode = returncode
        self.wait_called = False
        self.killed = False

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.wait_called = True
        if self._on_wait is not None:
            self._on_wait()
        return self._returncode

    def poll(self):
        return self._returncode if self.wait_called or self.killed else None

    def kill(self) -> None:
        self.killed = True


def _frames() -> tuple[RGBDFrame, RGBDFrame]:
    return (
        RGBDFrame(41, Path("color_41.png"), Path("depth_41.png"), 1.0, 1_000_000),
        RGBDFrame(43, Path("color_43.png"), Path("depth_43.png"), 1.0, 1_050_000),
    )


def _launch(tmp_path: Path) -> online.OnlineORBSLAM3Launch:
    return online.OnlineORBSLAM3Launch(
        work_dir=tmp_path,
        settings_path=tmp_path / "settings.yaml",
        trajectory_path=tmp_path / "CameraTrajectory.txt",
        protocol_path=tmp_path / "stream_protocol.txt",
        command=("wsl.exe", "-e", "rgbd_g305_stream_headless"),
        config=ORBSLAM3Config(minimum_tracked_fraction=1.0),
    )


def test_persistent_protocol_is_monotonic_and_parses_only_after_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = _frames()
    launch = _launch(tmp_path)

    def write_final_trajectory() -> None:
        assert not launch.trajectory_path.exists()
        launch.trajectory_path.write_text(
            "1.000000 0 0 0 0 0 0 1\n1.050000 0.001 0 0 0 0 0 1\n",
            encoding="utf-8",
        )

    process = _FakeProcess(
        stdout_lines=[
            "G305_STREAM_READY\n",
            "G305_STREAM_ACCEPTED 41\n",
            "G305_STREAM_ACCEPTED 43\n",
            "G305_STREAM_FINALIZED 43\n",
        ],
        on_wait=write_final_trajectory,
    )
    monkeypatch.setattr(online.subprocess, "Popen", lambda *_args, **_kwargs: process)

    runner = online.PersistentORBSLAM3Runner(launch)
    runner.start()
    runner.submit(frames[0], staged_color_path_wsl="/tmp/color41.png", staged_depth_path_wsl="/tmp/depth41.png")
    runner.submit(frames[1], staged_color_path_wsl="/tmp/color43.png", staged_depth_path_wsl="/tmp/depth43.png")
    trajectory = runner.finalize()

    assert process.wait_called
    assert runner.finalized
    assert trajectory.tracked_frame_ids == (41, 43)
    assert trajectory.poses_by_frame_id[43][0, 3] == pytest.approx(1.0)
    assert process.stdin.lines == [
        "FRAME 41 1.000000 /tmp/color41.png /tmp/depth41.png\n",
        "FRAME 43 1.050000 /tmp/color43.png /tmp/depth43.png\n",
        "FINALIZE\n",
    ]
    assert launch.protocol_path.read_text(encoding="utf-8").splitlines()[-1] == "FINALIZE"


def test_persistent_protocol_rejects_non_monotonic_frame_before_native_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _frames()
    process = _FakeProcess(stdout_lines=["G305_STREAM_READY\n", "G305_STREAM_ACCEPTED 43\n"])
    monkeypatch.setattr(online.subprocess, "Popen", lambda *_args, **_kwargs: process)
    runner = online.PersistentORBSLAM3Runner(_launch(tmp_path))
    runner.start()
    runner.submit(second, staged_color_path_wsl="/tmp/color43.png", staged_depth_path_wsl="/tmp/depth43.png")

    with pytest.raises(ORBSLAM3Error, match="strictly monotonic"):
        runner.submit(first, staged_color_path_wsl="/tmp/color41.png", staged_depth_path_wsl="/tmp/depth41.png")

    assert process.stdin.lines == ["FRAME 43 1.050000 /tmp/color43.png /tmp/depth43.png\n"]


def test_runner_rejects_unquoted_whitespace_in_staged_wsl_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _frames()[0]
    process = _FakeProcess(stdout_lines=["G305_STREAM_READY\n"])
    monkeypatch.setattr(online.subprocess, "Popen", lambda *_args, **_kwargs: process)
    runner = online.PersistentORBSLAM3Runner(_launch(tmp_path))
    runner.start()

    with pytest.raises(ValueError, match="cannot contain whitespace"):
        runner.submit(frame, staged_color_path_wsl="/tmp/color 41.png", staged_depth_path_wsl="/tmp/depth41.png")

    assert process.stdin.lines == []


def test_native_stream_failure_has_scalar_execution_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _frames()
    process = _FakeProcess(
        stdout_lines=[
            "G305_STREAM_READY\n",
            "G305_STREAM_ACCEPTED 41\n",
            "G305_STREAM_ACCEPTED 43\n",
            "G305_STREAM_FINALIZED 43\n",
        ],
        returncode=7,
    )
    process.stderr = _FakeStdout(["native tracking failure\n"])
    monkeypatch.setattr(online.subprocess, "Popen", lambda *_args, **_kwargs: process)
    runner = online.PersistentORBSLAM3Runner(_launch(tmp_path))
    runner.start()
    runner.submit(first, staged_color_path_wsl="/tmp/color41.png", staged_depth_path_wsl="/tmp/depth41.png")
    runner.submit(second, staged_color_path_wsl="/tmp/color43.png", staged_depth_path_wsl="/tmp/depth43.png")

    with pytest.raises(ORBSLAM3Error, match=r"failed \(7\)") as raised:
        runner.finalize()

    assert raised.value.attempt_audit[0]["returncode"] == 7
    assert raised.value.attempt_audit[0]["accepted"] is False
    assert raised.value.attempt_audit[0]["protocol"] == "persistent_stdin_finalize_only"


def test_prepare_online_runner_selects_stream_executable_and_writes_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        online, "_resolve_wsl_path", lambda _config, value: "/orb/" + str(value).replace("~", "home")
    )
    monkeypatch.setattr(online, "_windows_path_to_wsl", lambda _config, path: "/mnt/" + Path(path).name)
    monkeypatch.setattr(online, "_run_checked", lambda *_args, **_kwargs: None)
    intrinsics = CameraIntrinsics(848, 480, 500.0, 501.0, 424.0, 240.0, ())

    launch = online.prepare_online_orbslam3_runner(
        intrinsics=intrinsics,
        depth_scale_mm_per_unit=1.0,
        tracking_fps=20,
        work_dir=tmp_path,
    )

    assert launch.command[4].endswith("rgbd_g305_stream_headless")
    settings = launch.settings_path.read_text(encoding="utf-8")
    assert "Camera.fps: 20" in settings
    assert "RGBD.DepthMapFactor: 1000" in settings
