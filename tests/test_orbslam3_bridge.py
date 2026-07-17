from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import panorama_demo.orbslam3_bridge as bridge
import panorama_demo.stitch_sequence as sequence
from panorama_demo.session import RGBDSession, load_rgbd_session
from panorama_demo.synthetic import generate_sequence


def _session(tmp_path: Path) -> RGBDSession:
    root = generate_sequence(
        tmp_path / "session",
        frame_count=3,
        frame_width=64,
        frame_height=48,
        step=8,
        seed=31,
    )
    return load_rgbd_session(root)


def _patch_wsl_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bridge,
        "_resolve_wsl_path",
        lambda _config, value: "/orb/" + str(value).replace("~", "home"),
    )
    monkeypatch.setattr(
        bridge,
        "_windows_path_to_wsl",
        lambda _config, path: "/mnt/fixture/" + Path(path).name,
    )
    monkeypatch.setattr(
        bridge,
        "_run_checked",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            list(command), 0, stdout="", stderr=""
        ),
    )


def _write_complete_trajectory(stage_dir: Path, session: RGBDSession) -> None:
    lines = [
        f"{frame.timestamp_us / 1_000_000.0:.6f} "
        f"{index * 0.001:.6f} 0 0 0 0 0 1"
        for index, frame in enumerate(session.frames)
    ]
    (stage_dir / "CameraTrajectory.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def test_sigsegv_139_retries_once_with_a_fresh_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    _patch_wsl_runtime(monkeypatch)
    calls: list[Path] = []

    def fake_process(
        command: list[str] | tuple[str, ...],
        *,
        stage_dir: Path,
        timeout_seconds: float,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        del timeout_seconds
        calls.append(stage_dir)
        if len(calls) == 1:
            # A partial file in the crashed attempt must not be read by the
            # second, newly staged attempt.
            (stage_dir / "CameraTrajectory.txt").write_text(
                "0.000000 99 99 99 0 0 0 1\n", encoding="utf-8"
            )
            return subprocess.CompletedProcess(
                list(command), 139, stdout="", stderr="segfault"
            ), 1.25
        _write_complete_trajectory(stage_dir, session)
        return subprocess.CompletedProcess(
            list(command), 0, stdout="tracked", stderr=""
        ), 2.5

    monkeypatch.setattr(bridge, "_run_orbslam3_process", fake_process)
    trajectory = bridge.run_orbslam3_rgbd(
        session.frames,
        session.calibration,
        tmp_path / "orb-work",
    )

    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert trajectory.work_dir == calls[1]
    assert trajectory.trajectory_path.parent == calls[1]
    assert trajectory.tracked_frame_ids == tuple(frame.frame_id for frame in session.frames)
    assert trajectory.attempt_audit == (
        {
            "attempt_index": 1,
            "returncode": 139,
            "signal": 11,
            "elapsed_seconds": 1.25,
            "accepted": False,
            "retry_reason": "returncode_139_native_sigsegv_fresh_staging_retry",
        },
        {
            "attempt_index": 2,
            "returncode": 0,
            "signal": None,
            "elapsed_seconds": 2.5,
            "accepted": True,
            "retry_reason": None,
        },
    )
    report = trajectory.as_dict(input_frame_count=len(session.frames))
    assert report["execution_attempts"] == list(trajectory.attempt_audit)


def test_native_heap_abort_134_retries_once_with_a_fresh_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WSL malloc aborts are retried, but never replaced by a fake trajectory."""

    session = _session(tmp_path)
    _patch_wsl_runtime(monkeypatch)
    calls: list[Path] = []

    def fake_process(
        command: list[str] | tuple[str, ...],
        *,
        stage_dir: Path,
        timeout_seconds: float,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        del timeout_seconds
        calls.append(stage_dir)
        if len(calls) == 1:
            return subprocess.CompletedProcess(
                list(command), 134, stdout="", stderr="malloc(): corrupted"
            ), 0.5
        _write_complete_trajectory(stage_dir, session)
        return subprocess.CompletedProcess(list(command), 0, stdout="tracked", stderr=""), 1.0

    monkeypatch.setattr(bridge, "_run_orbslam3_process", fake_process)
    trajectory = bridge.run_orbslam3_rgbd(
        session.frames,
        session.calibration,
        tmp_path / "orb-work",
    )

    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert trajectory.attempt_audit[0] == {
        "attempt_index": 1,
        "returncode": 134,
        "signal": 6,
        "elapsed_seconds": 0.5,
        "accepted": False,
        "retry_reason": "returncode_134_native_heap_abort_fresh_staging_retry",
    }
    assert trajectory.attempt_audit[1]["accepted"] is True


def test_sigsegv_139_twice_fails_closed_after_two_fresh_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    _patch_wsl_runtime(monkeypatch)
    calls: list[Path] = []

    def fake_process(
        command: list[str] | tuple[str, ...],
        *,
        stage_dir: Path,
        timeout_seconds: float,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        del timeout_seconds
        calls.append(stage_dir)
        return subprocess.CompletedProcess(
            list(command), 139, stdout="", stderr="segfault"
        ), 0.5

    monkeypatch.setattr(bridge, "_run_orbslam3_process", fake_process)
    with pytest.raises(bridge.ORBSLAM3Error) as raised:
        bridge.run_orbslam3_rgbd(
            session.frames,
            session.calibration,
            tmp_path / "orb-work",
        )

    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert [row["returncode"] for row in raised.value.attempt_audit] == [139, 139]
    assert [row["retry_reason"] for row in raised.value.attempt_audit] == [
        "returncode_139_native_sigsegv_fresh_staging_retry",
        None,
    ]


def test_non_sigsegv_return_code_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    _patch_wsl_runtime(monkeypatch)
    calls: list[Path] = []

    def fake_process(
        command: list[str] | tuple[str, ...],
        *,
        stage_dir: Path,
        timeout_seconds: float,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        del timeout_seconds
        calls.append(stage_dir)
        return subprocess.CompletedProcess(
            list(command), 1, stdout="", stderr="ordinary failure"
        ), 0.25

    monkeypatch.setattr(bridge, "_run_orbslam3_process", fake_process)
    with pytest.raises(bridge.ORBSLAM3Error) as raised:
        bridge.run_orbslam3_rgbd(
            session.frames,
            session.calibration,
            tmp_path / "orb-work",
        )

    assert len(calls) == 1
    assert raised.value.attempt_audit[0]["returncode"] == 1
    assert raised.value.attempt_audit[0]["retry_reason"] is None


def test_successful_process_with_missing_trajectory_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _session(tmp_path)
    _patch_wsl_runtime(monkeypatch)
    calls: list[Path] = []

    def fake_process(
        command: list[str] | tuple[str, ...],
        *,
        stage_dir: Path,
        timeout_seconds: float,
    ) -> tuple[subprocess.CompletedProcess[str], float]:
        del timeout_seconds
        calls.append(stage_dir)
        return subprocess.CompletedProcess(
            list(command), 0, stdout="", stderr=""
        ), 0.75

    monkeypatch.setattr(bridge, "_run_orbslam3_process", fake_process)
    with pytest.raises(bridge.ORBSLAM3Error, match="did not write CameraTrajectory") as raised:
        bridge.run_orbslam3_rgbd(
            session.frames,
            session.calibration,
            tmp_path / "orb-work",
        )

    assert len(calls) == 1
    assert raised.value.attempt_audit[0]["returncode"] == 0
    assert raised.value.attempt_audit[0]["accepted"] is False


def test_failure_report_keeps_only_scalar_orb_attempt_audit(tmp_path: Path) -> None:
    error = bridge.ORBSLAM3Error(
        "two native attempts failed",
        attempt_audit=(
            {
                "attempt_index": 1,
                "returncode": 139,
                "signal": 11,
                "elapsed_seconds": 1.0,
                "accepted": False,
                "retry_reason": "returncode_139_native_sigsegv_fresh_staging_retry",
            },
            {
                "attempt_index": 2,
                "returncode": 139,
                "signal": 11,
                "elapsed_seconds": 1.1,
                "accepted": False,
                "retry_reason": None,
            },
        ),
    )

    sequence._write_failure_report(tmp_path / "output", tmp_path / "input", error)

    report = (tmp_path / "output" / "failure.json").read_text(encoding="utf-8")
    assert "orbslam3_execution_attempts" in report
    assert ".orbslam3_rgbd-attempt-" not in report
