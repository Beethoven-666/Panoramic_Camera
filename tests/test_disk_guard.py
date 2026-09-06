import pytest

from panorama_demo.disk_guard import DiskGuard
from panorama_demo.sdk_state import CaptureError


def test_preflight_refuses_before_session_creation(tmp_path):
    guard = DiskGuard(tmp_path / "capture", tmp_path / "output", space=lambda _: (100, 100))
    with pytest.raises(CaptureError) as exc:
        guard.preflight()
    assert exc.value.error_code == "LOW_DISK"
    assert not (tmp_path / "capture").exists()


def test_runtime_check_is_bounded_and_uses_committed_write_rate(tmp_path):
    clock = [0.0]
    available = [30 * 1024**3]
    calls = []

    def space(path):
        calls.append(path)
        return available[0], 10000

    guard = DiskGuard(tmp_path, tmp_path, clock=lambda: clock[0], space=space)
    guard.preflight()
    guard.committed(100 * 1024**2)
    clock[0] = 0.5
    guard.committed(100 * 1024**2)
    assert guard.check(1)
    assert len(calls) == 2
    clock[0] = 1.1
    available[0] = 20 * 1024**3
    assert not guard.check(2)
    assert guard.last_report["stop_threshold_bytes"] > available[0]
    assert available[0] > 0  # Stop before ENOSPC.


def test_zero_reserve_and_inode_exhaustion_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        DiskGuard(tmp_path, tmp_path, reserve_gib=0)
    with pytest.raises(CaptureError):
        DiskGuard(tmp_path, tmp_path, space=lambda _: (100 * 1024**3, 0)).preflight()
