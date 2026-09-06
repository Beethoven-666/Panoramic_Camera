import os
import subprocess
import sys

import pytest

from panorama_demo.camera_lease import CameraLease
from panorama_demo.sdk_state import CaptureError, SDKBusyError


def test_lease_content_does_not_grant_ownership_and_release_allows_next_client(tmp_path):
    first = CameraLease(tmp_path, "305", "one").acquire()
    try:
        with pytest.raises(SDKBusyError):
            CameraLease(tmp_path, "305", "two").acquire()
    finally:
        first.release()
    with CameraLease(tmp_path, "305", "two"):
        assert (tmp_path / "camera-305.lock").is_file()


def test_missing_root_is_not_created_or_silently_replaced(tmp_path):
    with pytest.raises(CaptureError) as exc:
        CameraLease(tmp_path / "absent", "305", "job").acquire()
    assert exc.value.error_code == "CAMERA_LOCK_ROOT_UNAVAILABLE"
    assert not (tmp_path / "absent").exists()


def test_process_death_releases_kernel_lease(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    command = [sys.executable, "-u", "-c",
               "import os,sys; from pathlib import Path; "
               "from panorama_demo.camera_lease import CameraLease; "
               "lease=CameraLease(Path(sys.argv[1]),'305','child').acquire(); "
               "print('locked',flush=True); sys.stdin.readline(); os._exit(9)", str(tmp_path)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=env)
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(SDKBusyError):
            CameraLease(tmp_path, "305", "parent").acquire()
        child.communicate("exit\n", timeout=10)
        assert child.returncode == 9
        with CameraLease(tmp_path, "305", "parent"):
            pass
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
