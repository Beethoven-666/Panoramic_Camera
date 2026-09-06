import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from panorama_demo.sdk_signals import cooperative_capture_signals
from panorama_demo.sdk_state import ThreeDProcessingError
from panorama_demo.video_sdk_3d import supervise_post_3d


@pytest.mark.parametrize("mode", ["exit", "timeout"])
def test_real_child_failure_and_watchdog_preserve_2d(tmp_path, monkeypatch, mode):
    two_d = tmp_path / "2d"
    two_d.mkdir()
    artifacts = {name: b"immutable" for name in (
        "video_delivery.json", "video_panorama.png", "video_pixel_provenance.npz")}
    for name, content in artifacts.items():
        (two_d / name).write_bytes(content)
    popen = subprocess.Popen
    children = []

    def spawn(_command, **kwargs):
        program = "raise SystemExit(7)" if mode == "exit" else "import time; time.sleep(60)"
        child = popen([sys.executable, "-c", program], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("panorama_demo.video_sdk_3d.subprocess.Popen", spawn)
    with pytest.raises(ThreeDProcessingError, match="2-D was preserved"):
        supervise_post_3d(session=tmp_path, two_d=two_d, output=two_d / "3d",
                          config_path=Path("unused.yaml"), timeout_seconds=0.2)
    assert children[0].poll() is not None
    assert (two_d / "3d/video_3d_failure.json").is_file()
    assert all((two_d / name).read_bytes() == content for name, content in artifacts.items())


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal delivery")
@pytest.mark.parametrize("number", [signal.SIGINT, signal.SIGTERM])
def test_cli_signal_only_requests_cooperative_stop(number):
    args = SimpleNamespace()
    original = signal.getsignal(number)
    with cooperative_capture_signals(args):
        os.kill(os.getpid(), number)
        assert args.cancel_event.is_set()
        assert args.stop_reason == signal.Signals(number).name
    assert signal.getsignal(number) == original
