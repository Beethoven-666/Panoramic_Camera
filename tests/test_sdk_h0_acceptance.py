from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = ROOT / "scripts/run_sdk_h0_acceptance.py"
    spec = importlib.util.spec_from_file_location("run_sdk_h0_acceptance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        artifact_root=root, runs=5, warmups=1, width=848, height=480, fps=60,
        duration=3.0, video_exposure_us=100, maximum_post_seconds=60.0,
        no_preview=True, post_3d_last=True,
    )


def test_h0_uses_public_sdk_and_keeps_all_five_runs(tmp_path: Path) -> None:
    module = _module()
    calls = {"capture": 0, "post": 0}

    class Job:
        state = SimpleNamespace(value="succeeded")
        def __init__(self, result): self._result = result
        def result(self): return self._result

    class SDK:
        def capture_and_process(self, capture_root, output_dir, **kwargs):
            calls["capture"] += 1
            assert kwargs["preview_window"] is False
            if calls["capture"] == 3:
                raise RuntimeError("retained failure")
            return Job(SimpleNamespace(
                session=Path(capture_root) / "session", output_dir=Path(output_dir),
                delivery_path=Path(output_dir) / "video_delivery.json",
                authority="ignore_pose_s013_v11_production", overall_grade="A",
                manual_review_required=False, primary_post_capture_seconds=1.0,
            ))
        def run_post_3d(self, session, two_d_output):
            calls["post"] += 1
            return SimpleNamespace(output_dir=Path(two_d_output) / "3d",
                                   delivery_path=Path(two_d_output) / "3d/video_3d_delivery.json")

    result = module.run(_args(tmp_path / "h0"), sdk_factory=SDK)
    assert calls == {"capture": 6, "post": 1}
    assert len([item for item in result["records"] if not item["warmup"]]) == 5
    assert result["successful_runs"] == 4
    assert result["status"] == "FAIL"
    assert (tmp_path / "h0/run_02/sdk_result.json").is_file()
    assert result["cli_crosscheck"]["status"] == "NOT_EXECUTED"
