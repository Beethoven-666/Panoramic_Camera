from __future__ import annotations

import argparse
import importlib.util
import json
import pytest
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
        artifact_root=root,
        runs=5,
        warmups=1,
        width=848,
        height=480,
        fps=60,
        duration=3.0,
        video_exposure_us=100,
        maximum_post_seconds=60.0,
        no_preview=True,
        post_3d_last=True,
    )


def test_h0_uses_public_sdk_and_keeps_all_five_runs(
    tmp_path: Path, monkeypatch
) -> None:
    module = _module()
    monkeypatch.setattr(module, "native_host", lambda: True)
    monkeypatch.setattr(module, "verify_capture_contracts", lambda paths: None)
    monkeypatch.setattr(module, "verify_3d", lambda *a, **k: None)
    monkeypatch.setattr(
        module,
        "verify_2d",
        lambda *a, **k: {"capture_stop_to_p3_published_seconds": 1.25},
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: None)
    calls = {"capture": 0, "post": 0}

    class Job:
        state = SimpleNamespace(value="succeeded")

        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class SDK:
        def doctor(self):
            return SimpleNamespace(as_dict=lambda: {})

        def capture_and_process(self, capture_root, output_dir, **kwargs):
            calls["capture"] += 1
            assert kwargs["preview_window"] is False
            if calls["capture"] == 3:
                raise RuntimeError("retained failure")
            return Job(
                SimpleNamespace(
                    session=Path(capture_root) / "session",
                    output_dir=Path(output_dir),
                    delivery_path=Path(output_dir) / "video_delivery.json",
                    authority="ignore_pose_s013_v11_production",
                    overall_grade="A",
                    manual_review_required=False,
                    primary_post_capture_seconds=1.0,
                )
            )

        def run_post_3d(self, session, two_d_output):
            calls["post"] += 1
            return SimpleNamespace(
                output_dir=Path(two_d_output) / "3d",
                delivery_path=Path(two_d_output) / "3d/video_3d_delivery.json",
            )

    args = _args(tmp_path / "h0")
    args.cli_crosscheck_root = tmp_path / "cli"
    args.capture_contracts = tmp_path / "contracts.json"
    args.capture_contracts.write_text(
        json.dumps({"fixed": "x", "auto": "y", "photo": "z"})
    )
    args.runtime_binding = tmp_path / "binding.json"
    args.runtime_binding.write_text("{}")
    args.post_3d_success = tmp_path / "success"
    args.post_3d_failure = tmp_path / "failure"
    result = module.run(args, sdk_factory=SDK)
    assert calls == {"capture": 6, "post": 1}
    assert len([item for item in result["records"] if not item["warmup"]]) == 5
    assert result["successful_runs"] == 4
    assert result["status"] == "FAIL"
    assert (tmp_path / "h0/run_02/sdk_result.json").is_file()
    assert result["cli_crosscheck"]["status"] == "PASS"
    assert result["records"][0]["capture_stop_to_p3_published_seconds"] == 1.25


@pytest.mark.parametrize(
    "native,runs,warmups,error",
    [
        (False, 5, 1, "WSL"),
        (True, 4, 1, "five"),
        (True, 5, 0, "warmup"),
        (True, 5, 1, "CLI"),
    ],
)
def test_h0_rejects_before_camera(tmp_path, monkeypatch, native, runs, warmups, error):
    module = _module()
    monkeypatch.setattr(module, "native_host", lambda: native)
    args = _args(tmp_path / "h0")
    args.runs = runs
    args.warmups = warmups
    with pytest.raises(ValueError, match=error):
        module.run(args, sdk_factory=lambda: pytest.fail("camera must not be opened"))
