from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from panorama_demo.stitch_sequence import (
    _clear_delivery_files,
    _ensure_publishable_quality,
    _parse_frame_ids,
    _write_failure_report,
    main,
)


def test_failure_report_removes_stale_deliverables(tmp_path: Path) -> None:
    for name in (
        "panorama.jpg",
        "report.json",
        "transforms.json",
        "render_transforms.json",
        "delivery.json",
    ):
        (tmp_path / name).write_bytes(b"stale")
    (tmp_path / "diagnostic_panorama.jpg").write_bytes(b"stale")
    (tmp_path / "diagnostic_report.json").write_bytes(b"stale")

    _write_failure_report(tmp_path, tmp_path / "input", RuntimeError("bad seam"))

    assert not (tmp_path / "panorama.jpg").exists()
    assert not (tmp_path / "delivery.json").exists()
    assert not (tmp_path / "diagnostic_panorama.jpg").exists()
    assert not (tmp_path / "diagnostic_report.json").exists()
    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert failure["message"] == "bad seam"


def test_clear_delivery_does_not_remove_diagnostics(tmp_path: Path) -> None:
    diagnostics = tmp_path / "pairs" / "pair.jpg"
    diagnostics.parent.mkdir()
    diagnostics.write_bytes(b"diagnostic")
    (tmp_path / ".report.pending.json").write_bytes(b"partial")

    _clear_delivery_files(tmp_path)

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
        _clear_delivery_files(tmp_path)

    assert not (tmp_path / "delivery.json").exists()


@pytest.mark.parametrize(
    ("capture_pass", "render_pass", "message"),
    [
        (False, True, "input capture quality"),
        (True, False, "final render quality"),
    ],
)
def test_diagnostic_gate_overrides_cannot_publish(
    capture_pass: bool, render_pass: bool, message: str
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _ensure_publishable_quality(
            {"quality_pass": capture_pass},
            {"quality_metrics": {"quality_pass": render_pass}},
        )


def test_manual_render_sources_cannot_reduce_delivery_to_one_frame() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _parse_frame_ids("42")


def test_feather_cli_cannot_publish_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "panorama.jpg").write_bytes(b"stale")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unistitch-sequence",
            str(tmp_path / "unused-input"),
            "--output",
            str(output),
            "--blend-mode",
            "feather",
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        main()

    assert not (output / "panorama.jpg").exists()
    assert not (output / "delivery.json").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert "diagnostic renderer" in failure["message"]


def test_manual_render_frame_cli_cannot_publish_partial_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unistitch-sequence",
            str(tmp_path / "unused-input"),
            "--output",
            str(output),
            "--render-frame-ids",
            "10,20",
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        main()

    assert not (output / "delivery.json").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert "cannot publish a complete" in failure["message"]


def test_unrestricted_auto_exposure_capture_cannot_publish_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "panorama-demo-session/v1",
                "capture_mode": "diagnostic_unrestricted_auto_exposure",
                "diagnostic_only": True,
                "formal_stitch_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    output.mkdir()
    (output / "delivery.json").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "unistitch-sequence",
            str(session),
            "--output",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="1"):
        main()

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert "diagnostic-only" in failure["message"]
    assert "--diagnostic-force" in failure["message"]
