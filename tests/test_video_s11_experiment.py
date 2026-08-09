from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import panorama_demo.video_s1_experiment as experiment
import panorama_demo.video_s1_renderer as renderer_module
from panorama_demo.quality import FrameQuality
from panorama_demo.session import CameraIntrinsics, RGBDFrame, RGBDSession
from panorama_demo.video_s1_layout import S1SourceLayout
from panorama_demo.video_s1_renderer import S11StageProvenance, VideoS11RenderResult
from panorama_demo.video_session import VideoSession


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/video_candidates/S011_object_safe_straight_handoff_v1.yaml"


def _provenance(height: int = 4, width: int = 6) -> S11StageProvenance:
    owner = np.full((height, width), 7, dtype=np.int32)
    valid = np.ones((height, width), dtype=bool)
    return S11StageProvenance(owner, valid, owner.copy(), valid.copy())


def test_stage_provenance_archive_has_exact_four_arrays(tmp_path: Path) -> None:
    target = tmp_path / "stage_a_provenance.npz"
    experiment._write_stage_provenance(target, _provenance())

    with np.load(target) as archive:
        assert set(archive.files) == {
            "geometric_owner_frame_id",
            "declared_owner_sample_valid",
            "final_owner_frame_id",
            "final_valid",
        }
        assert archive["final_owner_frame_id"].dtype == np.int32
        assert archive["final_valid"].dtype == np.bool_


def test_stage_provenance_rejects_foreign_owner_and_invalid_owner(tmp_path: Path) -> None:
    provenance = _provenance()
    foreign = provenance.final_owner_frame_id.copy()
    foreign[0, 0] = 8
    with pytest.raises(ValueError, match="geometric hard owner"):
        experiment._write_stage_provenance(
            tmp_path / "foreign.npz",
            S11StageProvenance(
                provenance.geometric_owner_frame_id,
                provenance.declared_owner_sample_valid,
                foreign,
                provenance.final_valid,
            ),
        )
    final_valid = provenance.final_valid.copy()
    final_valid[0, 0] = False
    with pytest.raises(ValueError, match="owner -1"):
        experiment._write_stage_provenance(
            tmp_path / "invalid.npz",
            S11StageProvenance(
                provenance.geometric_owner_frame_id,
                provenance.declared_owner_sample_valid,
                provenance.final_owner_frame_id,
                final_valid,
            ),
        )


def test_v2_run_requires_explicit_frozen_baseline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires --baseline"):
        experiment.run(tmp_path, CONFIG, tmp_path / "output")


def test_v2_run_writes_fixed_stages_provenance_report_and_performance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 4, 6
    color = tmp_path / "color.png"
    depth = tmp_path / "depth.png"
    assert cv2.imwrite(str(color), np.full((height, width, 3), 96, np.uint8))
    assert cv2.imwrite(str(depth), np.full((height, width), 1000, np.uint16))
    frame = RGBDFrame(7, color, depth, 1.0, 1000, 5, 1)
    calibration = CameraIntrinsics(
        width, height, 10.0, 10.0, width / 2, height / 2, (0.0,) * 8
    )
    session = VideoSession(
        RGBDSession(
            tmp_path,
            calibration,
            (frame,),
            {"capture_mode": "continuous_rgbd_video_auto"},
        ),
        "continuous_rgbd_video_auto",
        True,
        True,
    )
    quality = FrameQuality(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    layout = S1SourceLayout(7, 0, 3.0, 0, 6, 0, 6, 0, 6, 0.0)
    image = np.full((height, width, 3), 96, np.uint8)
    provenance = _provenance(height, width)
    result = VideoS11RenderResult(
        image,
        image.copy(),
        image.copy(),
        provenance,
        provenance,
        provenance,
        (layout,),
        (layout,),
        (layout,),
        (0,),
        (),
        1,
        {"render": 0.01},
        (),
    )
    verified = {
        "schema": "gemini305-video-s1-baseline-lock/v1",
        "run_name": "synthetic",
        "expected": {"session": {"manifest_sha256": "a"}, "artifacts": {"report_sha256": "b"}},
        "report": {"performance": {"stage_seconds": {"total": 10.0}}},
        "layout": {},
        "baseline_path": str(tmp_path / "baseline"),
        "lock_path": str(tmp_path / "lock.json"),
    }
    monkeypatch.setattr(experiment, "load_video_session", lambda *args, **kwargs: session)
    monkeypatch.setattr(experiment, "verify_s01_baseline", lambda *args, **kwargs: verified)
    monkeypatch.setattr(
        experiment,
        "analyse_video_scan",
        lambda *args, **kwargs: (
            [quality],
            [],
            {"start_index": 0, "end_index": 0, "scan_direction": 1},
        ),
    )
    received: dict[str, object] = {}

    def fake_render(*args, **kwargs):
        received.update(kwargs)
        return result

    monkeypatch.setattr(renderer_module, "render_video_s11", fake_render, raising=False)
    output = tmp_path / "out"
    report = experiment.run(
        tmp_path,
        CONFIG,
        output,
        baseline_path=tmp_path / "baseline",
    )

    assert received["baseline"] is verified
    assert report["schema"] == "gemini305-video-s1-experiment-report/v2"
    assert report["source_lineage"]["rescue_count"] == 0
    assert report["needs_s2_count"] == 0
    assert report["motion_role"] == "diagnostic_layout_evidence_only"
    for name in (
        "s11_nominal_midpoint_owner_only.png",
        "s0_panorama_owner_only.png",
        "s11_panorama_owner_only.png",
        "stage_a_provenance.npz",
        "stage_b_provenance.npz",
        "stage_c_provenance.npz",
        "source_layout.json",
        "config_snapshot.yaml",
        "performance.json",
        "report.json",
    ):
        assert (output / name).is_file(), name
    snapshot = (output / "config_snapshot.yaml").read_text(encoding="utf-8")
    assert "motion_role: diagnostic_layout_evidence_only" in snapshot

    minimal_report = experiment.run(
        tmp_path,
        CONFIG,
        output,
        baseline_path=tmp_path / "baseline",
        output_mode="minimal",
    )
    assert minimal_report["performance"]["mode"] == "minimal"
    assert not (output / "source_layout.json").exists()
    assert not (output / "config_snapshot.yaml").exists()


def test_v2_staged_publish_replaces_the_whole_bundle(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "pair_audit").mkdir()
    (output / "pair_audit" / "stale.json").write_text("audit", encoding="utf-8")
    staging = tmp_path / ".result.staging-test"
    staging.mkdir()
    (staging / "report.json").write_text("minimal", encoding="utf-8")

    experiment._publish_staged_bundle(staging, output)

    assert (output / "report.json").read_text(encoding="utf-8") == "minimal"
    assert not (output / "pair_audit").exists()
    assert not staging.exists()


def test_v2_failed_staging_keeps_previous_complete_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "result"
    output.mkdir()
    (output / "report.json").write_text("previous", encoding="utf-8")

    def fail(*args, **kwargs):
        Path(args[1], "partial.png").write_bytes(b"partial")
        raise RuntimeError("render failed")

    monkeypatch.setattr(experiment, "_run_s11", fail)
    with pytest.raises(RuntimeError, match="render failed"):
        experiment._run_s11_atomically(
            tmp_path,
            output,
            SimpleNamespace(),
            tmp_path / "baseline",
            started=0.0,
        )

    assert (output / "report.json").read_text(encoding="utf-8") == "previous"
    assert not (output / "partial.png").exists()
    assert not list(tmp_path.glob(".result.staging-*"))


def test_pair_manual_review_flag_is_preserved_fail_closed() -> None:
    assert experiment._pair_manual_review_required(
        SimpleNamespace(needs_s2=False, pair_status="unobservable_midpoint")
    )
    assert experiment._pair_manual_review_required(
        SimpleNamespace(needs_s2=True, pair_status="straight_seam_limited")
    )
    assert not experiment._pair_manual_review_required(
        SimpleNamespace(needs_s2=False, pair_status="midpoint_safe")
    )


def test_audit_writes_fixed_fast_lineage_crops(tmp_path: Path) -> None:
    image = np.full((8, 32, 3), 80, np.uint8)
    pair_results = tuple(
        SimpleNamespace(
            origin_pair_lineage=lineage,
            boundary_x=16,
            pair_index=index,
            pair_status="midpoint_safe",
            needs_s2=False,
            heldout_metrics={"evaluable_fraction": 1.0},
        )
        for index, lineage in enumerate(((69, 71), (106, 109), (109, 112)))
    )
    result = SimpleNamespace(
        pair_results=pair_results,
        stage_a_nominal_midpoint_owner_only=image,
        stage_b_handoff_only=image,
        stage_c_final=image,
    )

    experiment._write_fast_regressions(
        tmp_path,
        result,
        {"run_name": "run_20260807_140140"},
    )

    for left, right in ((69, 71), (106, 109), (109, 112)):
        crop = tmp_path / "fast_acceptance" / f"lineage_{left}_to_{right}"
        assert (crop / "stage_a_nominal_midpoint_crop.png").is_file()
        assert (crop / "stage_b_handoff_only_crop.png").is_file()
        assert (crop / "stage_c_final_crop.png").is_file()
        assert (crop / "handoff_overlay.png").is_file()
        assert json.loads((crop / "acceptance.json").read_text(encoding="utf-8"))[
            "origin_pair_lineage"
        ] == [left, right]


def test_slow_box_main_edge_peak_audit_rejects_a_parallel_duplicate() -> None:
    height, width = 111, 62
    y_grid, x_grid = np.mgrid[:height, :width]
    single = np.where(y_grid >= 66.0 + 1.1 * x_grid, 190, 40).astype(np.uint8)
    duplicated = np.where(
        (y_grid >= 62.0 + 1.1 * x_grid)
        & (y_grid < 70.0 + 1.1 * x_grid),
        190,
        40,
    ).astype(np.uint8)

    single_metrics = experiment._measure_box_main_edge_peaks(single)
    duplicate_metrics = experiment._measure_box_main_edge_peaks(duplicated)

    assert single_metrics["peak_count"] == 1
    assert duplicate_metrics["peak_count"] == 2
    assert len(duplicate_metrics["peak_intercepts"]) == 2
    assert (
        duplicate_metrics["peak_intercepts"][1]
        - duplicate_metrics["peak_intercepts"][0]
        >= 3.0
    )


def test_cli_forwards_baseline_and_output_mode_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(input_path, config_path, output, baseline_path=None, output_mode=None):
        captured.update({
            "input": input_path,
            "config": config_path,
            "output": output,
            "baseline": baseline_path,
            "output_mode": output_mode,
        })
        return {}

    monkeypatch.setattr(experiment, "run", fake_run)
    assert experiment.main([
        str(tmp_path / "input"),
        "--config", str(CONFIG),
        "--baseline", str(tmp_path / "baseline"),
        "--output-mode", "minimal",
        "--output", str(tmp_path / "output"),
    ]) == 0
    assert captured["baseline"] == tmp_path / "baseline"
    assert captured["output_mode"] == "minimal"
