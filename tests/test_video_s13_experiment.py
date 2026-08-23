from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from functools import wraps
from typing import Callable

import cv2
import numpy as np
import pytest

from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_bundle import sha256_file, verify_p0_completion
from panorama_demo.video_s13_experiment import run_s13_experiment
from panorama_demo.video_s13_motion import S13MotionEdge, S13MotionHypothesis
from panorama_demo.video_s13_session import load_s13_session, load_s13_session_bundle
from panorama_demo.video_s13_trajectory import load_s13_trajectory
from panorama_demo.video_trajectory_cache import session_input_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v3.yaml"
FORMAL_M6_CONFIG = (
    ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v4.yaml"
)


def _isolated_rss_process(test: Callable[..., None]) -> Callable[..., None]:
    """Run an RSS-capped integration test in a fresh pytest process.

    The production cap is an absolute process-RSS envelope.  A long-lived
    pytest process retains unrelated OpenCV/CUDA allocator pages from earlier
    tests, so RSS-capped integration tests need a process boundary to measure
    only their own test process.  The child executes the original body; the
    parent only relays its result.
    """

    isolated_key = f"{Path(__file__).name}::{test.__name__}"

    @wraps(test)
    def wrapper(*args: object, **kwargs: object) -> None:
        if os.environ.get("G305_PYTEST_RSS_ISOLATED_TEST") == isolated_key:
            test(*args, **kwargs)
            return
        environment = os.environ.copy()
        environment["G305_PYTEST_RSS_ISOLATED_TEST"] = isolated_key
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                f"{Path(__file__).as_posix()}::{test.__name__}",
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(
                "isolated RSS pytest process failed:\n" + result.stdout,
                pytrace=False,
            )

    wrapper.rss_process_isolated = True  # type: ignore[attr-defined]
    return wrapper


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _video_session(tmp_path: Path, *, frame_count: int = 8, step: int = 8) -> Path:
    root = generate_sequence(tmp_path / "video", frame_count=frame_count, frame_width=160, frame_height=96, step=step)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "capture_mode": "continuous_rgbd_video_fixed_exposure",
        "product_eligibility": {"photo_panorama": False, "video_panorama": True},
        "writer_errors": [],
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _run(session: Path, output: Path, *, all_fail: bool = False) -> dict:
    return run_s13_experiment(
        input_path=session,
        output=output,
        candidate_config=CONFIG,
        algorithm_spec=build_algorithm_spec(CONFIG, expected_role="candidate"),
        trajectory_cache=None,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=True,
        config_path=None,
        simulate_optimizer_all_fail=all_fail,
    )


def _run_formal_m6(
    session: Path,
    output: Path,
    *,
    run_m6: bool = True,
    resume_generation: Path | None = None,
) -> dict:
    return run_s13_experiment(
        input_path=session,
        output=output,
        candidate_config=FORMAL_M6_CONFIG,
        algorithm_spec=build_algorithm_spec(FORMAL_M6_CONFIG, expected_role="candidate"),
        trajectory_cache=None,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=True,
        config_path=None,
        run_m6=run_m6,
        resume_generation=resume_generation,
    )


def test_s13_synthetic_p0_has_unique_real_owner_and_full_provenance(tmp_path: Path) -> None:
    session, output = _video_session(tmp_path), tmp_path / "out"
    report = _run(session, output)
    generation = Path(report["generation"])
    completion = verify_p0_completion(generation / "P0")
    arrays = np.load(generation / "P0/base_pixel_provenance.npz")
    assert np.array_equal(arrays["valid"], arrays["owner_frame_id"] >= 0)
    assert np.all(arrays["owner_frame_id"][arrays["valid"]] >= 0)
    assert np.all(np.isfinite(arrays["source_u"][arrays["valid"]]))
    assert np.all(np.isfinite(arrays["source_v"][arrays["valid"]]))
    assert np.all(arrays["secondary_frame_id"] == -1)
    assert np.all(arrays["secondary_weight"] == 0)
    assert completion["duplicate_write_count"] == 0
    assert report["panorama_claim"] == "visual_nonmetric"
    assert (output / "current_base.json").is_file()
    assert not (output / "video_delivery.json").exists()
    assert not (output / "delivery.json").exists()


def test_depth_failure_enters_rgb_salvage_without_changing_visual_contract(tmp_path: Path) -> None:
    session, output = _video_session(tmp_path), tmp_path / "out"
    first_depth = next((session / "depth_aligned").glob("*.png"))
    depth = cv2.imread(str(first_depth), cv2.IMREAD_UNCHANGED)
    assert cv2.imwrite(str(first_depth), np.zeros_like(depth))
    report = _run(session, output)
    audit = json.loads((Path(report["generation"]) / "input_audit.json").read_text(encoding="utf-8"))
    assert audit["strict_session"] is False
    assert audit["depth_used_for_p0"] is False
    assert Path(report["panorama"]).is_file()


def test_missing_calibration_uses_raw_rgb_fallback(tmp_path: Path) -> None:
    session, output = _video_session(tmp_path), tmp_path / "out"
    (session / "calibration.json").unlink()
    report = _run(session, output)
    audit = json.loads((Path(report["generation"]) / "input_audit.json").read_text(encoding="utf-8"))
    assert audit["calibration_mode"] == "raw_rgb_fallback"
    arrays = np.load(Path(report["pixel_provenance"]))
    assert np.all(np.isfinite(arrays["source_u"][arrays["valid"]]))


def test_optimizer_all_fail_preserves_sealed_p0_hashes(tmp_path: Path) -> None:
    session, output = _video_session(tmp_path), tmp_path / "out"
    report = _run(session, output, all_fail=True)
    completion_path = Path(report["completion"])
    before = sha256_file(completion_path)
    completion = verify_p0_completion(completion_path.parent)
    after = sha256_file(completion_path)
    assert before == after
    assert report["optimizer"]["requested"] is True
    assert report["optimizer"]["attempt_count"] == 3
    assert report["optimizer"]["all_failed"] is True
    assert report["optimizer"]["p0_parent_preserved"] is True
    assert report["optimizer"]["p0_hash_unchanged"] is True
    assert completion["sealed"] is True


def test_valid_black_rgb_is_not_removed(tmp_path: Path) -> None:
    session = _video_session(tmp_path)
    for path in (session / "color").glob("*"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert cv2.imwrite(str(path), np.zeros_like(image))
    report = _run(session, tmp_path / "out")
    if report["render_state"] == "base_generated":
        arrays = np.load(Path(report["pixel_provenance"]))
        assert np.any(arrays["valid"])
    else:
        assert report["render_state"] == "temporal_preview_only"


def test_no_readable_rgb_writes_input_fatal_report(tmp_path: Path) -> None:
    session = tmp_path / "empty_session"
    session.mkdir()
    output = tmp_path / "out"
    try:
        _run(session, output)
    except ValueError as exc:
        assert "no decodable real RGB" in str(exc)
    else:
        raise AssertionError("empty S1.3 session unexpectedly rendered")
    failure = json.loads((output / "S013_failure.json").read_text(encoding="utf-8"))
    assert failure["render_state"] == "input_fatal"
    assert failure["panorama_claim"] == "none"


def test_single_calibrated_view_is_inverse_remapped_with_real_uv(tmp_path: Path) -> None:
    session, output = _video_session(tmp_path, frame_count=1), tmp_path / "out"
    report = _run(session, output)
    assert report["asset_type"] == "calibrated_single_view"
    arrays = np.load(Path(report["generation"]) / "nonpanorama/pixel_provenance.npz")
    assert np.array_equal(arrays["valid"], arrays["owner_frame_id"] >= 0)
    assert np.all(np.isfinite(arrays["source_u"][arrays["valid"]]))
    assert np.all(np.isfinite(arrays["source_v"][arrays["valid"]]))


def test_invalid_explicit_trajectory_degrades_to_rgb_p0(tmp_path: Path) -> None:
    session, output = _video_session(tmp_path), tmp_path / "out"
    invalid = tmp_path / "invalid_trajectory.json"
    invalid.write_text("{}", encoding="utf-8")
    report = run_s13_experiment(
        input_path=session,
        output=output,
        candidate_config=CONFIG,
        algorithm_spec=build_algorithm_spec(CONFIG, expected_role="candidate"),
        trajectory_cache=invalid,
        reuse_online_trajectory=False,
        run_offline_orb=False,
        ignore_pose=False,
        config_path=None,
    )
    audit = json.loads((Path(report["generation"]) / "trajectory_audit.json").read_text(encoding="utf-8"))
    assert audit["trajectory_valid"] is False
    assert audit["fallback"] == "rgb_motion_without_pose"
    assert report["panorama_claim"] == "visual_nonmetric"


def test_strict_s13_bundle_reuses_validated_rgb_without_second_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _video_session(tmp_path)
    from panorama_demo import video_s13_session

    monkeypatch.setattr(
        video_s13_session,
        "_decode",
        lambda _path: pytest.fail("strict S1.3 path decoded RGB a second time"),
    )
    bundle = load_s13_session_bundle(root, validation_workers=2)

    assert bundle.session.strict_video is not None
    assert bundle.performance["strict_rgb_decode_count"] == len(bundle.session.frames)
    assert bundle.performance["strict_depth_decode_count"] == len(bundle.session.frames)
    assert bundle.performance["s13_fallback_rgb_decode_count"] == 0
    images = bundle.validated_rgb_handoff.take_all()
    assert set(images) == {frame.frame_id for frame in bundle.session.frames}
    assert all(image.flags.writeable is False for image in images.values())
    with pytest.raises(RuntimeError, match="already consumed"):
        bundle.validated_rgb_handoff.take_all()


def test_online_trajectory_is_bound_to_committed_current_source_files(tmp_path: Path) -> None:
    root = _video_session(tmp_path)
    session = load_s13_session(root, validation_workers=2)
    assert session.strict_video is not None
    strict = {frame.frame_id: frame for frame in session.strict_video.rgbd.frames}
    payload = {
        "schema": "gemini305-online-orbslam3-trajectory/v1",
        "capture_origin": "writer_committed_files",
        "input_sha256": session_input_sha256(root),
        "pose_convention": "camera_to_world",
        "translation_unit": "mm",
        "tracked_frame_ids": [frame.frame_id for frame in session.frames],
        "camera_to_world": [np.eye(4).tolist() for _frame in session.frames],
        "source_file_sha256": [
            {
                "frame_id": frame.frame_id,
                "color_sha256": _sha256(frame.color_path),
                "aligned_depth_sha256": _sha256(strict[frame.frame_id].aligned_depth_path),
            }
            for frame in session.frames
        ],
    }
    online = root / "online_orbslam3_trajectory.json"
    online.write_text(json.dumps(payload), encoding="utf-8")
    trajectory = load_s13_trajectory(
        session, trajectory_cache=None, reuse_online_trajectory=True,
        run_offline_orb=False, ignore_pose=False,
    )
    assert trajectory.audit["complete_pose_coverage"] is True
    first = session.frames[0].color_path
    first.write_bytes(first.read_bytes() + b"changed")
    try:
        load_s13_trajectory(
            session, trajectory_cache=None, reuse_online_trajectory=True,
            run_offline_orb=False, ignore_pose=False,
        )
    except ValueError as exc:
        assert "source hashes" in str(exc)
    else:
        raise AssertionError("changed online RGB source was not rejected")


def _signed_edge(left: int, right: int, value: float) -> S13MotionEdge:
    hypothesis = S13MotionHypothesis(
        0, value, 0.0, 10.0, 0.1, 0.5, 0.5, 1.0, 0.8, 0.5, 0.5,
        f"signature-{left}", "grid_lk_weighted_histogram", 0.9, ("0:0",),
    )
    return S13MotionEdge(
        left, right, 1, value, 0.0, 12.0, 20, value, 0.0, 1.0, value, "grid_lk", False, (),
        motion_hypotheses=(hypothesis,),
    )


def test_l3_writes_only_nonpanorama_completion(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("panorama_demo.video_s13_experiment.measure_s13_motion", lambda *_args, **_kwargs: ())
    report = _run(_video_session(tmp_path, frame_count=4), tmp_path / "out")
    generation = Path(report["generation"])
    assert report["render_state"] == "temporal_preview_only"
    assert report["panorama"] is None
    assert (generation / "nonpanorama/completion.json").is_file()
    assert not (generation / "P0").exists()
    completion = json.loads((generation / "nonpanorama/completion.json").read_text(encoding="utf-8"))
    assert completion["layout"]["layout_level"] == "L3_temporal_slit"
    assert completion["layout"]["spatial"] is False
    assert not (tmp_path / "out/current_preview.json").exists()


def test_reverse_segments_publish_panel_set_not_first_panel_panorama(tmp_path: Path, monkeypatch) -> None:
    values = (8.0, 8.0, 8.0, -8.0, -8.0)
    monkeypatch.setattr(
        "panorama_demo.video_s13_experiment.measure_s13_motion",
        lambda *_args, **_kwargs: tuple(_signed_edge(index, index + 1, value) for index, value in enumerate(values)),
    )
    report = _run(_video_session(tmp_path, frame_count=6), tmp_path / "out")
    p0 = Path(report["completion"]).parent
    assert report["render_state"] == "spatial_panel_set"
    assert report["panorama_claim"] == "none"
    assert report["panorama"] is None
    assert Path(report["spatial_panel_set"]).is_dir()
    assert not (p0 / "base_panorama_owner_only.png").exists()
    assert (p0 / "panel_navigation_overview_explicit_gaps.png").is_file()
    assert report["m4"]["state"] == "not_run"
    assert report["m4"]["reason"] == "spatial_panel_set_requires_independent_panels"


def test_report_separates_motion_and_render_counts(tmp_path: Path) -> None:
    report = _run(_video_session(tmp_path), tmp_path / "out")
    required = {
        "raw_rgb_motion_edge_count", "raw_rgb_adjacent_edge_count", "render_source_count",
        "render_pair_count", "lineage_switch_count", "hypothesis_count_histogram",
        "fallback_counts", "source_u_statistics",
    }
    assert required.issubset(report)
    assert set(report["fallback_counts"]) == {f"F{index}" for index in range(8)}
    assert report["render_pair_count"] == report["render_source_count"] - 1


def test_m4_appends_sealed_p1_and_preserves_immutable_p0(tmp_path: Path) -> None:
    output = tmp_path / "out"
    report = _run(_video_session(tmp_path), output)
    assert report["m4"]["state"] == "P1_sealed"
    p0_completion = Path(report["completion"])
    p1_completion = Path(report["m4"]["completion"])
    p1 = json.loads(p1_completion.read_text(encoding="utf-8"))
    assert p1["p0_parent_sha256"] == sha256_file(p0_completion)
    assert p1["models"] == ["global_scalar_dy", "pair_local_row_residual"]
    assert "translation" in p1["excluded_models"]
    assert p1["formal_raw_rgb_remap_invocations"] == report["render_source_count"]
    assert report["optimizer"]["p0_hash_unchanged_after_m4"] is True
    assert report["m4"]["latest_stage"] == "P1"
    generation_report = json.loads(
        (Path(report["generation"]) / "report.json").read_text(encoding="utf-8")
    )
    assert generation_report["optimizer_state"] != "m5_pending_after_p0"
    assert generation_report["m4"] == report["m4"]
    assert generation_report["m5"] == report["m5"]


def test_m4_level_failure_keeps_p0_and_does_not_publish_preview(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "panorama_demo.video_s13_experiment.estimate_s13_vertical",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected vertical failure")),
    )
    output = tmp_path / "out"
    report = _run(_video_session(tmp_path), output)
    assert report["m4"]["state"] == "failed_p0_preserved"
    assert verify_p0_completion(Path(report["completion"]).parent)["sealed"] is True
    assert report["optimizer"]["p0_hash_unchanged_after_m4"] is True
    assert not (output / "current_preview.json").exists()
    latest = json.loads((output / "current_latest.json").read_text(encoding="utf-8"))
    assert latest["stage"] == "P0"
    assert (Path(report["generation"]) / "M4_failure.json").is_file()


def test_m5_failure_preserves_p0_and_p1_and_does_not_publish_preview(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "out"
    output.mkdir(parents=True)
    (output / "current_preview.json").write_text(
        '{"generation_id":"stale-generation"}\n', encoding="utf-8"
    )
    hashes_before_failure: dict[str, str] = {}

    def fail_m5(*_args, **_kwargs):
        generation = next((output / "generations").iterdir())
        hashes_before_failure["p0"] = _sha256(generation / "P0/P0_completion.json")
        hashes_before_failure["p1"] = _sha256(generation / "P1/P1_completion.json")
        raise RuntimeError("injected M5 failure")

    monkeypatch.setattr(
        "panorama_demo.video_s13_experiment.run_s13_m5",
        fail_m5,
    )
    report = _run(_video_session(tmp_path), output)
    generation = Path(report["generation"])
    p0_completion = generation / "P0/P0_completion.json"
    p1_completion = generation / "P1/P1_completion.json"
    p0_hash = hashes_before_failure["p0"]
    p1_hash = hashes_before_failure["p1"]
    p1 = json.loads(p1_completion.read_text(encoding="utf-8"))

    assert report["m5"]["state"] == "failed_parent_preserved"
    assert report["m5"]["error"] == "injected M5 failure"
    assert p1["p0_parent_sha256"] == p0_hash
    assert _sha256(p0_completion) == p0_hash
    assert _sha256(p1_completion) == p1_hash
    assert verify_p0_completion(p0_completion.parent)["sealed"] is True
    legacy_preview = json.loads((output / "current_preview.json").read_text(encoding="utf-8"))
    assert legacy_preview["generation_id"] == "stale-generation"
    latest = json.loads((output / "current_latest.json").read_text(encoding="utf-8"))
    assert latest["stage"] == "P1"
    failure = json.loads((generation / "M5_failure.json").read_text(encoding="utf-8"))
    assert failure["p0_parent_preserved"] is True
    assert failure["p1_candidate_parent_preserved"] is True


def test_m5_never_reestimates_or_reselects_m4(tmp_path: Path, monkeypatch) -> None:
    import panorama_demo.video_s13_experiment as experiment

    calls = {"estimate": 0, "select": 0}
    original_estimate = experiment.estimate_s13_vertical
    original_select = experiment.select_s13_vertical_parent

    def count_estimate(*args, **kwargs):
        calls["estimate"] += 1
        return original_estimate(*args, **kwargs)

    def count_select(*args, **kwargs):
        calls["select"] += 1
        return original_select(*args, **kwargs)

    monkeypatch.setattr(experiment, "estimate_s13_vertical", count_estimate)
    monkeypatch.setattr(experiment, "select_s13_vertical_parent", count_select)
    report = _run(_video_session(tmp_path), tmp_path / "out")

    assert report["m5"]["state"] == "P2_sealed"
    assert calls == {"estimate": 1, "select": 1}
    performance = json.loads(Path(report["m5"]["performance"]).read_text(encoding="utf-8"))
    assert performance["m4_reestimated_in_m5"] is False
    assert performance["gain_enumeration_count_in_m5"] == 0
    assert performance["m4_selection_full_resolution_render_count"] == 0
    assert performance["p2_full_resolution_render_count"] == 2


def test_formal_m6_v4_writes_stage_snapshots_and_completion_timing(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    report = _run_formal_m6(_video_session(tmp_path), output)

    assert report["final_stage"] == "P3"
    assert report["counts"] == {
        "png_write_count": 4,
        "jpg_write_count": 0,
        "json_write_count": 1,
        "npz_write_count": 0,
            "sha_call_count": 0,
            "m7_call_count": 0,
            "c2e_call_count": 6,
        }
    assert {path.name for path in output.iterdir()} == {
        "P0_base_panorama.png",
        "P1_vertical_panorama.png",
        "P2_geometry_and_seam_panorama.png",
        "P3_visual_panorama.png",
        "timing.json",
    }
    assert Path(report["panorama"]).name == "P3_visual_panorama.png"
    assert all(Path(path).is_file() for path in report["stage_images"])
    timing = json.loads(Path(report["timing_json"]).read_text(encoding="utf-8"))
    assert timing["schema"] == "gemini305-video-s13-panorama-timing/v1"
    assert timing["run_id"] == report["run_id"]
    assert timing["completed"] is True
    assert timing["completion_stage"] == "P3"
    assert timing["measurement_end"] == "all_stage_png_files_closed"
    assert timing["panorama_completion_wall_seconds"] == (
        report["panorama_completion_wall_seconds"]
    )
    assert timing["panorama_completion_wall_seconds"] >= (
        timing["stage_seconds"]["total"]
    )
    assert timing["stage_images"] == [Path(path).name for path in report["stage_images"]]
    assert set(report["c2e"]["selected"]) == {"C0_keep_standard"}
