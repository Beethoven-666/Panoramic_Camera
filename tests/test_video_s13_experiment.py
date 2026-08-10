from __future__ import annotations

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.synthetic import generate_sequence
from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_s13_bundle import sha256_file, verify_p0_completion
from panorama_demo.video_s13_experiment import run_s13_experiment
from panorama_demo.video_s13_session import load_s13_session
from panorama_demo.video_s13_trajectory import load_s13_trajectory
from panorama_demo.video_trajectory_cache import session_input_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/video_candidates/s013/S013_output_first_progressive_dense_central_slit_v3.yaml"


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
