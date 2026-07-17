from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

import panorama_demo.calibrated_rgb_pushbroom as pushbroom
from panorama_demo import geometry_pair_diagnostic as pair_diagnostic
import panorama_demo.stitch_sequence as sequence
from panorama_demo.calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomResult,
    render_geometry_pair_diagnostic,
)
from panorama_demo.session import RGBDSession, load_rgbd_session
from panorama_demo.synthetic import generate_sequence


def _diagnostic_input(
    tmp_path: Path,
) -> tuple[RGBDSession, list[np.ndarray], list[dict[str, object]]]:
    root = generate_sequence(
        tmp_path / "session",
        frame_count=5,
        frame_width=320,
        frame_height=200,
        step=16,
        seed=211,
    )
    # Keep a stable safe-wall fixture so the test focuses on full-chain A/B
    # plumbing rather than synthetic foreground texture selection.
    for index in range(5):
        image = np.full(
            (200, 320, 3),
            (180 + 2 * index, 182 + index, 184),
            dtype=np.uint8,
        )
        assert cv2.imwrite(str(root / "color" / f"{index:08d}.jpg"), image)
    session = load_rgbd_session(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    poses = [
        np.asarray(row["matrix_row_major"], dtype=np.float64).reshape(4, 4)
        for row in manifest["known_trajectory"]["poses"]
    ]
    motions = [{"dx": 16.0, "reliable": True} for _ in range(len(poses) - 1)]
    return session, poses, motions


def _assert_scalar_tree(value: object) -> None:
    assert not isinstance(value, np.ndarray)
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_scalar_tree(item)
    elif isinstance(value, list):
        for item in value:
            _assert_scalar_tree(item)
    else:
        assert value is None or isinstance(value, (bool, int, float, str))


def test_geometry_pair_diagnostic_renders_full_chain_baseline_then_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, poses, motions = _diagnostic_input(tmp_path)
    original = pushbroom.render_calibrated_rgb_pushbroom
    calls: list[dict[str, Any]] = []

    def traced_render(
        frames: object,
        poses_value: object,
        calibration: object,
        **kwargs: object,
    ) -> CalibratedRGBPushbroomResult:
        config = kwargs["config"]
        assert hasattr(config, "geometry_assisted_seam")
        calls.append(
            {
                "frame_count": len(frames),  # type: ignore[arg-type]
                "pose_count": len(poses_value),  # type: ignore[arg-type]
                "geometry_enabled": config.geometry_assisted_seam.enabled,
                "rgb_motions": kwargs["rgb_motions"],
                "quality_gate": kwargs["quality_gate"],
            }
        )
        return original(frames, poses_value, calibration, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(pushbroom, "render_calibrated_rgb_pushbroom", traced_render)
    result = render_geometry_pair_diagnostic(
        session.frames,
        poses,
        session.calibration,
        pair_index=1,
        rgb_motions=motions,
        quality_gate=False,
    )

    assert len(calls) == 2
    assert [call["geometry_enabled"] for call in calls] == [False, True]
    assert all(call["frame_count"] == len(session.frames) for call in calls)
    assert all(call["pose_count"] == len(poses) for call in calls)
    assert calls[0]["rgb_motions"] is calls[1]["rgb_motions"]
    assert all(call["quality_gate"] is False for call in calls)

    metadata = result.metadata
    roi = metadata["roi_global"]
    assert roi["anchor"] == "nominal_owner_boundary"
    assert roi["width"] > 0 and roi["height"] > 0
    assert result.panorama.shape == (roi["height"], roi["width"] * 2, 3)
    assert result.panorama.dtype == np.uint8
    assert metadata["panel_mapping"]["baseline"]["columns"] == [0, roi["width"]]
    assert metadata["panel_mapping"]["candidate"]["columns"] == [
        roi["width"],
        roi["width"] * 2,
    ]
    assert metadata["source_chain"] == {
        "source_count": len(session.frames),
        "pair_frame_ids": [1, 2],
        "baseline_candidate_frame_ids_equal": True,
        "baseline_geometry_assisted_enabled": False,
        "candidate_geometry_assisted_enabled": True,
    }
    assert metadata["pair_audits"]["baseline"]["triggered"] is False
    assert metadata["pair_audits"]["candidate"]["triggered"] is False
    assert (
        metadata["source_remap_gain_info"]["baseline"]
        ["full_resolution_output_remap_count"]
        == len(session.frames)
    )
    assert (
        metadata["source_remap_gain_info"]["candidate"]
        ["full_resolution_output_remap_count"]
        == len(session.frames)
    )
    _assert_scalar_tree(metadata)


def _fake_pair_render_result(
    *,
    color: tuple[int, int, int],
    geometry_enabled: bool,
) -> CalibratedRGBPushbroomResult:
    height, width = 20, 90
    reason = "accepted" if geometry_enabled else "geometry_assistance_disabled"
    pair = {
        "pair_index": 0,
        "frame_ids": [7, 8],
        "triggered": geometry_enabled,
        "corridor_x": [30, 50] if geometry_enabled else None,
        "warp_source_index": 1 if geometry_enabled else None,
        "accepted": geometry_enabled,
        "fallback": "none" if geometry_enabled else "disabled",
        "audit": {
            "reason": reason,
            "mesh_active_pixel_count": 12 if geometry_enabled else 0,
            "nested_scalar": {"p95": 0.5},
        },
    }
    metrics = {
        "source_remap_count": 2,
        "full_resolution_output_remap_count": 2,
        "analysis_preview_remap_count": 2,
        "geometry_trigger_preview_remap_count": 2,
        "geometry_flow_validation_preview_remap_count": 2 if geometry_enabled else 0,
        "exposure_gain_min": 0.9,
        "exposure_gain_max": 1.1,
        "exposure_gain_min_bgr": [0.9, 0.9, 0.9],
        "exposure_gain_max_bgr": [1.1, 1.1, 1.1],
    }
    return CalibratedRGBPushbroomResult(
        panorama=np.full((height, width, 3), color, dtype=np.uint8),
        metadata={
            "source_count": 2,
            "frame_ids": [7, 8],
            "crop": {"x": 5, "y": 3, "width": width, "height": height},
            "layout": {
                "width": 100,
                "height": 30,
                "frame_ids": [7, 8],
                "owner_boundaries_x": [41.5],
            },
            "geometry_assisted_seam": {
                "enabled": geometry_enabled,
                "pairs": [pair],
            },
            "quality_metrics": metrics,
            "color_gains": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            "color_gain_channel_order": "RGB",
        },
    )


def test_geometry_pair_diagnostic_uses_candidate_geometry_corridor_and_raw_rgb_hstack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _fake_pair_render_result(
        color=(11, 22, 33), geometry_enabled=False
    )
    candidate = _fake_pair_render_result(
        color=(101, 112, 123), geometry_enabled=True
    )

    def fake_render(*_: object, **kwargs: object) -> CalibratedRGBPushbroomResult:
        config = kwargs["config"]
        return candidate if config.geometry_assisted_seam.enabled else baseline

    monkeypatch.setattr(pushbroom, "render_calibrated_rgb_pushbroom", fake_render)
    result = render_geometry_pair_diagnostic(
        [SimpleNamespace(frame_id=7), SimpleNamespace(frame_id=8)],
        [np.eye(4), np.eye(4)],
        SimpleNamespace(),
        pair_index=0,
    )

    assert result.metadata["roi_global"] == {
        "x": 30,
        "y": 3,
        "width": 20,
        "height": 20,
        "requested_x": [30, 50],
        "anchor": "candidate_geometry_corridor",
    }
    assert result.panorama.shape == (20, 40, 3)
    # The helper itself adds no text, masks, or diagnostic colouring: panels
    # are a raw horizontal RGB concatenation of the returned renderer crops.
    assert np.all(result.panorama[:, :20] == (11, 22, 33))
    assert np.all(result.panorama[:, 20:] == (101, 112, 123))
    assert result.metadata["pair_audits"]["candidate"]["reason"] == "accepted"
    _assert_scalar_tree(result.metadata)


def test_geometry_pair_diagnostic_rejects_nonadjacent_pair_index_before_rendering() -> None:
    with pytest.raises(IndexError, match="not an adjacent pair"):
        render_geometry_pair_diagnostic(
            [SimpleNamespace(frame_id=7), SimpleNamespace(frame_id=8)],
            [np.eye(4), np.eye(4)],
            SimpleNamespace(),
            pair_index=1,
        )


def test_pair_diagnostic_parser_rejects_negative_pair_index() -> None:
    parser = pair_diagnostic._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["session", "--pair-index", "-1"])


@pytest.mark.parametrize("value", [None, -1, True, "not-an-integer"])
def test_pair_diagnostic_index_requires_nonnegative_integer(value: object) -> None:
    class _Args:
        pair_index = value

    with pytest.raises(ValueError, match="pair_index|--pair-index"):
        sequence._geometry_pair_diagnostic_index(_Args())  # type: ignore[arg-type]


def test_pair_diagnostic_refuses_open3d_or_saved_trajectory_fallback(
    tmp_path: Path,
) -> None:
    """The callback must never turn a fake/local graph into a diagnostic pose."""

    session_root = generate_sequence(
        tmp_path / "pair-session",
        frame_count=4,
        frame_width=320,
        frame_height=200,
        step=32,
        seed=71,
    )
    output = tmp_path / "pair-output"

    class _Open3DOnlyBackend:
        def __init__(self) -> None:
            manifest = json.loads(
                (session_root / "manifest.json").read_text(encoding="utf-8")
            )
            self.poses = {
                int(row["frame_id"]): np.asarray(
                    row["matrix_row_major"], dtype=np.float64
                ).reshape(4, 4)
                for row in manifest["known_trajectory"]["poses"]
            }

        def estimate_pair(self, *, reference, source, intrinsics, config):
            del intrinsics, config
            return {
                "source_to_reference": (
                    np.linalg.inv(self.poses[int(reference.frame_id)])
                    @ self.poses[int(source.frame_id)]
                ),
                "converged": True,
                "fitness": 0.99,
                "rmse_mm": 0.5,
                "information": np.eye(6, dtype=np.float64) * 100.0,
            }

        def optimize_pose_graph(
            self, *, node_ids, initial_camera_to_world, edges, config
        ):
            del node_ids, edges, config
            return tuple(initial_camera_to_world)

    called = False

    def forbidden_callback(**kwargs: object) -> object:
        nonlocal called
        del kwargs
        called = True
        raise AssertionError("pair renderer must not run without current ORB")

    args = pair_diagnostic._parser().parse_args(
        [str(session_root), "--output", str(output), "--pair-index", "1"]
    )
    with pytest.raises(RuntimeError, match="current full-scan ORB-SLAM3"):
        sequence.run(
            args,
            odometry_backend=_Open3DOnlyBackend(),
            geometry_pair_diagnostic_renderer=forbidden_callback,
        )

    assert called is False
    assert sorted(path.name for path in output.iterdir()) == ["failure.json"]


def test_pair_diagnostic_route_requires_full_current_orb_then_publishes_two_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestration route is full-chain even when its renderer is faked."""

    session_root = generate_sequence(
        tmp_path / "full-session",
        frame_count=4,
        frame_width=320,
        frame_height=200,
        step=32,
        seed=89,
    )
    output = tmp_path / "full-output"
    manifest = json.loads((session_root / "manifest.json").read_text(encoding="utf-8"))
    poses = {
        int(row["frame_id"]): np.asarray(
            row["matrix_row_major"], dtype=np.float64
        ).reshape(4, 4)
        for row in manifest["known_trajectory"]["poses"]
    }

    class _Trajectory:
        def as_dict(self, *, input_frame_count: int) -> dict[str, object]:
            return {
                "backend": "orbslam3_rgbd_wsl",
                "input_frame_count": input_frame_count,
                "tracked_frame_count": input_frame_count,
                "tracked_fraction": 1.0,
                "tracked_frame_ids": sorted(poses),
                "work_dir": "temporary",
                "settings_path": "temporary",
                "association_path": "temporary",
                "trajectory_path": "temporary",
                "stdout_path": "temporary",
                "stderr_path": "temporary",
                "command": ["orbslam3"],
            }

    class _Graph:
        def pose_for(self, frame_id: int) -> np.ndarray:
            return poses[int(frame_id)].copy()

    class _Quality:
        quality_pass = True
        failure_reasons: tuple[str, ...] = ()

        @staticmethod
        def as_dict() -> dict[str, object]:
            return {"quality_pass": True, "failure_reasons": [], "metrics": {}}

    def fake_edges(*args: object, **kwargs: object):
        del args, kwargs
        return [], []

    def fake_orb(frames, calibration, work_root, *, config):
        del calibration, work_root, config
        assert [frame.frame_id for frame in frames] == sorted(poses)
        return _Trajectory()

    def fake_optimize(*args: object, **kwargs: object) -> _Graph:
        del args, kwargs
        return _Graph()

    received: dict[str, object] = {}

    def fake_pair_renderer(**kwargs: object) -> SimpleNamespace:
        received.update(kwargs)
        return SimpleNamespace(
            panorama=np.full((20, 60, 3), 127, dtype=np.uint8),
            metadata={"renderer": "fake-geometry-pair"},
        )

    monkeypatch.setattr(sequence, "_estimate_pose_edges", fake_edges)
    monkeypatch.setattr(sequence, "run_orbslam3_rgbd", fake_orb)
    monkeypatch.setattr(sequence, "optimize_rgbd_pose_graph", fake_optimize)
    monkeypatch.setattr(
        sequence, "validate_pose_trajectory", lambda *_, **__: _Quality()
    )

    args = pair_diagnostic._parser().parse_args(
        [str(session_root), "--output", str(output), "--pair-index", "1"]
    )
    report = sequence.run(
        args, geometry_pair_diagnostic_renderer=fake_pair_renderer
    )

    assert [frame.frame_id for frame in received["render_frames"]] == sorted(poses)
    assert len(received["render_poses"]) == len(poses)
    assert received["pair_index"] == 1
    assert report["schema"] == "gemini305-geometry-pair-diagnostic/v1"
    assert report["trajectory_provenance"] == "current_orbslam3_rgbd_full_scan"
    assert report["deliverable_published"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "diagnostic_panorama.jpg",
        "diagnostic_report.json",
    ]
