from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo import foreground_deformation_diagnostic as diagnostic
import panorama_demo.stitch_sequence as sequence
from panorama_demo.calibrated_rgb_pushbroom import CalibratedRGBPushbroomResult
from panorama_demo.foreground_segments import SegmentOwnerPlan
from panorama_demo.synthetic import generate_sequence


def test_foreground_deformation_diagnostic_parser_rejects_negative_pair_index() -> None:
    with pytest.raises(SystemExit):
        diagnostic._parser().parse_args(["session", "--pair-index", "-1"])


@pytest.mark.parametrize("value", [None, -1, True, "not-an-integer"])
def test_foreground_deformation_diagnostic_index_requires_nonnegative_integer(
    value: object,
) -> None:
    class _Args:
        pair_index = value

    with pytest.raises(
        ValueError, match="foreground deformation diagnostic.*pair[-_]index"
    ):
        sequence._foreground_deformation_diagnostic_index(_Args())  # type: ignore[arg-type]


def test_pair_renderer_is_full_chain_read_only_and_publishes_scalar_ab_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The experimental renderer may observe one pair but renders all sources."""

    received: dict[str, object] = {}
    first = np.full((40, 128, 3), (20, 150, 220), dtype=np.uint8)
    second = first.copy()

    def fake_render(
        frames: object,
        poses: object,
        calibration: object,
        **kwargs: object,
    ) -> CalibratedRGBPushbroomResult:
        del calibration
        received["frame_count"] = len(frames)  # type: ignore[arg-type]
        received["pose_count"] = len(poses)  # type: ignore[arg-type]
        callback = kwargs["foreground_deformation_diagnostic_callback"]
        assert callable(callback)
        callback(
            pair_index=1,
            frame_ids=(11, 12),
            source_indices=(1, 2),
            camera_to_world=(np.eye(4), np.eye(4)),
            nominal_owner_boundary_x=164.0,
            overlap_x=(100, 228),
            first_bgr=first,
            second_bgr=second,
            first_valid=np.ones(first.shape[:2], dtype=bool),
            second_valid=np.ones(first.shape[:2], dtype=bool),
            foreground_fragments=(),
            foreground_owner_plan=SegmentOwnerPlan(
                component_owner_constraints=(),
                segments=(),
                spans=(),
                handoffs=(),
                rejected_associations=(),
                rejected_association_counts={},
                geometry_mode_counts={},
            ),
            preflight_window=(100, 228),
            geometry_triggered=False,
        )
        return CalibratedRGBPushbroomResult(
            panorama=np.full((40, 160, 3), (11, 22, 33), dtype=np.uint8),
            metadata={
                "crop": {"x": 100, "y": 0, "width": 160, "height": 40},
                "quality_metrics": {
                    "source_remap_count": 5,
                    "full_resolution_output_remap_count": 5,
                    "analysis_preview_remap_count": 5,
                },
                "layout": {
                    "width": 160,
                    "height": 40,
                    "frame_ids": [10, 11, 12, 13, 14],
                    "owner_boundaries_x": [100.0, 132.0, 164.0, 196.0],
                },
            },
        )

    monkeypatch.setattr(diagnostic, "render_calibrated_rgb_pushbroom", fake_render)
    frames = [SimpleNamespace(frame_id=10 + index) for index in range(5)]
    result = diagnostic.render_foreground_deformation_pair_diagnostic(
        frames,  # type: ignore[arg-type]
        [np.eye(4) for _ in frames],
        SimpleNamespace(),
        pair_index=1,
        experiment_config=diagnostic.ForegroundDeformationExperimentConfig(
            enabled=True
        ),
        quality_gate=False,
    )

    assert received == {"frame_count": 5, "pose_count": 5}
    assert result.panorama.shape == (40, 256, 3)
    assert np.all(result.panorama == (11, 22, 33))
    metadata = result.metadata
    assert metadata["source_chain"]["source_count"] == 5
    assert metadata["foreground_deformation"]["corridor_width_pixels"] == 128
    audit = metadata["foreground_deformation"]["foreground_deformation_audits"][0]
    assert audit["reason"] == "no_measurable_foreground_seam_residual"
    assert metadata["foreground_deformation"]["pose_rewrite_detected"] is False
    assert metadata["foreground_deformation"]["color_generation_detected"] is False
    assert metadata["foreground_deformation"]["alpha_blend_pixel_count"] == 0
    assert metadata["foreground_deformation"]["multiband_pixel_count"] == 0
    assert metadata["foreground_deformation"]["baseline_graphcut_background_retained"] is True


def test_panorama_renderer_observes_all_pairs_and_retains_baseline_without_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full experimental diagnostic remains an exact baseline when all gates reject."""

    received: dict[str, object] = {}
    first = np.full((40, 128, 3), (20, 150, 220), dtype=np.uint8)
    empty_plan = SegmentOwnerPlan(
        component_owner_constraints=(),
        segments=(),
        spans=(),
        handoffs=(),
        rejected_associations=(),
        rejected_association_counts={},
        geometry_mode_counts={},
    )

    def fake_render(
        frames: object,
        poses: object,
        calibration: object,
        **kwargs: object,
    ) -> CalibratedRGBPushbroomResult:
        del calibration
        received["frame_count"] = len(frames)  # type: ignore[arg-type]
        received["pose_count"] = len(poses)  # type: ignore[arg-type]
        received["pair_indices"] = kwargs[
            "foreground_deformation_diagnostic_pair_indices"
        ]
        callback = kwargs["foreground_deformation_diagnostic_callback"]
        assert callable(callback)
        for pair_index in (0, 1):
            callback(
                pair_index=pair_index,
                frame_ids=(10 + pair_index, 11 + pair_index),
                source_indices=(pair_index, pair_index + 1),
                camera_to_world=(np.eye(4), np.eye(4)),
                nominal_owner_boundary_x=64.0,
                overlap_x=(0, 128),
                first_bgr=first,
                second_bgr=first.copy(),
                first_valid=np.ones(first.shape[:2], dtype=bool),
                second_valid=np.ones(first.shape[:2], dtype=bool),
                foreground_fragments=(),
                foreground_owner_plan=empty_plan,
                preflight_window=(0, 128),
                geometry_triggered=False,
            )
        return CalibratedRGBPushbroomResult(
            panorama=np.full((40, 160, 3), (11, 22, 33), dtype=np.uint8),
            metadata={
                "crop": {"x": 0, "y": 0, "width": 160, "height": 40},
                "quality_metrics": {
                    "source_remap_count": 3,
                    "full_resolution_output_remap_count": 3,
                    "analysis_preview_remap_count": 3,
                },
                "layout": {
                    "width": 160,
                    "height": 40,
                    "frame_ids": [10, 11, 12],
                    "owner_boundaries_x": [64.0, 96.0],
                },
            },
        )

    monkeypatch.setattr(diagnostic, "render_calibrated_rgb_pushbroom", fake_render)
    frames = [SimpleNamespace(frame_id=10 + index) for index in range(3)]
    result = diagnostic.render_foreground_deformation_panorama_diagnostic(
        frames,  # type: ignore[arg-type]
        [np.eye(4) for _ in frames],
        SimpleNamespace(),
        experiment_config=diagnostic.ForegroundDeformationExperimentConfig(enabled=True),
        quality_gate=False,
    )

    assert received == {
        "frame_count": 3,
        "pose_count": 3,
        "pair_indices": (0, 1),
    }
    assert result.panorama.shape == (40, 160, 3)
    assert np.all(result.panorama == (11, 22, 33))
    metadata = result.metadata["foreground_deformation"]
    assert metadata["mode"] == "full_panorama_experimental_foreground_deformation"
    assert metadata["pair_count"] == 2
    assert metadata["mesh_accepted_foreground_instance_count"] == 0
    assert metadata["applied_foreground_instance_count"] == 0
    assert metadata["foreground_deformation_pixel_count"] == 0
    assert metadata["baseline_graphcut_background_retained"] is True
    assert len(metadata["foreground_deformation_audits"]) == 2


def test_panorama_collector_overlays_only_an_accepted_single_source_candidate() -> None:
    collector = diagnostic._ForegroundDeformationPanoramaCollector(
        pair_indices=(0,),
        config=diagnostic.ForegroundDeformationExperimentConfig(enabled=True),
    )
    pair = collector._collectors[0]
    pair.candidate_bgr = np.full((40, 128, 3), (77, 88, 99), dtype=np.uint8)
    pair.candidate_mask = np.zeros((40, 128), dtype=bool)
    pair.candidate_mask[10:20, 30:40] = True
    pair.metadata = {
        "pair_index": 0,
        "corridor_x": [0, 128],
        "accepted_foreground_instance_count": 1,
        "foreground_deformation_pixel_count": 100,
        "foreground_deformation_audits": [
            {"candidate": True, "accepted": True, "reason": "accepted"}
        ],
    }
    baseline = CalibratedRGBPushbroomResult(
        panorama=np.full((40, 160, 3), (11, 22, 33), dtype=np.uint8),
        metadata={"crop": {"x": 0, "y": 0, "width": 160, "height": 40}},
    )

    panorama, metadata = collector.panorama_from_baseline(baseline)

    assert np.all(panorama[10:20, 30:40] == (77, 88, 99))
    assert np.all(panorama[0:10] == (11, 22, 33))
    assert metadata["applied_foreground_instance_count"] == 1
    assert metadata["foreground_deformation_pixel_count"] == 100
    audit = metadata["foreground_deformation_audits"][0]
    assert audit["final_composite_applied"] is True
    assert audit["final_composite_reason"] == "accepted_single_source_foreground_inverse_mesh"


def test_foreground_deformation_whole_panorama_selector_requires_boolean() -> None:
    class _Args:
        whole_panorama = "yes"

    with pytest.raises(ValueError, match="whole_panorama"):
        sequence._foreground_deformation_diagnostic_whole_panorama(_Args())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("diagnostic_args", "expected_pair_index", "expected_whole_panorama", "strategy"),
    [
        (
            ["--pair-index", "1"],
            1,
            False,
            "full_sequence_foreground_deformation_pair_ab_crop",
        ),
        (
            ["--whole-panorama"],
            None,
            True,
            "full_sequence_foreground_deformation_experimental_panorama",
        ),
    ],
)
def test_diagnostic_route_requires_current_orb_and_writes_only_two_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic_args: list[str],
    expected_pair_index: int | None,
    expected_whole_panorama: bool,
    strategy: str,
) -> None:
    session_root = generate_sequence(
        tmp_path / "full-session",
        frame_count=4,
        frame_width=320,
        frame_height=200,
        step=32,
        seed=307,
    )
    output = tmp_path / "diagnostic-output"
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
        def __init__(self) -> None:
            self.node_ids = tuple(sorted(poses))
            self.camera_to_world = tuple(
                poses[frame_id].copy() for frame_id in self.node_ids
            )
            self.edges = tuple(
                SimpleNamespace(
                    reference_node_id=left,
                    source_node_id=right,
                    structurally_valid=True,
                )
                for left, right in zip(self.node_ids[:-1], self.node_ids[1:], strict=True)
            )
            self.optimized = True
            self.connected = True
            self.edge_residuals = tuple(
                {"translation_residual_mm": 0.0, "rotation_residual_deg": 0.0}
                for _ in self.edges
            )

        def pose_for(self, frame_id: int) -> np.ndarray:
            return poses[int(frame_id)].copy()

    class _Quality:
        quality_pass = True
        failure_reasons: tuple[str, ...] = ()
        thresholds = sequence.PoseQualityThresholds()
        metrics = {
            "scan_span_mm": 96.0,
            "maximum_reverse_step_mm": 0.0,
            "reverse_fraction": 0.0,
            "maximum_step_translation_mm": 32.0,
            "maximum_step_vertical_mm": 0.0,
            "maximum_step_forward_mm": 0.0,
            "maximum_total_vertical_drift_mm": 0.0,
            "maximum_total_forward_drift_mm": 0.0,
            "maximum_step_rotation_deg": 0.0,
            "maximum_total_rotation_deg": 0.0,
            "maximum_edge_translation_residual_mm": 0.0,
            "maximum_edge_rotation_residual_deg": 0.0,
        }

        @staticmethod
        def as_dict() -> dict[str, object]:
            return {
                "quality_pass": True,
                "failure_reasons": [],
                "metrics": dict(_Quality.metrics),
            }

    monkeypatch.setattr(sequence, "_estimate_pose_edges", lambda *_args, **_kwargs: ([], []))
    monkeypatch.setattr(
        sequence,
        "run_orbslam3_rgbd",
        lambda frames, *_args, **_kwargs: (
            _Trajectory()
            if [frame.frame_id for frame in frames] == sorted(poses)
            else (_ for _ in ()).throw(AssertionError("full chain was not passed to ORB"))
        ),
    )
    monkeypatch.setattr(
        sequence, "optimize_rgbd_pose_graph", lambda *_args, **_kwargs: _Graph()
    )
    monkeypatch.setattr(
        sequence, "validate_pose_trajectory", lambda *_args, **_kwargs: _Quality()
    )

    received: dict[str, object] = {}

    def fake_renderer(**kwargs: object) -> SimpleNamespace:
        received.update(kwargs)
        return SimpleNamespace(
            panorama=np.full((20, 60, 3), 127, dtype=np.uint8),
            metadata={
                "foreground_deformation_experiment": {"enabled": False},
                "foreground_deformation": {
                    "foreground_deformation_audits": [
                        {
                            "candidate": False,
                            "accepted": False,
                            "reason": "no_measurable_foreground_seam_residual",
                        }
                    ]
                },
            },
        )

    args = diagnostic._parser().parse_args(
        [str(session_root), "--output", str(output), *diagnostic_args]
    )
    report = sequence.run(
        args, foreground_deformation_diagnostic_renderer=fake_renderer
    )

    assert [frame.frame_id for frame in received["render_frames"]] == sorted(poses)
    assert len(received["render_poses"]) == len(poses)
    assert received["pair_index"] == expected_pair_index
    assert received["whole_panorama"] is expected_whole_panorama
    assert received["experiment_config"].enabled is False
    assert report["schema"] == "gemini305-foreground-deformation-diagnostic/v1"
    assert report["trajectory_provenance"] == "current_orbslam3_rgbd_full_scan"
    assert report["deliverable_published"] is False
    assert report["render_strategy"] == strategy
    assert report["foreground_deformation_audits"][0]["accepted"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "diagnostic_panorama.jpg",
        "diagnostic_report.json",
    ]
