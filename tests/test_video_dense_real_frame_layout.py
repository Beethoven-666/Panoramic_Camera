from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from panorama_demo.session import CameraIntrinsics, RGBDFrame
from panorama_demo.video_dense_pose_prior import ORBPoseAnchor
from panorama_demo.video_dense_real_frame_layout import (
    DenseAdjacentFrameAudit,
    DenseRealFrameLayoutConfig,
    DenseRealFrameLayoutError,
    build_dense_real_frame_layout,
    compact_support_masks_from_projection,
    verify_dense_owner_observability,
)


def _frame(frame_id: int, timestamp_us: int) -> RGBDFrame:
    return RGBDFrame(frame_id=frame_id, color_path=Path(f"{frame_id}.png"),
                     aligned_depth_path=Path(f"{frame_id}_depth.png"),
                     depth_scale_mm_per_unit=1.0, timestamp_us=timestamp_us)


def _pose(x_mm: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = x_mm
    return pose


def _calibration() -> CameraIntrinsics:
    return CameraIntrinsics(width=8, height=6, fx=4.0, fy=4.0, cx=3.5, cy=2.5,
                            distortion=())


def test_d1_layout_uses_only_real_frames_and_audited_bracketed_priors(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = (_frame(10, 0), _frame(12, 50_000), _frame(15, 100_000),
              _frame(18, 150_000), _frame(20, 200_000))
    observed_pairs: list[tuple[int, int]] = []

    def audited(left_frame, left_prior, right_frame, right_prior, *_args):
        observed_pairs.append((left_frame.frame_id, right_frame.frame_id))
        return DenseAdjacentFrameAudit(
            left_frame.frame_id, right_frame.frame_id,
            left_prior.source_pose_origin, right_prior.source_pose_origin,
            0.2, 0.3, 64, 64,
        )

    monkeypatch.setattr("panorama_demo.video_dense_real_frame_layout._dense_audit_for_adjacent_sources", audited)
    layout = build_dense_real_frame_layout(
        frames, (ORBPoseAnchor(10, 0, _pose(0.0)), ORBPoseAnchor(20, 200_000, _pose(20.0))), _calibration()
    )

    assert [frame.frame_id for frame in layout.frames] == [10, 12, 15, 18, 20]
    assert [prior.source_pose_origin for prior in layout.priors] == [
        "direct_orb_anchor", "interpolated_se3_prior", "interpolated_se3_prior",
        "interpolated_se3_prior", "direct_orb_anchor",
    ]
    # This proves D1 audits the adjacent dense real-frame chain, never one
    # intermediate against the two enclosing anchors as a union.
    assert observed_pairs == [(10, 12), (12, 15), (15, 18), (18, 20)]
    assert [(entry["left_frame_id"], entry["right_frame_id"])
            for entry in layout.audit["adjacent_real_frame_audits"]] == observed_pairs
    assert all(entry["accepted"] for entry in layout.audit["adjacent_real_frame_audits"])
    assert layout.audit["dense_prior_coverage"] == 1.0
    assert layout.as_dict()["no_extrapolated_poses"] is True


def test_d1_uses_configured_24_fps_cadence_from_real_60_fps_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = tuple(_frame(index, round(index * 1_000_000 / 60.0)) for index in range(61))
    anchors = tuple(
        ORBPoseAnchor(index, int(frames[index].timestamp_us), _pose(float(index)))
        for index in (0, 7, 15, 22, 30, 37, 45, 52, 60)
    )
    observed_pairs: list[tuple[int, int]] = []

    def audited(left_frame, left_prior, right_frame, right_prior, *_args):
        observed_pairs.append((left_frame.frame_id, right_frame.frame_id))
        return DenseAdjacentFrameAudit(
            left_frame.frame_id, right_frame.frame_id,
            left_prior.source_pose_origin, right_prior.source_pose_origin,
            0.2, 0.2, 64, 64,
        )

    monkeypatch.setattr("panorama_demo.video_dense_real_frame_layout._dense_audit_for_adjacent_sources", audited)
    layout = build_dense_real_frame_layout(
        frames, anchors, _calibration(),
        layout_config=DenseRealFrameLayoutConfig(real_source_fps=24.0),
    )

    ids = [frame.frame_id for frame in layout.frames]
    # 24 cadence targets map to actual 60 FPS files; no image, timestamp, or
    # pose is generated.  The endpoints remain observable D1 candidates.
    assert ids == [0, 3, 5, 7, 10, 12, 15, 18, 20, 22, 25, 27, 30, 33, 35,
                   37, 40, 42, 45, 48, 50, 52, 55, 57, 60]
    assert layout.audit["real_source_fps"] == 24.0
    assert layout.audit["scan_frame_count"] == 61
    assert layout.audit["candidate_source_count"] == 25
    assert layout.audit["dense_prior_coverage"] == 1.0
    # Dense evidence stays at native capture cadence: the 24 FPS list is
    # source selection only and must not turn into 42 ms residual tests.
    assert observed_pairs == list(zip(range(60), range(1, 61)))


def test_d1_omits_bad_60fps_incident_source_but_can_use_nearby_real_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = tuple(_frame(index, round(index * 1_000_000 / 60.0)) for index in range(61))
    anchors = tuple(ORBPoseAnchor(index, int(frame.timestamp_us), _pose(float(index)))
                    for index, frame in enumerate(frames))

    def audited(left_frame, left_prior, right_frame, right_prior, *_args):
        bad = 10 in {left_frame.frame_id, right_frame.frame_id}
        return DenseAdjacentFrameAudit(
            left_frame.frame_id, right_frame.frame_id,
            left_prior.source_pose_origin, right_prior.source_pose_origin,
            2.0 if bad else 0.2, 0.2, 64, 64,
        )

    monkeypatch.setattr("panorama_demo.video_dense_real_frame_layout._dense_audit_for_adjacent_sources", audited)
    layout = build_dense_real_frame_layout(frames, anchors, _calibration())
    assert 10 not in [item.frame_id for item in layout.frames]
    assert layout.audit["dense_prior_coverage"] >= 0.95
    selected = [row for row in layout.audit["grid_source_selection"] if row["selected"]]
    assert all(row["selected_frame_id"] != 10 for row in selected)
    # The full graph retains the rejected evidence rather than concealing it.
    assert any(not row["accepted"] for row in layout.audit["adjacent_real_frame_audits"])


def test_d1_keeps_unbracketed_selected_endpoint_visible_to_fail_closed_coverage() -> None:
    frames = tuple(_frame(index, round(index * 1_000_000 / 60.0)) for index in range(10))
    # The first selected real source has no left bracket.  D1 must not remove
    # it merely to claim coverage over a more convenient subset.
    anchors = (
        ORBPoseAnchor(1, int(frames[1].timestamp_us), _pose(1.0)),
        ORBPoseAnchor(9, int(frames[9].timestamp_us), _pose(9.0)),
    )
    with pytest.raises(DenseRealFrameLayoutError, match="auditable source coverage") as raised:
        build_dense_real_frame_layout(
            frames, anchors, _calibration(),
            layout_config=DenseRealFrameLayoutConfig(real_source_fps=24.0),
        )
    diagnostics = raised.value.diagnostics
    assert diagnostics["candidate_source_frame_ids"][0] == 0
    assert diagnostics["candidate_source_frame_ids"][-1] == 9
    assert diagnostics["dense_prior_coverage"] < diagnostics["dense_prior_coverage_gate"]


def test_d1_layout_fails_closed_when_adjacent_real_evidence_cannot_be_measured(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = (_frame(10, 0), _frame(15, 100_000), _frame(20, 200_000))
    monkeypatch.setattr("panorama_demo.video_dense_real_frame_layout._dense_audit_for_adjacent_sources",
                        lambda *_args: (_ for _ in ()).throw(ValueError("bad real evidence")))
    with pytest.raises(DenseRealFrameLayoutError, match="auditable source coverage"):
        build_dense_real_frame_layout(
            frames, (ORBPoseAnchor(10, 0, _pose(0.0)), ORBPoseAnchor(20, 200_000, _pose(20.0))),
            _calibration(), layout_config=DenseRealFrameLayoutConfig(minimum_dense_prior_coverage=1.0),
        )


def test_d1_layout_rejects_an_adjacent_real_pair_above_strict_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = (_frame(10, 0), _frame(15, 100_000), _frame(20, 200_000))
    monkeypatch.setattr(
        "panorama_demo.video_dense_real_frame_layout._dense_audit_for_adjacent_sources",
        lambda left_frame, left_prior, right_frame, right_prior, *_args: DenseAdjacentFrameAudit(
            left_frame.frame_id, right_frame.frame_id,
            left_prior.source_pose_origin, right_prior.source_pose_origin,
            0.2, 1.51, 64, 64,
        ),
    )
    with pytest.raises(DenseRealFrameLayoutError, match="auditable source coverage") as raised:
        build_dense_real_frame_layout(
            frames, (ORBPoseAnchor(10, 0, _pose(0.0)), ORBPoseAnchor(20, 200_000, _pose(20.0))),
            _calibration(),
        )
    diagnostics = raised.value.diagnostics
    assert diagnostics["stage"] == "dense_evidence_grid_source_coverage"
    assert [(row["left_frame_id"], row["right_frame_id"])
            for row in diagnostics["adjacent_real_frame_audits"]] == [(10, 15), (15, 20)]


def test_d1_owner_observability_rejects_unowned_and_undercovered_compact_support() -> None:
    with pytest.raises(DenseRealFrameLayoutError, match="unowned"):
        verify_dense_owner_observability(np.array([[10, -1]], dtype=np.int32))
    owner = np.array([[10, 11]], dtype=np.int32)
    assert verify_dense_owner_observability(owner, compact_support_masks={"fan": np.ones((1, 2), bool)})[
        "compact_object_support_coverage"]["fan"] == 1.0
    with pytest.raises(DenseRealFrameLayoutError, match="shape"):
        verify_dense_owner_observability(owner, compact_support_masks={"fan": np.ones((2, 1), bool)})


def test_d1_compact_support_requires_every_v2_compact_group_from_full_support_projection() -> None:
    annotations = {
        "objects": [
            {"id": "fan_249", "measurement_group": "fan", "role": "compact_foreground_single_owner"},
            {"id": "beam_249", "measurement_group": "beam", "role": "extended_background_structure"},
        ]
    }
    payload = {"objects": [{"id": "fan_249", "measurement_group": "fan", "mask_key": "objects__consensus__fan"}]}
    masks = {"objects__consensus__fan": np.ones((2, 3), dtype=bool)}
    selected = compact_support_masks_from_projection(annotations, payload, masks)
    assert set(selected) == {"fan"}
    with pytest.raises(DenseRealFrameLayoutError, match="omitted"):
        compact_support_masks_from_projection(annotations, {"objects": []}, masks)
