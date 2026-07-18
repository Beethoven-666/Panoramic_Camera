from __future__ import annotations

import csv
import json
from pathlib import Path
import struct

import numpy as np
import pytest

import panorama_demo.calibrated_rgb_pushbroom as pushbroom_module
import panorama_demo.stitch_sequence as sequence
from panorama_demo.handoff_continuity import (
    ForegroundOwnerHandoffOutcome,
    build_foreground_owner_handoff_audit,
)
from panorama_demo.synthetic import generate_sequence


def _test_glb(label: str) -> bytes:
    """Return a small, structurally valid glTF 2.0 binary for staging tests."""

    document = json.dumps(
        {"asset": {"version": "2.0", "generator": label}},
        separators=(",", ":"),
    ).encode("utf-8")
    document += b" " * (-len(document) % 4)
    return (
        struct.pack("<III", 0x46546C67, 2, 20 + len(document))
        + struct.pack("<II", len(document), 0x4E4F534A)
        + document
    )


class _DeliveryTestRGBDBackend:
    name = "delivery_test_rgbd"

    def __init__(self, session: Path, *, mode: str = "ok") -> None:
        manifest = json.loads(
            (session / "manifest.json").read_text(encoding="utf-8")
        )
        self.poses = {
            int(row["frame_id"]): np.asarray(
                row["matrix_row_major"], dtype=np.float64
            ).reshape(4, 4)
            for row in manifest["known_trajectory"]["poses"]
        }
        self.mode = mode

    def estimate_pair(self, *, reference, source, intrinsics, config):
        del intrinsics, config
        if self.mode == "odometry_error":
            raise RuntimeError("forced RGB-D odometry failure")
        reference_id = int(reference.frame_id)
        source_id = int(source.frame_id)
        return {
            "source_to_reference": (
                np.linalg.inv(self.poses[reference_id]) @ self.poses[source_id]
            ),
            "converged": self.mode != "disconnected_graph",
            "fitness": 0.0 if self.mode == "poor_quality" else 0.99,
            "rmse_mm": 500.0 if self.mode == "poor_quality" else 0.5,
            "information": np.eye(6, dtype=np.float64) * 100.0,
            "backend": self.name,
        }

    def optimize_pose_graph(
        self, *, node_ids, initial_camera_to_world, edges, config
    ):
        del node_ids, edges, config
        optimized = tuple(
            np.asarray(pose).copy() for pose in initial_camera_to_world
        )
        if self.mode == "reverse_motion":
            # Keep every pose finite and rigid, but make one real scan step
            # travel backwards.  This must stay a trajectory-structure F,
            # rather than being reclassified as a publishable C-quality result.
            reverse_pose = optimized[2].copy()
            reverse_pose[0, 3] = optimized[0][0, 3]
            optimized = (*optimized[:2], reverse_pose, *optimized[3:])
        return optimized


def _make_session(tmp_path: Path, *, seed: int = 41) -> Path:
    return generate_sequence(
        tmp_path / "session",
        frame_count=6,
        frame_width=320,
        frame_height=200,
        # The formal 32 px interior owner-search corridor needs a matching
        # synthetic source overlap; the old 60 px step left only a 4 px gap.
        step=32,
        seed=seed,
    )


def _write_stale_delivery(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "panorama.jpg",
        "report.json",
        "transforms.json",
        "render_transforms.json",
        "delivery.json",
        "foreground_mask.png",
        "background_exclusion_mask.png",
        "tsdf_foreground_mask.png",
        "foreground_source_id.png",
        "tsdf_mesh.glb",
        "tsdf_mesh_viewer.html",
    ):
        (output / name).write_bytes(b"stale")


def test_failure_report_removes_stale_deliverables(tmp_path: Path) -> None:
    _write_stale_delivery(tmp_path)
    (tmp_path / "diagnostic_panorama.jpg").write_bytes(b"stale")
    (tmp_path / "diagnostic_report.json").write_bytes(b"stale")
    staging = tmp_path / ".orbslam3_rgbd"
    staging.mkdir()
    (staging / "sensitive-staged-rgb.png").write_bytes(b"stale")

    sequence._write_failure_report(
        tmp_path, tmp_path / "input", RuntimeError("bad GraphCut seam")
    )

    assert not (tmp_path / "panorama.jpg").exists()
    assert not (tmp_path / "delivery.json").exists()
    assert not (tmp_path / "diagnostic_panorama.jpg").exists()
    assert not (tmp_path / "diagnostic_report.json").exists()
    assert not staging.exists()
    for legacy_artifact in (
        "foreground_mask.png",
        "background_exclusion_mask.png",
        "tsdf_foreground_mask.png",
        "foreground_source_id.png",
        "tsdf_mesh.glb",
        "tsdf_mesh_viewer.html",
    ):
        assert not (tmp_path / legacy_artifact).exists()
    failure = json.loads((tmp_path / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert failure["message"] == "bad GraphCut seam"


def test_clear_delivery_does_not_remove_nondelivery_diagnostics(
    tmp_path: Path,
) -> None:
    diagnostics = tmp_path / "pairs" / "pair.jpg"
    diagnostics.parent.mkdir()
    diagnostics.write_bytes(b"diagnostic")
    (tmp_path / ".report.pending.json").write_bytes(b"partial")

    sequence._clear_delivery_files(tmp_path)

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
        sequence._clear_delivery_files(tmp_path)

    assert not (tmp_path / "delivery.json").exists()


def test_run_invalidates_delivery_before_configuration_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "delivery.json").write_bytes(b"stale")

    def broken_config(_path: Path | None) -> dict[str, object]:
        assert not (output / "delivery.json").exists()
        raise ValueError("broken configuration")

    monkeypatch.setattr(sequence, "load_config", broken_config)
    args = sequence._parser().parse_args(
        [str(tmp_path / "unused-session"), "--output", str(output)]
    )

    with pytest.raises(ValueError, match="broken configuration"):
        sequence.run(args)

    assert not (output / "delivery.json").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["message"] == "broken configuration"
    assert failure["deliverable_published"] is False


@pytest.mark.parametrize(
    ("capture_pass", "pose_pass", "render_pass", "message"),
    [
        (False, True, True, "input capture quality"),
        (True, False, True, "pose trajectory quality"),
        (True, True, False, "final render quality"),
    ],
)
def test_legacy_strict_quality_helper_rejects_non_strict_results(
    capture_pass: bool,
    pose_pass: bool,
    render_pass: bool,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        sequence._ensure_publishable_quality(
            {"quality_pass": capture_pass},
            {"quality_metrics": {"quality_pass": render_pass}},
            {"quality_pass": pose_pass},
        )


def test_manual_render_sources_cannot_reduce_delivery_to_one_frame() -> None:
    with pytest.raises(ValueError, match="at least two"):
        sequence._parse_frame_ids("42")


@pytest.mark.parametrize(
    "legacy_arguments",
    [
        ["--blend-mode", "feather"],
        ["--model", "old.pth"],
        ["--device", "cuda"],
        ["--inference-width", "640"],
        ["--motion-model", "translation"],
        ["--translation-anchor-y", "0.85"],
        ["--strict-unistitch"],
        ["--no-pair-previews"],
    ],
)
def test_cli_does_not_expose_legacy_sequence_algorithm_options(
    legacy_arguments: list[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        sequence._parser().parse_args(["unused-session", *legacy_arguments])


def test_manual_render_frame_ids_cannot_publish_formal_delivery(
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    output = tmp_path / "output"
    _write_stale_delivery(output)
    args = sequence._parser().parse_args(
        [
            str(session),
            "--output",
            str(output),
            "--render-frame-ids",
            "0,1",
        ]
    )

    with pytest.raises(ValueError, match="cannot publish a complete"):
        sequence.run(args, odometry_backend=_DeliveryTestRGBDBackend(session))

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()


def test_diagnostic_capture_requires_force_and_invalidates_stale_delivery(
    tmp_path: Path,
) -> None:
    session = _make_session(tmp_path)
    manifest_path = session / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "capture_mode": "diagnostic_unrestricted_auto_exposure",
            "diagnostic_only": True,
            "formal_stitch_allowed": False,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = tmp_path / "output"
    _write_stale_delivery(output)
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(RuntimeError, match="diagnostic-only"):
        sequence.run(args, odometry_backend=_DeliveryTestRGBDBackend(session))

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()


@pytest.mark.parametrize("defect", ["calibration", "aligned_depth", "depth_scale"])
def test_diagnostic_force_cannot_bypass_strict_rgbd_session_contract(
    tmp_path: Path,
    defect: str,
) -> None:
    session = _make_session(tmp_path)
    backend = _DeliveryTestRGBDBackend(session)
    if defect == "calibration":
        (session / "calibration.json").unlink()
        expected = "Missing calibration.json"
    elif defect == "aligned_depth":
        with (session / "frames.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            first_row = next(csv.DictReader(handle))
        (session / first_row["aligned_depth_path"]).unlink()
        expected = "Missing aligned depth image"
    else:
        csv_path = session / "frames.csv"
        text = csv_path.read_text(encoding="utf-8")
        text = text.replace(",1.0,", ",0.0,", 1)
        csv_path.write_text(text, encoding="utf-8")
        expected = "depth scale"
    output = tmp_path / "output"
    _write_stale_delivery(output)
    args = sequence._parser().parse_args(
        [str(session), "--output", str(output), "--diagnostic-force"]
    )

    with pytest.raises((FileNotFoundError, ValueError), match=expected):
        sequence.run(args, odometry_backend=backend)

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "diagnostic_panorama.jpg").exists()


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ("odometry", "forced RGB-D odometry failure"),
        ("graph", "disconnected"),
        ("pushbroom", "forced calibrated RGB pushbroom failure"),
    ],
)
def test_rgbd_pipeline_stage_failure_never_leaves_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    message: str,
) -> None:
    session = _make_session(tmp_path)
    output = tmp_path / "output"
    _write_stale_delivery(output)
    mode = {
        "odometry": "odometry_error",
        "graph": "disconnected_graph",
        "pushbroom": "ok",
    }[stage]
    backend = _DeliveryTestRGBDBackend(session, mode=mode)
    if stage == "pushbroom":
        def fail_pushbroom(*args, **kwargs):
            del args, kwargs
            raise RuntimeError("forced calibrated RGB pushbroom failure")

        monkeypatch.setattr(
            sequence, "render_calibrated_rgb_pushbroom", fail_pushbroom
        )
    args = sequence._parser().parse_args(
        [str(session), "--output", str(output), "--diagnostic-force"]
    )

    with pytest.raises(RuntimeError, match=message):
        sequence.run(args, odometry_backend=backend)

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    assert not (output / "report.json").exists()
    assert not (output / "diagnostic_panorama.jpg").exists()
    assert not (output / "diagnostic_report.json").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert message in failure["message"]


def _complete_anchor_render_metadata(
    *, frame_ids: list[int], quality_pass: bool = True
) -> dict[str, object]:
    return {
        "quality_metrics": {"quality_pass": quality_pass},
        "residual_alignment": {
            "working_set_audit": {
                "foreground_segment_owner_plan": {
                    "foreground_owner_continuity_summary": {
                        "backend": "foreground_segment_owner_plan_v3",
                        "track_count": 0,
                        "multi_pair_track_count": 0,
                        "owner_run_count": 0,
                        "actual_owner_switch_count": 0,
                        "minimum_feasible_owner_switch_count": 0,
                        "avoidable_owner_switch_count": 0,
                        "current_valid_nonadjacent_owner_pixel_count": 0,
                        "foreground_blend_pixel_count": 0,
                        "foreground_deformation_pixel_count": 0,
                    }
                }
            }
        },
        "pairs": [
            {
                "first_frame_id": frame_ids[index],
                "second_frame_id": frame_ids[index + 1],
                "graphcut_used": True,
                "hard_cut_row_count": 0,
                "foreground_owner_handoff_audit": {
                    "policy": "foreground_owner_only_handoff_audit_v1",
                    "handoff_count": 0,
                    "audit_complete_count": 0,
                    "owner_only_no_deformation_count": 0,
                    "structurally_safe_count": 0,
                    "hard_owner_fallback_count": 0,
                    "audits": [],
                },
                "geometry_assistance": {
                    "triggered": False,
                    "accepted": False,
                    "fallback": "not_needed",
                    "audit": {"reason": "rgb_preview_below_geometry_trigger"},
                },
            }
            for index in range(len(frame_ids) - 1)
        ],
    }


def _foreground_handoff_record(
    *,
    outcome: str,
    outgoing_source: int = 0,
    incoming_source: int = 1,
    support: int = 8,
    corridor: int = 12,
) -> dict[str, object]:
    """Build scalar-only v3 foreground evidence for publication tests."""

    return {
        "pair_index": 0,
        "track_id": "track-0",
        "outgoing_run_id": "run-0",
        "incoming_run_id": "run-1",
        "outcome": outcome,
        "current_pair_source_indices": [0, 1],
        "outgoing_source_index": outgoing_source,
        "incoming_source_index": incoming_source,
        "handoff_pixel_count": support,
        "incoming_owner_pixel_count": support,
        "other_adjacent_owner_pixel_count": 0,
        "nonadjacent_owner_pixel_count": 0,
        "invalid_owner_pixel_count": 0,
        "pair_corridor_pixel_count": corridor,
        "coverage_pixel_count": support,
        "coverage_required_pixel_count": support,
        "incoming_owner_coverage_ratio": 1.0,
        "coverage_ratio": 1.0,
        "incoming_owner_is_adjacent": True,
        "outgoing_owner_is_adjacent": True,
        "incoming_owner_ownership_complete": True,
        "foreground_owner_only": True,
        "owner_only_without_deformation": True,
        "foreground_blend_pixel_count": 0,
        "foreground_deformation_pixel_count": 0,
        "apap_authorized": False,
        "flow_authorized": False,
        "local_deformation_allowed": False,
        "prohibited_apap_or_flow_requested": False,
        "prohibited_deformation_attempted": False,
        "audit_complete": True,
        "structurally_safe": True,
        "manual_review_required": outcome
        in {"pair_local_hard_owner", "full_corridor_hard_cut"},
        "selection_reasons": ["planned_owner_run"],
        "rejection_reasons": [],
        "evidence_storage": "scalar_only",
    }


def test_foreground_handoff_serializer_satisfies_delivery_validator() -> None:
    """Keep the scalar producer and formal v3 delivery consumer in lockstep."""

    audit = build_foreground_owner_handoff_audit(
        pair_index=35,
        track_id="hose_track_35",
        outgoing_run_id="run_35",
        incoming_run_id="run_36",
        outcome=ForegroundOwnerHandoffOutcome.ADJACENT_OWNER_HANDOFF,
        outgoing_source_index=35,
        incoming_source_index=36,
        handoff_support=np.ones((2, 2), dtype=bool),
        owner_assignments=np.full((2, 2), 36, dtype=np.int16),
        pair_corridor_pixel_count=8,
        current_pair_source_indices=(35, 36),
    )
    record = audit.as_dict()
    pair = {
        "foreground_owner_handoff_audit": {
            "policy": "foreground_owner_only_handoff_audit_v1",
            "handoff_count": 1,
            "audit_complete_count": 1,
            "owner_only_no_deformation_count": 1,
            "structurally_safe_count": 1,
            "hard_owner_fallback_count": 0,
            "audits": [record],
        }
    }

    assert sequence._validate_foreground_owner_handoff_audit(
        pair,
        pair_index=35,
        frame_ids=list(range(37)),
    ) == {
        "handoff_count": 1,
        "audit_complete_count": 1,
        "owner_only_no_deformation_count": 1,
        "hard_owner_fallback_count": 0,
    }


def _set_single_track_summary(metadata: dict[str, object]) -> None:
    summary = metadata["residual_alignment"]["working_set_audit"][
        "foreground_segment_owner_plan"
    ]["foreground_owner_continuity_summary"]
    summary.update(
        {
            "track_count": 1,
            "multi_pair_track_count": 1,
            "owner_run_count": 2,
            "actual_owner_switch_count": 1,
            "minimum_feasible_owner_switch_count": 1,
        }
    )


def test_publication_assessment_marks_complete_strict_failure_as_c() -> None:
    frame_ids = [10, 11]
    metadata = _complete_anchor_render_metadata(frame_ids=frame_ids, quality_pass=False)

    assessment = sequence._assess_publication(
        {"quality_pass": False},
        {"quality_pass": False},
        metadata,
        frame_ids,
    )

    assert assessment.strict_quality_pass is False
    assert assessment.quality_grade == "C"
    assert assessment.delivery_state == "published_degraded"
    assert assessment.manual_review_required is True
    assert assessment.handoff_fallback_summary == {
        "anchor": 1,
        "apap": 0,
        "flow_mesh": 0,
        "hard_cut": 0,
    }
    assert metadata["handoff_outcomes"] == [
        {
            "pair_index": 0,
            "frame_ids": [10, 11],
            "method": "anchor",
            "structurally_safe": True,
            "audit_complete": True,
            "reason": "rgb_preview_below_geometry_trigger",
            "foreground_owner_handoff_count": 0,
            "foreground_owner_handoff_audit_complete_count": 0,
            "foreground_owner_handoff_owner_only_count": 0,
            "foreground_owner_handoff_hard_owner_fallback_count": 0,
        }
    ]


def test_publication_assessment_preserves_renderer_strict_failure_reasons() -> None:
    frame_ids = [10, 11]
    metadata = _complete_anchor_render_metadata(frame_ids=frame_ids, quality_pass=False)
    metadata["quality_metrics"] = {
        "quality_pass": False,
        "strict_failure_reasons": [
            "safe white-wall owner-boundary P95 delta E00 exceeds 2.0",
            "periodic vertical stripe energy was not reduced by 40%",
        ],
    }

    assessment = sequence._assess_publication(
        {"quality_pass": True},
        {"quality_pass": True},
        metadata,
        frame_ids,
    )

    assert assessment.quality_grade == "C"
    assert assessment.strict_failure_reasons == (
        "final render quality did not pass",
        "final render: safe white-wall owner-boundary P95 delta E00 exceeds 2.0",
        "final render: periodic vertical stripe energy was not reduced by 40%",
    )
    assert assessment.as_dict()["strict_failure_reasons"] == list(
        assessment.strict_failure_reasons
    )


def test_publication_assessment_marks_accepted_local_mesh_as_b() -> None:
    frame_ids = [10, 11]
    metadata = _complete_anchor_render_metadata(frame_ids=frame_ids)
    pair = metadata["pairs"][0]
    pair["geometry_assistance"] = {
        "triggered": True,
        "accepted": True,
        "fallback": "none",
        "audit": {"reason": "accepted"},
    }

    assessment = sequence._assess_publication(
        {"quality_pass": True},
        {"quality_pass": True},
        metadata,
        frame_ids,
    )

    assert assessment.quality_grade == "B"
    assert assessment.delivery_state == "published"
    assert assessment.manual_review_required is False
    assert assessment.handoff_fallback_summary == {
        "anchor": 0,
        "apap": 0,
        "flow_mesh": 1,
        "hard_cut": 0,
    }


def test_publication_assessment_rejects_incomplete_handoff_audit() -> None:
    with pytest.raises(RuntimeError, match="complete handoff audit"):
        sequence._assess_publication(
            {"quality_pass": True},
            {"quality_pass": True},
            {"quality_metrics": {"quality_pass": True}, "pairs": []},
            [10, 11],
        )


def test_publication_assessment_requires_foreground_owner_handoff_audit() -> None:
    metadata = _complete_anchor_render_metadata(frame_ids=[10, 11])
    del metadata["pairs"][0]["foreground_owner_handoff_audit"]

    with pytest.raises(RuntimeError, match="foreground owner-run evidence"):
        sequence._assess_publication(
            {"quality_pass": True},
            {"quality_pass": True},
            metadata,
            [10, 11],
        )


@pytest.mark.parametrize(
    "outcome",
    ["pair_local_hard_owner", "full_corridor_hard_cut"],
)
def test_publication_assessment_marks_foreground_hard_owner_handoff_as_c(
    outcome: str,
) -> None:
    metadata = _complete_anchor_render_metadata(frame_ids=[10, 11])
    _set_single_track_summary(metadata)
    support = 12 if outcome == "full_corridor_hard_cut" else 8
    record = _foreground_handoff_record(outcome=outcome, support=support)
    metadata["pairs"][0]["foreground_owner_handoff_audit"] = {
        "policy": "foreground_owner_only_handoff_audit_v1",
        "handoff_count": 1,
        "audit_complete_count": 1,
        "owner_only_no_deformation_count": 1,
        "structurally_safe_count": 1,
        "hard_owner_fallback_count": int(
            outcome in {"pair_local_hard_owner", "full_corridor_hard_cut"}
        ),
        "audits": [record],
    }

    assessment = sequence._assess_publication(
        {"quality_pass": True},
        {"quality_pass": True},
        metadata,
        [10, 11],
    )

    assert assessment.quality_grade == "C"
    assert assessment.delivery_state == "published_degraded"
    assert assessment.manual_review_required is True
    assert assessment.handoff_fallback_summary == {
        "anchor": 0,
        "apap": 0,
        "flow_mesh": 0,
        "hard_cut": 1,
    }


@pytest.mark.parametrize(
    ("outcome", "mutation", "message"),
    [
        (
            "keep_source",
            lambda record: record.update(incoming_source_index=1),
            "KEEP_SOURCE handoff changes owner",
        ),
        (
            "adjacent_owner_handoff",
            lambda record: record.update(incoming_source_index=0),
            "does not change owner",
        ),
        (
            "full_corridor_hard_cut",
            lambda record: record.update(pair_corridor_pixel_count=13),
            "lacks complete corridor support",
        ),
        (
            "pair_local_hard_owner",
            lambda record: record.update(prohibited_apap_or_flow_requested=True),
            "violates the owner-only contract",
        ),
        (
            "pair_local_hard_owner",
            lambda record: record.update(manual_review_required=False),
            "manual-review flag is inconsistent",
        ),
    ],
)
def test_publication_assessment_revalidates_foreground_handoff_semantics(
    outcome: str,
    mutation: object,
    message: str,
) -> None:
    metadata = _complete_anchor_render_metadata(frame_ids=[10, 11])
    record = _foreground_handoff_record(
        outcome=outcome,
        incoming_source=0 if outcome == "keep_source" else 1,
        support=12 if outcome == "full_corridor_hard_cut" else 8,
    )
    assert callable(mutation)
    mutation(record)
    metadata["pairs"][0]["foreground_owner_handoff_audit"] = {
        "policy": "foreground_owner_only_handoff_audit_v1",
        "handoff_count": 1,
        "audit_complete_count": 1,
        "owner_only_no_deformation_count": 1,
        "structurally_safe_count": 1,
        "hard_owner_fallback_count": int(
            outcome in {"pair_local_hard_owner", "full_corridor_hard_cut"}
        ),
        "audits": [record],
    }

    with pytest.raises(RuntimeError, match=message):
        sequence._assess_publication(
            {"quality_pass": True},
            {"quality_pass": True},
            metadata,
            [10, 11],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("avoidable_owner_switch_count", 1),
        ("current_valid_nonadjacent_owner_pixel_count", 1),
        ("foreground_blend_pixel_count", 1),
        ("foreground_deformation_pixel_count", 1),
    ],
)
def test_publication_assessment_rejects_foreground_owner_invariant_violation(
    field: str, value: int
) -> None:
    metadata = _complete_anchor_render_metadata(frame_ids=[10, 11])
    summary = metadata["residual_alignment"]["working_set_audit"][
        "foreground_segment_owner_plan"
    ]["foreground_owner_continuity_summary"]
    summary[field] = value
    if field == "avoidable_owner_switch_count":
        summary["actual_owner_switch_count"] = value
        summary["owner_run_count"] = value + summary["track_count"]
        summary["minimum_feasible_owner_switch_count"] = 0

    with pytest.raises(RuntimeError, match="Foreground owner continuity violates"):
        sequence._assess_publication(
            {"quality_pass": True},
            {"quality_pass": True},
            metadata,
            [10, 11],
        )


@pytest.mark.parametrize(
    ("track_count", "owner_run_count", "actual_switch_count"),
    [(1, 0, 0), (1, 1, 1), (0, 1, 0)],
)
def test_publication_assessment_rejects_impossible_foreground_run_counts(
    track_count: int, owner_run_count: int, actual_switch_count: int
) -> None:
    metadata = _complete_anchor_render_metadata(frame_ids=[10, 11])
    summary = metadata["residual_alignment"]["working_set_audit"][
        "foreground_segment_owner_plan"
    ]["foreground_owner_continuity_summary"]
    summary.update(
        {
            "track_count": track_count,
            "owner_run_count": owner_run_count,
            "actual_owner_switch_count": actual_switch_count,
            "minimum_feasible_owner_switch_count": 0,
        }
    )

    with pytest.raises(RuntimeError, match="inconsistent run costs"):
        sequence._assess_publication(
            {"quality_pass": True},
            {"quality_pass": True},
            metadata,
            [10, 11],
        )


def test_structurally_valid_poor_pose_quality_publishes_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C still requires the staged display-only TSDF delivery pair."""

    session = _make_session(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sequence,
        "_export_display_only_tsdf_mesh",
        lambda *args, **kwargs: (
            _test_glb("degraded-delivery-test"),
            {
                "backend": "fake_tsdf_display_only",
                "display_only": True,
                "participates_in_panorama": False,
            },
        ),
    )
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    report = sequence.run(
        args,
        odometry_backend=_DeliveryTestRGBDBackend(session, mode="poor_quality"),
    )

    delivery = json.loads((output / "delivery.json").read_text(encoding="utf-8"))
    assert (output / "panorama.jpg").is_file()
    assert not (output / "failure.json").exists()
    assert delivery["delivery_state"] == "published_degraded"
    assert delivery["quality_grade"] == "C"
    assert delivery["strict_quality_pass"] is False
    assert delivery["quality_pass"] is False
    assert delivery["manual_review_required"] is True
    assert report["strict_failure_reasons"]
    visualization = report["tsdf_visualization"]
    assert visualization == {
        "backend": "fake_tsdf_display_only",
        "status": "published",
        "required_for_delivery": True,
        "display_only": True,
        "participates_in_panorama": False,
        "mesh": "tsdf_mesh.glb",
        "viewer": "tsdf_mesh_viewer.html",
    }
    assert delivery["tsdf_visualization"] == visualization
    assert (output / "tsdf_mesh.glb").read_bytes() == _test_glb(
        "degraded-delivery-test"
    )
    assert 'src="tsdf_mesh.glb"' in (
        output / "tsdf_mesh_viewer.html"
    ).read_text(encoding="utf-8")


def test_nonempty_invalid_tsdf_glb_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The required mesh must be a GLB envelope, not merely non-empty bytes."""

    session = _make_session(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sequence,
        "_export_display_only_tsdf_mesh",
        lambda *args, **kwargs: (
            b"glTF" + b"not-a-valid-glb" * 2,
            {
                "backend": "fake_tsdf_display_only",
                "display_only": True,
                "participates_in_panorama": False,
            },
        ),
    )
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(RuntimeError, match="GLB"):
        sequence.run(
            args,
            odometry_backend=_DeliveryTestRGBDBackend(session),
        )

    assert [path.name for path in output.iterdir()] == ["failure.json"]


def test_reverse_optimized_pose_is_structural_failure_not_degraded_delivery(
    tmp_path: Path,
) -> None:
    """A finite but physically reversing real trajectory is F, never C."""

    session = _make_session(tmp_path)
    output = tmp_path / "output"
    args = sequence._parser().parse_args([str(session), "--output", str(output)])

    with pytest.raises(
        RuntimeError, match="violates formal side-scan structure"
    ):
        sequence.run(
            args,
            odometry_backend=_DeliveryTestRGBDBackend(
                session, mode="reverse_motion"
            ),
        )

    assert not (output / "delivery.json").exists()
    assert not (output / "panorama.jpg").exists()
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert failure["deliverable_published"] is False
    assert "maximum_reverse_step_mm" in failure["message"]


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("extract_pair_evidence", "forced RGB preview evidence failure"),
        ("preflight_sequence_owners", "forced RGB owner preflight failure"),
    ],
)
def test_rgb_residual_stage_failure_is_atomic_and_leaves_no_analysis_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    message: str,
) -> None:
    """Every residual/owner stage fails before any formal or diagnostic publish."""

    session = _make_session(tmp_path)
    output = tmp_path / "output"
    _write_stale_delivery(output)

    def fail_stage(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError(message)

    monkeypatch.setattr(pushbroom_module, attribute, fail_stage)
    args = sequence._parser().parse_args(
        [str(session), "--output", str(output), "--diagnostic-force"]
    )

    with pytest.raises(RuntimeError, match=message):
        sequence.run(args, odometry_backend=_DeliveryTestRGBDBackend(session))

    assert [path.name for path in output.iterdir()] == ["failure.json"]
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    assert message in failure["message"]
