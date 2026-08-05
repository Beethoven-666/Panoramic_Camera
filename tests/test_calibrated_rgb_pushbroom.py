from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import panorama_demo.calibrated_rgb_pushbroom as pushbroom_module
from panorama_demo.calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomRenderer,
    CalibratedRGBPushbroomConfig,
    GeometryAssistedSeamConfig,
    LocalGeometryContribution,
    PushbroomContribution,
    RGBMotionScaleEstimate,
    _GeometryPairPlan,
    _ForegroundAnchorHandoff,
    _ForegroundAnchorReservation,
    _SourceAnchorEvidence,
    _boundary_rgb_risk_audit,
    _append_shared_source_signed_occlusion_anchors,
    _advance_completed_foreground_anchor_locks,
    _advance_completed_foreground_owner_run_locks,
    _audit_foreground_anchor_handoff_continuity,
    _audit_foreground_owner_run_handoffs,
    _build_direct_token_identity_inputs,
    _build_foreground_owner_run_reservations,
    _build_shared_source_anchor_reservations,
    _foreground_anchor_completed_lock_mask,
    _foreground_anchor_mask_and_prior,
    _verify_foreground_anchor_handoff,
    _write_foreground_anchor_fragment,
    _geometry_trigger_from_preview,
    _held_out_strong_rgb_edge_gate,
    _rgb_transparent_or_reflection_protection_from_preview,
    _rgb_texture_consistent_log_relation,
    _lift_rgb_flow_application_and_fit_support_to_geometry_tile,
    _lift_training_rgb_flow_consistency_to_geometry_tile,
    _force_monotonic_component_owners,
    _pair_level_hard_owner_options,
    _pair_level_hard_owner_masks,
    _preflight_failure_pair_index,
    _protected_component_anchor_pixel_counts,
    _raw_pixels_to_virtual_coordinates,
    _robust_scan_axis,
    _restrict_signed_occlusion_components_to_tile_footprint,
    _resolve_pair_level_hard_owners,
    _audit_suppressed_source_frames,
    _select_pair_level_hard_owner,
    _clip_foreground_owner_handoffs_to_incoming_valid,
    _component_constraint_label_raster,
    _suppress_redundant_pose_sources_in_foreground_plan,
    _PhotometricEdge,
    _adaptive_multiband_levels,
    _apply_linear_bgr_gain,
    _blend_safe_pair_zone,
    _graphcut_monotonic_owner,
    _solve_global_linear_rgb_gains,
    _save_source_anchor_labels,
    _srgb_to_linear_bgr,
    build_calibrated_rgb_pushbroom_layout,
    estimate_rgb_motion_pixels_per_mm,
    render_calibrated_rgb_pushbroom,
)
from panorama_demo.foreground_segments import (
    DepthAnchorToken,
    ForegroundFragment,
    ForegroundOwnerRun,
    GeometryMode,
    RawFootprintSummary,
    plan_foreground_owners,
)
from panorama_demo.rgb_residual_alignment import ProtectedComponentFragment
from panorama_demo.geometry_assisted_local_warp import (
    LocalMeshInverseWarp,
    SignedOcclusionForegroundComponents,
    TileBounds,
)
from panorama_demo.handoff_continuity import ForegroundOwnerHandoffOutcome
from panorama_demo.session import CameraIntrinsics, RGBDSession, load_rgbd_session
from panorama_demo.synthetic import generate_sequence


def _make_rgb_pushbroom_input(tmp_path: Path, *, seed: int) -> tuple[RGBDSession, list[np.ndarray]]:
    root = generate_sequence(
        tmp_path / "session",
        frame_count=5,
        frame_width=320,
        frame_height=200,
        # Dense real pose nodes leave a 64--160 px calibrated seam-search
        # corridor inside the unchanged 20% central source-band limit.
        step=16,
        seed=seed,
    )
    # The general renderer fixture isolates calibrated-strip, photometric and
    # ownership contracts from the generator's deliberately dense foreground
    # clutter.  Dedicated tests below cover protected foreground components.
    for index in range(5):
        image = np.full(
            (200, 320, 3),
            (180 + 2 * index, 182 + index, 184),
            dtype=np.uint8,
        )
        assert cv2.imwrite(str(root / "color" / f"{index:08d}.jpg"), image)
    session = load_rgbd_session(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    trajectory = manifest["known_trajectory"]
    assert trajectory["transform"] == "camera_to_world"
    return (
        session,
        [
            np.asarray(row["matrix_row_major"], dtype=np.float64).reshape(4, 4)
            for row in trajectory["poses"]
        ],
    )


def _reliable_adjacent_rgb_motions(frame_count: int) -> list[dict[str, object]]:
    """Synthetic frames advance by 16 RGB pixels at the far background."""

    return [{"dx": 16.0, "reliable": True} for _ in range(frame_count - 1)]


def test_fast_hard_owner_keeps_one_calibrated_remap_and_complete_owner_map(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=83)

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
        fast_hard_owner=True,
        central_strip_output_dir=tmp_path / "strips",
        central_strip_owner_only_output_dir=tmp_path / "owner_only",
    )

    metrics = result.metadata["quality_metrics"]
    assert metrics["quality_pass"] is False
    assert metrics["source_remap_count"] == len(session.frames)
    assert metrics["analysis_preview_remap_count"] == 0
    assert metrics["strict_owner_partition"] is True
    assert result.owner_frame_id is not None
    assert result.owner_frame_id.shape == result.panorama.shape[:2]
    assert np.all(result.owner_frame_id >= 0)
    assert result.metadata["central_strip_export"]["image_count"] == len(session.frames)


def test_fast_visual_owner_can_skip_depth_when_motion_risk_is_clear(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=89)

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
        fast_visual_owner=True,
        fast_visual_use_depth=False,
    )

    assert result.metadata["backend"] == "calibrated_rgb_pushbroom_fast_visual_owner"
    assert result.metadata["depth_used_for_local_geometry"] is False
    assert result.metadata["quality_metrics"]["strict_owner_partition"] is True
    assert result.metadata["quality_metrics"]["photometric_flow_evidence_complete"] is True


def test_unlimited_pose_admission_keeps_more_than_160_real_nodes(tmp_path: Path) -> None:
    """A null total limit is not silently converted back to the old 160 cap."""

    session, _ = _make_rgb_pushbroom_input(tmp_path, seed=79)
    count = 161
    poses = []
    for index in range(count):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(index * 10)
        poses.append(pose)
    layout = build_calibrated_rgb_pushbroom_layout(
        list(range(count)),
        poses,
        session.calibration,
        RGBMotionScaleEstimate(
            pixels_per_mm=1.0,
            valid_pair_count=count - 1,
            candidate_pair_count=count - 1,
            relative_mad=0.0,
            samples=(),
        ),
        CalibratedRGBPushbroomConfig(max_pose_count=None),
    )

    assert len(layout.frame_ids) == count
    assert layout.retained_owner_source_indices == tuple(range(count))


def test_sparse_depth_anchor_hands_off_to_a_later_valid_pair() -> None:
    """An old anchor survives only where a later pair has no valid RGB."""

    token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=3,
        right_direct_component_label=5,
    )
    first_anchor = np.zeros((3, 3), dtype=bool)
    first_anchor[:2, :2] = True
    second_anchor = np.zeros((3, 3), dtype=bool)
    second_anchor[1:, 1:] = True
    first_fragment = ForegroundFragment(
        pair_index=0,
        component_label=7,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        global_bbox=(4, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        depth_anchor_local_mask=first_anchor,
        depth_anchor_token=token,
        allowed_local_owners=(1,),
        preferred_local_owner=1,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    second_fragment = ForegroundFragment(
        pair_index=1,
        component_label=13,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        global_bbox=(6, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        depth_anchor_local_mask=second_anchor,
        depth_anchor_token=token,
        allowed_local_owners=(0,),
        preferred_local_owner=0,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    reservation = _ForegroundAnchorReservation(
        span_id=4,
        segment_id=2,
        source_index=1,
        frame_id=101,
        fragment_refs=((0, 7), (1, 13)),
        fragments=(first_fragment, second_fragment),
    )
    canvas = np.zeros((8, 16, 3), dtype=np.uint8)
    valid = np.zeros((8, 16), dtype=bool)
    owner = np.full((8, 16), -1, dtype=np.int16)
    contribution = PushbroomContribution(
        source_index=1,
        frame_id=101,
        x0=3,
        rgb=np.full((8, 10, 3), (20, 40, 60), dtype=np.uint8),
        valid_mask=np.ones((8, 10), dtype=bool),
    )

    first_reserved, first_prior, first_current = _foreground_anchor_mask_and_prior(
        (reservation,), pair_index=0, left=4, right=10, height=8
    )
    assert np.any(first_reserved)
    assert set(np.unique(first_prior[first_prior >= 0])) == {1}
    assert len(first_current) == 1
    _write_foreground_anchor_fragment(
        canvas, valid, owner, contribution, first_current[0][1]
    )
    completed = [first_current[0]]

    current_reserved, prior, current = _foreground_anchor_mask_and_prior(
        (reservation,), pair_index=1, left=4, right=10, height=8
    )
    completed_reserved = _foreground_anchor_completed_lock_mask(
        completed, left=4, right=10, height=8
    )
    # Pair 1 sees source 1 as its first source, while pair 0's fragment is
    # only a completed lock rather than a second owner prior.
    assert np.any(current_reserved)
    assert np.any(completed_reserved)
    assert set(np.unique(prior[prior >= 0])) == {0}
    assert len(current) == 1
    _write_foreground_anchor_fragment(
        canvas, valid, owner, contribution, current[0][1]
    )
    completed.append(current[0])

    later_current, later_prior, later_records = _foreground_anchor_mask_and_prior(
        (reservation,), pair_index=2, left=4, right=10, height=8
    )
    no_support_retained, no_support_handoffs = _advance_completed_foreground_anchor_locks(
        completed,
        pair_index=2,
        left=4,
        right=10,
        height=8,
        current_pair_valid=np.zeros((8, 6), dtype=bool),
        protected_components=(),
        component_owner_constraints={},
    )
    assert no_support_retained == tuple(completed)
    assert no_support_handoffs == ()

    later_retained, later_handoffs = _advance_completed_foreground_anchor_locks(
        no_support_retained,
        pair_index=2,
        left=4,
        right=10,
        height=8,
        current_pair_valid=np.ones((8, 6), dtype=bool),
        protected_components=(),
        component_owner_constraints={},
    )
    assert not np.any(later_current)
    assert not np.any(later_prior >= 0)
    assert later_records == ()
    assert later_retained == ()
    assert len(later_handoffs) == 2
    assert sum(handoff.pixel_count for handoff in later_handoffs) == 8

    # The valid later pair must replace every old non-adjacent anchor with an
    # owner from its own adjacent source set.  Its hard-owner guard is covered
    # by the renderer, while this narrow unit check exercises the ownership
    # transition itself.
    later_pair_rgb = np.full((8, 6, 3), (200, 10, 10), dtype=np.uint8)
    later_pair_valid = np.ones((8, 6), dtype=bool)
    canvas[:, 4:10][later_pair_valid] = later_pair_rgb[later_pair_valid]
    valid[:, 4:10][later_pair_valid] = True
    owner[:, 4:10][later_pair_valid] = 2
    for handoff in later_handoffs:
        _verify_foreground_anchor_handoff(valid, owner, handoff)
    assert np.all(owner[2:4, 4:6] == 2)
    assert np.all(owner[3:5, 7:9] == 2)


def test_completed_anchor_handoffs_before_a_nonadjacent_guard() -> None:
    """An old local owner ID cannot retain a non-adjacent RGB source."""

    token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=3,
        right_direct_component_label=5,
    )
    old_fragment = ForegroundFragment(
        pair_index=1,
        component_label=5,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        global_bbox=(8, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        depth_anchor_local_mask=np.ones((3, 3), dtype=bool),
        depth_anchor_token=token,
        allowed_local_owners=(0,),
        preferred_local_owner=0,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    earlier_fragment = ForegroundFragment(
        pair_index=0,
        component_label=3,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        global_bbox=(8, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        depth_anchor_local_mask=np.ones((3, 3), dtype=bool),
        depth_anchor_token=token,
        allowed_local_owners=(1,),
        preferred_local_owner=1,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    reservation = _ForegroundAnchorReservation(
        span_id=0,
        segment_id=0,
        source_index=1,
        frame_id=101,
        fragment_refs=((0, 3), (1, 5)),
        fragments=(earlier_fragment, old_fragment),
    )
    current_guard = ProtectedComponentFragment(
        pair_index=2,
        global_bbox=(8, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        allowed_owners=(0, 1),
        component_label=9,
        preferred_owner=0,
    )

    # Source 1 was old pair 1's local owner 0, but it is not a source of
    # current pair 2=(2,3).  The guard must hand off all nine exact pixels
    # rather than treating both local owner IDs as equivalent.
    retained, retirements = _advance_completed_foreground_anchor_locks(
        ((reservation, old_fragment),),
        pair_index=2,
        left=8,
        right=11,
        height=8,
        current_pair_valid=np.ones((8, 3), dtype=bool),
        protected_components=(current_guard,),
        component_owner_constraints={9: 0},
    )
    assert retained == ()
    assert len(retirements) == 1
    assert retirements[0].as_dict() == {
            "pair_index": 2,
            "span_id": 0,
            "source_index": 1,
            "fragment_ref": {"pair_index": 1, "component_label": 5},
            "retired_exact_anchor_pixel_count": 9,
            "current_pair_valid_overlap_pixel_count": 9,
            "protected_component_pixel_counts": {"9": 9},
            "reason": "nonadjacent_current_pair_hard_owner_handoff",
    }
    valid = np.ones((8, 16), dtype=bool)
    owner = np.full((8, 16), 2, dtype=np.int16)
    _verify_foreground_anchor_handoff(valid, owner, retirements[0])
    continuity = _audit_foreground_anchor_handoff_continuity(
        retirements,
        pair_index=2,
        frame_ids=[100, 101, 102, 103],
        left=8,
        right=11,
        height=8,
        valid_canvas=valid,
        owner_canvas=owner,
    )
    assert continuity["policy"] == (
        "foreground_owner_only_continuity_audit_no_local_deformation"
    )
    assert continuity["handoff_count"] == 1
    assert continuity["coverage_complete_count"] == 1
    assert continuity["local_deformation_attempted"] is False
    scalar_audit = continuity["audits"][0]
    assert scalar_audit["candidate_anchor_frame_id"] == 101
    assert scalar_audit["coverage_ratio"] == pytest.approx(1.0)
    assert scalar_audit["decision"] == "apap_flow_fallback"
    assert scalar_audit["foreground_owner_only"] is True
    assert scalar_audit["local_deformation_allowed"] is False
    assert scalar_audit["local_deformation_attempted"] is False
    assert scalar_audit["decision_consumed_as"] == (
        "owner_only_guarded_current_pair_rewrite"
    )
    owner[2, 8] = 1
    with pytest.raises(RuntimeError, match="retained its non-adjacent source"):
        _verify_foreground_anchor_handoff(valid, owner, retirements[0])


def test_empty_foreground_handoff_continuity_is_a_complete_scalar_mapping() -> None:
    valid = np.zeros((4, 8), dtype=bool)
    owner = np.full((4, 8), -1, dtype=np.int16)

    continuity = _audit_foreground_anchor_handoff_continuity(
        (),
        pair_index=0,
        frame_ids=[10, 11],
        left=2,
        right=6,
        height=4,
        valid_canvas=valid,
        owner_canvas=owner,
    )

    assert continuity == {
        "policy": "foreground_owner_only_continuity_audit_no_local_deformation",
        "handoff_count": 0,
        "continuity_audit_count": 0,
        "coverage_complete_count": 0,
        "owner_only_no_deformation_count": 0,
        "local_deformation_attempted": False,
        "audits": [],
    }


def _test_raw_footprint(source_index: int) -> RawFootprintSummary:
    rows, columns = np.mgrid[0:8, 0:8]
    footprint = RawFootprintSummary.from_source_coordinates(
        source_index=source_index,
        source_size=(64, 48),
        map_x=20.0 + columns.astype(np.float64),
        map_y=10.0 + rows.astype(np.float64),
        mask=np.ones(rows.shape, dtype=bool),
    )
    assert footprint is not None
    return footprint


def _test_anchor_plan(
    *,
    pair_index: int,
    token: DepthAnchorToken,
    local_owner: int,
    path: Path,
) -> _GeometryPairPlan:
    labels = np.zeros((8, 12), dtype=np.int32)
    labels[2:6, 3:7] = 1
    _save_source_anchor_labels(path, x0=20, labels=labels)
    return _GeometryPairPlan(
        pair_index=pair_index,
        frame_ids=(100 + pair_index, 101 + pair_index),
        triggered=True,
        corridor_x=(20, 32),
        warp_source_index=None,
        accepted=False,
        fallback="hard_owner",
        protected_mask_path=None,
        active_mask_path=None,
        audit={},
        source_anchor_labels_path=path,
        source_anchor_evidence={
            1: _SourceAnchorEvidence(
                token=token,
                local_owner=local_owner,
                footprint=_test_raw_footprint(token.shared_source_index),
            )
        },
    )


def _test_anchor_plan_with_evidence(
    *,
    pair_index: int,
    labels: np.ndarray,
    source_anchor_evidence: dict[int, _SourceAnchorEvidence],
    path: Path,
) -> _GeometryPairPlan:
    _save_source_anchor_labels(path, x0=20, labels=labels)
    return _GeometryPairPlan(
        pair_index=pair_index,
        frame_ids=(100 + pair_index, 101 + pair_index),
        triggered=True,
        corridor_x=(20, 32),
        warp_source_index=None,
        accepted=False,
        fallback="hard_owner",
        protected_mask_path=None,
        active_mask_path=None,
        audit={},
        source_anchor_labels_path=path,
        source_anchor_evidence=source_anchor_evidence,
    )


def _test_protected_component(pair_index: int) -> ProtectedComponentFragment:
    return ProtectedComponentFragment(
        pair_index=pair_index,
        global_bbox=(20, 0, 12, 8),
        local_mask=np.ones((8, 12), dtype=bool),
        allowed_owners=(0, 1),
        component_label=1,
        preferred_owner=0,
    )


def test_exact_token_tiles_compile_to_a_sparse_two_pair_reservation(tmp_path: Path) -> None:
    """The bridge retains token identity from label tile to final reservation."""

    token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    plans = (
        _test_anchor_plan(
            pair_index=0,
            token=token,
            local_owner=1,
            path=tmp_path / "left-labels.npz",
        ),
        _test_anchor_plan(
            pair_index=1,
            token=token,
            local_owner=0,
            path=tmp_path / "right-labels.npz",
        ),
    )

    reservations, constraints, audit = _build_shared_source_anchor_reservations(
        plans,
        ((_test_protected_component(0),), (_test_protected_component(1),)),
        canvas_height=8,
    )

    assert len(reservations) == 1
    assert reservations[0].fragments[0].depth_anchor_token == token
    assert reservations[0].fragments[1].depth_anchor_token == token
    assert reservations[0].pixel_count == 32
    assert constraints == ({1: 1}, {1: 0})
    assert audit["selected_reservation_count"] == 1


def test_v3_direct_token_enters_owner_plan_and_reservation_run() -> None:
    """The renderer bridge reserves planner runs, not an independent token vote."""

    token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    footprint = _test_raw_footprint(1)
    first = ForegroundFragment(
        pair_index=0,
        component_label=7,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        global_bbox=(20, 1, 4, 4),
        local_mask=np.ones((4, 4), dtype=bool),
        depth_anchor_tokens=(token,),
        allowed_local_owners=(1,),
        preferred_local_owner=1,
        raw_footprints=(None, footprint),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    second = ForegroundFragment(
        pair_index=1,
        component_label=11,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        global_bbox=(20, 1, 4, 4),
        local_mask=np.ones((4, 4), dtype=bool),
        depth_anchor_tokens=(token,),
        allowed_local_owners=(0,),
        preferred_local_owner=0,
        raw_footprints=(footprint, None),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )

    plan = plan_foreground_owners(((first,), (second,)))

    assert plan.accepted
    assert len(plan.tracks) == 1
    assert len(plan.owner_runs) == 1
    reservations = _build_foreground_owner_run_reservations(
        plan,
        ((first,), (second,)),
        plan.component_owner_constraints,
        {},
    )
    assert len(reservations) == 1
    reservation = reservations[0]
    assert reservation.track_id == plan.owner_runs[0].track_id
    assert reservation.run_id == plan.owner_runs[0].run_id
    assert reservation.source_index == 1
    assert reservation.uses_sparse_anchor_mask is False


def test_redundant_pose_source_cannot_become_persistent_foreground_owner() -> None:
    token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    fragments = (
        (
            ForegroundFragment(
                pair_index=0,
                component_label=7,
                frame_ids=(100, 101),
                source_indices=(0, 1),
                global_bbox=(20, 1, 4, 4),
                local_mask=np.ones((4, 4), dtype=bool),
                depth_anchor_tokens=(token,),
                allowed_local_owners=(0, 1),
                preferred_local_owner=1,
                geometry_mode=GeometryMode.DEPTH_OBSERVED,
                bidirectional_visibility_supported=True,
            ),
        ),
        (
            ForegroundFragment(
                pair_index=1,
                component_label=11,
                frame_ids=(101, 102),
                source_indices=(1, 2),
                global_bbox=(20, 1, 4, 4),
                local_mask=np.ones((4, 4), dtype=bool),
                depth_anchor_tokens=(token,),
                allowed_local_owners=(0, 1),
                preferred_local_owner=0,
                geometry_mode=GeometryMode.DEPTH_OBSERVED,
                bidirectional_visibility_supported=True,
            ),
        ),
    )

    filtered, audit = _suppress_redundant_pose_sources_in_foreground_plan(
        fragments,
        ({"source_index": 1},),
    )

    assert filtered[0][0].allowed_local_owners == (0,)
    assert filtered[1][0].allowed_local_owners == (1,)
    assert all(
        fragment.geometry_mode is GeometryMode.IMAGE_REGION
        for pair in filtered
        for fragment in pair
    )
    assert all(
        not fragment.depth_anchor_tokens
        for pair in filtered
        for fragment in pair
    )
    assert audit["suppressed_source_indices"] == [1]
    assert audit["changed_fragment_count"] == 2
    assert audit["stripped_depth_identity_fragment_count"] == 2


def test_v3_owner_run_preflight_uses_full_guards_without_sparse_anchor_masks() -> None:
    """Multiple owner runs may compare bundled-token guards without a legacy mask."""

    token01 = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    token12 = DepthAnchorToken(
        shared_source_index=2,
        left_pair_index=1,
        right_pair_index=2,
        left_direct_component_label=11,
        right_direct_component_label=13,
    )
    first = ForegroundFragment(
        pair_index=0,
        component_label=7,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        global_bbox=(20, 1, 4, 4),
        local_mask=np.ones((4, 4), dtype=bool),
        depth_anchor_tokens=(token01,),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    middle = ForegroundFragment(
        pair_index=1,
        component_label=11,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        # This overlaps the outgoing run's guard in canvas coordinates.  It
        # is legitimate only because the plan names this exact adjacent
        # run-to-run handoff for renderer-time retirement and audit.
        global_bbox=(20, 1, 4, 4),
        local_mask=np.ones((4, 4), dtype=bool),
        depth_anchor_tokens=(token01, token12),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    last = ForegroundFragment(
        pair_index=2,
        component_label=13,
        frame_ids=(102, 103),
        source_indices=(2, 3),
        global_bbox=(32, 1, 4, 4),
        local_mask=np.ones((4, 4), dtype=bool),
        depth_anchor_tokens=(token12,),
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )

    plan = plan_foreground_owners(((first,), (middle,), (last,)))

    assert plan.accepted
    assert [run.owner_source_index for run in plan.owner_runs] == [1, 2]
    assert all(
        fragment.depth_anchor_local_mask is None
        for fragment in (first, middle, last)
    )
    reservations = _build_foreground_owner_run_reservations(
        plan,
        ((first,), (middle,), (last,)),
        plan.component_owner_constraints,
        {},
    )
    assert len(reservations) == 2


def test_v3_direct_token_bridge_exposes_identity_inputs_not_owner_votes(
    tmp_path: Path,
) -> None:
    token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    plans = (
        _test_anchor_plan(
            pair_index=0,
            token=token,
            local_owner=1,
            path=tmp_path / "v3-left-labels.npz",
        ),
        _test_anchor_plan(
            pair_index=1,
            token=token,
            local_owner=0,
            path=tmp_path / "v3-right-labels.npz",
        ),
    )

    token_sets, modes, visibility, footprints, audit = _build_direct_token_identity_inputs(
        plans,
        ((_test_protected_component(0),), (_test_protected_component(1),)),
        canvas_height=8,
        pair_windows=((20, 32), (20, 32)),
    )

    assert token_sets == {(0, 1): (token,), (1, 1): (token,)}
    assert modes == {(0, 1): GeometryMode.DEPTH_OBSERVED, (1, 1): GeometryMode.DEPTH_OBSERVED}
    assert visibility == {(0, 1): True, (1, 1): True}
    assert sorted(footprints) == [(0, 1, 1), (1, 1, 0)]
    assert audit["policy"] == "direct_tokens_as_v3_identity_candidates_no_pair_owner_voting"
    assert audit["selected_identity_token_count"] == 1
    assert "component_owner_conflicts" not in audit


def test_v3_direct_token_bundle_retains_redundant_same_peer_evidence(
    tmp_path: Path,
) -> None:
    """Multiple exact claims for one component transition form one bundle."""

    first_token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    second_token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=8,
        right_direct_component_label=12,
    )
    first_footprint = _test_raw_footprint(1)
    second_footprint = RawFootprintSummary(
        source_index=first_footprint.source_index,
        source_size=first_footprint.source_size,
        bounds_xyxy=first_footprint.bounds_xyxy,
        sample_count=first_footprint.sample_count,
        occupancy=np.roll(first_footprint.occupancy, 1, axis=1),
    )
    assert first_footprint.as_dict() == second_footprint.as_dict()
    assert not np.array_equal(first_footprint.occupancy, second_footprint.occupancy)

    labels = np.zeros((8, 12), dtype=np.int32)
    labels[2:6, 0:4] = 1
    labels[2:6, 8:12] = 2
    plans = (
        _test_anchor_plan_with_evidence(
            pair_index=0,
            labels=labels,
            path=tmp_path / "bundle-left.npz",
            source_anchor_evidence={
                1: _SourceAnchorEvidence(
                    token=first_token,
                    local_owner=1,
                    footprint=first_footprint,
                ),
                2: _SourceAnchorEvidence(
                    token=second_token,
                    local_owner=1,
                    footprint=second_footprint,
                ),
            },
        ),
        _test_anchor_plan_with_evidence(
            pair_index=1,
            labels=labels,
            path=tmp_path / "bundle-right.npz",
            source_anchor_evidence={
                1: _SourceAnchorEvidence(
                    token=first_token,
                    local_owner=0,
                    footprint=first_footprint,
                ),
                2: _SourceAnchorEvidence(
                    token=second_token,
                    local_owner=0,
                    footprint=second_footprint,
                ),
            },
        ),
    )
    protected = (
        (
            ProtectedComponentFragment(
                pair_index=0,
                global_bbox=(20, 0, 12, 8),
                local_mask=np.ones((8, 12), dtype=bool),
                allowed_owners=(0, 1),
                component_label=10,
                preferred_owner=0,
            ),
        ),
        (
            ProtectedComponentFragment(
                pair_index=1,
                global_bbox=(20, 0, 12, 8),
                local_mask=np.ones((8, 12), dtype=bool),
                allowed_owners=(0, 1),
                component_label=20,
                preferred_owner=0,
            ),
        ),
    )

    token_sets, modes, visibility, footprints, audit = _build_direct_token_identity_inputs(
        plans,
        protected,
        canvas_height=8,
        pair_windows=((20, 32), (20, 32)),
    )

    assert token_sets == {
        (0, 10): (first_token, second_token),
        (1, 20): (first_token, second_token),
    }
    assert modes == {
        (0, 10): GeometryMode.DEPTH_OBSERVED,
        (1, 20): GeometryMode.DEPTH_OBSERVED,
    }
    assert visibility == {(0, 10): True, (1, 20): True}
    assert footprints == {}
    assert audit["candidate_identity_bundle_count"] == 1
    assert audit["selected_identity_bundle_count"] == 1
    assert audit["selected_identity_token_count"] == 2
    assert audit["rejected_competing_transition_token_count"] == 0
    assert audit["raw_footprint_retained_slot_count"] == 0
    assert audit["raw_footprint_omitted_nonidentical_slot_count"] == 2


def test_v3_direct_token_competing_component_transition_rejects_all_edges(
    tmp_path: Path,
) -> None:
    """One component cannot pick between two peer components in one interval."""

    first_token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    second_token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=8,
        right_direct_component_label=12,
    )
    footprint = _test_raw_footprint(1)
    left_labels = np.zeros((8, 12), dtype=np.int32)
    left_labels[2:6, 0:4] = 1
    left_labels[2:6, 8:12] = 2
    right_labels = np.zeros((8, 12), dtype=np.int32)
    right_labels[2:6, 0:4] = 1
    right_labels[2:6, 8:12] = 2
    plans = (
        _test_anchor_plan_with_evidence(
            pair_index=0,
            labels=left_labels,
            path=tmp_path / "competing-left.npz",
            source_anchor_evidence={
                1: _SourceAnchorEvidence(
                    token=first_token,
                    local_owner=1,
                    footprint=footprint,
                ),
                2: _SourceAnchorEvidence(
                    token=second_token,
                    local_owner=1,
                    footprint=footprint,
                ),
            },
        ),
        _test_anchor_plan_with_evidence(
            pair_index=1,
            labels=right_labels,
            path=tmp_path / "competing-right.npz",
            source_anchor_evidence={
                1: _SourceAnchorEvidence(
                    token=first_token,
                    local_owner=0,
                    footprint=footprint,
                ),
                2: _SourceAnchorEvidence(
                    token=second_token,
                    local_owner=0,
                    footprint=footprint,
                ),
            },
        ),
    )
    protected = (
        (
            ProtectedComponentFragment(
                pair_index=0,
                global_bbox=(20, 0, 12, 8),
                local_mask=np.ones((8, 12), dtype=bool),
                allowed_owners=(0, 1),
                component_label=10,
                preferred_owner=0,
            ),
        ),
        (
            ProtectedComponentFragment(
                pair_index=1,
                global_bbox=(20, 0, 6, 8),
                local_mask=np.ones((8, 6), dtype=bool),
                allowed_owners=(0, 1),
                component_label=20,
                preferred_owner=0,
            ),
            ProtectedComponentFragment(
                pair_index=1,
                global_bbox=(26, 0, 6, 8),
                local_mask=np.ones((8, 6), dtype=bool),
                allowed_owners=(0, 1),
                component_label=21,
                preferred_owner=0,
            ),
        ),
    )

    token_sets, modes, visibility, footprints, audit = _build_direct_token_identity_inputs(
        plans,
        protected,
        canvas_height=8,
        pair_windows=((20, 32), (20, 32)),
    )

    assert token_sets == {}
    assert modes == {}
    assert visibility == {}
    assert footprints == {}
    assert audit["candidate_identity_bundle_count"] == 2
    assert audit["selected_identity_bundle_count"] == 0
    assert audit["rejected_competing_transition_token_count"] == 2
    assert audit["rejected_competing_transition_bundle_count"] == 2
    assert audit["rejected_competing_transition_component_interval_count"] == 1
    assert audit["rejection_reasons"]["competing_component_transition"] == 2


def test_v3_terminal_owner_run_retires_without_a_synthetic_handoff() -> None:
    """A terminal track lifecycle is not a missing run-to-run handoff."""

    fragment = ForegroundFragment(
        pair_index=0,
        component_label=5,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        global_bbox=(2, 1, 2, 2),
        local_mask=np.ones((2, 2), dtype=bool),
        allowed_local_owners=(1,),
        preferred_local_owner=1,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    run = ForegroundOwnerRun(
        run_id=8,
        track_id=3,
        fragment_refs=(fragment.reference,),
        owner_source_index=1,
        owner_frame_id=101,
        local_owners=(1,),
    )
    reservation = _ForegroundAnchorReservation(
        span_id=8,
        segment_id=3,
        source_index=1,
        frame_id=101,
        fragment_refs=(fragment.reference,),
        fragments=(fragment,),
        track_id=3,
        run_id=8,
    )

    # Pair 1 has current valid RGB support, but the sole run ends at pair 0
    # and deliberately has no successor.  It must be released to pair 1's
    # ordinary adjacent owner solve, without fabricating an incoming run/audit.
    retained, handoffs = _advance_completed_foreground_owner_run_locks(
        ((reservation, fragment),),
        owner_runs=(run,),
        pair_index=1,
        left=2,
        right=6,
        height=4,
        current_pair_valid=np.ones((4, 4), dtype=bool),
    )

    assert retained == ()
    assert handoffs == ()


def test_v3_handoff_writes_incoming_run_before_owner_verification() -> None:
    """A current-run reservation is excluded from the ordinary pair write.

    The outgoing source is already present on the canvas.  At the planned
    boundary, the ordinary pair write deliberately skips the incoming run's
    reserved guard, so checking the handoff before the explicit current-run
    writer would incorrectly still observe that outgoing source.  The writer
    must therefore run before the handoff verifier.
    """

    height, left, right = 4, 2, 6
    outgoing_fragment = ForegroundFragment(
        pair_index=0,
        component_label=5,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        global_bbox=(2, 1, 2, 2),
        local_mask=np.ones((2, 2), dtype=bool),
        allowed_local_owners=(1,),
        preferred_local_owner=1,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    incoming_fragment = ForegroundFragment(
        pair_index=1,
        component_label=6,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        global_bbox=(2, 1, 2, 2),
        local_mask=np.ones((2, 2), dtype=bool),
        allowed_local_owners=(1,),
        preferred_local_owner=1,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    outgoing_run = ForegroundOwnerRun(
        run_id=8,
        track_id=3,
        fragment_refs=(outgoing_fragment.reference,),
        owner_source_index=1,
        owner_frame_id=101,
        local_owners=(1,),
    )
    incoming_run = ForegroundOwnerRun(
        run_id=9,
        track_id=3,
        fragment_refs=(incoming_fragment.reference,),
        owner_source_index=2,
        owner_frame_id=102,
        local_owners=(1,),
    )
    outgoing_reservation = _ForegroundAnchorReservation(
        span_id=8,
        segment_id=3,
        source_index=1,
        frame_id=101,
        fragment_refs=(outgoing_fragment.reference,),
        fragments=(outgoing_fragment,),
        track_id=3,
        run_id=8,
    )
    incoming_reservation = _ForegroundAnchorReservation(
        span_id=9,
        segment_id=3,
        source_index=2,
        frame_id=102,
        fragment_refs=(incoming_fragment.reference,),
        fragments=(incoming_fragment,),
        track_id=3,
        run_id=9,
    )
    retained, handoffs = _advance_completed_foreground_owner_run_locks(
        ((outgoing_reservation, outgoing_fragment),),
        owner_runs=(outgoing_run, incoming_run),
        pair_index=1,
        left=left,
        right=right,
        height=height,
        current_pair_valid=np.ones((height, right - left), dtype=bool),
    )
    assert retained == ()
    assert len(handoffs) == 1
    handoff = handoffs[0]
    assert handoff.source_index == 1
    assert handoff.incoming_source_index == 2
    second_valid_for_handoff = np.ones((height, right - left), dtype=bool)
    second_valid_for_handoff[1, 0] = False
    clipped_handoffs, clipped_handoff_pixels = (
        _clip_foreground_owner_handoffs_to_incoming_valid(
            handoffs,
            left=left,
            right=right,
            height=height,
            pair_index=1,
            owner_valid_masks=(
                np.ones((height, right - left), dtype=bool),
                second_valid_for_handoff,
            ),
        )
    )
    assert clipped_handoff_pixels == 1
    assert clipped_handoffs[0].pixel_count == handoff.pixel_count - 1

    reserved, owner_prior, current_fragments = _foreground_anchor_mask_and_prior(
        (incoming_reservation,),
        pair_index=1,
        left=left,
        right=right,
        height=height,
    )
    assert len(current_fragments) == 1
    assert np.all(owner_prior[reserved] == 1)
    second_valid = np.ones((height, right - left), dtype=bool)
    reserved_y, reserved_x = np.argwhere(reserved)[0]
    second_valid[reserved_y, reserved_x] = False
    clipped_reserved, clipped_prior, clipped_fragments = (
        _foreground_anchor_mask_and_prior(
            (incoming_reservation,),
            pair_index=1,
            left=left,
            right=right,
            height=height,
            owner_valid_masks=(
                np.ones((height, right - left), dtype=bool),
                second_valid,
            ),
        )
    )
    assert np.count_nonzero(clipped_reserved) == np.count_nonzero(reserved) - 1
    assert clipped_prior[reserved_y, reserved_x] == -1
    assert len(clipped_fragments) == 1

    def pair_canvas() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        canvas = np.zeros((height, 8, 3), dtype=np.uint8)
        valid = np.zeros((height, 8), dtype=bool)
        owners = np.full((height, 8), -1, dtype=np.int16)
        # This is the old completed run's prior output.
        valid[1:3, 2:4] = True
        owners[1:3, 2:4] = outgoing_run.owner_source_index
        # This mirrors the renderer's generic write: a current foreground
        # reservation is intentionally excluded and will be written below.
        pair_write = np.ones((height, right - left), dtype=bool) & ~reserved
        valid[:, left:right][pair_write] = True
        owners[:, left:right][pair_write] = incoming_run.owner_source_index
        return canvas, valid, owners

    _canvas, premature_valid, premature_owners = pair_canvas()
    with pytest.raises(RuntimeError, match="retained its non-adjacent source"):
        _verify_foreground_anchor_handoff(
            premature_valid,
            premature_owners,
            handoff,
        )

    canvas, valid, owners = pair_canvas()
    incoming_contribution = PushbroomContribution(
        source_index=2,
        frame_id=102,
        x0=left,
        rgb=np.full((height, right - left, 3), 127, dtype=np.uint8),
        valid_mask=np.ones((height, right - left), dtype=bool),
    )
    reservation, fragment = current_fragments[0]
    assert reservation.source_index == incoming_run.owner_source_index
    _write_foreground_anchor_fragment(
        canvas,
        valid,
        owners,
        incoming_contribution,
        fragment,
        anchor_only=False,
    )
    _verify_foreground_anchor_handoff(valid, owners, handoff)
    assert np.all(owners[1:3, 2:4] == incoming_run.owner_source_index)


def test_v3_foreground_handoff_audit_has_required_pair_schema() -> None:
    """A planned run boundary emits only the new scalar owner-only audit."""

    fragment = ForegroundFragment(
        pair_index=1,
        component_label=5,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        global_bbox=(2, 1, 2, 2),
        local_mask=np.ones((2, 2), dtype=bool),
        allowed_local_owners=(0, 1),
        preferred_local_owner=1,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    handoff = _ForegroundAnchorHandoff(
        target_pair_index=1,
        span_id=8,
        source_index=1,
        fragment=fragment,
        track_id=3,
        outgoing_run_id=8,
        incoming_run_id=9,
        incoming_source_index=2,
        planned_outcome=ForegroundOwnerHandoffOutcome.ADJACENT_OWNER_HANDOFF,
    )
    owners = np.full((4, 8), -1, dtype=np.int16)
    owners[1:3, 2:4] = 2
    audit = _audit_foreground_owner_run_handoffs(
        (handoff,),
        planned_handoffs=((3, 8, 9, 1, 2),),
        pair_index=1,
        left=2,
        right=6,
        height=4,
        pair_valid=np.ones((4, 4), dtype=bool),
        owner_canvas=owners,
        blend_zone=np.zeros((4, 4), dtype=bool),
        deformation_zone=np.zeros((4, 4), dtype=bool),
    )

    assert audit["policy"] == "foreground_owner_only_handoff_audit_v1"
    assert audit["handoff_count"] == 1
    assert audit["audit_complete_count"] == 1
    assert audit["owner_only_no_deformation_count"] == 1
    assert audit["structurally_safe_count"] == 1
    assert audit["hard_owner_fallback_count"] == 0
    assert audit["audits"][0]["incoming_owner_pixel_count"] == 4
    assert audit["audits"][0]["nonadjacent_owner_pixel_count"] == 0
    assert audit["audits"][0]["foreground_blend_pixel_count"] == 0
    assert audit["audits"][0]["foreground_deformation_pixel_count"] == 0


def test_v3_foreground_handoff_rejects_a_planned_source_tuple_mismatch() -> None:
    fragment = ForegroundFragment(
        pair_index=1,
        component_label=5,
        frame_ids=(101, 102),
        source_indices=(1, 2),
        global_bbox=(2, 1, 2, 2),
        local_mask=np.ones((2, 2), dtype=bool),
        allowed_local_owners=(0, 1),
        preferred_local_owner=0,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    actual = _ForegroundAnchorHandoff(
        target_pair_index=1,
        span_id=8,
        source_index=2,
        fragment=fragment,
        track_id=3,
        outgoing_run_id=8,
        incoming_run_id=9,
        incoming_source_index=1,
        planned_outcome=ForegroundOwnerHandoffOutcome.ADJACENT_OWNER_HANDOFF,
    )
    owners = np.full((4, 8), -1, dtype=np.int16)
    owners[1:3, 2:4] = 1

    with pytest.raises(RuntimeError, match="source tuple disagrees with the plan"):
        _audit_foreground_owner_run_handoffs(
            (actual,),
            planned_handoffs=((3, 8, 9, 1, 2),),
            pair_index=1,
            left=2,
            right=6,
            height=4,
            pair_valid=np.ones((4, 4), dtype=bool),
            owner_canvas=owners,
            blend_zone=np.zeros((4, 4), dtype=bool),
            deformation_zone=np.zeros((4, 4), dtype=bool),
        )


def test_v3_empty_foreground_handoff_audit_has_all_zero_counts() -> None:
    owners = np.full((4, 8), -1, dtype=np.int16)

    audit = _audit_foreground_owner_run_handoffs(
        (),
        pair_index=0,
        left=2,
        right=6,
        height=4,
        pair_valid=np.zeros((4, 4), dtype=bool),
        owner_canvas=owners,
        blend_zone=np.zeros((4, 4), dtype=bool),
        deformation_zone=np.zeros((4, 4), dtype=bool),
    )

    assert audit == {
        "policy": "foreground_owner_only_handoff_audit_v1",
        "handoff_count": 0,
        "audit_complete_count": 0,
        "owner_only_no_deformation_count": 0,
        "structurally_safe_count": 0,
        "hard_owner_fallback_count": 0,
        "audits": [],
    }


def test_v3_seven_pair_owner_run_lifecycle_audits_repeated_adjacent_switches() -> None:
    """Every run boundary stays adjacent, including a zero-support boundary.

    The third run is deliberately outside every rendered pair window.  Its
    planned boundary must still emit a complete zero-support audit, while the
    remaining boundaries prove that their actual pixels were rewritten by the
    specifically planned incoming owner rather than merely either pair source.
    """

    height, left, right = 4, 2, 6
    track_id = 19
    fragments: list[ForegroundFragment] = []
    runs: list[ForegroundOwnerRun] = []
    reservations: list[_ForegroundAnchorReservation] = []
    for pair_index in range(7):
        fragment = ForegroundFragment(
            pair_index=pair_index,
            component_label=pair_index + 1,
            frame_ids=(100 + pair_index, 101 + pair_index),
            source_indices=(pair_index, pair_index + 1),
            global_bbox=(20 if pair_index == 3 else 2, 1, 2, 2),
            local_mask=np.ones((2, 2), dtype=bool),
            allowed_local_owners=(1,),
            preferred_local_owner=1,
            geometry_mode=GeometryMode.DEPTH_OBSERVED,
            bidirectional_visibility_supported=True,
        )
        run = ForegroundOwnerRun(
            run_id=100 + pair_index,
            track_id=track_id,
            fragment_refs=(fragment.reference,),
            owner_source_index=pair_index + 1,
            owner_frame_id=101 + pair_index,
            local_owners=(1,),
        )
        reservation = _ForegroundAnchorReservation(
            span_id=run.run_id,
            segment_id=track_id,
            source_index=run.owner_source_index,
            frame_id=run.owner_frame_id,
            fragment_refs=run.fragment_refs,
            fragments=(fragment,),
            track_id=track_id,
            run_id=run.run_id,
        )
        fragments.append(fragment)
        runs.append(run)
        reservations.append(reservation)

    completed: tuple[tuple[_ForegroundAnchorReservation, ForegroundFragment], ...] = (
        (reservations[0], fragments[0]),
    )
    for pair_index in range(1, 7):
        retained, handoffs = _advance_completed_foreground_owner_run_locks(
            completed,
            owner_runs=tuple(runs),
            pair_index=pair_index,
            left=left,
            right=right,
            height=height,
            current_pair_valid=np.ones((height, right - left), dtype=bool),
        )
        owners = np.full((height, 8), -1, dtype=np.int16)
        if pair_index != 4:
            owners[1:3, 2:4] = pair_index + 1
            assert len(handoffs) == 1
            assert handoffs[0].source_index == pair_index
            assert handoffs[0].incoming_source_index == pair_index + 1
        else:
            assert handoffs == ()

        audit = _audit_foreground_owner_run_handoffs(
            handoffs,
            planned_handoffs=(
                (track_id, 99 + pair_index, 100 + pair_index, pair_index, pair_index + 1),
            ),
            pair_index=pair_index,
            left=left,
            right=right,
            height=height,
            pair_valid=np.ones((height, right - left), dtype=bool),
            owner_canvas=owners,
            blend_zone=np.zeros((height, right - left), dtype=bool),
            deformation_zone=np.zeros((height, right - left), dtype=bool),
        )

        assert audit["handoff_count"] == 1
        assert audit["audit_complete_count"] == 1
        assert audit["structurally_safe_count"] == 1
        assert audit["owner_only_no_deformation_count"] == 1
        record = audit["audits"][0]
        assert record["outcome"] == "adjacent_owner_handoff"
        assert record["outgoing_source_index"] == pair_index
        assert record["incoming_source_index"] == pair_index + 1
        assert record["incoming_owner_ownership_complete"] is True
        assert record["nonadjacent_owner_pixel_count"] == 0
        assert record["foreground_blend_pixel_count"] == 0
        assert record["foreground_deformation_pixel_count"] == 0
        assert record["handoff_pixel_count"] == (0 if pair_index == 4 else 4)
        assert record["incoming_owner_pixel_count"] == (0 if pair_index == 4 else 4)

        completed = (*retained, (reservations[pair_index], fragments[pair_index]))


def test_different_source_output_overlap_rejects_every_sparse_anchor(
    tmp_path: Path,
) -> None:
    """No unrelated direct support may pick an owner at a shared output pixel."""

    first_token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=3,
        right_direct_component_label=5,
    )
    second_token = DepthAnchorToken(
        shared_source_index=2,
        left_pair_index=1,
        right_pair_index=2,
        left_direct_component_label=7,
        right_direct_component_label=11,
    )
    first = _test_anchor_plan(
        pair_index=0,
        token=first_token,
        local_owner=1,
        path=tmp_path / "first-labels.npz",
    )
    middle_base = _test_anchor_plan(
        pair_index=1,
        token=first_token,
        local_owner=0,
        path=tmp_path / "middle-labels.npz",
    )
    with np.load(middle_base.source_anchor_labels_path, allow_pickle=False) as stored:
        middle_labels = np.asarray(stored["labels"], dtype=np.int32)
    middle_labels[2:6, 8:12] = 2
    _save_source_anchor_labels(
        middle_base.source_anchor_labels_path,
        x0=20,
        labels=middle_labels,
    )
    middle = _GeometryPairPlan(
        pair_index=1,
        frame_ids=(101, 102),
        triggered=True,
        corridor_x=(20, 32),
        warp_source_index=None,
        accepted=False,
        fallback="hard_owner",
        protected_mask_path=None,
        active_mask_path=None,
        audit={},
        source_anchor_labels_path=middle_base.source_anchor_labels_path,
        source_anchor_evidence={
            1: _SourceAnchorEvidence(
                token=first_token,
                local_owner=0,
                footprint=_test_raw_footprint(1),
            ),
            2: _SourceAnchorEvidence(
                token=second_token,
                local_owner=1,
                footprint=_test_raw_footprint(2),
            ),
        },
    )
    last = _test_anchor_plan(
        pair_index=2,
        token=second_token,
        local_owner=0,
        path=tmp_path / "last-labels.npz",
    )

    reservations, constraints, audit = _build_shared_source_anchor_reservations(
        (first, middle, last),
        ((_test_protected_component(0),), (), (_test_protected_component(2),)),
        canvas_height=8,
    )

    assert reservations == ()
    assert constraints == ({}, {}, {})
    assert audit["candidate_token_count"] == 2
    assert audit["selected_reservation_count"] == 0
    assert audit["rejected_token_count_due_to_output_overlap"] == 2
    assert audit["output_anchor_overlap_conflicts"] == [
        {
            "candidate_count": 2,
            "reservation_span_ids": [0, 1],
            "source_indices": [1, 2],
            "outcome": "ambiguous_exact_anchor_output_overlap_rejected",
        }
    ]


def test_anchor_owner_vote_counts_only_pixels_inside_that_protected_component() -> None:
    """A one-pixel touch cannot vote with the claim's full outside area."""

    token = DepthAnchorToken(
        shared_source_index=1,
        left_pair_index=0,
        right_pair_index=1,
        left_direct_component_label=3,
        right_direct_component_label=5,
    )
    claim = ForegroundFragment(
        pair_index=0,
        component_label=1,
        frame_ids=(100, 101),
        source_indices=(0, 1),
        global_bbox=(0, 0, 10, 10),
        local_mask=np.ones((10, 10), dtype=bool),
        depth_anchor_local_mask=np.ones((10, 10), dtype=bool),
        depth_anchor_token=token,
        allowed_local_owners=(1,),
        preferred_local_owner=1,
        geometry_mode=GeometryMode.DEPTH_OBSERVED,
        bidirectional_visibility_supported=True,
    )
    guard = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(9, 9, 1, 1),
        local_mask=np.ones((1, 1), dtype=bool),
        allowed_owners=(0, 1),
        component_label=7,
        preferred_owner=0,
    )

    counts = _protected_component_anchor_pixel_counts(claim, (guard,))

    assert counts == {7: 1}


def test_scoped_signed_occlusion_components_ignore_full_frame_evidence_outside_tile() -> None:
    """A raw component cannot win a corridor match from an unseen image area."""

    labels = np.zeros((20, 20), dtype=np.int32)
    labels[5:15, 5:15] = 1
    components = SignedOcclusionForegroundComponents(
        labels=labels,
        component_pixel_counts={1: 100},
        signed_occlusion_pixel_count=100,
        eroded_core_pixel_count=64,
        rejected_component_count=0,
    )
    map_x, map_y = np.meshgrid(
        np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32)
    )
    tile = LocalGeometryContribution(
        source_index=1,
        frame_id=101,
        x0=0,
        source_map_x=map_x,
        source_map_y=map_y,
        valid_mask=np.ones((4, 4), dtype=bool),
    )

    scoped = _restrict_signed_occlusion_components_to_tile_footprint(components, tile)

    assert scoped.component_count == 0
    assert not np.any(scoped.labels)


def test_shared_source_anchor_reserves_only_exact_raw_signed_occlusion_intersection(
    tmp_path: Path,
) -> None:
    """A matched component pair may not widen a sparse owner claim."""

    previous_labels = np.zeros((20, 20), dtype=np.int32)
    previous_labels[2:12, 2:12] = 1
    current_labels = np.zeros((20, 20), dtype=np.int32)
    current_labels[2:12, 5:15] = 1
    previous_components = SignedOcclusionForegroundComponents(
        labels=previous_labels,
        component_pixel_counts={1: 100},
        signed_occlusion_pixel_count=100,
        eroded_core_pixel_count=64,
        rejected_component_count=0,
    )
    current_components = SignedOcclusionForegroundComponents(
        labels=current_labels,
        component_pixel_counts={1: 100},
        signed_occlusion_pixel_count=100,
        eroded_core_pixel_count=64,
        rejected_component_count=0,
    )
    map_x, map_y = np.meshgrid(
        np.arange(20, dtype=np.float32), np.arange(20, dtype=np.float32)
    )
    tile = LocalGeometryContribution(
        source_index=1,
        frame_id=101,
        x0=0,
        source_map_x=map_x,
        source_map_y=map_y,
        valid_mask=np.ones((20, 20), dtype=bool),
    )
    previous_path = tmp_path / "previous-labels.npz"
    current_path = tmp_path / "current-labels.npz"
    _save_source_anchor_labels(previous_path, x0=0, labels=np.zeros((20, 20), dtype=np.int32))
    _save_source_anchor_labels(current_path, x0=0, labels=np.zeros((20, 20), dtype=np.int32))
    previous_plan = _GeometryPairPlan(
        pair_index=0,
        frame_ids=(100, 101),
        triggered=True,
        corridor_x=(0, 20),
        warp_source_index=None,
        accepted=False,
        fallback="hard_owner",
        protected_mask_path=None,
        active_mask_path=None,
        audit={},
        source_anchor_labels_path=previous_path,
    )
    current_plan = _GeometryPairPlan(
        pair_index=1,
        frame_ids=(101, 102),
        triggered=True,
        corridor_x=(0, 20),
        warp_source_index=None,
        accepted=False,
        fallback="hard_owner",
        protected_mask_path=None,
        active_mask_path=None,
        audit={},
        source_anchor_labels_path=current_path,
    )
    calibration = CameraIntrinsics(20, 20, 10.0, 10.0, 10.0, 10.0, ())

    previous_plan, current_plan = _append_shared_source_signed_occlusion_anchors(
        previous_plan,
        current_plan,
        previous_second_components=previous_components,
        current_first_components=current_components,
        previous_second_tile=tile,
        current_first_tile=tile,
        calibration=calibration,
    )

    with np.load(previous_path, allow_pickle=False) as stored:
        previous_claim = np.asarray(stored["labels"], dtype=np.int32)
    with np.load(current_path, allow_pickle=False) as stored:
        current_claim = np.asarray(stored["labels"], dtype=np.int32)
    # The native overlap is 7 columns x 10 rows, not either 10x10 component.
    assert int(np.count_nonzero(previous_claim)) == 70
    assert int(np.count_nonzero(current_claim)) == 70
    assert len(previous_plan.source_anchor_evidence) == 1
    assert len(current_plan.source_anchor_evidence) == 1


def test_geometry_trigger_requires_local_owner_boundary_evidence() -> None:
    evidence = SimpleNamespace(
        metrics={
            "edge_normal_step_p95_pixels": 4.0,
            "edge_normal_step_max_pixels": 8.0,
            "flow_fb_error_p95_pixels": 3.0,
        }
    )
    settings = GeometryAssistedSeamConfig()
    boundary_contract = {
        "preview_risk_policy": (
            "rgb_risk_at_nominal_owner_boundary_plus_or_minus_"
            "2_full_resolution_pixels"
        ),
        "boundary_high_risk_topology_policy": (
            "3x3_closed_8_connected_raw_structural_rgb_risk_seed_component_covering_"
            "nominal_centreline_minimum_72_full_resolution_pixels_18_rows_"
            "and_26_row_span"
        ),
        "preview_risk_preview_scale": 0.75,
        "boundary_risk_pixel_count": 0,
        "boundary_risk_row_count": 0,
        "boundary_common_row_count": 40,
        "common_row_count": 40,
        "boundary_risk_component_count": 0,
        "boundary_centreline_touching_risk_component_count": 0,
        "boundary_largest_risk_component_pixel_count": 0,
        "boundary_largest_risk_component_row_count": 0,
        "boundary_largest_risk_component_row_span": 0,
        "minimum_boundary_high_risk_component_pixel_count": 41,
        "minimum_boundary_high_risk_component_row_count": 14,
        "minimum_boundary_high_risk_component_row_span": 20,
        "boundary_qualifying_risk_component_count": 0,
        "boundary_largest_qualifying_risk_component_pixel_count": 0,
        "boundary_largest_qualifying_risk_component_row_count": 0,
        "boundary_largest_qualifying_risk_component_row_span": 0,
        "boundary_high_rgb_risk": False,
        "preview_hard_cut_row_count": 0,
        "preview_full_height_hard_cut": False,
        "preview_owner_probe_status": "not_needed_no_boundary_rgb_risk",
        "preview_owner_probe_graphcut_used": False,
    }

    triggered, audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 31,
            "edge_normal_step_p95_pixels": 4.0,
            "edge_normal_step_max_pixels": 8.0,
        },
    )

    assert not triggered
    assert audit["boundary_support_sufficient"] is False
    triggered, audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 32,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 2.1,
        },
    )
    assert triggered
    assert audit["trigger_reasons"] == ["boundary_edge_normal_step_max"]

    isolated_risk_triggered, isolated_risk_audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 0,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 0.0,
            "boundary_risk_pixel_count": 3,
            "boundary_risk_row_count": 2,
            "boundary_risk_component_count": 1,
            "boundary_largest_risk_component_pixel_count": 3,
            "boundary_largest_risk_component_row_count": 2,
            "boundary_largest_risk_component_row_span": 2,
            "preview_owner_probe_status": "not_needed_unqualified_boundary_rgb_risk",
        },
    )
    assert not isolated_risk_triggered
    assert isolated_risk_audit["trigger_reasons"] == []

    risk_triggered, risk_audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 0,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 0.0,
            "boundary_risk_pixel_count": 60,
            "boundary_risk_row_count": 20,
            "boundary_risk_component_count": 1,
            "boundary_centreline_touching_risk_component_count": 1,
            "boundary_largest_risk_component_pixel_count": 60,
            "boundary_largest_risk_component_row_count": 20,
            "boundary_largest_risk_component_row_span": 20,
            "boundary_qualifying_risk_component_count": 1,
            "boundary_largest_qualifying_risk_component_pixel_count": 60,
            "boundary_largest_qualifying_risk_component_row_count": 20,
            "boundary_largest_qualifying_risk_component_row_span": 20,
            "boundary_high_rgb_risk": True,
            "preview_owner_probe_status": "graphcut",
            "preview_owner_probe_graphcut_used": True,
        },
    )
    assert risk_triggered
    assert risk_audit["trigger_reasons"] == ["boundary_high_rgb_risk"]

    hard_cut_triggered, hard_cut_audit = _geometry_trigger_from_preview(
        evidence,
        settings,
        {
            **boundary_contract,
            "observable_pixel_count": 0,
            "edge_normal_step_p95_pixels": 0.0,
            "edge_normal_step_max_pixels": 0.0,
            "boundary_risk_pixel_count": 60,
            "boundary_risk_row_count": 20,
            "boundary_risk_component_count": 1,
            "boundary_centreline_touching_risk_component_count": 1,
            "boundary_largest_risk_component_pixel_count": 60,
            "boundary_largest_risk_component_row_count": 20,
            "boundary_largest_risk_component_row_span": 20,
            "boundary_qualifying_risk_component_count": 1,
            "boundary_largest_qualifying_risk_component_pixel_count": 60,
            "boundary_largest_qualifying_risk_component_row_count": 20,
            "boundary_largest_qualifying_risk_component_row_span": 20,
            "boundary_high_rgb_risk": True,
            "preview_hard_cut_row_count": 40,
            "preview_full_height_hard_cut": True,
            "preview_owner_probe_status": "full_height_hard_cut",
        },
    )
    assert hard_cut_triggered
    assert hard_cut_audit["trigger_reasons"] == [
        "boundary_high_rgb_risk",
        "preview_full_height_hard_cut",
    ]

    with pytest.raises(RuntimeError, match="RGB-risk policy"):
        _geometry_trigger_from_preview(
            evidence,
            settings,
            {"observable_pixel_count": 32},
        )


def test_boundary_risk_trigger_requires_one_material_connected_component() -> None:
    common = np.ones((48, 17), dtype=bool)
    fragmented = np.zeros_like(common)
    fragmented[4:8, 7:11] = True
    fragmented[18:22, 7:11] = True
    fragmented_audit = _boundary_rgb_risk_audit(
        fragmented,
        common,
        nominal_boundary=8,
        half_width_pixels=2,
        preview_scale=1.0,
    )
    # The two small components have 32 pixels in total, but neither persists
    # across the required eight rows.  They stay protected by RGB ownership
    # without spending a depth-assisted local-mesh attempt.
    assert fragmented_audit["boundary_risk_pixel_count"] == 32
    assert fragmented_audit["boundary_risk_component_count"] == 2
    assert fragmented_audit["boundary_qualifying_risk_component_count"] == 0
    assert fragmented_audit["boundary_high_rgb_risk"] is False

    material = np.zeros_like(common)
    material[8:34, 7:10] = True
    material_audit = _boundary_rgb_risk_audit(
        material,
        common,
        nominal_boundary=8,
        half_width_pixels=2,
        preview_scale=1.0,
    )
    assert material_audit["boundary_risk_pixel_count"] == 78
    assert material_audit["boundary_risk_component_count"] == 1
    assert material_audit["boundary_qualifying_risk_component_count"] == 1
    assert material_audit["boundary_high_rgb_risk"] is True

    preview_material = np.zeros_like(common)
    preview_material[8:28, 7:10] = True
    preview_audit = _boundary_rgb_risk_audit(
        preview_material,
        common,
        nominal_boundary=8,
        half_width_pixels=2,
        preview_scale=0.75,
    )
    assert preview_audit["minimum_boundary_high_risk_component_pixel_count"] == 41
    assert preview_audit["minimum_boundary_high_risk_component_row_count"] == 14
    assert preview_audit["minimum_boundary_high_risk_component_row_span"] == 20
    assert preview_audit["boundary_high_rgb_risk"] is True


def test_geometry_mesh_fit_mask_uses_training_only_bidirectional_rgb_flow() -> None:
    """Held-out or locally bad flow can never enable a full-res mesh sample."""

    accepted = np.ones((4, 6), dtype=bool)
    held_out = np.zeros_like(accepted)
    held_out[1, 2] = True
    uncertain = np.zeros_like(accepted)
    uncertain[2, 1] = True
    fb = np.full(accepted.shape, 0.20, dtype=np.float64)
    fb[0, 4] = 0.90
    # 0.50 px in a half-resolution preview is a one-full-resolution-pixel
    # discrepancy.  It must not pass the formal 0.75 px support gate.
    fb[3, 2] = 0.50
    evidence = SimpleNamespace(
        accepted_mask=accepted,
        held_out_mask=held_out,
        observable_mask=np.ones_like(accepted),
        flow_uncertain_mask=uncertain,
        forward_backward_error=fb,
    )

    mask, audit = _lift_training_rgb_flow_consistency_to_geometry_tile(
        evidence,
        preview_origin_x=5.0,
        preview_scale=0.5,
        corridor_x=(10, 22),
        canvas_height=8,
        maximum_full_resolution_fb_error_pixels=0.75,
    )

    assert mask.shape == (8, 12)
    # Each preview pixel maps to its 2x2 full-resolution footprint.  The
    # held-out location, an uncertain location, and a local FB violation are
    # all excluded before mesh fitting, while ordinary training support stays.
    assert not mask[2:4, 4:6].any()
    assert not mask[4:6, 2:4].any()
    assert not mask[0:2, 8:10].any()
    assert not mask[6:8, 4:6].any()
    assert mask[0:2, 0:2].all()
    assert audit["preview_training_flow_consistent_pixel_count"] == 20
    assert audit["full_resolution_flow_consistent_pixel_count"] == int(
        np.count_nonzero(mask)
    )
    assert audit["metric_unit"] == "full_resolution_pixels"
    assert audit["maximum_full_resolution_fb_error_pixels"] == pytest.approx(0.75)
    assert audit["maximum_preview_fb_error_pixels"] == pytest.approx(0.375)


def test_geometry_flow_application_keeps_safe_rgb_holdout_for_validation() -> None:
    """Held-out RGB evidence is withheld from fitting, not from safe sampling."""

    accepted = np.ones((4, 6), dtype=bool)
    held_out = np.zeros_like(accepted)
    held_out[1, 2] = True
    evidence = SimpleNamespace(
        accepted_mask=accepted,
        held_out_mask=held_out,
        observable_mask=np.ones_like(accepted),
        flow_uncertain_mask=np.zeros_like(accepted),
        forward_backward_error=np.full(accepted.shape, 0.20, dtype=np.float64),
    )

    application, fit_support, audit = (
        _lift_rgb_flow_application_and_fit_support_to_geometry_tile(
            evidence,
            preview_origin_x=5.0,
            preview_scale=0.5,
            corridor_x=(10, 22),
            canvas_height=8,
            maximum_full_resolution_fb_error_pixels=0.75,
        )
    )

    # The held-out preview pixel expands to a 2x2 full-resolution footprint.
    # It can receive a validated mesh sample, but it cannot contribute RGB
    # evidence to fitting the mesh nodes.
    assert application[2:4, 4:6].all()
    assert not fit_support[2:4, 4:6].any()
    assert np.all(fit_support <= application)
    assert audit["preview_application_flow_consistent_pixel_count"] == 24
    assert audit["preview_training_flow_consistent_pixel_count"] == 23
    assert audit["full_resolution_application_flow_consistent_pixel_count"] == int(
        np.count_nonzero(application)
    )
    assert audit["full_resolution_flow_consistent_pixel_count"] == int(
        np.count_nonzero(fit_support)
    )


def test_geometry_rgb_transparency_protection_keeps_unsafe_edge_hard_owned() -> None:
    """Visual transparency/reflection evidence is independent of wall depth."""

    shape = (32, 64)
    accepted = np.ones(shape, dtype=bool)
    uncertain = np.zeros(shape, dtype=bool)
    occluded = np.zeros(shape, dtype=bool)
    edge = np.full(shape, np.nan, dtype=np.float64)
    # A flow-uncertain diagonal highlight, a well-tracked strong optical edge,
    # and an independently occluded point stand in for a reflective/transparent
    # foreground object.  The stable edge is deliberately accepted: stable
    # RGB flow alone cannot prove that a transparent visual layer is the wall.
    # The remote white-wall interior has no strong edge and stays blend-eligible.
    diagonal = np.arange(8, 16)
    uncertain[diagonal, diagonal] = True
    accepted[diagonal, diagonal] = False
    edge[diagonal, diagonal] = 1.0
    stable = np.arange(8, 16)
    edge[stable, 48] = 1.0
    occluded[6, 40] = True
    evidence = SimpleNamespace(
        accepted_mask=accepted,
        flow_uncertain_mask=uncertain,
        occluded_mask=occluded,
        edge_normal_step_pixels=edge,
    )

    protected, audit = _rgb_transparent_or_reflection_protection_from_preview(
        evidence,
        preview_origin_x=0.0,
        preview_scale=0.5,
        corridor_x=(0, 128),
        canvas_height=64,
        guard_radius_pixels=8,
    )

    assert protected.shape == (64, 128)
    # The nearest mapping plus 8 px guard covers the unsafe diagonal/object.
    assert protected[20, 20]
    assert protected[24, 96]
    assert protected[12, 80]
    # A sufficiently remote smooth white wall is not converted into a global
    # protection mask merely because another component was unreliable.
    assert not protected[56, 116]
    assert audit["preview_strong_rgb_structure_pixel_count"] == 16
    assert audit["preview_uncertain_or_rejected_strong_edge_pixel_count"] == 8
    assert audit["preview_unsafe_pixel_count"] == 17
    assert audit["full_resolution_protected_pixel_count"] == int(
        np.count_nonzero(protected)
    )
    assert audit["guard_radius_pixels"] == 8


def test_geometry_background_selector_cannot_bridge_a_visual_owner_guard() -> None:
    """The one permitted mesh layer is selected after visual protection."""

    # The raw depth layer is connected, but a strong transparent/reflective
    # RGB guard cuts it at the nominal owner boundary.  Neither resulting
    # island has support on both source sides, so the pair must hard-own rather
    # than silently fitting two mesh islands from the former raw component.
    visual_safe = np.ones((24, 40), dtype=bool)
    visual_safe[:, 19:21] = False
    selected, audit = pushbroom_module._select_single_virtual_background_component(
        visual_safe,
        corridor_x=(100, 140),
        nominal_boundary_x=120,
    )

    assert not selected.any()
    assert audit["depth_safe_component_count"] == 2
    assert audit["boundary_crossing_component_count"] == 0
    assert audit["selected_component_pixel_count"] == 0


def test_geometry_candidate_requires_same_held_out_strong_edge_improvement() -> None:
    held_out = np.ones((3, 3), dtype=bool)
    accepted = np.ones_like(held_out)
    before = SimpleNamespace(
        held_out_mask=held_out,
        accepted_mask=accepted,
        edge_normal_step_pixels=np.full(held_out.shape, 1.40, dtype=np.float64),
    )
    after = SimpleNamespace(
        held_out_mask=held_out.copy(),
        accepted_mask=accepted.copy(),
        edge_normal_step_pixels=np.full(held_out.shape, 0.35, dtype=np.float64),
    )

    accepted_audit = _held_out_strong_rgb_edge_gate(
        before, after, settings=GeometryAssistedSeamConfig()
    )
    assert accepted_audit["accepted"] is True
    assert accepted_audit["held_out_strong_edge_pixel_count"] == 9
    assert accepted_audit["held_out_strong_edge_improvement_ratio"] == pytest.approx(0.75)

    rejected_after = SimpleNamespace(
        held_out_mask=held_out.copy(),
        accepted_mask=accepted.copy(),
        edge_normal_step_pixels=np.full(held_out.shape, 0.90, dtype=np.float64),
    )
    rejected_audit = _held_out_strong_rgb_edge_gate(
        before, rejected_after, settings=GeometryAssistedSeamConfig()
    )
    assert rejected_audit["accepted"] is False
    assert rejected_audit["reason"] == "held_out_strong_edge_p95_exceeded"


def test_geometry_strong_edge_gate_converts_preview_error_to_full_resolution() -> None:
    held_out = np.ones((3, 3), dtype=bool)
    accepted = np.ones_like(held_out)
    before = SimpleNamespace(
        held_out_mask=held_out,
        accepted_mask=accepted,
        edge_normal_step_pixels=np.full(held_out.shape, 1.00, dtype=np.float64),
    )
    after = SimpleNamespace(
        held_out_mask=held_out.copy(),
        accepted_mask=accepted.copy(),
        edge_normal_step_pixels=np.full(held_out.shape, 0.40, dtype=np.float64),
    )

    audit = _held_out_strong_rgb_edge_gate(
        before,
        after,
        settings=GeometryAssistedSeamConfig(),
        preview_scale=0.50,
    )

    assert audit["metric_unit"] == "full_resolution_pixels"
    assert audit["held_out_strong_edge_p95_after_preview_pixels"] == pytest.approx(
        0.40
    )
    assert audit["held_out_strong_edge_p95_after_pixels"] == pytest.approx(0.80)
    assert audit["accepted"] is False
    assert audit["reason"] == "held_out_strong_edge_p95_exceeded"


def test_actual_rgb_line_gate_rejects_visible_mesh_bend_without_candidate_rgb() -> None:
    """A baseline door-frame line sees raw forward-mesh bend in output space."""

    preview = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.line(preview, (24, 3), (24, 44), (255, 255, 255), 2)
    common = np.ones(preview.shape[:2], dtype=bool)
    active = np.ones_like(common)
    bounds = TileBounds(0.0, 0.0, 63.0, 63.0)
    grid = np.asarray((0.0, 32.0, 63.0), dtype=np.float64)
    identity = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.zeros((3, 3), dtype=np.float64),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
    )
    curved = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 1.9, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
    )
    settings = GeometryAssistedSeamConfig()

    identity_audit = pushbroom_module._actual_rgb_line_straightness_gate(
        baseline_first_rgb=preview,
        baseline_second_rgb=preview,
        common_preview=common,
        active_preview=active,
        preview_left=0,
        preview_scale=0.75,
        mesh_warp=identity,
        settings=settings,
    )
    curved_audit = pushbroom_module._actual_rgb_line_straightness_gate(
        baseline_first_rgb=preview,
        baseline_second_rgb=preview,
        common_preview=common,
        active_preview=active,
        preview_left=0,
        preview_scale=0.75,
        mesh_warp=curved,
        settings=settings,
    )

    assert identity_audit["accepted"] is True
    assert identity_audit["observed"] is True
    assert identity_audit["tested_line_run_count"] >= 1
    assert curved_audit["accepted"] is False
    assert curved_audit["observed"] is True
    assert curved_audit["reason"] == "rgb_actual_line_bend_exceeded"
    assert curved_audit["maximum_line_bend_pixels"] > 1.0


def test_actual_rgb_line_gate_records_safe_wall_without_a_line_as_not_observed() -> None:
    """Owner-only structures cannot be required as a mesh-line witness."""

    preview = np.full((48, 48, 3), 128, dtype=np.uint8)
    bounds = TileBounds(0.0, 0.0, 63.0, 63.0)
    grid = np.asarray((0.0, 32.0, 63.0), dtype=np.float64)
    warp = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
    )

    audit = pushbroom_module._actual_rgb_line_straightness_gate(
        baseline_first_rgb=preview,
        baseline_second_rgb=preview,
        common_preview=np.ones(preview.shape[:2], dtype=bool),
        active_preview=np.ones(preview.shape[:2], dtype=bool),
        preview_left=0,
        preview_scale=0.75,
        mesh_warp=warp,
        settings=GeometryAssistedSeamConfig(),
    )

    assert audit["accepted"] is True
    assert audit["observed"] is False
    assert audit["reason"] == "not_observed_no_solver_valid_line"
    assert audit["tested_line_run_count"] == 0
    assert audit["maximum_line_bend_pixels"] is None


def test_geometry_active_audit_tracks_only_actual_safe_nonidentity_pixels() -> None:
    """A nominally active cell cannot make a guarded pixel appear mesh-active."""

    bounds = TileBounds(0.0, 0.0, 63.0, 63.0)
    grid = np.asarray((0.0, 32.0, 63.0), dtype=np.float64)
    same_layer = np.ones((64, 64), dtype=bool)
    same_layer[30:35, 30:35] = False
    warp = LocalMeshInverseWarp(
        bounds=bounds,
        grid_x=grid,
        grid_y=grid,
        inverse_dx=np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.8, 0.0), (0.0, 0.0, 0.0)),
            dtype=np.float64,
        ),
        inverse_dy=np.zeros((3, 3), dtype=np.float64),
        active_cells=np.ones((2, 2), dtype=bool),
        same_layer_mask=same_layer,
        same_layer_origin_xy=(0.0, 0.0),
    )

    active = pushbroom_module._mesh_active_mask(
        warp, x0=0, x1=64, height=64
    )

    assert not active[32, 32]
    assert active[48, 48]


def test_geometry_topology_fallback_uses_exactly_one_fully_covering_source() -> None:
    first = np.ones((3, 4), dtype=bool)
    second = np.ones((3, 4), dtype=bool)
    second[0, 0] = False
    assert _select_pair_level_hard_owner(first, second) == 0

    first[1, 1] = False
    assert _select_pair_level_hard_owner(first, second) is None
    assert _preflight_failure_pair_index(
        "pair_local_component_owners_do_not_admit_a_monotonic_seam:17"
    ) == 17
    assert _preflight_failure_pair_index("unrelated_failure") is None


def test_geometry_topology_fallback_preserves_shared_sources_across_a_run() -> None:
    first = np.ones((3, 4), dtype=bool)
    second = np.ones((3, 4), dtype=bool)
    assert _pair_level_hard_owner_options(first, second) == (0, 1)

    resolved = _resolve_pair_level_hard_owners(
        (17, 18, 19, 42),
        {
            17: (0, 1),
            18: (1,),
            19: (0, 1),
            42: (0, 1),
        },
    )

    # The run prefers its second RGB source to retain every shared strip;
    # the isolated corridor prefers its first source.
    assert resolved == {17: 1, 18: 1, 19: 1, 42: 0}

    locked = _resolve_pair_level_hard_owners(
        (17, 18, 19, 42),
        {
            17: (0, 1),
            18: (0, 1),
            19: (0, 1),
            42: (0, 1),
        },
        required_owners={18: 0, 42: 1},
    )
    assert locked == {17: 1, 18: 0, 19: 1, 42: 1}


def test_geometry_component_fallback_preserves_safe_pair_coverage() -> None:
    left = ProtectedComponentFragment(
        pair_index=4,
        global_bbox=(10, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        allowed_owners=(0, 1),
        component_label=1,
        preferred_owner=1,
    )
    right = ProtectedComponentFragment(
        pair_index=4,
        global_bbox=(22, 2, 3, 3),
        local_mask=np.ones((3, 3), dtype=bool),
        allowed_owners=(0, 1),
        component_label=2,
        preferred_owner=0,
    )

    result = _force_monotonic_component_owners((left, right))

    assert result is not None
    forced, owners = result
    assert owners == {1: 0, 2: 1}
    assert [fragment.allowed_owners for fragment in forced] == [(0,), (1,)]

    locked_result = _force_monotonic_component_owners(
        (left, right),
        locked_owners={1: 1},
    )
    assert locked_result is not None
    locked_forced, locked_owners = locked_result
    assert locked_owners[1] == 1
    assert [fragment.allowed_owners for fragment in locked_forced][0] == (1,)


def test_geometry_pair_hard_owner_and_source_suppression_are_audited() -> None:
    first = np.ones((2, 3), dtype=bool)
    second = np.ones((2, 3), dtype=bool)
    owner0, owner1, cuts = _pair_level_hard_owner_masks(first, second, 1)
    assert not np.any(owner0)
    assert np.all(owner1)
    assert np.all(cuts == -1)

    suppressed = _audit_suppressed_source_frames(
        (100, 101, 102),
        (5, 0, 4),
        {0},
    )
    assert suppressed == [
        {
            "source_index": 1,
            "frame_id": 101,
            "adjacent_audited_owner_topology_pairs": [0],
            "reason": "fully_covered_by_complete_adjacent_rgb_owner_topology",
        }
    ]
    with pytest.raises(RuntimeError, match="without an audited adjacent owner decision"):
        _audit_suppressed_source_frames((100, 101), (1, 0), set())


def test_raw_geometry_coordinates_invert_the_calibrated_rgb_map_subpixel(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=13)
    settings = CalibratedRGBPushbroomConfig()
    scale = estimate_rgb_motion_pixels_per_mm(
        session.frames,
        poses,
        session.calibration,
        settings,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in session.frames],
        poses,
        session.calibration,
        scale,
        settings,
    )
    renderer = CalibratedRGBPushbroomRenderer(layout, session.calibration, poses)
    source_index = 1
    x0 = layout.support_left_x[source_index]
    x1 = min(x0 + 20, layout.support_right_x[source_index])
    tile = renderer.render_local_geometry_map(
        session.frames[source_index], source_index, x0=x0, x1=x1
    )
    yy, xx = np.nonzero(tile.valid_mask)
    raw = np.column_stack((tile.source_map_x[yy, xx], tile.source_map_y[yy, xx]))
    recovered = _raw_pixels_to_virtual_coordinates(
        raw,
        layout=layout,
        calibration=session.calibration,
        camera_to_world=poses[source_index],
        source_index=source_index,
        residual_warp=None,
    )
    expected = np.column_stack((x0 + xx, yy)).astype(np.float64)
    np.testing.assert_allclose(recovered, expected, atol=1e-4)


def test_robust_scan_axis_downweights_noisy_trajectory_endpoints() -> None:
    centres = np.column_stack(
        (
            np.arange(10, dtype=np.float64) * 10.0,
            np.zeros(10, dtype=np.float64),
            np.zeros(10, dtype=np.float64),
        )
    )
    centres[0, 1] = 40.0
    centres[-1, 1] = -40.0

    axis, audit = _robust_scan_axis(centres)

    assert axis[0] > 0.999
    assert abs(axis[1]) < 0.02
    assert abs(axis[2]) < 1e-9
    assert audit["method"] == "robust_irls_pca_camera_centres"
    assert audit["pose_smoothing_applied"] is False
    assert audit["pose_interpolation_applied"] is False
    assert audit["downweighted_pose_count"] >= 2


def test_pushbroom_uses_every_real_frame_once_with_bounded_strip_residency(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=17)
    motions = _reliable_adjacent_rgb_motions(len(session.frames))

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=motions,
    )

    metadata = result.metadata
    metrics = metadata["quality_metrics"]
    assert len(motions) == len(session.frames) - 1
    assert metadata["source_count"] == len(session.frames)
    assert metadata["frame_ids"] == [frame.frame_id for frame in session.frames]
    assert metadata["single_inverse_remap_per_source"] is True
    assert metadata["interpolated_pose_count"] == 0
    assert result.owner_frame_id is not None
    assert result.owner_frame_id.dtype == np.int64
    assert result.owner_frame_id.shape == result.panorama.shape[:2]
    assert set(np.unique(result.owner_frame_id)) <= {
        frame.frame_id for frame in session.frames
    }
    assert metadata["mosaicing_method"]["name_en"] == (
        "Trajectory-Constrained Depth-Aware Multi-Viewpoint Side-Scan Mosaicing"
    )
    assert metadata["mosaicing_method"]["real_pose_nodes_preserved"] is True
    assert metadata["mosaicing_method"]["pose_smoothing_applied"] is False
    assert metrics["source_remap_count"] == len(session.frames)
    assert 2 <= metrics["maximum_resident_strips"] <= 5
    assert len(metadata["rgb_motion_scale"]["samples"]) == len(session.frames) - 1
    assert metadata["layout"]["endpoint_policy"] == "outward_half_fov"
    assert metadata["layout"]["scan_frame_audit"]["method"] == (
        "robust_irls_pca_camera_centres"
    )
    assert metadata["layout"]["scan_frame_audit"]["pose_count"] == len(
        session.frames
    )
    assert len(metadata["layout"]["endpoint_outer_owner_intervals_x"]) == 2
    assert metadata["layout"]["maximum_source_strip_width"] > 64
    supports = metadata["layout"]["source_support_intervals_x"]
    assert all(right - left <= 64 for left, right in supports[1:-1])
    assert all(count > 0 for count in metadata["source_owner_pixel_counts"])
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )
    assert metrics["endpoint_outer_half_fov_preserved"] is True
    assert metrics["endpoint_outer_half_fov_trimmed_column_counts"] == [0, 0]
    assert metrics["endpoint_outer_half_fov_trimmed_invalid_pixel_counts"] == [0, 0]


def test_unified_content_mode_has_no_legacy_foreground_owner_reservations(
    tmp_path: Path,
) -> None:
    """v5.2 keeps all RGB in the single strip-owner domain."""

    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=23)
    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        config={"unified_content_mode": True},
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    metrics = result.metadata["quality_metrics"]
    geometry = result.metadata["geometry_assisted_seam"]
    assert result.metadata["mosaicing_method"]["unified_content_mode"] is True
    assert metrics["foreground_anchor_reservation_count"] == 0
    assert metrics["foreground_anchor_handoff_retirement_count"] == 0
    assert metrics["foreground_deformation_pixel_count"] == 0
    assert geometry["foreground_anchor_reservation_count"] == 0


@pytest.mark.parametrize("duplicate_distance_mm", [16.1, 15.9])
def test_pushbroom_suppresses_near_duplicate_pose_owner_but_remaps_it_once(
    tmp_path: Path, duplicate_distance_mm: float
) -> None:
    session, original_poses = _make_rgb_pushbroom_input(tmp_path, seed=18)
    scan_axis = (
        original_poses[-1][:3, 3] - original_poses[0][:3, 3]
    )
    scan_axis /= np.linalg.norm(scan_axis)
    distances_mm = (0.0, 16.0, duplicate_distance_mm, 32.0, 48.0)
    poses: list[np.ndarray] = []
    for original, distance_mm in zip(
        original_poses, distances_mm, strict=True
    ):
        pose = original.copy()
        pose[:3, 3] = original_poses[0][:3, 3] + scan_axis * distance_mm
        poses.append(pose)

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=[
            {"dx": 16.0, "reliable": True}
            for _ in range(len(session.frames) - 1)
        ],
        quality_gate=False,
    )

    layout = result.metadata["layout"]
    suppression = result.metadata["redundant_pose_node_suppression"]
    assert layout["retained_owner_source_indices"] == [0, 1, 3, 4]
    assert len(layout["redundant_pose_node_suppression"]) == 1
    assert layout["redundant_pose_node_suppression"][0]["source_index"] == 2
    assert layout["owner_intervals_x"][2][0] == layout["owner_intervals_x"][2][1]
    assert result.metadata["source_owner_pixel_counts"][2] == 0
    assert result.metadata["quality_metrics"]["full_resolution_output_remap_count"] == 5
    assert suppression["suppressed_source_count"] == 1
    assert suppression["full_resolution_remap_count"] == 5
    assert suppression["all_real_pose_nodes_preserved"] is True
    assert suppression["foreground_owner_suppression"][
        "suppressed_source_indices"
    ] == [2]
    assert suppression["foreground_owner_suppression"]["complete"] is True
    assert suppression["owner_rewrite_audits"][0]["uncovered_pixel_count"] == 0
    assert result.metadata["suppressed_source_frames"][0]["source_index"] == 2
    assert result.metadata["quality_metrics"]["quality_pass"] is False
    assert any(
        "near-duplicate real pose node owner suppression" in reason
        for reason in result.metadata["quality_metrics"]["strict_failure_reasons"]
    )


def test_pushbroom_keeps_outward_endpoint_coverage_when_virtual_x_is_reversed(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=19)
    reversed_poses = []
    for pose in poses:
        reversed_pose = pose.copy()
        reversed_pose[:3, :3] = np.diag((-1.0, 1.0, -1.0))
        reversed_poses.append(reversed_pose)

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        reversed_poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    layout = result.metadata["layout"]
    metrics = result.metadata["quality_metrics"]
    assert layout["temporal_to_virtual_x_sign"] < 0.0
    assert layout["presentation_horizontal_flip"] is True
    assert result.metadata["presentation_orientation"] == {
        "axis": "calibrated_virtual_camera_image_right",
        "temporal_order_preserved_in_internal_canvas": True,
        "horizontal_flip_applied": True,
        "reason": "temporal_scan_axis_opposes_calibrated_image_right",
    }
    assert result.owner_frame_id is not None
    # The source chain remains chronological internally, but a camera whose
    # image-right points opposite to scan time must be presented spatially,
    # so its final (later) frame appears at the delivered image's left edge.
    assert int(np.median(result.owner_frame_id[:, 0])) > int(
        np.median(result.owner_frame_id[:, -1])
    )
    assert layout["endpoint_policy"] == "outward_half_fov"
    assert all(
        count > 0
        for count in metrics["endpoint_outer_half_fov_owner_pixel_counts"]
    )


def test_pushbroom_pair_blends_are_narrow_and_never_include_rgb_risk(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=23)
    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    pairs = result.metadata["pairs"]
    metrics = result.metadata["quality_metrics"]
    assert len(pairs) == len(session.frames) - 1
    assert all(2 <= pair["blend_width_pixels"] <= 8 for pair in pairs)
    # Endpoint supports may be shortened by the outward-half-FOV policy, but
    # interior seams retain a 64--160 px search corridor that is independent
    # from the 2--8 px output blend band.
    assert all(2 <= pair["search_corridor_width_pixels"] <= 64 for pair in pairs)
    assert all(
        32 <= pair["search_corridor_width_pixels"] <= 64 for pair in pairs[1:-1]
    )
    assert all(
        pair["search_corridor_width_pixels"] > pair["blend_width_pixels"]
        for pair in pairs
    )
    assert all(pair["blend_zone_risk_pixel_count"] == 0 for pair in pairs)
    assert metrics["blend_zone_risk_pixel_count"] == 0
    assert metrics["blend_zone_risk_fraction"] == 0.0
    assert metrics["blend_zone_fraction"] <= 0.20


def test_pushbroom_crops_from_valid_mask_and_preserves_valid_black_rgb(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=31)
    black = np.zeros(
        (session.calibration.height, session.calibration.width, 3), dtype=np.uint8
    )
    # Preserve a neutral wall rail solely for the mandatory photometric
    # measurement; the large black region remains valid output content.
    # Eight 16x16 spatial tiles are mandatory for the v10 independent
    # photometric train/held-out audit.
    black[:64, :, :] = 190
    for frame in session.frames:
        assert cv2.imwrite(str(frame.color_path), black)
        # Rendering after this deletion proves output pixels are not read from
        # the strict session's aligned-depth files.
        frame.aligned_depth_path.unlink()

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
    )

    crop = result.metadata["crop"]
    assert result.metadata["depth_used_for_output_pixels"] is False
    assert result.panorama.shape[:2] == (crop["height"], crop["width"])
    assert crop["width"] > 0 and crop["height"] > 0
    assert np.any(np.all(result.panorama == 0, axis=2))


def test_pushbroom_rejects_insufficient_reliable_rgb_motion_scale(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=41)
    invalid_motions = [
        {"dx": 0.0, "reliable": False} for _ in range(len(session.frames) - 1)
    ]

    with pytest.raises(RuntimeError, match="too few reliable adjacent RGB motion"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=invalid_motions,
        )


def test_pushbroom_rejects_unstable_rgb_motion_scale(tmp_path: Path) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=47)
    unstable_motions = [
        {"dx": value, "reliable": True} for value in (60.0, 120.0, 1200.0, 2400.0)
    ]

    with pytest.raises(RuntimeError, match="RGB-motion scale is unstable"):
        estimate_rgb_motion_pixels_per_mm(
            session.frames,
            poses,
            session.calibration,
            CalibratedRGBPushbroomConfig(),
            rgb_motions=unstable_motions,
        )


def _legacy_photometric_tile_masks(
    first: np.ndarray,
    second: np.ndarray,
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Reproduce the former full-frame-per-tile mask semantics as a test oracle."""

    candidate_mask = np.asarray(candidates, dtype=bool)
    height, width = candidate_mask.shape
    yy, xx = np.indices((height, width), dtype=np.int32)
    linear0 = pushbroom_module._srgb_to_linear_bgr(first)
    linear1 = pushbroom_module._srgb_to_linear_bgr(second)
    ratios = np.log(np.maximum(linear0, 1e-4)) - np.log(
        np.maximum(linear1, 1e-4)
    )
    tile_y, tile_x = yy // 16, xx // 16
    stable = np.zeros_like(candidate_mask)
    for row in range((height + 15) // 16):
        for column in range((width + 15) // 16):
            tile = candidate_mask & (tile_y == row) & (tile_x == column)
            if int(np.count_nonzero(tile)) < 40:
                continue
            values = ratios[tile]
            centre = np.median(values, axis=0)
            tile_mad = np.median(
                np.abs(values - centre.reshape(1, 3)), axis=0
            )
            if np.any(tile_mad > 0.035):
                continue
            stable[tile] = np.all(
                np.abs(values - centre.reshape(1, 3)) <= 0.05, axis=1
            )
    held_out_seed = ((xx + 3 * yy) % 5) == 0
    keep = np.zeros_like(stable)
    spatial_tiles = 0
    for row in range((height + 15) // 16):
        for column in range((width + 15) // 16):
            tile = stable & (tile_y == row) & (tile_x == column)
            if int(np.count_nonzero(tile & ~held_out_seed)) >= 32:
                keep |= tile
                spatial_tiles += 1
    return (
        stable,
        keep,
        keep & ~held_out_seed,
        keep & held_out_seed,
        spatial_tiles,
    )


def test_audited_photometric_tile_slices_match_legacy_full_frame_golden() -> None:
    height, width = 53, 49
    rng = np.random.default_rng(20260726)
    first = rng.integers(48, 188, (height, width, 3), dtype=np.uint8)
    expected_gain = np.asarray((0.94, 1.06, 0.98), dtype=np.float64)
    second = pushbroom_module._linear_to_srgb_bgr(
        pushbroom_module._srgb_to_linear_bgr(first)
        / expected_gain.reshape(1, 1, 3)
    )
    candidate_storage = np.ones((height * 2, width * 2), dtype=bool)
    candidates = candidate_storage[::2, ::2]
    candidates[:16, :16] = False
    candidates[:16, :16].flat[:39] = True
    # One gross colour-ratio outlier must be removed without rejecting the
    # otherwise stable tile.
    second[20, 35] = (8, 247, 8)
    legacy_stable, legacy_keep, legacy_train, legacy_held_out, legacy_tiles = (
        _legacy_photometric_tile_masks(first, second, candidates)
    )
    diagnostic: dict[str, object] = {}

    observation = pushbroom_module._audited_photometric_relation(
        first,
        second,
        candidates,
        method="rgb_texture_consistent",
        same_layer_corridor_used=True,
        diagnostic=diagnostic,
    )

    assert observation is not None
    assert diagnostic == {
        "initial_candidate_pixel_count": int(np.count_nonzero(candidates)),
        "stable_candidate_pixel_count": int(np.count_nonzero(legacy_stable)),
    }
    assert observation.candidate_pixels == int(np.count_nonzero(legacy_stable))
    assert observation.support_pixels == int(np.count_nonzero(legacy_keep))
    assert observation.training_pixels == int(np.count_nonzero(legacy_train))
    assert observation.held_out_pixels == int(np.count_nonzero(legacy_held_out))
    assert observation.spatial_tile_count == legacy_tiles
    np.testing.assert_array_equal(observation.safe_mask, legacy_held_out)
    np.testing.assert_allclose(
        observation.log_relation_bgr, np.log(expected_gain), atol=0.025
    )
    assert observation.method == "rgb_texture_consistent"
    assert observation.same_layer_corridor_used is True


def test_audited_photometric_tile_thresholds_remain_inclusive() -> None:
    height = width = 64
    first = np.full((height, width, 3), (96, 128, 160), dtype=np.uint8)
    second = first.copy()
    yy, xx = np.indices((height, width), dtype=np.int32)
    held_out_seed = ((xx + 3 * yy) % 5) == 0
    candidates = np.ones((height, width), dtype=bool)

    def set_tile_candidates(
        row: int, column: int, *, training: int, held_out: int
    ) -> None:
        y0, x0 = row * 16, column * 16
        tile_seed = held_out_seed[y0 : y0 + 16, x0 : x0 + 16]
        local = np.zeros((16, 16), dtype=bool)
        local.flat[np.flatnonzero(~tile_seed)[:training]] = True
        local.flat[np.flatnonzero(tile_seed)[:held_out]] = True
        candidates[y0 : y0 + 16, x0 : x0 + 16] = local

    # Stable membership requires 40 candidates.  The second-stage tile gate
    # independently requires 32 training pixels, both with inclusive bounds.
    set_tile_candidates(0, 0, training=31, held_out=8)  # 39: not stable
    set_tile_candidates(0, 1, training=31, held_out=9)  # stable, not retained
    set_tile_candidates(0, 2, training=32, held_out=8)  # retained exactly
    legacy_stable, legacy_keep, legacy_train, legacy_held_out, legacy_tiles = (
        _legacy_photometric_tile_masks(first, second, candidates)
    )

    observation = pushbroom_module._audited_photometric_relation(
        first,
        second,
        candidates,
        method="safe_wall",
        same_layer_corridor_used=False,
    )

    assert observation is not None
    assert not np.any(legacy_stable[:16, :16])
    assert np.count_nonzero(legacy_stable[:16, 16:32]) == 40
    assert not np.any(legacy_keep[:16, 16:32])
    assert np.count_nonzero(legacy_keep[:16, 32:48]) == 40
    assert observation.spatial_tile_count == legacy_tiles == 14
    assert observation.candidate_pixels == int(np.count_nonzero(legacy_stable))
    assert observation.support_pixels == int(np.count_nonzero(legacy_keep))
    assert observation.training_pixels == int(np.count_nonzero(legacy_train))
    assert observation.held_out_pixels == int(np.count_nonzero(legacy_held_out))
    np.testing.assert_array_equal(observation.safe_mask, legacy_held_out)


@pytest.mark.parametrize("tile_count,accepted", [(7, False), (8, True)])
def test_audited_photometric_requires_eight_spatial_tiles(
    tile_count: int, accepted: bool
) -> None:
    first = np.full((64, 64, 3), 128, dtype=np.uint8)
    candidates = np.zeros((64, 64), dtype=bool)
    for index in range(tile_count):
        row, column = divmod(index, 4)
        candidates[row * 16 : (row + 1) * 16, column * 16 : (column + 1) * 16] = True
    diagnostic: dict[str, object] = {}

    observation = pushbroom_module._audited_photometric_relation(
        first,
        first,
        candidates,
        method="safe_wall",
        same_layer_corridor_used=False,
        diagnostic=diagnostic,
    )

    assert (observation is not None) is accepted
    if accepted:
        assert observation is not None
        assert observation.spatial_tile_count == 8
        assert "reason" not in diagnostic
    else:
        assert diagnostic["reason"] == "insufficient_tiles"
        assert diagnostic["spatial_tile_count"] == 7
        yy, xx = np.indices(candidates.shape, dtype=np.int32)
        held_out_seed = ((xx + 3 * yy) % 5) == 0
        assert diagnostic["training_pixel_count"] == int(
            np.count_nonzero(candidates & ~held_out_seed)
        )
        assert diagnostic["held_out_pixel_count"] == int(
            np.count_nonzero(candidates & held_out_seed)
        )


def test_global_linear_rgb_gain_solver_uses_verified_partial_edges() -> None:
    source_count = 5
    # Each BGR channel has its own linear log-gain slope.  The mean-zero gauge
    # is known analytically, and affine gain curves are not biased by the
    # second-difference regularizer.
    slope_bgr = np.array((0.045, -0.025, 0.030), dtype=np.float64)
    expected_log_gains = (
        np.arange(source_count, dtype=np.float64)[:, None] - 2.0
    ) * slope_bgr[None, :]
    edges = [
        _PhotometricEdge(
            log_relation_bgr=slope_bgr.copy(),
            support_pixels=1024,
            mad_bgr=np.full(3, 0.005, dtype=np.float64),
            raw_signed_l_delta=0.0,
        )
        for _ in range(source_count - 1)
    ]

    gains, metrics = _solve_global_linear_rgb_gains(source_count, edges)

    assert metrics["photometric_mode"] == (
        "source_aware_global_linear_rgb_v2_two_stage"
    )
    assert metrics["photometric_global_solver"] is True
    np.testing.assert_allclose(np.log(gains), expected_log_gains, atol=1e-8)
    assert not np.allclose(gains[:, 0], gains[:, 1])
    assert not np.allclose(gains[:, 1], gains[:, 2])

    fallback_gains, fallback_metrics = _solve_global_linear_rgb_gains(
        source_count, [edges[0], None, *edges[2:]]
    )
    assert not np.array_equal(fallback_gains, np.ones((source_count, 3)))
    assert fallback_metrics["photometric_global_solver"] is True
    assert fallback_metrics["photometric_partial_observation_solver"] is True
    assert fallback_metrics["photometric_global_identity_fallback"] is False
    assert fallback_metrics["photometric_identity_fallback_pair_count"] == 1


def test_rgb_texture_consistent_relation_uses_fixed_tiles_and_held_out_pixels() -> None:
    height, width = 96, 96
    yy, xx = np.indices((height, width), dtype=np.float32)
    # Deliberately chromatic material texture: no neutral-wall criterion may
    # be needed for a stable same-surface relation.
    first = np.stack(
        (
            48.0 + xx * 1.1,
            55.0 + yy * 0.8,
            70.0 + (xx + yy) * 0.55,
        ),
        axis=2,
    ).astype(np.uint8)
    expected_gain = np.array((0.91, 1.08, 0.96), dtype=np.float64)
    second = pushbroom_module._linear_to_srgb_bgr(
        pushbroom_module._srgb_to_linear_bgr(first) / expected_gain.reshape(1, 1, 3)
    )
    common = np.ones((height, width), dtype=bool)
    observation = _rgb_texture_consistent_log_relation(
        first, second, common, np.zeros_like(common)
    )

    assert observation is not None
    assert observation.method == "rgb_texture_consistent"
    assert observation.spatial_tile_count >= 8
    assert observation.training_pixels >= 256
    assert observation.held_out_pixels >= 64
    assert observation.held_out_log_residual_p95 is not None
    np.testing.assert_allclose(
        observation.log_relation_bgr, np.log(expected_gain), atol=0.025
    )


def test_safe_same_layer_background_accepts_chromatic_post_gain_support() -> None:
    first = np.full((64, 64, 3), (48, 112, 176), dtype=np.uint8)
    second = first.copy()
    common = np.ones((64, 64), dtype=bool)
    same_layer = np.zeros_like(common)
    same_layer[:, :40] = True
    risk_guard = np.zeros_like(common)
    risk_guard[20:44, 20:24] = True

    safe = pushbroom_module._safe_same_layer_background_mask(
        first,
        second,
        common,
        risk_guard,
        same_layer,
        low_gradient_quantile=0.45,
    )

    assert np.count_nonzero(safe) > 0
    assert not np.any(safe & risk_guard)
    assert not np.any(safe & ~same_layer)
    # The deliberately saturated orange material is not a neutral safe wall.
    assert not np.any(
        pushbroom_module._safe_white_wall_mask(
            first,
            second,
            common,
            risk_guard,
            low_gradient_quantile=0.45,
        )
    )


def test_post_gain_risk_retains_structure_without_replaying_old_lab_mask() -> None:
    first = np.full((32, 48, 3), 128, dtype=np.uint8)
    second = first.copy()
    valid = np.ones((32, 48), dtype=bool)
    current = pushbroom_module._rgb_risk_details(
        first,
        second,
        valid,
        valid,
    )
    assert not np.any(current.mask)

    raw_structure = np.zeros_like(valid)
    raw_structure[12:20, 23] = True
    retained = pushbroom_module._post_gain_risk_details(
        first,
        second,
        valid,
        valid,
        raw_structural_seed=raw_structure,
        raw_structural_edge_offset_p95=3.5,
        post_gain_risk=current,
    )

    assert np.all(retained.structural_seed_mask[raw_structure])
    assert np.any(retained.mask)
    assert retained.edge_offset_p95 == 3.5


def test_same_layer_owner_guard_does_not_reintroduce_single_source_geometry() -> None:
    common = np.zeros((8, 16), dtype=bool)
    common[:, 3:13] = True
    guard = np.zeros_like(common)
    guard[:, 5:11] = True
    same_layer = np.zeros_like(common)
    same_layer[:, 7:9] = True
    geometry = np.zeros_like(common)
    geometry[:, 1:4] = True
    geometry[:, 12:15] = True

    owner_guard = pushbroom_module._same_layer_owner_guard(
        guard,
        same_layer,
        geometry,
        common,
    )

    assert not np.any(owner_guard & ~common)
    assert not np.any(owner_guard[:, 7:9])
    assert np.all(owner_guard[:, 3])
    assert np.all(owner_guard[:, 12])
    assert np.all(owner_guard[:, 5:7])
    assert np.all(owner_guard[:, 9:11])


def test_safe_background_cut_can_move_across_unblended_protected_free_route() -> None:
    valid = np.ones((1, 12), dtype=bool)
    safe = np.zeros_like(valid)
    safe[:, 7:10] = True
    movable = np.zeros_like(valid)
    movable[:, 3:10] = True

    owner0, owner1, cuts, changed_rows, maximum_shift = (
        pushbroom_module._refine_cuts_on_contiguous_safe_background(
            valid,
            valid,
            safe,
            movable,
            np.array([2], dtype=np.int32),
            nominal_boundary=8,
            blend_width=3,
        )
    )

    assert cuts.tolist() == [8]
    assert changed_rows == 1
    assert maximum_shift == 6
    assert np.all(owner0[:, :9])
    assert np.all(owner1[:, 9:])
    assert not np.any(owner0 & owner1)

    movable[:, 5] = False
    _, _, blocked_cuts, blocked_rows, _ = (
        pushbroom_module._refine_cuts_on_contiguous_safe_background(
            valid,
            valid,
            safe,
            movable,
            np.array([2], dtype=np.int32),
            nominal_boundary=8,
            blend_width=3,
        )
    )
    assert blocked_cuts.tolist() == [2]
    assert blocked_rows == 0


def test_same_layer_cut_can_use_extended_half_corridor_without_relaxing_wall() -> None:
    valid = np.ones((1, 96), dtype=bool)
    safe = np.zeros_like(valid)
    safe[:, 76:79] = True
    movable = np.ones_like(valid)
    old = np.array([38], dtype=np.int32)

    _, _, wall_cuts, wall_rows, _ = (
        pushbroom_module._refine_cuts_on_contiguous_safe_background(
            valid,
            valid,
            safe,
            movable,
            old,
            nominal_boundary=38,
            blend_width=3,
        )
    )
    _, _, layer_cuts, layer_rows, maximum_shift = (
        pushbroom_module._refine_cuts_on_contiguous_safe_background(
            valid,
            valid,
            safe,
            movable,
            old,
            nominal_boundary=38,
            blend_width=3,
            extended_same_layer_background=safe,
        )
    )

    assert wall_cuts.tolist() == [38]
    assert wall_rows == 0
    assert layer_cuts.tolist() == [77]
    assert layer_rows == 1
    assert maximum_shift == 39


def test_content_aware_nominal_boundary_avoids_a_protected_object() -> None:
    common = np.ones((40, 96), dtype=bool)
    risk = np.zeros_like(common, dtype=np.uint8)
    risk[4:38, 45:53] = 255

    selected = pushbroom_module._content_aware_nominal_boundary(
        risk,
        common,
        48,
    )

    assert abs(selected - 48) <= 32
    assert not 45 <= selected < 53
    assert selected in {42, 43, 54, 55}

    sparse_risk = np.zeros_like(common, dtype=np.uint8)
    sparse_risk[10, 48] = 255
    assert (
        pushbroom_module._content_aware_nominal_boundary(
            sparse_risk,
            common,
            48,
        )
        == 48
    )


def test_preferred_safe_band_boundary_moves_before_graphcut() -> None:
    safe = np.zeros((40, 96), dtype=bool)
    safe[8:34, 70:76] = True
    common = np.ones_like(safe)

    boundary, rows = pushbroom_module._preferred_safe_band_boundary(
        safe,
        common,
        40,
        blend_width=4,
    )

    assert 71 <= boundary <= 73
    assert rows == 26
    sparse = np.zeros_like(safe)
    sparse[0:2, 70:76] = True
    assert pushbroom_module._preferred_safe_band_boundary(
        sparse,
        common,
        40,
        blend_width=4,
    ) == (40, 0)

    aperture, aperture_boundary, aperture_rows = (
        pushbroom_module._same_layer_seam_aperture(
            safe,
            common,
            40,
            blend_width=4,
        )
    )
    assert aperture_boundary == boundary
    assert aperture_rows == 26
    assert np.count_nonzero(aperture) > 26 * 4
    assert not np.any(aperture & ~safe)


def test_photometric_failure_uses_identity_gain_and_full_hard_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=49)
    monkeypatch.setattr(
        pushbroom_module,
        "_safe_wall_log_relation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pushbroom_module,
        "_rgb_texture_consistent_log_relation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pushbroom_module,
        "_degraded_background_log_relation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        pushbroom_module,
        "_safe_white_wall_mask",
        lambda first, *args, **kwargs: np.zeros(first.shape[:2], dtype=bool),
    )
    monkeypatch.setattr(
        pushbroom_module,
        "_safe_same_layer_background_mask",
        lambda first, *args, **kwargs: np.zeros(first.shape[:2], dtype=bool),
    )

    result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
        quality_gate=False,
    )

    assert result.metadata["color_gains"] == [
        [1.0, 1.0, 1.0] for _ in session.frames
    ]
    calibration = result.metadata["photometric_calibration"]
    assert calibration["identity_hard_owner_fallback_used"] is True
    assert calibration["identity_hard_owner_fallback_pair_count"] == (
        len(session.frames) - 1
    )
    assert all(
        pair["photometric_audit"]["method"] == "identity_hard_owner"
        and pair["photometric_audit"]["unit_gain_fallback"] is True
        and pair["photometric_audit"]["hard_owner_required"] is True
        and pair["photometric_hard_owner_fallback"] is True
        and pair["blend_zone_pixel_count"] == 0
        and pair["multiband_levels"] == 0
        for pair in result.metadata["pairs"]
    )
    assert result.metadata["quality_metrics"]["quality_pass"] is False
    assert any(
        "photometric identity hard-owner fallback" in reason
        for reason in result.metadata["quality_metrics"]["strict_failure_reasons"]
    )


def test_linear_rgb_gain_application_is_per_channel_not_gamma_scalar() -> None:
    encoded = np.full((4, 5, 3), (96, 144, 192), dtype=np.uint8)
    gain_bgr = np.array((1.20, 0.82, 1.08), dtype=np.float64)

    corrected = _apply_linear_bgr_gain(encoded, gain_bgr)
    expected_linear = _srgb_to_linear_bgr(encoded) * gain_bgr.reshape(1, 1, 3)
    actual_linear = _srgb_to_linear_bgr(corrected)

    # Encoding quantization is the only allowed discrepancy; a scalar applied
    # to gamma-encoded RGB would fail this linear-light comparison.
    np.testing.assert_allclose(actual_linear, expected_linear, atol=0.005)
    assert not np.all(corrected[:, :, 0] == corrected[:, :, 1])
    assert not np.all(corrected[:, :, 1] == corrected[:, :, 2])


def test_protected_foreground_component_is_owned_wholly_by_one_source() -> None:
    height, width = 32, 64
    first = np.full((height, width, 3), 190, dtype=np.uint8)
    second = np.full((height, width, 3), 190, dtype=np.uint8)
    valid = np.ones((height, width), dtype=bool)
    # A horizontal hose spans the nominal seam.  Its uniform interior is part
    # of the supplied connected protection component, so the GraphCut owner
    # must not split it at the nominal centre.
    protected = np.zeros((height, width), dtype=bool)
    protected[12:20, 18:46] = True
    first[protected] = (20, 20, 20)
    second[protected] = (20, 20, 20)

    owner0, owner1, cuts, _, _, split_count, boundary_guard_count = (
        _graphcut_monotonic_owner(
            first,
            second,
            valid,
            valid,
            protected,
            nominal_boundary=32,
        )
    )

    assert np.all(owner0[protected]) or np.all(owner1[protected])
    assert not np.any(owner0[protected] & owner1[protected])
    assert split_count == 0
    assert boundary_guard_count == 0
    assert np.all(cuts[12:20] < 18) or np.all(cuts[12:20] >= 46)


def test_graphcut_locks_only_complete_same_layer_safe_rows() -> None:
    height, width = 24, 64
    first = np.full((height, width, 3), 120, dtype=np.uint8)
    second = first.copy()
    valid = np.ones((height, width), dtype=bool)
    protected = np.zeros_like(valid)
    safe = np.zeros_like(valid)
    safe[5:19, 30:34] = True
    preferred_audit: dict[str, int] = {}

    _, _, cuts, _, _, split_count, boundary_guard_count = (
        _graphcut_monotonic_owner(
            first,
            second,
            valid,
            valid,
            protected,
            nominal_boundary=31,
            preferred_safe_background=safe,
            preferred_blend_width=4,
            preferred_safe_audit=preferred_audit,
        )
    )

    assert np.all(cuts[5:19] == 31)
    assert split_count == 0
    assert boundary_guard_count == 0
    assert preferred_audit == {
        "candidate_row_count": 14,
        "component_admitted_row_count": 14,
        "component_blocked_row_count": 0,
        "final_aligned_row_count": 14,
    }


def test_graphcut_relabels_preflight_constraints_after_guards_merge() -> None:
    height, width = 24, 64
    image = np.full((height, width, 3), 120, dtype=np.uint8)
    valid = np.ones((height, width), dtype=bool)
    protected = np.zeros_like(valid)
    protected[8:12, 10:20] = True
    protected[8:12, 30:40] = True
    protected[9:11, 20:30] = True
    first = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(10, 8, 10, 4),
        local_mask=np.ones((4, 10), dtype=bool),
        component_label=1,
    )
    second = ProtectedComponentFragment(
        pair_index=0,
        global_bbox=(30, 8, 10, 4),
        local_mask=np.ones((4, 10), dtype=bool),
        component_label=2,
    )
    source_labels = _component_constraint_label_raster(
        (first, second), left=0, right=width, height=height
    )

    owner0, owner1, *_ = _graphcut_monotonic_owner(
        image,
        image,
        valid,
        valid,
        protected,
        nominal_boundary=32,
        component_owner_constraints={1: 0, 2: 0},
        component_constraint_labels=source_labels,
    )

    assert np.all(owner0[protected])
    assert not np.any(owner1[protected])
    with pytest.raises(
        RuntimeError,
        match="merges conflicting owner constraints",
    ):
        _graphcut_monotonic_owner(
            image,
            image,
            valid,
            valid,
            protected,
            nominal_boundary=32,
            component_owner_constraints={1: 0, 2: 1},
            component_constraint_labels=source_labels,
        )


def test_local_multiband_uses_distinct_owner_masks_and_adaptive_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height, width = 12, 20
    first = np.full((height, width, 3), (80, 120, 160), dtype=np.uint8)
    second = np.full((height, width, 3), (120, 160, 200), dtype=np.uint8)
    common = np.ones((height, width), dtype=bool)
    protected = np.zeros((height, width), dtype=bool)
    safe_wall = common.copy()
    owner0 = np.zeros_like(common)
    owner0[:, :8] = True
    owner1 = common & ~owner0
    cuts = np.full(height, 7, dtype=np.int32)
    captured_masks: list[np.ndarray] = []
    captured_images: list[np.ndarray] = []

    class _CapturingBlender:
        def setNumBands(self, _: int) -> None:
            pass

        def prepare(self, _: tuple[int, int, int, int]) -> None:
            pass

        def feed(
            self, image: np.ndarray, mask: np.ndarray, _: tuple[int, int]
        ) -> None:
            captured_images.append(np.asarray(image).copy())
            captured_masks.append(np.asarray(mask).copy())

        def blend(
            self, _: None, __: None
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                captured_images[0],
                np.where(captured_masks[0] | captured_masks[1], 255, 0).astype(
                    np.uint8
                ),
            )

    monkeypatch.setattr(
        pushbroom_module.cv2,
        "detail_MultiBandBlender",
        _CapturingBlender,
    )
    (
        _,
        zone,
        pixels,
        levels,
        mask0_pixels,
        mask1_pixels,
        masks_distinct,
    ) = _blend_safe_pair_zone(
        first,
        second,
        common,
        protected,
        safe_wall,
        owner0,
        owner1,
        cuts,
        blend_width=6,
        levels=3,
    )

    assert pixels == int(np.count_nonzero(zone)) > 0
    assert levels == 3
    assert mask0_pixels > 0 and mask1_pixels > 0
    assert masks_distinct is True
    assert len(captured_masks) == 2
    assert not np.array_equal(captured_masks[0], captured_masks[1])
    assert _adaptive_multiband_levels(2, 3) == 1
    assert _adaptive_multiband_levels(4, 3) == 2
    assert _adaptive_multiband_levels(8, 3) == 3


def test_preview_residual_diagnostics_keep_identity_output_and_remap_counts_separate(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=53)
    motions = _reliable_adjacent_rgb_motions(len(session.frames))
    default_result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        rgb_motions=motions,
    )
    identity_result = render_calibrated_rgb_pushbroom(
        session.frames,
        poses,
        session.calibration,
        config={"residual_alignment": {"background_model": "identity"}},
        rgb_motions=motions,
    )

    # Preview evidence uses a separate low-resolution inverse remap.  It is
    # forbidden from changing a formal source sample while the selected model
    # remains identity.
    assert np.array_equal(default_result.panorama, identity_result.panorama)
    alignment = default_result.metadata["residual_alignment"]
    metrics = default_result.metadata["quality_metrics"]
    assert alignment["selected_model"] == "identity"
    assert alignment["analysis_preview_remap_count"] == len(session.frames)
    assert alignment["full_resolution_output_remap_count"] == len(session.frames)
    assert metrics["analysis_preview_remap_count"] == len(session.frames)
    assert metrics["full_resolution_output_remap_count"] == len(session.frames)
    assert len(alignment["evidence"]) == len(session.frames) - 1
    working = alignment["working_set_audit"]
    assert working["preview_streaming_maximum_resident_previews"] == 2
    assert (
        working["preview_evidence_pixel_count"]
        <= working["preview_evidence_hard_limit_pixels"]
    )
    assert working["preview_evidence_storage"] == "disk_backed_adjacent_analysis_only"


def test_preview_evidence_budget_fails_before_formal_full_resolution_remaps(
    tmp_path: Path,
) -> None:
    session, poses = _make_rgb_pushbroom_input(tmp_path, seed=59)

    with pytest.raises(RuntimeError, match="preview evidence exceeds"):
        render_calibrated_rgb_pushbroom(
            session.frames,
            poses,
            session.calibration,
            rgb_motions=_reliable_adjacent_rgb_motions(len(session.frames)),
            config={
                "residual_alignment": {
                    "maximum_evidence_megapixels": 0.000001,
                }
            },
        )
