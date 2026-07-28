from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from panorama_demo.inspection_identity_runtime import (
    InspectionIdentityRuntimeConfig,
    _bounded_exact_corridor_transfer,
    _expand_resolved_interval_rgb_context,
    _exclude_shelf_tracks_from_direct_preseam_candidates,
    _fastsam_cuda_audit,
    _format_shelf_unsat_context,
    _level4b_panel_evidence_score,
    _proposal_polygons,
    _resolve_fastsam_model,
    _resolve_shelf_native_owner_conflict_groups,
    _trim_fixed_corridor_boundary_overlap,
    build_inspection_identity_runtime,
)
from panorama_demo.inspection_identity_owner_planner import (
    InspectionIdentityOwnerFrame,
)
from panorama_demo.inspection_multiview import (
    InspectionForegroundIdentityOwner,
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    InspectionPreSeamHardOwnerInterval,
    VirtualPerspectivePanel,
)
from panorama_demo.session import CameraIntrinsics


def test_disabled_runtime_is_pixel_path_neutral() -> None:
    result = build_inspection_identity_runtime(
        (),
        (),
        CameraIntrinsics(
            width=10,
            height=8,
            fx=8.0,
            fy=8.0,
            cx=5.0,
            cy=4.0,
            distortion=(),
        ),
        inspection_config=InspectionMultiviewConfig(),
        runtime_config={"enabled": False},
    )

    assert result.pre_seam_hard_owner_intervals == ()
    assert result.foreground_owners == ()
    assert result.audit == {
        "schema": "inspection-identity-runtime/v1",
        "enabled": False,
        "executed": False,
        "applied": False,
        "foreground_identity_owner_count": 0,
        "pre_seam_hard_owner_interval_count": 0,
        "object_owner_application_count": 0,
    }


def _fixed_test_interval(
    mask: np.ndarray,
    *,
    background_panel_lock_required: bool = True,
) -> InspectionPreSeamHardOwnerInterval:
    return InspectionPreSeamHardOwnerInterval(
        track_id=9001,
        panel_index=0,
        frame_id=10,
        lock_mask=mask,
        union_footprint=mask,
        rgb_source_panel_index=0,
        rgb_transfer_mask=mask,
        owner_only_mask=mask,
        background_panel_lock_required=background_panel_lock_required,
    )


def test_fixed_corridor_trim_removes_only_non_object_row_boundary() -> None:
    support = np.zeros((8, 16), dtype=bool)
    support[2:6, 4:12] = True
    corridor = np.zeros_like(support)
    corridor[2:6, 2:14] = True
    fixed = np.zeros_like(support)
    fixed[2:6, 2:4] = True

    trimmed, accepted, audit = _trim_fixed_corridor_boundary_overlap(
        corridor,
        support,
        candidate_frame_id=20,
        fixed_intervals=(_fixed_test_interval(fixed),),
    )

    assert accepted is True
    assert not np.any(trimmed & fixed)
    assert np.all(trimmed[support])
    assert np.all(trimmed[2:6, 4:14])
    assert audit == [
        {
            "fixed_track_id": 9001,
            "fixed_frame_id": 10,
            "trimmed_pixel_count": 8,
            "requested_overlap_pixel_count": 8,
            "measured_support_overlap_pixel_count": 0,
            "measured_support_overlap_ratio": 0.0,
            "measured_overlap_in_fixed_exact_support_pixel_count": 0,
            "measured_overlap_on_both_inner_boundaries_pixel_count": 0,
            "measured_overlap_on_candidate_inner_boundary_pixel_count": 0,
            "cross_track_boundary_alias_partition": False,
            "cross_track_boundary_alias_absolute_limit_pixels": 4096,
            "cross_track_boundary_alias_relative_limit": 0.15,
            "delegated_measured_boundary_pixel_count": 0,
            "delegated_measured_boundary_owner": None,
            "guard_overlap_retained_for_decoupled_owner": False,
            "zero_measured_support_intersection": True,
            "all_member_supports_retained": True,
            "all_member_supports_single_owner_covered": True,
            "per_row_boundary_trim_passed": True,
            "subtraction_row_contiguous": True,
            "trimmed_pixel_owner": "existing_fixed_corridor",
            "accepted": True,
        }
    ]


def test_tiny_shared_instance_boundary_gets_one_existing_rgb_owner() -> None:
    support = np.zeros((120, 140), dtype=bool)
    support[10:110, 10:110] = True
    fixed = np.zeros_like(support)
    fixed[20:40, 10] = True

    trimmed, accepted, audit = _trim_fixed_corridor_boundary_overlap(
        support,
        support,
        candidate_frame_id=20,
        fixed_intervals=(
            _fixed_test_interval(
                fixed, background_panel_lock_required=False
            ),
        ),
    )

    assert accepted is True
    assert np.all(trimmed[fixed])
    assert audit[0]["measured_support_overlap_pixel_count"] == 20
    assert audit[0]["measured_support_overlap_ratio"] == pytest.approx(0.002)
    assert audit[0]["cross_track_boundary_alias_partition"] is True
    assert audit[0]["all_member_supports_retained"] is True
    assert audit[0]["all_member_supports_single_owner_covered"] is True
    assert audit[0]["guard_overlap_retained_for_decoupled_owner"] is True


def test_large_shared_instance_interior_remains_fail_closed() -> None:
    support = np.zeros((120, 140), dtype=bool)
    support[10:110, 10:110] = True
    fixed = np.zeros_like(support)
    fixed[10:110, 48:68] = True

    _, accepted, audit = _trim_fixed_corridor_boundary_overlap(
        support,
        support,
        candidate_frame_id=20,
        fixed_intervals=(
            _fixed_test_interval(
                fixed, background_panel_lock_required=False
            ),
        ),
    )

    assert accepted is False
    assert audit[0]["cross_track_boundary_alias_partition"] is False
    assert audit[0]["per_row_boundary_trim_passed"] is True


def test_bounded_relative_shared_label_is_stably_partitioned() -> None:
    support = np.zeros((120, 140), dtype=bool)
    support[10:110, 10:110] = True
    fixed = np.zeros_like(support)
    fixed[10:110, 48:54] = True

    trimmed, accepted, audit = _trim_fixed_corridor_boundary_overlap(
        support,
        support,
        candidate_frame_id=20,
        fixed_intervals=(
            _fixed_test_interval(
                fixed, background_panel_lock_required=False
            ),
        ),
    )

    assert accepted is True
    assert np.all(trimmed[support])
    assert audit[0]["measured_support_overlap_pixel_count"] == 600
    assert audit[0]["measured_support_overlap_ratio"] == pytest.approx(0.06)
    assert audit[0]["cross_track_boundary_alias_partition"] is True
    assert (
        audit[0]["delegated_measured_boundary_owner"]
        == "existing_fixed_corridor"
    )


def test_corridor_rgb_transfer_is_limited_to_exact_support_plus_two() -> None:
    support = np.zeros((20, 30), dtype=bool)
    support[8:12, 12:16] = True
    guard = np.zeros_like(support)
    guard[4:16, 6:24] = True
    panel_valid = np.ones_like(support)

    transfer, audit = _bounded_exact_corridor_transfer(
        support,
        guard,
        panel_valid,
        dilation_pixels=2,
    )

    assert transfer is not None
    assert np.all(transfer[support])
    assert not np.any(transfer & ~guard)
    assert np.count_nonzero(transfer) < np.count_nonzero(guard)
    assert audit["rgb_transfer_dilation_pixels"] == 2
    assert audit["all_member_measured_support_retained"] is True
    assert audit["guard_minus_rgb_transfer_pixel_count"] > 0
    assert audit["rgb_blended_or_generated"] is False


def test_corridor_optional_dilation_yields_to_foreign_measured_support() -> None:
    support = np.zeros((20, 30), dtype=bool)
    support[8:12, 12:16] = True
    foreign = np.zeros_like(support)
    foreign[8:12, 17:20] = True
    guard = np.ones_like(support)

    transfer, audit = _bounded_exact_corridor_transfer(
        support,
        guard,
        guard,
        dilation_pixels=2,
        reserved_foreign_support=foreign,
    )

    assert transfer is not None
    assert np.all(transfer[support])
    assert not np.any(transfer & foreign)
    assert (
        audit["excluded_optional_dilation_foreign_support_pixel_count"]
        > 0
    )
    assert audit["exact_member_foreign_support_overlap_pixel_count"] == 0
    assert audit["all_member_measured_support_retained"] is True


def test_corridor_exact_support_overlap_is_retained_for_final_assignment() -> None:
    support = np.zeros((20, 30), dtype=bool)
    support[8:12, 12:16] = True
    foreign = np.zeros_like(support)
    foreign[9:11, 14:18] = True
    guard = np.ones_like(support)

    transfer, audit = _bounded_exact_corridor_transfer(
        support,
        guard,
        guard,
        dilation_pixels=2,
        reserved_foreign_support=foreign,
    )

    assert transfer is not None
    assert np.all(transfer[support])
    assert audit["exact_member_foreign_support_overlap_pixel_count"] == 4
    assert (
        audit["exact_member_overlap_deferred_to_final_assignment_audit"]
        is True
    )
    assert audit["all_member_measured_support_retained"] is True


def test_corridor_member_context_fills_each_object_without_group_bridge() -> None:
    first = np.zeros((40, 80), dtype=bool)
    first[14:26, 8:22] = True
    first[17:23, 12:18] = False
    second = np.zeros_like(first)
    second[14:26, 58:72] = True
    second[17:23, 62:68] = False
    support = first | second
    guard = np.ones_like(support)

    transfer, audit = _bounded_exact_corridor_transfer(
        support,
        guard,
        guard,
        dilation_pixels=0,
        member_supports=(first, second),
        member_context_guard_pixels=0,
        maximum_member_context_ratio=1.5,
    )

    assert transfer is not None
    assert np.all(transfer[17:23, 12:18])
    assert np.all(transfer[17:23, 62:68])
    assert not np.any(transfer[:, 30:50])
    assert audit["member_context_member_count"] == 2
    assert all(
        row["accepted"] for row in audit["member_context_rows"]
    )
    assert audit["accepted_member_context_pixel_count"] > 0


def test_corridor_member_context_ratio_limit_falls_back_to_exact() -> None:
    member = np.zeros((20, 80), dtype=bool)
    member[10, 5] = True
    member[10, 70] = True
    guard = np.ones_like(member)

    transfer, audit = _bounded_exact_corridor_transfer(
        member,
        guard,
        guard,
        dilation_pixels=0,
        member_supports=(member,),
        member_context_guard_pixels=0,
        maximum_member_context_ratio=1.5,
    )

    assert transfer is not None
    assert np.array_equal(transfer, member)
    assert audit["member_context_rows"][0]["accepted"] is False
    assert (
        audit["member_context_rows"][0]["fallback"]
        == "exact_plus_requested_dilation"
    )


def test_rgb_context_is_added_only_after_exact_intervals_are_resolved() -> None:
    shape = (30, 60)
    first = np.zeros(shape, dtype=bool)
    first[10:20, 8:20] = True
    first[13:17, 11:17] = False
    second = np.zeros(shape, dtype=bool)
    second[10:20, 40:52] = True
    second[13:17, 43:49] = False
    frames = tuple(
        InspectionIdentityOwnerFrame(
            panel_index=index,
            source_index=index,
            frame_id=frame_id,
            image_bgr=np.zeros((*shape, 3), dtype=np.uint8),
            depth_mm=np.full(shape, 1000.0, dtype=np.float32),
            reliable_depth=np.ones(shape, dtype=bool),
            camera_to_world=np.eye(4, dtype=np.float64),
            panel_valid_mask=np.ones(shape, dtype=bool),
        )
        for index, frame_id in enumerate((10, 20))
    )
    intervals = tuple(
        InspectionPreSeamHardOwnerInterval(
            track_id=index,
            panel_index=index,
            frame_id=frame_id,
            lock_mask=np.ones(shape, dtype=bool),
            union_footprint=mask,
            rgb_transfer_mask=mask,
            owner_only_mask=np.ones(shape, dtype=bool),
            rgb_context_member_supports=(mask,),
            background_panel_lock_required=False,
        )
        for index, (frame_id, mask) in enumerate(
            ((10, first), (20, second))
        )
    )

    expanded, audit = _expand_resolved_interval_rgb_context(
        intervals,
        frames,
    )

    assert np.all(expanded[0].rgb_transfer_mask[13:17, 11:17])
    assert np.all(expanded[1].rgb_transfer_mask[13:17, 43:49])
    assert not np.any(
        expanded[0].rgb_transfer_mask
        & expanded[1].rgb_transfer_mask
    )
    assert audit["expanded_interval_count"] == 2
    assert audit["different_frame_overlap_pixel_count"] == 0
    assert audit["identity_or_csp_decision_modified"] is False


def test_level4b_evidence_rich_panel_beats_existing_corridor_frame() -> None:
    existing_score, existing_audit = _level4b_panel_evidence_score(
        (
            (190, True, (0.8, 0.9), 100, True),
            (225, False, None, 100, False),
            (226, False, None, 100, False),
            (234, False, None, 100, False),
        ),
        corridor_center_x=500.0,
        panel_center_x=500.0,
        panel_index=6,
        existing_corridor_frame=True,
    )
    rich_score, rich_audit = _level4b_panel_evidence_score(
        (
            (190, True, (0.7, 0.8), 100, True),
            (225, True, (0.7, 0.8), 100, True),
            (226, True, (0.7, 0.8), 100, True),
            (234, True, (0.7, 0.8), 100, True),
        ),
        corridor_center_x=500.0,
        panel_center_x=530.0,
        panel_index=8,
        existing_corridor_frame=False,
    )

    assert rich_score > existing_score
    assert existing_audit["eligible_complete_observation_count"] == 1
    assert rich_audit["eligible_complete_observation_count"] == 4
    assert rich_audit["any_reference_observation_count"] == 4
    assert (
        rich_audit["score_policy"]
        == "eligible_complete_count_then_corridor_center_then_support_"
        "weighted_selection_rank_then_any_reference_count_then_panel_then_"
        "existing_corridor_frame_last_tiebreak"
    )


def test_level4b_equally_complete_panel_prefers_geometric_center() -> None:
    off_center, _ = _level4b_panel_evidence_score(
        ((1, True, (0.99, 0.99), 100, True),),
        corridor_center_x=900.0,
        panel_center_x=700.0,
        panel_index=8,
        existing_corridor_frame=False,
    )
    centered, _ = _level4b_panel_evidence_score(
        ((1, True, (0.90, 0.90), 100, True),),
        corridor_center_x=900.0,
        panel_center_x=900.0,
        panel_index=9,
        existing_corridor_frame=False,
    )

    assert centered > off_center


def test_fixed_corridor_trim_vetoes_measured_object_support_overlap() -> None:
    support = np.zeros((8, 16), dtype=bool)
    support[2:6, 4:12] = True
    corridor = np.zeros_like(support)
    corridor[2:6, 2:14] = True
    fixed = np.zeros_like(support)
    fixed[2:6, 3:5] = True

    trimmed, accepted, audit = _trim_fixed_corridor_boundary_overlap(
        corridor,
        support,
        candidate_frame_id=20,
        fixed_intervals=(_fixed_test_interval(fixed),),
    )

    assert accepted is False
    assert np.array_equal(trimmed, corridor)
    assert audit[0]["measured_support_overlap_pixel_count"] == 4
    assert audit[0]["zero_measured_support_intersection"] is False
    assert audit[0]["accepted"] is False


def test_shelf_tracks_are_exclusive_from_direct_preseam_layers() -> None:
    mask = np.ones((4, 5), dtype=bool)

    def owner(track_id: int | None) -> InspectionForegroundIdentityOwner:
        return InspectionForegroundIdentityOwner(
            group_id=1,
            structure_id=track_id if track_id is not None else 99,
            structure_kind="test",
            identity_track_id=track_id,
            panel_index=0,
            target_panel_index=0,
            frame_id=10,
            source_index=0,
            source_mask=mask,
            target_footprint=mask,
            measured_depth_coverage_ratio=1.0,
            projected_in_bounds_ratio=1.0,
        )

    shelf_owner = owner(7)
    hierarchy_owner = owner(8)
    unrelated_direct_owner = owner(9)
    unrelated_untracked_owner = owner(None)
    retained, eligible, audit = (
        _exclude_shelf_tracks_from_direct_preseam_candidates(
            (
                shelf_owner,
                hierarchy_owner,
                unrelated_direct_owner,
                unrelated_untracked_owner,
            ),
            {7, 8, 9, 10},
            shelf_exclusive_track_ids={7, 8},
        )
    )

    assert retained == (unrelated_direct_owner, unrelated_untracked_owner)
    assert eligible == {9, 10}
    assert audit["excluded_direct_panel_native_track_ids"] == [7, 8]
    assert audit["excluded_direct_object_rich_track_ids"] == [7, 8]
    assert audit["retained_unrelated_direct_owner_count"] == 2
    assert audit["retained_unrelated_object_rich_track_ids"] == [9, 10]
    assert audit["ocr_owner_path_modified"] is False
    assert audit["cross_layer_duplicate_owner_allowed"] is False


def test_shelf_unsat_context_format_is_compact_stable_json() -> None:
    context = {
        "schema": "inspection-shelf-native-unsat-context/v1",
        "iteration": 2,
        "active_track_ids": [14, 22, 193],
        "minimal_unsat_core_track_ids": [22, 193],
        "target_corridor_bbox_xyxy": [100, 20, 180, 70],
        "target_corridor_width_pixels": 80,
        "target_corridor_area_pixels": 4000,
        "maximum_corridor_width_pixels": 1280,
        "maximum_corridor_area_pixels": 358400,
        "identity_frame_rejections": [
            {
                "frame_id": 68,
                "panel_index": 6,
                "outside_panel_valid_pixel_count": 12,
                "fixed_overlap_count": 1,
                "fixed_overlap_pixel_count": 60,
                "foreign_blocker_track_ids": [14],
                "closure_additions": [
                    {"track_id": 14, "support_pixel_count": 320}
                ],
                "veto_reason": (
                    "foreign_blocker_support_not_fully_selected_panel_valid"
                ),
            }
        ],
    }

    formatted = _format_shelf_unsat_context(context)

    assert json.loads(formatted) == context
    assert " " not in formatted
    assert formatted.startswith('{"active_track_ids":[14,22,193],')
    assert '"fixed_overlap_pixel_count":60' in formatted
    assert '"foreign_blocker_track_ids":[14]' in formatted
    assert (
        '"veto_reason":"foreign_blocker_support_not_fully_selected_panel_valid"'
        in formatted
    )


def test_fastsam_model_discovery_is_explicit_and_conflict_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured.onnx"
    environment = tmp_path / "environment.onnx"
    configured.write_bytes(b"model")
    environment.write_bytes(b"model")
    monkeypatch.setenv("G305_FASTSAM_ONNX", str(environment))

    with pytest.raises(ValueError, match="disagree"):
        _resolve_fastsam_model(
            InspectionIdentityRuntimeConfig(
                enabled=True,
                fastsam_model_path=str(configured),
            )
        )

    monkeypatch.delenv("G305_FASTSAM_ONNX")
    path, source = _resolve_fastsam_model(
        InspectionIdentityRuntimeConfig(
            enabled=True,
            fastsam_model_path=str(configured),
        )
    )
    assert path == configured.resolve()
    assert source == "configuration"


def test_fastsam_profile_requires_observed_cuda_heavy_compute(
    tmp_path: Path,
) -> None:
    passing = tmp_path / "passing.json"
    passing.write_text(
        json.dumps(
            [
                {
                    "name": "conv_kernel_time",
                    "dur": 7,
                    "args": {
                        "provider": "CUDAExecutionProvider",
                        "op_name": "Conv",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    audit = _fastsam_cuda_audit(passing)
    assert audit["pass"] is True
    assert audit["profile_retained"] is False

    failing = tmp_path / "failing.json"
    failing.write_text(
        json.dumps(
            [
                {
                    "name": "conv_kernel_time",
                    "dur": 7,
                    "args": {
                        "provider": "CPUExecutionProvider",
                        "op_name": "Conv",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    audit = _fastsam_cuda_audit(failing)
    assert audit["pass"] is False
    assert (
        "fastsam_convolution_not_executed_on_cuda"
        in audit["failures"]
    )
    assert "fastsam_heavy_operator_executed_on_cpu" in audit["failures"]


def test_proposal_filter_returns_bbox_local_exact_masks() -> None:
    proposals = [
        SimpleNamespace(
            score=0.9,
            bbox_xyxy=(0.0, 0.0, 20.0, 20.0),
            polygon_xy=np.asarray(
                [[0, 0], [20, 0], [20, 20], [0, 20]],
                dtype=np.float32,
            ),
            mask=np.ones((100, 100), dtype=bool),
        ),
        SimpleNamespace(
            score=0.8,
            bbox_xyxy=(0.0, 0.0, 99.0, 99.0),
            polygon_xy=np.asarray(
                [[0, 0], [99, 0], [99, 99], [0, 99]],
                dtype=np.float32,
            ),
            mask=np.ones((100, 100), dtype=bool),
        ),
    ]
    proposals = _proposal_polygons(
        proposals,
        image_pixels=10_000,
        config=InspectionIdentityRuntimeConfig(
            minimum_proposal_area_ratio=0.01,
            maximum_proposal_area_ratio=0.30,
        ),
    )
    assert len(proposals) == 1
    assert proposals[0].polygon_xy.shape == (4, 2)
    assert proposals[0].exact_mask_bbox.shape == (21, 21)
    assert proposals[0].exact_mask_bbox.nbytes < 100 * 100


def _shelf_native_conflict_fixture(
    *,
    common_complete_frame: bool,
    common_boundary_clear: bool = True,
) -> tuple[
    tuple[InspectionForegroundIdentityOwner, ...],
    dict[str, object],
    tuple[InspectionIdentityOwnerFrame, ...],
    InspectionMultiviewLayout,
    CameraIntrinsics,
    InspectionIdentityRuntimeConfig,
    tuple[np.ndarray, np.ndarray],
]:
    height, width = 80, 100
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=80.0,
        fy=80.0,
        cx=50.0,
        cy=40.0,
        distortion=(),
    )
    offsets = (0, 30, 60)
    layout = InspectionMultiviewLayout(
        width=160,
        height=height,
        reference_depth_mm=1000.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=index,
                anchor_scan_mm=0.0,
                canvas_offset_x=float(offset),
                center_world_mm=(0.0, 0.0, 0.0),
            )
            for index, offset in enumerate(offsets)
        ),
        panel_step_mm=100.0,
        canvas_megapixels=0.0128,
    )
    frames = []
    for panel_index, (frame_id, offset) in enumerate(
        zip((10, 20, 30), offsets, strict=True)
    ):
        valid = np.zeros((height, layout.width), dtype=bool)
        valid[:, offset : offset + width] = True
        frames.append(
            InspectionIdentityOwnerFrame(
                panel_index=panel_index,
                source_index=panel_index,
                frame_id=frame_id,
                image_bgr=np.full((height, width, 3), 128, dtype=np.uint8),
                depth_mm=np.full(
                    (height, width), 1000.0, dtype=np.float32
                ),
                reliable_depth=np.ones((height, width), dtype=bool),
                camera_to_world=np.eye(4, dtype=np.float64),
                panel_valid_mask=valid,
            )
        )

    first_panel_zero = np.zeros((height, width), dtype=bool)
    first_panel_zero[20:56, 45:76] = True
    first_panel_one = np.zeros((height, width), dtype=bool)
    first_panel_one[20:56, 20:56] = True
    second_panel_one = np.zeros((height, width), dtype=bool)
    second_panel_one[20:56, 35:71] = True
    second_panel_two = np.zeros((height, width), dtype=bool)
    second_panel_two[20:56, 10:41] = True

    def target(mask: np.ndarray, panel_index: int) -> np.ndarray:
        value = np.zeros((height, layout.width), dtype=bool)
        offset = offsets[panel_index]
        value[:, offset : offset + width] = mask
        return value

    owners = (
        InspectionForegroundIdentityOwner(
            group_id=77,
            structure_id=1,
            structure_kind="middle_yellow_shelf_stable_object_inventory",
            identity_track_id=1,
            panel_index=0,
            target_panel_index=0,
            frame_id=10,
            source_index=0,
            source_mask=first_panel_zero,
            target_footprint=target(first_panel_zero, 0),
            measured_depth_coverage_ratio=1.0,
            projected_in_bounds_ratio=1.0,
            reference_observation_masks=(
                (0, first_panel_zero),
                (1, first_panel_one),
            ),
        ),
        InspectionForegroundIdentityOwner(
            group_id=77,
            structure_id=2,
            structure_kind="middle_yellow_shelf_stable_object_inventory",
            identity_track_id=2,
            panel_index=2,
            target_panel_index=2,
            frame_id=30,
            source_index=2,
            source_mask=second_panel_two,
            target_footprint=target(second_panel_two, 2),
            measured_depth_coverage_ratio=1.0,
            projected_in_bounds_ratio=1.0,
            reference_observation_masks=(
                (1, second_panel_one),
                (2, second_panel_two),
            ),
        ),
    )

    def observation(
        frame_id: int,
        panel_index: int,
        eligible: bool,
        boundary_clear: bool = True,
    ) -> dict[str, object]:
        return {
            "frame_id": frame_id,
            "source_panel_index": panel_index,
            "eligible_complete_shelf_observation": eligible,
            "gates": {"source_boundary_clear": boundary_clear},
            "selection_rank": [
                0.2,
                0.9,
                0.9,
                1.0,
                1000.0,
                50.0,
                -float(panel_index),
            ],
        }

    planner_audit = {
        "track_dispositions": [
            {
                "track_id": 1,
                "observations": [
                    observation(10, 0, True),
                    observation(
                        20,
                        1,
                        common_complete_frame,
                        common_boundary_clear,
                    ),
                ],
            },
            {
                "track_id": 2,
                "observations": [
                    observation(
                        20,
                        1,
                        common_complete_frame,
                        common_boundary_clear,
                    ),
                    observation(30, 2, True),
                ],
            },
        ]
    }
    config = InspectionIdentityRuntimeConfig(
        panel_native_preseam_lock_enabled=True,
        panel_native_lock_guard_pixels=2,
        direct_source_boundary_margin_pixels=8,
    )
    return (
        owners,
        planner_audit,
        tuple(frames),
        layout,
        intrinsics,
        config,
        (first_panel_one, second_panel_one),
    )


def test_shelf_native_conflict_group_uses_one_complete_real_rgb_frame() -> None:
    (
        owners,
        planner_audit,
        frames,
        layout,
        intrinsics,
        config,
        common_masks,
    ) = _shelf_native_conflict_fixture(common_complete_frame=True)

    resolved, intervals, interval_audit, group_audit, corridor_tracks = (
        _resolve_shelf_native_owner_conflict_groups(
            owners,
            planner_audit,
            frames,
            layout=layout,
            intrinsics=intrinsics,
            config=config,
        )
    )

    assert {owner.frame_id for owner in resolved} == {20}
    assert {owner.panel_index for owner in resolved} == {1}
    assert {owner.source_index for owner in resolved} == {1}
    assert np.array_equal(resolved[0].source_mask, common_masks[0])
    assert np.array_equal(resolved[1].source_mask, common_masks[1])
    assert len(intervals) == 2
    assert {interval.frame_id for interval in intervals} == {20}
    transfer_masks = [
        np.asarray(interval.rgb_transfer_mask, dtype=bool)
        for interval in intervals
    ]
    assert np.any(transfer_masks[0] & transfer_masks[1])
    assert interval_audit["accepted_interval_count"] == 2
    assert (
        interval_audit[
            "final_different_real_frame_transfer_overlap_pixel_count"
        ]
        == 0
    )
    assert (
        interval_audit[
            "final_same_real_frame_transfer_overlap_pixel_count"
        ]
        > 0
    )
    assert (
        interval_audit[
            "all_transfer_overlaps_have_one_real_rgb_owner"
        ]
        is True
    )
    assert group_audit["pass"] is True
    assert group_audit["resolution_event_count"] == 1
    assert corridor_tracks == frozenset()
    event = group_audit["resolution_events"][0]
    assert event["resolution_level"] == (
        "level_1_pairwise_native_footprint_csp"
    )
    assert event["transitive_closure_constraint_used"] is False
    assert event["search_state_count_cumulative"] <= event[
        "search_state_limit"
    ]
    group = group_audit["conflict_groups"][0]
    assert group["member_track_ids"] == [1, 2]
    assert group["selected_frame_id"] == 20
    assert group["full_union_coverage_pass"] is True
    assert group["exact_member_masks_preserved"] is True
    assert group["sequential_override_used"] is False
    assert group["rgb_blended_or_generated"] is False


def test_shelf_native_conflict_group_uses_level_2_real_rgb_corridor() -> None:
    (
        owners,
        planner_audit,
        frames,
        layout,
        intrinsics,
        config,
        _,
    ) = _shelf_native_conflict_fixture(
        common_complete_frame=False,
        common_boundary_clear=True,
    )

    resolved, intervals, interval_audit, group_audit, corridor_tracks = (
        _resolve_shelf_native_owner_conflict_groups(
            owners,
            planner_audit,
            frames,
            layout=layout,
            intrinsics=intrinsics,
            config=config,
        )
    )

    assert resolved == owners
    assert corridor_tracks == frozenset({1, 2})
    assert len(intervals) == 1
    assert intervals[0].frame_id == 20
    assert intervals[0].panel_index == 1
    assert interval_audit["object_rich_corridor_interval_count"] == 1
    assert (
        interval_audit[
            "final_different_real_frame_transfer_overlap_pixel_count"
        ]
        == 0
    )
    assert group_audit["object_rich_corridor_group_count"] == 1
    event = group_audit["resolution_events"][0]
    assert event["resolution_level"] == "level_2_object_rich_corridor"
    assert event["member_track_ids"] == [1, 2]
    assert event["selected_frame_id"] == 20
    assert event["all_members_observed_and_boundary_clear"] is True
    assert event["full_panel_valid_inverse_map_coverage"] is True
    assert event["single_contiguous_real_rgb_corridor"] is True
    assert event["all_members_handled"] is True
    assert event["mesh_used"] is False
    assert event["graphcut_multiband_flow_allowed_inside"] is False
    assert event["rgb_blended_or_generated"] is False


def test_shelf_native_conflict_group_fails_without_common_boundary_frame() -> None:
    (
        owners,
        planner_audit,
        frames,
        layout,
        intrinsics,
        config,
        _,
    ) = _shelf_native_conflict_fixture(
        common_complete_frame=False,
        common_boundary_clear=False,
    )

    with pytest.raises(RuntimeError, match="no common boundary-clear"):
        _resolve_shelf_native_owner_conflict_groups(
            owners,
            planner_audit,
            frames,
            layout=layout,
            intrinsics=intrinsics,
            config=config,
        )


def test_shelf_fragment_pair_uses_small_geometry_supported_composite() -> None:
    (
        owners,
        _,
        frames,
        layout,
        intrinsics,
        config,
        _,
    ) = _shelf_native_conflict_fixture(
        common_complete_frame=False,
        common_boundary_clear=False,
    )
    second_mask = np.zeros_like(owners[1].source_mask)
    second_mask[20:56, 8:38] = True
    second_target = np.zeros((layout.height, layout.width), dtype=bool)
    second_target[:, 60:160] = second_mask
    owners = (
        replace(
            owners[0],
            reference_observation_masks=((0, owners[0].source_mask),),
        ),
        replace(
            owners[1],
            source_mask=second_mask,
            target_footprint=second_target,
            reference_observation_masks=((2, second_mask),),
        ),
    )

    def observed(frame_id: int, panel_index: int) -> dict[str, object]:
        return {
            "frame_id": frame_id,
            "source_panel_index": panel_index,
            "eligible_complete_shelf_observation": True,
            "gates": {"source_boundary_clear": True},
            "selection_rank": [
                0.2,
                0.9,
                0.9,
                1.0,
                1000.0,
                50.0,
                -float(panel_index),
            ],
        }

    planner_audit = {
        "track_dispositions": [
            {"track_id": 1, "observations": [observed(10, 0)]},
            {"track_id": 2, "observations": [observed(30, 2)]},
        ]
    }

    resolved, intervals, _, group_audit, handled = (
        _resolve_shelf_native_owner_conflict_groups(
            owners,
            planner_audit,
            frames,
            layout=layout,
            intrinsics=intrinsics,
            config=config,
        )
    )

    assert resolved == owners
    assert handled == frozenset({1, 2})
    assert len(intervals) == 1
    assert intervals[0].frame_id == 10
    assert intervals[0].panel_index == 0
    assert group_audit["fragment_composite_group_count"] == 1
    event = group_audit["resolution_events"][0]
    assert event["resolution_level"] == (
        "level_3_canonical_fragment_composite_corridor"
    )
    assert event["member_track_ids"] == [1, 2]
    assert event["no_common_observed_panel"] is True
    assert event["pairwise_csp_proven_unsat"] is True
    assert event["all_member_measured_support_retained"] is True
    assert event["full_selected_panel_valid_coverage"] is True
    assert event["suppression_semantics"].startswith(
        "tracks_alias_to_one_composite_entity"
    )


def test_no_coobservation_alone_does_not_suppress_distinct_masks() -> None:
    (
        owners,
        _,
        frames,
        layout,
        intrinsics,
        config,
        _,
    ) = _shelf_native_conflict_fixture(
        common_complete_frame=False,
        common_boundary_clear=False,
    )
    separated = np.zeros_like(owners[1].source_mask)
    separated[20:56, 35:66] = True
    separated_target = np.zeros((layout.height, layout.width), dtype=bool)
    separated_target[:, 60:160] = separated
    owners = (
        replace(
            owners[0],
            reference_observation_masks=((0, owners[0].source_mask),),
        ),
        replace(
            owners[1],
            source_mask=separated,
            target_footprint=separated_target,
            reference_observation_masks=((2, separated),),
        ),
    )
    planner_audit = {
        "track_dispositions": [
            {
                "track_id": 1,
                "observations": [
                    {
                        "frame_id": 10,
                        "eligible_complete_shelf_observation": True,
                        "gates": {"source_boundary_clear": True},
                        "selection_rank": [1.0] * 7,
                    }
                ],
            },
            {
                "track_id": 2,
                "observations": [
                    {
                        "frame_id": 30,
                        "eligible_complete_shelf_observation": True,
                        "gates": {"source_boundary_clear": True},
                        "selection_rank": [1.0] * 7,
                    }
                ],
            },
        ]
    }

    _, intervals, _, group_audit, handled = (
        _resolve_shelf_native_owner_conflict_groups(
            owners,
            planner_audit,
            frames,
            layout=layout,
            intrinsics=intrinsics,
            config=config,
        )
    )

    assert len(intervals) == 2
    assert handled == frozenset()
    assert group_audit["fragment_composite_group_count"] == 0


def test_global_csp_unsat_cycle_uses_bounded_minimal_core_corridor() -> None:
    height, width = 80, 100
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=80.0,
        fy=80.0,
        cx=50.0,
        cy=40.0,
        distortion=(),
    )
    offsets = (0, 20, 40, 57)
    frame_ids = (10, 20, 30, 40)
    layout = InspectionMultiviewLayout(
        width=157,
        height=height,
        reference_depth_mm=1000.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=index,
                anchor_scan_mm=0.0,
                canvas_offset_x=float(offset),
                center_world_mm=(0.0, 0.0, 0.0),
            )
            for index, offset in enumerate(offsets)
        ),
        panel_step_mm=100.0,
        canvas_megapixels=0.01256,
    )
    frames = []
    for panel_index, (frame_id, offset) in enumerate(
        zip(frame_ids, offsets, strict=True)
    ):
        valid = np.zeros((height, layout.width), dtype=bool)
        valid[:, offset : offset + width] = True
        frames.append(
            InspectionIdentityOwnerFrame(
                panel_index=panel_index,
                source_index=panel_index,
                frame_id=frame_id,
                image_bgr=np.full((height, width, 3), 128, dtype=np.uint8),
                depth_mm=np.full(
                    (height, width), 1000.0, dtype=np.float32
                ),
                reliable_depth=np.ones((height, width), dtype=bool),
                camera_to_world=np.eye(4, dtype=np.float64),
                panel_valid_mask=valid,
            )
        )

    def source_mask(x0: int) -> np.ndarray:
        value = np.zeros((height, width), dtype=bool)
        value[20:56, x0 : x0 + 30] = True
        return value

    masks_by_track = {
        1: {0: source_mask(35), 1: source_mask(15)},
        2: {
            0: source_mask(50),
            1: source_mask(30),
            2: source_mask(10),
        },
        3: {2: source_mask(20), 3: source_mask(8)},
        # The extra single-panel track makes the irreducible core depend on
        # the selected real source.  Whichever adjacent object is left out of
        # that core has no compatible owner outside the corridor and must be
        # absorbed by the foreign-blocker closure.
        4: {3: source_mask(8)},
    }

    def canvas(mask: np.ndarray, panel_index: int) -> np.ndarray:
        value = np.zeros((height, layout.width), dtype=bool)
        offset = offsets[panel_index]
        value[:, offset : offset + width] = mask
        return value

    selected_panels = {1: 0, 2: 1, 3: 2, 4: 3}
    owners = tuple(
        InspectionForegroundIdentityOwner(
            group_id=88,
            structure_id=track_id,
            structure_kind="middle_yellow_shelf_stable_object_inventory",
            identity_track_id=track_id,
            panel_index=selected_panels[track_id],
            target_panel_index=selected_panels[track_id],
            frame_id=frame_ids[selected_panels[track_id]],
            source_index=selected_panels[track_id],
            source_mask=masks_by_track[track_id][
                selected_panels[track_id]
            ],
            target_footprint=canvas(
                masks_by_track[track_id][selected_panels[track_id]],
                selected_panels[track_id],
            ),
            measured_depth_coverage_ratio=1.0,
            projected_in_bounds_ratio=1.0,
            reference_observation_masks=tuple(
                (panel, mask)
                for panel, mask
                in sorted(masks_by_track[track_id].items())
            ),
        )
        for track_id in (1, 2, 3, 4)
    )
    owners = (
        replace(
            owners[0],
            # Shared non-eligible reference evidence vetoes fragment aliasing
            # for the A/C edge without adding a CSP candidate.
            reference_observation_masks=(
                *owners[0].reference_observation_masks,
                (3, source_mask(8)),
            ),
        ),
        owners[1],
        owners[2],
        owners[3],
    )
    planner_audit = {
        "track_dispositions": [
            {
                "track_id": track_id,
                "observations": [
                    {
                        "frame_id": frame_ids[panel],
                        "eligible_complete_shelf_observation": True,
                        "gates": {"source_boundary_clear": True},
                        "selection_rank": [
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1000.0,
                            50.0,
                            -float(panel),
                        ],
                    }
                    for panel in sorted(masks_by_track[track_id])
                ],
            }
            for track_id in (1, 2, 3, 4)
        ]
    }
    config = InspectionIdentityRuntimeConfig(
        panel_native_preseam_lock_enabled=True,
        panel_native_lock_guard_pixels=2,
        direct_source_boundary_margin_pixels=8,
        object_rich_lock_guard_pixels=2,
    )

    _, intervals, _, audit, handled = (
        _resolve_shelf_native_owner_conflict_groups(
            owners,
            planner_audit,
            tuple(frames),
            layout=layout,
            intrinsics=intrinsics,
            config=config,
        )
    )

    # Track 1 remains a compatible native owner and is resolved separately;
    # the return value lists only tracks consumed by hard corridors.
    assert handled == frozenset({2, 3, 4})
    assert len(intervals) == 2
    event = next(
        row
        for row in audit["resolution_events"]
        if row["resolution_level"]
        == "level_4b_minimal_unsat_core_object_rich_corridor"
    )
    assert event["core_track_ids"] == [2, 4]
    assert event["closure_track_ids"] == [2, 3, 4]
    assert event["closure_fixed_point_reached"] is True
    assert event["closure_additions"] == [
        {
            "closure_iteration": 0,
            "added_track_id": 3,
            "reason": (
                "no_compatible_candidate_outside_same_real_rgb_corridor"
            ),
            "measured_target_support_pixel_count": 1080,
            "support_fully_panel_valid": True,
        }
    ]
    assert event["minimal_core_rechecked_unsat"] is True
    assert event["hard_resource_bounds_passed"] is True
    assert event["full_selected_panel_valid_coverage"] is True
    assert event["all_member_target_supports_fully_retained"] is True
    assert audit["transitive_closure_constraint_used"] is False
    assert audit["all_transfer_overlaps_have_one_real_rgb_owner"] is True


@pytest.mark.parametrize(
    "mapping",
    [
        {"maximum_frame_count": 161},
        {"maximum_proposals_per_frame": 0},
        {"maximum_identity_owner_count": 129},
        {
            "minimum_proposal_area_ratio": 0.5,
            "maximum_proposal_area_ratio": 0.4,
        },
    ],
)
def test_runtime_resource_bounds_are_closed(
    mapping: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        InspectionIdentityRuntimeConfig.from_mapping(mapping)
