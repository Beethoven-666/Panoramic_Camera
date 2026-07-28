from __future__ import annotations

from dataclasses import replace

import numpy as np

from panorama_demo.fastsam_dis_tracking import (
    FastSAMDISTrack,
    FastSAMDISTrackingResult,
)
from panorama_demo.inspection_fastsam_track import FastSAMRGBDCandidate
from panorama_demo.inspection_identity_owner_planner import (
    InspectionIdentityOwnerFrame,
    _project_structure,
    plan_direct_stable_track_identity_owners,
    plan_inspection_identity_owner_intervals,
    plan_middle_shelf_inventory_identity_owners,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
)
from panorama_demo.inspection_ocr_panel import OCRSeededPanel
from panorama_demo.session import CameraIntrinsics


def test_projected_structure_preserves_nonconvex_holes() -> None:
    intrinsics, layout, _, _, _ = _fixture()
    source = np.zeros((intrinsics.height, intrinsics.width), dtype=bool)
    source[50:111, 70:131] = True
    source[65:96, 85:116] = False
    yy, xx = np.nonzero(source)
    keep = (yy % 2 == 0) & (xx % 2 == 0)
    xx = xx[keep].astype(np.float64)
    yy = yy[keep].astype(np.float64)
    depth = np.full(xx.shape, 1000.0, dtype=np.float64)
    points = np.column_stack(
        (
            (xx - intrinsics.cx) * depth / intrinsics.fx,
            (yy - intrinsics.cy) * depth / intrinsics.fy,
            depth,
        )
    )
    panel_valid = np.zeros((layout.height, layout.width), dtype=bool)
    panel_valid[:, : intrinsics.width] = True

    projected = _project_structure(
        points,
        layout=layout,
        intrinsics=intrinsics,
        panel_index=0,
        panel_valid_mask=panel_valid,
        minimum_sample_count=30,
    )

    assert projected is not None
    assert projected.footprint[55, 75]
    assert projected.footprint[105, 125]
    assert not projected.footprint[80, 100]
    assert np.count_nonzero(projected.footprint) < 0.85 * 61 * 61


def _fixture():
    height, width = 200, 400
    intrinsics = CameraIntrinsics(
        width=width,
        height=height,
        fx=300.0,
        fy=300.0,
        cx=200.0,
        cy=100.0,
        distortion=(),
    )
    layout = InspectionMultiviewLayout(
        width=1200,
        height=height,
        reference_depth_mm=1000.0,
        scan_axis=(1.0, 0.0, 0.0),
        down_axis=(0.0, 1.0, 0.0),
        normal_axis=(0.0, 0.0, 1.0),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=index,
                anchor_scan_mm=0.0,
                canvas_offset_x=float(index * width),
                center_world_mm=(0.0, 0.0, 0.0),
            )
            for index in range(3)
        ),
        panel_step_mm=100.0,
        canvas_megapixels=0.24,
    )
    frames = []
    panels = []
    candidates: dict[int, FastSAMRGBDCandidate] = {}
    track_candidate_ids = [[], []]
    frame_ids = (10, 20, 30)
    for source_index, frame_id in enumerate(frame_ids):
        image = np.full((height, width, 3), 180, dtype=np.uint8)
        image[50:150, 50:180] = 235
        image[70:145, 190:235] = 30
        image[65:145, 245:290] = 45
        depth = np.full((height, width), 1000.0, dtype=np.float32)
        depth[50:150, 50:180] = 600.0
        depth[70:145, 190:235] = 600.0
        depth[65:145, 245:290] = 600.0
        reliable = np.ones((height, width), dtype=bool)
        panel_valid = np.zeros((height, layout.width), dtype=bool)
        x0 = source_index * width
        panel_valid[:, x0 : x0 + width] = True
        frames.append(
            InspectionIdentityOwnerFrame(
                panel_index=source_index,
                source_index=source_index,
                frame_id=frame_id,
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=np.eye(4, dtype=np.float64),
                panel_valid_mask=panel_valid,
            )
        )
        panel_mask = np.zeros((height, width), dtype=bool)
        panel_mask[50:150, 50:180] = True
        panels.append(
            OCRSeededPanel(
                frame_id=frame_id,
                source_index=source_index,
                mask=panel_mask,
                contour_xy=np.asarray(
                    [[50, 50], [179, 50], [179, 149], [50, 149]],
                    dtype=np.int32,
                ),
                bbox_xywh=(50, 50, 130, 100),
                world_points_mm=np.asarray(
                    [[0.0, 0.0, 600.0], [100.0, 50.0, 600.0]]
                ),
                world_centroid_mm=(0.0, 0.0, 600.0),
                world_extent_pca_mm=(260.0, 180.0),
                median_lab=(235.0, 128.0, 128.0),
                aspect_ratio=2.6,
                rectangularity=0.95,
                solidity=0.98,
                clarity_variance=200.0 + source_index,
                audit={"depth_coverage_ratio": 1.0, "pass": True},
            )
        )
        for track_index, (x1, y1, x2, y2, lightness) in enumerate(
            (
                (190, 70, 235, 145, 35.0),
                (245, 65, 290, 145, 50.0),
            )
        ):
            candidate_id = 100 * track_index + source_index
            polygon = np.asarray(
                [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]],
                dtype=np.int32,
            )
            candidates[candidate_id] = FastSAMRGBDCandidate(
                candidate_id=candidate_id,
                source_index=source_index,
                frame_id=frame_id,
                polygon_xy=polygon,
                bbox_xywh=(x1, y1, x2 - x1, y2 - y1),
                source_area_pixels=(x2 - x1) * (y2 - y1),
                depth_coverage_ratio=1.0,
                world_voxel_hashes=frozenset(),
                world_dilated_voxel_hashes=frozenset(),
                world_centroid_mm=(0.0, 0.0, 600.0),
                world_spans_mm=(80.0, 100.0, 1.0),
                median_lab=(lightness, 128.0, 128.0),
                aspect_ratio=(x2 - x1) / (y2 - y1),
                solidity=1.0,
            )
            track_candidate_ids[track_index].append(candidate_id)
    tracks = tuple(
        FastSAMDISTrack(
            track_id=700 + track_index,
            candidate_ids=tuple(ids),
            frame_ids=frame_ids,
            observation_count=3,
            stable_candidate_ids=tuple(ids),
            stable_frame_ids=frame_ids,
            maximum_area_ratio=1.0,
            minimum_flow_mask_iou=0.9,
            maximum_fb_p95_preview_pixels=0.2,
            merge_split_terminated=False,
        )
        for track_index, ids in enumerate(track_candidate_ids)
    )
    tracking = FastSAMDISTrackingResult(
        frames=(),
        tracks=tracks,
        stable_tracks=tracks,
        pair_audits=(),
        candidate_by_id=candidates,
    )
    return intrinsics, layout, tuple(frames), tracking, tuple(panels)


def test_planner_emits_one_real_panel_row_contiguous_owner_interval() -> None:
    intrinsics, layout, frames, tracking, panels = _fixture()
    image_before = [frame.image_bgr.copy() for frame in frames]
    depth_before = [frame.depth_mm.copy() for frame in frames]
    pose_before = [frame.camera_to_world.copy() for frame in frames]

    result = plan_inspection_identity_owner_intervals(
        frames=frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
        ocr_seeded_panels=panels,
    )

    assert result.audit["pass"] is True
    assert result.audit["delivery_grade_ceiling"] == "C"
    assert result.audit["handoff_outcome"] == "hard_cut_degraded"
    assert len(result.intervals) == 1
    assert len(result.foreground_owners) == 3
    interval = result.intervals[0]
    selected_frame = next(
        frame for frame in frames if frame.frame_id == interval.frame_id
    )
    assert interval.panel_index == selected_frame.panel_index
    assert interval.lock_mask.dtype == np.bool_
    assert interval.union_footprint.dtype == np.bool_
    assert np.all(~interval.union_footprint | interval.lock_mask)
    assert np.all(~interval.lock_mask | selected_frame.panel_valid_mask)
    for row in np.flatnonzero(np.any(interval.lock_mask, axis=1)):
        columns = np.flatnonzero(interval.lock_mask[row])
        assert columns.size == columns[-1] - columns[0] + 1
    group_ids = {owner.group_id for owner in result.foreground_owners}
    assert group_ids == {interval.track_id}
    assert [owner.structure_id for owner in result.foreground_owners] == [
        0,
        1,
        2,
    ]
    assert [
        owner.structure_kind for owner in result.foreground_owners
    ] == [
        "ocr_seeded_panel",
        "fastsam_stable_dis_track",
        "fastsam_stable_dis_track",
    ]
    for owner in result.foreground_owners:
        assert owner.panel_index == interval.panel_index
        assert owner.frame_id == interval.frame_id
        assert owner.source_index == selected_frame.source_index
        assert owner.source_mask.shape == (
            intrinsics.height,
            intrinsics.width,
        )
        assert owner.source_mask.dtype == np.bool_
        assert owner.target_footprint.shape == (
            layout.height,
            layout.width,
        )
        assert owner.target_footprint.dtype == np.bool_
        assert np.any(owner.source_mask)
        assert np.any(owner.target_footprint)
        assert np.all(
            ~owner.target_footprint | selected_frame.panel_valid_mask
        )
        assert owner.measured_depth_coverage_ratio >= 0.85
        assert owner.projected_in_bounds_ratio >= 0.90
    for first_index, first in enumerate(result.foreground_owners):
        for second in result.foreground_owners[first_index + 1 :]:
            assert not np.any(
                first.target_footprint & second.target_footprint
            )
            assert not np.any(first.source_mask & second.source_mask)
    independent_union = np.logical_or.reduce(
        [owner.target_footprint for owner in result.foreground_owners]
    )
    selected_row = 100
    structure_columns = np.flatnonzero(independent_union[selected_row])
    assert np.any(
        ~independent_union[
            selected_row,
            structure_columns[0] : structure_columns[-1] + 1,
        ]
    )
    assert np.all(
        interval.lock_mask[
            selected_row,
            structure_columns[0] : structure_columns[-1] + 1,
        ]
    )
    for frame, image, depth, pose in zip(
        frames, image_before, depth_before, pose_before, strict=True
    ):
        assert np.array_equal(frame.image_bgr, image)
        assert np.array_equal(frame.depth_mm, depth)
        assert np.array_equal(frame.camera_to_world, pose)


def test_middle_shelf_inventory_keeps_adjacent_objects_and_removes_hierarchy(
) -> None:
    intrinsics, layout, original_frames, original_tracking, _ = _fixture()
    frames = []
    for frame in original_frames:
        image = frame.image_bgr.copy()
        image[145:, :] = (0, 200, 255)
        frames.append(replace(frame, image_bgr=image))

    candidates = dict(original_tracking.candidate_by_id)
    nested_candidate_ids: list[int] = []
    for source_index, frame_id in enumerate((10, 20, 30)):
        candidate_id = 900 + source_index
        nested_candidate_ids.append(candidate_id)
        polygon = np.asarray(
            [[200, 85], [219, 85], [219, 129], [200, 129]],
            dtype=np.int32,
        )
        candidates[candidate_id] = FastSAMRGBDCandidate(
            candidate_id=candidate_id,
            source_index=source_index,
            frame_id=frame_id,
            polygon_xy=polygon,
            bbox_xywh=(200, 85, 20, 45),
            source_area_pixels=20 * 45,
            depth_coverage_ratio=1.0,
            world_voxel_hashes=frozenset(),
            world_dilated_voxel_hashes=frozenset(),
            world_centroid_mm=(0.0, 0.0, 600.0),
            world_spans_mm=(20.0, 45.0, 1.0),
            median_lab=(35.0, 128.0, 128.0),
            aspect_ratio=20 / 45,
            solidity=1.0,
        )
    nested = FastSAMDISTrack(
        track_id=999,
        candidate_ids=tuple(nested_candidate_ids),
        frame_ids=(10, 20, 30),
        observation_count=3,
        stable_candidate_ids=tuple(nested_candidate_ids),
        stable_frame_ids=(10, 20, 30),
        maximum_area_ratio=1.0,
        minimum_flow_mask_iou=0.9,
        maximum_fb_p95_preview_pixels=0.2,
        merge_split_terminated=False,
    )
    tracks = (*original_tracking.stable_tracks, nested)
    tracking = replace(
        original_tracking,
        tracks=tracks,
        stable_tracks=tracks,
        candidate_by_id=candidates,
    )

    result = plan_middle_shelf_inventory_identity_owners(
        frames=tuple(frames),
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
    )

    assert result.audit["reference_rgb_or_geometry_used"] is False
    assert result.audit["track_ids_hardcoded"] is False
    assert result.audit["all_stable_tracks_have_disposition"] is True
    assert result.audit["inventory_owner_candidate_count"] == 2
    assert result.audit["hierarchy_duplicate_count"] == 1
    assert {owner.identity_track_id for owner in result.foreground_owners} == {
        700,
        701,
    }
    assert all(
        owner.structure_kind
        == "middle_yellow_shelf_stable_object_inventory"
        for owner in result.foreground_owners
    )
    dispositions = {
        row["track_id"]: row for row in result.audit["track_dispositions"]
    }
    assert dispositions[999]["inventory_disposition"] == (
        "excluded_fastsam_hierarchy_duplicate"
    )
    assert dispositions[999]["hierarchy_parent_track_id"] == 700
    assert dispositions[700]["mesh_preflight_required"] is True
    assert dispositions[701]["mesh_preflight_required"] is True


def test_middle_shelf_inventory_requires_measured_yellow_shelf_contact() -> None:
    intrinsics, layout, frames, tracking, _ = _fixture()

    result = plan_middle_shelf_inventory_identity_owners(
        frames=frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
    )

    assert result.foreground_owners == ()
    assert result.audit["inventory_owner_candidate_count"] == 0
    assert result.audit["all_stable_tracks_have_disposition"] is True
    assert {
        row["inventory_disposition"]
        for row in result.audit["track_dispositions"]
    } == {"excluded_not_complete_shelf_object"}


def test_planner_without_ocr_anchor_is_empty_and_fail_closed() -> None:
    intrinsics, layout, frames, tracking, _ = _fixture()
    result = plan_inspection_identity_owner_intervals(
        frames=frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
    )
    assert result.intervals == ()
    assert result.foreground_owners == ()
    assert result.audit["pass"] is False
    assert (
        result.audit["rejection_reason"]
        == "stable_ocr_seeded_panel_anchor_unavailable"
    )


def test_planner_rejects_corridor_not_covered_by_selected_real_panel() -> None:
    intrinsics, layout, frames, tracking, panels = _fixture()
    invalid_frames = []
    for frame in frames:
        valid = frame.panel_valid_mask.copy()
        x0 = frame.panel_index * intrinsics.width
        valid[:, x0 + 220 : x0 + 225] = False
        invalid_frames.append(replace(frame, panel_valid_mask=valid))
    result = plan_inspection_identity_owner_intervals(
        frames=tuple(invalid_frames),
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
        ocr_seeded_panels=panels,
    )
    assert result.intervals == ()
    assert result.foreground_owners == ()
    assert result.audit["pass"] is False
    assert (
        result.audit["rejection_reason"]
        == "no_complete_single_real_panel_owner_corridor"
    )


def test_planner_rejects_non_rigid_pose_before_planning() -> None:
    intrinsics, layout, frames, tracking, panels = _fixture()
    bad_pose = frames[0].camera_to_world.copy()
    bad_pose[0, 0] = 2.0
    bad_frames = (replace(frames[0], camera_to_world=bad_pose), *frames[1:])
    try:
        plan_inspection_identity_owner_intervals(
            frames=bad_frames,
            tracking=tracking,
            layout=layout,
            intrinsics=intrinsics,
            ocr_seeded_panels=panels,
        )
    except ValueError as error:
        assert "rigid SE(3)" in str(error)
    else:
        raise AssertionError("Non-rigid pose was accepted")


def test_direct_track_planner_without_ocr_uses_median_target_panel() -> None:
    intrinsics, layout, frames, tracking, _ = _fixture()
    image_before = [frame.image_bgr.copy() for frame in frames]
    pose_before = [frame.camera_to_world.copy() for frame in frames]
    result = plan_direct_stable_track_identity_owners(
        frames=frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
    )
    assert result.audit["pass"] is True
    assert sorted(result.audit["accepted_track_ids"]) == [700, 701]
    assert result.audit["ranking"] == (
        "consistent_projection_count_then_selected_target_area_"
        "then_union_coverage"
    )
    assert len(result.foreground_owners) == 2
    for owner in result.foreground_owners:
        assert owner.structure_kind == "fastsam_stable_dis_track_direct"
        assert owner.target_panel_index == 1
        assert owner.panel_index == 0
        assert owner.frame_id == 10
        assert owner.source_index == 0
        assert owner.measured_depth_coverage_ratio >= 0.90
        assert owner.projected_in_bounds_ratio >= 0.90
        assert owner.source_mask.shape == (intrinsics.height, intrinsics.width)
        assert owner.target_footprint.shape == (
            layout.height,
            layout.width,
        )
        target_frame = next(
            frame
            for frame in frames
            if frame.panel_index == owner.target_panel_index
        )
        assert np.all(
            ~owner.target_footprint | target_frame.panel_valid_mask
        )
    for frame, image, pose in zip(
        frames, image_before, pose_before, strict=True
    ):
        assert np.array_equal(frame.image_bgr, image)
        assert np.array_equal(frame.camera_to_world, pose)
    audits = {
        int(item["track_id"]): item for item in result.audit["track_audits"]
    }
    assert audits[700]["consistent_projection_count"] == 3
    assert audits[700]["selected_target_union_coverage_ratio"] == 1.0
    assert audits[700]["translation_used"] is False
    assert audits[700]["pose_interpolation_used"] is False


def test_direct_track_planner_rejects_overlap_with_existing_ocr_owner() -> None:
    intrinsics, layout, frames, tracking, _ = _fixture()
    baseline = plan_direct_stable_track_identity_owners(
        frames=frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
    )
    existing = replace(
        baseline.foreground_owners[0],
        group_id=9999,
        structure_kind="ocr_seeded_panel",
        identity_track_id=None,
    )
    existing_track_id = baseline.foreground_owners[0].identity_track_id
    result = plan_direct_stable_track_identity_owners(
        frames=frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
        existing_foreground_owners=(existing,),
    )
    assert {
        owner.identity_track_id for owner in result.foreground_owners
    } == ({700, 701} - {existing_track_id})
    assert len(result.audit["conflict_rejections"]) == 1
    conflict = result.audit["conflict_rejections"][0]
    assert conflict["track_id"] == existing_track_id
    assert conflict["identity_owner_overlap_ratio"] > 0.15
    assert conflict["reason"] == (
        "direct_target_overlaps_existing_or_prior_identity_owner"
    )


def test_direct_track_planner_rejects_later_overlapping_track() -> None:
    intrinsics, layout, frames, tracking, _ = _fixture()
    source_track = tracking.stable_tracks[0]
    candidate_by_id = dict(tracking.candidate_by_id)
    duplicate_ids = []
    for offset, candidate_id in enumerate(source_track.candidate_ids):
        duplicate_id = 900 + offset
        duplicate_ids.append(duplicate_id)
        candidate_by_id[duplicate_id] = replace(
            candidate_by_id[candidate_id],
            candidate_id=duplicate_id,
        )
    duplicate = replace(
        source_track,
        track_id=999,
        candidate_ids=tuple(duplicate_ids),
        stable_candidate_ids=tuple(duplicate_ids),
    )
    overlapping_tracking = replace(
        tracking,
        tracks=(*tracking.tracks, duplicate),
        stable_tracks=(*tracking.stable_tracks, duplicate),
        candidate_by_id=candidate_by_id,
    )
    result = plan_direct_stable_track_identity_owners(
        frames=frames,
        tracking=overlapping_tracking,
        layout=layout,
        intrinsics=intrinsics,
    )
    assert {
        owner.identity_track_id for owner in result.foreground_owners
    } == {700, 701}
    conflicts = result.audit["conflict_rejections"]
    assert len(conflicts) == 1
    assert conflicts[0]["track_id"] == 999
    assert conflicts[0]["identity_owner_overlap_ratio"] > 0.15


def test_direct_track_planner_recomputes_point_nine_depth_gate() -> None:
    intrinsics, layout, frames, tracking, _ = _fixture()
    damaged = []
    for frame in frames:
        reliable = frame.reliable_depth.copy()
        reliable[70:145, 190:200] = False
        damaged.append(replace(frame, reliable_depth=reliable))
    result = plan_direct_stable_track_identity_owners(
        frames=tuple(damaged),
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
    )
    assert [owner.identity_track_id for owner in result.foreground_owners] == [
        701
    ]
    audits = {
        int(item["track_id"]): item for item in result.audit["track_audits"]
    }
    assert audits[700]["accepted"] is False
    assert all(
        item["source_depth_coverage_ratio"] < 0.90
        for item in audits[700]["source_observations"]
    )
