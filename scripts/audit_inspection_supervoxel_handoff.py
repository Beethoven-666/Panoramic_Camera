"""One-shot all-frame RGB-D supervoxel object-handoff diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.cuda_backend import pinhole_unproject, transform_points
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
    _read_rgbd,
    _undistortion_maps,
)
from panorama_demo.inspection_object_handoff import (
    build_object_owner_interval,
    project_complete_object_owner_from_rgbd,
)
from panorama_demo.inspection_supervoxel_track import (
    segment_world_supervoxels,
)
from panorama_demo.session import load_rgbd_session


def _layout_from_report(value: dict[str, object]) -> InspectionMultiviewLayout:
    return InspectionMultiviewLayout(
        width=int(value["width"]),
        height=int(value["height"]),
        reference_depth_mm=float(value["reference_depth_mm"]),
        scan_axis=tuple(float(item) for item in value["scan_axis_world"]),
        down_axis=tuple(float(item) for item in value["down_axis_world"]),
        normal_axis=tuple(float(item) for item in value["normal_axis_world"]),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=int(item["panel_index"]),
                anchor_scan_mm=float(item["anchor_scan_mm"]),
                canvas_offset_x=float(item["canvas_offset_x"]),
                center_world_mm=tuple(
                    float(component) for component in item["center_world_mm"]
                ),
            )
            for item in value["panels"]
        ),
        panel_step_mm=float(value["panel_step_mm"]),
        canvas_megapixels=float(value["canvas_megapixels"]),
    )


def _full_baseline(
    output: Path,
    layout: InspectionMultiviewLayout,
    crop: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(
        str(output / "mosaic_inspection.png"), cv2.IMREAD_COLOR
    )
    encoded = cv2.imread(
        str(output / "inspection_owner.png"), cv2.IMREAD_UNCHANGED
    )
    if (
        image is None
        or encoded is None
        or encoded.dtype != np.uint16
        or image.shape[:2] != encoded.shape
    ):
        raise RuntimeError("Formal inspection baseline is invalid")
    x, y, width, height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    full_image = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    full_owner = np.full((layout.height, layout.width), -1, dtype=np.int32)
    full_valid = np.zeros((layout.height, layout.width), dtype=bool)
    full_image[y : y + height, x : x + width] = image
    owner = encoded.astype(np.int32) - 1
    full_owner[y : y + height, x : x + width] = owner
    full_valid[y : y + height, x : x + width] = owner >= 0
    return full_image, full_owner, full_valid


def _sample_source(
    *,
    image: np.ndarray,
    depth: np.ndarray,
    reliable: np.ndarray,
    pose: np.ndarray,
    intrinsics: object,
    reference_depth_mm: float,
    stride: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    margin = max(35.0, 0.04 * reference_depth_mm)
    yy_grid, xx_grid = np.meshgrid(
        np.arange(2, depth.shape[0] - 2, stride, dtype=np.int32),
        np.arange(2, depth.shape[1] - 2, stride, dtype=np.int32),
        indexing="ij",
    )
    yy = yy_grid.reshape(-1)
    xx = xx_grid.reshape(-1)
    center_depth = depth[yy, xx]
    valid = (
        reliable[yy, xx]
        & np.isfinite(center_depth)
        & (center_depth < np.float32(reference_depth_mm - margin))
    )
    yy = yy[valid]
    xx = xx[valid]
    center_depth = center_depth[valid]
    if xx.size == 0:
        empty_points = np.empty((0, 3), dtype=np.float64)
        return (
            empty_points,
            np.empty((0, 3), dtype=np.float32),
            empty_points.copy(),
            np.empty(0, dtype=bool),
            np.empty((0, 2), dtype=np.int16),
        )
    camera = pinhole_unproject(
        xx,
        yy,
        center_depth,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    world = transform_points(camera, pose[:3, :3], pose[:3, 3])
    right_depth = depth[yy, xx + 2]
    down_depth = depth[yy + 2, xx]
    tolerance = np.maximum(20.0, 0.02 * center_depth)
    normal_valid = (
        reliable[yy, xx + 2]
        & reliable[yy + 2, xx]
        & np.isfinite(right_depth)
        & np.isfinite(down_depth)
        & (np.abs(right_depth - center_depth) <= tolerance)
        & (np.abs(down_depth - center_depth) <= tolerance)
    )
    right = pinhole_unproject(
        xx + 2,
        yy,
        right_depth,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    down = pinhole_unproject(
        xx,
        yy + 2,
        down_depth,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    normals = np.cross(right - camera, down - camera)
    length = np.linalg.norm(normals, axis=1)
    normal_valid &= np.isfinite(length) & (length > 1e-8)
    normals[normal_valid] /= length[normal_valid, None]
    normals[~normal_valid] = 0.0
    normals_world = normals @ pose[:3, :3].T
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab = lab_image[yy, xx].astype(np.float32)
    return (
        np.ascontiguousarray(world),
        np.ascontiguousarray(lab),
        np.ascontiguousarray(normals_world),
        np.ascontiguousarray(normal_valid),
        np.column_stack((xx, yy)).astype(np.int16),
    )


def _project_world_to_panel(
    points: np.ndarray,
    layout: InspectionMultiviewLayout,
    intrinsics: object,
    panel_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    panel = layout.panels[panel_index]
    relative = points - np.asarray(panel.center_world_mm, dtype=np.float64)
    scan = relative @ np.asarray(layout.scan_axis, dtype=np.float64)
    down = relative @ np.asarray(layout.down_axis, dtype=np.float64)
    normal = relative @ np.asarray(layout.normal_axis, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (
            panel.canvas_offset_x
            + intrinsics.cx
            + intrinsics.fx * scan / normal
        )
        y = intrinsics.cy + intrinsics.fy * down / normal
    return x, y, normal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("formal_output")
    parser.add_argument("--sample-stride", type=int, default=8)
    arguments = parser.parse_args()
    started = time.perf_counter()
    session_path = Path(arguments.session).expanduser().resolve()
    output = Path(arguments.formal_output).expanduser().resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    render = report["render"]
    layout = _layout_from_report(render["layout"])
    config = InspectionMultiviewConfig.from_mapping(render["config"])
    session = load_rgbd_session(session_path)
    frame_by_id = {int(frame.frame_id): frame for frame in session.frames}
    frame_ids = [int(value) for value in render["frame_ids"]]
    frames = [frame_by_id[value] for value in frame_ids]
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    poses = [pose_by_id[value] for value in frame_ids]
    full_image, full_owner, full_valid = _full_baseline(
        output, layout, render["crop"]
    )
    maps = _undistortion_maps(session.calibration)
    world_parts: list[np.ndarray] = []
    lab_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    normal_valid_parts: list[np.ndarray] = []
    source_parts: list[np.ndarray] = []
    pixel_parts: list[np.ndarray] = []
    for source_index, (frame, pose) in enumerate(zip(frames, poses, strict=True)):
        image, depth, geometric_valid = _read_rgbd(
            frame, session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        world, lab, normals, normal_valid, pixels = _sample_source(
            image=image,
            depth=depth,
            reliable=reliable,
            pose=pose,
            intrinsics=session.calibration,
            reference_depth_mm=layout.reference_depth_mm,
            stride=int(arguments.sample_stride),
        )
        if world.size:
            world_parts.append(world)
            lab_parts.append(lab)
            normal_parts.append(normals)
            normal_valid_parts.append(normal_valid)
            source_parts.append(
                np.full(world.shape[0], source_index, dtype=np.int16)
            )
            pixel_parts.append(pixels)
        if (source_index + 1) % 24 == 0:
            print(
                f"sampled {source_index + 1}/{len(frames)} real RGB-D frames",
                flush=True,
            )
    raw_world = np.concatenate(world_parts)
    raw_lab = np.concatenate(lab_parts)
    raw_normals = np.concatenate(normal_parts)
    raw_normal_valid = np.concatenate(normal_valid_parts)
    raw_source = np.concatenate(source_parts)
    raw_pixels = np.concatenate(pixel_parts)
    segmentation = segment_world_supervoxels(
        points_world_mm=raw_world,
        lab=raw_lab,
        normals_world=raw_normals,
        normal_valid=raw_normal_valid,
    )

    cache: OrderedDict[
        int, tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = OrderedDict()

    def load_source(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index in cache:
            cache.move_to_end(index)
            return cache[index]
        image, depth, geometric_valid = _read_rgbd(
            frames[index], session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        cache[index] = (image, depth, reliable)
        while len(cache) > 8:
            cache.popitem(last=False)
        return cache[index]

    valid_track = segmentation.sample_track_id >= 0
    order = np.argsort(
        segmentation.sample_track_id[valid_track], kind="stable"
    )
    valid_indices = np.flatnonzero(valid_track)[order]
    ordered_track = segmentation.sample_track_id[valid_indices]
    starts = np.flatnonzero(
        np.r_[True, ordered_track[1:] != ordered_track[:-1]]
    )
    ends = np.r_[starts[1:], len(valid_indices)]
    anchors_scan = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels],
        dtype=np.float64,
    )
    diagnostic = full_image.copy()
    accepted_mask = np.zeros(full_valid.shape, dtype=bool)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    candidate_count = 0
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    down_axis = np.asarray(layout.down_axis, dtype=np.float64)
    normal_axis = np.asarray(layout.normal_axis, dtype=np.float64)
    for start, end in zip(starts, ends, strict=True):
        indices = valid_indices[start:end]
        if indices.size < 30:
            continue
        source_ids, source_counts = np.unique(
            raw_source[indices], return_counts=True
        )
        if source_ids.size < 2:
            continue
        points = raw_world[indices]
        basis = np.column_stack(
            (
                points @ scan_axis,
                points @ down_axis,
                points @ normal_axis,
            )
        )
        spans = np.ptp(basis, axis=0)
        if (
            np.any(spans > np.asarray([450.0, 450.0, 350.0]))
            or np.count_nonzero(spans > 12.0) < 2
        ):
            continue
        panel_index = int(
            np.argmin(
                np.abs(anchors_scan - float(np.median(basis[:, 0])))
            )
        )
        approx_x, approx_y, approx_z = _project_world_to_panel(
            points, layout, session.calibration, panel_index
        )
        finite = (
            np.isfinite(approx_x)
            & np.isfinite(approx_y)
            & np.isfinite(approx_z)
            & (approx_z > 0.0)
        )
        if np.count_nonzero(finite) < 30:
            continue
        approx_mask = np.zeros(full_valid.shape, dtype=np.uint8)
        px = np.rint(approx_x[finite]).astype(np.int32)
        py = np.rint(approx_y[finite]).astype(np.int32)
        inside = (
            (px >= 0)
            & (px < layout.width)
            & (py >= 0)
            & (py < layout.height)
        )
        approx_mask[py[inside], px[inside]] = 1
        approx_mask = cv2.dilate(
            approx_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (
                    int(arguments.sample_stride) * 2 + 1,
                    int(arguments.sample_stride) * 2 + 1,
                ),
            ),
        )
        baseline_owners = np.unique(
            full_owner[(approx_mask > 0) & full_valid]
        )
        baseline_owners = baseline_owners[baseline_owners >= 0]
        if baseline_owners.size < 2:
            continue
        candidate_count += 1
        track_id = int(ordered_track[start])
        group_min = np.min(basis, axis=0) - 18.0
        group_max = np.max(basis, axis=0) + 18.0
        ranked_sources = [
            int(value)
            for value in source_ids[
                np.argsort(-source_counts, kind="stable")
            ][:4]
        ]
        projections = []
        source_audits: list[dict[str, object]] = []
        for source_index in ranked_sources:
            source_indices = indices[
                raw_source[indices] == source_index
            ]
            if source_indices.size < 18:
                continue
            image, depth, reliable = load_source(source_index)
            seed = np.zeros(depth.shape, dtype=np.uint8)
            source_xy = raw_pixels[source_indices]
            seed[source_xy[:, 1], source_xy[:, 0]] = 1
            candidate = cv2.dilate(
                seed,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (
                        int(arguments.sample_stride) * 2 + 1,
                        int(arguments.sample_stride) * 2 + 1,
                    ),
                ),
            )
            candidate = (candidate > 0) & reliable & (
                depth
                < np.float32(
                    layout.reference_depth_mm
                    - max(35.0, 0.04 * layout.reference_depth_mm)
                )
            )
            yy, xx = np.nonzero(candidate)
            if xx.size < 64:
                continue
            camera = pinhole_unproject(
                xx,
                yy,
                depth[yy, xx],
                fx=session.calibration.fx,
                fy=session.calibration.fy,
                cx=session.calibration.cx,
                cy=session.calibration.cy,
            )
            world = transform_points(
                camera,
                poses[source_index][:3, :3],
                poses[source_index][:3, 3],
            )
            world_basis = np.column_stack(
                (
                    world @ scan_axis,
                    world @ down_axis,
                    world @ normal_axis,
                )
            )
            within = np.all(
                (world_basis >= group_min) & (world_basis <= group_max),
                axis=1,
            )
            candidate[:] = False
            candidate[yy[within], xx[within]] = True
            candidate = cv2.morphologyEx(
                candidate.astype(np.uint8),
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ).astype(bool)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                candidate.astype(np.uint8), 8
            )
            best_label = 0
            best_support = 0
            for label in range(1, count):
                support = int(
                    np.count_nonzero((labels == label) & (seed > 0))
                )
                if support > best_support:
                    best_label = label
                    best_support = support
            if best_label == 0 or best_support < 18:
                continue
            source_mask = labels == best_label
            try:
                owner = project_complete_object_owner_from_rgbd(
                    source_image_bgr=image,
                    source_depth_mm=depth,
                    source_reliable_depth=reliable,
                    source_object_mask=source_mask,
                    camera_to_world=poses[source_index],
                    layout=layout,
                    intrinsics=session.calibration,
                    frame_id=int(frames[source_index].frame_id),
                    panel_index=panel_index,
                    minimum_cells=64,
                )
            except (RuntimeError, ValueError) as exc:
                source_audits.append(
                    {
                        "frame_id": int(frames[source_index].frame_id),
                        "reason": str(exc),
                    }
                )
                continue
            projections.append((source_index, source_mask, owner))
            source_audits.append(
                {
                    "frame_id": int(frames[source_index].frame_id),
                    "source_mask_pixel_count": int(
                        np.count_nonzero(source_mask)
                    ),
                    "target_pixel_count": int(
                        np.count_nonzero(owner.target_mask)
                    ),
                    "accepted": True,
                }
            )
        rejection_base = {
            "track_id": track_id,
            "raw_sample_count": int(indices.size),
            "world_source_support_count": int(source_ids.size),
            "world_spans_mm": [float(value) for value in spans],
            "selected_panel_index": panel_index,
            "baseline_owner_frame_ids": [
                int(value) for value in baseline_owners
            ],
            "source_audits": source_audits,
        }
        if len(projections) < 2:
            rejected.append(
                {
                    **rejection_base,
                    "reason": "fewer_than_two_complete_direct_rgbd_owners",
                }
            )
            continue
        best = None
        for candidate_index, candidate_item in enumerate(projections):
            candidate_target = candidate_item[2].target_mask
            consistent = [candidate_index]
            for other_index, other_item in enumerate(projections):
                if other_index == candidate_index:
                    continue
                intersection = int(
                    np.count_nonzero(
                        candidate_target & other_item[2].target_mask
                    )
                )
                union = int(
                    np.count_nonzero(
                        candidate_target | other_item[2].target_mask
                    )
                )
                if union and intersection / union >= 0.30:
                    consistent.append(other_index)
            if len(consistent) < 2:
                continue
            footprint_union = np.logical_or.reduce(
                [projections[index][2].target_mask for index in consistent]
            )
            coverage = float(
                np.count_nonzero(candidate_target & footprint_union)
                / max(1, np.count_nonzero(footprint_union))
            )
            score = (
                coverage,
                len(consistent),
                int(np.count_nonzero(candidate_target)),
                -int(candidate_item[2].frame_id),
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    candidate_item,
                    consistent,
                    footprint_union,
                )
        if best is None or best[0][0] < 0.95:
            rejected.append(
                {
                    **rejection_base,
                    "reason": (
                        "cross_view_direct_footprints_inconsistent_or_"
                        "selected_union_coverage_below_0_95"
                    ),
                }
            )
            continue
        _, selected, consistent, footprint_union = best
        interval = build_object_owner_interval(
            panel_index=panel_index,
            view_dependent_footprints=tuple(
                projections[index][2].target_mask for index in consistent
            ),
            selected_panel_valid_mask=full_valid,
        )
        owner = selected[2]
        interval_coverage = float(
            np.count_nonzero(owner.target_mask & interval.union_footprint)
            / max(1, np.count_nonzero(interval.union_footprint))
        )
        overlap = int(np.count_nonzero(owner.target_mask & accepted_mask))
        if interval_coverage < 0.95 or overlap:
            rejected.append(
                {
                    **rejection_base,
                    "reason": (
                        "selected_owner_does_not_cover_all_view_footprints_"
                        "or_overlaps_another_track"
                    ),
                    "interval_union_coverage_ratio": interval_coverage,
                    "accepted_track_overlap_pixel_count": overlap,
                }
            )
            continue
        diagnostic[owner.target_mask] = owner.target_image_bgr[
            owner.target_mask
        ]
        accepted_mask |= owner.target_mask
        target_y, target_x = np.nonzero(owner.target_mask)
        accepted.append(
            {
                **rejection_base,
                "selected_frame_id": int(owner.frame_id),
                "cross_view_complete_owner_count": len(consistent),
                "selected_cross_view_union_coverage_ratio": float(best[0][0]),
                "interval_union_coverage_ratio": interval_coverage,
                "target_pixel_count": int(
                    np.count_nonzero(owner.target_mask)
                ),
                "target_bbox_xywh": [
                    int(np.min(target_x)),
                    int(np.min(target_y)),
                    int(np.max(target_x) - np.min(target_x) + 1),
                    int(np.max(target_y) - np.min(target_y) + 1),
                ],
                "single_rgb_owner": True,
                "pose_modified": False,
                "rgb_generated": False,
                "blend_used": False,
                "interval_audit": interval.audit,
            }
        )

    crop = render["crop"]
    crop_x, crop_y, crop_width, crop_height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    diagnostic_crop = diagnostic[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    comparison = np.hstack(
        (
            full_image[
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
            ],
            diagnostic_crop,
        )
    )
    for item in accepted:
        x, y, width, height = item["target_bbox_xywh"]
        cv2.rectangle(
            comparison,
            (x - crop_x + crop_width, y - crop_y),
            (
                x + width - 1 - crop_x + crop_width,
                y + height - 1 - crop_y,
            ),
            (0, 255, 0),
            2,
        )
    image_path = output / "diagnostic_supervoxel_object_handoff.png"
    comparison_path = (
        output / "diagnostic_supervoxel_object_handoff_before_after.jpg"
    )
    audit_path = (
        output / "diagnostic_supervoxel_object_handoff_audit.json"
    )
    if not cv2.imwrite(str(image_path), diagnostic_crop):
        raise RuntimeError("Could not write supervoxel handoff diagnostic")
    if not cv2.imwrite(str(comparison_path), comparison):
        raise RuntimeError("Could not write supervoxel comparison")
    audit = {
        "schema": "inspection-supervoxel-object-handoff-diagnostic/v1",
        "formal_output_modified": False,
        "formal_acceptance": False,
        "formal_acceptance_reason": (
            "bounded_first_prototype_not_connected_to_formal_owner_chain"
        ),
        "frame_count": len(frames),
        "all_real_rgbd_frames_sampled": len(frames) == 145,
        "sample_stride": int(arguments.sample_stride),
        "manual_roi_used": False,
        "manual_frame_id_used": False,
        "semantic_model_used": False,
        "affine_used": False,
        "hole_fill_used": False,
        "generated_rgb_used": False,
        "segmentation": segmentation.audit,
        "candidate_track_count": candidate_count,
        "accepted_track_count": len(accepted),
        "rejected_track_count": len(rejected),
        "accepted_pixel_count": int(np.count_nonzero(accepted_mask)),
        "rejection_reason_counts": dict(
            Counter(item["reason"] for item in rejected)
        ),
        "accepted_tracks": accepted,
        "rejected_tracks": rejected,
        "thresholds": {
            "voxel_size_mm": 12.0,
            "maximum_lab_delta": 32.0,
            "maximum_normal_angle_degrees": 45.0,
            "maximum_plane_residual_mm": 14.0,
            "minimum_complete_view_count": 2,
            "minimum_cross_view_iou": 0.30,
            "minimum_selected_union_coverage_ratio": 0.95,
        },
        "silent_fallback_allowed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "diagnostic": image_path.name,
            "before_after": comparison_path.name,
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_path)
    print(image_path)
    print(comparison_path)
    print(
        json.dumps(
            {
                "raw_sample_count": int(raw_world.shape[0]),
                "raw_track_count": segmentation.audit["raw_track_count"],
                "candidate_track_count": candidate_count,
                "accepted_track_count": len(accepted),
                "rejected_track_count": len(rejected),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
