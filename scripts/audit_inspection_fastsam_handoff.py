"""One-shot FastSAM-contour plus measured RGB-D object handoff diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import json
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    build_fastsam_rgbd_candidate,
    parse_fastsam_polygons,
    polygon_mask,
    track_fastsam_rgbd_candidates,
)
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
    decoded = encoded.astype(np.int32) - 1
    full_owner[y : y + height, x : x + width] = decoded
    full_valid[y : y + height, x : x + width] = decoded >= 0
    return full_image, full_owner, full_valid


def _target_shape_consistent(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[bool, float, float]:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    iou = float(intersection / union) if union else 0.0
    first_contours, _ = cv2.findContours(
        first.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    second_contours, _ = cv2.findContours(
        second.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not first_contours or not second_contours:
        return False, iou, float("inf")
    first_contour = max(first_contours, key=cv2.contourArea)
    second_contour = max(second_contours, key=cv2.contourArea)
    contour_delta = float(
        cv2.matchShapes(
            first_contour, second_contour, cv2.CONTOURS_MATCH_I1, 0.0
        )
    )
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    area_ratio = max(first_area, second_area) / max(
        1, min(first_area, second_area)
    )
    return (
        iou >= 0.30 and contour_delta <= 0.35 and area_ratio <= 2.0,
        iou,
        contour_delta,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("formal_output")
    parser.add_argument("labels")
    arguments = parser.parse_args()
    started = time.perf_counter()
    session_path = Path(arguments.session).expanduser().resolve()
    output = Path(arguments.formal_output).expanduser().resolve()
    labels_path = Path(arguments.labels).expanduser().resolve()
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
    candidates: list[FastSAMRGBDCandidate] = []
    candidates_by_source: list[list[FastSAMRGBDCandidate]] = []
    raw_polygon_count = 0
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
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        polygons = parse_fastsam_polygons(
            labels_path / f"{int(frame.frame_id):08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        raw_polygon_count += len(polygons)
        selected: list[FastSAMRGBDCandidate] = []
        for polygon in polygons:
            candidate = build_fastsam_rgbd_candidate(
                candidate_id=len(candidates),
                source_index=source_index,
                frame_id=int(frame.frame_id),
                polygon_xy=polygon,
                image_bgr=image,
                lab_image=lab,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=pose,
                intrinsics=session.calibration,
                reference_depth_mm=layout.reference_depth_mm,
            )
            if candidate is not None:
                candidates.append(candidate)
                selected.append(candidate)
        candidates_by_source.append(selected)
        if (source_index + 1) % 24 == 0:
            print(
                f"described {source_index + 1}/{len(frames)} RGB-D frames",
                flush=True,
            )
    tracks = track_fastsam_rgbd_candidates(candidates_by_source)
    candidate_by_id = {item.candidate_id: item for item in candidates}
    panel_anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels],
        dtype=np.float64,
    )
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
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

    proposals: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for track in tracks:
        track_candidates = [
            candidate_by_id[value] for value in track.candidate_ids
        ]
        ranked = sorted(
            track_candidates,
            key=lambda item: (
                item.depth_coverage_ratio,
                len(item.world_voxel_hashes),
                item.source_area_pixels,
                -item.frame_id,
            ),
            reverse=True,
        )[:4]
        median_world = np.median(
            np.asarray(
                [item.world_centroid_mm for item in track_candidates]
            ),
            axis=0,
        )
        panel_index = int(
            np.argmin(
                np.abs(panel_anchors - float(median_world @ scan_axis))
            )
        )
        projections = []
        source_audits: list[dict[str, object]] = []
        for candidate in ranked:
            image, depth, reliable = load_source(candidate.source_index)
            source_mask = polygon_mask(candidate, depth.shape)
            try:
                owner = project_complete_object_owner_from_rgbd(
                    source_image_bgr=image,
                    source_depth_mm=depth,
                    source_reliable_depth=reliable,
                    source_object_mask=source_mask,
                    camera_to_world=poses[candidate.source_index],
                    layout=layout,
                    intrinsics=session.calibration,
                    frame_id=candidate.frame_id,
                    panel_index=panel_index,
                    minimum_cells=64,
                )
            except (RuntimeError, ValueError) as exc:
                source_audits.append(
                    {
                        "frame_id": candidate.frame_id,
                        "reason": str(exc),
                    }
                )
                continue
            projections.append((candidate, owner))
            source_audits.append(
                {
                    "frame_id": candidate.frame_id,
                    "source_mask_pixel_count": (
                        candidate.source_area_pixels
                    ),
                    "target_pixel_count": int(
                        np.count_nonzero(owner.target_mask)
                    ),
                    "accepted_direct_projection": True,
                }
            )
        rejection_base = {
            "track_id": int(track.track_id),
            "world_track_source_count": len(track.source_indices),
            "world_track_candidate_count": len(track.candidate_ids),
            "selected_panel_index": panel_index,
            "world_track": track.audit,
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
        pair_audits = []
        for first_index, (first_candidate, first_owner) in enumerate(
            projections
        ):
            consistent = [first_index]
            for second_index, (second_candidate, second_owner) in enumerate(
                projections
            ):
                if second_index == first_index:
                    continue
                accepted_pair, iou, contour_delta = (
                    _target_shape_consistent(
                        first_owner.target_mask, second_owner.target_mask
                    )
                )
                pair_audits.append(
                    {
                        "first_frame_id": first_candidate.frame_id,
                        "second_frame_id": second_candidate.frame_id,
                        "target_iou": iou,
                        "contour_match_i1": contour_delta,
                        "accepted": accepted_pair,
                    }
                )
                if accepted_pair:
                    consistent.append(second_index)
            if len(consistent) < 2:
                continue
            footprint_union = np.logical_or.reduce(
                [projections[index][1].target_mask for index in consistent]
            )
            coverage = float(
                np.count_nonzero(
                    first_owner.target_mask & footprint_union
                )
                / max(1, np.count_nonzero(footprint_union))
            )
            score = (
                coverage,
                len(consistent),
                first_candidate.depth_coverage_ratio,
                -first_candidate.frame_id,
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    first_candidate,
                    first_owner,
                    consistent,
                    footprint_union,
                )
        if best is None or best[0][0] < 0.95:
            rejected.append(
                {
                    **rejection_base,
                    "reason": (
                        "cross_view_world_track_masks_disagree_or_"
                        "union_coverage_below_0_95"
                    ),
                    "pair_audits": pair_audits,
                }
            )
            continue
        _, selected_candidate, selected_owner, consistent, footprint_union = (
            best
        )
        baseline_owners = np.unique(
            full_owner[footprint_union & full_valid]
        )
        baseline_owners = baseline_owners[baseline_owners >= 0]
        if baseline_owners.size < 2:
            continue
        try:
            interval = build_object_owner_interval(
                panel_index=panel_index,
                view_dependent_footprints=tuple(
                    projections[index][1].target_mask
                    for index in consistent
                ),
                selected_panel_valid_mask=full_valid,
            )
        except (RuntimeError, ValueError) as exc:
            rejected.append(
                {
                    **rejection_base,
                    "reason": f"owner_interval_rejected: {exc}",
                }
            )
            continue
        interval_coverage = float(
            np.count_nonzero(
                selected_owner.target_mask & interval.union_footprint
            )
            / max(1, np.count_nonzero(interval.union_footprint))
        )
        if interval_coverage < 0.95:
            rejected.append(
                {
                    **rejection_base,
                    "reason": "selected_owner_interval_coverage_below_0_95",
                    "interval_union_coverage_ratio": interval_coverage,
                }
            )
            continue
        target_y, target_x = np.nonzero(selected_owner.target_mask)
        proposals.append(
            {
                **rejection_base,
                "selected_candidate": selected_candidate,
                "selected_owner": selected_owner,
                "consistent_projection_count": len(consistent),
                "selected_cross_view_union_coverage_ratio": float(best[0][0]),
                "interval_union_coverage_ratio": interval_coverage,
                "baseline_owner_frame_ids": [
                    int(value) for value in baseline_owners
                ],
                "target_bbox_xywh": [
                    int(np.min(target_x)),
                    int(np.min(target_y)),
                    int(np.max(target_x) - np.min(target_x) + 1),
                    int(np.max(target_y) - np.min(target_y) + 1),
                ],
                "target_pixel_count": int(
                    np.count_nonzero(selected_owner.target_mask)
                ),
                "pair_audits": pair_audits,
                "interval_audit": interval.audit,
            }
        )

    diagnostic = full_image.copy()
    accepted_mask = np.zeros(full_valid.shape, dtype=bool)
    accepted: list[dict[str, object]] = []
    for proposal in sorted(
        proposals,
        key=lambda item: (
            item["consistent_projection_count"],
            item["selected_cross_view_union_coverage_ratio"],
            item["target_pixel_count"],
        ),
        reverse=True,
    ):
        owner = proposal.pop("selected_owner")
        selected_candidate = proposal.pop("selected_candidate")
        overlap = int(np.count_nonzero(owner.target_mask & accepted_mask))
        overlap_ratio = float(
            overlap / max(1, np.count_nonzero(owner.target_mask))
        )
        if overlap_ratio > 0.15:
            rejected.append(
                {
                    **proposal,
                    "reason": "duplicate_world_track_target_overlap",
                    "accepted_target_overlap_ratio": overlap_ratio,
                }
            )
            continue
        diagnostic[owner.target_mask] = owner.target_image_bgr[
            owner.target_mask
        ]
        accepted_mask |= owner.target_mask
        accepted.append(
            {
                **proposal,
                "selected_frame_id": int(selected_candidate.frame_id),
                "single_rgb_owner": True,
                "pose_modified": False,
                "rgb_generated": False,
                "blend_used": False,
            }
        )

    crop = render["crop"]
    crop_x, crop_y, crop_width, crop_height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    baseline_crop = full_image[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    diagnostic_crop = diagnostic[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    comparison = np.hstack((baseline_crop, diagnostic_crop))
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
    diagnostic_path = output / "diagnostic_fastsam_rgbd_handoff.png"
    comparison_path = (
        output / "diagnostic_fastsam_rgbd_handoff_before_after.jpg"
    )
    audit_path = output / "diagnostic_fastsam_rgbd_handoff_audit.json"
    if not cv2.imwrite(str(diagnostic_path), diagnostic_crop):
        raise RuntimeError("Could not write FastSAM RGB-D diagnostic")
    if not cv2.imwrite(str(comparison_path), comparison):
        raise RuntimeError("Could not write FastSAM RGB-D comparison")
    serializable_rejected = rejected
    audit = {
        "schema": "inspection-fastsam-rgbd-handoff-diagnostic/v1",
        "formal_output_modified": False,
        "formal_acceptance": False,
        "formal_acceptance_reason": (
            "bounded_first_prototype_not_connected_to_formal_owner_chain"
        ),
        "model_role": "polygon_contour_proposals_only",
        "model_weight_sha256": (
            "c9f78716a81c7aff0d608ccc73e1b82a"
            "b3aaad86005049f6a92106a0be6d0844"
        ),
        "frame_count": len(frames),
        "all_real_pose_rgbd_frames_used": len(frames) == 145,
        "raw_polygon_count": raw_polygon_count,
        "rgbd_candidate_count": len(candidates),
        "world_track_count": len(tracks),
        "accepted_track_count": len(accepted),
        "rejected_track_count": len(serializable_rejected),
        "accepted_pixel_count": int(np.count_nonzero(accepted_mask)),
        "rejection_reason_counts": dict(
            Counter(item["reason"] for item in serializable_rejected)
        ),
        "accepted_tracks": accepted,
        "rejected_tracks": serializable_rejected,
        "thresholds": {
            "world_voxel_size_mm": 20.0,
            "minimum_world_voxel_overlap_ratio": 0.25,
            "maximum_lab_delta": 30.0,
            "maximum_source_gap": 12,
            "minimum_view_count": 2,
            "minimum_target_iou": 0.30,
            "maximum_contour_match_i1": 0.35,
            "minimum_selected_union_coverage_ratio": 0.95,
        },
        "manual_roi_used": False,
        "manual_frame_id_used": False,
        "model_position_used": False,
        "model_rgb_used": False,
        "affine_used": False,
        "hole_fill_used": False,
        "generated_rgb_used": False,
        "silent_fallback_allowed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "diagnostic": diagnostic_path.name,
            "before_after": comparison_path.name,
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_path)
    print(diagnostic_path)
    print(comparison_path)
    print(
        json.dumps(
            {
                "raw_polygon_count": raw_polygon_count,
                "rgbd_candidate_count": len(candidates),
                "world_track_count": len(tracks),
                "accepted_track_count": len(accepted),
                "rejected_track_count": len(rejected),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
