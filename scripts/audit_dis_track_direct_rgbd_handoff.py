"""Direct RGB-D whole-object owners for every stable FastSAM+DIS track."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.dis_track_direct_handoff import (
    DirectHandoffConfig,
    DirectProjectedObservation,
    evaluate_direct_track,
)
from panorama_demo.inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    build_fastsam_rgbd_candidate,
    parse_fastsam_polygons,
    polygon_mask,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
    _read_rgbd,
    _undistortion_maps,
)
from panorama_demo.inspection_object_handoff import (
    project_complete_object_owner_from_rgbd,
)
from panorama_demo.session import load_rgbd_session


FOCUS_TRACK_IDS = (0, 49, 112, 479)


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
    formal_output: Path,
    layout: InspectionMultiviewLayout,
    crop: dict[str, object],
) -> np.ndarray:
    image = cv2.imread(
        str(formal_output / "mosaic_inspection.png"), cv2.IMREAD_COLOR
    )
    x, y, width, height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    if image is None or image.shape != (height, width, 3):
        raise RuntimeError("Formal inspection image is incomplete")
    full = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    full[y : y + height, x : x + width] = image
    return full


def _bbox(mask: np.ndarray) -> list[int]:
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return [0, 0, 0, 0]
    return [
        int(np.min(xx)),
        int(np.min(yy)),
        int(np.max(xx) - np.min(xx) + 1),
        int(np.max(yy) - np.min(yy) + 1),
    ]


def _build_selected_panel_candidates(
    *,
    panel_selection: list[dict[str, object]],
    frame_by_id: dict[int, object],
    pose_by_id: dict[int, np.ndarray],
    labels_path: Path,
    calibration: object,
    maps: tuple[np.ndarray, np.ndarray],
    config: InspectionMultiviewConfig,
    reference_depth_mm: float,
) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    local_candidate_id = 0
    for payload in panel_selection:
        panel_index = int(payload["panel_index"])
        frame_id = int(payload["frame_id"])
        frame = frame_by_id[frame_id]
        pose = pose_by_id[frame_id]
        image, depth, geometric_valid = _read_rgbd(
            frame, calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        candidates: list[FastSAMRGBDCandidate] = []
        for polygon in parse_fastsam_polygons(
            labels_path / f"{frame_id:08d}.txt",
            width=calibration.width,
            height=calibration.height,
        ):
            candidate = build_fastsam_rgbd_candidate(
                candidate_id=local_candidate_id,
                source_index=panel_index,
                frame_id=frame_id,
                polygon_xy=polygon,
                image_bgr=image,
                lab_image=lab,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=pose,
                intrinsics=calibration,
                reference_depth_mm=reference_depth_mm,
            )
            local_candidate_id += 1
            if candidate is not None:
                candidates.append(candidate)
        result[frame_id] = {
            "panel_index": panel_index,
            "frame_id": frame_id,
            "image_bgr": image,
            "depth_mm": depth,
            "reliable_depth": reliable,
            "pose": pose,
            "candidates": candidates,
        }
    return result


def _match_track_observations(
    track: dict[str, object],
    source_by_frame: dict[int, dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    matched: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for stable_candidate_id, frame_id, bbox in zip(
        track["selected_panel_candidate_ids"],
        track["selected_panel_frame_ids"],
        track["selected_panel_bboxes_xywh"],
        strict=True,
    ):
        source = source_by_frame.get(int(frame_id))
        if source is None:
            rejected.append(
                {
                    "candidate_id": int(stable_candidate_id),
                    "frame_id": int(frame_id),
                    "reason": "stable_track_panel_source_is_missing",
                }
            )
            continue
        candidates = [
            candidate
            for candidate in source["candidates"]
            if list(candidate.bbox_xywh) == list(bbox)
        ]
        if len(candidates) != 1:
            rejected.append(
                {
                    "candidate_id": int(stable_candidate_id),
                    "frame_id": int(frame_id),
                    "bbox_xywh": list(bbox),
                    "matching_current_candidate_count": len(candidates),
                    "reason": "stable_track_bbox_is_not_uniquely_reconstructed",
                }
            )
            continue
        matched.append(
            {
                "stable_candidate_id": int(stable_candidate_id),
                "source": source,
                "candidate": replace(
                    candidates[0], candidate_id=int(stable_candidate_id)
                ),
            }
        )
    return matched, rejected


def _track_contact_sheet(
    track_rows: list[dict[str, object]],
    source_by_frame: dict[int, dict[str, object]],
) -> np.ndarray:
    card_width, card_height = 225, 165
    columns = 4
    sheet = np.full(
        (max(1, len(track_rows)) * card_height, columns * card_width, 3),
        245,
        dtype=np.uint8,
    )
    for row, track in enumerate(track_rows):
        decision = track["decision"]
        accepted = bool(decision["accepted"])
        color = (0, 150, 0) if accepted else (0, 0, 210)
        focus = int(track["track_id"]) in FOCUS_TRACK_IDS
        for column, observation in enumerate(track["source_observations"][:3]):
            source = source_by_frame[int(observation["frame_id"])]
            image = np.asarray(source["image_bgr"])
            x, y, width, height = observation["bbox_xywh"]
            margin = 14
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1 = min(image.shape[1], x + width + margin)
            y1 = min(image.shape[0], y + height + margin)
            crop = image[y0:y1, x0:x1].copy()
            polygon = (
                np.asarray(observation["polygon_xy"], dtype=np.int32)
                - np.asarray([x0, y0], dtype=np.int32)
            )
            cv2.polylines(crop, [polygon], True, color, 2)
            scale = min(
                (card_width - 8) / max(1, crop.shape[1]),
                (card_height - 36) / max(1, crop.shape[0]),
            )
            resized = cv2.resize(
                crop,
                (
                    max(1, int(round(crop.shape[1] * scale))),
                    max(1, int(round(crop.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
            card = sheet[
                row * card_height : (row + 1) * card_height,
                column * card_width : (column + 1) * card_width,
            ]
            offset_x = (card_width - resized.shape[1]) // 2
            offset_y = 32 + (card_height - 32 - resized.shape[0]) // 2
            card[
                offset_y : offset_y + resized.shape[0],
                offset_x : offset_x + resized.shape[1],
            ] = resized
            cv2.putText(
                card,
                (
                    f"T{track['track_id']} F{observation['frame_id']} "
                    f"D{observation['depth_coverage_ratio']:.2f}"
                ),
                (5, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )
            if focus:
                cv2.rectangle(
                    card,
                    (1, 1),
                    (card_width - 2, card_height - 2),
                    (0, 190, 255),
                    2,
                )
        summary = sheet[
            row * card_height : (row + 1) * card_height,
            3 * card_width : 4 * card_width,
        ]
        summary[:] = (250, 250, 250)
        lines = [
            f"T{track['track_id']} {'PASS' if accepted else 'FAIL'}",
            str(decision.get("reason", ""))[:31],
            (
                "owner="
                + (
                    f"F{decision.get('selected_frame_id')}"
                    if accepted
                    else "none"
                )
            ),
        ]
        pair_values = [
            float(item["target_mask_iou"])
            for item in decision.get("pair_audits", [])
        ]
        lines.append(
            f"IoU max={max(pair_values):.3f}"
            if pair_values
            else "IoU max=n/a"
        )
        if "selected_target_union_coverage_ratio" in decision:
            lines.append(
                "coverage="
                f"{decision['selected_target_union_coverage_ratio']:.3f}"
            )
        for index, line in enumerate(lines):
            cv2.putText(
                summary,
                line,
                (6, 22 + 27 * index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )
        if focus:
            cv2.rectangle(
                summary,
                (1, 1),
                (card_width - 2, card_height - 2),
                (0, 190, 255),
                2,
            )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("formal_output", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("dis_track_audit", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    started = time.perf_counter()
    session_path = arguments.session.expanduser().resolve()
    formal_output = arguments.formal_output.expanduser().resolve()
    labels_path = arguments.labels.expanduser().resolve()
    dis_audit_path = arguments.dis_track_audit.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == formal_output:
        raise ValueError("Diagnostic output must not be the formal output")

    report = json.loads(
        (formal_output / "report.json").read_text(encoding="utf-8")
    )
    render = report["render"]
    transforms = json.loads(
        (formal_output / "transforms.json").read_text(encoding="utf-8")
    )
    dis_audit = json.loads(dis_audit_path.read_text(encoding="utf-8"))
    if dis_audit.get("schema") != "inspection-fastsam-dis-identity-tracks/v1":
        raise RuntimeError("Stable FastSAM+DIS track audit schema is unsupported")
    tracks = list(dis_audit["stable_selected_panel_tracks"])
    session = load_rgbd_session(session_path)
    frame_by_id = {int(frame.frame_id): frame for frame in session.frames}
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    config = InspectionMultiviewConfig.from_mapping(render["config"])
    layout = _layout_from_report(render["layout"])
    fixed = DirectHandoffConfig()
    fixed.validate()
    source_by_frame = _build_selected_panel_candidates(
        panel_selection=render["selected_panel_sources"],
        frame_by_id=frame_by_id,
        pose_by_id=pose_by_id,
        labels_path=labels_path,
        calibration=session.calibration,
        maps=_undistortion_maps(session.calibration),
        config=config,
        reference_depth_mm=layout.reference_depth_mm,
    )
    full_baseline = _full_baseline(
        formal_output, layout, render["crop"]
    )

    runtime_rows: list[dict[str, object]] = []
    public_rows: list[dict[str, object]] = []
    for index, track in enumerate(tracks):
        track_id = int(track["track_id"])
        matched, reconstruction_rejections = _match_track_observations(
            track, source_by_frame
        )
        source_panels = sorted(
            int(item["source"]["panel_index"]) for item in matched
        )
        target_panel_index = (
            source_panels[len(source_panels) // 2]
            if source_panels
            else -1
        )
        projections: list[DirectProjectedObservation] = []
        source_observations: list[dict[str, object]] = []
        projection_rejections = list(reconstruction_rejections)
        for item in matched:
            source = item["source"]
            candidate = item["candidate"]
            mask = polygon_mask(
                candidate, np.asarray(source["depth_mm"]).shape
            )
            source_count = int(np.count_nonzero(mask))
            reliable_count = int(
                np.count_nonzero(mask & source["reliable_depth"])
            )
            depth_hole_count = source_count - reliable_count
            depth_coverage = float(
                reliable_count / max(1, source_count)
            )
            gray = cv2.cvtColor(source["image_bgr"], cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            clarity = float(np.var(laplacian[mask]))
            source_observations.append(
                {
                    "candidate_id": int(candidate.candidate_id),
                    "frame_id": int(candidate.frame_id),
                    "panel_index": int(source["panel_index"]),
                    "bbox_xywh": list(candidate.bbox_xywh),
                    "polygon_xy": candidate.polygon_xy.tolist(),
                    "source_mask_pixel_count": source_count,
                    "reliable_depth_pixel_count": reliable_count,
                    "depth_hole_pixel_count": depth_hole_count,
                    "depth_hole_ratio": float(
                        depth_hole_count / max(1, source_count)
                    ),
                    "depth_coverage_ratio": depth_coverage,
                    "clarity_laplacian_variance": clarity,
                }
            )
            if target_panel_index < 0:
                continue
            try:
                owner = project_complete_object_owner_from_rgbd(
                    source_image_bgr=source["image_bgr"],
                    source_depth_mm=source["depth_mm"],
                    source_reliable_depth=source["reliable_depth"],
                    source_object_mask=mask,
                    camera_to_world=source["pose"],
                    layout=layout,
                    intrinsics=session.calibration,
                    frame_id=int(candidate.frame_id),
                    panel_index=target_panel_index,
                    minimum_cells=64,
                )
            except (RuntimeError, ValueError) as exc:
                projection_rejections.append(
                    {
                        "candidate_id": int(candidate.candidate_id),
                        "frame_id": int(candidate.frame_id),
                        "reason": "direct_rgbd_projection_failed",
                        "detail": str(exc),
                        "source_depth_coverage_ratio": depth_coverage,
                        "source_depth_hole_pixel_count": depth_hole_count,
                    }
                )
                continue
            projection_audit = {
                **owner.audit,
                "source_depth_hole_pixel_count": depth_hole_count,
                "source_depth_hole_ratio": float(
                    depth_hole_count / max(1, source_count)
                ),
            }
            projections.append(
                DirectProjectedObservation(
                    candidate_id=int(candidate.candidate_id),
                    frame_id=int(candidate.frame_id),
                    source_panel_index=int(source["panel_index"]),
                    target_panel_index=target_panel_index,
                    target_mask=owner.target_mask,
                    target_image_bgr=owner.target_image_bgr,
                    source_depth_coverage_ratio=depth_coverage,
                    clarity=clarity,
                    projection_audit=projection_audit,
                )
            )
        decision = evaluate_direct_track(
            track_id, projections, config=fixed
        )
        decision_audit = {
            **decision.audit,
            "stable_track_observation_count": int(
                track["observation_count"]
            ),
            "stable_selected_panel_observation_count": int(
                track["selected_panel_observation_count"]
            ),
            "stable_selected_panel_frame_ids": list(
                track["selected_panel_frame_ids"]
            ),
            "stable_dis_minimum_flow_mask_iou": float(
                track["minimum_flow_mask_iou"]
            ),
            "stable_dis_maximum_fb_p95_preview_pixels": float(
                track["maximum_fb_p95_preview_pixels"]
            ),
            "stable_dis_maximum_area_ratio": float(
                track["maximum_area_ratio"]
            ),
            "automatic_target_world_panel_index": target_panel_index,
            "projection_rejections": projection_rejections,
            "source_observations": [
                {
                    key: value
                    for key, value in observation.items()
                    if key != "polygon_xy"
                }
                for observation in source_observations
            ],
        }
        public_rows.append(decision_audit)
        runtime_rows.append(
            {
                "track_id": track_id,
                "decision": decision_audit,
                "selected_observation": decision.selected_observation,
                "source_observations": source_observations,
            }
        )
        print(
            f"track {index + 1}/{len(tracks)} T{track_id}: "
            f"{len(projections)} direct projections, "
            f"{'PASS' if decision.accepted else 'FAIL'} "
            f"{decision.audit['reason']}",
            flush=True,
        )

    diagnostic = full_baseline.copy()
    accepted_mask = np.zeros(full_baseline.shape[:2], dtype=bool)
    accepted_tracks: list[dict[str, object]] = []
    final_rejected_tracks: list[dict[str, object]] = []
    for runtime in sorted(
        runtime_rows,
        key=lambda item: (
            -int(
                item["decision"].get("consistent_projection_count", 0)
            ),
            -float(
                item["decision"].get(
                    "selected_target_union_coverage_ratio", 0.0
                )
            ),
            int(item["track_id"]),
        ),
    ):
        decision = runtime["decision"]
        observation = runtime["selected_observation"]
        if not decision["accepted"] or observation is None:
            final_rejected_tracks.append(decision)
            continue
        overlap = int(
            np.count_nonzero(observation.target_mask & accepted_mask)
        )
        overlap_ratio = float(
            overlap / max(1, np.count_nonzero(observation.target_mask))
        )
        if overlap_ratio > fixed.maximum_track_overlap_ratio:
            decision["accepted"] = False
            decision["reason"] = "direct_target_overlaps_prior_accepted_track"
            decision["accepted_track_overlap_pixel_count"] = overlap
            decision["accepted_track_overlap_ratio"] = overlap_ratio
            final_rejected_tracks.append(decision)
            continue
        diagnostic[observation.target_mask] = (
            observation.target_image_bgr[observation.target_mask]
        )
        accepted_mask |= observation.target_mask
        decision["selected_target_bbox_xywh"] = _bbox(
            observation.target_mask
        )
        decision["selected_target_pixel_count"] = int(
            np.count_nonzero(observation.target_mask)
        )
        accepted_tracks.append(decision)

    crop = render["crop"]
    crop_x, crop_y, crop_width, crop_height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    baseline_crop = full_baseline[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    diagnostic_crop = diagnostic[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    comparison = np.hstack((baseline_crop, diagnostic_crop))
    for item in accepted_tracks:
        x, y, width, height = item["selected_target_bbox_xywh"]
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
        cv2.putText(
            comparison,
            f"T{item['track_id']} F{item['selected_frame_id']}",
            (
                x - crop_x + crop_width,
                max(20, y - crop_y - 4),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    contact_sheet = _track_contact_sheet(runtime_rows, source_by_frame)
    diagnostic_path = output / "dis_track_direct_rgbd_handoff.png"
    comparison_path = output / "dis_track_direct_rgbd_before_after.png"
    contact_path = output / "dis_track_direct_rgbd_contact_sheet.jpg"
    audit_path = output / "dis_track_direct_rgbd_handoff_audit.json"
    if not cv2.imwrite(str(diagnostic_path), diagnostic_crop):
        raise RuntimeError("Could not write direct RGB-D handoff image")
    if not cv2.imwrite(str(comparison_path), comparison):
        raise RuntimeError("Could not write direct RGB-D before/after")
    if not cv2.imwrite(str(contact_path), contact_sheet):
        raise RuntimeError("Could not write direct RGB-D contact sheet")

    by_track = {int(item["track_id"]): item for item in public_rows}
    focus = {
        f"T{track_id}": (
            by_track.get(track_id)
            if track_id in by_track
            else {
                "track_id": track_id,
                "accepted": False,
                "reason": "not_in_stable_selected_panel_tracks",
            }
        )
        for track_id in FOCUS_TRACK_IDS
    }
    final_reason_counts = Counter(
        str(item["reason"]) for item in final_rejected_tracks
    )
    audit = {
        "schema": "fastsam-dis-direct-rgbd-handoff-diagnostic/v1",
        "formal_output_modified": False,
        "formal_renderer_connected": False,
        "stable_track_audit": str(dis_audit_path),
        "stable_track_count": len(tracks),
        "processed_stable_track_count": len(runtime_rows),
        "accepted_direct_owner_count": len(accepted_tracks),
        "rejected_direct_owner_count": len(final_rejected_tracks),
        "accepted_direct_owner_pixel_count": int(
            np.count_nonzero(accepted_mask)
        ),
        "rejection_reason_counts": dict(final_reason_counts),
        "focus_track_results": focus,
        "accepted_tracks": accepted_tracks,
        "rejected_tracks": final_rejected_tracks,
        "all_track_decisions_before_overlap_suppression": public_rows,
        "fixed_thresholds": {
            name: getattr(fixed, name)
            for name in fixed.__dataclass_fields__
        },
        "constraints": {
            "all_stable_tracks_processed_automatically": True,
            "hardcoded_track_selection_used": False,
            "focus_track_ids_used_for_reporting_only": list(FOCUS_TRACK_IDS),
            "aligned_depth_used": True,
            "immutable_real_se3_used": True,
            "direct_triangle_raster_used": True,
            "single_complete_rgb_owner": True,
            "translation_used": False,
            "affine_used": False,
            "fitted_warp_used": False,
            "pose_interpolation_used": False,
            "hole_fill_used": False,
            "generated_rgb_used": False,
            "blend_used": False,
            "tsdf_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "diagnostic": diagnostic_path.name,
            "before_after": comparison_path.name,
            "contact_sheet": contact_path.name,
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_path)
    print(
        json.dumps(
            {
                "stable_track_count": audit["stable_track_count"],
                "accepted_direct_owner_count": audit[
                    "accepted_direct_owner_count"
                ],
                "rejected_direct_owner_count": audit[
                    "rejected_direct_owner_count"
                ],
                "focus_track_results": {
                    key: {
                        "accepted": value["accepted"],
                        "reason": value["reason"],
                    }
                    for key, value in focus.items()
                },
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
