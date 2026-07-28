"""Read-only OCR-seeded WAVESHARE panel and shared-owner diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_fastsam_track import parse_fastsam_polygons
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
    _read_rgbd,
    _undistortion_maps,
    project_world_points_to_panels,
)
from panorama_demo.inspection_ocr_panel import (
    OCRSeededPanel,
    extract_ocr_seeded_panel,
    sample_mask_world_points,
    track_ocr_seeded_panels,
)
from panorama_demo.session import load_rgbd_session


def _layout_from_audit(
    value: dict[str, object],
) -> InspectionMultiviewLayout:
    return InspectionMultiviewLayout(
        width=int(value["width"]),
        height=int(value["height"]),
        reference_depth_mm=float(value["reference_depth_mm"]),
        scan_axis=tuple(
            float(item) for item in value["scan_axis_world"]
        ),
        down_axis=tuple(
            float(item) for item in value["down_axis_world"]
        ),
        normal_axis=tuple(
            float(item) for item in value["normal_axis_world"]
        ),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=int(item["panel_index"]),
                anchor_scan_mm=float(item["anchor_scan_mm"]),
                canvas_offset_x=float(item["canvas_offset_x"]),
                center_world_mm=tuple(
                    float(point) for point in item["center_world_mm"]
                ),
            )
            for item in value["panels"]
        ),
        panel_step_mm=float(value["panel_step_mm"]),
        canvas_megapixels=float(value["canvas_megapixels"]),
    )


def _polygon_mask(
    polygon_xy: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_xy.astype(np.int32)], 1)
    return mask.astype(bool)


def _project_owner_coverage(
    *,
    points_world_mm: np.ndarray,
    source_frame_id: int,
    expected_panel_index: int,
    layout: InspectionMultiviewLayout,
    intrinsics,
    crop: dict[str, object],
    owner_frame_id: np.ndarray,
) -> dict[str, object]:
    if points_world_mm.shape[0] == 0:
        return {
            "sample_count": 0,
            "pass": False,
            "rejection_reason": "no_measured_world_points",
        }
    x, y, q_normal, panel_index = project_world_points_to_panels(
        points_world_mm, layout, intrinsics
    )
    x = np.rint(x - int(crop["x"])).astype(np.int32)
    y = np.rint(y - int(crop["y"])).astype(np.int32)
    valid = (
        np.isfinite(q_normal)
        & (q_normal > 0.0)
        & (x >= 0)
        & (x < owner_frame_id.shape[1])
        & (y >= 0)
        & (y < owner_frame_id.shape[0])
    )
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        return {
            "sample_count": int(points_world_mm.shape[0]),
            "in_bounds_sample_count": 0,
            "pass": False,
            "rejection_reason": "all_points_project_outside_final_owner",
        }
    owners = owner_frame_id[y[valid], x[valid]]
    selected_owner_ratio = float(
        np.count_nonzero(owners == source_frame_id) / valid_count
    )
    expected_panel_ratio = float(
        np.count_nonzero(
            panel_index[valid] == expected_panel_index
        )
        / valid_count
    )
    projected_ratio = float(valid_count / points_world_mm.shape[0])
    accepted = bool(
        valid_count >= 30
        and projected_ratio >= 0.80
        and selected_owner_ratio >= 0.90
        and expected_panel_ratio >= 0.90
    )
    values, counts = np.unique(owners, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return {
        "sample_count": int(points_world_mm.shape[0]),
        "in_bounds_sample_count": valid_count,
        "projected_in_bounds_ratio": projected_ratio,
        "selected_source_owner_ratio": selected_owner_ratio,
        "expected_panel_projection_ratio": expected_panel_ratio,
        "owner_vote_top": [
            {
                "frame_id": int(value),
                "sample_count": int(count),
            }
            for value, count in zip(
                values[order][:5], counts[order][:5], strict=False
            )
        ],
        "pass": accepted,
        "rejection_reason": (
            None if accepted else "single_panel_source_owner_gate_failed"
        ),
    }


def _contact_sheet(
    display_rows: list[dict[str, object]],
    stable_ids: set[int],
    shared_source_frame_id: int | None,
) -> np.ndarray:
    width, height = 420, 260
    columns = 3
    ordered = sorted(
        display_rows,
        key=lambda row: (
            int(row["candidate_index"]) in stable_ids,
            int(row["frame_id"]) == shared_source_frame_id,
            float(row["panel"].clarity_variance),
        ),
        reverse=True,
    )[:18]
    row_count = max(1, math.ceil(len(ordered) / columns))
    sheet = np.full(
        (row_count * height, columns * width, 3),
        245,
        dtype=np.uint8,
    )
    for index, row in enumerate(ordered):
        image = np.asarray(row["image"]).copy()
        panel: OCRSeededPanel = row["panel"]
        cv2.polylines(
            image,
            [panel.contour_xy.astype(np.int32)],
            True,
            (20, 220, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.polylines(
            image,
            [np.asarray(row["ocr_polygon_xy"], dtype=np.int32)],
            True,
            (0, 220, 255),
            3,
            cv2.LINE_AA,
        )
        charger_bbox = row.get("charger_bbox_xywh")
        if charger_bbox is not None:
            x, y, box_width, box_height = charger_bbox
            cv2.rectangle(
                image,
                (x, y),
                (x + box_width - 1, y + box_height - 1),
                (255, 180, 20),
                4,
            )
        x, y, box_width, box_height = panel.bbox_xywh
        pad_x = max(30, int(0.15 * box_width))
        pad_y = max(30, int(0.30 * box_height))
        if charger_bbox is not None:
            charger_x, charger_y, charger_w, charger_h = charger_bbox
            x0 = min(x - pad_x, charger_x - 20)
            y0 = min(y - pad_y, charger_y - 20)
            x1 = max(
                x + box_width + pad_x,
                charger_x + charger_w + 20,
            )
            y1 = max(
                y + box_height + pad_y,
                charger_y + charger_h + 20,
            )
        else:
            x0, y0 = x - pad_x, y - pad_y
            x1 = x + box_width + pad_x
            y1 = y + box_height + pad_y
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(image.shape[1], x1)
        y1 = min(image.shape[0], y1)
        crop = image[y0:y1, x0:x1]
        scale = min(
            (width - 8) / crop.shape[1],
            (height - 54) / crop.shape[0],
        )
        resized = cv2.resize(
            crop,
            (
                max(1, int(round(crop.shape[1] * scale))),
                max(1, int(round(crop.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        column = index % columns
        row_index = index // columns
        left = column * width + (width - resized.shape[1]) // 2
        top = row_index * height + 48
        sheet[
            top : top + resized.shape[0],
            left : left + resized.shape[1],
        ] = resized
        stable = int(row["candidate_index"]) in stable_ids
        shared = int(row["frame_id"]) == shared_source_frame_id
        label = (
            f"F{int(row['frame_id'])} "
            f"{'STABLE ' if stable else ''}"
            f"{'SHARED OWNER' if shared else 'PANEL'} "
            f"rect={panel.rectangularity:.2f}"
        )
        cv2.putText(
            sheet,
            label,
            (column * width + 5, row_index * height + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 90, 0) if stable else (40, 40, 180),
            1,
            cv2.LINE_AA,
        )
    if not ordered:
        cv2.putText(
            sheet,
            "No OCR-seeded panel passed fixed structure gates",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 180),
            2,
            cv2.LINE_AA,
        )
    return sheet


def _write_jpeg_atomic(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode seeded panel contact sheet")
    pending = path.with_name(f".{path.name}.pending")
    pending.write_bytes(encoded.tobytes())
    os.replace(pending, path)


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(pending, path)


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
    audit_path = output / "diagnostic_waveshare_seeded_panel_audit.json"
    sheet_path = (
        output / "diagnostic_waveshare_seeded_panel_contact_sheet.jpg"
    )
    if audit_path.exists() or sheet_path.exists():
        raise RuntimeError(
            "Seeded panel outputs already exist; fixed-gate diagnostic "
            "will not be rerun or tuned"
        )

    ocr_audit = json.loads(
        (
            output / "diagnostic_waveshare_ocr_rgbd_audit.json"
        ).read_text(encoding="utf-8")
    )
    fastsam_audit = json.loads(
        (
            output / "diagnostic_fastsam_dis_tracks_audit.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    inspection = json.loads(
        (output / "inspection_meta.json").read_text(encoding="utf-8")
    )
    session = load_rgbd_session(session_path)
    frames = sorted(session.frames, key=lambda item: int(item.frame_id))
    frame_by_id = {int(item.frame_id): item for item in frames}
    source_index_by_id = {
        int(item.frame_id): index for index, item in enumerate(frames)
    }
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    maps = _undistortion_maps(session.calibration)
    ocr_by_frame = {
        int(item["frame_id"]): item["target_detections"]
        for item in ocr_audit["frame_audits"]
        if item["target_detections"]
    }

    candidates: list[OCRSeededPanel] = []
    candidate_rows: list[dict[str, object]] = []
    display_rows: list[dict[str, object]] = []
    for frame_id, detections in sorted(ocr_by_frame.items()):
        pose = pose_by_id.get(frame_id)
        if pose is None or frame_id not in frame_by_id:
            candidate_rows.append(
                {
                    "frame_id": frame_id,
                    "pass": False,
                    "rejection_reason": "real_pose_unavailable",
                }
            )
            continue
        image, depth, geometric_valid = _read_rgbd(
            frame_by_id[frame_id], session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        for detection in detections:
            panel, extraction_audit = extract_ocr_seeded_panel(
                frame_id=frame_id,
                source_index=source_index_by_id[frame_id],
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=reliable,
                ocr_polygon_xy=np.asarray(
                    detection["ocr_polygon_xy"], dtype=np.float32
                ),
                camera_to_world=pose,
                intrinsics=session.calibration,
            )
            row = {
                "frame_id": frame_id,
                "ocr": detection["text_audit"],
                "structure_extraction": extraction_audit,
                "pass": panel is not None,
            }
            if panel is not None:
                candidate_index = len(candidates)
                row["candidate_index"] = candidate_index
                candidates.append(panel)
                display_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "frame_id": frame_id,
                        "image": image.copy(),
                        "ocr_polygon_xy": detection[
                            "ocr_polygon_xy"
                        ],
                        "panel": panel,
                    }
                )
            candidate_rows.append(row)

    tracks = track_ocr_seeded_panels(candidates)
    stable_ids = {
        candidate_index
        for track in tracks
        for candidate_index in track
    }
    track_rows = [
        {
            "track_id": track_index,
            "candidate_indices": list(track),
            "frame_ids": [
                int(candidates[index].frame_id) for index in track
            ],
            "distinct_view_count": len(
                {candidates[index].frame_id for index in track}
            ),
        }
        for track_index, track in enumerate(tracks)
    ]

    t0 = next(
        item
        for item in fastsam_audit["stable_selected_panel_tracks"]
        if int(item["track_id"]) == 0
    )
    charger_observations = {
        int(frame_id): tuple(int(value) for value in bbox)
        for frame_id, bbox in zip(
            t0["selected_panel_frame_ids"],
            t0["selected_panel_bboxes_xywh"],
            strict=True,
        )
    }
    selected_panels = {
        int(item["frame_id"]): (
            int(item["panel_index"]),
            int(item["source_position"]),
        )
        for item in report["render"]["selected_panel_sources"]
    }
    layout = _layout_from_audit(inspection["renderer"]["layout"])
    crop = inspection["renderer"]["crop"]
    owner_encoded = cv2.imread(
        str(output / "inspection_owner.png"),
        cv2.IMREAD_UNCHANGED,
    )
    if owner_encoded is None or owner_encoded.dtype != np.uint16:
        raise RuntimeError("Could not decode uint16 inspection owner")
    owner = owner_encoded.astype(np.int32) - 1
    common_audits: list[dict[str, object]] = []
    display_by_frame = {
        int(item["frame_id"]): item for item in display_rows
    }
    for candidate_index in sorted(stable_ids):
        panel = candidates[candidate_index]
        frame_id = int(panel.frame_id)
        if (
            frame_id not in charger_observations
            or frame_id not in selected_panels
        ):
            continue
        image, depth, geometric_valid = _read_rgbd(
            frame_by_id[frame_id], session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        charger_bbox = charger_observations[frame_id]
        polygons = parse_fastsam_polygons(
            labels_path / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        charger_matches = [
            polygon
            for polygon in polygons
            if tuple(int(value) for value in cv2.boundingRect(polygon))
            == charger_bbox
        ]
        attempt: dict[str, object] = {
            "candidate_index": candidate_index,
            "frame_id": frame_id,
            "panel_index": selected_panels[frame_id][0],
            "source_position": selected_panels[frame_id][1],
            "charger_track_id": 0,
            "charger_bbox_xywh": list(charger_bbox),
            "charger_polygon_match_count": len(charger_matches),
        }
        if len(charger_matches) != 1:
            attempt.update(
                {
                    "pass": False,
                    "rejection_reason": (
                        "stable_t0_polygon_cannot_be_resolved_uniquely"
                    ),
                }
            )
            common_audits.append(attempt)
            continue
        charger_mask = _polygon_mask(
            charger_matches[0], depth.shape
        )
        charger_world = sample_mask_world_points(
            mask=charger_mask,
            depth_mm=depth,
            reliable_depth=reliable,
            camera_to_world=pose_by_id[frame_id],
            intrinsics=session.calibration,
        )
        expected_panel = selected_panels[frame_id][0]
        panel_coverage = _project_owner_coverage(
            points_world_mm=panel.world_points_mm,
            source_frame_id=frame_id,
            expected_panel_index=expected_panel,
            layout=layout,
            intrinsics=session.calibration,
            crop=crop,
            owner_frame_id=owner,
        )
        charger_coverage = _project_owner_coverage(
            points_world_mm=charger_world,
            source_frame_id=frame_id,
            expected_panel_index=expected_panel,
            layout=layout,
            intrinsics=session.calibration,
            crop=crop,
            owner_frame_id=owner,
        )
        x, y, box_width, box_height = charger_bbox
        charger_clarity = float(
            cv2.Laplacian(
                cv2.cvtColor(
                    image[y : y + box_height, x : x + box_width],
                    cv2.COLOR_BGR2GRAY,
                ),
                cv2.CV_64F,
            ).var()
        )
        accepted = bool(
            panel_coverage["pass"] and charger_coverage["pass"]
        )
        score = float(
            min(
                float(
                    panel_coverage.get(
                        "selected_source_owner_ratio", 0.0
                    )
                ),
                float(
                    charger_coverage.get(
                        "selected_source_owner_ratio", 0.0
                    )
                ),
            )
            + 0.01
            * math.log1p(
                min(panel.clarity_variance, charger_clarity)
            )
        )
        attempt.update(
            {
                "panel_owner_coverage": panel_coverage,
                "charger_owner_coverage": charger_coverage,
                "panel_clarity_laplacian_variance": (
                    panel.clarity_variance
                ),
                "charger_clarity_laplacian_variance": charger_clarity,
                "automatic_selection_score": score,
                "pass": accepted,
                "rejection_reason": (
                    None
                    if accepted
                    else "shared_panel_source_owner_gate_failed"
                ),
            }
        )
        common_audits.append(attempt)
        display_by_frame[frame_id]["charger_bbox_xywh"] = charger_bbox
    passing_common = sorted(
        [item for item in common_audits if item["pass"]],
        key=lambda item: (
            float(item["automatic_selection_score"]),
            -int(item["frame_id"]),
        ),
        reverse=True,
    )
    selected_common = passing_common[0] if passing_common else None
    selected_common_frame_id = (
        None
        if selected_common is None
        else int(selected_common["frame_id"])
    )
    sheet = _contact_sheet(
        display_rows, stable_ids, selected_common_frame_id
    )
    _write_jpeg_atomic(sheet_path, sheet)
    audit = {
        "schema": "inspection-waveshare-ocr-seeded-panel-owner/v1",
        "diagnostic_only": True,
        "formal_delivery_files_modified": False,
        "renderer_modified": False,
        "white_box_fastsam_complete_mask_used": False,
        "ocr_detection_reused_without_rerun": True,
        "frame_count": len(frames),
        "waveshare_ocr_detection_count": sum(
            len(value) for value in ocr_by_frame.values()
        ),
        "ocr_seeded_panel_candidate_count": len(candidates),
        "stable_panel_track_count": len(tracks),
        "stable_panel_structure_pass": bool(tracks),
        "panel_tracks": track_rows,
        "candidate_audits": candidate_rows,
        "charger_track": {
            "track_id": 0,
            "source_audit": (
                "diagnostic_fastsam_dis_tracks_audit.json"
            ),
            "selected_panel_observation_frame_ids": sorted(
                charger_observations
            ),
        },
        "shared_panel_source_owner_audits": common_audits,
        "shared_panel_source_owner_pass": selected_common is not None,
        "automatically_selected_shared_source": selected_common,
        "thresholds": {
            "minimum_lab_l": 145.0,
            "maximum_lab_chroma_distance": 28.0,
            "same_layer_depth_tolerance": "max(20_mm,0.02*seed_depth)",
            "maximum_rgb_gradient": 80.0,
            "minimum_rectangularity": 0.65,
            "minimum_solidity": 0.75,
            "minimum_rectangle_aspect_ratio": 2.0,
            "minimum_ocr_coverage": 0.90,
            "maximum_world_centroid_delta_mm": 80.0,
            "maximum_world_extent_ratio": 1.40,
            "maximum_cross_view_lab_delta": 20.0,
            "maximum_log_aspect_delta": 0.30,
            "maximum_source_gap": 12,
            "minimum_distinct_views": 2,
            "minimum_each_structure_owner_coverage": 0.90,
            "minimum_projected_in_bounds_ratio": 0.80,
            "minimum_expected_panel_projection_ratio": 0.90,
        },
        "manual_roi_used": False,
        "manual_frame_id_used": False,
        "threshold_tuning_used": False,
        "completed_real_dataset_scan_count": 1,
        "final_rgb_or_position_modified": False,
        "elapsed_seconds": time.perf_counter() - started,
        "files": {"contact_sheet": sheet_path.name},
    }
    _write_json_atomic(audit_path, audit)
    print(audit_path)
    print(sheet_path)
    print(
        json.dumps(
            {
                "ocr_seeded_panel_candidate_count": len(candidates),
                "stable_panel_track_count": len(tracks),
                "stable_panel_structure_pass": bool(tracks),
                "shared_panel_source_owner_attempt_count": len(
                    common_audits
                ),
                "shared_panel_source_owner_pass": (
                    selected_common is not None
                ),
                "selected_shared_source_frame_id": (
                    selected_common_frame_id
                ),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
