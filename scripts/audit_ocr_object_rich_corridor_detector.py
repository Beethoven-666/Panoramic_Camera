"""Fixed-gate OCR/RGB-D object-rich corridor detector validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    _read_rgbd,
    _undistortion_maps,
)
from panorama_demo.inspection_object_rich_corridor import (
    ObjectRichCorridor,
    extract_object_rich_corridor,
    interval_pair_metrics,
    track_object_rich_corridors,
)
from panorama_demo.inspection_ocr_panel import (
    extract_ocr_seeded_panel,
    track_ocr_seeded_panels,
)
from panorama_demo.session import load_rgbd_session


def _write_jpeg_atomic(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode corridor detector sheet")
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


def _contact_sheet(
    corridors: list[ObjectRichCorridor],
    image_by_frame: dict[int, np.ndarray],
    accepted_ids: set[int],
    selected_id: int | None,
) -> np.ndarray:
    card_width, card_height = 500, 300
    columns = 3
    ordered = sorted(
        enumerate(corridors),
        key=lambda item: (
            item[0] in accepted_ids,
            item[0] == selected_id,
            item[1].clarity_variance,
        ),
        reverse=True,
    )[:18]
    rows = max(1, (len(ordered) + columns - 1) // columns)
    sheet = np.full(
        (rows * card_height, columns * card_width, 3),
        245,
        dtype=np.uint8,
    )
    palette = (
        (255, 180, 20),
        (220, 30, 220),
        (20, 160, 255),
        (180, 100, 20),
    )
    for card_index, (corridor_id, corridor) in enumerate(ordered):
        image = image_by_frame[corridor.frame_id].copy()
        cv2.polylines(
            image,
            [corridor.panel.contour_xy.astype(np.int32)],
            True,
            (20, 220, 20),
            4,
            cv2.LINE_AA,
        )
        for structure_index, structure in enumerate(
            corridor.structures
        ):
            cv2.polylines(
                image,
                [structure.contour_xy.astype(np.int32)],
                True,
                palette[structure_index % len(palette)],
                4,
                cv2.LINE_AA,
            )
        x0, y0, x1, y1 = corridor.interval_xyxy
        cv2.rectangle(
            image, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 2
        )
        cv2.line(
            image, (corridor.left_endpoint_x, y0), (corridor.left_endpoint_x, y1), (0, 0, 255), 3
        )
        cv2.line(
            image, (corridor.right_endpoint_x, y0), (corridor.right_endpoint_x, y1), (0, 0, 255), 3
        )
        pad = 24
        crop_x0, crop_y0 = max(0, x0 - pad), max(0, y0 - pad)
        crop_x1 = min(image.shape[1], x1 + pad)
        crop_y1 = min(image.shape[0], y1 + pad)
        crop = image[crop_y0:crop_y1, crop_x0:crop_x1]
        scale = min(
            (card_width - 8) / crop.shape[1],
            (card_height - 48) / crop.shape[0],
        )
        resized = cv2.resize(
            crop,
            (
                max(1, int(round(crop.shape[1] * scale))),
                max(1, int(round(crop.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
        column = card_index % columns
        row = card_index // columns
        left = column * card_width + (
            card_width - resized.shape[1]
        ) // 2
        top = row * card_height + 42
        sheet[
            top : top + resized.shape[0],
            left : left + resized.shape[1],
        ] = resized
        selected = corridor_id == selected_id
        stable = corridor_id in accepted_ids
        label = (
            f"F{corridor.frame_id} "
            f"{'SELECTED' if selected else ('STABLE' if stable else 'SINGLE')} "
            f"N={len(corridor.structures)} "
            f"risk={max(corridor.left_endpoint_risk_ratio, corridor.right_endpoint_risk_ratio):.2f}"
        )
        cv2.putText(
            sheet,
            label,
            (column * card_width + 6, row * card_height + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 90, 0) if stable else (40, 40, 180),
            1,
            cv2.LINE_AA,
        )
    if not ordered:
        cv2.putText(
            sheet,
            "No OCR-anchored corridor passed fixed gates",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 180),
            2,
            cv2.LINE_AA,
        )
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("formal_output")
    arguments = parser.parse_args()
    started = time.perf_counter()
    session_path = Path(arguments.session).expanduser().resolve()
    output = Path(arguments.formal_output).expanduser().resolve()
    audit_path = (
        output
        / "diagnostic_ocr_object_rich_corridor_detector_audit.json"
    )
    sheet_path = (
        output
        / "diagnostic_ocr_object_rich_corridor_detector_contact_sheet.jpg"
    )
    if audit_path.exists() or sheet_path.exists():
        raise RuntimeError(
            "Corridor detector outputs already exist; fixed-gate "
            "validation will not be rerun or tuned"
        )
    ocr_audit = json.loads(
        (
            output / "diagnostic_waveshare_ocr_rgbd_audit.json"
        ).read_text(encoding="utf-8")
    )
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
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
    layout = report["render"]["layout"]
    reference_depth_mm = float(layout["reference_depth_mm"])
    scan_axis = tuple(float(value) for value in layout["scan_axis_world"])
    maps = _undistortion_maps(session.calibration)
    ocr_by_frame = {
        int(item["frame_id"]): item["target_detections"]
        for item in ocr_audit["frame_audits"]
        if item["target_detections"]
    }
    panels = []
    panel_source_rows = []
    image_by_frame: dict[int, np.ndarray] = {}
    rgbd_by_frame: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for frame_id, detections in sorted(ocr_by_frame.items()):
        if frame_id not in pose_by_id or frame_id not in frame_by_id:
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
            panel, panel_audit = extract_ocr_seeded_panel(
                frame_id=frame_id,
                source_index=source_index_by_id[frame_id],
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=reliable,
                ocr_polygon_xy=np.asarray(
                    detection["ocr_polygon_xy"], dtype=np.float32
                ),
                camera_to_world=pose_by_id[frame_id],
                intrinsics=session.calibration,
            )
            panel_source_rows.append(
                {
                    "frame_id": frame_id,
                    "pass": panel is not None,
                    "audit": panel_audit,
                }
            )
            if panel is not None:
                panels.append(panel)
                image_by_frame[frame_id] = image
                rgbd_by_frame[frame_id] = (
                    depth,
                    reliable,
                    geometric_valid,
                )
    panel_tracks = track_ocr_seeded_panels(panels)
    stable_panel_indices = {
        index for track in panel_tracks for index in track
    }
    corridors: list[ObjectRichCorridor] = []
    corridor_source_rows = []
    for panel_index in sorted(stable_panel_indices):
        panel = panels[panel_index]
        depth, reliable, geometric_valid = rgbd_by_frame[panel.frame_id]
        corridor, corridor_audit = extract_object_rich_corridor(
            panel=panel,
            image_bgr=image_by_frame[panel.frame_id],
            depth_mm=depth,
            reliable_depth=reliable,
            geometric_valid=geometric_valid,
            camera_to_world=pose_by_id[panel.frame_id],
            intrinsics=session.calibration,
            reference_depth_mm=reference_depth_mm,
            scan_axis_world=scan_axis,
        )
        corridor_source_rows.append(
            {
                "panel_candidate_index": panel_index,
                "frame_id": panel.frame_id,
                "pass": corridor is not None,
                "audit": corridor_audit,
            }
        )
        if corridor is not None:
            corridors.append(corridor)
    tracks = track_object_rich_corridors(corridors)
    accepted_ids = {
        index for track in tracks for index in track
    }
    pair_rows = []
    for track_index, track in enumerate(tracks):
        for first_offset, first_index in enumerate(track):
            for second_index in track[first_offset + 1 :]:
                first = corridors[first_index]
                second = corridors[second_index]
                iou, coverage = interval_pair_metrics(
                    first.relative_scan_range_mm,
                    second.relative_scan_range_mm,
                )
                pair_rows.append(
                    {
                        "track_index": track_index,
                        "first_frame_id": first.frame_id,
                        "second_frame_id": second.frame_id,
                        "relative_scan_range_iou": iou,
                        "smaller_range_coverage": coverage,
                        "pass": bool(iou >= 0.50 and coverage >= 0.75),
                    }
                )
    stable_track = (
        max(
            tracks,
            key=lambda track: (
                len(track),
                min(
                    corridors[index].inverse_map_coverage_ratio
                    for index in track
                ),
            ),
        )
        if tracks
        else ()
    )
    selected_id = (
        max(
            stable_track,
            key=lambda index: (
                corridors[index].inverse_map_coverage_ratio,
                -max(
                    corridors[index].left_endpoint_risk_ratio,
                    corridors[index].right_endpoint_risk_ratio,
                ),
                corridors[index].clarity_variance,
                -corridors[index].frame_id,
            ),
        )
        if stable_track
        else None
    )
    selected = None if selected_id is None else corridors[selected_id]
    sheet = _contact_sheet(
        corridors, image_by_frame, accepted_ids, selected_id
    )
    _write_jpeg_atomic(sheet_path, sheet)
    audit = {
        "schema": "inspection-ocr-object-rich-corridor-detector/v1",
        "diagnostic_only": True,
        "formal_delivery_files_modified": False,
        "renderer_modified": False,
        "detector_dependencies": [
            "OCR_polygon",
            "raw_RGB_gradient",
            "raw_RGB_Lab",
            "aligned_depth",
            "real_camera_to_world_SE3",
            "calibration_inverse_map_validity",
        ],
        "fastsam_audit_read": False,
        "dis_audit_read": False,
        "fastsam_labels_read": False,
        "ocr_audit_fields_consumed": [
            "frame_id",
            "target_detections[].ocr_polygon_xy",
            "target_detections[].text_audit",
        ],
        "ocr_panel_candidate_count": len(panels),
        "stable_ocr_panel_track_count": len(panel_tracks),
        "corridor_candidate_count": len(corridors),
        "stable_corridor_track_count": len(tracks),
        "stable_corridor_tracks": [
            {
                "track_index": track_index,
                "corridor_indices": list(track),
                "frame_ids": [
                    corridors[index].frame_id for index in track
                ],
                "distinct_view_count": len(
                    {corridors[index].frame_id for index in track}
                ),
            }
            for track_index, track in enumerate(tracks)
        ],
        "cross_view_pair_audits": pair_rows,
        "detector_validation_pass": selected is not None,
        "automatically_selected_source": (
            None
            if selected is None
            else {
                "corridor_index": selected_id,
                "frame_id": selected.frame_id,
                "source_index": selected.source_index,
                "interval_xyxy": list(selected.interval_xyxy),
                "selected_structure_count": len(selected.structures),
                "relative_scan_range_mm": list(
                    selected.relative_scan_range_mm
                ),
                "left_endpoint_risk_ratio": (
                    selected.left_endpoint_risk_ratio
                ),
                "right_endpoint_risk_ratio": (
                    selected.right_endpoint_risk_ratio
                ),
                "inverse_map_coverage_ratio": (
                    selected.inverse_map_coverage_ratio
                ),
                "clarity_laplacian_variance": (
                    selected.clarity_variance
                ),
                "selection_rule": (
                    "stable_track_length_then_full_inverse_coverage_"
                    "then_endpoint_risk_then_clarity"
                ),
            }
        ),
        "panel_candidate_audits": panel_source_rows,
        "corridor_candidate_audits": corridor_source_rows,
        "thresholds": {
            "maximum_structure_x_gap_pixels": 160,
            "minimum_structure_area_pixels": 300,
            "maximum_structure_area_image_ratio": 0.08,
            "maximum_broad_thin_aspect_ratio": 8.0,
            "minimum_structure_depth_coverage_ratio": 0.85,
            "minimum_internal_depth_continuity_ratio": 0.70,
            "foreground_depth_limit": (
                "reference_depth-max(60_mm,0.08*reference_depth)"
            ),
            "rgb_gradient_structure_threshold": 24.0,
            "maximum_endpoint_risk_ratio": 0.15,
            "required_inverse_map_coverage_ratio": 1.0,
            "minimum_relative_scan_range_iou": 0.50,
            "minimum_smaller_range_coverage": 0.75,
            "maximum_source_gap": 12,
            "minimum_distinct_views": 2,
        },
        "manual_roi_used": False,
        "manual_frame_id_used": False,
        "hard_coded_object_used": False,
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
                "ocr_panel_candidate_count": len(panels),
                "stable_ocr_panel_track_count": len(panel_tracks),
                "corridor_candidate_count": len(corridors),
                "stable_corridor_track_count": len(tracks),
                "detector_validation_pass": selected is not None,
                "selected_source_frame_id": (
                    None if selected is None else selected.frame_id
                ),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
