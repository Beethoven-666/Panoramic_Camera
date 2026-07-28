"""Read-only fixed-gate OCR + FastSAM + RGB-D WAVESHARE identity audit."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime
from rapidocr_onnxruntime import RapidOCR

from panorama_demo.inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    build_fastsam_rgbd_candidate,
    parse_fastsam_polygons,
    track_fastsam_rgbd_candidates,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    _read_rgbd,
    _undistortion_maps,
)
from panorama_demo.inspection_ocr_identity import (
    OCRTextDetection,
    audit_complete_white_mask,
    audit_waveshare_text,
    select_unambiguous_mask_association,
)
from panorama_demo.session import load_rgbd_session


def _as_detection(row: list[object]) -> OCRTextDetection | None:
    if len(row) != 3:
        return None
    polygon = np.asarray(row[0], dtype=np.float32)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        return None
    try:
        confidence = float(row[2])
    except (TypeError, ValueError):
        return None
    return OCRTextDetection(
        polygon_xy=np.ascontiguousarray(polygon),
        text=str(row[1]),
        confidence=confidence,
    )


def _candidate_row(
    candidate: FastSAMRGBDCandidate,
    text_audit: dict[str, object],
    association_audit: dict[str, object],
) -> dict[str, object]:
    return {
        "candidate_id": int(candidate.candidate_id),
        "frame_id": int(candidate.frame_id),
        "bbox_xywh": list(candidate.bbox_xywh),
        "source_area_pixels": int(candidate.source_area_pixels),
        "depth_coverage_ratio": float(candidate.depth_coverage_ratio),
        "world_centroid_mm": list(candidate.world_centroid_mm),
        "world_spans_mm": list(candidate.world_spans_mm),
        "median_lab": list(candidate.median_lab),
        "ocr": text_audit,
        "complete_mask_association": association_audit,
    }


def _draw_contact_sheet(
    display_rows: list[dict[str, object]],
    stable_candidate_ids: set[int],
) -> np.ndarray:
    card_width, card_height = 360, 240
    columns = 3
    ordered = sorted(
        display_rows,
        key=lambda row: (
            int(row.get("candidate_id", -1)) in stable_candidate_ids,
            bool(row.get("rgbd_candidate_pass", False)),
            float(row["text_audit"]["target_similarity"]),
            float(row["text_audit"]["confidence"]),
        ),
        reverse=True,
    )[:24]
    rows = max(1, (len(ordered) + columns - 1) // columns)
    sheet = np.full(
        (rows * card_height, columns * card_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, row in enumerate(ordered):
        image = np.asarray(row["image"])
        text_polygon = np.asarray(
            row["ocr_polygon_xy"], dtype=np.int32
        )
        candidate_polygon = row.get("candidate_polygon_xy")
        points = text_polygon.reshape(-1, 2)
        x0 = max(0, int(np.min(points[:, 0])) - 120)
        y0 = max(0, int(np.min(points[:, 1])) - 100)
        x1 = min(image.shape[1], int(np.max(points[:, 0])) + 120)
        y1 = min(image.shape[0], int(np.max(points[:, 1])) + 100)
        if candidate_polygon is not None:
            candidate_points = np.asarray(
                candidate_polygon, dtype=np.int32
            ).reshape(-1, 2)
            x0 = max(0, min(x0, int(np.min(candidate_points[:, 0])) - 16))
            y0 = max(0, min(y0, int(np.min(candidate_points[:, 1])) - 16))
            x1 = min(
                image.shape[1],
                max(x1, int(np.max(candidate_points[:, 0])) + 16),
            )
            y1 = min(
                image.shape[0],
                max(y1, int(np.max(candidate_points[:, 1])) + 16),
            )
        crop = image[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue
        shifted_text = text_polygon - np.asarray([x0, y0])
        cv2.polylines(
            crop, [shifted_text], True, (0, 220, 255), 3, cv2.LINE_AA
        )
        if candidate_polygon is not None:
            shifted_candidate = (
                np.asarray(candidate_polygon, dtype=np.int32)
                - np.asarray([x0, y0])
            )
            cv2.polylines(
                crop,
                [shifted_candidate],
                True,
                (20, 220, 20),
                3,
                cv2.LINE_AA,
            )
        available_height = card_height - 54
        scale = min(
            (card_width - 8) / crop.shape[1],
            available_height / crop.shape[0],
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
        left = column * card_width + (card_width - resized.shape[1]) // 2
        top = row_index * card_height + 48
        sheet[
            top : top + resized.shape[0],
            left : left + resized.shape[1],
        ] = resized
        candidate_id = int(row.get("candidate_id", -1))
        stable = candidate_id in stable_candidate_ids
        state = (
            "STABLE COMPLETE"
            if stable
            else (
                "COMPLETE SINGLE VIEW"
                if row.get("rgbd_candidate_pass", False)
                else "TEXT ONLY / MASK REJECT"
            )
        )
        label = (
            f"F{int(row['frame_id'])} {state} "
            f"{row['text_audit']['normalized_text']} "
            f"{float(row['text_audit']['confidence']):.2f}"
        )
        cv2.putText(
            sheet,
            label[:55],
            (column * card_width + 5, row_index * card_height + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 80, 0) if stable else (40, 40, 180),
            1,
            cv2.LINE_AA,
        )
    if not ordered:
        cv2.putText(
            sheet,
            "No OCR detection passed the fixed WAVESHARE text gate",
            (16, 60),
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
        raise RuntimeError("Could not encode OCR contact sheet")
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
    audit_path = output / "diagnostic_waveshare_ocr_rgbd_audit.json"
    sheet_path = (
        output / "diagnostic_waveshare_ocr_rgbd_contact_sheet.jpg"
    )
    if audit_path.exists() or sheet_path.exists():
        raise RuntimeError(
            "OCR diagnostic outputs already exist; fixed-gate audit "
            "will not be rerun or tuned"
        )

    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    reference_depth_mm = float(
        report["render"]["layout"]["reference_depth_mm"]
    )
    session = load_rgbd_session(session_path)
    frames = sorted(session.frames, key=lambda item: int(item.frame_id))
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    maps = _undistortion_maps(session.calibration)
    engine = RapidOCR()

    frame_audits: list[dict[str, object]] = []
    candidates: list[FastSAMRGBDCandidate] = []
    candidates_by_frame: list[list[FastSAMRGBDCandidate]] = []
    candidate_audit_by_id: dict[int, dict[str, object]] = {}
    display_rows: list[dict[str, object]] = []
    total_ocr_detections = 0
    target_text_detection_count = 0
    complete_mask_association_count = 0
    rgbd_candidate_rejection_count = 0

    for frame_index, frame in enumerate(frames):
        frame_id = int(frame.frame_id)
        image, depth, geometric_valid = _read_rgbd(
            frame, session.calibration, maps
        )
        raw_result, _ = engine(image)
        detections = [
            detection
            for row in (raw_result or [])
            if (detection := _as_detection(row)) is not None
        ]
        total_ocr_detections += len(detections)
        polygons = parse_fastsam_polygons(
            labels_path / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        frame_candidates: list[FastSAMRGBDCandidate] = []
        frame_target_rows: list[dict[str, object]] = []
        used_polygon_indices: set[int] = set()
        for detection in detections:
            text_audit = audit_waveshare_text(detection)
            if not text_audit["pass"]:
                continue
            target_text_detection_count += 1
            mask_audits = [
                audit_complete_white_mask(
                    ocr_polygon_xy=detection.polygon_xy,
                    candidate_polygon_xy=polygon,
                    lab_image=lab,
                )
                for polygon in polygons
            ]
            selected_polygon_index = select_unambiguous_mask_association(
                mask_audits
            )
            ranked_mask_audits = sorted(
                (
                    {
                        "polygon_index": index,
                        **mask_audit,
                    }
                    for index, mask_audit in enumerate(mask_audits)
                ),
                key=lambda row: float(row.get("score", -1.0)),
                reverse=True,
            )[:5]
            target_row: dict[str, object] = {
                "frame_id": frame_id,
                "ocr_polygon_xy": detection.polygon_xy.tolist(),
                "text_audit": text_audit,
                "fastsam_polygon_count": len(polygons),
                "passing_complete_mask_count": sum(
                    int(bool(row.get("pass", False)))
                    for row in mask_audits
                ),
                "selected_polygon_index": selected_polygon_index,
                "top_mask_audits": ranked_mask_audits,
                "rgbd_candidate_pass": False,
            }
            if (
                selected_polygon_index is None
                or selected_polygon_index in used_polygon_indices
            ):
                frame_target_rows.append(target_row)
                display_rows.append(
                    {
                        **target_row,
                        "image": image.copy(),
                    }
                )
                continue
            complete_mask_association_count += 1
            selected_polygon = polygons[selected_polygon_index]
            pose = pose_by_id.get(frame_id)
            candidate = None
            if pose is not None:
                candidate = build_fastsam_rgbd_candidate(
                    candidate_id=len(candidates),
                    source_index=frame_index,
                    frame_id=frame_id,
                    polygon_xy=selected_polygon,
                    image_bgr=image,
                    lab_image=lab,
                    depth_mm=depth,
                    reliable_depth=reliable,
                    camera_to_world=pose,
                    intrinsics=session.calibration,
                    reference_depth_mm=reference_depth_mm,
                )
            if candidate is None:
                rgbd_candidate_rejection_count += 1
                target_row["rgbd_rejection_reason"] = (
                    "real_pose_unavailable"
                    if pose is None
                    else "existing_strict_rgbd_candidate_gate_failed"
                )
            else:
                used_polygon_indices.add(selected_polygon_index)
                candidates.append(candidate)
                frame_candidates.append(candidate)
                target_row["rgbd_candidate_pass"] = True
                target_row["candidate_id"] = candidate.candidate_id
                candidate_audit_by_id[candidate.candidate_id] = (
                    _candidate_row(
                        candidate,
                        text_audit,
                        mask_audits[selected_polygon_index],
                    )
                )
            frame_target_rows.append(target_row)
            display_rows.append(
                {
                    **target_row,
                    "candidate_polygon_xy": selected_polygon.copy(),
                    "image": image.copy(),
                }
            )
        candidates_by_frame.append(frame_candidates)
        frame_audits.append(
            {
                "frame_id": frame_id,
                "ocr_detection_count": len(detections),
                "waveshare_text_detection_count": len(frame_target_rows),
                "complete_rgbd_candidate_count": len(frame_candidates),
                "target_detections": frame_target_rows,
            }
        )
        if (frame_index + 1) % 16 == 0:
            print(
                f"OCR audited {frame_index + 1}/{len(frames)} frames",
                flush=True,
            )

    tracks = track_fastsam_rgbd_candidates(
        candidates_by_frame,
        minimum_voxel_overlap_ratio=0.25,
        maximum_lab_delta=30.0,
        maximum_source_gap=12,
    )
    candidate_by_id = {
        candidate.candidate_id: candidate for candidate in candidates
    }
    track_rows: list[dict[str, object]] = []
    stable_candidate_ids: set[int] = set()
    for track in tracks:
        frame_ids = [
            int(candidate_by_id[value].frame_id)
            for value in track.candidate_ids
        ]
        stable_candidate_ids.update(track.candidate_ids)
        track_rows.append(
            {
                "track_id": int(track.track_id),
                "candidate_ids": list(track.candidate_ids),
                "frame_ids": frame_ids,
                "view_count": len(set(frame_ids)),
                "audit": track.audit,
                "candidates": [
                    candidate_audit_by_id[value]
                    for value in track.candidate_ids
                ],
            }
        )
    identity_pass = any(row["view_count"] >= 2 for row in track_rows)
    sheet = _draw_contact_sheet(display_rows, stable_candidate_ids)
    _write_jpeg_atomic(sheet_path, sheet)

    audit: dict[str, object] = {
        "schema": "inspection-waveshare-ocr-rgbd-identity/v1",
        "formal_delivery_files_modified": False,
        "renderer_modified": False,
        "diagnostic_only": True,
        "model": {
            "package": "rapidocr-onnxruntime",
            "version": importlib.metadata.version(
                "rapidocr-onnxruntime"
            ),
            "license": importlib.metadata.metadata(
                "rapidocr-onnxruntime"
            ).get("License"),
            "onnxruntime_version": onnxruntime.__version__,
            "available_execution_providers": (
                onnxruntime.get_available_providers()
            ),
            "role": "text_identity_anchor_only",
        },
        "frame_count": len(frames),
        "label_frame_count": len(list(labels_path.glob("*.txt"))),
        "real_pose_frame_count": len(pose_by_id),
        "ocr_detection_count": total_ocr_detections,
        "waveshare_text_detection_count": target_text_detection_count,
        "complete_fastsam_mask_association_count": (
            complete_mask_association_count
        ),
        "complete_rgbd_candidate_count": len(candidates),
        "rgbd_candidate_rejection_count": rgbd_candidate_rejection_count,
        "stable_complete_white_box_track_count": len(track_rows),
        "complete_white_box_identity_pass": identity_pass,
        "minimum_required_distinct_views": 2,
        "tracks": track_rows,
        "frame_audits": frame_audits,
        "thresholds": {
            "minimum_ocr_confidence": 0.45,
            "minimum_waveshare_edit_similarity": 0.70,
            "normalized_text_length_range": [7, 14],
            "minimum_ocr_polygon_coverage_by_mask": 0.80,
            "candidate_to_text_area_ratio_range": [4.0, 90.0],
            "candidate_to_text_width_ratio_range": [1.25, 8.0],
            "candidate_to_text_height_ratio_range": [1.50, 12.0],
            "minimum_candidate_median_lab_l": 145.0,
            "maximum_candidate_median_lab_chroma_distance": 28.0,
            "mask_association_ambiguity_margin": 0.05,
            "minimum_world_voxel_overlap_ratio": 0.25,
            "maximum_cross_view_lab_delta": 30.0,
            "maximum_source_frame_gap": 12,
            "minimum_distinct_views": 2,
        },
        "manual_roi_used": False,
        "manual_frame_id_used": False,
        "threshold_tuning_used": False,
        "completed_real_dataset_scan_count": 1,
        "pre_scan_initialization_timeout_restart_count": 1,
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
                "ocr_detection_count": total_ocr_detections,
                "waveshare_text_detection_count": (
                    target_text_detection_count
                ),
                "complete_rgbd_candidate_count": len(candidates),
                "stable_complete_white_box_track_count": len(track_rows),
                "complete_white_box_identity_pass": identity_pass,
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
