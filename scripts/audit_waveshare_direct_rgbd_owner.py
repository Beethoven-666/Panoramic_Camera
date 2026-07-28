"""Direct SE(3) complete-owner audit for stable OCR-seeded panels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.dis_track_direct_handoff import (
    DirectHandoffConfig,
    DirectProjectedObservation,
    evaluate_direct_track,
)
from panorama_demo.inspection_fastsam_track import parse_fastsam_polygons
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
from panorama_demo.inspection_ocr_panel import (
    audit_relative_world_geometry,
    extract_ocr_seeded_panel,
    sample_mask_world_points,
)
from panorama_demo.session import load_rgbd_session


def _layout_from_report(
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


def _write_jpeg_atomic(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode direct owner contact sheet")
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
    observations: list[DirectProjectedObservation],
    selected_frame_id: int | None,
) -> np.ndarray:
    card_width, card_height = 330, 230
    columns = 4
    selected = observations[:12]
    rows = max(1, (len(selected) + columns - 1) // columns)
    sheet = np.full(
        (rows * card_height, columns * card_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, observation in enumerate(selected):
        yy, xx = np.nonzero(observation.target_mask)
        if not xx.size:
            continue
        x0, x1 = max(0, int(np.min(xx)) - 12), min(
            observation.target_mask.shape[1], int(np.max(xx)) + 13
        )
        y0, y1 = max(0, int(np.min(yy)) - 12), min(
            observation.target_mask.shape[0], int(np.max(yy)) + 13
        )
        crop = observation.target_image_bgr[y0:y1, x0:x1].copy()
        mask = observation.target_mask[y0:y1, x0:x1]
        contour_rows, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(crop, contour_rows, -1, (20, 220, 20), 2)
        scale = min(
            (card_width - 8) / crop.shape[1],
            (card_height - 45) / crop.shape[0],
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
        row = index // columns
        left = column * card_width + (
            card_width - resized.shape[1]
        ) // 2
        top = row * card_height + 40
        sheet[
            top : top + resized.shape[0],
            left : left + resized.shape[1],
        ] = resized
        is_selected = observation.frame_id == selected_frame_id
        cv2.putText(
            sheet,
            (
                f"F{observation.frame_id} "
                f"{'SELECTED OWNER' if is_selected else 'DIRECT SE3'}"
            ),
            (column * card_width + 5, row * card_height + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 90, 0) if is_selected else (40, 40, 180),
            1,
            cv2.LINE_AA,
        )
    if not selected:
        cv2.putText(
            sheet,
            "No fixed-gate direct RGB-D projection survived",
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
    parser.add_argument("labels")
    arguments = parser.parse_args()
    started = time.perf_counter()
    session_path = Path(arguments.session).expanduser().resolve()
    output = Path(arguments.formal_output).expanduser().resolve()
    labels_path = Path(arguments.labels).expanduser().resolve()
    audit_path = (
        output / "diagnostic_waveshare_direct_rgbd_owner_audit.json"
    )
    sheet_path = (
        output / "diagnostic_waveshare_direct_rgbd_owner_contact_sheet.jpg"
    )
    if audit_path.exists() or sheet_path.exists():
        raise RuntimeError(
            "Direct owner outputs already exist; fixed-gate audit "
            "will not be rerun or tuned"
        )
    ocr_audit = json.loads(
        (
            output / "diagnostic_waveshare_ocr_rgbd_audit.json"
        ).read_text(encoding="utf-8")
    )
    seeded_audit = json.loads(
        (
            output / "diagnostic_waveshare_seeded_panel_audit.json"
        ).read_text(encoding="utf-8")
    )
    object_rich_audit = json.loads(
        (
            output
            / "diagnostic_waveshare_object_rich_corridor_audit.json"
        ).read_text(encoding="utf-8")
    )
    dis_audit = json.loads(
        (
            output / "diagnostic_fastsam_dis_tracks_audit.json"
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
    layout = _layout_from_report(report["render"]["layout"])
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    maps = _undistortion_maps(session.calibration)
    ocr_by_frame = {
        int(item["frame_id"]): item["target_detections"][0]
        for item in ocr_audit["frame_audits"]
        if item["target_detections"]
    }
    stable_frames = sorted(
        {
            int(frame_id)
            for track in seeded_audit["panel_tracks"]
            for frame_id in track["frame_ids"]
        }
    )
    cache: dict[int, dict[str, object]] = {}
    panels = []
    panel_centroids = {}
    for frame_id in stable_frames:
        if frame_id not in pose_by_id or frame_id not in ocr_by_frame:
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
        panel, _ = extract_ocr_seeded_panel(
            frame_id=frame_id,
            source_index=source_index_by_id[frame_id],
            image_bgr=image,
            depth_mm=depth,
            reliable_depth=reliable,
            ocr_polygon_xy=np.asarray(
                ocr_by_frame[frame_id]["ocr_polygon_xy"],
                dtype=np.float32,
            ),
            camera_to_world=pose_by_id[frame_id],
            intrinsics=session.calibration,
        )
        if panel is None:
            continue
        cache[frame_id] = {
            "image": image,
            "depth": depth,
            "reliable": reliable,
        }
        panels.append(panel)
        panel_centroids[frame_id] = panel.world_centroid_mm
    if len(panels) < 2:
        raise RuntimeError("Stable OCR panel masks cannot be reconstructed")
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    anchors = np.asarray(
        [item.anchor_scan_mm for item in layout.panels],
        dtype=np.float64,
    )
    median_panel_world = np.median(
        np.asarray([item.world_centroid_mm for item in panels]), axis=0
    )
    target_panel_index = int(
        np.argmin(
            np.abs(anchors - float(median_panel_world @ scan_axis))
        )
    )
    fixed = DirectHandoffConfig()
    fixed.validate()
    projections: list[DirectProjectedObservation] = []
    projection_rejections: list[dict[str, object]] = []
    for candidate_id, panel in enumerate(panels):
        frame_id = int(panel.frame_id)
        source = cache[frame_id]
        source_count = int(np.count_nonzero(panel.mask))
        reliable_count = int(
            np.count_nonzero(panel.mask & source["reliable"])
        )
        depth_coverage = float(
            reliable_count / max(1, source_count)
        )
        source_panel_index = int(
            np.argmin(
                np.abs(
                    anchors
                    - float(pose_by_id[frame_id][:3, 3] @ scan_axis)
                )
            )
        )
        try:
            owner = project_complete_object_owner_from_rgbd(
                source_image_bgr=source["image"],
                source_depth_mm=source["depth"],
                source_reliable_depth=source["reliable"],
                source_object_mask=panel.mask,
                camera_to_world=pose_by_id[frame_id],
                layout=layout,
                intrinsics=session.calibration,
                frame_id=frame_id,
                panel_index=target_panel_index,
                minimum_cells=64,
            )
        except (RuntimeError, ValueError) as exc:
            projection_rejections.append(
                {
                    "candidate_id": candidate_id,
                    "frame_id": frame_id,
                    "source_depth_coverage_ratio": depth_coverage,
                    "reason": "direct_rgbd_projection_failed",
                    "detail": str(exc),
                }
            )
            continue
        projections.append(
            DirectProjectedObservation(
                candidate_id=candidate_id,
                frame_id=frame_id,
                source_panel_index=source_panel_index,
                target_panel_index=target_panel_index,
                target_mask=owner.target_mask,
                target_image_bgr=owner.target_image_bgr,
                source_depth_coverage_ratio=depth_coverage,
                clarity=panel.clarity_variance,
                projection_audit=owner.audit,
            )
        )
    decision = evaluate_direct_track(
        target_panel_index,
        projections,
        config=fixed,
    )

    automatically_selected_tracks = [
        int(value)
        for value in object_rich_audit[
            "automatically_selected_object_track_ids"
        ]
    ]
    track_by_id = {
        int(item["track_id"]): item
        for item in dis_audit["stable_selected_panel_tracks"]
    }
    compact_track_id = min(
        automatically_selected_tracks,
        key=lambda track_id: float(
            np.median(
                track_by_id[track_id][
                    "selected_panel_source_area_pixels"
                ]
            )
        ),
    )
    compact_track = track_by_id[compact_track_id]
    object_centroids = {}
    relative_observation_rows = []
    for frame_id, bbox in zip(
        compact_track["selected_panel_frame_ids"],
        compact_track["selected_panel_bboxes_xywh"],
        strict=True,
    ):
        frame_id = int(frame_id)
        if frame_id not in panel_centroids:
            continue
        if frame_id not in cache:
            image, depth, geometric_valid = _read_rgbd(
                frame_by_id[frame_id], session.calibration, maps
            )
            reliable = (
                geometric_valid
                & np.isfinite(depth)
                & (depth >= config.minimum_depth_mm)
                & (depth <= config.maximum_depth_mm)
            )
        else:
            image = cache[frame_id]["image"]
            depth = cache[frame_id]["depth"]
            reliable = cache[frame_id]["reliable"]
        polygons = parse_fastsam_polygons(
            labels_path / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        bbox_tuple = tuple(int(value) for value in bbox)
        matches = [
            polygon
            for polygon in polygons
            if tuple(int(value) for value in cv2.boundingRect(polygon))
            == bbox_tuple
        ]
        if len(matches) != 1:
            continue
        mask = _polygon_mask(matches[0], depth.shape)
        world = sample_mask_world_points(
            mask=mask,
            depth_mm=depth,
            reliable_depth=reliable,
            camera_to_world=pose_by_id[frame_id],
            intrinsics=session.calibration,
        )
        if world.shape[0] < 30:
            continue
        centroid = np.median(world, axis=0)
        object_centroids[frame_id] = centroid.tolist()
        relative_observation_rows.append(
            {
                "frame_id": frame_id,
                "bbox_xywh": list(bbox_tuple),
                "world_sample_count": int(world.shape[0]),
                "world_centroid_mm": centroid.tolist(),
            }
        )
    relative_geometry = audit_relative_world_geometry(
        panel_centroids, object_centroids
    )
    overall_pass = bool(
        decision.accepted and relative_geometry["pass"]
    )
    selected_frame_id = (
        None
        if decision.selected_observation is None
        else int(decision.selected_observation.frame_id)
    )
    sheet = _contact_sheet(projections, selected_frame_id)
    _write_jpeg_atomic(sheet_path, sheet)
    audit = {
        "schema": "inspection-waveshare-direct-rgbd-owner/v1",
        "diagnostic_only": True,
        "formal_delivery_files_modified": False,
        "renderer_modified": False,
        "post_render_overlay_used": False,
        "projection_function": (
            "project_complete_object_owner_from_rgbd"
        ),
        "true_camera_to_world_se3_only": True,
        "stable_panel_mask_count": len(panels),
        "direct_projection_count": len(projections),
        "direct_projection_rejections": projection_rejections,
        "automatic_target_panel_index": target_panel_index,
        "direct_owner_decision": decision.audit,
        "complete_white_panel_direct_owner_pass": decision.accepted,
        "automatic_compact_neighbor_track_selection": {
            "candidate_track_ids_from_object_rich_audit": (
                automatically_selected_tracks
            ),
            "selection_rule": (
                "minimum_median_selected_panel_source_area"
            ),
            "selected_track_id": compact_track_id,
            "hard_coded_track_id_used": False,
        },
        "compact_neighbor_world_observations": relative_observation_rows,
        "panel_to_compact_neighbor_relative_world_geometry": (
            relative_geometry
        ),
        "relative_position_pass": relative_geometry["pass"],
        "overall_direct_owner_and_relative_position_pass": overall_pass,
        "thresholds": {
            **fixed.__dict__,
            "maximum_relative_vector_deviation_mm": 80.0,
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
                "stable_panel_mask_count": len(panels),
                "direct_projection_count": len(projections),
                "complete_white_panel_direct_owner_pass": (
                    decision.accepted
                ),
                "selected_owner_frame_id": selected_frame_id,
                "automatic_compact_neighbor_track_id": compact_track_id,
                "relative_position_pass": relative_geometry["pass"],
                "overall_pass": overall_pass,
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
