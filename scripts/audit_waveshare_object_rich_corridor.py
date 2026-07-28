"""Automatic pre-seam object-rich corridor audit without hard-coded IDs."""

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
    StableObjectTrackEvidence,
    audit_object_rich_interval,
    extract_ocr_seeded_panel,
    sample_mask_world_points,
    select_object_rich_neighbor_tracks,
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


def _resolve_polygon(
    polygons: list[np.ndarray],
    bbox_xywh: tuple[int, int, int, int],
) -> np.ndarray | None:
    matches = [
        polygon
        for polygon in polygons
        if tuple(int(value) for value in cv2.boundingRect(polygon))
        == bbox_xywh
    ]
    return matches[0] if len(matches) == 1 else None


def _project_to_specific_panel(
    points_world_mm: np.ndarray,
    *,
    layout: InspectionMultiviewLayout,
    panel_index: int,
    intrinsics,
) -> dict[str, object]:
    panel = layout.panels[panel_index]
    relative = points_world_mm - np.asarray(
        panel.center_world_mm, dtype=np.float64
    )
    q_scan = relative @ np.asarray(layout.scan_axis)
    q_down = relative @ np.asarray(layout.down_axis)
    q_normal = relative @ np.asarray(layout.normal_axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (
            panel.canvas_offset_x
            + intrinsics.cx
            + intrinsics.fx * q_scan / q_normal
        )
        y = intrinsics.cy + intrinsics.fy * q_down / q_normal
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(q_normal)
        & (q_normal > 0.0)
        & (x >= 0.0)
        & (x < layout.width)
        & (y >= 0.0)
        & (y < layout.height)
    )
    count = int(np.count_nonzero(valid))
    ratio = float(count / max(1, points_world_mm.shape[0]))
    return {
        "sample_count": int(points_world_mm.shape[0]),
        "in_panel_sample_count": count,
        "projected_in_bounds_ratio": ratio,
        "x_span_pixels": (
            [float(np.min(x[valid])), float(np.max(x[valid]))]
            if count
            else None
        ),
        "y_span_pixels": (
            [float(np.min(y[valid])), float(np.max(y[valid]))]
            if count
            else None
        ),
        "pass": bool(count >= 30 and ratio >= 0.90),
    }


def _current_owner_votes(
    points_world_mm: np.ndarray,
    *,
    layout: InspectionMultiviewLayout,
    intrinsics,
    crop: dict[str, object],
    owner_frame_id: np.ndarray,
) -> dict[str, object]:
    x, y, q_normal, _ = project_world_points_to_panels(
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
    values, counts = np.unique(
        owner_frame_id[y[valid], x[valid]], return_counts=True
    )
    order = np.argsort(-counts, kind="stable")
    return {
        "in_bounds_sample_count": int(np.count_nonzero(valid)),
        "owner_vote_top": [
            {"frame_id": int(value), "sample_count": int(count)}
            for value, count in zip(
                values[order][:5], counts[order][:5], strict=False
            )
        ],
    }


def _write_jpeg_atomic(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise RuntimeError("Could not encode object-rich contact sheet")
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
    rows: list[dict[str, object]],
    selected_frame_id: int | None,
) -> np.ndarray:
    card_width, card_height = 700, 390
    columns = 2
    height = max(
        card_height, math.ceil(max(1, len(rows)) / columns) * card_height
    )
    sheet = np.full(
        (height, columns * card_width, 3), 245, dtype=np.uint8
    )
    colours = ((20, 220, 20), (255, 180, 20), (220, 30, 220))
    for index, row in enumerate(rows):
        image = np.asarray(row["image"]).copy()
        contours = row["contours"]
        for contour, colour in zip(contours, colours, strict=True):
            cv2.polylines(
                image,
                [np.asarray(contour, dtype=np.int32)],
                True,
                colour,
                4,
                cv2.LINE_AA,
            )
        all_points = np.concatenate(
            [np.asarray(item).reshape(-1, 2) for item in contours]
        )
        x0 = max(0, int(np.min(all_points[:, 0])) - 40)
        y0 = max(0, int(np.min(all_points[:, 1])) - 40)
        x1 = min(image.shape[1], int(np.max(all_points[:, 0])) + 40)
        y1 = min(image.shape[0], int(np.max(all_points[:, 1])) + 40)
        crop = image[y0:y1, x0:x1]
        scale = min(
            (card_width - 10) / crop.shape[1],
            (card_height - 55) / crop.shape[0],
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
        left = column * card_width + (
            card_width - resized.shape[1]
        ) // 2
        top = row_index * card_height + 48
        sheet[
            top : top + resized.shape[0],
            left : left + resized.shape[1],
        ] = resized
        selected = int(row["frame_id"]) == selected_frame_id
        label = (
            f"F{int(row['frame_id'])} "
            f"{'SELECTED PRE-SEAM INTERVAL' if selected else 'CANDIDATE'} "
            f"pass={bool(row['pass'])}"
        )
        cv2.putText(
            sheet,
            label,
            (column * card_width + 7, row_index * card_height + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 90, 0) if selected else (40, 40, 180),
            2,
            cv2.LINE_AA,
        )
    if not rows:
        cv2.putText(
            sheet,
            "No automatic three-structure corridor candidate",
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
        output / "diagnostic_waveshare_object_rich_corridor_audit.json"
    )
    sheet_path = (
        output
        / "diagnostic_waveshare_object_rich_corridor_contact_sheet.jpg"
    )
    if audit_path.exists() or sheet_path.exists():
        raise RuntimeError(
            "Object-rich corridor outputs already exist; audit will "
            "not be rerun or tuned"
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
    tracks_audit = json.loads(
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
    selected_panels = {
        int(item["frame_id"]): int(item["panel_index"])
        for item in report["render"]["selected_panel_sources"]
    }
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    maps = _undistortion_maps(session.calibration)
    ocr_by_frame = {
        int(item["frame_id"]): item["target_detections"][0]
        for item in ocr_audit["frame_audits"]
        if item["target_detections"]
    }
    stable_panel_frames = {
        int(frame_id)
        for track in seeded_audit["panel_tracks"]
        for frame_id in track["frame_ids"]
    }
    candidate_frames = sorted(
        stable_panel_frames & set(selected_panels) & set(ocr_by_frame)
    )
    image_cache: dict[int, np.ndarray] = {}
    depth_cache: dict[int, np.ndarray] = {}
    reliable_cache: dict[int, np.ndarray] = {}
    polygons_cache: dict[int, list[np.ndarray]] = {}
    panel_by_frame = {}
    for frame_id in candidate_frames:
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
        image_cache[frame_id] = image
        depth_cache[frame_id] = depth
        reliable_cache[frame_id] = reliable
        polygons_cache[frame_id] = parse_fastsam_polygons(
            labels_path / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        panel_by_frame[frame_id] = panel

    evidence_rows: list[StableObjectTrackEvidence] = []
    observation_by_track_frame: dict[
        tuple[int, int], dict[str, object]
    ] = {}
    for track in tracks_audit["stable_selected_panel_tracks"]:
        track_id = int(track["track_id"])
        observations = []
        for frame_id, bbox, area in zip(
            track["selected_panel_frame_ids"],
            track["selected_panel_bboxes_xywh"],
            track["selected_panel_source_area_pixels"],
            strict=True,
        ):
            frame_id = int(frame_id)
            if frame_id not in panel_by_frame:
                continue
            bbox_tuple = tuple(int(value) for value in bbox)
            polygon = _resolve_polygon(
                polygons_cache[frame_id], bbox_tuple
            )
            if polygon is None:
                continue
            mask = _polygon_mask(
                polygon, depth_cache[frame_id].shape
            )
            reliable = reliable_cache[frame_id]
            depth_coverage = float(
                np.count_nonzero(mask & reliable)
                / max(1, np.count_nonzero(mask))
            )
            lab = cv2.cvtColor(
                image_cache[frame_id], cv2.COLOR_BGR2LAB
            )
            median_l = float(np.median(lab[..., 0][mask]))
            x, y, width, height = bbox_tuple
            clarity = float(
                cv2.Laplacian(
                    cv2.cvtColor(
                        image_cache[frame_id][
                            y : y + height, x : x + width
                        ],
                        cv2.COLOR_BGR2GRAY,
                    ),
                    cv2.CV_64F,
                ).var()
            )
            panel_x, panel_y, panel_width, panel_height = (
                panel_by_frame[frame_id].bbox_xywh
            )
            object_center_x = x + 0.5 * width
            object_center_y = y + 0.5 * height
            adjacent = bool(
                object_center_x
                >= panel_x + 0.45 * panel_width
                and x
                <= panel_x + panel_width + 0.75 * panel_width
                and panel_y - panel_height
                <= object_center_y
                <= panel_y + 2.0 * panel_height
            )
            observation = {
                "track_id": track_id,
                "frame_id": frame_id,
                "bbox_xywh": list(bbox_tuple),
                "polygon": polygon,
                "mask": mask,
                "source_area_pixels": int(area),
                "median_lab_l": median_l,
                "clarity_laplacian_variance": clarity,
                "depth_coverage_ratio": depth_coverage,
                "adjacent_to_panel": adjacent,
            }
            observations.append(observation)
            observation_by_track_frame[(track_id, frame_id)] = (
                observation
            )
        if observations:
            evidence_rows.append(
                StableObjectTrackEvidence(
                    track_id=track_id,
                    observation_count=int(track["observation_count"]),
                    selected_panel_observation_count=int(
                        track["selected_panel_observation_count"]
                    ),
                    common_frame_ids=tuple(
                        int(item["frame_id"]) for item in observations
                    ),
                    median_lab_l=float(
                        np.median(
                            [item["median_lab_l"] for item in observations]
                        )
                    ),
                    clarity_variance=float(
                        np.median(
                            [
                                item["clarity_laplacian_variance"]
                                for item in observations
                            ]
                        )
                    ),
                    minimum_depth_coverage_ratio=float(
                        min(
                            item["depth_coverage_ratio"]
                            for item in observations
                        )
                    ),
                    adjacent_to_panel=bool(
                        all(item["adjacent_to_panel"] for item in observations)
                    ),
                )
            )
    selected_track_ids = select_object_rich_neighbor_tracks(
        evidence_rows
    )
    common_frames = (
        sorted(
            set.intersection(
                *[
                    {
                        frame_id
                        for track_id, frame_id in observation_by_track_frame
                        if track_id == selected_id
                    }
                    for selected_id in selected_track_ids
                ],
                set(panel_by_frame),
            )
        )
        if selected_track_ids
        else []
    )
    layout = _layout_from_audit(inspection["renderer"]["layout"])
    crop = inspection["renderer"]["crop"]
    owner_encoded = cv2.imread(
        str(output / "inspection_owner.png"),
        cv2.IMREAD_UNCHANGED,
    )
    if owner_encoded is None or owner_encoded.dtype != np.uint16:
        raise RuntimeError("Could not decode inspection owner")
    owner = owner_encoded.astype(np.int32) - 1
    corridor_rows = []
    contact_rows = []
    for frame_id in common_frames:
        panel = panel_by_frame[frame_id]
        structure_points = [panel.world_points_mm]
        contours = [panel.contour_xy]
        depth_coverages = [
            float(panel.audit["depth_coverage_ratio"])
        ]
        clarities = [panel.clarity_variance]
        object_rows = []
        for track_id in selected_track_ids:
            observation = observation_by_track_frame[
                (track_id, frame_id)
            ]
            world = sample_mask_world_points(
                mask=observation["mask"],
                depth_mm=depth_cache[frame_id],
                reliable_depth=reliable_cache[frame_id],
                camera_to_world=pose_by_id[frame_id],
                intrinsics=session.calibration,
            )
            structure_points.append(world)
            contours.append(observation["polygon"])
            depth_coverages.append(
                float(observation["depth_coverage_ratio"])
            )
            clarities.append(
                float(observation["clarity_laplacian_variance"])
            )
            object_rows.append(
                {
                    key: value
                    for key, value in observation.items()
                    if key not in {"polygon", "mask"}
                }
            )
        projections = [
            _project_to_specific_panel(
                points,
                layout=layout,
                panel_index=selected_panels[frame_id],
                intrinsics=session.calibration,
            )
            for points in structure_points
        ]
        interval = audit_object_rich_interval(
            projected_x_spans=[
                tuple(item["x_span_pixels"])
                if item["x_span_pixels"] is not None
                else (0.0, 0.0)
                for item in projections
            ],
            projected_in_bounds_ratios=[
                float(item["projected_in_bounds_ratio"])
                for item in projections
            ],
            depth_coverage_ratios=depth_coverages,
            source_width_pixels=session.calibration.width,
        )
        accepted = bool(
            all(item["pass"] for item in projections)
            and interval["pass"]
        )
        score = float(
            min(
                item["projected_in_bounds_ratio"]
                for item in projections
            )
            + 0.01 * math.log1p(min(clarities))
        )
        current_votes = [
            _current_owner_votes(
                points,
                layout=layout,
                intrinsics=session.calibration,
                crop=crop,
                owner_frame_id=owner,
            )
            for points in structure_points
        ]
        row = {
            "frame_id": frame_id,
            "panel_index": selected_panels[frame_id],
            "source_position": source_index_by_id[frame_id],
            "selected_object_track_ids": list(selected_track_ids),
            "object_observations": object_rows,
            "structure_depth_coverage_ratios": depth_coverages,
            "direct_specific_panel_projections": projections,
            "pre_seam_hard_owner_interval": interval,
            "current_post_seam_owner_votes_diagnostic_only": current_votes,
            "minimum_structure_clarity_variance": min(clarities),
            "automatic_selection_score": score,
            "pass": accepted,
        }
        corridor_rows.append(row)
        contact_rows.append(
            {
                "frame_id": frame_id,
                "image": image_cache[frame_id],
                "contours": contours,
                "pass": accepted,
            }
        )
    passing = sorted(
        [row for row in corridor_rows if row["pass"]],
        key=lambda row: (
            float(row["automatic_selection_score"]),
            -int(row["frame_id"]),
        ),
        reverse=True,
    )
    selected = passing[0] if passing else None
    selected_frame_id = (
        None if selected is None else int(selected["frame_id"])
    )
    sheet = _contact_sheet(contact_rows, selected_frame_id)
    _write_jpeg_atomic(sheet_path, sheet)
    audit = {
        "schema": "inspection-waveshare-object-rich-corridor/v1",
        "diagnostic_only": True,
        "formal_delivery_files_modified": False,
        "renderer_modified": False,
        "post_render_overlay_used": False,
        "intended_integration_point": (
            "before_seam_solve_single_panel_hard_owner_interval"
        ),
        "white_box_fastsam_complete_mask_used": False,
        "hard_coded_frame_id_used": False,
        "hard_coded_track_id_used": False,
        "stable_panel_source_frame_ids": candidate_frames,
        "automatic_track_evidence": [
            {
                "track_id": item.track_id,
                "observation_count": item.observation_count,
                "selected_panel_observation_count": (
                    item.selected_panel_observation_count
                ),
                "common_frame_ids": list(item.common_frame_ids),
                "median_lab_l": item.median_lab_l,
                "clarity_variance": item.clarity_variance,
                "minimum_depth_coverage_ratio": (
                    item.minimum_depth_coverage_ratio
                ),
                "adjacent_to_panel": item.adjacent_to_panel,
            }
            for item in evidence_rows
        ],
        "automatically_selected_object_track_ids": list(
            selected_track_ids
        ),
        "common_candidate_source_frame_ids": common_frames,
        "corridor_candidates": corridor_rows,
        "object_rich_corridor_pass": selected is not None,
        "automatically_selected_source": selected,
        "thresholds": {
            "minimum_selected_panel_track_observations": 2,
            "minimum_common_panel_frames": 2,
            "maximum_neighbor_median_lab_l": 150.0,
            "minimum_object_depth_coverage_ratio": 0.85,
            "minimum_structure_projected_in_bounds_ratio": 0.90,
            "minimum_panel_structure_depth_coverage_ratio": 0.85,
            "maximum_inter_structure_gap_pixels": 160.0,
            "maximum_interval_width": "source_image_width",
            "automatic_track_ranking": (
                "observation_count_then_common_frame_count_then_"
                "selected_panel_observations_then_clarity"
            ),
            "automatic_source_ranking": (
                "minimum_projection_coverage_plus_log_minimum_clarity"
            ),
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
                "automatically_selected_object_track_ids": list(
                    selected_track_ids
                ),
                "common_candidate_source_frame_ids": common_frames,
                "corridor_candidate_count": len(corridor_rows),
                "object_rich_corridor_pass": selected is not None,
                "selected_source_frame_id": selected_frame_id,
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
