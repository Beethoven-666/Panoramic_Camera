"""Short-baseline FastSAM identity tracks using DIS only as evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    build_fastsam_rgbd_candidate,
    flow_forward_backward_consistency,
    flow_predict_mask,
    parse_fastsam_polygons,
    select_unambiguous_one_to_one_matches,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    _read_rgbd,
    _undistortion_maps,
)
from panorama_demo.session import load_rgbd_session


def _preview_mask(
    candidate: FastSAMRGBDCandidate,
    shape: tuple[int, int],
    scale: float,
) -> np.ndarray:
    polygon = np.rint(candidate.polygon_xy * scale).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, shape[1] - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, shape[0] - 1)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return float(intersection / union) if union else 0.0


def _candidate_pair_audit(
    first: FastSAMRGBDCandidate,
    second: FastSAMRGBDCandidate,
    predicted_mask: np.ndarray,
    second_mask: np.ndarray,
) -> tuple[bool, float, dict[str, object]]:
    iou = _mask_iou(predicted_mask, second_mask)
    lab_delta = float(
        np.linalg.norm(
            np.asarray(first.median_lab) - np.asarray(second.median_lab)
        )
    )
    centroid_delta = float(
        np.linalg.norm(
            np.asarray(first.world_centroid_mm)
            - np.asarray(second.world_centroid_mm)
        )
    )
    area_ratio = max(
        first.source_area_pixels, second.source_area_pixels
    ) / max(1, min(first.source_area_pixels, second.source_area_pixels))
    aspect_delta = abs(
        math.log(
            max(1e-6, first.aspect_ratio)
            / max(1e-6, second.aspect_ratio)
        )
    )
    contour_delta = float(
        cv2.matchShapes(
            first.polygon_xy,
            second.polygon_xy,
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
    )
    accepted = bool(
        iou >= 0.35
        and lab_delta <= 30.0
        and centroid_delta <= 80.0
        and area_ratio <= 2.0
        and aspect_delta <= 0.65
        and abs(first.solidity - second.solidity) <= 0.30
        and contour_delta <= 0.35
    )
    score = (
        iou
        - 0.01 * lab_delta
        - 0.0025 * centroid_delta
        - 0.20 * contour_delta
    )
    return accepted, score, {
        "flow_predicted_mask_iou": iou,
        "median_lab_delta": lab_delta,
        "world_centroid_delta_mm": centroid_delta,
        "source_area_ratio": area_ratio,
        "log_aspect_delta": aspect_delta,
        "solidity_delta": abs(first.solidity - second.solidity),
        "contour_match_i1": contour_delta,
        "score": score,
        "pass": accepted,
    }


def _contact_sheet(
    tracks: list[dict[str, object]],
    candidate_by_id: dict[int, FastSAMRGBDCandidate],
    selected_images: dict[int, np.ndarray],
) -> np.ndarray:
    card_width, card_height = 220, 155
    columns = 4
    selected_tracks = tracks[:30]
    sheet = np.full(
        (max(1, len(selected_tracks)) * card_height, columns * card_width, 3),
        245,
        dtype=np.uint8,
    )
    for row, track in enumerate(selected_tracks):
        observations = [
            candidate_by_id[value]
            for value in track["selected_panel_candidate_ids"][:columns]
        ]
        for column, candidate in enumerate(observations):
            image = selected_images[candidate.frame_id]
            x, y, width, height = candidate.bbox_xywh
            margin = 12
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1 = min(image.shape[1], x + width + margin)
            y1 = min(image.shape[0], y + height + margin)
            crop = image[y0:y1, x0:x1].copy()
            polygon = candidate.polygon_xy - np.asarray([x0, y0])
            cv2.polylines(crop, [polygon], True, (0, 255, 0), 2)
            scale = min(
                (card_width - 8) / max(1, crop.shape[1]),
                (card_height - 28) / max(1, crop.shape[0]),
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
            offset_y = 24 + (card_height - 24 - resized.shape[0]) // 2
            card[
                offset_y : offset_y + resized.shape[0],
                offset_x : offset_x + resized.shape[1],
            ] = resized
            cv2.putText(
                card,
                f"T{track['track_id']} F{candidate.frame_id}",
                (5, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                1,
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
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    session = load_rgbd_session(session_path)
    frames = sorted(session.frames, key=lambda item: int(item.frame_id))
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    selected_frame_ids = {
        int(item["frame_id"])
        for item in report["render"]["selected_panel_sources"]
    }
    maps = _undistortion_maps(session.calibration)
    preview_scale = 0.25
    preview_size = (
        int(round(session.calibration.width * preview_scale)),
        int(round(session.calibration.height * preview_scale)),
    )
    grays: list[np.ndarray] = []
    candidates_by_frame: list[list[FastSAMRGBDCandidate]] = []
    candidates: list[FastSAMRGBDCandidate] = []
    selected_images: dict[int, np.ndarray] = {}
    raw_polygon_count = 0
    missing_pose_polygon_count = 0
    for frame_index, frame in enumerate(frames):
        frame_id = int(frame.frame_id)
        image, depth, geometric_valid = _read_rgbd(
            frame, session.calibration, maps
        )
        gray = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            preview_size,
            interpolation=cv2.INTER_AREA,
        )
        grays.append(np.ascontiguousarray(gray))
        if frame_id in selected_frame_ids:
            selected_images[frame_id] = np.ascontiguousarray(image)
        polygons = parse_fastsam_polygons(
            labels_path / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        raw_polygon_count += len(polygons)
        frame_candidates: list[FastSAMRGBDCandidate] = []
        pose = pose_by_id.get(frame_id)
        if pose is None:
            missing_pose_polygon_count += len(polygons)
            candidates_by_frame.append(frame_candidates)
            continue
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        for polygon in polygons:
            candidate = build_fastsam_rgbd_candidate(
                candidate_id=len(candidates),
                source_index=frame_index,
                frame_id=frame_id,
                polygon_xy=polygon,
                image_bgr=image,
                lab_image=lab,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=pose,
                intrinsics=session.calibration,
                reference_depth_mm=float(
                    report["render"]["layout"]["reference_depth_mm"]
                ),
            )
            if candidate is not None:
                candidates.append(candidate)
                frame_candidates.append(candidate)
        candidates_by_frame.append(frame_candidates)
        if (frame_index + 1) % 32 == 0:
            print(
                f"described {frame_index + 1}/{len(frames)} frames",
                flush=True,
            )

    candidate_by_id = {item.candidate_id: item for item in candidates}
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    edge_audit: dict[str, dict[str, object]] = {}
    pair_audits: list[dict[str, object]] = []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    for pair_index in range(len(frames) - 1):
        first_candidates = candidates_by_frame[pair_index]
        second_candidates = candidates_by_frame[pair_index + 1]
        pair_row = {
            "first_frame_id": int(frames[pair_index].frame_id),
            "second_frame_id": int(frames[pair_index + 1].frame_id),
            "first_candidate_count": len(first_candidates),
            "second_candidate_count": len(second_candidates),
            "pose_gate_available": bool(
                int(frames[pair_index].frame_id) in pose_by_id
                and int(frames[pair_index + 1].frame_id) in pose_by_id
            ),
            "valid_edge_count": 0,
            "one_to_one_match_count": 0,
            "fb_rejected_source_count": 0,
        }
        if not first_candidates or not second_candidates:
            pair_audits.append(pair_row)
            continue
        forward = dis.calc(
            grays[pair_index], grays[pair_index + 1], None
        )
        backward = dis.calc(
            grays[pair_index + 1], grays[pair_index], None
        )
        first_masks = [
            _preview_mask(item, grays[pair_index].shape, preview_scale)
            for item in first_candidates
        ]
        second_masks = [
            _preview_mask(item, grays[pair_index + 1].shape, preview_scale)
            for item in second_candidates
        ]
        valid = np.zeros(
            (len(first_candidates), len(second_candidates)), dtype=bool
        )
        score = np.full(valid.shape, -np.inf, dtype=np.float64)
        metrics: dict[tuple[int, int], dict[str, object]] = {}
        for first_index, (first_candidate, first_mask) in enumerate(
            zip(first_candidates, first_masks, strict=True)
        ):
            fb = flow_forward_backward_consistency(
                first_mask,
                forward,
                backward,
                maximum_error_pixels=0.75,
            )
            if not fb["pass"]:
                pair_row["fb_rejected_source_count"] += 1
                continue
            predicted = flow_predict_mask(first_mask, backward)
            for second_index, (second_candidate, second_mask) in enumerate(
                zip(second_candidates, second_masks, strict=True)
            ):
                accepted, value, audit = _candidate_pair_audit(
                    first_candidate,
                    second_candidate,
                    predicted,
                    second_mask,
                )
                if not accepted:
                    continue
                valid[first_index, second_index] = True
                score[first_index, second_index] = value
                metrics[(first_index, second_index)] = {
                    **audit,
                    "flow_forward_backward": fb,
                }
        pair_row["valid_edge_count"] = int(np.count_nonzero(valid))
        matches = select_unambiguous_one_to_one_matches(
            valid, score, ambiguity_margin=0.05
        )
        pair_row["one_to_one_match_count"] = len(matches)
        for first_index, second_index in matches:
            first_id = first_candidates[first_index].candidate_id
            second_id = second_candidates[second_index].candidate_id
            outgoing[first_id] = second_id
            incoming[second_id] = first_id
            edge_audit[f"{first_id}:{second_id}"] = metrics[
                (first_index, second_index)
            ]
        pair_audits.append(pair_row)
        if (pair_index + 1) % 32 == 0:
            print(
                f"flow-audited {pair_index + 1}/{len(frames) - 1} pairs",
                flush=True,
            )

    tracks: list[dict[str, object]] = []
    visited: set[int] = set()
    for candidate in candidates:
        candidate_id = candidate.candidate_id
        if candidate_id in incoming or candidate_id in visited:
            continue
        sequence = []
        edge_rows = []
        current = candidate_id
        while current not in visited:
            visited.add(current)
            sequence.append(current)
            next_id = outgoing.get(current)
            if next_id is None:
                break
            edge_rows.append(edge_audit[f"{current}:{next_id}"])
            current = next_id
        if len(sequence) < 2:
            continue
        selected_ids = [
            value
            for value in sequence
            if candidate_by_id[value].frame_id in selected_frame_ids
        ]
        tracks.append(
            {
                "track_id": len(tracks),
                "candidate_ids": sequence,
                "frame_ids": [
                    int(candidate_by_id[value].frame_id)
                    for value in sequence
                ],
                "observation_count": len(sequence),
                "selected_panel_candidate_ids": selected_ids,
                "selected_panel_frame_ids": [
                    int(candidate_by_id[value].frame_id)
                    for value in selected_ids
                ],
                "selected_panel_bboxes_xywh": [
                    list(candidate_by_id[value].bbox_xywh)
                    for value in selected_ids
                ],
                "selected_panel_source_area_pixels": [
                    int(candidate_by_id[value].source_area_pixels)
                    for value in selected_ids
                ],
                "selected_panel_observation_count": len(selected_ids),
                "maximum_area_ratio": max(
                    float(row["source_area_ratio"]) for row in edge_rows
                ),
                "minimum_flow_mask_iou": min(
                    float(row["flow_predicted_mask_iou"])
                    for row in edge_rows
                ),
                "maximum_fb_p95_preview_pixels": max(
                    float(
                        row["flow_forward_backward"][
                            "p95_error_pixels"
                        ]
                    )
                    for row in edge_rows
                ),
                "merge_split_terminated": bool(
                    sequence[-1] not in outgoing
                    and candidate_by_id[sequence[-1]].source_index
                    < len(frames) - 1
                ),
            }
        )
    stable_selected = sorted(
        [
            item
            for item in tracks
            if item["selected_panel_observation_count"] >= 2
        ],
        key=lambda item: (
            item["selected_panel_observation_count"],
            item["observation_count"],
            -item["maximum_area_ratio"],
        ),
        reverse=True,
    )
    sheet = _contact_sheet(
        stable_selected, candidate_by_id, selected_images
    )
    sheet_path = output / "diagnostic_fastsam_dis_tracks_contact_sheet.jpg"
    audit_path = output / "diagnostic_fastsam_dis_tracks_audit.json"
    if not cv2.imwrite(str(sheet_path), sheet):
        raise RuntimeError("Could not write DIS track contact sheet")
    audit = {
        "schema": "inspection-fastsam-dis-identity-tracks/v1",
        "formal_output_modified": False,
        "model_role": "polygon_contour_candidates_only",
        "flow_role": "candidate_identity_evidence_only",
        "flow_used_to_warp_final_rgb_or_position": False,
        "frame_count": len(frames),
        "label_frame_count": len(
            list(labels_path.glob("*.txt"))
        ),
        "real_pose_frame_count": len(pose_by_id),
        "pose_unavailable_frame_ids": sorted(
            set(int(frame.frame_id) for frame in frames) - set(pose_by_id)
        ),
        "raw_polygon_count": raw_polygon_count,
        "pose_unavailable_polygon_count": missing_pose_polygon_count,
        "rgbd_candidate_count": len(candidates),
        "matched_edge_count": len(outgoing),
        "track_count": len(tracks),
        "stable_selected_panel_track_count": len(stable_selected),
        "pair_audits": pair_audits,
        "stable_selected_panel_tracks": stable_selected,
        "thresholds": {
            "preview_scale": preview_scale,
            "maximum_fb_error_preview_pixels": 0.75,
            "minimum_fb_consistent_ratio": 0.80,
            "minimum_flow_predicted_mask_iou": 0.35,
            "maximum_lab_delta": 30.0,
            "maximum_world_centroid_delta_mm": 80.0,
            "maximum_area_ratio": 2.0,
            "maximum_contour_match_i1": 0.35,
            "mutual_best_ambiguity_margin": 0.05,
            "minimum_selected_panel_observations": 2,
        },
        "manual_roi_used": False,
        "manual_frame_id_used": False,
        "translation_used": False,
        "affine_used": False,
        "final_rgb_warp_used": False,
        "hole_fill_used": False,
        "generated_rgb_used": False,
        "elapsed_seconds": time.perf_counter() - started,
        "files": {"contact_sheet": sheet_path.name},
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_path)
    print(sheet_path)
    print(
        json.dumps(
            {
                "raw_polygon_count": raw_polygon_count,
                "rgbd_candidate_count": len(candidates),
                "matched_edge_count": len(outgoing),
                "track_count": len(tracks),
                "stable_selected_panel_track_count": len(stable_selected),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
