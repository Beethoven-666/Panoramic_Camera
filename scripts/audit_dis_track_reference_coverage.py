from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_fastsam_track import (
    FastSAMRGBDCandidate,
    build_fastsam_rgbd_candidate,
    parse_fastsam_polygons,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    _read_rgbd,
    _reference_panel_inverse_maps,
    _undistortion_maps,
    estimate_inspection_layout,
)
from panorama_demo.session import load_rgbd_session


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of every observation in existing stable DIS "
            "tracks through the formal virtual-panel inverse maps."
        )
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("formal_output", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("track_audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def choose_best_panel_result(
    results: list[dict[str, object]],
) -> dict[str, object]:
    """Uniform panel choice: mapped support first, then world proximity."""

    if not results:
        raise ValueError("Panel result list is empty")
    return max(
        results,
        key=lambda item: (
            int(item["mapped_pixel_count"]),
            -float(item["anchor_distance_mm"]),
            -int(item["panel_index"]),
        ),
    )


def footprint_coverage_ratio(
    candidate_flat_indices: np.ndarray,
    union_flat_indices: np.ndarray,
) -> float:
    candidate = np.unique(np.asarray(candidate_flat_indices, dtype=np.int64))
    union = np.unique(np.asarray(union_flat_indices, dtype=np.int64))
    if union.size == 0:
        return 0.0
    return float(np.intersect1d(candidate, union, assume_unique=True).size / union.size)


def _candidate_mask(
    candidate: FastSAMRGBDCandidate,
    shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [candidate.polygon_xy], 1)
    return mask


def _canvas_indices(
    mapped_mask: np.ndarray,
    *,
    canvas_x0: int,
    canvas_width: int,
) -> np.ndarray:
    y, local_x = np.nonzero(mapped_mask)
    canvas_x = local_x + int(canvas_x0)
    keep = (canvas_x >= 0) & (canvas_x < canvas_width)
    return np.unique(
        (
            y[keep].astype(np.int64) * np.int64(canvas_width)
            + canvas_x[keep].astype(np.int64)
        ).astype(np.int32)
    )


def _contact_sheet(
    tracks: list[dict[str, object]],
    candidate_by_id: dict[int, FastSAMRGBDCandidate],
    images: dict[int, np.ndarray],
    layout_width: int,
    layout_height: int,
) -> np.ndarray:
    card_width = 320
    card_height = 175
    rows: list[np.ndarray] = []
    for track in tracks:
        card = np.zeros((card_height, card_width * 2, 3), dtype=np.uint8)
        best = track.get("selected_single_source")
        if isinstance(best, dict):
            candidate = candidate_by_id[int(best["candidate_id"])]
            image = images[int(candidate.frame_id)]
            x, y, width, height = candidate.bbox_xywh
            margin = 10
            x0 = max(0, x - margin)
            y0 = max(0, y - margin)
            x1 = min(image.shape[1], x + width + margin)
            y1 = min(image.shape[0], y + height + margin)
            crop = image[y0:y1, x0:x1].copy()
            cv2.polylines(
                crop,
                [candidate.polygon_xy - np.asarray((x0, y0))],
                True,
                (0, 255, 0),
                2,
            )
            scale = min(
                300.0 / max(1, crop.shape[1]),
                135.0 / max(1, crop.shape[0]),
                1.0,
            )
            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            card[30 : 30 + resized.shape[0], : resized.shape[1]] = resized
        footprint = np.zeros((layout_height, layout_width), dtype=np.uint8)
        union = np.asarray(track["union_flat_indices"], dtype=np.int64)
        footprint.reshape(-1)[union] = 100
        if isinstance(best, dict):
            selected = np.asarray(best["flat_indices"], dtype=np.int64)
            footprint.reshape(-1)[selected] = 255
        y, x = np.nonzero(footprint)
        if x.size:
            x0, y0 = int(np.min(x)), int(np.min(y))
            x1, y1 = int(np.max(x)) + 1, int(np.max(y)) + 1
            view = cv2.applyColorMap(
                footprint[y0:y1, x0:x1], cv2.COLORMAP_TURBO
            )
            scale = min(
                300.0 / max(1, view.shape[1]),
                135.0 / max(1, view.shape[0]),
                1.0,
            )
            resized = cv2.resize(
                view,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )
            card[
                30 : 30 + resized.shape[0],
                card_width : card_width + resized.shape[1],
            ] = resized
        label = (
            f"T{track['track_id']} obs={track['observation_count']} "
            f"cover={float(best['cross_view_footprint_coverage_ratio']):.3f}"
            if isinstance(best, dict)
            else f"T{track['track_id']} no source"
        )
        cv2.putText(
            card,
            label,
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rows.append(card)
    return np.vstack(rows) if rows else np.zeros((1, 1, 3), dtype=np.uint8)


def main() -> int:
    args = _arguments()
    started = time.perf_counter()
    session = load_rgbd_session(args.session)
    formal_output = args.formal_output.resolve()
    report = json.loads(
        (formal_output / "report.json").read_text(encoding="utf-8")
    )
    transforms = json.loads(
        (formal_output / "transforms.json").read_text(encoding="utf-8")
    )
    track_payload = json.loads(args.track_audit.read_text(encoding="utf-8"))
    stable_tracks = list(track_payload["stable_selected_panel_tracks"])
    stable_candidate_ids = {
        int(value)
        for track in stable_tracks
        for value in track["candidate_ids"]
    }
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    frames = sorted(session.frames, key=lambda item: int(item.frame_id))
    tracked_frames = [
        frame for frame in frames if int(frame.frame_id) in pose_by_id
    ]
    tracked_poses = [
        pose_by_id[int(frame.frame_id)] for frame in tracked_frames
    ]
    layout = estimate_inspection_layout(
        tracked_frames,
        tracked_poses,
        session.calibration,
        config=config,
    )
    if layout.as_dict() != report["render"]["layout"]:
        raise RuntimeError(
            "Reconstructed virtual-panel layout differs from formal report"
        )
    maps = _undistortion_maps(session.calibration)
    candidates: list[FastSAMRGBDCandidate] = []
    candidate_by_id: dict[int, FastSAMRGBDCandidate] = {}
    images: dict[int, np.ndarray] = {}
    depth_by_frame: dict[int, np.ndarray] = {}
    reliable_by_frame: dict[int, np.ndarray] = {}
    for source_index, frame in enumerate(frames):
        frame_id = int(frame.frame_id)
        image, depth, geometric_valid = _read_rgbd(
            frame, session.calibration, maps
        )
        pose = pose_by_id.get(frame_id)
        if pose is None:
            continue
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        polygons = parse_fastsam_polygons(
            args.labels / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        for polygon in polygons:
            candidate = build_fastsam_rgbd_candidate(
                candidate_id=len(candidates),
                source_index=source_index,
                frame_id=frame_id,
                polygon_xy=polygon,
                image_bgr=image,
                lab_image=lab,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=pose,
                intrinsics=session.calibration,
                reference_depth_mm=float(layout.reference_depth_mm),
            )
            if candidate is None:
                continue
            candidates.append(candidate)
            if candidate.candidate_id in stable_candidate_ids:
                candidate_by_id[candidate.candidate_id] = candidate
                images[frame_id] = image
                depth_by_frame[frame_id] = depth
                reliable_by_frame[frame_id] = reliable
    missing = sorted(stable_candidate_ids - set(candidate_by_id))
    if missing:
        raise RuntimeError(
            f"Could not reproduce stable candidate IDs: {missing[:16]}"
        )

    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    panel_anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels], dtype=np.float64
    )
    candidate_rows: dict[int, dict[str, object]] = {}
    candidates_by_frame: dict[int, list[FastSAMRGBDCandidate]] = {}
    for candidate in candidate_by_id.values():
        candidates_by_frame.setdefault(candidate.frame_id, []).append(candidate)
    for frame_id, frame_candidates in candidates_by_frame.items():
        pose = pose_by_id[frame_id]
        needed_panels: set[int] = set()
        panel_options: dict[int, list[int]] = {}
        for candidate in frame_candidates:
            scan = float(
                np.asarray(candidate.world_centroid_mm) @ scan_axis
            )
            nearest = np.argsort(np.abs(panel_anchors - scan))[:2]
            panel_options[candidate.candidate_id] = [
                int(value) for value in nearest
            ]
            needed_panels.update(panel_options[candidate.candidate_id])
        panel_maps = {
            panel_index: _reference_panel_inverse_maps(
                source_pose=pose,
                panel_index=panel_index,
                layout=layout,
                intrinsics=session.calibration,
            )
            for panel_index in sorted(needed_panels)
        }
        gray = cv2.cvtColor(images[frame_id], cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        for candidate in frame_candidates:
            source_mask = _candidate_mask(
                candidate,
                (
                    session.calibration.height,
                    session.calibration.width,
                ),
            )
            panel_results = []
            scan = float(
                np.asarray(candidate.world_centroid_mm) @ scan_axis
            )
            for panel_index in panel_options[candidate.candidate_id]:
                x0, map_x, map_y, valid, _ = panel_maps[panel_index]
                mapped = cv2.remap(
                    source_mask,
                    map_x,
                    map_y,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(bool)
                mapped &= valid
                flat = _canvas_indices(
                    mapped,
                    canvas_x0=x0,
                    canvas_width=layout.width,
                )
                panel_results.append(
                    {
                        "panel_index": panel_index,
                        "anchor_distance_mm": abs(
                            float(panel_anchors[panel_index]) - scan
                        ),
                        "mapped_pixel_count": int(flat.size),
                        "flat_indices": flat,
                    }
                )
            selected = choose_best_panel_result(panel_results)
            x, y, width, height = candidate.bbox_xywh
            boundary_complete = bool(
                x >= 8
                and y >= 8
                and x + width <= session.calibration.width - 8
                and y + height <= session.calibration.height - 8
            )
            blur = float(laplacian[source_mask.astype(bool)].var())
            candidate_rows[candidate.candidate_id] = {
                "candidate_id": int(candidate.candidate_id),
                "frame_id": int(frame_id),
                "selected_panel_index": int(selected["panel_index"]),
                "panel_anchor_distance_mm": float(
                    selected["anchor_distance_mm"]
                ),
                "mapped_pixel_count": int(selected["mapped_pixel_count"]),
                "flat_indices": np.asarray(
                    selected["flat_indices"], dtype=np.int32
                ),
                "source_boundary_complete": boundary_complete,
                "source_laplacian_variance": blur,
                "source_clear": bool(blur >= 50.0),
                "source_depth_coverage_ratio": float(
                    candidate.depth_coverage_ratio
                ),
                "forbidden_transform_used": False,
            }

    track_rows: list[dict[str, object]] = []
    for track in stable_tracks:
        candidate_ids = [int(value) for value in track["candidate_ids"]]
        observation_rows = [candidate_rows[value] for value in candidate_ids]
        union = np.unique(
            np.concatenate(
                [
                    np.asarray(item["flat_indices"], dtype=np.int32)
                    for item in observation_rows
                ]
            )
        )
        source_audits = []
        for item in observation_rows:
            coverage_ratio = footprint_coverage_ratio(
                np.asarray(item["flat_indices"]),
                union,
            )
            source_audits.append(
                {
                    **{
                        key: value
                        for key, value in item.items()
                        if key != "flat_indices"
                    },
                    "cross_view_footprint_coverage_ratio": coverage_ratio,
                    "covers_all_cross_view_footprints": bool(
                        coverage_ratio >= 0.98
                    ),
                    "complete_single_source": bool(
                        item["source_boundary_complete"]
                        and item["source_clear"]
                        and float(item["source_depth_coverage_ratio"]) >= 0.85
                        and coverage_ratio >= 0.98
                    ),
                }
            )
        selected = max(
            source_audits,
            key=lambda item: (
                float(item["cross_view_footprint_coverage_ratio"]),
                bool(item["source_boundary_complete"]),
                float(item["source_laplacian_variance"]),
                float(item["source_depth_coverage_ratio"]),
                -int(item["frame_id"]),
            ),
        )
        selected_internal = candidate_rows[int(selected["candidate_id"])]
        track_rows.append(
            {
                "track_id": int(track["track_id"]),
                "observation_count": len(candidate_ids),
                "all_short_baseline_frame_ids": [
                    int(value) for value in track["frame_ids"]
                ],
                "union_footprint_pixel_count": int(union.size),
                "candidate_source_count": len(source_audits),
                "complete_single_source_exists": any(
                    bool(item["complete_single_source"])
                    for item in source_audits
                ),
                "selected_single_source": {
                    **selected,
                    "flat_indices": selected_internal[
                        "flat_indices"
                    ].tolist(),
                },
                "source_audits": source_audits,
                "union_flat_indices": union.tolist(),
            }
        )
    highlighted = {
        str(track_id): next(
            item for item in track_rows if item["track_id"] == track_id
        )
        for track_id in (0, 49, 112, 479)
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sheet = _contact_sheet(
        track_rows,
        candidate_by_id,
        images,
        layout.width,
        layout.height,
    )
    cv2.imwrite(str(output / "dis_track_reference_coverage_contact_sheet.jpg"), sheet)
    serializable_tracks = []
    for track in track_rows:
        serializable_tracks.append(
            {
                key: value
                for key, value in track.items()
                if key not in {"union_flat_indices"}
            }
        )
        serializable_tracks[-1]["selected_single_source"].pop(
            "flat_indices", None
        )
    serializable_highlighted = {
        key: next(
            item
            for item in serializable_tracks
            if item["track_id"] == int(key)
        )
        for key in highlighted
    }
    audit = {
        "schema": "inspection-dis-track-reference-coverage/v1",
        "formal_renderer_modified": False,
        "selection_policy": (
            "all_short_baseline_observations_nearest_two_world_panels_"
            "maximum_existing_reference_inverse_map_support_then_"
            "single_source_maximum_union_footprint_coverage"
        ),
        "manual_track_roi_or_frame_selection_used": False,
        "translation_used": False,
        "affine_used": False,
        "additional_warp_used": False,
        "hole_fill_used": False,
        "generated_rgb_used": False,
        "track_count": len(serializable_tracks),
        "observation_count": sum(
            int(item["observation_count"]) for item in serializable_tracks
        ),
        "track_with_complete_single_source_count": sum(
            bool(item["complete_single_source_exists"])
            for item in serializable_tracks
        ),
        "highlighted_tracks": serializable_highlighted,
        "tracks": serializable_tracks,
        "thresholds": {
            "source_boundary_margin_pixels": 8,
            "source_clear_laplacian_variance": 50.0,
            "minimum_source_depth_coverage_ratio": 0.85,
            "minimum_cross_view_footprint_coverage_ratio": 0.98,
        },
        "layout": layout.as_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "contact_sheet": "dis_track_reference_coverage_contact_sheet.jpg"
        },
    }
    (output / "dis_track_reference_coverage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "track_count": audit["track_count"],
                "observation_count": audit["observation_count"],
                "track_with_complete_single_source_count": audit[
                    "track_with_complete_single_source_count"
                ],
                "highlighted": {
                    key: {
                        "frame_id": value["selected_single_source"][
                            "frame_id"
                        ],
                        "coverage": value["selected_single_source"][
                            "cross_view_footprint_coverage_ratio"
                        ],
                        "complete": value[
                            "complete_single_source_exists"
                        ],
                    }
                    for key, value in serializable_highlighted.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
