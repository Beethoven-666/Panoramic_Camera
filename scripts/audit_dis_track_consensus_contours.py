from __future__ import annotations

import argparse
from collections import Counter
import json
import math
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
from panorama_demo.inspection_object_handoff import build_object_owner_interval
from panorama_demo.session import load_rgbd_session


VOXEL_SIZE_MM = 20.0
MINIMUM_CONSENSUS_VIEWS = 2
MINIMUM_CONSENSUS_VIEW_FRACTION = 0.30
LAB_DISTANCE = 45.0
PROBABLE_DILATION_PIXELS = 5
MINIMUM_NATURAL_COVERAGE = 0.90


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("formal_output", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("track_audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def consensus_voxels(
    per_observation: list[set[tuple[int, int, int]]],
    *,
    minimum_views: int = MINIMUM_CONSENSUS_VIEWS,
) -> set[tuple[int, int, int]]:
    counts: Counter[tuple[int, int, int]] = Counter()
    for observation in per_observation:
        counts.update(set(observation))
    return {
        key for key, count in counts.items() if count >= int(minimum_views)
    }


def seeded_components(probable: np.ndarray, seed: np.ndarray) -> np.ndarray:
    candidate = np.asarray(probable, dtype=bool)
    seeds = np.asarray(seed, dtype=bool)
    count, labels, _, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), 8
    )
    accepted = np.zeros(candidate.shape, dtype=bool)
    for label in range(1, count):
        component = labels == label
        if np.any(component & seeds):
            accepted |= component
    return accepted


def _mask(candidate: FastSAMRGBDCandidate, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(result, [candidate.polygon_xy], 1)
    return result.astype(bool)


def _world_voxels(
    candidate: FastSAMRGBDCandidate,
    depth: np.ndarray,
    reliable: np.ndarray,
    pose: np.ndarray,
    intrinsics: object,
) -> set[tuple[int, int, int]]:
    mask = _mask(candidate, depth.shape)
    sampled = np.zeros(mask.shape, dtype=bool)
    sampled[::4, ::4] = True
    y, x = np.nonzero(mask & reliable & sampled)
    z = depth[y, x].astype(np.float64)
    camera = np.column_stack(
        (
            (x - intrinsics.cx) * z / intrinsics.fx,
            (y - intrinsics.cy) * z / intrinsics.fy,
            z,
        )
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    return {
        tuple(int(value) for value in key)
        for key in np.floor(world / VOXEL_SIZE_MM).astype(np.int32)
    }


def _dilated_voxels(
    values: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    return {
        (key[0] + dx, key[1] + dy, key[2] + dz)
        for key in values
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    }


def _reconstruct(
    candidate: FastSAMRGBDCandidate,
    *,
    image: np.ndarray,
    depth: np.ndarray,
    reliable: np.ndarray,
    pose: np.ndarray,
    intrinsics: object,
    consensus: set[tuple[int, int, int]],
) -> tuple[np.ndarray | None, dict[str, object]]:
    x0, y0, width, height = candidate.bbox_xywh
    original = _mask(candidate, depth.shape)[
        y0 : y0 + height, x0 : x0 + width
    ]
    yy, xx = np.indices((height, width), dtype=np.int32)
    source_x = xx + x0
    source_y = yy + y0
    z = depth[source_y, source_x].astype(np.float64)
    valid = reliable[source_y, source_x]
    camera = np.stack(
        (
            (source_x - intrinsics.cx) * z / intrinsics.fx,
            (source_y - intrinsics.cy) * z / intrinsics.fy,
            z,
        ),
        axis=-1,
    )
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    keys = np.floor(world / VOXEL_SIZE_MM).astype(np.int32)
    dilated = _dilated_voxels(consensus)
    flat_keys = keys.reshape(-1, 3)
    support = np.asarray(
        [
            tuple(int(value) for value in key) in dilated
            for key in flat_keys
        ],
        dtype=bool,
    ).reshape(height, width)
    seed = original & valid & support
    if np.count_nonzero(seed) < 24:
        return None, {
            "reason": "insufficient_projected_consensus_seed",
            "seed_pixel_count": int(np.count_nonzero(seed)),
        }
    lab = cv2.cvtColor(
        image[y0 : y0 + height, x0 : x0 + width],
        cv2.COLOR_BGR2LAB,
    ).astype(np.float32)
    median_lab = np.median(lab[seed], axis=0)
    lab_distance = np.linalg.norm(lab - median_lab, axis=2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (PROBABLE_DILATION_PIXELS, PROBABLE_DILATION_PIXELS),
    )
    near_seed = cv2.dilate(seed.astype(np.uint8), kernel) > 0
    probable = original & near_seed & (lab_distance <= LAB_DISTANCE)
    probable |= seed
    grab = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grab[probable] = cv2.GC_PR_FGD
    grab[seed] = cv2.GC_FGD
    try:
        cv2.grabCut(
            image[y0 : y0 + height, x0 : x0 + width],
            grab,
            None,
            np.zeros((1, 65), dtype=np.float64),
            np.zeros((1, 65), dtype=np.float64),
            2,
            cv2.GC_INIT_WITH_MASK,
        )
        reconstructed = (grab == cv2.GC_FGD) | (grab == cv2.GC_PR_FGD)
    except cv2.error:
        reconstructed = probable
    reconstructed = seeded_components(reconstructed, seed)
    if np.count_nonzero(reconstructed) < 300:
        return None, {
            "reason": "consensus_contour_below_minimum_area",
            "seed_pixel_count": int(np.count_nonzero(seed)),
            "reconstructed_pixel_count": int(
                np.count_nonzero(reconstructed)
            ),
        }
    full = np.zeros(depth.shape, dtype=bool)
    full[y0 : y0 + height, x0 : x0 + width] = reconstructed
    return full, {
        "reason": "accepted_consensus_contour",
        "seed_pixel_count": int(np.count_nonzero(seed)),
        "original_pixel_count": int(np.count_nonzero(original)),
        "reconstructed_pixel_count": int(np.count_nonzero(reconstructed)),
        "reconstructed_to_original_area_ratio": float(
            np.count_nonzero(reconstructed)
            / max(1, np.count_nonzero(original))
        ),
        "bbox_limited": True,
        "lab_distance_limit": LAB_DISTANCE,
        "probable_dilation_pixels": PROBABLE_DILATION_PIXELS,
    }


def _contour_candidate(
    source: FastSAMRGBDCandidate,
    reconstructed: np.ndarray,
    *,
    image: np.ndarray,
    depth: np.ndarray,
    reliable: np.ndarray,
    pose: np.ndarray,
    intrinsics: object,
    reference_depth_mm: float,
) -> FastSAMRGBDCandidate | None:
    contours, _ = cv2.findContours(
        reconstructed.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return build_fastsam_rgbd_candidate(
        candidate_id=source.candidate_id,
        source_index=source.source_index,
        frame_id=source.frame_id,
        polygon_xy=contour.reshape(-1, 2),
        image_bgr=image,
        lab_image=lab,
        depth_mm=depth,
        reliable_depth=reliable,
        camera_to_world=pose,
        intrinsics=intrinsics,
        reference_depth_mm=reference_depth_mm,
    )


def _natural(
    mask: np.ndarray,
    *,
    pose: np.ndarray,
    panel_index: int,
    layout: object,
    intrinsics: object,
) -> dict[str, object] | None:
    x0, map_x, map_y, valid, _ = _reference_panel_inverse_maps(
        source_pose=pose,
        panel_index=panel_index,
        layout=layout,
        intrinsics=intrinsics,
    )
    sampled = cv2.remap(
        mask.astype(np.uint8),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    target = (sampled > 0) & valid
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        target.astype(np.uint8), 8
    )
    components = [
        (int(stats[label, cv2.CC_STAT_AREA]), label)
        for label in range(1, count)
    ]
    if len(components) != 1:
        return None
    target = labels == components[0][1]
    y, x = np.nonzero(target)
    source_x = np.rint(map_x[y, x]).astype(np.int32)
    source_y = np.rint(map_y[y, x]).astype(np.int32)
    inside = (
        (source_x >= 0)
        & (source_x < mask.shape[1])
        & (source_y >= 0)
        & (source_y < mask.shape[0])
    )
    hit = np.zeros(mask.shape, dtype=bool)
    hit[source_y[inside], source_x[inside]] = True
    coverage = float(np.count_nonzero(hit & mask) / max(1, np.count_nonzero(mask)))
    full_mask = np.zeros((layout.height, layout.width), dtype=bool)
    full_valid = np.zeros_like(full_mask)
    x1 = x0 + target.shape[1]
    full_mask[:, x0:x1] = target
    full_valid[:, x0:x1] = valid
    return {
        "panel_index": int(panel_index),
        "corner_x": int(x0),
        "source_coverage_ratio": coverage,
        "full_mask": full_mask,
        "full_valid": full_valid,
    }


def main() -> int:
    args = _arguments()
    started = time.perf_counter()
    session = load_rgbd_session(args.session)
    output_root = args.formal_output.resolve()
    report = json.loads(
        (output_root / "report.json").read_text(encoding="utf-8")
    )
    transforms = json.loads(
        (output_root / "transforms.json").read_text(encoding="utf-8")
    )
    track_payload = json.loads(args.track_audit.read_text(encoding="utf-8"))
    tracks = list(track_payload["stable_selected_panel_tracks"])
    wanted_ids = {
        int(value) for track in tracks for value in track["candidate_ids"]
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
    frames = sorted(session.frames, key=lambda item: int(item.frame_id))
    tracked_frames = [item for item in frames if int(item.frame_id) in pose_by_id]
    tracked_poses = [pose_by_id[int(item.frame_id)] for item in tracked_frames]
    layout = estimate_inspection_layout(
        tracked_frames,
        tracked_poses,
        session.calibration,
        config=config,
    )
    maps = _undistortion_maps(session.calibration)
    candidates: list[FastSAMRGBDCandidate] = []
    candidate_by_id: dict[int, FastSAMRGBDCandidate] = {}
    image_by_frame: dict[int, np.ndarray] = {}
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
            item = build_fastsam_rgbd_candidate(
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
            if item is None:
                continue
            candidates.append(item)
            if item.candidate_id in wanted_ids:
                candidate_by_id[item.candidate_id] = item
                image_by_frame[frame_id] = image
                depth_by_frame[frame_id] = depth
                reliable_by_frame[frame_id] = reliable
    if set(candidate_by_id) != wanted_ids:
        raise RuntimeError("Stable DIS candidate IDs could not be reproduced")

    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    anchors = np.asarray(
        [item.anchor_scan_mm for item in layout.panels], dtype=np.float64
    )
    full_owner = np.full((layout.height, layout.width), -1, dtype=np.int32)
    crop = report["render"]["crop"]
    encoded_owner = cv2.imread(
        str(output_root / "inspection_owner.png"), cv2.IMREAD_UNCHANGED
    ).astype(np.int32) - 1
    crop_x, crop_y, crop_width, crop_height = (
        int(crop[key]) for key in ("x", "y", "width", "height")
    )
    full_owner[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ] = encoded_owner

    audit_tracks = []
    contact_rows = []
    for track in tracks:
        track_id = int(track["track_id"])
        observations = [
            candidate_by_id[int(value)] for value in track["candidate_ids"]
        ]
        voxel_sets = [
            _world_voxels(
                item,
                depth_by_frame[item.frame_id],
                reliable_by_frame[item.frame_id],
                pose_by_id[item.frame_id],
                session.calibration,
            )
            for item in observations
        ]
        required_consensus_views = max(
            MINIMUM_CONSENSUS_VIEWS,
            int(
                math.ceil(
                    MINIMUM_CONSENSUS_VIEW_FRACTION
                    * len(voxel_sets)
                )
            ),
        )
        consensus = consensus_voxels(
            voxel_sets,
            minimum_views=required_consensus_views,
        )
        reconstructed_rows = []
        footprints = []
        for item in observations:
            reconstructed, reconstruction_audit = _reconstruct(
                item,
                image=image_by_frame[item.frame_id],
                depth=depth_by_frame[item.frame_id],
                reliable=reliable_by_frame[item.frame_id],
                pose=pose_by_id[item.frame_id],
                intrinsics=session.calibration,
                consensus=consensus,
            )
            if reconstructed is None:
                reconstructed_rows.append(
                    {
                        "candidate_id": int(item.candidate_id),
                        "frame_id": int(item.frame_id),
                        "accepted": False,
                        "reconstruction": reconstruction_audit,
                    }
                )
                continue
            rebuilt = _contour_candidate(
                item,
                reconstructed,
                image=image_by_frame[item.frame_id],
                depth=depth_by_frame[item.frame_id],
                reliable=reliable_by_frame[item.frame_id],
                pose=pose_by_id[item.frame_id],
                intrinsics=session.calibration,
                reference_depth_mm=float(layout.reference_depth_mm),
            )
            if rebuilt is None:
                reconstructed_rows.append(
                    {
                        "candidate_id": int(item.candidate_id),
                        "frame_id": int(item.frame_id),
                        "accepted": False,
                        "reconstruction": {
                            **reconstruction_audit,
                            "reason": "rebuilt_rgbd_candidate_failed",
                        },
                    }
                )
                continue
            scan = float(np.asarray(rebuilt.world_centroid_mm) @ scan_axis)
            panel_results = []
            for panel_index in np.argsort(np.abs(anchors - scan))[:2]:
                natural = _natural(
                    reconstructed,
                    pose=pose_by_id[item.frame_id],
                    panel_index=int(panel_index),
                    layout=layout,
                    intrinsics=session.calibration,
                )
                if natural is not None:
                    panel_results.append(natural)
            if not panel_results:
                reconstructed_rows.append(
                    {
                        "candidate_id": int(item.candidate_id),
                        "frame_id": int(item.frame_id),
                        "accepted": False,
                        "reconstruction": {
                            **reconstruction_audit,
                            "reason": "no_connected_reference_inverse_map",
                        },
                    }
                )
                continue
            selected = max(
                panel_results,
                key=lambda value: (
                    float(value["source_coverage_ratio"]),
                    -int(value["panel_index"]),
                ),
            )
            footprints.append(selected["full_mask"])
            reconstructed_rows.append(
                {
                    "candidate_id": int(item.candidate_id),
                    "frame_id": int(item.frame_id),
                    "accepted": True,
                    "selected_panel_index": int(selected["panel_index"]),
                    "source_coverage_ratio": float(
                        selected["source_coverage_ratio"]
                    ),
                    "reconstruction": reconstruction_audit,
                    "_natural": selected,
                    "_mask": reconstructed,
                }
            )
        accepted_rows = [item for item in reconstructed_rows if item["accepted"]]
        options = []
        if len(footprints) >= 2:
            for row in accepted_rows:
                natural = row["_natural"]
                try:
                    interval = build_object_owner_interval(
                        panel_index=int(natural["panel_index"]),
                        view_dependent_footprints=tuple(footprints),
                        selected_panel_valid_mask=natural["full_valid"],
                    )
                except (RuntimeError, ValueError):
                    continue
                baseline_owners = np.unique(
                    full_owner[interval.union_footprint]
                )
                baseline_owners = baseline_owners[baseline_owners >= 0]
                options.append(
                    (
                        float(row["source_coverage_ratio"]),
                        -int(row["frame_id"]),
                        row,
                        interval,
                        baseline_owners,
                    )
                )
        best = max(options, default=None, key=lambda value: value[:2])
        direct_pass = bool(
            best is not None
            and best[0] >= MINIMUM_NATURAL_COVERAGE
            and best[4].size >= 2
        )
        public_observations = [
            {
                key: value
                for key, value in row.items()
                if not key.startswith("_")
            }
            for row in reconstructed_rows
        ]
        track_audit = {
            "track_id": track_id,
            "original_observation_count": len(observations),
            "consensus_voxel_count": len(consensus),
            "minimum_consensus_views": required_consensus_views,
            "reconstructed_observation_count": len(accepted_rows),
            "direct_single_owner_option_count": len(options),
            "direct_single_owner_pass": direct_pass,
            "best_source": (
                None
                if best is None
                else {
                    "candidate_id": int(best[2]["candidate_id"]),
                    "frame_id": int(best[2]["frame_id"]),
                    "panel_index": int(best[2]["selected_panel_index"]),
                    "source_coverage_ratio": float(best[0]),
                    "baseline_owner_frame_ids": [
                        int(value) for value in best[4]
                    ],
                }
            ),
            "observations": public_observations,
        }
        audit_tracks.append(track_audit)
        if best is not None:
            row = best[2]
            mask = row["_mask"]
            image = image_by_frame[int(row["frame_id"])].copy()
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
            y, x = np.nonzero(mask)
            x0, y0, x1, y1 = (
                int(np.min(x)),
                int(np.min(y)),
                int(np.max(x)) + 1,
                int(np.max(y)) + 1,
            )
            crop_image = image[y0:y1, x0:x1]
            scale = min(1.0, 260.0 / max(crop_image.shape[:2]))
            crop_image = cv2.resize(
                crop_image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            card = np.zeros((190, 300, 3), dtype=np.uint8)
            card[28 : 28 + min(160, crop_image.shape[0]), : min(300, crop_image.shape[1])] = crop_image[
                :160, :300
            ]
            cv2.putText(
                card,
                f"T{track_id} F{row['frame_id']} cov={best[0]:.3f}",
                (4, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            contact_rows.append(card)

    highlighted = {
        str(track_id): next(
            item for item in audit_tracks if item["track_id"] == track_id
        )
        for track_id in (0, 49, 112, 479)
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if contact_rows:
        columns = 4
        rows = []
        for index in range(0, len(contact_rows), columns):
            row = np.hstack(contact_rows[index : index + columns])
            row = cv2.copyMakeBorder(
                row,
                0,
                0,
                0,
                columns * 300 - row.shape[1],
                cv2.BORDER_CONSTANT,
            )
            rows.append(row)
        cv2.imwrite(
            str(output / "dis_consensus_contours_contact_sheet.jpg"),
            np.vstack(rows),
        )
    audit = {
        "schema": "inspection-dis-track-consensus-contours/v1",
        "formal_renderer_modified": False,
        "uniform_policy": (
            "20mm_world_voxels_two_view_consensus_bbox_limited_"
            "same_depth_layer_lab45_grabcut_then_existing_direct_owner_gate"
        ),
        "manual_track_roi_or_frame_selection_used": False,
        "translation_used": False,
        "affine_used": False,
        "additional_position_warp_used": False,
        "hole_fill_used": False,
        "track_count": len(audit_tracks),
        "direct_single_owner_pass_count": sum(
            bool(item["direct_single_owner_pass"]) for item in audit_tracks
        ),
        "highlighted_tracks": highlighted,
        "tracks": audit_tracks,
        "thresholds": {
            "world_voxel_size_mm": VOXEL_SIZE_MM,
            "minimum_consensus_views": MINIMUM_CONSENSUS_VIEWS,
            "minimum_consensus_view_fraction": (
                MINIMUM_CONSENSUS_VIEW_FRACTION
            ),
            "lab_distance": LAB_DISTANCE,
            "probable_dilation_pixels": PROBABLE_DILATION_PIXELS,
            "minimum_natural_source_coverage_ratio": (
                MINIMUM_NATURAL_COVERAGE
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "dis_consensus_contours_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "track_count": audit["track_count"],
                "direct_single_owner_pass_count": audit[
                    "direct_single_owner_pass_count"
                ],
                "highlighted": {
                    key: {
                        "coverage": (
                            None
                            if value["best_source"] is None
                            else value["best_source"][
                                "source_coverage_ratio"
                            ]
                        ),
                        "pass": value["direct_single_owner_pass"],
                    }
                    for key, value in highlighted.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
