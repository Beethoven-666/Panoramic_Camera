from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.session import load_rgbd_session


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of existing FastSAM polygon labels against "
            "automatically discovered low-coverage RGB-D world regions."
        )
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("transforms", type=Path)
    parser.add_argument("observability_report", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_poses(path: Path) -> tuple[list[int], list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    return (
        [int(item["node_id"]) for item in nodes],
        [
            np.asarray(item["camera_to_world"], dtype=np.float64)
            for item in nodes
        ],
    )


def _polygons(path: Path, width: int, height: int) -> list[np.ndarray]:
    if not path.exists():
        return []
    results: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = np.fromstring(line, sep=" ", dtype=np.float64)
        if values.size < 7 or (values.size - 1) % 2:
            continue
        points = values[1:].reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygon = np.rint(points).astype(np.int32)
        if polygon.shape[0] >= 3 and abs(cv2.contourArea(polygon)) >= 16.0:
            results.append(polygon)
    return results


def _project(
    world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(camera_to_world)
    camera = world @ inverse[:3, :3].T + inverse[:3, 3]
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = intrinsics.fx * camera[:, 0] / z + intrinsics.cx
        y = intrinsics.fy * camera[:, 1] / z + intrinsics.cy
    return x, y, z


def _candidate_mask(
    polygons: list[np.ndarray],
    seed_x: np.ndarray,
    seed_y: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray | None, int, int]:
    if seed_x.size == 0:
        return None, -1, 0
    seed_x0 = int(np.min(seed_x))
    seed_y0 = int(np.min(seed_y))
    seed_x1 = int(np.max(seed_x)) + 1
    seed_y1 = int(np.max(seed_y)) + 1
    best_index = -1
    best_overlap = 0
    for index, polygon in enumerate(polygons):
        px, py, pw, ph = cv2.boundingRect(polygon)
        if (
            px >= seed_x1
            or py >= seed_y1
            or px + pw <= seed_x0
            or py + ph <= seed_y0
        ):
            continue
        inside = np.asarray(
            [
                cv2.pointPolygonTest(
                    polygon,
                    (float(x), float(y)),
                    False,
                )
                >= 0.0
                for x, y in zip(seed_x, seed_y, strict=True)
            ],
            dtype=bool,
        )
        overlap = int(np.count_nonzero(inside))
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    if best_index < 0:
        return None, -1, 0
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygons[best_index]], 1)
    return mask.astype(bool), best_index, best_overlap


def main() -> int:
    args = _arguments()
    session = load_rgbd_session(args.session)
    frame_ids, poses = _load_poses(args.transforms)
    by_id = {int(frame.frame_id): frame for frame in session.frames}
    frames = [by_id[frame_id] for frame_id in frame_ids]
    source_report = json.loads(
        args.observability_report.read_text(encoding="utf-8")
    )
    inspection_metadata = json.loads(
        Path(source_report["input_inspection_meta"]).read_text(
            encoding="utf-8"
        )
    )
    renderer = inspection_metadata.get("renderer", inspection_metadata)
    layout = renderer["layout"]
    coverage = renderer["world_surface_coverage_audit"]
    source_regions = source_report["regions"]
    cell_to_region = {
        tuple(int(value) for value in key): int(region["region_id"])
        for region in source_regions
        for key in region["cell_keys"]
    }
    region_count = len(source_regions)
    intrinsics = session.calibration
    stride = int(coverage["sample_stride"])
    voxel_size = float(coverage["voxel_size_mm"])
    reference_depth = float(layout["reference_depth_mm"])
    near_margin = float(coverage["near_depth_margin_mm"])
    scan_axis = np.asarray(
        layout["scan_axis_world"],
        dtype=np.float64,
    )
    down_axis = np.asarray(
        layout["down_axis_world"],
        dtype=np.float64,
    )
    normal_axis = np.asarray(
        layout["normal_axis_world"],
        dtype=np.float64,
    )
    yy, xx = np.indices(
        (intrinsics.height, intrinsics.width), dtype=np.int32
    )
    sample_y = yy[::stride, ::stride].reshape(-1)
    sample_x = xx[::stride, ::stride].reshape(-1)
    region_frame_points: dict[
        tuple[int, int], list[tuple[int, int, np.ndarray]]
    ] = defaultdict(list)
    global_voxels: dict[int, set[tuple[int, int, int]]] = defaultdict(set)
    depth_images: list[np.ndarray] = []
    gray_images: list[np.ndarray] = []
    for frame_index, (frame, pose) in enumerate(
        zip(frames, poses, strict=True)
    ):
        encoded = cv2.imread(
            str(frame.aligned_depth_path), cv2.IMREAD_UNCHANGED
        )
        gray = cv2.imread(str(frame.color_path), cv2.IMREAD_GRAYSCALE)
        depth_images.append(encoded)
        gray_images.append(gray)
        depth = (
            encoded[::stride, ::stride].reshape(-1).astype(np.float64)
            * float(frame.depth_scale_mm_per_unit)
        )
        valid = (
            np.isfinite(depth)
            & (depth >= 200.0)
            & (depth <= 3000.0)
            & (depth < reference_depth - near_margin)
        )
        x = sample_x[valid]
        y = sample_y[valid]
        z = depth[valid]
        camera = np.column_stack(
            (
                (x - intrinsics.cx) * z / intrinsics.fx,
                (y - intrinsics.cy) * z / intrinsics.fy,
                z,
            )
        )
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        basis = np.column_stack(
            (
                world @ scan_axis,
                world @ down_axis,
                world @ normal_axis,
            )
        )
        keys = np.floor(basis / 80.0).astype(np.int32)
        for key in np.unique(keys, axis=0):
            region_id = cell_to_region.get(tuple(int(value) for value in key))
            if region_id is None:
                continue
            take = np.all(keys == key, axis=1)
            selected = world[take]
            region_frame_points[(region_id, frame_index)].extend(
                (
                    int(px),
                    int(py),
                    point,
                )
                for px, py, point in zip(
                    x[take], y[take], selected, strict=True
                )
            )
            global_voxels[region_id].update(
                tuple(int(value) for value in voxel)
                for voxel in np.floor(
                    selected / voxel_size
                ).astype(np.int32)
            )

    region_candidates: list[list[dict[str, object]]] = [
        [] for _ in range(region_count)
    ]
    for frame_index, (frame, pose) in enumerate(
        zip(frames, poses, strict=True)
    ):
        label_path = args.labels / f"{int(frame.frame_id):08d}.txt"
        polygons = _polygons(
            label_path, intrinsics.width, intrinsics.height
        )
        if not polygons:
            continue
        for region_id in range(region_count):
            points = region_frame_points.get((region_id, frame_index), [])
            if len(points) < 4:
                continue
            seed_x = np.asarray([item[0] for item in points], dtype=np.int32)
            seed_y = np.asarray([item[1] for item in points], dtype=np.int32)
            mask, polygon_index, overlap = _candidate_mask(
                polygons,
                seed_x,
                seed_y,
                intrinsics.width,
                intrinsics.height,
            )
            if mask is None or overlap < max(3, int(0.1 * len(points))):
                continue
            area = int(np.count_nonzero(mask))
            ys, xs = np.nonzero(mask)
            bbox = cv2.boundingRect(
                np.column_stack((xs, ys)).astype(np.int32)
            )
            boundary_complete = bool(
                bbox[0] >= 8
                and bbox[1] >= 8
                and bbox[0] + bbox[2] <= intrinsics.width - 8
                and bbox[1] + bbox[3] <= intrinsics.height - 8
            )
            laplacian = cv2.Laplacian(
                gray_images[frame_index], cv2.CV_32F
            )
            blur = float(laplacian[mask].var()) if area else 0.0
            measured_mask_depth = (
                depth_images[frame_index][mask].astype(np.float64)
                * float(frame.depth_scale_mm_per_unit)
            )
            aligned_depth_valid = (
                np.isfinite(measured_mask_depth)
                & (measured_mask_depth >= 200.0)
                & (measured_mask_depth <= 3000.0)
            )
            aligned_depth_valid_ratio = float(
                np.mean(aligned_depth_valid)
            )
            voxels = global_voxels[region_id]
            centers = (
                np.asarray(sorted(voxels), dtype=np.float64) + 0.5
            ) * voxel_size
            projected_x, projected_y, expected_z = _project(
                centers, pose, intrinsics
            )
            finite = (
                np.isfinite(projected_x)
                & np.isfinite(projected_y)
                & np.isfinite(expected_z)
                & (expected_z > 0.0)
            )
            in_bounds = (
                finite
                & (projected_x >= 0.0)
                & (projected_x < intrinsics.width)
                & (projected_y >= 0.0)
                & (projected_y < intrinsics.height)
            )
            inside = np.zeros(centers.shape[0], dtype=bool)
            if np.any(in_bounds):
                xi = np.clip(
                    np.rint(projected_x[in_bounds]).astype(np.intp),
                    0,
                    intrinsics.width - 1,
                )
                yi = np.clip(
                    np.rint(projected_y[in_bounds]).astype(np.intp),
                    0,
                    intrinsics.height - 1,
                )
                inside_indices = np.flatnonzero(in_bounds)
                inside[inside_indices] = mask[yi, xi]
            candidate_world_recall = float(np.mean(inside))
            if np.any(inside):
                xi = np.clip(
                    np.rint(projected_x[inside]).astype(np.intp),
                    0,
                    intrinsics.width - 1,
                )
                yi = np.clip(
                    np.rint(projected_y[inside]).astype(np.intp),
                    0,
                    intrinsics.height - 1,
                )
                measured = (
                    depth_images[frame_index][yi, xi].astype(np.float64)
                    * float(frame.depth_scale_mm_per_unit)
                )
                expected = expected_z[inside]
                valid = (
                    np.isfinite(measured)
                    & (measured >= 200.0)
                    & (measured <= 3000.0)
                )
                tolerance = np.maximum(20.0, 0.02 * expected)
                same_layer = valid & (
                    np.abs(measured - expected) <= tolerance
                )
                occluded = valid & (measured < expected - tolerance)
                same_layer_ratio = float(np.mean(same_layer))
                occluded_ratio = float(np.mean(occluded))
            else:
                same_layer_ratio = 0.0
                occluded_ratio = 0.0
            directly_observed_voxels = {
                tuple(
                    int(value)
                    for value in np.floor(point / voxel_size).astype(np.int32)
                )
                for x, y, point in points
                if mask[int(y), int(x)]
            }
            direct_world_recall = len(directly_observed_voxels) / max(
                1, len(voxels)
            )
            clear = blur >= 50.0
            depth_sufficient = bool(
                aligned_depth_valid_ratio >= 0.80
                and direct_world_recall >= 0.75
                and candidate_world_recall >= 0.75
                and same_layer_ratio >= 0.60
                and occluded_ratio <= 0.10
            )
            complete = bool(
                boundary_complete and clear and depth_sufficient
            )
            score = (
                0.20 * float(boundary_complete)
                + 0.15 * min(1.0, blur / 100.0)
                + 0.15 * aligned_depth_valid_ratio
                + 0.20 * direct_world_recall
                + 0.15 * candidate_world_recall
                + 0.10 * same_layer_ratio
                + 0.05 * (1.0 - min(1.0, occluded_ratio))
            )
            audit = {
                "frame_id": int(frame.frame_id),
                "polygon_index": int(polygon_index),
                "mask_area_pixels": area,
                "source_bbox_xywh": [int(value) for value in bbox],
                "seed_sample_count": len(points),
                "seed_overlap_count": overlap,
                "seed_overlap_ratio": float(overlap / len(points)),
                "boundary_complete": boundary_complete,
                "laplacian_variance": blur,
                "clear": bool(clear),
                "aligned_depth_valid_ratio": aligned_depth_valid_ratio,
                "direct_region_world_voxel_recall": float(
                    direct_world_recall
                ),
                "candidate_projected_world_voxel_recall": (
                    candidate_world_recall
                ),
                "same_layer_ratio": same_layer_ratio,
                "occluded_ratio": occluded_ratio,
                "depth_contour_sufficient": depth_sufficient,
                "complete_candidate": complete,
                "score": float(score),
            }
            region_candidates[region_id].append(audit)

    final_regions: list[dict[str, object]] = []
    contact_items: list[tuple[str, np.ndarray]] = []
    for region in source_regions:
        region_id = int(region["region_id"])
        candidates = sorted(
            region_candidates[region_id],
            key=lambda item: (-float(item["score"]), int(item["frame_id"])),
        )
        complete = [
            item for item in candidates if bool(item["complete_candidate"])
        ]
        consistent_pairs: list[dict[str, object]] = []
        for first_index, first in enumerate(complete):
            for second in complete[first_index + 1 :]:
                area_ratio = max(
                    int(first["mask_area_pixels"]),
                    int(second["mask_area_pixels"]),
                ) / max(
                    1,
                    min(
                        int(first["mask_area_pixels"]),
                        int(second["mask_area_pixels"]),
                    ),
                )
                recall_floor = min(
                    float(first["direct_region_world_voxel_recall"]),
                    float(second["direct_region_world_voxel_recall"]),
                )
                if area_ratio <= 2.5 and recall_floor >= 0.75:
                    consistent_pairs.append(
                        {
                            "frame_ids": [
                                int(first["frame_id"]),
                                int(second["frame_id"]),
                            ],
                            "mask_area_ratio": float(area_ratio),
                            "minimum_direct_world_recall": recall_floor,
                        }
                    )
        best = candidates[0] if candidates else None
        final_regions.append(
            {
                "region_id": region_id,
                "cell_count": int(region["cell_count"]),
                "world_voxel_count": int(region["world_voxel_count"]),
                "full_canvas_bbox_xywh": region[
                    "full_canvas_bbox_xywh"
                ],
                "candidate_count": len(candidates),
                "complete_candidate_count": len(complete),
                "consistent_two_frame_candidate": bool(consistent_pairs),
                "consistent_pairs": consistent_pairs,
                "best_candidate": best,
                "top_candidates": candidates[:5],
            }
        )
        if best is not None and len(contact_items) < 26:
            frame_id = int(best["frame_id"])
            frame = by_id[frame_id]
            polygons = _polygons(
                args.labels / f"{frame_id:08d}.txt",
                intrinsics.width,
                intrinsics.height,
            )
            polygon = polygons[int(best["polygon_index"])]
            image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
            cv2.drawContours(image, [polygon], -1, (0, 255, 0), 2)
            x0, y0, width, height = (
                int(value) for value in best["source_bbox_xywh"]
            )
            crop = image[y0 : y0 + height, x0 : x0 + width]
            if crop.size:
                contact_items.append(
                    (
                        f"R{region_id} F{best['frame_id']} "
                        f"pair={bool(consistent_pairs)}",
                        crop,
                    )
                )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "gemini305-fastsam-observability-audit/v1",
        "formal_renderer_modified": False,
        "torch_or_fastsam_inference_executed": False,
        "labels": str(args.labels.resolve()),
        "automatic_region_source": str(
            args.observability_report.resolve()
        ),
        "thresholds": {
            "mask_border_margin_pixels": 8,
            "clear_laplacian_variance": 50.0,
            "aligned_depth_valid_ratio": 0.80,
            "direct_region_world_voxel_recall": 0.75,
            "candidate_projected_world_voxel_recall": 0.75,
            "same_layer_ratio": 0.60,
            "maximum_occluded_ratio": 0.10,
            "consistent_pair_maximum_area_ratio": 2.5,
        },
        "region_count": region_count,
        "region_with_any_candidate_count": sum(
            bool(item["candidate_count"]) for item in final_regions
        ),
        "region_with_complete_candidate_count": sum(
            bool(item["complete_candidate_count"]) for item in final_regions
        ),
        "region_with_consistent_two_frame_candidate_count": sum(
            bool(item["consistent_two_frame_candidate"])
            for item in final_regions
        ),
        "regions": final_regions,
    }
    (output / "fastsam_observability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    tiles: list[np.ndarray] = []
    for label, crop in contact_items:
        scale = min(1.0, 300.0 / max(crop.shape[:2]))
        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
        tile = np.zeros((250, 330, 3), dtype=np.uint8)
        height = min(220, resized.shape[0])
        width = min(330, resized.shape[1])
        tile[25 : 25 + height, :width] = resized[:height, :width]
        cv2.putText(
            tile,
            label,
            (4, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if tiles:
        rows = []
        for index in range(0, len(tiles), 4):
            row = np.hstack(tiles[index : index + 4])
            row = cv2.copyMakeBorder(
                row,
                0,
                0,
                0,
                4 * 330 - row.shape[1],
                cv2.BORDER_CONSTANT,
            )
            rows.append(row)
        cv2.imwrite(
            str(output / "fastsam_candidate_contact_sheet.jpg"),
            np.vstack(rows),
        )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "region_count",
                    "region_with_any_candidate_count",
                    "region_with_complete_candidate_count",
                    "region_with_consistent_two_frame_candidate_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
