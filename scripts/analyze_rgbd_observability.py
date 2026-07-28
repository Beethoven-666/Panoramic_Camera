from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.session import load_rgbd_session


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only RGB-D observability audit for low-coverage inspection "
            "world cells. This command never changes a formal render."
        )
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("transforms", type=Path)
    parser.add_argument("inspection_meta", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _groups(
    cells: list[dict[str, object]],
) -> list[list[dict[str, object]]]:
    by_key = {
        tuple(int(value) for value in cell["cell_scan_down_normal"]): cell
        for cell in cells
    }
    remaining = set(by_key)
    groups: list[list[dict[str, object]]] = []
    offsets = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque((start,))
        selected: list[dict[str, object]] = []
        while queue:
            key = queue.popleft()
            selected.append(by_key[key])
            for offset in offsets:
                neighbour = tuple(
                    key[axis] + offset[axis] for axis in range(3)
                )
                if neighbour in remaining:
                    remaining.remove(neighbour)
                    queue.append(neighbour)
        groups.append(selected)
    return sorted(
        groups,
        key=lambda items: -sum(
            int(item["world_voxel_count"]) for item in items
        ),
    )


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


def _project(
    world: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(camera_to_world)
    camera = (
        world @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    z = camera[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = intrinsics.fx * camera[:, 0] / z + intrinsics.cx
        y = intrinsics.fy * camera[:, 1] / z + intrinsics.cy
    return x, y, z


def main() -> int:
    args = _arguments()
    session = load_rgbd_session(args.session)
    frame_ids, poses = _load_poses(args.transforms)
    by_id = {int(frame.frame_id): frame for frame in session.frames}
    frames = [by_id[frame_id] for frame_id in frame_ids]
    metadata = json.loads(args.inspection_meta.read_text(encoding="utf-8"))
    renderer = metadata.get("renderer", metadata)
    coverage = renderer["world_surface_coverage_audit"]
    layout = renderer["layout"]
    low_cells = list(coverage["low_coverage_cells"])
    grouped = _groups(low_cells)

    scan_axis = np.asarray(layout["scan_axis_world"], dtype=np.float64)
    down_axis = np.asarray(layout["down_axis_world"], dtype=np.float64)
    normal_axis = np.asarray(layout["normal_axis_world"], dtype=np.float64)
    cell_to_region: dict[tuple[int, int, int], int] = {}
    regions: list[dict[str, object]] = []
    for group_index, items in enumerate(grouped):
        for item in items:
            key = tuple(
                int(value) for value in item["cell_scan_down_normal"]
            )
            cell_to_region[key] = group_index
        bounds = np.asarray(
            [item["basis_bounds_mm"] for item in items], dtype=np.float64
        )
        canvas_boxes = np.asarray(
            [item["full_canvas_bbox_xywh"] for item in items],
            dtype=np.float64,
        )
        x0 = float(np.min(canvas_boxes[:, 0]))
        y0 = float(np.min(canvas_boxes[:, 1]))
        x1 = float(np.max(canvas_boxes[:, 0] + canvas_boxes[:, 2]))
        y1 = float(np.max(canvas_boxes[:, 1] + canvas_boxes[:, 3]))
        regions.append(
            {
                "region_id": group_index,
                "cell_count": len(items),
                "world_voxel_count": int(
                    sum(int(item["world_voxel_count"]) for item in items)
                ),
                "cell_keys": [
                    item["cell_scan_down_normal"] for item in items
                ],
                "basis_bounds_mm": [
                    np.min(bounds[:, 0, :], axis=0).tolist(),
                    np.max(bounds[:, 1, :], axis=0).tolist(),
                ],
                "full_canvas_bbox_xywh": [
                    int(np.floor(x0)),
                    int(np.floor(y0)),
                    int(np.ceil(x1 - x0)),
                    int(np.ceil(y1 - y0)),
                ],
            }
        )

    region_frame_points: dict[
        tuple[int, int], list[tuple[int, int, np.ndarray]]
    ] = defaultdict(list)
    region_world_voxels: dict[int, set[tuple[int, int, int]]] = defaultdict(
        set
    )
    stride = int(coverage["sample_stride"])
    voxel_size = float(coverage["voxel_size_mm"])
    reference_depth = float(layout["reference_depth_mm"])
    near_margin = float(coverage["near_depth_margin_mm"])
    yy_grid, xx_grid = np.indices(
        (
            session.calibration.height,
            session.calibration.width,
        ),
        dtype=np.int32,
    )
    sample_y = yy_grid[::stride, ::stride].reshape(-1)
    sample_x = xx_grid[::stride, ::stride].reshape(-1)
    intrinsics = session.calibration
    depth_images = [
        cv2.imread(str(frame.aligned_depth_path), cv2.IMREAD_UNCHANGED)
        for frame in frames
    ]
    gray_images = [
        cv2.imread(str(frame.color_path), cv2.IMREAD_GRAYSCALE)
        for frame in frames
    ]
    for frame_index, (frame, pose) in enumerate(
        zip(frames, poses, strict=True)
    ):
        encoded = depth_images[frame_index]
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
        if not np.any(valid):
            continue
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
        cell_keys = np.floor(basis / 80.0).astype(np.int32)
        for key in np.unique(cell_keys, axis=0):
            region_id = cell_to_region.get(tuple(int(v) for v in key))
            if region_id is None:
                continue
            take = np.all(cell_keys == key, axis=1)
            selected_world = world[take]
            region_frame_points[(region_id, frame_index)].extend(
                (
                    int(px),
                    int(py),
                    point,
                )
                for px, py, point in zip(
                    x[take], y[take], selected_world, strict=True
                )
            )
            voxel_keys = np.floor(
                selected_world / voxel_size
            ).astype(np.int32)
            region_world_voxels[region_id].update(
                tuple(int(value) for value in voxel)
                for voxel in voxel_keys
            )

    low_owner_components = []
    locks = renderer["background_seam_audit"][
        "foreground_component_locks"
    ]["components"]
    for component in locks:
        candidates = component["candidates"]
        complete_count = sum(
            bool(item["complete_reference_coverage"])
            for item in candidates
        )
        mesh_ratio = max(
            (
                int(item["depth_mesh_coverage_pixels"])
                / max(1, int(component["area_pixels"]))
                for item in candidates
            ),
            default=0.0,
        )
        if (
            int(component["raw_component_area_pixels"]) >= 100
            and (complete_count <= 1 or mesh_ratio < 0.8)
        ):
            low_owner_components.append(
                {
                    "component_id": int(component["component_id"]),
                    "center_x": float(component["center_x"]),
                    "area_pixels": int(component["area_pixels"]),
                    "complete_candidate_count": int(complete_count),
                    "maximum_depth_mesh_coverage_ratio": float(mesh_ratio),
                    "selected_frame_id": int(
                        component["selected_frame_id"]
                    ),
                }
            )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contact_items: list[tuple[str, np.ndarray]] = []
    final_regions: list[dict[str, object]] = []
    for region in regions:
        region_id = int(region["region_id"])
        global_voxels = region_world_voxels.get(region_id, set())
        if not global_voxels:
            region["status"] = "no_raw_samples_recovered"
            final_regions.append(region)
            continue
        world_centers = (
            np.asarray(sorted(global_voxels), dtype=np.float64) + 0.5
        ) * voxel_size
        frame_audits: list[dict[str, object]] = []
        for frame_index, (frame, pose) in enumerate(
            zip(frames, poses, strict=True)
        ):
            x, y, expected_z = _project(
                world_centers, pose, intrinsics
            )
            finite = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(expected_z)
                & (expected_z > 0.0)
            )
            in_bounds = (
                finite
                & (x >= 0.0)
                & (x < intrinsics.width)
                & (y >= 0.0)
                & (y < intrinsics.height)
            )
            in_bounds_ratio = float(np.mean(in_bounds))
            points = region_frame_points.get((region_id, frame_index), [])
            observed_voxels = {
                tuple(
                    int(value)
                    for value in np.floor(point / voxel_size).astype(np.int32)
                )
                for _, _, point in points
            }
            observed_world_ratio = len(observed_voxels) / max(
                1, len(global_voxels)
            )
            if np.any(in_bounds):
                xi = np.clip(
                    np.rint(x[in_bounds]).astype(np.intp),
                    0,
                    intrinsics.width - 1,
                )
                yi = np.clip(
                    np.rint(y[in_bounds]).astype(np.intp),
                    0,
                    intrinsics.height - 1,
                )
                encoded = depth_images[frame_index]
                measured = (
                    encoded[yi, xi].astype(np.float64)
                    * float(frame.depth_scale_mm_per_unit)
                )
                expected = expected_z[in_bounds]
                depth_valid = (
                    np.isfinite(measured)
                    & (measured >= 200.0)
                    & (measured <= 3000.0)
                )
                tolerance = np.maximum(20.0, 0.02 * expected)
                same = depth_valid & (
                    np.abs(measured - expected) <= tolerance
                )
                occluded = depth_valid & (
                    measured < expected - tolerance
                )
                depth_valid_ratio = float(np.mean(depth_valid))
                same_layer_ratio = float(np.mean(same))
                occluded_ratio = float(np.mean(occluded))
                xs = x[in_bounds]
                ys = y[in_bounds]
                x0 = max(0, int(np.floor(np.min(xs))) - 4)
                y0 = max(0, int(np.floor(np.min(ys))) - 4)
                x1 = min(
                    intrinsics.width, int(np.ceil(np.max(xs))) + 5
                )
                y1 = min(
                    intrinsics.height, int(np.ceil(np.max(ys))) + 5
                )
                crop = gray_images[frame_index][y0:y1, x0:x1]
                blur = (
                    float(
                        cv2.Laplacian(
                            crop,
                            cv2.CV_32F,
                        ).var()
                    )
                    if crop.size
                    else 0.0
                )
                border_margin = float(
                    min(
                        np.min(xs),
                        np.min(ys),
                        intrinsics.width - 1 - np.max(xs),
                        intrinsics.height - 1 - np.max(ys),
                    )
                )
                source_bbox = [x0, y0, x1 - x0, y1 - y0]
            else:
                depth_valid_ratio = 0.0
                same_layer_ratio = 0.0
                occluded_ratio = 0.0
                blur = 0.0
                border_margin = -1.0
                source_bbox = [0, 0, 0, 0]
            rgb_complete = (
                in_bounds_ratio >= 0.98 and border_margin >= 8.0
            )
            clear = blur >= 50.0
            depth_contour_sufficient = (
                observed_world_ratio >= 0.75
                and depth_valid_ratio >= 0.80
                and same_layer_ratio >= 0.60
                and occluded_ratio <= 0.10
            )
            complete_observation = bool(
                rgb_complete and clear and depth_contour_sufficient
            )
            score = (
                0.25 * in_bounds_ratio
                + 0.30 * observed_world_ratio
                + 0.20 * same_layer_ratio
                + 0.10 * depth_valid_ratio
                + 0.10 * min(1.0, blur / 100.0)
                + 0.05 * (1.0 - min(1.0, occluded_ratio))
            )
            frame_audits.append(
                {
                    "frame_id": int(frame.frame_id),
                    "source_bbox_xywh": source_bbox,
                    "in_bounds_ratio": in_bounds_ratio,
                    "border_margin_pixels": border_margin,
                    "observed_world_voxel_ratio": float(
                        observed_world_ratio
                    ),
                    "laplacian_variance": blur,
                    "depth_valid_ratio": depth_valid_ratio,
                    "same_layer_ratio": same_layer_ratio,
                    "occluded_ratio": occluded_ratio,
                    "rgb_complete": bool(rgb_complete),
                    "clear": bool(clear),
                    "depth_contour_sufficient": bool(
                        depth_contour_sufficient
                    ),
                    "complete_rgbd_observation": complete_observation,
                    "score": float(score),
                }
            )
        frame_audits.sort(
            key=lambda item: (-float(item["score"]), int(item["frame_id"]))
        )
        best = frame_audits[0]
        region_box = region["full_canvas_bbox_xywh"]
        x0 = float(region_box[0])
        x1 = x0 + float(region_box[2])
        attached = [
            item
            for item in low_owner_components
            if x0 - 40.0 <= float(item["center_x"]) <= x1 + 40.0
        ]
        region.update(
            {
                "status": (
                    "complete_clear_rgbd_source_exists"
                    if any(
                        bool(item["complete_rgbd_observation"])
                        for item in frame_audits
                    )
                    else "no_complete_clear_rgbd_source"
                ),
                "global_observed_world_voxel_count": len(global_voxels),
                "attached_low_coverage_owner_components": attached,
                "best_frame": best,
                "top_frames": frame_audits[:5],
                "complete_frame_ids": [
                    int(item["frame_id"])
                    for item in frame_audits
                    if bool(item["complete_rgbd_observation"])
                ],
            }
        )
        final_regions.append(region)
        if (
            len(contact_items) < 24
            and int(region["world_voxel_count"]) >= 80
            and int(best["source_bbox_xywh"][2]) > 0
        ):
            frame = by_id[int(best["frame_id"])]
            image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
            bx, by, bw, bh = (
                int(value) for value in best["source_bbox_xywh"]
            )
            crop = image[by : by + bh, bx : bx + bw]
            if crop.size:
                contact_items.append(
                    (
                        f"R{region_id} F{best['frame_id']} "
                        f"ok={best['complete_rgbd_observation']}",
                        crop,
                    )
                )

    report = {
        "schema": "gemini305-rgbd-observability-diagnostic/v1",
        "formal_render_modified": False,
        "input_session": str(session.root),
        "input_transforms": str(args.transforms.resolve()),
        "input_inspection_meta": str(args.inspection_meta.resolve()),
        "thresholds": {
            "rgb_in_bounds_ratio": 0.98,
            "rgb_border_margin_pixels": 8.0,
            "clear_laplacian_variance": 50.0,
            "observed_world_voxel_ratio": 0.75,
            "depth_valid_ratio": 0.80,
            "same_layer_ratio": 0.60,
            "maximum_occluded_ratio": 0.10,
        },
        "source_world_coverage": {
            key: coverage[key]
            for key in (
                "observed_world_voxel_count",
                "multiview_observed_world_voxel_count",
                "matched_multiview_world_voxel_count",
                "multiview_world_coverage_ratio",
                "low_coverage_cell_count",
            )
        },
        "automatic_region_policy": (
            "six_connected_low_coverage_80mm_world_cells_with_"
            "low_depth_mesh_owner_components_attached_by_canvas_x"
        ),
        "region_count": len(final_regions),
        "complete_region_count": sum(
            item.get("status")
            == "complete_clear_rgbd_source_exists"
            for item in final_regions
        ),
        "incomplete_region_count": sum(
            item.get("status") == "no_complete_clear_rgbd_source"
            for item in final_regions
        ),
        "regions": final_regions,
    }
    (output / "observability_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if contact_items:
        tiles = []
        for label, crop in contact_items:
            scale = min(1.0, 320.0 / max(crop.shape[:2]))
            tile = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.zeros((260, 340, 3), dtype=np.uint8)
            height, width = tile.shape[:2]
            canvas[25 : 25 + height, :width] = tile[
                : min(height, 235), : min(width, 340)
            ]
            cv2.putText(
                canvas,
                label,
                (4, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(canvas)
        rows = [
            np.hstack(tiles[index : index + 4])
            for index in range(0, len(tiles), 4)
        ]
        if len(rows[-1]) != 4 * 340:
            rows[-1] = cv2.copyMakeBorder(
                rows[-1],
                0,
                0,
                0,
                4 * 340 - rows[-1].shape[1],
                cv2.BORDER_CONSTANT,
            )
        cv2.imwrite(str(output / "best_frame_contact_sheet.jpg"), np.vstack(rows))
    print(
        json.dumps(
            {
                "region_count": report["region_count"],
                "complete_region_count": report["complete_region_count"],
                "incomplete_region_count": report[
                    "incomplete_region_count"
                ],
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
