"""Audit automatic RGB-D complete-object handoffs on a formal inspection.

This is an isolated diagnostic.  It reconstructs the renderer's selected
depth meshes, starts only from its measured foreground component labels, and
tests whether a baseline multi-owner component has at least two consistent
direct SE(3) projections.  It never changes the formal output or renderer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
    _build_depth_mesh_panel_remap,
    _build_foreground_component_owner_locks,
    _composite_reference_panel,
    _depth_confidence,
    _read_rgbd,
    _undistortion_maps,
)
from panorama_demo.inspection_object_handoff import (
    AutomaticObjectHandoffRejected,
    ObjectHandoffSource,
    select_automatic_complete_object_owner,
)
from panorama_demo.session import load_rgbd_session


def _layout_from_report(value: dict[str, object]) -> InspectionMultiviewLayout:
    return InspectionMultiviewLayout(
        width=int(value["width"]),
        height=int(value["height"]),
        reference_depth_mm=float(value["reference_depth_mm"]),
        scan_axis=tuple(float(item) for item in value["scan_axis_world"]),
        down_axis=tuple(float(item) for item in value["down_axis_world"]),
        normal_axis=tuple(float(item) for item in value["normal_axis_world"]),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=int(item["panel_index"]),
                anchor_scan_mm=float(item["anchor_scan_mm"]),
                canvas_offset_x=float(item["canvas_offset_x"]),
                center_world_mm=tuple(
                    float(component) for component in item["center_world_mm"]
                ),
            )
            for item in value["panels"]
        ),
        panel_step_mm=float(value["panel_step_mm"]),
        canvas_megapixels=float(value["canvas_megapixels"]),
    )


def _load_full_baseline(
    output: Path,
    layout: InspectionMultiviewLayout,
    crop: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cropped_image = cv2.imread(
        str(output / "mosaic_inspection.png"), cv2.IMREAD_COLOR
    )
    encoded_owner = cv2.imread(
        str(output / "inspection_owner.png"), cv2.IMREAD_UNCHANGED
    )
    if (
        cropped_image is None
        or encoded_owner is None
        or encoded_owner.dtype != np.uint16
        or cropped_image.shape[:2] != encoded_owner.shape
    ):
        raise RuntimeError("Formal inspection RGB/owner products are invalid")
    x0, y0, width, height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    if (
        cropped_image.shape[:2] != (height, width)
        or x0 < 0
        or y0 < 0
        or x0 + width > layout.width
        or y0 + height > layout.height
    ):
        raise RuntimeError("Formal inspection crop is inconsistent")
    full_image = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    full_owner = np.full((layout.height, layout.width), -1, dtype=np.int32)
    full_valid = np.zeros((layout.height, layout.width), dtype=bool)
    full_image[y0 : y0 + height, x0 : x0 + width] = cropped_image
    decoded_owner = encoded_owner.astype(np.int32) - 1
    full_owner[y0 : y0 + height, x0 : x0 + width] = decoded_owner
    full_valid[y0 : y0 + height, x0 : x0 + width] = decoded_owner >= 0
    return full_image, full_owner, full_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("formal_output")
    parser.add_argument(
        "--minimum-component-pixels", type=int, default=300
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    session_path = Path(arguments.session).expanduser().resolve()
    output = Path(arguments.formal_output).expanduser().resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    render = report["render"]
    layout = _layout_from_report(render["layout"])
    config = InspectionMultiviewConfig.from_mapping(render["config"])
    session = load_rgbd_session(session_path)
    frame_by_id = {int(frame.frame_id): frame for frame in session.frames}
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    full_image, full_owner, full_valid = _load_full_baseline(
        output, layout, render["crop"]
    )
    maps = _undistortion_maps(session.calibration)
    reference_rasters = []
    depth_mesh_candidates = []
    sources: list[ObjectHandoffSource] = []
    dummy_image = np.zeros_like(full_image)
    dummy_depth = np.full(
        (layout.height, layout.width), np.inf, dtype=np.float32
    )
    dummy_confidence = np.zeros(
        (layout.height, layout.width), dtype=np.float32
    )
    dummy_owner = np.full(
        (layout.height, layout.width), -1, dtype=np.int32
    )
    dummy_reliable = np.zeros(
        (layout.height, layout.width), dtype=bool
    )
    for selected in render["selected_panel_sources"]:
        panel_index = int(selected["panel_index"])
        frame_id = int(selected["frame_id"])
        frame = frame_by_id[frame_id]
        pose = pose_by_id[frame_id]
        image, depth, geometric_valid = _read_rgbd(
            frame, session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        confidence, edge = _depth_confidence(depth, reliable, config)
        foreground_margin = max(
            config.foreground_depth_margin_mm,
            config.foreground_depth_margin_ratio
            * layout.reference_depth_mm,
        )
        geometry_limit = min(
            layout.reference_depth_mm - foreground_margin,
            layout.reference_depth_mm
            * config.foreground_reference_depth_ratio,
        )
        geometry_depth = reliable & (depth < geometry_limit)
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        radius = np.sqrt(
            (
                (xx - session.calibration.cx)
                / max(1.0, session.calibration.width * 0.5)
            )
            ** 2
            + (
                (yy - session.calibration.cy)
                / max(1.0, session.calibration.height * 0.5)
            )
            ** 2
        )
        centrality = np.clip(1.0 - radius, 0.0, 1.0)
        confidence[geometry_depth] *= (
            0.35 + 0.65 * centrality[geometry_depth]
        )
        projection_valid = (
            geometry_depth & ~edge & (confidence >= np.float32(0.50))
        )
        _, raster = _composite_reference_panel(
            output_image=dummy_image,
            output_depth=dummy_depth,
            output_confidence=dummy_confidence,
            output_owner=dummy_owner,
            output_reliable_depth=dummy_reliable,
            source_image=image,
            source_protected_mask=~reliable,
            source_pose=pose,
            frame_id=frame_id,
            panel_index=panel_index,
            layout=layout,
            intrinsics=session.calibration,
            retain_reference_maps=False,
        )
        mesh = _build_depth_mesh_panel_remap(
            source_depth_mm=depth,
            source_solver_valid=projection_valid,
            source_pose=pose,
            panel_index=panel_index,
            layout=layout,
            intrinsics=session.calibration,
            config=config,
        )
        reference_rasters.append(raster)
        depth_mesh_candidates.append(
            (mesh, raster.image_bgr, confidence, frame_id)
        )
        sources.append(
            ObjectHandoffSource(
                frame_id=frame_id,
                panel_index=panel_index,
                image_bgr=np.ascontiguousarray(image),
                depth_mm=np.ascontiguousarray(depth),
                reliable_depth=np.ascontiguousarray(reliable),
                camera_to_world=np.ascontiguousarray(pose),
                mesh_corner_x=int(mesh.corner_x),
                mesh_map_x=np.ascontiguousarray(mesh.map_x),
                mesh_map_y=np.ascontiguousarray(mesh.map_y),
                mesh_valid_mask=np.ascontiguousarray(mesh.valid_mask),
                mesh_relative_depth_mm=np.ascontiguousarray(
                    mesh.relative_depth_mm
                ),
            )
        )
    locked, labels, lock_audit = _build_foreground_component_owner_locks(
        reference_rasters=reference_rasters,
        depth_mesh_candidates=depth_mesh_candidates,
        layout=layout,
        reference_depth_mm=layout.reference_depth_mm,
        config=config,
    )

    diagnostic = full_image.copy()
    accepted_mask = np.zeros(full_valid.shape, dtype=bool)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    candidate_count = 0
    for label in np.unique(labels):
        if int(label) <= 0:
            continue
        component = labels == int(label)
        area = int(np.count_nonzero(component))
        if area < int(arguments.minimum_component_pixels):
            continue
        owners = np.unique(full_owner[component & full_valid])
        owners = owners[owners >= 0]
        if owners.size < 2:
            continue
        component_y, component_x = np.nonzero(component)
        component_bbox = [
            int(np.min(component_x)),
            int(np.min(component_y)),
            int(np.max(component_x) - np.min(component_x) + 1),
            int(np.max(component_y) - np.min(component_y) + 1),
        ]
        candidate_count += 1
        panels = np.unique(locked[component])
        panels = panels[panels >= 0]
        if panels.size != 1:
            rejected.append(
                {
                    "component_label": int(label),
                    "area_pixels": area,
                    "bbox_xywh": component_bbox,
                    "baseline_owner_frame_ids": [
                        int(value) for value in owners
                    ],
                    "reason": "component_has_no_unique_locked_panel",
                }
            )
            continue
        try:
            handoff = select_automatic_complete_object_owner(
                target_component_mask=component,
                baseline_owner_frame_id=full_owner,
                sources=sources,
                target_panel_index=int(panels[0]),
                layout=layout,
                intrinsics=session.calibration,
                minimum_seed_pixels=max(80, min(400, area // 5)),
                minimum_target_component_recall=0.98,
                minimum_cross_view_iou=0.30,
                minimum_selected_union_coverage_ratio=0.95,
            )
            target = handoff.owner.target_mask
            overlap = int(np.count_nonzero(target & accepted_mask))
            if overlap:
                raise RuntimeError(
                    "automatic owner overlaps another accepted object"
                )
            component_covered = int(np.count_nonzero(target & component))
            component_coverage_ratio = float(component_covered / area)
            if component_coverage_ratio < 0.98:
                raise RuntimeError(
                    "automatic owner leaves a duplicate baseline footprint"
                )
            diagnostic[target] = handoff.owner.target_image_bgr[target]
            accepted_mask |= target
            yy_target, xx_target = np.nonzero(target)
            accepted.append(
                {
                    "component_label": int(label),
                    "area_pixels": area,
                    "bbox_xywh": [
                        int(np.min(xx_target)),
                        int(np.min(yy_target)),
                        int(np.max(xx_target) - np.min(xx_target) + 1),
                        int(np.max(yy_target) - np.min(yy_target) + 1),
                    ],
                    "component_coverage_ratio": component_coverage_ratio,
                    "remaining_baseline_component_pixel_count": int(
                        area - component_covered
                    ),
                    "accepted_owner_pixel_count": int(
                        np.count_nonzero(target)
                    ),
                    "baseline_owner_frame_ids": [
                        int(value) for value in owners
                    ],
                    "selected_panel_index": int(panels[0]),
                    "handoff": handoff.audit,
                }
            )
        except (RuntimeError, ValueError) as exc:
            rejection = {
                "component_label": int(label),
                "area_pixels": area,
                "bbox_xywh": component_bbox,
                "baseline_owner_frame_ids": [
                    int(value) for value in owners
                ],
                "selected_panel_index": int(panels[0]),
                "reason": str(exc),
            }
            if isinstance(exc, AutomaticObjectHandoffRejected):
                rejection["handoff_rejection_audit"] = exc.audit
            rejected.append(rejection)

    crop = render["crop"]
    crop_x, crop_y, crop_width, crop_height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    diagnostic_crop = diagnostic[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    overlay = diagnostic_crop.copy()
    for item in accepted:
        x, y, width, height = item["bbox_xywh"]
        cv2.rectangle(
            overlay,
            (max(0, x - crop_x), max(0, y - crop_y)),
            (
                min(crop_width - 1, x + width - 1 - crop_x),
                min(crop_height - 1, y + height - 1 - crop_y),
            ),
            (0, 255, 0),
            2,
        )
    diagnostic_path = (
        output / "diagnostic_automatic_rgbd_object_handoff.png"
    )
    overlay_path = (
        output / "diagnostic_automatic_rgbd_object_handoff_overlay.png"
    )
    audit_path = (
        output / "diagnostic_automatic_rgbd_object_handoff_audit.json"
    )
    if not cv2.imwrite(str(diagnostic_path), diagnostic_crop):
        raise RuntimeError("Could not write automatic object handoff image")
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError("Could not write automatic handoff overlay")
    audit = {
        "schema": "inspection-object-handoff-diagnostic/v1",
        "formal_output_modified": False,
        "formal_acceptance": False,
        "formal_acceptance_reason": (
            "isolated_diagnostic_not_connected_to_monotone_owner_chain"
        ),
        "policy": (
            "renderer_measured_multi_owner_component_automatic_mesh_seed_"
            "two_view_direct_rgbd_se3_one_rgb_owner"
        ),
        "manual_bbox_used": False,
        "manual_frame_id_used": False,
        "source_scope": "renderer_selected_full_fov_rgbd_panel_sources",
        "source_count": len(sources),
        "candidate_component_count": candidate_count,
        "accepted_component_count": len(accepted),
        "rejected_component_count": len(rejected),
        "accepted_pixel_count": int(np.count_nonzero(accepted_mask)),
        "accepted_components": accepted,
        "rejected_components": rejected,
        "foreground_lock_audit": lock_audit,
        "duplicate_prevention": {
            "minimum_target_component_coverage_ratio": 0.98,
            "minimum_selected_cross_view_union_coverage_ratio": 0.95,
            "accepted_owner_overlap_pixel_count": 0,
            "silent_fallback_allowed": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "diagnostic": diagnostic_path.name,
            "overlay": overlay_path.name,
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_path)
    print(diagnostic_path)
    print(overlay_path)
    print(
        json.dumps(
            {
                "candidate_component_count": candidate_count,
                "accepted_component_count": len(accepted),
                "rejected_component_count": len(rejected),
                "accepted_pixel_count": int(np.count_nonzero(accepted_mask)),
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
