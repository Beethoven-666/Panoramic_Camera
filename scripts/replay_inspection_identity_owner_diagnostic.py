"""Real-data replay of identity-owned RGB-D inverse meshes.

This diagnostic intentionally consumes retained FastSAM/OCR evidence so the
new renderer/compositor can be validated independently of model execution.
It never modifies a formal delivery.  A subsequent formal run must recreate
the same evidence in memory with the CUDA ONNX adapters.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.fastsam_dis_tracking import (
    FastSAMDISFrameInput,
    track_fastsam_dis_frames,
)
from panorama_demo.dis_track_direct_handoff import DirectHandoffConfig
from panorama_demo.inspection_fastsam_track import parse_fastsam_polygons
from panorama_demo.inspection_identity_owner_planner import (
    InspectionIdentityOwnerFrame,
    plan_direct_stable_track_identity_owners,
    plan_inspection_identity_owner_intervals,
)
from panorama_demo.inspection_identity_mesh import (
    InspectionIdentityMeshConfig,
    InspectionIdentityMeshSource,
    composite_inspection_identity_owners,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    _read_rgbd,
    _reference_panel_inverse_maps,
    _undistortion_maps,
    estimate_inspection_layout,
    render_inspection_multiview,
)
from panorama_demo.inspection_ocr_panel import extract_ocr_seeded_panel
from panorama_demo.session import load_rgbd_session


def _atomic_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f"Could not encode {path.name}")
    pending = path.with_name(f".{path.name}.pending")
    pending.write_bytes(encoded.tobytes())
    os.replace(pending, path)


def _atomic_json(path: Path, value: object) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(pending, path)


def _mesh_feasible_direct_owners(
    owners,
    *,
    identity_frames,
    layout,
    intrinsics,
    config,
):
    sources = {
        int(frame.frame_id): InspectionIdentityMeshSource(
            panel_index=int(frame.panel_index),
            frame_id=int(frame.frame_id),
            image_bgr=np.asarray(frame.image_bgr),
            depth_mm=np.asarray(frame.depth_mm),
            reliable_depth=np.asarray(frame.reliable_depth),
            camera_to_world=np.asarray(frame.camera_to_world),
        )
        for frame in identity_frames
    }
    accepted = []
    rejected = []
    shape = (int(layout.height), int(layout.width))
    mesh_config = InspectionIdentityMeshConfig(
        cell_size_pixels=int(config.identity_mesh_cell_size_pixels),
        maximum_fill_distance_pixels=float(
            config.identity_mesh_maximum_fill_distance_pixels
        ),
        minimum_depth_mm=float(config.minimum_depth_mm),
        maximum_depth_mm=float(config.maximum_depth_mm),
        minimum_jacobian=float(config.depth_mesh_min_jacobian),
        maximum_jacobian=float(config.depth_mesh_max_jacobian),
    )
    for owner in owners:
        try:
            composite_inspection_identity_owners(
                owners=(owner,),
                sources_by_frame_id=sources,
                layout=layout,
                intrinsics=intrinsics,
                output_image=np.zeros((*shape, 3), dtype=np.uint8),
                output_depth=np.full(shape, np.inf, dtype=np.float32),
                output_confidence=np.zeros(shape, dtype=np.float32),
                output_owner=np.full(shape, -1, dtype=np.int32),
                output_reliable_depth=np.zeros(shape, dtype=bool),
                output_overlay_mask=np.zeros(shape, dtype=bool),
                config=mesh_config,
            )
        except RuntimeError as error:
            rejected.append(
                {
                    "track_id": owner.identity_track_id,
                    "frame_id": int(owner.frame_id),
                    "reason": str(error),
                }
            )
        else:
            accepted.append(owner)
    return tuple(accepted), rejected


def _baseline_split_direct_owners(
    owners,
    *,
    baseline_owner_path,
    crop,
):
    baseline = cv2.imread(
        str(baseline_owner_path), cv2.IMREAD_UNCHANGED
    )
    if baseline is None or baseline.ndim != 2:
        raise RuntimeError("Baseline inspection owner image is unavailable")
    x0 = int(crop["x"])
    y0 = int(crop["y"])
    width = int(crop["width"])
    height = int(crop["height"])
    if baseline.shape != (height, width):
        raise RuntimeError("Baseline inspection owner/crop shape mismatch")
    accepted = []
    audits = []
    for owner in owners:
        footprint = np.asarray(owner.target_footprint, dtype=bool)
        local = footprint[y0 : y0 + height, x0 : x0 + width]
        values, counts = np.unique(baseline[local], return_counts=True)
        rows = [
            {
                "baseline_owner_frame_id": int(value) - 1,
                "pixel_count": int(count),
                "ratio": float(count / max(1, np.sum(counts))),
            }
            for value, count in zip(values, counts, strict=True)
        ]
        split = len(rows) >= 2
        audits.append(
            {
                "track_id": owner.identity_track_id,
                "source_frame_id": int(owner.frame_id),
                "baseline_owner_count": len(rows),
                "baseline_owners": rows,
                "crosses_existing_owner_boundary": split,
                "accepted": split,
                "reason": (
                    "baseline_graphcut_splits_true_rgbd_object_footprint"
                    if split
                    else "single_baseline_owner_needs_no_generic_handoff"
                ),
            }
        )
        if split:
            accepted.append(owner)
    return tuple(accepted), audits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("formal_output", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_output = args.formal_output.expanduser().resolve()
    labels = args.labels.expanduser().resolve()
    session = load_rgbd_session(args.session.expanduser().resolve())
    report = json.loads(
        (source_output / "report.json").read_text(encoding="utf-8")
    )
    transforms = json.loads(
        (source_output / "transforms.json").read_text(encoding="utf-8")
    )
    ocr_audit = json.loads(
        (
            source_output / "diagnostic_waveshare_ocr_rgbd_audit.json"
        ).read_text(encoding="utf-8")
    )
    node_rows = sorted(
        transforms["nodes"], key=lambda item: int(item["node_id"])
    )
    frame_by_id = {
        int(frame.frame_id): frame for frame in session.frames
    }
    frames = [frame_by_id[int(item["node_id"])] for item in node_rows]
    poses = [
        np.asarray(item["camera_to_world"], dtype=np.float64)
        for item in node_rows
    ]
    pose_by_frame = {
        int(frame.frame_id): pose
        for frame, pose in zip(frames, poses, strict=True)
    }
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    layout = estimate_inspection_layout(
        frames, poses, session.calibration, config=config
    )
    selected_rows = list(report["render"]["selected_panel_sources"])
    selected_by_frame = {
        int(item["frame_id"]): (
            int(item["panel_index"]),
            int(item["source_position"]),
        )
        for item in selected_rows
    }
    ocr_by_frame = {
        int(item["frame_id"]): item["target_detections"][0]
        for item in ocr_audit["frame_audits"]
        if item["target_detections"]
    }
    maps = _undistortion_maps(session.calibration)
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    tracking_inputs = []
    for frame in frames:
        frame_id = int(frame.frame_id)
        image, depth, geometric_valid = _read_rgbd(
            frame, session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= config.minimum_depth_mm)
            & (depth <= config.maximum_depth_mm)
        )
        cache[frame_id] = (image, depth, reliable)
        proposals = parse_fastsam_polygons(
            labels / f"{frame_id:08d}.txt",
            width=session.calibration.width,
            height=session.calibration.height,
        )
        tracking_inputs.append(
            FastSAMDISFrameInput(
                frame_id=frame_id,
                image_bgr=image,
                depth_mm=depth,
                camera_to_world=pose_by_frame[frame_id],
                proposals=proposals,
                geometric_valid=geometric_valid,
            )
        )
    tracking = track_fastsam_dis_frames(
        tracking_inputs,
        intrinsics=session.calibration,
        reference_depth_mm=layout.reference_depth_mm,
        stable_frame_ids=tuple(selected_by_frame),
    )

    identity_frames = []
    seeded_panels = []
    for frame_id, (panel_index, source_position) in selected_by_frame.items():
        if frame_id not in cache:
            continue
        image, depth, reliable = cache[frame_id]
        corner_x, _, _, local_valid, _ = _reference_panel_inverse_maps(
            source_pose=pose_by_frame[frame_id],
            panel_index=panel_index,
            layout=layout,
            intrinsics=session.calibration,
        )
        panel_valid = np.zeros(
            (layout.height, layout.width), dtype=bool
        )
        panel_valid[
            :, corner_x : corner_x + local_valid.shape[1]
        ] = local_valid
        identity_frames.append(
            InspectionIdentityOwnerFrame(
                panel_index=panel_index,
                source_index=source_position,
                frame_id=frame_id,
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=pose_by_frame[frame_id],
                panel_valid_mask=panel_valid,
            )
        )
        ocr = ocr_by_frame.get(frame_id)
        if ocr is not None:
            panel, _ = extract_ocr_seeded_panel(
                frame_id=frame_id,
                source_index=source_position,
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=reliable,
                ocr_polygon_xy=np.asarray(
                    ocr["ocr_polygon_xy"], dtype=np.float32
                ),
                camera_to_world=pose_by_frame[frame_id],
                intrinsics=session.calibration,
            )
            if panel is not None:
                seeded_panels.append(panel)
    plan = plan_inspection_identity_owner_intervals(
        frames=identity_frames,
        tracking=tracking,
        layout=layout,
        intrinsics=session.calibration,
        ocr_seeded_panels=seeded_panels,
    )
    if not plan.foreground_owners:
        raise RuntimeError(
            f"Identity owner planning failed: {plan.audit}"
        )
    direct_plan = plan_direct_stable_track_identity_owners(
        frames=identity_frames,
        tracking=tracking,
        layout=layout,
        intrinsics=session.calibration,
        existing_foreground_owners=plan.foreground_owners,
        config=DirectHandoffConfig(
            minimum_pair_target_iou=0.85,
        ),
    )
    feasible_direct_owners, mesh_rejections = (
        _mesh_feasible_direct_owners(
            direct_plan.foreground_owners,
            identity_frames=identity_frames,
            layout=layout,
            intrinsics=session.calibration,
            config=config,
        )
    )
    split_direct_owners, split_audits = (
        _baseline_split_direct_owners(
            feasible_direct_owners,
            baseline_owner_path=source_output / "inspection_owner.png",
            crop=report["render"]["crop"],
        )
    )
    combined_owners = (
        *plan.foreground_owners,
        *split_direct_owners,
    )
    result = render_inspection_multiview(
        frames,
        poses,
        session.calibration,
        config=config,
        foreground_identity_owners=combined_owners,
    )
    _atomic_image(output / "diagnostic_panorama.png", result.image_bgr)
    _atomic_image(
        output / "diagnostic_owner.png",
        np.asarray(result.owner_frame_id + 1, dtype=np.uint16),
    )
    _atomic_json(
        output / "diagnostic_report.json",
        {
            "schema": "inspection-identity-owner-replay/v1",
            "diagnostic_only": True,
            "historical_model_evidence_used": True,
            "formal_delivery_modified": False,
            "post_render_overlay_used": False,
            "rgb_translation_affine_or_alpha_used": False,
            "planner": plan.audit,
            "direct_stable_track_planner": direct_plan.audit,
            "direct_identity_mesh_rejections": mesh_rejections,
            "direct_baseline_owner_split_audits": split_audits,
            "foreground_identity_owner_count": len(combined_owners),
            "renderer": result.metadata,
        },
    )
    print(output / "diagnostic_panorama.png")


if __name__ == "__main__":
    main()
