from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.session import load_rgbd_session
from panorama_demo.world_surface_inspection import (
    WorldSurfaceInspectionConfig,
    colourise_owner,
    render_automatic_instance_candidates,
    render_world_surface_inspection,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the isolated all-view RGB-D world-surface prototype."
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("transforms", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cell-size",
        type=int,
        default=8,
        help="Source depth-mesh cell size in pixels (prototype default: 8).",
    )
    parser.add_argument(
        "--v9-inspection-meta",
        type=Path,
        help=(
            "Enable the isolated automatic-instance experiment using this "
            "read-only v9 inspection_meta.json and its sibling panorama."
        ),
    )
    return parser.parse_args()


def _load_poses(path: Path) -> tuple[list[int], list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("translation_unit") != "mm":
        raise ValueError("World-surface transforms must use millimetres")
    if not str(payload.get("pose_convention", "")).startswith("camera_to_world"):
        raise ValueError("World-surface transforms must be camera_to_world")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError("World-surface transforms contain too few nodes")
    frame_ids = [int(node["node_id"]) for node in nodes]
    poses = [
        np.asarray(node["camera_to_world"], dtype=np.float64) for node in nodes
    ]
    return frame_ids, poses


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)


def main() -> int:
    args = _arguments()
    session = load_rgbd_session(args.session)
    frame_ids, poses = _load_poses(args.transforms)
    frames_by_id = {int(frame.frame_id): frame for frame in session.frames}
    missing = sorted(set(frame_ids) - set(frames_by_id))
    if missing:
        raise ValueError(f"Session is missing transform frames: {missing}")
    frames = [frames_by_id[frame_id] for frame_id in frame_ids]
    started = time.perf_counter()
    result = render_world_surface_inspection(
        frames,
        poses,
        session.calibration,
        config=WorldSurfaceInspectionConfig(
            depth_mesh_cell_size_pixels=int(args.cell_size)
        ),
    )
    elapsed = time.perf_counter() - started
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "world_surface_inspection.png"
    owner_path = output / "world_surface_owner.png"
    valid_path = output / "world_surface_valid.png"
    depth_path = output / "world_surface_target_depth_mm.npy"
    owner_id_path = output / "world_surface_owner_frame_id.npy"
    component_image_path = output / "component_locked_inspection.png"
    component_owner_path = output / "component_locked_owner.png"
    component_label_path = output / "component_labels.png"
    if not cv2.imwrite(str(image_path), result.image_bgr):
        raise RuntimeError("Could not write world-surface inspection image")
    if not cv2.imwrite(str(owner_path), colourise_owner(result.owner_frame_id)):
        raise RuntimeError("Could not write world-surface owner diagnostic")
    if not cv2.imwrite(
        str(valid_path), result.valid_mask.astype(np.uint8) * 255
    ):
        raise RuntimeError("Could not write world-surface valid mask")
    np.save(depth_path, result.target_depth_mm, allow_pickle=False)
    np.save(owner_id_path, result.owner_frame_id, allow_pickle=False)
    if not cv2.imwrite(
        str(component_image_path), result.component_locked_image_bgr
    ):
        raise RuntimeError("Could not write component-locked inspection image")
    if not cv2.imwrite(
        str(component_owner_path),
        colourise_owner(result.component_locked_owner_frame_id),
    ):
        raise RuntimeError("Could not write component-locked owner image")
    if not cv2.imwrite(
        str(component_label_path),
        colourise_owner(
            np.where(result.component_label > 0, result.component_label, -1)
        ),
    ):
        raise RuntimeError("Could not write component label image")
    metadata = dict(result.metadata)
    metadata["elapsed_seconds"] = float(elapsed)
    metadata["artifacts"] = {
        "image": image_path.name,
        "owner": owner_path.name,
        "valid": valid_path.name,
        "target_depth_mm": depth_path.name,
        "owner_frame_id": owner_id_path.name,
        "component_locked_image": component_image_path.name,
        "component_locked_owner": component_owner_path.name,
        "component_labels": component_label_path.name,
    }
    if args.v9_inspection_meta is not None:
        v9_meta_path = args.v9_inspection_meta.resolve()
        v9_payload = json.loads(v9_meta_path.read_text(encoding="utf-8"))
        renderer = v9_payload.get("renderer")
        if not isinstance(renderer, dict):
            raise ValueError("v9 inspection metadata omitted renderer")
        v9_image_path = v9_meta_path.parent / "mosaic_inspection.png"
        v9_image = cv2.imread(str(v9_image_path), cv2.IMREAD_COLOR)
        if v9_image is None:
            raise ValueError(
                f"Could not decode sibling v9 panorama: {v9_image_path}"
            )
        automatic_started = time.perf_counter()
        automatic = render_automatic_instance_candidates(
            frames,
            poses,
            session.calibration,
            v9_renderer_metadata=renderer,
            v9_inspection_image_bgr=v9_image,
        )
        automatic_elapsed = time.perf_counter() - automatic_started
        automatic_image_path = output / "automatic_instance_inspection.png"
        automatic_owner_path = output / "automatic_instance_owner.png"
        automatic_label_path = output / "automatic_instance_labels.png"
        if not cv2.imwrite(
            str(automatic_image_path), automatic.image_bgr
        ):
            raise RuntimeError("Could not write automatic-instance panorama")
        if not cv2.imwrite(
            str(automatic_owner_path),
            colourise_owner(automatic.owner_frame_id),
        ):
            raise RuntimeError("Could not write automatic-instance owners")
        if not cv2.imwrite(
            str(automatic_label_path),
            colourise_owner(
                np.where(automatic.instance_label > 0, automatic.instance_label, -1)
            ),
        ):
            raise RuntimeError("Could not write automatic-instance labels")
        automatic_metadata = dict(automatic.metadata)
        automatic_metadata["elapsed_seconds"] = float(automatic_elapsed)
        automatic_metadata["v9_inspection_meta"] = str(v9_meta_path)
        automatic_metadata["v9_inspection_image"] = str(v9_image_path)
        automatic_metadata["artifacts"] = {
            "image": automatic_image_path.name,
            "owner": automatic_owner_path.name,
            "labels": automatic_label_path.name,
        }
        automatic_meta_path = output / "automatic_instance_meta.json"
        _atomic_write_json(automatic_meta_path, automatic_metadata)
        metadata["automatic_instance_prototype"] = {
            "enabled": True,
            "elapsed_seconds": float(automatic_elapsed),
            "accepted_instance_count": int(
                automatic.metadata["accepted_instance_count"]
            ),
            "metadata": automatic_meta_path.name,
            "image": automatic_image_path.name,
            "owner": automatic_owner_path.name,
            "labels": automatic_label_path.name,
        }
    _atomic_write_json(output / "world_surface_meta.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
