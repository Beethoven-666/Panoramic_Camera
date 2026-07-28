"""Replay a solved trajectory with the RGB-D world-object anchor enabled.

This is an isolated diagnostic.  It never writes formal delivery files and
does not run or replace Open3D/ORB-SLAM3 trajectory estimation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    render_inspection_multiview,
)
from panorama_demo.session import load_rgbd_session


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    pending = path.with_name(f".{path.name}.pending")
    pending.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(pending, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("transforms", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/demo.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    session = load_rgbd_session(arguments.session.resolve())
    transforms = json.loads(
        arguments.transforms.resolve().read_text(encoding="utf-8")
    )
    if transforms.get("translation_unit") != "mm":
        raise RuntimeError("Saved trajectory must use millimetres")
    nodes = transforms.get("nodes")
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise RuntimeError("Saved trajectory has too few pose nodes")
    frame_by_id = {int(frame.frame_id): frame for frame in session.frames}
    frame_ids = [int(node["node_id"]) for node in nodes]
    frames = [frame_by_id[frame_id] for frame_id in frame_ids]
    poses = [
        np.asarray(node["camera_to_world"], dtype=np.float64)
        for node in nodes
    ]
    payload = yaml.safe_load(
        arguments.config.resolve().read_text(encoding="utf-8")
    )
    base = InspectionMultiviewConfig.from_mapping(
        payload["stitch"]["inspection_multiview"]
    )
    diagnostic_config = replace(
        base,
        foreground_world_anchor_enabled=True,
    )
    started = time.perf_counter()
    result = render_inspection_multiview(
        frames,
        poses,
        session.calibration,
        config=diagnostic_config,
    )
    elapsed = time.perf_counter() - started

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / "diagnostic_panorama.png"
    owner_path = output / "diagnostic_owner.png"
    if not cv2.imwrite(str(image_path), result.image_bgr):
        raise RuntimeError("Could not write object-anchor diagnostic panorama")
    encoded_owner = np.where(
        result.owner_frame_id >= 0,
        result.owner_frame_id + 1,
        0,
    ).astype(np.uint16)
    if not cv2.imwrite(str(owner_path), encoded_owner):
        raise RuntimeError("Could not write object-anchor owner raster")
    audit = {
        "schema": "inspection-world-object-anchor-diagnostic/v1",
        "formal_publication": False,
        "saved_real_pose_replay": True,
        "pose_interpolation_count": 0,
        "frame_count": len(frames),
        "elapsed_seconds": elapsed,
        "renderer": result.metadata,
        "files": {
            "panorama": image_path.name,
            "owner": owner_path.name,
        },
    }
    _atomic_json(output / "diagnostic_report.json", audit)
    print(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "track_count": result.metadata[
                    "foreground_component_assignment"
                ]["object_world_anchor"]["track_count"],
                "visible_pixel_count": result.metadata[
                    "foreground_component_assignment"
                ]["object_world_anchor"]["visible_pixel_count"],
                "strict_v1_inspection_complete": result.metadata[
                    "strict_v1_inspection_complete"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
