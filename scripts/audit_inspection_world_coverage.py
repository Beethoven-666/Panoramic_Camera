"""Audit an existing formal inspection output against its original RGB-D scan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.inspection_multiview import (
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
)
from panorama_demo.inspection_world_coverage import (
    audit_inspection_world_coverage,
)
from panorama_demo.session import load_rgbd_session


def _layout_from_report(value: dict[str, object]) -> InspectionMultiviewLayout:
    panels = tuple(
        VirtualPerspectivePanel(
            panel_index=int(item["panel_index"]),
            anchor_scan_mm=float(item["anchor_scan_mm"]),
            canvas_offset_x=float(item["canvas_offset_x"]),
            center_world_mm=tuple(
                float(component) for component in item["center_world_mm"]
            ),
        )
        for item in value["panels"]
    )
    return InspectionMultiviewLayout(
        width=int(value["width"]),
        height=int(value["height"]),
        reference_depth_mm=float(value["reference_depth_mm"]),
        scan_axis=tuple(float(item) for item in value["scan_axis_world"]),
        down_axis=tuple(float(item) for item in value["down_axis_world"]),
        normal_axis=tuple(float(item) for item in value["normal_axis_world"]),
        panels=panels,
        panel_step_mm=float(value["panel_step_mm"]),
        canvas_megapixels=float(value["canvas_megapixels"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("formal_output")
    arguments = parser.parse_args()
    session_path = Path(arguments.session).expanduser().resolve()
    output = Path(arguments.formal_output).expanduser().resolve()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    transforms = json.loads(
        (output / "transforms.json").read_text(encoding="utf-8")
    )
    session = load_rgbd_session(session_path)
    frame_ids = [int(item) for item in report["render"]["frame_ids"]]
    frame_by_id = {int(frame.frame_id): frame for frame in session.frames}
    frames = [frame_by_id[frame_id] for frame_id in frame_ids]
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    poses = [pose_by_id[frame_id] for frame_id in frame_ids]
    layout = _layout_from_report(report["render"]["layout"])
    encoded_owner = cv2.imread(
        str(output / "inspection_owner.png"), cv2.IMREAD_UNCHANGED
    )
    if encoded_owner is None or encoded_owner.dtype != np.uint16:
        raise RuntimeError("Formal inspection owner PNG is missing or invalid")
    owner = encoded_owner.astype(np.int32) - 1
    crop = report["render"]["crop"]
    started = time.perf_counter()
    audit = audit_inspection_world_coverage(
        frames=frames,
        poses=poses,
        intrinsics=session.calibration,
        layout=layout,
        owner_frame_id=owner,
        crop_xywh=(
            crop["x"],
            crop["y"],
            crop["width"],
            crop["height"],
        ),
        selected_panel_sources=report["render"]["selected_panel_sources"],
    )
    audit["elapsed_seconds"] = time.perf_counter() - started
    audit_path = output / "inspection_world_coverage_diagnostic.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image = cv2.imread(str(output / "mosaic_inspection.png"), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != owner.shape:
        raise RuntimeError("Formal inspection RGB is missing or misaligned")
    overlay = image.copy()
    crop_x, crop_y = int(crop["x"]), int(crop["y"])
    for item in audit["low_coverage_cells"]:
        bbox = item["full_canvas_bbox_xywh"]
        if bbox is None:
            continue
        x, y, width, height = (int(value) for value in bbox)
        x0 = max(0, x - crop_x)
        y0 = max(0, y - crop_y)
        x1 = min(overlay.shape[1], x + width - crop_x)
        y1 = min(overlay.shape[0], y + height - crop_y)
        if x1 <= x0 or y1 <= y0:
            continue
        ratio = float(item["coverage_ratio"])
        colour = (0, int(round(255.0 * ratio)), 255)
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), colour, 1)
    blended = cv2.addWeighted(image, 0.72, overlay, 0.28, 0.0)
    overlay_path = output / "inspection_world_coverage_diagnostic.png"
    if not cv2.imwrite(str(overlay_path), blended):
        raise RuntimeError("Could not write inspection coverage overlay")
    print(audit_path)
    print(overlay_path)
    print(
        json.dumps(
            {
                "elapsed_seconds": audit["elapsed_seconds"],
                "observed_world_coverage_ratio": audit[
                    "observed_world_coverage_ratio"
                ],
                "multiview_world_coverage_ratio": audit[
                    "multiview_world_coverage_ratio"
                ],
                "low_coverage_cell_count": audit[
                    "low_coverage_cell_count"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
