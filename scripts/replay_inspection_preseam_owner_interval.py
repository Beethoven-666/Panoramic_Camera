"""Replay one audited pre-seam owner interval without changing RGB geometry.

This is a diagnostic bridge for validating the renderer integration.  It
consumes a previously generated object-rich corridor audit, reconstructs the
same immutable RGB-D/pose input, and asks the formal inspection renderer to
keep that entire row-contiguous interval owned by the audited real panel.
It never overlays, translates, fills, or synthesizes RGB.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    InspectionPreSeamHardOwnerInterval,
    estimate_inspection_layout,
    render_inspection_multiview,
)
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


def _interval_masks(
    selected: dict[str, object],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    projections = list(selected["direct_specific_panel_projections"])
    spans = [
        tuple(float(value) for value in item["x_span_pixels"])
        for item in projections
    ]
    y_spans = [
        tuple(float(value) for value in item["y_span_pixels"])
        for item in projections
    ]
    height, width = shape
    footprint = np.zeros(shape, dtype=bool)
    for x_span, y_span in zip(spans, y_spans, strict=True):
        x0 = max(0, int(np.floor(min(x_span))))
        x1 = min(width, int(np.ceil(max(x_span))) + 1)
        y0 = max(0, int(np.floor(min(y_span))))
        y1 = min(height, int(np.ceil(max(y_span))) + 1)
        footprint[y0:y1, x0:x1] = True
    rows = np.flatnonzero(np.any(footprint, axis=1))
    if rows.size == 0:
        raise RuntimeError("Audited interval has no projected footprint")
    x0 = max(0, int(np.floor(min(min(item) for item in spans))))
    x1 = min(width, int(np.ceil(max(max(item) for item in spans))) + 1)
    lock = np.zeros(shape, dtype=bool)
    lock[int(rows[0]) : int(rows[-1]) + 1, x0:x1] = True
    return lock, footprint


def _structure_masks(
    selected: dict[str, object],
    shape: tuple[int, int],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    height, width = shape
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for projection in selected["direct_specific_panel_projections"]:
        x_span = tuple(float(value) for value in projection["x_span_pixels"])
        y_span = tuple(float(value) for value in projection["y_span_pixels"])
        x0 = max(0, int(np.floor(min(x_span))))
        x1 = min(width, int(np.ceil(max(x_span))) + 1)
        y0 = max(0, int(np.floor(min(y_span))))
        y1 = min(height, int(np.ceil(max(y_span))) + 1)
        footprint = np.zeros(shape, dtype=bool)
        footprint[y0:y1, x0:x1] = True
        result.append((footprint.copy(), footprint))
    return tuple(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("formal_output", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--separate-structures",
        action="store_true",
        help="Lock each audited structure independently to the same panel.",
    )
    parser.add_argument(
        "--structure-index",
        type=int,
        choices=(0, 1, 2),
        help="Diagnostic: lock only one of the three audited structures.",
    )
    parser.add_argument(
        "--no-interval",
        action="store_true",
        help="Replay only the renderer's RGB-D foreground component owner path.",
    )
    args = parser.parse_args()
    source_output = args.formal_output.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    session = load_rgbd_session(args.session.expanduser().resolve())
    report = json.loads(
        (source_output / "report.json").read_text(encoding="utf-8")
    )
    transforms = json.loads(
        (source_output / "transforms.json").read_text(encoding="utf-8")
    )
    corridor = json.loads(
        (
            source_output
            / "diagnostic_waveshare_object_rich_corridor_audit.json"
        ).read_text(encoding="utf-8")
    )
    selected = corridor.get("automatically_selected_source")
    if not isinstance(selected, dict) or selected.get("pass") is not True:
        raise RuntimeError("Object-rich corridor audit has no accepted source")

    frame_by_id = {
        int(frame.frame_id): frame for frame in session.frames
    }
    node_rows = sorted(
        transforms["nodes"], key=lambda item: int(item["node_id"])
    )
    frames = [frame_by_id[int(item["node_id"])] for item in node_rows]
    poses = [
        np.asarray(item["camera_to_world"], dtype=np.float64)
        for item in node_rows
    ]
    config = InspectionMultiviewConfig.from_mapping(
        report["render"]["config"]
    )
    layout = estimate_inspection_layout(
        frames, poses, session.calibration, config=config
    )
    if args.no_interval:
        mask_pairs = ()
    elif args.separate_structures:
        mask_pairs = _structure_masks(
            selected, (layout.height, layout.width)
        )
        if args.structure_index is not None:
            mask_pairs = (mask_pairs[args.structure_index],)
    else:
        mask_pairs = (
            _interval_masks(selected, (layout.height, layout.width)),
        )
    intervals = tuple(
        InspectionPreSeamHardOwnerInterval(
            track_id=-(index + 1),
            panel_index=int(selected["panel_index"]),
            frame_id=int(selected["frame_id"]),
            lock_mask=lock,
            union_footprint=footprint,
        )
        for index, (lock, footprint) in enumerate(mask_pairs)
    )
    result = render_inspection_multiview(
        frames,
        poses,
        session.calibration,
        config=config,
        pre_seam_hard_owner_intervals=intervals,
    )
    _atomic_image(output / "diagnostic_panorama.png", result.image_bgr)
    owner = np.asarray(result.owner_frame_id, dtype=np.int32)
    _atomic_image(
        output / "diagnostic_owner.png",
        np.asarray(owner + 1, dtype=np.uint16),
    )
    _atomic_json(
        output / "diagnostic_report.json",
        {
            "schema": "inspection-preseam-owner-replay/v1",
            "diagnostic_only": True,
            "post_render_overlay_used": False,
            "rgb_translation_or_warp_used": False,
            "pose_or_depth_modified": False,
            "source_corridor_schema": corridor.get("schema"),
            "selected_source": selected,
            "pre_seam_interval_used": not args.no_interval,
            "renderer": result.metadata,
        },
    )
    print(output / "diagnostic_panorama.png")


if __name__ == "__main__":
    main()
