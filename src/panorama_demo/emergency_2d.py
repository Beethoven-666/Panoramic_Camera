"""Explicit non-production RGB evidence for a safe but insufficient prefix."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .commit_journal import file_sha256
from .sdk_state import atomic_json


def publish_emergency_2d(session: Path, rows: Sequence[dict[str, Any]], output: Path,
                         *, reason: str) -> dict[str, Any]:
    session, output = session.resolve(), output.resolve()
    if (output / "video_delivery.json").exists() or (output / "emergency_delivery.json").exists():
        raise FileExistsError("Emergency output cannot replace an existing delivery")
    candidates = []
    for row in rows:
        color = (session / row["color_path"]).resolve()
        color.relative_to(session)
        if file_sha256(color) != row["color_sha256"]:
            continue
        image = cv2.imread(str(color), cv2.IMREAD_COLOR)
        if image is None:
            continue
        depth_path = (session / row["aligned_depth_path"]).resolve()
        depth_path.relative_to(session)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        valid_depth = (0.0 if depth is None or depth.dtype != np.uint16 or depth.shape != image.shape[:2]
                       else float(np.mean(depth > 0)))
        sharpness = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        exposure = row["row"].get("color_exposure", 0)
        try:
            exposure_valid = float(exposure) > 0
        except (ValueError, TypeError):
            exposure_valid = False
        candidates.append((row, image.shape, (valid_depth, sharpness, exposure_valid, -int(row["frame_id"]))))
    if not candidates:
        raise ValueError("FAILED_NO_VALID_FRAME")
    selected = max(candidates, key=lambda item: item[2])
    image = cv2.imread(str(session / selected[0]["color_path"]), cv2.IMREAD_COLOR)
    owner = np.full(image.shape[:2], selected[0]["frame_id"], dtype=np.int32)
    mode = "best_complete_single_frame"
    motion_audit: dict[str, Any] = {"reliable_monotonic": False}
    # The emergency renderer is independent of the production renderer. Only
    # measured, reliable adjacent motion can authorize its narrow strips.
    if len(candidates) >= 2 and len({item[1] for item in candidates}) == 1:
        from .video_s13_motion import measure_s13_motion
        from .video_s13_session import S13RenderFrame

        frames = tuple(S13RenderFrame(frame_id=int(row["frame_id"]),
                                      color_path=session / row["color_path"],
                                      timestamp_us=int(row["timestamp_us"]),
                                      width=shape[1], height=shape[0])
                       for row, shape, _ in candidates)
        edges = [edge for left, right in zip(frames, frames[1:])
                 for edge in measure_s13_motion((left, right), steps=(1,))]
        advances = [edge.selected_advance_px for edge in edges]
        reliable = (len(edges) == len(candidates) - 1
                    and all(not edge.risk for edge in edges)
                    and all(value is not None and np.isfinite(value) and
                            0.25 <= abs(value) <= image.shape[1] * 0.20 for value in advances))
        if reliable and (all(value > 0 for value in advances) or all(value < 0 for value in advances)):
            widths = [max(1, min(image.shape[1] // 10, round(abs(value)))) for value in advances]
            widths.append(widths[-1])
            pieces, owners = [], []
            ordering = list(zip(candidates, widths))
            # The shared estimator already reports signed camera advance.
            if advances[0] < 0:
                ordering.reverse()
            for (row, _, _), width in ordering:
                rgb = cv2.imread(str(session / row["color_path"]), cv2.IMREAD_COLOR)
                left = (rgb.shape[1] - width) // 2
                pieces.append(rgb[:, left:left + width])
                owners.append(np.full((rgb.shape[0], width), row["frame_id"], np.int32))
            image, owner = np.concatenate(pieces, axis=1), np.concatenate(owners, axis=1)
            mode = "measured_central_narrow_strips"
            motion_audit = {"reliable_monotonic": True, "adjacent_advance_px": advances}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "emergency_panorama.png"
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError("Emergency PNG encoding failed")
    path.write_bytes(encoded.tobytes())
    provenance = output / "emergency_provenance.npz"
    np.savez_compressed(provenance, owner_frame_id=owner, valid=np.ones(owner.shape, dtype=bool))
    report = {"schema": "gemini305-sdk-emergency-2d/v1", "formal_2d_published": False,
              "authority": "committed_rgb_emergency_only", "mode": mode, "reason": reason,
              "manual_review_required": True, "source_frame_ids": np.unique(owner).tolist(),
              "motion": motion_audit, "completion_state": "EMERGENCY_2D_PUBLISHED",
              "emergency_2d_path": str(path), "provenance_path": str(provenance),
              "png_sha256": hashlib.sha256(encoded.tobytes()).hexdigest()}
    atomic_json(output / "emergency_report.json", report)
    atomic_json(output / "emergency_delivery.json", report, durable=True)
    return report
