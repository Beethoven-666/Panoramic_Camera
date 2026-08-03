"""Atomic independent publication for a video two-dimensional delivery."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIDEO_DELIVERY_SCHEMA = "gemini305-video-panorama-delivery/v1"
VIDEO_REPORT_SCHEMA = "gemini305-video-panorama-report/v1"


def invalidate_video_delivery(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "video_delivery.json", "video_report.json", "video_failure.json",
        "video_panorama.jpg", "video_panorama.png", "video_pixel_provenance.npz",
        "central_strips", ".central_strips.pending",
        "central_strips_owner_only", ".central_strips_owner_only.pending",
    ):
        path = output / name
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)


def write_video_failure(output: Path, input_path: Path, exc: Exception) -> None:
    invalidate_video_delivery(output)
    pending = output / ".video_failure.pending.json"
    pending.write_text(json.dumps({"schema": "gemini305-video-panorama-failure/v1", "input": str(input_path), "error_type": type(exc).__name__, "message": str(exc), "deliverable_published": False}, indent=2), encoding="utf-8")
    os.replace(pending, output / "video_failure.json")


def publish_video_2d(
    output: Path,
    panorama: np.ndarray,
    owner: np.ndarray,
    report: dict[str, Any],
    *,
    pending_central_strips: Path | None = None,
    pending_central_strips_owner_only: Path | None = None,
) -> dict[str, Any]:
    if panorama.dtype != np.uint8 or panorama.ndim != 3 or panorama.shape[2] != 3:
        raise ValueError("Video panorama must be an 8-bit BGR image")
    if owner.shape != panorama.shape[:2]:
        raise ValueError("Video owner map shape does not match panorama")
    if pending_central_strips is not None:
        export = report.get("central_strip_export")
        if not isinstance(export, dict) or not pending_central_strips.is_dir():
            raise RuntimeError("Video delivery lacks staged central-strip images")
        export["path"] = str(output / "central_strips")
    if pending_central_strips_owner_only is not None:
        owner_only_export = report.get("central_strip_owner_only_export")
        if not isinstance(owner_only_export, dict) or not pending_central_strips_owner_only.is_dir():
            raise RuntimeError("Video delivery lacks staged owner-only central-strip images")
        owner_only_export["path"] = str(output / "central_strips_owner_only")
    pending_jpg, pending_png = output / ".video_panorama.pending.jpg", output / ".video_panorama.pending.png"
    if not cv2.imwrite(str(pending_jpg), panorama, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError("Could not encode video panorama JPEG")
    if not cv2.imwrite(str(pending_png), panorama):
        raise OSError("Could not encode video panorama PNG")
    pending_prov = output / ".video_provenance.pending.npz"
    np.savez_compressed(pending_prov, owner_frame_id=owner.astype(np.int32, copy=False))
    report["schema"] = VIDEO_REPORT_SCHEMA
    report["panorama"] = str(output / "video_panorama.jpg")
    report["provenance"] = str(output / "video_pixel_provenance.npz")
    report_path = output / ".video_report.pending.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    delivery = {"schema": VIDEO_DELIVERY_SCHEMA, "delivery_state": report["delivery_state"], "quality_grade": report["quality_grade"], "manual_review_required": report["manual_review_required"], "report": "video_report.json"}
    if pending_central_strips is not None:
        delivery["central_strip_export"] = report["central_strip_export"]
    if pending_central_strips_owner_only is not None:
        delivery["central_strip_owner_only_export"] = report["central_strip_owner_only_export"]
    delivery_path = output / ".video_delivery.pending.json"
    delivery_path.write_text(json.dumps(delivery, indent=2), encoding="utf-8")
    os.replace(pending_jpg, output / "video_panorama.jpg")
    os.replace(pending_png, output / "video_panorama.png")
    os.replace(pending_prov, output / "video_pixel_provenance.npz")
    if pending_central_strips is not None:
        os.replace(pending_central_strips, output / "central_strips")
    if pending_central_strips_owner_only is not None:
        os.replace(pending_central_strips_owner_only, output / "central_strips_owner_only")
    os.replace(report_path, output / "video_report.json")
    os.replace(delivery_path, output / "video_delivery.json")
    return report
