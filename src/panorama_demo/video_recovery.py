"""Create a separately identified, closed session from a durable capture prefix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
from typing import Any

from .commit_journal import read_durable_prefix
from .sdk_state import atomic_json


def salvage_prefix(source: Path, destination: Path) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    source, destination = source.resolve(), destination.resolve()
    rows, checkpoint = read_durable_prefix(source)
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    # Never modify the capture's original shutdown or qualification evidence.
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / "calibration.json", destination / "calibration.json")
    for row in rows:
        for field in ("color_path", "aligned_depth_path"):
            relative = Path(row[field])
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, target)
    with (destination / "frames.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]["row"]))
        writer.writeheader()
        for item in rows:
            row = dict(item["row"])
            # Optional raw depth is not part of the durable RGB-D contract.
            if "raw_depth_path" in row:
                row["raw_depth_path"] = ""
            writer.writerow(row)
    regression = int(original.get("timestamp_regressions", 0)) != 0
    recovery = {"schema": "gemini305-sdk-prefix-recovery/v1", "source_session": str(source),
                "original_clean_shutdown": original.get("clean_shutdown"),
                "original_capture_error": original.get("capture_error"),
                "original_writer_errors": original.get("writer_errors", []),
                "original_timestamp_regressions": original.get("timestamp_regressions", 0),
                "committed_frame_count": len(rows),
                "truncated_frame_count": max(0, int(original.get("received_frames", len(rows))) - len(rows)),
                "checkpoint": checkpoint, "manual_review_required": True,
                "formal_eligible": not regression}
    manifest = {**original, "schema": "panorama-demo-session/v2", "clean_shutdown": True,
                "received_frames": len(rows), "written_frames": len(rows), "write_errors": 0,
                "writer_errors": [], "queue_drops": 0, "diagnostic_only": True,
                "formal_stitch_allowed": False, "recovery": recovery,
                "product_eligibility": {"photo_panorama": False, "video_panorama": not regression}}
    # This clean flag describes completion of this derived disk-only session.
    # It does not claim a successful physical shutdown of the source capture.
    manifest.pop("capture_error", None)
    atomic_json(destination / "manifest.json", manifest, durable=True)
    atomic_json(destination / "recovery.json", recovery, durable=True)
    return destination, rows, recovery
