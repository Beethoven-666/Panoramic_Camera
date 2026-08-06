"""Immutable development/validation/holdout split for the approved video run."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Mapping, Sequence


SPLIT_DEFINITION = {
    "schema": "gemini305-video-split/v1",
    "progress_coordinate": "cumulative_reliable_horizontal_motion",
    "development": [[0.00, 0.30], [0.48, 0.68]],
    "validation": [[0.30, 0.48], [0.68, 0.84]],
    "holdout": [[0.84, 1.00]],
}

SOURCE_PROGRESS_SCHEMA = "gemini305-video-source-progress-evidence/v1"


def _frame_id(frame: object) -> int:
    """Return a real source identifier without creating a synthetic node."""

    value = getattr(frame, "frame_id", frame)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Source progress evidence requires non-negative integer real frame ids")
    return int(value)


def _finite_motion_value(motion: object, name: str) -> float:
    value = getattr(motion, name, None)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Source progress evidence requires finite motion.{name}")
    return float(value)


def build_source_progress_evidence(
    frames: Sequence[object], motions: Sequence[object]
) -> dict[str, object]:
    """Derive a serializable source-frame progress map from real scan edges.

    Progress is *only* cumulative reliable horizontal motion.  The helper
    neither estimates a pose nor fills a missing frame; unreliable edges are
    retained in the evidence but contribute zero progress.  A reliable edge
    which reverses the measured one-way scan is rejected rather than clipped
    or silently repaired.
    """

    ids = [_frame_id(frame) for frame in frames]
    if len(ids) < 2 or len(motions) != len(ids) - 1:
        raise ValueError("Source progress evidence requires aligned real frames and edges")
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("Source progress evidence requires unique chronological frame ids")

    reliable_dx: list[float] = []
    raw_edges: list[tuple[float, float, bool]] = []
    for motion in motions:
        dx = _finite_motion_value(motion, "dx")
        dy = _finite_motion_value(motion, "dy")
        reliable = getattr(motion, "reliable", None)
        if not isinstance(reliable, bool):
            raise ValueError("Source progress evidence requires an explicit motion.reliable flag")
        raw_edges.append((dx, dy, reliable))
        if reliable and abs(dx) > 1e-9:
            reliable_dx.append(dx)
    if not reliable_dx:
        raise ValueError("Source progress evidence requires reliable nonzero horizontal motion")
    direction = 1 if sorted(reliable_dx)[len(reliable_dx) // 2] > 0.0 else -1

    cumulative = [0.0]
    edge_rows: list[dict[str, object]] = []
    for index, (dx, dy, reliable) in enumerate(raw_edges):
        signed_horizontal = float(direction * dx)
        if reliable and signed_horizontal < -1e-9:
            raise ValueError("Reliable scan edge reverses the measured horizontal direction")
        increment = max(0.0, signed_horizontal) if reliable else 0.0
        cumulative.append(cumulative[-1] + increment)
        edge_rows.append(
            {
                "from_frame_id": ids[index],
                "to_frame_id": ids[index + 1],
                "dx": dx,
                "dy": dy,
                "reliable": reliable,
                "reliable_horizontal_increment": increment,
            }
        )
    total = cumulative[-1]
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Source progress evidence has no cumulative reliable horizontal motion")
    frame_rows = [
        {"frame_id": frame_id, "scan_progress": value / total}
        for frame_id, value in zip(ids, cumulative)
    ]
    canonical = json.dumps(
        {"direction": direction, "frames": frame_rows, "edges": edge_rows},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": SOURCE_PROGRESS_SCHEMA,
        "coordinate": "cumulative_reliable_horizontal_motion",
        "measurement_only": True,
        "source_frame_count": len(frame_rows),
        "direction": "increasing_frame_order_rightward" if direction > 0 else "increasing_frame_order_leftward",
        "content_sha256": hashlib.sha256(canonical).hexdigest(),
        "frames": frame_rows,
        "edges": edge_rows,
    }


def source_progress_by_frame(evidence: Mapping[str, object]) -> dict[int, float]:
    """Validate evidence and expose its exact real-frame progress mapping."""

    if evidence.get("schema") != SOURCE_PROGRESS_SCHEMA:
        raise ValueError("Unsupported source progress evidence schema")
    if evidence.get("coordinate") != "cumulative_reliable_horizontal_motion":
        raise ValueError("Source progress evidence has an unsupported coordinate")
    if evidence.get("measurement_only") is not True:
        raise ValueError("Source progress evidence must remain measurement-only")
    direction_name = evidence.get("direction")
    if direction_name not in {
        "increasing_frame_order_rightward",
        "increasing_frame_order_leftward",
    }:
        raise ValueError("Source progress evidence direction is invalid")
    rows = evidence.get("frames")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("Source progress evidence requires at least two frame rows")
    if evidence.get("source_frame_count") != len(rows):
        raise ValueError("Source progress evidence frame count is invalid")
    mapping: dict[int, float] = {}
    previous = -1.0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Source progress evidence frame row is invalid")
        frame_id, progress = row.get("frame_id"), row.get("scan_progress")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("Source progress evidence frame id is invalid")
        if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(progress):
            raise ValueError("Source progress evidence progress is invalid")
        if not 0.0 <= float(progress) <= 1.0 or float(progress) < previous or frame_id in mapping:
            raise ValueError("Source progress evidence frame rows are not monotonic and unique")
        mapping[frame_id] = float(progress)
        previous = float(progress)
    if mapping and (next(iter(mapping.values())) != 0.0 or list(mapping.values())[-1] != 1.0):
        raise ValueError("Source progress evidence must span exactly [0, 1]")
    edges = evidence.get("edges")
    if not isinstance(edges, list) or len(edges) != len(mapping) - 1:
        raise ValueError("Source progress evidence edges are incomplete")
    expected_ids = list(mapping)
    for index, row in enumerate(edges):
        if not isinstance(row, Mapping):
            raise ValueError("Source progress evidence edge row is invalid")
        if row.get("from_frame_id") != expected_ids[index] or row.get("to_frame_id") != expected_ids[index + 1]:
            raise ValueError("Source progress evidence edge/frame linkage is invalid")
        for name in ("dx", "dy", "reliable_horizontal_increment"):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Source progress evidence edge numeric field is invalid")
        if float(row["reliable_horizontal_increment"]) < 0.0 or not isinstance(row.get("reliable"), bool):
            raise ValueError("Source progress evidence edge reliability is invalid")
    direction = 1 if direction_name.endswith("rightward") else -1
    canonical = json.dumps(
        {"direction": direction, "frames": rows, "edges": edges},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = evidence.get("content_sha256")
    if not isinstance(digest, str) or digest != hashlib.sha256(canonical).hexdigest():
        raise ValueError("Source progress evidence content digest is invalid")
    return mapping


def write_or_verify_source_progress_evidence(path: Path, evidence: Mapping[str, object]) -> dict[str, object]:
    """Atomically freeze validated evidence, rejecting a later different map."""

    source_progress_by_frame(evidence)
    serialized = json.dumps(dict(evidence), indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid source progress evidence: {path}") from exc
        source_progress_by_frame(saved)
        if saved != dict(evidence):
            raise ValueError("Source progress evidence is immutable and does not match the locked map")
        return dict(saved)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(evidence)


def write_or_verify_split(path: Path) -> dict[str, object]:
    """Create the split once, or reject any later mutation."""

    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid split definition: {path}") from exc
        if saved != SPLIT_DEFINITION:
            raise ValueError("Split definition is immutable and does not match the approved lock")
        return saved
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(SPLIT_DEFINITION, indent=2), encoding="utf-8")
    return dict(SPLIT_DEFINITION)
