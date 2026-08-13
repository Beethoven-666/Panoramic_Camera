"""Immutable source-frame annotation validation and preview rendering.

Annotations are measurement input, not rendering input.  Their coordinate
system is explicitly a real decoded source frame, so they cannot be confused
with a panorama crop or used to manufacture a render source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .video_split import source_progress_by_frame


ANNOTATION_SCHEMA = "gemini305-video-source-annotations/v1"
ANNOTATION_SCHEMA_V2 = "gemini305-video-source-annotations/v2"
SUPPORTED_ANNOTATION_SCHEMAS = frozenset((ANNOTATION_SCHEMA, ANNOTATION_SCHEMA_V2))

# These labels describe how a fixed *measurement* is evaluated.  They never
# select render sources, change an owner, or otherwise feed back into a video
# renderer.  v1 intentionally has no role field and keeps its historical
# compact-object semantics.
ANNOTATION_ROLES_BY_KIND = {
    "objects": frozenset(("compact_foreground_single_owner", "extended_background_structure")),
    "lines": frozenset(("long_line",)),
    "safe_background": frozenset(("safe_background",)),
}


class VideoAnnotationError(ValueError):
    """Raised when immutable manual measurement input is malformed."""


def load_source_annotations(path: str | Path) -> dict[str, Any]:
    annotation_path = Path(path).expanduser().resolve()
    try:
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VideoAnnotationError(f"Invalid annotation JSON: {annotation_path}") from exc
    if not isinstance(data, dict) or data.get("schema") not in SUPPORTED_ANNOTATION_SCHEMAS:
        raise VideoAnnotationError("Unsupported source annotation schema")
    frames = data.get("source_frames")
    if not isinstance(frames, Mapping) or not frames:
        raise VideoAnnotationError("Annotations require source_frames")
    for frame_id, descriptor in frames.items():
        if not str(frame_id).isdigit() or not isinstance(descriptor, Mapping):
            raise VideoAnnotationError("source_frames must map real integer frame ids to descriptors")
        if not isinstance(descriptor.get("color_path"), str):
            raise VideoAnnotationError("Every annotated frame needs color_path")
        progress = descriptor.get("scan_progress")
        if not isinstance(progress, (int, float)) or not 0.0 <= float(progress) <= 1.0:
            raise VideoAnnotationError("Annotated source scan_progress must be in [0, 1]")
    for group in ("objects", "lines", "safe_background"):
        entries = data.get(group)
        if not isinstance(entries, list) or not entries:
            raise VideoAnnotationError(f"Annotations require non-empty {group}")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise VideoAnnotationError(f"{group} entries must be mappings")
            _validate_entry(entry, frames, group, schema=str(data["schema"]))
    return data


def _validate_entry(entry: Mapping[str, Any], frames: Mapping[str, Any], group: str, *, schema: str) -> None:
    label = entry.get("id")
    if not isinstance(label, str) or not label:
        raise VideoAnnotationError(f"{group} entry requires id")
    frame_id = entry.get("frame_id")
    if not isinstance(frame_id, int) or str(frame_id) not in frames:
        raise VideoAnnotationError(f"{group} entry {label!r} must reference an annotated real source frame")
    # Paired real-source annotations may explicitly name one measurement
    # group.  The group is read-only evaluation metadata: it never selects a
    # source, changes a pose, or affects rendering.  Omission preserves the
    # historical one-entry-one-measurement behaviour.
    measurement_group = entry.get("measurement_group")
    if measurement_group is not None and (
        not isinstance(measurement_group, str) or not measurement_group.strip()
    ):
        raise VideoAnnotationError(
            f"{group} entry {label!r} measurement_group must be a non-empty string when present"
        )
    role = entry.get("role")
    if schema == ANNOTATION_SCHEMA_V2:
        if not isinstance(role, str) or role not in ANNOTATION_ROLES_BY_KIND[group]:
            allowed = ", ".join(sorted(ANNOTATION_ROLES_BY_KIND[group]))
            raise VideoAnnotationError(
                f"{group} entry {label!r} requires one of the v2 roles: {allowed}"
            )
    elif role is not None:
        raise VideoAnnotationError("v1 source annotations must not declare v2 measurement roles")
    if group == "lines":
        points = entry.get("points")
        if not isinstance(points, list) or len(points) != 2:
            raise VideoAnnotationError(f"Line annotation {label!r} requires exactly two points")
    else:
        points = entry.get("polygon")
        if not isinstance(points, list) or len(points) < 3:
            raise VideoAnnotationError(f"Polygon annotation {label!r} requires at least three points")
    for point in points:
        if (
            not isinstance(point, list)
            or len(point) != 2
            or not all(isinstance(value, (int, float)) and np.isfinite(value) for value in point)
        ):
            raise VideoAnnotationError(f"Annotation {label!r} has an invalid point")


def validate_annotation_coordinates(
    annotations: Mapping[str, Any], *, session_root: str | Path
) -> None:
    """Ensure every fixed polygon/line lies inside its decoded source image."""

    root = Path(session_root).expanduser().resolve()
    frames = annotations["source_frames"]
    for frame_id, descriptor in frames.items():
        image = cv2.imread(str(root / descriptor["color_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise VideoAnnotationError(f"Annotated source frame cannot be decoded: {frame_id}")
        height, width = image.shape[:2]
        for group in ("objects", "lines", "safe_background"):
            for entry in annotations[group]:
                if int(entry["frame_id"]) != int(frame_id):
                    continue
                points = entry.get("points", entry.get("polygon"))
                for x, y in points:
                    if not 0.0 <= float(x) < width or not 0.0 <= float(y) < height:
                        raise VideoAnnotationError(
                            f"Annotation {entry['id']!r} lies outside source frame {frame_id}"
                        )


def audit_annotation_source_progress(
    annotations: Mapping[str, Any],
    source_progress_by_frame: Mapping[int, float],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare immutable annotation progress with independently computed progress.

    The caller supplies the progress mapping produced from the real source
    sequence (for example, the locked cumulative-motion scan analysis).  This
    function does not derive, normalise, or repair either value: a mismatch is
    evidence of a split/annotation inconsistency, not a reason to move an
    annotation.
    """

    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0.0:
        raise VideoAnnotationError("Annotation progress tolerance must be a finite non-negative number")
    frames = annotations.get("source_frames")
    if not isinstance(frames, Mapping) or not frames:
        raise VideoAnnotationError("Annotations require source_frames")
    entries: list[dict[str, Any]] = []
    mismatches: list[int] = []
    for raw_frame_id, descriptor in frames.items():
        if not str(raw_frame_id).isdigit() or not isinstance(descriptor, Mapping):
            raise VideoAnnotationError("source_frames must map real integer frame ids to descriptors")
        frame_id = int(raw_frame_id)
        declared = descriptor.get("scan_progress")
        if not isinstance(declared, (int, float)) or not np.isfinite(declared):
            raise VideoAnnotationError(f"Annotated source frame {frame_id} has no finite scan_progress")
        if frame_id not in source_progress_by_frame:
            raise VideoAnnotationError(f"Real source progress mapping lacks annotated frame {frame_id}")
        observed = source_progress_by_frame[frame_id]
        if not isinstance(observed, (int, float)) or not np.isfinite(observed) or not 0.0 <= float(observed) <= 1.0:
            raise VideoAnnotationError(f"Real source progress for frame {frame_id} must be finite and in [0, 1]")
        difference = abs(float(declared) - float(observed))
        matched = difference <= float(tolerance)
        if not matched:
            mismatches.append(frame_id)
        entries.append(
            {
                "frame_id": frame_id,
                "declared_scan_progress": float(declared),
                "observed_scan_progress": float(observed),
                "absolute_difference": difference,
                "matched": matched,
            }
        )
    return {
        "schema": "gemini305-video-annotation-source-progress-audit/v1",
        "measurement_only": True,
        "tolerance": float(tolerance),
        "frame_count": len(entries),
        "verified": not mismatches,
        "mismatched_frame_ids": mismatches,
        "frames": entries,
    }


def verify_annotation_source_progress(
    annotations: Mapping[str, Any],
    source_progress_by_frame: Mapping[int, float],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Strictly return a source-progress audit or fail on any mismatch."""

    audit = audit_annotation_source_progress(
        annotations, source_progress_by_frame, tolerance=tolerance
    )
    if not audit["verified"]:
        values = ", ".join(str(frame_id) for frame_id in audit["mismatched_frame_ids"])
        raise VideoAnnotationError(f"Fixed annotation scan_progress mismatch for source frame(s): {values}")
    return audit


def verify_annotation_source_progress_evidence(
    annotations: Mapping[str, Any],
    evidence: Mapping[str, object],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Verify annotations against frozen real-source progress evidence only.

    This accepts the serializable evidence artifact rather than a hand-built
    mapping, so evaluation cannot quietly substitute inferred source progress.
    """

    try:
        mapping = source_progress_by_frame(evidence)
    except ValueError as exc:
        raise VideoAnnotationError(f"Invalid real source progress evidence: {exc}") from exc
    audit = verify_annotation_source_progress(annotations, mapping, tolerance=tolerance)
    return {**audit, "source_progress_evidence_sha256": evidence["content_sha256"]}


def write_annotation_preview(
    annotations: Mapping[str, Any], *, session_root: str | Path, output: str | Path
) -> Path:
    """Write a contact-sheet preview for human audit; never used by algorithms."""

    root, destination = Path(session_root).expanduser().resolve(), Path(output).expanduser().resolve()
    validate_annotation_coordinates(annotations, session_root=root)
    palette = {"objects": (0, 0, 255), "lines": (255, 0, 0), "safe_background": (0, 180, 0)}
    panels: list[np.ndarray] = []
    for frame_id, descriptor in annotations["source_frames"].items():
        image = cv2.imread(str(root / descriptor["color_path"]), cv2.IMREAD_COLOR)
        assert image is not None
        panel = image.copy()
        for group, color in palette.items():
            for entry in annotations[group]:
                if int(entry["frame_id"]) != int(frame_id):
                    continue
                points = np.asarray(entry.get("points", entry.get("polygon")), dtype=np.int32)
                if group == "lines":
                    cv2.line(panel, tuple(points[0]), tuple(points[1]), color, 2)
                else:
                    cv2.polylines(panel, [points], True, color, 2)
                cv2.putText(panel, entry["id"], tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
        panels.append(panel)
    preview = cv2.vconcat(panels)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), preview):
        raise VideoAnnotationError(f"Could not write annotation preview: {destination}")
    return destination


__all__ = [
    "ANNOTATION_SCHEMA",
    "ANNOTATION_SCHEMA_V2",
    "SUPPORTED_ANNOTATION_SCHEMAS",
    "ANNOTATION_ROLES_BY_KIND",
    "VideoAnnotationError",
    "load_source_annotations",
    "audit_annotation_source_progress",
    "verify_annotation_source_progress",
    "verify_annotation_source_progress_evidence",
    "validate_annotation_coordinates",
    "write_annotation_preview",
]
