"""Require real Preview publications during capture, never a final-file counter."""

from __future__ import annotations

import json
from pathlib import Path

from .observer import read_observation


def verify_native_preview(output_root: Path, *, observation_path: Path) -> dict:
    import cv2

    operation_root = Path(output_root).resolve()
    root = None
    start = stop = None
    updates = requests = 0
    last_generation = 0
    first_update = last_update = None
    for event in read_observation(observation_path):
        kind = event["kind"]
        if kind == "capture_started":
            if start is not None:
                raise ValueError("Multiple captures in one Preview observation")
            start = event.get("capture_started_monotonic_ns")
        elif kind == "capture_stopped":
            stop = event["monotonic_ns"]
        elif kind == "preview_request":
            requests += 1
            if (type(event.get("writer_queue_fraction")) not in (int, float)
                    or not 0 <= event["writer_queue_fraction"] <= 0.25
                    or event.get("writer_queue_drops") != 0
                    or event.get("preview_queue_capacity") != 1
                    or event.get("preview_queue_depth") not in {0, 1}):
                raise ValueError("Preview/writer queue gate failed")
        elif kind == "preview_state_published":
            observed = event.get("state", {})
            path = Path(event.get("path", "")).resolve()
            if not path.is_relative_to(operation_root) or path.name != "live_preview_state.json":
                raise ValueError("Preview observation belongs to another output")
            if root is not None and path.parent != root:
                raise ValueError("Multiple Preview outputs in one observation")
            root = path.parent
            if observed.get("authority") != "non_authoritative_live_preview":
                raise ValueError("Wrong observed Preview authority")
            if observed.get("capture_active") is not True:
                continue
            timestamp = observed.get("published_monotonic_ns")
            generation = observed.get("preview_generation")
            if (type(timestamp) is not int or type(generation) is not int
                    or generation != last_generation + 1 or event.get("image_exists") is not True):
                raise ValueError("Invalid or missed actual Preview publication")
            if start is None or timestamp < start or stop is not None:
                raise ValueError("Preview update is outside capture interval")
            last_generation = generation
            first_update = timestamp if first_update is None else first_update
            if last_update is not None and timestamp <= last_update:
                raise ValueError("Preview publication timestamp regressed")
            last_update = timestamp
            updates += 1
    if (updates < 2 or requests < updates or type(start) is not int or type(stop) is not int
            or last_update > stop or root is None):
        raise ValueError("At least two actual Preview updates within capture are required")
    if any(operation_root.rglob("live_preview_failure.json")):
        raise ValueError("Preview failure evidence exists")
    state = json.loads((root / "live_preview_state.json").read_text(encoding="utf-8"))
    if (state.get("authority") != "non_authoritative_live_preview"
            or state.get("preview_generation") != updates):
        raise ValueError("Wrong Preview authority or final update count")
    image = cv2.imread(str(root / "live_preview.jpg"), cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise ValueError("Preview image missing or invalid")
    for name in ("video_delivery.json", "video_report.json"):
        matches = list(operation_root.rglob(name))
        if len(matches) != 1:
            raise ValueError("Preview requires exactly one formal output in its operation")
        record = json.loads(matches[0].read_text(encoding="utf-8"))
        if record.get("authority") == "non_authoritative_live_preview":
            raise ValueError("Preview was assigned final authority")
        # Preview observability fields may mention its authority. Delivery file
        # references, however, must never use the Preview image or state.
        if name == "video_delivery.json":
            def preview_reference(value):
                if isinstance(value, dict):
                    return any(preview_reference(item) for item in value.values())
                if isinstance(value, list):
                    return any(preview_reference(item) for item in value)
                return isinstance(value, str) and Path(value).name in {"live_preview.jpg", "live_preview_state.json"}
            if preview_reference(record):
                raise ValueError("Delivery references non-authoritative Preview")
    return {"schema": "gemini305-native-h0-preview-verification/v1", "status": "PASS",
            "actual_updates": updates, "first_update_monotonic_ns": first_update,
            "last_update_monotonic_ns": last_update, "preview_queue_capacity": 1}
