import json

import cv2
import numpy as np
import pytest

from qualification.native_h0.preview_verifier import verify_native_preview


def _fixture(tmp_path):
    state = {"authority": "non_authoritative_live_preview", "preview_generation": 2,
             "capture_active": False}
    (tmp_path / "live_preview_state.json").write_text(json.dumps(state))
    cv2.imwrite(str(tmp_path / "live_preview.jpg"), np.zeros((12, 12, 3), np.uint8))
    for name in ("video_delivery.json", "video_report.json"):
        (tmp_path / name).write_text("{}")
    events = [{"kind": "capture_started", "capture_started_monotonic_ns": 10, "monotonic_ns": 10}]
    for generation in (1, 2):
        events.extend([
            {"kind": "preview_request", "monotonic_ns": generation * 20 - 1,
             "writer_queue_fraction": 0.1, "writer_queue_drops": 0,
             "preview_queue_capacity": 1, "preview_queue_depth": 1},
            {"kind": "preview_state_published", "monotonic_ns": generation * 20,
             "path": str(tmp_path / "live_preview_state.json"), "image_exists": True,
             "state": {**state, "capture_active": True, "preview_generation": generation,
                       "published_monotonic_ns": generation * 20}}])
    events.append({"kind": "capture_stopped", "monotonic_ns": 60})
    return events


def _write(tmp_path, events):
    path = tmp_path / "observation.json"
    (tmp_path / "observation.jsonl").write_text("".join(
        json.dumps({**event, "sequence": index}) + "\n" for index, event in enumerate(events, 1)))
    path.write_text(json.dumps({"schema": "gemini305-native-h0-observation/v1", "completed": True,
                              "observer_errors": [], "event_count": len(events),
                              "events_path": str(path.with_suffix(".jsonl"))}))
    return path


def test_real_updates_are_required(tmp_path):
    events = _fixture(tmp_path)
    result = verify_native_preview(tmp_path, observation_path=_write(tmp_path, events))
    assert result["actual_updates"] == 2


@pytest.mark.parametrize("failure", ["wrong_authority", "outside_capture", "one_update", "writer_block",
                                     "preview_failure", "delivery", "counter_only"])
def test_preview_negative_gates(tmp_path, failure):
    events = _fixture(tmp_path)
    if failure == "wrong_authority":
        events[2]["state"]["authority"] = "formal"
    elif failure == "outside_capture":
        events[2]["state"]["published_monotonic_ns"] = 5
    elif failure == "one_update":
        del events[3:5]
    elif failure == "writer_block":
        events[1]["writer_queue_fraction"] = 0.99
    elif failure == "preview_failure":
        (tmp_path / "live_preview_failure.json").write_text("{}")
    elif failure == "delivery":
        (tmp_path / "video_delivery.json").write_text(json.dumps({"panorama": "live_preview.jpg"}))
    else:
        events = [events[0], events[-1]]
    with pytest.raises(ValueError):
        verify_native_preview(tmp_path, observation_path=_write(tmp_path, events))
