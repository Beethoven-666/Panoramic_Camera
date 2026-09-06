import json

import pytest

from panorama_demo.sdk import SDKConfigurationError
from panorama_demo.sdk_state import JobRecord, JobState, StopReason


def test_transitions_are_validated_and_reloadable(tmp_path):
    record = JobRecord.create(tmp_path / "job", source_commit="abc", config_sha="def", owns_session=True)
    with pytest.raises(ValueError, match="Invalid job transition"):
        record.transition(JobState.COMPLETED)
    record.transition(JobState.WAITING_FOR_CAMERA)
    record.request_stop(StopReason.USER_REQUEST)
    record.request_stop(StopReason.INTERNAL_ERROR)
    record.transition(JobState.CANCELLED_NO_DATA)
    loaded = JobRecord.load(record.root)
    assert loaded.state is JobState.CANCELLED_NO_DATA
    assert loaded.payload["stop_reason"] == "USER_REQUEST"
    events = [json.loads(line) for line in (record.root / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["created", "transition", "stop_requested", "transition"]
    assert all(e["updated_utc"] and e["updated_monotonic_ns"] > 0 for e in events)
    with pytest.raises(ValueError):
        record.transition(JobState.FAILED)
    with pytest.raises(ValueError):
        record.update(state="CAPTURING")


def test_photo_error_remains_compatible_and_structured():
    error = SDKConfigurationError("bad", cause=OSError("cause"))
    assert isinstance(error, ValueError)
    assert error.as_dict() == {"error_code": "SDK_CONFIGURATION_ERROR", "stage": "CREATED",
                               "recoverable": False, "message": "bad", "artifact_path": None,
                               "cause_type": "OSError"}
