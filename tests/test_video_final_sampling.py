from __future__ import annotations

import numpy as np

from panorama_demo.video_final_sampling import VideoSamplingSource, render_video_final_once


def _source(frame_id: int, colour: tuple[int, int, int], shape: tuple[int, int]) -> VideoSamplingSource:
    height, width = shape
    raw = np.full((height, width, 3), colour, np.uint8)
    y, x = np.indices(shape, dtype=np.float32)
    return VideoSamplingSource(frame_id, raw, x, y, np.ones(shape, bool))


def test_final_sampling_remaps_each_raw_source_exactly_once_and_obeys_owner(monkeypatch) -> None:
    import panorama_demo.video_final_sampling as sampling

    calls: list[int] = []
    original = sampling.cv2.remap

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling.cv2, "remap", tracked)
    shape = (12, 16)
    owner = np.full(shape, 4, np.int32)
    owner[:, 8:] = 9
    result = render_video_final_once((_source(4, (0, 0, 0), shape), _source(9, (10, 20, 30), shape)), owner)

    assert len(calls) == 2
    assert result.audit.exactly_one_raw_rgb_sampling_per_source
    assert result.audit.strict_owner_partition
    assert np.all(result.bgr[:, :8] == 0)  # Black source content remains valid.
    assert np.all(result.bgr[:, 8:] == (10, 20, 30))


def test_final_sampling_rejects_unknown_or_invalid_owner_source() -> None:
    source = _source(2, (1, 2, 3), (4, 4))
    with np.testing.assert_raises_regex(ValueError, "unknown"):
        render_video_final_once((source,), np.full((4, 4), 7, np.int32))
