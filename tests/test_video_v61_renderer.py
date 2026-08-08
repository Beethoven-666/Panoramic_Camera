from __future__ import annotations

import numpy as np

from panorama_demo.video_final_sampling import VideoSamplingSource
from panorama_demo.video_v61_renderer import render_video_v61_real_sources


def _source(frame_id: int, value: int) -> VideoSamplingSource:
    y, x = np.indices((480, 160), dtype=np.float32)
    return VideoSamplingSource(frame_id, np.full((480, 160, 3), value, np.uint8), x, y, np.ones((480, 160), bool))


def test_v61_degraded_pair_still_outputs_full_canvas_once_per_real_source() -> None:
    result = render_video_v61_real_sources((_source(1, 0), _source(2, 32), _source(3, 64)))

    assert result.panorama.shape == (480, 160, 3)
    assert result.owner_frame_id is not None and np.all(result.owner_frame_id >= 0)
    assert result.metadata["raw_rgb_once_sampling"]["source_sampling_call_count"] == 3
    assert len(result.metadata["pair_states"]) == 2
    assert all(pair["gate_state"] == "hard_owner_degraded" for pair in result.metadata["pair_states"])
    assert result.panorama[0, 0, 0] == 64  # black input is still a valid source, not transparency
