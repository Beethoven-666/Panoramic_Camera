from __future__ import annotations

import numpy as np

from panorama_demo.video_final_sampling import VideoSamplingSource
from panorama_demo.video_v6_pair_renderer import render_video_v6_real_pair, render_video_v6_real_sources


def _source(frame_id: int, value: int) -> VideoSamplingSource:
    shape = (480, 120)
    y, x = np.indices(shape, dtype=np.float32)
    return VideoSamplingSource(frame_id, np.full((*shape, 3), value, np.uint8), x, y, np.ones(shape, bool))


def test_v6_pair_route_runs_one_sampling_dis_graphcut_guard_blend_and_quality() -> None:
    result = render_video_v6_real_pair(_source(1, 80), _source(2, 80))

    assert result.source_sampling_call_count == 2
    assert result.graphcut_audit.graphcut_called
    assert result.graphcut_audit.accepted
    assert result.dis_evidence.flow_forward.shape[:2] == (480, 120)
    assert result.valid_mask.all()
    assert result.quality.owner_topology_ok


def test_v6_source_chain_samples_each_real_source_only_once() -> None:
    result = render_video_v6_real_sources((_source(1, 80), _source(2, 80), _source(3, 80)))

    assert result.source_sampling_call_count == 3
    assert len(result.graphcut_audits) == 2
    assert all(audit.graphcut_called for audit in result.graphcut_audits)
    assert result.valid_mask.all()


def test_v6_chain_records_real_owner_expansion_instead_of_faking_a_dp_fallback(monkeypatch) -> None:
    import panorama_demo.video_v6_pair_renderer as renderer

    original = renderer.solve_video_graphcut_seam

    def rejected(*args, **kwargs):
        outcome = original(*args, **kwargs)
        return type(outcome)(outcome.choose_new, type(outcome.audit)(
            outcome.audit.graphcut_called, outcome.audit.rescue_corridor_used, outcome.audit.seam_x_by_row,
            outcome.audit.maximum_adjacent_row_step_px, outcome.audit.owner_island_count,
            outcome.audit.small_fragment_count, outcome.audit.valid_pixel_exactly_one_owner,
            False, "synthetic_topology_failure",
        ))

    monkeypatch.setattr(renderer, "solve_video_graphcut_seam", rejected)
    result = renderer.render_video_v6_real_sources((_source(1, 80), _source(2, 80)))

    assert result.expanded_real_owner_pair_frame_ids == ((1, 2),)
