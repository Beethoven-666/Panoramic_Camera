from __future__ import annotations

import numpy as np

from panorama_demo.video_final_sampling import VideoSamplingSource
from panorama_demo.video_v6_pair_renderer import (
    _apply_output_mesh_to_grid,
    _compact_object_owner_preference,
    _hard_frontality_supports,
    _low_structure_corridor_left,
    _photometric_matched_right,
    render_video_v6_real_pair,
    render_video_v6_real_sources,
)
from panorama_demo.video_object_mask import build_video_object_masks
from panorama_demo.video_object_patch_planning import VideoDirectSourceSupport
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _source(frame_id: int, value: int, *, width: int = 120) -> VideoSamplingSource:
    shape = (480, width)
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
    assert result.graphcut_audits[0].rejection_reason == "synthetic_topology_failure"


def test_v6_graphcut_retries_once_in_a_192px_corridor_using_cached_pair_evidence(monkeypatch) -> None:
    import panorama_demo.video_v6_pair_renderer as renderer

    original = renderer.solve_video_graphcut_seam
    widths: list[int] = []

    def reject_normal_then_use_rescue(*args, **kwargs):
        outcome = original(*args, **kwargs)
        widths.append(args[0].shape[1])
        if len(widths) == 1:
            return type(outcome)(outcome.choose_new, type(outcome.audit)(
                outcome.audit.graphcut_called, outcome.audit.rescue_corridor_used,
                outcome.audit.seam_x_by_row, outcome.audit.maximum_adjacent_row_step_px,
                outcome.audit.owner_island_count, outcome.audit.small_fragment_count,
                outcome.audit.valid_pixel_exactly_one_owner, False, "synthetic_normal_failure",
            ))
        return outcome

    monkeypatch.setattr(renderer, "solve_video_graphcut_seam", reject_normal_then_use_rescue)
    result = renderer.render_video_v6_real_sources((_source(1, 80, width=240), _source(2, 80, width=240)))

    assert widths == [160, 192]
    assert result.graphcut_audits[0].accepted
    assert result.graphcut_audits[0].rescue_corridor_used


def test_photometric_right_samples_follow_the_cached_forward_dis_flow() -> None:
    right = np.zeros((4, 5, 3), np.uint8)
    right[:, :, 0] = np.arange(5, dtype=np.uint8)
    flow = np.zeros((4, 5, 2), np.float32)
    flow[..., 0] = 1.0
    zeros = np.zeros((4, 5), np.float32)
    evidence = VideoDISPairEvidence(flow, flow.copy(), zeros, zeros, zeros, np.zeros((4, 5), bool), zeros, np.ones((4, 5), bool), np.zeros((4, 5, 4), np.uint8))

    matched, valid = _photometric_matched_right(right, np.ones((4, 5), bool), evidence)

    assert matched[0, 0, 0] == 1
    assert not valid[:, -1].any()


def test_bounded_mesh_changes_only_explicit_safe_background_grid_cells() -> None:
    source = _source(7, 90)
    mesh = np.zeros((3, 3, 2), np.float32)
    mesh[..., 0] = 1.0
    safe = np.zeros(source.valid_mask.shape, bool)
    safe[120:360, 30:90] = True

    adjusted = _apply_output_mesh_to_grid(source, mesh, safe, preview_scale=4)

    assert np.array_equal(adjusted.inverse_x[~safe], source.inverse_x[~safe])
    assert np.array_equal(adjusted.inverse_y[~safe], source.inverse_y[~safe])
    assert not np.array_equal(adjusted.inverse_x[safe], source.inverse_x[safe])


def test_hard_frontality_support_is_mapped_from_raw_columns_to_canvas_coordinates() -> None:
    source = _source(7, 90)

    support = _hard_frontality_supports((source,), {7: (30, 90)})

    assert support[0].frame_id == 7
    assert support[0].support_x == (30.0, 90.0)


def test_v6_corridor_prefers_lower_canny_structure_within_common_real_support() -> None:
    old = np.zeros((480, 240, 3), np.uint8)
    new = old.copy()
    old[:, 100:104] = 255
    new[:, 100:104] = 255

    left = _low_structure_corridor_left(
        old, new, np.ones((480, 240), bool), overlap_left=0, overlap_right=240, width=96,
        image_width=240,
    )

    assert not left <= 102 < left + 96


def test_compact_object_switches_only_to_one_complete_new_direct_source() -> None:
    height, width = 80, 120
    flow = np.zeros((height, width, 2), np.float32)
    flow[20:48, 85:115, 0] = 4.0
    backward = np.zeros_like(flow)
    backward[20:48, 89:119, 0] = -4.0
    zeros = np.zeros((height, width), np.float32)
    evidence = VideoDISPairEvidence(
        flow, backward, zeros, zeros, zeros, np.zeros((height, width), bool), zeros,
        np.ones((height, width), bool), np.zeros((height, width, 4), np.uint8),
    )
    masks = build_video_object_masks(evidence, strong_protection=np.zeros((height, width), bool))

    preferred = _compact_object_owner_preference(
        masks, old_frame_id=1, new_frame_id=2, canvas_left=0,
        supports=(VideoDirectSourceSupport(1, (0.0, 80.0)), VideoDirectSourceSupport(2, (79.0, 120.0))),
    )

    assert preferred[34, 100]
    assert preferred.sum() >= masks.candidate_mask.sum()
    assert not _compact_object_owner_preference(
        masks, old_frame_id=1, new_frame_id=2, canvas_left=0,
        supports=(VideoDirectSourceSupport(1, (0.0, 100.0)), VideoDirectSourceSupport(2, (100.0, 120.0))),
    ).any()
