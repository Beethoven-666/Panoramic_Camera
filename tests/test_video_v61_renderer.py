from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import panorama_demo.video_v61_renderer as renderer
from panorama_demo.video_final_sampling import VideoSamplingSource
from panorama_demo.video_graphcut_seam import VideoGraphCutAudit, VideoGraphCutResult
from panorama_demo.video_hard_guards import VideoHardGuardAudit, VideoHardGuards
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


def _source(frame_id: int, value: int, *, valid: np.ndarray | None = None) -> VideoSamplingSource:
    y, x = np.indices((480, 160), dtype=np.float32)
    support = np.ones((480, 160), bool) if valid is None else np.asarray(valid, bool)
    return VideoSamplingSource(
        frame_id,
        np.full((480, 160, 3), value, np.uint8),
        x,
        y,
        support,
    )


def _evidence(shape: tuple[int, int]) -> VideoDISPairEvidence:
    zeros = np.zeros(shape, np.float32)
    return VideoDISPairEvidence(
        np.zeros((*shape, 2), np.float32),
        np.zeros((*shape, 2), np.float32),
        zeros,
        zeros,
        zeros,
        np.zeros(shape, bool),
        np.ones(shape, np.float32),
        np.ones(shape, bool),
        np.zeros((*shape, 4), np.uint8),
    )


def _install_accepted_gates(monkeypatch, *, translation_preview_px: float = 0.0) -> None:
    monkeypatch.setattr(
        renderer,
        "video_dis_pair_evidence",
        lambda _old, _new, support: _evidence(np.asarray(support).shape),
    )
    alignment_audit = SimpleNamespace(
        accepted=True,
        selected_model="translation" if translation_preview_px else "identity",
        rejection_reason=None,
        maximum_displacement_px=abs(translation_preview_px),
    )
    matrix = np.array(
        ((1.0, 0.0, 0.0), (0.0, 1.0, translation_preview_px), (0.0, 0.0, 1.0)),
        np.float64,
    )
    monkeypatch.setattr(
        renderer,
        "fit_near_protected_alignment",
        lambda *_args, **_kwargs: SimpleNamespace(audit=alignment_audit, matrix=matrix),
    )

    def geometry(_old, _new, _evidence_value, *, support, protected, config):
        del protected
        return SimpleNamespace(
            accepted=True,
            rejection_reason=None,
            tail_guard=np.zeros(np.asarray(support).shape, bool),
            as_report_dict=lambda: {
                "accepted": True,
                "edge_p95_px": 0.5,
                "edge_abs_max_px": 2.0,
                "tail_count": 1,
                "configured_tail_threshold_px": config.tail_threshold_px,
            },
        )

    monkeypatch.setattr(renderer, "evaluate_v61_geometry_gate", geometry)

    def empty_guards(old, _new, _evidence_value, **_kwargs):
        shape = np.asarray(old).shape[:2]
        empty = np.zeros(shape, bool)
        audit = VideoHardGuardAudit(0, 0, 0, 0, 0, 0)
        return VideoHardGuards(
            empty.copy(), empty.copy(), empty.copy(), empty.copy(), empty.copy(),
            empty.copy(), empty.copy(), audit,
        )

    monkeypatch.setattr(renderer, "build_video_hard_guards", empty_guards)


def _accepted_graphcut(old_bgr, _new_bgr, old_valid, new_valid, **_kwargs):
    height, width = old_bgr.shape[:2]
    choose_new = np.zeros((height, width), bool)
    choose_new[:, width // 2 :] = True
    choose_new &= np.asarray(new_valid, bool)
    audit = VideoGraphCutAudit(
        True,
        False,
        (width // 2,) * height,
        0,
        0,
        0,
        bool(np.all(np.asarray(old_valid, bool) | np.asarray(new_valid, bool))),
        True,
        None,
    )
    return VideoGraphCutResult(choose_new=choose_new, audit=audit)


def test_v61_degraded_pair_outputs_real_full_canvas_and_measured_hard_seams(monkeypatch) -> None:
    _install_accepted_gates(monkeypatch)

    def fail_graphcut(*_args, **_kwargs):
        raise RuntimeError("synthetic graphcut failure")

    monkeypatch.setattr(renderer, "solve_video_graphcut_seam", fail_graphcut)
    result = renderer.render_video_v61_real_sources(
        (_source(1, 0), _source(2, 32), _source(3, 64)),
    )

    assert result.panorama.shape == (480, 160, 3)
    assert result.owner_frame_id is not None and np.all(result.owner_frame_id >= 0)
    assert result.metadata["raw_rgb_once_sampling"]["source_sampling_call_count"] == 3
    assert result.metadata["raw_rgb_once_sampling"]["full_resolution_inverse_remap_call_count_by_source"] == [1, 1, 1]
    assert len(result.metadata["pair_states"]) == 2
    assert all(pair["gate_state"] == "hard_owner_degraded" for pair in result.metadata["pair_states"])
    assert all(pair["blend_pixel_count"] == 0 for pair in result.metadata["pair_states"])
    assert all(pair["effective_owner_handoff"]["evaluated_seam_rows"] == 480 for pair in result.metadata["pair_states"])
    quality = result.metadata["quality_metrics"]
    assert quality["quality_pass"] is quality["strict_quality_pass"] is False
    assert quality["grade"] == "C"
    assert quality["seam_step_p95_px"] == 0.0
    assert isinstance(quality["double_edge_count"], int)
    assert isinstance(quality["ghost_count"], int)
    assert result.panorama[0, 0, 0] == 0
    assert result.owner_frame_id[0, 0] == 1  # black is valid owned RGB, not transparency
    assert result.panorama[0, -1, 0] == 64
    assert np.all(np.diff(result.owner_frame_id[0].astype(np.int64)) >= 0)
    json.dumps(result.metadata, allow_nan=False)


def test_v61_removes_only_tiny_disconnected_validity_specks_before_owner(
    monkeypatch,
) -> None:
    _install_accepted_gates(monkeypatch)
    monkeypatch.setattr(renderer, "solve_video_graphcut_seam", _accepted_graphcut)
    valid = np.ones((480, 160), bool)
    valid[:, 80:] = False
    valid[0, 159] = True

    result = renderer.render_video_v61_real_sources(
        (_source(1, 0, valid=valid), _source(2, 32)),
    )

    cleanup = result.metadata["support_fragment_cleanup"]
    assert cleanup["removed_pixel_count_by_source"][1] == 1
    assert result.owner_frame_id[0, 159] == 2
    assert result.metadata["owner_audit"]["owner_island_count"] == 0


def test_v61_reassigns_a_cut_created_tiny_owner_island_to_adjacent_support() -> None:
    owner = np.full((480, 160), 2, np.int32)
    owner[:, :80] = 1
    owner[100:113, 120] = 1
    support = np.ones(owner.shape, bool)

    repaired, count, dropped = renderer._repair_pair_owner_fragments(
        owner,
        old_frame_id=1,
        new_frame_id=2,
        supports={1: support, 2: support},
    )

    assert count == 13
    assert dropped == 0
    assert np.all(repaired[100:113, 120] == 2)
    assert renderer._component_counts(repaired == 1) == (0, 0)


def test_v61_drops_only_an_unsupported_tiny_cut_fragment_from_final_validity() -> None:
    owner = np.full((480, 160), 2, np.int32)
    owner[:, :80] = 1
    owner[100:103, 120] = 1
    old_support = owner == 1
    old_support[:, :80] = True
    new_support = np.ones(owner.shape, bool)
    new_support[100:103, 120] = False
    supports = {1: old_support.copy(), 2: new_support}

    repaired, reassigned, dropped = renderer._repair_pair_owner_fragments(
        owner,
        old_frame_id=1,
        new_frame_id=2,
        supports=supports,
        allow_unsupported_drop=True,
    )

    assert reassigned == 0
    assert dropped == 3
    assert np.all(repaired[100:103, 120] == -1)
    assert not np.any(supports[1][100:103, 120])


def test_v61_accepted_graphcut_applies_alignment_to_grid_before_one_final_sample(
    monkeypatch,
) -> None:
    _install_accepted_gates(monkeypatch, translation_preview_px=1.0)
    monkeypatch.setattr(renderer, "solve_video_graphcut_seam", _accepted_graphcut)
    actual_sample = renderer.sample_video_sources_once
    calls: list[tuple[VideoSamplingSource, ...]] = []

    def record_sample(sources):
        ordered = tuple(sources)
        calls.append(ordered)
        return actual_sample(ordered)

    monkeypatch.setattr(renderer, "sample_video_sources_once", record_sample)
    result = renderer.render_video_v61_real_sources(
        (_source(10, 0), _source(20, 80)),
        geometry_gate_config={
            "minimum_reliable_pixels": 128,
            "fb_p95_max_px": 1.25,
            "edge_p95_max_px": 0.75,
            "minimum_matched_edge_fraction": 0.85,
            "tail_threshold_px": 1.25,
            "tail_dilation_px": 3,
        },
    )

    assert len(calls) == 1
    assert not np.array_equal(calls[0][1].inverse_y, _source(20, 80).inverse_y)
    assert result.metadata["alignment_execution"][0]["matrix_composed_into_final_inverse_grid"] is True
    pair = result.metadata["pair_states"][0]
    assert pair["gate_state"] == "graphcut_accepted", pair["fallback_reason"]
    assert pair["alignment_grid_applied"] is True
    assert pair["geometry"]["edge_abs_max_px"] == 2.0
    quality = result.metadata["quality_metrics"]
    assert quality["quality_pass"] is quality["strict_quality_pass"] is True
    assert quality["grade"] == "B"
    assert result.metadata["candidate_run_state"] == "completed"
    assert result.metadata["selection_eligible"] is True
    assert result.metadata["component_execution"]["v61_tail_guarded_full_panorama"]["applied_to_output"] is True
    json.dumps(result.metadata, allow_nan=False)


def test_v61_graphcut_exception_becomes_audited_c_hard_owner(monkeypatch) -> None:
    _install_accepted_gates(monkeypatch)

    def fail_graphcut(*_args, **_kwargs):
        raise RuntimeError("synthetic graphcut failure")

    monkeypatch.setattr(renderer, "solve_video_graphcut_seam", fail_graphcut)
    result = renderer.render_video_v61_real_sources((_source(1, 12), _source(2, 24)))

    pair = result.metadata["pair_states"][0]
    assert pair["gate_state"] == "hard_owner_degraded"
    assert pair["graphcut_called"] is True
    assert pair["graphcut_accepted"] is False
    assert pair["fallback_reason"] == "graphcut_exception:RuntimeError"
    assert pair["blend_pixel_count"] == 0
    assert pair["effective_owner_handoff"]["evaluated"] is True
    assert pair["effective_owner_handoff"]["double_edge_count"] is not None
    assert pair["effective_owner_handoff"]["ghost_count"] is not None
    assert result.metadata["quality_metrics"]["seam_step_p95_px"] == 0.0
    assert result.metadata["quality_metrics"]["grade"] == "C"
    assert result.metadata["selection_eligible"] is False
    json.dumps(result.metadata, allow_nan=False)


def test_v61_candidate_reports_real_open3d_quality_and_rejects_wrong_adjacency(monkeypatch) -> None:
    sources = (_source(1, 10), _source(2, 20))
    monkeypatch.setattr(renderer, "build_v6_sampling_sources", lambda *_args, **_kwargs: sources)
    frames = (SimpleNamespace(frame_id=1), SimpleNamespace(frame_id=2))
    edge = SimpleNamespace(
        reference_node_id=1,
        source_node_id=2,
        structurally_valid=True,
        reliable=False,
        backend="open3d_tensor_cuda_rgbd",
        failure_reasons=("fitness_below_quality_gate",),
    )
    result = renderer.render_video_v61_candidate(
        frames,
        (np.eye(4), np.eye(4)),
        object(),
        pushbroom_config={},
        rgb_motions=[],
        motion_pixels_to_full_resolution=1.0,
        open3d_edges=(edge,),
    )

    evidence = result.metadata["component_evidence"]["open3d_rgbd_edges"]
    assert evidence["valid"] is True
    assert evidence["audit_passed"] is False
    assert evidence["edges"][0]["failure_reasons"] == ["fitness_below_quality_gate"]
    assert result.metadata["quality_metrics"]["quality_pass"] is False
    assert result.metadata["quality_metrics"]["grade"] == "C"
    assert result.metadata["selection_eligible"] is False

    wrong = SimpleNamespace(**{**vars(edge), "reference_node_id": 99})
    with pytest.raises(RuntimeError, match="does not bind"):
        renderer.render_video_v61_candidate(
            frames,
            (np.eye(4), np.eye(4)),
            object(),
            pushbroom_config={},
            rgb_motions=[],
            motion_pixels_to_full_resolution=1.0,
            open3d_edges=(wrong,),
        )
