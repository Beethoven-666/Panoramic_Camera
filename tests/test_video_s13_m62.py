from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from panorama_demo.video_s13_m6 import run_s13_m6
from panorama_demo.video_s13_m62_equivalence import compare_s13_m62_u8
from panorama_demo.video_s13_m62_plan import build_s13_m62_execution_plan
from panorama_demo.video_s13_m62_reference import execute_s13_m62_cpu_reference
from panorama_demo.video_s13_m62_runner import run_s13_m62
from panorama_demo.video_s13_cuda_runtime import S13CudaRuntime
from panorama_demo.cuda_backend import cuda_status
from panorama_demo.video_s13_replay import S13P2ReplayPair, S13VerifiedP2


def _p2(tmp_path: Path) -> S13VerifiedP2:
    shape = (4, 12)
    owner = np.broadcast_to((np.arange(12) >= 6).astype(np.int32), shape).copy()
    valid = np.ones(shape, bool)
    pair_shape = (4, 8)
    pair = S13P2ReplayPair(0, 0, 1, 10, 11, 2, 10, np.full(4, 6, np.int32),
        np.broadcast_to(np.arange(2, 10, dtype=np.float32), pair_shape).copy(),
        np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], pair_shape).copy(), np.ones(pair_shape, bool),
        np.broadcast_to(np.arange(2, 10, dtype=np.float32), pair_shape).copy(),
        np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], pair_shape).copy(), np.ones(pair_shape, bool),
        np.broadcast_to(np.arange(2, 10)[None, :] >= 6, pair_shape), 0, 0, "a" * 64)
    p2_image = np.repeat(np.where(owner[..., None] == 0, 80, 90), 3, axis=2).astype(np.uint8)
    return S13VerifiedP2(tmp_path, {"source_count": 2}, "b" * 64, p2_image, valid,
        {"owner_source_index": owner, "source_u": np.broadcast_to(np.arange(12, dtype=np.float32), shape).copy(),
         "source_v": np.broadcast_to(np.arange(4, dtype=np.float32)[:, None], shape).copy()}, ({},), (pair,), {})


def test_m62_plan_is_readonly_and_cpu_reference_matches_legacy(tmp_path: Path) -> None:
    p2 = _p2(tmp_path)
    images = {10: np.full((4, 12, 3), 80, np.uint8), 11: np.full((4, 12, 3), 90, np.uint8)}
    plan = build_s13_m62_execution_plan(p2, images.__getitem__, force_owner_only_pair_indices=frozenset({0}))
    assert plan.valid_mask.flags.writeable is False
    assert not hasattr(plan, "corrected_rois")
    assert plan.source_rois[0].map_u.flags.writeable is False
    assert plan.blend_plans[0].secondary_weight.flags.writeable is False
    profile = plan.evidence.performance
    assert profile["m62_evidence_reference_adjacent_remap_count"] == 2
    assert profile["m62_evidence_reference_bridge_remap_count"] == 0
    assert profile["m62_evidence_reference_remap_pixel_count"] == 64
    assert profile["m62_evidence_reference_remap_seconds"] >= 0.0
    assert profile["m62_evidence_packed_remap_count"] == 0
    assert profile["m62_evidence_adjacent_usage_count"] == 2
    assert plan.decision_timings["m62_evidence_reference_adjacent_remap_count"] == 2
    current = execute_s13_m62_cpu_reference(plan)
    legacy = run_s13_m6(p2, images.__getitem__, force_owner_only_pair_indices=frozenset({0}))
    np.testing.assert_array_equal(current.visual_panorama, legacy.visual_panorama)


def test_m62_u8_authority_requires_exact_bytes() -> None:
    reference = np.zeros((2, 2, 3), np.uint8)
    candidate = reference.copy()
    candidate[0, 0, 1] = 1
    result = compare_s13_m62_u8(reference, candidate)
    assert result["shadow_gate_passed"] is True
    assert result["authority_gate_passed"] is False
    assert result["differing_channel_count"] == 1


def test_candidate_single_pass_q0_b0_skips_all_pixel_executors(tmp_path: Path) -> None:
    p2 = _p2(tmp_path)
    images = {10: np.full((4, 12, 3), 80, np.uint8), 11: np.full((4, 12, 3), 90, np.uint8)}
    result, audit = run_s13_m62(
        p2, images.__getitem__, execution_mode="candidate_single_pass",
        force_owner_only_pair_indices=frozenset({0}),
    )
    np.testing.assert_array_equal(result.visual_panorama, p2.result_image)
    assert result.performance["legacy_m6_call_count"] == 0
    assert result.performance["gpu_shadow_call_count"] == 0
    assert result.performance["pixel_executor_count"] == 0
    assert result.performance["m6_owner_source_remap_count"] == 0
    assert result.performance["final_linear_full_d2h_count"] == 0
    assert audit["plan"].q0_b0_direct_return is True
    json.dumps(result.performance)


def test_audit_modes_are_explicit_and_isolated(tmp_path: Path) -> None:
    p2 = _p2(tmp_path)
    images = {10: np.full((4, 12, 3), 80, np.uint8), 11: np.full((4, 12, 3), 90, np.uint8)}
    shadow, _ = run_s13_m62(
        p2, images.__getitem__, execution_mode="shadow_audit",
        force_owner_only_pair_indices=frozenset({0}),
    )
    assert shadow.performance["gpu_shadow_call_count"] == 1
    assert shadow.performance["legacy_m6_call_count"] == 0
    parity, _ = run_s13_m62(
        p2, images.__getitem__, execution_mode="parity_test",
        force_owner_only_pair_indices=frozenset({0}),
    )
    assert parity.performance["gpu_shadow_call_count"] == 1
    assert parity.performance["legacy_m6_call_count"] == 1


def test_m62_b1_input_gates_do_not_require_cuda_context() -> None:
    # Bind the pure input-validation part without constructing a CUDA runtime.
    class Runtime:
        apply_b1_secondary_weight_roi_device = S13CudaRuntime.apply_b1_secondary_weight_roi_device
    left = np.zeros((2, 3, 3), np.float32)
    try:
        Runtime().apply_b1_secondary_weight_roi_device(canvas_device=np.zeros((2, 3, 3), np.float32),
            x0=0, x1=2, left_device=left, right_device=left, primary_owner_right_mask=np.zeros((2, 3), bool),
            secondary_weight=np.zeros((2, 3), np.float32))
    except ValueError as exc:
        assert "x0/x1" in str(exc)
    else:
        raise AssertionError("off-by-one corridor must be rejected before CUDA use")


def test_m62_cuda_b1_persistent_canvas_handles_both_primary_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G305_CUDA", "required")
    if not cuda_status(refresh=True).available:
        pytest.skip("CUDA unavailable")
    runtime = S13CudaRuntime()
    try:
        canvas = runtime.new_linear_canvas(2, 4)
        left = np.full((2, 2, 3), 0.2, np.float32)
        right = np.full((2, 2, 3), 0.8, np.float32)
        first_owner_right = np.array([[False, True], [True, False]])
        weight = np.full((2, 2), 0.1, np.float32)
        runtime.apply_b1_secondary_weight_roi_device(
            canvas_device=canvas, x0=0, x1=2, left_device=left, right_device=right,
            primary_owner_right_mask=first_owner_right, secondary_weight=weight,
            protected_mask=np.zeros((2, 2), bool),
        )
        runtime.apply_b1_secondary_weight_roi_device(
            canvas_device=canvas, x0=2, x1=4, left_device=left, right_device=right,
            primary_owner_right_mask=~first_owner_right, secondary_weight=weight,
            protected_mask=np.zeros((2, 2), bool),
        )
        result = runtime.download_linear_canvas(canvas)
        expected_first = np.repeat(np.where(first_owner_right[..., None], 0.8 * 0.9 + 0.2 * 0.1, 0.2 * 0.9 + 0.8 * 0.1), 3, axis=2)
        expected_second = np.repeat(np.where((~first_owner_right)[..., None], 0.8 * 0.9 + 0.2 * 0.1, 0.2 * 0.9 + 0.8 * 0.1), 3, axis=2)
        np.testing.assert_allclose(result[:, :2], expected_first, atol=1e-6)
        np.testing.assert_allclose(result[:, 2:], expected_second, atol=1e-6)
    finally:
        runtime.close()
