from __future__ import annotations

from pathlib import Path

import numpy as np

from panorama_demo.video_s13_m6 import run_s13_m6
from panorama_demo.video_s13_m62_equivalence import compare_s13_m62_u8
from panorama_demo.video_s13_m62_plan import build_s13_m62_execution_plan
from panorama_demo.video_s13_m62_reference import execute_s13_m62_cpu_reference
from panorama_demo.video_s13_cuda_runtime import S13CudaRuntime
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
    assert plan.corrected_rois[0].flags.writeable is False
    assert plan.blend_plans[0].secondary_weight.flags.writeable is False
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
