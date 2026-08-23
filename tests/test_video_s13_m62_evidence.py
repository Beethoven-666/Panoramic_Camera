from __future__ import annotations

import cv2
import numpy as np
import pytest

from panorama_demo.video_s13_m62_evidence import (
    S13M62EvidenceUse,
    S13M62PackedEvidencePlanError,
    build_s13_m62_packed_source_plan,
    sample_s13_m62_packed_source,
)


def _map_tile(height: int, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.broadcast_to(np.asarray(x_values, np.float32)[None, :], (height, len(x_values))).copy()
    v = np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], u.shape).copy()
    return u, v


def test_packed_source_samples_exact_original_tiles_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.arange(6 * 9 * 3, dtype=np.uint8).reshape(6, 9, 3)
    specifications = (
        ("bridge", 4, "right", 7, np.asarray([-1.0, 0.25, 8.0, 10.0])),
        ("adjacent", 3, "right", 2, np.asarray([6.5, 7.5])),
        ("adjacent", 2, "left", 1, np.asarray([0.0, 1.0, 2.0])),
        ("bridge", 4, "left", 7, np.asarray([2.25, 3.75, 4.5])),
    )
    uses = []
    for kind, logical_index, side, canvas_x0, x_values in specifications:
        map_u, map_v = _map_tile(6, x_values)
        uses.append(S13M62EvidenceUse(
            (kind, logical_index, side),
            kind,
            logical_index,
            side,
            5,
            42,
            canvas_x0,
            map_u,
            map_v,
        ))
    plan = build_s13_m62_packed_source_plan(uses)
    calls = 0
    original = cv2.remap

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "remap", counted)
    packed = sample_s13_m62_packed_source(raw, plan)
    assert calls == 1
    assert [item.usage_id for item in plan.placements] == [
        ("adjacent", 2, "left"),
        ("adjacent", 3, "right"),
        ("bridge", 4, "left"),
        ("bridge", 4, "right"),
    ]
    for usage in uses:
        expected = original(
            raw,
            usage.map_u,
            usage.map_v,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        np.testing.assert_array_equal(packed[usage.usage_id], expected)


def test_packed_source_rejects_frame_or_shape_invariants() -> None:
    map_u, map_v = _map_tile(4, np.asarray([0.0, 1.0]))
    first = S13M62EvidenceUse(
        ("adjacent", 0, "left"), "adjacent", 0, "left",
        0, 10, 0, map_u, map_v,
    )
    different_frame = S13M62EvidenceUse(
        ("adjacent", 0, "right"), "adjacent", 0, "right",
        0, 11, 0, map_u, map_v,
    )
    with pytest.raises(S13M62PackedEvidencePlanError, match="multiple frame"):
        build_s13_m62_packed_source_plan((first, different_frame))

    mismatched = S13M62EvidenceUse(
        ("bridge", 0, "left"), "bridge", 0, "left",
        0, 10, 0, map_u, map_v[:, :1],
    )
    with pytest.raises(S13M62PackedEvidencePlanError, match="shape mismatch"):
        build_s13_m62_packed_source_plan((mismatched,))


def test_pair_reference_and_packed_plan_evidence_are_exact(tmp_path) -> None:
    from test_video_s13_m62 import _p2
    from panorama_demo.video_s13_m62_plan import build_s13_m62_execution_plan

    p2 = _p2(tmp_path)
    images = {
        10: np.full((4, 12, 3), 80, np.uint8),
        11: np.full((4, 12, 3), 90, np.uint8),
    }
    reference = build_s13_m62_execution_plan(
        p2,
        images.__getitem__,
        force_owner_only_pair_indices=frozenset({0}),
    )
    packed = build_s13_m62_execution_plan(
        p2,
        images.__getitem__,
        force_owner_only_pair_indices=frozenset({0}),
        evidence_sampling_mode="source_packed_cv2",
    )

    assert packed.photometric_solution == reference.photometric_solution
    assert len(packed.evidence.adjacent_samples) == len(reference.evidence.adjacent_samples)
    for actual, expected in zip(
        packed.evidence.adjacent_samples,
        reference.evidence.adjacent_samples,
        strict=True,
    ):
        for name in (
            "train_left_rgb_linear",
            "train_right_rgb_linear",
            "heldout_left_rgb_linear",
            "heldout_right_rgb_linear",
            "train_canvas_xy",
            "heldout_canvas_xy",
            "safe_mask",
            "protected_mask",
            "common_mask",
            "left_linear_corridor",
            "right_linear_corridor",
        ):
            np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
        assert actual.gradient_limit == expected.gradient_limit
        assert actual.residual_limit == expected.residual_limit
        assert actual.edge_eligible == expected.edge_eligible
    profile = packed.evidence.performance
    assert profile["m62_evidence_reference_adjacent_remap_count"] == 0
    assert profile["m62_evidence_packed_source_count"] == 2
    assert profile["m62_evidence_packed_remap_count"] == 2
    assert profile["m62_evidence_adjacent_usage_count"] == 2
    assert profile["m62_evidence_fallback_count"] == 0


def test_packed_plan_error_falls_back_only_to_pair_reference(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import panorama_demo.video_s13_m62_evidence as evidence_module
    from test_video_s13_m62 import _p2
    from panorama_demo.video_s13_m62_plan import build_s13_m62_execution_plan

    p2 = _p2(tmp_path)
    images = {
        10: np.full((4, 12, 3), 80, np.uint8),
        11: np.full((4, 12, 3), 90, np.uint8),
    }

    def fail_plan(_uses):
        raise S13M62PackedEvidencePlanError("forced_shape_mismatch")

    monkeypatch.setattr(
        evidence_module, "build_s13_m62_packed_source_plan", fail_plan
    )
    result = build_s13_m62_execution_plan(
        p2,
        images.__getitem__,
        force_owner_only_pair_indices=frozenset({0}),
        evidence_sampling_mode="source_packed_cv2",
    )

    profile = result.evidence.performance
    assert profile["m62_evidence_fallback_count"] == 1
    assert profile["m62_evidence_reference_adjacent_remap_count"] == 2
    assert profile["m62_evidence_packed_remap_count"] == 0
    assert profile["m62_evidence_fallback_reasons"] == {"forced_shape_mismatch": 1}
