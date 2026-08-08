from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from panorama_demo.video_v61_geometry_gate import (
    V61GeometryGateConfig,
    evaluate_v61_geometry_gate,
)
from panorama_demo.video_visual_renderer import VideoDISPairEvidence


_SHAPE = (480, 160)


def _evidence(
    shape: tuple[int, int],
    *,
    fb_error: float = 0.0,
    reliable_mask: np.ndarray | None = None,
) -> VideoDISPairEvidence:
    height, width = shape
    flow = np.zeros((height, width, 2), np.float32)
    zeros = np.zeros(shape, np.float32)
    reliable = (
        np.ones(shape, bool)
        if reliable_mask is None
        else np.asarray(reliable_mask, bool)
    )
    return VideoDISPairEvidence(
        flow,
        -flow,
        np.full(shape, fb_error, np.float32),
        zeros,
        zeros,
        np.zeros(shape, bool),
        reliable.astype(np.float32),
        reliable,
        np.zeros((height, width, 4), np.uint8),
    )


def _striped_pair() -> tuple[np.ndarray, np.ndarray]:
    old = np.zeros((*_SHAPE, 3), np.uint8)
    for column in range(10, 151, 20):
        cv2.line(old, (column, 0), (column, _SHAPE[0] - 1), (255, 255, 255), 1)
    return old, old.copy()


def _evaluate(
    old: np.ndarray,
    new: np.ndarray,
    evidence: VideoDISPairEvidence | None = None,
    *,
    protected: np.ndarray | None = None,
) -> object:
    shape = old.shape[:2]
    return evaluate_v61_geometry_gate(
        old,
        new,
        evidence or _evidence(shape),
        support=np.ones(shape, bool),
        protected=(
            np.zeros(shape, bool)
            if protected is None
            else np.asarray(protected, bool)
        ),
    )


def test_tail_outlier_is_guarded_without_being_an_absolute_max_veto() -> None:
    old, new = _striped_pair()
    # Move only a six-row fragment by two pixels.  The two edge-normal tail
    # samples exceed 1.25px, while thousands of unchanged samples keep P95 at
    # zero.  This is the exact distribution that v1.2 incorrectly rejected.
    new[200:206, 88:93] = 0
    cv2.line(new, (92, 200), (92, 205), (255, 255, 255), 1)
    audit = _evaluate(old, new)

    assert audit.accepted
    assert audit.edge_p95_px is not None and audit.edge_p95_px <= 0.75
    assert audit.edge_abs_max_px is not None and audit.edge_abs_max_px > 1.25
    assert audit.tail_count > 0
    assert audit.tail_risk_pixel_count > 0
    assert audit.tail_guard_incremental_pixel_count > 0
    assert np.any(audit.tail_guard)
    assert np.any(audit.residual_px[audit.residual_sample_mask] > 1.25)
    assert audit.as_report_dict()["edge_abs_max_gate_role"] == "telemetry_only"


def test_tail_guard_is_only_the_increment_beyond_existing_edge_guard() -> None:
    old, new = _striped_pair()
    new[200:206, 88:93] = 0
    cv2.line(new, (92, 200), (92, 205), (255, 255, 255), 1)
    first = _evaluate(old, new)
    # Pretend part of the two-way source/target band was already protected by
    # the ordinary Canny/object guard.  The runtime tail guard must contain
    # only the still-unprotected increment, while tail_risk_mask preserves the
    # complete auditable band.
    protected = np.zeros(old.shape[:2], bool)
    protected[:, :91] = True
    audit = _evaluate(old, new, protected=protected)

    assert audit.tail_guard_intersection > 0
    assert audit.tail_guard_incremental_pixel_count > 0
    assert not np.any(audit.tail_guard & protected)
    assert np.array_equal(audit.tail_risk_mask, first.tail_risk_mask)
    assert np.array_equal(
        audit.tail_guard,
        audit.tail_risk_mask & ~protected,
    )


def test_new_only_edge_is_seen_by_backward_evidence_and_fails_coverage() -> None:
    old = np.zeros((*_SHAPE, 3), np.uint8)
    new = old.copy()
    cv2.line(new, (80, 0), (80, _SHAPE[0] - 1), (255, 255, 255), 1)
    audit = _evaluate(old, new)

    assert audit.forward_edge_sample_count == 0
    assert audit.backward_edge_sample_count > 0
    assert audit.backward_matched_edge_count == 0
    assert audit.matched_edge_fraction == 0.0
    assert not audit.accepted
    assert "matched_edge_fraction" in str(audit.rejection_reason)


def test_geometry_gate_rejects_true_edge_p95_failure() -> None:
    old = np.zeros((*_SHAPE, 3), np.uint8)
    new = np.zeros_like(old)
    for column in range(10, 151, 20):
        cv2.line(old, (column, 0), (column, _SHAPE[0] - 1), (255, 255, 255), 1)
        cv2.line(new, (column + 2, 0), (column + 2, _SHAPE[0] - 1), (255, 255, 255), 1)
    audit = _evaluate(old, new)

    assert audit.matched_edge_fraction == 1.0
    assert audit.edge_p95_px is not None and audit.edge_p95_px > 0.75
    assert not audit.accepted
    assert "edge_p95" in str(audit.rejection_reason)


def test_geometry_gate_rejects_true_fb_p95_failure() -> None:
    old, new = _striped_pair()
    audit = _evaluate(old, new, _evidence(old.shape[:2], fb_error=2.0))

    assert not audit.accepted
    assert audit.fb_p95_px == pytest.approx(2.0)
    assert "fb_p95" in str(audit.rejection_reason)


def test_geometry_gate_rejects_insufficient_reliable_support() -> None:
    old, new = _striped_pair()
    reliable = np.zeros(old.shape[:2], bool)
    reliable[:10, :10] = True
    audit = _evaluate(
        old,
        new,
        _evidence(old.shape[:2], reliable_mask=reliable),
    )

    assert audit.reliable_pixel_count == 100
    assert not audit.accepted
    assert "minimum_reliable_pixels" in str(audit.rejection_reason)


def test_scalar_report_is_json_safe_and_excludes_runtime_arrays() -> None:
    old, new = _striped_pair()
    audit = _evaluate(old, new)

    payload = audit.as_report_dict()
    assert audit.report_dict() == payload
    assert "tail_guard" not in payload
    assert "tail_risk_mask" not in payload
    assert "residual_px" not in payload
    assert "residual_sample_mask" not in payload
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"minimum_reliable_pixels": 127}, "minimum_reliable_pixels"),
        ({"minimum_reliable_pixels": True}, "minimum_reliable_pixels"),
        ({"fb_p95_max_px": float("nan")}, "fb_p95_max_px"),
        ({"fb_p95_max_px": 1.26}, "fb_p95_max_px"),
        ({"edge_p95_max_px": 0.76}, "edge_p95_max_px"),
        ({"minimum_matched_edge_fraction": 0.84}, "minimum_matched_edge_fraction"),
        ({"tail_threshold_px": 1.24}, "tail_threshold_px"),
        ({"tail_dilation_px": 0}, "tail_dilation_px"),
        ({"edge_normal_search_px": 17}, "edge_normal_search_px"),
        ({"edge_normal_sample_step_px": 0.01}, "edge_normal_sample_step_px"),
        ({"edge_correspondence_band_px": 4.1}, "edge_correspondence_band_px"),
        ({"edge_orientation_tolerance_deg": 46.0}, "edge_orientation_tolerance_deg"),
        ({"maximum_edge_samples_per_direction": 127}, "maximum_edge_samples_per_direction"),
    ],
)
def test_geometry_config_rejects_invalid_or_relaxed_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        V61GeometryGateConfig(**overrides)


def test_geometry_gate_rejects_malformed_corridor_inputs() -> None:
    old, new = _striped_pair()
    with pytest.raises(ValueError, match="matching uint8 BGR"):
        evaluate_v61_geometry_gate(
            old.astype(np.float32),
            new.astype(np.float32),
            _evidence(old.shape[:2]),
            support=np.ones(old.shape[:2], bool),
            protected=np.zeros(old.shape[:2], bool),
        )
    with pytest.raises(ValueError, match="support and protected"):
        evaluate_v61_geometry_gate(
            old,
            new,
            _evidence(old.shape[:2]),
            support=np.ones((10, 10), bool),
            protected=np.zeros(old.shape[:2], bool),
        )
