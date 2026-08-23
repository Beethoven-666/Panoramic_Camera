from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np
import pytest

import panorama_demo.video_s13_m5 as m5_module
from panorama_demo.video_s13_alignment import (
    filter_s13_correspondences_for_final_seam,
    reestimate_s13_final_corridor_alignment,
)
from panorama_demo.video_s13_m51_r2 import S13M51R2Config
from panorama_demo.session import CameraIntrinsics
from panorama_demo.video_s12_schedule import build_s012_schedule
from panorama_demo.video_s13_vertical import estimate_s13_vertical
from panorama_demo.video_s13_quality import (
    edge_registration_visual_suspect,
    pair_edge_registration_metrics,
    prepare_seam_structure,
)


def _config(**overrides: object) -> S13M51R2Config:
    values: dict[str, object] = {
        "enabled": True,
        "lk_error_sigma_multiplier": 3.0,
        "lk_error_mad_floor": 0.1,
        "lk_error_absolute_maximum": 5.0,
        "minimum_correspondence_count_for_error_filter": 4,
    }
    values.update(overrides)
    return S13M51R2Config(**values)


def test_v5_transaction_floats_are_canonical_before_hashing() -> None:
    transaction = {
        "schema": "gemini305-video-s13-m5-pair-transaction/v3",
        "transaction_id": "m5-pair-0000",
        "metrics": {
            "positive": 1.23456789,
            "numpy": np.float32(2.3456788),
            "negative_zero": -0.0,
            "nested": [0.123456789, None],
        },
    }

    canonical = m5_module._finalize_v5_transaction(transaction)

    assert canonical["metrics"] == {
        "positive": 1.234568,
        "numpy": 2.345679,
        "negative_zero": 0.0,
        "nested": [0.123457, None],
    }
    content = dict(canonical)
    digest = content.pop("result_stage_sha256")
    expected = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert digest == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_v5_transaction_rejects_nonfinite_float(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        m5_module._finalize_v5_transaction({"metric": value})


def test_lk_correspondences_reject_invalid_moved_targets_before_indexing(
    monkeypatch,
) -> None:
    left = np.zeros((12, 20, 3), dtype=np.uint8)
    right = left.copy()
    left_valid = np.ones((12, 20), dtype=bool)
    right_valid = np.ones((12, 20), dtype=bool)
    right_valid[5, 7] = False
    source = np.asarray(
        [[2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]], dtype=np.float32
    )
    moved = np.asarray(
        [[2, 2], [3, 3], [np.nan, 4], [20, 5], [6, 12], [7, 5]],
        dtype=np.float32,
    )
    status = np.asarray([[1], [0], [1], [1], [1], [1]], dtype=np.uint8)
    error = np.full((6, 1), 0.2, dtype=np.float32)
    calls = {"gftt": 0, "lk": 0}

    def fake_gftt(*_args, **_kwargs):
        calls["gftt"] += 1
        return source[:, None, :]

    def fake_lk(*_args, **_kwargs):
        calls["lk"] += 1
        return moved[:, None, :], status, error

    monkeypatch.setattr(m5_module.cv2, "goodFeaturesToTrack", fake_gftt)
    monkeypatch.setattr(m5_module.cv2, "calcOpticalFlowPyrLK", fake_lk)

    result = m5_module._pair_correspondences(
        left, right, left_valid, right_valid, x_offset=30, config=_config()
    )

    assert np.array_equal(result.reference_xy, np.asarray([[32.0, 2.0]]))
    assert np.array_equal(result.moving_xy, np.asarray([[32.0, 2.0]]))
    assert {name: result.audit[name] for name in (
        "detected_count", "status_count", "finite_count", "target_in_bounds_count",
        "target_valid_count", "error_filtered_count", "final_count", "error_median",
        "error_mad", "error_limit", "gftt_call_count", "forward_pyr_lk_call_count",
        "backward_pyr_lk_call_count",
    )} == {
        "detected_count": 6,
        "status_count": 5,
        "finite_count": 4,
        "target_in_bounds_count": 2,
        "target_valid_count": 1,
        "error_filtered_count": 1,
        "final_count": 1,
        "error_median": None,
        "error_mad": None,
        "error_limit": None,
        "gftt_call_count": 1,
        "forward_pyr_lk_call_count": 1,
        "backward_pyr_lk_call_count": 0,
    }
    assert calls == {"gftt": 1, "lk": 1}


def test_lk_error_filter_never_restores_rejected_tail(monkeypatch) -> None:
    image = np.zeros((20, 32, 3), dtype=np.uint8)
    valid = np.ones((20, 32), dtype=bool)
    source = np.column_stack(
        (np.arange(2, 10, dtype=np.float32), np.full(8, 5, dtype=np.float32))
    )
    moved = source + np.asarray((0.5, 0.0), dtype=np.float32)
    status = np.ones((8, 1), dtype=np.uint8)
    error = np.asarray([0.2, 0.2, 0.21, 0.19, 0.2, 20.0, 30.0, 40.0], np.float32)[:, None]
    monkeypatch.setattr(
        m5_module.cv2, "goodFeaturesToTrack", lambda *_a, **_k: source[:, None, :]
    )
    monkeypatch.setattr(
        m5_module.cv2,
        "calcOpticalFlowPyrLK",
        lambda *_a, **_k: (moved[:, None, :], status, error),
    )

    result = m5_module._pair_correspondences(
        image,
        image,
        valid,
        valid,
        x_offset=0,
        config=_config(minimum_correspondence_count_for_error_filter=4),
    )

    assert result.audit["target_valid_count"] == 8
    assert result.audit["error_filtered_count"] == 5
    assert result.audit["final_count"] == 5
    assert result.audit["error_limit"] < 1.0
    assert len(result.reference_xy) == 5


def test_disabled_lk_instrumentation_preserves_legacy_points_and_reports_error(
    monkeypatch,
) -> None:
    image = np.zeros((10, 16, 3), dtype=np.uint8)
    valid = np.ones((10, 16), dtype=bool)
    right_valid = valid.copy()
    right_valid[4, 4] = False
    source = np.asarray([[2, 2], [4, 4], [6, 6]], dtype=np.float32)
    moved = np.asarray([[2, 2], [4, 4], [50, 6]], dtype=np.float32)
    status = np.ones((3, 1), dtype=np.uint8)
    error = np.asarray([[0.1], [1.0], [10.0]], dtype=np.float32)
    monkeypatch.setattr(
        m5_module.cv2, "goodFeaturesToTrack", lambda *_a, **_k: source[:, None, :]
    )
    monkeypatch.setattr(
        m5_module.cv2,
        "calcOpticalFlowPyrLK",
        lambda *_a, **_k: (moved[:, None, :], status, error),
    )

    result = m5_module._pair_correspondences(
        image, image, valid, right_valid, x_offset=10, config=S13M51R2Config()
    )

    # The disabled instrumentation path deliberately preserves historic
    # status+finite decisions, including an invalid moved target.
    assert len(result.reference_xy) == 3
    assert np.array_equal(result.reference_xy[:, 0], source[:, 0] + 10)
    assert result.audit["legacy_decision_preserved"] is True
    assert result.audit["filter_enabled"] is False
    assert result.audit["target_valid_count"] == 1
    assert result.audit["raw_error_count"] == 3
    assert result.audit["raw_error_median"] == 1.0
    assert result.audit["raw_error_mad"] == pytest.approx(0.9)
    assert result.audit["raw_error_p95"] is not None
    assert result.audit["raw_error_maximum"] == 10.0


def test_seam_local_filter_uses_each_points_own_row_and_selects_16px() -> None:
    seam = 50 + np.arange(64, dtype=np.int32) // 8
    rows = np.arange(40, dtype=np.float64) % 64
    reference = np.column_stack((seam[rows.astype(np.int32)] - 12.0, rows))
    moving_rows = (rows + 3.0) % 64
    moving = np.column_stack((seam[moving_rows.astype(np.int32)] + 14.0, moving_rows))

    selected_reference, selected_moving, audit = (
        filter_s13_correspondences_for_final_seam(
            reference,
            moving,
            seam,
            evidence_half_widths_px=(16, 24),
            moving_margin_px=0,
            minimum_required_points=36,
        )
    )

    assert len(selected_reference) == len(selected_moving) == 40
    assert audit["selected_half_width_px"] == 16
    assert audit["mode"] == "seam_local"
    assert audit["full_shoulder_fallback_used"] is False


def test_seam_local_filter_expands_only_to_24px() -> None:
    seam = np.full(64, 60, dtype=np.int32)
    rows = np.arange(40, dtype=np.float64) % 64
    reference = np.column_stack((np.full(40, 40.0), rows))
    moving = np.column_stack((np.full(40, 80.0), rows))

    selected_reference, _selected_moving, audit = (
        filter_s13_correspondences_for_final_seam(
            reference, moving, seam, minimum_required_points=36
        )
    )

    assert len(selected_reference) == 40
    assert audit["selected_half_width_px"] == 24
    assert audit["requested_half_widths_px"] == [16, 24]
    assert audit["wide_evidence"] is True
    assert audit["full_shoulder_fallback_used"] is False


def test_seam_local_filter_reports_insufficient_without_full_shoulder_fallback() -> None:
    seam = np.full(64, 60, dtype=np.int32)
    rows = np.arange(50, dtype=np.float64) % 64
    reference = np.column_stack((np.full(50, 30.0), rows))
    moving = np.column_stack((np.full(50, 90.0), rows))

    selected_reference, selected_moving, audit = (
        filter_s13_correspondences_for_final_seam(
            reference, moving, seam, minimum_required_points=36
        )
    )

    assert selected_reference.shape == selected_moving.shape == (0, 2)
    assert audit["selected_half_width_px"] is None
    assert audit["sufficient_for_train_held_out"] is False
    assert audit["mode"] == "seam_local_insufficient"
    assert audit["full_shoulder_fallback_used"] is False


def test_final_corridor_reestimate_fits_only_seam_local_evidence() -> None:
    height, width = 64, 128
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    valid = np.ones((height, width), dtype=bool)
    seam = np.full(height, 64, dtype=np.int32)
    rows = np.arange(60, dtype=np.float64) % height
    near = np.column_stack((50.0 + rows % 5.0, rows))
    far = np.column_stack((26.0 + rows % 5.0, rows))
    reference = np.concatenate((near, far))
    moving = np.concatenate((near + (1.0, 0.0), far + (3.0, 0.0)))

    alignment = reestimate_s13_final_corridor_alignment(
        final_seam_x_by_row=seam,
        application_half_width_px=4,
        pair_index=0,
        pair_frame_ids=(1, 2),
        non_reference_side="right",
        p0_source_u=u,
        p0_source_v=v,
        p0_valid=valid,
        source_size=(width, height),
        reference_points_xy=reference,
        non_reference_points_xy=moving,
        alignment_shoulder=(16, 112),
        m51_r2_config=_config(),
    )

    assert alignment.seam_local_evidence["selected_half_width_px"] == 16
    assert alignment.seam_local_evidence["selected_count"] == 60
    translation = next(
        item for item in alignment.candidates if item.model == "C2_subpixel_translation"
    )
    assert translation.audit["translation_px"] < 1.1
    assert alignment.application_band.right_x_by_row.max() - seam.max() <= 5
    assert alignment.reestimated_for_final_seam is True


# Edge-normal registration tests intentionally use the two real source views,
# not an owner-composed preview.  They are isolated from the M5 orchestration
# tests above so their failures identify the quality primitive directly.
def _edge_pair(
    angle_degrees: float,
    shift_px: float,
    *,
    height: int = 96,
    width: int = 96,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.zeros((height, width, 3), dtype=np.uint8)
    center = np.asarray((width / 2.0, height / 2.0), dtype=np.float64)
    direction = np.asarray(
        (np.cos(np.deg2rad(angle_degrees)), np.sin(np.deg2rad(angle_degrees))),
        dtype=np.float64,
    )
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    half_length = 0.8 * max(height, width)

    def draw(image: np.ndarray) -> None:
        midpoint = center
        p0 = tuple(np.rint(midpoint - half_length * direction).astype(np.int32))
        p1 = tuple(np.rint(midpoint + half_length * direction).astype(np.int32))
        cv2.line(image, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)

    draw(left)
    transform = np.asarray(
        [[1.0, 0.0, shift_px * normal[0]], [0.0, 1.0, shift_px * normal[1]]],
        dtype=np.float32,
    )
    right = cv2.warpAffine(
        left, transform, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = np.ones((height, width), dtype=bool)
    seam = np.full(height, width // 2, dtype=np.int32)
    return left, right, valid, valid.copy(), seam


def test_pair_edge_registration_detects_oblique_normal_shift() -> None:
    left, right, left_valid, right_valid, seam = _edge_pair(30.0, 2.0)

    metrics = pair_edge_registration_metrics(
        left, right, left_valid, right_valid, seam, config=_config()
    )

    assert metrics["schema"] == "gemini305-video-s13-oblique-structure-audit/v1"
    assert metrics["evaluable"] is True
    assert metrics["supported_block_count"] >= 1
    assert 1.0 <= metrics["median_supported_abs_lag_px"] <= 3.0
    assert metrics["p95_supported_abs_lag_px"] >= 1.0


def test_pair_edge_registration_near_vertical_edge_searches_horizontal_normal() -> None:
    left, right, left_valid, right_valid, seam = _edge_pair(89.0, 2.0)

    metrics = pair_edge_registration_metrics(
        prepare_seam_structure(left),
        prepare_seam_structure(right),
        left_valid,
        right_valid,
        seam,
        config=_config(),
    )

    assert metrics["evaluable"] is True
    assert metrics["median_supported_abs_lag_px"] >= 1.0


def test_pair_edge_registration_weak_texture_is_neutral() -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    valid = np.ones((96, 96), dtype=bool)
    seam = np.full(96, 48, dtype=np.int32)

    metrics = pair_edge_registration_metrics(
        image, image, valid, valid, seam, config=_config()
    )

    assert metrics["evaluable"] is False
    assert metrics["supported_block_count"] == 0
    assert metrics["median_supported_abs_lag_px"] is None


def test_pair_edge_registration_is_black_safe_and_respects_valid_masks() -> None:
    left, right, left_valid, right_valid, seam = _edge_pair(45.0, 0.0)
    metrics = pair_edge_registration_metrics(
        left, right, left_valid, right_valid, seam, config=_config()
    )
    assert metrics["evaluable"] is True

    right_valid[:] = False
    excluded = pair_edge_registration_metrics(
        left, right, left_valid, right_valid, seam, config=_config()
    )
    assert excluded["evaluable"] is False
    assert excluded["supported_block_count"] == 0


@pytest.mark.parametrize("angle_degrees", [0.0, 15.0, 45.0, 75.0])
def test_pair_edge_registration_covers_multiple_edge_orientations(
    angle_degrees: float,
) -> None:
    left, right, left_valid, right_valid, seam = _edge_pair(angle_degrees, 2.0)

    metrics = pair_edge_registration_metrics(
        left, right, left_valid, right_valid, seam, config=_config()
    )

    assert metrics["evaluable"] is True
    assert metrics["p95_supported_abs_lag_px"] >= 1.0


def test_pair_edge_registration_uses_overlapping_32_row_blocks() -> None:
    height, width = 96, 96
    left = np.zeros((height, width, 3), dtype=np.uint8)
    # The edge crosses y=32, exactly where a non-overlapping implementation
    # would split its support into two insufficient fragments.
    cv2.line(left, (38, 26), (58, 38), (255, 255, 255), 2, cv2.LINE_AA)
    transform = np.asarray([[1.0, 0.0, -1.0], [0.0, 1.0, 1.75]], np.float32)
    right = cv2.warpAffine(left, transform, (width, height), flags=cv2.INTER_LINEAR)
    valid = np.ones((height, width), dtype=bool)
    seam = np.full(height, 48, dtype=np.int32)

    metrics = pair_edge_registration_metrics(
        left, right, valid, valid, seam, config=_config()
    )

    assert metrics["evaluable"] is True
    assert any(
        row["supported"] and row["y0"] <= 32 < row["y1"]
        for row in metrics["block_audits"]
    )


def test_pair_edge_registration_marks_competing_layers_ambiguous() -> None:
    left = np.zeros((96, 96, 3), dtype=np.uint8)
    for offset in (-4, 4):
        cv2.line(left, (32, 20 + offset), (64, 76 + offset), (255, 255, 255), 2, cv2.LINE_AA)
    right = left.copy()
    valid = np.ones((96, 96), dtype=bool)
    seam = np.full(96, 48, dtype=np.int32)

    metrics = pair_edge_registration_metrics(
        left, right, valid, valid, seam, config=_config()
    )

    assert metrics["multiple_layer_or_ambiguous"] is True


def test_pair_edge_registration_rejects_invalid_coordinate_domains() -> None:
    left, right, left_valid, right_valid, seam = _edge_pair(30.0, 0.0)

    with pytest.raises(ValueError, match="valid mask shape"):
        pair_edge_registration_metrics(
            left, right, left_valid[:-1], right_valid, seam, config=_config()
        )
    with pytest.raises(ValueError, match="integral"):
        pair_edge_registration_metrics(
            left, right, left_valid, right_valid, seam.astype(np.float64) + 0.25,
            config=_config(),
        )


@pytest.mark.parametrize("angle_degrees", [0.0, 15.0, 30.0, 45.0, 75.0, 89.0])
@pytest.mark.parametrize("shift_px", [0.0, 0.5, 1.0, 2.0, 3.0])
def test_pair_edge_registration_orientation_and_subpixel_shift_matrix(
    angle_degrees: float,
    shift_px: float,
) -> None:
    left, right, left_valid, right_valid, seam = _edge_pair(angle_degrees, shift_px)

    metrics = pair_edge_registration_metrics(
        left, right, left_valid, right_valid, seam, config=_config()
    )

    if shift_px == 0.0:
        if metrics["evaluable"] is True:
            assert metrics["median_supported_abs_lag_px"] <= 0.75
        assert edge_registration_visual_suspect(
            metrics, seam_local_lk_p95_px=0.0, config=_config()
        ) is False
    else:
        assert metrics["evaluable"] is True
        assert metrics["median_supported_abs_lag_px"] is not None
    if shift_px == 2.0:
        assert metrics["p95_supported_abs_lag_px"] > 1.0
        assert edge_registration_visual_suspect(
            metrics, seam_local_lk_p95_px=0.0, config=_config()
        ) is True


def test_edge_registration_visual_suspect_requires_unambiguous_support() -> None:
    clean = {
        "evaluable": True,
        "multiple_layer_or_ambiguous": False,
        "median_supported_abs_lag_px": 0.5,
        "p95_supported_abs_lag_px": 0.75,
    }
    assert edge_registration_visual_suspect(
        clean, seam_local_lk_p95_px=1.25, config=_config()
    ) is False
    assert edge_registration_visual_suspect(
        clean, seam_local_lk_p95_px=1.251, config=_config()
    ) is True
    assert edge_registration_visual_suspect(
        {**clean, "median_supported_abs_lag_px": 0.751},
        seam_local_lk_p95_px=None,
        config=_config(),
    ) is True
    assert edge_registration_visual_suspect(
        {**clean, "p95_supported_abs_lag_px": 1.001},
        seam_local_lk_p95_px=None,
        config=_config(),
    ) is True
    assert edge_registration_visual_suspect(
        {**clean, "evaluable": False, "p95_supported_abs_lag_px": 3.0},
        seam_local_lk_p95_px=3.0,
        config=_config(),
    ) is False
    assert edge_registration_visual_suspect(
        {**clean, "multiple_layer_or_ambiguous": True, "p95_supported_abs_lag_px": 3.0},
        seam_local_lk_p95_px=3.0,
        config=_config(),
    ) is False


def _small_m5_inputs():
    calibration = CameraIntrinsics(96, 64, 80.0, 80.0, 47.5, 31.5, ())
    schedule = build_s012_schedule(
        (0, 1, 2), (47.5, 55.5, 63.5), calibration, hard_internal_width_px=None
    )
    rng = np.random.default_rng(731)
    base = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
    images = {0: base, 1: np.roll(base, 1, axis=1), 2: np.roll(base, 2, axis=1)}
    vertical = estimate_s13_vertical(schedule, calibration, images.__getitem__)
    return calibration, schedule, images, vertical


def test_v5_transactions_use_v3_structure_fields() -> None:
    calibration, schedule, images, vertical = _small_m5_inputs()
    pairs = m5_module.estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="a" * 64, m51_r2_config=_config(),
    )
    required = {
        "correspondence_filter", "seam_local_evidence", "edge_registration",
        "selection_continued_for_structure", "selected_seam_rank",
        "selected_geometry_rank", "seam_fallback_used", "geometry_fallback_used",
        "topology_repair_used", "unresolved_oblique_structure",
        "unexpected_exception_fallback", "micro_rescue",
    }
    for pair in pairs:
        assert pair.transaction["schema"] == "gemini305-video-s13-m5-pair-transaction/v3"
        assert required <= set(pair.transaction)
        assert pair.transaction["unexpected_exception_fallback"] is False
        content = dict(pair.transaction)
        digest = content.pop("result_stage_sha256")
        assert digest == hashlib.sha256(
            json.dumps(
                content,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        pending = [content]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, float):
                assert value == round(value, 6)
                assert value != 0.0 or np.signbit(value) is np.False_


def test_v5_unexpected_program_error_is_not_converted_to_fallback(monkeypatch) -> None:
    calibration, schedule, images, vertical = _small_m5_inputs()
    monkeypatch.setattr(
        m5_module.cv2, "cvtColor",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("program bug")),
    )
    with pytest.raises(AssertionError, match="program bug"):
        m5_module.estimate_s13_m5_transactions(
            schedule, calibration, images.__getitem__, vertical,
            parent_stage_sha256="b" * 64, m51_r2_config=_config(),
        )


def test_v5_clean_first_candidate_keeps_early_stop() -> None:
    calibration, schedule, images, vertical = _small_m5_inputs()
    pairs = m5_module.estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="c" * 64, m51_r2_config=_config(),
    )
    for pair in pairs:
        assert pair.transaction["selection_continued_for_structure"] is False
        evaluations = pair.transaction["candidate_evaluations"]
        assert evaluations[0]["evaluation_status"] == "selected"
        assert all(
            row["evaluation_status"] == "skipped_after_higher_rank_safe_candidate"
            for row in evaluations[1:]
        )


def test_v5_all_suspect_candidates_continue_then_keep_hard_safe_baseline(
    monkeypatch,
) -> None:
    calibration, schedule, images, vertical = _small_m5_inputs()
    monkeypatch.setattr(
        m5_module, "edge_registration_visual_suspect", lambda *_a, **_k: True
    )
    pairs = m5_module.estimate_s13_m5_transactions(
        schedule, calibration, images.__getitem__, vertical,
        parent_stage_sha256="d" * 64, m51_r2_config=_config(),
    )
    for pair in pairs:
        assert pair.transaction["selection_continued_for_structure"] is True
        assert pair.transaction["unresolved_oblique_structure"] is True
        assert pair.transaction["decision"] == "applied"
        assert all(
            row["evaluation_status"] != "skipped_after_higher_rank_safe_candidate"
            for row in pair.transaction["candidate_evaluations"]
        )
