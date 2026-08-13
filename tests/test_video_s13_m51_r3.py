from __future__ import annotations

from dataclasses import asdict

import cv2
import numpy as np

from panorama_demo.video_s13_m51_r2 import S13M51R2Config, S13M51R3Config
from panorama_demo.video_s13_quality import (
    edge_registration_visual_suspect,
    pair_edge_registration_metrics,
)


def _config_v5() -> S13M51R2Config:
    return S13M51R2Config(
        enabled=True,
        lk_error_mad_floor=0.1,
        lk_error_absolute_maximum=5.0,
    )


def _config_v6() -> S13M51R3Config:
    return S13M51R3Config(
        enabled=True,
        lk_error_mad_floor=0.1,
        lk_error_absolute_maximum=5.0,
    )


def _shifted_line(
    p0: tuple[int, int],
    p1: tuple[int, int],
    shift: tuple[float, float],
    *,
    height: int = 96,
    width: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.zeros((height, width, 3), np.uint8)
    cv2.line(left, p0, p1, (255, 255, 255), 2, cv2.LINE_AA)
    transform = np.asarray(
        [[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], np.float32
    )
    right = cv2.warpAffine(left, transform, (width, height), flags=cv2.INTER_LINEAR)
    return left, right


def _metrics(
    left: np.ndarray,
    right: np.ndarray,
    config: S13M51R2Config,
) -> dict[str, object]:
    valid = np.ones(left.shape[:2], bool)
    seam = np.full(left.shape[0], left.shape[1] // 2, np.int32)
    return pair_edge_registration_metrics(
        left, right, valid, valid, seam, config=config
    )


def test_v6_deduplicates_one_physical_edge_seen_by_overlapping_blocks() -> None:
    left, right = _shifted_line((28, 8), (68, 88), (-1.8, 0.9))

    metrics = _metrics(left, right, _config_v6())

    assert metrics["evaluable"] is True
    assert metrics["supported_block_count"] >= 2
    assert metrics["supported_component_count"] == 1
    assert len(metrics["component_audits"]) == 1
    assert metrics["component_audits"][0]["observation_count"] >= 2


def test_v6_ambiguous_component_does_not_mask_clean_high_lag_component() -> None:
    left = np.zeros((128, 96, 3), np.uint8)
    # Top: two competing layers. Bottom: one separate, unambiguous edge.
    cv2.line(left, (32, 8), (50, 48), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(left, (43, 8), (61, 48), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(left, (32, 78), (62, 122), (255, 255, 255), 2, cv2.LINE_AA)
    transform = np.asarray([[1.0, 0.0, -1.7], [0.0, 1.0, 1.1]], np.float32)
    right = cv2.warpAffine(left, transform, (96, 128), flags=cv2.INTER_LINEAR)

    metrics = _metrics(left, right, _config_v6())

    assert metrics["ambiguous_component_count"] >= 1
    assert metrics["actionable_component_count"] >= 1
    assert metrics["p95_supported_abs_lag_px"] > 1.0
    assert edge_registration_visual_suspect(
        metrics, seam_local_lk_p95_px=0.0, config=_config_v6()
    ) is True


def test_v5_config_and_metric_path_remain_exactly_unchanged() -> None:
    expected_config = {
        "enabled": True,
        "collect_lk_error_statistics": True,
        "lk_error_sigma_multiplier": 3.0,
        "lk_error_mad_floor": 0.1,
        "lk_error_absolute_maximum": 5.0,
        "minimum_correspondence_count_for_error_filter": 16,
        "evidence_half_widths_px": (16, 24),
        "moving_evidence_margin_px": 8,
        "block_height_px": 32,
        "block_stride_px": 16,
        "seam_support_radius_px": 6,
        "feature_halo_radius_px": 12,
        "normal_search_maximum_px": 3.0,
        "normal_search_step_px": 0.5,
        "minimum_edge_component_length_px": 12,
        "minimum_edge_support_count": 12,
        "minimum_edge_correlation": 0.65,
        "minimum_uniqueness_fraction": 0.1,
        "maximum_orientation_difference_degrees": 10.0,
        "suspect_lk_p95_px": 1.25,
        "suspect_edge_median_px": 0.75,
        "suspect_edge_p95_px": 1.0,
        "equivalent_edge_p95_tolerance_px": 0.15,
        "micro_rescue_enabled": False,
    }
    config = _config_v5()
    left, right = _shifted_line((28, 8), (68, 88), (-1.8, 0.9))

    metrics = _metrics(left, right, config)

    assert asdict(config) == expected_config
    assert "component_audits" not in metrics
    assert "actionable_component_count" not in metrics
    assert metrics["schema"] == "gemini305-video-s13-oblique-structure-audit/v1"

