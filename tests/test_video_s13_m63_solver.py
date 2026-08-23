from __future__ import annotations

from dataclasses import replace

import numpy as np

from panorama_demo.video_s13_m63_solver import (
    Q1R_MODEL,
    Q4C_MODEL,
    S13M63Config,
    solve_s13_m63_centered_logs,
    solve_s13_m63_photometric,
)
from panorama_demo.video_s13_photometric import S13PhotometricSampleSet


def _sample(
    pair_index: int,
    left_source: int,
    right_source: int,
    left_level: float,
    right_level: float,
    *,
    count: int = 512,
    outlier: bool = False,
) -> S13PhotometricSampleSet:
    rng = np.random.default_rng(1000 + pair_index)
    texture = rng.uniform(0.12, 0.52, (count, 1))
    left = np.repeat(texture * np.exp(left_level), 3, axis=1)
    right = np.repeat(texture * np.exp(right_level), 3, axis=1)
    if outlier:
        right = rng.uniform(0.02, 0.95, (count, 3))
    split = count // 2
    empty_mask = np.zeros((2, 2), bool)
    return S13PhotometricSampleSet(
        pair_index=pair_index,
        left_source_index=left_source,
        right_source_index=right_source,
        train_left_rgb_linear=left[:split].astype(np.float32),
        train_right_rgb_linear=right[:split].astype(np.float32),
        heldout_left_rgb_linear=left[split:].astype(np.float32),
        heldout_right_rgb_linear=right[split:].astype(np.float32),
        train_canvas_xy=np.zeros((split, 2), np.int32),
        heldout_canvas_xy=np.zeros((count - split, 2), np.int32),
        safe_mask=empty_mask,
        protected_mask=empty_mask,
        common_mask=empty_mask,
        safe_mask_sha256="",
        protected_mask_sha256="",
        edge_eligible=True,
        edge_kind="adjacent",
    )


def _chain(levels: np.ndarray) -> tuple[S13PhotometricSampleSet, ...]:
    return tuple(
        _sample(index, index, index + 1, float(levels[index]), float(levels[index + 1]))
        for index in range(len(levels) - 1)
    )


def test_q1r_long_chain_is_centered_and_has_no_partial_fallback():
    levels = np.linspace(-0.06, 0.06, 107)
    samples = _chain(levels)
    config = S13M63Config(enabled=True)
    logs, evidence = solve_s13_m63_centered_logs(107, samples, config)
    weights = np.maximum(evidence, 1.0)
    assert abs(float(np.average(logs, weights=weights))) < 1e-10
    assert np.ptp(logs) > 0.10
    result = solve_s13_m63_photometric(
        samples, samples, frame_ids=range(107), config=config
    )
    assert result.solution.model_family == Q1R_MODEL
    assert result.audit["source_fallback_count"] == 0
    assert all(item.fallback_reason is None for item in result.solution.source_parameters)
    gains = np.asarray([item.gain_bgr[0] for item in result.solution.source_parameters])
    assert gains.min() >= config.minimum_gain
    assert gains.max() <= config.maximum_gain


def test_q1r_uses_atomic_alpha_shrink_when_raw_solution_is_too_strong():
    levels = np.linspace(-0.12, 0.12, 21)
    samples = _chain(levels)
    result = solve_s13_m63_photometric(
        samples, samples, frame_ids=range(21), config=S13M63Config(enabled=True)
    )
    assert result.solution.model_family == Q1R_MODEL
    assert result.audit["selected_alpha"] in (0.5, 0.25, 0.125)
    audits = [
        item for item in result.solution.candidate_audits if item["model"] == Q1R_MODEL
    ]
    assert audits[0]["out_of_bounds_source_count"] > 0
    assert sum(item["selected"] is True for item in audits) == 1


def test_quality_cut_removes_outlier_and_q4c_is_component_atomic():
    levels = np.asarray([-0.02, -0.01, 0.0, 0.0, 0.01, 0.02])
    samples = list(_chain(levels))
    samples[2] = _sample(2, 2, 3, 0.0, 0.0, outlier=True)
    result = solve_s13_m63_photometric(
        samples,
        samples,
        frame_ids=range(6),
        config=S13M63Config(enabled=True),
    )
    assert result.quality_cut_pair_indices == (2,)
    assert result.solution.model_family == Q4C_MODEL
    gains = np.asarray([item.gain_bgr[0] for item in result.solution.source_parameters])
    assert gains[2] == 1.0
    assert gains[3] == 1.0
    assert result.audit["source_fallback_count"] == 0


def test_nonfinite_or_unusable_model_falls_back_as_one_whole_candidate():
    samples = _chain(np.zeros(5))
    result = solve_s13_m63_photometric(
        samples,
        samples,
        frame_ids=range(5),
        config=S13M63Config(enabled=True),
        force_identity=True,
    )
    assert result.solution.model_family == "Q0_identity"
    assert all(item.fallback_reason is None for item in result.solution.source_parameters)
    assert result.audit["source_fallback_count"] == 0


def test_q4c_refits_when_one_pair_regresses_but_is_not_component_worst() -> None:
    levels = np.asarray([-0.06, -0.04, -0.02, 0.0, 0.0, 0.02, 0.04, 0.06])
    samples = list(_chain(levels))
    samples[3] = _sample(3, 3, 4, 0.0, 0.0, outlier=True)
    inconsistent = samples[1]
    samples[1] = replace(
        inconsistent,
        heldout_right_rgb_linear=inconsistent.heldout_left_rgb_linear.copy(),
    )

    result = solve_s13_m63_photometric(
        samples,
        samples,
        frame_ids=range(8),
        config=S13M63Config(enabled=True),
    )

    assert result.quality_cut_pair_indices == (1, 3)
    assert result.solution.model_family == Q4C_MODEL
    pair = next(item for item in result.quality_rows if item.pair_index == 1)
    assert pair.reasons == ("candidate_regression",)
    assert result.audit["pair_nonregression_fraction"] == 1.0
