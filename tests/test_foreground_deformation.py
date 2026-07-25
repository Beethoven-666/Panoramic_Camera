from __future__ import annotations

import numpy as np
import pytest

from panorama_demo.foreground_deformation import (
    ForegroundDeformationExperimentConfig,
    ForegroundTrackEvidence,
    attempt_foreground_deformation,
)
from panorama_demo.synthetic import generate_foreground_deformation_hose_pair


def _track(**overrides: object) -> ForegroundTrackEvidence:
    values: dict[str, object] = {
        "track_id": 7,
        "association_score": 0.99,
        "one_to_one": True,
        "no_split_merge": True,
        "complete_source_coverage": True,
        "bidirectional_visibility": True,
        "contour_correspondence": True,
        "centreline_correspondence": True,
        "no_real_joint": True,
        "no_object_endpoint": True,
        "no_occlusion_or_disocclusion": True,
        "native_resolution": True,
    }
    values.update(overrides)
    return ForegroundTrackEvidence(**values)  # type: ignore[arg-type]


def _hose_pair(
    *, offset_pixels: float = 1.2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return generate_foreground_deformation_hose_pair(offset_pixels=offset_pixels)


def test_default_experiment_is_disabled_and_only_exposes_enabled() -> None:
    config = ForegroundDeformationExperimentConfig.from_mapping({"enabled": False})

    assert config.enabled is False
    assert config.analysis_corridor_width_pixels == 128
    with pytest.raises(ValueError, match="only accepts enabled"):
        ForegroundDeformationExperimentConfig.from_mapping({"maximum_displacement_pixels": 9})
    with pytest.raises(ValueError, match="boolean"):
        ForegroundDeformationExperimentConfig.from_mapping({"enabled": 1})


def test_accepts_native_one_to_two_pixel_hose_residual_and_keeps_outside_identity() -> None:
    first, second, mask, second_mask = _hose_pair()

    result = attempt_foreground_deformation(
        first,
        second,
        mask,
        _track(),
        config=ForegroundDeformationExperimentConfig(enabled=True),
        source_foreground_mask=second_mask,
    )

    audit = result.as_dict()
    assert result.accepted
    assert audit["held_out_error_p95_before_pixels"] > 0.75
    assert audit["held_out_error_p95_after_pixels"] <= 0.75
    assert audit["held_out_error_max_after_pixels"] <= 2.0
    assert audit["held_out_improvement_ratio"] >= 0.30
    assert audit["maximum_displacement_pixels"] <= 2.0
    assert 0.95 <= audit["local_scale_min"] <= audit["local_scale_max"] <= 1.05
    assert audit["local_jacobian_determinant_min"] > 0.0
    assert not np.any(result.active_mask & ~mask)
    assert np.array_equal(result.warped_source_bgr[~mask], second[~mask])
    grid_x = np.arange(first.shape[1], dtype=np.float32)[None, :]
    grid_y = np.arange(first.shape[0], dtype=np.float32)[:, None]
    assert np.allclose(result.inverse_map_x[~mask], np.broadcast_to(grid_x, mask.shape)[~mask])
    assert np.allclose(result.inverse_map_y[~mask], np.broadcast_to(grid_y, mask.shape)[~mask])
    assert audit["boundary_pinned_zero_displacement"] is True
    assert audit["color_generation_detected"] is False
    assert audit["global_flow_or_apap_used"] is False
    assert audit["alpha_blend_pixel_count"] == 0
    assert audit["multiband_pixel_count"] == 0


def test_no_measurable_native_seam_is_rejected_without_a_warp() -> None:
    first, _second, mask, second_mask = _hose_pair(offset_pixels=0.0)

    result = attempt_foreground_deformation(
        first,
        first,
        mask,
        _track(),
        config=ForegroundDeformationExperimentConfig(enabled=True),
        source_foreground_mask=second_mask,
    )

    assert not result.accepted
    assert result.reason == "no_measurable_foreground_seam_residual"
    assert not np.any(result.active_mask)
    assert np.array_equal(result.warped_source_bgr, first)
    assert np.allclose(result.inverse_map_x, np.arange(first.shape[1])[None, :])


def test_protected_coupler_or_incomplete_track_falls_back_as_a_whole() -> None:
    first, second, mask, second_mask = _hose_pair()
    coupler = np.zeros_like(mask)
    coupler[70:90, 70:90] = True
    protected = attempt_foreground_deformation(
        first,
        second,
        mask,
        _track(),
        config=ForegroundDeformationExperimentConfig(enabled=True),
        source_foreground_mask=second_mask,
        protected_mask=coupler,
    )
    incomplete = attempt_foreground_deformation(
        first,
        second,
        mask,
        _track(no_split_merge=False),
        config=ForegroundDeformationExperimentConfig(enabled=True),
        source_foreground_mask=second_mask,
    )

    assert not protected.accepted
    assert protected.reason == "foreground_instance_intersects_protected_domain"
    assert not np.any(protected.active_mask)
    assert not incomplete.accepted
    assert incomplete.reason == "track_split_merge_or_nonunique_association"
    assert np.array_equal(incomplete.warped_source_bgr, second)


def test_occlusion_transparent_band_and_non_crossing_instance_are_rejected() -> None:
    first, second, mask, second_mask = _hose_pair()
    transparent_band = np.zeros_like(mask)
    transparent_band[:, 78:83] = True
    occluded_source_mask = second_mask.copy()
    occluded_source_mask[68:92, 72:92] = False
    non_crossing = mask.copy()
    non_crossing[:, 80:] = False

    transparent = attempt_foreground_deformation(
        first,
        second,
        mask,
        _track(),
        config=ForegroundDeformationExperimentConfig(enabled=True),
        source_foreground_mask=second_mask,
        protected_mask=transparent_band,
    )
    occluded = attempt_foreground_deformation(
        first,
        second,
        mask,
        _track(),
        config=ForegroundDeformationExperimentConfig(enabled=True),
        source_foreground_mask=occluded_source_mask,
    )
    separate = attempt_foreground_deformation(
        first,
        second,
        non_crossing,
        _track(),
        config=ForegroundDeformationExperimentConfig(enabled=True),
        source_foreground_mask=second_mask,
    )

    assert transparent.reason == "foreground_instance_intersects_protected_domain"
    assert occluded.reason in {
            "insufficient_foreground_contour_or_centreline_correspondence",
            "insufficient_held_out_strong_foreground_edges",
            "insufficient_held_out_strong_edges_inside_foreground_mesh",
            "foreground_mesh_maps_outside_complete_source_coverage",
    }
    assert not occluded.accepted
    assert separate.reason == "foreground_instance_does_not_uniquely_cross_owner_boundary"
    assert not np.any(transparent.active_mask | occluded.active_mask | separate.active_mask)
