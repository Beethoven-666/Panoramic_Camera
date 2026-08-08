from __future__ import annotations

from pathlib import Path

from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_cuda_photometric import CudaPhotometricConfig
from panorama_demo.video_cuda_photometric_bundle import CudaIlluminationFieldConfig, CudaPhotometricBundleError
from panorama_demo.video_pipeline import _cuda_v2_route_mode
from panorama_demo.video_visual_renderer_v2 import _finalize_component_execution


ROOT = Path(__file__).resolve().parents[1]


def test_c13_immutable_candidate_routes_to_its_own_cuda_data_plane() -> None:
    spec = build_algorithm_spec(
        ROOT / "configs/video_candidates/C13_robust_photometric_bundle.yaml",
        expected_role="candidate",
    )
    assert spec.algorithm_id == "C13_robust_photometric_bundle"
    assert _cuda_v2_route_mode(spec) == "c13_robust_photometric_bundle"
    assert spec.required_components[-1] == "c13_robust_photometric_bundle"


def test_c13_bounds_and_only_64x96_field_cells_are_closed() -> None:
    strict = CudaPhotometricConfig(
        gain_minimum=0.75, gain_maximum=1.35, bias_absolute_maximum=0.08,
        temporal_first_order_regularization=5e-4,
        temporal_second_order_regularization=2.5e-4,
        robust_huber_delta=0.02, robust_irls_iterations=3,
    ).validated()
    assert strict.gain_minimum == 0.75
    assert strict.gain_maximum == 1.35
    assert CudaIlluminationFieldConfig().validated().cell_width_pixels == 64
    try:
        CudaIlluminationFieldConfig(cell_width_pixels=48, cell_height_pixels=48).validated()
    except CudaPhotometricBundleError:
        pass
    else:  # pragma: no cover - protects the closed immutable C13 contract
        raise AssertionError("C13 accepted a non-64/96 illumination field")


def test_c13_identity_or_rejection_is_not_selection_eligible() -> None:
    audit = {
        "c1_constrained_owner": {"actual_output_mesh_pixel_count": 1, "pair_audits": [{}]},
        "c13_robust_photometric_bundle": {
            "accepted": False,
            "fail_closed_identity": True,
            "actual_safe_output_affected_pixel_count": 0,
            "per_source_fields": [],
        },
    }
    _finalize_component_execution(
        audit,
        required_components=("c1_constrained_owner", "c13_robust_photometric_bundle"),
    )
    assert audit["component_execution"]["c13_robust_photometric_bundle"]["applied_to_output"] is False
    assert audit["candidate_run_state"] == "invalid_component_execution"
