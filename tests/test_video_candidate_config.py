from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from panorama_demo.video_algorithm import (
    build_algorithm_spec,
    canonical_config_sha256,
    load_algorithm_config,
)
from panorama_demo.video_pipeline import _legacy_settings_for


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CHAIN = (
    ("C0_baseline_reference", None),
    ("C1_constrained_owner", "C0_baseline_reference"),
    ("C2_dis_rgb_mesh", "C1_constrained_owner"),
    ("C3_raft_rgb_mesh", "C1_constrained_owner"),
    ("C4_raft_rgbd_layered_mesh", "C3_raft_rgb_mesh"),
    ("C5_object_lock", "C4_raft_rgbd_layered_mesh"),
    ("C6_multiband", "C5_object_lock"),
    ("C7_photometric_graph", "C6_multiband"),
    ("C8_multilabel_window", "C7_photometric_graph"),
)


@pytest.mark.parametrize(("candidate_id", "parent_id"), EXPECTED_CHAIN)
def test_candidate_templates_are_self_contained_and_hash_valid(candidate_id, parent_id):
    path = ROOT / "configs" / "video_candidates" / f"{candidate_id}.yaml"
    config = load_algorithm_config(path)
    spec = build_algorithm_spec(path, expected_role="candidate")

    assert config["candidate_id"] == candidate_id
    assert config["parent_candidate_id"] == parent_id
    assert config["changed_components"] == list(config["changed_components"])
    assert spec.algorithm_id == candidate_id
    assert spec.config_sha256 == config["config_sha256"]


def test_c2_is_dis_only_and_object_lock_starts_at_c5():
    c2 = load_algorithm_config(ROOT / "configs" / "video_candidates" / "C2_dis_rgb_mesh.yaml")
    c4 = load_algorithm_config(ROOT / "configs" / "video_candidates" / "C4_raft_rgbd_layered_mesh.yaml")
    c5 = load_algorithm_config(ROOT / "configs" / "video_candidates" / "C5_object_lock.yaml")

    assert c2["components"]["residual_mesh"]["flow_backend"] == "dis"
    assert c2["components"]["residual_mesh"]["depth_layers"] is False
    assert c2["components"]["object_lock"]["enabled"] is False
    assert c4["components"]["object_lock"]["enabled"] is False
    assert c5["components"]["object_lock"]["enabled"] is True


def test_c0_reference_remains_runnable_with_disabled_optional_modules():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C0_baseline_reference.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["motion_resampling"]["normal_target_step_pixels"] == 20.0


def test_c1_selects_its_real_source_constrained_owner_renderer():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C1_constrained_owner.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["fast_renderer"] == "hard_owner_diagnostic"
    assert settings["candidate_c1_constrained_owner"] is True
    assert settings["motion_resampling"]["normal_target_step_pixels"] == 12.0
    assert settings["motion_resampling"]["risk_target_step_pixels"] == 5.0


def test_c2_runs_its_dis_mesh_evidence_without_approximating_it_away():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C2_dis_rgb_mesh.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["candidate_mesh_evidence"] == {
        "enabled": True,
        "flow_backend": "dis",
        "require_depth_safety": False,
    }


def test_c3_uses_the_locked_local_raft_model_for_mesh_evidence():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C3_raft_rgb_mesh.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["candidate_mesh_evidence"]["flow_backend"] == "raft"
    assert settings["candidate_mesh_evidence"]["model_sha256"] == spec.model_sha256[
        "torchvision_raft_small_C_T_V2"
    ]


def test_c4_requires_depth_safe_raft_mesh_evidence():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C4_raft_rgbd_layered_mesh.yaml",
        expected_role="candidate",
    )
    assert _legacy_settings_for(spec)["candidate_mesh_evidence"]["require_depth_safety"] is True


def test_c5_enables_real_source_depth_object_owner_locks():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C5_object_lock.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["candidate_object_owner_lock"] is True
    assert settings["candidate_mesh_evidence"]["require_depth_safety"] is True


def test_later_candidate_uses_its_own_cuda_c1_controls_but_inherits_c1_scan_step(tmp_path):
    """A C5 tuning declaration must not be silently replaced by C1 defaults."""

    document = load_algorithm_config(ROOT / "configs" / "video_candidates" / "C5_object_lock.yaml")
    document["components"]["cuda_c1"] = {
        "corridor_width_pixels": 160,
        "maximum_row_step_pixels": 1,
        "first_order_penalty": 20.0,
        "second_order_penalty": 10.0,
    }
    document["config_sha256"] = canonical_config_sha256(document)
    path = tmp_path / "C5_validation.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    settings = _legacy_settings_for(build_algorithm_spec(path, expected_role="candidate"))

    assert settings["motion_resampling"]["normal_target_step_pixels"] == 12.0
    assert settings["motion_resampling"]["risk_target_step_pixels"] == 5.0
    assert settings["candidate_c1_config"] == document["components"]["cuda_c1"]


def test_c6_enables_narrow_safe_multiband_only_after_object_lock():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C6_multiband.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["candidate_object_owner_lock"] is True
    assert settings["candidate_safe_multiband"] is True


def test_c7_enables_global_photometric_graph_after_safe_multiband():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C7_photometric_graph.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["candidate_safe_multiband"] is True
    assert settings["candidate_global_photometric"] is True


def test_c8_composes_c4_to_c7_safeguards_with_local_multilabel_optimisation():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "C8_multilabel_window.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["candidate_mesh_evidence"]["require_depth_safety"] is True
    assert settings["candidate_object_owner_lock"] is True
    assert settings["candidate_safe_multiband"] is True
    assert settings["candidate_global_photometric"] is True
    assert settings["candidate_multilabel_owner"] is True
