from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from panorama_demo import video_pipeline
from panorama_demo.video_algorithm import (
    build_algorithm_spec,
    canonical_config_sha256,
    load_algorithm_config,
)
from panorama_demo.video_pipeline import _legacy_settings_for, _spec_report
from panorama_demo.video_observability import ObservabilitySpec
from panorama_demo.video_candidate_manifest import canonical_candidate_manifest_sha256


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


def test_d2_is_a_candidate_only_d1_successor_not_a_c3_c4_mesh_alias():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "D2_monotonic_depth_layer_warp.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    assert settings["fast_renderer"] == "hard_owner_diagnostic"
    assert settings["candidate_dense_real_frame_layout"]["real_source_fps"] == 24
    d2 = settings["candidate_d2_monotonic_depth_layer_warp"]
    assert d2["layers"] == ["far", "mid", "near"]
    assert d2["multiband"] is False
    assert spec.replaces_output_components == (
        "c3_raft_mesh", "c4_depth_layered_mesh", "d1_dense_real_frame_hard_owner",
    )


def test_d3_is_a_d2_successor_with_a_real_source_owner_only_contract():
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / "D3_object_first_dense_source_compositor.yaml",
        expected_role="candidate",
    )
    settings = _legacy_settings_for(spec)
    d3 = settings["candidate_d3_object_first_dense_source_compositor"]
    assert settings["candidate_d2_monotonic_depth_layer_warp"]["multiband"] is False
    assert d3["object_flow_or_warp"] is False
    assert d3["object_multiband"] is False
    assert d3["source_support_gate"] == 0.98


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
    import json
    manifest = json.loads((ROOT / "configs" / "video_candidates" / "candidate_manifest.json").read_text(encoding="utf-8"))
    manifest["candidates"]["C5_object_lock"]["config_sha256"] = document["config_sha256"]
    manifest["manifest_sha256"] = canonical_candidate_manifest_sha256(manifest)
    (tmp_path / "candidate_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

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


@pytest.mark.parametrize(
    ("candidate_id", "tracking_fps"),
    (("V6_rgb_only_graphcut", 12.0), ("V6_rgb_only_graphcut_t2", 16.0)),
)
def test_v6_routes_keep_their_frozen_tracking_and_resampling_contracts(
    candidate_id: str,
    tracking_fps: float,
) -> None:
    spec = build_algorithm_spec(
        ROOT / "configs" / "video_candidates" / f"{candidate_id}.yaml",
        expected_role="candidate",
    )

    settings = _legacy_settings_for(spec)

    assert settings["fast_renderer"] == "v6_graphcut_candidate"
    assert settings["fast_orb_target_fps"] == tracking_fps
    assert settings["motion_resampling"]["normal_target_step_pixels"] == 8.0
    assert settings["motion_resampling"]["risk_target_step_pixels"] == 5.0
    assert settings["motion_resampling"]["maximum_step_pixels"] == 12.0


def test_v61_config_routes_immutable_tracking_geometry_and_report_contract() -> None:
    spec = build_algorithm_spec(
        ROOT
        / "configs"
        / "video_candidates"
        / "V61_tail_guarded_full_panorama.yaml",
        expected_role="candidate",
    )

    settings = _legacy_settings_for(spec)
    report = _spec_report(spec)

    assert settings["fast_renderer"] == "v61_tail_guarded_candidate"
    assert settings["fast_orb_target_fps"] == 12.0
    assert settings["motion_resampling"]["normal_target_step_pixels"] == 8.0
    assert settings["motion_resampling"]["risk_target_step_pixels"] == 5.0
    assert settings["motion_resampling"]["maximum_step_pixels"] == 12.0
    assert settings["candidate_v61_geometry_gate"] == {
        "minimum_reliable_pixels": 128,
        "fb_p95_max_px": 1.25,
        "edge_p95_max_px": 0.75,
        "minimum_matched_edge_fraction": 0.85,
        "tail_threshold_px": 1.25,
        "tail_dilation_px": 3,
    }
    assert report["required_evidence_components"] == [
        "orb_anchor_trajectory",
        "open3d_rgbd_edges",
        "dis_forward_backward",
    ]
    assert report["required_output_components"] == [
        "v61_tail_guarded_full_panorama"
    ]
    assert report["replaces_output_components"] == ["v6_rgb_only_graphcut"]
    assert report["working_tree_dirty"] is spec.working_tree_dirty
    assert report["candidate_manifest_path"] == str(spec.candidate_manifest_path)
    assert report["candidate_manifest_sha256"] == spec.candidate_manifest_sha256


@pytest.mark.parametrize(
    ("component", "invalid_value", "message"),
    (
        ("tracking_fps", 8, "tracking_fps=12"),
        (
            "geometry_gate",
            {"minimum_reliable_pixels": 128},
            "immutable geometry_gate",
        ),
    ),
)
def test_v61_route_rejects_mutated_runtime_contract(
    tmp_path: Path,
    component: str,
    invalid_value: object,
    message: str,
) -> None:
    original = ROOT / "configs" / "video_candidates" / "V61_tail_guarded_full_panorama.yaml"
    document = load_algorithm_config(original)
    document["components"][component] = invalid_value
    mutated = tmp_path / original.name
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    spec = build_algorithm_spec(original, expected_role="candidate")

    with pytest.raises(ValueError, match=message):
        _legacy_settings_for(replace(spec, config_path=mutated))


def test_v61_renderer_route_rejects_non_candidate_identity() -> None:
    spec = build_algorithm_spec(
        ROOT
        / "configs"
        / "video_candidates"
        / "V61_tail_guarded_full_panorama.yaml",
        expected_role="candidate",
    )

    with pytest.raises(ValueError, match="candidate-only"):
        _legacy_settings_for(replace(spec, role="production"))


def test_v61_config_reaches_only_its_candidate_runtime_route(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    candidate = (
        ROOT
        / "configs"
        / "video_candidates"
        / "V61_tail_guarded_full_panorama.yaml"
    )
    spec = build_algorithm_spec(candidate, expected_role="candidate")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(
        video_pipeline, "resolve_video_algorithm", lambda *_args, **_kwargs: spec
    )
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(
        video_pipeline, "write_observability_artifacts", lambda *_args, **_kwargs: {}
    )
    from panorama_demo import video_panorama

    def fake_legacy(args: object) -> dict[str, str]:
        runtime = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        captured["settings"] = runtime["stitch"]["video_panorama"]
        captured["identity"] = dict(args.algorithm_spec)
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)

    result = video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="candidate",
        candidate_config=candidate,
        observability=ObservabilitySpec(),
    )

    assert result == {"panorama": "fake"}
    settings = captured["settings"]
    assert settings["fast_renderer"] == "v61_tail_guarded_candidate"
    assert settings["fast_orb_target_fps"] == 12.0
    assert settings["candidate_v61_geometry_gate"]["tail_threshold_px"] == 1.25
    identity = captured["identity"]
    assert identity["role"] == "candidate"
    assert identity["required_output_components"] == [
        "v61_tail_guarded_full_panorama"
    ]
