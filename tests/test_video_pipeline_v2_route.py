from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from panorama_demo import video_pipeline
from panorama_demo import video_v2_route
from panorama_demo.video_algorithm import VideoAlgorithmSpec
from panorama_demo.video_algorithm import build_algorithm_spec
from panorama_demo.video_observability import ObservabilitySpec
from panorama_demo.video_v2_route import (
    VideoV2RouteError,
    is_cuda_c1_constrained_owner_implementation,
    is_cuda_c2_dis_residual_mesh_implementation,
    is_cuda_c3_raft_residual_mesh_implementation,
    is_cuda_c4_raft_rgbd_layered_mesh_implementation,
    is_cuda_c5_object_lock_implementation,
    is_cuda_c6_safe_multiband_implementation,
    is_cuda_c7_photometric_graph_implementation,
    is_cuda_c8_multilabel_window_implementation,
    is_cuda_c10_depth_conditioned_layout_implementation,
    is_cuda_c11_object_first_foreground_compositor_implementation,
    is_cuda_c12_joint_owner_final_grid_implementation,
    is_strict_cuda_strip_owner_implementation,
    render_cuda_c5_object_lock_v2,
    render_cuda_c6_safe_multiband_v2,
    render_cuda_c7_photometric_graph_v2,
    render_cuda_c8_multilabel_window_v2,
    render_cuda_c10_depth_conditioned_layout_v2,
    render_cuda_c11_object_first_foreground_compositor_v2,
    render_cuda_c12_joint_owner_final_grid_v2,
    _post_publication_measurement_context_if_c1_geometry_is_exact,
)


def _spec(tmp_path: Path, *, role: str, implementation_id: str) -> VideoAlgorithmSpec:
    return VideoAlgorithmSpec(
        role=role,  # type: ignore[arg-type]
        algorithm_id="C0_cuda_strict_strip_owner",
        implementation_id=implementation_id,
        config_path=tmp_path / "candidate.yaml",
        config_sha256="a" * 64,
        source_commit="test",
        model_sha256={},
        allow_baseline_fallback=False,
    )


def test_strict_cuda_route_accepts_only_the_dedicated_identity_for_candidate_or_locked_production(tmp_path):
    assert is_strict_cuda_strip_owner_implementation(
        role="candidate", implementation_id="torch_cuda_strip_owner_v2"
    )
    assert is_strict_cuda_strip_owner_implementation(
        role="production", implementation_id="torch_cuda_strip_owner_v2"
    )
    assert not video_pipeline._uses_cuda_strict_owner_route(  # noqa: SLF001
        _spec(tmp_path, role="candidate", implementation_id="video_visual_renderer_v2")
    )


def test_explicit_c0_cuda_candidate_config_resolves_to_the_narrow_route():
    spec = build_algorithm_spec("configs/video_candidates/C0_cuda_strict_strip_owner.yaml")

    assert spec.algorithm_id == "C0_cuda_strict_strip_owner"
    assert video_pipeline._uses_cuda_strict_owner_route(spec)  # noqa: SLF001


def test_c5_to_c8_routes_cannot_accept_manual_measurement_annotations():
    """Fixed labels are post-publication evaluator input, never render input."""

    for route in (
        render_cuda_c5_object_lock_v2,
        render_cuda_c6_safe_multiband_v2,
        render_cuda_c7_photometric_graph_v2,
        render_cuda_c8_multilabel_window_v2,
        render_cuda_c10_depth_conditioned_layout_v2,
        render_cuda_c11_object_first_foreground_compositor_v2,
        render_cuda_c12_joint_owner_final_grid_v2,
    ):
        assert "annotations" not in inspect.signature(route).parameters


def test_c4_mesh_output_fail_closes_c1_only_annotation_projection(monkeypatch):
    """A C1 inverse-grid projection cannot measure an accepted C4 mesh."""

    sentinel = object()
    monkeypatch.setattr(video_v2_route, "_post_publication_measurement_context", lambda **_: sentinel)
    common = {"sources": (), "strips": (), "calibration": None}
    mesh_result = SimpleNamespace(
        algorithm_audit={"c4_raft_rgbd_layered_mesh": {"actual_output_mesh_pixel_count": 1}}
    )
    c1_exact_result = SimpleNamespace(
        algorithm_audit={"c4_raft_rgbd_layered_mesh": {"actual_output_mesh_pixel_count": 0}}
    )

    assert _post_publication_measurement_context_if_c1_geometry_is_exact(
        **common, result=mesh_result
    ) is None
    assert _post_publication_measurement_context_if_c1_geometry_is_exact(
        **common, result=c1_exact_result
    ) is sentinel


def test_c1_to_c8_candidate_identities_enter_only_their_completed_cuda_routes(tmp_path):
    c1 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C1_constrained_owner",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c1.yaml",
        config_sha256="c" * 64,
        source_commit="test",
        model_sha256={},
        allow_baseline_fallback=False,
    )
    c2 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C2_dis_rgb_mesh",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c2.yaml",
        config_sha256="d" * 64,
        source_commit="test",
        model_sha256={},
        allow_baseline_fallback=False,
    )
    c3 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C3_raft_rgb_mesh",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c3.yaml",
        config_sha256="e" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )
    c4 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C4_raft_rgbd_layered_mesh",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c4.yaml",
        config_sha256="f" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )
    c5 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C5_object_lock",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c5.yaml",
        config_sha256="1" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )
    c6 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C6_multiband",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c6.yaml",
        config_sha256="2" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )
    c7 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C7_photometric_graph",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c7.yaml",
        config_sha256="3" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )
    c8 = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C8_multilabel_window",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c8.yaml",
        config_sha256="4" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )

    assert is_cuda_c1_constrained_owner_implementation(
        role="candidate", algorithm_id="C1_constrained_owner", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c1) == "c1_constrained_owner"  # noqa: SLF001
    assert is_cuda_c2_dis_residual_mesh_implementation(
        role="candidate", algorithm_id="C2_dis_rgb_mesh", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c2) == "c2_dis_residual_mesh"  # noqa: SLF001
    assert is_cuda_c3_raft_residual_mesh_implementation(
        role="candidate", algorithm_id="C3_raft_rgb_mesh", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c3) == "c3_raft_residual_mesh"  # noqa: SLF001
    assert is_cuda_c4_raft_rgbd_layered_mesh_implementation(
        role="candidate", algorithm_id="C4_raft_rgbd_layered_mesh", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c4) == "c4_raft_rgbd_layered_mesh"  # noqa: SLF001
    assert is_cuda_c5_object_lock_implementation(
        role="candidate", algorithm_id="C5_object_lock", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c5) == "c5_object_lock"  # noqa: SLF001
    assert is_cuda_c6_safe_multiband_implementation(
        role="candidate", algorithm_id="C6_multiband", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c6) == "c6_safe_multiband"  # noqa: SLF001
    assert is_cuda_c7_photometric_graph_implementation(
        role="candidate", algorithm_id="C7_photometric_graph", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c7) == "c7_photometric_graph"  # noqa: SLF001
    assert is_cuda_c8_multilabel_window_implementation(
        role="candidate", algorithm_id="C8_multilabel_window", implementation_id="video_visual_renderer_v2"
    )
    assert video_pipeline._cuda_v2_route_mode(c8) == "c8_multilabel_window"  # noqa: SLF001


def test_c10_config_is_immutable_candidate_only_and_enters_its_own_cuda_route():
    spec = build_algorithm_spec("configs/video_candidates/C10_depth_conditioned_multi_perspective_layout.yaml")

    assert spec.required_components == (
        "c1_constrained_owner", "c3_raft_mesh", "c4_depth_layered_mesh", "c10_depth_conditioned_layout",
    )
    assert is_cuda_c10_depth_conditioned_layout_implementation(
        role="candidate", algorithm_id=spec.algorithm_id, implementation_id=spec.implementation_id,
    )
    assert not is_cuda_c10_depth_conditioned_layout_implementation(
        role="production", algorithm_id=spec.algorithm_id, implementation_id=spec.implementation_id,
    )
    assert video_pipeline._cuda_v2_route_mode(spec) == "c10_depth_conditioned_layout"  # noqa: SLF001


def test_c11_config_is_candidate_only_and_requires_a_real_object_compositor_route():
    spec = build_algorithm_spec("configs/video_candidates/C11_object_first_single_view_foreground_compositor.yaml")

    assert spec.required_components[-2:] == (
        "c10_depth_conditioned_layout", "c11_object_first_foreground_compositor",
    )
    assert is_cuda_c11_object_first_foreground_compositor_implementation(
        role="candidate", algorithm_id=spec.algorithm_id, implementation_id=spec.implementation_id,
    )
    assert not is_cuda_c11_object_first_foreground_compositor_implementation(
        role="production", algorithm_id=spec.algorithm_id, implementation_id=spec.implementation_id,
    )
    assert video_pipeline._cuda_v2_route_mode(spec) == "c11_object_first_foreground_compositor"  # noqa: SLF001


def test_c12_config_is_candidate_only_and_requires_the_real_five_to_seven_source_route():
    spec = build_algorithm_spec("configs/video_candidates/C12_joint_owner_mesh_window.yaml")

    assert spec.required_components == ("c1_constrained_owner", "c12_joint_owner_final_grid")
    assert is_cuda_c12_joint_owner_final_grid_implementation(
        role="candidate", algorithm_id=spec.algorithm_id, implementation_id=spec.implementation_id,
    )
    assert not is_cuda_c12_joint_owner_final_grid_implementation(
        role="production", algorithm_id=spec.algorithm_id, implementation_id=spec.implementation_id,
    )
    assert video_pipeline._cuda_v2_route_mode(spec) == "c12_joint_owner_final_grid"  # noqa: SLF001
    settings = video_pipeline._legacy_settings_for(spec)  # noqa: SLF001
    assert settings["candidate_joint_owner_final_grid"] is True
    assert settings.get("candidate_depth_conditioned_layout", False) is False


def test_c12_route_rejects_a_non_window_source_count_before_any_renderer_fallback():
    with pytest.raises(VideoV2RouteError, match="5--7 source window"):
        render_cuda_c12_joint_owner_final_grid_v2(
            sources=(), camera_to_world=(), calibration=None, pushbroom_config={},
            selected_motions=(), motion_pixels_to_full_resolution=1.0,
        )


def test_pipeline_passes_dedicated_cuda_route_and_reports_actual_backend(monkeypatch, tmp_path):
    spec = _spec(tmp_path, role="candidate", implementation_id="torch_cuda_strip_owner_v2")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(video_pipeline, "_baseline_legacy_settings", lambda: {})

    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["route"] = args.v2_cuda_strict_owner
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})

    result = video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="candidate",
        candidate_config=tmp_path / "candidate.yaml",
        observability=ObservabilitySpec(),
    )

    assert result == {"panorama": "fake"}
    assert captured == {"route": True, "backend": "video_visual_renderer_v2_cuda"}


def test_pipeline_allows_verified_production_identity_to_reuse_v2_cuda_route(monkeypatch, tmp_path):
    spec = _spec(tmp_path, role="production", implementation_id="torch_cuda_strip_owner_v2")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "_baseline_legacy_settings", lambda: {})

    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["route"] = args.v2_cuda_strict_owner
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})

    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="production",
        observability=ObservabilitySpec(),
    )

    assert captured == {"route": True, "backend": "video_visual_renderer_v2_cuda"}


def test_pipeline_keeps_c1_identity_on_explicit_legacy_experiment_bridge(monkeypatch, tmp_path):
    spec = _spec(tmp_path, role="candidate", implementation_id="video_visual_renderer_v2")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(video_pipeline, "_legacy_settings_for", lambda _spec: {})

    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["route"] = args.v2_cuda_strict_owner
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})

    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="candidate",
        candidate_config=tmp_path / "candidate.yaml",
    )

    assert captured == {"route": False, "backend": "legacy_candidate_experiment_bridge"}


def test_pipeline_passes_c1_only_v2_mode_and_preserves_candidate_settings(monkeypatch, tmp_path):
    spec = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C1_constrained_owner",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c1.yaml",
        config_sha256="e" * 64,
        source_commit="test",
        model_sha256={},
        allow_baseline_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(
        video_pipeline,
        "_legacy_settings_for",
        lambda _spec: {"candidate_c1_constrained_owner": True},
    )

    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["strict"] = args.v2_cuda_strict_owner
        captured["mode"] = args.v2_cuda_renderer_mode
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})

    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="candidate",
        candidate_config=tmp_path / "c1.yaml",
    )

    assert captured == {
        "strict": False,
        "mode": "c1_constrained_owner",
        "backend": "video_visual_renderer_v2_cuda",
    }


def test_pipeline_passes_c2_v2_mode_with_its_dis_component_settings(monkeypatch, tmp_path):
    spec = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C2_dis_rgb_mesh",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c2.yaml",
        config_sha256="f" * 64,
        source_commit="test",
        model_sha256={},
        allow_baseline_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(
        video_pipeline,
        "_legacy_settings_for",
        lambda _spec: {
            "candidate_c1_constrained_owner": True,
            "candidate_mesh_evidence": {"flow_backend": "dis", "require_depth_safety": False},
        },
    )

    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["strict"] = args.v2_cuda_strict_owner
        captured["mode"] = args.v2_cuda_renderer_mode
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})

    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="candidate",
        candidate_config=tmp_path / "c2.yaml",
    )

    assert captured == {
        "strict": False,
        "mode": "c2_dis_residual_mesh",
        "backend": "video_visual_renderer_v2_cuda",
    }


def test_pipeline_passes_c3_v2_mode_with_locked_raft_component_settings(monkeypatch, tmp_path):
    spec = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C3_raft_rgb_mesh",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c3.yaml",
        config_sha256="b" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(
        video_pipeline,
        "_legacy_settings_for",
        lambda _spec: {
            "candidate_c1_constrained_owner": True,
            "candidate_mesh_evidence": {"flow_backend": "raft", "require_depth_safety": False},
        },
    )

    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["strict"] = args.v2_cuda_strict_owner
        captured["mode"] = args.v2_cuda_renderer_mode
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})

    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="candidate",
        candidate_config=tmp_path / "c3.yaml",
    )

    assert captured == {
        "strict": False,
        "mode": "c3_raft_residual_mesh",
        "backend": "video_visual_renderer_v2_cuda",
    }


def test_pipeline_passes_c4_v2_mode_only_with_raft_depth_layer_settings(monkeypatch, tmp_path):
    spec = VideoAlgorithmSpec(
        role="candidate",
        algorithm_id="C4_raft_rgbd_layered_mesh",
        implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c4.yaml",
        config_sha256="c" * 64,
        source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64},
        allow_baseline_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        video_pipeline,
        "_lock_paths",
        lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"),
    )
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(
        video_pipeline,
        "_legacy_settings_for",
        lambda _spec: {
            "candidate_c1_constrained_owner": True,
            "candidate_mesh_evidence": {"flow_backend": "raft", "require_depth_safety": True},
        },
    )
    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["strict"] = args.v2_cuda_strict_owner
        captured["mode"] = args.v2_cuda_renderer_mode
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})

    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session",
        output=tmp_path / "output",
        role="candidate",
        candidate_config=tmp_path / "c4.yaml",
    )

    assert captured == {
        "strict": False,
        "mode": "c4_raft_rgbd_layered_mesh",
        "backend": "video_visual_renderer_v2_cuda",
    }


def test_pipeline_passes_c5_v2_mode_only_with_its_c4_and_object_lock_settings(monkeypatch, tmp_path):
    spec = VideoAlgorithmSpec(
        role="candidate", algorithm_id="C5_object_lock", implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c5.yaml", config_sha256="5" * 64, source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64}, allow_baseline_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(video_pipeline, "_lock_paths", lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"))
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(video_pipeline, "_legacy_settings_for", lambda _spec: {
        "candidate_c1_constrained_owner": True,
        "candidate_mesh_evidence": {"flow_backend": "raft", "require_depth_safety": True},
        "candidate_object_owner_lock": True,
    })
    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["mode"] = args.v2_cuda_renderer_mode
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})
    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session", output=tmp_path / "output", role="candidate",
        candidate_config=tmp_path / "c5.yaml",
    )

    assert captured == {"mode": "c5_object_lock", "backend": "video_visual_renderer_v2_cuda"}


def test_pipeline_passes_c6_v2_mode_only_with_its_complete_c5_and_multiband_settings(monkeypatch, tmp_path):
    spec = VideoAlgorithmSpec(
        role="candidate", algorithm_id="C6_multiband", implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c6.yaml", config_sha256="6" * 64, source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64}, allow_baseline_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(video_pipeline, "_lock_paths", lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"))
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(video_pipeline, "_legacy_settings_for", lambda _spec: {
        "candidate_c1_constrained_owner": True,
        "candidate_mesh_evidence": {"flow_backend": "raft", "require_depth_safety": True},
        "candidate_object_owner_lock": True,
        "candidate_safe_multiband": True,
    })
    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["mode"] = args.v2_cuda_renderer_mode
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})
    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session", output=tmp_path / "output", role="candidate",
        candidate_config=tmp_path / "c6.yaml",
    )

    assert captured == {"mode": "c6_safe_multiband", "backend": "video_visual_renderer_v2_cuda"}


def test_pipeline_passes_c8_v2_mode_only_with_the_complete_c7_parent_chain(monkeypatch, tmp_path):
    spec = VideoAlgorithmSpec(
        role="candidate", algorithm_id="C8_multilabel_window", implementation_id="video_visual_renderer_v2",
        config_path=tmp_path / "c8.yaml", config_sha256="8" * 64, source_commit="test",
        model_sha256={"torchvision_raft_small_C_T_V2": "a" * 64}, allow_baseline_fallback=False,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(video_pipeline, "_lock_paths", lambda _config: ({}, tmp_path / "baseline.lock", tmp_path / "production.lock"))
    monkeypatch.setattr(video_pipeline, "resolve_video_algorithm", lambda *_a, **_k: spec)
    monkeypatch.setattr(video_pipeline, "verify_candidate_models", lambda _models: None)
    monkeypatch.setattr(video_pipeline, "_legacy_settings_for", lambda _spec: {
        "candidate_c1_constrained_owner": True,
        "candidate_mesh_evidence": {"flow_backend": "raft", "require_depth_safety": True},
        "candidate_object_owner_lock": True,
        "candidate_safe_multiband": True,
        "candidate_global_photometric": True,
        "candidate_multilabel_owner": True,
    })
    from panorama_demo import video_panorama

    def fake_legacy(args):
        captured["mode"] = args.v2_cuda_renderer_mode
        captured["backend"] = args.algorithm_spec["execution_backend"]
        return {"panorama": "fake"}

    monkeypatch.setattr(video_panorama, "run_legacy", fake_legacy)
    monkeypatch.setattr(video_pipeline, "write_observability_artifacts", lambda *_a, **_k: {})
    video_pipeline.run_video_algorithm(
        input_path=tmp_path / "session", output=tmp_path / "output", role="candidate",
        candidate_config=tmp_path / "c8.yaml",
    )

    assert captured == {"mode": "c8_multilabel_window", "backend": "video_visual_renderer_v2_cuda"}
