"""Role-safe video pipeline facade.

The historical renderer is deliberately kept behind this module while the
public command exposes only the immutable production lock.  This separation
lets a baseline remain byte-stable and prevents a mutable candidate YAML from
leaking into production.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import yaml

from .config import load_config
from .paths import PROJECT_ROOT
from .video_algorithm import VideoAlgorithmSpec, load_algorithm_config
from .video_algorithm_registry import resolve_video_algorithm
from .video_model_lock import verify_candidate_models
from .video_annotations import load_source_annotations
from .video_observability import (
    ObservabilitySpec,
    clear_observability_artifacts,
    write_audit_manifest,
    write_observability_artifacts,
)
from .video_v2_route import (
    is_cuda_c1_constrained_owner_implementation,
    is_cuda_c2_dis_residual_mesh_implementation,
    is_cuda_c3_raft_residual_mesh_implementation,
    is_cuda_c4_raft_rgbd_layered_mesh_implementation,
    is_cuda_c5_object_lock_implementation,
    is_cuda_c6_safe_multiband_implementation,
    is_cuda_c7_photometric_graph_implementation,
    is_cuda_c8_multilabel_window_implementation,
    is_strict_cuda_strip_owner_implementation,
)


def production_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the frozen production RGB-D video panorama")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/video_sequence"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--maximum-post-seconds", type=float)
    parser.add_argument("--reuse-online-trajectory", action="store_true")
    parser.add_argument("--trajectory-cache", type=Path)
    parser.add_argument("--online-state", type=Path)
    parser.add_argument("--defer-3d", action="store_true")
    return parser


def _lock_paths(config_path: Path | None) -> tuple[dict[str, Any], Path, Path]:
    config = load_config(config_path)
    settings = dict(dict(config.get("stitch", {})).get("video_panorama", {}))
    baseline_ref = settings.get("baseline_lock")
    production_ref = settings.get("production_lock")
    if not isinstance(baseline_ref, str) or not isinstance(production_ref, str):
        raise ValueError("video_panorama requires baseline_lock and production_lock")
    return config, (PROJECT_ROOT / baseline_ref).resolve(), (PROJECT_ROOT / production_ref).resolve()


def _baseline_legacy_settings() -> dict[str, Any]:
    document = load_algorithm_config(
        PROJECT_ROOT / "configs/video_algorithms/baseline_legacy_fast_b07b561.yaml"
    )
    settings = document.get("legacy_video_panorama")
    if not isinstance(settings, dict):
        raise ValueError("Frozen baseline config lacks legacy_video_panorama settings")
    return dict(settings)


def _legacy_settings_for(spec: VideoAlgorithmSpec) -> dict[str, Any]:
    """Translate the initial C0/C1 candidates into the frozen legacy engine.

    More advanced candidates are never silently approximated: their renderer
    implementation must be present and their model hashes locked before this
    bridge will run them.
    """

    settings = _baseline_legacy_settings()
    document = load_algorithm_config(spec.config_path)
    if spec.role == "baseline":
        return settings
    if spec.role == "production":
        legacy = document.get("legacy_video_panorama")
        if not isinstance(legacy, dict):
            raise ValueError("Frozen production config does not provide a runnable renderer contract")
        settings.update(legacy)
        return settings
    if spec.algorithm_id in {"C1_constrained_owner", "C2_dis_rgb_mesh", "C3_raft_rgb_mesh", "C4_raft_rgbd_layered_mesh", "C5_object_lock", "C6_multiband", "C7_photometric_graph", "C8_multilabel_window"}:
        # C1 has its own pair-local, second-order hard-owner compositor.  It
        # is an explicit experiment renderer, never an approximation of the
        # frozen curved visual seam.
        settings["fast_renderer"] = "hard_owner_diagnostic"
        settings["candidate_c1_constrained_owner"] = True
        source_document = (
            document
            if spec.algorithm_id == "C1_constrained_owner"
            else load_algorithm_config(PROJECT_ROOT / "configs/video_candidates/C1_constrained_owner.yaml")
        )
        components = source_document.get("components", {})
        step = components.get("scan_step") if isinstance(components, dict) else None
        if not isinstance(step, dict):
            raise ValueError("C1 candidate requires a scan_step component")
        normal, risk = step.get("normal_pixels"), step.get("risk_pixels")
        if not isinstance(normal, (int, float)) or not isinstance(risk, (int, float)):
            raise ValueError("C1 candidate scan_step requires numeric normal_pixels and risk_pixels")
        resampling = dict(settings.get("motion_resampling", {}))
        resampling["normal_target_step_pixels"] = float(normal)
        resampling["risk_target_step_pixels"] = float(risk)
        settings["motion_resampling"] = resampling
        # The CUDA v2 C1 data plane accepts only this closed, numeric seam
        # tuning mapping.  It is copied from the candidate declaration rather
        # than inferred from measurements, so fixed validation annotations
        # remain post-publication evidence only.
        # Keyframe density is intentionally inherited from the frozen C1
        # parent for C2--C8.  Seam controls, however, are a property of the
        # *current* candidate declaration: otherwise a C5--C8 validation
        # trial would silently execute C1's defaults even though its own
        # immutable config records different CUDA parameters.
        candidate_components = document.get("components", {})
        cuda_c1 = (
            candidate_components.get("cuda_c1")
            if isinstance(candidate_components, dict)
            else None
        )
        if cuda_c1 is not None:
            if not isinstance(cuda_c1, dict):
                raise ValueError("C1 candidate cuda_c1 must be a mapping")
            settings["candidate_c1_config"] = dict(cuda_c1)
        if spec.algorithm_id == "C2_dis_rgb_mesh":
            settings["candidate_mesh_evidence"] = {
                "enabled": True,
                "flow_backend": "dis",
                "require_depth_safety": False,
            }
        if spec.algorithm_id == "C3_raft_rgb_mesh":
            model_id = "torchvision_raft_small_C_T_V2"
            settings["candidate_mesh_evidence"] = {
                "enabled": True,
                "flow_backend": "raft",
                "require_depth_safety": False,
                "model_id": model_id,
                "model_sha256": spec.model_sha256[model_id],
            }
        if spec.algorithm_id == "C4_raft_rgbd_layered_mesh":
            model_id = "torchvision_raft_small_C_T_V2"
            settings["candidate_mesh_evidence"] = {
                "enabled": True,
                "flow_backend": "raft",
                "require_depth_safety": True,
                "model_id": model_id,
                "model_sha256": spec.model_sha256[model_id],
            }
        if spec.algorithm_id == "C5_object_lock":
            model_id = "torchvision_raft_small_C_T_V2"
            settings["candidate_mesh_evidence"] = {
                "enabled": True,
                "flow_backend": "raft",
                "require_depth_safety": True,
                "model_id": model_id,
                "model_sha256": spec.model_sha256[model_id],
            }
            settings["candidate_object_owner_lock"] = True
        if spec.algorithm_id == "C6_multiband":
            model_id = "torchvision_raft_small_C_T_V2"
            settings["candidate_mesh_evidence"] = {
                "enabled": True, "flow_backend": "raft", "require_depth_safety": True,
                "model_id": model_id, "model_sha256": spec.model_sha256[model_id],
            }
            settings["candidate_object_owner_lock"] = True
            settings["candidate_safe_multiband"] = True
        if spec.algorithm_id == "C7_photometric_graph":
            model_id = "torchvision_raft_small_C_T_V2"
            settings["candidate_mesh_evidence"] = {
                "enabled": True, "flow_backend": "raft", "require_depth_safety": True,
                "model_id": model_id, "model_sha256": spec.model_sha256[model_id],
            }
            settings["candidate_object_owner_lock"] = True
            settings["candidate_safe_multiband"] = True
            settings["candidate_global_photometric"] = True
        if spec.algorithm_id == "C8_multilabel_window":
            # C8 is the cumulative final ablation: it retains the audited
            # C4--C7 safeguards and adds only the local <=5-source owner
            # optimisation.  Running it as plain C1 would falsely represent
            # the YAML's declared parent lineage.
            model_id = "torchvision_raft_small_C_T_V2"
            settings["candidate_mesh_evidence"] = {
                "enabled": True, "flow_backend": "raft", "require_depth_safety": True,
                "model_id": model_id, "model_sha256": spec.model_sha256[model_id],
            }
            settings["candidate_object_owner_lock"] = True
            settings["candidate_safe_multiband"] = True
            settings["candidate_global_photometric"] = True
            settings["candidate_multilabel_owner"] = True
        return settings
    components = document.get("components", {})
    if not isinstance(components, dict):
        raise ValueError("Candidate components must be a mapping")
    step = components.get("scan_step")
    if isinstance(step, dict):
        normal, risk = step.get("normal_pixels"), step.get("risk_pixels")
        if isinstance(normal, (int, float)) and isinstance(risk, (int, float)):
            resampling = dict(settings.get("motion_resampling", {}))
            resampling["normal_target_step_pixels"] = float(normal)
            resampling["risk_target_step_pixels"] = float(risk)
            settings["motion_resampling"] = resampling
    advanced = {
        name
        for name in (
            "residual_mesh", "depth_confidence", "object_lock", "multiband",
            "photometric_graph", "multilabel",
        )
        if (value := components.get(name)) is not None
        and (not isinstance(value, dict) or bool(value.get("enabled", True)))
    }
    if advanced:
        raise RuntimeError(
            f"{spec.algorithm_id} requires video_visual_renderer_v2 with locked model assets; "
            "it cannot be approximated by the baseline renderer"
        )
    return settings


def _spec_report(
    spec: VideoAlgorithmSpec,
    *,
    fallback_used: bool = False,
    execution_backend: str = "legacy_video_renderer",
) -> dict[str, Any]:
    return {
        "role": spec.role,
        "algorithm_id": spec.algorithm_id,
        "implementation_id": spec.implementation_id,
        "config_sha256": spec.config_sha256,
        "source_commit": spec.source_commit,
        "model_sha256": dict(spec.model_sha256),
        "fallback_used": fallback_used,
        # The immutable algorithm declaration identifies intended component
        # lineage.  This separate field records the renderer that actually
        # produced pixels, preventing the legacy experiment bridge from being
        # mistaken for the CUDA v2 data plane during selection.
        "execution_backend": execution_backend,
    }


def _cuda_v2_route_mode(spec: VideoAlgorithmSpec) -> str | None:
    """Keep the v2 data plane explicit and immutable in production.

    C2--C8 use ``video_visual_renderer_v2`` as their intended implementation
    identity, but that is not evidence that every declared component is wired
    into the renderer.  C0 can use strict owner under a verified immutable
    identity; C1--C5 have bounded candidate-only CUDA integrations.  The
    C6 has a bounded C5-derived CUDA route and C7 extends that chain with its
    device-only photometric graph; C8 extends that exact candidate-only data
    plane with bounded CUDA multi-label owner recomposition.
    """

    if is_strict_cuda_strip_owner_implementation(
        role=spec.role, implementation_id=spec.implementation_id
    ):
        return "strict_owner"
    if is_cuda_c1_constrained_owner_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c1_constrained_owner"
    if is_cuda_c2_dis_residual_mesh_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c2_dis_residual_mesh"
    if is_cuda_c3_raft_residual_mesh_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c3_raft_residual_mesh"
    if is_cuda_c4_raft_rgbd_layered_mesh_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c4_raft_rgbd_layered_mesh"
    if is_cuda_c5_object_lock_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c5_object_lock"
    if is_cuda_c6_safe_multiband_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c6_safe_multiband"
    if is_cuda_c7_photometric_graph_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c7_photometric_graph"
    if is_cuda_c8_multilabel_window_implementation(
        role=spec.role,
        algorithm_id=spec.algorithm_id,
        implementation_id=spec.implementation_id,
    ):
        return "c8_multilabel_window"
    return None


def _uses_cuda_strict_owner_route(spec: VideoAlgorithmSpec) -> bool:
    """Backward-compatible predicate for C0-focused callers and tests."""

    return _cuda_v2_route_mode(spec) == "strict_owner"


def run_video_algorithm(
    *,
    input_path: Path,
    output: Path,
    role: str,
    candidate_config: Path | None = None,
    config_path: Path | None = None,
    observability: ObservabilitySpec | None = None,
    maximum_post_seconds: float | None = None,
    defer_3d: bool = False,
    reuse_online_trajectory: bool = False,
    trajectory_cache: Path | None = None,
    online_state: Path | None = None,
    scan_progress_interval: tuple[float, float] | None = None,
    evaluation_scope: str | None = None,
) -> dict[str, Any]:
    if role not in {"baseline", "candidate", "production"}:
        raise ValueError("algorithm role must be baseline, candidate, or production")
    _, baseline_lock, production_lock = _lock_paths(config_path)
    spec = resolve_video_algorithm(
        role, baseline_lock=baseline_lock, production_lock=production_lock,
        candidate_config=candidate_config,
    )
    # Candidate models are explicit local evidence.  This check is before any
    # session decoding or publishing, and the public production facade never
    # reaches a mutable candidate declaration.
    if spec.role == "candidate":
        verify_candidate_models(spec.model_sha256)
    observe = observability or ObservabilitySpec()
    output = output.expanduser().resolve()
    # The legacy publisher invalidates only its owned primary delivery and
    # central-strip archives.  Clear stale evidence here, before the primary
    # run, so a minimal/provenance rerun cannot inherit an old audit sidecar.
    clear_observability_artifacts(output)
    cuda_v2_route_mode = _cuda_v2_route_mode(spec)
    # The common orchestration settings are reused for real source selection,
    # ORB staging and Open3D audit.  The v2 route itself never receives the
    # historical CPU renderer's RGB output or settings toggles.
    legacy_settings = (
        _baseline_legacy_settings()
        if cuda_v2_route_mode == "strict_owner"
        else _legacy_settings_for(spec)
    )
    measurement_annotations = None
    if spec.role == "candidate":
        session_root = input_path.expanduser().resolve()
        session_root = session_root if session_root.is_dir() else session_root.parent
        annotation_path = PROJECT_ROOT / "benchmarks" / session_root.name / "annotations" / "objects.json"
        if annotation_path.is_file():
            # This feeds read-only post-publication measurement only.  It
            # cannot alter source selection, poses, RGB, owner, seam, or any
            # CUDA candidate renderer (including C5--C8 protection).
            measurement_annotations = load_source_annotations(annotation_path)
    # Artifact selection may request extra exported evidence but never changes
    # source selection, pose tracking, seam, or photometry.
    legacy_settings["fast_publish_auxiliary_exports"] = observe.artifact_level == "audit"
    custom = {"stitch": {"video_panorama": legacy_settings}}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", encoding="utf-8", delete=False) as handle:
        yaml.safe_dump(custom, handle, sort_keys=True)
        effective_config = Path(handle.name)
    try:
        from .video_panorama import run_legacy

        legacy_args = argparse.Namespace(
            input=input_path,
            output=output,
            config=effective_config,
            reuse_online_trajectory=reuse_online_trajectory,
            trajectory_cache=trajectory_cache,
            online_state=online_state,
            maximum_post_seconds=maximum_post_seconds,
            defer_3d=defer_3d,
            algorithm_spec=_spec_report(
                spec,
                execution_backend=(
                    "video_visual_renderer_v2_cuda"
                    if cuda_v2_route_mode is not None
                    else "legacy_candidate_experiment_bridge"
                    if spec.role == "candidate"
                    else "legacy_video_renderer"
                ),
            ),
            v2_cuda_strict_owner=cuda_v2_route_mode == "strict_owner",
            v2_cuda_renderer_mode=cuda_v2_route_mode,
            observability=observe.as_dict(),
            measurement_annotations=measurement_annotations,
            scan_progress_interval=scan_progress_interval,
            evaluation_scope=evaluation_scope,
        )
        published = run_legacy(legacy_args)
    finally:
        effective_config.unlink(missing_ok=True)

    # Evidence is post-publication, read-only work.  It receives only the
    # encoded primary artifacts, so it cannot influence renderer decisions.
    try:
        export = write_observability_artifacts(output, observe)
        if observe.artifact_level == "audit":
            audit_manifest = write_audit_manifest(output, observe, export)
            published["audit_manifest"] = str(output / "audit_manifest.json")
            published["audit_status"] = audit_manifest["status"]
    except Exception as exc:
        if observe.artifact_level != "audit":
            raise
        # A 2-D delivery has already been atomically published.  An audit
        # export failure is an evidence failure, never a reason to revoke it.
        audit_manifest = write_audit_manifest(output, observe, {}, error=exc)
        published["audit_manifest"] = str(output / "audit_manifest.json")
        published["audit_status"] = audit_manifest["status"]
        published["audit_error"] = str(exc)
    return published


def run_production(args: argparse.Namespace) -> dict[str, Any]:
    return run_video_algorithm(
        input_path=args.input,
        output=args.output,
        role="production",
        config_path=args.config,
        maximum_post_seconds=args.maximum_post_seconds,
        defer_3d=args.defer_3d,
        reuse_online_trajectory=args.reuse_online_trajectory,
        trajectory_cache=args.trajectory_cache,
        online_state=args.online_state,
    )
