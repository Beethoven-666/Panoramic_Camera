"""Narrow, fail-closed lifecycle adapter for the first v2 CUDA data plane.

The legacy video orchestrator remains responsible for strict session loading,
real-source selection, the complete ORB-SLAM3 chain, and adjacent Open3D
audits.  This module receives only those already-audited real sources and
replaces the final CPU pushbroom sampling with the CUDA strict-owner renderer.
It supports the narrow C0 strict-owner route, the separately bounded C1
curved hard-owner route, C2's local CUDA DIS residual mesh, C3's locked
bidirectional RAFT-small RGB residual mesh, and C4's C3 chain with resident
aligned-depth layer protection.  C5 adds a bounded resident object/depth
single-real-owner lock, and C6 extends that exact chain with safe-background
device-resident MultiBand.  C7 extends that exact chain with an anchored,
device-resident linear-light photometric graph.  C8 then adds only a bounded
2--5-source CUDA multi-label hard-owner recomposition over that exact chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from .calibrated_rgb_pushbroom import (
    CalibratedRGBPushbroomConfig,
    build_calibrated_rgb_pushbroom_layout,
    estimate_rgb_motion_pixels_per_mm,
)
from .session import CameraIntrinsics, RGBDFrame
from .video_algorithm_contract import PairPlan, VideoAlgorithmContractError
from .video_gpu_runtime import VideoGpuRuntimeConfig
from .video_visual_metrics import evaluate_visual_metrics
from .video_model_lock import verify_candidate_models
from .video_raft_runtime import RAFTSmallRuntimeConfig, TorchvisionRAFTSmallRuntime
from .video_visual_renderer_v2 import (
    CudaC1ConstrainedOwnerConfig,
    CudaRealSource,
    TorchCudaC4RAFTDepthLayeredMeshAlgorithm,
    TorchCudaC9PositiveJacobianLinePreservingLayeredMeshAlgorithm,
    TorchCudaC5ObjectLockAlgorithm,
    TorchCudaC6SafeMultiBandAlgorithm,
    TorchCudaC7PhotometricGraphAlgorithm,
    TorchCudaC8MultilabelWindowAlgorithm,
    TorchCudaC13RobustPhotometricBundleAlgorithm,
    TorchCudaC10DepthConditionedLayoutAlgorithm,
    TorchCudaC12JointOwnerFinalGridAlgorithm,
    TorchCudaC11ObjectFirstForegroundCompositorAlgorithm,
    TorchCudaC3RAFTResidualMeshAlgorithm,
    TorchCudaC2DisResidualMeshAlgorithm,
    TorchCudaC1ConstrainedOwnerAlgorithm,
    TorchCudaStripOwnerAlgorithm,
    build_cuda_strips_from_pushbroom_layout,
)
from .video_candidate_annotation_projection import (
    apply_final_grid_updates,
    build_candidate_annotation_projection,
    build_v2_c1_calibrated_inverse_sources,
)
from .video_v2_audit_export import V2CudaAuditExportContext


class VideoV2RouteError(RuntimeError):
    """A candidate asked the strict CUDA route to do unimplemented work."""


def _c1_config_from_mapping(config: Mapping[str, object] | None) -> CudaC1ConstrainedOwnerConfig:
    """Build the closed C1 CUDA controls from a candidate-only mapping.

    Candidate YAMLs are allowed to tune only the bounded C1 corridor and its
    curvature regularisation.  The function is deliberately strict: unknown
    keys cannot silently appear to influence a CUDA experiment, and no
    annotation-derived value can enter the renderer through this boundary.
    """

    if config is None:
        return CudaC1ConstrainedOwnerConfig()
    if not isinstance(config, Mapping):
        raise VideoV2RouteError("candidate_c1_config must be a mapping")
    allowed = {
        "corridor_width_pixels",
        "maximum_row_step_pixels",
        "first_order_penalty",
        "second_order_penalty",
    }
    unknown = set(config).difference(allowed)
    if unknown:
        raise VideoV2RouteError(f"candidate_c1_config has unsupported keys: {sorted(unknown)}")
    defaults = CudaC1ConstrainedOwnerConfig()
    try:
        return CudaC1ConstrainedOwnerConfig(
            corridor_width_pixels=int(config.get("corridor_width_pixels", defaults.corridor_width_pixels)),
            maximum_row_step_pixels=int(config.get("maximum_row_step_pixels", defaults.maximum_row_step_pixels)),
            first_order_penalty=float(config.get("first_order_penalty", defaults.first_order_penalty)),
            second_order_penalty=float(config.get("second_order_penalty", defaults.second_order_penalty)),
        )
    except (TypeError, ValueError, VideoAlgorithmContractError) as exc:
        raise VideoV2RouteError("candidate_c1_config is invalid") from exc


@dataclass(frozen=True)
class V2PostPublicationMeasurementContext:
    """Read-only geometry needed to project labels after primary publication.

    This deliberately carries no RGB, owner authority, seam controls, or
    mutable annotations.  ``video_panorama`` may consume it only after the
    atomic 2-D delivery exists.
    """

    sources: tuple[RGBDFrame, ...]
    strips: tuple[object, ...]
    calibration: CameraIntrinsics
    canvas_shape: tuple[int, int]
    corridor_width_pixels: int
    final_grid_updates: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class V2CudaStripRender:
    """Compatibility shape consumed by the common publication/report path."""

    panorama: np.ndarray
    owner_frame_id: np.ndarray
    metadata: dict[str, object]
    measurement_projection_payload: dict[str, object] | None = None
    measurement_projection_masks: dict[str, np.ndarray] | None = None
    post_publication_measurement_context: V2PostPublicationMeasurementContext | None = None
    audit_export_context: V2CudaAuditExportContext | None = None


def _v2_audit_export_context(
    *,
    sources: Sequence[RGBDFrame],
    strips: Sequence[object],
    calibration: CameraIntrinsics,
    result: object,
    include_adjacent_corridors: bool,
) -> V2CudaAuditExportContext:
    """Return serialisation-free source/grid evidence for audit-only staging."""

    corrections: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    audit = getattr(result, "algorithm_audit", {})
    c7 = audit.get("c7_global_photometric") if isinstance(audit, dict) else None
    raw_corrections = c7.get("export_corrections_bgr", []) if isinstance(c7, dict) else []
    if not isinstance(raw_corrections, list):
        raise VideoV2RouteError("C7 CUDA audit correction export is malformed")
    for item in raw_corrections:
        if not isinstance(item, Mapping):
            raise VideoV2RouteError("C7 CUDA audit correction record is malformed")
        try:
            frame_id = int(item["frame_id"])
            gain = tuple(float(value) for value in item["gain_bgr"])
            bias = tuple(float(value) for value in item["bias_bgr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise VideoV2RouteError("C7 CUDA audit correction record is invalid") from exc
        if len(gain) != 3 or len(bias) != 3 or not np.isfinite(gain).all() or not np.isfinite(bias).all():
            raise VideoV2RouteError("C7 CUDA audit correction must be finite BGR triplets")
        corrections[frame_id] = (gain, bias)
    return V2CudaAuditExportContext(
        sources=tuple(sources), strips=tuple(strips), calibration=calibration,
        renderer=str(audit.get("renderer", "unknown_v2_cuda_renderer")) if isinstance(audit, dict) else "unknown_v2_cuda_renderer",
        include_adjacent_corridors=bool(include_adjacent_corridors),
        c7_export_corrections_bgr=corrections or None,
    )


def _v2_quality_metrics(result: object, **route_audit: object) -> dict[str, object]:
    """Evaluate the already-final CUDA output without feeding it back.

    This is a read-only quality decision after the sole final D2H transfers;
    it neither changes RGB/owner pixels nor relaxes the established visual
    thresholds.  A v2 data plane must not be downgraded merely because its
    renderer forgot to expose the project's existing objective evaluator.
    """

    panorama = getattr(result, "panorama_bgr")
    owner = getattr(result, "owner_frame_id")
    metrics, grades = evaluate_visual_metrics(panorama, owner, structural_ok=True)
    return {
        "quality_pass": grades.structural == "A" and grades.visual == "A",
        "objective_visual_metrics": metrics,
        "objective_visual_grades": grades.as_dict(),
        **route_audit,
    }


def is_strict_cuda_strip_owner_implementation(
    *, role: str, implementation_id: str
) -> bool:
    """Return true only for the dedicated strict-owner implementation.

    ``production`` can reach this route only through an already verified,
    immutable production lock.  It never accepts a mutable candidate YAML.
    """

    return role in {"candidate", "production"} and implementation_id == "torch_cuda_strip_owner_v2"


def is_cuda_c1_constrained_owner_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C1 may use its v2 implementation only as a mutable candidate run."""

    return (
        role == "candidate"
        and algorithm_id == "C1_constrained_owner"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c2_dis_residual_mesh_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C2's full CUDA data plane remains candidate-only until selection."""

    return (
        role == "candidate"
        and algorithm_id == "C2_dis_rgb_mesh"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c3_raft_residual_mesh_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C3 remains candidate-only because its RAFT model is candidate evidence."""

    return (
        role == "candidate"
        and algorithm_id == "C3_raft_rgb_mesh"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c4_raft_rgbd_layered_mesh_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C4 is candidate-only and must execute its complete C1+C3+depth route."""

    return (
        role == "candidate"
        and algorithm_id == "C4_raft_rgbd_layered_mesh"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c9_positive_jacobian_line_mesh_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C9 is a candidate-only C4 extension with automatic line evidence."""

    return (
        role == "candidate"
        and algorithm_id == "C9_positive_jacobian_line_mesh"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c5_object_lock_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C5 is candidate-only and requires its entire C1+C3+C4+C5 chain."""

    return (
        role == "candidate"
        and algorithm_id == "C5_object_lock"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c6_safe_multiband_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C6 is candidate-only and must execute the complete C5+C6 chain."""

    return (
        role == "candidate"
        and algorithm_id == "C6_multiband"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c7_photometric_graph_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C7 is candidate-only and executes the complete C1+C3+C4+C5+C6+C7 chain."""

    return (
        role == "candidate"
        and algorithm_id == "C7_photometric_graph"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c8_multilabel_window_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C8 is candidate-only and executes the complete C1+C3+C4+C5+C6+C7+C8 chain."""

    return (
        role == "candidate"
        and algorithm_id == "C8_multilabel_window"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c13_robust_photometric_bundle_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C13 is candidate-only; its robust bundle is never a C7 fallback."""

    return (
        role == "candidate"
        and algorithm_id == "C13_robust_photometric_bundle"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c10_depth_conditioned_layout_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C10 is an experimental candidate-only depth-conditioned layout route."""

    return (
        role == "candidate"
        and algorithm_id == "C10_depth_conditioned_multi_perspective_layout"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c11_object_first_foreground_compositor_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C11 is a candidate-only C10 final-grid foreground compositor."""

    return (
        role == "candidate"
        and algorithm_id == "C11_object_first_single_view_foreground_compositor"
        and implementation_id == "video_visual_renderer_v2"
    )


def is_cuda_c12_joint_owner_final_grid_implementation(
    *, role: str, algorithm_id: str, implementation_id: str
) -> bool:
    """C12 is a candidate-only real 5--7 source final renderer route."""

    return (
        role == "candidate"
        and algorithm_id == "C12_joint_owner_mesh_window"
        and implementation_id == "video_visual_renderer_v2"
    )


def _decode_real_source(
    frame: RGBDFrame, calibration: CameraIntrinsics, pose: np.ndarray,
) -> CudaRealSource:
    color_bgr = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    if color_bgr is None or color_bgr.shape[:2] != (calibration.height, calibration.width):
        raise VideoV2RouteError(f"Could not decode calibrated real RGB source: {frame.color_path}")
    depth = cv2.imread(str(frame.aligned_depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16 or depth.shape != color_bgr.shape[:2]:
        raise VideoV2RouteError(f"Could not decode aligned uint16 depth source: {frame.aligned_depth_path}")
    if not np.isfinite(frame.depth_scale_mm_per_unit) or frame.depth_scale_mm_per_unit <= 0.0:
        raise VideoV2RouteError(f"Real source {frame.frame_id} has no positive depth unit")
    if frame.timestamp_us is None or frame.timestamp_us < 0:
        raise VideoV2RouteError(f"Real source {frame.frame_id} has no non-negative timestamp")
    return CudaRealSource(
        frame_id=int(frame.frame_id),
        timestamp_us=int(frame.timestamp_us),
        color_u8_rgb=np.ascontiguousarray(color_bgr[:, :, ::-1]),
        depth_mm=np.ascontiguousarray(depth.astype(np.float32) * float(frame.depth_scale_mm_per_unit)),
        camera_to_world=np.ascontiguousarray(np.asarray(pose, dtype=np.float64)),
        color_exposure_raw=(int(frame.color_exposure_raw) if frame.color_exposure_raw is not None else None),
    )


def _c1_measurement_projection(
    *,
    annotations: Mapping[str, object] | None,
    strips: Sequence[object],
    sources: Sequence[RGBDFrame],
    calibration: CameraIntrinsics,
    canvas_shape: tuple[int, int],
    final_owner_frame_id: np.ndarray,
    corridor_width_pixels: int,
    final_grid_updates: Sequence[Mapping[str, object]] = (),
) -> tuple[dict[str, object] | None, dict[str, np.ndarray] | None]:
    """Create C1 post-render measurement evidence without touching output.

    This intentionally runs only after the CUDA renderer has returned the
    final CPU provenance map.  The source maps are reconstructed from the
    exact C1 layout/grid equations. Projection is owner-independent, so
    annotations never become an owner/seam or colour input and a later owner
    decision cannot erase the measurement subject.
    """

    if annotations is None:
        return None, None
    frame_ids = {
        int(entry["frame_id"])
        for kind in ("objects", "lines", "safe_background")
        for entry in annotations.get(kind, [])
        if isinstance(entry, Mapping) and isinstance(entry.get("frame_id"), int)
    }
    source_shapes = {
        int(frame.frame_id): (int(calibration.height), int(calibration.width))
        for frame in sources
    }
    inverse_sources = build_v2_c1_calibrated_inverse_sources(
        strips=strips,
        source_shapes=source_shapes,
        canvas_shape=canvas_shape,
        calibration={
            "fx": calibration.fx, "fy": calibration.fy,
            "cx": calibration.cx, "cy": calibration.cy,
            "distortion": calibration.distortion,
        },
        annotation_frame_ids=tuple(sorted(frame_ids)),
        corridor_width_pixels=int(corridor_width_pixels),
    )
    inverse_sources = apply_final_grid_updates(inverse_sources, final_grid_updates)
    payload, masks = build_candidate_annotation_projection(
        annotations,
        sources=inverse_sources,
        final_owner_frame_id=np.asarray(final_owner_frame_id),
        crop_xywh=(0, 0, int(canvas_shape[1]), int(canvas_shape[0])),
        # The v2 C1 route returns its virtual canvas directly.  Unlike the
        # legacy pushbroom branch it performs no presentation-time flip.
        horizontal_flip=False,
    )
    payload = {
        **payload,
        "projection_method": "v2_cuda_final_inverse_grid_owner_independent_consensus",
        "v2_grid_reference": (
            "torch_cuda_final_inverse_grid_with_accepted_local_mesh_deltas"
            if final_grid_updates else "torch_cuda_c1_constrained_owner_v2._render_window"
        ),
        "source_map_count": len(inverse_sources),
        "final_grid_update_count": len(final_grid_updates),
        "final_owner_filter_applied": False,
    }
    return payload, masks


def build_v2_post_publication_measurement_projection(
    *,
    context: V2PostPublicationMeasurementContext,
    annotations: Mapping[str, object],
    final_owner_frame_id: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Project fixed labels only after a v2 primary delivery is immutable."""

    if tuple(int(value) for value in final_owner_frame_id.shape) != tuple(context.canvas_shape):
        raise VideoV2RouteError("post-publication measurement owner shape differs from its frozen v2 canvas")
    payload, masks = _c1_measurement_projection(
        annotations=annotations,
        strips=context.strips,
        sources=context.sources,
        calibration=context.calibration,
        canvas_shape=context.canvas_shape,
        final_owner_frame_id=final_owner_frame_id,
        corridor_width_pixels=context.corridor_width_pixels,
        final_grid_updates=context.final_grid_updates,
    )
    if payload is None or masks is None:
        raise VideoV2RouteError("post-publication v2 measurement projection requires fixed annotations")
    payload = {**payload, "projection_timing": "post_atomic_primary_delivery"}
    return payload, masks


def _post_publication_measurement_context(
    *,
    sources: Sequence[RGBDFrame],
    strips: Sequence[object],
    calibration: CameraIntrinsics,
    result: object,
    final_grid_updates: Sequence[Mapping[str, object]] = (),
) -> V2PostPublicationMeasurementContext:
    return V2PostPublicationMeasurementContext(
        sources=tuple(sources),
        strips=tuple(strips),
        calibration=calibration,
        canvas_shape=tuple(int(value) for value in getattr(result, "owner_frame_id").shape),
        corridor_width_pixels=CudaC1ConstrainedOwnerConfig().corridor_width_pixels,
        final_grid_updates=tuple(dict(item) for item in final_grid_updates),
    )


def _post_publication_measurement_context_from_final_grid(
    *,
    sources: Sequence[RGBDFrame],
    strips: Sequence[object],
    calibration: CameraIntrinsics,
    result: object,
) -> V2PostPublicationMeasurementContext | None:
    """Return a read-only context for C1 plus any actual final-grid deltas."""

    updates = getattr(result, "measurement_grid_updates", ())
    if not isinstance(updates, tuple) or not all(isinstance(item, Mapping) for item in updates):
        raise VideoV2RouteError("v2 renderer final measurement grid updates are malformed")
    return _post_publication_measurement_context(
        sources=sources, strips=strips, calibration=calibration, result=result,
        final_grid_updates=updates,
    )


def _post_publication_measurement_context_if_c1_geometry_is_exact(
    *,
    sources: Sequence[RGBDFrame],
    strips: Sequence[object],
    calibration: CameraIntrinsics,
    result: object,
) -> V2PostPublicationMeasurementContext | None:
    """Keep later C5--C8 evaluation fail-closed until their final grids exist."""

    updates = getattr(result, "measurement_grid_updates", None)
    if updates is None:
        # Older renderer results never emitted final-grid evidence; retain
        # their original C1-only fail-closed behaviour rather than treating
        # an accepted C4 mesh as if it were a nominal C1 sample.
        audit = getattr(result, "algorithm_audit", {})
        c4 = audit.get("c4_raft_rgbd_layered_mesh") if isinstance(audit, Mapping) else None
        if isinstance(c4, Mapping) and int(c4.get("actual_output_mesh_pixel_count", 0)) > 0:
            return None
        updates = ()
    if not isinstance(updates, tuple):
        raise VideoV2RouteError("v2 renderer final measurement grid updates are malformed")
    if updates:
        return None
    return _post_publication_measurement_context(
        sources=sources, strips=strips, calibration=calibration, result=result,
    )


def render_cuda_strict_owner_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Execute one calibrated CUDA strict-owner pass from actual source files.

    The layout is the same audited real-pose/RGB-motion layout used by the
    established renderer, but the CPU renderer is never invoked and supplies
    no RGB samples.  Near-duplicate source nodes retain a real pose and make
    one device-side calibrated remap, but receive no final owner pixels.
    """

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA route needs at least two aligned real sources and poses")
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(
        {**settings.__dict__, "max_pose_count": None}
    )
    scale = estimate_rgb_motion_pixels_per_mm(
        sources,
        camera_to_world,
        calibration,
        settings,
        rgb_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    try:
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(
            _decode_real_source(frame, calibration, pose)
            for frame, pose in zip(sources, camera_to_world, strict=True)
        )
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id),
                right_frame_id=int(right.frame_id),
                risk_level=0,
                flow_backend="none",
                use_raft_backward=False,
                use_depth_mesh=False,
                # Open3D was already executed by the common lifecycle before
                # this renderer begins; this declares that required audit.
                use_open3d=True,
                object_lock_required=False,
                seam_mode="hard_owner",
                blend_mode="none",
            )
            for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaStripOwnerAlgorithm(
            sources=decoded,
            strips=strips,
            output_height=calibration.height,
            output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx,
                "fy": calibration.fy,
                "cx": calibration.cx,
                "cy": calibration.cy,
                "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=1
            ),
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit,
        "layout": layout.as_dict(),
        "rgb_motion_scale": scale.as_dict(),
        "redundant_real_pose_nodes": [dict(value) for value in layout.redundant_pose_node_suppression],
        "quality_metrics": _v2_quality_metrics(result, strict_cuda_owner_route=True),
        "measurement_projection": "not_available_for_strict_owner_c0",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr),
        owner_frame_id=np.ascontiguousarray(result.owner_frame_id),
        metadata=metadata,
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=False,
        ),
    )


def render_cuda_c1_constrained_owner_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    annotations: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Run C1's real-source CUDA owner path; it never falls back to C0."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C1 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources,
        camera_to_world,
        calibration,
        settings,
        rgb_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError(
            "C1 CUDA requires each selected chronological real source to retain a non-empty owner strip"
        )
    try:
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(
            _decode_real_source(frame, calibration, pose)
            for frame, pose in zip(sources, camera_to_world, strict=True)
        )
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id),
                right_frame_id=int(right.frame_id),
                risk_level=1,
                flow_backend="none",
                use_raft_backward=False,
                use_depth_mesh=False,
                use_open3d=True,
                object_lock_required=False,
                seam_mode="curved_hard_owner",
                blend_mode="none",
            )
            for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaC1ConstrainedOwnerAlgorithm(
            sources=decoded,
            strips=strips,
            output_height=calibration.height,
            output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx,
                "fy": calibration.fy,
                "cx": calibration.cx,
                "cy": calibration.cy,
                "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2
            ),
            c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    try:
        projection_payload, projection_masks = _c1_measurement_projection(
            annotations=annotations,
            strips=strips,
            sources=sources,
            calibration=calibration,
            canvas_shape=tuple(int(value) for value in result.owner_frame_id.shape),
            final_owner_frame_id=result.owner_frame_id,
            corridor_width_pixels=resolved_c1_config.corridor_width_pixels,
        )
    except ValueError as exc:
        # Annotation projection is measurement-only.  It must never change a
        # successful C1 primary result, but an invalid projection is not
        # silently represented as usable evidence.
        raise VideoV2RouteError(f"C1 annotation projection failed: {exc}") from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit,
        "layout": layout.as_dict(),
        "rgb_motion_scale": scale.as_dict(),
        "quality_metrics": _v2_quality_metrics(result, cuda_c1_owner_route=True),
        "measurement_projection": (
            "v2_cuda_c1_calibrated_inverse_grid_owner_independent_consensus"
            if projection_payload is not None else "not_requested"
        ),
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr),
        owner_frame_id=np.ascontiguousarray(result.owner_frame_id),
        metadata=metadata,
        measurement_projection_payload=projection_payload,
        measurement_projection_masks=projection_masks,
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c2_dis_residual_mesh_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Run C2's C1 owner route with actual CUDA DIS mesh output sampling."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C2 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources,
        camera_to_world,
        calibration,
        settings,
        rgb_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError(
            "C2 CUDA requires each selected chronological real source to retain a non-empty owner strip"
        )
    try:
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(
            _decode_real_source(frame, calibration, pose)
            for frame, pose in zip(sources, camera_to_world, strict=True)
        )
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id),
                right_frame_id=int(right.frame_id),
                risk_level=1,
                flow_backend="dis",
                use_raft_backward=False,
                use_depth_mesh=False,
                use_open3d=True,
                object_lock_required=False,
                seam_mode="curved_hard_owner",
                blend_mode="none",
            )
            for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaC2DisResidualMeshAlgorithm(
            sources=decoded,
            strips=strips,
            output_height=calibration.height,
            output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx,
                "fy": calibration.fy,
                "cx": calibration.cx,
                "cy": calibration.cy,
                "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2
            ),
            c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit,
        "layout": layout.as_dict(),
        "rgb_motion_scale": scale.as_dict(),
        "quality_metrics": _v2_quality_metrics(result, cuda_c2_dis_mesh_route=True),
        "measurement_projection": "v2_post_publication_final_grid_context",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr),
        owner_frame_id=np.ascontiguousarray(result.owner_frame_id),
        metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_from_final_grid(
            sources=sources, strips=strips, calibration=calibration, result=result,
        ),
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c3_raft_residual_mesh_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Run C3's C1 owner chain plus a locked RAFT-small CUDA mesh.

    The model manifest is resolved again here to turn the candidate's declared
    SHA-256 into the exact local path consumed by RAFT.  This is intentionally
    a second read-only verification boundary: no cache/URL fetch is possible
    inside the renderer.
    """

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C3 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources,
        camera_to_world,
        calibration,
        settings,
        rgb_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError(
            "C3 CUDA requires each selected chronological real source to retain a non-empty owner strip"
        )
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C3 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(
                weights_path=locks[0].path,
                weights_sha256=locks[0].sha256,
                cuda_device=int(cuda_device),
            )
        )
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(
            _decode_real_source(frame, calibration, pose)
            for frame, pose in zip(sources, camera_to_world, strict=True)
        )
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id),
                right_frame_id=int(right.frame_id),
                risk_level=1,
                flow_backend="raft_small",
                use_raft_backward=True,
                use_depth_mesh=False,
                use_open3d=True,
                object_lock_required=False,
                seam_mode="curved_hard_owner",
                blend_mode="none",
            )
            for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaC3RAFTResidualMeshAlgorithm(
            sources=decoded,
            strips=strips,
            output_height=calibration.height,
            output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx,
                "fy": calibration.fy,
                "cx": calibration.cx,
                "cy": calibration.cy,
                "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2
            ),
            raft_runtime=raft_runtime,
            c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit,
        "layout": layout.as_dict(),
        "rgb_motion_scale": scale.as_dict(),
        "quality_metrics": _v2_quality_metrics(result, cuda_c3_raft_mesh_route=True),
        "measurement_projection": "v2_post_publication_final_grid_context",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr),
        owner_frame_id=np.ascontiguousarray(result.owner_frame_id),
        metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_from_final_grid(
            sources=sources, strips=strips, calibration=calibration, result=result,
        ),
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c4_raft_rgbd_layered_mesh_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Execute C4's C1/C3 chain plus calibrated resident depth protection."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C4 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources, camera_to_world, calibration, settings,
        rgb_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError(
            "C4 CUDA requires each selected chronological real source to retain a non-empty owner strip"
        )
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C4 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(
                weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device)
            )
        )
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(
            _decode_real_source(frame, calibration, pose)
            for frame, pose in zip(sources, camera_to_world, strict=True)
        )
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1,
                flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True,
                use_open3d=True, object_lock_required=False, seam_mode="curved_hard_owner", blend_mode="none",
            )
            for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaC4RAFTDepthLayeredMeshAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx,
                "cy": calibration.cy, "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2
            ),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit,
        "layout": layout.as_dict(),
        "rgb_motion_scale": scale.as_dict(),
        "quality_metrics": _v2_quality_metrics(result, cuda_c4_raft_rgbd_layered_mesh_route=True),
        "measurement_projection": "v2_post_publication_final_grid_context",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr),
        owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_from_final_grid(
            sources=sources, strips=strips, calibration=calibration, result=result,
        ),
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c9_positive_jacobian_line_mesh_v2(
    *,
    sources: Sequence[RGBDFrame], camera_to_world: Sequence[np.ndarray], calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object], selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float, c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0, long_line_minimum_length_px: int = 32,
) -> V2CudaStripRender:
    """Run C9's real-source C4 chain with automatic long-line CTF mesh."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C9 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(sources, camera_to_world, calibration, settings, rgb_motions=selected_motions, motion_pixels_to_full_resolution=motion_pixels_to_full_resolution)
    layout = build_calibrated_rgb_pushbroom_layout([frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings)
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError("C9 CUDA requires every selected real source to retain a non-empty owner strip")
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C9 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(RAFTSmallRuntimeConfig(weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device)))
        strips = build_cuda_strips_from_pushbroom_layout(layout, calibration_width=calibration.width, calibration_cx=calibration.cx)
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(PairPlan(left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1, flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True, use_open3d=True, object_lock_required=False, seam_mode="curved_hard_owner", blend_mode="none") for left, right in zip(sources[:-1], sources[1:], strict=True))
        algorithm = TorchCudaC9PositiveJacobianLinePreservingLayeredMeshAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={"fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx, "cy": calibration.cy, "distortion": calibration.distortion},
            runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config, long_line_minimum_length_px=int(long_line_minimum_length_px),
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit, "layout": layout.as_dict(), "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {"renderer_input": False, "post_publication_evaluation_only": True, "protection_input": "aligned_depth_only"},
        "quality_metrics": _v2_quality_metrics(result, cuda_c9_positive_jacobian_line_mesh_route=True),
        "measurement_projection": "not_available_for_cuda_c9",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr), owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_if_c1_geometry_is_exact(sources=sources, strips=strips, calibration=calibration, result=result),
        audit_export_context=_v2_audit_export_context(sources=sources, strips=strips, calibration=calibration, result=result, include_adjacent_corridors=True),
    )


def render_cuda_c10_depth_conditioned_layout_v2(
    *,
    sources: Sequence[RGBDFrame], camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics, pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object], motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None, cuda_device: int = 0,
) -> V2CudaStripRender:
    """Execute the C10 C1+C3+C4 chain with three real-depth layout grids."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C10 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources, camera_to_world, calibration, settings, rgb_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError("C10 CUDA requires every selected real source to retain a non-empty owner strip")
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C10 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device))
        )
        strips = build_cuda_strips_from_pushbroom_layout(layout, calibration_width=calibration.width, calibration_cx=calibration.cx)
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(PairPlan(
            left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1,
            flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True, use_open3d=True,
            object_lock_required=False, seam_mode="curved_hard_owner", blend_mode="none",
        ) for left, right in zip(sources[:-1], sources[1:], strict=True))
        algorithm = TorchCudaC10DepthConditionedLayoutAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={"fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx, "cy": calibration.cy, "distortion": calibration.distortion},
            runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit, "layout": layout.as_dict(), "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {"renderer_input": False, "post_publication_evaluation_only": True, "protection_input": "aligned_depth_only"},
        "quality_metrics": _v2_quality_metrics(result, cuda_c10_depth_conditioned_layout_route=True),
        "measurement_projection": "v2_post_publication_final_grid_context",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr), owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_from_final_grid(sources=sources, strips=strips, calibration=calibration, result=result),
        audit_export_context=_v2_audit_export_context(sources=sources, strips=strips, calibration=calibration, result=result, include_adjacent_corridors=True),
    )


def render_cuda_c12_joint_owner_final_grid_v2(
    *, sources: Sequence[RGBDFrame], camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics, pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object], motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None, cuda_device: int = 0,
) -> V2CudaStripRender:
    """Run C12's genuine 5--7 source resident owner/final-grid renderer."""

    if len(sources) != len(camera_to_world) or len(sources) < 5:
        raise VideoV2RouteError("C12 CUDA requires at least one matching genuine chronological 5--7 source window")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources, camera_to_world, calibration, settings, rgb_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError("C12 CUDA requires every genuine 5--7 source to retain a non-empty owner strip")
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C12 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device))
        )
        strips = build_cuda_strips_from_pushbroom_layout(layout, calibration_width=calibration.width, calibration_cx=calibration.cx)
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(PairPlan(
            left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1,
            flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True, use_open3d=True,
            object_lock_required=False, seam_mode="curved_hard_owner", blend_mode="none",
        ) for left, right in zip(sources[:-1], sources[1:], strict=True))
        algorithm = TorchCudaC12JointOwnerFinalGridAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={"fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx, "cy": calibration.cy, "distortion": calibration.distortion},
            runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=len(decoded)),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit, "layout": layout.as_dict(), "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {"renderer_input": False, "post_publication_evaluation_only": True, "protection_input": "aligned_depth_only"},
        "quality_metrics": _v2_quality_metrics(result, cuda_c12_joint_owner_final_grid_route=True),
        "measurement_projection": "v2_post_publication_final_grid_context",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr), owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_from_final_grid(sources=sources, strips=strips, calibration=calibration, result=result),
        audit_export_context=_v2_audit_export_context(sources=sources, strips=strips, calibration=calibration, result=result, include_adjacent_corridors=True),
    )


def render_cuda_c11_object_first_foreground_compositor_v2(
    *, sources: Sequence[RGBDFrame], camera_to_world: Sequence[np.ndarray], calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object], selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float, c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0, protection_margin_pixels: int = 10,
) -> V2CudaStripRender:
    """Execute C10 final-grid background plus C11 real object-owner tracks."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C11 route needs at least two aligned real sources and poses")
    if not isinstance(protection_margin_pixels, int) or not 8 <= protection_margin_pixels <= 12:
        raise VideoV2RouteError("C11 requires immutable protection_margin_pixels in [8, 12]")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(sources, camera_to_world, calibration, settings, rgb_motions=selected_motions, motion_pixels_to_full_resolution=motion_pixels_to_full_resolution)
    layout = build_calibrated_rgb_pushbroom_layout([frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings)
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError("C11 CUDA requires every selected real source to retain a non-empty owner strip")
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C11 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(RAFTSmallRuntimeConfig(weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device)))
        strips = build_cuda_strips_from_pushbroom_layout(layout, calibration_width=calibration.width, calibration_cx=calibration.cx)
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(PairPlan(int(left.frame_id), int(right.frame_id), 1, "raft_small", True, True, True, False, "curved_hard_owner", "none") for left, right in zip(sources[:-1], sources[1:], strict=True))
        algorithm = TorchCudaC11ObjectFirstForegroundCompositorAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={"fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx, "cy": calibration.cy, "distortion": calibration.distortion},
            runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config, protection_margin_pixels=protection_margin_pixels,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {**result.algorithm_audit, "layout": layout.as_dict(), "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {"renderer_input": False, "post_publication_evaluation_only": True},
        "quality_metrics": _v2_quality_metrics(result, cuda_c11_object_first_foreground_compositor_route=True),
        "measurement_projection": "v2_post_publication_final_grid_context"}
    return V2CudaStripRender(panorama=np.ascontiguousarray(result.panorama_bgr), owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_from_final_grid(sources=sources, strips=strips, calibration=calibration, result=result),
        audit_export_context=_v2_audit_export_context(sources=sources, strips=strips, calibration=calibration, result=result, include_adjacent_corridors=True))


def render_cuda_c5_object_lock_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Execute C5's C4 chain and aligned-depth-derived owner protection.

    Manual measurement annotations are deliberately absent from this route.
    They are allowed only after primary publication by the offline evaluator.
    """

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C5 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources, camera_to_world, calibration, settings,
        rgb_motions=selected_motions, motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError(
            "C5 CUDA requires each selected chronological real source to retain a non-empty owner strip"
        )
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C5 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(
                weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device)
            )
        )
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1,
                flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True,
                use_open3d=True, object_lock_required=True, seam_mode="curved_hard_owner", blend_mode="none",
            )
            for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaC5ObjectLockAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx,
                "cy": calibration.cy, "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2
            ),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit,
        "layout": layout.as_dict(),
        "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {
            "renderer_input": False,
            "post_publication_evaluation_only": True,
            "protection_input": "aligned_depth_only",
        },
        "quality_metrics": _v2_quality_metrics(result, cuda_c5_object_lock_route=True),
        "measurement_projection": "not_available_for_cuda_c5",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr),
        owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_if_c1_geometry_is_exact(
            sources=sources, strips=strips, calibration=calibration, result=result,
        ),
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c6_safe_multiband_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Execute C5 first, then bounded CUDA-only safe-background C6 blending."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("v2 CUDA C6 route needs at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources, camera_to_world, calibration, settings,
        rgb_motions=selected_motions, motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError(
            "C6 CUDA requires each selected chronological real source to retain a non-empty owner strip"
        )
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C6 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(
                weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device)
            )
        )
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1,
                flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True,
                use_open3d=True, object_lock_required=True, seam_mode="curved_hard_owner",
                blend_mode="safe_multiband",
            )
            for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaC6SafeMultiBandAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx,
                "cy": calibration.cy, "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=2
            ),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit,
        "layout": layout.as_dict(),
        "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {
            "renderer_input": False,
            "post_publication_evaluation_only": True,
            "protection_input": "aligned_depth_only",
        },
        "quality_metrics": _v2_quality_metrics(result, cuda_c6_safe_multiband_route=True),
        "measurement_projection": "not_available_for_cuda_c6",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr),
        owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_if_c1_geometry_is_exact(
            sources=sources, strips=strips, calibration=calibration, result=result,
        ),
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c7_photometric_graph_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Execute C6 then its bounded global CUDA photometric C7 extension."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError(
            "C7 CUDA requires at least two aligned real sources/poses for its global fit"
        )
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources, camera_to_world, calibration, settings,
        rgb_motions=selected_motions, motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError("C7 CUDA requires every selected chronological real source to retain a non-empty owner strip")
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C7 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device))
        )
        strips = build_cuda_strips_from_pushbroom_layout(layout, calibration_width=calibration.width, calibration_cx=calibration.cx)
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1,
                flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True,
                use_open3d=True, object_lock_required=True, seam_mode="curved_hard_owner", blend_mode="safe_multiband",
            ) for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        algorithm = TorchCudaC7PhotometricGraphAlgorithm(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={"fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx, "cy": calibration.cy, "distortion": calibration.distortion},
            # C7's global graph needs every selected real source to remain
            # resident from evidence fitting through final composition.  This
            # is derived from the audited sequence, never a fixed source
            # count or a source-dropping resampler; the renderer still checks
            # all established working-set/resource limits before upload.
            runtime_config=VideoGpuRuntimeConfig(cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=len(decoded)),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit, "layout": layout.as_dict(), "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {
            "renderer_input": False,
            "post_publication_evaluation_only": True,
            "protection_input": "aligned_depth_only",
        },
        "quality_metrics": _v2_quality_metrics(result, cuda_c7_photometric_graph_route=True),
        "measurement_projection": "not_available_for_cuda_c7",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr), owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_if_c1_geometry_is_exact(
            sources=sources, strips=strips, calibration=calibration, result=result,
        ),
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c8_multilabel_window_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
    _algorithm_type: type = TorchCudaC8MultilabelWindowAlgorithm,
) -> V2CudaStripRender:
    """Execute C7/C6 and bounded chronological real-owner C8 recomposition."""

    if len(sources) != len(camera_to_world) or len(sources) < 2:
        raise VideoV2RouteError("C8 CUDA requires at least two aligned real sources and poses")
    resolved_c1_config = _c1_config_from_mapping(c1_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping(pushbroom_config)
    settings = CalibratedRGBPushbroomConfig.from_mapping({**settings.__dict__, "max_pose_count": None})
    scale = estimate_rgb_motion_pixels_per_mm(
        sources, camera_to_world, calibration, settings,
        rgb_motions=selected_motions, motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in sources], camera_to_world, calibration, scale, settings
    )
    if layout.redundant_pose_node_suppression:
        raise VideoV2RouteError("C8 CUDA requires every selected chronological real source to retain a non-empty owner strip")
    model_id = "torchvision_raft_small_C_T_V2"
    expected_sha256 = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
    try:
        locks = verify_candidate_models({model_id: expected_sha256})
        if len(locks) != 1 or locks[0].model_id != model_id:
            raise VideoV2RouteError("C8 RAFT-small model lock is absent or ambiguous")
        raft_runtime = TorchvisionRAFTSmallRuntime(
            RAFTSmallRuntimeConfig(
                weights_path=locks[0].path, weights_sha256=locks[0].sha256, cuda_device=int(cuda_device)
            )
        )
        strips = build_cuda_strips_from_pushbroom_layout(
            layout, calibration_width=calibration.width, calibration_cx=calibration.cx
        )
        decoded = tuple(_decode_real_source(frame, calibration, pose) for frame, pose in zip(sources, camera_to_world, strict=True))
        plans = tuple(
            PairPlan(
                left_frame_id=int(left.frame_id), right_frame_id=int(right.frame_id), risk_level=1,
                flow_backend="raft_small", use_raft_backward=True, use_depth_mesh=True,
                use_open3d=True, object_lock_required=True, seam_mode="curved_hard_owner", blend_mode="safe_multiband",
            ) for left, right in zip(sources[:-1], sources[1:], strict=True)
        )
        # C8 evaluates one bounded owner window at a time, but it retains all
        # selected real sources through final recomposition.  This is derived
        # from the actual sequence, not a fixed resampling cap: evicting an
        # old source would require an impermissible second H2D upload.
        algorithm = _algorithm_type(
            sources=decoded, strips=strips, output_height=calibration.height, output_width=layout.canvas_width,
            calibration={
                "fx": calibration.fx, "fy": calibration.fy, "cx": calibration.cx,
                "cy": calibration.cy, "distortion": calibration.distortion,
            },
            runtime_config=VideoGpuRuntimeConfig(
                cuda_mode="required", cuda_device=int(cuda_device), maximum_resident_frames=len(decoded)
            ),
            raft_runtime=raft_runtime, c1_config=resolved_c1_config,
        )
        prepared = algorithm.prepare(session=None, online_state=None, context={"pair_plans": plans})
        result = algorithm.render(prepared)
    except (VideoAlgorithmContractError, ValueError) as exc:
        raise VideoV2RouteError(str(exc)) from exc
    metadata: dict[str, object] = {
        **result.algorithm_audit, "layout": layout.as_dict(), "rgb_motion_scale": scale.as_dict(),
        "manual_measurement_annotations": {
            "renderer_input": False,
            "post_publication_evaluation_only": True,
            "protection_input": "aligned_depth_only",
        },
        "quality_metrics": _v2_quality_metrics(result, cuda_c8_multilabel_window_route=True),
        "measurement_projection": "not_available_for_cuda_c8",
    }
    return V2CudaStripRender(
        panorama=np.ascontiguousarray(result.panorama_bgr), owner_frame_id=np.ascontiguousarray(result.owner_frame_id), metadata=metadata,
        post_publication_measurement_context=_post_publication_measurement_context_if_c1_geometry_is_exact(
            sources=sources, strips=strips, calibration=calibration, result=result,
        ),
        audit_export_context=_v2_audit_export_context(
            sources=sources, strips=strips, calibration=calibration, result=result,
            include_adjacent_corridors=True,
        ),
    )


def render_cuda_c13_robust_photometric_bundle_v2(
    *,
    sources: Sequence[RGBDFrame],
    camera_to_world: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    pushbroom_config: Mapping[str, object],
    selected_motions: Sequence[object],
    motion_pixels_to_full_resolution: float,
    c1_config: Mapping[str, object] | None = None,
    cuda_device: int = 0,
) -> V2CudaStripRender:
    """Run C13's actual C8 lineage plus robust pre-seam field, not C8 alone."""

    result = render_cuda_c8_multilabel_window_v2(
        sources=sources, camera_to_world=camera_to_world, calibration=calibration,
        pushbroom_config=pushbroom_config, selected_motions=selected_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
        c1_config=c1_config, cuda_device=cuda_device,
        _algorithm_type=TorchCudaC13RobustPhotometricBundleAlgorithm,
    )
    # The immutable C13 component is present only if the C13 renderer made
    # final-output audit evidence.  Do not paper over a C13 identity/reject
    # with the C8 route label.
    execution = result.metadata.get("component_execution")
    if not isinstance(execution, dict) or "c13_robust_photometric_bundle" not in execution:
        raise VideoV2RouteError("C13 CUDA renderer omitted robust photometric bundle execution evidence")
    result.metadata["cuda_c13_robust_photometric_bundle_route"] = True
    return result


__all__ = [
    "V2CudaStripRender",
    "V2PostPublicationMeasurementContext",
    "VideoV2RouteError",
    "build_v2_post_publication_measurement_projection",
    "is_strict_cuda_strip_owner_implementation",
    "is_cuda_c1_constrained_owner_implementation",
    "is_cuda_c2_dis_residual_mesh_implementation",
    "is_cuda_c3_raft_residual_mesh_implementation",
    "is_cuda_c4_raft_rgbd_layered_mesh_implementation",
    "is_cuda_c9_positive_jacobian_line_mesh_implementation",
    "is_cuda_c5_object_lock_implementation",
    "is_cuda_c6_safe_multiband_implementation",
    "is_cuda_c7_photometric_graph_implementation",
    "is_cuda_c8_multilabel_window_implementation",
    "is_cuda_c13_robust_photometric_bundle_implementation",
    "is_cuda_c10_depth_conditioned_layout_implementation",
    "is_cuda_c11_object_first_foreground_compositor_implementation",
    "is_cuda_c12_joint_owner_final_grid_implementation",
    "render_cuda_c10_depth_conditioned_layout_v2",
    "render_cuda_c11_object_first_foreground_compositor_v2",
    "render_cuda_c12_joint_owner_final_grid_v2",
    "render_cuda_c8_multilabel_window_v2",
    "render_cuda_c13_robust_photometric_bundle_v2",
    "render_cuda_c7_photometric_graph_v2",
    "render_cuda_c6_safe_multiband_v2",
    "render_cuda_c5_object_lock_v2",
    "render_cuda_c4_raft_rgbd_layered_mesh_v2",
    "render_cuda_c9_positive_jacobian_line_mesh_v2",
    "render_cuda_c3_raft_residual_mesh_v2",
    "render_cuda_c2_dis_residual_mesh_v2",
    "render_cuda_c1_constrained_owner_v2",
    "render_cuda_strict_owner_v2",
]
