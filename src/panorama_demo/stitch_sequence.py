from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import cv2
import numpy as np

from .calibrated_rgb_pushbroom import (
    _BOUNDARY_HIGH_RISK_TOPOLOGY_POLICY,
    _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_PIXELS,
    _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROWS,
    _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROW_SPAN,
    _PREVIEW_RISK_TRIGGER_POLICY,
    GeometryAssistedSeamConfig,
    render_calibrated_rgb_pushbroom,
)
from .config import load_config
from .rgb_residual_alignment import ResidualAlignmentConfig
from .orbslam3_bridge import (
    ORBSLAM3PoseGraphOptimizer,
    ORBSLAM3Trajectory,
    run_orbslam3_rgbd,
)
from .quality import (
    FrameQuality,
    MotionEstimate,
    assess_capture_quality,
    analyze_frame_quality,
    estimate_translation,
    resize_for_analysis,
    select_layout_from_motion_estimates,
    select_primary_scan_segment,
)
from .rgbd_odometry import (
    PoseGraphConfig,
    PoseQualityThresholds,
    RGBDOdometryConfig,
    estimate_pair_rgbd_odometry,
    optimize_rgbd_pose_graph,
    validate_pose_trajectory,
)
from .session import (
    CameraIntrinsics,
    RGBDFrame,
    load_rgbd_session,
    read_aligned_depth_mm,
)

if TYPE_CHECKING:
    # These types belong only to dormant legacy helpers used by the isolated
    # central-strip diagnostic and a focused compatibility test.  Keeping the
    # imports type-only ensures the formal RGB pushbroom route cannot import
    # the RGB-D projection module at process start.
    from .rgbd_projection import PinholeIntrinsics, RGBDProjectionFrame


_DELIVERY_FILES = (
    # The success marker must be invalidated before every other cleanup.
    "delivery.json",
    "panorama.jpg",
    "foreground_mask.png",
    "background_exclusion_mask.png",
    "tsdf_foreground_mask.png",
    "depth_fallback_mask.png",
    "depth_multiview_foreground_mask.png",
    "foreground_alpha.png",
    "foreground_source_id.png",
    "foreground_confidence.png",
    "background_source_id.png",
    "tsdf_mesh.glb",
    "tsdf_mesh_viewer.html",
    "report.json",
    "transforms.json",
    "render_transforms.json",
)

_DIAGNOSTIC_FILES = (
    "diagnostic_panorama.jpg",
    "diagnostic_foreground_mask.png",
    "diagnostic_tsdf_mesh.glb",
    "diagnostic_tsdf_mesh_viewer.html",
    "diagnostic_report.json",
)

_HARD_MAX_CANVAS_MEGAPIXELS = 200.0
_HARD_MAX_LAYOUT_FRAMES = 160
_HARD_MAX_RENDER_SOURCES = 32

_CENTRAL_STRIP_DIAGNOSTIC_DEFAULTS: dict[str, object] = {
    # This is deliberately disabled in the shared configuration.  The only
    # activation path is an injected renderer from the independent diagnostic
    # command below; g305-panorama never imports or injects that renderer.
    "enabled": False,
    "reference_scale_mode": "robust_aligned_depth_plane",
    "orientation_mode": "verified_camera_to_world",
    "maximum_central_band_fraction": 0.20,
    "minimum_pair_overlap_pixels": 96,
    "exposure_mode": "global_gain",
    "multiband_levels": 5,
}
_CENTRAL_STRIP_DIAGNOSTIC_KEYS = frozenset(_CENTRAL_STRIP_DIAGNOSTIC_DEFAULTS)


def _invalidate_delivery_marker(output: Path) -> None:
    """Invalidate any previous success before performing other task work."""

    output.mkdir(parents=True, exist_ok=True)
    (output / "delivery.json").unlink(missing_ok=True)


def _capture_manifest_summary(
    manifest: dict[str, Any] | None,
) -> dict[str, object]:
    if manifest is None:
        return {
            "capture_mode": "legacy_or_unknown",
            "diagnostic_only": False,
            "formal_stitch_allowed": True,
        }
    diagnostic_marker = manifest.get("diagnostic_only", False)
    formal_allowed = manifest.get("formal_stitch_allowed", True)
    if not isinstance(diagnostic_marker, bool):
        raise ValueError("Session manifest diagnostic_only must be a boolean")
    if not isinstance(formal_allowed, bool):
        raise ValueError("Session manifest formal_stitch_allowed must be a boolean")
    capture_mode = str(manifest.get("capture_mode", "legacy_or_unknown"))
    options = manifest.get("capture_options", {})
    option_marker = False
    if isinstance(options, dict):
        option_marker = options.get("diagnostic_unrestricted_auto_exposure", False)
        if not isinstance(option_marker, bool):
            raise ValueError(
                "Session manifest diagnostic exposure marker must be a boolean"
            )
    diagnostic_only = bool(
        diagnostic_marker
        or not formal_allowed
        or option_marker
        or capture_mode == "diagnostic_unrestricted_auto_exposure"
    )
    return {
        "capture_mode": capture_mode,
        "diagnostic_only": diagnostic_only,
        "formal_stitch_allowed": not diagnostic_only,
    }


def _clear_delivery_files(output: Path) -> None:
    _invalidate_delivery_marker(output)
    for name in _DELIVERY_FILES[1:]:
        (output / name).unlink(missing_ok=True)
    for pending in output.glob(".*.pending.*"):
        if pending.is_file():
            pending.unlink()
    # ORB-SLAM3 staging is never a deliverable.  Older runs created it below
    # ``output``; clean that known child atomically with the rest of failed or
    # superseded work without following a user-created symlink outside output.
    staging = output / ".orbslam3_rgbd"
    if staging.is_symlink() or staging.is_file():
        staging.unlink(missing_ok=True)
    elif staging.is_dir():
        shutil.rmtree(staging)


def _clear_diagnostic_files(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in _DIAGNOSTIC_FILES:
        (output / name).unlink(missing_ok=True)


def _write_failure_report(output: Path, input_path: Path, exc: Exception) -> None:
    _clear_delivery_files(output)
    _clear_diagnostic_files(output)
    payload = {
        "schema": "gemini305-panorama-failure/v2",
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.expanduser().resolve()),
        "error_type": type(exc).__name__,
        "message": str(exc),
        "deliverable_published": False,
    }
    attempt_audit = getattr(exc, "attempt_audit", ())
    if isinstance(attempt_audit, (list, tuple)):
        compact_attempt_audit = [
            dict(row) for row in attempt_audit if isinstance(row, Mapping)
        ]
        if compact_attempt_audit:
            # Keep the native process facts that explain a bounded retry, but
            # never leak a temporary RGB-D staging location into the output.
            payload["orbslam3_execution_attempts"] = compact_attempt_audit
    pending = output / ".failure.pending.json"
    pending.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(pending, output / "failure.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a quality-gated side-scan panorama from a calibrated, "
            "color-aligned Gemini 305 RGB-D session"
        )
    )
    parser.add_argument("input", type=Path, help="Calibrated RGB-D capture session")
    parser.add_argument("--output", type=Path, default=Path("outputs/sequence"))
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--render-frame-ids",
        help=(
            "Diagnostic-only comma-separated pose-node override; formal delivery "
            "always uses automatic full-coverage selection"
        ),
    )
    parser.add_argument(
        "--diagnostic-force",
        action="store_true",
        help=(
            "Bypass input, odometry-quality and final image-quality thresholds, "
            "but keep calibration, aligned depth, finite SE(3), graph connectivity, "
            "projection, topology, memory and atomic-delivery safety"
        ),
    )
    return parser


def _parse_frame_ids(value: object) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        result = [int(part) for part in parts]
    elif isinstance(value, (list, tuple)):
        result = [int(item) for item in value]
    else:
        raise ValueError("render_frame_ids must be a comma-separated string or list")
    if len(result) != len(set(result)):
        raise ValueError("render_frame_ids cannot contain duplicates")
    if len(result) == 1:
        raise ValueError("render_frame_ids must contain at least two frame ids")
    return sorted(result)


def _ensure_publishable_quality(
    capture_quality: dict[str, object],
    render_metadata: dict[str, Any],
    pose_quality: dict[str, Any] | None = None,
) -> None:
    """Final assertion that diagnostic overrides can never publish a delivery."""

    failures: list[str] = []
    if not bool(capture_quality.get("quality_pass", False)):
        failures.append("input capture quality did not pass")
    if pose_quality is not None and not bool(pose_quality.get("quality_pass", False)):
        failures.append("RGB-D pose trajectory quality did not pass")
    render_quality = render_metadata.get("quality_metrics")
    if not isinstance(render_quality, dict) or not bool(
        render_quality.get("quality_pass", False)
    ):
        failures.append("final render quality did not pass")
    if failures:
        raise RuntimeError("Delivery quality gate failed: " + "; ".join(failures))


def _compact_residual_alignment_for_transforms(
    render_metadata: dict[str, Any],
    settings: ResidualAlignmentConfig,
    frame_ids: list[int],
) -> dict[str, object]:
    """Return the reproducible, non-dense residual-map audit sidecar payload.

    ``report.json`` keeps the complete scalar evidence audit.  The standalone
    render-transform sidecar needs only the selected model, small per-source
    parameters, and the structural/held-out summaries required to reproduce
    the calibrated composite inverse map.  Preview RGB, flow, masks, and
    dense inverse maps must never escape the renderer's temporary workspace.
    """

    residual = render_metadata.get("residual_alignment")
    if not isinstance(residual, dict):
        raise RuntimeError(
            "Calibrated RGB pushbroom omitted the required residual alignment audit"
        )
    backend = residual.get("backend")
    if backend != settings.backend:
        raise RuntimeError("Residual alignment audit backend disagrees with config")
    selected_model = residual.get("selected_model")
    if settings.background_model != "identity" or selected_model != "identity":
        raise RuntimeError(
            "Formal residual-alignment sidecar must remain identity; "
            "verified RGB-D SE(3) is the sole global geometry"
        )

    preview_count = residual.get("preview_remap_count")
    full_resolution_count = residual.get("full_resolution_output_remap_count")
    expected_count = len(frame_ids)
    if (
        not isinstance(preview_count, int)
        or preview_count != expected_count
        or not isinstance(full_resolution_count, int)
        or full_resolution_count != expected_count
    ):
        raise RuntimeError(
            "Residual alignment audit did not account for every real source remap"
        )

    parameters = residual.get("per_source_parameters")
    if not isinstance(parameters, list) or len(parameters) != expected_count:
        raise RuntimeError(
            "Residual alignment audit has no one-to-one source parameters"
        )
    per_source_parameters: list[dict[str, object]] = []
    for source_index, (frame_id, parameter) in enumerate(
        zip(frame_ids, parameters, strict=True)
    ):
        if (
            not isinstance(parameter, dict)
            or parameter.get("source_index") != source_index
        ):
            raise RuntimeError(
                "Residual alignment audit source parameters are not in render order"
            )
        if parameter.get("identity") is not True:
            raise RuntimeError("Residual alignment sidecar contains a non-identity warp")
        for name in (
            "translation_x_pixels",
            "translation_y_pixels",
            "roll_degrees",
        ):
            value = parameter.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != 0.0
            ):
                raise RuntimeError(
                    "Residual alignment sidecar contains a non-identity global transform"
                )
        per_source_parameters.append({"frame_id": frame_id, **parameter})

    held_out_before = residual.get("held_out_metrics_before")
    held_out_after = residual.get("held_out_metrics_after")
    component_audit = residual.get("component_audit")
    topology_audit = residual.get("topology_audit")
    working_set_audit = residual.get("working_set_audit")
    if (
        not isinstance(held_out_before, dict)
        or not isinstance(held_out_after, dict)
        or not isinstance(component_audit, dict)
        or not isinstance(topology_audit, dict)
        or not isinstance(working_set_audit, dict)
        or topology_audit.get("accepted") is not True
    ):
        raise RuntimeError("Residual alignment structural audit is incomplete")

    return {
        "backend": backend,
        "selected_model": selected_model,
        "configuration": settings.as_dict(),
        "preview_remap_count": preview_count,
        "full_resolution_output_remap_count": full_resolution_count,
        "per_source_parameters": per_source_parameters,
        "held_out_metrics_before": dict(held_out_before),
        "held_out_metrics_after": dict(held_out_after),
        "component_audit": dict(component_audit),
        "topology_audit": dict(topology_audit),
        "working_set_audit": dict(working_set_audit),
    }


def _compact_geometry_assistance_for_transforms(
    render_metadata: dict[str, Any],
    settings: GeometryAssistedSeamConfig,
    frame_ids: list[int],
) -> dict[str, object]:
    """Publish scalar local-geometry evidence without paths, maps, or depth.

    The renderer's temporary geometry masks and aligned depth frames are never
    part of reproducibility sidecars.  The compact payload proves that depth
    was limited to adjacent seam geometry and that final colour remained RGB.
    """

    geometry = render_metadata.get("geometry_assisted_seam")
    if not isinstance(geometry, dict):
        raise RuntimeError("Calibrated pushbroom omitted geometry-assistance audit")
    if geometry.get("scope") != "adjacent_seam_corridors_only":
        raise RuntimeError("Geometry assistance scope is not adjacent seam corridors")
    if geometry.get("depth_used_for_output_pixels") is not False:
        raise RuntimeError("Geometry assistance incorrectly claims depth output pixels")
    config = geometry.get("config")
    if not isinstance(config, dict) or config != settings.as_dict():
        raise RuntimeError("Geometry assistance audit configuration disagrees with config")
    pairs = geometry.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(frame_ids) - 1:
        raise RuntimeError("Geometry assistance audit does not cover all adjacent pairs")
    compact_pairs: list[dict[str, object]] = []
    forbidden_keys = {
        "aligned_depth_path",
        "raw_depth_path",
        "depth_path",
        "source_map_x",
        "source_map_y",
        "depth_mm",
        "protected_mask",
        "active_mask",
    }

    def compact_scalar_audit(value: object, path: tuple[str, ...] = ()) -> object:
        """Copy only finite scalar evidence; reject nested dense geometry."""

        if value is None or isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(
                    "Geometry assistance sidecar contains a non-finite scalar audit value"
                )
            return numeric
        if isinstance(value, str):
            if "/" in value or "\\" in value:
                raise RuntimeError(
                    "Geometry assistance sidecar attempted to publish a path"
                )
            return value
        if isinstance(value, np.ndarray):
            raise RuntimeError("Geometry assistance sidecar attempted to publish dense data")
        if isinstance(value, (list, tuple)):
            # Scalar audits may expose the small reason vocabulary, but never
            # arbitrary nested Python lists: those could smuggle an otherwise
            # rejected dense mask/depth map through JSON serialization.
            if not path or path[-1] != "trigger_reasons":
                raise RuntimeError(
                    "Geometry assistance sidecar attempted to publish dense list data"
                )
            return [
                compact_scalar_audit(item, (*path, str(index)))
                for index, item in enumerate(value)
            ]
        if isinstance(value, dict):
            compact: dict[str, object] = {}
            for key, nested in value.items():
                if not isinstance(key, str):
                    raise RuntimeError("Geometry assistance audit keys must be strings")
                normalised = key.lower()
                if (
                    key in forbidden_keys
                    or normalised.endswith("_path")
                    or normalised.endswith("_mask")
                    or normalised.endswith("_map")
                    or normalised.endswith("_map_x")
                    or normalised.endswith("_map_y")
                    or normalised in {"depth_mm", "depth_image", "depth_frame"}
                ):
                    raise RuntimeError(
                        "Geometry assistance sidecar attempted to publish dense data"
                    )
                compact[key] = compact_scalar_audit(nested, (*path, key))
            return compact
        raise RuntimeError(
            "Geometry assistance sidecar contains a non-scalar audit value"
        )

    def required_finite_scalar(
        mapping: Mapping[str, object], key: str, *, context: str
    ) -> float:
        value = mapping.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RuntimeError(f"{context} lacks finite {key}")
        return float(value)

    def required_integer(
        mapping: Mapping[str, object], key: str, *, context: str
    ) -> int:
        value = mapping.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"{context} lacks integer {key}")
        return int(value)

    def required_boolean(
        mapping: Mapping[str, object], key: str, *, context: str
    ) -> bool:
        value = mapping.get(key)
        if type(value) is not bool:
            raise RuntimeError(f"{context} lacks boolean {key}")
        return bool(value)

    def validate_accepted_trigger_audit(audit: Mapping[str, object]) -> None:
        """Bind a published mesh to its narrow structural trigger evidence."""

        context = "Accepted geometry trigger audit"
        reasons = audit.get("trigger_reasons")
        allowed_reasons = {
            "boundary_edge_normal_step_p95",
            "boundary_edge_normal_step_max",
            "boundary_high_rgb_risk",
            "preview_full_height_hard_cut",
        }
        if (
            not isinstance(reasons, list)
            or not reasons
            or any(not isinstance(reason, str) for reason in reasons)
            or any(reason not in allowed_reasons for reason in reasons)
            or len(set(reasons)) != len(reasons)
        ):
            raise RuntimeError("Accepted geometry mesh lacks a valid trigger reason")
        if audit.get("preview_risk_policy") != _PREVIEW_RISK_TRIGGER_POLICY:
            raise RuntimeError("Accepted geometry mesh lacks RGB-risk trigger policy")
        if (
            audit.get("boundary_high_risk_topology_policy")
            != _BOUNDARY_HIGH_RISK_TOPOLOGY_POLICY
        ):
            raise RuntimeError(
                "Accepted geometry mesh lacks structural RGB-risk trigger policy"
            )
        preview_scale = required_finite_scalar(
            audit, "preview_risk_preview_scale", context=context
        )
        if not math.isclose(
            preview_scale,
            float(settings.flow_validation_preview_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Accepted geometry mesh has inconsistent trigger scale")
        expected_minimums = {
            "minimum_boundary_high_risk_component_pixel_count": max(
                1,
                int(
                    math.ceil(
                        _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_PIXELS
                        * preview_scale
                        * preview_scale
                    )
                ),
            ),
            "minimum_boundary_high_risk_component_row_count": max(
                1,
                int(
                    math.ceil(
                        _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROWS * preview_scale
                    )
                ),
            ),
            "minimum_boundary_high_risk_component_row_span": max(
                1,
                int(
                    math.ceil(
                        _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROW_SPAN
                        * preview_scale
                    )
                ),
            ),
        }
        if any(
            required_integer(audit, key, context=context) != expected
            for key, expected in expected_minimums.items()
        ):
            raise RuntimeError(
                "Accepted geometry mesh has inconsistent structural-risk threshold"
            )
        component_count = required_integer(
            audit, "boundary_risk_component_count", context=context
        )
        centreline_count = required_integer(
            audit,
            "boundary_centreline_touching_risk_component_count",
            context=context,
        )
        qualifying_count = required_integer(
            audit, "boundary_qualifying_risk_component_count", context=context
        )
        high_risk = required_boolean(audit, "boundary_high_rgb_risk", context=context)
        if (
            component_count < 0
            or centreline_count < 0
            or qualifying_count < 0
            or centreline_count > component_count
            or qualifying_count > centreline_count
            or high_risk != bool(qualifying_count)
            or ("boundary_high_rgb_risk" in reasons) != high_risk
        ):
            raise RuntimeError("Accepted geometry mesh has invalid structural-risk topology")
        for key, minimum_key in (
            (
                "boundary_largest_qualifying_risk_component_pixel_count",
                "minimum_boundary_high_risk_component_pixel_count",
            ),
            (
                "boundary_largest_qualifying_risk_component_row_count",
                "minimum_boundary_high_risk_component_row_count",
            ),
            (
                "boundary_largest_qualifying_risk_component_row_span",
                "minimum_boundary_high_risk_component_row_span",
            ),
        ):
            value = required_integer(audit, key, context=context)
            minimum = required_integer(audit, minimum_key, context=context)
            if (qualifying_count == 0 and value != 0) or (
                qualifying_count > 0 and value < minimum
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has invalid qualifying risk support"
                )
        full_hard_cut = required_boolean(
            audit, "preview_full_height_hard_cut", context=context
        )
        probe_status = audit.get("preview_owner_probe_status")
        probe_graphcut = required_boolean(
            audit, "preview_owner_probe_graphcut_used", context=context
        )
        hard_cut_rows = required_integer(
            audit, "preview_hard_cut_row_count", context=context
        )
        common_rows = required_integer(audit, "common_row_count", context=context)
        if (
            not isinstance(probe_status, str)
            or not probe_status
            or hard_cut_rows > common_rows
            or ("preview_full_height_hard_cut" in reasons) != full_hard_cut
            or (full_hard_cut and not high_risk)
            or (
                not high_risk
                and (
                    hard_cut_rows != 0
                    or full_hard_cut
                    or probe_graphcut
                    or probe_status
                    not in {
                        "not_needed_no_boundary_rgb_risk",
                        "not_needed_unqualified_boundary_rgb_risk",
                    }
                )
            )
        ):
            raise RuntimeError("Accepted geometry mesh has inconsistent owner probe")

    def validate_accepted_depth_layer_audit(audit: Mapping[str, object]) -> None:
        """Require scalar proof that a mesh stayed on one safe wall layer."""

        context = "Accepted geometry depth-layer audit"
        pair_geometry = audit.get("geometry")
        if not isinstance(pair_geometry, dict):
            raise RuntimeError("Accepted geometry mesh lacks bidirectional depth audit")
        for direction in ("first_to_second", "second_to_first"):
            directed = pair_geometry.get(direction)
            if not isinstance(directed, dict):
                raise RuntimeError("Accepted geometry mesh lacks directed depth audit")
            mutual_count = required_integer(
                directed,
                "mutual_consistent_pixel_count",
                context=context,
            )
            residual_p95 = required_finite_scalar(
                directed,
                "mutual_depth_residual_ratio_p95",
                context=context,
            )
            residual_maximum = required_finite_scalar(
                directed,
                "mutual_depth_residual_ratio_max",
                context=context,
            )
            if (
                mutual_count <= 0
                or residual_p95 < 0.0
                or residual_maximum < residual_p95
                or residual_maximum > 1.0 + 1e-6
            ):
                raise RuntimeError(
                    "Accepted geometry mesh violates mutual depth-residual gate"
                )
        for key in ("first_surface_safety", "second_surface_safety"):
            safety = pair_geometry.get(key)
            if (
                not isinstance(safety, dict)
                or safety.get("policy")
                != (
                    "one_dominant_far_depth_component_mesh_safe_"
                    "near_and_ambiguous_components_hard_owner"
                )
            ):
                raise RuntimeError(
                    "Accepted geometry mesh lacks conservative surface-safety audit"
                )
            mesh_safe_count = required_integer(
                safety, "mesh_safe_pixel_count", context=context
            )
            dominant_count = required_integer(
                safety, "dominant_component_pixel_count", context=context
            )
            hard_owner_count = required_integer(
                safety, "hard_owner_pixel_count", context=context
            )
            near_count = required_integer(
                safety, "near_foreground_pixel_count", context=context
            )
            ambiguous_count = required_integer(
                safety,
                "ambiguous_or_unreliable_pixel_count",
                context=context,
            )
            component_count = required_integer(
                safety, "component_count", context=context
            )
            material_component_count = required_integer(
                safety, "material_component_count", context=context
            )
            near_component_count = required_integer(
                safety,
                "near_foreground_component_count",
                context=context,
            )
            depth_anchor_count = required_integer(
                safety,
                "depth_anchor_component_count",
                context=context,
            )
            analysis_count = required_integer(
                safety, "analysis_pixel_count", context=context
            )
            base_safe_count = required_integer(
                safety, "base_safe_pixel_count", context=context
            )
            dominant_fraction = required_finite_scalar(
                safety, "dominant_component_fraction", context=context
            )
            dominant_depth = safety.get("dominant_component_median_depth_mm")
            if (
                safety.get("analysis_scope") != "adjacent_seam_raw_footprint"
                or mesh_safe_count <= 0
                or dominant_count != mesh_safe_count
                or hard_owner_count < 0
                or hard_owner_count != near_count + ambiguous_count
                or near_count < 0
                or ambiguous_count < 0
                or component_count < 0
                or material_component_count < 0
                or near_component_count < 0
                or near_component_count > material_component_count
                or material_component_count > component_count
                or depth_anchor_count <= 0
                or depth_anchor_count > material_component_count
                or analysis_count <= 0
                or base_safe_count <= 0
                or base_safe_count > analysis_count
                or not 0.50 <= dominant_fraction <= 1.0
                or isinstance(dominant_depth, bool)
                or not isinstance(dominant_depth, (int, float))
                or not math.isfinite(float(dominant_depth))
                or float(dominant_depth) <= 0.0
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has invalid surface-safety support"
                )
        component = audit.get("virtual_background_component")
        if (
            not isinstance(component, dict)
            or component.get("policy")
            != "one_4_connected_bilateral_background_component_crossing_nominal_owner_boundary"
        ):
            raise RuntimeError(
                "Accepted geometry mesh lacks one-component background selection"
            )
        depth_component_count = required_integer(
            component, "depth_safe_component_count", context=context
        )
        crossing_component_count = required_integer(
            component, "boundary_crossing_component_count", context=context
        )
        selected_label = required_integer(
            component, "selected_component_label", context=context
        )
        selected_count = required_integer(
            component, "selected_component_pixel_count", context=context
        )
        nonselected_count = required_integer(
            component, "nonselected_depth_safe_pixel_count", context=context
        )
        candidate_count = required_integer(
            audit, "candidate_depth_same_layer_pixel_count", context=context
        )
        before_rgb_count = required_integer(
            audit,
            "depth_same_layer_before_rgb_protection_pixel_count",
            context=context,
        )
        depth_same_layer_count = required_integer(
            audit, "depth_same_layer_pixel_count", context=context
        )
        same_layer_count = required_integer(
            audit, "same_layer_pixel_count", context=context
        )
        fit_support_count = required_integer(
            audit, "mesh_fit_support_pixel_count", context=context
        )
        flow_application_count = required_integer(
            audit, "rgb_flow_application_pixel_count", context=context
        )
        fit_excluded_count = required_integer(
            audit,
            "rgb_flow_fit_excluded_same_layer_pixel_count",
            context=context,
        )
        mesh_candidate_count = required_integer(
            audit, "mesh_candidate_pixel_count", context=context
        )
        mesh_active_count = required_integer(
            audit, "mesh_active_pixel_count", context=context
        )
        if (
            depth_component_count <= 0
            or crossing_component_count <= 0
            or crossing_component_count > depth_component_count
            or selected_label <= 0
            or selected_count <= 0
            or candidate_count != selected_count + nonselected_count
            or candidate_count > before_rgb_count
            or depth_same_layer_count != selected_count
            or same_layer_count <= 0
            or same_layer_count > depth_same_layer_count
            or flow_application_count <= 0
            or same_layer_count > flow_application_count
            or fit_support_count <= 0
            or fit_support_count > same_layer_count
            or fit_excluded_count != same_layer_count - fit_support_count
            or mesh_candidate_count <= 0
            or mesh_candidate_count > same_layer_count
            or mesh_active_count <= 0
            or mesh_active_count > mesh_candidate_count
        ):
            raise RuntimeError(
                "Accepted geometry mesh has invalid virtual background support"
            )
        if required_integer(
            audit,
            "protected_active_overlap_pixel_count",
            context=context,
        ) != 0:
            raise RuntimeError(
                "Accepted geometry mesh overlaps an owner-protected component"
            )
        if required_integer(
            audit,
            "active_non_same_layer_overlap_pixel_count",
            context=context,
        ) != 0:
            raise RuntimeError(
                "Accepted geometry mesh moves a non-same-layer component"
            )

    mesh_settings = settings.mesh_warp_config()

    triggered_count = 0
    accepted_count = 0
    hard_owner_fallback_count = 0
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise RuntimeError("Geometry assistance pair audit is malformed")
        if pair.get("pair_index") != index or pair.get("frame_ids") != frame_ids[index : index + 2]:
            raise RuntimeError("Geometry assistance pair order disagrees with render sources")
        if forbidden_keys & set(pair):
            raise RuntimeError("Geometry assistance sidecar attempted to publish dense data")
        audit = pair.get("audit")
        if not isinstance(audit, dict) or "reason" not in audit:
            raise RuntimeError("Geometry assistance pair audit lacks a decision reason")
        compact_audit = compact_scalar_audit(audit)
        if not isinstance(compact_audit, dict):  # Kept explicit for type safety.
            raise RuntimeError("Geometry assistance pair audit is malformed")
        triggered = pair.get("triggered")
        accepted = pair.get("accepted")
        if type(triggered) is not bool or type(accepted) is not bool:
            raise RuntimeError("Geometry assistance pair decision flags must be booleans")
        fallback = pair.get("fallback")
        corridor = pair.get("corridor_x")
        if triggered:
            if corridor is None:
                # A high-risk preview may correctly trigger while the two
                # immutable 20%-band strips have no 96--160 px common field.
                # That is a declared hard-owner fallback, never a reason to
                # expand the corridor or relax the strip budget.
                if (
                    accepted
                    or fallback != "hard_owner"
                    or audit.get("reason")
                    != "insufficient_calibrated_geometry_corridor"
                ):
                    raise RuntimeError(
                        "Geometry pair without a calibrated corridor must hard-own"
                    )
            elif (
                not isinstance(corridor, list)
                or len(corridor) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in corridor
                )
                or not 96 <= int(corridor[1]) - int(corridor[0]) <= 160
            ):
                raise RuntimeError("Triggered geometry pair has an invalid 96-160px corridor")
            triggered_count += 1
        elif corridor is not None:
            raise RuntimeError("Untriggered geometry pair must not retain a corridor")
        if accepted:
            accepted_count += 1
        if fallback == "hard_owner":
            hard_owner_fallback_count += 1
        if accepted and not triggered:
            raise RuntimeError("Accepted geometry pair was not triggered")
        if triggered and not accepted and (
            fallback != "hard_owner" or pair.get("warp_source_index") is not None
        ):
            raise RuntimeError("Rejected triggered geometry pair must use hard-owner fallback")
        if accepted:
            if audit.get("reason") != "accepted":
                raise RuntimeError("Accepted geometry pair lacks an accepted decision")
            if pair.get("warp_source_index") != index + 1 or pair.get("fallback") != "none":
                raise RuntimeError("Accepted geometry pair has an invalid local-warp owner")
            validate_accepted_trigger_audit(compact_audit)
            mesh = compact_audit.get("mesh")
            if not isinstance(mesh, dict):
                raise RuntimeError("Accepted geometry mesh lacks a scalar mesh audit")
            if mesh.get("accepted") is not True:
                raise RuntimeError("Accepted geometry pair disagrees with its mesh audit")
            if mesh.get("metric_unit") != "full_resolution_pixels":
                raise RuntimeError("Accepted geometry mesh lacks full-resolution metric units")
            straightness = mesh.get("maximum_straight_line_deviation_pixels")
            if (
                isinstance(straightness, bool)
                or not isinstance(straightness, (int, float))
                or not math.isfinite(float(straightness))
                or float(straightness) < 0.0
            ):
                raise RuntimeError("Accepted geometry mesh lacks a finite straight-line audit")
            if float(straightness) > float(
                settings.maximum_straight_line_deviation_pixels
            ):
                raise RuntimeError("Accepted geometry mesh exceeds straight-line gate")
            validate_accepted_depth_layer_audit(compact_audit)
            transparency = compact_audit.get("rgb_transparency_protection")
            if (
                not isinstance(transparency, dict)
                or transparency.get("policy")
                != "rgb_occlusion_or_any_strong_rgb_structure_dilated_guard"
            ):
                raise RuntimeError(
                    "Accepted geometry mesh lacks RGB transparency/reflection protection"
                )
            if required_integer(
                transparency,
                "guard_radius_pixels",
                context="Accepted geometry RGB transparency protection",
            ) != int(settings.edge_guard_radius_pixels):
                raise RuntimeError(
                    "Accepted geometry mesh has an inconsistent RGB transparency guard"
                )
            transparency_counts = {
                key: required_integer(
                    transparency,
                    key,
                    context="Accepted geometry RGB transparency protection",
                )
                for key in (
                    "preview_occluded_pixel_count",
                    "preview_strong_rgb_structure_pixel_count",
                    "preview_uncertain_or_rejected_strong_edge_pixel_count",
                    "preview_unsafe_pixel_count",
                    "full_resolution_unguarded_pixel_count",
                    "full_resolution_protected_pixel_count",
                    "full_resolution_tile_pixel_count",
                )
            }
            if (
                any(value < 0 for value in transparency_counts.values())
                or transparency_counts["preview_unsafe_pixel_count"]
                < transparency_counts["preview_occluded_pixel_count"]
                or transparency_counts["preview_unsafe_pixel_count"]
                < transparency_counts[
                    "preview_strong_rgb_structure_pixel_count"
                ]
                or transparency_counts[
                    "preview_strong_rgb_structure_pixel_count"
                ]
                < transparency_counts[
                    "preview_uncertain_or_rejected_strong_edge_pixel_count"
                ]
                or transparency_counts["full_resolution_protected_pixel_count"]
                < transparency_counts["full_resolution_unguarded_pixel_count"]
                or transparency_counts["full_resolution_protected_pixel_count"]
                > transparency_counts["full_resolution_tile_pixel_count"]
                or transparency_counts["full_resolution_tile_pixel_count"] <= 0
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has an invalid RGB transparency protection audit"
                )
            for key in (
                "depth_protected_pixel_count",
                "rgb_transparent_or_reflection_protected_pixel_count",
                "protected_pixel_count",
            ):
                if required_integer(
                    compact_audit, key, context="Accepted geometry protection audit"
                ) < 0:
                    raise RuntimeError(
                        "Accepted geometry mesh has a negative protection count"
                    )
            if required_integer(
                compact_audit,
                "protected_pixel_count",
                context="Accepted geometry protection audit",
            ) < max(
                required_integer(
                    compact_audit,
                    "depth_protected_pixel_count",
                    context="Accepted geometry protection audit",
                ),
                required_integer(
                    compact_audit,
                    "rgb_transparent_or_reflection_protected_pixel_count",
                    context="Accepted geometry protection audit",
                ),
            ):
                raise RuntimeError(
                    "Accepted geometry mesh protection union disagrees with its inputs"
                )
            final_owner = compact_audit.get("final_owner")
            if (
                not isinstance(final_owner, dict)
                or final_owner.get("policy")
                != "final_nominal_owner_boundary_rgb_risk_and_hard_cut_closure"
            ):
                raise RuntimeError(
                    "Accepted geometry mesh lacks final RGB owner-closure audit"
                )
            final_owner_counts = {
                key: required_integer(
                    final_owner,
                    key,
                    context="Accepted geometry final owner audit",
                )
                for key in (
                    "nominal_boundary_x",
                    "nominal_boundary_half_width_pixels",
                    "final_nominal_boundary_risk_pixel_count",
                    "final_nominal_boundary_risk_row_count",
                    "final_boundary_common_row_count",
                    "final_common_row_count",
                    "final_hard_cut_row_count",
                )
            }
            final_full_hard_cut = final_owner.get("final_full_height_hard_cut")
            final_graphcut_used = final_owner.get("final_graphcut_used")
            final_pair_level = final_owner.get("final_pair_level_hard_owner")
            if (
                type(final_full_hard_cut) is not bool
                or type(final_graphcut_used) is not bool
                or type(final_pair_level) is not bool
                or final_owner_counts["nominal_boundary_half_width_pixels"] != 2
                or final_owner_counts["final_nominal_boundary_risk_row_count"]
                > final_owner_counts["final_boundary_common_row_count"]
                or final_owner_counts["final_boundary_common_row_count"]
                > final_owner_counts["final_common_row_count"]
                or (
                    not final_pair_level
                    and final_owner_counts["final_hard_cut_row_count"]
                    > final_owner_counts["final_common_row_count"]
                )
                or (
                    final_full_hard_cut
                    and not final_pair_level
                    and (
                        final_owner_counts["final_common_row_count"] <= 0
                        or final_owner_counts["final_hard_cut_row_count"]
                        < final_owner_counts["final_common_row_count"]
                    )
                )
                or (final_pair_level and not final_full_hard_cut)
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has an invalid final RGB owner-closure audit"
                )
            active_cells = mesh.get("active_cell_count")
            largest_component = mesh.get("largest_connected_active_cell_count")
            if (
                not isinstance(active_cells, int)
                or not isinstance(largest_component, int)
                or active_cells < int(settings.minimum_active_mesh_cells)
                or largest_component < int(settings.minimum_active_mesh_cells)
            ):
                raise RuntimeError(
                    "Accepted geometry mesh lacks four-cell connected support"
                )
            if mesh.get("straight_line_audit_policy") != (
                "raw_same_layer_active_centrelines_internal_grid_edges_and_cell_diagonals"
            ):
                raise RuntimeError("Accepted geometry mesh lacks straight-line audit policy")
            correspondence_count = required_integer(
                mesh, "correspondence_count", context="Accepted geometry mesh"
            )
            if correspondence_count < int(settings.minimum_mutual_correspondences):
                raise RuntimeError("Accepted geometry mesh lacks mutual correspondence support")
            training_count = required_integer(
                mesh, "training_count", context="Accepted geometry mesh"
            )
            held_out_count = required_integer(
                mesh, "held_out_count", context="Accepted geometry mesh"
            )
            expected_held_out = max(
                1,
                int(round(correspondence_count * float(mesh_settings.held_out_fraction))),
            )
            if (
                training_count <= 0
                or held_out_count != expected_held_out
                or training_count + held_out_count != correspondence_count
            ):
                raise RuntimeError("Accepted geometry mesh lacks an independent held-out partition")
            if required_integer(
                mesh, "free_node_count", context="Accepted geometry mesh"
            ) <= 0:
                raise RuntimeError("Accepted geometry mesh lacks free mesh nodes")
            held_before = required_finite_scalar(
                mesh,
                "held_out_error_p95_before_pixels",
                context="Accepted geometry mesh",
            )
            held_after = required_finite_scalar(
                mesh,
                "held_out_error_p95_after_pixels",
                context="Accepted geometry mesh",
            )
            held_max = required_finite_scalar(
                mesh,
                "held_out_error_max_after_pixels",
                context="Accepted geometry mesh",
            )
            if (
                held_before < 0.0
                or held_after < 0.0
                or held_max < 0.0
                or held_after > float(settings.maximum_held_out_error_pixels)
                or held_max > float(settings.maximum_held_out_maximum_error_pixels)
            ):
                raise RuntimeError("Accepted geometry mesh exceeds held-out error gate")
            improvement = held_before - held_after
            improvement_ratio = improvement / held_before if held_before > 1e-9 else 0.0
            if (
                improvement < float(settings.minimum_held_out_improvement_pixels)
                or improvement_ratio < float(settings.minimum_held_out_improvement_ratio)
            ):
                raise RuntimeError("Accepted geometry mesh lacks held-out improvement")
            displacement = required_finite_scalar(
                mesh, "maximum_displacement_pixels", context="Accepted geometry mesh"
            )
            if not 0.0 <= displacement <= float(settings.maximum_local_displacement_pixels):
                raise RuntimeError("Accepted geometry mesh exceeds displacement gate")
            minimum_det = required_finite_scalar(
                mesh, "minimum_jacobian_determinant", context="Accepted geometry mesh"
            )
            maximum_det = required_finite_scalar(
                mesh, "maximum_jacobian_determinant", context="Accepted geometry mesh"
            )
            maximum_condition = required_finite_scalar(
                mesh, "maximum_jacobian_condition", context="Accepted geometry mesh"
            )
            if not (
                float(mesh_settings.minimum_jacobian_determinant)
                <= minimum_det
                <= maximum_det
                <= float(mesh_settings.maximum_jacobian_determinant)
                and 1.0 <= maximum_condition <= float(mesh_settings.maximum_jacobian_condition)
            ):
                raise RuntimeError("Accepted geometry mesh violates Jacobian gate")
            boundary_identity = required_finite_scalar(
                mesh,
                "boundary_identity_maximum_error_pixels",
                context="Accepted geometry mesh",
            )
            if not 0.0 <= boundary_identity <= 1e-9:
                raise RuntimeError("Accepted geometry mesh violates boundary-identity gate")
            flow_fit_support = compact_audit.get("rgb_flow_fit_support")
            if not isinstance(flow_fit_support, dict) or flow_fit_support.get("policy") != (
                "training_only_accepted_bidirectional_rgb_flow_and_epipolar_support"
            ):
                raise RuntimeError("Accepted geometry mesh lacks flow-consistent fit support")
            fit_full_resolution_count = required_integer(
                flow_fit_support,
                "full_resolution_flow_consistent_pixel_count",
                context="Accepted geometry flow-fit mask",
            )
            application_full_resolution_count = required_integer(
                flow_fit_support,
                "full_resolution_application_flow_consistent_pixel_count",
                context="Accepted geometry flow-application mask",
            )
            training_preview_count = required_integer(
                flow_fit_support,
                "preview_training_flow_consistent_pixel_count",
                context="Accepted geometry flow-fit mask",
            )
            application_preview_count = required_integer(
                flow_fit_support,
                "preview_application_flow_consistent_pixel_count",
                context="Accepted geometry flow-application mask",
            )
            held_out_preview_count = required_integer(
                flow_fit_support,
                "preview_held_out_pixel_count",
                context="Accepted geometry flow-fit mask",
            )
            flow_tile_count = required_integer(
                flow_fit_support,
                "full_resolution_tile_pixel_count",
                context="Accepted geometry flow-fit mask",
            )
            audited_flow_application_count = required_integer(
                compact_audit,
                "rgb_flow_application_pixel_count",
                context="Accepted geometry flow-application audit",
            )
            audited_fit_support_count = required_integer(
                compact_audit,
                "mesh_fit_support_pixel_count",
                context="Accepted geometry flow-fit audit",
            )
            if (
                fit_full_resolution_count <= 0
                or application_full_resolution_count <= 0
                or training_preview_count <= 0
                or application_preview_count < training_preview_count
                or held_out_preview_count <= 0
                or flow_tile_count <= 0
                or fit_full_resolution_count > application_full_resolution_count
                or application_full_resolution_count > flow_tile_count
                or audited_flow_application_count
                != application_full_resolution_count
                or audited_fit_support_count > fit_full_resolution_count
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has inconsistent independent flow-fit support"
                )
            fit_preview_scale = required_finite_scalar(
                flow_fit_support, "preview_scale", context="Accepted geometry flow-fit mask"
            )
            fit_maximum_full_resolution_fb = required_finite_scalar(
                flow_fit_support,
                "maximum_full_resolution_fb_error_pixels",
                context="Accepted geometry flow-fit mask",
            )
            fit_maximum_preview_fb = required_finite_scalar(
                flow_fit_support,
                "maximum_preview_fb_error_pixels",
                context="Accepted geometry flow-fit mask",
            )
            if not (
                0.0 < fit_preview_scale <= 1.0
                and flow_fit_support.get("metric_unit")
                == "full_resolution_pixels"
                and math.isclose(
                    fit_maximum_full_resolution_fb,
                    float(settings.maximum_held_out_flow_fb_error_pixels),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    fit_maximum_preview_fb,
                    fit_maximum_full_resolution_fb * fit_preview_scale,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError("Accepted geometry mesh has inconsistent flow-fit policy")
            flow = compact_audit.get("rgb_flow_validation")
            if (
                not isinstance(flow, dict)
                or flow.get("accepted") is not True
                or flow.get("reason") != "accepted"
                or flow.get("metric_unit") != "full_resolution_pixels"
            ):
                raise RuntimeError("Accepted geometry mesh lacks held-out RGB-flow acceptance")
            flow_count = required_integer(
                flow,
                "held_out_observable_flow_pixel_count",
                context="Accepted geometry RGB-flow audit",
            )
            flow_p95 = required_finite_scalar(
                flow,
                "held_out_flow_fb_error_p95_pixels",
                context="Accepted geometry RGB-flow audit",
            )
            if (
                flow_count < int(settings.minimum_held_out_flow_validation_pixels)
                or flow_p95 < 0.0
                or flow_p95 > float(settings.maximum_held_out_flow_fb_error_pixels)
            ):
                raise RuntimeError("Accepted geometry mesh violates held-out RGB-flow gate")
            flow_preview_scale = required_finite_scalar(
                flow, "preview_scale", context="Accepted geometry RGB-flow audit"
            )
            if not math.isclose(
                flow_preview_scale,
                float(settings.flow_validation_preview_scale),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError("Accepted geometry mesh has inconsistent flow-validation scale")
            flow_maximum = required_finite_scalar(
                flow,
                "maximum_held_out_flow_fb_error_pixels",
                context="Accepted geometry RGB-flow audit",
            )
            if not math.isclose(
                flow_maximum,
                float(settings.maximum_held_out_flow_fb_error_pixels),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has inconsistent RGB-flow threshold"
                )
            strong_edge = flow.get("held_out_strong_edge")
            if (
                not isinstance(strong_edge, dict)
                or strong_edge.get("accepted") is not True
                or strong_edge.get("reason") != "accepted"
                or strong_edge.get("policy")
                != "same_held_out_flow_epipolar_supported_strong_rgb_edges"
                or strong_edge.get("metric_unit") != "full_resolution_pixels"
            ):
                raise RuntimeError("Accepted geometry mesh lacks held-out strong-edge acceptance")
            strong_count = required_integer(
                strong_edge,
                "held_out_strong_edge_pixel_count",
                context="Accepted geometry strong-edge audit",
            )
            strong_after = required_finite_scalar(
                strong_edge,
                "held_out_strong_edge_p95_after_pixels",
                context="Accepted geometry strong-edge audit",
            )
            strong_max = required_finite_scalar(
                strong_edge,
                "held_out_strong_edge_maximum_after_pixels",
                context="Accepted geometry strong-edge audit",
            )
            strong_improvement = required_finite_scalar(
                strong_edge,
                "held_out_strong_edge_improvement_pixels",
                context="Accepted geometry strong-edge audit",
            )
            strong_ratio = required_finite_scalar(
                strong_edge,
                "held_out_strong_edge_improvement_ratio",
                context="Accepted geometry strong-edge audit",
            )
            strong_before = required_finite_scalar(
                strong_edge,
                "held_out_strong_edge_p95_before_pixels",
                context="Accepted geometry strong-edge audit",
            )
            recomputed_strong_improvement = strong_before - strong_after
            recomputed_strong_ratio = (
                recomputed_strong_improvement / strong_before
                if strong_before > 1e-9
                else 0.0
            )
            if (
                strong_count
                < int(settings.minimum_held_out_strong_edge_validation_pixels)
                or strong_before < 0.0
                or strong_after < 0.0
                or strong_max < 0.0
                or strong_after > float(settings.maximum_held_out_error_pixels)
                or strong_max > float(settings.maximum_held_out_maximum_error_pixels)
                or not math.isclose(
                    strong_improvement,
                    recomputed_strong_improvement,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    strong_ratio,
                    recomputed_strong_ratio,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or recomputed_strong_improvement
                < float(settings.minimum_held_out_improvement_pixels)
                or recomputed_strong_ratio
                < float(settings.minimum_held_out_improvement_ratio)
            ):
                raise RuntimeError("Accepted geometry mesh violates held-out strong-edge gate")
            strong_preview_scale = required_finite_scalar(
                strong_edge,
                "preview_scale",
                context="Accepted geometry strong-edge audit",
            )
            if not math.isclose(
                strong_preview_scale,
                flow_preview_scale,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has inconsistent strong-edge scale"
                )
            actual_line = flow.get("rgb_actual_line_straightness")
            if (
                not isinstance(actual_line, dict)
                or actual_line.get("accepted") is not True
                or actual_line.get("metric_unit") != "full_resolution_pixels"
                or actual_line.get("policy")
                != (
                    "baseline_second_rgb_dual_source_canny_hough_"
                    "raw_forward_inverse_chord_bend"
                )
            ):
                raise RuntimeError(
                    "Accepted geometry mesh lacks actual RGB line-straightness acceptance"
                )
            observed_line = actual_line.get("observed")
            if type(observed_line) is not bool:
                raise RuntimeError(
                    "Accepted geometry mesh lacks an explicit actual RGB line observation state"
                )
            actual_line_counts = {
                key: required_integer(
                    actual_line,
                    key,
                    context="Accepted geometry actual RGB line audit",
                )
                for key in (
                    "raw_hough_segment_count",
                    "deduplicated_hough_segment_count",
                    "eligible_hough_segment_count",
                    "tested_line_run_count",
                )
            }
            if (
                actual_line_counts["raw_hough_segment_count"]
                < actual_line_counts["deduplicated_hough_segment_count"]
                or actual_line_counts["raw_hough_segment_count"] < 0
                or actual_line_counts["deduplicated_hough_segment_count"] < 0
                or actual_line_counts["eligible_hough_segment_count"] < 0
                or actual_line_counts["tested_line_run_count"] < 0
                or actual_line_counts["deduplicated_hough_segment_count"]
                > int(settings.maximum_actual_rgb_line_segments)
                or actual_line_counts["eligible_hough_segment_count"]
                > actual_line_counts["deduplicated_hough_segment_count"]
            ):
                raise RuntimeError(
                    "Accepted geometry mesh has invalid actual RGB line support"
                )
            actual_line_length = required_finite_scalar(
                actual_line,
                "minimum_actual_rgb_line_length_pixels",
                context="Accepted geometry actual RGB line audit",
            )
            actual_line_support = required_finite_scalar(
                actual_line,
                "minimum_actual_rgb_line_support_fraction",
                context="Accepted geometry actual RGB line audit",
            )
            if (
                not math.isclose(
                    actual_line_length,
                    float(settings.minimum_actual_rgb_line_length_pixels),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    actual_line_support,
                    float(settings.minimum_actual_rgb_line_support_fraction),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or required_integer(
                    actual_line,
                    "maximum_actual_rgb_line_segments",
                    context="Accepted geometry actual RGB line audit",
                )
                != int(settings.maximum_actual_rgb_line_segments)
                or required_integer(
                    actual_line,
                    "inverse_maximum_iterations",
                    context="Accepted geometry actual RGB line audit",
                )
                != int(settings.actual_rgb_line_inverse_maximum_iterations)
                or not math.isclose(
                    required_finite_scalar(
                        actual_line,
                        "inverse_maximum_residual_pixels",
                        context="Accepted geometry actual RGB line audit",
                    ),
                    float(settings.actual_rgb_line_inverse_maximum_residual_pixels),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    required_finite_scalar(
                        actual_line,
                        "maximum_straight_line_deviation_pixels",
                        context="Accepted geometry actual RGB line audit",
                    ),
                    float(settings.maximum_straight_line_deviation_pixels),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise RuntimeError(
                    "Accepted geometry mesh violates actual RGB line-straightness gate"
                )
            if not observed_line:
                if (
                    actual_line.get("reason")
                    != "not_observed_no_solver_valid_line"
                    or actual_line_counts["tested_line_run_count"] != 0
                    or any(
                        actual_line.get(key) is not None
                        for key in (
                            "maximum_line_bend_pixels",
                            "p95_line_bend_pixels",
                            "maximum_inverse_residual_pixels",
                            "p95_inverse_residual_pixels",
                        )
                    )
                ):
                    raise RuntimeError(
                        "Accepted geometry mesh has an invalid unobserved RGB line audit"
                    )
            else:
                if (
                    actual_line.get("reason") != "accepted"
                    or actual_line_counts["eligible_hough_segment_count"] <= 0
                    or actual_line_counts["tested_line_run_count"] <= 0
                ):
                    raise RuntimeError(
                        "Accepted geometry mesh has invalid observed RGB line support"
                    )
                actual_line_max_bend = required_finite_scalar(
                    actual_line,
                    "maximum_line_bend_pixels",
                    context="Accepted geometry actual RGB line audit",
                )
                actual_line_p95_bend = required_finite_scalar(
                    actual_line,
                    "p95_line_bend_pixels",
                    context="Accepted geometry actual RGB line audit",
                )
                actual_line_max_residual = required_finite_scalar(
                    actual_line,
                    "maximum_inverse_residual_pixels",
                    context="Accepted geometry actual RGB line audit",
                )
                actual_line_p95_residual = required_finite_scalar(
                    actual_line,
                    "p95_inverse_residual_pixels",
                    context="Accepted geometry actual RGB line audit",
                )
                if (
                    min(
                        actual_line_max_bend,
                        actual_line_p95_bend,
                        actual_line_max_residual,
                        actual_line_p95_residual,
                    )
                    < 0.0
                    or actual_line_max_bend
                    > float(settings.maximum_straight_line_deviation_pixels)
                    or actual_line_p95_bend > actual_line_max_bend
                    or actual_line_max_residual
                    > float(settings.actual_rgb_line_inverse_maximum_residual_pixels)
                    or actual_line_p95_residual > actual_line_max_residual
                ):
                    raise RuntimeError(
                        "Accepted geometry mesh violates actual RGB line-straightness gate"
                    )
        compact_pairs.append(
            {
                "pair_index": index,
                "frame_ids": list(frame_ids[index : index + 2]),
                "triggered": triggered,
                "corridor_x": pair.get("corridor_x"),
                "warp_source_index": pair.get("warp_source_index"),
                "accepted": accepted,
                "fallback": pair.get("fallback"),
                "audit": compact_audit,
            }
        )
    if required_integer(
        geometry, "triggered_pair_count", context="Geometry assistance aggregate"
    ) != triggered_count:
        raise RuntimeError("Geometry assistance triggered aggregate disagrees with pair audits")
    if required_integer(
        geometry, "accepted_pair_count", context="Geometry assistance aggregate"
    ) != accepted_count:
        raise RuntimeError("Geometry assistance accepted aggregate disagrees with pair audits")
    if required_integer(
        geometry,
        "hard_owner_fallback_pair_count",
        context="Geometry assistance aggregate",
    ) != hard_owner_fallback_count:
        raise RuntimeError("Geometry assistance fallback aggregate disagrees with pair audits")
    if geometry.get("depth_used_for_local_geometry") is not bool(triggered_count):
        raise RuntimeError("Geometry assistance depth-use flag disagrees with pair audits")
    return {
        "backend": "rgbd_bidirectional_visibility_local_inverse_mesh",
        "configuration": settings.as_dict(),
        "scope": "adjacent_seam_corridors_only",
        "depth_used_for_output_pixels": False,
        "depth_used_for_local_geometry": bool(
            geometry.get("depth_used_for_local_geometry", False)
        ),
        "triggered_pair_count": int(geometry.get("triggered_pair_count", 0)),
        "accepted_pair_count": int(geometry.get("accepted_pair_count", 0)),
        "hard_owner_fallback_pair_count": int(
            geometry.get("hard_owner_fallback_pair_count", 0)
        ),
        "pairs": compact_pairs,
    }


def _read_bgr(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"OpenCV could not decode color image: {path}")
    return image


def _write_bgr(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 95] if suffix in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(suffix, image, parameters)
    if not success:
        raise OSError(f"OpenCV could not encode panorama: {path}")
    path.write_bytes(encoded.tobytes())
    return path


def _write_mask(path: Path, mask: np.ndarray) -> Path:
    """Write an explicit lossless foreground ownership mask for inspection."""

    binary = np.where(np.asarray(mask, dtype=bool), 255, 0).astype(np.uint8)
    return _write_bgr(path, binary)


def _write_dense_audit_image(path: Path, image: np.ndarray) -> Path:
    """Write a lossless dense-fusion audit raster with a stable ID encoding."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("Dense audit image must be a single-channel raster")
    if array.dtype == bool:
        return _write_mask(path, array)
    if np.issubdtype(array.dtype, np.signedinteger):
        # -1 means no source owner.  PNG zero therefore remains an explicit
        # empty sentinel and a real frame ID N is encoded as N+1.
        if np.any(array < -1):
            raise ValueError("Dense audit source IDs cannot be below -1")
        array = np.where(array >= 0, array + 1, 0).astype(np.uint16)
    elif array.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise ValueError(f"Unsupported dense audit image dtype: {array.dtype}")
    return _write_bgr(path, array)


def _write_bytes(path: Path, data: bytes) -> Path:
    """Write a binary deliverable using the caller's pending-file protocol."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _mesh_viewer_html(mesh_filename: str) -> str:
    """Build a self-contained entry page for a locally served GLB mesh."""

    mesh_url = html.escape(mesh_filename, quote=True)
    return f"""<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>TSDF 三维网格</title>
  <script type=\"module\" src=\"https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js\"></script>
  <style>
    html, body, model-viewer {{ width: 100%; height: 100%; margin: 0; background: #16181d; }}
    model-viewer {{ --poster-color: #16181d; }}
  </style>
</head>
<body>
  <model-viewer src=\"{mesh_url}\" alt=\"TSDF RGB-D mesh\" camera-controls
      auto-rotate shadow-intensity=\"0.7\" exposure=\"1\"></model-viewer>
</body>
</html>
"""


def _export_display_only_tsdf_mesh(
    frames: list[RGBDFrame],
    poses: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    config: dict[str, Any],
) -> tuple[bytes, dict[str, object]]:
    """Build the optional 3-D inspection asset after RGB rendering is complete.

    This delayed import is intentionally below the RGB panorama quality gate.
    The exported mesh has no return path into calibrated RGB pushbroom, seam,
    blending, crop, or delivery-quality decisions.
    """

    from .dense_fusion import export_tsdf_mesh

    return export_tsdf_mesh(
        frames,
        poses,
        _pinhole_intrinsics(intrinsics),
        config=config,
    )


def _intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, object]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "distortion": list(intrinsics.distortion),
    }


def _pinhole_intrinsics(intrinsics: CameraIntrinsics) -> "PinholeIntrinsics":
    # Kept solely for legacy isolated projection tests.  The formal sequence
    # import and calibrated RGB pushbroom route never import this depth module.
    from .rgbd_projection import PinholeIntrinsics

    return PinholeIntrinsics(
        width=intrinsics.width,
        height=intrinsics.height,
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
        distortion=intrinsics.distortion,
    )


def _analyse_session(
    frames: tuple[RGBDFrame, ...],
    analysis_width: int,
) -> tuple[list[FrameQuality], list[MotionEstimate], tuple[int, int, int]]:
    qualities: list[FrameQuality] = []
    motions: list[MotionEstimate] = []
    previous: np.ndarray | None = None
    shape: tuple[int, int, int] | None = None
    for frame in frames:
        analysis = resize_for_analysis(_read_bgr(frame.color_path), analysis_width)
        if shape is None:
            shape = analysis.shape
        elif analysis.shape != shape:
            raise ValueError(
                f"Frame {frame.frame_id} has inconsistent analysis shape {analysis.shape}"
            )
        qualities.append(analyze_frame_quality(analysis))
        if previous is not None:
            motions.append(estimate_translation(previous, analysis))
        previous = analysis
    if shape is None:
        raise ValueError("RGB-D session contains no analysable color frame")
    return qualities, motions, shape


def _select_pose_nodes(
    frames: tuple[RGBDFrame, ...],
    qualities: list[FrameQuality],
    motions: list[MotionEstimate],
    image_width: int,
    stitch_config: dict[str, Any],
) -> tuple[
    list[RGBDFrame],
    list[FrameQuality],
    dict[str, Any],
]:
    segment = select_primary_scan_segment(
        motions,
        image_width=image_width,
        maximum_fraction=float(
            stitch_config.get("layout_max_displacement_fraction", 0.28)
        ),
    )
    scan_frames = frames[segment.start_index : segment.end_index + 1]
    scan_qualities = qualities[segment.start_index : segment.end_index + 1]
    scan_motions = motions[segment.start_index : segment.end_index]
    selection = select_layout_from_motion_estimates(
        scan_motions,
        frame_count=len(scan_frames),
        image_width=image_width,
        target_fraction=float(
            stitch_config.get("layout_target_displacement_fraction", 0.18)
        ),
        maximum_fraction=float(
            stitch_config.get("layout_max_displacement_fraction", 0.28)
        ),
        max_selected=int(stitch_config.get("layout_max_frames", 160)),
    )
    dense_chain = bool(stitch_config.get("dense_rgbd_pose_chain", True))
    maximum_nodes = int(stitch_config.get("layout_max_frames", 160))
    if dense_chain:
        if len(scan_frames) > maximum_nodes:
            raise RuntimeError(
                "The primary RGB-D scan contains more frames than the "
                "configured dense pose-node safety limit"
            )
        # Every captured frame is a real pose node.  This gives close-range
        # projection and fusion a short-baseline trajectory rather than asking
        # a handful of widely spaced frames to represent the whole sweep.
        pose_frames = list(scan_frames)
        pose_qualities = list(scan_qualities)
    else:
        pose_frames = [scan_frames[index] for index in selection.indices]
        pose_qualities = [scan_qualities[index] for index in selection.indices]
    if len(pose_frames) < 2:
        raise RuntimeError("Adaptive RGB-D layout selected fewer than two pose nodes")
    metadata = selection.as_dict()
    metadata.update(
        {
            "mode": "adaptive_rgbd_pose_nodes",
            "pose_node_strategy": (
                "dense_consecutive_rgbd_chain" if dense_chain else "sparse_layout"
            ),
            "selected_pose_frame_count": len(pose_frames),
            "frame_ids": [frame.frame_id for frame in pose_frames],
            "segment": {
                **segment.as_dict(),
                "start_frame_id": scan_frames[0].frame_id,
                "end_frame_id": scan_frames[-1].frame_id,
            },
            "motion": [motion.as_dict() for motion in scan_motions],
        }
    )
    return pose_frames, pose_qualities, metadata


def _estimate_pose_edges(
    pose_frames: list[RGBDFrame],
    intrinsics: CameraIntrinsics,
    odometry_config: RGBDOdometryConfig,
    *,
    backend: object | None,
    nonadjacent_gap: int,
    scan_frames: tuple[RGBDFrame, ...] | None = None,
    short_baseline_audit: list[dict[str, object]] | None = None,
    short_baseline_initialization: bool = True,
) -> tuple[list[Any], list[dict[str, object]]]:
    """Measure selected pose edges, seeding wide edges from real short RGB-D edges.

    A sparse rendering layout may put 10--20 captured frames between two pose
    nodes.  At close range, solving that wide pair from identity can converge
    to a visually plausible but wrong local minimum.  We therefore compose
    only consecutive, measured RGB-D edges to initialise the same direct
    measurement.  The composed motion is never inserted as a synthetic graph
    edge and no pose is interpolated.
    """

    frames_by_id = (
        {frame.frame_id: (index, frame) for index, frame in enumerate(scan_frames)}
        if scan_frames is not None
        else {}
    )
    supports_seed = short_baseline_initialization and (backend is None or bool(
        getattr(backend, "supports_initial_source_to_reference", False)
    ))

    def short_baseline_seed(
        reference: RGBDFrame, source: RGBDFrame
    ) -> np.ndarray | None:
        if not supports_seed:
            return None
        reference_row = frames_by_id.get(reference.frame_id)
        source_row = frames_by_id.get(source.frame_id)
        if reference_row is None or source_row is None:
            return None
        start, _ = reference_row
        stop, _ = source_row
        if stop <= start + 1:
            return None
        composed = np.eye(4, dtype=np.float64)
        short_edges = 0
        try:
            for frame_index in range(start + 1, stop + 1):
                short_reference = scan_frames[frame_index - 1]
                short_source = scan_frames[frame_index]
                short_edge = estimate_pair_rgbd_odometry(
                    short_reference,
                    short_source,
                    intrinsics,
                    config=odometry_config,
                    backend=backend,
                )
                if not short_edge.reliable:
                    raise RuntimeError(
                        "short edge "
                        f"{short_reference.frame_id}<->{short_source.frame_id} "
                        "is unreliable: "
                        + "; ".join(short_edge.failure_reasons)
                    )
                composed = composed @ short_edge.source_to_reference
                short_edges += 1
        except Exception as exc:
            if short_baseline_audit is not None:
                short_baseline_audit.append(
                    {
                        "reference_node_id": reference.frame_id,
                        "source_node_id": source.frame_id,
                        "used": False,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
            return None
        if short_baseline_audit is not None:
            short_baseline_audit.append(
                {
                    "reference_node_id": reference.frame_id,
                    "source_node_id": source.frame_id,
                    "used": True,
                    "short_edge_count": short_edges,
                    "initial_source_to_reference": composed.tolist(),
                }
            )
        return composed

    edges: list[Any] = []
    optional_failures: list[dict[str, object]] = []
    pair_total = len(pose_frames) - 1
    for index in range(1, len(pose_frames)):
        reference = pose_frames[index - 1]
        source = pose_frames[index]
        started = time.perf_counter()
        initial_source_to_reference = short_baseline_seed(reference, source)
        edge = estimate_pair_rgbd_odometry(
            reference,
            source,
            intrinsics,
            config=odometry_config,
            backend=backend,
            initial_source_to_reference=initial_source_to_reference,
        )
        edges.append(edge)
        print(
            f"[{index}/{pair_total}] RGB-D {reference.frame_id}->{source.frame_id}: "
            f"fitness={edge.fitness:.3f}, rmse={edge.rmse_mm:.1f} mm, "
            f"{time.perf_counter() - started:.2f}s"
        )
    # Non-adjacent constraints are optional loop candidates.  A zero gap is
    # intentional for the default open side-scan chain.
    maximum_gap = max(0, int(nonadjacent_gap))
    for gap in range(2, maximum_gap + 1):
        for source_index in range(gap, len(pose_frames)):
            reference = pose_frames[source_index - gap]
            source = pose_frames[source_index]
            try:
                edge = estimate_pair_rgbd_odometry(
                    reference,
                    source,
                    intrinsics,
                    config=odometry_config,
                    backend=backend,
                    uncertain=True,
                )
            except Exception as exc:
                optional_failures.append(
                    {
                        "reference_node_id": reference.frame_id,
                        "source_node_id": source.frame_id,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                continue
            edges.append(edge)
    return edges, optional_failures


def _working_projection_frames(
    frames: list[RGBDFrame],
    poses: list[np.ndarray],
    intrinsics: CameraIntrinsics,
    working_width: int,
) -> tuple[list["RGBDProjectionFrame"], "PinholeIntrinsics"]:
    # Kept solely for the independent central-strip diagnostic callback.  Its
    # delayed import prevents the formal RGB renderer from loading or using a
    # depth projection path.
    from .rgbd_projection import PinholeIntrinsics, RGBDProjectionFrame

    target_width = min(intrinsics.width, int(working_width))
    if target_width < 64:
        raise ValueError("RGB-D footprint working width must be at least 64")
    scale = target_width / float(intrinsics.width)
    target_height = max(1, int(round(intrinsics.height * scale)))
    scaled = PinholeIntrinsics(
        width=target_width,
        height=target_height,
        fx=intrinsics.fx * scale,
        fy=intrinsics.fy * (target_height / float(intrinsics.height)),
        cx=intrinsics.cx * scale,
        cy=intrinsics.cy * (target_height / float(intrinsics.height)),
        distortion=(),
    )
    placeholder = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    result: list[RGBDProjectionFrame] = []
    for frame, pose in zip(frames, poses, strict=True):
        depth = read_aligned_depth_mm(frame)
        if depth.shape != (target_height, target_width):
            depth = cv2.resize(
                depth,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        result.append(
            RGBDProjectionFrame(
                frame_id=frame.frame_id,
                rgb=placeholder,
                depth_mm=np.ascontiguousarray(depth),
                camera_to_world=pose,
            )
        )
    return result, scaled


def _full_projection_frames(
    frames: list[RGBDFrame], poses: list[np.ndarray]
) -> list["RGBDProjectionFrame"]:
    # Legacy unit-test helper; no formal rendering call reaches this function.
    from .rgbd_projection import RGBDProjectionFrame

    return [
        RGBDProjectionFrame(
            frame_id=frame.frame_id,
            rgb=_read_bgr(frame.color_path),
            depth_mm=read_aligned_depth_mm(frame),
            camera_to_world=pose,
        )
        for frame, pose in zip(frames, poses, strict=True)
    ]


def _validate_backend_config(stitch_config: dict[str, Any]) -> None:
    pose_backend = str(stitch_config.get("pose_backend", "hybrid_orbslam3_rgbd"))
    if pose_backend not in {"open3d_rgbd", "hybrid_orbslam3_rgbd"}:
        raise ValueError(
            "pose_backend must be open3d_rgbd or hybrid_orbslam3_rgbd"
        )
    if (
        str(stitch_config.get("sequence_blend_mode", "calibrated_rgb_pushbroom"))
        != "calibrated_rgb_pushbroom"
    ):
        raise ValueError(
            "calibrated_rgb_pushbroom is the only formal sequence render mode"
        )
    if any(
        name in stitch_config
        for name in ("dense_fusion_backend", "dense_tsdf", "rgbd_projection")
    ):
        raise ValueError(
            "Formal calibrated RGB pushbroom rendering rejects TSDF and RGB-D "
            "projection configuration"
        )
    visualization = stitch_config.get("tsdf_visualization", {})
    if not isinstance(visualization, dict):
        raise ValueError("tsdf_visualization must be a mapping")
    pushbroom = dict(stitch_config.get("calibrated_rgb_pushbroom", {}))
    if str(pushbroom.get("mode", "calibrated_rgb_pushbroom")) != (
        "calibrated_rgb_pushbroom"
    ):
        raise ValueError(
            "Formal renderer mode must remain calibrated_rgb_pushbroom"
        )
    seam = dict(stitch_config.get("scan_seam", {}))
    if str(seam.get("backend", "rgb_monotonic_hard_owner_graphcut")) != (
        "rgb_monotonic_hard_owner_graphcut"
    ):
        raise ValueError(
            "Formal scan seam backend must be rgb_monotonic_hard_owner_graphcut"
        )


def _validate_safety_envelope(
    stitch_config: dict[str, Any], *, diagnostic_force: bool
) -> None:
    """Reject configuration values that relax non-bypassable safety bounds."""

    canvas_limit = float(
        stitch_config.get("max_canvas_megapixels", _HARD_MAX_CANVAS_MEGAPIXELS)
    )
    if not np.isfinite(canvas_limit) or not 0.0 < canvas_limit <= _HARD_MAX_CANVAS_MEGAPIXELS:
        raise ValueError("max_canvas_megapixels cannot exceed the 200 MP hard limit")
    layout_limit = int(
        stitch_config.get("layout_max_frames", _HARD_MAX_LAYOUT_FRAMES)
    )
    if not 2 <= layout_limit <= _HARD_MAX_LAYOUT_FRAMES:
        raise ValueError("layout_max_frames must remain within the 2-160 hard budget")

    pushbroom = dict(stitch_config.get("calibrated_rgb_pushbroom", {}))
    pushbroom_canvas_limit = float(
        pushbroom.get("max_canvas_megapixels", canvas_limit)
    )
    if (
        not np.isfinite(pushbroom_canvas_limit)
        or not 0.0 < pushbroom_canvas_limit <= _HARD_MAX_CANVAS_MEGAPIXELS
    ):
        raise ValueError(
            "calibrated_rgb_pushbroom.max_canvas_megapixels cannot exceed 200 MP"
        )
    pushbroom_aggregate_limit = float(
        pushbroom.get("max_aggregate_megapixels", canvas_limit)
    )
    if (
        not np.isfinite(pushbroom_aggregate_limit)
        or not 0.0 < pushbroom_aggregate_limit <= _HARD_MAX_CANVAS_MEGAPIXELS
    ):
        raise ValueError(
            "calibrated_rgb_pushbroom.max_aggregate_megapixels cannot exceed "
            "200 MP"
        )
    pushbroom_pose_limit = int(
        pushbroom.get("max_pose_count", _HARD_MAX_LAYOUT_FRAMES)
    )
    if not 2 <= pushbroom_pose_limit <= _HARD_MAX_LAYOUT_FRAMES:
        raise ValueError(
            "calibrated_rgb_pushbroom.max_pose_count must remain within the "
            "2-160 hard budget"
        )
    resident_limit = int(pushbroom.get("max_resident_frames", 5))
    if not 2 <= resident_limit <= 5:
        raise ValueError(
            "calibrated_rgb_pushbroom.max_resident_frames must remain within "
            "the 2-5 streaming budget"
        )
    try:
        residual_alignment = ResidualAlignmentConfig.from_mapping(
            pushbroom.get("residual_alignment")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid calibrated_rgb_pushbroom.residual_alignment") from exc
    try:
        geometry_assisted_seam = GeometryAssistedSeamConfig.from_mapping(
            pushbroom.get("geometry_assisted_seam")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid calibrated_rgb_pushbroom.geometry_assisted_seam") from exc

    # This is structural even for --diagnostic-force: a global RGB residual
    # would create a second pose model and violate the SE(3)-only contract.
    if residual_alignment.background_model != "identity":
        raise ValueError(
            "Calibrated RGB pushbroom requires identity residual alignment; "
            "verified RGB-D SE(3) is the sole global geometry"
        )

    if diagnostic_force:
        return

    visualization = dict(stitch_config.get("tsdf_visualization", {}))
    if visualization.get("enabled", True) is not True:
        raise ValueError(
            "Formal delivery requires the display-only tsdf_visualization export"
        )

    # These remain structural/model limits even for a later diagnostic A/B
    # run.  Formal delivery additionally requires a real held-out partition
    # and cross-pair owner tracking rather than allowing the model to certify
    # itself on the same evidence it used to select a residual.
    if residual_alignment.held_out_fraction < 0.20:
        raise ValueError(
            "Formal residual alignment must retain at least a 20% held-out partition"
        )
    if not residual_alignment.owner_track_consistency:
        raise ValueError(
            "Formal residual alignment requires cross-pair owner track consistency"
        )
    if not geometry_assisted_seam.enabled:
        raise ValueError("Formal delivery requires geometry_assisted_seam.enabled")
    if geometry_assisted_seam.trigger_edge_offset_p95_pixels > 1.0:
        raise ValueError(
            "Formal geometry-assist trigger cannot exceed 1.0 pixels"
        )
    if geometry_assisted_seam.mutual_reprojection_tolerance_pixels > 0.40:
        raise ValueError(
            "Formal geometry-assist mutual reprojection tolerance cannot exceed 0.40 pixels"
        )
    if not np.isclose(
        geometry_assisted_seam.absolute_depth_tolerance_mm,
        20.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Formal geometry-assist absolute depth tolerance must equal 20 mm"
        )
    if not np.isclose(
        geometry_assisted_seam.relative_depth_tolerance,
        0.02,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Formal geometry-assist relative depth tolerance must equal 2%"
        )
    if not np.isclose(
        geometry_assisted_seam.depth_noise_mm,
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Formal geometry-assist cannot use uncalibrated depth-noise tolerance"
        )
    if geometry_assisted_seam.maximum_held_out_flow_fb_error_pixels > 0.75:
        raise ValueError(
            "Formal geometry-assist held-out RGB flow FB error cannot exceed 0.75 pixels"
        )
    if geometry_assisted_seam.minimum_held_out_strong_edge_validation_pixels < 8:
        raise ValueError(
            "Formal geometry-assist requires at least eight held-out strong-edge samples"
        )
    if geometry_assisted_seam.minimum_trigger_boundary_observable_pixels < 32:
        raise ValueError(
            "Formal geometry-assist trigger requires at least 32 boundary-observable pixels"
        )
    if geometry_assisted_seam.minimum_active_mesh_cells < 4:
        raise ValueError(
            "Formal geometry-assist requires at least four active mesh cells"
        )
    if geometry_assisted_seam.minimum_held_out_improvement_ratio < 0.30:
        raise ValueError(
            "Formal geometry-assist requires at least 30% held-out improvement"
        )
    if geometry_assisted_seam.maximum_held_out_error_pixels > 0.75:
        raise ValueError(
            "Formal geometry-assist held-out P95 cannot exceed 0.75 pixels"
        )
    if geometry_assisted_seam.maximum_held_out_maximum_error_pixels > 2.0:
        raise ValueError(
            "Formal geometry-assist held-out maximum cannot exceed 2 pixels"
        )
    if geometry_assisted_seam.maximum_straight_line_deviation_pixels > 1.0:
        raise ValueError(
            "Formal geometry-assist straight-line deviation cannot exceed 1 pixel"
        )
    if geometry_assisted_seam.minimum_actual_rgb_line_length_pixels < 24.0:
        raise ValueError(
            "Formal geometry-assist actual RGB line support must be at least 24 pixels"
        )
    if geometry_assisted_seam.minimum_actual_rgb_line_support_fraction < 0.80:
        raise ValueError(
            "Formal geometry-assist actual RGB line support fraction must be at least 80%"
        )
    if geometry_assisted_seam.maximum_actual_rgb_line_segments > 32:
        raise ValueError(
            "Formal geometry-assist actual RGB Hough segment budget cannot exceed 32"
        )
    if geometry_assisted_seam.actual_rgb_line_inverse_maximum_iterations > 8:
        raise ValueError(
            "Formal geometry-assist actual RGB line inverse solve cannot exceed 8 iterations"
        )
    if geometry_assisted_seam.actual_rgb_line_inverse_maximum_residual_pixels > 0.05:
        raise ValueError(
            "Formal geometry-assist actual RGB line inverse residual cannot exceed 0.05 pixels"
        )

    if not bool(stitch_config.get("adaptive_layout", True)):
        raise ValueError("Formal delivery cannot disable adaptive_layout")
    if not bool(stitch_config.get("dense_rgbd_pose_chain", True)):
        raise ValueError(
            "Formal calibrated RGB pushbroom rendering requires every real "
            "RGB-D pose node"
        )
    if not bool(stitch_config.get("input_quality_gate", True)):
        raise ValueError("Formal delivery cannot disable input_quality_gate")
    exposure_limit = float(
        stitch_config.get("maximum_motion_exposure_us", 1200.0)
    )
    if not np.isfinite(exposure_limit) or not 0.0 < exposure_limit <= 1200.0:
        raise ValueError("Formal exposure rejection limit cannot exceed 1200 us")
    exposure_unit = float(stitch_config.get("color_exposure_unit_us", 100.0))
    if not np.isclose(exposure_unit, 100.0):
        raise ValueError("Formal color exposure metadata unit must remain 100 us")
    band_fraction = float(pushbroom.get("maximum_central_band_fraction", 0.20))
    if not np.isfinite(band_fraction) or not 0.0 < band_fraction <= 0.20:
        raise ValueError(
            "Formal central RGB strip fraction cannot exceed 0.20"
        )
    if pushbroom.get("endpoint_outer_half_fov", True) is not True:
        raise ValueError(
            "Formal calibrated RGB pushbroom requires endpoint_outer_half_fov"
        )
    seam_search_width = int(pushbroom.get("seam_search_width_pixels", 64))
    if not 32 <= seam_search_width <= 64:
        raise ValueError(
            "Formal calibrated RGB seam search width must remain within 32-64 "
            "pixels"
        )
    if int(pushbroom.get("minimum_valid_scale_pairs", 3)) < 3:
        raise ValueError(
            "Formal RGB motion scale requires at least three valid adjacent pairs"
        )
    if float(pushbroom.get("scale_minimum_response", 0.10)) < 0.10:
        raise ValueError(
            "Formal RGB motion scale response threshold cannot be relaxed below 0.10"
        )
    if float(pushbroom.get("scale_max_relative_mad", 0.35)) > 0.35:
        raise ValueError(
            "Formal RGB motion scale relative MAD cannot exceed 0.35"
        )
    low_gradient_quantile = float(
        pushbroom.get("scale_low_gradient_quantile", 0.45)
    )
    if (
        not np.isfinite(low_gradient_quantile)
        or not 0.0 < low_gradient_quantile <= 0.45
    ):
        raise ValueError(
            "Formal safe-wall low-gradient quantile cannot exceed 0.45"
        )
    scale_fraction = float(pushbroom.get("scale_central_fraction", 0.20))
    if not np.isfinite(scale_fraction) or not 0.0 < scale_fraction <= 0.20:
        raise ValueError(
            "Formal RGB motion scale central fraction cannot exceed 0.20"
        )

    seam = dict(stitch_config.get("scan_seam", {}))
    if not bool(seam.get("quality_gate", True)):
        raise ValueError("Formal delivery cannot disable the scan-seam quality gate")
    levels = int(seam.get("multiband_levels", 3))
    if not 1 <= levels <= 3:
        raise ValueError("Formal MultiBand level count must remain within 1-3")
    if str(seam.get("exposure_mode", "safe_wall_global_linear_rgb")) != (
        "safe_wall_global_linear_rgb"
    ):
        raise ValueError(
            "Formal exposure compensation mode must be "
            "safe_wall_global_linear_rgb"
        )

    odometry = RGBDOdometryConfig.from_mapping(
        stitch_config.get("rgbd_odometry")
    )
    baseline_odometry = RGBDOdometryConfig()
    minimum_odometry = (
        ("working_width", odometry.working_width, baseline_odometry.working_width),
        (
            "minimum_depth_mm",
            odometry.minimum_depth_mm,
            baseline_odometry.minimum_depth_mm,
        ),
        (
            "minimum_valid_depth_ratio",
            odometry.minimum_valid_depth_ratio,
            baseline_odometry.minimum_valid_depth_ratio,
        ),
        ("minimum_fitness", odometry.minimum_fitness, baseline_odometry.minimum_fitness),
    )
    maximum_odometry = (
        ("maximum_depth_mm", odometry.maximum_depth_mm, baseline_odometry.maximum_depth_mm),
        (
            "maximum_depth_difference_mm",
            odometry.maximum_depth_difference_mm,
            baseline_odometry.maximum_depth_difference_mm,
        ),
        (
            "evaluation_distance_mm",
            odometry.evaluation_distance_mm,
            baseline_odometry.evaluation_distance_mm,
        ),
        (
            "maximum_inlier_rmse_mm",
            odometry.maximum_inlier_rmse_mm,
            baseline_odometry.maximum_inlier_rmse_mm,
        ),
        (
            "maximum_pair_translation_mm",
            odometry.maximum_pair_translation_mm,
            baseline_odometry.maximum_pair_translation_mm,
        ),
        (
            "maximum_pair_vertical_mm",
            odometry.maximum_pair_vertical_mm,
            baseline_odometry.maximum_pair_vertical_mm,
        ),
        (
            "maximum_pair_forward_mm",
            odometry.maximum_pair_forward_mm,
            baseline_odometry.maximum_pair_forward_mm,
        ),
        (
            "maximum_pair_rotation_deg",
            odometry.maximum_pair_rotation_deg,
            baseline_odometry.maximum_pair_rotation_deg,
        ),
    )
    for name, value, baseline in minimum_odometry:
        if value < baseline:
            raise ValueError(f"Formal rgbd_odometry.{name} cannot be relaxed")
    for name, value, baseline in maximum_odometry:
        if value > baseline:
            raise ValueError(f"Formal rgbd_odometry.{name} cannot be relaxed")
    if any(
        value < baseline
        for value, baseline in zip(
            odometry.iteration_number_per_pyramid_level,
            baseline_odometry.iteration_number_per_pyramid_level,
            strict=True,
        )
    ):
        raise ValueError("Formal RGB-D odometry iteration schedule cannot be reduced")

    pose_limits = PoseQualityThresholds.from_mapping(
        stitch_config.get("pose_quality")
    )
    baseline_pose = PoseQualityThresholds()
    if pose_limits.minimum_scan_span_mm < baseline_pose.minimum_scan_span_mm:
        raise ValueError("Formal minimum pose scan span cannot be relaxed")
    for name in (
        "maximum_reverse_step_mm",
        "maximum_reverse_fraction",
        "maximum_step_translation_mm",
        "maximum_step_vertical_mm",
        "maximum_step_forward_mm",
        "maximum_total_vertical_drift_mm",
        "maximum_total_forward_drift_mm",
        "maximum_step_rotation_deg",
        "maximum_total_rotation_deg",
        "maximum_edge_translation_residual_mm",
        "maximum_edge_rotation_residual_deg",
        "maximum_consecutive_unreliable_edges",
    ):
        if getattr(pose_limits, name) > getattr(baseline_pose, name):
            raise ValueError(f"Formal pose_quality.{name} cannot be relaxed")
    if not pose_limits.require_all_adjacent_edges:
        raise ValueError("Formal pose graph requires every adjacent RGB-D edge")

    graph_mapping = dict(stitch_config.get("pose_graph", {}))
    graph_mapping.pop("nonadjacent_max_gap", None)
    graph = PoseGraphConfig.from_mapping(graph_mapping)
    baseline_graph = PoseGraphConfig()
    if (
        graph.maximum_correspondence_distance_mm
        > baseline_graph.maximum_correspondence_distance_mm
        or graph.edge_prune_threshold < baseline_graph.edge_prune_threshold
        or not np.isclose(
            graph.preference_loop_closure,
            baseline_graph.preference_loop_closure,
        )
    ):
        raise ValueError("Formal pose-graph optimizer settings cannot be relaxed")


def _central_strip_diagnostic_config(
    stitch_config: dict[str, Any], *, diagnostic_renderer: object | None
) -> dict[str, object] | None:
    """Validate the closed central-strip diagnostic configuration boundary.

    The shared YAML keeps this experimental renderer disabled.  A value of
    ``enabled: true`` is never a way for the formal entry point to select it:
    only the independent command can inject a renderer.  The callback receives
    an effective, explicitly enabled copy of this fixed configuration.
    """

    value = stitch_config.get("central_strip_diagnostic", {})
    if not isinstance(value, dict):
        raise ValueError("stitch.central_strip_diagnostic must be a mapping")
    unknown = sorted(set(value) - _CENTRAL_STRIP_DIAGNOSTIC_KEYS)
    if unknown:
        raise ValueError(
            "Unknown central_strip_diagnostic configuration keys: "
            + ", ".join(unknown)
        )
    config = dict(_CENTRAL_STRIP_DIAGNOSTIC_DEFAULTS)
    config.update(value)

    if type(config["enabled"]) is not bool:
        raise ValueError("central_strip_diagnostic.enabled must be a boolean")
    for name, expected in (
        ("reference_scale_mode", "robust_aligned_depth_plane"),
        ("orientation_mode", "verified_camera_to_world"),
        ("exposure_mode", "global_gain"),
    ):
        if config[name] != expected:
            raise ValueError(
                f"central_strip_diagnostic.{name} must remain {expected!r}"
            )

    fraction = config["maximum_central_band_fraction"]
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not np.isfinite(float(fraction))
        or not np.isclose(float(fraction), 0.20)
    ):
        raise ValueError(
            "central_strip_diagnostic.maximum_central_band_fraction must remain 0.20"
        )
    overlap = config["minimum_pair_overlap_pixels"]
    if type(overlap) is not int or overlap != 96:
        raise ValueError(
            "central_strip_diagnostic.minimum_pair_overlap_pixels must remain 96"
        )
    levels = config["multiband_levels"]
    if type(levels) is not int or levels != 5:
        raise ValueError("central_strip_diagnostic.multiband_levels must remain 5")

    if diagnostic_renderer is None:
        if bool(config["enabled"]):
            raise ValueError(
                "central_strip_diagnostic.enabled can only be activated by "
                "g305-central-strip-diagnostic"
            )
        return None
    if not callable(diagnostic_renderer):
        raise TypeError("diagnostic_renderer must be callable")

    # The callback itself is the independent command's explicit opt-in.  Do not
    # mutate the loaded config, since it is also used by formal validation and
    # report construction in callers/tests.
    config["enabled"] = True
    return config


def _sanitized_diagnostic_trajectory(
    trajectory: ORBSLAM3Trajectory | None, *, input_frame_count: int
) -> dict[str, object] | None:
    """Keep tracking evidence without retaining a temporary ORB work path."""

    if trajectory is None:
        return None
    payload = trajectory.as_dict(input_frame_count=input_frame_count)
    for name in (
        "work_dir",
        "settings_path",
        "association_path",
        "trajectory_path",
        "stdout_path",
        "stderr_path",
        "command",
    ):
        payload.pop(name, None)
    return payload


def _validate_central_strip_result(
    result: object,
) -> tuple[np.ndarray, dict[str, object]]:
    """Validate the minimal callback result before it can be atomically published."""

    try:
        panorama = np.asarray(getattr(result, "panorama"))
        metadata = getattr(result, "metadata")
    except AttributeError as exc:
        raise TypeError(
            "diagnostic_renderer must return an object with panorama and metadata"
        ) from exc
    if (
        panorama.ndim != 3
        or panorama.shape[2] != 3
        or panorama.dtype != np.uint8
        or panorama.shape[0] <= 0
        or panorama.shape[1] <= 0
    ):
        raise ValueError(
            "Central-strip diagnostic panorama must be a non-empty BGR uint8 image"
        )
    if not isinstance(metadata, dict):
        raise TypeError("Central-strip diagnostic metadata must be a dictionary")
    return panorama, dict(metadata)


def _publish_central_strip_diagnostic(
    output: Path, panorama: np.ndarray, report: dict[str, object]
) -> None:
    """Atomically publish the two and only two central-strip diagnostics."""

    pending_panorama = _write_bgr(output / ".diagnostic_panorama.pending.jpg", panorama)
    pending_report = output / ".diagnostic_report.pending.json"
    pending_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(pending_panorama, output / "diagnostic_panorama.jpg")
    os.replace(pending_report, output / "diagnostic_report.json")


def _geometry_pair_diagnostic_index(args: argparse.Namespace) -> int:
    """Return one adjacent-pair index without accepting a frame subset.

    The geometry A/B diagnostic deliberately renders the complete real pose
    chain.  Its selector therefore names a *pair position* in that chain,
    never a two-frame render override that would turn both sources into
    endpoint strips or weaken the motion-scale contract.
    """

    value = getattr(args, "pair_index", None)
    if isinstance(value, bool):
        raise ValueError("geometry pair diagnostic pair_index must be an integer")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "geometry pair diagnostic requires an integer --pair-index"
        ) from exc
    if index < 0:
        raise ValueError("geometry pair diagnostic pair_index must be non-negative")
    return index


def _run_pipeline(
    args: argparse.Namespace,
    *,
    odometry_backend: object | None = None,
    diagnostic_renderer: object | None = None,
    geometry_pair_diagnostic_renderer: object | None = None,
    orb_work_root: Path | None = None,
) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    # This is deliberately the first filesystem action in a task.  Even a
    # malformed configuration must not leave a previous success marker live.
    _invalidate_delivery_marker(output)
    if (
        diagnostic_renderer is not None
        and geometry_pair_diagnostic_renderer is not None
    ):
        raise ValueError("Only one diagnostic renderer may be active in a task")
    if (
        geometry_pair_diagnostic_renderer is not None
        and not callable(geometry_pair_diagnostic_renderer)
    ):
        raise TypeError("geometry_pair_diagnostic_renderer must be callable")
    config = load_config(getattr(args, "config", None))
    stitch_config = dict(config["stitch"])
    central_strip_config = _central_strip_diagnostic_config(
        stitch_config, diagnostic_renderer=diagnostic_renderer
    )
    _validate_backend_config(stitch_config)
    diagnostic_force = bool(
        getattr(args, "diagnostic_force", False)
        or stitch_config.get("diagnostic_force", False)
    )
    if geometry_pair_diagnostic_renderer is not None and diagnostic_force:
        raise ValueError(
            "Geometry pair diagnostic cannot use diagnostic-force; its complete "
            "RGB-D/Open3D/ORB-SLAM3 chain remains quality-gated"
        )
    _validate_safety_envelope(
        stitch_config, diagnostic_force=diagnostic_force
    )
    _clear_delivery_files(output)
    _clear_diagnostic_files(output)
    (output / "failure.json").unlink(missing_ok=True)

    session = load_rgbd_session(args.input)
    capture_summary = _capture_manifest_summary(session.manifest)
    if bool(capture_summary["diagnostic_only"]) and not diagnostic_force:
        raise RuntimeError(
            "Input capture session is diagnostic-only; rerun with "
            "--diagnostic-force to write only diagnostic artifacts"
        )
    manual_render_ids = _parse_frame_ids(
        getattr(args, "render_frame_ids", None)
        or stitch_config.get("render_frame_ids")
    )
    if geometry_pair_diagnostic_renderer is not None and manual_render_ids:
        raise ValueError(
            "Geometry pair diagnostic always renders the complete scan; "
            "render_frame_ids is not permitted"
        )
    if manual_render_ids and not diagnostic_force:
        raise ValueError(
            "render_frame_ids cannot publish a complete quality-gated delivery"
        )

    analysis_width = int(stitch_config.get("analysis_width", 320))
    qualities, motions, analysis_shape = _analyse_session(
        session.frames, analysis_width
    )
    pose_frames, pose_qualities, layout_metadata = _select_pose_nodes(
        session.frames,
        qualities,
        motions,
        analysis_shape[1],
        stitch_config,
    )
    segment_row = layout_metadata["segment"]
    segment_start = int(segment_row["start_index"])
    segment_stop = int(segment_row["end_index"]) + 1
    scan_frames = session.frames[segment_start:segment_stop]
    capture_quality = assess_capture_quality(
        qualities[segment_start:segment_stop],
        [
            frame.color_exposure_raw
            for frame in session.frames[segment_start:segment_stop]
        ],
        exposure_unit_us=float(stitch_config.get("color_exposure_unit_us", 100.0)),
        maximum_exposure_us=float(
            stitch_config.get("maximum_motion_exposure_us", 1200.0)
        ),
    )
    if (
        not diagnostic_force
        and bool(stitch_config.get("input_quality_gate", True))
        and not bool(capture_quality["quality_pass"])
    ):
        reasons = "; ".join(
            str(value) for value in capture_quality["failure_reasons"]
        )
        raise RuntimeError("Input capture quality gate failed: " + reasons)

    odometry_config = RGBDOdometryConfig.from_mapping(
        stitch_config.get("rgbd_odometry")
    )
    graph_mapping = dict(stitch_config.get("pose_graph", {}))
    nonadjacent_gap = int(graph_mapping.pop("nonadjacent_max_gap", 2))
    graph_config = PoseGraphConfig.from_mapping(graph_mapping)
    pose_thresholds = PoseQualityThresholds.from_mapping(
        stitch_config.get("pose_quality")
    )
    started = time.perf_counter()
    short_baseline_audit: list[dict[str, object]] = []
    edges, optional_edge_failures = _estimate_pose_edges(
        pose_frames,
        session.calibration,
        odometry_config,
        backend=odometry_backend,
        nonadjacent_gap=nonadjacent_gap,
        scan_frames=scan_frames,
        short_baseline_audit=short_baseline_audit,
        short_baseline_initialization=bool(
            stitch_config.get("short_baseline_initialization", True)
        ),
    )
    pose_backend = str(stitch_config.get("pose_backend", "hybrid_orbslam3_rgbd"))
    if (
        geometry_pair_diagnostic_renderer is not None
        and pose_backend != "hybrid_orbslam3_rgbd"
    ):
        raise RuntimeError(
            "Geometry pair diagnostic requires hybrid_orbslam3_rgbd and a "
            "current full-scan ORB-SLAM3 trajectory"
        )
    orbslam3_trajectory: ORBSLAM3Trajectory | None = None
    global_pose_backend: object | None = odometry_backend
    if pose_backend == "hybrid_orbslam3_rgbd" and odometry_backend is None:
        print(
            "[ORB-SLAM3] solving the global RGB-D trajectory from the complete "
            "short-baseline scan"
        )
        if (
            diagnostic_renderer is not None
            or geometry_pair_diagnostic_renderer is not None
        ) and orb_work_root is None:
            raise RuntimeError(
                "Diagnostic rendering requires an isolated ORB staging directory"
            )
        if orb_work_root is not None:
            orbslam3_trajectory = run_orbslam3_rgbd(
                scan_frames,
                session.calibration,
                orb_work_root,
                config=stitch_config.get("orbslam3_rgbd"),
            )
        else:
            # The formal output directory must contain only final pending or
            # delivered artifacts.  RGB-D staging is sensitive analysis input
            # and belongs to a system temporary directory even on success.
            with tempfile.TemporaryDirectory(prefix="g305-panorama-orbslam3-") as root:
                orbslam3_trajectory = run_orbslam3_rgbd(
                    scan_frames,
                    session.calibration,
                    Path(root),
                    config=stitch_config.get("orbslam3_rgbd"),
                )
        global_pose_backend = ORBSLAM3PoseGraphOptimizer(orbslam3_trajectory)
    pose_graph = optimize_rgbd_pose_graph(
        pose_frames,
        edges,
        config=graph_config,
        backend=global_pose_backend,
        enforce_edge_quality=not diagnostic_force,
    )
    pose_quality_result = validate_pose_trajectory(
        pose_graph, thresholds=pose_thresholds
    )
    pose_quality = pose_quality_result.as_dict()
    if not diagnostic_force and not pose_quality_result.quality_pass:
        raise RuntimeError(
            "RGB-D pose trajectory quality gate failed: "
            + "; ".join(pose_quality_result.failure_reasons)
        )
    pose_values = [pose_graph.pose_for(frame.frame_id) for frame in pose_frames]

    if (
        geometry_pair_diagnostic_renderer is not None
        and orbslam3_trajectory is None
    ):
        raise RuntimeError(
            "Geometry pair diagnostic requires a current full-scan ORB-SLAM3 "
            "trajectory; it cannot reuse a saved transform sidecar or an "
            "Open3D-only pose graph"
        )

    if diagnostic_renderer is not None:
        # The legacy reference-plane diagnostic remains isolated behind its
        # injected callback.  It may use its own depth-only layout estimator;
        # the normal g305-panorama path below does not import or invoke it.
        from .rgbd_projection import estimate_projection_canvas

        legacy_projection_config = dict(stitch_config.get("rgbd_projection", {}))
        maximum_projection_depth_mm = float(
            legacy_projection_config.get("maximum_projection_depth_mm", 2000.0)
        )
        if odometry_backend is not None:
            maximum_projection_depth_mm = max(
                maximum_projection_depth_mm, odometry_config.maximum_depth_mm
            )
        footprint_frames, footprint_intrinsics = _working_projection_frames(
            pose_frames,
            pose_values,
            session.calibration,
            int(
                legacy_projection_config.get(
                    "footprint_working_width", odometry_config.working_width
                )
            ),
        )
        footprint_canvas = estimate_projection_canvas(
            footprint_frames,
            footprint_intrinsics,
            max_canvas_megapixels=1_000_000.0,
            max_aggregate_megapixels=1_000_000_000.0,
            maximum_depth_mm=maximum_projection_depth_mm,
        )
        full_resolution_scale = (
            session.calibration.width / float(footprint_intrinsics.width)
        )
        estimated_full_canvas_mp = (
            footprint_canvas.canvas_megapixels * full_resolution_scale**2
        )
        del footprint_frames
        if (
            not np.isfinite(estimated_full_canvas_mp)
            or estimated_full_canvas_mp > _HARD_MAX_CANVAS_MEGAPIXELS
        ):
            raise RuntimeError(
                "Central-strip diagnostic estimated canvas exceeds the hard "
                f"{_HARD_MAX_CANVAS_MEGAPIXELS:.0f} MP limit"
            )
        # The central-strip renderer composes only a narrow calibrated band.
        # Its overlap requirement is deliberately denser than the full-FOV
        # delivery selector, so feed it chronological *real* pose nodes rather
        # than allowing that selector to collapse the chain to endpoints.  The
        # hard cap remains non-bypassable; when more nodes are available, take
        # an evenly spaced subset that includes both chronological endpoints.
        if len(pose_frames) <= _HARD_MAX_RENDER_SOURCES:
            render_indices = list(range(len(pose_frames)))
        else:
            render_indices = sorted(
                {
                    int(value)
                    for value in np.linspace(
                        0,
                        len(pose_frames) - 1,
                        _HARD_MAX_RENDER_SOURCES,
                    ).round()
                }
            )
        if len(render_indices) < 2:
            raise RuntimeError(
                "Central-strip diagnostic requires at least two optimized pose nodes"
            )
        render_selection: dict[str, object] = {
            "mode": "central_strip_real_pose_nodes",
            "frame_ids": [pose_frames[index].frame_id for index in render_indices],
            "interpolated_pose_count": 0,
            "source_cap": _HARD_MAX_RENDER_SOURCES,
        }
    elif manual_render_ids:
        index_by_id = {frame.frame_id: index for index, frame in enumerate(pose_frames)}
        missing = [frame_id for frame_id in manual_render_ids if frame_id not in index_by_id]
        if missing:
            raise ValueError(
                "Manual diagnostic render ids are not optimized pose nodes: "
                f"{missing}"
            )
        render_indices = [index_by_id[frame_id] for frame_id in manual_render_ids]
        render_selection: dict[str, object] = {
            "mode": "diagnostic_manual_pose_nodes",
            "frame_ids": manual_render_ids,
            "interpolated_pose_count": 0,
        }
    else:
        # The pushbroom's bounded strip cache removes the former 32-source
        # full-canvas limit.  Every chronological, optimized real pose node
        # contributes one narrow calibrated RGB strip; no pose is synthesized,
        # reordered, or interpolated.
        render_indices = list(range(len(pose_frames)))
        render_selection = {
            "mode": "calibrated_rgb_pushbroom_all_real_pose_nodes",
            "frame_ids": [frame.frame_id for frame in pose_frames],
            "interpolated_pose_count": 0,
            "source_cap": int(
                dict(stitch_config.get("calibrated_rgb_pushbroom", {})).get(
                    "max_pose_count", _HARD_MAX_LAYOUT_FRAMES
                )
            ),
            "streaming": True,
        }
    render_frames = [pose_frames[index] for index in render_indices]
    render_poses = [pose_values[index] for index in render_indices]
    render_qualities = [pose_qualities[index] for index in render_indices]
    # Existing RGB thumbnail motion is a local scalar observation only.  It is
    # scaled to native colour pixels and paired with the already verified real
    # SE(3) camera-centre displacement inside the renderer; it never becomes a
    # 2-D pose, transform, ordering rule, or interpolation source.
    all_real_sources = (
        len(render_frames) == len(scan_frames)
        and [frame.frame_id for frame in render_frames]
        == [frame.frame_id for frame in scan_frames]
    )
    render_motions: list[MotionEstimate] | None = (
        motions[segment_start:segment_stop] if all_real_sources else None
    )
    if diagnostic_renderer is not None:
        assert central_strip_config is not None
        callback_result = diagnostic_renderer(
            plane_frames=pose_frames,
            plane_poses=pose_values,
            render_frames=render_frames,
            render_poses=render_poses,
            calibration=session.calibration,
            config=central_strip_config,
            sharpness_scores=[quality.sharpness for quality in render_qualities],
        )
        panorama, render_metadata = _validate_central_strip_result(callback_result)
        render_metadata.setdefault(
            "frame_ids", [frame.frame_id for frame in render_frames]
        )
        render_metadata.setdefault("selection", render_selection)
        render_metadata.setdefault("interpolated_pose_count", 0)
        render_metadata.setdefault(
            "source_quality",
            {
                str(frame.frame_id): quality.as_dict()
                for frame, quality in zip(
                    render_frames, render_qualities, strict=True
                )
            },
        )
        transforms_payload = pose_graph.as_dict()
        transforms_payload["layout_selection"] = layout_metadata
        transforms_payload["optional_edge_failures"] = optional_edge_failures
        transforms_payload["short_baseline_initialization"] = short_baseline_audit
        transforms_payload["global_trajectory"] = _sanitized_diagnostic_trajectory(
            orbslam3_trajectory, input_frame_count=len(scan_frames)
        )
        panorama_path = output / "diagnostic_panorama.jpg"
        report_path = output / "diagnostic_report.json"
        report: dict[str, Any] = {
            "schema": "gemini305-central-strip-diagnostic/v1",
            "input": str(args.input.expanduser().resolve()),
            "panorama": str(panorama_path),
            "report": str(report_path),
            "diagnostic_only": True,
            "deliverable_published": False,
            "geometry_claim": "reference_plane_only",
            "input_capture": capture_summary,
            "rgbd_session": {
                "root": str(session.root),
                "frame_count": len(session.frames),
                "depth_alignment": session.depth_alignment,
                "depth_unit": "mm",
                "calibration": _intrinsics_payload(session.calibration),
            },
            "input_quality": capture_quality,
            "layout_selection": layout_metadata,
            "odometry": {
                "backend": "open3d_rgbd",
                "config": {
                    "working_width": odometry_config.working_width,
                    "require_aligned_depth": odometry_config.require_aligned_depth,
                    "require_calibration": odometry_config.require_calibration,
                },
                "edges": [edge.as_dict() for edge in edges],
                "optional_edge_failures": optional_edge_failures,
                "short_baseline_initialization": short_baseline_audit,
            },
            "global_trajectory": transforms_payload["global_trajectory"],
            "pose_graph": transforms_payload,
            "pose_quality": pose_quality,
            "render_strategy": "central_strip_plane_diagnostic",
            "central_strip_config": central_strip_config,
            "central_strip": render_metadata,
            "render": render_metadata,
            "diagnostic_overrides": (
                {
                    "input_quality_thresholds_bypassed": True,
                    "odometry_quality_thresholds_bypassed": True,
                    "pose_quality_thresholds_bypassed": True,
                    "final_image_quality_thresholds_bypassed": True,
                    "calibration_aligned_depth_finite_se3_graph_connectivity_"
                    "projection_topology_memory_atomic_safety_required": True,
                }
                if diagnostic_force
                else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _publish_central_strip_diagnostic(output, panorama, report)
        return report

    if geometry_pair_diagnostic_renderer is not None:
        if not all_real_sources:
            raise RuntimeError(
                "Geometry pair diagnostic requires every real full-scan pose node"
            )
        pair_index = _geometry_pair_diagnostic_index(args)
        if pair_index >= len(render_frames) - 1:
            raise ValueError(
                "Geometry pair diagnostic pair_index must select one adjacent "
                f"pair in [0, {len(render_frames) - 2}]"
            )
        seam_config = dict(stitch_config.get("scan_seam", {}))
        pushbroom_config = dict(stitch_config.get("calibrated_rgb_pushbroom", {}))
        callback_result = geometry_pair_diagnostic_renderer(
            render_frames=render_frames,
            render_poses=render_poses,
            calibration=session.calibration,
            config=pushbroom_config,
            rgb_motions=render_motions,
            motion_pixels_to_full_resolution=(
                session.calibration.width / float(analysis_shape[1])
            ),
            multiband_levels=int(seam_config.get("multiband_levels", 3)),
            pair_index=pair_index,
        )
        panorama, render_metadata = _validate_central_strip_result(callback_result)
        panorama_path = output / "diagnostic_panorama.jpg"
        report_path = output / "diagnostic_report.json"
        trajectory_summary = _sanitized_diagnostic_trajectory(
            orbslam3_trajectory, input_frame_count=len(scan_frames)
        )
        report = {
            "schema": "gemini305-geometry-pair-diagnostic/v1",
            "input": str(args.input.expanduser().resolve()),
            "panorama": str(panorama_path),
            "report": str(report_path),
            "diagnostic_only": True,
            "deliverable_published": False,
            "trajectory_provenance": "current_orbslam3_rgbd_full_scan",
            "input_capture": capture_summary,
            "rgbd_session": {
                "frame_count": len(session.frames),
                "depth_alignment": session.depth_alignment,
                "depth_unit": "mm",
                "calibration": _intrinsics_payload(session.calibration),
            },
            "odometry": {
                "backend": "open3d_rgbd",
                "edge_count": len(edges),
                "required_adjacent_edge_count": len(pose_frames) - 1,
                "all_adjacent_edges_required": True,
                "optional_edge_failure_count": len(optional_edge_failures),
            },
            "global_trajectory": trajectory_summary,
            "pose_quality": pose_quality,
            "render_strategy": "full_sequence_geometry_pair_ab_crop",
            "geometry_pair_diagnostic": render_metadata,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _publish_central_strip_diagnostic(output, panorama, report)
        return report

    seam_config = dict(stitch_config.get("scan_seam", {}))
    pushbroom_config = dict(stitch_config.get("calibrated_rgb_pushbroom", {}))
    pushbroom_result = render_calibrated_rgb_pushbroom(
        render_frames,
        render_poses,
        session.calibration,
        config=pushbroom_config,
        rgb_motions=render_motions,
        motion_pixels_to_full_resolution=(
            session.calibration.width / float(analysis_shape[1])
        ),
        multiband_levels=int(seam_config.get("multiband_levels", 3)),
        quality_gate=(
            not diagnostic_force and bool(seam_config.get("quality_gate", True))
        ),
    )
    panorama = pushbroom_result.panorama
    render_metadata = dict(pushbroom_result.metadata)
    render_metadata["frame_ids"] = [frame.frame_id for frame in render_frames]
    render_metadata["selection"] = render_selection
    render_metadata["source_quality"] = {
        str(frame.frame_id): quality.as_dict()
        for frame, quality in zip(render_frames, render_qualities, strict=True)
    }
    if not diagnostic_force:
        _ensure_publishable_quality(
            capture_quality,
            render_metadata,
            pose_quality,
        )

    # TSDF is deliberately an output-only 3-D inspection stage.  It executes
    # only after the RGB-only panorama has passed all of its own gates, and it
    # receives real RGB-D poses without returning any result to that renderer.
    tsdf_visualization: dict[str, object] | None = None
    tsdf_mesh_glb: bytes | None = None
    if not diagnostic_force:
        visualization_config = dict(stitch_config.get("tsdf_visualization", {}))
        if bool(visualization_config.get("enabled", True)):
            tsdf_mesh_glb, mesh_metadata = _export_display_only_tsdf_mesh(
                pose_frames,
                pose_values,
                session.calibration,
                visualization_config,
            )
            tsdf_visualization = {
                **mesh_metadata,
                "mesh": str(output / "tsdf_mesh.glb"),
                "viewer": str(output / "tsdf_mesh_viewer.html"),
            }

    panorama_path = output / (
        "diagnostic_panorama.jpg" if diagnostic_force else "panorama.jpg"
    )
    report_path = output / (
        "diagnostic_report.json" if diagnostic_force else "report.json"
    )
    transforms_payload = pose_graph.as_dict()
    transforms_payload["layout_selection"] = layout_metadata
    transforms_payload["optional_edge_failures"] = optional_edge_failures
    transforms_payload["short_baseline_initialization"] = short_baseline_audit
    transforms_payload["global_trajectory"] = _sanitized_diagnostic_trajectory(
        orbslam3_trajectory, input_frame_count=len(scan_frames)
    )
    residual_settings = ResidualAlignmentConfig.from_mapping(
        pushbroom_config.get("residual_alignment")
    )
    compact_residual_alignment = _compact_residual_alignment_for_transforms(
        render_metadata,
        residual_settings,
        [frame.frame_id for frame in render_frames],
    )
    geometry_settings = GeometryAssistedSeamConfig.from_mapping(
        pushbroom_config.get("geometry_assisted_seam")
    )
    compact_geometry_assistance = _compact_geometry_assistance_for_transforms(
        render_metadata,
        geometry_settings,
        [frame.frame_id for frame in render_frames],
    )
    render_transforms_payload = {
        "schema": "calibrated-rgb-pushbroom/v7",
        "translation_unit": "mm",
        "pixel_source": "calibrated_rgb_source_samples",
        "depth_used_for_output_pixels": False,
        "depth_used_for_local_geometry": compact_geometry_assistance[
            "depth_used_for_local_geometry"
        ],
        "layout": render_metadata.get("layout"),
        "rgb_motion_scale": render_metadata.get("rgb_motion_scale"),
        "selection": render_selection,
        "residual_alignment": compact_residual_alignment,
        "geometry_assisted_seam": compact_geometry_assistance,
        "sources": [
            {
                "frame_id": frame.frame_id,
                "color_path": str(frame.color_path),
                "camera_to_world": pose.tolist(),
            }
            for frame, pose in zip(render_frames, render_poses, strict=True)
        ],
    }
    report: dict[str, Any] = {
        "schema": "gemini305-calibrated-rgb-pushbroom/v8",
        "input": str(args.input.expanduser().resolve()),
        "panorama": str(panorama_path),
        "report": str(report_path),
        "diagnostic_only": diagnostic_force,
        "deliverable_published": not diagnostic_force,
        "input_capture": capture_summary,
        "rgbd_session": {
            "root": str(session.root),
            "frame_count": len(session.frames),
            "depth_alignment": session.depth_alignment,
            "depth_unit": "mm",
            "calibration": _intrinsics_payload(session.calibration),
        },
        "input_quality": capture_quality,
        "layout_selection": layout_metadata,
        "odometry": {
            "backend": "open3d_rgbd",
            "config": {
                "working_width": odometry_config.working_width,
                "require_aligned_depth": odometry_config.require_aligned_depth,
                "require_calibration": odometry_config.require_calibration,
            },
            "edges": [edge.as_dict() for edge in edges],
            "optional_edge_failures": optional_edge_failures,
            "short_baseline_initialization": short_baseline_audit,
        },
        "global_trajectory": transforms_payload["global_trajectory"],
        "pose_graph": transforms_payload,
        "pose_quality": pose_quality,
        "projection": render_transforms_payload,
        "render_strategy": "calibrated_rgb_pushbroom",
        "render": render_metadata,
        "tsdf_visualization": tsdf_visualization,
        "diagnostic_overrides": (
            {
                "input_quality_thresholds_bypassed": True,
                "odometry_quality_thresholds_bypassed": True,
                "pose_quality_thresholds_bypassed": True,
                "final_image_quality_thresholds_bypassed": True,
                "calibration_aligned_depth_finite_se3_graph_connectivity_"
                "projection_topology_memory_atomic_safety_required": True,
            }
            if diagnostic_force
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }

    if diagnostic_force:
        pending_panorama = _write_bgr(
            output / ".diagnostic_panorama.pending.jpg", panorama
        )
        pending_report = output / ".diagnostic_report.pending.json"
        pending_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(pending_panorama, panorama_path)
        os.replace(pending_report, report_path)
        return report

    pending_panorama = _write_bgr(output / ".panorama.pending.jpg", panorama)
    pending_mesh: Path | None = None
    pending_mesh_viewer: Path | None = None
    if tsdf_mesh_glb is not None:
        pending_mesh = _write_bytes(output / ".tsdf_mesh.pending.glb", tsdf_mesh_glb)
        pending_mesh_viewer = output / ".tsdf_mesh_viewer.pending.html"
        pending_mesh_viewer.write_text(
            _mesh_viewer_html("tsdf_mesh.glb"), encoding="utf-8"
        )
    pending_transforms = output / ".transforms.pending.json"
    pending_render_transforms = output / ".render_transforms.pending.json"
    pending_report = output / ".report.pending.json"
    pending_transforms.write_text(
        json.dumps(transforms_payload, indent=2), encoding="utf-8"
    )
    pending_render_transforms.write_text(
        json.dumps(render_transforms_payload, indent=2), encoding="utf-8"
    )
    pending_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(pending_panorama, output / "panorama.jpg")
    if pending_mesh is not None and pending_mesh_viewer is not None:
        os.replace(pending_mesh, output / "tsdf_mesh.glb")
        os.replace(pending_mesh_viewer, output / "tsdf_mesh_viewer.html")
    os.replace(pending_transforms, output / "transforms.json")
    os.replace(pending_render_transforms, output / "render_transforms.json")
    os.replace(pending_report, output / "report.json")
    delivery = {
        "schema": "gemini305-panorama-delivery/v8",
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "quality_pass": True,
        "pose_backend": (
            "hybrid_orbslam3_rgbd"
            if orbslam3_trajectory is not None
            else "open3d_rgbd"
        ),
        "projection": "calibrated_rgb_pushbroom",
        "alignment_backend": compact_residual_alignment["backend"],
        "alignment_model": compact_residual_alignment["selected_model"],
        "seam_backend": "rgb_monotonic_hard_owner_graphcut",
        "blend_backend": "safe_wall_local_multiband_narrow_owner_boundary",
        "pixel_source": "calibrated_rgb_source_samples",
        "depth_used_for_output_pixels": False,
        "depth_used_for_local_geometry": compact_geometry_assistance[
            "depth_used_for_local_geometry"
        ],
        "geometry_assistance_backend": compact_geometry_assistance["backend"],
        "geometry_assistance_gate": {
            "minimum_active_mesh_cells": int(
                geometry_settings.minimum_active_mesh_cells
            ),
            "maximum_straight_line_deviation_pixels": float(
                geometry_settings.maximum_straight_line_deviation_pixels
            ),
            "rgb_flow_application_policy": (
                "accepted_bidirectional_rgb_flow_and_epipolar_support_"
                "including_held_out"
            ),
            "rgb_flow_fit_support_policy": (
                "training_only_accepted_bidirectional_rgb_flow_and_epipolar_support"
            ),
            "actual_rgb_line_observation_policy": (
                "observed_hough_solver_line_veto_or_not_observed_non_veto"
            ),
        },
        "panorama": str(output / "panorama.jpg"),
        "tsdf_visualization": tsdf_visualization,
        "report": str(output / "report.json"),
    }
    pending_delivery = output / ".delivery.pending.json"
    pending_delivery.write_text(json.dumps(delivery, indent=2), encoding="utf-8")
    os.replace(pending_delivery, output / "delivery.json")
    return report


def run(
    args: argparse.Namespace,
    *,
    odometry_backend: object | None = None,
    diagnostic_renderer: object | None = None,
    geometry_pair_diagnostic_renderer: object | None = None,
) -> dict[str, Any]:
    """Run one task and persist fail-closed state for every ordinary error.

    ``diagnostic_renderer`` is reserved for the independent central-strip
    command.  ``geometry_pair_diagnostic_renderer`` is a separate diagnostic
    seam for a full-sequence RGB-only A/B crop.  Neither can reach formal
    delivery.
    """

    output = args.output.expanduser().resolve()
    try:
        if (
            diagnostic_renderer is None
            and geometry_pair_diagnostic_renderer is None
        ):
            return _run_pipeline(args, odometry_backend=odometry_backend)

        # Preserve the first-action delivery invalidation invariant before
        # creating an external ORB staging directory.  The bridge receives this
        # temporary root only on the callback route, so a successful diagnostic
        # output never retains .orbslam3_rgbd under the requested output path.
        _invalidate_delivery_marker(output)
        prefix = (
            "g305-central-strip-orbslam3-"
            if diagnostic_renderer is not None
            else "g305-geometry-pair-orbslam3-"
        )
        with tempfile.TemporaryDirectory(prefix=prefix) as root:
            return _run_pipeline(
                args,
                odometry_backend=odometry_backend,
                diagnostic_renderer=diagnostic_renderer,
                geometry_pair_diagnostic_renderer=geometry_pair_diagnostic_renderer,
                orb_work_root=Path(root),
            )
    except Exception as exc:
        _write_failure_report(output, args.input, exc)
        raise


def main() -> None:
    args = _parser().parse_args()
    if "unistitch-sequence" in Path(sys.argv[0]).name.lower():
        print(
            "WARNING: unistitch-sequence is deprecated; use g305-panorama. "
            "Both commands run the same RGB-D Open3D pipeline.",
            file=sys.stderr,
        )
    try:
        report = run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Panorama: {report['panorama']}")
    print(f"Report: {report['report']}")
    if bool(report.get("diagnostic_only", False)):
        print("Diagnostic only: no delivery.json was published")


if __name__ == "__main__":
    main()
