"""Streaming calibrated pushbroom panorama renderer with local RGB-D seam aid.

This module deliberately separates *pose evidence* from *pixel generation*.
The caller supplies already-validated, real camera-to-world SE(3) poses (normally
from the RGB-D Open3D/ORB-SLAM3 stages).  Final panorama colour always originates
from a calibrated RGB source sample.  Aligned depth is read only by the explicitly
audited geometry-assist analysis: it may classify a narrow adjacent seam as the
same surface, an occlusion, a disocclusion, or an unreliable depth/transparent
region, and it may supply a bounded *inverse sampling* correction.  It never
replaces a real pose, creates colour, fills a hole, constructs a TSDF, or supplies
a global projection.

Each source is inverse-remapped exactly once from raw calibrated RGB into a narrow
pose-levelled strip.  Narrow strips are spooled to a temporary directory, allowing
the exposure and seam passes to keep only adjacent strips in memory.  This is a
pushbroom image: its scan-coordinate scale is a robust *local RGB motion per real
camera-centre displacement* estimate, not an estimated 2-D camera trajectory.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .calibrated_remap import camera_points_to_source_pixels
from .foreground_segments import (
    GeometryMode,
    build_foreground_fragments,
    plan_foreground_owners,
)
from .geometry_assisted_local_warp import (
    GeometryAssistConfig,
    LocalMeshInverseWarp,
    LocalMeshWarpConfig,
    TileBounds,
    analyze_adjacent_rgbd_pair,
    fit_local_mesh_inverse_warp,
    mutually_consistent_correspondences,
    solve_active_mesh_forward_inverse,
)
from .render import largest_valid_rectangle
from .rgb_residual_alignment import (
    ProtectedComponentFragment,
    ResidualAlignmentConfig,
    ResidualAlignmentResult,
    SourceResidualWarp,
    audit_source_warps,
    extract_protected_component_fragments,
    extract_pair_evidence,
    measure_owner_boundary_geometry,
    preflight_sequence_owners,
)
from .session import CameraIntrinsics, RGBDFrame, read_aligned_depth_mm


_HARD_MAX_POSES = 160
_HARD_MAX_CANVAS_MEGAPIXELS = 200.0
_HARD_MAX_RESIDENT_STRIPS = 5


@dataclass(frozen=True)
class GeometryAssistedSeamConfig:
    """Closed limits for depth-assisted local seam correction.

    The whole renderer remains calibrated RGB pushbroom.  This configuration
    merely controls whether a high-risk adjacent corridor may receive an
    evidence-backed local inverse sampling field before the source's one output
    RGB remap.  Values are deliberately closed because changing a local map
    without a geometry/held-out audit is worse than retaining a visible hard
    cut.
    """

    enabled: bool = True
    analysis_corridor_width_pixels: int = 128
    trigger_edge_offset_p95_pixels: float = 1.0
    absolute_depth_tolerance_mm: float = 20.0
    relative_depth_tolerance: float = 0.02
    depth_noise_mm: float = 0.0
    mutual_reprojection_tolerance_pixels: float = 0.40
    edge_guard_radius_pixels: int = 8
    minimum_trigger_boundary_observable_pixels: int = 32
    flow_validation_preview_scale: float = 0.75
    minimum_held_out_flow_validation_pixels: int = 8
    minimum_held_out_strong_edge_validation_pixels: int = 8
    maximum_held_out_flow_fb_error_pixels: float = 0.75
    mesh_cell_pixels: int = 16
    minimum_mutual_correspondences: int = 30
    minimum_active_mesh_cells: int = 4
    maximum_local_displacement_pixels: float = 8.0
    maximum_straight_line_deviation_pixels: float = 1.0
    minimum_actual_rgb_line_length_pixels: float = 24.0
    minimum_actual_rgb_line_support_fraction: float = 0.80
    maximum_actual_rgb_line_segments: int = 32
    actual_rgb_line_inverse_maximum_iterations: int = 8
    actual_rgb_line_inverse_maximum_residual_pixels: float = 0.05
    maximum_held_out_error_pixels: float = 0.75
    maximum_held_out_maximum_error_pixels: float = 2.0
    minimum_held_out_improvement_pixels: float = 0.05
    minimum_held_out_improvement_ratio: float = 0.30

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "GeometryAssistedSeamConfig":
        supplied = {} if value is None else dict(value)
        allowed = {
            "enabled",
            "analysis_corridor_width_pixels",
            "trigger_edge_offset_p95_pixels",
            "absolute_depth_tolerance_mm",
            "relative_depth_tolerance",
            "depth_noise_mm",
            "mutual_reprojection_tolerance_pixels",
            "edge_guard_radius_pixels",
            "minimum_trigger_boundary_observable_pixels",
            "flow_validation_preview_scale",
            "minimum_held_out_flow_validation_pixels",
            "minimum_held_out_strong_edge_validation_pixels",
            "maximum_held_out_flow_fb_error_pixels",
            "mesh_cell_pixels",
            "minimum_mutual_correspondences",
            "minimum_active_mesh_cells",
            "maximum_local_displacement_pixels",
            "maximum_straight_line_deviation_pixels",
            "minimum_actual_rgb_line_length_pixels",
            "minimum_actual_rgb_line_support_fraction",
            "maximum_actual_rgb_line_segments",
            "actual_rgb_line_inverse_maximum_iterations",
            "actual_rgb_line_inverse_maximum_residual_pixels",
            "maximum_held_out_error_pixels",
            "maximum_held_out_maximum_error_pixels",
            "minimum_held_out_improvement_pixels",
            "minimum_held_out_improvement_ratio",
        }
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(
                "Unknown geometry_assisted_seam configuration keys: "
                + ", ".join(unknown)
            )
        try:
            result = cls(**supplied)
        except TypeError as exc:
            raise ValueError("Invalid geometry_assisted_seam configuration") from exc
        result._validate()
        return result

    def _validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("geometry_assisted_seam.enabled must be a boolean")
        if not 96 <= int(self.analysis_corridor_width_pixels) <= 160:
            raise ValueError(
                "geometry_assisted_seam.analysis_corridor_width_pixels must be in [96, 160]"
            )
        if int(self.mesh_cell_pixels) not in {16, 32}:
            raise ValueError("geometry_assisted_seam.mesh_cell_pixels must be 16 or 32")
        if not 8 <= int(self.edge_guard_radius_pixels) <= 12:
            raise ValueError(
                "geometry_assisted_seam.edge_guard_radius_pixels must be in [8, 12]"
            )
        if int(self.minimum_trigger_boundary_observable_pixels) < 32:
            raise ValueError(
                "geometry_assisted_seam.minimum_trigger_boundary_observable_pixels must be at least 32"
            )
        if not 0.50 <= float(self.flow_validation_preview_scale) <= 0.75:
            raise ValueError(
                "geometry_assisted_seam.flow_validation_preview_scale must be in [0.50, 0.75]"
            )
        if int(self.minimum_held_out_flow_validation_pixels) < 8:
            raise ValueError(
                "geometry_assisted_seam.minimum_held_out_flow_validation_pixels must be at least 8"
            )
        if int(self.minimum_held_out_strong_edge_validation_pixels) < 8:
            raise ValueError(
                "geometry_assisted_seam.minimum_held_out_strong_edge_validation_pixels must be at least 8"
            )
        if float(self.minimum_actual_rgb_line_length_pixels) < 24.0:
            raise ValueError(
                "geometry_assisted_seam.minimum_actual_rgb_line_length_pixels must be at least 24"
            )
        if not 0.80 <= float(self.minimum_actual_rgb_line_support_fraction) <= 1.0:
            raise ValueError(
                "geometry_assisted_seam.minimum_actual_rgb_line_support_fraction must be in [0.80, 1]"
            )
        if not 1 <= int(self.maximum_actual_rgb_line_segments) <= 32:
            raise ValueError(
                "geometry_assisted_seam.maximum_actual_rgb_line_segments must be in [1, 32]"
            )
        if not 1 <= int(self.actual_rgb_line_inverse_maximum_iterations) <= 8:
            raise ValueError(
                "geometry_assisted_seam.actual_rgb_line_inverse_maximum_iterations must be in [1, 8]"
            )
        if int(self.minimum_mutual_correspondences) < 30:
            raise ValueError(
                "geometry_assisted_seam.minimum_mutual_correspondences must be at least 30"
            )
        if (
            not isinstance(self.minimum_active_mesh_cells, (int, np.integer))
            or int(self.minimum_active_mesh_cells) < 4
        ):
            raise ValueError(
                "geometry_assisted_seam.minimum_active_mesh_cells must be at least 4"
            )
        numeric = np.asarray(
            (
                self.trigger_edge_offset_p95_pixels,
                self.absolute_depth_tolerance_mm,
                self.relative_depth_tolerance,
                self.depth_noise_mm,
                self.mutual_reprojection_tolerance_pixels,
                self.flow_validation_preview_scale,
                self.maximum_held_out_flow_fb_error_pixels,
                self.maximum_local_displacement_pixels,
                self.maximum_straight_line_deviation_pixels,
                self.minimum_actual_rgb_line_length_pixels,
                self.minimum_actual_rgb_line_support_fraction,
                self.actual_rgb_line_inverse_maximum_residual_pixels,
                self.maximum_held_out_error_pixels,
                self.maximum_held_out_maximum_error_pixels,
                self.minimum_held_out_improvement_pixels,
                self.minimum_held_out_improvement_ratio,
            ),
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
            raise ValueError("geometry_assisted_seam settings must be finite and non-negative")
        if self.trigger_edge_offset_p95_pixels <= 0.0:
            raise ValueError("geometry_assisted_seam.trigger_edge_offset_p95_pixels must be positive")
        # These two terms define the formal max(20 mm, 2% depth, 3 sigma)
        # contract.  A user-supplied larger value would silently turn an
        # occlusion guard into a permissive depth match, so the formal renderer
        # fixes them rather than treating them as tuneable quality knobs.
        if not math.isclose(
            float(self.absolute_depth_tolerance_mm), 20.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "geometry_assisted_seam.absolute_depth_tolerance_mm must equal 20"
            )
        if not math.isclose(
            float(self.relative_depth_tolerance), 0.02, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "geometry_assisted_seam.relative_depth_tolerance must equal 0.02"
            )
        # The current strict session contract has no validated per-pixel noise
        # calibration.  Retain the 3-sigma term in GeometryAssistConfig, but
        # force its unknown sigma to zero rather than accepting an unverified
        # wider tolerance in either formal or diagnostic seam decisions.
        if not math.isclose(
            float(self.depth_noise_mm), 0.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                "geometry_assisted_seam.depth_noise_mm must be zero until calibrated noise provenance is available"
            )
        if not 0.0 < self.mutual_reprojection_tolerance_pixels <= 0.40:
            raise ValueError(
                "geometry_assisted_seam.mutual_reprojection_tolerance_pixels must be in (0, 0.40]"
            )
        if not 0.0 < self.maximum_held_out_flow_fb_error_pixels <= 0.75:
            raise ValueError(
                "geometry_assisted_seam.maximum_held_out_flow_fb_error_pixels must be in (0, 0.75]"
            )
        if not 0.0 < self.maximum_local_displacement_pixels <= 8.0:
            raise ValueError(
                "geometry_assisted_seam.maximum_local_displacement_pixels must be in (0, 8]"
            )
        if not 0.0 < self.maximum_straight_line_deviation_pixels <= 1.0:
            raise ValueError(
                "geometry_assisted_seam.maximum_straight_line_deviation_pixels "
                "must be in (0, 1]"
            )
        if not 0.0 < self.actual_rgb_line_inverse_maximum_residual_pixels <= 0.05:
            raise ValueError(
                "geometry_assisted_seam.actual_rgb_line_inverse_maximum_residual_pixels "
                "must be in (0, 0.05]"
            )
        if not 0.0 < self.maximum_held_out_error_pixels <= 0.75:
            raise ValueError(
                "geometry_assisted_seam.maximum_held_out_error_pixels must be in (0, 0.75]"
            )
        if not 0.0 < self.maximum_held_out_maximum_error_pixels <= 2.0:
            raise ValueError(
                "geometry_assisted_seam.maximum_held_out_maximum_error_pixels must be in (0, 2]"
            )
        if self.maximum_held_out_error_pixels > self.maximum_held_out_maximum_error_pixels:
            raise ValueError(
                "geometry_assisted_seam held-out P95 bound cannot exceed its maximum bound"
            )
        if not 0.30 <= self.minimum_held_out_improvement_ratio <= 1.0:
            raise ValueError(
                "geometry_assisted_seam.minimum_held_out_improvement_ratio must be in [0.30, 1]"
            )

    def geometry_config(self) -> GeometryAssistConfig:
        return GeometryAssistConfig(
            absolute_depth_tolerance_mm=float(self.absolute_depth_tolerance_mm),
            relative_depth_tolerance=float(self.relative_depth_tolerance),
            depth_noise_mm=float(self.depth_noise_mm),
            mutual_pixel_tolerance=float(self.mutual_reprojection_tolerance_pixels),
            edge_guard_radius_pixels=int(self.edge_guard_radius_pixels),
        )

    def mesh_warp_config(self) -> LocalMeshWarpConfig:
        return LocalMeshWarpConfig(
            grid_spacing_pixels=int(self.mesh_cell_pixels),
            minimum_correspondences=int(self.minimum_mutual_correspondences),
            minimum_active_cells=int(self.minimum_active_mesh_cells),
            maximum_displacement_pixels=float(self.maximum_local_displacement_pixels),
            maximum_straight_line_deviation_pixels=float(
                self.maximum_straight_line_deviation_pixels
            ),
            maximum_held_out_error_pixels=float(self.maximum_held_out_error_pixels),
            maximum_held_out_maximum_error_pixels=float(
                self.maximum_held_out_maximum_error_pixels
            ),
            minimum_held_out_improvement_pixels=float(
                self.minimum_held_out_improvement_pixels
            ),
            minimum_held_out_improvement_ratio=float(
                self.minimum_held_out_improvement_ratio
            ),
        )

    def as_dict(self) -> dict[str, object]:
        """Return auditable limits, never a depth image or map."""

        return {
            "enabled": bool(self.enabled),
            "analysis_corridor_width_pixels": int(self.analysis_corridor_width_pixels),
            "trigger_edge_offset_p95_pixels": float(self.trigger_edge_offset_p95_pixels),
            "absolute_depth_tolerance_mm": float(self.absolute_depth_tolerance_mm),
            "relative_depth_tolerance": float(self.relative_depth_tolerance),
            "depth_noise_mm": float(self.depth_noise_mm),
            "mutual_reprojection_tolerance_pixels": float(
                self.mutual_reprojection_tolerance_pixels
            ),
            "edge_guard_radius_pixels": int(self.edge_guard_radius_pixels),
            "minimum_trigger_boundary_observable_pixels": int(
                self.minimum_trigger_boundary_observable_pixels
            ),
            "flow_validation_preview_scale": float(
                self.flow_validation_preview_scale
            ),
            "minimum_held_out_flow_validation_pixels": int(
                self.minimum_held_out_flow_validation_pixels
            ),
            "minimum_held_out_strong_edge_validation_pixels": int(
                self.minimum_held_out_strong_edge_validation_pixels
            ),
            "maximum_held_out_flow_fb_error_pixels": float(
                self.maximum_held_out_flow_fb_error_pixels
            ),
            "mesh_cell_pixels": int(self.mesh_cell_pixels),
            "minimum_mutual_correspondences": int(self.minimum_mutual_correspondences),
            "minimum_active_mesh_cells": int(self.minimum_active_mesh_cells),
            "mesh_boundary_identity_policy": "tile_edges_fixed_zero_and_protected_samples_pointwise_identity",
            "maximum_local_displacement_pixels": float(
                self.maximum_local_displacement_pixels
            ),
            "maximum_straight_line_deviation_pixels": float(
                self.maximum_straight_line_deviation_pixels
            ),
            "minimum_actual_rgb_line_length_pixels": float(
                self.minimum_actual_rgb_line_length_pixels
            ),
            "minimum_actual_rgb_line_support_fraction": float(
                self.minimum_actual_rgb_line_support_fraction
            ),
            "maximum_actual_rgb_line_segments": int(
                self.maximum_actual_rgb_line_segments
            ),
            "actual_rgb_line_inverse_maximum_iterations": int(
                self.actual_rgb_line_inverse_maximum_iterations
            ),
            "actual_rgb_line_inverse_maximum_residual_pixels": float(
                self.actual_rgb_line_inverse_maximum_residual_pixels
            ),
            "maximum_held_out_error_pixels": float(self.maximum_held_out_error_pixels),
            "maximum_held_out_maximum_error_pixels": float(
                self.maximum_held_out_maximum_error_pixels
            ),
            "minimum_held_out_improvement_pixels": float(
                self.minimum_held_out_improvement_pixels
            ),
            "minimum_held_out_improvement_ratio": float(
                self.minimum_held_out_improvement_ratio
            ),
        }


@dataclass(frozen=True)
class CalibratedRGBPushbroomConfig:
    """Closed safety/configuration surface for the RGB pushbroom renderer."""

    maximum_central_band_fraction: float = 0.20
    endpoint_outer_half_fov: bool = True
    # This is deliberately a *read-only* seam-search corridor, not a blend
    # width.  The two adjacent source strips may each reserve half of it as
    # calibrated support while their hard-owned cores remain unchanged.  The
    # actual common corridor is audited because the 20% central-strip limit
    # can make a requested width unavailable on sparse captures.
    seam_search_width_pixels: int = 64
    max_canvas_megapixels: float = 200.0
    max_aggregate_megapixels: float = 200.0
    max_pose_count: int = _HARD_MAX_POSES
    max_resident_frames: int = _HARD_MAX_RESIDENT_STRIPS
    minimum_valid_scale_pairs: int = 3
    scale_central_fraction: float = 0.20
    scale_low_gradient_quantile: float = 0.45
    scale_minimum_response: float = 0.10
    scale_max_relative_mad: float = 0.35
    residual_alignment: ResidualAlignmentConfig = field(
        default_factory=ResidualAlignmentConfig
    )
    geometry_assisted_seam: GeometryAssistedSeamConfig = field(
        default_factory=GeometryAssistedSeamConfig
    )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None
    ) -> "CalibratedRGBPushbroomConfig":
        supplied = {} if value is None else dict(value)
        supplied.pop("mode", None)
        alignment_value = supplied.pop("residual_alignment", None)
        geometry_value = supplied.pop("geometry_assisted_seam", None)
        allowed = {
            "maximum_central_band_fraction",
            "endpoint_outer_half_fov",
            "seam_search_width_pixels",
            "max_canvas_megapixels",
            "max_aggregate_megapixels",
            "max_pose_count",
            "max_resident_frames",
            "minimum_valid_scale_pairs",
            "scale_central_fraction",
            "scale_low_gradient_quantile",
            "scale_minimum_response",
            "scale_max_relative_mad",
        }
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise ValueError(
                "Unknown calibrated_rgb_pushbroom configuration keys: "
                + ", ".join(unknown)
            )
        try:
            result = cls(
                **supplied,
                residual_alignment=(
                    alignment_value
                    if isinstance(alignment_value, ResidualAlignmentConfig)
                    else ResidualAlignmentConfig.from_mapping(alignment_value)
                ),
                geometry_assisted_seam=(
                    geometry_value
                    if isinstance(geometry_value, GeometryAssistedSeamConfig)
                    else GeometryAssistedSeamConfig.from_mapping(geometry_value)
                ),
            )
        except TypeError as exc:
            raise ValueError("Invalid calibrated_rgb_pushbroom configuration") from exc
        result._validate()
        return result

    def _validate(self) -> None:
        finite = np.asarray(
            [
                self.maximum_central_band_fraction,
                self.max_canvas_megapixels,
                self.max_aggregate_megapixels,
                self.scale_central_fraction,
                self.scale_low_gradient_quantile,
                self.scale_minimum_response,
                self.scale_max_relative_mad,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(finite).all():
            raise ValueError("Calibrated RGB pushbroom settings must be finite")
        if not 0.0 < self.maximum_central_band_fraction <= 0.20:
            raise ValueError("maximum_central_band_fraction must be in (0, 0.20]")
        if type(self.endpoint_outer_half_fov) is not bool:
            raise ValueError("endpoint_outer_half_fov must be a boolean")
        if not 32 <= int(self.seam_search_width_pixels) <= 64:
            raise ValueError("seam_search_width_pixels must be in [32, 64]")
        if not 0.0 < self.max_canvas_megapixels <= _HARD_MAX_CANVAS_MEGAPIXELS:
            raise ValueError("max_canvas_megapixels must be in (0, 200]")
        if not 0.0 < self.max_aggregate_megapixels <= _HARD_MAX_CANVAS_MEGAPIXELS:
            raise ValueError("max_aggregate_megapixels must be in (0, 200]")
        if not 2 <= int(self.max_pose_count) <= _HARD_MAX_POSES:
            raise ValueError("max_pose_count must be in [2, 160]")
        if not 2 <= int(self.max_resident_frames) <= _HARD_MAX_RESIDENT_STRIPS:
            raise ValueError("max_resident_frames must be in [2, 5]")
        if int(self.minimum_valid_scale_pairs) < 1:
            raise ValueError("minimum_valid_scale_pairs must be positive")
        if not 0.0 < self.scale_central_fraction <= 1.0:
            raise ValueError("scale_central_fraction must be in (0, 1]")
        if not 0.0 < self.scale_low_gradient_quantile <= 1.0:
            raise ValueError("scale_low_gradient_quantile must be in (0, 1]")
        if self.scale_minimum_response < 0.0:
            raise ValueError("scale_minimum_response cannot be negative")
        if not 0.0 <= self.scale_max_relative_mad <= 1.0:
            raise ValueError("scale_max_relative_mad must be in [0, 1]")
        if not isinstance(self.residual_alignment, ResidualAlignmentConfig):
            raise ValueError("residual_alignment must be a ResidualAlignmentConfig")
        self.residual_alignment._validate()
        if self.residual_alignment.background_model != "identity":
            raise ValueError(
                "Calibrated RGB pushbroom requires identity residual alignment; "
                "verified RGB-D SE(3) is the sole global geometry"
            )
        if not isinstance(self.geometry_assisted_seam, GeometryAssistedSeamConfig):
            raise ValueError(
                "geometry_assisted_seam must be a GeometryAssistedSeamConfig"
            )
        self.geometry_assisted_seam._validate()


@dataclass(frozen=True)
class RGBMotionScaleEstimate:
    """The RGB-only convention used to convert real millimetres to strip pixels."""

    pixels_per_mm: float
    valid_pair_count: int
    candidate_pair_count: int
    relative_mad: float
    samples: tuple[dict[str, float | int | bool], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": "adjacent_rgb_motion_divided_by_real_se3_camera_displacement",
            "pixels_per_mm": self.pixels_per_mm,
            "valid_pair_count": self.valid_pair_count,
            "candidate_pair_count": self.candidate_pair_count,
            "relative_mad": self.relative_mad,
            "samples": [dict(value) for value in self.samples],
        }


@dataclass(frozen=True)
class PushbroomLayout:
    """All real-pose strip positions in one calibrated, pose-levelled canvas."""

    frame_ids: tuple[int, ...]
    source_scan_positions_mm: tuple[float, ...]
    source_centres_x: tuple[float, ...]
    owner_left_x: tuple[float, ...]
    owner_right_x: tuple[float, ...]
    support_left_x: tuple[int, ...]
    support_right_x: tuple[int, ...]
    owner_boundaries_x: tuple[float, ...]
    endpoint_outer_owner_intervals_x: tuple[tuple[float, float], ...]
    canvas_width: int
    canvas_height: int
    canvas_megapixels: float
    aggregate_megapixels: float
    maximum_source_strip_width: int
    pixels_per_mm: float
    temporal_scan_axis: tuple[float, float, float]
    level_camera_to_world_rotation: np.ndarray
    temporal_to_virtual_x_sign: float

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinate_system": {
                "canvas_x": "real camera-centre scan position scaled by local RGB motion",
                "canvas_y": "pose-levelled calibrated virtual camera row",
                "translation_unit": "mm",
                "pixel_scale": "RGB pixels per real camera-centre millimetre",
            },
            "frame_ids": list(self.frame_ids),
            "source_scan_positions_mm": list(self.source_scan_positions_mm),
            "source_centres_x": list(self.source_centres_x),
            "owner_intervals_x": [
                [left, right]
                for left, right in zip(self.owner_left_x, self.owner_right_x, strict=True)
            ],
            "source_support_intervals_x": [
                [left, right]
                for left, right in zip(
                    self.support_left_x, self.support_right_x, strict=True
                )
            ],
            "owner_boundaries_x": list(self.owner_boundaries_x),
            "endpoint_policy": (
                "outward_half_fov"
                if self.endpoint_outer_owner_intervals_x
                else "central_strip_only"
            ),
            "endpoint_outer_owner_intervals_x": [
                [left, right] for left, right in self.endpoint_outer_owner_intervals_x
            ],
            "width": self.canvas_width,
            "height": self.canvas_height,
            "canvas_megapixels": self.canvas_megapixels,
            "aggregate_megapixels": self.aggregate_megapixels,
            "maximum_source_strip_width": self.maximum_source_strip_width,
            "pixels_per_mm": self.pixels_per_mm,
            "temporal_scan_axis": list(self.temporal_scan_axis),
            "level_camera_to_world_rotation": self.level_camera_to_world_rotation.tolist(),
            "temporal_to_virtual_x_sign": self.temporal_to_virtual_x_sign,
        }


@dataclass(frozen=True)
class PushbroomContribution:
    """One narrow RGB strip kept independently of all other source images."""

    source_index: int
    frame_id: int
    x0: int
    rgb: np.ndarray
    valid_mask: np.ndarray

    @property
    def x1(self) -> int:
        return self.x0 + int(self.rgb.shape[1])


@dataclass(frozen=True)
class PreviewContribution:
    """Low-resolution RGB-only analysis strip kept outside the output path.

    ``x0`` is expressed in the common preview-canvas grid.  The two inverse
    maps are raw calibrated source-pixel coordinates, retained only in the
    temporary preview spool so RGB residual evidence can be related back to
    the real camera image.  Preview samples are never copied into the formal
    panorama.
    """

    source_index: int
    frame_id: int
    x0: int
    canvas_scale: float
    rgb: np.ndarray
    valid_mask: np.ndarray
    source_map_x: np.ndarray
    source_map_y: np.ndarray

    @property
    def x1(self) -> int:
        return self.x0 + int(self.rgb.shape[1])


@dataclass(frozen=True)
class LocalGeometryContribution:
    """One adjacent-pair geometry tile, deliberately without RGB samples.

    ``source_map_x/y`` are the same calibrated raw-colour inverse map that a
    later output remap would use.  They let the geometry stage sample aligned
    depth nearest-neighbour and translate mutually visible raw-depth matches
    back into the virtual pushbroom coordinate system.  Tiles are created and
    discarded one adjacent pair at a time; neither depth nor an RGB tile is
    retained in a delivery.
    """

    source_index: int
    frame_id: int
    x0: int
    source_map_x: np.ndarray
    source_map_y: np.ndarray
    valid_mask: np.ndarray

    @property
    def x1(self) -> int:
        return self.x0 + int(self.source_map_x.shape[1])


@dataclass(frozen=True)
class _RawVirtualLookup:
    """A deterministic native-depth-pixel to virtual-tile lookup."""

    valid_mask: np.ndarray
    virtual_x: np.ndarray
    virtual_y: np.ndarray


@dataclass(frozen=True)
class _GeometryPairPlan:
    """Compact geometry decision plus a temporary protected-mask path."""

    pair_index: int
    frame_ids: tuple[int, int]
    triggered: bool
    corridor_x: tuple[int, int] | None
    warp_source_index: int | None
    accepted: bool
    fallback: str
    protected_mask_path: Path | None
    active_mask_path: Path | None
    audit: Mapping[str, object]


@dataclass(frozen=True)
class CalibratedRGBPushbroomResult:
    panorama: np.ndarray
    metadata: dict[str, object]


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-9:
        raise RuntimeError(f"{label} is degenerate")
    return value / norm


def validate_camera_to_world(pose: np.ndarray) -> np.ndarray:
    """Validate one finite rigid camera-to-world transform without depth data."""

    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("camera_to_world must be a finite 4x4 matrix")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise ValueError("camera_to_world must have homogeneous final row [0, 0, 0, 1]")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("camera_to_world rotation must be orthonormal")
    if not np.isclose(float(np.linalg.det(rotation)), 1.0, atol=1e-5):
        raise ValueError("camera_to_world rotation must have determinant +1")
    return value.copy()


def _trajectory_axes(poses: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, float]:
    """Derive a level virtual-camera orientation from real poses only."""

    checked = [validate_camera_to_world(pose) for pose in poses]
    if len(checked) < 2:
        raise ValueError("Calibrated RGB pushbroom requires at least two poses")
    centres = np.asarray([pose[:3, 3] for pose in checked], dtype=np.float64)
    temporal = _unit(centres[-1] - centres[0], "camera-centre scan span")
    signed_steps = np.diff(centres, axis=0) @ temporal
    if not np.all(np.isfinite(signed_steps)) or np.any(signed_steps <= 1e-4):
        raise RuntimeError(
            "Calibrated RGB pushbroom requires strictly monotonic real camera centres"
        )

    down_candidates = np.asarray([pose[:3, 1] for pose in checked], dtype=np.float64)
    down = np.median(down_candidates, axis=0)
    down -= temporal * float(np.dot(down, temporal))
    down = _unit(down, "levelled camera-down direction")
    virtual_right = temporal.copy()
    virtual_forward = _unit(
        np.cross(virtual_right, down), "levelled camera-forward direction"
    )
    mean_forward = _unit(
        np.median(np.asarray([pose[:3, 2] for pose in checked]), axis=0),
        "mean camera-forward direction",
    )
    if float(np.dot(virtual_forward, mean_forward)) < 0.0:
        virtual_right *= -1.0
        virtual_forward = _unit(
            np.cross(virtual_right, down), "levelled camera-forward direction"
        )
    level_rotation = np.column_stack((virtual_right, down, virtual_forward))
    if not np.isclose(float(np.linalg.det(level_rotation)), 1.0, atol=1e-6):
        raise RuntimeError("Levelled RGB virtual camera is not right handed")
    return temporal, level_rotation, float(np.dot(temporal, virtual_right))


def _motion_row_value(value: object, name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _read_bgr(path: Path, calibration: CameraIntrinsics) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode RGB source image: {path}")
    if image.shape[:2] != (calibration.height, calibration.width):
        raise ValueError(
            f"RGB source {path} has shape {image.shape[:2]}, expected "
            f"{(calibration.height, calibration.width)}"
        )
    return image


def _phase_motion_pixels(
    first: np.ndarray, second: np.ndarray, central_fraction: float
) -> tuple[float, float]:
    """Fallback local RGB shift measurement; it never produces a pose."""

    height, width = first.shape[:2]
    fraction = float(np.clip(central_fraction, 0.05, 1.0))
    half = max(8, min(width // 2, int(round(width * fraction * 0.5))))
    centre = width // 2
    x0, x1 = max(0, centre - half), min(width, centre + half)
    y0, y1 = max(0, int(round(height * 0.08))), min(height, int(round(height * 0.92)))
    first_gray = cv2.cvtColor(first[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    second_gray = cv2.cvtColor(second[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    if first_gray.shape[0] < 8 or first_gray.shape[1] < 8:
        return 0.0, 0.0
    window = cv2.createHanningWindow(
        (first_gray.shape[1], first_gray.shape[0]), cv2.CV_32F
    )
    try:
        shift, response = cv2.phaseCorrelate(first_gray, second_gray, window)
    except cv2.error:
        return 0.0, 0.0
    return abs(float(shift[0])), float(response)


def estimate_rgb_motion_pixels_per_mm(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    config: CalibratedRGBPushbroomConfig,
    *,
    rgb_motions: Sequence[object] | None = None,
    motion_pixels_to_full_resolution: float = 1.0,
) -> RGBMotionScaleEstimate:
    """Estimate one layout-only RGB pixel scale from adjacent real SE(3) motion.

    ``rgb_motions`` may contain the pipeline's existing local thumbnail motion
    estimates.  They are used only as independent RGB measurements for a scalar
    scale; they never move, order, or create camera poses.  When unavailable,
    local phase correlation is used as a fail-closed fallback.
    """

    if len(frames) != len(poses) or len(frames) < 2:
        raise ValueError("RGB scale frames and poses must align and contain two frames")
    if not np.isfinite(float(motion_pixels_to_full_resolution)) or (
        motion_pixels_to_full_resolution <= 0.0
    ):
        raise ValueError("motion_pixels_to_full_resolution must be finite and positive")
    temporal, _, _ = _trajectory_axes(poses)
    centres = np.asarray([validate_camera_to_world(pose)[:3, 3] for pose in poses])
    displacements = np.diff(centres, axis=0) @ temporal
    samples: list[dict[str, float | int | bool]] = []
    candidates: list[float] = []
    for index, displacement in enumerate(displacements):
        pixels = 0.0
        response = 0.0
        used_thumbnail_motion = False
        reliable = True
        if rgb_motions is not None and index < len(rgb_motions):
            row = rgb_motions[index]
            raw_dx = _motion_row_value(row, "dx")
            raw_reliable = _motion_row_value(row, "reliable")
            try:
                pixels = abs(float(raw_dx)) * float(motion_pixels_to_full_resolution)
                response = 1.0
                reliable = bool(raw_reliable) if raw_reliable is not None else True
                used_thumbnail_motion = True
            except (TypeError, ValueError):
                pixels = 0.0
        if not used_thumbnail_motion:
            first = _read_bgr(frames[index].color_path, calibration)
            second = _read_bgr(frames[index + 1].color_path, calibration)
            pixels, response = _phase_motion_pixels(
                first, second, config.scale_central_fraction
            )
        accepted = bool(
            reliable
            and displacement > 1e-4
            and pixels > 0.25
            and response >= config.scale_minimum_response
            and np.isfinite(pixels)
        )
        pixels_per_mm = pixels / float(displacement) if accepted else 0.0
        samples.append(
            {
                "pair_index": index,
                "camera_displacement_mm": float(displacement),
                "rgb_motion_pixels": float(pixels),
                "pixels_per_mm": float(pixels_per_mm),
                "response": float(response),
                "used_thumbnail_motion": used_thumbnail_motion,
                "accepted": accepted,
            }
        )
        if accepted:
            candidates.append(float(pixels_per_mm))
    if len(candidates) < int(config.minimum_valid_scale_pairs):
        raise RuntimeError(
            "Calibrated RGB pushbroom has too few reliable adjacent RGB motion "
            "measurements for a real-SE(3) strip scale"
        )
    values = np.asarray(candidates, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    relative_mad = mad / max(median, 1e-9)
    if (
        not np.isfinite(median)
        or median <= 1e-6
        or relative_mad > config.scale_max_relative_mad
    ):
        raise RuntimeError(
            "Calibrated RGB pushbroom RGB-motion scale is unstable; no depth or "
            "2-D fallback is permitted"
        )
    return RGBMotionScaleEstimate(
        pixels_per_mm=median,
        valid_pair_count=len(candidates),
        candidate_pair_count=len(samples),
        relative_mad=relative_mad,
        samples=tuple(samples),
    )


def build_calibrated_rgb_pushbroom_layout(
    frame_ids: Sequence[int],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    scale: RGBMotionScaleEstimate,
    config: CalibratedRGBPushbroomConfig,
) -> PushbroomLayout:
    """Allocate contiguous real-pose owner intervals without a depth plane."""

    if len(frame_ids) != len(poses) or not 2 <= len(poses) <= config.max_pose_count:
        raise ValueError("Pushbroom layout requires 2-160 aligned real pose nodes")
    temporal, level_rotation, virtual_sign = _trajectory_axes(poses)
    centres_mm = np.asarray(
        [validate_camera_to_world(pose)[:3, 3] for pose in poses], dtype=np.float64
    )
    positions = centres_mm @ temporal
    if np.any(np.diff(positions) <= 1e-4):
        raise RuntimeError("Pushbroom camera centres are not strictly monotonic")
    centres_x_unshifted = (positions - positions[0]) * scale.pixels_per_mm
    steps = np.diff(centres_x_unshifted)
    if not np.isfinite(steps).all() or np.any(steps <= 0.25):
        raise RuntimeError("Pushbroom RGB scale produces collapsed adjacent strips")
    owner_edges = np.concatenate(
        (
            [centres_x_unshifted[0] - 0.5 * steps[0]],
            0.5 * (centres_x_unshifted[:-1] + centres_x_unshifted[1:]),
            [centres_x_unshifted[-1] + 0.5 * steps[-1]],
        )
    )
    if config.endpoint_outer_half_fov:
        # The first and last source own the scene that lies outward from the
        # scan.  Extend only those exterior intervals to the calibrated image
        # edge; every intermediate source remains a central strip.  The sign
        # accounts for a levelled virtual camera whose image-right direction
        # is opposite the chronological scan axis.
        if virtual_sign >= 0.0:
            first_outward_extent = float(calibration.cx)
            first_inward_extent = float(calibration.width - 1 - calibration.cx)
            last_inward_extent = float(calibration.cx)
            last_outward_extent = float(calibration.width - 1 - calibration.cx)
        else:
            first_outward_extent = float(calibration.width - 1 - calibration.cx)
            first_inward_extent = float(calibration.cx)
            last_inward_extent = float(calibration.width - 1 - calibration.cx)
            last_outward_extent = float(calibration.cx)
        owner_edges[0] = centres_x_unshifted[0] - first_outward_extent
        owner_edges[-1] = centres_x_unshifted[-1] + last_outward_extent
        if owner_edges[1] > centres_x_unshifted[0] + first_inward_extent:
            raise RuntimeError(
                "First endpoint cannot cover its midpoint owner interval within "
                "the calibrated RGB field of view"
            )
        if owner_edges[-2] < centres_x_unshifted[-1] - last_inward_extent:
            raise RuntimeError(
                "Last endpoint cannot cover its midpoint owner interval within "
                "the calibrated RGB field of view"
            )
    if np.any(np.diff(owner_edges) <= 1e-4):
        raise RuntimeError("Pushbroom endpoint and midpoint owner intervals collapse")
    origin = math.floor(float(owner_edges[0]))
    owner_edges -= origin
    centres_x = centres_x_unshifted - origin
    width = int(math.ceil(float(owner_edges[-1])))
    if width < 2:
        raise RuntimeError("Pushbroom RGB canvas is degenerate")
    megapixels = width * calibration.height / 1_000_000.0
    if megapixels > config.max_canvas_megapixels:
        raise MemoryError(
            f"Calibrated RGB pushbroom canvas is {megapixels:.1f} MP, above "
            f"the {config.max_canvas_megapixels:.1f} MP hard limit"
        )
    # Every intermediate raw source contributes only a genuinely narrow
    # central band.  The two scan endpoints are allowed to own their outward
    # calibrated half-FOV so the captured scene does not silently disappear at
    # either edge of the panorama.  If an intermediate hard-owned interval
    # cannot fit, fail closed and ask for denser real pose sampling or a lower
    # output scale.
    maximum_band_width = max(
        2, int(math.floor(calibration.width * config.maximum_central_band_fraction))
    )
    owner_left_int = np.ceil(owner_edges[:-1]).astype(np.int32)
    owner_right_int = np.ceil(owner_edges[1:]).astype(np.int32)
    owner_widths = owner_right_int - owner_left_int
    limited_owner_widths = (
        owner_widths[1:-1] if config.endpoint_outer_half_fov else owner_widths
    )
    if np.any(limited_owner_widths > maximum_band_width):
        raise RuntimeError(
            "An intermediate calibrated RGB pushbroom owner strip exceeds the "
            "configured narrow central-band limit; capture more real pose frames "
            "or lower output scale"
        )
    spare_width = np.maximum(0, maximum_band_width - owner_widths)
    # Keep the complete permitted central support on disk for photometric
    # statistics.  The *owner search* is selected separately below and remains
    # exactly 32--64 px wide, so this does not broaden a hard-owned strip or a
    # 2--8 px blend band.  Using the full 20% central strip for the global
    # colour solve avoids deriving a frame gain from a handful of pixels when a
    # foreground happens to cross the narrow seam corridor.
    left_padding = spare_width // 2
    right_padding = spare_width - left_padding
    # Endpoint exterior pixels are a direct calibrated copy, never a seam
    # support region.  Do not ask the inverse remap to sample beyond a physical
    # image edge merely to add seam padding.
    left_padding[0] = 0
    right_padding[-1] = 0
    support_left = owner_left_int - left_padding
    support_right = owner_right_int + right_padding
    support_left = np.clip(support_left, 0, width)
    support_right = np.clip(support_right, 0, width)
    if np.any(support_right - support_left < 2):
        raise RuntimeError("Pushbroom central strip support collapsed")
    canvas_megapixels = width * calibration.height / 1_000_000.0
    maximum_source_strip_width = int(np.max(support_right - support_left))
    aggregate_megapixels = (
        canvas_megapixels
        + 2.0 * calibration.height * maximum_source_strip_width / 1_000_000.0
    )
    if aggregate_megapixels > config.max_aggregate_megapixels:
        raise MemoryError(
            "Calibrated RGB pushbroom aggregate working set is "
            f"{aggregate_megapixels:.1f} MP, above the "
            f"{config.max_aggregate_megapixels:.1f} MP hard limit"
        )
    return PushbroomLayout(
        frame_ids=tuple(int(value) for value in frame_ids),
        source_scan_positions_mm=tuple(float(value) for value in positions),
        source_centres_x=tuple(float(value) for value in centres_x),
        owner_left_x=tuple(float(value) for value in owner_edges[:-1]),
        owner_right_x=tuple(float(value) for value in owner_edges[1:]),
        support_left_x=tuple(int(value) for value in support_left),
        support_right_x=tuple(int(value) for value in support_right),
        owner_boundaries_x=tuple(float(value) for value in owner_edges[1:-1]),
        endpoint_outer_owner_intervals_x=(
            (
                float(owner_edges[0]),
                float(centres_x[0]),
            ),
            (
                float(centres_x[-1]),
                float(owner_edges[-1]),
            ),
        )
        if config.endpoint_outer_half_fov
        else (),
        canvas_width=width,
        canvas_height=int(calibration.height),
        canvas_megapixels=float(canvas_megapixels),
        aggregate_megapixels=float(aggregate_megapixels),
        maximum_source_strip_width=maximum_source_strip_width,
        pixels_per_mm=float(scale.pixels_per_mm),
        temporal_scan_axis=tuple(float(value) for value in temporal),
        level_camera_to_world_rotation=level_rotation,
        temporal_to_virtual_x_sign=float(virtual_sign),
    )


def _as_virtual_coordinate_grids(
    canvas_x: np.ndarray, canvas_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise one-dimensional or matching two-dimensional virtual grids."""

    x = np.asarray(canvas_x, dtype=np.float64)
    y = np.asarray(canvas_y, dtype=np.float64)
    if x.ndim == 1 and y.ndim == 1:
        return (
            np.broadcast_to(x[None, :], (y.size, x.size)),
            np.broadcast_to(y[:, None], (y.size, x.size)),
        )
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Virtual inverse-map coordinates must be matching 2-D grids")
    return x, y


def _build_nominal_inverse_map(
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    camera_to_world: np.ndarray,
    source_index: int,
    canvas_x: np.ndarray,
    canvas_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a nominal levelled virtual strip into raw calibrated RGB pixels.

    This is intentionally the pre-existing calibrated mapping written as an
    independently testable constructor.  With one-dimensional coordinates its
    arithmetic and sampling grid are bit-for-bit the former ``render_frame``
    path, which lets the identity residual model prove it has not changed any
    output pixel.
    """

    if not 0 <= source_index < len(layout.frame_ids):
        raise IndexError("Pushbroom source index is out of range")
    pose = validate_camera_to_world(camera_to_world)
    x, y = _as_virtual_coordinate_grids(canvas_x, canvas_y)
    virtual_u = calibration.cx + layout.temporal_to_virtual_x_sign * (
        x - layout.source_centres_x[source_index]
    )
    ray_x = (virtual_u - calibration.cx) / calibration.fx
    ray_y = (y - calibration.cy) / calibration.fy
    rays = np.empty((*x.shape, 3), dtype=np.float64)
    rays[:, :, 0] = ray_x
    rays[:, :, 1] = ray_y
    rays[:, :, 2] = 1.0
    world_rays = rays @ layout.level_camera_to_world_rotation.T
    source_rays = world_rays @ pose[:3, :3]
    map_x, map_y, positive_z = camera_points_to_source_pixels(source_rays, calibration)
    valid = (
        positive_z
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= calibration.width - 1)
        & (map_y >= 0.0)
        & (map_y <= calibration.height - 1)
    )
    return (
        np.ascontiguousarray(map_x),
        np.ascontiguousarray(map_y),
        np.ascontiguousarray(valid),
    )


def _build_composite_inverse_map(
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    camera_to_world: np.ndarray,
    source_index: int,
    canvas_x: np.ndarray,
    canvas_y: np.ndarray,
    *,
    residual_warp: object | None,
    geometry_warp: object | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compose local geometry then residual inverse maps before ray mapping.

    Neither map is a pose and neither alters the real SE(3) supplied by the
    pose pipeline.  Geometry correction is source-local and bounded to an
    adjacent seam tile; the optional RGB residual remains source-global.  The
    no-warp path intentionally calls the nominal constructor directly so its
    floating-point arithmetic stays byte-identical to the original renderer.
    """

    residual_identity = residual_warp is None or bool(
        getattr(residual_warp, "is_identity", False)
    )
    geometry_identity = geometry_warp is None or bool(
        getattr(geometry_warp, "is_identity", False)
    )
    if residual_identity and geometry_identity:
        return _build_nominal_inverse_map(
            layout,
            calibration,
            camera_to_world,
            source_index,
            canvas_x,
            canvas_y,
        )
    x, y = _as_virtual_coordinate_grids(canvas_x, canvas_y)
    warped_x, warped_y = x, y
    for label, warp, identity in (
        ("geometry", geometry_warp, geometry_identity),
        ("residual", residual_warp, residual_identity),
    ):
        if identity:
            continue
        inverse = getattr(warp, "inverse_virtual_coordinates", None)
        if not callable(inverse):
            raise TypeError(f"{label.capitalize()} warp must provide inverse_virtual_coordinates")
        try:
            next_x, next_y = inverse(warped_x, warped_y)
        except Exception as exc:
            raise RuntimeError(
                f"{label.capitalize()} inverse warp could not map virtual coordinates"
            ) from exc
        warped_x = np.asarray(next_x, dtype=np.float64)
        warped_y = np.asarray(next_y, dtype=np.float64)
        if (
            warped_x.shape != x.shape
            or warped_y.shape != y.shape
            or not np.isfinite(warped_x).all()
            or not np.isfinite(warped_y).all()
        ):
            raise RuntimeError(
                f"{label.capitalize()} inverse warp returned invalid virtual coordinates"
            )
    return _build_nominal_inverse_map(
        layout,
        calibration,
        camera_to_world,
        source_index,
        warped_x,
        warped_y,
    )


class CalibratedRGBPushbroomRenderer:
    """Single-remap calibrated RGB strip generator plus geometry-map analysis."""

    def __init__(
        self,
        layout: PushbroomLayout,
        calibration: CameraIntrinsics,
        poses: Sequence[np.ndarray],
        residual_warps: Sequence[object | None] | None = None,
        geometry_warps: Sequence[object | None] | None = None,
    ) -> None:
        if len(poses) != len(layout.frame_ids):
            raise ValueError("Pushbroom renderer poses must match layout sources")
        if residual_warps is not None and len(residual_warps) != len(poses):
            raise ValueError("Pushbroom residual warps must match layout sources")
        if geometry_warps is not None and len(geometry_warps) != len(poses):
            raise ValueError("Pushbroom geometry warps must match layout sources")
        self.layout = layout
        self.calibration = calibration
        self.poses = tuple(validate_camera_to_world(pose) for pose in poses)
        self.residual_warps = (
            tuple(residual_warps) if residual_warps is not None else (None,) * len(poses)
        )
        self.geometry_warps = (
            tuple(geometry_warps) if geometry_warps is not None else (None,) * len(poses)
        )
        # ``remap_count`` is kept as the legacy full-resolution-output counter.
        # Preview analysis is deliberately reported separately and cannot
        # satisfy the formal one-source/one-output-remap invariant.
        self.remap_count = 0
        self.full_resolution_output_remap_count = 0
        self.analysis_preview_remap_count = 0

    def _inverse_map(
        self,
        source_index: int,
        global_x: np.ndarray,
        virtual_v: np.ndarray,
        *,
        include_geometry: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the sole calibrated inverse map for a virtual strip grid."""

        warp = self.residual_warps[source_index]
        geometry_warp = self.geometry_warps[source_index] if include_geometry else None
        return _build_composite_inverse_map(
            self.layout,
            self.calibration,
            self.poses[source_index],
            source_index,
            global_x,
            virtual_v,
            residual_warp=warp,
            geometry_warp=geometry_warp,
        )

    def render_frame(self, frame: RGBDFrame, source_index: int) -> PushbroomContribution:
        if not 0 <= source_index < len(self.poses):
            raise IndexError("Pushbroom source index is out of range")
        if int(frame.frame_id) != self.layout.frame_ids[source_index]:
            raise ValueError("Pushbroom frame order must match real-pose layout order")
        image = _read_bgr(frame.color_path, self.calibration)
        x0 = self.layout.support_left_x[source_index]
        x1 = self.layout.support_right_x[source_index]
        height = self.layout.canvas_height
        global_x = np.arange(x0, x1, dtype=np.float64)
        virtual_v = np.arange(height, dtype=np.float64)
        map_x, map_y, valid = self._inverse_map(source_index, global_x, virtual_v)
        rgb = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        self.remap_count += 1
        self.full_resolution_output_remap_count += 1
        return PushbroomContribution(
            source_index=source_index,
            frame_id=int(frame.frame_id),
            x0=x0,
            rgb=np.ascontiguousarray(rgb),
            valid_mask=np.ascontiguousarray(valid),
        )

    def render_local_geometry_map(
        self,
        frame: RGBDFrame,
        source_index: int,
        *,
        x0: int,
        x1: int,
    ) -> LocalGeometryContribution:
        """Build a narrow raw-depth lookup map without reading or remapping RGB.

        Geometry planning deliberately uses the residual map but excludes any
        geometry correction currently under consideration.  It is therefore an
        analysis operation, not a second output sampling pass.
        """

        if not 0 <= source_index < len(self.poses):
            raise IndexError("Pushbroom geometry source index is out of range")
        if int(frame.frame_id) != self.layout.frame_ids[source_index]:
            raise ValueError("Pushbroom geometry frame order must match real-pose layout order")
        start, stop = int(x0), int(x1)
        if (
            start < self.layout.support_left_x[source_index]
            or stop > self.layout.support_right_x[source_index]
            or stop <= start
        ):
            raise ValueError("Local geometry map must stay inside calibrated source support")
        map_x, map_y, valid = self._inverse_map(
            source_index,
            np.arange(start, stop, dtype=np.float64),
            np.arange(self.layout.canvas_height, dtype=np.float64),
            include_geometry=False,
        )
        return LocalGeometryContribution(
            source_index=source_index,
            frame_id=int(frame.frame_id),
            x0=start,
            source_map_x=np.ascontiguousarray(map_x),
            source_map_y=np.ascontiguousarray(map_y),
            valid_mask=np.ascontiguousarray(valid),
        )

    def render_preview_frame(
        self,
        frame: RGBDFrame,
        source_index: int,
        *,
        canvas_scale: float,
    ) -> PreviewContribution:
        """Inverse-remap one analysis-only preview in a shared canvas grid.

        The preview grid is derived from full virtual coordinates, rather than
        resizing a completed output strip.  Consequently its raw-coordinate
        inverse maps remain valid evidence for calibrated RGB/SE(3) checks.
        """

        if not 0 <= source_index < len(self.poses):
            raise IndexError("Pushbroom preview source index is out of range")
        if int(frame.frame_id) != self.layout.frame_ids[source_index]:
            raise ValueError("Pushbroom preview order must match real-pose layout order")
        scale = float(canvas_scale)
        if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
            raise ValueError("Pushbroom preview canvas scale must be in (0, 1]")
        source_x0 = self.layout.support_left_x[source_index]
        source_x1 = self.layout.support_right_x[source_index]
        x0 = int(math.floor(source_x0 * scale))
        x1 = int(math.ceil(source_x1 * scale))
        height = max(1, int(math.ceil(self.layout.canvas_height * scale)))
        preview_x = np.arange(x0, x1, dtype=np.float64)
        global_x = (preview_x + 0.5) / scale - 0.5
        preview_y = np.arange(height, dtype=np.float64)
        virtual_v = (preview_y + 0.5) / scale - 0.5
        map_x, map_y, valid = self._inverse_map(source_index, global_x, virtual_v)
        image = _read_bgr(frame.color_path, self.calibration)
        rgb = cv2.remap(
            image,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        self.analysis_preview_remap_count += 1
        return PreviewContribution(
            source_index=source_index,
            frame_id=int(frame.frame_id),
            x0=x0,
            canvas_scale=scale,
            rgb=np.ascontiguousarray(rgb),
            valid_mask=np.ascontiguousarray(valid),
            source_map_x=np.ascontiguousarray(map_x),
            source_map_y=np.ascontiguousarray(map_y),
        )


def _finite_metric(value: object) -> float | None:
    """Return one finite scalar metric without accepting booleans as numbers."""

    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


_PREVIEW_RISK_TRIGGER_POLICY = (
    "rgb_risk_at_nominal_owner_boundary_plus_or_minus_2_full_resolution_pixels"
)
_BOUNDARY_HIGH_RISK_TOPOLOGY_POLICY = (
    "3x3_closed_8_connected_raw_structural_rgb_risk_seed_component_covering_"
    "nominal_centreline_minimum_72_full_resolution_pixels_18_rows_and_26_row_span"
)
# A dilated risk sample must keep a seam out of MultiBand, but it does not
# contain enough spatial evidence to justify reading depth or fitting a local
# mesh.  Geometry triggering therefore uses only the pre-close/pre-dilation
# *structural* RGB-risk seed (edge displacement/orientation), not a Lab-only
# photometric residual.  The latter remains fully guarded by RGB ownership
# and safe-wall gain estimation, but cannot bend an otherwise straight scene.
# Express the support in full-resolution units, then project it into the
# bounded 0.50--0.75 preview.  This keeps the policy scale invariant without
# exposing another user knob.
_MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_PIXELS = 72
_MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROWS = 18
_MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROW_SPAN = 26


def _boundary_rgb_risk_audit(
    risk_mask: np.ndarray,
    common_mask: np.ndarray,
    *,
    nominal_boundary: int,
    half_width_pixels: int,
    preview_scale: float = 1.0,
) -> dict[str, object]:
    """Summarise raw RGB-risk seed topology at one nominal owner seam.

    The caller passes ``_RGBRiskDetails.structural_seed_mask`` rather than
    the enlarged risk guard.  A 3x3 close joins a genuinely interrupted edge but never
    creates the 5x5 guard used by ownership/MultiBand.  This distinction keeps
    a one-pixel colour fluctuation safely hard-owned without escalating it to
    a depth-assisted local warp request.
    """

    risk = np.asarray(risk_mask, dtype=bool)
    common = np.asarray(common_mask, dtype=bool)
    radius = int(half_width_pixels)
    scale = float(preview_scale)
    if (
        risk.shape != common.shape
        or risk.ndim != 2
        or radius < 0
        or not math.isfinite(scale)
        or not 0.0 < scale <= 1.0
    ):
        raise ValueError("Boundary RGB-risk audit inputs are malformed")
    boundary = int(np.clip(int(nominal_boundary), 0, max(0, risk.shape[1] - 1)))
    x = np.arange(risk.shape[1], dtype=np.int32)[None, :]
    local_common = common & (np.abs(x - boundary) <= radius)
    local_risk = risk & local_common
    closed_risk = cv2.morphologyEx(
        local_risk.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    ) > 0
    closed_risk &= local_common
    component_count, labels, _stats, _centres = cv2.connectedComponentsWithStats(
        closed_risk.astype(np.uint8), connectivity=8
    )
    minimum_component_pixels = max(
        1,
        int(
            math.ceil(
                _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_PIXELS * scale * scale
            )
        ),
    )
    minimum_component_rows = max(
        1,
        int(math.ceil(_MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROWS * scale)),
    )
    minimum_component_span = max(
        1,
        int(math.ceil(_MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROW_SPAN * scale)),
    )
    components: list[tuple[int, int, int, bool]] = []
    for label in range(1, component_count):
        component = labels == label
        raw_component = local_risk & component
        raw_rows = np.flatnonzero(np.any(raw_component, axis=1))
        raw_pixels = int(np.count_nonzero(raw_component))
        if raw_pixels == 0 or raw_rows.size == 0:
            # A morphological close may introduce a bridge, but it must never
            # invent a geometry-triggering component without a raw seed.
            continue
        row_span = int(raw_rows[-1] - raw_rows[0] + 1)
        covers_centreline = bool(np.any(component[:, boundary]))
        components.append(
            (raw_pixels, int(raw_rows.size), row_span, covers_centreline)
        )
    component_pixels = np.asarray([item[0] for item in components], dtype=np.int32)
    component_rows = np.asarray([item[1] for item in components], dtype=np.int32)
    component_spans = np.asarray([item[2] for item in components], dtype=np.int32)
    component_centreline = np.asarray(
        [item[3] for item in components], dtype=bool
    )
    qualifying_components = (
        (component_pixels >= minimum_component_pixels)
        & (component_rows >= minimum_component_rows)
        & (component_spans >= minimum_component_span)
        & component_centreline
    )
    qualifying_pixels = component_pixels[qualifying_components]
    qualifying_rows = component_rows[qualifying_components]
    qualifying_spans = component_spans[qualifying_components]
    return {
        "preview_risk_policy": _PREVIEW_RISK_TRIGGER_POLICY,
        "boundary_high_risk_topology_policy": _BOUNDARY_HIGH_RISK_TOPOLOGY_POLICY,
        "boundary_nominal_x": boundary,
        "boundary_half_width_pixels": radius,
        "preview_risk_preview_scale": scale,
        "boundary_risk_pixel_count": int(np.count_nonzero(local_risk)),
        "boundary_risk_row_count": int(np.count_nonzero(np.any(local_risk, axis=1))),
        "boundary_common_row_count": int(
            np.count_nonzero(np.any(local_common, axis=1))
        ),
        "common_row_count": int(np.count_nonzero(np.any(common, axis=1))),
        "boundary_risk_component_count": int(component_pixels.size),
        "boundary_centreline_touching_risk_component_count": int(
            np.count_nonzero(component_centreline)
        ),
        "boundary_largest_risk_component_pixel_count": int(
            np.max(component_pixels) if component_pixels.size else 0
        ),
        "boundary_largest_risk_component_row_count": int(
            np.max(component_rows) if component_rows.size else 0
        ),
        "boundary_largest_risk_component_row_span": int(
            np.max(component_spans) if component_spans.size else 0
        ),
        "minimum_boundary_high_risk_component_pixel_count": minimum_component_pixels,
        "minimum_boundary_high_risk_component_row_count": minimum_component_rows,
        "minimum_boundary_high_risk_component_row_span": minimum_component_span,
        "boundary_qualifying_risk_component_count": int(
            np.count_nonzero(qualifying_components)
        ),
        "boundary_largest_qualifying_risk_component_pixel_count": int(
            np.max(qualifying_pixels) if qualifying_pixels.size else 0
        ),
        "boundary_largest_qualifying_risk_component_row_count": int(
            np.max(qualifying_rows) if qualifying_rows.size else 0
        ),
        "boundary_largest_qualifying_risk_component_row_span": int(
            np.max(qualifying_spans) if qualifying_spans.size else 0
        ),
        "boundary_high_rgb_risk": bool(np.any(qualifying_components)),
    }


def _normalise_preview_pair_for_rgb_risk(
    first: np.ndarray,
    second: np.ndarray,
    common_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Remove only robust low-gradient pair exposure before risk triggering.

    The later formal renderer estimates a full gain curve from safe wall
    support.  Trigger planning happens before that curve exists; comparing raw
    preview brightness would otherwise label a uniformly exposed wall as
    foreground risk at every owner boundary.  This small, RGB-only, trimmed
    pair normalisation is analysis evidence only and never changes a source
    strip, an output pixel, or the eventual global gain solution.
    """

    image0 = np.asarray(first, dtype=np.uint8)
    image1 = np.asarray(second, dtype=np.uint8)
    common = np.asarray(common_mask, dtype=bool)
    if image0.shape != image1.shape or image0.ndim != 3 or image0.shape[2] != 3:
        raise ValueError("Preview RGB risk exposure inputs are malformed")
    if common.shape != image0.shape[:2]:
        raise ValueError("Preview RGB risk exposure support is malformed")
    if int(np.count_nonzero(common)) < 64:
        return image1, {
            "preview_risk_exposure_normalized": False,
            "preview_risk_exposure_support_pixel_count": int(np.count_nonzero(common)),
            "preview_risk_exposure_gain_min": None,
            "preview_risk_exposure_gain_max": None,
        }
    gradient0 = _gradient_magnitude(image0)
    gradient1 = _gradient_magnitude(image1)
    gradient_values = np.concatenate((gradient0[common], gradient1[common]))
    limit = float(np.percentile(gradient_values, 45.0))
    unclipped = (
        (np.min(image0, axis=2) >= 10)
        & (np.min(image1, axis=2) >= 10)
        & (np.max(image0, axis=2) <= 245)
        & (np.max(image1, axis=2) <= 245)
    )
    support = common & unclipped & (gradient0 <= limit) & (gradient1 <= limit)
    if int(np.count_nonzero(support)) < 64:
        return image1, {
            "preview_risk_exposure_normalized": False,
            "preview_risk_exposure_support_pixel_count": int(np.count_nonzero(support)),
            "preview_risk_exposure_gain_min": None,
            "preview_risk_exposure_gain_max": None,
        }
    linear0 = _srgb_to_linear_bgr(image0)
    linear1 = _srgb_to_linear_bgr(image1)
    logarithmic = np.log(np.maximum(linear0[support], 1e-4)) - np.log(
        np.maximum(linear1[support], 1e-4)
    )
    gains = np.ones(3, dtype=np.float64)
    for channel in range(3):
        samples = logarithmic[:, channel]
        median = float(np.median(samples))
        mad = float(np.median(np.abs(samples - median)))
        kept = samples[np.abs(samples - median) <= max(0.004, 3.0 * 1.4826 * mad)]
        if kept.size < 32:
            return image1, {
                "preview_risk_exposure_normalized": False,
                "preview_risk_exposure_support_pixel_count": int(np.count_nonzero(support)),
                "preview_risk_exposure_gain_min": None,
                "preview_risk_exposure_gain_max": None,
            }
        gains[channel] = math.exp(float(np.median(kept)))
    if not np.isfinite(gains).all() or np.any(gains < 0.45) or np.any(gains > 2.20):
        return image1, {
            "preview_risk_exposure_normalized": False,
            "preview_risk_exposure_support_pixel_count": int(np.count_nonzero(support)),
            "preview_risk_exposure_gain_min": None,
            "preview_risk_exposure_gain_max": None,
        }
    adjusted = _linear_to_srgb_bgr(
        linear1 * gains.reshape(1, 1, 3).astype(np.float32)
    )
    return adjusted, {
        "preview_risk_exposure_normalized": True,
        "preview_risk_exposure_support_pixel_count": int(np.count_nonzero(support)),
        "preview_risk_exposure_gain_min": float(np.min(gains)),
        "preview_risk_exposure_gain_max": float(np.max(gains)),
    }


def _preview_geometry_trigger_boundary_audit(
    first: np.ndarray,
    second: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    *,
    nominal_boundary: int,
    preview_scale: float,
) -> dict[str, object]:
    """Probe preview risk/owner topology before narrow RGB-D mesh planning.

    This is a scalar-only analysis of already-rendered previews.  It never
    adds an RGB remap or writes a preview/mask into delivery state.  Any
    owner-probe structural failure is treated as a full-height hard-cut risk,
    thereby requesting the bounded geometry analysis rather than silently
    passing an unsafe seam through as ordinary wall blending.
    """

    scale = float(preview_scale)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("Geometry trigger preview scale is invalid")
    common = np.asarray(valid0, dtype=bool) & np.asarray(valid1, dtype=bool)
    risk_second, exposure_audit = _normalise_preview_pair_for_rgb_risk(
        first, second, common
    )
    risk = _rgb_risk_details(first, risk_second, valid0, valid1)
    # This stream is explicitly rendered at the geometry 0.50--0.75 analysis
    # scale, so a true full-resolution boundary edge survives without turning
    # the entire 64 px owner search corridor into a geometry request.  The
    # later final-owner closure catches any remaining scale disagreement.
    boundary = _boundary_rgb_risk_audit(
        risk.structural_seed_mask,
        common,
        nominal_boundary=int(nominal_boundary),
        half_width_pixels=max(1, int(math.ceil(2.0 * scale))),
        preview_scale=scale,
    )
    common_rows = int(boundary["common_row_count"])
    # Remote preview risk belongs to its own source region and must not make a
    # nominal owner boundary request depth geometry.  Only a risk component
    # that actually reaches the local boundary window can make the owner probe
    # relevant; otherwise the final RGB guard keeps it out of MultiBand.
    if not bool(boundary["boundary_high_rgb_risk"]):
        no_seed = int(boundary["boundary_risk_pixel_count"]) == 0
        return {
            **boundary,
            **exposure_audit,
            "preview_hard_cut_row_count": 0,
            "preview_full_height_hard_cut": False,
            "preview_owner_probe_status": (
                "not_needed_no_boundary_rgb_risk"
                if no_seed
                else "not_needed_unqualified_boundary_rgb_risk"
            ),
            "preview_owner_probe_graphcut_used": False,
        }
    try:
        guard, _radius, _components = _risk_guard(
            risk,
            common,
            blend_width=8,
            requested_levels=3,
        )
        owner0, owner1, _cuts, graphcut_used, hard_rows, _split, _guard_hits = (
            _graphcut_monotonic_owner(
                first,
                risk_second,
                valid0,
                valid1,
                guard,
                int(nominal_boundary),
            )
        )
        common_owned_by_both = bool(
            np.any(owner0 & common) and np.any(owner1 & common)
        )
        full_height = bool(
            common_rows > 0
            and (
                int(hard_rows) >= common_rows
                or not common_owned_by_both
            )
        )
        return {
            **boundary,
            **exposure_audit,
            "preview_hard_cut_row_count": int(hard_rows),
            "preview_full_height_hard_cut": full_height,
            "preview_owner_probe_status": (
                "full_height_hard_cut"
                if full_height
                else ("graphcut" if graphcut_used else "partial_hard_cut")
            ),
            "preview_owner_probe_graphcut_used": bool(graphcut_used),
        }
    except (RuntimeError, cv2.error):
        # There is no legitimate way to reinterpret a preview topology failure
        # as a safe blend.  Trigger only the narrow local geometry inspection;
        # it may still reject to a fully audited hard owner later.
        return {
            **boundary,
            **exposure_audit,
            "preview_hard_cut_row_count": common_rows,
            "preview_full_height_hard_cut": bool(common_rows > 0),
            "preview_owner_probe_status": "no_safe_owner_channel",
            "preview_owner_probe_graphcut_used": False,
        }


def _required_nonnegative_integer(
    mapping: Mapping[str, object], key: str, *, context: str
) -> int:
    """Read one strict scalar count without treating malformed as zero."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise RuntimeError(f"{context} lacks integer {key}")
    result = int(value)
    if result < 0:
        raise RuntimeError(f"{context} has negative {key}")
    return result


def _geometry_trigger_from_preview(
    evidence: object,
    settings: GeometryAssistedSeamConfig,
    boundary_evidence: Mapping[str, object] | None,
) -> tuple[bool, dict[str, object]]:
    """Turn *boundary-local* RGB evidence into a geometry trigger.

    A global preview maximum often comes from a remote foreground object and
    must not cause a depth guard to hard-own an unrelated seam.  Geometry is
    considered only when the existing owner-boundary measurement has enough
    observable support and reports a local edge offset.  RGB flow remains a
    post-mesh held-out gate rather than a broad trigger.
    """

    metrics_value = getattr(evidence, "metrics", {})
    metrics = dict(metrics_value) if isinstance(metrics_value, Mapping) else {}
    edge_p95 = _finite_metric(metrics.get("edge_normal_step_p95_pixels"))
    edge_max = _finite_metric(metrics.get("edge_normal_step_max_pixels"))
    flow_fb_p95 = _finite_metric(metrics.get("flow_fb_error_p95_pixels"))
    boundary = (
        dict(boundary_evidence)
        if isinstance(boundary_evidence, Mapping)
        else {}
    )
    boundary_observable = _required_nonnegative_integer(
        boundary,
        "observable_pixel_count",
        context="Geometry preview boundary evidence",
    )
    if boundary.get("preview_risk_policy") != _PREVIEW_RISK_TRIGGER_POLICY:
        raise RuntimeError("Geometry preview boundary evidence lacks RGB-risk policy")
    if (
        boundary.get("boundary_high_risk_topology_policy")
        != _BOUNDARY_HIGH_RISK_TOPOLOGY_POLICY
    ):
        raise RuntimeError(
            "Geometry preview boundary evidence lacks high-risk topology policy"
        )
    risk_preview_scale = _finite_metric(boundary.get("preview_risk_preview_scale"))
    if (
        risk_preview_scale is None
        or not math.isclose(
            risk_preview_scale,
            float(settings.flow_validation_preview_scale),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "Geometry preview boundary evidence has an inconsistent risk-preview scale"
        )
    boundary_risk_pixels = _required_nonnegative_integer(
        boundary,
        "boundary_risk_pixel_count",
        context="Geometry preview boundary evidence",
    )
    boundary_risk_rows = _required_nonnegative_integer(
        boundary,
        "boundary_risk_row_count",
        context="Geometry preview boundary evidence",
    )
    boundary_common_rows = _required_nonnegative_integer(
        boundary,
        "boundary_common_row_count",
        context="Geometry preview boundary evidence",
    )
    common_rows = _required_nonnegative_integer(
        boundary,
        "common_row_count",
        context="Geometry preview boundary evidence",
    )
    risk_components = _required_nonnegative_integer(
        boundary,
        "boundary_risk_component_count",
        context="Geometry preview boundary evidence",
    )
    centreline_risk_components = _required_nonnegative_integer(
        boundary,
        "boundary_centreline_touching_risk_component_count",
        context="Geometry preview boundary evidence",
    )
    largest_risk_component = _required_nonnegative_integer(
        boundary,
        "boundary_largest_risk_component_pixel_count",
        context="Geometry preview boundary evidence",
    )
    largest_risk_rows = _required_nonnegative_integer(
        boundary,
        "boundary_largest_risk_component_row_count",
        context="Geometry preview boundary evidence",
    )
    largest_risk_span = _required_nonnegative_integer(
        boundary,
        "boundary_largest_risk_component_row_span",
        context="Geometry preview boundary evidence",
    )
    minimum_risk_component = _required_nonnegative_integer(
        boundary,
        "minimum_boundary_high_risk_component_pixel_count",
        context="Geometry preview boundary evidence",
    )
    minimum_risk_rows = _required_nonnegative_integer(
        boundary,
        "minimum_boundary_high_risk_component_row_count",
        context="Geometry preview boundary evidence",
    )
    minimum_risk_span = _required_nonnegative_integer(
        boundary,
        "minimum_boundary_high_risk_component_row_span",
        context="Geometry preview boundary evidence",
    )
    qualifying_risk_components = _required_nonnegative_integer(
        boundary,
        "boundary_qualifying_risk_component_count",
        context="Geometry preview boundary evidence",
    )
    largest_qualifying_component = _required_nonnegative_integer(
        boundary,
        "boundary_largest_qualifying_risk_component_pixel_count",
        context="Geometry preview boundary evidence",
    )
    largest_qualifying_rows = _required_nonnegative_integer(
        boundary,
        "boundary_largest_qualifying_risk_component_row_count",
        context="Geometry preview boundary evidence",
    )
    largest_qualifying_span = _required_nonnegative_integer(
        boundary,
        "boundary_largest_qualifying_risk_component_row_span",
        context="Geometry preview boundary evidence",
    )
    boundary_high_rgb_risk = boundary.get("boundary_high_rgb_risk")
    if type(boundary_high_rgb_risk) is not bool:
        raise RuntimeError(
            "Geometry preview boundary evidence lacks high-RGB-risk flag"
        )
    hard_cut_rows = _required_nonnegative_integer(
        boundary,
        "preview_hard_cut_row_count",
        context="Geometry preview boundary evidence",
    )
    preview_full_height = boundary.get("preview_full_height_hard_cut")
    if type(preview_full_height) is not bool:
        raise RuntimeError(
            "Geometry preview boundary evidence lacks full-height hard-cut flag"
        )
    probe_status = boundary.get("preview_owner_probe_status")
    if not isinstance(probe_status, str) or not probe_status:
        raise RuntimeError("Geometry preview boundary evidence lacks owner-probe status")
    probe_graphcut_used = boundary.get("preview_owner_probe_graphcut_used")
    if type(probe_graphcut_used) is not bool:
        raise RuntimeError(
            "Geometry preview boundary evidence lacks owner-probe GraphCut flag"
        )
    expected_minimum_component = max(
        1,
        int(
            math.ceil(
                _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_PIXELS
                * risk_preview_scale
                * risk_preview_scale
            )
        ),
    )
    expected_minimum_rows = max(
        1,
        int(math.ceil(_MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROWS * risk_preview_scale)),
    )
    expected_minimum_span = max(
        1,
        int(
            math.ceil(
                _MINIMUM_BOUNDARY_HIGH_RISK_RAW_SEED_ROW_SPAN * risk_preview_scale
            )
        ),
    )
    if (
        boundary_risk_rows > boundary_common_rows
        or boundary_common_rows > common_rows
        or (boundary_risk_pixels == 0) != (risk_components == 0)
        or centreline_risk_components > risk_components
        or (risk_components == 0 and (
            largest_risk_component or largest_risk_rows or largest_risk_span
        ))
        or (risk_components > 0 and (
            largest_risk_component == 0
            or largest_risk_rows == 0
            or largest_risk_span == 0
        ))
        or risk_components > boundary_risk_pixels
        or largest_risk_component > boundary_risk_pixels
        or largest_risk_rows > boundary_risk_rows
        or qualifying_risk_components > centreline_risk_components
        or minimum_risk_component != expected_minimum_component
        or minimum_risk_rows != expected_minimum_rows
        or minimum_risk_span != expected_minimum_span
        or bool(qualifying_risk_components) != boundary_high_rgb_risk
        or (qualifying_risk_components == 0 and (
            largest_qualifying_component
            or largest_qualifying_rows
            or largest_qualifying_span
        ))
        or (qualifying_risk_components > 0 and (
            largest_qualifying_component < minimum_risk_component
            or largest_qualifying_rows < minimum_risk_rows
            or largest_qualifying_span < minimum_risk_span
            or largest_qualifying_component > largest_risk_component
            or largest_qualifying_rows > largest_risk_rows
            or largest_qualifying_span > largest_risk_span
        ))
        or hard_cut_rows > common_rows
        or (preview_full_height and common_rows <= 0)
        or (preview_full_height and not boundary_high_rgb_risk)
        or (
            preview_full_height
            and hard_cut_rows < common_rows
            and probe_status
            not in {"no_safe_owner_channel", "full_height_hard_cut"}
        )
        or (
            not boundary_high_rgb_risk
            and (
                hard_cut_rows != 0
                or preview_full_height
                or probe_graphcut_used
                or probe_status
                not in {
                    "not_needed_no_boundary_rgb_risk",
                    "not_needed_unqualified_boundary_rgb_risk",
                }
            )
        )
    ):
        raise RuntimeError("Geometry preview boundary evidence has inconsistent risk rows")
    boundary_edge_p95 = _finite_metric(
        boundary.get("edge_normal_step_p95_pixels")
    )
    boundary_edge_max = _finite_metric(
        boundary.get("edge_normal_step_max_pixels")
    )
    reasons: list[str] = []
    has_boundary_support = (
        boundary_observable
        >= int(settings.minimum_trigger_boundary_observable_pixels)
    )
    if has_boundary_support:
        if (
            boundary_edge_p95 is not None
            and boundary_edge_p95 >= settings.trigger_edge_offset_p95_pixels
        ):
            reasons.append("boundary_edge_normal_step_p95")
        if boundary_edge_max is not None and boundary_edge_max > 2.0:
            reasons.append("boundary_edge_normal_step_max")
    if boundary_high_rgb_risk:
        reasons.append("boundary_high_rgb_risk")
    if preview_full_height:
        reasons.append("preview_full_height_hard_cut")
    return (
        bool(reasons),
        {
            "preview_edge_normal_step_p95_pixels": edge_p95,
            "preview_edge_normal_step_max_pixels": edge_max,
            "preview_flow_forward_backward_p95_pixels": flow_fb_p95,
            "boundary_observable_pixel_count": boundary_observable,
            "minimum_boundary_observable_pixel_count": int(
                settings.minimum_trigger_boundary_observable_pixels
            ),
            "boundary_edge_normal_step_p95_pixels": boundary_edge_p95,
            "boundary_edge_normal_step_max_pixels": boundary_edge_max,
            "preview_risk_policy": _PREVIEW_RISK_TRIGGER_POLICY,
            "boundary_high_risk_topology_policy": (
                _BOUNDARY_HIGH_RISK_TOPOLOGY_POLICY
            ),
            "boundary_risk_pixel_count": boundary_risk_pixels,
            "boundary_risk_row_count": boundary_risk_rows,
            "boundary_common_row_count": boundary_common_rows,
            "common_row_count": common_rows,
            "preview_risk_preview_scale": risk_preview_scale,
            "boundary_risk_component_count": risk_components,
            "boundary_centreline_touching_risk_component_count": (
                centreline_risk_components
            ),
            "boundary_largest_risk_component_pixel_count": largest_risk_component,
            "boundary_largest_risk_component_row_count": largest_risk_rows,
            "boundary_largest_risk_component_row_span": largest_risk_span,
            "minimum_boundary_high_risk_component_pixel_count": (
                minimum_risk_component
            ),
            "minimum_boundary_high_risk_component_row_count": minimum_risk_rows,
            "minimum_boundary_high_risk_component_row_span": minimum_risk_span,
            "boundary_qualifying_risk_component_count": qualifying_risk_components,
            "boundary_largest_qualifying_risk_component_pixel_count": (
                largest_qualifying_component
            ),
            "boundary_largest_qualifying_risk_component_row_count": (
                largest_qualifying_rows
            ),
            "boundary_largest_qualifying_risk_component_row_span": (
                largest_qualifying_span
            ),
            "boundary_high_rgb_risk": boundary_high_rgb_risk,
            "preview_hard_cut_row_count": hard_cut_rows,
            "preview_full_height_hard_cut": preview_full_height,
            "preview_owner_probe_status": probe_status,
            "preview_owner_probe_graphcut_used": probe_graphcut_used,
            "boundary_support_sufficient": has_boundary_support,
            "trigger_reasons": reasons,
        },
    )


def _select_geometry_analysis_corridor(
    layout: PushbroomLayout,
    pair_index: int,
    requested_width: int,
) -> tuple[int, int] | None:
    """Return the 96--160 px real support shared by one adjacent source pair."""

    left = max(
        int(layout.support_left_x[pair_index]),
        int(layout.support_left_x[pair_index + 1]),
    )
    right = min(
        int(layout.support_right_x[pair_index]),
        int(layout.support_right_x[pair_index + 1]),
    )
    available = right - left
    if available < int(requested_width):
        return None
    centre = int(round(float(layout.owner_boundaries_x[pair_index])))
    start = int(np.clip(centre - int(requested_width) // 2, left, right - requested_width))
    return start, start + int(requested_width)


def _raw_virtual_lookup(
    contribution: LocalGeometryContribution,
    raw_shape: tuple[int, int],
) -> _RawVirtualLookup:
    """Invert one narrow raw inverse map with deterministic collision handling."""

    height, width = raw_shape
    map_x = np.asarray(contribution.source_map_x, dtype=np.float64)
    map_y = np.asarray(contribution.source_map_y, dtype=np.float64)
    valid = np.asarray(contribution.valid_mask, dtype=bool)
    if map_x.shape != map_y.shape or map_x.shape != valid.shape or map_x.ndim != 2:
        raise RuntimeError("Local geometry inverse map is malformed")
    usable = (
        valid
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= width - 1)
        & (map_y >= 0.0)
        & (map_y <= height - 1)
    )
    raw_valid = np.zeros(raw_shape, dtype=bool)
    virtual_x = np.full(raw_shape, np.nan, dtype=np.float32)
    virtual_y = np.full(raw_shape, np.nan, dtype=np.float32)
    flat_output = np.flatnonzero(usable)
    if not flat_output.size:
        return _RawVirtualLookup(raw_valid, virtual_x, virtual_y)
    rounded_x = np.rint(map_x.reshape(-1)[flat_output]).astype(np.int64)
    rounded_y = np.rint(map_y.reshape(-1)[flat_output]).astype(np.int64)
    inside = (
        (rounded_x >= 0)
        & (rounded_x < width)
        & (rounded_y >= 0)
        & (rounded_y < height)
    )
    flat_output = flat_output[inside]
    rounded_x = rounded_x[inside]
    rounded_y = rounded_y[inside]
    if not flat_output.size:
        return _RawVirtualLookup(raw_valid, virtual_x, virtual_y)
    raw_flat = rounded_y * width + rounded_x
    map_flat_x = map_x.reshape(-1)[flat_output]
    map_flat_y = map_y.reshape(-1)[flat_output]
    distance2 = np.square(map_flat_x - rounded_x) + np.square(map_flat_y - rounded_y)
    # Each raw depth pixel owns at most one virtual representative.  The one
    # nearest its native centre wins; ties use the lower virtual flat index.
    order = np.lexsort((flat_output, distance2, raw_flat))
    ordered_raw = raw_flat[order]
    winners = np.empty(order.size, dtype=bool)
    winners[0] = True
    winners[1:] = ordered_raw[1:] != ordered_raw[:-1]
    selected = order[winners]
    selected_raw = raw_flat[selected]
    output_y, output_x = np.unravel_index(flat_output[selected], map_x.shape)
    raw_valid.reshape(-1)[selected_raw] = True
    virtual_x.reshape(-1)[selected_raw] = (
        int(contribution.x0) + output_x
    ).astype(np.float32)
    virtual_y.reshape(-1)[selected_raw] = output_y.astype(np.float32)
    return _RawVirtualLookup(raw_valid, virtual_x, virtual_y)


def _sample_raw_boolean_nearest(
    raw_mask: np.ndarray,
    contribution: LocalGeometryContribution,
) -> np.ndarray:
    """Nearest-sample a raw geometry label through a calibrated strip map."""

    raw = np.asarray(raw_mask, dtype=bool)
    map_x = np.asarray(contribution.source_map_x, dtype=np.float64)
    map_y = np.asarray(contribution.source_map_y, dtype=np.float64)
    valid = np.asarray(contribution.valid_mask, dtype=bool)
    if raw.ndim != 2:
        raise RuntimeError("Geometry protection source mask is malformed")
    result = np.zeros(map_x.shape, dtype=bool)
    usable = (
        valid
        & np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= raw.shape[1] - 1)
        & (map_y >= 0.0)
        & (map_y <= raw.shape[0] - 1)
    )
    positions = np.flatnonzero(usable)
    if not positions.size:
        return result
    x = np.rint(map_x.reshape(-1)[positions]).astype(np.int64)
    y = np.rint(map_y.reshape(-1)[positions]).astype(np.int64)
    inside = (x >= 0) & (x < raw.shape[1]) & (y >= 0) & (y < raw.shape[0])
    if np.any(inside):
        result.reshape(-1)[positions[inside]] = raw[y[inside], x[inside]]
    return result


def _raw_pixels_to_virtual_coordinates(
    raw_points_xy: np.ndarray,
    *,
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    camera_to_world: np.ndarray,
    source_index: int,
    residual_warp: object | None,
) -> np.ndarray:
    """Invert one calibrated raw RGB map analytically at sub-pixel precision.

    The previous nearest raw-pixel lookup was intentionally conservative for
    depth visibility, but its integer quantisation alone can contribute up to
    roughly 1.4 px of apparent mesh residual.  Geometry fitting instead uses
    the exact inverse of the same calibrated ray construction as the RGB
    renderer: distorted raw pixel -> undistorted source ray -> levelled
    virtual ray -> pushbroom coordinate.  The raw validity mask remains
    nearest-neighbour/depth-safe; this helper only maps coordinates and never
    interpolates a depth measurement or a colour sample.
    """

    points = np.asarray(raw_points_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("Raw geometry points must be an N x 2 array")
    if not 0 <= int(source_index) < len(layout.frame_ids):
        raise IndexError("Raw geometry source index is out of range")
    pose = validate_camera_to_world(camera_to_world)
    result = np.full(points.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(points).all(axis=1)
    if not np.any(finite):
        return result
    native = points[finite]
    distortion = tuple(float(value) for value in calibration.distortion)
    if distortion and any(value != 0.0 for value in distortion):
        normalised = cv2.undistortPoints(
            native.reshape(-1, 1, 2),
            calibration.matrix,
            np.asarray(distortion, dtype=np.float64),
        ).reshape(-1, 2)
    else:
        normalised = np.column_stack(
            (
                (native[:, 0] - calibration.cx) / calibration.fx,
                (native[:, 1] - calibration.cy) / calibration.fy,
            )
        )
    source_rays = np.column_stack(
        (normalised, np.ones(len(normalised), dtype=np.float64))
    )
    world_rays = source_rays @ pose[:3, :3].T
    virtual_rays = world_rays @ layout.level_camera_to_world_rotation
    positive_z = np.isfinite(virtual_rays).all(axis=1) & (virtual_rays[:, 2] > 1e-9)
    if not np.any(positive_z):
        return result
    virtual_u = np.full(len(native), np.nan, dtype=np.float64)
    virtual_v = np.full(len(native), np.nan, dtype=np.float64)
    virtual_u[positive_z] = (
        calibration.fx
        * virtual_rays[positive_z, 0]
        / virtual_rays[positive_z, 2]
        + calibration.cx
    )
    virtual_v[positive_z] = (
        calibration.fy
        * virtual_rays[positive_z, 1]
        / virtual_rays[positive_z, 2]
        + calibration.cy
    )
    virtual_x = (
        layout.source_centres_x[int(source_index)]
        + layout.temporal_to_virtual_x_sign * (virtual_u - calibration.cx)
    )
    if residual_warp is not None and not bool(
        getattr(residual_warp, "is_identity", False)
    ):
        forward = getattr(residual_warp, "forward_virtual_coordinates", None)
        if not callable(forward):
            raise TypeError(
                "Residual warp must provide forward_virtual_coordinates for geometry aid"
            )
        virtual_x, virtual_v = forward(virtual_x, virtual_v)
        virtual_x = np.asarray(virtual_x, dtype=np.float64)
        virtual_v = np.asarray(virtual_v, dtype=np.float64)
        if virtual_x.shape != virtual_u.shape or virtual_v.shape != virtual_u.shape:
            raise RuntimeError("Residual forward warp returned malformed geometry coordinates")
    converted = np.column_stack((virtual_x, virtual_v))
    finite_positions = np.flatnonzero(finite)
    result[finite_positions] = converted
    return result


def _virtual_geometry_correspondences(
    first_to_second: object,
    first_lookup: _RawVirtualLookup,
    second_lookup: _RawVirtualLookup,
    *,
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    first_source_index: int,
    second_source_index: int,
    first_residual_warp: object | None,
    second_residual_warp: object | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Translate guarded RGB-D matches into sub-pixel virtual map pairs."""

    source_raw, target_raw = mutually_consistent_correspondences(first_to_second)
    if not len(source_raw):
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    source_x = np.rint(source_raw[:, 0]).astype(np.int64)
    source_y = np.rint(source_raw[:, 1]).astype(np.int64)
    target_x = np.rint(target_raw[:, 0]).astype(np.int64)
    target_y = np.rint(target_raw[:, 1]).astype(np.int64)
    source_inside = (
        (source_x >= 0)
        & (source_x < first_lookup.valid_mask.shape[1])
        & (source_y >= 0)
        & (source_y < first_lookup.valid_mask.shape[0])
    )
    target_inside = (
        (target_x >= 0)
        & (target_x < second_lookup.valid_mask.shape[1])
        & (target_y >= 0)
        & (target_y < second_lookup.valid_mask.shape[0])
    )
    usable = source_inside & target_inside
    positions = np.flatnonzero(usable)
    if not positions.size:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    source_x, source_y = source_x[positions], source_y[positions]
    target_x, target_y = target_x[positions], target_y[positions]
    usable = (
        first_lookup.valid_mask[source_y, source_x]
        & second_lookup.valid_mask[target_y, target_x]
    )
    if not np.any(usable):
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    source_raw = source_raw[positions][usable]
    target_raw = target_raw[positions][usable]
    output = _raw_pixels_to_virtual_coordinates(
        source_raw,
        layout=layout,
        calibration=calibration,
        camera_to_world=first_pose,
        source_index=first_source_index,
        residual_warp=first_residual_warp,
    )
    sample = _raw_pixels_to_virtual_coordinates(
        target_raw,
        layout=layout,
        calibration=calibration,
        camera_to_world=second_pose,
        source_index=second_source_index,
        residual_warp=second_residual_warp,
    )
    finite = np.isfinite(output).all(axis=1) & np.isfinite(sample).all(axis=1)
    return output[finite], sample[finite]


def _lift_preview_boolean_mask_to_geometry_tile(
    preview_mask: np.ndarray,
    *,
    preview_origin_x: float,
    preview_scale: float,
    corridor_x: tuple[int, int],
    canvas_height: int,
) -> np.ndarray:
    """Nearest-lift one conservative preview classification into a seam tile.

    Preview evidence is deliberately analysis-only, whereas geometry masks live
    on the full virtual canvas.  A full-resolution pixel with no directly
    corresponding preview sample is always false.  Callers therefore must
    begin from a *positive* unsafe/support classification; this helper never
    invents a classification by interpolation.
    """

    mask = np.asarray(preview_mask, dtype=bool)
    x0, x1 = (int(corridor_x[0]), int(corridor_x[1]))
    height = int(canvas_height)
    origin = float(preview_origin_x)
    scale = float(preview_scale)
    if mask.ndim != 2 or not mask.size:
        raise RuntimeError("Geometry preview classification mask is empty or malformed")
    if x1 <= x0 or height <= 0:
        raise RuntimeError("Geometry preview classification tile is empty")
    if (
        not math.isfinite(origin)
        or not math.isfinite(scale)
        or not 0.0 < scale <= 1.0
    ):
        raise RuntimeError("Geometry preview classification mapping is invalid")
    preview_x = np.rint(
        (np.arange(x0, x1, dtype=np.float64) + 0.5) * scale - 0.5 - origin
    ).astype(np.int64)
    preview_y = np.rint(
        (np.arange(height, dtype=np.float64) + 0.5) * scale - 0.5
    ).astype(np.int64)
    xx, yy = np.meshgrid(preview_x, preview_y)
    inside = (
        (xx >= 0)
        & (xx < mask.shape[1])
        & (yy >= 0)
        & (yy < mask.shape[0])
    )
    result = np.zeros(inside.shape, dtype=bool)
    result[inside] = mask[yy[inside], xx[inside]]
    return np.ascontiguousarray(result)


def _lift_rgb_flow_application_and_fit_support_to_geometry_tile(
    evidence: object,
    *,
    preview_origin_x: float,
    preview_scale: float,
    corridor_x: tuple[int, int],
    canvas_height: int,
    maximum_full_resolution_fb_error_pixels: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Lift RGB-flow application and independent fit support into one tile.

    Geometry correspondences are never sufficient by themselves to enable a
    local mesh.  This converts the already-calibrated RGB preview evidence
    back into the full virtual canvas.  The first result is the safe
    application domain: every pixel has texture/epipolar-accepted,
    bidirectional flow.  The second result is its strict training subset,
    excluding the deterministic held-out partition.  A later local mesh uses
    the former for pointwise safe sampling but the latter *only* to fit its
    nodes, so RGB held-out samples remain eligible for independent validation
    instead of being forced identity merely because they were not training
    evidence.

    The nearest preview sample is conservative: a full-resolution output
    pixel without a directly corresponding preview support pixel is false,
    hence cannot receive a non-identity local inverse map.
    """

    x0, x1 = (int(corridor_x[0]), int(corridor_x[1]))
    height = int(canvas_height)
    if x1 <= x0 or height <= 0:
        raise RuntimeError("Geometry flow-consistency tile is empty")
    origin = float(preview_origin_x)
    scale = float(preview_scale)
    maximum_fb_full_resolution = float(maximum_full_resolution_fb_error_pixels)
    if (
        not math.isfinite(origin)
        or not math.isfinite(scale)
        or not 0.0 < scale <= 1.0
        or not math.isfinite(maximum_fb_full_resolution)
        or maximum_fb_full_resolution <= 0.0
    ):
        raise RuntimeError("Geometry flow-consistency mapping is invalid")
    try:
        accepted = np.asarray(getattr(evidence, "accepted_mask"), dtype=bool)
        held_out = np.asarray(getattr(evidence, "held_out_mask"), dtype=bool)
        observable = np.asarray(getattr(evidence, "observable_mask"), dtype=bool)
        uncertain = np.asarray(getattr(evidence, "flow_uncertain_mask"), dtype=bool)
        fb_error = np.asarray(
            getattr(evidence, "forward_backward_error"), dtype=np.float64
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Geometry flow preview evidence is malformed") from exc
    if (
        accepted.ndim != 2
        or held_out.shape != accepted.shape
        or observable.shape != accepted.shape
        or uncertain.shape != accepted.shape
        or fb_error.shape != accepted.shape
    ):
        raise RuntimeError("Geometry flow preview evidence has inconsistent shapes")
    # Flow is measured in this preview's coordinate system, while the formal
    # 0.75 px contract is expressed in full-resolution RGB pixels.  Convert
    # the threshold before selecting training support; otherwise a 0.75-scale
    # preview would silently admit a one-full-resolution-pixel error.
    maximum_fb_preview = maximum_fb_full_resolution * scale
    application_flow = (
        accepted
        & observable
        & ~uncertain
        & np.isfinite(fb_error)
        & (fb_error <= maximum_fb_preview)
    )
    training_flow = application_flow & ~held_out
    application = _lift_preview_boolean_mask_to_geometry_tile(
        application_flow,
        preview_origin_x=origin,
        preview_scale=scale,
        corridor_x=(x0, x1),
        canvas_height=height,
    )
    fit_support = _lift_preview_boolean_mask_to_geometry_tile(
        training_flow,
        preview_origin_x=origin,
        preview_scale=scale,
        corridor_x=(x0, x1),
        canvas_height=height,
    )
    return (
        np.ascontiguousarray(application),
        np.ascontiguousarray(fit_support),
        {
            "policy": (
                "training_only_accepted_bidirectional_rgb_flow_and_epipolar_support"
            ),
            "preview_origin_x": origin,
            "preview_scale": scale,
            "metric_unit": "full_resolution_pixels",
            "maximum_full_resolution_fb_error_pixels": maximum_fb_full_resolution,
            "maximum_preview_fb_error_pixels": maximum_fb_preview,
            "preview_training_flow_consistent_pixel_count": int(
                np.count_nonzero(training_flow)
            ),
            "preview_application_flow_consistent_pixel_count": int(
                np.count_nonzero(application_flow)
            ),
            "preview_held_out_pixel_count": int(np.count_nonzero(held_out)),
            "preview_flow_uncertain_pixel_count": int(np.count_nonzero(uncertain)),
            "full_resolution_flow_consistent_pixel_count": int(
                np.count_nonzero(fit_support)
            ),
            "full_resolution_application_flow_consistent_pixel_count": int(
                np.count_nonzero(application)
            ),
            "full_resolution_tile_pixel_count": int(application.size),
        },
    )


def _lift_training_rgb_flow_consistency_to_geometry_tile(
    evidence: object,
    *,
    preview_origin_x: float,
    preview_scale: float,
    corridor_x: tuple[int, int],
    canvas_height: int,
    maximum_full_resolution_fb_error_pixels: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return the strict training subset for backwards-compatible callers."""

    _application, fit_support, audit = (
        _lift_rgb_flow_application_and_fit_support_to_geometry_tile(
            evidence,
            preview_origin_x=preview_origin_x,
            preview_scale=preview_scale,
            corridor_x=corridor_x,
            canvas_height=canvas_height,
            maximum_full_resolution_fb_error_pixels=(
                maximum_full_resolution_fb_error_pixels
            ),
        )
    )
    return fit_support, audit


def _rgb_transparent_or_reflection_protection_from_preview(
    evidence: object,
    *,
    preview_origin_x: float,
    preview_scale: float,
    corridor_x: tuple[int, int],
    canvas_height: int,
    guard_radius_pixels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Protect RGB-visible transparent/reflective candidates from blending.

    A plausible wall depth is not evidence that a transparent extinguisher
    cover, reflective cabinet, or unreliable-depth edge belongs to the wall.
    We therefore treat visual occlusion and every *strong* RGB structure as an
    independently protected component.  In particular, a well-tracked strong
    RGB edge with no depth discontinuity is a visual/depth disagreement, not
    proof that a transparent or reflective layer may be blended with the wall.
    The mask is only an owner constraint: it neither creates colour nor makes
    a geometric correspondence.

    Low-texture safe wall interior remains eligible for the usual narrow
    exposure/MultiBand path; only an explicitly visual discontinuity grows by
    the configured 8--12 pixel guard before it reaches owner selection.
    """

    radius = int(guard_radius_pixels)
    if not 8 <= radius <= 12:
        raise RuntimeError("RGB transparency protection guard radius is invalid")
    try:
        accepted = np.asarray(getattr(evidence, "accepted_mask"), dtype=bool)
        uncertain = np.asarray(getattr(evidence, "flow_uncertain_mask"), dtype=bool)
        occluded = np.asarray(getattr(evidence, "occluded_mask"), dtype=bool)
        edge_step = np.asarray(
            getattr(evidence, "edge_normal_step_pixels"), dtype=np.float64
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("RGB transparency protection evidence is malformed") from exc
    shape = accepted.shape
    if (
        accepted.ndim != 2
        or uncertain.shape != shape
        or occluded.shape != shape
        or edge_step.shape != shape
    ):
        raise RuntimeError("RGB transparency protection evidence has inconsistent shapes")
    strong_rgb_structure = np.isfinite(edge_step)
    uncertain_or_rejected_strong_edge = strong_rgb_structure & (
        uncertain | ~accepted
    )
    # The geometry planner separately proves that a prospective mesh region
    # lies on one mutual depth surface.  If that otherwise wall-like region
    # still contains a strong RGB structure, the optical layer is ambiguous:
    # use one RGB owner rather than treating its stable flow as permission to
    # interpolate across it.  This includes the uncertain/rejected subset.
    preview_unsafe = occluded | strong_rgb_structure
    unguarded = _lift_preview_boolean_mask_to_geometry_tile(
        preview_unsafe,
        preview_origin_x=float(preview_origin_x),
        preview_scale=float(preview_scale),
        corridor_x=corridor_x,
        canvas_height=int(canvas_height),
    )
    protected = cv2.dilate(
        unguarded.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        ),
    ) > 0
    return np.ascontiguousarray(protected), {
        "policy": "rgb_occlusion_or_any_strong_rgb_structure_dilated_guard",
        "guard_radius_pixels": radius,
        "preview_occluded_pixel_count": int(np.count_nonzero(occluded)),
        "preview_strong_rgb_structure_pixel_count": int(
            np.count_nonzero(strong_rgb_structure)
        ),
        "preview_uncertain_or_rejected_strong_edge_pixel_count": int(
            np.count_nonzero(uncertain_or_rejected_strong_edge)
        ),
        "preview_unsafe_pixel_count": int(np.count_nonzero(preview_unsafe)),
        "full_resolution_unguarded_pixel_count": int(np.count_nonzero(unguarded)),
        "full_resolution_protected_pixel_count": int(np.count_nonzero(protected)),
        "full_resolution_tile_pixel_count": int(protected.size),
    }


def _mesh_active_mask(
    warp: LocalMeshInverseWarp,
    *,
    x0: int,
    x1: int,
    height: int,
) -> np.ndarray:
    """Rasterise only non-identity mesh cells for source-scope protection."""

    x, y = _as_virtual_coordinate_grids(
        np.arange(x0, x1, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    mapped_x, mapped_y = warp.inverse_virtual_coordinates(x, y)
    return np.ascontiguousarray(
        np.hypot(np.asarray(mapped_x) - x, np.asarray(mapped_y) - y) > 1e-9
    )


def _select_single_virtual_background_component(
    bilateral_mesh_safe: np.ndarray,
    *,
    corridor_x: tuple[int, int],
    nominal_boundary_x: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Keep one 4-connected safe background component across an owner seam.

    The raw-depth classifier already rejects near/ambiguous layers.  A single
    source tile can nevertheless contain separate safe wall islands divided by
    holes or an owner-protected foreground object.  The mesh receives only the
    largest component that physically spans the nominal adjacent-source
    boundary; every other island remains identity/hard-owner.
    """

    safe = np.asarray(bilateral_mesh_safe, dtype=bool)
    if safe.ndim != 2:
        raise ValueError("Virtual background component mask must be two-dimensional")
    x0, x1 = (int(corridor_x[0]), int(corridor_x[1]))
    if x1 <= x0 or safe.shape[1] != x1 - x0:
        raise ValueError("Virtual background component corridor is inconsistent")
    boundary = int(nominal_boundary_x) - x0
    if not 0 < boundary < safe.shape[1]:
        return np.zeros_like(safe), {
            "policy": "one_4_connected_bilateral_background_component_crossing_nominal_owner_boundary",
            "nominal_boundary_x": int(nominal_boundary_x),
            "depth_safe_component_count": 0,
            "boundary_crossing_component_count": 0,
            "selected_component_label": None,
            "selected_component_pixel_count": 0,
            "nonselected_depth_safe_pixel_count": int(np.count_nonzero(safe)),
            "reason": "nominal_owner_boundary_outside_geometry_corridor",
        }
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        safe.astype(np.uint8), connectivity=4
    )
    candidates: list[tuple[int, int]] = []
    for label in range(1, int(count)):
        component = labels == label
        # A component must have real support on both source sides.  Touching
        # only a guard pixel at the nominal centre cannot justify changing
        # either source's sampling coordinates.
        if not np.any(component[:, :boundary]) or not np.any(component[:, boundary:]):
            continue
        candidates.append((int(stats[label, cv2.CC_STAT_AREA]), label))
    selected_label: int | None = None
    if candidates:
        selected_label = max(candidates, key=lambda item: (item[0], -item[1]))[1]
    selected = labels == selected_label if selected_label is not None else np.zeros_like(safe)
    selected_count = int(np.count_nonzero(selected))
    return np.ascontiguousarray(selected), {
        "policy": "one_4_connected_bilateral_background_component_crossing_nominal_owner_boundary",
        "nominal_boundary_x": int(nominal_boundary_x),
        "depth_safe_component_count": int(count - 1),
        "boundary_crossing_component_count": int(len(candidates)),
        "selected_component_label": selected_label,
        "selected_component_pixel_count": selected_count,
        "nonselected_depth_safe_pixel_count": int(
            np.count_nonzero(safe & ~selected)
        ),
    }


def _plan_geometry_assisted_seams(
    *,
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    layout: PushbroomLayout,
    residual_warps: Sequence[object | None],
    preview_evidence: Sequence[object],
    preview_boundary_geometry: Sequence[Mapping[str, object]],
    preview_pair_origins: Sequence[tuple[float, float]],
    settings: GeometryAssistedSeamConfig,
    root: Path,
) -> tuple[tuple[object | None, ...], tuple[_GeometryPairPlan, ...]]:
    """Build only triggered adjacent mesh corrections and depth protection masks.

    This is intentionally a sequential, two-frame operation.  A pair's full
    aligned depths live only for the duration of its z-buffer classification;
    the temporary spool contains compact boolean virtual masks, never depth or
    colour.  Lack of same-layer confidence produces a *declared* hard-owner
    fallback.  A structurally unreadable strict depth frame remains fatal.
    """

    if len(preview_evidence) != len(frames) - 1:
        raise RuntimeError("Geometry planning requires one RGB preview evidence object per pair")
    if len(preview_boundary_geometry) != len(preview_evidence):
        raise RuntimeError(
            "Geometry planning requires one RGB owner-boundary audit per pair"
        )
    if len(preview_pair_origins) != len(preview_evidence):
        raise RuntimeError(
            "Geometry planning requires one RGB preview origin/scale per pair"
        )
    warps: list[object | None] = [None] * len(frames)
    plans: list[_GeometryPairPlan] = []
    geometry_renderer = CalibratedRGBPushbroomRenderer(
        layout,
        calibration,
        poses,
        residual_warps=residual_warps,
    )
    for pair_index, evidence in enumerate(preview_evidence):
        triggered, trigger_audit = _geometry_trigger_from_preview(
            evidence,
            settings,
            preview_boundary_geometry[pair_index],
        )
        frame_ids = (int(frames[pair_index].frame_id), int(frames[pair_index + 1].frame_id))
        base_audit: dict[str, object] = {
            **trigger_audit,
            "mesh_cell_pixels": int(settings.mesh_cell_pixels),
            "depth_tolerance": {
                "absolute_mm": float(settings.absolute_depth_tolerance_mm),
                "relative": float(settings.relative_depth_tolerance),
                "noise_mm": float(settings.depth_noise_mm),
            },
            "depth_edge_guard_radius_pixels": int(settings.edge_guard_radius_pixels),
        }
        if not settings.enabled:
            plans.append(
                _GeometryPairPlan(
                    pair_index, frame_ids, False, None, None, False, "disabled", None, None,
                    {**base_audit, "reason": "geometry_assistance_disabled"},
                )
            )
            continue
        if not triggered:
            plans.append(
                _GeometryPairPlan(
                    pair_index, frame_ids, False, None, None, False, "not_needed", None, None,
                    {**base_audit, "reason": "rgb_preview_below_geometry_trigger"},
                )
            )
            continue
        corridor = _select_geometry_analysis_corridor(
            layout, pair_index, int(settings.analysis_corridor_width_pixels)
        )
        if corridor is None:
            plans.append(
                _GeometryPairPlan(
                    pair_index, frame_ids, True, None, None, False, "hard_owner", None, None,
                    {
                        **base_audit,
                        "reason": "insufficient_calibrated_geometry_corridor",
                        "required_corridor_width_pixels": int(
                            settings.analysis_corridor_width_pixels
                        ),
                    },
                )
            )
            continue
        x0, x1 = corridor
        first_tile = geometry_renderer.render_local_geometry_map(
            frames[pair_index], pair_index, x0=x0, x1=x1
        )
        second_tile = geometry_renderer.render_local_geometry_map(
            frames[pair_index + 1], pair_index + 1, x0=x0, x1=x1
        )
        # Strict-session depth failures cannot be reclassified as a visual
        # seam fallback: the intended geometry evidence is structurally absent.
        first_depth = read_aligned_depth_mm(frames[pair_index])
        second_depth = read_aligned_depth_mm(frames[pair_index + 1])
        expected_shape = (int(calibration.height), int(calibration.width))
        if first_depth.shape != expected_shape or second_depth.shape != expected_shape:
            raise RuntimeError("Geometry assist read an aligned depth image with unexpected dimensions")
        first_lookup = _raw_virtual_lookup(first_tile, expected_shape)
        second_lookup = _raw_virtual_lookup(second_tile, expected_shape)
        geometry = analyze_adjacent_rgbd_pair(
            first_depth,
            second_depth,
            calibration,
            poses[pair_index],
            poses[pair_index + 1],
            first_valid_mask=first_lookup.valid_mask,
            second_valid_mask=second_lookup.valid_mask,
            first_analysis_mask=first_lookup.valid_mask,
            second_analysis_mask=second_lookup.valid_mask,
            analysis_scope="adjacent_seam_raw_footprint",
            config=settings.geometry_config(),
        )
        first_depth_protected = _sample_raw_boolean_nearest(
            geometry.first_to_second.protected_mask, first_tile
        ) | _sample_raw_boolean_nearest(
            geometry.first_surface_safety.hard_owner_mask, first_tile
        )
        second_depth_protected = _sample_raw_boolean_nearest(
            geometry.second_to_first.protected_mask, second_tile
        ) | _sample_raw_boolean_nearest(
            geometry.second_surface_safety.hard_owner_mask, second_tile
        )
        depth_protected = first_depth_protected | second_depth_protected
        # `protected_mask` already contains this condition, but keep the
        # bilateral layer predicate explicit at the renderer boundary.  The
        # source/sampling maps address different raw pixels for the two real
        # cameras, so both directions must survive their own z-buffer,
        # depth-consistency and round-trip checks before a virtual pixel is
        # eligible for any non-identity local inverse sampling coordinate.
        first_mutual = _sample_raw_boolean_nearest(
            geometry.first_to_second.mutual_consistent, first_tile
        )
        second_mutual = _sample_raw_boolean_nearest(
            geometry.second_to_first.mutual_consistent, second_tile
        )
        bilateral_mutual = first_mutual & second_mutual
        first_mesh_safe = _sample_raw_boolean_nearest(
            geometry.first_surface_safety.mesh_safe_mask, first_tile
        )
        second_mesh_safe = _sample_raw_boolean_nearest(
            geometry.second_surface_safety.mesh_safe_mask, second_tile
        )
        bilateral_mesh_safe = first_mesh_safe & second_mesh_safe
        preview_origin_x, preview_scale = preview_pair_origins[pair_index]
        rgb_transparency_protected, rgb_transparency_audit = (
            _rgb_transparent_or_reflection_protection_from_preview(
                evidence,
                preview_origin_x=float(preview_origin_x),
                preview_scale=float(preview_scale),
                corridor_x=(x0, x1),
                canvas_height=layout.canvas_height,
                guard_radius_pixels=int(settings.edge_guard_radius_pixels),
            )
        )
        depth_same_layer_before_rgb_protection = (
            np.asarray(first_tile.valid_mask, dtype=bool)
            & np.asarray(second_tile.valid_mask, dtype=bool)
            & bilateral_mutual
            & bilateral_mesh_safe
            & ~depth_protected
        )
        # Do this before component selection.  A visually strong diagonal or
        # reflective layer must split the virtual background rather than
        # allowing a mesh's regularisation to bridge across it.
        candidate_depth_same_layer = (
            depth_same_layer_before_rgb_protection & ~rgb_transparency_protected
        )
        selected_background, component_selection = (
            _select_single_virtual_background_component(
                candidate_depth_same_layer,
                corridor_x=(x0, x1),
                nominal_boundary_x=int(layout.owner_boundaries_x[pair_index]),
            )
        )
        # A safe-but-separate component is not a second mesh layer.  Treat it
        # as an explicit owner-only region rather than letting GraphCut/MultiBand
        # reinterpret it as ordinary wall overlap.
        nonselected_depth_safe = candidate_depth_same_layer & ~selected_background
        # The RGB and depth classifications are independent fail-closed owner
        # constraints.  A visually unreliable layer must not become mesh
        # support merely because its depth happens to agree with the wall.
        protected = (
            depth_protected | nonselected_depth_safe | rgb_transparency_protected
        )
        depth_same_layer = selected_background & ~protected
        flow_application, flow_fit_support, flow_fit_audit = (
            _lift_rgb_flow_application_and_fit_support_to_geometry_tile(
                evidence,
                preview_origin_x=float(preview_origin_x),
                preview_scale=float(preview_scale),
                corridor_x=(x0, x1),
                canvas_height=layout.canvas_height,
                maximum_full_resolution_fb_error_pixels=float(
                    settings.maximum_held_out_flow_fb_error_pixels
                ),
            )
        )
        # A local field can only *apply* on one bilateral safe depth layer
        # with RGB flow support.  Its node fit is further restricted to the
        # independent training subset.  Do not make RGB held-out pixels
        # identity merely because they were withheld from fitting: they remain
        # safe application samples for the later independent validation pass.
        same_layer = depth_same_layer & flow_application
        fit_support = same_layer & flow_fit_support
        output_points, sample_points = _virtual_geometry_correspondences(
            geometry.first_to_second,
            first_lookup,
            second_lookup,
            layout=layout,
            calibration=calibration,
            first_pose=poses[pair_index],
            second_pose=poses[pair_index + 1],
            first_source_index=pair_index,
            second_source_index=pair_index + 1,
            first_residual_warp=residual_warps[pair_index],
            second_residual_warp=residual_warps[pair_index + 1],
        )
        fit = fit_local_mesh_inverse_warp(
            output_points,
            sample_points,
            TileBounds(float(x0), 0.0, float(x1 - 1), float(layout.canvas_height - 1)),
            same_layer_mask=same_layer,
            same_layer_origin_xy=(float(x0), 0.0),
            fit_support_mask=fit_support,
            fit_support_origin_xy=(float(x0), 0.0),
            config=settings.mesh_warp_config(),
        )
        active = (
            _mesh_active_mask(
                fit.warp, x0=x0, x1=x1, height=layout.canvas_height
            )
            if fit.warp is not None
            else np.zeros_like(same_layer)
        )
        mesh_accepted = fit.warp is not None and bool(fit.audit.accepted)
        protected_active_overlap = int(np.count_nonzero(active & protected))
        if protected_active_overlap:
            raise RuntimeError(
                "A local mesh attempted to move an owner-protected depth layer"
            )
        # ``active`` is the actual non-identity sampling footprint, not merely
        # a grid-cell bounding box.  Keep an explicit scalar proof that the
        # mesh evaluator did not move a pixel outside the final bilateral
        # depth/RGB same-layer mask even when its surrounding cell was fitted.
        active_non_same_layer_overlap = int(np.count_nonzero(active & ~same_layer))
        if active_non_same_layer_overlap:
            raise RuntimeError(
                "A local mesh attempted to move a non-same-layer pixel"
            )
        flow_validation: dict[str, object]
        if mesh_accepted:
            assert fit.warp is not None
            flow_validation = _validate_geometry_candidate_rgb_flow(
                frames=frames,
                poses=poses,
                calibration=calibration,
                layout=layout,
                residual_warps=residual_warps,
                pair_index=pair_index,
                mesh_warp=fit.warp,
                corridor_x=(x0, x1),
                active_mask=active,
                settings=settings,
            )
        else:
            flow_validation = {
                "accepted": False,
                "reason": "mesh_held_out_gate_rejected",
                "analysis_preview_remap_count": 0,
            }
        accepted = bool(mesh_accepted and flow_validation.get("accepted") is True)
        # A candidate that fails RGB held-out flow remains a protected
        # hard-owner corridor.  Do not leave its non-identity field resident
        # for a later pair or accidentally report it as active.
        accepted_active = active if accepted else np.zeros_like(active)
        protected_path = root / f"geometry-protected-{pair_index:04d}.npz"
        np.savez_compressed(
            protected_path,
            x0=np.asarray(x0, dtype=np.int32),
            protected=protected.astype(np.uint8),
            active=accepted_active.astype(np.uint8),
        )
        if accepted:
            # Exactly one source-local field is selected for each pair: the
            # second source samples toward the first.  The mesh itself rejects
            # any active-cell overlap with protected layers and remains
            # identity outside the pair tile.
            if warps[pair_index + 1] is not None:
                raise RuntimeError("Overlapping local geometry mesh for one source is forbidden")
            warps[pair_index + 1] = fit.warp
        plans.append(
            _GeometryPairPlan(
                pair_index=pair_index,
                frame_ids=frame_ids,
                triggered=True,
                corridor_x=(x0, x1),
                warp_source_index=(pair_index + 1 if accepted else None),
                accepted=accepted,
                fallback=("none" if accepted else "hard_owner"),
                protected_mask_path=protected_path,
                active_mask_path=protected_path,
                audit={
                    **base_audit,
                    "reason": (
                        "accepted"
                        if accepted
                        else (
                            str(flow_validation.get("reason"))
                            if mesh_accepted
                            else fit.audit.reason
                        )
                    ),
                    "geometry": geometry.audit.as_dict(),
                    "depth_protected_pixel_count": int(
                        np.count_nonzero(depth_protected)
                    ),
                    "first_direction_depth_protected_pixel_count": int(
                        np.count_nonzero(first_depth_protected)
                    ),
                    "second_direction_depth_protected_pixel_count": int(
                        np.count_nonzero(second_depth_protected)
                    ),
                    "first_direction_mutual_depth_pixel_count": int(
                        np.count_nonzero(first_mutual)
                    ),
                    "second_direction_mutual_depth_pixel_count": int(
                        np.count_nonzero(second_mutual)
                    ),
                    "bilateral_mutual_depth_pixel_count": int(
                        np.count_nonzero(bilateral_mutual)
                    ),
                    "first_direction_mesh_safe_depth_pixel_count": int(
                        np.count_nonzero(first_mesh_safe)
                    ),
                    "second_direction_mesh_safe_depth_pixel_count": int(
                        np.count_nonzero(second_mesh_safe)
                    ),
                    "bilateral_mesh_safe_depth_pixel_count": int(
                        np.count_nonzero(bilateral_mesh_safe)
                    ),
                    "candidate_depth_same_layer_pixel_count": int(
                        np.count_nonzero(candidate_depth_same_layer)
                    ),
                    "depth_same_layer_before_rgb_protection_pixel_count": int(
                        np.count_nonzero(depth_same_layer_before_rgb_protection)
                    ),
                    "virtual_background_component": component_selection,
                    "nonselected_depth_safe_pixel_count": int(
                        np.count_nonzero(nonselected_depth_safe)
                    ),
                    "rgb_transparent_or_reflection_protected_pixel_count": int(
                        np.count_nonzero(rgb_transparency_protected)
                    ),
                    "rgb_transparency_protection": rgb_transparency_audit,
                    "depth_same_layer_pixel_count": int(
                        np.count_nonzero(depth_same_layer)
                    ),
                    "rgb_flow_application_pixel_count": int(
                        np.count_nonzero(flow_application)
                    ),
                    "rgb_flow_fit_excluded_same_layer_pixel_count": int(
                        np.count_nonzero(same_layer & ~fit_support)
                    ),
                    "rgb_flow_fit_support": flow_fit_audit,
                    "same_layer_pixel_count": int(np.count_nonzero(same_layer)),
                    "mesh_fit_support_pixel_count": int(
                        np.count_nonzero(fit_support)
                    ),
                    "protected_pixel_count": int(np.count_nonzero(protected)),
                    "mesh_candidate_pixel_count": int(np.count_nonzero(active)),
                    "mesh_active_pixel_count": int(np.count_nonzero(accepted_active)),
                    "protected_active_overlap_pixel_count": protected_active_overlap,
                    "active_non_same_layer_overlap_pixel_count": (
                        active_non_same_layer_overlap
                    ),
                    "mesh": fit.audit.as_dict(),
                    "rgb_flow_validation": flow_validation,
                },
            )
        )
    return tuple(warps), tuple(plans)


def _geometry_pair_masks(
    plan: _GeometryPairPlan,
    *,
    left: int,
    right: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one temporary pair mask into an arbitrary overlapping strip span."""

    shape = (int(height), max(0, int(right) - int(left)))
    protected_result = np.zeros(shape, dtype=bool)
    active_result = np.zeros(shape, dtype=bool)
    if plan.protected_mask_path is None or not plan.protected_mask_path.exists():
        return protected_result, active_result
    with np.load(plan.protected_mask_path, allow_pickle=False) as stored:
        stored_x0 = int(stored["x0"])
        protected = np.asarray(stored["protected"], dtype=bool)
        active = np.asarray(stored["active"], dtype=bool)
    if (
        protected.shape != active.shape
        or protected.ndim != 2
        or protected.shape[0] != int(height)
    ):
        raise RuntimeError("Temporary geometry protection mask is malformed")
    stored_x1 = stored_x0 + protected.shape[1]
    common_left, common_right = max(int(left), stored_x0), min(int(right), stored_x1)
    if common_right <= common_left:
        return protected_result, active_result
    destination = slice(common_left - int(left), common_right - int(left))
    source = slice(common_left - stored_x0, common_right - stored_x0)
    protected_result[:, destination] = protected[:, source]
    active_result[:, destination] = active[:, source]
    return protected_result, active_result


def _geometry_out_of_scope_first_mask(
    plans: Sequence[_GeometryPairPlan],
    pair_index: int,
    *,
    left: int,
    right: int,
    height: int,
) -> np.ndarray:
    """Exclude a prior pair's warped source from its next pair's seam.

    A mesh is fitted only to make source ``i + 1`` agree with source ``i`` in
    pair ``i``.  That same physical RGB strip can be the *first* source in
    pair ``i + 1``.  It must not silently carry the earlier correction into a
    different ownership decision.  The corrected source is therefore treated
    as unavailable exactly where its prior mesh is non-identity; the next
    adjacent source must own that pixel or the normal hard-owner gate fails.
    """

    shape = (int(height), max(0, int(right) - int(left)))
    if pair_index <= 0:
        return np.zeros(shape, dtype=bool)
    prior = plans[pair_index - 1]
    if not prior.accepted or prior.warp_source_index != pair_index:
        return np.zeros(shape, dtype=bool)
    _protected, active = _geometry_pair_masks(
        prior,
        left=left,
        right=right,
        height=height,
    )
    return active


def _pair_level_hard_owner_options(
    valid_first: np.ndarray, valid_second: np.ndarray
) -> tuple[int, ...]:
    """Return every fully covering source for a geometry-topology fallback.

    This is deliberately more conservative than an arbitrary per-pixel
    fallback: if the protected geometry components cannot admit one monotonic
    GraphCut boundary, the whole local corridor comes from *one* source RGB
    strip.  If neither strip completely covers the corridor, no synthetic
    coverage is permitted and the caller must fail.  Keep both valid choices
    when available: adjacent fallback corridors share a source, and resolving
    them coherently avoids needlessly dropping that source's only strip.
    """

    first = np.asarray(valid_first, dtype=bool)
    second = np.asarray(valid_second, dtype=bool)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Pair-level hard-owner validity masks must match")
    required = first | second
    first_covers = bool(np.all(~required | first))
    second_covers = bool(np.all(~required | second))
    return tuple(
        source_index
        for source_index, covers in ((0, first_covers), (1, second_covers))
        if covers
    )


def _select_pair_level_hard_owner(
    valid_first: np.ndarray, valid_second: np.ndarray
) -> int | None:
    """Return the first covering fallback source for small standalone callers."""

    options = _pair_level_hard_owner_options(valid_first, valid_second)
    return options[0] if options else None


def _pair_level_hard_owner_masks(
    valid_first: np.ndarray,
    valid_second: np.ndarray,
    owner: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a fully-covered, unblended adjacent-pair owner decision."""

    first = np.asarray(valid_first, dtype=bool)
    second = np.asarray(valid_second, dtype=bool)
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Pair-level hard-owner validity masks must match")
    required = first | second
    if owner == 0:
        if np.any(required & ~first):
            raise RuntimeError("Pair-level first-source owner lost valid coverage")
        return (
            required,
            np.zeros_like(required),
            np.full(required.shape[0], first.shape[1] - 1, dtype=np.int32),
        )
    if owner == 1:
        if np.any(required & ~second):
            raise RuntimeError("Pair-level second-source owner lost valid coverage")
        return (
            np.zeros_like(required),
            required,
            np.full(required.shape[0], -1, dtype=np.int32),
        )
    raise ValueError("Pair-level hard owner must be source zero or one")


def _audit_suppressed_source_frames(
    frame_ids: Sequence[int],
    owner_pixel_counts: Sequence[int],
    hard_owner_topology_pairs: set[int],
) -> list[dict[str, object]]:
    """Permit a zero-colour source only beside an audited hard-owner decision."""

    if len(frame_ids) != len(owner_pixel_counts):
        raise ValueError("Frame IDs and source owner counts must have equal length")
    suppressed: list[dict[str, object]] = []
    for source_index, (frame_id, pixel_count) in enumerate(
        zip(frame_ids, owner_pixel_counts, strict=True)
    ):
        if int(pixel_count) != 0:
            continue
        adjacent_pairs = tuple(
            pair_index
            for pair_index in (source_index - 1, source_index)
            if pair_index in hard_owner_topology_pairs
        )
        if not adjacent_pairs:
            raise RuntimeError(
                "Calibrated RGB pushbroom crop removed all owned pixels from "
                f"a source without an audited geometry seam decision: {int(frame_id)}"
            )
        suppressed.append(
            {
                "source_index": int(source_index),
                "frame_id": int(frame_id),
                "adjacent_hard_owner_topology_pairs": list(adjacent_pairs),
                "reason": "fully_covered_by_audited_hard_owner_topology_decision",
            }
        )
    return suppressed


def _resolve_pair_level_hard_owners(
    pair_indices: Sequence[int],
    options_by_pair: Mapping[int, Sequence[int]],
) -> dict[int, int]:
    """Choose bounded hard owners while retaining shared-source participation.

    A run of neighbouring hard-owner fallbacks shares a camera strip between
    every two corridors.  When both RGB sources cover a corridor, select its
    second source inside a run so that the shared source remains represented
    in the next corridor.  For a singleton, prefer the first source, whose
    preceding ordinary corridor may have been eliminated by the same topology
    conflict.  Forced one-source choices remain untouched.
    """

    selected = sorted({int(index) for index in pair_indices})
    resolved: dict[int, int] = {}
    run_start = 0
    while run_start < len(selected):
        run_end = run_start + 1
        while (
            run_end < len(selected)
            and selected[run_end] == selected[run_end - 1] + 1
        ):
            run_end += 1
        run = selected[run_start:run_end]
        preferred = 1 if len(run) > 1 else 0
        for pair_index in run:
            options = tuple(int(value) for value in options_by_pair[pair_index])
            if not options:
                raise RuntimeError(
                    "Pair-level hard-owner fallback has no fully covering RGB source"
                )
            resolved[pair_index] = preferred if preferred in options else options[0]
        run_start = run_end
    return resolved


def _component_owner_assignment_is_monotonic(
    fragments: Sequence[ProtectedComponentFragment],
    owners: Mapping[int, int],
) -> bool:
    """Check one exact hard-owner decision without relaxing component masks."""

    lower_by_row: dict[int, int] = {}
    upper_by_row: dict[int, int] = {}
    for fragment in fragments:
        label = fragment.component_label
        if label is None or label not in owners:
            return False
        owner = int(owners[label])
        if owner not in fragment.allowed_owners:
            return False
        x0, y0, _, _ = fragment.global_bbox
        mask = np.asarray(fragment.local_mask, dtype=bool)
        for local_y in np.flatnonzero(np.any(mask, axis=1)):
            local_x = np.flatnonzero(mask[int(local_y)])
            if not local_x.size:
                continue
            row = int(y0 + int(local_y))
            if owner == 0:
                lower_by_row[row] = max(
                    lower_by_row.get(row, -1),
                    int(x0 + int(local_x[-1])),
                )
            else:
                upper_by_row[row] = min(
                    upper_by_row.get(row, int(x0 + mask.shape[1]) - 1),
                    int(x0 + int(local_x[0]) - 1),
                )
    return all(
        lower <= upper_by_row.get(row, lower)
        for row, lower in lower_by_row.items()
    )


def _force_monotonic_component_owners(
    fragments: Sequence[ProtectedComponentFragment],
) -> tuple[tuple[ProtectedComponentFragment, ...], dict[int, int]] | None:
    """Resolve a failed geometry neighbourhood at component, not pair, scope.

    Every protected component remains a single RGB owner.  We search only
    the owners that fully cover each component and retain the first monotonic
    assignment with the fewest deviations from the nominal owner decisions.
    This preserves ordinary safe pixels in the corridor; a whole-pair owner
    is reserved for the rare case where no component-level assignment exists.
    """

    checked = tuple(fragments)
    if not checked:
        return (), {}
    labels = [fragment.component_label for fragment in checked]
    if any(label is None for label in labels):
        return None
    numeric_labels = tuple(int(label) for label in labels if label is not None)
    if len(set(numeric_labels)) != len(numeric_labels):
        return None

    candidates: list[dict[int, int]] = []
    preferred = {
        int(fragment.component_label): int(
            fragment.preferred_owner
            if fragment.preferred_owner in fragment.allowed_owners
            else fragment.allowed_owners[0]
        )
        for fragment in checked
        if fragment.component_label is not None
    }
    candidates.append(preferred)

    # A left-to-right threshold covers the common case with many components
    # without exponential work.  It is exact because each candidate is still
    # checked against the full per-row component masks below.
    centres = [
        (
            int(fragment.component_label),
            float(fragment.global_bbox[0]) + 0.5 * float(fragment.global_bbox[2] - 1),
            fragment,
        )
        for fragment in checked
        if fragment.component_label is not None
    ]
    thresholds = [-math.inf] + sorted({centre for _, centre, _ in centres}) + [math.inf]
    for threshold in thresholds:
        assignment: dict[int, int] = {}
        for label, centre, fragment in centres:
            desired = 0 if centre <= threshold else 1
            if desired not in fragment.allowed_owners:
                break
            assignment[label] = desired
        else:
            candidates.append(assignment)

    # Geometry guards are deliberately narrow.  For the few genuinely
    # overlapping components that a simple threshold cannot resolve, search
    # all legal owner choices with a fixed bound instead of widening a seam.
    if len(checked) <= 12:
        option_rows = [tuple(int(owner) for owner in fragment.allowed_owners) for fragment in checked]
        for selected in product(*option_rows):
            candidates.append(
                {
                    int(fragment.component_label): int(owner)
                    for fragment, owner in zip(checked, selected, strict=True)
                    if fragment.component_label is not None
                }
            )

    accepted = [
        assignment
        for assignment in candidates
        if _component_owner_assignment_is_monotonic(checked, assignment)
    ]
    if not accepted:
        return None
    selected = min(
        accepted,
        key=lambda assignment: (
            # Prefer a real inter-source seam whenever the component geometry
            # permits one; collapsing all components to one strip is the
            # later, explicitly audited pair-level fallback.
            0 if len({assignment[label] for label in numeric_labels}) == 2 else 1,
            sum(assignment[label] != preferred[label] for label in numeric_labels),
            tuple(assignment[label] for label in numeric_labels),
        ),
    )
    forced = tuple(
        replace(
            fragment,
            allowed_owners=(selected[int(fragment.component_label)],),
            preferred_owner=selected[int(fragment.component_label)],
        )
        for fragment in checked
        if fragment.component_label is not None
    )
    return forced, selected


def _preflight_failure_pair_index(reason: object) -> int | None:
    """Return a local geometry pair index only for a known topology failure."""

    prefix = "pair_local_component_owners_do_not_admit_a_monotonic_seam:"
    text = str(reason)
    if not text.startswith(prefix):
        return None
    try:
        value = int(text.removeprefix(prefix))
    except ValueError:
        return None
    return value if value >= 0 else None


def _sample_geometry_active_in_preview(
    active_mask: np.ndarray,
    *,
    corridor_x0: int,
    preview_left: int,
    preview_right: int,
    preview_height: int,
    canvas_scale: float,
) -> np.ndarray:
    """Nearest-sample a full-res active mesh mask onto a preview lattice."""

    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 2 or not active.size:
        raise RuntimeError("Geometry active mesh mask is malformed")
    if preview_right <= preview_left or preview_height <= 0:
        raise RuntimeError("Geometry flow preview is empty")
    scale = float(canvas_scale)
    if not np.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise RuntimeError("Geometry flow preview scale is invalid")
    preview_x = np.arange(preview_left, preview_right, dtype=np.float64)
    preview_y = np.arange(preview_height, dtype=np.float64)
    virtual_x = (preview_x + 0.5) / scale - 0.5
    virtual_y = (preview_y + 0.5) / scale - 0.5
    xx, yy = np.meshgrid(virtual_x, virtual_y)
    ix = np.rint(xx).astype(np.int64) - int(corridor_x0)
    iy = np.rint(yy).astype(np.int64)
    inside = (
        (ix >= 0)
        & (ix < active.shape[1])
        & (iy >= 0)
        & (iy < active.shape[0])
    )
    result = np.zeros(inside.shape, dtype=bool)
    result[inside] = active[iy[inside], ix[inside]]
    return result


def _normalise_hough_segment(
    segment: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """Return a deterministic endpoint order plus compact segment geometry."""

    values = np.asarray(segment, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Hough segment must contain four finite coordinates")
    x0, y0, x1, y1 = (float(value) for value in values)
    if (x1, y1) < (x0, y0):
        x0, y0, x1, y1 = x1, y1, x0, y0
    length = float(math.hypot(x1 - x0, y1 - y0))
    return x0, y0, x1, y1, length, (x0 + x1) * 0.5, (y0 + y1) * 0.5


def _deduplicate_hough_segments(
    segments: np.ndarray,
) -> tuple[tuple[float, float, float, float, float, float, float], ...]:
    """Keep stable distinct Hough lines without silently truncating support."""

    normalised = [_normalise_hough_segment(item) for item in np.asarray(segments)]
    ordered = sorted(
        normalised,
        key=lambda item: (-item[4], item[5], item[6], item[0], item[1], item[2], item[3]),
    )
    unique: list[tuple[float, float, float, float, float, float, float]] = []
    for candidate in ordered:
        candidate_angle = float(
            math.atan2(candidate[3] - candidate[1], candidate[2] - candidate[0])
            % math.pi
        )
        candidate_midpoint = np.asarray(
            ((candidate[0] + candidate[2]) * 0.5, (candidate[1] + candidate[3]) * 0.5),
            dtype=np.float64,
        )
        duplicate = False
        for prior in unique:
            prior_angle = float(
                math.atan2(prior[3] - prior[1], prior[2] - prior[0]) % math.pi
            )
            angle_delta = abs((candidate_angle - prior_angle + math.pi / 2.0) % math.pi - math.pi / 2.0)
            prior_midpoint = np.asarray(
                ((prior[0] + prior[2]) * 0.5, (prior[1] + prior[3]) * 0.5),
                dtype=np.float64,
            )
            if (
                angle_delta <= math.radians(3.0)
                and np.linalg.norm(candidate_midpoint - prior_midpoint) <= 3.0
                and abs(candidate[4] - prior[4]) <= 4.0
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return tuple(unique)


def _sample_segment_points(
    segment: tuple[float, float, float, float, float, float, float],
) -> np.ndarray:
    """Sample a preview Hough segment at no more than one pixel spacing."""

    x0, y0, x1, y1, length, _mid_x, _mid_y = segment
    count = max(2, int(math.ceil(length)) + 1)
    return np.column_stack(
        (
            np.linspace(x0, x1, count, dtype=np.float64),
            np.linspace(y0, y1, count, dtype=np.float64),
        )
    )


def _contiguous_valid_runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive valid sample runs for one line without joining holes."""

    values = np.asarray(mask, dtype=bool).reshape(-1)
    runs: list[tuple[int, int]] = []
    begin: int | None = None
    for index, enabled in enumerate(values):
        if enabled and begin is None:
            begin = index
        elif not enabled and begin is not None:
            runs.append((begin, index - 1))
            begin = None
    if begin is not None:
        runs.append((begin, len(values) - 1))
    return tuple(runs)


def _actual_rgb_line_straightness_gate(
    *,
    baseline_first_rgb: np.ndarray,
    baseline_second_rgb: np.ndarray,
    common_preview: np.ndarray,
    active_preview: np.ndarray,
    preview_left: int,
    preview_scale: float,
    mesh_warp: LocalMeshInverseWarp,
    settings: GeometryAssistedSeamConfig,
) -> dict[str, object]:
    """Veto a mesh only when observed baseline RGB lines visibly bend.

    Hough detection intentionally happens on the *unwarped second-source*
    baseline.  A candidate image can split an already bent door frame into
    short segments and make the defect invisible.  For a baseline source line
    ``p(t)``, the raw active mesh solver finds ``q(t)=M^-1(p(t))`` and measures
    the final-output chord deviation in full virtual pixels.  Identity
    fallbacks, protected samples, folds, and ambiguity are never accepted as
    a straight-line solution.  A geometry-safe wall need not contain a
    solver-valid 24 px RGB line after foreground/transparent structures have
    been hard-owned.  In that case this is an explicit ``not_observed`` audit,
    not a licence to bypass the always-required mesh centreline/edge/diagonal
    straightness audit.
    """

    first = np.asarray(baseline_first_rgb)
    second = np.asarray(baseline_second_rgb)
    common = np.asarray(common_preview, dtype=bool)
    active = np.asarray(active_preview, dtype=bool)
    scale = float(preview_scale)
    if (
        first.ndim != 3
        or first.shape[2] != 3
        or second.shape != first.shape
        or common.shape != first.shape[:2]
        or active.shape != first.shape[:2]
        or not math.isfinite(scale)
        or not 0.0 < scale <= 1.0
    ):
        raise RuntimeError("Actual RGB line-straightness inputs are malformed")
    minimum_full_length = float(settings.minimum_actual_rgb_line_length_pixels)
    minimum_preview_length = minimum_full_length * scale
    support_fraction = float(settings.minimum_actual_rgb_line_support_fraction)
    maximum_segments = int(settings.maximum_actual_rgb_line_segments)
    search_radius = max(1, int(math.ceil(
        float(settings.maximum_local_displacement_pixels) * scale
    )))
    gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    gray0 = cv2.GaussianBlur(gray0, (3, 3), 0.0)
    gray1 = cv2.GaussianBlur(gray1, (3, 3), 0.0)
    edge0 = (cv2.Canny(gray0, 30, 90, L2gradient=True) > 0) & common
    edge1 = (cv2.Canny(gray1, 30, 90, L2gradient=True) > 0) & common
    first_nearby = cv2.dilate(
        edge0.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) > 0
    active_search = cv2.dilate(
        active.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * search_radius + 1, 2 * search_radius + 1)
        ),
    ) > 0
    dual_source_edges = edge1 & first_nearby & active_search
    raw = cv2.HoughLinesP(
        dual_source_edges.astype(np.uint8) * 255,
        rho=1.0,
        theta=math.pi / 180.0,
        threshold=max(12, int(math.ceil(0.60 * minimum_preview_length))),
        minLineLength=minimum_preview_length,
        maxLineGap=2,
    )
    raw_segments = (
        np.empty((0, 4), dtype=np.float64)
        if raw is None
        else np.asarray(raw, dtype=np.float64).reshape(-1, 4)
    )
    segments = _deduplicate_hough_segments(raw_segments)
    base_audit: dict[str, object] = {
        "policy": (
            "baseline_second_rgb_dual_source_canny_hough_"
            "raw_forward_inverse_chord_bend"
        ),
        "preview_scale": scale,
        "metric_unit": "full_resolution_pixels",
        "minimum_actual_rgb_line_length_pixels": minimum_full_length,
        "minimum_actual_rgb_line_support_fraction": support_fraction,
        "maximum_actual_rgb_line_segments": maximum_segments,
        "maximum_straight_line_deviation_pixels": float(
            settings.maximum_straight_line_deviation_pixels
        ),
        "inverse_maximum_iterations": int(
            settings.actual_rgb_line_inverse_maximum_iterations
        ),
        "inverse_maximum_residual_pixels": float(
            settings.actual_rgb_line_inverse_maximum_residual_pixels
        ),
        "active_search_radius_preview_pixels": search_radius,
        "dual_source_edge_pixel_count": int(np.count_nonzero(dual_source_edges)),
        "raw_hough_segment_count": int(len(raw_segments)),
        "deduplicated_hough_segment_count": int(len(segments)),
    }
    if len(segments) > maximum_segments:
        return {
            **base_audit,
            "accepted": False,
            "observed": False,
            "reason": "actual_rgb_hough_segment_budget_exceeded",
            "eligible_hough_segment_count": 0,
            "tested_line_run_count": 0,
            "maximum_line_bend_pixels": None,
            "p95_line_bend_pixels": None,
            "maximum_inverse_residual_pixels": None,
            "p95_inverse_residual_pixels": None,
        }
    eligible: list[tuple[tuple[float, float, float, float, float, float, float], np.ndarray]] = []
    for segment in segments:
        samples = _sample_segment_points(segment)
        xs = np.rint(samples[:, 0]).astype(np.int64)
        ys = np.rint(samples[:, 1]).astype(np.int64)
        inside = (
            (xs >= 0)
            & (xs < dual_source_edges.shape[1])
            & (ys >= 0)
            & (ys < dual_source_edges.shape[0])
        )
        edge_support = np.zeros(len(samples), dtype=bool)
        edge_support[inside] = dual_source_edges[ys[inside], xs[inside]]
        if float(np.mean(edge_support)) >= support_fraction:
            eligible.append((segment, samples))
    if not eligible:
        return {
            **base_audit,
            "accepted": True,
            "observed": False,
            "reason": "not_observed_no_solver_valid_line",
            "eligible_hough_segment_count": 0,
            "tested_line_run_count": 0,
            "maximum_line_bend_pixels": None,
            "p95_line_bend_pixels": None,
            "maximum_inverse_residual_pixels": None,
            "p95_inverse_residual_pixels": None,
        }
    bends: list[float] = []
    residuals: list[float] = []
    tested_runs = 0
    for _segment, preview_points in eligible:
        source_points = np.column_stack(
            (
                (float(preview_left) + preview_points[:, 0] + 0.5) / scale - 0.5,
                (preview_points[:, 1] + 0.5) / scale - 0.5,
            )
        )
        solved = solve_active_mesh_forward_inverse(
            mesh_warp,
            source_points,
            maximum_iterations=int(settings.actual_rgb_line_inverse_maximum_iterations),
            maximum_residual_pixels=float(
                settings.actual_rgb_line_inverse_maximum_residual_pixels
            ),
        )
        for begin, end in _contiguous_valid_runs(solved.valid_mask):
            output_points = solved.output_points_xy[begin : end + 1]
            if len(output_points) < 2:
                continue
            chord = output_points[-1] - output_points[0]
            chord_length = float(np.linalg.norm(chord))
            if chord_length < minimum_full_length:
                continue
            perpendicular = np.abs(
                chord[0] * (output_points[:, 1] - output_points[0, 1])
                - chord[1] * (output_points[:, 0] - output_points[0, 0])
            ) / chord_length
            if not np.isfinite(perpendicular).all():
                continue
            run_residuals = solved.residual_pixels[begin : end + 1]
            if not np.isfinite(run_residuals).all():
                continue
            tested_runs += 1
            bends.extend(float(value) for value in perpendicular)
            residuals.extend(float(value) for value in run_residuals)
    if not tested_runs:
        return {
            **base_audit,
            "accepted": True,
            "observed": False,
            "reason": "not_observed_no_solver_valid_line",
            "eligible_hough_segment_count": int(len(eligible)),
            "tested_line_run_count": 0,
            "maximum_line_bend_pixels": None,
            "p95_line_bend_pixels": None,
            "maximum_inverse_residual_pixels": None,
            "p95_inverse_residual_pixels": None,
        }
    maximum_bend = float(np.max(bends))
    p95_bend = float(np.percentile(bends, 95.0))
    maximum_residual = float(np.max(residuals))
    p95_residual = float(np.percentile(residuals, 95.0))
    accepted = maximum_bend <= float(settings.maximum_straight_line_deviation_pixels)
    return {
        **base_audit,
        "accepted": bool(accepted),
        "observed": True,
        "reason": "accepted" if accepted else "rgb_actual_line_bend_exceeded",
        "eligible_hough_segment_count": int(len(eligible)),
        "tested_line_run_count": int(tested_runs),
        "maximum_line_bend_pixels": maximum_bend,
        "p95_line_bend_pixels": p95_bend,
        "maximum_inverse_residual_pixels": maximum_residual,
        "p95_inverse_residual_pixels": p95_residual,
    }


def _held_out_strong_rgb_edge_gate(
    before_evidence: object,
    after_evidence: object,
    *,
    settings: GeometryAssistedSeamConfig,
    preview_scale: float = 1.0,
) -> dict[str, object]:
    """Audit the same held-out, flow-supported strong RGB edges before/after.

    ``edge_normal_step_pixels`` is finite only at the strong Canny/Scharr edge
    support selected by :func:`extract_pair_evidence`.  Intersecting its
    deterministic held-out partition and accepted flow/epipolar support before
    and after the candidate prevents a mesh from claiming improvement merely
    by moving an edge outside its own evaluation set.
    """

    scale = float(preview_scale)
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise ValueError("Geometry strong-edge preview scale must be in (0, 1]")
    try:
        before_held = np.asarray(getattr(before_evidence, "held_out_mask"), dtype=bool)
        after_held = np.asarray(getattr(after_evidence, "held_out_mask"), dtype=bool)
        before_accepted = np.asarray(
            getattr(before_evidence, "accepted_mask"), dtype=bool
        )
        after_accepted = np.asarray(
            getattr(after_evidence, "accepted_mask"), dtype=bool
        )
        before_edge = np.asarray(
            getattr(before_evidence, "edge_normal_step_pixels"), dtype=np.float64
        )
        after_edge = np.asarray(
            getattr(after_evidence, "edge_normal_step_pixels"), dtype=np.float64
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Geometry strong-edge evidence is malformed") from exc
    shape = before_held.shape
    if (
        before_held.ndim != 2
        or after_held.shape != shape
        or before_accepted.shape != shape
        or after_accepted.shape != shape
        or before_edge.shape != shape
        or after_edge.shape != shape
        or not np.array_equal(before_held, after_held)
    ):
        raise RuntimeError("Geometry strong-edge evidence has inconsistent held-out support")
    support = (
        before_held
        & before_accepted
        & after_accepted
        & np.isfinite(before_edge)
        & np.isfinite(after_edge)
    )
    before_preview_values = before_edge[support]
    after_preview_values = after_edge[support]
    # Edge displacement is measured in the validation preview.  All formal
    # limits and delivery-side audit fields use full-resolution RGB pixels.
    before_values = before_preview_values / scale
    after_values = after_preview_values / scale
    count = int(before_values.size)
    minimum = int(settings.minimum_held_out_strong_edge_validation_pixels)
    before_p95 = float(np.percentile(before_values, 95.0)) if count else None
    after_p95 = float(np.percentile(after_values, 95.0)) if count else None
    after_max = float(np.max(after_values)) if count else None
    if before_p95 is None or after_p95 is None:
        improvement = None
        improvement_ratio = None
    else:
        improvement = float(before_p95 - after_p95)
        improvement_ratio = (
            float(improvement / before_p95) if before_p95 > 1e-9 else 0.0
        )
    accepted = bool(
        count >= minimum
        and after_p95 is not None
        and after_max is not None
        and improvement is not None
        and improvement_ratio is not None
        and after_p95 <= float(settings.maximum_held_out_error_pixels)
        and after_max <= float(settings.maximum_held_out_maximum_error_pixels)
        and improvement >= float(settings.minimum_held_out_improvement_pixels)
        and improvement_ratio >= float(settings.minimum_held_out_improvement_ratio)
    )
    if accepted:
        reason = "accepted"
    elif count < minimum:
        reason = "insufficient_held_out_strong_edge_support"
    elif after_p95 is None or after_p95 > float(settings.maximum_held_out_error_pixels):
        reason = "held_out_strong_edge_p95_exceeded"
    elif after_max is None or after_max > float(settings.maximum_held_out_maximum_error_pixels):
        reason = "held_out_strong_edge_maximum_exceeded"
    elif improvement is None or improvement < float(settings.minimum_held_out_improvement_pixels):
        reason = "held_out_strong_edge_absolute_improvement_insufficient"
    else:
        reason = "held_out_strong_edge_relative_improvement_insufficient"
    return {
        "accepted": accepted,
        "reason": reason,
        "policy": "same_held_out_flow_epipolar_supported_strong_rgb_edges",
        "metric_unit": "full_resolution_pixels",
        "preview_scale": scale,
        "held_out_strong_edge_pixel_count": count,
        "minimum_held_out_strong_edge_validation_pixels": minimum,
        "held_out_strong_edge_p95_before_pixels": before_p95,
        "held_out_strong_edge_p95_after_pixels": after_p95,
        "held_out_strong_edge_maximum_after_pixels": after_max,
        "held_out_strong_edge_p95_before_preview_pixels": (
            float(np.percentile(before_preview_values, 95.0)) if count else None
        ),
        "held_out_strong_edge_p95_after_preview_pixels": (
            float(np.percentile(after_preview_values, 95.0)) if count else None
        ),
        "held_out_strong_edge_maximum_after_preview_pixels": (
            float(np.max(after_preview_values)) if count else None
        ),
        "held_out_strong_edge_improvement_pixels": improvement,
        "held_out_strong_edge_improvement_ratio": improvement_ratio,
        "maximum_held_out_error_pixels": float(settings.maximum_held_out_error_pixels),
        "maximum_held_out_maximum_error_pixels": float(
            settings.maximum_held_out_maximum_error_pixels
        ),
        "minimum_held_out_improvement_pixels": float(
            settings.minimum_held_out_improvement_pixels
        ),
        "minimum_held_out_improvement_ratio": float(
            settings.minimum_held_out_improvement_ratio
        ),
    }


def _validate_geometry_candidate_rgb_flow(
    *,
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    layout: PushbroomLayout,
    residual_warps: Sequence[object | None],
    pair_index: int,
    mesh_warp: LocalMeshInverseWarp,
    corridor_x: tuple[int, int],
    active_mask: np.ndarray,
    settings: GeometryAssistedSeamConfig,
) -> dict[str, object]:
    """Verify a proposed mesh with held-out RGB flow, never use it to fit.

    The two images here are deliberately low-resolution analysis previews.
    They are never copied into a panorama and the DIS result never changes a
    mesh node or a camera pose.  Geometry remains the correspondence source;
    RGB merely rejects an otherwise geometric candidate when its safe layer
    has no mutually consistent visual motion.
    """

    if not 0 <= pair_index < len(frames) - 1:
        raise IndexError("Geometry flow validation pair index is out of range")
    if len(poses) != len(frames) or len(residual_warps) != len(frames):
        raise RuntimeError("Geometry flow validation inputs are not aligned")
    x0, x1 = (int(corridor_x[0]), int(corridor_x[1]))
    if x1 <= x0:
        raise RuntimeError("Geometry flow validation corridor is invalid")
    scale = float(settings.flow_validation_preview_scale)
    candidate_warps: list[object | None] = [None] * len(frames)
    candidate_warps[pair_index + 1] = mesh_warp
    baseline_renderer = CalibratedRGBPushbroomRenderer(
        layout,
        calibration,
        poses,
        residual_warps=residual_warps,
    )
    candidate_renderer = CalibratedRGBPushbroomRenderer(
        layout,
        calibration,
        poses,
        residual_warps=residual_warps,
        geometry_warps=candidate_warps,
    )

    def pair_evidence_for_renderer(
        renderer: CalibratedRGBPushbroomRenderer,
        *,
        retain_line_inputs: bool = False,
    ) -> tuple[
        object,
        int,
        int,
        int,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None,
    ]:
        first = renderer.render_preview_frame(
            frames[pair_index], pair_index, canvas_scale=scale
        )
        second = renderer.render_preview_frame(
            frames[pair_index + 1], pair_index + 1, canvas_scale=scale
        )
        (
            preview_left,
            preview_right,
            rgb0,
            rgb1,
            valid0,
            valid1,
            map0_x,
            map0_y,
            map1_x,
            map1_y,
        ) = _preview_pair_overlap(first, second)
        active_preview = _sample_geometry_active_in_preview(
            active_mask,
            corridor_x0=x0,
            preview_left=preview_left,
            preview_right=preview_right,
            preview_height=rgb0.shape[0],
            canvas_scale=scale,
        )
        common = (
            np.asarray(valid0, dtype=bool)
            & np.asarray(valid1, dtype=bool)
            & active_preview
        )
        evidence = extract_pair_evidence(
            first_rgb=rgb0,
            second_rgb=rgb1,
            first_valid=common,
            second_valid=common,
            first_inverse_map=(map0_x, map0_y),
            second_inverse_map=(map1_x, map1_y),
            intrinsics=calibration,
            first_camera_to_world=poses[pair_index],
            second_camera_to_world=poses[pair_index + 1],
            pair_index=pair_index,
            frame_ids=(
                int(frames[pair_index].frame_id),
                int(frames[pair_index + 1].frame_id),
            ),
            config=ResidualAlignmentConfig(
                maximum_flow_fb_error_pixels=float(
                    settings.maximum_held_out_flow_fb_error_pixels * scale
                )
            ),
        )
        line_inputs = (
            (
                np.ascontiguousarray(rgb0),
                np.ascontiguousarray(rgb1),
                np.ascontiguousarray(common),
                np.ascontiguousarray(active_preview),
                int(preview_left),
            )
            if retain_line_inputs
            else None
        )
        return (
            evidence,
            int(rgb0.shape[0] * rgb0.shape[1]),
            int(np.count_nonzero(active_preview)),
            int(np.count_nonzero(common)),
            line_inputs,
        )

    try:
        (
            baseline_evidence,
            baseline_overlap_pixels,
            baseline_active_pixels,
            baseline_common_pixels,
            baseline_line_inputs,
        ) = pair_evidence_for_renderer(baseline_renderer, retain_line_inputs=True)
        if baseline_line_inputs is None:
            raise RuntimeError("Baseline RGB line evidence was not retained")
        line_straightness = _actual_rgb_line_straightness_gate(
            baseline_first_rgb=baseline_line_inputs[0],
            baseline_second_rgb=baseline_line_inputs[1],
            common_preview=baseline_line_inputs[2],
            active_preview=baseline_line_inputs[3],
            preview_left=baseline_line_inputs[4],
            preview_scale=scale,
            mesh_warp=mesh_warp,
            settings=settings,
        )
        # Candidate RGB remaps cannot rescue an observed baseline line that
        # fails its bend audit.  A baseline with no solver-valid line is an
        # explicitly audited non-veto result; intrinsic mesh
        # centreline/edge/diagonal checks remain mandatory in either case.
        # Release preview pixels before the candidate pass; only scalar
        # Hough/solver evidence survives.
        del baseline_line_inputs
        if line_straightness.get("accepted") is not True:
            return {
                "accepted": False,
                "reason": str(line_straightness.get("reason")),
                "preview_scale": scale,
                "preview_overlap_pixel_count": baseline_overlap_pixels,
                "active_preview_pixel_count": baseline_active_pixels,
                "common_preview_pixel_count": baseline_common_pixels,
                "held_out_observable_flow_pixel_count": 0,
                "held_out_flow_fb_error_p95_pixels": None,
                "held_out_flow_fb_error_max_pixels": None,
                "maximum_held_out_flow_fb_error_pixels": float(
                    settings.maximum_held_out_flow_fb_error_pixels
                ),
                "minimum_held_out_flow_validation_pixels": int(
                    settings.minimum_held_out_flow_validation_pixels
                ),
                "held_out_accepted_pixel_count": 0,
                "held_out_strong_edge": {
                    "accepted": False,
                    "reason": "not_evaluated_after_actual_rgb_line_rejection",
                    "policy": "same_held_out_flow_epipolar_supported_strong_rgb_edges",
                },
                "rgb_actual_line_straightness": line_straightness,
                "analysis_preview_remap_count": int(
                    baseline_renderer.analysis_preview_remap_count
                ),
            }
        (
            evidence,
            preview_overlap_pixels,
            active_preview_pixels,
            common_preview_pixels,
            _candidate_line_inputs,
        ) = pair_evidence_for_renderer(candidate_renderer)
        strong_edge = _held_out_strong_rgb_edge_gate(
            baseline_evidence,
            evidence,
            settings=settings,
            preview_scale=scale,
        )
    except (ValueError, RuntimeError, cv2.error) as exc:
        return {
            "accepted": False,
            "reason": "rgb_held_out_validation_unavailable",
            "preview_scale": scale,
            "analysis_preview_remap_count": int(
                baseline_renderer.analysis_preview_remap_count
                + candidate_renderer.analysis_preview_remap_count
            ),
            "detail": str(exc),
        }
    # `accepted_mask` carries target-valid, texture, bidirectional-flow and
    # real-SE(3) epipolar support.  A merely finite flow value cannot certify
    # a mesh after one of those independent checks has rejected it.
    flow_mask = (
        np.asarray(evidence.held_out_mask, dtype=bool)
        & np.asarray(evidence.accepted_mask, dtype=bool)
        & np.isfinite(np.asarray(evidence.forward_backward_error, dtype=np.float64))
    )
    flow_preview_values = np.asarray(
        evidence.forward_backward_error, dtype=np.float64
    )[flow_mask]
    flow_values = flow_preview_values / scale
    held_out_flow_count = int(flow_values.size)
    flow_p95 = (
        float(np.percentile(flow_values, 95.0)) if held_out_flow_count else None
    )
    flow_max = float(np.max(flow_values)) if held_out_flow_count else None
    flow_accepted = bool(
        held_out_flow_count >= int(settings.minimum_held_out_flow_validation_pixels)
        and flow_p95 is not None
        and flow_p95 <= float(settings.maximum_held_out_flow_fb_error_pixels)
    )
    flow_reason = (
        "accepted"
        if flow_accepted
        else (
            "insufficient_held_out_rgb_flow_support"
            if held_out_flow_count < int(settings.minimum_held_out_flow_validation_pixels)
            else "held_out_rgb_flow_fb_error_exceeded"
        )
    )
    accepted = bool(
        flow_accepted
        and strong_edge["accepted"] is True
        and line_straightness["accepted"] is True
    )
    reason = "accepted" if accepted else (
        flow_reason
        if not flow_accepted
        else (
            str(strong_edge["reason"])
            if strong_edge["accepted"] is not True
            else str(line_straightness["reason"])
        )
    )
    return {
        "accepted": accepted,
        "reason": reason,
        "preview_scale": scale,
        "preview_overlap_pixel_count": preview_overlap_pixels,
        "active_preview_pixel_count": active_preview_pixels,
        "common_preview_pixel_count": common_preview_pixels,
        "held_out_observable_flow_pixel_count": held_out_flow_count,
        "held_out_flow_fb_error_p95_pixels": flow_p95,
        "held_out_flow_fb_error_max_pixels": flow_max,
        "held_out_flow_fb_error_p95_preview_pixels": (
            float(np.percentile(flow_preview_values, 95.0))
            if held_out_flow_count
            else None
        ),
        "held_out_flow_fb_error_max_preview_pixels": (
            float(np.max(flow_preview_values)) if held_out_flow_count else None
        ),
        "metric_unit": "full_resolution_pixels",
        "maximum_held_out_flow_fb_error_pixels": float(
            settings.maximum_held_out_flow_fb_error_pixels
        ),
        "minimum_held_out_flow_validation_pixels": int(
            settings.minimum_held_out_flow_validation_pixels
        ),
        "held_out_accepted_pixel_count": int(evidence.held_out_accepted_count),
        "held_out_strong_edge": strong_edge,
        "rgb_actual_line_straightness": line_straightness,
        "analysis_preview_remap_count": int(
            baseline_renderer.analysis_preview_remap_count
            + candidate_renderer.analysis_preview_remap_count
        ),
    }


def _save_contribution(path: Path, contribution: PushbroomContribution) -> None:
    np.savez_compressed(
        path,
        source_index=np.asarray(contribution.source_index, dtype=np.int32),
        frame_id=np.asarray(contribution.frame_id, dtype=np.int32),
        x0=np.asarray(contribution.x0, dtype=np.int32),
        rgb=contribution.rgb,
        valid=contribution.valid_mask.astype(np.uint8),
    )


def _load_contribution(path: Path) -> PushbroomContribution:
    with np.load(path, allow_pickle=False) as stored:
        return PushbroomContribution(
            source_index=int(stored["source_index"]),
            frame_id=int(stored["frame_id"]),
            x0=int(stored["x0"]),
            rgb=np.ascontiguousarray(stored["rgb"]),
            valid_mask=np.ascontiguousarray(stored["valid"].astype(bool)),
        )


def _save_preview(path: Path, contribution: PreviewContribution) -> None:
    """Spool analysis-only preview evidence; never publish it with a delivery."""

    np.savez_compressed(
        path,
        source_index=np.asarray(contribution.source_index, dtype=np.int32),
        frame_id=np.asarray(contribution.frame_id, dtype=np.int32),
        x0=np.asarray(contribution.x0, dtype=np.int32),
        canvas_scale=np.asarray(contribution.canvas_scale, dtype=np.float64),
        rgb=contribution.rgb,
        valid=contribution.valid_mask.astype(np.uint8),
        source_map_x=contribution.source_map_x,
        source_map_y=contribution.source_map_y,
    )


def _load_preview(path: Path) -> PreviewContribution:
    with np.load(path, allow_pickle=False) as stored:
        return PreviewContribution(
            source_index=int(stored["source_index"]),
            frame_id=int(stored["frame_id"]),
            x0=int(stored["x0"]),
            canvas_scale=float(stored["canvas_scale"]),
            rgb=np.ascontiguousarray(stored["rgb"]),
            valid_mask=np.ascontiguousarray(stored["valid"].astype(bool)),
            source_map_x=np.ascontiguousarray(stored["source_map_x"]),
            source_map_y=np.ascontiguousarray(stored["source_map_y"]),
        )


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.magnitude(
        cv2.Scharr(gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray, cv2.CV_32F, 0, 1),
    )


def _srgb_to_linear_bgr(image: np.ndarray) -> np.ndarray:
    """Decode a uint8 BGR image into the linear-light domain exactly once."""

    encoded = np.asarray(image, dtype=np.float32) / 255.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb_bgr(linear: np.ndarray) -> np.ndarray:
    """Encode finite linear BGR samples back to uint8 sRGB."""

    values = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(encoded * 255.0, 0.0, 255.0)).astype(np.uint8)


def _apply_linear_bgr_gain(image: np.ndarray, gain_bgr: np.ndarray) -> np.ndarray:
    """Apply one finite three-channel gain in linear RGB, never gamma RGB."""

    gain = np.asarray(gain_bgr, dtype=np.float32).reshape(1, 1, 3)
    if not np.isfinite(gain).all() or np.any(gain < 0.45) or np.any(gain > 2.20):
        raise RuntimeError("Linear RGB photometric gain is outside the formal range")
    return _linear_to_srgb_bgr(_srgb_to_linear_bgr(image) * gain)


@dataclass(frozen=True)
class _RGBRiskDetails:
    mask: np.ndarray
    # ``mask`` is the dilated owner/MultiBand guard.  ``seed_mask`` remains
    # private analysis evidence for deciding whether a boundary contains
    # enough real RGB structure to justify an RGB-D local mesh attempt.
    # Never publish either dense array.
    seed_mask: np.ndarray
    # Edge displacement/orientation seeds only.  A Lab-only colour residual
    # remains a hard-owner/MultiBand guard but is photometric evidence, not a
    # licence to introduce a geometric local deformation.
    structural_seed_mask: np.ndarray
    edge_offset_p95: float
    seed_pixel_count: int
    component_count: int


@dataclass(frozen=True)
class _PhotometricEdge:
    """One reliable adjacent safe-wall observation in linear BGR log space."""

    log_relation_bgr: np.ndarray
    support_pixels: int
    mad_bgr: np.ndarray
    raw_signed_l_delta: float
    safe_mask: np.ndarray | None = None


def _fill_risk_components(seed: np.ndarray, common: np.ndarray, bridge_radius: int) -> np.ndarray:
    """Close exposed risk edges and fill bounded foreground components."""

    bridge = max(3, int(bridge_radius) * 2 + 1)
    closed = cv2.morphologyEx(
        (np.asarray(seed, dtype=bool) & common).astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bridge, bridge)),
    ) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed.astype(np.uint8), connectivity=8
    )
    filled = closed.copy()
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) < 2:
            continue
        x0 = int(stats[label, cv2.CC_STAT_LEFT])
        y0 = int(stats[label, cv2.CC_STAT_TOP])
        x1 = x0 + int(stats[label, cv2.CC_STAT_WIDTH])
        y1 = y0 + int(stats[label, cv2.CC_STAT_HEIGHT])
        component = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        interior = np.zeros_like(component)
        cv2.drawContours(interior, contours, -1, 1, thickness=cv2.FILLED)
        filled[y0:y1, x0:x1] |= interior > 0
    return filled & common


def _rgb_risk_details(
    image0: np.ndarray,
    image1: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    *,
    supplied_risk_mask: np.ndarray | None = None,
) -> _RGBRiskDetails:
    """Build RGB-only foreground risk from colour, edge distance and structure.

    A high Lab residual alone is deliberately sufficient.  The other two
    independent seeds detect a displaced dark object's exposed outline even
    when its uniform interior carries almost no gradient or Lab residual.
    """

    common = np.asarray(valid0, dtype=bool) & np.asarray(valid1, dtype=bool)
    if not np.any(common):
        return _RGBRiskDetails(
            mask=np.zeros(common.shape, dtype=np.uint8),
            seed_mask=np.zeros(common.shape, dtype=bool),
            structural_seed_mask=np.zeros(common.shape, dtype=bool),
            edge_offset_p95=0.0,
            seed_pixel_count=0,
            component_count=0,
        )
    lab0 = cv2.cvtColor(image0, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(image1, cv2.COLOR_BGR2LAB).astype(np.float32)
    difference = np.linalg.norm(lab0 - lab1, axis=2)
    values = difference[common]
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    lab_threshold = max(18.0, median + 3.5 * max(mad, 1.0))
    lab_seed = common & (difference >= lab_threshold)

    gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gx0, gy0 = (
        cv2.Scharr(gray0, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray0, cv2.CV_32F, 0, 1),
    )
    gx1, gy1 = (
        cv2.Scharr(gray1, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray1, cv2.CV_32F, 0, 1),
    )
    magnitude0 = cv2.magnitude(gx0, gy0)
    magnitude1 = cv2.magnitude(gx1, gy1)
    magnitudes = np.concatenate((magnitude0[common], magnitude1[common]))
    # A displaced outline is caught by the symmetric edge-distance seed below.
    # The structure term is intentionally reserved for genuinely strong edges;
    # treating every modest Scharr fluctuation as foreground would turn normal
    # wall texture/illumination noise into a full-width risk component.
    gradient_threshold = max(80.0, float(np.percentile(magnitudes, 90.0)))
    denominator = np.maximum(magnitude0 * magnitude1, 1e-6)
    cosine = (gx0 * gx1 + gy0 * gy1) / denominator
    strong = (magnitude0 >= gradient_threshold) & (magnitude1 >= gradient_threshold)
    gradient_seed = common & strong & (
        (np.abs(magnitude0 - magnitude1) >= np.maximum(64.0, 0.65 * np.maximum(magnitude0, magnitude1)))
        | (cosine < 0.25)
    )

    edge0 = cv2.Canny(
        cv2.GaussianBlur(gray0, (3, 3), 0), 30, 90, L2gradient=True
    ) > 0
    edge1 = cv2.Canny(
        cv2.GaussianBlur(gray1, (3, 3), 0), 30, 90, L2gradient=True
    ) > 0
    distance0 = cv2.distanceTransform((~edge0).astype(np.uint8), cv2.DIST_L2, 3)
    distance1 = cv2.distanceTransform((~edge1).astype(np.uint8), cv2.DIST_L2, 3)
    edge_offset = (edge0 & common & (distance1 > 1.5)) | (
        edge1 & common & (distance0 > 1.5)
    )
    offsets = np.concatenate((distance1[edge0 & common], distance0[edge1 & common]))
    finite_offsets = offsets[
        np.isfinite(offsets)
        & (offsets > 0.0)
        # OpenCV uses FLT_MAX when an image contains no opposing edge at all;
        # that is not a measurable local edge displacement.
        & (offsets <= float(max(image0.shape[:2])))
    ]
    # P95 is a local registration-displacement statistic.  An edge with no
    # counterpart anywhere in the corridor remains a risk seed, but must not
    # turn a single unrelated missing edge into a 400 px global guard radius.
    local_offsets = finite_offsets[finite_offsets <= 16.0]
    edge_offset_p95 = (
        float(np.percentile(local_offsets, 95.0)) if local_offsets.size else 0.0
    )

    supplied = np.zeros(common.shape, dtype=bool)
    if supplied_risk_mask is not None:
        supplied_array = np.asarray(supplied_risk_mask)
        if supplied_array.shape != common.shape:
            raise ValueError("RGB disparity risk mask must match the strip overlap")
        if supplied_array.dtype not in {np.dtype(np.uint8), np.dtype(np.bool_)}:
            raise ValueError("RGB disparity risk mask must be uint8 or bool")
        supplied = supplied_array > 0
    seed = common & (lab_seed | edge_offset | gradient_seed | supplied)
    structural_seed = common & (edge_offset | gradient_seed)
    # Keep the bridge deliberately local.  A large p95 will subsequently make
    # the guard large and the constrained seam fail rather than silently grow
    # an arbitrary wall into a foreground mask.
    bridge_radius = min(8, max(2, int(math.ceil(edge_offset_p95))))
    filled = _fill_risk_components(seed, common, bridge_radius)
    risk = cv2.dilate(
        filled.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    risk[~common] = 0
    component_count = int(cv2.connectedComponents((risk > 0).astype(np.uint8))[0] - 1)
    return _RGBRiskDetails(
        mask=np.ascontiguousarray(risk),
        seed_mask=np.ascontiguousarray(seed),
        structural_seed_mask=np.ascontiguousarray(structural_seed),
        edge_offset_p95=edge_offset_p95,
        seed_pixel_count=int(np.count_nonzero(seed)),
        component_count=component_count,
    )


def _adaptive_multiband_levels(blend_width: int, requested_levels: int) -> int:
    if blend_width <= 2:
        natural = 1
    elif blend_width <= 4:
        natural = 2
    else:
        natural = 3
    return min(natural, max(1, int(requested_levels)))


def _risk_guard(
    risk: _RGBRiskDetails,
    common: np.ndarray,
    blend_width: int,
    requested_levels: int,
    *,
    geometry_protected: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Fill/dilate RGB risk and union non-blendable geometry protection."""

    levels = _adaptive_multiband_levels(blend_width, requested_levels)
    pyramid_support = 1 << max(0, levels - 1)
    radius = max(2, int(math.ceil(risk.edge_offset_p95)) + pyramid_support + 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    guard = cv2.dilate((risk.mask > 0).astype(np.uint8), kernel) > 0
    # Treat a one-pixel residual gap between adjacent protected fragments as
    # part of the same foreground component.  Otherwise two independently
    # assigned fragments can leave an unusable one-pixel "safe" slit that no
    # 2--8 px owner boundary can traverse without touching the guard.
    guard = cv2.morphologyEx(
        guard.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    usable = np.asarray(common, dtype=bool)
    if geometry_protected is not None:
        protected = np.asarray(geometry_protected, dtype=bool)
        if protected.shape != usable.shape:
            raise ValueError("Geometry protection mask must match the seam corridor")
        guard |= protected
    guard &= usable
    component_count = int(cv2.connectedComponents(guard.astype(np.uint8))[0] - 1)
    return guard, radius, component_count


def _safe_white_wall_mask(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    risk_guard: np.ndarray,
    *,
    low_gradient_quantile: float,
) -> np.ndarray:
    """Return the conservative low-texture neutral wall support for one pair."""

    usable = np.asarray(common, dtype=bool) & ~np.asarray(risk_guard, dtype=bool)
    if int(np.count_nonzero(usable)) < 16:
        return np.zeros_like(usable)
    lab0 = cv2.cvtColor(first, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(second, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad0, grad1 = _gradient_magnitude(first), _gradient_magnitude(second)
    gradient_values = np.concatenate((grad0[usable], grad1[usable]))
    if not gradient_values.size:
        return np.zeros_like(usable)
    gradient_limit = float(
        np.clip(
            np.percentile(gradient_values, 100.0 * low_gradient_quantile),
            18.0,
            150.0,
        )
    )
    hsv0 = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)
    valid_distance = cv2.distanceTransform(usable.astype(np.uint8), cv2.DIST_L2, 3)
    unclipped = (
        (np.min(first, axis=2) >= 10)
        & (np.min(second, axis=2) >= 10)
        & (np.max(first, axis=2) <= 250)
        & (np.max(second, axis=2) <= 250)
        & (lab0[:, :, 0] >= 42.0)
        & (lab1[:, :, 0] >= 42.0)
        & (hsv0[:, :, 1] <= 48)
        & (hsv1[:, :, 1] <= 48)
        & (np.maximum(np.abs(lab0[:, :, 1] - 128.0), np.abs(lab0[:, :, 2] - 128.0)) <= 20.0)
        & (np.maximum(np.abs(lab1[:, :, 1] - 128.0), np.abs(lab1[:, :, 2] - 128.0)) <= 20.0)
    )
    candidates = (
        usable
        & (valid_distance >= 3.0)
        & unclipped
        & (grad0 <= gradient_limit)
        & (grad1 <= gradient_limit)
    )
    if int(np.count_nonzero(candidates)) < 16:
        return np.zeros_like(usable)
    linear0 = _srgb_to_linear_bgr(first)
    linear1 = _srgb_to_linear_bgr(second)
    provisional = np.median(
        np.log(np.maximum(linear0[candidates], 1e-4))
        - np.log(np.maximum(linear1[candidates], 1e-4)),
        axis=0,
    )
    adjusted_second = _linear_to_srgb_bgr(
        linear1 * np.exp(np.asarray(provisional, dtype=np.float32)).reshape(1, 1, 3)
    )
    adjusted_lab = cv2.cvtColor(adjusted_second, cv2.COLOR_BGR2LAB).astype(np.float32)
    residual = np.linalg.norm(lab0 - adjusted_lab, axis=2)
    residual_values = residual[candidates]
    residual_median = float(np.median(residual_values))
    residual_mad = float(np.median(np.abs(residual_values - residual_median)))
    residual_limit = max(3.0, residual_median + 3.5 * max(residual_mad, 0.5))
    return candidates & (residual <= residual_limit)


def _robust_log_channel_relation(values: np.ndarray, minimum: int) -> tuple[float, float] | None:
    """Return a trimmed Huber log-ratio estimate and its robust MAD."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < minimum:
        return None
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    keep = np.abs(finite - median) <= max(0.004, 3.0 * 1.4826 * mad)
    trimmed = finite[keep]
    # The initial safe support must meet the formal minimum.  Huber trimming
    # may legitimately discard a minority of tiny remap/interpolation
    # outliers, so require a still-substantial half of that support rather
    # than falsely declaring an otherwise clean narrow wall unusable.
    if trimmed.size < max(32, int(math.ceil(minimum * 0.5))):
        return None
    centre = float(np.median(trimmed))
    scale = max(0.0025, 1.4826 * float(np.median(np.abs(trimmed - centre))))
    weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(trimmed - centre), 1e-9))
    estimate = float(np.sum(weights * trimmed) / np.sum(weights))
    if not np.isfinite(estimate):
        return None
    return estimate, mad


def _safe_wall_log_relation(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    risk_guard: np.ndarray,
    *,
    low_gradient_quantile: float,
) -> _PhotometricEdge | None:
    """Measure all three linear-BGR gain relations from safe neutral wall only."""

    safe = _safe_white_wall_mask(
        first,
        second,
        common,
        risk_guard,
        low_gradient_quantile=low_gradient_quantile,
    )
    support = int(np.count_nonzero(safe))
    # A narrow but genuinely clean corridor can contain only a few dozen wall
    # pixels; robust trimming still leaves a useful estimate at 64 samples.
    # Anything below that is a structural no-support failure.
    minimum = 64
    if support < minimum:
        return None
    linear0 = _srgb_to_linear_bgr(first)
    linear1 = _srgb_to_linear_bgr(second)
    ratios = np.log(np.maximum(linear0, 1e-4)) - np.log(np.maximum(linear1, 1e-4))
    relation = np.empty(3, dtype=np.float64)
    mad = np.empty(3, dtype=np.float64)
    for channel in range(3):
        robust = _robust_log_channel_relation(ratios[:, :, channel][safe], minimum)
        if robust is None:
            return None
        relation[channel], mad[channel] = robust
    lab0 = cv2.cvtColor(first, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(second, cv2.COLOR_BGR2LAB).astype(np.float32)
    signed_l_delta = float(np.median((lab0[:, :, 0] - lab1[:, :, 0])[safe]) * (100.0 / 255.0))
    return _PhotometricEdge(
        log_relation_bgr=relation,
        support_pixels=support,
        mad_bgr=mad,
        raw_signed_l_delta=signed_l_delta,
        safe_mask=np.ascontiguousarray(safe),
    )


def _pair_overlap(
    first: PushbroomContribution, second: PushbroomContribution
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left, right = max(first.x0, second.x0), min(first.x1, second.x1)
    if right <= left:
        raise RuntimeError("Adjacent calibrated RGB strips have no real overlap")
    first_slice = slice(left - first.x0, right - first.x0)
    second_slice = slice(left - second.x0, right - second.x0)
    return (
        left,
        right,
        first.rgb[:, first_slice],
        second.rgb[:, second_slice],
        first.valid_mask[:, first_slice],
        second.valid_mask[:, second_slice],
    )


def _preview_pair_overlap(
    first: PreviewContribution, second: PreviewContribution
) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Return adjacent preview overlap plus both raw-coordinate inverse maps."""

    left, right = max(first.x0, second.x0), min(first.x1, second.x1)
    if right <= left:
        raise RuntimeError("Adjacent calibrated RGB previews have no real overlap")
    first_slice = slice(left - first.x0, right - first.x0)
    second_slice = slice(left - second.x0, right - second.x0)
    return (
        left,
        right,
        first.rgb[:, first_slice],
        second.rgb[:, second_slice],
        first.valid_mask[:, first_slice],
        second.valid_mask[:, second_slice],
        first.source_map_x[:, first_slice],
        first.source_map_y[:, first_slice],
        second.source_map_x[:, second_slice],
        second.source_map_y[:, second_slice],
    )


def _preview_canvas_scale(
    layout: PushbroomLayout, settings: ResidualAlignmentConfig
) -> float:
    """Bound preview analysis by formal width *and* pixel working-set caps."""

    full_pixels = int(layout.canvas_height) * int(layout.canvas_width)
    if full_pixels <= 0:
        raise RuntimeError("Calibrated RGB preview canvas has no pixels")
    width_scale = min(1.0, float(settings.analysis_width) / float(layout.canvas_width))
    pixel_scale = min(
        1.0,
        math.sqrt(
            float(settings.maximum_preview_megapixels) * 1_000_000.0
            / float(full_pixels)
        ),
    )
    return min(width_scale, pixel_scale)


def _residual_support_margins(layout: PushbroomLayout) -> tuple[float, ...]:
    """Return calibrated support spare for each source-owned interval."""

    margins: list[float] = []
    for left, right, support_left, support_right in zip(
        layout.owner_left_x,
        layout.owner_right_x,
        layout.support_left_x,
        layout.support_right_x,
        strict=True,
    ):
        # Endpoint outer FOV may sit directly on a calibrated image edge.  It
        # has no deformation spare, so a non-identity residual must pin that
        # source to identity instead of silently consuming an owner core.
        margins.append(max(0.0, min(float(left - support_left), float(support_right - right))))
    return tuple(margins)


def _residual_support_bounds(
    layout: PushbroomLayout,
) -> tuple[tuple[float, float, float, float], ...]:
    """Conservative virtual rectangles used to audit full SE(2) deformation."""

    bottom = float(layout.canvas_height - 1)
    return tuple(
        (
            float(left),
            float(right - 1),
            0.0,
            bottom,
        )
        for left, right in zip(
            layout.support_left_x, layout.support_right_x, strict=True
        )
    )


def _minimum_residual_support_margin(layout: PushbroomLayout) -> float:
    return min(_residual_support_margins(layout), default=0.0)


def _held_out_edge_metrics(
    evidence: Sequence[object],
) -> dict[str, float | int | None]:
    """Aggregate only immutable held-out edge samples for model comparison."""

    values: list[np.ndarray] = []
    accepted = 0
    held_out = 0
    for item in evidence:
        # Keep this helper structural: it only consumes the public evidence
        # arrays and does not need a renderer-specific class import.
        held_mask = np.asarray(getattr(item, "held_out_mask"), dtype=bool)
        accepted_mask = np.asarray(getattr(item, "accepted_mask"), dtype=bool)
        edge_steps = np.asarray(getattr(item, "edge_normal_step_pixels"), dtype=np.float64)
        samples = edge_steps[held_mask & accepted_mask]
        samples = samples[np.isfinite(samples)]
        if samples.size:
            values.append(samples)
        accepted += int(np.count_nonzero(held_mask & accepted_mask))
        held_out += int(np.count_nonzero(held_mask))
    merged = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    return {
        "held_out_correspondence_count": held_out,
        "held_out_accepted_count": accepted,
        "held_out_edge_step_p50_pixels": (
            float(np.percentile(merged, 50.0)) if merged.size else None
        ),
        "held_out_edge_step_p95_pixels": (
            float(np.percentile(merged, 95.0)) if merged.size else None
        ),
        "held_out_edge_step_max_pixels": float(np.max(merged)) if merged.size else None,
    }


def _identity_residual_alignment_result(
    layout: PushbroomLayout,
    calibration: CameraIntrinsics,
    settings: ResidualAlignmentConfig,
    evidence: Sequence[object],
    *,
    preview_remap_count: int,
    support_bounds: Sequence[tuple[float, float, float, float]] | None = None,
    held_out_additional: Mapping[str, float | int | str | bool | None] | None = None,
    working_set_additional: Mapping[str, int | float | str | bool | None] | None = None,
) -> ResidualAlignmentResult:
    """Build the Phase-0 selected model without changing any output mapping."""

    warps = tuple(
        SourceResidualWarp.identity(
            index,
            centre_x=float(layout.source_centres_x[index]),
            centre_y=float(calibration.cy),
        )
        for index in range(len(layout.frame_ids))
    )
    topology = audit_source_warps(
        warps,
        settings,
        support_margin_pixels=_minimum_residual_support_margin(layout),
        support_bounds=support_bounds,
    )
    if not topology.accepted:
        raise RuntimeError("Identity residual-alignment topology audit failed")
    held_out = _held_out_edge_metrics(evidence)
    if held_out_additional:
        held_out.update(held_out_additional)
    working_set: dict[str, int | float | bool | str | None] = {
        "analysis_preview_remap_count": int(preview_remap_count),
        "preview_streaming_maximum_resident_strips": 2,
        "preview_canvas_scale": _preview_canvas_scale(layout, settings),
        "preview_output_pixels": int(
            math.ceil(layout.canvas_height * _preview_canvas_scale(layout, settings))
            * math.ceil(layout.canvas_width * _preview_canvas_scale(layout, settings))
        ),
    }
    if working_set_additional:
        working_set.update(working_set_additional)
    return ResidualAlignmentResult(
        selected_model="identity",
        source_warps=warps,
        pair_evidence=tuple(evidence),
        held_out_metrics_before=held_out,
        held_out_metrics_after=dict(held_out),
        component_tracks=(),
        topology_audit=topology,
        working_set_audit=working_set,
    )


def _select_seam_search_corridor(
    left: int,
    right: int,
    first: np.ndarray,
    second: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    *,
    nominal_boundary_x: float,
    requested_width: int,
    allow_short_endpoint_corridor: bool = False,
) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Restrict a wide calibrated overlap to the formal owner-search band.

    Full central-strip overlap is retained for reliable global photometry only.
    This helper is the sole path by which GraphCut, risk guards, owners, and
    MultiBand see a corridor, keeping their search at 32--64 px even when the
    source's legal 20% central support is substantially wider.
    """

    available = int(right - left)
    if available <= 0:
        raise RuntimeError("Adjacent calibrated RGB strips have no seam support")
    if available < 32 and not allow_short_endpoint_corridor:
        raise RuntimeError(
            "Interior calibrated RGB seam support is below the 32-pixel "
            "formal search minimum"
        )
    width = min(int(requested_width), available)
    if width < 2:
        raise RuntimeError("Calibrated RGB seam-search support collapsed")
    centre = int(round(float(nominal_boundary_x)))
    start = centre - width // 2
    start = int(np.clip(start, left, right - width))
    stop = start + width
    source = slice(start - left, stop - left)
    return (
        start,
        stop,
        first[:, source],
        second[:, source],
        valid0[:, source],
        valid1[:, source],
    )


def _solve_global_linear_rgb_gains(
    source_count: int, edges: Sequence[_PhotometricEdge | None]
) -> tuple[np.ndarray, dict[str, int | float | bool]]:
    """Jointly solve all 3N linear-light gains with robust smoothness.

    Every adjacent relation is mandatory.  Unlike sequential accumulation,
    this makes all measurements participate in a single IRLS problem and uses
    a second-difference penalty only on the *gain curve*, never on image
    pixels.  A mean-zero gauge preserves the overall scene brightness.
    """

    if len(edges) != source_count - 1:
        raise ValueError("Photometric observations must cover every adjacent pair")
    missing = [index for index, edge in enumerate(edges) if edge is None]
    if missing:
        raise RuntimeError(
            "No reliable safe white-wall photometric support for adjacent pair(s): "
            + ", ".join(str(index) for index in missing)
        )
    checked = [edge for edge in edges if edge is not None]
    assert len(checked) == source_count - 1
    relations = np.asarray([edge.log_relation_bgr for edge in checked], dtype=np.float64)
    mads = np.asarray([edge.mad_bgr for edge in checked], dtype=np.float64)
    support = np.asarray([edge.support_pixels for edge in checked], dtype=np.float64)
    if (
        relations.shape != (source_count - 1, 3)
        or not np.isfinite(relations).all()
        or not np.isfinite(mads).all()
        or np.any(mads < 0.0)
        or np.any(support <= 0.0)
    ):
        raise RuntimeError("Safe white-wall photometric observations are invalid")

    log_gains = np.empty((source_count, 3), dtype=np.float64)
    residuals = np.empty_like(relations)
    maximum_condition = 0.0
    iterations = 4
    # The robust wall statistics are measured from thousands of pixels on most
    # edges.  Keep a light second-difference prior to suppress an isolated bad
    # edge without biasing the channel relations enough to leave a visible
    # colour-temperature step at every owner boundary.
    smoothness = 0.08
    for channel in range(3):
        confidence = np.clip(support / 512.0, 0.25, 12.0) / np.maximum(
            np.square(np.maximum(mads[:, channel], 0.004) / 0.012), 1.0
        )
        base_weight = confidence / max(float(np.median(confidence)), 1e-9)
        base_weight = np.clip(base_weight, 0.10, 10.0)
        robust_weight = np.ones(source_count - 1, dtype=np.float64)
        solution = np.zeros(source_count, dtype=np.float64)
        for _ in range(iterations):
            rows: list[np.ndarray] = []
            rhs: list[float] = []
            for index, relation in enumerate(relations[:, channel]):
                row = np.zeros(source_count, dtype=np.float64)
                weight = math.sqrt(float(base_weight[index] * robust_weight[index]))
                row[index] = -weight
                row[index + 1] = weight
                rows.append(row)
                rhs.append(weight * float(relation))
            if source_count >= 3:
                curvature_weight = math.sqrt(smoothness)
                for index in range(1, source_count - 1):
                    row = np.zeros(source_count, dtype=np.float64)
                    row[index - 1 : index + 2] = (
                        curvature_weight,
                        -2.0 * curvature_weight,
                        curvature_weight,
                    )
                    rows.append(row)
                    rhs.append(0.0)
            gauge = np.full(source_count, math.sqrt(100.0 / source_count), dtype=np.float64)
            rows.append(gauge)
            rhs.append(0.0)
            matrix = np.vstack(rows)
            target = np.asarray(rhs, dtype=np.float64)
            rank = int(np.linalg.matrix_rank(matrix))
            condition = float(np.linalg.cond(matrix))
            maximum_condition = max(maximum_condition, condition)
            if rank != source_count or not np.isfinite(condition) or condition > 1e8:
                raise RuntimeError("Global RGB photometric solver is rank-deficient or ill-conditioned")
            solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
            if not np.isfinite(solution).all():
                raise RuntimeError("Global RGB photometric solver returned non-finite gains")
            residual = np.diff(solution) - relations[:, channel]
            scale = max(0.0025, 1.4826 * float(np.median(np.abs(residual))))
            robust_weight = np.minimum(
                1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-9)
            )
        log_gains[:, channel] = solution
        residuals[:, channel] = np.diff(solution) - relations[:, channel]

    gains = np.exp(log_gains)
    if not np.isfinite(gains).all() or np.any(gains < 0.45) or np.any(gains > 2.20):
        raise RuntimeError("Global linear RGB gains exceed the formal 0.45-2.20 range")
    return gains, {
        "photometric_mode": "safe_wall_global_linear_rgb",
        "safe_exposure_pair_count": len(checked),
        "safe_exposure_pair_fraction": len(checked) / max(1, source_count - 1),
        "safe_exposure_support_pixel_count": int(np.sum(support)),
        "safe_exposure_support_pixel_count_per_channel": int(np.sum(support) * 3),
        "photometric_global_solver": True,
        "photometric_solver_irls_iterations": iterations,
        "photometric_solver_second_difference_lambda": smoothness,
        "photometric_solver_condition": maximum_condition,
        "photometric_log_residual_p95": float(np.percentile(np.abs(residuals), 95.0)),
        "photometric_edge_log_mad_p95": float(np.percentile(mads, 95.0)),
    }


def _graphcut_monotonic_owner(
    first: np.ndarray,
    second: np.ndarray,
    valid0: np.ndarray,
    valid1: np.ndarray,
    protected: np.ndarray,
    nominal_boundary: int,
    *,
    component_owner_constraints: Mapping[int, int] | None = None,
    owner_prior: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, int, int]:
    """Constrain GraphCut to whole protected components and validate its seam.

    The previous implementation split a protected component at the nominal
    boundary and then rewrote the result with a row-wise ``max``.  Here the
    GraphCut masks receive an all-or-nothing owner for every connected risk
    component.  Optional sequence preflight constraints are keyed by the
    local connected-component label and use ``0`` / ``1`` for first / second
    owner.  Its returned owner map is used directly only when it is a valid
    monotonic prefix; otherwise a safe unblended hard cut is allowed.
    """

    height, width = valid0.shape
    common = np.asarray(valid0, dtype=bool) & np.asarray(valid1, dtype=bool)
    protected = np.asarray(protected, dtype=bool) & common
    nominal_boundary = int(np.clip(nominal_boundary, 0, width - 1))
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        protected.astype(np.uint8), connectivity=8
    )
    constraints = {} if component_owner_constraints is None else dict(component_owner_constraints)
    for label, owner in constraints.items():
        if not isinstance(label, (int, np.integer)) or not 1 <= int(label) < labels_count:
            raise ValueError("Component owner constraint references an unknown label")
        if int(owner) not in {0, 1}:
            raise ValueError("Component owner constraints must select owner 0 or 1")
    lower = np.full(height, -1, dtype=np.int32)
    upper = np.full(height, width - 1, dtype=np.int32)
    force0 = np.zeros_like(common)
    force1 = np.zeros_like(common)
    if owner_prior is not None:
        prior = np.asarray(owner_prior)
        if prior.shape != common.shape:
            raise ValueError("Owner prior must match the GraphCut corridor")
        if not np.isin(prior, (-1, 0, 1)).all():
            raise ValueError("Owner prior values must be -1, 0, or 1")
        force0 |= common & (prior == 0)
        force1 |= common & (prior == 1)
        if np.any(force0 & force1):
            raise ValueError("Owner prior selects both sources for one pixel")
        for row in range(height):
            first_columns = np.flatnonzero(force0[row])
            second_columns = np.flatnonzero(force1[row])
            if first_columns.size:
                lower[row] = max(lower[row], int(first_columns[-1]))
            if second_columns.size:
                upper[row] = min(upper[row], int(second_columns[0]) - 1)
    # Preserve a real contribution from both adjacent strips whenever a safe
    # corridor exists.  GraphCut otherwise has no terminal costs in this small
    # two-image overlap and can legally choose the same image for every row,
    # silently erasing a narrow intermediate pose node.  These are single-pixel
    # terminal constraints at the outermost safe columns, not a restriction of
    # the 32--64 px seam search itself.  Rows occupied completely by one
    # protected component deliberately get no such anchors; assigning that
    # component as a whole is safer than slicing it at an artificial terminal.
    anchor0 = np.full(height, -1, dtype=np.int32)
    anchor1 = np.full(height, -1, dtype=np.int32)
    for label in range(1, labels_count):
        if int(stats[label, cv2.CC_STAT_AREA]) == 0:
            continue
        component = labels == label
        xs = np.flatnonzero(np.any(component, axis=0))
        if not xs.size:
            continue
        left, right = int(xs[0]), int(xs[-1])
        constrained_owner = constraints.get(label)
        if constrained_owner is not None:
            choose_first = int(constrained_owner) == 0
        elif right <= nominal_boundary:
            choose_first = True
        elif left > nominal_boundary:
            choose_first = False
        else:
            # For a component straddling the nominal centre, choose exactly
            # one side according to the smaller necessary seam displacement.
            choose_first = (right - nominal_boundary) <= (nominal_boundary - left + 1)
        rows = np.flatnonzero(np.any(component, axis=1))
        if choose_first:
            force0 |= component
            for row in rows:
                lower[row] = max(lower[row], int(np.flatnonzero(component[row])[-1]))
        else:
            force1 |= component
            for row in rows:
                upper[row] = min(upper[row], int(np.flatnonzero(component[row])[0]) - 1)
    if np.any(force0 & force1):
        raise RuntimeError("Sequence owner constraints conflict in a protected component")
    safe_common = common & ~protected
    for row in range(height):
        safe_columns = np.flatnonzero(safe_common[row])
        if safe_columns.size < 2:
            continue
        # A left terminal can only belong to the first image and a right
        # terminal only to the second.  Keep them strictly ordered so a one
        # pixel slit cannot manufacture an impossible seam.
        left_candidates = safe_columns[safe_columns <= nominal_boundary]
        right_candidates = safe_columns[safe_columns > nominal_boundary]
        if not left_candidates.size or not right_candidates.size:
            continue
        left_terminal = int(left_candidates[0])
        right_terminal = int(right_candidates[-1])
        if left_terminal >= right_terminal:
            continue
        anchor0[row] = left_terminal
        anchor1[row] = right_terminal
    if np.any(lower > upper):
        raise RuntimeError("No safe hard-owner corridor around a protected RGB component")

    masks = [valid0.astype(np.uint8) * 255, valid1.astype(np.uint8) * 255]
    masks[1][force0] = 0
    masks[0][force1] = 0
    for row in range(height):
        if anchor0[row] >= 0:
            masks[1][row, anchor0[row]] = 0
        if anchor1[row] >= 0:
            masks[0][row, anchor1[row]] = 0

    def owners_from_prefix(
        candidate: np.ndarray, *, enforce_terminals: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        candidate = np.asarray(candidate, dtype=bool)
        if enforce_terminals:
            for row in range(height):
                if anchor0[row] >= 0 and not candidate[row, anchor0[row]]:
                    raise RuntimeError("GraphCut discarded a first-owner terminal")
                if anchor1[row] >= 0 and candidate[row, anchor1[row]]:
                    raise RuntimeError("GraphCut discarded a second-owner terminal")
        owner0 = valid0 & (~valid1 | candidate)
        owner1 = valid1 & (~valid0 | ~candidate)
        union = valid0 | valid1
        if np.any(union & ~(owner0 | owner1)) or np.any(owner0 & owner1):
            raise RuntimeError("RGB hard-owner seam left an invalid or multiply-owned pixel")
        cuts = np.full(height, nominal_boundary, dtype=np.int32)
        boundary_guard_pixels = 0
        for row in range(height):
            indices = np.flatnonzero(common[row])
            if not indices.size:
                continue
            prefix = owner0[row, indices]
            false_indices = np.flatnonzero(~prefix)
            if false_indices.size and np.any(prefix[false_indices[0] + 1 :]):
                raise RuntimeError("GraphCut returned a non-monotonic hard-owner seam")
            cut = int(indices[np.flatnonzero(prefix)[-1]]) if np.any(prefix) else int(indices[0] - 1)
            if cut < lower[row] or cut > upper[row]:
                raise RuntimeError("GraphCut violated a protected-component owner constraint")
            cuts[row] = cut
            # A component can legitimately own the complete endpoint overlap.
            # With only one owner present there is no actual inter-source
            # boundary in this row, so it cannot violate the boundary guard.
            if np.any(prefix) and np.any(~prefix):
                for position in (cut, cut + 1):
                    if 0 <= position < width and protected[row, position]:
                        boundary_guard_pixels += 1
        split_count = 0
        for label in range(1, labels_count):
            component = labels == label
            if np.any(owner0 & component) and np.any(owner1 & component):
                split_count += 1
        if split_count:
            raise RuntimeError("A protected RGB component was split between hard owners")
        if boundary_guard_pixels:
            raise RuntimeError("A hard-owner boundary intersects the RGB risk guard")
        return owner0, owner1, cuts, split_count, boundary_guard_pixels

    graphcut_candidate: np.ndarray | None = None
    try:
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        work_masks = [
            np.ascontiguousarray(masks[0]),
            np.ascontiguousarray(masks[1]),
        ]
        output = finder.find(
            [
                np.ascontiguousarray(first, dtype=np.float32),
                np.ascontiguousarray(second, dtype=np.float32),
            ],
            [(0, 0), (0, 0)],
            work_masks,
        )
        result_masks = work_masks if output is None else output
        if len(result_masks) == 2:
            candidate = result_masks[0].get() if hasattr(result_masks[0], "get") else result_masks[0]
            candidate = np.asarray(candidate, dtype=np.uint8)
            if candidate.shape == valid0.shape:
                graphcut_candidate = candidate > 0
    except cv2.error:
        graphcut_candidate = None

    if graphcut_candidate is not None:
        try:
            owner0, owner1, cuts, split_count, boundary_guard = owners_from_prefix(
                graphcut_candidate, enforce_terminals=True
            )
            return owner0, owner1, cuts, True, 0, split_count, boundary_guard
        except RuntimeError:
            # Do not repair a non-monotonic GraphCut by taking a row-wise
            # maximum.  The only permitted fallback is a safe direct hard cut.
            pass

    hard_candidate = np.zeros_like(common)
    hard_rows = 0
    for row in range(height):
        indices = np.flatnonzero(common[row])
        if not indices.size:
            continue
        if lower[row] >= width - 1:
            hard_candidate[row] = True
            hard_rows += 1
            continue
        if upper[row] < 0:
            hard_rows += 1
            continue
        candidates = np.arange(max(0, lower[row]), min(width - 1, upper[row]) + 1)
        candidates = candidates[
            ~protected[row, candidates]
            & ~protected[row, np.minimum(candidates + 1, width - 1)]
        ]
        if not candidates.size:
            # A protected component may span the complete search corridor in
            # this row (for example a horizontal hose).  It can still remain
            # intact if the entire row is assigned to its chosen owner; no
            # inter-source boundary then crosses the guard.  Competing forced
            # owners remain a genuine no-safe-channel structural failure.
            if lower[row] >= 0 and upper[row] == width - 1:
                hard_candidate[row] = True
                hard_rows += 1
                continue
            if upper[row] < width - 1 and lower[row] < 0:
                hard_rows += 1
                continue
            raise RuntimeError("No safe direct hard cut exists outside the RGB risk guard")
        cut = int(candidates[np.argmin(np.abs(candidates - nominal_boundary))])
        hard_candidate[row] = np.arange(width) <= cut
        hard_rows += 1
    owner0, owner1, cuts, split_count, boundary_guard = owners_from_prefix(hard_candidate)
    return owner0, owner1, cuts, False, hard_rows, split_count, boundary_guard


def _blend_safe_pair_zone(
    first: np.ndarray,
    second: np.ndarray,
    common: np.ndarray,
    protected: np.ndarray,
    safe_wall: np.ndarray,
    owner0: np.ndarray,
    owner1: np.ndarray,
    cuts: np.ndarray,
    blend_width: int,
    levels: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, bool]:
    """Use complementary owner masks in one local safe-wall MultiBand pass."""

    direct = np.zeros_like(first)
    direct[owner0] = first[owner0]
    direct[owner1] = second[owner1]
    if blend_width < 2:
        raise ValueError("Safe RGB blend width must be at least two pixels")
    x = np.arange(first.shape[1], dtype=np.int32)[None, :]
    left_span = (int(blend_width) - 1) // 2
    right_span = int(blend_width) - left_span - 1
    zone = (
        common
        & ~protected
        & safe_wall
        & (x >= cuts[:, None] - left_span)
        & (x <= cuts[:, None] + right_span)
    )
    effective_levels = _adaptive_multiband_levels(blend_width, levels)
    if not np.any(zone):
        return direct, zone, 0, effective_levels, 0, 0, False
    # Keep the pyramid local, but make each source's mask begin with its own
    # hard owner.  The two dilations overlap only within safe wall support;
    # they are never the same full-support 50/50 mask.
    support_radius = max(1, 1 << max(0, effective_levels - 1))
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (support_radius * 2 + 1, support_radius * 2 + 1)
    )
    support = cv2.dilate(zone.astype(np.uint8), support_kernel) > 0
    support &= common & ~protected & safe_wall
    owner_radius = max(1, int(math.ceil(blend_width / 2.0)))
    owner_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (owner_radius * 2 + 1, owner_radius * 2 + 1)
    )
    mask0 = cv2.dilate((owner0 & safe_wall).astype(np.uint8), owner_kernel) > 0
    mask1 = cv2.dilate((owner1 & safe_wall).astype(np.uint8), owner_kernel) > 0
    mask0 &= support
    mask1 &= support
    zone &= mask0 & mask1
    if not np.any(zone):
        return (
            direct,
            zone,
            0,
            effective_levels,
            int(np.count_nonzero(mask0)),
            int(np.count_nonzero(mask1)),
            False,
        )
    if np.array_equal(mask0, mask1):
        # A tiny legal corridor (common in dense pose-node sequences) can make
        # both dilations cover the exact same support.  Feeding that pair would
        # recreate the forbidden 50/50 average, so retain the audited hard cut
        # and omit MultiBand for this pair instead of inventing a feather mask.
        return (
            direct,
            np.zeros_like(zone),
            0,
            effective_levels,
            int(np.count_nonzero(mask0)),
            int(np.count_nonzero(mask1)),
            False,
        )
    mask0_u8 = np.where(mask0, 255, 0).astype(np.uint8)
    mask1_u8 = np.where(mask1, 255, 0).astype(np.uint8)
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(effective_levels)
    blender.prepare((0, 0, first.shape[1], first.shape[0]))
    try:
        blender.feed(np.ascontiguousarray(first, dtype=np.int16), mask0_u8, (0, 0))
        blender.feed(np.ascontiguousarray(second, dtype=np.int16), mask1_u8, (0, 0))
        blended, output_mask = blender.blend(None, None)
    except cv2.error as exc:
        raise RuntimeError("Safe RGB local MultiBand blending failed") from exc
    if hasattr(blended, "get"):
        blended = blended.get()
    if hasattr(output_mask, "get"):
        output_mask = output_mask.get()
    blended = np.clip(np.asarray(blended), 0, 255).astype(np.uint8)
    output_mask = np.asarray(output_mask, dtype=np.uint8)
    if blended.shape != first.shape or output_mask.shape != common.shape:
        raise RuntimeError("Safe RGB local MultiBand returned an invalid output")
    if np.any(zone & (output_mask == 0)):
        raise RuntimeError("Safe RGB local MultiBand produced a zero-weight wedge")
    if np.any(zone & protected):
        raise RuntimeError("RGB risk protection reached a MultiBand zone")
    direct[zone] = blended[zone]
    return (
        direct,
        zone,
        int(np.count_nonzero(zone)),
        effective_levels,
        int(np.count_nonzero(mask0)),
        int(np.count_nonzero(mask1)),
        True,
    )


def _delta_e00(lab0: np.ndarray, lab1: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 for standard (not uint8-encoded) CIE Lab."""

    first = np.asarray(lab0, dtype=np.float64)
    second = np.asarray(lab1, dtype=np.float64)
    l1, a1, b1 = first[..., 0], first[..., 1], first[..., 2]
    l2, a2, b2 = second[..., 0], second[..., 1], second[..., 2]
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    mean_c = 0.5 * (c1 + c2)
    mean_c7 = np.power(mean_c, 7)
    g = 0.5 * (1.0 - np.sqrt(mean_c7 / (mean_c7 + 25.0**7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.mod(np.degrees(np.arctan2(b1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(b2, a2p)), 360.0)
    delta_l = l2 - l1
    delta_c = c2p - c1p
    hue_difference = h2p - h1p
    hue_difference = np.where(hue_difference > 180.0, hue_difference - 360.0, hue_difference)
    hue_difference = np.where(hue_difference < -180.0, hue_difference + 360.0, hue_difference)
    hue_difference = np.where((c1p * c2p) == 0.0, 0.0, hue_difference)
    delta_h = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(hue_difference / 2.0))
    mean_l = 0.5 * (l1 + l2)
    mean_cp = 0.5 * (c1p + c2p)
    mean_hp = 0.5 * (h1p + h2p)
    mean_hp = np.where(np.abs(h1p - h2p) > 180.0, mean_hp + 180.0, mean_hp)
    mean_hp = np.where((c1p * c2p) == 0.0, h1p + h2p, mean_hp)
    mean_hp = np.mod(mean_hp, 360.0)
    t = (
        1.0
        - 0.17 * np.cos(np.radians(mean_hp - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * mean_hp))
        + 0.32 * np.cos(np.radians(3.0 * mean_hp + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * mean_hp - 63.0))
    )
    delta_theta = 30.0 * np.exp(-np.square((mean_hp - 275.0) / 25.0))
    rc = 2.0 * np.sqrt(np.power(mean_cp, 7) / (np.power(mean_cp, 7) + 25.0**7))
    sl = 1.0 + (0.015 * np.square(mean_l - 50.0)) / np.sqrt(20.0 + np.square(mean_l - 50.0))
    sc = 1.0 + 0.045 * mean_cp
    sh = 1.0 + 0.015 * mean_cp * t
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc
    return np.sqrt(
        np.square(delta_l / sl)
        + np.square(delta_c / sc)
        + np.square(delta_h / sh)
        + rt * (delta_c / sc) * (delta_h / sh)
    )


def _bgr_to_cie_lab(image: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR to continuous CIE Lab for a quality measurement."""

    values = np.ascontiguousarray(np.asarray(image, dtype=np.float32) / 255.0)
    return cv2.cvtColor(values, cv2.COLOR_BGR2LAB)


def _robust_wall_patch_samples(
    lab0: np.ndarray,
    lab1: np.ndarray,
    samples: np.ndarray,
    *,
    tile_height: int = 64,
    tile_width: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local median wall colours without filtering any rendered pixels.

    The metric is intentionally computed from independently safe pixels in an
    64x4 neighbourhood.  It rejects isolated inverse-remap/JPEG sample noise
    while retaining an owner-boundary colour step: a global per-frame colour
    shift is coherent in both dimensions and therefore survives each local
    median.  This helper is only a quality measurement; it never feeds a
    blend, owner decision, or output image.
    """

    safe = np.asarray(samples, dtype=bool)
    if lab0.shape != lab1.shape or safe.shape != lab0.shape[:2]:
        raise ValueError("Wall quality samples must match both Lab images")
    height, width = safe.shape
    first_patches: list[np.ndarray] = []
    second_patches: list[np.ndarray] = []
    height_step = max(2, int(tile_height))
    width_step = max(2, int(tile_width))
    for y0 in range(0, height, height_step):
        y1 = min(height, y0 + height_step)
        for x0 in range(0, width, width_step):
            x1 = min(width, x0 + width_step)
            patch = safe[y0:y1, x0:x1]
            minimum = max(8, int(math.ceil(0.25 * patch.size)))
            if int(np.count_nonzero(patch)) < minimum:
                continue
            first_patches.append(np.median(lab0[y0:y1, x0:x1][patch], axis=0))
            second_patches.append(np.median(lab1[y0:y1, x0:x1][patch], axis=0))
    if len(first_patches) < 4:
        # Sparse wall support is still a valid photometric observation, but it
        # cannot form four independent tiles.  Report its robust whole-support
        # medians rather than reverting to individual JPEG/remap samples.
        return (
            np.median(lab0[safe], axis=0, keepdims=True),
            np.median(lab1[safe], axis=0, keepdims=True),
        )
    return np.asarray(first_patches), np.asarray(second_patches)


def _white_wall_pair_metrics(
    first: np.ndarray,
    second: np.ndarray,
    safe_wall: np.ndarray,
    cuts: np.ndarray,
) -> tuple[
    float | None,
    float | None,
    float | None,
    int,
    np.ndarray,
    np.ndarray,
]:
    """Measure corrected ΔL*/ΔE00 near a real hard-owner boundary."""

    x = np.arange(first.shape[1], dtype=np.int32)[None, :]
    boundary = np.abs(x - cuts[:, None]) <= 2
    samples = np.asarray(safe_wall, dtype=bool) & boundary
    # A pair can legitimately have no neutral wall right at a foreground
    # detour.  Its wider safe-wall support still measures colour continuity.
    if int(np.count_nonzero(samples)) < 32:
        samples = np.asarray(safe_wall, dtype=bool)
    count = int(np.count_nonzero(samples))
    if count < 32:
        return (
            None,
            None,
            None,
            count,
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    lab0 = _bgr_to_cie_lab(first)
    lab1 = _bgr_to_cie_lab(second)
    first_samples, second_samples = _robust_wall_patch_samples(lab0, lab1, samples)
    delta_l = first_samples[:, 0] - second_samples[:, 0]
    delta_e = _delta_e00(first_samples, second_samples)
    return (
        float(np.percentile(np.abs(delta_l), 95.0)),
        float(np.percentile(delta_e, 95.0)),
        float(np.median(delta_l)),
        count,
        np.asarray(np.abs(delta_l), dtype=np.float64),
        np.asarray(delta_e, dtype=np.float64),
    )


def _periodic_stripe_energy(values: Sequence[float]) -> float | None:
    """High-frequency residual energy after removing an exposure drift line."""

    samples = np.asarray(values, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size < 5:
        return None
    x = np.arange(samples.size, dtype=np.float64)
    trend = np.polyval(np.polyfit(x, samples, 1), x)
    residual = samples - trend
    smooth = np.convolve(np.pad(residual, (2, 2), mode="edge"), np.ones(5) / 5.0, mode="valid")
    return float(np.sqrt(np.mean(np.square(residual - smooth))))


def _write_owner_core(
    canvas: np.ndarray,
    valid_canvas: np.ndarray,
    owner_canvas: np.ndarray,
    contribution: PushbroomContribution,
    image: np.ndarray,
    left: float,
    right: float,
) -> None:
    x0 = max(contribution.x0, int(math.ceil(left)))
    x1 = min(contribution.x1, int(math.ceil(right)))
    if x1 <= x0:
        return
    source_slice = slice(x0 - contribution.x0, x1 - contribution.x0)
    destination = slice(x0, x1)
    usable = contribution.valid_mask[:, source_slice]
    if not np.any(usable):
        return
    source = image[:, source_slice]
    canvas_view = canvas[:, destination]
    valid_view = valid_canvas[:, destination]
    owner_view = owner_canvas[:, destination]
    canvas_view[usable] = source[usable]
    valid_view[usable] = True
    owner_view[usable] = contribution.source_index


def render_calibrated_rgb_pushbroom(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    *,
    config: Mapping[str, object] | CalibratedRGBPushbroomConfig | None = None,
    rgb_motions: Sequence[object] | None = None,
    motion_pixels_to_full_resolution: float = 1.0,
    multiband_levels: int = 3,
    quality_gate: bool = True,
) -> CalibratedRGBPushbroomResult:
    """Render real pose frames as a bounded-memory RGB-colour pushbroom.

    Every sample written to ``panorama`` originates from a source frame's RGB
    array.  Aligned depth is read only by the audited adjacent-corridor geometry
    assist for visibility, layer protection and an inverse RGB sampling field;
    it never contributes a colour, hole fill or pose update.
    """

    settings = (
        config
        if isinstance(config, CalibratedRGBPushbroomConfig)
        else CalibratedRGBPushbroomConfig.from_mapping(config)
    )
    if len(frames) != len(poses) or not 2 <= len(frames) <= settings.max_pose_count:
        raise ValueError("Calibrated RGB pushbroom requires 2-160 aligned frames/poses")
    if not 1 <= int(multiband_levels) <= 3:
        raise ValueError("Calibrated RGB pushbroom MultiBand levels must be in [1, 3]")
    checked_poses = [validate_camera_to_world(pose) for pose in poses]
    scale = estimate_rgb_motion_pixels_per_mm(
        frames,
        checked_poses,
        calibration,
        settings,
        rgb_motions=rgb_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
    )
    layout = build_calibrated_rgb_pushbroom_layout(
        [frame.frame_id for frame in frames], checked_poses, calibration, scale, settings
    )
    residual_support_bounds = _residual_support_bounds(layout)
    strip_paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="g305-rgb-pushbroom-") as temporary:
        root = Path(temporary)
        # Phase 0/preview pass: calibrated low-resolution RGB evidence is
        # spooled independently and never enters the formal output canvas.
        analysis_renderer = CalibratedRGBPushbroomRenderer(
            layout, calibration, checked_poses
        )
        preview_scale = _preview_canvas_scale(layout, settings.residual_alignment)
        # Risk/owner trigger evidence cannot inherit the much smaller global
        # residual-analysis width: a one-pixel foreground edge can disappear
        # there yet close a full-resolution 64 px owner corridor later.  A
        # second *analysis-only* stream uses the already-closed 0.50--0.75
        # geometry validation scale, retains just an adjacent pair, and never
        # reaches output colour or a pose/residual model.
        geometry_trigger_preview_scale = float(
            settings.geometry_assisted_seam.flow_validation_preview_scale
        )
        geometry_trigger_renderer = CalibratedRGBPushbroomRenderer(
            layout, calibration, checked_poses
        )
        preview_evidence: list[object] = []
        preview_boundary_geometry: list[dict[str, object]] = []
        preview_pair_origins: list[tuple[float, float]] = []
        preview_evidence_pixels = 0
        preview_evidence_limit_pixels = int(
            math.floor(
                settings.residual_alignment.maximum_evidence_megapixels * 1_000_000.0
            )
        )
        previous_preview_path: Path | None = None
        previous_geometry_trigger_preview: PreviewContribution | None = None
        for index, frame in enumerate(frames):
            preview = analysis_renderer.render_preview_frame(
                frame, index, canvas_scale=preview_scale
            )
            geometry_trigger_preview = geometry_trigger_renderer.render_preview_frame(
                frame,
                index,
                canvas_scale=geometry_trigger_preview_scale,
            )
            path = root / f"preview-{index:04d}.npz"
            _save_preview(path, preview)
            if previous_preview_path is None:
                previous_preview_path = path
                previous_geometry_trigger_preview = geometry_trigger_preview
                continue
            if previous_geometry_trigger_preview is None:
                raise RuntimeError("Geometry trigger preview stream lost its first source")
            first_preview = _load_preview(previous_preview_path)
            second_preview = _load_preview(path)
            (
                preview_left,
                _preview_right,
                preview_rgb0,
                preview_rgb1,
                preview_valid0,
                preview_valid1,
                preview_map0_x,
                preview_map0_y,
                preview_map1_x,
                preview_map1_y,
            ) = _preview_pair_overlap(first_preview, second_preview)
            (
                geometry_preview_left,
                _geometry_preview_right,
                geometry_preview_rgb0,
                geometry_preview_rgb1,
                geometry_preview_valid0,
                geometry_preview_valid1,
                _geometry_preview_map0_x,
                _geometry_preview_map0_y,
                _geometry_preview_map1_x,
                _geometry_preview_map1_y,
            ) = _preview_pair_overlap(
                previous_geometry_trigger_preview, geometry_trigger_preview
            )
            evidence_pixels = int(preview_rgb0.shape[0]) * int(preview_rgb0.shape[1])
            preview_evidence_pixels += evidence_pixels
            if preview_evidence_pixels > preview_evidence_limit_pixels:
                raise RuntimeError(
                    "RGB residual preview evidence exceeds the formal bounded-memory "
                    "pixel limit"
                )
            pair_index = index - 1
            preview_pair_origins.append((float(preview_left), float(preview_scale)))
            evidence = extract_pair_evidence(
                first_rgb=preview_rgb0,
                second_rgb=preview_rgb1,
                first_valid=preview_valid0,
                second_valid=preview_valid1,
                first_inverse_map=(preview_map0_x, preview_map0_y),
                second_inverse_map=(preview_map1_x, preview_map1_y),
                intrinsics=calibration,
                first_camera_to_world=checked_poses[pair_index],
                second_camera_to_world=checked_poses[index],
                pair_index=pair_index,
                frame_ids=(first_preview.frame_id, second_preview.frame_id),
                config=settings.residual_alignment,
            )
            preview_evidence.append(evidence)
            preview_nominal_boundary = int(
                round(
                    layout.owner_boundaries_x[pair_index]
                    * geometry_trigger_preview_scale
                    - geometry_preview_left
                )
            )
            preview_trigger_probe = _preview_geometry_trigger_boundary_audit(
                geometry_preview_rgb0,
                geometry_preview_rgb1,
                geometry_preview_valid0,
                geometry_preview_valid1,
                nominal_boundary=preview_nominal_boundary,
                preview_scale=geometry_trigger_preview_scale,
            )
            preview_boundary_geometry.append(
                {
                    "pair_index": pair_index,
                    "frame_ids": [first_preview.frame_id, second_preview.frame_id],
                    **measure_owner_boundary_geometry(
                        evidence,
                        boundary_x=(
                            layout.owner_boundaries_x[pair_index] * preview_scale
                            - preview_left
                        ),
                    ),
                    **preview_trigger_probe,
                }
            )
            # Only the immediately adjacent preview pair is retained on disk;
            # dense evidence itself is bounded globally by the explicit pixel
            # limit above and contains no output RGB strip.
            previous_preview_path.unlink(missing_ok=True)
            previous_preview_path = path
            previous_geometry_trigger_preview = geometry_trigger_preview
        if previous_preview_path is not None:
            previous_preview_path.unlink(missing_ok=True)
        preview_working_set = {
            "preview_evidence_pixel_count": preview_evidence_pixels,
            "preview_evidence_hard_limit_pixels": preview_evidence_limit_pixels,
            "preview_evidence_estimated_bytes": preview_evidence_pixels * 64,
            "preview_evidence_storage": "bounded_in_memory_analysis_only",
            "preview_streaming_maximum_resident_previews": 2,
            "geometry_trigger_preview_scale": geometry_trigger_preview_scale,
            "geometry_trigger_preview_remap_count": (
                geometry_trigger_renderer.analysis_preview_remap_count
            ),
            "geometry_trigger_preview_storage": "bounded_adjacent_analysis_only",
            "geometry_trigger_preview_streaming_maximum_resident_previews": 2,
        }
        # RGB previews are retained as evidence only.  Never select a global
        # RGB SE(2) residual: the validated RGB-D camera_to_world chain is the
        # one and only global geometry, and local geometry assistance below is
        # limited to accepted adjacent seam corridors.
        residual_alignment = _identity_residual_alignment_result(
            layout,
            calibration,
            settings.residual_alignment,
            preview_evidence,
            preview_remap_count=analysis_renderer.analysis_preview_remap_count,
            support_bounds=residual_support_bounds,
            working_set_additional={
                **preview_working_set,
                "global_rgb_residual_model": "forbidden_identity_only",
            },
        )
        # Geometry aid is considered only after immutable RGB preview evidence
        # exists.  It creates no colour and does not modify a pose; accepted
        # meshes simply participate in the same source inverse map that will
        # be used for that source's one RGB output remap.
        geometry_warps, planned_geometry = _plan_geometry_assisted_seams(
            frames=frames,
            poses=checked_poses,
            calibration=calibration,
            layout=layout,
            residual_warps=residual_alignment.source_warps,
            preview_evidence=preview_evidence,
            preview_boundary_geometry=preview_boundary_geometry,
            preview_pair_origins=preview_pair_origins,
            settings=settings.geometry_assisted_seam,
            root=root,
        )
        # Final owner decisions add scalar closure evidence below.  The mesh
        # decision itself remains immutable; replacing this compact plan never
        # alters a rendered coordinate map or creates a second remap.
        geometry_plans = list(planned_geometry)
        # The only non-identity coordinate correction that can reach the
        # one-map/one-remap path is an accepted local seam mesh.  It is
        # separately held-out audited and has an identity branch outside its
        # flow-consistent same-layer cells.
        renderer = CalibratedRGBPushbroomRenderer(
            layout,
            calibration,
            checked_poses,
            residual_warps=residual_alignment.source_warps,
            geometry_warps=geometry_warps,
        )
        for index, frame in enumerate(frames):
            contribution = renderer.render_frame(frame, index)
            path = root / f"{index:04d}.npz"
            _save_contribution(path, contribution)
            strip_paths.append(path)

        photometric_edges: list[_PhotometricEdge | None] = []
        raw_risk_paths: list[Path] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            (
                full_left,
                full_right,
                rgb0,
                rgb1,
                valid0,
                valid1,
            ) = _pair_overlap(first, second)
            geometry_protected, _geometry_active = _geometry_pair_masks(
                geometry_plans[index],
                left=full_left,
                right=full_right,
                height=layout.canvas_height,
            )
            geometry_out_of_scope_first = _geometry_out_of_scope_first_mask(
                geometry_plans,
                index,
                left=full_left,
                right=full_right,
                height=layout.canvas_height,
            )
            # A mesh belongs only to the adjacent pair that validated it.
            # Do not let source ``index`` reuse a prior pair's correction as
            # the first member of this pair's photometric/owner evidence.
            valid0 = np.ascontiguousarray(valid0 & ~geometry_out_of_scope_first)
            raw_risk = _rgb_risk_details(rgb0, rgb1, valid0, valid1)
            # The safe-wall estimator excludes the complete protection band,
            # including the largest permitted local pyramid footprint.
            preliminary_guard, _, _ = _risk_guard(
                raw_risk,
                valid0 & valid1,
                blend_width=8,
                requested_levels=multiband_levels,
                geometry_protected=geometry_protected,
            )
            photometric_edges.append(
                _safe_wall_log_relation(
                    rgb0,
                    rgb1,
                    valid0 & valid1,
                    preliminary_guard,
                    low_gradient_quantile=settings.scale_low_gradient_quantile,
                )
            )
            (
                _seam_left,
                _seam_right,
                seam_rgb0,
                seam_rgb1,
                seam_valid0,
                seam_valid1,
            ) = _select_seam_search_corridor(
                full_left,
                full_right,
                rgb0,
                rgb1,
                valid0,
                valid1,
                nominal_boundary_x=layout.owner_boundaries_x[index],
                requested_width=settings.seam_search_width_pixels,
                allow_short_endpoint_corridor=index in {0, len(frames) - 2},
            )
            # Store the raw risk as evaluated in the exact owner-search
            # corridor.  A broad photometric overlap may join two foreground
            # regions outside that corridor; importing that remote connection
            # into GraphCut would falsely make a local safe path impossible.
            seam_raw_risk = _rgb_risk_details(
                seam_rgb0, seam_rgb1, seam_valid0, seam_valid1
            )
            raw_risk_path = root / f"risk-{index:04d}.npz"
            np.savez_compressed(
                raw_risk_path,
                mask=seam_raw_risk.mask,
                seed_mask=seam_raw_risk.seed_mask,
                structural_seed_mask=seam_raw_risk.structural_seed_mask,
                edge_offset_p95=np.asarray(
                    seam_raw_risk.edge_offset_p95, dtype=np.float64
                ),
                seed_pixel_count=np.asarray(
                    seam_raw_risk.seed_pixel_count, dtype=np.int32
                ),
                component_count=np.asarray(
                    seam_raw_risk.component_count, dtype=np.int32
                ),
            )
            raw_risk_paths.append(raw_risk_path)
        gains, exposure_metrics = _solve_global_linear_rgb_gains(
            len(frames), photometric_edges
        )

        # Rewrite each temporary calibrated strip after one linear-light
        # three-channel compensation.  Subsequent owner and blending passes
        # only consume these corrected strips, never compound a gain.
        for index, path in enumerate(strip_paths):
            contribution = _load_contribution(path)
            _save_contribution(
                path,
                PushbroomContribution(
                    source_index=contribution.source_index,
                    frame_id=contribution.frame_id,
                    x0=contribution.x0,
                    rgb=_apply_linear_bgr_gain(contribution.rgb, gains[index]),
                    valid_mask=contribution.valid_mask,
                ),
            )

        # Compare the same preselected safe-wall samples before and after the
        # single global gain pass.  This makes the stripe-energy audit an
        # apples-to-apples frame-sequence measurement rather than mixing a
        # broad photometric support with a later, foreground-detoured seam.
        raw_stripe_samples = [
            edge.raw_signed_l_delta
            for edge in photometric_edges
            if edge is not None
        ]
        corrected_stripe_samples: list[float] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            edge = photometric_edges[index]
            if edge is None or edge.safe_mask is None:
                raise RuntimeError("Global RGB photometric audit lost safe-wall support")
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            _, _, rgb0, rgb1, _, _ = _pair_overlap(first, second)
            safe = np.asarray(edge.safe_mask, dtype=bool)
            if safe.shape != rgb0.shape[:2] or not np.any(safe):
                raise RuntimeError("Global RGB photometric audit has an invalid safe-wall mask")
            lab0 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2LAB).astype(np.float32)
            corrected_stripe_samples.append(
                float(np.median((lab0[:, :, 0] - lab1[:, :, 0])[safe]) * (100.0 / 255.0))
            )

        # Preflight all pair-local protected components before any GraphCut is
        # permitted to write a final owner.  This pass only loads two corrected
        # strips at a time; it produces constraints/track summaries, not an
        # output owner image or an extra source remap.
        preflight_fragments_by_pair: list[tuple[ProtectedComponentFragment, ...]] = []
        # Every adjacent search corridor may need a strictly RGB pair-level
        # owner if component topology proves that no monotonic GraphCut seam
        # exists.  Geometry-triggered pairs still retain their additional
        # depth-protection audit, but a non-geometry foreground component must
        # not turn a safe hard cut into a sequence failure merely because it
        # cannot be split monotonically.
        geometry_pair_hard_owner_options: dict[int, tuple[int, ...]] = {}
        geometry_neighbourhood_pairs: set[int] = set()
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            (
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
            ) = _pair_overlap(first, second)
            geometry_out_of_scope_first = _geometry_out_of_scope_first_mask(
                geometry_plans,
                index,
                left=full_left,
                right=full_right,
                height=layout.canvas_height,
            )
            full_valid0 = np.ascontiguousarray(
                full_valid0 & ~geometry_out_of_scope_first
            )
            left, right, image0, image1, valid0, valid1 = _select_seam_search_corridor(
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
                nominal_boundary_x=layout.owner_boundaries_x[index],
                requested_width=settings.seam_search_width_pixels,
                allow_short_endpoint_corridor=index in {0, len(frames) - 2},
            )
            geometry_protected, _geometry_active = _geometry_pair_masks(
                geometry_plans[index],
                left=left,
                right=right,
                height=layout.canvas_height,
            )
            with np.load(raw_risk_paths[index], allow_pickle=False) as stored_risk:
                raw_risk_mask = np.asarray(stored_risk["mask"], dtype=np.uint8)
                raw_risk_seed = np.asarray(stored_risk["seed_mask"], dtype=bool)
                raw_risk_structural_seed = np.asarray(
                    stored_risk["structural_seed_mask"], dtype=bool
                )
                raw_edge_offset = float(stored_risk["edge_offset_p95"])
            if (
                raw_risk_seed.shape != image0.shape[:2]
                or raw_risk_structural_seed.shape != image0.shape[:2]
            ):
                raise RuntimeError("Stored raw RGB-risk seed shape is inconsistent")
            risk = _rgb_risk_details(
                image0,
                image1,
                valid0,
                valid1,
                supplied_risk_mask=raw_risk_mask,
            )
            risk = _RGBRiskDetails(
                mask=risk.mask,
                seed_mask=risk.seed_mask,
                structural_seed_mask=risk.structural_seed_mask,
                edge_offset_p95=max(raw_edge_offset, risk.edge_offset_p95),
                seed_pixel_count=risk.seed_pixel_count,
                component_count=risk.component_count,
            )
            owner_width = min(
                layout.owner_right_x[index] - layout.owner_left_x[index],
                layout.owner_right_x[index + 1] - layout.owner_left_x[index + 1],
            )
            blend_width = int(np.clip(math.floor(0.20 * owner_width), 2, 8))
            guard, _, _ = _risk_guard(
                risk,
                valid0 & valid1,
                blend_width,
                multiband_levels,
                geometry_protected=geometry_protected,
            )
            # A local warp belongs to one pair only, but the source it uses is
            # shared with either immediate neighbour.  A protected component
            # can therefore make the sequence-level owner constraint surface
            # one corridor away.  Regardless of whether geometry triggered,
            # pre-authorise a hard-owner fallback only when one unmodified RGB
            # source covers the entire valid union.
            geometry_neighbourhood = (
                geometry_plans[index].triggered
                or (
                    index > 0
                    and geometry_plans[index - 1].triggered
                )
                or (
                    index + 1 < len(geometry_plans)
                    and geometry_plans[index + 1].triggered
                )
            )
            if geometry_neighbourhood:
                geometry_neighbourhood_pairs.add(index)
            options = _pair_level_hard_owner_options(valid0, valid1)
            if options:
                geometry_pair_hard_owner_options[index] = options
            preflight_fragments_by_pair.append(
                extract_protected_component_fragments(
                    guard,
                    valid0,
                    valid1,
                    pair_index=index,
                    global_x0=left,
                    nominal_boundary_x=layout.owner_boundaries_x[index],
                )
            )
        # v2 foreground planning is deliberately separate from the renderer:
        # it turns protected components into auditable spans/handoff candidates
        # and can only add a complete-coverage hard owner.  This capture is an
        # aligned-depth legacy session, so its RGB-risk foreground remains
        # IMAGE_REGION owner-only unless a future capture supplies the raw
        # footprint and bidirectional-visibility evidence required to promote
        # a DEPTH_OBSERVED span.
        foreground_fragments = build_foreground_fragments(
            preflight_fragments_by_pair,
            frame_ids=[frame.frame_id for frame in frames],
            geometry_modes=(GeometryMode.IMAGE_REGION,) * len(
                preflight_fragments_by_pair
            ),
        )
        foreground_owner_plan = plan_foreground_owners(foreground_fragments)
        if not foreground_owner_plan.accepted:
            raise RuntimeError(
                "Foreground segment owner planning failed: "
                + str(foreground_owner_plan.structural_failure_reason)
            )
        segment_locked_constraints = [
            dict(constraints)
            for constraints in foreground_owner_plan.component_owner_constraints
        ]
        preflight_fragments_for_solver = list(preflight_fragments_by_pair)
        pair_level_hard_owner_pairs: set[int] = set()
        pair_level_hard_owner_reasons: dict[int, str] = {}
        component_hard_owner_fallback_pairs: set[int] = set()
        component_hard_owner_fallbacks: dict[int, dict[int, int]] = {}
        component_hard_owner_fallback_reasons: dict[int, str] = {}
        while True:
            sequence_owner_preflight = preflight_sequence_owners(
                preflight_fragments_for_solver,
                locked_component_owner_constraints=segment_locked_constraints,
            )
            if sequence_owner_preflight.accepted:
                break
            failed_pair = _preflight_failure_pair_index(
                sequence_owner_preflight.structural_failure_reason
            )
            if failed_pair is None:
                raise RuntimeError(
                    "RGB sequence owner preflight failed: "
                    + str(sequence_owner_preflight.structural_failure_reason)
                )
            # A conflict can surface between adjacent depth-assisted pairs,
            # but it can also arise from an ordinary RGB foreground component
            # (for example a hose crossing the nominal centreline).  First
            # retain as much normal owner space as possible by assigning each
            # complete component to one source.  This preserves the RGB risk
            # guard; it is not a blend or a relaxed component split.
            candidates = [failed_pair, failed_pair - 1, failed_pair + 1]
            component_fallback_pair = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate not in component_hard_owner_fallback_pairs
                    and 0 <= candidate < len(preflight_fragments_by_pair)
                ),
                None,
            )
            if component_fallback_pair is not None:
                forced_components = _force_monotonic_component_owners(
                    preflight_fragments_by_pair[component_fallback_pair]
                )
                component_hard_owner_fallback_pairs.add(component_fallback_pair)
                if forced_components is not None:
                    replacement, owners = forced_components
                    preflight_fragments_for_solver[component_fallback_pair] = replacement
                    segment_locked_constraints[component_fallback_pair].update(owners)
                    component_hard_owner_fallbacks[component_fallback_pair] = owners
                    component_hard_owner_fallback_reasons[component_fallback_pair] = (
                        (
                            "geometry_protection_monotonic_component_owner:"
                            if component_fallback_pair in geometry_neighbourhood_pairs
                            else "rgb_protection_monotonic_component_owner:"
                        )
                        + f"{failed_pair}"
                    )
                    continue
            fallback_pair = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate not in pair_level_hard_owner_pairs
                    and candidate in geometry_pair_hard_owner_options
                ),
                None,
            )
            if fallback_pair is None:
                raise RuntimeError(
                    "RGB sequence owner preflight failed: "
                    + str(sequence_owner_preflight.structural_failure_reason)
                )
            # Neither a geometry-protected nor an ordinary RGB-protected
            # component may be weakened or blended across.  Use the declared
            # fully covered single-source fallback instead.
            pair_level_hard_owner_pairs.add(fallback_pair)
            pair_level_hard_owner_reasons[fallback_pair] = (
                (
                    "geometry_protection_nonmonotonic_topology:"
                    if fallback_pair in geometry_neighbourhood_pairs
                    else "rgb_protection_nonmonotonic_topology:"
                )
                + f"{failed_pair}"
            )
            preflight_fragments_for_solver[fallback_pair] = ()
            segment_locked_constraints[fallback_pair] = {}
        pair_level_hard_owners = _resolve_pair_level_hard_owners(
            tuple(pair_level_hard_owner_pairs),
            geometry_pair_hard_owner_options,
        )
        runtime_pair_level_hard_owners: dict[int, int] = {}
        runtime_pair_level_hard_owner_reasons: dict[int, str] = {}
        residual_alignment = ResidualAlignmentResult(
            selected_model=residual_alignment.selected_model,
            source_warps=residual_alignment.source_warps,
            pair_evidence=residual_alignment.pair_evidence,
            held_out_metrics_before=residual_alignment.held_out_metrics_before,
            held_out_metrics_after=residual_alignment.held_out_metrics_after,
            component_tracks=sequence_owner_preflight.component_tracks,
            topology_audit=residual_alignment.topology_audit,
            working_set_audit={
                **residual_alignment.working_set_audit,
                "sequence_owner_preflight": sequence_owner_preflight.as_dict(),
                "foreground_segment_owner_plan": {
                    **foreground_owner_plan.as_dict(),
                    "compiled_component_owner_constraints": [
                        {
                            str(label): int(owner)
                            for label, owner in constraints.items()
                        }
                        for constraints in segment_locked_constraints
                    ],
                },
                "geometry_pair_level_hard_owners": {
                    str(index): int(owner)
                    for index, owner in pair_level_hard_owners.items()
                },
                "geometry_pair_level_hard_owner_reasons": dict(
                    pair_level_hard_owner_reasons
                ),
            },
        )

        canvas = np.zeros(
            (layout.canvas_height, layout.canvas_width, 3), dtype=np.uint8
        )
        valid_canvas = np.zeros((layout.canvas_height, layout.canvas_width), dtype=bool)
        owner_canvas = np.full(
            (layout.canvas_height, layout.canvas_width), -1, dtype=np.int16
        )
        for index, path in enumerate(strip_paths):
            contribution = _load_contribution(path)
            _write_owner_core(
                canvas,
                valid_canvas,
                owner_canvas,
                contribution,
                contribution.rgb,
                layout.owner_left_x[index],
                layout.owner_right_x[index],
            )

        blend_pixels = 0
        maximum_blend_risk = 0.0
        hard_cut_pairs = 0
        graphcut_pairs = 0
        protected_split_components = 0
        owner_boundary_guard_pixels = 0
        geometry_protected_pixels = 0
        geometry_active_pixels = 0
        geometry_out_of_scope_first_pixels = 0
        geometry_blend_pixels = 0
        geometry_late_owner_closure_pairs = 0
        white_wall_l_samples: list[np.ndarray] = []
        white_wall_delta_e00_samples: list[np.ndarray] = []
        actual_search_widths: list[int] = []
        applied_multiband_levels: list[int] = []
        pair_metadata: list[dict[str, object]] = []
        for index, (first_path, second_path) in enumerate(
            zip(strip_paths[:-1], strip_paths[1:], strict=True)
        ):
            first, second = _load_contribution(first_path), _load_contribution(second_path)
            (
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
            ) = _pair_overlap(first, second)
            geometry_out_of_scope_first = _geometry_out_of_scope_first_mask(
                geometry_plans,
                index,
                left=full_left,
                right=full_right,
                height=layout.canvas_height,
            )
            full_valid0 = np.ascontiguousarray(
                full_valid0 & ~geometry_out_of_scope_first
            )
            left, right, image0, image1, valid0, valid1 = _select_seam_search_corridor(
                full_left,
                full_right,
                full_image0,
                full_image1,
                full_valid0,
                full_valid1,
                nominal_boundary_x=layout.owner_boundaries_x[index],
                requested_width=settings.seam_search_width_pixels,
                allow_short_endpoint_corridor=index in {0, len(frames) - 2},
            )
            geometry_protected, geometry_active = _geometry_pair_masks(
                geometry_plans[index],
                left=left,
                right=right,
                height=layout.canvas_height,
            )
            geometry_out_of_scope_first = _geometry_out_of_scope_first_mask(
                geometry_plans,
                index,
                left=left,
                right=right,
                height=layout.canvas_height,
            )
            with np.load(raw_risk_paths[index], allow_pickle=False) as stored_risk:
                raw_risk_mask = np.asarray(stored_risk["mask"], dtype=np.uint8)
                raw_risk_seed = np.asarray(stored_risk["seed_mask"], dtype=bool)
                raw_risk_structural_seed = np.asarray(
                    stored_risk["structural_seed_mask"], dtype=bool
                )
                raw_edge_offset = float(stored_risk["edge_offset_p95"])
            if (
                raw_risk_seed.shape != image0.shape[:2]
                or raw_risk_structural_seed.shape != image0.shape[:2]
            ):
                raise RuntimeError("Stored raw RGB-risk seed shape is inconsistent")
            # Re-evaluate risk after the global linear-RGB correction while
            # retaining every raw risk component as an irreversible guard.
            post_gain_risk = _rgb_risk_details(image0, image1, valid0, valid1)
            risk = _rgb_risk_details(
                image0,
                image1,
                valid0,
                valid1,
                supplied_risk_mask=raw_risk_mask,
            )
            risk = _RGBRiskDetails(
                mask=risk.mask,
                # The full guard retains every pre-gain risk component, but
                # the geometry/late-closure topology must remain the union of
                # true pre/post-gain *seeds*, never the dilated guard itself.
                seed_mask=np.ascontiguousarray(post_gain_risk.seed_mask | raw_risk_seed),
                structural_seed_mask=np.ascontiguousarray(
                    post_gain_risk.structural_seed_mask | raw_risk_structural_seed
                ),
                edge_offset_p95=max(raw_edge_offset, risk.edge_offset_p95),
                seed_pixel_count=int(
                    np.count_nonzero(post_gain_risk.seed_mask | raw_risk_seed)
                ),
                component_count=risk.component_count,
            )
            owner_width = min(
                layout.owner_right_x[index] - layout.owner_left_x[index],
                layout.owner_right_x[index + 1] - layout.owner_left_x[index + 1],
            )
            # This is a total blend-zone width, deliberately not a +/- radius:
            # using the latter would place roughly 40% of every narrow owner
            # strip inside a pyramid.  The capped 2--8 px zone stays below the
            # 20% global budget while retaining the requested adaptive rule.
            blend_width = int(np.clip(math.floor(0.20 * owner_width), 2, 8))
            guard, guard_radius, guard_component_count = _risk_guard(
                risk,
                valid0 & valid1,
                blend_width,
                multiband_levels,
                geometry_protected=geometry_protected,
            )
            nominal = int(round(layout.owner_boundaries_x[index])) - left
            # Endpoint outward half-FOV ownership can leave the midpoint one
            # pixel beyond the overlap support.  The nearest real corridor
            # column is the only valid nominal reference in that case.
            nominal = int(np.clip(nominal, 0, image0.shape[1] - 1))
            safe_wall = _safe_white_wall_mask(
                image0,
                image1,
                valid0 & valid1,
                guard,
                low_gradient_quantile=settings.scale_low_gradient_quantile,
            )
            pair_level_hard_owner = pair_level_hard_owners.get(index)
            if pair_level_hard_owner is None:
                try:
                    (
                        owner0,
                        owner1,
                        cuts,
                        used_graphcut,
                        hard_rows,
                        split_count,
                        boundary_guard_count,
                    ) = _graphcut_monotonic_owner(
                        image0,
                        image1,
                        valid0,
                        valid1,
                        guard,
                        nominal,
                        component_owner_constraints=(
                            sequence_owner_preflight.component_owner_constraints[index]
                        ),
                    )
                except RuntimeError as exc:
                    # Component-level ownership is preferred because it leaves
                    # every safe background pixel under the normal GraphCut
                    # and white-wall path.  If its exact guard admits no
                    # physical hard boundary, use a whole-corridor source only
                    # when it covers every valid RGB pixel; otherwise fail.
                    options = geometry_pair_hard_owner_options.get(index, ())
                    if not options:
                        raise RuntimeError(
                            "No structurally safe RGB hard-owner seam for adjacent "
                            f"pair {index} ({first.frame_id}, {second.frame_id})"
                        ) from exc
                    pair_level_hard_owner = int(options[0])
                    runtime_pair_level_hard_owners[index] = pair_level_hard_owner
                    runtime_pair_level_hard_owner_reasons[index] = (
                        "geometry_component_owner_no_safe_boundary"
                        if index in geometry_neighbourhood_pairs
                        else "rgb_component_owner_no_safe_boundary"
                    )
            if pair_level_hard_owner is not None:
                owner0, owner1, cuts = _pair_level_hard_owner_masks(
                    valid0, valid1, pair_level_hard_owner
                )
                required = valid0 | valid1
                pair_image = np.zeros_like(image0)
                pair_image[owner0] = image0[owner0]
                pair_image[owner1] = image1[owner1]
                blend_zone = np.zeros_like(required)
                pair_blend_pixels = 0
                pair_levels = 0
                mask0_pixels = 0
                mask1_pixels = 0
                masks_distinct = True
                used_graphcut = False
                hard_rows = int(required.shape[0])
                split_count = 0
                boundary_guard_count = 0
            else:
                (
                    pair_image,
                    blend_zone,
                    pair_blend_pixels,
                    pair_levels,
                    mask0_pixels,
                    mask1_pixels,
                    masks_distinct,
                ) = _blend_safe_pair_zone(
                    image0,
                    image1,
                    valid0 & valid1,
                    guard,
                    safe_wall,
                    owner0,
                    owner1,
                    cuts,
                    blend_width,
                    multiband_levels,
                )
            final_boundary = _boundary_rgb_risk_audit(
                risk.structural_seed_mask,
                valid0 & valid1,
                nominal_boundary=nominal,
                half_width_pixels=2,
            )
            boundary_x = np.arange(image0.shape[1], dtype=np.int32)[None, :]
            final_guard = (
                (risk.mask > 0)
                & (valid0 & valid1)
                & (np.abs(boundary_x - nominal) <= 2)
            )
            final_common_rows = int(final_boundary["common_row_count"])
            final_full_height_hard_cut = bool(
                pair_level_hard_owner is not None
                or (
                    final_common_rows > 0
                    and not used_graphcut
                    and int(hard_rows) >= final_common_rows
                )
            )
            final_high_rgb_risk = bool(
                final_boundary["boundary_high_rgb_risk"]
            )
            late_post_gain_owner_closure = bool(
                final_high_rgb_risk and not geometry_plans[index].triggered
            )
            final_owner_audit = {
                "policy": "final_nominal_owner_boundary_rgb_risk_and_hard_cut_closure",
                "nominal_boundary_x": int(final_boundary["boundary_nominal_x"]),
                "nominal_boundary_half_width_pixels": int(
                    final_boundary["boundary_half_width_pixels"]
                ),
                "final_nominal_boundary_risk_pixel_count": int(
                    final_boundary["boundary_risk_pixel_count"]
                ),
                "final_nominal_boundary_guard_pixel_count": int(
                    np.count_nonzero(final_guard)
                ),
                "final_nominal_boundary_risk_row_count": int(
                    final_boundary["boundary_risk_row_count"]
                ),
                "final_boundary_common_row_count": int(
                    final_boundary["boundary_common_row_count"]
                ),
                "final_common_row_count": final_common_rows,
                "final_hard_cut_row_count": int(hard_rows),
                "final_full_height_hard_cut": final_full_height_hard_cut,
                "final_graphcut_used": bool(used_graphcut),
                "final_pair_level_hard_owner": bool(
                    pair_level_hard_owner is not None
                ),
                "late_post_gain_owner_closure": late_post_gain_owner_closure,
            }
            plan = geometry_plans[index]
            # Exposure is estimated only after preview geometry planning.  A
            # post-gain *topology-qualified raw seed* can therefore appear
            # where the immutable pre-gain preview was clean.  It cannot
            # retroactively install a mesh without a second source remap, so
            # retain the final guarded owner decision (the full dilated guard
            # never reaches MultiBand) and record that exceptional closure.
            # Isolated expanded guard pixels deliberately do not count here:
            # they are already safe under RGB ownership and do not establish
            # a reason to fit geometry.
            geometry_late_owner_closure_pairs += int(late_post_gain_owner_closure)
            geometry_plans[index] = replace(
                plan,
                audit={**dict(plan.audit), "final_owner": final_owner_audit},
            )
            if np.any(blend_zone & (risk.mask > 0)):
                raise RuntimeError("RGB risk pixels reached a MultiBand blend zone")
            if np.any(blend_zone & geometry_protected):
                raise RuntimeError("Geometry-protected pixels reached a MultiBand blend zone")
            if split_count or boundary_guard_count:
                raise RuntimeError("Hard-owner protection audit failed")
            if pair_level_hard_owner is None:
                (
                    l_delta,
                    delta_e00,
                    _signed_l_delta,
                    white_support,
                    l_samples,
                    delta_e00_samples,
                ) = _white_wall_pair_metrics(image0, image1, safe_wall, cuts)
                if l_delta is not None:
                    white_wall_l_samples.append(l_samples)
                if delta_e00 is not None:
                    white_wall_delta_e00_samples.append(delta_e00_samples)
            else:
                l_delta = None
                delta_e00 = None
                white_support = 0
            pair_valid = owner0 | owner1
            canvas_view = canvas[:, left:right]
            valid_view = valid_canvas[:, left:right]
            owner_view = owner_canvas[:, left:right]
            canvas_view[pair_valid] = pair_image[pair_valid]
            valid_view[pair_valid] = True
            owner_view[owner0] = index
            owner_view[owner1] = index + 1
            if not used_graphcut or hard_rows:
                hard_cut_pairs += 1
            graphcut_pairs += int(used_graphcut)
            blend_pixels += pair_blend_pixels
            protected_split_components += split_count
            owner_boundary_guard_pixels += boundary_guard_count
            geometry_protected_pixels += int(np.count_nonzero(geometry_protected))
            geometry_active_pixels += int(np.count_nonzero(geometry_active))
            geometry_out_of_scope_first_pixels += int(
                np.count_nonzero(geometry_out_of_scope_first)
            )
            geometry_blend_pixels += int(np.count_nonzero(blend_zone & geometry_protected))
            actual_search_widths.append(int(right - left))
            applied_multiband_levels.append(pair_levels)
            photometric = photometric_edges[index]
            pair_metadata.append(
                {
                    "first_frame_id": first.frame_id,
                    "second_frame_id": second.frame_id,
                    "overlap_x": [left, right],
                    "search_corridor_width_pixels": int(right - left),
                    "search_corridor_target_width_pixels": int(
                        settings.seam_search_width_pixels
                    ),
                    "blend_width_pixels": blend_width,
                    "multiband_levels": pair_levels,
                    "blend_zone_pixel_count": pair_blend_pixels,
                    "blend_zone_risk_pixel_count": 0,
                    "risk_pixel_count": int(np.count_nonzero(risk.mask)),
                    "risk_seed_pixel_count": risk.seed_pixel_count,
                    "risk_edge_offset_p95_pixels": risk.edge_offset_p95,
                    "protected_guard_pixel_count": int(np.count_nonzero(guard)),
                    "protected_guard_radius_pixels": guard_radius,
                    "protected_component_count": guard_component_count,
                    "geometry_protected_pixel_count": int(
                        np.count_nonzero(geometry_protected)
                    ),
                    "geometry_active_mesh_pixel_count": int(
                        np.count_nonzero(geometry_active)
                    ),
                    "geometry_out_of_scope_first_source_pixel_count": int(
                        np.count_nonzero(geometry_out_of_scope_first)
                    ),
                    "geometry_blend_zone_pixel_count": 0,
                    "geometry_component_hard_owner_fallback": index
                    in component_hard_owner_fallbacks,
                    "geometry_component_hard_owner_fallback_reason": (
                        component_hard_owner_fallback_reasons.get(index)
                    ),
                    "geometry_component_hard_owner_constraints": {
                        str(label): int(owner)
                        for label, owner in component_hard_owner_fallbacks.get(
                            index, {}
                        ).items()
                    },
                    "geometry_pair_level_hard_owner": pair_level_hard_owner,
                    "geometry_pair_level_hard_owner_reason": (
                        pair_level_hard_owner_reasons.get(index)
                        or runtime_pair_level_hard_owner_reasons.get(index)
                        if pair_level_hard_owner is not None
                        else None
                    ),
                    "preflight_component_constraint_count": len(
                        sequence_owner_preflight.component_owner_constraints[index]
                    ),
                    "split_protected_component_count": split_count,
                    "owner_boundary_risk_guard_pixel_count": boundary_guard_count,
                    "safe_white_wall_pixel_count": white_support,
                    "safe_white_wall_delta_l_p95": l_delta,
                    "safe_white_wall_delta_e00_p95": delta_e00,
                    "multiband_mask0_pixel_count": mask0_pixels,
                    "multiband_mask1_pixel_count": mask1_pixels,
                    "multiband_masks_complementary": bool(
                        pair_blend_pixels == 0 or masks_distinct
                    ),
                    "photometric_safe_wall_support_pixels": (
                        photometric.support_pixels if photometric is not None else 0
                    ),
                    "photometric_log_mad_bgr": (
                        photometric.mad_bgr.tolist() if photometric is not None else None
                    ),
                    "graphcut_used": used_graphcut,
                    "hard_cut_row_count": hard_rows,
                    "geometry_assistance": {
                        "triggered": geometry_plans[index].triggered,
                        "accepted": geometry_plans[index].accepted,
                        "fallback": geometry_plans[index].fallback,
                        "audit": dict(geometry_plans[index].audit),
                    },
                }
            )

        if protected_split_components or owner_boundary_guard_pixels:
            raise RuntimeError("RGB hard-owner structural protection audit failed")
        white_wall_l_p95 = (
            float(np.percentile(np.concatenate(white_wall_l_samples), 95.0))
            if white_wall_l_samples
            else None
        )
        white_wall_delta_e00_p95 = (
            float(np.percentile(np.concatenate(white_wall_delta_e00_samples), 95.0))
            if white_wall_delta_e00_samples
            else None
        )
        white_wall_measurement_count = int(
            sum(samples.size for samples in white_wall_delta_e00_samples)
        )
        stripe_energy_before = _periodic_stripe_energy(raw_stripe_samples)
        stripe_energy_after = _periodic_stripe_energy(corrected_stripe_samples)
        stripe_reduction_required = (
            stripe_energy_before is not None and stripe_energy_before >= 0.20
        )
        stripe_energy_ratio = (
            stripe_energy_after / stripe_energy_before
            if stripe_energy_before is not None
            and stripe_energy_after is not None
            and stripe_energy_before > 1e-9
            else None
        )

        if np.any(valid_canvas & (owner_canvas < 0)):
            raise RuntimeError("RGB pushbroom valid output contains an unowned pixel")
        if not np.any(valid_canvas):
            raise RuntimeError("Calibrated RGB pushbroom produced no valid RGB pixels")
        crop = largest_valid_rectangle(valid_canvas)
        panorama = canvas[crop.y : crop.y + crop.height, crop.x : crop.x + crop.width]
        cropped_owner = owner_canvas[
            crop.y : crop.y + crop.height, crop.x : crop.x + crop.width
        ]
        source_owner_pixel_counts = [
            int(np.count_nonzero(cropped_owner == source_index))
            for source_index in range(len(frames))
        ]
        effective_pair_level_hard_owners = {
            **pair_level_hard_owners,
            **runtime_pair_level_hard_owners,
        }
        effective_pair_level_hard_owner_reasons = {
            **pair_level_hard_owner_reasons,
            **runtime_pair_level_hard_owner_reasons,
        }
        owner_topology_fallback_pairs = set(effective_pair_level_hard_owners) | set(
            component_hard_owner_fallbacks
        )
        hard_owner_topology_pairs = owner_topology_fallback_pairs | {
            int(plan.pair_index) for plan in geometry_plans if plan.triggered
        }
        suppressed_source_frames = _audit_suppressed_source_frames(
            [int(frame.frame_id) for frame in frames],
            source_owner_pixel_counts,
            hard_owner_topology_pairs,
        )
        endpoint_outer_owner_pixel_counts: list[int] = []
        endpoint_outer_trimmed_invalid_pixel_counts: list[int] = []
        endpoint_outer_trimmed_column_counts: list[int] = []
        if layout.endpoint_outer_owner_intervals_x:
            for source_index, (left, right) in zip(
                (0, len(frames) - 1),
                layout.endpoint_outer_owner_intervals_x,
                strict=True,
            ):
                x0 = max(crop.x, int(math.ceil(left)))
                x1 = min(crop.x + crop.width, int(math.ceil(right)))
                count = (
                    int(
                        np.count_nonzero(
                            owner_canvas[
                                crop.y : crop.y + crop.height,
                                x0:x1,
                            ]
                            == source_index
                        )
                    )
                    if x1 > x0
                    else 0
                )
                endpoint_outer_owner_pixel_counts.append(count)
                requested_left = max(0, int(math.ceil(left)))
                requested_right = min(layout.canvas_width, int(math.ceil(right)))
                trimmed_spans = (
                    (requested_left, min(requested_right, crop.x)),
                    (max(requested_left, crop.x + crop.width), requested_right),
                )
                trimmed_invalid = 0
                fully_valid_trimmed_columns = 0
                for trim_left, trim_right in trimmed_spans:
                    if trim_right <= trim_left:
                        continue
                    trimmed = valid_canvas[
                        crop.y : crop.y + crop.height,
                        trim_left:trim_right,
                    ]
                    trimmed_invalid += int(trimmed.size - np.count_nonzero(trimmed))
                    fully_valid_trimmed_columns += int(
                        np.count_nonzero(np.all(trimmed, axis=0))
                    )
                if fully_valid_trimmed_columns:
                    raise RuntimeError(
                        "Calibrated RGB pushbroom crop discarded a fully valid "
                        "endpoint outward calibrated field-of-view column"
                    )
                endpoint_outer_trimmed_invalid_pixel_counts.append(trimmed_invalid)
                endpoint_outer_trimmed_column_counts.append(
                    sum(max(0, right - left) for left, right in trimmed_spans)
                )
        if layout.endpoint_outer_owner_intervals_x and any(
            count == 0 for count in endpoint_outer_owner_pixel_counts
        ):
            raise RuntimeError(
                "Calibrated RGB pushbroom crop removed an endpoint outward "
                "calibrated field-of-view contribution"
            )
        crop_height_ratio = crop.height / float(layout.canvas_height)
        crop_width_ratio = crop.width / float(layout.canvas_width)
        blend_fraction = blend_pixels / max(1, int(np.count_nonzero(valid_canvas)))
        geometry_flow_validation_preview_remap_count = int(
            sum(
                int(
                    dict(plan.audit.get("rgb_flow_validation", {})).get(
                        "analysis_preview_remap_count", 0
                    )
                )
                for plan in geometry_plans
            )
        )
        quality_metrics: dict[str, float | int | bool | None] = {
            "quality_pass": True,
            "crop_height_ratio": float(crop_height_ratio),
            "crop_width_ratio": float(crop_width_ratio),
            "blend_zone_fraction": float(blend_fraction),
            "blend_zone_risk_fraction": float(maximum_blend_risk),
            "blend_zone_risk_pixel_count": 0,
            "geometry_blend_zone_pixel_count": int(geometry_blend_pixels),
            "geometry_protected_pixel_count": int(geometry_protected_pixels),
            "geometry_active_mesh_pixel_count": int(geometry_active_pixels),
            "geometry_out_of_scope_first_source_pixel_count": int(
                geometry_out_of_scope_first_pixels
            ),
            "geometry_triggered_pair_count": int(
                sum(plan.triggered for plan in geometry_plans)
            ),
            "geometry_accepted_pair_count": int(
                sum(plan.accepted for plan in geometry_plans)
            ),
            "geometry_hard_owner_fallback_pair_count": int(
                sum(plan.fallback == "hard_owner" for plan in geometry_plans)
            ),
            "geometry_pair_level_hard_owner_count": int(
                sum(
                    reason.startswith("geometry_")
                    for reason in effective_pair_level_hard_owner_reasons.values()
                )
            ),
            "geometry_component_hard_owner_fallback_pair_count": int(
                sum(
                    reason.startswith("geometry_")
                    for reason in component_hard_owner_fallback_reasons.values()
                )
            ),
            "rgb_owner_topology_pair_level_hard_owner_count": int(
                sum(
                    reason.startswith("rgb_")
                    for reason in effective_pair_level_hard_owner_reasons.values()
                )
            ),
            "rgb_owner_topology_component_hard_owner_fallback_pair_count": int(
                sum(
                    reason.startswith("rgb_")
                    for reason in component_hard_owner_fallback_reasons.values()
                )
            ),
            "geometry_late_post_gain_owner_closure_pair_count": int(
                geometry_late_owner_closure_pairs
            ),
            "strict_owner_partition": True,
            "graphcut_pair_count": graphcut_pairs,
            "hard_cut_pair_count": hard_cut_pairs,
            "multiband_levels": max(applied_multiband_levels, default=0),
            "multiband_requested_max_levels": int(multiband_levels),
            "search_corridor_target_width_pixels": int(settings.seam_search_width_pixels),
            "search_corridor_width_min_pixels": min(actual_search_widths, default=0),
            "search_corridor_width_max_pixels": max(actual_search_widths, default=0),
            "search_corridor_target_pair_fraction": float(
                sum(width >= 32 for width in actual_search_widths)
                / max(1, len(actual_search_widths))
            ),
            "owner_boundary_risk_guard_pixel_count": owner_boundary_guard_pixels,
            "split_protected_component_count": protected_split_components,
            "safe_white_wall_boundary_l_delta_p95": white_wall_l_p95,
            "safe_white_wall_boundary_delta_e00_p95": white_wall_delta_e00_p95,
            "safe_white_wall_boundary_measurement_count": white_wall_measurement_count,
            "safe_white_wall_boundary_metric_tile_pixels": [64, 4],
            "periodic_stripe_energy_before": stripe_energy_before,
            "periodic_stripe_energy_after": stripe_energy_after,
            "periodic_stripe_energy_ratio": stripe_energy_ratio,
            "periodic_stripe_reduction_required": stripe_reduction_required,
            "periodic_stripe_reduction_target_ratio": 0.60,
            "source_remap_count": renderer.remap_count,
            "analysis_preview_remap_count": analysis_renderer.analysis_preview_remap_count,
            "geometry_trigger_preview_remap_count": (
                geometry_trigger_renderer.analysis_preview_remap_count
            ),
            "geometry_flow_validation_preview_remap_count": (
                geometry_flow_validation_preview_remap_count
            ),
            "full_resolution_output_remap_count": renderer.full_resolution_output_remap_count,
            "maximum_resident_strips": 2,
            "maximum_resident_geometry_depth_frames": 2,
            "all_sources_have_owned_pixels": not bool(suppressed_source_frames),
            "all_real_pose_nodes_rendered": True,
            "suppressed_source_frame_count": int(len(suppressed_source_frames)),
            "endpoint_outer_half_fov_owner_pixel_counts": endpoint_outer_owner_pixel_counts,
            "endpoint_outer_half_fov_trimmed_invalid_pixel_counts": endpoint_outer_trimmed_invalid_pixel_counts,
            "endpoint_outer_half_fov_trimmed_column_counts": endpoint_outer_trimmed_column_counts,
            "endpoint_outer_half_fov_preserved": bool(
                layout.endpoint_outer_owner_intervals_x
            ),
            **exposure_metrics,
            "exposure_gain_min": float(np.min(gains)),
            "exposure_gain_max": float(np.max(gains)),
            "exposure_gain_min_bgr": [float(value) for value in np.min(gains, axis=0)],
            "exposure_gain_max_bgr": [float(value) for value in np.max(gains, axis=0)],
        }
        failures: list[str] = []
        if crop_height_ratio < 0.85:
            failures.append("less than 85% of levelled calibrated RGB height remains")
        if crop_width_ratio < 0.95:
            failures.append("less than 95% of the calibrated RGB strip canvas remains")
        if blend_fraction > 0.20:
            failures.append("RGB MultiBand zone exceeds 20% of valid output")
        if maximum_blend_risk != 0.0:
            failures.append("RGB risk entered a MultiBand blend zone")
        if geometry_blend_pixels:
            failures.append("geometry-protected pixels entered a MultiBand blend zone")
        if white_wall_l_p95 is None or white_wall_l_p95 > 1.5:
            failures.append("safe white-wall owner-boundary P95 |delta L*| exceeds 1.5")
        if white_wall_delta_e00_p95 is None or white_wall_delta_e00_p95 > 2.0:
            failures.append("safe white-wall owner-boundary P95 delta E00 exceeds 2.0")
        if (
            stripe_reduction_required
            and (stripe_energy_ratio is None or stripe_energy_ratio > 0.60)
        ):
            failures.append("periodic vertical stripe energy was not reduced by 40%")
        quality_metrics["quality_pass"] = not failures
        if quality_gate and failures:
            raise RuntimeError("Calibrated RGB pushbroom quality gate failed: " + "; ".join(failures))
        if analysis_renderer.analysis_preview_remap_count != len(frames):
            raise RuntimeError("RGB residual preview did not remap every real source once")
        if geometry_trigger_renderer.analysis_preview_remap_count != len(frames):
            raise RuntimeError(
                "Geometry trigger preview did not remap every real source once"
            )
        if renderer.full_resolution_output_remap_count != len(frames):
            raise RuntimeError(
                "Calibrated RGB pushbroom did not perform exactly one full-resolution "
                "output remap per real source"
            )
        residual_metadata = residual_alignment.as_dict()
        residual_metadata.update(
            {
                "backend": settings.residual_alignment.backend,
                "config": settings.residual_alignment.as_dict(),
                "analysis_preview_remap_count": analysis_renderer.analysis_preview_remap_count,
                "preview_remap_count": analysis_renderer.analysis_preview_remap_count,
                "full_resolution_output_remap_count": renderer.full_resolution_output_remap_count,
                "owner_boundary_geometry": preview_boundary_geometry,
            }
        )
        working_set_audit = residual_metadata.get("working_set_audit")
        if isinstance(working_set_audit, dict):
            working_set_audit.update(
                {
                    "geometry_component_hard_owner_fallbacks": {
                        str(pair_index): {
                            str(label): int(owner)
                            for label, owner in owners.items()
                        }
                        for pair_index, owners in component_hard_owner_fallbacks.items()
                    },
                    "geometry_component_hard_owner_fallback_reasons": {
                        str(pair_index): reason
                        for pair_index, reason in component_hard_owner_fallback_reasons.items()
                    },
                    "geometry_runtime_pair_level_hard_owners": {
                        str(pair_index): int(owner)
                        for pair_index, owner in runtime_pair_level_hard_owners.items()
                    },
                    "geometry_runtime_pair_level_hard_owner_reasons": {
                        str(pair_index): reason
                        for pair_index, reason in runtime_pair_level_hard_owner_reasons.items()
                    },
                }
            )
        metadata: dict[str, object] = {
            "backend": "calibrated_rgb_pushbroom",
            "pixel_source": "calibrated_rgb_source_samples",
            "depth_used_for_output_pixels": False,
            "depth_used_for_local_geometry": bool(
                any(plan.triggered for plan in geometry_plans)
            ),
            "local_geometry_scope": "adjacent_seam_corridors_only",
            "point_cloud_constructed": False,
            "tsdf_constructed": False,
            "reference_plane_fitted": False,
            "single_inverse_remap_per_source": True,
            "interpolated_pose_count": 0,
            "layout": layout.as_dict(),
            "rgb_motion_scale": scale.as_dict(),
            "crop": crop.as_dict(),
            "source_count": len(frames),
            "frame_ids": [frame.frame_id for frame in frames],
            "source_owner_pixel_counts": source_owner_pixel_counts,
            "suppressed_source_frames": suppressed_source_frames,
            "color_gain_channel_order": "RGB",
            "color_gain_application": "single_linear_rgb_pass",
            "color_gains": [
                [float(value) for value in gain[::-1]] for gain in gains
            ],
            "residual_alignment": residual_metadata,
            "geometry_assisted_seam": {
                "enabled": bool(settings.geometry_assisted_seam.enabled),
                "config": settings.geometry_assisted_seam.as_dict(),
                "scope": "adjacent_seam_corridors_only",
                "depth_used_for_output_pixels": False,
                "depth_used_for_local_geometry": bool(
                    any(plan.triggered for plan in geometry_plans)
                ),
                "triggered_pair_count": int(sum(plan.triggered for plan in geometry_plans)),
                "accepted_pair_count": int(sum(plan.accepted for plan in geometry_plans)),
                "hard_owner_fallback_pair_count": int(
                    sum(plan.fallback == "hard_owner" for plan in geometry_plans)
                ),
                "pair_level_hard_owner_count": int(
                    len(pair_level_hard_owners) + len(runtime_pair_level_hard_owners)
                ),
                "component_hard_owner_fallback_pair_count": int(
                    len(component_hard_owner_fallbacks)
                ),
                "suppressed_source_frame_count": int(len(suppressed_source_frames)),
                "pairs": [
                    {
                        "pair_index": int(plan.pair_index),
                        "frame_ids": [int(value) for value in plan.frame_ids],
                        "triggered": bool(plan.triggered),
                        "corridor_x": (
                            list(plan.corridor_x) if plan.corridor_x is not None else None
                        ),
                        "warp_source_index": plan.warp_source_index,
                        "accepted": bool(plan.accepted),
                        "fallback": plan.fallback,
                        "audit": dict(plan.audit),
                    }
                    for plan in geometry_plans
                ],
            },
            "pairs": pair_metadata,
            "quality_metrics": quality_metrics,
        }
        return CalibratedRGBPushbroomResult(panorama=panorama, metadata=metadata)


def _compact_geometry_diagnostic_scalar(value: object, *, context: str) -> object:
    """Copy a bounded scalar-only audit value for an in-memory A/B result.

    The formal renderer intentionally retains dense masks and temporary maps
    only inside its temporary directory.  A geometry-pair diagnostic may
    expose the *decision* that was made, but it must never accidentally grow
    into a second image/depth sidecar API.  Keep this converter deliberately
    narrow so a future audit field containing an ndarray, path, or long dense
    sequence fails the diagnostic rather than being silently published.
    """

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RuntimeError(f"{context} contains a non-finite scalar")
        return float(value)
    if isinstance(value, np.ndarray):
        raise RuntimeError(f"{context} attempted to expose a dense array")
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RuntimeError(f"{context} contains a non-string audit key")
            # Paths are not scalar quality evidence and may disclose temporary
            # planner state.  They are intentionally absent rather than
            # stringified into a diagnostic payload.
            if "path" in key.lower():
                continue
            result[key] = _compact_geometry_diagnostic_scalar(
                item, context=f"{context}.{key}"
            )
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            raise RuntimeError(f"{context} attempted to expose a long audit sequence")
        return [
            _compact_geometry_diagnostic_scalar(
                item, context=f"{context}[{index}]"
            )
            for index, item in enumerate(value)
        ]
    raise RuntimeError(
        f"{context} contains a non-scalar diagnostic value "
        f"({type(value).__name__})"
    )


def _geometry_pair_diagnostic_entry(
    metadata: Mapping[str, object],
    *,
    pair_index: int,
    label: str,
) -> dict[str, object]:
    """Extract one compact adjacent-pair audit from a renderer result."""

    geometry = metadata.get("geometry_assisted_seam")
    if not isinstance(geometry, Mapping):
        raise RuntimeError(f"{label} render lacks geometry-assistance metadata")
    pairs = geometry.get("pairs")
    if not isinstance(pairs, list) or not 0 <= pair_index < len(pairs):
        raise RuntimeError(f"{label} render lacks the requested adjacent-pair audit")
    pair = pairs[pair_index]
    if not isinstance(pair, Mapping):
        raise RuntimeError(f"{label} geometry pair audit is malformed")
    reported_index = pair.get("pair_index")
    if isinstance(reported_index, bool) or not isinstance(
        reported_index, (int, np.integer)
    ) or int(reported_index) != pair_index:
        raise RuntimeError(f"{label} geometry pair order is inconsistent")
    frame_ids = pair.get("frame_ids")
    if (
        not isinstance(frame_ids, (list, tuple))
        or len(frame_ids) != 2
        or any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in frame_ids)
    ):
        raise RuntimeError(f"{label} geometry pair frame IDs are malformed")
    audit = pair.get("audit")
    if not isinstance(audit, Mapping):
        raise RuntimeError(f"{label} geometry pair lacks a scalar audit")
    return {
        "pair_index": int(pair_index),
        "frame_ids": [int(value) for value in frame_ids],
        "triggered": bool(pair.get("triggered", False)),
        "corridor_x": _compact_geometry_diagnostic_scalar(
            pair.get("corridor_x"), context=f"{label}.corridor_x"
        ),
        "warp_source_index": _compact_geometry_diagnostic_scalar(
            pair.get("warp_source_index"),
            context=f"{label}.warp_source_index",
        ),
        "accepted": bool(pair.get("accepted", False)),
        "fallback": _compact_geometry_diagnostic_scalar(
            pair.get("fallback"), context=f"{label}.fallback"
        ),
        "reason": _compact_geometry_diagnostic_scalar(
            audit.get("reason"), context=f"{label}.audit.reason"
        ),
        "audit": _compact_geometry_diagnostic_scalar(
            audit, context=f"{label}.audit"
        ),
    }


def _diagnostic_crop_bounds(
    metadata: Mapping[str, object], *, label: str
) -> tuple[int, int, int, int]:
    """Return one formal output crop in global calibrated-canvas coordinates."""

    crop = metadata.get("crop")
    if not isinstance(crop, Mapping):
        raise RuntimeError(f"{label} render lacks calibrated crop metadata")
    values: dict[str, int] = {}
    for name in ("x", "y", "width", "height"):
        value = crop.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeError(f"{label} crop {name} is malformed")
        values[name] = int(value)
    if values["width"] <= 0 or values["height"] <= 0:
        raise RuntimeError(f"{label} render has an empty calibrated crop")
    return (
        values["x"],
        values["y"],
        values["x"] + values["width"],
        values["y"] + values["height"],
    )


def _geometry_pair_diagnostic_render_summary(
    result: CalibratedRGBPushbroomResult,
    *,
    label: str,
) -> dict[str, object]:
    """Return the source/remap/gain scalars that explain an A/B panel."""

    metadata = result.metadata
    metrics = metadata.get("quality_metrics")
    if not isinstance(metrics, Mapping):
        raise RuntimeError(f"{label} render lacks quality metrics")
    source_count = metadata.get("source_count")
    gains = metadata.get("color_gains")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, (int, np.integer))
        or int(source_count) <= 0
        or not isinstance(gains, list)
        or len(gains) != int(source_count)
    ):
        raise RuntimeError(f"{label} render source/gain audit is malformed")
    scalar_metrics = {
        name: metrics.get(name)
        for name in (
            "source_remap_count",
            "full_resolution_output_remap_count",
            "analysis_preview_remap_count",
            "geometry_trigger_preview_remap_count",
            "geometry_flow_validation_preview_remap_count",
            "exposure_gain_min",
            "exposure_gain_max",
            "exposure_gain_min_bgr",
            "exposure_gain_max_bgr",
        )
    }
    return {
        "source_count": int(source_count),
        "source_remap_count": _compact_geometry_diagnostic_scalar(
            scalar_metrics["source_remap_count"],
            context=f"{label}.source_remap_count",
        ),
        "full_resolution_output_remap_count": _compact_geometry_diagnostic_scalar(
            scalar_metrics["full_resolution_output_remap_count"],
            context=f"{label}.full_resolution_output_remap_count",
        ),
        "analysis_preview_remap_count": _compact_geometry_diagnostic_scalar(
            scalar_metrics["analysis_preview_remap_count"],
            context=f"{label}.analysis_preview_remap_count",
        ),
        "geometry_trigger_preview_remap_count": _compact_geometry_diagnostic_scalar(
            scalar_metrics["geometry_trigger_preview_remap_count"],
            context=f"{label}.geometry_trigger_preview_remap_count",
        ),
        "geometry_flow_validation_preview_remap_count": _compact_geometry_diagnostic_scalar(
            scalar_metrics["geometry_flow_validation_preview_remap_count"],
            context=f"{label}.geometry_flow_validation_preview_remap_count",
        ),
        "color_gain_count": int(len(gains)),
        "color_gain_channel_order": _compact_geometry_diagnostic_scalar(
            metadata.get("color_gain_channel_order"),
            context=f"{label}.color_gain_channel_order",
        ),
        "exposure_gain_min": _compact_geometry_diagnostic_scalar(
            scalar_metrics["exposure_gain_min"],
            context=f"{label}.exposure_gain_min",
        ),
        "exposure_gain_max": _compact_geometry_diagnostic_scalar(
            scalar_metrics["exposure_gain_max"],
            context=f"{label}.exposure_gain_max",
        ),
        "exposure_gain_min_bgr": _compact_geometry_diagnostic_scalar(
            scalar_metrics["exposure_gain_min_bgr"],
            context=f"{label}.exposure_gain_min_bgr",
        ),
        "exposure_gain_max_bgr": _compact_geometry_diagnostic_scalar(
            scalar_metrics["exposure_gain_max_bgr"],
            context=f"{label}.exposure_gain_max_bgr",
        ),
    }


def render_geometry_pair_diagnostic(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    calibration: CameraIntrinsics,
    *,
    pair_index: int,
    config: Mapping[str, object] | CalibratedRGBPushbroomConfig | None = None,
    rgb_motions: Sequence[object] | None = None,
    motion_pixels_to_full_resolution: float = 1.0,
    multiband_levels: int = 3,
    quality_gate: bool = True,
) -> CalibratedRGBPushbroomResult:
    """Render an in-memory full-chain RGB A/B crop for one adjacent seam.

    This helper is deliberately a renderer-level diagnostic primitive, not a
    delivery route.  It runs the complete supplied real-pose source chain
    twice: first with geometry assistance disabled, then with the caller's
    normal configuration.  It returns only an RGB horizontal concatenation of
    their common, globally aligned crop and compact scalar evidence.  It does
    not read sidecars, write files, fit a replacement pose, expose depth/masks,
    or change the formal ``g305-panorama`` path.
    """

    source_frames = tuple(frames)
    source_poses = tuple(poses)
    if len(source_frames) != len(source_poses) or len(source_frames) < 2:
        raise ValueError(
            "Geometry pair diagnostic requires matching full frame/pose chains "
            "with at least two sources"
        )
    if isinstance(pair_index, bool) or not isinstance(pair_index, (int, np.integer)):
        raise TypeError("Geometry pair diagnostic pair_index must be an integer")
    requested_pair = int(pair_index)
    if not 0 <= requested_pair < len(source_frames) - 1:
        raise IndexError("Geometry pair diagnostic pair_index is not an adjacent pair")
    source_motions = tuple(rgb_motions) if rgb_motions is not None else None
    settings = (
        config
        if isinstance(config, CalibratedRGBPushbroomConfig)
        else CalibratedRGBPushbroomConfig.from_mapping(config)
    )
    baseline_settings = replace(
        settings,
        geometry_assisted_seam=replace(
            settings.geometry_assisted_seam,
            enabled=False,
        ),
    )

    # Both passes receive the exact same full chain and motion evidence.  The
    # only algorithmic difference is the local-geometry enabled flag; this
    # preserves the real SE(3) layout, exposure solve and source order needed
    # for a meaningful A/B crop.
    baseline = render_calibrated_rgb_pushbroom(
        source_frames,
        source_poses,
        calibration,
        config=baseline_settings,
        rgb_motions=source_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
        multiband_levels=multiband_levels,
        quality_gate=quality_gate,
    )
    candidate = render_calibrated_rgb_pushbroom(
        source_frames,
        source_poses,
        calibration,
        config=settings,
        rgb_motions=source_motions,
        motion_pixels_to_full_resolution=motion_pixels_to_full_resolution,
        multiband_levels=multiband_levels,
        quality_gate=quality_gate,
    )
    baseline_metadata = baseline.metadata
    candidate_metadata = candidate.metadata
    baseline_pair = _geometry_pair_diagnostic_entry(
        baseline_metadata, pair_index=requested_pair, label="baseline"
    )
    candidate_pair = _geometry_pair_diagnostic_entry(
        candidate_metadata, pair_index=requested_pair, label="candidate"
    )
    expected_pair_frame_ids = [
        int(source_frames[requested_pair].frame_id),
        int(source_frames[requested_pair + 1].frame_id),
    ]
    if (
        baseline_pair["frame_ids"] != expected_pair_frame_ids
        or candidate_pair["frame_ids"] != expected_pair_frame_ids
    ):
        raise RuntimeError("Geometry pair diagnostic source order disagrees with pair audit")

    baseline_frame_ids = baseline_metadata.get("frame_ids")
    candidate_frame_ids = candidate_metadata.get("frame_ids")
    expected_frame_ids = [int(frame.frame_id) for frame in source_frames]
    if (
        baseline_frame_ids != expected_frame_ids
        or candidate_frame_ids != expected_frame_ids
    ):
        raise RuntimeError("Geometry pair diagnostic did not retain the full source chain")
    baseline_layout = baseline_metadata.get("layout")
    candidate_layout = candidate_metadata.get("layout")
    if not isinstance(baseline_layout, Mapping) or not isinstance(candidate_layout, Mapping):
        raise RuntimeError("Geometry pair diagnostic lacks calibrated layout metadata")
    for name in ("width", "height", "frame_ids", "owner_boundaries_x"):
        if baseline_layout.get(name) != candidate_layout.get(name):
            raise RuntimeError("Geometry pair diagnostic baseline/candidate layouts differ")
    canvas_width = candidate_layout.get("width")
    canvas_height = candidate_layout.get("height")
    if (
        isinstance(canvas_width, bool)
        or isinstance(canvas_height, bool)
        or not isinstance(canvas_width, (int, np.integer))
        or not isinstance(canvas_height, (int, np.integer))
        or int(canvas_width) <= 0
        or int(canvas_height) <= 0
    ):
        raise RuntimeError("Geometry pair diagnostic canvas dimensions are malformed")

    candidate_corridor = candidate_pair["corridor_x"]
    if candidate_corridor is not None:
        if (
            not isinstance(candidate_corridor, list)
            or len(candidate_corridor) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in candidate_corridor
            )
        ):
            raise RuntimeError("Geometry pair diagnostic corridor audit is malformed")
        requested_x0, requested_x1 = candidate_corridor
        roi_anchor = "candidate_geometry_corridor"
    else:
        boundaries = candidate_layout.get("owner_boundaries_x")
        if (
            not isinstance(boundaries, (list, tuple))
            or len(boundaries) != len(source_frames) - 1
            or not isinstance(boundaries[requested_pair], (int, float, np.number))
        ):
            raise RuntimeError("Geometry pair diagnostic lacks nominal owner boundary")
        boundary = float(boundaries[requested_pair])
        if not math.isfinite(boundary):
            raise RuntimeError("Geometry pair diagnostic nominal owner boundary is non-finite")
        requested_width = int(settings.geometry_assisted_seam.analysis_corridor_width_pixels)
        requested_x0 = int(math.floor(boundary - requested_width / 2.0))
        requested_x1 = requested_x0 + requested_width
        roi_anchor = "nominal_owner_boundary"
    requested_x0 = max(0, int(requested_x0))
    requested_x1 = min(int(canvas_width), int(requested_x1))
    if requested_x1 <= requested_x0:
        raise RuntimeError("Geometry pair diagnostic requested an empty global seam ROI")

    baseline_crop = _diagnostic_crop_bounds(baseline_metadata, label="baseline")
    candidate_crop = _diagnostic_crop_bounds(candidate_metadata, label="candidate")
    common_x0 = max(baseline_crop[0], candidate_crop[0])
    common_y0 = max(baseline_crop[1], candidate_crop[1])
    common_x1 = min(baseline_crop[2], candidate_crop[2])
    common_y1 = min(baseline_crop[3], candidate_crop[3])
    roi_x0 = max(requested_x0, common_x0)
    roi_x1 = min(requested_x1, common_x1)
    if roi_x1 <= roi_x0 or common_y1 <= common_y0:
        raise RuntimeError(
            "Geometry pair diagnostic has no common calibrated RGB crop around "
            "the requested seam"
        )

    def panel_from(result: CalibratedRGBPushbroomResult, crop: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, _x1, _y1 = crop
        panel = result.panorama[
            common_y0 - y0 : common_y1 - y0,
            roi_x0 - x0 : roi_x1 - x0,
        ]
        expected_shape = (common_y1 - common_y0, roi_x1 - roi_x0, 3)
        if panel.shape != expected_shape or panel.dtype != np.uint8:
            raise RuntimeError("Geometry pair diagnostic RGB panel is malformed")
        return np.ascontiguousarray(panel)

    baseline_panel = panel_from(baseline, baseline_crop)
    candidate_panel = panel_from(candidate, candidate_crop)
    panorama = np.ascontiguousarray(np.hstack((baseline_panel, candidate_panel)))

    baseline_summary = _geometry_pair_diagnostic_render_summary(
        baseline, label="baseline"
    )
    candidate_summary = _geometry_pair_diagnostic_render_summary(
        candidate, label="candidate"
    )
    for summary, label in ((baseline_summary, "baseline"), (candidate_summary, "candidate")):
        if summary["source_count"] != len(source_frames):
            raise RuntimeError(f"Geometry pair diagnostic {label} source count changed")
        if summary["full_resolution_output_remap_count"] != len(source_frames):
            raise RuntimeError(
                f"Geometry pair diagnostic {label} did not render every source once"
            )

    roi = {
        "x": int(roi_x0),
        "y": int(common_y0),
        "width": int(roi_x1 - roi_x0),
        "height": int(common_y1 - common_y0),
        "requested_x": [int(requested_x0), int(requested_x1)],
        "anchor": roi_anchor,
    }
    panel_width = int(roi_x1 - roi_x0)
    metadata: dict[str, object] = {
        "backend": "calibrated_rgb_pushbroom_geometry_pair_ab_diagnostic",
        "diagnostic_only": True,
        "deliverable_published": False,
        "pixel_source": "calibrated_rgb_source_samples",
        "depth_used_for_output_pixels": False,
        "pair_index": int(requested_pair),
        "pair_frame_ids": expected_pair_frame_ids,
        "quality_gate_enabled": bool(quality_gate),
        "roi_global": roi,
        "panel_mapping": {
            "layout": "horizontal_baseline_then_candidate",
            "baseline": {
                "columns": [0, panel_width],
                "global_roi": dict(roi),
            },
            "candidate": {
                "columns": [panel_width, panel_width * 2],
                "global_roi": dict(roi),
            },
        },
        "source_chain": {
            "source_count": int(len(source_frames)),
            "pair_frame_ids": expected_pair_frame_ids,
            "baseline_candidate_frame_ids_equal": True,
            "baseline_geometry_assisted_enabled": False,
            "candidate_geometry_assisted_enabled": bool(
                settings.geometry_assisted_seam.enabled
            ),
        },
        "pair_audits": {
            "baseline": baseline_pair,
            "candidate": candidate_pair,
        },
        "source_remap_gain_info": {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
        },
    }
    return CalibratedRGBPushbroomResult(panorama=panorama, metadata=metadata)
