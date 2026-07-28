"""Depth-aware multi-view inspection renderer for a long linear side scan.

The renderer uses overlapping virtual perspective panels.  Every RGB-D sample
is reconstructed with its real ``camera_to_world`` pose and then projected to
the nearest panel in the straightened display path.  A reference-depth plane
maps to the same canvas coordinate from adjacent panels, while real near
surfaces retain perspective and are resolved by depth/owner logic.

This is deliberately independent from the metric raster and TSDF products.
It consumes the original RGB-D frames and the immutable real trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import re
from typing import Mapping, Sequence

import cv2
import numpy as np

from .cuda_backend import (
    linear_to_srgb_bgr,
    pinhole_unproject,
    remap as accelerated_remap,
    srgb_to_linear_bgr,
    transform_points,
)
from .inspection_chain_seam import (
    ChainSeamConfig,
    PairCorridorEvidence,
    PanelLocalEvidence,
    select_adaptive_nominal_boundaries,
    solve_adjacent_panel_chain,
)
from .inspection_identity_mesh import (
    InspectionIdentityMeshConfig,
    InspectionIdentityMeshSource,
    composite_inspection_identity_owners,
)
from .foreground_object_anchor import (
    ForegroundAnchorSource,
    ForegroundObjectAnchorPlan,
    overlay_foreground_object_anchors,
    plan_foreground_object_anchors,
)
from .inspection_world_coverage import (
    InspectionWorldCoverageConfig,
    audit_inspection_world_coverage,
)
from .rgbd_projection import estimate_world_axes
from .render import largest_valid_rectangle
from .session import CameraIntrinsics, RGBDFrame, read_aligned_depth_mm


@dataclass(frozen=True)
class InspectionMultiviewConfig:
    enabled: bool = True
    minimum_depth_mm: float = 200.0
    maximum_depth_mm: float = 3000.0
    reference_depth_quantile: float = 0.85
    background_panel_overlap: float = 0.95
    minimum_depth_panel_overlap: float = 0.30
    preview_stride: int = 8
    chunk_rows: int = 128
    maximum_canvas_megapixels: float = 200.0
    maximum_working_bytes: int = 4_000_000_000
    temporal_absolute_tolerance_mm: float = 20.0
    temporal_relative_tolerance: float = 0.02
    foreground_reference_depth_ratio: float = 0.70
    foreground_depth_margin_mm: float = 60.0
    foreground_depth_margin_ratio: float = 0.08
    foreground_guard_radius_pixels: int = 12
    minimum_foreground_component_pixels: int = 16
    foreground_world_anchor_enabled: bool = False
    depth_mesh_cell_size_pixels: int = 8
    depth_mesh_min_jacobian: float = 0.01
    depth_mesh_max_jacobian: float = 64.0
    depth_mesh_boundary_margin_pixels: int = 1
    identity_mesh_cell_size_pixels: int = 4
    identity_mesh_maximum_fill_distance_pixels: float = 2.5
    graphcut_preview_scale: float = 0.25
    chain_seam_corridor_width_pixels: int = 96
    chain_seam_maximum_row_step_pixels: int = 1
    chain_seam_adaptive_boundary_maximum_shift_pixels: int = 64
    chain_seam_adaptive_boundary_risk_guard_pixels: int = 12
    chain_seam_adaptive_boundary_minimum_common_coverage_ratio: float = 0.50
    chain_seam_adaptive_boundary_shift_penalty: float = 0.05
    chain_seam_hard_cut_fallback_enabled: bool = True
    dis_preview_scale: float = 0.25
    dis_maximum_motion_pixels: float = 2.0
    dis_maximum_fb_error_pixels: float = 1.0
    dis_maximum_rgb_residual: float = 12.0
    dis_maximum_gradient: float = 24.0
    maximum_background_owner_boundary_lab_p95: float = 30.0

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None = None
    ) -> "InspectionMultiviewConfig":
        payload = {} if value is None else dict(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"Unknown inspection_multiview configuration keys: {unknown}"
            )
        try:
            config = cls(**payload)
        except TypeError as exc:
            raise ValueError("Invalid inspection_multiview configuration") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if self.enabled is not True:
            raise ValueError("Formal multi-view inspection output cannot be disabled")
        if type(self.foreground_world_anchor_enabled) is not bool:
            raise ValueError(
                "inspection_multiview.foreground_world_anchor_enabled "
                "must be boolean"
            )
        finite_positive = {
            "minimum_depth_mm": self.minimum_depth_mm,
            "maximum_depth_mm": self.maximum_depth_mm,
            "maximum_canvas_megapixels": self.maximum_canvas_megapixels,
            "temporal_absolute_tolerance_mm": (
                self.temporal_absolute_tolerance_mm
            ),
            "temporal_relative_tolerance": self.temporal_relative_tolerance,
            "foreground_depth_margin_mm": self.foreground_depth_margin_mm,
            "foreground_depth_margin_ratio": (
                self.foreground_depth_margin_ratio
            ),
        }
        for name, value in finite_positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"inspection_multiview.{name} must be finite and positive"
                )
        if self.maximum_depth_mm <= self.minimum_depth_mm:
            raise ValueError("inspection_multiview depth range is empty")
        if not 0.5 <= self.reference_depth_quantile < 1.0:
            raise ValueError(
                "inspection_multiview.reference_depth_quantile must be in [0.5, 1)"
            )
        for name, value in (
            ("background_panel_overlap", self.background_panel_overlap),
            ("minimum_depth_panel_overlap", self.minimum_depth_panel_overlap),
        ):
            if not 0.0 < float(value) < 1.0:
                raise ValueError(f"inspection_multiview.{name} must be in (0, 1)")
        if self.preview_stride < 1 or self.chunk_rows < 1:
            raise ValueError(
                "inspection_multiview preview_stride/chunk_rows must be positive"
            )
        if (
            type(self.maximum_working_bytes) is not int
            or self.maximum_working_bytes <= 0
        ):
            raise ValueError(
                "inspection_multiview.maximum_working_bytes "
                "must be a positive integer"
            )
        if not 0.0 < self.foreground_reference_depth_ratio < 1.0:
            raise ValueError(
                "inspection_multiview.foreground_reference_depth_ratio "
                "must be in (0, 1)"
            )
        if not 0.0 < self.foreground_depth_margin_ratio < 1.0:
            raise ValueError(
                "inspection_multiview.foreground_depth_margin_ratio "
                "must be in (0, 1)"
            )
        if self.minimum_foreground_component_pixels < 1:
            raise ValueError(
                "inspection_multiview.minimum_foreground_component_pixels "
                "must be positive"
            )
        if not 1 <= self.foreground_guard_radius_pixels <= 32:
            raise ValueError(
                "inspection_multiview.foreground_guard_radius_pixels "
                "must be in [1, 32]"
            )
        if not 2 <= self.depth_mesh_cell_size_pixels <= 32:
            raise ValueError(
                "inspection_multiview.depth_mesh_cell_size_pixels "
                "must be in [2, 32]"
            )
        if (
            not math.isfinite(self.depth_mesh_min_jacobian)
            or not math.isfinite(self.depth_mesh_max_jacobian)
            or self.depth_mesh_min_jacobian <= 0.0
            or self.depth_mesh_max_jacobian
            <= self.depth_mesh_min_jacobian
        ):
            raise ValueError(
                "inspection_multiview depth-mesh Jacobian bounds are invalid"
            )
        if not 0 <= self.depth_mesh_boundary_margin_pixels <= 8:
            raise ValueError(
                "inspection_multiview.depth_mesh_boundary_margin_pixels "
                "must be in [0, 8]"
            )
        if not 2 <= int(self.identity_mesh_cell_size_pixels) <= 16:
            raise ValueError(
                "inspection_multiview.identity_mesh_cell_size_pixels "
                "must be in [2, 16]"
            )
        if not (
            math.isfinite(
                float(self.identity_mesh_maximum_fill_distance_pixels)
            )
            and 0.0
            < float(self.identity_mesh_maximum_fill_distance_pixels)
            <= 2.5
        ):
            raise ValueError(
                "inspection_multiview."
                "identity_mesh_maximum_fill_distance_pixels must be in "
                "(0, 2.5]"
            )
        if not 0.125 <= float(self.graphcut_preview_scale) <= 1.0:
            raise ValueError(
                "inspection_multiview.graphcut_preview_scale must be in "
                "[0.125, 1]"
            )
        if not 96 <= int(self.chain_seam_corridor_width_pixels) <= 160:
            raise ValueError(
                "inspection_multiview.chain_seam_corridor_width_pixels "
                "must be in [96, 160]"
            )
        if not 0 <= int(self.chain_seam_maximum_row_step_pixels) <= 16:
            raise ValueError(
                "inspection_multiview.chain_seam_maximum_row_step_pixels "
                "must be in [0, 16]"
            )
        if not (
            0
            <= int(
                self.chain_seam_adaptive_boundary_maximum_shift_pixels
            )
            <= 160
        ):
            raise ValueError(
                "inspection_multiview."
                "chain_seam_adaptive_boundary_maximum_shift_pixels "
                "must be in [0, 160]"
            )
        if not (
            1
            <= int(
                self.chain_seam_adaptive_boundary_risk_guard_pixels
            )
            <= 32
        ):
            raise ValueError(
                "inspection_multiview."
                "chain_seam_adaptive_boundary_risk_guard_pixels "
                "must be in [1, 32]"
            )
        if not (
            0.0
            < float(
                self.chain_seam_adaptive_boundary_minimum_common_coverage_ratio
            )
            <= 1.0
        ):
            raise ValueError(
                "inspection_multiview."
                "chain_seam_adaptive_boundary_minimum_common_coverage_ratio "
                "must be in (0, 1]"
            )
        if (
            not math.isfinite(
                float(self.chain_seam_adaptive_boundary_shift_penalty)
            )
            or float(self.chain_seam_adaptive_boundary_shift_penalty) < 0.0
        ):
            raise ValueError(
                "inspection_multiview."
                "chain_seam_adaptive_boundary_shift_penalty "
                "must be finite and non-negative"
            )
        if type(self.chain_seam_hard_cut_fallback_enabled) is not bool:
            raise ValueError(
                "inspection_multiview.chain_seam_hard_cut_fallback_enabled "
                "must be boolean"
            )
        if not 0.125 <= float(self.dis_preview_scale) <= 1.0:
            raise ValueError(
                "inspection_multiview.dis_preview_scale must be in [0.125, 1]"
            )
        for name, value in (
            ("dis_maximum_motion_pixels", self.dis_maximum_motion_pixels),
            ("dis_maximum_fb_error_pixels", self.dis_maximum_fb_error_pixels),
            ("dis_maximum_rgb_residual", self.dis_maximum_rgb_residual),
            ("dis_maximum_gradient", self.dis_maximum_gradient),
            (
                "maximum_background_owner_boundary_lab_p95",
                self.maximum_background_owner_boundary_lab_p95,
            ),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"inspection_multiview.{name} must be finite and positive"
                )


def _background_owner_boundary_audit(
    image_bgr: np.ndarray,
    owner_frame_id: np.ndarray,
    valid_mask: np.ndarray,
    foreground_mask: np.ndarray,
    config: InspectionMultiviewConfig,
    *,
    owner_only_guard_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure visible RGB discontinuity where the background owner changes."""

    image = np.asarray(image_bgr, dtype=np.uint8)
    owner = np.asarray(owner_frame_id, dtype=np.int32)
    valid = np.asarray(valid_mask, dtype=bool)
    foreground = np.asarray(foreground_mask, dtype=bool)
    owner_only_guard = (
        np.zeros(valid.shape, dtype=bool)
        if owner_only_guard_mask is None
        else np.asarray(owner_only_guard_mask, dtype=bool)
    )
    if (
        image.shape[:2] != owner.shape
        or owner.shape != valid.shape
        or valid.shape != foreground.shape
        or owner_only_guard.shape != valid.shape
    ):
        raise RuntimeError("Inspection boundary-audit rasters are misaligned")
    guard_radius = max(1, int(config.foreground_guard_radius_pixels))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * guard_radius + 1, 2 * guard_radius + 1),
    )
    excluded_guard = foreground | owner_only_guard
    guarded_foreground = cv2.dilate(
        excluded_guard.astype(np.uint8), kernel
    ).astype(bool)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    samples: list[np.ndarray] = []
    axis_audits: list[dict[str, object]] = []
    pair_samples: dict[tuple[int, int], list[np.ndarray]] = {}
    for axis, name in ((1, "horizontal_neighbor"), (0, "vertical_neighbor")):
        count = owner.shape[axis] - 1
        first_owner = np.take(owner, range(count), axis=axis)
        second_owner = np.take(owner, range(1, count + 1), axis=axis)
        first_valid = np.take(valid, range(count), axis=axis)
        second_valid = np.take(valid, range(1, count + 1), axis=axis)
        first_guard = np.take(guarded_foreground, range(count), axis=axis)
        second_guard = np.take(
            guarded_foreground, range(1, count + 1), axis=axis
        )
        boundary = (
            first_valid
            & second_valid
            & ~first_guard
            & ~second_guard
            & (first_owner != second_owner)
        )
        first_lab = np.take(lab, range(count), axis=axis)
        second_lab = np.take(lab, range(1, count + 1), axis=axis)
        delta = np.linalg.norm(first_lab - second_lab, axis=2)
        values = np.asarray(delta[boundary], dtype=np.float32)
        boundary_first_owner = first_owner[boundary]
        boundary_second_owner = second_owner[boundary]
        if values.size:
            pair_keys = np.column_stack(
                (
                    np.minimum(
                        boundary_first_owner, boundary_second_owner
                    ),
                    np.maximum(
                        boundary_first_owner, boundary_second_owner
                    ),
                )
            )
            for pair in np.unique(pair_keys, axis=0):
                pair_key = (int(pair[0]), int(pair[1]))
                selected_pair = np.all(pair_keys == pair[None, :], axis=1)
                pair_samples.setdefault(pair_key, []).append(
                    values[selected_pair]
                )
        samples.append(values)
        axis_audits.append(
            {
                "direction": name,
                "sample_count": int(values.size),
                "median_lab_delta": (
                    float(np.median(values)) if values.size else 0.0
                ),
                "p95_lab_delta": (
                    float(np.percentile(values, 95.0)) if values.size else 0.0
                ),
            }
        )
    combined = np.concatenate(samples) if any(item.size for item in samples) else (
        np.empty(0, dtype=np.float32)
    )
    p95 = float(np.percentile(combined, 95.0)) if combined.size else 0.0
    threshold = float(config.maximum_background_owner_boundary_lab_p95)
    pair_audits: list[dict[str, object]] = []
    for (left_frame_id, right_frame_id), chunks in sorted(
        pair_samples.items()
    ):
        values = np.concatenate(chunks)
        pair_p95 = float(np.percentile(values, 95.0))
        pair_audits.append(
            {
                "left_frame_id": left_frame_id,
                "right_frame_id": right_frame_id,
                "sample_count": int(values.size),
                "median_lab_delta": float(np.median(values)),
                "p95_lab_delta": pair_p95,
                "pass": pair_p95 <= threshold,
            }
        )
    return {
        "method": (
            "cie_lab_neighbor_delta_at_owner_change_excluding_dilated_"
            "foreground_and_owner_only_guard"
        ),
        "foreground_guard_radius_pixels": guard_radius,
        "owner_only_guard_pixel_count": int(
            np.count_nonzero(owner_only_guard)
        ),
        "dilated_excluded_guard_pixel_count": int(
            np.count_nonzero(guarded_foreground)
        ),
        "sample_count": int(combined.size),
        "median_lab_delta": (
            float(np.median(combined)) if combined.size else 0.0
        ),
        "p95_lab_delta": p95,
        "maximum_allowed_p95_lab_delta": threshold,
        "pass": bool(combined.size == 0 or p95 <= threshold),
        "axes": axis_audits,
        "pairs": pair_audits,
    }


@dataclass(frozen=True)
class VirtualPerspectivePanel:
    panel_index: int
    anchor_scan_mm: float
    canvas_offset_x: float
    center_world_mm: tuple[float, float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "panel_index": self.panel_index,
            "anchor_scan_mm": self.anchor_scan_mm,
            "canvas_offset_x": self.canvas_offset_x,
            "center_world_mm": list(self.center_world_mm),
        }


@dataclass(frozen=True)
class InspectionMultiviewLayout:
    width: int
    height: int
    reference_depth_mm: float
    scan_axis: tuple[float, float, float]
    down_axis: tuple[float, float, float]
    normal_axis: tuple[float, float, float]
    panels: tuple[VirtualPerspectivePanel, ...]
    panel_step_mm: float
    canvas_megapixels: float
    canvas_offset_y: float = 0.0
    panel_center_policy: str = "fixed_median_down_normal_world_side_scan"

    def as_dict(self) -> dict[str, object]:
        return {
            "model": "overlapping_straightened_virtual_perspective_panels",
            "width": self.width,
            "height": self.height,
            "reference_depth_mm": self.reference_depth_mm,
            "scan_axis_world": list(self.scan_axis),
            "down_axis_world": list(self.down_axis),
            "normal_axis_world": list(self.normal_axis),
            "panel_step_mm": self.panel_step_mm,
            "panel_count": len(self.panels),
            "panels": [panel.as_dict() for panel in self.panels],
            "canvas_megapixels": self.canvas_megapixels,
            "canvas_offset_y": self.canvas_offset_y,
            "panel_center_policy": self.panel_center_policy,
            "equations": {
                "local_q": (
                    "[dot(P-Cv,S), dot(P-Cv,D), dot(P-Cv,N)]"
                ),
                "canvas_x": (
                    "(fx/reference_depth)*(anchor-a0)+cx+fx*qS/qN"
                ),
                "canvas_y": "canvas_offset_y+cy+fy*qD/qN",
            },
        }


@dataclass(frozen=True)
class InspectionResourceEstimate:
    """Byte-bounded contract for the corridor-local inspection target."""

    model: str
    canvas_width: int
    canvas_height: int
    canvas_pixel_count: int
    panel_count: int
    corridor_width_pixels: int
    fixed_canvas_bytes: int
    adjacent_panel_bytes: int
    corridor_workspace_bytes: int
    graphcut_preview_bytes: int
    persistent_panel_summary_bytes: int
    estimated_peak_bytes: int
    maximum_working_bytes: int
    per_panel_full_canvas_array_count: int
    per_pair_full_canvas_array_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "estimate_role": "required_corridor_local_target_contract",
            "runtime_peak_measured": False,
            "canvas_width": self.canvas_width,
            "canvas_height": self.canvas_height,
            "canvas_pixel_count": self.canvas_pixel_count,
            "panel_count": self.panel_count,
            "corridor_width_pixels": self.corridor_width_pixels,
            "fixed_canvas_bytes": self.fixed_canvas_bytes,
            "adjacent_panel_bytes": self.adjacent_panel_bytes,
            "corridor_workspace_bytes": self.corridor_workspace_bytes,
            "graphcut_preview_bytes": self.graphcut_preview_bytes,
            "persistent_panel_summary_bytes": (
                self.persistent_panel_summary_bytes
            ),
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "maximum_working_bytes": self.maximum_working_bytes,
            "within_budget": (
                self.estimated_peak_bytes <= self.maximum_working_bytes
            ),
            "maximum_resident_adjacent_panels": 2,
            "maximum_resident_pair_corridors": 1,
            "per_panel_full_canvas_array_count": (
                self.per_panel_full_canvas_array_count
            ),
            "per_pair_full_canvas_array_count": (
                self.per_pair_full_canvas_array_count
            ),
            "panel_canvas_product_bytes": 0,
            "pair_canvas_product_bytes": 0,
        }


def estimate_inspection_working_set(
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    *,
    config: InspectionMultiviewConfig | Mapping[str, object] | None = None,
    per_panel_full_canvas_array_count: int = 0,
    per_pair_full_canvas_array_count: int = 0,
) -> InspectionResourceEstimate:
    """Estimate the required corridor-local streaming working set.

    This is a target allocation contract, not a measurement of the legacy
    compose implementation.  A plan that retains even one full-canvas array
    per panel or per adjacent pair is rejected independently of the byte
    budget, because its memory grows as ``P * H * W`` rather than with one
    adjacent corridor.
    """

    selected = (
        config
        if isinstance(config, InspectionMultiviewConfig)
        else InspectionMultiviewConfig.from_mapping(config)
    )
    selected.validate()
    panel_count = len(layout.panels)
    if (
        layout.width <= 0
        or layout.height <= 0
        or panel_count < 2
        or intrinsics.width <= 0
        or intrinsics.height <= 0
    ):
        raise ValueError(
            "Inspection resource estimate needs a valid layout and camera"
        )
    if (
        type(per_panel_full_canvas_array_count) is not int
        or type(per_pair_full_canvas_array_count) is not int
        or per_panel_full_canvas_array_count < 0
        or per_pair_full_canvas_array_count < 0
    ):
        raise ValueError(
            "Inspection full-canvas allocation counts must be "
            "non-negative integers"
        )
    if (
        per_panel_full_canvas_array_count
        or per_pair_full_canvas_array_count
    ):
        raise MemoryError(
            "Inspection resource plan rejects panel-scaled or pair-scaled "
            "full-canvas resident arrays; use corridor-local storage"
        )

    canvas_pixels = int(layout.width * layout.height)
    source_pixels = int(intrinsics.width * intrinsics.height)
    corridor_width = int(selected.chain_seam_corridor_width_pixels)
    preview_width = max(
        1, int(math.ceil(intrinsics.width * selected.graphcut_preview_scale))
    )
    preview_height = max(
        1, int(math.ceil(intrinsics.height * selected.graphcut_preview_scale))
    )
    preview_pixels = int(preview_width * preview_height)

    # Conservative target-model constants include output delivery copies,
    # owner/depth/confidence/valid rasters, topology scratch, two complete
    # adjacent RGB-D/remap panels, one pair corridor, and persistent preview
    # evidence.  They intentionally do not include any P x full-canvas term.
    fixed_canvas_bytes = canvas_pixels * 72
    adjacent_panel_bytes = 2 * source_pixels * 56
    corridor_workspace_bytes = (
        int(layout.height) * corridor_width * 32
    )
    graphcut_preview_bytes = panel_count * preview_pixels * 18
    persistent_panel_summary_bytes = panel_count * preview_pixels * 2
    estimated_peak_bytes = int(
        fixed_canvas_bytes
        + adjacent_panel_bytes
        + corridor_workspace_bytes
        + graphcut_preview_bytes
        + persistent_panel_summary_bytes
    )
    estimate = InspectionResourceEstimate(
        model="corridor_local_adjacent_pair_streaming/v1",
        canvas_width=int(layout.width),
        canvas_height=int(layout.height),
        canvas_pixel_count=canvas_pixels,
        panel_count=panel_count,
        corridor_width_pixels=corridor_width,
        fixed_canvas_bytes=fixed_canvas_bytes,
        adjacent_panel_bytes=adjacent_panel_bytes,
        corridor_workspace_bytes=corridor_workspace_bytes,
        graphcut_preview_bytes=graphcut_preview_bytes,
        persistent_panel_summary_bytes=persistent_panel_summary_bytes,
        estimated_peak_bytes=estimated_peak_bytes,
        maximum_working_bytes=int(selected.maximum_working_bytes),
        per_panel_full_canvas_array_count=0,
        per_pair_full_canvas_array_count=0,
    )
    if estimate.estimated_peak_bytes > estimate.maximum_working_bytes:
        raise MemoryError(
            "Inspection corridor-local estimated working set exceeds its "
            f"byte budget: {estimate.estimated_peak_bytes} > "
            f"{estimate.maximum_working_bytes}"
        )
    return estimate


@dataclass(frozen=True)
class InspectionMultiviewResult:
    image_bgr: np.ndarray
    owner_frame_id: np.ndarray
    valid_mask: np.ndarray
    relative_depth_mm: np.ndarray
    full_extent_bgra: np.ndarray
    full_extent_owner_frame_id: np.ndarray
    metadata: dict[str, object]
    source_uv: np.ndarray | None = None

    def validate(self) -> None:
        shape = self.image_bgr.shape[:2]
        if self.image_bgr.dtype != np.uint8 or self.image_bgr.shape != (*shape, 3):
            raise RuntimeError("Inspection RGB must be an HxWx3 uint8 image")
        if (
            self.owner_frame_id.shape != shape
            or self.owner_frame_id.dtype != np.int32
            or self.valid_mask.shape != shape
            or self.relative_depth_mm.shape != shape
            or self.relative_depth_mm.dtype != np.float32
        ):
            raise RuntimeError("Inspection output rasters are not pixel-aligned")
        valid = np.asarray(self.valid_mask, dtype=bool)
        if not np.any(valid):
            raise RuntimeError("Inspection output contains no valid RGB-D surface")
        if np.any(self.owner_frame_id[valid] < 0) or np.any(
            self.owner_frame_id[~valid] != -1
        ):
            raise RuntimeError("Inspection owner validity contract failed")
        if np.any(~np.isfinite(self.relative_depth_mm[valid])) or np.any(
            np.isfinite(self.relative_depth_mm[~valid])
        ):
            raise RuntimeError("Inspection depth validity contract failed")
        full_bgra = np.asarray(self.full_extent_bgra)
        full_owner = np.asarray(self.full_extent_owner_frame_id)
        if self.source_uv is not None and (
            self.source_uv.dtype != np.float32
            or self.source_uv.shape != (*shape, 2)
        ):
            raise RuntimeError("Inspection provenance UV raster is misaligned")
        if (
            full_bgra.dtype != np.uint8
            or full_bgra.ndim != 3
            or full_bgra.shape[2] != 4
            or full_owner.dtype != np.int32
            or full_owner.shape != full_bgra.shape[:2]
        ):
            raise RuntimeError(
                "Inspection full-extent RGB/owner products are not aligned"
            )
        full_valid = full_owner >= 0
        if (
            not np.any(full_valid)
            or np.any(full_bgra[..., 3][full_valid] != 255)
            or np.any(full_bgra[..., 3][~full_valid] != 0)
            or np.any(full_bgra[..., :3][~full_valid] != 0)
        ):
            raise RuntimeError(
                "Inspection full-extent alpha/owner contract failed"
            )


@dataclass(frozen=True)
class InspectionPreSeamHardOwnerInterval:
    """One externally audited, row-contiguous panel lock.

    This record contains no model or RGB payload. ``lock_mask`` constrains
    the monotone spatial chain, while the optional ``rgb_transfer_mask`` and
    ``owner_only_mask`` keep source RGB replacement and blend protection
    limited to the measured object rather than its row-contiguous seam hull.
    """

    track_id: int
    panel_index: int
    frame_id: int
    lock_mask: np.ndarray
    union_footprint: np.ndarray
    rgb_source_panel_index: int | None = None
    rgb_transfer_mask: np.ndarray | None = None
    owner_only_mask: np.ndarray | None = None
    rgb_context_member_supports: tuple[np.ndarray, ...] = ()
    protect_from_background_seam: bool = True
    deferred_true_depth_identity_overlay: bool = False
    background_panel_lock_required: bool = True


@dataclass(frozen=True)
class InspectionForegroundIdentityOwner:
    """One measured foreground structure owned by one real RGB panel.

    ``source_mask`` is an unchanged OCR/FastSAM mask in source coordinates.
    ``target_footprint`` is its aligned-depth/true-pose projection in canvas
    coordinates.  The record contains no RGB payload and cannot alter pose.
    """

    group_id: int
    structure_id: int
    structure_kind: str
    identity_track_id: int | None
    panel_index: int
    frame_id: int
    source_index: int
    source_mask: np.ndarray
    target_footprint: np.ndarray
    measured_depth_coverage_ratio: float
    projected_in_bounds_ratio: float
    target_panel_index: int | None = None
    # Exact source-coordinate observations for every selected reference panel
    # that saw this stable track.  They are used only to remove old
    # reference-plane copies before the background seam.
    reference_observation_masks: tuple[
        tuple[int, np.ndarray], ...
    ] = ()


@dataclass(frozen=True)
class _ReferencePanelRaster:
    panel_index: int
    frame_id: int
    corner_x: int
    image_bgr: np.ndarray
    valid_mask: np.ndarray
    protected_mask: np.ndarray
    confidence: np.ndarray
    reference_map_x: np.ndarray | None = None
    reference_map_y: np.ndarray | None = None


def _prepare_pre_seam_hard_owner_intervals(
    intervals: Sequence[InspectionPreSeamHardOwnerInterval],
    rasters: Sequence[_ReferencePanelRaster],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Validate and merge mandatory single-panel locks without RGB mutation."""

    locked = np.full(shape, -1, dtype=np.int16)
    owner_only_guard = np.zeros(shape, dtype=bool)
    panel_by_index: dict[int, _ReferencePanelRaster] = {}
    for position, raster in enumerate(rasters):
        panel_index = int(raster.panel_index)
        if panel_index != position or panel_index in panel_by_index:
            raise RuntimeError(
                "Inspection reference panels are not a unique ordered chain"
            )
        panel_by_index[panel_index] = raster

    seen_track_ids: set[int] = set()
    rows: list[dict[str, object]] = []
    same_panel_overlap = 0
    rgb_transfer_owner = np.full(shape, -1, dtype=np.int32)
    same_rgb_owner_transfer_overlap = 0
    for interval in intervals:
        track_id = int(interval.track_id)
        panel_index = int(interval.panel_index)
        frame_id = int(interval.frame_id)
        if track_id in seen_track_ids:
            raise ValueError("Inspection pre-seam track IDs must be unique")
        seen_track_ids.add(track_id)
        raster = panel_by_index.get(panel_index)
        if raster is None:
            raise ValueError(
                "Inspection pre-seam interval references an unknown panel"
            )
        rgb_source_panel_index = (
            panel_index
            if interval.rgb_source_panel_index is None
            else int(interval.rgb_source_panel_index)
        )
        deferred_overlay = bool(
            interval.deferred_true_depth_identity_overlay
        )
        background_panel_lock_required = bool(
            interval.background_panel_lock_required
        )
        background_owner_decoupled = bool(
            deferred_overlay or not background_panel_lock_required
        )
        rgb_source_raster = panel_by_index.get(rgb_source_panel_index)
        if rgb_source_raster is None:
            raise ValueError(
                "Inspection pre-seam interval references an unknown RGB "
                "source panel"
            )
        if (
            not deferred_overlay
            and int(rgb_source_raster.frame_id) != frame_id
        ):
            raise RuntimeError(
                "Inspection pre-seam interval RGB source panel/frame mapping "
                "is inconsistent"
            )
        lock = np.asarray(interval.lock_mask)
        footprint = np.asarray(interval.union_footprint)
        transfer = (
            lock
            if interval.rgb_transfer_mask is None
            else np.asarray(interval.rgb_transfer_mask)
        )
        owner_only = (
            lock
            if interval.owner_only_mask is None
            else np.asarray(interval.owner_only_mask)
        )
        if (
            lock.dtype != np.bool_
            or footprint.dtype != np.bool_
            or transfer.dtype != np.bool_
            or owner_only.dtype != np.bool_
            or lock.shape != shape
            or footprint.shape != shape
            or transfer.shape != shape
            or owner_only.shape != shape
        ):
            raise ValueError(
                "Inspection pre-seam interval masks must be canvas-aligned bool"
            )
        if (
            not np.any(lock)
            or not np.any(footprint)
            or not np.any(transfer)
            or not np.any(owner_only)
        ):
            raise ValueError("Inspection pre-seam interval masks cannot be empty")
        footprint_outside_lock = int(
            np.count_nonzero(footprint & ~lock)
        )
        if footprint_outside_lock:
            raise RuntimeError(
                "Inspection pre-seam footprint is not contained by its row interval"
            )
        transfer_outside_lock = int(
            np.count_nonzero(transfer & ~lock)
        )
        owner_only_outside_lock = int(
            np.count_nonzero(owner_only & ~lock)
        )
        transfer_outside_owner_only = int(
            np.count_nonzero(transfer & ~owner_only)
        )
        if (
            transfer_outside_lock
            or owner_only_outside_lock
            or transfer_outside_owner_only
        ):
            raise RuntimeError(
                "Inspection pre-seam RGB transfer/owner-only masks are "
                "not nested inside the row interval"
            )
        affected_rows = np.flatnonzero(np.any(lock, axis=1))
        noncontiguous_rows = 0
        for row in affected_rows:
            columns = np.flatnonzero(lock[row])
            noncontiguous_rows += int(
                columns.size != int(columns[-1] - columns[0] + 1)
            )
        if noncontiguous_rows:
            raise RuntimeError(
                "Inspection pre-seam owner interval is not row-contiguous"
            )

        x0 = int(raster.corner_x)
        x1 = x0 + int(raster.valid_mask.shape[1])
        selected_panel_coverage_mask = (
            transfer if background_owner_decoupled else lock
        )
        missing_coverage = int(
            np.count_nonzero(selected_panel_coverage_mask[:, :x0])
        )
        missing_coverage += int(
            np.count_nonzero(selected_panel_coverage_mask[:, x1:])
        )
        missing_coverage += int(
            np.count_nonzero(
                selected_panel_coverage_mask[:, x0:x1]
                & ~np.asarray(raster.valid_mask, dtype=bool)
            )
        )
        if missing_coverage:
            raise RuntimeError(
                "Inspection pre-seam selected panel lacks complete valid coverage"
            )
        source_missing_coverage = 0
        if not background_owner_decoupled:
            source_x0 = int(rgb_source_raster.corner_x)
            source_x1 = source_x0 + int(
                rgb_source_raster.valid_mask.shape[1]
            )
            source_missing_coverage = int(
                np.count_nonzero(lock[:, :source_x0])
                + np.count_nonzero(lock[:, source_x1:])
            )
            source_missing_coverage += int(
                np.count_nonzero(
                    lock[:, source_x0:source_x1]
                    & ~np.asarray(
                        rgb_source_raster.valid_mask, dtype=bool
                    )
                )
            )
        if source_missing_coverage:
            raise RuntimeError(
                "Inspection pre-seam RGB source panel lacks complete lock "
                "coverage"
            )
        # A decoupled object RGB owner is independent of the background panel
        # partition.  This includes both deferred true-depth overlays and
        # same-panel reference-raster transfers.  Their object/guard masks
        # still exclude GraphCut, MultiBand, and flow, but their
        # row-contiguous guards must not force the background to carry the
        # object's source panel: guards of two distinct shelf objects can
        # legitimately overlap even though their measured RGB footprints do
        # not.  Deferred owners are audited after their inverse-mesh overlay;
        # reference-raster owners are audited after their exact transfer.
        if not background_owner_decoupled:
            conflict = lock & (locked >= 0) & (locked != panel_index)
            if np.any(conflict):
                raise RuntimeError(
                    "Inspection pre-seam hard-owner intervals overlap with "
                    "different panels"
                )
            same_panel_overlap += int(
                np.count_nonzero(lock & (locked == panel_index))
            )
            locked[lock] = np.int16(panel_index)
        if not deferred_overlay:
            rgb_transfer_conflict = (
                transfer
                & (rgb_transfer_owner >= 0)
                & (rgb_transfer_owner != frame_id)
            )
            if np.any(rgb_transfer_conflict):
                raise RuntimeError(
                    "Inspection pre-seam RGB transfer footprints overlap with "
                    "different real RGB owners"
                )
            same_rgb_owner_transfer_overlap += int(
                np.count_nonzero(
                    transfer & (rgb_transfer_owner == frame_id)
                )
            )
            rgb_transfer_owner[transfer] = np.int32(frame_id)
        if interval.protect_from_background_seam:
            owner_only_guard |= owner_only
        rows.append(
            {
                "track_id": track_id,
                "panel_index": panel_index,
                "rgb_source_panel_index": rgb_source_panel_index,
                "frame_id": frame_id,
                "lock_pixel_count": int(np.count_nonzero(lock)),
                "union_footprint_pixel_count": int(
                    np.count_nonzero(footprint)
                ),
                "rgb_transfer_pixel_count": int(
                    np.count_nonzero(transfer)
                ),
                "owner_only_pixel_count": int(
                    np.count_nonzero(owner_only)
                ),
                "affected_row_count": int(affected_rows.size),
                "row_contiguous": True,
                "union_footprint_outside_lock_pixel_count": 0,
                "rgb_transfer_outside_lock_pixel_count": 0,
                "owner_only_outside_lock_pixel_count": 0,
                "rgb_transfer_outside_owner_only_pixel_count": 0,
                "selected_panel_missing_valid_pixel_count": 0,
                "selected_panel_coverage_mask": (
                    "rgb_transfer"
                    if background_owner_decoupled
                    else "row_contiguous_lock"
                ),
                "rgb_source_panel_missing_valid_pixel_count": 0,
                "deferred_true_depth_identity_overlay": deferred_overlay,
                "background_spatial_panel_lock_applied": (
                    not background_owner_decoupled
                ),
                "background_panel_owner_decoupled_from_rgb_owner": (
                    background_owner_decoupled
                ),
                "protected_from_background_seam": bool(
                    interval.protect_from_background_seam
                ),
                "post_background_hard_owner": bool(
                    not interval.protect_from_background_seam
                    and not deferred_overlay
                ),
            }
        )
    return locked, owner_only_guard, {
        "schema": "inspection-pre-seam-hard-owner-interval/v1",
        "policy": (
            "externally_audited_row_contiguous_single_reference_panel_"
            "owner_before_adjacent_chain"
        ),
        "used": bool(rows),
        "interval_count": len(rows),
        "locked_pixel_count": int(np.count_nonzero(locked >= 0)),
        "owner_only_pixel_count": int(
            np.count_nonzero(owner_only_guard)
        ),
        "same_panel_overlap_pixel_count": int(same_panel_overlap),
        "different_panel_conflict_pixel_count": 0,
        "same_rgb_owner_transfer_overlap_pixel_count": int(
            same_rgb_owner_transfer_overlap
        ),
        "different_rgb_owner_transfer_conflict_pixel_count": 0,
        "intervals": rows,
    }


@dataclass(frozen=True)
class _DepthMeshPanelRemap:
    """A target-panel to source-image map derived only from real RGB-D.

    ``map_x``/``map_y`` are sampling coordinates, not colours or poses.  They
    are finite exactly where ``valid_mask`` is true and never extrapolate
    outside a solver-valid source depth cell.
    """

    corner_x: int
    map_x: np.ndarray
    map_y: np.ndarray
    relative_depth_mm: np.ndarray
    valid_mask: np.ndarray
    audit: dict[str, object]


def _validate_inputs(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
) -> list[np.ndarray]:
    if len(frames) < 2 or len(frames) != len(poses):
        raise ValueError("Inspection needs at least two aligned RGB-D poses")
    ids = [int(frame.frame_id) for frame in frames]
    if len(set(ids)) != len(ids):
        raise ValueError("Inspection frame IDs must be unique")
    checked: list[np.ndarray] = []
    for pose in poses:
        matrix = np.asarray(pose, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Inspection poses must be finite 4x4 matrices")
        rotation = matrix[:3, :3]
        if (
            not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-4)
        ):
            raise ValueError("Inspection poses must be rigid camera_to_world SE(3)")
        checked.append(matrix)
    return checked


def _reference_depth(
    frames: Sequence[RGBDFrame],
    config: InspectionMultiviewConfig,
) -> float:
    per_frame: list[float] = []
    for frame in frames:
        depth = read_aligned_depth_mm(frame)
        sampled = depth[:: config.preview_stride, :: config.preview_stride]
        valid = sampled[
            np.isfinite(sampled)
            & (sampled >= config.minimum_depth_mm)
            & (sampled <= config.maximum_depth_mm)
        ]
        if valid.size:
            per_frame.append(
                float(np.quantile(valid, config.reference_depth_quantile))
            )
    if len(per_frame) < 2:
        raise RuntimeError(
            "Inspection cannot estimate a stable RGB-D reference depth"
        )
    reference = float(np.median(per_frame))
    if not math.isfinite(reference):
        raise RuntimeError("Inspection reference depth is non-finite")
    return float(
        np.clip(reference, config.minimum_depth_mm, config.maximum_depth_mm)
    )


def estimate_inspection_layout(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    config: InspectionMultiviewConfig | Mapping[str, object] | None = None,
) -> InspectionMultiviewLayout:
    selected = (
        config
        if isinstance(config, InspectionMultiviewConfig)
        else InspectionMultiviewConfig.from_mapping(config)
    )
    selected.validate()
    checked = _validate_inputs(frames, poses)
    scan, up, normal = estimate_world_axes(checked)
    down = -up
    centers = np.stack([pose[:3, 3] for pose in checked])
    center_scan = centers @ scan
    center_down = centers @ down
    center_normal = centers @ normal
    minimum_scan = float(np.min(center_scan))
    maximum_scan = float(np.max(center_scan))
    span = maximum_scan - minimum_scan
    if span <= 0.0:
        raise ValueError("Inspection trajectory has no positive scan span")
    reference = _reference_depth(frames, selected)
    background_fov_mm = reference * intrinsics.width / intrinsics.fx
    minimum_fov_mm = (
        selected.minimum_depth_mm * intrinsics.width / intrinsics.fx
    )
    requested_step = min(
        background_fov_mm * (1.0 - selected.background_panel_overlap),
        minimum_fov_mm * (1.0 - selected.minimum_depth_panel_overlap),
    )
    panel_count = min(
        len(frames),
        max(2, int(math.ceil(span / requested_step)) + 1),
    )
    projected_span_pixels = intrinsics.fx * span / reference
    maximum_nonoverlapping_corridor_panel_count = max(
        2,
        int(
            math.floor(
                projected_span_pixels
                / float(selected.chain_seam_corridor_width_pixels)
            )
        )
        + 1,
    )
    panel_count = min(
        panel_count, maximum_nonoverlapping_corridor_panel_count
    )
    anchors = np.linspace(minimum_scan, maximum_scan, panel_count)
    panel_step = float(anchors[1] - anchors[0])
    median_down = float(np.median(center_down))
    median_normal = float(np.median(center_normal))
    panels: list[VirtualPerspectivePanel] = []
    for index, anchor in enumerate(anchors):
        center = (
            float(anchor) * scan
            + median_down * down
            + median_normal * normal
        )
        offset = (intrinsics.fx / reference) * (
            float(anchor) - float(anchors[0])
        )
        panels.append(
            VirtualPerspectivePanel(
                panel_index=index,
                anchor_scan_mm=float(anchor),
                canvas_offset_x=float(offset),
                center_world_mm=tuple(float(value) for value in center),
            )
        )
    width = int(
        math.ceil(panels[-1].canvas_offset_x + intrinsics.width)
    )
    # Preserve the full common fixed-world vertical support.  Camera vertical
    # drift must not be hidden by per-panel re-centering (which would bend the
    # shelf); one global y offset expands the shared canvas instead.
    center_scan_values = np.asarray(center_scan, dtype=np.float64)
    selected_source_indices: list[int] = []
    used_source_indices: set[int] = set()
    for panel in panels:
        candidates = np.argsort(
            np.abs(center_scan_values - panel.anchor_scan_mm),
            kind="stable",
        )
        selected_source = next(
            int(value)
            for value in candidates
            if int(value) not in used_source_indices
        )
        used_source_indices.add(selected_source)
        selected_source_indices.append(selected_source)
    source_corners = np.asarray(
        [
            [0.0, 0.0, reference],
            [intrinsics.width - 1.0, 0.0, reference],
            [0.0, intrinsics.height - 1.0, reference],
            [
                intrinsics.width - 1.0,
                intrinsics.height - 1.0,
                reference,
            ],
        ],
        dtype=np.float64,
    )
    source_corners[:, 0] = (
        source_corners[:, 0] - intrinsics.cx
    ) * reference / intrinsics.fx
    source_corners[:, 1] = (
        source_corners[:, 1] - intrinsics.cy
    ) * reference / intrinsics.fy
    projected_y_values: list[float] = [
        0.0,
        float(intrinsics.height - 1),
    ]
    for panel, source_index in zip(
        panels, selected_source_indices, strict=True
    ):
        pose = checked[source_index]
        world_corners = (
            source_corners @ pose[:3, :3].T + pose[:3, 3]
        )
        relative = world_corners - np.asarray(
            panel.center_world_mm, dtype=np.float64
        )
        q_down = relative @ down
        q_normal = relative @ normal
        finite = np.isfinite(q_down) & np.isfinite(q_normal) & (q_normal > 0)
        projected_y_values.extend(
            (
                intrinsics.cy
                + intrinsics.fy * q_down[finite] / q_normal[finite]
            ).tolist()
        )
    minimum_canvas_y = int(math.floor(min(projected_y_values)))
    maximum_canvas_y = int(math.ceil(max(projected_y_values))) + 1
    canvas_offset_y = float(-minimum_canvas_y)
    height = int(maximum_canvas_y - minimum_canvas_y)
    megapixels = width * height / 1_000_000.0
    if megapixels > selected.maximum_canvas_megapixels:
        raise MemoryError(
            "Inspection virtual-panel canvas exceeds its resource limit: "
            f"{width}x{height} ({megapixels:.2f} MP)"
        )
    layout = InspectionMultiviewLayout(
        width=width,
        height=height,
        reference_depth_mm=reference,
        scan_axis=tuple(float(value) for value in scan),
        down_axis=tuple(float(value) for value in down),
        normal_axis=tuple(float(value) for value in normal),
        panels=tuple(panels),
        panel_step_mm=panel_step,
        canvas_megapixels=megapixels,
        canvas_offset_y=canvas_offset_y,
        panel_center_policy="fixed_median_down_normal_world_side_scan",
    )
    estimate_inspection_working_set(
        layout,
        intrinsics,
        config=selected,
    )
    return layout


def project_world_points_to_panels(
    points_world_mm: np.ndarray,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to their nearest straightened virtual panel."""

    points = np.asarray(points_world_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("World points must be a finite Nx3 array")
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    down_axis = np.asarray(layout.down_axis, dtype=np.float64)
    normal_axis = np.asarray(layout.normal_axis, dtype=np.float64)
    anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels], dtype=np.float64
    )
    point_scan = points @ scan_axis
    insertion = np.searchsorted(anchors, point_scan)
    right = np.clip(insertion, 0, anchors.size - 1)
    left = np.clip(insertion - 1, 0, anchors.size - 1)
    choose_right = np.abs(anchors[right] - point_scan) < np.abs(
        anchors[left] - point_scan
    )
    panel_index = np.where(choose_right, right, left).astype(np.int32)
    centers = np.asarray(
        [panel.center_world_mm for panel in layout.panels], dtype=np.float64
    )
    offsets = np.asarray(
        [panel.canvas_offset_x for panel in layout.panels], dtype=np.float64
    )
    relative = points - centers[panel_index]
    q_scan = relative @ scan_axis
    q_down = relative @ down_axis
    q_normal = relative @ normal_axis
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (
            offsets[panel_index]
            + intrinsics.cx
            + intrinsics.fx * q_scan / q_normal
        )
        y = (
            layout.canvas_offset_y
            + intrinsics.cy
            + intrinsics.fy * q_down / q_normal
        )
    return x, y, q_normal, panel_index


def _project_world_points_to_panel(
    points_world_mm: np.ndarray,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    panel_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world_mm, dtype=np.float64)
    panel = layout.panels[int(panel_index)]
    center = np.asarray(panel.center_world_mm, dtype=np.float64)
    relative = points - center
    q_scan = relative @ np.asarray(layout.scan_axis, dtype=np.float64)
    q_down = relative @ np.asarray(layout.down_axis, dtype=np.float64)
    q_normal = relative @ np.asarray(layout.normal_axis, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (
            panel.canvas_offset_x
            + intrinsics.cx
            + intrinsics.fx * q_scan / q_normal
        )
        y = (
            layout.canvas_offset_y
            + intrinsics.cy
            + intrinsics.fy * q_down / q_normal
        )
    return x, y, q_normal


def _select_panel_sources(
    poses: Sequence[np.ndarray],
    layout: InspectionMultiviewLayout,
) -> list[tuple[int, int]]:
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    center_scan = np.asarray(
        [np.asarray(pose, dtype=np.float64)[:3, 3] @ scan_axis for pose in poses]
    )
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for panel in layout.panels:
        candidates = np.argsort(
            np.abs(center_scan - panel.anchor_scan_mm), kind="stable"
        )
        source_index = next(
            int(value) for value in candidates if int(value) not in used
        )
        used.add(source_index)
        selected.append((panel.panel_index, source_index))
    return selected


def _nearest_panel_indices(
    points_world_mm: np.ndarray,
    layout: InspectionMultiviewLayout,
) -> np.ndarray:
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    anchors = np.asarray(
        [panel.anchor_scan_mm for panel in layout.panels], dtype=np.float64
    )
    point_scan = np.asarray(points_world_mm, dtype=np.float64) @ scan_axis
    insertion = np.searchsorted(anchors, point_scan)
    right = np.clip(insertion, 0, anchors.size - 1)
    left = np.clip(insertion - 1, 0, anchors.size - 1)
    choose_right = np.abs(anchors[right] - point_scan) < np.abs(
        anchors[left] - point_scan
    )
    return np.where(choose_right, right, left).astype(np.int32)


def _undistortion_maps(
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not intrinsics.distortion or not np.any(
        np.asarray(intrinsics.distortion, dtype=np.float64)
    ):
        return None
    return cv2.initUndistortRectifyMap(
        intrinsics.matrix,
        np.asarray(intrinsics.distortion, dtype=np.float64),
        None,
        intrinsics.matrix,
        (intrinsics.width, intrinsics.height),
        cv2.CV_32FC1,
    )


def _read_rgbd(
    frame: RGBDFrame,
    intrinsics: CameraIntrinsics,
    maps: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(frame.color_path), cv2.IMREAD_COLOR)
    if image is None or image.shape != (intrinsics.height, intrinsics.width, 3):
        raise ValueError(
            f"Inspection could not decode calibrated RGB frame {frame.frame_id}"
        )
    depth = read_aligned_depth_mm(frame)
    if depth.shape != image.shape[:2]:
        raise ValueError("Inspection aligned depth dimensions do not match RGB")
    if maps is None:
        return image, depth.astype(np.float32, copy=False), np.ones(
            depth.shape, dtype=bool
        )
    map_x, map_y = maps
    image = accelerated_remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    depth = accelerated_remap(
        depth.astype(np.float32, copy=False),
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    geometric_valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0.0)
        & (map_x <= intrinsics.width - 1)
        & (map_y >= 0.0)
        & (map_y <= intrinsics.height - 1)
    )
    return image, depth, geometric_valid


def _depth_confidence(
    depth: np.ndarray,
    valid: np.ndarray,
    config: InspectionMultiviewConfig,
) -> tuple[np.ndarray, np.ndarray]:
    neighbourhood = cv2.boxFilter(
        valid.astype(np.float32),
        -1,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    local_score = np.clip(neighbourhood / 9.0, 0.0, 1.0)
    maximum = cv2.dilate(np.where(valid, depth, 0.0), np.ones((3, 3), np.uint8))
    sentinel = np.float32(config.maximum_depth_mm * 2.0)
    minimum = cv2.erode(
        np.where(valid, depth, sentinel), np.ones((3, 3), np.uint8)
    )
    tolerance = np.maximum(
        config.temporal_absolute_tolerance_mm,
        config.temporal_relative_tolerance * np.maximum(depth, 0.0),
    )
    edge = valid & ((maximum - minimum) > tolerance)
    near_taper = max(50.0, config.minimum_depth_mm * 0.1)
    far_taper = max(200.0, config.maximum_depth_mm * 0.1)
    range_score = np.minimum(
        np.clip((depth - config.minimum_depth_mm) / near_taper, 0.0, 1.0),
        np.clip((config.maximum_depth_mm - depth) / far_taper, 0.0, 1.0),
    )
    confidence = np.zeros(depth.shape, dtype=np.float32)
    confidence[valid] = np.minimum(
        range_score[valid], 0.25 + 0.75 * local_score[valid]
    )
    confidence[edge] = np.minimum(confidence[edge], 0.25)
    confidence[valid & (confidence <= 0.0)] = 1.0 / 65535.0
    return confidence, edge


def _apply_gain(
    image: np.ndarray,
    gain_rgb: Sequence[float] | None,
) -> np.ndarray:
    if gain_rgb is None:
        return image
    gain = np.asarray(gain_rgb, dtype=np.float32)
    if gain.shape != (3,) or not np.isfinite(gain).all() or np.any(gain <= 0):
        raise ValueError("Inspection source RGB gain must be finite and positive")
    linear = srgb_to_linear_bgr(image)
    linear *= gain[::-1]
    return linear_to_srgb_bgr(linear)


def _reference_panel_inverse_maps(
    *,
    source_pose: np.ndarray,
    panel_index: int,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact virtual-D0 inverse map for one reference panel."""

    panel = layout.panels[panel_index]
    rows, columns = np.indices(
        (layout.height, intrinsics.width), dtype=np.float64
    )
    q_scan = (
        (columns - intrinsics.cx)
        * layout.reference_depth_mm
        / intrinsics.fx
    )
    q_down = (
        (rows - layout.canvas_offset_y - intrinsics.cy)
        * layout.reference_depth_mm
        / intrinsics.fy
    )
    center = np.asarray(panel.center_world_mm, dtype=np.float64)
    world = (
        center[None, None, :]
        + q_scan[..., None]
        * np.asarray(layout.scan_axis, dtype=np.float64)[None, None, :]
        + q_down[..., None]
        * np.asarray(layout.down_axis, dtype=np.float64)[None, None, :]
        + layout.reference_depth_mm
        * np.asarray(layout.normal_axis, dtype=np.float64)[None, None, :]
    )
    camera = (world - source_pose[:3, 3]) @ source_pose[:3, :3]
    camera_z = camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        map_x = (
            intrinsics.fx * camera[..., 0] / camera_z + intrinsics.cx
        ).astype(np.float32)
        map_y = (
            intrinsics.fy * camera[..., 1] / camera_z + intrinsics.cy
        ).astype(np.float32)
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (camera_z > 0.0)
        & (map_x >= 0.0)
        & (map_x <= intrinsics.width - 1)
        & (map_y >= 0.0)
        & (map_y <= intrinsics.height - 1)
    )
    x0 = int(round(panel.canvas_offset_x))
    x1 = min(layout.width, x0 + intrinsics.width)
    local_width = x1 - x0
    if local_width <= 0:
        raise RuntimeError(
            "Inspection reference panel has no canvas intersection"
        )
    return (
        x0,
        np.ascontiguousarray(map_x[:, :local_width]),
        np.ascontiguousarray(map_y[:, :local_width]),
        np.ascontiguousarray(valid[:, :local_width]),
        np.ascontiguousarray(columns[:, :local_width]),
    )


def _composite_reference_panel(
    *,
    output_image: np.ndarray,
    output_depth: np.ndarray,
    output_confidence: np.ndarray,
    output_owner: np.ndarray,
    output_reliable_depth: np.ndarray,
    source_image: np.ndarray,
    source_protected_mask: np.ndarray,
    source_pose: np.ndarray,
    frame_id: int,
    panel_index: int,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    retain_reference_maps: bool = True,
) -> tuple[int, _ReferencePanelRaster]:
    """Inverse-sample one complete RGB view on the virtual D0 plane."""

    (
        x0,
        map_x,
        map_y,
        valid,
        local_columns,
    ) = _reference_panel_inverse_maps(
        source_pose=source_pose,
        panel_index=panel_index,
        layout=layout,
        intrinsics=intrinsics,
    )
    sampled = accelerated_remap(
        source_image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_protected = accelerated_remap(
        source_protected_mask.astype(np.uint8) * 255,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    x1 = x0 + map_x.shape[1]
    sampled_protected = sampled_protected > 0
    centrality = np.clip(
        1.0
        - np.abs(local_columns - intrinsics.cx)
        / max(1.0, intrinsics.width * 0.5),
        0.0,
        1.0,
    ).astype(np.float32)
    candidate_confidence = 0.05 + 0.20 * centrality
    region_confidence = output_confidence[:, x0:x1]
    take = valid & (
        (output_owner[:, x0:x1] < 0)
        | (
            (~output_reliable_depth[:, x0:x1])
            & (candidate_confidence > region_confidence + 1e-6)
        )
    )
    output_image[:, x0:x1][take] = sampled[take]
    output_depth[:, x0:x1][take] = np.float32(layout.reference_depth_mm)
    output_confidence[:, x0:x1][take] = candidate_confidence[take]
    output_owner[:, x0:x1][take] = int(frame_id)
    output_reliable_depth[:, x0:x1][take] = False
    return int(np.count_nonzero(take)), _ReferencePanelRaster(
        panel_index=int(panel_index),
        frame_id=int(frame_id),
        corner_x=x0,
        image_bgr=np.ascontiguousarray(sampled),
        valid_mask=np.ascontiguousarray(valid),
        protected_mask=np.ascontiguousarray(valid & sampled_protected),
        confidence=np.ascontiguousarray(candidate_confidence),
        reference_map_x=(
            np.ascontiguousarray(map_x)
            if retain_reference_maps
            else None
        ),
        reference_map_y=(
            np.ascontiguousarray(map_y)
            if retain_reference_maps
            else None
        ),
    )


def _rasterize_inverse_triangle(
    *,
    map_x: np.ndarray,
    map_y: np.ndarray,
    relative_depth: np.ndarray,
    target_xy: np.ndarray,
    source_xy: np.ndarray,
    target_depth: np.ndarray,
) -> int:
    """Rasterize one positive-orientation triangle as target->source maps."""

    x0 = max(0, int(math.floor(float(np.min(target_xy[:, 0])))))
    x1 = min(
        map_x.shape[1] - 1,
        int(math.ceil(float(np.max(target_xy[:, 0])))),
    )
    y0 = max(0, int(math.floor(float(np.min(target_xy[:, 1])))))
    y1 = min(
        map_x.shape[0] - 1,
        int(math.ceil(float(np.max(target_xy[:, 1])))),
    )
    if x1 < x0 or y1 < y0:
        return 0
    a, b, c = target_xy.astype(np.float64, copy=False)
    determinant = (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )
    if not math.isfinite(float(determinant)) or determinant <= 0.0:
        return 0
    yy, xx = np.indices((y1 - y0 + 1, x1 - x0 + 1), dtype=np.float64)
    xx += x0
    yy += y0
    # Barycentric weights at integer target pixel centres.  The small
    # tolerance makes the two triangles of a cell share their diagonal
    # without opening a one-pixel crack.
    weight_b = (
        (xx - a[0]) * (c[1] - a[1])
        - (yy - a[1]) * (c[0] - a[0])
    ) / determinant
    weight_c = (
        (b[0] - a[0]) * (yy - a[1])
        - (b[1] - a[1]) * (xx - a[0])
    ) / determinant
    weight_a = 1.0 - weight_b - weight_c
    inside = (
        (weight_a >= -1e-6)
        & (weight_b >= -1e-6)
        & (weight_c >= -1e-6)
    )
    if not np.any(inside):
        return 0
    candidate_depth = (
        weight_a * target_depth[0]
        + weight_b * target_depth[1]
        + weight_c * target_depth[2]
    )
    destination_depth = relative_depth[y0 : y1 + 1, x0 : x1 + 1]
    take = (
        inside
        & np.isfinite(candidate_depth)
        & (candidate_depth > 0.0)
        & (
            ~np.isfinite(destination_depth)
            | (candidate_depth < destination_depth)
        )
    )
    if not np.any(take):
        return 0
    candidate_x = (
        weight_a * source_xy[0, 0]
        + weight_b * source_xy[1, 0]
        + weight_c * source_xy[2, 0]
    )
    candidate_y = (
        weight_a * source_xy[0, 1]
        + weight_b * source_xy[1, 1]
        + weight_c * source_xy[2, 1]
    )
    map_x[y0 : y1 + 1, x0 : x1 + 1][take] = candidate_x[take]
    map_y[y0 : y1 + 1, x0 : x1 + 1][take] = candidate_y[take]
    destination_depth[take] = candidate_depth[take].astype(np.float32)
    return int(np.count_nonzero(take))


def _build_depth_mesh_panel_remap(
    *,
    source_depth_mm: np.ndarray,
    source_solver_valid: np.ndarray,
    source_pose: np.ndarray,
    panel_index: int,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    config: InspectionMultiviewConfig,
) -> _DepthMeshPanelRemap:
    """Build a continuous local inverse mesh from one real RGB-D source.

    Only complete, non-boundary depth cells are accepted.  Each accepted
    source cell is projected with the immutable ``camera_to_world`` pose and
    split into two positive-Jacobian triangles.  Rasterization stores only a
    target-to-source sampling map; RGB is sampled once by the caller.
    """

    depth = np.asarray(source_depth_mm, dtype=np.float32)
    solver_valid = np.asarray(source_solver_valid, dtype=bool)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth.shape != expected_shape or solver_valid.shape != expected_shape:
        raise ValueError("Inspection depth mesh inputs do not match intrinsics")
    pose = np.asarray(source_pose, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Inspection depth mesh pose must be finite 4x4")
    panel = layout.panels[int(panel_index)]
    corner_x = int(round(panel.canvas_offset_x))
    local_width = min(intrinsics.width, layout.width - corner_x)
    if local_width <= 0:
        raise RuntimeError("Inspection depth mesh panel is outside the canvas")
    map_x = np.full(
        (layout.height, local_width), np.nan, dtype=np.float32
    )
    map_y = np.full_like(map_x, np.nan)
    relative_depth = np.full_like(map_x, np.inf)

    step = int(config.depth_mesh_cell_size_pixels)
    margin = int(config.depth_mesh_boundary_margin_pixels)
    xs = list(range(margin, intrinsics.width - margin, step))
    ys = list(range(margin, intrinsics.height - margin, step))
    last_x = intrinsics.width - 1 - margin
    last_y = intrinsics.height - 1 - margin
    if not xs or xs[-1] != last_x:
        xs.append(last_x)
    if not ys or ys[-1] != last_y:
        ys.append(last_y)
    xs_array = np.asarray(xs, dtype=np.int32)
    ys_array = np.asarray(ys, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs_array, ys_array)
    grid_depth = depth[grid_y, grid_x].astype(np.float64)
    camera_points = pinhole_unproject(
        grid_x.reshape(-1),
        grid_y.reshape(-1),
        grid_depth.reshape(-1),
        fx=intrinsics.fx,
        fy=intrinsics.fy,
        cx=intrinsics.cx,
        cy=intrinsics.cy,
    )
    world = transform_points(
        camera_points, pose[:3, :3], pose[:3, 3]
    )
    target_x, target_y, target_z = _project_world_points_to_panel(
        world, layout, intrinsics, panel_index
    )
    target_x = target_x.reshape(grid_x.shape) - float(corner_x)
    target_y = target_y.reshape(grid_y.shape)
    target_z = target_z.reshape(grid_y.shape)
    world = world.reshape((*grid_y.shape, 3))

    cell_count = 0
    accepted_cell_count = 0
    rejected_invalid = 0
    rejected_discontinuous = 0
    rejected_panel_owner = 0
    rejected_jacobian = 0
    triangle_count = 0
    rasterized_pixel_updates = 0
    minimum_jacobian = math.inf
    maximum_jacobian = 0.0
    for row in range(len(ys) - 1):
        source_y0 = ys[row]
        source_y1 = ys[row + 1]
        for column in range(len(xs) - 1):
            cell_count += 1
            source_x0 = xs[column]
            source_x1 = xs[column + 1]
            cell_valid = solver_valid[
                source_y0 : source_y1 + 1,
                source_x0 : source_x1 + 1,
            ]
            if not np.all(cell_valid):
                rejected_invalid += 1
                continue
            cell_depth = depth[
                source_y0 : source_y1 + 1,
                source_x0 : source_x1 + 1,
            ]
            depth_median = float(np.median(cell_depth))
            depth_tolerance = max(
                config.temporal_absolute_tolerance_mm,
                config.temporal_relative_tolerance * depth_median,
            )
            if (
                not np.isfinite(cell_depth).all()
                or float(np.max(cell_depth) - np.min(cell_depth))
                > depth_tolerance
            ):
                rejected_discontinuous += 1
                continue
            node_indices = (
                (row, column),
                (row, column + 1),
                (row + 1, column + 1),
                (row + 1, column),
            )
            target = np.asarray(
                [
                    (target_x[node], target_y[node])
                    for node in node_indices
                ],
                dtype=np.float64,
            )
            z = np.asarray(
                [target_z[node] for node in node_indices], dtype=np.float64
            )
            if (
                not np.isfinite(target).all()
                or not np.isfinite(z).all()
                or np.any(z < config.minimum_depth_mm)
                or np.any(z > config.maximum_depth_mm)
            ):
                rejected_invalid += 1
                continue
            cell_world_center = np.mean(
                np.asarray([world[node] for node in node_indices]), axis=0
            )
            if int(
                _nearest_panel_indices(
                    cell_world_center[None, :], layout
                )[0]
            ) != int(panel_index):
                rejected_panel_owner += 1
                continue
            source = np.asarray(
                [
                    (source_x0, source_y0),
                    (source_x1, source_y0),
                    (source_x1, source_y1),
                    (source_x0, source_y1),
                ],
                dtype=np.float64,
            )
            source_area = float(
                (source_x1 - source_x0) * (source_y1 - source_y0)
            )
            triangle_indices = ((0, 1, 2), (0, 2, 3))
            triangle_payload: list[
                tuple[np.ndarray, np.ndarray, np.ndarray]
            ] = []
            cell_jacobians: list[float] = []
            for indices in triangle_indices:
                selected_target = target[list(indices)]
                vector_ab = selected_target[1] - selected_target[0]
                vector_ac = selected_target[2] - selected_target[0]
                doubled_area = float(
                    vector_ab[0] * vector_ac[1]
                    - vector_ab[1] * vector_ac[0]
                )
                jacobian = doubled_area / source_area
                cell_jacobians.append(jacobian)
                triangle_payload.append(
                    (
                        selected_target,
                        source[list(indices)],
                        z[list(indices)],
                    )
                )
            if any(
                not math.isfinite(value)
                or value < config.depth_mesh_min_jacobian
                or value > config.depth_mesh_max_jacobian
                for value in cell_jacobians
            ):
                rejected_jacobian += 1
                continue
            accepted_cell_count += 1
            minimum_jacobian = min(minimum_jacobian, *cell_jacobians)
            maximum_jacobian = max(maximum_jacobian, *cell_jacobians)
            for selected_target, selected_source, selected_z in triangle_payload:
                triangle_count += 1
                rasterized_pixel_updates += _rasterize_inverse_triangle(
                    map_x=map_x,
                    map_y=map_y,
                    relative_depth=relative_depth,
                    target_xy=selected_target,
                    source_xy=selected_source,
                    target_depth=selected_z,
                )
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & np.isfinite(relative_depth)
        & (map_x >= margin)
        & (map_x <= intrinsics.width - 1 - margin)
        & (map_y >= margin)
        & (map_y <= intrinsics.height - 1 - margin)
    )
    map_x[~valid] = np.nan
    map_y[~valid] = np.nan
    relative_depth[~valid] = np.inf
    return _DepthMeshPanelRemap(
        corner_x=corner_x,
        map_x=map_x,
        map_y=map_y,
        relative_depth_mm=relative_depth,
        valid_mask=valid,
        audit={
            "model": (
                "reliable_depth_cell_piecewise_affine_target_to_source"
            ),
            "cell_size_pixels": step,
            "candidate_cell_count": cell_count,
            "accepted_cell_count": accepted_cell_count,
            "rejected_invalid_or_boundary_cell_count": rejected_invalid,
            "rejected_discontinuous_cell_count": rejected_discontinuous,
            "rejected_nonlocal_panel_cell_count": rejected_panel_owner,
            "rejected_jacobian_cell_count": rejected_jacobian,
            "accepted_triangle_count": triangle_count,
            "minimum_accepted_jacobian": (
                None if not accepted_cell_count else minimum_jacobian
            ),
            "maximum_accepted_jacobian": (
                None if not accepted_cell_count else maximum_jacobian
            ),
            "rasterized_pixel_update_count": rasterized_pixel_updates,
            "valid_target_pixel_count": int(np.count_nonzero(valid)),
            "rgb_generated": False,
            "pose_modified": False,
            "source_boundary_margin_pixels": margin,
        },
    )


def _composite_depth_mesh_panel(
    *,
    mesh: _DepthMeshPanelRemap,
    source_image: np.ndarray,
    source_confidence: np.ndarray,
    output_image: np.ndarray,
    output_depth: np.ndarray,
    output_confidence: np.ndarray,
    output_owner: np.ndarray,
    output_reliable_depth: np.ndarray,
    frame_id: int,
    config: InspectionMultiviewConfig,
    require_existing_owner: bool = False,
    write_rgb: bool = True,
) -> tuple[int, int]:
    """Sample and z-compose one accepted inverse mesh without RGB synthesis."""

    safe_map_x = np.where(mesh.valid_mask, mesh.map_x, -1.0).astype(
        np.float32, copy=False
    )
    safe_map_y = np.where(mesh.valid_mask, mesh.map_y, -1.0).astype(
        np.float32, copy=False
    )
    sampled_image = (
        accelerated_remap(
            source_image,
            safe_map_x,
            safe_map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        if write_rgb
        else None
    )
    sampled_confidence = accelerated_remap(
        source_confidence.astype(np.float32, copy=False),
        safe_map_x,
        safe_map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    x0 = mesh.corner_x
    x1 = x0 + mesh.valid_mask.shape[1]
    current_depth = output_depth[:, x0:x1]
    current_confidence = output_confidence[:, x0:x1]
    current_owner = output_owner[:, x0:x1]
    current_reliable = output_reliable_depth[:, x0:x1]
    candidate_depth = mesh.relative_depth_mm
    existing = current_owner >= 0
    comparison_depth = np.where(
        existing, current_depth, candidate_depth
    )
    tolerance = np.maximum(
        config.temporal_absolute_tolerance_mm,
        config.temporal_relative_tolerance
        * np.maximum(candidate_depth, comparison_depth),
    )
    depth_delta = np.full(candidate_depth.shape, np.inf, dtype=np.float32)
    depth_delta[mesh.valid_mask] = np.abs(
        candidate_depth[mesh.valid_mask]
        - comparison_depth[mesh.valid_mask]
    )
    nearer = np.zeros(mesh.valid_mask.shape, dtype=bool)
    nearer[mesh.valid_mask] = (
        candidate_depth[mesh.valid_mask]
        < comparison_depth[mesh.valid_mask] - tolerance[mesh.valid_mask]
    )
    same_layer = (
        mesh.valid_mask
        & existing
        & current_reliable
        & (depth_delta <= tolerance)
    )
    take = mesh.valid_mask & (
        ~existing
        | ~current_reliable
        | nearer
        | (
            (depth_delta <= tolerance)
            & (sampled_confidence > current_confidence + 1e-6)
        )
    )
    if require_existing_owner:
        take &= current_owner == int(frame_id)
    if write_rgb:
        assert sampled_image is not None
        output_image[:, x0:x1][take] = sampled_image[take]
    current_depth[take] = candidate_depth[take]
    current_confidence[take] = sampled_confidence[take]
    current_owner[take] = int(frame_id)
    current_reliable[take] = True
    return int(np.count_nonzero(take)), int(np.count_nonzero(same_layer))


def _dis_safe_background_recovery(
    rasters: Sequence[_ReferencePanelRaster],
    config: InspectionMultiviewConfig,
) -> tuple[list[np.ndarray], list[dict[str, object]]]:
    """Recover only flat, RGB-consistent invalid-depth background with DIS."""

    recovered = [
        np.zeros(item.valid_mask.shape, dtype=bool) for item in rasters
    ]
    audits: list[dict[str, object]] = []
    scale = float(config.dis_preview_scale)
    for index in range(len(rasters) - 1):
        left = rasters[index]
        right = rasters[index + 1]
        overlap_x0 = max(left.corner_x, right.corner_x)
        overlap_x1 = min(
            left.corner_x + left.image_bgr.shape[1],
            right.corner_x + right.image_bgr.shape[1],
        )
        if overlap_x1 <= overlap_x0:
            audits.append(
                {
                    "left_frame_id": int(left.frame_id),
                    "right_frame_id": int(right.frame_id),
                    "overlap_pixel_count": 0,
                    "recovered_safe_background_pixel_count": 0,
                    "status": "no_overlap",
                }
            )
            continue
        left_slice = slice(
            overlap_x0 - left.corner_x,
            overlap_x1 - left.corner_x,
        )
        right_slice = slice(
            overlap_x0 - right.corner_x,
            overlap_x1 - right.corner_x,
        )
        left_image = left.image_bgr[:, left_slice]
        right_image = right.image_bgr[:, right_slice]
        common = (
            left.valid_mask[:, left_slice]
            & right.valid_mask[:, right_slice]
        )
        target_width = max(8, int(round(left_image.shape[1] * scale)))
        target_height = max(8, int(round(left_image.shape[0] * scale)))
        size = (target_width, target_height)
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        left_small = cv2.resize(
            left_gray, size, interpolation=cv2.INTER_AREA
        )
        right_small = cv2.resize(
            right_gray, size, interpolation=cv2.INTER_AREA
        )
        common_small = (
            cv2.resize(
                common.astype(np.uint8),
                size,
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        forward_solver = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
        )
        backward_solver = cv2.DISOpticalFlow_create(
            cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
        )
        forward = forward_solver.calc(left_small, right_small, None)
        backward = backward_solver.calc(right_small, left_small, None)
        rows, columns = np.indices(left_small.shape, dtype=np.float32)
        backward_at_forward = cv2.remap(
            backward,
            columns + forward[..., 0],
            rows + forward[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        motion = np.linalg.norm(forward, axis=2) / scale
        fb_error = np.linalg.norm(
            forward + backward_at_forward, axis=2
        ) / scale
        dis_safe_small = (
            common_small
            & np.isfinite(motion)
            & np.isfinite(fb_error)
            & (motion <= config.dis_maximum_motion_pixels)
            & (fb_error <= config.dis_maximum_fb_error_pixels)
        )
        dis_safe = (
            cv2.resize(
                dis_safe_small.astype(np.uint8),
                (left_image.shape[1], left_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
        residual = np.mean(
            cv2.absdiff(left_image, right_image), axis=2
        )
        left_gradient = np.maximum(
            np.abs(
                cv2.Sobel(left_gray, cv2.CV_32F, 1, 0, ksize=3)
            ),
            np.abs(
                cv2.Sobel(left_gray, cv2.CV_32F, 0, 1, ksize=3)
            ),
        )
        right_gradient = np.maximum(
            np.abs(
                cv2.Sobel(right_gray, cv2.CV_32F, 1, 0, ksize=3)
            ),
            np.abs(
                cv2.Sobel(right_gray, cv2.CV_32F, 0, 1, ksize=3)
            ),
        )
        originally_protected = (
            left.protected_mask[:, left_slice]
            | right.protected_mask[:, right_slice]
        )
        safe = (
            common
            & originally_protected
            & dis_safe
            & (residual <= config.dis_maximum_rgb_residual)
            & (left_gradient <= config.dis_maximum_gradient)
            & (right_gradient <= config.dis_maximum_gradient)
        )
        recovered[index][:, left_slice] |= safe
        recovered[index + 1][:, right_slice] |= safe
        audits.append(
            {
                "left_frame_id": int(left.frame_id),
                "right_frame_id": int(right.frame_id),
                "overlap_pixel_count": int(np.count_nonzero(common)),
                "dis_safe_pixel_count": int(np.count_nonzero(dis_safe & common)),
                "recovered_safe_background_pixel_count": int(
                    np.count_nonzero(safe)
                ),
                "maximum_motion_pixels": float(
                    np.max(motion[common_small], initial=0.0)
                ),
                "maximum_fb_error_pixels": float(
                    np.max(fb_error[common_small], initial=0.0)
                ),
                "status": "audited",
            }
        )
    return recovered, audits


def _apply_continuous_canvas_exposure_curve(
    image_bgr: np.ndarray,
    safe_background: np.ndarray,
    application_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Estimate on safe background, then apply one curve to every RGB owner."""

    estimate_mask = np.asarray(safe_background, dtype=bool)
    apply_mask = (
        estimate_mask
        if application_mask is None
        else np.asarray(application_mask, dtype=bool)
    )
    if (
        estimate_mask.shape != image_bgr.shape[:2]
        or apply_mask.shape != image_bgr.shape[:2]
        or np.any(estimate_mask & ~apply_mask)
    ):
        raise RuntimeError(
            "Inspection continuous exposure estimate/application masks are "
            "not canvas-aligned"
        )
    linear = srgb_to_linear_bgr(image_bgr)
    luma = (
        0.0722 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.2126 * linear[..., 2]
    )
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gradient = np.maximum(
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)),
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)),
    )
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    neutral_chroma = np.sqrt(
        (lab[..., 1] - 128.0) ** 2 + (lab[..., 2] - 128.0) ** 2
    )
    support_mask = (
        estimate_mask
        & (gradient <= 24.0)
        & (luma > 0.03)
        & (luma < 0.95)
        & (neutral_chroma <= 24.0)
    )
    width = image_bgr.shape[1]
    observed = np.full((width, 3), np.nan, dtype=np.float32)
    support_counts = np.count_nonzero(support_mask, axis=0)
    for column in np.flatnonzero(support_counts >= 24):
        observed[column] = np.median(
            linear[support_mask[:, column], column],
            axis=0,
        )
    known = np.flatnonzero(np.all(np.isfinite(observed), axis=1))
    if known.size < max(8, width // 20):
        return image_bgr, {
            "applied": False,
            "reason": "insufficient_neutral_safe_background_columns",
            "supported_column_count": int(known.size),
            "corrected_pixel_count": 0,
            "preserved_pixel_count": int(image_bgr.shape[0] * width),
            "reference_rgb_used": False,
        }
    supported_luma = luma[support_mask]
    measured_safe_luma = float(np.median(supported_luma))
    # Source-domain compensation above is the only exposure operation.  Do
    # not apply a post-composition global gain: it would recolour already
    # locked object owners and contradict the v4 photometric contract.
    global_gain = 1.0
    gain = np.full((width, 3), global_gain, dtype=np.float32)
    corrected_all = linear_to_srgb_bgr(linear * gain[None, :, :])
    corrected = np.ascontiguousarray(image_bgr.copy())
    corrected[apply_mask] = corrected_all[apply_mask]
    channel_names = ("B", "G", "R")
    return corrected, {
        "applied": False,
        "method": (
            "disabled_post_composition_gain_v4_source_domain_only"
        ),
        "supported_column_count": int(known.size),
        "estimate_pixel_count": int(np.count_nonzero(estimate_mask)),
        "corrected_pixel_count": int(np.count_nonzero(apply_mask)),
        "preserved_pixel_count": int(
            apply_mask.size - np.count_nonzero(apply_mask)
        ),
        "safe_background_median_linear_luma_before": measured_safe_luma,
        "global_normalization_gain": global_gain,
        "target_safe_background_linear_luma": None,
        "maximum_adjacent_column_gain_delta": 0.0,
        "column_varying_gain_used": False,
        "minimum_gain": float(np.min(gain)),
        "maximum_gain": float(np.max(gain)),
        "median_gain": float(np.median(gain)),
        "channel_gain_statistics_bgr": [
            {
                "channel": channel_names[channel],
                "minimum": float(np.min(gain[:, channel])),
                "median": float(np.median(gain[:, channel])),
                "maximum": float(np.max(gain[:, channel])),
            }
            for channel in range(3)
        ],
        "owner_pixels_mixed": 0,
        "reference_rgb_used": False,
    }


def _plan_chain_foreground_owner_locks(
    foreground_mask: np.ndarray,
    panel_valid_evidence: Sequence[PanelLocalEvidence],
    panel_confidence_evidence: Sequence[PanelLocalEvidence],
    nominal_boundaries_x: Sequence[float],
    *,
    minimum_component_pixels: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Lock each seam-risk component to one nearby fully covering panel."""

    foreground = np.asarray(foreground_mask, dtype=bool)
    if (
        len(panel_valid_evidence) < 2
        or len(panel_valid_evidence) != len(panel_confidence_evidence)
        or any(
            value.height != foreground.shape[0]
            or value.canvas_width != foreground.shape[1]
            for value in (
                *panel_valid_evidence,
                *panel_confidence_evidence,
            )
        )
    ):
        raise RuntimeError(
            "Inspection foreground chain-lock inputs are misaligned"
        )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    locked = np.full(foreground.shape, -1, dtype=np.int16)
    component_audits: list[dict[str, object]] = []
    locked_pixels = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(minimum_component_pixels):
            continue
        component = labels == label
        centroid_x = float(centroids[label, 0])
        preferred = int(
            np.searchsorted(
                np.asarray(nominal_boundaries_x, dtype=np.float64),
                centroid_x,
                side="right",
            )
        )
        candidates: list[tuple[float, int, int]] = []
        candidate_rows: list[dict[str, object]] = []
        for panel_index, (valid, confidence) in enumerate(
            zip(
                panel_valid_evidence,
                panel_confidence_evidence,
                strict=True,
            )
        ):
            if (
                valid.corner_x != confidence.corner_x
                or valid.width != confidence.width
            ):
                raise RuntimeError(
                    "Inspection panel valid/confidence footprints differ"
                )
            x0 = int(valid.corner_x)
            x1 = x0 + int(valid.width)
            component_local = component[:, x0:x1]
            valid_local = np.asarray(valid.values, dtype=bool)
            coverage = int(
                np.count_nonzero(component_local & valid_local)
            )
            complete = coverage == area
            local_adjacent = abs(panel_index - preferred) <= 1
            mean_confidence = (
                float(
                    np.mean(
                        np.asarray(
                            confidence.values,
                            dtype=np.float32,
                        )[component_local]
                    )
                )
                if complete
                else 0.0
            )
            candidate_rows.append(
                {
                    "panel_index": panel_index,
                    "coverage_pixel_count": coverage,
                    "complete_coverage": bool(complete),
                    "near_nominal_owner": bool(local_adjacent),
                    "mean_confidence": mean_confidence,
                }
            )
            if complete and local_adjacent:
                candidates.append(
                    (
                        mean_confidence,
                        -abs(panel_index - preferred),
                        -panel_index,
                    )
                )
        if not candidates:
            component_audits.append(
                {
                    "component_id": int(label),
                    "area_pixels": area,
                    "centroid_x": centroid_x,
                    "preferred_panel_index": preferred,
                    "selected_panel_index": None,
                    "complete_adjacent_panel_coverage": False,
                    "candidates": candidate_rows,
                }
            )
            continue
        selected_panel = -max(candidates)[2]
        locked[component] = np.int16(selected_panel)
        locked_pixels += area
        component_audits.append(
            {
                "component_id": int(label),
                "area_pixels": area,
                "centroid_x": centroid_x,
                "preferred_panel_index": preferred,
                "selected_panel_index": selected_panel,
                "complete_adjacent_panel_coverage": True,
                "candidates": candidate_rows,
            }
        )
    return locked, {
        "policy": (
            "connected_foreground_depth_risk_component_locked_to_one_"
            "nearby_fully_covering_real_panel_before_seam_dp"
        ),
        "component_count": len(component_audits),
        "locked_pixel_count": int(locked_pixels),
        "unassigned_component_count": int(
            sum(
                item["selected_panel_index"] is None
                for item in component_audits
            )
        ),
        "all_components_assigned": bool(
            all(
                item["selected_panel_index"] is not None
                for item in component_audits
            )
        ),
        "components": component_audits,
    }


def _compose_reference_panels_graphcut_multiband(
    rasters: Sequence[_ReferencePanelRaster],
    foreground_source_images: Sequence[np.ndarray],
    layout: InspectionMultiviewLayout,
    global_foreground_mask: np.ndarray,
    locked_foreground_panel_index: np.ndarray,
    foreground_lock_audit: Mapping[str, object],
    pre_seam_hard_owner_intervals: Sequence[
        InspectionPreSeamHardOwnerInterval
    ],
    config: InspectionMultiviewConfig,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
    list[_ReferencePanelRaster],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    if len(rasters) < 2:
        raise RuntimeError("Inspection GraphCut needs at least two panels")
    if len(foreground_source_images) != len(rasters):
        raise RuntimeError(
            "Inspection foreground source images are not aligned with panels"
        )
    corners = [(int(item.corner_x), 0) for item in rasters]
    recovered_safe, dis_pair_audits = _dis_safe_background_recovery(
        rasters, config
    )
    full_valid = np.zeros((layout.height, layout.width), dtype=bool)
    safe_coverage = np.zeros((layout.height, layout.width), dtype=np.uint16)
    for item, recovered in zip(rasters, recovered_safe, strict=True):
        x0 = int(item.corner_x)
        x1 = x0 + item.valid_mask.shape[1]
        full_valid[:, x0:x1] |= item.valid_mask
        safe_coverage[:, x0:x1] += (
            item.valid_mask & (~item.protected_mask | recovered)
        ).astype(np.uint16)
    foreground = np.asarray(global_foreground_mask, dtype=bool)
    if foreground.shape != full_valid.shape:
        raise RuntimeError(
            "Inspection foreground protection mask does not match canvas"
        )
    # A geometrically near surface is not automatically a fragile object.
    # When two or more panels independently recover the same low-motion,
    # low-gradient, photometrically consistent support, it is a stable
    # structural plane (for example the yellow shelf) and may use the safe
    # background seam/blend path.  Object edges, depth holes and view-variant
    # content remain protected.
    protected_foreground = foreground & (safe_coverage < 2)
    foreground_locks = np.asarray(
        locked_foreground_panel_index, dtype=np.int16
    )
    if foreground_locks.shape != full_valid.shape:
        raise RuntimeError(
            "Inspection foreground owner locks do not match canvas"
        )
    if np.any(
        (foreground_locks < -1)
        | (foreground_locks >= len(rasters))
    ):
        raise RuntimeError(
            "Inspection foreground owner locks contain an invalid panel"
        )
    if foreground_lock_audit.get("all_components_locked") is not True:
        raise RuntimeError(
            "Inspection foreground component lacks a complete real panel"
        )
    base_foreground_locks = foreground_locks.copy()
    # External identity intervals are validated and admitted one at a time
    # after the adaptive boundary evidence is ready.  One impossible object
    # lock must not erase another valid lock or turn an otherwise auditable
    # C-grade hard-owner result into a generic structural failure.
    (
        pre_seam_locks,
        pre_seam_owner_only_guard,
        pre_seam_interval_audit,
    ) = _prepare_pre_seam_hard_owner_intervals(
        (),
        rasters,
        full_valid.shape,
    )
    # Reliable foreground is a real occluding layer over the monotone
    # background chain.  Two depth-layer components can legitimately
    # interleave in image x on the same row, so forcing their frame IDs into
    # the background chain would cut one of the objects.  The selected panel
    # is therefore consumed by the inverse-mesh foreground overlay below;
    # only the background seams themselves must remain monotone.
    # Protect the depth-mesh foreground at its final virtual location.  Source
    # guards are view-dependent and move with parallax; taking their union
    # across all panels would black-list most of a long scan.  Pixels for which
    # every real panel is unsafe remain hard-owner as well.
    protected_union = protected_foreground | (
        full_valid & (safe_coverage == 0)
    )

    # A protected pixel in any source is excluded from every exposure/seam
    # observation.  This prevents another panel from blending across a
    # foreground silhouette, depth hole, transparent surface, or depth edge.
    safe_masks: list[np.ndarray] = []
    exposure_masks: list[np.ndarray] = []
    for item, recovered in zip(rasters, recovered_safe, strict=True):
        x0 = int(item.corner_x)
        x1 = x0 + item.valid_mask.shape[1]
        safe = (
            item.valid_mask
            & (~item.protected_mask | recovered)
            & ~protected_foreground[:, x0:x1]
        )
        safe_masks.append(np.ascontiguousarray(safe.astype(np.uint8) * 255))
        # Exposure estimation needs pairwise safe overlap, not safety against
        # every panel at once.  A foreground shifted by parallax in a distant
        # panel must not erase otherwise valid wall observations from an
        # adjacent pair.  The resulting gains may be applied globally, while
        # actual blending below still excludes the global protected union.
        exposure_safe = item.valid_mask & (
            ~item.protected_mask | recovered
        )
        exposure_masks.append(
            np.ascontiguousarray(exposure_safe.astype(np.uint8) * 255)
        )

    compensated_images = [
        np.ascontiguousarray(item.image_bgr.copy()) for item in rasters
    ]
    exposure_applied = False
    exposure_reason = "insufficient_safe_background"
    exposure_gain_stats: list[dict[str, object]] = []
    channel_gains_bgr = np.ones((len(rasters), 3), dtype=np.float32)
    if all(np.count_nonzero(mask) >= 64 for mask in exposure_masks):
        try:
            compensator = cv2.detail.ExposureCompensator_createDefault(
                cv2.detail.ExposureCompensator_CHANNELS
            )
            compensator.feed(corners, compensated_images, exposure_masks)
            gains = compensator.getMatGains()
            for index, (item, image, mask) in enumerate(
                zip(rasters, compensated_images, exposure_masks, strict=True)
            ):
                adjusted = compensator.apply(
                    index, corners[index], image, mask
                )
                if adjusted is not None:
                    compensated_images[index] = np.ascontiguousarray(adjusted)
                gain = np.asarray(gains[index], dtype=np.float32)
                # OpenCV's CHANNELS compensator exposes a homogeneous fourth
                # slot whose value is zero; only B, G and R are image gains.
                color_gain = gain.reshape(-1)[:3]
                if (
                    color_gain.size != 3
                    or not np.isfinite(color_gain).all()
                    or np.any(color_gain <= 0.0)
                ):
                    raise RuntimeError(
                        "Inspection exposure compensation produced invalid gains"
                    )
                channel_gains_bgr[index] = color_gain
                exposure_gain_stats.append(
                    {
                        "frame_id": int(item.frame_id),
                        "minimum": float(np.min(color_gain)),
                        "median": float(np.median(color_gain)),
                        "maximum": float(np.max(color_gain)),
                    }
                )
            exposure_applied = True
            exposure_reason = "safe_background_global_gain"
        except (cv2.error, RuntimeError) as exc:
            compensated_images = [
                np.ascontiguousarray(item.image_bgr.copy()) for item in rasters
            ]
            exposure_gain_stats = []
            channel_gains_bgr.fill(1.0)
            exposure_reason = (
                "opencv_exposure_compensation_failed:"
                f"{type(exc).__name__}:{str(exc)}"
            )

    adjacent_residual_gains = np.ones(len(rasters), dtype=np.float32)
    adjacent_pair_audits: list[dict[str, object]] = []
    if exposure_applied:
        cumulative_log = np.zeros(len(rasters), dtype=np.float64)
        for index in range(len(rasters) - 1):
            left = rasters[index]
            right = rasters[index + 1]
            overlap_x0 = max(left.corner_x, right.corner_x)
            overlap_x1 = min(
                left.corner_x + left.image_bgr.shape[1],
                right.corner_x + right.image_bgr.shape[1],
            )
            ratio = 1.0
            support = 0
            if overlap_x1 > overlap_x0:
                left_slice = slice(
                    overlap_x0 - left.corner_x,
                    overlap_x1 - left.corner_x,
                )
                right_slice = slice(
                    overlap_x0 - right.corner_x,
                    overlap_x1 - right.corner_x,
                )
                common = (
                    exposure_masks[index][:, left_slice] > 0
                ) & (exposure_masks[index + 1][:, right_slice] > 0)
                left_linear = srgb_to_linear_bgr(
                    compensated_images[index][:, left_slice]
                )
                right_linear = srgb_to_linear_bgr(
                    compensated_images[index + 1][:, right_slice]
                )
                left_luma = (
                    0.0722 * left_linear[..., 0]
                    + 0.7152 * left_linear[..., 1]
                    + 0.2126 * left_linear[..., 2]
                )
                right_luma = (
                    0.0722 * right_linear[..., 0]
                    + 0.7152 * right_linear[..., 1]
                    + 0.2126 * right_linear[..., 2]
                )
                photometric = (
                    common
                    & (left_luma > 0.03)
                    & (right_luma > 0.03)
                    & (left_luma < 0.95)
                    & (right_luma < 0.95)
                )
                support = int(np.count_nonzero(photometric))
                if support >= 64:
                    samples = np.clip(
                        left_luma[photometric]
                        / np.maximum(right_luma[photometric], 1e-6),
                        0.5,
                        2.0,
                    )
                    ratio = float(np.median(samples))
            applied_ratio = float(np.clip(ratio, 0.97, 1.03))
            cumulative_log[index + 1] = (
                cumulative_log[index]
                + math.log(max(applied_ratio, 1e-6))
            )
            adjacent_pair_audits.append(
                {
                    "left_frame_id": int(left.frame_id),
                    "right_frame_id": int(right.frame_id),
                    "safe_support_pixel_count": support,
                    "left_to_right_residual_gain": ratio,
                    "regularized_applied_ratio": applied_ratio,
                }
            )
        cumulative_log -= float(np.median(cumulative_log))
        adjacent_residual_gains = np.clip(
            np.exp(cumulative_log), 0.92, 1.08
        ).astype(np.float32)
        for index, residual_gain in enumerate(adjacent_residual_gains):
            linear = srgb_to_linear_bgr(compensated_images[index])
            linear *= residual_gain
            compensated_images[index] = linear_to_srgb_bgr(linear)
        exposure_reason = (
            "opencv_global_gain_then_adjacent_safe_background_residual_chain"
        )

    # A near structural plane may have been protected solely because it is
    # closer than D0.  Admit it to the background blender only where two
    # adjacent compensated panels directly prove low colour residual and low
    # gradient at the same canvas pixels.  This removes hard-owner scallops on
    # flat shelves/walls without relaxing protection at object silhouettes.
    stable_structural_canvas = np.zeros(full_valid.shape, dtype=bool)
    stable_structural_pair_audits: list[dict[str, object]] = []
    for index in range(len(rasters) - 1):
        left = rasters[index]
        right = rasters[index + 1]
        overlap_x0 = max(left.corner_x, right.corner_x)
        overlap_x1 = min(
            left.corner_x + left.image_bgr.shape[1],
            right.corner_x + right.image_bgr.shape[1],
        )
        accepted = np.zeros(
            (layout.height, max(0, overlap_x1 - overlap_x0)), dtype=bool
        )
        if overlap_x1 > overlap_x0:
            left_slice = slice(
                overlap_x0 - left.corner_x,
                overlap_x1 - left.corner_x,
            )
            right_slice = slice(
                overlap_x0 - right.corner_x,
                overlap_x1 - right.corner_x,
            )
            left_image = compensated_images[index][:, left_slice]
            right_image = compensated_images[index + 1][:, right_slice]
            left_lab = cv2.cvtColor(left_image, cv2.COLOR_BGR2LAB).astype(
                np.float32
            )
            right_lab = cv2.cvtColor(right_image, cv2.COLOR_BGR2LAB).astype(
                np.float32
            )
            residual = np.linalg.norm(left_lab - right_lab, axis=2)
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
            left_gradient = np.maximum(
                np.abs(
                    cv2.Sobel(
                        left_gray, cv2.CV_32F, 1, 0, ksize=3
                    )
                ),
                np.abs(
                    cv2.Sobel(
                        left_gray, cv2.CV_32F, 0, 1, ksize=3
                    )
                ),
            )
            right_gradient = np.maximum(
                np.abs(
                    cv2.Sobel(
                        right_gray, cv2.CV_32F, 1, 0, ksize=3
                    )
                ),
                np.abs(
                    cv2.Sobel(
                        right_gray, cv2.CV_32F, 0, 1, ksize=3
                    )
                ),
            )
            common = (
                left.valid_mask[:, left_slice]
                & right.valid_mask[:, right_slice]
            )
            accepted = (
                common
                & (
                    residual
                    <= np.float32(
                        config.dis_maximum_rgb_residual * 3.0
                    )
                )
                & (
                    left_gradient
                    <= np.float32(config.dis_maximum_gradient)
                )
                & (
                    right_gradient
                    <= np.float32(config.dis_maximum_gradient)
                )
            )
            safe_masks[index][:, left_slice][accepted] = 255
            safe_masks[index + 1][:, right_slice][accepted] = 255
            stable_structural_canvas[
                :, overlap_x0:overlap_x1
            ] |= accepted
        stable_structural_pair_audits.append(
            {
                "left_frame_id": int(left.frame_id),
                "right_frame_id": int(right.frame_id),
                "accepted_pixel_count": int(np.count_nonzero(accepted)),
            }
        )
    protected_union &= ~stable_structural_canvas
    per_panel_exposure_safe_pixel_counts = [
        int(np.count_nonzero(mask)) for mask in exposure_masks
    ]
    dis_recovered_safe_background_pixel_count = int(
        sum(np.count_nonzero(mask) for mask in recovered_safe)
    )
    # Exposure masks and DIS recovery masks have completed their only
    # full-resolution roles.  Do not carry one extra panel-local mask per
    # source into GraphCut and MultiBand, where native pyramid storage is the
    # actual peak working-set consumer.
    del exposure_masks
    del recovered_safe

    input_masks = [mask.copy() for mask in safe_masks]
    graphcut_scale = float(config.graphcut_preview_scale)
    if graphcut_scale < 1.0:
        # Convert one source at a time before resizing.  Retaining a float32
        # full-resolution copy for every panel would add 12 bytes per source
        # pixel even though GraphCut consumes only the configured preview.
        graphcut_images = [
            cv2.resize(
                np.ascontiguousarray(image, dtype=np.float32),
                None,
                fx=graphcut_scale,
                fy=graphcut_scale,
                interpolation=cv2.INTER_AREA,
            )
            for image in compensated_images
        ]
        graphcut_masks = [
            cv2.resize(
                mask,
                (graphcut_images[index].shape[1], graphcut_images[index].shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            for index, mask in enumerate(input_masks)
        ]
        graphcut_corners = [
            (int(round(x * graphcut_scale)), int(round(y * graphcut_scale)))
            for x, y in corners
        ]
    else:
        graphcut_images = [
            np.ascontiguousarray(image, dtype=np.float32)
            for image in compensated_images
        ]
        graphcut_masks = input_masks
        graphcut_corners = corners
    try:
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        output = finder.find(
            graphcut_images, graphcut_corners, graphcut_masks
        )
    except cv2.error as exc:
        raise RuntimeError("Inspection background GraphCut failed") from exc
    result_masks = graphcut_masks if output is None else list(output)
    masks: list[np.ndarray] = []
    for item, mask in zip(rasters, result_masks, strict=True):
        value = mask.get() if hasattr(mask, "get") else mask
        value = np.asarray(value, dtype=np.uint8)
        if graphcut_scale < 1.0:
            value = cv2.resize(
                value,
                (item.valid_mask.shape[1], item.valid_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            value = cv2.bitwise_and(value, input_masks[len(masks)])
        if value.shape != item.valid_mask.shape:
            raise RuntimeError("Inspection GraphCut returned an invalid mask shape")
        masks.append(np.ascontiguousarray(value))

    # OpenCV supplies the photometric/gradient seam preference, but its
    # unconstrained multi-image masks may jump backwards or select nonadjacent
    # panels.  Project that evidence onto one closed left-to-right chain of
    # N-1 adjacent, full-height seams before any RGB is composed.
    graphcut_hint_masks = [mask.copy() for mask in masks]
    panel_valid_evidence: list[PanelLocalEvidence] = []
    graphcut_hint_evidence: list[PanelLocalEvidence] = []
    for item, hint in zip(rasters, graphcut_hint_masks, strict=True):
        panel_valid_evidence.append(
            PanelLocalEvidence(
                corner_x=int(item.corner_x),
                values=np.asarray(item.valid_mask, dtype=bool),
                canvas_width=int(layout.width),
            )
        )
        graphcut_hint_evidence.append(
            PanelLocalEvidence(
                corner_x=int(item.corner_x),
                values=np.asarray(hint > 0, dtype=bool),
                canvas_width=int(layout.width),
            )
        )
    # The compact boolean hint evidence above is the only GraphCut result
    # consumed by the closed-chain solver.  Release preview float images,
    # native finder/output wrappers, and the duplicate input masks before
    # constructing pair corridors.
    del graphcut_images
    del graphcut_masks
    del result_masks
    del input_masks
    del graphcut_hint_masks
    del finder
    del output
    panel_centers = [
        float(item.corner_x) + 0.5 * float(item.valid_mask.shape[1] - 1)
        for item in rasters
    ]
    geometric_nominal_boundaries = [
        0.5 * (panel_centers[index] + panel_centers[index + 1])
        for index in range(len(panel_centers) - 1)
    ]
    endpoint_boundary_outward_bias = min(
        320.0,
        max(
            0.0,
            2.0
            * (
                geometric_nominal_boundaries[1]
                - geometric_nominal_boundaries[0]
            ),
        )
        if len(geometric_nominal_boundaries) > 1
        else 0.0,
    )
    if endpoint_boundary_outward_bias > 0.0:
        geometric_nominal_boundaries[0] -= (
            endpoint_boundary_outward_bias
        )
    chain_config = ChainSeamConfig(
        corridor_width_pixels=int(
            config.chain_seam_corridor_width_pixels
        ),
        maximum_row_step_pixels=int(
            config.chain_seam_maximum_row_step_pixels
        ),
        smoothness_penalty=2.0,
        adaptive_boundary_maximum_shift_pixels=int(
            config.chain_seam_adaptive_boundary_maximum_shift_pixels
        ),
        adaptive_boundary_risk_guard_pixels=int(
            config.chain_seam_adaptive_boundary_risk_guard_pixels
        ),
        adaptive_boundary_minimum_common_coverage_ratio=float(
            config.chain_seam_adaptive_boundary_minimum_common_coverage_ratio
        ),
        adaptive_boundary_shift_penalty=float(
            config.chain_seam_adaptive_boundary_shift_penalty
        ),
        hard_cut_fallback_enabled=bool(
            config.chain_seam_hard_cut_fallback_enabled
        ),
    )
    # Boundary planning sees the full virtual depth-mesh union, plus pixels
    # that no panel can expose safely.  Each adjacent pair also contributes
    # both reference-panel protected masks so depth holes, transparent/
    # reflective objects and invalid-depth devices cannot be cut merely
    # because they produced no reliable foreground mesh.
    globally_unsafe = full_valid & (safe_coverage == 0)
    global_pair_risk = protected_foreground | globally_unsafe
    global_pair_risk_count = int(np.count_nonzero(global_pair_risk))
    pair_boundary_risk_evidence: list[PairCorridorEvidence] = []
    pair_boundary_risk_pixel_counts: list[int] = []
    for index in range(len(rasters) - 1):
        pair_items = (rasters[index], rasters[index + 1])
        pair_x0 = min(int(item.corner_x) for item in pair_items)
        pair_x1 = max(
            int(item.corner_x) + item.valid_mask.shape[1]
            for item in pair_items
        )
        base_local = global_pair_risk[:, pair_x0:pair_x1]
        risk_local = base_local.copy()
        for item in pair_items:
            item_x0 = int(item.corner_x)
            item_x1 = item_x0 + item.valid_mask.shape[1]
            risk_local[
                :,
                item_x0 - pair_x0 : item_x1 - pair_x0,
            ] |= item.protected_mask & item.valid_mask
        pair_boundary_risk_pixel_counts.append(
            global_pair_risk_count
            - int(np.count_nonzero(base_local))
            + int(np.count_nonzero(risk_local))
        )
        original_center = int(
            round(geometric_nominal_boundaries[index])
        )
        maximum_shift = int(
            chain_config.adaptive_boundary_maximum_shift_pixels
        )
        corridor_width = int(chain_config.corridor_width_pixels)
        minimum_center = max(
            corridor_width // 2,
            original_center - maximum_shift,
        )
        maximum_center = min(
            layout.width - (corridor_width - corridor_width // 2),
            original_center + maximum_shift,
        )
        planning_x0 = max(
            0,
            min(
                minimum_center - corridor_width // 2,
                minimum_center
                - int(
                    chain_config.adaptive_boundary_risk_guard_pixels
                ),
            ),
        )
        planning_x1 = min(
            layout.width,
            max(
                maximum_center
                + (corridor_width - corridor_width // 2),
                maximum_center
                + int(
                    chain_config.adaptive_boundary_risk_guard_pixels
                )
                + 1,
            ),
        )
        planning_risk = global_pair_risk[
            :, planning_x0:planning_x1
        ].copy()
        for item in pair_items:
            item_x0 = max(int(item.corner_x), planning_x0)
            item_x1 = min(
                int(item.corner_x) + item.valid_mask.shape[1],
                planning_x1,
            )
            if item_x1 <= item_x0:
                continue
            source = slice(
                item_x0 - int(item.corner_x),
                item_x1 - int(item.corner_x),
            )
            planning_risk[
                :,
                item_x0 - planning_x0 : item_x1 - planning_x0,
            ] |= (
                item.protected_mask[:, source]
                & item.valid_mask[:, source]
            )
        pair_boundary_risk_evidence.append(
            PairCorridorEvidence(
                corner_x=planning_x0,
                values=np.ascontiguousarray(planning_risk),
                canvas_width=int(layout.width),
            )
        )
    adaptive_boundaries = select_adaptive_nominal_boundaries(
        panel_valid_evidence,
        geometric_nominal_boundaries,
        pair_boundary_risk_evidence,
        target_valid_mask=full_valid,
        locked_owner_panel_index=None,
        config=chain_config,
    )
    requested_pre_seam_interval_count = len(
        pre_seam_hard_owner_intervals
    )
    decoupled_pre_seam_intervals = tuple(
        interval
        for interval in pre_seam_hard_owner_intervals
        if interval.deferred_true_depth_identity_overlay
        or not interval.background_panel_lock_required
    )
    ordinary_pre_seam_intervals = tuple(
        interval
        for interval in pre_seam_hard_owner_intervals
        if not interval.deferred_true_depth_identity_overlay
        and interval.background_panel_lock_required
    )
    accepted_pre_seam_intervals: list[
        InspectionPreSeamHardOwnerInterval
    ] = []
    rejected_pre_seam_intervals: list[dict[str, object]] = []
    for interval in ordinary_pre_seam_intervals:
        trial_intervals = (
            *accepted_pre_seam_intervals,
            interval,
        )
        trial_locks, _, _ = _prepare_pre_seam_hard_owner_intervals(
            trial_intervals,
            rasters,
            full_valid.shape,
        )
        try:
            select_adaptive_nominal_boundaries(
                panel_valid_evidence,
                geometric_nominal_boundaries,
                pair_boundary_risk_evidence,
                target_valid_mask=full_valid,
                locked_owner_panel_index=trial_locks,
                config=chain_config,
            )
        except RuntimeError as exc:
            if "no lock-compatible candidate" not in str(exc):
                raise
            rejected_pre_seam_intervals.append(
                {
                    "track_id": int(interval.track_id),
                    "panel_index": int(interval.panel_index),
                    "frame_id": int(interval.frame_id),
                    "lock_pixel_count": int(
                        np.count_nonzero(interval.lock_mask)
                    ),
                    "reason": str(exc),
                    "outcome": "hard_cut_degraded_not_applied",
                }
            )
            continue
        accepted_pre_seam_intervals.append(interval)
    pre_seam_hard_owner_intervals = (
        *accepted_pre_seam_intervals,
        *decoupled_pre_seam_intervals,
    )
    (
        pre_seam_locks,
        pre_seam_owner_only_guard,
        pre_seam_interval_audit,
    ) = _prepare_pre_seam_hard_owner_intervals(
        pre_seam_hard_owner_intervals,
        rasters,
        full_valid.shape,
    )
    foreground_pre_seam_conflict = (
        (base_foreground_locks >= 0)
        & (pre_seam_locks >= 0)
        & (base_foreground_locks != pre_seam_locks)
    )
    foreground_pre_seam_conflict_count = int(
        np.count_nonzero(foreground_pre_seam_conflict)
    )
    foreground_locks = np.where(
        pre_seam_locks >= 0,
        pre_seam_locks,
        base_foreground_locks,
    ).astype(np.int16, copy=False)
    pre_seam_interval_audit.update(
        {
            "requested_interval_count": len(
                accepted_pre_seam_intervals
            ) + len(rejected_pre_seam_intervals),
            "rejected_interval_count": len(
                rejected_pre_seam_intervals
            ),
            "rejected_intervals": rejected_pre_seam_intervals,
            "existing_foreground_lock_conflict_pixel_count": (
                foreground_pre_seam_conflict_count
            ),
            "identity_lock_precedence_over_anonymous_component": True,
        }
    )
    nominal_boundaries = list(adaptive_boundaries.selected_boundaries_x)
    corridor_width = int(config.chain_seam_corridor_width_pixels)
    corridor_bounds: list[tuple[int, int]] = []
    for nominal_x in nominal_boundaries:
        corridor_x0 = int(round(nominal_x)) - corridor_width // 2
        corridor_x0 = min(
            max(0, corridor_x0), layout.width - corridor_width
        )
        corridor_bounds.append(
            (corridor_x0, corridor_x0 + corridor_width)
        )
    pair_cost_evidence: list[PairCorridorEvidence] = []
    for index, (nominal_x, corridor) in enumerate(
        zip(nominal_boundaries, corridor_bounds, strict=True)
    ):
        left = rasters[index]
        right = rasters[index + 1]
        overlap_x0 = max(int(left.corner_x), int(right.corner_x))
        overlap_x1 = min(
            int(left.corner_x) + left.image_bgr.shape[1],
            int(right.corner_x) + right.image_bgr.shape[1],
        )
        corridor_x0, corridor_x1 = corridor
        cost = np.full(
            (layout.height, corridor_width),
            np.float32(255.0),
            dtype=np.float32,
        )
        if overlap_x1 > overlap_x0:
            left_slice = slice(
                overlap_x0 - int(left.corner_x),
                overlap_x1 - int(left.corner_x),
            )
            right_slice = slice(
                overlap_x0 - int(right.corner_x),
                overlap_x1 - int(right.corner_x),
            )
            left_lab = cv2.cvtColor(
                compensated_images[index][:, left_slice],
                cv2.COLOR_BGR2LAB,
            ).astype(np.float32)
            right_lab = cv2.cvtColor(
                compensated_images[index + 1][:, right_slice],
                cv2.COLOR_BGR2LAB,
            ).astype(np.float32)
            residual = np.linalg.norm(left_lab - right_lab, axis=2)
            common = (
                panel_valid_evidence[index].window(
                    overlap_x0,
                    overlap_x1,
                    dtype=np.dtype(bool),
                )
                & panel_valid_evidence[index + 1].window(
                    overlap_x0,
                    overlap_x1,
                    dtype=np.dtype(bool),
                )
            )
            local_cost = np.where(common, residual, np.float32(255.0))
            # Keep the monotone path near OpenCV's adjacent GraphCut choice
            # whenever that choice exists on the row.
            local_x = np.arange(overlap_x0, overlap_x1, dtype=np.float32)
            left_hint = graphcut_hint_evidence[index].window(
                overlap_x0,
                overlap_x1,
                dtype=np.dtype(bool),
            )
            right_hint = graphcut_hint_evidence[index + 1].window(
                overlap_x0,
                overlap_x1,
                dtype=np.dtype(bool),
            )
            for row in range(layout.height):
                left_columns = np.flatnonzero(left_hint[row])
                right_columns = np.flatnonzero(right_hint[row])
                if left_columns.size and right_columns.size:
                    hint_x = 0.5 * (
                        float(left_columns[-1]) + float(right_columns[0])
                    ) + overlap_x0
                else:
                    hint_x = float(nominal_x)
                local_cost[row] += (
                    np.abs(local_x - hint_x).astype(np.float32) * 0.10
                )
            cost_x0 = max(overlap_x0, corridor_x0)
            cost_x1 = min(overlap_x1, corridor_x1)
            if cost_x1 > cost_x0:
                cost[
                    :,
                    cost_x0 - corridor_x0 : cost_x1 - corridor_x0,
                ] = local_cost[
                    :,
                    cost_x0 - overlap_x0 : cost_x1 - overlap_x0,
                ]
        pair_cost_evidence.append(
            PairCorridorEvidence(
                corner_x=int(corridor_x0),
                values=np.ascontiguousarray(cost),
                canvas_width=int(layout.width),
            )
        )
    protected_evidence: list[PanelLocalEvidence] = []
    confidence_evidence: list[PanelLocalEvidence] = []
    for item in rasters:
        protected_evidence.append(
            PanelLocalEvidence(
                corner_x=int(item.corner_x),
                values=np.asarray(item.protected_mask, dtype=bool),
                canvas_width=int(layout.width),
            )
        )
        confidence_evidence.append(
            PanelLocalEvidence(
                corner_x=int(item.corner_x),
                values=np.asarray(item.confidence, dtype=np.float32),
                canvas_width=int(layout.width),
            )
        )
    invalid_depth_owner_only_mask = np.zeros(full_valid.shape, dtype=bool)
    for pair_index, (corridor_x0, corridor_x1) in enumerate(
        corridor_bounds
    ):
        invalid_depth_owner_only_mask[
            :, corridor_x0:corridor_x1
        ] |= (
            protected_evidence[pair_index].window(
                corridor_x0,
                corridor_x1,
                dtype=np.dtype(bool),
            )
            | protected_evidence[pair_index + 1].window(
                corridor_x0,
                corridor_x1,
                dtype=np.dtype(bool),
            )
        ) & ~protected_foreground[
            :, corridor_x0:corridor_x1
        ] & ~stable_structural_canvas[:, corridor_x0:corridor_x1]
    (
        invalid_depth_locks,
        invalid_depth_lock_audit,
    ) = _plan_chain_foreground_owner_locks(
        invalid_depth_owner_only_mask,
        panel_valid_evidence,
        confidence_evidence,
        nominal_boundaries,
        minimum_component_pixels=int(
            config.minimum_foreground_component_pixels
        ),
    )
    chain_target = np.zeros(full_valid.shape, dtype=bool)
    for index, panel_valid in enumerate(panel_valid_evidence):
        core_x0 = 0 if index == 0 else corridor_bounds[index - 1][1]
        core_x1 = (
            layout.width
            if index == len(panel_valid_evidence) - 1
            else corridor_bounds[index][0]
        )
        if core_x1 > core_x0:
            chain_target[:, core_x0:core_x1] = panel_valid.window(
                core_x0,
                core_x1,
                dtype=np.dtype(bool),
            )
    for index, (corridor_x0, corridor_x1) in enumerate(corridor_bounds):
        chain_target[:, corridor_x0:corridor_x1] = (
            panel_valid_evidence[index].window(
                corridor_x0,
                corridor_x1,
                dtype=np.dtype(bool),
            )
            | panel_valid_evidence[index + 1].window(
                corridor_x0,
                corridor_x1,
                dtype=np.dtype(bool),
            )
        )
    # Adaptive boundary compatibility is necessary but not sufficient: a
    # lock can still make the full-height, bounded-row-step DP infeasible.
    # Admit each interval only after the complete chain solver succeeds.
    fully_accepted_pre_seam_intervals: list[
        InspectionPreSeamHardOwnerInterval
    ] = []
    for interval in accepted_pre_seam_intervals:
        trial_intervals = (
            *fully_accepted_pre_seam_intervals,
            interval,
        )
        trial_locks, _, _ = _prepare_pre_seam_hard_owner_intervals(
            trial_intervals,
            rasters,
            full_valid.shape,
        )
        try:
            solve_adjacent_panel_chain(
                panel_valid_evidence,
                nominal_boundaries,
                pair_costs=pair_cost_evidence,
                seam_forbidden_masks=None,
                target_valid_mask=chain_target,
                locked_owner_panel_index=trial_locks,
                config=chain_config,
            )
        except RuntimeError as exc:
            if not any(
                marker in str(exc)
                for marker in (
                    "without a feasible closed boundary",
                    "has no top-to-bottom feasible path",
                    "could not produce a closed monotone topology",
                    "has a row without a feasible closed boundary",
                )
            ):
                raise
            rejected_pre_seam_intervals.append(
                {
                    "track_id": int(interval.track_id),
                    "panel_index": int(interval.panel_index),
                    "frame_id": int(interval.frame_id),
                    "lock_pixel_count": int(
                        np.count_nonzero(interval.lock_mask)
                    ),
                    "reason": str(exc),
                    "outcome": "hard_cut_degraded_not_applied",
                }
            )
            continue
        fully_accepted_pre_seam_intervals.append(interval)
    pre_seam_hard_owner_intervals = (
        *fully_accepted_pre_seam_intervals,
        *decoupled_pre_seam_intervals,
    )
    (
        pre_seam_locks,
        pre_seam_owner_only_guard,
        pre_seam_interval_audit,
    ) = _prepare_pre_seam_hard_owner_intervals(
        pre_seam_hard_owner_intervals,
        rasters,
        full_valid.shape,
    )
    foreground_pre_seam_conflict = (
        (base_foreground_locks >= 0)
        & (pre_seam_locks >= 0)
        & (base_foreground_locks != pre_seam_locks)
    )
    foreground_pre_seam_conflict_count = int(
        np.count_nonzero(foreground_pre_seam_conflict)
    )
    foreground_locks = np.where(
        pre_seam_locks >= 0,
        pre_seam_locks,
        base_foreground_locks,
    ).astype(np.int16, copy=False)
    pre_seam_interval_audit.update(
        {
            "requested_interval_count": (
                requested_pre_seam_interval_count
            ),
            "rejected_interval_count": len(
                rejected_pre_seam_intervals
            ),
            "rejected_intervals": rejected_pre_seam_intervals,
            "existing_foreground_lock_conflict_pixel_count": (
                foreground_pre_seam_conflict_count
            ),
            "identity_lock_precedence_over_anonymous_component": True,
        }
    )
    # Foreground/protected components are forbidden seam support.  Once a
    # seam cannot cross a component, the monotone background partition does
    # not need to reproduce the foreground z-order.  Reliable foreground is
    # inverse-mesh overlaid from its selected source after background
    # composition; invalid-depth content remains the single panel naturally
    # selected on the one side of the avoiding seam.
    lock_candidates: list[
        tuple[int, int, int, int, int, np.ndarray]
    ] = []
    for panel_index in range(len(rasters)):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (foreground_locks == panel_index).astype(np.uint8), 8
        )
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            if area < 1000 or width < 40 or height < 24:
                continue
            selected_lock = labels == label
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            lock_candidates.append(
                (
                    panel_index,
                    area,
                    x0,
                    x0 + width,
                    int(stats[label, cv2.CC_STAT_TOP]),
                    selected_lock,
                )
            )
    selected_candidate_ids: set[int] = set()
    half_corridor = int(config.chain_seam_corridor_width_pixels) // 2
    for pair_index, boundary in enumerate(nominal_boundaries):
        corridor_x0 = int(round(boundary)) - half_corridor
        corridor_x1 = corridor_x0 + int(
            config.chain_seam_corridor_width_pixels
        )
        eligible = [
            (candidate_id, candidate)
            for candidate_id, candidate in enumerate(lock_candidates)
            if candidate[0] in {pair_index, pair_index + 1}
            and candidate[2] < corridor_x1
            and candidate[3] > corridor_x0
        ]
        if not eligible:
            continue
        candidate_id, candidate = max(
            eligible, key=lambda item: item[1][1]
        )
        selected_candidate_ids.add(candidate_id)
    protected_union |= (
        invalid_depth_owner_only_mask | pre_seam_owner_only_guard
    )
    # Solve every geometrically covered RGB pixel, including the valid
    # non-rectangular upper/lower extent. The formal inspection product is
    # still cropped later with largest_valid_rectangle(), while the full-
    # extent RGBA browse product can retain real edge objects without filling
    # invalid corners or moving their coordinates.
    excluded_locked_pixels = int(np.count_nonzero(
        (foreground_locks >= 0) & ~chain_target
    ))
    if excluded_locked_pixels:
        raise RuntimeError(
            "Inspection rectangular target excludes locked foreground pixels"
        )
    pre_seam_target_excluded = int(
        np.count_nonzero(pre_seam_owner_only_guard & ~chain_target)
    )
    if pre_seam_target_excluded:
        raise RuntimeError(
            "Inspection rectangular target excludes a pre-seam hard-owner "
            "interval"
        )
    pre_seam_interval_audit["excluded_by_target_pixel_count"] = 0

    def solve_with_candidate_ids(
        candidate_ids: Sequence[int],
        *,
        candidate_records: Sequence[
            tuple[int, int, int, int, int, np.ndarray]
        ] = lock_candidates,
        target_mask: np.ndarray = chain_target,
        valid_evidence: Sequence[
            PanelLocalEvidence
        ] = panel_valid_evidence,
        cost_evidence: Sequence[
            PairCorridorEvidence
        ] = pair_cost_evidence,
    ) -> tuple[object, np.ndarray]:
        # RGB object ownership is deliberately independent of the spatial
        # background panel chain.  Pre-seam intervals protect their footprints
        # and are copied/audited after the background composition; feeding
        # their panel index into this solver can make an otherwise valid
        # closed background boundary impossible when the RGB owner originates
        # from a different physical view.  Only optional foreground candidates
        # below may constrain the background chain, and each is admitted
        # transactionally.
        combined = np.full(pre_seam_locks.shape, -1, dtype=np.int16)
        for candidate_id in candidate_ids:
            panel_index, _, _, _, _, selected_lock = (
                candidate_records[candidate_id]
            )
            available = selected_lock & (
                (combined < 0) | (combined == panel_index)
            )
            combined[available] = np.int16(panel_index)
        combined[~target_mask] = -1
        result = solve_adjacent_panel_chain(
            valid_evidence,
            nominal_boundaries,
            pair_costs=cost_evidence,
            seam_forbidden_masks=None,
            target_valid_mask=target_mask,
            locked_owner_panel_index=(
                combined if np.any(combined >= 0) else None
            ),
            config=chain_config,
        )
        return result, combined

    # A conflict between two noisy candidate components must not erase every
    # otherwise feasible object lock.  Add candidates in descending measured
    # area and retain each only when the complete closed seam chain remains
    # feasible.  Every rejection is explicit in the audit and later makes
    # strict object-completeness false; there is no silent all-lock fallback.
    try:
        chain_result, combined_foreground_locks = solve_with_candidate_ids(())
    except RuntimeError as exc:
        # A virtual panel can have a one-row non-rectangular lower/upper edge
        # that no adjacent pair jointly covers.  It cannot contribute to the
        # final largest-valid crop.  Exclude it only when it is in the outer
        # ten percent and contains no required object guard, then re-solve the
        # complete background chain; all interior/object rows remain fail-closed.
        match = re.search(r"first_failed_rows=\[([^\]]*)\]", str(exc))
        rows = (
            []
            if match is None or not match.group(1).strip()
            else [int(value.strip()) for value in match.group(1).split(",")]
        )
        edge_limit = max(1, int(round(0.10 * layout.height)))
        edge_rows = [
            row for row in rows
            if row < edge_limit or row >= layout.height - edge_limit
        ]
        if (
            not rows
            or rows != edge_rows
            or np.any(pre_seam_owner_only_guard[edge_rows])
        ):
            raise
        chain_target[edge_rows, :] = False
        chain_result, combined_foreground_locks = solve_with_candidate_ids(())
    accepted_candidate_ids: list[int] = []
    rejected_candidate_ids: list[int] = []
    lock_rejection_reasons: dict[int, str] = {}
    candidate_area_by_id = {
        int(candidate_id): int(lock_candidates[candidate_id][1])
        for candidate_id in selected_candidate_ids
    }
    ordered_candidate_ids = sorted(
        selected_candidate_ids,
        key=lambda candidate_id: (
            -candidate_area_by_id[int(candidate_id)],
            int(candidate_id),
        ),
    )
    for candidate_id in ordered_candidate_ids:
        trial_ids = (*accepted_candidate_ids, candidate_id)
        try:
            trial_result, trial_locks = solve_with_candidate_ids(
                trial_ids
            )
        except RuntimeError as exc:
            if not any(
                marker in str(exc)
                for marker in (
                    "without a feasible closed boundary",
                    "has no top-to-bottom feasible path",
                    "could not produce a closed monotone topology",
                    "has a row without a feasible closed boundary",
                )
            ):
                raise
            rejected_candidate_ids.append(candidate_id)
            lock_rejection_reasons[candidate_id] = str(exc)
            continue
        accepted_candidate_ids.append(candidate_id)
        chain_result = trial_result
        combined_foreground_locks = trial_locks
    retained_lock_component_count = len(accepted_candidate_ids)
    retained_lock_pixel_count = int(
        np.count_nonzero(combined_foreground_locks >= 0)
    )
    retained_chain_locks_applied = bool(
        accepted_candidate_ids
    )
    chain_lock_candidate_count = len(selected_candidate_ids)
    rejected_chain_lock_audits = [
        {
            "candidate_id": int(candidate_id),
            "panel_index": int(lock_candidates[candidate_id][0]),
            "area_pixels": int(lock_candidates[candidate_id][1]),
            "reason": lock_rejection_reasons[candidate_id],
        }
        for candidate_id in rejected_candidate_ids
    ]
    del solve_with_candidate_ids
    full_valid = chain_result.valid_mask.copy()
    pre_seam_chain_mismatch = int(
        np.count_nonzero(
            (pre_seam_locks >= 0)
            & (chain_result.owner_panel_index != pre_seam_locks)
        )
    )
    if pre_seam_chain_mismatch:
        raise RuntimeError(
            "Inspection chain seam did not preserve a pre-seam hard-owner "
            "interval"
        )
    pre_seam_interval_audit[
        "solver_locked_owner_mismatch_pixel_count"
    ] = 0
    protected_union &= full_valid
    masks = []
    for index, item in enumerate(rasters):
        x0 = int(item.corner_x)
        x1 = x0 + item.valid_mask.shape[1]
        selected = (
            chain_result.owner_panel_index[:, x0:x1] == int(index)
        ) & item.valid_mask
        masks.append(np.ascontiguousarray(selected.astype(np.uint8) * 255))

    hard_owner = np.full((layout.height, layout.width), -1, dtype=np.int32)
    hard_valid = np.zeros((layout.height, layout.width), dtype=bool)
    multiply_owned = 0
    for item, mask in zip(rasters, masks, strict=True):
        x0 = item.corner_x
        x1 = x0 + mask.shape[1]
        selected = mask > 0
        existing = hard_valid[:, x0:x1]
        multiply_owned += int(np.count_nonzero(existing & selected))
        take = selected & ~existing
        hard_owner[:, x0:x1][take] = int(item.frame_id)
        hard_valid[:, x0:x1][take] = True
    if multiply_owned:
        # OpenCV may leave a narrow shared support band.  It is valid for
        # MultiBand but the audit owner remains deterministic first-panel.
        pass

    compact_evidence_storage_audit = {
        "model": "panel_local_and_pair_corridor/v1",
        "per_panel_full_canvas_array_count": 0,
        "per_pair_full_canvas_array_count": 0,
        "panel_valid_bytes": int(
            sum(
                np.asarray(item.values).nbytes
                for item in panel_valid_evidence
            )
        ),
        "graphcut_hint_bytes": int(
            sum(
                np.asarray(item.values).nbytes
                for item in graphcut_hint_evidence
            )
        ),
        "pair_boundary_risk_bytes": int(
            sum(
                np.asarray(item.values).nbytes
                for item in pair_boundary_risk_evidence
            )
        ),
        "pair_cost_bytes": int(
            sum(
                np.asarray(item.values).nbytes
                for item in pair_cost_evidence
            )
        ),
        "protected_bytes": int(
            sum(
                np.asarray(item.values).nbytes
                for item in protected_evidence
            )
        ),
        "confidence_bytes": int(
            sum(
                np.asarray(item.values).nbytes
                for item in confidence_evidence
            )
        ),
    }
    invalid_depth_seam_risk_pixel_count = int(
        np.count_nonzero(invalid_depth_owner_only_mask)
    )
    invalid_depth_locked_pixel_count = int(
        np.count_nonzero(invalid_depth_locks >= 0)
    )
    invalid_depth_cropped_lock_pixel_count = int(
        np.count_nonzero((invalid_depth_locks >= 0) & ~chain_target)
    )
    # Every decision derived from these compact arrays is now immutable in
    # chain_result or in the scalar audit above.  Releasing them before DIS
    # and MultiBand prevents the solver evidence from overlapping native
    # optical-flow and pyramid allocations at the working-set peak.
    del panel_valid_evidence
    del graphcut_hint_evidence
    del pair_boundary_risk_evidence
    del pair_cost_evidence
    del protected_evidence
    del confidence_evidence
    del invalid_depth_owner_only_mask
    del invalid_depth_locks
    del combined_foreground_locks
    del chain_target
    del lock_candidates

    local_dis_alignment_audits: list[dict[str, object]] = []
    for pair_index, seam in enumerate(chain_result.seams):
        left = rasters[pair_index]
        right = rasters[pair_index + 1]
        overlap_x0 = max(left.corner_x, right.corner_x)
        overlap_x1 = min(
            left.corner_x + left.image_bgr.shape[1],
            right.corner_x + right.image_bgr.shape[1],
        )
        applied_count = 0
        maximum_motion = 0.0
        maximum_fb_error = 0.0
        if overlap_x1 > overlap_x0:
            left_slice = slice(
                overlap_x0 - left.corner_x,
                overlap_x1 - left.corner_x,
            )
            right_slice = slice(
                overlap_x0 - right.corner_x,
                overlap_x1 - right.corner_x,
            )
            left_image = compensated_images[pair_index][:, left_slice]
            right_image = compensated_images[pair_index + 1][:, right_slice]
            scale = float(config.dis_preview_scale)
            small_size = (
                max(8, int(round(left_image.shape[1] * scale))),
                max(8, int(round(left_image.shape[0] * scale))),
            )
            left_small = cv2.resize(
                cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY),
                small_size,
                interpolation=cv2.INTER_AREA,
            )
            right_small = cv2.resize(
                cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY),
                small_size,
                interpolation=cv2.INTER_AREA,
            )
            dis = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
            )
            forward = dis.calc(left_small, right_small, None)
            backward = dis.calc(right_small, left_small, None)
            small_y, small_x = np.indices(
                left_small.shape, dtype=np.float32
            )
            backward_on_forward = cv2.remap(
                backward,
                small_x + forward[..., 0],
                small_y + forward[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            fb_small = np.linalg.norm(
                forward + backward_on_forward, axis=2
            )
            scale_x = left_image.shape[1] / float(small_size[0])
            scale_y = left_image.shape[0] / float(small_size[1])
            flow_x = cv2.resize(
                forward[..., 0],
                (left_image.shape[1], left_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) * np.float32(scale_x)
            flow_y = cv2.resize(
                forward[..., 1],
                (left_image.shape[1], left_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) * np.float32(scale_y)
            fb_error = cv2.resize(
                fb_small,
                (left_image.shape[1], left_image.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            ) * np.float32(max(scale_x, scale_y))
            motion = np.sqrt(flow_x * flow_x + flow_y * flow_y)
            local_y, local_x = np.indices(
                left_image.shape[:2], dtype=np.float32
            )
            aligned_right = accelerated_remap(
                right_image,
                local_x + flow_x,
                local_y + flow_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            seam_x = np.asarray(seam.seam_x_by_row, dtype=np.int32)
            global_columns = np.arange(
                overlap_x0, overlap_x1, dtype=np.int32
            )
            seam_band = (
                np.abs(global_columns[None, :] - seam_x[:, None]) <= 4
            )
            candidate = (
                seam_band
                & stable_structural_canvas[:, overlap_x0:overlap_x1]
                & left.valid_mask[:, left_slice]
                & right.valid_mask[:, right_slice]
                & ~pre_seam_owner_only_guard[
                    :, overlap_x0:overlap_x1
                ]
            )
            safe = (
                candidate
                & (motion <= np.float32(config.dis_maximum_motion_pixels))
                & (
                    fb_error
                    <= np.float32(config.dis_maximum_fb_error_pixels)
                )
                & left.valid_mask[:, left_slice]
                & right.valid_mask[:, right_slice]
            )
            right_region = compensated_images[pair_index + 1][
                :, right_slice
            ]
            right_region[safe] = aligned_right[safe]
            applied_count = int(np.count_nonzero(safe))
            maximum_motion = float(np.max(motion[safe], initial=0.0))
            maximum_fb_error = float(
                np.max(fb_error[safe], initial=0.0)
            )
            candidate_motion_p50 = float(
                np.quantile(motion[candidate], 0.50)
            ) if np.any(candidate) else 0.0
            candidate_motion_p95 = float(
                np.quantile(motion[candidate], 0.95)
            ) if np.any(candidate) else 0.0
            candidate_fb_p50 = float(
                np.quantile(fb_error[candidate], 0.50)
            ) if np.any(candidate) else 0.0
            candidate_fb_p95 = float(
                np.quantile(fb_error[candidate], 0.95)
            ) if np.any(candidate) else 0.0
        else:
            candidate_motion_p50 = 0.0
            candidate_motion_p95 = 0.0
            candidate_fb_p50 = 0.0
            candidate_fb_p95 = 0.0
        local_dis_alignment_audits.append(
            {
                "left_frame_id": int(left.frame_id),
                "right_frame_id": int(right.frame_id),
                "inverse_sampled_pixel_count": applied_count,
                "maximum_motion_pixels": maximum_motion,
                "maximum_fb_error_pixels": maximum_fb_error,
                "candidate_motion_p50_pixels": candidate_motion_p50,
                "candidate_motion_p95_pixels": candidate_motion_p95,
                "candidate_fb_error_p50_pixels": candidate_fb_p50,
                "candidate_fb_error_p95_pixels": candidate_fb_p95,
                "maximum_band_half_width_pixels": 4,
                "protected_pixel_count": 0,
                "pre_seam_hard_owner_intersection_pixel_count": int(
                    np.count_nonzero(
                        safe
                        & pre_seam_owner_only_guard[
                            :, overlap_x0:overlap_x1
                        ]
                    )
                ) if overlap_x1 > overlap_x0 else 0,
            }
        )

    blend_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    blend_masks = [
        cv2.bitwise_and(
            cv2.dilate(mask, blend_kernel),
            safe_mask,
        )
        for mask, safe_mask in zip(masks, safe_masks, strict=True)
    ]
    per_panel_global_safe_blend_pixel_counts = [
        int(np.count_nonzero(mask)) for mask in safe_masks
    ]
    del safe_masks
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(3)
    blender.prepare((0, 0, layout.width, layout.height))
    try:
        for item, image, mask in zip(
            rasters, compensated_images, blend_masks, strict=True
        ):
            blender.feed(
                np.ascontiguousarray(image, dtype=np.int16),
                mask,
                (int(item.corner_x), 0),
            )
        blended, blended_mask = blender.blend(None, None)
    except cv2.error as exc:
        raise RuntimeError("Inspection background MultiBand failed") from exc
    blended = np.clip(np.asarray(blended), 0, 255).astype(np.uint8)
    blended_valid = np.asarray(blended_mask, dtype=np.uint8) > 0
    if blended.shape != (layout.height, layout.width, 3):
        raise RuntimeError("Inspection MultiBand returned an invalid canvas")
    # Fill protected and otherwise uncovered support from exactly one
    # exposure-adjusted real RGB panel.  Confidence/first-panel tie breaking
    # is deterministic and never averages owners.
    hard_image = np.zeros_like(blended)
    hard_score = np.full(
        (layout.height, layout.width), -np.inf, dtype=np.float32
    )
    for item, image, mask in zip(
        rasters, compensated_images, masks, strict=True
    ):
        x0 = int(item.corner_x)
        x1 = x0 + item.valid_mask.shape[1]
        seam_selected = mask > 0
        seam_take = seam_selected & (
            hard_score[:, x0:x1] == -np.inf
        )
        hard_image[:, x0:x1][seam_take] = image[seam_take]
        hard_score[:, x0:x1][seam_take] = item.confidence[seam_take] + 2.0
    del hard_score
    del blend_masks
    effective_blended_valid = (
        blended_valid & ~protected_union & full_valid
    )
    direct_copy = protected_union | ~effective_blended_valid
    zero_weight_wedge = (
        effective_blended_valid
        & np.all(blended == 0, axis=2)
        & np.any(hard_image != 0, axis=2)
    )
    blended_luma = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    hard_luma = cv2.cvtColor(hard_image, cv2.COLOR_BGR2GRAY)
    dark_weight_wedge = (
        effective_blended_valid
        & (hard_luma >= 16)
        & (
            blended_luma.astype(np.float32)
            < 0.25 * hard_luma.astype(np.float32)
        )
    )
    direct_copy |= zero_weight_wedge | dark_weight_wedge
    direct_copy &= full_valid
    blended[direct_copy] = hard_image[direct_copy]
    output_valid = effective_blended_valid | direct_copy
    missing_owner = output_valid & (hard_owner < 0)
    if np.any(missing_owner):
        raise RuntimeError(
            "Inspection hard owner is incomplete after safe background blending"
        )
    hard_owner[~output_valid] = -1
    cross_panel_rgb_transfer_rows: list[dict[str, object]] = []
    for interval in pre_seam_hard_owner_intervals:
        if interval.deferred_true_depth_identity_overlay:
            continue
        spatial_panel_index = int(interval.panel_index)
        source_panel_index = (
            spatial_panel_index
            if interval.rgb_source_panel_index is None
            else int(interval.rgb_source_panel_index)
        )
        lock = (
            np.asarray(interval.lock_mask, dtype=bool)
            if interval.rgb_transfer_mask is None
            else np.asarray(interval.rgb_transfer_mask, dtype=bool)
        )
        source_raster = rasters[source_panel_index]
        source_x0 = int(source_raster.corner_x)
        source_x1 = source_x0 + int(source_raster.valid_mask.shape[1])
        local_lock = lock[:, source_x0:source_x1]
        if int(np.count_nonzero(local_lock)) != int(
            np.count_nonzero(lock)
        ):
            raise RuntimeError(
                "Inspection pre-seam RGB source lock escapes its panel"
            )
        if np.any(local_lock & ~source_raster.valid_mask):
            raise RuntimeError(
                "Inspection pre-seam RGB source has invalid lock pixels"
            )
        source_image = compensated_images[source_panel_index]
        destination_image = blended[:, source_x0:source_x1]
        exact_local = np.asarray(
            interval.union_footprint[:, source_x0:source_x1],
            dtype=bool,
        )
        exact_guard = cv2.dilate(
            exact_local.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ).astype(bool)
        source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
        destination_gray = cv2.cvtColor(
            destination_image, cv2.COLOR_BGR2GRAY
        )
        source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
        destination_lab = cv2.cvtColor(
            destination_image, cv2.COLOR_BGR2LAB
        )
        source_gradient = (
            np.abs(
                cv2.Sobel(
                    source_gray,
                    cv2.CV_16S,
                    1,
                    0,
                    ksize=3,
                )
            )
            + np.abs(
                cv2.Sobel(
                    source_gray,
                    cv2.CV_16S,
                    0,
                    1,
                    ksize=3,
                )
            )
        )
        destination_gradient = (
            np.abs(
                cv2.Sobel(
                    destination_gray,
                    cv2.CV_16S,
                    1,
                    0,
                    ksize=3,
                )
            )
            + np.abs(
                cv2.Sobel(
                    destination_gray,
                    cv2.CV_16S,
                    0,
                    1,
                    ksize=3,
                )
            )
        )
        calibration_support = (
            local_lock
            & ~exact_guard
            & output_valid[:, source_x0:source_x1]
            & (source_gray >= 24)
            & (source_gray <= 232)
            & (destination_gray >= 24)
            & (destination_gray <= 232)
            & (source_gradient <= 24)
            & (destination_gradient <= 24)
            & (
                np.abs(
                    source_lab[..., 0].astype(np.int16)
                    - destination_lab[..., 0].astype(np.int16)
                )
                <= 16
            )
            & (
                np.abs(
                    source_lab[..., 1].astype(np.int16)
                    - destination_lab[..., 1].astype(np.int16)
                )
                <= 12
            )
            & (
                np.abs(
                    source_lab[..., 2].astype(np.int16)
                    - destination_lab[..., 2].astype(np.int16)
                )
                <= 12
            )
        )
        calibration_pixel_count = int(
            np.count_nonzero(calibration_support)
        )
        local_gain = np.ones(3, dtype=np.float32)
        local_offset = np.zeros(3, dtype=np.float32)
        if calibration_pixel_count >= 64:
            source_luma_values = source_gray[
                calibration_support
            ].astype(np.float32)
            destination_luma_values = destination_gray[
                calibration_support
            ].astype(np.float32)
            valid_luma_ratio = source_luma_values >= 16.0
            luma_ratios = (
                destination_luma_values[valid_luma_ratio]
                / source_luma_values[valid_luma_ratio]
            )
            robust_luma_gain = 1.0
            if luma_ratios.size >= 64:
                lower, upper = np.quantile(
                    luma_ratios, (0.10, 0.90)
                )
                trimmed_luma = luma_ratios[
                    (luma_ratios >= lower)
                    & (luma_ratios <= upper)
                ]
                if trimmed_luma.size:
                    robust_luma_gain = float(
                        np.clip(
                            np.median(trimmed_luma),
                            0.85,
                            1.18,
                        )
                    )
            for channel in range(3):
                source_values = source_image[..., channel][
                    calibration_support
                ].astype(np.float32)
                destination_values = destination_image[..., channel][
                    calibration_support
                ].astype(np.float32)
                valid_ratio = source_values >= 16.0
                ratios = (
                    destination_values[valid_ratio]
                    / source_values[valid_ratio]
                )
                if ratios.size < 64:
                    continue
                lower, upper = np.quantile(ratios, (0.10, 0.90))
                trimmed = ratios[
                    (ratios >= lower) & (ratios <= upper)
                ]
                if trimmed.size:
                    local_gain[channel] = np.float32(
                        np.clip(
                            np.median(trimmed),
                            max(0.85, robust_luma_gain - 0.04),
                            min(1.18, robust_luma_gain + 0.04),
                        )
                    )
        corrected_source = np.clip(
            source_image.astype(np.float32)
            * local_gain[None, None, :],
            0.0,
            255.0,
        )
        corrected_source = np.clip(
            corrected_source + local_offset[None, None, :],
            0.0,
            255.0,
        ).astype(np.uint8)
        # The object itself remains a hard copy from one real RGB owner.
        # Only a narrow low-gradient, chromatically consistent background
        # strip at the outside of its context corridor may use MultiBand.
        # This hides exposure steps without averaging object contours, text,
        # depth boundaries, or the exact FastSAM support.
        boundary_distance = cv2.distanceTransform(
            local_lock.astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        safe_background_boundary = (
            local_lock
            & (boundary_distance <= 6.0)
            & ~exact_guard
            & output_valid[:, source_x0:source_x1]
            & (source_gradient <= 24)
            & (destination_gradient <= 24)
            & (
                np.abs(
                    source_lab[..., 1].astype(np.int16)
                    - destination_lab[..., 1].astype(np.int16)
                )
                <= 12
            )
            & (
                np.abs(
                    source_lab[..., 2].astype(np.int16)
                    - destination_lab[..., 2].astype(np.int16)
                )
                <= 12
            )
        )
        local_multiband_pixel_count = int(
            np.count_nonzero(safe_background_boundary)
        )
        transfer_source = corrected_source.copy()
        if local_multiband_pixel_count:
            yy, xx = np.nonzero(local_lock)
            y0 = max(0, int(yy.min()) - 8)
            y1 = min(local_lock.shape[0], int(yy.max()) + 9)
            x0 = max(0, int(xx.min()) - 8)
            x1 = min(local_lock.shape[1], int(xx.max()) + 9)
            roi_destination = np.ascontiguousarray(
                destination_image[y0:y1, x0:x1]
            )
            roi_source = np.ascontiguousarray(
                corrected_source[y0:y1, x0:x1]
            )
            roi_valid = (
                output_valid[:, source_x0:source_x1][
                    y0:y1, x0:x1
                ].astype(np.uint8)
                * np.uint8(255)
            )
            roi_source_mask = (
                local_lock[y0:y1, x0:x1].astype(np.uint8)
                * np.uint8(255)
            )
            local_blender = cv2.detail_MultiBandBlender()
            local_blender.setNumBands(3)
            local_blender.prepare(
                (0, 0, int(x1 - x0), int(y1 - y0))
            )
            try:
                local_blender.feed(
                    roi_destination.astype(np.int16),
                    roi_valid,
                    (0, 0),
                )
                local_blender.feed(
                    roi_source.astype(np.int16),
                    roi_source_mask,
                    (0, 0),
                )
                local_blended, local_blended_mask = (
                    local_blender.blend(None, None)
                )
            except cv2.error as exc:
                raise RuntimeError(
                    "Inspection object-corridor background MultiBand failed"
                ) from exc
            local_blended = np.clip(
                np.asarray(local_blended), 0, 255
            ).astype(np.uint8)
            local_blended_valid = (
                np.asarray(local_blended_mask, dtype=np.uint8) > 0
            )
            local_safe = (
                safe_background_boundary[y0:y1, x0:x1]
                & local_blended_valid
            )
            transfer_source[y0:y1, x0:x1][local_safe] = (
                local_blended[local_safe]
            )
        blended[:, source_x0:source_x1][local_lock] = (
            transfer_source[local_lock]
        )
        hard_owner[:, source_x0:source_x1][local_lock] = int(
            interval.frame_id
        )
        cross_panel_rgb_transfer_rows.append(
            {
                "track_id": int(interval.track_id),
                "spatial_panel_index": spatial_panel_index,
                "rgb_source_panel_index": source_panel_index,
                "rgb_source_frame_id": int(interval.frame_id),
                "pixel_count": int(np.count_nonzero(lock)),
                "local_context_photometric_calibration_pixel_count": (
                    calibration_pixel_count
                ),
                "local_context_linear_bgr_gain": [
                    float(value) for value in local_gain
                ],
                "local_context_linear_bgr_offset": [
                    float(value) for value in local_offset
                ],
                "local_context_gain_applied": bool(
                    np.any(np.abs(local_gain - 1.0) > 1e-6)
                    or np.any(np.abs(local_offset) > 1e-6)
                ),
                "safe_background_boundary_multiband_pixel_count": (
                    local_multiband_pixel_count
                ),
                "safe_background_boundary_multiband_width_pixels": 6,
                "safe_background_boundary_multiband_band_count": 3,
                "exact_object_support_multiband_intersection_pixel_count": 0,
                "existing_reference_inverse_raster_only": True,
                "novel_view_warp_used": False,
                "alpha_or_multiband_used": False,
            }
        )
    pre_seam_final_owner_mismatch = 0
    pre_seam_final_invalid = 0
    for interval in pre_seam_hard_owner_intervals:
        if interval.deferred_true_depth_identity_overlay:
            continue
        lock = (
            np.asarray(interval.lock_mask, dtype=bool)
            if interval.rgb_transfer_mask is None
            else np.asarray(interval.rgb_transfer_mask, dtype=bool)
        )
        pre_seam_final_owner_mismatch += int(
            np.count_nonzero(lock & (hard_owner != int(interval.frame_id)))
        )
        pre_seam_final_invalid += int(
            np.count_nonzero(lock & ~output_valid)
        )
    if pre_seam_final_owner_mismatch or pre_seam_final_invalid:
        raise RuntimeError(
            "Inspection final background owner did not preserve a pre-seam "
            "hard-owner interval"
        )
    pre_seam_interval_audit["final_owner_mismatch_pixel_count"] = 0
    pre_seam_interval_audit["final_invalid_pixel_count"] = 0
    pre_seam_interval_audit["rgb_source_transfers"] = (
        cross_panel_rgb_transfer_rows
    )
    pre_seam_interval_audit["spatial_panel_and_rgb_source_decoupled"] = bool(
        any(
            int(row["spatial_panel_index"])
            != int(row["rgb_source_panel_index"])
            for row in cross_panel_rgb_transfer_rows
        )
    )
    compensated_rasters = [
        _ReferencePanelRaster(
            panel_index=int(item.panel_index),
            frame_id=int(item.frame_id),
            corner_x=int(item.corner_x),
            image_bgr=np.ascontiguousarray(image),
            valid_mask=item.valid_mask,
            protected_mask=item.protected_mask,
            confidence=item.confidence,
            reference_map_x=item.reference_map_x,
            reference_map_y=item.reference_map_y,
        )
        for item, image in zip(
            rasters, compensated_images, strict=True
        )
    ]
    compensated_foreground_sources: list[np.ndarray] = []
    for source_image, channel_gain, residual_gain in zip(
        foreground_source_images,
        channel_gains_bgr,
        adjacent_residual_gains,
        strict=True,
    ):
        channel_adjusted = np.clip(
            np.asarray(source_image, dtype=np.float32)
            * channel_gain[None, None, :],
            0.0,
            255.0,
        ).astype(np.uint8)
        linear = srgb_to_linear_bgr(channel_adjusted)
        linear *= np.float32(residual_gain)
        compensated_foreground_sources.append(
            np.ascontiguousarray(linear_to_srgb_bgr(linear))
        )
    protected_blend_intersection = int(
        np.count_nonzero(effective_blended_valid & protected_union)
    )
    pre_seam_blend_intersection = int(
        np.count_nonzero(
            effective_blended_valid & pre_seam_owner_only_guard
        )
    )
    pre_seam_flow_intersection = int(
        sum(
            int(item["pre_seam_hard_owner_intersection_pixel_count"])
            for item in local_dis_alignment_audits
        )
    )
    if pre_seam_blend_intersection or pre_seam_flow_intersection:
        raise RuntimeError(
            "Inspection pre-seam hard-owner interval intersected blend or flow"
        )
    pre_seam_interval_audit[
        "multiband_intersection_pixel_count"
    ] = 0
    pre_seam_interval_audit["dis_flow_intersection_pixel_count"] = 0
    return (
        blended,
        hard_owner,
        output_valid,
        {
            "backend": (
                "opencv_safe_background_exposure_graphcut_then_multiband"
            ),
            "graphcut_used": True,
            "graphcut_preview_scale": graphcut_scale,
            "graphcut_role": (
                "adjacent_pair_photometric_gradient_hint_for_closed_"
                "monotone_chain"
            ),
            "panel_chain_topology": {
                **chain_result.audit,
                "adaptive_boundary_selection": {
                    **adaptive_boundaries.as_dict(),
                    "risk_sources": [
                        "full_depth_mesh_union",
                        "globally_unsafe_reference_coverage",
                        "left_right_reference_protected_masks",
                    ],
                    "risk_pixel_counts": (
                        pair_boundary_risk_pixel_counts
                    ),
                    "risk_is_seam_forbidden": False,
                    "risk_usage": (
                        "adaptive_nominal_boundary_selection; foreground_"
                        "components_use_explicit_single_panel_owner_locks"
                    ),
                },
            },
            "compact_evidence_storage": compact_evidence_storage_audit,
            "foreground_component_locks": {
                **dict(foreground_lock_audit),
                "excluded_by_target_pixel_count": excluded_locked_pixels,
                "solver_role": (
                    "large_complete_component_single_panel_chain_lock"
                ),
                "retained_chain_lock_component_count": (
                    retained_lock_component_count
                ),
                "retained_chain_lock_pixel_count": (
                    retained_lock_pixel_count
                ),
                "candidate_chain_lock_component_count": (
                    chain_lock_candidate_count
                ),
                "rejected_chain_lock_component_count": len(
                    rejected_chain_lock_audits
                ),
                "rejected_chain_lock_candidates": (
                    rejected_chain_lock_audits
                ),
                "retained_chain_locks_applied": (
                    retained_chain_locks_applied
                ),
                "solver_locked_owner_mismatch_pixel_count": 0,
            },
            "pre_seam_hard_owner_intervals": pre_seam_interval_audit,
            "invalid_depth_owner_only_locks": {
                **invalid_depth_lock_audit,
                "seam_risk_pixel_count": (
                    invalid_depth_seam_risk_pixel_count
                ),
                "locked_pixel_count": invalid_depth_locked_pixel_count,
                "cropped_outside_largest_closed_rectangle_pixel_count": (
                    invalid_depth_cropped_lock_pixel_count
                ),
                "reference_plane_rgb_allowed": True,
                "multiband_allowed": False,
            },
            "panel_chain_seams": [
                seam.as_dict() for seam in chain_result.seams
            ],
            "multiband_used": True,
            "multiband_levels": 3,
            "exposure_compensation_used": exposure_applied,
            "exposure_compensation_method": exposure_reason,
            "exposure_gain_statistics": exposure_gain_stats,
            "adjacent_residual_gain_minimum": float(
                np.min(adjacent_residual_gains)
            ),
            "adjacent_residual_gain_maximum": float(
                np.max(adjacent_residual_gains)
            ),
            "adjacent_residual_gain_statistics": [
                {
                    "frame_id": int(item.frame_id),
                    "gain": float(gain),
                }
                for item, gain in zip(
                    rasters, adjacent_residual_gains, strict=True
                )
            ],
            "adjacent_exposure_pair_audits": adjacent_pair_audits,
            "stable_structural_plane_recovery": {
                "policy": (
                    "adjacent_compensated_low_lab_residual_low_gradient_"
                    "shared_rgb_support"
                ),
                "accepted_pixel_count": int(
                    np.count_nonzero(stable_structural_canvas)
                ),
                "pairs": stable_structural_pair_audits,
            },
            "local_dis_seam_alignment": {
                "policy": (
                    "bidirectional_fb_audited_inverse_sampling_only_in_"
                    "stable_structural_four_pixel_seam_band"
                ),
                "inverse_sampled_pixel_count": int(
                    sum(
                        item["inverse_sampled_pixel_count"]
                        for item in local_dis_alignment_audits
                    )
                ),
                "protected_pixel_count": 0,
                "pairs": local_dis_alignment_audits,
            },
            "continuous_canvas_exposure": {
                "applied": False,
                "reason": "deferred_until_after_foreground_owner_replacement",
            },
            "per_panel_exposure_safe_pixel_counts": (
                per_panel_exposure_safe_pixel_counts
            ),
            "per_panel_global_safe_blend_pixel_counts": (
                per_panel_global_safe_blend_pixel_counts
            ),
            "panel_count": len(rasters),
            "dis_optical_flow_used": True,
            "dis_policy": (
                "adjacent_bidirectional_fb_low_motion_low_rgb_residual_"
                "low_gradient_invalid_depth_background_recovery"
            ),
            "dis_pair_audits": dis_pair_audits,
            "dis_recovered_safe_background_pixel_count": (
                dis_recovered_safe_background_pixel_count
            ),
            "graphcut_shared_support_pixel_count": multiply_owned,
            "protected_pixel_count": int(np.count_nonzero(protected_union)),
            "protected_blend_intersection_pixel_count": (
                protected_blend_intersection
            ),
            "direct_single_owner_pixel_count": int(
                np.count_nonzero(direct_copy)
            ),
            "valid_pixel_count": int(np.count_nonzero(output_valid)),
        },
        compensated_rasters,
        compensated_foreground_sources,
        np.ascontiguousarray(protected_union),
        np.ascontiguousarray(chain_result.owner_panel_index),
    )


def _update_global_surface(
    *,
    output_image: np.ndarray,
    output_depth: np.ndarray,
    output_confidence: np.ndarray,
    output_owner: np.ndarray,
    output_reliable_depth: np.ndarray,
    source_image: np.ndarray,
    source_confidence: np.ndarray,
    source_edge: np.ndarray,
    source_reliable_depth: np.ndarray,
    frame_id: int,
    source_pixel_index: np.ndarray,
    canvas_x: np.ndarray,
    canvas_y: np.ndarray,
    relative_depth: np.ndarray,
    config: InspectionMultiviewConfig,
) -> tuple[int, int, int]:
    width = output_depth.shape[1]
    flat = canvas_y.astype(np.int64) * width + canvas_x.astype(np.int64)
    # Resolve collisions inside this source/chunk before touching the shared
    # z-buffer.  Edge confidence and source pixel index are stable tie-breaks.
    candidate_confidence = source_confidence.reshape(-1)[source_pixel_index]
    candidate_reliable = source_reliable_depth.reshape(-1)[source_pixel_index]
    order = np.lexsort(
        (source_pixel_index, -candidate_confidence, relative_depth, flat)
    )
    sorted_flat = flat[order]
    first = np.empty(sorted_flat.size, dtype=bool)
    first[0] = True
    first[1:] = sorted_flat[1:] != sorted_flat[:-1]
    selected = order[first]
    destination = flat[selected]
    candidate_depth = relative_depth[selected].astype(np.float32)
    candidate_confidence = candidate_confidence[selected]
    candidate_reliable = candidate_reliable[selected]
    output_depth_flat = output_depth.reshape(-1)
    output_confidence_flat = output_confidence.reshape(-1)
    output_owner_flat = output_owner.reshape(-1)
    output_reliable_flat = output_reliable_depth.reshape(-1)
    existing = output_owner_flat[destination] >= 0
    existing_reliable = output_reliable_flat[destination]
    existing_depth = output_depth_flat[destination]
    comparison_depth = np.where(existing, existing_depth, candidate_depth)
    tolerance = np.maximum(
        config.temporal_absolute_tolerance_mm,
        config.temporal_relative_tolerance
        * np.maximum(candidate_depth, comparison_depth),
    )
    nearer = existing & (candidate_depth < comparison_depth - tolerance)
    same_layer = existing & (
        np.abs(candidate_depth - comparison_depth) <= tolerance
    )
    better_same = same_layer & (
        candidate_confidence > output_confidence_flat[destination] + 1e-6
    )
    reliability_upgrade = existing & candidate_reliable & ~existing_reliable
    same_reliability_class = candidate_reliable == existing_reliable
    take = (
        (~existing)
        | reliability_upgrade
        | (existing & same_reliability_class & (nearer | better_same))
    )
    chosen_destination = destination[take]
    chosen = selected[take]
    chosen_source_pixel = source_pixel_index[chosen]
    output_image.reshape(-1, 3)[chosen_destination] = (
        source_image.reshape(-1, 3)[chosen_source_pixel]
    )
    output_depth_flat[chosen_destination] = relative_depth[chosen].astype(
        np.float32
    )
    output_confidence_flat[chosen_destination] = (
        source_confidence.reshape(-1)[chosen_source_pixel]
    )
    output_owner_flat[chosen_destination] = int(frame_id)
    output_reliable_flat[chosen_destination] = (
        source_reliable_depth.reshape(-1)[chosen_source_pixel]
    )
    edge_count = int(
        np.count_nonzero(source_edge.reshape(-1)[chosen_source_pixel])
    )
    return int(np.count_nonzero(take)), int(np.count_nonzero(same_layer)), edge_count


def _foreground_component_mask(
    valid: np.ndarray,
    depth: np.ndarray,
    reference_depth_mm: float,
    config: InspectionMultiviewConfig,
) -> np.ndarray:
    # Depth edges and their guards remain protected from GraphCut/MultiBand,
    # but they must not bridge separate objects into one giant ownership
    # component.  Only connected reliable near-surface interiors define a
    # foreground object for the whole-component owner rule.
    foreground = valid & (
        depth < reference_depth_mm * config.foreground_reference_depth_ratio
    )
    return np.ascontiguousarray(foreground)


def _foreground_depth_layer_components(
    valid: np.ndarray,
    depth: np.ndarray,
    reference_depth_mm: float,
    config: InspectionMultiviewConfig,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Label connected near surfaces without joining distinct depth layers."""

    foreground = _foreground_component_mask(
        valid, depth, reference_depth_mm, config
    )
    labels = np.zeros(foreground.shape, dtype=np.int32)
    components: list[tuple[int, int]] = []
    if not np.any(foreground):
        return labels, components
    physical_tolerance = max(
        config.temporal_absolute_tolerance_mm,
        config.temporal_relative_tolerance * reference_depth_mm,
    )
    # Twice the required same-layer tolerance avoids noisy one-bin fragments,
    # while surfaces separated by a meaningful depth discontinuity cannot be
    # bridged merely because their image masks touch.
    band_width = max(1.0, 2.0 * physical_tolerance)
    depth_band = np.floor(
        np.where(foreground, np.maximum(depth, 0.0), 0.0) / band_width
    ).astype(np.int32)
    next_label = 1
    for band in np.unique(depth_band[foreground]):
        band_mask = foreground & (depth_band == int(band))
        count, local_labels, stats, _ = cv2.connectedComponentsWithStats(
            band_mask.astype(np.uint8), connectivity=8
        )
        for local_label in range(1, count):
            area = int(stats[local_label, cv2.CC_STAT_AREA])
            labels[local_labels == local_label] = next_label
            components.append((next_label, area))
            next_label += 1
    return labels, components


def _build_foreground_component_owner_locks(
    *,
    reference_rasters: Sequence[_ReferencePanelRaster],
    depth_mesh_candidates: Sequence[
        tuple[_DepthMeshPanelRemap, np.ndarray, np.ndarray, int]
    ],
    layout: InspectionMultiviewLayout,
    reference_depth_mm: float,
    config: InspectionMultiviewConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Lock each reliable foreground component to one complete real panel.

    Locks are produced before the adjacent seam chain is solved.  The seam
    solver must move the appropriate left/right boundaries around the locked
    component, so the final owner remains one monotone panel interval rather
    than a post-composition RGB island.
    """

    if len(reference_rasters) != len(depth_mesh_candidates):
        raise RuntimeError(
            "Inspection foreground lock sources are not aligned"
        )
    shape = (layout.height, layout.width)
    nearest_depth = np.full(shape, np.inf, dtype=np.float32)
    evidence = np.zeros(shape, dtype=bool)
    for mesh, _, _, _ in depth_mesh_candidates:
        x0 = int(mesh.corner_x)
        x1 = x0 + mesh.valid_mask.shape[1]
        local_depth = nearest_depth[:, x0:x1]
        take = mesh.valid_mask & (
            ~np.isfinite(local_depth)
            | (mesh.relative_depth_mm < local_depth)
        )
        local_depth[take] = mesh.relative_depth_mm[take]
        evidence[:, x0:x1] |= mesh.valid_mask
    labels, components = _foreground_depth_layer_components(
        evidence,
        nearest_depth,
        reference_depth_mm,
        config,
    )
    locked = np.full(shape, -1, dtype=np.int16)
    locked_component_labels = np.zeros(shape, dtype=np.int32)
    panel_centers = np.asarray(
        [
            float(item.corner_x)
            + 0.5 * float(item.valid_mask.shape[1] - 1)
            for item in reference_rasters
        ],
        dtype=np.float64,
    )
    nominal_boundaries = 0.5 * (
        panel_centers[:-1] + panel_centers[1:]
    )
    audits: list[dict[str, object]] = []
    unassigned_pixels = 0
    component_close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)
    )
    for label, area in components:
        if area < config.minimum_foreground_component_pixels:
            continue
        raw_component = labels == int(label)
        component = (
            cv2.morphologyEx(
                raw_component.astype(np.uint8),
                cv2.MORPH_CLOSE,
                component_close_kernel,
            )
            > 0
        )
        component &= locked_component_labels == 0
        area = int(np.count_nonzero(component))
        if area < config.minimum_foreground_component_pixels:
            continue
        component_x = np.flatnonzero(np.any(component, axis=0))
        if not component_x.size:
            continue
        center_x = float(np.median(component_x))
        nominal_panel = int(np.count_nonzero(
            center_x > nominal_boundaries
        ))
        eligible: list[tuple[int, int, float, int]] = []
        candidates: list[dict[str, object]] = []
        for source_index, (
            raster,
            (mesh, _, _, frame_id),
        ) in enumerate(
            zip(reference_rasters, depth_mesh_candidates, strict=True)
        ):
            if int(raster.frame_id) != int(frame_id):
                raise RuntimeError(
                    "Inspection foreground lock frame IDs are misaligned"
                )
            raster_x0 = int(raster.corner_x)
            raster_x1 = raster_x0 + raster.valid_mask.shape[1]
            local_component = component[:, raster_x0:raster_x1]
            inside = int(np.count_nonzero(local_component))
            reference_coverage = int(np.count_nonzero(
                local_component & raster.valid_mask
            ))
            complete = inside == area and reference_coverage == area
            mesh_x0 = int(mesh.corner_x)
            mesh_x1 = mesh_x0 + mesh.valid_mask.shape[1]
            mesh_coverage = int(np.count_nonzero(
                component[:, mesh_x0:mesh_x1] & mesh.valid_mask
            ))
            candidates.append(
                {
                    "panel_index": int(raster.panel_index),
                    "frame_id": int(frame_id),
                    "complete_reference_coverage": bool(complete),
                    "reference_coverage_pixels": reference_coverage,
                    "depth_mesh_coverage_pixels": mesh_coverage,
                }
            )
            if complete and mesh_coverage > 0:
                eligible.append(
                    (
                        mesh_coverage,
                        -abs(int(raster.panel_index) - nominal_panel),
                        -int(raster.panel_index),
                        source_index,
                    )
                )
        if not eligible:
            unassigned_pixels += area
            audits.append(
                {
                    "component_id": int(label),
                    "area_pixels": area,
                    "center_x": center_x,
                    "nominal_panel_index": nominal_panel,
                    "selected_panel_index": None,
                    "selected_frame_id": None,
                    "locked_before_seam": False,
                    "candidates": candidates,
                }
            )
            continue
        selected_index = max(eligible)[-1]
        selected = reference_rasters[selected_index]
        locked[component] = np.int16(selected.panel_index)
        locked_component_labels[component] = np.int32(label)
        audits.append(
            {
                "component_id": int(label),
                "area_pixels": area,
                "raw_component_area_pixels": int(
                    np.count_nonzero(raw_component)
                ),
                "center_x": center_x,
                "nominal_panel_index": nominal_panel,
                "selected_panel_index": int(selected.panel_index),
                "selected_frame_id": int(selected.frame_id),
                "locked_before_seam": True,
                "candidates": candidates,
            }
        )

    topology_conflicts = 0
    conflict_rows: list[int] = []
    for y in range(locked.shape[0]):
        sequence = locked[y, locked[y] >= 0]
        if sequence.size and np.any(np.diff(sequence) < 0):
            topology_conflicts += int(np.count_nonzero(np.diff(sequence) < 0))
            conflict_rows.append(y)
    return locked, locked_component_labels, {
        "policy": (
            "reliable_near_depth_layer_component_locked_to_one_complete_"
            "real_panel_before_adjacent_seam_solve"
        ),
        "component_count": len(audits),
        "locked_component_count": sum(
            item["locked_before_seam"] for item in audits
        ),
        "unassigned_component_count": sum(
            not item["locked_before_seam"] for item in audits
        ),
        "locked_pixel_count": int(np.count_nonzero(locked >= 0)),
        "unassigned_pixel_count": int(unassigned_pixels),
        "lock_backward_transition_count": topology_conflicts,
        "lock_conflict_example_rows": conflict_rows[:32],
        "all_components_locked": unassigned_pixels == 0,
        "lock_rows_monotonic": topology_conflicts == 0,
        "components": audits,
    }


def _owner_topology_audit(
    owner: np.ndarray,
    valid: np.ndarray,
    ordered_frame_ids: Sequence[int],
) -> dict[str, object]:
    """Audit final left-to-right owner order on every valid output row."""

    owner_array = np.asarray(owner, dtype=np.int32)
    valid_array = np.asarray(valid, dtype=bool)
    if owner_array.shape != valid_array.shape:
        raise RuntimeError("Inspection owner topology rasters are misaligned")
    frame_order = [int(value) for value in ordered_frame_ids]
    if len(frame_order) != len(set(frame_order)):
        raise RuntimeError("Inspection owner topology frame order is not unique")
    rank = np.full(owner_array.shape, -1, dtype=np.int32)
    for index, frame_id in enumerate(frame_order):
        rank[owner_array == frame_id] = int(index)
    unknown = valid_array & (rank < 0)
    backward_transitions = 0
    repeated_islands = 0
    transition_count = 0
    backward_rows: list[int] = []
    repeated_rows: list[int] = []
    for y in range(owner_array.shape[0]):
        sequence = rank[y, valid_array[y]]
        if sequence.size < 2:
            continue
        runs = sequence[
            np.r_[True, sequence[1:] != sequence[:-1]]
        ]
        transition_count += max(0, int(runs.size) - 1)
        backward = int(np.count_nonzero(np.diff(runs) < 0))
        backward_transitions += backward
        if backward:
            backward_rows.append(y)
        if runs.size:
            _, counts = np.unique(runs[runs >= 0], return_counts=True)
            repeated = int(np.sum(np.maximum(counts - 1, 0)))
        else:
            repeated = 0
        repeated_islands += repeated
        if repeated:
            repeated_rows.append(y)
    passed = (
        not np.any(unknown)
        and backward_transitions == 0
        and repeated_islands == 0
    )
    return {
        "policy": (
            "final_cropped_owner_rows_follow_virtual_panel_order_once"
        ),
        "ordered_frame_ids": frame_order,
        "valid_pixel_count": int(np.count_nonzero(valid_array)),
        "unknown_owner_pixel_count": int(np.count_nonzero(unknown)),
        "owner_transition_count": transition_count,
        "backward_transition_count": backward_transitions,
        "repeated_owner_island_count": repeated_islands,
        "rows_with_backward_transition_count": len(backward_rows),
        "rows_with_repeated_owner_island_count": len(repeated_rows),
        "backward_transition_example_rows": backward_rows[:32],
        "repeated_owner_island_example_rows": repeated_rows[:32],
        "all_rows_monotonic": backward_transitions == 0,
        "no_repeated_owner_islands": repeated_islands == 0,
        "audit_complete": True,
        "pass": bool(passed),
    }


def _composite_locked_foreground_mesh_rgb(
    *,
    locked_panel_index: np.ndarray,
    locked_component_labels: np.ndarray,
    reference_rasters: Sequence[_ReferencePanelRaster],
    depth_mesh_candidates: Sequence[
        tuple[_DepthMeshPanelRemap, np.ndarray, np.ndarray, int]
    ],
    compensated_source_images: Sequence[np.ndarray],
    output_image: np.ndarray,
    output_depth: np.ndarray | None = None,
    output_confidence: np.ndarray,
    output_owner: np.ndarray,
    output_reliable_depth: np.ndarray | None = None,
    output_foreground_overlay_mask: np.ndarray | None = None,
    config: InspectionMultiviewConfig,
) -> dict[str, object]:
    """Inverse-sample one real RGB source over every pre-seam locked component."""

    if not (
        len(reference_rasters)
        == len(depth_mesh_candidates)
        == len(compensated_source_images)
    ):
        raise RuntimeError(
            "Inspection locked foreground RGB sources are not aligned"
        )
    if (output_depth is None) != (output_reliable_depth is None):
        raise RuntimeError(
            "Inspection foreground depth/reliability outputs must be paired"
        )
    if (
        output_foreground_overlay_mask is not None
        and output_foreground_overlay_mask.shape != output_owner.shape
    ):
        raise RuntimeError(
            "Inspection foreground overlay mask is not canvas-aligned"
        )
    audits: list[dict[str, object]] = []
    maximum_fill_distance = float(
        max(1, 8 * int(config.depth_mesh_cell_size_pixels))
    )
    for component_label in np.unique(locked_component_labels):
        if int(component_label) <= 0:
            continue
        component = locked_component_labels == int(component_label)
        panel_values = np.unique(locked_panel_index[component])
        if panel_values.size != 1 or int(panel_values[0]) < 0:
            raise RuntimeError(
                "Inspection locked foreground component has no unique panel"
            )
        panel_index = int(panel_values[0])
        raster = reference_rasters[panel_index]
        mesh, _, source_confidence, frame_id = depth_mesh_candidates[
            panel_index
        ]
        if (
            int(raster.panel_index) != panel_index
            or int(raster.frame_id) != int(frame_id)
        ):
            raise RuntimeError(
                "Inspection locked foreground panel sources are misaligned"
            )
        x0 = int(mesh.corner_x)
        x1 = x0 + mesh.valid_mask.shape[1]
        local_component = component[:, x0:x1]
        area = int(np.count_nonzero(component))
        if int(np.count_nonzero(local_component)) != area:
            raise RuntimeError(
                "Inspection locked foreground component escapes its mesh panel"
            )
        seed = local_component & mesh.valid_mask
        seed_count = int(np.count_nonzero(seed))
        if seed_count == 0:
            raise RuntimeError(
                "Inspection foreground component has no inverse mesh support"
            )
        if (
            raster.reference_map_x is not None
            and raster.reference_map_y is not None
        ):
            source_mask = np.zeros(
                compensated_source_images[panel_index].shape[:2],
                dtype=np.uint8,
            )
            source_x = np.rint(mesh.map_x[seed]).astype(np.int32)
            source_y = np.rint(mesh.map_y[seed]).astype(np.int32)
            source_valid = (
                (source_x >= 0)
                & (source_x < source_mask.shape[1])
                & (source_y >= 0)
                & (source_y < source_mask.shape[0])
            )
            source_mask[source_y[source_valid], source_x[source_valid]] = 255
            close_size = 2 * int(config.depth_mesh_cell_size_pixels) + 1
            source_mask = cv2.morphologyEx(
                source_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (close_size, close_size)
                ),
            )
            source_mask = cv2.dilate(
                source_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            contours, _ = cv2.findContours(
                source_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            source_mask[:] = 0
            retained_contours = [
                contour
                for contour in contours
                if cv2.contourArea(contour) >= 4.0
            ]
            if retained_contours:
                cv2.drawContours(
                    source_mask,
                    retained_contours,
                    -1,
                    255,
                    thickness=cv2.FILLED,
                )
            source_points_y, source_points_x = np.nonzero(source_mask)
            if source_points_x.size >= 3:
                hull_points = np.column_stack(
                    (source_points_x, source_points_y)
                ).astype(np.int32)
                hull = cv2.convexHull(hull_points)
                hull_area = float(cv2.contourArea(hull))
                mask_area = float(np.count_nonzero(source_mask))
                # A bounded convex envelope restores the complete natural
                # silhouette and a small same-frame context margin.  Highly
                # concave/elongated masks retain their filled contours so a
                # cable loop cannot claim a large unrelated background area.
                if hull_area <= 3.0 * max(1.0, mask_area):
                    cv2.drawContours(
                        source_mask,
                        [hull],
                        -1,
                        255,
                        thickness=cv2.FILLED,
                    )
            reference_component = (
                accelerated_remap(
                    source_mask,
                    raster.reference_map_x,
                    raster.reference_map_y,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            ) & raster.valid_mask
            reference_area = int(np.count_nonzero(reference_component))
            if reference_area == 0:
                raise RuntimeError(
                    "Inspection foreground source component has no "
                    "reference-panel projection"
                )
            area_ratio = reference_area / float(max(1, area))
            if not 0.20 <= area_ratio <= 5.0:
                raise RuntimeError(
                    "Inspection foreground reference-panel component area "
                    f"ratio is unsafe: {area_ratio}"
                )
            raster_x0 = int(raster.corner_x)
            raster_x1 = raster_x0 + raster.valid_mask.shape[1]
            image_region = output_image[:, raster_x0:raster_x1]
            confidence_region = output_confidence[:, raster_x0:raster_x1]
            owner_region = output_owner[:, raster_x0:raster_x1]
            image_region[reference_component] = raster.image_bgr[
                reference_component
            ]
            confidence_region[reference_component] = raster.confidence[
                reference_component
            ]
            owner_region[reference_component] = int(frame_id)
            if output_reliable_depth is not None:
                output_reliable_depth[
                    :, raster_x0:raster_x1
                ][reference_component] = False
            if output_foreground_overlay_mask is not None:
                output_foreground_overlay_mask[
                    :, raster_x0:raster_x1
                ][reference_component] = True
            audits.append(
                {
                    "component_id": int(component_label),
                    "panel_index": panel_index,
                    "frame_id": int(frame_id),
                    "area_pixels": area,
                    "direct_inverse_mesh_pixel_count": seed_count,
                    "same_layer_map_fill_pixel_count": 0,
                    "maximum_map_fill_distance_pixels": 0.0,
                    "maximum_allowed_map_fill_distance_pixels": (
                        maximum_fill_distance
                    ),
                    "reference_panel_overlay_pixel_count": reference_area,
                    "reference_panel_area_ratio": area_ratio,
                    "reference_plane_rgb_fallback_pixel_count": 0,
                    "rgb_sampling_model": (
                        "depth_component_selected_single_source_mask_"
                        "projected_into_its_natural_reference_panel"
                    ),
                    "rgb_generated": False,
                    "owner_modified_after_seam": True,
                }
            )
            continue
        ys, xs = np.nonzero(local_component)
        y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
        local_x0, local_x1 = int(np.min(xs)), int(np.max(xs)) + 1
        component_roi = local_component[y0:y1, local_x0:local_x1]
        seed_roi = seed[y0:y1, local_x0:local_x1]
        distance_input = np.ones(component_roi.shape, dtype=np.uint8)
        distance_input[seed_roi] = 0
        distances, nearest_labels = cv2.distanceTransformWithLabels(
            distance_input,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
        label_to_map_x: dict[int, float] = {}
        label_to_map_y: dict[int, float] = {}
        label_to_depth: dict[int, float] = {}
        seed_y, seed_x = np.nonzero(seed_roi)
        for sy, sx in zip(seed_y, seed_x, strict=True):
            nearest_label = int(nearest_labels[sy, sx])
            label_to_map_x[nearest_label] = float(
                mesh.map_x[y0 + sy, local_x0 + sx]
            )
            label_to_map_y[nearest_label] = float(
                mesh.map_y[y0 + sy, local_x0 + sx]
            )
            label_to_depth[nearest_label] = float(
                mesh.relative_depth_mm[y0 + sy, local_x0 + sx]
            )
        fill = component_roi & ~seed_roi
        fill_distance = distances[fill]
        max_observed_fill = float(
            np.max(fill_distance, initial=0.0)
        )
        if max_observed_fill > maximum_fill_distance:
            raise RuntimeError(
                "Inspection foreground inverse mesh hole exceeds bounded "
                f"same-layer fill distance: {max_observed_fill}"
            )
        map_x = np.full(component_roi.shape, -1.0, dtype=np.float32)
        map_y = np.full(component_roi.shape, -1.0, dtype=np.float32)
        component_depth = np.full(
            component_roi.shape, np.inf, dtype=np.float32
        )
        map_x[seed_roi] = mesh.map_x[
            y0:y1, local_x0:local_x1
        ][seed_roi]
        map_y[seed_roi] = mesh.map_y[
            y0:y1, local_x0:local_x1
        ][seed_roi]
        component_depth[seed_roi] = mesh.relative_depth_mm[
            y0:y1, local_x0:local_x1
        ][seed_roi]
        fill_y, fill_x = np.nonzero(fill)
        for fy, fx in zip(fill_y, fill_x, strict=True):
            nearest_label = int(nearest_labels[fy, fx])
            map_x[fy, fx] = np.float32(
                label_to_map_x[nearest_label]
            )
            map_y[fy, fx] = np.float32(
                label_to_map_y[nearest_label]
            )
            component_depth[fy, fx] = np.float32(
                label_to_depth[nearest_label]
            )
        # Nearest-seed filling is only an initialization.  Smooth filled map
        # coordinates inside the same component while keeping every measured
        # inverse-mesh seed exact.  This removes cell-shaped RGB plateaus
        # without blending colours or crossing a depth-layer boundary.
        smoothing_size = 2 * int(config.depth_mesh_cell_size_pixels) + 1
        component_weight = component_roi.astype(np.float32)
        smoothed_weight = cv2.GaussianBlur(
            component_weight,
            (smoothing_size, smoothing_size),
            sigmaX=max(1.0, smoothing_size / 4.0),
            sigmaY=max(1.0, smoothing_size / 4.0),
            borderType=cv2.BORDER_REPLICATE,
        )
        for field in (map_x, map_y, component_depth):
            smoothed_value = cv2.GaussianBlur(
                np.where(component_roi, field, 0.0).astype(np.float32),
                (smoothing_size, smoothing_size),
                sigmaX=max(1.0, smoothing_size / 4.0),
                sigmaY=max(1.0, smoothing_size / 4.0),
                borderType=cv2.BORDER_REPLICATE,
            )
            interpolated = smoothed_value / np.maximum(
                smoothed_weight, np.float32(1e-6)
            )
            field[fill] = interpolated[fill]
        if (
            np.any(~np.isfinite(map_x[component_roi]))
            or np.any(~np.isfinite(map_y[component_roi]))
            or np.any(~np.isfinite(component_depth[component_roi]))
        ):
            raise RuntimeError(
                "Inspection foreground inverse mesh fill is incomplete"
            )
        sampled_image = accelerated_remap(
            compensated_source_images[panel_index],
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        sampled_confidence = accelerated_remap(
            source_confidence.astype(np.float32, copy=False),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        output_roi = output_image[
            y0:y1, x0 + local_x0:x0 + local_x1
        ]
        confidence_roi = output_confidence[
            y0:y1, x0 + local_x0:x0 + local_x1
        ]
        owner_roi = output_owner[
            y0:y1, x0 + local_x0:x0 + local_x1
        ]
        output_roi[component_roi] = sampled_image[component_roi]
        confidence_roi[component_roi] = sampled_confidence[component_roi]
        owner_roi[component_roi] = int(frame_id)
        if output_depth is not None and output_reliable_depth is not None:
            depth_roi = output_depth[
                y0:y1, x0 + local_x0:x0 + local_x1
            ]
            reliable_roi = output_reliable_depth[
                y0:y1, x0 + local_x0:x0 + local_x1
            ]
            depth_roi[component_roi] = component_depth[component_roi]
            reliable_roi[component_roi] = True
        audits.append(
            {
                "component_id": int(component_label),
                "panel_index": panel_index,
                "frame_id": int(frame_id),
                "area_pixels": area,
                "direct_inverse_mesh_pixel_count": seed_count,
                "same_layer_map_fill_pixel_count": int(
                    np.count_nonzero(fill)
                ),
                "maximum_map_fill_distance_pixels": max_observed_fill,
                "maximum_allowed_map_fill_distance_pixels": (
                    maximum_fill_distance
                ),
                "reference_plane_rgb_fallback_pixel_count": 0,
                "rgb_generated": False,
                "owner_modified_after_seam": True,
            }
        )
    return {
        "policy": (
            "depth_component_selects_one_real_source_then_natural_"
            "reference_panel_rgb_owner_overlay"
        ),
        "component_count": len(audits),
        "inverse_sampled_pixel_count": int(
            sum(item["area_pixels"] for item in audits)
        ),
        "same_layer_map_fill_pixel_count": int(
            sum(item["same_layer_map_fill_pixel_count"] for item in audits)
        ),
        "reference_plane_rgb_fallback_pixel_count": 0,
        "all_components_inverse_sampled": True,
        "components": audits,
    }


def _component_owner_preserves_row_topology(
    component: np.ndarray,
    owner: np.ndarray,
    ordered_frame_ids: Sequence[int],
    candidate_frame_id: int,
) -> bool:
    """Return whether replacing a whole component preserves affected rows."""

    frame_order = [int(value) for value in ordered_frame_ids]
    rank_by_frame = {
        frame_id: index for index, frame_id in enumerate(frame_order)
    }
    candidate_rank = rank_by_frame.get(int(candidate_frame_id))
    if candidate_rank is None:
        return False
    affected_rows = np.flatnonzero(np.any(component, axis=1))
    for y in affected_rows:
        row_owner = owner[y]
        row_valid = row_owner >= 0
        row_rank = np.full(row_owner.shape, -1, dtype=np.int32)
        for frame_id, frame_rank in rank_by_frame.items():
            row_rank[row_owner == frame_id] = int(frame_rank)
        row_rank[component[y]] = int(candidate_rank)
        sequence = row_rank[row_valid]
        if np.any(sequence < 0) or np.any(np.diff(sequence) < 0):
            return False
    return True


def _enforce_foreground_components_single_owner(
    *,
    output_image: np.ndarray,
    output_depth: np.ndarray,
    output_confidence: np.ndarray,
    output_owner: np.ndarray,
    output_reliable_depth: np.ndarray,
    reference_rasters: Sequence[_ReferencePanelRaster],
    depth_mesh_candidates: Sequence[
        tuple[_DepthMeshPanelRemap, np.ndarray, np.ndarray, int]
    ],
    reference_depth_mm: float,
    config: InspectionMultiviewConfig,
) -> dict[str, object]:
    """Replace each multi-owner foreground component from one real source.

    The component is discovered from the already rendered reliable near
    surface/depth-edge mask, not from the union of every candidate mesh.  A
    replacement is allowed only when one source's actual reference RGB raster
    covers the complete component.  The selected source contributes its
    continuous full-FOV panel RGB so sparse depth cannot pixelate the object's
    appearance.  Its accepted mesh supplies depth/visibility evidence, never
    replacement colour.
    """

    if len(reference_rasters) != len(depth_mesh_candidates):
        raise RuntimeError(
            "Inspection foreground component sources are not aligned"
        )
    ordered_rasters = sorted(
        reference_rasters, key=lambda item: int(item.panel_index)
    )
    ordered_frame_ids = [int(item.frame_id) for item in ordered_rasters]
    if len(ordered_frame_ids) != len(set(ordered_frame_ids)):
        raise RuntimeError(
            "Inspection foreground component frame order is not unique"
        )
    valid = output_owner >= 0
    labels, components = _foreground_depth_layer_components(
        valid & output_reliable_depth,
        output_depth,
        reference_depth_mm,
        config,
    )
    audits: list[dict[str, object]] = []
    replaced_pixels = 0
    unassigned_pixels = 0
    for label, area in components:
        if area < config.minimum_foreground_component_pixels:
            continue
        component = labels == label
        existing_owners = np.unique(output_owner[component])
        existing_owners = existing_owners[existing_owners >= 0]
        if existing_owners.size == 1:
            audits.append(
                {
                    "component_id": int(label),
                    "area_pixels": area,
                    "owners_before": [int(existing_owners[0])],
                    "selected_frame_id": int(existing_owners[0]),
                    "replacement_applied": False,
                    "complete_real_rgb_coverage": True,
                }
            )
            continue

        eligible: list[tuple[int, int, float, int, int, int]] = []
        candidate_audits: list[dict[str, object]] = []
        for source_index, (
            raster,
            (mesh, _, _, frame_id),
        ) in enumerate(
            zip(reference_rasters, depth_mesh_candidates, strict=True)
        ):
            if int(raster.frame_id) != int(frame_id):
                raise RuntimeError(
                    "Inspection foreground raster/mesh frame IDs are misaligned"
                )
            raster_x0 = int(raster.corner_x)
            raster_x1 = raster_x0 + raster.valid_mask.shape[1]
            local_component = component[:, raster_x0:raster_x1]
            inside = int(np.count_nonzero(local_component))
            reference_coverage = int(
                np.count_nonzero(local_component & raster.valid_mask)
            )
            complete = inside == area and reference_coverage == area
            mesh_x0 = int(mesh.corner_x)
            mesh_x1 = mesh_x0 + mesh.valid_mask.shape[1]
            mesh_component = component[:, mesh_x0:mesh_x1]
            mesh_coverage = int(
                np.count_nonzero(mesh_component & mesh.valid_mask)
            )
            existing_coverage = int(
                np.count_nonzero(component & (output_owner == int(frame_id)))
            )
            mean_confidence = (
                float(np.mean(raster.confidence[local_component]))
                if complete
                else 0.0
            )
            topology_preserved = (
                complete
                and _component_owner_preserves_row_topology(
                    component,
                    output_owner,
                    ordered_frame_ids,
                    int(frame_id),
                )
            )
            candidate_audits.append(
                {
                    "frame_id": int(frame_id),
                    "reference_coverage_pixels": reference_coverage,
                    "depth_mesh_coverage_pixels": mesh_coverage,
                    "existing_owner_pixels": existing_coverage,
                    "complete_reference_coverage": bool(complete),
                    "row_owner_topology_preserved": bool(
                        topology_preserved
                    ),
                }
            )
            if complete and topology_preserved:
                eligible.append(
                    (
                        mesh_coverage,
                        existing_coverage,
                        mean_confidence,
                        -int(raster.panel_index),
                        -int(frame_id),
                        source_index,
                    )
                )
        if not eligible:
            unassigned_pixels += area
            audits.append(
                {
                    "component_id": int(label),
                    "area_pixels": area,
                    "owners_before": [
                        int(value) for value in existing_owners
                    ],
                    "selected_frame_id": None,
                    "replacement_applied": False,
                    "complete_real_rgb_coverage": False,
                    "row_owner_topology_preserved": False,
                    "candidates": candidate_audits,
                }
            )
            continue

        selected_index = max(eligible)[-1]
        raster = reference_rasters[selected_index]
        mesh, _, _, frame_id = (
            depth_mesh_candidates[selected_index]
        )
        raster_x0 = int(raster.corner_x)
        raster_x1 = raster_x0 + raster.valid_mask.shape[1]
        local_component = component[:, raster_x0:raster_x1]
        output_image[:, raster_x0:raster_x1][local_component] = (
            raster.image_bgr[local_component]
        )
        output_confidence[:, raster_x0:raster_x1][local_component] = (
            raster.confidence[local_component]
        )
        output_owner[:, raster_x0:raster_x1][local_component] = int(frame_id)
        mesh_x0 = int(mesh.corner_x)
        mesh_x1 = mesh_x0 + mesh.valid_mask.shape[1]
        mesh_component = component[:, mesh_x0:mesh_x1]
        mesh_take = mesh_component & mesh.valid_mask
        replaced_pixels += area
        audits.append(
            {
                "component_id": int(label),
                "area_pixels": area,
                "owners_before": [int(value) for value in existing_owners],
                "selected_frame_id": int(frame_id),
                "replacement_applied": True,
                "complete_real_rgb_coverage": True,
                "row_owner_topology_preserved": True,
                "selected_depth_mesh_coverage_pixels": int(
                    np.count_nonzero(mesh_take)
                ),
                "candidates": candidate_audits,
            }
        )
    return {
        "policy": (
            "rendered_reliable_foreground_component_one_fully_covering_"
            "real_rgb_source"
        ),
        "component_count": len(audits),
        "multi_owner_component_count_before": sum(
            len(item["owners_before"]) > 1 for item in audits
        ),
        "replaced_component_count": sum(
            bool(item["replacement_applied"]) for item in audits
        ),
        "unassigned_component_count": sum(
            item["selected_frame_id"] is None for item in audits
        ),
        "topology_rejected_component_count": sum(
            item["selected_frame_id"] is None
            and any(
                bool(candidate["complete_reference_coverage"])
                and not bool(candidate["row_owner_topology_preserved"])
                for candidate in item.get("candidates", [])
            )
            for item in audits
        ),
        "replaced_pixel_count": int(replaced_pixels),
        "unassigned_pixel_count": int(unassigned_pixels),
        "all_components_assigned": unassigned_pixels == 0,
        "components": audits,
    }


def _foreground_owner_audit(
    valid: np.ndarray,
    depth: np.ndarray,
    owner: np.ndarray,
    reference_depth_mm: float,
    config: InspectionMultiviewConfig,
) -> dict[str, object]:
    labels, components = _foreground_depth_layer_components(
        valid, depth, reference_depth_mm, config
    )
    audits: list[dict[str, object]] = []
    single_owner_pixels = 0
    audited_pixels = 0
    for label, area in components:
        if area < config.minimum_foreground_component_pixels:
            continue
        component = labels == label
        owners, owner_counts = np.unique(owner[component], return_counts=True)
        owners = owners[owners >= 0]
        owner_counts = owner_counts[-owners.size :] if owners.size else owner_counts[:0]
        is_single = owners.size == 1
        audited_pixels += area
        if is_single:
            single_owner_pixels += area
        audits.append(
            {
                "component_id": int(label),
                "area_pixels": area,
                "owner_frame_ids": [int(value) for value in owners],
                "owner_count": int(owners.size),
                "single_owner": bool(is_single),
                "dominant_owner_fraction": (
                    float(np.max(owner_counts) / area) if owner_counts.size else 0.0
                ),
            }
        )
    single_components = sum(bool(item["single_owner"]) for item in audits)
    return {
        "policy": (
            "connected_reliable_near_depth_layer_component_single_real_frame_owner"
        ),
        "component_count": len(audits),
        "single_owner_component_count": single_components,
        "multi_owner_component_count": len(audits) - single_components,
        "single_owner_component_ratio": (
            float(single_components / len(audits)) if audits else 1.0
        ),
        "audited_foreground_pixel_count": audited_pixels,
        "single_owner_foreground_pixel_ratio": (
            float(single_owner_pixels / audited_pixels)
            if audited_pixels
            else 1.0
        ),
        "foreground_blend_pixel_count": 0,
        "audit_complete": True,
        "all_components_single_owner": single_components == len(audits),
        "components": audits,
    }


def _crop_valid(
    image: np.ndarray,
    depth: np.ndarray,
    confidence: np.ndarray,
    owner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    valid = owner >= 0
    if not np.any(valid):
        raise RuntimeError("Inspection rendering produced no valid crop")
    crop = largest_valid_rectangle(valid)
    x0, y0 = int(crop.x), int(crop.y)
    x1, y1 = x0 + int(crop.width), y0 + int(crop.height)
    return (
        np.ascontiguousarray(image[y0:y1, x0:x1]),
        np.ascontiguousarray(depth[y0:y1, x0:x1]),
        np.ascontiguousarray(confidence[y0:y1, x0:x1]),
        np.ascontiguousarray(owner[y0:y1, x0:x1]),
        (x0, y0, x1 - x0, y1 - y0),
    )


def _replace_object_reference_footprints(
    *,
    image: np.ndarray,
    owner: np.ndarray,
    valid: np.ndarray,
    rasters: Sequence[_ReferencePanelRaster],
    exclusions: Sequence[np.ndarray],
    output_footprint_mask: np.ndarray,
    preserve_owner_mask: np.ndarray | None = None,
) -> dict[str, object]:
    """Remove reference-plane copies of anchored objects using another view."""

    if len(rasters) != len(exclusions):
        raise RuntimeError("Object exclusions are not aligned with panels")
    if output_footprint_mask.shape != valid.shape:
        raise RuntimeError("Object footprint audit mask is misaligned")
    preserve = (
        np.zeros(valid.shape, dtype=bool)
        if preserve_owner_mask is None
        else np.asarray(preserve_owner_mask, dtype=bool)
    )
    if preserve.shape != valid.shape:
        raise RuntimeError("Object owner preservation mask is misaligned")
    bad = np.zeros(valid.shape, dtype=bool)
    for raster, exclusion in zip(rasters, exclusions, strict=True):
        if exclusion.shape != raster.valid_mask.shape:
            raise RuntimeError("Object exclusion shape does not match panel")
        x0 = int(raster.corner_x)
        x1 = x0 + exclusion.shape[1]
        bad[:, x0:x1] |= (
            (owner[:, x0:x1] == int(raster.frame_id))
            & exclusion
            & valid[:, x0:x1]
        )
    # A pre-seam object/corridor already has a complete, audited real RGB
    # owner.  It is not a stale reference-plane copy and must never be
    # reselected by this generic anchored-object cleanup.
    bad &= ~preserve
    replacement_image = np.zeros_like(image)
    replacement_owner = np.full(owner.shape, -1, dtype=np.int32)
    replacement_score = np.full(owner.shape, -np.inf, dtype=np.float32)
    for raster, exclusion in zip(rasters, exclusions, strict=True):
        x0 = int(raster.corner_x)
        x1 = x0 + exclusion.shape[1]
        available = (
            raster.valid_mask
            & ~exclusion
            & bad[:, x0:x1]
            & (owner[:, x0:x1] != int(raster.frame_id))
        )
        score = raster.confidence
        take = available & (score > replacement_score[:, x0:x1])
        replacement_image[:, x0:x1][take] = raster.image_bgr[take]
        replacement_owner[:, x0:x1][take] = int(raster.frame_id)
        replacement_score[:, x0:x1][take] = score[take]
    resolved = bad & (replacement_owner >= 0)
    unresolved = bad & ~resolved
    output_footprint_mask[:] = bad
    image[resolved] = replacement_image[resolved]
    owner[resolved] = replacement_owner[resolved]
    if np.any(unresolved):
        image[unresolved] = 0
        owner[unresolved] = -1
        valid[unresolved] = False
    return {
        "policy": (
            "remove_reference_plane_object_copy_with_nonexcluded_real_"
            "adjacent_panel_rgb"
        ),
        "footprint_pixel_count": int(np.count_nonzero(bad)),
        "replacement_pixel_count": int(np.count_nonzero(resolved)),
        "unresolved_pixel_count": int(np.count_nonzero(unresolved)),
        "preserved_single_owner_pixel_count": int(np.count_nonzero(preserve)),
        "all_footprints_replaced": not np.any(unresolved),
    }


def render_inspection_multiview(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    color_gains_rgb: Sequence[Sequence[float]] | None = None,
    pre_seam_hard_owner_intervals: Sequence[
        InspectionPreSeamHardOwnerInterval
    ] = (),
    foreground_identity_owners: Sequence[
        InspectionForegroundIdentityOwner
    ] = (),
    config: InspectionMultiviewConfig | Mapping[str, object] | None = None,
) -> InspectionMultiviewResult:
    """Render a non-fixed-strip inspection image from original RGB-D."""

    selected = (
        config
        if isinstance(config, InspectionMultiviewConfig)
        else InspectionMultiviewConfig.from_mapping(config)
    )
    selected.validate()
    checked = _validate_inputs(frames, poses)
    layout = estimate_inspection_layout(
        frames, checked, intrinsics, config=selected
    )
    resource_estimate = estimate_inspection_working_set(
        layout,
        intrinsics,
        config=selected,
    )
    gains: np.ndarray | None
    if color_gains_rgb is None:
        gains = None
    else:
        gains = np.asarray(color_gains_rgb, dtype=np.float32)
        if gains.shape != (len(frames), 3):
            raise ValueError(
                "Inspection color gains must align with all real source frames"
            )
    output_image = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    output_depth = np.full(
        (layout.height, layout.width), np.inf, dtype=np.float32
    )
    output_confidence = np.zeros(
        (layout.height, layout.width), dtype=np.float32
    )
    output_owner = np.full(
        (layout.height, layout.width), -1, dtype=np.int32
    )
    output_reliable_depth = np.zeros(
        (layout.height, layout.width), dtype=bool
    )
    maps = _undistortion_maps(intrinsics)
    source_audits: list[dict[str, object]] = []
    reference_rasters: list[_ReferencePanelRaster] = []
    foreground_anchor_sources: list[ForegroundAnchorSource] = []
    depth_mesh_candidates: list[
        tuple[_DepthMeshPanelRemap, np.ndarray, np.ndarray, int]
    ] = []
    panel_sources = _select_panel_sources(checked, layout)
    panel_source_by_frame = {
        int(frames[source_position].frame_id): (
            int(panel_index),
            int(source_position),
        )
        for panel_index, source_position in panel_sources
    }
    identity_structures: set[tuple[int, int]] = set()
    identity_sources_by_frame: dict[int, InspectionIdentityMeshSource] = {}
    identity_panel_indices = {
        int(item.panel_index) for item in foreground_identity_owners
    } | {
        (
            int(item.panel_index)
            if item.target_panel_index is None
            else int(item.target_panel_index)
        )
        for item in foreground_identity_owners
    }
    identity_reference_map_panel_indices = set(identity_panel_indices)
    for item in foreground_identity_owners:
        identity_reference_map_panel_indices.update(
            int(panel_index)
            for panel_index, _ in item.reference_observation_masks
        )
    identity_frame_ids = {
        int(frames[source_position].frame_id)
        for panel_index, source_position in panel_sources
        if int(panel_index) in identity_panel_indices
    }
    for identity_owner in foreground_identity_owners:
        key = (
            int(identity_owner.group_id),
            int(identity_owner.structure_id),
        )
        if key in identity_structures:
            raise ValueError(
                "Inspection foreground identity structures must be unique"
            )
        identity_structures.add(key)
        expected = panel_source_by_frame.get(int(identity_owner.frame_id))
        if expected is None or expected != (
            int(identity_owner.panel_index),
            int(identity_owner.source_index),
        ):
            raise RuntimeError(
                "Inspection foreground identity owner does not map to one "
                "selected real reference panel"
            )
        source_mask = np.asarray(identity_owner.source_mask)
        target_footprint = np.asarray(identity_owner.target_footprint)
        if (
            source_mask.dtype != np.bool_
            or source_mask.shape != (intrinsics.height, intrinsics.width)
            or target_footprint.dtype != np.bool_
            or target_footprint.shape != (layout.height, layout.width)
            or not np.any(source_mask)
            or not np.any(target_footprint)
        ):
            raise ValueError(
                "Inspection foreground identity masks are empty or "
                "not source/canvas-aligned bool"
            )
    for panel_index, source_position in panel_sources:
        frame = frames[source_position]
        pose = checked[source_position]
        image, depth, geometric_valid = _read_rgbd(frame, intrinsics, maps)
        image = _apply_gain(
            image, None if gains is None else gains[source_position]
        )
        reliable_depth = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= selected.minimum_depth_mm)
            & (depth <= selected.maximum_depth_mm)
        )
        if int(frame.frame_id) in identity_frame_ids:
            identity_sources_by_frame[int(frame.frame_id)] = (
                InspectionIdentityMeshSource(
                    panel_index=int(panel_index),
                    frame_id=int(frame.frame_id),
                    image_bgr=np.ascontiguousarray(image.copy()),
                    depth_mm=np.ascontiguousarray(depth.copy()),
                    reliable_depth=np.ascontiguousarray(
                        reliable_depth.copy()
                    ),
                    camera_to_world=np.asarray(
                        pose, dtype=np.float64
                    ).copy(),
                )
            )
        confidence, edge = _depth_confidence(
            depth, reliable_depth, selected
        )
        # Depth is advisory outside its reliable domain.  A full-FOV
        # reference-plane fallback keeps transparent/invalid-depth RGB owned
        # by one real frame instead of punching black holes.  Any reliable
        # observation always supersedes this low-confidence display fallback,
        # even when it is geometrically farther.
        foreground_margin = max(
            selected.foreground_depth_margin_mm,
            selected.foreground_depth_margin_ratio
            * layout.reference_depth_mm,
        )
        geometry_depth_limit = min(
            layout.reference_depth_mm - foreground_margin,
            layout.reference_depth_mm
            * selected.foreground_reference_depth_ratio,
        )
        geometry_depth = reliable_depth & (depth < geometry_depth_limit)
        guard_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                selected.foreground_guard_radius_pixels * 2 + 1,
                selected.foreground_guard_radius_pixels * 2 + 1,
            ),
        )
        foreground_guard = (
            cv2.dilate(geometry_depth.astype(np.uint8), guard_kernel) > 0
        )
        fallback = geometric_valid & ~geometry_depth
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        normalised_radius = np.sqrt(
            ((xx - intrinsics.cx) / max(1.0, intrinsics.width * 0.5)) ** 2
            + ((yy - intrinsics.cy) / max(1.0, intrinsics.height * 0.5)) ** 2
        )
        view_centrality = np.clip(1.0 - normalised_radius, 0.0, 1.0)
        confidence[geometry_depth] *= (
            0.35 + 0.65 * view_centrality[geometry_depth]
        )
        confidence[fallback] = 0.02 + 0.03 * view_centrality[fallback]
        (
            reference_panel_pixel_count,
            reference_raster,
        ) = _composite_reference_panel(
            output_image=output_image,
            output_depth=output_depth,
            output_confidence=output_confidence,
            output_owner=output_owner,
            output_reliable_depth=output_reliable_depth,
            source_image=image,
            source_protected_mask=(
                foreground_guard | ~reliable_depth
            ),
            source_pose=pose,
            frame_id=int(frame.frame_id),
            panel_index=panel_index,
            layout=layout,
            intrinsics=intrinsics,
            retain_reference_maps=bool(
                selected.foreground_world_anchor_enabled
                or int(panel_index)
                in identity_reference_map_panel_indices
            ),
        )
        reference_rasters.append(reference_raster)
        if selected.foreground_world_anchor_enabled:
            if (
                reference_raster.reference_map_x is None
                or reference_raster.reference_map_y is None
            ):
                raise RuntimeError(
                    "Inspection reference panel lacks inverse maps for "
                    "object anchoring"
                )
            foreground_anchor_sources.append(
                ForegroundAnchorSource(
                    source_index=len(foreground_anchor_sources),
                    panel_index=int(panel_index),
                    frame_id=int(frame.frame_id),
                    image_bgr=np.ascontiguousarray(image),
                    depth_mm=np.ascontiguousarray(depth),
                    reliable_depth=np.ascontiguousarray(reliable_depth),
                    camera_to_world=np.asarray(pose, dtype=np.float64),
                    reference_map_x=np.ascontiguousarray(
                        reference_raster.reference_map_x
                    ),
                    reference_map_y=np.ascontiguousarray(
                        reference_raster.reference_map_y
                    ),
                )
            )
        # Only locally supported, non-boundary near geometry is allowed to
        # replace the complete hard-owner panel.  Continuous depth cells are
        # inverse-rasterized to a target->source remap; no rounded point
        # splats, RGB generation, or pose modification is involved.
        projection_valid = (
            geometry_depth & ~edge & (confidence >= np.float32(0.50))
        )
        depth_mesh = _build_depth_mesh_panel_remap(
            source_depth_mm=depth,
            source_solver_valid=projection_valid,
            source_pose=pose,
            panel_index=panel_index,
            layout=layout,
            intrinsics=intrinsics,
            config=selected,
        )
        depth_mesh_candidates.append(
            (
                depth_mesh,
                (
                    np.ascontiguousarray(image)
                    if selected.foreground_world_anchor_enabled
                    else reference_raster.image_bgr
                ),
                np.ascontiguousarray(confidence),
                int(frame.frame_id),
            )
        )
        selected_count = 0
        same_layer_count = 0
        projected_count = int(
            depth_mesh.audit["valid_target_pixel_count"]
        )
        foreground_rejected_count = int(
            depth_mesh.audit["rejected_nonlocal_panel_cell_count"]
        )
        source_audits.append(
            {
                "frame_id": int(frame.frame_id),
                "source_position": int(source_position),
                "virtual_panel_index": int(panel_index),
                "input_valid_depth_pixel_count": int(
                    np.count_nonzero(reliable_depth)
                ),
                "reference_plane_fallback_pixel_count": int(
                    np.count_nonzero(fallback)
                ),
                "reference_panel_selected_pixel_count": int(
                    reference_panel_pixel_count
                ),
                "reliable_foreground_geometry_pixel_count": int(
                    np.count_nonzero(geometry_depth)
                ),
                "solver_valid_foreground_geometry_pixel_count": int(
                    np.count_nonzero(projection_valid)
                ),
                "foreground_guard_pixel_count": int(
                    np.count_nonzero(foreground_guard)
                ),
                "foreground_guard_other_panel_rejected_sample_count": int(
                    foreground_rejected_count
                ),
                "projected_in_canvas_sample_count": projected_count,
                "global_surface_update_count": selected_count,
                "same_layer_collision_count": same_layer_count,
                "selected_depth_edge_pixel_count": 0,
                "depth_mesh": depth_mesh.audit,
                "foreground_sampling_model": (
                    "single_full_fov_panel_rgb_with_depth_mesh_visibility"
                ),
                "full_width_source_sampling": True,
                "central_twenty_percent_only": False,
            }
        )

    if set(identity_sources_by_frame) != identity_frame_ids:
        raise RuntimeError(
            "Inspection foreground identity source collection is incomplete"
        )

    retained_reference_map_panel_indices = [
        int(raster.panel_index)
        for raster in reference_rasters
        if (
            raster.reference_map_x is not None
            and raster.reference_map_y is not None
        )
    ]
    partial_reference_map_panel_indices = [
        int(raster.panel_index)
        for raster in reference_rasters
        if (
            (raster.reference_map_x is None)
            != (raster.reference_map_y is None)
        )
    ]
    if partial_reference_map_panel_indices:
        raise RuntimeError(
            "Inspection reference panel retained only one inverse map"
        )
    expected_retained_reference_map_panel_indices = (
        set(range(len(reference_rasters)))
        if selected.foreground_world_anchor_enabled
        else identity_reference_map_panel_indices
    )
    if set(retained_reference_map_panel_indices) != (
        expected_retained_reference_map_panel_indices
    ):
        raise RuntimeError(
            "Inspection retained reference inverse maps do not match their "
            "world-anchor/identity-owner consumers"
        )
    if selected.foreground_world_anchor_enabled:
        if len(retained_reference_map_panel_indices) != len(
            reference_rasters
        ):
            raise RuntimeError(
                "Inspection world-anchor source lacks retained inverse maps"
            )
    reference_inverse_map_audit = {
        "policy": (
            "retain_only_for_enabled_world_anchor_or_identity_exclusion_"
            "consumer"
        ),
        "foreground_world_anchor_enabled": bool(
            selected.foreground_world_anchor_enabled
        ),
        "reference_panel_count": len(reference_rasters),
        "retained_panel_count": len(
            retained_reference_map_panel_indices
        ),
        "retained_panel_indices": retained_reference_map_panel_indices,
        "identity_owner_retained_panel_indices": sorted(
            identity_reference_map_panel_indices
        ),
        "retained_bytes": int(
            sum(
                np.asarray(raster.reference_map_x).nbytes
                + np.asarray(raster.reference_map_y).nbytes
                for raster in reference_rasters
                if (
                    raster.reference_map_x is not None
                    and raster.reference_map_y is not None
                )
            )
        ),
        "lazy_recomputed_panel_count": 0,
        "unused_map_retention_count": 0,
        "depth_mesh_source_image_policy": (
            "original_rgb_retained_for_enabled_world_anchor"
            if selected.foreground_world_anchor_enabled
            else (
                "original_rgb_retained_only_for_identity_owner_source"
                if foreground_identity_owners
                else (
                    "reference_panel_placeholder_no_rgb_read_in_"
                    "write_rgb_false_path"
                )
            )
        ),
    }

    background_anchor_source_count = len(reference_rasters)
    # Object tracking uses denser real viewpoints than the appearance
    # background.  The background keeps a small, stable panel chain; extra
    # RGB-D views only improve world-cluster support and complete-object
    # source selection, and never enter GraphCut or MultiBand.
    scan_axis = np.asarray(layout.scan_axis, dtype=np.float64)
    center_scan = np.asarray(
        [pose[:3, 3] @ scan_axis for pose in checked], dtype=np.float64
    )
    dense_targets = (
        np.linspace(
            float(np.min(center_scan)),
            float(np.max(center_scan)),
            min(len(frames), max(24, len(panel_sources))),
        )
        if selected.foreground_world_anchor_enabled
        else np.empty(0, dtype=np.float64)
    )
    dense_positions: list[int] = []
    used_dense: set[int] = set()
    for target in dense_targets:
        for candidate in np.argsort(
            np.abs(center_scan - target), kind="stable"
        ):
            position = int(candidate)
            if position not in used_dense:
                used_dense.add(position)
                dense_positions.append(position)
                break
    background_positions = {
        int(source_position) for _, source_position in panel_sources
    }
    for source_position in dense_positions:
        if source_position in background_positions:
            continue
        frame = frames[source_position]
        pose = checked[source_position]
        image, depth, geometric_valid = _read_rgbd(
            frame, intrinsics, maps
        )
        image = _apply_gain(
            image, None if gains is None else gains[source_position]
        )
        reliable_depth = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= selected.minimum_depth_mm)
            & (depth <= selected.maximum_depth_mm)
        )
        source_scan = float(center_scan[source_position])
        panel_index = int(
            np.argmin(
                np.abs(
                    np.asarray(
                        [
                            panel.anchor_scan_mm
                            for panel in layout.panels
                        ],
                        dtype=np.float64,
                    )
                    - source_scan
                )
            )
        )
        _, anchor_raster = _composite_reference_panel(
            output_image=output_image,
            output_depth=output_depth,
            output_confidence=output_confidence,
            output_owner=output_owner,
            output_reliable_depth=output_reliable_depth,
            source_image=image,
            source_protected_mask=~reliable_depth,
            source_pose=pose,
            frame_id=int(frame.frame_id),
            panel_index=panel_index,
            layout=layout,
            intrinsics=intrinsics,
        )
        if (
            anchor_raster.reference_map_x is None
            or anchor_raster.reference_map_y is None
        ):
            raise RuntimeError(
                "Dense object-anchor source lacks inverse reference maps"
            )
        foreground_anchor_sources.append(
            ForegroundAnchorSource(
                source_index=len(foreground_anchor_sources),
                panel_index=panel_index,
                frame_id=int(frame.frame_id),
                image_bgr=np.ascontiguousarray(image),
                depth_mm=np.ascontiguousarray(depth),
                reliable_depth=np.ascontiguousarray(reliable_depth),
                camera_to_world=np.asarray(pose, dtype=np.float64),
                reference_map_x=np.ascontiguousarray(
                    anchor_raster.reference_map_x
                ),
                reference_map_y=np.ascontiguousarray(
                    anchor_raster.reference_map_y
                ),
            )
        )

    if selected.foreground_world_anchor_enabled:
        foreground_anchor_plan = plan_foreground_object_anchors(
            foreground_anchor_sources,
            layout,
            intrinsics,
            minimum_component_pixels=max(
                600, int(selected.minimum_foreground_component_pixels)
            ),
        )
    else:
        foreground_anchor_plan = ForegroundObjectAnchorPlan(
            anchors=(),
            observations=(),
            background_exclusion_masks=tuple(
                np.zeros(raster.valid_mask.shape, dtype=bool)
                for raster in reference_rasters
            ),
            target_mask=np.zeros(
                (int(layout.height), int(layout.width)), dtype=bool
            ),
            audit={
                "policy": (
                    "rgbd_world_component_track_one_complete_rgb_owner_"
                    "constrained_similarity"
                ),
                "enabled": False,
                "reason": (
                    "formal_default_disabled_until_cross_scene_object_"
                    "identity_validation"
                ),
                "near_depth_margin_mm": max(
                    35.0, 0.04 * float(layout.reference_depth_mm)
                ),
                "canvas_boundary_margin_pixels": 0,
                "raw_sample_point_count": 0,
                "voxel_size_mm": 8.0,
                "voxel_count": 0,
                "removed_structural_plane_count": 0,
                "raw_world_cluster_count": 0,
                "rejected_world_cluster_count": 0,
                "world_cluster_rejection_reasons": {},
                "rejected_world_clusters": [],
                "observation_count": 0,
                "track_count": 0,
                "background_exclusion_pixel_counts": [
                    0 for _ in reference_rasters
                ],
                "tracks": [],
            },
        )
    foreground_mask = foreground_anchor_plan.target_mask.copy()
    # Identity owners carry their own pre-seam owner-only guards and are
    # composited later from one true-depth RGB source.  They must not be
    # converted into background panel risks: doing so couples independent
    # objects to the monotone panel chain and can eliminate every feasible
    # background seam.
    for mesh, _, _, _ in depth_mesh_candidates:
        x0 = int(mesh.corner_x)
        x1 = x0 + mesh.valid_mask.shape[1]
        foreground_mask[:, x0:x1] |= mesh.valid_mask
    (
        locked_foreground_panel_index,
        _,
        foreground_lock_audit,
    ) = _build_foreground_component_owner_locks(
        reference_rasters=reference_rasters,
        depth_mesh_candidates=depth_mesh_candidates,
        layout=layout,
        reference_depth_mm=layout.reference_depth_mm,
        config=selected,
    )
    (
        background_image,
        background_owner,
        background_valid,
        background_seam_audit,
        compensated_reference_rasters,
        compensated_foreground_sources,
        background_owner_only_guard,
        background_spatial_owner_panel_index,
    ) = _compose_reference_panels_graphcut_multiband(
        reference_rasters,
        [item[1] for item in depth_mesh_candidates],
        layout,
        foreground_mask,
        locked_foreground_panel_index,
        foreground_lock_audit,
        pre_seam_hard_owner_intervals,
        selected,
    )
    pre_seam_value = background_seam_audit.get(
        "pre_seam_hard_owner_intervals"
    )
    if not isinstance(pre_seam_value, Mapping):
        raise RuntimeError("Inspection pre-seam interval audit is malformed")
    accepted_pre_seam_track_ids = {
        int(row["track_id"])
        for row in pre_seam_value.get("intervals", [])
        if isinstance(row, Mapping) and "track_id" in row
    }
    pre_seam_hard_owner_intervals = tuple(
        interval
        for interval in pre_seam_hard_owner_intervals
        if int(interval.track_id) in accepted_pre_seam_track_ids
    )
    object_reference_footprint_mask = np.zeros(
        background_valid.shape, dtype=bool
    )
    identity_background_exclusions = [
        np.zeros(raster.valid_mask.shape, dtype=bool)
        for raster in reference_rasters
    ]
    for identity_owner in foreground_identity_owners:
        observations = (
            identity_owner.reference_observation_masks
            if identity_owner.reference_observation_masks
            else (
                (
                    int(identity_owner.panel_index),
                    np.asarray(identity_owner.source_mask, dtype=bool),
                ),
            )
        )
        seen_observation_panels: set[int] = set()
        for panel_index_value, observation_mask in observations:
            panel_index = int(panel_index_value)
            if panel_index in seen_observation_panels:
                raise RuntimeError(
                    "Inspection identity owner repeats a reference observation "
                    "panel"
                )
            seen_observation_panels.add(panel_index)
            raster = reference_rasters[panel_index]
            if (
                raster.reference_map_x is None
                or raster.reference_map_y is None
            ):
                raise RuntimeError(
                    "Inspection identity owner lacks its reference-plane "
                    "exclusion map"
                )
            source_observation = np.asarray(observation_mask, dtype=bool)
            if source_observation.shape != (
                intrinsics.height,
                intrinsics.width,
            ):
                raise RuntimeError(
                    "Inspection identity reference observation is not "
                    "source-aligned"
                )
            projected_source_mask = (
                accelerated_remap(
                    source_observation.astype(np.uint8) * np.uint8(255),
                    raster.reference_map_x,
                    raster.reference_map_y,
                    cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                > 0
            )
            # One pixel protects nearest/linear sampling without turning a
            # non-convex instance or an internal hole into a filled blob.
            projected_source_mask = cv2.dilate(
                projected_source_mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            ).astype(bool)
            identity_background_exclusions[panel_index] |= (
                projected_source_mask & raster.valid_mask
            )
    combined_background_exclusions = [
        np.asarray(anchor_exclusion, dtype=bool)
        | identity_background_exclusions[index]
        for index, anchor_exclusion in enumerate(
            foreground_anchor_plan.background_exclusion_masks[
                :background_anchor_source_count
            ]
        )
    ]
    pre_seam_preserve_owner_mask = np.zeros(
        background_valid.shape, dtype=bool
    )
    for interval in pre_seam_hard_owner_intervals:
        if interval.deferred_true_depth_identity_overlay:
            continue
        transfer = (
            np.asarray(interval.lock_mask, dtype=bool)
            if interval.rgb_transfer_mask is None
            else np.asarray(interval.rgb_transfer_mask, dtype=bool)
        )
        pre_seam_preserve_owner_mask |= transfer
    object_footprint_audit = _replace_object_reference_footprints(
        image=background_image,
        owner=background_owner,
        valid=background_valid,
        rasters=compensated_reference_rasters,
        exclusions=combined_background_exclusions,
        output_footprint_mask=object_reference_footprint_mask,
        preserve_owner_mask=pre_seam_preserve_owner_mask,
    )
    object_footprint_audit["identity_owner_reference_exclusion_pixel_count"] = (
        int(
            sum(
                np.count_nonzero(value)
                for value in identity_background_exclusions
            )
        )
    )
    background_seam_audit["object_reference_footprint_replacement"] = (
        object_footprint_audit
    )
    output_image = background_image
    output_owner = background_owner
    output_source_uv = np.full(
        (*output_owner.shape, 2), np.nan, dtype=np.float32
    )
    # Rebuild the deterministic reference-panel inverse map only for the
    # selected real owner frame.  This records the actual target->source UV
    # used by the background/corridor RGB path without retaining twelve full
    # maps throughout GraphCut and MultiBand.
    for panel_index, source_position in panel_sources:
        frame_id = int(frames[source_position].frame_id)
        corner_x, map_x, map_y, map_valid, _ = _reference_panel_inverse_maps(
            source_pose=checked[source_position],
            panel_index=int(panel_index),
            layout=layout,
            intrinsics=intrinsics,
        )
        x1 = corner_x + map_x.shape[1]
        local_owner = output_owner[:, corner_x:x1] == frame_id
        take = local_owner & map_valid
        output_source_uv[:, corner_x:x1][take] = np.stack(
            (map_x[take], map_y[take]), axis=1
        )
    output_depth = np.where(
        background_valid,
        np.float32(layout.reference_depth_mm),
        np.float32(np.inf),
    )
    output_confidence = np.where(
        background_valid, np.float32(0.10), np.float32(0.0)
    )
    output_reliable_depth = np.zeros(background_valid.shape, dtype=bool)
    # The safe-background seam chooses the initial RGB owner.  Each depth mesh
    # first refines only that frame's owner region.  A final component pass
    # below resolves any object crossed by an adjacent background owner seam.
    for audit, (mesh, source_image, source_confidence, frame_id) in zip(
        source_audits, depth_mesh_candidates, strict=True
    ):
        selected_count, same_layer_count = _composite_depth_mesh_panel(
            mesh=mesh,
            source_image=source_image,
            source_confidence=source_confidence,
            output_image=output_image,
            output_depth=output_depth,
            output_confidence=output_confidence,
            output_owner=output_owner,
            output_reliable_depth=output_reliable_depth,
            frame_id=frame_id,
            config=selected,
            require_existing_owner=True,
            write_rgb=False,
        )
        audit["global_surface_update_count"] = selected_count
        audit["same_layer_collision_count"] = same_layer_count

    compensated_anchor_sources = list(foreground_anchor_sources)
    for index, image in enumerate(
        compensated_foreground_sources[
            : min(
                len(compensated_foreground_sources),
                len(compensated_anchor_sources),
            )
        ]
    ):
        compensated_anchor_sources[index] = replace(
            compensated_anchor_sources[index],
            image_bgr=np.ascontiguousarray(image),
        )
    foreground_overlay = overlay_foreground_object_anchors(
        plan=foreground_anchor_plan,
        sources=compensated_anchor_sources,
        output_image=output_image,
        output_owner=output_owner,
        output_depth=output_depth,
        output_confidence=output_confidence,
    )
    identity_overlay_mask = np.zeros(output_owner.shape, dtype=bool)
    compensated_identity_sources = {
        frame_id: replace(
            source,
            image_bgr=np.ascontiguousarray(
                compensated_foreground_sources[int(source.panel_index)]
            ),
        )
        for frame_id, source in identity_sources_by_frame.items()
    }
    identity_overlay_audit = composite_inspection_identity_owners(
        owners=foreground_identity_owners,
        sources_by_frame_id=compensated_identity_sources,
        layout=layout,
        intrinsics=intrinsics,
        output_image=output_image,
        output_depth=output_depth,
        output_confidence=output_confidence,
        output_owner=output_owner,
        output_reliable_depth=output_reliable_depth,
        output_overlay_mask=identity_overlay_mask,
        output_source_uv=output_source_uv,
        config=InspectionIdentityMeshConfig(
            cell_size_pixels=int(
                selected.identity_mesh_cell_size_pixels
            ),
            maximum_fill_distance_pixels=float(
                selected.identity_mesh_maximum_fill_distance_pixels
            ),
            minimum_depth_mm=float(selected.minimum_depth_mm),
            maximum_depth_mm=float(selected.maximum_depth_mm),
            minimum_jacobian=float(selected.depth_mesh_min_jacobian),
            maximum_jacobian=float(selected.depth_mesh_max_jacobian),
        ),
    )
    deferred_identity_lock_rows: list[dict[str, object]] = []
    owner_by_track_id = {
        int(owner.identity_track_id): owner
        for owner in foreground_identity_owners
        if owner.identity_track_id is not None
    }
    for interval in pre_seam_hard_owner_intervals:
        if not interval.deferred_true_depth_identity_overlay:
            continue
        owner_value = owner_by_track_id.get(int(interval.track_id))
        if owner_value is None:
            raise RuntimeError(
                "Deferred inspection identity lock lacks its mesh owner"
            )
        footprint = np.asarray(owner_value.target_footprint, dtype=bool)
        rgb_owner_mismatch = int(
            np.count_nonzero(
                footprint & (output_owner != int(interval.frame_id))
            )
        )
        overlay_missing = int(
            np.count_nonzero(footprint & ~identity_overlay_mask)
        )
        if rgb_owner_mismatch or overlay_missing:
            raise RuntimeError(
                "Deferred inspection identity owner did not preserve its "
                "complete true-depth RGB owner"
            )
        deferred_identity_lock_rows.append(
            {
                "track_id": int(interval.track_id),
                "spatial_panel_index": int(interval.panel_index),
                "rgb_owner_frame_id": int(interval.frame_id),
                "spatial_lock_mismatch_pixel_count": 0,
                "background_spatial_panel_lock_required": False,
                "background_panel_owner_decoupled_from_rgb_owner": True,
                "rgb_owner_mismatch_pixel_count": 0,
                "overlay_missing_pixel_count": 0,
            }
        )
    identity_overlay_audit["deferred_spatial_lock_closure"] = {
        "component_count": len(deferred_identity_lock_rows),
        "components": deferred_identity_lock_rows,
        "pass": True,
    }
    foreground_overlay_mask = (
        foreground_overlay.visible_mask | identity_overlay_mask
    )
    output_reliable_depth[:] = False
    output_reliable_depth[foreground_overlay_mask] = True
    output_depth[
        (output_owner >= 0) & ~output_reliable_depth
    ] = np.float32(layout.reference_depth_mm)

    foreground_component_assignment = {
        **foreground_overlay.audit,
        "assignment_stage": (
            "rgbd_world_track_before_seam_then_depth_ordered_hard_"
            "object_overlay_after_monotone_background_chain"
        ),
        "post_composition_foreground_overlay_component_count": int(
            len(foreground_anchor_plan.anchors)
            + int(identity_overlay_audit["component_count"])
        ),
        "all_components_assigned": bool(
            foreground_overlay.audit["all_tracks_visible"]
            and identity_overlay_audit[
                "all_components_single_real_owner"
            ]
        ),
        "rgb_photometric_domain": (
            "same_opencv_channels_and_adjacent_residual_compensated_"
            "raw_source_rgb_as_background"
        ),
        "object_world_anchor": foreground_overlay.audit,
        "identity_owner_inverse_mesh": identity_overlay_audit,
        "category_counts": {
            "reliable_mesh_component_count": int(
                len(foreground_anchor_plan.anchors)
                + int(identity_overlay_audit["component_count"])
            ),
            "reliable_mesh_inverse_sampled_pixel_count": int(
                np.count_nonzero(foreground_overlay_mask)
            ),
            "invalid_depth_owner_only_component_count": int(
                background_seam_audit[
                    "invalid_depth_owner_only_locks"
                ]["component_count"]
            ),
            "invalid_depth_owner_only_locked_pixel_count": int(
                background_seam_audit[
                    "invalid_depth_owner_only_locks"
                ]["locked_pixel_count"]
            ),
        },
    }

    full_valid_before_crop = output_owner >= 0
    full_foreground = _foreground_component_mask(
        full_valid_before_crop & output_reliable_depth,
        output_depth,
        layout.reference_depth_mm,
        selected,
    ) | foreground_overlay_mask
    photometric_owner_only_guard = (
        background_owner_only_guard | full_foreground
    )
    photometric_background = (
        full_valid_before_crop & ~photometric_owner_only_guard
    )
    output_image, canvas_exposure_audit = (
        _apply_continuous_canvas_exposure_curve(
            output_image,
            photometric_background,
            full_valid_before_crop,
        )
    )
    background_seam_audit["continuous_canvas_exposure"] = {
        **canvas_exposure_audit,
        "application_order": (
            "after_all_foreground_component_owner_replacements"
        ),
        "applied_uniformly_to_background_and_foreground": True,
        "foreground_rgb_preserved_from_selected_real_owner": True,
        "owner_only_guard_pixel_count": int(
            np.count_nonzero(photometric_owner_only_guard)
        ),
        "corrected_owner_only_guard_intersection_pixel_count": int(
            np.count_nonzero(
                full_valid_before_crop & photometric_owner_only_guard
            )
        ),
    }

    full_extent_valid = output_owner >= 0
    full_rows, full_columns = np.nonzero(full_extent_valid)
    if not full_columns.size:
        raise RuntimeError(
            "Inspection rendering produced no full-extent RGB support"
        )
    full_x0 = int(np.min(full_columns))
    full_x1 = int(np.max(full_columns)) + 1
    full_y0 = int(np.min(full_rows))
    full_y1 = int(np.max(full_rows)) + 1
    full_extent_valid_crop = np.ascontiguousarray(
        full_extent_valid[full_y0:full_y1, full_x0:full_x1]
    )
    full_extent_bgr = np.ascontiguousarray(
        output_image[full_y0:full_y1, full_x0:full_x1].copy()
    )
    full_extent_bgr[~full_extent_valid_crop] = 0
    full_extent_bgra = np.dstack(
        (
            full_extent_bgr,
            full_extent_valid_crop.astype(np.uint8) * np.uint8(255),
        )
    )
    full_extent_owner = np.ascontiguousarray(
        output_owner[full_y0:full_y1, full_x0:full_x1]
    )

    image, depth, confidence, owner, crop = _crop_valid(
        output_image, output_depth, output_confidence, output_owner
    )
    source_uv_crop = np.ascontiguousarray(
        output_source_uv[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
    )
    provenance_known = np.isfinite(source_uv_crop).all(axis=2)
    provenance_unknown = int(np.count_nonzero((owner >= 0) & ~provenance_known))
    provenance_out_of_bounds = int(
        np.count_nonzero(
            (owner >= 0)
            & provenance_known
            & (
                (source_uv_crop[..., 0] < 0.0)
                | (source_uv_crop[..., 0] > float(intrinsics.width - 1))
                | (source_uv_crop[..., 1] < 0.0)
                | (source_uv_crop[..., 1] > float(intrinsics.height - 1))
            )
        )
    )
    pre_seam_crop_rows: list[dict[str, object]] = []
    for interval in pre_seam_hard_owner_intervals:
        lock = np.asarray(interval.lock_mask, dtype=bool)
        transfer = (
            lock
            if interval.rgb_transfer_mask is None
            else np.asarray(interval.rgb_transfer_mask, dtype=bool)
        )
        local_transfer = transfer[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
        transfer_before_count = int(np.count_nonzero(transfer))
        transfer_after_count = int(np.count_nonzero(local_transfer))
        # A deferred RGB-D owner uses ``lock`` only as a seam-protection
        # guard.  Its factual object support is the inverse-mesh transfer
        # footprint, so crop preservation must be measured against that
        # footprint rather than against padding which may legitimately sit
        # outside the largest-valid rectangle.  Ordinary panel/corridor
        # owners remain governed by their complete row-contiguous lock.
        preserve_mask = (
            transfer
            if interval.deferred_true_depth_identity_overlay
            else lock
        )
        local_preserve_mask = preserve_mask[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
        before_count = int(np.count_nonzero(preserve_mask))
        after_count = int(np.count_nonzero(local_preserve_mask))
        owner_mismatch = int(
            np.count_nonzero(
                local_transfer & (owner != int(interval.frame_id))
            )
        )
        # A pre-seam panel lock can later be superseded only by a verified
        # foreground inverse-mesh owner.  It is not a GraphCut/MultiBand
        # rewrite: the mesh owns the exact target footprint with one real RGB
        # source after its depth/visibility checks.
        local_foreground_overlay = foreground_overlay_mask[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
        owner_mismatch_mask = local_transfer & (owner != int(interval.frame_id))
        mesh_owner_override = int(
            np.count_nonzero(owner_mismatch_mask & local_foreground_overlay)
        )
        unexplained_owner_mismatch = int(
            np.count_nonzero(owner_mismatch_mask & ~local_foreground_overlay)
        )
        if (
            after_count != before_count
            or transfer_after_count != transfer_before_count
            or unexplained_owner_mismatch
        ):
            raise RuntimeError(
                "Inspection crop removed or changed a pre-seam hard-owner "
                "interval; "
                f"track_id={int(interval.track_id)}, frame_id={int(interval.frame_id)}, "
                f"deferred_true_depth={bool(interval.deferred_true_depth_identity_overlay)}, "
                f"preserve_before={before_count}, preserve_after={after_count}, "
                f"transfer_before={transfer_before_count}, "
                f"transfer_after={transfer_after_count}, owner_mismatch={owner_mismatch}, "
                f"mesh_owner_override={mesh_owner_override}, "
                f"unexplained_owner_mismatch={unexplained_owner_mismatch}, "
                f"crop={crop}"
            )
        pre_seam_crop_rows.append(
            {
                "track_id": int(interval.track_id),
                "frame_id": int(interval.frame_id),
                "crop_preservation_mask": (
                    "rgbd_target_footprint"
                    if interval.deferred_true_depth_identity_overlay
                    else "row_contiguous_owner_lock"
                ),
                "pre_crop_pixel_count": before_count,
                "post_crop_pixel_count": after_count,
                "rgb_transfer_pre_crop_pixel_count": (
                    transfer_before_count
                ),
                "rgb_transfer_post_crop_pixel_count": (
                    transfer_after_count
                ),
                "owner_mismatch_pixel_count": owner_mismatch,
                "mesh_owner_override_pixel_count": mesh_owner_override,
                "unexplained_owner_mismatch_pixel_count": (
                    unexplained_owner_mismatch
                ),
            }
        )
    pre_seam_audit_value = background_seam_audit[
        "pre_seam_hard_owner_intervals"
    ]
    if not isinstance(pre_seam_audit_value, dict):
        raise RuntimeError("Inspection pre-seam interval audit is malformed")
    pre_seam_audit_value["crop_preserved_all_locked_pixels"] = True
    pre_seam_audit_value["post_crop_owner_mismatch_pixel_count"] = int(
        sum(int(row["owner_mismatch_pixel_count"]) for row in pre_seam_crop_rows)
    )
    pre_seam_audit_value["post_crop_mesh_owner_override_pixel_count"] = int(
        sum(
            int(row["mesh_owner_override_pixel_count"])
            for row in pre_seam_crop_rows
        )
    )
    pre_seam_audit_value["post_crop_unexplained_owner_mismatch_pixel_count"] = int(
        sum(
            int(row["unexplained_owner_mismatch_pixel_count"])
            for row in pre_seam_crop_rows
        )
    )
    pre_seam_audit_value["crop_intervals"] = pre_seam_crop_rows
    reliable_depth_crop = np.ascontiguousarray(
        output_reliable_depth[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
    )
    foreground_overlay_crop = np.ascontiguousarray(
        foreground_overlay_mask[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
    )
    identity_overlay_crop = np.ascontiguousarray(
        identity_overlay_mask[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
    )
    owner_only_guard_crop = np.ascontiguousarray(
        background_owner_only_guard[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
    )
    spatial_owner_crop = np.ascontiguousarray(
        background_spatial_owner_panel_index[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
    )
    object_reference_footprint_crop = np.ascontiguousarray(
        object_reference_footprint_mask[
            crop[1] : crop[1] + crop[3],
            crop[0] : crop[0] + crop[2],
        ]
    )
    foreground_overlay_pre_crop_pixel_count = int(
        np.count_nonzero(foreground_overlay_mask)
    )
    foreground_overlay_post_crop_pixel_count = int(
        np.count_nonzero(foreground_overlay_crop)
    )
    valid = owner >= 0
    depth[~valid] = np.nan
    base_foreground_audit = _foreground_owner_audit(
        valid & reliable_depth_crop & ~identity_overlay_crop,
        depth,
        owner,
        layout.reference_depth_mm,
        selected,
    )
    identity_component_count = int(
        identity_overlay_audit["component_count"]
    )
    identity_pixel_count = int(np.count_nonzero(identity_overlay_crop))
    base_component_count = int(
        base_foreground_audit["component_count"]
    )
    base_single_component_count = int(
        base_foreground_audit["single_owner_component_count"]
    )
    base_audited_pixels = int(
        base_foreground_audit["audited_foreground_pixel_count"]
    )
    base_single_pixels = int(round(
        float(
            base_foreground_audit[
                "single_owner_foreground_pixel_ratio"
            ]
        )
        * base_audited_pixels
    ))
    foreground_audit = {
        "policy": (
            "ordinary_near_depth_components_plus_stable_identity_"
            "structures_each_require_one_real_rgb_owner"
        ),
        "component_count": (
            base_component_count + identity_component_count
        ),
        "single_owner_component_count": (
            base_single_component_count + identity_component_count
        ),
        "multi_owner_component_count": int(
            base_foreground_audit["multi_owner_component_count"]
        ),
        "single_owner_component_ratio": float(
            (
                base_single_component_count + identity_component_count
            )
            / max(1, base_component_count + identity_component_count)
        ),
        "audited_foreground_pixel_count": (
            base_audited_pixels + identity_pixel_count
        ),
        "single_owner_foreground_pixel_ratio": float(
            (base_single_pixels + identity_pixel_count)
            / max(1, base_audited_pixels + identity_pixel_count)
        ),
        "foreground_blend_pixel_count": 0,
        "audit_complete": True,
        "all_components_single_owner": bool(
            base_foreground_audit["all_components_single_owner"]
            and identity_overlay_audit[
                "all_components_single_real_owner"
            ]
        ),
        "ordinary_near_depth_components": base_foreground_audit,
        "stable_identity_structures": identity_overlay_audit,
    }
    owner_topology_audit = _owner_topology_audit(
        spatial_owner_crop,
        valid
        & ~foreground_overlay_crop
        & ~object_reference_footprint_crop,
        list(range(len(compensated_reference_rasters))),
    )
    owner_topology_audit["owner_domain"] = (
        "monotone_spatial_panel_index_separate_from_true_rgb_source_frame_id"
    )
    owner_topology_audit["rgb_source_owner_frame_id_audited_separately"] = True
    final_foreground = _foreground_component_mask(
        valid & reliable_depth_crop,
        depth,
        layout.reference_depth_mm,
        selected,
    ) | foreground_overlay_crop
    owner_boundary_audit = _background_owner_boundary_audit(
        image,
        owner,
        valid,
        final_foreground,
        selected,
        owner_only_guard_mask=owner_only_guard_crop,
    )
    background_seam_audit["owner_boundary_visual_audit"] = (
        owner_boundary_audit
    )
    world_coverage_audit = audit_inspection_world_coverage(
        frames=frames,
        poses=checked,
        intrinsics=intrinsics,
        layout=layout,
        owner_frame_id=owner,
        crop_xywh=crop,
        selected_panel_sources=[
            {
                "panel_index": int(panel_index),
                "source_position": int(source_position),
                "frame_id": int(frames[source_position].frame_id),
            }
            for panel_index, source_position in panel_sources
        ],
        config=InspectionWorldCoverageConfig(
            minimum_depth_mm=float(selected.minimum_depth_mm),
            maximum_depth_mm=float(selected.maximum_depth_mm),
        ),
    )
    minimum_multiview_world_coverage_ratio = 0.80
    world_coverage_audit["minimum_multiview_world_coverage_ratio"] = (
        minimum_multiview_world_coverage_ratio
    )
    world_coverage_audit["pass"] = bool(
        float(world_coverage_audit["multiview_world_coverage_ratio"])
        >= minimum_multiview_world_coverage_ratio
    )
    strict_incomplete_reasons: list[str] = []
    if pre_seam_hard_owner_intervals:
        strict_incomplete_reasons.append(
            "pre_seam_single_panel_hard_owner_interval_used"
        )
    pre_seam_audit = background_seam_audit[
        "pre_seam_hard_owner_intervals"
    ]
    if int(pre_seam_audit.get("rejected_interval_count", 0)) > 0:
        strict_incomplete_reasons.append(
            "pre_seam_object_lock_rejected_by_closed_seam_topology"
        )
    if foreground_identity_owners:
        strict_incomplete_reasons.append(
            "foreground_identity_single_owner_inverse_mesh_used"
        )
    if not foreground_component_assignment["all_components_assigned"]:
        strict_incomplete_reasons.append(
            "foreground_component_has_no_single_source_full_rgb_coverage"
        )
    if (
        foreground_overlay_post_crop_pixel_count
        != foreground_overlay_pre_crop_pixel_count
    ):
        strict_incomplete_reasons.append(
            "valid_crop_removed_rgbd_world_anchored_object_pixels"
        )
    if object_footprint_audit["unresolved_pixel_count"]:
        strict_incomplete_reasons.append(
            "object_reference_footprint_has_no_alternate_background_view"
        )
    if not foreground_audit["all_components_single_owner"]:
        strict_incomplete_reasons.append(
            "foreground_components_have_multiple_rgb_owners"
        )
    if not owner_topology_audit["pass"]:
        strict_incomplete_reasons.append(
            "background_owner_rows_are_nonmonotonic_or_repeat_panels"
        )
    if (
        background_seam_audit[
            "protected_blend_intersection_pixel_count"
        ]
        != 0
    ):
        strict_incomplete_reasons.append(
            "protected_pixels_intersect_background_multiband"
        )
    if not background_seam_audit["exposure_compensation_used"]:
        strict_incomplete_reasons.append(
            "background_exposure_compensation_not_applied"
        )
    if not owner_boundary_audit["pass"]:
        strict_incomplete_reasons.append(
            "background_owner_boundary_has_visible_color_discontinuity"
        )
    if (
        int(
            background_seam_audit["foreground_component_locks"][
                "rejected_chain_lock_component_count"
            ]
        )
        > 0
    ):
        strict_incomplete_reasons.append(
            "foreground_object_lock_rejected_by_closed_seam_topology"
        )
    if not world_coverage_audit["pass"]:
        strict_incomplete_reasons.append(
            "multiview_observed_near_world_surface_missing_from_final_rgb_owner"
        )
    noncentral_source_count = sum(
        int(item["projected_in_canvas_sample_count"]) > 0
        for item in source_audits
    )
    metadata: dict[str, object] = {
        "schema": "gemini305-inspection-multiview/v1",
        "method": (
            "trajectory_constrained_depth_aware_multi_viewpoint_"
            "side_scan_mosaicing"
        ),
        "backend": "overlapping_virtual_perspective_panels_rgbd",
        "fixed_strip_pushbroom": False,
        "ordinary_2d_panorama": False,
        "metric_raster_used_for_rgb": False,
        "tsdf_used_for_rgb": False,
        "real_pose_count": len(checked),
        "selected_full_fov_source_count": len(panel_sources),
        "selected_panel_sources": [
            {
                "panel_index": int(panel_index),
                "source_position": int(source_position),
                "frame_id": int(frames[source_position].frame_id),
            }
            for panel_index, source_position in panel_sources
        ],
        "pose_interpolation_count": 0,
        "layout": layout.as_dict(),
        "resource_estimate": resource_estimate.as_dict(),
        "reference_inverse_maps": reference_inverse_map_audit,
        "config": asdict(selected),
        "crop": {
            "x": crop[0],
            "y": crop[1],
            "width": crop[2],
            "height": crop[3],
        },
        "full_extent_browse_product": {
            "crop_x": full_x0,
            "crop_y": full_y0,
            "width": int(full_x1 - full_x0),
            "height": int(full_y1 - full_y0),
            "valid_pixel_count": int(
                np.count_nonzero(full_extent_valid_crop)
            ),
            "invalid_transparent_pixel_count": int(
                full_extent_valid_crop.size
                - np.count_nonzero(full_extent_valid_crop)
            ),
            "alpha_policy": (
                "255_real_single_rgb_owner_0_geometrically_uncovered"
            ),
            "rgb_fill_or_generation_used": False,
            "formal_quality_crop_replaced": False,
        },
        "source_audits": source_audits,
        "source_with_full_fov_contribution_count": noncentral_source_count,
        "rgb_policy": (
            "single_real_rgb_owner_per_pixel_nearest_local_depth_"
            "then_confidence"
        ),
        "background_graphcut_applied": True,
        "background_multiband_applied": True,
        "background_seam_audit": background_seam_audit,
        "foreground_component_assignment": (
            {
                **foreground_component_assignment,
                "pre_crop_visible_pixel_count": (
                    foreground_overlay_pre_crop_pixel_count
                ),
                "post_crop_visible_pixel_count": (
                    foreground_overlay_post_crop_pixel_count
                ),
                "crop_preserved_all_object_pixels": bool(
                    foreground_overlay_post_crop_pixel_count
                    == foreground_overlay_pre_crop_pixel_count
                ),
            }
        ),
        "foreground_owner_continuity_summary": foreground_audit,
        "owner_topology_audit": {
            **owner_topology_audit,
            "scope": (
                "background_pixels_excluding_rgbd_object_overlay_and_"
                "audited_reference_footprint_replacements"
            ),
            "foreground_overlay_pixel_count": int(
                np.count_nonzero(foreground_overlay_crop)
            ),
            "foreground_owner_islands_allowed": True,
        },
        "world_surface_coverage_audit": world_coverage_audit,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "invalid_pixel_count": int(valid.size - np.count_nonzero(valid)),
        "color_gain_applied": gains is not None,
        "color_gain_channel_order": "RGB",
        "reliable_depth_owner_pixel_count": int(
            np.count_nonzero(valid & reliable_depth_crop)
        ),
        "reference_plane_fallback_owner_pixel_count": int(
            np.count_nonzero(valid & ~reliable_depth_crop)
        ),
        "reference_plane_fallback_policy": (
            "invalid_or_unreliable_depth_rgb_keeps_one_real_owner_at_D0_"
            "reliable_depth_always_supersedes_no_cross_owner_average"
        ),
        "strict_v1_inspection_complete": not strict_incomplete_reasons,
        "strict_incomplete_reasons": strict_incomplete_reasons,
        "pixel_provenance_v1": {
            "sampling_model": "per_pixel_real_rgb_inverse_map",
            "unknown_source_pixel_count": provenance_unknown,
            "out_of_bounds_source_pixel_count": provenance_out_of_bounds,
            "border_replicated_pixel_count": 0,
            "synthetic_or_fake_coverage_pixel_count": 0,
        },
    }
    result = InspectionMultiviewResult(
        image_bgr=image,
        owner_frame_id=owner,
        valid_mask=valid,
        relative_depth_mm=depth.astype(np.float32, copy=False),
        full_extent_bgra=np.ascontiguousarray(full_extent_bgra),
        full_extent_owner_frame_id=full_extent_owner,
        metadata=metadata,
        source_uv=source_uv_crop,
    )
    result.validate()
    return result


__all__ = [
    "InspectionForegroundIdentityOwner",
    "InspectionMultiviewConfig",
    "InspectionMultiviewLayout",
    "InspectionPreSeamHardOwnerInterval",
    "InspectionResourceEstimate",
    "InspectionMultiviewResult",
    "VirtualPerspectivePanel",
    "estimate_inspection_working_set",
    "estimate_inspection_layout",
    "project_world_points_to_panels",
    "render_inspection_multiview",
]
