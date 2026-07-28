"""Bounded in-memory CUDA identity planning for formal inspection output.

FastSAM and DIS supply instance identity evidence.  RapidOCR may seed a
stable same-layer object group.  Every emitted owner is reconstructed from
aligned depth and the immutable camera-to-world trajectory by the existing
identity-owner planners.  This module never stores labels, changes a pose,
or carries RGB into a post-render overlay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import cv2
import numpy as np

from .dis_track_direct_handoff import DirectHandoffConfig
from .fastsam_dis_tracking import (
    FastSAMDISConfig,
    FastSAMExactMaskProposal,
    FastSAMDISFrameInput,
    FastSAMDISTrackingResult,
    track_fastsam_dis_frames,
)
from .inspection_fastsam_track import polygon_mask
from .fastsam_onnx import (
    FastSAMOnnxConfig,
    FastSAMOnnxRunner,
    summarize_onnxruntime_profile as summarize_fastsam_profile,
)
from .cuda_backend import remap as accelerated_remap
from .inspection_identity_mesh import (
    InspectionIdentityMeshConfig,
    InspectionIdentityMeshSource,
    composite_inspection_identity_owners,
)
from .inspection_identity_owner_planner import (
    InspectionIdentityOwnerFrame,
    _project_structure,
    plan_direct_stable_track_identity_owners,
    plan_inspection_identity_owner_intervals,
    plan_middle_shelf_inventory_identity_owners,
)
from .inspection_multiview import (
    InspectionForegroundIdentityOwner,
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    InspectionPreSeamHardOwnerInterval,
    _read_rgbd,
    _reference_panel_inverse_maps,
    _select_panel_sources,
    _undistortion_maps,
    estimate_inspection_layout,
)
from .inspection_ocr_panel import (
    extract_ocr_seeded_panel,
    sample_mask_world_points,
)
from .rapidocr_onnx_adapter import (
    RapidOCRModels,
    RapidOCROnnxAdapter,
    RapidOCRRuntime,
)
from .session import CameraIntrinsics, RGBDFrame


_FASTSAM_MODEL_ENVIRONMENT = "G305_FASTSAM_ONNX"
_RAPIDOCR_MODEL_DIRECTORY_ENVIRONMENT = "G305_RAPIDOCR_MODEL_DIR"


def _format_shelf_unsat_context(context: Mapping[str, object]) -> str:
    """Return stable compact JSON suitable for a fail-closed exception."""

    return json.dumps(
        dict(context),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class InspectionIdentityRuntimeConfig:
    """Closed formal switches and resource limits for identity planning."""

    enabled: bool = False
    fastsam_model_path: str | None = None
    rapidocr_model_directory: str | None = None
    rapidocr_enabled: bool = True
    direct_rgbd_owner_application_enabled: bool = False
    panel_native_preseam_lock_enabled: bool = False
    panel_native_minimum_source_coverage_ratio: float = 0.90
    panel_native_minimum_component_ratio: float = 0.98
    panel_native_lock_guard_pixels: int = 8
    object_rich_preseam_lock_enabled: bool = True
    object_rich_minimum_horizontal_overlap_ratio: float = 0.70
    object_rich_maximum_vertical_gap_pixels: int = 16
    object_rich_minimum_depth_coverage_ratio: float = 0.95
    object_rich_lock_guard_pixels: int = 8
    cuda_device_id: int = 0
    maximum_frame_count: int = 160
    maximum_proposals_per_frame: int = 80
    minimum_proposal_area_ratio: float = 0.001
    maximum_proposal_area_ratio: float = 0.30
    minimum_ocr_score: float = 0.50
    maximum_identity_owner_count: int = 64
    maximum_runtime_bytes: int = 1_500_000_000
    minimum_direct_mask_compactness: float = 0.65
    minimum_direct_bbox_aspect_ratio: float = 0.40
    maximum_direct_bbox_aspect_ratio: float = 2.00
    direct_source_boundary_margin_pixels: int = 8

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object] | None = None
    ) -> "InspectionIdentityRuntimeConfig":
        payload = {} if value is None else dict(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                f"Unknown identity_owner_runtime configuration keys: {unknown}"
            )
        try:
            selected = cls(**payload)
        except TypeError as exc:
            raise ValueError(
                "Invalid identity_owner_runtime configuration"
            ) from exc
        selected.validate()
        return selected

    def validate(self) -> None:
        if (
            type(self.enabled) is not bool
            or type(self.rapidocr_enabled) is not bool
            or type(self.direct_rgbd_owner_application_enabled) is not bool
            or type(self.panel_native_preseam_lock_enabled) is not bool
            or type(self.object_rich_preseam_lock_enabled) is not bool
        ):
            raise ValueError(
                "identity_owner_runtime enabled switches must be boolean"
            )
        if type(self.cuda_device_id) is not int or self.cuda_device_id < 0:
            raise ValueError(
                "identity_owner_runtime.cuda_device_id must be non-negative"
            )
        if not 2 <= int(self.maximum_frame_count) <= 160:
            raise ValueError(
                "identity_owner_runtime.maximum_frame_count must be in [2, 160]"
            )
        if not 1 <= int(self.maximum_proposals_per_frame) <= 300:
            raise ValueError(
                "identity_owner_runtime.maximum_proposals_per_frame must be "
                "in [1, 300]"
            )
        if not (
            math.isfinite(float(self.minimum_proposal_area_ratio))
            and math.isfinite(float(self.maximum_proposal_area_ratio))
            and 0.0
            < float(self.minimum_proposal_area_ratio)
            < float(self.maximum_proposal_area_ratio)
            <= 1.0
        ):
            raise ValueError(
                "identity_owner_runtime proposal area ratios are invalid"
            )
        if not (
            math.isfinite(float(self.minimum_ocr_score))
            and 0.0 <= float(self.minimum_ocr_score) <= 1.0
        ):
            raise ValueError(
                "identity_owner_runtime.minimum_ocr_score must be in [0, 1]"
            )
        if not 1 <= int(self.maximum_identity_owner_count) <= 128:
            raise ValueError(
                "identity_owner_runtime.maximum_identity_owner_count must be "
                "in [1, 128]"
            )
        if (
            type(self.maximum_runtime_bytes) is not int
            or self.maximum_runtime_bytes <= 0
        ):
            raise ValueError(
                "identity_owner_runtime.maximum_runtime_bytes must be positive"
            )
        if not (
            0.0 <= float(self.minimum_direct_mask_compactness) <= 1.0
            and 0.0
            < float(self.minimum_direct_bbox_aspect_ratio)
            <= float(self.maximum_direct_bbox_aspect_ratio)
        ):
            raise ValueError(
                "identity_owner_runtime direct compactness/aspect gates "
                "are invalid"
            )
        if not 0 <= int(self.direct_source_boundary_margin_pixels) <= 64:
            raise ValueError(
                "identity_owner_runtime direct source boundary margin is "
                "invalid"
            )
        if not (
            0.0
            < float(self.panel_native_minimum_source_coverage_ratio)
            <= 1.0
            and 0.0
            < float(self.panel_native_minimum_component_ratio)
            <= 1.0
        ):
            raise ValueError(
                "identity_owner_runtime panel-native coverage gates are "
                "invalid"
            )
        if not 0 <= int(self.panel_native_lock_guard_pixels) <= 32:
            raise ValueError(
                "identity_owner_runtime panel-native guard is invalid"
            )
        if not (
            0.0
            < float(self.object_rich_minimum_horizontal_overlap_ratio)
            <= 1.0
            and 0.0
            < float(self.object_rich_minimum_depth_coverage_ratio)
            <= 1.0
        ):
            raise ValueError(
                "identity_owner_runtime object-rich ratio gates are invalid"
            )
        if not 0 <= int(
            self.object_rich_maximum_vertical_gap_pixels
        ) <= 128:
            raise ValueError(
                "identity_owner_runtime object-rich vertical gap is invalid"
            )
        if not 0 <= int(self.object_rich_lock_guard_pixels) <= 32:
            raise ValueError(
                "identity_owner_runtime object-rich guard is invalid"
            )
        for name in ("fastsam_model_path", "rapidocr_model_directory"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"identity_owner_runtime.{name} must be a non-empty path"
                )


@dataclass(frozen=True)
class InspectionIdentityRuntimeResult:
    pre_seam_hard_owner_intervals: tuple[
        InspectionPreSeamHardOwnerInterval, ...
    ]
    foreground_owners: tuple[InspectionForegroundIdentityOwner, ...]
    audit: dict[str, object]


def _filter_compact_direct_owners(
    owners: Sequence[InspectionForegroundIdentityOwner],
    *,
    config: InspectionIdentityRuntimeConfig,
) -> tuple[
    tuple[InspectionForegroundIdentityOwner, ...],
    dict[str, object],
]:
    accepted: list[InspectionForegroundIdentityOwner] = []
    rows: list[dict[str, object]] = []
    for owner in owners:
        mask = np.asarray(owner.source_mask, dtype=bool)
        yy, xx = np.nonzero(mask)
        if xx.size == 0:
            raise ValueError("Direct identity owner source mask is empty")
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        width = x1 - x0
        height = y1 - y0
        aspect = float(width / max(1, height))
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        perimeter = float(
            sum(cv2.arcLength(contour, True) for contour in contours)
        )
        compactness = float(
            4.0
            * math.pi
            * int(np.count_nonzero(mask))
            / max(1.0, perimeter * perimeter)
        )
        margin = int(config.direct_source_boundary_margin_pixels)
        boundary_clear = bool(
            x0 >= margin
            and y0 >= margin
            and x1 <= mask.shape[1] - margin
            and y1 <= mask.shape[0] - margin
        )
        geometric_gate_passed = bool(
            compactness >= config.minimum_direct_mask_compactness
            and config.minimum_direct_bbox_aspect_ratio
            <= aspect
            <= config.maximum_direct_bbox_aspect_ratio
            and boundary_clear
        )
        passed = bool(
            geometric_gate_passed
            and config.direct_rgbd_owner_application_enabled
        )
        rows.append(
            {
                "identity_track_id": owner.identity_track_id,
                "frame_id": int(owner.frame_id),
                "source_bbox_xywh": [x0, y0, width, height],
                "source_mask_pixel_count": int(np.count_nonzero(mask)),
                "source_mask_compactness": compactness,
                "source_bbox_aspect_ratio": aspect,
                "source_boundary_clear": boundary_clear,
                "accepted": passed,
                "geometric_candidate_gate_passed": (
                    geometric_gate_passed
                ),
                "direct_rgbd_owner_application_enabled": bool(
                    config.direct_rgbd_owner_application_enabled
                ),
                "outcome": (
                    "direct_rgbd_owner_candidate"
                    if passed
                    else "hard_cut_degraded_not_applied"
                ),
                "reason": (
                    "compact_closed_nonstructural_instance_gate_passed"
                    if passed
                    else (
                        "direct_owner_application_disabled_until_"
                        "photometric_boundary_closure_is_validated"
                        if geometric_gate_passed
                        else "elongated_open_or_boundary_structural_mask"
                    )
                ),
            }
        )
        if passed:
            accepted.append(owner)
    return tuple(accepted), {
        "policy": (
            "reject_elongated_open_or_image_boundary_structural_masks_"
            "before_true_rgbd_owner_composition"
        ),
        "candidate_count": len(owners),
        "accepted_count": len(accepted),
        "rejected_count": len(owners) - len(accepted),
        "minimum_mask_compactness": float(
            config.minimum_direct_mask_compactness
        ),
        "bbox_aspect_ratio_range": [
            float(config.minimum_direct_bbox_aspect_ratio),
            float(config.maximum_direct_bbox_aspect_ratio),
        ],
        "source_boundary_margin_pixels": int(
            config.direct_source_boundary_margin_pixels
        ),
        "direct_rgbd_owner_application_enabled": bool(
            config.direct_rgbd_owner_application_enabled
        ),
        "tracks": rows,
    }


def _exclude_shelf_tracks_from_direct_preseam_candidates(
    owners: Sequence[InspectionForegroundIdentityOwner],
    eligible_object_rich_track_ids: set[int],
    *,
    shelf_exclusive_track_ids: set[int],
) -> tuple[
    tuple[InspectionForegroundIdentityOwner, ...],
    set[int],
    dict[str, object],
]:
    """Keep required shelf tracks on exactly one owner-planning path."""

    retained_owners = tuple(
        owner
        for owner in owners
        if owner.identity_track_id is None
        or int(owner.identity_track_id) not in shelf_exclusive_track_ids
    )
    excluded_owner_track_ids = sorted(
        {
            int(owner.identity_track_id)
            for owner in owners
            if owner.identity_track_id is not None
            and int(owner.identity_track_id) in shelf_exclusive_track_ids
        }
    )
    excluded_object_rich_track_ids = sorted(
        eligible_object_rich_track_ids & shelf_exclusive_track_ids
    )
    retained_eligible_track_ids = (
        set(eligible_object_rich_track_ids) - shelf_exclusive_track_ids
    )
    return retained_owners, retained_eligible_track_ids, {
        "policy": (
            "required_shelf_inventory_and_hierarchy_tracks_use_only_the_"
            "resolved_shelf_native_mesh_or_single_source_corridor_path"
        ),
        "shelf_exclusive_track_ids": sorted(shelf_exclusive_track_ids),
        "excluded_direct_panel_native_track_ids": (
            excluded_owner_track_ids
        ),
        "excluded_direct_object_rich_track_ids": (
            excluded_object_rich_track_ids
        ),
        "retained_unrelated_direct_owner_count": len(retained_owners),
        "retained_unrelated_object_rich_track_ids": sorted(
            retained_eligible_track_ids
        ),
        "ocr_owner_path_modified": False,
        "cross_layer_duplicate_owner_allowed": False,
    }


def _row_contiguous_guard(
    footprint: np.ndarray,
    *,
    guard_pixels: int,
) -> np.ndarray:
    """Return the minimum per-row horizontal owner interval around a mask."""

    source = np.asarray(footprint, dtype=bool)
    if source.ndim != 2 or not np.any(source):
        raise ValueError("Panel-native footprint must be a non-empty mask")
    if guard_pixels:
        size = 2 * int(guard_pixels) + 1
        source = cv2.dilate(
            source.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (size, size),
            ),
        ).astype(bool)
    result = np.zeros_like(source)
    for row in np.flatnonzero(np.any(source, axis=1)):
        columns = np.flatnonzero(source[row])
        result[row, int(columns[0]) : int(columns[-1]) + 1] = True
    return np.ascontiguousarray(result)


def _trim_fixed_corridor_boundary_overlap(
    corridor: np.ndarray,
    measured_support: np.ndarray,
    *,
    candidate_frame_id: int,
    fixed_intervals: Sequence[InspectionPreSeamHardOwnerInterval],
) -> tuple[np.ndarray, bool, list[dict[str, object]]]:
    """Give a fixed corridor only non-object pixels on row boundaries.

    A different-frame fixed corridor may touch the padding around a later
    object-rich corridor.  Removing that padding is safe when it cannot punch
    an interior hole in any row.  A tiny overlap between two *measured*
    instance masks is also resolvable only when every shared pixel is already
    covered by the fixed interval's exact real-RGB support and the duplicate
    label occupies at most 384 pixels, or at most 4096 pixels and 15% of the
    candidate's measured support.  The relative alternative makes the
    partition stable to sub-pixel pose/rasterization variation without
    admitting a large interior identity collision.  One image pixel cannot
    belong to two exclusive instance owners; this partitions that bounded
    label ambiguity without deleting, blending, or generating its RGB.
    That is an exclusive partition of ambiguous segmentation boundary pixels,
    not deletion or blending.  Interior or larger overlaps remain fail-closed.
    """

    result = np.ascontiguousarray(np.asarray(corridor, dtype=bool).copy())
    support = np.asarray(measured_support, dtype=bool)
    if result.shape != support.shape or not np.any(result):
        raise ValueError("Corridor and measured support must be non-empty peers")
    rows: list[dict[str, object]] = []
    for fixed in fixed_intervals:
        if int(fixed.frame_id) == int(candidate_frame_id):
            continue
        fixed_mask = np.asarray(fixed.rgb_transfer_mask, dtype=bool)
        if fixed_mask.shape != result.shape:
            raise ValueError("Fixed corridor mask shape mismatch")
        overlap = result & fixed_mask
        overlap_pixels = int(np.count_nonzero(overlap))
        if overlap_pixels == 0:
            continue
        measured_overlap_pixels = int(
            np.count_nonzero(overlap & support)
        )
        fixed_exact_support = np.asarray(
            fixed.union_footprint, dtype=bool
        )
        if fixed_exact_support.shape != result.shape:
            raise ValueError("Fixed exact-support mask shape mismatch")
        measured_overlap = overlap & support
        support_pixels = int(np.count_nonzero(support))
        measured_overlap_ratio = float(
            measured_overlap_pixels / max(1, support_pixels)
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        support_boundary = support & ~cv2.erode(
            support.astype(np.uint8), kernel
        ).astype(bool)
        fixed_boundary = fixed_exact_support & ~cv2.erode(
            fixed_exact_support.astype(np.uint8), kernel
        ).astype(bool)
        measured_overlap_in_fixed_exact = int(
            np.count_nonzero(measured_overlap & fixed_exact_support)
        )
        measured_overlap_on_both_boundaries = int(
            np.count_nonzero(
                measured_overlap & support_boundary & fixed_boundary
            )
        )
        measured_overlap_on_candidate_boundary = int(
            np.count_nonzero(measured_overlap & support_boundary)
        )
        boundary_alias_partition = bool(
            measured_overlap_pixels > 0
            and (
                measured_overlap_pixels <= 384
                or (
                    measured_overlap_pixels <= 4096
                    and measured_overlap_ratio <= 0.15
                )
            )
            and measured_overlap_in_fixed_exact == measured_overlap_pixels
        )
        decoupled_guard = bool(
            not fixed.background_panel_lock_required
        )
        candidate = (
            result.copy()
            if decoupled_guard
            else result & ~fixed_mask
        )
        candidate_supports_retained = bool(
            np.any(candidate) and not np.any(support & ~candidate)
        )
        all_support_single_owner_covered = bool(
            np.any(candidate)
            and not np.any(
                support & ~(candidate | fixed_exact_support)
            )
        )
        row_boundary_passed = all_support_single_owner_covered
        if row_boundary_passed and not decoupled_guard:
            for row in np.flatnonzero(np.any(overlap, axis=1)):
                remaining_columns = np.flatnonzero(candidate[row])
                removed_columns = np.flatnonzero(overlap[row])
                if remaining_columns.size == 0:
                    row_boundary_passed = False
                    break
                left = int(remaining_columns[0])
                right = int(remaining_columns[-1])
                if (
                    int(remaining_columns.size) != right - left + 1
                    or np.any(
                        (removed_columns >= left)
                        & (removed_columns <= right)
                    )
                ):
                    row_boundary_passed = False
                    break
        accepted = bool(
            (
                (
                    measured_overlap_pixels == 0
                    and candidate_supports_retained
                )
                or boundary_alias_partition
            )
            and all_support_single_owner_covered
            and row_boundary_passed
        )
        rows.append(
            {
                "fixed_track_id": int(fixed.track_id),
                "fixed_frame_id": int(fixed.frame_id),
                "trimmed_pixel_count": (
                    overlap_pixels
                    if accepted and not decoupled_guard
                    else 0
                ),
                "requested_overlap_pixel_count": overlap_pixels,
                "measured_support_overlap_pixel_count": (
                    measured_overlap_pixels
                ),
                "measured_support_overlap_ratio": measured_overlap_ratio,
                "measured_overlap_in_fixed_exact_support_pixel_count": (
                    measured_overlap_in_fixed_exact
                ),
                "measured_overlap_on_both_inner_boundaries_pixel_count": (
                    measured_overlap_on_both_boundaries
                ),
                "measured_overlap_on_candidate_inner_boundary_pixel_count": (
                    measured_overlap_on_candidate_boundary
                ),
                "cross_track_boundary_alias_partition": (
                    boundary_alias_partition
                ),
                "cross_track_boundary_alias_absolute_limit_pixels": 4096,
                "cross_track_boundary_alias_relative_limit": 0.15,
                "delegated_measured_boundary_pixel_count": (
                    measured_overlap_pixels
                    if boundary_alias_partition
                    else 0
                ),
                "delegated_measured_boundary_owner": (
                    "existing_fixed_corridor"
                    if boundary_alias_partition
                    else None
                ),
                "guard_overlap_retained_for_decoupled_owner": (
                    decoupled_guard
                ),
                "zero_measured_support_intersection": bool(
                    measured_overlap_pixels == 0
                ),
                "all_member_supports_retained": (
                    candidate_supports_retained
                ),
                "all_member_supports_single_owner_covered": (
                    all_support_single_owner_covered
                ),
                "per_row_boundary_trim_passed": row_boundary_passed,
                "subtraction_row_contiguous": row_boundary_passed,
                "trimmed_pixel_owner": "existing_fixed_corridor",
                "accepted": accepted,
            }
        )
        if not accepted:
            return result, False, rows
        result = np.ascontiguousarray(candidate)
    return result, True, rows


def _bounded_exact_corridor_transfer(
    measured_support: np.ndarray,
    owner_guard: np.ndarray,
    panel_valid_mask: np.ndarray,
    *,
    dilation_pixels: int = 2,
    reserved_foreign_support: np.ndarray | None = None,
    member_supports: Sequence[np.ndarray] | None = None,
    member_context_guard_pixels: int = 3,
    maximum_member_context_ratio: float = 1.50,
    member_context_mode: str = "row_hull",
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Limit RGB replacement to exact support plus bounded per-member context.

    ``measured_support`` remains the immutable, fail-closed evidence.  Optional
    context is built for every member independently (close, then a per-row
    solid hull) before the members are combined.  This fills segmentation
    holes without bridging a whole object-rich group into the rectangular RGB
    patches produced by a hull of the group union.

    Cosmetic dilation and member context may yield to another object's exact
    support.  Exact-support overlap is retained because that foreign object may
    still select another real-frame observation; the unconditional final
    assignment audit remains fail-closed if the overlap persists.
    """

    support = np.asarray(measured_support, dtype=bool)
    guard = np.asarray(owner_guard, dtype=bool)
    panel_valid = np.asarray(panel_valid_mask, dtype=bool)
    if support.shape != guard.shape or support.shape != panel_valid.shape:
        raise ValueError("Corridor transfer masks must have matching shapes")
    reserved = (
        np.zeros_like(support)
        if reserved_foreign_support is None
        else np.asarray(reserved_foreign_support, dtype=bool)
    )
    if reserved.shape != support.shape:
        raise ValueError("Reserved foreign support mask shape mismatch")
    if not np.any(support) or not np.any(guard):
        raise ValueError("Corridor transfer support and guard must be non-empty")
    dilation = int(dilation_pixels)
    if not 0 <= dilation <= 2:
        raise ValueError("Corridor RGB transfer dilation must be in [0, 2]")
    context_guard = int(member_context_guard_pixels)
    if not 0 <= context_guard <= 4:
        raise ValueError(
            "Corridor member context guard must be in [0, 4]"
        )
    context_ratio = float(maximum_member_context_ratio)
    if not np.isfinite(context_ratio) or not 1.0 <= context_ratio <= 2.0:
        raise ValueError(
            "Corridor member context ratio must be finite and in [1, 2]"
        )
    context_mode = str(member_context_mode)
    if context_mode not in {
        "row_hull",
        "convex_hull",
        "bounding_box",
    }:
        raise ValueError(
            "Corridor member context mode must be row_hull, convex_hull, "
            "or bounding_box"
        )
    missing_guard = int(np.count_nonzero(support & ~guard))
    missing_panel = int(np.count_nonzero(support & ~panel_valid))
    exact_foreign_overlap = int(np.count_nonzero(support & reserved))
    normalized_member_supports: list[np.ndarray] = []
    if member_supports is not None:
        for member_support in member_supports:
            member = np.asarray(member_support, dtype=bool).copy()
            if member.shape != support.shape:
                raise ValueError(
                    "Corridor member support mask shape mismatch"
                )
            member &= support
            if np.any(member):
                normalized_member_supports.append(
                    np.ascontiguousarray(member)
                )
        member_union = (
            np.logical_or.reduce(normalized_member_supports)
            if normalized_member_supports
            else np.zeros_like(support)
        )
        if np.any(support & ~member_union):
            normalized_member_supports.append(
                np.ascontiguousarray(support & ~member_union)
            )
    audit = {
        "owner_only_guard_pixel_count": int(np.count_nonzero(guard)),
        "exact_member_transfer_support_pixel_count": int(
            np.count_nonzero(support)
        ),
        "rgb_transfer_dilation_pixels": dilation,
        "missing_member_support_from_owner_guard_pixel_count": (
            missing_guard
        ),
        "missing_member_support_from_panel_valid_pixel_count": (
            missing_panel
        ),
        "all_member_measured_support_retained": bool(
            missing_guard == 0 and missing_panel == 0
        ),
        "reserved_foreign_support_pixel_count": int(
            np.count_nonzero(reserved)
        ),
        "exact_member_foreign_support_overlap_pixel_count": (
            exact_foreign_overlap
        ),
        "exact_member_overlap_deferred_to_final_assignment_audit": bool(
            exact_foreign_overlap > 0
        ),
        "only_optional_dilation_may_yield_to_foreign_support": True,
        "member_context_requested": bool(normalized_member_supports),
        "member_context_member_count": len(normalized_member_supports),
        "member_context_guard_pixels": context_guard,
        "maximum_member_context_ratio": context_ratio,
        "member_context_mode": context_mode,
        "rgb_blended_or_generated": False,
    }
    if missing_guard or missing_panel:
        return None, audit
    transfer = np.ascontiguousarray(support.copy())
    if dilation:
        size = 2 * dilation + 1
        transfer = cv2.dilate(
            support.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)),
        ).astype(bool)
    member_context = np.zeros_like(support)
    member_context_rows: list[dict[str, object]] = []
    for member_index, member in enumerate(normalized_member_supports):
        closed = cv2.morphologyEx(
            member.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ).astype(bool)
        if context_mode == "row_hull":
            candidate = _row_contiguous_guard(
                closed,
                guard_pixels=context_guard,
            )
        elif context_mode == "convex_hull":
            contours, _ = cv2.findContours(
                closed.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            points = np.concatenate(contours, axis=0)
            candidate = np.zeros_like(closed)
            cv2.fillConvexPoly(
                candidate,
                cv2.convexHull(points),
                1,
            )
            if context_guard:
                size = 2 * context_guard + 1
                candidate = cv2.dilate(
                    candidate.astype(np.uint8),
                    cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (size, size),
                    ),
                ).astype(bool)
        else:
            yy, xx = np.nonzero(closed)
            y0 = max(0, int(yy.min()) - context_guard)
            y1 = min(
                closed.shape[0],
                int(yy.max()) + context_guard + 1,
            )
            x0 = max(0, int(xx.min()) - context_guard)
            x1 = min(
                closed.shape[1],
                int(xx.max()) + context_guard + 1,
            )
            candidate = np.zeros_like(closed)
            candidate[y0:y1, x0:x1] = True
        exact_pixels = int(np.count_nonzero(member))
        candidate_pixels = int(np.count_nonzero(candidate))
        ratio = float(candidate_pixels / max(1, exact_pixels))
        accepted = bool(ratio <= context_ratio)
        if accepted:
            member_context |= candidate
        member_context_rows.append(
            {
                "member_index": member_index,
                "exact_support_pixel_count": exact_pixels,
                "candidate_context_pixel_count": candidate_pixels,
                "candidate_to_exact_ratio": ratio,
                "accepted": accepted,
                "fallback": (
                    None if accepted else "exact_plus_requested_dilation"
                ),
            }
        )
    transfer |= member_context
    bounded_before_reservation = transfer & guard & panel_valid
    optional_dilation_reservation = reserved & ~support
    excluded_optional_dilation = int(
        np.count_nonzero(
            bounded_before_reservation & optional_dilation_reservation
        )
    )
    transfer = np.ascontiguousarray(
        bounded_before_reservation & ~optional_dilation_reservation
    )
    if np.any(support & ~transfer):
        raise RuntimeError("Bounded corridor transfer lost measured support")
    audit["rgb_transfer_pixel_count"] = int(np.count_nonzero(transfer))
    audit["member_context_rows"] = member_context_rows
    audit["accepted_member_context_pixel_count"] = int(
        np.count_nonzero(member_context & ~support)
    )
    audit["excluded_optional_dilation_foreign_support_pixel_count"] = (
        excluded_optional_dilation
    )
    audit["guard_minus_rgb_transfer_pixel_count"] = int(
        np.count_nonzero(guard & ~transfer)
    )
    return transfer, audit


def _expand_resolved_interval_rgb_context(
    intervals: Sequence[InspectionPreSeamHardOwnerInterval],
    identity_frames: Sequence[InspectionIdentityOwnerFrame],
    *,
    reserved_external_support: np.ndarray | None = None,
    bounding_box_track_ids: set[int] | None = None,
) -> tuple[
    tuple[InspectionPreSeamHardOwnerInterval, ...],
    dict[str, object],
]:
    """Add bounded real-RGB context only after exact owner resolution."""

    selected = tuple(intervals)
    if not selected:
        return (), {
            "policy": "post_resolution_per_member_real_rgb_context",
            "interval_count": 0,
            "expanded_interval_count": 0,
            "different_frame_overlap_pixel_count": 0,
            "pass": True,
        }
    frames = {int(frame.frame_id): frame for frame in identity_frames}
    exact_masks = tuple(
        np.asarray(
            interval.lock_mask
            if interval.rgb_transfer_mask is None
            else interval.rgb_transfer_mask,
            dtype=bool,
        )
        for interval in selected
    )
    external_reserved = (
        np.zeros_like(exact_masks[0])
        if reserved_external_support is None
        else np.asarray(reserved_external_support, dtype=bool)
    )
    if external_reserved.shape != exact_masks[0].shape:
        raise ValueError("External RGB context reservation shape mismatch")
    expanded: list[InspectionPreSeamHardOwnerInterval] = []
    rows: list[dict[str, object]] = []
    bounding_tracks = (
        None
        if bounding_box_track_ids is None
        else set(bounding_box_track_ids)
    )
    for interval_index, (interval, exact) in enumerate(
        zip(selected, exact_masks, strict=True)
    ):
        frame = frames.get(int(interval.frame_id))
        if frame is None:
            raise RuntimeError(
                "Resolved RGB context interval lacks its selected frame"
            )
        reserved_masks = [external_reserved]
        reserved_masks.extend(
            other_exact
            for other_index, (other, other_exact) in enumerate(
                zip(selected, exact_masks, strict=True)
            )
            if other_index != interval_index
            and int(other.frame_id) != int(interval.frame_id)
        )
        reserved_masks.extend(
            np.asarray(other.rgb_transfer_mask, dtype=bool)
            for other in expanded
            if int(other.frame_id) != int(interval.frame_id)
            and other.rgb_transfer_mask is not None
        )
        reserved = (
            np.logical_or.reduce(reserved_masks)
            if reserved_masks
            else np.zeros_like(exact)
        )
        member_supports = tuple(
            np.ascontiguousarray(
                np.asarray(member, dtype=bool) & exact
            )
            for member in interval.rgb_context_member_supports
            if np.any(np.asarray(member, dtype=bool) & exact)
        )
        expanded_guard_seed = np.asarray(
            interval.lock_mask, dtype=bool
        ).copy()
        for member in member_supports:
            yy, xx = np.nonzero(member)
            y0 = max(0, int(yy.min()) - 3)
            y1 = min(member.shape[0], int(yy.max()) + 4)
            x0 = max(0, int(xx.min()) - 3)
            x1 = min(member.shape[1], int(xx.max()) + 4)
            expanded_guard_seed[y0:y1, x0:x1] = True
        expanded_lock = _row_contiguous_guard(
            expanded_guard_seed,
            guard_pixels=0,
        )
        raw_track_id = int(interval.track_id)
        identity_track_id = (
            raw_track_id - 1_000_000
            if 1_000_000 <= raw_track_id < 2_000_000
            else raw_track_id
        )
        member_context_mode = (
            "bounding_box"
            if (
                bounding_tracks is None
                or identity_track_id in bounding_tracks
            )
            else "convex_hull"
        )
        transfer, transfer_audit = _bounded_exact_corridor_transfer(
            exact,
            expanded_lock,
            frame.panel_valid_mask,
            dilation_pixels=2 if member_supports else 0,
            reserved_foreign_support=reserved,
            member_supports=member_supports or None,
            maximum_member_context_ratio=2.0,
            member_context_mode=member_context_mode,
        )
        if transfer is None:
            raise RuntimeError(
                "Post-resolution object RGB context lost exact support"
            )
        expanded_interval = replace(
            interval,
            lock_mask=np.ascontiguousarray(expanded_lock),
            rgb_transfer_mask=np.ascontiguousarray(transfer),
            owner_only_mask=np.ascontiguousarray(
                np.asarray(interval.owner_only_mask, dtype=bool)
                | transfer
            ),
        )
        expanded.append(expanded_interval)
        rows.append(
            {
                "track_id": int(interval.track_id),
                "frame_id": int(interval.frame_id),
                "exact_support_pixel_count": int(
                    np.count_nonzero(exact)
                ),
                "final_transfer_pixel_count": int(
                    np.count_nonzero(transfer)
                ),
                "selected_member_context_mode": member_context_mode,
                **transfer_audit,
            }
        )
    different_frame_overlap_pixels = 0
    for first_index, first in enumerate(expanded):
        first_transfer = np.asarray(
            first.rgb_transfer_mask, dtype=bool
        )
        for second in expanded[first_index + 1 :]:
            if int(first.frame_id) == int(second.frame_id):
                continue
            different_frame_overlap_pixels += int(
                np.count_nonzero(
                    first_transfer
                    & np.asarray(second.rgb_transfer_mask, dtype=bool)
                )
            )
    if different_frame_overlap_pixels:
        raise RuntimeError(
            "Post-resolution object RGB context created cross-source overlap"
        )
    return tuple(expanded), {
        "policy": (
            "post_resolution_per_member_convex_hull_real_rgb_context_with_"
            "bounding_box_only_for_low_coverage_completion_foreign_exact_"
            "and_transfer_reserved"
        ),
        "interval_count": len(selected),
        "expanded_interval_count": sum(
            int(
                row["final_transfer_pixel_count"]
                > row["exact_support_pixel_count"]
            )
            for row in rows
        ),
        "different_frame_overlap_pixel_count": 0,
        "reserved_external_support_pixel_count": int(
            np.count_nonzero(external_reserved)
        ),
        "rows": rows,
        "identity_or_csp_decision_modified": False,
        "rgb_blended_or_generated": False,
        "pass": True,
    }


def _level4b_panel_evidence_score(
    track_evidence: Sequence[
        tuple[int, bool, Sequence[float] | None, int, bool]
    ],
    *,
    corridor_center_x: float,
    panel_center_x: float,
    panel_index: int,
    existing_corridor_frame: bool,
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Rank Level4b panels by member evidence before layout reuse."""

    complete = [row for row in track_evidence if row[1]]
    rank_widths = {
        len(row[2]) for row in complete if row[2] is not None
    }
    if len(rank_widths) > 1:
        raise ValueError("Level4b completeness ranks have mixed widths")
    rank_width = next(iter(rank_widths), 0)
    total_weight = float(sum(max(1, int(row[3])) for row in complete))
    weighted_rank = tuple(
        float(
            sum(
                max(1, int(row[3])) * float(row[2][index])
                for row in complete
                if row[2] is not None
            )
            / max(1.0, total_weight)
        )
        for index in range(rank_width)
    )
    reference_count = sum(int(row[4]) for row in track_evidence)
    score = (
        float(len(complete)),
        -abs(float(corridor_center_x) - float(panel_center_x)),
        *weighted_rank,
        float(reference_count),
        -float(panel_index),
        float(int(existing_corridor_frame)),
    )
    return score, {
        "eligible_complete_observation_count": len(complete),
        "eligible_complete_observation_track_ids": [
            int(row[0]) for row in complete
        ],
        "weighted_complete_observation_rank": list(weighted_rank),
        "weighted_complete_observation_support_pixels": int(
            total_weight
        ),
        "any_reference_observation_count": reference_count,
        "any_reference_observation_track_ids": [
            int(row[0]) for row in track_evidence if row[4]
        ],
        "existing_corridor_frame_last_tiebreak_only": bool(
            existing_corridor_frame
        ),
        "score_policy": (
            "eligible_complete_count_then_corridor_center_then_support_"
            "weighted_selection_rank_then_any_reference_count_then_panel_then_"
            "existing_corridor_frame_last_tiebreak"
        ),
    }


def _build_panel_native_preseam_intervals(
    owners: Sequence[InspectionForegroundIdentityOwner],
    identity_frames: Sequence[InspectionIdentityOwnerFrame],
    *,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    geometric_track_ids: set[int],
    config: InspectionIdentityRuntimeConfig,
    bounded_context_completion_track_ids: set[int] | None = None,
) -> tuple[
    tuple[InspectionPreSeamHardOwnerInterval, ...],
    dict[str, object],
]:
    """Keep same-panel objects in the existing reference-panel RGB raster.

    Aligned depth and DIS establish identity and select the real panel.  This
    function never performs a novel-view object warp: the source mask reaches
    the canvas only through that panel's existing reference-plane inverse map.
    """

    frame_by_panel = {
        int(frame.panel_index): frame for frame in identity_frames
    }
    intervals: list[InspectionPreSeamHardOwnerInterval] = []
    rows: list[dict[str, object]] = []
    seen_tracks: set[int] = set()
    bounded_context_tracks = (
        set()
        if bounded_context_completion_track_ids is None
        else set(bounded_context_completion_track_ids)
    )
    for owner in owners:
        track_id = (
            None
            if owner.identity_track_id is None
            else int(owner.identity_track_id)
        )
        row: dict[str, object] = {
            "identity_track_id": track_id,
            "frame_id": int(owner.frame_id),
            "source_panel_index": int(owner.panel_index),
            "target_panel_index": (
                None
                if owner.target_panel_index is None
                else int(owner.target_panel_index)
            ),
            "accepted": False,
            "rgb_sampling": (
                "existing_reference_panel_inverse_map_only"
            ),
            "novel_view_object_warp_used": False,
        }
        if track_id is None or track_id not in geometric_track_ids:
            row["reason"] = "direct_structural_gate_not_passed"
            rows.append(row)
            continue
        if not config.panel_native_preseam_lock_enabled:
            row["reason"] = "panel_native_preseam_lock_disabled"
            rows.append(row)
            continue
        if (
            owner.target_panel_index is None
            or int(owner.panel_index) != int(owner.target_panel_index)
        ):
            row["reason"] = "source_target_panel_mismatch"
            rows.append(row)
            continue
        if track_id in seen_tracks:
            row["reason"] = "duplicate_identity_track_candidate"
            rows.append(row)
            continue
        frame = frame_by_panel.get(int(owner.panel_index))
        if frame is None or int(frame.frame_id) != int(owner.frame_id):
            raise RuntimeError(
                "Panel-native identity owner panel/frame mapping changed"
            )
        (
            corner_x,
            map_x,
            map_y,
            map_valid,
            _,
        ) = _reference_panel_inverse_maps(
            source_pose=np.asarray(frame.camera_to_world, dtype=np.float64),
            panel_index=int(owner.panel_index),
            layout=layout,
            intrinsics=intrinsics,
        )
        source_mask = np.asarray(owner.source_mask, dtype=bool)
        safe_x = np.where(map_valid, map_x, -1.0).astype(
            np.float32, copy=False
        )
        safe_y = np.where(map_valid, map_y, -1.0).astype(
            np.float32, copy=False
        )
        local = (
            accelerated_remap(
                source_mask.astype(np.uint8),
                safe_x,
                safe_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            > 0
        ) & map_valid
        local_count = int(np.count_nonzero(local))
        if local_count == 0:
            row["reason"] = "reference_panel_map_has_no_object_support"
            rows.append(row)
            continue
        rounded_x = np.rint(map_x[local]).astype(np.int32)
        rounded_y = np.rint(map_y[local]).astype(np.int32)
        inside = (
            (rounded_x >= 0)
            & (rounded_x < source_mask.shape[1])
            & (rounded_y >= 0)
            & (rounded_y < source_mask.shape[0])
        )
        represented = np.zeros(source_mask.shape, dtype=bool)
        represented[rounded_y[inside], rounded_x[inside]] = True
        source_count = int(np.count_nonzero(source_mask))
        source_coverage = float(
            np.count_nonzero(represented & source_mask)
            / max(1, source_count)
        )
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            local.astype(np.uint8), 8
        )
        dominant_ratio = float(
            int(np.max(stats[1:, cv2.CC_STAT_AREA], initial=0))
            / max(1, local_count)
        )
        row.update(
            {
                "source_mask_pixel_count": source_count,
                "mapped_footprint_pixel_count": local_count,
                "inverse_source_coverage_ratio": source_coverage,
                "mapped_component_count": int(component_count - 1),
                "dominant_mapped_component_ratio": dominant_ratio,
            }
        )
        bounded_context_completion = bool(
            track_id in bounded_context_tracks
            and source_coverage >= 0.85
            and dominant_ratio
            >= config.panel_native_minimum_component_ratio
        )
        row["bounded_context_completion_used"] = bool(
            bounded_context_completion
            and source_coverage
            < config.panel_native_minimum_source_coverage_ratio
        )
        if (
            source_coverage
            < config.panel_native_minimum_source_coverage_ratio
            and not bounded_context_completion
        ):
            row["reason"] = "whole_source_mask_not_represented"
            rows.append(row)
            continue
        if dominant_ratio < config.panel_native_minimum_component_ratio:
            row["reason"] = "mapped_mask_not_one_complete_component"
            rows.append(row)
            continue
        footprint = np.zeros(
            (int(layout.height), int(layout.width)), dtype=bool
        )
        footprint[
            :, corner_x : corner_x + local.shape[1]
        ] = local
        lock = _row_contiguous_guard(
            footprint,
            guard_pixels=int(config.panel_native_lock_guard_pixels),
        )
        missing_panel_coverage = int(
            np.count_nonzero(lock & ~frame.panel_valid_mask)
        )
        row.update(
            {
                "lock_pixel_count": int(np.count_nonzero(lock)),
                "selected_panel_missing_valid_pixel_count": (
                    missing_panel_coverage
                ),
                "guard_pixels": int(
                    config.panel_native_lock_guard_pixels
                ),
                "row_contiguous": True,
            }
        )
        if missing_panel_coverage:
            row["reason"] = "selected_panel_lacks_complete_lock_coverage"
            rows.append(row)
            continue
        intervals.append(
            InspectionPreSeamHardOwnerInterval(
                track_id=1_000_000 + track_id,
                panel_index=int(owner.panel_index),
                frame_id=int(owner.frame_id),
                lock_mask=lock,
                union_footprint=np.ascontiguousarray(footprint),
                rgb_transfer_mask=np.ascontiguousarray(footprint),
                owner_only_mask=np.ascontiguousarray(lock),
                rgb_context_member_supports=(
                    np.ascontiguousarray(footprint),
                ),
                background_panel_lock_required=False,
            )
        )
        seen_tracks.add(track_id)
        row["accepted"] = True
        row["reason"] = (
            "same_panel_bounded_context_completion_lock_candidate"
            if row["bounded_context_completion_used"]
            else "same_panel_reference_raster_lock_candidate"
        )
        rows.append(row)
    return tuple(intervals), {
        "schema": "inspection-panel-native-preseam-owner-plan/v1",
        "policy": (
            "stable_identity_same_source_target_panel_existing_reference_"
            "inverse_map_row_contiguous_hard_owner_before_graphcut"
        ),
        "enabled": bool(config.panel_native_preseam_lock_enabled),
        "candidate_count": len(owners),
        "accepted_interval_count": len(intervals),
        "rejected_count": len(rows) - len(intervals),
        "minimum_source_coverage_ratio": float(
            config.panel_native_minimum_source_coverage_ratio
        ),
        "minimum_component_ratio": float(
            config.panel_native_minimum_component_ratio
        ),
        "bounded_context_completion_minimum_source_coverage_ratio": 0.85,
        "guard_pixels": int(config.panel_native_lock_guard_pixels),
        "tracks": rows,
        "rgb_generated": False,
        "pose_modified": False,
        "true_depth_object_overlay_used": False,
    }


def _resolve_shelf_native_owner_conflict_groups(
    owners: Sequence[InspectionForegroundIdentityOwner],
    planner_audit: Mapping[str, object],
    identity_frames: Sequence[InspectionIdentityOwnerFrame],
    *,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    config: InspectionIdentityRuntimeConfig,
) -> tuple[
    tuple[InspectionForegroundIdentityOwner, ...],
    tuple[InspectionPreSeamHardOwnerInterval, ...],
    dict[str, object],
    dict[str, object],
    frozenset[int],
]:
    """Assign every overlapping shelf group one complete real RGB frame.

    Preliminary exact reference-raster footprints are used only to discover
    conflicts.  A bounded global CSP chooses one independently complete
    observation per track; only recomputed final footprints which still
    overlap must share a real frame.  It never transitive-closes the initial
    overlap graph.  A pair proven incompatible may use a small canonical-union
    composite corridor only when one member's real panel fully covers both
    measured supports and no co-observed independent-instance evidence exists.
    Co-observed distinct pairs use their exact common-frame source union.
    Remaining higher-order UNSAT is deletion-reduced to a bounded core whose
    target-union corridor must fit one complete selected real panel.  A
    two-object level-2 boundary-clear corridor remains the last native-raster
    fallback.  Corridors cannot enter the true-depth mesh.
    """

    owner_by_track: dict[int, InspectionForegroundIdentityOwner] = {}
    owner_order: list[int] = []
    for owner in owners:
        if owner.identity_track_id is None:
            raise ValueError("Shelf inventory owner is missing its track ID")
        track_id = int(owner.identity_track_id)
        if track_id in owner_by_track:
            raise ValueError("Shelf inventory owner track IDs must be unique")
        owner_by_track[track_id] = owner
        owner_order.append(track_id)
    frame_by_id = {int(frame.frame_id): frame for frame in identity_frames}
    frame_by_panel = {
        int(frame.panel_index): frame for frame in identity_frames
    }
    if (
        len(frame_by_id) != len(identity_frames)
        or len(frame_by_panel) != len(identity_frames)
    ):
        raise ValueError("Shelf inventory RGB frames are not one-to-one")
    dispositions = {
        int(row["track_id"]): row
        for row in planner_audit["track_dispositions"]
    }
    eligible_frames_by_track: dict[int, dict[int, dict[str, object]]] = {}
    boundary_clear_frames_by_track: dict[
        int, dict[int, dict[str, object]]
    ] = {}
    for track_id in owner_order:
        disposition = dispositions.get(track_id)
        if disposition is None:
            raise RuntimeError(
                "Shelf inventory owner lacks its planner disposition"
            )
        eligible_frames_by_track[track_id] = {
            int(row["frame_id"]): row
            for row in disposition["observations"]
            if row["eligible_complete_shelf_observation"] is True
        }
        if not eligible_frames_by_track[track_id]:
            raise RuntimeError(
                "Shelf inventory owner lacks a complete real RGB observation"
            )
        boundary_clear_frames_by_track[track_id] = {
            int(row["frame_id"]): row
            for row in disposition["observations"]
            if bool(row.get("gates", {}).get("source_boundary_clear", False))
        }

    native_owner_reselection_rows: list[dict[str, object]] = []
    for track_id in owner_order:
        current_owner = owner_by_track[track_id]
        current_intervals, _ = _build_panel_native_preseam_intervals(
            (current_owner,),
            identity_frames,
            layout=layout,
            intrinsics=intrinsics,
            geometric_track_ids={track_id},
            config=config,
            bounded_context_completion_track_ids=set(owner_order),
        )
        if len(current_intervals) == 1:
            continue
        source_masks_by_panel = {
            int(panel_index): np.asarray(mask, dtype=bool)
            for panel_index, mask
            in current_owner.reference_observation_masks
        }
        alternatives: list[
            tuple[
                tuple[float, ...],
                InspectionForegroundIdentityOwner,
                int,
            ]
        ] = []
        for frame_id, observation in eligible_frames_by_track[
            track_id
        ].items():
            frame = frame_by_id.get(int(frame_id))
            if frame is None:
                continue
            source_mask = source_masks_by_panel.get(
                int(frame.panel_index)
            )
            if source_mask is None or not np.any(source_mask):
                continue
            proposed = replace(
                current_owner,
                panel_index=int(frame.panel_index),
                target_panel_index=int(frame.panel_index),
                frame_id=int(frame.frame_id),
                source_index=int(frame.source_index),
                source_mask=np.ascontiguousarray(source_mask),
            )
            trial_intervals, _ = _build_panel_native_preseam_intervals(
                (proposed,),
                identity_frames,
                layout=layout,
                intrinsics=intrinsics,
                geometric_track_ids={track_id},
                config=config,
            )
            if len(trial_intervals) != 1:
                continue
            alternatives.append(
                (
                    tuple(
                        float(value)
                        for value in observation["selection_rank"]
                    ),
                    proposed,
                    int(np.count_nonzero(source_mask)),
                )
            )
        if not alternatives:
            continue
        alternatives.sort(
            key=lambda item: (
                item[0],
                item[2],
                -int(item[1].panel_index),
            ),
            reverse=True,
        )
        _, selected_owner, selected_pixels = alternatives[0]
        owner_by_track[track_id] = selected_owner
        native_owner_reselection_rows.append(
            {
                "track_id": int(track_id),
                "original_frame_id": int(current_owner.frame_id),
                "selected_frame_id": int(selected_owner.frame_id),
                "selected_panel_index": int(selected_owner.panel_index),
                "selected_source_mask_pixel_count": int(selected_pixels),
                "eligible_complete_alternative_count": len(alternatives),
                "reason": (
                    "original_same_panel_inverse_map_incomplete_reselected_"
                    "to_complete_real_rgb_observation"
                ),
                "rgb_generated": False,
                "pose_modified": False,
            }
        )

    corridor_intervals: list[InspectionPreSeamHardOwnerInterval] = []
    corridor_handled_track_ids: set[int] = set()
    def build_current_intervals() -> tuple[
        tuple[InspectionPreSeamHardOwnerInterval, ...],
        dict[str, object],
    ]:
        current = tuple(
            owner_by_track[value]
            for value in owner_order
            if value not in corridor_handled_track_ids
        )
        base_intervals, base_audit = _build_panel_native_preseam_intervals(
            current,
            identity_frames,
            layout=layout,
            intrinsics=intrinsics,
            geometric_track_ids={
                value
                for value in owner_order
                if value not in corridor_handled_track_ids
            },
            config=config,
            bounded_context_completion_track_ids=set(owner_order),
        )
        return (
            (*base_intervals, *corridor_intervals),
            {
                **base_audit,
                "accepted_interval_count": (
                    int(base_audit["accepted_interval_count"])
                    + len(corridor_intervals)
                ),
                "object_rich_corridor_interval_count": len(
                    corridor_intervals
                ),
                "object_rich_corridor_handled_track_ids": sorted(
                    corridor_handled_track_ids
                ),
            },
        )

    def interval_track_id(
        interval: InspectionPreSeamHardOwnerInterval,
    ) -> int:
        value = int(interval.track_id)
        return value - 1_000_000 if value >= 1_000_000 else value

    def overlap_components(
        intervals: Sequence[InspectionPreSeamHardOwnerInterval],
    ) -> list[tuple[int, ...]]:
        selected = {
            interval_track_id(interval): interval
            for interval in intervals
            if interval_track_id(interval)
            not in deferred_post_background_track_ids
        }
        selected_order = [
            track_id for track_id in owner_order if track_id in selected
        ]
        adjacency = {track_id: set() for track_id in selected_order}
        for position, first_id in enumerate(selected_order):
            first = selected[first_id]
            first_mask = (
                np.asarray(first.lock_mask, dtype=bool)
                if first.rgb_transfer_mask is None
                else np.asarray(first.rgb_transfer_mask, dtype=bool)
            )
            for second_id in selected_order[position + 1 :]:
                second = selected[second_id]
                second_mask = (
                    np.asarray(second.lock_mask, dtype=bool)
                    if second.rgb_transfer_mask is None
                    else np.asarray(second.rgb_transfer_mask, dtype=bool)
                )
                if np.any(first_mask & second_mask):
                    adjacency[first_id].add(second_id)
                    adjacency[second_id].add(first_id)
        components: list[tuple[int, ...]] = []
        unseen = set(selected_order)
        while unseen:
            seed = min(unseen)
            stack = [seed]
            component: set[int] = set()
            while stack:
                value = stack.pop()
                if value in component:
                    continue
                component.add(value)
                stack.extend(adjacency[value] - component)
            unseen -= component
            if len(component) > 1:
                components.append(tuple(sorted(component)))
        return components

    resolution_events: list[dict[str, object]] = []
    initial_intervals, initial_interval_audit = build_current_intervals()
    deferred_post_background_track_ids = {
        int(row["track_id"]) for row in native_owner_reselection_rows
    } | {
        int(row["identity_track_id"])
        for row in initial_interval_audit.get("tracks", [])
        if row.get("identity_track_id") is not None
        and bool(row.get("bounded_context_completion_used", False))
    }
    if not deferred_post_background_track_ids.issubset(
        {
            interval_track_id(interval)
            for interval in initial_intervals
        }
    ):
        raise RuntimeError(
            "Deferred shelf RGB owners lack a complete panel-native interval"
        )
    csp_search_state_count = 0
    csp_search_state_limit = 200_000
    mus_search_state_count = 0
    mus_search_state_limit = 200_000
    maximum_iterations = max(1, len(owner_order))
    final_intervals: tuple[InspectionPreSeamHardOwnerInterval, ...] = ()
    final_interval_audit: dict[str, object] = {}
    for iteration in range(maximum_iterations):
        final_intervals, final_interval_audit = build_current_intervals()
        components = overlap_components(final_intervals)
        conflicting_components = [
            component
            for component in components
            if len(
                {
                    int(owner_by_track[track_id].frame_id)
                    for track_id in component
                }
            )
            > 1
        ]
        active_native_track_ids = tuple(
            track_id
            for track_id in owner_order
            if track_id not in corridor_handled_track_ids
            and track_id not in deferred_post_background_track_ids
            and any(
                interval_track_id(interval) == track_id
                for interval in final_intervals
            )
        )
        fixed_corridor_conflict = any(
            int(fixed.frame_id) != int(candidate.frame_id)
            and interval_track_id(candidate) in active_native_track_ids
            and interval_track_id(fixed)
            not in deferred_post_background_track_ids
            and np.any(
                np.asarray(fixed.rgb_transfer_mask, dtype=bool)
                & np.asarray(candidate.rgb_transfer_mask, dtype=bool)
            )
            for fixed in corridor_intervals
            for candidate in final_intervals
            if candidate is not fixed
        )
        # Solve one global assignment over every active native shelf owner.
        # A local component reassignment can move a footprint into a
        # previously disjoint component and oscillate.  Global pair
        # compatibility plus the unconditional final all-pairs audit closes
        # that hole without imposing any transitive same-frame constraint.
        unresolved = (
            [active_native_track_ids]
            if conflicting_components or fixed_corridor_conflict
            else []
        )
        if not unresolved:
            break
        for component in unresolved:
            option_domains: dict[
                int,
                list[
                    tuple[
                        tuple[float, ...],
                        InspectionForegroundIdentityOwner,
                        InspectionPreSeamHardOwnerInterval,
                        dict[str, object],
                    ]
                ],
            ] = {}
            for track_id in component:
                current_owner = owner_by_track[track_id]
                source_masks = {
                    int(value_panel): np.asarray(value_mask, dtype=bool)
                    for value_panel, value_mask
                    in current_owner.reference_observation_masks
                }
                options = []
                for frame_id, observation in sorted(
                    eligible_frames_by_track[track_id].items()
                ):
                    frame = frame_by_id.get(frame_id)
                    if frame is None:
                        continue
                    panel_index = int(frame.panel_index)
                    source_mask = source_masks.get(panel_index)
                    if source_mask is None or not np.any(source_mask):
                        continue
                    points = sample_mask_world_points(
                        mask=source_mask,
                        depth_mm=frame.depth_mm,
                        reliable_depth=frame.reliable_depth,
                        camera_to_world=frame.camera_to_world,
                        intrinsics=intrinsics,
                        stride=2,
                    )
                    projection = _project_structure(
                        points,
                        layout=layout,
                        intrinsics=intrinsics,
                        panel_index=panel_index,
                        panel_valid_mask=frame.panel_valid_mask,
                        minimum_sample_count=30,
                    )
                    if (
                        projection is None
                        or float(projection.in_bounds_ratio) < 0.90
                    ):
                        continue
                    proposed = replace(
                        current_owner,
                        panel_index=panel_index,
                        target_panel_index=panel_index,
                        frame_id=int(frame.frame_id),
                        source_index=int(frame.source_index),
                        source_mask=np.ascontiguousarray(source_mask),
                        target_footprint=np.ascontiguousarray(
                            projection.footprint.copy()
                        ),
                        projected_in_bounds_ratio=float(
                            projection.in_bounds_ratio
                        ),
                        measured_depth_coverage_ratio=float(
                            np.count_nonzero(
                                source_mask
                                & np.asarray(
                                    frame.reliable_depth, dtype=bool
                                )
                                & np.isfinite(frame.depth_mm)
                                & (np.asarray(frame.depth_mm) > 0.0)
                            )
                            / max(1, np.count_nonzero(source_mask))
                        ),
                    )
                    trial_intervals, trial_audit = (
                        _build_panel_native_preseam_intervals(
                            (proposed,),
                            identity_frames,
                            layout=layout,
                            intrinsics=intrinsics,
                            geometric_track_ids={track_id},
                            config=config,
                        )
                    )
                    if len(trial_intervals) != 1:
                        continue
                    trial_transfer = np.asarray(
                        trial_intervals[0].rgb_transfer_mask, dtype=bool
                    )
                    if any(
                        int(fixed.frame_id) != int(frame.frame_id)
                        and np.any(
                            trial_transfer
                            & np.asarray(
                                fixed.rgb_transfer_mask, dtype=bool
                            )
                        )
                        for fixed in corridor_intervals
                    ):
                        continue
                    rank = tuple(
                        float(value)
                        for value in observation["selection_rank"]
                    )
                    options.append(
                        (
                            rank,
                            proposed,
                            trial_intervals[0],
                            {
                                "frame_id": int(frame.frame_id),
                                "panel_index": panel_index,
                                "source_index": int(frame.source_index),
                                "projected_in_bounds_ratio": float(
                                    projection.in_bounds_ratio
                                ),
                                "measured_depth_coverage_ratio": float(
                                    proposed.measured_depth_coverage_ratio
                                ),
                                "native_interval_plan": trial_audit,
                            },
                        )
                    )
                options.sort(key=lambda item: item[0], reverse=True)
                option_domains[track_id] = options

            csp_order = sorted(
                component,
                key=lambda value: (
                    len(option_domains[value]),
                    value,
                ),
            )
            csp_assignment: dict[
                int,
                tuple[
                    tuple[float, ...],
                    InspectionForegroundIdentityOwner,
                    InspectionPreSeamHardOwnerInterval,
                    dict[str, object],
                ],
            ] = {}
            csp_solution: dict[
                int,
                tuple[
                    tuple[float, ...],
                    InspectionForegroundIdentityOwner,
                    InspectionPreSeamHardOwnerInterval,
                    dict[str, object],
                ],
            ] | None = None
            csp_budget_exhausted = False

            def options_compatible(
                first: tuple[
                    tuple[float, ...],
                    InspectionForegroundIdentityOwner,
                    InspectionPreSeamHardOwnerInterval,
                    dict[str, object],
                ],
                second: tuple[
                    tuple[float, ...],
                    InspectionForegroundIdentityOwner,
                    InspectionPreSeamHardOwnerInterval,
                    dict[str, object],
                ],
            ) -> bool:
                _, first_owner, first_interval, _ = first
                _, second_owner, second_interval, _ = second
                if int(first_owner.frame_id) == int(second_owner.frame_id):
                    return True
                return not np.any(
                    np.asarray(
                        first_interval.rgb_transfer_mask, dtype=bool
                    )
                    & np.asarray(
                        second_interval.rgb_transfer_mask, dtype=bool
                    )
                )

            def search_csp(position: int) -> bool:
                nonlocal csp_budget_exhausted
                nonlocal csp_search_state_count
                nonlocal csp_solution
                if csp_search_state_count >= csp_search_state_limit:
                    csp_budget_exhausted = True
                    return False
                if position == len(csp_order):
                    csp_solution = dict(csp_assignment)
                    return True
                track_id = csp_order[position]
                for option in option_domains[track_id]:
                    csp_search_state_count += 1
                    if any(
                        not options_compatible(option, other_option)
                        for other_option in csp_assignment.values()
                    ):
                        continue
                    csp_assignment[track_id] = option
                    forward_valid = True
                    for future_track_id in csp_order[position + 1 :]:
                        if not any(
                            all(
                                options_compatible(
                                    future_option, assigned_option
                                )
                                for assigned_option
                                in csp_assignment.values()
                            )
                            for future_option
                            in option_domains[future_track_id]
                        ):
                            forward_valid = False
                            break
                    if not forward_valid:
                        del csp_assignment[track_id]
                        continue
                    if search_csp(position + 1):
                        return True
                    del csp_assignment[track_id]
                return False

            search_csp(0)
            if csp_solution is None and csp_budget_exhausted:
                raise RuntimeError(
                    "Shelf native owner global CSP search exhausted its "
                    "hard state budget before proving a solution or UNSAT"
                )
            if csp_solution is not None:
                original_frame_ids = sorted(
                    {
                        int(owner_by_track[track_id].frame_id)
                        for track_id in component
                    }
                )
                selected_rows: list[dict[str, object]] = []
                for track_id in component:
                    _, proposed, _, option_audit = csp_solution[track_id]
                    owner_by_track[track_id] = proposed
                    selected_rows.append(
                        {
                            "track_id": int(track_id),
                            **option_audit,
                        }
                    )
                resolution_events.append(
                    {
                        "iteration": int(iteration),
                        "resolution_level": (
                            "level_1_pairwise_native_footprint_csp"
                        ),
                        "member_track_ids": list(component),
                        "original_frame_ids": original_frame_ids,
                        "candidate_frames_by_track": {
                            str(track_id): [
                                int(option[1].frame_id)
                                for option in option_domains[track_id]
                            ]
                            for track_id in component
                        },
                        "search_variable_order": csp_order,
                        "search_state_count_cumulative": (
                            csp_search_state_count
                        ),
                        "search_state_limit": csp_search_state_limit,
                        "constraint": (
                            "only_recomputed_final_transfer_overlap_requires_"
                            "the_same_real_rgb_frame"
                        ),
                        "transitive_closure_constraint_used": False,
                        "selected_assignments": selected_rows,
                        "all_members_handled": True,
                        "sequential_override_used": False,
                        "rgb_blended_or_generated": False,
                    }
                )
                continue
            pairwise_unsat: list[tuple[int, int]] = []
            for first_position, first_track_id in enumerate(component):
                for second_track_id in component[first_position + 1 :]:
                    if not option_domains[first_track_id] or not option_domains[
                        second_track_id
                    ]:
                        continue
                    if not any(
                        options_compatible(first_option, second_option)
                        for first_option in option_domains[first_track_id]
                        for second_option in option_domains[second_track_id]
                    ):
                        pairwise_unsat.append(
                            (first_track_id, second_track_id)
                        )
            composite_candidates: list[
                tuple[
                    tuple[float, ...],
                    tuple[int, int],
                    InspectionPreSeamHardOwnerInterval,
                    dict[str, object],
                ]
            ] = []
            current_interval_by_track = {
                interval_track_id(interval): interval
                for interval in final_intervals
                if interval_track_id(interval) in owner_by_track
            }

            def reserved_foreign_support(
                member_track_ids: set[int],
            ) -> np.ndarray:
                masks = [
                    np.asarray(interval.rgb_transfer_mask, dtype=bool)
                    for track_id, interval
                    in current_interval_by_track.items()
                    if track_id not in member_track_ids
                ]
                if not masks:
                    return np.zeros(
                        (layout.height, layout.width), dtype=bool
                    )
                return np.ascontiguousarray(np.logical_or.reduce(masks))

            coobserved_corridor_candidates: list[
                tuple[
                    tuple[float, ...],
                    tuple[int, int],
                    InspectionPreSeamHardOwnerInterval,
                    dict[str, object],
                ]
            ] = []
            existing_corridor_frame_ids = {
                int(interval.frame_id) for interval in corridor_intervals
            }
            for first_track_id, second_track_id in pairwise_unsat:
                common_complete_frame_ids = sorted(
                    set(eligible_frames_by_track[first_track_id])
                    & set(eligible_frames_by_track[second_track_id])
                )
                for frame_id in common_complete_frame_ids:
                    frame = frame_by_id.get(frame_id)
                    if frame is None:
                        continue
                    panel_index = int(frame.panel_index)
                    source_masks: list[np.ndarray] = []
                    member_rows: list[dict[str, object]] = []
                    ranks: list[tuple[float, ...]] = []
                    for track_id in (first_track_id, second_track_id):
                        mask_by_panel = {
                            int(value_panel): np.asarray(
                                value_mask, dtype=bool
                            )
                            for value_panel, value_mask
                            in owner_by_track[
                                track_id
                            ].reference_observation_masks
                        }
                        source_mask = mask_by_panel.get(panel_index)
                        if source_mask is None or not np.any(source_mask):
                            source_masks = []
                            break
                        observation = eligible_frames_by_track[track_id][
                            frame_id
                        ]
                        source_masks.append(source_mask)
                        ranks.append(
                            tuple(
                                float(value)
                                for value
                                in observation["selection_rank"]
                            )
                        )
                        member_rows.append(
                            {
                                "track_id": int(track_id),
                                "source_mask_pixel_count": int(
                                    np.count_nonzero(source_mask)
                                ),
                                "complete_observation_gate_passed": True,
                            }
                        )
                    if len(source_masks) != 2:
                        continue
                    source_union = source_masks[0] | source_masks[1]
                    member_native_owners = tuple(
                        replace(
                            owner_by_track[track_id],
                            panel_index=panel_index,
                            target_panel_index=panel_index,
                            frame_id=int(frame.frame_id),
                            source_index=int(frame.source_index),
                            source_mask=np.ascontiguousarray(source_mask),
                        )
                        for track_id, source_mask in zip(
                            (first_track_id, second_track_id),
                            source_masks,
                            strict=True,
                        )
                    )
                    (
                        member_native_intervals,
                        member_native_audit,
                    ) = _build_panel_native_preseam_intervals(
                        member_native_owners,
                        identity_frames,
                        layout=layout,
                        intrinsics=intrinsics,
                        geometric_track_ids={
                            first_track_id,
                            second_track_id,
                        },
                        config=config,
                    )
                    if len(member_native_intervals) != 2:
                        continue
                    exact_member_transfer = np.logical_or.reduce(
                        [
                            np.asarray(
                                member_interval.rgb_transfer_mask,
                                dtype=bool,
                            )
                            for member_interval
                            in member_native_intervals
                        ]
                    )
                    yy, xx = np.nonzero(source_union)
                    margin = int(config.object_rich_lock_guard_pixels)
                    x0 = int(xx.min()) - margin
                    x1 = int(xx.max()) + margin + 1
                    y0 = int(yy.min()) - margin
                    y1 = int(yy.max()) + margin + 1
                    if (
                        x0 < 0
                        or y0 < 0
                        or x1 > intrinsics.width
                        or y1 > intrinsics.height
                    ):
                        continue
                    source_corridor = np.zeros_like(source_union)
                    source_corridor[y0:y1, x0:x1] = True
                    payload = (
                        f"coobserved:{first_track_id},{second_track_id}"
                    ).encode("ascii")
                    group_track_id = (
                        500_000_000
                        + int.from_bytes(
                            hashlib.blake2s(
                                payload, digest_size=4
                            ).digest(),
                            "little",
                        )
                        % 100_000_000
                    )
                    seed = owner_by_track[first_track_id]
                    corridor_owner = replace(
                        seed,
                        structure_id=group_track_id,
                        structure_kind=(
                            "middle_shelf_coobserved_object_rich_corridor"
                        ),
                        identity_track_id=group_track_id,
                        panel_index=panel_index,
                        target_panel_index=panel_index,
                        frame_id=int(frame.frame_id),
                        source_index=int(frame.source_index),
                        source_mask=np.ascontiguousarray(source_corridor),
                        reference_observation_masks=(
                            (
                                panel_index,
                                np.ascontiguousarray(source_corridor),
                            ),
                        ),
                    )
                    trial_intervals, trial_audit = (
                        _build_panel_native_preseam_intervals(
                            (corridor_owner,),
                            identity_frames,
                            layout=layout,
                            intrinsics=intrinsics,
                            geometric_track_ids={group_track_id},
                            config=config,
                        )
                    )
                    if len(trial_intervals) != 1:
                        continue
                    interval = trial_intervals[0]
                    corridor_competition_mask = np.asarray(
                        interval.rgb_transfer_mask, dtype=bool
                    )
                    if any(
                        int(fixed.frame_id) != int(frame.frame_id)
                        and np.any(
                            corridor_competition_mask
                            & np.asarray(
                                fixed.rgb_transfer_mask, dtype=bool
                            )
                        )
                        for fixed in corridor_intervals
                    ):
                        continue
                    foreign_rows: list[dict[str, object]] = []
                    foreign_compatible = True
                    for other_track_id in component:
                        if other_track_id in {
                            first_track_id,
                            second_track_id,
                        }:
                            continue
                        compatible_count = sum(
                            int(
                                int(option[1].frame_id)
                                == int(frame.frame_id)
                                or not np.any(
                                    corridor_competition_mask
                                    & np.asarray(
                                        option[2].rgb_transfer_mask,
                                        dtype=bool,
                                    )
                                )
                            )
                            for option in option_domains[other_track_id]
                        )
                        if compatible_count == 0:
                            foreign_compatible = False
                            break
                        foreign_rows.append(
                            {
                                "track_id": int(other_track_id),
                                "compatible_candidate_count": (
                                    compatible_count
                                ),
                            }
                        )
                    if not foreign_compatible:
                        continue
                    rgb_transfer, transfer_audit = (
                        _bounded_exact_corridor_transfer(
                            exact_member_transfer,
                            interval.lock_mask,
                            frame.panel_valid_mask,
                            dilation_pixels=0,
                            reserved_foreign_support=(
                                reserved_foreign_support(
                                    {first_track_id, second_track_id}
                                )
                            ),
                        )
                    )
                    if rgb_transfer is None:
                        continue
                    interval = replace(
                        interval,
                        union_footprint=np.ascontiguousarray(
                            exact_member_transfer
                        ),
                        rgb_transfer_mask=rgb_transfer,
                        owner_only_mask=np.ascontiguousarray(
                            interval.lock_mask
                        ),
                        rgb_context_member_supports=tuple(
                            np.asarray(
                                member_interval.rgb_transfer_mask,
                                dtype=bool,
                            )
                            for member_interval in member_native_intervals
                        ),
                    )
                    score = (
                        float(
                            int(
                                int(frame.frame_id)
                                in existing_corridor_frame_ids
                            )
                        ),
                        *min(ranks),
                        -float(np.count_nonzero(source_corridor)),
                        -float(panel_index),
                    )
                    coobserved_corridor_candidates.append(
                        (
                            score,
                            (first_track_id, second_track_id),
                            interval,
                            {
                                "resolution_level": (
                                    "level_4a_coobserved_distinct_object_"
                                    "corridor"
                                ),
                                "member_track_ids": [
                                    first_track_id,
                                    second_track_id,
                                ],
                                "distinct_objects_preserved": True,
                                "common_complete_frame_ids": (
                                    common_complete_frame_ids
                                ),
                                "selected_frame_id": int(frame.frame_id),
                                "selected_panel_index": panel_index,
                                "selected_source_index": int(
                                    frame.source_index
                                ),
                                "preferred_existing_corridor_frame": bool(
                                    int(frame.frame_id)
                                    in existing_corridor_frame_ids
                                ),
                                "source_union_pixel_count": int(
                                    np.count_nonzero(source_union)
                                ),
                                "source_corridor_bbox_xyxy": [
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                ],
                                "source_corridor_margin_pixels": margin,
                                "corridor_transfer_pixel_count": int(
                                    np.count_nonzero(rgb_transfer)
                                ),
                                "corridor_owner_guard_pixel_count": int(
                                    np.count_nonzero(interval.lock_mask)
                                ),
                                "corridor_transfer_audit": transfer_audit,
                                "member_native_interval_plan": (
                                    member_native_audit
                                ),
                                "member_observations": member_rows,
                                "full_panel_valid_inverse_map_coverage": True,
                                "foreign_required_constraints": foreign_rows,
                                "all_members_handled": True,
                                "mesh_used": False,
                                "graphcut_multiband_flow_allowed_inside": (
                                    False
                                ),
                                "rgb_blended_or_generated": False,
                                "native_interval_plan": trial_audit,
                            },
                        )
                    )
            if coobserved_corridor_candidates:
                coobserved_corridor_candidates.sort(
                    key=lambda item: item[0], reverse=True
                )
                _, members, interval, corridor_audit = (
                    coobserved_corridor_candidates[0]
                )
                corridor_intervals.append(interval)
                corridor_handled_track_ids.update(members)
                resolution_events.append(
                    {
                        "iteration": int(iteration),
                        **corridor_audit,
                    }
                )
                continue
            global_unsat_cycle_break_pairs: list[tuple[int, int]] = []
            for first_position, first_track_id in enumerate(component):
                for second_track_id in component[first_position + 1 :]:
                    first_interval = current_interval_by_track.get(
                        first_track_id
                    )
                    second_interval = current_interval_by_track.get(
                        second_track_id
                    )
                    if first_interval is None or second_interval is None:
                        continue
                    first_transfer = np.asarray(
                        first_interval.rgb_transfer_mask, dtype=bool
                    )
                    second_transfer = np.asarray(
                        second_interval.rgb_transfer_mask, dtype=bool
                    )
                    overlap_pixels = int(
                        np.count_nonzero(first_transfer & second_transfer)
                    )
                    if overlap_pixels <= 0:
                        continue
                    smaller_pixels = min(
                        int(np.count_nonzero(first_transfer)),
                        int(np.count_nonzero(second_transfer)),
                    )
                    if (
                        float(overlap_pixels / max(1, smaller_pixels))
                        < 0.02
                    ):
                        continue
                    common_observed_panels = {
                        int(value[0])
                        for value in owner_by_track[
                            first_track_id
                        ].reference_observation_masks
                    } & {
                        int(value[0])
                        for value in owner_by_track[
                            second_track_id
                        ].reference_observation_masks
                    }
                    if common_observed_panels:
                        continue
                    global_unsat_cycle_break_pairs.append(
                        (first_track_id, second_track_id)
                    )
            fragment_candidate_pairs = sorted(
                set(pairwise_unsat)
                | set(global_unsat_cycle_break_pairs)
            )
            for first_track_id, second_track_id in fragment_candidate_pairs:
                first_interval = current_interval_by_track[first_track_id]
                second_interval = current_interval_by_track[second_track_id]
                first_transfer = np.asarray(
                    first_interval.rgb_transfer_mask, dtype=bool
                )
                second_transfer = np.asarray(
                    second_interval.rgb_transfer_mask, dtype=bool
                )
                overlap_pixels = int(
                    np.count_nonzero(first_transfer & second_transfer)
                )
                if overlap_pixels <= 0:
                    continue
                smaller_pixels = min(
                    int(np.count_nonzero(first_transfer)),
                    int(np.count_nonzero(second_transfer)),
                )
                canonical_overlap_ratio = float(
                    overlap_pixels / max(1, smaller_pixels)
                )
                if canonical_overlap_ratio < 0.02:
                    continue
                common_observed_panels = {
                    int(value[0])
                    for value in owner_by_track[
                        first_track_id
                    ].reference_observation_masks
                } & {
                    int(value[0])
                    for value in owner_by_track[
                        second_track_id
                    ].reference_observation_masks
                }
                if common_observed_panels:
                    # Co-observed masks must remain distinct unless the normal
                    # hierarchy gate already suppressed one in the planner.
                    continue
                union_target = first_transfer | second_transfer
                corridor = _row_contiguous_guard(
                    union_target, guard_pixels=2
                )
                component_count, _ = cv2.connectedComponents(
                    corridor.astype(np.uint8), 8
                )
                if component_count != 2:
                    continue
                group_payload = (
                    f"{first_track_id},{second_track_id}"
                ).encode("ascii")
                group_track_id = (
                    600_000_000
                    + (
                        int.from_bytes(
                            hashlib.blake2s(
                                group_payload, digest_size=4
                            ).digest(),
                            "little",
                        )
                        % 100_000_000
                    )
                )
                member_options = (
                    *option_domains[first_track_id],
                    *option_domains[second_track_id],
                )
                for rank, proposed, _, _ in member_options:
                    frame = frame_by_id[int(proposed.frame_id)]
                    if np.any(
                        corridor
                        & ~np.asarray(frame.panel_valid_mask, dtype=bool)
                    ):
                        continue
                    foreign_rows: list[dict[str, object]] = []
                    foreign_compatible = True
                    for other_track_id in component:
                        if other_track_id in {
                            first_track_id,
                            second_track_id,
                        }:
                            continue
                        overlapping_options = [
                            option
                            for option in option_domains[other_track_id]
                            if np.any(
                                corridor
                                & np.asarray(
                                    option[2].rgb_transfer_mask,
                                    dtype=bool,
                                )
                            )
                        ]
                        if not overlapping_options:
                            continue
                        compatible_options = [
                            option
                            for option in option_domains[other_track_id]
                            if (
                                not np.any(
                                    corridor
                                    & np.asarray(
                                        option[2].rgb_transfer_mask,
                                        dtype=bool,
                                    )
                                )
                                or int(option[1].frame_id)
                                == int(proposed.frame_id)
                            )
                        ]
                        if not compatible_options:
                            foreign_compatible = False
                            break
                        foreign_rows.append(
                            {
                                "track_id": int(other_track_id),
                                "overlapping_candidate_count": len(
                                    overlapping_options
                                ),
                                "compatible_candidate_count": len(
                                    compatible_options
                                ),
                            }
                        )
                    if not foreign_compatible:
                        continue
                    rgb_transfer, transfer_audit = (
                        _bounded_exact_corridor_transfer(
                            union_target,
                            corridor,
                            frame.panel_valid_mask,
                            dilation_pixels=0,
                            reserved_foreign_support=(
                                reserved_foreign_support(
                                    {first_track_id, second_track_id}
                                )
                            ),
                        )
                    )
                    if rgb_transfer is None:
                        continue
                    interval = InspectionPreSeamHardOwnerInterval(
                        track_id=group_track_id,
                        panel_index=int(proposed.panel_index),
                        frame_id=int(proposed.frame_id),
                        lock_mask=np.ascontiguousarray(corridor),
                        union_footprint=np.ascontiguousarray(union_target),
                        rgb_source_panel_index=int(proposed.panel_index),
                        rgb_transfer_mask=rgb_transfer,
                        owner_only_mask=np.ascontiguousarray(corridor),
                        rgb_context_member_supports=(
                            np.ascontiguousarray(first_transfer),
                            np.ascontiguousarray(second_transfer),
                        ),
                        background_panel_lock_required=False,
                    )
                    score = (
                        *rank,
                        -float(np.count_nonzero(corridor)),
                        -float(proposed.panel_index),
                    )
                    composite_candidates.append(
                        (
                            score,
                            (first_track_id, second_track_id),
                            interval,
                            {
                                "resolution_level": (
                                    "level_3_canonical_fragment_composite_"
                                    "corridor"
                                ),
                                "member_track_ids": [
                                    first_track_id,
                                    second_track_id,
                                ],
                                "suppression_semantics": (
                                    "tracks_alias_to_one_composite_entity_"
                                    "without_deleting_either_measured_support"
                                ),
                                "candidate_frames_by_track": {
                                    str(first_track_id): [
                                        int(option[1].frame_id)
                                        for option
                                        in option_domains[first_track_id]
                                    ],
                                    str(second_track_id): [
                                        int(option[1].frame_id)
                                        for option
                                        in option_domains[second_track_id]
                                    ],
                                },
                                "no_common_observed_panel": True,
                                "trigger": (
                                    "global_csp_unsat_cycle_break"
                                    if (
                                        first_track_id,
                                        second_track_id,
                                    )
                                    in global_unsat_cycle_break_pairs
                                    else "pairwise_csp_proven_unsat"
                                ),
                                "pairwise_csp_proven_unsat": bool(
                                    (
                                        first_track_id,
                                        second_track_id,
                                    )
                                    in pairwise_unsat
                                ),
                                "canonical_overlap_pixel_count": (
                                    overlap_pixels
                                ),
                                "canonical_smaller_support_overlap_ratio": (
                                    canonical_overlap_ratio
                                ),
                                "selected_frame_id": int(
                                    proposed.frame_id
                                ),
                                "selected_panel_index": int(
                                    proposed.panel_index
                                ),
                                "selected_source_index": int(
                                    proposed.source_index
                                ),
                                "corridor_margin_pixels": 2,
                                "corridor_transfer_pixel_count": int(
                                    np.count_nonzero(rgb_transfer)
                                ),
                                "corridor_owner_guard_pixel_count": int(
                                    np.count_nonzero(corridor)
                                ),
                                "corridor_transfer_audit": transfer_audit,
                                "full_selected_panel_valid_coverage": True,
                                "foreign_required_constraints": foreign_rows,
                                "all_member_measured_support_retained": True,
                                "all_members_handled": True,
                                "mesh_used": False,
                                "cross_panel_warp_used": False,
                                "graphcut_multiband_flow_allowed_inside": (
                                    False
                                ),
                                "rgb_blended_or_generated": False,
                            },
                        )
                    )
            if composite_candidates:
                composite_candidates.sort(
                    key=lambda item: item[0], reverse=True
                )
                _, members, interval, composite_audit = (
                    composite_candidates[0]
                )
                corridor_intervals.append(interval)
                corridor_handled_track_ids.update(members)
                resolution_events.append(
                    {
                        "iteration": int(iteration),
                        **composite_audit,
                    }
                )
                continue

            def subset_has_solution(
                track_ids: Sequence[int],
            ) -> bool:
                nonlocal mus_search_state_count
                order = sorted(
                    track_ids,
                    key=lambda value: (
                        len(option_domains[value]),
                        value,
                    ),
                )
                assigned: list[
                    tuple[
                        tuple[float, ...],
                        InspectionForegroundIdentityOwner,
                        InspectionPreSeamHardOwnerInterval,
                        dict[str, object],
                    ]
                ] = []

                def visit(position: int) -> bool:
                    nonlocal mus_search_state_count
                    if mus_search_state_count >= mus_search_state_limit:
                        raise RuntimeError(
                            "Shelf minimal UNSAT core derivation exhausted "
                            "its hard search-state budget"
                        )
                    if position == len(order):
                        return True
                    for option in option_domains[order[position]]:
                        mus_search_state_count += 1
                        if any(
                            not options_compatible(option, other)
                            for other in assigned
                        ):
                            continue
                        assigned.append(option)
                        if visit(position + 1):
                            return True
                        assigned.pop()
                    return False

                return visit(0)

            core = list(component)
            mus_corridor_rejection_rows: list[dict[str, object]] = []
            mus_corridor_geometry_context: dict[str, object] = {}
            core_derivation_rows: list[dict[str, object]] = []
            changed = True
            while changed and len(core) > 2:
                changed = False
                for track_id in tuple(core):
                    trial_core = [
                        value for value in core if value != track_id
                    ]
                    trial_satisfiable = subset_has_solution(trial_core)
                    core_derivation_rows.append(
                        {
                            "removed_track_id": int(track_id),
                            "remaining_track_ids": list(trial_core),
                            "remaining_satisfiable": trial_satisfiable,
                        }
                    )
                    if not trial_satisfiable:
                        core = trial_core
                        changed = True
                        break
            core_is_unsat = not subset_has_solution(core)
            mus_corridor_candidates: list[
                tuple[
                    tuple[float, ...],
                    tuple[int, ...],
                    InspectionPreSeamHardOwnerInterval,
                    dict[str, object],
                ]
            ] = []
            if core_is_unsat and len(core) >= 2:
                target_union = np.logical_or.reduce(
                    [
                        np.asarray(
                            current_interval_by_track[
                                track_id
                            ].rgb_transfer_mask,
                            dtype=bool,
                        )
                        for track_id in core
                    ]
                )
                target_corridor = _row_contiguous_guard(
                    target_union,
                    guard_pixels=int(
                        config.object_rich_lock_guard_pixels
                    ),
                )
                yy, xx = np.nonzero(target_corridor)
                corridor_width = int(xx.max() - xx.min() + 1)
                corridor_area = int(np.count_nonzero(target_corridor))
                target_corridor_bbox_xyxy = [
                    int(xx.min()),
                    int(yy.min()),
                    int(xx.max()) + 1,
                    int(yy.max()) + 1,
                ]
                maximum_corridor_width = int(intrinsics.width)
                maximum_corridor_area = int(
                    round(
                        0.35
                        * float(intrinsics.width)
                        * float(intrinsics.height)
                    )
                )
                mus_corridor_geometry_context = {
                    "target_corridor_bbox_xyxy": (
                        target_corridor_bbox_xyxy
                    ),
                    "target_corridor_width_pixels": corridor_width,
                    "target_corridor_area_pixels": corridor_area,
                    "maximum_corridor_width_pixels": (
                        maximum_corridor_width
                    ),
                    "maximum_corridor_area_pixels": maximum_corridor_area,
                }
                bounded = bool(
                    corridor_width <= maximum_corridor_width
                    and corridor_area <= maximum_corridor_area
                )
                if not bounded:
                    mus_corridor_rejection_rows.extend(
                        {
                            "frame_id": int(frame.frame_id),
                            "panel_index": int(frame.panel_index),
                            "outside_panel_valid_pixel_count": None,
                            "fixed_overlap_count": 0,
                            "fixed_overlap_pixel_count": 0,
                            "foreign_blocker_track_ids": [],
                            "closure_additions": [],
                            "veto_reason": (
                                "initial_corridor_resource_bounds_failed"
                            ),
                        }
                        for frame in identity_frames
                    )
                if bounded:
                    payload = (
                        "mus:" + ",".join(str(value) for value in core)
                    ).encode("ascii")
                    group_track_id = (
                        400_000_000
                        + int.from_bytes(
                            hashlib.blake2s(
                                payload, digest_size=4
                            ).digest(),
                            "little",
                        )
                        % 100_000_000
                    )
                    for frame in identity_frames:
                        frame_rejection: dict[str, object] = {
                            "frame_id": int(frame.frame_id),
                            "panel_index": int(frame.panel_index),
                            "outside_panel_valid_pixel_count": 0,
                            "fixed_overlap_count": 0,
                            "fixed_overlap_pixel_count": 0,
                            "foreign_blocker_track_ids": [],
                            "closure_additions": [],
                            "veto_reason": None,
                        }
                        closure_track_ids = set(core)
                        closure_union = np.ascontiguousarray(
                            target_union.copy()
                        )
                        closure_additions: list[dict[str, object]] = []
                        foreign_rows: list[dict[str, object]] = []
                        fixed_boundary_trim_rows: list[
                            dict[str, object]
                        ] = []
                        closure_valid = False
                        maximum_closure_members = min(
                            16, len(component)
                        )
                        maximum_closure_iterations = min(
                            16, len(component)
                        )
                        target_corridor = np.zeros_like(
                            closure_union, dtype=bool
                        )
                        for closure_iteration in range(
                            maximum_closure_iterations
                        ):
                            target_corridor = _row_contiguous_guard(
                                closure_union,
                                guard_pixels=int(
                                    config.object_rich_lock_guard_pixels
                                ),
                            )
                            (
                                target_corridor,
                                fixed_boundary_trim_valid,
                                fixed_boundary_trim_rows,
                            ) = _trim_fixed_corridor_boundary_overlap(
                                target_corridor,
                                closure_union,
                                candidate_frame_id=int(frame.frame_id),
                                fixed_intervals=corridor_intervals,
                            )
                            frame_rejection["fixed_overlap_count"] = len(
                                fixed_boundary_trim_rows
                            )
                            frame_rejection[
                                "fixed_overlap_pixel_count"
                            ] = sum(
                                int(
                                    row[
                                        "requested_overlap_pixel_count"
                                    ]
                                )
                                for row in fixed_boundary_trim_rows
                            )
                            frame_rejection["fixed_overlap_details"] = [
                                {
                                    "fixed_track_id": int(
                                        row["fixed_track_id"]
                                    ),
                                    "fixed_frame_id": int(
                                        row["fixed_frame_id"]
                                    ),
                                    "requested_overlap_pixel_count": int(
                                        row[
                                            "requested_overlap_pixel_count"
                                        ]
                                    ),
                                    "measured_support_overlap_pixel_count": (
                                        int(
                                            row[
                                                "measured_support_overlap_"
                                                "pixel_count"
                                            ]
                                        )
                                    ),
                                    "measured_support_overlap_ratio": float(
                                        row[
                                            "measured_support_overlap_ratio"
                                        ]
                                    ),
                                    "measured_overlap_in_fixed_exact_support_"
                                    "pixel_count": int(
                                        row[
                                            "measured_overlap_in_fixed_exact_"
                                            "support_pixel_count"
                                        ]
                                    ),
                                    "cross_track_boundary_alias_partition": (
                                        bool(
                                            row[
                                                "cross_track_boundary_alias_"
                                                "partition"
                                            ]
                                        )
                                    ),
                                    "all_member_supports_retained": bool(
                                        row[
                                            "all_member_supports_retained"
                                        ]
                                    ),
                                    "per_row_boundary_trim_passed": bool(
                                        row[
                                            "per_row_boundary_trim_passed"
                                        ]
                                    ),
                                }
                                for row in fixed_boundary_trim_rows
                            ]
                            if not fixed_boundary_trim_valid:
                                frame_rejection["veto_reason"] = (
                                    "fixed_corridor_overlap_not_safe_to_trim"
                                )
                                break
                            closure_component_count, _ = (
                                cv2.connectedComponents(
                                    target_corridor.astype(np.uint8), 8
                                )
                            )
                            yy, xx = np.nonzero(target_corridor)
                            corridor_width = int(
                                xx.max() - xx.min() + 1
                            )
                            corridor_area = int(
                                np.count_nonzero(target_corridor)
                            )
                            outside_panel_valid_pixel_count = int(
                                np.count_nonzero(
                                    closure_union
                                    & ~np.asarray(
                                        frame.panel_valid_mask,
                                        dtype=bool,
                                    )
                                )
                            )
                            frame_rejection[
                                "outside_panel_valid_pixel_count"
                            ] = outside_panel_valid_pixel_count
                            if (
                                corridor_width
                                > maximum_corridor_width
                                or corridor_area
                                > maximum_corridor_area
                                or outside_panel_valid_pixel_count > 0
                            ):
                                frame_rejection["veto_reason"] = (
                                    "corridor_resource_bounds_failed"
                                    if (
                                        corridor_width
                                        > maximum_corridor_width
                                        or corridor_area
                                        > maximum_corridor_area
                                    )
                                    else "outside_selected_panel_valid_mask"
                                )
                                break
                            blockers: list[int] = []
                            foreign_rows = []
                            for other_track_id in component:
                                if other_track_id in closure_track_ids:
                                    continue
                                compatible_count = sum(
                                    int(
                                        int(option[1].frame_id)
                                        == int(frame.frame_id)
                                        or not np.any(
                                            closure_union
                                            & np.asarray(
                                                option[
                                                    2
                                                ].rgb_transfer_mask,
                                                dtype=bool,
                                            )
                                        )
                                    )
                                    for option
                                    in option_domains[other_track_id]
                                )
                                foreign_rows.append(
                                    {
                                        "track_id": int(other_track_id),
                                        "compatible_candidate_count": (
                                            compatible_count
                                        ),
                                    }
                                )
                                if compatible_count == 0:
                                    blockers.append(other_track_id)
                            frame_rejection[
                                "foreign_blocker_track_ids"
                            ] = [int(value) for value in blockers]
                            if not blockers:
                                closure_valid = True
                                break
                            if (
                                len(closure_track_ids)
                                + len(blockers)
                                > maximum_closure_members
                            ):
                                frame_rejection["veto_reason"] = (
                                    "closure_member_limit_exceeded"
                                )
                                break
                            additions_valid = True
                            for blocker_track_id in blockers:
                                support = np.asarray(
                                    current_interval_by_track[
                                        blocker_track_id
                                    ].rgb_transfer_mask,
                                    dtype=bool,
                                )
                                if (
                                    not np.any(support)
                                    or np.any(
                                        support
                                        & ~np.asarray(
                                            frame.panel_valid_mask,
                                            dtype=bool,
                                        )
                                    )
                                ):
                                    additions_valid = False
                                    frame_rejection["veto_reason"] = (
                                        "foreign_blocker_support_not_fully_"
                                        "selected_panel_valid"
                                    )
                                    break
                                closure_union |= support
                                closure_track_ids.add(
                                    blocker_track_id
                                )
                                closure_additions.append(
                                    {
                                        "closure_iteration": int(
                                            closure_iteration
                                        ),
                                        "added_track_id": int(
                                            blocker_track_id
                                        ),
                                        "reason": (
                                            "no_compatible_candidate_outside_"
                                            "same_real_rgb_corridor"
                                        ),
                                        "measured_target_support_pixel_count": (
                                            int(np.count_nonzero(support))
                                        ),
                                        "support_fully_panel_valid": True,
                                    }
                                )
                            if not additions_valid:
                                break
                            frame_rejection["closure_additions"] = [
                                {
                                    "track_id": int(
                                        row["added_track_id"]
                                    ),
                                    "support_pixel_count": int(
                                        row[
                                            "measured_target_support_"
                                            "pixel_count"
                                        ]
                                    ),
                                }
                                for row in closure_additions
                            ]
                        if not closure_valid:
                            if frame_rejection["veto_reason"] is None:
                                frame_rejection["veto_reason"] = (
                                    "closure_iteration_limit_exhausted"
                                )
                            frame_rejection["closure_additions"] = [
                                {
                                    "track_id": int(
                                        row["added_track_id"]
                                    ),
                                    "support_pixel_count": int(
                                        row[
                                            "measured_target_support_"
                                            "pixel_count"
                                        ]
                                    ),
                                }
                                for row in closure_additions
                            ]
                            mus_corridor_rejection_rows.append(
                                frame_rejection
                            )
                            continue
                        delegated_closure_support = np.zeros_like(
                            closure_union, dtype=bool
                        )
                        fixed_by_track_id = {
                            int(fixed.track_id): fixed
                            for fixed in corridor_intervals
                        }
                        for row in fixed_boundary_trim_rows:
                            if not bool(
                                row.get(
                                    "cross_track_boundary_alias_partition",
                                    False,
                                )
                            ):
                                continue
                            fixed = fixed_by_track_id.get(
                                int(row["fixed_track_id"])
                            )
                            if fixed is None:
                                continue
                            delegated_closure_support |= (
                                closure_union
                                & np.asarray(
                                    fixed.union_footprint, dtype=bool
                                )
                                & np.asarray(
                                    fixed.rgb_transfer_mask, dtype=bool
                                )
                            )
                        delegated_closure_support = np.ascontiguousarray(
                            delegated_closure_support
                        )
                        effective_closure_support = np.ascontiguousarray(
                            closure_union & ~delegated_closure_support
                        )
                        delegated_support_pixels = int(
                            np.count_nonzero(delegated_closure_support)
                        )
                        delegated_rows = [
                            row
                            for row in fixed_boundary_trim_rows
                            if int(
                                row[
                                    "measured_support_overlap_pixel_count"
                                ]
                            )
                            > 0
                        ]
                        support_partition_complete = bool(
                            not np.any(
                                effective_closure_support
                                & delegated_closure_support
                            )
                            and np.array_equal(
                                effective_closure_support
                                | delegated_closure_support,
                                closure_union,
                            )
                        )
                        delegated_transfer_frame = np.full(
                            closure_union.shape, -1, dtype=np.int32
                        )
                        delegated_exact_frame = np.full(
                            closure_union.shape, -1, dtype=np.int32
                        )
                        delegated_transfer_frame_conflict = False
                        delegated_exact_frame_conflict = False
                        for fixed in corridor_intervals:
                            fixed_frame_id = int(fixed.frame_id)
                            fixed_transfer_coverage = (
                                delegated_closure_support
                                & np.asarray(
                                    fixed.rgb_transfer_mask, dtype=bool
                                )
                            )
                            fixed_exact_coverage = (
                                delegated_closure_support
                                & np.asarray(
                                    fixed.union_footprint, dtype=bool
                                )
                            )
                            delegated_transfer_frame_conflict = bool(
                                delegated_transfer_frame_conflict
                                or np.any(
                                    fixed_transfer_coverage
                                    & (delegated_transfer_frame >= 0)
                                    & (
                                        delegated_transfer_frame
                                        != fixed_frame_id
                                    )
                                )
                            )
                            delegated_exact_frame_conflict = bool(
                                delegated_exact_frame_conflict
                                or np.any(
                                    fixed_exact_coverage
                                    & (delegated_exact_frame >= 0)
                                    & (
                                        delegated_exact_frame
                                        != fixed_frame_id
                                    )
                                )
                            )
                            delegated_transfer_frame[
                                fixed_transfer_coverage
                                & (delegated_transfer_frame < 0)
                            ] = fixed_frame_id
                            delegated_exact_frame[
                                fixed_exact_coverage
                                & (delegated_exact_frame < 0)
                            ] = fixed_frame_id
                        delegated_single_owner_covered = bool(
                            delegated_support_pixels == 0
                            or (
                                bool(delegated_rows)
                                and all(
                                    bool(
                                        row.get(
                                            "cross_track_boundary_alias_"
                                            "partition",
                                            False,
                                        )
                                    )
                                    for row in delegated_rows
                                )
                                and not delegated_transfer_frame_conflict
                                and not delegated_exact_frame_conflict
                                and np.all(
                                    delegated_transfer_frame[
                                        delegated_closure_support
                                    ]
                                    >= 0
                                )
                                and np.array_equal(
                                    delegated_transfer_frame[
                                        delegated_closure_support
                                    ],
                                    delegated_exact_frame[
                                        delegated_closure_support
                                    ],
                                )
                            )
                        )
                        member_partition_rows = {}
                        for track_id in closure_track_ids:
                            member_support = np.asarray(
                                current_interval_by_track[
                                    track_id
                                ].rgb_transfer_mask,
                                dtype=bool,
                            )
                            effective_pixels = int(
                                np.count_nonzero(
                                    member_support
                                    & effective_closure_support
                                )
                            )
                            delegated_pixels = int(
                                np.count_nonzero(
                                    member_support
                                    & delegated_closure_support
                                )
                            )
                            total_pixels = int(
                                np.count_nonzero(member_support)
                            )
                            member_partition_rows[str(track_id)] = {
                                "original_support_pixel_count": total_pixels,
                                "effective_support_pixel_count": (
                                    effective_pixels
                                ),
                                "delegated_support_pixel_count": (
                                    delegated_pixels
                                ),
                                "partition_complete": bool(
                                    effective_pixels + delegated_pixels
                                    == total_pixels
                                ),
                            }
                        all_member_partitions_complete = all(
                            bool(row["partition_complete"])
                            for row in member_partition_rows.values()
                        )
                        if not (
                            support_partition_complete
                            and delegated_single_owner_covered
                            and all_member_partitions_complete
                        ):
                            frame_rejection["veto_reason"] = (
                                "measured_support_delegation_not_audited"
                            )
                            mus_corridor_rejection_rows.append(
                                frame_rejection
                            )
                            continue
                        context_member_supports = tuple(
                            np.ascontiguousarray(
                                np.asarray(
                                    current_interval_by_track[
                                        track_id
                                    ].rgb_transfer_mask,
                                    dtype=bool,
                                )
                                & effective_closure_support
                            )
                            for track_id in sorted(closure_track_ids)
                        )
                        rgb_transfer, transfer_audit = (
                            _bounded_exact_corridor_transfer(
                                effective_closure_support,
                                target_corridor,
                                frame.panel_valid_mask,
                                dilation_pixels=0,
                                reserved_foreign_support=(
                                    reserved_foreign_support(
                                        set(closure_track_ids)
                                    )
                                ),
                            )
                        )
                        if rgb_transfer is None:
                            frame_rejection["veto_reason"] = (
                                "exact_member_transfer_support_not_retained"
                            )
                            mus_corridor_rejection_rows.append(
                                frame_rejection
                            )
                            continue
                        interval = InspectionPreSeamHardOwnerInterval(
                            track_id=group_track_id,
                            panel_index=int(frame.panel_index),
                            frame_id=int(frame.frame_id),
                            lock_mask=np.ascontiguousarray(
                                target_corridor
                            ),
                            union_footprint=np.ascontiguousarray(
                                effective_closure_support
                            ),
                            rgb_source_panel_index=int(frame.panel_index),
                            rgb_transfer_mask=rgb_transfer,
                            owner_only_mask=np.ascontiguousarray(
                                target_corridor
                            ),
                            rgb_context_member_supports=(
                                context_member_supports
                            ),
                            background_panel_lock_required=False,
                        )
                        panel_center = (
                            float(layout.panels[frame.panel_index].canvas_offset_x)
                            + 0.5 * float(intrinsics.width)
                        )
                        corridor_center = float(np.median(xx))
                        track_evidence = []
                        for track_id in sorted(closure_track_ids):
                            observation = eligible_frames_by_track[
                                track_id
                            ].get(int(frame.frame_id))
                            reference_panels = {
                                int(value[0])
                                for value in owner_by_track[
                                    track_id
                                ].reference_observation_masks
                            }
                            track_evidence.append(
                                (
                                    int(track_id),
                                    observation is not None,
                                    (
                                        None
                                        if observation is None
                                        else tuple(
                                            float(value)
                                            for value
                                            in observation[
                                                "selection_rank"
                                            ]
                                        )
                                    ),
                                    int(
                                        np.count_nonzero(
                                            np.asarray(
                                                current_interval_by_track[
                                                    track_id
                                                ].rgb_transfer_mask,
                                                dtype=bool,
                                            )
                                        )
                                    ),
                                    int(frame.panel_index)
                                    in reference_panels,
                                )
                            )
                        score, evidence_audit = (
                            _level4b_panel_evidence_score(
                                track_evidence,
                                corridor_center_x=corridor_center,
                                panel_center_x=panel_center,
                                panel_index=int(frame.panel_index),
                                existing_corridor_frame=bool(
                                    int(frame.frame_id)
                                    in existing_corridor_frame_ids
                                ),
                            )
                        )
                        mus_corridor_candidates.append(
                            (
                                score,
                                tuple(sorted(closure_track_ids)),
                                interval,
                                {
                                    "resolution_level": (
                                        "level_4b_minimal_unsat_core_"
                                        "object_rich_corridor"
                                    ),
                                    "core_track_ids": list(core),
                                    "core_member_count": len(core),
                                    "closure_track_ids": sorted(
                                        closure_track_ids
                                    ),
                                    "closure_member_count": len(
                                        closure_track_ids
                                    ),
                                    "closure_additions": closure_additions,
                                    "closure_iteration_limit": (
                                        maximum_closure_iterations
                                    ),
                                    "closure_member_limit": (
                                        maximum_closure_members
                                    ),
                                    "closure_fixed_point_reached": True,
                                    "target_corridor_connected": bool(
                                        closure_component_count == 2
                                    ),
                                    "target_corridor_row_contiguous": True,
                                    "fixed_corridor_boundary_trim_count": (
                                        len(fixed_boundary_trim_rows)
                                    ),
                                    "fixed_corridor_boundary_trimmed_"
                                    "pixel_count": sum(
                                        int(row["trimmed_pixel_count"])
                                        for row
                                        in fixed_boundary_trim_rows
                                    ),
                                    "fixed_corridor_boundary_trims": (
                                        fixed_boundary_trim_rows
                                    ),
                                    "fixed_corridor_trim_zero_measured_"
                                    "support_intersection": all(
                                        bool(
                                            row[
                                                "zero_measured_support_"
                                                "intersection"
                                            ]
                                        )
                                        for row
                                        in fixed_boundary_trim_rows
                                    ),
                                    "fixed_corridor_trim_row_contiguous_"
                                    "pass": all(
                                        bool(
                                            row[
                                                "subtraction_row_"
                                                "contiguous"
                                            ]
                                        )
                                        for row
                                        in fixed_boundary_trim_rows
                                    ),
                                    "foreign_compatibility_rechecked_after_"
                                    "fixed_boundary_trim": bool(
                                        fixed_boundary_trim_rows
                                    ),
                                    "core_derivation": core_derivation_rows,
                                    "global_csp_proven_unsat": True,
                                    "minimal_core_rechecked_unsat": True,
                                    "selected_frame_id": int(
                                        frame.frame_id
                                    ),
                                    "selected_panel_index": int(
                                        frame.panel_index
                                    ),
                                    "selected_source_index": int(
                                        frame.source_index
                                    ),
                                    "panel_evidence_selection": (
                                        {
                                            **evidence_audit,
                                            "corridor_center_x": (
                                                corridor_center
                                            ),
                                            "panel_center_x": panel_center,
                                        }
                                    ),
                                    "all_selected_real_panels_searched": True,
                                    "target_union_pixel_count": int(
                                        np.count_nonzero(target_union)
                                    ),
                                    "effective_member_transfer_support_"
                                    "pixel_count": int(
                                        np.count_nonzero(
                                            effective_closure_support
                                        )
                                    ),
                                    "delegated_cross_track_boundary_"
                                    "pixel_count": delegated_support_pixels,
                                    "all_member_support_pixels_have_exactly_"
                                    "one_real_rgb_owner": bool(
                                        support_partition_complete
                                        and delegated_single_owner_covered
                                        and all_member_partitions_complete
                                    ),
                                    "delegated_support_single_real_rgb_"
                                    "owner_covered": (
                                        delegated_single_owner_covered
                                    ),
                                    "delegated_transfer_frame_conflict": (
                                        delegated_transfer_frame_conflict
                                    ),
                                    "delegated_exact_frame_conflict": (
                                        delegated_exact_frame_conflict
                                    ),
                                    "member_support_partitions": (
                                        member_partition_rows
                                    ),
                                    "target_corridor_pixel_count": (
                                        corridor_area
                                    ),
                                    "corridor_rgb_transfer_pixel_count": int(
                                        np.count_nonzero(rgb_transfer)
                                    ),
                                    "corridor_transfer_audit": (
                                        transfer_audit
                                    ),
                                    "target_corridor_width_pixels": (
                                        corridor_width
                                    ),
                                    "maximum_corridor_width_pixels": (
                                        maximum_corridor_width
                                    ),
                                    "maximum_corridor_area_pixels": (
                                        maximum_corridor_area
                                    ),
                                    "hard_resource_bounds_passed": True,
                                    "full_selected_panel_valid_coverage": True,
                                    "selected_panel_coverage_applies_to_"
                                    "actual_rgb_support_not_decoupled_guard": (
                                        True
                                    ),
                                    "member_target_supports_retained": {
                                        str(track_id): int(
                                            np.count_nonzero(
                                                np.asarray(
                                                    current_interval_by_track[
                                                        track_id
                                                    ].rgb_transfer_mask,
                                                    dtype=bool,
                                                )
                                                & target_corridor
                                            )
                                        )
                                        for track_id in closure_track_ids
                                    },
                                    "member_target_supports_delegated_to_"
                                    "existing_fixed_owner": {
                                        str(track_id): int(
                                            np.count_nonzero(
                                                np.asarray(
                                                    current_interval_by_track[
                                                        track_id
                                                    ].rgb_transfer_mask,
                                                    dtype=bool,
                                                )
                                                & delegated_closure_support
                                            )
                                        )
                                        for track_id in closure_track_ids
                                    },
                                    "all_member_target_supports_fully_"
                                    "retained": bool(
                                        all_member_partitions_complete
                                        and delegated_single_owner_covered
                                    ),
                                    "foreign_required_constraints": (
                                        foreign_rows
                                    ),
                                    "all_members_handled": True,
                                    "distinct_objects_preserved": True,
                                    "mesh_used": False,
                                    "graphcut_multiband_flow_allowed_inside": (
                                        False
                                    ),
                                    "rgb_blended_or_generated": False,
                                },
                            )
                        )
            if mus_corridor_candidates:
                maximum_complete_observations = max(
                    int(
                        item[3]["panel_evidence_selection"][
                            "eligible_complete_observation_count"
                        ]
                    )
                    for item in mus_corridor_candidates
                )
                minimum_endpoint_evidence = max(
                    1,
                    int(math.ceil(0.60 * maximum_complete_observations)),
                )
                endpoint_candidates = [
                    item
                    for item in mus_corridor_candidates
                    if int(
                        item[3]["panel_evidence_selection"][
                            "eligible_complete_observation_count"
                        ]
                    )
                    >= minimum_endpoint_evidence
                ]
                target_center_x = float(
                    endpoint_candidates[0][3][
                        "panel_evidence_selection"
                    ]["corridor_center_x"]
                )
                endpoint_direction = (
                    1.0
                    if target_center_x >= 0.5 * float(layout.width)
                    else -1.0
                )
                endpoint_candidates.sort(
                    key=lambda item: (
                        endpoint_direction
                        * float(item[2].panel_index),
                        item[0],
                    ),
                    reverse=True,
                )
                _, closure_members, interval, mus_audit = (
                    endpoint_candidates[0]
                )
                mus_audit = {
                    **mus_audit,
                    "endpoint_preservation_selection": {
                        "maximum_complete_observation_count": (
                            maximum_complete_observations
                        ),
                        "minimum_qualified_complete_observation_count": (
                            minimum_endpoint_evidence
                        ),
                        "qualified_candidate_count": len(
                            endpoint_candidates
                        ),
                        "target_canvas_side": (
                            "right"
                            if endpoint_direction > 0.0
                            else "left"
                        ),
                        "same_side_outer_panel_preferred": True,
                        "reference_image_used": False,
                        "frame_id_hardcoded": False,
                    },
                }
                corridor_intervals.append(interval)
                corridor_handled_track_ids.update(closure_members)
                resolution_events.append(
                    {
                        "iteration": int(iteration),
                        **mus_audit,
                    }
                )
                continue
            common_frame_ids = set(
                eligible_frames_by_track[component[0]]
            )
            for track_id in component[1:]:
                common_frame_ids &= set(
                    eligible_frames_by_track[track_id]
                )
            candidate_rows: list[
                tuple[
                    tuple[float, ...],
                    int,
                    tuple[InspectionForegroundIdentityOwner, ...],
                    dict[str, object],
                ]
            ] = []
            for frame_id in sorted(common_frame_ids):
                frame = frame_by_id.get(int(frame_id))
                if frame is None:
                    continue
                panel_index = int(frame.panel_index)
                proposed: list[InspectionForegroundIdentityOwner] = []
                union_source = np.zeros(
                    (intrinsics.height, intrinsics.width), dtype=bool
                )
                member_rows: list[dict[str, object]] = []
                member_ranks: list[tuple[float, ...]] = []
                valid_candidate = True
                for track_id in component:
                    current_owner = owner_by_track[track_id]
                    source_masks = {
                        int(value_panel): np.asarray(value_mask, dtype=bool)
                        for value_panel, value_mask
                        in current_owner.reference_observation_masks
                    }
                    source_mask = source_masks.get(panel_index)
                    if (
                        source_mask is None
                        or source_mask.shape
                        != (intrinsics.height, intrinsics.width)
                        or not np.any(source_mask)
                    ):
                        valid_candidate = False
                        break
                    points = sample_mask_world_points(
                        mask=source_mask,
                        depth_mm=frame.depth_mm,
                        reliable_depth=frame.reliable_depth,
                        camera_to_world=frame.camera_to_world,
                        intrinsics=intrinsics,
                        stride=2,
                    )
                    projection = _project_structure(
                        points,
                        layout=layout,
                        intrinsics=intrinsics,
                        panel_index=panel_index,
                        panel_valid_mask=frame.panel_valid_mask,
                        minimum_sample_count=30,
                    )
                    if (
                        projection is None
                        or float(projection.in_bounds_ratio) < 0.90
                    ):
                        valid_candidate = False
                        break
                    reliable = (
                        np.asarray(frame.reliable_depth, dtype=bool)
                        & np.isfinite(frame.depth_mm)
                        & (np.asarray(frame.depth_mm) > 0.0)
                    )
                    depth_coverage = float(
                        np.count_nonzero(source_mask & reliable)
                        / max(1, np.count_nonzero(source_mask))
                    )
                    proposed.append(
                        replace(
                            current_owner,
                            panel_index=panel_index,
                            target_panel_index=panel_index,
                            frame_id=int(frame.frame_id),
                            source_index=int(frame.source_index),
                            source_mask=np.ascontiguousarray(source_mask),
                            target_footprint=np.ascontiguousarray(
                                projection.footprint.copy()
                            ),
                            measured_depth_coverage_ratio=depth_coverage,
                            projected_in_bounds_ratio=float(
                                projection.in_bounds_ratio
                            ),
                        )
                    )
                    union_source |= source_mask
                    observation = eligible_frames_by_track[track_id][
                        frame_id
                    ]
                    rank = tuple(
                        float(value)
                        for value in observation["selection_rank"]
                    )
                    member_ranks.append(rank)
                    member_rows.append(
                        {
                            "track_id": int(track_id),
                            "source_mask_pixel_count": int(
                                np.count_nonzero(source_mask)
                            ),
                            "source_depth_coverage_ratio": depth_coverage,
                            "projected_in_bounds_ratio": float(
                                projection.in_bounds_ratio
                            ),
                            "complete_observation_gate_passed": True,
                        }
                    )
                if not valid_candidate or len(proposed) != len(component):
                    continue
                yy, xx = np.nonzero(union_source)
                if not xx.size:
                    continue
                margin = int(config.direct_source_boundary_margin_pixels)
                union_boundary_clear = bool(
                    int(xx.min()) >= margin
                    and int(yy.min()) >= margin
                    and int(xx.max()) < intrinsics.width - margin
                    and int(yy.max()) < intrinsics.height - margin
                )
                if not union_boundary_clear:
                    continue
                trial_intervals, trial_audit = (
                    _build_panel_native_preseam_intervals(
                        proposed,
                        identity_frames,
                        layout=layout,
                        intrinsics=intrinsics,
                        geometric_track_ids=set(component),
                        config=config,
                    )
                )
                if len(trial_intervals) != len(component):
                    continue
                trial_masks = [
                    (
                        np.asarray(interval.lock_mask, dtype=bool)
                        if interval.rgb_transfer_mask is None
                        else np.asarray(
                            interval.rgb_transfer_mask, dtype=bool
                        )
                    )
                    for interval in trial_intervals
                ]
                same_source_overlap = 0
                for first_index, first_mask in enumerate(trial_masks):
                    for second_mask in trial_masks[first_index + 1 :]:
                        same_source_overlap += int(
                            np.count_nonzero(first_mask & second_mask)
                        )
                worst_member_rank = min(member_ranks)
                score = (
                    *worst_member_rank,
                    float(np.count_nonzero(union_source)),
                    -float(panel_index),
                )
                candidate_rows.append(
                    (
                        score,
                        frame_id,
                        tuple(proposed),
                        {
                            "selected_frame_id": int(frame.frame_id),
                            "selected_panel_index": panel_index,
                            "selected_source_index": int(frame.source_index),
                            "union_source_mask_pixel_count": int(
                                np.count_nonzero(union_source)
                            ),
                            "union_source_boundary_clear": True,
                            "full_union_coverage_pass": True,
                            "member_complete_coverage": member_rows,
                            "native_interval_plan": trial_audit,
                            "same_source_transfer_overlap_pixel_count": (
                                same_source_overlap
                            ),
                        },
                    )
                )
            if not candidate_rows:
                if len(component) > 2:
                    context = {
                        "schema": (
                            "inspection-shelf-native-unsat-context/v1"
                        ),
                        "iteration": int(iteration),
                        "active_track_ids": [
                            int(value) for value in component
                        ],
                        "minimal_unsat_core_track_ids": [
                            int(value) for value in core
                        ],
                        **mus_corridor_geometry_context,
                        "identity_frame_rejections": (
                            mus_corridor_rejection_rows
                        ),
                    }
                    raise RuntimeError(
                        "Shelf native owner global CSP is UNSAT and no "
                        "geometry-supported fragment composite corridor "
                        "preserves every distinct object; context="
                        + _format_shelf_unsat_context(context)
                    )
                if not config.object_rich_preseam_lock_enabled:
                    raise RuntimeError(
                        "Overlapping shelf object group lacks a common "
                        "complete real RGB frame and object-rich corridor "
                        "fallback is disabled"
                    )
                common_observed_frame_ids = set(
                    boundary_clear_frames_by_track[component[0]]
                )
                for track_id in component[1:]:
                    common_observed_frame_ids &= set(
                        boundary_clear_frames_by_track[track_id]
                    )
                corridor_candidates: list[
                    tuple[
                        tuple[float, ...],
                        InspectionPreSeamHardOwnerInterval,
                        dict[str, object],
                    ]
                ] = []
                for frame_id in sorted(common_observed_frame_ids):
                    frame = frame_by_id.get(int(frame_id))
                    if frame is None:
                        continue
                    panel_index = int(frame.panel_index)
                    union_source = np.zeros(
                        (intrinsics.height, intrinsics.width), dtype=bool
                    )
                    member_rows: list[dict[str, object]] = []
                    member_ranks: list[tuple[float, ...]] = []
                    valid_candidate = True
                    for track_id in component:
                        owner = owner_by_track[track_id]
                        source_masks = {
                            int(value_panel): np.asarray(
                                value_mask, dtype=bool
                            )
                            for value_panel, value_mask
                            in owner.reference_observation_masks
                        }
                        source_mask = source_masks.get(panel_index)
                        observation = boundary_clear_frames_by_track[
                            track_id
                        ].get(frame_id)
                        if (
                            source_mask is None
                            or observation is None
                            or source_mask.shape
                            != (intrinsics.height, intrinsics.width)
                            or not np.any(source_mask)
                        ):
                            valid_candidate = False
                            break
                        union_source |= source_mask
                        rank = tuple(
                            float(value)
                            for value in observation["selection_rank"]
                        )
                        member_ranks.append(rank)
                        member_rows.append(
                            {
                                "track_id": int(track_id),
                                "observed_frame_id": int(frame_id),
                                "source_mask_pixel_count": int(
                                    np.count_nonzero(source_mask)
                                ),
                                "source_boundary_clear": True,
                                "complete_observation_gate_passed": bool(
                                    observation[
                                        "eligible_complete_shelf_observation"
                                    ]
                                ),
                            }
                        )
                    if not valid_candidate or not np.any(union_source):
                        continue
                    yy, xx = np.nonzero(union_source)
                    margin = int(config.object_rich_lock_guard_pixels)
                    x0 = int(xx.min()) - margin
                    x1 = int(xx.max()) + margin + 1
                    y0 = int(yy.min()) - margin
                    y1 = int(yy.max()) + margin + 1
                    if (
                        x0 < 0
                        or y0 < 0
                        or x1 > intrinsics.width
                        or y1 > intrinsics.height
                    ):
                        continue
                    source_corridor = np.zeros_like(union_source)
                    source_corridor[y0:y1, x0:x1] = True
                    group_payload = ",".join(
                        str(value) for value in component
                    ).encode("ascii")
                    group_track_id = (
                        700_000_000
                        + (
                            int.from_bytes(
                                hashlib.blake2s(
                                    group_payload, digest_size=4
                                ).digest(),
                                "little",
                            )
                            % 100_000_000
                        )
                    )
                    seed_owner = owner_by_track[component[0]]
                    corridor_owner = replace(
                        seed_owner,
                        structure_id=group_track_id,
                        structure_kind=(
                            "middle_shelf_object_rich_single_source_corridor"
                        ),
                        identity_track_id=group_track_id,
                        panel_index=panel_index,
                        target_panel_index=panel_index,
                        frame_id=int(frame.frame_id),
                        source_index=int(frame.source_index),
                        source_mask=np.ascontiguousarray(source_corridor),
                        reference_observation_masks=(
                            (
                                panel_index,
                                np.ascontiguousarray(source_corridor),
                            ),
                        ),
                    )
                    trial_intervals, trial_audit = (
                        _build_panel_native_preseam_intervals(
                            (corridor_owner,),
                            identity_frames,
                            layout=layout,
                            intrinsics=intrinsics,
                            geometric_track_ids={group_track_id},
                            config=config,
                        )
                    )
                    if len(trial_intervals) != 1:
                        continue
                    interval = trial_intervals[0]
                    transfer = np.asarray(
                        interval.rgb_transfer_mask, dtype=bool
                    )
                    if (
                        np.any(
                            transfer
                            & ~np.asarray(
                                frame.panel_valid_mask, dtype=bool
                            )
                        )
                        or np.any(
                            np.asarray(interval.lock_mask, dtype=bool)
                            & ~np.asarray(
                                frame.panel_valid_mask, dtype=bool
                            )
                        )
                    ):
                        continue
                    foreign_overlap_rows: list[dict[str, object]] = []
                    for other_interval in final_intervals:
                        other_track_id = interval_track_id(other_interval)
                        if (
                            other_track_id in component
                            or other_track_id not in owner_by_track
                        ):
                            continue
                        other_transfer = (
                            np.asarray(
                                other_interval.lock_mask, dtype=bool
                            )
                            if other_interval.rgb_transfer_mask is None
                            else np.asarray(
                                other_interval.rgb_transfer_mask, dtype=bool
                            )
                        )
                        overlap_pixels = int(
                            np.count_nonzero(transfer & other_transfer)
                        )
                        if overlap_pixels:
                            foreign_overlap_rows.append(
                                {
                                    "track_id": int(other_track_id),
                                    "overlap_pixel_count": overlap_pixels,
                                }
                            )
                    if foreign_overlap_rows:
                        continue
                    worst_member_rank = min(member_ranks)
                    score = (
                        *worst_member_rank,
                        -float(np.count_nonzero(source_corridor)),
                        -float(panel_index),
                    )
                    corridor_candidates.append(
                        (
                            score,
                            interval,
                            {
                                "resolution_level": (
                                    "level_2_object_rich_corridor"
                                ),
                                "member_track_ids": list(component),
                                "member_observed_frames": {
                                    str(track_id): sorted(
                                        boundary_clear_frames_by_track[
                                            track_id
                                        ]
                                    )
                                    for track_id in component
                                },
                                "common_boundary_clear_observed_frame_ids": (
                                    sorted(common_observed_frame_ids)
                                ),
                                "selected_frame_id": int(frame.frame_id),
                                "selected_panel_index": panel_index,
                                "selected_source_index": int(
                                    frame.source_index
                                ),
                                "source_union_mask_pixel_count": int(
                                    np.count_nonzero(union_source)
                                ),
                                "source_corridor_bbox_xyxy": [
                                    x0,
                                    y0,
                                    x1,
                                    y1,
                                ],
                                "source_corridor_margin_pixels": margin,
                                "source_corridor_pixel_count": int(
                                    np.count_nonzero(source_corridor)
                                ),
                                "canvas_corridor_transfer_pixel_count": int(
                                    np.count_nonzero(transfer)
                                ),
                                "member_observations": member_rows,
                                "all_members_observed_and_boundary_clear": (
                                    True
                                ),
                                "full_panel_valid_inverse_map_coverage": True,
                                "foreign_required_object_overlap_pixel_count": (
                                    0
                                ),
                                "foreign_required_object_overlaps": [],
                                "single_contiguous_real_rgb_corridor": True,
                                "native_interval_plan": trial_audit,
                                "all_members_handled": True,
                                "mesh_used": False,
                                "cross_panel_warp_used": False,
                                "graphcut_multiband_flow_allowed_inside": (
                                    False
                                ),
                                "sequential_override_used": False,
                                "rgb_blended_or_generated": False,
                            },
                        )
                    )
                if not corridor_candidates:
                    raise RuntimeError(
                        "Overlapping shelf object group has no common "
                        "boundary-clear observed real RGB frame for an "
                        "object-rich corridor"
                    )
                corridor_candidates.sort(
                    key=lambda item: item[0], reverse=True
                )
                _, corridor_interval, corridor_audit = (
                    corridor_candidates[0]
                )
                corridor_intervals.append(corridor_interval)
                corridor_handled_track_ids.update(component)
                resolution_events.append(
                    {
                        "iteration": int(iteration),
                        "original_frame_ids": sorted(
                            {
                                int(owner_by_track[track_id].frame_id)
                                for track_id in component
                            }
                        ),
                        "common_complete_frame_ids": sorted(
                            common_frame_ids
                        ),
                        **corridor_audit,
                    }
                )
                continue
            candidate_rows.sort(key=lambda item: item[0], reverse=True)
            _, selected_frame_id, proposed, selected_audit = candidate_rows[0]
            original_frame_ids = sorted(
                {
                    int(owner_by_track[track_id].frame_id)
                    for track_id in component
                }
            )
            for owner in proposed:
                assert owner.identity_track_id is not None
                owner_by_track[int(owner.identity_track_id)] = owner
            resolution_events.append(
                {
                    "iteration": int(iteration),
                    "member_track_ids": list(component),
                    "original_frame_ids": original_frame_ids,
                    "common_complete_frame_ids": sorted(common_frame_ids),
                    **selected_audit,
                    "selected_frame_id": int(selected_frame_id),
                    "all_members_from_one_real_rgb_frame": True,
                    "all_member_masks_preserved": True,
                    "sequential_override_used": False,
                    "rgb_blended_or_generated": False,
                }
            )
    else:
        raise RuntimeError(
            "Shelf single-source conflict grouping did not converge"
        )

    final_intervals, final_interval_audit = build_current_intervals()
    # Shelf RGB is composited as an exclusive hard owner after the monotone
    # background chain.  Therefore its measured footprint must not make the
    # *background* seam infeasible: the later single-source replacement
    # removes whatever background choice was made underneath it.  Cross-object
    # RGB ownership remains fail-closed in the CSP and overlap audits below.
    post_background_track_ids = {
        interval_track_id(interval) for interval in final_intervals
    }
    final_intervals = tuple(
        replace(
            interval,
            protect_from_background_seam=False,
        )
        if interval_track_id(interval) in post_background_track_ids
        else interval
        for interval in final_intervals
    )
    post_background_overlap_rows: list[dict[str, object]] = []
    partitioned_intervals = list(final_intervals)
    for priority_index, priority in enumerate(partitioned_intervals):
        priority_track_id = interval_track_id(priority)
        if priority_track_id not in post_background_track_ids:
            continue
        priority_transfer = np.asarray(
            priority.rgb_transfer_mask, dtype=bool
        )
        for other_index, other in enumerate(partitioned_intervals):
            if other_index == priority_index:
                continue
            other_track_id = interval_track_id(other)
            if (
                other_track_id in post_background_track_ids
                or int(other.frame_id) == int(priority.frame_id)
            ):
                continue
            other_transfer = np.asarray(
                other.rgb_transfer_mask, dtype=bool
            )
            overlap = priority_transfer & other_transfer
            overlap_pixels = int(np.count_nonzero(overlap))
            if overlap_pixels == 0:
                continue
            other_pixels = int(np.count_nonzero(other_transfer))
            overlap_ratio = float(
                overlap_pixels / max(1, other_pixels)
            )
            bounded_alias = bool(
                overlap_pixels <= 384
                or (
                    overlap_pixels <= 4096
                    and overlap_ratio <= 0.15
                )
            )
            reduced_transfer = other_transfer & ~overlap
            if not bounded_alias or not np.any(reduced_transfer):
                raise RuntimeError(
                    "Deferred post-background shelf owner has an unbounded "
                    "exact-support collision with another distinct object; "
                    f"priority_track_id={priority_track_id}, "
                    f"other_track_id={other_track_id}, "
                    f"overlap_pixels={overlap_pixels}, "
                    f"other_support_ratio={overlap_ratio:.6f}"
                )
            partitioned_intervals[other_index] = replace(
                other,
                rgb_transfer_mask=np.ascontiguousarray(
                    reduced_transfer
                ),
            )
            post_background_overlap_rows.append(
                {
                    "priority_track_id": int(priority_track_id),
                    "priority_frame_id": int(priority.frame_id),
                    "delegated_track_id": int(other_track_id),
                    "delegated_frame_id": int(other.frame_id),
                    "overlap_pixel_count": overlap_pixels,
                    "delegated_support_overlap_ratio": overlap_ratio,
                    "absolute_limit_pixels": 4096,
                    "relative_limit": 0.15,
                    "exclusive_rgb_owner": int(priority.frame_id),
                    "rgb_blended_or_generated": False,
                    "accepted": True,
                }
            )
    final_intervals = tuple(partitioned_intervals)
    final_interval_audit = {
        **final_interval_audit,
        "post_background_hard_owner_track_ids": sorted(
            post_background_track_ids
        ),
        "post_background_hard_owner_count": sum(
            int(
                interval_track_id(interval)
                in post_background_track_ids
            )
            for interval in final_intervals
        ),
        "post_background_overlap_partition_count": len(
            post_background_overlap_rows
        ),
        "post_background_overlap_partitions": (
            post_background_overlap_rows
        ),
    }
    final_same_frame_overlap_pixels = 0
    final_different_frame_overlap_pixels = 0
    final_overlap_rows: list[dict[str, object]] = []
    for first_index, first in enumerate(final_intervals):
        first_transfer = (
            np.asarray(first.lock_mask, dtype=bool)
            if first.rgb_transfer_mask is None
            else np.asarray(first.rgb_transfer_mask, dtype=bool)
        )
        for second in final_intervals[first_index + 1 :]:
            second_transfer = (
                np.asarray(second.lock_mask, dtype=bool)
                if second.rgb_transfer_mask is None
                else np.asarray(second.rgb_transfer_mask, dtype=bool)
            )
            overlap_pixels = int(
                np.count_nonzero(first_transfer & second_transfer)
            )
            if not overlap_pixels:
                continue
            same_frame = int(first.frame_id) == int(second.frame_id)
            if same_frame:
                final_same_frame_overlap_pixels += overlap_pixels
            else:
                final_different_frame_overlap_pixels += overlap_pixels
            final_overlap_rows.append(
                {
                    "first_interval_track_id": int(first.track_id),
                    "second_interval_track_id": int(second.track_id),
                    "first_frame_id": int(first.frame_id),
                    "second_frame_id": int(second.frame_id),
                    "overlap_pixel_count": overlap_pixels,
                    "same_real_rgb_frame": same_frame,
                }
            )
    if final_different_frame_overlap_pixels:
        raise RuntimeError(
            "Shelf object-rich corridor overlaps another interval from a "
            "different real RGB frame"
        )
    final_interval_audit = {
        **final_interval_audit,
        "final_same_real_frame_transfer_overlap_pixel_count": (
            final_same_frame_overlap_pixels
        ),
        "final_different_real_frame_transfer_overlap_pixel_count": 0,
        "final_transfer_overlap_pairs": final_overlap_rows,
        "all_transfer_overlaps_have_one_real_rgb_owner": True,
    }
    final_components = overlap_components(final_intervals)
    final_rows: list[dict[str, object]] = []
    interval_by_track = {
        interval_track_id(interval): interval for interval in final_intervals
    }
    for group_index, component in enumerate(final_components):
        frame_ids = {
            int(owner_by_track[track_id].frame_id)
            for track_id in component
        }
        panel_indices = {
            int(owner_by_track[track_id].panel_index)
            for track_id in component
        }
        if len(frame_ids) != 1 or len(panel_indices) != 1:
            raise RuntimeError(
                "Shelf conflict group did not close to one real RGB owner"
            )
        masks = [
            (
                np.asarray(interval_by_track[track_id].lock_mask, dtype=bool)
                if interval_by_track[track_id].rgb_transfer_mask is None
                else np.asarray(
                    interval_by_track[track_id].rgb_transfer_mask, dtype=bool
                )
            )
            for track_id in component
        ]
        overlap_pixels = 0
        for first_index, first_mask in enumerate(masks):
            for second_mask in masks[first_index + 1 :]:
                overlap_pixels += int(
                    np.count_nonzero(first_mask & second_mask)
                )
        final_rows.append(
            {
                "conflict_group_id": int(group_index),
                "member_track_ids": list(component),
                "selected_frame_id": int(next(iter(frame_ids))),
                "selected_panel_index": int(next(iter(panel_indices))),
                "member_count": len(component),
                "same_real_rgb_transfer_overlap_pixel_count": overlap_pixels,
                "all_members_from_one_real_rgb_frame": True,
                "full_union_coverage_pass": True,
                "exact_member_masks_preserved": True,
                "overlap_rgb_write_is_identical_reference_raster": True,
                "sequential_override_used": False,
                "rgb_blended_or_generated": False,
            }
        )
    represented_native_track_ids = {
        interval_track_id(interval)
        for interval in final_intervals
        if interval_track_id(interval) in owner_by_track
    } | set(corridor_handled_track_ids)
    external_owner_masks = [
        np.asarray(owner_by_track[track_id].target_footprint, dtype=bool)
        for track_id in owner_order
        if track_id not in represented_native_track_ids
    ]
    external_owner_support = (
        np.logical_or.reduce(external_owner_masks)
        if external_owner_masks
        else np.zeros(
            (int(layout.height), int(layout.width)), dtype=bool
        )
    )
    (
        final_intervals,
        post_resolution_rgb_context_audit,
    ) = _expand_resolved_interval_rgb_context(
        final_intervals,
        identity_frames,
        reserved_external_support=external_owner_support,
        bounding_box_track_ids=deferred_post_background_track_ids,
    )
    final_interval_audit = {
        **final_interval_audit,
        "post_resolution_rgb_context": (
            post_resolution_rgb_context_audit
        ),
    }
    return (
        tuple(owner_by_track[value] for value in owner_order),
        final_intervals,
        final_interval_audit,
        {
            "schema": "inspection-shelf-single-source-conflict-group/v1",
            "policy": (
                "global_pairwise_native_footprint_csp_then_geometry_"
                "supported_fragment_composite_then_coobserved_pair_or_"
                "bounded_minimal_unsat_core_real_rgb_corridor_zero_blend"
            ),
            "required_owner_count": len(owner_order),
            "native_owner_reselection_count": len(
                native_owner_reselection_rows
            ),
            "native_owner_reselections": native_owner_reselection_rows,
            "resolution_event_count": len(resolution_events),
            "global_csp_search_state_count": csp_search_state_count,
            "global_csp_search_state_limit": csp_search_state_limit,
            "global_csp_budget_exhausted": False,
            "minimal_unsat_core_search_state_count": (
                mus_search_state_count
            ),
            "minimal_unsat_core_search_state_limit": (
                mus_search_state_limit
            ),
            "transitive_closure_constraint_used": False,
            "fragment_composite_group_count": sum(
                int(
                    row.get("resolution_level")
                    == "level_3_canonical_fragment_composite_corridor"
                )
                for row in resolution_events
            ),
            "coobserved_distinct_object_corridor_count": sum(
                int(
                    row.get("resolution_level")
                    == "level_4a_coobserved_distinct_object_corridor"
                )
                for row in resolution_events
            ),
            "minimal_unsat_core_corridor_count": sum(
                int(
                    row.get("resolution_level")
                    == "level_4b_minimal_unsat_core_object_rich_corridor"
                )
                for row in resolution_events
            ),
            "object_rich_corridor_group_count": len(
                corridor_intervals
            ),
            "object_rich_corridor_handled_track_ids": sorted(
                corridor_handled_track_ids
            ),
            "final_same_real_frame_transfer_overlap_pixel_count": (
                final_same_frame_overlap_pixels
            ),
            "final_different_real_frame_transfer_overlap_pixel_count": 0,
            "all_transfer_overlaps_have_one_real_rgb_owner": True,
            "conflict_group_count": len(final_rows),
            "resolution_events": resolution_events,
            "conflict_groups": final_rows,
            "post_resolution_rgb_context": (
                post_resolution_rgb_context_audit
            ),
            "pass": True,
            "fail_closed_without_common_complete_frame": True,
            "sequential_override_used": False,
            "rgb_blended_or_generated": False,
        },
        frozenset(corridor_handled_track_ids),
    )


def _build_object_rich_preseam_intervals(
    tracking: FastSAMDISTrackingResult,
    identity_frames: Sequence[InspectionIdentityOwnerFrame],
    *,
    layout: InspectionMultiviewLayout,
    intrinsics: CameraIntrinsics,
    eligible_track_ids: set[int],
    stable_target_owners: Sequence[
        InspectionForegroundIdentityOwner
    ],
    config: InspectionIdentityRuntimeConfig,
) -> tuple[
    tuple[InspectionPreSeamHardOwnerInterval, ...],
    dict[str, object],
]:
    """Find a complete vertically adjacent object group in one real view.

    The selected RGB frame may differ from the monotone spatial panel.  Its
    unchanged existing reference raster is copied only inside the accepted
    row-contiguous interval; no true-depth colour warp or generated pixel is
    permitted.
    """

    if not config.object_rich_preseam_lock_enabled:
        return (), {
            "schema": "inspection-object-rich-preseam-owner-plan/v1",
            "enabled": False,
            "candidate_group_count": 0,
            "accepted_interval_count": 0,
            "groups": [],
        }
    frames_by_id = {
        int(frame.frame_id): frame for frame in identity_frames
    }
    selected_frame_ids = set(frames_by_id)
    tracks = {
        int(track.track_id): track
        for track in tracking.stable_tracks
        if int(track.track_id) in eligible_track_ids
    }
    observations: dict[int, dict[int, object]] = {}
    for track_id, track in tracks.items():
        by_frame: dict[int, object] = {}
        for candidate_id in track.stable_candidate_ids:
            candidate = tracking.candidate_by_id[int(candidate_id)]
            if int(candidate.frame_id) in selected_frame_ids:
                by_frame[int(candidate.frame_id)] = candidate
        observations[track_id] = by_frame
    panel_centers = np.asarray(
        [
            float(panel.canvas_offset_x)
            + 0.5 * float(intrinsics.width - 1)
            for panel in layout.panels
        ],
        dtype=np.float64,
    )
    panel_boundaries = 0.5 * (
        panel_centers[:-1] + panel_centers[1:]
    )
    group_rows: list[dict[str, object]] = []
    accepted_candidates: list[
        tuple[
            float,
            InspectionPreSeamHardOwnerInterval,
            dict[str, object],
        ]
    ] = []
    track_ids = sorted(observations)
    for first_position, first_track_id in enumerate(track_ids):
        for second_track_id in track_ids[first_position + 1 :]:
            common_frames = sorted(
                set(observations[first_track_id])
                & set(observations[second_track_id])
            )
            for frame_id in common_frames:
                first = observations[first_track_id][frame_id]
                second = observations[second_track_id][frame_id]
                first_x, first_y, first_w, first_h = (
                    int(value) for value in first.bbox_xywh
                )
                second_x, second_y, second_w, second_h = (
                    int(value) for value in second.bbox_xywh
                )
                horizontal_overlap = max(
                    0,
                    min(first_x + first_w, second_x + second_w)
                    - max(first_x, second_x),
                )
                horizontal_overlap_ratio = float(
                    horizontal_overlap / max(1, min(first_w, second_w))
                )
                upper_bottom = min(
                    first_y + first_h,
                    second_y + second_h,
                )
                lower_top = max(first_y, second_y)
                vertical_gap = max(0, lower_top - upper_bottom)
                vertical_center_delta = abs(
                    (first_y + 0.5 * first_h)
                    - (second_y + 0.5 * second_h)
                )
                frame = frames_by_id[frame_id]
                first_mask = polygon_mask(
                    first, frame.depth_mm.shape
                )
                second_mask = polygon_mask(
                    second, frame.depth_mm.shape
                )
                smaller = min(
                    int(np.count_nonzero(first_mask)),
                    int(np.count_nonzero(second_mask)),
                )
                mask_overlap_ratio = float(
                    np.count_nonzero(first_mask & second_mask)
                    / max(1, smaller)
                )
                row: dict[str, object] = {
                    "track_ids": [first_track_id, second_track_id],
                    "frame_id": frame_id,
                    "rgb_source_panel_index": int(frame.panel_index),
                    "horizontal_bbox_overlap_ratio": (
                        horizontal_overlap_ratio
                    ),
                    "vertical_gap_pixels": int(vertical_gap),
                    "vertical_center_delta_pixels": float(
                        vertical_center_delta
                    ),
                    "mask_smaller_overlap_ratio": mask_overlap_ratio,
                    "accepted": False,
                }
                if (
                    horizontal_overlap_ratio
                    < config.object_rich_minimum_horizontal_overlap_ratio
                    or vertical_gap
                    > config.object_rich_maximum_vertical_gap_pixels
                    or vertical_center_delta
                    < 0.30 * min(first_h, second_h)
                    or mask_overlap_ratio > 0.20
                ):
                    row["reason"] = (
                        "not_a_vertically_adjacent_nonoverlapping_group"
                    )
                    group_rows.append(row)
                    continue
                source_union = first_mask | second_mask
                depth_coverage = float(
                    np.count_nonzero(source_union & frame.reliable_depth)
                    / max(1, np.count_nonzero(source_union))
                )
                row["source_union_depth_coverage_ratio"] = depth_coverage
                if (
                    depth_coverage
                    < config.object_rich_minimum_depth_coverage_ratio
                ):
                    row["reason"] = "object_group_depth_coverage_below_gate"
                    group_rows.append(row)
                    continue
                points = sample_mask_world_points(
                    mask=source_union,
                    depth_mm=frame.depth_mm,
                    reliable_depth=frame.reliable_depth,
                    camera_to_world=frame.camera_to_world,
                    intrinsics=intrinsics,
                    stride=2,
                )
                projected = _project_structure(
                    points,
                    layout=layout,
                    intrinsics=intrinsics,
                    panel_index=int(frame.panel_index),
                    panel_valid_mask=frame.panel_valid_mask,
                    minimum_sample_count=30,
                )
                if projected is None:
                    row["reason"] = "object_group_true_projection_failed"
                    group_rows.append(row)
                    continue
                (
                    corner_x,
                    map_x,
                    map_y,
                    map_valid,
                    _,
                ) = _reference_panel_inverse_maps(
                    source_pose=np.asarray(
                        frame.camera_to_world, dtype=np.float64
                    ),
                    panel_index=int(frame.panel_index),
                    layout=layout,
                    intrinsics=intrinsics,
                )
                reference_local = (
                    accelerated_remap(
                        source_union.astype(np.uint8),
                        np.where(map_valid, map_x, -1.0).astype(
                            np.float32, copy=False
                        ),
                        np.where(map_valid, map_y, -1.0).astype(
                            np.float32, copy=False
                        ),
                        cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    > 0
                ) & map_valid
                reference_footprint = np.zeros(
                    (int(layout.height), int(layout.width)), dtype=bool
                )
                reference_footprint[
                    :, corner_x : corner_x + reference_local.shape[1]
                ] = reference_local
                combined_footprint = (
                    np.asarray(projected.footprint, dtype=bool)
                    | reference_footprint
                )
                yy, xx = np.nonzero(combined_footprint)
                if xx.size == 0:
                    row["reason"] = "object_group_canvas_footprint_empty"
                    group_rows.append(row)
                    continue
                center_x = float(np.median(xx))
                spatial_panel_index = int(
                    np.count_nonzero(center_x > panel_boundaries)
                )
                lock = _row_contiguous_guard(
                    combined_footprint,
                    guard_pixels=int(
                        config.object_rich_lock_guard_pixels
                    ),
                )
                transfer_mask = np.ascontiguousarray(
                    reference_footprint
                )
                owner_only_mask = (
                    cv2.dilate(
                        transfer_mask.astype(np.uint8),
                        cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE,
                            (
                                2
                                * int(
                                    config.object_rich_lock_guard_pixels
                                )
                                + 1,
                                2
                                * int(
                                    config.object_rich_lock_guard_pixels
                                )
                                + 1,
                            ),
                        ),
                    )
                    > 0
                ) & lock
                foreign_overlap_rows: list[dict[str, object]] = []
                foreign_overlap_pixel_count = 0
                for other_owner in stable_target_owners:
                    other_track_id = other_owner.identity_track_id
                    if (
                        other_track_id is None
                        or int(other_track_id)
                        in {first_track_id, second_track_id}
                    ):
                        continue
                    other_footprint = np.asarray(
                        other_owner.target_footprint,
                        dtype=bool,
                    )
                    overlap_pixels = int(
                        np.count_nonzero(
                            transfer_mask & other_footprint
                        )
                    )
                    if overlap_pixels <= 0:
                        continue
                    foreign_overlap_pixel_count += overlap_pixels
                    foreign_overlap_rows.append(
                        {
                            "identity_track_id": int(other_track_id),
                            "frame_id": int(other_owner.frame_id),
                            "overlap_pixel_count": overlap_pixels,
                            "other_footprint_overlap_ratio": float(
                                overlap_pixels
                                / max(
                                    1,
                                    np.count_nonzero(
                                        other_footprint
                                    ),
                                )
                            ),
                            "transfer_overlap_ratio": float(
                                overlap_pixels
                                / max(
                                    1,
                                    np.count_nonzero(
                                        transfer_mask
                                    ),
                                )
                            ),
                        }
                    )
                spatial_frame = identity_frames[spatial_panel_index]
                missing_spatial = int(
                    np.count_nonzero(
                        lock & ~spatial_frame.panel_valid_mask
                    )
                )
                missing_source = int(
                    np.count_nonzero(lock & ~frame.panel_valid_mask)
                )
                crossed_boundaries = int(
                    np.count_nonzero(
                        (panel_boundaries >= float(np.min(xx)))
                        & (panel_boundaries <= float(np.max(xx)))
                    )
                )
                row.update(
                    {
                        "spatial_panel_index": spatial_panel_index,
                        "canvas_bbox_xyxy": [
                            int(np.min(xx)),
                            int(np.min(yy)),
                            int(np.max(xx)) + 1,
                            int(np.max(yy)) + 1,
                        ],
                        "lock_pixel_count": int(np.count_nonzero(lock)),
                        "spatial_panel_missing_valid_pixel_count": (
                            missing_spatial
                        ),
                        "rgb_source_panel_missing_valid_pixel_count": (
                            missing_source
                        ),
                        "crossed_nominal_boundary_count": (
                            crossed_boundaries
                        ),
                        "foreign_stable_track_overlap_pixel_count": (
                            foreign_overlap_pixel_count
                        ),
                        "foreign_stable_track_overlaps": (
                            foreign_overlap_rows
                        ),
                    }
                )
                if missing_spatial or missing_source:
                    row["reason"] = "object_group_panel_coverage_incomplete"
                    group_rows.append(row)
                    continue
                if crossed_boundaries < 1:
                    row["reason"] = "object_group_already_inside_one_panel_domain"
                    group_rows.append(row)
                    continue
                if foreign_overlap_pixel_count:
                    row["reason"] = (
                        "source_object_transfer_would_occlude_another_"
                        "stable_track"
                    )
                    group_rows.append(row)
                    continue
                if int(frame.panel_index) != spatial_panel_index:
                    row["reason"] = (
                        "cross_panel_reference_plane_rgb_transfer_rejected_"
                        "without_true_depth_inverse_mesh_boundary_closure"
                    )
                    group_rows.append(row)
                    continue
                group_track_id = (
                    2_000_000
                    + min(first_track_id, second_track_id) * 1000
                    + max(first_track_id, second_track_id)
                )
                interval = InspectionPreSeamHardOwnerInterval(
                    track_id=group_track_id,
                    panel_index=spatial_panel_index,
                    frame_id=frame_id,
                    lock_mask=lock,
                    union_footprint=np.ascontiguousarray(
                        combined_footprint
                    ),
                    rgb_source_panel_index=int(frame.panel_index),
                    rgb_transfer_mask=transfer_mask,
                    owner_only_mask=np.ascontiguousarray(
                        owner_only_mask
                    ),
                )
                row["accepted"] = True
                row["reason"] = (
                    "audited_object_rich_hard_cut_candidate"
                )
                score = float(
                    np.count_nonzero(source_union)
                    * depth_coverage
                    / max(1, np.count_nonzero(lock))
                )
                row["ranking_score"] = score
                group_rows.append(row)
                accepted_candidates.append((score, interval, row))
    accepted_candidates.sort(
        key=lambda item: (-item[0], int(item[1].track_id))
    )
    accepted = (
        (accepted_candidates[0][1],)
        if accepted_candidates
        else ()
    )
    selected_track_id = (
        None if not accepted else int(accepted[0].track_id)
    )
    for row in group_rows:
        if (
            row.get("accepted") is True
            and (
                2_000_000
                + min(row["track_ids"]) * 1000
                + max(row["track_ids"])
            )
            != selected_track_id
        ):
            row["accepted"] = False
            row["reason"] = "lower_ranked_overlapping_group_not_selected"
    return accepted, {
        "schema": "inspection-object-rich-preseam-owner-plan/v1",
        "enabled": True,
        "policy": (
            "vertically_adjacent_stable_tracks_one_complete_real_rgb_"
            "source_decoupled_from_monotone_spatial_panel"
        ),
        "candidate_group_count": len(group_rows),
        "accepted_interval_count": len(accepted),
        "selected_track_id": selected_track_id,
        "groups": group_rows,
        "rgb_generated": False,
        "pose_modified": False,
        "true_depth_color_warp_used": False,
    }


def _resolve_fastsam_model(
    config: InspectionIdentityRuntimeConfig,
) -> tuple[Path, str]:
    configured = config.fastsam_model_path
    environment = os.environ.get(_FASTSAM_MODEL_ENVIRONMENT)
    if configured is not None and environment:
        first = Path(configured).expanduser().resolve()
        second = Path(environment).expanduser().resolve()
        if first != second:
            raise ValueError(
                "Configured FastSAM model and G305_FASTSAM_ONNX disagree"
            )
    raw = configured or environment
    if raw is None:
        raise FileNotFoundError(
            "Formal identity runtime is enabled but no FastSAM ONNX model "
            "was supplied by identity_owner_runtime.fastsam_model_path or "
            "G305_FASTSAM_ONNX"
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".onnx":
        raise FileNotFoundError(f"FastSAM ONNX model was not found: {path}")
    return path, ("configuration" if configured is not None else "environment")


def _rapidocr_model_directory(
    config: InspectionIdentityRuntimeConfig,
) -> tuple[Path, str]:
    configured = config.rapidocr_model_directory
    environment = os.environ.get(_RAPIDOCR_MODEL_DIRECTORY_ENVIRONMENT)
    if configured is not None and environment:
        first = Path(configured).expanduser().resolve()
        second = Path(environment).expanduser().resolve()
        if first != second:
            raise ValueError(
                "Configured RapidOCR model directory and "
                "G305_RAPIDOCR_MODEL_DIR disagree"
            )
    raw = configured or environment
    if raw is not None:
        return (
            Path(raw).expanduser().resolve(),
            "configuration" if configured is not None else "environment",
        )
    specification = importlib.util.find_spec("rapidocr_onnxruntime")
    if specification is None or specification.origin is None:
        raise FileNotFoundError(
            "RapidOCR model discovery failed; install rapidocr-onnxruntime "
            "or supply identity_owner_runtime.rapidocr_model_directory"
        )
    return Path(specification.origin).resolve().parent / "models", "package"


def _resolve_rapidocr_models(
    config: InspectionIdentityRuntimeConfig,
) -> tuple[RapidOCRModels, str]:
    directory, provenance = _rapidocr_model_directory(config)
    models = RapidOCRModels(
        detection=directory / "ch_PP-OCRv4_det_infer.onnx",
        classification=directory / "ch_ppocr_mobile_v2.0_cls_infer.onnx",
        recognition=directory / "ch_PP-OCRv4_rec_infer.onnx",
    ).validated()
    return models, provenance


def _fastsam_cuda_audit(profile_path: Path) -> dict[str, object]:
    summary = summarize_fastsam_profile(profile_path)
    providers = {
        str(provider): dict(values)
        for provider, values in dict(summary.get("providers", {})).items()
    }
    operators = {
        str(operator): {str(value) for value in values}
        for operator, values in dict(
            summary.get("operator_providers", {})
        ).items()
    }
    cuda = "CUDAExecutionProvider"
    cpu = "CPUExecutionProvider"
    heavy = {"Conv", "ConvTranspose", "Gemm", "MatMul"}
    allowed_cpu = {
        "Add",
        "Cast",
        "Concat",
        "Div",
        "Floor",
        "Gather",
        "Mul",
        "Reshape",
        "Shape",
        "Slice",
        "Squeeze",
        "Unsqueeze",
    }
    heavy_cpu = sorted(
        operator
        for operator, assigned in operators.items()
        if operator in heavy and cpu in assigned
    )
    unexpected_cpu = sorted(
        operator
        for operator, assigned in operators.items()
        if cpu in assigned
        and operator not in allowed_cpu
        and operator not in heavy
    )
    cuda_events = int(providers.get(cuda, {}).get("node_events", 0))
    cpu_events = int(providers.get(cpu, {}).get("node_events", 0))
    cuda_duration = int(providers.get(cuda, {}).get("duration_us", 0))
    cpu_duration = int(providers.get(cpu, {}).get("duration_us", 0))
    failures: list[str] = []
    if cuda_events <= 0 or cuda not in operators.get("Conv", set()):
        failures.append("fastsam_convolution_not_executed_on_cuda")
    if heavy_cpu:
        failures.append("fastsam_heavy_operator_executed_on_cpu")
    if unexpected_cpu:
        failures.append("fastsam_unapproved_cpu_operator")
    audit = {
        "pass": not failures,
        "failures": failures,
        "provider_node_events": {
            cuda: cuda_events,
            cpu: cpu_events,
        },
        "provider_duration_us": {
            cuda: cuda_duration,
            cpu: cpu_duration,
        },
        "operator_providers": {
            key: sorted(value) for key, value in operators.items()
        },
        "heavy_cpu_operators": heavy_cpu,
        "unexpected_cpu_operators": unexpected_cpu,
        "allowed_cpu_shape_index_control_operators": sorted(allowed_cpu),
        "cpu_duration_fraction": float(
            cpu_duration / max(1, cuda_duration + cpu_duration)
        ),
    }
    return {
        **audit,
        "profile_retained": False,
        "requested_provider": "CUDAExecutionProvider",
        "silent_cpu_fallback_allowed": False,
    }


def _proposal_polygons(
    proposals: Sequence[object],
    *,
    image_pixels: int,
    config: InspectionIdentityRuntimeConfig,
) -> tuple[FastSAMExactMaskProposal, ...]:
    accepted: list[tuple[float, FastSAMExactMaskProposal]] = []
    for proposal in proposals:
        polygon = np.asarray(
            getattr(proposal, "polygon_xy"), dtype=np.float32
        )
        bbox = tuple(float(value) for value in getattr(proposal, "bbox_xyxy"))
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            continue
        area = max(0.0, bbox[2] - bbox[0]) * max(
            0.0, bbox[3] - bbox[1]
        )
        ratio = area / max(1, image_pixels)
        if not (
            config.minimum_proposal_area_ratio
            <= ratio
            <= config.maximum_proposal_area_ratio
        ):
            continue
        score = float(getattr(proposal, "score"))
        mask = np.asarray(getattr(proposal, "mask"), dtype=bool)
        if mask.ndim != 2:
            continue
        x, y, width, height = cv2.boundingRect(
            np.rint(polygon).astype(np.int32)
        )
        if (
            x < 0
            or y < 0
            or x + width > mask.shape[1]
            or y + height > mask.shape[0]
        ):
            continue
        exact = np.ascontiguousarray(
            mask[y : y + height, x : x + width]
        )
        if not np.any(exact):
            continue
        accepted.append(
            (
                score,
                FastSAMExactMaskProposal(
                    polygon_xy=np.ascontiguousarray(
                        polygon, dtype=np.float32
                    ),
                    bbox_xywh=(x, y, width, height),
                    exact_mask_bbox=exact,
                ),
            )
        )
    accepted.sort(key=lambda item: item[0], reverse=True)
    return tuple(
        proposal
        for _, proposal in accepted[: config.maximum_proposals_per_frame]
    )


def _preflight_identity_mesh(
    *,
    owners: Sequence[InspectionForegroundIdentityOwner],
    identity_frames: Sequence[InspectionIdentityOwnerFrame],
    layout: object,
    intrinsics: CameraIntrinsics,
    inspection_config: InspectionMultiviewConfig,
) -> tuple[
    tuple[InspectionForegroundIdentityOwner, ...],
    dict[str, object],
]:
    if not owners:
        return (), {
            "pass": True,
            "candidate_owner_count": 0,
            "accepted_owner_count": 0,
            "rejected_owner_count": 0,
            "accepted_owner_audits": [],
            "rejected_owners": [],
            "reason": "no_identity_owner_selected",
        }
    sources = {
        int(frame.frame_id): InspectionIdentityMeshSource(
            panel_index=int(frame.panel_index),
            frame_id=int(frame.frame_id),
            image_bgr=np.asarray(frame.image_bgr),
            depth_mm=np.asarray(frame.depth_mm),
            reliable_depth=np.asarray(frame.reliable_depth),
            camera_to_world=np.asarray(frame.camera_to_world),
        )
        for frame in identity_frames
    }
    shape = (int(layout.height), int(layout.width))
    mesh_config = InspectionIdentityMeshConfig(
        cell_size_pixels=int(
            inspection_config.identity_mesh_cell_size_pixels
        ),
        maximum_fill_distance_pixels=float(
            inspection_config.identity_mesh_maximum_fill_distance_pixels
        ),
        minimum_depth_mm=float(inspection_config.minimum_depth_mm),
        maximum_depth_mm=float(inspection_config.maximum_depth_mm),
        minimum_jacobian=float(
            inspection_config.depth_mesh_min_jacobian
        ),
        maximum_jacobian=float(
            inspection_config.depth_mesh_max_jacobian
        ),
    )
    accepted: list[InspectionForegroundIdentityOwner] = []
    accepted_audits: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for owner in owners:
        try:
            audit = composite_inspection_identity_owners(
                owners=(owner,),
                sources_by_frame_id=sources,
                layout=layout,
                intrinsics=intrinsics,
                output_image=np.zeros((*shape, 3), dtype=np.uint8),
                output_depth=np.full(shape, np.inf, dtype=np.float32),
                output_confidence=np.zeros(shape, dtype=np.float32),
                output_owner=np.full(shape, -1, dtype=np.int32),
                output_reliable_depth=np.zeros(shape, dtype=bool),
                output_overlay_mask=np.zeros(shape, dtype=bool),
                config=mesh_config,
            )
        except RuntimeError as error:
            rejected.append(
                {
                    "identity_track_id": owner.identity_track_id,
                    "group_id": int(owner.group_id),
                    "structure_id": int(owner.structure_id),
                    "frame_id": int(owner.frame_id),
                    "outcome": "hard_cut_degraded_not_applied",
                    "reason": str(error),
                }
            )
        else:
            accepted.append(owner)
            accepted_audits.append(audit)
    return tuple(accepted), {
        "pass": not rejected,
        "candidate_owner_count": len(owners),
        "accepted_owner_count": len(accepted),
        "rejected_owner_count": len(rejected),
        "accepted_owner_audits": accepted_audits,
        "rejected_owners": rejected,
        "rejected_owner_policy": (
            "do_not_modify_rgb_report_hard_cut_degraded_for_manual_review"
        ),
    }


def build_inspection_identity_runtime(
    frames: Sequence[RGBDFrame],
    poses: Sequence[np.ndarray],
    intrinsics: CameraIntrinsics,
    *,
    inspection_config: InspectionMultiviewConfig | Mapping[str, object],
    runtime_config: InspectionIdentityRuntimeConfig
    | Mapping[str, object]
    | None = None,
) -> InspectionIdentityRuntimeResult:
    """Build formal identity owners before rendering, entirely in memory."""

    selected = (
        runtime_config
        if isinstance(runtime_config, InspectionIdentityRuntimeConfig)
        else InspectionIdentityRuntimeConfig.from_mapping(runtime_config)
    )
    selected.validate()
    if (
        not selected.enabled
        and os.environ.get(_FASTSAM_MODEL_ENVIRONMENT)
    ):
        selected = replace(selected, enabled=True)
    if not selected.enabled:
        return InspectionIdentityRuntimeResult(
            pre_seam_hard_owner_intervals=(),
            foreground_owners=(),
            audit={
                "schema": "inspection-identity-runtime/v1",
                "enabled": False,
                "executed": False,
                "applied": False,
                "foreground_identity_owner_count": 0,
                "pre_seam_hard_owner_interval_count": 0,
                "object_owner_application_count": 0,
            },
        )
    renderer_config = (
        inspection_config
        if isinstance(inspection_config, InspectionMultiviewConfig)
        else InspectionMultiviewConfig.from_mapping(inspection_config)
    )
    if len(frames) != len(poses) or len(frames) < 2:
        raise ValueError(
            "Formal identity runtime requires aligned RGB-D poses"
        )
    if len(frames) > selected.maximum_frame_count:
        raise MemoryError(
            "Formal identity runtime frame count exceeds its hard bound: "
            f"{len(frames)} > {selected.maximum_frame_count}"
        )
    image_pixels = int(intrinsics.width * intrinsics.height)
    preview_pixels = int(
        math.ceil(intrinsics.width * 0.25)
        * math.ceil(intrinsics.height * 0.25)
    )
    estimated_bytes = int(
        len(frames)
        * (
            image_pixels * 9
            + selected.maximum_proposals_per_frame * preview_pixels
        )
    )
    if estimated_bytes > selected.maximum_runtime_bytes:
        raise MemoryError(
            "Formal identity runtime estimated working set exceeds its "
            f"bound: {estimated_bytes} > {selected.maximum_runtime_bytes}"
        )
    checked_poses = [
        np.asarray(pose, dtype=np.float64) for pose in poses
    ]
    layout = estimate_inspection_layout(
        frames, checked_poses, intrinsics, config=renderer_config
    )
    panel_sources = _select_panel_sources(checked_poses, layout)
    selected_frame_ids = {
        int(frames[source_index].frame_id)
        for _, source_index in panel_sources
    }
    fastsam_model, fastsam_model_provenance = _resolve_fastsam_model(
        selected
    )
    fastsam_model_sha256 = hashlib.sha256(
        fastsam_model.read_bytes()
    ).hexdigest()
    rapidocr_models: RapidOCRModels | None = None
    rapidocr_model_provenance: str | None = None
    if selected.rapidocr_enabled:
        rapidocr_models, rapidocr_model_provenance = (
            _resolve_rapidocr_models(selected)
        )

    maps = _undistortion_maps(intrinsics)
    cache: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    tracking_inputs: list[FastSAMDISFrameInput] = []
    proposal_counts: list[int] = []
    ocr_seeded_panels: list[object] = []
    ocr_frame_audits: list[dict[str, object]] = []
    rapidocr_audit: dict[str, object] | None = None
    with tempfile.TemporaryDirectory(
        prefix="g305_identity_runtime_"
    ) as profile_directory_text:
        profile_directory = Path(profile_directory_text)
        fastsam = FastSAMOnnxRunner(
            fastsam_model,
            device_id=selected.cuda_device_id,
            allow_cpu_diagnostic_fallback=False,
            config=FastSAMOnnxConfig(
                max_detections=selected.maximum_proposals_per_frame
            ),
            enable_profiling=True,
            profile_directory=profile_directory,
        )
        rapidocr = (
            RapidOCROnnxAdapter(
                rapidocr_models,
                RapidOCRRuntime(
                    device_id=selected.cuda_device_id,
                    profile_directory=profile_directory,
                ),
            )
            if rapidocr_models is not None
            else None
        )
        for source_index, (frame, pose) in enumerate(
            zip(frames, checked_poses, strict=True)
        ):
            image, depth, geometric_valid = _read_rgbd(
                frame, intrinsics, maps
            )
            reliable = (
                geometric_valid
                & np.isfinite(depth)
                & (depth >= renderer_config.minimum_depth_mm)
                & (depth <= renderer_config.maximum_depth_mm)
            )
            cache.append(
                (
                    np.ascontiguousarray(image),
                    np.ascontiguousarray(depth),
                    np.ascontiguousarray(geometric_valid),
                    np.ascontiguousarray(reliable),
                )
            )
            raw_proposals = fastsam.predict(image)
            polygons = _proposal_polygons(
                raw_proposals,
                image_pixels=image_pixels,
                config=selected,
            )
            proposal_counts.append(len(polygons))
            tracking_inputs.append(
                FastSAMDISFrameInput(
                    frame_id=int(frame.frame_id),
                    image_bgr=image,
                    depth_mm=depth,
                    camera_to_world=pose,
                    proposals=polygons,
                    geometric_valid=geometric_valid,
                )
            )
            if rapidocr is not None and int(frame.frame_id) in selected_frame_ids:
                detections = rapidocr.predict(image)
                accepted_detection_count = 0
                extraction_rows: list[dict[str, object]] = []
                for detection in detections:
                    if float(detection.score) < selected.minimum_ocr_score:
                        continue
                    panel, extraction_audit = extract_ocr_seeded_panel(
                        frame_id=int(frame.frame_id),
                        source_index=source_index,
                        image_bgr=image,
                        depth_mm=depth,
                        reliable_depth=reliable,
                        ocr_polygon_xy=detection.polygon_xy,
                        camera_to_world=pose,
                        intrinsics=intrinsics,
                    )
                    extraction_rows.append(extraction_audit)
                    if panel is not None:
                        ocr_seeded_panels.append(panel)
                        accepted_detection_count += 1
                ocr_frame_audits.append(
                    {
                        "frame_id": int(frame.frame_id),
                        "detection_count": len(detections),
                        "accepted_seed_count": accepted_detection_count,
                        "extractions": extraction_rows,
                    }
                )
        fastsam_profile = fastsam.end_profiling()
        if fastsam_profile is None or not fastsam_profile.is_file():
            raise RuntimeError(
                "FastSAM formal execution profile was not produced"
            )
        fastsam_audit = _fastsam_cuda_audit(fastsam_profile)
        if fastsam_audit["pass"] is not True:
            raise RuntimeError(
                "FastSAM actual execution did not satisfy CUDA policy: "
                f"{fastsam_audit['failures']}"
            )
        if rapidocr is not None:
            rapidocr_audit = rapidocr.audit()
            if (
                rapidocr_audit.get("execution_verified") is not True
                or not isinstance(rapidocr_audit.get("execution"), Mapping)
                or rapidocr_audit["execution"].get("pass") is not True
            ):
                raise RuntimeError(
                    "RapidOCR actual execution did not satisfy CUDA policy"
                )
            rapid_execution = dict(rapidocr_audit["execution"])
            rapid_execution.pop("profile_paths", None)
            rapidocr_audit = {
                **rapidocr_audit,
                "execution": rapid_execution,
                "profile_paths": None,
            }

    tracking = track_fastsam_dis_frames(
        tracking_inputs,
        intrinsics=intrinsics,
        reference_depth_mm=float(layout.reference_depth_mm),
        stable_frame_ids=sorted(selected_frame_ids),
        config=FastSAMDISConfig(
            preview_scale=0.25,
            minimum_depth_mm=float(renderer_config.minimum_depth_mm),
            maximum_depth_mm=float(renderer_config.maximum_depth_mm),
        ),
    )
    identity_frames: list[InspectionIdentityOwnerFrame] = []
    for panel_index, source_index in panel_sources:
        frame = frames[source_index]
        image, depth, _, reliable = cache[source_index]
        corner_x, _, _, local_valid, _ = _reference_panel_inverse_maps(
            source_pose=checked_poses[source_index],
            panel_index=panel_index,
            layout=layout,
            intrinsics=intrinsics,
        )
        panel_valid = np.zeros(
            (layout.height, layout.width), dtype=bool
        )
        panel_valid[:, corner_x : corner_x + local_valid.shape[1]] = (
            local_valid
        )
        identity_frames.append(
            InspectionIdentityOwnerFrame(
                panel_index=int(panel_index),
                source_index=int(source_index),
                frame_id=int(frame.frame_id),
                image_bgr=image,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=checked_poses[source_index],
                panel_valid_mask=panel_valid,
            )
        )
    ocr_plan = plan_inspection_identity_owner_intervals(
        frames=identity_frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
        ocr_seeded_panels=ocr_seeded_panels,
    )
    direct_plan = plan_direct_stable_track_identity_owners(
        frames=identity_frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
        existing_foreground_owners=ocr_plan.foreground_owners,
        config=DirectHandoffConfig(minimum_pair_target_iou=0.85),
    )
    shelf_inventory_plan = plan_middle_shelf_inventory_identity_owners(
        frames=identity_frames,
        tracking=tracking,
        layout=layout,
        intrinsics=intrinsics,
    )
    (
        resolved_shelf_inventory_owners,
        shelf_inventory_native_intervals,
        shelf_inventory_native_plan,
        shelf_single_source_conflict_groups,
        shelf_corridor_handled_track_ids,
    ) = _resolve_shelf_native_owner_conflict_groups(
        shelf_inventory_plan.foreground_owners,
        shelf_inventory_plan.audit,
        identity_frames,
        layout=layout,
        intrinsics=intrinsics,
        config=selected,
    )
    shelf_inventory_plan = replace(
        shelf_inventory_plan,
        foreground_owners=resolved_shelf_inventory_owners,
        audit={
            **shelf_inventory_plan.audit,
            "single_source_conflict_groups": (
                shelf_single_source_conflict_groups
            ),
        },
    )
    shelf_inventory_track_ids = {
        int(value)
        for value in shelf_inventory_plan.audit["inventory_track_ids"]
    }
    shelf_hierarchy_track_ids = {
        int(value)
        for value in shelf_inventory_plan.audit[
            "hierarchy_duplicate_track_ids"
        ]
    }
    shelf_exclusive_track_ids = (
        shelf_inventory_track_ids | shelf_hierarchy_track_ids
    )
    compact_direct_owners, direct_structure_gate = (
        _filter_compact_direct_owners(
            direct_plan.foreground_owners,
            config=selected,
        )
    )
    geometric_track_ids = {
        int(row["identity_track_id"])
        for row in direct_structure_gate["tracks"]
        if row["identity_track_id"] is not None
        and row["geometric_candidate_gate_passed"] is True
    }
    eligible_object_rich_track_ids = {
        int(row["track_id"])
        for row in direct_plan.audit["track_audits"]
        if row["accepted"] is True
    }
    (
        direct_preseam_owners,
        eligible_object_rich_track_ids,
        shelf_direct_preseam_exclusivity,
    ) = _exclude_shelf_tracks_from_direct_preseam_candidates(
        direct_plan.foreground_owners,
        eligible_object_rich_track_ids,
        shelf_exclusive_track_ids=shelf_exclusive_track_ids,
    )
    (
        panel_native_intervals,
        panel_native_interval_plan,
    ) = _build_panel_native_preseam_intervals(
        direct_preseam_owners,
        identity_frames,
        layout=layout,
        intrinsics=intrinsics,
        geometric_track_ids=geometric_track_ids,
        config=selected,
    )
    panel_native_by_track_id = {
        int(interval.track_id): interval
        for interval in panel_native_intervals
    }
    for interval in shelf_inventory_native_intervals:
        panel_native_by_track_id.setdefault(
            int(interval.track_id), interval
        )
    panel_native_intervals = tuple(
        panel_native_by_track_id[key]
        for key in sorted(panel_native_by_track_id)
    )
    (
        object_rich_intervals,
        object_rich_interval_plan,
    ) = _build_object_rich_preseam_intervals(
        tracking,
        identity_frames,
        layout=layout,
        intrinsics=intrinsics,
        eligible_track_ids=eligible_object_rich_track_ids,
        stable_target_owners=direct_preseam_owners,
        config=selected,
    )
    pre_seam_intervals = (
        *ocr_plan.intervals,
        *panel_native_intervals,
        *object_rich_intervals,
    )
    # A successful object-rich interval already preserves unchanged RGB from
    # one complete reference panel.  Do not additionally novel-view overlay
    # the same structures.  The older true-depth overlay remains separately
    # gated and disabled by default until its visibility/photometric closure
    # is proven.
    # Every required shelf inventory track already has exactly one audited
    # panel-native or object-rich pre-seam RGB owner.  Sending those same
    # tracks through the later true-depth mesh would create a second colour
    # writer and can split an otherwise complete object when the mesh clips at
    # a visibility boundary.  Keep true-depth mesh available for unrelated
    # compact foreground, while shelf inventory remains single-source.
    combined = tuple(
        owner
        for owner in compact_direct_owners
        if owner.identity_track_id is None
        or int(owner.identity_track_id)
        not in shelf_inventory_track_ids | shelf_hierarchy_track_ids
    )
    if len(combined) > selected.maximum_identity_owner_count:
        raise MemoryError(
            "Formal identity owner count exceeds its hard bound: "
            f"{len(combined)} > {selected.maximum_identity_owner_count}"
        )
    feasible_owners, mesh_preflight = _preflight_identity_mesh(
        owners=combined,
        identity_frames=identity_frames,
        layout=layout,
        intrinsics=intrinsics,
        inspection_config=renderer_config,
    )
    accepted_mesh_track_ids = {
        int(owner.identity_track_id)
        for owner in feasible_owners
        if owner.identity_track_id is not None
    }
    native_fallback_track_ids = {
        int(interval.track_id) - 1_000_000
        for interval in shelf_inventory_native_intervals
        if int(interval.track_id) >= 1_000_000
    } - accepted_mesh_track_ids
    handled_shelf_inventory_track_ids = (
        accepted_mesh_track_ids
        | native_fallback_track_ids
        | set(shelf_corridor_handled_track_ids)
    ) & shelf_inventory_track_ids
    raw_mesh_rejections = list(mesh_preflight["rejected_owners"])
    native_owner_track_ids = {
        int(interval.track_id) - 1_000_000
        for interval in panel_native_intervals
        if int(interval.track_id) >= 1_000_000
    }
    resolved_mesh_rejections = [
        row
        for row in raw_mesh_rejections
        if row.get("identity_track_id") is not None
        and int(row["identity_track_id"]) in native_owner_track_ids
    ]
    unresolved_mesh_rejections = [
        row
        for row in raw_mesh_rejections
        if row not in resolved_mesh_rejections
    ]
    mesh_preflight = {
        **mesh_preflight,
        "pass": not unresolved_mesh_rejections,
        "accepted_inverse_mesh_owner_count": len(feasible_owners),
        "accepted_same_panel_reference_rgb_owner_count": len(
            resolved_mesh_rejections
        ),
        "externally_handled_object_rich_corridor_owner_count": len(
            shelf_corridor_handled_track_ids
        ),
        "accepted_owner_count": (
            len(feasible_owners)
            + len(resolved_mesh_rejections)
        ),
        "rejected_owner_count": len(unresolved_mesh_rejections),
        "rejected_owners": unresolved_mesh_rejections,
        "inverse_mesh_rejections_resolved_by_same_panel_reference_rgb": (
            resolved_mesh_rejections
        ),
        "same_panel_reference_rgb_policy": (
            "exact_source_mask_existing_reference_inverse_map_single_rgb_"
            "owner_background_panel_decoupled_zero_blend"
        ),
    }
    unhandled_shelf_inventory_track_ids = sorted(
        shelf_inventory_track_ids - handled_shelf_inventory_track_ids
    )
    shelf_inventory_mesh_closure = {
        "required_track_ids": sorted(shelf_inventory_track_ids),
        "accepted_track_ids": sorted(
            shelf_inventory_track_ids & handled_shelf_inventory_track_ids
        ),
        "accepted_true_depth_mesh_track_ids": sorted(
            shelf_inventory_track_ids & accepted_mesh_track_ids
        ),
        "accepted_same_panel_reference_rgb_track_ids": sorted(
            shelf_inventory_track_ids
            & (
                native_fallback_track_ids
                | set(shelf_corridor_handled_track_ids)
            )
        ),
        "accepted_object_rich_corridor_track_ids": sorted(
            shelf_inventory_track_ids
            & set(shelf_corridor_handled_track_ids)
        ),
        "unhandled_track_ids": unhandled_shelf_inventory_track_ids,
        "pass": not unhandled_shelf_inventory_track_ids,
        "failure_policy": (
            "formal_main_flow_must_fail_closed_when_any_inventory_owner_"
            "mesh_is_not_accepted"
        ),
    }
    shelf_inventory_dispositions = []
    for value in shelf_inventory_plan.audit["track_dispositions"]:
        row = dict(value)
        track_id = int(row["track_id"])
        row["planner_disposition"] = row["inventory_disposition"]
        if track_id in shelf_inventory_track_ids:
            if track_id in accepted_mesh_track_ids:
                disposition = "required_owner_mesh_accepted"
            elif track_id in native_fallback_track_ids:
                disposition = (
                    "required_owner_same_panel_reference_rgb_accepted"
                )
            elif track_id in shelf_corridor_handled_track_ids:
                disposition = (
                    "required_owner_object_rich_corridor_accepted"
                )
            else:
                disposition = "required_owner_unhandled_fail_closed"
            row["inventory_disposition"] = disposition
            row["mesh_preflight_accepted"] = bool(
                track_id in accepted_mesh_track_ids
            )
            row["same_panel_reference_rgb_fallback_accepted"] = bool(
                track_id in native_fallback_track_ids
            )
            row["object_rich_corridor_accepted"] = bool(
                track_id in shelf_corridor_handled_track_ids
            )
        else:
            row["mesh_preflight_accepted"] = None
        shelf_inventory_dispositions.append(row)
    deferred_identity_intervals = tuple(
        InspectionPreSeamHardOwnerInterval(
            track_id=int(owner.identity_track_id),
            panel_index=(
                int(owner.panel_index)
                if owner.target_panel_index is None
                else int(owner.target_panel_index)
            ),
            frame_id=int(owner.frame_id),
            lock_mask=_row_contiguous_guard(
                np.asarray(owner.target_footprint, dtype=bool),
                guard_pixels=2,
            ),
            union_footprint=np.ascontiguousarray(
                np.asarray(owner.target_footprint, dtype=bool)
            ),
            rgb_source_panel_index=int(owner.panel_index),
            rgb_transfer_mask=np.ascontiguousarray(
                np.asarray(owner.target_footprint, dtype=bool)
            ),
            owner_only_mask=_row_contiguous_guard(
                np.asarray(owner.target_footprint, dtype=bool),
                guard_pixels=2,
            ),
            deferred_true_depth_identity_overlay=True,
        )
        for owner in feasible_owners
    )
    deferred_track_ids = {
        int(owner.identity_track_id)
        for owner in feasible_owners
        if owner.identity_track_id is not None
    }
    pre_seam_intervals = tuple(
        interval
        for interval in pre_seam_intervals
        if (
            int(interval.track_id) not in deferred_track_ids
            and (
                int(interval.track_id) < 1_000_000
                or int(interval.track_id) - 1_000_000
                not in deferred_track_ids
            )
        )
    )
    pre_seam_intervals = (
        *pre_seam_intervals,
        *deferred_identity_intervals,
    )
    return InspectionIdentityRuntimeResult(
        pre_seam_hard_owner_intervals=tuple(pre_seam_intervals),
        foreground_owners=feasible_owners,
        audit={
            "schema": "inspection-identity-runtime/v1",
            "enabled": True,
            "executed": True,
            "applied": bool(pre_seam_intervals or feasible_owners),
            "configuration": asdict(selected),
            "frame_count": len(frames),
            "selected_reference_panel_frame_count": len(panel_sources),
            "estimated_peak_input_and_tracking_bytes": estimated_bytes,
            "model_provenance": {
                "fastsam": {
                    "path": str(fastsam_model),
                    "path_source": fastsam_model_provenance,
                    "sha256": fastsam_model_sha256,
                },
                "rapidocr": (
                    {
                        "path_source": rapidocr_model_provenance,
                        "discovery": (
                            "explicit_directory_or_installed_package_models"
                        ),
                    }
                    if rapidocr_models is not None
                    else None
                ),
            },
            "cuda_execution": {
                "fastsam": fastsam_audit,
                "rapidocr": rapidocr_audit,
                "actual_provider_profile_required": True,
                "silent_cpu_fallback_allowed": False,
            },
            "proposal_count": int(sum(proposal_counts)),
            "maximum_frame_proposal_count": int(
                max(proposal_counts, default=0)
            ),
            "stable_dis_track_count": len(tracking.stable_tracks),
            "ocr_seeded_panel_count": len(ocr_seeded_panels),
            "ocr_frame_audits": ocr_frame_audits,
            "ocr_owner_plan": ocr_plan.audit,
            "direct_owner_plan": direct_plan.audit,
            "middle_shelf_inventory_owner_plan": (
                shelf_inventory_plan.audit
            ),
            "middle_shelf_same_panel_reference_rgb_plan": (
                shelf_inventory_native_plan
            ),
            "shelf_direct_preseam_owner_exclusivity": (
                shelf_direct_preseam_exclusivity
            ),
            "shelf_object_inventory": {
                "required_track_count": len(shelf_inventory_track_ids),
                "owner_candidate_count": len(
                    shelf_inventory_plan.foreground_owners
                ),
                "accepted_owner_count": len(
                    shelf_inventory_track_ids
                    & handled_shelf_inventory_track_ids
                ),
                "accepted_true_depth_mesh_owner_count": len(
                    shelf_inventory_track_ids & accepted_mesh_track_ids
                ),
                "accepted_same_panel_reference_rgb_owner_count": len(
                    shelf_inventory_track_ids
                    & (
                        native_fallback_track_ids
                        | set(shelf_corridor_handled_track_ids)
                    )
                ),
                "accepted_object_rich_corridor_owner_count": len(
                    shelf_inventory_track_ids
                    & set(shelf_corridor_handled_track_ids)
                ),
                "unhandled_required_track_count": len(
                    unhandled_shelf_inventory_track_ids
                ),
                "required_track_ids": sorted(shelf_inventory_track_ids),
                "unhandled_required_track_ids": (
                    unhandled_shelf_inventory_track_ids
                ),
                "dispositions": shelf_inventory_dispositions,
                "all_stable_tracks_have_disposition": (
                    shelf_inventory_plan.audit[
                        "all_stable_tracks_have_disposition"
                    ]
                ),
                "pass": not unhandled_shelf_inventory_track_ids,
                "reference_rgb_or_geometry_used": False,
                "track_ids_hardcoded": False,
                "single_source_conflict_groups": (
                    shelf_single_source_conflict_groups
                ),
            },
            "direct_structural_mask_gate": direct_structure_gate,
            "panel_native_preseam_owner_plan": (
                panel_native_interval_plan
            ),
            "object_rich_preseam_owner_plan": (
                object_rich_interval_plan
            ),
            "mesh_preflight": mesh_preflight,
            "middle_shelf_inventory_mesh_closure": (
                shelf_inventory_mesh_closure
            ),
            "foreground_identity_owner_candidate_count": len(combined),
            "foreground_identity_owner_count": len(feasible_owners),
            "pre_seam_hard_owner_interval_count": len(
                pre_seam_intervals
            ),
            "object_owner_application_count": (
                len(feasible_owners) + len(pre_seam_intervals)
            ),
            "owner_frame_ids": sorted(
                {
                    *(int(owner.frame_id) for owner in feasible_owners),
                    *(
                        int(interval.frame_id)
                        for interval in pre_seam_intervals
                    ),
                }
            ),
            "seam_integration": {
                "target_footprints_protect_graphcut_and_multiband_before_"
                "background_solve": bool(
                    pre_seam_intervals or feasible_owners
                ),
                "locked_owner_panel_index_applied_to_background_chain": bool(
                    pre_seam_intervals
                ),
                "identity_rgb_composition_stage": (
                    "existing_reference_panel_owner_before_monotone_"
                    "background_chain"
                    if pre_seam_intervals
                    else "true_rgbd_inverse_mesh_after_monotone_background_chain"
                ),
                "reason": (
                    "same_panel_identity_uses_existing_reference_raster_"
                    "and_graphcut_owner_lock"
                    if pre_seam_intervals
                    else (
                        "identity_source_rgb_uses_true_depth_projection_and_"
                        "is_not_the_source_panel_reference_plane_raster"
                    )
                ),
                "planner_interval_candidate_count": len(
                    pre_seam_intervals
                ),
                "post_render_overlay_used": False,
            },
            "delivery_grade_ceiling": (
                "C"
                if (
                    pre_seam_intervals
                    or feasible_owners
                    or mesh_preflight["rejected_owner_count"]
                    or direct_structure_gate["rejected_count"]
                )
                else None
            ),
            "manual_review_required_when_applied": bool(
                pre_seam_intervals
                or feasible_owners
                or mesh_preflight["rejected_owner_count"]
                or direct_structure_gate["rejected_count"]
            ),
            "rgb_generated": False,
            "post_render_overlay_used": False,
            "pose_modified": False,
            "depth_or_metric_feedback_to_rgb": False,
        },
    )


__all__ = [
    "InspectionIdentityRuntimeConfig",
    "InspectionIdentityRuntimeResult",
    "build_inspection_identity_runtime",
]
