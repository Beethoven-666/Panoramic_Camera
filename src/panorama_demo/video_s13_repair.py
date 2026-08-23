"""Frozen-Q Core repair candidates for the authorized S013 M7 stage.

This module deliberately contains no Optional implementation.  R2/R3 accept
only an independently replayed candidate carrying the required immutable-P0
audit declarations; they never estimate a seam, geometry, threshold, or source
set inside M7.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np


R0_KEEP_P3 = "R0_keep_p3"
R1_B1_TO_B0 = "R1_b1_2px_to_b0_owner_only"
R2_SEAM_DOWNGRADE = "R2_downgrade_seam_reestimate_geometry"
R3_GEOMETRY_DOWNGRADE = "R3_downgrade_geometry_reselect_seam"
CORE_CANDIDATES = (
    R0_KEEP_P3,
    R1_B1_TO_B0,
    R2_SEAM_DOWNGRADE,
    R3_GEOMETRY_DOWNGRADE,
)
OPTIONAL_CANDIDATES = frozenset({
    "constrained_two_label_graphcut",
    "depth_risk_veto",
    "bounded_local_mesh",
})
FORBIDDEN_CANDIDATES = frozenset({
    "Q4", "low_frequency_field", "feather_1px", "expanded_feather",
    "multiband", "cross_pair_joint_repair", "source_rescue",
    "interpolated_frame", "free_dense_flow", "inpainting",
})


@dataclass(frozen=True)
class S13CoreCandidate:
    model: str
    image: np.ndarray
    provenance: Mapping[str, np.ndarray]
    audit: Mapping[str, Any]


def validate_core_registry(candidates: tuple[str, ...]) -> None:
    if candidates != CORE_CANDIDATES:
        raise ValueError("S013 M7 Core registry must be exactly ordered R0-R3")
    lowered = {value.lower() for value in candidates}
    if lowered & {value.lower() for value in OPTIONAL_CANDIDATES | FORBIDDEN_CANDIDATES}:
        raise ValueError("S013 M7 Core registry contains an Optional/forbidden candidate")


def _copy_provenance(value: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(array).copy() for name, array in value.items()}


def r0_keep_p3(image: np.ndarray, provenance: Mapping[str, np.ndarray]) -> S13CoreCandidate:
    return S13CoreCandidate(
        model=R0_KEEP_P3,
        image=np.asarray(image).copy(),
        provenance=_copy_provenance(provenance),
        audit={"pixel_change_count": 0, "provenance_change_count": 0},
    )


def r1_b1_to_b0_owner_only(
    p3_image: np.ndarray,
    owner_only_image: np.ndarray,
    p3_provenance: Mapping[str, np.ndarray],
    affected_mask: np.ndarray,
    *,
    pair_index: int,
) -> S13CoreCandidate:
    """Disable exactly one handoff-bound B1 pair without changing Q/geometry/seam."""

    image = np.asarray(p3_image)
    owner_only = np.asarray(owner_only_image)
    mask = np.asarray(affected_mask, bool)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("S013 R1 requires uint8 BGR P3 pixels")
    if owner_only.shape != image.shape or owner_only.dtype != image.dtype or mask.shape != image.shape[:2]:
        raise ValueError("S013 R1 image/mask shape changed")
    provenance = _copy_provenance(p3_provenance)
    required = {
        "valid", "owner_source_index", "secondary_frame_id", "secondary_source_index",
        "secondary_source_u", "secondary_source_v", "secondary_weight", "blend_transaction_id",
    }
    if not required.issubset(provenance):
        raise ValueError("S013 R1 provenance is incomplete")
    valid = np.asarray(provenance["valid"], bool)
    secondary = np.asarray(provenance["secondary_source_index"], np.int32)
    owner = np.asarray(provenance["owner_source_index"], np.int32)
    weight = np.asarray(provenance["secondary_weight"], np.float32)
    pair_mask = (weight > 0) & (np.minimum(owner, secondary) == int(pair_index))
    if not np.any(mask) or not np.array_equal(mask, pair_mask):
        raise ValueError("S013 R1 affected mask is not the exact frozen B1 pair mask")
    if np.any(mask & ~valid) or np.any((weight[mask] <= 0) | (weight[mask] >= 0.5)):
        raise ValueError("S013 R1 affected pixels are not valid two-source B1 pixels")
    result = image.copy()
    result[mask] = owner_only[mask]
    provenance["secondary_frame_id"][mask] = -1
    provenance["secondary_source_index"][mask] = -1
    provenance["secondary_source_u"][mask] = np.nan
    provenance["secondary_source_v"][mask] = np.nan
    provenance["secondary_weight"][mask] = 0.0
    provenance["blend_transaction_id"][mask] = -1
    return S13CoreCandidate(
        model=R1_B1_TO_B0,
        image=result,
        provenance=provenance,
        audit={
            "pair_index": int(pair_index),
            "affected_pixel_count": int(np.count_nonzero(mask)),
            "pixel_change_count": int(np.count_nonzero(np.any(result != image, axis=2))),
            "q_parameters_frozen": True,
            "geometry_frozen": True,
            "seam_frozen": True,
            "b1_total_width_before_px": 2,
            "b1_total_width_after_px": 0,
        },
    )


def validate_replayed_r2_candidate(candidate: S13CoreCandidate) -> None:
    audit = candidate.audit
    if candidate.model != R2_SEAM_DOWNGRADE:
        raise ValueError("S013 R2 model identity changed")
    if (
        audit.get("seam_direction") not in {"S2_to_S1", "S2_to_S0", "S1_to_S0"}
        or audit.get("geometry_reestimated_from_immutable_p0") is not True
        or audit.get("final_corridor_geometry_reestimated") is not True
        or audit.get("q_parameters_frozen") is not True
        or audit.get("blend_model") != "B0_owner_only"
        or audit.get("thresholds_unchanged") is not True
        or audit.get("source_set_unchanged") is not True
    ):
        raise ValueError("S013 R2 replay did not prove frozen-Q seam downgrade invariants")


def validate_replayed_r3_candidate(candidate: S13CoreCandidate) -> None:
    audit = candidate.audit
    if candidate.model != R3_GEOMETRY_DOWNGRADE:
        raise ValueError("S013 R3 model identity changed")
    if (
        audit.get("geometry_complexity_decreased") is not True
        or audit.get("geometry_replayed_from_immutable_p0") is not True
        or audit.get("seam_reselected_for_candidate_geometry") is not True
        or audit.get("q_parameters_frozen") is not True
        or audit.get("blend_model") != "B0_owner_only"
        or audit.get("thresholds_unchanged") is not True
        or audit.get("source_set_unchanged") is not True
    ):
        raise ValueError("S013 R3 replay did not prove frozen-Q geometry downgrade invariants")


def ghost_material_count(reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> int:
    """Use the frozen M6.1 gradient-ghost predicate on an explicit component mask."""

    reference_gray = cv2.cvtColor(np.asarray(reference), cv2.COLOR_BGR2GRAY).astype(np.float32)
    candidate_gray = cv2.cvtColor(np.asarray(candidate), cv2.COLOR_BGR2GRAY).astype(np.float32)
    reference_gradient = np.hypot(
        cv2.Sobel(reference_gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(reference_gray, cv2.CV_32F, 0, 1),
    )
    candidate_gradient = np.hypot(
        cv2.Sobel(candidate_gray, cv2.CV_32F, 1, 0),
        cv2.Sobel(candidate_gray, cv2.CV_32F, 0, 1),
    )
    active = np.asarray(mask, bool)
    return int(np.count_nonzero(active & (candidate_gradient > reference_gradient * 1.5 + 4) & (reference_gradient >= 2)))


__all__ = [
    "CORE_CANDIDATES", "FORBIDDEN_CANDIDATES", "OPTIONAL_CANDIDATES",
    "R0_KEEP_P3", "R1_B1_TO_B0", "R2_SEAM_DOWNGRADE", "R3_GEOMETRY_DOWNGRADE",
    "S13CoreCandidate", "ghost_material_count", "r0_keep_p3",
    "r1_b1_to_b0_owner_only", "validate_core_registry",
    "validate_replayed_r2_candidate", "validate_replayed_r3_candidate",
]
